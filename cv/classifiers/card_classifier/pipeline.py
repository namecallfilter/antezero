"""
Unified Two-Stage Card Recognition Pipeline
===========================================
End-to-end inference orchestrator:
1. Stage 1 (YOLO Detector): Localizes all cards in the hand screenshot.
2. Stage 2 (Multi-Head Classifier): Batched attribute classification on card crops.
Returns sorted left-to-right list of CardResult objects.
"""

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from ultralytics import YOLO

CURRENT_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = CURRENT_DIR / "weights"

try:
    from classifiers.card_classifier.stage2_classifier.config import (
        IDX_TO_RANK, IDX_TO_SUIT, IDX_TO_EDITION, IDX_TO_ENHANCEMENT, IDX_TO_SEAL,
        CROP_WIDTH, CROP_HEIGHT
    )
    from classifiers.card_classifier.stage2_classifier.dataset import IMAGENET_MEAN, IMAGENET_STD
    from classifiers.card_classifier.stage2_classifier.model import (
        MultiHeadCardClassifier,
        infer_backbone_from_checkpoint,
    )
except ImportError:
    from stage2_classifier.config import (
        IDX_TO_RANK, IDX_TO_SUIT, IDX_TO_EDITION, IDX_TO_ENHANCEMENT, IDX_TO_SEAL,
        CROP_WIDTH, CROP_HEIGHT
    )
    from stage2_classifier.dataset import IMAGENET_MEAN, IMAGENET_STD
    from stage2_classifier.model import (
        MultiHeadCardClassifier,
        infer_backbone_from_checkpoint,
    )


@dataclass
class CardResult:
    index: int
    rank: str
    suit: str
    edition: str
    enhancement: str
    seal: str
    selected: bool
    debuffed: bool
    face_down: bool
    box: list[int]  # [x1, y1, x2, y2]
    confidences: dict[str, float]
    top_probs: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        parts = []
        if self.face_down:
            parts.append("[FACE DOWN]")
        else:
            if self.debuffed:
                parts.append("[DEBUFFED]")
            if self.edition != "None":
                parts.append(self.edition)
            if self.enhancement != "None":
                parts.append(self.enhancement)
            if self.rank == "Stone":
                parts.append("Stone Card")
            else:
                parts.append(f"{self.rank} of {self.suit}")
            if self.seal != "None":
                parts.append(f"({self.seal} Seal)")
        if self.selected:
            parts.append("[Selected]")
        return " ".join(parts)


def _fit_clamped_arch(
    design_x: np.ndarray,
    u: np.ndarray,
    y_target: np.ndarray,
    scale: float,
) -> tuple[float, float, float]:
    """Fits a gently curved, asymmetric-edge upward-convex arch `c0 + c1*u + c2*u^2`."""
    coef, _, _, _ = np.linalg.lstsq(design_x, y_target, rcond=None)
    c0, c1, c2 = float(coef[0]), float(coef[1]), float(coef[2])
    c2 = max(0.0, min(48.0 * scale, c2))
    # Allow negative c1 (leftmost card #0 occluded on its top-right corner shifts YOLO y1 downward).
    c1 = max(-16.0 * scale, min(2.0 * scale, c1))
    c0 = float(np.median(y_target - c1 * u - c2 * (u ** 2)))
    return c0, c1, c2


def determine_card_selection(
    boxes: list[list[int]],
    img_height: int,
    *,
    min_selected: int = 0,
    max_selected: int = 5,
) -> list[bool]:
    """
    Geometrically determines which cards are selected (raised upward) along Balatro's
    arched hand curve, optionally bounded to `[min_selected, max_selected]`.

    Selected cards shift upward by ~38-56px (at 322px crop height) relative to their
    resting position on the curved hand arch, whereas outer unselected cards naturally
    droop ~20-40px below center unselected cards.
    """
    n = len(boxes)
    if n == 0:
        return []

    eff_max = min(n, max(0, int(max_selected)))
    eff_min = min(eff_max, max(0, int(min_selected)))
    scale = (img_height / 322.0) if img_height > 0 else 1.0
    ys = np.array([b[1] for b in boxes], dtype=np.float64)
    cxs = np.array([(b[0] + b[2]) / 2.0 for b in boxes], dtype=np.float64)

    if n == 1:
        if eff_min == 1:
            return [True]
        if eff_max == 0:
            return [False]
        return [bool(ys[0] < 43.0 * scale)]
    if n == 2:
        if eff_min == 2:
            return [True, True]
        if eff_max == 0:
            return [False, False]
        if eff_min == 1 and eff_max == 1:
            return [True, False] if ys[0] <= ys[1] else [False, True]
        if float(ys.max() - ys.min()) > 28.0 * scale:
            midpoint = float(ys.min() + ys.max()) / 2.0
            return [bool(y < midpoint) for y in ys]
        res = [bool(y < 43.0 * scale) for y in ys]
        if sum(res) < eff_min:
            res = [True, False] if ys[0] <= ys[1] else [False, True]
        return res

    cx_min, cx_max = float(cxs.min()), float(cxs.max())
    span = max(1.0, (cx_max - cx_min) / 2.0)
    u = (cxs - (cx_max + cx_min) / 2.0) / span
    design_x = np.column_stack([np.ones(n), u, u ** 2])
    candidate_deltas = (
        40.0 * scale,
        46.0 * scale,
        52.0 * scale,
        58.0 * scale,
    )

    def _eval_mask(mask: list[bool]) -> tuple[float, tuple[float, float, float, float]]:
        sel = np.asarray(mask, dtype=np.float64)
        num_sel = int(sel.sum())
        if num_sel < eff_min or num_sel > eff_max:
            return float("inf"), (0.0, 0.0, 0.0, 0.0)
        if num_sel == 0 or num_sel == n:
            c0, c1, c2 = _fit_clamped_arch(design_x, u, ys, scale)
            if num_sel == 0 and c0 < 40.0 * scale:
                return float("inf"), (c0, c1, c2, 0.0)
            if num_sel == n and c0 >= 40.0 * scale:
                return float("inf"), (c0, c1, c2, 0.0)
            pred = c0 + c1 * u + c2 * (u ** 2)
            mse = float(np.mean((ys - pred) ** 2))
            if num_sel == 0 and float(np.max(np.abs(ys - pred))) > 22.0 * scale:
                mse += 100.0 * (scale ** 2)
            penalty = 0.0 if num_sel == 0 else 4.0 * (scale ** 2)
            return mse + penalty, (c0, c1, c2, 0.0)

        best_mse = float("inf")
        best_params = (0.0, 0.0, 0.0, 46.0 * scale)
        for delta in candidate_deltas:
            y_adj = ys + delta * sel
            c0, c1, c2 = _fit_clamped_arch(design_x, u, y_adj, scale)
            # An unselected arch baseline in hand/pack crops never sits above 43px*scale.
            c0 = max(43.0 * scale, c0)
            pred = c0 + c1 * u + c2 * (u ** 2) - delta * sel
            mse = float(np.mean((ys - pred) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_params = (c0, c1, c2, delta)
        return best_mse + 1.5 * (scale ** 2), best_params

    uniq_ys = sorted(set(ys.tolist()))
    candidates: list[list[bool]] = [[False] * n, [True] * n]
    for idx in range(len(uniq_ys) - 1):
        thresh = (uniq_ys[idx] + uniq_ys[idx + 1]) / 2.0
        candidates.append([bool(y < thresh) for y in ys])
    order_y = np.argsort(ys)
    for k in range(max(1, eff_min), eff_max + 1):
        cand_k = [False] * n
        for idx in order_y[:k]:
            cand_k[int(idx)] = True
        candidates.append(cand_k)

    best_score = float("inf")
    best_mask = [False] * n
    best_params = (0.0, 0.0, 0.0, 0.0)
    for cand in candidates:
        score, params = _eval_mask(cand)
        if score < best_score:
            best_score, best_mask, best_params = score, cand, params

    # Coordinate refinement flips mixed center+edge selections missed by pure 1D splits.
    for _ in range(3):
        c0, c1, c2, delta = best_params
        if delta <= 0.0:
            break
        arch = c0 + c1 * u + c2 * (u ** 2)
        raises = arch - ys
        refined = [bool(raises[i] > delta * 0.5) for i in range(n)]
        sel_cnt = sum(refined)
        if sel_cnt > eff_max:
            top_indices = set(int(i) for i in np.argsort(-raises)[:eff_max])
            refined = [i in top_indices for i in range(n)]
        elif sel_cnt < eff_min:
            top_indices = set(int(i) for i in np.argsort(-raises)[:eff_min])
            refined = [i in top_indices for i in range(n)]
        score, params = _eval_mask(refined)
        if score < best_score:
            best_score, best_mask, best_params = score, refined, params
        else:
            break

    return best_mask


def is_valid_hand_card_box(box: list[int] | tuple[int, int, int, int]) -> bool:
    """Rejects sub-card edge slivers (e.g. adjacent pack tops bleeding into pack_hand ROI)."""
    return (box[2] - box[0]) >= 35 and (box[3] - box[1]) >= 100


def decode_card_prediction(
    preds: dict,
    pred_idx: int,
    index: int,
    box: list[int],
    selected: bool = False,
) -> CardResult:
    """Decodes one row from MultiHeadCardClassifier.predict_attributes into a CardResult."""
    confs = {
        "rank": float(preds["rank_conf"][pred_idx].item()),
        "suit": float(preds["suit_conf"][pred_idx].item()),
        "edition": float(preds["edition_conf"][pred_idx].item()),
        "enhancement": float(preds["enhancement_conf"][pred_idx].item()),
        "seal": float(preds["seal_conf"][pred_idx].item()),
        "selected": 1.0,
        "debuffed": float(preds["debuffed_conf"][pred_idx].item()) if "debuffed_conf" in preds else 1.0,
        "face_down": float(preds["face_down_conf"][pred_idx].item()) if "face_down_conf" in preds else 1.0,
    }
    top_probs: dict[str, dict[str, float]] = {}
    head_maps = (
        ("rank", "rank_probs", IDX_TO_RANK),
        ("suit", "suit_probs", IDX_TO_SUIT),
        ("edition", "edition_probs", IDX_TO_EDITION),
        ("enhancement", "enhancement_probs", IDX_TO_ENHANCEMENT),
        ("seal", "seal_probs", IDX_TO_SEAL),
    )
    for head_name, probs_key, idx_map in head_maps:
        if probs_key in preds:
            row = preds[probs_key][pred_idx].tolist()
            top_probs[head_name] = {
                idx_map[c_i]: float(p)
                for c_i, p in enumerate(row)
                if p >= 0.005 and c_i in idx_map
            }

    return CardResult(
        index=index,
        rank=IDX_TO_RANK.get(preds["rank_idx"][pred_idx].item(), "Unknown"),
        suit=IDX_TO_SUIT.get(preds["suit_idx"][pred_idx].item(), "Unknown"),
        edition=IDX_TO_EDITION.get(preds["edition_idx"][pred_idx].item(), "None"),
        enhancement=IDX_TO_ENHANCEMENT.get(preds["enhancement_idx"][pred_idx].item(), "None"),
        seal=IDX_TO_SEAL.get(preds["seal_idx"][pred_idx].item(), "None"),
        selected=selected,
        debuffed=bool(preds["debuffed"][pred_idx].item()) if "debuffed" in preds else False,
        face_down=bool(preds["face_down"][pred_idx].item()) if "face_down" in preds else False,
        box=box,
        confidences=confs,
        top_probs=top_probs,
    )


class CardRecognitionPipeline:
    def __init__(self, detector_path=None, classifier_path=None, backbone="mobilenet_v3_small", device=None):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 1. Resolve Stage 1 Detector Weights
        if detector_path is None:
            detector_path = WEIGHTS_DIR / "detector_best.pt"
            if not detector_path.exists():
                candidates = list(CURRENT_DIR.glob("stage1_detector/runs/*/weights/best.pt"))
                if candidates:
                    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    detector_path = candidates[0]

        # 2. Resolve Stage 2 Classifier Weights
        if classifier_path is None:
            classifier_path = WEIGHTS_DIR / "classifier_best.pt"
            prod_cls = CURRENT_DIR.parent / "prod_weights" / "card_classifier.pt"
            if not classifier_path.exists() and prod_cls.exists():
                classifier_path = prod_cls
            elif not classifier_path.exists():
                candidates = list(CURRENT_DIR.glob("stage2_classifier/runs/*/weights/best.pt"))
                if candidates:
                    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    classifier_path = candidates[0]

        print(f"[Card Pipeline] Loading Detector   : {detector_path}")
        print(f"[Card Pipeline] Loading Classifier : {classifier_path}")

        self.detector = YOLO(str(detector_path))
        
        ckpt = torch.load(str(classifier_path), map_location=self.device)
        resolved_backbone = infer_backbone_from_checkpoint(ckpt, default_backbone=backbone or "mobilenet_v3_small")
        self.classifier = MultiHeadCardClassifier(backbone_name=resolved_backbone, pretrained=False).to(self.device)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        # Filter out head_selected in case loading an older checkpoint prior to retraining
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head_selected")}
        self.classifier.load_state_dict(filtered_state_dict, strict=False)
        self.classifier.eval()

        self.crop_transform = T.Compose([
            T.Resize((CROP_HEIGHT, CROP_WIDTH), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

        # Warm up GPU contexts & CUDA kernels
        if self.device.type == "cuda":
            dummy = Image.new("RGB", (CROP_WIDTH * 2, CROP_HEIGHT), (0, 0, 0))
            _ = self.predict(dummy, conf=0.50)

    def predict(self, image_input, conf=0.60, return_timing: bool = False):
        """
        Runs two-stage detection and classification.
        Args:
            image_input: PIL.Image, numpy ndarray, or file path.
            conf: Confidence threshold for Stage 1 card detector.
            return_timing: If True, returns (list[CardResult], timing_dict).
        Returns:
            list[CardResult] sorted left to right by x position (or (results, timing_dict)).
        """
        import time

        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input).convert("RGB")
        else:
            pil_img = image_input.convert("RGB")

        use_cuda = (self.device.type == "cuda")

        # 1. Stage 1: Card Detection (YOLO)
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        det_results = self.detector(pil_img, conf=conf, verbose=False)

        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        stage1_ms = (t1 - t0) * 1000.0

        boxes_data = det_results[0].boxes

        if len(boxes_data) == 0:
            timings = {
                "stage1_detector_ms": stage1_ms,
                "stage2_classifier_ms": 0.0,
                "postprocess_ms": 0.0,
                "total_inference_ms": stage1_ms
            }
            return ([], timings) if return_timing else []

        # Sort detected card bounding boxes left-to-right
        xyxy_boxes = boxes_data.xyxy.cpu().numpy()
        order = np.argsort(xyxy_boxes[:, 0])
        sorted_boxes = xyxy_boxes[order]

        # 2. Extract and batch card crops
        crop_tensors = []
        iw, ih = pil_img.size
        valid_boxes = []

        for b in sorted_boxes:
            x1 = max(0, min(iw - 1, int(b[0])))
            y1 = max(0, min(ih - 1, int(b[1])))
            x2 = max(x1 + 1, min(iw, int(b[2])))
            y2 = max(y1 + 1, min(ih, int(b[3])))
            if not is_valid_hand_card_box([x1, y1, x2, y2]):
                continue

            crop = pil_img.crop((x1, y1, x2, y2))
            crop_tensor = self.crop_transform(crop)
            crop_tensors.append(crop_tensor)
            valid_boxes.append([x1, y1, x2, y2])

        if not crop_tensors:
            timings = {
                "stage1_detector_ms": stage1_ms,
                "stage2_classifier_ms": 0.0,
                "postprocess_ms": 0.0,
                "total_inference_ms": stage1_ms,
            }
            return ([], timings) if return_timing else []

        batched_crops = torch.stack(crop_tensors, dim=0).to(self.device)

        # 3. Stage 2: Batched Attribute Classification (CNN)
        if use_cuda:
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        with torch.no_grad():
            preds = self.classifier.predict_attributes(batched_crops)

        if use_cuda:
            torch.cuda.synchronize()
        t3 = time.perf_counter()
        stage2_ms = (t3 - t2) * 1000.0

        # 4. Geometrically determine card selection (raised upward)
        selected_flags = determine_card_selection(valid_boxes, ih)

        # 5. Assemble CardResult objects
        results = [
            decode_card_prediction(preds, i, index=i + 1, box=valid_boxes[i], selected=selected_flags[i])
            for i in range(len(valid_boxes))
        ]

        t4 = time.perf_counter()
        total_ms = (t4 - t0) * 1000.0

        timings = {
            "stage1_detector_ms": stage1_ms,
            "stage2_classifier_ms": stage2_ms,
            "total_inference_ms": total_ms,
            "cards_detected": len(valid_boxes)
        }

        if return_timing:
            return results, timings
        return results

    def classify_crops(self, pil_crops: list[Image.Image], boxes: list[list[int]] | None = None) -> list[CardResult]:
        """
        Runs Stage 2 multi-head card classifier directly on pre-cropped card images.
        Bypasses Stage 1 YOLO detection entirely.
        """
        if not pil_crops:
            return []

        crop_tensors = [self.crop_transform(c.convert("RGB")) for c in pil_crops]
        batched_crops = torch.stack(crop_tensors, dim=0).to(self.device)

        with torch.no_grad():
            preds = self.classifier.predict_attributes(batched_crops)

        results = [
            decode_card_prediction(
                preds,
                i,
                index=i + 1,
                box=boxes[i] if boxes and i < len(boxes) else [0, 0, pil_crops[i].width, pil_crops[i].height],
                selected=False,
            )
            for i in range(len(pil_crops))
        ]

        return results
