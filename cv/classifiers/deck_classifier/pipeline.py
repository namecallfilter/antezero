"""
Unified Two-Stage View Deck Recognition Pipeline
================================================
End-to-end inference orchestrator for Balatro's View Deck overlay:
1. Stage 1 (Deck YOLO Detector): Localizes left-edge card slices across suit rows.
2. Stage 2 (Multi-Head Card Classifier): Batched attribute classification on slices.
Returns cards sorted row-by-row top-to-bottom, and left-to-right within each row.
"""

from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from ultralytics import YOLO

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
WEIGHTS_DIR = CURRENT_DIR / "weights"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classifiers.card_classifier.pipeline import CardResult, decode_card_prediction
from classifiers.card_classifier.stage2_classifier.config import CROP_HEIGHT, CROP_WIDTH
from classifiers.card_classifier.stage2_classifier.dataset import IMAGENET_MEAN, IMAGENET_STD
from classifiers.card_classifier.stage2_classifier.model import (
    MultiHeadCardClassifier,
    infer_backbone_from_checkpoint,
)


def is_valid_deck_card_box(box: list[int] | tuple[int, int, int, int]) -> bool:
    """Filters out tiny degenerate slivers while permitting narrow left-edge slices."""
    return (box[2] - box[0]) >= 12 and (box[3] - box[1]) >= 45


def sort_deck_boxes(boxes: list[list[int]] | np.ndarray) -> list[list[int]]:
    """
    Clusters detected boxes into horizontal suit rows (Y-center within 0.45 * median height),
    orders rows top-to-bottom, and sorts left-to-right by x1 within each row.
    """
    box_list = [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in boxes]
    if not box_list:
        return []

    heights = [max(1.0, float(b[3] - b[1])) for b in box_list]
    median_h = float(np.median(heights))
    row_tol = 0.45 * median_h

    enriched = []
    for b in box_list:
        cy = (b[1] + b[3]) / 2.0
        enriched.append((cy, b))
    enriched.sort(key=lambda item: item[0])

    rows: list[tuple[float, list[list[int]]]] = []
    for cy, b in enriched:
        if not rows or abs(cy - rows[-1][0]) > row_tol:
            rows.append((cy, [b]))
        else:
            prev_mean, prev_boxes = rows[-1]
            n = len(prev_boxes)
            new_mean = (prev_mean * n + cy) / (n + 1)
            prev_boxes.append(b)
            rows[-1] = (new_mean, prev_boxes)

    rows.sort(key=lambda r: r[0])
    ordered: list[list[int]] = []
    for _, row_boxes in rows:
        row_boxes.sort(key=lambda b: (b[0], b[1]))
        ordered.extend(row_boxes)
    return ordered


class DeckRecognitionPipeline:
    def __init__(
        self,
        detector_path: str | Path | None = None,
        classifier_path: str | Path | None = None,
        backbone: str = "mobilenet_v3_small",
        device: str | None = None,
    ):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if detector_path is None:
            detector_path = WEIGHTS_DIR / "detector_best.pt"
            prod_det = PROJECT_ROOT / "classifiers" / "prod_weights" / "deck_detector.pt"
            if not detector_path.exists() and prod_det.exists():
                detector_path = prod_det
            elif not detector_path.exists():
                candidates = list(CURRENT_DIR.glob("stage1_detector/runs/*/weights/best.pt"))
                if candidates:
                    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    detector_path = candidates[0]

        if classifier_path is None:
            card_weights_dir = PROJECT_ROOT / "classifiers" / "card_classifier" / "weights"
            classifier_path = card_weights_dir / "classifier_best.pt"
            prod_cls = PROJECT_ROOT / "classifiers" / "prod_weights" / "card_classifier.pt"
            if not classifier_path.exists() and prod_cls.exists():
                classifier_path = prod_cls
            elif not classifier_path.exists():
                candidates = list(
                    (PROJECT_ROOT / "classifiers" / "card_classifier").glob("stage2_classifier/runs/*/weights/best.pt")
                )
                if candidates:
                    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    classifier_path = candidates[0]

        print(f"[Deck Pipeline] Loading Detector   : {detector_path}")
        print(f"[Deck Pipeline] Loading Classifier : {classifier_path}")

        self.detector = YOLO(str(detector_path))

        ckpt = torch.load(str(classifier_path), map_location=self.device)
        resolved_backbone = infer_backbone_from_checkpoint(ckpt, default_backbone=backbone or "mobilenet_v3_small")
        self.classifier = MultiHeadCardClassifier(backbone_name=resolved_backbone, pretrained=False).to(self.device)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head_selected")}
        self.classifier.load_state_dict(filtered_state_dict, strict=False)
        self.classifier.eval()

        self.crop_transform = T.Compose([
            T.Resize((CROP_HEIGHT, CROP_WIDTH), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        if self.device.type == "cuda":
            dummy = Image.new("RGB", (CROP_WIDTH * 2, CROP_HEIGHT), (0, 0, 0))
            _ = self.predict(dummy, conf=0.45)

    def predict(
        self,
        image_input,
        conf: float = 0.45,
        iou: float = 0.55,
        max_det: int = 150,
        return_timing: bool = False,
    ):
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input).convert("RGB")
        else:
            pil_img = image_input.convert("RGB")

        use_cuda = (self.device.type == "cuda")
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        det_results = self.detector(pil_img, conf=conf, iou=iou, max_det=max_det, verbose=False)

        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        stage1_ms = (t1 - t0) * 1000.0

        boxes_data = det_results[0].boxes
        if boxes_data is None or len(boxes_data) == 0:
            timings = {
                "stage1_detector_ms": stage1_ms,
                "stage2_classifier_ms": 0.0,
                "total_inference_ms": stage1_ms,
                "cards_detected": 0,
            }
            return ([], timings) if return_timing else []

        iw, ih = pil_img.size
        raw_boxes = boxes_data.xyxy.cpu().numpy()
        valid_boxes = []
        for b in raw_boxes:
            x1 = max(0, min(iw - 1, int(round(b[0]))))
            y1 = max(0, min(ih - 1, int(round(b[1]))))
            x2 = max(x1 + 1, min(iw, int(round(b[2]))))
            y2 = max(y1 + 1, min(ih, int(round(b[3]))))
            box = [x1, y1, x2, y2]
            if is_valid_deck_card_box(box):
                valid_boxes.append(box)

        sorted_boxes = sort_deck_boxes(valid_boxes)
        if not sorted_boxes:
            timings = {
                "stage1_detector_ms": stage1_ms,
                "stage2_classifier_ms": 0.0,
                "total_inference_ms": stage1_ms,
                "cards_detected": 0,
            }
            return ([], timings) if return_timing else []

        crop_tensors = [self.crop_transform(pil_img.crop(tuple(box))) for box in sorted_boxes]
        batched_crops = torch.stack(crop_tensors, dim=0).to(self.device)

        if use_cuda:
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        with torch.no_grad():
            preds = self.classifier.predict_attributes(batched_crops)

        if use_cuda:
            torch.cuda.synchronize()
        t3 = time.perf_counter()
        stage2_ms = (t3 - t2) * 1000.0

        results: list[CardResult] = [
            decode_card_prediction(preds, i, index=i + 1, box=sorted_boxes[i], selected=False)
            for i in range(len(sorted_boxes))
        ]

        total_ms = (time.perf_counter() - t0) * 1000.0
        timings = {
            "stage1_detector_ms": stage1_ms,
            "stage2_classifier_ms": stage2_ms,
            "total_inference_ms": total_ms,
            "cards_detected": len(results),
        }
        return (results, timings) if return_timing else results
