"""
Unified Two-Stage Shop Item Recognition Pipeline
================================================
End-to-end inference orchestrator for Balatro shop screens:
1. Stage 1 (YOLO Detector): Localizes all cards/items in any shop ROI crop (joker bar, consumables, shop, pack).
2. Stage 2 (Multi-Head Classifier): Batched attribute classification on item crops.
Returns sorted left-to-right list of ShopItemResult objects.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as T
from PIL import Image
from ultralytics import YOLO

CURRENT_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = CURRENT_DIR / "weights"

try:
    from classifiers.shop_classifier.stage2_classifier.config import (
        CROP_HEIGHT,
        CROP_WIDTH,
        IDX_TO_EDITION,
        IDX_TO_KEY,
    )
    from classifiers.shop_classifier.stage2_classifier.dataset import (
        IMAGENET_MEAN,
        IMAGENET_STD,
    )
    from classifiers.shop_classifier.stage2_classifier.model import (
        MultiHeadItemClassifier,
        infer_backbone_from_checkpoint,
    )
except ImportError:
    from stage2_classifier.config import (
        CROP_HEIGHT,
        CROP_WIDTH,
        IDX_TO_EDITION,
        IDX_TO_KEY,
    )
    from stage2_classifier.dataset import IMAGENET_MEAN, IMAGENET_STD
    from stage2_classifier.model import (
        MultiHeadItemClassifier,
        infer_backbone_from_checkpoint,
    )


@dataclass
class ShopItemResult:
    index: int
    card_type: str  # 'Joker', 'Tarot', 'Spectral', 'Planet', 'Voucher', 'Booster', 'PlayingCard'
    key: str  # 'j_blueprint', 'c_fool', 'King of Hearts', etc.
    edition: str  # 'None', 'Foil', 'Holo', 'Polychrome', 'Negative'
    eternal: bool
    perishable: bool
    rental: bool
    debuffed: bool
    box: list[int]  # [x1, y1, x2, y2]
    confidences: dict[str, float]
    card_details: Any = None
    top_keys: list[tuple[str, float]] = field(default_factory=list)

    @property
    def display_name(self) -> str:
        if self.card_type == "PlayingCard" and self.card_details is not None:
            return getattr(
                self.card_details, "display_name", f"Playing Card ({self.key})"
            )

        parts = []
        if self.debuffed:
            parts.append("[DEBUFFED]")
        if self.edition != "None":
            parts.append(self.edition)
        if self.eternal:
            parts.append("[Eternal]")
        if self.perishable:
            parts.append("[Perishable]")
        if self.rental:
            parts.append("[Rental]")

        # Clean key to readable title (e.g. j_half_joker -> Half Joker, v_hone -> Hone, p_arcana_mega_1 -> Arcana Mega 1)
        clean_key = self.key
        if (
            clean_key.startswith("j_")
            or clean_key.startswith("c_")
            or clean_key.startswith("v_")
            or clean_key.startswith("p_")
        ):
            clean_key = clean_key[2:]
        readable_title = clean_key.replace("_", " ").title()
        parts.append(readable_title)

        return " ".join(parts)


def filter_shop_items(
    items: list[ShopItemResult],
    *,
    min_width: int = 45,
    key_threshold: float = 0.0,
    dedup_dx: float = 85.0,
    dedup_dy: float = 35.0,
    max_x_overlap_ratio: float = 1.0,
) -> list[ShopItemResult]:
    """Filters sliver detections, low-confidence keys, and split-half card duplicates."""
    candidates = [
        it
        for it in items
        if (it.box[2] - it.box[0]) >= min_width
        and it.confidences.get("key", 1.0) >= key_threshold
    ]
    # Prefer higher Stage-2 key confidence (with detection confidence tie-breaker)
    # so a full/better-framed half beats a clipped split-half duplicate.
    ranked = sorted(
        candidates,
        key=lambda it: (
            it.confidences.get("key", 1.0),
            it.confidences.get("detection", 0.0),
        ),
        reverse=True,
    )
    kept: list[ShopItemResult] = []
    for it in ranked:
        cx = 0.5 * (it.box[0] + it.box[2])
        cy = 0.5 * (it.box[1] + it.box[3])
        is_dup = False
        for prev in kept:
            if it.card_type != prev.card_type:
                continue
            pcx = 0.5 * (prev.box[0] + prev.box[2])
            pcy = 0.5 * (prev.box[1] + prev.box[3])
            if abs(cy - pcy) < dedup_dy:
                if abs(cx - pcx) < dedup_dx:
                    is_dup = True
                    break
                if max_x_overlap_ratio < 1.0:
                    x_ov = max(
                        0, min(it.box[2], prev.box[2]) - max(it.box[0], prev.box[0])
                    )
                    min_w = max(
                        1, min(it.box[2] - it.box[0], prev.box[2] - prev.box[0])
                    )
                    if (x_ov / min_w) >= max_x_overlap_ratio:
                        is_dup = True
                        break
        if not is_dup:
            kept.append(it)

    kept.sort(key=lambda it: it.box[0])
    for idx, it in enumerate(kept):
        it.index = idx
    return kept


class ShopRecognitionPipeline:
    def __init__(
        self,
        detector_path=None,
        classifier_path=None,
        backbone="mobilenet_v3_small",
        device=None,
        card_pipeline=None,
    ):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.card_pipeline = card_pipeline
        prod_weights_dir = CURRENT_DIR.parent / "prod_weights"
        if detector_path is not None:
            detector_path = Path(detector_path)
        else:
            detector_path = WEIGHTS_DIR / "detector_best.pt"
            if (
                not detector_path.exists()
                and (prod_weights_dir / "shop_detector.pt").exists()
            ):
                detector_path = prod_weights_dir / "shop_detector.pt"
            elif not detector_path.exists():
                candidates = list(
                    CURRENT_DIR.glob("stage1_detector/runs/*/weights/best.pt")
                )
                if candidates:
                    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    detector_path = candidates[0]

        if classifier_path is not None:
            classifier_path = Path(classifier_path)
        else:
            classifier_path = WEIGHTS_DIR / "classifier_best.pt"
            if (
                not classifier_path.exists()
                and (prod_weights_dir / "shop_classifier.pt").exists()
            ):
                classifier_path = prod_weights_dir / "shop_classifier.pt"
            elif not classifier_path.exists():
                candidates = list(CURRENT_DIR.glob("stage2_classifier/runs/*/best.pt"))
                if candidates:
                    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    classifier_path = candidates[0]

        print(f"[Pipeline] Loading Stage 1 detector: {detector_path}")
        self.detector = YOLO(str(detector_path))

        if classifier_path.exists():
            checkpoint = torch.load(classifier_path, map_location=self.device)
            resolved_backbone = infer_backbone_from_checkpoint(checkpoint, default_backbone=backbone or "mobilenet_v3_small")
            print(f"[Pipeline] Loading Stage 2 classifier: {classifier_path} ({resolved_backbone})")
            self.classifier = MultiHeadItemClassifier(
                backbone_name=resolved_backbone, pretrained=False
            ).to(self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            self.classifier.load_state_dict(state_dict)
            self.classifier.eval()
            print("[Pipeline] Weights loaded successfully.")
        else:
            print(f"[Pipeline] Loading Stage 2 classifier: {classifier_path} ({backbone})")
            self.classifier = MultiHeadItemClassifier(
                backbone_name=backbone, pretrained=False
            ).to(self.device)
            print(
                f"[Pipeline Warning] Classifier weights not found at: {classifier_path}"
            )

        self.transform = T.Compose(
            [
                T.Resize(
                    (CROP_HEIGHT, CROP_WIDTH),
                    interpolation=T.InterpolationMode.BILINEAR,
                ),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        self.class_names = {
            0: "Joker",
            1: "Tarot",
            2: "Spectral",
            3: "Planet",
            4: "Voucher",
            5: "Booster",
            6: "PlayingCard",
        }

    def predict(
        self,
        image_input,
        conf_threshold: float = 0.15,
        iou_threshold: float = 0.45,
        key_threshold: float = 0.30,
        dedup_dx: float = 85.0,
        min_width: int = 45,
        max_x_overlap_ratio: float = 1.0,
    ) -> list[ShopItemResult]:
        """
        Runs full two-stage inference on an ROI crop or screenshot.
        Args:
            image_input: PIL Image or path to image
        Returns:
            list[ShopItemResult] sorted left to right
        """
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise TypeError("Expected PIL Image or file path.")

        # 1. Stage 1 YOLO Detection
        det_results = self.detector.predict(
            source=pil_img,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device.type if self.device.type == "cpu" else 0,
            verbose=False,
        )

        boxes_with_info = []
        if det_results and len(det_results) > 0:
            boxes_obj = det_results[0].boxes
            if boxes_obj is not None and len(boxes_obj) > 0:
                xyxy = boxes_obj.xyxy.cpu().numpy()
                confs = boxes_obj.conf.cpu().numpy()
                classes = boxes_obj.cls.cpu().numpy()

                for i in range(len(xyxy)):
                    box = [int(round(coord)) for coord in xyxy[i]]
                    cls_id = int(classes[i])
                    card_type = self.class_names.get(cls_id, "Joker")
                    det_conf = float(confs[i])
                    boxes_with_info.append((box, card_type, det_conf))

        if not boxes_with_info:
            return []

        # Sort items strictly left-to-right by horizontal coordinate x1
        boxes_with_info.sort(key=lambda item: item[0][0])

        # Separate regular shop items and playing cards
        shop_indices = []
        shop_crop_tensors = []
        playing_card_indices = []
        playing_card_crops = []

        iw, ih = pil_img.size
        for i, (box, card_type, det_conf) in enumerate(boxes_with_info):
            x1, y1, x2, y2 = box
            x1 = max(0, min(iw - 1, x1))
            y1 = max(0, min(ih - 1, y1))
            x2 = max(x1 + 1, min(iw, x2))
            y2 = max(y1 + 1, min(ih, y2))

            crop = pil_img.crop((x1, y1, x2, y2))
            if card_type == "PlayingCard":
                playing_card_indices.append(i)
                playing_card_crops.append(crop)
            else:
                shop_indices.append(i)
                shop_crop_tensors.append(self.transform(crop))

        item_predictions = {}
        if shop_crop_tensors:
            batch_tensor = torch.stack(shop_crop_tensors).to(self.device)
            attr_preds = self.classifier.predict_attributes(batch_tensor)
            for idx, attrs in zip(shop_indices, attr_preds):
                item_predictions[idx] = attrs

        card_predictions = {}
        if playing_card_crops:
            if self.card_pipeline is None:
                try:
                    from classifiers.inference import get_card_pipeline

                    self.card_pipeline = get_card_pipeline(device=str(self.device))
                except Exception as e:
                    print(
                        f"[Pipeline Warning] Could not load CardRecognitionPipeline: {e}"
                    )

            if self.card_pipeline and hasattr(self.card_pipeline, "classify_crops"):
                card_boxes = [boxes_with_info[idx][0] for idx in playing_card_indices]
                card_results = self.card_pipeline.classify_crops(
                    playing_card_crops, boxes=card_boxes
                )
                for idx, c_res in zip(playing_card_indices, card_results):
                    card_predictions[idx] = c_res

        # 3. Assemble Output Results in original sorted left-to-right order
        results = []
        for idx, (box, card_type, det_conf) in enumerate(boxes_with_info):
            if card_type == "PlayingCard":
                c_res = card_predictions.get(idx)
                if c_res is not None:
                    card_key = (
                        f"{c_res.rank} of {c_res.suit}"
                        if c_res.rank != "Stone"
                        else "Stone Card"
                    )
                    edition = c_res.edition
                    debuffed = c_res.debuffed
                    conf_dict = {"detection": det_conf, **c_res.confidences}
                    card_det = c_res
                else:
                    card_key = "playing_card"
                    edition = "None"
                    debuffed = False
                    conf_dict = {"detection": det_conf}
                    card_det = None

                results.append(
                    ShopItemResult(
                        index=idx,
                        card_type=card_type,
                        key=card_key,
                        edition=edition,
                        eternal=False,
                        perishable=False,
                        rental=False,
                        debuffed=debuffed,
                        box=box,
                        confidences=conf_dict,
                        card_details=card_det,
                    )
                )
            else:
                attrs = item_predictions.get(
                    idx,
                    {
                        "key": "unknown",
                        "edition": "None",
                        "eternal": False,
                        "perishable": False,
                        "rental": False,
                        "debuffed": False,
                        "key_conf": 0.0,
                        "edition_conf": 0.0,
                        "eternal_conf": 0.0,
                        "perishable_conf": 0.0,
                        "rental_conf": 0.0,
                        "debuffed_conf": 0.0,
                    },
                )
                results.append(
                    ShopItemResult(
                        index=idx,
                        card_type=card_type,
                        key=attrs["key"],
                        edition=attrs["edition"],
                        eternal=attrs["eternal"],
                        perishable=attrs["perishable"],
                        rental=attrs["rental"],
                        debuffed=attrs["debuffed"],
                        box=box,
                        confidences={
                            "detection": det_conf,
                            "key": attrs["key_conf"],
                            "edition": attrs["edition_conf"],
                            "eternal": attrs["eternal_conf"],
                            "perishable": attrs["perishable_conf"],
                            "rental": attrs["rental_conf"],
                            "debuffed": attrs["debuffed_conf"],
                        },
                        card_details=None,
                        top_keys=list(attrs.get("top_keys", ())),
                    )
                )

        return filter_shop_items(
            results,
            min_width=min_width,
            key_threshold=key_threshold,
            dedup_dx=dedup_dx,
            max_x_overlap_ratio=max_x_overlap_ratio,
        )
