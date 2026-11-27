"""
Balatro Frame Recognition Interface for Post-Processing Pipeline
================================================================
Exposes lightweight, high-performance recognition functions intended for direct
import and execution in `post_processing.py` (or live bot loops).

Uses the production models stored in `classifiers/prod_weights/`:
  - card_detector.pt + card_classifier.pt
  - shop_detector.pt + shop_classifier.pt
  - end_round_classifier.pt

Supports:
  1. Single-frame immediate inference (`recognize_hand_cards`, `recognize_shop`)
  2. High-throughput grouped batching (`batch_recognize_hands`, `batch_recognize_shops`)
     Group-batches frames by identical ROI shelf dimensions so the GPU processes
     hundreds of crops in uniform tensor batches at full saturation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence, Union
import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROD_WEIGHTS_DIR = PROJECT_ROOT / "classifiers" / "prod_weights"

# Explicit relative imports from pipelines
# pyrefly: ignore [missing-import]
from classifiers.card_classifier.pipeline import (
    CardRecognitionPipeline,
    CardResult,
    decode_card_prediction,
    determine_card_selection,
    is_valid_hand_card_box,
)

# pyrefly: ignore [missing-import]
from classifiers.shop_classifier.pipeline import (
    ShopRecognitionPipeline,
    ShopItemResult,
    filter_shop_items,
)

# pyrefly: ignore [missing-import]
from classifiers.end_round_classifier.config import PLAY_AREA_ROI

# pyrefly: ignore [missing-import]
from classifiers.end_round_classifier.pipeline import (
    EndRoundRecognitionPipeline,
    EndRoundResult,
)

# pyrefly: ignore [missing-import]
from classifiers.end_round_classifier.video_io import (
    frames_sequence_to_tensor,
    read_video_segments_rgb_batch,
    resample_frame_indices,
)

# pyrefly: ignore [missing-import]
from classifiers.common.video_utils import scale_roi_pixels

# pyrefly: ignore [missing-import]
from classifiers.hud_ocr.config import (
    DEFAULT_OCR_CONF_THRESHOLD,
    HUD_ROIS,
    HUD_ROI_FIELDS,
)

# pyrefly: ignore [missing-import]
from classifiers.hud_ocr.pipeline import HudOcrPipeline, HudOcrResult

# pyrefly: ignore [missing-import]
from classifiers.deck_classifier.config import resolve_view_deck_roi

from eval_template import detect_view_deck_tab_mode

BASE_WIDTH = 1920.0
BASE_HEIGHT = 1080.0

# 1920x1080 Balatro UI ROI coordinates (calibrated in balatro_mod/dataset/boxes.lua)
ROIS = {
    "card_hand": {"x": 508.319934, "y": 567.839093, "w": 1071.056521, "h": 322.734447},
    "pack_hand": {"x": 490.0, "y": 300.0, "w": 1100.0, "h": 340.0},
    "joker_bar": {"x": 512.701385, "y": 20.409878, "w": 857.0, "h": 259.967609},
    "consumeables": {
        "x": 1369.783812,
        "y": 30.836806,
        "w": 425.206356,
        "h": 270.406328,
    },
    "shop": {"x": 818.386186, "y": 369.181989, "w": 720.772049, "h": 280.979925},
    "voucher": {"x": 656.059547, "y": 714.459732, "w": 360.743773, "h": 284.420985},
    "booster": {"x": 1071.358597, "y": 724.761198, "w": 427.707911, "h": 300.000000},
    "pack": {"x": 510.0, "y": 575.0, "w": 1070.0, "h": 330.0},
    "view_deck": resolve_view_deck_roi(),
    "play_area": dict(PLAY_AREA_ROI),
    **{k: dict(v) for k, v in HUD_ROIS.items()},
}


# =====================================================================
# Output Data Structures
# =====================================================================


@dataclass(slots=True)
class HandRecognitionOutput:
    """Output results for player hand detection."""

    frame_index: int
    timestamp_seconds: float
    cards: list[CardResult] = field(default_factory=list)
    roi: tuple[int, int, int, int] = (0, 0, 0, 0)  # (x, y, w, h)


@dataclass(slots=True)
class ShopRecognitionOutput:
    """Complete shop frame evaluation results across all shelves."""

    frame_index: int
    timestamp_seconds: float
    mode: str = "shop"  # 'shop' or 'pack'
    joker_bar: list[ShopItemResult] = field(default_factory=list)
    consumeables: list[ShopItemResult] = field(default_factory=list)
    shop_items: list[ShopItemResult] = field(default_factory=list)
    vouchers: list[ShopItemResult] = field(default_factory=list)
    boosters: list[ShopItemResult] = field(default_factory=list)
    pack_items: list[ShopItemResult] = field(default_factory=list)

    @property
    def all_items(self) -> list[ShopItemResult]:
        """Flattened list of all detected shop items across all regions."""
        return (
            self.joker_bar
            + self.consumeables
            + self.shop_items
            + self.vouchers
            + self.boosters
            + self.pack_items
        )


@dataclass(slots=True)
class HudRecognitionOutput:
    """HUD OCR predictions and parsed numeric values for a single frame."""

    frame_index: int
    timestamp_seconds: float
    results: dict[str, HudOcrResult] = field(default_factory=dict)
    values: dict[str, int | float | None] = field(default_factory=dict)

    @property
    def round_score(self) -> int | float | None:
        return self.values.get("hud_round_score")

    @property
    def money(self) -> int | float | None:
        return self.values.get("hud_money")

    @property
    def hands(self) -> int | float | None:
        return self.values.get("hud_hands")

    @property
    def discards(self) -> int | float | None:
        return self.values.get("hud_discards")

    @property
    def ante(self) -> int | float | None:
        return self.values.get("hud_ante")

    @property
    def round(self) -> int | float | None:
        return self.values.get("hud_round")

    @property
    def reroll_cost(self) -> int | float | None:
        return self.values.get("hud_reroll_cost")

    @property
    def blind_target(self) -> int | float | None:
        return self.values.get("hud_blind_target")


@dataclass(slots=True)
class DeckRecognitionOutput:
    """Output results for full-deck / remaining-deck View Deck modal recognition."""

    frame_index: int
    timestamp_seconds: float
    cards: list[CardResult] = field(default_factory=list)
    view_mode: str = "full_deck"  # "full_deck" or "remaining"
    roi: tuple[int, int, int, int] = (0, 0, 0, 0)


# =====================================================================
# Model Cache & Lazy Initialization
# =====================================================================

_CARD_PIPELINE: CardRecognitionPipeline | None = None
_SHOP_PIPELINE: ShopRecognitionPipeline | None = None
_END_ROUND_PIPELINE: EndRoundRecognitionPipeline | None = None
_HUD_OCR_PIPELINE: HudOcrPipeline | None = None
_SEED_OCR_PIPELINE: Any = None
_DECK_PIPELINE: Any = None


def get_card_pipeline(device: str | None = None) -> CardRecognitionPipeline:
    """Returns singleton CardRecognitionPipeline loaded from classifiers/prod_weights/."""
    global _CARD_PIPELINE
    if _CARD_PIPELINE is None:
        det_path = PROD_WEIGHTS_DIR / "card_detector.pt"
        cls_path = PROD_WEIGHTS_DIR / "card_classifier.pt"
        if not det_path.exists():
            raise FileNotFoundError(
                f"Missing production card detector weights at {det_path}"
            )
        if not cls_path.exists():
            raise FileNotFoundError(
                f"Missing production card classifier weights at {cls_path}"
            )

        _CARD_PIPELINE = CardRecognitionPipeline(
            detector_path=det_path, classifier_path=cls_path, device=device
        )
    return _CARD_PIPELINE


def get_shop_pipeline(device: str | None = None) -> ShopRecognitionPipeline:
    """Returns singleton ShopRecognitionPipeline loaded from classifiers/prod_weights/."""
    global _SHOP_PIPELINE
    if _SHOP_PIPELINE is None:
        det_path = PROD_WEIGHTS_DIR / "shop_detector.pt"
        cls_path = PROD_WEIGHTS_DIR / "shop_classifier.pt"
        if not det_path.exists():
            raise FileNotFoundError(
                f"Missing production shop detector weights at {det_path}"
            )
        if not cls_path.exists():
            raise FileNotFoundError(
                f"Missing production shop classifier weights at {cls_path}"
            )

        _SHOP_PIPELINE = ShopRecognitionPipeline(
            detector_path=det_path, classifier_path=cls_path, device=device
        )
    return _SHOP_PIPELINE


def get_end_round_pipeline(device: str | None = None) -> EndRoundRecognitionPipeline:
    """Returns singleton EndRoundRecognitionPipeline loaded from classifiers/prod_weights/."""
    global _END_ROUND_PIPELINE
    if _END_ROUND_PIPELINE is None:
        ckpt_path = PROD_WEIGHTS_DIR / "end_round_classifier.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Missing production END_ROUND classifier weights at {ckpt_path}"
            )
        _END_ROUND_PIPELINE = EndRoundRecognitionPipeline(
            weights_path=ckpt_path,
            device=device,
        )
    return _END_ROUND_PIPELINE


def get_hud_ocr_pipeline(device: str | None = None) -> HudOcrPipeline:
    """Returns singleton HudOcrPipeline loaded from classifiers/prod_weights/."""
    global _HUD_OCR_PIPELINE
    if _HUD_OCR_PIPELINE is None:
        ckpt_path = PROD_WEIGHTS_DIR / "hud_ocr.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Missing production HUD OCR weights at {ckpt_path}"
            )
        _HUD_OCR_PIPELINE = HudOcrPipeline(
            weights_path=ckpt_path,
            device=device,
        )
    return _HUD_OCR_PIPELINE


def get_seed_ocr_pipeline(device: str | None = None):
    """Returns singleton SeedOcrPipeline loaded from classifiers/prod_weights/seed_ocr.pt."""
    global _SEED_OCR_PIPELINE
    if _SEED_OCR_PIPELINE is None:
        from classifiers.seed_ocr.pipeline import SeedOcrPipeline

        ckpt_path = PROD_WEIGHTS_DIR / "seed_ocr.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Missing production Seed OCR weights at {ckpt_path}"
            )
        _SEED_OCR_PIPELINE = SeedOcrPipeline(
            weights_path=ckpt_path,
            device=device,
        )
    return _SEED_OCR_PIPELINE


def get_deck_pipeline(device: str | None = None):
    """Lazily loads singleton DeckRecognitionPipeline from classifiers/prod_weights/."""
    global _DECK_PIPELINE
    if _DECK_PIPELINE is None:
        from classifiers.deck_classifier.pipeline import DeckRecognitionPipeline

        det_path = PROD_WEIGHTS_DIR / "deck_detector.pt"
        cls_path = PROD_WEIGHTS_DIR / "card_classifier.pt"
        if not det_path.exists():
            raise FileNotFoundError(
                f"Missing production deck detector weights at {det_path}"
            )
        if not cls_path.exists():
            raise FileNotFoundError(
                f"Missing production card classifier weights at {cls_path}"
            )
        _DECK_PIPELINE = DeckRecognitionPipeline(
            detector_path=det_path,
            classifier_path=cls_path,
            device=device,
        )
    return _DECK_PIPELINE


def _extract_frame_image_and_meta(video_frame: Any) -> tuple[np.ndarray, int, float]:
    """Helper to unwrap VideoFrame, tuple, or raw np.ndarray."""
    if hasattr(video_frame, "image") and hasattr(video_frame, "index"):
        return (
            video_frame.image,
            getattr(video_frame, "index", 0),
            getattr(video_frame, "timestamp_seconds", 0.0),
        )
    elif isinstance(video_frame, tuple) and len(video_frame) >= 2:
        return (
            video_frame[0],
            video_frame[1],
            (video_frame[2] if len(video_frame) > 2 else 0.0),
        )
    elif isinstance(video_frame, np.ndarray):
        return video_frame, 0, 0.0
    else:
        raise TypeError(
            f"Unsupported frame type: {type(video_frame)}. Expected VideoFrame or np.ndarray."
        )


def _crop_roi_rgb(
    full_bgr: np.ndarray, roi_key: str
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crops a region from full frame BGR array into an RGB uint8 ndarray."""
    fh, fw = full_bgr.shape[:2]
    rx, ry, rw, rh = scale_roi_pixels(ROIS[roi_key], fw, fh, BASE_WIDTH, BASE_HEIGHT)
    crop_bgr = full_bgr[ry : ry + rh, rx : rx + rw]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return crop_rgb, (rx, ry, rw, rh)


def _crop_roi_to_pil(
    full_bgr: np.ndarray, roi_key: str
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Crops a region from full frame BGR array to PIL Image in RGB format."""
    crop_rgb, coords = _crop_roi_rgb(full_bgr, roi_key)
    return Image.fromarray(crop_rgb), coords


def _offset_box(box: list[int], rx: int, ry: int) -> list[int]:
    """Translates an ROI-local [x1, y1, x2, y2] box into full-frame pixel coordinates."""
    return [box[0] + rx, box[1] + ry, box[2] + rx, box[3] + ry]


def _translate_item_boxes(items: Sequence[Any], rx: int, ry: int) -> Any:
    """Translates item.box in-place from ROI-local pixels into full-frame pixels."""
    for item in items:
        item.box = _offset_box(item.box, rx, ry)
    return items


# =====================================================================
# Single Frame Recognition Functions
# =====================================================================


def recognize_hand_cards(
    video_frame: Any,
    conf_threshold: float = 0.50,
    device: str | None = None,
    roi_key: str = "card_hand",
) -> HandRecognitionOutput:
    """Evaluates the player hand on a single video frame (boxes returned in full-frame pixels)."""
    img_bgr, f_idx, ts = _extract_frame_image_and_meta(video_frame)
    pipeline = get_card_pipeline(device=device)

    crop_pil, roi_coords = _crop_roi_to_pil(img_bgr, roi_key)
    cards = pipeline.predict(crop_pil, conf=conf_threshold)
    _translate_item_boxes(cards, roi_coords[0], roi_coords[1])

    return HandRecognitionOutput(
        frame_index=f_idx, timestamp_seconds=ts, cards=cards, roi=roi_coords
    )


def recognize_full_deck(
    video_frame: Any,
    *,
    view_mode: str = "full_deck",
    conf_threshold: float = 0.45,
    iou_threshold: float = 0.55,
    device: str | None = None,
) -> DeckRecognitionOutput:
    """Evaluates Balatro's View Deck overlay on a single frame (boxes returned in full-frame pixels)."""
    img_bgr, f_idx, ts = _extract_frame_image_and_meta(video_frame)
    pipeline = get_deck_pipeline(device=device)

    crop_pil, roi_coords = _crop_roi_to_pil(img_bgr, "view_deck")
    cards = pipeline.predict(crop_pil, conf=conf_threshold, iou=iou_threshold)
    _translate_item_boxes(cards, roi_coords[0], roi_coords[1])

    return DeckRecognitionOutput(
        frame_index=f_idx,
        timestamp_seconds=ts,
        cards=cards,
        view_mode=view_mode,
        roi=roi_coords,
    )


def _get_shelves_for_mode(mode: str) -> list[str]:
    """Returns the ROI shelf keys evaluated for a given shop/panel recognition mode."""
    if mode == "joker_only":
        return ["joker_bar"]
    if mode == "bars":
        return ["joker_bar", "consumeables"]
    if mode == "pack_only":
        return ["pack"]
    if mode == "pack":
        return ["joker_bar", "consumeables", "pack"]
    return ["joker_bar", "consumeables", "shop", "voucher", "booster"]


def _filter_kwargs_for_shelf(shelf_key: str, key_threshold: float) -> dict[str, Any]:
    """Returns shelf-appropriate deduplication thresholds; dense fanned bars allow tighter spacing."""
    kwargs: dict[str, Any] = {"key_threshold": key_threshold}
    if shelf_key in ("joker_bar", "consumeables"):
        kwargs["dedup_dx"] = 38.0
        kwargs["max_x_overlap_ratio"] = 0.50
    if shelf_key == "joker_bar":
        kwargs["min_width"] = 50
    return kwargs


def recognize_shop(
    video_frame: Any,
    mode: str = "shop",
    conf_threshold: float = 0.15,
    device: str | None = None,
    key_threshold: float = 0.30,
) -> ShopRecognitionOutput:
    """Evaluates all relevant shelves on a shop, pack, or HUD frame (boxes returned in full-frame pixels)."""
    img_bgr, f_idx, ts = _extract_frame_image_and_meta(video_frame)
    pipeline = get_shop_pipeline(device=device)

    output = ShopRecognitionOutput(frame_index=f_idx, timestamp_seconds=ts, mode=mode)

    shelves = _get_shelves_for_mode(mode)
    shelf_attr_map = {
        "joker_bar": "joker_bar",
        "consumeables": "consumeables",
        "pack": "pack_items",
        "shop": "shop_items",
        "voucher": "vouchers",
        "booster": "boosters",
    }
    for shelf_key in shelves:
        crop_pil, roi = _crop_roi_to_pil(img_bgr, shelf_key)
        fkw = _filter_kwargs_for_shelf(shelf_key, key_threshold)
        if isinstance(pipeline, ShopRecognitionPipeline):
            raw_items = pipeline.predict(
                crop_pil,
                conf_threshold=conf_threshold,
                **fkw,
            )
        else:
            if key_threshold < 0.30:
                raw_items = pipeline.predict(
                    crop_pil,
                    conf_threshold=conf_threshold,
                    key_threshold=key_threshold,
                )
            else:
                raw_items = pipeline.predict(crop_pil, conf_threshold=conf_threshold)
            raw_items = filter_shop_items(raw_items, **fkw)
        items = _translate_item_boxes(raw_items, roi[0], roi[1])
        setattr(output, shelf_attr_map[shelf_key], items)

    return output


# =====================================================================
# High-Throughput Grouped Batch Inference Functions
# =====================================================================


def batch_recognize_hands(
    frames: Sequence[Any],
    conf_threshold: float = 0.50,
    batch_size: int = 32,
    device: str | None = None,
    roi_key: str = "card_hand",
) -> list[HandRecognitionOutput]:
    """
    Batches player hand card detection across a collected list of frames.
    All hand crops for a given `roi_key` share identical dimensions, so they are batched
    directly into YOLO Stage 1 and Stage 2 CNN in parallel.
    """
    if not frames:
        return []

    pipeline = get_card_pipeline(device=device)
    unwrapped = [_extract_frame_image_and_meta(f) for f in frames]

    outputs: list[HandRecognitionOutput] = []

    for chunk_start in range(0, len(unwrapped), batch_size):
        chunk = unwrapped[chunk_start : chunk_start + batch_size]
        pil_crops = []
        metas = []

        for img_bgr, f_idx, ts in chunk:
            crop_pil, roi_coords = _crop_roi_to_pil(img_bgr, roi_key)
            pil_crops.append(crop_pil)
            metas.append((f_idx, ts, roi_coords))

        # Batch Stage 1 YOLO detection
        det_results = pipeline.detector.predict(
            source=pil_crops,
            conf=conf_threshold,
            device=pipeline.device.type if pipeline.device.type == "cpu" else 0,
            verbose=False,
        )

        # Collect card crops for global Stage 2 CNN batching
        chunk_cards_per_frame: list[list[CardResult]] = []

        all_crops_tensors = []
        crop_metadata = []  # (frame_idx_in_chunk, box, ih)

        for f_i, det in enumerate(det_results):
            boxes_obj = det.boxes
            if boxes_obj is None or len(boxes_obj) == 0:
                continue

            xyxy = boxes_obj.xyxy.cpu().numpy()
            order = np.argsort(xyxy[:, 0])
            sorted_boxes = xyxy[order]

            cur_pil = pil_crops[f_i]
            iw, ih = cur_pil.size

            for b in sorted_boxes:
                x1 = max(0, min(iw - 1, int(b[0])))
                y1 = max(0, min(ih - 1, int(b[1])))
                x2 = max(x1 + 1, min(iw, int(b[2])))
                y2 = max(y1 + 1, min(ih, int(b[3])))
                if not is_valid_hand_card_box([x1, y1, x2, y2]):
                    continue

                crop = cur_pil.crop((x1, y1, x2, y2))
                all_crops_tensors.append(pipeline.crop_transform(crop))
                crop_metadata.append((f_i, [x1, y1, x2, y2], ih))

        # Stage 2 Batched CNN forward pass across all cards in this chunk
        all_card_preds = {}
        if all_crops_tensors:
            batched_cards = torch.stack(all_crops_tensors).to(pipeline.device)
            with torch.no_grad():
                all_card_preds = pipeline.classifier.predict_attributes(batched_cards)

        # Assemble per-frame outputs
        frame_cards_map: dict[int, list[CardResult]] = {
            i: [] for i in range(len(chunk))
        }
        for pred_idx, (f_i, box, _) in enumerate(crop_metadata):
            card_res = decode_card_prediction(
                all_card_preds,
                pred_idx,
                index=len(frame_cards_map[f_i]) + 1,
                box=box,
                selected=False,  # Updated below
            )
            frame_cards_map[f_i].append(card_res)

        # Calculate card raise/selection per frame before translating boxes to full-frame space
        for f_i, cards_list in frame_cards_map.items():
            f_idx, ts, roi_coords = metas[f_i]
            if cards_list:
                rx, ry, _, rh = roi_coords
                crop_ih = max(1, rh)  # roi_coords is (rx, ry, rw, rh)
                boxes_only = [c.box for c in cards_list]
                selected_flags = determine_card_selection(boxes_only, crop_ih)
                for c_obj, is_sel in zip(cards_list, selected_flags):
                    c_obj.selected = is_sel
                _translate_item_boxes(cards_list, rx, ry)

            outputs.append(
                HandRecognitionOutput(
                    frame_index=f_idx,
                    timestamp_seconds=ts,
                    cards=cards_list,
                    roi=roi_coords,
                )
            )

    return outputs


def batch_recognize_shops(
    frames: Sequence[Any],
    mode: str = "shop",
    conf_threshold: float = 0.15,
    batch_size: int = 32,
    device: str | None = None,
    key_threshold: float = 0.30,
) -> list[ShopRecognitionOutput]:
    """
    Group-batches shop detection across collected frames (boxes returned in full-frame pixels).
    Groups crops by shelf region (joker_bar, shop, voucher, booster, etc.) so that
    each batch consists of uniformly-sized images sent to the GPU together.
    """
    if not frames:
        return []

    pipeline = get_shop_pipeline(device=device)
    unwrapped = [_extract_frame_image_and_meta(f) for f in frames]

    outputs = [
        ShopRecognitionOutput(frame_index=f_idx, timestamp_seconds=ts, mode=mode)
        for _, f_idx, ts in unwrapped
    ]

    shelves = _get_shelves_for_mode(mode)

    for shelf_key in shelves:
        # 1. Extract uniformly shaped crops for this specific shelf across all frames
        shelf_crops = []
        shelf_rois = []
        for img_bgr, _, _ in unwrapped:
            crop_pil, roi_coords = _crop_roi_to_pil(img_bgr, shelf_key)
            shelf_crops.append(crop_pil)
            shelf_rois.append(roi_coords)

        # 2. Batch process in chunks of `batch_size`
        for chunk_start in range(0, len(shelf_crops), batch_size):
            chunk_crops = shelf_crops[chunk_start : chunk_start + batch_size]
            chunk_indices = list(range(chunk_start, chunk_start + len(chunk_crops)))

            det_results = pipeline.detector.predict(
                source=chunk_crops,
                conf=conf_threshold,
                device=pipeline.device.type if pipeline.device.type == "cpu" else 0,
                verbose=False,
            )

            # Collect Stage 2 crops across the chunk, separating regular items and playing cards
            all_stage2_tensors = []
            shop_metadata = []  # (global_idx, box, card_type, det_conf)
            playing_card_crops = []
            card_metadata = []  # (global_idx, box, card_type, det_conf)

            for local_idx, det in enumerate(det_results):
                global_idx = chunk_indices[local_idx]
                cur_pil = chunk_crops[local_idx]
                iw, ih = cur_pil.size

                boxes_obj = det.boxes
                if boxes_obj is None or len(boxes_obj) == 0:
                    continue

                xyxy = boxes_obj.xyxy.cpu().numpy()
                confs = boxes_obj.conf.cpu().numpy()
                classes = boxes_obj.cls.cpu().numpy()

                for b, c_id, conf in zip(xyxy, classes, confs):
                    x1 = max(0, min(iw - 1, int(round(b[0]))))
                    y1 = max(0, min(ih - 1, int(round(b[1]))))
                    x2 = max(x1 + 1, min(iw, int(round(b[2]))))
                    y2 = max(y1 + 1, min(ih, int(round(b[3]))))

                    crop = cur_pil.crop((x1, y1, x2, y2))
                    c_type = pipeline.class_names.get(int(c_id), "Joker")
                    if c_type == "PlayingCard":
                        playing_card_crops.append(crop)
                        card_metadata.append(
                            (global_idx, [x1, y1, x2, y2], c_type, float(conf))
                        )
                    else:
                        all_stage2_tensors.append(pipeline.transform(crop))
                        shop_metadata.append(
                            (global_idx, [x1, y1, x2, y2], c_type, float(conf))
                        )

            # Batch Stage 2 MobileNet inference for shop items
            attr_preds = []
            if all_stage2_tensors:
                batched_tensor = torch.stack(all_stage2_tensors).to(pipeline.device)
                with torch.no_grad():
                    attr_preds = pipeline.classifier.predict_attributes(batched_tensor)

            # Classify playing cards via CardRecognitionPipeline
            card_preds = []
            if playing_card_crops:
                card_pipe = get_card_pipeline(device=device)
                card_preds = card_pipe.classify_crops(playing_card_crops)

            # Distribute regular shop items back into their respective frame output
            for (g_idx, box, card_type, det_conf), attrs in zip(
                shop_metadata, attr_preds
            ):
                rx, ry = shelf_rois[g_idx][:2]
                item_res = ShopItemResult(
                    index=0,  # Ordered below
                    card_type=card_type,
                    key=attrs["key"],
                    edition=attrs["edition"],
                    eternal=attrs["eternal"],
                    perishable=attrs["perishable"],
                    rental=attrs["rental"],
                    debuffed=attrs["debuffed"],
                    box=_offset_box(box, rx, ry),
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

                out_obj = outputs[g_idx]
                if shelf_key == "joker_bar":
                    out_obj.joker_bar.append(item_res)
                elif shelf_key == "consumeables":
                    out_obj.consumeables.append(item_res)
                elif shelf_key == "shop":
                    out_obj.shop_items.append(item_res)
                elif shelf_key == "voucher":
                    out_obj.vouchers.append(item_res)
                elif shelf_key == "booster":
                    out_obj.boosters.append(item_res)
                elif shelf_key == "pack":
                    out_obj.pack_items.append(item_res)

            # Distribute playing cards back into their respective frame output
            for (g_idx, box, card_type, det_conf), c_res in zip(
                card_metadata, card_preds
            ):
                rx, ry = shelf_rois[g_idx][:2]
                card_key = (
                    f"{c_res.rank} of {c_res.suit}"
                    if c_res.rank != "Stone"
                    else "Stone Card"
                )
                item_res = ShopItemResult(
                    index=0,
                    card_type=card_type,
                    key=card_key,
                    edition=c_res.edition,
                    eternal=False,
                    perishable=False,
                    rental=False,
                    debuffed=c_res.debuffed,
                    box=_offset_box(box, rx, ry),
                    confidences={"detection": det_conf, **c_res.confidences},
                    card_details=c_res,
                )

                out_obj = outputs[g_idx]
                if shelf_key == "pack":
                    out_obj.pack_items.append(item_res)
                else:
                    out_obj.shop_items.append(item_res)

        # Filter slivers/low-key/split-half duplicates and sort left-to-right
        attr_name = (
            "pack_items"
            if shelf_key == "pack"
            else (
                "shop_items"
                if shelf_key == "shop"
                else (
                    "vouchers"
                    if shelf_key == "voucher"
                    else ("boosters" if shelf_key == "booster" else shelf_key)
                )
            )
        )
        filter_kwargs = _filter_kwargs_for_shelf(shelf_key, key_threshold)
        for out_obj in outputs:
            filtered_list = filter_shop_items(
                getattr(out_obj, attr_name),
                **filter_kwargs,
            )
            setattr(out_obj, attr_name, filtered_list)

    return outputs


def process_frames_iterator(
    frames_iterator: Iterable[Any],
    target: str = "hand",
    mode: str = "shop",
    conf_threshold: float | None = None,
) -> Iterator[Union[HandRecognitionOutput, ShopRecognitionOutput]]:
    """Yields frame-by-frame recognition outputs from an iterator."""
    if target == "hand":
        thresh = conf_threshold if conf_threshold is not None else 0.50
        for f in frames_iterator:
            yield recognize_hand_cards(f, conf_threshold=thresh)
    elif target == "shop":
        thresh = conf_threshold if conf_threshold is not None else 0.15
        for f in frames_iterator:
            yield recognize_shop(f, mode=mode, conf_threshold=thresh)
    else:
        raise ValueError(f"Unknown target '{target}'. Must be 'hand' or 'shop'.")


def recognize_end_round_segment(
    video_path: str | Path,
    start_frame: int,
    end_frame: int,
    *,
    device: str | None = None,
) -> EndRoundResult:
    """Runs temporal END_ROUND detection over one `[start_frame, end_frame)` video window."""
    pipeline = get_end_round_pipeline(device=device)
    return pipeline.predict_video_segment(
        video_path,
        start_frame=start_frame,
        end_frame=end_frame,
    )


def batch_recognize_end_rounds(
    segments: Sequence[tuple[int, int]],
    *,
    video_path: str | Path | None = None,
    preloaded_frames: dict[int, Any] | None = None,
    fps: float = 60.0,
    batch_size: int = 8,
    device: str | None = None,
    show_progress: bool = False,
) -> list[EndRoundResult]:
    """Batches temporal END_ROUND detection across multiple `[PLAY_HAND, CASH_OUT]` segments."""
    if not segments:
        return []
    if video_path is None and preloaded_frames is None:
        return []

    pbar = None
    if show_progress:
        from tqdm import tqdm

        pbar = tqdm(
            total=len(segments),
            desc="Detecting END_ROUND transitions",
            unit="segment",
        )
        pbar.set_postfix_str("reading video")

    if video_path is not None:
        pipeline = get_end_round_pipeline(device=device)
        clip_tensors, src_fps, sampled_indices_list = read_video_segments_rgb_batch(
            video_path,
            segments,
            target_fps=pipeline.target_fps,
            roi_base_1080p=pipeline.roi_base_1080p,
            out_width=pipeline.input_width,
            out_height=pipeline.input_height,
            progress_callback=pbar.update if pbar is not None else None,
        )
    else:
        assert preloaded_frames is not None
        sorted_avail = sorted(preloaded_frames.keys())
        avail_per_seg: list[list[int]] = []
        for s_frame, e_frame in segments:
            s_i, e_i = int(s_frame), int(e_frame)
            in_range = [fid for fid in sorted_avail if s_i <= fid < e_i]
            if len(in_range) < 2:
                # Sparse button-frame test fixtures do not hold contiguous segment frames.
                if pbar is not None:
                    pbar.close()
                return []
            avail_per_seg.append(in_range)

        pipeline = get_end_round_pipeline(device=device)
        src_fps = float(fps) if fps > 0 else 60.0
        clip_tensors = []
        sampled_indices_list = []
        for (s_frame, e_frame), in_range in zip(segments, avail_per_seg):
            desired = resample_frame_indices(
                int(s_frame), int(e_frame), src_fps, pipeline.target_fps
            )
            if len(desired) >= 2 and all(idx in preloaded_frames for idx in desired):
                chosen_ids = desired
            else:
                chosen_ids = in_range
            bgr_seq = [
                _extract_frame_image_and_meta(preloaded_frames[fid])[0]
                for fid in chosen_ids
            ]
            clip_tchw = frames_sequence_to_tensor(
                bgr_seq,
                is_bgr=True,
                already_cropped=False,
                roi_base_1080p=pipeline.roi_base_1080p,
                out_width=pipeline.input_width,
                out_height=pipeline.input_height,
            )
            clip_tensors.append(clip_tchw)
            sampled_indices_list.append(chosen_ids)

    chunk_size = max(1, int(batch_size))
    results: list[EndRoundResult] = []
    if pbar is not None:
        pbar.reset(total=len(clip_tensors))
        pbar.set_postfix_str("inferring")
    for chunk_start in range(0, len(clip_tensors), chunk_size):
        chunk_clips = clip_tensors[chunk_start : chunk_start + chunk_size]
        chunk_indices = sampled_indices_list[chunk_start : chunk_start + chunk_size]
        bsz = len(chunk_clips)
        max_t = max(int(t.shape[0]) for t in chunk_clips)
        _, c, h, w = chunk_clips[0].shape
        padded = torch.zeros((bsz, max_t, c, h, w), dtype=chunk_clips[0].dtype)
        lengths = torch.tensor([int(t.shape[0]) for t in chunk_clips], dtype=torch.long)
        for i, clip_t in enumerate(chunk_clips):
            padded[i, : int(clip_t.shape[0])] = clip_t

        results.extend(
            pipeline.predict_batch(
                padded,
                lengths=lengths,
                fps_list=[src_fps] * bsz,
                sampled_video_frames_list=chunk_indices,
            )
        )
        if pbar is not None:
            pbar.update(bsz)

    if pbar is not None:
        pbar.close()

    return results


def recognize_hud(
    video_frame: Any,
    fields: Sequence[str] | None = None,
    conf_threshold: float = DEFAULT_OCR_CONF_THRESHOLD,
    device: str | None = None,
) -> HudRecognitionOutput:
    """Runs batched HUD OCR across selected HUD ROIs on a single video frame."""
    target_fields = tuple(fields) if fields is not None else HUD_ROI_FIELDS
    return batch_recognize_hud(
        [video_frame],
        fields=target_fields,
        conf_threshold=conf_threshold,
        batch_size=max(1, len(target_fields)),
        device=device,
    )[0]


def batch_recognize_hud(
    frames: Sequence[Any],
    fields_by_frame: dict[int, Sequence[str]] | None = None,
    fields: Sequence[str] | None = None,
    conf_threshold: float = DEFAULT_OCR_CONF_THRESHOLD,
    batch_size: int = 128,
    device: str | None = None,
) -> list[HudRecognitionOutput]:
    """Batches HUD OCR inference across multiple frames and per-frame ROI field subsets."""
    if not frames:
        return []

    default_fields = tuple(fields) if fields is not None else HUD_ROI_FIELDS
    pipeline = get_hud_ocr_pipeline(device=device)
    unwrapped = [_extract_frame_image_and_meta(f) for f in frames]

    outputs = [
        HudRecognitionOutput(frame_index=f_idx, timestamp_seconds=ts)
        for _, f_idx, ts in unwrapped
    ]

    all_crops: list[np.ndarray] = []
    all_field_names: list[str] = []
    crop_owners: list[tuple[int, str]] = []

    for out_idx, (img_bgr, f_idx, _) in enumerate(unwrapped):
        frame_fields = (
            tuple(fields_by_frame.get(f_idx, default_fields))
            if fields_by_frame is not None
            else default_fields
        )
        for fname in frame_fields:
            crop_rgb, _ = _crop_roi_rgb(img_bgr, fname)
            all_crops.append(crop_rgb)
            all_field_names.append(fname)
            crop_owners.append((out_idx, fname))

    if not all_crops:
        return outputs

    chunk_size = max(1, int(batch_size))
    preds: list[HudOcrResult] = []
    for start in range(0, len(all_crops), chunk_size):
        preds.extend(
            pipeline.predict_crops(
                all_crops[start : start + chunk_size],
                field_names=all_field_names[start : start + chunk_size],
                conf_threshold=conf_threshold,
            )
        )

    for (out_idx, fname), res in zip(crop_owners, preds):
        outputs[out_idx].results[fname] = res
        outputs[out_idx].values[fname] = res.parsed_value

    return outputs
