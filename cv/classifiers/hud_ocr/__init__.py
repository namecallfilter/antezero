"""Lightweight GPU-batched HUD OCR package for Balatro telemetry reconstruction."""

from .config import (
    BLANK_IDX,
    CHAR_TO_IDX,
    DEFAULT_OCR_CONF_THRESHOLD,
    DEFAULT_WEIGHT_CANDIDATES,
    HUD_ROIS,
    HUD_ROI_FIELDS,
    IDX_TO_CHAR,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    REROLL_MAX_RGB_DISTANCE,
    REROLL_MIN_GREEN_DOMINANCE,
    REROLL_TARGET_RGB,
    VOCAB,
)
from .model import HudOcrCRNN, ctc_greedy_decode
from .pipeline import (
    HudOcrPipeline,
    HudOcrResult,
    is_reroll_button_color,
    parse_hud_value,
    resolve_weights_path,
)

__all__ = [
    "BLANK_IDX",
    "CHAR_TO_IDX",
    "DEFAULT_OCR_CONF_THRESHOLD",
    "DEFAULT_WEIGHT_CANDIDATES",
    "HUD_ROIS",
    "HUD_ROI_FIELDS",
    "HudOcrCRNN",
    "HudOcrPipeline",
    "HudOcrResult",
    "IDX_TO_CHAR",
    "INPUT_HEIGHT",
    "INPUT_WIDTH",
    "REROLL_MAX_RGB_DISTANCE",
    "REROLL_MIN_GREEN_DOMINANCE",
    "REROLL_TARGET_RGB",
    "VOCAB",
    "ctc_greedy_decode",
    "is_reroll_button_color",
    "parse_hud_value",
    "resolve_weights_path",
]
