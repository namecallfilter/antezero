"""Configuration and constants for Balatro HUD OCR."""

from __future__ import annotations

from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
CLASSIFIERS_DIR = CURRENT_DIR.parent
PROJECT_ROOT = CLASSIFIERS_DIR.parent

DEFAULT_WEIGHT_CANDIDATES = (
    CLASSIFIERS_DIR / "prod_weights" / "hud_ocr.pt",
    CURRENT_DIR / "weights" / "hud_ocr_best.pt",
)

DEFAULT_FONTS_DIR = CURRENT_DIR / "fonts"
DEFAULT_FONT_PATH = PROJECT_ROOT / "balatro_source" / "resources" / "fonts" / "m6x11plus.ttf"
DEFAULT_MOD_DATASET_DIR = (
    Path.home() / "AppData" / "Roaming" / "Balatro" / "Mods" / "gamestate_dataset_cv" / "dataset_hud_ocr"
)

# CTC vocabulary: index 0 is reserved for CTC <blank>.
VOCAB: tuple[str, ...] = (
    "<blank>",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "$",
    ",",
    ".",
    "e",
    "+",
    "-",
    "/",
)

BLANK_IDX: int = 0
CHAR_TO_IDX: dict[str, int] = {ch: idx for idx, ch in enumerate(VOCAB) if idx != BLANK_IDX}
IDX_TO_CHAR: dict[int, str] = {idx: ch for idx, ch in enumerate(VOCAB)}
NUM_CLASSES: int = len(VOCAB)

# Fixed crop input geometry (HxW) for batched GPU inference.
# Downsampling in HudOcrCRNN maps 40x192 -> 1x48 (48 CTC timesteps).
INPUT_HEIGHT: int = 40
INPUT_WIDTH: int = 192

NORM_MEAN: tuple[float, float, float] = (0.5, 0.5, 0.5)
NORM_STD: tuple[float, float, float] = (0.5, 0.5, 0.5)

DEFAULT_OCR_CONF_THRESHOLD: float = 0.90
REROLL_TARGET_RGB: tuple[int, int, int] = (78, 177, 127)
REROLL_MAX_RGB_DISTANCE: float = 65.0
REROLL_MIN_GREEN_DOMINANCE: float = 15.0

HUD_ROI_FIELDS: tuple[str, ...] = (
    "hud_round_score",
    "hud_money",
    "hud_hands",
    "hud_discards",
    "hud_ante",
    "hud_round",
    "hud_reroll_cost",
    "hud_blind_target",
)

# Per-field valid characters (excluding CTC <blank>) so decoding stays strictly numeric.
FIELD_ALLOWED_CHARS: dict[str, str] = {
    "hud_round_score": "0123456789,.e+",
    "hud_money": "0123456789-",
    "hud_hands": "0123456789",
    "hud_discards": "0123456789",
    "hud_ante": "0123456789",
    "hud_round": "0123456789",
    "hud_reroll_cost": "0123456789",
    "hud_blind_target": "0123456789,.e+",
}

# Base 1920x1080 ROI coordinates (updated once calibrated via balatro_mod HUD OCR mode).
HUD_ROIS: dict[str, dict[str, float]] = {
    "hud_round_score": {"x": 220.599731, "y": 389.166317, "w": 310.609893, "h": 91.386205},
    "hud_money": {"x": 235.579213, "y": 810.532096, "w": 258.355712, "h": 109.781902},
    "hud_hands": {"x": 245.551355, "y": 721.540544, "w": 103.609385, "h": 84.604535},
    "hud_discards": {"x": 376.543146, "y": 722.718153, "w": 104.654328, "h": 79.453563},
    "hud_ante": {"x": 237.682514, "y": 955.549886, "w": 118.302089, "h": 75.922924},
    "hud_round": {"x": 376.753403, "y": 955.475887, "w": 105.791305, "h": 74.350088},
    "hud_reroll_cost": {"x": 629.031965, "y": 575.298322, "w": 122.196835, "h": 65.468198},
    "hud_blind_target": {"x": 245.537867, "y": 233.218735, "w": 253.485902, "h": 88.705984},
}
