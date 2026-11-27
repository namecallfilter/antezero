"""Configuration and constants for dedicated Balatro end-of-run Seed OCR."""

from __future__ import annotations

from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
CLASSIFIERS_DIR = CURRENT_DIR.parent
PROJECT_ROOT = CLASSIFIERS_DIR.parent

DEFAULT_WEIGHT_CANDIDATES: tuple[Path, ...] = (
    CLASSIFIERS_DIR / "prod_weights" / "seed_ocr.pt",
    CURRENT_DIR / "weights" / "seed_ocr_best.pt",
)

DEFAULT_MOD_DATASET_DIR: Path = (
    Path.home()
    / "AppData"
    / "Roaming"
    / "Balatro"
    / "Mods"
    / "gamestate_dataset_cv"
    / "dataset_seed_ocr"
)

# Exact 34-character Balatro random seed alphabet (no '0' and no 'O').
SEED_ALPHABET: str = "123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"
SEED_VOCAB: tuple[str, ...] = ("<blank>", *SEED_ALPHABET)
SEED_BLANK_IDX: int = 0
SEED_CHAR_TO_IDX: dict[str, int] = {
    ch: idx for idx, ch in enumerate(SEED_VOCAB) if idx != SEED_BLANK_IDX
}
SEED_IDX_TO_CHAR: dict[int, str] = {idx: ch for idx, ch in enumerate(SEED_VOCAB)}
SEED_NUM_CLASSES: int = len(SEED_VOCAB)
SEED_LENGTH: int = 8

INPUT_HEIGHT: int = 40
INPUT_WIDTH: int = 192
NORM_MEAN: tuple[float, float, float] = (0.5, 0.5, 0.5)
NORM_STD: tuple[float, float, float] = (0.5, 0.5, 0.5)

DEFAULT_SEED_CONF_THRESHOLD: float = 0.85

# Calibrated 1920x1080 Seed pill ROIs (Win & Game Over modals) and Copy-button relative anchor.
SEED_ROIS: dict[str, dict[str, float]] = {
    "seed_win": {"x": 1066.0, "y": 700.0, "w": 173.0, "h": 58.0},
    "seed_game_over": {"x": 1066.0, "y": 660.0, "w": 173.0, "h": 58.0},
}
COPY_BUTTON_TO_SEED_OFFSET_1080P: dict[str, float] = {
    "dx": 90.0,
    "dy": -64.0,
    "w": 173.0,
    "h": 58.0,
}
