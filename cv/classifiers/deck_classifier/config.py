"""Configuration and ROI resolution for the View Deck classifier."""

import os
import re
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

BASE_WIDTH = 1920.0
BASE_HEIGHT = 1080.0

# Default 1920x1080 ROI matching balatro_mod/src/config.lua (DEFAULT_CONFIG.rois.view_deck).
VIEW_DECK_ROI = {
    "x": 600.0,
    "y": 200.0,
    "w": 1140.0,
    "h": 660.0,
}

DECK_CLASS_NAMES = {0: "card"}
DEFAULT_IMGSZ = 960

_VIEW_DECK_LINE_RE = re.compile(
    r"view_deck\s*=\s*\{\s*"
    r"x\s*=\s*([-+]?\d+(?:\.\d+)?)\s*,\s*"
    r"y\s*=\s*([-+]?\d+(?:\.\d+)?)\s*,\s*"
    r"w\s*=\s*([-+]?\d+(?:\.\d+)?)\s*,\s*"
    r"h\s*=\s*([-+]?\d+(?:\.\d+)?)\s*,?\s*\}"
)


def resolve_view_deck_roi(boxes_path: Path | None = None) -> dict[str, float]:
    """
    Returns the View Deck ROI from the Balatro mod's `dataset/boxes.lua` if available,
    falling back to `VIEW_DECK_ROI`.
    """
    candidates = []
    if boxes_path is not None:
        candidates.append(Path(boxes_path))
    else:
        candidates.append(
            Path(os.path.expandvars(r"%APPDATA%\Balatro\Mods\gamestate_dataset_cv\dataset\boxes.lua"))
        )
        candidates.append(PROJECT_ROOT / "balatro_mod" / "dataset" / "boxes.lua")

    for path in candidates:
        if path.exists():
            try:
                text = path.read_text(encoding="utf-8")
                match = _VIEW_DECK_LINE_RE.search(text)
                if match:
                    return {
                        "x": float(match.group(1)),
                        "y": float(match.group(2)),
                        "w": float(match.group(3)),
                        "h": float(match.group(4)),
                    }
            except OSError:
                continue

    return dict(VIEW_DECK_ROI)
