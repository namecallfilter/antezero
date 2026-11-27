r"""
Sync Shop Dataset from Balatro Mod to Shop Classifier
=====================================================
Copies newly captured images and labels from the Balatro Mod:
- `dataset_shop/images/*.png` -> `dataset/images/`
- `dataset_shop/labels/*.txt` -> `dataset/labels_detect/`
- `dataset_shop/labels/*.json` -> `dataset/labels_json/`
"""

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
TARGET_DATASET_DIR = CURRENT_DIR / "dataset"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# pyrefly: ignore [missing-import]
from classifiers.common.dataset_sync import sync_mod_dataset


def sync_dataset(clean: bool = False):
    return sync_mod_dataset(
        source_subfolder="dataset_shop",
        target_dataset_dir=TARGET_DATASET_DIR,
        labels_subdir_name="labels_detect",
        clean=clean
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync newly generated dataset from Balatro mod")
    parser.add_argument("--clean", action="store_true", help="Wipe target dataset folders before copying")
    args = parser.parse_args()
    sync_dataset(clean=args.clean)
