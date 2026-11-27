"""
Sync View Deck Dataset from Balatro Mod to Deck Classifier (and optionally Card Classifier)
==========================================================================================
Copies newly generated images and labels from `balatro_mod/dataset_deck/`
into `classifiers/deck_classifier/dataset/`:
- `dataset_deck/images/*.png` -> `dataset/images/`
- `dataset_deck/labels/*.txt` -> `dataset/labels_deck/`
- `dataset_deck/labels/*.json` -> `dataset/labels_json/`
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


def sync_dataset(clean: bool = False, also_sync_card: bool = False) -> int:
    deck_count = sync_mod_dataset(
        source_subfolder="dataset_deck",
        target_dataset_dir=TARGET_DATASET_DIR,
        labels_subdir_name="labels_deck",
        clean=clean,
    )
    print(f"[Deck Dataset Sync] Synced {deck_count} view-deck frames into {TARGET_DATASET_DIR}.")

    if also_sync_card:
        from classifiers.card_classifier.sync_dataset import sync_deck_frames_only

        card_deck_count = sync_deck_frames_only(clean=clean)
        print(f"[Deck Dataset Sync] Also synced {card_deck_count} view-deck frames into card_classifier/dataset/deck_frames.")

    return deck_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync View Deck Dataset from Balatro Mod")
    parser.add_argument("--clean", action="store_true", help="Clear existing deck dataset images/labels before syncing")
    parser.add_argument(
        "--also-sync-card",
        action="store_true",
        help="Also sync View Deck frames into classifiers/card_classifier/dataset/deck_frames/",
    )
    args = parser.parse_args()
    sync_dataset(clean=args.clean, also_sync_card=args.also_sync_card)
