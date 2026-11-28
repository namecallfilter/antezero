"""
Master Two-Stage Training Script for Balatro Card Classifier
============================================================
Orchestrates training for:
- Stage 1: YOLO Card Localization Detector (single class: "card")
- Stage 2: Multi-Head Card Attribute Classifier (Rank, Suit, Edition, Enhancement, Seal)

Usage:
  python train.py --stage all        # Trains Stage 1 and Stage 2 end-to-end
  python train.py --stage detector   # Trains Stage 1 card detector only
  python train.py --stage classifier # Trains Stage 2 multi-head attribute model only
"""

import sys
import argparse
from pathlib import Path
import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
CURSOR_PATH = PROJECT_ROOT / "assets" / "classifiers" / "cursor.png"
DATASET_DIR = CURRENT_DIR / "dataset"
WEIGHTS_DIR = CURRENT_DIR / "weights"

# Ensure submodules are in python path
sys.path.append(str(CURRENT_DIR / "stage1_detector"))
sys.path.append(str(CURRENT_DIR / "stage2_classifier"))

from stage1_detector.train_detector import run_train_detector
from stage2_classifier.train_classifier import run_train_classifier
from convert_legacy_dataset import convert_dataset
from sync_dataset import sync_dataset


def resolve_model_weights(model_name: str) -> str:
    """Resolves model weights path, checking classifiers/card_classifier/weights first."""
    p = Path(model_name)
    if p.exists():
        return str(p)
    cand_weights = WEIGHTS_DIR / model_name
    if cand_weights.exists():
        return str(cand_weights)
    cand_curr = CURRENT_DIR / model_name
    if cand_curr.exists():
        return str(cand_curr)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    return str(WEIGHTS_DIR / model_name)


def ensure_dataset_ready():
    """Checks that Stage 1 and Stage 2 labels exist, converting if necessary."""
    labels_card = DATASET_DIR / "labels_card"
    labels_json = DATASET_DIR / "labels_json"
    if not labels_card.exists() or not labels_json.exists() or len(list(labels_json.glob("*.json"))) == 0:
        print("[Dataset] Converting labels to Stage 1 and Stage 2 formats...")
        convert_dataset()
    else:
        print(f"[Dataset OK] Found existing Stage 1 & Stage 2 formatted labels in {DATASET_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Master Training Script for Two-Stage Card Classifier")
    parser.add_argument("--stage", type=str, default="all", choices=["all", "detector", "classifier"], help="Which stage to train")
    parser.add_argument("--epochs-detector", type=int, default=15, help="Epochs for Stage 1 detector (default: 15)")
    parser.add_argument("--epochs-classifier", type=int, default=25, help="Epochs for Stage 2 classifier (default: 25)")
    parser.add_argument("--batch-detector", type=int, default=64, help="Batch size for Stage 1 (default: 64)")
    parser.add_argument("--batch-classifier", type=int, default=512, help="Batch size for Stage 2 (default: 512)")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader CPU worker processes for Stage 2 (default: 8)")
    parser.add_argument("--backbone", type=str, default="mobilenet_v3_large", choices=["mobilenet_v3_small", "mobilenet_v3_large", "resnet18"], help="Classifier backbone")
    parser.add_argument("--max-deck-crops-per-frame", type=int, default=32, help="Max crops sampled per View Deck frame (0 = all)")
    parser.add_argument("--exist-ok", action="store_true", help="Overwrite existing runs")
    parser.add_argument("--sync", action="store_true", help="Sync latest generated images and labels from balatro_mod/dataset first")
    parser.add_argument("--clean-sync", action="store_true", help="Clean dataset directory before syncing from balatro_mod/dataset")
    args = parser.parse_args()

    if args.sync or args.clean_sync:
        sync_dataset(clean=args.clean_sync)

    ensure_dataset_ready()

    if args.stage in ("all", "detector"):
        print("\n" + "=" * 60)
        print("STARTING STAGE 1: CARD DETECTOR TRAINING")
        print("=" * 60)
        run_train_detector(
            epochs=args.epochs_detector,
            batch=args.batch_detector,
            exist_ok=args.exist_ok
        )

    if args.stage in ("all", "classifier"):
        print("\n" + "=" * 60)
        print("STARTING STAGE 2: MULTI-HEAD CLASSIFIER TRAINING")
        print("=" * 60)
        run_train_classifier(
            epochs=args.epochs_classifier,
            batch_size=args.batch_classifier,
            backbone=args.backbone,
            exist_ok=args.exist_ok,
            num_workers=args.workers,
            max_deck_crops_per_frame=(None if args.max_deck_crops_per_frame <= 0 else args.max_deck_crops_per_frame),
        )

    print("\n" + "=" * 60)
    print("ALL REQUESTED TRAINING STAGES COMPLETE!")
    print(f"Weights Available in: {WEIGHTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
