"""
Master Two-Stage Training Script for Balatro Shop Classifier
============================================================
Orchestrates end-to-end training:
- Stage 1: YOLO Shop Item Localization Detector (Joker, Tarot, Spectral, Planet)
- Stage 2: Multi-Head Item Attribute Classifier (Key, Edition, Eternal, Perishable, Rental, Debuffed)

Usage:
  python train.py --stage all        # Trains Stage 1 and Stage 2 end-to-end
  python train.py --stage detector   # Trains Stage 1 item detector only
  python train.py --stage classifier # Trains Stage 2 multi-head attribute model only
  python train.py --sync             # Syncs latest mod dataset before training
"""

import sys
import argparse
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
DATASET_DIR = CURRENT_DIR / "dataset"
WEIGHTS_DIR = CURRENT_DIR / "weights"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# Ensure submodules are in python path
sys.path.append(str(CURRENT_DIR / "stage1_detector"))
sys.path.append(str(CURRENT_DIR / "stage2_classifier"))

from stage1_detector.train_detector import run_train_detector
from stage2_classifier.train_classifier import run_train_classifier
from sync_dataset import sync_dataset


def ensure_dataset_ready():
    """Checks that Stage 1 and Stage 2 labels exist."""
    labels_detect = DATASET_DIR / "labels_detect"
    labels_json = DATASET_DIR / "labels_json"
    images_dir = DATASET_DIR / "images"

    has_images = images_dir.exists() and len(list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))) > 0
    has_detect = labels_detect.exists() and len(list(labels_detect.glob("*.txt"))) > 0
    has_json = labels_json.exists() and len(list(labels_json.glob("*.json"))) > 0

    if not has_images or not has_detect or not has_json:
        print("[Dataset] Local dataset folder incomplete, running sync_dataset()...")
        sync_dataset(clean=False)
    else:
        print(f"[Dataset OK] Found existing images and labels in {DATASET_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Master Training Script for Two-Stage Shop Classifier")
    parser.add_argument("--stage", type=str, default="all", choices=["all", "detector", "classifier"], help="Which stage to train")
    parser.add_argument("--epochs-detector", type=int, default=25, help="Epochs for Stage 1 detector (default: 25)")
    parser.add_argument("--epochs-classifier", type=int, default=20, help="Epochs for Stage 2 classifier (default: 20)")
    parser.add_argument("--batch-detector", type=int, default=64, help="Batch size for Stage 1 (default: 64)")
    parser.add_argument("--batch-classifier", type=int, default=512, help="Batch size for Stage 2 (default: 512)")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader CPU workers for Stage 2 (default: 8)")
    parser.add_argument("--backbone", type=str, default="mobilenet_v3_large", choices=["mobilenet_v3_small", "mobilenet_v3_large", "resnet18"])
    parser.add_argument("--exist-ok", action="store_true", help="Overwrite existing runs")
    parser.add_argument("--sync", action="store_true", help="Sync latest generated images and labels from mod first")
    parser.add_argument("--clean-sync", action="store_true", help="Clean dataset directory before syncing from mod")
    args = parser.parse_args()

    if args.sync or args.clean_sync:
        sync_dataset(clean=args.clean_sync)

    ensure_dataset_ready()

    if args.stage in ("all", "detector"):
        print("\n" + "=" * 60)
        print("STARTING STAGE 1: SHOP ITEM DETECTOR TRAINING")
        print("=" * 60)
        run_train_detector(
            epochs=args.epochs_detector,
            batch=args.batch_detector,
            exist_ok=args.exist_ok
        )

    if args.stage in ("all", "classifier"):
        print("\n" + "=" * 60)
        print("STARTING STAGE 2: MULTI-HEAD SHOP ITEM CLASSIFIER TRAINING")
        print("=" * 60)
        run_train_classifier(
            epochs=args.epochs_classifier,
            batch_size=args.batch_classifier,
            backbone=args.backbone,
            exist_ok=args.exist_ok,
            num_workers=args.workers
        )

    print("\n" + "=" * 60)
    print("ALL REQUESTED TRAINING STAGES COMPLETE!")
    print(f"Weights Available in: {WEIGHTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
