"""
Master Two-Stage Training Script for Balatro View Deck Classifier
=================================================================
Orchestrates end-to-end training:
- Stage 1: YOLO Left-Edge Card Localization Detector on View Deck screens
           (`classifiers/deck_classifier/weights/detector_best.pt`)
- Stage 2: Shared Multi-Head Card Attribute Classifier retrained on Hand + Shop Crops + View Deck
           (`classifiers/card_classifier/weights/classifier_best.pt`)

Usage:
  python classifiers/deck_classifier/train.py --sync --stage all
  python classifiers/deck_classifier/train.py --sync --stage detector
  python classifiers/deck_classifier/train.py --sync --stage classifier
"""

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
DECK_DATASET_DIR = CURRENT_DIR / "dataset"
DECK_WEIGHTS_DIR = CURRENT_DIR / "weights"
CARD_CLASSIFIER_DIR = PROJECT_ROOT / "classifiers" / "card_classifier"
CARD_DECK_FRAMES_DIR = CARD_CLASSIFIER_DIR / "dataset" / "deck_frames"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from classifiers.card_classifier.stage2_classifier.train_classifier import run_train_classifier
from classifiers.card_classifier.sync_dataset import sync_deck_frames_only
from classifiers.deck_classifier.stage1_detector.train_detector import run_train_detector
from classifiers.deck_classifier.sync_dataset import sync_dataset


def _has_files(directory: Path, pattern: str) -> bool:
    return directory.exists() and any(directory.glob(pattern))


def ensure_datasets_ready(stage: str):
    """Auto-syncs missing View Deck files without ever deleting existing hand/shop datasets."""
    if stage in ("all", "detector"):
        has_imgs = _has_files(DECK_DATASET_DIR / "images", "*.png") or _has_files(DECK_DATASET_DIR / "images", "*.jpg")
        has_lbls = _has_files(DECK_DATASET_DIR / "labels_deck", "*.txt")
        if not has_imgs or not has_lbls:
            print("[Deck Dataset] Local deck_classifier dataset incomplete, syncing from mod...")
            sync_dataset(clean=False, also_sync_card=False)
        else:
            print(f"[Deck Dataset OK] Found existing View Deck detector files in {DECK_DATASET_DIR}")

    if stage in ("all", "classifier"):
        has_card_deck_imgs = _has_files(CARD_DECK_FRAMES_DIR / "images", "*.png") or _has_files(
            CARD_DECK_FRAMES_DIR / "images", "*.jpg"
        )
        has_card_deck_json = _has_files(CARD_DECK_FRAMES_DIR / "labels_json", "*.json")
        if not has_card_deck_imgs or not has_card_deck_json:
            print("[Card Classifier Deck Frames] deck_frames incomplete, syncing dataset_deck...")
            sync_deck_frames_only(clean=False)
        else:
            print(f"[Card Classifier Deck Frames OK] Found existing View Deck frames in {CARD_DECK_FRAMES_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Master Training Script for View Deck Classifier")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "detector", "classifier"],
        help="Which stage to train (detector, classifier, or all)",
    )
    parser.add_argument("--epochs-detector", type=int, default=30, help="Epochs for Stage 1 deck detector (default: 30)")
    parser.add_argument(
        "--epochs-classifier",
        type=int,
        default=25,
        help="Epochs for Stage 2 card classifier (default: 25)",
    )
    parser.add_argument("--batch-detector", type=int, default=32, help="Batch size for Stage 1 detector (default: 32)")
    parser.add_argument("--batch-classifier", type=int, default=512, help="Batch size for Stage 2 classifier (default: 512)")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO training image size (default: 960)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base YOLO weights (default: yolov8n.pt)")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader CPU workers for Stage 2 (default: 8)")
    parser.add_argument(
        "--backbone",
        type=str,
        default="mobilenet_v3_large",
        choices=["mobilenet_v3_small", "mobilenet_v3_large", "resnet18"],
        help="Stage 2 classifier backbone",
    )
    parser.add_argument(
        "--max-deck-crops-per-frame",
        type=int,
        default=32,
        help="Max crops sampled per View Deck frame for Stage 2 training (0 = all)",
    )
    parser.add_argument("--exist-ok", action="store_true", help="Overwrite existing runs")
    parser.add_argument("--sync", action="store_true", help="Sync latest View Deck captures from Balatro mod first")
    parser.add_argument(
        "--clean-sync",
        action="store_true",
        help="Clean ONLY deck_classifier/dataset and card_classifier/dataset/deck_frames before syncing (preserves hand & shop data)",
    )
    args = parser.parse_args()

    if args.sync or args.clean_sync:
        clean_flag = bool(args.clean_sync)
        if args.stage in ("all", "detector"):
            sync_dataset(clean=clean_flag, also_sync_card=(args.stage == "all"))
        elif args.stage == "classifier":
            sync_deck_frames_only(clean=clean_flag)

    ensure_datasets_ready(args.stage)

    if args.stage in ("all", "detector"):
        print("\n" + "=" * 60)
        print("STARTING STAGE 1: VIEW DECK LEFT-EDGE DETECTOR TRAINING")
        print("=" * 60)
        run_train_detector(
            model_name=args.model,
            epochs=args.epochs_detector,
            imgsz=args.imgsz,
            batch=args.batch_detector,
            exist_ok=args.exist_ok,
        )

    if args.stage in ("all", "classifier"):
        print("\n" + "=" * 60)
        print("STARTING STAGE 2: CARD CLASSIFIER RETRAINING (HAND + SHOP + DECK)")
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
    print("ALL REQUESTED VIEW DECK TRAINING STAGES COMPLETE!")
    print(f"Deck Detector Weights : {DECK_WEIGHTS_DIR / 'detector_best.pt'}")
    print(f"Card Classifier Weights: {CARD_CLASSIFIER_DIR / 'weights' / 'classifier_best.pt'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
