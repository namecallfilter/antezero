"""
Train Stage 1 YOLO View Deck Card Localization Model
=====================================================
Trains a single-class (`card`) YOLOv8 model to detect the visible left-edge slice
of every card in Balatro's View Deck screen (15-100 cards across suit rows).
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

CURRENT_DIR = Path(__file__).resolve().parent
CLASSIFIER_ROOT = CURRENT_DIR.parent
PROJECT_ROOT = CLASSIFIER_ROOT.parent.parent
DATA_YAML = CURRENT_DIR / "data.yaml"
RUNS_DIR = CURRENT_DIR / "runs"
WEIGHTS_DIR = CLASSIFIER_ROOT / "weights"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from classifiers.augmentations import attach_batch_augmentations
except ImportError:
    attach_batch_augmentations = None

try:
    from .prepare_detector_data import prepare_detector_data
except ImportError:
    from prepare_detector_data import prepare_detector_data


def resolve_base_weights(model_name: str) -> str:
    """Resolves local YOLO base weights before falling back to Ultralytics download."""
    p = Path(model_name)
    if p.exists():
        return str(p)
    for candidate in (
        WEIGHTS_DIR / model_name,
        CLASSIFIER_ROOT / model_name,
        PROJECT_ROOT / "classifiers" / "shop_classifier" / model_name,
        PROJECT_ROOT / "classifiers" / "card_classifier" / model_name,
    ):
        if candidate.exists():
            return str(candidate)
    return model_name


def run_train_detector(
    model_name: str = "yolov8n.pt",
    epochs: int = 30,
    imgsz: int = 960,
    batch: int = 32,
    name: str = "deck_detector",
    exist_ok: bool = False,
    compression_prob: float = 0.5,
    cursor_prob: float = 0.3,
):
    print("Preparing Stage 1 View Deck dataset split from dataset/labels_deck...")
    prepare_detector_data()

    print("\n" + "=" * 60)
    print("Stage 1: YOLO View Deck Card Localization Training")
    print("=" * 60)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[CUDA OK] GPU Device : {gpu_name} ({vram_gb:.2f} GB VRAM)")
        device = 0
        if vram_gb >= 20.0 and batch <= 16:
            batch = 64
            print(f"[AUTO] High-VRAM GPU detected ({vram_gb:.1f} GB); scaling detector batch size to {batch}.")
    else:
        print("[WARNING] CUDA not available, training on CPU.")
        device = "cpu"

    resolved_weights = resolve_base_weights(model_name)
    print(f"Base weights : {resolved_weights}")
    print(f"Target epochs: {epochs}")
    print(f"Image size   : {imgsz}")
    print(f"Batch size   : {batch}")
    print(f"Data config  : {DATA_YAML}")
    print(f"Output dir   : {RUNS_DIR}")
    print("=" * 60 + "\n")

    model = YOLO(resolved_weights)

    if attach_batch_augmentations and (compression_prob > 0 or cursor_prob > 0):
        attach_batch_augmentations(model, compression_prob=compression_prob, cursor_prob=cursor_prob)

    # Disable horizontal/vertical flips because left-edge card slices become invalid if mirrored.
    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=RUNS_DIR.as_posix(),
        name=name,
        workers=min(8, os.cpu_count() or 4),
        cache="ram",
        amp=True,
        val=True,
        save=True,
        exist_ok=exist_ok,
        verbose=True,
        fliplr=0.0,
        flipud=0.0,
        mosaic=0.25,
        close_mosaic=10,
        degrees=0.0,
        scale=0.25,
        max_det=150,
    )

    actual_run_dir = Path(results.save_dir) if hasattr(results, "save_dir") else (RUNS_DIR / name)
    best_pt = actual_run_dir / "weights" / "best.pt"

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    target_best = WEIGHTS_DIR / "detector_best.pt"
    if best_pt.exists():
        shutil.copy2(best_pt, target_best)
        print(f"\n[Success] Stage 1 deck detector best weights copied to: {target_best}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stage 1 YOLO View Deck Detector")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs (default: 30)")
    parser.add_argument("--batch", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--imgsz", type=int, default=960, help="Image size (default: 960)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model weights")
    parser.add_argument("--name", type=str, default="deck_detector", help="Run name")
    parser.add_argument("--exist-ok", action="store_true", help="Overwrite existing run")
    args = parser.parse_args()

    run_train_detector(
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        exist_ok=args.exist_ok,
    )
