"""
Train Stage 1 YOLO Shop Item Localization Model
================================================
Trains a 4-class (Joker, Tarot, Spectral, Planet) YOLOv8 model to detect cards/items
across the 4 shop regions (joker bar, consumables area, main shop, booster packs).
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
DATA_YAML = CURRENT_DIR / "data.yaml"
RUNS_DIR = CURRENT_DIR / "runs"
WEIGHTS_DIR = CLASSIFIER_ROOT / "weights"

try:
    from .prepare_detector_data import prepare_detector_data
except ImportError:
    from prepare_detector_data import prepare_detector_data


def run_train_detector(
    model_name="yolov8n.pt",
    epochs=25,
    imgsz=640,
    batch=16,
    name="shop_detector",
    exist_ok=False
):
    print("Preparing Stage 1 dataset split from dataset/labels_detect...")
    prepare_detector_data()

    print("\n" + "=" * 60)
    print("Stage 1: YOLO Shop Item Localization Training")
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

    print(f"Base weights : {model_name}")
    print(f"Target epochs: {epochs}")
    print(f"Batch size   : {batch}")
    print(f"Data config  : {DATA_YAML}")
    print(f"Output dir   : {RUNS_DIR}")
    model = YOLO(model_name)

    try:
        from classifiers.augmentations import attach_batch_augmentations
        attach_batch_augmentations(model, compression_prob=0.5, cursor_prob=0.3)
    except Exception as e:
        print(f"[Notice] Could not attach YOLO batch augmentations: {e}")

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
    )

    actual_run_dir = Path(results.save_dir) if hasattr(results, "save_dir") else (RUNS_DIR / name)
    best_pt = actual_run_dir / "weights" / "best.pt"

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    target_best = WEIGHTS_DIR / "detector_best.pt"
    if best_pt.exists():
        shutil.copy2(best_pt, target_best)
        print(f"\n[Success] Stage 1 detector best weights copied to: {target_best}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stage 1 YOLO Shop Detector")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--exist-ok", action="store_true", help="Overwrite existing run")
    args = parser.parse_args()

    run_train_detector(
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        exist_ok=args.exist_ok
    )
