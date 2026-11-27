"""
Wait for Balatro Dataset Generation and Auto-Launch Retraining
==============================================================
Monitors dataset generation until target image count is reached,
then automatically launches the end-to-end retraining pipeline:
    python classifiers/{shop_classifier|card_classifier}/train.py --sync --stage all

Supported Models:
    --model shop   (default: monitors dataset_shop/images, trains shop_classifier)
    --model card   (monitors dataset/images, trains card_classifier)
    --model deck   (monitors dataset_deck/images, trains deck_classifier + card_classifier Stage 2)
    --model both   (monitors dataset_shop/images, trains shop_classifier AND card_classifier)
    --model auto   (detects active generation directory automatically)

Usage:
    python tools/auto_retrain.py
    python tools/auto_retrain.py --model shop --target 10000 --interval 5
    python tools/auto_retrain.py --model card --target 10000
    python tools/auto_retrain.py --model deck --target 2500
    python tools/auto_retrain.py --model both --clean-shop --target 10000
"""

import os
import sys
import time
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent.parent

APPDATA_MOD_DIR = Path(os.path.expandvars(r"%APPDATA%\Balatro\Mods\gamestate_dataset_cv"))
REPO_MOD_DIR = ROOT_DIR / "balatro_mod"


def format_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def _latest_image_mtime(dirs: list[Path]) -> float:
    latest = 0.0
    for d in dirs:
        if d.exists():
            for p in d.iterdir():
                if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    mt = p.stat().st_mtime
                    if mt > latest:
                        latest = mt
    return latest


def resolve_images_directory(model_type: str = "shop") -> tuple[Path, str, Path]:
    """
    Finds the active dataset images directory, model name, and train script.
    Checks both AppData roaming directory and repo directory.
    """
    if model_type == "auto":
        # Check which directory has newly growing files
        shop_dirs = [APPDATA_MOD_DIR / "dataset_shop" / "images", REPO_MOD_DIR / "dataset_shop" / "images"]
        card_dirs = [APPDATA_MOD_DIR / "dataset" / "images", REPO_MOD_DIR / "dataset" / "images"]
        deck_dirs = [APPDATA_MOD_DIR / "dataset_deck" / "images", REPO_MOD_DIR / "dataset_deck" / "images"]

        candidates = [
            ("shop", _latest_image_mtime(shop_dirs)),
            ("card", _latest_image_mtime(card_dirs)),
            ("deck", _latest_image_mtime(deck_dirs)),
        ]
        model_type = max(candidates, key=lambda item: item[1])[0]

    if model_type in ("shop", "both"):
        folder_name = "dataset_shop"
        train_script = ROOT_DIR / "classifiers" / "shop_classifier" / "train.py"
    elif model_type == "deck":
        folder_name = "dataset_deck"
        train_script = ROOT_DIR / "classifiers" / "deck_classifier" / "train.py"
    else:
        folder_name = "dataset"
        train_script = ROOT_DIR / "classifiers" / "card_classifier" / "train.py"

    appdata_candidate = APPDATA_MOD_DIR / folder_name / "images"
    repo_candidate = REPO_MOD_DIR / folder_name / "images"

    # Prefer AppData if files exist there, otherwise check repo
    if appdata_candidate.exists() and len(list(appdata_candidate.glob("*.png")) + list(appdata_candidate.glob("*.jpg"))) > 0:
        active_dir = appdata_candidate
    elif repo_candidate.exists() and len(list(repo_candidate.glob("*.png")) + list(repo_candidate.glob("*.jpg"))) > 0:
        active_dir = repo_candidate
    else:
        # Default to AppData (where Love2D writes by default)
        active_dir = appdata_candidate

    return active_dir, model_type, train_script


import re

FRAME_REGEX = re.compile(r"frame_(\d+)", re.IGNORECASE)


def get_dataset_counts(directory: Path) -> tuple[int, int]:
    """
    Returns (unique_frames_count, total_images_count).
    Extracts frame numbers using regex 'frame_(\\d+)' so that multiple crop files
    per frame (e.g. joker, consumables, shop/pack) map to 1 unique frame.
    """
    if not directory.exists():
        return 0, 0
    frame_ids = set()
    total_images = 0
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg"):
            total_images += 1
            m = FRAME_REGEX.search(p.name)
            if m:
                frame_ids.add(int(m.group(1)))
            else:
                frame_ids.add(p.name)
    return len(frame_ids), total_images


def wait_for_images(images_dir: Path, model_type: str, target_count: int = 10000, check_interval: int = 5):
    print("\n" + "=" * 70)
    print("BALATRO DATASET MONITOR & AUTO-RETRAIN ORCHESTRATOR")
    print("=" * 70)
    print(f"Target Model         : {model_type.upper()} CLASSIFIER")
    print(f"Monitoring Directory : {images_dir}")
    print(f"Target Frame Count   : {target_count:,} frames")
    print(f"Polling Interval     : Every {check_interval} seconds")
    print(f"Start Time           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")

    images_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    last_frames = -1
    consecutive_done_checks = 0

    while True:
        # Count unique frames and total images in monitored directory
        frames, total_imgs = get_dataset_counts(images_dir)

        # If AppData is empty but repo has files (or vice versa), check both
        alt_dir = REPO_MOD_DIR / images_dir.parent.name / "images" if "AppData" in str(images_dir) else APPDATA_MOD_DIR / images_dir.parent.name / "images"
        alt_frames, alt_imgs = get_dataset_counts(alt_dir)

        total_frames = max(frames, alt_frames)
        total_crops = max(total_imgs, alt_imgs)

        elapsed = time.time() - start_time

        if total_frames != last_frames:
            pct = (total_frames / target_count) * 100.0 if target_count > 0 else 0.0
            rate = total_frames / elapsed if elapsed > 0 else 0.0
            remaining_secs = (target_count - total_frames) / rate if rate > 0 and total_frames < target_count else 0
            eta_str = format_elapsed(remaining_secs) if total_frames < target_count else "Complete"

            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"Frames: {total_frames:5,d} / {target_count:,d} ({pct:5.1f}%) | "
                f"Crops: {total_crops:5,d} | "
                f"Speed: {rate:4.1f} frames/s | "
                f"Elapsed: {format_elapsed(elapsed)} | "
                f"ETA: {eta_str}"
            )
            last_frames = total_frames

        if total_frames >= target_count:
            consecutive_done_checks += 1
            if consecutive_done_checks >= 2:
                print(f"\n[TARGET REACHED] Generated {total_frames:,} frames ({total_crops:,} crop images) in {format_elapsed(elapsed)}!")
                break
        else:
            consecutive_done_checks = 0

        time.sleep(check_interval)


def clean_shop_datasets():
    """
    Deletes old shop dataset and card-crop output directories (leaving in-hand
    card classifier data untouched) so shop frames can be regenerated cleanly
    with updated ROIs.
    """
    targets = [
        APPDATA_MOD_DIR / "dataset_shop",
        REPO_MOD_DIR / "dataset_shop",
        APPDATA_MOD_DIR / "dataset_card_crops",
        REPO_MOD_DIR / "dataset_card_crops",
        ROOT_DIR / "classifiers" / "shop_classifier" / "dataset",
        ROOT_DIR / "classifiers" / "card_classifier" / "dataset" / "shop_crops",
    ]
    print("\n" + "=" * 70)
    print("CLEANING OLD SHOP & CROP DATASETS (PRESERVING HAND CARD DATA)")
    print("=" * 70)
    for target in targets:
        if target.exists():
            print(f"Removing: {target}")
            shutil.rmtree(target)
        else:
            print(f"Already clean: {target}")
    # Recreate expected output directories so an already-running Balatro instance can write immediately
    for sub in [
        APPDATA_MOD_DIR / "dataset_shop" / "images",
        APPDATA_MOD_DIR / "dataset_shop" / "labels",
        APPDATA_MOD_DIR / "dataset_card_crops" / "images",
        APPDATA_MOD_DIR / "dataset_card_crops" / "labels",
    ]:
        sub.mkdir(parents=True, exist_ok=True)
    print("=" * 70 + "\n")


def clean_deck_datasets():
    """
    Deletes old View Deck dataset output directories (preserving hand card and shop
    datasets) so deck frames can be regenerated cleanly with updated ROIs.
    """
    targets = [
        APPDATA_MOD_DIR / "dataset_deck",
        REPO_MOD_DIR / "dataset_deck",
        ROOT_DIR / "classifiers" / "deck_classifier" / "dataset",
        ROOT_DIR / "classifiers" / "card_classifier" / "dataset" / "deck_frames",
    ]
    print("\n" + "=" * 70)
    print("CLEANING OLD VIEW DECK DATASETS (PRESERVING HAND CARD & SHOP DATA)")
    print("=" * 70)
    for target in targets:
        if target.exists():
            print(f"Removing: {target}")
            shutil.rmtree(target)
        else:
            print(f"Already clean: {target}")
    for sub in [
        APPDATA_MOD_DIR / "dataset_deck" / "images",
        APPDATA_MOD_DIR / "dataset_deck" / "labels",
    ]:
        sub.mkdir(parents=True, exist_ok=True)
    print("=" * 70 + "\n")


def launch_training(
    train_script: Path,
    model_type: str,
    stage: str = "all",
    epochs_det: int | None = None,
    epochs_cls: int | None = None,
    backbone: str | None = None,
    clean_sync: bool = False,
):
    print("\n" + "=" * 70)
    print(f"LAUNCHING {model_type.upper()} CLASSIFIER RETRAINING PIPELINE (stage={stage})")
    print("=" * 70)
    sync_flag = "--clean-sync" if clean_sync else "--sync"
    cmd = [
        sys.executable,
        str(train_script),
        sync_flag,
        "--stage", stage,
        "--batch-classifier", "512",
        "--workers", "8",
    ]
    if epochs_det is not None:
        cmd.extend(["--epochs-detector", str(epochs_det)])
    if epochs_cls is not None:
        cmd.extend(["--epochs-classifier", str(epochs_cls)])
    if backbone:
        cmd.extend(["--backbone", backbone])
    print(f"Executing: {' '.join(cmd)}\n" + "=" * 70 + "\n")

    # Run and stream output directly to terminal
    proc = subprocess.run(cmd, cwd=str(train_script.parent))
    if proc.returncode == 0:
        print("\n" + "=" * 70)
        print(f"ALL {model_type.upper()} TRAINING STAGES COMPLETED SUCCESSFULLY!")
        print(f"Weights saved in: {train_script.parent / 'weights'}")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70 + "\n")
    else:
        print(f"\n[ERROR] Training process exited with return code: {proc.returncode}")
        sys.exit(proc.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wait for dataset generation and trigger retraining")
    parser.add_argument("--model", type=str, default="shop", choices=["shop", "card", "deck", "both", "auto"], help="Model(s) to train (default: shop)")
    parser.add_argument("--target", type=int, default=10000, help="Target image count (default: 10000)")
    parser.add_argument("--interval", type=int, default=5, help="Poll interval in seconds (default: 5)")
    parser.add_argument("--epochs-detector", type=int, default=None, help="Stage 1 detector epochs (defaults to per-model train.py default)")
    parser.add_argument("--epochs-classifier", type=int, default=None, help="Stage 2 classifier epochs (defaults to per-model train.py default)")
    parser.add_argument("--backbone", type=str, default=None, choices=["mobilenet_v3_small", "mobilenet_v3_large", "resnet18"], help="Stage 2 classifier backbone override")
    parser.add_argument("--clean-shop", action="store_true", help="Delete existing shop & shop_crops datasets before monitoring/syncing (preserves hand card dataset)")
    parser.add_argument("--clean-deck", action="store_true", help="Delete existing View Deck datasets before monitoring/syncing (preserves hand card & shop datasets)")
    parser.add_argument("--shop-stage", type=str, default="all", choices=["all", "detector", "classifier"], help="Training stage for shop_classifier (default: all)")
    parser.add_argument("--card-stage", type=str, default=None, choices=["all", "detector", "classifier"], help="Training stage for card_classifier (default: classifier when --model both, else all)")
    parser.add_argument("--deck-stage", type=str, default="all", choices=["all", "detector", "classifier"], help="Training stage for deck_classifier (default: all)")
    args = parser.parse_args()

    if args.clean_shop:
        clean_shop_datasets()
    if args.clean_deck:
        clean_deck_datasets()

    images_dir, model_type, train_script = resolve_images_directory(args.model)
    wait_for_images(images_dir, model_type, target_count=args.target, check_interval=args.interval)

    if model_type == "both":
        shop_script = ROOT_DIR / "classifiers" / "shop_classifier" / "train.py"
        card_script = ROOT_DIR / "classifiers" / "card_classifier" / "train.py"
        card_stage = args.card_stage or "classifier"
        launch_training(
            shop_script,
            "shop",
            stage=args.shop_stage,
            epochs_det=args.epochs_detector,
            epochs_cls=args.epochs_classifier,
            backbone=args.backbone,
            clean_sync=args.clean_shop,
        )
        # Never --clean-sync card_classifier here so existing hand card data is preserved
        launch_training(
            card_script,
            "card",
            stage=card_stage,
            epochs_det=args.epochs_detector,
            epochs_cls=args.epochs_classifier,
            backbone=args.backbone,
            clean_sync=False,
        )
    else:
        if model_type == "shop":
            stage = args.shop_stage
            clean_sync = args.clean_shop
        elif model_type == "deck":
            stage = args.deck_stage
            clean_sync = args.clean_deck
        else:
            stage = args.card_stage or "all"
            clean_sync = False
        launch_training(
            train_script,
            model_type,
            stage=stage,
            epochs_det=args.epochs_detector,
            epochs_cls=args.epochs_classifier,
            backbone=args.backbone,
            clean_sync=clean_sync,
        )
