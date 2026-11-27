"""
Train Stage 2 Multi-Head Shop Item Classifier
=============================================
Trains multi-task CNN (MobileNetV3-Small / MobileNetV3-Large / ResNet18) to predict:
- Item Key (230 classes: Jokers + Consumables)
- Edition (5 classes: None, Foil, Holo, Polychrome, Negative)
- Stickers / Status (Eternal, Perishable, Rental, Debuffed)

Saves checkpoints and copies best weights to weights/classifier_best.pt.
"""

import sys
import shutil
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
CLASSIFIER_ROOT = CURRENT_DIR.parent
REPO_ROOT = CLASSIFIER_ROOT.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classifiers.augmentations import apply_gpu_batch_augmentations
CLASSIFIER_ROOT = CURRENT_DIR.parent
DATASET_DIR = CLASSIFIER_ROOT / "dataset"
RUNS_DIR = CURRENT_DIR / "runs"
WEIGHTS_DIR = CLASSIFIER_ROOT / "weights"

try:
    from .config import ITEM_KEYS, EDITIONS, CROP_WIDTH, CROP_HEIGHT
    # pyrefly: ignore [missing-import]
    from .dataset import ItemCropDataset
    # pyrefly: ignore [missing-import]
    from .model import MultiHeadItemClassifier
except ImportError:
    from config import ITEM_KEYS, EDITIONS, CROP_WIDTH, CROP_HEIGHT
    from dataset import ItemCropDataset
    # pyrefly: ignore [missing-import]
    from model import MultiHeadItemClassifier


def get_next_run_dir(base_name="shop_classifier", exist_ok=False) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if exist_ok:
        d = RUNS_DIR / base_name
        d.mkdir(parents=True, exist_ok=True)
        return d
    idx = 1
    target = RUNS_DIR / base_name
    while target.exists():
        idx += 1
        target = RUNS_DIR / f"{base_name}-{idx}"
    target.mkdir(parents=True, exist_ok=True)
    return target


def run_train_classifier(
    epochs=20,
    batch_size=256,
    lr=1e-3,
    backbone="mobilenet_v3_large",
    run_name="shop_classifier",
    exist_ok=False,
    num_workers=4
):
    print("\n" + "=" * 60)
    print("Stage 2: Multi-Head Shop Item Classifier Training")
    print("=" * 60)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if vram_gb >= 16.0 and batch_size < 512:
            batch_size = 512
        print(f"[CUDA OK] GPU Device : {gpu_name} ({vram_gb:.2f} GB VRAM, AMP={amp_dtype})")
    else:
        print("[WARNING] CUDA not available, training on CPU.")

    print(f"Dataset root : {DATASET_DIR}")
    print(f"Backbone     : {backbone}")
    print(f"Epochs       : {epochs}")
    print(f"Batch size   : {batch_size}")
    print(f"Item Keys    : {len(ITEM_KEYS)} classes")
    print("=" * 60 + "\n")

    # 1. Datasets & Loaders (8% halo crops on train for bounding-box jitter)
    train_dataset = ItemCropDataset(DATASET_DIR, split="train", val_ratio=0.15, pad_ratio=0.08)
    val_dataset = ItemCropDataset(DATASET_DIR, split="val", val_ratio=0.15, pad_ratio=0.0)

    print(f"Loaded {len(train_dataset):,} train samples, {len(val_dataset):,} validation samples.")
    if len(train_dataset) == 0:
        print("[Error] No training samples found. Please sync the dataset first.")
        return None

    def _make_loader(ds, shuffle, drop_last):
        if getattr(ds, "uses_ram_cache", False):
            return None
        kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": (device.type == "cuda"),
            "drop_last": drop_last,
        }
        if num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 4
        return DataLoader(ds, **kwargs)

    def _iter_batches(ds, loader, shuffle, drop_last):
        if not getattr(ds, "uses_ram_cache", False):
            yield from loader
            return
        n = len(ds)
        perm = torch.randperm(n) if shuffle else torch.arange(n)
        limit = (n // batch_size) * batch_size if (drop_last and n > batch_size) else n
        for start in range(0, limit, batch_size):
            idx = perm[start:start + batch_size]
            imgs = ds._cached_images[idx]
            tgts = {k: v[idx] for k, v in ds._targets.items()}
            yield imgs, tgts

    train_loader = _make_loader(train_dataset, shuffle=True, drop_last=(len(train_dataset) > batch_size))
    val_loader = _make_loader(val_dataset, shuffle=False, drop_last=False)

    # 2. Model & Optimizer
    model = MultiHeadItemClassifier(backbone_name=backbone, pretrained=True).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss()

    run_dir = get_next_run_dir(base_name=run_name, exist_ok=exist_ok)
    best_weights_path = run_dir / "best.pt"
    best_score = 0.0

    # 3. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        train_batches = 0

        for images, targets in _iter_batches(train_dataset, train_loader, shuffle=True, drop_last=(len(train_dataset) > batch_size)):
            images = images.to(device, non_blocking=True)
            if train_dataset.raw_tensor_mode:
                images = apply_gpu_batch_augmentations(
                    images,
                    is_train=True,
                    compression_prob=0.5,
                    cursor_prob=0.3,
                    color_jitter=True,
                    geom_prob=0.45,
                    pad_crop_size=(CROP_HEIGHT, CROP_WIDTH),
                )
            if device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            for k in targets:
                targets[k] = targets[k].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                logits = model(images)

                loss_key = ce_loss(logits["key"], targets["key"])
                loss_edition = ce_loss(logits["edition"], targets["edition"])
                loss_eternal = bce_loss(logits["eternal"], targets["eternal"])
                loss_perishable = bce_loss(logits["perishable"], targets["perishable"])
                loss_rental = bce_loss(logits["rental"], targets["rental"])
                loss_debuffed = bce_loss(logits["debuffed"], targets["debuffed"])

                # Weighted sum: prioritize item identity key, then edition & stickers
                loss = (
                    loss_key * 2.0 +
                    loss_edition * 1.0 +
                    loss_eternal * 0.5 +
                    loss_perishable * 0.5 +
                    loss_rental * 0.5 +
                    loss_debuffed * 0.5
                )

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            train_batches += 1

        scheduler.step()
        avg_train_loss = total_loss / max(1, train_batches)

        # 4. Validation Loop
        model.eval()
        val_loss = 0.0
        val_batches = 0
        correct_key = 0
        correct_edition = 0
        correct_eternal = 0
        correct_perishable = 0
        correct_rental = 0
        correct_debuffed = 0
        total_samples = 0

        with torch.no_grad():
            for images, targets in _iter_batches(val_dataset, val_loader, shuffle=False, drop_last=False):
                images = images.to(device, non_blocking=True)
                if val_dataset.raw_tensor_mode:
                    images = apply_gpu_batch_augmentations(images, is_train=False)
                if device.type == "cuda":
                    images = images.contiguous(memory_format=torch.channels_last)
                for k in targets:
                    targets[k] = targets[k].to(device, non_blocking=True)

                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    logits = model(images)

                    l_k = ce_loss(logits["key"], targets["key"])
                    l_e = ce_loss(logits["edition"], targets["edition"])
                    l_et = bce_loss(logits["eternal"], targets["eternal"])
                    l_p = bce_loss(logits["perishable"], targets["perishable"])
                    l_r = bce_loss(logits["rental"], targets["rental"])
                    l_d = bce_loss(logits["debuffed"], targets["debuffed"])

                    v_loss = l_k * 2.0 + l_e + 0.5 * (l_et + l_p + l_r + l_d)
                val_loss += v_loss.item()
                val_batches += 1

                # Accuracies
                pred_key = torch.argmax(logits["key"], dim=-1)
                pred_ed = torch.argmax(logits["edition"], dim=-1)
                pred_et = (torch.sigmoid(logits["eternal"]) > 0.5).float()
                pred_p = (torch.sigmoid(logits["perishable"]) > 0.5).float()
                pred_r = (torch.sigmoid(logits["rental"]) > 0.5).float()
                pred_d = (torch.sigmoid(logits["debuffed"]) > 0.5).float()

                n = images.size(0)
                total_samples += n
                correct_key += (pred_key == targets["key"]).sum().item()
                correct_edition += (pred_ed == targets["edition"]).sum().item()
                correct_eternal += (pred_et == targets["eternal"]).sum().item()
                correct_perishable += (pred_p == targets["perishable"]).sum().item()
                correct_rental += (pred_r == targets["rental"]).sum().item()
                correct_debuffed += (pred_d == targets["debuffed"]).sum().item()

        avg_val_loss = val_loss / max(1, val_batches)
        acc_key = (correct_key / max(1, total_samples)) * 100.0
        acc_edition = (correct_edition / max(1, total_samples)) * 100.0
        acc_et = (correct_eternal / max(1, total_samples)) * 100.0
        acc_p = (correct_perishable / max(1, total_samples)) * 100.0
        acc_r = (correct_rental / max(1, total_samples)) * 100.0
        acc_d = (correct_debuffed / max(1, total_samples)) * 100.0

        # Composite score prioritizing item key recognition
        composite_score = acc_key * 0.6 + acc_edition * 0.2 + (acc_et + acc_p + acc_r + acc_d) * 0.05

        print(
            f"Epoch {epoch:2d}/{epochs:2d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Key Acc: {acc_key:5.1f}% | "
            f"Ed Acc: {acc_edition:5.1f}% | "
            f"Stickers: {((acc_et + acc_p + acc_r) / 3):5.1f}%"
        )

        if composite_score >= best_score:
            best_score = composite_score
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "backbone": backbone,
                "best_score": best_score,
                "acc_key": acc_key,
                "acc_edition": acc_edition,
            }, best_weights_path)

    # 5. Export Best Weights to Root Weights Directory
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    target_best = WEIGHTS_DIR / "classifier_best.pt"
    if best_weights_path.exists():
        shutil.copy2(best_weights_path, target_best)
        print(f"\n[Success] Stage 2 classifier best weights copied to: {target_best}")

    print("\n" + "=" * 60)
    print("STAGE 2 CLASSIFIER TRAINING COMPLETE!")
    print(f"Best Validation Composite Score: {best_score:.2f}%")
    print("=" * 60)
    return best_weights_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stage 2 Multi-Head Shop Item Classifier")
    parser.add_argument("--epochs", type=int, default=20, help="Epoch count (default: 20)")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size (default: 512)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--backbone", type=str, default="mobilenet_v3_large", choices=["mobilenet_v3_small", "mobilenet_v3_large", "resnet18"])
    parser.add_argument("--exist-ok", action="store_true", help="Overwrite existing runs")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader CPU workers (default: 8)")
    args = parser.parse_args()

    run_train_classifier(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        backbone=args.backbone,
        exist_ok=args.exist_ok,
        num_workers=args.num_workers
    )
