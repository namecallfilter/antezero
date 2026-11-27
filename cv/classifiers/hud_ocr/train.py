"""Training script for Balatro HUD OCR (CNN + BiGRU + CTC)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

try:
    from .config import (
        BLANK_IDX,
        CURRENT_DIR,
        DEFAULT_MOD_DATASET_DIR,
        INPUT_HEIGHT,
        INPUT_WIDTH,
        NUM_CLASSES,
    )
    from .dataset import (
        ManifestHudOcrDataset,
        ModHudOcrDataset,
        SyntheticHudOcrDataset,
        collate_ocr_batch,
    )
    from .model import HudOcrCRNN, compute_cer, ctc_greedy_decode
    from .synth import resolve_font_path
except ImportError:
    from config import (  # type: ignore[no-redef]
        BLANK_IDX,
        CURRENT_DIR,
        DEFAULT_MOD_DATASET_DIR,
        INPUT_HEIGHT,
        INPUT_WIDTH,
        NUM_CLASSES,
    )
    from dataset import (  # type: ignore[no-redef]
        ManifestHudOcrDataset,
        ModHudOcrDataset,
        SyntheticHudOcrDataset,
        collate_ocr_batch,
    )
    from model import HudOcrCRNN, compute_cer, ctc_greedy_decode  # type: ignore[no-redef]
    from synth import resolve_font_path  # type: ignore[no-redef]


def _next_run_dir(runs_root: Path) -> Path:
    runs_root.mkdir(parents=True, exist_ok=True)
    idx = 1
    while (runs_root / f"exp{idx}").exists():
        idx += 1
    run_dir = runs_root / f"exp{idx}"
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    return run_dir


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compute_mixed_ctc_loss(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    criterion: nn.CTCLoss,
) -> torch.Tensor:
    """Compute weighted CTCLoss over non-empty targets and all-blank NLL over empty targets."""
    bsz, t_steps, _ = log_probs.shape
    non_empty_mask = target_lengths > 0
    empty_mask = ~non_empty_mask
    n_non = int(non_empty_mask.sum().item())
    n_empty = int(empty_mask.sum().item())

    lp_float = log_probs.float()
    total_loss = torch.zeros((), dtype=torch.float32, device=log_probs.device)

    if n_non > 0:
        lp_ne = lp_float[non_empty_mask].permute(1, 0, 2)  # (T, B_ne, C)
        input_lengths_ne = torch.full((n_non,), t_steps, dtype=torch.long, device=log_probs.device)
        target_lengths_ne = target_lengths[non_empty_mask]
        loss_ne = criterion(lp_ne, targets, input_lengths_ne, target_lengths_ne)
        total_loss = total_loss + loss_ne * (n_non / max(1, bsz))

    if n_empty > 0:
        loss_e = -lp_float[empty_mask, :, BLANK_IDX].mean()
        total_loss = total_loss + loss_e * (n_empty / max(1, bsz))

    return total_loss


@torch.inference_mode()
def evaluate(
    model: HudOcrCRNN,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    exact_matches = 0
    total_samples = 0
    all_preds: list[str] = []
    all_targets: list[str] = []

    for images, targets, target_lengths, texts in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        target_lengths = target_lengths.to(device, non_blocking=True)

        log_probs = model(images)  # (B, T, C)
        loss = _compute_mixed_ctc_loss(log_probs, targets, target_lengths, criterion)
        total_loss += float(loss.item())
        total_batches += 1

        decoded = ctc_greedy_decode(log_probs)
        preds = [d[0] for d in decoded]
        for pred, tgt in zip(preds, texts):
            if pred == tgt:
                exact_matches += 1
            total_samples += 1
        all_preds.extend(preds)
        all_targets.extend(texts)

    return {
        "loss": total_loss / max(1, total_batches),
        "exact_acc": exact_matches / max(1, total_samples),
        "cer": compute_cer(all_preds, all_targets),
    }


def _resolve_mod_dataset_dir(explicit_dir: str | None, synth_only: bool) -> Path | None:
    if synth_only:
        return None
    if explicit_dir is not None:
        p = Path(explicit_dir)
        if not (p / "labels").exists():
            raise FileNotFoundError(f"Mod HUD OCR labels directory not found: {p / 'labels'}")
        return p
    if (DEFAULT_MOD_DATASET_DIR / "labels").exists() and any((DEFAULT_MOD_DATASET_DIR / "labels").glob("frame_*.json")):
        return DEFAULT_MOD_DATASET_DIR
    return None


def train_ocr(args: argparse.Namespace) -> Path:
    _set_seed(args.seed)
    torch.backends.cudnn.benchmark = False

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    font_resolved = resolve_font_path(args.font_path)
    mod_dir = _resolve_mod_dataset_dir(
        getattr(args, "dataset_dir", None),
        bool(getattr(args, "synth_only", False)),
    )

    datasets_train: list[torch.utils.data.Dataset] = []
    datasets_val: list[torch.utils.data.Dataset] = []

    if mod_dir is not None:
        mod_train = ModHudOcrDataset(mod_dir, split="train", seed=args.seed, augment=True)
        mod_val = ModHudOcrDataset(mod_dir, split="val", seed=args.seed, augment=True)
        datasets_train.append(mod_train)
        if len(mod_val) > 0:
            datasets_val.append(mod_val)
        synth_mix = int(getattr(args, "synth_samples", 0) or 0)
        if synth_mix > 0:
            datasets_train.append(
                SyntheticHudOcrDataset(
                    num_samples=synth_mix,
                    seed=args.seed,
                    fixed=False,
                    augment=True,
                    font_path=args.font_path,
                )
            )
        source_desc = f"mod={mod_dir} (train={len(mod_train)}, val={len(mod_val)}, synth_mix={synth_mix})"
    else:
        synth_n = int(getattr(args, "synth_samples", None) or args.train_samples)
        datasets_train.append(
            SyntheticHudOcrDataset(
                num_samples=synth_n,
                seed=args.seed,
                fixed=False,
                augment=True,
                font_path=args.font_path,
            )
        )
        datasets_val.append(
            SyntheticHudOcrDataset(
                num_samples=args.val_samples,
                seed=args.seed + 500_000,
                fixed=True,
                augment=True,
                font_path=args.font_path,
            )
        )
        source_desc = f"synthetic (train={synth_n}, val={args.val_samples})"

    if args.manifest_dir:
        real_train = ManifestHudOcrDataset(args.manifest_dir, split="train")
        real_val = ManifestHudOcrDataset(args.manifest_dir, split="val")
        if len(real_train) > 0:
            datasets_train.append(real_train)
        if len(real_val) > 0:
            if args.val_real_only:
                datasets_val = [real_val]
            else:
                datasets_val.append(real_val)

    train_ds: torch.utils.data.Dataset = (
        datasets_train[0] if len(datasets_train) == 1 else ConcatDataset(datasets_train)
    )
    val_ds: torch.utils.data.Dataset = (
        datasets_val[0] if len(datasets_val) == 1 else ConcatDataset(datasets_val)
    )

    print(
        f"[HUD OCR] device={device} | input={INPUT_HEIGHT}x{INPUT_WIDTH} | "
        f"{source_desc} | font={font_resolved if font_resolved else 'PIL default fallback'}",
        flush=True,
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_ocr_batch,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_ocr_batch,
        persistent_workers=args.num_workers > 0,
    )

    model = HudOcrCRNN(
        num_classes=NUM_CLASSES,
        rnn_hidden=args.rnn_hidden,
        rnn_layers=args.rnn_layers,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.05)

    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    run_dir = _next_run_dir(Path(args.runs_dir) if getattr(args, "runs_dir", None) else (CURRENT_DIR / "runs"))
    metrics_path = run_dir / "metrics.csv"
    weights_out = Path(args.weights_out)
    weights_out.parent.mkdir(parents=True, exist_ok=True)
    run_best_path = run_dir / "weights" / "best.pt"

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "lr", "train_loss", "val_loss", "val_exact_acc", "val_cer", "epoch_sec"])

    best_exact = -1.0
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        running_loss = 0.0
        batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for images, targets, target_lengths, _texts in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            target_lengths = target_lengths.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                log_probs = model(images)  # (B, T, C)

            loss = _compute_mixed_ctc_loss(log_probs, targets, target_lengths, criterion)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            batches += 1
            pbar.set_postfix(loss=f"{running_loss / batches:.4f}")

        scheduler.step()
        train_loss = running_loss / max(1, batches)
        val_metrics = evaluate(model, val_loader, criterion, device)
        elapsed = time.perf_counter() - t0
        cur_lr = optimizer.param_groups[0]["lr"]

        with metrics_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{cur_lr:.6e}",
                f"{train_loss:.6f}",
                f"{val_metrics['loss']:.6f}",
                f"{val_metrics['exact_acc']:.6f}",
                f"{val_metrics['cer']:.6f}",
                f"{elapsed:.2f}",
            ])

        improved = (val_metrics["exact_acc"] > best_exact) or (
            abs(val_metrics["exact_acc"] - best_exact) < 1e-6 and val_metrics["loss"] < best_loss
        )
        marker = ""
        if improved:
            best_exact = val_metrics["exact_acc"]
            best_loss = val_metrics["loss"]
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "num_classes": NUM_CLASSES,
                    "rnn_hidden": args.rnn_hidden,
                    "rnn_layers": args.rnn_layers,
                    "input_height": INPUT_HEIGHT,
                    "input_width": INPUT_WIDTH,
                },
                "val_metrics": val_metrics,
            }
            torch.save(ckpt, str(run_best_path))
            torch.save(ckpt, str(weights_out))
            marker = " *"

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | val_exact={val_metrics['exact_acc']:.4f} | "
            f"val_cer={val_metrics['cer']:.4f} | {elapsed:.1f}s{marker}",
            flush=True,
        )

    print(f"[HUD OCR] Training complete. Best weights saved to: {weights_out}", flush=True)
    return weights_out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Balatro HUD OCR (CNN + BiGRU + CTC).")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--train-samples", type=int, default=40000)
    p.add_argument("--val-samples", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--rnn-hidden", type=int, default=128)
    p.add_argument("--rnn-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--font-path", type=str, default=None)
    p.add_argument("--dataset-dir", type=str, default=None, help="Path to mod-captured dataset_hud_ocr directory")
    p.add_argument("--synth-only", action="store_true", help="Force synthetic-only training even if mod dataset exists")
    p.add_argument("--synth-samples", type=int, default=None, help="Extra synthetic samples to mix with mod dataset")
    p.add_argument("--manifest-dir", type=str, default=None)
    p.add_argument("--val-real-only", action="store_true")
    p.add_argument("--weights-out", type=str, default=str(CURRENT_DIR / "weights" / "hud_ocr_best.pt"))
    p.add_argument("--runs-dir", type=str, default=None)
    return p


if __name__ == "__main__":
    train_ocr(build_arg_parser().parse_args())
