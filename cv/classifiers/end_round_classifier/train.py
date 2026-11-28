"""Training entry point for the temporal END_ROUND recognition model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .config import (
        BOUNDARY_SIGMA_FRAMES,
        DEFAULT_BACKBONE,
        DEFAULT_BATCH_SIZE,
        DEFAULT_BOUNDARY_LOSS_WEIGHT,
        DEFAULT_BOUNDARY_REFINE_ALPHA,
        DEFAULT_BOUNDARY_REFINE_RADIUS,
        DEFAULT_DROPOUT,
        DEFAULT_EPOCHS,
        DEFAULT_FRAME_FEAT_DIM,
        DEFAULT_LR,
        DEFAULT_MAX_TRAIN_SEQ_LEN,
        DEFAULT_MIN_POST_PLAY_FRAMES,
        DEFAULT_MIN_SCORING_FRAMES,
        DEFAULT_NUM_GRU_LAYERS,
        DEFAULT_SEED,
        DEFAULT_SPATIAL_POOL_W,
        DEFAULT_TEMPORAL_HIDDEN_DIM,
        DEFAULT_TRANSITION_SMOOTH_WEIGHT,
        DEFAULT_VAL_RATIO,
        DEFAULT_WEIGHT_DECAY,
        IGNORE_INDEX,
        INPUT_HEIGHT,
        INPUT_WIDTH,
        NUM_STATES,
        PLAY_AREA_ROI,
        STATES,
        TARGET_FPS,
    )
    from .dataset import (
        EndRoundClipDataset,
        collate_end_round_batch,
        load_prepared_manifest,
        split_manifest_records,
    )
    from .model import SUPPORTED_BACKBONES, TemporalEndRoundNet, predict_end_round_index
    from .prepare_dataset import PREPARED_DIR, prepare_dataset
    from .sync_dataset import sync_dataset
    from .temporal_augmentations import TemporalCursorOverlay, apply_temporal_clip_augmentations
except ImportError:
    from config import (  # type: ignore[no-redef]
        BOUNDARY_SIGMA_FRAMES,
        DEFAULT_BACKBONE,
        DEFAULT_BATCH_SIZE,
        DEFAULT_BOUNDARY_LOSS_WEIGHT,
        DEFAULT_BOUNDARY_REFINE_ALPHA,
        DEFAULT_BOUNDARY_REFINE_RADIUS,
        DEFAULT_DROPOUT,
        DEFAULT_EPOCHS,
        DEFAULT_FRAME_FEAT_DIM,
        DEFAULT_LR,
        DEFAULT_MAX_TRAIN_SEQ_LEN,
        DEFAULT_MIN_POST_PLAY_FRAMES,
        DEFAULT_MIN_SCORING_FRAMES,
        DEFAULT_NUM_GRU_LAYERS,
        DEFAULT_SEED,
        DEFAULT_SPATIAL_POOL_W,
        DEFAULT_TEMPORAL_HIDDEN_DIM,
        DEFAULT_TRANSITION_SMOOTH_WEIGHT,
        DEFAULT_VAL_RATIO,
        DEFAULT_WEIGHT_DECAY,
        IGNORE_INDEX,
        INPUT_HEIGHT,
        INPUT_WIDTH,
        NUM_STATES,
        PLAY_AREA_ROI,
        STATES,
        TARGET_FPS,
    )
    from dataset import (  # type: ignore[no-redef]
        EndRoundClipDataset,
        collate_end_round_batch,
        load_prepared_manifest,
        split_manifest_records,
    )
    from model import SUPPORTED_BACKBONES, TemporalEndRoundNet, predict_end_round_index  # type: ignore[no-redef]
    from prepare_dataset import PREPARED_DIR, prepare_dataset  # type: ignore[no-redef]
    from sync_dataset import sync_dataset  # type: ignore[no-redef]
    from temporal_augmentations import (  # type: ignore[no-redef]
        TemporalCursorOverlay,
        apply_temporal_clip_augmentations,
    )

CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR = CURRENT_DIR / "runs"
DEFAULT_WEIGHTS_OUT = CURRENT_DIR / "weights" / "end_round_best.pt"


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_run_dir(runs_dir: Path, name: str | None, exist_ok: bool) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    if name:
        candidate = runs_dir / name
        if candidate.exists() and not exist_ok:
            idx = 2
            while (runs_dir / f"{name}_{idx}").exists():
                idx += 1
            candidate = runs_dir / f"{name}_{idx}"
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    idx = 1
    while True:
        suffix = "" if idx == 1 else str(idx)
        candidate = runs_dir / f"exp{suffix}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        idx += 1


def recommend_batch_size(device: torch.device, requested: int) -> int:
    if requested > 0:
        return requested
    if device.type != "cuda":
        return 4
    vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    if vram_gb >= 24.0:
        return 16
    if vram_gb >= 12.0:
        return 12
    if vram_gb >= 8.0:
        return 8
    return 4


def compute_multitask_loss(
    outputs: dict[str, torch.Tensor],
    state_labels: torch.Tensor,
    boundary_targets: torch.Tensor,
    end_round_frames: list[int | None],
    *,
    boundary_weight: float = DEFAULT_BOUNDARY_LOSS_WEIGHT,
    smooth_weight: float = DEFAULT_TRANSITION_SMOOTH_WEIGHT,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute masked state CE + positive-weighted boundary BCE + transition smoothness."""
    state_logits = outputs["state_logits"]
    boundary_logits = outputs["boundary_logits"]
    valid_mask = outputs["valid_mask"]

    state_mask = valid_mask & (state_labels != IGNORE_INDEX)
    if bool(state_mask.any()):
        state_loss = F.cross_entropy(
            state_logits[state_mask],
            state_labels[state_mask],
            label_smoothing=0.03,
        )
    else:
        state_loss = state_logits.sum() * 0.0

    # Only supervise boundary head on clips with a known end_round_frame
    clip_has_boundary = torch.tensor(
        [er is not None for er in end_round_frames],
        dtype=torch.bool,
        device=valid_mask.device,
    )
    b_mask = valid_mask & clip_has_boundary.unsqueeze(1)
    if bool(b_mask.any()):
        b_logits_sel = boundary_logits[b_mask]
        b_targs_sel = boundary_targets[b_mask]
        bce = F.binary_cross_entropy_with_logits(b_logits_sel, b_targs_sel, reduction="none")
        # Emphasize sparse Gaussian positive frames near END_ROUND
        pos_weights = 1.0 + 4.0 * b_targs_sel
        boundary_loss = (bce * pos_weights).mean()
    else:
        boundary_loss = boundary_logits.sum() * 0.0

    # Temporal truncated MSE smoothness over adjacent valid frame pairs
    if state_logits.shape[1] > 1:
        pair_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
        if bool(pair_mask.any()):
            log_probs = F.log_softmax(state_logits.float(), dim=-1)
            diff_sq = (log_probs[:, 1:] - log_probs[:, :-1]).pow(2)
            diff_clamped = torch.clamp(diff_sq, max=16.0).mean(dim=-1)
            smooth_loss = diff_clamped[pair_mask].mean()
        else:
            smooth_loss = state_logits.sum() * 0.0
    else:
        smooth_loss = state_logits.sum() * 0.0

    total_loss = state_loss + boundary_weight * boundary_loss + smooth_weight * smooth_loss
    stats = {
        "total": float(total_loss.detach().item()),
        "state": float(state_loss.detach().item()),
        "boundary": float(boundary_loss.detach().item()),
        "smooth": float(smooth_loss.detach().item()),
    }
    return total_loss, stats


def build_checkpoint_payload(
    model: TemporalEndRoundNet,
    *,
    epoch: int,
    metrics: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
        "model_config": {
            "backbone_name": model.backbone_name,
            "spatial_pool_w": model.spatial_pool_w,
            "frame_feat_dim": model.frame_feat_dim,
            "temporal_hidden_dim": model.temporal_hidden_dim,
            "num_gru_layers": model.num_gru_layers,
            "dropout": float(args.dropout),
            "num_states": model.num_states,
            "states": list(STATES),
            "input_width": INPUT_WIDTH,
            "input_height": INPUT_HEIGHT,
            "target_fps": TARGET_FPS,
            "roi_base_1080p": dict(PLAY_AREA_ROI),
            "viterbi_params": {
                "min_scoring_frames": DEFAULT_MIN_SCORING_FRAMES,
                "min_post_play_frames": DEFAULT_MIN_POST_PLAY_FRAMES,
                "refine_radius": DEFAULT_BOUNDARY_REFINE_RADIUS,
                "refine_alpha": DEFAULT_BOUNDARY_REFINE_ALPHA,
            },
        },
        "train_args": {
            key: str(val) if isinstance(val, Path) else val
            for key, val in vars(args).items()
        },
    }


def save_training_plots(
    history: list[dict[str, Any]],
    conf_mat: np.ndarray,
    run_dir: Path,
) -> None:
    if not history:
        return
    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(epochs, [h["train_loss"] for h in history], label="Train Total")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="Val Total")
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, [h["val_frame_acc"] for h in history], label="Val State Acc")
    axes[1].plot(epochs, [h["val_tol_2_acc"] for h in history], label="Val <=2f Acc")
    axes[1].plot(epochs, [h["val_tol_1_acc"] for h in history], label="Val <=1f Acc")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("State & Event Recall")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(epochs, [h["val_mae_frames"] for h in history], label="Val MAE (frames)", color="tab:red")
    axes[2].set_title("END_ROUND Error (frames @15fps)")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(run_dir / "training_curves.png", dpi=140)
    plt.close(fig)

    # State confusion matrix
    fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4.8))
    row_sums = np.maximum(conf_mat.sum(axis=1, keepdims=True), 1)
    norm_cm = conf_mat.astype(np.float64) / row_sums
    im = ax_cm.imshow(norm_cm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax_cm.set_xticks(range(NUM_STATES))
    ax_cm.set_yticks(range(NUM_STATES))
    short_names = ["PRE", "SCOR", "POST", "EVAL"]
    ax_cm.set_xticklabels(short_names)
    ax_cm.set_yticklabels(short_names)
    ax_cm.set_xlabel("Predicted State")
    ax_cm.set_ylabel("True State")
    ax_cm.set_title("Validation Frame State Confusion")
    for r in range(NUM_STATES):
        for c in range(NUM_STATES):
            val = norm_cm[r, c]
            color = "white" if val > 0.55 else "black"
            ax_cm.text(c, r, f"{val:.2f}\n({conf_mat[r, c]})", ha="center", va="center", color=color, fontsize=8)
    fig_cm.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    fig_cm.tight_layout()
    fig_cm.savefig(run_dir / "confusion_matrix.png", dpi=140)
    plt.close(fig_cm)


def run_epoch(
    model: TemporalEndRoundNet,
    loader: DataLoader,
    *,
    device: torch.device,
    is_train: bool,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    amp_dtype: torch.dtype,
    cursor_overlay: TemporalCursorOverlay | None,
    use_gpu_aug: bool,
    boundary_weight: float,
    smooth_weight: float,
) -> tuple[dict[str, float], np.ndarray]:
    model.train(is_train)

    total_loss_sum = 0.0
    state_loss_sum = 0.0
    boundary_loss_sum = 0.0
    n_batches = 0

    correct_frames = 0
    total_labeled_frames = 0
    conf_mat = np.zeros((NUM_STATES, NUM_STATES), dtype=np.int64)

    frame_errors: list[float] = []
    ms_errors: list[float] = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    pbar = tqdm(
        loader,
        desc="train" if is_train else "val",
        leave=False,
        unit="batch",
        dynamic_ncols=True,
    )
    with context:
        for batch in pbar:
            frames_u8 = batch["frames"].to(device, non_blocking=True)
            frames = frames_u8.float() / 255.0
            state_labels = batch["state_labels"].to(device, non_blocking=True)
            boundary_targets = batch["boundary_targets"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)
            end_round_frames = batch["end_round_frames"]
            fps_list = batch["fps"]

            if is_train and use_gpu_aug:
                frames = apply_temporal_clip_augmentations(
                    frames,
                    lengths,
                    cursor_overlay=cursor_overlay,
                )

            if is_train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                outputs = model(frames, lengths=lengths)
                loss, loss_parts = compute_multitask_loss(
                    outputs,
                    state_labels,
                    boundary_targets,
                    end_round_frames,
                    boundary_weight=boundary_weight,
                    smooth_weight=smooth_weight,
                )

            if is_train and optimizer is not None:
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

            total_loss_sum += loss_parts["total"]
            state_loss_sum += loss_parts["state"]
            boundary_loss_sum += loss_parts["boundary"]
            n_batches += 1
            pbar.set_postfix(loss=f"{total_loss_sum / n_batches:.4f}", refresh=False)

            # Evaluate frame state accuracy & confusion matrix
            s_logits = outputs["state_logits"].detach().float().cpu()
            b_logits = outputs["boundary_logits"].detach().float().cpu()
            valid_mask_cpu = outputs["valid_mask"].detach().cpu()
            s_labels_cpu = state_labels.detach().cpu()

            preds_all = s_logits.argmax(dim=-1)
            labeled_mask = valid_mask_cpu & (s_labels_cpu != IGNORE_INDEX)
            if bool(labeled_mask.any()):
                y_true = s_labels_cpu[labeled_mask].numpy()
                y_pred = preds_all[labeled_mask].numpy()
                correct_frames += int((y_true == y_pred).sum())
                total_labeled_frames += int(len(y_true))
                np.add.at(conf_mat, (y_true, y_pred), 1)

            if not is_train:
                for b in range(len(end_round_frames)):
                    gt_er = end_round_frames[b]
                    if gt_er is None:
                        continue
                    t_len = int(batch["lengths"][b].item())
                    dec = predict_end_round_index(
                        s_logits[b, :t_len].numpy(),
                        b_logits[b, :t_len].numpy(),
                    )
                    err_f = abs(float(dec["end_round_frame"]) - float(gt_er))
                    clip_fps = max(1e-3, float(fps_list[b]))
                    frame_errors.append(err_f)
                    ms_errors.append(err_f * (1000.0 / clip_fps))

    denom = max(1, n_batches)
    metrics: dict[str, float] = {
        "loss": total_loss_sum / denom,
        "state_loss": state_loss_sum / denom,
        "boundary_loss": boundary_loss_sum / denom,
        "frame_acc": float(correct_frames) / max(1, total_labeled_frames),
    }
    if frame_errors:
        err_arr = np.asarray(frame_errors, dtype=np.float64)
        ms_arr = np.asarray(ms_errors, dtype=np.float64)
        metrics["mae_frames"] = float(np.mean(err_arr))
        metrics["median_frames"] = float(np.median(err_arr))
        metrics["mae_ms"] = float(np.mean(ms_arr))
        metrics["tol_1_acc"] = float(np.mean(err_arr <= 1.0))
        metrics["tol_2_acc"] = float(np.mean(err_arr <= 2.0))
        metrics["tol_3_acc"] = float(np.mean(err_arr <= 3.0))
    else:
        metrics["mae_frames"] = 0.0
        metrics["median_frames"] = 0.0
        metrics["mae_ms"] = 0.0
        metrics["tol_1_acc"] = 0.0
        metrics["tol_2_acc"] = 0.0
        metrics["tol_3_acc"] = 0.0

    return metrics, conf_mat


def train(args: argparse.Namespace) -> Path:
    seed_everything(args.seed)

    if args.sync:
        sync_dataset()
    if args.prepare:
        prepare_dataset()

    dataset_dir, records = load_prepared_manifest(args.dataset_dir)
    if not records:
        raise RuntimeError(f"Prepared manifest at {dataset_dir} contains 0 clips.")

    train_records, val_records = split_manifest_records(
        records,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    if not val_records:
        print("[warn] Dataset has <=1 group; reusing train split for validation sanity checks.")
        val_records = list(train_records)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = recommend_batch_size(device, args.batch_size)
    use_amp = (device.type == "cuda") and (not args.no_amp)
    if use_amp and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    elif use_amp:
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    if device.type == "cuda":
        # Sequences have variable sum(lengths) per batch; cudnn.benchmark=True
        # re-autotunes convolutions on every new shape (~9s/batch penalty).
        torch.backends.cudnn.benchmark = False

    train_ds = EndRoundClipDataset(
        train_records,
        dataset_dir,
        is_train=True,
        max_seq_len=args.max_seq_len,
        sigma=BOUNDARY_SIGMA_FRAMES,
        cache_in_ram=not args.no_ram_cache,
        use_temporal_aug=not args.no_augment,
        seed=args.seed,
    )
    val_ds = EndRoundClipDataset(
        val_records,
        dataset_dir,
        is_train=False,
        max_seq_len=None,
        sigma=BOUNDARY_SIGMA_FRAMES,
        cache_in_ram=not args.no_ram_cache,
        use_temporal_aug=False,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=min(batch_size, max(1, len(train_ds))),
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_end_round_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(batch_size, max(1, len(val_ds))),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_end_round_batch,
    )

    model = TemporalEndRoundNet(
        backbone_name=args.backbone,
        pretrained=not args.no_pretrained,
        spatial_pool_w=args.spatial_pool_w,
        frame_feat_dim=args.frame_feat_dim,
        temporal_hidden_dim=args.temporal_hidden_dim,
        num_gru_layers=args.num_gru_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=max(1e-6, args.lr * 0.02),
    )

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"[resume] Loaded checkpoint from {args.resume} (continuing from epoch {start_epoch})")

    cursor_overlay = TemporalCursorOverlay(device=device) if (not args.no_augment) else None

    run_dir = resolve_run_dir(Path(args.runs_dir), args.name, args.exist_ok)
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_path = weights_dir / "best.pt"
    last_path = weights_dir / "last.pt"

    print(
        f"[train] device={device.type} amp={amp_dtype if use_amp else 'off'} "
        f"backbone={args.backbone} train_clips={len(train_ds)} val_clips={len(val_ds)} "
        f"batch_size={batch_size} run_dir={run_dir}",
        flush=True,
    )

    history: list[dict[str, Any]] = []
    best_score = (float("inf"), float("inf"))  # Minimize (mae_frames, val_loss)
    best_conf_mat = np.zeros((NUM_STATES, NUM_STATES), dtype=np.int64)
    epochs_without_improve = 0

    csv_path = run_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "epoch",
                "lr",
                "train_loss",
                "train_frame_acc",
                "val_loss",
                "val_frame_acc",
                "val_mae_frames",
                "val_mae_ms",
                "val_tol_1_acc",
                "val_tol_2_acc",
                "val_tol_3_acc",
                "elapsed_s",
            ],
        )
        writer.writeheader()

        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.perf_counter()
            train_metrics, _ = run_epoch(
                model,
                train_loader,
                device=device,
                is_train=True,
                optimizer=optimizer,
                scaler=scaler,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                cursor_overlay=cursor_overlay,
                use_gpu_aug=not args.no_augment,
                boundary_weight=args.boundary_weight,
                smooth_weight=args.smooth_weight,
            )
            scheduler.step()

            val_metrics, val_cm = run_epoch(
                model,
                val_loader,
                device=device,
                is_train=False,
                optimizer=None,
                scaler=scaler,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                cursor_overlay=None,
                use_gpu_aug=False,
                boundary_weight=args.boundary_weight,
                smooth_weight=args.smooth_weight,
            )

            elapsed = time.perf_counter() - t0
            current_lr = float(optimizer.param_groups[0]["lr"])
            row = {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train_metrics["loss"],
                "train_frame_acc": train_metrics["frame_acc"],
                "val_loss": val_metrics["loss"],
                "val_frame_acc": val_metrics["frame_acc"],
                "val_mae_frames": val_metrics["mae_frames"],
                "val_mae_ms": val_metrics["mae_ms"],
                "val_tol_1_acc": val_metrics["tol_1_acc"],
                "val_tol_2_acc": val_metrics["tol_2_acc"],
                "val_tol_3_acc": val_metrics["tol_3_acc"],
                "elapsed_s": elapsed,
            }
            history.append(row)
            writer.writerow(row)
            csv_file.flush()

            print(
                f"Epoch {epoch:03d}/{args.epochs:03d} ({elapsed:.1f}s) | "
                f"train loss={row['train_loss']:.4f} acc={row['train_frame_acc']:.3f} | "
                f"val loss={row['val_loss']:.4f} acc={row['val_frame_acc']:.3f} "
                f"MAE={row['val_mae_frames']:.2f}f ({row['val_mae_ms']:.0f}ms) "
                f"<=2f={row['val_tol_2_acc']:.3f}",
                flush=True,
            )

            payload = build_checkpoint_payload(model, epoch=epoch, metrics=row, args=args)
            torch.save(payload, last_path)

            score = (val_metrics["mae_frames"], val_metrics["loss"])
            if score < best_score:
                best_score = score
                best_conf_mat = val_cm.copy()
                epochs_without_improve = 0
                torch.save(payload, best_path)
                if args.weights_out:
                    weights_out = Path(args.weights_out)
                    weights_out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(best_path, weights_out)
                print(
                    f"  [best] Saved new best checkpoint -> {best_path} "
                    f"(MAE={val_metrics['mae_frames']:.2f}f)"
                )
            else:
                epochs_without_improve += 1

            if args.patience > 0 and epochs_without_improve >= args.patience:
                print(f"[early-stop] No improvement for {args.patience} epochs.")
                break

    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    save_training_plots(history, best_conf_mat, run_dir)
    print(f"[done] Best checkpoint: {best_path}")
    return best_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train temporal END_ROUND recognition model.")
    parser.add_argument("--sync", action="store_true", help="Sync raw clips from Balatro mod first.")
    parser.add_argument("--prepare", action="store_true", help="Rebuild prepared .pt tensors first.")
    parser.add_argument("--dataset-dir", type=Path, default=PREPARED_DIR)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--weights-out", type=Path, default=DEFAULT_WEIGHTS_OUT)
    parser.add_argument("--name", type=str, default=None, help="Run subdirectory name.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reusing existing --name directory.")
    parser.add_argument("--resume", type=Path, default=None, help="Resume weights from checkpoint.")

    parser.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE, choices=SUPPORTED_BACKBONES)
    parser.add_argument("--spatial-pool-w", type=int, default=DEFAULT_SPATIAL_POOL_W)
    parser.add_argument("--frame-feat-dim", type=int, default=DEFAULT_FRAME_FEAT_DIM)
    parser.add_argument("--temporal-hidden-dim", type=int, default=DEFAULT_TEMPORAL_HIDDEN_DIM)
    parser.add_argument("--num-gru-layers", type=int, default=DEFAULT_NUM_GRU_LAYERS)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)

    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="0 = auto from VRAM.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--boundary-weight", type=float, default=DEFAULT_BOUNDARY_LOSS_WEIGHT)
    parser.add_argument("--smooth-weight", type=float, default=DEFAULT_TRANSITION_SMOOTH_WEIGHT)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_TRAIN_SEQ_LEN)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 works best with RAM cache on Windows).")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience (0 disables).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument("--no-pretrained", action="store_true", help="Do not load ImageNet weights.")
    parser.add_argument("--no-ram-cache", action="store_true", help="Stream clips from disk instead of caching in RAM.")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA automatic mixed precision.")
    parser.add_argument("--no-augment", action="store_true", help="Disable temporal/photometric training augmentations.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
