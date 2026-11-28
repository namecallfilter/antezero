"""Evaluation, error profiling, and timeline visualization for END_ROUND recognition."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from .config import DEFAULT_SEED, DEFAULT_VAL_RATIO, STATES
    from .dataset import EndRoundClipDataset, load_prepared_manifest, split_manifest_records
    from .pipeline import EndRoundRecognitionPipeline, EndRoundResult
    from .prepare_dataset import PREPARED_DIR
except ImportError:
    from config import DEFAULT_SEED, DEFAULT_VAL_RATIO, STATES  # type: ignore[no-redef]
    from dataset import EndRoundClipDataset, load_prepared_manifest, split_manifest_records  # type: ignore[no-redef]
    from pipeline import EndRoundRecognitionPipeline, EndRoundResult  # type: ignore[no-redef]
    from prepare_dataset import PREPARED_DIR  # type: ignore[no-redef]

CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_EVAL_OUT = CURRENT_DIR / "figures" / "eval"


def plot_clip_timeline(
    result: EndRoundResult,
    out_path: Path,
    *,
    title: str,
    gt_step: int | None = None,
) -> None:
    """Save stacked state-probability + Gaussian boundary probability timeline."""
    if result.state_probs is None or result.boundary_probs is None:
        return

    t_len = result.state_probs.shape[0]
    t_axis = np.arange(t_len, dtype=np.float64) / max(1e-3, result.src_fps if result.sampled_video_frames is None else 15.0)
    if result.sampled_video_frames is not None and len(result.sampled_video_frames) == t_len:
        t_axis = np.asarray(result.sampled_video_frames, dtype=np.float64) / max(1e-3, result.src_fps)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)

    colors = ["tab:gray", "tab:orange", "tab:green", "tab:blue"]
    for idx, state_name in enumerate(STATES):
        ax1.plot(t_axis, result.state_probs[:, idx], label=state_name, color=colors[idx], linewidth=1.6)
    ax1.set_ylabel("State Prob")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title(title)

    ax2.plot(t_axis, result.boundary_probs, label="boundary_prob", color="tab:red", linewidth=1.8)
    pred_x = float(t_axis[min(result.model_step_index, t_len - 1)])
    ax2.axvline(pred_x, color="black", linestyle="--", linewidth=1.5, label=f"Pred ({pred_x:.2f}s)")
    if gt_step is not None and 0 <= gt_step < t_len:
        gt_x = float(t_axis[gt_step])
        ax2.axvline(gt_x, color="tab:purple", linestyle=":", linewidth=2.0, label=f"GT ({gt_x:.2f}s)")
    ax2.set_ylabel("Boundary Prob")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def evaluate_dataset(args: argparse.Namespace) -> dict[str, Any]:
    pipeline = EndRoundRecognitionPipeline(weights_path=args.weights)
    dataset_dir, records = load_prepared_manifest(args.dataset_dir)
    train_records, val_records = split_manifest_records(
        records,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    if args.split == "val":
        target_records = val_records or train_records
    elif args.split == "train":
        target_records = train_records
    else:
        target_records = records

    ds = EndRoundClipDataset(
        target_records,
        dataset_dir,
        is_train=False,
        max_seq_len=None,
        cache_in_ram=False,
        use_temporal_aug=False,
        seed=args.seed,
    )

    rows: list[dict[str, Any]] = []
    worst_cases: list[tuple[float, dict[str, Any], EndRoundResult]] = []

    for idx in range(len(ds)):
        item = ds[idx]
        gt_er = item["end_round_frame"]
        res = pipeline.predict_tensor(
            item["frames"],
            src_fps=item["fps"],
            return_profiles=True,
        )
        if gt_er is None:
            continue

        err_f = abs(float(res.model_step_index) - float(gt_er))
        err_ms = err_f * (1000.0 / max(1e-3, item["fps"]))
        rec_out = {
            "clip_id": item["clip_id"],
            "source": item["source"],
            "fps": float(item["fps"]),
            "num_frames": int(item["frames"].shape[0]),
            "gt_step": int(gt_er),
            "pred_step": int(res.model_step_index),
            "err_frames": float(err_f),
            "err_ms": float(err_ms),
            "confidence": float(res.confidence),
        }
        rows.append(rec_out)
        worst_cases.append((err_f, rec_out, res))

    if not rows:
        raise RuntimeError(f"No clips with ground-truth end_round_frame found in split '{args.split}'.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def summarize(subset: list[dict[str, Any]]) -> dict[str, float]:
        ef = np.asarray([r["err_frames"] for r in subset], dtype=np.float64)
        em = np.asarray([r["err_ms"] for r in subset], dtype=np.float64)
        return {
            "count": float(len(subset)),
            "mae_frames": float(np.mean(ef)),
            "median_frames": float(np.median(ef)),
            "p90_frames": float(np.percentile(ef, 90)),
            "mae_ms": float(np.mean(em)),
            "tol_1_acc": float(np.mean(ef <= 1.0)),
            "tol_2_acc": float(np.mean(ef <= 2.0)),
            "tol_3_acc": float(np.mean(ef <= 3.0)),
        }

    overall = summarize(rows)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)

    summary = {
        "split": args.split,
        "weights": str(pipeline.weights_path),
        "overall": overall,
        "by_source": {src: summarize(sub) for src, sub in sorted(by_source.items())},
        "clips": rows,
    }
    (out_dir / f"eval_{args.split}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"[eval:{args.split}] n={int(overall['count'])} MAE={overall['mae_frames']:.2f}f "
        f"({overall['mae_ms']:.0f}ms) | median={overall['median_frames']:.2f}f | "
        f"P90={overall['p90_frames']:.2f}f | <=1f={overall['tol_1_acc']:.3f} "
        f"<=2f={overall['tol_2_acc']:.3f} <=3f={overall['tol_3_acc']:.3f}"
    )
    for src, stats in summary["by_source"].items():
        print(
            f"  [{src}] n={int(stats['count'])} MAE={stats['mae_frames']:.2f}f "
            f"({stats['mae_ms']:.0f}ms) <=2f={stats['tol_2_acc']:.3f}"
        )

    # Error distribution histogram
    err_vals = [r["err_frames"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(0, max(6.0, float(max(err_vals)) + 2.0)) - 0.5
    ax.hist(err_vals, bins=bins, color="steelblue", edgecolor="black", alpha=0.85)
    ax.set_xlabel("|Pred - GT| Error (frames @ clip fps)")
    ax.set_ylabel("Clip Count")
    ax.set_title(f"END_ROUND Frame Error Distribution ({args.split})")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"error_hist_{args.split}.png", dpi=140)
    plt.close(fig)

    # Save timeline plots for worst clips
    worst_cases.sort(key=lambda item: item[0], reverse=True)
    for rank, (_, rec_out, res) in enumerate(worst_cases[: max(0, int(args.plot_worst))], start=1):
        safe_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in rec_out["clip_id"])
        title = (
            f"#{rank} {rec_out['clip_id']} ({rec_out['source']}) | "
            f"GT={rec_out['gt_step']} Pred={rec_out['pred_step']} Err={rec_out['err_frames']:.1f}f"
        )
        plot_clip_timeline(
            res,
            out_dir / f"worst_{rank:02d}_{safe_id}.png",
            title=title,
            gt_step=rec_out["gt_step"],
        )

    return summary


def evaluate_video_segment(args: argparse.Namespace) -> EndRoundResult:
    pipeline = EndRoundRecognitionPipeline(weights_path=args.weights)
    res = pipeline.predict_video_segment(
        args.video,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        start_ts=args.start_ts,
        end_ts=args.end_ts,
        already_cropped=args.already_cropped,
        return_profiles=True,
    )
    print(
        f"[video] predicted end_round_frame={res.end_round_frame} "
        f"(t={res.end_round_timestamp:.3f}s @ {res.src_fps:.2f}fps, "
        f"model_step={res.model_step_index}/{res.num_sampled_frames}, conf={res.confidence:.3f})"
    )

    if res.decoded_states is not None and len(res.decoded_states) > 0:
        cur = int(res.decoded_states[0])
        seg_start = 0
        vf = res.sampled_video_frames or list(range(len(res.decoded_states)))
        print("Decoded segments:")
        for idx in range(1, len(res.decoded_states) + 1):
            nxt = int(res.decoded_states[idx]) if idx < len(res.decoded_states) else -1
            if nxt != cur:
                t0 = vf[seg_start] / max(1e-3, res.src_fps)
                t1 = vf[idx - 1] / max(1e-3, res.src_fps)
                print(f"  {STATES[cur]:16s}: frames [{vf[seg_start]}..{vf[idx - 1]}] ({t0:.2f}s - {t1:.2f}s)")
                seg_start = idx
                cur = nxt

    if args.plot_out:
        gt_step = None
        if args.gt_frame is not None and res.sampled_video_frames:
            arr = np.asarray(res.sampled_video_frames, dtype=np.int64)
            gt_step = int(np.argmin(np.abs(arr - int(args.gt_frame))))
        plot_clip_timeline(
            res,
            Path(args.plot_out),
            title=f"{Path(args.video).name} -> END_ROUND @ {res.end_round_timestamp:.2f}s (conf={res.confidence:.2f})",
            gt_step=gt_step,
        )
        print(f"[video] Saved timeline plot -> {args.plot_out}")

    return res


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained END_ROUND recognition model.")
    parser.add_argument("--weights", type=Path, default=None, help="Path to checkpoint (auto-discovers best weights if omitted).")
    parser.add_argument("--dataset-dir", type=Path, default=PREPARED_DIR)
    parser.add_argument("--split", type=str, default="val", choices=("val", "train", "all"))
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_EVAL_OUT)
    parser.add_argument("--plot-worst", type=int, default=5, help="Save timeline plots for top N error clips.")

    # Optional single video-segment inspection mode
    parser.add_argument("--video", type=Path, default=None, help="Optional video file to run segment inference on.")
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--start-ts", type=float, default=None)
    parser.add_argument("--end-ts", type=float, default=None)
    parser.add_argument("--gt-frame", type=int, default=None, help="Optional GT video frame for plotting.")
    parser.add_argument("--already-cropped", action="store_true")
    parser.add_argument("--plot-out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.video is not None:
        evaluate_video_segment(args)
    else:
        evaluate_dataset(args)


if __name__ == "__main__":
    main()
