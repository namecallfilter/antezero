"""
Prepare End-Round Temporal Dataset
==================================
Converts raw clip directories (from `dataset/raw/` and optional `dataset/real_vod/`)
into uniform, training-ready `.pt` clip tensors under `dataset/prepared/`:
  - Crops `play_area` if `already_cropped` is False
  - Resizes frames to `(INPUT_WIDTH, INPUT_HEIGHT)` (`256x72`)
  - Temporally resamples clips near `TARGET_FPS` (`15.0`)
  - Generates per-frame 4-state labels and Gaussian boundary heatmaps via `schema.py`
  - Writes `dataset/prepared/manifest.json` with clip paths and metadata
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .config import (
        BOUNDARY_SIGMA_FRAMES,
        IGNORE_INDEX,
        INPUT_HEIGHT,
        INPUT_WIDTH,
        STATES,
        TARGET_FPS,
    )
    from .schema import ClipMeta, build_frame_targets
    from .video_io import crop_and_resize_rgb, read_video_segment_rgb, resample_frame_indices
except ImportError:
    from config import (
        BOUNDARY_SIGMA_FRAMES,
        IGNORE_INDEX,
        INPUT_HEIGHT,
        INPUT_WIDTH,
        STATES,
        TARGET_FPS,
    )
    from schema import ClipMeta, build_frame_targets
    from video_io import crop_and_resize_rgb, read_video_segment_rgb, resample_frame_indices

DATASET_DIR = CURRENT_DIR / "dataset"
RAW_DIR = DATASET_DIR / "raw"
REAL_VOD_DIR = DATASET_DIR / "real_vod"
PREPARED_DIR = DATASET_DIR / "prepared"


def _resolve_payload_path(clip_dir: Path, payload_file: str | None, fallbacks: tuple[str, ...]) -> Path:
    if payload_file:
        p = Path(payload_file)
        if not p.is_absolute():
            p = clip_dir / p
        if p.exists():
            return p
    for name in fallbacks:
        cand = clip_dir / name
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Could not find payload in {clip_dir} (payload_file={payload_file!r})")


def _load_raw_clip_frames(clip_dir: Path, meta: ClipMeta) -> tuple[torch.Tensor, float]:
    """Loads clip frames as `(T, 3, INPUT_HEIGHT, INPUT_WIDTH)` uint8 RGB tensor and effective FPS."""
    fmt = meta.frame_format

    if fmt in ("raw_rgb_u8", "raw_rgba_u8"):
        channels = 3 if fmt == "raw_rgb_u8" else 4
        bin_path = _resolve_payload_path(clip_dir, meta.payload_file, ("frames.bin", "frames.raw"))
        expected_bytes = meta.num_frames * meta.height * meta.width * channels
        raw = np.fromfile(bin_path, dtype=np.uint8)
        if raw.size != expected_bytes:
            raise ValueError(
                f"{bin_path}: expected {expected_bytes} bytes for ({meta.num_frames}, {meta.height}, {meta.width}, {channels}), got {raw.size}"
            )
        arr = raw.reshape(meta.num_frames, meta.height, meta.width, channels)
        if channels == 4:
            arr = arr[..., :3]

        keep = resample_frame_indices(0, meta.num_frames, meta.fps, TARGET_FPS)
        arr = arr[keep]
        resized = [
            crop_and_resize_rgb(
                arr[i],
                already_cropped=meta.already_cropped,
                roi_base_1080p=meta.roi_base_1080p,
                out_width=INPUT_WIDTH,
                out_height=INPUT_HEIGHT,
            )
            for i in range(arr.shape[0])
        ]
        stacked = np.stack(resized, axis=0)
        eff_fps = min(meta.fps, TARGET_FPS) if meta.fps > TARGET_FPS * 1.02 else meta.fps
        return torch.from_numpy(np.ascontiguousarray(stacked.transpose(0, 3, 1, 2))), eff_fps

    if fmt == "png_seq":
        candidate_dirs = (
            [clip_dir / meta.payload_file]
            if meta.payload_file
            else [clip_dir / "frames", clip_dir]
        )
        img_paths: list[Path] = []
        for cand_dir in candidate_dirs:
            if cand_dir.is_dir():
                found = list(cand_dir.glob("*.png")) + list(cand_dir.glob("*.jpg"))
                if found:
                    img_paths = sorted(
                        found,
                        key=lambda p: (
                            int("".join(ch for ch in p.stem if ch.isdigit()) or -1),
                            p.name,
                        ),
                    )
                    break
        if not img_paths:
            raise FileNotFoundError(f"No PNG/JPG frames found in {[str(d) for d in candidate_dirs]}")
        keep = resample_frame_indices(0, len(img_paths), meta.fps, TARGET_FPS)
        resized = []
        for idx in keep:
            bgr = cv2.imread(str(img_paths[idx]), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"Failed to read frame image: {img_paths[idx]}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            resized.append(
                crop_and_resize_rgb(
                    rgb,
                    already_cropped=meta.already_cropped,
                    roi_base_1080p=meta.roi_base_1080p,
                    out_width=INPUT_WIDTH,
                    out_height=INPUT_HEIGHT,
                )
            )
        stacked = np.stack(resized, axis=0)
        eff_fps = min(meta.fps, TARGET_FPS) if meta.fps > TARGET_FPS * 1.02 else meta.fps
        return torch.from_numpy(np.ascontiguousarray(stacked.transpose(0, 3, 1, 2))), eff_fps

    if fmt == "video":
        video_path = _resolve_payload_path(
            clip_dir, meta.payload_file, ("clip.mp4", "frames.mp4", "clip.mkv", "clip.avi")
        )
        start_f = meta.source_start_frame if meta.source_start_frame is not None else 0
        end_f = meta.source_end_frame if meta.source_end_frame is not None else (start_f + meta.num_frames)
        tensor, src_fps, sampled = read_video_segment_rgb(
            video_path,
            start_frame=start_f,
            end_frame=end_f,
            target_fps=TARGET_FPS,
            already_cropped=meta.already_cropped,
            roi_base_1080p=meta.roi_base_1080p,
            out_width=INPUT_WIDTH,
            out_height=INPUT_HEIGHT,
        )
        eff_fps = min(src_fps, TARGET_FPS) if src_fps > TARGET_FPS * 1.02 else src_fps
        if len(sampled) < 2:
            raise RuntimeError(f"Video clip {clip_dir.name} decoded < 2 frames")
        return tensor, eff_fps

    raise ValueError(f"Unsupported frame_format={fmt!r}")


def _prepare_one_clip(
    clip_dir: Path,
    out_clips_dir: Path,
    *,
    source_tag: str,
    overwrite: bool,
) -> dict[str, Any]:
    meta_path = clip_dir / "meta.json"
    meta = ClipMeta.from_json(meta_path)
    safe_id = f"{source_tag}__{meta.clip_id}"
    out_pt = out_clips_dir / f"{safe_id}.pt"

    if out_pt.exists() and not overwrite:
        payload = torch.load(out_pt, map_location="cpu", weights_only=False)
        return {
            "clip_id": safe_id,
            "orig_clip_id": meta.clip_id,
            "source": meta.source or source_tag,
            "pt_file": f"clips/{out_pt.name}",
            "num_frames": int(payload["frames"].shape[0]),
            "fps": float(payload["fps"]),
            "end_round_frame": int(payload["events"]["end_round_frame"]),
            "metadata": meta.metadata,
        }

    frames_tchw, eff_fps = _load_raw_clip_frames(clip_dir, meta)
    num_out = int(frames_tchw.shape[0])

    state_labels, boundary_targets, mapped_events = build_frame_targets(
        num_frames=num_out,
        events=meta.events,
        src_fps=meta.fps,
        dst_fps=eff_fps,
        sigma=BOUNDARY_SIGMA_FRAMES,
    )

    record = {
        "clip_id": safe_id,
        "source": meta.source or source_tag,
        "fps": float(eff_fps),
        "frames": frames_tchw,  # uint8 (T, 3, H, W)
        "state_labels": torch.from_numpy(state_labels),         # int64 (T,)
        "boundary_targets": torch.from_numpy(boundary_targets), # float32 (T,)
        "events": mapped_events,
        "metadata": meta.metadata,
    }
    torch.save(record, out_pt)

    return {
        "clip_id": safe_id,
        "orig_clip_id": meta.clip_id,
        "source": meta.source or source_tag,
        "pt_file": f"clips/{out_pt.name}",
        "num_frames": num_out,
        "fps": float(eff_fps),
        "end_round_frame": int(mapped_events["end_round_frame"]),
        "metadata": meta.metadata,
    }


def discover_clip_dirs(
    raw_dir: Path = RAW_DIR,
    real_vod_dir: Path = REAL_VOD_DIR,
) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    for source_tag, root in (("mod", raw_dir), ("real_vod", real_vod_dir)):
        if not root.exists():
            continue
        for p in sorted(root.iterdir()):
            if p.is_dir() and (p / "meta.json").exists():
                found.append((source_tag, p))
    return found


def prepare_dataset(
    clean: bool = False,
    overwrite: bool = False,
    workers: int = 4,
    *,
    raw_dir: Path = RAW_DIR,
    real_vod_dir: Path = REAL_VOD_DIR,
    prepared_dir: Path = PREPARED_DIR,
) -> Path | None:
    clips = discover_clip_dirs(raw_dir=raw_dir, real_vod_dir=real_vod_dir)
    if not clips:
        print(
            "[Prepare] No clip directories found in "
            f"{raw_dir} or {real_vod_dir}. Run sync_dataset.py or add annotated clips first."
        )
        return None

    out_clips_dir = prepared_dir / "clips"
    if clean and prepared_dir.exists():
        print(f"[Prepare] Cleaning {prepared_dir}...")
        shutil.rmtree(prepared_dir)
    out_clips_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    failures: list[tuple[Path, str]] = []

    max_workers = max(1, min(workers, os.cpu_count() or 4))
    if max_workers == 1:
        for source_tag, clip_dir in tqdm(clips, desc="Preparing clips"):
            try:
                entries.append(
                    _prepare_one_clip(clip_dir, out_clips_dir, source_tag=source_tag, overwrite=overwrite)
                )
            except Exception as exc:
                failures.append((clip_dir, str(exc)))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                (clip_dir, pool.submit(_prepare_one_clip, clip_dir, out_clips_dir, source_tag=source_tag, overwrite=overwrite))
                for source_tag, clip_dir in clips
            ]
            for clip_dir, fut in tqdm(futures, desc="Preparing clips"):
                try:
                    entries.append(fut.result())
                except Exception as exc:
                    failures.append((clip_dir, str(exc)))

    entries.sort(key=lambda e: e["clip_id"])
    manifest_path = prepared_dir / "manifest.json"
    manifest = {
        "target_fps": TARGET_FPS,
        "input_width": INPUT_WIDTH,
        "input_height": INPUT_HEIGHT,
        "states": STATES,
        "ignore_index": IGNORE_INDEX,
        "num_clips": len(entries),
        "clips": entries,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    src_counts = Counter(e["source"] for e in entries)
    total_frames = sum(e["num_frames"] for e in entries)
    print(
        f"[Prepare Complete] {len(entries)} clips ({total_frames} frames) written to {prepared_dir}. "
        f"Sources: {dict(src_counts)}"
    )
    if failures:
        print(f"[Prepare Warnings] {len(failures)} clips failed:")
        for p, err in failures[:10]:
            print(f"  - {p}: {err}")
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Temporal End-Round Dataset")
    parser.add_argument("--clean", action="store_true", help="Rebuild dataset/prepared from scratch")
    parser.add_argument("--overwrite", action="store_true", help="Re-process existing .pt clips")
    parser.add_argument("--workers", type=int, default=4, help="Parallel clip preprocessing threads")
    args = parser.parse_args()
    prepare_dataset(clean=args.clean, overwrite=args.overwrite, workers=args.workers)
