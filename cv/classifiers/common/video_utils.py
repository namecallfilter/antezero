"""
Common Utilities and Video Extraction Helpers for Balatro Classifiers
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
from typing import Any, Callable, Iterable

import cv2
import numpy as np


def parse_timestamp(ts_str: str) -> float:
    """Parses timestamps like '04:00', '1:23:45', or '240.5' into float seconds."""
    ts_str = str(ts_str).strip()
    if ":" in ts_str:
        parts = [float(p) for p in ts_str.split(":")]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        elif len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        else:
            raise ValueError(f"Invalid timestamp format: {ts_str}")
    return float(ts_str)


def format_seconds(seconds: float) -> str:
    """Formats float seconds into HH:MM:SS.mmm string."""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"
    return f"{int(m):02d}:{s:06.3f}"


def extract_video_frame(video_path: Path, timestamp_sec: float) -> tuple[np.ndarray, float, int]:
    """Extracts a single BGR frame from a video at the closest timestamp."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame = int(round(timestamp_sec * fps))
    target_frame = max(0, min(total_frames - 1, target_frame))

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    actual_pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
    actual_sec = actual_pos_msec / 1000.0 if actual_pos_msec > 0 else (target_frame / fps)
    cap.release()

    if not ret or frame is None:
        raise RuntimeError(f"Failed to read frame at timestamp {timestamp_sec:.2f}s from {video_path}")

    return frame, actual_sec, target_frame


def _extract_video_frames_decord(
    video_path: Path | str,
    requested: list[int],
    *,
    batch_size: int = 16,
    progress_callback: Callable[[int], None] | None = None,
    max_workers: int = 4,
) -> dict[int, np.ndarray]:
    """Fast decord-backed implementation of `extract_video_frames`."""
    import decord

    vr: Any | None = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
    try:
        total_frames = len(vr)
        if total_frames <= 0:
            raise RuntimeError(f"Video has no frames: {video_path}")

        target_to_origs: dict[int, list[int]] = {}
        for orig_idx in requested:
            target = max(0, min(total_frames - 1, orig_idx))
            target_to_origs.setdefault(target, []).append(orig_idx)

        unique_targets = sorted(target_to_origs.keys())
        step = max(1, int(batch_size))
        cb_lock = threading.Lock()

        def _decode_targets(reader: Any, target_slice: list[int]) -> dict[int, np.ndarray]:
            local_frames: dict[int, np.ndarray] = {}
            for i in range(0, len(target_slice), step):
                chunk_targets = target_slice[i : i + step]
                batch_rgb = reader.get_batch(chunk_targets).asnumpy()
                chunk_count = 0
                for target, rgb in zip(chunk_targets, batch_rgb):
                    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
                    orig_list = target_to_origs[target]
                    for orig_idx in orig_list:
                        local_frames[orig_idx] = bgr
                    chunk_count += len(orig_list)
                if progress_callback is not None and chunk_count > 0:
                    with cb_lock:
                        progress_callback(chunk_count)
            return local_frames

        n_workers = min(max(1, int(max_workers)), len(unique_targets) // 16)
        if len(unique_targets) < 32 or n_workers <= 1:
            return _decode_targets(vr, unique_targets)

        # Release the probe reader before spawning parallel per-thread readers on Windows.
        del vr
        vr = None

        k, m = divmod(len(unique_targets), n_workers)
        slices = [
            unique_targets[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
            for i in range(n_workers)
        ]

        def _worker(target_slice: list[int]) -> dict[int, np.ndarray]:
            w_vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
            try:
                return _decode_targets(w_vr, target_slice)
            finally:
                del w_vr

        frames: dict[int, np.ndarray] = {}
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_worker, s) for s in slices if s]
            for fut in futures:
                frames.update(fut.result())
        return frames
    finally:
        # Release Windows file handle immediately so temp files can be cleaned up.
        if vr is not None:
            del vr


def extract_video_frames(
    video_path: Path | str,
    frame_indices: Iterable[int],
    sequential_grab_threshold: int = 240,
    progress_callback: Callable[[int], None] | None = None,
) -> dict[int, np.ndarray]:
    """Extracts specific BGR frames in one pass over a video file.

    Uses `cv2.VideoCapture` with `cap.grab()` for intra-GOP steps (matching `main.py`
    frame-for-frame; `decord.VideoReader` exhibits H.264 Open-GOP seeking drift on
    YouTube MP4s and is intentionally bypassed).
    Returns `{requested_frame_idx: bgr_ndarray}`.
    """
    requested = sorted({int(idx) for idx in frame_indices})
    if not requested:
        return {}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    frames: dict[int, np.ndarray] = {}
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cur_pos: int | None = None
        last_clamped: int | None = None
        last_frame: np.ndarray | None = None

        for orig_idx in requested:
            target = max(0, orig_idx)
            if total_frames > 0:
                target = min(total_frames - 1, target)

            if last_clamped == target and last_frame is not None:
                frames[orig_idx] = last_frame
                if progress_callback is not None:
                    progress_callback(1)
                continue

            if (
                cur_pos is None
                or target < cur_pos
                or (target - cur_pos) > sequential_grab_threshold
            ):
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                cur_pos = target
            else:
                while cur_pos < target:
                    if not cap.grab():
                        break
                    cur_pos += 1

            ret, frame = cap.read()
            cur_pos += 1
            if not ret or frame is None:
                raise RuntimeError(
                    f"Failed to read frame {target} (requested {orig_idx}) from {video_path}"
                )

            frames[orig_idx] = frame
            last_clamped = target
            last_frame = frame
            if progress_callback is not None:
                progress_callback(1)
    finally:
        cap.release()

    return frames


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """
    Computes Intersection-over-Union (IoU) between two bounding boxes.
    Boxes are in [x1, y1, x2, y2] format.
    """
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    area_a = max(0.0, (box_a[2] - box_a[0])) * max(0.0, (box_a[3] - box_a[1]))
    area_b = max(0.0, (box_b[2] - box_b[0])) * max(0.0, (box_b[3] - box_b[1]))

    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def scale_roi_pixels(roi: dict, frame_w: int, frame_h: int, base_w: float = 1920.0, base_h: float = 1080.0) -> tuple[int, int, int, int]:
    """Calculates pixel ROI coordinates scaled to the frame's actual resolution."""
    scale_x = frame_w / base_w
    scale_y = frame_h / base_h

    rx = int(round(roi["x"] * scale_x))
    ry = int(round(roi["y"] * scale_y))
    rw = int(round(roi["w"] * scale_x))
    rh = int(round(roi["h"] * scale_y))

    rx = max(0, min(frame_w - 1, rx))
    ry = max(0, min(frame_h - 1, ry))
    rw = min(frame_w - rx, rw)
    rh = min(frame_h - ry, rh)
    return rx, ry, rw, rh
