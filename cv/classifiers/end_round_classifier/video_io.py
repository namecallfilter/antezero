"""
Video Segment & Frame Payload I/O Helpers
=========================================
Shared sequential video-segment decoder, ROI cropper, and clip payload loader
used by both `prepare_dataset.py` and `pipeline.py` (inference).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading
from typing import Any, Callable, Sequence

import cv2
import numpy as np
import torch

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# pyrefly: ignore [missing-import]
from classifiers.common.video_utils import scale_roi_pixels

try:
    from .config import BASE_HEIGHT, BASE_WIDTH, INPUT_HEIGHT, INPUT_WIDTH, PLAY_AREA_ROI, TARGET_FPS
except ImportError:
    from config import BASE_HEIGHT, BASE_WIDTH, INPUT_HEIGHT, INPUT_WIDTH, PLAY_AREA_ROI, TARGET_FPS


def crop_and_resize_rgb(
    frame_rgb: np.ndarray,
    *,
    already_cropped: bool = False,
    roi_base_1080p: dict[str, float] | None = None,
    out_width: int = INPUT_WIDTH,
    out_height: int = INPUT_HEIGHT,
) -> np.ndarray:
    """Crops `play_area` (if uncropped) and resizes an RGB uint8 frame to `(out_height, out_width, 3)`."""
    h, w = frame_rgb.shape[:2]
    if not already_cropped:
        roi = roi_base_1080p or PLAY_AREA_ROI
        rx, ry, rw, rh = scale_roi_pixels(roi, w, h, BASE_WIDTH, BASE_HEIGHT)
        if rw > 0 and rh > 0:
            frame_rgb = frame_rgb[ry : ry + rh, rx : rx + rw]

    if frame_rgb.shape[1] != out_width or frame_rgb.shape[0] != out_height:
        interp = cv2.INTER_AREA if (frame_rgb.shape[1] > out_width and frame_rgb.shape[0] > out_height) else cv2.INTER_LINEAR
        frame_rgb = cv2.resize(frame_rgb, (out_width, out_height), interpolation=interp)
    return np.ascontiguousarray(frame_rgb, dtype=np.uint8)


def resample_frame_indices(
    start_frame: int,
    end_frame_exclusive: int,
    src_fps: float,
    target_fps: float,
) -> list[int]:
    """Computes monotonically increasing source frame indices sampled near `target_fps`."""
    span = max(0, end_frame_exclusive - start_frame)
    if span == 0:
        return []
    if src_fps <= 0 or target_fps <= 0 or src_fps <= target_fps * 1.02:
        return list(range(start_frame, end_frame_exclusive))

    step = src_fps / target_fps
    n_out = max(2, int(round(span / step)))
    indices: list[int] = []
    last_idx = -1
    for i in range(n_out):
        idx = int(round(start_frame + i * step))
        idx = max(start_frame, min(end_frame_exclusive - 1, idx))
        if idx > last_idx:
            indices.append(idx)
            last_idx = idx
    return indices


def read_video_segment_rgb(
    video_path: Path | str,
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
    target_fps: float = TARGET_FPS,
    already_cropped: bool = False,
    roi_base_1080p: dict[str, float] | None = None,
    out_width: int = INPUT_WIDTH,
    out_height: int = INPUT_HEIGHT,
) -> tuple[torch.Tensor, float, list[int]]:
    """Sequentially decodes `[start_frame, end_frame)` at `target_fps` into uint8 `(T, 3, H, W)` RGB tensor.

    Returns `(frames_tchw, src_fps, sampled_source_frame_indices)`.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 60.0)
        if src_fps <= 0:
            src_fps = 60.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        s_frame = max(0, int(start_frame))
        e_frame = int(end_frame) if end_frame is not None else total_frames
        if total_frames > 0:
            e_frame = min(total_frames, e_frame)
        if e_frame <= s_frame:
            raise ValueError(f"Empty video segment [{s_frame}, {e_frame}) for {video_path}")

        sampled = resample_frame_indices(s_frame, e_frame, src_fps, target_fps)
        if not sampled:
            raise ValueError(f"No frames sampled from [{s_frame}, {e_frame}) in {video_path}")

        if s_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, s_frame)
        cur_pos = s_frame

        out_frames: list[np.ndarray] = []
        actual_indices: list[int] = []
        for target_idx in sampled:
            while cur_pos < target_idx:
                if not cap.grab():
                    break
                cur_pos += 1
            if cur_pos != target_idx:
                break
            ret, bgr = cap.read()
            cur_pos += 1
            if not ret or bgr is None:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            cropped = crop_and_resize_rgb(
                rgb,
                already_cropped=already_cropped,
                roi_base_1080p=roi_base_1080p,
                out_width=out_width,
                out_height=out_height,
            )
            out_frames.append(cropped)
            actual_indices.append(target_idx)

        if not out_frames:
            raise RuntimeError(f"Failed to decode any frames from {video_path} [{s_frame}, {e_frame})")

        arr = np.stack(out_frames, axis=0)  # (T, H, W, 3)
        tensor = torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 3, 1, 2)))  # uint8 (T, 3, H, W)
        return tensor, src_fps, actual_indices
    finally:
        cap.release()


def _read_video_segments_rgb_batch_decord(
    video_path: Path | str,
    segments: Sequence[tuple[int, int]],
    *,
    target_fps: float = TARGET_FPS,
    already_cropped: bool = False,
    roi_base_1080p: dict[str, float] | None = None,
    out_width: int = INPUT_WIDTH,
    out_height: int = INPUT_HEIGHT,
    progress_callback: Callable[[int], None] | None = None,
    max_workers: int = 4,
) -> tuple[list[torch.Tensor], float, list[list[int]]]:
    """Fast decord-backed implementation of `read_video_segments_rgb_batch`."""
    import decord

    vr: Any | None = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
    try:
        src_fps = float(vr.get_avg_fps() or 60.0)
        if src_fps <= 0:
            src_fps = 60.0
        total_frames = len(vr)
        if total_frames <= 0:
            raise RuntimeError(f"Video has no frames: {video_path}")

        ordered_segs = sorted(enumerate(segments), key=lambda item: (int(item[1][0]), int(item[1][1])))
        tasks: list[tuple[int, list[int]]] = []
        for orig_idx, (raw_start, raw_end) in ordered_segs:
            s_frame = max(0, int(raw_start))
            e_frame = min(total_frames, int(raw_end))
            if e_frame <= s_frame:
                raise ValueError(f"Empty video segment [{s_frame}, {e_frame}) for {video_path}")

            sampled = resample_frame_indices(s_frame, e_frame, src_fps, target_fps)
            if not sampled:
                raise ValueError(f"No frames sampled from [{s_frame}, {e_frame}) in {video_path}")
            tasks.append((orig_idx, sampled))

        cb_lock = threading.Lock()

        def _decode_segment_group(
            reader: Any, group_tasks: Sequence[tuple[int, list[int]]]
        ) -> list[tuple[int, torch.Tensor, list[int]]]:
            results: list[tuple[int, torch.Tensor, list[int]]] = []
            for orig_idx, sampled in group_tasks:
                batch_rgb = reader.get_batch(sampled).asnumpy()
                out_frames = [
                    crop_and_resize_rgb(
                        rgb,
                        already_cropped=already_cropped,
                        roi_base_1080p=roi_base_1080p,
                        out_width=out_width,
                        out_height=out_height,
                    )
                    for rgb in batch_rgb
                ]
                arr = np.stack(out_frames, axis=0)
                tensor = torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 3, 1, 2)))
                results.append((orig_idx, tensor, sampled))
                if progress_callback is not None:
                    with cb_lock:
                        progress_callback(1)
            return results

        tensors_out: list[torch.Tensor | None] = [None] * len(segments)
        indices_out: list[list[int]] = [[] for _ in segments]

        n_workers = min(max(1, int(max_workers)), len(tasks))
        if len(tasks) < 4 or n_workers <= 1:
            decoded = _decode_segment_group(vr, tasks)
        else:
            del vr
            vr = None

            k, m = divmod(len(tasks), n_workers)
            slices = [
                tasks[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
                for i in range(n_workers)
            ]

            def _worker(group_tasks: list[tuple[int, list[int]]]) -> list[tuple[int, torch.Tensor, list[int]]]:
                w_vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
                try:
                    return _decode_segment_group(w_vr, group_tasks)
                finally:
                    del w_vr

            decoded = []
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = [pool.submit(_worker, s) for s in slices if s]
                for fut in futures:
                    decoded.extend(fut.result())

        for orig_idx, tensor, sampled in decoded:
            tensors_out[orig_idx] = tensor
            indices_out[orig_idx] = sampled

        return [t for t in tensors_out if t is not None], src_fps, indices_out
    finally:
        if vr is not None:
            del vr


def read_video_segments_rgb_batch(
    video_path: Path | str,
    segments: Sequence[tuple[int, int]],
    *,
    target_fps: float = TARGET_FPS,
    already_cropped: bool = False,
    roi_base_1080p: dict[str, float] | None = None,
    out_width: int = INPUT_WIDTH,
    out_height: int = INPUT_HEIGHT,
    progress_callback: Callable[[int], None] | None = None,
) -> tuple[list[torch.Tensor], float, list[list[int]]]:
    """Decode multiple `[start_frame, end_frame)` segments using OpenCV VideoCapture.

    Uses OpenCV directly because `decord.VideoReader` exhibits H.264 Open-GOP
    seeking drift on YouTube MP4s (`_read_video_segments_rgb_batch_decord` is
    kept defined for compatibility).
    """
    if not segments:
        return [], 60.0, []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 60.0)
        if src_fps <= 0:
            src_fps = 60.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        tensors_out: list[torch.Tensor | None] = [None] * len(segments)
        indices_out: list[list[int]] = [[] for _ in segments]

        ordered_segs = sorted(enumerate(segments), key=lambda item: (int(item[1][0]), int(item[1][1])))
        cur_pos = 0
        has_started = False

        for orig_idx, (raw_start, raw_end) in ordered_segs:
            s_frame = max(0, int(raw_start))
            e_frame = int(raw_end)
            if total_frames > 0:
                e_frame = min(total_frames, e_frame)
            if e_frame <= s_frame:
                raise ValueError(f"Empty video segment [{s_frame}, {e_frame}) for {video_path}")

            sampled = resample_frame_indices(s_frame, e_frame, src_fps, target_fps)
            if not sampled:
                raise ValueError(f"No frames sampled from [{s_frame}, {e_frame}) in {video_path}")

            out_frames: list[np.ndarray] = []
            actual_indices: list[int] = []
            for target_idx in sampled:
                if not has_started and target_idx > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                    cur_pos = target_idx
                    has_started = True
                elif target_idx < cur_pos or (target_idx - cur_pos) > 240:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                    cur_pos = target_idx
                    has_started = True
                else:
                    has_started = True
                    while cur_pos < target_idx:
                        if not cap.grab():
                            break
                        cur_pos += 1

                if cur_pos != target_idx:
                    break
                ret, bgr = cap.read()
                cur_pos += 1
                if not ret or bgr is None:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                cropped = crop_and_resize_rgb(
                    rgb,
                    already_cropped=already_cropped,
                    roi_base_1080p=roi_base_1080p,
                    out_width=out_width,
                    out_height=out_height,
                )
                out_frames.append(cropped)
                actual_indices.append(target_idx)

            if not out_frames:
                raise RuntimeError(f"Failed to decode any frames from {video_path} [{s_frame}, {e_frame})")

            arr = np.stack(out_frames, axis=0)
            tensors_out[orig_idx] = torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 3, 1, 2)))
            indices_out[orig_idx] = actual_indices
            if progress_callback is not None:
                progress_callback(1)

        return [t for t in tensors_out if t is not None], src_fps, indices_out
    finally:
        cap.release()


def frames_sequence_to_tensor(
    frames: Sequence[np.ndarray],
    *,
    is_bgr: bool = True,
    already_cropped: bool = False,
    roi_base_1080p: dict[str, float] | None = None,
    out_width: int = INPUT_WIDTH,
    out_height: int = INPUT_HEIGHT,
) -> torch.Tensor:
    """Converts an in-memory sequence of HWC uint8 frames into a `(T, 3, H, W)` RGB uint8 tensor."""
    processed: list[np.ndarray] = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if is_bgr else frame
        processed.append(
            crop_and_resize_rgb(
                rgb,
                already_cropped=already_cropped,
                roi_base_1080p=roi_base_1080p,
                out_width=out_width,
                out_height=out_height,
            )
        )
    arr = np.stack(processed, axis=0)
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 3, 1, 2)))
