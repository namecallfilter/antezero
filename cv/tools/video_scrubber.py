"""
Shared Video Scrubber Engine for `tools/` Annotation Scripts
==========================================================
Provides:
  - `FrameRange`: bounded `[start_frame, end_frame]` trackbar <-> VOD frame math
  - `decode_scrubber_key`: normalizes `cv2.waitKeyEx` codes (ASCII + arrow keys)
  - `ScrubberVideoSession`: single-open `decord.VideoReader` with LRU exact-frame
    cache plus cancellable background thumbnail caching for fast trackbar scrubbing
  - `ScrubberController`: manages OpenCV `"Frame"` and `"Speed %"` trackbars,
    play/pause timing, scrub throttling, and clean window lifecycle
  - `draw_hud_panel` & `draw_segment_timeline`: crisp HUD + timeline markers
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import threading
import time
from typing import Iterable, Sequence

import cv2
import numpy as np

# Extended `cv2.waitKeyEx` scan codes across Windows, X11/GTK, and macOS backends.
_LEFT_ARROW_CODES = {2424832, 65361, 81, 2}
_RIGHT_ARROW_CODES = {2555904, 65363, 83, 3}
_UP_ARROW_CODES = {2490368, 65362, 82}
_DOWN_ARROW_CODES = {2621440, 65364, 84}


def decode_scrubber_key(raw_key: int) -> str:
    """Normalizes a `cv2.waitKeyEx` return value into a canonical key name.

    Returns `""` when no key was pressed, canonical names (`"enter"`, `"esc"`,
    `"space"`, `"left"`, `"right"`, `"up"`, `"down"`), or lowercase single
    ASCII characters (e.g. `"a"`, `"d"`, `","`, `"."`).
    """
    if raw_key == -1:
        return ""
    if raw_key in _LEFT_ARROW_CODES:
        return "left"
    if raw_key in _RIGHT_ARROW_CODES:
        return "right"
    if raw_key in _UP_ARROW_CODES:
        return "up"
    if raw_key in _DOWN_ARROW_CODES:
        return "down"

    low = raw_key & 0xFF
    if low in (10, 13):
        return "enter"
    if low == 27:
        return "esc"
    if low == 32:
        return "space"
    if 33 <= low <= 126:
        return chr(low).lower()
    return ""


def format_video_timestamp(frame_idx: int, fps: float) -> str:
    """Formats a 0-based frame index as `MM:SS.mmm` or `HH:MM:SS.mmm`."""
    safe_fps = fps if fps > 0 else 60.0
    total_ms = max(0, round((frame_idx / safe_fps) * 1000.0))
    total_sec, ms = divmod(total_ms, 1000)
    minutes, sec = divmod(total_sec, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}.{ms:03d}"
    return f"{minutes:02d}:{sec:02d}.{ms:03d}"


@dataclass(frozen=True, slots=True)
class FrameRange:
    """Inclusive `[start_frame, end_frame]` bounds mapped to `[0, trackbar_max]`."""

    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        if self.end_frame < self.start_frame:
            raise ValueError(
                f"FrameRange requires end_frame >= start_frame, got {self.start_frame}..{self.end_frame}"
            )

    @property
    def span(self) -> int:
        """Number of steps between `start_frame` and `end_frame` (`0` if single frame)."""
        return max(0, self.end_frame - self.start_frame)

    @property
    def num_frames(self) -> int:
        """Inclusive number of frames covered by this range."""
        return self.span + 1

    @property
    def trackbar_max(self) -> int:
        """OpenCV trackbars require max >= 1 even when `span == 0`."""
        return max(1, self.span)

    def clamp_abs(self, frame_idx: int) -> int:
        return max(self.start_frame, min(self.end_frame, int(frame_idx)))

    def to_rel(self, abs_frame: int) -> int:
        return self.clamp_abs(abs_frame) - self.start_frame

    def to_abs(self, rel_frame: int) -> int:
        return self.clamp_abs(self.start_frame + int(rel_frame))


class ScrubberVideoSession:
    """Manages one open `decord.VideoReader` with LRU exact cache & thumbnail worker."""

    def __init__(
        self,
        video_path: str,
        *,
        max_exact_frames: int = 180,
        thumb_width: int = 640,
        thumb_height: int = 360,
    ) -> None:
        from decord import VideoReader, cpu

        self.video_path = str(video_path)
        self._vr = VideoReader(self.video_path, ctx=cpu(0))
        self.total_frames = len(self._vr)
        if self.total_frames <= 0:
            raise RuntimeError(f"Video has no frames: {self.video_path}")
        raw_fps = float(self._vr.get_avg_fps() or 60.0)
        self.fps = raw_fps if raw_fps > 0 else 60.0

        first = self._vr[0].asnumpy()
        self.height, self.width = first.shape[:2]
        self._exact_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._exact_cache[0] = cv2.cvtColor(first, cv2.COLOR_RGB2BGR)
        self._max_exact_frames = max(8, int(max_exact_frames))

        self._thumb_width = thumb_width
        self._thumb_height = thumb_height
        self._vr_thumb = None
        self._thumb_cache: dict[int, np.ndarray] = {}
        self._thumb_stride = 1
        self._thumb_range: FrameRange | None = None
        self._thumb_lock = threading.Lock()
        self._thumb_cancel = threading.Event()
        self._thumb_thread: threading.Thread | None = None
        self.caching_progress = 0.0

    def _ensure_thumb_reader(self):
        if self._vr_thumb is None:
            from decord import VideoReader, cpu

            self._vr_thumb = VideoReader(
                self.video_path,
                ctx=cpu(0),
                width=self._thumb_width,
                height=self._thumb_height,
            )
        return self._vr_thumb

    def start_thumbnail_cache(
        self,
        *,
        frame_range: FrameRange | None = None,
        stride: int | None = None,
        max_samples: int = 280,
        chunk_size: int = 48,
    ) -> None:
        """Starts background thumbnail extraction for the given range (cancels prior)."""
        self.stop_thumbnail_cache()

        target_range = frame_range or FrameRange(0, self.total_frames - 1)
        s = max(0, target_range.start_frame)
        e = min(self.total_frames - 1, target_range.end_frame)
        if e < s:
            return

        if stride is None:
            computed_stride = max(1, (e - s + 1) // max(1, max_samples))
        else:
            computed_stride = max(1, int(stride))

        with self._thumb_lock:
            self._thumb_cache.clear()
            self._thumb_stride = computed_stride
            self._thumb_range = FrameRange(s, e)
            self.caching_progress = 0.0

        cancel_evt = threading.Event()
        self._thumb_cancel = cancel_evt

        def worker() -> None:
            try:
                vr_t = self._ensure_thumb_reader()
                indices = list(range(s, e + 1, computed_stride))
                if indices and indices[-1] != e:
                    indices.append(e)
                total = max(1, len(indices))
                for pos in range(0, len(indices), chunk_size):
                    if cancel_evt.is_set():
                        return
                    batch_idx = indices[pos : pos + chunk_size]
                    batch_rgb = vr_t.get_batch(batch_idx).asnumpy()
                    if cancel_evt.is_set():
                        return
                    bgr_frames = {
                        idx: cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        for idx, rgb in zip(batch_idx, batch_rgb)
                    }
                    with self._thumb_lock:
                        if cancel_evt.is_set():
                            return
                        self._thumb_cache.update(bgr_frames)
                        self.caching_progress = min(100.0, (pos + len(batch_idx)) / total * 100.0)
            except Exception as exc:
                if not cancel_evt.is_set():
                    print(f"[Scrubber] Thumbnail cache warning: {exc}")

        self._thumb_thread = threading.Thread(target=worker, daemon=True)
        self._thumb_thread.start()

    def stop_thumbnail_cache(self) -> None:
        self._thumb_cancel.set()
        if self._thumb_thread is not None and self._thumb_thread.is_alive():
            self._thumb_thread.join(timeout=0.5)
        self._thumb_thread = None

    def get_exact_frame(self, frame_idx: int) -> np.ndarray:
        """Returns a native-resolution BGR frame using an in-memory LRU cache."""
        clamped = max(0, min(self.total_frames - 1, int(frame_idx)))
        cached = self._exact_cache.get(clamped)
        if cached is not None:
            self._exact_cache.move_to_end(clamped)
            return cached

        rgb = self._vr[clamped].asnumpy()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self._exact_cache[clamped] = bgr
        if len(self._exact_cache) > self._max_exact_frames:
            self._exact_cache.popitem(last=False)
        return bgr

    def get_display_frame(
        self,
        frame_idx: int,
        *,
        prefer_thumbnail: bool = False,
    ) -> tuple[np.ndarray, bool]:
        """Returns `(bgr_image, is_thumbnail)`. Falls back to exact frame cleanly."""
        clamped = max(0, min(self.total_frames - 1, int(frame_idx)))
        if prefer_thumbnail:
            # Prefer an already-cached exact frame over an upscaled thumbnail
            exact = self._exact_cache.get(clamped)
            if exact is not None:
                self._exact_cache.move_to_end(clamped)
                return exact, False

            with self._thumb_lock:
                if self._thumb_cache and self._thumb_range is not None:
                    rel = clamped - self._thumb_range.start_frame
                    nearest = self._thumb_range.start_frame + (
                        (max(0, rel) // self._thumb_stride) * self._thumb_stride
                    )
                    thumb = self._thumb_cache.get(nearest)
                    if thumb is None:
                        thumb = self._thumb_cache.get(self._thumb_range.end_frame)
                    if thumb is not None:
                        if thumb.shape[1] != self.width or thumb.shape[0] != self.height:
                            thumb = cv2.resize(thumb, (self.width, self.height))
                        return thumb, True

        return self.get_exact_frame(clamped), False

    def close(self) -> None:
        self.stop_thumbnail_cache()
        self._exact_cache.clear()
        with self._thumb_lock:
            self._thumb_cache.clear()
        self._vr = None
        self._vr_thumb = None


class ScrubberController:
    """Drives OpenCV trackbar synchronization, playback timing, and scrub state."""

    FRAME_TB = "Frame"
    SPEED_TB = "Speed %"

    def __init__(
        self,
        window_name: str,
        *,
        window_width: int = 1280,
        window_height: int = 720,
        initial_range: FrameRange | None = None,
        initial_speed_pct: int = 100,
        scrub_settle_sec: float = 0.12,
    ) -> None:
        self.window_name = window_name
        self.frame_range = initial_range or FrameRange(0, 1)
        self.frame_idx = self.frame_range.start_frame
        self._frame_float = float(self.frame_idx)
        self.playing = False
        self._last_tick = time.time()
        self._last_scrub_time = 0.0
        self._scrub_settle_sec = scrub_settle_sec
        self._suppress_tb_cb = False
        self._last_tb_pos = 0

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, window_width, window_height)
        cv2.createTrackbar(
            self.FRAME_TB,
            self.window_name,
            0,
            self.frame_range.trackbar_max,
            self._on_frame_trackbar,
        )
        cv2.createTrackbar(
            self.SPEED_TB,
            self.window_name,
            initial_speed_pct,
            400,
            lambda _val: None,
        )

    def _on_frame_trackbar(self, rel_val: int) -> None:
        if self._suppress_tb_cb:
            return
        self.set_frame(self.frame_range.to_abs(rel_val), update_trackbar=False, is_scrub=True)

    def set_range(self, frame_range: FrameRange, *, initial_frame: int | None = None) -> None:
        """Switches the bounded `[start_frame, end_frame]` window cleanly."""
        self.playing = False
        self.frame_range = frame_range
        target = frame_range.start_frame if initial_frame is None else initial_frame
        self._suppress_tb_cb = True
        try:
            cv2.setTrackbarMax(self.FRAME_TB, self.window_name, frame_range.trackbar_max)
        finally:
            self._suppress_tb_cb = False
        self.set_frame(target, update_trackbar=True, is_scrub=False)
        self._last_scrub_time = 0.0

    def set_frame(
        self,
        abs_frame: int,
        *,
        update_trackbar: bool = True,
        is_scrub: bool = False,
    ) -> None:
        clamped = self.frame_range.clamp_abs(abs_frame)
        self.frame_idx = clamped
        self._frame_float = float(clamped)
        rel = self.frame_range.to_rel(clamped)
        self._last_tb_pos = rel
        self._last_scrub_time = time.time() if is_scrub else 0.0

        if update_trackbar:
            self._suppress_tb_cb = True
            try:
                cv2.setTrackbarPos(self.FRAME_TB, self.window_name, rel)
            finally:
                self._suppress_tb_cb = False

    def step_frames(self, delta: int, *, scrub_hint: bool = False) -> None:
        self.playing = False
        self.set_frame(self.frame_idx + delta, update_trackbar=True, is_scrub=scrub_hint)

    def toggle_play(self) -> None:
        if not self.playing and self.frame_idx >= self.frame_range.end_frame:
            self.set_frame(self.frame_range.start_frame, update_trackbar=True)
        self.playing = not self.playing
        self._last_tick = time.time()

    def advance_playback(self, fps: float) -> bool:
        """Advances playback or polls trackbar state. Returns `prefer_thumbnail` hint."""
        now = time.time()
        if self.playing:
            dt = max(0.0, now - self._last_tick)
            self._last_tick = now

            speed_pct = max(10, cv2.getTrackbarPos(self.SPEED_TB, self.window_name))
            speed_mul = speed_pct / 100.0
            self._frame_float += fps * dt * speed_mul
            if self._frame_float >= self.frame_range.end_frame:
                self._frame_float = float(self.frame_range.end_frame)
                self.playing = False
            self.frame_idx = self.frame_range.clamp_abs(int(self._frame_float))
            rel = self.frame_range.to_rel(self.frame_idx)
            if rel != self._last_tb_pos:
                self._suppress_tb_cb = True
                try:
                    cv2.setTrackbarPos(self.FRAME_TB, self.window_name, rel)
                finally:
                    self._suppress_tb_cb = False
                self._last_tb_pos = rel
            return (dt * fps * speed_mul) > 2.0

        # Handle trackbar drag edge cases where HighGUI didn't deliver a trailing callback
        tb_rel = cv2.getTrackbarPos(self.FRAME_TB, self.window_name)
        if tb_rel >= 0 and tb_rel != self._last_tb_pos and tb_rel <= self.frame_range.span:
            self.set_frame(self.frame_range.to_abs(tb_rel), update_trackbar=False, is_scrub=True)
            return True

        return (now - self._last_scrub_time) < self._scrub_settle_sec

    def is_window_open(self) -> bool:
        try:
            return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False


def draw_hud_panel(
    canvas: np.ndarray,
    lines: Sequence[tuple[str, tuple[int, int, int]]],
    *,
    origin: tuple[int, int] = (20, 18),
    font_scale: float = 0.65,
    line_height: int = 28,
    padding: int = 12,
    alpha: float = 0.72,
) -> None:
    """Renders a dark translucent HUD panel with crisp text rows onto `canvas`."""
    if not lines:
        return
    h, w = canvas.shape[:2]
    x0, y0 = origin
    box_h = padding * 2 + line_height * len(lines)
    max_text_w = max(
        cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0][0]
        for text, _color in lines
    )
    box_w = max_text_w + padding * 2
    x1 = min(w - 1, x0 + box_w)
    y1 = min(h - 1, y0 + box_h)
    x0 = max(0, x0)
    y0 = max(0, y0)
    if x1 <= x0 or y1 <= y0:
        return

    roi = canvas[y0:y1, x0:x1]
    overlay = np.zeros_like(roi)
    canvas[y0:y1, x0:x1] = cv2.addWeighted(roi, 1.0 - alpha, overlay, alpha, 0)

    for idx, (text, color) in enumerate(lines):
        ty = y0 + padding + (idx * line_height) + int(line_height * 0.72)
        tx = x0 + padding
        if ty < y1:
            cv2.putText(
                canvas,
                text,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                2,
                cv2.LINE_AA,
            )


def draw_segment_timeline(
    canvas: np.ndarray,
    frame_range: FrameRange,
    current_abs: int,
    *,
    markers: Iterable[tuple[int, tuple[int, int, int], str]] = (),
    bottom_margin: int = 24,
    side_margin: int = 60,
) -> None:
    """Draws a bounded horizontal progress bar with event ticks and playhead."""
    h, w = canvas.shape[:2]
    x0 = min(w - 20, max(10, side_margin))
    x1 = max(x0 + 20, w - side_margin)
    y = max(30, h - bottom_margin)
    bar_w = x1 - x0
    span = max(1, frame_range.span)

    def frame_to_x(abs_frame: int) -> int:
        clamped = frame_range.clamp_abs(abs_frame)
        frac = (clamped - frame_range.start_frame) / span
        return int(round(x0 + frac * bar_w))

    # Dark backing band for high contrast against gameplay visuals
    bg_y0 = max(0, y - 28)
    bg_y1 = min(h, y + 18)
    bg_x0 = max(0, x0 - 16)
    bg_x1 = min(w, x1 + 16)
    roi = canvas[bg_y0:bg_y1, bg_x0:bg_x1]
    canvas[bg_y0:bg_y1, bg_x0:bg_x1] = cv2.addWeighted(roi, 0.35, np.zeros_like(roi), 0.65, 0)

    cv2.line(canvas, (x0, y), (x1, y), (140, 140, 140), 4, cv2.LINE_AA)

    for m_frame, m_color, m_label in markers:
        if frame_range.start_frame <= m_frame <= frame_range.end_frame:
            mx = frame_to_x(m_frame)
            cv2.line(canvas, (mx, y - 10), (mx, y + 10), m_color, 2, cv2.LINE_AA)
            if m_label:
                cv2.putText(
                    canvas,
                    m_label,
                    (max(bg_x0 + 2, mx - 24), y - 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    m_color,
                    1,
                    cv2.LINE_AA,
                )

    px = frame_to_x(current_abs)
    cv2.line(canvas, (x0, y), (px, y), (0, 220, 255), 4, cv2.LINE_AA)
    cv2.circle(canvas, (px, y), 7, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, (px, y), 7, (0, 180, 255), 2, cv2.LINE_AA)
