from collections.abc import Iterator
from dataclasses import dataclass
from math import ceil
from pathlib import Path
import shutil
import subprocess

import cv2
from cv2.typing import MatLike
import numpy as np
from typing_extensions import Self

from geometry import ImageSize


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    path: Path
    resolution: ImageSize
    fps: float
    frame_count: int
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class VideoFrame:
    image: MatLike
    index: int
    timestamp_seconds: float


def _probe_video_codec(path: Path) -> str | None:
    """Queries the primary video stream codec via `ffprobe` when available."""
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        return None
    try:
        res = subprocess.run(
            [
                ffprobe_bin,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "default=nw=1:nk=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
        codec = res.stdout.strip().splitlines()[0].strip().lower() if res.stdout.strip() else None
        return codec or None
    except Exception:
        return None


class VideoReader(Iterator[VideoFrame]):
    """Iterate through a video while validating resolution and skipping frames."""

    def __init__(
        self,
        path: str | Path,
        expected_resolution: ImageSize,
        skip_frames: int = 0,
        *,
        start_frame: int | None = None,
        end_frame: int | None = None,
        start_timestamp_seconds: float | None = None,
        end_timestamp_seconds: float | None = None,
        use_hwaccel: bool = True,
    ) -> None:
        if skip_frames < 0:
            raise ValueError("skip_frames must be non-negative")
        if start_frame is not None and start_timestamp_seconds is not None:
            raise ValueError("Specify either start_frame or start_timestamp_seconds")
        if end_frame is not None and end_timestamp_seconds is not None:
            raise ValueError("Specify either end_frame or end_timestamp_seconds")
        if start_frame is not None and start_frame < 0:
            raise ValueError("start_frame must be non-negative")
        if end_frame is not None and end_frame < 0:
            raise ValueError("end_frame must be non-negative")
        if start_timestamp_seconds is not None and start_timestamp_seconds < 0:
            raise ValueError("start_timestamp_seconds must be non-negative")
        if end_timestamp_seconds is not None and end_timestamp_seconds < 0:
            raise ValueError("end_timestamp_seconds must be non-negative")

        self.path = Path(path)
        self.expected_resolution = expected_resolution
        self.skip_frames = skip_frames
        self.capture = cv2.VideoCapture(str(self.path))
        self._ffmpeg_proc: subprocess.Popen[bytes] | None = None
        self._nv12_bytes = 0
        self._frames_yielded = 0

        if not self.capture.isOpened():
            raise ValueError(f"Could not open video: {self.path}")

        width = round(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = round(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        actual_resolution = ImageSize(width=width, height=height)
        if actual_resolution != expected_resolution:
            self.close()
            raise ValueError(
                f"Video resolution is {width}x{height}; expected "
                f"{expected_resolution.width}x{expected_resolution.height}."
            )

        fps = self.capture.get(cv2.CAP_PROP_FPS)
        frame_count = round(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

        self.metadata = VideoMetadata(
            path=self.path,
            resolution=actual_resolution,
            fps=fps,
            frame_count=frame_count,
            duration_seconds=frame_count / fps if fps > 0 else 0.0,
        )

        if start_timestamp_seconds is not None:
            if fps <= 0:
                self.close()
                raise ValueError(
                    "Cannot use timestamp bounds when video FPS is unavailable"
                )
            start_frame = ceil(start_timestamp_seconds * fps)
        if end_timestamp_seconds is not None:
            if fps <= 0:
                self.close()
                raise ValueError(
                    "Cannot use timestamp bounds when video FPS is unavailable"
                )
            end_frame = ceil(end_timestamp_seconds * fps)

        if start_frame is None:
            start_frame = 0
        if end_frame is not None and end_frame < start_frame:
            self.close()
            raise ValueError("end_frame must be greater than or equal to start_frame")

        self._end_frame = end_frame
        self._next_index = start_frame

        step = self.skip_frames + 1
        expected_kept = (
            ceil((end_frame - start_frame) / step)
            if end_frame is not None
            else None
        )

        ffmpeg_bin = shutil.which("ffmpeg") if use_hwaccel else None
        can_use_nvdec = (
            ffmpeg_bin is not None
            and (expected_kept is None or expected_kept >= 16)
            and start_frame <= 3600
        )

        if can_use_nvdec:
            self._start_ffmpeg_nvdec(
                ffmpeg_bin=ffmpeg_bin,  # type: ignore[arg-type]
                width=width,
                height=height,
                start_frame=start_frame,
                step=step,
                expected_kept=expected_kept,
            )
            if self._ffmpeg_proc is not None:
                self.capture.release()
                return

        if start_frame:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    def _start_ffmpeg_nvdec(
        self,
        *,
        ffmpeg_bin: str,
        width: int,
        height: int,
        start_frame: int,
        step: int,
        expected_kept: int | None,
    ) -> None:
        codec = _probe_video_codec(self.path)
        cmd: list[str] = [ffmpeg_bin, "-loglevel", "error", "-hwaccel", "cuda"]
        if codec == "av1":
            # FFmpeg defaults AV1 to software libdav1d unless `-c:v av1` is specified.
            cmd.extend(["-c:v", "av1"])
        cmd.extend(["-i", str(self.path)])

        if start_frame == 0 and step > 1:
            cmd.extend(["-vf", f"select=not(mod(n\\,{step}))", "-vsync", "0"])
        elif start_frame > 0:
            if step == 1:
                cmd.extend(["-vf", f"select=gte(n\\,{start_frame})", "-vsync", "0"])
            else:
                cmd.extend(
                    [
                        "-vf",
                        f"select=gte(n\\,{start_frame})*not(mod(n-{start_frame}\\,{step}))",
                        "-vsync",
                        "0",
                    ]
                )

        if expected_kept is not None:
            cmd.extend(["-frames:v", str(expected_kept)])

        cmd.extend(["-f", "rawvideo", "-pix_fmt", "nv12", "pipe:1"])
        self._nv12_bytes = (width * height * 3) // 2
        self._ffmpeg_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=width * height * 6,
        )

    def _stop_ffmpeg(self) -> None:
        proc = self._ffmpeg_proc
        self._ffmpeg_proc = None
        if proc is None:
            return
        try:
            if proc.stdout is not None:
                proc.stdout.close()
        except Exception:
            pass
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=1.0)
        except Exception:
            pass

    def _next_cv2(self) -> VideoFrame:
        if self._end_frame is not None and self._next_index >= self._end_frame:
            self.close()
            raise StopIteration

        success, image = self.capture.read()
        if not success:
            self.close()
            raise StopIteration

        frame_index = self._next_index
        timestamp_seconds = self.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000

        self._next_index += 1

        for _ in range(self.skip_frames):
            if not self.capture.grab():
                break
            self._next_index += 1

        self._frames_yielded += 1
        return VideoFrame(
            image=image,
            index=frame_index,
            timestamp_seconds=timestamp_seconds,
        )

    def __iter__(self) -> "VideoReader":
        return self

    def __next__(self) -> VideoFrame:
        if self._ffmpeg_proc is None:
            return self._next_cv2()

        if self._end_frame is not None and self._next_index >= self._end_frame:
            self.close()
            raise StopIteration

        stdout = self._ffmpeg_proc.stdout
        buf = stdout.read(self._nv12_bytes) if stdout is not None else b""
        if len(buf) < self._nv12_bytes:
            if self._frames_yielded == 0:
                # Fall back to OpenCV if NVDEC failed before yielding any frame.
                self._stop_ffmpeg()
                self.capture = cv2.VideoCapture(str(self.path))
                if not self.capture.isOpened():
                    raise StopIteration
                if self._next_index:
                    self.capture.set(cv2.CAP_PROP_POS_FRAMES, self._next_index)
                return self._next_cv2()
            self.close()
            raise StopIteration

        w = self.metadata.resolution.width
        h = self.metadata.resolution.height
        nv12 = np.frombuffer(buf, dtype=np.uint8).reshape(((h * 3) // 2, w))
        image = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)

        frame_index = self._next_index
        fps = self.metadata.fps
        timestamp_seconds = frame_index / fps if fps > 0 else 0.0
        self._next_index += self.skip_frames + 1
        self._frames_yielded += 1
        return VideoFrame(
            image=image,
            index=frame_index,
            timestamp_seconds=timestamp_seconds,
        )

    def close(self) -> None:
        self._stop_ffmpeg()
        if self.capture.isOpened():
            self.capture.release()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def iter_video_frames(
    path: str | Path,
    expected_resolution: ImageSize,
    skip_frames: int = 0,
    *,
    start_frame: int | None = None,
    end_frame: int | None = None,
    start_timestamp_seconds: float | None = None,
    end_timestamp_seconds: float | None = None,
    use_hwaccel: bool = True,
) -> Iterator[VideoFrame]:
    """Yield a bounded range of video frames and always release the capture.

    Frame endpoints are zero-based and the end endpoint is exclusive. Timestamp
    endpoints are measured in seconds and are converted to the first frame at
    or after each timestamp.
    """
    with VideoReader(
        path,
        expected_resolution,
        skip_frames,
        start_frame=start_frame,
        end_frame=end_frame,
        start_timestamp_seconds=start_timestamp_seconds,
        end_timestamp_seconds=end_timestamp_seconds,
        use_hwaccel=use_hwaccel,
    ) as reader:
        yield from reader
