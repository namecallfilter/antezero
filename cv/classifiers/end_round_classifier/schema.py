"""
Dataset Contract & Metadata Schema for Balatro Mod / Real-VOD Hook-In
=====================================================================
Defines the standardized `meta.json` clip specification expected by `sync_dataset.py`
and `prepare_dataset.py`, plus label generation helpers for both full mod clips
and partially-annotated real VOD clips.

Future Balatro mod exports should place one folder per clip under:
    %APPDATA%/Balatro/Mods/gamestate_dataset_cv/dataset_end_round/<clip_id>/
containing:
  - `meta.json` (matching `ClipMeta.to_dict()`)
  - frame payload in one of:
      1. `raw_rgb_u8` / `raw_rgba_u8`: packed binary file (`frames.bin`) of shape (T, H, W, 3|4) uint8
      2. `png_seq`: sorted `frame_0000.png` ... (or `.jpg`) files
      3. `video`: a clip video file (`clip.mp4`) or slice `[source_start_frame, source_end_frame]`
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .config import (
        BOUNDARY_SIGMA_FRAMES,
        IGNORE_INDEX,
        PLAY_AREA_ROI,
        SCHEMA_VERSION,
        STATE_TO_IDX,
        TARGET_FPS,
    )
except ImportError:
    from config import (
        BOUNDARY_SIGMA_FRAMES,
        IGNORE_INDEX,
        PLAY_AREA_ROI,
        SCHEMA_VERSION,
        STATE_TO_IDX,
        TARGET_FPS,
    )

VALID_FRAME_FORMATS = {"png_seq", "raw_rgb_u8", "raw_rgba_u8", "video"}


@dataclass(slots=True)
class ClipEvents:
    """Frame indices (0-based relative to the extracted clip at native clip FPS)."""

    end_round_frame: int
    play_hand_frame: int = 0
    first_card_in_play_frame: int | None = None
    draw_to_discard_frame: int | None = None
    cards_cleared_frame: int | None = None
    round_eval_enter_frame: int | None = None
    round_eval_settled_frame: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClipEvents":
        if "end_round_frame" not in data or data["end_round_frame"] is None:
            raise ValueError("ClipEvents requires 'end_round_frame'")
        return cls(
            end_round_frame=int(data["end_round_frame"]),
            play_hand_frame=int(data.get("play_hand_frame", 0) or 0),
            first_card_in_play_frame=_opt_int(data.get("first_card_in_play_frame")),
            draw_to_discard_frame=_opt_int(data.get("draw_to_discard_frame")),
            cards_cleared_frame=_opt_int(data.get("cards_cleared_frame")),
            round_eval_enter_frame=_opt_int(data.get("round_eval_enter_frame")),
            round_eval_settled_frame=_opt_int(data.get("round_eval_settled_frame")),
        )


@dataclass(slots=True)
class ClipMeta:
    """Top-level `meta.json` contract produced by the mod or VOD annotation tool."""

    clip_id: str
    fps: float
    num_frames: int
    width: int
    height: int
    frame_format: str
    events: ClipEvents
    schema_version: int = SCHEMA_VERSION
    source: str = "balatro_mod"  # e.g. "balatro_mod" or "real_vod"
    roi_base_1080p: dict[str, float] = field(default_factory=lambda: dict(PLAY_AREA_ROI))
    already_cropped: bool = True
    payload_file: str | None = None  # e.g. "frames.bin" or "clip.mp4" or external VOD path
    source_start_frame: int | None = None  # optional slice start when frame_format == "video"
    source_end_frame: int | None = None    # optional inclusive/exclusive slice end
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.clip_id:
            raise ValueError("clip_id must be non-empty")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.num_frames < 2:
            raise ValueError(f"num_frames must be >= 2, got {self.num_frames}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Invalid frame dimensions {self.width}x{self.height}")
        if self.frame_format not in VALID_FRAME_FORMATS:
            raise ValueError(
                f"Unsupported frame_format={self.frame_format!r}. Expected one of {sorted(VALID_FRAME_FORMATS)}"
            )
        if not (0 <= self.events.end_round_frame < self.num_frames):
            raise ValueError(
                f"end_round_frame ({self.events.end_round_frame}) out of bounds [0, {self.num_frames})"
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_json(cls, json_path: Path | str) -> "ClipMeta":
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        meta = cls(
            clip_id=str(raw["clip_id"]),
            fps=float(raw["fps"]),
            num_frames=int(raw["num_frames"]),
            width=int(raw["width"]),
            height=int(raw["height"]),
            frame_format=str(raw["frame_format"]),
            events=ClipEvents.from_dict(raw["events"]),
            schema_version=int(raw.get("schema_version", SCHEMA_VERSION)),
            source=str(raw.get("source", "balatro_mod")),
            roi_base_1080p=dict(raw.get("roi_base_1080p") or PLAY_AREA_ROI),
            already_cropped=bool(raw.get("already_cropped", True)),
            payload_file=raw.get("payload_file"),
            source_start_frame=_opt_int(raw.get("source_start_frame")),
            source_end_frame=_opt_int(raw.get("source_end_frame")),
            metadata=dict(raw.get("metadata") or {}),
        )
        meta.validate()
        return meta


def _opt_int(val: Any) -> int | None:
    if val is None:
        return None
    return int(val)


def resample_event_index(frame_idx: int | None, src_fps: float, dst_fps: float, dst_len: int) -> int | None:
    """Maps a 0-based frame index from `src_fps` to `dst_fps`, clamped to `[0, dst_len - 1]`."""
    if frame_idx is None:
        return None
    if dst_len <= 0:
        return 0
    if abs(src_fps - dst_fps) < 1e-6:
        return max(0, min(dst_len - 1, int(frame_idx)))
    mapped = round(frame_idx * (dst_fps / src_fps))
    return max(0, min(dst_len - 1, int(mapped)))


def build_frame_targets(
    num_frames: int,
    events: ClipEvents,
    *,
    src_fps: float = TARGET_FPS,
    dst_fps: float = TARGET_FPS,
    sigma: float = BOUNDARY_SIGMA_FRAMES,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | None]]:
    """Builds per-frame 4-state labels and Gaussian boundary heatmap for a clip.

    Handles both densely annotated mod clips (all boundaries known) and lightly
    annotated real-VOD clips (where only `end_round_frame` is mandatory; unknown
    `PRE_PLAY`/`ROUND_EVAL` transitions are masked with `IGNORE_INDEX`).
    """
    end_round = resample_event_index(events.end_round_frame, src_fps, dst_fps, num_frames)
    assert end_round is not None
    first_in_play = resample_event_index(events.first_card_in_play_frame, src_fps, dst_fps, num_frames)
    round_eval_enter = resample_event_index(events.round_eval_enter_frame, src_fps, dst_fps, num_frames)
    round_eval_settled = resample_event_index(events.round_eval_settled_frame, src_fps, dst_fps, num_frames)

    pre_play_idx = STATE_TO_IDX["PRE_PLAY"]
    scoring_idx = STATE_TO_IDX["SCORING"]
    post_play_idx = STATE_TO_IDX["POST_PLAY_EMPTY"]
    round_eval_idx = STATE_TO_IDX["ROUND_EVAL"]

    state_labels = np.full((num_frames,), IGNORE_INDEX, dtype=np.int64)

    # 1. Pre-end_round region: PRE_PLAY -> SCORING
    if first_in_play is not None:
        split_in_play = max(0, min(end_round, first_in_play))
        if split_in_play > 0:
            state_labels[:split_in_play] = pre_play_idx
        state_labels[split_in_play:end_round] = scoring_idx
    else:
        # Unknown PRE_PLAY arrival (e.g., manual real-VOD clip): ignore earliest ~0.5s,
        # supervise SCORING safely once cards have landed up to end_round.
        safe_scoring_start = min(end_round, max(1, round(0.5 * dst_fps)))
        if safe_scoring_start < end_round:
            state_labels[safe_scoring_start:end_round] = scoring_idx

    # 2. Post-end_round region: POST_PLAY_EMPTY -> ROUND_EVAL
    if round_eval_enter is not None:
        eval_start = max(end_round + 1, min(num_frames, round_eval_enter))
        state_labels[end_round:eval_start] = post_play_idx
        if eval_start < num_frames:
            state_labels[eval_start:] = round_eval_idx
    else:
        # Unknown ROUND_EVAL transition: supervise immediate POST_PLAY_EMPTY (~0.4s)
        # and leave later frames as IGNORE_INDEX.
        safe_empty_end = min(num_frames, end_round + max(2, round(0.4 * dst_fps)))
        state_labels[end_round:safe_empty_end] = post_play_idx

    # 3. Gaussian boundary heatmap peaking at 1.0 on `end_round`
    t_idx = np.arange(num_frames, dtype=np.float32)
    denom = max(1e-6, 2.0 * (sigma ** 2))
    boundary_targets = np.exp(-((t_idx - float(end_round)) ** 2) / denom).astype(np.float32)

    mapped_events = {
        "first_card_in_play_frame": first_in_play,
        "end_round_frame": end_round,
        "round_eval_enter_frame": round_eval_enter,
        "round_eval_settled_frame": round_eval_settled,
    }
    return state_labels, boundary_targets, mapped_events
