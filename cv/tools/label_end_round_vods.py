"""
Batch Real-VOD End-Round Clip Discovery & Manual Scrubber Labeler
=================================================================
Scans all Balatro gameplay videos in a folder (default: `assets/sample_inputs/videos/`)
using the first-pass CV pipeline from `main.py`, pairs each `Cash Out` click with its
immediately preceding `Play Hand` click, and steps through every `[Play Hand, Cash Out]`
window in a bounded OpenCV/`decord` video scrubber.

Because the OpenCV trackbar spans strictly `[play_hand, cash_out]` (typically ~5-20s)
instead of the entire multi-hour VOD, 1px of mouse scrub moves smoothly at near frame-level
precision. Pressing `Enter` (or `M`) saves a `real_vod` clip specification in
`classifiers/end_round_classifier/dataset/real_vod/<clip_id>/meta.json` ready for
`classifiers.end_round_classifier.prepare_dataset` and automatically advances to the next clip.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any, Sequence

# Ensure repository root is importable when invoked as `py tools/label_end_round_vods.py`.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2

from classifiers.common.video_utils import scale_roi_pixels
from classifiers.end_round_classifier.config import PLAY_AREA_ROI, SCHEMA_VERSION
from classifiers.end_round_classifier.schema import ClipEvents, ClipMeta
from config import Config, configure_logging
from main import detect_video_actions
from post_processing import remove_duplicates
from template import template_loader
from tools.video_scrubber import (
    FrameRange,
    ScrubberController,
    ScrubberVideoSession,
    decode_scrubber_key,
    draw_hud_panel,
    draw_segment_timeline,
    format_video_timestamp,
)

DEFAULT_VIDEOS_DIR = REPO_ROOT / "assets" / "sample_inputs" / "videos"
DEFAULT_ACTIONS_CACHE_DIR = REPO_ROOT / "output" / "vod_actions"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "classifiers" / "end_round_classifier" / "dataset" / "real_vod"
)
DEFAULT_MANIFEST_NAME = "_annotation_manifest.json"
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}


@dataclass(frozen=True, slots=True)
class EndRoundCandidate:
    """One `[Play Hand, Cash Out]` clip candidate extracted from first-pass actions."""

    clip_id: str
    video_path: Path
    play_hand_frame: int
    cash_out_frame: int
    range_start: int
    range_end: int
    fps: float
    width: int
    height: int


def sanitize_clip_slug(text: str) -> str:
    """Normalizes a video filename stem into a filesystem-safe clip prefix."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")
    return cleaned or "vod"


def build_candidate_clip_id(
    video_stem: str, play_hand_frame: int, cash_out_frame: int
) -> str:
    slug = sanitize_clip_slug(video_stem)
    return f"{slug}_f{play_hand_frame:07d}_{cash_out_frame:07d}"


def discover_videos(
    videos_dir: Path | None = None,
    explicit_videos: Sequence[Path] | None = None,
) -> list[Path]:
    """Returns sorted unique video paths from explicit arguments or `videos_dir`."""
    found: list[Path] = []
    seen: set[Path] = set()

    if explicit_videos:
        for vp in explicit_videos:
            resolved = vp.resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"Video file not found: {vp}")
            if resolved not in seen:
                seen.add(resolved)
                found.append(resolved)
        return found

    root = (videos_dir or DEFAULT_VIDEOS_DIR).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Videos directory not found: {root}")

    candidates = sorted(
        (
            p
            for p in root.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        ),
        key=lambda p: p.name.lower(),
    )
    return [p.resolve() for p in candidates]


def probe_video_metadata(video_path: Path) -> tuple[float, int, int, int]:
    """Quickly queries `(fps, total_frames, width, height)` via OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 60.0)
        if fps <= 0:
            fps = 60.0
        total_frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920))
        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080))
        if total_frames <= 0:
            raise RuntimeError(f"Video reports zero frames: {video_path}")
        return fps, total_frames, width, height
    finally:
        cap.release()


def build_action_cache_fingerprint(
    video_path: Path,
    *,
    start_timestamp: float | None,
    end_timestamp: float | None,
    skip_frames: int,
    elem_conf_thresh: float,
) -> dict[str, Any]:
    stat = video_path.stat()
    return {
        "video_name": video_path.name,
        "video_size": int(stat.st_size),
        "video_mtime_ns": int(stat.st_mtime_ns),
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "skip_frames": int(skip_frames),
        "elem_conf_thresh": float(elem_conf_thresh),
    }


def _cache_matches_fingerprint(
    cached: dict[str, Any], expected: dict[str, Any]
) -> bool:
    if "key" not in cached or "frames" not in cached:
        return False
    meta = cached.get("meta")
    if not isinstance(meta, dict):
        return False
    for k, v in expected.items():
        if meta.get(k) != v:
            return False
    return True


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def load_or_scan_video_actions(
    video_path: Path,
    *,
    actions_cache_dir: Path,
    base_config: Config,
    templates: Sequence[Any] | None = None,
    explicit_actions_json: Path | None = None,
    reprocess: bool = False,
    skip_scan: bool = False,
    progress_label: str = "Processing video",
) -> dict[str, Any] | None:
    """Loads cached first-pass actions for `video_path`, or runs `detect_video_actions`."""
    if explicit_actions_json is not None:
        if not explicit_actions_json.exists():
            raise FileNotFoundError(
                f"Explicit actions JSON not found: {explicit_actions_json}"
            )
        return _load_json(explicit_actions_json)

    cache_path = actions_cache_dir / f"{video_path.stem}.json"
    expected_fp = build_action_cache_fingerprint(
        video_path,
        start_timestamp=base_config.start_timestamp,
        end_timestamp=base_config.end_timestamp,
        skip_frames=base_config.skip_frames,
        elem_conf_thresh=base_config.elem_conf_thresh,
    )

    if cache_path.exists() and not reprocess:
        try:
            cached = _load_json(cache_path)
            if skip_scan or _cache_matches_fingerprint(cached, expected_fp):
                print(
                    f"[Cache] {video_path.name}: loaded {len(cached.get('frames', {}))} raw detections from {cache_path}"
                )
                return cached
        except Exception as exc:
            print(f"[Cache] Warning reading {cache_path}: {exc}; rescanning.")

    if skip_scan:
        print(f"[Skip Scan] No cached actions for {video_path.name} at {cache_path}")
        return None

    scan_cfg = Config(
        template_json_path=base_config.template_json_path,
        template_img_dir=base_config.template_img_dir,
        input_img_path=base_config.input_img_path,
        input_video_path=str(video_path),
        output_dir=str(actions_cache_dir),
        resolution_width=base_config.resolution_width,
        resolution_height=base_config.resolution_height,
        matching_method=base_config.matching_method,
        elem_conf_thresh=base_config.elem_conf_thresh,
        expand_roi_factor=base_config.expand_roi_factor,
        shadow_region_height=base_config.shadow_region_height,
        shadow_threshold=base_config.shadow_threshold,
        skip_frames=base_config.skip_frames,
        start_timestamp=base_config.start_timestamp,
        end_timestamp=base_config.end_timestamp,
        debug_plot=False,
        setup_mode=False,
        run_benchmark=False,
    )
    print(f"[Pass 1] Scanning actions in {video_path.name}...")
    payload, _written = detect_video_actions(
        scan_cfg,
        write_output=True,
        output_json_path=str(cache_path),
        templates=templates,
        extra_metadata=expected_fp,
        progress_desc=progress_label,
    )
    return payload


def _extract_action_id(entry: int | dict[str, Any]) -> int:
    return int(entry["action"]) if isinstance(entry, dict) else int(entry)


def extract_play_hand_cash_out_candidates(
    actions_payload: dict[str, Any],
    *,
    video_path: Path,
    fps: float,
    total_frames: int,
    width: int,
    height: int,
    play_hand_label: str = "Play Hand",
    cash_out_label: str = "Cash Out",
    dedup_max_gap_frames: int | None = None,
    pad_before_sec: float = 0.0,
    pad_after_sec: float = 0.0,
    max_window_sec: float | None = None,
) -> list[EndRoundCandidate]:
    """Pairs each `Cash Out` click with the latest preceding `Play Hand` click.

    When multiple `Play Hand` clicks occur before a single `Cash Out` (multi-hand
    blinds), only the final blind-clearing `Play Hand` is paired. Each `Cash Out`
    consumes its pending `Play Hand` so two `Cash Out`s never pair to the same hand.
    """
    key_map = actions_payload.get("key") or {}
    if play_hand_label not in key_map or cash_out_label not in key_map:
        available = ", ".join(sorted(key_map.keys()))
        raise ValueError(
            f"Actions JSON missing {play_hand_label!r} or {cash_out_label!r}. Available: [{available}]"
        )

    play_hand_id = int(key_map[play_hand_label])
    cash_out_id = int(key_map[cash_out_label])
    raw_frames = actions_payload.get("frames") or {}
    if not raw_frames:
        return []

    gap = (
        int(dedup_max_gap_frames)
        if dedup_max_gap_frames is not None
        else max(15, round(0.35 * fps))
    )
    # Collapse contiguous multi-frame detections to the final click-release frame.
    cleaned = remove_duplicates({str(k): v for k, v in raw_frames.items()}, max_frame_gap=gap)
    ordered = sorted(cleaned.items(), key=lambda kv: int(kv[0]))

    pad_before = max(0, round(max(0.0, pad_before_sec) * fps))
    pad_after = max(0, round(max(0.0, pad_after_sec) * fps))
    max_frame_idx = max(0, total_frames - 1)

    candidates: list[EndRoundCandidate] = []
    seen_ids: set[str] = set()
    pending_play_hand: int | None = None

    for frame_str, entry in ordered:
        frame_idx = max(0, min(max_frame_idx, int(frame_str)))
        action_id = _extract_action_id(entry)

        if action_id == play_hand_id:
            pending_play_hand = frame_idx
        elif action_id == cash_out_id:
            if pending_play_hand is not None and frame_idx > pending_play_hand:
                duration_sec = (frame_idx - pending_play_hand) / max(1e-6, fps)
                if max_window_sec is None or duration_sec <= max_window_sec:
                    range_start = max(0, pending_play_hand - pad_before)
                    range_end = min(max_frame_idx, frame_idx + pad_after)
                    if range_end > range_start:
                        clip_id = build_candidate_clip_id(
                            video_path.stem, pending_play_hand, frame_idx
                        )
                        if clip_id in seen_ids:
                            suffix = 2
                            while f"{clip_id}_v{suffix}" in seen_ids:
                                suffix += 1
                            clip_id = f"{clip_id}_v{suffix}"
                        seen_ids.add(clip_id)

                        candidates.append(
                            EndRoundCandidate(
                                clip_id=clip_id,
                                video_path=video_path,
                                play_hand_frame=pending_play_hand,
                                cash_out_frame=frame_idx,
                                range_start=range_start,
                                range_end=range_end,
                                fps=fps,
                                width=width,
                                height=height,
                            )
                        )
            # Clear even if unmatched/skipped so no stale Play Hand carries across rounds.
            pending_play_hand = None

    return candidates


def compute_relative_payload_path(video_path: Path, clip_dir: Path) -> str:
    """Computes a portable POSIX relative path from `clip_dir` to `video_path`."""
    try:
        rel = os.path.relpath(video_path.resolve(), clip_dir.resolve())
        return Path(rel).as_posix()
    except ValueError:
        return str(video_path.resolve())


def save_real_vod_clip(
    candidate: EndRoundCandidate,
    *,
    marked_abs_frame: int,
    output_dir: Path,
) -> Path:
    """Validates and writes `<output_dir>/<clip_id>/meta.json` in `real_vod` schema."""
    if not (candidate.range_start <= marked_abs_frame <= candidate.range_end):
        raise ValueError(
            f"Marked frame {marked_abs_frame} outside candidate bounds "
            f"[{candidate.range_start}, {candidate.range_end}]"
        )

    clip_dir = output_dir / candidate.clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    payload_ref = compute_relative_payload_path(candidate.video_path, clip_dir)

    start_f = candidate.range_start
    end_f_exclusive = candidate.range_end + 1
    num_frames = end_f_exclusive - start_f

    meta = ClipMeta(
        schema_version=SCHEMA_VERSION,
        clip_id=candidate.clip_id,
        source="real_vod",
        fps=float(candidate.fps),
        num_frames=num_frames,
        width=int(candidate.width),
        height=int(candidate.height),
        frame_format="video",
        already_cropped=False,
        roi_base_1080p=dict(PLAY_AREA_ROI),
        payload_file=payload_ref,
        source_start_frame=start_f,
        source_end_frame=end_f_exclusive,
        events=ClipEvents(
            play_hand_frame=candidate.play_hand_frame - start_f,
            end_round_frame=marked_abs_frame - start_f,
        ),
        metadata={
            "video_filename": candidate.video_path.name,
            "abs_play_hand_frame": candidate.play_hand_frame,
            "abs_cash_out_frame": candidate.cash_out_frame,
            "abs_end_round_frame": marked_abs_frame,
            "play_hand_timestamp_sec": round(candidate.play_hand_frame / candidate.fps, 4),
            "cash_out_timestamp_sec": round(candidate.cash_out_frame / candidate.fps, 4),
            "end_round_timestamp_sec": round(marked_abs_frame / candidate.fps, 4),
        },
    )
    meta.validate()

    meta_path = clip_dir / "meta.json"
    _atomic_write_json(meta_path, meta.to_dict())
    return meta_path


class AnnotationManifest:
    """Tracks labeled/skipped status per `clip_id` cleanly across resumable runs."""

    def __init__(self, manifest_path: Path, output_dir: Path) -> None:
        self.manifest_path = manifest_path
        self.output_dir = output_dir
        self.records: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if self.manifest_path.exists():
            data = _load_json(self.manifest_path)
            raw = data.get("clips")
            if isinstance(raw, dict):
                self.records = dict(raw)

        # Also recognize any clip folders already present in `output_dir`.
        if self.output_dir.exists():
            for p in self.output_dir.iterdir():
                meta_path = p / "meta.json"
                if p.is_dir() and meta_path.exists() and p.name not in self.records:
                    try:
                        meta = ClipMeta.from_json(meta_path)
                        abs_end = meta.metadata.get("abs_end_round_frame")
                        if abs_end is None:
                            start_f = meta.source_start_frame or 0
                            abs_end = start_f + meta.events.end_round_frame
                        self.records[p.name] = {
                            "status": "labeled",
                            "abs_end_round_frame": int(abs_end),
                            "meta_path": str(meta_path),
                        }
                    except Exception:
                        continue

    def save(self) -> None:
        _atomic_write_json(
            self.manifest_path,
            {"version": 1, "clips": self.records},
        )

    def status_of(self, clip_id: str) -> str:
        rec = self.records.get(clip_id)
        if not rec:
            return "pending"
        return str(rec.get("status", "pending"))

    def saved_frame_of(self, clip_id: str) -> int | None:
        rec = self.records.get(clip_id)
        if rec and rec.get("status") == "labeled" and rec.get("abs_end_round_frame") is not None:
            return int(rec["abs_end_round_frame"])
        return None

    def mark_labeled(
        self, candidate: EndRoundCandidate, marked_abs_frame: int, meta_path: Path
    ) -> None:
        self.records[candidate.clip_id] = {
            "status": "labeled",
            "video": candidate.video_path.name,
            "play_hand_frame": candidate.play_hand_frame,
            "cash_out_frame": candidate.cash_out_frame,
            "abs_end_round_frame": int(marked_abs_frame),
            "meta_path": str(meta_path),
            "updated_at": int(time.time()),
        }
        self.save()

    def mark_skipped(self, candidate: EndRoundCandidate) -> None:
        clip_dir = self.output_dir / candidate.clip_id
        if clip_dir.exists():
            shutil.rmtree(clip_dir, ignore_errors=True)
        self.records[candidate.clip_id] = {
            "status": "skipped",
            "video": candidate.video_path.name,
            "play_hand_frame": candidate.play_hand_frame,
            "cash_out_frame": candidate.cash_out_frame,
            "updated_at": int(time.time()),
        }
        self.save()


def filter_pending_candidates(
    candidates: Sequence[EndRoundCandidate],
    manifest: AnnotationManifest,
    *,
    review_all: bool = False,
) -> list[EndRoundCandidate]:
    if review_all:
        return list(candidates)
    return [c for c in candidates if manifest.status_of(c.clip_id) == "pending"]


def _draw_play_area_box(canvas: Any) -> None:
    h, w = canvas.shape[:2]
    rx, ry, rw, rh = scale_roi_pixels(PLAY_AREA_ROI, w, h, 1920.0, 1080.0)
    if rw > 0 and rh > 0:
        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (60, 220, 60), 1, cv2.LINE_AA)


def run_interactive_labeler(
    queue: Sequence[EndRoundCandidate],
    *,
    all_candidates_count: int,
    manifest: AnnotationManifest,
    output_dir: Path,
) -> tuple[int, int]:
    """Opens the bounded OpenCV/`decord` scrubber across `queue`.

    Returns `(session_labeled_count, session_skipped_count)`.
    """
    if not queue:
        print("[Done] No pending clips to label in this queue.")
        return 0, 0

    win_name = "End-Round Clip Labeler"
    scrubber = ScrubberController(
        win_name,
        window_width=1280,
        window_height=720,
        initial_range=FrameRange(queue[0].range_start, queue[0].range_end),
    )

    show_roi = True
    show_help = True
    session_labeled = 0
    session_skipped = 0

    active_video_path: Path | None = None
    session: ScrubberVideoSession | None = None
    pos = 0
    loaded_pos = -1

    try:
        while 0 <= pos < len(queue) and scrubber.is_window_open():
            cand = queue[pos]

            if pos != loaded_pos:
                if session is None or active_video_path != cand.video_path:
                    if session is not None:
                        session.close()
                    session = ScrubberVideoSession(
                        str(cand.video_path),
                        max_exact_frames=180,
                        thumb_width=640,
                        thumb_height=360,
                    )
                    active_video_path = cand.video_path

                seg_range = FrameRange(cand.range_start, cand.range_end)
                saved_abs = manifest.saved_frame_of(cand.clip_id)
                init_abs = saved_abs if saved_abs is not None else cand.play_hand_frame
                scrubber.set_range(seg_range, initial_frame=init_abs)
                session.start_thumbnail_cache(frame_range=seg_range, max_samples=260)
                loaded_pos = pos

            assert session is not None
            prefer_thumb = scrubber.advance_playback(cand.fps)
            frame, used_thumb = session.get_display_frame(
                scrubber.frame_idx, prefer_thumbnail=prefer_thumb
            )
            display = frame.copy()

            if show_roi:
                _draw_play_area_box(display)

            rel_idx = scrubber.frame_range.to_rel(scrubber.frame_idx)
            span = scrubber.frame_range.span
            rel_sec = rel_idx / max(1e-6, cand.fps)
            total_sec = span / max(1e-6, cand.fps)
            abs_ts = format_video_timestamp(scrubber.frame_idx, cand.fps)
            status_tag = manifest.status_of(cand.clip_id).upper()

            lines: list[tuple[str, tuple[int, int, int]]] = [
                (
                    f"Clip [{pos + 1}/{len(queue)}] (Total: {all_candidates_count}) | "
                    f"Video: {cand.video_path.name} | Status: {status_tag} | "
                    f"+{session_labeled} labeled / +{session_skipped} skipped",
                    (80, 255, 80),
                ),
                (
                    f"Rel: +{rel_idx}f ({rel_sec:.2f}s) / {span}f ({total_sec:.2f}s) | "
                    f"VOD Frame: {scrubber.frame_idx} ({abs_ts}) @ {cand.fps:.1f}fps"
                    + ("  [LONG >60s]" if total_sec > 60.0 else ""),
                    (0, 190, 255) if total_sec > 60.0 else (245, 245, 245),
                ),
            ]
            if show_help:
                lines.extend(
                    [
                        (
                            "ENTER/M: Mark EndRound & Next | SPACE: Play/Pause | "
                            "A/D or Arrows: -/+1f | ,/.: -/+0.25s | J/L: -/+2s",
                            (0, 245, 245),
                        ),
                        (
                            "R: Jump to PlayHand | S: Skip Clip | B: Prev Clip | "
                            "X: ROI Box | H: Toggle Help | Q/ESC: Quit",
                            (0, 245, 245),
                        ),
                    ]
                )

            draw_hud_panel(display, lines, origin=(18, 14), font_scale=0.58, line_height=26)

            markers: list[tuple[int, tuple[int, int, int], str]] = [
                (cand.play_hand_frame, (80, 255, 80), "PLAY"),
                (cand.cash_out_frame, (80, 80, 255), "CASH"),
            ]
            saved_abs = manifest.saved_frame_of(cand.clip_id)
            if saved_abs is not None:
                markers.append((saved_abs, (255, 100, 255), "SAVED"))

            draw_segment_timeline(
                display,
                scrubber.frame_range,
                scrubber.frame_idx,
                markers=markers,
            )

            if used_thumb:
                cv2.putText(
                    display,
                    "SCRUBBING",
                    (28, display.shape[0] - 52),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(win_name, display)
            key = decode_scrubber_key(cv2.waitKeyEx(1 if scrubber.playing else 25))

            if key == "space":
                scrubber.toggle_play()
            elif key in ("a", "left"):
                scrubber.step_frames(-1)
            elif key in ("d", "right"):
                scrubber.step_frames(1)
            elif key in (",", "["):
                scrubber.step_frames(-max(1, round(cand.fps * 0.25)), scrub_hint=True)
            elif key in (".", "]"):
                scrubber.step_frames(max(1, round(cand.fps * 0.25)), scrub_hint=True)
            elif key == "j":
                scrubber.step_frames(-max(1, round(cand.fps * 2.0)), scrub_hint=True)
            elif key == "l":
                scrubber.step_frames(max(1, round(cand.fps * 2.0)), scrub_hint=True)
            elif key == "r":
                scrubber.playing = False
                scrubber.set_frame(cand.play_hand_frame, update_trackbar=True)
            elif key == "x":
                show_roi = not show_roi
            elif key == "h":
                show_help = not show_help
            elif key in ("enter", "m"):
                scrubber.playing = False
                exact_frame = scrubber.frame_idx
                meta_path = save_real_vod_clip(
                    cand,
                    marked_abs_frame=exact_frame,
                    output_dir=output_dir,
                )
                manifest.mark_labeled(cand, exact_frame, meta_path)
                session_labeled += 1

                flash = session.get_exact_frame(exact_frame).copy()
                rel_marked = scrubber.frame_range.to_rel(exact_frame)
                hh, ww = flash.shape[:2]
                cv2.putText(
                    flash,
                    f"MARKED +{rel_marked}f",
                    (max(40, ww // 2 - 220), hh // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.6,
                    (0, 255, 0),
                    4,
                    cv2.LINE_AA,
                )
                cv2.imshow(win_name, flash)
                cv2.waitKey(110)
                pos += 1
            elif key == "s":
                scrubber.playing = False
                manifest.mark_skipped(cand)
                session_skipped += 1
                pos += 1
            elif key == "b":
                if pos > 0:
                    pos -= 1
            elif key in ("q", "esc"):
                break
    finally:
        if session is not None:
            session.close()
        cv2.destroyAllWindows()
        manifest.save()

    return session_labeled, session_skipped


def build_arg_parser() -> argparse.ArgumentParser:
    default_cfg = Config()
    parser = argparse.ArgumentParser(
        description="First-pass VOD action scan + bounded end-round clip scrubber labeler."
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=DEFAULT_VIDEOS_DIR,
        help=f"Directory of Balatro gameplay videos (default: {DEFAULT_VIDEOS_DIR})",
    )
    parser.add_argument(
        "--video",
        type=Path,
        nargs="+",
        default=None,
        help="Optional explicit video file(s) instead of scanning --videos-dir.",
    )
    parser.add_argument(
        "--actions-dir",
        type=Path,
        default=DEFAULT_ACTIONS_CACHE_DIR,
        help=f"Directory to cache per-video first-pass action JSONs (default: {DEFAULT_ACTIONS_CACHE_DIR})",
    )
    parser.add_argument(
        "--actions-json",
        type=Path,
        default=None,
        help="Optional precomputed main.py JSON path (only valid with a single --video).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Target real_vod dataset directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--template-json",
        type=str,
        default=default_cfg.template_json_path,
        help="Path to templates.json used for first-pass detection.",
    )
    parser.add_argument(
        "--template-dir",
        type=str,
        default=default_cfg.template_img_dir,
        help="Directory of template PNGs used for first-pass detection.",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=default_cfg.skip_frames,
        help="Frames skipped between template evaluations during Pass 1.",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=default_cfg.elem_conf_thresh,
        help="Template confidence threshold during Pass 1.",
    )
    parser.add_argument(
        "--start-timestamp",
        type=float,
        default=0.0,
        help="Optional scan start timestamp in seconds (default: 0.0).",
    )
    parser.add_argument(
        "--end-timestamp",
        type=float,
        default=None,
        help="Optional scan end timestamp in seconds (default: None = full video).",
    )
    parser.add_argument(
        "--pad-before-sec",
        type=float,
        default=0.0,
        help="Optional seconds before Play Hand included in scrubber range (default: 0.0).",
    )
    parser.add_argument(
        "--pad-after-sec",
        type=float,
        default=0.0,
        help="Optional seconds after Cash Out included in scrubber range (default: 0.0).",
    )
    parser.add_argument(
        "--max-window-sec",
        type=float,
        default=None,
        help="Optional cap on Play Hand -> Cash Out window duration in seconds.",
    )
    parser.add_argument(
        "--play-hand-label",
        type=str,
        default="Play Hand",
        help="Template label name for Play Hand clicks.",
    )
    parser.add_argument(
        "--cash-out-label",
        type=str,
        default="Cash Out",
        help="Template label name for Cash Out clicks.",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Force re-running first-pass CV detection even if cached JSON exists.",
    )
    parser.add_argument(
        "--skip-scan",
        action="store_true",
        help="Use existing cached action JSONs only; never run Pass 1 scan.",
    )
    parser.add_argument(
        "--review-all",
        action="store_true",
        help="Re-open previously labeled and skipped candidates too.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Run Pass 1 (if needed) / load caches, print candidate counts, and exit without GUI.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    args = build_arg_parser().parse_args(argv)

    videos = discover_videos(args.videos_dir, args.video)
    if not videos:
        print(f"No video files found in {args.videos_dir}.")
        return 1
    if args.actions_json is not None and len(videos) != 1:
        raise ValueError("--actions-json can only be used when specifying a single --video.")

    base_cfg = Config(
        template_json_path=args.template_json,
        template_img_dir=args.template_dir,
        output_dir=str(args.actions_dir),
        skip_frames=args.skip_frames,
        elem_conf_thresh=args.conf_thresh,
        start_timestamp=args.start_timestamp,
        end_timestamp=args.end_timestamp,
        run_benchmark=False,
    )

    manifest = AnnotationManifest(args.output_dir / DEFAULT_MANIFEST_NAME, args.output_dir)
    shared_templates = None
    if not args.skip_scan and args.actions_json is None:
        shared_templates = template_loader(
            base_cfg.template_json_path,
            base_cfg.template_img_dir,
            base_cfg.resolution,
        )

    all_candidates: list[EndRoundCandidate] = []
    n_videos = len(videos)
    for idx, vp in enumerate(videos, start=1):
        actions_payload = load_or_scan_video_actions(
            vp,
            actions_cache_dir=args.actions_dir,
            base_config=base_cfg,
            templates=shared_templates,
            explicit_actions_json=args.actions_json,
            reprocess=args.reprocess,
            skip_scan=args.skip_scan,
            progress_label=f"[{idx}/{n_videos}] {vp.name}",
        )
        if actions_payload is None:
            continue

        fps, total_frames, width, height = probe_video_metadata(vp)
        cands = extract_play_hand_cash_out_candidates(
            actions_payload,
            video_path=vp,
            fps=fps,
            total_frames=total_frames,
            width=width,
            height=height,
            play_hand_label=args.play_hand_label,
            cash_out_label=args.cash_out_label,
            pad_before_sec=args.pad_before_sec,
            pad_after_sec=args.pad_after_sec,
            max_window_sec=args.max_window_sec,
        )
        all_candidates.extend(cands)
        print(f"  -> {vp.name}: {len(cands)} Play Hand -> Cash Out candidate(s)")

    queue = filter_pending_candidates(
        all_candidates, manifest, review_all=args.review_all
    )
    labeled_n = sum(1 for c in all_candidates if manifest.status_of(c.clip_id) == "labeled")
    skipped_n = sum(1 for c in all_candidates if manifest.status_of(c.clip_id) == "skipped")
    pending_n = sum(1 for c in all_candidates if manifest.status_of(c.clip_id) == "pending")

    print(
        f"\nSummary: {len(videos)} video(s), {len(all_candidates)} total candidates "
        f"({labeled_n} labeled, {skipped_n} skipped, {pending_n} pending)."
    )
    print(f"Output dataset directory: {args.output_dir}")

    if args.stats:
        return 0

    added_labeled, added_skipped = run_interactive_labeler(
        queue,
        all_candidates_count=len(all_candidates),
        manifest=manifest,
        output_dir=args.output_dir,
    )
    print(
        f"Session finished: +{added_labeled} labeled, +{added_skipped} skipped. "
        f"Manifest saved to {manifest.manifest_path}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
