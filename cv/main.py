import argparse
import concurrent.futures
import itertools
import json
import logging
import math
import os
from pathlib import Path
import queue
import threading
import time
from typing import Any, Sequence

import cv2
from tqdm import tqdm

from config import Config, configure_logging
from tools.debug_plotter import plot
from eval_template import detect_view_deck_tab_mode, eval, format_timestamp
from frame import iter_video_frames
from post_processing import (
    build_run_trajectory,
    default_trajectory_output_path,
    parse_deck,
    parse_stake,
    serialize_trajectory_summary,
)
from screen_state import CoarseScreenState, ScreenStateTracker
from template import template_loader


def build_output_json_path(
    output_dir: str, input_video_path: str, timestamp: int | None = None
) -> str:
    """Builds the output JSON path from any Windows or POSIX video file path."""
    ts = int(time.time()) if timestamp is None else timestamp
    base_name = Path(input_video_path).stem
    return str(Path(output_dir) / f"{base_name}_{ts}.json")


def get_expected_frame_count(config: Config) -> int | None:
    cap = cv2.VideoCapture(config.input_video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps <= 0:
        return None

    start_frame = (
        math.ceil(config.start_timestamp * fps)
        if config.start_timestamp is not None
        else 0
    )
    end_frame = (
        math.ceil(config.end_timestamp * fps)
        if config.end_timestamp is not None
        else total_frames
    )
    end_frame = min(end_frame, total_frames)

    frame_span = max(0, end_frame - start_frame)
    step = config.skip_frames + 1
    return math.ceil(frame_span / step)


def _process_frame(
    video_frame: Any,
    active_templates: Sequence[Any],
    template_key: dict[str, int],
    config: Config,
    logger: logging.Logger,
) -> tuple[int, list, dict[str, float], dict | None, str | None]:
    frame_idx = video_frame.index
    need_ts = config.debug_plot or config.setup_mode
    ts_str = (
        format_timestamp(video_frame.timestamp_seconds, frame_idx) if need_ts else ""
    )

    if logger.isEnabledFor(logging.INFO):
        logger.info(f"Processing frame {frame_idx}...")

    t0 = time.perf_counter()
    img_gray = cv2.cvtColor(video_frame.image, cv2.COLOR_BGR2GRAY)
    img_gray_half = (
        cv2.resize(
            img_gray,
            (img_gray.shape[1] // 2, img_gray.shape[0] // 2),
            interpolation=cv2.INTER_NEAREST,
        )
        if config.coarse_to_fine and not need_ts
        else None
    )
    t1 = time.perf_counter()

    matches = []
    template_clicked = None
    clicked_name = None
    best_clicked_conf = -1.0
    cash_out_box: list[int] | None = None
    cash_out_present = False
    open_pack_present = False
    for template in active_templates:
        result = eval(
            template,
            video_frame.image,
            img_gray,
            config,
            video_timestamp_str=ts_str,
            img_gray_half=img_gray_half,
        )
        # When "Sell Card" sits directly above "Use Consumable", its 10px shadow box
        # collides with the top border of "Use"; re-evaluate only within the gap.
        eff_min_luma = (
            float(template.min_luminance_ratio)
            if getattr(template, "min_luminance_ratio", None) is not None
            else config.min_luminance_ratio
        )
        eff_max_luma = (
            float(template.max_luminance_ratio)
            if getattr(template, "max_luminance_ratio", None) is not None
            else config.max_luminance_ratio
        )
        luma_ratio = getattr(result, "luminance_ratio", None)
        luma_ok = (
            not config.use_luminance_check
            or luma_ratio is None
            or eff_min_luma <= luma_ratio <= eff_max_luma
        )
        if (
            result.found
            and not result.clicked
            and luma_ok
            and result.template_name == "Sell Card"
            and result.box is not None
        ):
            use_tpl = next(
                (t for t in active_templates if t.name == "Use Consumable"), None
            )
            if use_tpl is not None:
                use_res = eval(
                    use_tpl,
                    video_frame.image,
                    img_gray,
                    config,
                    video_timestamp_str=ts_str,
                    img_gray_half=img_gray_half,
                )
                if use_res.found and use_res.box is not None:
                    sx1, sy1, sx2, sy2 = (int(v) for v in result.box)
                    ux1, uy1, ux2, _ = (int(v) for v in use_res.box)
                    horiz_overlap = max(0, min(sx2, ux2) - max(sx1, ux1))
                    min_w = max(1, min(sx2 - sx1, ux2 - ux1))
                    if horiz_overlap >= 0.5 * min_w and (sy2 - 2) <= uy1 <= (
                        sy2 + config.shadow_region_height + 2
                    ):
                        y0 = max(0, sy2 + int(getattr(template, "shadow_dy", 0)))
                        y1 = min(img_gray.shape[0], uy1 - 1)
                        if y1 - y0 < 2:
                            sell_clicked = True
                        else:
                            from shadow_evaluator import analyze_shadow_presence

                            clipped = img_gray[
                                y0:y1, max(0, sx1) : min(img_gray.shape[1], sx2)
                            ]
                            sell_clicked = not analyze_shadow_presence(
                                clipped, threshold=config.shadow_threshold
                            )
                        from dataclasses import is_dataclass, replace as _dc_replace

                        if is_dataclass(result):
                            result = _dc_replace(result, clicked=sell_clicked)
                        else:
                            result.clicked = sell_clicked
        if (
            result.template_name == "Open Pack"
            and result.found
            and not result.clicked
        ):
            open_pack_present = True
        if (
            result.template_name == "Cash Out"
            and result.found
            and not result.clicked
            and result.box is not None
            and result.match is not None
            and int(result.box[1]) <= 425
        ):
            if result.match.confidence >= 0.85:
                cash_out_present = True
            if result.match.confidence >= 0.95:
                cash_out_box = [int(v) for v in result.box]
        if hasattr(result, "plot_kwargs") and result.found and result.plot_kwargs:
            matches.append((template, result.plot_kwargs))
        if result.found and result.clicked:
            cand_conf = (
                float(result.match.confidence) if result.match is not None else 1.0
            )
            if cand_conf > best_clicked_conf:
                best_clicked_conf = cand_conf
                clicked_name = result.template_name
                template_clicked = {
                    "action": template_key[result.template_name],
                    "box": list(result.box) if result.box is not None else None,
                }
                if (
                    result.template_name == "Full Deck"
                    and result.box is not None
                ):
                    template_clicked["metadata"] = {
                        "view_mode": detect_view_deck_tab_mode(
                            video_frame.image, result.box
                        )
                    }
    t2 = time.perf_counter()

    metrics: dict[str, Any] = {
        "grayscale_time": t1 - t0,
        "eval_time": t2 - t1,
    }
    if cash_out_present:
        metrics["cash_out_present"] = True
    if cash_out_box is not None:
        metrics["cash_out_box"] = cash_out_box
    if open_pack_present:
        metrics["open_pack_present"] = True
    return frame_idx, matches, metrics, template_clicked, clicked_name


def _prune_isolated_single_frame_clicks(
    action_frames: dict[int, dict[str, Any]],
    template_key: dict[str, int],
    *,
    template_names: tuple[str, ...],
    skip_frames: int,
) -> dict[int, dict[str, Any]]:
    """Removes isolated length-1 click blips (animation-blur spikes) for non-transition templates."""
    target_ids = {template_key[name] for name in template_names if name in template_key}
    if not target_ids or skip_frames > 2 or not action_frames:
        return action_frames

    max_gap = max(4, (skip_frames + 1) * 3)
    runs: list[list[tuple[int, dict[str, Any]]]] = []
    for frame_idx, entry in sorted(action_frames.items()):
        if (
            runs
            and runs[-1][-1][1].get("action") == entry.get("action")
            and (frame_idx - runs[-1][-1][0]) <= max_gap
        ):
            runs[-1].append((frame_idx, entry))
        else:
            runs.append([(frame_idx, entry)])

    kept: dict[int, dict[str, Any]] = {}
    for run in runs:
        if len(run) == 1 and run[0][1].get("action") in target_ids:
            continue
        for frame_idx, entry in run:
            kept[frame_idx] = entry
    return kept


def _chunked_iterable(iterable, size: int):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk


def _prefetch_generator(iterable, prefetch: int = 64):
    q = queue.Queue(maxsize=prefetch)
    sentinel = object()
    producer_exc: list[BaseException | None] = [None]

    def producer():
        try:
            for item in iterable:
                q.put(item)
        except BaseException as exc:
            producer_exc[0] = exc
        finally:
            q.put(sentinel)

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    while True:
        item = q.get()
        if item is sentinel:
            if producer_exc[0] is not None:
                raise producer_exc[0]
            break
        yield item


def detect_video_actions(
    config: Config,
    *,
    write_output: bool = True,
    output_json_path: str | None = None,
    templates: Sequence[Any] | None = None,
    extra_metadata: dict[str, Any] | None = None,
    progress_desc: str = "Processing video",
) -> tuple[dict[str, Any], str | None]:
    """Runs first-pass template action detection on `config.input_video_path`.

    Returns `(output_payload, written_json_path)` where `output_payload` always
    contains `"key"` and `"frames"` (plus `"meta"` when `extra_metadata` is supplied).
    """
    logger = logging.getLogger(__name__)
    loaded_templates = (
        list(templates)
        if templates is not None
        else template_loader(
            config.template_json_path,
            config.template_img_dir,
            config.resolution,
        )
    )
    template_key = {template.name: i for i, template in enumerate(loaded_templates)}
    # Starting mid-run is out of scope: screen gating begins in PRE_GAME.
    screen_tracker = ScreenStateTracker()

    raw_frames = iter_video_frames(
        config.input_video_path,
        config.resolution,
        config.skip_frames,
        start_timestamp_seconds=config.start_timestamp,
        end_timestamp_seconds=config.end_timestamp,
        use_hwaccel=config.use_hwaccel,
    )

    video_extract_times = []

    def timed_frames(it):
        while True:
            t0 = time.perf_counter()
            try:
                frame = next(it)
            except StopIteration:
                break
            t1 = time.perf_counter()
            if config.run_benchmark:
                video_extract_times.append(t1 - t0)
            yield frame

    prefetched_frames = _prefetch_generator(raw_frames, prefetch=128)
    frames = timed_frames(prefetched_frames)

    start = time.perf_counter()
    results = []
    total_grayscale_time = 0.0
    total_eval_time = 0.0

    num_workers = 1
    action_frames: dict[int, dict[str, Any]] = {}
    cash_out_visible_streak = 0
    cash_out_missing_streak = 0
    cash_out_first_missing_frame: int | None = None
    cash_out_last_box: list[int] | None = None
    open_pack_visible_streak = 0
    open_pack_missing_streak = 0
    blind_deck_tab_streak = 0
    blind_deck_tab_x_thresh = int(0.468 * config.resolution.width)
    shop_last_activity_frame: int | None = None
    shop_pack_fallback_frames = 45

    def _reset_cash_out_watch() -> None:
        nonlocal \
            cash_out_visible_streak, \
            cash_out_missing_streak, \
            cash_out_first_missing_frame
        cash_out_visible_streak = 0
        cash_out_missing_streak = 0
        cash_out_first_missing_frame = None

    def _reset_open_pack_watch() -> None:
        nonlocal open_pack_visible_streak, open_pack_missing_streak
        open_pack_visible_streak = 0
        open_pack_missing_streak = 0

    def _apply_frame_detection(
        frame_idx: int,
        template_clicked: dict[str, Any] | None,
        clicked_name: str | None,
        metrics: dict[str, Any],
    ) -> bool:
        """Records detected click (or synthetic Cash Out disappearance) and returns True if state changed."""
        nonlocal cash_out_visible_streak, cash_out_missing_streak
        nonlocal cash_out_first_missing_frame, cash_out_last_box
        nonlocal open_pack_visible_streak, open_pack_missing_streak
        nonlocal blind_deck_tab_streak
        nonlocal shop_last_activity_frame

        prev_state = (
            screen_tracker.state,
            screen_tracker.allow_play_game,
            screen_tracker.pre_pack_location,
            screen_tracker.last_cash_out_frame,
            screen_tracker.last_held_item_action_frame,
        )
        if template_clicked is not None and clicked_name is not None:
            action_frames[frame_idx] = template_clicked
            screen_tracker.on_template_clicked(clicked_name, frame_idx=frame_idx)
            if clicked_name == "Full Deck":
                fd_box = template_clicked.get("box")
                fd_view_mode = (template_clicked.get("metadata") or {}).get(
                    "view_mode"
                )
                if fd_view_mode == "remaining" or (
                    fd_box is not None and int(fd_box[0]) >= blind_deck_tab_x_thresh
                ):
                    blind_deck_tab_streak += 1
                    if blind_deck_tab_streak >= 3:
                        screen_tracker.on_blind_view_deck_observed()
                else:
                    blind_deck_tab_streak = 0
            else:
                blind_deck_tab_streak = 0
            _reset_open_pack_watch()
            if screen_tracker.state == CoarseScreenState.SHOP:
                shop_last_activity_frame = frame_idx
            else:
                shop_last_activity_frame = None
            if clicked_name in (
                "Open Run Info Menu",
                "Back (Run Info)",
                "Full Deck",
                "Open Options Menu",
                "Cash Out",
                "Play Hand",
                "Discard",
                "Select Blind",
                "Next Round",
            ):
                _reset_cash_out_watch()
        else:
            blind_deck_tab_streak = 0
            cash_box = metrics.get("cash_out_box")
            if screen_tracker.state in (
                CoarseScreenState.BLIND,
                CoarseScreenState.CASH_OUT,
            ):
                if cash_box is not None:
                    cash_out_visible_streak += 1
                    cash_out_missing_streak = 0
                    cash_out_first_missing_frame = None
                    cash_out_last_box = cash_box
                elif metrics.get("cash_out_present"):
                    cash_out_missing_streak = 0
                    cash_out_first_missing_frame = None
                elif cash_out_visible_streak >= 2:
                    if cash_out_missing_streak == 0:
                        cash_out_first_missing_frame = frame_idx
                    cash_out_missing_streak += 1
                    if cash_out_missing_streak >= 10 and "Cash Out" in template_key:
                        syn_frame = (
                            cash_out_first_missing_frame
                            if cash_out_first_missing_frame is not None
                            else frame_idx
                        )
                        if syn_frame not in action_frames:
                            action_frames[syn_frame] = {
                                "action": template_key["Cash Out"],
                                "box": cash_out_last_box,
                                "metadata": {"source": "cash_out_disappearance"},
                            }
                        screen_tracker.on_template_clicked(
                            "Cash Out", frame_idx=syn_frame
                        )
                        _reset_cash_out_watch()
                        _reset_open_pack_watch()
                        shop_last_activity_frame = syn_frame
            else:
                _reset_cash_out_watch()

            if screen_tracker.state == CoarseScreenState.SHOP:
                if metrics.get("open_pack_present"):
                    open_pack_visible_streak += 1
                    open_pack_missing_streak = 0
                elif open_pack_visible_streak >= 1:
                    open_pack_missing_streak += 1
                    if open_pack_missing_streak >= 2:
                        screen_tracker.on_pack_possibly_opened()
                        _reset_open_pack_watch()
                        shop_last_activity_frame = None
                if (
                    screen_tracker.state == CoarseScreenState.SHOP
                    and shop_last_activity_frame is not None
                    and (frame_idx - shop_last_activity_frame)
                    >= shop_pack_fallback_frames
                ):
                    screen_tracker.on_shop_idle_fallback()
                    _reset_open_pack_watch()
                    shop_last_activity_frame = None
            else:
                _reset_open_pack_watch()
                shop_last_activity_frame = None

        cur_state = (
            screen_tracker.state,
            screen_tracker.allow_play_game,
            screen_tracker.pre_pack_location,
            screen_tracker.last_cash_out_frame,
            screen_tracker.last_held_item_action_frame,
        )
        return cur_state != prev_state

    if config.debug_plot or config.setup_mode:
        # Sequential processing to ensure safe UI/Plotting on the main thread
        for video_frame in frames:
            active_templates = screen_tracker.filter_templates(
                loaded_templates, frame_idx=video_frame.index
            )
            frame_idx, matches, metrics, template_clicked, clicked_name = (
                _process_frame(
                    video_frame, active_templates, template_key, config, logger
                )
            )
            _apply_frame_detection(frame_idx, template_clicked, clicked_name, metrics)
            total_grayscale_time += metrics["grayscale_time"]
            total_eval_time += metrics["eval_time"]
            for template, plot_kwargs in matches:
                should_plot = config.debug_plot or (
                    config.setup_mode and not template.adjusted
                )
                if should_plot:
                    plot(**plot_kwargs)
                    try:
                        with open(
                            config.template_json_path, "r", encoding="utf-8"
                        ) as f:
                            data = json.load(f)
                        if template.name in data:
                            template.adjusted = data[template.name].get(
                                "adjusted", template.adjusted
                            )
                            template.shadow_dy = data[template.name].get(
                                "shadow_dy", template.shadow_dy
                            )
                            template.shadow_dw = data[template.name].get(
                                "shadow_dw", template.shadow_dw
                            )
                            template.shadow_dh = data[template.name].get(
                                "shadow_dh", template.shadow_dh
                            )
                            if "shadow_threshold" in data[template.name]:
                                raw_t = data[template.name]["shadow_threshold"]
                                template.shadow_threshold = (
                                    float(raw_t) if raw_t is not None else None
                                )
                    except Exception as e:
                        logger.error(
                            f"Failed to reload template data for {template.name}: {e}"
                        )
            results.append(frame_idx)
    else:
        # Chunked multithreaded processing for CPU maximum speed (avoids OOM on large videos)
        total_frames = get_expected_frame_count(config)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=config.max_workers
        ) as executor:
            num_workers = executor._max_workers
            with tqdm(total=total_frames, desc=progress_desc, unit="frame") as pbar:
                for chunk in _chunked_iterable(frames, 64):
                    pending = list(chunk)
                    while pending:
                        batch = pending
                        remainder: list[Any] = []
                        cooldown_cutoffs: list[int] = []
                        if screen_tracker.last_cash_out_frame is not None:
                            cooldown_cutoffs.append(
                                screen_tracker.last_cash_out_frame
                                + screen_tracker.shop_next_round_cooldown_frames
                            )
                        if screen_tracker.last_held_item_action_frame is not None:
                            cooldown_cutoffs.append(
                                screen_tracker.last_held_item_action_frame
                                + screen_tracker.cash_out_after_held_item_cooldown_frames
                            )
                        matching_cutoffs = [
                            c
                            for c in cooldown_cutoffs
                            if pending[0].index < c <= pending[-1].index
                        ]
                        if matching_cutoffs:
                            cutoff = min(matching_cutoffs)
                            batch = [vf for vf in pending if vf.index < cutoff]
                            remainder = [vf for vf in pending if vf.index >= cutoff]

                        active_templates = screen_tracker.filter_templates(
                            loaded_templates, frame_idx=batch[0].index
                        )
                        chunk_outputs = list(
                            executor.map(
                                lambda vf, tmpls=active_templates: _process_frame(
                                    vf, tmpls, template_key, config, logger
                                ),
                                batch,
                            )
                        )
                        transitioned_at = None
                        for idx_in_chunk, (
                            frame_idx,
                            matches,
                            metrics,
                            template_clicked,
                            clicked_name,
                        ) in enumerate(chunk_outputs):
                            results.append(frame_idx)
                            total_grayscale_time += metrics["grayscale_time"]
                            total_eval_time += metrics["eval_time"]
                            pbar.update(1)
                            if _apply_frame_detection(
                                frame_idx, template_clicked, clicked_name, metrics
                            ):
                                transitioned_at = idx_in_chunk
                                break
                        if transitioned_at is not None:
                            # Re-run only the suffix after a state transition under the updated whitelist
                            pending = batch[transitioned_at + 1 :] + remainder
                        else:
                            pending = remainder

    action_frames = _prune_isolated_single_frame_clicks(
        action_frames,
        template_key,
        template_names=("Sell Card", "Sort by Rank", "Sort by Suit"),
        skip_frames=config.skip_frames,
    )

    output_payload: dict[str, Any] = {
        "key": template_key,
        "frames": action_frames,
    }
    if extra_metadata is not None:
        output_payload["meta"] = extra_metadata

    num_frames = len(results)
    out_path: str | None = None
    if num_frames > 0:
        elapsed = time.perf_counter() - start
        avg_wall_time = elapsed / num_frames
        print(f"Finished in {elapsed:.2f} seconds; {avg_wall_time:.4f} per frame.")
        if config.run_benchmark:
            wall_extract = sum(video_extract_times) / num_frames
            wall_gray = (total_grayscale_time / num_frames) / num_workers
            wall_eval = (total_eval_time / num_frames) / num_workers
            wall_other = avg_wall_time - (wall_extract + wall_gray + wall_eval)

            pct_extract = (wall_extract / avg_wall_time) * 100
            pct_gray = (wall_gray / avg_wall_time) * 100
            pct_eval = (wall_eval / avg_wall_time) * 100
            pct_other = (wall_other / avg_wall_time) * 100

            print("\n--- Benchmark Breakdown (Effective Wall Time per frame) ---")
            print(f"Video Extraction:   {wall_extract:.5f}s ({pct_extract:4.1f}%)")
            print(f"Grayscale Convert:  {wall_gray:.5f}s ({pct_gray:4.1f}%)")
            print(f"Template Matching:  {wall_eval:.5f}s ({pct_eval:4.1f}%)")
            print(f"Other (Overhead):   {wall_other:.5f}s ({pct_other:4.1f}%)")
            print("-----------------------------------------------------------")

        if write_output:
            out_path = output_json_path or build_output_json_path(
                config.output_dir, config.input_video_path
            )
            os.makedirs(Path(out_path).parent, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(output_payload, f, indent=2)

    return output_payload, out_path


def build_cli_parser() -> argparse.ArgumentParser:
    """Builds the unified end-to-end CLI parser for video detection + trajectory reconstruction."""
    default_cfg = Config()
    p = argparse.ArgumentParser(
        description="Botlatro End-to-End Video-to-GameState Pipeline (Template Detection + CV + HUD OCR)."
    )

    # Video & template matching options
    p.add_argument(
        "--input-video",
        type=str,
        default=default_cfg.input_video_path,
        help="Path to gameplay video",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=default_cfg.output_dir,
        help="Directory for output JSON files",
    )
    p.add_argument("--template-json", type=str, default=default_cfg.template_json_path)
    p.add_argument("--template-dir", type=str, default=default_cfg.template_img_dir)
    p.add_argument(
        "--start-timestamp",
        type=float,
        default=default_cfg.start_timestamp,
        help="Start time in seconds",
    )
    p.add_argument(
        "--end-timestamp",
        type=float,
        default=default_cfg.end_timestamp,
        help="End time in seconds (omit or use --full-video for entire video)",
    )
    p.add_argument(
        "--full-video",
        action="store_true",
        help="Process the entire video (overrides --end-timestamp to None)",
    )
    p.add_argument(
        "--skip-frames",
        type=int,
        default=default_cfg.skip_frames,
        help="Skip N frames between template scans",
    )
    p.add_argument(
        "--use-hwaccel",
        action=argparse.BooleanOptionalAction,
        default=default_cfg.use_hwaccel,
    )
    p.add_argument(
        "--coarse-to-fine",
        action=argparse.BooleanOptionalAction,
        default=default_cfg.coarse_to_fine,
    )
    p.add_argument("--coarse-margin", type=float, default=default_cfg.coarse_margin)
    p.add_argument("--max-workers", type=int, default=default_cfg.max_workers)
    p.add_argument(
        "--debug-plot",
        action=argparse.BooleanOptionalAction,
        default=default_cfg.debug_plot,
    )
    p.add_argument(
        "--setup-mode",
        action=argparse.BooleanOptionalAction,
        default=default_cfg.setup_mode,
    )
    p.add_argument(
        "--run-benchmark",
        action=argparse.BooleanOptionalAction,
        default=default_cfg.run_benchmark,
    )

    # Run metadata & post-processing options
    p.add_argument("--seed", type=str, default="", help="Optional Balatro run seed")
    p.add_argument(
        "--deck",
        type=str,
        default="red",
        help="Deck name (e.g. red, blue, plasma, abandoned)",
    )
    p.add_argument(
        "--stake", type=str, default="white", help="Stake name (e.g. white, red, gold)"
    )
    p.add_argument(
        "--actions-json",
        type=str,
        default=None,
        help="Existing main.py action JSON (skips Stage 1 template matching)",
    )
    p.add_argument(
        "--actions-out",
        type=str,
        default=None,
        help="Optional output path for Stage 1 action JSON",
    )
    p.add_argument(
        "--trajectory-out",
        type=str,
        default=None,
        help="Optional output path for reconstructed trajectory JSON",
    )
    p.add_argument(
        "--skip-post-processing",
        action="store_true",
        help="Run only Stage 1 template action detection",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override source video FPS for post-processing",
    )
    p.add_argument(
        "--conf-hand",
        type=float,
        default=0.50,
        help="Hand card detector confidence threshold",
    )
    p.add_argument(
        "--conf-shop",
        type=float,
        default=0.15,
        help="Shop/joker detector confidence threshold",
    )
    p.add_argument(
        "--conf-ocr", type=float, default=0.90, help="HUD OCR confidence threshold"
    )
    p.add_argument("--batch-size", type=int, default=32, help="CV inference batch size")
    p.add_argument(
        "--device", type=str, default=None, help="Torch device (e.g. cuda, cpu)"
    )
    p.add_argument(
        "--expressive",
        action="store_true",
        help="Show progress bars and end-of-run diagnostics (OCR drift overrides, mismatches, timings)",
    )
    return p


def config_from_cli_args(args: argparse.Namespace) -> Config:
    """Constructs a `Config` instance from parsed CLI arguments."""
    return Config(
        template_json_path=args.template_json,
        template_img_dir=args.template_dir,
        input_video_path=args.input_video,
        output_dir=args.output_dir,
        start_timestamp=args.start_timestamp,
        end_timestamp=None if args.full_video else args.end_timestamp,
        skip_frames=args.skip_frames,
        use_hwaccel=args.use_hwaccel,
        coarse_to_fine=args.coarse_to_fine,
        coarse_margin=args.coarse_margin,
        max_workers=max(1, int(args.max_workers)),
        debug_plot=args.debug_plot,
        setup_mode=args.setup_mode,
        run_benchmark=args.run_benchmark,
    )


def run_end_to_end(args: argparse.Namespace) -> dict[str, Any]:
    """Runs Stage 1 template detection (unless `--actions-json` is given) and Stage 2 post-processing."""
    config = config_from_cli_args(args)
    deck = parse_deck(args.deck)
    stake = parse_stake(args.stake)

    if args.actions_json:
        actions_path = args.actions_json
        with open(actions_path, encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload, actions_path = detect_video_actions(
            config,
            write_output=True,
            output_json_path=args.actions_out,
        )

    if args.skip_post_processing:
        return {
            "actions_payload": payload,
            "actions_json": actions_path,
        }

    state, trajectory = build_run_trajectory(
        payload,
        video_path=config.input_video_path,
        seed=args.seed,
        deck=deck,
        stake=stake,
        fps=args.fps,
        conf_threshold_hand=args.conf_hand,
        conf_threshold_shop=args.conf_shop,
        conf_threshold_ocr=args.conf_ocr,
        batch_size=args.batch_size,
        device=args.device,
        expressive=args.expressive,
    )

    summary = serialize_trajectory_summary(state, trajectory)
    if args.trajectory_out:
        traj_path = Path(args.trajectory_out)
    elif actions_path:
        traj_path = default_trajectory_output_path(actions_path)
    else:
        out_dir = Path(config.output_dir)
        traj_dir = (
            out_dir.parent / "trajectories"
            if out_dir.name == "vod_actions"
            else out_dir / "trajectories"
        )
        traj_path = traj_dir / f"{Path(config.input_video_path).stem}_trajectory.json"

    traj_path.parent.mkdir(parents=True, exist_ok=True)
    with traj_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"\n[Pipeline Complete] {summary['num_steps']} steps | "
        f"Deck={deck.value} | Stake={stake.value} | "
        f"Ante {state.ante} Round {state.round} ({state.current_blind.value}) | "
        f"Money=${state.money} | Trajectory: {traj_path}"
    )

    return {
        "actions_payload": payload,
        "actions_json": actions_path,
        "trajectory_json": str(traj_path),
        "state": state,
        "trajectory": trajectory,
    }


if __name__ == "__main__":
    configure_logging()
    run_end_to_end(build_cli_parser().parse_args())
