"""Overlapped concurrent downloader + Stage 1/2 processor for Balatro VODs."""

from __future__ import annotations

from collections import Counter
import concurrent.futures
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Sequence

from post_processing import parse_deck, parse_stake
from vod_pipeline.metadata import (
    ExtractedRunMetadata,
    compact_raw_metadata,
    extract_run_metadata,
    fetch_metadata_batch,
)

# VideoReader requires 1920x1080 and audio is unused; prefer H.264/MP4 for fastest decode.
DEFAULT_VIDEO_FORMAT: str = (
    "bestvideo[width=1920][height=1080][vcodec^=avc1]/"
    "bestvideo[width=1920][height=1080][ext=mp4]/"
    "bestvideo[width=1920][height=1080]/"
    "best[width=1920][height=1080]/"
    "bestvideo[height=1080]/"
    "best[height=1080]"
)


@dataclass
class PipelineConfig:
    """Configuration for autonomous VOD batch processing."""

    work_dir: Path = Path("batch_runs")
    max_backlog: int = 5
    download_workers: int = 3
    metadata_workers: int = 6
    concurrent_fragments: int = 4
    video_format: str = DEFAULT_VIDEO_FORMAT
    keep_videos: bool = False
    resume: bool = True
    dry_run: bool = False
    allow_deck_template_fallback: bool = True
    allow_ocr_seed_fallback: bool = False
    stage1_workers: int = 1
    stage1_threads_per_worker: int | None = None
    yt_dlp_retries: int = 3
    yt_dlp_retry_backoff_sec: float = 3.0
    sleep_requests: float | None = None
    sleep_interval: float | None = None
    cookies_from_browser: str | None = None
    cookies_file: Path | None = None
    use_metadata_cache: bool = True
    skip_frames: int = 1
    batch_size: int = 32
    conf_hand: float = 0.50
    conf_shop: float = 0.15
    conf_ocr: float = 0.90
    device: str | None = None
    expressive: bool = False
    min_duration_sec: float = 1800.0
    metadata_timeout_sec: int = 60
    download_timeout_sec: int = 3600


def _cleanup_partial_downloads(downloads_dir: Path, video_id: str) -> None:
    """Removes partial/temporary yt-dlp files for `video_id` after error or abort."""
    if not downloads_dir.exists():
        return
    for candidate in downloads_dir.glob(f"{video_id}*"):
        if candidate.is_file() and (
            candidate.name.endswith(".part")
            or candidate.name.endswith(".ytdl")
            or ".temp." in candidate.name
        ):
            candidate.unlink(missing_ok=True)


def _find_downloaded_file(downloads_dir: Path, video_id: str) -> Path | None:
    """Finds a completed media file for `video_id` inside `downloads_dir`."""
    candidates = [
        p
        for p in sorted(downloads_dir.glob(f"{video_id}.*"))
        if p.is_file()
        and not p.name.endswith((".part", ".ytdl", ".json"))
        and ".temp." not in p.name
        and p.stat().st_size > 0
    ]
    if not candidates:
        return None
    for p in candidates:
        if p.suffix.lower() == ".mp4":
            return p
    return candidates[0]


def _sleep_with_stop(seconds: float, stop_event: threading.Event | None) -> None:
    if seconds <= 0:
        return
    deadline = time.monotonic() + seconds
    while True:
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError("Sleep cancelled by stop_event")
        rem = deadline - time.monotonic()
        if rem <= 0:
            break
        time.sleep(min(0.25, rem))


def download_video(
    url: str,
    video_id: str,
    downloads_dir: Path,
    *,
    video_format: str = DEFAULT_VIDEO_FORMAT,
    concurrent_fragments: int = 4,
    timeout_sec: int = 3600,
    retries: int = 3,
    retry_backoff_sec: float = 3.0,
    sleep_requests: float | None = None,
    sleep_interval: float | None = None,
    cookies_from_browser: str | None = None,
    cookies_file: str | Path | None = None,
    stop_event: threading.Event | None = None,
) -> Path:
    """Downloads a single 1080p video (no audio needed) into `downloads_dir` via `yt-dlp` with retries."""
    downloads_dir.mkdir(parents=True, exist_ok=True)
    existing = _find_downloaded_file(downloads_dir, video_id)
    if existing is not None:
        return existing

    if stop_event is not None and stop_event.is_set():
        raise InterruptedError("Download aborted before start")

    out_tmpl = str(downloads_dir / f"{video_id}.%(ext)s")
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--no-warnings",
        "--newline",
        "--concurrent-fragments",
        str(max(1, int(concurrent_fragments))),
        "-f",
        video_format,
        "--merge-output-format",
        "mp4",
    ]
    if cookies_from_browser:
        cmd.extend(["--cookies-from-browser", str(cookies_from_browser)])
    elif cookies_file:
        cmd.extend(["--cookies", str(cookies_file)])
    if sleep_requests is not None and sleep_requests > 0:
        cmd.extend(["--sleep-requests", str(sleep_requests)])
    if sleep_interval is not None and sleep_interval > 0:
        cmd.extend(["--sleep-interval", str(sleep_interval)])
    cmd.extend(["-o", out_tmpl, "--print", "after_move:filepath", url])

    max_attempts = 1 + max(0, int(retries))
    last_exc: BaseException | None = None
    for attempt in range(max_attempts):
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError(f"Download aborted before attempt {attempt + 1}")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        deadline = time.monotonic() + max(10, int(timeout_sec))
        try:
            while proc.poll() is None:
                if stop_event is not None and stop_event.is_set():
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    raise InterruptedError(f"Download cancelled for {video_id}")
                if time.monotonic() > deadline:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    raise TimeoutError(
                        f"yt-dlp download timed out after {timeout_sec}s for {video_id}"
                    )
                time.sleep(0.25)

            stdout_text, stderr_text = proc.communicate()
            if proc.returncode != 0:
                err_msg = (
                    stderr_text or stdout_text or f"yt-dlp exit {proc.returncode}"
                ).strip()
                raise RuntimeError(f"yt-dlp failed ({proc.returncode}): {err_msg[:500]}")

            for line in reversed(
                [ln.strip() for ln in stdout_text.splitlines() if ln.strip()]
            ):
                p = Path(line)
                if p.exists() and p.is_file():
                    return p
            found = _find_downloaded_file(downloads_dir, video_id)
            if found is None:
                raise FileNotFoundError(
                    f"yt-dlp finished for {video_id} cleanly, but no output file was found in {downloads_dir}"
                )
            return found
        except InterruptedError:
            _cleanup_partial_downloads(downloads_dir, video_id)
            raise
        except BaseException as exc:
            _cleanup_partial_downloads(downloads_dir, video_id)
            last_exc = exc
            if attempt + 1 < max_attempts:
                backoff = float(retry_backoff_sec) * (2**attempt)
                _sleep_with_stop(backoff, stop_event)
                continue
            raise
    assert last_exc is not None
    raise last_exc


def _has_valid_actions_json(path: Path) -> bool:
    """Returns True when `path` is a readable Stage-1 actions JSON with `key` and `frames`."""
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return (
            isinstance(payload, dict)
            and isinstance(payload.get("key"), dict)
            and isinstance(payload.get("frames"), dict)
        )
    except Exception:
        return False


def _summarize_trajectory_cleanliness(source: Any) -> dict[str, Any]:
    """Computes clean step & Ante-bucket telemetry from a `RunTrajectory` or saved JSON path."""
    steps_iter: list[tuple[int | None, bool, Sequence[str]]] = []
    cur_ante = 1
    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.exists():
            return {}
        data = json.loads(p.read_text(encoding="utf-8"))
        init_st = data.get("initial_state") or {}
        cur_ante = int(init_st.get("ante") or 1)
        for st in data.get("steps") or ():
            act = st.get("action") or {}
            meta = act.get("metadata") or {}
            steps_iter.append(
                (
                    st.get("ante"),
                    bool(meta.get("corrupted")),
                    list(meta.get("corruption_reasons") or ()),
                )
            )
    elif source is not None and hasattr(source, "deltas"):
        try:
            st0 = source.get_state(0) if source.deltas else None
            cur_ante = int(getattr(st0, "ante", 1) or 1)
        except Exception:
            cur_ante = 1
        for dl in source.deltas:
            act = getattr(dl, "action", None)
            meta = (getattr(act, "metadata", None) or {}) if act is not None else {}
            steps_iter.append(
                (
                    getattr(dl, "ante", None),
                    bool(meta.get("corrupted")),
                    list(meta.get("corruption_reasons") or ()),
                )
            )
    else:
        return {}

    total = len(steps_iter)
    clean = 0
    buckets: dict[str, list[int]] = {
        "a13": [0, 0],
        "a46": [0, 0],
        "a78": [0, 0],
    }
    reasons_cnt: Counter[str] = Counter()
    cascade_step: int | None = None

    for idx, (ante_val, corrupted, reasons) in enumerate(steps_iter):
        if ante_val is not None:
            cur_ante = int(ante_val)
        b_key = "a13" if cur_ante <= 3 else ("a46" if cur_ante <= 6 else "a78")
        buckets[b_key][1] += 1
        if corrupted:
            for r in reasons:
                reasons_cnt[str(r)] += 1
            if cascade_step is None and "downstream_cascade_quarantine" in reasons:
                cascade_step = idx
        else:
            clean += 1
            buckets[b_key][0] += 1

    a78_clean, a78_total = buckets["a78"]
    return {
        "clean_steps": clean,
        "total_steps": total,
        "clean_pct": round(100.0 * clean / max(1, total), 2),
        "ante13_clean": buckets["a13"][0],
        "ante13_total": buckets["a13"][1],
        "ante46_clean": buckets["a46"][0],
        "ante46_total": buckets["a46"][1],
        "ante78_clean": a78_clean,
        "ante78_total": a78_total,
        "ante78_pct": round(100.0 * a78_clean / max(1, a78_total), 2),
        "cascade_step": cascade_step,
        "corruption_reasons": dict(reasons_cnt.most_common()),
    }


def _release_runtime_memory() -> None:
    """Reclaims Python objects and cached CUDA memory between sequential VOD runs."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def process_downloaded_video(
    video_path: Path,
    meta: ExtractedRunMetadata,
    cfg: PipelineConfig,
    *,
    actions_json_override: Path | None = None,
    stage1_elapsed_sec: float | None = None,
    actions_reused_override: bool | None = None,
) -> dict[str, Any]:
    """Runs `main.run_end_to_end` (Stage 1+2 or Stage 2 from `--actions-json`) on a VOD."""
    import main as vod_main

    actions_dir = cfg.work_dir / "actions"
    trajectories_dir = cfg.work_dir / "trajectories"
    actions_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    actions_out = actions_dir / f"{meta.video_id}_actions.json"
    trajectory_out = trajectories_dir / f"{meta.video_id}_trajectory.json"

    # Validate enums before spawning Stage 1/2
    parse_deck(meta.deck_cli_value)
    parse_stake(meta.stake_cli_value)

    use_actions_json: Path | None = None
    actions_reused = False
    if actions_json_override is not None:
        use_actions_json = Path(actions_json_override)
        actions_reused = bool(actions_reused_override)
    elif cfg.resume and _has_valid_actions_json(actions_out):
        use_actions_json = actions_out
        actions_reused = True

    argv: list[str] = [
        "--input-video",
        str(video_path),
        "--output-dir",
        str(actions_dir),
        "--trajectory-out",
        str(trajectory_out),
        "--deck",
        meta.deck_cli_value,
        "--stake",
        meta.stake_cli_value,
        "--seed",
        meta.seed,
        "--skip-frames",
        str(cfg.skip_frames),
        "--batch-size",
        str(cfg.batch_size),
        "--conf-hand",
        str(cfg.conf_hand),
        "--conf-shop",
        str(cfg.conf_shop),
        "--conf-ocr",
        str(cfg.conf_ocr),
    ]
    if use_actions_json is not None:
        argv.extend(["--actions-json", str(use_actions_json)])
    else:
        argv.extend(["--full-video", "--actions-out", str(actions_out)])
    if cfg.device:
        argv.extend(["--device", cfg.device])
    if cfg.expressive:
        argv.append("--expressive")

    args = vod_main.build_cli_parser().parse_args(argv)
    t0 = time.perf_counter()
    res = vod_main.run_end_to_end(args)
    elapsed = time.perf_counter() - t0
    traj = res.get("trajectory")
    st = res.get("state")
    traj_path = Path(res.get("trajectory_json") or trajectory_out)
    clean_summary = _summarize_trajectory_cleanliness(
        traj if traj is not None else traj_path
    )
    stage1_sec = (
        round(float(stage1_elapsed_sec), 2)
        if stage1_elapsed_sec is not None
        else (0.0 if actions_reused else None)
    )
    stage2_sec = round(elapsed, 2) if use_actions_json is not None else None
    total_sec = (
        round(float(stage1_elapsed_sec) + elapsed, 2)
        if stage1_elapsed_sec is not None
        else round(elapsed, 2)
    )
    out: dict[str, Any] = {
        "actions_json": str(res.get("actions_json") or (use_actions_json or actions_out)),
        "trajectory_json": str(traj_path),
        "actions_reused": actions_reused,
        "num_steps": len(traj.deltas) if traj is not None else clean_summary.get("total_steps"),
        "final_ante": int(st.ante) if st is not None else None,
        "final_round": int(st.round) if st is not None else None,
        "elapsed_sec": total_sec,
        **clean_summary,
    }
    if stage1_sec is not None:
        out["stage1_sec"] = stage1_sec
    if stage2_sec is not None:
        out["stage2_sec"] = stage2_sec
    return out


def _run_stage1_subprocess(
    video_path: Path,
    meta: ExtractedRunMetadata,
    cfg: PipelineConfig,
    actions_out: Path,
    *,
    threads_per_worker: int,
    stop_event: threading.Event | None = None,
) -> float:
    """Runs CPU-only `main.py --skip-post-processing` in an isolated subprocess."""
    actions_out.parent.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg.work_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{meta.video_id}_stage1.log"

    repo_root = Path(__file__).resolve().parent.parent
    deck_arg = meta.deck_cli_value if meta.deck is not None else "Red Deck"
    stake_arg = meta.stake_cli_value if meta.stake is not None else "Gold Stake"
    seed_arg = meta.seed or "00000000"

    cmd = [
        sys.executable,
        "-u",
        str(repo_root / "main.py"),
        "--input-video",
        str(video_path),
        "--full-video",
        "--skip-post-processing",
        "--output-dir",
        str(actions_out.parent),
        "--actions-out",
        str(actions_out),
        "--deck",
        deck_arg,
        "--stake",
        stake_arg,
        "--seed",
        seed_arg,
        "--skip-frames",
        str(cfg.skip_frames),
        "--max-workers",
        str(max(1, int(threads_per_worker))),
    ]
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")

    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8", errors="replace") as log_f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        while proc.poll() is None:
            if stop_event is not None and stop_event.is_set():
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                raise InterruptedError("Stage 1 aborted due to stop_event")
            time.sleep(0.25)

    elapsed = time.perf_counter() - t0
    if proc.returncode != 0 or not _has_valid_actions_json(actions_out):
        raise RuntimeError(
            f"Stage 1 failed (exit={proc.returncode}); see {log_path}"
        )
    return elapsed


_DECK_RECOVERABLE_SKIP_REASONS = frozenset(
    {"missing_deck", "ambiguous_multiple_decks", "ambiguous_deck_conflict"}
)


def _try_deck_template_fallback(video_path: Path, cfg: PipelineConfig) -> Any | None:
    """Samples a few frames of `video_path` to identify the deck back when metadata lacks it."""
    try:
        from vod_pipeline.deck_matcher import detect_deck_from_video

        return detect_deck_from_video(video_path)
    except Exception:
        return None


def _try_ocr_seed_fallback(video_path: Path, cfg: PipelineConfig) -> tuple[str, float] | None:
    """Attempts end-of-run Seed OCR recovery when `--allow-ocr-seed-fallback` is enabled."""
    try:
        from classifiers.seed_ocr.pipeline import (
            extract_seed_from_video,
            resolve_seed_weights_path,
        )

        weights_path = resolve_seed_weights_path()
        if weights_path is None:
            return None
        res = extract_seed_from_video(video_path, device=cfg.device)
        if res is not None and res.seed:
            return res.seed, float(res.confidence)
    except Exception:
        return None
    return None


def _append_manifest_row(manifest_path: Path, row: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _default_fetch_all(urls: Sequence[str], cfg: PipelineConfig) -> list[tuple[dict[str, Any], ExtractedRunMetadata]]:
    meta_dir = cfg.work_dir / "metadata"
    return fetch_metadata_batch(
        urls,
        max_workers=cfg.metadata_workers,
        save_dir=meta_dir,
        timeout=cfg.metadata_timeout_sec,
        retries=cfg.yt_dlp_retries,
        retry_backoff_sec=cfg.yt_dlp_retry_backoff_sec,
        cookies_from_browser=cfg.cookies_from_browser,
        cookies_file=cfg.cookies_file,
        sleep_requests=cfg.sleep_requests,
        use_cache=cfg.use_metadata_cache,
    )


def _aggregate_cleanliness_summary(processed: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregates clean step counts, Ante buckets, and corruption reasons across processed VODs."""
    total_steps = 0
    clean_steps = 0
    a13_c = a13_t = a46_c = a46_t = a78_c = a78_t = 0
    cascades: list[str] = []
    reasons: Counter[str] = Counter()
    with_telemetry = 0

    for row in processed:
        ts = row.get("total_steps")
        if ts is None:
            continue
        with_telemetry += 1
        total_steps += int(ts)
        clean_steps += int(row.get("clean_steps") or 0)
        a13_c += int(row.get("ante13_clean") or 0)
        a13_t += int(row.get("ante13_total") or 0)
        a46_c += int(row.get("ante46_clean") or 0)
        a46_t += int(row.get("ante46_total") or 0)
        a78_c += int(row.get("ante78_clean") or 0)
        a78_t += int(row.get("ante78_total") or 0)
        if row.get("cascade_step") is not None:
            cascades.append(str(row.get("video_id")))
        for k, v in (row.get("corruption_reasons") or {}).items():
            reasons[str(k)] += int(v)

    return {
        "videos_with_telemetry": with_telemetry,
        "clean_steps": clean_steps,
        "total_steps": total_steps,
        "clean_pct": round(100.0 * clean_steps / max(1, total_steps), 2) if total_steps else 0.0,
        "ante13_clean": a13_c,
        "ante13_total": a13_t,
        "ante13_pct": round(100.0 * a13_c / max(1, a13_t), 2) if a13_t else 0.0,
        "ante46_clean": a46_c,
        "ante46_total": a46_t,
        "ante46_pct": round(100.0 * a46_c / max(1, a46_t), 2) if a46_t else 0.0,
        "ante78_clean": a78_c,
        "ante78_total": a78_t,
        "ante78_pct": round(100.0 * a78_c / max(1, a78_t), 2) if a78_t else 0.0,
        "cascade_videos": cascades,
        "corruption_reasons": dict(reasons.most_common()),
    }


def _print_batch_summary(report: dict[str, Any]) -> None:
    counts = report["counts"]
    print("\n================ VOD Batch Pipeline Summary ================")
    print(
        f"Total URLs: {counts['total']} | Ready: {counts['eligible']} | "
        f"Processed: {counts['processed']} | Skipped: {counts['skipped']} | "
        f"Failed: {counts['failed']} | Dry-run: {report['config']['dry_run']} | "
        f"Elapsed: {report.get('elapsed_sec')}s"
    )
    csum = report.get("cleanliness_summary") or {}
    if csum.get("total_steps"):
        print(
            f"Cleanliness: {csum['clean_steps']}/{csum['total_steps']} ({csum['clean_pct']:.2f}%) | "
            f"A1-3: {csum['ante13_clean']}/{csum['ante13_total']} ({csum['ante13_pct']:.2f}%) | "
            f"A4-6: {csum['ante46_clean']}/{csum['ante46_total']} ({csum['ante46_pct']:.2f}%) | "
            f"A7-8: {csum['ante78_clean']}/{csum['ante78_total']} ({csum['ante78_pct']:.2f}%)"
        )
        if csum.get("cascade_videos"):
            print(f"Cascades: {', '.join(csum['cascade_videos'])}")
    if report["processed"]:
        print("\n[Processed]")
        for row in report["processed"]:
            clean_str = ""
            if row.get("total_steps"):
                clean_str = (
                    f" clean={row['clean_steps']}/{row['total_steps']} ({row['clean_pct']:.1f}%,"
                    f" A7-8={row['ante78_clean']}/{row['ante78_total']})"
                )
            t_str = f" t={row.get('elapsed_sec')}s"
            if row.get("stage1_sec") is not None and row.get("stage2_sec") is not None:
                t_str = f" s1={row['stage1_sec']}s s2={row['stage2_sec']}s"
            print(
                f"  - {row['video_id']} | {row['deck']} ({row.get('deck_source', '')}) | "
                f"{row['stake']} | Seed={row['seed']} ({row.get('seed_source', '')}) | "
                f"steps={row.get('num_steps')}{clean_str}{t_str} -> {row.get('trajectory_json')}"
            )
    if report["skipped"]:
        print("\n[Skipped]")
        for row in report["skipped"]:
            print(
                f"  - {row['video_id']} | reason={row['skip_reason']} | "
                f"date={row.get('upload_date') or '?'} | title={row.get('title')!r}"
            )
    if report["failed"]:
        print("\n[Failed]")
        for row in report["failed"]:
            print(
                f"  - {row['video_id']} ({row['stage']}): {row['error']}"
            )
    print(f"\nReport saved: {report['report_path']}")
    print("============================================================\n")


def run_batch_pipeline(
    urls: Sequence[str],
    cfg: PipelineConfig,
    *,
    fetch_metadata_fn: Callable[[Sequence[str], PipelineConfig], list[tuple[dict[str, Any], ExtractedRunMetadata]]] | None = None,
    download_fn: Callable[[str, str, Path, PipelineConfig, threading.Event], Path] | None = None,
    process_fn: Callable[[Path, ExtractedRunMetadata, PipelineConfig], dict[str, Any]] | None = None,
    deck_matcher_fn: Callable[[Path, PipelineConfig], Any] | None = None,
    stage1_fn: Callable[[Path, ExtractedRunMetadata, PipelineConfig, Path], float] | None = None,
    stage2_fn: Callable[[Path, ExtractedRunMetadata, PipelineConfig, Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Executes metadata gating + overlapped bounded-backlog download, Stage-1 & Stage-2 processing."""
    t_start = time.perf_counter()
    work_dir = Path(cfg.work_dir)
    meta_dir = work_dir / "metadata"
    downloads_dir = work_dir / "downloads"
    actions_dir = work_dir / "actions"
    trajectories_dir = work_dir / "trajectories"
    reports_dir = work_dir / "reports"
    meta_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    if not cfg.dry_run:
        downloads_dir.mkdir(parents=True, exist_ok=True)
        actions_dir.mkdir(parents=True, exist_ok=True)
        trajectories_dir.mkdir(parents=True, exist_ok=True)

    fetcher = fetch_metadata_fn or _default_fetch_all
    downloader = download_fn or (
        lambda u, vid, ddir, c, ev: download_video(
            u,
            vid,
            ddir,
            video_format=c.video_format,
            concurrent_fragments=c.concurrent_fragments,
            timeout_sec=c.download_timeout_sec,
            retries=c.yt_dlp_retries,
            retry_backoff_sec=c.yt_dlp_retry_backoff_sec,
            sleep_requests=c.sleep_requests,
            sleep_interval=c.sleep_interval,
            cookies_from_browser=c.cookies_from_browser,
            cookies_file=c.cookies_file,
            stop_event=ev,
        )
    )
    processor = process_fn or process_downloaded_video
    deck_matcher = deck_matcher_fn or _try_deck_template_fallback

    fetched = fetcher(urls, cfg)
    # Persist metadata if caller injected a custom fetcher
    if fetch_metadata_fn is not None and fetched:
        idx_rows = []
        for raw, meta in fetched:
            p = meta_dir / f"{meta.video_id}.json"
            payload = dict(raw) if "title" in raw else compact_raw_metadata(raw)
            payload["extracted"] = meta.to_dict()
            with p.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            idx_rows.append(meta.to_dict())
        with (meta_dir / "index.json").open("w", encoding="utf-8") as f:
            json.dump(idx_rows, f, indent=2, ensure_ascii=False)

    seed_ocr_weights_available = False
    if cfg.allow_ocr_seed_fallback:
        try:
            from classifiers.seed_ocr.pipeline import resolve_seed_weights_path

            seed_ocr_weights_available = resolve_seed_weights_path() is not None
        except Exception:
            seed_ocr_weights_available = False

    eligible: list[ExtractedRunMetadata] = []
    skipped: list[dict[str, Any]] = []
    for _raw, meta in fetched:
        traj_path = trajectories_dir / f"{meta.video_id}_trajectory.json"
        if cfg.resume and traj_path.exists() and traj_path.stat().st_size > 0:
            row = meta.to_dict()
            row["skip_reason"] = "already_processed"
            row["trajectory_json"] = str(traj_path)
            skipped.append(row)
            continue

        if (
            cfg.min_duration_sec > 0
            and meta.duration_seconds is not None
            and meta.duration_seconds < cfg.min_duration_sec
        ):
            row = meta.to_dict()
            row["skip_reason"] = (
                f"too_short_under_{int(round(cfg.min_duration_sec / 60))}m"
            )
            skipped.append(row)
            continue

        if meta.ready:
            eligible.append(meta)
            continue

        # Recover missing/ambiguous deck via sparse video frame template matching when enabled
        if (
            cfg.allow_deck_template_fallback
            and meta.skip_reason in _DECK_RECOVERABLE_SKIP_REASONS
            and meta.seed
        ):
            meta.deck = None
            meta.deck_source = "pending_deck_template_match"
            eligible.append(meta)
            continue

        # Optional OCR fallback ONLY if deck is valid, missing_seed is the sole gap, and weights exist
        if (
            cfg.allow_ocr_seed_fallback
            and seed_ocr_weights_available
            and meta.skip_reason == "missing_seed"
            and meta.deck is not None
        ):
            meta.seed_source = "pending_ocr_end_of_run"
            eligible.append(meta)
            continue

        if (
            cfg.allow_deck_template_fallback
            and cfg.allow_ocr_seed_fallback
            and seed_ocr_weights_available
            and (meta.skip_reason == "missing_deck_and_seed" or meta.skip_reason in _DECK_RECOVERABLE_SKIP_REASONS)
            and not meta.seed
        ):
            meta.deck = None
            meta.deck_source = "pending_deck_template_match"
            meta.seed_source = "pending_ocr_end_of_run"
            eligible.append(meta)
            continue

        skipped.append(meta.to_dict())

    processed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    manifest_path = reports_dir / "manifest.jsonl"

    if not cfg.dry_run and eligible:
        backlog = max(1, int(cfg.max_backlog))
        dl_workers = max(1, min(int(cfg.download_workers), backlog))
        s1_workers = max(1, int(cfg.stage1_workers))
        use_split_stage1 = (process_fn is None and (s1_workers > 1 or stage1_fn is not None or stage2_fn is not None))
        threads_per_s1 = (
            max(1, int(cfg.stage1_threads_per_worker))
            if cfg.stage1_threads_per_worker is not None
            else max(2, (os.cpu_count() or 8) // s1_workers)
        )

        slot_sem = threading.Semaphore(backlog)
        ready_queue: queue.Queue[tuple[ExtractedRunMetadata, Path, float] | None] = queue.Queue()
        stage2_queue: queue.Queue[
            tuple[ExtractedRunMetadata, Path, float, Path, float, bool] | None
        ] = queue.Queue()
        stop_event = threading.Event()
        lock = threading.Lock()

        prev_sigint = signal.getsignal(signal.SIGINT)
        interrupted = [False]

        def _handle_sigint(signum, frame) -> None:
            interrupted[0] = True
            stop_event.set()
            raise KeyboardInterrupt

        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, _handle_sigint)

        def _resolve_pending_deck(item: ExtractedRunMetadata, vpath: Path) -> bool:
            if item.deck is None and cfg.allow_deck_template_fallback:
                deck_match = deck_matcher(vpath, cfg)
                if deck_match is None:
                    skip_row = {
                        **item.to_dict(),
                        "skip_reason": "deck_template_not_found",
                    }
                    with lock:
                        skipped.append(skip_row)
                        _append_manifest_row(manifest_path, {"status": "skipped", **skip_row})
                    return False
                if hasattr(deck_match, "deck"):
                    item.deck = deck_match.deck
                    item.deck_source = (
                        f"deck_template_match(conf={deck_match.confidence:.2f},"
                        f"{deck_match.matched_frames}/{deck_match.sampled_frames})"
                    )
                else:
                    item.deck = deck_match
                    item.deck_source = "deck_template_match"
                if item.seed:
                    item.skip_reason = None
                    item.ready = True
            return True

        def _resolve_pending_seed(item: ExtractedRunMetadata, vpath: Path) -> bool:
            if not item.seed and cfg.allow_ocr_seed_fallback:
                ocr_res = _try_ocr_seed_fallback(vpath, cfg)
                if ocr_res is None:
                    skip_row = {**item.to_dict(), "skip_reason": "ocr_seed_not_found"}
                    with lock:
                        skipped.append(skip_row)
                        _append_manifest_row(manifest_path, {"status": "skipped", **skip_row})
                    return False
                item.seed, conf = ocr_res
                item.seed_source = f"seed_ocr(conf={conf:.2f})"
                item.skip_reason = None
                item.ready = True
            return True

        def _download_one(item: ExtractedRunMetadata) -> None:
            t_dl0 = time.perf_counter()
            handed_off = False
            try:
                vpath = downloader(item.url, item.video_id, downloads_dir, cfg, stop_event)
                ready_queue.put((item, vpath, time.perf_counter() - t_dl0))
                handed_off = True
            except Exception as exc:
                _cleanup_partial_downloads(downloads_dir, item.video_id)
                fail_row = {
                    **item.to_dict(),
                    "stage": "download",
                    "error": str(exc),
                }
                with lock:
                    failed.append(fail_row)
                    _append_manifest_row(manifest_path, {"status": "failed", **fail_row})
            finally:
                if not handed_off:
                    slot_sem.release()

        def _dispatcher() -> None:
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=dl_workers) as pool:
                    futures: list[concurrent.futures.Future[Any]] = []
                    for item in eligible:
                        acquired = False
                        while not stop_event.is_set():
                            if slot_sem.acquire(timeout=0.25):
                                acquired = True
                                break
                        if not acquired:
                            break
                        if stop_event.is_set():
                            slot_sem.release()
                            break
                        futures.append(pool.submit(_download_one, item))
                    for fut in futures:
                        try:
                            fut.result()
                        except Exception:
                            pass
            finally:
                ready_queue.put(None)

        disp_thread = threading.Thread(target=_dispatcher, name="vod-download-dispatcher", daemon=True)
        disp_thread.start()

        s1_feeder_thread: threading.Thread | None = None
        if use_split_stage1:
            def _stage1_one(item: ExtractedRunMetadata, vpath: Path, dl_sec: float) -> None:
                handed_off = False
                try:
                    if not _resolve_pending_deck(item, vpath):
                        return
                    actions_out = actions_dir / f"{item.video_id}_actions.json"
                    if cfg.resume and _has_valid_actions_json(actions_out):
                        s1_sec = 0.0
                        reused = True
                    else:
                        reused = False
                        if stage1_fn is not None:
                            s1_sec = float(stage1_fn(vpath, item, cfg, actions_out))
                        else:
                            s1_sec = _run_stage1_subprocess(
                                vpath,
                                item,
                                cfg,
                                actions_out,
                                threads_per_worker=threads_per_s1,
                                stop_event=stop_event,
                            )
                    stage2_queue.put((item, vpath, dl_sec, actions_out, s1_sec, reused))
                    handed_off = True
                except Exception as exc:
                    fail_row = {
                        **item.to_dict(),
                        "stage": "stage1",
                        "download_sec": round(dl_sec, 2),
                        "error": str(exc),
                    }
                    with lock:
                        failed.append(fail_row)
                        _append_manifest_row(manifest_path, {"status": "failed", **fail_row})
                finally:
                    if not handed_off:
                        if not cfg.keep_videos and vpath.exists():
                            vpath.unlink(missing_ok=True)
                        slot_sem.release()

            def _stage1_feeder() -> None:
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=s1_workers) as s1_pool:
                        s1_futs: list[concurrent.futures.Future[Any]] = []
                        while True:
                            try:
                                pkt = ready_queue.get(timeout=0.5)
                            except queue.Empty:
                                if stop_event.is_set() and not disp_thread.is_alive():
                                    break
                                continue
                            if pkt is None:
                                break
                            item, vpath, dl_sec = pkt
                            if stop_event.is_set():
                                if not cfg.keep_videos and vpath.exists():
                                    vpath.unlink(missing_ok=True)
                                slot_sem.release()
                                continue
                            s1_futs.append(s1_pool.submit(_stage1_one, item, vpath, dl_sec))
                        for fut in s1_futs:
                            try:
                                fut.result()
                            except Exception:
                                pass
                finally:
                    stage2_queue.put(None)

            s1_feeder_thread = threading.Thread(
                target=_stage1_feeder, name="vod-stage1-feeder", daemon=True
            )
            s1_feeder_thread.start()

        try:
            if use_split_stage1:
                while True:
                    try:
                        pkt2 = stage2_queue.get(timeout=0.5)
                    except queue.Empty:
                        if stop_event.is_set() and s1_feeder_thread is not None and not s1_feeder_thread.is_alive():
                            break
                        continue
                    if pkt2 is None:
                        break
                    item, vpath, dl_sec, actions_out, s1_sec, reused = pkt2
                    try:
                        if not _resolve_pending_seed(item, vpath):
                            continue
                        if stage2_fn is not None:
                            stats = dict(stage2_fn(vpath, item, cfg, actions_out))
                            if "stage1_sec" not in stats:
                                stats["stage1_sec"] = round(s1_sec, 2)
                            stats.setdefault("actions_reused", reused)
                        else:
                            stats = process_downloaded_video(
                                vpath,
                                item,
                                cfg,
                                actions_json_override=actions_out,
                                stage1_elapsed_sec=s1_sec,
                                actions_reused_override=reused,
                            )
                        ok_row = {
                            **item.to_dict(),
                            "download_sec": round(dl_sec, 2),
                            **stats,
                        }
                        with lock:
                            processed.append(ok_row)
                            _append_manifest_row(manifest_path, {"status": "processed", **ok_row})
                    except Exception as exc:
                        fail_row = {
                            **item.to_dict(),
                            "stage": "processing",
                            "download_sec": round(dl_sec, 2),
                            "stage1_sec": round(s1_sec, 2),
                            "error": str(exc),
                        }
                        with lock:
                            failed.append(fail_row)
                            _append_manifest_row(manifest_path, {"status": "failed", **fail_row})
                    finally:
                        _release_runtime_memory()
                        if not cfg.keep_videos and vpath.exists():
                            vpath.unlink(missing_ok=True)
                        slot_sem.release()
            else:
                while True:
                    try:
                        pkt = ready_queue.get(timeout=0.5)
                    except queue.Empty:
                        if stop_event.is_set() and not disp_thread.is_alive():
                            break
                        continue
                    if pkt is None:
                        break

                    item, vpath, dl_sec = pkt
                    try:
                        if not _resolve_pending_deck(item, vpath):
                            continue
                        if not _resolve_pending_seed(item, vpath):
                            continue
                        stats = processor(vpath, item, cfg)
                        ok_row = {
                            **item.to_dict(),
                            "download_sec": round(dl_sec, 2),
                            **stats,
                        }
                        processed.append(ok_row)
                        _append_manifest_row(manifest_path, {"status": "processed", **ok_row})
                    except Exception as exc:
                        fail_row = {
                            **item.to_dict(),
                            "stage": "processing",
                            "download_sec": round(dl_sec, 2),
                            "error": str(exc),
                        }
                        with lock:
                            failed.append(fail_row)
                            _append_manifest_row(manifest_path, {"status": "failed", **fail_row})
                    finally:
                        _release_runtime_memory()
                        if not cfg.keep_videos and vpath.exists():
                            vpath.unlink(missing_ok=True)
                        slot_sem.release()
        except KeyboardInterrupt:
            stop_event.set()
            while not ready_queue.empty():
                leftover = ready_queue.get_nowait()
                if leftover is not None:
                    _, lvpath, _ = leftover
                    if not cfg.keep_videos and lvpath.exists():
                        lvpath.unlink(missing_ok=True)
                    slot_sem.release()
            while not stage2_queue.empty():
                leftover2 = stage2_queue.get_nowait()
                if leftover2 is not None:
                    _, lvpath, _, _, _, _ = leftover2
                    if not cfg.keep_videos and lvpath.exists():
                        lvpath.unlink(missing_ok=True)
                    slot_sem.release()
        finally:
            stop_event.set()
            disp_thread.join(timeout=10.0)
            if s1_feeder_thread is not None:
                s1_feeder_thread.join(timeout=15.0)
            if threading.current_thread() is threading.main_thread():
                signal.signal(signal.SIGINT, prev_sigint)

    ts_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = reports_dir / f"batch_report_{ts_tag}.json"
    cfg_dict = asdict(cfg)
    cfg_dict["work_dir"] = str(cfg.work_dir)
    if cfg.cookies_file is not None:
        cfg_dict["cookies_file"] = str(cfg.cookies_file)
    report: dict[str, Any] = {
        "generated_at_utc": ts_tag,
        "elapsed_sec": round(time.perf_counter() - t_start, 2),
        "config": cfg_dict,
        "counts": {
            "total": len(fetched),
            "eligible": len(eligible),
            "processed": len(processed),
            "skipped": len(skipped),
            "failed": len(failed),
        },
        "cleanliness_summary": _aggregate_cleanliness_summary(processed),
        "eligible_preview": [m.to_dict() for m in eligible] if cfg.dry_run else [],
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "report_path": str(report_path),
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    _print_batch_summary(report)
    return report
