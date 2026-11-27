"""CLI entry point for autonomous Balatro YouTube VOD batch reconstruction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from vod_pipeline.runner import DEFAULT_VIDEO_FORMAT, PipelineConfig, run_batch_pipeline


def _load_urls(positional: Sequence[str], urls_file: str | None) -> list[str]:
    urls: list[str] = []
    if urls_file:
        with Path(urls_file).open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    urls.append(stripped)
    for item in positional:
        stripped = item.strip()
        if stripped and not stripped.startswith("#"):
            urls.append(stripped)
    return urls


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download YouTube Balatro VODs concurrently (bounded backlog), "
        "extract deck/seed/stake metadata, run main.py Stage 1/2, and clean up processed videos."
    )
    p.add_argument(
        "urls",
        nargs="*",
        help="YouTube video or playlist URLs",
    )
    p.add_argument(
        "--urls-file",
        type=str,
        default=None,
        help="Optional newline-delimited file of YouTube URLs (# comments supported)",
    )
    p.add_argument(
        "--work-dir",
        type=str,
        default="batch_runs",
        help="Gitignored root directory for downloads, metadata, actions, trajectories, and reports",
    )
    p.add_argument(
        "--max-backlog",
        type=int,
        default=5,
        help="Maximum simultaneous downloaded/in-flight videos buffered on disk (default: 5)",
    )
    p.add_argument(
        "--download-workers",
        type=int,
        default=3,
        help="Maximum concurrent yt-dlp downloads (default: 3)",
    )
    p.add_argument(
        "--metadata-workers",
        type=int,
        default=6,
        help="Concurrent workers for yt-dlp metadata pre-scan (default: 6)",
    )
    p.add_argument(
        "--concurrent-fragments",
        type=int,
        default=4,
        help="yt-dlp --concurrent-fragments per video (default: 4)",
    )
    p.add_argument(
        "--video-format",
        type=str,
        default=DEFAULT_VIDEO_FORMAT,
        help="yt-dlp -f format selector (must decode at 1920x1080 for main.py VideoReader)",
    )
    p.add_argument(
        "--keep-videos",
        action="store_true",
        help="Keep downloaded MP4 files after processing instead of deleting them",
    )
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip videos that already have a completed trajectory JSON in work-dir/trajectories",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch metadata, validate deck/seed/stake gating, and print report without downloading",
    )
    p.add_argument(
        "--deck-template-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When deck is missing/ambiguous in metadata, sample a few video frames to template-match the deck back (default: enabled)",
    )
    p.add_argument(
        "--allow-ocr-seed-fallback",
        action="store_true",
        help="When seed is absent from description, allow downloading if seed_ocr.pt exists to read end screen",
    )
    p.add_argument(
        "--stage1-workers",
        type=int,
        default=2,
        help="Concurrent Stage-1 video scanner subprocesses (fed into serial GPU Stage 2; default: 2)",
    )
    p.add_argument(
        "--stage1-threads-per-worker",
        type=int,
        default=None,
        help="Template-matching threads per Stage-1 worker (auto-scaled when omitted)",
    )
    p.add_argument("--yt-dlp-retries", type=int, default=3)
    p.add_argument("--yt-dlp-retry-backoff", type=float, default=3.0)
    p.add_argument("--sleep-requests", type=float, default=None)
    p.add_argument("--sleep-interval", type=float, default=None)
    p.add_argument("--cookies-from-browser", type=str, default=None)
    p.add_argument("--cookies-file", type=str, default=None)
    p.add_argument(
        "--metadata-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse valid saved metadata JSON files in work-dir/metadata (default: enabled)",
    )
    p.add_argument("--skip-frames", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--conf-hand", type=float, default=0.50)
    p.add_argument("--conf-shop", type=float, default=0.15)
    p.add_argument("--conf-ocr", type=float, default=0.90)
    p.add_argument(
        "--min-duration-minutes",
        type=float,
        default=30.0,
        help="Skip videos shorter than this duration in minutes (default: 30; set 0 to disable)",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--expressive", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    urls = _load_urls(args.urls, args.urls_file)
    if not urls:
        parser.error("Provide at least one URL via positional arguments or --urls-file")

    cfg = PipelineConfig(
        work_dir=Path(args.work_dir),
        max_backlog=max(1, int(args.max_backlog)),
        download_workers=max(1, int(args.download_workers)),
        metadata_workers=max(1, int(args.metadata_workers)),
        concurrent_fragments=max(1, int(args.concurrent_fragments)),
        video_format=args.video_format,
        keep_videos=bool(args.keep_videos),
        resume=bool(args.resume),
        dry_run=bool(args.dry_run),
        allow_deck_template_fallback=bool(args.deck_template_fallback),
        allow_ocr_seed_fallback=bool(args.allow_ocr_seed_fallback),
        stage1_workers=max(1, int(args.stage1_workers)),
        stage1_threads_per_worker=args.stage1_threads_per_worker,
        yt_dlp_retries=max(0, int(args.yt_dlp_retries)),
        yt_dlp_retry_backoff_sec=max(0.0, float(args.yt_dlp_retry_backoff)),
        sleep_requests=args.sleep_requests,
        sleep_interval=args.sleep_interval,
        cookies_from_browser=args.cookies_from_browser,
        cookies_file=Path(args.cookies_file) if args.cookies_file else None,
        use_metadata_cache=bool(args.metadata_cache),
        skip_frames=int(args.skip_frames),
        batch_size=int(args.batch_size),
        conf_hand=float(args.conf_hand),
        conf_shop=float(args.conf_shop),
        conf_ocr=float(args.conf_ocr),
        min_duration_sec=max(0.0, float(args.min_duration_minutes) * 60.0),
        device=args.device,
        expressive=bool(args.expressive),
    )
    run_batch_pipeline(urls, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
