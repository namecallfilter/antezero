"""Autonomous YouTube VOD download, metadata extraction, and replay orchestration package."""

from vod_pipeline.deck_matcher import (
    DeckFrameMatch,
    DeckVideoMatch,
    detect_deck_from_video,
    match_deck_in_frame,
)
from vod_pipeline.metadata import (
    BALATRO_SEED_ALPHABET,
    ExtractedRunMetadata,
    expand_input_urls,
    extract_run_metadata,
    extract_video_id,
    fetch_metadata_batch,
    fetch_video_metadata,
)
from vod_pipeline.runner import (
    DEFAULT_VIDEO_FORMAT,
    PipelineConfig,
    download_video,
    process_downloaded_video,
    run_batch_pipeline,
)

__all__ = [
    "BALATRO_SEED_ALPHABET",
    "DEFAULT_VIDEO_FORMAT",
    "DeckFrameMatch",
    "DeckVideoMatch",
    "ExtractedRunMetadata",
    "PipelineConfig",
    "detect_deck_from_video",
    "download_video",
    "expand_input_urls",
    "extract_run_metadata",
    "extract_video_id",
    "fetch_metadata_batch",
    "fetch_video_metadata",
    "match_deck_in_frame",
    "process_downloaded_video",
    "run_batch_pipeline",
]
