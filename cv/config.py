import argparse
from dataclasses import dataclass, field
import logging
import os

import cv2

from geometry import ImageSize


def _default_max_workers() -> int:
    # Match physical core count to avoid SMT/GIL contention with the NVDEC reader thread.
    return max(1, min(8, (os.cpu_count() or 8) // 2))


@dataclass
class Config:
    template_json_path: str = "assets/templates/drspectred/templates.json"
    template_img_dir: str = "assets/templates/drspectred"
    input_img_path: str = "assets/sample_inputs/scraped_frames/test_frame_1.jpg"
    input_video_path: str = "assets/sample_inputs/videos/bu_anaglyph_deck_10-31-26.mp4"
    output_dir: str = "output"
    resolution_width: int = 1920
    resolution_height: int = 1080
    matching_method: int = cv2.TM_CCORR_NORMED

    elem_conf_thresh: float = 0.92  # found element? confidence threshold
    expand_roi_factor: float = 1.1
    shadow_region_height: int = 10  # pixels
    shadow_threshold: float = 52.0
    use_luminance_check: bool = True
    min_luminance_ratio: float = 0.70
    max_luminance_ratio: float = 1.30
    skip_frames: int = 1  # skip n frames then use 1

    start_timestamp: float | None = None
    end_timestamp: float | None = None

    use_hwaccel: bool = True
    coarse_to_fine: bool = True
    coarse_margin: float = 0.04
    max_workers: int = field(default_factory=_default_max_workers)
    # A torch device ("cuda") that matches the templates of a batch of frames at once, in
    # place of OpenCV on the CPU (`matcher_gpu`).
    match_device: str | None = None

    debug_plot: bool = False
    setup_mode: bool = False
    run_benchmark: bool = True

    @property
    def resolution(self) -> ImageSize:
        return ImageSize(width=self.resolution_width, height=self.resolution_height)


def get_config() -> Config:
    default_cfg = Config()
    parser = argparse.ArgumentParser(description="Botlatro Video Processing Config")
    parser.add_argument(
        "--template-json", type=str, default=default_cfg.template_json_path
    )
    parser.add_argument(
        "--template-dir", type=str, default=default_cfg.template_img_dir
    )
    parser.add_argument("--input-img", type=str, default=default_cfg.input_img_path)
    parser.add_argument("--input-video", type=str, default=default_cfg.input_video_path)
    parser.add_argument(
        "--start-timestamp", type=float, default=default_cfg.start_timestamp
    )
    parser.add_argument(
        "--end-timestamp", type=float, default=default_cfg.end_timestamp
    )
    parser.add_argument("--skip-frames", type=int, default=default_cfg.skip_frames)
    parser.add_argument(
        "--luminance-check",
        action=argparse.BooleanOptionalAction,
        default=default_cfg.use_luminance_check,
    )
    parser.add_argument(
        "--min-luminance-ratio",
        type=float,
        default=default_cfg.min_luminance_ratio,
    )
    parser.add_argument(
        "--max-luminance-ratio",
        type=float,
        default=default_cfg.max_luminance_ratio,
    )
    parser.add_argument(
        "--use-hwaccel",
        action=argparse.BooleanOptionalAction,
        default=default_cfg.use_hwaccel,
    )
    parser.add_argument(
        "--coarse-to-fine",
        action=argparse.BooleanOptionalAction,
        default=default_cfg.coarse_to_fine,
    )
    parser.add_argument(
        "--coarse-margin", type=float, default=default_cfg.coarse_margin
    )
    parser.add_argument("--max-workers", type=int, default=default_cfg.max_workers)
    parser.add_argument(
        "--debug-plot",
        action=argparse.BooleanOptionalAction,
        default=default_cfg.debug_plot,
    )
    parser.add_argument(
        "--setup-mode",
        action=argparse.BooleanOptionalAction,
        default=default_cfg.setup_mode,
    )
    parser.add_argument(
        "--run-benchmark",
        action=argparse.BooleanOptionalAction,
        default=default_cfg.run_benchmark,
    )

    args, unknown = parser.parse_known_args()

    return Config(
        template_json_path=args.template_json,
        template_img_dir=args.template_dir,
        input_img_path=args.input_img,
        input_video_path=args.input_video,
        start_timestamp=args.start_timestamp,
        end_timestamp=args.end_timestamp,
        skip_frames=args.skip_frames,
        use_luminance_check=args.luminance_check,
        min_luminance_ratio=args.min_luminance_ratio,
        max_luminance_ratio=args.max_luminance_ratio,
        use_hwaccel=args.use_hwaccel,
        coarse_to_fine=args.coarse_to_fine,
        coarse_margin=args.coarse_margin,
        max_workers=max(1, int(args.max_workers)),
        debug_plot=args.debug_plot,
        setup_mode=args.setup_mode,
        run_benchmark=args.run_benchmark,
    )


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="[%(levelname)s] %(message)s",
    )
