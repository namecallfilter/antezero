"""
Evaluates a given template for a given frame.
"""

import json
import logging
from dataclasses import dataclass

import cv2
import cv2.mat_wrapper
import numpy as np

from config import Config
from tools.debug_plotter import plot
from geometry import PixelROI, crop_image_to_roi
from matcher import TemplateMatch, find_element, find_element_coarse_to_fine
from shadow_evaluator import analyze_shadow_presence
from template import Template

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    template_name: str
    found: bool
    match: TemplateMatch | None
    clicked: bool | None
    template_filename: str | None = None
    plot_kwargs: dict | None = None
    box: tuple[int, int, int, int] | None = None
    luminance_ratio: float | None = None


def detect_view_deck_tab_mode(
    img_bgr: np.ndarray, full_deck_box: tuple[int, int, int, int] | list[int]
) -> str:
    """Distinguishes active 'Full Deck' tab (red pointer triangle above button) from 'Remaining'."""
    x1, y1, x2, _ = (int(v) for v in full_deck_box)
    y0 = max(0, y1 - 45)
    y1_strip = max(0, y1 - 2)
    x0 = max(0, x1)
    x1_strip = min(img_bgr.shape[1], x2)
    if y1_strip <= y0 or x1_strip <= x0:
        return "remaining"
    strip = img_bgr[y0:y1_strip, x0:x1_strip]
    red_px = int(
        np.sum(
            (strip[:, :, 2] > 180) & (strip[:, :, 1] < 90) & (strip[:, :, 0] < 90)
        )
    )
    return "full_deck" if red_px >= 40 else "remaining"


def format_timestamp(timestamp_seconds: float, frame_index: int) -> str:
    sec = int(timestamp_seconds)
    hh = sec // 3600
    mm = (sec % 3600) // 60
    ss = sec % 60
    ff = frame_index % 60
    return f"{timestamp_seconds:.2f}s | {hh:02}:{mm:02}:{ss:02}:{ff:02}"


def eval(
    template: Template,
    img: cv2.mat_wrapper.Mat,
    img_gray: cv2.mat_wrapper.Mat,
    config: Config,
    video_timestamp_str: str = "",
    img_gray_half: cv2.mat_wrapper.Mat | None = None,
) -> EvaluationResult:
    if img is None or img_gray is None:
        raise ValueError("Got None for image")

    img_h, img_w = img.shape[:2]
    should_plot = config.debug_plot or (config.setup_mode and not template.adjusted)
    use_coarse = config.coarse_to_fine and not should_plot

    if use_coarse and img_gray_half is None and img_w >= 16 and img_h >= 16:
        img_gray_half = cv2.resize(
            img_gray, (img_w // 2, img_h // 2), interpolation=cv2.INTER_NEAREST
        )

    debug_enabled = logger.isEnabledFor(logging.DEBUG)
    info_enabled = logger.isEnabledFor(logging.INFO)
    eff_conf_thresh = (
        float(template.conf_threshold)
        if getattr(template, "conf_threshold", None) is not None
        else config.elem_conf_thresh
    )
    specs = template.get_compiled_specs(img_w, img_h, config.expand_roi_factor)

    for spec in specs:
        w_inp = spec.template_size.width
        h_inp = spec.template_size.height
        x1, y1, x2, y2 = spec.x1, spec.y1, spec.x2, spec.y2

        roi_img_gray = img_gray[y1:y2, x1:x2]
        roi_img = img[y1:y2, x1:x2]

        if use_coarse and spec.can_coarse and img_gray_half is not None:
            match, (dx, dy) = find_element_coarse_to_fine(
                roi_img_gray,
                roi_img,
                spec.image_gray,
                spec.image_bgr,
                spec.mask,
                img_gray_half[spec.hy1 : spec.hy2, spec.hx1 : spec.hx2],
                spec.image_gray_half,
                spec.mask_half,
                spec.template_size,
                config.matching_method,
                spec.px_roi,
                eff_conf_thresh,
                coarse_margin=config.coarse_margin,
            )
            top_left = (match.top_left[0] + x1 + dx, match.top_left[1] + y1 + dy)
        else:
            match = find_element(
                roi_img_gray,
                spec.image_gray,
                spec.mask,
                roi_img,
                spec.image_bgr,
                spec.template_size,
                config.matching_method,
                spec.px_roi,
                eff_conf_thresh,
            )
            top_left = (match.top_left[0] + x1, match.top_left[1] + y1)

        luminance_ratio: float | None = None
        if match.found:
            matched_gray = img_gray[
                top_left[1] : top_left[1] + h_inp,
                top_left[0] : top_left[0] + w_inp,
            ]
            if (
                spec.interior_mask is not None
                and matched_gray.shape == spec.interior_mask.shape
            ):
                roi_med = float(np.median(matched_gray[spec.interior_mask]))
            elif matched_gray.size > 0:
                roi_med = float(np.median(matched_gray))
            else:
                roi_med = spec.template_median_luma
            luminance_ratio = (
                roi_med / spec.template_median_luma
                if spec.template_median_luma > 0.0
                else 1.0
            )
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
            luma_ok = (not config.use_luminance_check) or (
                eff_min_luma <= luminance_ratio <= eff_max_luma
            )

            bottom_left = (top_left[0], top_left[1] + h_inp)
            bottom_right = (top_left[0] + w_inp, top_left[1] + h_inp)

            dy = template.shadow_dy
            dw = template.shadow_dw
            dh = template.shadow_dh
            height = config.shadow_region_height
            shadow_region = PixelROI(
                x_min=bottom_left[0],
                y_min=bottom_left[1] + dy,
                x_max=bottom_right[0] + dw,
                y_max=bottom_right[1] + height + dy + dh,
            )

            eff_shadow_threshold = (
                template.shadow_threshold
                if getattr(template, "shadow_threshold", None) is not None
                else config.shadow_threshold
            )
            if not getattr(template, "check_shadow", True):
                clicked = True
            elif not luma_ok:
                clicked = False
            else:
                shadow_img_gray = crop_image_to_roi(img_gray, shadow_region)
                has_shadow = analyze_shadow_presence(
                    shadow_img_gray, threshold=eff_shadow_threshold
                )
                clicked = not has_shadow

            if info_enabled:
                logger.info(
                    f"FOUND {template.name} at {top_left} with confidence of {match.confidence:.4f}; "
                    f"luma_ratio={luminance_ratio:.3f}; Clicked={clicked}"
                )
        else:
            if debug_enabled:
                logger.debug(
                    f"Did not find {template.name} in ROI; highest confidence of {match.confidence:.4f}"
                )
            clicked = False

        if should_plot and match.found:
            plot_kwargs = {
                "img": img,
                "template": spec.image_bgr,
                "roi": (x1, y1, x2, y2),
                "found_target": match.found,
                "match": match.score_map,
                "top_left": top_left,
                "max_val": match.confidence,
                "clicked": clicked,
                "label": template.name,
                "shadow_threshold": eff_shadow_threshold,
                "template_name": template.name,
                "template_json_path": config.template_json_path,
                "current_shadow_dy": template.shadow_dy,
                "current_shadow_dw": template.shadow_dw,
                "current_shadow_dh": template.shadow_dh,
                "shadow_height": config.shadow_region_height,
                "video_timestamp_str": video_timestamp_str,
            }
            if match.found:
                plot_kwargs["shadow_region"] = shadow_region  # type:ignore

            plot(**plot_kwargs)

            try:
                with open(config.template_json_path, "r", encoding="utf-8") as f:
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
                logger.error(f"Failed to reload template data for {template.name}: {e}")

        if match.found:
            return EvaluationResult(
                template_name=template.name,
                found=True,
                match=match,
                clicked=clicked,
                template_filename=None,
                box=(
                    top_left[0],
                    top_left[1],
                    top_left[0] + w_inp,
                    top_left[1] + h_inp,
                ),
                luminance_ratio=luminance_ratio,
            )

    return EvaluationResult(
        template_name=template.name,
        found=False,
        match=None,
        clicked=None,
        template_filename=None,
    )
