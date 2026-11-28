from dataclasses import dataclass

import cv2
from cv2.typing import MatLike

from geometry import ImageSize, PixelROI


@dataclass(frozen=True, slots=True)
class TemplateMatch:
    found: bool
    top_left: tuple[int, int]
    confidence: float
    score_map: MatLike | None
    roi: PixelROI
    template_size: ImageSize


def find_element(
    img_gray,
    template_gray,
    mask,
    img_bgr,
    template_bgr,
    template_size,
    method,
    roi: PixelROI,
    confidence_threshold: float,
) -> TemplateMatch:
    if mask is not None:
        result: MatLike = cv2.matchTemplate(img_gray, template_gray, method, mask=mask)
    else:
        result: MatLike = cv2.matchTemplate(img_gray, template_gray, method)

    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    # Grayscale fast reject (allow 0.03 margin of error for color differences)
    if max_val < confidence_threshold - 0.03:
        return TemplateMatch(False, tuple(max_loc), max_val, result, roi, template_size)

    # Second pass BGR for true confidence
    if mask is not None:
        result_bgr = cv2.matchTemplate(img_bgr, template_bgr, method, mask=mask)
    else:
        result_bgr = cv2.matchTemplate(img_bgr, template_bgr, method)

    _, max_val_bgr, _, max_loc_bgr = cv2.minMaxLoc(result_bgr)
    found_target = max_val_bgr > confidence_threshold

    return TemplateMatch(found_target, tuple(max_loc_bgr), max_val_bgr, result_bgr, roi, template_size)


def find_element_coarse_to_fine(
    img_gray_roi: MatLike,
    img_bgr_roi: MatLike,
    template_gray: MatLike,
    template_bgr: MatLike,
    mask: MatLike | None,
    img_gray_half_roi: MatLike,
    template_gray_half: MatLike,
    mask_half: MatLike | None,
    template_size: ImageSize,
    method: int,
    roi: PixelROI,
    confidence_threshold: float,
    coarse_margin: float = 0.04,
    refine_pad: int = 4,
) -> tuple[TemplateMatch, tuple[int, int]]:
    """Pre-screens at 2x downscale and refines full-res matching inside a small window.

    Returns `(match, (dx, dy))` where `(dx, dy)` is the refinement window origin relative
    to `img_gray_roi`.
    """
    if mask_half is not None:
        c_res = cv2.matchTemplate(img_gray_half_roi, template_gray_half, method, mask=mask_half)
    else:
        c_res = cv2.matchTemplate(img_gray_half_roi, template_gray_half, method)

    _, c_max, _, c_loc = cv2.minMaxLoc(c_res)
    if c_max < confidence_threshold - coarse_margin:
        return (
            TemplateMatch(
                False,
                (c_loc[0] * 2, c_loc[1] * 2),
                c_max,
                None,
                roi,
                template_size,
            ),
            (0, 0),
        )

    roi_h, roi_w = img_gray_roi.shape[:2]
    w_inp, h_inp = template_size.width, template_size.height
    cx = c_loc[0] * 2
    cy = c_loc[1] * 2

    rx0 = max(0, cx - refine_pad)
    rx1 = min(roi_w, cx + w_inp + refine_pad)
    if (rx1 - rx0) < w_inp:
        rx0 = max(0, min(rx0, roi_w - w_inp))
        rx1 = min(roi_w, rx0 + w_inp)

    ry0 = max(0, cy - refine_pad)
    ry1 = min(roi_h, cy + h_inp + refine_pad)
    if (ry1 - ry0) < h_inp:
        ry0 = max(0, min(ry0, roi_h - h_inp))
        ry1 = min(roi_h, ry0 + h_inp)

    match = find_element(
        img_gray_roi[ry0:ry1, rx0:rx1],
        template_gray,
        mask,
        img_bgr_roi[ry0:ry1, rx0:rx1],
        template_bgr,
        template_size,
        method,
        roi,
        confidence_threshold,
    )
    return match, (rx0, ry0)
