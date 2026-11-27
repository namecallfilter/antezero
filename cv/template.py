from dataclasses import dataclass, field
from itertools import product
import json
import logging
from typing import Any

import cv2
from cv2.typing import MatLike
import numpy as np

from geometry import ImageSize, NormalizedROI, PixelROI

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CompiledTemplateSpec:
    """Precomputed contiguous buffers & clamped ROI slices for one variant x ROI."""

    image_bgr: MatLike
    image_gray: MatLike
    mask: MatLike | None
    image_gray_half: MatLike
    mask_half: MatLike | None
    template_size: ImageSize
    px_roi: PixelROI
    x1: int
    y1: int
    x2: int
    y2: int
    hx1: int
    hy1: int
    hx2: int
    hy2: int
    can_coarse: bool
    interior_mask: np.ndarray | None
    template_median_luma: float


@dataclass
class Template:
    name: str
    filenames: list[str]
    norm_rois: list[NormalizedROI]
    px_rois: list[PixelROI]
    images: list[MatLike]
    images_gray: list[MatLike]
    masks: list[MatLike]
    shadow_dy: int
    shadow_dw: int
    shadow_dh: int
    adjusted: bool
    shadow_threshold: float | None = None
    conf_threshold: float | None = None
    min_luminance_ratio: float | None = None
    max_luminance_ratio: float | None = None
    check_shadow: bool = True
    _compiled_specs_cache: dict[tuple[int, int, float], tuple[CompiledTemplateSpec, ...]] = field(
        default_factory=dict, repr=False, compare=False
    )

    def get_compiled_specs(
        self, img_w: int, img_h: int, expand_roi_factor: float
    ) -> tuple[CompiledTemplateSpec, ...]:
        """Returns cached static ROI bounds & contiguous template slices for `(img_w, img_h)`."""
        key = (img_w, img_h, round(expand_roi_factor, 4))
        cached = self._compiled_specs_cache.get(key)
        if cached is not None:
            return cached

        specs: list[CompiledTemplateSpec] = []
        half_w = img_w // 2
        half_h = img_h // 2

        for (image, image_gray, mask), px_roi in product(
            zip(self.images, self.images_gray, self.masks), self.px_rois
        ):
            h_inp, w_inp = image.shape[:2]
            ex1, ey1, ex2, ey2 = px_roi.expanded_bounds(expand_roi_factor)

            y1, y2 = max(0, ey1), min(img_h, ey2)
            x1, x2 = max(0, ex1), min(img_w, ex2)

            if (y2 - y1) < h_inp:
                diff = h_inp - (y2 - y1)
                y1 = max(0, y1 - diff // 2)
                y2 = y1 + h_inp
                if y2 > img_h:
                    y2 = img_h
                    y1 = max(0, y2 - h_inp)
            if (x2 - x1) < w_inp:
                diff = w_inp - (x2 - x1)
                x1 = max(0, x1 - diff // 2)
                x2 = x1 + w_inp
                if x2 > img_w:
                    x2 = img_w
                    x1 = max(0, x2 - w_inp)

            if (y2 - y1) < h_inp or (x2 - x1) < w_inp:
                logger.warning("ROI is too small and cannot be expanded!")
                continue

            # Align full-res ROI start down to an even pixel so half-res coordinates map exactly by 2x.
            x1 &= ~1
            y1 &= ~1
            hx1 = x1 // 2
            hy1 = y1 // 2
            hx2 = min(half_w, (x2 + 1) // 2)
            hy2 = min(half_h, (y2 + 1) // 2)

            im_bgr = np.ascontiguousarray(image[:, :, :3])
            im_gray = np.ascontiguousarray(image_gray)
            im_mask = np.ascontiguousarray(mask) if mask is not None else None
            im_gray_half = np.ascontiguousarray(im_gray[::2, ::2])
            im_mask_half = (
                np.ascontiguousarray(im_mask[::2, ::2]) if im_mask is not None else None
            )

            can_coarse = (
                im_gray_half.shape[0] >= 8
                and im_gray_half.shape[1] >= 8
                and (hy2 - hy1) >= im_gray_half.shape[0]
                and (hx2 - hx1) >= im_gray_half.shape[1]
            )
            interior_mask = (im_mask > 127) if im_mask is not None else None
            if interior_mask is not None and np.any(interior_mask):
                template_median_luma = float(np.median(im_gray[interior_mask]))
            else:
                interior_mask = None
                template_median_luma = float(np.median(im_gray))

            specs.append(
                CompiledTemplateSpec(
                    image_bgr=im_bgr,
                    image_gray=im_gray,
                    mask=im_mask,
                    image_gray_half=im_gray_half,
                    mask_half=im_mask_half,
                    template_size=ImageSize(width=w_inp, height=h_inp),
                    px_roi=px_roi,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    hx1=hx1,
                    hy1=hy1,
                    hx2=hx2,
                    hy2=hy2,
                    can_coarse=can_coarse,
                    interior_mask=interior_mask,
                    template_median_luma=template_median_luma,
                )
            )

        compiled = tuple(specs)
        self._compiled_specs_cache[key] = compiled
        return compiled


def template_loader(
    template_json_path: str, template_img_dir: str, resolution: ImageSize
) -> list[Template]:
    """Load all templates from json."""
    with open(template_json_path, "r") as f:
        TEMPLATES: dict[str, dict[str, Any]] = json.load(f)

    templates = []

    for name, data in TEMPLATES.items():
        filenames, rois = data["filenames"], data["rois"]
        filenames: list[str]
        rois: list[list[float]]

        norm_rois = []
        px_rois = []
        for roi in rois:
            normalized_roi = NormalizedROI(*roi)
            norm_rois.append(normalized_roi)
            px_rois.append(
                PixelROI(
                    round(normalized_roi.x_min * resolution.width),
                    round(normalized_roi.y_min * resolution.height),
                    round(normalized_roi.x_max * resolution.width),
                    round(normalized_roi.y_max * resolution.height),
                )
            )

        shadow_dy = data.get("shadow_dy", 0)
        shadow_dw = data.get("shadow_dw", 0)
        shadow_dh = data.get("shadow_dh", 0)
        raw_thresh = data.get("shadow_threshold", None)
        shadow_threshold = float(raw_thresh) if raw_thresh is not None else None
        raw_conf = data.get("conf_threshold", None)
        conf_threshold = float(raw_conf) if raw_conf is not None else None
        raw_min_luma = data.get("min_luminance_ratio", None)
        min_luminance_ratio = float(raw_min_luma) if raw_min_luma is not None else None
        raw_max_luma = data.get("max_luminance_ratio", None)
        max_luminance_ratio = float(raw_max_luma) if raw_max_luma is not None else None
        check_shadow = bool(data.get("check_shadow", True))
        adjusted = data.get("adjusted", False)

        images = [
            img
            for f in filenames
            if (img := cv2.imread(f"{template_img_dir}/{f}", cv2.IMREAD_UNCHANGED))
            is not None
        ]

        images_gray = []
        masks = []
        for img in images:
            images_gray.append(cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY))
            masks.append(img[:, :, 3] if img.shape[2] == 4 else None)

        templates.append(
            Template(
                name,
                filenames,
                norm_rois,
                px_rois,
                images,
                images_gray,
                masks,
                shadow_dy,
                shadow_dw,
                shadow_dh,
                adjusted,
                shadow_threshold,
                conf_threshold,
                min_luminance_ratio,
                max_luminance_ratio,
                check_shadow,
            )
        )

    return templates
