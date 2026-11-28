"""`matcher.find_element_coarse_to_fine` for a batch of frames at once, on a GPU.

The score of a masked TM_CCORR_NORMED match at a position is

    sum(I * T * M) / (|T * M| * sqrt(sum(I * I * M)))

over the template's pixels. OpenCV and this compute the two sums with a float64 FFT, but OpenCV
rounds them to float32, so the scores differ in their seventh digit. A match whose outcome a
difference of `TOLERANCE` could change is not decided here: a score that close to a threshold, a
second position that close to the best one (a flat area of the screen scores the same at each of
its positions, and which of them OpenCV takes is a matter of rounding), or a position where the
frame is black under the template, whose score is 0/0. The result has None for its template,
and the caller asks OpenCV.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import cv2
import numpy as np
from scipy.fft import next_fast_len
import torch

from config import Config
from matcher import TemplateMatch
from template import CompiledTemplateSpec, Template

# OpenCV's scores are within 2e-7 of this module's on every position of the 21,000 coarse maps
# of a 10 minute video.
TOLERANCE = 1e-5
REFINE_PAD = 4

# What `eval_template.eval` needs of one variant of a template in one frame: the match, and its
# top left corner in the frame.
Matched = tuple[TemplateMatch, tuple[int, int]]
# One frame: its BGR image, its grayscale image, and that at half size.
Frame = tuple[np.ndarray, np.ndarray, np.ndarray]


class _Kernel:
    """The pixels of a template under its mask, gray or BGR, as its score needs them."""

    def __init__(self, image: np.ndarray, mask: np.ndarray, device: torch.device) -> None:
        pixels = torch.from_numpy(np.atleast_3d(image)).to(device, torch.float64).permute(2, 0, 1)
        self.mask = (torch.from_numpy(mask).to(device) > 0).double()
        self.template = pixels * self.mask
        self.norm = float(self.template.norm())
        self.height, self.width = image.shape[:2]
        self._spectra: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def spectra(self, size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        """What multiplies the spectrum of an image of `size`, and of its squares, into the two sums of the score."""
        if size not in self._spectra:
            self._spectra[size] = (torch.fft.rfft2(self.template, s=size).conj(), torch.fft.rfft2(self.mask, s=size).conj())
        return self._spectra[size]


def _fast(images: torch.Tensor) -> tuple[int, int]:
    return next_fast_len(images.shape[-2], real=True), next_fast_len(images.shape[-1], real=True)


def _coarse(images: torch.Tensor, kernels: list[_Kernel]) -> tuple[torch.Tensor, torch.Tensor]:
    """`_best_two` of gray images (n, height, width) for each of the gray kernels."""
    images = images.double()
    size = _fast(images)
    templates = torch.stack([kernel.spectra(size)[0][0] for kernel in kernels])
    masks = torch.stack([kernel.spectra(size)[1] for kernel in kernels])
    products = torch.fft.irfft2(torch.fft.rfft2(images, s=size)[:, None] * templates, s=size)
    energy = torch.fft.irfft2(torch.fft.rfft2(images * images, s=size)[:, None] * masks, s=size)
    return _best_two(products, energy, kernels, images.shape[-2:])


def _fine(grays: torch.Tensor, colors: torch.Tensor, gray: _Kernel, bgr: _Kernel) -> tuple[torch.Tensor, torch.Tensor]:
    """`_best_two` of windows as gray (n, height, width) and as BGR (n, height, width, 3) images, for a template's two kernels."""
    images = torch.cat([grays[:, None], colors.permute(0, 3, 1, 2)], dim=1).double()
    size = _fast(images)
    (gray_template, mask), (bgr_template, _) = gray.spectra(size), bgr.spectra(size)
    spectrum = torch.fft.rfft2(images, s=size)
    products = torch.fft.irfft2(torch.stack([spectrum[:, 0] * gray_template[0], (spectrum[:, 1:] * bgr_template).sum(1)], dim=1), s=size)
    squares = images * images
    energy = torch.fft.irfft2(torch.fft.rfft2(torch.stack([squares[:, 0], squares[:, 1:].sum(1)], dim=1), s=size) * mask, s=size)
    return _best_two(products, energy, [gray, bgr], images.shape[-2:])


def _best_two(products: torch.Tensor, energy: torch.Tensor, kernels: list[_Kernel], image: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """The two highest scores of each image for each kernel (n, kernels, 2), and the position (x, y) of the highest.

    `products` and `energy` are the two sums of the score, at every position. The scores of an
    image that is black under a kernel's template somewhere are not numbers.
    """
    rows = image[0] - min(kernel.height for kernel in kernels) + 1
    columns = image[1] - min(kernel.width for kernel in kernels) + 1
    products, energy = products[..., :rows, :columns], energy[..., :rows, :columns]
    norms = torch.tensor([kernel.norm for kernel in kernels], device=products.device)[:, None, None]
    # A sum of squares of whole numbers under 0.5 is 0.
    scores = torch.where(energy < 0.5, torch.nan, products / (energy.sqrt() * norms))
    for at, kernel in enumerate(kernels):
        scores[:, at, image[0] - kernel.height + 1:] = -torch.inf
        scores[:, at, :, image[1] - kernel.width + 1:] = -torch.inf
    scores = scores.flatten(2)
    scores = torch.where(scores.isnan().any(2, keepdim=True), torch.nan, scores)
    if scores.shape[2] == 1:
        scores = torch.cat([scores, torch.full_like(scores, -torch.inf)], dim=2)
    values, places = scores.topk(2, dim=2)
    return values, torch.stack([places[..., 0] % columns, places[..., 0] // columns], dim=-1)


def _padded(images: np.ndarray) -> torch.Tensor:
    """The images, and black ones up to the next power of two of them: cuFFT plans a transform anew for every count."""
    count = 1 << (len(images) - 1).bit_length()
    return torch.from_numpy(np.concatenate([images, np.zeros((count - len(images), *images.shape[1:]), images.dtype)]))


class GpuMatcher:
    def __init__(self, device: str, config: Config) -> None:
        if not config.coarse_to_fine or config.matching_method != cv2.TM_CCORR_NORMED:
            raise ValueError("the GPU matches coarse to fine and by TM_CCORR_NORMED only")
        self.device = torch.device(device)
        self.config = config
        self._kernels: dict[tuple[str, int], tuple[_Kernel, _Kernel, _Kernel]] = {}

    def _kernel(self, template: Template, variant: int, spec: CompiledTemplateSpec) -> tuple[_Kernel, _Kernel, _Kernel]:
        key = (template.name, variant)
        if key not in self._kernels:
            self._kernels[key] = (_Kernel(spec.image_gray_half, spec.mask_half, self.device),
                                  _Kernel(spec.image_gray, spec.mask, self.device),
                                  _Kernel(spec.image_bgr, spec.mask, self.device))
        return self._kernels[key]

    def match(self, frames: Sequence[Frame], templates: Sequence[Template]) -> list[dict[str, list[Matched] | None]]:
        """For each frame and by the template's name, the matches of the template's variants, or None where OpenCV has to decide."""
        height, width = frames[0][1].shape
        jobs = [(template, variant, spec) for template in templates
                for variant, spec in enumerate(template.get_compiled_specs(width, height, self.config.expand_roi_factor))]
        # A variant too small for the coarse pass is OpenCV's, and so is one without a mask:
        # OpenCV scores it by other means.
        undecided = {(at, template.name) for template, _, spec in jobs if not spec.can_coarse or spec.mask is None
                     for at in range(len(frames))}
        jobs = [job for job in jobs if job[2].can_coarse and job[2].mask is not None]
        results: dict[tuple[int, str], dict[int, Matched]] = defaultdict(dict)
        if not jobs:
            return self._decided(frames, templates, results, undecided)

        # The coarse pass, the variants that look at one area of the frame together.
        areas: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
        for at_job, (_, _, spec) in enumerate(jobs):
            areas[spec.hy1, spec.hy2, spec.hx1, spec.hx2].append(at_job)
        halves = _padded(np.stack([frame[2] for frame in frames])).to(self.device)
        coarse = [_coarse(halves[:, y1:y2, x1:x2], [self._kernel(*jobs[at_job])[0] for at_job in members])
                  for (y1, y2, x1, x2), members in areas.items()]
        order = [at_job for members in areas.values() for at_job in members]
        values = torch.cat([value for value, _ in coarse], dim=1).cpu().numpy()
        places = torch.cat([place for _, place in coarse], dim=1).cpu().numpy()

        # The windows the fine pass looks at, by the variant and the window's size.
        windows: dict[tuple[int, int, int], list[tuple[int, int, int]]] = defaultdict(list)
        for column, at_job in enumerate(order):
            template, variant, spec = jobs[at_job]
            threshold = self._threshold(template) - self.config.coarse_margin
            size = spec.template_size
            for at in range(len(frames)):
                best, second = values[at, column]
                if self._unsure(best, threshold):
                    undecided.add((at, template.name))
                elif best < threshold:
                    results[at, template.name][variant] = self._missed(spec, best)
                elif best - second < TOLERANCE:
                    undecided.add((at, template.name))
                else:
                    x0, x1 = _window(2 * int(places[at, column, 0]), size.width, spec.x2 - spec.x1)
                    y0, y1 = _window(2 * int(places[at, column, 1]), size.height, spec.y2 - spec.y1)
                    windows[at_job, y1 - y0, x1 - x0].append((at, spec.x1 + x0, spec.y1 + y0))
        if not windows:
            return self._decided(frames, templates, results, undecided)

        fine = []
        for (at_job, tall, wide), found in windows.items():
            _, gray, bgr = self._kernel(*jobs[at_job])
            grays = _padded(np.stack([frames[at][1][y:y + tall, x:x + wide] for at, x, y in found]))
            colors = _padded(np.stack([frames[at][0][y:y + tall, x:x + wide] for at, x, y in found]))
            fine_values, fine_places = _fine(grays.to(self.device), colors.to(self.device), gray, bgr)
            fine.append((fine_values[:len(found)], fine_places[:len(found), 1]))
        values = iter(torch.cat([value for value, _ in fine]).cpu().numpy())
        places = iter(torch.cat([place for _, place in fine]).cpu().numpy())
        for (at_job, _, _), found in windows.items():
            template, variant, spec = jobs[at_job]
            threshold = self._threshold(template)
            for (at, x, y), ((gray_value, _), (best, second)), place in zip(found, values, places):
                if self._unsure(gray_value, threshold - 0.03):
                    undecided.add((at, template.name))
                elif gray_value < threshold - 0.03:
                    results[at, template.name][variant] = self._missed(spec, gray_value)
                elif self._unsure(best, threshold):
                    undecided.add((at, template.name))
                elif best <= threshold:
                    results[at, template.name][variant] = self._missed(spec, best)
                elif best - second < TOLERANCE:
                    undecided.add((at, template.name))
                else:
                    match = TemplateMatch(True, (int(place[0]), int(place[1])), float(best), None, spec.px_roi, spec.template_size)
                    results[at, template.name][variant] = (match, (x + int(place[0]), y + int(place[1])))
        return self._decided(frames, templates, results, undecided)

    @staticmethod
    def _decided(frames: Sequence[Frame], templates: Sequence[Template], results: dict[tuple[int, str], dict[int, Matched]],
                 undecided: set[tuple[int, str]]) -> list[dict[str, list[Matched] | None]]:
        def variants(at: int, name: str) -> list[Matched] | None:
            if (at, name) in undecided:
                return None
            return [results[at, name][variant] for variant in sorted(results[at, name])]

        return [{template.name: variants(at, template.name) for template in templates} for at in range(len(frames))]

    def _threshold(self, template: Template) -> float:
        return float(template.conf_threshold) if template.conf_threshold is not None else self.config.elem_conf_thresh

    @staticmethod
    def _unsure(score: float, threshold: float) -> bool:
        return not np.isfinite(score) or abs(score - threshold) < TOLERANCE

    @staticmethod
    def _missed(spec: CompiledTemplateSpec, score: float) -> Matched:
        return TemplateMatch(False, (0, 0), float(score), None, spec.px_roi, spec.template_size), (0, 0)


def _window(start: int, size: int, limit: int) -> tuple[int, int]:
    """The fine pass's window along one axis: `REFINE_PAD` around a template at `start`, inside 0..limit."""
    low = max(0, start - REFINE_PAD)
    high = min(limit, start + size + REFINE_PAD)
    if high - low < size:
        low = max(0, min(low, limit - size))
        high = min(limit, low + size)
    return low, high
