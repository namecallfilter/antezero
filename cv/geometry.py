import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class NormalizedROI:
    """Saves ROIs (Regions of Interest) as screen size ratios"""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        bounds = (
            self.x_min,
            self.y_min,
            self.x_max,
            self.y_max,
        )
        if any(c <= 0 for c in bounds):
            raise ValueError(f"At least one invalid ROI boundary: {bounds}.")

        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("ROI minimums must be less than maximums.")


@dataclass(frozen=True, slots=True)
class PixelROI:
    """Saves ROIs (Regions of Interest) as specific pixel counts"""

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def __post_init__(self) -> None:
        bounds = (
            self.x_min,
            self.y_min,
            self.x_max,
            self.y_max,
        )
        if any(c <= 0 for c in bounds):
            raise ValueError(f"At least one invalid ROI boundary: {bounds}.")

        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("ROI minimums must be less than maximums.")

    def expanded_bounds(self, factor: float) -> tuple[int, int, int, int]:
        if factor < 1.0:
            raise ValueError("ROI expansion factor must be at least 1.0")

        center_x = (self.x_min + self.x_max) / 2
        center_y = (self.y_min + self.y_max) / 2
        half_width = self.width * factor / 2
        half_height = self.height * factor / 2

        return (
            round(center_x - half_width),
            round(center_y - half_height),
            round(center_x + half_width),
            round(center_y + half_height),
        )

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    @property
    def top_left(self) -> tuple[int, int]:
        return self.x_min, self.y_min

    @property
    def bottom_right(self) -> tuple[int, int]:
        return self.x_max, self.y_max


@dataclass(frozen=True, slots=True)
class ImageSize:
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Image dimensions must be positive")

    @classmethod
    def from_image(cls, image: np.ndarray) -> "ImageSize":
        height, width = image.shape[:2]
        return cls(width=width, height=height)


def crop_image_to_roi(
    image: np.ndarray,
    roi: PixelROI,
    *,
    copy: bool = False,
) -> np.ndarray:
    """Return the image region described by a PixelROI."""
    cropped = image[roi.y_min : roi.y_max, roi.x_min : roi.x_max]
    return cropped.copy() if copy else cropped
