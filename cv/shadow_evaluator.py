import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Converts a BGR/BGRA/grayscale image slice to 2D grayscale."""
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def detect_horizontal_edges(image: np.ndarray) -> np.ndarray:
    """
    Runs a Y-directional Sobel edge detection to find horizontal lines,
    and returns the sum of gradients across each row.

    Args:
        image: A pre-sliced numpy array image (ROI).

    Returns:
        A 1D numpy array containing the sum of absolute Y-gradients for each row.
    """
    if image is None or image.size == 0:
        logger.warning("detect_horizontal_edges received an empty or None image.")
        return np.array([])

    height, width = image.shape[:2]

    # Shadow slices are expected to be thin and wide (aspect ratio W/H >> 1)
    if height == 0 or width == 0:
        logger.warning(f"Invalid image dimensions: {width}x{height}.")
    elif width / height <= 2:  # 2:1 aspect ratio or worse
        logger.warning(
            f"Unexpected aspect ratio for horizontal edge detection. Image is {width}x{height} (W <= H). Expected a wide, thin slice."
        )
    elif height < 3 or width < 10:
        logger.warning(
            f"Image slice might be too small for reliable edge detection: {width}x{height}."
        )

    gray = _to_gray(image)

    # Run Sobel in Y direction (dx=0, dy=1) to isolate horizontal edges
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Take absolute value to get magnitude of gradients
    abs_sobel_y = np.abs(sobel_y)

    # Use median instead of mean to ignore sparse noise spikes and only trigger on consistent lines
    row_sums = np.median(abs_sobel_y, axis=1)

    return row_sums


def analyze_shadow_presence(image: np.ndarray, threshold: float = 1000.0) -> bool:
    """
    Analyzes a pre-sliced image to determine if a shadow is present,
    based on finding a strong horizontal edge.

    Args:
        image: A pre-sliced numpy array image directly below the button.
        threshold: The gradient sum threshold to consider a row as a valid shadow edge.
                   This may need to be tuned based on image width and contrast.

    Returns:
        True if a distinct horizontal shadow edge is found (unpressed), False otherwise.
    """
    if image is None or image.size == 0:
        return False

    gray = _to_gray(image)
    row_sums = detect_horizontal_edges(gray)

    if len(row_sums) == 0:
        return False

    # Check both Sobel edge strength and dark-trough-to-background intensity steps
    # so dark Boss Blind backgrounds and slight motion blur don't mask real shadows.
    if len(row_sums) >= 5 and gray.shape[0] >= 3 and gray.shape[1] >= 1:
        # Skip rows 0..1 (button bottom border / 1px dark bevel spikes) on all strips
        # with >= 5 rows while preserving 4-row strips like Sort by Rank (shadow_dh=-6).
        max_gradient = float(np.max(row_sums[2:]))
    else:
        max_gradient = float(np.max(row_sums))

    if max_gradient > threshold:
        return True

    if gray.shape[0] >= 3 and gray.shape[1] >= 1:
        row_medians = np.median(gray, axis=1)
        min_step = max(10.0, threshold * (10.0 / 52.0))
        # Restrict fallback to genuine dark shadows near the button bottom (e.g. Boss Blind);
        # prevents card borders deeper inside a bright unshadowed strip from false-triggering.
        for i in range(min(4, len(row_medians))):
            dark = float(row_medians[i])
            if dark > 85.0:
                continue
            if i > 0 and float(np.max(row_medians[:i])) > 150.0:
                continue
            for j in range(i + 1, min(len(row_medians), i + 4)):
                step = float(row_medians[j]) - dark
                if (
                    step >= min_step
                    and (float(row_medians[j]) + 1.0) / (dark + 1.0) >= 1.45
                ):
                    return True

    return False
