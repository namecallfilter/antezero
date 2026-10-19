"""Small statistics helpers shared by training and evaluation."""

from __future__ import annotations

import math


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% (by default) Wilson score interval for ``k`` successes in ``n``."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def episodes_needed(p: float, half_width: float, z: float = 1.96) -> int:
    """Episodes for a +-``half_width`` interval around a win rate ``p``."""
    return math.ceil(z * z * p * (1 - p) / (half_width * half_width))
