"""Temporal End-Round Classifier Workspace.

Detects the implicit ``ActionType.END_ROUND`` timestamp between ``PLAY_HAND``
and ``CASH_OUT`` using only the central ``G.play`` screen region (``play_area`` ROI).
"""

from __future__ import annotations

from typing import Any

from .config import PLAY_AREA_ROI, STATES

__all__ = [
    "PLAY_AREA_ROI",
    "STATES",
    "EndRoundRecognitionPipeline",
    "EndRoundResult",
    "TemporalEndRoundNet",
]


def __getattr__(name: str) -> Any:
    if name in ("EndRoundRecognitionPipeline", "EndRoundResult"):
        from .pipeline import EndRoundRecognitionPipeline, EndRoundResult

        return EndRoundRecognitionPipeline if name == "EndRoundRecognitionPipeline" else EndRoundResult
    if name == "TemporalEndRoundNet":
        from .model import TemporalEndRoundNet

        return TemporalEndRoundNet
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
