"""Policy architectures for antezero_agent."""

from __future__ import annotations

import torch

from ..config import ModelConfig
from .action_heads import ACTION_HEADS, ActionHead, FactoredActionHead, HandBuilderActionHead, build_action_head
from .embed import load_text_embeddings


def build_model(cfg: ModelConfig, fx: torch.Tensor | None = None):
    if cfg.arch == "run95":
        from .run95_policy import Run95Net

        return Run95Net(cfg, fx)
    if cfg.arch == "antezero":
        from ..model import Net

        return Net(cfg, fx)
    raise ValueError(f"unknown model.arch {cfg.arch!r}")


def __getattr__(name: str):
    if name == "Net":
        from ..model import Net

        return Net
    if name == "Run95Net":
        from .run95_policy import Run95Net

        return Run95Net
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ACTION_HEADS",
    "ActionHead",
    "FactoredActionHead",
    "HandBuilderActionHead",
    "Net",
    "Run95Net",
    "build_action_head",
    "build_model",
    "load_text_embeddings",
]
