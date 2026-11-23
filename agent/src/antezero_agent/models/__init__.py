"""Policy architectures for antezero_agent."""

from __future__ import annotations

import torch

from ..config import ModelConfig
from ..model import Net
from .embed import load_text_embeddings
from .run95_policy import Run95Net


def build_model(cfg: ModelConfig, fx: torch.Tensor | None = None) -> Net:
    if cfg.arch == "run95":
        return Run95Net(cfg, fx)
    if cfg.arch == "antezero":
        return Net(cfg, fx)
    raise ValueError(f"unknown model.arch {cfg.arch!r}")


__all__ = ["Net", "Run95Net", "build_model", "load_text_embeddings"]
