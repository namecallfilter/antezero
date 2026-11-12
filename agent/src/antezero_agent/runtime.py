"""What training, evaluation and play share: the device, the environment
factory and checkpoints."""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

import antezero_sim as az
import torch

from .config import Config
from .model import Net


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@contextlib.contextmanager
def seeded(seed: int, device="cpu"):
    """Runs the block with Torch's RNG (of the CPU, and of ``device`` if it
    is a GPU) seeded with ``seed``; the previous RNG state is restored."""
    dev = torch.device(device)
    gpus = [dev.index if dev.index is not None else torch.cuda.current_device()] if dev.type == "cuda" else []
    with torch.random.fork_rng(devices=gpus):
        torch.manual_seed(seed)
        yield


def make_env(cfg: Config, seed: int | None = None, num_envs: int | None = None, **over) -> az.VecEnv:
    """A ``VecEnv`` as configured by ``cfg.env`` (``over`` replaces any of
    its keyword arguments)."""
    e = cfg.env
    kw = dict(
        backs=list(e.backs),
        stakes=list(e.stakes),
        seed=e.seed if seed is None else seed,
        endless=e.endless,
        max_ante=e.max_ante,
        step_limit=e.step_limit,
        gamma=cfg.ppo.gamma,
        r_win=e.r_win,
        r_endless=e.r_endless,
        r_loss=e.r_loss,
        w_blind=e.w_blind,
        w_score=e.w_score,
        threads=e.threads,
        play_candidates=e.play_candidates,
        deck_tokens=e.deck_tokens,
        snapshot_prob=e.snapshot_prob,
        snapshot_min_ante=e.snapshot_min_ante,
        allow_skip=e.allow_skip,
        build_previews=e.build_previews,
        terminal_phi=e.terminal_phi,
    )
    kw.update(over)
    return az.VecEnv(num_envs or e.num_envs, **kw)


def obs_schema() -> dict:
    """The observation / action layout the network is built for (stored in
    checkpoints: a policy is only valid for the schema it was trained on)."""
    names = ["MAX_TOKENS", "TOK_INT", "TOK_FLOAT", "GLOB_INT", "GLOB_FLOAT", "MAX_OPS", "OP_INT", "OP_FLOAT", "MAX_HAND", "N_ZONES", "N_CENTERS", "N_OPS_KINDS"]
    return {**{n: getattr(az, n) for n in names}, "TOK_VOCAB": list(az.TOK_VOCAB), "GLOB_VOCAB": list(az.GLOB_VOCAB)}


def save_checkpoint(path: Path, model: Net, opt, cfg: Config, update: int, steps: int, extra: dict | None = None) -> None:
    """Atomically writes a training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ck = {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "config": json.loads(cfg.to_json()),
        "update": update,
        "steps": steps,
        "obs_schema": obs_schema(),
        **(extra or {}),
    }
    tmp = path.with_suffix(".tmp")
    torch.save(ck, tmp)
    os.replace(tmp, path)


def read_checkpoint(path: str | Path) -> dict:
    """A checkpoint's contents (on the CPU), after checking it matches this
    simulator's observation schema."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck.get("obs_schema") != obs_schema():
        raise ValueError(f"{path} was trained on another observation schema than this simulator's")
    return ck


def load_model(path: str | Path, device="cpu") -> tuple[Net, Config, dict]:
    """The network of a checkpoint (in eval mode, on ``device``), its config
    and its metadata (update, steps). ``a.pt+b.pt+...``: their greedy
    :class:`~.ensemble.Ens`, with the first one's config and metadata."""
    if "+" in str(path):
        from .ensemble import Ens

        loaded = [load_model(p, device) for p in str(path).split("+")]
        obs = lambda c: (c.env.play_candidates, c.env.deck_tokens, c.env.build_previews, c.env.allow_skip)  # noqa: E731
        if any(obs(c) != obs(loaded[0][1]) for _, c, _ in loaded):
            raise ValueError(f"{path}: the ensemble's members were trained on different observations")
        return Ens([m for m, _, _ in loaded]).eval(), loaded[0][1], loaded[0][2]
    ck = read_checkpoint(path)
    cfg = Config.from_dict(ck["config"])
    model = Net(cfg.model)
    model.load_state_dict(ck["model"])
    meta = {k: ck[k] for k in ("update", "steps")}
    return model.to(device).eval(), cfg, meta
