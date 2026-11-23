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
        practice_prob=e.practice_prob,
        practice_eternal=e.practice_eternal,
        allow_skip=e.allow_skip,
        null_sorts=e.null_sorts,
        build_previews=e.build_previews,
        terminal_phi=e.terminal_phi,
        order_gain=e.order_gain,
        order_label_rate=e.order_label_rate,
        obs_v2=e.obs_v2,
        max_rearranges=e.max_rearranges,
    )
    kw.update(over)
    return az.VecEnv(num_envs or e.num_envs, **kw)


def obs_schema(obs_v2: bool = False) -> dict:
    """The observation / action layout the network is built for (stored in
    checkpoints: a policy is only valid for the schema it was trained on)."""
    names = ["MAX_TOKENS", "TOK_INT", "TOK_FLOAT", "GLOB_INT", "GLOB_FLOAT", "MAX_OPS", "OP_INT", "OP_FLOAT", "MAX_HAND", "N_ZONES", "N_CENTERS", "N_OPS_KINDS"]
    d = {**{n: getattr(az, n) for n in names}, "TOK_VOCAB": list(az.TOK_VOCAB), "GLOB_VOCAB": list(az.GLOB_VOCAB)}
    if obs_v2:
        d["GLOB_FLOAT"] = az.GLOB_FLOAT_V2
    return d


def save_checkpoint(path: Path, model: Net, opt, cfg: Config, update: int, steps: int, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ck = {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "config": json.loads(cfg.to_json()),
        "update": update,
        "steps": steps,
        "obs_schema": obs_schema(cfg.model.obs_v2),
        **(extra or {}),
    }
    tmp = path.with_suffix(".tmp")
    torch.save(ck, tmp)
    os.replace(tmp, path)


def read_checkpoint(path: str | Path) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    obs_v2 = bool(ck.get("config", {}).get("model", {}).get("obs_v2", False))
    if ck.get("obs_schema") != obs_schema(obs_v2):
        raise ValueError(f"{path} was trained on another observation schema than this simulator's")
    return ck


def load_model(path: str | Path, device="cpu") -> tuple[Net, Config, dict]:
    ck = read_checkpoint(path)
    cfg = Config.from_dict(ck["config"])
    model = Net(cfg.model, ck["model"].get("fx"))
    model.load_state_dict(ck["model"])
    meta = {k: ck[k] for k in ("update", "steps")}
    return model.to(device).eval(), cfg, meta
