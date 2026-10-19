"""Throughput benchmark: environment alone, then full PPO updates (rollout
collection + learning) for a preset on a device. Measure this on the
training machine before choosing a budget.

    uv run python -m antezero_agent.bench --preset base --device cuda --updates 5
    uv run python -m antezero_agent.bench --preset cpu --set env.num_envs=64
"""

from __future__ import annotations

import argparse
import csv
import tempfile
import time
from pathlib import Path

import numpy as np

from .config import parse_overrides, preset
from .runtime import make_env
from .train import train


def env_throughput(cfg, seconds: float = 5.0) -> float:
    """Env steps/s with a trivial policy (first op, no cards; illegal
    composites fall back to a legal action)."""
    env = make_env(cfg)
    N = env.num_envs
    op = np.zeros(N, np.int32)
    cards = np.zeros(N, np.uint32)
    dest = np.zeros(N, np.int32)
    env.step(op, cards, dest)
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        env.step(op, cards, dest)
        n += N
    return n / (time.perf_counter() - t0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="base", choices=["smoke", "cpu", "base", "large"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--updates", type=int, default=4, help="the first one is warm-up and not counted")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args(argv)
    cfg = preset(a.preset)
    cfg.device = a.device
    try:
        parse_overrides(cfg, a.set)
        cfg.validate()
    except ValueError as e:
        ap.error(str(e))
    print(f"env only: {env_throughput(cfg):,.0f} steps/s ({cfg.env.num_envs} envs, candidates={cfg.env.play_candidates})", flush=True)
    with tempfile.TemporaryDirectory() as d:
        cfg.run_dir, cfg.name = d, "bench"
        cfg.save_every = cfg.eval_every = 0
        cfg.ppo.total_steps = 10**15
        train(cfg, max_updates=a.updates)
        with open(Path(d) / "bench" / "log.csv") as f:
            rows = list(csv.DictReader(f))
    rows = rows[1:] or rows
    fps = float(np.mean([float(r["fps"]) for r in rows]))
    roll = float(np.mean([float(r["rollout_s"]) for r in rows]))
    steps = cfg.env.num_envs * cfg.ppo.rollout_len
    print(f"ppo: {fps:,.0f} env-steps/s end to end; rollout {roll:.2f}s per update of {steps:,} steps")
    print(f"=> {1e9 / max(1.0, fps) / 3600:,.1f} hours per 1e9 env steps")


if __name__ == "__main__":
    main()
