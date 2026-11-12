"""Faster rollout collection for CUDA training (``ppo.fast_rollout``).

1: the policy's moves are replayed from CUDA graphs (one per padded token count),
   observations are uploaded once per step from pinned memory (the first T tokens,
   int32 on the wire) and the actions come back in one copy. Same samples, log-probs,
   values and observations as train.py's loop, bitwise (the noise is drawn eagerly,
   in act's order and shapes, before each replay).
2: also two half-size VecEnvs: one steps (a worker thread; the simulator releases the
   GIL) while the GPU picks for the other. Same training math, other games (each half
   has its own seeds) and batch-256 forward passes, so not bitwise.

Measured on an RTX 4090 + EPYC 9684X (u13200, 512 envs x 128): rollout 4.61 s ->
4.09 s (1) and 3.08 s (2); see .scratch/speed/rollout/REPORT.md.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import antezero_sim as az
import numpy as np
import torch

from .config import Config
from .model import MAX_HAND, Net, Obs, noise_shapes, obs_from_numpy
from .runtime import make_env

_INT32 = ("tok_i", "glob_i", "ops", "n_ops", "n_tokens", "n_hand")


def _shapes(n: int) -> dict[str, tuple[int, ...]]:
    return {
        "tok_i": (n, az.MAX_TOKENS, az.TOK_INT),
        "tok_f": (n, az.MAX_TOKENS, az.TOK_FLOAT),
        "glob_i": (n, az.GLOB_INT),
        "glob_f": (n, az.GLOB_FLOAT),
        "ops": (n, az.MAX_OPS, az.OP_INT),
        "ops_f": (n, az.MAX_OPS, az.OP_FLOAT),
        "n_ops": (n,),
        "n_tokens": (n,),
        "n_hand": (n,),
    }


class _Upload:
    """Pinned staging and fixed device buffers for one env's observation (its first T
    tokens). The device buffers are rewritten every step; stream order protects their
    readers, and the action copy-back syncs every step before the staging is reused."""

    def __init__(self, n: int, device):
        self.n = n
        self.h, self.d = {}, {}
        for k, s in _shapes(n).items():
            dt = torch.int32 if k in _INT32 else torch.float32
            self.h[k] = torch.empty(int(np.prod(s)), dtype=dt, pin_memory=True)
            self.d[k] = torch.empty(int(np.prod(s)), dtype=dt, device=device)

    def __call__(self, env: az.VecEnv) -> int:
        """Uploads ``env``'s observation; returns its max token count."""
        t_max = int(np.asarray(env.n_tokens).max())
        T = max(t_max, MAX_HAND, 1)
        for k in self.d:
            src = np.asarray(getattr(env, k))
            if k in ("tok_i", "tok_f"):
                src = src[:, :T]
            hv = self.h[k][: src.size].view(src.shape)
            np.copyto(hv.numpy(), src)
            self.d[k][: src.size].view(src.shape).copy_(hv, non_blocking=True)
        return t_max

    def views(self, t_max: int, host: bool = False) -> dict[str, torch.Tensor]:
        """The last upload (int32 / float32, T tokens): its device buffers, or (``host``) its pinned staging."""
        T = max(t_max, MAX_HAND, 1)
        buf = self.h if host else self.d
        out = {}
        for k, s in _shapes(self.n).items():
            s = (s[0], T, s[2]) if k in ("tok_i", "tok_f") else s
            out[k] = buf[k][: int(np.prod(s))].view(s)
        return out


def _obs(v: dict[str, torch.Tensor]) -> Obs:
    return Obs(**{k: x.long() if k in _INT32 else x for k, x in v.items()})


class _GraphAct:
    """``Net.act`` (sampling) on an ``_Upload``'s buffers, replayed from CUDA graphs:
    one per padded token count, captured on first use into a shared memory pool."""

    def __init__(self, model: Net, up: _Upload, explore: float, critic=None):
        self.m, self.up, self.explore, self.critic = model, up, explore, critic
        dev = up.d["tok_f"].device
        self.spec = noise_shapes(up.n, explore)
        self.qs = [torch.empty(s, device=dev) for s, _ in self.spec]
        self.graphs: dict[int, tuple[torch.cuda.CUDAGraph, dict]] = {}
        self.pool = torch.cuda.graph_pool_handle()

    def _act(self, T: int) -> dict:
        o = _obs(self.up.views(T))
        out = self.m.act(o, explore=self.explore, T=T, noise=self.qs)
        if self.critic is not None:  # in the same replay: the device is idle while the simulator steps
            out["c_val"] = self.critic.value(o, T).float()
        # One copy back: op, card mask (u32 bits), destination.
        out["acts"] = torch.stack([out["op"].to(torch.int32), out["cards"].to(torch.int32), out["dest"].to(torch.int32)])
        return out

    def __call__(self, t_max: int) -> dict:
        T = max(t_max, MAX_HAND, 1)  # graphs depend on the padded count only
        if T not in self.graphs:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):  # warm-up: compiles, allocations, cuBLAS handles
                    self._act(T)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                out = self._act(T)
            self.graphs[T] = (g, out)
        g, out = self.graphs[T]
        for (_s, uniform), q in zip(self.spec, self.qs):  # act's own draws, in its order
            q.uniform_() if uniform else q.exponential_(1)
        g.replay()
        return out


class Collector:
    """Fills a ``Rollout`` with ``T`` steps of ``cfg.env.num_envs`` games, like train.py's
    loop (see the module docstring for ``mode``)."""

    def __init__(self, cfg: Config, model: Net, device, seed: int, mode: int, value_of, critic=None):
        """``value_of(obs) -> [B]``: the value truncated episodes bootstrap from. ``critic``: a frozen
        critic whose value of every state is stored with the rollout (``roll.c_val``)."""
        self.cfg, self.m, self.dev, self.value_of = cfg, model, device, value_of
        N = cfg.env.num_envs
        halves = 2 if mode == 2 else 1
        if N % halves:
            raise ValueError("ppo.fast_rollout=2 needs an even env.num_envs")
        nb = N // halves
        self.parts = [slice(h * nb, (h + 1) * nb) for h in range(halves)]
        self.envs = [make_env(cfg, seed=seed + 7_000_001 * h, num_envs=nb) for h in range(halves)]
        self.up = _Upload(nb, device)
        self.act = _GraphAct(model, self.up, cfg.ppo.explore_redeem, critic)
        self.acts = [torch.empty(3, nb, dtype=torch.int32, pin_memory=True) for _ in self.parts]
        T = cfg.ppo.rollout_len
        self.rew = torch.empty(T, N, dtype=torch.float32, pin_memory=True)
        self.done = torch.empty(T, N, dtype=torch.float32, pin_memory=True)
        self.pool = ThreadPoolExecutor(1) if halves > 1 else None
        self.roll, self.width = None, {}  # the rollout being filled; token slots last written per (step, half)

    def invalid_actions(self) -> int:
        return sum(e.invalid_actions() for e in self.envs)

    def last_obs(self) -> list[Obs]:
        return [obs_from_numpy(e, self.dev) for e in self.envs]

    def _pick(self, roll, t: int, h: int):
        """Uploads half ``h``'s observation, stores it and the sampled actions at step ``t``."""
        sl, env = self.parts[h], self.envs[h]
        t_max = self.up(env)
        out = self.act(t_max)
        a = self.acts[h]
        a.copy_(out["acts"], non_blocking=True)
        # Rows in host memory (ppo.rollout_obs_on_cpu) are copied from the pinned staging, on the CPU while the GPU picks.
        v, hv = self.up.views(t_max), self.up.views(t_max, host=True)
        T = v["tok_i"].shape[1]
        if roll is not self.roll:
            self.roll, self.width = roll, {}
        W = self.width.get((t, h), az.MAX_TOKENS)  # the slots past the last write here are already zero
        self.width[t, h] = T
        for k in ("tok_i", "tok_f", "glob_i", "glob_f", "ops", "ops_f", "n_ops", "n_tokens", "n_hand"):
            x = getattr(roll, k)[t, sl]
            src = (hv if x.is_cpu else v)[k]
            if k in ("tok_i", "tok_f"):
                x[:, :T], x[:, T:W] = src, 0  # zeros past T: what the simulator writes there
            else:
                x.copy_(src)
        roll.op[t, sl], roll.card_seq[t, sl], roll.dest[t, sl] = out["op"], out["card_seq"], out["dest"]
        roll.explored[t, sl], roll.logp[t, sl], roll.value[t, sl] = out["explored"], out["logp"].float(), out["value"].float()
        roll.sure[t, sl] = out["sure"].float()
        if "c_val" in out:
            roll.c_val[t, sl] = out["c_val"]
        torch.cuda.current_stream().synchronize()
        return a[0].numpy(), a[1].numpy().view(np.uint32), a[2].numpy()

    def _store(self, t: int, h: int, res, boot: list, on_episodes) -> None:
        rew, done, trunc = res
        sl, env = self.parts[h], self.envs[h]
        self.rew[t, sl] = torch.from_numpy(rew)
        self.done[t, sl] = torch.from_numpy(done.astype(np.float32))
        if trunc.any():
            # Not a real end: bootstrap from the final state's value (added after the upload).
            rows = np.flatnonzero(trunc)
            boot.append((t, sl.start + rows, self.value_of(obs_from_numpy(env, self.dev, final=True, rows=rows))))
        on_episodes(env.pop_episodes())

    @torch.no_grad()
    def collect(self, roll, on_episodes) -> None:
        """Steps every env ``roll.T`` times; ``on_episodes(list)`` gets the finished episodes."""
        gamma, boot = self.cfg.ppo.gamma, []
        futs = [None] * len(self.parts)
        for t in range(roll.T):
            for h in range(len(self.parts)):
                if futs[h] is not None:
                    self._store(t - 1, h, futs[h].result(), boot, on_episodes)
                a = self._pick(roll, t, h)
                if self.pool is None:
                    self._store(t, h, self.envs[h].step(*a), boot, on_episodes)
                else:
                    futs[h] = self.pool.submit(self.envs[h].step, *a)
        for h, f in enumerate(futs):
            if f is not None:
                self._store(roll.T - 1, h, f.result(), boot, on_episodes)
        roll.reward.copy_(self.rew, non_blocking=True)
        roll.done.copy_(self.done, non_blocking=True)
        for t, rows, v_fin in boot:
            roll.reward[t, torch.as_tensor(rows, device=self.dev)] += gamma * v_fin
        torch.cuda.current_stream().synchronize()  # the pinned buffers are rewritten next rollout
