"""PPO training of the antezero transformer on the Rust simulator.

    uv run python -m antezero_agent.train --preset base --name run1
    uv run python -m antezero_agent.train --preset cpu --set env.num_envs=128 ppo.lr=2e-4
    uv run python -m antezero_agent.train --resume agent/runs/run1/last.pt

Rollouts come from ``antezero_sim.VecEnv`` (parallel Rust games writing
numpy buffers); the policy samples factored actions (op, card set,
destination) and PPO uses teacher-forced log-probabilities of the stored
actions. Truncated episodes (step limit, endless ante cap) bootstrap from
the value of their final observation; only real game ends are terminal.

A resumed run continues the model, optimizer, schedule, Torch RNG and stake
curriculum; the environments restart on fresh seeds (their states are not
saved).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from collections import deque
from pathlib import Path

import antezero_sim as az
import numpy as np
import torch

from .collect import Collector
from .config import Config, parse_overrides, preset
from .critic import Critic, critic_state
from .model import MAX_CARDS, Net, Obs, count_params, obs_from_numpy
from .runtime import make_env, pick_device, read_checkpoint, save_checkpoint
from .stats import wilson


class Rollout:
    """Fixed-size on-device storage for ``T`` steps of ``N`` envs. Integer
    features use compact (lossless) dtypes; floats stay float32 so training
    sees exactly the observations the actions were sampled on.

    ``obs_on_cpu`` (CUDA devices; ``ppo.rollout_obs_on_cpu``): the tokens and
    ops (33.0 of the 33.9 KB per state) live in host memory instead, and
    :meth:`obs` gathers them there."""

    def __init__(self, T: int, N: int, device: torch.device, obs_on_cpu: bool = False):
        z = lambda *s, dt: torch.zeros(*s, dtype=dt, device=device)  # noqa: E731
        self.host = obs_on_cpu and device.type == "cuda"
        # Pageable: torch rounds pinned blocks up to a power of two (tok_f: 11.7 -> 16 GiB at 4,096 x 128).
        big = (lambda *s, dt: torch.zeros(*s, dtype=dt)) if self.host else z  # noqa: E731
        self.tok_i = big(T, N, az.MAX_TOKENS, az.TOK_INT, dt=torch.int16)
        self.tok_f = big(T, N, az.MAX_TOKENS, az.TOK_FLOAT, dt=torch.float32)
        self.glob_i = z(T, N, az.GLOB_INT, dt=torch.int16)
        self.glob_f = z(T, N, az.GLOB_FLOAT, dt=torch.float32)
        self.ops = big(T, N, az.MAX_OPS, az.OP_INT, dt=torch.int32)
        self.ops_f = big(T, N, az.MAX_OPS, az.OP_FLOAT, dt=torch.float32)
        self.n_ops = z(T, N, dt=torch.int16)
        self.n_tokens = z(T, N, dt=torch.int16)
        self.n_hand = z(T, N, dt=torch.int16)
        self.op = z(T, N, dt=torch.int64)
        self.card_seq = z(T, N, MAX_CARDS, dt=torch.int64)
        self.dest = z(T, N, dt=torch.int64)
        self.explored = z(T, N, dt=torch.bool)
        self.logp = z(T, N, dt=torch.float32)
        self.sure = z(T, N, dt=torch.float32)
        self.value = z(T, N, dt=torch.float32)
        self.c_val = z(T, N, dt=torch.float32)  # the frozen critic's values (ppo.critic)
        self.reward = z(T, N, dt=torch.float32)
        self.done = z(T, N, dt=torch.float32)
        self.T, self.N = T, N

    def put_obs(self, t: int, o: Obs):
        self.tok_i[t] = o.tok_i
        self.tok_f[t] = o.tok_f
        self.glob_i[t] = o.glob_i
        self.glob_f[t] = o.glob_f
        self.ops[t] = o.ops
        self.ops_f[t] = o.ops_f
        self.n_ops[t] = o.n_ops
        self.n_tokens[t] = o.n_tokens
        self.n_hand[t] = o.n_hand

    def obs(self, idx, idx_cpu=None, T=None, O=None) -> Obs:
        """Flat indices over T*N -> Obs (float32 / int64). Host storage gathers
        on the CPU with ``idx_cpu`` (the same indices; else ``idx.cpu()``, a
        sync), and only the first ``T`` tokens and ``O`` ops: all that
        :meth:`Net.encode` reads when given the same T and O."""
        if self.host and idx_cpu is None:
            idx_cpu = idx.cpu()

        def f(x, n=None):
            x = x.flatten(0, 1)
            if n == 0:  # none of it is read (the critic takes no ops)
                return x.new_empty((len(idx), 0, *x.shape[2:]), device=idx.device)
            if x.device == idx.device:
                return x[idx]
            # Only the first n slots cross (states use ~40 of the 200 token slots), gathered into pinned memory so the
            # copy does not hold up the CPU (the allocator hands the block out again once it is done). They land in
            # full-width rows: encode's [:, :T] views keep today's strides, hence today's kernels (bitwise).
            full, x = x.shape[1:], x[:, :n]
            pin = torch.empty((len(idx_cpu), *x.shape[1:]), dtype=x.dtype, pin_memory=True)
            torch.index_select(x, 0, idx_cpu, out=pin)
            d = torch.empty((len(idx_cpu), *full), dtype=x.dtype, device=idx.device)
            d[:, : x.shape[1]].copy_(pin, non_blocking=True)
            return d

        return Obs(
            tok_i=f(self.tok_i, T).long(),
            tok_f=f(self.tok_f, T),
            glob_i=f(self.glob_i).long(),
            glob_f=f(self.glob_f),
            ops=f(self.ops, O).long(),
            ops_f=f(self.ops_f, O),
            n_ops=f(self.n_ops).long(),
            n_tokens=f(self.n_tokens).long(),
            n_hand=f(self.n_hand).long(),
        )


def gae(r: Rollout, last_value: torch.Tensor, gamma: float, lam: float, value: torch.Tensor | None = None):
    """Generalized advantage estimates and returns over ``value`` [T, N]
    (default ``r.value``). ``r.done`` ends the recursion (episodes do not leak
    into the next one); truncated steps already carry ``gamma * V(final
    state)`` in their reward."""
    value = r.value if value is None else value
    adv = torch.zeros_like(r.reward)
    last = torch.zeros_like(last_value)
    for t in reversed(range(r.T)):
        nonterm = 1.0 - r.done[t]
        nv = last_value if t == r.T - 1 else value[t + 1]
        delta = r.reward[t] + gamma * nv * nonterm - value[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + value


def mc_returns(r: Rollout, gamma: float):
    """The discounted returns of the steps whose episode ends inside the rollout,
    which no value function enters (a truncated one's keeps its bootstrap) ->
    (mask [T, N], returns [n])."""
    ends = r.done.flip(0).cummax(0).values.flip(0) > 0
    zero = torch.zeros_like(r.reward)
    return ends, gae(r, zero[0], gamma, 1.0, zero)[1][ends]


class EpisodeStats:
    """Rolling statistics of training episodes that started from a fresh
    run (snapshot restarts are only counted)."""

    FIELDS = ["episodes", "snapshot_episodes", "truncated", "win_rate", "win_lo", "win_hi", "ante_mean", "ante_max", "reach_ante9", "reach_ante12", "blinds_mean", "ep_len", "ep_return"]

    def __init__(self, window: int = 2000):
        self.eps = deque(maxlen=window)
        self.total = 0
        self.snapshot_eps = 0
        self.truncated = 0

    def add(self, eps):
        for e in eps:
            # (slot, seed, back, stake, ante, won, blinds, steps, return,
            #  truncated, from_snapshot)
            self.truncated += bool(e[9])
            if e[10]:
                self.snapshot_eps += 1
                continue
            self.eps.append(e)
            self.total += 1

    def summary(self) -> dict:
        """All of ``FIELDS`` (``None`` before the first fresh episode)."""
        out = dict.fromkeys(self.FIELDS)
        out.update(episodes=self.total, snapshot_episodes=self.snapshot_eps, truncated=self.truncated)
        if not self.eps:
            return out
        a = np.array([e[4] for e in self.eps])
        won = np.array([e[5] for e in self.eps], dtype=float)
        lo, hi = wilson(won.sum(), len(won))
        out.update(
            win_rate=float(won.mean()),
            win_lo=lo,
            win_hi=hi,
            ante_mean=float(a.mean()),
            ante_max=int(a.max()),
            reach_ante9=float((a >= 9).mean()),
            reach_ante12=float((a >= 12).mean()),
            blinds_mean=float(np.mean([e[6] for e in self.eps])),
            ep_len=float(np.mean([e[7] for e in self.eps])),
            ep_return=float(np.mean([e[8] for e in self.eps])),
        )
        return out


LOG_FIELDS = [
    "update",
    "steps",
    "fps",
    "rollout_s",
    "lr",
    "ent_coef",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clipfrac",
    "opt_steps",
    "logp_drift",
    "invalid_actions",
    "stakes",
    *EpisodeStats.FIELDS,
]
# With ppo.critic: the explained variance and the mean error of the critic's and
# the policy head's values against `mc_returns` (the same value-free targets).
# Kept only by ending in this rollout, so read them as critic vs head: both
# biases carry the same offset, and the first updates after a start are void.
CRITIC_FIELDS = ["critic_ev", "value_ev", "critic_bias", "value_bias"]


def _log_writer(f, path: Path, fields: list[str] = LOG_FIELDS) -> csv.DictWriter:
    """A writer of ``fields`` rows appending to ``path`` (whose header, if
    any, must match)."""
    if path.stat().st_size:
        with open(path, newline="") as r:
            header = next(csv.reader(r), [])
        if header != fields:
            raise ValueError(f"{path} has other columns than this version logs; move it away to resume")
    w = csv.DictWriter(f, fieldnames=fields)
    if not path.stat().st_size:
        w.writeheader()
    return w


def schedule(cfg: Config, update: int, n_updates: int):
    p = cfg.ppo
    frac = min(1.0, update / max(1, n_updates))
    if update < p.warmup_updates:
        lr = p.lr * (update + 1) / p.warmup_updates
    else:
        lr = p.lr_final + 0.5 * (p.lr - p.lr_final) * (1 + math.cos(math.pi * frac))
    ent = p.ent_coef + (p.ent_coef_final - p.ent_coef) * frac
    return lr, ent


def _mean(v: list[float]) -> float:
    return float(np.mean(v)) if v else float("nan")


def train(cfg: Config, resume: str | None = None, max_updates: int | None = None) -> dict:
    """Trains (or resumes, from a checkpoint path) until ``ppo.total_steps``
    or ``max_updates`` more updates; returns the last log row."""
    cfg.validate()
    device = pick_device(cfg.device)
    torch.set_float32_matmul_precision("high")
    run_dir = Path(cfg.run_dir) / cfg.name
    ck = read_checkpoint(resume) if resume else None
    if ck is None and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"{run_dir} already holds a run: choose another --name, or --resume it")
    if ck is not None:
        old = Config.from_dict(ck["config"])
        if old.env.num_envs * old.ppo.rollout_len != cfg.env.num_envs * cfg.ppo.rollout_len:
            raise ValueError("cannot resume with another env.num_envs * ppo.rollout_len (the update count would change meaning)")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.csv"
    log_path.touch()

    torch.manual_seed(cfg.env.seed)
    model = Net(cfg.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.ppo.lr, weight_decay=cfg.ppo.weight_decay, eps=1e-5)
    critic = None  # not compiled: it would share (and fill) the policy encoder's Dynamo cache
    if cfg.ppo.critic:  # built before a resumed RNG state is restored (its init draws from it)
        critic = Critic(cfg.model).to(device).eval().requires_grad_(False)
        critic.load_state_dict(ck["critic"] if ck is not None and "critic" in ck else critic_state(cfg.ppo.critic))
        if device.type == "cuda":
            critic.encoder.half()  # ~4x faster, values within ~1e-3 (its inputs are float16-rounded already)
    start_update, steps = 0, 0
    # Outcomes of the latest fresh episodes on the top stake (curriculum).
    top = deque(maxlen=cfg.env.curriculum_min_episodes)
    if ck is not None:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_update, steps = ck["update"], ck["steps"]
        top.extend(ck.get("curriculum", []))
        if "torch_rng" in ck:
            torch.set_rng_state(ck["torch_rng"])
        if "cuda_rng" in ck and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
    if cfg.ppo.compile:
        model.encoder.compile()  # in place: parameter names stay the same
    amp = cfg.ppo.amp and ((device.type == "cuda" and torch.cuda.is_bf16_supported()) or (device.type == "cpu" and cfg.ppo.amp_cpu))
    if cfg.ppo.amp and device.type == "cuda" and not amp:
        print("[antezero] this GPU has no bf16: training in float32", flush=True)
    autocast = lambda: torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp)  # noqa: E731

    def save(update: int):
        extra = {"curriculum": list(top), "torch_rng": torch.get_rng_state()}
        if torch.cuda.is_available():
            extra["cuda_rng"] = torch.cuda.get_rng_state_all()
        if critic is not None:
            extra["critic"] = critic.state_dict()
        save_checkpoint(run_dir / "last.pt", model, opt, cfg, update, steps, extra)

    def value_of(o: Obs) -> torch.Tensor:
        """The advantages' value function: the critic's, else the policy's value head."""
        if critic is not None:
            return critic.value(o).float()
        cls, _tok, _ope, _lg = model.encode(o)
        return model.value(cls).squeeze(-1).float()

    fast = cfg.ppo.fast_rollout
    if fast and (device.type != "cuda" or amp):
        raise ValueError("ppo.fast_rollout needs a CUDA device and ppo.amp=false")
    # Mode 2 has the critic value the states as it acts; modes 0 and 1 stay bitwise equal (a pass after the rollout).
    new_env = lambda seed: Collector(cfg, model, device, seed, fast, value_of, critic if fast == 2 else None) if fast else make_env(cfg, seed=seed)  # noqa: E731
    env = new_env(cfg.env.seed + 1_000_003 * start_update)
    N, T = cfg.env.num_envs, cfg.ppo.rollout_len
    roll = Rollout(T, N, device, cfg.ppo.rollout_obs_on_cpu)
    # The schedule spans the whole run; `max_updates` only stops early.
    n_updates = cfg.ppo.total_steps // (N * T)
    stop_at = n_updates if max_updates is None else min(n_updates, start_update + max_updates)
    stats = EpisodeStats()
    B = N * T
    mb = B // cfg.ppo.minibatches
    fields = LOG_FIELDS + (CRITIC_FIELDS if critic is not None else [])
    print(f"[antezero] device={device} params={count_params(model):,} envs={N} rollout={T} updates={start_update}..{stop_at} of {n_updates}", flush=True)

    last: dict = {}
    completed = start_update
    with open(log_path, "a", newline="") as log_f:
        log_w = _log_writer(log_f, log_path, fields)
        (run_dir / "config.json").write_text(cfg.to_json())
        for update in range(start_update, stop_at):
            t0 = time.perf_counter()
            lr, ent_coef = schedule(cfg, update, n_updates)
            for g in opt.param_groups:
                g["lr"] = lr
            top_stake = max(cfg.env.stakes)
            # ------------------------------------------------------ rollout
            model.eval()

            def on_episodes(new_eps):
                stats.add(new_eps)
                top.extend(bool(x[5]) for x in new_eps if not x[10] and x[3] == top_stake)

            if fast:
                env.collect(roll, on_episodes)
            for t in range(0 if fast else T):
                o = obs_from_numpy(env, device)
                roll.put_obs(t, o)
                with autocast():
                    out = model.act(o, explore=cfg.ppo.explore_redeem)
                roll.op[t] = out["op"]
                roll.card_seq[t] = out["card_seq"]
                roll.dest[t] = out["dest"]
                roll.explored[t] = out["explored"]
                roll.logp[t] = out["logp"].float()
                roll.sure[t] = out["sure"].float()
                roll.value[t] = out["value"].float()
                rew, done, trunc = env.step(
                    out["op"].to(torch.int32).cpu().numpy(),
                    out["cards"].cpu().numpy().astype(np.uint32),
                    out["dest"].to(torch.int32).cpu().numpy(),
                )
                rew = torch.as_tensor(rew, device=device)
                if trunc.any():
                    # Not a real end: bootstrap from the final state's value.
                    rows = np.flatnonzero(trunc)
                    with torch.no_grad(), autocast():
                        v_fin = value_of(obs_from_numpy(env, device, final=True, rows=rows))
                    rew[torch.as_tensor(rows, device=device)] += cfg.ppo.gamma * v_fin
                roll.reward[t] = rew
                roll.done[t] = torch.as_tensor(done, device=device, dtype=torch.float32)
                on_episodes(env.pop_episodes())
            with torch.no_grad(), autocast():
                vals = []
                for o in env.last_obs() if fast else [obs_from_numpy(env, device)]:
                    vals.append(value_of(o))
                last_value = torch.cat(vals)
            flat_op, flat_seq, flat_dest = roll.op.flatten(), roll.card_seq.flatten(0, 1), roll.dest.flatten()
            logp_drift = torch.zeros((), device=device)
            c_val = roll.c_val if critic is not None else None
            sweep = critic is not None and not (fast and env.act.critic is not None)
            if cfg.ppo.recompute_logp or sweep:
                # Slices of states sorted by token count, each at its longest token and op lists (one GPU sync).
                order = roll.n_tokens.flatten().argsort(stable=True).view(-1, mb)
                lens = torch.stack([roll.n_tokens.flatten()[order].amax(1), roll.n_ops.flatten()[order].amax(1)], -1).clamp(min=1).tolist()
                order_cpu = order.cpu() if roll.host else None
                with torch.no_grad(), autocast():
                    for k in range(cfg.ppo.minibatches):
                        idx = order[k]
                        # The critic alone reads no ops.
                        o = roll.obs(idx, order_cpu[k] if roll.host else None, lens[k][0], lens[k][1] if cfg.ppo.recompute_logp else 0)
                        tt, nn_ = idx // N, idx % N
                        if cfg.ppo.recompute_logp:
                            lp, _e, v = model.evaluate(o, flat_op[idx], flat_seq[idx], flat_dest[idx], *lens[k])
                            # Kept on the device (no sync per slice: the next slice is gathered while this one runs).
                            logp_drift = torch.maximum(logp_drift, (lp.float() - roll.logp[tt, nn_]).abs().max())
                            roll.logp[tt, nn_] = lp.float()
                            roll.value[tt, nn_] = v.float()
                        if sweep:
                            c_val[tt, nn_] = critic.value(o, lens[k][0]).float()
            adv, ret = gae(roll, last_value, cfg.ppo.gamma, cfg.ppo.gae_lambda, c_val)
            c_log = {}
            if critic is not None:
                # Graded on value-free targets: the critic's own lambda-returns would favour it (staleness).
                ends, mc = mc_returns(roll, cfg.ppo.gamma)
                for k, v in (("critic", c_val), ("value", roll.value)):
                    e = v[ends] - mc
                    c_log[f"{k}_ev"], c_log[f"{k}_bias"] = (float(1 - e.var() / mc.var()), float(e.mean())) if len(mc) > 1 else (math.nan, math.nan)
            x_log = ""
            if cfg.ppo.explore_redeem:  # explored redeems: count, mean normalised advantage (biased low until V knows vouchers)
                x_adv = ((adv[roll.explored] - adv.mean()) / adv.std()).mean()
                x_log = f" explored={int(roll.explored.sum())} x_adv={float(x_adv):+.3f}"
            t_roll = time.perf_counter() - t0
            steps += B

            # ------------------------------------------------------- update
            model.train()
            flat_adv, flat_ret = adv.flatten(), ret.flatten()
            flat_logp, flat_val, flat_x = roll.logp.flatten(), roll.value.flatten(), roll.explored.flatten()
            flat_nt, flat_no = roll.n_tokens.flatten().long(), roll.n_ops.flatten().long()
            flat_sure = roll.sure.flatten().masked_fill(flat_x, -torch.inf)  # forced redeems are always trained on
            A = cfg.ppo.grad_accum
            kb = max(A, int(mb * cfg.ppo.learn_frac) // A * A)  # samples of a minibatch that are trained on
            kept = kb / mb
            kls, clipfracs, pls, vls, ents = [], [], [], [], []
            for _epoch in range(cfg.ppo.epochs):
                perm = torch.randperm(B, device=device).view(-1, mb)
                # Each minibatch's states by token count, so its equal chunks pad less (same summed gradient).
                perm = perm.gather(1, flat_nt[perm].argsort(dim=1))
                full = perm  # advantages are normalised over the whole minibatch
                if kb < mb:  # ppo.learn_frac: the least sure states, still by token count
                    perm = perm.gather(1, flat_sure[perm].topk(kb, dim=1, largest=False).indices.sort(dim=1).values)
                ch = perm.view(-1, A, kb // A)
                lens = torch.stack([flat_nt[ch].amax(-1), flat_no[ch].amax(-1)], -1).clamp(min=1).tolist()  # one GPU sync per epoch
                perm = perm.flatten()
                perm_cpu = perm.cpu() if roll.host else perm  # what host storage gathers with
                for k in range(cfg.ppo.minibatches):
                    idx = perm[k * kb : (k + 1) * kb]
                    a_mean, a_std = flat_adv[full[k]].mean(), flat_adv[full[k]].std() + 1e-8
                    opt.zero_grad(set_to_none=True)
                    parts = []
                    finite = torch.ones((), dtype=torch.bool, device=device)
                    # Equal chunks of means: their summed gradients equal the whole minibatch's.
                    for c, c_cpu, (n_tok, n_op) in zip(idx.chunk(A), perm_cpu[k * kb : (k + 1) * kb].chunk(A), lens[k]):
                        o = roll.obs(c, c_cpu, n_tok, n_op)
                        with autocast():
                            logp, ent, val = model.evaluate(o, flat_op[c], flat_seq[c], flat_dest[c], n_tok, n_op)
                        logp, ent, val = logp.float(), ent.float(), val.float()
                        x = flat_x[c]
                        with torch.no_grad():
                            lr_ = logp - flat_logp[c]
                            # Explored actions have pi ~ 0: their ratio is no measure of policy change.
                            kl_terms = ((lr_.exp() - 1) - lr_)[~x]
                            kl = kl_terms.mean() if kl_terms.numel() else lr_.new_zeros(())
                        a = (flat_adv[c] - a_mean) / a_std
                        ratio = (logp - flat_logp[c]).exp()
                        clipped = ratio.clamp(1 - cfg.ppo.clip, 1 + cfg.ppo.clip) * a
                        # Explored redeems: no importance weight (it would be ~0), and no unclipped side
                        # (shared weights carry their ratios far outside the band, which would amplify them).
                        surr = torch.where(x, cfg.ppo.explore_weight * clipped, torch.min(ratio * a, clipped))
                        pl = -surr.mean()
                        if cfg.ppo.value_clip > 0:
                            v_old = flat_val[c]
                            v_clip = v_old + (val - v_old).clamp(-cfg.ppo.value_clip, cfg.ppo.value_clip)
                            vl = 0.5 * torch.max((val - flat_ret[c]) ** 2, (v_clip - flat_ret[c]) ** 2).mean()
                        else:
                            vl = 0.5 * ((val - flat_ret[c]) ** 2).mean()
                        # glob_i column 0 is the phase: 2 = shop, 3 = booster pack.
                        shop = (o.glob_i[:, 0] == 2) | (o.glob_i[:, 0] == 3)
                        loss = pl + cfg.ppo.vf_coef * vl - ent_coef * (ent * torch.where(shop, cfg.ppo.shop_ent_mult, 1.0)).mean()
                        finite &= torch.isfinite(loss)
                        (loss * kept / A).backward()
                        clipfrac = ((ratio.detach() - 1).abs() > cfg.ppo.clip).float().mean()
                        parts.append(torch.stack([kl * kept, clipfrac * kept, pl.detach(), vl.detach(), ent.detach().mean() * kept]))
                    # Fail instead of training on garbage (on Vast this also stops the instance).
                    if not finite:
                        raise FloatingPointError(f"non-finite loss at update {update + 1}")
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.ppo.max_grad_norm)
                    opt.step()
                    for v, s in zip((kls, clipfracs, pls, vls, ents), torch.stack(parts).mean(0).tolist()):
                        v.append(s)
                # Judged per epoch: one noisy minibatch no longer cuts the update short.
                if cfg.ppo.target_kl and _mean(kls[-cfg.ppo.minibatches :]) > 1.5 * cfg.ppo.target_kl:
                    break
            dt = time.perf_counter() - t0
            row = {
                "update": update + 1,
                "steps": steps,
                "fps": round(B / dt),
                "rollout_s": round(t_roll, 3),
                "lr": lr,
                "ent_coef": ent_coef,
                "policy_loss": _mean(pls),
                "value_loss": _mean(vls),
                "entropy": _mean(ents),
                "approx_kl": _mean(kls),
                "clipfrac": _mean(clipfracs),
                "opt_steps": len(pls),
                "logp_drift": float(logp_drift),
                "invalid_actions": env.invalid_actions(),
                "stakes": ",".join(map(str, cfg.env.stakes)),
                **stats.summary(),
            }
            row.update(c_log)
            last = row
            log_w.writerow(row)
            log_f.flush()
            if cfg.log_every and (update + 1) % cfg.log_every == 0:
                s = row
                num = lambda k, spec: "-" if s[k] is None else format(s[k], spec)  # noqa: E731
                print(
                    f"[{s['update']}] steps={s['steps']:,} fps={s['fps']:,} win={num('win_rate', '.3f')} "
                    f"[{num('win_lo', '.3f')},{num('win_hi', '.3f')}] ante={num('ante_mean', '.2f')} max={num('ante_max', 'd')} "
                    f"ret={num('ep_return', '.3f')} ent={s['entropy']:.3f} kl={s['approx_kl']:.4f} vl={s['value_loss']:.4f}{x_log}"
                    + (f" cev={s['critic_ev']:.3f} vev={s['value_ev']:.3f} cbias={s['critic_bias']:+.3f} vbias={s['value_bias']:+.3f}" if critic is not None else ""),
                    flush=True,
                )
            e = cfg.env
            if e.curriculum and top_stake < 8 and len(top) == top.maxlen:
                # The latest fresh episodes on the top stake (sampled actions,
                # so a conservative estimate of greedy play).
                lo, _hi = wilson(sum(top), len(top))
                if lo >= e.curriculum_target:
                    e.stakes = sorted(set(e.stakes) | {top_stake + 1})
                    print(f"[curriculum] stakes -> {e.stakes}", flush=True)
                    env = new_env(e.seed + 7919 * (update + 1))
                    top.clear()
                    (run_dir / "config.json").write_text(cfg.to_json())
            completed = update + 1
            if cfg.save_every and completed % cfg.save_every == 0:
                save(completed)
                # Every saved checkpoint is kept (an average of them beats the last one), whether or not it is
                # evaluated: the in-run eval costs wall time and can be rarer.
                shutil.copyfile(run_dir / "last.pt", run_dir / f"u{completed}.pt")
            if cfg.eval_every and completed % cfg.eval_every == 0:
                from .eval import evaluate

                res = evaluate(model, cfg, episodes=cfg.eval_episodes, device=device)
                with open(run_dir / "eval.jsonl", "a") as f:
                    f.write(json.dumps({"update": completed, "steps": steps, **res}) + "\n")
                print(f"[eval {completed}] {res['summary']}", flush=True)
        save(completed)
    return last


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="base", choices=["smoke", "cpu", "base", "large"])
    ap.add_argument("--name", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. ppo.lr=1e-4 env.backs=b_red,b_blue")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-updates", type=int, default=None)
    a = ap.parse_args(argv)
    try:
        cfg = Config.from_dict(read_checkpoint(a.resume)["config"]) if a.resume else preset(a.preset)
        if a.name:
            cfg.name = a.name
        if a.device:
            cfg.device = a.device
        parse_overrides(cfg, a.set)
        cfg.validate()
    except ValueError as e:
        ap.error(str(e))
    train(cfg, resume=a.resume, max_updates=a.max_updates)


if __name__ == "__main__":
    main()
