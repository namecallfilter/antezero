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
saved). A file named STOP in the run directory ends the run after the
current update, with a checkpoint to resume from.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import shutil
import signal
import time
from collections import deque
from pathlib import Path

import antezero_sim as az
import numpy as np
import torch
import torch.nn as nn

from .collect import Collector
from .config import Config, parse_overrides, preset
from .critic import Critic, Learner, critic_state
from .model import GAIN, MAX_CARDS, Net, Obs, count_params, obs_from_numpy
from .models import build_model, load_text_embeddings
from .runtime import make_env, pick_device, read_checkpoint, save_checkpoint
from .stats import wilson


class Rollout:
    """Integer features use compact (lossless) dtypes; floats stay float32 so
    training sees exactly the observations the actions were sampled on.

    ``obs_on_cpu`` (CUDA devices; ``ppo.rollout_obs_on_cpu``): the tokens and
    ops (33.0 of the 33.9 KB per state) live in host memory instead, and
    :meth:`obs` gathers them there."""

    def __init__(self, T: int, N: int, device: torch.device, obs_on_cpu: bool = False, glob_float: int = az.GLOB_FLOAT):
        z = lambda *s, dt: torch.zeros(*s, dtype=dt, device=device)  # noqa: E731
        self.host = obs_on_cpu and device.type == "cuda"
        # Pageable: torch rounds pinned blocks up to a power of two (tok_f: 11.7 -> 16 GiB at 4,096 x 128).
        big = (lambda *s, dt: torch.zeros(*s, dtype=dt)) if self.host else z  # noqa: E731
        self.tok_i = big(T, N, az.MAX_TOKENS, az.TOK_INT, dt=torch.int16)
        self.tok_f = big(T, N, az.MAX_TOKENS, az.TOK_FLOAT, dt=torch.float32)
        self.glob_i = z(T, N, az.GLOB_INT, dt=torch.int16)
        self.glob_f = z(T, N, glob_float, dt=torch.float32)
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
        # The joker-order labels of the actions (VecEnv.aux_kind / aux_dest; env.order_label_rate).
        self.aux_kind = z(T, N, dt=torch.int8) - 1
        self.aux_dest = z(T, N, az.ORDER_JOKERS, dt=torch.uint8)
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
            # full-width rows so that encode's [:, :T] views keep the on-device strides, hence the same kernels (bitwise).
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


def gae(r: Rollout, last_value: torch.Tensor, gamma: float, lam: float, value: torch.Tensor | None = None, cut: torch.Tensor | None = None):
    """Generalized advantage estimates and returns over ``value`` [T, N]
    (default ``r.value``). ``r.done`` ends the recursion (episodes do not leak
    into the next one); truncated steps already carry ``gamma * V(final
    state)`` in their reward. ``cut`` [T, N]: steps whose action the policy
    did not choose (explored); an earlier step's estimate stops at the value
    of the state before them, as what follows is not the policy's doing."""
    value = r.value if value is None else value
    adv = torch.zeros_like(r.reward)
    last = torch.zeros_like(last_value)
    for t in reversed(range(r.T)):
        nonterm = 1.0 - r.done[t]
        nv = last_value if t == r.T - 1 else value[t + 1]
        delta = r.reward[t] + gamma * nv * nonterm - value[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
        if cut is not None:
            last = last * ~cut[t]
    return adv, adv + value


def mc_returns(r: Rollout, gamma: float):
    """The discounted returns of the steps whose episode ends inside the rollout,
    which no value function enters (a truncated one's keeps its bootstrap) ->
    (mask [T, N], returns [T, N]; of the other steps: the part of the return
    inside the rollout)."""
    ends = r.done.flip(0).cummax(0).values.flip(0) > 0
    zero = torch.zeros_like(r.reward)
    return ends, gae(r, zero[0], gamma, 1.0, zero)[1]


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
# With env.order_label_rate, of the rollout's labelled played hands and the policy that played them: how many had a
# gaining single joker move and how many had none; on the former, the median lead of the labelled moves' logit over
# the best other op and the share where greedy play makes a labelled move to one of its slots; on the latter, the
# mean probability of moving a joker.
ORDER_FIELDS = ["order_pos", "order_zero", "order_lead", "order_hit", "order_stray"]
EXPLORE_FIELDS = ["explored", "x_adv", "x_pos"]
ORDER_CLIP = 0.5  # the ordering loss's own gradient-norm clip


def _log_writer(f, path: Path, fields: list[str] = LOG_FIELDS) -> csv.DictWriter:
    if path.stat().st_size:
        with open(path, newline="") as r:
            header = next(csv.reader(r), [])
        if header != fields:
            raise ValueError(f"{path} has other columns than this version logs; move it away to resume")
    w = csv.DictWriter(f, fieldnames=fields)
    if not path.stat().st_size:
        w.writeheader()
    return w


class Optimizers:
    """Several optimizers stepped as one (param_groups, state_dict, step, zero_grad)."""

    def __init__(self, *opts):
        self.opts = opts

    @property
    def param_groups(self):
        return [g for o in self.opts for g in o.param_groups]

    def step(self):
        for o in self.opts:
            o.step()

    def zero_grad(self, set_to_none: bool = False):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return [o.state_dict() for o in self.opts]

    def load_state_dict(self, d):
        for o, s in zip(self.opts, d):
            o.load_state_dict(s)


def muon_params(model: nn.Module, cfg: Config) -> list[nn.Parameter]:
    """2-D hidden matrices optimized by Muon (excludes output heads and, when embed_adam, embedding tables)."""
    table_ids = {id(m.weight) for m in model.modules() if isinstance(m, nn.Embedding)}
    heads = ("op_score.", "ptr_stop.", "value.")
    return [
        p
        for n, p in model.named_parameters()
        if p.ndim == 2 and not n.startswith(heads) and "emb" not in n and not (cfg.ppo.embed_adam and id(p) in table_ids)
    ]


def build_optimizer(model: nn.Module, cfg: Config, device: torch.device):
    p = cfg.ppo
    if p.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=p.lr, weight_decay=p.weight_decay, eps=1e-5)
    hidden = muon_params(model, cfg)
    h_ids = {id(q) for q in hidden}
    rest = [q for q in model.parameters() if id(q) not in h_ids]
    fast = {"fused": True} if device.type == "cuda" else {}
    adam = torch.optim.Adam(rest, lr=p.lr, eps=1e-5, **fast)
    if not hidden:
        return Optimizers(adam)
    muon = torch.optim.Muon(hidden, lr=p.lr, weight_decay=p.muon_wd, adjust_lr_fn="match_rms_adamw")
    return Optimizers(muon, adam)


def update_kl_controller(cfg: Config, update: int, kl: float, kl_ctl: dict) -> None:
    p = cfg.ppo
    if p.target_kl > 0 and not math.isnan(kl):
        kl_ctl["ema"] = kl if kl_ctl["ema"] is None else (1 - p.kl_ema) * kl_ctl["ema"] + p.kl_ema * kl
        if update >= p.kl_start:
            if kl_ctl["ema"] > p.kl_high * p.target_kl:
                kl_ctl["m"] *= 1 - p.kl_step
            elif kl_ctl["ema"] < p.kl_low * p.target_kl:
                kl_ctl["m"] = min(p.kl_max_factor, kl_ctl["m"] * (1 + p.kl_step))


def schedule(cfg: Config, update: int, n_updates: int, kl_ctl: dict | None = None):
    p = cfg.ppo
    frac = min(1.0, update / max(1, n_updates))
    if p.warmup_updates > 0 and update < p.warmup_updates:
        lr = p.lr * (update + 1) / p.warmup_updates
    elif p.lr_schedule == "exp":
        tau = -p.lr_decay_updates / math.log(max(1e-6, p.lr_final / max(1e-6, p.lr - p.lr_final)))
        lr = p.lr_final + (p.lr - p.lr_final) * math.exp(-update / max(1.0, tau))
    else:
        lr = p.lr_final + 0.5 * (p.lr - p.lr_final) * (1 + math.cos(math.pi * frac))
    if kl_ctl is not None:
        lr *= kl_ctl["m"]
    ent = p.ent_coef + (p.ent_coef_final - p.ent_coef) * frac
    return lr, ent


def _mean(v: list[float]) -> float:
    return float(np.mean(v)) if v else float("nan")


def order_log(model: Net, roll: Rollout, batch: int) -> dict:
    """``ORDER_FIELDS`` of a rollout (one pass without gradients over its labelled states, ``batch`` at a time)."""
    kind, bits = roll.aux_kind.flatten(), roll.aux_dest.flatten(0, 1)
    nt, no = roll.n_tokens.flatten(), roll.n_ops.flatten()
    lead, hit, stray = [], [], []
    with torch.no_grad():
        labelled = (kind >= 0).nonzero().squeeze(1)
        for idx in labelled.split(batch) if len(labelled) else ():
            n_tok, n_op = int(nt[idx].max()), int(no[idx].max())
            _loss, ld, ht, pm = model.order_aux(roll.obs(idx, None, n_tok, n_op), bits[idx], n_tok, n_op)
            pos = kind[idx] == 1
            lead.append(ld[pos])
            hit.append(ht[pos].float())
            stray.append(pm[~pos])
    lead, hit, stray = (torch.cat(v) if v else torch.zeros(0) for v in (lead, hit, stray))
    stat = lambda v, f: float(f(v)) if len(v) else math.nan  # noqa: E731
    return {
        "order_pos": int((kind == 1).sum()),
        "order_zero": int((kind == 0).sum()),
        "order_lead": stat(lead, torch.median),
        "order_hit": stat(hit, torch.mean),
        "order_stray": stat(stray, torch.mean),
    }


def add_order_grad(model: Net, parts: list, coef: float) -> None:
    """``ppo.aux_order``: adds the gradient of the ordering loss (``Net.order_aux``) to ``.grad``, clipped to
    ``ORDER_CLIP`` on its own. Only the order-gain input column of ``op_feat`` and the destination head receive
    it. ``parts``: the states of a minibatch whose labels have a gaining move, per chunk (observations, destination
    bits, their rows of the chunk's ``Net.encode`` output or None to encode them here); the loss is their mean."""
    n = sum(len(bits) for _o, bits, _enc in parts)
    if not n:
        return
    loss = coef * sum(model.order_aux(o, bits, enc=enc)[0].sum() for o, bits, enc in parts if len(bits)) / n
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite ordering loss")
    params = [model.op_feat.weight, *model.dst_q.parameters(), *model.dst_k.parameters(), *model.dst_pos.parameters()]
    grads = list(torch.autograd.grad(loss, params))
    column = torch.zeros_like(grads[0])
    column[:, GAIN] = grads[0][:, GAIN]
    grads[0] = column
    scale = (ORDER_CLIP / (torch.sqrt(sum((g * g).sum() for g in grads)) + 1e-6)).clamp(max=1.0)
    for p, g in zip(params, grads):
        p.grad.add_(g * scale)


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
        if old.model != cfg.model:
            raise ValueError("cannot resume with another model: the checkpoint's weights and effect table are its own")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.csv"
    log_path.touch()

    torch.manual_seed(cfg.env.seed)
    model = build_model(cfg.model, ck["model"].get("fx") if ck is not None else None).to(device)
    if ck is None and cfg.model.embed_from:
        tables = load_text_embeddings(model, cfg.model.embed_from, device)
        print(f"[antezero] embeddings from {cfg.model.embed_from}: {', '.join(tables)}", flush=True)
    opt = build_optimizer(model, cfg, device)
    critic = learner = None  # not compiled: it would share (and fill) the policy encoder's Dynamo cache
    if cfg.ppo.critic:  # built before a resumed RNG state is restored (its init draws from it)
        state = ck["critic"] if ck is not None and "critic" in ck else critic_state(cfg.ppo.critic)
        critic = Critic(cfg.model, state.get("fx")).to(device).eval().requires_grad_(False)
        critic.load_state_dict(state)
        if cfg.ppo.critic_learn:  # copied before the float16 cast below
            learner = Learner(critic)
            if ck is not None and "critic_learner" in ck:
                learner.load(ck["critic_learner"])
        if device.type == "cuda":
            critic.encoder.half()  # ~4x faster, values within ~1e-3 (its inputs are float16-rounded already)
    start_update, steps = 0, 0
    kl_ctl = {"m": 1.0, "ema": None}
    # Outcomes of the latest fresh episodes on the top stake (curriculum).
    top = deque(maxlen=cfg.env.curriculum_min_episodes)
    if ck is not None:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_update, steps = ck["update"], ck["steps"]
        if "kl_ctl" in ck:
            kl_ctl = ck["kl_ctl"]
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
        extra = {"curriculum": list(top), "kl_ctl": kl_ctl, "torch_rng": torch.get_rng_state()}
        if torch.cuda.is_available():
            extra["cuda_rng"] = torch.cuda.get_rng_state_all()
        if critic is not None:
            extra["critic"] = critic.state_dict()
        if learner is not None:
            extra["critic_learner"] = learner.state()
        save_checkpoint(run_dir / "last.pt", model, opt, cfg, update, steps, extra)

    def value_of(o: Obs) -> torch.Tensor:
        if critic is not None:
            return critic.value(o).float()
        cls, _tok, _ope, _lg = model.encode(o)
        return model.value(cls).squeeze(-1).float()

    explore = (cfg.ppo.explore_redeem, cfg.ppo.explore_buy)
    fast = cfg.ppo.fast_rollout
    if fast and (device.type != "cuda" or amp):
        raise ValueError("ppo.fast_rollout needs a CUDA device and ppo.amp=false")
    # Mode 2 has the critic value the states as it acts; modes 0 and 1 stay bitwise equal (a pass after the rollout).
    new_env = lambda seed: Collector(cfg, model, device, seed, fast, value_of, critic if fast == 2 else None) if fast else make_env(cfg, seed=seed)  # noqa: E731
    env = new_env(cfg.env.seed + 1_000_003 * start_update)
    N, T = cfg.env.num_envs, cfg.ppo.rollout_len
    glob_float = az.GLOB_FLOAT_V2 if cfg.model.obs_v2 else az.GLOB_FLOAT
    roll = Rollout(T, N, device, cfg.ppo.rollout_obs_on_cpu, glob_float=glob_float)
    # The schedule spans the whole run; `max_updates` only stops early.
    n_updates = cfg.ppo.total_steps // (N * T)
    stop_at = n_updates if max_updates is None else min(n_updates, start_update + max_updates)
    stats = EpisodeStats()
    B = N * T
    mb = B // cfg.ppo.minibatches
    labels = cfg.env.order_label_rate > 0
    fields = LOG_FIELDS + (CRITIC_FIELDS if critic is not None else []) + (Learner.FIELDS if learner is not None else []) + (ORDER_FIELDS if labels else []) + (EXPLORE_FIELDS if any(explore) else [])
    print(f"[antezero] device={device} params={count_params(model):,} envs={N} rollout={T} updates={start_update}..{stop_at} of {n_updates}", flush=True)

    last: dict = {}
    completed = start_update
    wb = None
    if cfg.wandb:
        import wandb

        wb = wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb,
            name=f"antezero-{cfg.name}",
            group="antezero",
            id=cfg.wandb_id or f"antezero-{cfg.name}",
            resume="allow",
            config=dataclasses.asdict(cfg),
            dir=str(run_dir),
        )
        if not wb.resumed and log_path.stat().st_size > 0:
            with open(log_path, newline="") as rf:
                for r in csv.DictReader(rf):
                    wb.log(
                        {
                            f"train/{k}": float(v)
                            for k, v in r.items()
                            if k not in ("update", "stakes") and v not in ("", "nan", "None")
                        },
                        step=int(r["update"]),
                    )
    w_groups = [
        (
            "weights/rms_muon" if isinstance(o, torch.optim.Muon) else "weights/rms_adam",
            [q for g in o.param_groups for q in g["params"]],
        )
        for o in getattr(opt, "opts", [opt])
    ]
    combo_recent: dict[tuple[str, int], deque[int]] = {}
    stop_requested = False
    prev_sigterm = signal.getsignal(signal.SIGTERM)

    def _on_sigterm(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        with open(log_path, "a", newline="") as log_f:
            log_w = _log_writer(log_f, log_path, fields)
            (run_dir / "config.json").write_text(cfg.to_json())
            for update in range(start_update, stop_at):
                    t0 = time.perf_counter()
                    lr, ent_coef = schedule(cfg, update, n_updates, kl_ctl if cfg.ppo.kl_controller else None)
                    for g in opt.param_groups:
                        g["lr"] = lr
                    top_stake = max(cfg.env.stakes)
                    model.eval()
                    cells: dict[str, list[int]] = {}  # this update's fresh episodes per deck/stake: [won, played]

                    def on_episodes(new_eps):
                        stats.add(new_eps)
                        top.extend(bool(x[5]) for x in new_eps if not x[10] and x[3] == top_stake)
                        for x in new_eps:
                            if not x[10]:
                                c = cells.setdefault(f"{x[2]}/{x[3]}", [0, 0])
                                c[0] += bool(x[5])
                                c[1] += 1
                                combo_recent.setdefault((x[2], int(x[3])), deque(maxlen=200)).append(int(bool(x[5])))

                    if fast:
                        env.collect(roll, on_episodes)
                    for t in range(0 if fast else T):
                        o = obs_from_numpy(env, device)
                        roll.put_obs(t, o)
                        with autocast():
                            out = model.act(o, explore=explore)
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
                        roll.aux_kind[t] = torch.as_tensor(np.asarray(env.aux_kind), device=device)
                        roll.aux_dest[t] = torch.as_tensor(np.asarray(env.aux_dest), device=device)
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
                    adv, ret = gae(roll, last_value, cfg.ppo.gamma, cfg.ppo.gae_lambda, c_val, roll.explored)
                    c_log = {}
                    if critic is not None:
                        # Graded on value-free targets: the critic's own lambda-returns would favour it (staleness).
                        ends, part = mc_returns(roll, cfg.ppo.gamma)
                        mc = part[ends]
                        for k, v in (("critic", c_val), ("value", roll.value)):
                            e = v[ends] - mc
                            c_log[f"{k}_ev"], c_log[f"{k}_bias"] = (float(1 - e.var() / mc.var()), float(e.mean())) if len(mc) > 1 else (math.nan, math.nan)
                        if learner is not None:
                            c_log.update(learner.step(roll, ends, part, (c_val, roll.value), cfg.ppo.gamma, cfg.env.r_win))
                            if cfg.ppo.critic_learn == 2:  # the fitted copy values the next rollout
                                if not all(bool(torch.isfinite(p).all()) for p in learner.net.parameters()):
                                    raise FloatingPointError("the online critic's weights are not finite")
                                critic.load_state_dict(learner.net.state_dict())
                    x_log = ""
                    if any(explore):  # explored steps: count, mean normalised advantage, share better than expected
                        xa = (adv[roll.explored] - adv.mean()) / adv.std()
                        c_log.update(explored=int(roll.explored.sum()), x_adv=float(xa.mean()) if len(xa) else 0.0, x_pos=float((xa > 0).float().mean()) if len(xa) else 0.0)
                        x_log = f" explored={c_log['explored']} x_adv={c_log['x_adv']:+.3f} x_pos={c_log['x_pos']:.2f}"
                    if labels:
                        c_log.update(order_log(model, roll, mb // cfg.ppo.grad_accum))
                    t_roll = time.perf_counter() - t0
                    steps += B

                    model.train()
                    flat_adv, flat_ret = adv.flatten(), ret.flatten()
                    flat_logp, flat_val, flat_x = roll.logp.flatten(), roll.value.flatten(), roll.explored.flatten()
                    flat_nt, flat_no = roll.n_tokens.flatten().long(), roll.n_ops.flatten().long()
                    flat_kind, flat_bits = roll.aux_kind.flatten(), roll.aux_dest.flatten(0, 1)
                    # Forced redeems and the hands a joker move would have gained on are always trained on.
                    flat_sure = roll.sure.flatten().masked_fill(flat_x | (flat_kind == 1), -torch.inf)
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
                            parts, gaining = [], []
                            finite = torch.ones((), dtype=torch.bool, device=device)
                            # Equal chunks of means: their summed gradients equal the whole minibatch's.
                            for c, c_cpu, (n_tok, n_op) in zip(idx.chunk(A), perm_cpu[k * kb : (k + 1) * kb].chunk(A), lens[k]):
                                o = roll.obs(c, c_cpu, n_tok, n_op)
                                with autocast():
                                    enc = model.encode(o, n_tok, n_op)
                                    logp, ent, val = model.evaluate(o, flat_op[c], flat_seq[c], flat_dest[c], n_tok, n_op, enc)
                                if cfg.ppo.aux_order:  # the ordering loss reads this pass's encodings of its states
                                    g = (flat_kind[c] == 1).nonzero().squeeze(1)
                                    if len(g):
                                        gaining.append((Obs(**{f: getattr(o, f)[g] for f in Obs.__dataclass_fields__}), flat_bits[c][g], tuple(x[g].detach() for x in enc[:3])))
                                logp, ent, val = logp.float(), ent.float(), val.float()
                                x = flat_x[c]
                                with torch.no_grad():
                                    lr_ = logp - flat_logp[c]
                                    # Explored actions have pi ~ 0: their ratio is no measure of policy change.
                                    kl_terms = ((lr_.exp() - 1) - lr_)[~x]
                                    kl = kl_terms.mean() if kl_terms.numel() else lr_.new_zeros(())
                                a = (flat_adv[c] - a_mean) / a_std
                                drift = logp - flat_logp[c]
                                # An explored step's old probability is ~0, so its ratio can overflow: it stays out of the
                                # exponential, whose gradient would be NaN even through the branch `where` does not select.
                                ratio = drift.masked_fill(x, 0.0).exp()
                                clipped = ratio.clamp(1 - cfg.ppo.clip, 1 + cfg.ppo.clip) * a
                                # Explored steps: no importance weight (it would be ~0). Their log-probability moves with the
                                # sign of the advantage, by at most the clip per update either way (a one-sided term would
                                # raise every tried action whose returns are noisy).
                                tried = cfg.ppo.explore_weight * drift.clamp(-cfg.ppo.clip, cfg.ppo.clip) * a
                                surr = torch.where(x, tried, torch.min(ratio * a, clipped))
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
                            if cfg.ppo.aux_order:
                                add_order_grad(model, gaining, cfg.ppo.aux_order)
                            opt.step()
                            for v, s in zip((kls, clipfracs, pls, vls, ents), torch.stack(parts).mean(0).tolist()):
                                v.append(s)
                        # Judged per epoch, so one noisy minibatch does not cut the update short.
                        if not cfg.ppo.kl_controller and cfg.ppo.target_kl and _mean(kls[-cfg.ppo.minibatches :]) > 1.5 * cfg.ppo.target_kl:
                            break
                    if cfg.ppo.kl_controller:
                        update_kl_controller(cfg, update, _mean(kls), kl_ctl)
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
                    if wb is not None:
                        wb_d = {
                            f"train/{k}": float(v)
                            for k, v in row.items()
                            if k not in ("update", "stakes") and v is not None and not (isinstance(v, float) and math.isnan(v))
                        }
                        with torch.no_grad():
                            for k_, ps in w_groups:
                                if ps:
                                    wb_d[k_] = math.sqrt(
                                        float(torch.stack([q.float().square().sum() for q in ps]).sum())
                                        / sum(q.numel() for q in ps)
                                    )
                        if cfg.ppo.kl_controller and cfg.ppo.target_kl > 0:
                            wb_d["train/kl_factor"] = kl_ctl["m"]
                        for (deck, stake_), q in combo_recent.items():
                            if q:
                                d_short = deck[2:] if deck.startswith("b_") else deck
                                wb_d[f"combo/{d_short}_{stake_}"] = float(np.mean(q))
                        wb.log(wb_d, step=row["update"])
                    if len(set(cfg.env.backs)) * len(cfg.env.stakes) > 1:
                        with open(run_dir / "cells.jsonl", "a") as f:
                            f.write(json.dumps({"update": update + 1, **cells}) + "\n")
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
                        if wb is not None and isinstance(res.get("summary"), dict):
                            ev_d = {
                                f"eval/{k}": float(v)
                                for k, v in res["summary"].items()
                                if isinstance(v, (int, float)) and not math.isnan(v)
                            }
                            if ev_d:
                                wb.log(ev_d, step=completed)
                    if stop_requested or (run_dir / "STOP").exists():
                        if (run_dir / "STOP").exists():
                            (run_dir / "STOP").unlink()
                        break
            save(completed)
            if completed >= n_updates and not stop_requested:
                print("done", flush=True)
    finally:
        signal.signal(signal.SIGTERM, prev_sigterm)
        if wb is not None:
            wb.finish(exit_code=143 if stop_requested else 0)
    return last


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="base", choices=["smoke", "cpu", "base", "large", "run95"])
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
