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
import time
from collections import deque
from pathlib import Path

import antezero_sim as az
import numpy as np
import torch

from .config import Config, parse_overrides, preset
from .model import MAX_CARDS, Net, Obs, count_params, obs_from_numpy
from .runtime import make_env, pick_device, read_checkpoint, save_checkpoint
from .stats import wilson


class Rollout:
    """Fixed-size on-device storage for ``T`` steps of ``N`` envs. Integer
    features use compact (lossless) dtypes; floats stay float32 so training
    sees exactly the observations the actions were sampled on."""

    def __init__(self, T: int, N: int, device: torch.device):
        z = lambda *s, dt: torch.zeros(*s, dtype=dt, device=device)  # noqa: E731
        self.tok_i = z(T, N, az.MAX_TOKENS, az.TOK_INT, dt=torch.int16)
        self.tok_f = z(T, N, az.MAX_TOKENS, az.TOK_FLOAT, dt=torch.float32)
        self.glob_i = z(T, N, az.GLOB_INT, dt=torch.int16)
        self.glob_f = z(T, N, az.GLOB_FLOAT, dt=torch.float32)
        self.ops = z(T, N, az.MAX_OPS, az.OP_INT, dt=torch.int32)
        self.ops_f = z(T, N, az.MAX_OPS, az.OP_FLOAT, dt=torch.float32)
        self.n_ops = z(T, N, dt=torch.int16)
        self.n_tokens = z(T, N, dt=torch.int16)
        self.n_hand = z(T, N, dt=torch.int16)
        self.op = z(T, N, dt=torch.int64)
        self.card_seq = z(T, N, MAX_CARDS, dt=torch.int64)
        self.dest = z(T, N, dt=torch.int64)
        self.logp = z(T, N, dt=torch.float32)
        self.value = z(T, N, dt=torch.float32)
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

    def obs(self, idx) -> Obs:
        """Flat indices over T*N -> Obs (float32 / int64)."""
        f = lambda x: x.flatten(0, 1)[idx]  # noqa: E731
        return Obs(
            tok_i=f(self.tok_i).long(),
            tok_f=f(self.tok_f),
            glob_i=f(self.glob_i).long(),
            glob_f=f(self.glob_f),
            ops=f(self.ops).long(),
            ops_f=f(self.ops_f),
            n_ops=f(self.n_ops).long(),
            n_tokens=f(self.n_tokens).long(),
            n_hand=f(self.n_hand).long(),
        )


def gae(r: Rollout, last_value: torch.Tensor, gamma: float, lam: float):
    """Generalized advantage estimates and returns. ``r.done`` ends the
    recursion (episodes do not leak into the next one); truncated steps
    already carry ``gamma * V(final state)`` in their reward."""
    adv = torch.zeros_like(r.reward)
    last = torch.zeros_like(last_value)
    for t in reversed(range(r.T)):
        nonterm = 1.0 - r.done[t]
        nv = last_value if t == r.T - 1 else r.value[t + 1]
        delta = r.reward[t] + gamma * nv * nonterm - r.value[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + r.value


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


def _log_writer(f, path: Path) -> csv.DictWriter:
    """A writer of ``LOG_FIELDS`` rows appending to ``path`` (whose header,
    if any, must match)."""
    if path.stat().st_size:
        with open(path, newline="") as r:
            header = next(csv.reader(r), [])
        if header != LOG_FIELDS:
            raise ValueError(f"{path} has other columns than this version logs; move it away to resume")
    w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
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
        if (old.env.num_envs, old.ppo.rollout_len) != (cfg.env.num_envs, cfg.ppo.rollout_len):
            raise ValueError("cannot resume with another env.num_envs / ppo.rollout_len (the update count would change meaning)")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.csv"
    log_path.touch()

    torch.manual_seed(cfg.env.seed)
    model = Net(cfg.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.ppo.lr, weight_decay=cfg.ppo.weight_decay, eps=1e-5)
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
        save_checkpoint(run_dir / "last.pt", model, opt, cfg, update, steps, extra)

    env = make_env(cfg, seed=cfg.env.seed + 1_000_003 * start_update)
    N, T = env.num_envs, cfg.ppo.rollout_len
    roll = Rollout(T, N, device)
    # The schedule spans the whole run; `max_updates` only stops early.
    n_updates = cfg.ppo.total_steps // (N * T)
    stop_at = n_updates if max_updates is None else min(n_updates, start_update + max_updates)
    stats = EpisodeStats()
    B = N * T
    mb = B // cfg.ppo.minibatches
    print(f"[antezero] device={device} params={count_params(model):,} envs={N} rollout={T} updates={start_update}..{stop_at} of {n_updates}", flush=True)

    last: dict = {}
    completed = start_update
    with open(log_path, "a", newline="") as log_f:
        log_w = _log_writer(log_f, log_path)
        (run_dir / "config.json").write_text(cfg.to_json())
        for update in range(start_update, stop_at):
            t0 = time.perf_counter()
            lr, ent_coef = schedule(cfg, update, n_updates)
            for g in opt.param_groups:
                g["lr"] = lr
            top_stake = max(cfg.env.stakes)
            # ------------------------------------------------------ rollout
            model.eval()
            for t in range(T):
                o = obs_from_numpy(env, device)
                roll.put_obs(t, o)
                with autocast():
                    out = model.act(o)
                roll.op[t] = out["op"]
                roll.card_seq[t] = out["card_seq"]
                roll.dest[t] = out["dest"]
                roll.logp[t] = out["logp"].float()
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
                        cls, _tok, _ope, _lg = model.encode(obs_from_numpy(env, device, final=True, rows=rows))
                        v_fin = model.value(cls).squeeze(-1).float()
                    rew[torch.as_tensor(rows, device=device)] += cfg.ppo.gamma * v_fin
                roll.reward[t] = rew
                roll.done[t] = torch.as_tensor(done, device=device, dtype=torch.float32)
                new_eps = env.pop_episodes()
                stats.add(new_eps)
                top.extend(bool(x[5]) for x in new_eps if not x[10] and x[3] == top_stake)
            with torch.no_grad(), autocast():
                cls, _tok, _ope, _lg = model.encode(obs_from_numpy(env, device))
                last_value = model.value(cls).squeeze(-1).float()
            flat_op, flat_seq, flat_dest = roll.op.flatten(), roll.card_seq.flatten(0, 1), roll.dest.flatten()
            logp_drift = 0.0
            if cfg.ppo.recompute_logp:
                with torch.no_grad(), autocast():
                    for k in range(cfg.ppo.minibatches):
                        idx = torch.arange(k * mb, (k + 1) * mb, device=device)
                        lp, _e, v = model.evaluate(roll.obs(idx), flat_op[idx], flat_seq[idx], flat_dest[idx])
                        tt, nn_ = idx // N, idx % N
                        logp_drift = max(logp_drift, float((lp.float() - roll.logp[tt, nn_]).abs().max()))
                        roll.logp[tt, nn_] = lp.float()
                        roll.value[tt, nn_] = v.float()
            adv, ret = gae(roll, last_value, cfg.ppo.gamma, cfg.ppo.gae_lambda)
            t_roll = time.perf_counter() - t0
            steps += B

            # ------------------------------------------------------- update
            model.train()
            flat_adv, flat_ret = adv.flatten(), ret.flatten()
            flat_logp, flat_val = roll.logp.flatten(), roll.value.flatten()
            kls, clipfracs, pls, vls, ents = [], [], [], [], []
            stop = False
            for _epoch in range(cfg.ppo.epochs):
                perm = torch.randperm(B, device=device)
                for k in range(cfg.ppo.minibatches):
                    idx = perm[k * mb : (k + 1) * mb]
                    with autocast():
                        logp, ent, val = model.evaluate(roll.obs(idx), flat_op[idx], flat_seq[idx], flat_dest[idx])
                    logp, ent, val = logp.float(), ent.float(), val.float()
                    with torch.no_grad():
                        lr_ = logp - flat_logp[idx]
                        kls.append(float(((lr_.exp() - 1) - lr_).mean()))
                    # Stop before stepping on a policy already too far away.
                    if cfg.ppo.target_kl and kls[-1] > 1.5 * cfg.ppo.target_kl:
                        stop = True
                        break
                    a = flat_adv[idx]
                    a = (a - a.mean()) / (a.std() + 1e-8)
                    ratio = (logp - flat_logp[idx]).exp()
                    pl = -torch.min(ratio * a, ratio.clamp(1 - cfg.ppo.clip, 1 + cfg.ppo.clip) * a).mean()
                    if cfg.ppo.value_clip > 0:
                        v_old = flat_val[idx]
                        v_clip = v_old + (val - v_old).clamp(-cfg.ppo.value_clip, cfg.ppo.value_clip)
                        vl = 0.5 * torch.max((val - flat_ret[idx]) ** 2, (v_clip - flat_ret[idx]) ** 2).mean()
                    else:
                        vl = 0.5 * ((val - flat_ret[idx]) ** 2).mean()
                    loss = pl + cfg.ppo.vf_coef * vl - ent_coef * ent.mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.ppo.max_grad_norm)
                    opt.step()
                    clipfracs.append(float(((ratio.detach() - 1).abs() > cfg.ppo.clip).float().mean()))
                    pls.append(float(pl.detach()))
                    vls.append(float(vl.detach()))
                    ents.append(float(ent.detach().mean()))
                if stop:
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
                "logp_drift": logp_drift,
                "invalid_actions": env.invalid_actions(),
                "stakes": ",".join(map(str, cfg.env.stakes)),
                **stats.summary(),
            }
            last = row
            log_w.writerow(row)
            log_f.flush()
            if cfg.log_every and (update + 1) % cfg.log_every == 0:
                s = row
                num = lambda k, spec: "-" if s[k] is None else format(s[k], spec)  # noqa: E731
                print(
                    f"[{s['update']}] steps={s['steps']:,} fps={s['fps']:,} win={num('win_rate', '.3f')} "
                    f"[{num('win_lo', '.3f')},{num('win_hi', '.3f')}] ante={num('ante_mean', '.2f')} max={num('ante_max', 'd')} "
                    f"ret={num('ep_return', '.3f')} ent={s['entropy']:.3f} kl={s['approx_kl']:.4f} vl={s['value_loss']:.4f}",
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
                    env = make_env(cfg, seed=e.seed + 7919 * (update + 1))
                    top.clear()
                    (run_dir / "config.json").write_text(cfg.to_json())
            completed = update + 1
            if cfg.save_every and completed % cfg.save_every == 0:
                save(completed)
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
