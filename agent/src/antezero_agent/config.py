"""Configuration for the antezero agent (model, environment, PPO).

Presets:

* ``smoke``  - tiny, for tests (seconds on a laptop CPU).
* ``cpu``    - small model / few envs, usable for debugging on a laptop.
* ``base``   - the intended starting point on a single modern GPU.
* ``large``  - a bigger model and batch for long runs once ``base``
  plateaus (still a single device: there is no multi-GPU training).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field

import antezero_sim as az


@dataclass
class ModelConfig:
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    ff_mult: int = 4
    # Must stay 0 for PPO (the behaviour and the updated policy would see
    # different dropout masks); kept for other uses of the network.
    dropout: float = 0.0


@dataclass
class EnvConfig:
    num_envs: int = 512
    backs: list[str] = field(default_factory=lambda: ["b_red"])
    stakes: list[int] = field(default_factory=lambda: [1])
    # Episodes end at the win (ante 8 boss) unless `endless`, which keeps
    # playing (truncated, with bootstrapping, beyond `max_ante`). The policy
    # sees the flag, so one network can serve both objectives.
    endless: bool = False
    max_ante: int = 16
    step_limit: int = 5000
    # True reward: winning the run, each blind cleared after the win (endless
    # objective), losing. Progress shaping is potential-based (policy
    # invariant): Phi = w_blind * blinds cleared + w_score * round progress,
    # F = gamma * Phi(s') - Phi(s), Phi(terminal) = 0. With w_blind ~ r_win /
    # 24 the potential is a rough prior on the value (the win is 24 blinds).
    r_win: float = 1.0
    r_endless: float = 0.0
    r_loss: float = 0.0
    w_blind: float = 0.04
    w_score: float = 0.02
    threads: int = 0  # 0 = all cores
    # Offer the K best plays (sim score on chance scenarios that are shared
    # by all plays, face-up cards only) as extra ops with score statistics.
    play_candidates: int = 8
    # Tokenize the remaining deck / discard pile card by card (otherwise
    # only histograms in the global features; ~3x fewer tokens).
    deck_tokens: bool = False
    seed: int = 1
    # Restart a fraction of episodes from ante-start snapshots of recent
    # episodes (hidden state resampled) so late antes get trained more.
    # Such episodes are excluded from the win-rate statistics.
    snapshot_prob: float = 0.0
    snapshot_min_ante: int = 2
    # Stake curriculum: the next stake is added once the 95% Wilson lower
    # bound of the win rate over the latest `curriculum_min_episodes` fresh
    # episodes on the top stake (since the last change) reaches the target.
    curriculum: bool = False
    curriculum_target: float = 0.8
    curriculum_min_episodes: int = 5000


@dataclass
class PPOConfig:
    total_steps: int = 2_000_000_000
    rollout_len: int = 128
    gamma: float = 0.999
    gae_lambda: float = 0.95
    epochs: int = 3
    # `num_envs * rollout_len` must split into this many equal minibatches
    # of at least 2 samples.
    minibatches: int = 8
    clip: float = 0.2
    value_clip: float = 0.0  # 0 = plain value MSE
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    ent_coef_final: float = 0.002
    lr: float = 3e-4
    lr_final: float = 3e-5
    warmup_updates: int = 20
    weight_decay: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    # bf16 autocast (CUDA devices that support bf16, and CPU when amp_cpu).
    amp: bool = True
    amp_cpu: bool = True
    compile: bool = False
    # Recompute the behaviour log-probs / values after each rollout with the
    # training forward pass (minibatch-sized batches under the same autocast)
    # instead of keeping the sampling ones. Under bf16 this removes most of
    # the sampling / training rounding mismatch from the PPO ratio; batch
    # composition still differs, so the first ratio is close to, not exactly,
    # 1 (`logp_drift` in the log is the largest change).
    recompute_logp: bool = True


@dataclass
class Config:
    name: str = "base"
    model: ModelConfig = field(default_factory=ModelConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    device: str = "auto"
    log_every: int = 1
    save_every: int = 50
    eval_every: int = 200
    eval_episodes: int = 256
    run_dir: str = "agent/runs"

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)

    @staticmethod
    def from_dict(d: dict) -> Config:
        """The inverse of ``to_json``; unknown fields are an error."""
        sections = {"model": ModelConfig, "env": EnvConfig, "ppo": PPOConfig}
        known = {f.name for f in dataclasses.fields(Config)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown config fields {sorted(unknown)}")
        kw = {}
        for k, v in d.items():
            if k in sections:
                cls = sections[k]
                bad = set(v) - {f.name for f in dataclasses.fields(cls)}
                if bad:
                    raise ValueError(f"unknown {k} fields {sorted(bad)}")
                kw[k] = cls(**v)
            else:
                kw[k] = v
        return Config(**kw)

    def validate(self) -> None:
        """Rejects settings training cannot run with (before anything is
        allocated or written)."""
        e, p = self.env, self.ppo
        checks = [
            (e.num_envs > 0 and p.rollout_len > 0, "env.num_envs and ppo.rollout_len must be positive"),
            (p.minibatches > 0 and p.epochs > 0, "ppo.minibatches and ppo.epochs must be positive"),
            (
                (e.num_envs * p.rollout_len) % max(1, p.minibatches) == 0 and e.num_envs * p.rollout_len // max(1, p.minibatches) >= 2,
                f"num_envs * rollout_len ({e.num_envs * p.rollout_len}) must split into {p.minibatches} minibatches of >= 2 samples",
            ),
            (0.0 < p.gamma <= 1.0 and 0.0 <= p.gae_lambda <= 1.0, "ppo.gamma must be in (0, 1] and ppo.gae_lambda in [0, 1]"),
            (self.model.dropout == 0.0, "model.dropout must be 0 for PPO"),
            (0 <= e.play_candidates <= az.MAX_CANDIDATES, f"env.play_candidates must be in 0..={az.MAX_CANDIDATES}"),
            (bool(e.backs) and bool(e.stakes), "env.backs and env.stakes must not be empty"),
            (not e.curriculum or e.curriculum_min_episodes > 0, "env.curriculum_min_episodes must be positive"),
        ]
        for ok, msg in checks:
            if not ok:
                raise ValueError(msg)


def preset(name: str) -> Config:
    c = Config(name=name)
    if name == "smoke":
        c.model = ModelConfig(d_model=32, n_layers=1, n_heads=2)
        c.env.num_envs = 8
        c.env.threads = 2
        c.ppo.rollout_len = 8
        c.ppo.minibatches = 2
        c.ppo.epochs = 1
        c.ppo.total_steps = 8 * 8 * 2
        c.ppo.warmup_updates = 0
        c.eval_episodes = 4
        c.save_every = 1
        c.eval_every = 0
    elif name == "cpu":
        c.model = ModelConfig(d_model=128, n_layers=3, n_heads=4)
        c.env.num_envs = 64
        c.ppo.rollout_len = 64
        c.ppo.minibatches = 4
    elif name == "base":
        pass
    elif name == "large":
        c.model = ModelConfig(d_model=512, n_layers=10, n_heads=8)
        c.env.num_envs = 2048
        c.ppo.rollout_len = 64
        c.ppo.minibatches = 16
        c.ppo.lr = 2e-4
    else:
        raise ValueError(f"unknown preset {name!r}")
    return c


def _parse_value(cur, val: str):
    if isinstance(cur, bool):
        low = val.lower()
        if low in ("1", "true", "yes", "on"):
            return True
        if low in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"not a boolean: {val!r}")
    if isinstance(cur, int):
        try:
            return int(val)
        except ValueError:
            f = float(val)  # allow 2e9
            if not f.is_integer():
                raise ValueError(f"not an integer: {val!r}") from None
            return int(f)
    if isinstance(cur, float):
        return float(val)
    if isinstance(cur, list):
        elem = type(cur[0]) if cur else str
        return [_parse_value(elem(), x) if elem is not str else x for x in val.split(",")]
    return val


def parse_overrides(cfg: Config, sets: list[str]) -> None:
    """Applies ``section.field=value`` overrides (``ValueError`` on unknown
    fields or values of the wrong type)."""
    for s in sets:
        key, sep, val = s.partition("=")
        if not sep:
            raise ValueError(f"override {s!r} is not key=value")
        obj = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p, None)
            if not dataclasses.is_dataclass(obj):
                raise ValueError(f"unknown config section in {key!r}")
        if parts[-1] not in {f.name for f in dataclasses.fields(obj)} or dataclasses.is_dataclass(getattr(obj, parts[-1])):
            raise ValueError(f"unknown config field {key!r}")
        try:
            setattr(obj, parts[-1], _parse_value(getattr(obj, parts[-1]), val))
        except ValueError as e:
            raise ValueError(f"{key}: {e}") from None
