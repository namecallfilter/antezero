"""Training configuration: every setting with its default (the headline run's recipe) and what it does."""
import ast
import dataclasses
from dataclasses import dataclass, fields
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    # ---- run ---------------------------------------------------------------------------------------------------------
    name: str = "run"
    out_dir: str = str(ROOT / "runs")
    seed: int = 0
    resume: bool = True
    init_from: str = ""
    save_every: int = 100
    keep_ckpt_every: int = 1000
    total_steps: int = 1_966_080_000

    # ---- environment -------------------------------------------------------------------------------------------------
    num_envs: int = 512
    num_steps: int = 128
    env_threads: int = 32
    max_steps: int = 3000
    max_free_actions: int = 10
    endless_ante: int = 16
    disable_softlocks: bool = True
    obs_hide_face_down: bool = False

    # ---- reward ------------------------------------------------------------------------------------------------------
    reward_mode: str = "blinds"
    win_log: bool = True
    reward_skip_deferred: bool = True
    money_per_reward: float = 15.0

    # ---- deck / stake (the curriculum) -------------------------------------------------------------------------------
    deck: str = "b_red"
    stake: int = 1
    challenge: str = "none"
    all_stakes: bool = True
    stake_only: int = 8
    all_white: bool = False
    curriculum: bool = False
    all_decks: bool = False
    unlock_rate: float = 0.02
    unlock_window: int = 200
    curriculum_uniform: bool = False

    # ---- start pool (mid-run states of recorded games) ---------------------------------------------------------------
    start_pool: str = "pools/human_gold_v2_ante6"
    start_frac: float = 0.2
    start_filter: str = ""
    start_ante_min: int = 0
    start_ante_max: int = 99
    start_reseed: bool = True

    # ---- model -------------------------------------------------------------------------------------------------------
    d_model: int = 256
    layers: int = 6
    heads: int = 8
    embed_from: str = "data/text_embeddings_256.pt"
    embed_freeze_updates: int = 0

    # ---- PPO and optimizer -------------------------------------------------------------------------------------------
    lr: float = 6e-4
    lr_schedule: str = "exp"
    lr_floor: float = 5e-5
    lr_decay_updates: int = 800
    optimizer: str = "muon"
    muon_wd: float = 0.05
    embed_adam: bool = True
    target_kl: float = 0.0012
    kl_start: int = 2000
    kl_high: float = 1.25
    kl_low: float = 0.75
    kl_step: float = 0.03
    kl_max_factor: float = 2.0
    kl_ema: float = 0.1
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 1
    minibatch_size: int = 4096
    microbatch_size: int = 2048
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # ---- speed -------------------------------------------------------------------------------------------------------
    bf16: bool = True
    compile: bool = True
    compile_mode: str = "default"
    len_buckets: int = 2
    len_round: int = 0
    fast_opt: bool = True
    buf_device: str = ""
    pinned_gather: bool = False

    def check(self):
        def bad(msg):
            raise SystemExit(f"config: {msg}")
        if self.reward_mode not in ("blinds", "best_score", "money"):
            bad("reward_mode: one of blinds, best_score, money")
        if sum(map(bool, (self.curriculum, self.all_white, self.all_stakes))) > 1:
            bad("curriculum, all_white and all_stakes are exclusive")
        if self.all_decks and not self.curriculum:
            bad("all_decks needs curriculum=1")
        if self.stake_only and not self.all_stakes:
            bad("stake_only needs all_stakes=1")
        if self.challenge != "none" and (self.curriculum or self.all_white or self.all_stakes):
            bad("challenge: not with curriculum / all_white / all_stakes (they include Jokerless already)")
        if self.minibatch_size > self.num_envs * self.num_steps:
            bad("minibatch_size is larger than an update's batch (num_envs * num_steps)")
        return self


def _value(text, current):
    try:
        v = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        v = text
    if isinstance(current, bool) and isinstance(v, str) and v.lower() in ("true", "false"):
        v = v.lower() == "true"
    elif isinstance(current, bool) and isinstance(v, int):
        v = bool(v)
    elif isinstance(current, float) and isinstance(v, int):
        v = float(v)
    if v is None and isinstance(current, str):
        v = ""
    if type(v) is not type(current):
        raise SystemExit(f"config: {text!r} is a {type(v).__name__}, expected {type(current).__name__}")
    return v


_ALIASES = {
    "model.d_model": "d_model",
    "model.n_layers": "layers",
    "model.n_heads": "heads",
    "ppo.total_steps": "total_steps",
    "ppo.num_envs": "num_envs",
    "ppo.num_steps": "num_steps",
    "ppo.minibatch_size": "minibatch_size",
}


def parse(argv=None):
    """Config from key=value overrides (plus compatibility with `--name`, `--preset run95`, `--set`, `--resume`)."""
    cfg = Config()
    names = {f.name for f in fields(Config)}

    def set_(k, raw):
        k = _ALIASES.get(k, k)
        if k == "grad_accum" or k == "ppo.grad_accum":
            ga = max(1, int(raw))
            cfg.microbatch_size = max(1, cfg.minibatch_size // ga)
            return
        if k not in names:
            raise SystemExit(f"config: unknown key {k!r}")
        setattr(cfg, k, _value(raw, getattr(cfg, k)))

    args = list(sys.argv[1:] if argv is None else argv)
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("--name", "-n") and i + 1 < len(args):
            cfg.name = args[i + 1]
            i += 2
        elif a == "--preset" and i + 1 < len(args):
            if args[i + 1] in ("run95", "0095", "0098"):
                cfg.d_model, cfg.layers, cfg.heads = 384, 8, 12
                cfg.embed_from = "data/text_embeddings_384_v2.pt"
                cfg.microbatch_size = 4096
                cfg.obs_hide_face_down = True
                cfg.keep_ckpt_every = 500
                cfg.total_steps = 1_310_720_000
            i += 2
        elif a == "--resume":
            cfg.resume = True
            if i + 1 < len(args) and "=" not in args[i + 1] and not args[i + 1].startswith("-"):
                i += 2
            else:
                i += 1
        elif a == "--set" and i + 1 < len(args):
            i += 1
            while i < len(args) and not args[i].startswith("-"):
                if "=" in args[i]:
                    k, v = args[i].split("=", 1)
                    set_(k.lstrip("-").replace("-", "_"), v)
                i += 1
        elif "=" in a:
            k, v = a.split("=", 1)
            set_(k.lstrip("-").replace("-", "_"), v)
            i += 1
        elif a.endswith((".yaml", ".yml")):
            import yaml
            for k, v in (yaml.safe_load(Path(a).read_text()) or {}).items():
                set_(k, repr(v) if not isinstance(v, str) else v)
            i += 1
        else:
            raise SystemExit(f"config: expected key=value or a .yaml file, got {a!r}")
    if cfg.d_model == 384 and cfg.embed_from == "data/text_embeddings_256.pt":
        cfg.embed_from = "data/text_embeddings_384_v2.pt"
    return cfg.check()


def as_dict(cfg):
    return dataclasses.asdict(cfg)
