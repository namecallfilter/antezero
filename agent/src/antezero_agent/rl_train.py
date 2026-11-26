"""PPO training (RLBalatro recipe on the antezero Rust engine)."""
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn

from . import speed as SP
from .curriculum import Combos, StartStates
from .loggers import CompletionTracker, ConsTracker, HandTracker, JokerTracker, PlayStats, RunLogger
from .models import hand_select as HS
from .models.rl_policy import Policy, _hand_tables
from .ppo import PPOTrainer, build_optimizer, compute_lr, update_kl_controller
from .rl_config import as_dict, parse
from .rl_env import BalatroEnv, BalatroVecEnv

PKG = Path(__file__).resolve().parent
AGENT_ROOT = PKG.parents[1]


def resolve(path):
    """A relative path as given (from the working directory), else inside agent/, else the repo."""
    p = Path(path)
    for cand in (p, AGENT_ROOT / p, PKG / p, PKG.parent / p, AGENT_ROOT.parent / p):
        if cand.exists():
            return cand
    raise FileNotFoundError(path)


def code_version():
    """The git commit (and whether the tree had local changes) and a hash of the package's sources."""
    out = {}
    try:
        g = lambda *a: subprocess.run(["git", "-C", str(PKG), *a], capture_output=True, text=True, timeout=10)  # noqa: E731
        r = g("rev-parse", "HEAD")
        if r.returncode == 0:
            out["commit"] = r.stdout.strip()
            out["dirty"] = bool(g("status", "--porcelain", "--untracked-files=no").stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    h = hashlib.sha256()
    for f in sorted(PKG.rglob("*.py")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    out["source_sha256"] = h.hexdigest()[:16]
    return out


def env_kwargs(cfg):
    """The BalatroEnv settings of a run (training, evaluation and replays use the same)."""
    return dict(max_steps=cfg["max_steps"], max_free_actions=cfg["max_free_actions"],
                endless_ante=cfg["endless_ante"], disable_softlocks=cfg["disable_softlocks"],
                obs_hide_face_down=cfg["obs_hide_face_down"], reward_win_log=cfg["win_log"],
                reward_skip_deferred=cfg["reward_skip_deferred"], reward_money=cfg["reward_mode"] == "money",
                money_per_reward=cfg["money_per_reward"], reward_best_score=cfg["reward_mode"] == "best_score")


def model_config(cfg):
    return {"d_model": cfg.d_model, "layers": cfg.layers, "heads": cfg.heads,
            "deck_emb": bool(cfg.curriculum or cfg.all_white or cfg.all_stakes)}


def load_model(path, groups, dev="cpu"):
    """A Policy from a model file ({"model": state dict, "config": model_config}) or a run directory."""
    p = Path(path)
    if p.is_dir():
        p = p / "model.pt" if (p / "model.pt").exists() else p / "ckpt.pt"
    ck = torch.load(p, map_location=dev, weights_only=True)
    mc = ck["config"]
    model = Policy(groups, mc["d_model"], mc["layers"], mc["heads"], deck_emb=mc["deck_emb"]).to(dev)
    model.load_state_dict(ck["model"])
    return model, mc


def copy_text_embeddings(model, path, dev):
    """The embedding tables of `path` (text embeddings of the game's cards) copied into the model's."""
    sd = torch.load(resolve(path), map_location=dev)
    params = []
    with torch.no_grad():
        for n, m in model.named_modules():
            if isinstance(m, nn.Embedding) and f"{n}.weight" in sd:
                m.weight.copy_(sd[f"{n}.weight"])
                params.append(m.weight)
    print(f"embeddings from {path}: {len(params)} tables", flush=True)
    return params


def compile_policy(model, cfg, dev):
    """compile=1: torch.compile'd forward / card picker sites (each falls back to eager if compiling fails)."""
    model.len_buckets = cfg.len_buckets
    model.len_round = cfg.len_round or (8 if cfg.len_buckets > 1 or cfg.compile else 1)
    if not cfg.compile:
        return model.pick_cards, model.pick_cards
    if SP.toolchain_problem():
        print(f"WARNING: compile=1: {SP.toolchain_problem()}; trying anyway", flush=True)
    SP.COMPILE_MODE["mode"] = cfg.compile_mode
    SP.configure()
    _hand_tables(dev), HS._const(dev)
    model.rows_fn = {False: SP.CompiledSite("fwd_roll", model._rows), True: SP.CompiledSite("fwd_train", model._rows)}
    return SP.CompiledSite("pick_roll", model.pick_cards), SP.CompiledSite("pick_train", model.pick_cards)


def compile_probe(model, x, mask, spec_t, hctx_t, hctx_np, amp, opt, pick_roll, pick_train):
    """Run every compiled site once on the first observations."""
    rng = torch.get_rng_state(), (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    t0 = time.time()
    try:
        a, _, _, pk, sp, pm = model.act(x, mask, spec_t, hctx_t, hctx_np, amp=amp, pick_fn=pick_roll)
        model(x)
        model.train()
        lp, v, ent = model.evaluate_actions(x, mask, a, sp, pk, pm, amp=amp, pick_fn=pick_train)
        if not SP.is_off():
            (lp.float().sum() + v.sum() + ent.float()).backward()
    except Exception as ex:   # noqa: BLE001
        SP.disable(f"probe: {type(ex).__name__}: {str(ex).strip()[:300]}")
    opt.zero_grad()
    model.eval()
    torch.set_rng_state(rng[0])
    if rng[1] is not None:
        torch.cuda.set_rng_state_all(rng[1])
    print(f"compile probe {time.time() - t0:.1f} s; {SP.summary()}", flush=True)


class GameStats:
    """Per-env totals of the game in progress, and the last 2000 finished games."""

    def __init__(self, E):
        self.ret, self.blinds, self.money, self.best = (np.zeros(E) for _ in range(4))
        self.recent, self.recent_starts = deque(maxlen=2000), deque(maxlen=2000)
        self.best_score = 0.0

    def step(self, r, info):
        self.ret += r
        self.blinds += info["blinds"]
        self.money += info["earned"]
        np.maximum(self.best, info["hand_score"], out=self.best)

    def game_over(self, i, info, start_ante, endless_ante):
        won, ante, score = bool(info["won"][i]), int(info["ante"][i]), float(info["score"][i])
        if start_ante is not None:
            self.recent_starts.append({"win": won, "blinds": self.blinds[i], "ante": ante,
                                       "ante_gain": ante - start_ante, "return": self.ret[i]})
        else:
            self.recent.append({"return": self.ret[i], "ante": ante, "win": won, "blinds": self.blinds[i],
                                "game_steps": info["steps"][i], "win_log10": np.log10(max(1.0, score)) if won else np.nan,
                                "win_endless": won and endless_ante > 8 and ante > endless_ante,
                                "money": self.money[i], "best_hand": np.log10(max(1.0, self.best[i]))})
            if won:
                self.best_score = max(self.best_score, score)
        self.ret[i] = self.blinds[i] = self.money[i] = self.best[i] = 0

    def row(self):
        def means(q, prefix=""):
            keys = q[0].keys() if q else ()
            return {prefix + k: float(np.nanmean([g[k] for g in q])) if any(not np.isnan(g[k]) for g in q) else np.nan
                    for k in keys}
        out = means(self.recent)
        if self.recent_starts:
            n = len(self.recent_starts)
            out["start_share"] = n / (n + len(self.recent))
            out.update(means(self.recent_starts, "start_"))
        return out


def main(argv=None):
    cfg = parse(argv)
    cfgd = as_dict(cfg)
    run = Path(cfg.out_dir) / cfg.name
    run.mkdir(parents=True, exist_ok=True)
    json.dump({**cfgd, "code": code_version()}, open(run / "config.json", "w"), indent=1)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    amp = dev.type == "cuda" and cfg.bf16
    E, T = cfg.num_envs, cfg.num_steps
    challenge = None if cfg.challenge == "none" else cfg.challenge
    deck = "b_challenge" if challenge else cfg.deck

    # ---- env, model, optimizer -------------------------------------------------------------------------------------
    env = BalatroVecEnv(E, deck=deck, stake=cfg.stake, threads=cfg.env_threads, seed_start=cfg.seed * 1_000_000_000,
                        challenge=challenge, **env_kwargs(cfgd))
    model = Policy(env.groups, cfg.d_model, cfg.layers, cfg.heads, deck_emb=model_config(cfg)["deck_emb"]).to(dev)
    pick_roll, pick_train = compile_policy(model, cfg, dev)
    emb_params = copy_text_embeddings(model, cfg.embed_from, dev) if cfg.embed_from else []
    opt, emb_ids = build_optimizer(model, cfg, dev, emb_params)
    obs_pin = None
    if dev.type == "cuda":
        obs_pin = torch.zeros(env.obs_buffer.shape, dtype=torch.float32, pin_memory=True)
        env._obs = obs_pin.numpy()

    def inputs(o, info):
        b = (obs_pin if obs_pin is not None else torch.from_numpy(env.obs_buffer)).to(dev)
        return (torch.cat([b, torch.from_numpy(o["selection"]).to(dev)], 1), torch.from_numpy(info["action_mask"]).to(dev),
                torch.from_numpy(info["sel_spec"]).to(dev).long(), torch.from_numpy(info["hand_ctx"]).to(dev),
                info["hand_ctx"])

    # ---- state: fresh, initialized from weights, or resumed ------------------------------------------------------
    combos = Combos(cfg, env, np.random.default_rng(cfg.seed))
    completion = CompletionTracker(JokerTracker(1).key_index)
    st = {"steps": 0, "games": 0, "update": 0, "blinds_attempted": 0, "kl_ctl": {"m": 1.0, "ema": None}}
    if cfg.resume and (run / "ckpt.pt").exists():
        ck = torch.load(run / "ckpt.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        st = ck["state"]
        env.next_seed = st["next_seed"]
        combos.load(ck["combos"])
        completion.load(ck["completion"])
        combos.rng = np.random.default_rng(cfg.seed + st["update"])
        print(f"resumed from {run / 'ckpt.pt'} at update {st['update']}", flush=True)
    elif cfg.init_from:
        src = resolve(cfg.init_from) if Path(cfg.init_from).suffix == ".pt" else Path(cfg.out_dir) / cfg.init_from
        init, _ = load_model(src, env.groups, dev)
        missing = model.load_state_dict(init.state_dict(), strict=False).missing_keys
        assert all(k.startswith(("deck_emb", "stake_emb")) for k in missing), missing
        with torch.no_grad():
            for m in (model.deck_emb, model.stake_emb) if missing else ():
                m.weight.zero_()
        print(f"initialized from {src}", flush=True)
    print(f"{cfg.name}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params on {dev}, {E} envs x {T} "
          f"steps, reward {cfg.reward_mode}", flush=True)

    starts = StartStates(cfg, env, resolve(cfg.start_pool) if cfg.start_frac > 0 else None, cfg.seed, st["update"])
    starts.start()
    combos.start()
    o, info = env.reset()
    starts.after_reset()
    x, mask, spec_t, hctx_t, hctx_np = inputs(o, info)
    if cfg.compile:
        compile_probe(model, x, mask, spec_t, hctx_t, hctx_np, amp, opt, pick_roll, pick_train)

    trainer = PPOTrainer(model, opt, cfg, dev, torch.device(cfg.buf_device) if cfg.buf_device else dev,
                         x.shape[1], env.n_actions, amp=amp, pick_train=pick_train,
                         pgather=SP.PinnedGather(dev) if cfg.pinned_gather and cfg.buf_device == "cpu" else None)
    logger = RunLogger(run, cfgd, opt, cfg.resume)
    cons, jok, hands, plays, games = ConsTracker(env.actions), JokerTracker(E), HandTracker(E), PlayStats(), GameStats(E)
    best = json.load(open(run / "best.json")) if cfg.resume and (run / "best.json").exists() else {}
    (run / "best").mkdir(exist_ok=True)
    game_seed = np.arange(env.next_seed - E, env.next_seed, dtype=np.uint64)
    game_acts = np.zeros((E, cfg.max_steps + 16), np.int16)
    game_picks = np.full((E, cfg.max_steps + 16, 5), -1, np.int8)
    game_len = np.zeros(E, np.int64)

    def save_best(i, c, score):
        d, s = c
        e = BalatroEnv(deck="b_challenge" if d == "jokerless" else d, stake=s, record=True,
                       challenge="jokerless" if d == "jokerless" else challenge, **env_kwargs(cfgd))
        e.reset(seed=int(game_seed[i]))
        for a_, pk in zip(game_acts[i, :game_len[i]], game_picks[i, :game_len[i]]):
            *_, inf = e.step((int(a_), [int(q) for q in pk if q >= 0]))
        if inf.get("won"):
            (run / "best" / f"{d}.bal").write_text(e.script())
            best[d] = {"stake": int(s), "score": float(score), "seed": int(game_seed[i]), "update": st["update"]}
            json.dump(best, open(run / "best.json", "w"), indent=1)
        e.close()

    prev_round = np.zeros(E, np.int32)
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    t_start, s_start, games_at_save = time.time(), st["steps"], st["games"]
    while st["steps"] < cfg.total_steps and not stop:
        # ---- learning rate ---------------------------------------------------------------------------------------
        lr = compute_lr(cfg, st["update"], st["steps"], st["kl_ctl"])
        frozen = bool(emb_params) and (cfg.embed_freeze_updates < 0 or st["update"] < cfg.embed_freeze_updates)
        for gp in opt.param_groups:
            gp["lr"] = 0.0 if frozen and any(id(p) in emb_ids for p in gp["params"]) else lr
        # ---- rollout: T steps of E envs -----------------------------------------------------------------------------
        model.eval()
        illegal = dead = softlocks = 0
        for t in range(T):
            a, lp, v, pk, sp, pm = model.act(x, mask, spec_t, hctx_t, hctx_np, amp=amp, pick_fn=pick_roll)
            trainer.buffer.store_step(t, x, mask, a, lp, v, pk, sp, pm)
            apk = torch.cat([a[:, None], pk], 1).cpu().numpy()
            a_np, pk_np = apk[:, 0], np.ascontiguousarray(apk[:, 1:], dtype=np.int32)
            step_idx = np.minimum(game_len, game_acts.shape[1] - 1)
            game_acts[np.arange(E), step_idx], game_picks[np.arange(E), step_idx] = a_np, pk_np
            game_len += 1
            cons.before(o, a_np)
            jok.observe(o)
            hands.observe(o)
            o, r, term, trunc, info = env.step(a_np, pk_np)
            done = term | trunc
            ok = ~info["illegal"].astype(bool)
            plays.add(info, ok)
            cons.after(info["illegal"])
            games.step(r, info)
            jok.blinds_beaten(info["blinds"])
            illegal += int((~ok).sum())
            dead += int(info["dead_end"].sum())
            softlocks += int(info["softlocks_avoided"].sum())
            st["blinds_attempted"] += int(np.maximum(0, info["round"] - prev_round).sum())
            prev_round = np.where(done, 0, info["round"])
            for i in np.flatnonzero(done):
                won = bool(info["won"][i])
                start_ante = starts.game_over(i)
                new_run = start_ante is None
                if won and new_run:
                    completion.won(combos.cur[i][1], jok.last[i])
                    c, prev = combos.cur[i], best.get(combos.cur[i][0])
                    if prev is None or (c[1], float(info["score"][i])) > (prev["stake"], prev["score"]):
                        save_best(i, c, info["score"][i])
                jok.game_over(i, won, int(info["ante"][i]))
                hands.game_over(i, won)
                games.game_over(i, info, start_ante, cfg.endless_ante)
                combos.game_over(i, won, info["ante"][i], new_run)
                game_seed[i], game_len[i] = env.reset_seeds[i], 0
            st["games"] += int(done.sum())
            trainer.buffer.store_outcome(t, r, done)
            x, mask, spec_t, hctx_t, hctx_np = inputs(o, info)
        st["steps"] += T * E
        # ---- PPO update ------------------------------------------------------------------------------------------
        pg, vl, ent, kl = trainer.update(x)
        st["update"] += 1
        update_kl_controller(cfg, st["update"], kl, st["kl_ctl"])
        # ---- log ---------------------------------------------------------------------------------------------------
        sps = (st["steps"] - s_start) / (time.time() - t_start)
        stake = cfg.stake_only if cfg.all_stakes and cfg.stake_only else cfg.stake
        row = {"update": st["update"], "steps": st["steps"], "games": st["games"], "sps": round(sps), **games.row(),
               "illegal": illegal, "pg": pg, "vl": vl, "ent": ent, "kl": kl, "lr": lr, "kl_factor": st["kl_ctl"]["m"],
               "best_score": games.best_score, "unlocked": len(combos.unlocked), "passed": len(combos.passed),
               "combos_won": sum(1 for c in combos.unlocked if combos.wins.get(c)), "dead_ends": dead,
               "softlocks_avoided": softlocks, "blinds_attempted": st["blinds_attempted"],
               "jokers_won1": completion.coverage(stake, 1), "jokers_won100": completion.coverage(stake, 100),
               **plays.row()}
        logger.write(row)
        plays.reset()
        if cfg.compile and st["update"] <= 2:
            print(f"update {st['update']}: {SP.summary()}", flush=True)
        f = lambda k, spec: format(row[k], spec) if k in row else "nan"   # noqa: E731
        print(f"update {st['update']:5d}  steps {st['steps']:>12,}  games {st['games']:>9,}  {sps:6.0f} sps  "
              f"return {f('return', '.3f')}  ante {f('ante', '.2f')}  win {f('win', '.2%')}  blinds/g "
              f"{f('blinds', '.2f')}  money ${f('money', '.2f')}  best hand 10^{f('best_hand', '.3f')}  "
              f"ent {ent:.2f}  kl {kl:.4f}" + (f"  starts: win {f('start_win', '.2%')} ante +{f('start_ante_gain', '.2f')}"
                                               if starts.on else ""), flush=True)
        # ---- checkpoint ------------------------------------------------------------------------------------------
        keep = cfg.keep_ckpt_every > 0 and st["update"] % cfg.keep_ckpt_every == 0
        last = st["steps"] >= cfg.total_steps or stop
        if st["update"] % cfg.save_every == 0 or keep or last:
            st["next_seed"] = env.next_seed
            torch.save({"model": model.state_dict(), "config": model_config(cfg), "opt": opt.state_dict(),
                        "state": st, "combos": combos.state(), "completion": completion.state()}, run / "ckpt.tmp")
            os.replace(run / "ckpt.tmp", run / "ckpt.pt")
            (run / "last.pt").unlink(missing_ok=True)
            os.link(run / "ckpt.pt", run / "last.pt")
            logger.interval_tables(st["update"], st["steps"], st["games"], st["games"] - games_at_save, completion,
                                   cons, jok, hands, combos, best)
            games_at_save = st["games"]
            if keep:
                (run / "ckpts").mkdir(exist_ok=True)
                kp = run / "ckpts" / f"ckpt_u{st['update']:05d}.pt"
                kp.unlink(missing_ok=True)
                os.link(run / "ckpt.pt", kp)
    torch.save({"model": model.state_dict(), "config": model_config(cfg)}, run / "model.pt")
    if cfg.compile:
        print(SP.summary(), flush=True)
    logger.finish(stopped=bool(stop))
    if stop:
        print(f"stopped by SIGTERM at update {st['update']}; checkpoint saved", flush=True)
        raise SystemExit(143)
    print("done", flush=True)


if __name__ == "__main__":
    main()
