"""Evaluation: win rate (with 95% Wilson intervals) and endless survival on
held-out seeds.

    uv run python -m antezero_agent.eval agent/runs/run1/last.pt --episodes 10000
    uv run python -m antezero_agent.eval CKPT --backs b_red,b_blue --stakes 1,8
    uv run python -m antezero_agent.eval CKPT --endless --max-ante 20
    uv run python -m antezero_agent.eval CKPT --episodes 200 --search   # ablation
    uv run python -m antezero_agent.eval CKPT --episodes 512 --round-search --device cuda
    uv run python -m antezero_agent.eval CKPT --episodes 512 --round-search --endgame-search --device cuda

The episodes are played in batches of at most ``--batch`` parallel games
(batch ``k`` uses seed ``--seed + k``, so plain, ``--search`` and
``--round-search`` evaluation play the same games). Every env slot counts
exactly one episode (finished slots keep running but are ignored), so short
losing games do not bias the statistics. Seeds come from a range disjoint
from training. The policy only sees player-visible information (see
``sim/src/obs``).

``win_rate`` counts runs cut off by the step limit (``unresolved``) as not
won: it is the rate of runs won within the limit, a lower bound on the
policy's win rate.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from collections import Counter
from collections.abc import Iterator

import antezero_sim as az
import numpy as np
import torch

from .config import Config
from .model import Net, obs_from_numpy
from .runtime import load_model, make_env, seeded
from .stats import episodes_needed, wilson

EVAL_SEED = 7_777_777
BATCH = 1024


def _envs(cfg: Config, episodes: int, seed: int, backs, stakes, max_ante, endless, batch: int) -> Iterator[az.VecEnv]:
    """The evaluation games, ``batch`` at a time."""
    e = cfg.env
    for k, start in enumerate(range(0, episodes, batch)):
        yield make_env(
            cfg,
            seed=seed + k,
            num_envs=min(batch, episodes - start),
            backs=list(backs or e.backs),
            stakes=list(stakes or e.stakes),
            endless=e.endless if endless is None else endless,
            max_ante=max_ante or e.max_ante,
            snapshot_prob=0.0,
        )


def _rate(won: list[bool]) -> dict:
    k, n = int(sum(won)), len(won)
    lo, hi = wilson(k, n)
    return {"n": n, "wins": k, "win_rate": k / n if n else 0.0, "ci95": [round(lo, 4), round(hi, 4)]}


def summarize(eps: list[tuple], invalid: int = 0) -> dict:
    """Statistics of episode tuples ``(slot, seed, back, stake, ante, won,
    blinds, steps, return, truncated, from_snapshot)``."""
    antes = np.array([x[4] for x in eps]) if eps else np.zeros(0, dtype=int)
    won = [bool(x[5]) for x in eps]
    top = int(antes.max()) if len(antes) else 0
    by_cfg: dict[str, list] = {}
    for x in eps:
        by_cfg.setdefault(f"{x[2]}/{x[3]}", []).append(bool(x[5]))
    rate = _rate(won)
    return {
        "summary": {
            "episodes": len(eps),
            **rate,
            "ante_mean": float(antes.mean()) if len(eps) else 0.0,
            "ante_max": top,
            "blinds_mean": float(np.mean([x[6] for x in eps])) if eps else 0.0,
            "truncated": int(sum(bool(x[9]) for x in eps)),
            # Cut off before an outcome (counted as not won above).
            "unresolved": int(sum(bool(x[9]) and not x[5] for x in eps)),
            "invalid_actions": invalid,
            # Episodes for a +-1% interval at the observed rate.
            "episodes_for_1pct": episodes_needed(min(max(rate["win_rate"], 0.01), 0.99), 0.01),
        },
        "survival_past_ante": {str(a): float((antes > a).mean()) for a in range(1, max(top, 9) + 1)} if len(eps) else {},
        "by_deck_stake": {k: _rate(v) for k, v in sorted(by_cfg.items())},
    }


@torch.no_grad()
def evaluate(
    model: Net,
    cfg: Config,
    episodes: int = 1000,
    device="cpu",
    deterministic: bool = True,
    seed: int = EVAL_SEED,
    backs: list[str] | None = None,
    stakes: list[int] | None = None,
    max_ante: int | None = None,
    endless: bool | None = None,
    batch: int = BATCH,
) -> dict:
    """Plays one episode per game with the raw policy (sampling, if not
    ``deterministic``, is seeded with ``seed``). ``endless`` defaults to the
    training config."""
    model.eval()
    eps, invalid = [], 0
    rng = contextlib.nullcontext() if deterministic else seeded(seed, device)
    with rng:
        for env in _envs(cfg, episodes, seed, backs, stakes, max_ante, endless, batch):
            first: dict[int, tuple] = {}
            steps = 0
            while len(first) < env.num_envs and steps <= cfg.env.step_limit + 10:
                out = model.act(obs_from_numpy(env, device), deterministic=deterministic)
                env.step(
                    out["op"].to(torch.int32).cpu().numpy(),
                    out["cards"].cpu().numpy().astype(np.uint32),
                    out["dest"].to(torch.int32).cpu().numpy(),
                )
                steps += 1
                for ep in env.pop_episodes():
                    first.setdefault(ep[0], ep)
            eps += first.values()
            invalid += env.invalid_actions()
    return summarize(eps, invalid)


@torch.no_grad()
def evaluate_search(
    model: Net,
    cfg: Config,
    episodes: int = 200,
    device="cpu",
    seed: int = EVAL_SEED,
    backs: list[str] | None = None,
    stakes: list[int] | None = None,
    max_ante: int | None = None,
    endless: bool | None = None,
    batch: int = BATCH,
    **search_kw,
) -> dict:
    """The games of :func:`evaluate`, played with determinized lookahead
    (sequential, much slower)."""
    from .search import choose

    e = cfg.env
    cap = max_ante or e.max_ante
    eps = []
    for env in _envs(cfg, episodes, seed, backs, stakes, max_ante, endless, batch):
        for i in range(env.num_envs):
            g = env.game(i)
            steps = 0
            while not g.is_terminal() and steps < e.step_limit and not (g.endless and g.ante > cap):
                g.step(choose(model, g, cfg=cfg, seed=steps, device=device, **search_kw))
                steps += 1
            trunc = not g.is_terminal()
            eps.append((len(eps), g.seed, g.back, g.stake, g.ante, g.run_won, g.blinds_cleared, steps, 0.0, trunc, False))
    return summarize(eps)


@torch.no_grad()
def evaluate_round_search(
    model: Net,
    cfg: Config,
    episodes: int = 512,
    device="cpu",
    seed: int = EVAL_SEED,
    backs: list[str] | None = None,
    stakes: list[int] | None = None,
    max_ante: int | None = None,
    endless: bool | None = None,
    batch: int = BATCH,
    **search_kw,
) -> dict:
    """The games of :func:`evaluate`, played with the risk-gated round search
    (``round_search``: batched across games, ~100x the policy forwards;
    ``endgame=True`` adds the ante-8 shop / pack search)."""
    from .round_search import RoundSearch

    rs = RoundSearch(model, cfg, device, **search_kw)
    cap = max_ante or cfg.env.max_ante
    eps, st = [], Counter()
    keys = ("gated", "risky", "overrides", "illegal") + (("eg_gated", "eg_risky", "eg_overrides") if rs.endgame else ())
    for k, env in enumerate(_envs(cfg, episodes, seed, backs, stakes, max_ante, endless, batch)):
        games = [env.game(i) for i in range(env.num_envs)]
        for r in rs.run([rs.play(g, k * batch + i, max_ante=cap) for i, g in enumerate(games)]):
            g = games[r["slot"] - k * batch]
            eps.append((r["slot"], g.seed, g.back, g.stake, g.ante, g.run_won, g.blinds_cleared, r["steps"], 0.0, r["truncated"], False))
            st.update({x: r[x] for x in keys})
    res = summarize(sorted(eps))
    res["search"] = {**{f"{k}_per_game": v / max(1, len(eps)) for k, v in st.items()}, "rows_per_game": sum(rs.rows.values()) / max(1, len(eps))}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=BATCH, help="parallel games")
    ap.add_argument("--backs", default=None)
    ap.add_argument("--stakes", default=None)
    ap.add_argument("--max-ante", type=int, default=None)
    ap.add_argument("--endless", action=argparse.BooleanOptionalAction, default=None, help="default: as trained")
    ap.add_argument("--seed", type=int, default=EVAL_SEED)
    ap.add_argument("--sample", action="store_true", help="sample instead of argmax")
    ap.add_argument("--search", action="store_true", help="determinized lookahead (slow)")
    ap.add_argument("--round-search", action="store_true", help="risk-gated within-round search (round_search; use a GPU)")
    ap.add_argument("--endgame-search", action="store_true", help="with --round-search: also search ante-8 shop and pack decisions")
    ap.add_argument("--endgame-ante", type=int, default=8, help="with --endgame-search: the first ante whose shops and packs are searched (7: about +2 pts, 3x the cost)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.episodes <= 0 or a.batch <= 0:
        ap.error("--episodes and --batch must be positive")
    if a.search and a.round_search:
        ap.error("--search and --round-search are exclusive")
    if a.endgame_search and not a.round_search:
        ap.error("--endgame-search needs --round-search")
    if not 1 <= a.endgame_ante <= 8:
        ap.error("--endgame-ante must be 1..8")
    if a.endgame_ante != 8 and not a.endgame_search:
        ap.error("--endgame-ante needs --endgame-search")
    model, cfg, _ = load_model(a.checkpoint, a.device)
    kw = dict(
        episodes=a.episodes,
        device=a.device,
        seed=a.seed,
        backs=a.backs.split(",") if a.backs else None,
        stakes=[int(s) for s in a.stakes.split(",")] if a.stakes else None,
        max_ante=a.max_ante,
        endless=a.endless,
        batch=a.batch,
    )
    if a.round_search:
        res = evaluate_round_search(model, cfg, endgame=a.endgame_search, eg_min_ante=a.endgame_ante, **kw)
    else:
        res = evaluate_search(model, cfg, **kw) if a.search else evaluate(model, cfg, deterministic=not a.sample, **kw)
    print(json.dumps(res, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
