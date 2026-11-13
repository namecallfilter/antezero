"""Measure a trained agent in the real game (Steam Balatro via the bridge):
seeded games played without rendering, for the win rate and for any
divergence between the game and the simulator. To watch the agent play, use
the in-game bot button (``bot.py``).

    uv run python -m antezero_agent.play_real CKPT --seeds 20 --deck b_red --stake 1 [--search]
    uv run python -m antezero_agent.play_real CKPT --seeds 20 --round-search --device cuda
    uv run python -m antezero_agent.play_real CKPT --seeds 20 --round-search --endgame-search --device cuda

The simulator runs in lockstep with the real game (same seed, same
actions): the policy reads the simulator's observation, the chosen action
is sent to the real game, and both canonical states are compared after the
reset and after every step (with the oracle's rules: only the outcome once
the game is over), so any divergence is reported immediately instead of
silently corrupting the agent's view. The searches only roll out
``resample_hidden`` copies of the simulator, which knows the real seed.
Requires the modding stack from ``bridge`` (lovely + Steamodded +
BalatroBot + the AnteZero mod).
"""

from __future__ import annotations

import argparse
import json
import random

import antezero_sim as az
import torch

from .model import obs_from_numpy
from .round_search import RoundSearch
from .runtime import load_model, pick_device
from .search import choose


def random_seed(rng: random.Random) -> str:
    return "".join(rng.choice("123456789ABCDEFGHIJKLMNPQRSTUVWXYZ") for _ in range(8))


def greedy(model, cfg, sim: az.Game, dev) -> tuple[str | None, dict, object]:
    """The policy's own choice in ``sim``: its action code, ``Net.act``'s
    output and the observation it was made from."""
    K = cfg.env.play_candidates
    o = obs_from_numpy(sim.encode(K, cfg.env.deck_tokens, cfg.env.build_previews), dev)
    out = model.act(o, deterministic=True)
    return sim.factored_to_code(int(out["op"][0]), int(out["cards"][0]), int(out["dest"][0]), K), out, o


@torch.no_grad()
def play(model, cfg, env, seed: str, deck: str, stake: int, max_ante: int, use_search: bool, max_steps: int | None = None, log=print, round_search: RoundSearch | None = None, game: int = 0) -> dict:
    """One game; ``outcome`` is ``won``, ``lost``, ``diverged`` (the engines
    disagree) or ``stopped`` (``max_ante`` or ``max_steps`` reached).
    ``game`` numbers the game for ``round_search``'s world tags."""
    from antezero_bridge import canon

    dev = next(model.parameters()).device
    max_steps = cfg.env.step_limit if max_steps is None else max_steps
    env.reset(seed, deck, stake)
    sim = az.Game(seed, deck, stake, endless=cfg.env.endless)
    sim.allow_skip = cfg.env.allow_skip  # the policy must see the action set it was trained on
    sim.steamodded = True  # the bridge's game runs Steamodded, whose ×Mult rounds differently
    steps, code = 0, "reset"

    def result(outcome: str) -> dict:
        return {"seed": seed, "steps": steps, "ante": sim.ante, "won": sim.run_won, "outcome": outcome}

    while True:
        d = canon.diff(sim.canonical(), env.canonical())
        if d:
            log(f"[{seed}] divergence after step {steps} ({code}): {d[:10]}")
            return result("diverged")
        if sim.is_terminal():
            return result("won" if sim.run_won else "lost")
        if sim.ante > max_ante or steps >= max_steps:
            return result("stopped")
        if round_search is not None:
            code = round_search.choose(sim, f"{game}_{steps}")
        elif use_search:
            code = choose(model, sim, cfg=cfg, seed=steps, device=dev)
        else:
            code = greedy(model, cfg, sim, dev)[0]
        if code is None:
            raise RuntimeError(f"policy produced an illegal action at step {steps}")
        env.step(code)
        sim.step(code)
        steps += 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed-list", default=None, help="comma-separated game seeds")
    ap.add_argument("--deck", default="b_red")
    ap.add_argument("--stake", type=int, default=1)
    ap.add_argument("--max-ante", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=None, help="default: the training step limit")
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--round-search", action="store_true", help="risk-gated within-round search (use a GPU)")
    ap.add_argument("--endgame-search", action="store_true", help="with --round-search: also search ante-8 shop and pack decisions")
    ap.add_argument("--endgame-ante", type=int, default=8, help="with --endgame-search: the first ante whose shops and packs are searched (7: about +2 pts, longer pauses)")
    ap.add_argument("--device", default="cpu", help="cpu, cuda, ... or auto")
    ap.add_argument("--attach", action="store_true", help="use an already running game")
    a = ap.parse_args(argv)
    if a.search and a.round_search:
        ap.error("--search and --round-search are exclusive")
    if a.endgame_search and not a.round_search:
        ap.error("--endgame-search needs --round-search")
    if not 1 <= a.endgame_ante <= 8:
        ap.error("--endgame-ante must be 1..8")
    if a.endgame_ante != 8 and not a.endgame_search:
        ap.error("--endgame-ante needs --endgame-search")
    from antezero_bridge.game_env import GameEnv

    dev = pick_device(a.device)
    model, cfg, _ = load_model(a.checkpoint, dev)
    rs = RoundSearch(model, cfg, dev, endgame=a.endgame_search, eg_min_ante=a.endgame_ante) if a.round_search else None
    rng = random.Random(1234)
    seeds = a.seed_list.split(",") if a.seed_list else [random_seed(rng) for _ in range(a.seeds)]
    env = GameEnv() if a.attach else GameEnv.launch()
    results = []
    try:
        for gi, s in enumerate(seeds):
            r = play(model, cfg, env, s, a.deck, a.stake, a.max_ante, a.search, a.max_steps, round_search=rs, game=gi)
            print(json.dumps(r), flush=True)
            results.append(r)
    finally:
        env.close()
    count = {k: sum(r["outcome"] == k for r in results) for k in ("won", "lost", "diverged", "stopped")}
    finished = count["won"] + count["lost"]
    # The win rate is over games that finished in agreement with the sim.
    print(json.dumps({"games": len(results), **count, "win_rate": count["won"] / max(1, finished)}))


if __name__ == "__main__":
    main()
