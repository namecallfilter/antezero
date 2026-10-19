"""Play a trained agent in the real game (Steam Balatro via the bridge).

    uv run python -m antezero_agent.play_real CKPT --seeds 20 --deck b_red --stake 1 [--search]

The simulator runs in lockstep with the real game (same seed, same
actions): the policy reads the simulator's observation, the chosen action
is sent to the real game, and both canonical states are compared after the
reset and after every step (with the oracle's rules: only the outcome once
the game is over), so any divergence is reported immediately instead of
silently corrupting the agent's view. Requires the modding stack from
``agent/bridge`` (lovely + Steamodded + BalatroBot + companion mod).
"""

from __future__ import annotations

import argparse
import json
import random

import antezero_sim as az
import torch

from .model import obs_from_numpy
from .runtime import load_model
from .search import choose


def random_seed(rng: random.Random) -> str:
    return "".join(rng.choice("123456789ABCDEFGHIJKLMNPQRSTUVWXYZ") for _ in range(8))


@torch.no_grad()
def play(model, cfg, env, seed: str, deck: str, stake: int, max_ante: int, use_search: bool, max_steps: int | None = None, log=print) -> dict:
    """One game; ``outcome`` is ``won``, ``lost``, ``diverged`` (the engines
    disagree) or ``stopped`` (``max_ante`` or ``max_steps`` reached)."""
    from antezero_bridge import canon

    K = cfg.env.play_candidates
    max_steps = cfg.env.step_limit if max_steps is None else max_steps
    env.reset(seed, deck, stake)
    sim = az.Game(seed, deck, stake, endless=cfg.env.endless)
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
        if use_search:
            code = choose(model, sim, cfg=cfg, seed=steps)
        else:
            out = model.act(obs_from_numpy(sim.encode(K, cfg.env.deck_tokens)), deterministic=True)
            code = sim.factored_to_code(int(out["op"][0]), int(out["cards"][0]), int(out["dest"][0]), K)
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
    ap.add_argument("--attach", action="store_true", help="use an already running game")
    a = ap.parse_args(argv)
    from antezero_bridge.game_env import GameEnv

    model, cfg, _ = load_model(a.checkpoint)
    rng = random.Random(1234)
    seeds = a.seed_list.split(",") if a.seed_list else [random_seed(rng) for _ in range(a.seeds)]
    env = GameEnv() if a.attach else GameEnv.launch()
    results = []
    try:
        for s in seeds:
            r = play(model, cfg, env, s, a.deck, a.stake, a.max_ante, a.search, a.max_steps)
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
