"""The agent behind the in-game bot button.

    uv run python -m antezero_agent.bot --install CKPT [--device auto]

installs the AnteZero mod and tells it how to start this agent. Then start
Balatro as usual: on the New Run screen, turn on the bot toggle on the Play
button and press Play. The mod starts the agent (``python -m
antezero_agent.bot CKPT --port PORT``), which plays every such run until the
game closes. Search and the bot's speed are settings of the mod (Mods >
AnteZero > Config in the game).

As in ``play_real``, the simulator runs in lockstep with the game (same seed,
deck, stake and actions): the policy reads the simulator's observation, and
the bot stops, leaving the run to the player, if the two ever disagree.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import traceback
from pathlib import Path

import antezero_sim as az
import httpx
import torch

from .critic import Critic
from .play_real import greedy
from .round_search import RoundSearch
from .runtime import load_model, pick_device, read_checkpoint


def install(a) -> None:
    from antezero_bridge.launch import BB_DISABLED, SAVE_DIR, install_companion_mod

    ckpt = Path(a.checkpoint).resolve()
    read_checkpoint(ckpt)  # it must load in this simulator
    cmd = [sys.executable, "-m", "antezero_agent.bot", str(ckpt), "--device", a.device]
    (SAVE_DIR / "antezero.json").write_text(json.dumps({"command": subprocess.list2cmdline(cmd)}), encoding="utf-8")
    print("installed", install_companion_mod())
    if BB_DISABLED.parent.exists():  # it mutes the game and overrides its settings whenever it loads
        BB_DISABLED.touch()
        print("BalatroBot stays disabled outside the developer tools")


@torch.no_grad()
def play_run(client, env, run: dict, model, cfg, dev, rs: RoundSearch | None, critic: Critic | None) -> str | None:
    """Plays the bot run ``run`` of the mod; returns how it ended for the
    bot (the text the game shows), or ``None`` if the player left it."""
    from antezero_bridge import canon
    from antezero_bridge.client import RPCError
    from antezero_bridge.game_env import ActionRejected

    sim = az.Game(run["seed"], run["back"], int(run["stake"]), endless=cfg.env.endless)
    sim.allow_skip = cfg.env.allow_skip  # the policy must see the action set it was trained on
    sim.steamodded = True  # the game runs Steamodded, whose xMult rounds differently
    steps = 0
    try:
        env.refresh()
        while True:
            if sim.is_terminal() and sim.run_won:
                return "won the run"  # the game shows its win screen, which the harness would dismiss
            d = canon.diff(sim.canonical(), env.canonical())
            if d:
                print(f"[{run['seed']}] divergence after step {steps}: {d[:10]}", flush=True)
                return "stopped: the game and the simulator disagree"
            if sim.is_terminal():
                return "lost the run"
            while True:  # a menu the player opened pauses the bot
                s = client.az_bot(run=run["id"], status="thinking")
                if not s["active"] or s["run"]["id"] != run["id"]:
                    return None
                if not s["paused"]:
                    break
                time.sleep(0.2)
            code, out, o = greedy(model, cfg, sim, dev)
            note = f"{math.exp(float(out['logp'][0])):.0%} sure"
            if rs is not None and s["search"]:
                pick = rs.choose(sim, f"{run['id']}_{steps}")
                if pick != code:
                    code, note = pick, "found by search"
            if critic is not None:
                note += f", win {float(critic.win_prob(o)[0]):.0%}"
            if code is None:
                return "stopped: the policy has no legal action"
            env.step(code, run=run["id"], note=note)
            sim.step(code)
            steps += 1
    except (ActionRejected, RPCError) as e:  # the run ended under the action
        print(f"[{run['seed']}] {e}", flush=True)
        return None


def serve(a) -> None:
    from antezero_bridge.client import BridgeClient
    from antezero_bridge.game_env import GameEnv

    client = BridgeClient(port=a.port)
    client.patience = 24 * 3600.0
    env = GameEnv(client)
    try:
        client.az_bot(status="loading the model...")
    except httpx.HTTPError:
        pass
    dev = pick_device(a.device)
    model, cfg, _ = load_model(a.checkpoint, dev)
    # The mod's search setting, where there is a GPU for it (~100x the forward passes of greedy play).
    rs = RoundSearch(model, cfg, dev, endgame=True, eg_min_ante=7) if dev.type == "cuda" else None
    critic = None
    ck = read_checkpoint(a.checkpoint)
    if "critic" in ck:
        critic = Critic(cfg.model)
        critic.load_state_dict(ck["critic"])
        critic.to(dev).eval()
    del ck
    print(f"ready on {dev}: {a.checkpoint}, search {'available' if rs else 'unavailable (no CUDA GPU)'}", flush=True)
    done, misses = None, 0
    while misses < 5:  # else the game closed
        try:
            s = client.az_bot()
            misses = 0
            run = s.get("run")
            if s["active"] and run["id"] != done:
                done = run["id"]
                print(f"playing {run}", flush=True)
                try:
                    end = play_run(client, env, run, model, cfg, dev, rs, critic)
                except httpx.HTTPError:
                    raise
                except Exception as e:
                    traceback.print_exc()
                    end = f"stopped: {type(e).__name__}"
                print(f"[{run['seed']}] {end}", flush=True)
                if end:
                    client.az_bot(run=run["id"], stop=end)
        except httpx.HTTPError:
            misses += 1
        time.sleep(0.3)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--install", action="store_true", help="install the mod and remember this checkpoint")
    ap.add_argument("--device", default="auto", help="cpu, cuda, ... or auto")
    ap.add_argument("--port", type=int, default=12346)
    a = ap.parse_args(argv)
    install(a) if a.install else serve(a)


if __name__ == "__main__":
    main()
