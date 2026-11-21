"""The agent behind the in-game bot button.

    uv run python -m antezero_agent.bot --install CKPT [--device auto]

installs the AnteZero mod and tells it how to start this agent. Then start
Balatro as usual and, in a run, turn on "Bot plays this run" in the options
menu (Esc). The mod starts the agent (``python -m antezero_agent.bot CKPT
--port PORT``), which plays the run from there, and any other the toggle is
turned on in, until the game closes. Search and the bot's speed are
settings of the mod (Mods > AnteZero > Config in the game).

Before every decision the simulator's copy of the run is rebuilt from the
game's state (the mod's state dump), and the policy reads its observation:
whatever the game holds (the profile's locked cards, unlocks during the run,
moves the player made) is what the bot plays. The copy is checked first,
against the dump as the bridge reads it and against what the simulator
expected the bot's last action to do; a copy that fails either stops the bot.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import antezero_sim as az
import httpx
import torch

from .critic import Critic
from .play_real import greedy
from .round_search import FREE, RoundSearch
from .runtime import load_model, pick_device, read_checkpoint


def command_line(args: list[str]) -> str:
    """``args`` as the mod runs them, through cmd.exe: every one quoted, so
    that ``&``, ``(`` or ``^`` in a path stay in it."""
    return " ".join(f'"{a}"' for a in args)


def install(a) -> None:
    from antezero_bridge.launch import BB_DISABLED, SAVE_DIR, install_companion_mod

    ckpt = Path(a.checkpoint).resolve()
    read_checkpoint(ckpt)  # it must load in this simulator
    cmd = [sys.executable, "-m", "antezero_agent.bot", str(ckpt), "--device", a.device]
    (SAVE_DIR / "antezero.json").write_text(json.dumps({"command": command_line(cmd)}), encoding="utf-8")
    print("installed", install_companion_mod())
    if BB_DISABLED.parent.exists():  # it mutes the game and overrides its settings whenever it loads
        BB_DISABLED.touch()
        print("BalatroBot stays disabled outside the developer tools")


# What the game draws from its pools may differ from the simulator's draw with
# nothing wrong: a profile's locked cards are not offered (the simulator offers
# every card), and Steamodded rolls a few things its own way (the Soul's rarity
# roll, a used_jokers mark for a redeemed voucher). So the expected state is
# compared but for new cards, the next voucher and skip tags, used_jokers and
# the RNG streams, and not at all after the game made other cards than the
# simulator (a joker's own effects, such as hand size, can show anywhere).
# Steamodded also keeps The Pillar's mark through an enhancement, and the
# card stays debuffed.
LOOSE = ("rng.", "used_jokers", "voucher", "blind_tags.")


def surprises(before: dict, expected: dict, game: dict) -> list[tuple[str, str, str]]:
    """Where ``game`` differs from ``expected``, the simulator's prediction
    of the action taken in ``before`` (canonical states)."""
    from antezero_bridge import canon

    def card(path: str) -> str:
        return ".".join(path.split(".")[:2])

    last = max((float(v) for k, v in before.items() if k.endswith(".id")), default=0.0)
    new = {card(k) for c in (expected, game) for k, v in c.items() if k.endswith(".id") and float(v) > last}
    if any(expected.get(f"{c}.key") != game.get(f"{c}.key") for c in new):
        return []
    diffs = canon.diff(expected, game)
    pillar = {f"{card(p)}.{f}" for p, _, _ in diffs if p.endswith(".played_this_ante") for f in ("played_this_ante", "debuff")}
    return [d for d in diffs if not d[0].startswith(LOOSE) and d[0] not in pillar and card(d[0]) not in new]


def _stop(run: dict, why: str, diffs: list) -> str:
    print(f"[{run['seed']}] {why} (simulator, game):", *(f"  {p}: {a} | {b}" for p, a, b in diffs[:20]), sep="\n", flush=True)
    return f"stopped: {why}"


@torch.no_grad()
def play_run(client, env, run: dict, model, cfg, dev, rs: RoundSearch | None, critic: Critic | None) -> str | None:
    """Plays the bot's turn ``run`` at a run of the mod; returns how it
    ended for the bot (the text the game shows), or ``None`` if the player
    took the run back or left it."""
    from antezero_bridge import canon
    from antezero_bridge.client import RPCError
    from antezero_bridge.game_env import ActionRejected

    sim = az.Game(run["seed"], run["back"], int(run["stake"]), endless=cfg.env.endless)
    sim.allow_skip, sim.null_sorts = cfg.env.allow_skip, cfg.env.null_sorts  # the action set the policy was trained on
    sim.steamodded = True  # the game runs Steamodded, whose xMult rounds differently
    steps, free = 0, 0  # free: the bot's rearranges in a row, which the policy is held to as in training
    expected = None  # (the game before the bot's last action, what the simulator expected after it)
    stale = True  # the game is to be read anew: at the start, and after a menu was open
    try:
        while True:
            while True:  # a menu the player opened (the win screen too) pauses the bot
                s = client.az_bot(run=run["id"], status="thinking")
                if not s["active"] or s["run"]["id"] != run["id"]:
                    return None
                stale = stale or s["interrupted"]
                if not s["paused"]:
                    break
                time.sleep(0.2)
            if stale:  # the game as the player left it
                env.refresh()
                expected, stale = None, False
            game = env.canonical()
            sim.load_state(env.state_json())
            sim.rearranges = free
            # The bot waits on the win screen: a won run it reads goes on in Endless Mode.
            sim.endless = sim.endless or sim.run_won
            if wrong := canon.diff(sim.canonical(), game):
                return _stop(run, "the simulator misread the game", wrong)
            if sim.is_terminal():
                return f"lost in Endless Mode at ante {sim.ante}" if sim.run_won else "lost the run"
            if expected and (wrong := surprises(*expected, game)):
                return _stop(run, "the game did not do what the simulator expected", wrong)
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
            nxt = sim.clone()
            nxt.step(code)
            expected = (game, nxt.canonical())
            try:
                env.step(code, run=run["id"], note=note)
            except ActionRejected as e:
                s = client.az_bot(run=run["id"])
                if not s["active"] or s["run"]["id"] != run["id"]:
                    return None
                if s["interrupted"]:  # decided before a menu was opened: decide again
                    stale = True
                    continue
                print(f"[{run['seed']}] {e}", flush=True)
                return "stopped: the game refused the bot's action"
            free = free + 1 if code.startswith(FREE) else 0
            steps += 1
    except RPCError as e:  # the run ended under the action
        print(f"[{run['seed']}] {e}", flush=True)
        s = client.az_bot(run=run["id"])
        return "stopped: the game did not finish the bot's action" if s["active"] and s["run"]["id"] == run["id"] else None


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
        critic = Critic(cfg.model, ck["critic"].get("fx"))
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
                except httpx.HTTPError as e:  # handed back if the game is still there, else the stop fails too
                    end = f"stopped: lost contact with the game ({type(e).__name__})"
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
