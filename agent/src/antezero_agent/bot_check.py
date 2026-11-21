"""The in-game bot's loop (``bot.play_run``) against the headless game: the
Lua oracle of ``sim/oracle`` (the game's own scripts), read through the
mod's ``state.lua`` as the bot reads the real game. Every stop the game did
not earn is a false stop of the bot's checks (the import check and the
expected next state); the steps the expected-state check excused are
counted by why.

    python -m antezero_agent.bot_check CKPT [--games-per 1] [--search]
        [--max-ante 12] [--shard I/N] [--out FILE]

plays a game per deck and stake (``--games-per`` each), with the
checkpoint's greedy policy or with search, on to Endless Mode after a win
until ``--max-ante``. With ``--drain FRAMES`` the game also runs that many
frames after every settled reply and counts the fields that changed: what a
reply the mod calls settled leaves to happen. Needs the game's ``lua51.dll``
(``BALATRO_LUA51``) and scripts (``BALATRO_SRC``), as the oracle tests do.
"""

from __future__ import annotations

import argparse
import ctypes
import faulthandler
import json
import os
import time
from collections import Counter
from pathlib import Path

import antezero_sim as az
import torch

from . import bot
from .round_search import RoundSearch
from .runtime import load_model

ROOT = Path(__file__).resolve().parents[3]
LUA51 = Path(os.environ.get("BALATRO_LUA51", r"C:\Program Files (x86)\Steam\steamapps\common\Balatro\lua51.dll"))
SRC = Path(os.environ.get("BALATRO_SRC", ROOT / "reference" / "balatro"))


class Lua:
    """A LuaJIT state of the game's ``lua51.dll``."""

    def __init__(self) -> None:
        lib = ctypes.CDLL(str(LUA51))
        lib.luaL_newstate.restype = ctypes.c_void_p
        lib.luaL_openlibs.argtypes = [ctypes.c_void_p]
        lib.luaL_loadbuffer.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p]
        lib.lua_pcall.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.lua_tolstring.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_size_t)]
        lib.lua_tolstring.restype = ctypes.c_void_p
        lib.lua_settop.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib, self.state = lib, lib.luaL_newstate()
        lib.luaL_openlibs(self.state)

    def run(self, code: str) -> str | None:
        """Runs ``code`` (``return x`` gives ``x`` as text); a Lua error raises ``RuntimeError``."""
        lib, L = self.lib, self.state
        b = code.encode()
        # LuaJIT raises Lua errors as Windows exceptions, which faulthandler
        # reports as fatal even when a pcall catches them.
        reporting = faulthandler.is_enabled()
        faulthandler.disable()
        try:
            rc = lib.luaL_loadbuffer(L, b, len(b), b"bot_check") or lib.lua_pcall(L, 0, 1, 0)
        finally:
            if reporting:
                faulthandler.enable()
        n = ctypes.c_size_t()
        p = lib.lua_tolstring(L, -1, ctypes.byref(n))
        out = ctypes.string_at(p, n.value).decode() if p else None
        lib.lua_settop(L, 0)
        if rc:
            raise RuntimeError(out or "?")
        return out


def _lua_text(p: Path) -> str:
    return json.dumps(p.as_posix())


def headless_game() -> Lua:
    """The oracle harness with the mod's state dump writer."""
    lua = Lua()
    lua.run(
        f"HARNESS_SRC = {_lua_text(SRC)}\nHARNESS_DIR = {_lua_text(ROOT / 'sim' / 'oracle' / 'lua')}\n"
        "package.path = HARNESS_DIR..'/?.lua;'..HARNESS_SRC..'/?.lua;'..package.path\nrequire('harness')"
    )
    mod = ROOT / "mod" / "src"
    lua.run(
        f"""SMODS = {{}}
        AZB = {{json = dofile({_lua_text(mod / 'json.lua')})}}
        function AZB.state_name(s) for k, v in pairs(G.STATES) do if v == s then return k end end end
        function AZB.vanilla_state_name() return AZB.state_name(G.STATE) end
        function AZB.is_settled() return true end
        function AZB.is_bridge() return false end
        function AZB.smods_only_rng() return false end
        dofile({_lua_text(mod / 'state.lua')})"""
    )
    return lua


class OracleClient:
    """What ``GameEnv`` and ``play_run`` ask of the mod, answered by the
    headless game: one bot run, active until the run passes ``max_ante``."""

    def __init__(self, lua: Lua, run: dict, max_ante: int, drain: int = 0):
        self.lua, self.run, self.max_ante, self.ante, self.drain = lua, run, max_ante, 1, drain
        self.actions = 0
        self.drained = Counter()  # fields a settled reply left to change, by path

    def _dump(self):
        from antezero_bridge.client import StateDump

        raw = self.lua.run("return AZB.state()")
        self.ante = json.loads(raw)["ante"]
        if self.drain:
            from antezero_bridge import canon

            later = self.lua.run(f"H.step({self.drain}) return AZB.state()")
            for path, _, _ in canon.diff(canon.from_lua(raw), canon.from_lua(later)):
                self.drained[path] += 1
        return StateDump(raw=raw, settled=True, reason="settled", frames=0)

    def start(self) -> None:
        r = self.run
        self.lua.run(f"H.start({json.dumps(r['seed'])}, {json.dumps(r['back'])}, {int(r['stake'])})")

    def az_state(self, settle=True, settle_timeout=None):
        return self._dump()

    def az_act(self, action, settle_timeout=None, **bot):
        from antezero_bridge.actions import lua_table
        from antezero_bridge.client import RPCError

        try:
            self.lua.run(f"H.act({lua_table(action)})")
        except RuntimeError as e:
            raise RPCError("BAD_REQUEST", str(e).splitlines()[0], -1) from e
        self.actions += 1
        return self._dump()

    def az_bot(self, **params):
        active = self.ante <= self.max_ante
        return {"active": active, "run": self.run, "paused": False, "interrupted": False, "search": True}


class Excused:
    """Counts the steps ``bot.surprises`` excused, by why."""

    def __init__(self) -> None:
        self.counts = Counter()
        self.surprises = bot.surprises

    def __call__(self, before, expected, game):
        from antezero_bridge import canon

        out = self.surprises(before, expected, game)
        diffs = canon.diff(expected, game)
        if diffs and not out:
            kinds = {next((p for p in bot.LOOSE if d[0].startswith(p)), "new cards or The Pillar") for d in diffs}
            for k in kinds:
                self.counts[k] += 1
        return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--games-per", type=int, default=1, help="games per deck and stake")
    ap.add_argument("--stakes", default="1-8")
    ap.add_argument("--search", action="store_true", help="choose with RoundSearch (on the CPU: slow)")
    ap.add_argument("--max-ante", type=int, default=12, help="play on in Endless Mode until this ante")
    ap.add_argument("--shard", default="0/1", help="I/N: every Nth game from the Ith")
    ap.add_argument("--seed", default="CHK")
    ap.add_argument("--drain", type=int, default=0, metavar="FRAMES")
    ap.add_argument("--out", help="a JSON line per game")
    a = ap.parse_args(argv)
    from antezero_bridge.game_env import GameEnv

    torch.set_num_threads(1)
    model, cfg, _ = load_model(a.checkpoint, "cpu")
    rs = RoundSearch(model, cfg, "cpu", endgame=True, eg_min_ante=7, threads=2) if a.search else None
    lo, hi = (int(x) for x in a.stakes.split("-"))
    specs = [(back, stake, k) for back in az.backs() for stake in range(lo, hi + 1) for k in range(a.games_per)]
    i, n = (int(x) for x in a.shard.split("/"))
    excused = Excused()
    bot.surprises = excused
    lua = headless_game()
    ends, drained = Counter(), Counter()
    out = open(a.out, "a", encoding="utf-8") if a.out else None
    for j, (back, stake, k) in enumerate(specs):
        if j % n != i:
            continue
        run = {"id": j + 1, "seed": f"{a.seed}{j:04d}{k}", "back": back, "stake": stake}
        client = OracleClient(lua, run, a.max_ante, a.drain)
        client.start()
        t = time.time()
        try:
            end = bot.play_run(client, GameEnv(client), run, model, cfg, "cpu", rs, None)
        except Exception as e:  # noqa: BLE001 - reported per game
            end = f"error: {type(e).__name__}: {e}"
        end = end or f"reached ante {a.max_ante + 1}"
        ends[end.split(" at ante")[0] if end.startswith("lost in Endless") else end] += 1
        drained += client.drained
        rec = {**run, "end": end, "actions": client.actions, "ante": client.ante, "seconds": round(time.time() - t, 1), "drained": dict(client.drained)}
        print(json.dumps(rec), flush=True)
        if out:
            out.write(json.dumps(rec) + "\n")
            out.flush()
    print("ends:", dict(ends), "excused steps:", dict(excused.counts), "changed after settling:", dict(drained), flush=True)


if __name__ == "__main__":
    main()
