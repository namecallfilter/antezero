"""``GameEnv``: the :class:`~antezero_bridge.env.Env` interface over the real
game (BalatroBot JSON-RPC server + the antezero companion mod).

Routes
------
``route="harness"`` (default)
    Every action goes through the companion's ``az_act``, which performs it
    exactly like the oracle harness's ``H.act`` (same game entry points, same
    select -> settle -> rebase -> press -> settle timing protocol). This is
    the route for sim parity checks, and it avoids every known BalatroBot
    endpoint hang.
``route="balatrobot"``
    Actions BalatroBot can express are sent to its native endpoints
    (``select``, ``play``, ``buy``, ``rearrange``, ...); the rest, and cases
    with known BalatroBot hangs/rejections, fall back to ``az_act``. After
    every native call the bridge waits for the harness settle condition via
    ``az_state``. The companion rebases timers before every native mutating
    call too, but native calls highlight and press in the same frame
    (without the harness's intermediate settle), so timing can differ from
    the oracle.

Injections use ``az_inject`` (exact ``H.inject``) unless
``inject_route="balatrobot"`` (``add``/``set``; vouchers always companion).
"""

from __future__ import annotations

from typing import Any

from . import canon
from .actions import Action, parse_action
from .client import BridgeClient, RPCError, StateDump
from .env import Canon, Obs, back_key, stake_num
from .injects import Inject, parse_inject
from .launch import PARITY, GameProcess, LaunchConfig, launch

PACK_STATES = {"TAROT_PACK", "PLANET_PACK", "SPECTRAL_PACK", "STANDARD_PACK", "BUFFOON_PACK", "SMODS_BOOSTER_OPENED"}
# Skip tags that open a booster immediately: BalatroBot's `skip` then waits
# for BLIND_SELECT while holding the only connection (deadlock), and its
# `pack` endpoint waits for SHOP (coder/balatrobot#233).
PACK_TAGS = {"tag_charm", "tag_meteor", "tag_ethereal", "tag_standard", "tag_buffoon"}


# BalatroBot / companion error names of requests refused before anything
# was applied (the companion validates first; see its act.lua).
REJECTIONS = ("BAD_REQUEST", "NOT_ALLOWED", "INVALID_STATE")


class ActionRejected(RuntimeError):
    """The game (or BalatroBot) refused the action; nothing was applied.
    Other errors (``RPCError``, timeouts) may leave it partly applied: the
    environment then re-reads the state before using it again."""


class GameEnv:
    def __init__(
        self,
        client: BridgeClient | None = None,
        *,
        route: str = "harness",
        inject_route: str = "harness",
        settle_timeout: float = 120.0,
        process: GameProcess | None = None,
    ):
        self._check_routes(route, inject_route)
        self.client = client or (process.client if process else BridgeClient())
        self.process = process
        self.route = route
        self.inject_route = inject_route
        self.settle_timeout = settle_timeout
        self._dump: StateDump | None = None
        self._pack_from_shop = False
        self.last_route: str | None = None

    @staticmethod
    def _check_routes(route: str, inject_route: str) -> None:
        for r in (route, inject_route):
            if r not in ("harness", "balatrobot"):
                raise ValueError(f"unknown route {r!r}")

    @classmethod
    def launch(cls, cfg: LaunchConfig = PARITY, **kw: Any) -> GameEnv:
        """Start the game (parity settings by default) and own the process."""
        cls._check_routes(kw.get("route", "harness"), kw.get("inject_route", "harness"))
        proc = launch(cfg)
        try:
            return cls(process=proc, **kw)
        except BaseException:
            proc.stop()
            raise

    # -- Env interface ----------------------------------------------------
    def reset(self, seed: str, deck: str = "b_red", stake: int | str = 1) -> Obs:
        self._set(self.client.az_start(seed, back_key(deck), stake_num(stake), settle_timeout=self.settle_timeout))
        self._pack_from_shop = False
        self.last_route = "harness"
        return self.state()

    def step(self, action: Action | str) -> Obs:
        a = parse_action(action)
        prev = self._state_name()
        how = self.route_for(a)
        self.last_route = how
        try:
            if how == "balatrobot":
                d = self.state()
                method, params = a.balatrobot(len(d.get("jokers") or []), len(d.get("hand") or []))  # type: ignore[misc]
                self.client.call(method, params, timeout=self.settle_timeout + 30)
                dump = self.client.az_state(settle=True, settle_timeout=self.settle_timeout)
            else:
                dump = self.client.az_act(a.harness(), settle_timeout=self.settle_timeout)
        except RPCError as e:
            if e.name in REJECTIONS:
                raise ActionRejected(f"{a.code()} [{how}]: {e.message}") from e
            self._dump = None
            raise
        except Exception:
            self._dump = None
            raise
        self._set(dump)
        now = self._state_name()
        if now in PACK_STATES and prev not in PACK_STATES:
            self._pack_from_shop = prev == "SHOP"
        elif now not in PACK_STATES:
            self._pack_from_shop = False
        return self.state()

    def inject(self, inj: Inject | str) -> Obs:
        i = parse_inject(inj)
        bb = None
        # BalatroBot's `add` only works in these states (and `set` too).
        if self.inject_route == "balatrobot" and self._state_name() in ("SELECTING_HAND", "SHOP", "ROUND_EVAL"):
            bb = i.balatrobot(dollars_now=self.state().get("dollars"))
        self.last_route = "balatrobot" if bb is not None else "harness"
        try:
            if bb is not None:
                self.client.call(*bb, timeout=self.settle_timeout + 30)
                dump = self.client.az_state(settle=True, settle_timeout=self.settle_timeout)
            else:
                dump = self.client.az_inject(i.harness(), settle_timeout=self.settle_timeout)
        except RPCError as e:
            if e.name in REJECTIONS:
                raise ActionRejected(f"{i.code} [{self.last_route}]: {e.message}") from e
            self._dump = None
            raise
        except Exception:
            self._dump = None
            raise
        self._set(dump)
        return self.state()

    def legal_actions(self) -> list[Action]:
        codes, _ = self.client.az_legal(settle_timeout=self.settle_timeout)
        out = []
        for c in codes:
            a = Action.parse(c)
            if a is not None:
                out.append(a)
        return out

    def state(self) -> Obs:
        if self._dump is None:
            self.refresh()
        assert self._dump is not None
        return self._dump.data

    def state_json(self) -> str:
        """The verbatim in-game JSON (feed this to Rust ``canon::from_lua``)."""
        if self._dump is None:
            self.refresh()
        assert self._dump is not None
        return self._dump.raw

    def canonical(self) -> Canon:
        return canon.from_lua(self.state_json())

    @property
    def last_frames(self) -> int:
        """Game frames the last settle took (0 if unknown)."""
        return 0 if self._dump is None else self._dump.frames

    def is_terminal(self) -> bool:
        s = self.state()
        return s.get("state") == "GAME_OVER" or (bool(s.get("won")) and s.get("state") == "ROUND_EVAL")

    def refresh(self, settle: bool = True) -> Obs:
        self._set(self.client.az_state(settle=settle, settle_timeout=self.settle_timeout))
        return self.state()

    def close(self) -> None:
        if self.process is not None:
            self.process.stop()
            self.process = None
        else:
            self.client.close()

    def __enter__(self) -> GameEnv:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- routing ------------------------------------------------------------
    def route_for(self, a: Action) -> str:
        """Which route `step` would use for `a` in the current state."""
        if self.route == "harness" or a.op in ("reroll_boss", "sort", "buy_use"):
            return "harness"  # (BalatroBot has no such endpoint)
        s = self.state()
        st = s.get("state")
        op, args = a.op, a.args
        jokers, cons = s.get("jokers") or [], s.get("consumables") or []
        if op in ("use", "sell_joker", "sell_consumable") and st not in ("SELECTING_HAND", "SHOP"):
            return "harness"  # coder/balatrobot#202-#206
        if op == "sell_joker" and args[0] < len(jokers) and jokers[args[0]].get("key") == "j_invisible":
            return "harness"  # #195: sell hangs (the sold card spawns a copy)
        if op == "skip":
            tag = (s.get("blind_tags") or {}).get(s.get("blind_on_deck") or "", "") if isinstance(s.get("blind_tags"), dict) else ""
            if tag in PACK_TAGS:
                return "harness"
        if op == "move_joker" and st not in ("SELECTING_HAND", "SHOP") and st not in PACK_STATES:
            return "harness"  # #207
        if op == "move_hand" and st != "SELECTING_HAND" and st not in PACK_STATES:
            return "harness"
        if op == "buy":
            shop = s.get("shop_jokers") or []
            if args[0] < len(shop):
                c = shop[args[0]]
                if c.get("edition") == "negative" and (len(jokers) >= (s.get("joker_limit") or 0) or len(cons) >= (s.get("consumable_limit") or 0)):
                    return "harness"  # #208: BalatroBot ignores the negative slot
        if op == "open":
            packs = s.get("shop_booster") or []
            key = packs[args[0]].get("key", "") if args[0] < len(packs) else ""
            if "celestial" in key:
                return "harness"  # #199: hangs if Black Hole is the first card
            if ("arcana" in key or "spectral" in key) and len(s.get("deck") or []) < (s.get("hand_limit") or 0):
                return "harness"  # #198: hangs with thin decks
        if op in ("pick", "skip_pack"):
            if not self._pack_from_shop:
                return "harness"  # #233: tag packs; `pack` waits for SHOP
            pack = s.get("pack_cards") or []
            if any(c.get("key") == "c_black_hole" for c in pack):
                return "harness"  # #199
            if op == "pick" and args[0] < len(pack):
                c = pack[args[0]]
                if c.get("ability", {}).get("set") == "Joker" and c.get("edition") == "negative" and len(jokers) >= (s.get("joker_limit") or 0):
                    return "harness"  # #208
        return "balatrobot"

    # -- internals ------------------------------------------------------------
    def _set(self, d: StateDump) -> None:
        self._dump = d

    def _state_name(self) -> str | None:
        return None if self._dump is None else self._dump.data.get("state")
