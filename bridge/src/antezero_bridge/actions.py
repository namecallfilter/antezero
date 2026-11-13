"""The sim's action vocabulary (``sim/src/actions.rs``) in Python.

Text codes are exactly those of the Rust ``Action`` ``Display`` / ``FromStr``
(also used in history files), with the same strict grammar. Hand-card
selections are bitmasks over the *current* hand order (bit i =
``G.hand.cards[i+1]``); every other index is a 0-based position in the
current area order.

Two translations are provided:

* :meth:`Action.harness` -> the oracle harness command table (``H.act``),
  identical to ``action_to_lua`` in ``sim/oracle/src/diff.rs`` (1-based).
  The companion mod's ``az_act`` executes it with the harness's timing
  protocol.
* :meth:`Action.balatrobot` -> a native BalatroBot ``(method, params)`` call
  (0-based), or ``None`` when BalatroBot cannot express the action.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# op -> bit widths of its integer operands (u8 indices, u32 card masks), in
# code order. `sort` takes 0 (by rank) or 1 (by suit).
OPERANDS: dict[str, tuple[int, ...]] = {
    "select": (),
    "skip": (),
    "reroll_boss": (),
    "play": (32,),
    "discard": (32,),
    "use": (8, 32),
    "sell_joker": (8,),
    "sell_consumable": (8,),
    "move_joker": (8, 8),
    "move_hand": (8, 8),
    "sort": (1,),
    "buy": (8,),
    "buy_use": (8, 32),
    "redeem": (8,),
    "open": (8,),
    "reroll": (),
    "end_shop": (),
    "cash_out": (),
    "pick": (8, 32),
    "skip_pack": (),
}
# op -> number of operands.
ARITY: dict[str, int] = {op: len(w) for op, w in OPERANDS.items()}

# What Rust's unsigned integer `FromStr` accepts.
_UINT = re.compile(r"\+?[0-9]+")


def mask_indices(m: int) -> list[int]:
    """0-based hand indices of a selection mask, ascending (like the sim)."""
    return [i for i in range(32) if m >> i & 1]


@dataclass(frozen=True, slots=True)
class Action:
    """One sim action. ``op`` is the code keyword, ``args`` its integers."""

    op: str
    args: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        widths = OPERANDS.get(self.op)
        if widths is None:
            raise ValueError(f"unknown action op {self.op!r}")
        if len(self.args) != len(widths):
            raise ValueError(f"{self.op} takes {len(widths)} args, got {self.args!r}")
        for v, w in zip(self.args, widths):
            if not isinstance(v, int) or isinstance(v, bool) or not 0 <= v < 1 << w:
                raise ValueError(f"{self.op}: operand {v!r} is not a {w}-bit unsigned integer")

    # -- constructors -----------------------------------------------------
    @staticmethod
    def select() -> Action:
        return Action("select")

    @staticmethod
    def skip() -> Action:
        return Action("skip")

    @staticmethod
    def reroll_boss() -> Action:
        return Action("reroll_boss")

    @staticmethod
    def play(mask: int) -> Action:
        return Action("play", (mask,))

    @staticmethod
    def discard(mask: int) -> Action:
        return Action("discard", (mask,))

    @staticmethod
    def use(index: int, targets: int = 0) -> Action:
        return Action("use", (index, targets))

    @staticmethod
    def sell_joker(i: int) -> Action:
        return Action("sell_joker", (i,))

    @staticmethod
    def sell_consumable(i: int) -> Action:
        return Action("sell_consumable", (i,))

    @staticmethod
    def move_joker(frm: int, to: int) -> Action:
        return Action("move_joker", (frm, to))

    @staticmethod
    def move_hand(frm: int, to: int) -> Action:
        return Action("move_hand", (frm, to))

    @staticmethod
    def sort(by_suit: bool) -> Action:
        return Action("sort", (1 if by_suit else 0,))

    @staticmethod
    def buy(i: int) -> Action:
        return Action("buy", (i,))

    @staticmethod
    def buy_use(index: int, targets: int = 0) -> Action:
        return Action("buy_use", (index, targets))

    @staticmethod
    def redeem(i: int) -> Action:
        return Action("redeem", (i,))

    @staticmethod
    def open(i: int) -> Action:
        return Action("open", (i,))

    @staticmethod
    def reroll() -> Action:
        return Action("reroll")

    @staticmethod
    def end_shop() -> Action:
        return Action("end_shop")

    @staticmethod
    def cash_out() -> Action:
        return Action("cash_out")

    @staticmethod
    def pick(index: int, targets: int = 0) -> Action:
        return Action("pick", (index, targets))

    @staticmethod
    def skip_pack() -> Action:
        return Action("skip_pack")

    # -- text codes -------------------------------------------------------
    def code(self) -> str:
        """``Action::code`` text form, e.g. ``play 11``."""
        return " ".join([self.op, *map(str, self.args)])

    def __str__(self) -> str:
        return self.code()

    @staticmethod
    def parse(s: str) -> Action | None:
        """The Rust ``Action`` ``FromStr``: ``None`` for anything that is not
        exactly an action code (unknown op, wrong operand count, operand out
        of range)."""
        toks = s.split()
        if not toks or toks[0] not in OPERANDS:
            return None
        op, ops = toks[0], toks[1:]
        widths = OPERANDS[op]
        if len(ops) != len(widths) or not all(_UINT.fullmatch(t) for t in ops):
            return None
        vals = tuple(int(t) for t in ops)
        if any(v >= 1 << w for v, w in zip(vals, widths)):
            return None
        return Action(op, vals)

    # -- translations -----------------------------------------------------
    def harness(self) -> dict[str, Any]:
        """The oracle ``H.act`` table (1-based), as in diff.rs ``action_to_lua``."""
        op, a = self.op, self.args

        def cards(m: int) -> list[int]:
            return [i + 1 for i in mask_indices(m)]

        match op:
            case "select":
                return {"type": "select_blind"}
            case "skip":
                return {"type": "skip_blind"}
            case "reroll_boss":
                return {"type": "reroll_boss"}
            case "play":
                return {"type": "play", "cards": cards(a[0])}
            case "discard":
                return {"type": "discard", "cards": cards(a[0])}
            case "use":
                return {"type": "use", "index": a[0] + 1, "cards": cards(a[1])}
            case "sell_joker":
                return {"type": "sell_joker", "index": a[0] + 1}
            case "sell_consumable":
                return {"type": "sell_consumable", "index": a[0] + 1}
            case "move_joker":
                return {"type": "move_joker", "from": a[0] + 1, "to": a[1] + 1}
            case "move_hand":
                return {"type": "move_hand", "from": a[0] + 1, "to": a[1] + 1}
            case "sort":
                return {"type": "sort_hand", "by": "suit" if a[0] else "value"}
            case "buy":
                return {"type": "buy", "index": a[0] + 1}
            case "buy_use":
                return {"type": "buy_and_use", "index": a[0] + 1, "cards": cards(a[1])}
            case "redeem":
                return {"type": "redeem", "index": a[0] + 1}
            case "open":
                return {"type": "open", "index": a[0] + 1}
            case "reroll":
                return {"type": "reroll"}
            case "end_shop":
                return {"type": "end_shop"}
            case "cash_out":
                return {"type": "cash_out"}
            case "pick":
                return {"type": "pick", "index": a[0] + 1, "cards": cards(a[1])}
            case "skip_pack":
                return {"type": "skip_pack"}
        raise AssertionError(op)

    def harness_lua(self) -> str:
        """The exact ``H.act{...}`` Lua command diff.rs sends to the oracle."""
        return f"H.act({lua_table(self.harness())})"

    def balatrobot(self, n_jokers: int = 0, n_hand: int = 0) -> tuple[str, dict[str, Any]] | None:
        """Native BalatroBot call for this action, or ``None`` if inexpressible.

        ``n_jokers`` / ``n_hand`` (current area sizes) are needed to turn a
        move into the full permutation that ``rearrange`` takes.
        """
        op, a = self.op, self.args
        match op:
            case "select":
                return "select", {}
            case "skip":
                return "skip", {}
            case "play":
                return "play", {"cards": mask_indices(a[0])}
            case "discard":
                return "discard", {"cards": mask_indices(a[0])}
            case "use":
                p: dict[str, Any] = {"consumable": a[0]}
                if a[1]:
                    p["cards"] = mask_indices(a[1])
                return "use", p
            case "sell_joker":
                return "sell", {"joker": a[0]}
            case "sell_consumable":
                return "sell", {"consumable": a[0]}
            case "move_joker":
                return "rearrange", {"jokers": move_permutation(n_jokers, a[0], a[1])}
            case "move_hand":
                return "rearrange", {"hand": move_permutation(n_hand, a[0], a[1])}
            case "buy":
                return "buy", {"card": a[0]}
            case "redeem":
                return "buy", {"voucher": a[0]}
            case "open":
                return "buy", {"pack": a[0]}
            case "reroll":
                return "reroll", {}
            case "end_shop":
                return "next_round", {}
            case "cash_out":
                return "cash_out", {}
            case "pick":
                p = {"card": a[0]}
                if a[1]:
                    p["targets"] = mask_indices(a[1])
                return "pack", p
            case "skip_pack":
                return "pack", {"skip": True}
            case "reroll_boss" | "sort" | "buy_use":
                return None
        raise AssertionError(op)


def move_permutation(n: int, frm: int, to: int) -> list[int]:
    """``rearrange`` permutation (new[i] = old[perm[i]]) for remove+insert."""
    if not (0 <= frm < n and 0 <= to < n):
        raise ValueError(f"move {frm}->{to} out of range for {n} cards")
    perm = list(range(n))
    perm.insert(to, perm.pop(frm))
    return perm


def lua_value(v: Any) -> str:
    if v is None:
        return "nil"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v) if v != int(v) else str(int(v))
    if isinstance(v, str):
        return "'" + v.replace("\\", "\\\\").replace("'", "\\'") + "'"
    if isinstance(v, (list, tuple)):
        return "{" + ",".join(lua_value(x) for x in v) + "}"
    if isinstance(v, dict):
        return lua_table(v)
    raise TypeError(type(v))


def lua_table(d: dict[str, Any]) -> str:
    """A Lua table constructor (keys as identifiers; ``from`` etc. are fine)."""
    parts = [f"{k}={lua_value(v)}" for k, v in d.items() if v is not None]
    return "{" + ", ".join(parts) + "}"


def parse_action(a: Action | str) -> Action:
    if isinstance(a, Action):
        return a
    p = Action.parse(a)
    if p is None:
        raise ValueError(f"not an action code: {a!r}")
    return p
