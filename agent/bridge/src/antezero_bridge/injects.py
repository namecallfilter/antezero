"""Scenario injections (``sim/src/inject.rs``): parsing of history lines
(the same strict grammar as the Rust ``Inject`` ``FromStr``) and translation
to the oracle harness ``H.inject`` table (executed by the companion mod's
``az_inject``) or, where possible, to native BalatroBot ``add``/``set``.

History codes::

    inject_joker <joker idx> <edition idx> <eternal 0|1>
    inject_cons  <0 tarot|1 planet|2 spectral> <idx> <edition idx>
    inject_card  <suit idx> <rank 2..14> <enhancement idx> <edition idx> <seal 0..4>
    inject_dollars <finite amount>
    inject_voucher <voucher idx>
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from . import _codes
from .actions import lua_table

EDITIONS = (None, "foil", "holo", "polychrome", "negative")  # Edition as u8
SUIT_LETTERS = ("S", "H", "C", "D")  # Suit::ALL
ENHANCEMENTS = (  # Enhancement as u8
    "c_base", "m_bonus", "m_mult", "m_wild", "m_glass", "m_steel", "m_stone", "m_gold", "m_lucky",
)
SEALS = (None, "Gold", "Red", "Blue", "Purple")  # 0 = none, else Seal as u8 + 1
_BAD = "\0"  # an out-of-range index while parsing
CONSUMABLE_SETS = (_codes.TAROTS, _codes.PLANETS, _codes.SPECTRALS)

_BB_EDITION = {"foil": "FOIL", "holo": "HOLO", "polychrome": "POLYCHROME", "negative": "NEGATIVE"}
_BB_ENH = {
    "m_bonus": "BONUS", "m_mult": "MULT", "m_wild": "WILD", "m_glass": "GLASS",
    "m_steel": "STEEL", "m_stone": "STONE", "m_gold": "GOLD", "m_lucky": "LUCKY",
}


# What Rust's integer / f64 `FromStr` accept.
_UINT = re.compile(r"\+?[0-9]+")
_FLOAT = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


def rank_char(r: int) -> str:
    if 2 <= r <= 9:
        return str(r)
    return {10: "T", 11: "J", 12: "Q", 13: "K"}.get(r, "A")


@dataclass(frozen=True, slots=True)
class Inject:
    """A parsed injection: ``kind`` + the oracle harness table."""

    code: str
    kind: str  # joker | consumable | card | dollars | voucher
    table: dict[str, Any]

    @staticmethod
    def parse(s: str) -> Inject | None:
        """``None`` for anything that is not exactly an injection code."""
        toks = s.split()
        if not toks:
            return None
        op, args = toks[0], toks[1:]
        code = s.strip()
        arity = {"inject_joker": 3, "inject_cons": 3, "inject_card": 5, "inject_dollars": 1, "inject_voucher": 1}
        if arity.get(op) != len(args):
            return None
        if op == "inject_dollars":
            if not _FLOAT.fullmatch(args[0]) or not math.isfinite(amt := float(args[0])):
                return None
            return Inject(code, "dollars", {"kind": "dollars", "amount": int(amt) if amt == int(amt) else amt})
        if not all(_UINT.fullmatch(t) for t in args):
            return None
        v = [int(t) for t in args]

        def get(table, i):
            return table[i] if i < len(table) else _BAD

        if op == "inject_joker":
            if v[2] > 0xFF:
                return None
            t = {"kind": "joker", "key": get(_codes.JOKERS, v[0]), "edition": get(EDITIONS, v[1]), "eternal": v[2] != 0}
        elif op == "inject_cons":
            if v[0] > 2:
                return None
            t = {"kind": "consumable", "key": get(CONSUMABLE_SETS[v[0]], v[1]), "edition": get(EDITIONS, v[2])}
        elif op == "inject_card":
            if not 2 <= v[1] <= 14:
                return None
            t = {
                "kind": "card",
                "key": f"{get(SUIT_LETTERS, v[0])}_{rank_char(v[1])}",
                "center": get(ENHANCEMENTS, v[2]),
                "edition": get(EDITIONS, v[3]),
                "seal": get(SEALS, v[4]),
            }
        else:
            t = {"kind": "voucher", "key": get(_codes.VOUCHERS, v[0])}
        if _BAD in t.values() or _BAD in t["key"]:
            return None
        return Inject(code, t["kind"], t)

    def harness(self) -> dict[str, Any]:
        """``H.inject`` table (nil fields dropped)."""
        return {k: v for k, v in self.table.items() if v is not None}

    def harness_lua(self) -> str:
        """The exact ``H.inject{...}`` Lua command (as diff.rs inject_to_lua)."""
        return f"H.inject({lua_table(self.table)})"

    def balatrobot(self, dollars_now: float | None = None) -> tuple[str, dict[str, Any]] | None:
        """Closest native BalatroBot call, or ``None`` (vouchers: ``add`` puts a
        voucher in the shop instead of redeeming it). NOTE: BalatroBot ``add``
        goes through ``SMODS.add_card``, not vanilla ``add_joker`` /
        ``create_playing_card``, so RNG use and placement can differ from the
        oracle; prefer the companion route for parity work."""
        t = self.table
        if self.kind in ("joker", "consumable"):
            p: dict[str, Any] = {"key": t["key"]}
            if t.get("edition"):
                p["edition"] = _BB_EDITION[t["edition"]]
            if t.get("eternal"):
                p["eternal"] = True
            return "add", p
        if self.kind == "card":
            p = {"key": t["key"]}
            if t.get("center") and t["center"] != "c_base":
                p["enhancement"] = _BB_ENH[t["center"]]
            if t.get("edition"):
                p["edition"] = _BB_EDITION[t["edition"]]
            if t.get("seal"):
                p["seal"] = t["seal"].upper()
            return "add", p
        if self.kind == "dollars":
            if dollars_now is None:
                return None
            total = dollars_now + float(t["amount"])
            if total != int(total):
                return None
            return "set", {"money": int(total)}
        return None


def parse_inject(s: Inject | str) -> Inject:
    if isinstance(s, Inject):
        return s
    p = Inject.parse(s)
    if p is None:
        raise ValueError(f"not an inject code: {s!r}")
    return p
