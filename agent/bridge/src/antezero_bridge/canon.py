"""Python port of ``sim/oracle/src/canon.rs::from_lua`` and ``canon::diff``.

The companion mod emits exactly the JSON of the oracle harness's ``H.state``
(see ``sim/oracle/lua/harness.lua``), so the Rust ``canon::from_lua`` can
consume real-game dumps unchanged. This module reproduces that flattening in
Python (string formatting included) so real-game runs can be compared against
oracle dumps, or against each other, without building the Rust side.

Keep in sync with canon.rs: ``JOKER_FIELDS``, ``JOKER_COMMON``, ``NUMS``,
``HAND_TYPES`` and ``normalize_key`` mirror the Rust definitions.
"""

from __future__ import annotations

import json
import math
from decimal import Decimal
from typing import Any

Canon = dict[str, str]

HAND_TYPES = (
    "Flush Five",
    "Flush House",
    "Five of a Kind",
    "Straight Flush",
    "Four of a Kind",
    "Full House",
    "Flush",
    "Straight",
    "Three of a Kind",
    "Two Pair",
    "Pair",
    "High Card",
)

JOKER_COMMON = ("mult", "x_mult", "extra_value", "eternal", "rental", "couponed")

JOKER_FIELDS: dict[str, tuple[str, ...]] = {
    "j_runner": ("extra.chips",),
    "j_ice_cream": ("extra.chips",),
    "j_square": ("extra.chips",),
    "j_castle": ("extra.chips",),
    "j_wee": ("extra.chips",),
    "j_rocket": ("extra.dollars",),
    "j_selzer": ("extra",),
    "j_turtle_bean": ("extra.h_size",),
    "j_invisible": ("invis_rounds",),
    "j_yorick": ("yorick_discards",),
    "j_caino": ("caino_xmult",),
    "j_loyalty_card": ("loyalty_remaining",),
    "j_todo_list": ("to_do_poker_hand",),
    "j_steel_joker": ("steel_tally",),
    "j_stone": ("stone_tally",),
    "j_drivers_license": ("driver_tally",),
    "j_cloud_9": ("nine_tally",),
}

NUMS = (
    "ante",
    "round",
    "dollars",
    "chips",
    "hands_left",
    "discards_left",
    "reroll_cost",
    "free_rerolls",
    "rr_hands",
    "rr_discards",
    "rr_reroll_cost",
    "skips",
    "interest_cap",
    "interest_amount",
    "discount_percent",
    "bankrupt_at",
    "probabilities",
    "shop_joker_max",
    "tarot_rate",
    "planet_rate",
    "edition_rate",
    "playing_card_rate",
    "spectral_rate",
    "joker_limit",
    "consumable_limit",
    "hand_limit",
    "hands_played",
    "hands_played_round",
    "discards_used",
    "unused_discards",
    "ecto_minus",
    "starting_deck_size",
    "round_dollars",
    "pack_size",
)

AREAS = (
    ("hand", "hand"),
    ("deck", "deck"),
    ("discard", "discard"),
    ("jokers", "jokers"),
    ("consumables", "consumables"),
    ("shop_jokers", "shop_jokers"),
    ("shop_vouchers", "shop_vouchers"),
    ("shop_booster", "shop_booster"),
    ("pack", "pack_cards"),
)

_NORMALIZE = {
    "p_buffoon_normal_2": "p_buffoon_normal_1",
    "p_arcana_mega_2": "p_arcana_mega_1",
    "p_celestial_mega_2": "p_celestial_mega_1",
}


def normalize_key(k: str) -> str:
    return _NORMALIZE.get(k, k)


def fmt(v: float) -> str:
    """Rust ``format!("{v}")`` for f64 (shortest round-trip, no exponent)."""
    v = float(v)
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    s = format(Decimal(repr(v)), "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def _b(v: bool) -> str:
    return "true" if v else "false"


# Accessors with the semantics of sim/oracle/src/json.rs.
def _get(j: Any, k: str) -> Any:
    return j.get(k) if isinstance(j, dict) else None


def _num(j: Any) -> float | None:
    if isinstance(j, bool):
        return None
    if isinstance(j, (int, float)):
        return float(j)
    if j == "inf":
        return math.inf
    if j == "-inf":
        return -math.inf
    if j == "nan":
        return math.nan
    return None


def _str(j: Any) -> str | None:
    return j if isinstance(j, str) else None


def _bool(j: Any) -> bool:
    return j is True


def _arr(j: Any) -> list:
    return j if isinstance(j, list) else []


def _num_or_nil(j: Any) -> str:
    n = _num(j)
    return "nil" if n is None else fmt(n)


def _lua_field(ab: Any, name: str) -> str:
    if name.startswith("extra."):
        v = _get(_get(ab, "extra"), name[len("extra.") :])
    else:
        v = _get(ab, name)
    if v is None:
        return "false" if name in ("eternal", "rental", "couponed", "perishable") else "nil"
    if isinstance(v, bool):
        return _b(v)
    if isinstance(v, (int, float)):
        return fmt(v)
    if isinstance(v, str):
        return v
    return "?"


def _lua_card(out: Canon, p: str, c: Any, area: str) -> None:
    ab = _get(c, "ability")
    sid = _num(_get(c, "sort_id"))
    out[f"{p}.id"] = fmt(-1.0 if sid is None else sid)
    out[f"{p}.key"] = normalize_key(_str(_get(c, "key")) or "?")
    out[f"{p}.edition"] = _str(_get(c, "edition")) or "nil"
    if area not in ("deck", "discard"):
        cost = _num(_get(c, "cost"))
        sell = _num(_get(c, "sell_cost"))
        out[f"{p}.cost"] = fmt(-1.0 if cost is None else cost)
        out[f"{p}.sell_cost"] = fmt(-1.0 if sell is None else sell)
    st = _str(_get(ab, "set")) or ""
    if st in ("Default", "Enhanced"):
        out[f"{p}.card"] = _str(_get(c, "card")) or "?"
        out[f"{p}.seal"] = _str(_get(c, "seal")) or "nil"
        pb = _num(_get(ab, "perma_bonus"))
        out[f"{p}.perma_bonus"] = fmt(0.0 if pb is None else pb)
        if area != "pack":
            out[f"{p}.debuff"] = _b(_bool(_get(c, "debuff")))
        if area == "hand":
            out[f"{p}.facing"] = _str(_get(c, "facing")) or "?"
            out[f"{p}.forced"] = _b(_bool(_get(ab, "forced_selection")))
        out[f"{p}.played_this_ante"] = _b(_bool(_get(ab, "played_this_ante")))
    if st == "Joker":
        out[f"{p}.debuff"] = _b(_bool(_get(c, "debuff")))
        if area == "jokers":
            out[f"{p}.facing"] = _str(_get(c, "facing")) or "?"
        key = _str(_get(c, "key")) or ""
        for name in JOKER_COMMON + JOKER_FIELDS.get(key, ()):
            out[f"{p}.ab.{name}"] = _lua_field(ab, name)
        if _bool(_get(ab, "perishable")):
            out[f"{p}.ab.perish_tally"] = _lua_field(ab, "perish_tally")
    if st in ("Tarot", "Planet", "Spectral"):
        ev = _num(_get(ab, "extra_value"))
        out[f"{p}.ab.extra_value"] = fmt(0.0 if ev is None else ev)


def from_lua(j: dict | str) -> Canon:
    """Flatten an ``H.state``-shaped dump (dict or JSON text) like canon::from_lua."""
    if isinstance(j, str):
        j = json.loads(j)
    o: Canon = {}
    o["state"] = _str(_get(j, "state")) or "?"
    for k in NUMS:
        o[k] = _num_or_nil(_get(j, k))
    o["pack_choices"] = _num_or_nil(_get(j, "pack_choices"))
    bl = _get(j, "blind")
    o["blind.chips"] = _num_or_nil(_get(bl, "chips"))
    o["blind.dollars"] = _num_or_nil(_get(bl, "dollars"))
    o["blind.name"] = _str(_get(bl, "name")) or ""
    o["blind.disabled"] = _b(_bool(_get(bl, "disabled")))
    o["blind_on_deck"] = _str(_get(j, "blind_on_deck")) or "nil"
    for n in ("Small", "Big", "Boss"):
        o[f"blind_states.{n}"] = _str(_get(_get(j, "blind_states"), n)) or "nil"
    o["blind_choices.Boss"] = _str(_get(_get(j, "blind_choices"), "Boss")) or "nil"
    o["blind_tags.Small"] = _str(_get(_get(j, "blind_tags"), "Small")) or "nil"
    o["blind_tags.Big"] = _str(_get(_get(j, "blind_tags"), "Big")) or "nil"
    o["voucher"] = _str(_get(j, "voucher")) or "nil"
    o["won"] = _b(_bool(_get(j, "won")))
    o["boss_rerolled"] = _b(_bool(_get(j, "boss_rerolled")))
    o["gros_michel_extinct"] = _b(_bool(_get(j, "gros_michel_extinct")))
    o["last_hand_played"] = _str(_get(j, "last_hand_played")) or "nil"
    o["last_tarot_planet"] = _str(_get(j, "last_tarot_planet")) or "nil"
    o["most_played"] = _str(_get(j, "most_played")) or "nil"
    o["idol.rank"] = _num_or_nil(_get(_get(j, "idol"), "rank"))
    o["idol.suit"] = _str(_get(_get(j, "idol"), "suit")) or "nil"
    o["mail"] = _num_or_nil(_get(j, "mail"))
    o["ancient"] = _str(_get(j, "ancient")) or "nil"
    o["castle"] = _str(_get(j, "castle")) or "nil"
    o["sort_mode"] = _str(_get(j, "sort_mode")) or "desc"
    for n in HAND_TYPES:
        lv = _get(_get(j, "hands"), n)
        for k in ("level", "chips", "mult", "played", "played_this_round"):
            o[f"hands.{n}.{k}"] = _num_or_nil(_get(lv, k))
        o[f"hands.{n}.visible"] = _b(_bool(_get(lv, "visible")))
    for i, t in enumerate(_arr(_get(j, "tag_info"))):
        key = _str(_get(t, "key")) or "?"
        o[f"tags.{i}.key"] = key
        o[f"tags.{i}.triggered"] = _b(_bool(_get(t, "triggered")))
        if key == "tag_orbital":
            o[f"tags.{i}.orbital"] = _str(_get(t, "orbital")) or "nil"
    uv = [v for v in _arr(_get(j, "used_vouchers")) if isinstance(v, str)]
    o["used_vouchers"] = ",".join(uv)
    uj = sorted(
        k
        for k in _arr(_get(j, "used_jokers"))
        if isinstance(k, str) and k.startswith(("j_", "c_", "v_", "p_")) and k != "c_base"
    )
    o["used_jokers"] = ",".join(uj)
    for name, key in AREAS:
        v = _arr(_get(j, key))
        o[f"{name}.len"] = str(len(v))
        for i, c in enumerate(v):
            _lua_card(o, f"{name}.{i}", c, name)
    rng = _get(j, "rng")
    if isinstance(rng, dict):
        for k, v in rng.items():
            if k in ("seed", "hashed_seed"):
                continue
            n = _num(v)
            if n is not None:
                o[f"rng.{k}"] = fmt(n)
    return o


def diff(a: Canon, b: Canon) -> list[tuple[str, str, str]]:
    """Differences ``(path, a, b)`` with the GAME_OVER rule of canon::diff."""
    if a.get("state") == "GAME_OVER" or b.get("state") == "GAME_OVER":
        if a.get("state") != b.get("state"):
            return [("state", a.get("state", "<missing>"), b.get("state", "<missing>"))]
        return []
    out = []
    for k, v in sorted(a.items()):
        w = b.get(k)
        if w is None:
            out.append((k, v, "<missing>"))
        elif w != v:
            out.append((k, v, w))
    for k, w in sorted(b.items()):
        if k not in a:
            out.append((k, "<missing>", w))
    return out
