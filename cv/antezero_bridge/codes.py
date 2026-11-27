"""The numbers AnteZero's injection codes give to jokers and consumables."""

from __future__ import annotations

import antezero_sim

_CENTERS = [key for key, _, _ in antezero_sim.centers()]
JOKERS = [k for k in _CENTERS if k.startswith("j_")]
_CONSUMABLES = [k for k in _CENTERS if k.startswith("c_")]
# `inject_cons <set> <card>`: the tarots, planets and spectrals are numbered apart, each in the game's order.
TAROTS = _CONSUMABLES[: _CONSUMABLES.index("c_mercury")]
PLANETS = _CONSUMABLES[_CONSUMABLES.index("c_mercury"): _CONSUMABLES.index("c_familiar")]
SPECTRALS = _CONSUMABLES[_CONSUMABLES.index("c_familiar"):]


def consumable_code(key: str) -> str:
    for number, cards in enumerate((TAROTS, PLANETS, SPECTRALS)):
        if key in cards:
            return f"{number} {cards.index(key)}"
    raise ValueError(f"unknown consumable {key!r}")
