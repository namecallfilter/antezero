"""Chooses the simulator action that a vision read of a click stands for.

Stage 1 of the video pipeline names the button that was clicked, the classifiers read which
cards or which slot it applied to, and the HUD shows the result. With the seed known the
simulator lists the legal actions of that button and predicts the result of each, so a read
that is illegal, or whose result is not the one on screen, is replaced by the nearest action
that is.
"""

from __future__ import annotations

import json
from typing import Any

import antezero_sim

# What the pipeline can read off the screen after an action, by level of effort: the HUD's
# numbers (its OCR model), then the faces of the hand cards (its card classifier), then which
# jokers, consumables, shop items and pack cards are there (its shop classifier).
HUD = ("dollars", "chips", "hands_left", "discards_left", "ante", "round", "reroll_cost", "state")
AREAS = ("jokers", "consumables", "shop_jokers", "shop_booster", "shop_vouchers", "pack_cards")


def observe(game: antezero_sim.Game, hand: bool = False, items: bool = False) -> dict[str, Any]:
    """The simulator's prediction of what the screen shows."""
    state = json.loads(game.state_json())
    seen = {key: state[key] for key in HUD}
    seen["blind_chips"] = state["blind"]["chips"]
    for area in AREAS:
        seen[area] = len(state[area] or [])
    seen["hand_size"] = len(state["hand"])
    if hand:
        seen["hand"] = [_face(c) for c in state["hand"]]
    if items:
        for area in AREAS:
            seen[f"{area}_cards"] = [(c.get("card", c["key"]), c["edition"]) for c in state[area] or []]
    return seen


def _face(card: dict[str, Any]) -> tuple[Any, ...] | None:
    if card["facing"] != "front":
        return None
    return (card["card"], card["key"], card["edition"], card["seal"])


def verb(code: str) -> str:
    return code.split()[0]


def candidates(game: antezero_sim.Game, button: str) -> list[str]:
    """The legal actions of one button."""
    if button == "sort":
        # `legal_codes` leaves out the free reorders; the two sort buttons are there in every round.
        return ["sort 0", "sort 1"] if game.phase == "round" else []
    return [code for code in game.legal_codes() if verb(code) == button]


def consistent(game: antezero_sim.Game, codes: list[str], screen: dict[str, Any]) -> list[str]:
    """The actions whose result is the one on `screen` (an `observe` of the next state, or part of one)."""
    out = []
    for code in codes:
        after = game.clone()
        after.step(code)
        predicted = observe(after, hand="hand" in screen, items="jokers_cards" in screen)
        if all(predicted[key] == value for key, value in screen.items()):
            out.append(code)
    return out


def distance(a: str, b: str) -> int:
    """How far a read is from an action of the same button: slots that differ, plus hand cards that differ."""
    x, y = a.split()[1:], b.split()[1:]
    if len(x) != len(y):
        return 99
    if not x:
        return 0
    cards = bin(int(x[-1]) ^ int(y[-1])).count("1") if len(x) == 2 or verb(a) in ("play", "discard") else None
    if cards is None:
        return sum(p != q for p, q in zip(x, y))
    return cards + sum(p != q for p, q in zip(x[:-1], y[:-1]))


def repair(game: antezero_sim.Game, read: str, screen: dict[str, Any]) -> str | None:
    """The action to record for `read`: the legal action of its button nearest to it among those
    that give `screen`; None if no action of that button does (the click or the HUD was misread)."""
    fits = consistent(game, candidates(game, verb(read)), screen)
    return min(fits, key=lambda code: distance(read, code), default=None)
