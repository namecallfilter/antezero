"""Chooses the simulator action that a vision read of a click stands for.

Stage 1 of the video pipeline names the button that was clicked, the classifiers read which
cards or which slot it applied to, and the HUD shows the result. With the seed known the
simulator lists the legal actions of that button and predicts the result of each, so a read
that is illegal, or whose result is not the one on screen, is replaced by the nearest action
that is.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
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


# The HUD's numbers that the video walk reads on every screen, by their names in the simulator's state.
HUD_NUMBERS = ("dollars", "chips", "hands_left", "discards_left", "ante", "round")
# How many of them a screen must show for it to be checked against the simulator at all: the OCR
# leaves a number unread when it is not sure of it (on some screens the ante and the round score).
MIN_AGREEMENT = 4
# The cards the detectors read when the HUD leaves a choice.
CARDS = ("jokers", "consumables", "hand")
# From here the game prints a number as four digits and an exponent (`number_format`).
E_SWITCH_POINT = 1e11


class Unresolved(Exception):
    """The screen does not say what a click did: nothing is recorded for it, and the walk stops."""


@dataclass(frozen=True)
class Settled:
    # The action to record; None for a click that did nothing.
    code: str | None
    # "forced", "read", "repaired" or "rejected", as `video.Walk.outcomes` counts them.
    outcome: str
    # The other actions with another result that the screen fits as well. Only a read that is
    # itself one of the fits is recorded with any: it then rests on the classifiers alone.
    rivals: int = 0


def predicted(game: antezero_sim.Game) -> dict[str, Any]:
    """What the HUD OCR and the detectors should read in `game`, and its state."""
    state = json.loads(game.state_json())
    return {**{name: state[name] for name in HUD_NUMBERS},
            "jokers": sorted(c["key"] for c in state["jokers"]),
            "consumables": sorted(c["key"] for c in state["consumables"]),
            "hand": sorted(c["card"] for c in state["hand"] if c["facing"] == "front"), "state": state}


def _shown(expected: Any, read: Any) -> bool:
    numbers = isinstance(expected, (int, float)) and isinstance(read, (int, float))
    if numbers and abs(expected) >= E_SWITCH_POINT:
        return abs(expected - read) <= 1e-3 * abs(expected)
    return expected == read


def agreement(expected: dict[str, Any], screen: dict[str, Any]) -> float:
    """How many of the things read off the screen are as expected (a hand counts by the share of its cards)."""
    score = 0.0
    for key, value in screen.items():
        if key == "hand":
            both = Counter(value) & Counter(expected["hand"])
            score += sum(both.values()) / max(len(value), len(expected["hand"]), 1)
        else:
            score += _shown(expected[key], value)
    return score


def shows(game: antezero_sim.Game, hud: dict[str, Any]) -> bool:
    """Whether a screen with the HUD numbers `hud` is `game`: `MIN_AGREEMENT` numbers were read, and each is the game's."""
    return len(hud) >= MIN_AGREEMENT and agreement(predicted(game), hud) == len(hud)


def _fits(scores: dict[str | None, float]) -> list[str | None]:
    return [option for option, score in scores.items() if score == max(scores.values())]


def settle(game: antezero_sim.Game, legal: list[str], guess: str, hud: dict[str, Any],
           cards: Callable[[dict[str, Any]], dict[str, Any]]) -> Settled:
    """What the click of a button did, by the screen before the next click.

    `legal` are the button's legal actions, `guess` the classifiers' read of the click, and `hud`
    the HUD numbers read on that screen, by their names in `HUD_NUMBERS`. `cards` reads the
    screen's jokers, consumables and hand, given the state the simulator expects there; it is
    called only if the HUD leaves a choice that those cards would settle.

    The result is the read if it is one of the actions that fit the screen best, and otherwise the
    one result that does (another action, or nothing at all). Raises `Unresolved` if the fits
    differ in a HUD number that was not read, or if the read is not a fit and the fits differ in
    their results. The caller holds the result to the screen (`shows`).
    """
    after = {None: predicted(game)}
    for code in legal:
        ahead = game.clone()
        ahead.step(code)
        after[code] = predicted(ahead)
    scores = {option: agreement(view, hud) for option, view in after.items()}
    best = _fits(scores)
    if len({json.dumps([after[option][area] for area in CARDS]) for option in best}) > 1:
        seen = cards(after[guess if guess in best else best[-1]]["state"])
        scores = {option: scores[option] + agreement(after[option], seen) for option in best}
        best = _fits(scores)
    for name in HUD_NUMBERS:
        if name not in hud and len({after[option][name] for option in best}) > 1:
            raise Unresolved(f"the HUD's {name} was not read, and the results that fit the screen differ in it")

    results = {option: json.dumps(after[option]["state"], sort_keys=True) for option in best}
    if guess in best:
        others = {result for option, result in results.items() if option is not None} - {results[guess]}
        return Settled(guess, "forced" if len(legal) == 1 else "read", len(others))
    if len(set(results.values())) > 1:
        raise Unresolved(f"the read {guess} does not fit the screen, and {len(best)} results that differ fit it equally")
    if None in best:
        return Settled(None, "rejected")
    return Settled(min(best, key=lambda other: distance(guess, other)), "repaired")
