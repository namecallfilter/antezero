"""Chooses the simulator action that a vision read of a click stands for.

Stage 1 of the video pipeline names the button that was clicked, the classifiers read which
cards or which slot it applied to, and the HUD shows the result. With the seed known the
simulator lists the legal actions of that button and predicts the result of each, so a read
that is illegal, or whose result is not the one on screen, is replaced by the nearest action
that is.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from typing import Any, Protocol

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
    return [code for code in game.legal_codes() if code == button or code.startswith(button + " ")]


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
# From here the game prints a number as a few digits and an exponent (`number_format`).
E_SWITCH_POINT = 1e11
# Clicks the detector can miss without the game stopping: each has one outcome.
BRIDGES = ("cash_out", "end_shop", "skip_pack", "select")


class Unresolved(Exception):
    """The screen does not say what a click did: nothing is recorded for it, and the walk stops."""


class Screen(Protocol):
    """What the classifiers read on the frame just before a click."""

    # The HUD numbers that were read, by their names in `HUD_NUMBERS`.
    hud: dict[str, Any]

    def cards(self, states: list[dict[str, Any]]) -> dict[str, Any]:
        """The jokers, consumables and hand on the screen, which shows one of `states`."""

    def sales(self, game: antezero_sim.Game) -> list[str]:
        """The sales that turn `game` into what the screen shows, if their clicks were missed."""


@dataclass(frozen=True)
class Settled:
    # The action to record; None for a click that did nothing.
    code: str | None
    # "forced", "read", "repaired" or "rejected", as `video.Walk.outcomes` counts them.
    outcome: str
    # The other results that the screen fits as well (another action's, or nothing happening).
    # Only the classifiers' read is recorded with any: it then rests on them alone.
    rivals: int = 0
    # Sales the screen shows after it, whose clicks the detector missed.
    sales: tuple[str, ...] = ()


def predicted(game: antezero_sim.Game) -> dict[str, Any]:
    """What the HUD OCR and the detectors should read in `game`, and its state."""
    state = json.loads(game.state_json())
    return {**{name: state[name] for name in HUD_NUMBERS},
            "jokers": sorted(c["key"] for c in state["jokers"]),
            "consumables": sorted(c["key"] for c in state["consumables"]),
            "hand": sorted(c["card"] for c in state["hand"] if c["facing"] == "front"), "state": state}


def _displays(number: float) -> set[float]:
    """What the HUD's text for `number` reads as.

    From `E_SWITCH_POINT` the game prints a mantissa and an exponent (`number_format`): vanilla
    three decimals, Steamodded two (one from 1e100).
    """
    if number < E_SWITCH_POINT:
        return {number}
    rounded = float(f"{number:.4g}")
    exponent = math.floor(math.log10(rounded))
    mantissa = math.floor(rounded / 10**exponent * 1000 + 0.4999999999999994) / 1000
    if mantissa >= 10:
        mantissa, exponent = mantissa / 10, exponent + 1
    return {float(f"{mantissa:.{decimals}f}e{exponent}") for decimals in (3, 1 if exponent >= 100 else 2)}


def agreement(expected: dict[str, Any], hud: dict[str, Any]) -> int:
    """How many of the HUD numbers read are the ones the HUD shows for `expected`."""
    return sum(value in _displays(expected[name]) for name, value in hud.items())


def _overlap(expected: dict[str, Any], seen: dict[str, Any]) -> float:
    """How well the cards read fit: the areas that are as expected, the hand by the share of its cards."""
    score = 0.0
    for area, cards in seen.items():
        both = Counter(cards) & Counter(expected[area])
        size = max(len(cards), len(expected[area]))
        score += (sum(both.values()) / size if size else 1.0) if area == "hand" else cards == expected[area]
    return score


def shows(game: antezero_sim.Game, hud: dict[str, Any]) -> bool:
    """Whether a screen with the HUD numbers `hud` is `game`: `MIN_AGREEMENT` numbers were read, and each is the game's."""
    return len(hud) >= MIN_AGREEMENT and agreement(predicted(game), hud) == len(hud)


def playable(game: antezero_sim.Game, buttons: tuple[str, ...]) -> tuple[antezero_sim.Game, str | None, list[str]]:
    """The legal actions of a clicked button, the game they are legal in, and the click missed before it.

    The game is `game`, or `game` after the one forced click of `BRIDGES` that makes the button
    legal, which the detector then missed. No legal actions if neither has any.
    """
    legal = [code for button in buttons for code in candidates(game, button)]
    if legal:
        return game, None, legal
    for bridge in BRIDGES:
        if candidates(game, bridge) == [bridge]:
            ahead = game.clone()
            ahead.step(bridge)
            legal = [code for button in buttons for code in candidates(ahead, button)]
            if legal:
                return ahead, bridge, legal
    return game, None, []


def _best(scores: dict[str | None, float]) -> list[str | None]:
    return [option for option, score in scores.items() if score == max(scores.values())]


def settle(game: antezero_sim.Game, legal: list[str], guess: str | None, screen: Screen,
           following: tuple[str, ...]) -> Settled:
    """What the click of a button did, by the screen before the next click.

    `legal` are the button's legal actions, `guess` the classifiers' read of the click (None if
    they read nothing), and `following` the buttons the next click may be of.

    A result (of an action, or of nothing happening, with the sales `screen` shows after it)
    fits if no other has more of the HUD's numbers and the next click's button is legal after it
    (rather after it than after a click missed in between). Where the fits differ in the cards
    on screen, the cards are read: the read is recorded if it is among the fits the cards suit
    best, and with no such read only a fit whose cards are exactly the ones read. A read that
    the cards do not suit best, and that no exact fit replaces, stays the read of the fits
    before the cards.

    Raises `Unresolved` if the next click's button is legal after no result, if the fits differ
    in a HUD number that was not read, or if the read is not a fit and the fits differ in their
    results. The caller holds the result to the screen (`shows`).
    """
    views, sold, reach = {}, {}, {}
    for option in [None, *legal]:
        after = game.clone()
        if option is not None:
            after.step(option)
        sold[option] = tuple(screen.sales(after))
        for code in sold[option]:
            after.step(code)
        views[option] = predicted(after)
        _, missed, clicks = playable(after, following)
        reach[option] = 0 if not clicks else 1 if missed else 2

    best = _best({option: agreement(views[option], screen.hud) for option in views})
    if not any(reach[option] for option in best):
        raise Unresolved("the next click's button is legal after no result that fits the screen")
    best = _best({option: reach[option] for option in best})
    if len({json.dumps([views[option][area] for area in CARDS]) for option in best}) > 1:
        seen = screen.cards([views[option]["state"] for option in best])
        suited = _best({option: _overlap(views[option], seen) for option in best})
        exact = [option for option in suited if all(views[option][area] == cards for area, cards in seen.items())]
        if guess in suited or exact:
            best = suited if guess in suited else exact
    for name in HUD_NUMBERS:
        if name not in screen.hud and len({views[option][name] for option in best}) > 1:
            raise Unresolved(f"the HUD's {name} was not read, and the results that fit the screen differ in it")

    results = {option: json.dumps(views[option]["state"], sort_keys=True) for option in best}
    if guess is not None and guess in best:
        rivals = set(results.values()) - {results[guess]}
        return Settled(guess, "forced" if len(legal) == 1 else "read", len(rivals), sold[guess])
    if len(set(results.values())) > 1:
        raise Unresolved(f"the read {guess} does not fit the screen, and {len(best)} results that differ fit it equally")
    actions = [option for option in best if option is not None]
    if not actions or (None in best and not sold[None]):
        return Settled(None, "rejected", sales=sold[None])
    code = min(actions, key=lambda other: distance(guess, other) if guess else 0)
    return Settled(code, "repaired", sales=sold[code])
