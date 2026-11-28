"""Puts the logged game's state into the simulator after the two have diverged, so the rest of
a telemetry log can still be compared one transition at a time.

Only what the log records can be restored: the counters, the RNG streams and the cards the
player owns. A shop or an open pack that differs cannot, and ends the replay.
"""

from __future__ import annotations

import json
from typing import Any

import antezero_sim

from antezero_bridge import codes
from antezero_bridge.telemetry import EDITION_IDS, EDITIONS, RANK_CODES, compare, dump

SUITS = ["Spades", "Hearts", "Clubs", "Diamonds"]
ENHANCEMENTS = ["c_base", "m_bonus", "m_mult", "m_wild", "m_glass", "m_steel", "m_stone", "m_gold", "m_lucky"]
SEALS = ["none", "gold", "red", "blue", "purple"]
RANK_IDS = {**{str(n): n for n in range(2, 11)}, "Jack": 11, "Queen": 12, "King": 13, "Ace": 14}
JOKER_STATE = ("mult", "x_mult", "extra_value", "to_do_poker_hand", "loyalty_remaining", "invis_rounds",
               "yorick_discards", "eternal", "perishable", "rental")


class Unsupported(Exception):
    """The log does not record enough to restore this state."""


def _fresh(code: str, area: str) -> dict[str, Any]:
    game = antezero_sim.Game("RESYNC")
    game.inject(code)
    cards = dump(game)[area]
    return cards[0] if area == "deck" else cards[-1]


def _playing_card(have: dict[str, Any] | None, card: dict[str, Any], sort_id: int) -> dict[str, Any]:
    if have is None or "card" not in have or EDITIONS[card["edition"]] != have["edition"]:
        have = _fresh(
            f"inject_card {SUITS.index(card['suit'])} {RANK_IDS[card['value']]} "
            f"{ENHANCEMENTS.index(card['key'])} {EDITION_IDS[card['edition']]} {SEALS.index(card['seal'])}",
            "deck",
        )
    have["sort_id"] = sort_id
    have["suit"] = card["suit"]
    have["rank"] = RANK_IDS[card["value"]]
    have["card"] = f"{card['suit'][0]}_{RANK_CODES.get(card['value'], card['value'])}"
    have["key"] = card["key"]
    have["ability"]["set"] = "Default" if card["key"] == "c_base" else "Enhanced"
    have["seal"] = None if card["seal"] == "none" else card["seal"].capitalize()
    have["debuff"] = card["debuffed"]
    have["perma_bonus"] = have["ability"]["perma_bonus"] = card["bonus_chips"]
    return have


def _owned(have: dict[str, Any] | None, card: dict[str, Any], sort_id: int) -> dict[str, Any]:
    if have is None or have["key"] != card["key"] or EDITIONS[card["edition"]] != have["edition"]:
        edition = EDITION_IDS[card["edition"]]
        if card["set"] == "Joker":
            have = _fresh(f"inject_joker {codes.JOKERS.index(card['key'])} {edition} 0", "jokers")
        else:
            have = _fresh(f"inject_cons {codes.consumable_code(card['key'])} {edition}", "consumables")
    have["sort_id"] = sort_id
    have["facing"] = "back" if card["facing_back"] else "front"
    have["debuff"] = card["debuffed"]
    have["cost"], have["sell_cost"] = card["cost"], card["sell_cost"]
    for key in JOKER_STATE:
        if key in have["ability"] and card.get(key) is not None:
            have["ability"][key] = card[key]
    return have


def resync(game: antezero_sim.Game, lua: dict[str, Any], offset: int) -> None:
    """Loads the logged state `lua` into `game`; `offset` is the game's card number minus the simulator's."""
    sim = dump(game)
    areas = {"hand": "hand", "deck": "draw_pile", "discard": "discard_pile", "jokers": "jokers",
             "consumables": "consumables"}
    pool = {c["sort_id"]: c for area in areas for c in sim[area]}
    before = {c["key"] for area in ("jokers", "consumables") for c in sim[area]}
    for area, logged in areas.items():
        cards = []
        for card in lua[logged]:
            sort_id = card["sort_id"] - offset
            have = pool.pop(sort_id, None)
            if card["suit"] != "none":
                new = _playing_card(have, card, sort_id)
                new["facing"] = "back" if area == "deck" or card["facing_back"] else "front"
            else:
                new = _owned(have, card, sort_id)
            cards.append(new)
        sim[area] = cards
    sim["sort_id"] = max(sim["sort_id"], *(c["sort_id"] for area in areas for c in sim[area]))

    after = {c["key"] for area in ("jokers", "consumables") for c in sim[area]}
    sim["used_jokers"] = sorted((set(sim["used_jokers"]) - (before - after)) | (after - before))
    for area, limit in (("jokers", "joker_limit"), ("consumables", "consumable_limit")):
        negatives = sum(c["edition"] == "negative" for c in sim[area])
        was = sum(c["edition"] == "negative" for c in json.loads(game.state_json())[area])
        sim[limit] += negatives - was

    sim["dollars"] = lua["money"]
    sim["hands_left"], sim["discards_left"] = lua["hands_left"], lua["discards_left"]
    sim["rr_hands"], sim["rr_discards"] = lua["base_hands"], lua["base_discards"]
    sim["chips"] = lua["current_chips"]
    sim["blind"]["chips"] = lua["target_chips"]
    sim["reroll_cost"], sim["free_rerolls"] = lua["reroll_cost"], lua["free_rerolls"]
    sim["unused_discards"] = lua["unused_discards"]
    sim["hand_real_limit"] += lua["hand_max_size"] - sim["hand_limit"]
    sim["hand_limit"] = lua["hand_max_size"]
    sim["hands_played_round"] = lua["hands_played_this_round"]
    sim["discards_used"] = lua["discards_used_this_round"]
    for hand, levels in sim["hands"].items():
        if levels["level"] != lua["hand_levels"][hand]:
            raise Unsupported(f"the level of {hand} differs")
        levels["played"] = lua["hand_play_counts"][hand]
    sim["rng"] = {"seed": sim["rng"]["seed"], "hashed_seed": sim["rng"]["hashed_seed"],
                  **{k: v for k, v in lua["prng_streams"].items()
                     if k != "hashed_seed" and not k.startswith("soul_smods_")}}
    try:
        game.load_state(json.dumps(sim))
    except ValueError as e:
        raise Unsupported(str(e)) from e
    left = compare(dump(game), lua)
    if left:
        raise Unsupported("; ".join(left[:4]))
