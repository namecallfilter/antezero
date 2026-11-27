"""Replays a telemetry log of the in-game mod (`balatro_mod/src/run_tracker.lua`) in
AnteZero's simulator and compares the two after every action.

An entry holds the game's state at the moment the player clicked an action, so entry
`i`'s state is compared with the simulator after actions `1..i-1`. Unlike
`testing/test_engine.py`, nothing of the log is fed back into the simulator: it starts
from the seed, deck and stake and is given only the player's actions.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import antezero_sim

from antezero_bridge import codes

EDITIONS = {"none": None, "foil": "foil", "holographic": "holo", "polychrome": "polychrome", "negative": "negative"}
EDITION_IDS = {"none": 0, "foil": 1, "holographic": 2, "polychrome": 3, "negative": 4}
RANK_CODES = {"10": "T", "Jack": "J", "Queen": "Q", "King": "K", "Ace": "A"}
LOCATIONS = {
    "BLIND_SELECT": "select",
    "SELECTING_HAND": "blind",
    "ROUND_EVAL": "blind",
    "GAME_OVER": "blind",
    "SHOP": "shop",
}
# Above this, the game's numbers are compared to 12 digits: the log is JSON written by Lua.
REL_TOL = 1e-11


def load(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def back_key(deck_name: str) -> str:
    return "b_" + deck_name.lower().removesuffix(" deck").strip().replace(" ", "_")


def dump(game: antezero_sim.Game) -> dict[str, Any]:
    return json.loads(game.state_json())


def start_game(entries: list[dict[str, Any]]) -> antezero_sim.Game:
    """The simulator at the log's first entry.

    A log that opens with the mod's "Setup Test Suite" (`balatro_mod/src/ui.lua`) starts
    from a run the mod edited: jokers and consumables are added in the order the mod made
    them, then Ceremonial Dagger's place and the deck's enhancements are set in the dump.
    """
    first = entries[0]["state"]
    game = antezero_sim.Game(first["seed"], back_key(first["deck_name"]), int(first["stake"]))
    # The real game ran under Steamodded, which rounds xMult differently from vanilla.
    game.steamodded = True
    if entries[0]["action"] != "Setup Test Suite":
        return game

    fresh = dump(game)
    owned = {c["sort_id"] for c in fresh["consumables"]}
    for joker in sorted(first["jokers"], key=lambda c: c["sort_id"]):
        game.inject(f"inject_joker {codes.JOKERS.index(joker['key'])} {EDITION_IDS[joker['edition']]} 0")
    spawned = sorted(first["consumables"], key=lambda c: c["sort_id"])[len(owned):]
    for card in spawned:
        game.inject(f"inject_cons {codes.consumable_code(card['key'])} {EDITION_IDS[card['edition']]}")

    state = dump(game)
    by_rank = dict(zip(sorted(c["sort_id"] for c in first["jokers"]), sorted(c["sort_id"] for c in state["jokers"])))
    by_id = {c["sort_id"]: c for c in state["jokers"]}
    state["jokers"] = [by_id[by_rank[c["sort_id"]]] for c in first["jokers"]]
    enhancement = {c["playing_card"]: c["key"] for c in first["deck"]}
    for card in state["deck"]:
        card["key"] = enhancement[card["sort_id"]]
        card["ability"]["set"] = "Default" if card["key"] == "c_base" else "Enhanced"
    # The mod makes its cards from its settings menu, and the game does not count a card made
    # under a menu as seen: the pools still offer it.
    state["used_jokers"] = fresh["used_jokers"]
    # `card_limit = max(card_limit, #cards)`, before the negative editions have added their slots.
    for area in ("joker", "consumable"):
        real = max(state[f"{area}_real_limit"], len(state[f"{area}s"]))
        state[f"{area}_limit"] += real - state[f"{area}_real_limit"]
        state[f"{area}_real_limit"] = real
    game.load_state(json.dumps(state))
    return game


def _lua_card(card: dict[str, Any]) -> tuple[Any, ...]:
    """What identifies a card of the log: its face, center, edition and seal."""
    face = None
    if card.get("suit", "none") != "none":
        face = f"{card['suit'][0]}_{RANK_CODES.get(card['value'], card['value'])}"
    return (face, _center(card.get("key")), EDITIONS[card["edition"]], None if card["seal"] == "none" else card["seal"])


def _sim_card(card: dict[str, Any]) -> tuple[Any, ...]:
    seal = card.get("seal")
    return (card.get("card"), _center(card["key"]), card["edition"], seal.lower() if seal else None)


def _center(key: str | None) -> str | None:
    # The first shop's Buffoon Pack takes its art (`_1` or `_2`) from `math.random`, not the seed.
    return "p_buffoon_normal" if key in ("p_buffoon_normal_1", "p_buffoon_normal_2") else key


def _close(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return a == b
    return abs(a - b) <= REL_TOL * max(1.0, abs(a), abs(b))


def _cards(name: str, sim: list[dict], lua: list[dict], out: list[str], extra: tuple[str, ...] = ()) -> None:
    a = [_sim_card(c) for c in sim]
    b = [_lua_card(c) for c in lua]
    if a != b:
        if Counter(a) == Counter(b):
            out.append(f"{name}: same cards, different order")
        else:
            out.append(f"{name}: sim-only {sorted(map(str, (Counter(a) - Counter(b)).elements()))} "
                       f"game-only {sorted(map(str, (Counter(b) - Counter(a)).elements()))}")
        return
    for i, (s, g) in enumerate(zip(sim, lua)):
        for key in extra:
            sv = s["ability"].get(key, s.get(key)) if key not in ("cost", "sell_cost") else s[key]
            if key == "debuffed":
                sv = s["debuff"]
            if key == "bonus_chips":
                sv = s["ability"].get("perma_bonus", 0)
            if key in g and g[key] is not None and sv is not None and not _close(sv, g[key]):
                out.append(f"{name}[{i}] {g.get('key')} {key}: sim {sv} game {g[key]}")


def location(state: dict[str, Any]) -> str:
    return LOCATIONS.get(state["state"], "pack")


def compare(sim: dict[str, Any], lua: dict[str, Any]) -> list[str]:
    """Every field of the log's state that differs from the simulator's dump."""
    out: list[str] = []

    def num(name: str, a: Any, b: Any) -> None:
        if not _close(a, b):
            out.append(f"{name}: sim {a} game {b}")

    num("money", sim["dollars"], lua["money"])
    num("ante", sim["ante"], lua["ante"])
    num("round", sim["round"], lua["round"])
    num("hands_left", sim["hands_left"], lua["hands_left"])
    num("discards_left", sim["discards_left"], lua["discards_left"])
    num("current_chips", sim["chips"], lua["current_chips"])
    num("target_chips", sim["blind"]["chips"], lua["target_chips"])
    num("reroll_cost", sim["reroll_cost"], lua["reroll_cost"])
    num("free_rerolls", sim["free_rerolls"], lua["free_rerolls"])
    num("blinds_skipped", sim["skips"], lua["blinds_skipped"])
    num("unused_discards", sim["unused_discards"], lua["unused_discards"])
    num("hand_max_size", sim["hand_limit"], lua["hand_max_size"])
    num("joker_slots", sim["joker_limit"], lua["joker_slots"])
    num("consumable_slots", sim["consumable_limit"], lua["consumable_slots"])
    num("location", location(sim), lua["location"])
    num("blind", sim["blind"]["name"], lua["blind"])
    num("boss_blind", sim["blind_choices"]["Boss"], lua["boss_blind"])
    num("small_blind_tag", sim["blind_tags"]["Small"], lua["small_blind_tag"])
    num("big_blind_tag", sim["blind_tags"]["Big"], lua["big_blind_tag"])
    num("castle_suit", sim["castle"], lua["castle_suit"])
    num("ancient_suit", sim["ancient"], lua["ancient_suit"])
    num("mail_card_rank", sim["mail_card"]["rank"], lua["mail_card_rank"])
    num("idol_card", (sim["idol_card"]["rank"], sim["idol_card"]["suit"]),
        (lua["idol_card"]["rank"], lua["idol_card"]["suit"]))
    num("vouchers", sorted(sim["used_vouchers"]), sorted(lua["vouchers"]))
    num("tags", [t["key"] for t in sim["tag_info"]], [t["key"] for t in lua["active_tags"]])
    for hand, levels in sim["hands"].items():
        num(f"level of {hand}", levels["level"], lua["hand_levels"][hand])
        num(f"plays of {hand}", levels["played"], lua["hand_play_counts"][hand])

    playing = ("debuffed", "bonus_chips")
    owned = ("cost", "sell_cost", "debuffed", "extra_value")
    _cards("hand", sim["hand"], lua["hand"], out, playing)
    _cards("draw_pile", sim["deck"], lua["draw_pile"], out, playing)
    _cards("discard_pile", sim["discard"], lua["discard_pile"], out, playing)
    _cards("jokers", sim["jokers"], lua["jokers"], out,
           (*owned, "mult", "x_mult", "eternal", "perishable", "rental", "to_do_poker_hand",
            "loyalty_remaining", "invis_rounds", "yorick_discards"))
    _cards("consumables", sim["consumables"], lua["consumables"], out, owned)
    _cards("shop", sim["shop_jokers"], lua["shop"], out, ("cost", "eternal", "perishable", "rental"))
    _cards("shop_vouchers", sim["shop_vouchers"], lua["shop_vouchers"], out, ("cost",))
    _cards("shop_packs", sim["shop_booster"], lua["shop_packs"], out, ("cost",))
    _cards("pack_cards", sim["pack_cards"] or [], lua["pack_cards"], out)
    if lua["pack_cards"]:
        num("pack_choices_remaining", sim["pack_choices"], lua["pack_choices_remaining"])

    game_rng = lua["prng_streams"]
    for key in sorted(set(sim["rng"]) | set(game_rng)):
        # Steamodded rolls its own hidden cards (`soul_smods_<set><ante>`) beside the game's Soul
        # roll; with no modded cards in the pools the roll decides nothing.
        if key == "seed" or key.startswith("soul_smods_"):
            continue
        a, b = sim["rng"].get(key), game_rng.get(key)
        if a is None or b is None or not _close(a, b):
            out.append(f"rng {key}: sim {a} game {b}")
    return out


def _moves(current: list[int], target: list[int], verb: str) -> list[str]:
    """The drags that turn `current` into `target` (the same ids in another order)."""
    order = list(current)
    out = []
    for to, card in enumerate(target):
        frm = order.index(card)
        if frm != to:
            order.insert(to, order.pop(frm))
            out.append(f"{verb} {frm} {to}")
    return out


def reorders(sim: dict[str, Any], lua: dict[str, Any]) -> list[str]:
    """The joker and hand drags the player made since the last action (the log has no entry for them)."""
    out = []
    for area, verb in (("jokers", "move_joker"), ("hand", "move_hand")):
        a, b = sim[area], lua[area]
        cards, logged = [_sim_card(c) for c in a], [_lua_card(c) for c in b]
        if cards == logged or Counter(cards) != Counter(logged):
            continue
        # Both engines number cards in the order they were made, the game from a higher start.
        rank = dict(zip(sorted(c["sort_id"] for c in b), sorted(c["sort_id"] for c in a)))
        out += _moves([c["sort_id"] for c in a], [rank[c["sort_id"]] for c in b], verb)
    return out


def _mask(lua: dict[str, Any]) -> int:
    return sum(1 << i for i, card in enumerate(lua["hand"]) if card.get("selected"))


def action_code(entry: dict[str, Any]) -> str | None:
    """The simulator's code of a logged action; None for an entry that is not the player's."""
    name, meta, lua = entry["action"], entry["action_metadata"], entry["state"]
    index = meta.get("target_idx")
    match name:
        case "Select Blind":
            return "select"
        case "Skip Blind":
            return "skip"
        case "Reroll Boss":
            # The Boss Tag rerolls by itself, through the function the button calls.
            return None if any(t["key"] == "tag_boss" for t in lua["active_tags"]) else "reroll_boss"
        case "Play Hand":
            return f"play {_mask(lua)}"
        case "Discard":
            # The Hook's discard is the game's, logged from inside the play.
            return None if meta.get("hook") else f"discard {_mask(lua)}"
        case "Use Consumable":
            return f"use {index} {_mask(lua)}"
        case "Sell Card":
            jokers = len(lua["jokers"])
            return f"sell_joker {index}" if index < jokers else f"sell_consumable {index - jokers}"
        case "Buy Card":
            return f"buy {index}"
        case "Buy and Use":
            return f"buy_use {index} {_mask(lua)}"
        case "Redeem Voucher":
            return f"redeem {index}"
        case "Open Pack":
            # A skip tag's pack opens by itself: the entry has no shop slot.
            return None if index is None else f"open {index}"
        case "Use In Pack":
            return f"pick {index} {_mask(lua)}"
        case "Skip Pack":
            return "skip_pack"
        case "Reroll Shop":
            return "reroll"
        case "Next Round":
            return "end_shop"
        case "Cash Out":
            return "cash_out"
        case "Sort by Rank":
            return "sort 0"
        case "Sort by Suit":
            return "sort 1"
        case "Setup Test Suite":
            return None
    raise ValueError(f"unknown telemetry action {name!r}")
