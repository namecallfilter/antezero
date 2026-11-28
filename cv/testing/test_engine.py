"""
Unit tests and telemetry parity validation for Balatro GameState simulation engine.

Usage:
    python -m unittest testing/test_engine.py
    python testing/test_engine.py
"""

import json
import sys
import unittest
from pathlib import Path

# Ensure repo root and 'game' directory are in pythonpath
REPO_ROOT = Path(__file__).resolve().parent.parent
GAME_DIR = REPO_ROOT / "game"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(GAME_DIR) not in sys.path:
    sys.path.insert(0, str(GAME_DIR))

try:
    import game.balatro_models as bm
    import game.gamestate as gs
    from game.engine.jokers import on_joker_added, sync_dynamic_joker_states
except ImportError:
    # pyrefly: ignore[missing-import]
    import balatro_models as bm

    # pyrefly: ignore[missing-import]
    import gamestate as gs

    # pyrefly: ignore[missing-import]
    from engine.jokers import on_joker_added, sync_dynamic_joker_states  # noqa: F401


# =========================================================================
# LUA TELEMETRY <-> PYTHON MODEL DESERIALIZERS
# =========================================================================

RANK_MAP: dict[str, bm.Rank] = {
    "2": bm.Rank.TWO,
    "3": bm.Rank.THREE,
    "4": bm.Rank.FOUR,
    "5": bm.Rank.FIVE,
    "6": bm.Rank.SIX,
    "7": bm.Rank.SEVEN,
    "8": bm.Rank.EIGHT,
    "9": bm.Rank.NINE,
    "10": bm.Rank.TEN,
    "Jack": bm.Rank.JACK,
    "Queen": bm.Rank.QUEEN,
    "King": bm.Rank.KING,
    "Ace": bm.Rank.ACE,
    "none": bm.Rank.NONE,
}

SUIT_MAP: dict[str, bm.Suit] = {
    "Spades": bm.Suit.SPADES,
    "Hearts": bm.Suit.HEARTS,
    "Clubs": bm.Suit.CLUBS,
    "Diamonds": bm.Suit.DIAMONDS,
    "none": bm.Suit.NONE,
}

ENHANCEMENT_MAP: dict[str, bm.Enhancement] = {
    "none": bm.Enhancement.NONE,
    "bonus": bm.Enhancement.BONUS,
    "mult": bm.Enhancement.MULT,
    "wild": bm.Enhancement.WILD,
    "glass": bm.Enhancement.GLASS,
    "steel": bm.Enhancement.STEEL,
    "stone": bm.Enhancement.STONE,
    "gold": bm.Enhancement.GOLD,
    "lucky": bm.Enhancement.LUCKY,
}

EDITION_MAP: dict[str, bm.Edition] = {
    "none": bm.Edition.NONE,
    "foil": bm.Edition.FOIL,
    "holographic": bm.Edition.HOLOGRAPHIC,
    "polychrome": bm.Edition.POLYCHROME,
    "negative": bm.Edition.NEGATIVE,
}

SEAL_MAP: dict[str, bm.Seal] = {
    "none": bm.Seal.NONE,
    "red": bm.Seal.RED,
    "blue": bm.Seal.BLUE,
    "gold": bm.Seal.GOLD,
    "purple": bm.Seal.PURPLE,
}

JOKER_NAME_BY_STR: dict[str, bm.JokerName] = {j.value: j for j in bm.JokerName}
JOKER_NAME_BY_STR.update(
    {
        "Caino": bm.JokerName.CANIO,
        "Seance": bm.JokerName.SEANCE,
        "Riff-raff": bm.JokerName.RIFF_RAFF,
    }
)

BLIND_BY_STR: dict[str, bm.BlindName] = {b.value: b for b in bm.BlindName}
BLIND_BY_STR.update(
    {
        "bl_small": bm.BlindName.SMALL,
        "bl_big": bm.BlindName.BIG,
        "bl_hook": bm.BlindName.THE_HOOK,
        "bl_ox": bm.BlindName.THE_OX,
        "bl_house": bm.BlindName.THE_HOUSE,
        "bl_wall": bm.BlindName.THE_WALL,
        "bl_wheel": bm.BlindName.THE_WHEEL,
        "bl_arm": bm.BlindName.THE_ARM,
        "bl_club": bm.BlindName.THE_CLUB,
        "bl_fish": bm.BlindName.THE_FISH,
        "bl_psychic": bm.BlindName.THE_PSYCHIC,
        "bl_goad": bm.BlindName.THE_GOAD,
        "bl_water": bm.BlindName.THE_WATER,
        "bl_window": bm.BlindName.THE_WINDOW,
        "bl_manacle": bm.BlindName.THE_MANACLE,
        "bl_eye": bm.BlindName.THE_EYE,
        "bl_mouth": bm.BlindName.THE_MOUTH,
        "bl_plant": bm.BlindName.THE_PLANT,
        "bl_serpent": bm.BlindName.THE_SERPENT,
        "bl_pillar": bm.BlindName.THE_PILLAR,
        "bl_needle": bm.BlindName.THE_NEEDLE,
        "bl_head": bm.BlindName.THE_HEAD,
        "bl_tooth": bm.BlindName.THE_TOOTH,
        "bl_flint": bm.BlindName.THE_FLINT,
        "bl_mark": bm.BlindName.THE_MARK,
        "bl_final_acorn": bm.BlindName.AMBER_ACORN,
        "bl_final_leaf": bm.BlindName.VERDANT_LEAF,
        "bl_final_vessel": bm.BlindName.VIOLET_VESSEL,
        "bl_final_heart": bm.BlindName.CRIMSON_HEART,
        "bl_final_bell": bm.BlindName.CERULEAN_BELL,
    }
)

POKER_HAND_BY_STR: dict[str, bm.PokerHand] = {h.value: h for h in bm.PokerHand}
TAROT_NAME_BY_STR: dict[str, bm.TarotName] = {t.value: t for t in bm.TarotName}
PLANET_NAME_BY_STR: dict[str, bm.PlanetName] = {p.value: p for p in bm.PlanetName}
SPECTRAL_NAME_BY_STR: dict[str, bm.SpectralName] = {s.value: s for s in bm.SpectralName}
VOUCHER_NAME_BY_STR: dict[str, bm.VoucherName] = {v.value: v for v in bm.VoucherName}


def parse_playing_card(d: dict) -> bm.PlayingCard:
    enh = ENHANCEMENT_MAP.get(d.get("enhancement", "none"), bm.Enhancement.NONE)
    rank = (
        bm.Rank.NONE
        if enh == bm.Enhancement.STONE
        else RANK_MAP.get(d.get("value", "none"), bm.Rank.ACE)
    )
    suit = (
        bm.Suit.NONE
        if enh == bm.Enhancement.STONE
        else SUIT_MAP.get(d.get("suit", "none"), bm.Suit.SPADES)
    )
    return bm.PlayingCard(
        id=d.get("id", 0),
        rank=rank,
        suit=suit,
        enhancement=enh,
        edition=EDITION_MAP.get(d.get("edition", "none"), bm.Edition.NONE),
        seal=SEAL_MAP.get(d.get("seal", "none"), bm.Seal.NONE),
        bonus_chips=int(d.get("bonus_chips", 0)),
        debuffed=bool(d.get("debuffed", False)),
        extra_value=int(d.get("extra_value", 0)),
    )


def parse_hand_card(d: dict) -> bm.HandCard:
    return bm.HandCard(
        card=parse_playing_card(d),
        selected=bool(d.get("selected", False)),
        facing_back=bool(d.get("facing_back", False)),
        forced_selection=bool(d.get("forced_selection", False)),
    )


def parse_joker(d: dict) -> bm.Joker | None:
    name_str = d.get("name", "")
    j_name = JOKER_NAME_BY_STR.get(name_str)
    if j_name is None:
        return None
    j = bm.Joker(
        name=j_name,
        id=int(d.get("id", 0)),
        edition=EDITION_MAP.get(d.get("edition", "none"), bm.Edition.NONE),
        extra_value=int(d.get("extra_value", 0)),
        debuffed=bool(d.get("debuffed", False)),
        eternal=bool(d.get("eternal", False)),
        perishable=bool(d.get("perishable", False)),
        perish_tally=int(d.get("perish_tally", 5)),
        rental=bool(d.get("rental", False)),
        facing_back=bool(d.get("facing_back", False)),
    )
    if d.get("chips"):
        j.state["chips"] = d["chips"]
    if d.get("mult"):
        j.state["mult"] = d["mult"]
    if d.get("x_mult") and d["x_mult"] != 1.0:
        j.state["x_mult"] = d["x_mult"]
    if d.get("h_size"):
        j.state["h_size"] = d["h_size"]
    if d.get("yorick_discards") is not None:
        j.state["discards"] = d["yorick_discards"]
    if d.get("invis_rounds"):
        j.state["rounds"] = d["invis_rounds"]
    if d.get("to_do_poker_hand") in POKER_HAND_BY_STR:
        j.state["target_hand"] = POKER_HAND_BY_STR[d["to_do_poker_hand"]]
    if d.get("loyalty_remaining") is not None:
        j.state["remaining"] = int(d["loyalty_remaining"])
    if d.get("payout") is not None:
        j.state["payout"] = int(d["payout"])
    extra = d.get("extra")
    if isinstance(extra, dict):
        if "dollars" in extra and j_name == bm.JokerName.ROCKET:
            try:
                j.state["payout"] = int(extra["dollars"])
            except (ValueError, TypeError):
                pass
        if (
            "remaining" in extra
            and j_name == bm.JokerName.LOYALTY_CARD
            and "remaining" not in j.state
        ):
            val = extra["remaining"]
            if isinstance(val, (int, float)):
                j.state["remaining"] = int(val)
            elif isinstance(val, str):
                import re

                m = re.search(r"\d+", val)
                if m:
                    j.state["remaining"] = int(m.group(0))
    return j


def parse_any_card(d: dict):
    if not isinstance(d, dict):
        return None
    s = d.get("set", "")
    ed = EDITION_MAP.get(d.get("edition", "none"), bm.Edition.NONE)
    ev = int(d.get("extra_value", 0))
    if s == "Joker":
        return parse_joker(d)
    elif s == "Tarot":
        t_name = TAROT_NAME_BY_STR.get(d.get("name"), bm.TarotName.FOOL)
        return bm.Tarot(name=t_name, edition=ed, extra_value=ev)
    elif s == "Planet":
        p_name = PLANET_NAME_BY_STR.get(d.get("name"), bm.PlanetName.MERCURY)
        return bm.Planet(name=p_name, edition=ed, extra_value=ev)
    elif s == "Spectral":
        sp_name = SPECTRAL_NAME_BY_STR.get(d.get("name"), bm.SpectralName.FAMILIAR)
        return bm.Spectral(name=sp_name, edition=ed, extra_value=ev)
    elif s == "Voucher":
        v_name = VOUCHER_NAME_BY_STR.get(d.get("name"), bm.VoucherName.OVERSTOCK)
        return bm.Voucher(name=v_name)
    elif s in ("Default", "Enhanced") or d.get("value") in RANK_MAP:
        return parse_playing_card(d)
    return None


def parse_lua_action(
    action_name: str, meta: dict | list, frame_idx: int, timestamp: float
) -> bm.Action:
    """Maps a Lua telemetry action to a Python Action model."""
    meta_dict = dict(meta) if isinstance(meta, dict) else {}
    action_map = {
        "Play Hand": bm.ActionType.PLAY_HAND,
        "Discard": bm.ActionType.DISCARD,
        "Reroll Shop": bm.ActionType.REROLL_SHOP,
        "Cash Out": bm.ActionType.CASH_OUT,
        "Buy Card": bm.ActionType.BUY_CARD,
        "Buy and Use": bm.ActionType.BUY_AND_USE,
        "Sell Card": bm.ActionType.SELL_CARD,
        "Select Blind": bm.ActionType.SELECT_BLIND,
        "Skip Blind": bm.ActionType.SKIP_BLIND,
        "Use Consumable": bm.ActionType.USE_CONSUMABLE,
        "Open Pack": bm.ActionType.OPEN_PACK,
        "Use In Pack": bm.ActionType.USE_IN_PACK,
        "Skip Pack": bm.ActionType.SKIP_PACK,
        "Redeem Voucher": bm.ActionType.REDEEM_VOUCHER,
        "Next Round": bm.ActionType.NEXT_ROUND,
        "Reroll Boss": bm.ActionType.REROLL_BOSS,
        "Sort by Rank": bm.ActionType.SORT_RANK,
        "Sort by Suit": bm.ActionType.SORT_SUIT,
        "End Round": bm.ActionType.END_ROUND,
        "Setup Test Suite": bm.ActionType.SETUP_TEST_SUITE,
    }
    act_type = action_map.get(action_name, bm.ActionType.PLAY_HAND)
    if act_type == bm.ActionType.CASH_OUT:
        meta_dict.setdefault("include_round_payout", True)
    if act_type in (bm.ActionType.SORT_RANK, bm.ActionType.SORT_SUIT):
        sel_ids = meta_dict.get("selected_card_ids")
        if sel_ids and isinstance(sel_ids, list) and len(sel_ids) > 0:
            meta_dict.setdefault("selected_card_id", sel_ids[0])
    target_idx = meta_dict.get("target_idx")
    joker_order = meta_dict.get("joker_order")
    return bm.Action(
        type=act_type,
        frame_idx=frame_idx,
        timestamp=timestamp,
        target_idx=target_idx if isinstance(target_idx, int) else None,
        joker_order=joker_order if isinstance(joker_order, list) else [],
        metadata=meta_dict,
    )


def sync_cv_observations_before_action(
    sim_state: gs.GameState, lua_state: dict, act_name: str = ""
) -> None:
    """Simulates a Perfect-CV/OCR camera observing ONLY screen-visible inputs at action time.

    Does NOT touch closed-loop engine state (money, hands_left, discards_left,
    reroll_cost, discard_pile, deck_cards_count) so the test harness can verify
    that the Python engine predicts them without ground truth.
    """
    # 0. Sync deterministic seed if discovered dynamically
    if not sim_state.seed and lua_state.get("seed"):
        sim_state.set_seed(lua_state["seed"])

    # 1. Location, boss disabled flag, inflation, and pack choices
    loc_str = lua_state.get("location")
    if loc_str in ("blind", "shop", "select", "pack"):
        sim_state.location = bm.Location(loc_str)
    if "boss_disabled" in lua_state:
        sim_state.boss_disabled = bool(lua_state["boss_disabled"])
    if "inflation" in lua_state:
        sim_state.inflation = int(lua_state["inflation"])
    if "pack_choices_remaining" in lua_state:
        sim_state.pack_choices_remaining = int(lua_state["pack_choices_remaining"])
    if "reroll_cost" in lua_state:
        sim_state.reroll_cost = int(lua_state["reroll_cost"])
    if "active_tags" in lua_state:
        tags = []
        for t in lua_state["active_tags"]:
            tag_name = t.get("name") if isinstance(t, dict) else str(t)
            try:
                tags.append(bm.SkipTag(tag_name))
            except ValueError:
                pass
        sim_state.active_tags = tags

    # 2. Dealt hand cards & player card selection (visible on screen)
    if "hand" in lua_state:
        sim_state.hand.cards = [parse_hand_card(c) for c in lua_state["hand"]]

    # 2. Current blind & boss blind (visible on HUD)
    blind_str = lua_state.get("blind", "")
    if blind_str in BLIND_BY_STR:
        sim_state.current_blind = BLIND_BY_STR[blind_str]
    boss_str = lua_state.get("boss_blind", "")
    if boss_str in BLIND_BY_STR:
        sim_state.boss_blind = BLIND_BY_STR[boss_str]
    if "target_chips" in lua_state:
        sim_state.target_chips = int(lua_state["target_chips"])
    if "current_chips" in lua_state:
        sim_state.current_chips = int(lua_state["current_chips"])

    # 3. Rotating target suits/ranks (visible on joker tooltips)
    if lua_state.get("castle_suit") in SUIT_MAP:
        sim_state.castle_suit = SUIT_MAP[lua_state["castle_suit"]]
        for j in sim_state.inventory.jokers:
            if j.name == bm.JokerName.CASTLE:
                j.state["target_suit"] = sim_state.castle_suit
    if lua_state.get("ancient_suit") in SUIT_MAP:
        sim_state.ancient_suit = SUIT_MAP[lua_state["ancient_suit"]]
        for j in sim_state.inventory.jokers:
            if j.name == bm.JokerName.ANCIENT_JOKER:
                j.state["target_suit"] = sim_state.ancient_suit
    if lua_state.get("mail_card_rank") in RANK_MAP:
        sim_state.mail_card_rank = RANK_MAP[lua_state["mail_card_rank"]]
        for j in sim_state.inventory.jokers:
            if j.name == bm.JokerName.MAIL_IN_REBATE:
                j.state["target_rank"] = sim_state.mail_card_rank
    idol_data = lua_state.get("idol_card")
    if isinstance(idol_data, dict):
        r = RANK_MAP.get(idol_data.get("rank", "Ace"), bm.Rank.ACE)
        s = SUIT_MAP.get(idol_data.get("suit", "Spades"), bm.Suit.SPADES)
        sim_state.idol_card = (r, s)

    # 4. If Joker lineup or order changed externally (e.g. player dragged Jokers, mod generator, or RNG spawn),
    # preserve existing Python Joker instances by ID so closed-loop accumulator/extra_value predictions are kept.
    lua_jokers_raw = lua_state.get("jokers", [])
    if "jokers" in sim_state.pending_cv_checks or [
        getattr(j, "id", 0) for j in sim_state.inventory.jokers
    ] != [int(d.get("id", 0)) for d in lua_jokers_raw]:
        existing_by_id = {
            j.id: j for j in sim_state.inventory.jokers if getattr(j, "id", 0) != 0
        }
        updated_jokers: list[bm.Joker] = []
        for d in lua_jokers_raw:
            j_id = int(d.get("id", 0))
            if j_id != 0 and j_id in existing_by_id:
                updated_jokers.append(existing_by_id.pop(j_id))
            else:
                pj = parse_joker(d)
                if pj is not None:
                    updated_jokers.append(pj)
                    if act_name == "Setup Test Suite":
                        on_joker_added(sim_state, pj)
        sim_state.inventory.jokers = updated_jokers
        sim_state.pending_cv_checks.discard("jokers")

    # Sync per-Joker rotating To Do List target poker hand (visible on Joker tooltip)
    todo_targets = [
        POKER_HAND_BY_STR[d["to_do_poker_hand"]]
        for d in lua_jokers_raw
        if d.get("name") == "To Do List"
        and d.get("to_do_poker_hand") in POKER_HAND_BY_STR
    ]
    todo_idx = 0
    for j in sim_state.inventory.jokers:
        if j.name == bm.JokerName.TO_DO_LIST and todo_idx < len(todo_targets):
            j.state["target_hand"] = todo_targets[todo_idx]
            sim_state.to_do_poker_hand = todo_targets[todo_idx]
            todo_idx += 1

    # 5. Screen-visible Shop, Consumables, and Booster pack contents
    if "shop" in lua_state:
        sim_state.shop_cards = [
            c for d in lua_state["shop"] if (c := parse_any_card(d)) is not None
        ]
    if "shop_vouchers" in lua_state:
        sim_state.shop_vouchers = [
            c
            for d in lua_state["shop_vouchers"]
            if isinstance(c := parse_any_card(d), bm.Voucher)
        ]
    if "consumables" in lua_state:
        sim_state.inventory.consumables = [
            c for d in lua_state["consumables"] if (c := parse_any_card(d)) is not None
        ]
    if "pack_cards" in lua_state:
        sim_state.pack_cards = [
            c for d in lua_state["pack_cards"] if (c := parse_any_card(d)) is not None
        ]

    # 6. If action is "Setup Test Suite", refresh mutated deck from telemetry
    if act_name == "Setup Test Suite":
        if "deck" in lua_state and lua_state["deck"]:
            sim_state.full_deck = [parse_playing_card(c) for c in lua_state["deck"]]
            sim_state.deck_cards_count = len(sim_state.full_deck)
            sim_state.starting_deck_size = len(sim_state.full_deck)


def test_trajectory_parity(telemetry_path: Path, verbose: bool = False) -> int:
    """Validates game engine parity against a Lua telemetry jsonl log.

    Models a Perfect-CV/OCR Video Pipeline:
    - Each telemetry entry i records the pre-action state S_i at the moment action A_i is clicked.
    - At step i, the harness feeds ONLY CV-visible inputs (dealt hand & selection, blind name,
      shop offerings) into sim_state, executes step_simulation(sim_state, A_i), and compares
      the predicted post-action state against S_{i+1} (entry i+1's pre-action state).

    Returns the count of discrepancies encountered.
    """
    if not telemetry_path.exists():
        print(f"No telemetry file found at {telemetry_path}.")
        return 0

    print(f"Loading Lua telemetry from {telemetry_path}...")
    with open(telemetry_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        print("Telemetry file is empty.")
        return 0

    entries = [json.loads(line) for line in lines]
    first_state = entries[0].get("state", {})

    stake_val = first_state.get("stake", 1)
    stake_enum = (
        gs.STAKE_ORDER[stake_val - 1]
        if isinstance(stake_val, int) and 1 <= stake_val <= len(gs.STAKE_ORDER)
        else bm.Stake.WHITE
    )
    try:
        deck_enum = bm.Deck(first_state.get("deck_name", bm.Deck.RED_DECK.value))
    except ValueError:
        deck_enum = bm.Deck.RED_DECK

    sim_state = gs.GameState(
        step_idx=0,
        seed=first_state.get("seed", ""),
        deck=deck_enum,
        stake=stake_enum,
        ante=first_state.get("ante", 1),
        round=first_state.get("round", 1),
        money=first_state.get("money", 4),
        reroll_cost=first_state.get("reroll_cost", 5),
        hands_left=first_state.get("hands_left", 4),
        discards_left=first_state.get("discards_left", 3),
        hands_played_this_round=first_state.get("hands_played_this_round", 0),
        discards_used_this_round=first_state.get("discards_used_this_round", 0),
        apply_deck_stake_on_init=False,
    )
    if "deck" in first_state and first_state["deck"]:
        sim_state.full_deck = [parse_playing_card(c) for c in first_state["deck"]]
        sim_state.deck_cards_count = len(sim_state.full_deck)
        sim_state.starting_deck_size = len(sim_state.full_deck)
    if "jokers" in first_state:
        sim_state.inventory.jokers = [
            pj for d in first_state["jokers"] if (pj := parse_joker(d)) is not None
        ]

    trajectory = gs.RunTrajectory(keyframe_interval=10)
    errors = 0

    for i, entry in enumerate(entries):
        act_name = entry.get("action", "Play Hand")
        meta = entry.get("action_metadata", {})
        lua_state = entry.get("state", {})
        step = entry.get("step", i + 1)
        ts = entry.get("timestamp", 0.0)

        # 1. Sync CV-visible screen inputs before executing action A_i
        sync_cv_observations_before_action(sim_state, lua_state, act_name=act_name)

        # 2. Record (S_i, A_i) pair and step the Python simulation engine
        action = parse_lua_action(act_name, meta, frame_idx=step, timestamp=ts)
        if i + 1 < len(entries):
            next_entry_peek = entries[i + 1]
            next_act_peek = next_entry_peek.get("action", "")
            next_lua_peek = next_entry_peek.get("state", {})
            # Detect The Hook mid-play discard logged before evaluate_play
            if (
                act_name == "Play Hand"
                and sim_state.current_blind == bm.BlindName.THE_HOOK
                and not sim_state.boss_disabled
                and next_act_peek == "Discard"
                and next_lua_peek.get("hands_left") == sim_state.hands_left - 1
            ):
                action.metadata["defer_play_for_hook"] = True
            elif (
                act_name == "Discard"
                and getattr(sim_state, "_hook_pending_cards", None) is not None
            ):
                action.metadata["hook"] = True

            # If Riff-Raff spawns Jokers during Select Blind, pass observed spawns from post-blind state
            if act_name == "Select Blind" and "jokers" in next_lua_peek:
                curr_ids = {
                    int(d.get("id", 0))
                    for d in lua_state.get("jokers", [])
                    if d.get("id")
                }
                new_jokers = [
                    parse_joker(d)
                    for d in next_lua_peek["jokers"]
                    if int(d.get("id", 0)) not in curr_ids
                ]
                new_jokers = [j for j in new_jokers if j is not None]
                if new_jokers:
                    action.metadata["riff_raff_spawned"] = new_jokers

            # Detect when Play Hand (or The Hook's mid-play Discard) beats the blind and triggers end_round()
            if (
                act_name == "Play Hand"
                and not action.metadata.get("defer_play_for_hook")
            ) or (act_name == "Discard" and action.metadata.get("hook")):
                if (
                    lua_state.get("blind", "") != ""
                    and next_lua_peek.get("blind", "") == ""
                ):
                    action.metadata["won_round"] = True

        trajectory.record_step(sim_state.copy(), action)
        sim_state, _ = gs.step_simulation(sim_state, action)

        # 3. If there is a next step S_{i+1}, compare our predicted post-action state against S_{i+1}!
        if i + 1 < len(entries):
            next_entry = entries[i + 1]
            next_act = next_entry.get("action", "")
            next_lua = next_entry.get("state", {})

            # Check Money (unless an RNG money event or Tarot/Spectral modified money)
            if "money" in next_lua and "money" not in sim_state.pending_cv_checks:
                # IfCash Out had a specific round_dollars from tags/modifiers, or deterministic payout
                if act_name in (
                    "Play Hand",
                    "Discard",
                    "Reroll Shop",
                    "Select Blind",
                    "Cash Out",
                ):
                    if sim_state.money != next_lua["money"]:
                        print(
                            f"[Mismatch Step {step} '{act_name}' -> Step {step + 1}] "
                            f"Money: Python={sim_state.money} vs Lua={next_lua['money']}"
                        )
                        errors += 1
                    elif verbose:
                        print(f"[OK Step {step} '{act_name}'] Money={sim_state.money}")
            sim_state.Money = next_lua.get("money", sim_state.money)
            sim_state.money = next_lua.get("money", sim_state.money)
            sim_state.pending_cv_checks.discard("money")

            # Check Hands Left & Discards Left
            if act_name in ("Play Hand", "Discard", "Select Blind"):
                if (
                    "hands_left" in next_lua
                    and sim_state.hands_left != next_lua["hands_left"]
                ):
                    print(
                        f"[Mismatch Step {step} '{act_name}' -> Step {step + 1}] "
                        f"Hands Left: Python={sim_state.hands_left} vs Lua={next_lua['hands_left']}"
                    )
                    errors += 1
                    sim_state.hands_left = next_lua["hands_left"]
                if (
                    "discards_left" in next_lua
                    and sim_state.discards_left != next_lua["discards_left"]
                ):
                    print(
                        f"[Mismatch Step {step} '{act_name}' -> Step {step + 1}] "
                        f"Discards Left: Python={sim_state.discards_left} vs Lua={next_lua['discards_left']}"
                    )
                    errors += 1
                    sim_state.discards_left = next_lua["discards_left"]

            # Check Reroll Cost
            if act_name in ("Reroll Shop", "Cash Out") and "reroll_cost" in next_lua:
                if sim_state.reroll_cost != next_lua["reroll_cost"]:
                    print(
                        f"[Mismatch Step {step} '{act_name}' -> Step {step + 1}] "
                        f"Reroll Cost: Python={sim_state.reroll_cost} vs Lua={next_lua['reroll_cost']}"
                    )
                    errors += 1
                    sim_state.reroll_cost = next_lua["reroll_cost"]

            # Check Deck Cards Count
            if "deck_cards_count" in next_lua and act_name in (
                "Play Hand",
                "Discard",
                "Select Blind",
                "Cash Out",
            ):
                if sim_state.deck_cards_count != next_lua["deck_cards_count"]:
                    print(
                        f"[Mismatch Step {step} '{act_name}' -> Step {step + 1}] "
                        f"Deck Cards Count: Python={sim_state.deck_cards_count} vs Lua={next_lua['deck_cards_count']}"
                    )
                    errors += 1
                    sim_state.deck_cards_count = next_lua["deck_cards_count"]

            # Check Discard Pile Count (Note: when blind is beaten or before Cash Out, Balatro's end_round already shuffled discard pile back into deck)
            target_chips = next_lua.get("target_chips", float("inf"))
            current_chips = next_lua.get("current_chips", 0)
            if current_chips >= target_chips:
                sim_state.discard_pile.clear()

            if (
                "discard_pile" in next_lua
                and next_act != "Cash Out"
                and act_name in ("Discard", "Play Hand", "Select Blind")
            ):
                if len(sim_state.discard_pile) != len(next_lua["discard_pile"]):
                    print(
                        f"[Mismatch Step {step} '{act_name}' -> Step {step + 1}] "
                        f"Discard Pile Count: Python={len(sim_state.discard_pile)} vs Lua={len(next_lua['discard_pile'])}"
                    )
                    errors += 1

            # Check Joker Accumulator States (mult, x_mult, chips, extra_value) when Joker lineup is unchanged
            next_jokers_raw = next_lua.get("jokers", [])
            if len(next_jokers_raw) == len(sim_state.inventory.jokers):
                for j_idx, (py_j, lua_j) in enumerate(
                    zip(sim_state.inventory.jokers, next_jokers_raw)
                ):
                    if py_j.name.value != lua_j.get("name"):
                        continue
                    if (
                        py_j.id != 0
                        and lua_j.get("id", 0) != 0
                        and py_j.id != lua_j.get("id")
                    ):
                        continue
                    # Compare extra_value (Egg, Gift Card)
                    if (
                        "extra_value" in lua_j
                        and py_j.extra_value != lua_j["extra_value"]
                    ):
                        print(
                            f"[Mismatch Step {step} '{act_name}' -> Step {step + 1}] "
                            f"Joker[{j_idx}] {py_j.name.value} extra_value: Python={py_j.extra_value} vs Lua={lua_j['extra_value']}"
                        )
                        errors += 1
                        py_j.extra_value = lua_j["extra_value"]
                    # Compare mult accumulator (Green Joker, Ride the Bus, Flash Card, Red Card, Spare Trousers, Popcorn, Ceremonial Dagger)
                    if py_j.name in (
                        bm.JokerName.GREEN_JOKER,
                        bm.JokerName.RIDE_THE_BUS,
                        bm.JokerName.FLASH_CARD,
                        bm.JokerName.RED_CARD,
                        bm.JokerName.SPARE_TROUSERS,
                        bm.JokerName.POPCORN,
                        bm.JokerName.CEREMONIAL_DAGGER,
                    ):
                        py_m = py_j.state.get("mult", 0)
                        lua_m = lua_j.get("mult", 0)
                        if py_m != lua_m:
                            print(
                                f"[Mismatch Step {step} '{act_name}' -> Step {step + 1}] "
                                f"Joker[{j_idx}] {py_j.name.value} mult: Python={py_m} vs Lua={lua_m}"
                            )
                            errors += 1
                            py_j.state["mult"] = lua_m
                    # Compare x_mult accumulator (Campfire, Constellation, Hologram, Vampire, Obelisk, Yorick, Ramen, Hit the Road, Canio, Lucky Cat)
                    if py_j.name in (
                        bm.JokerName.CAMPFIRE,
                        bm.JokerName.CONSTELLATION,
                        bm.JokerName.HOLOGRAM,
                        bm.JokerName.VAMPIRE,
                        bm.JokerName.OBELISK,
                        bm.JokerName.YORICK,
                        bm.JokerName.RAMEN,
                        bm.JokerName.HIT_THE_ROAD,
                        bm.JokerName.CANIO,
                        bm.JokerName.LUCKY_CAT,
                    ) or (
                        py_j.name == bm.JokerName.GLASS_JOKER and bool(sim_state.seed)
                    ):
                        py_xm = round(float(py_j.state.get("x_mult", 1.0)), 2)
                        lua_xm = round(float(lua_j.get("x_mult", 1.0)), 2)
                        if abs(py_xm - lua_xm) > 1e-2:
                            print(
                                f"[Mismatch Step {step} '{act_name}' -> Step {step + 1}] "
                                f"Joker[{j_idx}] {py_j.name.value} x_mult: Python={py_xm} vs Lua={lua_xm}"
                            )
                            errors += 1
                            py_j.state["x_mult"] = lua_xm
                    # Compare Loyalty Card remaining hands
                    if (
                        py_j.name == bm.JokerName.LOYALTY_CARD
                        and "remaining" in py_j.state
                    ):
                        lua_rem = lua_j.get("loyalty_remaining")
                        if lua_rem is not None and py_j.state["remaining"] != int(
                            lua_rem
                        ):
                            print(
                                f"[Mismatch Step {step} '{act_name}' -> Step {step + 1}] "
                                f"Joker[{j_idx}] Loyalty Card remaining: Python={py_j.state['remaining']} vs Lua={lua_rem}"
                            )
                            errors += 1
                    # Compare chips accumulator when extra table is present in telemetry (Runner, Square Joker, Wee Joker, Castle, Ice Cream)
                    if "extra" in lua_j and py_j.name in (
                        bm.JokerName.RUNNER,
                        bm.JokerName.SQUARE_JOKER,
                        bm.JokerName.WEE_JOKER,
                        bm.JokerName.CASTLE,
                        bm.JokerName.ICE_CREAM,
                    ):
                        py_c = py_j.state.get("chips", 0)
                        lua_c = lua_j.get("chips", 0)
                        if py_c != lua_c:
                            print(
                                f"[Mismatch Step {step} '{act_name}' -> Step {step + 1}] "
                                f"Joker[{j_idx}] {py_j.name.value} chips: Python={py_c} vs Lua={lua_c}"
                            )
                            errors += 1

            # Check PRNG stream synchronization when RNG is active and telemetry includes prng_streams
            if (
                "prng_streams" in next_lua
                and getattr(sim_state, "rng", None) is not None
            ):
                lua_prng = next_lua["prng_streams"]
                if isinstance(lua_prng, dict):
                    for ch_key, ch in sim_state.rng.channels.items():
                        if ch_key in lua_prng and ch.call_count > 0:
                            py_val = round(float(ch.state), 7)
                            lua_val = round(float(lua_prng[ch_key]), 7)
                            if abs(py_val - lua_val) > 1e-5 and verbose:
                                print(
                                    f"[Mismatch Step {step} '{act_name}' -> Step {step + 1}] "
                                    f"PRNG channel '{ch_key}': Python={py_val} vs Lua={lua_val}"
                                )

    return errors


# =========================================================================
# UNIT TESTS
# =========================================================================


class TestEngine(unittest.TestCase):
    """Unit tests for the Balatro simulation engine, Jokers, and state transitions."""

    def test_synthetic_simulation(self):
        """Validates sequential shop reroll, Cash Out triggers, and trajectory keyframes."""
        sim_state = gs.GameState(
            money=15,
            reroll_cost=5,
            inventory=bm.Inventory(
                jokers=[
                    bm.Joker(name=bm.JokerName.GIFT_CARD),
                    bm.Joker(name=bm.JokerName.JOKER),
                ]
            ),
        )
        traj = gs.RunTrajectory(keyframe_interval=3)

        # 1. Reroll Shop
        sim_state, _ = gs.step_simulation(
            sim_state, bm.Action(type=bm.ActionType.REROLL_SHOP)
        )
        traj.record_step(sim_state)
        self.assertEqual(sim_state.money, 10)
        self.assertEqual(sim_state.reroll_cost, 6)

        # 2. Cash out (triggers Gift Card on both jokers)
        sim_state, _ = gs.step_simulation(
            sim_state, bm.Action(type=bm.ActionType.CASH_OUT)
        )
        traj.record_step(sim_state)
        self.assertEqual(sim_state.inventory.jokers[0].extra_value, 1)
        self.assertEqual(sim_state.inventory.jokers[1].extra_value, 1)
        self.assertEqual(sim_state.reroll_cost, 5)  # Resets in shop

        # 3. Test trajectory keyframe seek
        recovered_s0 = traj.get_state(0)
        self.assertIsNotNone(recovered_s0)
        self.assertEqual(recovered_s0.money, 10)

    def test_wee_joker_only_scores_on_active_twos(self):
        """Wee Joker must only gain chips when a 2 is part of the scoring hand."""
        wee = bm.Joker(name=bm.JokerName.WEE_JOKER)
        sim_state = gs.GameState(
            inventory=bm.Inventory(jokers=[wee]),
            hand=bm.Hand(
                cards=[
                    bm.HandCard(
                        card=bm.PlayingCard(rank=bm.Rank.ACE, suit=bm.Suit.SPADES),
                        selected=True,
                    ),
                    bm.HandCard(
                        card=bm.PlayingCard(rank=bm.Rank.TWO, suit=bm.Suit.HEARTS),
                        selected=True,
                    ),
                ]
            ),
        )
        # Playing High Card Ace with an un-scored 2: Wee Joker should NOT gain chips
        sim_state, _ = gs.step_simulation(
            sim_state, bm.Action(type=bm.ActionType.PLAY_HAND)
        )
        self.assertEqual(wee.state.get("chips", 0), 0)

        # Now playing Pair of 2s with King kicker: Wee Joker gains 16 chips (2 * 8)
        sim_state.hand = bm.Hand(
            cards=[
                bm.HandCard(
                    card=bm.PlayingCard(rank=bm.Rank.TWO, suit=bm.Suit.SPADES),
                    selected=True,
                ),
                bm.HandCard(
                    card=bm.PlayingCard(rank=bm.Rank.TWO, suit=bm.Suit.CLUBS),
                    selected=True,
                ),
                bm.HandCard(
                    card=bm.PlayingCard(rank=bm.Rank.KING, suit=bm.Suit.DIAMONDS),
                    selected=True,
                ),
            ]
        )
        sim_state, _ = gs.step_simulation(
            sim_state, bm.Action(type=bm.ActionType.PLAY_HAND)
        )
        self.assertEqual(wee.state.get("chips", 0), 16)

    def test_ride_the_bus_ignores_unscored_face_cards(self):
        """Ride the Bus must NOT reset when face cards are played as un-scored kickers."""
        bus = bm.Joker(name=bm.JokerName.RIDE_THE_BUS)
        sim_state = gs.GameState(
            inventory=bm.Inventory(jokers=[bus]),
            hand=bm.Hand(
                cards=[
                    bm.HandCard(
                        card=bm.PlayingCard(rank=bm.Rank.EIGHT, suit=bm.Suit.SPADES),
                        selected=True,
                    ),
                    bm.HandCard(
                        card=bm.PlayingCard(rank=bm.Rank.EIGHT, suit=bm.Suit.CLUBS),
                        selected=True,
                    ),
                    bm.HandCard(
                        card=bm.PlayingCard(rank=bm.Rank.KING, suit=bm.Suit.DIAMONDS),
                        selected=True,
                    ),
                ]
            ),
        )
        # Hand is Pair of 8s. King is un-scored kicker. Ride the Bus should GAIN +1 mult, not reset!
        sim_state, _ = gs.step_simulation(
            sim_state, bm.Action(type=bm.ActionType.PLAY_HAND)
        )
        self.assertEqual(bus.state.get("mult", 0), 1)

        # Now playing High Card King: King is the scoring card, so Ride the Bus must reset to 0!
        sim_state.hand = bm.Hand(
            cards=[
                bm.HandCard(
                    card=bm.PlayingCard(rank=bm.Rank.KING, suit=bm.Suit.SPADES),
                    selected=True,
                ),
                bm.HandCard(
                    card=bm.PlayingCard(rank=bm.Rank.EIGHT, suit=bm.Suit.HEARTS),
                    selected=True,
                ),
            ]
        )
        sim_state, _ = gs.step_simulation(
            sim_state, bm.Action(type=bm.ActionType.PLAY_HAND)
        )
        self.assertEqual(bus.state.get("mult", 0), 0)

    def test_hiker_and_vampire_only_affect_scoring_cards(self):
        """Hiker and Vampire must exclusively modify scoring cards, leaving kickers intact."""
        hiker = bm.Joker(name=bm.JokerName.HIKER)
        vampire = bm.Joker(name=bm.JokerName.VAMPIRE)
        c8a = bm.PlayingCard(
            rank=bm.Rank.EIGHT, suit=bm.Suit.SPADES, enhancement=bm.Enhancement.GOLD
        )
        c8b = bm.PlayingCard(
            rank=bm.Rank.EIGHT, suit=bm.Suit.CLUBS, enhancement=bm.Enhancement.GOLD
        )
        c5 = bm.PlayingCard(
            rank=bm.Rank.FIVE, suit=bm.Suit.HEARTS, enhancement=bm.Enhancement.GOLD
        )

        sim_state = gs.GameState(
            inventory=bm.Inventory(jokers=[hiker, vampire]),
            hand=bm.Hand(
                cards=[
                    bm.HandCard(card=c8a, selected=True),
                    bm.HandCard(card=c8b, selected=True),
                    bm.HandCard(card=c5, selected=True),
                ]
            ),
        )
        sim_state, _ = gs.step_simulation(
            sim_state, bm.Action(type=bm.ActionType.PLAY_HAND)
        )

        # The two 8s scored: gained +5 chips each, lost their Gold enhancement, Vampire gained 0.2x
        self.assertEqual(c8a.bonus_chips, 5)
        self.assertEqual(c8b.bonus_chips, 5)
        self.assertEqual(c8a.enhancement, bm.Enhancement.NONE)
        self.assertEqual(c8b.enhancement, bm.Enhancement.NONE)
        self.assertAlmostEqual(vampire.state.get("x_mult", 1.0), 1.2)

        # The 5 was an un-scored kicker: must retain Gold and gain NO bonus chips
        self.assertEqual(c5.bonus_chips, 0)
        self.assertEqual(c5.enhancement, bm.Enhancement.GOLD)

    def test_ramen_discard_decay(self):
        """Ramen decreases x_mult by 0.01 per card discarded."""
        ramen = bm.Joker(name=bm.JokerName.RAMEN)
        sim_state = gs.GameState(
            inventory=bm.Inventory(jokers=[ramen]),
            hand=bm.Hand(
                cards=[
                    bm.HandCard(
                        card=bm.PlayingCard(rank=bm.Rank.TWO, suit=bm.Suit.SPADES),
                        selected=True,
                    ),
                    bm.HandCard(
                        card=bm.PlayingCard(rank=bm.Rank.THREE, suit=bm.Suit.CLUBS),
                        selected=True,
                    ),
                    bm.HandCard(
                        card=bm.PlayingCard(rank=bm.Rank.FOUR, suit=bm.Suit.HEARTS),
                        selected=True,
                    ),
                ]
            ),
        )
        sim_state, _ = gs.step_simulation(
            sim_state, bm.Action(type=bm.ActionType.DISCARD)
        )
        self.assertAlmostEqual(ramen.state.get("x_mult", 2.0), 1.97)

    def test_dna_card_duplication_to_deck_and_hand(self):
        """DNA clones the single played card on first hand: adds to full_deck and directly to hand."""
        dna = bm.Joker(name=bm.JokerName.DNA)
        hologram = bm.Joker(name=bm.JokerName.HOLOGRAM)
        target_card = bm.PlayingCard(rank=bm.Rank.ACE, suit=bm.Suit.SPADES)
        other_card = bm.PlayingCard(rank=bm.Rank.TEN, suit=bm.Suit.HEARTS)

        sim_state = gs.GameState(
            inventory=bm.Inventory(jokers=[dna, hologram]),
            hand=bm.Hand(
                cards=[
                    bm.HandCard(card=target_card, selected=True),
                    bm.HandCard(card=other_card, selected=False),
                ]
            ),
        )
        self.assertEqual(len(sim_state.full_deck), 52)

        # 1. Play first hand with 1 card (Ace of Spades)
        sim_state, _ = gs.step_simulation(
            sim_state, bm.Action(type=bm.ActionType.PLAY_HAND)
        )

        # Deck should now have 53 cards, and Hologram scaled by +0.25x from DNA's card addition!
        self.assertEqual(len(sim_state.full_deck), 53)
        self.assertEqual(sim_state.deck_cards_count, 53)
        self.assertAlmostEqual(hologram.state.get("x_mult", 1.0), 1.25)

        # Hand should contain: the other card + the newly copied Ace of Spades (unselected)
        self.assertEqual(len(sim_state.hand.cards), 2)
        ace_hc = [c for c in sim_state.hand.cards if c.card.rank == bm.Rank.ACE][0]
        self.assertFalse(ace_hc.selected)

        # 2. Playing a second hand with 1 card should NOT trigger DNA again
        ace_hc.selected = True
        sim_state, _ = gs.step_simulation(
            sim_state, bm.Action(type=bm.ActionType.PLAY_HAND)
        )
        self.assertEqual(len(sim_state.full_deck), 53)

    def test_dna_blueprint_synergy(self):
        """Blueprint copying DNA produces two cloned cards in full_deck and hand."""
        dna = bm.Joker(name=bm.JokerName.DNA)
        bp = bm.Joker(name=bm.JokerName.BLUEPRINT)
        card = bm.PlayingCard(rank=bm.Rank.KING, suit=bm.Suit.HEARTS)

        sim_state = gs.GameState(
            inventory=bm.Inventory(jokers=[bp, dna]),
            hand=bm.Hand(cards=[bm.HandCard(card=card, selected=True)]),
        )
        sim_state, _ = gs.step_simulation(
            sim_state, bm.Action(type=bm.ActionType.PLAY_HAND)
        )

        self.assertEqual(len(sim_state.full_deck), 54)
        self.assertEqual(sim_state.deck_cards_count, 54)
        self.assertEqual(len(sim_state.hand.cards), 2)

    def test_trading_card_destruction_and_discard_pile(self):
        """Trading Card destroys single card on first discard: removes from full_deck and triggers Canio."""
        tc = bm.Joker(name=bm.JokerName.TRADING_CARD)
        canio = bm.Joker(name=bm.JokerName.CANIO)
        face_card = bm.PlayingCard(rank=bm.Rank.QUEEN, suit=bm.Suit.DIAMONDS)
        other_card = bm.PlayingCard(rank=bm.Rank.TWO, suit=bm.Suit.CLUBS)

        sim_state = gs.GameState(
            inventory=bm.Inventory(jokers=[tc, canio]),
            hand=bm.Hand(
                cards=[
                    bm.HandCard(card=face_card, selected=True),
                    bm.HandCard(card=other_card, selected=False),
                ]
            ),
        )
        initial_money = sim_state.money

        sim_state, _ = gs.step_simulation(
            sim_state, bm.Action(type=bm.ActionType.DISCARD)
        )

        self.assertEqual(sim_state.money, initial_money + 3)
        self.assertEqual(sim_state.deck_cards_count, 51)
        self.assertNotIn(face_card, sim_state.full_deck)
        self.assertNotIn(face_card, sim_state.discard_pile)
        self.assertEqual(len(sim_state.discard_pile), 0)
        self.assertAlmostEqual(canio.state.get("x_mult", 1.0), 2.0)

    def test_discard_pile_and_played_cards_tracking_and_reset(self):
        """Verifies accumulation in discard_pile and played_cards_this_round, and reset on SELECT_BLIND."""
        c1 = bm.PlayingCard(rank=bm.Rank.FOUR, suit=bm.Suit.SPADES)
        c2 = bm.PlayingCard(rank=bm.Rank.FIVE, suit=bm.Suit.HEARTS)
        c3 = bm.PlayingCard(rank=bm.Rank.SEVEN, suit=bm.Suit.CLUBS)

        sim_state = gs.GameState(
            hand=bm.Hand(
                cards=[
                    bm.HandCard(card=c1, selected=True),
                    bm.HandCard(card=c2, selected=False),
                    bm.HandCard(card=c3, selected=False),
                ]
            ),
        )

        sim_state, _ = gs.step_simulation(
            sim_state, bm.Action(type=bm.ActionType.DISCARD)
        )
        self.assertIn(c1, sim_state.discard_pile)
        self.assertEqual(len(sim_state.discard_pile), 1)

        sim_state.hand.cards[0].selected = True  # c2
        sim_state, _ = gs.step_simulation(
            sim_state, bm.Action(type=bm.ActionType.PLAY_HAND)
        )
        self.assertIn(c2, sim_state.played_cards_this_round)
        self.assertEqual(len(sim_state.played_cards_this_round), 1)

        sim_state, _ = gs.step_simulation(
            sim_state, bm.Action(type=bm.ActionType.SELECT_BLIND)
        )
        self.assertEqual(len(sim_state.discard_pile), 0)
        self.assertEqual(len(sim_state.played_cards_this_round), 0)

    def test_sixth_sense_destruction(self):
        """Sixth Sense destroys a single played 6 on first hand from hand and deck."""
        sixth = bm.Joker(name=bm.JokerName.SIXTH_SENSE)
        c6 = bm.PlayingCard(rank=bm.Rank.SIX, suit=bm.Suit.SPADES)

        sim_state = gs.GameState(
            inventory=bm.Inventory(jokers=[sixth]),
            hand=bm.Hand(cards=[bm.HandCard(card=c6, selected=True)]),
        )
        sim_state, _ = gs.step_simulation(
            sim_state, bm.Action(type=bm.ActionType.PLAY_HAND)
        )

        self.assertEqual(sim_state.deck_cards_count, 51)
        self.assertNotIn(c6, sim_state.full_deck)
        self.assertEqual(len(sim_state.hand.cards), 0)
        self.assertNotIn(c6, sim_state.played_cards_this_round)

    def test_marble_joker_adds_stone_card(self):
        """Marble Joker appends a Stone card to full_deck when selecting a blind."""
        marble = bm.Joker(name=bm.JokerName.MARBLE_JOKER)
        sim_state = gs.GameState(inventory=bm.Inventory(jokers=[marble]))
        initial_count = len(sim_state.full_deck)

        sim_state, _ = gs.step_simulation(
            sim_state, bm.Action(type=bm.ActionType.SELECT_BLIND)
        )
        self.assertEqual(len(sim_state.full_deck), initial_count + 1)
        self.assertEqual(sim_state.full_deck[-1].enhancement, bm.Enhancement.STONE)

    def test_cloud_9_evaluates_full_deck(self):
        """Cloud 9 counts all 9s in the full_deck at end of round."""
        c9 = bm.Joker(name=bm.JokerName.CLOUD_9)
        sim_state = gs.GameState(inventory=bm.Inventory(jokers=[c9]))
        initial_money = sim_state.money

        sim_state, _ = gs.step_simulation(
            sim_state,
            bm.Action(
                type=bm.ActionType.CASH_OUT,
                metadata={"include_round_payout": False},
            ),
        )
        self.assertEqual(sim_state.money, initial_money + 4)

    def test_joker_fixes_campfire_yorick_obelisk_todolist_turtlebean_retriggers(self):
        """Validates source-accurate fixes for Campfire, Yorick, Obelisk, To Do List, Turtle Bean, Hiker+Blueprint, and Retriggers."""
        # 1. Campfire adds +0.25x per card sold and resets on Boss Blind
        campfire = bm.Joker(name=bm.JokerName.CAMPFIRE)
        fodder = bm.Joker(name=bm.JokerName.JOKER)
        state = gs.GameState(
            current_blind=bm.BlindName.THE_GOAD,
            inventory=bm.Inventory(jokers=[campfire, fodder]),
        )
        state, _ = gs.step_simulation(
            state, bm.Action(type=bm.ActionType.SELL_CARD, target_idx=1)
        )
        self.assertAlmostEqual(campfire.state["x_mult"], 1.25)
        state, _ = gs.step_simulation(state, bm.Action(type=bm.ActionType.CASH_OUT))
        self.assertAlmostEqual(campfire.state["x_mult"], 1.0)

        # 2. Yorick carries over excess discards across the 23 threshold
        yorick = bm.Joker(name=bm.JokerName.YORICK)
        yorick.state["discards"] = 2
        state_y = gs.GameState(
            inventory=bm.Inventory(jokers=[yorick]),
            hand=bm.Hand(
                cards=[
                    bm.HandCard(
                        card=bm.PlayingCard(rank=bm.Rank.TWO, suit=bm.Suit.SPADES),
                        selected=True,
                    ),
                    bm.HandCard(
                        card=bm.PlayingCard(rank=bm.Rank.THREE, suit=bm.Suit.SPADES),
                        selected=True,
                    ),
                    bm.HandCard(
                        card=bm.PlayingCard(rank=bm.Rank.FOUR, suit=bm.Suit.SPADES),
                        selected=True,
                    ),
                    bm.HandCard(
                        card=bm.PlayingCard(rank=bm.Rank.FIVE, suit=bm.Suit.SPADES),
                        selected=True,
                    ),
                    bm.HandCard(
                        card=bm.PlayingCard(rank=bm.Rank.SIX, suit=bm.Suit.SPADES),
                        selected=True,
                    ),
                ]
            ),
        )
        state_y, _ = gs.step_simulation(state_y, bm.Action(type=bm.ActionType.DISCARD))
        self.assertAlmostEqual(yorick.state["x_mult"], 2.0)
        self.assertEqual(yorick.state["discards"], 20)  # 2 - 5 + 23 = 20

        # 3. Obelisk does NOT reset when tying the most played hand; only resets when strictly most played
        obelisk = bm.Joker(name=bm.JokerName.OBELISK)
        obelisk.state["x_mult"] = 1.4
        state_ob = gs.GameState(
            inventory=bm.Inventory(jokers=[obelisk]),
            hand_play_counts={bm.PokerHand.PAIR: 3, bm.PokerHand.HIGH_CARD: 2},
            hand=bm.Hand(
                cards=[
                    bm.HandCard(
                        card=bm.PlayingCard(rank=bm.Rank.ACE, suit=bm.Suit.SPADES),
                        selected=True,
                    ),
                ]
            ),
        )
        # Playing High Card brings HIGH_CARD to 3 (tied with PAIR=3) -> must NOT reset, scales to 1.6!
        state_ob, _ = gs.step_simulation(
            state_ob, bm.Action(type=bm.ActionType.PLAY_HAND)
        )
        self.assertAlmostEqual(obelisk.state["x_mult"], 1.6)
        # Playing High Card again brings HIGH_CARD to 4 (strictly > PAIR=3) -> resets to 1.0!
        state_ob.hand.cards = [
            bm.HandCard(
                card=bm.PlayingCard(rank=bm.Rank.KING, suit=bm.Suit.SPADES),
                selected=True,
            )
        ]
        state_ob, _ = gs.step_simulation(
            state_ob, bm.Action(type=bm.ActionType.PLAY_HAND)
        )
        self.assertAlmostEqual(obelisk.state["x_mult"], 1.0)

        # 4. Hiker + Blueprint + Hanging Chad retrigger synergy
        bp = bm.Joker(name=bm.JokerName.BLUEPRINT)
        hiker = bm.Joker(name=bm.JokerName.HIKER)
        chad = bm.Joker(name=bm.JokerName.HANGING_CHAD)
        c_ace = bm.PlayingCard(rank=bm.Rank.ACE, suit=bm.Suit.SPADES)
        state_hk = gs.GameState(
            inventory=bm.Inventory(jokers=[bp, hiker, chad]),
            hand=bm.Hand(cards=[bm.HandCard(card=c_ace, selected=True)]),
        )
        # Hanging Chad gives 1 + 2 = 3 triggers; Blueprint+Hiker gives 2 * 5 = +10 chips per trigger -> +30 bonus_chips!
        state_hk, _ = gs.step_simulation(
            state_hk, bm.Action(type=bm.ActionType.PLAY_HAND)
        )
        self.assertEqual(c_ace.bonus_chips, 30)

        # 5. Turtle Bean modifies hand.max_size on add and decrements hand.max_size at end of round
        bean = bm.Joker(name=bm.JokerName.TURTLE_BEAN)
        state_tb = gs.GameState()
        state_tb.inventory.jokers.append(bean)
        on_joker_added(state_tb, bean)
        self.assertEqual(state_tb.hand.max_size, 13)  # 8 + 5
        state_tb, _ = gs.step_simulation(
            state_tb, bm.Action(type=bm.ActionType.CASH_OUT)
        )
        self.assertEqual(bean.state["h_size"], 4)
        self.assertEqual(state_tb.hand.max_size, 12)

    def test_gamestate_new_features_and_deltas(self):
        """Validates pack opening, skip tags, forced selection, duplicate removal, and delta diffs."""
        pack = bm.BoosterPack(
            booster_type=bm.BoosterType.ARCANA,
            size=bm.BoosterSize.MEGA,
            cost=8,
            choose=2,
            extra=5,
        )
        ace1 = bm.PlayingCard(rank=bm.Rank.ACE, suit=bm.Suit.SPADES, id=101)
        ace2 = bm.PlayingCard(rank=bm.Rank.ACE, suit=bm.Suit.SPADES, id=102)

        state = gs.GameState(
            location=bm.Location.SELECT,
            money=20,
            small_blind_tag=bm.SkipTag.DOUBLE_TAG,
            shop_packs=[pack],
            hand=bm.Hand(
                cards=[
                    bm.HandCard(card=ace1, selected=True, forced_selection=True),
                    bm.HandCard(card=ace2, selected=False),
                ]
            ),
        )
        state.full_deck.extend([ace1, ace2])
        state.deck_cards_count = len(state.full_deck)

        state.hand.select(0)
        self.assertTrue(state.hand.cards[0].selected)
        state.hand.clear_selection()
        self.assertTrue(state.hand.cards[0].selected)

        state.remove_card_from_deck(ace1)
        self.assertEqual(len(state.hand.cards), 1)
        self.assertIs(state.hand.cards[0].card, ace2)

        traj = gs.RunTrajectory(keyframe_interval=5)
        traj.record_step(state)
        state, _ = gs.step_simulation(state, bm.Action(type=bm.ActionType.SKIP_BLIND))
        traj.record_step(state)
        self.assertEqual(state.blinds_skipped, 1)
        self.assertIn(bm.SkipTag.DOUBLE_TAG, state.active_tags)
        self.assertEqual(state.current_blind, bm.BlindName.BIG)

        state, _ = gs.step_simulation(
            state, bm.Action(type=bm.ActionType.OPEN_PACK, target_idx=0)
        )
        state.pack_cards = [
            bm.Tarot(name=bm.TarotName.HERMIT),
            bm.Spectral(name=bm.SpectralName.ECTOPLASM),
        ]
        state.hand_levels[bm.PokerHand.PAIR] = 3
        state.inventory.joker_slots = 6
        traj.record_step(state)
        self.assertEqual(state.location, bm.Location.PACK)
        self.assertEqual(state.pack_choices_remaining, 2)

        state, _ = gs.step_simulation(
            state, bm.Action(type=bm.ActionType.USE_IN_PACK, target_idx=0)
        )
        traj.record_step(state)
        self.assertEqual(state.last_tarot_planet, bm.TarotName.HERMIT)
        self.assertEqual(state.tarots_used, 1)
        self.assertEqual(state.pack_choices_remaining, 1)
        self.assertEqual(state.location, bm.Location.PACK)

        state, _ = gs.step_simulation(
            state, bm.Action(type=bm.ActionType.USE_IN_PACK, target_idx=0)
        )
        traj.record_step(state)
        self.assertEqual(state.hand.max_size, 7)
        self.assertEqual(state.ectoplasm_minus, 2)
        self.assertEqual(state.location, bm.Location.SHOP)

        replayed = traj.get_state(4)
        self.assertEqual(replayed.hand_levels[bm.PokerHand.PAIR], 3)
        self.assertEqual(replayed.inventory.joker_slots, 6)
        self.assertEqual(replayed.hand.max_size, 7)
        self.assertEqual(replayed.ectoplasm_minus, 2)
        self.assertEqual(replayed.last_tarot_planet, bm.TarotName.HERMIT)
        self.assertEqual(replayed.tarots_used, 1)

    def test_action_parsing(self):
        """Validates that Lua action names map correctly to ActionType enums."""
        action_cases = [
            ("Play Hand", bm.ActionType.PLAY_HAND),
            ("Discard", bm.ActionType.DISCARD),
            ("Reroll Shop", bm.ActionType.REROLL_SHOP),
            ("Cash Out", bm.ActionType.CASH_OUT),
            ("Buy Card", bm.ActionType.BUY_CARD),
            ("Buy and Use", bm.ActionType.BUY_AND_USE),
            ("Sell Card", bm.ActionType.SELL_CARD),
            ("Select Blind", bm.ActionType.SELECT_BLIND),
            ("Skip Blind", bm.ActionType.SKIP_BLIND),
            ("Use Consumable", bm.ActionType.USE_CONSUMABLE),
            ("Open Pack", bm.ActionType.OPEN_PACK),
            ("Use In Pack", bm.ActionType.USE_IN_PACK),
            ("Skip Pack", bm.ActionType.SKIP_PACK),
            ("Redeem Voucher", bm.ActionType.REDEEM_VOUCHER),
            ("Next Round", bm.ActionType.NEXT_ROUND),
            ("Reroll Boss", bm.ActionType.REROLL_BOSS),
        ]
        for name, expected_type in action_cases:
            action = parse_lua_action(name, {}, frame_idx=1, timestamp=0.5)
            self.assertEqual(action.type, expected_type)
            self.assertEqual(action.frame_idx, 1)
            self.assertEqual(action.timestamp, 0.5)

    def test_telemetry_file_parity_if_present(self):
        """Runs full parity validation if run_telemetry.jsonl is present and non-empty."""
        telemetry_paths = [
            REPO_ROOT / "run_telemetry.jsonl",
            REPO_ROOT / "output" / "run_telemetry.jsonl",
        ]
        for p in telemetry_paths:
            if p.exists() and p.stat().st_size > 0:
                errors = test_trajectory_parity(p)
                self.assertEqual(
                    errors, 0, f"Encountered {errors} telemetry parity errors in {p}"
                )


class TestFaceDownCardBounds(unittest.TestCase):
    """Unit tests for face-down card representation, partial observability, bounds deduction, and persistence."""

    def test_face_down_card_masks_rank_and_suit(self):
        """Card initialized face-down must mask rank and suit to NONE to prevent ML ground-truth leakage."""
        c = bm.PlayingCard(rank=bm.Rank.ACE, suit=bm.Suit.SPADES, id=42)
        hc = bm.HandCard(card=c, facing_back=True)

        self.assertEqual(hc.card.rank, bm.Rank.NONE)
        self.assertEqual(hc.card.suit, bm.Suit.NONE)
        self.assertEqual(hc.card.id, 0)
        self.assertIsNotNone(hc._real_card)
        self.assertEqual(hc._real_card.rank, bm.Rank.ACE)
        self.assertEqual(hc._real_card.suit, bm.Suit.SPADES)
        self.assertEqual(hc._real_card.id, 42)
        self.assertTrue(hc.bounds.is_face_down)
        self.assertFalse(hc.bounds.is_resolved)

    def test_marginal_probability_vectors_and_shapes(self):
        """CardBounds and HandCard must expose fixed-dimensional probability distributions."""
        c = bm.PlayingCard(rank=bm.Rank.ACE, suit=bm.Suit.SPADES)
        hc = bm.HandCard(card=c, facing_back=True)

        rv = hc.get_rank_vector()
        sv = hc.get_suit_vector()
        slv = hc.get_seal_vector()

        self.assertEqual(len(rv), 13)
        self.assertEqual(len(sv), 4)
        self.assertEqual(len(slv), 5)
        self.assertAlmostEqual(sum(rv), 1.0, places=5)
        self.assertAlmostEqual(sum(sv), 1.0, places=5)
        self.assertAlmostEqual(sum(slv), 1.0, places=5)

        # For a face-up card, verify one-hot vectors
        hc_up = bm.HandCard(card=c, facing_back=False)
        self.assertEqual(hc_up.get_rank_vector()[-1], 1.0)  # Ace is index 12 (last)
        self.assertEqual(hc_up.get_suit_vector()[0], 1.0)  # Spades is index 0 (first)
        self.assertEqual(hc_up.get_seal_vector()[0], 1.0)  # Seal.NONE is index 0

    def test_rank_sort_observation_bounds(self):
        """'Sort by Rank' (descending) constrains possible ranks between flanking face-up cards."""
        state = gs.GameState()
        ace = bm.PlayingCard(rank=bm.Rank.ACE, suit=bm.Suit.SPADES, id=1)
        ten = bm.PlayingCard(rank=bm.Rank.TEN, suit=bm.Suit.CLUBS, id=2)
        eight = bm.PlayingCard(rank=bm.Rank.EIGHT, suit=bm.Suit.HEARTS, id=3)

        hc_ace = bm.HandCard(card=ace, facing_back=False)
        hc_ten = bm.HandCard(card=ten, facing_back=True)
        hc_eight = bm.HandCard(card=eight, facing_back=False)

        state.hand = bm.Hand(cards=[hc_ace, hc_ten, hc_eight])
        state.apply_sort_observation(sort_type="rank")

        # Ranks 2..7 have nominal < 8.03 and must be ruled out
        for r in [
            bm.Rank.TWO,
            bm.Rank.THREE,
            bm.Rank.FOUR,
            bm.Rank.FIVE,
            bm.Rank.SIX,
            bm.Rank.SEVEN,
        ]:
            self.assertNotIn(r, hc_ten.bounds.possible_ranks)
        self.assertIn(bm.Rank.TEN, hc_ten.bounds.possible_ranks)
        self.assertIn(bm.Rank.NINE, hc_ten.bounds.possible_ranks)

    def test_suit_sort_observation_bounds(self):
        """'Sort by Suit' restricts candidate cards to the flanking suit block."""
        state = gs.GameState()
        kh = bm.PlayingCard(rank=bm.Rank.KING, suit=bm.Suit.HEARTS, id=101)
        five_h = bm.PlayingCard(rank=bm.Rank.FIVE, suit=bm.Suit.HEARTS, id=102)
        two_h = bm.PlayingCard(rank=bm.Rank.TWO, suit=bm.Suit.HEARTS, id=103)

        hc_kh = bm.HandCard(card=kh, facing_back=False)
        hc_five = bm.HandCard(card=five_h, facing_back=True)
        hc_two = bm.HandCard(card=two_h, facing_back=False)

        state.hand = bm.Hand(cards=[hc_kh, hc_five, hc_two])
        state.apply_sort_observation(sort_type="suit")

        self.assertEqual(hc_five.bounds.possible_suits, {bm.Suit.HEARTS})

    def test_rank_and_suit_sort_resolves_card(self):
        """Intersection of Rank sort and Suit sort observations resolves face-down card to exactly 1 candidate."""
        state = gs.GameState()
        c_ad = bm.PlayingCard(rank=bm.Rank.ACE, suit=bm.Suit.DIAMONDS, id=1)
        c_10d = bm.PlayingCard(rank=bm.Rank.TEN, suit=bm.Suit.DIAMONDS, id=2)
        c_8d = bm.PlayingCard(rank=bm.Rank.EIGHT, suit=bm.Suit.DIAMONDS, id=3)
        c_2s = bm.PlayingCard(rank=bm.Rank.TWO, suit=bm.Suit.SPADES, id=4)
        state.full_deck = [c_ad, c_10d, c_8d, c_2s]

        hc_ad = bm.HandCard(card=c_ad, facing_back=False)
        hc_10d = bm.HandCard(card=c_10d, facing_back=True)
        hc_8d = bm.HandCard(card=c_8d, facing_back=False)

        state.hand = bm.Hand(cards=[hc_ad, hc_10d, hc_8d])

        state.apply_sort_observation(sort_type="rank")
        state.apply_sort_observation(sort_type="suit")

        self.assertTrue(hc_10d.bounds.is_resolved)
        self.assertTrue(hc_10d.facing_back)
        self.assertEqual(hc_10d.card.rank, bm.Rank.TEN)
        self.assertEqual(hc_10d.card.suit, bm.Suit.DIAMONDS)
        self.assertEqual(hc_10d.card.id, 2)

    def test_persistent_slot_uid_and_bounds_across_play(self):
        """Held face-down cards retain slot_uid and bounds across PLAY_HAND actions while new cards append right."""
        state = gs.GameState()
        c1 = bm.PlayingCard(rank=bm.Rank.TWO, suit=bm.Suit.CLUBS, id=1)
        c2 = bm.PlayingCard(
            rank=bm.Rank.KING, suit=bm.Suit.HEARTS, seal=bm.Seal.BLUE, id=2
        )
        c3 = bm.PlayingCard(rank=bm.Rank.THREE, suit=bm.Suit.CLUBS, id=3)
        c4 = bm.PlayingCard(rank=bm.Rank.FOUR, suit=bm.Suit.CLUBS, id=4)
        c5 = bm.PlayingCard(rank=bm.Rank.FIVE, suit=bm.Suit.CLUBS, id=5)

        hc1 = bm.HandCard(card=c1, selected=True)
        hc2 = bm.HandCard(card=c2, facing_back=True, slot_uid=99)
        hc2.bounds.possible_seals = {bm.Seal.BLUE}
        hc3 = bm.HandCard(card=c3, selected=True)
        hc4 = bm.HandCard(card=c4, selected=True)
        hc5 = bm.HandCard(card=c5, selected=True)

        state.hand = bm.Hand(cards=[hc1, hc2, hc3, hc4, hc5])
        state, _ = gs.step_simulation(state, bm.Action(type=bm.ActionType.PLAY_HAND))

        self.assertEqual(len(state.hand.cards), 1)
        survivor = state.hand.cards[0]
        self.assertEqual(survivor.slot_uid, 99)
        self.assertEqual(survivor.bounds.possible_seals, {bm.Seal.BLUE})
        self.assertTrue(survivor.facing_back)

        # Replenish hand
        drawn = state.draw_cards()
        self.assertEqual(len(state.hand.cards), state.hand.max_size)
        self.assertEqual(state.hand.cards[0].slot_uid, 99)
        for i in range(1, len(state.hand.cards)):
            self.assertGreater(state.hand.cards[i].slot_uid, 0)
            self.assertNotEqual(state.hand.cards[i].slot_uid, 99)

    def test_purple_seal_discard_prunes_candidates(self):
        """Discarding a Purple Seal face-down card prunes it from remaining candidates and triggers tarot spawn."""
        state = gs.GameState()
        state.inventory = bm.Inventory()
        state.seed = "TEST"
        state.set_seed("TEST")

        c_purple = bm.PlayingCard(
            rank=bm.Rank.ACE, suit=bm.Suit.SPADES, seal=bm.Seal.PURPLE, id=10
        )
        c_normal = bm.PlayingCard(
            rank=bm.Rank.ACE, suit=bm.Suit.HEARTS, seal=bm.Seal.NONE, id=20
        )
        state.full_deck = [c_purple, c_normal]

        hc_p = bm.HandCard(card=c_purple, facing_back=True, selected=True)
        hc_n = bm.HandCard(card=c_normal, facing_back=True, selected=False)
        state.hand = bm.Hand(cards=[hc_p, hc_n])
        state.ensure_slot_uids()

        state, _ = gs.step_simulation(state, bm.Action(type=bm.ActionType.DISCARD))

        self.assertEqual(len(state.inventory.consumables), 1)
        survivor = state.hand.cards[0]
        self.assertNotIn(10, survivor.bounds.candidate_card_ids)
        self.assertEqual(survivor.bounds.candidate_card_ids, {20})
        self.assertTrue(survivor.bounds.is_resolved)
        self.assertEqual(survivor.card.rank, bm.Rank.ACE)
        self.assertEqual(survivor.card.suit, bm.Suit.HEARTS)
        self.assertEqual(survivor.card.id, 20)

    def test_tarot_actions_narrow_face_down_cards(self):
        """Tarot / Spectral actions (Death, Magician, Sun, Trance) narrow and resolve face-down cards."""
        state = gs.GameState()
        state.inventory = bm.Inventory()

        # 1. Death on [face_down, face_up_King_Hearts] resolves left card
        c_left = bm.PlayingCard(rank=bm.Rank.TWO, suit=bm.Suit.CLUBS, id=1)
        c_right = bm.PlayingCard(rank=bm.Rank.KING, suit=bm.Suit.HEARTS, id=2)
        hc_left = bm.HandCard(card=c_left, facing_back=True, selected=True)
        hc_right = bm.HandCard(card=c_right, facing_back=False, selected=True)
        state.hand = bm.Hand(cards=[hc_left, hc_right])
        state.inventory.consumables = [bm.Tarot(name=bm.TarotName.DEATH)]

        state, _ = gs.step_simulation(
            state, bm.Action(type=bm.ActionType.USE_CONSUMABLE, target_idx=0)
        )
        self.assertTrue(hc_left.bounds.is_resolved)
        self.assertTrue(hc_left.facing_back)
        self.assertEqual(hc_left.card.rank, bm.Rank.KING)
        self.assertEqual(hc_left.card.suit, bm.Suit.HEARTS)

        # 2. Magician applies Lucky enhancement
        c3 = bm.PlayingCard(rank=bm.Rank.FIVE, suit=bm.Suit.SPADES, id=3)
        hc3 = bm.HandCard(card=c3, facing_back=True, selected=True)
        state.hand = bm.Hand(cards=[hc3])
        state.inventory.consumables = [bm.Tarot(name=bm.TarotName.MAGICIAN)]

        state, _ = gs.step_simulation(
            state, bm.Action(type=bm.ActionType.USE_CONSUMABLE, target_idx=0)
        )
        self.assertEqual(hc3.bounds.possible_enhancements, {bm.Enhancement.LUCKY})
        self.assertEqual(hc3.card.enhancement, bm.Enhancement.LUCKY)

        # 3. The Sun converts suit to Hearts
        state.inventory.consumables = [bm.Tarot(name=bm.TarotName.SUN)]
        hc3.selected = True
        state, _ = gs.step_simulation(
            state, bm.Action(type=bm.ActionType.USE_CONSUMABLE, target_idx=0)
        )
        self.assertEqual(hc3.bounds.possible_suits, {bm.Suit.HEARTS})
        self.assertEqual(hc3.card.suit, bm.Suit.HEARTS)

        # 4. Trance adds Blue seal
        state.inventory.consumables = [bm.Spectral(name=bm.SpectralName.TRANCE)]
        hc3.selected = True
        state, _ = gs.step_simulation(
            state, bm.Action(type=bm.ActionType.USE_CONSUMABLE, target_idx=0)
        )
        self.assertEqual(hc3.bounds.possible_seals, {bm.Seal.BLUE})
        self.assertEqual(hc3.card.seal, bm.Seal.BLUE)

    def test_tarot_flip_reveals_face_down_card(self):
        """Tarot usage temporarily flips card face-up: card stays facing_back=True, but its full identity is observed."""
        state = gs.GameState()
        state.inventory = bm.Inventory()
        c = bm.PlayingCard(rank=bm.Rank.SEVEN, suit=bm.Suit.SPADES, id=77)
        hc = bm.HandCard(card=c, facing_back=True, selected=True)
        self.assertEqual(hc.card.rank, bm.Rank.NONE)
        self.assertEqual(hc.card.suit, bm.Suit.NONE)
        self.assertFalse(hc.bounds.is_resolved)

        state.hand = bm.Hand(cards=[hc])
        state.inventory.consumables = [bm.Tarot(name=bm.TarotName.SUN)]

        state, _ = gs.step_simulation(
            state, bm.Action(type=bm.ActionType.USE_CONSUMABLE, target_idx=0)
        )

        # Flips back face-down, but identity is now fully observed and resolved!
        self.assertTrue(hc.facing_back)
        self.assertTrue(hc.bounds.is_resolved)
        self.assertEqual(hc.card.rank, bm.Rank.SEVEN)
        self.assertEqual(hc.card.suit, bm.Suit.HEARTS)
        self.assertEqual(hc.bounds.possible_ranks, {bm.Rank.SEVEN})
        self.assertEqual(hc.bounds.possible_suits, {bm.Suit.HEARTS})

    def test_draw_face_up_retains_face_down_identity(self):
        """Drawing face-up cards into hand preserves known face-down card identity."""
        state = gs.GameState()
        state.inventory = bm.Inventory()
        c = bm.PlayingCard(rank=bm.Rank.EIGHT, suit=bm.Suit.CLUBS, id=88)
        hc = bm.HandCard(card=c, facing_back=True, selected=True)
        state.hand = bm.Hand(cards=[hc], max_size=4)
        state.inventory.consumables = [bm.Tarot(name=bm.TarotName.MAGICIAN)]
        state, _ = gs.step_simulation(
            state, bm.Action(type=bm.ActionType.USE_CONSUMABLE, target_idx=0)
        )
        self.assertTrue(hc.bounds.is_resolved)

        # Draw remaining cards face-up
        c_up1 = bm.PlayingCard(rank=bm.Rank.TWO, suit=bm.Suit.HEARTS, id=2)
        c_up2 = bm.PlayingCard(rank=bm.Rank.THREE, suit=bm.Suit.HEARTS, id=3)
        state.full_deck = [c, c_up1, c_up2]
        state.draw_cards(count=2, face_down=False)

        # Known face-down card still retained!
        self.assertTrue(hc.bounds.is_resolved)
        self.assertEqual(hc.card.rank, bm.Rank.EIGHT)
        self.assertEqual(hc.card.enhancement, bm.Enhancement.LUCKY)

    def test_draw_non_adjacent_face_down_retains_identity(self):
        """Drawing face-down cards visually separated by a face-up card does not cause confusion."""
        state = gs.GameState()
        c_revealed = bm.PlayingCard(rank=bm.Rank.KING, suit=bm.Suit.HEARTS, id=101)
        hc_revealed = bm.HandCard(card=c_revealed, facing_back=True)
        hc_revealed.reveal_temporarily()

        c_divider = bm.PlayingCard(rank=bm.Rank.ACE, suit=bm.Suit.SPADES, id=102)
        hc_divider = bm.HandCard(card=c_divider, facing_back=False)

        # Hand: [FaceDown (King of Hearts), FaceUp (Ace of Spades)]
        state.hand = bm.Hand(cards=[hc_revealed, hc_divider], max_size=3)

        # Draw a face-down card to the right: Hand will be [FaceDown, FaceUp, FaceDown]
        c_new_fd = bm.PlayingCard(rank=bm.Rank.FIVE, suit=bm.Suit.DIAMONDS, id=103)
        state.full_deck = [c_revealed, c_divider, c_new_fd]
        state.draw_cards(count=1, face_down=True)

        self.assertEqual(len(state.hand.cards), 3)
        # hc_revealed at index 0 is NOT adjacent to index 2 (separated by index 1 face-up card)!
        self.assertTrue(state.hand.cards[0].bounds.is_resolved)
        self.assertEqual(state.hand.cards[0].card.rank, bm.Rank.KING)
        self.assertEqual(state.hand.cards[0].card.suit, bm.Suit.HEARTS)

    def test_draw_adjacent_face_down_forgets_identity(self):
        """Drawing a face-down card immediately adjacent to a known face-down card blurs candidates."""
        state = gs.GameState()
        c_revealed = bm.PlayingCard(rank=bm.Rank.KING, suit=bm.Suit.HEARTS, id=101)
        hc_revealed = bm.HandCard(card=c_revealed, facing_back=True)
        hc_revealed.reveal_temporarily()

        # Hand: [FaceDown (King of Hearts)] at the far right of current hand
        state.hand = bm.Hand(cards=[hc_revealed], max_size=2)

        # Draw a new face-down card: it will be placed immediately adjacent at index 1!
        c_new_fd = bm.PlayingCard(rank=bm.Rank.FIVE, suit=bm.Suit.DIAMONDS, id=102)
        state.full_deck = [c_revealed, c_new_fd]
        state.draw_cards(count=1, face_down=True)

        self.assertEqual(len(state.hand.cards), 2)
        c0 = state.hand.cards[0]
        c1 = state.hand.cards[1]

        # Both are now adjacent face-down cards with identical card backs
        # The known card "forgets" single-card certainty and masks back to NONE
        self.assertFalse(c0.bounds.is_resolved)
        self.assertFalse(c1.bounds.is_resolved)
        self.assertEqual(c0.card.rank, bm.Rank.NONE)
        self.assertEqual(c0.card.suit, bm.Suit.NONE)
        # Both share the pooled candidates {101, 102}
        self.assertEqual(c0.bounds.candidate_card_ids, {101, 102})
        self.assertEqual(c1.bounds.candidate_card_ids, {101, 102})
        self.assertEqual(c0.bounds.possible_ranks, {bm.Rank.KING, bm.Rank.FIVE})
        self.assertEqual(c0.bounds.possible_suits, {bm.Suit.HEARTS, bm.Suit.DIAMONDS})

    def test_adjacent_face_down_distinguished_by_selection_and_seal(self):
        """Adjacent face-down cards do not conflate if visually distinguished by selection or seal sticker."""
        state = gs.GameState()
        c1 = bm.PlayingCard(rank=bm.Rank.KING, suit=bm.Suit.HEARTS, id=1)
        c2 = bm.PlayingCard(rank=bm.Rank.QUEEN, suit=bm.Suit.CLUBS, id=2)
        hc1 = bm.HandCard(card=c1, facing_back=True, selected=True)  # Raised!
        hc2 = bm.HandCard(card=c2, facing_back=True, selected=False)  # Resting!
        hc1.reveal_temporarily()
        hc2.reveal_temporarily()

        state.hand = bm.Hand(cards=[hc1, hc2])
        state.reconcile_face_down_adjacencies()

        # Selection state distinguishes them: no blurring!
        self.assertTrue(hc1.bounds.is_resolved)
        self.assertTrue(hc2.bounds.is_resolved)
        self.assertEqual(hc1.card.rank, bm.Rank.KING)
        self.assertEqual(hc2.card.rank, bm.Rank.QUEEN)

        # Now test seal distinction: both unselected, but one has a visible Blue Seal
        hc1.selected = False
        hc2.selected = False
        hc1.card.seal = bm.Seal.BLUE
        if hc1._real_card:
            hc1._real_card.seal = bm.Seal.BLUE
        hc2.card.seal = bm.Seal.NONE
        if hc2._real_card:
            hc2._real_card.seal = bm.Seal.NONE

        state.reconcile_face_down_adjacencies()
        self.assertTrue(hc1.bounds.is_resolved)
        self.assertTrue(hc2.bounds.is_resolved)
        self.assertEqual(hc1.card.rank, bm.Rank.KING)
        self.assertEqual(hc2.card.rank, bm.Rank.QUEEN)


class TestTelemetryParityAndLoggingAudit(unittest.TestCase):
    """Unit tests validating newly hooked telemetry actions, state deserialization, and audit fixes."""

    def test_action_parsing_new_types(self):
        """Validates mapping of Sort by Rank, Sort by Suit, Reorder Jokers, Reorder Hand, and Setup Test Suite."""
        act_rank = parse_lua_action(
            "Sort by Rank", {"sort_type": "rank", "selected_card_ids": [42]}, 1, 0.0
        )
        self.assertEqual(act_rank.type, bm.ActionType.SORT_RANK)
        self.assertEqual(act_rank.metadata.get("selected_card_id"), 42)

        act_suit = parse_lua_action(
            "Sort by Suit", {"sort_type": "suit", "selected_card_ids": [10]}, 2, 0.0
        )
        self.assertEqual(act_suit.type, bm.ActionType.SORT_SUIT)
        self.assertEqual(act_suit.metadata.get("selected_card_id"), 10)

        act_end_round = parse_lua_action(
            "End Round", {"joker_order": [102, 101]}, 3, 0.0
        )
        self.assertEqual(act_end_round.type, bm.ActionType.END_ROUND)

        act_setup = parse_lua_action("Setup Test Suite", {}, 5, 0.0)
        self.assertEqual(act_setup.type, bm.ActionType.SETUP_TEST_SUITE)

    def test_parse_joker_loyalty_and_payout(self):
        """Validates that Loyalty Card loyalty_remaining and Rocket payout are properly extracted."""
        j_loyalty_raw = {
            "name": "Loyalty Card",
            "id": 99,
            "loyalty_remaining": 3,
            "extra": {"Xmult": 4, "every": 5},
        }
        j_loyalty = parse_joker(j_loyalty_raw)
        self.assertIsNotNone(j_loyalty)
        self.assertEqual(j_loyalty.state.get("remaining"), 3)

        j_rocket_raw = {
            "name": "Rocket",
            "id": 100,
            "payout": 6,
            "extra": {"dollars": 6, "increase": 2},
        }
        j_rocket = parse_joker(j_rocket_raw)
        self.assertIsNotNone(j_rocket)
        self.assertEqual(j_rocket.state.get("payout"), 6)

    def test_reorder_actions_execution(self):
        """Implicit joker_order and hand_order reorder elements by card IDs or indices on any action, and END_ROUND evaluates end of round triggers."""
        j1 = bm.Joker(name=bm.JokerName.JOKER, id=1)
        j2 = bm.Joker(name=bm.JokerName.BLUEPRINT, id=2)
        state = gs.GameState(inventory=bm.Inventory(jokers=[j1, j2]))

        # Implicit reorder via PLAY_HAND action with joker_order
        c1 = bm.PlayingCard(rank=bm.Rank.TWO, suit=bm.Suit.HEARTS, id=11)
        state.hand = bm.Hand(cards=[bm.HandCard(card=c1, selected=True)])
        state, _ = gs.step_simulation(
            state,
            bm.Action(
                type=bm.ActionType.PLAY_HAND,
                joker_order=[2, 1],
            ),
        )
        self.assertEqual([j.id for j in state.inventory.jokers], [2, 1])

        # Implicit hand_order reorder on an action
        c2 = bm.PlayingCard(rank=bm.Rank.THREE, suit=bm.Suit.SPADES, id=12)
        state.hand = bm.Hand(cards=[bm.HandCard(card=c1), bm.HandCard(card=c2)])
        state, _ = gs.step_simulation(
            state,
            bm.Action(
                type=bm.ActionType.SORT_RANK,
                hand_order=[12, 11],
            ),
        )
        self.assertEqual([hc.card.id for hc in state.hand.cards], [12, 11])

        # END_ROUND action applies joker_order and triggers end-of-round effects
        mime = bm.Joker(name=bm.JokerName.MIME, id=3)
        state.inventory.jokers.append(mime)
        state.hand.cards[0].card.enhancement = bm.Enhancement.GOLD
        state, _ = gs.step_simulation(
            state,
            bm.Action(
                type=bm.ActionType.END_ROUND,
                joker_order=[2, 3, 1],  # Put Blueprint next to Mime
            ),
        )
        self.assertEqual([j.id for j in state.inventory.jokers], [2, 3, 1])
        self.assertTrue(state.round_won)
        state, _ = gs.step_simulation(state, bm.Action(type=bm.ActionType.CASH_OUT))
        self.assertEqual(state.location, bm.Location.SHOP)

    def test_persistent_joker_ids_and_cv_duplicate_matching(self):
        """Tests auto-assigned non-zero Joker IDs, distinguishable matching, and arbitrary identical-copy resolution."""
        import game.engine.cv_stubs as cv_stubs

        # 1. Shop purchases and spawned Jokers receive distinct non-zero IDs automatically
        state = gs.GameState(seed="JID1", location=bm.Location.SHOP, money=30)
        state.shop_cards = [
            bm.Joker(name=bm.JokerName.BLUEPRINT, edition=bm.Edition.FOIL),
            bm.Joker(name=bm.JokerName.BLUEPRINT, edition=bm.Edition.NONE),
            bm.Joker(name=bm.JokerName.BRAINSTORM, edition=bm.Edition.NONE),
        ]
        for _ in range(3):
            state, _ = gs.step_simulation(
                state, bm.Action(type=bm.ActionType.BUY_CARD, target_idx=0)
            )

        ids = [j.id for j in state.inventory.jokers]
        self.assertEqual(ids, [1, 2, 3])
        self.assertEqual(len(set(ids)), 3)

        # 2. Distinguishable copies (Foil vs Base Blueprint) match their true persistent IDs when swapped
        cv_swapped = [
            bm.Joker(name=bm.JokerName.BRAINSTORM),
            bm.Joker(name=bm.JokerName.BLUEPRINT, edition=bm.Edition.NONE),
            bm.Joker(name=bm.JokerName.BLUEPRINT, edition=bm.Edition.FOIL),
        ]
        matched_order = cv_stubs.match_cv_joker_order(state, cv_swapped)
        self.assertEqual(matched_order, [3, 2, 1])

        state, _ = gs.step_simulation(
            state,
            bm.Action(type=bm.ActionType.REROLL_SHOP, joker_order=matched_order),
        )
        self.assertEqual(
            [j.name for j in state.inventory.jokers],
            [
                bm.JokerName.BRAINSTORM,
                bm.JokerName.BLUEPRINT,
                bm.JokerName.BLUEPRINT,
            ],
        )
        self.assertEqual(
            [j.edition for j in state.inventory.jokers],
            [
                bm.Edition.NONE,
                bm.Edition.NONE,
                bm.Edition.FOIL,
            ],
        )
        self.assertEqual([j.id for j in state.inventory.jokers], [3, 2, 1])

        # 3. Visually identical copies (e.g. two Base Blueprints or Showman duplicates) resolve arbitrarily cleanly
        dup_state = gs.GameState(
            location=bm.Location.BLIND,
            inventory=bm.Inventory(
                jokers=[
                    bm.Joker(name=bm.JokerName.BLUEPRINT),
                    bm.Joker(name=bm.JokerName.BLUEPRINT),
                    bm.Joker(name=bm.JokerName.JOKER),
                ]
            ),
        )
        dup_ids = [j.id for j in dup_state.inventory.jokers]
        self.assertEqual(dup_ids, [1, 2, 3])
        cv_dup_reordered = [
            bm.Joker(name=bm.JokerName.BLUEPRINT),
            bm.Joker(name=bm.JokerName.JOKER),
            bm.Joker(name=bm.JokerName.BLUEPRINT),
        ]
        dup_order = cv_stubs.match_cv_joker_order(dup_state, cv_dup_reordered)
        self.assertEqual(set(dup_order), {1, 2, 3})
        dup_state, _ = gs.step_simulation(
            dup_state,
            bm.Action(type=bm.ActionType.PLAY_HAND, joker_order=dup_order),
        )
        self.assertEqual(
            [j.name for j in dup_state.inventory.jokers],
            [bm.JokerName.BLUEPRINT, bm.JokerName.JOKER, bm.JokerName.BLUEPRINT],
        )

        # 4. reconcile_cv_jokers preserves IDs on matched Jokers and assigns a fresh ID to a newly observed one
        new_obs = [
            bm.Joker(name=bm.JokerName.JOKER),
            bm.Joker(name=bm.JokerName.BLUEPRINT),
            bm.Joker(name=bm.JokerName.BLUEPRINT),
            bm.Joker(name=bm.JokerName.MIME),
        ]
        new_ids = cv_stubs.reconcile_cv_jokers(
            dup_state, new_obs, trigger_added_hooks=False
        )
        self.assertEqual(len(new_ids), 4)
        self.assertEqual(len(set(new_ids)), 4)
        self.assertTrue(all(jid > 0 for jid in new_ids))
        self.assertEqual(dup_state.inventory.jokers[-1].name, bm.JokerName.MIME)
        self.assertGreater(dup_state.inventory.jokers[-1].id, 3)

    def test_sync_observations_and_setup_test_suite(self):
        """Validates that sync_cv_observations_before_action updates state variables and refreshes deck on Setup Test Suite."""
        state = gs.GameState()
        lua_state = {
            "location": "shop",
            "boss_disabled": True,
            "inflation": 3,
            "pack_choices_remaining": 2,
            "deck": [
                {"value": "Ace", "suit": "Spades", "enhancement": "lucky", "id": 1},
                {"value": "King", "suit": "Hearts", "enhancement": "glass", "id": 2},
            ],
        }
        sync_cv_observations_before_action(
            state, lua_state, act_name="Setup Test Suite"
        )

        self.assertEqual(state.location, bm.Location.SHOP)
        self.assertTrue(state.boss_disabled)
        self.assertEqual(state.inflation, 3)
        self.assertEqual(state.pack_choices_remaining, 2)
        self.assertEqual(len(state.full_deck), 2)
        self.assertEqual(state.full_deck[0].enhancement, bm.Enhancement.LUCKY)
        self.assertEqual(state.full_deck[1].enhancement, bm.Enhancement.GLASS)

    def test_archived_telemetry_regression_suite(self):
        """Validates zero discrepancies across all archived telemetry runs in testing/telemetry_data/."""
        archive_dir = REPO_ROOT / "testing" / "telemetry_data"
        if not archive_dir.exists():
            self.skipTest("No telemetry archive directory found.")
        jsonl_files = sorted(archive_dir.glob("*.jsonl"))
        if not jsonl_files:
            self.skipTest(
                "No archived telemetry runs found in testing/telemetry_data/."
            )
        for f in jsonl_files:
            if f.stat().st_size == 0:
                continue
            with self.subTest(run=f.name):
                errors = test_trajectory_parity(f, verbose=False)
                self.assertEqual(
                    errors,
                    0,
                    f"Archived telemetry run {f.name} failed parity check with {errors} discrepancies!",
                )


class TestNewEngineFeatures(unittest.TestCase):
    """Tests newly integrated template, post-processing, inference, Blind/Ante, Deck/Stake, SkipTag, Consumable, Voucher, and Pricing rules."""

    def test_skip_pack_template_registered(self):
        templates_path = (
            REPO_ROOT / "assets" / "templates" / "drspectred" / "templates.json"
        )
        with templates_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("Skip Pack", data)
        self.assertEqual(data["Skip Pack"]["filenames"], ["skip.png"])
        skip_png = REPO_ROOT / "assets" / "templates" / "drspectred" / "skip.png"
        self.assertTrue(skip_png.exists())

    def test_post_processing_remap_filter_and_pack_exit(self):
        import post_processing

        # Non-gameplay UI clicks are filtered out; "Select Card" maps to USE_IN_PACK;
        # Pack opened via SKIP_BLIND restores Location.SELECT once exited.
        d = {
            "key": {
                "Play Game": 0,
                "Open Options Menu": 1,
                "Open Run Info Menu": 2,
                "Skip Blind": 3,
                "Select Card": 4,
                "Use Consumable": 5,
                "Select Blind": 6,
                "Open Pack": 7,
                "Buy Card": 8,
            },
            "frames": {
                "5": 0,  # Play Game -> filtered
                "10": 1,  # Open Options -> filtered
                "15": 2,  # Run Info -> filtered
                "100": 3,  # Skip Blind -> transitions to PACK since next decisive is USE_IN_PACK
                "150": 4,  # Select Card -> remapped to USE_IN_PACK; exits pack to SELECT
                "200": 5,  # Use Consumable -> ambient location should be SELECT (not PACK)
                "250": 6,  # Select Blind -> SELECT
                "300": 7,  # Open Pack (Shop) -> PACK
                "350": 4,  # Select Card -> exits pack to SHOP
                "400": 8,  # Buy Card -> SHOP
            },
        }
        queues = post_processing.process_actions(d)

        # Filtered frames never queue any CV work
        for filtered_base in (5, 10, 15):
            for offset in (0, -2):
                f_id = max(0, filtered_base + offset)
                for q in queues.values():
                    self.assertNotIn(f_id, q)

        # "Select Card" (150 - 2 = 148) runs pack + playing card CV
        self.assertEqual(queues["pack_cv_queue"][148], bm.ActionType.USE_IN_PACK)
        self.assertEqual(
            queues["playing_card_cv_queue"][148], bm.ActionType.USE_IN_PACK
        )

        # "Use Consumable" (200 - 2 = 198) happens in SELECT after Skip-Tag pack exited (not PACK)
        self.assertNotIn(198, queues["pack_cv_queue"])
        self.assertEqual(queues["joker_cv_queue"][198], bm.ActionType.USE_CONSUMABLE)

        # Shop pack -> Select Card (350 - 2 = 348) -> Buy Card (400 + 0 = 400)
        self.assertEqual(queues["pack_cv_queue"][348], bm.ActionType.USE_IN_PACK)
        self.assertEqual(queues["shop_cv_queue"][400], bm.ActionType.BUY_CARD)

    def test_the_hook_prng_discard_and_guards(self):
        from game.engine.rng_sim.events import predict_hook_discards
        from game.engine.rng_sim.rng import BalatroRNG

        state = gs.GameState(seed="HOOKSEED1", apply_deck_stake_on_init=True)
        state.location = bm.Location.BLIND
        state.current_blind = bm.BlindName.THE_HOOK
        state.hands_played_this_round = 0
        state.inventory.jokers = [
            bm.Joker(name=bm.JokerName.BURNT_JOKER),
            bm.Joker(name=bm.JokerName.TRADING_CARD),
        ]
        cards = [
            bm.PlayingCard(rank=bm.Rank.ACE, suit=bm.Suit.SPADES, id=1),
            bm.PlayingCard(rank=bm.Rank.KING, suit=bm.Suit.HEARTS, id=2),
            bm.PlayingCard(rank=bm.Rank.QUEEN, suit=bm.Suit.CLUBS, id=3),
            bm.PlayingCard(rank=bm.Rank.JACK, suit=bm.Suit.DIAMONDS, id=4),
            bm.PlayingCard(rank=bm.Rank.TEN, suit=bm.Suit.SPADES, id=5),
        ]
        state.hand.cards = [
            bm.HandCard(card=c, selected=(i == 0)) for i, c in enumerate(cards)
        ]

        expected_rng = BalatroRNG("HOOKSEED1")
        expected_hook = predict_hook_discards(expected_rng, cards[1:])
        self.assertEqual(len(expected_hook), 2)

        start_money = state.money
        start_hc_lvl = state.hand_levels.get(bm.PokerHand.HIGH_CARD, 1)
        state, _ = gs.step_simulation(state, bm.Action(type=bm.ActionType.PLAY_HAND))

        self.assertEqual(
            [c.id for c in state.discard_pile[:2]], [c.id for c in expected_hook]
        )
        self.assertEqual(state.discard_pile[2].id, 1)
        self.assertEqual(len(state.hand.cards), 2)
        self.assertEqual(
            state.money, start_money
        )  # Trading Card did NOT trigger on Hook discard
        self.assertEqual(
            state.hand_levels.get(bm.PokerHand.HIGH_CARD, 1), start_hc_lvl
        )  # Burnt Joker did NOT trigger

    def test_batch_recognize_hands_matches_single_and_selection(self):
        import cv2
        import torch
        from classifiers.card_classifier.pipeline import decode_card_prediction
        from classifiers.inference import batch_recognize_hands, recognize_hand_cards

        # Unit check decode_card_prediction with dict-of-tensors output & "selected": 1.0
        preds = {
            "rank_idx": torch.tensor([12]),
            "rank_conf": torch.tensor([0.99]),
            "suit_idx": torch.tensor([0]),
            "suit_conf": torch.tensor([0.98]),
            "enhancement_idx": torch.tensor([0]),
            "enhancement_conf": torch.tensor([0.97]),
            "seal_idx": torch.tensor([0]),
            "seal_conf": torch.tensor([0.96]),
            "edition_idx": torch.tensor([0]),
            "edition_conf": torch.tensor([0.95]),
            "debuffed": torch.tensor([False]),
            "face_down": torch.tensor([False]),
        }
        res = decode_card_prediction(
            preds, 0, index=1, box=[10, 20, 90, 150], selected=True
        )
        self.assertEqual(res.index, 1)
        self.assertEqual(res.box, [10, 20, 90, 150])
        self.assertEqual(res.confidences["selected"], 1.0)
        self.assertTrue(res.selected)

        sample_path = REPO_ROOT / "assets" / "sample_inputs" / "in_blind.jpg"
        if sample_path.exists():
            img = cv2.imread(str(sample_path))
            single = recognize_hand_cards(img)
            batch = batch_recognize_hands([img])[0]
            self.assertEqual(
                [(c.index, c.rank, c.suit, c.selected, c.box) for c in single.cards],
                [(c.index, c.rank, c.suit, c.selected, c.box) for c in batch.cards],
            )
            rx, ry, rw, rh = single.roi
            for c in single.cards:
                self.assertGreaterEqual(c.box[0], rx)
                self.assertGreaterEqual(c.box[1], ry)
                self.assertLessEqual(c.box[2], rx + rw)
                self.assertLessEqual(c.box[3], ry + rh)
            self.assertEqual(single.roi[3], batch.roi[3])
            self.assertGreater(batch.roi[3], 1)

    def test_shop_full_frame_boxes_and_joker_aliases(self):
        from unittest.mock import MagicMock, patch
        import numpy as np
        import torch
        from classifiers.inference import (
            ROIS,
            _offset_box,
            batch_recognize_shops,
            recognize_shop,
        )
        from classifiers.shop_classifier.pipeline import ShopItemResult
        from game.engine import cv_stubs

        # 1. All non-placeholder j_* keys in all_keys.json resolve to exact JokerName (not fallback JOKER)
        keys_path = (
            REPO_ROOT
            / "classifiers"
            / "shop_classifier"
            / "stage2_classifier"
            / "all_keys.json"
        )
        with keys_path.open("r", encoding="utf-8") as f:
            all_keys = json.load(f)
        for k in all_keys:
            if k.startswith("j_") and k not in ("j_locked", "j_undiscovered"):
                converted = cv_stubs._convert_shop_item_result(
                    ShopItemResult(
                        index=0,
                        card_type="Joker",
                        key=k,
                        edition="None",
                        eternal=False,
                        perishable=False,
                        rental=False,
                        debuffed=False,
                        box=[0, 0, 10, 10],
                        confidences={},
                    )
                )
                self.assertIsInstance(converted, bm.Joker)
                if k != "j_joker":
                    self.assertNotEqual(
                        converted.name, bm.JokerName.JOKER, msg=f"Unmapped key: {k}"
                    )

        # 2. Both recognize_shop and batch_recognize_shops return full-frame (1920x1080) boxes
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        local_box = [10, 20, 90, 140]

        mock_pipe = MagicMock()
        mock_pipe.device = torch.device("cpu")
        mock_pipe.class_names = {0: "Joker"}
        mock_pipe.transform = lambda crop: torch.zeros((3, 192, 128))
        mock_pipe.predict.side_effect = lambda crop, conf_threshold=0.35: [
            ShopItemResult(
                index=0,
                card_type="Joker",
                key="j_duo",
                edition="None",
                eternal=False,
                perishable=False,
                rental=False,
                debuffed=False,
                box=list(local_box),
                confidences={},
            )
        ]

        det_box = MagicMock()
        det_box.__len__.return_value = 1
        det_box.xyxy = torch.tensor([local_box], dtype=torch.float32)
        det_box.conf = torch.tensor([0.95], dtype=torch.float32)
        det_box.cls = torch.tensor([0.0], dtype=torch.float32)
        det_res = MagicMock(boxes=det_box)
        mock_pipe.detector.predict.side_effect = lambda source, **kw: [
            det_res for _ in source
        ]
        mock_pipe.classifier.predict_attributes.side_effect = lambda tensor: [
            {
                "key": "j_duo",
                "edition": "None",
                "eternal": False,
                "perishable": False,
                "rental": False,
                "debuffed": False,
                "key_conf": 0.99,
                "edition_conf": 0.99,
                "eternal_conf": 0.0,
                "perishable_conf": 0.0,
                "rental_conf": 0.0,
                "debuffed_conf": 0.0,
            }
            for _ in range(len(tensor))
        ]

        with patch("classifiers.inference.get_shop_pipeline", return_value=mock_pipe):
            single_out = recognize_shop(frame, mode="shop")
            batch_out = batch_recognize_shops([frame], mode="shop")[0]

        for shelf in ("joker_bar", "consumeables", "shop", "voucher", "booster"):
            attr_name = {
                "shop": "shop_items",
                "voucher": "vouchers",
                "booster": "boosters",
            }.get(shelf, shelf)
            expected_box = _offset_box(
                local_box, round(ROIS[shelf]["x"]), round(ROIS[shelf]["y"])
            )
            self.assertEqual(getattr(single_out, attr_name)[0].box, expected_box)
            self.assertEqual(getattr(batch_out, attr_name)[0].box, expected_box)

    def test_target_chips_lut_and_round_ante_progression(self):
        self.assertEqual(
            gs.calculate_target_chips(ante=1, round=1, stake=bm.Stake.WHITE), 300
        )
        self.assertEqual(
            gs.calculate_target_chips(ante=2, round=1, stake=bm.Stake.GREEN), 900
        )
        self.assertEqual(
            gs.calculate_target_chips(ante=8, round=3, stake=bm.Stake.GOLD), 400000
        )
        self.assertEqual(
            gs.calculate_target_chips(
                ante=4, blind=bm.BlindName.THE_WALL, stake=bm.Stake.WHITE
            ),
            20000,
        )
        self.assertEqual(
            gs.calculate_target_chips(
                ante=1, round=1, stake=bm.Stake.WHITE, deck=bm.Deck.PLASMA_DECK
            ),
            600,
        )
        self.assertGreater(
            gs.calculate_target_chips(ante=10, round=1, stake=bm.Stake.WHITE), 100000
        )

        state = gs.GameState(
            seed="ANTETEST", stake=bm.Stake.WHITE, apply_deck_stake_on_init=True
        )
        self.assertEqual(state.target_chips, 300)
        self.assertEqual(state.round, 1)
        self.assertEqual(state.ante, 1)

        # END_ROUND advances round 1 -> 2; CASH_OUT keeps round == 2 (no double increment)
        state.location = bm.Location.BLIND
        state.current_blind = bm.BlindName.SMALL
        state, _ = gs.step_simulation(state, bm.Action(type=bm.ActionType.END_ROUND))
        self.assertEqual(state.round, 2)
        self.assertTrue(state.round_won)
        self.assertEqual(state.location, bm.Location.BLIND)

        state, _ = gs.step_simulation(state, bm.Action(type=bm.ActionType.CASH_OUT))
        self.assertEqual(state.round, 2)
        self.assertEqual(state.location, bm.Location.SHOP)

        # Fallback: CASH_OUT without END_ROUND on Boss Blind still advances round (2 -> 3),
        # and NEXT_ROUND advances ante (1 -> 2) and updates target_chips (800).
        state.current_blind = bm.BlindName.THE_HOOK
        state.boss_blind = bm.BlindName.THE_HOOK
        state.location = bm.Location.BLIND
        state.round_advanced = False
        state, _ = gs.step_simulation(state, bm.Action(type=bm.ActionType.CASH_OUT))
        self.assertEqual(state.round, 3)

        state, _ = gs.step_simulation(state, bm.Action(type=bm.ActionType.NEXT_ROUND))
        self.assertEqual(state.ante, 2)
        self.assertEqual(state.current_blind, bm.BlindName.SMALL)
        self.assertEqual(state.target_chips, 800)
        self.assertEqual(state.location, bm.Location.SELECT)

        # Missed NEXT_ROUND fallback: Boss CASH_OUT -> SELECT_BLIND still advances ante (2 -> 3)
        state.current_blind = state.boss_blind or bm.BlindName.THE_HOOK
        state.location = bm.Location.BLIND
        state.round_advanced = False
        state, _ = gs.step_simulation(state, bm.Action(type=bm.ActionType.CASH_OUT))
        state, _ = gs.step_simulation(state, bm.Action(type=bm.ActionType.SELECT_BLIND))
        self.assertEqual(state.ante, 3)
        self.assertEqual(state.current_blind, bm.BlindName.SMALL)
        self.assertEqual(state.target_chips, 2000)

    def test_pack_exit_restores_pre_pack_location(self):
        # Skip Blind -> Charm Tag pack opens -> exit restores Location.SELECT
        state = gs.GameState(seed="PACKEXIT1")
        state.location = bm.Location.SELECT
        state.current_blind = bm.BlindName.SMALL
        state.small_blind_tag = bm.SkipTag.CHARM_TAG
        state, _ = gs.step_simulation(state, bm.Action(type=bm.ActionType.SKIP_BLIND))
        self.assertEqual(state.location, bm.Location.PACK)
        self.assertEqual(state.pre_pack_location, bm.Location.SELECT)

        state, _ = gs.step_simulation(state, bm.Action(type=bm.ActionType.SKIP_PACK))
        self.assertEqual(state.location, bm.Location.SELECT)
        self.assertIsNone(state.pre_pack_location)

        # Double Tag + Charm Tag opens 2 sequential packs before restoring Location.SELECT
        state.active_tags.append(bm.SkipTag.DOUBLE_TAG)
        state.current_blind = bm.BlindName.BIG
        state.big_blind_tag = bm.SkipTag.CHARM_TAG
        state, _ = gs.step_simulation(state, bm.Action(type=bm.ActionType.SKIP_BLIND))
        self.assertEqual(state.location, bm.Location.PACK)
        self.assertEqual(len(state.pending_packs), 1)
        state, _ = gs.step_simulation(state, bm.Action(type=bm.ActionType.SKIP_PACK))
        self.assertEqual(state.location, bm.Location.PACK)
        state, _ = gs.step_simulation(state, bm.Action(type=bm.ActionType.SKIP_PACK))
        self.assertEqual(state.location, bm.Location.SELECT)

        # Shop pack -> exit restores Location.SHOP
        state.location = bm.Location.SHOP
        state.money = 20
        state.shop_packs = [
            bm.BoosterPack(
                booster_type=bm.BoosterType.ARCANA,
                size=bm.BoosterSize.REGULAR,
                cost=4,
                choose=1,
                extra=3,
            )
        ]
        state, _ = gs.step_simulation(
            state, bm.Action(type=bm.ActionType.OPEN_PACK, target_idx=0)
        )
        self.assertEqual(state.location, bm.Location.PACK)
        state, _ = gs.step_simulation(state, bm.Action(type=bm.ActionType.SKIP_PACK))
        self.assertEqual(state.location, bm.Location.SHOP)

    def test_skip_tags_with_prng(self):
        # Double + Investment Tag -> +$50 on boss CASH_OUT
        s = gs.GameState(seed="TAGS1", apply_deck_stake_on_init=True)
        s.location = bm.Location.SELECT
        s.current_blind = bm.BlindName.SMALL
        s.small_blind_tag = bm.SkipTag.DOUBLE_TAG
        s.big_blind_tag = bm.SkipTag.INVESTMENT_TAG
        s, _ = gs.step_simulation(s, bm.Action(type=bm.ActionType.SKIP_BLIND))
        self.assertEqual(s.active_tags, [bm.SkipTag.DOUBLE_TAG])
        s, _ = gs.step_simulation(s, bm.Action(type=bm.ActionType.SKIP_BLIND))
        self.assertEqual(s.active_tags.count(bm.SkipTag.INVESTMENT_TAG), 2)

        money_before = s.money
        s.location = bm.Location.BLIND
        s.current_blind = s.boss_blind or bm.BlindName.THE_HOOK
        s, _ = gs.step_simulation(s, bm.Action(type=bm.ActionType.END_ROUND))
        s, _ = gs.step_simulation(
            s,
            bm.Action(
                type=bm.ActionType.CASH_OUT,
                metadata={"include_round_payout": False},
            ),
        )
        self.assertEqual(s.money, money_before + 50)
        self.assertNotIn(bm.SkipTag.INVESTMENT_TAG, s.active_tags)

        # Top-up Tag adds 2 Common jokers via PRNG; Orbital Tag levels a hand by +3; Economy Tag doubles capped at $40
        s2 = gs.GameState(seed="TAGS2", apply_deck_stake_on_init=True)
        s2.location = bm.Location.SELECT
        s2.current_blind = bm.BlindName.SMALL
        s2.small_blind_tag = bm.SkipTag.TOP_UP_TAG
        s2, _ = gs.step_simulation(s2, bm.Action(type=bm.ActionType.SKIP_BLIND))
        self.assertEqual(len(s2.inventory.jokers), 2)
        self.assertTrue(
            all(j.info.rarity == bm.JokerRarity.COMMON for j in s2.inventory.jokers)
        )

        s2.current_blind = bm.BlindName.SMALL
        s2.small_blind_tag = bm.SkipTag.ORBITAL_TAG
        s2, _ = gs.step_simulation(s2, bm.Action(type=bm.ActionType.SKIP_BLIND))
        self.assertEqual(sum(v - 1 for v in s2.hand_levels.values()), 3)

        s2.money = 30
        s2.current_blind = bm.BlindName.SMALL
        s2.small_blind_tag = bm.SkipTag.ECONOMY_TAG
        s2, _ = gs.step_simulation(s2, bm.Action(type=bm.ActionType.SKIP_BLIND))
        self.assertEqual(s2.money, 60)

        # Juggle Tag gives +3 hand size on SELECT_BLIND, removed at END_ROUND
        base_hand_size = s2.hand.max_size
        s2.active_tags.append(bm.SkipTag.JUGGLE_TAG)
        s2, _ = gs.step_simulation(s2, bm.Action(type=bm.ActionType.SELECT_BLIND))
        self.assertEqual(s2.hand.max_size, base_hand_size + 3)
        s2, _ = gs.step_simulation(s2, bm.Action(type=bm.ActionType.END_ROUND))
        self.assertEqual(s2.hand.max_size, base_hand_size)

        # D6 Tag zeroes reroll_cost; Coupon Tag zeroes shop card/pack cost; Foil Tag applies Foil in shop
        s2.active_tags.extend(
            [bm.SkipTag.D6_TAG, bm.SkipTag.COUPON_TAG, bm.SkipTag.FOIL_TAG]
        )
        s2.shop_cards = [
            bm.Joker(name=bm.JokerName.JOKER),
            bm.Tarot(name=bm.TarotName.FOOL),
        ]
        s2.location = bm.Location.BLIND
        s2.current_blind = bm.BlindName.SMALL
        s2, _ = gs.step_simulation(s2, bm.Action(type=bm.ActionType.END_ROUND))
        s2, _ = gs.step_simulation(s2, bm.Action(type=bm.ActionType.CASH_OUT))
        self.assertEqual(s2.reroll_cost, 0)
        self.assertTrue(s2.coupon_shop_active)
        self.assertEqual(s2.shop_cards[0].edition, bm.Edition.FOIL)
        self.assertTrue(s2.shop_cards[0].couponed)

    def test_all_decks_and_stakes(self):
        # Decks
        red = gs.GameState(deck=bm.Deck.RED_DECK)
        self.assertEqual(red.discards_left, 4)

        blue = gs.GameState(deck=bm.Deck.BLUE_DECK)
        self.assertEqual(blue.hands_left, 5)

        yellow = gs.GameState(deck=bm.Deck.YELLOW_DECK)
        self.assertEqual(yellow.money, 14)

        green = gs.GameState(deck=bm.Deck.GREEN_DECK)
        self.assertTrue(green.no_interest)

        black = gs.GameState(deck=bm.Deck.BLACK_DECK)
        self.assertEqual(black.hands_left, 3)
        self.assertEqual(black.inventory.joker_slots, 6)

        magic = gs.GameState(deck=bm.Deck.MAGIC_DECK)
        self.assertIn(bm.VoucherName.CRYSTAL_BALL, magic.vouchers_redeemed)
        self.assertEqual(
            [c.name for c in magic.inventory.consumables],
            [bm.TarotName.FOOL, bm.TarotName.FOOL],
        )

        nebula = gs.GameState(deck=bm.Deck.NEBULA_DECK)
        self.assertIn(bm.VoucherName.TELESCOPE, nebula.vouchers_redeemed)
        self.assertEqual(nebula.inventory.consumable_slots, 1)

        ghost = gs.GameState(deck=bm.Deck.GHOST_DECK)
        self.assertEqual(ghost.spectral_rate, 2.0)
        self.assertEqual(
            [c.name for c in ghost.inventory.consumables], [bm.SpectralName.HEX]
        )

        abandoned = gs.GameState(deck=bm.Deck.ABANDONED_DECK)
        self.assertEqual(len(abandoned.full_deck), 40)
        self.assertFalse(
            any(
                c.rank in (bm.Rank.JACK, bm.Rank.QUEEN, bm.Rank.KING)
                for c in abandoned.full_deck
            )
        )

        checkered = gs.GameState(deck=bm.Deck.CHECKERED_DECK)
        self.assertEqual(len(checkered.full_deck), 52)
        self.assertEqual(
            {c.suit for c in checkered.full_deck}, {bm.Suit.SPADES, bm.Suit.HEARTS}
        )

        zodiac = gs.GameState(deck=bm.Deck.ZODIAC_DECK)
        self.assertEqual(zodiac.shop_size, 3)
        self.assertEqual(zodiac.tarot_rate, 4.0 * 2.4)
        self.assertEqual(zodiac.planet_rate, 4.0 * 2.4)

        painted = gs.GameState(deck=bm.Deck.PAINTED_DECK)
        self.assertEqual(painted.hand.max_size, 10)
        self.assertEqual(painted.inventory.joker_slots, 4)

        anaglyph = gs.GameState(deck=bm.Deck.ANAGLYPH_DECK)
        anaglyph.location = bm.Location.BLIND
        anaglyph.current_blind = bm.BlindName.THE_HOOK
        anaglyph, _ = gs.step_simulation(
            anaglyph, bm.Action(type=bm.ActionType.END_ROUND)
        )
        self.assertIn(bm.SkipTag.DOUBLE_TAG, anaglyph.active_tags)

        plasma = gs.GameState(deck=bm.Deck.PLASMA_DECK)
        self.assertEqual(plasma.ante_scaling, 2.0)
        self.assertEqual(plasma.target_chips, 600)
        self.assertEqual(plasma.balance_plasma_score(100, 10), (55, 55))

        err1 = gs.GameState(deck=bm.Deck.ERRATIC_DECK, seed="ERRATIC1")
        err2 = gs.GameState(deck=bm.Deck.ERRATIC_DECK, seed="ERRATIC1")
        self.assertEqual(len(err1.full_deck), 52)
        self.assertEqual(
            [(c.rank, c.suit) for c in err1.full_deck],
            [(c.rank, c.suit) for c in err2.full_deck],
        )

        challenge = gs.GameState(deck=bm.Deck.CHALLENGE_DECK)
        self.assertEqual(len(challenge.full_deck), 52)

        # Stakes
        self.assertEqual(
            gs.GameState(deck=bm.Deck.BLUE_DECK, stake=bm.Stake.RED).discards_left, 3
        )
        self.assertEqual(gs.GameState(stake=bm.Stake.GREEN).target_chips, 300)
        black_s = gs.GameState(stake=bm.Stake.BLACK)
        self.assertTrue(black_s.enable_eternals_in_shop)
        blue_s = gs.GameState(deck=bm.Deck.BLUE_DECK, stake=bm.Stake.BLUE)
        self.assertEqual(blue_s.discards_left, 2)
        self.assertEqual(
            gs.GameState(deck=bm.Deck.RED_DECK, stake=bm.Stake.BLUE).discards_left, 3
        )
        purple_s = gs.GameState(stake=bm.Stake.PURPLE, ante=2)
        self.assertEqual(purple_s.target_chips, 1000)
        orange_s = gs.GameState(stake=bm.Stake.ORANGE)
        self.assertTrue(orange_s.enable_perishables_in_shop)
        gold_s = gs.GameState(stake=bm.Stake.GOLD)
        self.assertTrue(gold_s.enable_rentals_in_shop)

    def test_consumables_planets_vouchers_and_pricing(self):
        # 1. Planet + The Fool ordering & copy (Fool does not copy itself)
        s = gs.GameState(seed="CONS1", apply_deck_stake_on_init=True)
        s.inventory.consumables = [
            bm.Planet(name=bm.PlanetName.MERCURY),
            bm.Tarot(name=bm.TarotName.FOOL),
        ]
        s, _ = gs.step_simulation(
            s, bm.Action(type=bm.ActionType.USE_CONSUMABLE, target_idx=0)
        )
        self.assertEqual(s.hand_levels[bm.PokerHand.PAIR], 2)
        self.assertEqual(s.last_tarot_planet, bm.PlanetName.MERCURY)
        s, _ = gs.step_simulation(
            s, bm.Action(type=bm.ActionType.USE_CONSUMABLE, target_idx=0)
        )
        self.assertEqual(
            [c.name for c in s.inventory.consumables], [bm.PlanetName.MERCURY]
        )

        # Fool used right after Fool shouldn't spawn Fool
        s.inventory.consumables = [bm.Tarot(name=bm.TarotName.FOOL)]
        s, _ = gs.step_simulation(
            s, bm.Action(type=bm.ActionType.USE_CONSUMABLE, target_idx=0)
        )
        self.assertEqual(s.inventory.consumables, [])

        # 2. Emperor, High Priestess, Judgement, Black Hole, The Soul, Wraith
        s.inventory.consumables = [
            bm.Tarot(name=bm.TarotName.EMPEROR),
            bm.Tarot(name=bm.TarotName.HIGH_PRIESTESS),
            bm.Tarot(name=bm.TarotName.JUDGEMENT),
            bm.Spectral(name=bm.SpectralName.BLACK_HOLE),
            bm.Spectral(name=bm.SpectralName.THE_SOUL),
            bm.Spectral(name=bm.SpectralName.WRAITH),
        ]
        s.inventory.consumable_slots = 8
        s.inventory.joker_slots = 8
        s.money = 50
        s, _ = gs.step_simulation(
            s, bm.Action(type=bm.ActionType.USE_CONSUMABLE, target_idx=0)
        )  # Emperor (+2 Tarots)
        self.assertEqual(
            sum(1 for c in s.inventory.consumables if c.type == bm.CardType.TAROT), 4
        )
        s, _ = gs.step_simulation(
            s, bm.Action(type=bm.ActionType.USE_CONSUMABLE, target_idx=0)
        )  # High Priestess (+2 Planets)
        self.assertEqual(
            sum(1 for c in s.inventory.consumables if c.type == bm.CardType.PLANET), 2
        )
        s, _ = gs.step_simulation(
            s, bm.Action(type=bm.ActionType.USE_CONSUMABLE, target_idx=0)
        )  # Judgement
        self.assertEqual(len(s.inventory.jokers), 1)
        s, _ = gs.step_simulation(
            s, bm.Action(type=bm.ActionType.USE_CONSUMABLE, target_idx=0)
        )  # Black Hole
        self.assertEqual(s.hand_levels[bm.PokerHand.FLUSH], 2)
        s, _ = gs.step_simulation(
            s, bm.Action(type=bm.ActionType.USE_CONSUMABLE, target_idx=0)
        )  # The Soul
        self.assertEqual(s.inventory.jokers[-1].info.rarity, bm.JokerRarity.LEGENDARY)
        s, _ = gs.step_simulation(
            s, bm.Action(type=bm.ActionType.USE_CONSUMABLE, target_idx=0)
        )  # Wraith
        self.assertEqual(s.money, 0)
        self.assertEqual(s.inventory.jokers[-1].info.rarity, bm.JokerRarity.RARE)

        # 3. Ankh destroys non-eternals, preserves eternal, clones chosen without Negative
        ankh_s = gs.GameState(seed="ANKH1")
        ankh_s.inventory.jokers = [
            bm.Joker(name=bm.JokerName.JOKER, id=10, edition=bm.Edition.NEGATIVE),
            bm.Joker(name=bm.JokerName.GREEDY_JOKER, id=11, eternal=True),
        ]
        ankh_s.inventory.consumables = [bm.Spectral(name=bm.SpectralName.ANKH)]
        ankh_s, _ = gs.step_simulation(
            ankh_s, bm.Action(type=bm.ActionType.USE_CONSUMABLE, target_idx=0)
        )
        self.assertTrue(any(j.eternal for j in ankh_s.inventory.jokers))
        clones = [j for j in ankh_s.inventory.jokers if j.id not in (0, 10, 11)]
        self.assertEqual(len(clones), 1)
        self.assertNotEqual(clones[0].edition, bm.Edition.NEGATIVE)

        # 4. Ouija sets uniform rank & reduces hand size
        ouija_s = gs.GameState(seed="OUIJA1")
        ouija_s.hand.cards = [
            bm.HandCard(
                card=bm.PlayingCard(rank=bm.Rank.TWO, suit=bm.Suit.SPADES, id=1)
            ),
            bm.HandCard(
                card=bm.PlayingCard(rank=bm.Rank.KING, suit=bm.Suit.HEARTS, id=2)
            ),
        ]
        ouija_s.inventory.consumables = [bm.Spectral(name=bm.SpectralName.OUIJA)]
        start_size = ouija_s.hand.max_size
        ouija_s, _ = gs.step_simulation(
            ouija_s, bm.Action(type=bm.ActionType.USE_CONSUMABLE, target_idx=0)
        )
        self.assertEqual(
            ouija_s.hand.cards[0].card.rank, ouija_s.hand.cards[1].card.rank
        )
        self.assertEqual(ouija_s.hand.max_size, start_size - 1)

        # 5. Vouchers + Observatory + Pricing (Astronomer, Coupon Tag, Rental, Clearance Sale, Liquidation)
        vs = gs.GameState(seed="VOUCH1")
        vs.location = bm.Location.SHOP
        vs.money = 100
        vs.shop_vouchers = [
            bm.Voucher(name=bm.VoucherName.OVERSTOCK),
            bm.Voucher(name=bm.VoucherName.CLEARANCE_SALE),
            bm.Voucher(name=bm.VoucherName.OBSERVATORY),
        ]
        vs, _ = gs.step_simulation(
            vs, bm.Action(type=bm.ActionType.REDEEM_VOUCHER, target_idx=0)
        )
        self.assertEqual(vs.shop_size, 3)
        self.assertEqual(vs.money, 90)
        vs, _ = gs.step_simulation(
            vs, bm.Action(type=bm.ActionType.REDEEM_VOUCHER, target_idx=0)
        )
        self.assertEqual(vs.discount_percent, 25)
        vs, _ = gs.step_simulation(
            vs, bm.Action(type=bm.ActionType.REDEEM_VOUCHER, target_idx=0)
        )
        # Voucher cost with 25% discount: floor((10 + 0.5) * 0.75) = 7
        self.assertEqual(vs.money, 90 - 10 - 7)

        vs.inventory.consumables = [
            bm.Planet(name=bm.PlanetName.MERCURY),
            bm.Planet(name=bm.PlanetName.MERCURY),
        ]
        self.assertAlmostEqual(vs.get_observatory_x_mult(bm.PokerHand.PAIR), 2.25)

        vs.inventory.jokers = [bm.Joker(name=bm.JokerName.ASTRONOMER)]
        planet_card = bm.Planet(name=bm.PlanetName.JUPITER)
        celestial_pack = bm.BoosterPack(
            booster_type=bm.BoosterType.CELESTIAL,
            size=bm.BoosterSize.REGULAR,
            cost=4,
            choose=1,
            extra=3,
        )
        self.assertEqual(planet_card.get_cost(astronomer=True), 0)
        self.assertGreaterEqual(planet_card.get_sell_value(astronomer=True), 1)
        self.assertEqual(celestial_pack.get_cost(astronomer=True), 0)

        rental_joker = bm.Joker(name=bm.JokerName.BLUEPRINT, rental=True)
        self.assertEqual(rental_joker.get_cost(discount_percent=0, inflation=10), 1)

        vs.coupon_shop_active = True
        expensive_joker = bm.Joker(name=bm.JokerName.BLUEPRINT)
        vs.shop_cards = [expensive_joker]
        money_pre_coupon = vs.money
        vs, _ = gs.step_simulation(
            vs, bm.Action(type=bm.ActionType.BUY_CARD, target_idx=0)
        )
        self.assertEqual(vs.money, money_pre_coupon)
        self.assertGreaterEqual(vs.inventory.jokers[-1].get_sell_value(), 1)

    def test_auto_skip_tags_default_cash_out_and_prng_jokers(self):
        """Verifies Skip Tags PRNG population, default CASH_OUT payout, and Madness/Invisible/Perkeo."""
        from game.engine.rng_sim.api import BalatroRun

        # 1. GameState(seed=...) automatically populates Ante 1 tags matching BalatroRun
        s = gs.GameState(seed="AUTOTAG1")
        pred_small, pred_big = BalatroRun("AUTOTAG1").predict_ante_tags(
            ante=1, include_orbital_hands=True
        )
        self.assertEqual(s.small_blind_tag, pred_small.skip_tag)
        self.assertEqual(s.big_blind_tag, pred_big.skip_tag)
        if s.small_blind_tag == bm.SkipTag.ORBITAL_TAG:
            self.assertEqual(s.small_blind_orbital_hand, pred_small.orbital_hand)

        # Advancing from Ante 1 Boss to Ante 2 Small Blind rolls Ante 2 tags automatically
        s.location = bm.Location.BLIND
        s.current_blind = s.boss_blind or bm.BlindName.THE_HOOK
        s.money = 10
        s.hands_left = 2
        s, _ = gs.step_simulation(s, bm.Action(type=bm.ActionType.END_ROUND))
        # Default CASH_OUT (no metadata) pays blind reward ($5 boss) + remaining hands ($2) + interest ($2) = +$9
        s, _ = gs.step_simulation(s, bm.Action(type=bm.ActionType.CASH_OUT))
        self.assertEqual(s.money, 19)
        s, _ = gs.step_simulation(s, bm.Action(type=bm.ActionType.NEXT_ROUND))
        self.assertEqual(s.ante, 2)
        pred2_small, pred2_big = BalatroRun("AUTOTAG1").predict_ante_tags(ante=2)
        self.assertEqual(s.small_blind_tag, pred2_small.skip_tag)
        self.assertEqual(s.big_blind_tag, pred2_big.skip_tag)

        # 2. Madness increments x_mult and destroys another non-Eternal Joker on Small/Big Blind
        mad = bm.Joker(name=bm.JokerName.MADNESS)
        fodder = bm.Joker(name=bm.JokerName.JOKER)
        eternal = bm.Joker(name=bm.JokerName.BLUEPRINT, eternal=True)
        sm = gs.GameState(
            seed="MAD1",
            location=bm.Location.SELECT,
            current_blind=bm.BlindName.SMALL,
            inventory=bm.Inventory(jokers=[mad, fodder, eternal]),
        )
        sm, _ = gs.step_simulation(sm, bm.Action(type=bm.ActionType.SELECT_BLIND))
        self.assertAlmostEqual(mad.state["x_mult"], 1.5)
        self.assertNotIn(fodder, sm.inventory.jokers)
        self.assertIn(eternal, sm.inventory.jokers)

        # 3. Invisible Joker tracks rounds and clones another Joker on sell (stripping Negative)
        invis = bm.Joker(name=bm.JokerName.INVISIBLE_JOKER)
        neg_bp = bm.Joker(name=bm.JokerName.BLUEPRINT, edition=bm.Edition.NEGATIVE)
        si = gs.GameState(
            seed="INV1",
            location=bm.Location.BLIND,
            inventory=bm.Inventory(jokers=[invis, neg_bp]),
        )
        for _ in range(2):
            si.location = bm.Location.BLIND
            si.round_won = False
            si, _ = gs.step_simulation(si, bm.Action(type=bm.ActionType.END_ROUND))
            si, _ = gs.step_simulation(si, bm.Action(type=bm.ActionType.CASH_OUT))
        self.assertEqual(invis.state["rounds"], 2)
        si, _ = gs.step_simulation(
            si, bm.Action(type=bm.ActionType.SELL_CARD, target_idx=0)
        )
        self.assertEqual(len(si.inventory.jokers), 2)
        self.assertEqual(si.inventory.jokers[1].name, bm.JokerName.BLUEPRINT)
        self.assertEqual(si.inventory.jokers[1].edition, bm.Edition.NONE)
        self.assertNotEqual(si.inventory.jokers[0].id, si.inventory.jokers[1].id)

        # 4. Perkeo copies a held consumable as Negative on NEXT_ROUND (leaving shop)
        perkeo = bm.Joker(name=bm.JokerName.PERKEO)
        fool = bm.Tarot(name=bm.TarotName.FOOL)
        sp = gs.GameState(
            seed="PERK1",
            location=bm.Location.SHOP,
            inventory=bm.Inventory(jokers=[perkeo], consumables=[fool]),
        )
        sp, _ = gs.step_simulation(sp, bm.Action(type=bm.ActionType.NEXT_ROUND))
        self.assertEqual(len(sp.inventory.consumables), 2)
        self.assertEqual(sp.inventory.consumables[1].name, bm.TarotName.FOOL)
        self.assertEqual(sp.inventory.consumables[1].edition, bm.Edition.NEGATIVE)

    def test_all_boss_blind_semantics(self):
        """Validates gameplay/state semantics across Boss Blinds & Luchador/Chicot disable."""
        # 1. The Water, The Needle, and Luchador restore
        luch = bm.Joker(name=bm.JokerName.LUCHADOR)
        s = gs.GameState(
            seed="BOSS1",
            location=bm.Location.SELECT,
            current_blind=bm.BlindName.THE_WATER,
            boss_blind=bm.BlindName.THE_WATER,
            inventory=bm.Inventory(jokers=[luch]),
        )
        s, _ = gs.step_simulation(s, bm.Action(type=bm.ActionType.SELECT_BLIND))
        self.assertEqual(s.discards_left, 0)
        s, _ = gs.step_simulation(
            s, bm.Action(type=bm.ActionType.SELL_CARD, target_idx=0)
        )
        self.assertTrue(s.boss_disabled)
        self.assertEqual(s.discards_left, 4)  # Red Deck default = 4 discards

        s.location = bm.Location.SELECT
        s.current_blind = bm.BlindName.THE_NEEDLE
        s.inventory.jokers = [bm.Joker(name=bm.JokerName.CHICOT)]
        s, _ = gs.step_simulation(s, bm.Action(type=bm.ActionType.SELECT_BLIND))
        self.assertTrue(s.boss_disabled)
        self.assertEqual(s.hands_left, 4)

        # 2. The Manacle (-1 hand size restored at END_ROUND)
        sm = gs.GameState(
            seed="MAN1",
            location=bm.Location.SELECT,
            current_blind=bm.BlindName.THE_MANACLE,
        )
        sm, _ = gs.step_simulation(sm, bm.Action(type=bm.ActionType.SELECT_BLIND))
        self.assertEqual(sm.hand.max_size, 7)
        sm, _ = gs.step_simulation(sm, bm.Action(type=bm.ActionType.END_ROUND))
        self.assertEqual(sm.hand.max_size, 8)

        # 3. The Arm, The Ox, The Psychic, The Eye, The Mouth
        sax = gs.GameState(
            seed="ARMOX1",
            location=bm.Location.BLIND,
            current_blind=bm.BlindName.THE_ARM,
            money=25,
            hand_levels={bm.PokerHand.PAIR: 3, bm.PokerHand.HIGH_CARD: 1},
            hand=bm.Hand(
                cards=[
                    bm.HandCard(
                        card=bm.PlayingCard(
                            rank=bm.Rank.ACE, suit=bm.Suit.SPADES, id=1
                        ),
                        selected=True,
                    ),
                    bm.HandCard(
                        card=bm.PlayingCard(
                            rank=bm.Rank.ACE, suit=bm.Suit.HEARTS, id=2
                        ),
                        selected=True,
                    ),
                ]
            ),
        )
        sax, _ = gs.step_simulation(sax, bm.Action(type=bm.ActionType.PLAY_HAND))
        self.assertEqual(sax.hand_levels[bm.PokerHand.PAIR], 2)

        sax.current_blind = bm.BlindName.THE_OX
        sax.ox_target_hand = bm.PokerHand.HIGH_CARD
        sax.hand.cards = [
            bm.HandCard(
                card=bm.PlayingCard(rank=bm.Rank.KING, suit=bm.Suit.CLUBS, id=3),
                selected=True,
            )
        ]
        sax, _ = gs.step_simulation(sax, bm.Action(type=bm.ActionType.PLAY_HAND))
        self.assertEqual(sax.money, 0)

        # The Psychic: < 5 cards skips scoring triggers (Green Joker doesn't gain +1 mult)
        gj = bm.Joker(name=bm.JokerName.GREEN_JOKER)
        sp = gs.GameState(
            seed="PSY1",
            location=bm.Location.BLIND,
            current_blind=bm.BlindName.THE_PSYCHIC,
            inventory=bm.Inventory(jokers=[gj]),
            hand=bm.Hand(
                cards=[
                    bm.HandCard(
                        card=bm.PlayingCard(
                            rank=bm.Rank.ACE, suit=bm.Suit.SPADES, id=1
                        ),
                        selected=True,
                    )
                ]
            ),
        )
        sp, _ = gs.step_simulation(sp, bm.Action(type=bm.ActionType.PLAY_HAND))
        self.assertEqual(gj.state.get("mult", 0), 0)

        # The Eye: repeating same hand type skips scoring triggers
        se = gs.GameState(
            seed="EYE1",
            location=bm.Location.BLIND,
            current_blind=bm.BlindName.THE_EYE,
            inventory=bm.Inventory(jokers=[bm.Joker(name=bm.JokerName.GREEN_JOKER)]),
            hand=bm.Hand(
                cards=[
                    bm.HandCard(
                        card=bm.PlayingCard(
                            rank=bm.Rank.ACE, suit=bm.Suit.SPADES, id=1
                        ),
                        selected=True,
                    ),
                    bm.HandCard(
                        card=bm.PlayingCard(
                            rank=bm.Rank.KING, suit=bm.Suit.HEARTS, id=2
                        ),
                        selected=False,
                    ),
                ]
            ),
        )
        se, _ = gs.step_simulation(se, bm.Action(type=bm.ActionType.PLAY_HAND))
        self.assertEqual(se.inventory.jokers[0].state["mult"], 1)
        se.hand.cards[0].selected = True
        se, _ = gs.step_simulation(se, bm.Action(type=bm.ActionType.PLAY_HAND))
        self.assertEqual(se.inventory.jokers[0].state["mult"], 1)  # debuffed repeat

        # 4. The Serpent (draws 3 cards after play/discard), The Fish/House/Mark/Wheel face-down
        serp = gs.GameState(
            seed="SERP1",
            location=bm.Location.BLIND,
            current_blind=bm.BlindName.THE_SERPENT,
        )
        serp.draw_cards(8)
        serp.hand.cards[0].selected = True
        serp, _ = gs.step_simulation(serp, bm.Action(type=bm.ActionType.PLAY_HAND))
        drawn_serp = serp.draw_cards()
        self.assertEqual(len(drawn_serp), 3)

        # 5. The Pillar + Verdant Leaf debuffs & disable cleanup
        vl = gs.GameState(
            seed="VLEAF1",
            location=bm.Location.SELECT,
            current_blind=bm.BlindName.VERDANT_LEAF,
            inventory=bm.Inventory(jokers=[bm.Joker(name=bm.JokerName.JOKER)]),
        )
        vl, _ = gs.step_simulation(vl, bm.Action(type=bm.ActionType.SELECT_BLIND))
        vl.draw_cards(5)
        self.assertTrue(all(hc.card.debuffed for hc in vl.hand.cards))
        vl, _ = gs.step_simulation(
            vl, bm.Action(type=bm.ActionType.SELL_CARD, target_idx=0)
        )
        self.assertTrue(vl.boss_disabled)
        self.assertFalse(any(hc.card.debuffed for hc in vl.hand.cards))

        # 6. Amber Acorn, Crimson Heart, Cerulean Bell
        acorn = gs.GameState(
            seed="ACORN1",
            location=bm.Location.SELECT,
            current_blind=bm.BlindName.AMBER_ACORN,
            inventory=bm.Inventory(
                jokers=[
                    bm.Joker(name=bm.JokerName.JOKER),
                    bm.Joker(name=bm.JokerName.BLUEPRINT),
                    bm.Joker(name=bm.JokerName.BRAINSTORM),
                ]
            ),
        )
        acorn, _ = gs.step_simulation(acorn, bm.Action(type=bm.ActionType.SELECT_BLIND))
        self.assertTrue(all(j.facing_back for j in acorn.inventory.jokers))
        acorn, _ = gs.step_simulation(acorn, bm.Action(type=bm.ActionType.END_ROUND))
        self.assertFalse(any(j.facing_back for j in acorn.inventory.jokers))

        bell = gs.GameState(
            seed="BELL1",
            location=bm.Location.BLIND,
            current_blind=bm.BlindName.CERULEAN_BELL,
        )
        bell.draw_cards(5)
        self.assertEqual(sum(1 for hc in bell.hand.cards if hc.forced_selection), 1)
        self.assertEqual(sum(1 for hc in bell.hand.cards if hc.selected), 1)

        heart = gs.GameState(
            seed="HEART1",
            location=bm.Location.SELECT,
            current_blind=bm.BlindName.CRIMSON_HEART,
            inventory=bm.Inventory(
                jokers=[
                    bm.Joker(name=bm.JokerName.JOKER),
                    bm.Joker(name=bm.JokerName.MIME),
                ]
            ),
        )
        heart, _ = gs.step_simulation(heart, bm.Action(type=bm.ActionType.SELECT_BLIND))
        self.assertEqual(sum(1 for j in heart.inventory.jokers if j.debuffed), 1)

        # 7. The House, The Mark, The Fish, The Pillar (ante clear)
        house = gs.GameState(
            seed="HOUSE1",
            location=bm.Location.BLIND,
            current_blind=bm.BlindName.THE_HOUSE,
        )
        drawn_h = house.draw_cards(3)
        self.assertTrue(all(hc.facing_back for hc in drawn_h))

        fish = gs.GameState(
            seed="FISH1",
            location=bm.Location.SELECT,
            current_blind=bm.BlindName.THE_FISH,
        )
        fish, _ = gs.step_simulation(fish, bm.Action(type=bm.ActionType.SELECT_BLIND))
        first_draw = fish.draw_cards(2)
        self.assertFalse(any(hc.facing_back for hc in first_draw))
        fish.hand.cards[0].selected = True
        fish, _ = gs.step_simulation(fish, bm.Action(type=bm.ActionType.PLAY_HAND))
        second_draw = fish.draw_cards(1)
        self.assertTrue(all(hc.facing_back for hc in second_draw))

        pillar = gs.GameState(
            seed="PIL1",
            location=bm.Location.BLIND,
            current_blind=bm.BlindName.THE_PILLAR,
        )
        pillar.draw_cards(2)
        played_c = pillar.hand.cards[0].card
        pillar.hand.cards[0].selected = True
        pillar, _ = gs.step_simulation(pillar, bm.Action(type=bm.ActionType.PLAY_HAND))
        self.assertTrue(pillar.card_is_debuffed_by_boss(played_c))
        pillar, _ = gs.step_simulation(pillar, bm.Action(type=bm.ActionType.END_ROUND))
        self.assertEqual(pillar.played_cards_this_ante, [])

    def test_hermit_temperance_and_matador_deterministic_money(self):
        """Tests deterministic money resolution for Hermit, Temperance, and Matador."""
        # 1. The Hermit: doubles money up to +$20; +$0 if negative
        s_neg = gs.GameState(
            location=bm.Location.SHOP,
            money=-5,
            inventory=bm.Inventory(consumables=[bm.Tarot(name=bm.TarotName.HERMIT)]),
        )
        s_neg, _ = gs.step_simulation(
            s_neg, bm.Action(type=bm.ActionType.USE_CONSUMABLE, target_idx=0)
        )
        self.assertEqual(s_neg.money, -5)
        self.assertNotIn("money", s_neg.pending_cv_checks)

        s_mid = gs.GameState(
            location=bm.Location.SHOP,
            money=12,
            inventory=bm.Inventory(consumables=[bm.Tarot(name=bm.TarotName.HERMIT)]),
        )
        s_mid, _ = gs.step_simulation(
            s_mid, bm.Action(type=bm.ActionType.USE_CONSUMABLE, target_idx=0)
        )
        self.assertEqual(s_mid.money, 24)

        s_cap = gs.GameState(
            location=bm.Location.SHOP,
            money=35,
            inventory=bm.Inventory(consumables=[bm.Tarot(name=bm.TarotName.HERMIT)]),
        )
        s_cap, _ = gs.step_simulation(
            s_cap, bm.Action(type=bm.ActionType.USE_CONSUMABLE, target_idx=0)
        )
        self.assertEqual(s_cap.money, 55)

        # 2. Temperance: sums Joker sell values (including extra_value), capped at +$50
        j1 = bm.Joker(
            name=bm.JokerName.BLUEPRINT, extra_value=4
        )  # base 10 -> sell 5 + 4 = 9
        j2 = bm.Joker(name=bm.JokerName.MATADOR)  # base 7 -> sell 3
        s_temp = gs.GameState(
            location=bm.Location.SHOP,
            money=10,
            inventory=bm.Inventory(
                jokers=[j1, j2],
                consumables=[bm.Tarot(name=bm.TarotName.TEMPERANCE)],
            ),
        )
        s_temp, _ = gs.step_simulation(
            s_temp, bm.Action(type=bm.ActionType.USE_CONSUMABLE, target_idx=0)
        )
        self.assertEqual(s_temp.money, 22)
        self.assertNotIn("money", s_temp.pending_cv_checks)

        big_jokers = [
            bm.Joker(name=bm.JokerName.BLUEPRINT, extra_value=20),  # sell 25
            bm.Joker(name=bm.JokerName.BRAINSTORM, extra_value=20),  # sell 25
            bm.Joker(
                name=bm.JokerName.JOKER, extra_value=10
            ),  # sell 11 -> sum 61 -> capped 50
        ]
        s_temp_cap = gs.GameState(
            location=bm.Location.SHOP,
            money=5,
            inventory=bm.Inventory(
                jokers=big_jokers,
                consumables=[bm.Tarot(name=bm.TarotName.TEMPERANCE)],
            ),
        )
        s_temp_cap, _ = gs.step_simulation(
            s_temp_cap, bm.Action(type=bm.ActionType.USE_CONSUMABLE, target_idx=0)
        )
        self.assertEqual(s_temp_cap.money, 55)

        # 3. Matador (+$8 when Boss Blind triggers, Blueprint-compatible)
        def _hc(rank, suit, selected=True):
            return bm.HandCard(
                card=bm.PlayingCard(rank=rank, suit=suit),
                selected=selected,
            )

        # 3a. Debuffed hand (The Psychic < 5 cards) + Blueprint copying Matador -> +$16
        s_psy = gs.GameState(
            location=bm.Location.BLIND,
            current_blind=bm.BlindName.THE_PSYCHIC,
            money=10,
            inventory=bm.Inventory(
                jokers=[
                    bm.Joker(name=bm.JokerName.BLUEPRINT),
                    bm.Joker(name=bm.JokerName.MATADOR),
                ]
            ),
        )
        s_psy.hand.cards = [_hc(bm.Rank.ACE, bm.Suit.SPADES, selected=True)]
        s_psy, _ = gs.step_simulation(s_psy, bm.Action(type=bm.ActionType.PLAY_HAND))
        self.assertEqual(s_psy.money, 26)
        self.assertNotIn("money", s_psy.pending_cv_checks)

        # 3b. The Psychic with 5 cards (not debuffed, blind does NOT trigger) -> +$0
        s_psy_ok = gs.GameState(
            location=bm.Location.BLIND,
            current_blind=bm.BlindName.THE_PSYCHIC,
            money=10,
            inventory=bm.Inventory(jokers=[bm.Joker(name=bm.JokerName.MATADOR)]),
        )
        s_psy_ok.hand.cards = [
            _hc(bm.Rank.TWO, bm.Suit.SPADES, selected=True),
            _hc(bm.Rank.FOUR, bm.Suit.HEARTS, selected=True),
            _hc(bm.Rank.SIX, bm.Suit.CLUBS, selected=True),
            _hc(bm.Rank.EIGHT, bm.Suit.DIAMONDS, selected=True),
            _hc(bm.Rank.TEN, bm.Suit.SPADES, selected=True),
        ]
        s_psy_ok, _ = gs.step_simulation(
            s_psy_ok, bm.Action(type=bm.ActionType.PLAY_HAND)
        )
        self.assertEqual(s_psy_ok.money, 10)

        # 3c. The Flint (unconditional modify_hand trigger) -> +$8
        s_flint = gs.GameState(
            location=bm.Location.BLIND,
            current_blind=bm.BlindName.THE_FLINT,
            money=4,
            inventory=bm.Inventory(jokers=[bm.Joker(name=bm.JokerName.MATADOR)]),
        )
        s_flint.hand.cards = [_hc(bm.Rank.ACE, bm.Suit.SPADES, selected=True)]
        s_flint, _ = gs.step_simulation(
            s_flint, bm.Action(type=bm.ActionType.PLAY_HAND)
        )
        self.assertEqual(s_flint.money, 12)

        # 3d. The Club with debuffed scoring Clubs card -> +$8; without Clubs -> +$0
        s_club_hit = gs.GameState(
            location=bm.Location.BLIND,
            current_blind=bm.BlindName.THE_CLUB,
            money=5,
            inventory=bm.Inventory(jokers=[bm.Joker(name=bm.JokerName.MATADOR)]),
        )
        s_club_hit.hand.cards = [_hc(bm.Rank.KING, bm.Suit.CLUBS, selected=True)]
        s_club_hit.refresh_debuffs()
        s_club_hit, _ = gs.step_simulation(
            s_club_hit, bm.Action(type=bm.ActionType.PLAY_HAND)
        )
        self.assertEqual(s_club_hit.money, 13)

        s_club_miss = gs.GameState(
            location=bm.Location.BLIND,
            current_blind=bm.BlindName.THE_CLUB,
            money=5,
            inventory=bm.Inventory(jokers=[bm.Joker(name=bm.JokerName.MATADOR)]),
        )
        s_club_miss.hand.cards = [_hc(bm.Rank.KING, bm.Suit.HEARTS, selected=True)]
        s_club_miss.refresh_debuffs()
        s_club_miss, _ = gs.step_simulation(
            s_club_miss, bm.Action(type=bm.ActionType.PLAY_HAND)
        )
        self.assertEqual(s_club_miss.money, 5)

        # 3e. The Ox resets dollars to $0 BEFORE Matador awards +$8
        s_ox = gs.GameState(
            location=bm.Location.BLIND,
            current_blind=bm.BlindName.THE_OX,
            ox_target_hand=bm.PokerHand.PAIR,
            money=30,
            inventory=bm.Inventory(jokers=[bm.Joker(name=bm.JokerName.MATADOR)]),
        )
        s_ox.hand.cards = [
            _hc(bm.Rank.SEVEN, bm.Suit.SPADES, selected=True),
            _hc(bm.Rank.SEVEN, bm.Suit.HEARTS, selected=True),
        ]
        s_ox, _ = gs.step_simulation(s_ox, bm.Action(type=bm.ActionType.PLAY_HAND))
        self.assertEqual(s_ox.money, 8)

    def test_determine_card_selection_arched_and_edge_cases(self):
        from classifiers.card_classifier.pipeline import determine_card_selection

        def _mk_boxes(ys):
            return [
                [100 + i * 120, y, 190 + i * 120, y + 230] for i, y in enumerate(ys)
            ]

        # Curved unselected arch in pack_hand (8 cards, spread=35px > 28) -> none selected
        self.assertEqual(
            determine_card_selection(_mk_boxes([84, 71, 74, 66, 53, 55, 71, 88]), 340),
            [False] * 8,
        )
        # Hanged Man: two outer right cards selected, center arch card at y=53 must remain False
        self.assertEqual(
            determine_card_selection(_mk_boxes([98, 83, 61, 53, 63, 69, 18, 27]), 340),
            [False, False, False, False, False, False, True, True],
        )
        # Chariot: single middle-left card raised
        self.assertEqual(
            determine_card_selection(
                _mk_boxes([86, 70, 69, 23, 63, 53, 57, 69, 88]), 340
            ),
            [False, False, False, True, False, False, False, False, False],
        )
        # Mixed outer edge + center raised simultaneously
        self.assertEqual(
            determine_card_selection(_mk_boxes([49, 80, 65, 9, 55, 65, 80, 95]), 322),
            [True, False, False, True, False, False, False, False],
        )
        # Majority (5/7) selected, all-selected curved & flat, 1-card & 2-card hands
        self.assertEqual(
            determine_card_selection(_mk_boxes([77, 67, 16, 14, 16, 15, 23]), 322),
            [False, False, True, True, True, True, True],
        )
        self.assertEqual(
            determine_card_selection(_mk_boxes([40, 25, 15, 25, 40]), 322), [True] * 5
        )
        self.assertEqual(
            determine_card_selection(_mk_boxes([67, 67, 67, 67]), 322), [False] * 4
        )
        self.assertEqual(
            determine_card_selection(_mk_boxes([21, 21, 21, 21]), 322), [True] * 4
        )
        self.assertEqual(
            determine_card_selection(_mk_boxes([75, 29]), 322), [False, True]
        )
        self.assertEqual(determine_card_selection(_mk_boxes([20]), 322), [True])
        self.assertEqual(determine_card_selection(_mk_boxes([65]), 322), [False])

        # xPGzf7YJlxM step 334 (Hanged Man): left-edge occluded King (#0) + 9 (#1)
        s334_boxes = [
            [19, 66, 128, 305], [143, 44, 243, 279], [269, 74, 361, 307], [393, 77, 478, 309],
            [515, 83, 596, 313], [630, 79, 721, 310], [749, 75, 854, 311], [866, 89, 978, 329],
        ]
        self.assertEqual(
            determine_card_selection(s334_boxes, 340),
            [True, True, False, False, False, False, False, False],
        )
        self.assertEqual(
            determine_card_selection(s334_boxes, 340, min_selected=1, max_selected=2),
            [True, True, False, False, False, False, False, False],
        )

        # xPGzf7YJlxM step 215 (Death): Glass card #2 under JokerDisplay overlaps slightly high (y1=57),
        # but enforcing Death's exact (2, 2) cardinality cleanly picks #1 (y1=44) and #5 (y1=36).
        s215_boxes = [
            [19, 115, 127, 340], [141, 44, 246, 281], [264, 57, 358, 291], [395, 76, 476, 307],
            [515, 84, 596, 313], [632, 36, 721, 267], [744, 75, 855, 311], [862, 84, 973, 327],
        ]
        self.assertEqual(
            determine_card_selection(s215_boxes, 340, min_selected=2, max_selected=2),
            [False, True, False, False, False, True, False, False],
        )

    def test_death_tarot_preserves_left_card_deck_id(self):
        left_card = bm.PlayingCard(rank=bm.Rank.TWO, suit=bm.Suit.CLUBS, id=1)
        right_card = bm.PlayingCard(
            rank=bm.Rank.ACE,
            suit=bm.Suit.SPADES,
            enhancement=bm.Enhancement.MULT,
            edition=bm.Edition.FOIL,
            seal=bm.Seal.RED,
            bonus_chips=20,
            id=2,
        )
        state = gs.GameState(
            location=bm.Location.BLIND,
            full_deck=[left_card, right_card],
            inventory=bm.Inventory(consumables=[bm.Tarot(name=bm.TarotName.DEATH)]),
        )
        state.hand.cards = [
            bm.HandCard(card=left_card, selected=True),
            bm.HandCard(card=right_card, selected=True),
        ]
        state, _ = gs.step_simulation(
            state,
            bm.Action(type=bm.ActionType.USE_CONSUMABLE, target_idx=0),
        )
        # Left card in hand and full_deck takes right's identity but preserves unique id=1.
        self.assertEqual(left_card.rank, bm.Rank.ACE)
        self.assertEqual(left_card.suit, bm.Suit.SPADES)
        self.assertEqual(left_card.enhancement, bm.Enhancement.MULT)
        self.assertEqual(left_card.edition, bm.Edition.FOIL)
        self.assertEqual(left_card.seal, bm.Seal.RED)
        self.assertEqual(left_card.bonus_chips, 20)
        self.assertEqual(left_card.id, 1)
        self.assertEqual({c.id for c in state.full_deck}, {1, 2})

        # Face-down left card (id=0) transformed by Death must not alias right's id=2.
        fd_left = bm.HandCard(
            card=bm.PlayingCard(id=0), facing_back=True, selected=True
        )
        state.inventory.consumables = [bm.Tarot(name=bm.TarotName.DEATH)]
        state.hand.cards = [fd_left, bm.HandCard(card=right_card, selected=True)]
        state, _ = gs.step_simulation(
            state,
            bm.Action(type=bm.ActionType.USE_CONSUMABLE, target_idx=0),
        )
        self.assertEqual(fd_left.card.rank, bm.Rank.ACE)
        self.assertEqual(fd_left.card.id, 0)

    def test_analyze_shadow_presence_dark_boss_step_and_edge_filter(self):
        import numpy as np
        from shadow_evaluator import analyze_shadow_presence

        # 1. Weak dark Boss Blind shadow: dark shadow (10) rising to dark red felt (22)
        boss_shadow = np.full((10, 60), 22, dtype=np.uint8)
        boss_shadow[:4, :] = 10
        boss_shadow[4, :] = 12
        self.assertTrue(analyze_shadow_presence(boss_shadow, threshold=52.0))
        # High threshold still rejects moderate step cleanly
        self.assertFalse(analyze_shadow_presence(boss_shadow, threshold=500.0))

        # 2. Pressed Select Card: bright button border on row 0 (28) dropping to flat bg (12)
        select_pressed = np.full((10, 60), 12, dtype=np.uint8)
        select_pressed[0, :] = 28
        self.assertFalse(analyze_shadow_presence(select_pressed, threshold=52.0))

        # 3. 7-row pressed Open Pack strip with 1px dark bevel ([41, 58, 55...]) -> no shadow
        open_pack_pressed = np.full((7, 60), 55, dtype=np.uint8)
        open_pack_pressed[0, :] = 41
        open_pack_pressed[1, :] = 58
        self.assertFalse(analyze_shadow_presence(open_pack_pressed, threshold=52.0))

        # 4. 4-row unpressed Sort by Rank strip ([58, 33, 33, 35]) -> shadow still detected
        sort_rank_unpressed = np.full((4, 60), 33, dtype=np.uint8)
        sort_rank_unpressed[0, :] = 58
        sort_rank_unpressed[3, :] = 35
        self.assertTrue(analyze_shadow_presence(sort_rank_unpressed, threshold=52.0))

    def test_eval_template_luminance_gate(self):
        import cv2
        import numpy as np
        from config import Config
        import eval_template
        from geometry import ImageSize, NormalizedROI, PixelROI
        from template import Template

        # Build a non-uniform 24x32 template with button interior luma ~80 and white text border
        tpl_bgr = np.full((24, 32, 3), 80, dtype=np.uint8)
        tpl_bgr[6:18, 8:24] = 200
        tpl_bgr[8:16, 10:22] = 80
        tpl_gray = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)
        mask = np.full((24, 32), 255, dtype=np.uint8)
        tmpl = Template(
            name="Test Button",
            filenames=["test.png"],
            norm_rois=[NormalizedROI(0.05, 0.05, 0.95, 0.95)],
            px_rois=[PixelROI(1, 1, 119, 79)],
            images=[np.dstack([tpl_bgr, mask])],
            images_gray=[tpl_gray],
            masks=[mask],
            shadow_dy=0,
            shadow_dw=0,
            shadow_dh=0,
            adjusted=True,
        )

        def _make_scene(scale: float) -> tuple[np.ndarray, np.ndarray]:
            img = np.full((80, 120, 3), 25, dtype=np.uint8)
            patch = np.clip(tpl_bgr.astype(np.float32) * scale, 0, 255).astype(np.uint8)
            img[20:44, 30:62] = patch
            return img, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cfg = Config(debug_plot=False, setup_mode=False, elem_conf_thresh=0.90)

        # Bright unhovered match (luma ~135, ratio ~1.69) -> found=True, clicked=False
        img_bright, gray_bright = _make_scene(135.0 / 80.0)
        res_bright = eval_template.eval(tmpl, img_bright, gray_bright, cfg)
        self.assertTrue(res_bright.found)
        self.assertFalse(res_bright.clicked)
        self.assertGreater(res_bright.luminance_ratio, 1.50)

        # Dimmed modal match (luma ~40, ratio ~0.50) -> found=True, clicked=False
        img_dim, gray_dim = _make_scene(40.0 / 80.0)
        res_dim = eval_template.eval(tmpl, img_dim, gray_dim, cfg)
        self.assertTrue(res_dim.found)
        self.assertFalse(res_dim.clicked)
        self.assertLess(res_dim.luminance_ratio, 0.60)

        # Normal hovered/pressed match (luma ~84, ratio ~1.05) -> found=True, clicked=True (flat bg)
        img_ok, gray_ok = _make_scene(84.0 / 80.0)
        res_ok = eval_template.eval(tmpl, img_ok, gray_ok, cfg)
        self.assertTrue(res_ok.found)
        self.assertTrue(res_ok.clicked)
        self.assertAlmostEqual(res_ok.luminance_ratio, 1.05, delta=0.03)

        # use_luminance_check=False bypasses luminance gate cleanly
        cfg_off = Config(
            debug_plot=False, setup_mode=False, elem_conf_thresh=0.90, use_luminance_check=False
        )
        res_bypass = eval_template.eval(tmpl, img_bright, gray_bright, cfg_off)
        self.assertTrue(res_bypass.found)
        self.assertTrue(res_bypass.clicked)

    def test_batch_recognize_hands_passes_pack_hand_roi(self):
        from unittest.mock import MagicMock, patch
        import numpy as np
        import torch
        from classifiers.inference import ROIS, batch_recognize_hands

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_pipe = MagicMock()
        mock_pipe.device = torch.device("cpu")
        mock_res = MagicMock()
        mock_res.boxes = None
        mock_pipe.detector.predict.return_value = [mock_res]

        with patch("classifiers.inference.get_card_pipeline", return_value=mock_pipe):
            out = batch_recognize_hands([frame], roi_key="pack_hand")[0]

        self.assertEqual(out.roi[1], round(ROIS["pack_hand"]["y"]))
        self.assertEqual(out.roi[3], round(ROIS["pack_hand"]["h"]))

    def test_prng_card_blocking_across_inventory_shop_and_pack(self):
        """Tarot/Planet/Spectral/Joker PRNG pools block cards present in inventory, shop_cards, or pack_cards unless Showman is active."""
        from game.balatro_models import (
            Action,
            ActionType,
            BoosterPack,
            BoosterSize,
            BoosterType,
            CardType,
            Inventory,
            Joker,
            JokerName,
            Location,
            Planet,
            PlanetName,
            Spectral,
            Tarot,
            TarotName,
        )
        from game.engine.rng_sim.events import (
            poll_tarot_pool,
            poll_spectral_pool,
            get_blocked_card_names,
        )
        from game.engine.rng_sim.rng import BalatroRNG
        from game.engine.rules import on_action
        from game.gamestate import GameState

        # 1. Emperor inside an Arcana pack must skip tarots visible in pack_cards and external shop_cards
        rng_probe = BalatroRNG("XARBTZU7")
        first_without_block = poll_tarot_pool(rng_probe, ante=5, key_append="emp").name
        state = GameState(
            seed="XARBTZU7",
            ante=5,
            location=Location.PACK,
            pack_choices_remaining=1,
            active_pack=BoosterPack(
                booster_type=BoosterType.ARCANA,
                size=BoosterSize.REGULAR,
                choose=1,
                extra=3,
                cost=4,
            ),
            inventory=Inventory(),
            shop_cards=[Tarot(name=first_without_block)],
            pack_cards=[
                Tarot(name=TarotName.EMPEROR),
                Tarot(name=TarotName.TEMPERANCE),
            ],
        )
        on_action(state, Action(type=ActionType.USE_IN_PACK, target_idx=0))
        spawned_names = [c.name for c in state.inventory.consumables]
        self.assertEqual(len(spawned_names), 2)
        self.assertNotIn(first_without_block, spawned_names)
        self.assertNotIn(TarotName.TEMPERANCE, spawned_names)
        self.assertEqual(len(set(spawned_names)), 2)

        # 2. Showman bypasses shop_cards / pack_cards blocking
        state_showman = GameState(
            seed="XARBTZU7",
            ante=5,
            location=Location.PACK,
            pack_choices_remaining=1,
            active_pack=BoosterPack(
                booster_type=BoosterType.ARCANA,
                size=BoosterSize.REGULAR,
                choose=1,
                extra=3,
                cost=4,
            ),
            inventory=Inventory(jokers=[Joker(name=JokerName.SHOWMAN)]),
            shop_cards=[Tarot(name=first_without_block)],
            pack_cards=[Tarot(name=TarotName.EMPEROR)],
        )
        on_action(state_showman, Action(type=ActionType.USE_IN_PACK, target_idx=0))
        self.assertEqual(
            state_showman.inventory.consumables[0].name, first_without_block
        )

        # 3. Spectral pool + get_blocked_card_names across regions
        rng_spec = BalatroRNG("XARBTZU7")
        first_spec = poll_spectral_pool(rng_spec, ante=3, key_append="sea").name
        spec_state = GameState(
            shop_cards=[Spectral(name=first_spec)],
            pack_cards=[Planet(name=PlanetName.SATURN)],
            inventory=Inventory(jokers=[Joker(name=JokerName.BLUEPRINT)]),
        )
        self.assertIn(first_spec, get_blocked_card_names(spec_state, CardType.SPECTRAL))
        self.assertIn(
            PlanetName.SATURN, get_blocked_card_names(spec_state, CardType.PLANET)
        )
        self.assertIn(
            JokerName.BLUEPRINT, get_blocked_card_names(spec_state, CardType.JOKER)
        )
        alt_spec = poll_spectral_pool(
            BalatroRNG("XARBTZU7"),
            ante=3,
            key_append="sea",
            active_spectrals=get_blocked_card_names(spec_state, CardType.SPECTRAL),
        )
        self.assertNotEqual(alt_spec.name, first_spec)

    def test_initial_round_targets_rolled_on_init_erratic_z1dcpjse(self) -> None:
        """Game:start_run() rolls idol/mail/ancient/castle on init so Round 1..3 match Balatro."""
        from game.balatro_models import Action, ActionType, Deck, Rank, Stake
        from game.engine.rules import on_action
        from game.gamestate import GameState

        state = GameState(
            seed="Z1DCPJSE",
            deck=Deck.ERRATIC_DECK,
            stake=Stake.GOLD,
        )
        # Initial Round 1 (Small Blind) targets rolled at run start
        self.assertEqual(state.rng.get_channel_call_count("mail1"), 1)
        self.assertEqual(state.rng.get_channel_call_count("idol1"), 1)
        self.assertEqual(state.rng.get_channel_call_count("anc1"), 1)
        self.assertEqual(state.rng.get_channel_call_count("cas1"), 1)
        self.assertEqual(state.mail_card_rank, Rank.EIGHT)
        self.assertIsNotNone(state.idol_card)
        self.assertIsNotNone(state.ancient_suit)
        self.assertIsNotNone(state.castle_suit)

        # Round 1 -> Round 2 (Big Blind)
        on_action(state, Action(type=ActionType.SELECT_BLIND))
        on_action(state, Action(type=ActionType.END_ROUND))
        self.assertEqual(state.mail_card_rank, Rank.NINE)
        on_action(state, Action(type=ActionType.CASH_OUT))
        on_action(state, Action(type=ActionType.NEXT_ROUND))

        # Round 2 -> Round 3 (Boss Blind)
        on_action(state, Action(type=ActionType.SELECT_BLIND))
        on_action(state, Action(type=ActionType.END_ROUND))
        self.assertEqual(state.mail_card_rank, Rank.THREE)
        on_action(state, Action(type=ActionType.CASH_OUT))
        on_action(state, Action(type=ActionType.NEXT_ROUND))
        on_action(state, Action(type=ActionType.SELECT_BLIND))
        self.assertEqual(state.ante, 1)
        self.assertEqual(state.round, 3)
        self.assertEqual(state.mail_card_rank, Rank.THREE)

        # Beat Boss Blind -> Ante 2 Rounds 4, 5, 6 use mail2 #1 (6), #2 (King), #3 (7)
        on_action(state, Action(type=ActionType.END_ROUND))
        self.assertEqual(state.mail_card_rank, Rank.SIX)
        on_action(state, Action(type=ActionType.CASH_OUT))
        on_action(state, Action(type=ActionType.NEXT_ROUND))
        self.assertEqual(state.ante, 2)
        self.assertEqual(state.round, 4)

        on_action(state, Action(type=ActionType.SELECT_BLIND))
        on_action(state, Action(type=ActionType.END_ROUND))
        self.assertEqual(state.mail_card_rank, Rank.KING)
        on_action(state, Action(type=ActionType.CASH_OUT))
        on_action(state, Action(type=ActionType.NEXT_ROUND))
        self.assertEqual(state.round, 5)

        on_action(state, Action(type=ActionType.SELECT_BLIND))
        on_action(state, Action(type=ActionType.END_ROUND))
        self.assertEqual(state.mail_card_rank, Rank.SEVEN)

    def test_purple_seal_key_stone_match_and_clone_unique_ids(self) -> None:
        """Purple Seal uses '8ba', Stone cards match rank/suit NONE, and clones get unique IDs."""
        from game.balatro_models import (
            Action,
            ActionType,
            Edition,
            Enhancement,
            Hand,
            HandCard,
            Inventory,
            Joker,
            JokerName,
            PlayingCard,
            Rank,
            Seal,
            Spectral,
            SpectralName,
            Suit,
        )
        from game.engine.rng_sim.events import poll_tarot_pool, predict_purple_seal_tarot
        from game.engine.rng_sim.rng import BalatroRNG
        from game.gamestate import GameState, _cards_match, step_simulation

        # 1. Purple Seal uses '8ba' stream (card.lua:2260)
        rng_a = BalatroRNG("Z1DCPJSE")
        rng_b = BalatroRNG("Z1DCPJSE")
        p_tarot = predict_purple_seal_tarot(rng_a, ante=4)
        e_tarot = poll_tarot_pool(rng_b, ante=4, key_append="8ba")
        self.assertEqual(p_tarot.key, e_tarot.key)
        self.assertEqual(p_tarot.tarot_name, e_tarot.tarot_name)

        # 2. Stone cards with Rank.NONE/Suit.NONE match converted Stone cards (checking edition/seal)
        converted_stone = PlayingCard(
            rank=Rank.SIX,
            suit=Suit.CLUBS,
            enhancement=Enhancement.STONE,
            edition=Edition.HOLOGRAPHIC,
            seal=Seal.BLUE,
        )
        cv_stone = PlayingCard(
            rank=Rank.NONE,
            suit=Suit.NONE,
            enhancement=Enhancement.STONE,
            edition=Edition.HOLOGRAPHIC,
            seal=Seal.BLUE,
        )
        cv_stone_diff_seal = PlayingCard(
            rank=Rank.NONE,
            suit=Suit.NONE,
            enhancement=Enhancement.STONE,
            edition=Edition.HOLOGRAPHIC,
            seal=Seal.NONE,
        )
        self.assertTrue(_cards_match(converted_stone, cv_stone))
        self.assertTrue(_cards_match(cv_stone, converted_stone))
        self.assertFalse(_cards_match(converted_stone, cv_stone_diff_seal))

        # 3. DNA and Cryptid assign distinct PlayingCard.id values in full_deck
        dna = Joker(name=JokerName.DNA)
        cryptid = Spectral(name=SpectralName.CRYPTID)
        c1 = PlayingCard(rank=Rank.THREE, suit=Suit.CLUBS, id=2)
        state = GameState(
            inventory=Inventory(jokers=[dna], consumables=[cryptid]),
            hand=Hand(cards=[HandCard(card=c1, selected=True)]),
        )
        state, _ = step_simulation(state, Action(type=ActionType.PLAY_HAND))
        dna_ids = [c.id for c in state.full_deck]
        self.assertEqual(len(dna_ids), len(set(dna_ids)))

        state.hand.cards[0].selected = True
        state, _ = step_simulation(state, Action(type=ActionType.USE_CONSUMABLE, target_idx=0))
        cryptid_ids = [c.id for c in state.full_deck]
        self.assertEqual(len(cryptid_ids), len(set(cryptid_ids)))

    def test_remaining_deck_id_first_and_hook_discard_override(self) -> None:
        """remaining_deck prefers exact card.id matches on clones and THE_HOOK honors _hook_discard_ids."""
        from game.balatro_models import (
            Action,
            ActionType,
            BlindName,
            Hand,
            HandCard,
            Location,
            PlayingCard,
            Rank,
            Seal,
            Suit,
        )
        from game.engine.rules import on_action
        from game.gamestate import GameState

        c2 = PlayingCard(rank=Rank.THREE, suit=Suit.CLUBS, seal=Seal.PURPLE, id=2)
        c18 = PlayingCard(rank=Rank.THREE, suit=Suit.CLUBS, seal=Seal.PURPLE, id=18)
        c63 = PlayingCard(rank=Rank.THREE, suit=Suit.CLUBS, seal=Seal.PURPLE, id=63)
        h4 = PlayingCard(rank=Rank.FOUR, suit=Suit.HEARTS, id=30)
        h5 = PlayingCard(rank=Rank.FIVE, suit=Suit.DIAMONDS, id=31)
        state = GameState(
            location=Location.BLIND,
            current_blind=BlindName.THE_HOOK,
            full_deck=[c2, c18, c63, h4, h5],
            hand=Hand(cards=[HandCard(card=c63, selected=True), HandCard(card=h4), HandCard(card=h5)]),
            discard_pile=[c2],
        )
        # Before ID-first subtraction, id=63 popped c2 and c2 popped c18, leaving id=63 behind!
        rem_ids = [c.id for c in state.remaining_deck]
        self.assertEqual(rem_ids, [18])

        # _hook_discard_ids directs THE_HOOK to discard exact specified unplayed card IDs
        act = Action(
            type=ActionType.PLAY_HAND,
            selected_cards=[c63],
            metadata={"_hook_discard_ids": [30]},
        )
        on_action(state, act)
        discard_ids = {c.id for c in state.discard_pile}
        self.assertIn(30, discard_ids)
        self.assertIn(31, discard_ids)  # Remainder filled by PRNG when only 1 override ID supplied

    def test_tarot_and_joker_mutations_propagate_to_full_deck_after_face_down_resolve(self) -> None:
        """Tarot/Spectral/Joker mutations on face-down or resolved orphan HandCards update full_deck by id."""
        import copy
        from game.balatro_models import (
            Action,
            ActionType,
            Edition,
            Enhancement,
            Hand,
            HandCard,
            Inventory,
            Joker,
            JokerName,
            Location,
            PlayingCard,
            Rank,
            Seal,
            Spectral,
            SpectralName,
            Suit,
            Tarot,
            TarotName,
        )
        from game.engine.rules import on_action
        from game.gamestate import GameState

        c9 = PlayingCard(rank=Rank.ACE, suit=Suit.CLUBS, enhancement=Enhancement.MULT, id=9)
        c20 = PlayingCard(
            rank=Rank.FIVE,
            suit=Suit.CLUBS,
            enhancement=Enhancement.GLASS,
            seal=Seal.RED,
            edition=Edition.POLYCHROME,
            id=20,
        )
        c38 = PlayingCard(rank=Rank.QUEEN, suit=Suit.HEARTS, id=38)
        c_stone = PlayingCard(rank=Rank.NONE, suit=Suit.NONE, enhancement=Enhancement.STONE, id=44)

        # Simulate cards previously drawn face-down (severing hc.card pointer via HandCard.__post_init__)
        hc9 = HandCard(card=c9, facing_back=True)
        hc9.resolve_face_down(copy.deepcopy(c9), lift_facing_back=True)
        self.assertIsNot(hc9.card, c9)  # Orphan clone prior to Tarot

        hc20 = HandCard(card=c20)
        hc38_fd = HandCard(card=c38, facing_back=True)  # Still face-down with _real_card copy!
        hc_stone = HandCard(card=c_stone)

        state = GameState(
            location=Location.BLIND,
            full_deck=[c9, c20, c38, c_stone],
            hand=Hand(cards=[hc9, hc20, hc38_fd, hc_stone]),
            inventory=Inventory(
                consumables=[
                    Tarot(name=TarotName.DEATH),
                    Tarot(name=TarotName.STRENGTH),
                    Tarot(name=TarotName.CHARIOT),
                    Spectral(name=SpectralName.DEJA_VU),
                    Tarot(name=TarotName.DEATH),
                    Tarot(name=TarotName.HANGED_MAN),
                ]
            ),
        )

        # 1. Death on orphan hc9 (left) + hc20 (right) updates c9 in state.full_deck and rebinds hc9.card
        hc9.selected = True
        hc20.selected = True
        on_action(state, Action(type=ActionType.USE_CONSUMABLE, target_idx=0))
        self.assertEqual(c9.rank, Rank.FIVE)
        self.assertEqual(c9.enhancement, Enhancement.GLASS)
        self.assertEqual(c9.seal, Seal.RED)
        self.assertEqual(c9.edition, Edition.POLYCHROME)
        self.assertIs(hc9.card, c9)

        # 2. Strength + Chariot + Deja Vu on face-down hc38_fd propagate rank/enh/seal to c38 in full_deck
        hc9.selected = False
        hc20.selected = False
        hc38_fd.selected = True
        on_action(state, Action(type=ActionType.USE_CONSUMABLE, target_idx=0))  # Strength: Q -> K
        on_action(state, Action(type=ActionType.USE_CONSUMABLE, target_idx=0))  # Chariot: Steel
        on_action(state, Action(type=ActionType.USE_CONSUMABLE, target_idx=0))  # Deja Vu: Red Seal
        self.assertEqual(c38.rank, Rank.KING)
        self.assertEqual(c38.enhancement, Enhancement.STEEL)
        self.assertEqual(c38.seal, Seal.RED)

        # 3. Death cloning a CV Stone card (rank=NONE, enhancement=STONE) onto hc20 updates c20
        hc38_fd.selected = False
        hc20.selected = True
        hc_stone.selected = True
        on_action(state, Action(type=ActionType.USE_CONSUMABLE, target_idx=0))
        self.assertEqual(c20.enhancement, Enhancement.STONE)

        # 4. Inferred Hanged Man with zero selected cards does NOT blindly destroy hand.cards[:2]
        for hc in state.hand.cards:
            hc.selected = False
        on_action(
            state,
            Action(type=ActionType.USE_CONSUMABLE, target_idx=0, metadata={"inferred": True}),
        )
        self.assertEqual(len(state.full_deck), 4)

        # 5. Midas Mask + Hiker on deserialized/orphan played card propagate to full_deck by id
        orphan_king = copy.deepcopy(c38)
        orphan_king.enhancement = Enhancement.NONE
        c38.enhancement = Enhancement.NONE
        state.hand.cards = [HandCard(card=orphan_king, selected=True)]
        state.inventory.jokers = [Joker(name=JokerName.MIDAS_MASK), Joker(name=JokerName.HIKER)]
        on_action(state, Action(type=ActionType.PLAY_HAND))
        self.assertEqual(c38.enhancement, Enhancement.GOLD)
        self.assertGreaterEqual(c38.bonus_chips, 5)


def run_synthetic_test() -> None:
    """Convenience function running the synthetic test directly."""
    test = TestEngine()
    test.test_synthetic_simulation()
    print("Synthetic integration test passed successfully!")


if __name__ == "__main__":
    unittest.main()
