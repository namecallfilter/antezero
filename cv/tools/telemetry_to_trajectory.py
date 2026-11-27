"""Convert a Lua `run_telemetry.jsonl` log into a 1:1 schema-v2 `*_trajectory.json`.

Replays the telemetry actions through the exact same `GameState` + `on_action`
+ `RunTrajectory` + `serialize_trajectory` pipeline used by `post_processing.py`,
modeling an ideal zero-error CV/OCR pipeline for ground-truth trajectory
comparison. Also handles concatenated/contaminated JSONL files by segmenting
individual runs cleanly.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import game.balatro_models as bm
import game.gamestate as gs
from game.engine.cv_stubs import _BOOSTER_SIZE_MAP, _BOOSTER_TYPE_MAP
from game.engine.jokers import on_joker_added
from game.engine.rng_sim.events import TAG_BY_KEY, VOUCHER_BY_KEY
from game.engine.rules import on_action
from game.trajectory_codec import serialize_trajectory
from post_processing import (
    FACE_DOWN_BOSS_BLINDS,
    _resolve_target_card_name,
    _serialize_shop_shelves,
)
from testing.test_engine import (
    BLIND_BY_STR,
    POKER_HAND_BY_STR,
    RANK_MAP,
    SUIT_MAP,
    VOUCHER_NAME_BY_STR,
    parse_any_card,
    parse_hand_card,
    parse_joker,
    parse_lua_action,
    parse_playing_card,
    sync_cv_observations_before_action,
)


def _parse_voucher_name(raw: Any) -> bm.VoucherName | None:
    if not raw:
        return None
    s = str(raw).strip()
    if s in VOUCHER_BY_KEY:
        vn = VOUCHER_BY_KEY[s].name
        return vn if isinstance(vn, bm.VoucherName) else bm.VoucherName(vn)
    if s in VOUCHER_NAME_BY_STR:
        return VOUCHER_NAME_BY_STR[s]
    return None


def _parse_skip_tag(raw: Any) -> bm.SkipTag | None:
    if not raw:
        return None
    s = str(raw).strip()
    if s in TAG_BY_KEY:
        return TAG_BY_KEY[s].skip_tag
    try:
        return bm.SkipTag(s)
    except ValueError:
        return None


def _parse_booster_pack(d: dict[str, Any]) -> bm.BoosterPack | None:
    if not isinstance(d, dict):
        return None
    text = f"{d.get('name', '')} {d.get('key', '')}".lower()
    b_type = next((v for k, v in _BOOSTER_TYPE_MAP.items() if k in text), None)
    if b_type is None:
        return None
    b_size = next(
        (v for k, v in _BOOSTER_SIZE_MAP.items() if k in text),
        bm.BoosterSize.REGULAR,
    )
    # `d["cost"]` in Lua telemetry is the on-screen (possibly discounted) price, whereas
    # `GameState` / `rules.py` applies `discount_percent` onto the base pack cost.
    base_cost = {
        bm.BoosterSize.REGULAR: 4,
        bm.BoosterSize.JUMBO: 6,
        bm.BoosterSize.MEGA: 8,
    }[b_size]
    raw_cost = d.get("cost")
    is_couponed = bool(d.get("couponed", False)) or (raw_cost == 0)
    return bm.BoosterPack(
        booster_type=b_type,
        size=b_size,
        choose=int(d.get("choose", 1)),
        extra=int(d.get("extra", 3)),
        cost=base_cost,
        couponed=is_couponed,
    )


def _parse_stake(raw: Any) -> bm.Stake:
    if isinstance(raw, int) and 1 <= raw <= len(gs.STAKE_ORDER):
        return gs.STAKE_ORDER[raw - 1]
    if isinstance(raw, str):
        for st in bm.Stake:
            if st.value.lower() == raw.lower() or st.name.lower() == raw.lower():
                return st
    return bm.Stake.WHITE


def _parse_deck(raw: Any) -> bm.Deck:
    try:
        return bm.Deck(str(raw or bm.Deck.RED_DECK.value))
    except ValueError:
        return bm.Deck.RED_DECK


def load_telemetry_entries(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                entries.append(json.loads(stripped))
    return entries


def split_telemetry_runs(entries: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Segments a possibly concatenated/polluted JSONL log into distinct runs."""
    if not entries:
        return []
    runs: list[list[dict[str, Any]]] = [[entries[0]]]
    for entry in entries[1:]:
        prev = runs[-1][-1]
        ps = prev.get("state", {})
        cs = entry.get("state", {})
        is_new = False
        if (
            entry.get("run_id")
            and prev.get("run_id")
            and entry["run_id"] != prev["run_id"]
        ):
            is_new = True
        elif int(entry.get("step", 0)) <= int(prev.get("step", 0)):
            is_new = True
        elif (
            cs.get("seed") != ps.get("seed")
            or cs.get("deck_name") != ps.get("deck_name")
            or cs.get("stake") != ps.get("stake")
        ):
            is_new = True
        elif entry.get("action") == "Setup Test Suite":
            is_new = True
        elif (
            entry.get("run_id")
            and prev.get("run_id")
            and entry["run_id"] == prev["run_id"]
        ):
            # Same live run_id with strictly increasing step (e.g. Hieroglyph/Petroglyph ante -1)
            is_new = False
        else:
            p_ante = int(ps.get("ante", 1))
            c_ante = int(cs.get("ante", 1))
            c_hands = int(cs.get("hands_left", 0))
            c_base_h = int(cs.get("base_hands", 4))
            c_disc = int(cs.get("discards_left", 0))
            c_base_d = int(cs.get("base_discards", 3))
            # Catch dataset-generator contamination (allow -1 ante for Hieroglyph/Petroglyph)
            if (
                c_ante < p_ante - 1
                or c_ante > p_ante + 1
                or c_hands > c_base_h + 8
                or c_disc > c_base_d + 8
                or int(cs.get("round", 0)) < int(ps.get("round", 0))
            ):
                is_new = True
        if is_new:
            runs.append([entry])
        else:
            runs[-1].append(entry)
    return runs


def select_run(
    runs: list[list[dict[str, Any]]],
    *,
    run_index: int | None = None,
    seed: str | None = None,
) -> list[dict[str, Any]]:
    if not runs:
        raise ValueError("Telemetry file contains no entries.")
    candidates = runs
    if seed:
        candidates = [
            r
            for r in runs
            if str(r[0].get("state", {}).get("seed", "")).upper() == seed.upper()
        ]
        if not candidates:
            raise ValueError(f"No telemetry run found matching seed {seed!r}.")
    if run_index is not None:
        return candidates[run_index]

    # Auto-pick the longest run starting from step 1 at ante 1 (prefer real starts over leftover generator fragments)
    def _score(run: list[dict[str, Any]]) -> tuple[int, int, int]:
        s0 = run[0].get("state", {})
        starts_at_ante1 = 1 if int(s0.get("ante", 1)) == 1 else 0
        non_synthetic = (
            1 if int(s0.get("hands_left", 4)) <= int(s0.get("base_hands", 4)) + 4 else 0
        )
        return (non_synthetic, starts_at_ante1, len(run))

    return max(candidates, key=_score)


def _sync_extra_lua_fields(
    state: gs.GameState, lua_state: dict[str, Any], *, is_mid_setup: bool = False
) -> None:
    """Syncs screen-visible & pack/shop/tag metadata that `sync_cv_observations_before_action` skips."""
    if "round_won" in lua_state:
        state.round_won = bool(lua_state["round_won"])
    if lua_state.get("pre_pack_location") in ("shop", "select"):
        state.pre_pack_location = bm.Location(lua_state["pre_pack_location"])
    if isinstance(lua_state.get("active_pack"), dict):
        ap = _parse_booster_pack(lua_state["active_pack"])
        if ap is not None:
            state.active_pack = ap

    if "shop_packs" in lua_state:
        packs: list[bm.BoosterPack] = []
        for raw_bp in lua_state["shop_packs"]:
            bp = _parse_booster_pack(raw_bp)
            if bp is not None:
                packs.append(bp)
        state.shop_packs = packs

    if "small_blind_tag" in lua_state:
        tag = _parse_skip_tag(lua_state["small_blind_tag"])
        if tag is not None:
            state.small_blind_tag = tag
    if "big_blind_tag" in lua_state:
        tag = _parse_skip_tag(lua_state["big_blind_tag"])
        if tag is not None:
            state.big_blind_tag = tag

    if "joker_slots" in lua_state and state.inventory is not None:
        state.inventory.joker_slots = int(lua_state["joker_slots"])
    if "consumable_slots" in lua_state and state.inventory is not None:
        state.inventory.consumable_slots = int(lua_state["consumable_slots"])
    if "hand_max_size" in lua_state:
        state.hand.max_size = int(lua_state["hand_max_size"])

    if (
        "ox_target_hand" in lua_state
        and lua_state["ox_target_hand"] in POKER_HAND_BY_STR
    ):
        state.ox_target_hand = POKER_HAND_BY_STR[lua_state["ox_target_hand"]]
    if (
        "locked_hand_type" in lua_state
        and lua_state["locked_hand_type"] in POKER_HAND_BY_STR
    ):
        state.locked_hand_type = POKER_HAND_BY_STR[lua_state["locked_hand_type"]]
    if (
        "small_blind_orbital_hand" in lua_state
        and lua_state["small_blind_orbital_hand"] in POKER_HAND_BY_STR
    ):
        state.small_blind_orbital_hand = POKER_HAND_BY_STR[
            lua_state["small_blind_orbital_hand"]
        ]
    if (
        "big_blind_orbital_hand" in lua_state
        and lua_state["big_blind_orbital_hand"] in POKER_HAND_BY_STR
    ):
        state.big_blind_orbital_hand = POKER_HAND_BY_STR[
            lua_state["big_blind_orbital_hand"]
        ]

    if "ante_scaling" in lua_state:
        state.ante_scaling = float(lua_state["ante_scaling"])
    if "temp_hand_size_bonus" in lua_state:
        state.temp_hand_size_bonus = int(lua_state["temp_hand_size_bonus"])
    if "gros_michel_extinct" in lua_state:
        state.gros_michel_extinct = bool(lua_state["gros_michel_extinct"])

    if is_mid_setup and "vouchers" in lua_state:
        v_list = [
            vn
            for v in lua_state["vouchers"]
            if (vn := _parse_voucher_name(v)) is not None
        ]
        if v_list:
            state.vouchers_redeemed = v_list

    # Preserve existing full_deck PlayingCard instances by card.id inside hand
    deck_by_id = {c.id: c for c in state.full_deck if c.id != 0}
    for hc in state.hand.cards:
        if hc.card.id in deck_by_id:
            deck_card = deck_by_id[hc.card.id]
            deck_card.rank = hc.card.rank
            deck_card.suit = hc.card.suit
            deck_card.enhancement = hc.card.enhancement
            deck_card.edition = hc.card.edition
            deck_card.seal = hc.card.seal
            deck_card.bonus_chips = hc.card.bonus_chips
            deck_card.debuffed = hc.card.debuffed
            hc.card = deck_card


def _build_hud_ocr_meta(lua_state: dict[str, Any], loc: bm.Location) -> dict[str, int]:
    hud: dict[str, int] = {
        "hud_money": int(lua_state.get("money", 4)),
        "hud_hands": int(lua_state.get("hands_left", 4)),
        "hud_discards": int(lua_state.get("discards_left", 3)),
        "hud_ante": int(lua_state.get("ante", 1)),
        "hud_round": max(1, int(lua_state.get("round", 1))),
    }
    if loc == bm.Location.BLIND:
        hud["hud_round_score"] = int(lua_state.get("current_chips", 0))
        hud["hud_blind_target"] = int(lua_state.get("target_chips", 0))
    elif loc == bm.Location.SHOP:
        hud["hud_reroll_cost"] = int(lua_state.get("reroll_cost", 5))
    return hud


def _check_post_step_parity(
    step_label: str,
    act_name: str,
    sim_state: gs.GameState,
    next_lua: dict[str, Any],
    *,
    reconcile_truth: bool,
) -> list[str]:
    warnings: list[str] = []
    if "money" in next_lua and sim_state.money != int(next_lua["money"]):
        warnings.append(
            f"{step_label} money: sim={sim_state.money} vs lua={next_lua['money']}"
        )
        if reconcile_truth:
            sim_state.money = int(next_lua["money"])
    # Python keeps hands_left/discards_left until CASH_OUT & resets on SELECT_BLIND,
    # whereas Lua resets them immediately during ROUND_EVAL / SHOP.
    next_loc = next_lua.get("location")
    if (
        next_loc == "blind"
        and not next_lua.get("round_won")
        and next_lua.get("blind", "") != ""
    ):
        if "hands_left" in next_lua and sim_state.hands_left != int(
            next_lua["hands_left"]
        ):
            warnings.append(
                f"{step_label} hands_left: sim={sim_state.hands_left} vs lua={next_lua['hands_left']}"
            )
            if reconcile_truth:
                sim_state.hands_left = int(next_lua["hands_left"])
        if "discards_left" in next_lua and sim_state.discards_left != int(
            next_lua["discards_left"]
        ):
            warnings.append(
                f"{step_label} discards_left: sim={sim_state.discards_left} vs lua={next_lua['discards_left']}"
            )
            if reconcile_truth:
                sim_state.discards_left = int(next_lua["discards_left"])
    if (
        act_name in ("Reroll Shop", "Cash Out")
        and "reroll_cost" in next_lua
        and sim_state.reroll_cost != int(next_lua["reroll_cost"])
    ):
        warnings.append(
            f"{step_label} reroll_cost: sim={sim_state.reroll_cost} vs lua={next_lua['reroll_cost']}"
        )
        if reconcile_truth:
            sim_state.reroll_cost = int(next_lua["reroll_cost"])
    if "deck" in next_lua and next_lua["deck"] is not None:
        lua_deck_len = len(next_lua["deck"])
        if len(sim_state.full_deck) != lua_deck_len:
            warnings.append(
                f"{step_label} full_deck count: sim={len(sim_state.full_deck)} vs lua={lua_deck_len}"
            )
            if reconcile_truth:
                sim_state.full_deck = [parse_playing_card(c) for c in next_lua["deck"]]
                sim_state.deck_cards_count = len(sim_state.full_deck)
        elif reconcile_truth:
            # Reconcile exact RNG card mutations (enhancements, seals, editions, chip upgrades, ranks/suits)
            lua_cards = [parse_playing_card(c) for c in next_lua["deck"]]
            sim_state.full_deck = lua_cards
            sim_state.deck_cards_count = len(lua_cards)

    if reconcile_truth and "jokers" in next_lua:
        parsed_j = [
            pj for d in next_lua["jokers"] if (pj := parse_joker(d)) is not None
        ]
        if [j.name for j in sim_state.inventory.jokers] != [j.name for j in parsed_j]:
            warnings.append(
                f"{step_label} jokers: sim={[j.name.value for j in sim_state.inventory.jokers]} vs lua={[j.name.value for j in parsed_j]}"
            )
        sim_state.inventory.jokers = parsed_j
    if reconcile_truth and "consumables" in next_lua:
        parsed_c = [
            pc for d in next_lua["consumables"] if (pc := parse_any_card(d)) is not None
        ]
        sim_state.inventory.consumables = parsed_c
    if reconcile_truth and "vouchers" in next_lua:
        parsed_v = [
            vn
            for v in next_lua["vouchers"]
            if (vn := _parse_voucher_name(v)) is not None
        ]
        if parsed_v:
            sim_state.vouchers_redeemed = parsed_v
    return warnings


def build_trajectory_from_telemetry(
    entries: list[dict[str, Any]],
    *,
    inject_end_round: bool = True,
    keep_all_sorts: bool = False,
    reconcile_truth: bool = True,
) -> tuple[gs.GameState, gs.RunTrajectory, list[str]]:
    """Converts a single run's telemetry entries into `(final_state, RunTrajectory, warnings)`."""
    if not entries:
        raise ValueError("Cannot build trajectory from empty telemetry entries.")

    first_entry = entries[0]
    first_state = first_entry.get("state", {})
    seed = str(first_state.get("seed", ""))
    deck_enum = _parse_deck(first_state.get("deck_name"))
    stake_enum = _parse_stake(first_state.get("stake", 1))
    starts_mid_run = (
        first_entry.get("action") == "Setup Test Suite"
        or int(first_state.get("ante", 1)) > 1
        or first_state.get("location", "select") != "select"
    )

    first_loc = bm.Location(first_state.get("location", "select"))
    first_blind_raw = first_state.get("blind", "")
    first_blind = BLIND_BY_STR.get(first_blind_raw, bm.BlindName.SMALL)

    state = gs.GameState(
        step_idx=0,
        frame_idx=0,
        timestamp=0.0,
        seed=seed,
        deck=deck_enum,
        stake=stake_enum,
        ante=int(first_state.get("ante", 1)) if starts_mid_run else 1,
        round=max(1, int(first_state.get("round", 1))) if starts_mid_run else 1,
        current_blind=first_blind,
        location=first_loc,
        apply_deck_stake_on_init=not starts_mid_run,
    )
    if starts_mid_run:
        state.money = int(first_state.get("money", 4))
        if first_state.get("deck"):
            state.full_deck = [parse_playing_card(c) for c in first_state["deck"]]
            state.deck_cards_count = len(state.full_deck)
            state.starting_deck_size = len(state.full_deck)
        sync_cv_observations_before_action(
            state, first_state, act_name=first_entry.get("action", "")
        )
        _sync_extra_lua_fields(state, first_state, is_mid_setup=True)
    else:
        # Step 0 matches post_processing.py baseline initial state (before first blind selection).
        saved_target = state.target_chips
        sync_cv_observations_before_action(
            state, first_state, act_name=first_entry.get("action", "")
        )
        _sync_extra_lua_fields(state, first_state, is_mid_setup=False)
        if state.target_chips == 0 and saved_target > 0:
            state.target_chips = saved_target

    trajectory = gs.RunTrajectory()
    trajectory.record_step(state, action=None)

    base_ts = float(first_entry.get("timestamp", 0.0))
    warnings: list[str] = []

    i = 0
    while i < len(entries):
        entry = entries[i]
        act_name = entry.get("action", "Play Hand")
        meta = dict(entry.get("action_metadata") or {})
        lua_state = entry.get("state", {})
        step_num = int(entry.get("frame_idx") or entry.get("step", i + 1))
        rel_ts = max(0.0, float(entry.get("timestamp", 0.0)) - base_ts)

        saved_target = state.target_chips
        prev_boss_blind = state.boss_blind
        prev_reroll_cost = state.reroll_cost
        sync_cv_observations_before_action(state, lua_state, act_name=act_name)
        _sync_extra_lua_fields(
            state, lua_state, is_mid_setup=(act_name == "Setup Test Suite")
        )
        lua_ante = int(lua_state.get("ante", state.ante))
        # In post-boss shop/pack screens, Lua has already rolled the next ante's boss while Python waits for Next Round.
        if prev_boss_blind is not None and lua_ante != state.ante:
            state.boss_blind = prev_boss_blind
        if state.location != bm.Location.SHOP and prev_reroll_cost is not None:
            state.reroll_cost = prev_reroll_cost
        if state.target_chips == 0 and saved_target > 0:
            state.target_chips = saved_target

        action = parse_lua_action(act_name, meta, frame_idx=step_num, timestamp=rel_ts)
        is_face_down_boss = (
            state.current_blind in FACE_DOWN_BOSS_BLINDS and not state.boss_disabled
        )
        if (
            not keep_all_sorts
            and action.type in (bm.ActionType.SORT_RANK, bm.ActionType.SORT_SUIT)
            and not is_face_down_boss
        ):
            i += 1
            continue

        # Clear stale hands on blind/cash-out transitions identically to post_processing.py:1886
        if action.type in (
            bm.ActionType.SELECT_BLIND,
            bm.ActionType.SKIP_BLIND,
            bm.ActionType.CASH_OUT,
        ):
            state.hand.cards.clear()

        if lua_state.get("hand"):
            action.hand_order = [hc.card.id for hc in state.hand.cards]
        if action.type == bm.ActionType.PLAY_HAND:
            action.selected_cards = [
                copy.deepcopy(hc._real_card if hc._real_card is not None else hc.card)
                for hc in state.hand.cards
                if hc.selected
            ]
        else:
            action.selected_cards = [
                copy.deepcopy(hc.card) for hc in state.hand.cards if hc.selected
            ]

        if "target_card" not in action.metadata:
            tname = _resolve_target_card_name(state, action)
            if tname is not None:
                action.metadata["target_card"] = tname
            elif isinstance(meta.get("card"), str):
                action.metadata["target_card"] = meta["card"]

        if state.location == bm.Location.SHOP:
            action.metadata["observed_shop"] = _serialize_shop_shelves(state)
        elif state.location == bm.Location.PACK and state.pack_cards:
            from game.trajectory_codec import _serialize_card

            action.metadata["observed_pack"] = [
                _serialize_card(c) for c in state.pack_cards
            ]
        action.metadata["hud_ocr"] = _build_hud_ocr_meta(lua_state, state.location)

        next_lua = entries[i + 1].get("state", {}) if i + 1 < len(entries) else None
        next_act = entries[i + 1].get("action", "") if i + 1 < len(entries) else ""

        fold_hook_callback = (
            act_name == "Play Hand"
            and state.current_blind == bm.BlindName.THE_HOOK
            and not state.boss_disabled
            and next_act == "Discard"
            and next_lua is not None
            and int(next_lua.get("hands_left", 0)) == (state.hands_left or 0) - 1
        )
        if fold_hook_callback:
            action.metadata["defer_play_for_hook"] = True
        elif (
            act_name == "Discard"
            and getattr(state, "_hook_pending_cards", None) is not None
        ):
            action.metadata["hook"] = True

        if act_name == "Select Blind" and next_lua is not None and "jokers" in next_lua:
            curr_ids = {
                int(d.get("id", 0))
                for d in lua_state.get("jokers", [])
                if d.get("id")
            }
            spawned = [
                pj
                for d in next_lua["jokers"]
                if int(d.get("id", 0)) not in curr_ids
                and (pj := parse_joker(d)) is not None
            ]
            if spawned:
                action.metadata["riff_raff_spawned"] = spawned

        state.frame_idx = action.frame_idx
        state.timestamp = action.timestamp
        prev_loc = state.location
        on_action(state, action)

        if fold_hook_callback:
            action.metadata.pop("defer_play_for_hook", None)
            hook_entry = entries[i + 1]
            hook_lua = hook_entry.get("state", {})
            hook_meta = dict(hook_entry.get("action_metadata") or {})
            hook_step_num = int(hook_entry.get("frame_idx") or step_num)
            hook_rel_ts = max(0.0, float(hook_entry.get("timestamp", 0.0)) - base_ts)

            saved_boss_h = state.boss_blind
            saved_reroll_h = state.reroll_cost
            sync_cv_observations_before_action(state, hook_lua, act_name="Discard")
            _sync_extra_lua_fields(state, hook_lua, is_mid_setup=False)
            if saved_boss_h is not None:
                state.boss_blind = saved_boss_h
            if saved_reroll_h is not None:
                state.reroll_cost = saved_reroll_h

            hook_action = parse_lua_action(
                "Discard",
                hook_meta,
                frame_idx=hook_step_num,
                timestamp=hook_rel_ts,
            )
            hook_action.metadata["hook"] = True
            hook_action.selected_cards = [
                copy.deepcopy(hc.card) for hc in state.hand.cards if hc.selected
            ]
            on_action(state, hook_action)
            i += 1
            next_lua = entries[i + 1].get("state", {}) if i + 1 < len(entries) else None

        if action.type == bm.ActionType.NEXT_ROUND or int(lua_state.get("ante", state.ante)) == state.ante:
            boss_str = lua_state.get("boss_blind", "") or (
                next_lua.get("boss_blind", "") if next_lua is not None else ""
            )
            if boss_str in BLIND_BY_STR:
                state.boss_blind = BLIND_BY_STR[boss_str]

        won_round = (
            next_lua is not None
            and lua_state.get("blind", "") != ""
            and next_lua.get("blind", "") == ""
            and (
                act_name == "Play Hand"
                or (act_name == "Discard" and action.metadata.get("hook"))
            )
        )
        if won_round and not inject_end_round:
            action.metadata["won_round"] = True

        # Match post_processing.py:2213-2216 temporary hand cleanup
        if action.type in (
            bm.ActionType.END_ROUND,
            bm.ActionType.CASH_OUT,
            bm.ActionType.OPEN_PACK,
        ):
            state.hand.cards.clear()
        elif prev_loc == bm.Location.PACK and state.location != bm.Location.PACK:
            state.hand.cards.clear()

        if won_round and inject_end_round:
            # Record winning hand first (post_processing.py emits PLAY_HAND before END_ROUND)
            trajectory.record_step(state, copy.deepcopy(action))
            end_action = bm.Action(
                type=bm.ActionType.END_ROUND,
                frame_idx=action.frame_idx,
                timestamp=action.timestamp,
            )
            on_action(state, end_action)
            state.hand.cards.clear()
            if next_lua is not None:
                warnings.extend(
                    _check_post_step_parity(
                        f"step {entry.get('step', i + 1)} (End Round)",
                        "End Round",
                        state,
                        next_lua,
                        reconcile_truth=reconcile_truth,
                    )
                )
            trajectory.record_step(state, end_action)
        else:
            if next_lua is not None:
                warnings.extend(
                    _check_post_step_parity(
                        f"step {entry.get('step', i + 1)} ({act_name})",
                        act_name,
                        state,
                        next_lua,
                        reconcile_truth=reconcile_truth,
                    )
                )
            trajectory.record_step(state, copy.deepcopy(action))

        i += 1

    trajectory.impute_retroactive_boss_blinds()
    return state, trajectory, warnings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Balatro mod run_telemetry.jsonl into a 1:1 schema-v2 *_trajectory.json."
    )
    parser.add_argument(
        "telemetry_jsonl",
        type=str,
        nargs="?",
        default="assets/local/run_telemetry.jsonl",
        help="Path to run_telemetry.jsonl",
    )
    parser.add_argument(
        "--out", type=str, default=None, help="Output trajectory JSON path"
    )
    parser.add_argument(
        "--seed", type=str, default=None, help="Select run matching this seed"
    )
    parser.add_argument(
        "--run-index",
        type=int,
        default=None,
        help="0-based index of run segment to extract (default: longest non-synthetic run)",
    )
    parser.add_argument(
        "--list-runs", action="store_true", help="List detected run segments and exit"
    )
    parser.add_argument(
        "--keep-all-sorts", action="store_true", help="Keep non-face-down Sort steps"
    )
    parser.add_argument(
        "--no-inject-end-round",
        action="store_true",
        help="Do not inject END_ROUND before CASH_OUT",
    )
    parser.add_argument(
        "--strict-sim",
        action="store_true",
        help="Do not reconcile RNG/hidden post-step diffs against Lua",
    )
    parser.add_argument("--verbose", action="store_true", help="Print all parity notes")
    args = parser.parse_args()

    in_path = Path(args.telemetry_jsonl)
    entries = load_telemetry_entries(in_path)
    runs = split_telemetry_runs(entries)

    if args.list_runs:
        print(f"Found {len(runs)} run segment(s) in {in_path}:")
        for idx, run in enumerate(runs):
            s0 = run[0].get("state", {})
            s1 = run[-1].get("state", {})
            print(
                f"  [{idx}] seed={s0.get('seed')} deck={s0.get('deck_name')} stake={s0.get('stake')} "
                f"steps={run[0].get('step')}..{run[-1].get('step')} (len={len(run)}) "
                f"ante={s0.get('ante')}->{s1.get('ante')} hands0={s0.get('hands_left')}/{s0.get('base_hands')}"
            )
        return

    chosen = select_run(runs, run_index=args.run_index, seed=args.seed)
    state, traj, warnings = build_trajectory_from_telemetry(
        chosen,
        inject_end_round=not args.no_inject_end_round,
        keep_all_sorts=args.keep_all_sorts,
        reconcile_truth=not args.strict_sim,
    )
    payload = serialize_trajectory(state, traj)

    out_path = (
        Path(args.out)
        if args.out
        else (
            REPO_ROOT
            / "output"
            / "trajectories"
            / f"run_{state.seed}_{len(chosen)}steps_gt_trajectory.json"
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(
        f"Converted {len(chosen)} telemetry actions -> {payload['num_steps']} trajectory steps | "
        f"Seed={state.seed} Deck={state.deck.value} Stake={state.stake.value} | "
        f"Ante {state.ante} Round {state.round} Money=${state.money} Deck={len(state.full_deck)} | "
        f"Saved: {out_path}"
    )
    if warnings:
        print(f"Parity notes ({len(warnings)}):")
        for w in warnings if args.verbose else warnings[:10]:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
