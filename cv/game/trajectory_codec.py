"""Sparse GameStateDelta JSON codec and full GameState rebuilder.

Serializes a `(GameState, RunTrajectory)` into a compact, delta-only JSON format:
- `initial_state`: baseline configuration for step 0
- `steps`: one sparse `GameStateDelta` entry per step, recording only changed fields
  (e.g., `hand_diff` when hand cards change, `jokers_diff` with editions/stickers/state
  when jokers change, omitting unchanged fields such as `reroll_cost` or `money`)

Provides `rebuild_trajectory` and `rebuild_gamestate` to reconstruct the full
`RunTrajectory` / `GameState` from the serialized JSON.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from game.balatro_models import (
    Action,
    ActionType,
    BlindName,
    BoosterPack,
    BoosterSize,
    BoosterType,
    Card,
    CardBounds,
    CardType,
    Deck,
    Edition,
    Enhancement,
    HandCard,
    Joker,
    JokerName,
    Location,
    Planet,
    PlanetName,
    PlayingCard,
    PokerHand,
    Rank,
    Seal,
    SkipTag,
    Spectral,
    SpectralName,
    Stake,
    Suit,
    Tarot,
    TarotName,
    Voucher,
    VoucherName,
)
from game.gamestate import (
    _TRACKED_STATE_FIELDS,
    GameState,
    GameStateDelta,
    RunTrajectory,
    apply_delta,
)

SCHEMA_VERSION = 2

# Fields already represented as dedicated top-level properties on GameStateDelta.
_TOP_LEVEL_DELTA_FIELD_ALIASES: set[str] = {
    "location",
    "ante",
    "round",
    "current_blind",
    "boss_blind",
    "target_chips",
    "current_chips",
    "deck_cards_count",
    "full_deck",
    "discard_pile",
    "played_cards_this_round",
    "vouchers_redeemed",
    "hand_levels",
    "hand_play_counts",
    "unique_planets_used",
}


def _serialize_playing_card(card: PlayingCard) -> dict[str, Any]:
    out: dict[str, Any] = {
        "rank": card.rank.value,
        "suit": card.suit.value,
    }
    if card.enhancement != Enhancement.NONE:
        out["enhancement"] = card.enhancement.value
    if card.seal != Seal.NONE:
        out["seal"] = card.seal.value
    if card.edition != Edition.NONE:
        out["edition"] = card.edition.value
    if card.bonus_chips:
        out["bonus_chips"] = card.bonus_chips
    if card.id:
        out["id"] = card.id
    if card.debuffed:
        out["debuffed"] = True
    if card.destroyed:
        out["destroyed"] = True
    if card.couponed:
        out["couponed"] = True
    if card.rental:
        out["rental"] = True
    if card.extra_value:
        out["extra_value"] = card.extra_value
    return out


def _deserialize_playing_card(d: dict[str, Any]) -> PlayingCard:
    return PlayingCard(
        rank=Rank(d.get("rank", Rank.NONE.value)),
        suit=Suit(d.get("suit", Suit.NONE.value)),
        enhancement=Enhancement(d.get("enhancement", Enhancement.NONE.value)),
        seal=Seal(d.get("seal", Seal.NONE.value)),
        edition=Edition(d.get("edition", Edition.NONE.value)),
        bonus_chips=int(d.get("bonus_chips", 0)),
        id=int(d.get("id", 0)),
        debuffed=bool(d.get("debuffed", False)),
        destroyed=bool(d.get("destroyed", False)),
        couponed=bool(d.get("couponed", False)),
        rental=bool(d.get("rental", False)),
        extra_value=int(d.get("extra_value", 0)),
    )


def _serialize_playing_card_list_diff(
    prev_cards: list[PlayingCard] | None,
    curr_cards: list[PlayingCard],
) -> Any:
    """Serializes an incremental diff between `prev_cards` and `curr_cards`.

    Returns `[]` when cleared, a compact dict (`added`, `removed_ids`,
    `debuffed_ids`, `undebuffed_ids`, `updated`) when an incremental diff
    applies cleanly, or a full serialized list as fallback.
    """
    if len(curr_cards) == 0:
        return []

    curr_ser = [_serialize_playing_card(c) for c in curr_cards]
    if prev_cards is None:
        return curr_ser

    prev_ser = [_serialize_playing_card(c) for c in prev_cards]
    # Fast path 1: pure prefix append (covers almost all played/discarded pile updates)
    if len(curr_ser) >= len(prev_ser) and curr_ser[:len(prev_ser)] == prev_ser:
        return {"added": curr_ser[len(prev_ser):]}

    # Fast path 2: ID-keyed diff when all cards carry unique non-zero IDs
    if (
        all(c.id != 0 for c in prev_cards)
        and all(c.id != 0 for c in curr_cards)
        and len({c.id for c in prev_cards}) == len(prev_cards)
        and len({c.id for c in curr_cards}) == len(curr_cards)
    ):
        prev_by_id = {c.id: (c, s) for c, s in zip(prev_cards, prev_ser)}
        curr_by_id = {c.id: (c, s) for c, s in zip(curr_cards, curr_ser)}
        surviving_prev_ids = [c.id for c in prev_cards if c.id in curr_by_id]
        curr_existing_ids = [c.id for c in curr_cards if c.id in prev_by_id]
        curr_new_ids = [c.id for c in curr_cards if c.id not in prev_by_id]

        # Preserve relative order of surviving cards, with new cards appended at the end
        if (
            surviving_prev_ids == curr_existing_ids
            and [c.id for c in curr_cards] == curr_existing_ids + curr_new_ids
        ):
            removed_ids = [c.id for c in prev_cards if c.id not in curr_by_id]
            debuffed_ids: list[int] = []
            undebuffed_ids: list[int] = []
            updated: list[dict[str, Any]] = []

            for cid in curr_existing_ids:
                p_card, p_ser = prev_by_id[cid]
                c_card, c_ser = curr_by_id[cid]
                if p_ser == c_ser:
                    continue
                p_toggled = dict(p_ser)
                if c_card.debuffed:
                    p_toggled["debuffed"] = True
                else:
                    p_toggled.pop("debuffed", None)
                if p_toggled == c_ser:
                    if c_card.debuffed:
                        debuffed_ids.append(cid)
                    else:
                        undebuffed_ids.append(cid)
                else:
                    updated.append(c_ser)

            added = [curr_by_id[cid][1] for cid in curr_new_ids]
            diff_out: dict[str, Any] = {}
            if removed_ids:
                diff_out["removed_ids"] = removed_ids
            if debuffed_ids:
                diff_out["debuffed_ids"] = debuffed_ids
            if undebuffed_ids:
                diff_out["undebuffed_ids"] = undebuffed_ids
            if updated:
                diff_out["updated"] = updated
            if added:
                diff_out["added"] = added
            return diff_out

    return curr_ser


def _deserialize_playing_card_list_diff(
    raw: Any,
    prev_cards: list[PlayingCard] | None = None,
) -> list[PlayingCard]:
    """Reconstructs a `PlayingCard` list from an incremental diff dict or full list."""
    if isinstance(raw, list):
        return [_deserialize_playing_card(c) for c in raw]
    if not isinstance(raw, dict):
        raise TypeError(f"Expected list or dict for PlayingCard list diff, got {type(raw)!r}")

    cards = [copy.deepcopy(c) for c in (prev_cards or ())]
    if "removed_ids" in raw:
        rem = set(raw["removed_ids"])
        cards = [c for c in cards if c.id not in rem]
    if "debuffed_ids" in raw:
        deb = set(raw["debuffed_ids"])
        for c in cards:
            if c.id in deb:
                c.debuffed = True
    if "undebuffed_ids" in raw:
        undeb = set(raw["undebuffed_ids"])
        for c in cards:
            if c.id in undeb:
                c.debuffed = False
    if "updated" in raw:
        upd_map = {int(u["id"]): _deserialize_playing_card(u) for u in raw["updated"]}
        cards = [upd_map.get(c.id, c) for c in cards]
    if "added" in raw:
        cards.extend(_deserialize_playing_card(a) for a in raw["added"])
    return cards


def _serialize_poker_hand_int_diff(
    prev_map: dict[PokerHand, int] | None,
    curr_map: dict[PokerHand, int],
) -> dict[str, int]:
    """Serializes only entries in `curr_map` whose value differs from `prev_map`."""
    if prev_map is None:
        return {h.value: v for h, v in curr_map.items()}
    return {h.value: v for h, v in curr_map.items() if prev_map.get(h) != v}


def _deserialize_poker_hand_int_diff(
    raw: dict[str, Any],
    prev_map: dict[PokerHand, int] | None = None,
) -> dict[PokerHand, int]:
    """Reconstructs a `{PokerHand: int}` mapping by merging incremental entries onto `prev_map`."""
    out = dict(prev_map) if prev_map is not None else {}
    out.update({PokerHand(k): int(v) for k, v in raw.items()})
    return out


def _serialize_card_bounds(bounds: CardBounds) -> dict[str, Any]:
    out: dict[str, Any] = {"is_face_down": bounds.is_face_down}
    if bounds.possible_ranks != (set(Rank) - {Rank.NONE}):
        out["possible_ranks"] = sorted(r.value for r in bounds.possible_ranks)
    if bounds.possible_suits != (set(Suit) - {Suit.NONE}):
        out["possible_suits"] = sorted(s.value for s in bounds.possible_suits)
    if bounds.possible_enhancements != set(Enhancement):
        out["possible_enhancements"] = sorted(e.value for e in bounds.possible_enhancements)
    if bounds.possible_editions != set(Edition):
        out["possible_editions"] = sorted(e.value for e in bounds.possible_editions)
    if bounds.possible_seals != set(Seal):
        out["possible_seals"] = sorted(s.value for s in bounds.possible_seals)
    if bounds.candidate_card_ids:
        out["candidate_card_ids"] = sorted(bounds.candidate_card_ids)
    if bounds.candidate_cards:
        out["candidate_cards"] = [_serialize_playing_card(c) for c in bounds.candidate_cards]
    return out


def _deserialize_card_bounds(d: dict[str, Any]) -> CardBounds:
    return CardBounds(
        is_face_down=bool(d.get("is_face_down", True)),
        possible_ranks=(
            {Rank(v) for v in d["possible_ranks"]}
            if "possible_ranks" in d
            else set(Rank) - {Rank.NONE}
        ),
        possible_suits=(
            {Suit(v) for v in d["possible_suits"]}
            if "possible_suits" in d
            else set(Suit) - {Suit.NONE}
        ),
        possible_enhancements=(
            {Enhancement(v) for v in d["possible_enhancements"]}
            if "possible_enhancements" in d
            else set(Enhancement)
        ),
        possible_editions=(
            {Edition(v) for v in d["possible_editions"]}
            if "possible_editions" in d
            else set(Edition)
        ),
        possible_seals=(
            {Seal(v) for v in d["possible_seals"]}
            if "possible_seals" in d
            else set(Seal)
        ),
        candidate_card_ids=set(d.get("candidate_card_ids", ())),
        candidate_cards=[
            _deserialize_playing_card(c) for c in d.get("candidate_cards", ())
        ],
    )


def _serialize_hand_card(hc: HandCard) -> dict[str, Any]:
    out: dict[str, Any] = {"card": _serialize_playing_card(hc.card)}
    if hc.selected:
        out["selected"] = True
    if hc.facing_back:
        out["facing_back"] = True
    if hc.forced_selection:
        out["forced_selection"] = True
    if hc.slot_uid:
        out["slot_uid"] = hc.slot_uid
    if hc.bounds is not None:
        out["bounds"] = _serialize_card_bounds(hc.bounds)
    if hc._real_card is not None:
        out["_real_card"] = _serialize_playing_card(hc._real_card)
    return out


def _deserialize_hand_card(d: dict[str, Any]) -> HandCard:
    real_card = (
        _deserialize_playing_card(d["_real_card"])
        if d.get("_real_card") is not None
        else None
    )
    bounds = (
        _deserialize_card_bounds(d["bounds"])
        if d.get("bounds") is not None
        else None
    )
    return HandCard(
        card=_deserialize_playing_card(d["card"]),
        selected=bool(d.get("selected", False)),
        facing_back=bool(d.get("facing_back", False)),
        forced_selection=bool(d.get("forced_selection", False)),
        bounds=bounds,
        slot_uid=int(d.get("slot_uid", 0)),
        _real_card=real_card,
    )


def _serialize_joker(j: Joker) -> dict[str, Any]:
    out: dict[str, Any] = {"name": j.name.value}
    if j.edition != Edition.NONE:
        out["edition"] = j.edition.value
    if j.eternal:
        out["eternal"] = True
    if j.perishable:
        out["perishable"] = True
    if j.perishable or j.perish_tally != 5:
        out["perish_tally"] = j.perish_tally
    if j.rental:
        out["rental"] = True
    if j.debuffed:
        out["debuffed"] = True
    if j.facing_back:
        out["facing_back"] = True
    if j.couponed:
        out["couponed"] = True
    if j.extra_value:
        out["extra_value"] = j.extra_value
    if j.state:
        out["state"] = copy.deepcopy(j.state)
    if j.id:
        out["id"] = j.id
    return out


def _deserialize_joker(d: dict[str, Any]) -> Joker:
    return Joker(
        name=JokerName(d["name"]),
        edition=Edition(d.get("edition", Edition.NONE.value)),
        eternal=bool(d.get("eternal", False)),
        perishable=bool(d.get("perishable", False)),
        perish_tally=int(d.get("perish_tally", 5)),
        rental=bool(d.get("rental", False)),
        debuffed=bool(d.get("debuffed", False)),
        facing_back=bool(d.get("facing_back", False)),
        couponed=bool(d.get("couponed", False)),
        extra_value=int(d.get("extra_value", 0)),
        state=copy.deepcopy(d.get("state", {})),
        id=int(d.get("id", 0)),
    )


def _serialize_card(card: Card) -> dict[str, Any]:
    if isinstance(card, Joker):
        out = _serialize_joker(card)
        out["type"] = CardType.JOKER.value
        return out
    if isinstance(card, PlayingCard):
        out = _serialize_playing_card(card)
        out["type"] = CardType.PLAYING_CARD.value
        return out
    if isinstance(card, Tarot):
        out = {"type": CardType.TAROT.value, "name": card.name.value}
    elif isinstance(card, Planet):
        out = {"type": CardType.PLANET.value, "name": card.name.value}
    elif isinstance(card, Spectral):
        out = {"type": CardType.SPECTRAL.value, "name": card.name.value}
    else:
        raise TypeError(f"Unsupported Card subclass: {type(card)!r}")

    if card.edition != Edition.NONE:
        out["edition"] = card.edition.value
    if card.couponed:
        out["couponed"] = True
    if card.rental:
        out["rental"] = True
    if card.extra_value:
        out["extra_value"] = card.extra_value
    return out


def _deserialize_card(d: dict[str, Any]) -> Card:
    ctype = CardType(d["type"])
    if ctype == CardType.JOKER:
        return _deserialize_joker(d)
    if ctype == CardType.PLAYING_CARD:
        return _deserialize_playing_card(d)
    common = dict(
        edition=Edition(d.get("edition", Edition.NONE.value)),
        couponed=bool(d.get("couponed", False)),
        rental=bool(d.get("rental", False)),
        extra_value=int(d.get("extra_value", 0)),
    )
    if ctype == CardType.TAROT:
        return Tarot(name=TarotName(d["name"]), **common)
    if ctype == CardType.PLANET:
        return Planet(name=PlanetName(d["name"]), **common)
    if ctype == CardType.SPECTRAL:
        return Spectral(name=SpectralName(d["name"]), **common)
    raise ValueError(f"Unknown CardType: {ctype!r}")


def _serialize_voucher(v: Voucher) -> dict[str, Any]:
    out: dict[str, Any] = {"name": v.name.value}
    if v.price != 10:
        out["price"] = v.price
    return out


def _deserialize_voucher(d: dict[str, Any]) -> Voucher:
    return Voucher(name=VoucherName(d["name"]), price=int(d.get("price", 10)))


def _serialize_booster_pack(bp: BoosterPack) -> dict[str, Any]:
    out: dict[str, Any] = {
        "booster_type": bp.booster_type.value,
        "size": bp.size.value,
        "cost": bp.cost,
        "choose": bp.choose,
        "extra": bp.extra,
    }
    if bp.couponed:
        out["couponed"] = True
    return out


def _deserialize_booster_pack(d: dict[str, Any]) -> BoosterPack:
    return BoosterPack(
        booster_type=BoosterType(d["booster_type"]),
        size=BoosterSize(d["size"]),
        cost=int(d["cost"]),
        choose=int(d["choose"]),
        extra=int(d["extra"]),
        couponed=bool(d.get("couponed", False)),
    )


def _serialize_action(act: Action) -> dict[str, Any]:
    out: dict[str, Any] = {"type": act.type.value}
    if act.target_idx is not None:
        out["target_idx"] = act.target_idx
    if act.selected_cards:
        out["selected_cards"] = [_serialize_playing_card(c) for c in act.selected_cards]
    if act.joker_order:
        out["joker_order"] = list(act.joker_order)
    if act.hand_order:
        out["hand_order"] = list(act.hand_order)
    if act.metadata:
        out["metadata"] = copy.deepcopy(act.metadata)
    return out


def _deserialize_action(
    d: dict[str, Any], *, frame_idx: int, timestamp: float
) -> Action:
    return Action(
        type=ActionType(d["type"]),
        frame_idx=frame_idx,
        timestamp=timestamp,
        target_idx=d.get("target_idx"),
        selected_cards=[
            _deserialize_playing_card(c) for c in d.get("selected_cards", ())
        ],
        joker_order=list(d.get("joker_order", ())),
        hand_order=list(d.get("hand_order", ())),
        metadata=copy.deepcopy(d.get("metadata", {})),
    )


def _serialize_field_value(
    key: str, val: Any, prev_state: GameState | None = None
) -> Any:
    if val is None:
        return None
    if key in ("shop_cards", "pack_cards"):
        return [_serialize_card(c) for c in val]
    if key == "shop_vouchers":
        return [_serialize_voucher(v) for v in val]
    if key in ("shop_packs", "pending_packs"):
        return [_serialize_booster_pack(bp) for bp in val]
    if key == "active_pack":
        return _serialize_booster_pack(val)
    if key in ("full_deck", "discard_pile", "played_cards_this_round", "played_cards_this_ante"):
        prev_cards = getattr(prev_state, key, None) if prev_state is not None else None
        return _serialize_playing_card_list_diff(prev_cards, val)
    if key == "active_tags":
        return [t.value if hasattr(t, "value") else str(t) for t in val]
    if key == "vouchers_redeemed":
        return [v.value for v in val]
    if key in ("unique_planets_used", "pending_cv_checks"):
        return sorted(val)
    if key == "hands_played_this_round_by_type":
        return sorted(h.value for h in val)
    if key in ("hand_levels", "hand_play_counts"):
        return {h.value: v for h, v in val.items()}
    if key == "idol_card":
        return [val[0].value, val[1].value]
    if hasattr(val, "value"):
        return val.value
    return copy.deepcopy(val)


def _deserialize_field_value(
    key: str, val: Any, prev_state: GameState | None = None
) -> Any:
    if val is None:
        return None
    if key == "location" or key == "pre_pack_location":
        return Location(val)
    if key == "stake":
        return Stake(val)
    if key == "deck":
        return Deck(val)
    if key in ("current_blind", "boss_blind"):
        return BlindName(val)
    if key in ("shop_cards", "pack_cards"):
        return [_deserialize_card(c) for c in val]
    if key == "shop_vouchers":
        return [_deserialize_voucher(v) for v in val]
    if key in ("shop_packs", "pending_packs"):
        return [_deserialize_booster_pack(bp) for bp in val]
    if key == "active_pack":
        return _deserialize_booster_pack(val)
    if key in ("full_deck", "discard_pile", "played_cards_this_round", "played_cards_this_ante"):
        prev_cards = getattr(prev_state, key, None) if prev_state is not None else None
        return _deserialize_playing_card_list_diff(val, prev_cards)
    if key in ("small_blind_tag", "big_blind_tag"):
        return SkipTag(val)
    if key == "active_tags":
        return [SkipTag(t) for t in val]
    if key in (
        "small_blind_orbital_hand",
        "big_blind_orbital_hand",
        "to_do_poker_hand",
        "locked_hand_type",
        "ox_target_hand",
    ):
        return PokerHand(val)
    if key == "vouchers_redeemed":
        return [VoucherName(v) for v in val]
    if key == "last_tarot_planet":
        try:
            return TarotName(val)
        except ValueError:
            return PlanetName(val)
    if key in ("unique_planets_used", "pending_cv_checks"):
        return set(val)
    if key == "hands_played_this_round_by_type":
        return {PokerHand(h) for h in val}
    if key in ("hand_levels", "hand_play_counts"):
        return {PokerHand(k): int(v) for k, v in val.items()}
    if key == "idol_card":
        return (Rank(val[0]), Suit(val[1]))
    if key == "mail_card_rank":
        return Rank(val)
    if key in ("ancient_suit", "castle_suit"):
        return Suit(val)
    return copy.deepcopy(val)


def serialize_delta(
    delta: GameStateDelta, prev_state: GameState | None = None
) -> dict[str, Any]:
    """Serializes one `GameStateDelta` into a sparse dict containing only changed fields."""
    out: dict[str, Any] = {
        "step_idx": delta.step_idx,
        "frame_idx": delta.frame_idx,
        "timestamp": float(delta.timestamp),
    }
    if delta.action is not None:
        out["action"] = _serialize_action(delta.action)

    if delta.location is not None:
        out["location"] = delta.location.value
    if delta.ante is not None:
        out["ante"] = delta.ante
    if delta.round is not None:
        out["round"] = delta.round
    if delta.current_blind is not None:
        out["current_blind"] = delta.current_blind.value
    if delta.boss_blind is not None:
        out["boss_blind"] = delta.boss_blind.value
    if delta.target_chips is not None:
        out["target_chips"] = delta.target_chips
    if delta.current_chips is not None:
        out["current_chips"] = delta.current_chips
    if delta.deck_cards_count is not None:
        out["deck_cards_count"] = delta.deck_cards_count
    if delta.full_deck_diff is not None:
        prev_deck = prev_state.full_deck if prev_state is not None else None
        out["full_deck_diff"] = _serialize_playing_card_list_diff(prev_deck, delta.full_deck_diff)
    if delta.discard_pile_diff is not None:
        prev_discard = prev_state.discard_pile if prev_state is not None else None
        out["discard_pile_diff"] = _serialize_playing_card_list_diff(
            prev_discard, delta.discard_pile_diff
        )
    if delta.played_cards_diff is not None:
        prev_played = prev_state.played_cards_this_round if prev_state is not None else None
        out["played_cards_diff"] = _serialize_playing_card_list_diff(
            prev_played, delta.played_cards_diff
        )

    if delta.field_diffs:
        filtered_diffs = {
            k: _serialize_field_value(k, v, prev_state=prev_state)
            for k, v in delta.field_diffs.items()
            if k not in _TOP_LEVEL_DELTA_FIELD_ALIASES
        }
        if filtered_diffs:
            out["field_diffs"] = filtered_diffs

    if delta.jokers_diff is not None:
        out["jokers_diff"] = [_serialize_joker(j) for j in delta.jokers_diff]
    if delta.consumables_diff is not None:
        out["consumables_diff"] = [_serialize_card(c) for c in delta.consumables_diff]
    if delta.joker_slots_diff is not None:
        out["joker_slots_diff"] = delta.joker_slots_diff
    if delta.consumable_slots_diff is not None:
        out["consumable_slots_diff"] = delta.consumable_slots_diff
    if delta.hand_diff is not None:
        out["hand_diff"] = [_serialize_hand_card(hc) for hc in delta.hand_diff]
    if delta.hand_max_size_diff is not None:
        out["hand_max_size_diff"] = delta.hand_max_size_diff
    if delta.vouchers_diff is not None:
        out["vouchers_diff"] = [v.value for v in delta.vouchers_diff]
    if delta.hand_levels_diff is not None:
        prev_levels = prev_state.hand_levels if prev_state is not None else None
        out["hand_levels_diff"] = _serialize_poker_hand_int_diff(
            prev_levels, delta.hand_levels_diff
        )
    if delta.hand_play_counts_diff is not None:
        prev_counts = prev_state.hand_play_counts if prev_state is not None else None
        out["hand_play_counts_diff"] = _serialize_poker_hand_int_diff(
            prev_counts, delta.hand_play_counts_diff
        )
    if delta.unique_planets_diff is not None:
        out["unique_planets_diff"] = sorted(delta.unique_planets_diff)

    return out


def deserialize_delta(
    d: dict[str, Any], prev_state: GameState | None = None
) -> GameStateDelta:
    """Reconstructs a `GameStateDelta` from a serialized step dict."""
    frame_idx = int(d["frame_idx"])
    timestamp = float(d["timestamp"])
    act = (
        _deserialize_action(d["action"], frame_idx=frame_idx, timestamp=timestamp)
        if d.get("action") is not None
        else None
    )
    field_diffs = (
        {
            k: _deserialize_field_value(k, v, prev_state=prev_state)
            for k, v in d["field_diffs"].items()
        }
        if d.get("field_diffs") is not None
        else None
    )
    prev_deck = prev_state.full_deck if prev_state is not None else None
    prev_discard = prev_state.discard_pile if prev_state is not None else None
    prev_played = prev_state.played_cards_this_round if prev_state is not None else None
    prev_levels = prev_state.hand_levels if prev_state is not None else None
    prev_counts = prev_state.hand_play_counts if prev_state is not None else None

    return GameStateDelta(
        step_idx=int(d["step_idx"]),
        frame_idx=frame_idx,
        timestamp=timestamp,
        action=act,
        location=Location(d["location"]) if d.get("location") is not None else None,
        ante=int(d["ante"]) if d.get("ante") is not None else None,
        round=int(d["round"]) if d.get("round") is not None else None,
        current_blind=(
            BlindName(d["current_blind"]) if d.get("current_blind") is not None else None
        ),
        boss_blind=BlindName(d["boss_blind"]) if d.get("boss_blind") is not None else None,
        target_chips=d.get("target_chips"),
        current_chips=d.get("current_chips"),
        deck_cards_count=d.get("deck_cards_count"),
        full_deck_diff=(
            _deserialize_playing_card_list_diff(d["full_deck_diff"], prev_deck)
            if d.get("full_deck_diff") is not None
            else None
        ),
        discard_pile_diff=(
            _deserialize_playing_card_list_diff(d["discard_pile_diff"], prev_discard)
            if d.get("discard_pile_diff") is not None
            else None
        ),
        played_cards_diff=(
            _deserialize_playing_card_list_diff(d["played_cards_diff"], prev_played)
            if d.get("played_cards_diff") is not None
            else None
        ),
        field_diffs=field_diffs,
        jokers_diff=(
            [_deserialize_joker(j) for j in d["jokers_diff"]]
            if d.get("jokers_diff") is not None
            else None
        ),
        consumables_diff=(
            [_deserialize_card(c) for c in d["consumables_diff"]]
            if d.get("consumables_diff") is not None
            else None
        ),
        joker_slots_diff=d.get("joker_slots_diff"),
        consumable_slots_diff=d.get("consumable_slots_diff"),
        hand_diff=(
            [_deserialize_hand_card(hc) for hc in d["hand_diff"]]
            if d.get("hand_diff") is not None
            else None
        ),
        hand_max_size_diff=d.get("hand_max_size_diff"),
        vouchers_diff=(
            [VoucherName(v) for v in d["vouchers_diff"]]
            if d.get("vouchers_diff") is not None
            else None
        ),
        hand_levels_diff=(
            _deserialize_poker_hand_int_diff(d["hand_levels_diff"], prev_levels)
            if d.get("hand_levels_diff") is not None
            else None
        ),
        hand_play_counts_diff=(
            _deserialize_poker_hand_int_diff(d["hand_play_counts_diff"], prev_counts)
            if d.get("hand_play_counts_diff") is not None
            else None
        ),
        unique_planets_diff=(
            set(d["unique_planets_diff"])
            if d.get("unique_planets_diff") is not None
            else None
        ),
    )


def _serialize_initial_state(state0: GameState) -> dict[str, Any]:
    baseline = GameState(
        seed=state0.seed,
        deck=state0.deck,
        stake=state0.stake,
        ante=state0.ante,
        round=state0.round,
        current_blind=state0.current_blind,
        location=state0.location,
        apply_deck_stake_on_init=True,
    )
    out: dict[str, Any] = {
        "step_idx": state0.step_idx,
        "frame_idx": state0.frame_idx,
        "timestamp": float(state0.timestamp),
        "seed": state0.seed,
        "deck": state0.deck.value,
        "stake": state0.stake.value,
        "ante": state0.ante,
        "round": state0.round,
        "current_blind": state0.current_blind.value,
        "location": state0.location.value,
    }
    overrides: dict[str, Any] = {}
    for attr in _TRACKED_STATE_FIELDS:
        bval = getattr(baseline, attr)
        sval = getattr(state0, attr)
        if bval != sval:
            overrides[attr] = _serialize_field_value(attr, sval)
    if state0.inventory.jokers != baseline.inventory.jokers:
        overrides["jokers"] = [_serialize_joker(j) for j in state0.inventory.jokers]
    if state0.inventory.consumables != baseline.inventory.consumables:
        overrides["consumables"] = [_serialize_card(c) for c in state0.inventory.consumables]
    if state0.inventory.joker_slots != baseline.inventory.joker_slots:
        overrides["joker_slots"] = state0.inventory.joker_slots
    if state0.inventory.consumable_slots != baseline.inventory.consumable_slots:
        overrides["consumable_slots"] = state0.inventory.consumable_slots
    if state0.hand.cards != baseline.hand.cards:
        overrides["hand_cards"] = [_serialize_hand_card(hc) for hc in state0.hand.cards]
    if state0.hand.max_size != baseline.hand.max_size:
        overrides["hand_max_size"] = state0.hand.max_size
    if overrides:
        out["overrides"] = overrides
    return out


def _deserialize_initial_state(d: dict[str, Any]) -> GameState:
    state0 = GameState(
        step_idx=int(d.get("step_idx", 0)),
        frame_idx=int(d.get("frame_idx", 0)),
        timestamp=float(d.get("timestamp", 0.0)),
        seed=str(d.get("seed", "")),
        deck=Deck(d.get("deck", Deck.RED_DECK.value)),
        stake=Stake(d.get("stake", Stake.WHITE.value)),
        ante=int(d.get("ante", 1)),
        round=int(d.get("round", 1)),
        current_blind=BlindName(d.get("current_blind", BlindName.SMALL.value)),
        location=Location(d.get("location", Location.SELECT.value)),
        apply_deck_stake_on_init=True,
    )
    overrides = d.get("overrides") or {}
    for k, v in overrides.items():
        if k == "jokers":
            state0.inventory.jokers = [_deserialize_joker(j) for j in v]
        elif k == "consumables":
            state0.inventory.consumables = [_deserialize_card(c) for c in v]
        elif k == "joker_slots":
            state0.inventory.joker_slots = int(v)
        elif k == "consumable_slots":
            state0.inventory.consumable_slots = int(v)
        elif k == "hand_cards":
            state0.hand.cards = [_deserialize_hand_card(hc) for hc in v]
        elif k == "hand_max_size":
            state0.hand.max_size = int(v)
        else:
            setattr(state0, k, _deserialize_field_value(k, v))
    return state0


def serialize_trajectory(state: GameState, trajectory: RunTrajectory) -> dict[str, Any]:
    """Serializes `(GameState, RunTrajectory)` into a sparse, delta-based JSON dict."""
    state0 = trajectory.get_state(0) if trajectory.deltas else state
    steps: list[dict[str, Any]] = []
    for i, delta in enumerate(trajectory.deltas):
        prev_st = trajectory.get_state(i - 1) if i > 0 else state0
        steps.append(serialize_delta(delta, prev_state=prev_st))
    return {
        "schema_version": SCHEMA_VERSION,
        "initial_state": _serialize_initial_state(state0),
        "final_state": {
            "seed": state.seed,
            "deck": state.deck.value,
            "stake": state.stake.value,
            "location": state.location.value,
            "ante": state.ante,
            "round": state.round,
            "blind": state.current_blind.value,
            "money": state.money,
            "current_chips": state.current_chips,
            "target_chips": state.target_chips,
            "hands_left": state.hands_left,
            "discards_left": state.discards_left,
            "reroll_cost": state.reroll_cost,
            "jokers": [_serialize_joker(j) for j in state.inventory.jokers],
            "vouchers_redeemed": [v.value for v in state.vouchers_redeemed],
        },
        "num_steps": len(steps),
        "steps": steps,
    }


def rebuild_trajectory(
    data: dict[str, Any], *, keyframe_interval: int = 20
) -> RunTrajectory:
    """Rebuilds a full `RunTrajectory` from a serialized `trajectory.json` dict."""
    if "initial_state" not in data or "steps" not in data:
        raise ValueError(
            "Trajectory JSON is missing 'initial_state' or 'steps' (expected schema_version >= 2)."
        )

    traj = RunTrajectory(keyframe_interval=keyframe_interval)
    steps: list[dict[str, Any]] = data["steps"]
    if not steps:
        return traj

    curr = _deserialize_initial_state(data["initial_state"])
    delta0 = deserialize_delta(steps[0], prev_state=curr)
    curr = apply_delta(curr, delta0)
    traj.keyframes[0] = curr.copy()
    traj.deltas.append(delta0)

    for step_dict in steps[1:]:
        delta = deserialize_delta(step_dict, prev_state=curr)
        curr = apply_delta(curr, delta)
        if delta.step_idx % traj.keyframe_interval == 0:
            traj.keyframes[delta.step_idx] = curr.copy()
        traj.deltas.append(delta)

    return traj


def rebuild_gamestate(
    data: dict[str, Any], *, step_idx: int | None = None
) -> GameState:
    """Rebuilds the full `GameState` at `step_idx` (defaults to final step) from `trajectory.json`."""
    traj = rebuild_trajectory(data)
    if not traj.deltas:
        return _deserialize_initial_state(data["initial_state"])
    target = len(traj.deltas) - 1 if step_idx is None else step_idx
    return traj.get_state(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild and inspect a GameState from trajectory.json.")
    parser.add_argument("trajectory_json", type=str, help="Path to *_trajectory.json")
    parser.add_argument("--step", type=int, default=None, help="Optional step_idx to reconstruct (default: final)")
    args = parser.parse_args()

    with Path(args.trajectory_json).open(encoding="utf-8") as f:
        payload = json.load(f)

    rebuilt = rebuild_gamestate(payload, step_idx=args.step)
    print(
        f"Rebuilt GameState step={rebuilt.step_idx} | Deck={rebuilt.deck.value} | "
        f"Stake={rebuilt.stake.value} | Ante {rebuilt.ante} Round {rebuilt.round} "
        f"({rebuilt.current_blind.value}) | Loc={rebuilt.location.value} | "
        f"Money=${rebuilt.money} | Jokers={len(rebuilt.inventory.jokers)} | "
        f"Hand={len(rebuilt.hand.cards)}"
    )
