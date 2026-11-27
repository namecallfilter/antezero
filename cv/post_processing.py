"""
Post-processing and video-to-GameState trajectory orchestration pipeline.

Converts raw template-matching click logs (`main.py` output JSON) plus gameplay
video frames into batched CV observations and a deterministic `(GameState, RunTrajectory)`
sequence.

Scope notes:
- Starting mid-run is out of scope; replay reconstruction assumes a fresh run
  starting at Ante 1, Small Blind (`Location.SELECT`).
- Endless mode is out of scope; once OCR and simulation agree the ante has
  ticked over to Ante 9, processing stops immediately and truncates to the end
  of Ante 8.
- Exact chip/mult round score calculation is intentionally omitted; `current_chips`
  and HUD counters (money, hands, discards, round, ante, reroll cost) are left
  for OCR integration hooks.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from config import configure_logging, get_config
from game.balatro_models import (
    Action,
    ActionType,
    BlindName,
    BoosterPack,
    BoosterSize,
    BoosterType,
    Card,
    CardBounds,
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
    Spectral,
    SpectralName,
    Stake,
    Suit,
    Tarot,
    TarotName,
    Voucher,
)
from game.engine.cv_stubs import (
    _PLANET_LOOKUP,
    _clean_str,
    _convert_card_result_to_hand_card,
    _convert_items_of_type,
)
from game.engine.jokers import (
    _get_visible_poker_hands,
    on_joker_added,
    resolve_copy_target,
)
from game.engine.poker_hands import evaluate_poker_hand, get_scoring_cards
from game.engine.rng_sim.events import get_to_do_list_hand
from game.engine.rules import _effective_ante, on_action, trigger_end_of_round_effects
from game.gamestate import (
    GameState,
    RunTrajectory,
    _cards_match,
    calculate_target_chips,
)


ACTION_REMAP: dict[str, ActionType] = {
    "Select Card": ActionType.USE_IN_PACK,
}

NON_GAMEPLAY_ACTIONS: set[str] = {
    ActionType.OPEN_OPTIONS.value,
    ActionType.RUN_INFO.value,
    ActionType.PLAY_GAME.value,
    "Full Deck",
    "Back (Run Info)",
}

# Frame offsets relative to the logged button press frame.
# Negative offsets sample just before the click so dissolving/flying animations have not started.
# Positive offsets sample after animations (like sort) have settled.
FRAME_OFFSETS: dict[str | ActionType, int] = {
    "post_sort": 50,
    "pre_sort": -15,
    ActionType.PLAY_HAND: 0,
    ActionType.DISCARD: 0,
    ActionType.BUY_CARD: 0,
    ActionType.REROLL_SHOP: -10,
    # HUMAN TODO: manually verify remaining offsets below (above verified empirically)
    ActionType.BUY_AND_USE: -2,
    ActionType.OPEN_PACK: -2,
    ActionType.USE_IN_PACK: -2,
    ActionType.SKIP_PACK: -2,
    ActionType.REDEEM_VOUCHER: -2,
    ActionType.NEXT_ROUND: -2,
    ActionType.SELECT_BLIND: -2,
    ActionType.SKIP_BLIND: -2,
    ActionType.REROLL_BOSS: -2,
    ActionType.SELL_CARD: -2,
    ActionType.USE_CONSUMABLE: -2,
    ActionType.END_ROUND: -2,
    ActionType.CASH_OUT: -2,
}

# Actions that strictly belong to / define a specific location.
# NOTE: Location.CASH_OUT is only used internally by post_processing.py to separate
# the post-blind reward screen from active blind play when scheduling CV queues.
# When translating into GameState calls, use Location.BLIND with round_won=True.
LOCATION_DEFINING_MAP: dict[ActionType, Location] = {
    ActionType.SELECT_BLIND: Location.SELECT,
    ActionType.SKIP_BLIND: Location.SELECT,
    ActionType.REROLL_BOSS: Location.SELECT,
    ActionType.REROLL_SHOP: Location.SHOP,
    ActionType.NEXT_ROUND: Location.SHOP,
    ActionType.BUY_CARD: Location.SHOP,
    ActionType.BUY_AND_USE: Location.SHOP,
    ActionType.OPEN_PACK: Location.SHOP,
    ActionType.REDEEM_VOUCHER: Location.SHOP,
    ActionType.SKIP_PACK: Location.PACK,
    ActionType.USE_IN_PACK: Location.PACK,
    ActionType.PLAY_HAND: Location.BLIND,
    ActionType.DISCARD: Location.BLIND,
    ActionType.SORT_RANK: Location.BLIND,
    ActionType.SORT_SUIT: Location.BLIND,
    ActionType.END_ROUND: Location.BLIND,
    ActionType.CASH_OUT: Location.CASH_OUT,
}

# Legal `(min_selected, max_selected)` hand-card bounds for targeted Tarots and Spectrals.
_CONSUMABLE_HIGHLIGHT_LIMITS: dict[Any, tuple[int, int]] = {
    TarotName.DEATH: (2, 2),
    TarotName.HANGED_MAN: (1, 2),
    TarotName.STRENGTH: (1, 2),
    TarotName.MAGICIAN: (1, 2),
    TarotName.EMPRESS: (1, 2),
    TarotName.HIEROPHANT: (1, 2),
    TarotName.LOVERS: (1, 1),
    TarotName.CHARIOT: (1, 1),
    TarotName.JUSTICE: (1, 1),
    TarotName.DEVIL: (1, 1),
    TarotName.TOWER: (1, 1),
    TarotName.STAR: (1, 3),
    TarotName.MOON: (1, 3),
    TarotName.SUN: (1, 3),
    TarotName.WORLD: (1, 3),
    SpectralName.AURA: (1, 1),
    SpectralName.DEJA_VU: (1, 1),
    SpectralName.TALISMAN: (1, 1),
    SpectralName.MEDIUM: (1, 1),
    SpectralName.TRANCE: (1, 1),
    SpectralName.CRYPTID: (1, 1),
}

_HAND_REQUIRING_CONSUMABLES: frozenset[Any] = frozenset(
    _CONSUMABLE_HIGHLIGHT_LIMITS
) | frozenset(
    {
        SpectralName.FAMILIAR,
        SpectralName.GRIM,
        SpectralName.INCANTATION,
        SpectralName.IMMOLATE,
        SpectralName.SIGIL,
        SpectralName.OUIJA,
    }
)
_HAND_REQUIRING_CONSUMABLE_NAMES: frozenset[str] = frozenset(
    c.value for c in _HAND_REQUIRING_CONSUMABLES
)


@dataclass(slots=True)
class ScheduledAction:
    """Represents one filtered gameplay click and the video frames scheduled for CV evaluation."""

    idx: int
    frame_id: int
    action_type: ActionType
    target_frame: int
    ambient_location: Location
    button_box: list[int] | None = None
    pre_sort_frame: int | None = None
    post_sort_frame: int | None = None
    pre_pack_location_hint: Location | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _extract_action_id(entry: int | dict) -> int:
    return entry["action"] if isinstance(entry, dict) else entry


def _extract_box(entry: int | dict) -> list[int] | None:
    return entry.get("box") if isinstance(entry, dict) else None


def _boxes_match(
    box_a: list[int] | None, box_b: list[int] | None, tol_px: int = 28
) -> bool:
    """Returns True if both boxes are within `tol_px` on every coordinate (or if either is missing)."""
    if box_a is None or box_b is None or len(box_a) != 4 or len(box_b) != 4:
        return True
    return max(abs(a - b) for a, b in zip(box_a, box_b)) <= tol_px


@dataclass(slots=True)
class FullDeckObservation:
    """A grouped contiguous window where View Deck (`Full Deck` tab) is visible on screen."""

    start_frame: int
    end_frame: int
    sample_frame: int
    view_mode: str = "unknown"  # "full_deck", "remaining", or "unknown"
    box: list[int] | None = None


def _extract_full_deck_observations(
    raw_frames: dict[str, Any],
    key: dict[str, int],
    *,
    max_frame_gap: int = 20,
) -> list[FullDeckObservation]:
    """Groups raw `'Full Deck'` detections into candidate modal windows for lazy inference."""
    full_deck_id = key.get("Full Deck")
    if full_deck_id is None or not raw_frames:
        return []

    hits: list[tuple[int, int | dict]] = []
    for f_str, entry in raw_frames.items():
        if _extract_action_id(entry) == full_deck_id:
            hits.append((int(f_str), entry))
    if not hits:
        return []

    hits.sort(key=lambda item: item[0])
    windows: list[list[tuple[int, int | dict]]] = []
    for f_idx, entry in hits:
        if windows and (f_idx - windows[-1][-1][0]) <= max_frame_gap:
            windows[-1].append((f_idx, entry))
        else:
            windows.append([(f_idx, entry)])

    observations: list[FullDeckObservation] = []
    for win in windows:
        full_entries: list[tuple[int, int | dict]] = []
        rem_entries: list[tuple[int, int | dict]] = []
        for f_idx, entry in win:
            meta = entry.get("metadata") if isinstance(entry, dict) else None
            vm = (meta or {}).get("view_mode")
            if vm == "full_deck":
                full_entries.append((f_idx, entry))
            elif vm == "remaining":
                rem_entries.append((f_idx, entry))
        if full_entries:
            chosen = full_entries
            chosen_mode = "full_deck"
        elif rem_entries:
            chosen = rem_entries
            chosen_mode = "remaining"
        else:
            chosen = win
            chosen_mode = "unknown"

        mid_frame, mid_entry = chosen[len(chosen) // 2]
        observations.append(
            FullDeckObservation(
                start_frame=win[0][0],
                end_frame=win[-1][0],
                sample_frame=mid_frame,
                view_mode=chosen_mode,
                box=_extract_box(mid_entry),
            )
        )
    return observations


def remove_duplicates(
    d: dict[str, int | dict],
    *,
    max_frame_gap: int = 15,
) -> dict[str, int | dict]:
    """Collapses contiguous held button presses to the final click-release frame.

    Two detections of the same action and box are only merged if they occur within
    `max_frame_gap` frames of each other, so repeated distinct clicks (e.g. multiple
    Reroll Shop or consecutive Buy Card clicks at the same slot) are preserved.
    """
    new_dict: dict[str, int | dict] = {}
    prev_frame: int | None = None
    prev_action: int | None = None
    prev_box: list[int] | None = None

    # Reverse chronological order keeps the LAST frame (mouse release triggering the action)
    ordered_items = sorted(d.items(), key=lambda kv: int(kv[0]), reverse=True)
    for frame, entry in ordered_items:
        cur_frame = int(frame)
        action_id = _extract_action_id(entry)
        box = _extract_box(entry)
        is_same_hold = (
            prev_action == action_id
            and _boxes_match(prev_box, box)
            and prev_frame is not None
            and (prev_frame - cur_frame) <= max_frame_gap
        )
        if not is_same_hold:
            new_dict[frame] = entry
        prev_frame = cur_frame
        prev_action = action_id
        prev_box = box

    return dict(reversed(new_dict.items()))


def filter_premature_next_rounds(
    frames: dict[str, int | dict],
    key: dict[str, int],
    *,
    min_shop_frames: int = 60,
    diag: dict[str, Any] | None = None,
) -> dict[str, int | dict]:
    """Filters false-positive `Next Round` clicks fired while Shop UI slides up after `Cash Out`."""
    inv_key = {v: k for k, v in key.items()}
    ordered = sorted(frames.items(), key=lambda kv: int(kv[0]))
    filtered: dict[str, int | dict] = {}
    last_cash_out_frame: int | None = None

    for frame_str, entry in ordered:
        frame_idx = int(frame_str)
        raw_label = inv_key.get(_extract_action_id(entry))
        action = _resolve_action(raw_label) if raw_label is not None else None

        if action == ActionType.CASH_OUT:
            last_cash_out_frame = frame_idx
        elif action == ActionType.NEXT_ROUND:
            if (
                last_cash_out_frame is not None
                and 0 <= (frame_idx - last_cash_out_frame) < min_shop_frames
            ):
                if diag is not None:
                    diag["filtered_premature_next_rounds"].append(frame_idx)
                continue
            last_cash_out_frame = None
        elif action in (
            ActionType.SELECT_BLIND,
            ActionType.SKIP_BLIND,
            ActionType.PLAY_HAND,
        ):
            last_cash_out_frame = None

        filtered[frame_str] = entry

    return filtered


def filter_false_early_cash_outs(
    frames: dict[str, int | dict],
    key: dict[str, int],
    *,
    max_gap_frames: int = 15,
    diag: dict[str, Any] | None = None,
) -> dict[str, int | dict]:
    """Filters false `Cash Out` detections fired right after `Use Consumable`/`Sell Card`.

    Intended to run after `inject_end_round_actions` so `[PLAY_HAND, CASH_OUT]` windows remain
    intact for round-end detection while subsequent `Use Consumable`/`Sell Card` actions remain
    in `Location.CASH_OUT` until the first shop action triggers `preclicked_cash_out`.
    """
    inv_key = {v: k for k, v in key.items()}
    ordered = sorted(frames.items(), key=lambda kv: int(kv[0]))
    filtered: dict[str, int | dict] = {}
    last_held_item_frame: int | None = None

    for frame_str, entry in ordered:
        frame_idx = int(frame_str)
        raw_label = inv_key.get(_extract_action_id(entry))
        action = _resolve_action(raw_label) if raw_label is not None else None

        if action in (ActionType.USE_CONSUMABLE, ActionType.SELL_CARD):
            last_held_item_frame = frame_idx
        elif action == ActionType.CASH_OUT:
            is_disappearance = (
                isinstance(entry, dict)
                and isinstance(entry.get("metadata"), dict)
                and entry["metadata"].get("source") == "cash_out_disappearance"
            )
            if (
                not is_disappearance
                and last_held_item_frame is not None
                and 0 < (frame_idx - last_held_item_frame) <= max_gap_frames
            ):
                if diag is not None:
                    diag.setdefault("filtered_early_cash_outs", []).append(frame_idx)
                continue
            last_held_item_frame = None
        elif action in (
            ActionType.PLAY_HAND,
            ActionType.DISCARD,
            ActionType.SELECT_BLIND,
            ActionType.SKIP_BLIND,
            ActionType.NEXT_ROUND,
        ):
            last_held_item_frame = None

        filtered[frame_str] = entry

    return filtered


def _resolve_action(raw_label: str) -> ActionType | None:
    if raw_label in NON_GAMEPLAY_ACTIONS:
        return None
    if raw_label in ACTION_REMAP:
        return ACTION_REMAP[raw_label]
    return ActionType(raw_label)


def find_missing_end_round_segments(d: dict[str, Any]) -> list[tuple[int, int]]:
    """Finds `[PLAY_HAND, CASH_OUT]` frame windows lacking an intervening `END_ROUND`."""
    key = d.get("key") or {}
    frames = d.get("frames") or {}
    inv_key = {v: k for k, v in key.items()}

    ordered = sorted(frames.items(), key=lambda kv: int(kv[0]))
    segments: list[tuple[int, int]] = []
    pending_play_hand: int | None = None

    for frame_str, frame_entry in ordered:
        frame_idx = int(frame_str)
        raw_label = inv_key.get(_extract_action_id(frame_entry))
        if raw_label is None:
            continue
        action = _resolve_action(raw_label)
        if action is None:
            continue

        if action == ActionType.PLAY_HAND:
            pending_play_hand = frame_idx
        elif action in (
            ActionType.END_ROUND,
            ActionType.SELECT_BLIND,
            ActionType.SKIP_BLIND,
            ActionType.NEXT_ROUND,
        ):
            pending_play_hand = None
        elif action == ActionType.CASH_OUT:
            if pending_play_hand is not None and frame_idx > pending_play_hand + 1:
                segments.append((pending_play_hand, frame_idx))
            pending_play_hand = None

    return segments


def inject_end_round_actions(
    d: dict[str, Any],
    *,
    video_path: str | Path | None = None,
    preloaded_frames: dict[int, Any] | None = None,
    fps: float = 60.0,
    batch_size: int = 8,
    device: str | None = None,
    expressive: bool = False,
) -> dict[str, Any]:
    """Predicts and injects missing `ActionType.END_ROUND` events using the temporal model."""
    segments = find_missing_end_round_segments(d)
    if not segments or (video_path is None and preloaded_frames is None):
        return d

    from classifiers.inference import batch_recognize_end_rounds

    results = batch_recognize_end_rounds(
        segments,
        video_path=video_path,
        preloaded_frames=preloaded_frames,
        fps=fps,
        batch_size=batch_size,
        device=device,
        show_progress=expressive,
    )
    if not results:
        return d

    key = dict(d["key"])
    frames = dict(d["frames"])
    end_round_label = ActionType.END_ROUND.value
    end_round_id = key.setdefault(end_round_label, max(key.values(), default=-1) + 1)

    for (play_f, cash_f), res in zip(segments, results):
        pred_f = max(play_f + 1, min(cash_f - 1, int(res.end_round_frame)))
        if str(pred_f) in frames:
            chosen: int | None = None
            max_offset = max(pred_f - (play_f + 1), (cash_f - 1) - pred_f)
            for delta in range(1, max_offset + 1):
                for cand in (pred_f + delta, pred_f - delta):
                    if play_f < cand < cash_f and str(cand) not in frames:
                        chosen = cand
                        break
                if chosen is not None:
                    break
            if chosen is None:
                continue
            pred_f = chosen

        frames[str(pred_f)] = {
            "action": end_round_id,
            "box": None,
            "metadata": {
                "source": "end_round_classifier",
                "confidence": float(res.confidence),
                "end_round_timestamp": float(res.end_round_timestamp),
                "play_hand_frame": play_f,
                "cash_out_frame": cash_f,
            },
        }

    ordered_frames = dict(sorted(frames.items(), key=lambda kv: int(kv[0])))
    return {"key": key, "frames": ordered_frames}


FACE_DOWN_BOSS_BLINDS: frozenset[BlindName] = frozenset(
    {
        BlindName.THE_HOUSE,
        BlindName.THE_WHEEL,
        BlindName.THE_FISH,
        BlindName.THE_MARK,
    }
)


def plan_action_pipeline(d: dict[str, Any]) -> dict[str, Any]:
    """Parses cleaned action frames into location-aware CV queues and scheduled steps."""
    A = ActionType
    key = d["key"]
    frames = d["frames"]
    inv_key = {v: k for k, v in key.items()}

    parsed_frames: list[tuple[int, int | dict, ActionType]] = []
    for frame_str, frame_entry in frames.items():
        str_action = inv_key[_extract_action_id(frame_entry)]
        action = _resolve_action(str_action)
        if action is None:
            continue
        parsed_frames.append((int(frame_str), frame_entry, action))

    action_list = [act for _, _, act in parsed_frames]

    def peek_next_location(start_idx: int) -> Location | None:
        for act in action_list[start_idx:]:
            if act in LOCATION_DEFINING_MAP:
                return LOCATION_DEFINING_MAP[act]
        return None

    location = peek_next_location(0) or Location.SELECT

    def get_next_location(action: ActionType, current_idx: int) -> Location | None:
        match action:
            case A.NEXT_ROUND:
                return Location.SELECT
            case A.SELECT_BLIND:
                return Location.BLIND
            case A.END_ROUND:
                return Location.CASH_OUT
            case A.CASH_OUT:
                return Location.SHOP
            case A.OPEN_PACK:
                return Location.PACK
            case A.SKIP_BLIND:
                next_loc = peek_next_location(current_idx + 1)
                return Location.PACK if next_loc == Location.PACK else Location.SELECT
            case A.SKIP_PACK:
                return peek_next_location(current_idx + 1) or Location.SELECT
            case A.USE_IN_PACK:
                next_loc = peek_next_location(current_idx + 1)
                return Location.PACK if next_loc == Location.PACK else next_loc
            case _:
                return None

    playing_card_cv_queue: dict[int, ActionType] = {}
    joker_cv_queue: dict[int, ActionType] = {}
    shop_cv_queue: dict[int, ActionType] = {}
    pack_cv_queue: dict[int, ActionType] = {}
    hud_ocr_queue: dict[int, ActionType] = {}
    scheduled_actions: list[ScheduledAction] = []
    pre_pack_loc: Location | None = None
    blind_idx_in_ante = 0  # 0=Small, 1=Big, 2=Boss

    for idx, (frame_id, frame_entry, action) in enumerate(parsed_frames):
        button_box = _extract_box(frame_entry)

        if action in LOCATION_DEFINING_MAP:
            location = LOCATION_DEFINING_MAP[action]

        action_offset = FRAME_OFFSETS.get(action, -2)
        target_frame = max(0, frame_id + action_offset)
        pre_sort_frame: int | None = None
        post_sort_frame: int | None = None

        match action:
            case A.PLAY_HAND | A.DISCARD:
                playing_card_cv_queue[target_frame] = action

            case A.SORT_RANK | A.SORT_SUIT:
                if blind_idx_in_ante == 2:
                    pre_sort_frame = max(0, frame_id + FRAME_OFFSETS["pre_sort"])
                    post_sort_frame = max(0, frame_id + FRAME_OFFSETS["post_sort"])
                    playing_card_cv_queue[pre_sort_frame] = action
                    playing_card_cv_queue[post_sort_frame] = action

            case (
                A.BUY_CARD
                | A.BUY_AND_USE
                | A.OPEN_PACK
                | A.REDEEM_VOUCHER
                | A.REROLL_SHOP
                | A.NEXT_ROUND
            ):
                shop_cv_queue[target_frame] = action

            case A.USE_IN_PACK | A.SKIP_PACK:
                pack_cv_queue[target_frame] = action
                if action == A.USE_IN_PACK and location == Location.PACK:
                    playing_card_cv_queue[target_frame] = action

            case A.SELL_CARD | A.USE_CONSUMABLE:
                if location == Location.BLIND:
                    playing_card_cv_queue[target_frame] = action
                elif location == Location.SHOP:
                    shop_cv_queue[target_frame] = action
                elif location == Location.PACK:
                    pack_cv_queue[target_frame] = action
                    playing_card_cv_queue[target_frame] = action

            case (
                A.SELECT_BLIND | A.SKIP_BLIND | A.REROLL_BOSS | A.CASH_OUT | A.END_ROUND
            ):
                pass

        if action not in {A.SORT_RANK, A.SORT_SUIT}:
            hud_ocr_queue[target_frame] = action
            if target_frame not in shop_cv_queue and target_frame not in pack_cv_queue:
                joker_cv_queue[target_frame] = action

        extra_meta = (
            dict(frame_entry.get("metadata", {}))
            if isinstance(frame_entry, dict)
            and isinstance(frame_entry.get("metadata"), dict)
            else {}
        )
        scheduled_actions.append(
            ScheduledAction(
                idx=idx,
                frame_id=frame_id,
                action_type=action,
                target_frame=target_frame,
                ambient_location=location,
                button_box=button_box,
                pre_sort_frame=pre_sort_frame,
                post_sort_frame=post_sort_frame,
                pre_pack_location_hint=pre_pack_loc
                if location == Location.PACK
                else None,
                metadata=extra_meta,
            )
        )

        new_loc = get_next_location(action, idx)
        if new_loc is not None:
            if new_loc == Location.PACK and location != Location.PACK:
                pre_pack_loc = (
                    Location.SHOP if action == A.OPEN_PACK else Location.SELECT
                )
            elif new_loc != Location.PACK:
                pre_pack_loc = None
            location = new_loc

        if action in (A.END_ROUND, A.SKIP_BLIND):
            blind_idx_in_ante = (blind_idx_in_ante + 1) % 3

    return {
        "scheduled_actions": scheduled_actions,
        "playing_card_cv_queue": playing_card_cv_queue,
        "joker_cv_queue": joker_cv_queue,
        "shop_cv_queue": shop_cv_queue,
        "pack_cv_queue": pack_cv_queue,
        "hud_ocr_queue": hud_ocr_queue,
    }


def _pack_shows_hand_cards(pack_objects: Sequence[Any]) -> bool | None:
    """Infers from converted pack items whether the open pack shows hand cards underneath.

    Returns True for Arcana/Spectral packs (Tarot or Spectral items present),
    False for Celestial/Buffoon/Standard packs, or None if pack items are empty.
    """
    if not pack_objects:
        return None
    if any(isinstance(item, (Tarot, Spectral)) for item in pack_objects):
        return True
    if any(isinstance(item, (Planet, Joker, PlayingCard)) for item in pack_objects):
        return False
    return None


def _infer_booster_pack_from_items(pack_objects: Sequence[Any]) -> BoosterPack:
    """Synthesizes a fallback BoosterPack descriptor when a pack opens without a seed."""
    count = len(pack_objects)
    if any(isinstance(item, Spectral) for item in pack_objects):
        b_type = BoosterType.SPECTRAL
        size = BoosterSize.MEGA if count >= 4 else BoosterSize.REGULAR
    elif any(isinstance(item, Planet) for item in pack_objects):
        b_type = BoosterType.CELESTIAL
        size = BoosterSize.MEGA if count >= 5 else BoosterSize.REGULAR
    elif any(isinstance(item, Joker) for item in pack_objects):
        b_type = BoosterType.BUFFOON
        size = BoosterSize.MEGA if count >= 4 else BoosterSize.REGULAR
    elif any(isinstance(item, PlayingCard) for item in pack_objects):
        b_type = BoosterType.STANDARD
        size = BoosterSize.MEGA if count >= 5 else BoosterSize.REGULAR
    else:
        b_type = BoosterType.ARCANA
        size = BoosterSize.MEGA if count >= 5 else BoosterSize.REGULAR

    return BoosterPack(
        booster_type=b_type,
        size=size,
        cost=0,
        choose=2 if size == BoosterSize.MEGA else 1,
        extra=max(count, 2),
    )


def match_button_to_item_index(
    button_box: Sequence[int] | None,
    items: Sequence[Any],
    *,
    attach: str = "bottom_center",
) -> int | None:
    """Matches a clicked button `[x1, y1, x2, y2]` box to the closest detected item box.

    In Balatro (`card.lua:Card:highlight` and `UI_definitions.lua:use_and_sell_buttons`):
    - Shop / voucher / booster / pack buttons align `'bmi'` (centered horizontally under card).
    - Joker / consumable `'Sell'` and `'Use'` buttons align `'cr'` (anchored on card's right edge).
    """
    if not items:
        return None
    if button_box is None or len(button_box) != 4:
        return 0 if len(items) == 1 else None

    btn_cx = (button_box[0] + button_box[2]) / 2.0
    btn_cy = (button_box[1] + button_box[3]) / 2.0

    best_idx: int | None = None
    best_dist = float("inf")
    median_y1: float | None = None
    if attach == "center_right" and len(items) > 1:
        y1s = sorted(
            float(b[1])
            for it in items
            if (b := getattr(it, "box", None)) and len(b) == 4 and b != [0, 0, 0, 0]
        )
        if y1s:
            median_y1 = y1s[len(y1s) // 2]

    for idx, item in enumerate(items):
        box = getattr(item, "box", None)
        if not box or len(box) != 4 or box == [0, 0, 0, 0]:
            continue
        if attach == "center_right":
            item_w = max(1.0, float(box[2] - box[0]))
            anchor_x = float(box[2]) + 1.20 * item_w
            anchor_y = (box[1] + box[3]) / 2.0
        else:
            anchor_x = (box[0] + box[2]) / 2.0
            anchor_y = (box[1] + box[3]) / 2.0

        dist = abs(btn_cx - anchor_x) + 0.25 * abs(btn_cy - anchor_y)
        if attach == "center_right" and len(items) > 1:
            if btn_cx < float(box[0]):
                dist += 1000.0
            if (
                median_y1 is not None
                and float(box[1]) <= median_y1 - 10.0
                and btn_cx >= float(box[0])
            ):
                dist -= 40.0
        if dist < best_dist:
            best_dist = dist
            best_idx = idx

    if best_idx is None and len(items) == 1:
        return 0
    return best_idx


def _run_batched_cv(
    scheduled_actions: Sequence[ScheduledAction],
    queues: dict[str, dict[int, ActionType]],
    frames_by_idx: dict[int, Any],
    *,
    fps: float,
    conf_threshold_hand: float,
    conf_threshold_shop: float,
    conf_threshold_ocr: float = 0.90,
    batch_size: int,
    device: str | None,
    expressive: bool = False,
) -> tuple[dict[int, Any], dict[int, Any], dict[int, Any]]:
    """Runs shop/pack/joker-bar CV first, selectively runs hand CV, and batches HUD OCR."""
    from classifiers.inference import (
        batch_recognize_hands,
        batch_recognize_hud,
        batch_recognize_shops,
    )

    def _pack_frame_inputs(frame_ids: Sequence[int]) -> list[tuple[Any, int, float]]:
        return [
            (frames_by_idx[fid], fid, fid / fps)
            for fid in frame_ids
            if fid in frames_by_idx
        ]

    shop_ids = sorted(queues["shop_cv_queue"].keys())
    pack_ids = sorted(queues["pack_cv_queue"].keys())
    joker_ids = sorted(queues["joker_cv_queue"].keys())
    hud_queue = queues.get("hud_ocr_queue", {})
    hud_ids = sorted(hud_queue.keys())
    hand_candidates = sorted(queues["playing_card_cv_queue"].keys())

    pbar = (
        tqdm(
            total=len(shop_ids)
            + len(pack_ids)
            + len(joker_ids)
            + len(hand_candidates)
            + len(hud_ids),
            desc="Batched CV & OCR",
            unit="frame",
        )
        if expressive
        else None
    )

    shop_res_by_frame: dict[int, Any] = {}

    if shop_ids:
        if pbar is not None:
            pbar.set_postfix_str("shop")
        shop_outs = batch_recognize_shops(
            _pack_frame_inputs(shop_ids),
            mode="shop",
            conf_threshold=conf_threshold_shop,
            batch_size=batch_size,
            device=device,
        )
        for out in shop_outs:
            shop_res_by_frame[out.frame_index] = out
        if pbar is not None:
            pbar.update(len(shop_ids))

    if pack_ids:
        if pbar is not None:
            pbar.set_postfix_str("pack")
        pack_outs = batch_recognize_shops(
            _pack_frame_inputs(pack_ids),
            mode="pack",
            conf_threshold=conf_threshold_shop,
            batch_size=batch_size,
            device=device,
        )
        for out in pack_outs:
            shop_res_by_frame.setdefault(out.frame_index, out)
        if pbar is not None:
            pbar.update(len(pack_ids))

    if joker_ids:
        if pbar is not None:
            pbar.set_postfix_str("bars")
        bar_outs = batch_recognize_shops(
            _pack_frame_inputs(joker_ids),
            mode="bars",
            conf_threshold=conf_threshold_shop,
            batch_size=batch_size,
            device=device,
        )
        for out in bar_outs:
            shop_res_by_frame.setdefault(out.frame_index, out)
        if pbar is not None:
            pbar.update(len(joker_ids))

    # Dual-CV pack optimization: inspect recognized pack items first so Celestial/Buffoon/Standard
    # packs skip unnecessary hand card detection.
    pack_shows_hand_by_frame: dict[int, bool | None] = {}
    last_pack_hint: bool | None = None
    for sa in scheduled_actions:
        if sa.ambient_location != Location.PACK:
            last_pack_hint = None
            continue
        pack_out = shop_res_by_frame.get(sa.target_frame)
        converted_pack = (
            _convert_items_of_type(
                getattr(pack_out, "pack_items", None),
                (Joker, Tarot, Planet, Spectral, PlayingCard),
            )
            if pack_out is not None
            else []
        )
        hint = _pack_shows_hand_cards(converted_pack)
        if hint is not None:
            last_pack_hint = hint
        pack_shows_hand_by_frame[sa.target_frame] = (
            hint if hint is not None else last_pack_hint
        )

    filtered_hand_ids: list[int] = []
    skipped_hand = 0
    for fid in hand_candidates:
        if fid in queues["pack_cv_queue"]:
            if pack_shows_hand_by_frame.get(fid) is False:
                skipped_hand += 1
                continue
        filtered_hand_ids.append(fid)
    if pbar is not None and skipped_hand:
        pbar.update(skipped_hand)

    hand_res_by_frame: dict[int, Any] = {}
    if filtered_hand_ids:
        blind_hand_ids = [
            fid for fid in filtered_hand_ids if fid not in queues["pack_cv_queue"]
        ]
        pack_hand_ids = [
            fid for fid in filtered_hand_ids if fid in queues["pack_cv_queue"]
        ]
        if blind_hand_ids:
            if pbar is not None:
                pbar.set_postfix_str("hand")
            hand_outs = batch_recognize_hands(
                _pack_frame_inputs(blind_hand_ids),
                conf_threshold=conf_threshold_hand,
                batch_size=batch_size,
                device=device,
            )
            for out in hand_outs:
                hand_res_by_frame[out.frame_index] = out
            if pbar is not None:
                pbar.update(len(blind_hand_ids))
        if pack_hand_ids:
            if pbar is not None:
                pbar.set_postfix_str("pack_hand")
            pack_hand_outs = batch_recognize_hands(
                _pack_frame_inputs(pack_hand_ids),
                conf_threshold=min(conf_threshold_hand, 0.30),
                batch_size=batch_size,
                device=device,
                roi_key="pack_hand",
            )
            for out in pack_hand_outs:
                hand_res_by_frame[out.frame_index] = out
            if pbar is not None:
                pbar.update(len(pack_hand_ids))

    hud_res_by_frame: dict[int, Any] = {}
    if hud_ids:
        if pbar is not None:
            pbar.set_postfix_str("hud_ocr")
        base_hud_fields = (
            "hud_money",
            "hud_hands",
            "hud_discards",
            "hud_ante",
            "hud_round",
        )
        fields_by_frame: dict[int, list[str]] = {}
        for sa in scheduled_actions:
            if sa.target_frame not in hud_queue:
                continue
            f_list = fields_by_frame.setdefault(sa.target_frame, list(base_hud_fields))
            if (
                sa.ambient_location == Location.BLIND
                or sa.action_type == ActionType.END_ROUND
            ):
                for extra_field in ("hud_round_score", "hud_blind_target"):
                    if extra_field not in f_list:
                        f_list.append(extra_field)
            if sa.ambient_location == Location.SHOP and "hud_reroll_cost" not in f_list:
                f_list.append("hud_reroll_cost")

        hud_outs = batch_recognize_hud(
            _pack_frame_inputs(hud_ids),
            fields_by_frame=fields_by_frame,
            conf_threshold=conf_threshold_ocr,
            batch_size=max(batch_size, 128),
            device=device,
        )
        for out in hud_outs:
            hud_res_by_frame[out.frame_index] = out
        if pbar is not None:
            pbar.update(len(hud_ids))

    if pbar is not None:
        pbar.close()

    return shop_res_by_frame, hand_res_by_frame, hud_res_by_frame


def _has_end_round_after(
    scheduled_actions: Sequence[ScheduledAction], start_idx: int
) -> bool:
    """Returns True if an explicit END_ROUND action follows before CASH_OUT/SELECT_BLIND."""
    for nxt in scheduled_actions[start_idx + 1 :]:
        if nxt.action_type == ActionType.END_ROUND:
            return True
        if nxt.action_type in (
            ActionType.CASH_OUT,
            ActionType.SELECT_BLIND,
            ActionType.SKIP_BLIND,
            ActionType.NEXT_ROUND,
        ):
            return False
    return False


def _init_diagnostics() -> dict[str, Any]:
    """Creates a mutable diagnostics dictionary for `--expressive` reporting."""
    return {
        "timings": {},
        "end_round_segments_checked": 0,
        "end_rounds_injected": 0,
        "filtered_premature_next_rounds": [],
        "filtered_early_cash_outs": [],
        "missing_frames": [],
        "hud_frames_checked": 0,
        "ocr_score_updates": 0,
        "ocr_none_by_field": {},
        "ocr_overrides_total": 0,
        "ocr_overrides_by_field": {},
        "ocr_override_samples": [],
        "unexplained_joker_changes": [],
        "unexplained_consumable_changes": [],
        "unexpected_shop_transitions": [],
        "unmatched_button_targets": [],
        "location_mismatches": [],
        "cashout_without_round_won": [],
        "empty_hand_cv_actions": [],
        "missing_shop_cv_actions": [],
        "hand_size_over_15_actions": [],
        "unexpected_deck_additions": [],
        "large_ocr_overrides": [],
        "corrupted_actions": [],
        "quarantine_watermark": None,
        "quarantined_downstream_count": 0,
        "stopped_at_ante_9": None,
        "deck_constrained_remaps": 0,
        "face_down_plays_resolved": 0,
        "inferred_cash_outs": 0,
        "inferred_next_rounds": 0,
        "inferred_skip_packs": 0,
        "inferred_joker_sells": 0,
        "inferred_joker_buys": 0,
        "rng_cons_reconciled": 0,
        "rng_jokers_reconciled": 0,
        "reconciled_lucky_money": 0,
        "reconciled_rebate_money": 0,
        "shop_rescan_windows": 0,
        "shop_rescan_frames_scanned": 0,
        "shop_rescan_recovered": 0,
        "full_deck_views_detected": 0,
        "full_deck_views_skipped": 0,
        "full_deck_syncs": [],
        "full_deck_quarantine_recoveries": 0,
    }


def _record_ocr_override(
    diag: dict[str, Any] | None,
    *,
    frame_id: int,
    action_type: ActionType,
    field_name: str,
    prev_val: Any,
    new_val: Any,
) -> None:
    """Records an OCR override when observed HUD state differs from simulated state."""
    if diag is None or prev_val == new_val:
        return
    diag["ocr_overrides_total"] += 1
    by_field = diag["ocr_overrides_by_field"]
    by_field[field_name] = by_field.get(field_name, 0) + 1
    samples = diag["ocr_override_samples"]
    if len(samples) < 12:
        samples.append((frame_id, action_type.value, field_name, prev_val, new_val))

    is_large = False
    if field_name == "ante":
        is_large = True
    elif (
        field_name == "round"
        and isinstance(prev_val, int)
        and isinstance(new_val, int)
        and abs(new_val - prev_val) >= 2
    ):
        is_large = True
    elif (
        field_name == "money"
        and action_type != ActionType.CASH_OUT
        and isinstance(prev_val, (int, float))
        and isinstance(new_val, (int, float))
        and abs(new_val - prev_val) >= 15
    ):
        is_large = True
    if is_large:
        diag["large_ocr_overrides"].append(
            (frame_id, action_type.value, field_name, prev_val, new_val)
        )


def _print_expressive_report(diag: dict[str, Any]) -> None:
    """Prints end-of-run diagnostics when `--expressive` is enabled."""
    print("\n=== Expressive Post-Processing Diagnostics ===")
    timings = diag.get("timings", {})
    if timings:
        parts = [f"{k}={v:.2f}s" for k, v in timings.items()]
        print(f"Timings: {', '.join(parts)}")

    segs = diag.get("end_round_segments_checked", 0)
    inj = diag.get("end_rounds_injected", 0)
    if segs or inj:
        print(
            f"END_ROUND transitions: {inj} injected across {segs} [PLAY_HAND, CASH_OUT] segment(s)"
        )

    premature_nr = diag.get("filtered_premature_next_rounds", [])
    if premature_nr:
        print(
            f"Filtered premature Next Round detections (<60f after Cash Out): "
            f"{len(premature_nr)} (frames {premature_nr[:10]})"
        )

    early_co = diag.get("filtered_early_cash_outs", [])
    if early_co:
        print(
            f"Filtered false early Cash Out detections (<=15f after held-item click): "
            f"{len(early_co)} (frames {early_co[:10]})"
        )

    missing_frames = diag.get("missing_frames", [])
    if missing_frames:
        print(
            f"[WARN] Missing CV frames: {len(missing_frames)} requested frame(s) could not be read "
            f"(e.g. {missing_frames[:5]})"
        )

    hud_checked = diag.get("hud_frames_checked", 0)
    score_updates = diag.get("ocr_score_updates", 0)
    overrides = diag.get("ocr_overrides_total", 0)
    by_field = diag.get("ocr_overrides_by_field", {})
    print(
        f"OCR drift overrides: {overrides} field override(s) across {hud_checked} HUD-checked action(s) "
        f"(+{score_updates} blind score update(s))"
    )
    if by_field:
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(by_field.items()))
        print(f"  Override breakdown: {breakdown}")
        for fid, act, fld, old_v, new_v in diag.get("ocr_override_samples", []):
            print(f"  - frame {fid} ({act}) {fld}: sim={old_v} -> ocr={new_v}")

    none_counts = diag.get("ocr_none_by_field", {})
    if none_counts:
        dropped = ", ".join(f"{k}={v}" for k, v in sorted(none_counts.items()))
        print(f"OCR skipped/low-conf (None) counts: {dropped}")

    unexp_jokers = diag.get("unexplained_joker_changes", [])
    if unexp_jokers:
        print(
            f"[WARN] Unexplained Joker inventory changes: {len(unexp_jokers)} time(s)"
        )
        for fid, act, lost, gained in unexp_jokers[:10]:
            print(f"  - frame {fid} ({act}): lost={lost}, gained={gained}")

    unexp_cons = diag.get("unexplained_consumable_changes", [])
    if unexp_cons:
        print(
            f"[WARN] Unexplained Consumable inventory changes: {len(unexp_cons)} time(s)"
        )
        for fid, act, lost, gained in unexp_cons[:10]:
            print(f"  - frame {fid} ({act}): lost={lost}, gained={gained}")

    unmatched = diag.get("unmatched_button_targets", [])
    if unmatched:
        print(
            f"[WARN] Unmatched spatial button targets: {len(unmatched)} action(s) had target_idx=None"
        )
        for fid, act, loc in unmatched[:8]:
            print(f"  - frame {fid} ({act} @ {loc})")

    unexp_shop = diag.get("unexpected_shop_transitions", [])
    if unexp_shop:
        print(f"[WARN] Unexpected shop transitions: {len(unexp_shop)} time(s)")
        for item in unexp_shop[:12]:
            reasons_str = "; ".join(item["reasons"])
            print(
                f"  - frame {item['prev_frame']} ({item['prev_action']}) -> "
                f"frame {item['frame']} ({item['action']}): {reasons_str}"
            )

    loc_mismatches = diag.get("location_mismatches", [])
    if loc_mismatches:
        print(
            f"[WARN] Location drift: {len(loc_mismatches)} action(s) had ambient_location != state.location"
        )
        for fid, act, exp_loc, got_loc in loc_mismatches[:8]:
            print(f"  - frame {fid} ({act}): expected={exp_loc}, state={got_loc}")

    cashout_warn = diag.get("cashout_without_round_won", [])
    if cashout_warn:
        print(
            f"[WARN] CASH_OUT before round_won=True: {len(cashout_warn)} time(s) "
            f"(frames {cashout_warn[:8]})"
        )

    empty_hand = diag.get("empty_hand_cv_actions", [])
    if empty_hand:
        print(
            f"[WARN] PLAY_HAND/DISCARD with 0 detected hand cards: {len(empty_hand)} time(s) (frames {empty_hand[:8]})"
        )

    missing_shop = diag.get("missing_shop_cv_actions", [])
    if missing_shop:
        print(
            f"[WARN] SHOP/PACK actions missing CV output: {len(missing_shop)} time(s) (frames {missing_shop[:8]})"
        )

    over_15 = diag.get("hand_size_over_15_actions", [])
    if over_15:
        sample_str = ", ".join(
            f"f={fid} (exp={exp}, det={det})" for fid, _, exp, det in over_15[:6]
        )
        print(
            f"[WARN] Hand size > 15 (out-of-distribution CV): {len(over_15)} action(s) [{sample_str}]"
        )

    corrupted = diag.get("corrupted_actions", [])
    if corrupted:
        print(
            f"[WARN] Corrupted step(s) flagged for training exclusion: {len(corrupted)} "
            f"(frames {[fid for fid, _, _ in corrupted[:8]]})"
        )

    unexp_deck = diag.get("unexpected_deck_additions", [])
    if unexp_deck:
        total_adds = sum(cnt for _, _, cnt in unexp_deck)
        print(
            f"[WARN] Unexpected deck additions during hand sync: +{total_adds} card(s) "
            f"across {len(unexp_deck)} step(s) (e.g. frames {[f for f, _, _ in unexp_deck[:6]]})"
        )

    wm = diag.get("quarantine_watermark")
    if wm is not None:
        print(
            f"[WARN] Cascade quarantine triggered from step {wm} -> end "
            f"(+{diag.get('quarantined_downstream_count', 0)} downstream tag(s))"
        )

    ante9_stop = diag.get("stopped_at_ante_9")
    if ante9_stop:
        print(
            f"Ante 9 safeguard triggered at frame {ante9_stop['trigger_frame_id']} "
            f"({ante9_stop['trigger_action']}): stopped Endless Mode and kept "
            f"{ante9_stop['kept_steps']} step(s) up to end of Ante 8 "
            f"(dropped {ante9_stop['dropped_steps']} post-Ante-8 step(s))"
        )

    fd_det = diag.get("full_deck_views_detected", 0)
    fd_skip = diag.get("full_deck_views_skipped", 0)
    fd_syncs = diag.get("full_deck_syncs", [])
    fd_recov = diag.get("full_deck_quarantine_recoveries", 0)
    if fd_det or fd_syncs or fd_recov:
        print(
            f"Full Deck CV sync: views_detected={fd_det}, skipped={fd_skip}, "
            f"synced={len(fd_syncs)}, quarantine_recoveries={fd_recov}"
        )
        for s_info in fd_syncs[:8]:
            print(
                f"  - frame {s_info['frame_id']} ({s_info['view_mode']}): "
                f"{s_info['before']}->{s_info['after']} "
                f"(+{s_info['added']}, -{s_info['removed']}, ~{s_info['updated']}) "
                f"triggers={s_info['reasons']}"
            )

    remaps = diag.get("deck_constrained_remaps", 0)
    fd_resolved = diag.get("face_down_plays_resolved", 0)
    inf_co = diag.get("inferred_cash_outs", 0)
    inf_nr = diag.get("inferred_next_rounds", 0)
    inf_sp = diag.get("inferred_skip_packs", 0)
    inf_js = diag.get("inferred_joker_sells", 0)
    inf_jb = diag.get("inferred_joker_buys", 0)
    rng_cr = diag.get("rng_cons_reconciled", 0)
    rng_jr = diag.get("rng_jokers_reconciled", 0)
    rec_lm = diag.get("reconciled_lucky_money", 0)
    rec_rm = diag.get("reconciled_rebate_money", 0)
    rs_win = diag.get("shop_rescan_windows", 0)
    rs_rec = diag.get("shop_rescan_recovered", 0)
    if (
        remaps
        or fd_resolved
        or inf_co
        or inf_nr
        or inf_sp
        or inf_js
        or inf_jb
        or rng_cr
        or rng_jr
        or rec_lm
        or rec_rm
        or rs_win
        or rs_rec
    ):
        print(
            f"Recovery hooks: deck_remaps={remaps}, fd_plays_resolved={fd_resolved}, "
            f"inferred_cash_outs={inf_co}, inferred_next_rounds={inf_nr}, inferred_skip_packs={inf_sp}, "
            f"inferred_joker_sells={inf_js}, inferred_joker_buys={inf_jb}, "
            f"rng_cons_reconciled={rng_cr}, rng_jokers_reconciled={rng_jr}, "
            f"lucky_money={rec_lm}, rebate_money={rec_rm}, "
            f"shop_rescans={rs_win} (+{rs_rec} frames)"
        )
    print("==============================================")


def _get_hud_val(hud_out: Any, field_name: str) -> Any:
    """Safely extracts a parsed HUD OCR metric from `HudRecognitionOutput` or a test double."""
    if hud_out is None:
        return None
    values = getattr(hud_out, "values", None)
    if isinstance(values, dict) and values.get(field_name) is not None:
        return values[field_name]
    short = field_name[4:] if field_name.startswith("hud_") else field_name
    val = getattr(hud_out, short, None)
    if val is not None:
        return val
    return getattr(hud_out, field_name, None)


def _scores_indicate_no_hand_scored(
    cur_score: Any, nxt_score: Any, frame_delta: int
) -> bool:
    """Returns True if consecutive round-score OCR reads show no hand scored between them."""
    if cur_score is None or nxt_score is None:
        return True
    cur_i = int(cur_score)
    nxt_i = int(nxt_score)
    if nxt_i <= cur_i:
        return True
    # Minimum possible valid Balatro hand scores >= 5 chips (High Card); smaller delta is OCR jitter
    if (nxt_i - cur_i) <= 4:
        return True
    if frame_delta <= 180:
        s1 = str(cur_i)
        s2 = str(nxt_i)
        if len(s1) == len(s2) and sum(a != b for a, b in zip(s1, s2)) <= 1:
            return True
    return False


_SOFTLOCKED_PLANET_HANDS: dict[str, PokerHand] = {
    "c_planet_x": PokerHand.FIVE_OF_A_KIND,
    "c_ceres": PokerHand.FLUSH_HOUSE,
    "c_eris": PokerHand.FLUSH_FIVE,
}

_ZERO_TARGET_CONSUMABLES: frozenset[str] = frozenset(
    {
        "The Hermit",
        "Temperance",
        "Wheel of Fortune",
        "The Emperor",
        "The High Priestess",
        "The Fool",
        "Judgement",
        "Immolate",
        "Wraith",
        "Sigil",
        "Ouija",
        "Ectoplasm",
        "Ankh",
        "Hex",
        "The Soul",
        "Black Hole",
        "Familiar",
        "Grim",
        "Incantation",
        *(p.value for p in PlanetName),
    }
)

_DECK_MUTATING_CONSUMABLES: frozenset[str] = frozenset(
    {
        "Strength",
        "The Hanged Man",
        "Death",
        "The Lovers",
        "The Empress",
        "The Hierophant",
        "The Chariot",
        "Justice",
        "The Magician",
        "The Devil",
        "The Tower",
        "The Star",
        "The Moon",
        "The Sun",
        "The World",
        "Aura",
        "Deja Vu",
        "Talisman",
        "Trance",
        "Medium",
        "Cryptid",
    }
)


def _remap_softlocked_planets(raw_items: Sequence[Any], state: GameState) -> list[Any]:
    """Remaps softlocked secret-hand Planets (Planet X/Ceres/Eris) using Stage-2 top-k logits."""
    hand_counts = getattr(state, "hand_play_counts", None) or {}
    out: list[Any] = []
    for item in raw_items:
        key = getattr(item, "key", None)
        req_hand = _SOFTLOCKED_PLANET_HANDS.get(key) if key else None
        if req_hand is not None and hand_counts.get(req_hand, 0) <= 0:
            top_keys = getattr(item, "top_keys", None) or []
            remapped = None
            for cand_key, cand_prob in top_keys:
                if (
                    cand_prob < 0.05
                    or not isinstance(cand_key, str)
                    or not cand_key.startswith("c_")
                ):
                    continue
                cand_req = _SOFTLOCKED_PLANET_HANDS.get(cand_key)
                if cand_req is not None and hand_counts.get(cand_req, 0) <= 0:
                    continue
                if _clean_str(cand_key[2:]) not in _PLANET_LOOKUP:
                    continue
                cloned = copy.copy(item)
                cloned.key = cand_key
                cloned.card_type = "Planet"
                cloned_confs = dict(getattr(item, "confidences", None) or {})
                cloned_confs["key"] = float(cand_prob)
                cloned.confidences = cloned_confs
                remapped = cloned
                break
            out.append(remapped if remapped is not None else item)
        else:
            out.append(item)
    return out


def _active_joker_copies(state: GameState, target_name: JokerName) -> int:
    """Returns count of undebuffed `target_name` jokers plus Blueprint/Brainstorm copies."""
    if not state.inventory or not state.inventory.jokers:
        return 0
    jokers = state.inventory.jokers
    copies = 0
    for j in jokers:
        if j.debuffed:
            continue
        if j.name == target_name:
            copies += 1
        elif j.name in (JokerName.BLUEPRINT, JokerName.BRAINSTORM):
            target = resolve_copy_target(j, jokers)
            if (
                target is not None
                and not target.debuffed
                and target.name == target_name
            ):
                copies += 1
    return copies


def _mail_in_rebate_payout_per_card(state: GameState) -> int:
    """Returns active per-matching-discard dollar payout from all Mail-In Rebate & copy jokers."""
    return 5 * _active_joker_copies(state, JokerName.MAIL_IN_REBATE)


def _is_plausible_hud_round(
    new_round: int,
    cur_round: int,
    cur_ante: int,
    next_hud_round: int | None = None,
    *,
    ante_jumped: bool = False,
) -> bool:
    """Rejects CTC trailing-digit duplications (e.g. 18->188) and impossible jumps in hud_round."""
    if new_round < 1 or new_round < cur_round:
        return False
    if (
        cur_round > 0
        and new_round >= 10 * cur_round
        and str(new_round).startswith(str(cur_round))
    ):
        return False
    if cur_ante <= 8 and new_round > 3 * max(1, cur_ante) + 6:
        return False
    if not ante_jumped and new_round > cur_round + 2:
        if next_hud_round is None or next_hud_round < new_round:
            return False
    if new_round > cur_round and next_hud_round is not None:
        if next_hud_round < new_round - 1:
            return False
        if (
            not ante_jumped
            and new_round == cur_round + 2
            and next_hud_round < new_round
        ):
            return False
    return True


def _sync_hud_into_state(
    state: GameState,
    hud_out: Any,
    sa: ScheduledAction,
    action: Action,
    *,
    has_end_round_after: bool = False,
    next_hud_money: int | None = None,
    next_hud_moneys: Sequence[int] | None = None,
    next_hud_round: int | None = None,
    stale_money_drop_guard: bool = False,
    suppress_hands_discards: bool = False,
    rebate_discard_info: tuple[Any, ...] | None = None,
    diag: dict[str, Any] | None = None,
) -> int | float | None:
    """Applies validated HUD OCR readings into `state`, falling back cleanly on `None`.

    Returns a deferred `current_chips` value when `PLAY_HAND` observes a score already at or
    above `target_chips` while an `END_ROUND` action is scheduled afterwards.
    """
    values = getattr(hud_out, "values", None)
    if not isinstance(values, dict) or not values:
        return None

    if diag is not None:
        diag["hud_frames_checked"] += 1
        none_by_field = diag["ocr_none_by_field"]
        for k, v in values.items():
            if v is None:
                none_by_field[k] = none_by_field.get(k, 0) + 1

    observed = {k: v for k, v in values.items() if v is not None}
    if observed:
        action.metadata["hud_ocr"] = observed

    money_val = values.get("hud_money")
    if money_val is not None:
        if sa.action_type == ActionType.CASH_OUT and not state.round_won:
            if diag is not None:
                diag["cashout_without_round_won"].append(sa.frame_id)
            trigger_end_of_round_effects(state, action)
        new_money = int(money_val)
        if (
            sa.ambient_location == Location.BLIND
            and sa.action_type != ActionType.SELECT_BLIND
            and state.hands_played_this_round > 0
            and state.last_hand_played is not None
            and state.inventory is not None
            and new_money - state.money == 4
        ):
            for j in state.inventory.jokers:
                if (
                    j.name == JokerName.TO_DO_LIST
                    and not j.debuffed
                    and j.state.get("target_hand") != state.last_hand_played
                ):
                    j.state["target_hand"] = state.last_hand_played
                    state.to_do_poker_hand = state.last_hand_played
                    state.money += 4
                    break
        lucky_reconciled = False
        lucky_applied = False
        lucky_triggers = int(getattr(state, "_last_play_lucky_triggers", 0) or 0)
        sim_lucky_hits = int(getattr(state, "_last_play_lucky_money_hits", 0) or 0)
        if (
            lucky_triggers > 0
            and sa.ambient_location in (Location.BLIND, Location.CASH_OUT)
            and sa.action_type != ActionType.SELECT_BLIND
        ):
            lucky_diff = new_money - state.money
            lucky_k = int(round(lucky_diff / 20.0)) if lucky_diff != 0 else 0
            lucky_res = abs(lucky_diff - 20 * lucky_k)
            if (
                lucky_k != 0
                and lucky_res <= 5
                and 0 <= sim_lucky_hits + lucky_k <= lucky_triggers
            ):
                if lucky_res == 0:
                    state.money = new_money
                    lucky_reconciled = True
                else:
                    state.money += 20 * lucky_k
                lucky_applied = True
                if state.inventory and state.inventory.jokers:
                    for j in state.inventory.jokers:
                        if j.name == JokerName.LUCKY_CAT and not j.debuffed:
                            prev_xm = float(j.state.get("x_mult", 1.0))
                            j.state["x_mult"] = round(
                                max(1.0, prev_xm + 0.25 * lucky_k), 2
                            )
                state._last_play_lucky_triggers = 0
                state._last_play_lucky_money_hits = 0
                if diag is not None:
                    diag["reconciled_lucky_money"] += 1

        rebate_reconciled = False
        if (
            not lucky_applied
            and rebate_discard_info is not None
            and sa.ambient_location in (Location.BLIND, Location.CASH_OUT)
            and sa.action_type != ActionType.SELECT_BLIND
        ):
            if len(rebate_discard_info) >= 5:
                (
                    rebate_round,
                    rebate_unit,
                    disc_rank_counts,
                    sim_rebate_matches,
                    n_fd_discards,
                ) = rebate_discard_info[:5]
            else:
                (
                    rebate_round,
                    rebate_unit,
                    disc_rank_counts,
                    sim_rebate_matches,
                ) = rebate_discard_info[:4]
                n_fd_discards = 0
            total_disc_non_stone = sum(disc_rank_counts.values()) + n_fd_discards
            rebate_diff = new_money - state.money
            if (
                state.round == rebate_round
                and rebate_unit > 0
                and rebate_diff != 0
                and rebate_diff % rebate_unit == 0
            ):
                true_matches = sim_rebate_matches + (rebate_diff // rebate_unit)
                valid_rebate = (true_matches == 0 and sim_rebate_matches > 0) or (
                    0 < true_matches <= total_disc_non_stone
                    and (
                        any(
                            cnt <= true_matches <= cnt + n_fd_discards
                            for cnt in disc_rank_counts.values()
                        )
                        or n_fd_discards >= true_matches
                    )
                )
                if valid_rebate:
                    if true_matches > 0 and true_matches > n_fd_discards:
                        cand_ranks = [
                            r
                            for r, cnt in disc_rank_counts.items()
                            if cnt <= true_matches <= cnt + n_fd_discards
                        ]
                        if len(cand_ranks) == 1:
                            true_rank = cand_ranks[0]
                            state.mail_card_rank = true_rank
                            if state.inventory and state.inventory.jokers:
                                for j in state.inventory.jokers:
                                    if j.name == JokerName.MAIL_IN_REBATE:
                                        j.state["target_rank"] = true_rank
                    state.money = new_money
                    state._rebate_reconciled = True
                    rebate_reconciled = True
                    if diag is not None:
                        diag["reconciled_rebate_money"] += 1

        next_money_window = (
            list(next_hud_moneys)
            if next_hud_moneys is not None
            else ([next_hud_money] if next_hud_money is not None else [])
        )
        expected_after = state.money
        if (
            sa.action_type == ActionType.REROLL_SHOP
            and getattr(state, "free_rerolls", 0) == 0
        ):
            expected_after = state.money - int(getattr(state, "reroll_cost", 5) or 0)

        if lucky_reconciled or rebate_reconciled:
            pass
        elif (
            sa.action_type != ActionType.CASH_OUT
            and next_hud_money is not None
            and abs(new_money - state.money) >= 15
            and abs(next_hud_money - new_money) >= 15
            and min(
                abs(next_hud_money - state.money),
                abs(next_hud_money - expected_after),
            )
            <= 10
        ):
            # Guard against single-frame rolling-digit OCR truncations (e.g. $58 -> $5 -> $55).
            new_money = state.money
        elif (
            stale_money_drop_guard
            and new_money < state.money
            and (not next_money_window or max(next_money_window) >= state.money)
        ):
            # Ignore stale pre-animation money OCR right after Temperance/Hermit.
            new_money = state.money
        else:
            _record_ocr_override(
                diag,
                frame_id=sa.frame_id,
                action_type=sa.action_type,
                field_name="money",
                prev_val=state.money,
                new_val=new_money,
            )
            state.money = new_money

    hands_val = values.get("hud_hands")
    if (
        not suppress_hands_discards
        and hands_val is not None
        and sa.ambient_location == Location.BLIND
        and sa.action_type != ActionType.SELECT_BLIND
        and int(hands_val) >= 0
    ):
        new_hands = int(hands_val)
        _record_ocr_override(
            diag,
            frame_id=sa.frame_id,
            action_type=sa.action_type,
            field_name="hands_left",
            prev_val=state.hands_left,
            new_val=new_hands,
        )
        state.hands_left = new_hands

    discards_val = values.get("hud_discards")
    if (
        not suppress_hands_discards
        and discards_val is not None
        and sa.ambient_location == Location.BLIND
        and sa.action_type != ActionType.SELECT_BLIND
        and int(discards_val) >= 0
    ):
        new_discards = int(discards_val)
        _record_ocr_override(
            diag,
            frame_id=sa.frame_id,
            action_type=sa.action_type,
            field_name="discards_left",
            prev_val=state.discards_left,
            new_val=new_discards,
        )
        state.discards_left = new_discards

    blind_target_val = values.get("hud_blind_target")
    if (
        blind_target_val is not None
        and sa.ambient_location == Location.BLIND
        and blind_target_val > 0
    ):
        _record_ocr_override(
            diag,
            frame_id=sa.frame_id,
            action_type=sa.action_type,
            field_name="target_chips",
            prev_val=state.target_chips,
            new_val=blind_target_val,
        )
        state.target_chips = blind_target_val

    deferred_round_score: int | float | None = None
    round_score_val = values.get("hud_round_score")
    if (
        round_score_val is not None
        and sa.ambient_location == Location.BLIND
        and round_score_val >= 0
    ):
        if (
            sa.action_type == ActionType.PLAY_HAND
            and has_end_round_after
            and state.target_chips
            and round_score_val >= state.target_chips
        ):
            deferred_round_score = round_score_val
        else:
            if diag is not None and state.current_chips != round_score_val:
                diag["ocr_score_updates"] += 1
            state.current_chips = round_score_val

    prev_ante = state.ante
    ante_val = values.get("hud_ante")
    if (
        ante_val is not None
        and sa.ambient_location == Location.BLIND
        and not state.round_won
        and not getattr(state, "pending_blind_advance", False)
        and 1 <= int(ante_val) <= 8
        and int(ante_val) >= state.ante - 1
    ):
        new_ante = int(ante_val)
        _record_ocr_override(
            diag,
            frame_id=sa.frame_id,
            action_type=sa.action_type,
            field_name="ante",
            prev_val=state.ante,
            new_val=new_ante,
        )
        state.ante = new_ante

    ante_jumped = state.ante != prev_ante
    round_val = values.get("hud_round")
    if (
        round_val is not None
        and sa.ambient_location == Location.BLIND
        and sa.action_type != ActionType.END_ROUND
        and not getattr(state, "round_advanced", False)
        and _is_plausible_hud_round(
            int(round_val),
            state.round,
            state.ante,
            next_hud_round,
            ante_jumped=ante_jumped,
        )
    ):
        new_round = int(round_val)
        _record_ocr_override(
            diag,
            frame_id=sa.frame_id,
            action_type=sa.action_type,
            field_name="round",
            prev_val=state.round,
            new_val=new_round,
        )
        state.round = new_round

    reroll_val = values.get("hud_reroll_cost")
    if (
        reroll_val is not None
        and sa.ambient_location == Location.SHOP
        and int(reroll_val) >= 0
    ):
        reroll_cost = int(reroll_val)
        if sa.action_type == ActionType.REROLL_SHOP and (
            (state.free_rerolls == 0 and reroll_cost == state.reroll_cost + 1)
            or (state.free_rerolls > 0 and reroll_cost == state.reroll_cost)
        ):
            # At frame_id - 2 on Reroll Shop, the depressed button already shows
            # the post-reroll cost that on_action(REROLL_SHOP) is about to set.
            pass
        elif reroll_cost == 0 and state.free_rerolls > 0:
            pass
        elif reroll_cost > 0 and state.free_rerolls > 0:
            _record_ocr_override(
                diag,
                frame_id=sa.frame_id,
                action_type=sa.action_type,
                field_name="reroll_cost",
                prev_val=state.reroll_cost,
                new_val=reroll_cost,
            )
            state.free_rerolls = 0
            state.reroll_cost = reroll_cost
        else:
            _record_ocr_override(
                diag,
                frame_id=sa.frame_id,
                action_type=sa.action_type,
                field_name="reroll_cost",
                prev_val=state.reroll_cost,
                new_val=reroll_cost,
            )
            state.reroll_cost = reroll_cost

    return deferred_round_score


def _display_card_name(card: Any) -> str:
    """Returns a concise human-readable name for any card, joker, voucher, or pack."""
    if card is None:
        return "None"
    rank = getattr(card, "rank", None)
    suit = getattr(card, "suit", None)
    if rank is not None and suit is not None:
        r_str = getattr(rank, "value", str(rank))
        s_str = getattr(suit, "value", str(suit))
        return f"{r_str} of {s_str}"
    name = getattr(card, "name", None)
    if name is not None:
        return str(getattr(name, "value", name))
    return str(card)


def _counter_diff_names(
    before: Sequence[Any], after: Sequence[Any]
) -> tuple[list[str], list[str]]:
    """Returns `(lost_names, gained_names)` multiset differences in original order."""
    after_counts = Counter(_display_card_name(x) for x in after)
    lost: list[str] = []
    for item in before:
        label = _display_card_name(item)
        if after_counts[label] > 0:
            after_counts[label] -= 1
        else:
            lost.append(label)

    before_counts = Counter(_display_card_name(x) for x in before)
    gained: list[str] = []
    for item in after:
        label = _display_card_name(item)
        if before_counts[label] > 0:
            before_counts[label] -= 1
        else:
            gained.append(label)
    return lost, gained


def _is_high_conf_bar_det(raw_item: Any) -> bool:
    """Returns True when a top-bar raw CV item meets minimum detection and class confidence."""
    confs = getattr(raw_item, "confidences", None) or {}
    return (
        float(confs.get("detection", 1.0)) >= 0.60
        and float(confs.get("key", 1.0)) >= 0.85
    )


def _pack_item_sig(item: Any) -> tuple[Any, ...] | None:
    """Canonical signature for booster pack items, ignoring corrupt rank/suit NONE crops."""
    if isinstance(item, PlayingCard):
        if item.enhancement != Enhancement.STONE and (
            item.rank == Rank.NONE or item.suit == Suit.NONE
        ):
            return None
        return (
            "card",
            item.rank,
            item.suit,
            item.enhancement,
            item.edition,
            item.seal,
        )
    return (
        type(item).__name__,
        getattr(item, "name", None),
        getattr(item, "edition", None),
    )


def _filter_bar_observations(
    raw_items: Sequence[Any],
    target_types: type | tuple[type, ...],
    sig_fn: Callable[[Any], Any],
    prev_items: Sequence[Any],
    next_obs_counts: Counter[Any] | None = None,
) -> tuple[list[Any], list[Any]]:
    """Keeps high-confidence new bar items while anchoring slots matching prior inventory."""
    raw_obs: list[tuple[Any, Any]] = []
    for raw_item in raw_items:
        conv = _convert_items_of_type([raw_item], target_types)
        if conv:
            raw_obs.append((raw_item, conv[0]))

    prev_counts = Counter(sig_fn(item) for item in prev_items)
    kept_indices: set[int] = set()
    for sig, prev_n in prev_counts.items():
        ranked_idxs = sorted(
            (i for i, (_, item) in enumerate(raw_obs) if sig_fn(item) == sig),
            key=lambda i: (
                float(
                    (getattr(raw_obs[i][0], "confidences", None) or {}).get(
                        "detection", 1.0
                    )
                ),
                float(
                    (getattr(raw_obs[i][0], "confidences", None) or {}).get("key", 1.0)
                ),
            ),
            reverse=True,
        )
        kept_indices.update(ranked_idxs[:prev_n])

    kept_raw: list[Any] = []
    kept_items: list[Any] = []
    for i, (raw_item, conv_item) in enumerate(raw_obs):
        if i not in kept_indices:
            sig = sig_fn(conv_item)
            next_has_new = (
                next_obs_counts is not None and next_obs_counts[sig] > prev_counts[sig]
            )
            if not (_is_high_conf_bar_det(raw_item) or next_has_new):
                continue
        kept_raw.append(raw_item)
        kept_items.append(conv_item)
    return kept_raw, kept_items


def _preserve_flickering_jokers(
    obs_jokers: Sequence[Joker],
    prev_jokers: Sequence[Joker],
    next_obs_joker_counts: Counter[Any],
) -> list[Joker]:
    """Restores transient missing Jokers (e.g. END_ROUND modal fade) in original slot order."""
    effective = list(obs_jokers)
    obs_counts = Counter(j.name for j in effective)
    prev_counts = Counter(j.name for j in prev_jokers)
    missing: Counter[Any] = Counter()
    for name, prev_n in prev_counts.items():
        expected_keep = min(prev_n, next_obs_joker_counts[name])
        if obs_counts[name] < expected_keep:
            missing[name] = expected_keep - obs_counts[name]
    if missing:
        for orig_idx, prev_j in enumerate(prev_jokers):
            if missing[prev_j.name] > 0:
                effective.insert(min(orig_idx, len(effective)), copy.deepcopy(prev_j))
                missing[prev_j.name] -= 1
    return effective


_SELF_DESTROY_JOKERS: frozenset[JokerName] = frozenset(
    {
        JokerName.GROS_MICHEL,
        JokerName.CAVENDISH,
        JokerName.ICE_CREAM,
        JokerName.POPCORN,
        JokerName.RAMEN,
        JokerName.SELTZER,
        JokerName.TURTLE_BEAN,
        JokerName.MR_BONES,
        JokerName.INVISIBLE_JOKER,
        JokerName.DIET_COLA,
        JokerName.LUCHADOR,
    }
)


def _looks_like_face_down_joker_bar(
    obs_joker_counts: Counter[Any],
    held_jokers: Sequence[Joker],
    location: Location,
) -> bool:
    """Detects Amber Acorn / face-down joker bars when blind tracking missed Amber Acorn."""
    if location != Location.BLIND or len(held_jokers) < 2:
        return False
    held_counts = Counter(j.name for j in held_jokers)
    cert_obs = obs_joker_counts.get(JokerName.CERTIFICATE, 0)
    if cert_obs >= 3 and cert_obs > held_counts.get(JokerName.CERTIFICATE, 0):
        return True
    missing_total = sum(
        max(0, cnt - obs_joker_counts.get(name, 0)) for name, cnt in held_counts.items()
    )
    if missing_total >= 3 and sum(obs_joker_counts.values()) >= len(held_jokers) - 1:
        return True
    return False


def _keep_matching_jokers_only(
    existing: Sequence[Joker], observed: Sequence[Joker]
) -> list[Joker]:
    """Drops missing jokers in original slot order without admitting unconfirmed new jokers."""
    rem = Counter(j.name for j in observed)
    kept: list[Joker] = []
    for prev_j in existing:
        if rem[prev_j.name] > 0:
            kept.append(prev_j)
            rem[prev_j.name] -= 1
    return kept


def _sync_bars_into_state(
    state: GameState,
    shop_out: Any,
    action: Action,
    diag: dict[str, Any] | None = None,
    *,
    next_obs_cons_counts: Counter[tuple[Any, Any]] | None = None,
    next_obs_joker_counts: Counter[Any] | None = None,
    preserve_joker_flickers: bool = False,
    suppress_joker_change_diag: bool = False,
    suppress_consumable_loss_diag: bool = False,
    unconfirmed_rng_cons_ids: set[int] | None = None,
    pending_rng_cons_budget: Counter[type] | None = None,
    unconfirmed_rng_joker_ids: set[int] | None = None,
    pending_rng_joker_budget: int = 0,
    hud_money_val: int | float | None = None,
    allow_new_jokers: bool = True,
    sync_jokers: bool = True,
    sync_consumables: bool = True,
) -> tuple[list[Any], list[Any]]:
    """Syncs Joker and Consumable bars from CV into `state` & records non-identity `joker_order`."""
    raw_jokers = list(getattr(shop_out, "joker_bar", None) or [])
    if sync_jokers:
        prev_jokers = list(state.inventory.jokers) if state.inventory else []
        prev_joker_ids = [j.id for j in prev_jokers]
        raw_jokers, obs_jokers = _filter_bar_observations(
            raw_jokers,
            Joker,
            lambda j: j.name,
            prev_jokers,
            next_obs_joker_counts,
        )
        effective_obs_jokers = (
            _preserve_flickering_jokers(obs_jokers, prev_jokers, next_obs_joker_counts)
            if preserve_joker_flickers
            and next_obs_joker_counts is not None
            and prev_jokers
            else obs_jokers
        )

        if state.inventory and state.inventory.jokers:
            eff_joker_counts = Counter(j.name for j in effective_obs_jokers)
            if (
                state.current_blind == BlindName.AMBER_ACORN
                and not state.boss_disabled
                and any(j.facing_back for j in state.inventory.jokers)
            ) or _looks_like_face_down_joker_bar(
                eff_joker_counts, state.inventory.jokers, state.location
            ):
                # Amber Acorn flips owned jokers face-down, hiding their faces from CV;
                # Stage 2 forces red card backs into bogus classes so preserve tracked jokers.
                pass
            else:
                matched_ids = state.match_cv_joker_order(effective_obs_jokers)
                if matched_ids:
                    if matched_ids != prev_joker_ids:
                        action.joker_order = matched_ids
                elif allow_new_jokers:
                    state.reconcile_cv_jokers(
                        effective_obs_jokers, trigger_added_hooks=False
                    )
                elif next_obs_joker_counts is not None:
                    # On END_ROUND/eval screens, never admit bogus victory-overlay 'Joker's;
                    # only drop jokers that are missing in the subsequent frame too.
                    state.inventory.jokers = _keep_matching_jokers_only(
                        state.inventory.jokers, effective_obs_jokers
                    )
        elif effective_obs_jokers and allow_new_jokers:
            state.reconcile_cv_jokers(effective_obs_jokers, trigger_added_hooks=False)

        if state.inventory is not None and unconfirmed_rng_joker_ids:
            surviving_j_ids = {id(j) for j in state.inventory.jokers}
            removed_unconf_jokers = [
                j
                for j in prev_jokers
                if id(j) in unconfirmed_rng_joker_ids and id(j) not in surviving_j_ids
            ]
            for j in state.inventory.jokers:
                unconfirmed_rng_joker_ids.discard(id(j))
        else:
            removed_unconf_jokers = []

        if (
            diag is not None
            and state.inventory is not None
            and not suppress_joker_change_diag
        ):
            lost_jokers, gained_jokers = _counter_diff_names(
                prev_jokers, state.inventory.jokers
            )
            if removed_unconf_jokers:
                unconfirmed_rng_joker_ids.difference_update(
                    id(j) for j in removed_unconf_jokers
                )
                for rj in removed_unconf_jokers:
                    rj_name = _display_card_name(rj)
                    if rj_name in lost_jokers:
                        lost_jokers.remove(rj_name)
                    if gained_jokers:
                        gained_jokers.pop(0)
                diag["rng_jokers_reconciled"] += len(removed_unconf_jokers)
            if not lost_jokers and gained_jokers and pending_rng_joker_budget > 0:
                reconciled_n = min(len(gained_jokers), pending_rng_joker_budget)
                del gained_jokers[:reconciled_n]
                diag["rng_jokers_reconciled"] += reconciled_n
            if lost_jokers or gained_jokers:
                diag["unexplained_joker_changes"].append(
                    (action.frame_idx, action.type.value, lost_jokers, gained_jokers)
                )

    raw_cons = _remap_softlocked_planets(
        list(getattr(shop_out, "consumeables", None) or []), state
    )
    if sync_consumables:
        prev_cons = list(state.inventory.consumables) if state.inventory else []
        # Prioritize higher-confidence detections when matching existing inventory slots
        # so low-confidence edge slivers cannot displace real held cards into "new" slots.
        raw_cons, obs_cons = _filter_bar_observations(
            raw_cons,
            (Tarot, Planet, Spectral),
            _consumable_sig,
            prev_cons,
            next_obs_cons_counts,
        )
        if state.inventory and (obs_cons or state.inventory.consumables):
            effective_obs_cons = list(obs_cons)
            if next_obs_cons_counts is not None:
                obs_counts = Counter(_consumable_sig(c) for c in effective_obs_cons)
                prev_counts = Counter(
                    _consumable_sig(c) for c in state.inventory.consumables
                )
                for prev_c in state.inventory.consumables:
                    sig = _consumable_sig(prev_c)
                    expected_keep = min(prev_counts[sig], next_obs_cons_counts[sig])
                    if obs_counts[sig] < expected_keep:
                        effective_obs_cons.append(copy.deepcopy(prev_c))
                        obs_counts[sig] += 1

            if len(effective_obs_cons) == len(
                state.inventory.consumables
            ) and _consumable_multiset_matches(
                state.inventory.consumables, effective_obs_cons
            ):
                state.inventory.consumables = _reorder_existing_consumables(
                    state.inventory.consumables, effective_obs_cons
                )
            elif effective_obs_cons:
                state.inventory.consumables = _reconcile_existing_consumables(
                    state.inventory.consumables, effective_obs_cons
                )
            elif next_obs_cons_counts is None or not next_obs_cons_counts:
                state.inventory.consumables = []

        if state.inventory is not None and unconfirmed_rng_cons_ids:
            surviving_ids = {id(c) for c in state.inventory.consumables}
            removed_unconf = [
                c
                for c in prev_cons
                if id(c) in unconfirmed_rng_cons_ids and id(c) not in surviving_ids
            ]
            # Any unconfirmed RNG consumable preserved in state is now CV-confirmed.
            for c in state.inventory.consumables:
                unconfirmed_rng_cons_ids.discard(id(c))
        else:
            removed_unconf = []

        if diag is not None and state.inventory is not None:
            lost_cons, gained_cons = _counter_diff_names(
                prev_cons, state.inventory.consumables
            )
            if removed_unconf:
                unconfirmed_rng_cons_ids.difference_update(
                    id(c) for c in removed_unconf
                )
                for rc in removed_unconf:
                    rc_name = _display_card_name(rc)
                    if rc_name in lost_cons:
                        lost_cons.remove(rc_name)
                    if gained_cons:
                        gained_cons.pop(0)
                diag["rng_cons_reconciled"] += len(removed_unconf)
            if suppress_consumable_loss_diag:
                lost_cons = []
            if not lost_cons and gained_cons and pending_rng_cons_budget:
                prev_id_set = {id(c) for c in prev_cons}
                new_objs_by_name: dict[str, list[Any]] = {}
                for c in state.inventory.consumables:
                    if id(c) not in prev_id_set:
                        new_objs_by_name.setdefault(_display_card_name(c), []).append(c)
                remaining_gained: list[str] = []
                for g_name in gained_cons:
                    cands = new_objs_by_name.get(g_name)
                    matched_type = type(cands.pop(0)) if cands else None
                    if (
                        matched_type is not None
                        and pending_rng_cons_budget[matched_type] > 0
                    ):
                        pending_rng_cons_budget[matched_type] -= 1
                        diag["rng_cons_reconciled"] += 1
                    else:
                        remaining_gained.append(g_name)
                gained_cons = remaining_gained
            if (
                not lost_cons
                and len(gained_cons) == 1
                and state.location == Location.SHOP
                and hud_money_val is not None
                and int(hud_money_val) < state.money
            ):
                prev_id_set = {id(c) for c in prev_cons}
                new_cons_objs = [
                    c for c in state.inventory.consumables if id(c) not in prev_id_set
                ]
                if len(new_cons_objs) == 1:
                    has_astro = any(
                        j.name == JokerName.ASTRONOMER and not j.debuffed
                        for j in state.inventory.jokers
                    )
                    est_cost = new_cons_objs[0].get_cost(
                        discount_percent=state.discount_percent,
                        inflation=state.inflation,
                        astronomer=has_astro,
                    )
                    if est_cost > 0 and abs((state.money - int(hud_money_val)) - est_cost) <= 1:
                        state.money -= est_cost
                        gained_cons = []
                        diag["inferred_joker_buys"] += 1
            if lost_cons or gained_cons:
                diag["unexplained_consumable_changes"].append(
                    (action.frame_idx, action.type.value, lost_cons, gained_cons)
                )

    return raw_jokers, raw_cons


def _consumable_sig(card: Any) -> tuple[Any, Any]:
    return (getattr(card, "name", None), getattr(card, "edition", None))


def _consumable_multiset_matches(a: Sequence[Any], b: Sequence[Any]) -> bool:
    if len(a) != len(b):
        return False
    used: set[int] = set()
    for item_b in b:
        sig_b = _consumable_sig(item_b)
        found = False
        for i, item_a in enumerate(a):
            if i not in used and _consumable_sig(item_a) == sig_b:
                used.add(i)
                found = True
                break
        if not found:
            return False
    return True


def _reconcile_existing_consumables(
    existing: Sequence[Any], observed: Sequence[Any]
) -> list[Any]:
    used: set[int] = set()
    reconciled: list[Any] = []
    for obs in observed:
        sig_obs = _consumable_sig(obs)
        match_obj = None
        for i, prev in enumerate(existing):
            if i not in used and _consumable_sig(prev) == sig_obs:
                used.add(i)
                match_obj = prev
                break
        reconciled.append(match_obj if match_obj is not None else copy.deepcopy(obs))
    return reconciled


def _reorder_existing_consumables(
    existing: Sequence[Any], observed: Sequence[Any]
) -> list[Any]:
    return _reconcile_existing_consumables(existing, observed)


_HAND_SELECTION_ACTIONS: frozenset[ActionType] = frozenset(
    {
        ActionType.PLAY_HAND,
        ActionType.DISCARD,
        ActionType.USE_CONSUMABLE,
        ActionType.USE_IN_PACK,
    }
)


_RANDOM_DESTROY_SPECTRALS = frozenset({"Immolate", "Familiar", "Grim", "Incantation"})


def _collect_pack_surviving_hand_ids(
    state: GameState,
    scheduled_actions: Sequence[ScheduledAction],
    sa_idx: int,
    hand_res_by_frame: dict[int, Any],
) -> list[int]:
    """Identifies card IDs in current hand that still appear in the immediate next pack observation."""
    if not state.hand.cards or sa_idx + 1 >= len(scheduled_actions):
        return []
    next_sa = scheduled_actions[sa_idx + 1]
    cur_sa = scheduled_actions[sa_idx]
    still_in_same_pack = (
        state.location == Location.PACK
        and next_sa.ambient_location == Location.PACK
        and (
            cur_sa.action_type == ActionType.USE_CONSUMABLE
            or state.pack_choices_remaining > 1
        )
    )
    if not still_in_same_pack:
        return []
    next_hand = hand_res_by_frame.get(next_sa.target_frame)
    if next_hand is None or not getattr(next_hand, "cards", None):
        return []
    obs_cards = [_convert_card_result_to_hand_card(c) for c in next_hand.cards]
    matched_ids: list[int] = []
    used_indices: set[int] = set()
    for obs in obs_cards:
        if obs.facing_back:
            continue
        hit_idx = next(
            (
                j
                for j, hc in enumerate(state.hand.cards)
                if j not in used_indices and _cards_match(hc.card, obs.card)
            ),
            None,
        )
        if hit_idx is None:
            hit_idx = next(
                (
                    j
                    for j, hc in enumerate(state.hand.cards)
                    if j not in used_indices
                    and hc.card.rank == obs.card.rank
                    and hc.card.suit == obs.card.suit
                ),
                None,
            )
        if hit_idx is not None:
            used_indices.add(hit_idx)
            cid = state.hand.cards[hit_idx].card.id
            if cid:
                matched_ids.append(cid)
    return matched_ids


def _collect_hook_discard_ids(
    state: GameState,
    scheduled_actions: Sequence[ScheduledAction],
    sa_idx: int,
    hand_res_by_frame: dict[int, Any],
) -> list[int]:
    """Identifies up to 2 unplayed hand card IDs that vanished under The Hook's forced discard."""
    if (
        state.current_blind != BlindName.THE_HOOK
        or state.boss_disabled
        or not state.hand.cards
    ):
        return []
    unplayed = [hc for hc in state.hand.cards if not hc.selected]
    if not unplayed:
        return []

    next_hand = None
    for nxt in scheduled_actions[sa_idx + 1 :]:
        if nxt.action_type in (
            ActionType.END_ROUND,
            ActionType.CASH_OUT,
            ActionType.SELECT_BLIND,
            ActionType.SKIP_BLIND,
            ActionType.NEXT_ROUND,
        ):
            break
        if nxt.ambient_location != Location.BLIND:
            break
        if nxt.action_type in (ActionType.SORT_RANK, ActionType.SORT_SUIT):
            continue
        next_hand = hand_res_by_frame.get(nxt.target_frame)
        break

    if next_hand is None or not getattr(next_hand, "cards", None):
        return []

    obs_cards = [_convert_card_result_to_hand_card(c) for c in next_hand.cards]
    used_next: set[int] = set()
    vanished_ids: list[int] = []
    for hc in unplayed:
        hit_idx = next(
            (
                j
                for j, obs in enumerate(obs_cards)
                if j not in used_next
                and not obs.facing_back
                and _cards_match(hc.card, obs.card)
            ),
            None,
        )
        if hit_idx is None:
            hit_idx = next(
                (
                    j
                    for j, obs in enumerate(obs_cards)
                    if j not in used_next
                    and not obs.facing_back
                    and hc.card.rank == obs.card.rank
                    and hc.card.suit == obs.card.suit
                ),
                None,
            )
        if hit_idx is not None:
            used_next.add(hit_idx)
        elif hc.card.id:
            vanished_ids.append(hc.card.id)
    return vanished_ids[:2]


def _sync_hand_into_state(
    state: GameState,
    obs_hand: Sequence[HandCard],
    *,
    action: Action | None = None,
    selected_slot_uid: int | None = None,
    apply_direct_order: bool = False,
    pending_cert_card_ids: set[int] | None = None,
) -> int | None:
    """Reconciles CV-observed hand cards into `state.hand.cards` and populates Action fields.

    Preserves persistent `PlayingCard.id` and `HandCard.slot_uid` across draws and drags.
    Returns the `slot_uid` of the first raised/selected card if any.
    """
    state.ensure_slot_uids()
    existing = list(state.hand.cards)
    used_existing: set[int] = set()
    matched_slots: list[HandCard | None] = [None] * len(obs_hand)

    # 1. If tracking a pre-sort selected face-down slot across a sort, anchor that slot first
    if selected_slot_uid is not None:
        for i, obs in enumerate(obs_hand):
            if obs.selected and obs.facing_back:
                for j, ex in enumerate(existing):
                    if j not in used_existing and ex.slot_uid == selected_slot_uid:
                        matched_slots[i] = ex
                        used_existing.add(j)
                        break
                break

    # 2. Match visible face-up cards against previously held face-up cards
    for i, obs in enumerate(obs_hand):
        if matched_slots[i] is not None or obs.facing_back:
            continue
        for j, ex in enumerate(existing):
            if j in used_existing or ex.facing_back:
                continue
            if _cards_match(ex.card, obs.card):
                matched_slots[i] = ex
                used_existing.add(j)
                break

    # 3. Match face-up observations against previously face-down slots (e.g., after resolve/reveal)
    live_deck_ids = {dc.id for dc in state.full_deck if dc.id != 0}

    def _is_live_candidate(c: PlayingCard) -> bool:
        return not getattr(c, "destroyed", False) and (
            c.id == 0 or c.id in live_deck_ids
        )

    for i, obs in enumerate(obs_hand):
        if matched_slots[i] is not None or obs.facing_back:
            continue
        for j, ex in enumerate(existing):
            if j in used_existing:
                continue
            cand = ex._real_card or ex.card
            if (
                cand.rank != Rank.NONE
                and _is_live_candidate(cand)
                and _cards_match(cand, obs.card)
            ):
                ex.resolve_face_down(copy.deepcopy(obs.card), lift_facing_back=True)
                state.sync_card_to_deck(ex.card, rebind_hand_card=ex)
                matched_slots[i] = ex
                used_existing.add(j)
                break
            if ex.bounds and any(
                _is_live_candidate(c) and _cards_match(c, obs.card)
                for c in ex.bounds.candidate_cards
            ):
                match_cand = next(
                    c
                    for c in ex.bounds.candidate_cards
                    if _is_live_candidate(c) and _cards_match(c, obs.card)
                )
                ex.resolve_face_down(match_cand, lift_facing_back=True)
                state.sync_card_to_deck(ex.card, rebind_hand_card=ex)
                matched_slots[i] = ex
                used_existing.add(j)
                break

    # 4. Match remaining face-down observations against remaining unused face-down slots
    for i, obs in enumerate(obs_hand):
        if matched_slots[i] is not None or not obs.facing_back:
            continue
        for j, ex in enumerate(existing):
            if j not in used_existing and ex.facing_back:
                matched_slots[i] = ex
                used_existing.add(j)
                break

    # 5. Any remaining unmatched observations are newly drawn cards from remaining_deck
    claimed_deck_ids = {
        hc.card.id
        for hc in matched_slots
        if hc is not None and hc.card and hc.card.id != 0
    }
    newly_drawn: list[HandCard] = []
    next_deck_id = max((c.id for c in state.full_deck), default=0) + 1

    for i, obs in enumerate(obs_hand):
        hc = matched_slots[i]
        if hc is not None:
            hc.selected = bool(obs.selected)
            if not obs.facing_back:
                hc.facing_back = False
                hc.card.debuffed = bool(obs.card.debuffed)
            if pending_cert_card_ids and hc.card.id:
                pending_cert_card_ids.discard(hc.card.id)
            continue

        if obs.facing_back:
            new_hc = HandCard(
                card=copy.deepcopy(obs.card),
                selected=bool(obs.selected),
                facing_back=True,
                slot_uid=state._get_next_slot_uid(),
                bounds=CardBounds(is_face_down=True),
            )
            new_hc.bounds.set_candidates(
                [
                    c
                    for c in state.get_face_down_candidate_pool(new_hc)
                    if c.id not in claimed_deck_ids
                ]
            )
        else:
            rem_match = None
            for deck_card in state.remaining_deck:
                if deck_card.id not in claimed_deck_ids and _cards_match(
                    deck_card, obs.card
                ):
                    rem_match = deck_card
                    break
            if (
                rem_match is None
                and not obs.facing_back
                and obs.card.enhancement != Enhancement.NONE
            ):
                for deck_card in state.remaining_deck:
                    if (
                        deck_card.id not in claimed_deck_ids
                        and deck_card.rank == obs.card.rank
                        and deck_card.suit == obs.card.suit
                        and deck_card.seal == obs.card.seal
                        and deck_card.edition == obs.card.edition
                        and deck_card.enhancement != Enhancement.NONE
                    ):
                        rem_match = deck_card
                        break
            if (
                rem_match is None
                and state.current_blind == BlindName.THE_HOOK
                and not state.boss_disabled
                and state.discard_pile
            ):
                played_ids = {
                    c.id for c in state.played_cards_this_round if getattr(c, "id", 0)
                }
                for d_idx, disc_card in enumerate(state.discard_pile):
                    if (
                        disc_card.id not in claimed_deck_ids
                        and disc_card.id not in played_ids
                        and _cards_match(disc_card, obs.card)
                    ):
                        rem_match = state.discard_pile.pop(d_idx)
                        break
            if rem_match is None:
                is_corrupt_crop = (
                    not obs.facing_back
                    and obs.card.enhancement != Enhancement.STONE
                    and (obs.card.rank == Rank.NONE or obs.card.suit == Suit.NONE)
                )
                if is_corrupt_crop:
                    continue
                if (
                    pending_cert_card_ids
                    and not obs.facing_back
                    and obs.card.seal != Seal.NONE
                ):
                    cert_match = next(
                        (
                            dc
                            for dc in state.full_deck
                            if dc.id in pending_cert_card_ids
                            and dc.id not in claimed_deck_ids
                        ),
                        None,
                    )
                    if cert_match is not None:
                        cert_match.rank = obs.card.rank
                        cert_match.suit = obs.card.suit
                        cert_match.seal = obs.card.seal
                        cert_match.enhancement = obs.card.enhancement
                        cert_match.edition = obs.card.edition
                        cert_match.debuffed = bool(obs.card.debuffed)
                        pending_cert_card_ids.discard(cert_match.id)
                        rem_match = cert_match
                if rem_match is None:
                    rem_match = copy.deepcopy(obs.card)
                    rem_match.id = next_deck_id
                    next_deck_id += 1
                    state.full_deck.append(rem_match)
            else:
                rem_match.debuffed = bool(obs.card.debuffed)
            if pending_cert_card_ids and rem_match.id:
                pending_cert_card_ids.discard(rem_match.id)
            claimed_deck_ids.add(rem_match.id)
            new_hc = HandCard(
                card=rem_match,
                selected=bool(obs.selected),
                facing_back=False,
                slot_uid=state._get_next_slot_uid(),
            )
        matched_slots[i] = new_hc
        newly_drawn.append(new_hc)

    ordered_hand = [hc for hc in matched_slots if hc is not None]
    retained_in_prev_order = [ex for j, ex in enumerate(existing) if j in used_existing]
    pre_reorder = retained_in_prev_order + newly_drawn

    if apply_direct_order:
        state.hand.cards = ordered_hand
    else:
        state.hand.cards = pre_reorder
        if action is not None and ordered_hand:
            obj_pos = {id(hc): idx for idx, hc in enumerate(pre_reorder)}
            hand_order = [obj_pos[id(hc)] for hc in ordered_hand]
            if hand_order != list(range(len(ordered_hand))):
                action.hand_order = hand_order

    if action is not None and action.type in _HAND_SELECTION_ACTIONS:
        action.selected_cards = [
            copy.deepcopy(hc.card) for hc in ordered_hand if hc.selected
        ]

    state.reconcile_face_down_adjacencies()

    first_selected = next((hc for hc in ordered_hand if hc.selected), None)
    return first_selected.slot_uid if first_selected is not None else None


def _sync_full_deck_into_state(
    state: GameState,
    obs_cards: Sequence[PlayingCard],
    *,
    obs_suit_confs: Sequence[float | None] | None = None,
    obs_rank_confs: Sequence[float | None] | None = None,
    view_mode: str = "full_deck",
    phantom_adds: int = 0,
    phantom_card_ids: set[int] | None = None,
    diag: dict[str, Any] | None = None,
    frame_id: int = 0,
    reasons: Sequence[str] = (),
) -> dict[str, Any] | None:
    """Reconciles View Deck CV predictions into `state.full_deck` on suspected drift.

    Preserves existing `PlayingCard.id` values, live card object references held in
    `hand`/`discard_pile`/`played_cards_this_round`, and relative deck order where matched.
    Returns a diagnostic summary dict when applied, or None if rejected by sanity checks.
    """
    clean_obs: list[PlayingCard] = []
    clean_suit_confs: list[float | None] = []
    clean_rank_confs: list[float | None] = []
    for idx_obs, c in enumerate(obs_cards):
        if c.enhancement != Enhancement.STONE and (
            c.rank == Rank.NONE or c.suit == Suit.NONE
        ):
            continue
        clean_obs.append(c)
        clean_suit_confs.append(
            obs_suit_confs[idx_obs]
            if obs_suit_confs is not None and idx_obs < len(obs_suit_confs)
            else None
        )
        clean_rank_confs.append(
            obs_rank_confs[idx_obs]
            if obs_rank_confs is not None and idx_obs < len(obs_rank_confs)
            else None
        )
    if not clean_obs:
        return None

    before_len = len(state.full_deck)
    extra_phantom = max(0, int(phantom_adds))
    if view_mode == "remaining":
        cur_rem_len = len(state.remaining_deck)
        if not (max(0, cur_rem_len - 12) <= len(clean_obs) <= cur_rem_len + 12):
            return None
    else:
        min_allowed = max(5, before_len - 8 - extra_phantom)
        if extra_phantom > 0 and len(clean_obs) < min(
            15, max(5, before_len - extra_phantom)
        ):
            return None
        if not (min_allowed <= len(clean_obs) <= before_len + 12):
            return None

    active_hand_ids = {hc.card.id for hc in state.hand.cards if hc.card.id}
    active_hand_ptrs = {id(hc.card) for hc in state.hand.cards}
    discard_ids = {c.id for c in state.discard_pile if getattr(c, "id", 0)}
    discard_ptrs = {id(c) for c in state.discard_pile}
    if view_mode == "remaining":
        protected_ids = (
            active_hand_ids
            | discard_ids
            | {c.id for c in state.played_cards_this_round if getattr(c, "id", 0)}
        )
        protected_ptrs = (
            active_hand_ptrs
            | discard_ptrs
            | {id(c) for c in state.played_cards_this_round}
        )
    else:
        protected_ids = set(active_hand_ids)
        protected_ptrs = set(active_hand_ptrs)

    def _is_protected(idx: int) -> bool:
        c = state.full_deck[idx]
        return (c.id != 0 and c.id in protected_ids) or (id(c) in protected_ptrs)

    target_indices = [
        idx
        for idx in range(len(state.full_deck))
        if view_mode != "remaining" or not _is_protected(idx)
    ]
    priority_order = sorted(
        target_indices,
        key=lambda idx: (
            0
            if (
                (
                    state.full_deck[idx].id != 0
                    and state.full_deck[idx].id in (active_hand_ids | discard_ids)
                )
                or id(state.full_deck[idx]) in (active_hand_ptrs | discard_ptrs)
            )
            else 1,
            idx,
        ),
    )

    matched_obs: set[int] = set()
    matched_target: set[int] = set()
    pending_updates: list[tuple[int, PlayingCard, bool]] = []
    obs_match_order = sorted(
        range(len(clean_obs)),
        key=lambda i: (
            0
            if (
                (clean_suit_confs[i] is None or clean_suit_confs[i] >= 0.90)
                and (clean_rank_confs[i] is None or clean_rank_confs[i] >= 0.92)
            )
            else 1,
            i,
        ),
    )

    # Pass 1: exact matches (high-confidence ranks/suits first, preferring live hand/discard cards)
    for i in obs_match_order:
        obs = clean_obs[i]
        for dk_idx in priority_order:
            if dk_idx in matched_target:
                continue
            if _cards_match(state.full_deck[dk_idx], obs):
                matched_obs.add(i)
                matched_target.add(dk_idx)
                break

    # Pass 2: same (rank, suit) in-place attribute reconciliation preserving card.id (staged)
    for i in obs_match_order:
        obs = clean_obs[i]
        if i in matched_obs or obs.rank == Rank.NONE or obs.suit == Suit.NONE:
            continue
        r_conf = clean_rank_confs[i]
        if r_conf is not None and r_conf < 0.92:
            continue
        for dk_idx in priority_order:
            if dk_idx in matched_target:
                continue
            cand = state.full_deck[dk_idx]
            if cand.rank == obs.rank and cand.suit == obs.suit:
                pending_updates.append((dk_idx, obs, True))
                matched_obs.add(i)
                matched_target.add(dk_idx)
                break

    # Pass 2b: When CV suit confidence is low (< 0.90), preserve simulated suit if
    # (rank, enhancement, edition, seal) already matches an unmatched deck card.
    if obs_suit_confs is not None:
        for i, obs in enumerate(clean_obs):
            if i in matched_obs or obs.rank == Rank.NONE:
                continue
            s_conf = clean_suit_confs[i]
            if s_conf is None or s_conf >= 0.90:
                continue
            for dk_idx in priority_order:
                if dk_idx in matched_target:
                    continue
                cand = state.full_deck[dk_idx]
                if (
                    cand.rank == obs.rank
                    and cand.enhancement == obs.enhancement
                    and cand.edition == obs.edition
                    and cand.seal == obs.seal
                ):
                    pending_updates.append((dk_idx, obs, False))
                    matched_obs.add(i)
                    matched_target.add(dk_idx)
                    break

    # Pass 2c: When CV rank confidence is low (< 0.94) on occluded View Deck corners,
    # preserve simulated rank if (suit, enhancement, edition, seal) matches an unmatched deck card.
    if obs_rank_confs is not None:
        for i, obs in enumerate(clean_obs):
            if i in matched_obs or obs.suit == Suit.NONE:
                continue
            r_conf = clean_rank_confs[i]
            if r_conf is None or r_conf >= 0.94:
                continue
            for dk_idx in priority_order:
                if dk_idx in matched_target:
                    continue
                cand = state.full_deck[dk_idx]
                if (
                    cand.suit == obs.suit
                    and cand.enhancement == obs.enhancement
                    and cand.edition == obs.edition
                    and cand.seal == obs.seal
                ):
                    pending_updates.append((dk_idx, obs, False))
                    matched_obs.add(i)
                    matched_target.add(dk_idx)
                    break

    # Pass 3: compute unmatched deletions & additions BEFORE committing any state mutations
    remove_indices = {
        dk_idx
        for dk_idx in target_indices
        if dk_idx not in matched_target and not _is_protected(dk_idx)
    }
    new_obs_indices = [i for i in range(len(clean_obs)) if i not in matched_obs]

    # Guard against View-Deck occlusion churn when the ONLY trigger was phantom hand additions:
    # never overwrite known deterministic non-phantom cards with unseen View-Deck rank misreads.
    if (
        phantom_card_ids
        and reasons
        and set(reasons) <= {"unexpected_deck_addition"}
        and len(new_obs_indices) >= 1
    ):
        phantom_remove = {
            dk_idx
            for dk_idx in remove_indices
            if state.full_deck[dk_idx].id in phantom_card_ids
        }
        if phantom_remove:
            remove_indices = phantom_remove
            new_obs_indices = []
            pending_updates = [
                (dk_idx, obs, upd)
                for dk_idx, obs, upd in pending_updates
                if state.full_deck[dk_idx].id in phantom_card_ids
            ]

    removed_cnt = len(remove_indices)
    added_cnt = len(new_obs_indices)
    after_len_est = before_len - removed_cnt + added_cnt
    net_shrink = before_len - after_len_est
    if removed_cnt > (6 + extra_phantom) and net_shrink > (4 + extra_phantom):
        return None

    updated_cnt = 0
    for dk_idx, obs, update_attrs in pending_updates:
        cand = state.full_deck[dk_idx]
        if update_attrs:
            cand.enhancement = obs.enhancement
            cand.edition = obs.edition
            cand.seal = obs.seal
            updated_cnt += 1
        cand.debuffed = bool(obs.debuffed)

    if remove_indices:
        state.full_deck = [
            c for idx, c in enumerate(state.full_deck) if idx not in remove_indices
        ]

    # Pass 4: append remaining unmatched observed cards with fresh unique IDs
    next_id = max((c.id for c in state.full_deck), default=0) + 1
    for i in new_obs_indices:
        new_card = copy.deepcopy(clean_obs[i])
        new_card.id = next_id
        next_id += 1
        state.full_deck.append(new_card)

    summary = {
        "frame_id": int(frame_id),
        "view_mode": view_mode,
        "reasons": list(reasons),
        "before": before_len,
        "after": len(state.full_deck),
        "added": added_cnt,
        "removed": removed_cnt,
        "updated": updated_cnt,
    }
    if diag is not None:
        diag["full_deck_syncs"].append(summary)
    return summary


def _resolve_video_fps(video_path: str | Path | None, fps: float | None) -> float:
    resolved_fps = fps
    if resolved_fps is None and video_path is not None:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        cap_fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
        cap.release()
        resolved_fps = float(cap_fps) if cap_fps and cap_fps > 0 else 60.0
    if not resolved_fps or resolved_fps <= 0:
        resolved_fps = 60.0
    return resolved_fps


def _resolve_target_card_name(state: GameState, action: Action) -> str | None:
    """Resolves the display name of `action.target_idx` right before `on_action` runs."""
    idx = action.target_idx
    if idx is None or idx < 0:
        return None
    A = ActionType
    if action.type in (A.BUY_CARD, A.BUY_AND_USE):
        pool: Sequence[Any] = state.shop_cards
    elif action.type == A.OPEN_PACK:
        pool = state.shop_packs
    elif action.type == A.REDEEM_VOUCHER:
        pool = state.shop_vouchers
    elif action.type == A.USE_IN_PACK:
        pool = state.pack_cards
    elif action.type == A.USE_CONSUMABLE:
        pool = state.inventory.consumables if state.inventory else []
    elif action.type == A.SELL_CARD:
        jokers = list(state.inventory.jokers) if state.inventory else []
        if action.joker_order and jokers:
            by_id = {j.id: j for j in jokers if j.id != 0}
            if len(action.joker_order) == len(jokers) and all(
                jid in by_id for jid in action.joker_order
            ):
                jokers = [by_id[jid] for jid in action.joker_order]
            elif len(action.joker_order) == len(jokers) and all(
                isinstance(x, int) and 0 <= x < len(jokers) for x in action.joker_order
            ):
                jokers = [jokers[i] for i in action.joker_order]
        consumables = list(state.inventory.consumables) if state.inventory else []
        pool = jokers + consumables
    else:
        return None

    if idx < len(pool):
        return _display_card_name(pool[idx])
    return None


def _serialize_shop_shelves(state: GameState) -> dict[str, list[dict[str, Any]]]:
    """Serializes `state`'s visible shop shelves for pre-action metadata snapshots."""
    from game.trajectory_codec import (
        _serialize_booster_pack,
        _serialize_card,
        _serialize_voucher,
    )

    return {
        "cards": [_serialize_card(c) for c in state.shop_cards],
        "vouchers": [_serialize_voucher(v) for v in state.shop_vouchers],
        "packs": [_serialize_booster_pack(bp) for bp in state.shop_packs],
    }


def _detect_unexpected_shop_transition(
    prev_obs: dict[str, Any] | None,
    cur_cards: Sequence[str],
    cur_vouchers: Sequence[str],
    cur_packs: Sequence[str],
    *,
    actions_between: Sequence[ActionType],
    shop_visit_opens: int,
    cashout_frame: int | None = None,
) -> list[str]:
    """Detects unexpected shop shelf changes between consecutive shop observations."""
    if not (cur_cards or cur_vouchers or cur_packs):
        return []
    if prev_obs is None:
        if shop_visit_opens == 0 and len(cur_packs) < 2:
            return [
                f"first_shop_obs_missing_packs (seen={list(cur_packs)}, expected=2)"
            ]
        return []

    n_reroll = sum(1 for a in actions_between if a == ActionType.REROLL_SHOP)
    n_buy = sum(
        1 for a in actions_between if a in (ActionType.BUY_CARD, ActionType.BUY_AND_USE)
    )
    n_open = sum(1 for a in actions_between if a == ActionType.OPEN_PACK)
    n_redeem = sum(1 for a in actions_between if a == ActionType.REDEEM_VOUCHER)

    def _multiset_diff(
        before: Sequence[str], after: Sequence[str]
    ) -> tuple[list[str], list[str]]:
        a_cnt = Counter(after)
        lost = []
        for x in before:
            if a_cnt[x] > 0:
                a_cnt[x] -= 1
            else:
                lost.append(x)
        b_cnt = Counter(before)
        gain = []
        for x in after:
            if b_cnt[x] > 0:
                b_cnt[x] -= 1
            else:
                gain.append(x)
        return lost, gain

    c_lost, c_gain = _multiset_diff(prev_obs["cards"], cur_cards)
    v_lost, v_gain = _multiset_diff(prev_obs["vouchers"], cur_vouchers)
    p_lost, p_gain = _multiset_diff(prev_obs["packs"], cur_packs)

    reasons: list[str] = []
    if n_reroll == 0:
        if len(prev_obs["cards"]) - len(cur_cards) > n_buy:
            reasons.append(f"cards_lost={c_lost} (buys={n_buy})")
        if (
            c_gain
            and n_redeem == 0
            and (
                len(cur_cards) > len(prev_obs["cards"])
                or len(c_lost) > n_buy
                or (len(prev_obs["cards"]) >= 2 and len(c_gain) >= 2)
            )
        ):
            # Ignore single-card slot refills/swaps after a Buy/Buy-and-Use without net shop growth
            if not (
                n_buy >= 1
                and len(cur_cards) <= len(prev_obs["cards"])
                and len(c_gain) == 1
            ):
                reasons.append(f"cards_gained={c_gain} (rerolls=0)")
    if len(v_lost) > n_redeem:
        reasons.append(f"vouchers_lost={v_lost} (redeems={n_redeem})")
    if v_gain:
        reasons.append(f"vouchers_gained={v_gain}")
    if len(p_lost) > n_open:
        reasons.append(f"packs_lost={p_lost} (opens={n_open})")
    if p_gain:
        reasons.append(f"packs_gained={p_gain}")
    return reasons


_EDITION_CLS_LABELS: dict[Edition, str] = {
    Edition.NONE: "None",
    Edition.FOIL: "Foil",
    Edition.HOLOGRAPHIC: "Holo",
    Edition.POLYCHROME: "Polychrome",
    Edition.NEGATIVE: "None",
}

_SUIT_CLS_LABELS: dict[Suit, str] = {
    Suit.HEARTS: "Hearts",
    Suit.DIAMONDS: "Diamonds",
    Suit.CLUBS: "Clubs",
    Suit.SPADES: "Spades",
    Suit.NONE: "None",
}

_CLS_HEADS: tuple[str, ...] = ("rank", "suit", "enhancement", "edition", "seal")


def _playing_card_cls_labels(card: PlayingCard) -> dict[str, str]:
    rank_lbl = (
        "Stone"
        if card.enhancement == Enhancement.STONE
        else ("None" if card.rank == Rank.NONE else card.rank.value)
    )
    suit_lbl = _SUIT_CLS_LABELS.get(card.suit, "None")
    enh_lbl = (
        "None"
        if card.enhancement == Enhancement.NONE
        else card.enhancement.value.capitalize()
    )
    ed_lbl = _EDITION_CLS_LABELS.get(card.edition, "None")
    seal_lbl = "None" if card.seal == Seal.NONE else card.seal.value.capitalize()
    return {
        "rank": rank_lbl,
        "suit": suit_lbl,
        "enhancement": enh_lbl,
        "edition": ed_lbl,
        "seal": seal_lbl,
    }


def _card_attr_sig(card: PlayingCard) -> tuple[Rank, Suit, Enhancement, Edition, Seal]:
    return (card.rank, card.suit, card.enhancement, card.edition, card.seal)


def _raw_head_prob(
    raw_card: Any, head_name: str, target_label: str, top1_label: str
) -> float:
    top_probs = getattr(raw_card, "top_probs", None) or {}
    head_dict = top_probs.get(head_name)
    if isinstance(head_dict, dict) and target_label in head_dict:
        return float(head_dict[target_label])
    if target_label == top1_label:
        confs = getattr(raw_card, "confidences", None) or {}
        return float(confs.get(head_name, 0.99))
    return 0.0


def _flag_corrupted_action(
    action: Action,
    reason: str,
    *,
    diag: dict[str, Any] | None = None,
) -> None:
    """Marks an Action/step as corrupted so downstream training pipelines can exclude it."""
    action.metadata["corrupted"] = True
    reasons = action.metadata.setdefault("corruption_reasons", [])
    if reason not in reasons:
        reasons.append(reason)
    if diag is not None:
        entry = (action.frame_idx, action.type.value, reason)
        if entry not in diag["corrupted_actions"]:
            diag["corrupted_actions"].append(entry)


def _check_over_15_hand_size(
    state: GameState,
    action: Action,
    obs_cards: Sequence[HandCard],
    *,
    has_hand_cv: bool,
    diag: dict[str, Any] | None = None,
) -> bool:
    """Flags hand-observing steps as corrupted when expected or observed hand size exceeds 15."""
    expected_hand = (
        min(
            state.hand.max_size,
            len(state.hand.cards) + len(state.remaining_deck),
        )
        if state.full_deck
        else state.hand.max_size
    )
    in_hand_view = (
        state.location == Location.BLIND and not state.round_won and has_hand_cv
    ) or len(obs_cards) > 0
    if not in_hand_view or (expected_hand <= 15 and len(obs_cards) <= 15):
        return False

    action.metadata["expected_hand_max_size"] = int(expected_hand)
    action.metadata["detected_hand_cards"] = int(len(obs_cards))
    _flag_corrupted_action(action, "hand_size_over_15", diag=diag)
    if diag is not None:
        diag["hand_size_over_15_actions"].append(
            (
                action.frame_idx,
                action.type.value,
                int(expected_hand),
                int(len(obs_cards)),
            )
        )
    return True


def _scan_and_quarantine_corrupted_segments(
    trajectory: RunTrajectory,
    step_records: Sequence[dict[str, Any]],
    *,
    diag: dict[str, Any] | None = None,
) -> int | None:
    """Scans trajectory steps for local, round-scoped, and cascading permanent corruption.

    Marks affected steps via `_flag_corrupted_action` so training datasets can exclude
    poisoned states (such as corrupted `full_deck` / inventory after an anomalous segment).
    Returns the earliest permanent-tail quarantine step index, or None if not triggered.
    """
    n_steps = len(trajectory.deltas)
    if n_steps <= 1 or len(step_records) != n_steps:
        return None

    # 1. Identify & reconcile transient self-healing inventory flickers (within <= 2 steps)
    unexp_joker_by_step: dict[int, list[tuple[list[str], list[str]]]] = {
        s: list(rec.get("unexplained_joker", ())) for s, rec in enumerate(step_records)
    }
    unexp_cons_by_step: dict[int, list[tuple[list[str], list[str]]]] = {
        s: list(rec.get("unexplained_cons", ())) for s, rec in enumerate(step_records)
    }
    transient_glitch_steps: set[int] = set()

    def _reconcile_flickers(
        by_step: dict[int, list[tuple[list[str], list[str]]]],
        *,
        ignore_equal_count_swaps: bool = False,
    ) -> tuple[set[int], set[int]]:
        unresolved: set[int] = set()
        unresolved_losses: set[int] = set()
        reconciled_idxs: set[tuple[int, int]] = set()
        for s in range(1, n_steps):
            entries = by_step.get(s, [])
            for idx_a, (lost_a, gained_a) in enumerate(entries):
                if (s, idx_a) in reconciled_idxs:
                    continue
                if (
                    ignore_equal_count_swaps
                    and lost_a
                    and gained_a
                    and len(lost_a) == len(gained_a)
                ):
                    continue
                if lost_a and not gained_a:
                    matched = False
                    for s2 in range(s + 1, min(n_steps, s + 3)):
                        for idx_b, (lost_b, gained_b) in enumerate(by_step.get(s2, [])):
                            if (s2, idx_b) in reconciled_idxs:
                                continue
                            if not lost_b and Counter(gained_b) == Counter(lost_a):
                                reconciled_idxs.add((s, idx_a))
                                reconciled_idxs.add((s2, idx_b))
                                transient_glitch_steps.update(range(s, s2 + 1))
                                matched = True
                                break
                        if matched:
                            break
                    if not matched:
                        unresolved.add(s)
                        unresolved_losses.add(s)
                else:
                    unresolved.add(s)
                    if lost_a:
                        unresolved_losses.add(s)
        return unresolved, unresolved_losses

    unresolved_joker_steps, unresolved_joker_loss_steps = _reconcile_flickers(
        unexp_joker_by_step
    )
    unresolved_cons_steps, _ = _reconcile_flickers(
        unexp_cons_by_step, ignore_equal_count_swaps=True
    )

    # 2. Build active anomaly categories per step
    cats_by_step: list[set[str]] = [set() for _ in range(n_steps)]
    u_adds_by_step: list[int] = [0] * n_steps
    for s in range(1, n_steps):
        rec = step_records[s]
        u_adds = int(rec.get("unexpected_adds", 0))
        u_adds_by_step[s] = u_adds
        if rec.get("hand_over_15"):
            cats_by_step[s].add("hand_size_over_15")
        if u_adds > 0:
            cats_by_step[s].add("unexpected_deck_addition")
        if s in unresolved_joker_steps:
            cats_by_step[s].add("unexplained_joker")
        if s in unresolved_cons_steps:
            cats_by_step[s].add("unexplained_cons")
        if rec.get("location_mismatch"):
            cats_by_step[s].add("location_mismatch")
        if rec.get("unmatched_target"):
            cats_by_step[s].add("unmatched_button_target")
        if rec.get("empty_hand_cv"):
            cats_by_step[s].add("empty_hand_cv")
        if rec.get("missing_shop_cv"):
            cats_by_step[s].add("missing_shop_cv")
        if rec.get("cashout_no_round_won"):
            cats_by_step[s].add("cashout_without_round_won")
        if rec.get("large_ocr_override"):
            cats_by_step[s].add("large_ocr_override")

    # 3. Compute earliest permanent-tail quarantine watermark `t_perm`
    t_perm: int | None = None
    deck_perm_starts: set[int] = set()
    t_non_deck_perm: int | None = None

    def _update_t_perm(cand: int) -> None:
        nonlocal t_perm
        if t_perm is None or cand < t_perm:
            t_perm = cand

    def _record_deck_perm(cand: int) -> None:
        deck_perm_starts.add(cand)

    def _update_t_non_deck_perm(cand: int) -> None:
        nonlocal t_non_deck_perm
        if t_non_deck_perm is None or cand < t_non_deck_perm:
            t_non_deck_perm = cand

    # Rule C1a: Deck discovery burst in any 5-step sliding window (>= 3 cards or >= 2 steps)
    win_size = 5
    for start in range(1, n_steps):
        end = min(n_steps, start + win_size)
        add_steps = [k for k in range(start, end) if u_adds_by_step[k] > 0]
        if len(add_steps) >= 2 or sum(u_adds_by_step[k] for k in add_steps) >= 3:
            _record_deck_perm(add_steps[0])

    # Group step indices into contiguous round segments (preserves order even if ante/round repeats)
    round_groups: list[list[int]] = []
    cur_group: list[int] = []
    cur_rk: tuple[int, int] | None = None
    for s in range(1, n_steps):
        rk = step_records[s]["round_key"]
        if cur_rk is None or rk != cur_rk:
            if cur_group:
                round_groups.append(cur_group)
            cur_group = [s]
            cur_rk = rk
        else:
            cur_group.append(s)
    if cur_group:
        round_groups.append(cur_group)

    round_end_for_step: dict[int, int] = {}
    for grp in round_groups:
        grp_end = grp[-1]
        for k in grp:
            round_end_for_step[k] = grp_end
    full_deck_sync_steps = sorted(
        s for s, rec in enumerate(step_records) if rec.get("full_deck_synced")
    )

    for s in range(1, n_steps):
        rec = step_records[s]
        if rec.get("unobserved_deck_tarot"):
            _record_deck_perm(s)
        if rec.get("fd_deck_tarot"):
            next_r_start = round_end_for_step.get(s, s) + 1
            if next_r_start < n_steps and not any(
                s < ts <= next_r_start for ts in full_deck_sync_steps
            ):
                _record_deck_perm(next_r_start)

    # Rule C1b: Consecutive rounds both experiencing unexpected deck additions
    for idx in range(len(round_groups) - 1):
        g_prev = round_groups[idx]
        g_next = round_groups[idx + 1]
        prev_adds = [k for k in g_prev if u_adds_by_step[k] > 0]
        next_adds = [k for k in g_next if u_adds_by_step[k] > 0]
        if prev_adds and next_adds:
            _record_deck_perm(prev_adds[0])

    # Rule C2: hand_size_over_15 combined with deck mutation or hand/target failure in same round
    for grp in round_groups:
        h15_steps = [k for k in grp if "hand_size_over_15" in cats_by_step[k]]
        if h15_steps:
            if any(
                "empty_hand_cv" in cats_by_step[k]
                or "unmatched_button_target" in cats_by_step[k]
                for k in grp
            ):
                _update_t_non_deck_perm(h15_steps[0])
            elif any(u_adds_by_step[k] > 0 for k in grp):
                _record_deck_perm(h15_steps[0])

    # Rule C3: Multi-signal anomaly co-occurrence (>= 2 distinct categories) in 5-step window or same round
    _DECK_ONLY_CATS = frozenset(
        {
            "unexpected_deck_addition",
            "large_ocr_override",
            "unexplained_cons",
            "hand_size_over_15",
        }
    )
    for start in range(1, n_steps):
        end = min(n_steps, start + win_size)
        win_cats: set[str] = set()
        first_anomalous: int | None = None
        for k in range(start, end):
            if cats_by_step[k]:
                if first_anomalous is None:
                    first_anomalous = k
                win_cats.update(cats_by_step[k])
        if len(win_cats) >= 2 and first_anomalous is not None:
            if win_cats <= _DECK_ONLY_CATS:
                _record_deck_perm(first_anomalous)
            else:
                _update_t_non_deck_perm(first_anomalous)

    for grp in round_groups:
        grp_cats: set[str] = set()
        first_anomalous = None
        for k in grp:
            if cats_by_step[k]:
                if first_anomalous is None:
                    first_anomalous = k
                grp_cats.update(cats_by_step[k])
        if len(grp_cats) >= 2 and first_anomalous is not None:
            if grp_cats <= _DECK_ONLY_CATS:
                _record_deck_perm(first_anomalous)
            else:
                _update_t_non_deck_perm(first_anomalous)

    # Rule C4: Unresolved Joker loss or Joker drift co-occurring with downstream failure,
    # or >= 2 consecutive location mismatches
    step_to_round_group: dict[int, list[int]] = {}
    for grp in round_groups:
        for k in grp:
            step_to_round_group[k] = grp
    _DOWNSTREAM_FAILURE_CATS = frozenset(
        {
            "location_mismatch",
            "empty_hand_cv",
            "missing_shop_cv",
            "unmatched_button_target",
            "unexpected_deck_addition",
            "hand_size_over_15",
            "cashout_without_round_won",
        }
    )
    for s in sorted(unresolved_joker_steps):
        has_downstream_failure = any(
            cats_by_step[k] & _DOWNSTREAM_FAILURE_CATS
            for k in range(s, min(n_steps, s + 5))
        ) or any(
            cats_by_step[k] & _DOWNSTREAM_FAILURE_CATS
            for k in step_to_round_group.get(s, ())
        )
        if has_downstream_failure:
            _update_t_non_deck_perm(s)
    for s in range(1, n_steps - 1):
        if (
            "location_mismatch" in cats_by_step[s]
            and "location_mismatch" in cats_by_step[s + 1]
        ):
            _update_t_non_deck_perm(s)

    # De-escalate pure deck-drift episodes per segment when a subsequent Full Deck CV sync stabilizes
    bounded_deck_ranges: list[tuple[int, int]] = []
    if deck_perm_starts:
        covered_until = -1
        for d_start in sorted(deck_perm_starts):
            if d_start < covered_until:
                continue
            recovered_sync = None
            for t_sync in full_deck_sync_steps:
                if t_sync <= d_start:
                    continue
                min_after_h15 = max(
                    (
                        round_end_for_step.get(k, k)
                        for k in range(d_start, t_sync + 1)
                        if "hand_size_over_15" in cats_by_step[k]
                    ),
                    default=d_start,
                )
                if t_sync <= min_after_h15:
                    continue
                if not any(
                    u_adds_by_step[k] > 0
                    for k in range(t_sync + 1, min(n_steps, t_sync + 3))
                ):
                    recovered_sync = t_sync
                    break
            if recovered_sync is not None:
                bounded_deck_ranges.append((d_start, recovered_sync))
                covered_until = recovered_sync
                if diag is not None:
                    diag["full_deck_quarantine_recoveries"] = (
                        diag.get("full_deck_quarantine_recoveries", 0) + 1
                    )
            else:
                _update_t_perm(d_start)
                break

    if t_non_deck_perm is not None:
        _update_t_perm(t_non_deck_perm)

    # 4. Flag Scope A (transient), Scope B (round-scoped), and Scope C (permanent tail)
    def _flag_step(s: int, reason: str) -> bool:
        act = trajectory.deltas[s].action
        if act is None:
            return False
        already = bool(act.metadata.get("corrupted"))
        _flag_corrupted_action(act, reason, diag=diag)
        return not already

    # Scope A
    for s in sorted(transient_glitch_steps):
        _flag_step(s, "transient_inventory_glitch")
    for s in sorted(unresolved_joker_steps):
        _flag_step(s, "unexplained_joker_drift")
    for s in range(1, n_steps):
        if "large_ocr_override" in cats_by_step[s]:
            _flag_step(s, "large_ocr_override")
        if "location_mismatch" in cats_by_step[s]:
            _flag_step(s, "location_mismatch")
        if u_adds_by_step[s] >= 2:
            _flag_step(s, "unexpected_deck_addition")
    for b_start, b_end in bounded_deck_ranges:
        for s in range(b_start, b_end):
            _flag_step(s, "deck_drift_before_full_deck_resync")

    # Scope B (round-scoped through end of current round)
    round_scoped_reasons = (
        ("hand_size_over_15", "hand_size_over_15"),
        ("empty_hand_cv", "empty_hand_cv"),
        ("missing_shop_cv", "missing_shop_cv"),
        ("cashout_without_round_won", "cashout_without_round_won"),
    )
    for s in range(1, n_steps):
        end_r = round_end_for_step.get(s, s)
        for cat, reason_label in round_scoped_reasons:
            if cat in cats_by_step[s]:
                for k in range(s, end_r + 1):
                    _flag_step(k, reason_label if k == s else f"round_{reason_label}")

    # Scope C (permanent tail watermark)
    downstream_added = 0
    if t_perm is not None:
        for s in range(t_perm, n_steps):
            if _flag_step(s, "downstream_cascade_quarantine"):
                downstream_added += 1

    if diag is not None:
        diag["quarantine_watermark"] = t_perm
        diag["quarantined_downstream_count"] = downstream_added
    return t_perm


def _resolve_hand_with_deck_constraints(
    state: GameState,
    raw_cards: Sequence[Any],
    obs_hand: list[HandCard],
    *,
    diag: dict[str, Any] | None = None,
) -> list[HandCard]:
    """Refines ambiguous or conflicting face-up card classifications against `state`'s known deck.

    Leaves high-confidence non-conflicting cards (including genuinely new cards from a missed
    Standard Pack take) untouched via strict per-head probability plausibility gates.
    """
    if (
        not obs_hand
        or len(obs_hand) != len(raw_cards)
        or len(obs_hand) > 15
        or not state.full_deck
    ):
        return obs_hand

    faceup_indices = [i for i, hc in enumerate(obs_hand) if not hc.facing_back]
    if not faceup_indices:
        return obs_hand

    pool: list[PlayingCard] = []
    seen_keys: set[tuple[int, int]] = set()
    candidates_src: list[PlayingCard] = [
        hc.card for hc in state.hand.cards if not hc.facing_back
    ]
    candidates_src.extend(
        hc._real_card
        for hc in state.hand.cards
        if hc.facing_back and hc._real_card is not None
    )
    candidates_src.extend(state.remaining_deck)
    for dc in candidates_src:
        if getattr(dc, "destroyed", False):
            continue
        k = (1, dc.id) if dc.id != 0 else (0, id(dc))
        if k not in seen_keys:
            seen_keys.add(k)
            pool.append(dc)

    if not pool:
        return obs_hand

    obs_sig_counts = Counter(_card_attr_sig(obs_hand[i].card) for i in faceup_indices)
    pool_sig_counts = Counter(_card_attr_sig(c) for c in pool)

    m = len(faceup_indices)
    p = len(pool)
    inf_cost = 1e6
    new_card_penalty = 4.2
    cost_matrix = [[inf_cost] * (p + m) for _ in range(m)]

    for r_idx, obs_idx in enumerate(faceup_indices):
        raw = raw_cards[obs_idx]
        obs_c = obs_hand[obs_idx].card
        obs_lbl = _playing_card_cls_labels(obs_c)
        top_probs = getattr(raw, "top_probs", None) or {}
        confs = getattr(raw, "confidences", None) or {}
        top1_nll = sum(
            -math.log(max(float(confs.get(h, 0.99)), 1e-6)) for h in _CLS_HEADS
        )
        cost_matrix[r_idx][p + r_idx] = top1_nll + new_card_penalty
        has_conflict = (
            obs_sig_counts[_card_attr_sig(obs_c)]
            > pool_sig_counts[_card_attr_sig(obs_c)]
        )

        for c_idx, cand in enumerate(pool):
            if _cards_match(cand, obs_c):
                cost_matrix[r_idx][c_idx] = top1_nll
                continue
            if not top_probs:
                continue

            cand_lbl = _playing_card_cls_labels(cand)
            diffs = [h for h in _CLS_HEADS if cand_lbl[h] != obs_lbl[h]]
            if len(diffs) != 1:
                continue

            p_r = _raw_head_prob(raw, "rank", cand_lbl["rank"], obs_lbl["rank"])
            p_s = _raw_head_prob(raw, "suit", cand_lbl["suit"], obs_lbl["suit"])
            p_enh = _raw_head_prob(
                raw, "enhancement", cand_lbl["enhancement"], obs_lbl["enhancement"]
            )
            p_ed = _raw_head_prob(
                raw, "edition", cand_lbl["edition"], obs_lbl["edition"]
            )
            p_seal = _raw_head_prob(raw, "seal", cand_lbl["seal"], obs_lbl["seal"])
            min_enh_prob = 0.02 if (has_conflict and diffs == ["enhancement"]) else 0.10
            min_ed_prob = 0.02 if (has_conflict and diffs == ["edition"]) else 0.10
            if not (
                p_r >= 0.03
                and p_s >= 0.10
                and p_enh >= min_enh_prob
                and p_ed >= min_ed_prob
                and p_seal >= 0.10
            ):
                continue

            diff_head = diffs[0]
            top1_diff_conf = float(
                confs.get(
                    diff_head,
                    (top_probs.get(diff_head) or {}).get(obs_lbl[diff_head], 1.0),
                )
            )
            if has_conflict or top1_diff_conf < 0.985:
                cost_matrix[r_idx][c_idx] = sum(
                    -math.log(max(val, 1e-6)) for val in (p_r, p_s, p_enh, p_ed, p_seal)
                )

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    for r_idx, c_idx in zip(row_ind, col_ind):
        if c_idx >= p or cost_matrix[r_idx][c_idx] >= inf_cost * 0.5:
            continue
        obs_idx = faceup_indices[r_idx]
        cand = pool[c_idx]
        if not _cards_match(cand, obs_hand[obs_idx].card):
            updated = copy.deepcopy(cand)
            updated.debuffed = bool(obs_hand[obs_idx].card.debuffed)
            obs_hand[obs_idx].card = updated
            if diag is not None:
                diag["deck_constrained_remaps"] += 1

    return obs_hand


def _collect_play_area_probes(
    scheduled_actions: Sequence[ScheduledAction],
    hand_res_by_frame: dict[int, Any],
) -> dict[int, list[int]]:
    """Collects post-click `play_area` probe frame indices for PLAY_HAND actions with face-down cards."""
    probes_by_sa: dict[int, list[int]] = {}
    n_actions = len(scheduled_actions)
    for sa_idx, sa in enumerate(scheduled_actions):
        if sa.action_type != ActionType.PLAY_HAND:
            continue
        hand_out = hand_res_by_frame.get(sa.target_frame)
        if hand_out is None:
            continue
        cards = getattr(hand_out, "cards", None) or []
        if not any(
            getattr(c, "selected", False) and getattr(c, "face_down", False)
            for c in cards
        ):
            continue
        next_bound = (
            scheduled_actions[sa_idx + 1].frame_id - 4
            if sa_idx + 1 < n_actions
            else sa.frame_id + 120
        )
        probes = [
            sa.frame_id + off for off in (40, 65, 90) if sa.frame_id + off < next_bound
        ]
        if not probes and (next_bound - sa.frame_id) >= 12:
            probes = [sa.frame_id + (next_bound - sa.frame_id) // 2]
        if probes:
            probes_by_sa[sa_idx] = probes
    return probes_by_sa


def _run_play_area_cv_for_face_down_plays(
    probes_by_sa_idx: dict[int, list[int]],
    *,
    video_path: str | Path | None,
    preloaded_frames: dict[int, Any] | None,
    fps: float,
    conf_threshold_hand: float,
    batch_size: int,
    device: str | None,
) -> dict[int, list[Any]]:
    """Runs on-demand `ROIS["play_area"]` hand recognition across probe frames for face-down plays."""
    if not probes_by_sa_idx:
        return {}

    unique_frames = sorted({fid for fids in probes_by_sa_idx.values() for fid in fids})
    if preloaded_frames is not None:
        probe_frames_by_idx = {
            fid: preloaded_frames[fid]
            for fid in unique_frames
            if fid in preloaded_frames
        }
    elif video_path is not None and unique_frames:
        from classifiers.common.video_utils import extract_video_frames

        probe_frames_by_idx = extract_video_frames(video_path, unique_frames)
    else:
        probe_frames_by_idx = {}

    if not probe_frames_by_idx:
        return {}

    from classifiers.inference import batch_recognize_hands

    ordered_ids = sorted(probe_frames_by_idx.keys())
    frame_inputs = [(probe_frames_by_idx[fid], fid, fid / fps) for fid in ordered_ids]
    outs = batch_recognize_hands(
        frame_inputs,
        conf_threshold=conf_threshold_hand,
        batch_size=batch_size,
        device=device,
        roi_key="play_area",
    )
    out_by_frame = {out.frame_index: out for out in outs}
    return {
        sa_idx: [out_by_frame[fid] for fid in fids if fid in out_by_frame]
        for sa_idx, fids in probes_by_sa_idx.items()
    }


def _select_consensus_play_area_cards(
    probe_outputs: Sequence[Any], expected_count: int
) -> list[Any] | None:
    """Selects consensus left-to-right revealed cards from multiple `play_area` probe frames."""
    if expected_count <= 0:
        return None
    valid = [
        out
        for out in probe_outputs
        if len(getattr(out, "cards", None) or ()) == expected_count
        and all(not getattr(c, "face_down", False) for c in out.cards)
    ]
    if not valid:
        return None

    chosen: list[Any] = []
    for k in range(expected_count):
        scores: dict[tuple[Any, ...], float] = {}
        best_sample: dict[tuple[Any, ...], tuple[float, Any]] = {}
        for out in valid:
            c = out.cards[k]
            confs = getattr(c, "confidences", None) or {}
            w = float(confs.get("rank", 1.0)) * float(confs.get("suit", 1.0))
            sig = (c.rank, c.suit, c.enhancement, c.edition, c.seal)
            scores[sig] = scores.get(sig, 0.0) + w
            prev_best = best_sample.get(sig)
            if prev_best is None or w > prev_best[0]:
                best_sample[sig] = (w, c)
        win_sig = max(scores, key=lambda s: scores[s])
        chosen.append(best_sample[win_sig][1])
    return chosen


def _resolve_played_face_down_cards(
    state: GameState,
    action: Action,
    revealed_raw: Sequence[Any],
    *,
    diag: dict[str, Any] | None = None,
) -> bool:
    """Resolves face-down selected cards on PLAY_HAND using revealed `play_area` CV observations."""
    selected_hcs = [hc for hc in state.hand.cards if hc.selected]
    if not selected_hcs or not any(hc.facing_back for hc in selected_hcs):
        return False
    if len(revealed_raw) != len(selected_hcs):
        return False

    revealed_hcs = [_convert_card_result_to_hand_card(c) for c in revealed_raw]
    _resolve_hand_with_deck_constraints(state, revealed_raw, revealed_hcs, diag=diag)

    used_rev: set[int] = set()
    for hc in selected_hcs:
        if hc.facing_back:
            continue
        hit = next(
            (
                j
                for j, rev in enumerate(revealed_hcs)
                if j not in used_rev and _cards_match(hc.card, rev.card)
            ),
            None,
        )
        if hit is None:
            hit = next(
                (
                    j
                    for j, rev in enumerate(revealed_hcs)
                    if j not in used_rev
                    and hc.card.rank == rev.card.rank
                    and hc.card.suit == rev.card.suit
                ),
                None,
            )
        if hit is not None:
            used_rev.add(hit)

    remaining_fd_hcs = [hc for hc in selected_hcs if hc.facing_back]
    remaining_rev_hcs = [rev for j, rev in enumerate(revealed_hcs) if j not in used_rev]
    if len(remaining_fd_hcs) != len(remaining_rev_hcs):
        return False

    claimed_ids: set[int] = {
        hc.card.id for hc in state.hand.cards if not hc.facing_back and hc.card.id != 0
    }
    claimed_ids.update(
        hc._real_card.id
        for hc in state.hand.cards
        if hc.facing_back and hc._real_card is not None and hc._real_card.id != 0
    )
    next_deck_id = max((c.id for c in state.full_deck), default=0) + 1

    def _find_matching_deck_card(
        target: PlayingCard, cands: Sequence[PlayingCard]
    ) -> PlayingCard | None:
        for dc in cands:
            if (
                not getattr(dc, "destroyed", False)
                and dc.id not in claimed_ids
                and _cards_match(dc, target)
            ):
                return dc
        for dc in cands:
            if (
                not getattr(dc, "destroyed", False)
                and dc.id not in claimed_ids
                and dc.rank == target.rank
                and dc.suit == target.suit
                and dc.enhancement == target.enhancement
            ):
                return dc
        return None

    for hc, rev_hc in zip(remaining_fd_hcs, remaining_rev_hcs):
        obs_card = rev_hc.card
        resolved: PlayingCard | None = None
        if hc.bounds and hc.bounds.candidate_cards:
            resolved = _find_matching_deck_card(obs_card, hc.bounds.candidate_cards)
        if resolved is None:
            resolved = _find_matching_deck_card(obs_card, state.remaining_deck)
        if resolved is None:
            resolved = copy.deepcopy(obs_card)
            resolved.id = next_deck_id
            next_deck_id += 1
            state.full_deck.append(resolved)
        resolved.debuffed = bool(obs_card.debuffed)
        claimed_ids.add(resolved.id)
        hc.resolve_face_down(resolved, lift_facing_back=True)
        state.sync_card_to_deck(hc.card, rebind_hand_card=hc)

    action.selected_cards = [
        copy.deepcopy(hc.card) for hc in state.hand.cards if hc.selected
    ]
    action.metadata["revealed_played_cards"] = [
        (
            "Stone"
            if c.enhancement == Enhancement.STONE
            else f"{c.rank.value}{c.suit.value[0].upper() if c.suit != Suit.NONE else ''}"
        )
        for c in action.selected_cards
    ]
    state.update_face_down_candidates()
    state.reconcile_face_down_adjacencies()
    if diag is not None:
        diag["face_down_plays_resolved"] += 1
    return True


def _sync_on_screen_to_do_list(
    state: GameState,
    cards_list: Sequence[Any],
    todo_unbought: list[PokerHand],
) -> None:
    """Ensures every visible To Do List in shop/pack has a single deterministic `'to_do'` roll."""
    todo_jokers = [
        c for c in cards_list if isinstance(c, Joker) and c.name == JokerName.TO_DO_LIST
    ]
    if not todo_jokers:
        return
    if getattr(state, "rng", None) is not None:
        visible = _get_visible_poker_hands(state)
        while len(todo_unbought) < len(todo_jokers):
            rolled = get_to_do_list_hand(
                state.rng,
                current_hand=None,
                visible_hands=visible,
                is_initial_spawn=True,
            )
            todo_unbought.append(rolled)
    for idx, joker in enumerate(todo_jokers):
        if idx < len(todo_unbought):
            joker.state["target_hand"] = todo_unbought[idx]


def _find_todo_relative_index(
    cards_list: Sequence[Any], target_idx: int | None
) -> int | None:
    """Returns which k-th To Do List in `cards_list` sits at `target_idx`, or None."""
    if target_idx is None or target_idx < 0 or target_idx >= len(cards_list):
        return None
    card = cards_list[target_idx]
    if not (isinstance(card, Joker) and card.name == JokerName.TO_DO_LIST):
        return None
    return sum(
        1
        for i, c in enumerate(cards_list[:target_idx])
        if isinstance(c, Joker) and c.name == JokerName.TO_DO_LIST
    )


def _read_valid_hud_ante(hud_out: Any) -> int | None:
    """Returns parsed integer HUD ante in `[1, 39]`, or None if missing/invalid."""
    raw = _get_hud_val(hud_out, "hud_ante")
    if raw is None:
        return None
    try:
        ante_int = int(raw)
    except (TypeError, ValueError):
        return None
    return ante_int if 1 <= ante_int <= 39 else None


def _sim_is_in_ante_9(state: GameState) -> bool:
    """Returns True once the simulated run has reached Ante 9 / Endless Mode."""
    return state.ante >= 9 or _effective_ante(state) >= 9


def _snapshot_diag_state(diag: dict[str, Any] | None) -> dict[str, Any] | None:
    """Snapshots list lengths and counters in `diag` for clean Ante 9 rollback."""
    if diag is None:
        return None
    list_lens: dict[str, int] = {}
    scalars: dict[str, int] = {}
    subdicts: dict[str, dict[Any, Any]] = {}
    for k, v in diag.items():
        if isinstance(v, list):
            list_lens[k] = len(v)
        elif isinstance(v, int) and not isinstance(v, bool):
            scalars[k] = v
        elif k in ("ocr_none_by_field", "ocr_overrides_by_field") and isinstance(
            v, dict
        ):
            subdicts[k] = dict(v)
    return {"list_lens": list_lens, "scalars": scalars, "subdicts": subdicts}


def _restore_diag_snapshot(
    diag: dict[str, Any] | None, snap: dict[str, Any] | None
) -> None:
    """Restores `diag` lists and counters to a prior snapshot."""
    if diag is None or snap is None:
        return
    for k, n in snap["list_lens"].items():
        if isinstance(diag.get(k), list):
            del diag[k][n:]
    for k, val in snap["scalars"].items():
        diag[k] = val
    for k, d_copy in snap["subdicts"].items():
        diag[k] = dict(d_copy)


def _halt_and_truncate_to_ante_8(
    trajectory: RunTrajectory,
    step_records: list[dict[str, Any]],
    *,
    ante_8_end_step_idx: int,
    ante_8_diag_snap: dict[str, Any] | None,
    diag: dict[str, Any] | None,
    diag_cursor: dict[str, int],
    tracked_diag_keys: Sequence[str],
    trigger_sa: ScheduledAction,
    trigger_sa_idx: int,
    total_scheduled: int,
) -> GameState:
    """Truncates recorded steps/diag back to end of Ante 8 and returns that GameState."""
    keep_len = ante_8_end_step_idx + 1
    rolled_back = max(0, len(trajectory.deltas) - keep_len)
    trajectory.deltas = trajectory.deltas[:keep_len]
    trajectory.keyframes = {
        k: kf for k, kf in trajectory.keyframes.items() if k < keep_len
    }
    del step_records[keep_len:]
    _restore_diag_snapshot(diag, ante_8_diag_snap)
    if diag is not None:
        for k in tracked_diag_keys:
            diag_cursor[k] = len(diag[k])
        skipped = max(0, total_scheduled - trigger_sa_idx)
        diag["stopped_at_ante_9"] = {
            "trigger_frame_id": trigger_sa.frame_id,
            "trigger_action": trigger_sa.action_type.value,
            "ante_8_end_step_idx": ante_8_end_step_idx,
            "kept_steps": keep_len,
            "rolled_back_steps": rolled_back,
            "skipped_actions": skipped,
            "dropped_steps": rolled_back + skipped,
        }
    return trajectory.get_state(ante_8_end_step_idx)


_SHOP_RESCAN_TEMPLATE_NAMES: tuple[str, ...] = (
    "Buy Card",
    "Buy and Use",
    "Open Pack",
    "Redeem Voucher",
    "Reroll Shop",
    "Select Card",
    "Use In Pack",
    "Skip Pack",
    "Sell Card",
    "Use Consumable",
)


_SHOP_RESCAN_BLIND_TRANSITION_TEMPLATES: tuple[str, ...] = (
    "Next Round",
    "Select Blind",
    "Skip Blind",
    "Play Hand",
    "Discard",
    "Cash Out",
)


def _rescan_unexpected_shop_windows(
    d: dict[str, Any],
    video_path: str | Path,
    unexpected_transitions: Sequence[dict[str, Any]],
    *,
    max_frames_per_window: int = 2000,
    max_total_frames: int = 4000,
) -> tuple[dict[str, Any] | None, dict[str, int]]:
    """Re-scans missed-click shop intervals at stride 1 with standard template thresholds."""
    raw_intervals: list[tuple[int, int]] = []
    include_blind_transitions = False
    for item in unexpected_transitions:
        if any(
            str(r).startswith("vouchers_gained=") or str(r).startswith("packs_gained=")
            for r in (item.get("reasons") or ())
        ):
            include_blind_transitions = True
        pf = item.get("prev_frame")
        cf = item.get("frame")
        if pf is None or cf is None or int(cf) <= int(pf):
            continue
        lo = max(0, int(pf) - 2)
        hi = int(cf) + 2
        if hi - lo + 1 > max_frames_per_window:
            lo = hi - max_frames_per_window + 1
        raw_intervals.append((lo, hi))

    if not raw_intervals:
        return None, {"windows": 0, "frames_scanned": 0, "recovered_frames": 0}

    raw_intervals.sort()
    merged_intervals: list[list[int]] = []
    for lo, hi in raw_intervals:
        if merged_intervals and lo <= merged_intervals[-1][1] + 1:
            merged_intervals[-1][1] = max(merged_intervals[-1][1], hi)
        else:
            merged_intervals.append([lo, hi])

    effective_max_total = (
        max(max_total_frames, 6000) if include_blind_transitions else max_total_frames
    )

    def _select_window_frames(lo: int, hi: int) -> list[int]:
        span = hi - lo + 1
        if span <= max_frames_per_window:
            return list(range(lo, hi + 1))
        if not include_blind_transitions:
            return list(range(hi - max_frames_per_window + 1, hi + 1))
        edge = min(max_frames_per_window // 2, 1200)
        prefix_end = min(hi, lo + edge - 1)
        suffix_start = max(prefix_end + 1, hi - edge + 1)
        selected = list(range(lo, prefix_end + 1))
        if suffix_start > prefix_end + 1:
            mid_span = suffix_start - (prefix_end + 1)
            mid_budget = max(
                0,
                max_frames_per_window - len(selected) - (hi - suffix_start + 1),
            )
            stride = 2 if mid_budget <= 0 else max(2, math.ceil(mid_span / mid_budget))
            selected.extend(range(prefix_end + 1, suffix_start, stride))
        selected.extend(range(suffix_start, hi + 1))
        return selected

    existing_frames = d.get("frames") or {}
    candidate_frames: list[int] = []
    seen_candidates: set[int] = set()
    for lo, hi in merged_intervals:
        for f in _select_window_frames(lo, hi):
            if f in seen_candidates:
                continue
            if (
                str(f) in existing_frames
                or str(f - 1) in existing_frames
                or str(f + 1) in existing_frames
            ):
                continue
            seen_candidates.add(f)
            candidate_frames.append(f)
            if len(candidate_frames) >= effective_max_total:
                break
        if len(candidate_frames) >= effective_max_total:
            break

    stats = {
        "windows": len(merged_intervals),
        "frames_scanned": len(candidate_frames),
        "recovered_frames": 0,
    }
    if not candidate_frames:
        return None, stats

    import concurrent.futures
    import cv2
    from types import SimpleNamespace
    from classifiers.common.video_utils import extract_video_frames
    from eval_template import eval as eval_tmpl
    from template import template_loader

    imgs = extract_video_frames(video_path, candidate_frames)
    if not imgs:
        return None, stats

    first_img = next(iter(imgs.values()))
    res_h, res_w = first_img.shape[:2]
    cfg = get_config()
    all_tmpls = template_loader(
        cfg.template_json_path,
        cfg.template_img_dir,
        SimpleNamespace(width=res_w, height=res_h),
    )
    allowed_names = set(_SHOP_RESCAN_TEMPLATE_NAMES)
    if include_blind_transitions:
        allowed_names.update(_SHOP_RESCAN_BLIND_TRANSITION_TEMPLATES)
    active_tmpls = [t for t in all_tmpls if t.name in allowed_names]
    if not active_tmpls:
        return None, stats

    def _scan_one(f_idx: int) -> tuple[int, str | None, list[int] | None]:
        img = imgs.get(f_idx)
        if img is None:
            return f_idx, None, None
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray_half = (
            cv2.resize(
                img_gray,
                (img_gray.shape[1] // 2, img_gray.shape[0] // 2),
                interpolation=cv2.INTER_NEAREST,
            )
            if cfg.coarse_to_fine
            else None
        )
        best_conf = -1.0
        best_name: str | None = None
        best_box: list[int] | None = None
        for t in active_tmpls:
            res = eval_tmpl(t, img, img_gray, cfg, img_gray_half=img_gray_half)
            if res.found and res.clicked:
                cand_conf = (
                    float(res.match.confidence) if res.match is not None else 1.0
                )
                if cand_conf > best_conf:
                    best_conf = cand_conf
                    best_name = res.template_name
                    best_box = list(res.box) if res.box is not None else None
        return f_idx, best_name, best_box

    workers = max(1, int(getattr(cfg, "max_workers", 4)))
    if workers > 1 and len(candidate_frames) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            scan_results = list(pool.map(_scan_one, candidate_frames))
    else:
        scan_results = [_scan_one(f_idx) for f_idx in candidate_frames]

    merged_key = dict(d.get("key") or {})
    merged_frames = dict(existing_frames)
    recovered_count = 0
    for f_idx, best_name, best_box in scan_results:
        if best_name is not None:
            if best_name not in merged_key:
                merged_key[best_name] = max(merged_key.values(), default=-1) + 1
            merged_frames[str(f_idx)] = {
                "action": merged_key[best_name],
                "box": best_box,
            }
            recovered_count += 1

    stats["recovered_frames"] = recovered_count
    if recovered_count == 0:
        return None, stats
    return {"key": merged_key, "frames": merged_frames}, stats


def _build_run_trajectory_with_plan(
    d: dict[str, Any],
    video_path: str | Path | None = None,
    *,
    preloaded_frames: dict[int, Any] | None = None,
    seed: str = "",
    deck: Deck = Deck.RED_DECK,
    stake: Stake = Stake.WHITE,
    fps: float | None = None,
    conf_threshold_hand: float = 0.50,
    conf_threshold_shop: float = 0.15,
    conf_threshold_ocr: float = 0.90,
    batch_size: int = 32,
    device: str | None = None,
    deduplicate: bool = True,
    expressive: bool = False,
    rescan_shop_transitions: bool = True,
    _is_rescan_pass: bool = False,
) -> tuple[GameState, RunTrajectory, dict[str, Any]]:
    from classifiers.common.video_utils import extract_video_frames

    diag = _init_diagnostics()
    resolved_fps = _resolve_video_fps(video_path, fps)
    dedup_frames = remove_duplicates(d["frames"]) if deduplicate else d["frames"]
    cleaned_input = {
        "key": d["key"],
        "frames": filter_premature_next_rounds(dedup_frames, d["key"], diag=diag),
    }
    diag["end_round_segments_checked"] = len(
        find_missing_end_round_segments(cleaned_input)
    )
    pre_frames_count = len(cleaned_input["frames"])

    t0 = time.perf_counter()
    cleaned_input = inject_end_round_actions(
        cleaned_input,
        video_path=video_path,
        preloaded_frames=preloaded_frames,
        fps=resolved_fps,
        batch_size=max(1, batch_size // 4),
        device=device,
        expressive=expressive,
    )
    if diag is not None:
        diag["timings"]["end_round"] = time.perf_counter() - t0
        diag["end_rounds_injected"] = max(
            0, len(cleaned_input["frames"]) - pre_frames_count
        )

    cleaned_input["frames"] = filter_false_early_cash_outs(
        cleaned_input["frames"], cleaned_input["key"], diag=diag
    )
    plan = plan_action_pipeline(cleaned_input)
    full_deck_observations = _extract_full_deck_observations(
        d.get("frames") or {},
        d.get("key") or {},
    )
    plan["full_deck_observations"] = full_deck_observations
    if diag is not None:
        diag["full_deck_views_detected"] = len(full_deck_observations)
    scheduled_actions: list[ScheduledAction] = plan["scheduled_actions"]

    needed_frames = sorted(
        set(plan["playing_card_cv_queue"])
        | set(plan["joker_cv_queue"])
        | set(plan["shop_cv_queue"])
        | set(plan["pack_cv_queue"])
        | set(plan["hud_ocr_queue"])
    )
    t1 = time.perf_counter()
    if preloaded_frames is not None:
        frames_by_idx = {
            idx: preloaded_frames[idx]
            for idx in needed_frames
            if idx in preloaded_frames
        }
    elif video_path is not None and needed_frames:
        if expressive:
            with tqdm(
                total=len(needed_frames), desc="Extracting CV frames", unit="frame"
            ) as ex_bar:
                frames_by_idx = extract_video_frames(
                    video_path,
                    needed_frames,
                    progress_callback=ex_bar.update,
                )
        else:
            frames_by_idx = extract_video_frames(video_path, needed_frames)
    else:
        frames_by_idx = {}
    if diag is not None:
        diag["timings"]["extract_frames"] = time.perf_counter() - t1
        diag["missing_frames"] = [
            fid for fid in needed_frames if fid not in frames_by_idx
        ]

    t2 = time.perf_counter()
    shop_res_by_frame, hand_res_by_frame, hud_res_by_frame = _run_batched_cv(
        scheduled_actions,
        plan,
        frames_by_idx,
        fps=resolved_fps,
        conf_threshold_hand=conf_threshold_hand,
        conf_threshold_shop=conf_threshold_shop,
        conf_threshold_ocr=conf_threshold_ocr,
        batch_size=batch_size,
        device=device,
        expressive=expressive,
    )
    play_area_probes = _collect_play_area_probes(scheduled_actions, hand_res_by_frame)
    play_area_res_by_sa = _run_play_area_cv_for_face_down_plays(
        play_area_probes,
        video_path=video_path,
        preloaded_frames=preloaded_frames,
        fps=resolved_fps,
        conf_threshold_hand=conf_threshold_hand,
        batch_size=batch_size,
        device=device,
    )
    if diag is not None:
        diag["timings"]["batched_cv_ocr"] = time.perf_counter() - t2

    state = GameState(
        seed=seed,
        deck=deck,
        stake=stake,
        ante=1,
        round=1,
        current_blind=BlindName.SMALL,
        location=Location.SELECT,
        apply_deck_stake_on_init=True,
    )
    trajectory = RunTrajectory()
    step_records: list[dict[str, Any]] = []
    _tracked_diag_keys = (
        "hand_size_over_15_actions",
        "unexpected_deck_additions",
        "unexplained_joker_changes",
        "unexplained_consumable_changes",
        "unmatched_button_targets",
        "location_mismatches",
        "cashout_without_round_won",
        "empty_hand_cv_actions",
        "missing_shop_cv_actions",
        "large_ocr_overrides",
    )
    _diag_cursor: dict[str, int] = {k: len(diag[k]) for k in _tracked_diag_keys}
    ante_8_end_step_idx = 0
    ante_8_diag_snap = _snapshot_diag_state(diag)
    ocr_saw_ante_9_in_sim9 = False

    def _record_tracked_step(
        st: GameState,
        act: Action | None,
        *,
        unexpected_adds: int = 0,
        full_deck_synced: bool = False,
        unobserved_deck_tarot: bool = False,
        fd_deck_tarot: bool = False,
    ) -> None:
        nonlocal ante_8_end_step_idx, ante_8_diag_snap, ocr_saw_ante_9_in_sim9
        trajectory.record_step(st, act)
        new_slices = {k: diag[k][_diag_cursor[k] :] for k in _tracked_diag_keys}
        for k in _tracked_diag_keys:
            _diag_cursor[k] = len(diag[k])
        step_records.append(
            {
                "step_idx": len(trajectory.deltas) - 1,
                "round_key": (st.ante, st.round),
                "location": st.location,
                "round_won": bool(st.round_won),
                "hand_over_15": bool(new_slices["hand_size_over_15_actions"]),
                "unexpected_adds": int(unexpected_adds),
                "full_deck_synced": bool(full_deck_synced),
                "unobserved_deck_tarot": bool(unobserved_deck_tarot),
                "fd_deck_tarot": bool(fd_deck_tarot),
                "unexplained_joker": [
                    (lost, gained)
                    for _, _, lost, gained in new_slices["unexplained_joker_changes"]
                ],
                "unexplained_cons": [
                    (lost, gained)
                    for _, _, lost, gained in new_slices[
                        "unexplained_consumable_changes"
                    ]
                ],
                "unmatched_target": bool(new_slices["unmatched_button_targets"]),
                "location_mismatch": bool(new_slices["location_mismatches"]),
                "cashout_no_round_won": bool(new_slices["cashout_without_round_won"]),
                "empty_hand_cv": bool(new_slices["empty_hand_cv_actions"]),
                "missing_shop_cv": bool(new_slices["missing_shop_cv_actions"]),
                "large_ocr_override": bool(new_slices["large_ocr_overrides"]),
            }
        )
        if not _sim_is_in_ante_9(st):
            ante_8_end_step_idx = len(trajectory.deltas) - 1
            ante_8_diag_snap = _snapshot_diag_state(diag)
            ocr_saw_ante_9_in_sim9 = False
        cur_rec = step_records[-1]
        cur_has_large_ocr = bool(cur_rec["large_ocr_override"])
        cur_has_non_swap_cons = any(
            not (lost_c and gained_c and len(lost_c) == len(gained_c))
            for lost_c, gained_c in cur_rec["unexplained_cons"]
        )
        if cur_has_large_ocr or cur_has_non_swap_cons:
            n_recs = len(step_records)
            cur_rk = cur_rec["round_key"]
            recent_indices = set(range(max(1, n_recs - 5), n_recs))
            k = n_recs - 1
            while k >= 1 and step_records[k]["round_key"] == cur_rk:
                recent_indices.add(k)
                k -= 1
            seen_ocr = any(
                step_records[i]["large_ocr_override"] for i in recent_indices
            )
            seen_cons = any(
                any(
                    not (lc and gc and len(lc) == len(gc))
                    for lc, gc in step_records[i]["unexplained_cons"]
                )
                for i in recent_indices
            )
            if seen_ocr and seen_cons:
                _mark_deck_suspect("ocr_and_consumable_cooccurrence")

    def _maybe_halt_ante_9(
        cur_sa_idx: int, cur_sa: ScheduledAction, hud_ante_val: int | None
    ) -> bool:
        nonlocal state, ocr_saw_ante_9_in_sim9
        if not _sim_is_in_ante_9(state):
            return False
        if hud_ante_val is not None and hud_ante_val >= 9:
            ocr_saw_ante_9_in_sim9 = True
        if not ocr_saw_ante_9_in_sim9:
            return False
        if expressive and hasattr(action_iter, "close"):
            action_iter.close()
        state = _halt_and_truncate_to_ante_8(
            trajectory,
            step_records,
            ante_8_end_step_idx=ante_8_end_step_idx,
            ante_8_diag_snap=ante_8_diag_snap,
            diag=diag,
            diag_cursor=_diag_cursor,
            tracked_diag_keys=_tracked_diag_keys,
            trigger_sa=cur_sa,
            trigger_sa_idx=cur_sa_idx,
            total_scheduled=len(scheduled_actions),
        )
        return True

    _record_tracked_step(state, None)

    t3 = time.perf_counter()
    A = ActionType
    targeted_actions = {
        A.BUY_CARD,
        A.BUY_AND_USE,
        A.OPEN_PACK,
        A.REDEEM_VOUCHER,
        A.USE_IN_PACK,
        A.USE_CONSUMABLE,
        A.SELL_CARD,
    }
    obs_cons_counts_by_sa_idx: dict[int, Counter[tuple[Any, Any]]] = {}
    obs_joker_counts_by_sa_idx: dict[int, Counter[Any]] = {}
    obs_pack_counts_by_sa_idx: dict[int, Counter[tuple[Any, ...]]] = {}
    obs_booster_counts_by_sa_idx: dict[int, Counter[tuple[Any, Any]]] = {}
    obs_voucher_counts_by_sa_idx: dict[int, Counter[Any]] = {}
    for i, s_act in enumerate(scheduled_actions):
        s_out = shop_res_by_frame.get(s_act.target_frame)
        if s_out is not None:
            raw_c = [
                rc
                for rc in (getattr(s_out, "consumeables", None) or [])
                if _is_high_conf_bar_det(rc)
            ]
            obs_c = _convert_items_of_type(raw_c, (Tarot, Planet, Spectral))
            obs_cons_counts_by_sa_idx[i] = Counter(_consumable_sig(c) for c in obs_c)
            raw_j = [
                rj
                for rj in (getattr(s_out, "joker_bar", None) or [])
                if _is_high_conf_bar_det(rj)
            ]
            obs_j = _convert_items_of_type(raw_j, Joker)
            obs_joker_counts_by_sa_idx[i] = Counter(j.name for j in obs_j)
            raw_p = list(getattr(s_out, "pack_items", None) or [])
            obs_p = _convert_items_of_type(
                raw_p, (Joker, Tarot, Planet, Spectral, PlayingCard)
            )
            obs_pack_counts_by_sa_idx[i] = Counter(
                sig for sig in (_pack_item_sig(p) for p in obs_p) if sig is not None
            )
            raw_b = list(getattr(s_out, "boosters", None) or [])
            obs_b = _convert_items_of_type(raw_b, BoosterPack)
            obs_booster_counts_by_sa_idx[i] = Counter(
                (b.booster_type, b.size) for b in obs_b
            )
            raw_v = list(getattr(s_out, "vouchers", None) or [])
            obs_v = _convert_items_of_type(raw_v, Voucher)
            obs_voucher_counts_by_sa_idx[i] = Counter(v.name for v in obs_v)

    def _next_obs_counts(
        cur_idx: int, by_idx: dict[int, Counter[Any]]
    ) -> Counter[Any] | None:
        for j in range(cur_idx + 1, len(scheduled_actions)):
            if j in by_idx:
                return by_idx[j]
        return None

    def _next_non_sort_idx(cur_idx: int) -> int | None:
        for j in range(cur_idx + 1, len(scheduled_actions)):
            if scheduled_actions[j].action_type not in (A.SORT_RANK, A.SORT_SUIT):
                return j
        return None

    def _next_hud_money(cur_idx: int) -> int | None:
        for j in range(cur_idx + 1, len(scheduled_actions)):
            nxt_hud = hud_res_by_frame.get(scheduled_actions[j].target_frame)
            m = _get_hud_val(nxt_hud, "hud_money")
            if m is not None:
                return int(m)
        return None

    def _next_k_hud_money(cur_idx: int, k: int = 3) -> list[int]:
        vals: list[int] = []
        for j in range(cur_idx + 1, len(scheduled_actions)):
            nxt_hud = hud_res_by_frame.get(scheduled_actions[j].target_frame)
            m = _get_hud_val(nxt_hud, "hud_money")
            if m is not None:
                vals.append(int(m))
                if len(vals) >= k:
                    break
        return vals

    def _next_same_blind_hud_money(cur_idx: int) -> int | None:
        for j in range(cur_idx + 1, len(scheduled_actions)):
            nxt = scheduled_actions[j]
            if nxt.action_type in (A.SORT_RANK, A.SORT_SUIT):
                continue
            if nxt.action_type not in (A.PLAY_HAND, A.DISCARD, A.END_ROUND):
                return None
            nxt_hud = hud_res_by_frame.get(nxt.target_frame)
            m = _get_hud_val(nxt_hud, "hud_money")
            return int(m) if m is not None else None
        return None

    def _next_hud_round(cur_idx: int) -> int | None:
        for j in range(cur_idx + 1, len(scheduled_actions)):
            nxt_hud = hud_res_by_frame.get(scheduled_actions[j].target_frame)
            r = _get_hud_val(nxt_hud, "hud_round")
            if r is not None:
                return int(r)
        return None

    last_consumable_mutated_frame: int = -1
    last_money_consumable_frame: int = -1
    last_cash_out_frame: int = -1
    steps_since_cash_out: int = 999
    last_joker_destroy_frame: int = -1
    shop_reroll_seq = 0
    shop_todo_ctx: tuple[int, int, int] | None = None
    shop_todo_unbought: list[PokerHand] = []
    pack_todo_ctx: tuple[int, int, int] | None = None
    pack_todo_unbought: list[PokerHand] = []
    shop_visit_active = False
    shop_visit_cashout_frame: int | None = None
    last_shop_obs: dict[str, Any] | None = None
    shop_visit_actions: list[ActionType] = []
    shop_visit_opens = 0

    fd_obs_idx = 0
    deck_suspect = False
    deck_suspect_reasons: list[str] = []
    phantom_deck_adds_since_sync = 0
    phantom_deck_add_ids: set[int] = set()
    unconfirmed_rng_cons_ids: set[int] = set()
    pending_rng_cons_budget: Counter[type] = Counter()
    unconfirmed_rng_joker_ids: set[int] = set()
    pending_rng_joker_budget = 0
    pending_cert_card_ids: set[int] = set()
    last_rebate_discard_info: tuple[int, int, Counter[Rank], int, int] | None = None
    halted_ante_9 = False

    def _mark_deck_suspect(reason: str) -> None:
        nonlocal deck_suspect
        deck_suspect = True
        if reason not in deck_suspect_reasons:
            deck_suspect_reasons.append(reason)

    def _run_lazy_full_deck_obs(
        obs: FullDeckObservation, target_action: Action | None
    ) -> bool:
        nonlocal deck_suspect, phantom_deck_adds_since_sync
        if not deck_suspect or obs.view_mode == "remaining":
            if diag is not None:
                diag["full_deck_views_skipped"] += 1
            return False

        alt_frame = (
            obs.start_frame + ((obs.end_frame - obs.start_frame) * 3) // 4
            if preloaded_frames is None
            and video_path is not None
            and (obs.end_frame - obs.start_frame) >= 6
            else None
        )
        if alt_frame == obs.sample_frame:
            alt_frame = None

        frame_img = (
            preloaded_frames.get(obs.sample_frame)
            if preloaded_frames is not None
            else None
        )
        extracted: dict[int, Any] = {}
        if frame_img is None and video_path is not None:
            want = [obs.sample_frame] + ([alt_frame] if alt_frame is not None else [])
            extracted = extract_video_frames(video_path, want)
            frame_img = extracted.get(obs.sample_frame)
        if frame_img is None:
            return False

        from classifiers.inference import recognize_full_deck
        from eval_template import detect_view_deck_tab_mode

        view_mode = obs.view_mode
        chosen_frame = obs.sample_frame
        used_alt_for_mode = False
        if view_mode == "unknown":
            view_mode = detect_view_deck_tab_mode(frame_img, obs.box)
            if view_mode != "full_deck":
                if (
                    alt_frame is not None
                    and alt_frame in extracted
                    and detect_view_deck_tab_mode(extracted[alt_frame], obs.box)
                    == "full_deck"
                ):
                    frame_img = extracted[alt_frame]
                    chosen_frame = alt_frame
                    view_mode = "full_deck"
                    used_alt_for_mode = True
                else:
                    if diag is not None:
                        diag["full_deck_views_skipped"] += 1
                    return False

        deck_out = recognize_full_deck(
            (frame_img, chosen_frame, chosen_frame / resolved_fps),
            view_mode=view_mode,
            device=device,
        )
        raw_deck_cards = list(getattr(deck_out, "cards", None) or ())

        def _deck_disagreement_score(raw_list: Sequence[Any]) -> tuple[int, int]:
            sim_cnt = Counter(_card_attr_sig(c) for c in state.full_deck)
            obs_cnt = Counter(
                _card_attr_sig(_convert_card_result_to_hand_card(rc).card)
                for rc in raw_list
            )
            matched = sum((sim_cnt & obs_cnt).values())
            unmatched = max(0, len(state.full_deck) - matched) + max(
                0, len(raw_list) - matched
            )
            return unmatched, abs(len(raw_list) - len(state.full_deck))

        cur_score = _deck_disagreement_score(raw_deck_cards)
        if (
            not used_alt_for_mode
            and alt_frame is not None
            and alt_frame in extracted
            and (
                abs(len(raw_deck_cards) - len(state.full_deck)) > 2
                or cur_score[0] > max(2, phantom_deck_adds_since_sync + 1)
            )
        ):
            alt_img = extracted[alt_frame]
            alt_mode = (
                detect_view_deck_tab_mode(alt_img, obs.box)
                if obs.view_mode == "unknown"
                else view_mode
            )
            if alt_mode == "full_deck":
                alt_out = recognize_full_deck(
                    (alt_img, alt_frame, alt_frame / resolved_fps),
                    view_mode=alt_mode,
                    device=device,
                )
                alt_cards = list(getattr(alt_out, "cards", None) or ())
                if _deck_disagreement_score(alt_cards) < cur_score:
                    deck_out = alt_out
                    raw_deck_cards = alt_cards
                    chosen_frame = alt_frame

        obs_playing_cards = [
            _convert_card_result_to_hand_card(c).card for c in raw_deck_cards
        ]
        obs_suit_confs = [
            float(confs["suit"])
            if isinstance((confs := getattr(c, "confidences", None)), dict)
            and "suit" in confs
            else None
            for c in raw_deck_cards
        ]
        obs_rank_confs = [
            float(confs["rank"])
            if isinstance((confs := getattr(c, "confidences", None)), dict)
            and "rank" in confs
            else None
            for c in raw_deck_cards
        ]
        summary = _sync_full_deck_into_state(
            state,
            obs_playing_cards,
            obs_suit_confs=obs_suit_confs,
            obs_rank_confs=obs_rank_confs,
            view_mode=getattr(deck_out, "view_mode", view_mode),
            phantom_adds=phantom_deck_adds_since_sync,
            phantom_card_ids=phantom_deck_add_ids,
            diag=diag,
            frame_id=chosen_frame,
            reasons=deck_suspect_reasons,
        )
        if summary is not None:
            deck_suspect = False
            deck_suspect_reasons.clear()
            phantom_deck_adds_since_sync = 0
            phantom_deck_add_ids.clear()
            if target_action is not None:
                target_action.metadata["full_deck_synced"] = summary
            return True
        return False

    def _consume_full_deck_views_up_to(
        limit_frame: int | None, target_action: Action | None
    ) -> bool:
        nonlocal fd_obs_idx
        synced_any = False
        while fd_obs_idx < len(full_deck_observations):
            obs = full_deck_observations[fd_obs_idx]
            if limit_frame is not None and obs.sample_frame > limit_frame:
                break
            fd_obs_idx += 1
            if _run_lazy_full_deck_obs(obs, target_action):
                synced_any = True
        return synced_any

    action_iter = (
        tqdm(
            enumerate(scheduled_actions),
            total=len(scheduled_actions),
            desc="Simulating trajectory",
            unit="action",
        )
        if expressive
        else enumerate(scheduled_actions)
    )
    for sa_idx, sa in action_iter:
        hud_peek = hud_res_by_frame.get(sa.target_frame)
        cur_hud_ante = _read_valid_hud_ante(hud_peek)
        if _maybe_halt_ante_9(sa_idx, sa, cur_hud_ante):
            halted_ante_9 = True
            break

        is_face_down_boss = (
            state.current_blind in FACE_DOWN_BOSS_BLINDS and not state.boss_disabled
        )
        if sa.action_type in (A.SORT_RANK, A.SORT_SUIT) and not is_face_down_boss:
            continue

        shop_synced_this_step = False

        action = Action(
            type=sa.action_type,
            frame_idx=sa.frame_id,
            timestamp=sa.frame_id / resolved_fps,
            metadata=dict(sa.metadata),
        )
        # Preserve CASH_OUT round payout calculation unless caller explicitly provided OCR override
        if (
            sa.action_type == A.CASH_OUT
            and "include_round_payout" not in action.metadata
        ):
            action.metadata["include_round_payout"] = True

        # Drop false-positive / cancelled Discard, cancelled Play Hand drag-offs, and
        # false-positive Reroll Shop BEFORE mutating hand/bars/deck/HUD.
        if sa.action_type == A.DISCARD:
            hud_disc = _get_hud_val(hud_peek, "hud_discards")
            if (state.discards_left or 0) <= 0 and hud_disc in (None, 0):
                for hc in state.hand.cards:
                    hc.selected = False
                continue
            cur_round = _get_hud_val(hud_peek, "hud_round")
            if hud_disc is not None and int(hud_disc) > 0:
                cancelled_discard = False
                for nxt in scheduled_actions[sa_idx + 1 :]:
                    if nxt.action_type in (
                        A.END_ROUND,
                        A.CASH_OUT,
                        A.SELECT_BLIND,
                        A.SKIP_BLIND,
                        A.NEXT_ROUND,
                    ):
                        break
                    if nxt.target_frame <= sa.frame_id:
                        continue
                    nxt_hud = hud_res_by_frame.get(nxt.target_frame)
                    nxt_disc = _get_hud_val(nxt_hud, "hud_discards")
                    nxt_round = _get_hud_val(nxt_hud, "hud_round")
                    if nxt_disc is not None:
                        if (
                            cur_round is None
                            or nxt_round is None
                            or int(nxt_round) == int(cur_round)
                        ) and int(nxt_disc) == int(hud_disc):
                            cancelled_discard = True
                        break
                if cancelled_discard:
                    for hc in state.hand.cards:
                        hc.selected = False
                    continue

        if (
            sa.action_type == A.REROLL_SHOP
            and state.free_rerolls == 0
            and _get_hud_val(hud_peek, "hud_reroll_cost") != 0
            and sa_idx + 1 < len(scheduled_actions)
        ):
            nxt = scheduled_actions[sa_idx + 1]
            if nxt.ambient_location == Location.SHOP and nxt.target_frame > sa.frame_id:
                nxt_hud = hud_res_by_frame.get(nxt.target_frame)
                nxt_rr = _get_hud_val(nxt_hud, "hud_reroll_cost")
                cur_m = _get_hud_val(hud_peek, "hud_money")
                nxt_m = _get_hud_val(nxt_hud, "hud_money")
                if (
                    nxt_rr is not None
                    and 0 < int(nxt_rr) <= state.reroll_cost
                    and (cur_m is None or nxt_m is None or int(nxt_m) >= int(cur_m))
                ):
                    continue

        if sa.action_type == A.PLAY_HAND:
            cur_hands = _get_hud_val(hud_peek, "hud_hands")
            cur_round = _get_hud_val(hud_peek, "hud_round")
            cur_score = _get_hud_val(hud_peek, "hud_round_score")
            if cur_hands is not None:
                cancelled_play = False
                for nxt in scheduled_actions[sa_idx + 1 :]:
                    if nxt.action_type in (
                        A.END_ROUND,
                        A.CASH_OUT,
                        A.SELECT_BLIND,
                        A.SKIP_BLIND,
                        A.NEXT_ROUND,
                    ):
                        break
                    if nxt.action_type == A.PLAY_HAND:
                        nxt_hud = hud_res_by_frame.get(nxt.target_frame)
                        nxt_hands = _get_hud_val(nxt_hud, "hud_hands")
                        nxt_round = _get_hud_val(nxt_hud, "hud_round")
                        nxt_score = _get_hud_val(nxt_hud, "hud_round_score")
                        if (
                            nxt_hands is not None
                            and int(nxt_hands) == int(cur_hands)
                            and (
                                cur_round is None
                                or nxt_round is None
                                or int(nxt_round) == int(cur_round)
                            )
                            and _scores_indicate_no_hand_scored(
                                cur_score, nxt_score, nxt.frame_id - sa.frame_id
                            )
                        ):
                            cancelled_play = True
                        break
                if cancelled_play:
                    for hc in state.hand.cards:
                        hc.selected = False
                    continue

        # Drop cancelled SELL_CARD hover/click directly above an immediate USE_CONSUMABLE click
        if sa.action_type == A.SELL_CARD and sa_idx + 1 < len(scheduled_actions):
            nxt = scheduled_actions[sa_idx + 1]
            if (
                nxt.action_type == A.USE_CONSUMABLE
                and 0 < (nxt.frame_id - sa.frame_id) <= 60
                and sa.button_box
                and nxt.button_box
            ):
                sx = (sa.button_box[0] + sa.button_box[2]) * 0.5
                sy = (sa.button_box[1] + sa.button_box[3]) * 0.5
                ux = (nxt.button_box[0] + nxt.button_box[2]) * 0.5
                uy = (nxt.button_box[1] + nxt.button_box[3]) * 0.5
                if abs(sx - ux) <= 80 and 0 < (uy - sy) <= 160:
                    cur_m = _get_hud_val(hud_peek, "hud_money")
                    nxt_m = _get_hud_val(
                        hud_res_by_frame.get(nxt.target_frame), "hud_money"
                    )
                    if cur_m is None or nxt_m is None or int(nxt_m) <= int(cur_m):
                        continue

        # Ensure blind transition leaves no stale hand cards from the prior round
        if sa.action_type in (A.SELECT_BLIND, A.SKIP_BLIND, A.CASH_OUT):
            state.hand.cards.clear()

        # Infer preclicked CASH_OUT when entering SHOP directly from BLIND without CASH_OUT
        if (
            state.location == Location.BLIND
            and sa.ambient_location == Location.SHOP
            and sa.action_type != A.CASH_OUT
        ):
            prev_fid = scheduled_actions[sa_idx - 1].frame_id if sa_idx > 0 else 0
            inferred_co_frame = max(prev_fid + 1, sa.frame_id - 1)
            synth_co = Action(
                type=A.CASH_OUT,
                frame_idx=inferred_co_frame,
                timestamp=inferred_co_frame / resolved_fps,
                metadata={
                    "inferred": True,
                    "source": "preclicked_cash_out",
                    "include_round_payout": True,
                },
            )
            state.hand.cards.clear()
            state.frame_idx = synth_co.frame_idx
            state.timestamp = synth_co.timestamp
            if not state.round_won:
                trigger_end_of_round_effects(state, synth_co)
            on_action(state, synth_co)
            _record_tracked_step(state, synth_co)
            shop_reroll_seq = 0
            shop_todo_ctx = None
            shop_todo_unbought.clear()
            shop_visit_active = True
            shop_visit_cashout_frame = synth_co.frame_idx
            last_shop_obs = None
            shop_visit_actions = []
            shop_visit_opens = 0
            last_cash_out_frame = synth_co.frame_idx
            steps_since_cash_out = 0
            if diag is not None:
                diag["inferred_cash_outs"] += 1
            if _maybe_halt_ante_9(sa_idx, sa, cur_hud_ante):
                break

        # Infer missed Next Round (and verified zero-skip Select Blind) when entering SELECT/BLIND from SHOP
        def _step_inferred_next_round(nr_frame: int) -> None:
            nonlocal shop_visit_active, shop_visit_cashout_frame, last_shop_obs
            nonlocal \
                shop_visit_actions, \
                shop_visit_opens, \
                last_cash_out_frame, \
                steps_since_cash_out
            synth_nr = Action(
                type=A.NEXT_ROUND,
                frame_idx=nr_frame,
                timestamp=nr_frame / resolved_fps,
                metadata={"inferred": True, "source": "preclicked_next_round"},
            )
            state.frame_idx = synth_nr.frame_idx
            state.timestamp = synth_nr.timestamp
            on_action(state, synth_nr)
            _record_tracked_step(state, synth_nr)
            shop_visit_active = False
            shop_visit_cashout_frame = None
            last_shop_obs = None
            shop_visit_actions = []
            shop_visit_opens = 0
            last_cash_out_frame = -1
            steps_since_cash_out = 999
            if diag is not None:
                diag["inferred_next_rounds"] += 1

        if (
            state.location == Location.SHOP
            and sa.ambient_location == Location.SELECT
            and sa.action_type != A.NEXT_ROUND
        ):
            prev_fid = scheduled_actions[sa_idx - 1].frame_id if sa_idx > 0 else 0
            inferred_nr_frame = max(prev_fid + 1, sa.frame_id - 1)
            _step_inferred_next_round(inferred_nr_frame)
            if _maybe_halt_ante_9(sa_idx, sa, cur_hud_ante):
                halted_ante_9 = True
                break
        elif (
            state.location == Location.SHOP
            and sa.ambient_location == Location.BLIND
            and sa.action_type in (A.PLAY_HAND, A.DISCARD, A.SORT_RANK, A.SORT_SUIT)
        ):
            exp_ante = (
                state.ante + 1
                if state.current_blind not in (BlindName.SMALL, BlindName.BIG)
                else state.ante
            )
            if state.current_blind == BlindName.SMALL:
                exp_blind = BlindName.BIG
            elif state.current_blind == BlindName.BIG:
                exp_blind = state.boss_blind or BlindName.THE_HOOK
            else:
                exp_blind = BlindName.SMALL
            exp_target = calculate_target_chips(
                ante=exp_ante,
                round=state.round,
                stake=state.stake,
                blind=exp_blind,
                deck=state.deck,
            )
            hud_ante_chk = _get_hud_val(hud_peek, "hud_ante")
            hud_round_chk = _get_hud_val(hud_peek, "hud_round")
            hud_target_chk = _get_hud_val(hud_peek, "hud_blind_target")
            if (
                hud_target_chk is not None
                and int(hud_target_chk) == exp_target
                and (hud_ante_chk is None or int(hud_ante_chk) == exp_ante)
                and (hud_round_chk is None or int(hud_round_chk) == state.round)
            ):
                prev_fid = scheduled_actions[sa_idx - 1].frame_id if sa_idx > 0 else 0
                inferred_nr_frame = max(prev_fid + 1, sa.frame_id - 2)
                _step_inferred_next_round(inferred_nr_frame)
                inferred_sb_frame = max(inferred_nr_frame + 1, sa.frame_id - 1)
                synth_sb = Action(
                    type=A.SELECT_BLIND,
                    frame_idx=inferred_sb_frame,
                    timestamp=inferred_sb_frame / resolved_fps,
                    metadata={"inferred": True, "source": "preclicked_select_blind"},
                )
                state.hand.cards.clear()
                state.frame_idx = synth_sb.frame_idx
                state.timestamp = synth_sb.timestamp
                on_action(state, synth_sb)
                _record_tracked_step(state, synth_sb)
                if _maybe_halt_ante_9(sa_idx, sa, cur_hud_ante):
                    halted_ante_9 = True
                    break

        shop_out = shop_res_by_frame.get(sa.target_frame)
        if state.location == Location.PACK and sa.ambient_location in (
            Location.SHOP,
            Location.SELECT,
        ):
            recovered_buffoon_pick = False
            if (
                state.active_pack is not None
                and state.active_pack.booster_type == BoosterType.BUFFOON
                and shop_out is not None
                and state.inventory is not None
            ):
                obs_j_list = _convert_items_of_type(
                    list(getattr(shop_out, "joker_bar", None) or []), Joker
                )
                prev_counts = Counter(
                    (j.name, j.edition, j.eternal, j.perishable, j.rental)
                    for j in state.inventory.jokers
                )
                gained_from_pack: list[Joker] = []
                for oj in obs_j_list:
                    sig = (oj.name, oj.edition, oj.eternal, oj.perishable, oj.rental)
                    if prev_counts[sig] > 0:
                        prev_counts[sig] -= 1
                    else:
                        gained_from_pack.append(oj)
                if 1 <= len(gained_from_pack) <= max(1, state.active_pack.choose):
                    recovered_buffoon_pick = True
                    for gj in gained_from_pack:
                        new_j = copy.deepcopy(gj)
                        state.inventory.jokers.append(new_j)
                        on_joker_added(state, new_j)
                    state.ensure_joker_ids()

            obs_cons_here = obs_cons_counts_by_sa_idx.get(sa_idx)
            held_cons_here = Counter(
                _consumable_sig(c)
                for c in (state.inventory.consumables if state.inventory else ())
            )
            cons_gained_in_pack = obs_cons_here is not None and any(
                obs_cons_here[sig] > held_cons_here[sig] for sig in obs_cons_here
            )
            if (
                not recovered_buffoon_pick
                and not cons_gained_in_pack
                and state.pack_choices_remaining > 0
            ):
                prev_fid = scheduled_actions[sa_idx - 1].frame_id if sa_idx > 0 else 0
                inferred_skip_frame = max(prev_fid + 1, sa.frame_id - 1)
                max_skips = 1 + len(getattr(state, "pending_packs", None) or ())
                skips_done = 0
                while state.location == Location.PACK and skips_done < max_skips:
                    skip_action = Action(
                        type=A.SKIP_PACK,
                        frame_idx=inferred_skip_frame + skips_done,
                        timestamp=(inferred_skip_frame + skips_done) / resolved_fps,
                        metadata={"inferred": True, "source": "preclicked_skip_pack"},
                    )
                    state.hand.cards.clear()
                    state.frame_idx = skip_action.frame_idx
                    state.timestamp = skip_action.timestamp
                    on_action(state, skip_action)
                    _record_tracked_step(state, skip_action)
                    skips_done += 1
                    if diag is not None:
                        diag["inferred_skip_packs"] += 1

            state.active_pack = None
            state.pack_cards.clear()
            state.pack_choices_remaining = 0
            if getattr(state, "pending_packs", None):
                state.pending_packs.clear()
            state.location = sa.ambient_location
            state.pre_pack_location = None
            pack_todo_ctx = None
            pack_todo_unbought.clear()

        raw_jokers: list[Any] = []
        raw_cons: list[Any] = []
        next_cons_counts = _next_obs_counts(sa_idx, obs_cons_counts_by_sa_idx)
        next_joker_counts = _next_obs_counts(sa_idx, obs_joker_counts_by_sa_idx)
        stale_cons_frame = (
            last_consumable_mutated_frame >= 0
            and sa.target_frame <= last_consumable_mutated_frame + 10
        )
        if shop_out is not None:
            cur_cons_counts = obs_cons_counts_by_sa_idx.get(sa_idx, Counter())
            if (
                unconfirmed_rng_cons_ids
                and state.inventory
                and state.inventory.consumables
            ):
                rem_obs_cons = Counter(cur_cons_counts)
                # Confirm already-confirmed slots first so an unconfirmed duplicate doesn't steal confirmation.
                for held_c in state.inventory.consumables:
                    if id(held_c) not in unconfirmed_rng_cons_ids:
                        sig = _consumable_sig(held_c)
                        if rem_obs_cons[sig] > 0:
                            rem_obs_cons[sig] -= 1
                for held_c in state.inventory.consumables:
                    if id(held_c) in unconfirmed_rng_cons_ids:
                        sig = _consumable_sig(held_c)
                        if rem_obs_cons[sig] > 0:
                            rem_obs_cons[sig] -= 1
                            unconfirmed_rng_cons_ids.discard(id(held_c))

            # Retroactively infer missed USE_CONSUMABLE actions when a held consumable
            # disappears in both the current and next observed CV frame.
            if (
                not stale_cons_frame
                and sa.action_type != A.SELL_CARD
                and next_cons_counts is not None
                and state.inventory
                and state.inventory.consumables
            ):
                prev_frame_id = (
                    scheduled_actions[sa_idx - 1].frame_id if sa_idx > 0 else 0
                )
                inferred_frame = max(prev_frame_id + 1, sa.frame_id - 1)
                hud_for_infer = hud_res_by_frame.get(sa.target_frame)
                hud_money_val = _get_hud_val(hud_for_infer, "hud_money")
                cur_cons_total = sum(cur_cons_counts.values())
                next_cons_total = sum(next_cons_counts.values())
                orig_consumables = list(state.inventory.consumables)
                for held_c in orig_consumables:
                    idx_cursor = next(
                        (
                            i
                            for i, c in enumerate(state.inventory.consumables)
                            if c is held_c
                        ),
                        None,
                    )
                    if idx_cursor is None:
                        continue
                    sig = _consumable_sig(held_c)
                    held_sig_total = sum(
                        1
                        for c in state.inventory.consumables
                        if _consumable_sig(c) == sig
                    )
                    next_count = next_cons_counts[sig]
                    c_name = getattr(held_c, "name", None)
                    inv_len = len(state.inventory.consumables)
                    if id(held_c) in unconfirmed_rng_cons_ids:
                        if (
                            sa.action_type != A.USE_CONSUMABLE
                            and cur_cons_counts[sig] < held_sig_total
                            and next_count < held_sig_total
                            and cur_cons_total < inv_len
                            and next_cons_total < inv_len
                        ):
                            state.inventory.consumables.pop(idx_cursor)
                            unconfirmed_rng_cons_ids.discard(id(held_c))
                            if diag is not None:
                                diag["rng_cons_reconciled"] += 1
                        continue
                    if sa.action_type == A.USE_CONSUMABLE:
                        if not (
                            isinstance(held_c, Planet) or isinstance(c_name, PlanetName)
                        ):
                            continue
                        if cur_cons_total < 1:
                            continue
                        if cur_cons_counts[sig] > 0:
                            if next_count >= cur_cons_counts[sig]:
                                continue
                        elif next_count != 0:
                            continue
                    sell_val = max(1, int(getattr(held_c, "sell_value", 1)))
                    is_buy_replace_sell = (
                        sa.action_type == A.BUY_CARD
                        and cur_cons_total < inv_len
                        and hud_money_val is not None
                        and hud_money_val == state.money + sell_val
                        and c_name not in (TarotName.HERMIT, TarotName.TEMPERANCE)
                    )
                    if (
                        cur_cons_counts[sig] < held_sig_total
                        and next_count < held_sig_total
                        and cur_cons_total < inv_len
                        and (next_cons_total < inv_len or is_buy_replace_sell)
                    ):
                        is_sell = is_buy_replace_sell or (
                            hud_money_val is not None
                            and hud_money_val >= state.money + sell_val
                            and c_name not in (TarotName.HERMIT, TarotName.TEMPERANCE)
                        )
                        if sa.action_type == A.USE_CONSUMABLE and is_sell:
                            continue
                        if is_sell:
                            joker_offset = len(state.inventory.jokers)
                            inferred_action = Action(
                                type=A.SELL_CARD,
                                frame_idx=inferred_frame,
                                timestamp=inferred_frame / resolved_fps,
                                target_idx=joker_offset + idx_cursor,
                                metadata={
                                    "inferred": True,
                                    "target_card": str(c_name or ""),
                                },
                            )
                        else:
                            inferred_action = Action(
                                type=A.USE_CONSUMABLE,
                                frame_idx=inferred_frame,
                                timestamp=inferred_frame / resolved_fps,
                                target_idx=idx_cursor,
                                metadata={
                                    "inferred": True,
                                    "target_card": str(c_name or ""),
                                },
                            )
                        inferred_unobs_tarot = (
                            not is_sell and c_name in _HAND_REQUIRING_CONSUMABLES
                        )
                        if inferred_unobs_tarot:
                            _mark_deck_suspect("inferred_deck_mutating_consumable")
                        state.frame_idx = inferred_action.frame_idx
                        state.timestamp = inferred_action.timestamp
                        on_action(state, inferred_action)
                        _record_tracked_step(
                            state,
                            inferred_action,
                            unobserved_deck_tarot=inferred_unobs_tarot,
                        )

            # Retroactively infer missed SELL_CARD actions when a non-eternal held Joker
            # disappears in both the current and next observed CV frame.
            cur_joker_counts = obs_joker_counts_by_sa_idx.get(sa_idx, Counter())
            is_amber_hidden = (
                state.current_blind == BlindName.AMBER_ACORN
                and not state.boss_disabled
                and state.inventory is not None
                and any(j.facing_back for j in state.inventory.jokers)
            ) or (
                state.inventory is not None
                and _looks_like_face_down_joker_bar(
                    cur_joker_counts, state.inventory.jokers, state.location
                )
            )
            ankh_hex_cooldown = (
                last_joker_destroy_frame >= 0
                and sa.frame_id <= last_joker_destroy_frame + 300
            )
            if (
                sa.action_type not in (A.SELL_CARD, A.END_ROUND)
                and not is_amber_hidden
                and not ankh_hex_cooldown
                and state.inventory
                and state.inventory.jokers
                and next_joker_counts is not None
            ):
                held_j_counts = Counter(j.name for j in state.inventory.jokers)
                held_total = len(state.inventory.jokers)
                shop_joker_names: set[Any] = set()
                if (
                    state.location == Location.SHOP
                    and sa.ambient_location == Location.SHOP
                ):
                    for sc in state.shop_cards or ():
                        if isinstance(sc, Joker):
                            shop_joker_names.add(sc.name)
                    for obs_sj in _convert_items_of_type(
                        list(getattr(shop_out, "shop_items", None) or []), Joker
                    ):
                        shop_joker_names.add(obs_sj.name)
                cur_shop_buy_gains = sum(
                    max(
                        0,
                        min(cur_joker_counts[n], next_joker_counts[n])
                        - held_j_counts[n],
                    )
                    for n in shop_joker_names
                )
                next_shop_buy_gains = sum(
                    max(0, next_joker_counts[n] - held_j_counts[n])
                    for n in shop_joker_names
                )
                is_buffoon_pick = (
                    sa.action_type == A.USE_IN_PACK
                    and state.active_pack is not None
                    and state.active_pack.booster_type == BoosterType.BUFFOON
                )
                next_allowed_gain = next_shop_buy_gains
                if (
                    sa.action_type == A.BUY_CARD and shop_joker_names
                ) or is_buffoon_pick:
                    next_allowed_gain = max(1, next_allowed_gain)
                eff_cur_total = sum(cur_joker_counts.values()) - cur_shop_buy_gains
                eff_next_total = sum(next_joker_counts.values()) - next_allowed_gain
                if eff_cur_total < held_total and eff_next_total < held_total:
                    rem_need = {
                        name: max(
                            0,
                            cnt - max(cur_joker_counts[name], next_joker_counts[name]),
                        )
                        for name, cnt in held_j_counts.items()
                    }
                    sell_candidates: list[Joker] = []
                    for held_j in state.inventory.jokers:
                        if (
                            not held_j.eternal
                            and id(held_j) not in unconfirmed_rng_joker_ids
                            and rem_need.get(held_j.name, 0) > 0
                        ):
                            sell_candidates.append(held_j)
                            rem_need[held_j.name] -= 1
                    if 0 < len(sell_candidates) < 3:
                        prev_frame_id = (
                            scheduled_actions[sa_idx - 1].frame_id if sa_idx > 0 else 0
                        )
                        inferred_frame = max(prev_frame_id + 1, sa.frame_id - 1)
                        hud_money_val = _get_hud_val(
                            hud_res_by_frame.get(sa.target_frame), "hud_money"
                        )
                        has_astronomer = any(
                            jj.name == JokerName.ASTRONOMER and not jj.debuffed
                            for jj in state.inventory.jokers
                        )
                        for held_j in sell_candidates:
                            if held_j.name in _SELF_DESTROY_JOKERS:
                                sell_val = held_j.get_sell_value(
                                    discount_percent=state.discount_percent,
                                    inflation=state.inflation,
                                    astronomer=has_astronomer,
                                )
                                if (
                                    hud_money_val is None
                                    or hud_money_val < state.money + sell_val
                                ):
                                    continue
                            j_cursor = next(
                                (
                                    i
                                    for i, j in enumerate(state.inventory.jokers)
                                    if j is held_j
                                ),
                                None,
                            )
                            if j_cursor is None:
                                continue
                            j_name = held_j.name
                            inferred_sell = Action(
                                type=A.SELL_CARD,
                                frame_idx=inferred_frame,
                                timestamp=inferred_frame / resolved_fps,
                                target_idx=j_cursor,
                                metadata={
                                    "inferred": True,
                                    "source": "missing_joker_bar",
                                    "target_card": str(
                                        getattr(j_name, "value", j_name)
                                    ),
                                },
                            )
                            state.frame_idx = inferred_sell.frame_idx
                            state.timestamp = inferred_sell.timestamp
                            on_action(state, inferred_sell)
                            _record_tracked_step(state, inferred_sell)
                            if diag is not None:
                                diag["inferred_joker_sells"] += 1

            # Retroactively infer missed BUY_CARD actions when a Joker seen on the shop shelf
            # vanishes from shop_items and appears in joker_bar in current and next CV frames.
            if (
                state.location == Location.SHOP
                and sa.ambient_location == Location.SHOP
                and sa.action_type != A.END_ROUND
                and (
                    sa.action_type != A.BUY_CARD
                    or match_button_to_item_index(
                        sa.button_box,
                        list(getattr(shop_out, "shop_items", None) or []),
                        attach="bottom_center",
                    )
                    is not None
                )
                and not is_amber_hidden
                and state.inventory is not None
                and state.shop_cards
            ):
                prev_shop_jokers = [c for c in state.shop_cards if isinstance(c, Joker)]
                if prev_shop_jokers:
                    cur_shop_joker_counts = Counter(
                        j.name
                        for j in _convert_items_of_type(
                            list(getattr(shop_out, "shop_items", None) or []), Joker
                        )
                    )
                    prev_shop_joker_counts = Counter(j.name for j in prev_shop_jokers)
                    held_j_counts = Counter(j.name for j in state.inventory.jokers)
                    prev_frame_id = (
                        scheduled_actions[sa_idx - 1].frame_id if sa_idx > 0 else 0
                    )
                    inferred_buy_frame = max(prev_frame_id + 1, sa.frame_id - 1)
                    for shop_j in prev_shop_jokers:
                        j_name = shop_j.name
                        if (
                            prev_shop_joker_counts[j_name]
                            > cur_shop_joker_counts[j_name]
                            and cur_joker_counts[j_name] > held_j_counts[j_name]
                            and (
                                next_joker_counts is None
                                or next_joker_counts[j_name] > held_j_counts[j_name]
                            )
                        ):
                            s_idx = next(
                                (
                                    i
                                    for i, c in enumerate(state.shop_cards)
                                    if c is shop_j
                                ),
                                None,
                            )
                            if s_idx is None:
                                continue
                            inferred_buy = Action(
                                type=A.BUY_CARD,
                                frame_idx=inferred_buy_frame,
                                timestamp=inferred_buy_frame / resolved_fps,
                                target_idx=s_idx,
                                metadata={
                                    "inferred": True,
                                    "source": "shop_to_joker_bar",
                                    "target_card": str(
                                        getattr(j_name, "value", j_name)
                                    ),
                                },
                            )
                            state.frame_idx = inferred_buy.frame_idx
                            state.timestamp = inferred_buy.timestamp
                            on_action(state, inferred_buy)
                            state.ensure_joker_ids()
                            _record_tracked_step(state, inferred_buy)
                            if shop_visit_active:
                                shop_visit_actions.append(A.BUY_CARD)
                            if diag is not None:
                                diag["inferred_joker_buys"] += 1
                            prev_shop_joker_counts[j_name] -= 1
                            held_j_counts[j_name] += 1

                prev_shop_cons = [
                    c for c in state.shop_cards if isinstance(c, (Tarot, Planet, Spectral))
                ]
                if prev_shop_cons:
                    cur_shop_cons_counts = Counter(
                        _consumable_sig(c)
                        for c in _convert_items_of_type(
                            list(getattr(shop_out, "shop_items", None) or []),
                            (Tarot, Planet, Spectral),
                        )
                    )
                    prev_shop_cons_counts = Counter(
                        _consumable_sig(c) for c in prev_shop_cons
                    )
                    held_c_counts = Counter(
                        _consumable_sig(c) for c in state.inventory.consumables
                    )
                    prev_frame_id = (
                        scheduled_actions[sa_idx - 1].frame_id if sa_idx > 0 else 0
                    )
                    inferred_buy_frame = max(prev_frame_id + 1, sa.frame_id - 1)
                    for shop_c in prev_shop_cons:
                        c_sig = _consumable_sig(shop_c)
                        if (
                            prev_shop_cons_counts[c_sig] > cur_shop_cons_counts[c_sig]
                            and cur_cons_counts[c_sig] > held_c_counts[c_sig]
                            and (
                                next_cons_counts is None
                                or next_cons_counts[c_sig] > held_c_counts[c_sig]
                            )
                        ):
                            s_idx = next(
                                (
                                    i
                                    for i, c in enumerate(state.shop_cards)
                                    if c is shop_c
                                ),
                                None,
                            )
                            if s_idx is None:
                                continue
                            c_disp = _display_card_name(shop_c)
                            inferred_buy_c = Action(
                                type=A.BUY_CARD,
                                frame_idx=inferred_buy_frame,
                                timestamp=inferred_buy_frame / resolved_fps,
                                target_idx=s_idx,
                                metadata={
                                    "inferred": True,
                                    "source": "shop_to_cons_bar",
                                    "target_card": c_disp,
                                },
                            )
                            state.frame_idx = inferred_buy_c.frame_idx
                            state.timestamp = inferred_buy_c.timestamp
                            on_action(state, inferred_buy_c)
                            _record_tracked_step(state, inferred_buy_c)
                            if shop_visit_active:
                                shop_visit_actions.append(A.BUY_CARD)
                            if diag is not None:
                                diag["inferred_joker_buys"] += 1
                            prev_shop_cons_counts[c_sig] -= 1
                            held_c_counts[c_sig] += 1

            cur_hud_money = _get_hud_val(
                hud_res_by_frame.get(sa.target_frame), "hud_money"
            )
            sync_cons_now = sa.action_type != A.END_ROUND and not stale_cons_frame
            raw_jokers, raw_cons = _sync_bars_into_state(
                state,
                shop_out,
                action,
                diag=diag,
                next_obs_cons_counts=next_cons_counts,
                next_obs_joker_counts=next_joker_counts,
                preserve_joker_flickers=(
                    sa.action_type == A.END_ROUND
                    or (
                        sa.action_type != A.SELL_CARD
                        and not ankh_hex_cooldown
                        and not (
                            state.location == Location.SHOP
                            and cur_hud_money is not None
                            and cur_hud_money > state.money
                        )
                    )
                ),
                suppress_joker_change_diag=ankh_hex_cooldown,
                unconfirmed_rng_cons_ids=unconfirmed_rng_cons_ids,
                pending_rng_cons_budget=pending_rng_cons_budget,
                unconfirmed_rng_joker_ids=unconfirmed_rng_joker_ids,
                pending_rng_joker_budget=pending_rng_joker_budget,
                hud_money_val=cur_hud_money,
                allow_new_jokers=(sa.action_type != A.END_ROUND),
                sync_jokers=True,
                sync_consumables=sync_cons_now,
            )
            pending_rng_joker_budget = 0
            if sync_cons_now:
                pending_rng_cons_budget.clear()

        # Sync main shop shelves before matching/stepping shop actions
        if sa.ambient_location == Location.SHOP and shop_out is not None:
            shop_synced_this_step = True
            pre_cv_shelves = _serialize_shop_shelves(state)
            raw_shop_items = _remap_softlocked_planets(
                list(getattr(shop_out, "shop_items", None) or []), state
            )
            # Guard against shop_classifier mislabeling a shop-shelf card as a Voucher or BoosterPack,
            # keeping `raw_shop_items` and `state.shop_cards` 1-to-1 aligned for `match_button_to_item_index`.
            _SHOP_CARD_TYPES = (Joker, Tarot, Planet, Spectral, PlayingCard)
            filtered_shop_pairs = [
                (raw_item, conv_obj)
                for raw_item, conv_obj in zip(
                    raw_shop_items, _convert_items_of_type(raw_shop_items)
                )
                if isinstance(conv_obj, _SHOP_CARD_TYPES)
            ]
            raw_shop_items = [pair[0] for pair in filtered_shop_pairs]
            raw_vouchers = list(getattr(shop_out, "vouchers", None) or [])
            raw_boosters = list(getattr(shop_out, "boosters", None) or [])

            state.shop_cards = [pair[1] for pair in filtered_shop_pairs]
            state.shop_vouchers = _convert_items_of_type(raw_vouchers, Voucher)
            state.shop_packs = _convert_items_of_type(raw_boosters, BoosterPack)
            if getattr(state, "coupon_shop_active", False):
                for item in state.shop_cards + state.shop_packs:
                    item.couponed = True
            cur_shop_ctx = (state.ante, state.round, shop_reroll_seq)
            if cur_shop_ctx != shop_todo_ctx:
                shop_todo_ctx = cur_shop_ctx
                shop_todo_unbought.clear()
            _sync_on_screen_to_do_list(state, state.shop_cards, shop_todo_unbought)
            post_cv_shelves = _serialize_shop_shelves(state)
            if any(post_cv_shelves.values()) and (
                post_cv_shelves != pre_cv_shelves
                or sa.action_type in (A.NEXT_ROUND, A.SELECT_BLIND)
            ):
                action.metadata["observed_shop"] = post_cv_shelves

            if sa.action_type == A.BUY_CARD:
                action.target_idx = match_button_to_item_index(
                    sa.button_box, raw_shop_items, attach="bottom_center"
                )
            elif sa.action_type == A.BUY_AND_USE:
                consumable_indices = [
                    idx
                    for idx, card_obj in enumerate(state.shop_cards)
                    if isinstance(card_obj, (Tarot, Planet, Spectral))
                    and idx < len(raw_shop_items)
                ]
                if consumable_indices:
                    rel_idx = match_button_to_item_index(
                        sa.button_box,
                        [raw_shop_items[idx] for idx in consumable_indices],
                        attach="center_right",
                    )
                    action.target_idx = (
                        consumable_indices[rel_idx] if rel_idx is not None else None
                    )
                else:
                    action.target_idx = match_button_to_item_index(
                        sa.button_box, raw_shop_items, attach="center_right"
                    )
            elif sa.action_type == A.OPEN_PACK:
                action.target_idx = match_button_to_item_index(
                    sa.button_box, raw_boosters, attach="bottom_center"
                )
            elif sa.action_type == A.REDEEM_VOUCHER:
                action.target_idx = match_button_to_item_index(
                    sa.button_box, raw_vouchers, attach="bottom_center"
                )

        # Sync booster pack contents before matching/stepping pack actions
        if sa.ambient_location == Location.PACK and shop_out is not None:
            raw_pack_items = _remap_softlocked_planets(
                list(getattr(shop_out, "pack_items", None) or []), state
            )
            pack_objs = _convert_items_of_type(
                raw_pack_items, (Joker, Tarot, Planet, Spectral, PlayingCard)
            )
            state.pack_cards = pack_objs
            if state.active_pack is None:
                inferred = _infer_booster_pack_from_items(pack_objs)
                state.active_pack = inferred
                state.pack_choices_remaining = inferred.choose
                state.pre_pack_location = sa.pre_pack_location_hint or Location.SHOP
            state.location = Location.PACK
            cur_pack_ctx = (id(state.active_pack), state.ante, state.round)
            if cur_pack_ctx != pack_todo_ctx:
                pack_todo_ctx = cur_pack_ctx
                pack_todo_unbought.clear()
            _sync_on_screen_to_do_list(state, state.pack_cards, pack_todo_unbought)
            if pack_objs:
                from game.trajectory_codec import _serialize_card

                action.metadata["observed_pack"] = [
                    _serialize_card(p) for p in pack_objs
                ]

            if sa.action_type == A.USE_IN_PACK:
                action.target_idx = match_button_to_item_index(
                    sa.button_box, raw_pack_items, attach="bottom_center"
                )

        # Spatial matching for top-bar actions (USE_CONSUMABLE, SELL_CARD)
        if sa.action_type == A.USE_CONSUMABLE:
            action.target_idx = match_button_to_item_index(
                sa.button_box, raw_cons, attach="center_right"
            )
        elif sa.action_type == A.SELL_CARD:
            combined_bar = raw_jokers + raw_cons
            action.target_idx = match_button_to_item_index(
                sa.button_box, combined_bar, attach="center_right"
            )

        if sa.action_type in targeted_actions and action.target_idx is None:
            nxt_idx_u = _next_non_sort_idx(sa_idx)
            if nxt_idx_u is not None:
                nxt_u = scheduled_actions[nxt_idx_u]
                if (
                    nxt_u.action_type == sa.action_type
                    and nxt_u.ambient_location == sa.ambient_location
                    and 0 < (nxt_u.frame_id - sa.frame_id) <= 45
                ):
                    continue

        # Drop cancelled SELL_CARD and duplicate/cancelled USE_CONSUMABLE clicks before hand/HUD/sim mutations
        if (
            sa.action_type == A.SELL_CARD
            and action.target_idx is not None
            and state.inventory is not None
        ):
            nxt_idx = _next_non_sort_idx(sa_idx)
            if nxt_idx is not None:
                nxt = scheduled_actions[nxt_idx]
                if nxt.target_frame > sa.frame_id and nxt.action_type not in (
                    A.BUY_CARD,
                    A.BUY_AND_USE,
                    A.USE_IN_PACK,
                    A.OPEN_PACK,
                    A.DISCARD,
                    A.END_ROUND,
                ):
                    cur_m = _get_hud_val(hud_peek, "hud_money")
                    nxt_m = _get_hud_val(
                        hud_res_by_frame.get(nxt.target_frame), "hud_money"
                    )
                    money_not_gained = (
                        cur_m is None or nxt_m is None or int(nxt_m) <= int(cur_m)
                    )
                    if (
                        0 <= action.target_idx < len(state.inventory.jokers)
                        and nxt_idx in obs_joker_counts_by_sa_idx
                    ):
                        j_name = state.inventory.jokers[action.target_idx].name
                        cur_cnt = obs_joker_counts_by_sa_idx.get(sa_idx, Counter())[
                            j_name
                        ]
                        nxt_cnt = obs_joker_counts_by_sa_idx[nxt_idx][j_name]
                        if cur_cnt > 0 and nxt_cnt >= cur_cnt and money_not_gained:
                            continue
                    else:
                        c_idx = action.target_idx - len(state.inventory.jokers)
                        if (
                            0 <= c_idx < len(state.inventory.consumables)
                            and nxt_idx in obs_cons_counts_by_sa_idx
                        ):
                            c_sig = _consumable_sig(state.inventory.consumables[c_idx])
                            cur_cnt = obs_cons_counts_by_sa_idx.get(sa_idx, Counter())[
                                c_sig
                            ]
                            nxt_cnt = obs_cons_counts_by_sa_idx[nxt_idx][c_sig]
                            if cur_cnt > 0 and nxt_cnt >= cur_cnt and money_not_gained:
                                continue
        elif (
            sa.action_type == A.USE_CONSUMABLE
            and action.target_idx is not None
            and state.inventory is not None
            and 0 <= action.target_idx < len(state.inventory.consumables)
        ):
            if (
                state.location in (Location.SHOP, Location.SELECT, Location.CASH_OUT)
                and sa.ambient_location
                in (Location.SHOP, Location.SELECT, Location.CASH_OUT)
                and getattr(
                    state.inventory.consumables[action.target_idx], "name", None
                )
                in _HAND_REQUIRING_CONSUMABLES
            ):
                continue
            nxt_idx = _next_non_sort_idx(sa_idx)
            if nxt_idx is not None:
                nxt = scheduled_actions[nxt_idx]
                if (
                    nxt.frame_id > sa.frame_id
                    and (
                        nxt.ambient_location == sa.ambient_location
                        or (nxt.frame_id - sa.frame_id) <= 300
                    )
                    and nxt.target_frame > sa.frame_id
                    and nxt_idx in obs_cons_counts_by_sa_idx
                ):
                    c_obj = state.inventory.consumables[action.target_idx]
                    c_name = getattr(c_obj, "name", None)
                    c_sig = _consumable_sig(c_obj)
                    cur_obs_c = obs_cons_counts_by_sa_idx.get(sa_idx, Counter())
                    nxt_obs_c = obs_cons_counts_by_sa_idx[nxt_idx]
                    cur_cnt = cur_obs_c[c_sig]
                    nxt_cnt = nxt_obs_c[c_sig]
                    cur_m = _get_hud_val(hud_peek, "hud_money")
                    nxt_m = _get_hud_val(
                        hud_res_by_frame.get(nxt.target_frame), "hud_money"
                    )
                    money_gained = (
                        cur_m is not None
                        and nxt_m is not None
                        and int(nxt_m) > int(cur_m)
                    )
                    econ_blocked = (
                        c_name
                        in (
                            TarotName.HERMIT,
                            TarotName.TEMPERANCE,
                            SpectralName.IMMOLATE,
                        )
                        and money_gained
                    )
                    drop_cancelled_use = False
                    if nxt.action_type == A.USE_CONSUMABLE:
                        same_btn = False
                        if sa.button_box and nxt.button_box:
                            cx1 = (sa.button_box[0] + sa.button_box[2]) * 0.5
                            cy1 = (sa.button_box[1] + sa.button_box[3]) * 0.5
                            cx2 = (nxt.button_box[0] + nxt.button_box[2]) * 0.5
                            cy2 = (nxt.button_box[1] + nxt.button_box[3]) * 0.5
                            same_btn = abs(cx1 - cx2) <= 60 and abs(cy1 - cy2) <= 60
                        nxt2_idx = _next_non_sort_idx(nxt_idx)
                        after_only_one_consumed = (
                            nxt2_idx is not None
                            and nxt2_idx in obs_cons_counts_by_sa_idx
                            and obs_cons_counts_by_sa_idx[nxt2_idx][c_sig]
                            >= cur_cnt - 1
                        )
                        drop_cancelled_use = (
                            cur_cnt > 0
                            and nxt_cnt >= cur_cnt
                            and not econ_blocked
                            and (same_btn or cur_cnt == 1 or after_only_one_consumed)
                        )
                    elif nxt.action_type == A.SELL_CARD:
                        drop_cancelled_use = (
                            cur_cnt > 0
                            and nxt_cnt >= cur_cnt
                            and sum(nxt_obs_c.values()) >= sum(cur_obs_c.values())
                        )
                    elif nxt.action_type in (
                        A.USE_IN_PACK,
                        A.SKIP_PACK,
                        A.PLAY_HAND,
                        A.DISCARD,
                        A.REROLL_SHOP,
                        A.BUY_CARD,
                        A.BUY_AND_USE,
                        A.OPEN_PACK,
                        A.REDEEM_VOUCHER,
                        A.NEXT_ROUND,
                        A.SELECT_BLIND,
                    ):
                        drop_cancelled_use = (
                            cur_cnt > 0
                            and c_name
                            not in (
                                TarotName.EMPEROR,
                                TarotName.HIGH_PRIESTESS,
                                TarotName.FOOL,
                            )
                            and nxt_obs_c == cur_obs_c
                            and not econ_blocked
                        )
                    if drop_cancelled_use:
                        for hc in state.hand.cards:
                            hc.selected = False
                        continue
        elif (
            sa.action_type == A.USE_IN_PACK
            and action.target_idx is not None
            and 0 <= action.target_idx < len(state.pack_cards)
        ):
            nxt_idx = _next_non_sort_idx(sa_idx)
            if nxt_idx is not None:
                nxt = scheduled_actions[nxt_idx]
                if (
                    nxt.action_type
                    in (A.USE_IN_PACK, A.SKIP_PACK, A.USE_CONSUMABLE, A.SELL_CARD)
                    and nxt.ambient_location == Location.PACK
                    and nxt.target_frame > sa.frame_id
                    and nxt_idx in obs_pack_counts_by_sa_idx
                ):
                    sig = _pack_item_sig(state.pack_cards[action.target_idx])
                    cur_c = obs_pack_counts_by_sa_idx.get(sa_idx, Counter())
                    nxt_c = obs_pack_counts_by_sa_idx[nxt_idx]
                    if (
                        sig is not None
                        and cur_c[sig] > 0
                        and nxt_c[sig] >= cur_c[sig]
                        and sum(nxt_c.values()) >= sum(cur_c.values())
                    ):
                        for hc in state.hand.cards:
                            hc.selected = False
                        continue
        elif (
            sa.ambient_location == Location.SHOP
            and sa.action_type in (A.OPEN_PACK, A.REDEEM_VOUCHER)
            and action.target_idx is not None
        ):
            nxt_idx = _next_non_sort_idx(sa_idx)
            if nxt_idx is not None:
                nxt = scheduled_actions[nxt_idx]
                if (
                    nxt.action_type == sa.action_type
                    and nxt.ambient_location == Location.SHOP
                    and nxt.target_frame > sa.frame_id
                ):
                    counts_map = (
                        obs_booster_counts_by_sa_idx
                        if sa.action_type == A.OPEN_PACK
                        else obs_voucher_counts_by_sa_idx
                    )
                    cur_cnt_map = counts_map.get(sa_idx, Counter())
                    nxt_cnt_map = counts_map.get(nxt_idx, Counter())
                    cur_m = _get_hud_val(hud_peek, "hud_money")
                    nxt_m = _get_hud_val(
                        hud_res_by_frame.get(nxt.target_frame), "hud_money"
                    )
                    if (
                        sum(cur_cnt_map.values()) > 0
                        and sum(nxt_cnt_map.values()) >= sum(cur_cnt_map.values())
                        and (cur_m is None or nxt_m is None or int(nxt_m) >= int(cur_m))
                    ):
                        continue
        elif (
            sa.ambient_location == Location.SHOP
            and sa.action_type in (A.BUY_CARD, A.BUY_AND_USE)
            and action.target_idx is not None
            and 0 <= action.target_idx < len(state.shop_cards)
        ):
            nxt_idx = _next_non_sort_idx(sa_idx)
            if nxt_idx is not None:
                nxt = scheduled_actions[nxt_idx]
                if (
                    nxt.ambient_location == Location.SHOP
                    and nxt.target_frame > sa.frame_id
                ):
                    cur_shop_out = shop_res_by_frame.get(sa.target_frame)
                    nxt_shop_out = shop_res_by_frame.get(nxt.target_frame)
                    cur_shop_cnt = Counter(
                        (getattr(it, "card_type", None), getattr(it, "key", None))
                        for it in (getattr(cur_shop_out, "shop_items", None) or ())
                    )
                    nxt_shop_cnt = Counter(
                        (getattr(it, "card_type", None), getattr(it, "key", None))
                        for it in (getattr(nxt_shop_out, "shop_items", None) or ())
                    )
                    shelf_still_present = sum(cur_shop_cnt.values()) > 0 and all(
                        nxt_shop_cnt[k] >= v for k, v in cur_shop_cnt.items()
                    )
                    if shelf_still_present:
                        cur_m = _get_hud_val(hud_peek, "hud_money")
                        nxt_m = _get_hud_val(
                            hud_res_by_frame.get(nxt.target_frame), "hud_money"
                        )
                        money_unchanged = (
                            cur_m is None
                            or nxt_m is None
                            or (
                                int(nxt_m) >= int(cur_m)
                                if sa.action_type == A.BUY_CARD
                                else int(nxt_m) == int(cur_m)
                            )
                        )
                        if money_unchanged:
                            bought_item = state.shop_cards[action.target_idx]
                            inv_unchanged = True
                            if (
                                isinstance(bought_item, Joker)
                                and nxt_idx in obs_joker_counts_by_sa_idx
                            ):
                                cur_jc = obs_joker_counts_by_sa_idx.get(
                                    sa_idx, Counter()
                                )[bought_item.name]
                                nxt_jc = obs_joker_counts_by_sa_idx[nxt_idx][
                                    bought_item.name
                                ]
                                inv_unchanged = nxt_jc <= cur_jc
                            elif (
                                sa.action_type == A.BUY_CARD
                                and not isinstance(bought_item, PlayingCard)
                                and nxt_idx in obs_cons_counts_by_sa_idx
                            ):
                                c_sig = _consumable_sig(bought_item)
                                cur_cc = obs_cons_counts_by_sa_idx.get(
                                    sa_idx, Counter()
                                )[c_sig]
                                nxt_cc = obs_cons_counts_by_sa_idx[nxt_idx][c_sig]
                                inv_unchanged = nxt_cc <= cur_cc
                            if inv_unchanged:
                                continue

        # Hand synchronization & sort handling
        step_full_deck_synced = _consume_full_deck_views_up_to(sa.frame_id, action)
        pre_selected_uid: int | None = None
        step_unexpected_adds = 0
        if sa.action_type in (A.SORT_RANK, A.SORT_SUIT):
            pre_hand = (
                hand_res_by_frame.get(sa.pre_sort_frame)
                if sa.pre_sort_frame is not None
                else None
            )
            if pre_hand is not None:
                obs_pre = [_convert_card_result_to_hand_card(c) for c in pre_hand.cards]
                if _check_over_15_hand_size(
                    state, action, obs_pre, has_hand_cv=True, diag=diag
                ):
                    _mark_deck_suspect("hand_size_over_15")
                _resolve_hand_with_deck_constraints(
                    state, pre_hand.cards, obs_pre, diag=diag
                )
                pre_sync_deck_len = (
                    len(state.full_deck) if state.full_deck is not None else 0
                )
                pre_selected_uid = _sync_hand_into_state(
                    state,
                    obs_pre,
                    action=action,
                    pending_cert_card_ids=pending_cert_card_ids,
                )
                if state.full_deck is not None:
                    added_now = max(0, len(state.full_deck) - pre_sync_deck_len)
                    if added_now > 0:
                        step_unexpected_adds += added_now
                        phantom_deck_adds_since_sync += added_now
                        phantom_deck_add_ids.update(
                            c.id
                            for c in state.full_deck[pre_sync_deck_len:]
                            if c.id != 0
                        )
                        diag["unexpected_deck_additions"].append(
                            (action.frame_idx, action.type.value, added_now)
                        )
                        _mark_deck_suspect("unexpected_deck_addition")
                if pre_selected_uid is not None:
                    action.metadata["selected_slot_uid"] = pre_selected_uid
        else:
            # If a round transition was missed, return prior round's discard/played cards
            # to draw_pile before syncing the new round's hand so full_deck doesn't bloat.
            hud_pre = hud_res_by_frame.get(sa.target_frame)
            hud_r_val = _get_hud_val(hud_pre, "hud_round")
            if (
                hud_r_val is not None
                and hud_r_val > state.round
                and _is_plausible_hud_round(
                    int(hud_r_val), state.round, state.ante, _next_hud_round(sa_idx)
                )
                and (state.discard_pile or state.played_cards_this_round)
            ):
                state.hand.cards.clear()
                state.discard_pile.clear()
                state.played_cards_this_round.clear()
                state.clear_boss_round_modifiers()
            hand_out = hand_res_by_frame.get(sa.target_frame)
            obs_cards = (
                [_convert_card_result_to_hand_card(c) for c in hand_out.cards]
                if hand_out is not None
                else []
            )
            if _check_over_15_hand_size(
                state,
                action,
                obs_cards,
                has_hand_cv=(
                    hand_out is not None or sa.action_type in (A.PLAY_HAND, A.DISCARD)
                ),
                diag=diag,
            ):
                _mark_deck_suspect("hand_size_over_15")
            if hand_out is not None:
                # Re-solve selection within legal Tarot/Spectral highlight bounds when out of range.
                if (
                    sa.action_type in (A.USE_CONSUMABLE, A.USE_IN_PACK)
                    and action.target_idx is not None
                    and obs_cards
                ):
                    target_cons = None
                    if (
                        sa.action_type == A.USE_CONSUMABLE
                        and state.inventory is not None
                        and 0 <= action.target_idx < len(state.inventory.consumables)
                    ):
                        target_cons = state.inventory.consumables[action.target_idx]
                    elif (
                        sa.action_type == A.USE_IN_PACK
                        and 0 <= action.target_idx < len(state.pack_cards)
                    ):
                        target_cons = state.pack_cards[action.target_idx]
                    bounds = (
                        _CONSUMABLE_HIGHLIGHT_LIMITS.get(
                            getattr(target_cons, "name", None)
                        )
                        if target_cons is not None
                        else None
                    )
                    if bounds is not None:
                        min_sel, max_sel = bounds
                        cur_sel_cnt = sum(1 for hc in obs_cards if hc.selected)
                        if cur_sel_cnt < min_sel or cur_sel_cnt > max_sel:
                            raw_boxes = [
                                getattr(c, "box", None) for c in hand_out.cards
                            ]
                            if len(raw_boxes) == len(obs_cards) and all(
                                isinstance(b, (list, tuple))
                                and len(b) == 4
                                and any(int(v) != 0 for v in b)
                                for b in raw_boxes
                            ):
                                from classifiers.card_classifier.pipeline import (
                                    determine_card_selection,
                                )

                                roi = getattr(hand_out, "roi", None)
                                if isinstance(roi, (list, tuple)) and len(roi) == 4:
                                    rx, ry, _, rh = roi
                                else:
                                    rx, ry, rh = 0, 0, 340
                                local_boxes = [
                                    [
                                        int(b[0]) - int(rx),
                                        int(b[1]) - int(ry),
                                        int(b[2]) - int(rx),
                                        int(b[3]) - int(ry),
                                    ]
                                    for b in raw_boxes
                                ]
                                bounded_sel = determine_card_selection(
                                    local_boxes,
                                    max(1, int(rh)),
                                    min_selected=min_sel,
                                    max_selected=max_sel,
                                )
                                for hc_obj, is_sel in zip(obs_cards, bounded_sel):
                                    hc_obj.selected = is_sel
                _resolve_hand_with_deck_constraints(
                    state, hand_out.cards, obs_cards, diag=diag
                )
                pre_sync_deck_len = (
                    len(state.full_deck) if state.full_deck is not None else 0
                )
                _sync_hand_into_state(
                    state,
                    obs_cards,
                    action=action,
                    pending_cert_card_ids=pending_cert_card_ids,
                )
                if (
                    sa.action_type == A.PLAY_HAND
                    and sa_idx in play_area_res_by_sa
                    and any(hc.selected and hc.facing_back for hc in state.hand.cards)
                ):
                    expected_cnt = sum(1 for hc in state.hand.cards if hc.selected)
                    consensus_raw = _select_consensus_play_area_cards(
                        play_area_res_by_sa[sa_idx], expected_cnt
                    )
                    if consensus_raw is not None:
                        _resolve_played_face_down_cards(
                            state, action, consensus_raw, diag=diag
                        )
                if state.full_deck is not None:
                    added_now = max(0, len(state.full_deck) - pre_sync_deck_len)
                    if added_now > 0:
                        step_unexpected_adds += added_now
                        phantom_deck_adds_since_sync += added_now
                        phantom_deck_add_ids.update(
                            c.id
                            for c in state.full_deck[pre_sync_deck_len:]
                            if c.id != 0
                        )
                        diag["unexpected_deck_additions"].append(
                            (action.frame_idx, action.type.value, added_now)
                        )
                        _mark_deck_suspect("unexpected_deck_addition")
                if (
                    state.hand.cards
                    and state.full_deck
                    and len(state.hand.cards) > len(state.full_deck)
                ):
                    _mark_deck_suspect("deck_size_mismatch")
            if diag is not None and sa.action_type in (A.PLAY_HAND, A.DISCARD):
                if hand_out is None or not getattr(hand_out, "cards", None):
                    diag["empty_hand_cv_actions"].append(sa.frame_id)

        if diag is not None:
            if (
                sa.ambient_location in (Location.SHOP, Location.PACK)
                and shop_out is None
            ):
                diag["missing_shop_cv_actions"].append(sa.frame_id)
            if sa.action_type in targeted_actions and action.target_idx is None:
                diag["unmatched_button_targets"].append(
                    (sa.frame_id, sa.action_type.value, sa.ambient_location.value)
                )
            expected_loc = (
                Location.BLIND
                if sa.ambient_location == Location.CASH_OUT
                else sa.ambient_location
            )
            if expected_loc != state.location:
                diag["location_mismatches"].append(
                    (
                        sa.frame_id,
                        sa.action_type.value,
                        sa.ambient_location.value,
                        state.location.value,
                    )
                )

        if sa.action_type in targeted_actions and "target_card" not in action.metadata:
            target_card_name = _resolve_target_card_name(state, action)
            if target_card_name is not None:
                action.metadata["target_card"] = target_card_name

        if (
            sa.action_type in (A.USE_CONSUMABLE, A.USE_IN_PACK)
            and action.metadata.get("target_card") in _ZERO_TARGET_CONSUMABLES
        ):
            action.selected_cards = []
            for hc in state.hand.cards:
                hc.selected = False

        # Sync HUD OCR counters before deterministic simulation (falling back on None predictions)
        hud_out = hud_res_by_frame.get(sa.target_frame)
        deferred_round_score: int | float | None = None
        if hud_out is not None:
            state._rebate_reconciled = False
            next_ns_idx = _next_non_sort_idx(sa_idx)
            is_end_round_victory_risk = sa.action_type == A.END_ROUND and (
                next_ns_idx is None
                or scheduled_actions[next_ns_idx].ambient_location == Location.CASH_OUT
            )
            deferred_round_score = _sync_hud_into_state(
                state,
                hud_out,
                sa,
                action,
                has_end_round_after=_has_end_round_after(scheduled_actions, sa_idx),
                next_hud_money=_next_hud_money(sa_idx),
                next_hud_moneys=_next_k_hud_money(sa_idx, k=3),
                next_hud_round=_next_hud_round(sa_idx),
                stale_money_drop_guard=(
                    (
                        last_money_consumable_frame >= 0
                        and sa.frame_id <= last_money_consumable_frame + 120
                    )
                    or (
                        last_cash_out_frame >= 0
                        and sa.action_type in (A.USE_CONSUMABLE, A.SELL_CARD)
                        and (
                            sa.frame_id <= last_cash_out_frame + 240
                            or steps_since_cash_out <= 3
                        )
                    )
                ),
                suppress_hands_discards=is_end_round_victory_risk,
                rebate_discard_info=last_rebate_discard_info,
                diag=diag,
            )
            if getattr(state, "_rebate_reconciled", False):
                last_rebate_discard_info = None
                state._rebate_reconciled = False

        if (
            sa.action_type in (A.USE_CONSUMABLE, A.USE_IN_PACK)
            and action.metadata.get("target_card") in _RANDOM_DESTROY_SPECTRALS
        ):
            survivors = _collect_pack_surviving_hand_ids(
                state, scheduled_actions, sa_idx, hand_res_by_frame
            )
            if survivors:
                action.metadata["_surviving_hand_ids"] = survivors
        if (
            sa.action_type == A.PLAY_HAND
            and state.current_blind == BlindName.THE_HOOK
            and not state.boss_disabled
        ):
            hook_ids = _collect_hook_discard_ids(
                state, scheduled_actions, sa_idx, hand_res_by_frame
            )
            if hook_ids:
                action.metadata["_hook_discard_ids"] = hook_ids

        # Check for unexpected shop shelf changes after cancelled-action filtering
        if shop_synced_this_step and shop_visit_active:
            cur_shop_cards = [_display_card_name(c) for c in state.shop_cards]
            cur_shop_vouchers = [
                str(getattr(v.name, "value", v.name)) for v in state.shop_vouchers
            ]
            cur_shop_packs = [
                f"{getattr(bp.size, 'value', bp.size)} {getattr(bp.booster_type, 'value', bp.booster_type)}"
                for bp in state.shop_packs
            ]
            unexp_shop_reasons = _detect_unexpected_shop_transition(
                last_shop_obs,
                cur_shop_cards,
                cur_shop_vouchers,
                cur_shop_packs,
                actions_between=shop_visit_actions,
                shop_visit_opens=shop_visit_opens,
                cashout_frame=shop_visit_cashout_frame,
            )
            if unexp_shop_reasons:
                if diag is not None:
                    diag["unexpected_shop_transitions"].append(
                        {
                            "prev_frame": (
                                last_shop_obs["frame_id"]
                                if last_shop_obs is not None
                                else shop_visit_cashout_frame
                            ),
                            "prev_action": (
                                last_shop_obs["action_type"]
                                if last_shop_obs is not None
                                else A.CASH_OUT.value
                            ),
                            "frame": action.frame_idx,
                            "action": action.type.value,
                            "ante": state.ante,
                            "round": state.round,
                            "actions_between": [a.value for a in shop_visit_actions],
                            "reasons": unexp_shop_reasons,
                        }
                    )
                action.metadata["unexpected_shop_transition"] = list(unexp_shop_reasons)
                if any(
                    r == "first_shop_obs_missing_packs"
                    or (
                        r.startswith("packs_lost=")
                        and any(p in r for p in ("Standard", "Arcana", "Spectral"))
                    )
                    for r in unexp_shop_reasons
                ):
                    _mark_deck_suspect("missed_deck_mutating_pack")
            if cur_shop_cards or cur_shop_vouchers or cur_shop_packs:
                last_shop_obs = {
                    "frame_id": action.frame_idx,
                    "action_type": action.type.value,
                    "cards": cur_shop_cards,
                    "vouchers": cur_shop_vouchers,
                    "packs": cur_shop_packs,
                }
                shop_visit_actions = []

        # Advance simulation
        bought_shop_td_k = (
            _find_todo_relative_index(state.shop_cards, action.target_idx)
            if sa.action_type == A.BUY_CARD
            else None
        )
        bought_pack_td_k = (
            _find_todo_relative_index(state.pack_cards, action.target_idx)
            if sa.action_type == A.USE_IN_PACK
            else None
        )
        state.frame_idx = action.frame_idx
        state.timestamp = action.timestamp
        prev_loc = state.location
        prev_cons_len = len(state.inventory.consumables) if state.inventory else 0
        if (
            sa.action_type in (A.USE_CONSUMABLE, A.USE_IN_PACK)
            and action.metadata.get("target_card") == TarotName.FOOL.value
            and state.inventory is not None
            and next_cons_counts is not None
        ):
            post_fool_held = Counter(
                _consumable_sig(c)
                for idx, c in enumerate(state.inventory.consumables)
                if not (sa.action_type == A.USE_CONSUMABLE and idx == action.target_idx)
            )
            new_sigs = [
                sig
                for sig, cnt in (next_cons_counts - post_fool_held).items()
                if cnt > 0
                and isinstance(sig[0], (TarotName, PlanetName))
                and sig[0] != TarotName.FOOL
            ]
            if len(new_sigs) == 1:
                state.last_tarot_planet = new_sigs[0][0]
        pre_cons_obj_ids = (
            {id(c) for c in state.inventory.consumables}
            if state.inventory is not None
            else set()
        )
        pre_joker_obj_ids = (
            {id(j) for j in state.inventory.jokers}
            if state.inventory is not None
            else set()
        )
        has_active_cert = bool(
            sa.action_type == A.SELECT_BLIND
            and state.inventory is not None
            and any(
                j.name == JokerName.CERTIFICATE and not j.debuffed
                for j in state.inventory.jokers
            )
        )
        pre_cert_deck_ids = (
            {c.id for c in state.full_deck} if has_active_cert else set()
        )
        selected_hand_snapshot = [hc for hc in state.hand.cards if hc.selected]
        held_hand_snapshot = [hc for hc in state.hand.cards if not hc.selected]
        step_unobserved_deck_tarot = False
        step_fd_deck_tarot = False
        if (
            sa.action_type in (A.USE_CONSUMABLE, A.USE_IN_PACK)
            and action.metadata.get("target_card") in _HAND_REQUIRING_CONSUMABLE_NAMES
        ):
            t_cname = action.metadata.get("target_card")
            min_req = next(
                (
                    lo
                    for k, (lo, _) in _CONSUMABLE_HIGHLIGHT_LIMITS.items()
                    if k.value == t_cname
                ),
                0,
            )
            if (
                state.hand.cards
                and min_req > 0
                and len(selected_hand_snapshot) < min_req
            ):
                _mark_deck_suspect("underselected_deck_tarot")
                step_unobserved_deck_tarot = True
            elif state.hand.cards:
                affected_hcs = (
                    selected_hand_snapshot if min_req > 0 else list(state.hand.cards)
                )
                if (
                    t_cname == TarotName.DEATH.value
                    and len(selected_hand_snapshot) >= 2
                ):
                    right_hc = selected_hand_snapshot[1]
                    right_src = (
                        right_hc._real_card if right_hc._real_card else right_hc.card
                    )
                    if (
                        right_src.rank == Rank.NONE
                        and right_src.enhancement != Enhancement.STONE
                    ):
                        _mark_deck_suspect("face_down_tarot_mutation")
                        step_unobserved_deck_tarot = True
                if not step_unobserved_deck_tarot and any(
                    hc.facing_back
                    and hc.card.id == 0
                    and (hc._real_card is None or hc._real_card.id == 0)
                    for hc in affected_hcs
                ):
                    _mark_deck_suspect("face_down_tarot_mutation")
                    step_fd_deck_tarot = True
        pre_action_money = state.money
        hud_money_before = _get_hud_val(hud_out, "hud_money")
        deck_has_purple = any(
            c.seal == Seal.PURPLE and not getattr(c, "debuffed", False)
            for c in state.full_deck
        )
        deck_has_blue = any(
            c.seal == Seal.BLUE and not getattr(c, "debuffed", False)
            for c in state.full_deck
        )
        if sa.action_type == A.DISCARD and state.inventory is not None:
            rebate_unit = _mail_in_rebate_payout_per_card(state)
            if rebate_unit > 0:
                disc_rank_counts = Counter(
                    hc.card.rank
                    for hc in state.hand.cards
                    if hc.selected
                    and not hc.facing_back
                    and not hc.card.debuffed
                    and hc.card.enhancement != Enhancement.STONE
                    and hc.card.rank != Rank.NONE
                )
                n_fd_discards = sum(
                    1
                    for hc in state.hand.cards
                    if hc.selected and hc.facing_back and not hc.card.debuffed
                )
                sim_rebate_matches = disc_rank_counts.get(state.mail_card_rank, 0)
                last_rebate_discard_info = (
                    state.round,
                    rebate_unit,
                    disc_rank_counts,
                    sim_rebate_matches,
                    n_fd_discards,
                )
            else:
                last_rebate_discard_info = None
        elif sa.action_type in (
            A.SELECT_BLIND,
            A.SKIP_BLIND,
            A.END_ROUND,
            A.CASH_OUT,
            A.NEXT_ROUND,
            A.PLAY_HAND,
        ):
            last_rebate_discard_info = None

        on_action(state, action)
        target_card_name = action.metadata.get("target_card")
        new_sim_cons: list[Any] = []
        if (
            sa.action_type
            in (A.SELECT_BLIND, A.PLAY_HAND, A.DISCARD, A.OPEN_PACK, A.END_ROUND)
            or (
                sa.action_type in (A.USE_CONSUMABLE, A.USE_IN_PACK, A.BUY_AND_USE)
                and target_card_name
                in (TarotName.EMPEROR.value, TarotName.HIGH_PRIESTESS.value)
            )
        ) and state.inventory is not None:
            for c in state.inventory.consumables:
                if id(c) not in pre_cons_obj_ids:
                    unconfirmed_rng_cons_ids.add(id(c))
                    new_sim_cons.append(c)

        if state.inventory is not None and (
            sa.action_type in (A.SELECT_BLIND, A.SKIP_BLIND)
            or (
                sa.action_type == A.SELL_CARD
                and target_card_name == JokerName.INVISIBLE_JOKER.value
            )
            or (
                sa.action_type in (A.USE_CONSUMABLE, A.USE_IN_PACK, A.BUY_AND_USE)
                and target_card_name
                in (
                    TarotName.JUDGEMENT.value,
                    SpectralName.WRAITH.value,
                    SpectralName.THE_SOUL.value,
                    SpectralName.ANKH.value,
                )
            )
        ):
            new_sim_joker_count = 0
            for j in state.inventory.jokers:
                if id(j) not in pre_joker_obj_ids:
                    unconfirmed_rng_joker_ids.add(id(j))
                    new_sim_joker_count += 1
            free_j_slots = max(
                0, state.inventory.max_jokers - len(state.inventory.jokers)
            )
            if (
                sa.action_type in (A.USE_CONSUMABLE, A.USE_IN_PACK, A.BUY_AND_USE)
                and new_sim_joker_count == 0
                and free_j_slots > 0
            ):
                pending_rng_joker_budget += 1
            elif sa.action_type == A.SELECT_BLIND:
                rr_copies = _active_joker_copies(state, JokerName.RIFF_RAFF)
                if rr_copies > 0 and free_j_slots > 0:
                    pending_rng_joker_budget += min(free_j_slots, 2 * rr_copies)

        if state.inventory is not None:
            free_c_slots = max(
                0, state.inventory.max_consumables - len(state.inventory.consumables)
            )
            if free_c_slots > 0:
                if sa.action_type == A.OPEN_PACK:
                    halu = _active_joker_copies(state, JokerName.HALLUCINATION)
                    if halu > 0:
                        pending_rng_cons_budget[Tarot] += min(free_c_slots, halu)
                elif sa.action_type == A.SELECT_BLIND:
                    cart = _active_joker_copies(state, JokerName.CARTOMANCER)
                    if cart > 0:
                        pending_rng_cons_budget[Tarot] += min(free_c_slots, cart)
                elif sa.action_type == A.PLAY_HAND and selected_hand_snapshot:
                    vag = _active_joker_copies(state, JokerName.VAGABOND)
                    if vag > 0 and (
                        pre_action_money <= 4
                        or (hud_money_before is not None and int(hud_money_before) <= 4)
                    ):
                        pending_rng_cons_budget[Tarot] += min(free_c_slots, vag)
                    eb = _active_joker_copies(state, JokerName.EIGHT_BALL)
                    if eb > 0:
                        n_8 = sum(
                            1
                            for hc in selected_hand_snapshot
                            if hc.facing_back
                            or (
                                hc.card.rank == Rank.EIGHT
                                and hc.card.enhancement != Enhancement.STONE
                                and not hc.card.debuffed
                            )
                        )
                        if n_8 > 0:
                            pending_rng_cons_budget[Tarot] += min(
                                free_c_slots, eb * n_8
                            )
                    has_fd_play = any(hc.facing_back for hc in selected_hand_snapshot)
                    sup = _active_joker_copies(state, JokerName.SUPERPOSITION)
                    sea = _active_joker_copies(state, JokerName.SEANCE)
                    if (sup > 0 or sea > 0) and selected_hand_snapshot:
                        jokers_now = state.inventory.jokers
                        if has_fd_play:
                            if sup > 0:
                                pending_rng_cons_budget[Tarot] += min(free_c_slots, sup)
                            if sea > 0:
                                pending_rng_cons_budget[Spectral] += min(
                                    free_c_slots, sea
                                )
                        else:
                            played_cards_now = [
                                hc.card for hc in selected_hand_snapshot
                            ]
                            p_hand = evaluate_poker_hand(played_cards_now, jokers_now)
                            if (
                                sup > 0
                                and p_hand
                                in (PokerHand.STRAIGHT, PokerHand.STRAIGHT_FLUSH)
                                and any(
                                    c.rank == Rank.ACE
                                    and c.enhancement != Enhancement.STONE
                                    for c in get_scoring_cards(
                                        played_cards_now, jokers_now
                                    )
                                )
                            ):
                                pending_rng_cons_budget[Tarot] += min(free_c_slots, sup)
                            if sea > 0 and p_hand == PokerHand.STRAIGHT_FLUSH:
                                pending_rng_cons_budget[Spectral] += min(
                                    free_c_slots, sea
                                )
                    has_sixth = any(
                        j.name == JokerName.SIXTH_SENSE and not j.debuffed
                        for j in state.inventory.jokers
                    )
                    if (
                        has_sixth
                        and state.hands_played_this_round <= 1
                        and len(selected_hand_snapshot) == 1
                        and (
                            selected_hand_snapshot[0].facing_back
                            or (
                                selected_hand_snapshot[0].card.rank == Rank.SIX
                                and selected_hand_snapshot[0].card.enhancement
                                != Enhancement.STONE
                            )
                        )
                    ):
                        pending_rng_cons_budget[Spectral] += min(free_c_slots, 1)
                elif sa.action_type == A.DISCARD and selected_hand_snapshot:
                    purple_cnt = sum(
                        1
                        for hc in selected_hand_snapshot
                        if (not hc.card.debuffed and hc.card.seal == Seal.PURPLE)
                        or (hc.facing_back and deck_has_purple)
                    )
                    if purple_cnt > 0:
                        pending_rng_cons_budget[Tarot] += min(free_c_slots, purple_cnt)

                if (
                    sa.action_type == A.END_ROUND
                    or (sa.action_type == A.PLAY_HAND and state.round_won)
                ) and held_hand_snapshot:
                    blue_cnt = sum(
                        1
                        for hc in held_hand_snapshot
                        if (not hc.card.debuffed and hc.card.seal == Seal.BLUE)
                        or (hc.facing_back and deck_has_blue)
                    )
                    if blue_cnt > 0:
                        mimes = _active_joker_copies(state, JokerName.MIME)
                        pending_rng_cons_budget[Planet] += min(
                            free_c_slots, blue_cnt * (2 + mimes)
                        )

                if sa.action_type in (A.USE_CONSUMABLE, A.USE_IN_PACK, A.BUY_AND_USE):
                    if target_card_name == TarotName.EMPEROR.value:
                        rem_cap = min(free_c_slots, max(0, 2 - len(new_sim_cons)))
                        if rem_cap > 0:
                            pending_rng_cons_budget[Tarot] += rem_cap
                    elif target_card_name == TarotName.HIGH_PRIESTESS.value:
                        rem_cap = min(free_c_slots, max(0, 2 - len(new_sim_cons)))
                        if rem_cap > 0:
                            pending_rng_cons_budget[Planet] += rem_cap

            if sa.action_type == A.NEXT_ROUND and state.inventory.consumables:
                perkeo = _active_joker_copies(state, JokerName.PERKEO)
                if perkeo > 0 and not new_sim_cons:
                    for c_type in {type(c) for c in state.inventory.consumables}:
                        pending_rng_cons_budget[c_type] += perkeo
        if has_active_cert:
            for c in state.full_deck:
                if c.id not in pre_cert_deck_ids and c.id:
                    pending_cert_card_ids.add(c.id)
        if sa.action_type in (
            A.SELECT_BLIND,
            A.SKIP_BLIND,
            A.CASH_OUT,
            A.NEXT_ROUND,
        ):
            state._last_play_lucky_triggers = 0
            state._last_play_lucky_money_hits = 0
        if sa.action_type in (A.END_ROUND, A.CASH_OUT, A.NEXT_ROUND):
            pending_cert_card_ids.clear()
        if _maybe_halt_ante_9(sa_idx, sa, cur_hud_ante):
            halted_ante_9 = True
            break
        if bought_shop_td_k is not None and 0 <= bought_shop_td_k < len(
            shop_todo_unbought
        ):
            shop_todo_unbought.pop(bought_shop_td_k)
        if bought_pack_td_k is not None and 0 <= bought_pack_td_k < len(
            pack_todo_unbought
        ):
            pack_todo_unbought.pop(bought_pack_td_k)
        if sa.action_type == A.REROLL_SHOP:
            shop_reroll_seq += 1
        elif sa.action_type in (
            A.CASH_OUT,
            A.SELECT_BLIND,
            A.SKIP_BLIND,
            A.NEXT_ROUND,
        ):
            shop_reroll_seq = 0
            shop_todo_ctx = None
            shop_todo_unbought.clear()
        if sa.action_type == A.CASH_OUT:
            shop_visit_active = True
            shop_visit_cashout_frame = action.frame_idx
            last_shop_obs = None
            shop_visit_actions = []
            shop_visit_opens = 0
            last_cash_out_frame = action.frame_idx
            steps_since_cash_out = 0
        elif sa.action_type in (
            A.NEXT_ROUND,
            A.SELECT_BLIND,
            A.SKIP_BLIND,
            A.END_ROUND,
        ):
            shop_visit_active = False
            shop_visit_cashout_frame = None
            last_shop_obs = None
            shop_visit_actions = []
            shop_visit_opens = 0
            if sa.action_type in (A.NEXT_ROUND, A.SELECT_BLIND, A.SKIP_BLIND):
                last_cash_out_frame = -1
                steps_since_cash_out = 999
            else:
                steps_since_cash_out += 1
        else:
            if shop_visit_active:
                shop_visit_actions.append(sa.action_type)
                if sa.action_type == A.OPEN_PACK:
                    shop_visit_opens += 1
            steps_since_cash_out += 1
        if state.location != Location.PACK:
            pack_todo_ctx = None
            pack_todo_unbought.clear()
        post_cons_len = len(state.inventory.consumables) if state.inventory else 0
        if sa.action_type == A.USE_CONSUMABLE or (
            sa.action_type == A.SELL_CARD and post_cons_len < prev_cons_len
        ):
            last_consumable_mutated_frame = action.frame_idx
        if sa.action_type == A.USE_CONSUMABLE and action.metadata.get(
            "target_card"
        ) in (
            TarotName.HERMIT.value,
            TarotName.TEMPERANCE.value,
        ):
            last_money_consumable_frame = action.frame_idx
        if sa.action_type in (A.USE_CONSUMABLE, A.USE_IN_PACK) and action.metadata.get(
            "target_card"
        ) in ("Ankh", "Hex"):
            last_joker_destroy_frame = action.frame_idx
        action.metadata.pop("_surviving_hand_ids", None)
        action.metadata.pop("_hook_discard_ids", None)
        if (
            sa.action_type == A.END_ROUND
            and shop_out is not None
            and not stale_cons_frame
        ):
            _, raw_cons = _sync_bars_into_state(
                state,
                shop_out,
                action,
                diag=diag,
                next_obs_cons_counts=next_cons_counts,
                unconfirmed_rng_cons_ids=unconfirmed_rng_cons_ids,
                pending_rng_cons_budget=pending_rng_cons_budget,
                sync_jokers=False,
                sync_consumables=True,
            )
            pending_rng_cons_budget.clear()
        if deferred_round_score is not None:
            if diag is not None and state.current_chips != deferred_round_score:
                diag["ocr_score_updates"] += 1
            state.current_chips = deferred_round_score

        # Post-sort reconciliation aligns actual settled positions + tightens face-down CardBounds
        if (
            is_face_down_boss
            and sa.action_type in (A.SORT_RANK, A.SORT_SUIT)
            and sa.post_sort_frame is not None
            and sa.post_sort_frame in hand_res_by_frame
        ):
            post_hand = hand_res_by_frame[sa.post_sort_frame]
            obs_post = [_convert_card_result_to_hand_card(c) for c in post_hand.cards]
            if _check_over_15_hand_size(
                state, action, obs_post, has_hand_cv=True, diag=diag
            ):
                _mark_deck_suspect("hand_size_over_15")
            _resolve_hand_with_deck_constraints(
                state, post_hand.cards, obs_post, diag=diag
            )
            pre_post_sort_len = (
                len(state.full_deck) if state.full_deck is not None else 0
            )
            _sync_hand_into_state(
                state,
                obs_post,
                action=None,
                selected_slot_uid=pre_selected_uid,
                apply_direct_order=True,
            )
            if state.full_deck is not None:
                added_post = max(0, len(state.full_deck) - pre_post_sort_len)
                if added_post > 0:
                    step_unexpected_adds += added_post
                    phantom_deck_adds_since_sync += added_post
                    phantom_deck_add_ids.update(
                        c.id
                        for c in state.full_deck[pre_post_sort_len:]
                        if c.id != 0
                    )
                    diag["unexpected_deck_additions"].append(
                        (action.frame_idx, action.type.value, added_post)
                    )
                    _mark_deck_suspect("unexpected_deck_addition")
            state.apply_sort_observation(
                sort_type="rank" if sa.action_type == A.SORT_RANK else "suit",
                selected_slot_uid=pre_selected_uid,
            )

        # Clear temporary hands when closing a booster pack or ending a round
        if sa.action_type in (A.END_ROUND, A.CASH_OUT, A.OPEN_PACK):
            state.hand.cards.clear()
        elif prev_loc == Location.PACK and state.location != Location.PACK:
            state.hand.cards.clear()

        _record_tracked_step(
            state,
            action,
            unexpected_adds=step_unexpected_adds,
            full_deck_synced=step_full_deck_synced,
            unobserved_deck_tarot=step_unobserved_deck_tarot,
            fd_deck_tarot=step_fd_deck_tarot,
        )

    if not halted_ante_9 and deck_suspect:
        _consume_full_deck_views_up_to(None, None)

    _scan_and_quarantine_corrupted_segments(trajectory, step_records, diag=diag)
    trajectory.impute_retroactive_boss_blinds()

    rescan_targets: list[dict[str, Any]] = list(
        diag.get("unexpected_shop_transitions") or ()
    )
    loc_mm_list = diag.get("location_mismatches") or ()
    if loc_mm_list and video_path is not None:
        mm_covered_until = -1
        for mm_fid, _mm_act, exp_loc, got_loc in loc_mm_list:
            if mm_fid <= mm_covered_until or got_loc != Location.SHOP.value:
                continue
            if exp_loc not in (Location.BLIND.value, Location.SELECT.value):
                continue
            mm_idx = next(
                (
                    i
                    for i, sa_item in enumerate(scheduled_actions)
                    if sa_item.frame_id == mm_fid
                ),
                None,
            )
            if mm_idx is None:
                continue
            prev_shop_fid = 0
            for j in range(mm_idx - 1, -1, -1):
                if scheduled_actions[j].ambient_location == Location.SHOP:
                    prev_shop_fid = scheduled_actions[j].frame_id
                    break
            rescan_targets.append(
                {
                    "prev_frame": prev_shop_fid,
                    "frame": mm_fid,
                    "reasons": ["vouchers_gained=location_mismatch"],
                }
            )
            mm_covered_until = mm_fid

    if (
        rescan_shop_transitions
        and not _is_rescan_pass
        and video_path is not None
        and rescan_targets
    ):
        t_rs = time.perf_counter()
        merged_payload, rs_stats = _rescan_unexpected_shop_windows(
            d, video_path, rescan_targets
        )
        rs_elapsed = time.perf_counter() - t_rs
        if merged_payload is not None:
            state, trajectory, plan = _build_run_trajectory_with_plan(
                merged_payload,
                video_path=video_path,
                preloaded_frames=preloaded_frames,
                seed=seed,
                deck=deck,
                stake=stake,
                fps=fps,
                conf_threshold_hand=conf_threshold_hand,
                conf_threshold_shop=conf_threshold_shop,
                conf_threshold_ocr=conf_threshold_ocr,
                batch_size=batch_size,
                device=device,
                deduplicate=deduplicate,
                expressive=expressive,
                rescan_shop_transitions=False,
                _is_rescan_pass=True,
            )
            diag = plan.setdefault("diagnostics", _init_diagnostics())
        diag["shop_rescan_windows"] = rs_stats["windows"]
        diag["shop_rescan_frames_scanned"] = rs_stats["frames_scanned"]
        diag["shop_rescan_recovered"] = rs_stats["recovered_frames"]
        diag["timings"]["shop_rescan"] = rs_elapsed

    if expressive:
        if _is_rescan_pass:
            diag["timings"]["simulation"] = time.perf_counter() - t3
            plan["diagnostics"] = diag
        else:
            diag["timings"].setdefault("simulation", time.perf_counter() - t3)
            plan["diagnostics"] = diag
            _print_expressive_report(diag)
    return state, trajectory, plan


def build_run_trajectory(
    d: dict[str, Any],
    video_path: str | Path | None = None,
    *,
    preloaded_frames: dict[int, Any] | None = None,
    seed: str = "",
    deck: Deck = Deck.RED_DECK,
    stake: Stake = Stake.WHITE,
    fps: float | None = None,
    conf_threshold_hand: float = 0.50,
    conf_threshold_shop: float = 0.15,
    conf_threshold_ocr: float = 0.90,
    batch_size: int = 32,
    device: str | None = None,
    deduplicate: bool = True,
    expressive: bool = False,
) -> tuple[GameState, RunTrajectory]:
    """Reconstructs a complete `(GameState, RunTrajectory)` from action JSON + video frames.

    Note: Starting mid-run is out of scope; always initializes from Ante 1 / Round 1.
    """
    state, trajectory, _ = _build_run_trajectory_with_plan(
        d,
        video_path=video_path,
        preloaded_frames=preloaded_frames,
        seed=seed,
        deck=deck,
        stake=stake,
        fps=fps,
        conf_threshold_hand=conf_threshold_hand,
        conf_threshold_shop=conf_threshold_shop,
        conf_threshold_ocr=conf_threshold_ocr,
        batch_size=batch_size,
        device=device,
        deduplicate=deduplicate,
        expressive=expressive,
    )
    return state, trajectory


def process_actions(
    d: dict[str, Any],
    video_path: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Schedules CV queues for action frames and optionally reconstructs full GameState trajectory.

    When `video_path` and `preloaded_frames` are omitted, returns the 5 CV queue dicts.
    When either is provided, automatically detects missing `END_ROUND` transitions, runs batched
    CV + HUD OCR, and returns the scheduled queues alongside `state` and `trajectory`.
    """
    if video_path is not None or kwargs.get("preloaded_frames") is not None:
        state, trajectory, plan = _build_run_trajectory_with_plan(
            d, video_path=video_path, **kwargs
        )
        return {
            "playing_card_cv_queue": plan["playing_card_cv_queue"],
            "joker_cv_queue": plan["joker_cv_queue"],
            "shop_cv_queue": plan["shop_cv_queue"],
            "pack_cv_queue": plan["pack_cv_queue"],
            "hud_ocr_queue": plan["hud_ocr_queue"],
            "state": state,
            "trajectory": trajectory,
        }
    plan = plan_action_pipeline(d)
    return {
        "playing_card_cv_queue": plan["playing_card_cv_queue"],
        "joker_cv_queue": plan["joker_cv_queue"],
        "shop_cv_queue": plan["shop_cv_queue"],
        "pack_cv_queue": plan["pack_cv_queue"],
        "hud_ocr_queue": plan["hud_ocr_queue"],
    }


def _normalize_enum_token(value: str) -> str:
    return "".join(ch for ch in value.strip().lower() if ch.isalnum())


def parse_deck(value: str | Deck) -> Deck:
    """Parses a CLI deck string (e.g. 'red', 'Red Deck', 'PLASMA_DECK') into `Deck`."""
    if isinstance(value, Deck):
        return value
    norm = _normalize_enum_token(value)
    for member in Deck:
        if norm in (
            _normalize_enum_token(member.name),
            _normalize_enum_token(member.value),
        ):
            return member
        short_name = _normalize_enum_token(member.name.removesuffix("_DECK"))
        if norm == short_name:
            return member
    valid = ", ".join(
        m.name.removesuffix("_DECK").lower() for m in Deck if m != Deck.CHALLENGE_DECK
    )
    raise ValueError(f"Unknown deck {value!r}. Valid choices: {valid}")


def parse_stake(value: str | Stake) -> Stake:
    """Parses a CLI stake string (e.g. 'white', 'White Stake', 'GOLD') into `Stake`."""
    if isinstance(value, Stake):
        return value
    norm = _normalize_enum_token(value)
    for member in Stake:
        if norm in (
            _normalize_enum_token(member.name),
            _normalize_enum_token(member.value),
        ):
            return member
    valid = ", ".join(m.name.lower() for m in Stake)
    raise ValueError(f"Unknown stake {value!r}. Valid choices: {valid}")


def serialize_trajectory_summary(
    state: GameState, trajectory: RunTrajectory
) -> dict[str, Any]:
    """Serializes reconstructed `(GameState, RunTrajectory)` into sparse delta JSON."""
    from game.trajectory_codec import serialize_trajectory

    return serialize_trajectory(state, trajectory)


def _build_post_processing_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Reconstruct Balatro GameState trajectory from action JSON + video."
    )
    p.add_argument(
        "--actions-json",
        type=str,
        required=True,
        help="Path to main.py action detection JSON",
    )
    p.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to gameplay video (.mp4/.mkv/.avi)",
    )
    p.add_argument("--seed", type=str, default="", help="Optional run seed")
    p.add_argument(
        "--deck",
        type=str,
        default="red",
        help="Deck name (e.g. red, blue, plasma, abandoned)",
    )
    p.add_argument(
        "--stake", type=str, default="white", help="Stake name (e.g. white, red, gold)"
    )
    p.add_argument("--fps", type=float, default=None, help="Override source video FPS")
    p.add_argument(
        "--conf-hand",
        type=float,
        default=0.50,
        help="Hand card detector confidence threshold",
    )
    p.add_argument(
        "--conf-shop",
        type=float,
        default=0.15,
        help="Shop detector confidence threshold",
    )
    p.add_argument(
        "--conf-ocr", type=float, default=0.90, help="HUD OCR confidence threshold"
    )
    p.add_argument("--batch-size", type=int, default=32, help="CV inference batch size")
    p.add_argument(
        "--device", type=str, default=None, help="Torch device (e.g. cuda, cpu)"
    )
    p.add_argument(
        "--expressive",
        action="store_true",
        help="Show progress bars and end-of-run diagnostics (OCR drift overrides, mismatches, timings)",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output JSON path for trajectory summary",
    )
    return p


def default_trajectory_output_path(actions_path: str | Path) -> Path:
    """Resolves default trajectory JSON path under a `trajectories/` folder."""
    ap = Path(actions_path)
    traj_dir = (
        ap.parent.parent / "trajectories"
        if ap.parent.name == "vod_actions"
        else ap.parent / "trajectories"
    )
    return traj_dir / f"{ap.stem}_trajectory.json"


if __name__ == "__main__":
    configure_logging()
    args = _build_post_processing_arg_parser().parse_args()
    with open(args.actions_json, encoding="utf-8") as f:
        data = json.load(f)

    state, trajectory = build_run_trajectory(
        data,
        video_path=args.video,
        seed=args.seed,
        deck=parse_deck(args.deck),
        stake=parse_stake(args.stake),
        fps=args.fps,
        conf_threshold_hand=args.conf_hand,
        conf_threshold_shop=args.conf_shop,
        conf_threshold_ocr=args.conf_ocr,
        batch_size=args.batch_size,
        device=args.device,
        expressive=args.expressive,
    )
    summary = serialize_trajectory_summary(state, trajectory)
    out_path = (
        Path(args.out)
        if args.out
        else default_trajectory_output_path(args.actions_json)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(
        f"Reconstructed {summary['num_steps']} steps | "
        f"Ante {state.ante} Round {state.round} ({state.current_blind.value}) | "
        f"Money=${state.money} | Saved: {out_path}"
    )
