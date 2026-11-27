"""Non-cascading trajectory comparison & telemetry test framework.

Reconstructs ground-truth trajectories from Lua `run_telemetry.jsonl` logs
(via `tools/telemetry_to_trajectory.py`) or pre-converted `*_trajectory.json`
files and compares them against scraped `*_trajectory.json` trajectories.

Comparison design minimizes false cascades:
  1. Sequence alignment (Needleman-Wunsch DP) matches actions by step context so
     missed/extra steps align without shifting every subsequent step.
  2. Persistent state (`full_deck`, `money`, `jokers`, `consumables`,
     `vouchers_redeemed`, `active_tags`, `hand_levels`, `hand_play_counts`) is
     compared using per-step transitions (deltas between `prev -> curr`) rather
     than cumulative snapshots, using ID-independent canonical card signatures.
  3. Pre-existing deck card property drift (e.g., an earlier missed enhancement)
     and Cash Out interest bracket offsets due to an earlier `$money` mismatch are
     detected and suppressed from cascading into later steps.

CLI Usage:
    python tools/compare_trajectories.py --actual output/trajectories/run.json --telemetry assets/local/run_telemetry.jsonl
    python tools/compare_trajectories.py --actual scraped.json --expected gt.json --verbose
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from game.balatro_models import (
    ActionType,
    BlindName,
    Card,
    Deck,
    Enhancement,
    Joker,
    JokerName,
    Location,
    PlayingCard,
    PokerHand,
    VoucherName,
)
from game.gamestate import GameState, GameStateDelta, RunTrajectory, apply_delta
from game.trajectory_codec import rebuild_trajectory

FACE_DOWN_BOSS_BLINDS: frozenset[BlindName] = frozenset(
    {
        BlindName.THE_HOUSE,
        BlindName.THE_WHEEL,
        BlindName.THE_FISH,
        BlindName.THE_MARK,
    }
)

_TARGET_ACTION_TYPES: frozenset[ActionType] = frozenset(
    {
        ActionType.BUY_CARD,
        ActionType.BUY_AND_USE,
        ActionType.SELL_CARD,
        ActionType.USE_CONSUMABLE,
        ActionType.OPEN_PACK,
        ActionType.USE_IN_PACK,
        ActionType.REDEEM_VOUCHER,
    }
)

_SELECTED_ACTION_TYPES: frozenset[ActionType] = frozenset(
    {
        ActionType.PLAY_HAND,
        ActionType.DISCARD,
        ActionType.USE_CONSUMABLE,
        ActionType.USE_IN_PACK,
    }
)

_HAND_CHECK_ACTION_TYPES: frozenset[ActionType] = frozenset(
    {
        ActionType.PLAY_HAND,
        ActionType.DISCARD,
        ActionType.USE_CONSUMABLE,
        ActionType.USE_IN_PACK,
        ActionType.SKIP_PACK,
    }
)

_TARGET_JOKER_FIELDS: tuple[tuple[JokerName, str, str], ...] = (
    (JokerName.MAIL_IN_REBATE, "mail_card_rank", "Mail-In Rebate Rank"),
    (JokerName.THE_IDOL, "idol_card", "The Idol Card"),
    (JokerName.ANCIENT_JOKER, "ancient_suit", "Ancient Joker Suit"),
    (JokerName.CASTLE, "castle_suit", "Castle Suit"),
    (JokerName.TO_DO_LIST, "to_do_poker_hand", "To Do List Hand"),
)

_SUIT_SYMBOLS: dict[str, str] = {
    "spades": "♠",
    "hearts": "♥",
    "clubs": "♣",
    "diamonds": "♦",
}

_RANK_SHORT: dict[str, str] = {
    "Ace": "A",
    "King": "K",
    "Queen": "Q",
    "Jack": "J",
}


# ---------------------------------------------------------------------------
# Data Containers
# ---------------------------------------------------------------------------


@dataclass
class MaterializedStep:
    step_idx: int
    state: GameState
    delta: GameStateDelta


@dataclass
class _StepFeature:
    orig_idx: int
    action_type: str
    ante: int
    round_num: int
    location: str
    blind: str
    target_norm: str
    selected_sigs: Counter[tuple[Any, ...]]
    hands_left: int | None
    discards_left: int | None


@dataclass
class Discrepancy:
    category: str
    kind: str
    field: str
    expected: Any
    actual: Any
    message: str
    cascaded: bool = False
    expected_step_idx: int | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "category": self.category,
            "kind": self.kind,
            "field": self.field,
            "expected": self.expected,
            "actual": self.actual,
            "message": self.message,
            "cascaded": self.cascaded,
            "expected_step_idx": self.expected_step_idx,
        }
        if self.detail:
            out["detail"] = self.detail
        return out


# ---------------------------------------------------------------------------
# Trajectory & Telemetry Loading
# ---------------------------------------------------------------------------


def _extract_seed_hint(source: Any) -> str | None:
    if isinstance(source, RunTrajectory):
        if source.deltas:
            return source.get_state(0).seed or None
        return None
    if isinstance(source, dict):
        seed = str(source.get("initial_state", {}).get("seed") or "").strip()
        return seed or None
    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.is_file() and p.suffix.lower() == ".json":
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                seed = str(raw.get("initial_state", {}).get("seed") or "").strip()
                return seed or None
            except Exception:
                return None
    return None


def load_trajectory_source(
    source: RunTrajectory | dict[str, Any] | str | Path,
    *,
    seed: str | None = None,
    actual_seed_hint: str | None = None,
    run_index: int | None = None,
    keep_all_sorts: bool = False,
    inject_end_round: bool = True,
    reconcile_truth: bool = True,
) -> tuple[RunTrajectory, dict[str, Any]]:
    """Loads a `RunTrajectory` from a `.json`, `.jsonl` telemetry file, dict, or instance."""
    if isinstance(source, RunTrajectory):
        st0 = source.get_state(0) if source.deltas else GameState(apply_deck_stake_on_init=False)
        return source, {
            "source_kind": "in_memory",
            "seed": st0.seed,
            "deck": st0.deck.value,
            "stake": st0.stake.value,
            "num_steps": len(source.deltas),
            "parity_warnings": [],
        }

    if isinstance(source, dict):
        traj = rebuild_trajectory(source, keyframe_interval=20)
        st0 = traj.get_state(0) if traj.deltas else GameState(apply_deck_stake_on_init=False)
        return traj, {
            "source_kind": "trajectory_dict",
            "seed": st0.seed,
            "deck": st0.deck.value,
            "stake": st0.stake.value,
            "num_steps": len(traj.deltas),
            "parity_warnings": [],
        }

    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Trajectory or telemetry file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        from tools.telemetry_to_trajectory import (
            build_trajectory_from_telemetry,
            load_telemetry_entries,
            select_run,
            split_telemetry_runs,
        )

        entries = load_telemetry_entries(path)
        runs = split_telemetry_runs(entries)
        chosen: list[dict[str, Any]] | None = None
        seed_mismatch = False
        if seed is None and run_index is None and actual_seed_hint:
            try:
                chosen = select_run(runs, seed=actual_seed_hint)
            except ValueError:
                chosen = select_run(runs, run_index=None, seed=None)
                seed_mismatch = True
        else:
            chosen = select_run(runs, run_index=run_index, seed=seed)

        final_state, traj, warnings = build_trajectory_from_telemetry(
            chosen,
            inject_end_round=inject_end_round,
            keep_all_sorts=keep_all_sorts,
            reconcile_truth=reconcile_truth,
        )
        st0 = traj.get_state(0) if traj.deltas else final_state
        return traj, {
            "source_kind": "telemetry_jsonl",
            "source_path": str(path.resolve()),
            "source_name": path.name,
            "seed": st0.seed,
            "deck": st0.deck.value,
            "stake": st0.stake.value,
            "num_steps": len(traj.deltas),
            "selected_run_actions": len(chosen),
            "available_runs": len(runs),
            "seed_mismatch": seed_mismatch,
            "parity_warnings": warnings,
        }

    raw = json.loads(path.read_text(encoding="utf-8"))
    traj = rebuild_trajectory(raw, keyframe_interval=20)
    st0 = traj.get_state(0) if traj.deltas else GameState(apply_deck_stake_on_init=False)
    return traj, {
        "source_kind": "trajectory_json",
        "source_path": str(path.resolve()),
        "source_name": path.name,
        "seed": st0.seed,
        "deck": st0.deck.value,
        "stake": st0.stake.value,
        "num_steps": len(traj.deltas),
        "parity_warnings": [],
    }


def _materialize_steps(traj: RunTrajectory) -> list[MaterializedStep]:
    """Reconstructs every `(GameState, GameStateDelta)` in a single O(N) replay pass."""
    if not traj.deltas:
        return []
    curr = traj.get_state(0)
    out = [MaterializedStep(step_idx=0, state=curr, delta=traj.deltas[0])]
    prev = curr
    for idx in range(1, len(traj.deltas)):
        delta = traj.deltas[idx]
        nxt = apply_delta(prev.copy(), delta)
        out.append(MaterializedStep(step_idx=idx, state=nxt, delta=delta))
        prev = nxt
    return out


# ---------------------------------------------------------------------------
# Canonical ID-Independent Signatures & Formatting
# ---------------------------------------------------------------------------


def _enum_str(val: Any) -> str:
    if val is None:
        return ""
    return str(val.value) if hasattr(val, "value") else str(val)


def _playing_card_sig(card: PlayingCard) -> tuple[Any, ...]:
    enh = _enum_str(card.enhancement) or "none"
    ed = _enum_str(card.edition) or "none"
    seal = _enum_str(card.seal) or "none"
    chips = int(getattr(card, "bonus_chips", 0) or 0)
    if enh == Enhancement.STONE.value:
        return ("Stone", "stone", "stone", ed, seal, chips)
    rank = _enum_str(card.rank) or "none"
    suit = _enum_str(card.suit) or "none"
    return (rank, suit, enh, ed, seal, chips)


def _format_card_sig(sig: tuple[Any, ...]) -> str:
    rank, suit, enh, ed, seal, chips = sig
    if enh == "stone":
        base = "Stone"
    else:
        r = _RANK_SHORT.get(str(rank), str(rank))
        s = _SUIT_SYMBOLS.get(str(suit).lower(), str(suit))
        base = f"{r}{s}"
    extras: list[str] = []
    if enh not in ("none", "stone", ""):
        extras.append(str(enh).title())
    if ed not in ("none", ""):
        extras.append(str(ed).title())
    if seal not in ("none", ""):
        extras.append(f"{str(seal).title()} Seal")
    if chips:
        extras.append(f"+{chips}")
    return f"{base} ({', '.join(extras)})" if extras else base


def _joker_sig(joker: Joker) -> tuple[Any, ...]:
    return (
        _enum_str(joker.name),
        _enum_str(joker.edition) or "none",
        bool(joker.eternal),
        bool(joker.perishable),
        bool(joker.rental),
    )


def _format_joker_sig(sig: tuple[Any, ...]) -> str:
    name, ed, eternal, perishable, rental = sig
    tags: list[str] = []
    if ed not in ("none", ""):
        tags.append(str(ed).title())
    if eternal:
        tags.append("Eternal")
    if perishable:
        tags.append("Perishable")
    if rental:
        tags.append("Rental")
    return f"{name} [{', '.join(tags)}]" if tags else str(name)


def _consumable_sig(card: Card) -> tuple[Any, ...]:
    return (
        _enum_str(getattr(card, "type", "")),
        _enum_str(getattr(card, "name", "")),
        _enum_str(getattr(card, "edition", "none")) or "none",
    )


def _format_consumable_sig(sig: tuple[Any, ...]) -> str:
    _, name, ed = sig
    if ed not in ("none", ""):
        return f"{name} [{str(ed).title()}]"
    return str(name)


def _format_counter_items(counter: Counter[Any], fmt_fn: Any) -> list[str]:
    out: list[str] = []
    for sig, cnt in sorted(counter.items(), key=lambda kv: str(kv[0])):
        label = fmt_fn(sig)
        for _ in range(cnt):
            out.append(label)
    return out


_BOOSTER_REPR_TYPE_RE = re.compile(r"booster_type=[^'\"]*['\"]([^'\"]+)['\"]")
_BOOSTER_REPR_SIZE_RE = re.compile(r"size=[^'\"]*['\"]([^'\"]+)['\"]")


def _clean_target_label(raw: str | None) -> str:
    s = str(raw or "").strip()
    if s.startswith("BoosterPack("):
        tm = _BOOSTER_REPR_TYPE_RE.search(s)
        sm = _BOOSTER_REPR_SIZE_RE.search(s)
        if tm:
            sz = sm.group(1) if sm else ""
            return f"{sz} {tm.group(1)} Pack".strip()
    return s


def _normalize_target_name(raw: str | None) -> str:
    s = " ".join(_clean_target_label(raw).lower().split())
    if not s:
        return ""
    if "pack" in s:
        s = re.sub(r"^(?:normal|regular)\s+", "", s)
    return s


def _card_display_label(card_obj: Any) -> str:
    if isinstance(card_obj, dict):
        if card_obj.get("name"):
            return str(card_obj["name"])
        if card_obj.get("booster_type"):
            return f"{card_obj.get('size', '')} {card_obj.get('booster_type', '')} Pack".strip()
        if card_obj.get("rank") or card_obj.get("suit"):
            return _format_card_sig(
                (
                    str(card_obj.get("rank", "none")),
                    str(card_obj.get("suit", "none")),
                    str(card_obj.get("enhancement", "none")),
                    str(card_obj.get("edition", "none")),
                    str(card_obj.get("seal", "none")),
                    int(card_obj.get("bonus_chips", 0) or 0),
                )
            )
        return ""
    if isinstance(card_obj, PlayingCard):
        return _format_card_sig(_playing_card_sig(card_obj))
    if hasattr(card_obj, "booster_type"):
        return f"{_enum_str(card_obj.size)} {_enum_str(card_obj.booster_type)} Pack".strip()
    if hasattr(card_obj, "name"):
        return _enum_str(card_obj.name)
    return str(card_obj)


def _resolve_target_from_prev(state: GameState | None, delta: GameStateDelta) -> str:
    act = delta.action
    if act is None:
        return ""
    meta = act.metadata or {}
    tc = meta.get("target_card")
    if isinstance(tc, str) and tc.strip():
        return _clean_target_label(tc)
    if isinstance(meta.get("card"), str) and meta["card"].strip():
        return _clean_target_label(meta["card"])
    if state is None or act.target_idx is None or act.target_idx < 0:
        return ""
    idx = act.target_idx
    A = ActionType
    if act.type in (A.BUY_CARD, A.BUY_AND_USE):
        pool: list[Any] = list(state.shop_cards)
    elif act.type == A.OPEN_PACK:
        pool = list(state.shop_packs)
    elif act.type == A.REDEEM_VOUCHER:
        pool = list(state.shop_vouchers)
    elif act.type == A.USE_IN_PACK:
        pool = list(state.pack_cards)
    elif act.type == A.USE_CONSUMABLE:
        pool = list(state.inventory.consumables) if state.inventory else []
    elif act.type == A.SELL_CARD:
        jokers = list(state.inventory.jokers) if state.inventory else []
        if act.joker_order and jokers:
            by_id = {j.id: j for j in jokers if j.id != 0}
            if len(act.joker_order) == len(jokers) and all(jid in by_id for jid in act.joker_order):
                jokers = [by_id[jid] for jid in act.joker_order]
            elif len(act.joker_order) == len(jokers) and all(
                isinstance(i, int) and 0 <= i < len(jokers) for i in act.joker_order
            ):
                jokers = [jokers[i] for i in act.joker_order]
        consumables = list(state.inventory.consumables) if state.inventory else []
        pool = jokers + consumables
    else:
        return ""
    if 0 <= idx < len(pool):
        return _card_display_label(pool[idx])
    return ""


def _selected_cards_counter(step: MaterializedStep) -> Counter[tuple[Any, ...]]:
    act = step.delta.action
    if act is None or not act.selected_cards:
        return Counter()
    return Counter(_playing_card_sig(c) for c in act.selected_cards)


def _visible_hand_counter(state: GameState) -> Counter[tuple[Any, ...]]:
    return Counter(
        _playing_card_sig(hc.card)
        for hc in state.hand.cards
        if not hc.facing_back
    )


def _deck_counter(state: GameState | None) -> Counter[tuple[Any, ...]]:
    if state is None or not state.full_deck:
        return Counter()
    return Counter(_playing_card_sig(c) for c in state.full_deck)


def _jokers_counter(state: GameState | None) -> Counter[tuple[Any, ...]]:
    if state is None or state.inventory is None:
        return Counter()
    return Counter(_joker_sig(j) for j in state.inventory.jokers)


def _consumables_counter(state: GameState | None) -> Counter[tuple[Any, ...]]:
    if state is None or state.inventory is None:
        return Counter()
    return Counter(_consumable_sig(c) for c in state.inventory.consumables)


def _vouchers_counter(state: GameState | None) -> Counter[str]:
    if state is None:
        return Counter()
    return Counter(_enum_str(v) for v in (state.vouchers_redeemed or []))


def _tags_counter(state: GameState | None) -> Counter[str]:
    if state is None:
        return Counter()
    return Counter(_enum_str(t) for t in (state.active_tags or []))


# ---------------------------------------------------------------------------
# Sequence Alignment (Needleman-Wunsch DP)
# ---------------------------------------------------------------------------


def _is_benign_sort(step: MaterializedStep, prev_state: GameState | None) -> bool:
    act = step.delta.action
    if step.step_idx == 0 or act is None:
        return False
    if act.type not in (ActionType.SORT_RANK, ActionType.SORT_SUIT):
        return False
    ref = prev_state or step.state
    is_face_down = ref.current_blind in FACE_DOWN_BOSS_BLINDS and not ref.boss_disabled
    return not is_face_down


def _build_features(
    steps: list[MaterializedStep],
    *,
    ignore_benign_sorts: bool = True,
) -> list[_StepFeature]:
    feats: list[_StepFeature] = []
    for i, st in enumerate(steps):
        prev_st = steps[i - 1].state if i > 0 else None
        if ignore_benign_sorts and _is_benign_sort(st, prev_st):
            continue
        act = st.delta.action
        act_type = _enum_str(act.type) if act is not None else "Init"
        target_raw = _resolve_target_from_prev(prev_st, st.delta)
        feats.append(
            _StepFeature(
                orig_idx=st.step_idx,
                action_type=act_type,
                ante=int(st.state.ante),
                round_num=int(st.state.round),
                location=_enum_str(st.state.location),
                blind=_enum_str(st.state.current_blind),
                target_norm=_normalize_target_name(target_raw),
                selected_sigs=_selected_cards_counter(st),
                hands_left=st.state.hands_left,
                discards_left=st.state.discards_left,
            )
        )
    return feats


def _pair_score(a: _StepFeature, e: _StepFeature) -> float:
    if abs(a.ante - e.ante) > 1:
        return -25.0
    score = 12.0 if a.action_type == e.action_type else -8.0
    if a.ante == e.ante and a.round_num == e.round_num:
        score += 4.0
    elif a.ante == e.ante:
        score += 2.0
    else:
        score -= 6.0

    score += 3.0 if a.location == e.location else -1.5
    if a.blind == e.blind:
        score += 2.5

    if a.target_norm and e.target_norm:
        score += 5.0 if a.target_norm == e.target_norm else -2.0

    if a.selected_sigs and e.selected_sigs:
        inter = sum((a.selected_sigs & e.selected_sigs).values())
        union = sum((a.selected_sigs | e.selected_sigs).values())
        if union > 0:
            score += 5.0 * (inter / union)

    return score


def _align_features(
    a_feats: list[_StepFeature],
    e_feats: list[_StepFeature],
) -> list[tuple[str, int | None, int | None]]:
    """Needleman-Wunsch sequence alignment keeping step 0 anchored to step 0."""
    if not a_feats and not e_feats:
        return []
    if not a_feats:
        return [("missing_in_actual", None, ef.orig_idx) for ef in e_feats]
    if not e_feats:
        return [("extra_in_actual", af.orig_idx, None) for af in a_feats]

    ops: list[tuple[str, int | None, int | None]] = [
        ("match", a_feats[0].orig_idx, e_feats[0].orig_idx)
    ]
    a_tail = a_feats[1:]
    e_tail = e_feats[1:]
    m = len(a_tail)
    n = len(e_tail)

    if m == 0 and n == 0:
        return ops
    if m == 0:
        return ops + [("missing_in_actual", None, ef.orig_idx) for ef in e_tail]
    if n == 0:
        return ops + [("extra_in_actual", af.orig_idx, None) for af in a_tail]

    gap_penalty = -5.0
    dp: list[list[float]] = [[0.0] * (n + 1) for _ in range(m + 1)]
    back: list[list[int]] = [[0] * (n + 1) for _ in range(m + 1)]  # 0=diag, 1=up(extra A), 2=left(miss E)

    for i in range(1, m + 1):
        dp[i][0] = i * gap_penalty
        back[i][0] = 1
    for j in range(1, n + 1):
        dp[0][j] = j * gap_penalty
        back[0][j] = 2

    for i in range(1, m + 1):
        ai = a_tail[i - 1]
        dp_i = dp[i]
        dp_prev = dp[i - 1]
        back_i = back[i]
        for j in range(1, n + 1):
            ej = e_tail[j - 1]
            s_diag = dp_prev[j - 1] + _pair_score(ai, ej)
            s_up = dp_prev[j] + gap_penalty
            s_left = dp_i[j - 1] + gap_penalty
            if s_diag >= s_up and s_diag >= s_left:
                dp_i[j] = s_diag
                back_i[j] = 0
            elif s_up >= s_left:
                dp_i[j] = s_up
                back_i[j] = 1
            else:
                dp_i[j] = s_left
                back_i[j] = 2

    tail_ops: list[tuple[str, int | None, int | None]] = []
    i, j = m, n
    while i > 0 or j > 0:
        move = back[i][j]
        if i > 0 and j > 0 and move == 0:
            tail_ops.append(("match", a_tail[i - 1].orig_idx, e_tail[j - 1].orig_idx))
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or move == 1):
            tail_ops.append(("extra_in_actual", a_tail[i - 1].orig_idx, None))
            i -= 1
        else:
            tail_ops.append(("missing_in_actual", None, e_tail[j - 1].orig_idx))
            j -= 1

    tail_ops.reverse()
    return ops + tail_ops


# ---------------------------------------------------------------------------
# Non-Cascading Diff Helpers
# ---------------------------------------------------------------------------


def _suppress_property_drift(
    actual_c: Counter[tuple[Any, ...]],
    expected_c: Counter[tuple[Any, ...]],
    prev_drift_a: Counter[tuple[Any, ...]],
    prev_drift_e: Counter[tuple[Any, ...]],
) -> tuple[Counter[tuple[Any, ...]], Counter[tuple[Any, ...]], int]:
    """Suppresses differences explained by pre-existing deck card property drift.

    For example, if an earlier step failed to apply `Mult` onto `A♠`, `prev_drift_a`
    holds `A♠(none)` and `prev_drift_e` holds `A♠(mult)`. When that card appears in
    later `hand` or `selected_cards`, both share rank+suit `(Ace, spades)` and match
    existing drift slots, preventing endless cascade errors across future hands.
    Genuine rank/suit misreads (e.g. `K♣` vs `Q♣`) do NOT share `(rank, suit)` and
    are never suppressed.
    """
    rem_a = actual_c - expected_c
    rem_e = expected_c - actual_c
    if not rem_a or not rem_e or not prev_drift_a or not prev_drift_e:
        return rem_a, rem_e, 0

    avail_a = Counter(prev_drift_a)
    avail_e = Counter(prev_drift_e)
    suppressed = 0

    for sig_a in sorted(rem_a.keys(), key=str):
        if rem_a[sig_a] <= 0 or avail_a[sig_a] <= 0:
            continue
        for sig_e in sorted(rem_e.keys(), key=str):
            if rem_e[sig_e] <= 0 or avail_e[sig_e] <= 0:
                continue
            if sig_a[:2] != sig_e[:2]:
                continue
            # Require at least one property difference on matching rank+suit (or both Stone)
            matched = min(rem_a[sig_a], rem_e[sig_e], avail_a[sig_a], avail_e[sig_e])
            if matched > 0:
                rem_a[sig_a] -= matched
                rem_e[sig_e] -= matched
                avail_a[sig_a] -= matched
                avail_e[sig_e] -= matched
                suppressed += matched
                if rem_a[sig_a] <= 0 or avail_a[sig_a] <= 0:
                    break

    rem_a = +rem_a
    rem_e = +rem_e
    return rem_a, rem_e, suppressed


def _filter_collection_cascade(
    prev_a_c: Counter[Any],
    curr_a_c: Counter[Any],
    prev_e_c: Counter[Any],
    curr_e_c: Counter[Any],
    *,
    suppress_rank_suit_drift: bool = False,
) -> tuple[bool, bool]:
    """Checks whether a collection transition diff is purely self-correction / pre-existing drift.

    Returns `(has_transition_diff, is_cascaded)`.
    """
    a_gain = curr_a_c - prev_a_c
    a_loss = prev_a_c - curr_a_c
    e_gain = curr_e_c - prev_e_c
    e_loss = prev_e_c - curr_e_c
    if (a_gain, a_loss) == (e_gain, e_loss):
        return False, False

    prev_missing_in_a = prev_e_c - prev_a_c
    prev_extra_in_a = prev_a_c - prev_e_c

    extra_a_gain = (a_gain - e_gain) - prev_missing_in_a
    missing_a_gain = (e_gain - a_gain) - prev_extra_in_a
    extra_a_loss = (a_loss - e_loss) - prev_extra_in_a
    missing_a_loss = (e_loss - a_loss) - prev_missing_in_a

    if suppress_rank_suit_drift and prev_extra_in_a and prev_missing_in_a:
        extra_a_loss, missing_a_loss, _ = _suppress_property_drift(
            extra_a_loss, missing_a_loss, prev_extra_in_a, prev_missing_in_a
        )
        extra_a_gain, missing_a_gain, _ = _suppress_property_drift(
            extra_a_gain, missing_a_gain, prev_extra_in_a, prev_missing_in_a
        )

    is_cascade = not (extra_a_gain or missing_a_gain or extra_a_loss or missing_a_loss)
    return True, is_cascade


def _calc_interest(state: GameState) -> int:
    if state.no_interest or state.deck == Deck.GREEN_DECK:
        return 0
    money = int(state.money or 0)
    if money < 5:
        return 0
    vouchers = set(state.vouchers_redeemed or [])
    if VoucherName.MONEY_TREE in vouchers:
        cap = 20
    elif VoucherName.SEED_MONEY in vouchers:
        cap = 10
    else:
        cap = 5
    to_the_moon = sum(
        1
        for j in (state.inventory.jokers if state.inventory else [])
        if j.name == JokerName.TO_THE_MOON and not j.debuffed
    )
    return (1 + to_the_moon) * min(money // 5, cap)


def _format_deck_delta(gained: Counter[tuple[Any, ...]], lost: Counter[tuple[Any, ...]]) -> str:
    parts: list[str] = []
    if gained:
        parts.append("+[" + ", ".join(_format_counter_items(gained, _format_card_sig)) + "]")
    if lost:
        parts.append("-[" + ", ".join(_format_counter_items(lost, _format_card_sig)) + "]")
    return " ".join(parts) if parts else "no change"


def _format_signed_delta(d: int, prefix: str = "") -> str:
    if d > 0:
        return f"+{prefix}{d}"
    if d < 0:
        return f"-{prefix}{abs(d)}"
    return f"{prefix}0"


def _summarize_action(step: MaterializedStep, prev_state: GameState | None) -> str:
    act = step.delta.action
    if act is None:
        return "Init"
    atype = _enum_str(act.type)
    target = _resolve_target_from_prev(prev_state, step.delta)
    if target:
        return f"{atype} ({target})"
    if act.selected_cards:
        sel = ", ".join(_format_card_sig(_playing_card_sig(c)) for c in act.selected_cards)
        return f"{atype} [{sel}]"
    return atype


def _shop_item_sig(item: Any, fallback_kind: str) -> tuple[Any, ...]:
    if isinstance(item, dict):
        if item.get("booster_type"):
            return ("pack", _normalize_target_name(f"{item.get('size', '')} {item.get('booster_type', '')} Pack"))
        if fallback_kind == "voucher" or item.get("type") == "voucher":
            return ("voucher", str(item.get("name", "")).strip().lower())
        if item.get("rank") or item.get("suit"):
            return (
                "card",
                str(item.get("rank", "none")),
                str(item.get("suit", "none")),
                str(item.get("enhancement", "none")),
                str(item.get("edition", "none")),
            )
        return (
            str(item.get("type", fallback_kind)),
            str(item.get("name", "")).strip().lower(),
            str(item.get("edition", "none")).lower(),
        )
    if isinstance(item, PlayingCard):
        return ("playing_card",) + _playing_card_sig(item)
    if hasattr(item, "booster_type"):
        return ("pack", _normalize_target_name(f"{_enum_str(item.size)} {_enum_str(item.booster_type)} Pack"))
    if hasattr(item, "name"):
        return (
            fallback_kind,
            _enum_str(item.name).strip().lower(),
            _enum_str(getattr(item, "edition", "none")).lower(),
        )
    return (fallback_kind, str(item).strip().lower())


def _extract_observed_shop_counter(step: MaterializedStep) -> Counter[tuple[Any, ...]] | None:
    act = step.delta.action
    if act is None:
        return None
    obs = (act.metadata or {}).get("observed_shop")
    if not isinstance(obs, dict):
        return None
    out: Counter[tuple[Any, ...]] = Counter()
    for c in obs.get("cards") or ():
        out[_shop_item_sig(c, "card")] += 1
    for v in obs.get("vouchers") or ():
        out[_shop_item_sig(v, "voucher")] += 1
    for p in obs.get("packs") or ():
        out[_shop_item_sig(p, "pack")] += 1
    return out


def _extract_observed_pack_counter(step: MaterializedStep) -> Counter[tuple[Any, ...]] | None:
    act = step.delta.action
    if act is None:
        return None
    obs = (act.metadata or {}).get("observed_pack")
    if not isinstance(obs, list) or not obs:
        return None
    out: Counter[tuple[Any, ...]] = Counter()
    for item in obs:
        out[_shop_item_sig(item, "pack_card")] += 1
    return out


def _format_shop_sig(sig: tuple[Any, ...]) -> str:
    kind = sig[0]
    if (
        kind in ("playing_card", "card")
        and len(sig) >= 5
        and (sig[1] in _RANK_SHORT or sig[2] in _SUIT_SYMBOLS or str(sig[1]).isdigit())
    ):
        return f"{sig[1]} of {sig[2]}"
    if len(sig) >= 2:
        ed = sig[2] if len(sig) >= 3 and sig[2] not in ("none", "") else ""
        return f"{sig[1]} [{ed}]" if ed else str(sig[1])
    return str(sig)


# ---------------------------------------------------------------------------
# Per-Step Discrepancy Collectors
# ---------------------------------------------------------------------------


def _check_step0(a0: GameState, e0: GameState) -> list[Discrepancy]:
    out: list[Discrepancy] = []
    if a0.seed and e0.seed and a0.seed.upper() != e0.seed.upper():
        out.append(
            Discrepancy(
                category="blind_hud",
                kind="initial_mismatch",
                field="seed",
                expected=e0.seed,
                actual=a0.seed,
                message=f"Seed mismatch: expected {e0.seed!r}, actual {a0.seed!r}",
                expected_step_idx=0,
            )
        )
    if a0.deck != e0.deck:
        out.append(
            Discrepancy(
                category="blind_hud",
                kind="initial_mismatch",
                field="deck",
                expected=e0.deck.value,
                actual=a0.deck.value,
                message=f"Deck mismatch: expected {e0.deck.value}, actual {a0.deck.value}",
                expected_step_idx=0,
            )
        )
    if a0.stake != e0.stake:
        out.append(
            Discrepancy(
                category="blind_hud",
                kind="initial_mismatch",
                field="stake",
                expected=e0.stake.value,
                actual=a0.stake.value,
                message=f"Stake mismatch: expected {e0.stake.value}, actual {a0.stake.value}",
                expected_step_idx=0,
            )
        )
    if a0.money is not None and e0.money is not None and a0.money != e0.money:
        out.append(
            Discrepancy(
                category="economy",
                kind="initial_mismatch",
                field="money",
                expected=e0.money,
                actual=a0.money,
                message=f"Starting money mismatch: expected ${e0.money}, actual ${a0.money}",
                expected_step_idx=0,
            )
        )
    ad = _deck_counter(a0)
    ed = _deck_counter(e0)
    if ad != ed:
        extra = _format_counter_items(ad - ed, _format_card_sig)
        missing = _format_counter_items(ed - ad, _format_card_sig)
        out.append(
            Discrepancy(
                category="deck",
                kind="initial_mismatch",
                field="full_deck",
                expected=missing or len(e0.full_deck or []),
                actual=extra or len(a0.full_deck or []),
                message=f"Starting deck mismatch: +actual={extra or []}, -missing={missing or []}",
                expected_step_idx=0,
            )
        )
    aj = _jokers_counter(a0)
    ej = _jokers_counter(e0)
    if aj != ej:
        out.append(
            Discrepancy(
                category="inventory",
                kind="initial_mismatch",
                field="jokers",
                expected=_format_counter_items(ej, _format_joker_sig),
                actual=_format_counter_items(aj, _format_joker_sig),
                message="Starting jokers mismatch",
                expected_step_idx=0,
            )
        )
    ac = _consumables_counter(a0)
    ec = _consumables_counter(e0)
    if ac != ec:
        out.append(
            Discrepancy(
                category="inventory",
                kind="initial_mismatch",
                field="consumables",
                expected=_format_counter_items(ec, _format_consumable_sig),
                actual=_format_counter_items(ac, _format_consumable_sig),
                message="Starting consumables mismatch",
                expected_step_idx=0,
            )
        )
    av = _vouchers_counter(a0)
    ev = _vouchers_counter(e0)
    if av != ev:
        out.append(
            Discrepancy(
                category="inventory",
                kind="initial_mismatch",
                field="vouchers_redeemed",
                expected=sorted(ev.elements()),
                actual=sorted(av.elements()),
                message="Starting vouchers mismatch",
                expected_step_idx=0,
            )
        )
    return out


def _check_aligned_pair(
    curr_a_step: MaterializedStep,
    prev_a_step: MaterializedStep,
    curr_e_step: MaterializedStep,
    prev_e_step: MaterializedStep,
    next_a_step: MaterializedStep | None = None,
) -> list[Discrepancy]:
    out: list[Discrepancy] = []
    curr_a, prev_a = curr_a_step.state, prev_a_step.state
    curr_e, prev_e = curr_e_step.state, prev_e_step.state
    act_a = curr_a_step.delta.action
    act_e = curr_e_step.delta.action
    exp_idx = curr_e_step.step_idx

    prev_deck_a = _deck_counter(prev_a)
    curr_deck_a = _deck_counter(curr_a)
    prev_deck_e = _deck_counter(prev_e)
    curr_deck_e = _deck_counter(curr_e)
    prev_drift_a = prev_deck_a - prev_deck_e
    prev_drift_e = prev_deck_e - prev_deck_a

    # 1. Action checks
    a_type = act_a.type if act_a is not None else None
    e_type = act_e.type if act_e is not None else None
    if a_type != e_type:
        out.append(
            Discrepancy(
                category="action",
                kind="action_mismatch",
                field="action.type",
                expected=_enum_str(e_type),
                actual=_enum_str(a_type),
                message=f"Action type mismatch: expected {_enum_str(e_type)}, actual {_enum_str(a_type)}",
                expected_step_idx=exp_idx,
            )
        )
    else:
        if a_type in _TARGET_ACTION_TYPES:
            t_a = _resolve_target_from_prev(prev_a, curr_a_step.delta)
            t_e = _resolve_target_from_prev(prev_e, curr_e_step.delta)
            na, ne = _normalize_target_name(t_a), _normalize_target_name(t_e)
            if (na or ne) and na != ne:
                out.append(
                    Discrepancy(
                        category="action",
                        kind="target_mismatch",
                        field="action.target_card",
                        expected=t_e or "(none)",
                        actual=t_a or "(none)",
                        message=f"Target mismatch on {_enum_str(a_type)}: expected {t_e or '(none)'}, actual {t_a or '(none)'}",
                        expected_step_idx=exp_idx,
                    )
                )

        if a_type in _SELECTED_ACTION_TYPES:
            sel_a = _selected_cards_counter(curr_a_step)
            sel_e = _selected_cards_counter(curr_e_step)
            if sel_a or sel_e:
                rem_a, rem_e, suppressed = _suppress_property_drift(
                    sel_a, sel_e, prev_drift_a, prev_drift_e
                )
                if rem_a or rem_e:
                    out.append(
                        Discrepancy(
                            category="action",
                            kind="selected_cards_mismatch",
                            field="action.selected_cards",
                            expected=_format_counter_items(sel_e, _format_card_sig),
                            actual=_format_counter_items(sel_a, _format_card_sig),
                            message=(
                                f"Selected cards mismatch on {_enum_str(a_type)}: "
                                f"expected {(_format_counter_items(rem_e, _format_card_sig) if rem_e else '∅')}, "
                                f"actual {(_format_counter_items(rem_a, _format_card_sig) if rem_a else '∅')}"
                            ),
                            expected_step_idx=exp_idx,
                        )
                    )
                elif suppressed > 0:
                    out.append(
                        Discrepancy(
                            category="action",
                            kind="selected_cards_drift_cascade",
                            field="action.selected_cards",
                            expected=_format_counter_items(sel_e, _format_card_sig),
                            actual=_format_counter_items(sel_a, _format_card_sig),
                            message="Selected cards differed only on pre-existing deck card properties",
                            cascaded=True,
                            expected_step_idx=exp_idx,
                        )
                    )

    # 2. Full Deck transition (per-step delta with self-correction suppression)
    has_deck_diff, deck_cascaded = _filter_collection_cascade(
        prev_deck_a,
        curr_deck_a,
        prev_deck_e,
        curr_deck_e,
        suppress_rank_suit_drift=True,
    )
    if has_deck_diff:
        a_deck_gain = curr_deck_a - prev_deck_a
        a_deck_loss = prev_deck_a - curr_deck_a
        e_deck_gain = curr_deck_e - prev_deck_e
        e_deck_loss = prev_deck_e - curr_deck_e
        out.append(
            Discrepancy(
                category="deck",
                kind="drift_reconciliation" if deck_cascaded else "transition_mismatch",
                field="full_deck_delta",
                expected=_format_deck_delta(e_deck_gain, e_deck_loss),
                actual=_format_deck_delta(a_deck_gain, a_deck_loss),
                message=(
                    f"Deck transition mismatch: expected {_format_deck_delta(e_deck_gain, e_deck_loss)}, "
                    f"actual {_format_deck_delta(a_deck_gain, a_deck_loss)}"
                ),
                cascaded=deck_cascaded,
                expected_step_idx=exp_idx,
                detail={
                    "expected_deck_size": len(curr_e.full_deck or []),
                    "actual_deck_size": len(curr_a.full_deck or []),
                },
            )
        )

    # 3. Economy transition (per-step Δmoney + Δreroll_cost with convergence & timing guards)
    if None not in (prev_a.money, curr_a.money, prev_e.money, curr_e.money):
        dm_a = int(curr_a.money) - int(prev_a.money)
        dm_e = int(curr_e.money) - int(prev_e.money)
        if dm_a != dm_e:
            interest_diff = _calc_interest(prev_a) - _calc_interest(prev_e)
            is_interest_cascade = (
                a_type in (ActionType.CASH_OUT, ActionType.END_ROUND)
                and int(prev_a.money) != int(prev_e.money)
                and (dm_a - dm_e) == interest_diff
            )
            is_self_correction = (
                int(prev_a.money) != int(prev_e.money)
                and (
                    int(curr_a.money) == int(curr_e.money)
                    or (
                        dm_e == 0
                        and abs(int(curr_a.money) - int(curr_e.money))
                        < abs(int(prev_a.money) - int(prev_e.money))
                    )
                )
            )
            next_hud_money = (
                (next_a_step.delta.action.metadata or {}).get("hud_ocr", {}).get("hud_money")
                if (next_a_step is not None and next_a_step.delta.action is not None)
                else None
            )
            is_hud_timing = (
                a_type in (ActionType.PLAY_HAND, ActionType.DISCARD, ActionType.END_ROUND)
                and next_hud_money is not None
                and int(next_hud_money) == int(curr_e.money)
            )
            is_money_cascade = is_interest_cascade or is_self_correction or is_hud_timing
            kind_str = (
                "interest_cascade"
                if is_interest_cascade
                else (
                    "self_correction"
                    if is_self_correction
                    else ("hud_timing_cascade" if is_hud_timing else "delta_mismatch")
                )
            )
            out.append(
                Discrepancy(
                    category="economy",
                    kind=kind_str,
                    field="money_delta",
                    expected=dm_e,
                    actual=dm_a,
                    message=(
                        f"Money delta mismatch: expected {_format_signed_delta(dm_e, '$')}, "
                        f"actual {_format_signed_delta(dm_a, '$')} "
                        f"(cumulative: actual=${curr_a.money} vs GT=${curr_e.money})"
                    ),
                    cascaded=is_money_cascade,
                    expected_step_idx=exp_idx,
                )
            )

    if a_type == ActionType.REROLL_SHOP:
        if None not in (prev_a.reroll_cost, curr_a.reroll_cost, prev_e.reroll_cost, curr_e.reroll_cost):
            dr_a = int(curr_a.reroll_cost) - int(prev_a.reroll_cost)
            dr_e = int(curr_e.reroll_cost) - int(prev_e.reroll_cost)
            if dr_a != dr_e:
                is_reroll_cascade = (
                    int(prev_a.reroll_cost) != int(prev_e.reroll_cost)
                    and int(curr_a.reroll_cost) == int(curr_e.reroll_cost)
                )
                out.append(
                    Discrepancy(
                        category="economy",
                        kind="self_correction" if is_reroll_cascade else "delta_mismatch",
                        field="reroll_cost_delta",
                        expected=dr_e,
                        actual=dr_a,
                        message=(
                            f"Reroll cost delta mismatch: expected {_format_signed_delta(dr_e, '$')}, "
                            f"actual {_format_signed_delta(dr_a, '$')}"
                        ),
                        cascaded=is_reroll_cascade,
                        expected_step_idx=exp_idx,
                    )
                )

    # 4. Inventory transitions (jokers, consumables, vouchers, tags)
    aj_prev, aj_curr = _jokers_counter(prev_a), _jokers_counter(curr_a)
    ej_prev, ej_curr = _jokers_counter(prev_e), _jokers_counter(curr_e)
    has_j_diff, j_cascaded = _filter_collection_cascade(aj_prev, aj_curr, ej_prev, ej_curr)
    if has_j_diff:
        aj_gain, aj_loss = aj_curr - aj_prev, aj_prev - aj_curr
        ej_gain, ej_loss = ej_curr - ej_prev, ej_prev - ej_curr
        exp_desc = (
            f"+{_format_counter_items(ej_gain, _format_joker_sig)} "
            f"-{_format_counter_items(ej_loss, _format_joker_sig)}"
        )
        act_desc = (
            f"+{_format_counter_items(aj_gain, _format_joker_sig)} "
            f"-{_format_counter_items(aj_loss, _format_joker_sig)}"
        )
        out.append(
            Discrepancy(
                category="inventory",
                kind="drift_reconciliation" if j_cascaded else "transition_mismatch",
                field="jokers_delta",
                expected=exp_desc,
                actual=act_desc,
                message=f"Jokers transition mismatch: expected {exp_desc}, actual {act_desc}",
                cascaded=j_cascaded,
                expected_step_idx=exp_idx,
            )
        )

    ac_prev, ac_curr = _consumables_counter(prev_a), _consumables_counter(curr_a)
    ec_prev, ec_curr = _consumables_counter(prev_e), _consumables_counter(curr_e)
    has_c_diff, c_cascaded = _filter_collection_cascade(ac_prev, ac_curr, ec_prev, ec_curr)
    if has_c_diff:
        ac_gain, ac_loss = ac_curr - ac_prev, ac_prev - ac_curr
        ec_gain, ec_loss = ec_curr - ec_prev, ec_prev - ec_curr
        exp_desc = (
            f"+{_format_counter_items(ec_gain, _format_consumable_sig)} "
            f"-{_format_counter_items(ec_loss, _format_consumable_sig)}"
        )
        act_desc = (
            f"+{_format_counter_items(ac_gain, _format_consumable_sig)} "
            f"-{_format_counter_items(ac_loss, _format_consumable_sig)}"
        )
        out.append(
            Discrepancy(
                category="inventory",
                kind="drift_reconciliation" if c_cascaded else "transition_mismatch",
                field="consumables_delta",
                expected=exp_desc,
                actual=act_desc,
                message=f"Consumables transition mismatch: expected {exp_desc}, actual {act_desc}",
                cascaded=c_cascaded,
                expected_step_idx=exp_idx,
            )
        )

    av_prev, av_curr = _vouchers_counter(prev_a), _vouchers_counter(curr_a)
    ev_prev, ev_curr = _vouchers_counter(prev_e), _vouchers_counter(curr_e)
    has_v_diff, v_cascaded = _filter_collection_cascade(av_prev, av_curr, ev_prev, ev_curr)
    if has_v_diff:
        av_gain, av_loss = av_curr - av_prev, av_prev - av_curr
        ev_gain, ev_loss = ev_curr - ev_prev, ev_prev - ev_curr
        out.append(
            Discrepancy(
                category="inventory",
                kind="drift_reconciliation" if v_cascaded else "transition_mismatch",
                field="vouchers_delta",
                expected=f"+{sorted(ev_gain.elements())} -{sorted(ev_loss.elements())}",
                actual=f"+{sorted(av_gain.elements())} -{sorted(av_loss.elements())}",
                message=(
                    f"Vouchers transition mismatch: "
                    f"expected +{sorted(ev_gain.elements())}, actual +{sorted(av_gain.elements())}"
                ),
                cascaded=v_cascaded,
                expected_step_idx=exp_idx,
            )
        )

    at_prev, at_curr = _tags_counter(prev_a), _tags_counter(curr_a)
    et_prev, et_curr = _tags_counter(prev_e), _tags_counter(curr_e)
    has_t_diff, t_cascaded = _filter_collection_cascade(at_prev, at_curr, et_prev, et_curr)
    if has_t_diff:
        at_gain, at_loss = at_curr - at_prev, at_prev - at_curr
        et_gain, et_loss = et_curr - et_prev, et_prev - et_curr
        out.append(
            Discrepancy(
                category="inventory",
                kind="drift_reconciliation" if t_cascaded else "transition_mismatch",
                field="active_tags_delta",
                expected=f"+{sorted(et_gain.elements())} -{sorted(et_loss.elements())}",
                actual=f"+{sorted(at_gain.elements())} -{sorted(at_loss.elements())}",
                message=(
                    f"Skip tags transition mismatch: "
                    f"expected +{sorted(et_gain.elements())} -{sorted(et_loss.elements())}, "
                    f"actual +{sorted(at_gain.elements())} -{sorted(at_loss.elements())}"
                ),
                cascaded=t_cascaded,
                expected_step_idx=exp_idx,
            )
        )

    # 5. Poker hand levels & play counts transitions (suppress self-correction convergence)
    changed_hands_primary: dict[str, dict[str, tuple[int, int]]] = {}
    changed_hands_cascaded: dict[str, dict[str, tuple[int, int]]] = {}
    for ph in PokerHand:
        pl_a, cl_a = prev_a.hand_levels.get(ph, 1), curr_a.hand_levels.get(ph, 1)
        pl_e, cl_e = prev_e.hand_levels.get(ph, 1), curr_e.hand_levels.get(ph, 1)
        pp_a, cp_a = prev_a.hand_play_counts.get(ph, 0), curr_a.hand_play_counts.get(ph, 0)
        pp_e, cp_e = prev_e.hand_play_counts.get(ph, 0), curr_e.hand_play_counts.get(ph, 0)
        dl_a, dl_e = cl_a - pl_a, cl_e - pl_e
        dp_a, dp_e = cp_a - pp_a, cp_e - pp_e
        entry_pri: dict[str, tuple[int, int]] = {}
        entry_cas: dict[str, tuple[int, int]] = {}
        if dl_a != dl_e:
            if pl_a != pl_e and cl_a == cl_e:
                entry_cas["level_delta"] = (dl_e, dl_a)
            else:
                entry_pri["level_delta"] = (dl_e, dl_a)
        if dp_a != dp_e:
            if pp_a != pp_e and cp_a == cp_e:
                entry_cas["play_count_delta"] = (dp_e, dp_a)
            else:
                entry_pri["play_count_delta"] = (dp_e, dp_a)
        if entry_pri:
            changed_hands_primary[ph.value] = entry_pri
        elif entry_cas:
            changed_hands_cascaded[ph.value] = entry_cas

    if changed_hands_primary or changed_hands_cascaded:
        chosen = changed_hands_primary if changed_hands_primary else changed_hands_cascaded
        is_ph_cascaded = not changed_hands_primary
        summary_parts = [
            f"{hn}: "
            + ", ".join(
                f"{k} exp={v[0]:+d} act={v[1]:+d}" for k, v in hdiffs.items()
            )
            for hn, hdiffs in chosen.items()
        ]
        out.append(
            Discrepancy(
                category="poker_hands",
                kind="self_correction" if is_ph_cascaded else "delta_mismatch",
                field="poker_hands_delta",
                expected={h: {k: v[0] for k, v in d.items()} for h, d in chosen.items()},
                actual={h: {k: v[1] for k, v in d.items()} for h, d in chosen.items()},
                message="Poker hand delta mismatch: " + "; ".join(summary_parts),
                cascaded=is_ph_cascaded,
                expected_step_idx=exp_idx,
            )
        )

    # 6. Blind & HUD transitions
    d_ante_a = curr_a.ante - prev_a.ante
    d_ante_e = curr_e.ante - prev_e.ante
    if d_ante_a != d_ante_e:
        is_ante_casc = prev_a.ante != prev_e.ante and curr_a.ante == curr_e.ante
        out.append(
            Discrepancy(
                category="blind_hud",
                kind="self_correction" if is_ante_casc else "delta_mismatch",
                field="ante_delta",
                expected=d_ante_e,
                actual=d_ante_a,
                message=f"Ante delta mismatch: expected {d_ante_e:+d}, actual {d_ante_a:+d} (actual={curr_a.ante}, GT={curr_e.ante})",
                cascaded=is_ante_casc,
                expected_step_idx=exp_idx,
            )
        )
    d_round_a = curr_a.round - prev_a.round
    d_round_e = curr_e.round - prev_e.round
    if d_round_a != d_round_e:
        is_round_casc = prev_a.round != prev_e.round and curr_a.round == curr_e.round
        out.append(
            Discrepancy(
                category="blind_hud",
                kind="self_correction" if is_round_casc else "delta_mismatch",
                field="round_delta",
                expected=d_round_e,
                actual=d_round_a,
                message=f"Round delta mismatch: expected {d_round_e:+d}, actual {d_round_a:+d} (actual={curr_a.round}, GT={curr_e.round})",
                cascaded=is_round_casc,
                expected_step_idx=exp_idx,
            )
        )

    if curr_a.location != curr_e.location and (
        prev_a.location == prev_e.location
        or curr_a.location != prev_a.location
        or curr_e.location != prev_e.location
    ):
        out.append(
            Discrepancy(
                category="blind_hud",
                kind="transition_mismatch",
                field="location",
                expected=curr_e.location.value,
                actual=curr_a.location.value,
                message=f"Location mismatch: expected {curr_e.location.value}, actual {curr_a.location.value}",
                expected_step_idx=exp_idx,
            )
        )

    if (
        curr_a.current_blind != prev_a.current_blind
        or curr_e.current_blind != prev_e.current_blind
        or a_type in (ActionType.SELECT_BLIND, ActionType.SKIP_BLIND)
    ) and curr_a.current_blind != curr_e.current_blind:
        out.append(
            Discrepancy(
                category="blind_hud",
                kind="transition_mismatch",
                field="current_blind",
                expected=curr_e.current_blind.value,
                actual=curr_a.current_blind.value,
                message=f"Current blind mismatch: expected {curr_e.current_blind.value}, actual {curr_a.current_blind.value}",
                expected_step_idx=exp_idx,
            )
        )

    if (curr_a.boss_blind != prev_a.boss_blind or curr_e.boss_blind != prev_e.boss_blind) and (
        curr_a.boss_blind != curr_e.boss_blind
    ):
        if curr_a.boss_blind is not None or curr_e.boss_blind is not None:
            out.append(
                Discrepancy(
                    category="blind_hud",
                    kind="transition_mismatch",
                    field="boss_blind",
                    expected=_enum_str(curr_e.boss_blind) or None,
                    actual=_enum_str(curr_a.boss_blind) or None,
                    message=(
                        f"Boss blind mismatch: expected {_enum_str(curr_e.boss_blind) or 'None'}, "
                        f"actual {_enum_str(curr_a.boss_blind) or 'None'}"
                    ),
                    expected_step_idx=exp_idx,
                )
            )

    if (curr_a.location == Location.BLIND or curr_e.location == Location.BLIND) and (
        curr_a.target_chips != prev_a.target_chips or curr_e.target_chips != prev_e.target_chips
    ) and curr_a.target_chips != curr_e.target_chips:
        out.append(
            Discrepancy(
                category="blind_hud",
                kind="transition_mismatch",
                field="target_chips",
                expected=curr_e.target_chips,
                actual=curr_a.target_chips,
                message=f"Blind target_chips mismatch: expected {curr_e.target_chips}, actual {curr_a.target_chips}",
                expected_step_idx=exp_idx,
            )
        )

    if (
        prev_a.location == Location.BLIND
        and curr_a.location == Location.BLIND
        and prev_e.location == Location.BLIND
        and curr_e.location == Location.BLIND
        and a_type not in (ActionType.END_ROUND, ActionType.CASH_OUT)
        and e_type not in (ActionType.END_ROUND, ActionType.CASH_OUT)
    ):
        if None not in (prev_a.hands_left, curr_a.hands_left, prev_e.hands_left, curr_e.hands_left):
            dh_a = int(curr_a.hands_left) - int(prev_a.hands_left)
            dh_e = int(curr_e.hands_left) - int(prev_e.hands_left)
            if dh_a != dh_e:
                is_h_casc = int(prev_a.hands_left) != int(prev_e.hands_left) and int(curr_a.hands_left) == int(curr_e.hands_left)
                out.append(
                    Discrepancy(
                        category="blind_hud",
                        kind="self_correction" if is_h_casc else "delta_mismatch",
                        field="hands_left_delta",
                        expected=dh_e,
                        actual=dh_a,
                        message=f"Hands left delta mismatch: expected {dh_e:+d}, actual {dh_a:+d}",
                        cascaded=is_h_casc,
                        expected_step_idx=exp_idx,
                    )
                )
        if None not in (prev_a.discards_left, curr_a.discards_left, prev_e.discards_left, curr_e.discards_left):
            dd_a = int(curr_a.discards_left) - int(prev_a.discards_left)
            dd_e = int(curr_e.discards_left) - int(prev_e.discards_left)
            if dd_a != dd_e:
                is_d_casc = int(prev_a.discards_left) != int(prev_e.discards_left) and int(curr_a.discards_left) == int(curr_e.discards_left)
                out.append(
                    Discrepancy(
                        category="blind_hud",
                        kind="self_correction" if is_d_casc else "delta_mismatch",
                        field="discards_left_delta",
                        expected=dd_e,
                        actual=dd_a,
                        message=f"Discards left delta mismatch: expected {dd_e:+d}, actual {dd_a:+d}",
                        cascaded=is_d_casc,
                        expected_step_idx=exp_idx,
                    )
                )

    # 7. Visible hand comparison (with deck property drift suppression)
    if (
        a_type in _HAND_CHECK_ACTION_TYPES
        and curr_a.location in (Location.BLIND, Location.PACK)
        and curr_e.location in (Location.BLIND, Location.PACK)
    ):
        hand_a = _visible_hand_counter(curr_a)
        hand_e = _visible_hand_counter(curr_e)
        if hand_a or hand_e:
            rem_ha, rem_he, suppressed_h = _suppress_property_drift(
                hand_a, hand_e, prev_drift_a, prev_drift_e
            )
            if rem_ha or rem_he:
                exp_list = _format_counter_items(rem_he, _format_card_sig)
                act_list = _format_counter_items(rem_ha, _format_card_sig)
                out.append(
                    Discrepancy(
                        category="hand",
                        kind="hand_mismatch",
                        field="hand",
                        expected=exp_list,
                        actual=act_list,
                        message=(
                            f"Visible hand mismatch: expected +[{', '.join(exp_list)}], "
                            f"actual +[{', '.join(act_list)}]"
                        ),
                        expected_step_idx=exp_idx,
                    )
                )
            elif suppressed_h > 0:
                out.append(
                    Discrepancy(
                        category="hand",
                        kind="hand_drift_cascade",
                        field="hand",
                        expected=[],
                        actual=[],
                        message="Hand differed only on pre-existing deck card properties",
                        cascaded=True,
                        expected_step_idx=exp_idx,
                    )
                )

    # 8. Dynamic round joker targets (only when changed AND corresponding joker is owned)
    joker_names = {j.name for j in (curr_a.inventory.jokers if curr_a.inventory else [])} | {
        j.name for j in (curr_e.inventory.jokers if curr_e.inventory else [])
    }
    for jname, attr, label in _TARGET_JOKER_FIELDS:
        if jname not in joker_names:
            continue
        pa_val, ca_val = getattr(prev_a, attr), getattr(curr_a, attr)
        pe_val, ce_val = getattr(prev_e, attr), getattr(curr_e, attr)
        if (ca_val != pa_val or ce_val != pe_val) and ca_val != ce_val:
            fmt = lambda v: (
                f"{_enum_str(v[0])} of {_enum_str(v[1])}" if isinstance(v, tuple) else _enum_str(v)
            )
            out.append(
                Discrepancy(
                    category="targets",
                    kind="target_joker_mismatch",
                    field=attr,
                    expected=fmt(ce_val),
                    actual=fmt(ca_val),
                    message=f"{label} mismatch: expected {fmt(ce_val)}, actual {fmt(ca_val)}",
                    expected_step_idx=exp_idx,
                )
            )

    # 9. Observed Shop / Pack shelves (checked on entry / reroll or new items only)
    obs_shop_a = _extract_observed_shop_counter(curr_a_step)
    obs_shop_e = _extract_observed_shop_counter(curr_e_step)
    if obs_shop_a is not None and obs_shop_e is not None and a_type != ActionType.NEXT_ROUND:
        prev_obs_a = _extract_observed_shop_counter(prev_a_step) if prev_a.location == Location.SHOP else None
        prev_obs_e = _extract_observed_shop_counter(prev_e_step) if prev_e.location == Location.SHOP else None
        prev_was_reroll = prev_a_step.delta.action is not None and prev_a_step.delta.action.type == ActionType.REROLL_SHOP
        if prev_obs_a is None or prev_obs_e is None or prev_was_reroll:
            new_a = obs_shop_a - obs_shop_e
            new_e = obs_shop_e - obs_shop_a
        else:
            # Mid-shop: ignore items newly spotted by CV that match what GT already had in shop
            new_a = (obs_shop_a - prev_obs_a) - obs_shop_e
            new_e = (obs_shop_e - prev_obs_e) - obs_shop_a
        if new_a or new_e:
            out.append(
                Discrepancy(
                    category="shop_pack",
                    kind="observed_shop_mismatch",
                    field="observed_shop",
                    expected=_format_counter_items(new_e, _format_shop_sig),
                    actual=_format_counter_items(new_a, _format_shop_sig),
                    message=(
                        f"Observed shop mismatch: expected +{_format_counter_items(new_e, _format_shop_sig)}, "
                        f"actual +{_format_counter_items(new_a, _format_shop_sig)}"
                    ),
                    expected_step_idx=exp_idx,
                )
            )

    obs_pack_a = _extract_observed_pack_counter(curr_a_step)
    obs_pack_e = _extract_observed_pack_counter(curr_e_step)
    if obs_pack_a is not None and obs_pack_e is not None:
        prev_pack_a = _extract_observed_pack_counter(prev_a_step) if prev_a.location == Location.PACK else None
        prev_pack_e = _extract_observed_pack_counter(prev_e_step) if prev_e.location == Location.PACK else None
        if prev_pack_a is None or prev_pack_e is None:
            new_pa, new_pe = obs_pack_a, obs_pack_e
        else:
            new_pa = obs_pack_a - prev_pack_a
            new_pe = obs_pack_e - prev_pack_e
        if new_pa != new_pe and (new_pa or new_pe):
            out.append(
                Discrepancy(
                    category="shop_pack",
                    kind="observed_pack_mismatch",
                    field="observed_pack",
                    expected=_format_counter_items(new_pe - new_pa, _format_shop_sig),
                    actual=_format_counter_items(new_pa - new_pe, _format_shop_sig),
                    message=(
                        f"Observed pack mismatch: expected +{_format_counter_items(new_pe - new_pa, _format_shop_sig)}, "
                        f"actual +{_format_counter_items(new_pa - new_pe, _format_shop_sig)}"
                    ),
                    expected_step_idx=exp_idx,
                )
            )

    return out


# ---------------------------------------------------------------------------
# Public Comparison Entry Point
# ---------------------------------------------------------------------------


def compare_trajectories(
    actual: RunTrajectory | dict[str, Any] | str | Path,
    expected: RunTrajectory | dict[str, Any] | str | Path,
    *,
    seed: str | None = None,
    run_index: int | None = None,
    ignore_benign_sorts: bool = True,
    include_cascaded: bool = False,
    keep_all_sorts: bool = False,
    inject_end_round: bool = True,
    reconcile_truth: bool = True,
) -> dict[str, Any]:
    """Compares `actual` run trajectory against `expected` GT trajectory/telemetry without cascading."""
    actual_seed_hint = _extract_seed_hint(actual)
    traj_a, meta_a = load_trajectory_source(actual)
    traj_e, meta_e = load_trajectory_source(
        expected,
        seed=seed,
        actual_seed_hint=actual_seed_hint if seed is None else None,
        run_index=run_index,
        keep_all_sorts=keep_all_sorts,
        inject_end_round=inject_end_round,
        reconcile_truth=reconcile_truth,
    )

    steps_a = _materialize_steps(traj_a)
    steps_e = _materialize_steps(traj_e)
    feats_a = _build_features(steps_a, ignore_benign_sorts=ignore_benign_sorts)
    feats_e = _build_features(steps_e, ignore_benign_sorts=ignore_benign_sorts)
    ops = _align_features(feats_a, feats_e)

    discrepancies_by_step: dict[int, list[Discrepancy]] = {}
    missing_actions = 0
    extra_actions = 0
    aligned_pairs = 0
    suppressed_cascades = 0
    last_actual_idx = 0

    def _add_disc(actual_idx: int, d: Discrepancy) -> None:
        nonlocal suppressed_cascades
        if d.cascaded and not include_cascaded:
            suppressed_cascades += 1
            return
        discrepancies_by_step.setdefault(actual_idx, []).append(d)

    for kind, idx_a, idx_e in ops:
        if kind == "match" and idx_a is not None and idx_e is not None:
            aligned_pairs += 1
            last_actual_idx = idx_a
            if idx_a == 0 and idx_e == 0:
                for d in _check_step0(steps_a[0].state, steps_e[0].state):
                    _add_disc(0, d)
            else:
                prev_a = steps_a[max(0, idx_a - 1)]
                prev_e = steps_e[max(0, idx_e - 1)]
                next_a = steps_a[idx_a + 1] if idx_a + 1 < len(steps_a) else None
                for d in _check_aligned_pair(
                    steps_a[idx_a], prev_a, steps_e[idx_e], prev_e, next_a
                ):
                    _add_disc(idx_a, d)
        elif kind == "extra_in_actual" and idx_a is not None:
            extra_actions += 1
            last_actual_idx = idx_a
            prev_a = steps_a[max(0, idx_a - 1)]
            act_summary = _summarize_action(steps_a[idx_a], prev_a.state)
            _add_disc(
                idx_a,
                Discrepancy(
                    category="action",
                    kind="extra_action",
                    field="action",
                    expected=None,
                    actual=act_summary,
                    message=f"Extra action not present in ground truth: {act_summary}",
                ),
            )
        elif kind == "missing_in_actual" and idx_e is not None:
            missing_actions += 1
            prev_e = steps_e[max(0, idx_e - 1)]
            gt_summary = _summarize_action(steps_e[idx_e], prev_e.state)
            _add_disc(
                last_actual_idx,
                Discrepancy(
                    category="action",
                    kind="missing_action",
                    field="action",
                    expected=gt_summary,
                    actual=None,
                    message=f"Missing ground-truth action after step {last_actual_idx} (GT step {idx_e}): {gt_summary}",
                    expected_step_idx=idx_e,
                    detail={"after_actual_step_idx": last_actual_idx},
                ),
            )

    by_category: dict[str, int] = {}
    errors: list[dict[str, Any]] = []
    steps_by_actual_idx: dict[str, list[dict[str, Any]]] = {}
    total_discrepancies = 0

    for step_idx in sorted(discrepancies_by_step.keys()):
        d_list = discrepancies_by_step[step_idx]
        if not d_list:
            continue
        step_obj = steps_a[step_idx]
        st = step_obj.state
        act = step_obj.delta.action
        ser_list: list[dict[str, Any]] = []
        cats: list[str] = []
        exp_step_idx: int | None = None
        for d in d_list:
            by_category[d.category] = by_category.get(d.category, 0) + 1
            total_discrepancies += 1
            if d.category not in cats:
                cats.append(d.category)
            if exp_step_idx is None and d.expected_step_idx is not None:
                exp_step_idx = d.expected_step_idx
            ser_list.append(d.to_dict())

        steps_by_actual_idx[str(step_idx)] = ser_list
        errors.append(
            {
                "actual_step_idx": step_idx,
                "expected_step_idx": exp_step_idx,
                "frame_idx": int(st.frame_idx),
                "timestamp": float(st.timestamp),
                "action_type": _enum_str(act.type) if act is not None else "Init",
                "location": st.location.value,
                "ante": int(st.ante),
                "round": int(st.round),
                "categories": cats,
                "discrepancies": ser_list,
            }
        )

    warnings: list[str] = []
    if meta_e.get("seed_mismatch"):
        warnings.append(
            f"No telemetry run matched actual seed {actual_seed_hint!r}; "
            f"selected run with seed {meta_e.get('seed')!r}."
        )
    if meta_e.get("parity_warnings"):
        warnings.append(
            f"Telemetry replay noted {len(meta_e['parity_warnings'])} internal parity adjustment(s)."
        )

    return {
        "actual_meta": meta_a,
        "expected_meta": meta_e,
        "warnings": warnings,
        "summary": {
            "total_actual_steps": len(steps_a),
            "total_expected_steps": len(steps_e),
            "aligned_pairs": aligned_pairs,
            "missing_actions": missing_actions,
            "extra_actions": extra_actions,
            "error_step_count": len(errors),
            "total_discrepancies": total_discrepancies,
            "suppressed_cascades": suppressed_cascades,
            "by_category": by_category,
            "error_step_indices": [e["actual_step_idx"] for e in errors],
        },
        "errors": errors,
        "steps_by_actual_idx": steps_by_actual_idx,
    }


def format_report_text(report: dict[str, Any], *, verbose: bool = False) -> str:
    """Formats a `compare_trajectories` report into a clean human-readable CLI summary."""
    am = report["actual_meta"]
    em = report["expected_meta"]
    sm = report["summary"]
    lines = [
        f"Actual:   seed={am.get('seed') or '—'} | {am.get('deck')} | {am.get('stake')} | {sm['total_actual_steps']} steps",
        f"Expected: seed={em.get('seed') or '—'} | {em.get('deck')} | {em.get('stake')} | {sm['total_expected_steps']} steps ({em.get('source_kind')})",
        (
            f"Alignment: {sm['aligned_pairs']} paired, {sm['missing_actions']} missing GT, "
            f"{sm['extra_actions']} extra actual | Suppressed cascades: {sm['suppressed_cascades']}"
        ),
        (
            f"Result: {sm['error_step_count']} error step(s), "
            f"{sm['total_discrepancies']} discrepancy(ies) "
            f"{sm['by_category'] if sm['by_category'] else ''}"
        ).strip(),
    ]
    for w in report.get("warnings", ()):
        lines.append(f"Warning: {w}")
    if report["errors"]:
        lines.append("")
        for err in report["errors"]:
            gt_s = f" (GT step {err['expected_step_idx']})" if err["expected_step_idx"] is not None else ""
            lines.append(
                f"[Step {err['actual_step_idx']}{gt_s} | f={err['frame_idx']} | "
                f"A{err['ante']}R{err['round']} {err['location']} | {err['action_type']}]"
            )
            for d in err["discrepancies"]:
                tag = f" ({d['category']})"
                casc = " [cascaded]" if d.get("cascaded") else ""
                lines.append(f"  - {d['message']}{tag}{casc}")
                if verbose:
                    lines.append(f"      field={d['field']} expected={d['expected']!r} actual={d['actual']!r}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare a scraped Balatro trajectory against Lua telemetry or a GT trajectory without cascading errors."
    )
    parser.add_argument("--actual", "-a", type=Path, required=True, help="Scraped *_trajectory.json to test")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--telemetry", "-t", type=Path, help="Lua run_telemetry.jsonl file")
    group.add_argument("--expected", "-e", type=Path, help="Pre-converted ground-truth *_trajectory.json")
    parser.add_argument("--seed", type=str, default=None, help="Select telemetry run matching this seed")
    parser.add_argument("--run-index", type=int, default=None, help="0-based telemetry run segment index")
    parser.add_argument("--keep-all-sorts", action="store_true", help="Do not filter benign Sort actions")
    parser.add_argument("--no-inject-end-round", action="store_true", help="Do not inject END_ROUND on telemetry replay")
    parser.add_argument("--strict-sim", action="store_true", help="Do not reconcile telemetry RNG/hidden state")
    parser.add_argument("--include-cascaded", action="store_true", help="Include suppressed secondary cascades in output")
    parser.add_argument("--json-out", type=Path, default=None, help="Write structured JSON comparison report")
    parser.add_argument("--no-fail", action="store_true", help="Always exit 0 even when discrepancies exist")
    parser.add_argument("--verbose", action="store_true", help="Print detailed field values")
    args = parser.parse_args(argv)

    gt_source = args.telemetry if args.telemetry is not None else args.expected
    report = compare_trajectories(
        args.actual,
        gt_source,
        seed=args.seed,
        run_index=args.run_index,
        ignore_benign_sorts=not args.keep_all_sorts,
        include_cascaded=args.include_cascaded,
        keep_all_sorts=args.keep_all_sorts,
        inject_end_round=not args.no_inject_end_round,
        reconcile_truth=not args.strict_sim,
    )
    print(format_report_text(report, verbose=args.verbose))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote JSON report: {args.json_out}")
    if report["summary"]["error_step_count"] > 0 and not args.no_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
