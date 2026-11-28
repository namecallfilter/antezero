"""Interactive Web UI + API Server for inspecting Balatro run trajectories.

Reconstructs full step-by-step GameState snapshots from compressed trajectory
JSONs (`output/trajectories/*_trajectory.json`), computes diffs from each previous
action, pairs each step with its matched VOD video frame via an LRU frame cache,
and serves a responsive single-screen inspection studio using Balatro's 2x
sprite sheets (`balatro_source/resources/textures/2x/`).

Usage:
    python tools/visualize_trajectory.py
    python tools/visualize_trajectory.py --trajectory output/trajectories/...json
    python tools/visualize_trajectory.py --trajectory ...json --video ...mp4 --port 8765
"""

from __future__ import annotations

import argparse
import copy
import json
import mimetypes
import os
import re
import sys
import threading
import uuid
import webbrowser
from collections import OrderedDict
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import parse_qs, unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from game.balatro_models import (
    Action,
    ActionType,
    BlindName,
    BoosterPack,
    BoosterSize,
    BoosterType,
    Card,
    CardType,
    Deck,
    Edition,
    Enhancement,
    HandCard,
    Joker,
    JokerName,
    Location,
    POKER_HAND_DEFAULTS,
    PlayingCard,
    PokerHand,
    Rank,
    Seal,
    SkipTag,
    Stake,
    Suit,
    Voucher,
    VoucherName,
)
from game.gamestate import GameState, RunTrajectory, _cards_match
from game.trajectory_codec import (
    _deserialize_booster_pack,
    _deserialize_card,
    _deserialize_voucher,
    _serialize_action,
    _serialize_booster_pack,
    _serialize_card,
    _serialize_hand_card,
    _serialize_joker,
    _serialize_playing_card,
    _serialize_voucher,
    rebuild_trajectory,
)

DEFAULT_TRAJECTORIES_DIR = REPO_ROOT / "output" / "trajectories"
DEFAULT_VIDEOS_DIR = REPO_ROOT / "assets" / "sample_inputs" / "videos"
DEFAULT_TELEMETRY_DIRS = (
    REPO_ROOT / "assets" / "local",
    REPO_ROOT / "testing" / "telemetry_data",
)
TEXTURES_2X_DIR = REPO_ROOT / "balatro_source" / "resources" / "textures" / "2x"
GAME_LUA_PATH = REPO_ROOT / "balatro_source" / "game.lua"

VIDEO_EXTENSIONS = (".mp4", ".mkv", ".mov", ".avi", ".webm")
_VIDEO_STREAM_CHUNK_BYTES = 262144

# Verified 2x texture sheet dimensions matching balatro_source/resources/textures/2x/*.png
SPRITE_SHEETS: dict[str, dict[str, int]] = {
    "Jokers.png": {"cols": 10, "rows": 16, "tw": 142, "th": 190, "w": 1420, "h": 3040},
    "Tarots.png": {"cols": 10, "rows": 6, "tw": 142, "th": 190, "w": 1420, "h": 1140},
    "Vouchers.png": {"cols": 9, "rows": 4, "tw": 142, "th": 190, "w": 1278, "h": 760},
    "boosters.png": {"cols": 4, "rows": 9, "tw": 142, "th": 190, "w": 568, "h": 1710},
    "8BitDeck.png": {"cols": 13, "rows": 4, "tw": 142, "th": 190, "w": 1846, "h": 760},
    "Enhancers.png": {"cols": 7, "rows": 5, "tw": 142, "th": 190, "w": 994, "h": 950},
    "stickers.png": {"cols": 5, "rows": 3, "tw": 142, "th": 190, "w": 710, "h": 570},
    "tags.png": {"cols": 6, "rows": 5, "tw": 68, "th": 68, "w": 408, "h": 340},
    "BlindChips.png": {
        "cols": 21,
        "rows": 31,
        "tw": 68,
        "th": 68,
        "w": 1428,
        "h": 2108,
    },
    "chips.png": {"cols": 5, "rows": 2, "tw": 58, "th": 58, "w": 290, "h": 116},
}

_NAME_RE = re.compile(r"""\bname\s*=\s*(?:"([^"]*)"|'([^']*)')""")
_SET_RE = re.compile(r"""\bset\s*=\s*(?:"([^"]*)"|'([^']*)')""")
_KIND_RE = re.compile(r"""\bkind\s*=\s*(?:"([^"]*)"|'([^']*)')""")
_VALUE_RE = re.compile(r"""\bvalue\s*=\s*(?:"([^"]*)"|'([^']*)')""")
_SUIT_RE = re.compile(r"""\bsuit\s*=\s*(?:"([^"]*)"|'([^']*)')""")
_POS_RE = re.compile(r"""\bpos\s*=\s*\{\s*x\s*=\s*(\d+)\s*,\s*y\s*=\s*(\d+)\s*\}""")
_SOUL_POS_RE = re.compile(
    r"""\bsoul_pos\s*=\s*\{\s*x\s*=\s*(\d+)\s*,\s*y\s*=\s*(\d+)\s*\}"""
)
_HEX_RE = re.compile(r"""\bboss_colour\s*=\s*HEX\(\s*['"]([0-9a-fA-F]+)['"]\s*\)""")
_ENTRY_START_RE = re.compile(r"""([A-Za-z0-9_]+)\s*=\s*\{""")


def _first_group(match: re.Match[str] | None) -> str | None:
    if not match:
        return None
    return match.group(1) if match.group(1) is not None else match.group(2)


def _strip_lua_comments(text: str) -> str:
    """Removes single-line Lua `--` comments while respecting string literals."""
    out: list[str] = []
    i = 0
    n = len(text)
    in_quote: str | None = None
    while i < n:
        ch = text[i]
        if in_quote:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == in_quote:
                in_quote = None
            i += 1
        elif ch in ('"', "'"):
            in_quote = ch
            out.append(ch)
            i += 1
        elif ch == "-" and i + 1 < n and text[i + 1] == "-":
            while i < n and text[i] != "\n":
                i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _extract_lua_table_body(lua_text: str, marker_regex: str) -> str:
    """Extracts the brace-delimited body for a `self.P_* = { ... }` table."""
    m = re.search(marker_regex, lua_text)
    if not m:
        raise ValueError(f"Could not find Lua table matching {marker_regex!r}")
    start_brace = lua_text.find("{", m.start())
    if start_brace < 0:
        raise ValueError(f"Malformed Lua table for {marker_regex!r}")
    depth = 0
    in_quote: str | None = None
    i = start_brace
    n = len(lua_text)
    while i < n:
        ch = lua_text[i]
        if in_quote:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == in_quote:
                in_quote = None
        else:
            if ch in ('"', "'"):
                in_quote = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return lua_text[start_brace + 1 : i]
        i += 1
    raise ValueError(f"Unclosed Lua table for {marker_regex!r}")


def _iter_lua_entries(table_body: str) -> list[tuple[str, str]]:
    """Iterates top-level `key = { ... }` entries inside a Lua table body."""
    entries: list[tuple[str, str]] = []
    i = 0
    n = len(table_body)
    in_quote: str | None = None
    while i < n:
        ch = table_body[i]
        if in_quote:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == in_quote:
                in_quote = None
            i += 1
            continue
        if ch in ('"', "'"):
            in_quote = ch
            i += 1
            continue
        if ch.isalpha() or ch == "_":
            m = _ENTRY_START_RE.match(table_body, i)
            if m:
                key = m.group(1)
                brace_pos = m.end() - 1
                depth = 1
                j = brace_pos + 1
                q: str | None = None
                while j < n and depth > 0:
                    c2 = table_body[j]
                    if q:
                        if c2 == "\\" and j + 1 < n:
                            j += 2
                            continue
                        if c2 == q:
                            q = None
                    else:
                        if c2 in ('"', "'"):
                            q = c2
                        elif c2 == "{":
                            depth += 1
                        elif c2 == "}":
                            depth -= 1
                    j += 1
                entries.append((key, table_body[brace_pos + 1 : j - 1]))
                i = j
                continue
        i += 1
    return entries


def _parse_pos(
    entry_body: str, pattern: re.Pattern[str] = _POS_RE
) -> tuple[int, int] | None:
    m = pattern.search(entry_body)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _normalize_enhancement_name(raw_name: str) -> str:
    clean = raw_name.strip().lower().removesuffix(" card").strip()
    return clean


@lru_cache(maxsize=1)
def _build_sprite_manifest_cached() -> dict[str, Any]:
    """Parses Balatro's `game.lua` once and constructs sprite lookup tables."""
    raw_lua = GAME_LUA_PATH.read_text(encoding="utf-8")
    clean_lua = _strip_lua_comments(raw_lua)

    jokers: dict[str, dict[str, Any]] = {}
    tarots: dict[str, dict[str, Any]] = {}
    planets: dict[str, dict[str, Any]] = {}
    spectrals: dict[str, dict[str, Any]] = {}
    vouchers: dict[str, dict[str, Any]] = {}
    backs: dict[str, dict[str, Any]] = {}
    boosters: dict[str, dict[str, Any]] = {}
    enhancements: dict[str, dict[str, Any]] = {}
    tags: dict[str, dict[str, Any]] = {}
    blinds: dict[str, dict[str, Any]] = {}
    stakes: dict[str, dict[str, Any]] = {}
    playing_suits: dict[str, int] = {}
    playing_ranks: dict[str, int] = {}
    seals: dict[str, dict[str, Any]] = {}
    stickers: dict[str, dict[str, Any]] = {}
    overlays: dict[str, dict[str, Any]] = {}

    # 1. P_CENTERS: Jokers, Tarots, Planets, Spectrals, Vouchers, Backs, Enhancements, Boosters
    centers_body = _extract_lua_table_body(clean_lua, r"\bself\.P_CENTERS\s*=")
    for key, body in _iter_lua_entries(centers_body):
        item_set = _first_group(_SET_RE.search(body))
        name = _first_group(_NAME_RE.search(body))
        pos = _parse_pos(body, _POS_RE)
        if not item_set or not pos:
            continue
        x, y = pos

        if item_set == "Joker" and name:
            entry: dict[str, Any] = {"sheet": "Jokers.png", "x": x, "y": y}
            soul = _parse_pos(body, _SOUL_POS_RE)
            if soul:
                entry["soul"] = {"sheet": "Jokers.png", "x": soul[0], "y": soul[1]}
            jokers[name] = entry
        elif item_set == "Tarot" and name:
            tarots[name] = {"sheet": "Tarots.png", "x": x, "y": y}
        elif item_set == "Planet" and name:
            planets[name] = {"sheet": "Tarots.png", "x": x, "y": y}
        elif item_set == "Spectral" and name:
            spectrals[name] = {"sheet": "Tarots.png", "x": x, "y": y}
        elif item_set == "Voucher" and name:
            vouchers[name] = {"sheet": "Vouchers.png", "x": x, "y": y}
        elif item_set == "Back" and name:
            backs[name] = {"sheet": "Enhancers.png", "x": x, "y": y}
        elif item_set == "Default":
            enhancements["none"] = {"sheet": "Enhancers.png", "x": x, "y": y}
        elif item_set == "Enhanced" and name:
            enhancements[_normalize_enhancement_name(name)] = {
                "sheet": "Enhancers.png",
                "x": x,
                "y": y,
            }
        elif item_set == "Booster":
            kind = _first_group(_KIND_RE.search(body))
            if kind:
                size_tier = "Regular"
                if "_jumbo_" in key:
                    size_tier = "Jumbo"
                elif "_mega_" in key:
                    size_tier = "Mega"
                combo_key = f"{size_tier} {kind}"
                boosters.setdefault(
                    combo_key, {"sheet": "boosters.png", "x": x, "y": y}
                )
                if name:
                    boosters.setdefault(name, {"sheet": "boosters.png", "x": x, "y": y})

    # Enum spelling aliases vs Balatro Lua names
    _joker_aliases = {
        "Séance": "Seance",
        "Canio": "Caino",
        "Riff-Raff": "Riff-raff",
        "Driver": "Driver's License",
    }
    for enum_name, lua_name in _joker_aliases.items():
        if lua_name in jokers:
            jokers[enum_name] = jokers[lua_name]

    # 2. P_TAGS
    tags_body = _extract_lua_table_body(clean_lua, r"\bself\.P_TAGS\s*=")
    for _key, body in _iter_lua_entries(tags_body):
        name = _first_group(_NAME_RE.search(body))
        pos = _parse_pos(body, _POS_RE)
        if name and pos:
            tags[name] = {"sheet": "tags.png", "x": pos[0], "y": pos[1]}
    if "Skip Tag" in tags:
        tags["Speed Tag"] = tags["Skip Tag"]

    # 3. P_STAKES
    stakes_body = _extract_lua_table_body(clean_lua, r"\bself\.P_STAKES\s*=")
    for _key, body in _iter_lua_entries(stakes_body):
        name = _first_group(_NAME_RE.search(body))
        pos = _parse_pos(body, _POS_RE)
        if name and pos:
            sprite = {"sheet": "chips.png", "x": pos[0], "y": pos[1]}
            stakes[name] = sprite
            if name.endswith(" Chip"):
                stakes[name[:-5] + " Stake"] = sprite

    # 4. P_BLINDS
    blinds_body = _extract_lua_table_body(clean_lua, r"\bself\.P_BLINDS\s*=")
    for _key, body in _iter_lua_entries(blinds_body):
        name = _first_group(_NAME_RE.search(body))
        pos = _parse_pos(body, _POS_RE)
        if name and pos:
            b_entry: dict[str, Any] = {
                "sheet": "BlindChips.png",
                "x": pos[0],
                "y": pos[1],
            }
            hex_m = _HEX_RE.search(body)
            if hex_m:
                b_entry["color"] = f"#{hex_m.group(1)}"
            blinds[name] = b_entry

    # 5. P_CARDS
    cards_body = _extract_lua_table_body(clean_lua, r"\bself\.P_CARDS\s*=")
    for _key, body in _iter_lua_entries(cards_body):
        val = _first_group(_VALUE_RE.search(body))
        suit = _first_group(_SUIT_RE.search(body))
        pos = _parse_pos(body, _POS_RE)
        if val and suit and pos:
            playing_ranks.setdefault(val, pos[0])
            playing_suits.setdefault(suit.lower(), pos[1])

    # 6. Shared overlays, stickers, and seals
    debuff_m = re.search(
        r"self\.shared_debuff\s*=\s*Sprite\([^)]*\{\s*x\s*=\s*(\d+)\s*,\s*y\s*=\s*(\d+)\s*\}",
        clean_lua,
    )
    if debuff_m:
        overlays["debuff"] = {
            "sheet": "Enhancers.png",
            "x": int(debuff_m.group(1)),
            "y": int(debuff_m.group(2)),
        }

    for sticker_key in ("eternal", "perishable", "rental"):
        st_m = re.search(
            rf"self\.shared_sticker_{sticker_key}\s*=\s*Sprite\([^)]*\{{\s*x\s*=\s*(\d+)\s*,\s*y\s*=\s*(\d+)\s*\}}",
            clean_lua,
        )
        if st_m:
            stickers[sticker_key] = {
                "sheet": "stickers.png",
                "x": int(st_m.group(1)),
                "y": int(st_m.group(2)),
            }

    seals_body = _extract_lua_table_body(clean_lua, r"\bself\.shared_seals\s*=")
    for seal_name in ("Gold", "Purple", "Red", "Blue"):
        sl_m = re.search(
            rf"\b{seal_name}\s*=\s*Sprite\([^)]*\{{\s*x\s*=\s*(\d+)\s*,\s*y\s*=\s*(\d+)\s*\}}",
            seals_body,
        )
        if sl_m:
            seals[seal_name.lower()] = {
                "sheet": "Enhancers.png",
                "x": int(sl_m.group(1)),
                "y": int(sl_m.group(2)),
            }

    return {
        "sheets": SPRITE_SHEETS,
        "jokers": jokers,
        "tarots": tarots,
        "planets": planets,
        "spectrals": spectrals,
        "vouchers": vouchers,
        "backs": backs,
        "boosters": boosters,
        "enhancements": enhancements,
        "tags": tags,
        "blinds": blinds,
        "stakes": stakes,
        "seals": seals,
        "stickers": stickers,
        "overlays": overlays,
        "playing_card_grid": {
            "sheet": "8BitDeck.png",
            "suits": playing_suits,
            "ranks": playing_ranks,
        },
    }


def load_balatro_sprite_manifest() -> dict[str, Any]:
    """Returns a fresh copy of the parsed Balatro 2x sprite atlas manifest."""
    return copy.deepcopy(_build_sprite_manifest_cached())


# ---------------------------------------------------------------------------
# Trajectory Visualization Payload & Step Diff Engine
# ---------------------------------------------------------------------------

_SCALAR_DIFF_FIELDS: tuple[str, ...] = (
    "location",
    "ante",
    "round",
    "current_blind",
    "boss_blind",
    "boss_disabled",
    "round_won",
    "current_chips",
    "target_chips",
    "money",
    "hands_left",
    "discards_left",
    "hands_played_this_round",
    "discards_used_this_round",
    "reroll_cost",
    "free_rerolls",
    "inflation",
    "bankrupt_at",
    "shop_size",
    "joker_slots",
    "max_jokers",
    "consumable_slots",
    "max_consumables",
    "hand_max_size",
    "small_blind_tag",
    "big_blind_tag",
    "small_blind_orbital_hand",
    "big_blind_orbital_hand",
    "blinds_skipped",
    "temp_hand_size_bonus",
    "last_tarot_planet",
    "gros_michel_extinct",
    "tarots_used",
    "unused_discards",
    "ectoplasm_minus",
    "coupon_shop_active",
    "idol_card",
    "mail_card_rank",
    "ancient_suit",
    "castle_suit",
    "to_do_poker_hand",
    "locked_hand_type",
    "ox_target_hand",
    "active_pack",
    "pack_choices_remaining",
    "full_deck_count",
    "draw_pile_count",
    "discard_pile_count",
    "played_round_count",
    "played_ante_count",
)


def _format_playing_card_label(card: PlayingCard | dict[str, Any]) -> str:
    if isinstance(card, dict):
        rank = card.get("rank", "none")
        suit = card.get("suit", "none")
        enh = card.get("enhancement", "none")
    else:
        rank = card.rank.value
        suit = card.suit.value
        enh = card.enhancement.value
    if enh == "stone" or (rank == "none" and suit == "none"):
        return "Stone Card" if enh == "stone" else "Unknown Card"
    suit_sym = {"spades": "♠", "hearts": "♥", "clubs": "♣", "diamonds": "♦"}.get(
        suit, suit
    )
    return f"{rank}{suit_sym}"


def _populate_joker_fields(d: dict[str, Any], j: Joker) -> None:
    """Ensures serialized Joker dicts carry sticker, edition, and state metadata uniformly."""
    d["edition"] = j.edition.value
    d["eternal"] = j.eternal
    d["perishable"] = j.perishable
    d["perish_tally"] = j.perish_tally
    d["rental"] = j.rental
    d["debuffed"] = j.debuffed
    d["facing_back"] = j.facing_back
    d["extra_value"] = j.extra_value
    d["rarity"] = j.info.rarity.value
    d["blueprint_compat"] = j.blueprint_compat


def _enrich_joker(j: Joker, state: GameState) -> dict[str, Any]:
    d = _serialize_joker(j)
    astronomer = any(
        jk.name == JokerName.ASTRONOMER and not jk.debuffed
        for jk in state.inventory.jokers
    )
    _populate_joker_fields(d, j)
    d["sell_value"] = j.get_sell_value(
        discount_percent=state.discount_percent,
        inflation=state.inflation,
        astronomer=astronomer,
    )
    return d


def _enrich_card(card: Card, state: GameState) -> dict[str, Any]:
    d = _serialize_card(card)
    astronomer = any(
        jk.name == JokerName.ASTRONOMER and not jk.debuffed
        for jk in state.inventory.jokers
    )
    d["edition"] = card.edition.value
    d["couponed"] = card.couponed
    d["rental"] = card.rental
    d["extra_value"] = card.extra_value
    d["buy_cost"] = card.get_cost(
        discount_percent=state.discount_percent,
        inflation=state.inflation,
        astronomer=astronomer,
    )
    d["sell_value"] = card.get_sell_value(
        discount_percent=state.discount_percent,
        inflation=state.inflation,
        astronomer=astronomer,
    )
    if isinstance(card, Joker):
        _populate_joker_fields(d, card)
    elif isinstance(card, PlayingCard):
        d["enhancement"] = card.enhancement.value
        d["seal"] = card.seal.value
        d["bonus_chips"] = card.bonus_chips
        d["debuffed"] = card.debuffed
        d["destroyed"] = card.destroyed
    return d


def _enrich_hand_card(hc: HandCard) -> dict[str, Any]:
    d = _serialize_hand_card(hc)
    c = hc.card
    d["card"] = {
        "rank": c.rank.value,
        "suit": c.suit.value,
        "enhancement": c.enhancement.value,
        "seal": c.seal.value,
        "edition": c.edition.value,
        "bonus_chips": c.bonus_chips,
        "debuffed": c.debuffed,
        "id": c.id,
    }
    d["selected"] = hc.selected
    d["facing_back"] = hc.facing_back
    d["forced_selection"] = hc.forced_selection
    d["slot_uid"] = hc.slot_uid
    d["resolved"] = True
    if hc.bounds is not None:
        d["resolved"] = hc.bounds.is_resolved
        d["candidate_count"] = len(
            hc.bounds.candidate_cards or hc.bounds.candidate_card_ids
        )
    return d


def _enrich_voucher(v: Voucher, state: GameState) -> dict[str, Any]:
    d = _serialize_voucher(v)
    d["buy_cost"] = v.get_cost(
        discount_percent=state.discount_percent, inflation=state.inflation
    )
    return d


def _enrich_pack(bp: BoosterPack, state: GameState) -> dict[str, Any]:
    d = _serialize_booster_pack(bp)
    astronomer = any(
        jk.name == JokerName.ASTRONOMER and not jk.debuffed
        for jk in state.inventory.jokers
    )
    d["buy_cost"] = bp.get_cost(
        discount_percent=state.discount_percent,
        inflation=state.inflation,
        astronomer=astronomer,
    )
    return d


def _pop_matching_card(target: PlayingCard, pool: list[PlayingCard]) -> bool:
    """Consumes one matching card from `pool`, prioritizing persistent `id` matches."""
    if target.id:
        for i, cand in enumerate(pool):
            if cand.id == target.id:
                pool.pop(i)
                return True
    for i, cand in enumerate(pool):
        if _cards_match(target, cand):
            pool.pop(i)
            return True
    return False


def _serialize_playing_card_full(
    pc: PlayingCard, zone: str | None = None
) -> dict[str, Any]:
    d: dict[str, Any] = {
        "id": pc.id,
        "rank": pc.rank.value,
        "suit": pc.suit.value,
    }
    if pc.enhancement != Enhancement.NONE:
        d["enhancement"] = pc.enhancement.value
    if pc.seal != Seal.NONE:
        d["seal"] = pc.seal.value
    if pc.edition != Edition.NONE:
        d["edition"] = pc.edition.value
    if pc.bonus_chips:
        d["bonus_chips"] = pc.bonus_chips
    if pc.debuffed:
        d["debuffed"] = True
    if pc.destroyed:
        d["destroyed"] = True
    if zone is not None:
        d["zone"] = zone
    return d


def _serialize_state_snapshot(state: GameState) -> dict[str, Any]:
    """Serializes a complete, self-contained UI snapshot for a single step."""
    # Zone pools matching GameState.remaining_deck ownership semantics
    hand_pool = [
        (hc._real_card if (hc.facing_back and hc._real_card is not None) else hc.card)
        for hc in state.hand.cards
    ]
    played_pool = list(state.played_cards_this_round)
    discard_pool = list(state.discard_pile)

    full_deck_list: list[dict[str, Any]] = []
    for pc in state.full_deck or []:
        if _pop_matching_card(pc, hand_pool):
            zone = "in_hand"
        elif _pop_matching_card(pc, played_pool):
            zone = "played_round"
        elif _pop_matching_card(pc, discard_pool):
            zone = "discarded"
        else:
            zone = "in_draw_pile"
        full_deck_list.append(_serialize_playing_card_full(pc, zone=zone))

    draw_pile = [
        _serialize_playing_card_full(c, zone="in_draw_pile")
        for c in (state.remaining_deck if state.full_deck is not None else [])
    ]
    discard_pile = [
        _serialize_playing_card_full(c, zone="discarded") for c in state.discard_pile
    ]
    played_round = [
        _serialize_playing_card_full(c, zone="played_round")
        for c in state.played_cards_this_round
    ]
    played_ante = [
        _serialize_playing_card_full(c) for c in state.played_cards_this_ante
    ]

    poker_hands: list[dict[str, Any]] = []
    for hand_enum, defaults in POKER_HAND_DEFAULTS.items():
        lvl = state.hand_levels.get(hand_enum, 1)
        plays = state.hand_play_counts.get(hand_enum, 0)
        chips = defaults.base_chips + (lvl - 1) * defaults.l_chips
        mult = defaults.base_mult + (lvl - 1) * defaults.l_mult
        poker_hands.append(
            {
                "hand": hand_enum.value,
                "level": lvl,
                "chips": chips,
                "mult": mult,
                "play_count": plays,
                "played_this_round": hand_enum in state.hands_played_this_round_by_type,
            }
        )

    return {
        "step_idx": state.step_idx,
        "frame_idx": state.frame_idx,
        "timestamp": state.timestamp,
        "location": state.location.value,
        "deck": state.deck.value,
        "stake": state.stake.value,
        "seed": state.seed,
        "ante": state.ante,
        "round": state.round,
        "current_blind": state.current_blind.value,
        "boss_blind": state.boss_blind.value if state.boss_blind is not None else None,
        "boss_disabled": state.boss_disabled,
        "round_won": state.round_won,
        "current_chips": state.current_chips,
        "target_chips": state.target_chips,
        "money": state.money,
        "hands_left": state.hands_left,
        "discards_left": state.discards_left,
        "hands_played_this_round": state.hands_played_this_round,
        "discards_used_this_round": state.discards_used_this_round,
        "reroll_cost": state.reroll_cost,
        "free_rerolls": state.free_rerolls,
        "inflation": state.inflation,
        "bankrupt_at": state.bankrupt_at,
        "shop_size": state.shop_size,
        "joker_slots": state.inventory.joker_slots,
        "max_jokers": state.inventory.max_jokers,
        "consumable_slots": state.inventory.consumable_slots,
        "max_consumables": state.inventory.max_consumables,
        "hand_max_size": state.hand.max_size,
        "small_blind_tag": state.small_blind_tag.value
        if state.small_blind_tag
        else None,
        "big_blind_tag": state.big_blind_tag.value if state.big_blind_tag else None,
        "small_blind_orbital_hand": (
            state.small_blind_orbital_hand.value
            if state.small_blind_orbital_hand
            else None
        ),
        "big_blind_orbital_hand": (
            state.big_blind_orbital_hand.value if state.big_blind_orbital_hand else None
        ),
        "active_tags": [t.value for t in state.active_tags],
        "vouchers_redeemed": [v.value for v in state.vouchers_redeemed],
        "blinds_skipped": state.blinds_skipped,
        "temp_hand_size_bonus": state.temp_hand_size_bonus,
        "last_tarot_planet": state.last_tarot_planet.value
        if state.last_tarot_planet
        else None,
        "gros_michel_extinct": state.gros_michel_extinct,
        "tarots_used": state.tarots_used,
        "unused_discards": state.unused_discards,
        "ectoplasm_minus": state.ectoplasm_minus,
        "unique_planets_used": sorted(state.unique_planets_used),
        "coupon_shop_active": state.coupon_shop_active,
        "no_interest": state.no_interest,
        "ante_scaling": state.ante_scaling,
        "enable_eternals_in_shop": state.enable_eternals_in_shop,
        "enable_perishables_in_shop": state.enable_perishables_in_shop,
        "enable_rentals_in_shop": state.enable_rentals_in_shop,
        "tarot_rate": state.tarot_rate,
        "planet_rate": state.planet_rate,
        "spectral_rate": state.spectral_rate,
        "playing_card_rate": state.playing_card_rate,
        "edition_rate": state.edition_rate,
        "idol_card": [state.idol_card[0].value, state.idol_card[1].value],
        "mail_card_rank": state.mail_card_rank.value,
        "ancient_suit": state.ancient_suit.value,
        "castle_suit": state.castle_suit.value,
        "to_do_poker_hand": state.to_do_poker_hand.value,
        "locked_hand_type": state.locked_hand_type.value
        if state.locked_hand_type
        else None,
        "ox_target_hand": state.ox_target_hand.value if state.ox_target_hand else None,
        "hands_played_this_round_by_type": sorted(
            h.value for h in state.hands_played_this_round_by_type
        ),
        "pending_cv_checks": sorted(state.pending_cv_checks),
        "jokers": [_enrich_joker(j, state) for j in state.inventory.jokers],
        "consumables": [_enrich_card(c, state) for c in state.inventory.consumables],
        "hand": [_enrich_hand_card(hc) for hc in state.hand.cards],
        "shop_cards": [_enrich_card(c, state) for c in state.shop_cards],
        "shop_vouchers": [_enrich_voucher(v, state) for v in state.shop_vouchers],
        "shop_packs": [_enrich_pack(bp, state) for bp in state.shop_packs],
        "active_pack": _enrich_pack(state.active_pack, state)
        if state.active_pack
        else None,
        "pack_choices_remaining": state.pack_choices_remaining,
        "pre_pack_location": state.pre_pack_location.value
        if state.pre_pack_location
        else None,
        "pending_packs": [_enrich_pack(bp, state) for bp in state.pending_packs],
        "pack_cards": [_enrich_card(c, state) for c in state.pack_cards],
        "poker_hands": poker_hands,
        "full_deck": full_deck_list,
        "draw_pile": draw_pile,
        "discard_pile": discard_pile,
        "played_cards_this_round": played_round,
        "played_cards_this_ante": played_ante,
        "full_deck_count": len(full_deck_list),
        "draw_pile_count": len(draw_pile),
        "discard_pile_count": len(discard_pile),
        "played_round_count": len(played_round),
        "played_ante_count": len(played_ante),
    }


def _ordered_jokers_for_action(
    jokers: list[dict[str, Any]],
    j_order: Sequence[int] | None,
) -> list[dict[str, Any]]:
    """Applies action joker reordering before resolving SELL_CARD indices, matching `rules.py`."""
    if (
        not j_order
        or not jokers
        or len(j_order) != len(jokers)
        or len(set(j_order)) != len(j_order)
    ):
        return jokers
    by_id = {j.get("id"): j for j in jokers if j.get("id")}
    if all(jid in by_id for jid in j_order):
        return [by_id[jid] for jid in j_order]
    if all(isinstance(x, int) and 0 <= x < len(jokers) for x in j_order):
        return [jokers[x] for x in j_order]
    return jokers


def _enrich_observed_shop(raw_obs: Any, cost_state: GameState) -> dict[str, Any] | None:
    """Enriches pre-action `observed_shop` metadata with full costs, editions, and stickers."""
    if not isinstance(raw_obs, dict):
        return None
    return {
        "cards": [
            _enrich_card(_deserialize_card(c), cost_state)
            for c in raw_obs.get("cards", ())
        ],
        "vouchers": [
            {"type": "voucher", **_enrich_voucher(_deserialize_voucher(v), cost_state)}
            for v in raw_obs.get("vouchers", ())
        ],
        "packs": [
            {
                "type": "booster_pack",
                **_enrich_pack(_deserialize_booster_pack(p), cost_state),
            }
            for p in raw_obs.get("packs", ())
        ],
    }


def _find_gained_shop_item_by_name(
    prev_snap: dict[str, Any] | None,
    curr_snap: dict[str, Any],
    target_name: str,
) -> dict[str, Any] | None:
    """Finds a newly acquired joker or consumable in `curr_snap` matching `target_name`."""
    prev_j_ids = {j.get("id") for j in (prev_snap or {}).get("jokers", ())}
    for j in curr_snap.get("jokers", ()):
        if j.get("name") == target_name and j.get("id") not in prev_j_ids:
            return {"type": "joker", **j}
    for c in curr_snap.get("consumables", ()):
        if c.get("name") == target_name:
            return c
    return None


def _attach_hand_exit_breakdown(
    hand_diff: dict[str, Any],
    act_dict: dict[str, Any],
    prev_snap: dict[str, Any],
    curr_snap: dict[str, Any],
) -> None:
    """Annotates `hand_diff` with `played_cards` and `discarded_cards` (e.g., Boss Blind The Hook)."""
    prev_disc = prev_snap.get("discard_pile", ())
    curr_disc = curr_snap.get("discard_pile", ())
    selected = list(act_dict.get("selected_cards") or ())
    if not selected and prev_snap.get("hand"):
        selected = [hc["card"] for hc in prev_snap["hand"] if hc.get("selected")]

    newly_discarded: list[dict[str, Any]] = []
    if len(curr_disc) >= len(prev_disc) and list(curr_disc[: len(prev_disc)]) == list(
        prev_disc
    ):
        newly_discarded = list(curr_disc[len(prev_disc) :])

    if act_dict["type"] == ActionType.PLAY_HAND.value:
        hand_diff["played_cards"] = selected
        played_pool = list(selected)
        extra_discards: list[dict[str, Any]] = []
        for card_d in newly_discarded:
            matched_idx = None
            cid = card_d.get("id")
            if cid:
                for pi, pc in enumerate(played_pool):
                    if pc.get("id") == cid:
                        matched_idx = pi
                        break
            if matched_idx is None:
                for pi, pc in enumerate(played_pool):
                    if pc.get("rank") == card_d.get("rank") and pc.get(
                        "suit"
                    ) == card_d.get("suit"):
                        matched_idx = pi
                        break
            if matched_idx is not None:
                played_pool.pop(matched_idx)
            else:
                extra_discards.append(card_d)
        hand_diff["discarded_cards"] = extra_discards
    elif act_dict["type"] == ActionType.DISCARD.value:
        hand_diff["played_cards"] = []
        hand_diff["discarded_cards"] = newly_discarded if newly_discarded else selected


def _resolve_action_with_target(
    act: Action | None,
    prev_snap: dict[str, Any] | None,
    curr_snap: dict[str, Any],
    cost_state: GameState | None = None,
) -> dict[str, Any] | None:
    """Serializes an Action along with pre-action `target_item` and `observed_shop` metadata."""
    if act is None:
        return None
    out = _serialize_action(act)
    # Keep selected cards clean and fully labeled even when compact
    if act.selected_cards:
        out["selected_cards"] = [
            _serialize_playing_card_full(c) for c in act.selected_cards
        ]

    obs_shop = (
        _enrich_observed_shop(act.metadata.get("observed_shop"), cost_state)
        if act.metadata and cost_state is not None
        else None
    )
    if obs_shop is not None:
        out["observed_shop"] = obs_shop
    if act.metadata and act.metadata.get("observed_pack"):
        out["observed_pack"] = list(act.metadata["observed_pack"])

    ref = prev_snap if prev_snap is not None else curr_snap
    idx = act.target_idx
    meta_tc = act.metadata.get("target_card") if act.metadata else None
    target_item: dict[str, Any] | None = None

    if act.type in (ActionType.BUY_CARD, ActionType.BUY_AND_USE) and idx is not None:
        obs_cards = obs_shop["cards"] if obs_shop else []
        shop_items = obs_cards or ref.get("shop_cards", [])
        pack_items = ref.get("pack_cards", [])
        if (
            not obs_cards
            and ref.get("location") == "pack"
            and 0 <= idx < len(pack_items)
        ):
            item = pack_items[idx]
            target_item = {
                "area": "pack_cards",
                "idx": idx,
                "label": item.get("name") or _format_playing_card_label(item),
                "item": item,
            }
        elif 0 <= idx < len(shop_items):
            item = shop_items[idx]
            item_label = item.get("name") or _format_playing_card_label(item)
            if not obs_cards and meta_tc and item_label != meta_tc:
                gained = _find_gained_shop_item_by_name(prev_snap, curr_snap, meta_tc)
                target_item = {
                    "area": "shop_cards",
                    "idx": idx,
                    "label": meta_tc,
                    **({"item": gained} if gained is not None else {}),
                }
            else:
                target_item = {
                    "area": "shop_cards",
                    "idx": idx,
                    "label": item_label,
                    "item": item,
                }
        else:
            target_item = {
                "area": "shop_cards",
                "idx": idx,
                "label": meta_tc or f"Card Option #{idx}",
            }
    elif act.type == ActionType.REDEEM_VOUCHER and idx is not None:
        obs_vouchers = obs_shop["vouchers"] if obs_shop else []
        items = obs_vouchers or ref.get("shop_vouchers", [])
        if 0 <= idx < len(items):
            item = items[idx]
            target_item = {
                "area": "shop_vouchers",
                "idx": idx,
                "label": item.get("name", "Voucher"),
                "item": {"type": "voucher", **item},
            }
        elif meta_tc:
            target_item = {"area": "shop_vouchers", "idx": idx, "label": meta_tc}
    elif act.type == ActionType.OPEN_PACK:
        obs_packs = obs_shop["packs"] if obs_shop else []
        items = obs_packs or ref.get("shop_packs", [])
        if idx is not None and 0 <= idx < len(items):
            item = items[idx]
            target_item = {
                "area": "shop_packs",
                "idx": idx,
                "label": f"{item['size']} {item['booster_type']} Pack",
                "item": {"type": "booster_pack", **item},
            }
        elif curr_snap.get("active_pack"):
            item = curr_snap["active_pack"]
            target_item = {
                "area": "pack",
                "idx": idx,
                "label": f"{item['size']} {item['booster_type']} Pack",
                "item": {"type": "booster_pack", **item},
            }
    elif act.type == ActionType.USE_IN_PACK and idx is not None:
        items = ref.get("pack_cards", [])
        if 0 <= idx < len(items):
            item = items[idx]
            target_item = {
                "area": "pack_cards",
                "idx": idx,
                "label": item.get("name") or _format_playing_card_label(item),
                "item": item,
            }
        else:
            target_item = {
                "area": "pack_cards",
                "idx": idx,
                "label": meta_tc or f"Pack Option #{idx}",
            }
    elif act.type == ActionType.USE_CONSUMABLE and idx is not None:
        items = ref.get("consumables", [])
        if 0 <= idx < len(items):
            item = items[idx]
            target_item = {
                "area": "consumables",
                "idx": idx,
                "label": item.get("name", "Consumable"),
                "item": item,
            }
    elif act.type == ActionType.SELL_CARD and idx is not None:
        jokers = _ordered_jokers_for_action(ref.get("jokers", []), act.joker_order)
        consumables = ref.get("consumables", [])
        if 0 <= idx < len(jokers):
            item = jokers[idx]
            target_item = {
                "area": "jokers",
                "idx": idx,
                "label": item.get("name", "Joker"),
                "item": {"type": "joker", **item},
            }
        else:
            c_idx = idx - len(jokers)
            if 0 <= c_idx < len(consumables):
                item = consumables[c_idx]
                target_item = {
                    "area": "consumables",
                    "idx": idx,
                    "label": item.get("name", "Consumable"),
                    "item": item,
                }
    elif act.type in (ActionType.PLAY_HAND, ActionType.DISCARD):
        cards = out.get("selected_cards") or [
            hc["card"] for hc in ref.get("hand", []) if hc.get("selected")
        ]
        if cards:
            target_item = {
                "area": "hand",
                "label": ", ".join(_format_playing_card_label(c) for c in cards),
                "cards": cards,
            }
    elif act.type == ActionType.SELECT_BLIND:
        blind = ref.get("current_blind", "Small Blind")
        display = (
            ref.get("boss_blind")
            if blind not in ("Small Blind", "Big Blind")
            else blind
        )
        target_item = {
            "area": "blind",
            "label": display or blind,
            "blind": display or blind,
        }
    elif act.type == ActionType.SKIP_BLIND:
        blind = ref.get("current_blind", "Blind")
        tag = (
            ref.get("small_blind_tag")
            if blind == "Small Blind"
            else ref.get("big_blind_tag")
        )
        target_item = {
            "area": "blind",
            "label": f"Skipped {blind}" + (f" ({tag})" if tag else ""),
            "blind": blind,
            "tag": tag,
        }

    out["target_item"] = target_item
    return out


def _diff_jokers(
    prev_list: list[dict[str, Any]], curr_list: list[dict[str, Any]]
) -> dict[str, Any]:
    prev_by_id = {j["id"]: j for j in prev_list if j.get("id")}
    curr_by_id = {j["id"]: j for j in curr_list if j.get("id")}

    # Fallback positional diff only if IDs are absent
    if len(prev_by_id) != len(prev_list) or len(curr_by_id) != len(curr_list):
        changed = prev_list != curr_list
        added = [i for i, j in enumerate(curr_list) if j not in prev_list]
        removed = [j for j in prev_list if j not in curr_list]
        return {
            "added_indices": added,
            "removed": removed,
            "updated": {},
            "reordered": False,
            "changed": changed,
        }

    added_ids = [j["id"] for j in curr_list if j["id"] not in prev_by_id]
    removed = [j for j in prev_list if j["id"] not in curr_by_id]
    updated: dict[str, dict[str, Any]] = {}

    compare_fields = (
        "name",
        "edition",
        "eternal",
        "perishable",
        "perish_tally",
        "rental",
        "debuffed",
        "facing_back",
        "extra_value",
        "sell_value",
        "state",
    )
    for j in curr_list:
        jid = j["id"]
        pj = prev_by_id.get(jid)
        if pj is None:
            continue
        diffs: dict[str, Any] = {}
        for f in compare_fields:
            if pj.get(f) != j.get(f):
                diffs[f] = {"prev": pj.get(f), "curr": j.get(f)}
        if diffs:
            updated[str(jid)] = diffs

    common_prev = [j["id"] for j in prev_list if j["id"] in curr_by_id]
    common_curr = [j["id"] for j in curr_list if j["id"] in prev_by_id]
    reordered = common_prev != common_curr

    return {
        "added_ids": added_ids,
        "removed": removed,
        "updated": updated,
        "reordered": reordered,
        "changed": bool(added_ids or removed or updated or reordered),
    }


def _diff_simple_items(
    prev_list: list[dict[str, Any]],
    curr_list: list[dict[str, Any]],
    *,
    identity_fn: Any,
    compare_fields: tuple[str, ...],
) -> dict[str, Any]:
    """Matches list elements conservatively in order and reports added/removed/updated indices."""
    prev_used = [False] * len(prev_list)
    added_indices: list[int] = []
    updated: dict[str, dict[str, Any]] = {}
    matched_prev_order: list[int] = []

    for ci, c_item in enumerate(curr_list):
        c_key = identity_fn(c_item)
        match_pi = None
        # Prefer exact match first, then identity-key match
        for pi, p_item in enumerate(prev_list):
            if not prev_used[pi] and p_item == c_item:
                match_pi = pi
                break
        if match_pi is None:
            for pi, p_item in enumerate(prev_list):
                if not prev_used[pi] and identity_fn(p_item) == c_key:
                    match_pi = pi
                    break
        if match_pi is None:
            added_indices.append(ci)
        else:
            prev_used[match_pi] = True
            matched_prev_order.append(match_pi)
            p_item = prev_list[match_pi]
            diffs: dict[str, Any] = {}
            for f in compare_fields:
                if p_item.get(f) != c_item.get(f):
                    diffs[f] = {"prev": p_item.get(f), "curr": c_item.get(f)}
            if diffs:
                updated[str(ci)] = diffs

    removed = [prev_list[pi] for pi, used in enumerate(prev_used) if not used]
    reordered = matched_prev_order != sorted(matched_prev_order)
    return {
        "added_indices": added_indices,
        "removed": removed,
        "updated": updated,
        "reordered": reordered,
        "changed": bool(added_indices or removed or updated or reordered),
    }


def _diff_hand(
    prev_hand: list[dict[str, Any]], curr_hand: list[dict[str, Any]]
) -> dict[str, Any]:
    def _hand_id(hc: dict[str, Any]) -> Any:
        if hc.get("slot_uid"):
            return ("uid", hc["slot_uid"])
        c = hc.get("card", {})
        if c.get("id"):
            return ("card_id", c["id"])
        return (
            "rs",
            c.get("rank"),
            c.get("suit"),
            c.get("enhancement"),
            c.get("seal"),
            c.get("edition"),
        )

    prev_used = [False] * len(prev_hand)
    added_indices: list[int] = []
    updated: dict[str, dict[str, Any]] = {}
    matched_prev_order: list[int] = []

    for ci, chc in enumerate(curr_hand):
        ckey = _hand_id(chc)
        mpi = None
        for pi, phc in enumerate(prev_hand):
            if not prev_used[pi] and _hand_id(phc) == ckey:
                mpi = pi
                break
        if mpi is None:
            added_indices.append(ci)
        else:
            prev_used[mpi] = True
            matched_prev_order.append(mpi)
            phc = prev_hand[mpi]
            diffs: dict[str, Any] = {}
            for f in ("selected", "facing_back", "forced_selection", "resolved"):
                if phc.get(f) != chc.get(f):
                    diffs[f] = {"prev": phc.get(f), "curr": chc.get(f)}
            pc, cc = phc.get("card", {}), chc.get("card", {})
            for cf in (
                "rank",
                "suit",
                "enhancement",
                "seal",
                "edition",
                "bonus_chips",
                "debuffed",
            ):
                if pc.get(cf) != cc.get(cf):
                    diffs[f"card.{cf}"] = {"prev": pc.get(cf), "curr": cc.get(cf)}
            if diffs:
                updated[str(ci)] = diffs

    removed = [prev_hand[pi] for pi, used in enumerate(prev_used) if not used]
    reordered = matched_prev_order != sorted(matched_prev_order)
    return {
        "added_indices": added_indices,
        "removed": removed,
        "updated": updated,
        "reordered": reordered,
        "changed": bool(added_indices or removed or updated or reordered),
    }


def _diff_full_deck(
    prev_deck: list[dict[str, Any]], curr_deck: list[dict[str, Any]]
) -> dict[str, Any]:
    prev_by_id = {c["id"]: c for c in prev_deck if c.get("id")}
    curr_by_id = {c["id"]: c for c in curr_deck if c.get("id")}

    added_ids: list[int] = []
    removed: list[dict[str, Any]] = []
    modified: dict[str, dict[str, Any]] = {}
    zone_changed: dict[str, dict[str, str]] = {}

    deck_field_defaults: dict[str, Any] = {
        "rank": "none",
        "suit": "none",
        "enhancement": "none",
        "seal": "none",
        "edition": "none",
        "bonus_chips": 0,
        "debuffed": False,
    }
    if len(prev_by_id) == len(prev_deck) and len(curr_by_id) == len(curr_deck):
        for cid, cc in curr_by_id.items():
            pc = prev_by_id.get(cid)
            if pc is None:
                added_ids.append(cid)
                continue
            mods: dict[str, Any] = {}
            for f, default_val in deck_field_defaults.items():
                if pc.get(f, default_val) != cc.get(f, default_val):
                    mods[f] = {
                        "prev": pc.get(f, default_val),
                        "curr": cc.get(f, default_val),
                    }
            if mods:
                modified[str(cid)] = mods
            if pc.get("zone") != cc.get("zone"):
                zone_changed[str(cid)] = {
                    "prev": pc.get("zone", ""),
                    "curr": cc.get("zone", ""),
                }
        for cid, pc in prev_by_id.items():
            if cid not in curr_by_id:
                removed.append(pc)
    else:
        if prev_deck != curr_deck:
            added_ids = [
                c.get("id", i) for i, c in enumerate(curr_deck) if c not in prev_deck
            ]
            removed = [c for c in prev_deck if c not in curr_deck]

    return {
        "added_ids": added_ids,
        "removed": removed,
        "modified": modified,
        "zone_changed": zone_changed,
        "changed": bool(added_ids or removed or modified or zone_changed),
    }


def _compute_step_diff(
    prev_snap: dict[str, Any] | None, curr_snap: dict[str, Any]
) -> dict[str, Any]:
    """Computes structured changes from `prev_snap` to `curr_snap`."""
    if prev_snap is None:
        return {
            "has_changes": False,
            "changed_fields": {},
            "jokers": {
                "added_ids": [],
                "removed": [],
                "updated": {},
                "reordered": False,
                "changed": False,
            },
            "consumables": {
                "added_indices": [],
                "removed": [],
                "updated": {},
                "reordered": False,
                "changed": False,
            },
            "hand": {
                "added_indices": [],
                "removed": [],
                "updated": {},
                "reordered": False,
                "changed": False,
            },
            "shop_cards": {
                "added_indices": [],
                "removed": [],
                "updated": {},
                "reordered": False,
                "changed": False,
            },
            "shop_vouchers": {"added": [], "removed": [], "changed": False},
            "shop_packs": {"added": [], "removed": [], "changed": False},
            "pack_cards": {
                "added_indices": [],
                "removed": [],
                "updated": {},
                "reordered": False,
                "changed": False,
            },
            "active_tags": {"added": [], "removed": [], "changed": False},
            "vouchers_redeemed": {"added": [], "removed": [], "changed": False},
            "poker_hands": {},
            "full_deck": {
                "added_ids": [],
                "removed": [],
                "modified": {},
                "zone_changed": {},
                "changed": False,
            },
            "badges": [],
            "summary_items": [],
        }

    changed_fields: dict[str, dict[str, Any]] = {}
    for field_name in _SCALAR_DIFF_FIELDS:
        ov = prev_snap.get(field_name)
        nv = curr_snap.get(field_name)
        if ov != nv:
            changed_fields[field_name] = {"prev": ov, "curr": nv}

    jokers_diff = _diff_jokers(prev_snap["jokers"], curr_snap["jokers"])
    consumables_diff = _diff_simple_items(
        prev_snap["consumables"],
        curr_snap["consumables"],
        identity_fn=lambda c: (c.get("type"), c.get("name")),
        compare_fields=("edition", "extra_value", "sell_value", "rental", "couponed"),
    )
    hand_diff = _diff_hand(prev_snap["hand"], curr_snap["hand"])
    shop_cards_diff = _diff_simple_items(
        prev_snap["shop_cards"],
        curr_snap["shop_cards"],
        identity_fn=lambda c: (
            c.get("type"),
            c.get("name"),
            c.get("rank"),
            c.get("suit"),
            c.get("id"),
        ),
        compare_fields=(
            "buy_cost",
            "edition",
            "couponed",
            "rental",
            "eternal",
            "perishable",
        ),
    )
    pack_cards_diff = _diff_simple_items(
        prev_snap["pack_cards"],
        curr_snap["pack_cards"],
        identity_fn=lambda c: (
            c.get("type"),
            c.get("name"),
            c.get("rank"),
            c.get("suit"),
            c.get("id"),
        ),
        compare_fields=("edition", "enhancement", "seal"),
    )

    prev_vouchers = [v["name"] for v in prev_snap["shop_vouchers"]]
    curr_vouchers = [v["name"] for v in curr_snap["shop_vouchers"]]
    shop_vouchers_diff = {
        "added": [v for v in curr_vouchers if v not in prev_vouchers],
        "removed": [v for v in prev_vouchers if v not in curr_vouchers],
        "changed": prev_snap["shop_vouchers"] != curr_snap["shop_vouchers"],
    }

    prev_packs = [f"{p['size']} {p['booster_type']}" for p in prev_snap["shop_packs"]]
    curr_packs = [f"{p['size']} {p['booster_type']}" for p in curr_snap["shop_packs"]]
    shop_packs_diff = {
        "added": [p for p in curr_packs if p not in prev_packs],
        "removed": [p for p in prev_packs if p not in curr_packs],
        "changed": prev_snap["shop_packs"] != curr_snap["shop_packs"],
    }

    prev_tags = prev_snap["active_tags"]
    curr_tags = curr_snap["active_tags"]
    active_tags_diff = {
        "added": [t for t in curr_tags if curr_tags.count(t) > prev_tags.count(t)],
        "removed": [t for t in prev_tags if prev_tags.count(t) > curr_tags.count(t)],
        "changed": prev_tags != curr_tags,
    }

    prev_redeemed = prev_snap["vouchers_redeemed"]
    curr_redeemed = curr_snap["vouchers_redeemed"]
    vouchers_redeemed_diff = {
        "added": [v for v in curr_redeemed if v not in prev_redeemed],
        "removed": [v for v in prev_redeemed if v not in curr_redeemed],
        "changed": prev_redeemed != curr_redeemed,
    }

    poker_hands_diff: dict[str, dict[str, Any]] = {}
    prev_ph_map = {h["hand"]: h for h in prev_snap["poker_hands"]}
    for ch in curr_snap["poker_hands"]:
        ph = prev_ph_map.get(ch["hand"])
        if ph is None:
            continue
        diffs: dict[str, Any] = {}
        for f in ("level", "play_count", "played_this_round"):
            if ph.get(f) != ch.get(f):
                diffs[f] = {"prev": ph.get(f), "curr": ch.get(f)}
        if diffs:
            poker_hands_diff[ch["hand"]] = diffs

    full_deck_diff = _diff_full_deck(prev_snap["full_deck"], curr_snap["full_deck"])

    # Concise summary badges & feed bullets
    badges: list[str] = []
    summary: list[str] = []

    if "location" in changed_fields:
        badges.append("location")
        summary.append(
            f"Location: {changed_fields['location']['prev']} -> {changed_fields['location']['curr']}"
        )
    if "ante" in changed_fields or "round" in changed_fields:
        badges.append("blind")
    if "current_blind" in changed_fields:
        badges.append("blind")
        summary.append(
            f"Blind: {changed_fields['current_blind']['prev']} -> {changed_fields['current_blind']['curr']}"
        )
    if "money" in changed_fields:
        badges.append("money")
        summary.append(
            f"Money: ${changed_fields['money']['prev']} -> ${changed_fields['money']['curr']}"
        )
    if "current_chips" in changed_fields:
        badges.append("score")
        summary.append(
            f"Score: {changed_fields['current_chips']['prev']} -> {changed_fields['current_chips']['curr']}"
        )
    if "hands_left" in changed_fields:
        summary.append(
            f"Hands: {changed_fields['hands_left']['prev']} -> {changed_fields['hands_left']['curr']}"
        )
    if "discards_left" in changed_fields:
        summary.append(
            f"Discards: {changed_fields['discards_left']['prev']} -> {changed_fields['discards_left']['curr']}"
        )
    if jokers_diff["changed"]:
        badges.append("jokers")
        curr_j_by_id = {j["id"]: j for j in curr_snap["jokers"] if j.get("id")}
        for jid in jokers_diff.get("added_ids", []):
            if jid in curr_j_by_id:
                summary.append(f"+Joker: {curr_j_by_id[jid]['name']}")
        for rj in jokers_diff.get("removed", []):
            summary.append(f"-Joker: {rj['name']}")
        for jid, udiffs in jokers_diff.get("updated", {}).items():
            jname = (
                curr_j_by_id.get(int(jid), {}).get("name", f"Joker #{jid}")
                if jid.isdigit()
                else f"Joker #{jid}"
            )
            summary.append(f"~Joker {jname}: {', '.join(udiffs.keys())}")
        if jokers_diff.get("reordered") and not (
            jokers_diff.get("added_ids")
            or jokers_diff.get("removed")
            or jokers_diff.get("updated")
        ):
            summary.append("Jokers reordered")
    if consumables_diff["changed"]:
        badges.append("consumables")
        for ci in consumables_diff.get("added_indices", []):
            summary.append(f"+Consumable: {curr_snap['consumables'][ci].get('name')}")
        for rc in consumables_diff.get("removed", []):
            summary.append(f"-Consumable: {rc.get('name')}")
    if hand_diff["changed"]:
        badges.append("hand")
    if (
        shop_cards_diff["changed"]
        or shop_vouchers_diff["changed"]
        or shop_packs_diff["changed"]
    ):
        badges.append("shop")
    if pack_cards_diff["changed"] or "active_pack" in changed_fields:
        badges.append("pack")
    if active_tags_diff["changed"]:
        badges.append("tags")
        for t in active_tags_diff["added"]:
            summary.append(f"+Tag: {t}")
        for t in active_tags_diff["removed"]:
            summary.append(f"-Tag: {t}")
    if vouchers_redeemed_diff["changed"]:
        badges.append("vouchers")
        for v in vouchers_redeemed_diff["added"]:
            summary.append(f"+Voucher: {v}")
    if poker_hands_diff:
        badges.append("poker_hands")
        for hname, hdiff in poker_hands_diff.items():
            if "level" in hdiff:
                summary.append(
                    f"{hname}: Lvl {hdiff['level']['prev']} -> {hdiff['level']['curr']}"
                )
            if "play_count" in hdiff:
                summary.append(
                    f"{hname}: Played {hdiff['play_count']['prev']} -> {hdiff['play_count']['curr']}"
                )
    if (
        full_deck_diff["added_ids"]
        or full_deck_diff["removed"]
        or full_deck_diff["modified"]
        or full_deck_diff["zone_changed"]
    ):
        badges.append("deck")
        if full_deck_diff["added_ids"]:
            summary.append(f"+{len(full_deck_diff['added_ids'])} card(s) to deck")
        if full_deck_diff["removed"]:
            summary.append(f"-{len(full_deck_diff['removed'])} card(s) from deck")
        if full_deck_diff["modified"]:
            summary.append(f"~{len(full_deck_diff['modified'])} deck card(s) modified")

    has_changes = bool(changed_fields or badges)
    return {
        "has_changes": has_changes,
        "changed_fields": changed_fields,
        "jokers": jokers_diff,
        "consumables": consumables_diff,
        "hand": hand_diff,
        "shop_cards": shop_cards_diff,
        "shop_vouchers": shop_vouchers_diff,
        "shop_packs": shop_packs_diff,
        "pack_cards": pack_cards_diff,
        "active_tags": active_tags_diff,
        "vouchers_redeemed": vouchers_redeemed_diff,
        "poker_hands": poker_hands_diff,
        "full_deck": full_deck_diff,
        "badges": badges,
        "summary_items": summary,
    }


def build_visualization_payload(
    data: dict[str, Any],
    *,
    video_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Builds the full step-indexed JSON payload consumed by the visualizer UI."""
    traj: RunTrajectory = rebuild_trajectory(data, keyframe_interval=20)
    num_steps = len(traj.deltas)

    steps: list[dict[str, Any]] = []
    run_stats: list[dict[str, Any]] = []
    prev_raw_state: GameState | None = None
    prev_snap: dict[str, Any] | None = None
    first: dict[str, Any] = {}
    last: dict[str, Any] = {}
    for i in range(num_steps):
        raw_state = traj.get_state(i)
        snap = _serialize_state_snapshot(raw_state)
        if i == 0:
            first = snap
        last = snap
        delta = traj.deltas[i]
        act_dict = _resolve_action_with_target(
            delta.action, prev_snap, snap, cost_state=prev_raw_state or raw_state
        )
        step_diff = _compute_step_diff(prev_snap, snap)
        if act_dict is not None and prev_snap is not None:
            _attach_hand_exit_breakdown(step_diff["hand"], act_dict, prev_snap, snap)
        steps.append(
            {
                "step_idx": snap["step_idx"],
                "frame_idx": snap["frame_idx"],
                "timestamp": snap["timestamp"],
                "action": act_dict,
                "state": snap,
                "diff": step_diff,
            }
        )
        act_meta = (act_dict.get("metadata") or {}) if act_dict else {}
        warn_reasons = list(act_meta.get("corruption_reasons") or [])
        run_stats.append(
            {
                "step_idx": snap["step_idx"],
                "frame_idx": snap["frame_idx"],
                "ante": snap["ante"],
                "round": snap["round"],
                "location": snap["location"],
                "money": snap["money"],
                "current_chips": snap["current_chips"],
                "target_chips": snap["target_chips"],
                "deck_size": snap["full_deck_count"],
                "joker_count": len(snap["jokers"]),
                "action_type": act_dict["type"] if act_dict else "Init",
                "has_changes": step_diff["has_changes"],
                "corrupted": bool(act_meta.get("corrupted")) or bool(warn_reasons),
                "warning_reasons": warn_reasons,
            }
        )
        prev_raw_state = raw_state
        prev_snap = snap
    return {
        "schema_version": data.get("schema_version"),
        "num_steps": num_steps,
        "run_meta": {
            "seed": first.get("seed", ""),
            "deck": first.get("deck", "Red Deck"),
            "stake": first.get("stake", "White Stake"),
            "final_ante": last.get("ante", 1),
            "final_round": last.get("round", 1),
            "final_location": last.get("location", "blind"),
            "final_blind": last.get("current_blind", "Small Blind"),
            "final_money": last.get("money", 0),
            "deck_size": last.get("full_deck_count", 52),
            "joker_count": len(last.get("jokers", [])),
            "vouchers_redeemed": last.get("vouchers_redeemed", []),
        },
        "video": video_info or {"available": False},
        "run_stats": run_stats,
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# Video Discovery & Thread-Safe LRU JPEG Frame Cache
# ---------------------------------------------------------------------------


def find_matching_video(
    trajectory_path: Path,
    explicit_video: str | Path | None = None,
    video_dir: Path = DEFAULT_VIDEOS_DIR,
) -> Path | None:
    """Matches a trajectory file to its corresponding source MP4 video."""
    if explicit_video is not None:
        p = Path(explicit_video)
        return p.resolve() if p.is_file() else None

    stem = trajectory_path.stem
    if stem.endswith("_trajectory"):
        stem = stem[: -len("_trajectory")]
    # Strip post_processing epoch suffix like `_1794699311`
    stripped_ts = re.sub(r"_\d{8,}$", "", stem)

    search_dirs = [trajectory_path.parent, video_dir]
    candidates = [stem, stripped_ts]
    for sdir in search_dirs:
        if not sdir.is_dir():
            continue
        for cand in candidates:
            for ext in VIDEO_EXTENSIONS:
                vp = sdir / f"{cand}{ext}"
                if vp.is_file():
                    return vp.resolve()

    # Fallback prefix match against videos in `video_dir`
    if video_dir.is_dir():
        videos = [
            p for p in video_dir.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS
        ]
        videos.sort(key=lambda p: len(p.stem), reverse=True)
        for vp in videos:
            if stripped_ts.startswith(vp.stem) or vp.stem.startswith(stripped_ts):
                return vp.resolve()
    return None


def discover_trajectories(
    traj_dir: Path = DEFAULT_TRAJECTORIES_DIR,
    video_dir: Path = DEFAULT_VIDEOS_DIR,
) -> list[dict[str, Any]]:
    """Lists available trajectory JSONs sorted newest-first."""
    if not traj_dir.is_dir():
        return []
    files = sorted(
        (p for p in traj_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    items: list[dict[str, Any]] = []
    for p in files:
        matched = find_matching_video(p, video_dir=video_dir)
        items.append(
            {
                "name": p.name,
                "path": str(p.resolve()),
                "size_kb": round(p.stat().st_size / 1024.0, 1),
                "video": matched.name if matched else None,
            }
        )
    return items


def discover_telemetry_sources(
    traj_dir: Path = DEFAULT_TRAJECTORIES_DIR,
    extra_paths: Sequence[Path | None] = (),
) -> list[dict[str, Any]]:
    """Lists available telemetry `.jsonl` and `*_gt_trajectory.json` comparison sources."""
    candidates: list[Path] = []
    for p in extra_paths:
        if p is not None and p.is_file():
            candidates.append(p)
    for tdir in DEFAULT_TELEMETRY_DIRS:
        if tdir.is_dir():
            candidates.extend(
                p
                for p in tdir.iterdir()
                if p.is_file() and p.suffix.lower() == ".jsonl"
            )
    if traj_dir.is_dir():
        candidates.extend(
            p
            for p in traj_dir.iterdir()
            if p.is_file() and p.name.lower().endswith("_gt_trajectory.json")
        )
    appdata = os.environ.get("APPDATA")
    if appdata:
        balatro_log = Path(appdata) / "Balatro" / "run_telemetry.jsonl"
        if balatro_log.is_file():
            candidates.append(balatro_log)
    root_log = REPO_ROOT / "run_telemetry.jsonl"
    if root_log.is_file():
        candidates.append(root_log)

    seen: set[Path] = set()
    unique: list[Path] = []
    for p in candidates:
        try:
            rp = p.resolve()
        except OSError:
            continue
        if rp not in seen and rp.is_file():
            seen.add(rp)
            unique.append(rp)

    unique.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    items: list[dict[str, Any]] = []
    for rp in unique:
        kind = "jsonl" if rp.suffix.lower() == ".jsonl" else "gt_json"
        try:
            rel_label = str(rp.relative_to(REPO_ROOT)).replace("\\", "/")
        except ValueError:
            rel_label = rp.name
        items.append(
            {
                "name": rp.name,
                "label": rel_label,
                "path": str(rp),
                "kind": kind,
                "size_kb": round(rp.stat().st_size / 1024.0, 1),
            }
        )
    return items


class VideoFrameCache:
    """Thread-safe OpenCV VideoCapture wrapper with priority preloading & LRU cache."""

    def __init__(self, video_path: Path | None, max_items: int = 512) -> None:
        self.video_path = video_path
        self.max_items = max(1, max_items)
        self._lock = threading.Lock()
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        self._pinned_frames: set[int] = set()
        self._cap: Any = None
        self._last_pos: int | None = None
        self._preload_stop: threading.Event | None = None
        self._preload_thread: threading.Thread | None = None
        self._preload_priority_targets: list[int] = []
        self._preload_all_targets: list[int] = []
        self._preload_history: list[int] = []
        self._preload_done: bool = True
        self.meta: dict[str, Any] = {
            "available": False,
            "path": None,
            "filename": None,
            "cache_key": uuid.uuid4().hex[:12],
        }
        if video_path is not None and video_path.is_file():
            self._init_capture(video_path)

    def _init_capture(self, video_path: Path) -> None:
        try:
            import cv2  # type: ignore
        except ImportError:
            self.meta = {"available": False, "error": "opencv-python not installed"}
            return
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.meta = {
                "available": False,
                "error": f"Failed to open {video_path.name}",
            }
            return
        self._cap = cap
        self.meta = {
            "available": True,
            "path": str(video_path),
            "filename": video_path.name,
            "cache_key": uuid.uuid4().hex[:12],
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 60.0),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        }

    def _clamp_frame(self, frame_idx: int) -> int:
        total = self.meta.get("total_frames") or 0
        return (
            max(0, min(int(frame_idx), total - 1))
            if total > 0
            else max(0, int(frame_idx))
        )

    def _evict_if_needed(self) -> None:
        """Evicts oldest non-pinned cached frame while over `max_items` (caller holds `_lock`)."""
        while len(self._cache) > self.max_items:
            evict_key = next(
                (k for k in self._cache if k not in self._pinned_frames), None
            )
            if evict_key is None:
                self._cache.popitem(last=False)
            else:
                self._cache.pop(evict_key)

    def get_frame_jpeg(self, frame_idx: int, quality: int = 88) -> bytes | None:
        """Reads and JPEG-encodes `frame_idx` cleanly with LRU caching."""
        with self._lock:
            if self._cap is None:
                return None
            clamped = self._clamp_frame(frame_idx)
            cached = self._cache.get(clamped)
            if cached is not None:
                self._cache.move_to_end(clamped)
                return cached

            import cv2  # type: ignore

            if self._last_pos != clamped:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, clamped)
            ok, frame = self._cap.read()
            if not ok or frame is None:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, clamped)
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    self._last_pos = None
                    return None
            self._last_pos = clamped + 1
            ok_enc, buf = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            )
            if not ok_enc:
                return None
            jpg_bytes = buf.tobytes()
            self._cache[clamped] = jpg_bytes
            self._evict_if_needed()
            return jpg_bytes

    def stop_preload(self) -> None:
        """Signals any running background preloader to stop and waits for exit."""
        stop_ev = self._preload_stop
        thread = self._preload_thread
        if stop_ev is not None:
            stop_ev.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)
        self._preload_stop = None
        self._preload_thread = None

    def start_preload(
        self,
        priority_frames: Sequence[int],
        secondary_frames: Sequence[int] = (),
        *,
        quality: int = 88,
    ) -> None:
        """Starts background JPEG pre-decoding with `priority_frames` decoded first."""
        self.stop_preload()
        if not self.meta.get("available") or self.video_path is None:
            with self._lock:
                self._preload_priority_targets = []
                self._preload_all_targets = []
                self._preload_history = []
                self._preload_done = True
            return

        seen: set[int] = set()
        priority: list[int] = []
        for f in priority_frames:
            cf = self._clamp_frame(f)
            if cf not in seen:
                seen.add(cf)
                priority.append(cf)
        secondary: list[int] = []
        for f in secondary_frames:
            cf = self._clamp_frame(f)
            if cf not in seen:
                seen.add(cf)
                secondary.append(cf)

        ordered = priority + secondary
        with self._lock:
            self._pinned_frames = set(priority)
            self.max_items = max(self.max_items, len(ordered) + 64)
            self._preload_priority_targets = priority
            self._preload_all_targets = ordered
            self._preload_history = []
            self._preload_done = len(ordered) == 0

        if not ordered:
            return

        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._preload_worker,
            args=(stop_event, ordered, quality),
            daemon=True,
            name="VideoFramePreloader",
        )
        self._preload_stop = stop_event
        self._preload_thread = thread
        thread.start()

    def _preload_worker(
        self,
        stop_event: threading.Event,
        ordered_frames: list[int],
        quality: int,
    ) -> None:
        try:
            import cv2  # type: ignore
        except ImportError:
            with self._lock:
                self._preload_done = True
            return

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            with self._lock:
                self._preload_done = True
            return

        last_pos: int | None = None
        try:
            for clamped in ordered_frames:
                if stop_event.is_set():
                    return
                with self._lock:
                    already = clamped in self._cache
                if already:
                    with self._lock:
                        self._preload_history.append(clamped)
                    continue

                if last_pos != clamped:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, clamped)
                ok, frame = cap.read()
                if (not ok or frame is None) and not stop_event.is_set():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, clamped)
                    ok, frame = cap.read()
                if stop_event.is_set():
                    return
                if not ok or frame is None:
                    last_pos = None
                    continue

                last_pos = clamped + 1
                ok_enc, buf = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                )
                if not ok_enc:
                    continue
                jpg_bytes = buf.tobytes()
                with self._lock:
                    if clamped not in self._cache:
                        self._cache[clamped] = jpg_bytes
                        self._evict_if_needed()
                    self._preload_history.append(clamped)
        finally:
            cap.release()
            with self._lock:
                if not stop_event.is_set():
                    self._preload_done = True

    def preload_status(self) -> dict[str, Any]:
        with self._lock:
            p_total = len(self._preload_priority_targets)
            p_cached = sum(
                1 for f in self._preload_priority_targets if f in self._cache
            )
            a_total = len(self._preload_all_targets)
            a_cached = sum(1 for f in self._preload_all_targets if f in self._cache)
            return {
                "available": bool(self.meta.get("available")),
                "priority_cached": p_cached,
                "priority_total": p_total,
                "total_cached": a_cached,
                "total_targets": a_total,
                "done": self._preload_done,
                "cached_count": len(self._cache),
                "decoded_order": list(self._preload_history),
            }

    def close(self) -> None:
        self.stop_preload()
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            self._cache.clear()
            self._pinned_frames.clear()


# ---------------------------------------------------------------------------
# HTTP Server & Live Trajectory Switching
# ---------------------------------------------------------------------------


def _parse_range_header(range_header: str, file_size: int) -> tuple[int, int] | None:
    """Parses a single HTTP `Range: bytes=...` header into inclusive `(start, end)`."""
    if not range_header or "=" not in range_header:
        return None
    unit, _, spec = range_header.strip().partition("=")
    if unit.lower() != "bytes" or "," in spec:
        return None
    start_s, _, end_s = spec.strip().partition("-")
    try:
        if start_s == "":
            if end_s == "":
                return None
            suffix_len = int(end_s)
            if suffix_len <= 0 or file_size <= 0:
                return None
            start = max(0, file_size - suffix_len)
            end = file_size - 1
        else:
            start = int(start_s)
            end = int(end_s) if end_s else file_size - 1
    except ValueError:
        return None
    if start < 0 or end < start or start >= file_size:
        return None
    return start, min(end, file_size - 1)


class _VisualizerState:
    """Thread-safe holder for the active trajectory payload, manifest, comparison, and video cache."""

    def __init__(
        self,
        *,
        trajectory_path: Path | None,
        video_path: Path | None,
        traj_dir: Path,
        video_dir: Path,
        telemetry_path: Path | None = None,
        telemetry_seed: str | None = None,
        telemetry_run_index: int | None = None,
    ) -> None:
        self.lock = threading.Lock()
        self.traj_dir = traj_dir
        self.video_dir = video_dir
        self.telemetry_path: Path | None = (
            telemetry_path.resolve() if telemetry_path is not None else None
        )
        self.telemetry_seed = telemetry_seed
        self.telemetry_run_index = telemetry_run_index
        self.manifest = load_balatro_sprite_manifest()
        self.trajectory_path: Path | None = None
        self._raw_trajectory_data: dict[str, Any] | None = None
        self.frame_cache: VideoFrameCache | None = None
        self.comparison: dict[str, Any] | None = None
        self.comparison_source: str | None = None
        self.payload: dict[str, Any] = {
            "schema_version": None,
            "num_steps": 0,
            "run_meta": {},
            "video": {"available": False},
            "run_stats": [],
            "steps": [],
            "comparison": None,
        }
        if trajectory_path is not None and trajectory_path.is_file():
            should_compare = (
                self.telemetry_path is not None and self.telemetry_path.is_file()
            )
            self.load_trajectory(
                trajectory_path,
                explicit_video=video_path,
                start_preload=not should_compare,
            )
            if should_compare:
                self.run_comparison(
                    self.telemetry_path,
                    seed=self.telemetry_seed,
                    run_index=self.telemetry_run_index,
                )

    def _trigger_frame_preload(self) -> None:
        with self.lock:
            cache = self.frame_cache
            steps = self.payload.get("steps", [])
            comp = self.comparison
        if cache is None or not cache.meta.get("available"):
            return
        err_indices = (
            set(comp.get("summary", {}).get("error_step_indices", []))
            if comp
            else set()
        )
        priority: list[int] = []
        secondary: list[int] = []
        for idx, st in enumerate(steps):
            frame_idx = int(
                st.get("frame_idx") or st.get("state", {}).get("frame_idx") or 0
            )
            if idx in err_indices:
                priority.append(frame_idx)
            else:
                secondary.append(frame_idx)
        cache.start_preload(priority, secondary)

    def load_trajectory(
        self,
        traj_path: Path,
        explicit_video: Path | str | None = None,
        *,
        start_preload: bool = True,
    ) -> dict[str, Any]:
        traj_path = traj_path.resolve()
        data = json.loads(traj_path.read_text(encoding="utf-8"))
        matched_video = find_matching_video(
            traj_path, explicit_video, video_dir=self.video_dir
        )
        new_cache = VideoFrameCache(matched_video)
        new_payload = build_visualization_payload(data, video_info=new_cache.meta)
        new_payload["trajectory_name"] = traj_path.name
        new_payload["trajectory_path"] = str(traj_path)
        new_payload["comparison"] = None

        with self.lock:
            old_cache = self.frame_cache
            self.trajectory_path = traj_path
            self._raw_trajectory_data = data
            self.frame_cache = new_cache
            self.payload = new_payload
            self.comparison = None
            self.comparison_source = None
        if old_cache is not None:
            old_cache.close()
        if start_preload:
            self._trigger_frame_preload()
        return new_payload

    def run_comparison(
        self,
        source_path: Path | None,
        *,
        seed: str | None = None,
        run_index: int | None = None,
    ) -> dict[str, Any] | None:
        if source_path is None:
            with self.lock:
                self.comparison = None
                self.comparison_source = None
                self.payload["comparison"] = None
            self._trigger_frame_preload()
            return None

        resolved = source_path.resolve()
        with self.lock:
            raw_data = self._raw_trajectory_data
            traj_path = self.trajectory_path
        actual_source = raw_data if raw_data is not None else traj_path
        if actual_source is None:
            raise ValueError("No trajectory is currently loaded.")

        from tools.compare_trajectories import compare_trajectories

        report = compare_trajectories(
            actual_source,
            resolved,
            seed=seed,
            run_index=run_index,
        )
        report["source_path"] = str(resolved)
        report["source_name"] = resolved.name

        with self.lock:
            self.comparison = report
            self.comparison_source = str(resolved)
            self.payload["comparison"] = report
        self._trigger_frame_preload()
        return report

    def list_telemetry_sources(self) -> dict[str, Any]:
        extra = [self.telemetry_path]
        if self.trajectory_path is not None:
            p_dir = self.trajectory_path.parent
            if p_dir.is_dir():
                for cand in p_dir.iterdir():
                    if cand.is_file() and (
                        cand.suffix.lower() == ".jsonl"
                        or cand.name.lower().endswith("_gt_trajectory.json")
                    ):
                        extra.append(cand)
        items = discover_telemetry_sources(self.traj_dir, extra_paths=extra)
        with self.lock:
            active_source = self.comparison_source
        return {"active_source": active_source, "items": items}

    def get_video_path(self) -> Path | None:
        with self.lock:
            cache = self.frame_cache
            return cache.video_path if cache is not None else None

    def list_trajectories(self) -> dict[str, Any]:
        items = discover_trajectories(self.traj_dir, self.video_dir)
        with self.lock:
            active_name = self.trajectory_path.name if self.trajectory_path else None
            active_path = str(self.trajectory_path) if self.trajectory_path else None
        return {"active_name": active_name, "active_path": active_path, "items": items}

    def close(self) -> None:
        with self.lock:
            cache = self.frame_cache
            self.frame_cache = None
        if cache is not None:
            cache.close()


def _resolve_compare_source(
    raw_source: str,
    app_state: _VisualizerState,
) -> tuple[Path | None, int, str | None]:
    clean = raw_source.strip()
    if clean.lower() in ("", "none", "off", "clear"):
        return None, 200, None

    if ".." in Path(clean).parts:
        return None, 403, "Forbidden telemetry source path"

    allowed_roots: list[Path] = [REPO_ROOT.resolve(), app_state.traj_dir.resolve()]
    with app_state.lock:
        if app_state.trajectory_path is not None:
            allowed_roots.append(app_state.trajectory_path.parent.resolve())
        startup_tel = app_state.telemetry_path
    appdata = os.environ.get("APPDATA")
    if appdata:
        allowed_roots.append((Path(appdata) / "Balatro").resolve())

    discovered_paths = {
        Path(it["path"]).resolve() for it in app_state.list_telemetry_sources()["items"]
    }
    if startup_tel is not None:
        discovered_paths.add(startup_tel.resolve())

    cand = Path(clean)
    if not cand.is_file() and not cand.is_absolute():
        search_candidates = [REPO_ROOT / clean, app_state.traj_dir / clean]
        for sdir in (*DEFAULT_TELEMETRY_DIRS, app_state.traj_dir):
            search_candidates.append(sdir / cand.name)
        for sc in search_candidates:
            if sc.is_file():
                cand = sc
                break

    try:
        resolved = cand.resolve()
    except OSError:
        return None, 400, f"Invalid telemetry path: {raw_source}"

    if resolved.suffix.lower() not in (".jsonl", ".json"):
        return None, 403, "Forbidden source file extension"

    is_allowed = resolved in discovered_paths
    if not is_allowed:
        for root in allowed_roots:
            try:
                resolved.relative_to(root)
                is_allowed = True
                break
            except ValueError:
                continue
    if not is_allowed:
        return None, 403, "Forbidden telemetry source path"
    if not resolved.is_file():
        return None, 404, f"Telemetry or GT file not found: {raw_source}"
    return resolved, 200, None


def _make_request_handler(app_state: _VisualizerState, *, verbose: bool = False):
    class VisualizerHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            if verbose:
                super().log_message(format, *args)

        def _send_bytes(
            self,
            status: int,
            content_type: str,
            body: bytes,
            *,
            cache_control: str = "no-store",
        ) -> None:
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", cache_control)
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                self.close_connection = True

        def _send_json(self, status: int, obj: Any) -> None:
            raw = json.dumps(obj, separators=(",", ":")).encode("utf-8")
            self._send_bytes(status, "application/json; charset=utf-8", raw)

        def _serve_video(self, *, send_body: bool) -> None:
            video_path = app_state.get_video_path()
            if video_path is None or not video_path.is_file():
                if send_body:
                    self._send_json(
                        404, {"error": "No video loaded for active trajectory"}
                    )
                else:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                return

            file_size = video_path.stat().st_size
            ctype, _ = mimetypes.guess_type(str(video_path))
            content_type = ctype or "video/mp4"
            range_hdr = self.headers.get("Range")

            if range_hdr:
                parsed_range = _parse_range_header(range_hdr, file_size)
                if parsed_range is None:
                    self.send_response(416)
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Range", f"bytes */{file_size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                start, end = parsed_range
                length = end - start + 1
                self.send_response(206)
                self.send_header("Content-Type", content_type)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(length))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
            else:
                start = 0
                length = file_size
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(length))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()

            if not send_body or length == 0:
                return

            try:
                with video_path.open("rb") as vf:
                    vf.seek(start)
                    remaining = length
                    while remaining > 0:
                        chunk = vf.read(min(_VIDEO_STREAM_CHUNK_BYTES, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return

        def do_HEAD(self) -> None:
            path = unquote(urlparse(self.path).path)
            if path == "/api/video":
                self._serve_video(send_body=False)
                return
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            qs = parse_qs(parsed.query)

            if path == "/" or path == "/index.html":
                html_bytes = _render_html_page().encode("utf-8")
                self._send_bytes(200, "text/html; charset=utf-8", html_bytes)
                return

            if path == "/api/manifest":
                self._send_json(200, app_state.manifest)
                return

            if path == "/api/trajectories":
                self._send_json(200, app_state.list_trajectories())
                return

            if path == "/api/telemetry-sources":
                self._send_json(200, app_state.list_telemetry_sources())
                return

            if path == "/api/compare":
                raw_src = qs.get("source", [""])[0]
                seed_arg = qs.get("seed", [None])[0]
                run_idx_raw = qs.get("run_index", [None])[0]
                run_idx_arg = (
                    int(run_idx_raw) if run_idx_raw and run_idx_raw.isdigit() else None
                )
                resolved, status, err_msg = _resolve_compare_source(raw_src, app_state)
                if status != 200:
                    self._send_json(
                        status, {"error": err_msg or "Invalid comparison request"}
                    )
                    return
                try:
                    report = app_state.run_comparison(
                        resolved, seed=seed_arg, run_index=run_idx_arg
                    )
                except Exception as exc:  # noqa: BLE001
                    self._send_json(500, {"error": str(exc)})
                    return
                self._send_json(200, {"comparison": report})
                return

            if path == "/api/preload-status":
                with app_state.lock:
                    cache = app_state.frame_cache
                if cache is None:
                    self._send_json(
                        200,
                        {
                            "available": False,
                            "priority_cached": 0,
                            "priority_total": 0,
                            "total_cached": 0,
                            "total_targets": 0,
                            "done": True,
                            "cached_count": 0,
                        },
                    )
                    return
                status_payload = cache.preload_status()
                status_payload.pop("decoded_order", None)
                self._send_json(200, status_payload)
                return

            if path == "/api/video":
                self._serve_video(send_body=True)
                return

            if path == "/api/trajectory":
                req_name = qs.get("name", [None])[0] or qs.get("path", [None])[0]
                if req_name:
                    candidate = Path(req_name)
                    if not candidate.is_file():
                        candidate = app_state.traj_dir / Path(req_name).name
                    if not candidate.is_file():
                        self._send_json(
                            404, {"error": f"Trajectory not found: {req_name}"}
                        )
                        return
                    try:
                        payload = app_state.load_trajectory(candidate)
                    except Exception as exc:  # noqa: BLE001
                        self._send_json(500, {"error": str(exc)})
                        return
                    self._send_json(200, payload)
                    return

                with app_state.lock:
                    payload = app_state.payload
                self._send_json(200, payload)
                return

            if path.startswith("/api/frame/"):
                idx_str = path[len("/api/frame/") :].strip("/")
                try:
                    frame_idx = int(idx_str)
                except ValueError:
                    self._send_json(400, {"error": f"Invalid frame index: {idx_str}"})
                    return
                with app_state.lock:
                    cache = app_state.frame_cache
                if cache is None or not cache.meta.get("available"):
                    self._send_json(
                        404, {"error": "No video loaded for active trajectory"}
                    )
                    return
                jpg = cache.get_frame_jpeg(frame_idx)
                if jpg is None:
                    self._send_json(
                        404, {"error": f"Failed to decode frame {frame_idx}"}
                    )
                    return
                self._send_bytes(
                    200, "image/jpeg", jpg, cache_control="no-store"
                )
                return

            if path.startswith("/textures/"):
                rel = path[len("/textures/") :].lstrip("/")
                if not rel or ".." in Path(rel).parts:
                    self._send_json(403, {"error": "Forbidden texture path"})
                    return
                base = TEXTURES_2X_DIR.resolve()
                target = (base / rel).resolve()
                try:
                    target.relative_to(base)
                except ValueError:
                    self._send_json(403, {"error": "Forbidden texture path"})
                    return
                if not target.is_file():
                    self._send_json(404, {"error": f"Texture not found: {rel}"})
                    return
                ctype, _ = mimetypes.guess_type(str(target))
                self._send_bytes(
                    200,
                    ctype or "image/png",
                    target.read_bytes(),
                    cache_control="public, max-age=86400",
                )
                return

            self._send_json(404, {"error": f"Not found: {path}"})

    return VisualizerHandler


class _QuietThreadingHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that suppresses noisy client-disconnect socket errors."""

    # On Windows, SO_REUSEADDR allows a second server to bind the same port concurrently;
    # disabling it guarantees port-in-use detection so a new visualizer never collides on 8765.
    allow_reuse_address = os.name != "nt"

    def handle_error(self, request: Any, client_address: Any) -> None:
        _, exc, _ = sys.exc_info()
        if isinstance(
            exc, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)
        ):
            return
        super().handle_error(request, client_address)


class TrajectoryVisualizerServer:
    """Wrapper around `ThreadingHTTPServer` for programmatic / CLI lifecycle control."""

    def __init__(
        self,
        *,
        trajectory_path: Path | None = None,
        video_path: Path | None = None,
        traj_dir: Path = DEFAULT_TRAJECTORIES_DIR,
        video_dir: Path = DEFAULT_VIDEOS_DIR,
        telemetry_path: Path | None = None,
        telemetry_seed: str | None = None,
        telemetry_run_index: int | None = None,
        host: str = "127.0.0.1",
        port: int = 8765,
        verbose: bool = False,
    ) -> None:
        self.app_state = _VisualizerState(
            trajectory_path=trajectory_path,
            video_path=video_path,
            traj_dir=traj_dir,
            video_dir=video_dir,
            telemetry_path=telemetry_path,
            telemetry_seed=telemetry_seed,
            telemetry_run_index=telemetry_run_index,
        )
        handler = _make_request_handler(self.app_state, verbose=verbose)
        self.httpd = _QuietThreadingHTTPServer((host, port), handler)
        self.httpd.daemon_threads = True
        self.host, self.port = self.httpd.server_address[:2]

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def serve_forever(self) -> None:
        try:
            self.httpd.serve_forever()
        finally:
            self.close()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.app_state.close()


# ---------------------------------------------------------------------------
# Embedded Interactive Single-Page Frontend
# ---------------------------------------------------------------------------


def _render_html_page() -> str:
    """Returns the standalone responsive HTML/CSS/JS trajectory visualizer UI."""
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Balatro Trajectory Visualizer</title>
<style>
  :root {
    --bg: #0c1018;
    --panel: #151c28;
    --panel-alt: #1b2434;
    --card-bg: #222d40;
    --border: #2d3b55;
    --text: #e6edf7;
    --muted: #8ea0be;
    --gold: #f59e0b;
    --gold-bg: rgba(245, 158, 11, 0.16);
    --green: #22c55e;
    --green-bg: rgba(34, 197, 94, 0.16);
    --red: #ef4444;
    --red-bg: rgba(239, 68, 68, 0.18);
    --blue: #38bdf8;
    --blue-bg: rgba(56, 189, 248, 0.16);
    --purple: #c084fc;
    --chip-color: #3b82f6;
    --mult-color: #ef4444;
  }

  * { box-sizing: border-box; }
  html, body {
    margin: 0; padding: 0; height: 100vh; width: 100vw;
    overflow: hidden; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 13px;
  }

  .app-shell {
    display: grid;
    grid-template-rows: auto 1fr auto;
    height: 100vh;
    width: 100vw;
  }

  /* Header Bar */
  .topbar {
    display: flex; align-items: center; justify-content: space-between;
    min-height: 46px; padding: 4px 12px; gap: 8px; flex-wrap: wrap; row-gap: 4px;
    background: #101622; border-bottom: 1px solid var(--border);
  }
  .topbar-group { display: flex; align-items: center; gap: 7px; min-width: 0; flex-wrap: wrap; }
  .brand { font-weight: 700; letter-spacing: 0.4px; color: var(--gold); white-space: nowrap; }
  select, input[type="number"], input[type="text"], button {
    background: var(--panel-alt); color: var(--text); border: 1px solid var(--border);
    border-radius: 5px; padding: 4px 8px; font-size: 12px; font-family: inherit;
  }
  button { cursor: pointer; font-weight: 600; transition: background 0.12s, border-color 0.12s; }
  button:hover { background: #28354d; border-color: #4b6189; }
  button.primary { background: #1d4ed8; border-color: #3b82f6; }
  button.primary:hover { background: #2563eb; }
  button.accent { background: rgba(245,158,11,0.18); border-color: var(--gold); color: #fde68a; }
  button.active-toggle { background: rgba(56,189,248,0.22); border-color: var(--blue); color: #bae6fd; }

  .meta-pill {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 2px 7px; border-radius: 4px; background: var(--panel-alt);
    border: 1px solid var(--border); font-size: 12px; white-space: nowrap;
  }
  .seed-btn { font-family: monospace; color: #93c5fd; cursor: pointer; }

  /* 3-Column Body */
  .main-grid {
    display: grid;
    grid-template-columns: 270px 1fr 440px;
    min-height: 0;
    overflow: hidden;
  }

  .col-left, .col-center, .col-right {
    display: flex; flex-direction: column; min-height: 0; overflow: hidden;
  }
  .col-left {
    background: var(--panel); border-right: 1px solid var(--border);
    padding: 8px; gap: 8px; overflow-y: auto;
  }
  .col-center {
    background: radial-gradient(circle at 50% 35%, #162925 0%, #0d1616 70%, #0b1018 100%);
    padding: 8px 10px; gap: 8px; overflow-y: auto;
  }
  .col-right {
    background: var(--panel); border-left: 1px solid var(--border);
    padding: 8px; gap: 8px;
  }

  .section-box {
    background: rgba(15, 21, 32, 0.72); border: 1px solid var(--border);
    border-radius: 6px; padding: 7px 9px;
  }
  .section-title {
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.7px;
    color: var(--muted); font-weight: 700; margin-bottom: 5px;
    display: flex; align-items: center; justify-content: space-between;
  }

  /* Change Highlighting */
  .changed-glow {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 1px var(--gold), 0 0 10px rgba(245, 158, 11, 0.35);
    background-color: rgba(245, 158, 11, 0.08);
  }
  .added-glow {
    border-color: var(--green) !important;
    box-shadow: 0 0 0 1px var(--green), 0 0 10px rgba(34, 197, 94, 0.35);
  }
  .mod-glow {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 1px var(--blue), 0 0 10px rgba(56, 189, 248, 0.35);
  }
  .badge {
    display: inline-flex; align-items: center; gap: 3px;
    padding: 1px 5px; border-radius: 3px; font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.3px;
  }
  .badge-gold { background: var(--gold-bg); color: #fcd34d; border: 1px solid rgba(245,158,11,0.45); }
  .badge-green { background: var(--green-bg); color: #86efac; border: 1px solid rgba(34,197,94,0.45); }
  .badge-red { background: var(--red-bg); color: #fca5a5; border: 1px solid rgba(239,68,68,0.45); }
  .badge-blue { background: var(--blue-bg); color: #7dd3fc; border: 1px solid rgba(56,189,248,0.45); }
  .badge-purple { background: rgba(192,132,252,0.18); color: #e9d5ff; border: 1px solid rgba(192,132,252,0.45); }
  .badge-muted { background: #1e293b; color: #cbd5e1; border: 1px solid #334155; }

  .delta-pill {
    font-size: 10px; font-weight: 700; color: #fde68a; background: rgba(245,158,11,0.22);
    border: 1px solid rgba(245,158,11,0.5); border-radius: 3px; padding: 0 4px; margin-left: 4px;
    font-variant-numeric: tabular-nums;
  }

  /* Left HUD Stats */
  .blind-header {
    display: flex; align-items: center; gap: 9px; margin-bottom: 6px;
  }
  .score-bar-track {
    height: 8px; background: #0c1018; border-radius: 4px; overflow: hidden;
    border: 1px solid var(--border); margin: 4px 0 6px 0;
  }
  .score-bar-fill {
    height: 100%; background: linear-gradient(90deg, #3b82f6, #f59e0b); transition: width 0.2s;
  }
  .stat-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 5px;
  }
  .stat-cell {
    background: var(--panel-alt); border: 1px solid var(--border); border-radius: 5px;
    padding: 4px 7px; display: flex; flex-direction: column;
  }
  .stat-label { font-size: 10px; color: var(--muted); text-transform: uppercase; }
  .stat-val { font-size: 15px; font-weight: 700; font-variant-numeric: tabular-nums; }
  .val-money { color: #fbbf24; }
  .val-hands { color: #60a5fa; }
  .val-discards { color: #f87171; }

  .kv-mini {
    display: flex; justify-content: space-between; align-items: center;
    padding: 2px 4px; border-radius: 3px; font-size: 12px;
  }
  .kv-mini + .kv-mini { border-top: 1px solid rgba(45, 59, 85, 0.4); }

  /* Sprite Containers */
  .sprite-box {
    position: relative; background-repeat: no-repeat; image-rendering: pixelated; flex-shrink: 0;
  }
  .sprite-layer {
    position: absolute; inset: 0; background-repeat: no-repeat; image-rendering: pixelated; pointer-events: none;
  }
  .debuff-x-overlay {
    position: absolute; inset: 0; pointer-events: none;
    background: rgba(127, 29, 29, 0.35);
    border-radius: 4px;
  }
  .debuff-x-overlay::before, .debuff-x-overlay::after {
    content: ""; position: absolute; left: 50%; top: 50%;
    width: 82%; height: 4px; background: #ef4444;
    box-shadow: 0 0 4px rgba(0, 0, 0, 0.9);
    border-radius: 2px;
  }
  .debuff-x-overlay::before { transform: translate(-50%, -50%) rotate(45deg); }
  .debuff-x-overlay::after { transform: translate(-50%, -50%) rotate(-45deg); }

  /* Cards Row & Shelves */
  .board-shelf {
    display: grid; grid-template-columns: 1.6fr 1fr; gap: 8px; align-items: start;
  }
  .card-tray {
    display: flex; flex-wrap: wrap; gap: 8px; align-items: flex-end; min-height: 106px;
  }
  .hand-tray {
    display: flex; flex-wrap: wrap; gap: 7px; align-items: flex-end; min-height: 120px;
    padding-top: 16px;
  }
  .vis-card {
    position: relative; display: flex; flex-direction: column; align-items: center;
    background: rgba(15, 22, 34, 0.85); border: 1.5px solid var(--border);
    border-radius: 6px; padding: 3px; transition: transform 0.14s, border-color 0.14s;
    cursor: pointer; user-select: none;
  }
  .vis-card:hover { transform: translateY(-3px); border-color: #64748b; z-index: 4; }
  .vis-card.selected {
    transform: translateY(-14px);
    border-color: var(--gold);
    box-shadow: 0 6px 14px rgba(245,158,11,0.35);
  }
  .vis-card.ghost-removed {
    opacity: 0.62; border: 1.5px dashed var(--red); background: rgba(127, 29, 29, 0.2);
  }
  .card-caption {
    font-size: 10px; font-weight: 600; max-width: 78px; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis; text-align: center; margin-top: 2px;
  }
  .card-sub {
    font-size: 9px; color: var(--muted); display: flex; gap: 3px; flex-wrap: wrap; justify-content: center;
  }
  .tag-corner {
    position: absolute; top: -5px; right: -5px; z-index: 5; font-size: 9px; font-weight: 800;
    padding: 0 4px; border-radius: 3px; box-shadow: 0 1px 4px rgba(0,0,0,0.6);
  }
  .chip-pill {
    position: absolute; bottom: 20px; left: 3px; z-index: 4;
    font-size: 9px; font-weight: 800; padding: 0 3px; border-radius: 3px;
    background: rgba(30, 64, 175, 0.9); color: #dbeafe; border: 1px solid #60a5fa;
  }

  /* Edition effects */
  .ed-foil { box-shadow: inset 0 0 12px rgba(96, 165, 250, 0.75); border-color: #60a5fa; }
  .ed-holographic { box-shadow: inset 0 0 12px rgba(244, 114, 182, 0.8); border-color: #f472b6; }
  .ed-polychrome {
    box-shadow: inset 0 0 14px rgba(168, 85, 247, 0.85), 0 0 8px rgba(52, 211, 153, 0.5);
    border-color: #e879f9;
  }
  .ed-negative {
    filter: invert(0.88) hue-rotate(180deg);
    border-color: #e2e8f0;
  }

  /* Right Column Video + Tabs */
  .video-card {
    background: #070a10; border: 1px solid var(--border); border-radius: 6px; overflow: hidden;
    display: flex; flex-direction: column;
  }
  .video-stage {
    position: relative; width: 100%; aspect-ratio: 16 / 9; background: #05070b;
    display: flex; align-items: center; justify-content: center; overflow: hidden; cursor: zoom-in;
  }
  .video-stage.video-mode { cursor: default; }
  .video-stage img, .video-stage video { width: 100%; height: 100%; object-fit: contain; display: block; background: #000; }
  .video-toolbar {
    display: flex; align-items: center; justify-content: space-between; gap: 6px;
    padding: 4px 8px; background: #101622; border-top: 1px solid var(--border); font-size: 11px;
  }
  .preload-pill {
    font-size: 10px; font-weight: 600; padding: 1px 6px; border-radius: 3px;
    background: var(--panel-alt); border: 1px solid var(--border); color: var(--muted); white-space: nowrap;
  }
  .preload-pill.ready { background: var(--green-bg); border-color: rgba(34,197,94,0.45); color: #86efac; }

  .err-pill-ok { background: var(--green-bg); border-color: rgba(34,197,94,0.5); color: #86efac; }
  .err-pill-bad { background: var(--red-bg); border-color: rgba(239,68,68,0.65); color: #fecaca; font-weight: 700; }
  .err-nav-btn { background: rgba(127,29,29,0.35); border-color: rgba(239,68,68,0.55); color: #fecaca; }
  .err-nav-btn:hover { background: rgba(185,28,28,0.55); border-color: #ef4444; }
  .tab-btn.has-errors { border-color: rgba(239,68,68,0.65); color: #fca5a5; }
  .tab-btn.has-errors.active { background: #5c1922; border-color: #ef4444; color: #fff; }
  .err-banner {
    margin-top: 6px; padding: 6px 8px; border-radius: 5px;
    background: rgba(127, 29, 29, 0.25); border: 1px solid rgba(239, 68, 68, 0.55);
    font-size: 11.5px; color: #fecaca;
  }
  .err-row {
    padding: 6px 8px; border: 1px solid var(--border); border-radius: 5px;
    background: var(--panel-alt); margin-bottom: 5px; cursor: pointer;
    transition: border-color 0.12s, background 0.12s;
  }
  .err-row:hover { border-color: #ef4444; background: rgba(127, 29, 29, 0.22); }
  .err-row.active { border-color: var(--red); background: rgba(239, 68, 68, 0.16); box-shadow: 0 0 0 1px rgba(239,68,68,0.4); }
  .err-chip {
    padding: 2px 7px; border-radius: 12px; font-size: 10.5px; font-weight: 700;
    background: var(--panel-alt); border: 1px solid var(--border); color: var(--muted);
  }
  .err-chip.active { background: rgba(239,68,68,0.22); border-color: var(--red); color: #fecaca; }

  .tab-bar { display: flex; gap: 4px; border-bottom: 1px solid var(--border); padding-bottom: 4px; }
  .tab-btn {
    flex: 1; padding: 4px 6px; font-size: 11px; background: var(--panel-alt);
    border: 1px solid var(--border); border-radius: 4px; color: var(--muted);
  }
  .tab-btn.active { background: #1e3a5f; color: #fff; border-color: var(--blue); }
  .tab-body { flex: 1; min-height: 0; overflow-y: auto; }

  table.compact { width: 100%; border-collapse: collapse; font-size: 11.5px; }
  table.compact th, table.compact td {
    padding: 3px 5px; border-bottom: 1px solid rgba(45, 59, 85, 0.5); text-align: left;
  }
  table.compact th { color: var(--muted); font-size: 10.5px; text-transform: uppercase; position: sticky; top: 0; background: var(--panel); }
  tr.row-changed { background: rgba(245, 158, 11, 0.14); }

  /* Bottom Timeline Scrubber */
  .timeline-bar {
    background: #101622; border-top: 1px solid var(--border);
    padding: 4px 12px; display: flex; flex-direction: column; justify-content: center; gap: 3px;
  }
  .diag-track {
    position: relative; height: 11px; background: #080c13; border: 1px solid var(--border);
    border-radius: 3px; cursor: pointer; display: flex; align-items: stretch; overflow: hidden;
  }
  .diag-tick {
    position: relative; flex: 1; height: 100%; min-width: 1px; border-right: 1px solid rgba(0,0,0,0.25);
    display: flex; flex-direction: column;
  }
  .diag-seg-err { flex: 1; width: 100%; background: #ef4444; box-shadow: 0 0 4px rgba(239, 68, 68, 0.85); }
  .diag-seg-warn { flex: 1; width: 100%; background: #f97316; box-shadow: 0 0 4px rgba(249, 115, 22, 0.75); }
  .diag-seg-cascade { flex: 1; width: 100%; background: rgba(249, 115, 22, 0.32); }
  .scrubber-track {
    position: relative; height: 24px; background: #0a0e16; border: 1px solid var(--border);
    border-radius: 4px; cursor: pointer; display: flex; align-items: flex-end; overflow: hidden;
  }
  .scrub-tick {
    position: relative; flex: 1; height: 100%; min-width: 1px; border-right: 1px solid rgba(0,0,0,0.25);
    display: flex; align-items: flex-end;
  }
  .scrub-tick-bar { width: 100%; transition: opacity 0.1s; }
  .scrub-err-mark {
    position: absolute; top: 0; left: 0; right: 0; height: 5px;
    background: #ef4444; box-shadow: 0 0 4px rgba(239, 68, 68, 0.9);
    pointer-events: none; z-index: 2;
  }
  .scrub-playhead {
    position: absolute; top: 0; bottom: 0; width: 3px; background: #fff;
    box-shadow: 0 0 6px #fff; pointer-events: none; z-index: 4;
  }

  /* Modals (Deck Visualizer & Video Zoom) */
  .modal-backdrop {
    position: fixed; inset: 0; background: rgba(6, 9, 15, 0.86); z-index: 50;
    display: none; align-items: center; justify-content: center; padding: 18px;
  }
  .modal-backdrop.open { display: flex; }
  .modal-window {
    width: 94vw; height: 90vh; background: var(--panel); border: 1px solid #475569;
    border-radius: 10px; display: flex; flex-direction: column; overflow: hidden;
    box-shadow: 0 18px 48px rgba(0,0,0,0.75);
  }
  .modal-header {
    padding: 10px 14px; background: #101622; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap;
  }
  .modal-body { flex: 1; min-height: 0; overflow-y: auto; padding: 12px; }
  .deck-group-title {
    font-size: 12px; font-weight: 700; color: var(--muted); text-transform: uppercase;
    margin: 8px 0 6px 0; border-bottom: 1px solid var(--border); padding-bottom: 3px;
    display: flex; align-items: center; justify-content: space-between;
  }
  .deck-grid { display: flex; flex-wrap: wrap; gap: 7px; }
  .zone-dot {
    font-size: 9px; font-weight: 700; padding: 0 4px; border-radius: 3px; margin-top: 1px;
  }
  .zone-in_draw_pile { background: #1e293b; color: #93c5fd; }
  .zone-in_hand { background: rgba(34,197,94,0.22); color: #86efac; }
  .zone-played_round { background: rgba(245,158,11,0.22); color: #fde68a; }
  .zone-discarded { background: rgba(148,163,184,0.2); color: #cbd5e1; }
</style>
</head>
<body>
<div class="app-shell">
  <!-- TOP CONTROL BAR -->
  <header class="topbar">
    <div class="topbar-group">
      <span class="brand">BALATRO TRAJECTORY</span>
      <select id="trajSelect" title="Switch trajectory JSON" style="max-width:210px;"></select>
      <div id="runMetaBadges" class="topbar-group"></div>
    </div>

    <div class="topbar-group">
      <select id="telemetrySelect" title="Compare active run against ground-truth telemetry (.jsonl) or GT trajectory (.json)" style="max-width:175px;">
        <option value="">Compare: None</option>
      </select>
      <span id="compareStatusPill" class="meta-pill" style="display:none; cursor:pointer;" title="Open Errors inspector tab"></span>
      <button id="btnPrevErr" class="err-nav-btn" style="display:none;" title="Jump to previous test error ([)">&lt; Err</button>
      <button id="btnNextErr" class="err-nav-btn" style="display:none;" title="Jump to next test error (])">Err &gt;</button>
      <select id="filterSelect" title="Filter Prev/Next navigation">
        <option value="all">Filter: All Steps</option>
        <option value="test_errors">Test Errors Only</option>
        <option value="changed">Changed Only</option>
        <option value="play_discard">Play / Discard</option>
        <option value="shop_pack">Shop / Pack</option>
        <option value="blind_round">Blind / Round</option>
        <option value="deck_mod">Deck Modified</option>
      </select>
      <button id="btnFirst" title="First step (Home)">|&lt;&lt;</button>
      <button id="btnPrev" title="Previous step (Left Arrow)">&lt; Prev</button>
      <button id="btnPlay" class="primary" title="Play / Pause (Space)">Play</button>
      <button id="btnNext" title="Next step (Right Arrow)">Next &gt;</button>
      <button id="btnLast" title="Last step (End)">&gt;&gt;|</button>
      <select id="speedSelect" title="Playback speed">
        <option value="1200">0.5x</option>
        <option value="650" selected>1x</option>
        <option value="320">2x</option>
        <option value="140">4x</option>
      </select>
      <span class="meta-pill">
        Step <input id="stepInput" type="number" min="0" value="0" style="width:54px; padding:1px 4px;" />
        / <span id="stepTotal">0</span>
      </span>
      <button id="btnDeckModal" class="accent" title="Open Full Deck Visualizer (D)">Deck (0)</button>
    </div>
  </header>

  <!-- 3-COLUMN WORKSPACE -->
  <main class="main-grid">
    <!-- LEFT COLUMN: HUD & CHANGES -->
    <aside class="col-left">
      <div id="hudPanel" class="section-box"></div>
      <div id="targetsPanel" class="section-box"></div>
      <div id="tagsVouchersPanel" class="section-box"></div>
      <div id="changeFeedPanel" class="section-box" style="flex:1; min-height:110px;"></div>
    </aside>

    <!-- CENTER COLUMN: VISUAL BOARD -->
    <section class="col-center">
      <div class="board-shelf">
        <div id="jokersPanel" class="section-box"></div>
        <div id="consumablesPanel" class="section-box"></div>
      </div>

      <div id="middleStagePanel" class="section-box"></div>

      <div class="section-box" style="display:flex; gap:10px; align-items:flex-end; justify-content:space-between;">
        <div id="handPanel" style="flex:1; min-width:0;"></div>
        <div id="deckCornerPanel" style="flex-shrink:0;"></div>
      </div>
    </section>

    <!-- RIGHT COLUMN: VIDEO + ACTION + INSPECTOR TABS -->
    <aside class="col-right">
      <div class="video-card">
        <div id="videoStage" class="video-stage" title="Click to enlarge video frame">
          <span id="videoPlaceholder" style="color:var(--muted);">No video matched</span>
          <img id="videoImg" alt="VOD frame" style="display:none;" />
          <video id="videoPlayer" controls preload="metadata" style="display:none;"></video>
        </div>
        <div class="video-toolbar">
          <div style="display:flex; align-items:center; gap:6px; min-width:0; overflow:hidden;">
            <span id="videoCaption" style="white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">Frame -</span>
            <span id="preloadBadge" class="preload-pill" style="display:none;"></span>
          </div>
          <div style="display:flex; align-items:center; gap:4px; flex-shrink:0;">
            <button id="btnVideoMode" title="Toggle between frame screenshots and HTML5 video player (V)">Video Player</button>
            <button id="btnSyncVideo" style="display:none;" title="Seek video player to current step timestamp">Sync</button>
            <span id="frameNudgeGroup" style="display:flex; gap:4px;">
              <button data-nudge="-5">-5f</button>
              <button data-nudge="-1">-1f</button>
              <button data-nudge="0">Reset</button>
              <button data-nudge="1">+1f</button>
              <button data-nudge="5">+5f</button>
            </span>
          </div>
        </div>
      </div>

      <div id="actionPanel" class="section-box"></div>

      <div class="section-box" style="flex:1; min-height:0; display:flex; flex-direction:column; gap:6px;">
        <div class="tab-bar">
          <button class="tab-btn active" data-tab="diff">Changes</button>
          <button class="tab-btn" data-tab="hands">Poker Hands</button>
          <button class="tab-btn" data-tab="piles">Piles</button>
          <button class="tab-btn" data-tab="raw">Modifiers / Raw</button>
          <button class="tab-btn" data-tab="errors" id="tabBtnErrors">Errors</button>
        </div>
        <div id="tabBody" class="tab-body"></div>
      </div>
    </aside>
  </main>

  <!-- BOTTOM SCRUB TIMELINE -->
  <footer class="timeline-bar">
    <div id="diagTrack" class="diag-track" title="Errors (red) &amp; Expressive Warnings (orange)"></div>
    <div id="scrubberTrack" class="scrubber-track"></div>
    <div style="display:flex; justify-content:space-between; font-size:11px; color:var(--muted);">
      <span id="timelineHoverInfo">Hover or drag timeline to scrub steps &nbsp;|&nbsp; Shortcuts: &larr;/&rarr; Step, [ / ] Err, V Video, Space Play, D Deck, 1-5 Tabs</span>
      <span id="timelineLocationLegend">
        <span id="legendErrItem" style="color:#ef4444; display:none;">&#9632; Test Error &nbsp;</span>
        <span id="legendWarnItem" style="color:#f97316; display:none;">&#9632; Warning &nbsp;</span>
        <span style="color:#60a5fa;">&#9632; Blind</span> &nbsp;
        <span style="color:#f59e0b;">&#9632; Shop</span> &nbsp;
        <span style="color:#c084fc;">&#9632; Pack</span> &nbsp;
        <span style="color:#22c55e;">&#9632; Select</span>
      </span>
    </div>
  </footer>
</div>

<!-- DECK VISUALIZER MODAL -->
<div id="deckModal" class="modal-backdrop">
  <div class="modal-window">
    <div class="modal-header">
      <div style="display:flex; align-items:center; gap:10px;">
        <strong style="font-size:15px; color:var(--gold);">Deck Visualizer</strong>
        <span id="deckModalSummary" class="muted"></span>
      </div>
      <div style="display:flex; align-items:center; gap:6px; flex-wrap:wrap;">
        <select id="deckZoneFilter">
          <option value="all">Zone: All Cards</option>
          <option value="in_draw_pile">Draw Pile</option>
          <option value="in_hand">In Hand</option>
          <option value="played_round">Played Round</option>
          <option value="discarded">Discarded</option>
          <option value="modified_cards">Has Mod (Enh/Seal/Ed/+Chips)</option>
          <option value="step_changed">Changed This Step</option>
        </select>
        <select id="deckGroupBy">
          <option value="suit">Group: Suit</option>
          <option value="zone">Group: Zone</option>
          <option value="enhancement">Group: Enhancement</option>
          <option value="none">Flat List (Sort by ID)</option>
        </select>
        <input id="deckSearch" type="text" placeholder="Filter rank, seal, ed..." style="width:160px;" />
        <button id="btnCloseDeckModal">Close (Esc)</button>
      </div>
    </div>
    <div id="deckStatsStrip" style="padding:7px 14px; background:var(--panel-alt); border-bottom:1px solid var(--border); display:flex; gap:10px; flex-wrap:wrap; align-items:center;"></div>
    <div id="deckModalBody" class="modal-body"></div>
    <div id="deckInspectorBar" style="padding:8px 14px; background:#101622; border-top:1px solid var(--border); font-size:12px; color:var(--muted);">
      Hover over any card to inspect full attributes &amp; step diff.
    </div>
  </div>
</div>

<!-- VIDEO ZOOM LIGHTBOX -->
<div id="videoModal" class="modal-backdrop">
  <div style="position:relative; max-width:94vw; max-height:92vh; background:#05070b; border:1px solid #475569; border-radius:8px; padding:8px;">
    <img id="videoModalImg" alt="Enlarged VOD frame" style="max-width:90vw; max-height:84vh; display:block;" />
    <div style="display:flex; justify-content:space-between; align-items:center; padding-top:6px;">
      <span id="videoModalCaption"></span>
      <button id="btnCloseVideoModal">Close (Esc)</button>
    </div>
  </div>
</div>

<script>
const APP = {
  manifest: null,
  payload: null,
  trajectories: null,
  telemetrySources: null,
  stepIdx: 0,
  frameOffset: 0,
  activeTab: "diff",
  errorCategoryFilter: "all",
  videoMode: "frame",
  activeVideoKey: "",
  playTimer: null,
  preloadTimer: null,
};

function esc(v) {
  if (v === null || v === undefined) return "";
  return String(v).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function fmtNum(n) {
  if (n === null || n === undefined) return "—";
  return Number(n).toLocaleString();
}

function spriteStyle(pos, widthPx, heightPx) {
  if (!pos || !APP.manifest || !APP.manifest.sheets[pos.sheet]) return "";
  const sh = APP.manifest.sheets[pos.sheet];
  const pxPct = sh.cols > 1 ? (pos.x / (sh.cols - 1)) * 100 : 0;
  const pyPct = sh.rows > 1 ? (pos.y / (sh.rows - 1)) * 100 : 0;
  return `width:${widthPx}px;height:${heightPx}px;background-image:url('/textures/${pos.sheet}');background-size:${sh.cols * 100}% ${sh.rows * 100}%;background-position:${pxPct.toFixed(4)}% ${pyPct.toFixed(4)}%;`;
}

function renderSprite(pos, w, h, extraClass = "", extraLayersHtml = "") {
  return `<div class="sprite-box ${extraClass}" style="${spriteStyle(pos, w, h)}">${extraLayersHtml}</div>`;
}

function renderPlayingCardSprite(card, w, h, { facingBack = false } = {}) {
  const m = APP.manifest;
  if (!m) return "";
  if (facingBack) {
    const deckName = APP.payload?.run_meta?.deck || "Red Deck";
    const backPos = m.backs[deckName] || m.backs["Red Deck"];
    return renderSprite(backPos, w, h);
  }
  const enhKey = (card.enhancement || "none").toLowerCase();
  const basePos = m.enhancements[enhKey] || m.enhancements["none"];
  let layers = "";

  if (enhKey !== "stone") {
    const grid = m.playing_card_grid;
    const suitLower = (card.suit || "none").toLowerCase();
    const rankStr = String(card.rank || "none");
    if (rankStr !== "none" && suitLower !== "none" && grid.ranks[rankStr] !== undefined && grid.suits[suitLower] !== undefined) {
      const facePos = { sheet: grid.sheet, x: grid.ranks[rankStr], y: grid.suits[suitLower] };
      layers += `<div class="sprite-layer" style="${spriteStyle(facePos, w, h)}"></div>`;
    }
  }
  const sealKey = (card.seal || "none").toLowerCase();
  if (sealKey !== "none" && m.seals[sealKey]) {
    layers += `<div class="sprite-layer" style="${spriteStyle(m.seals[sealKey], w, h)}"></div>`;
  }
  if (card.debuffed) {
    layers += `<div class="debuff-x-overlay" title="Debuffed"></div>`;
  }
  const ed = (card.edition || "none").toLowerCase();
  const edClass = ed !== "none" ? `ed-${ed}` : "";
  return renderSprite(basePos, w, h, edClass, layers);
}

function lookupCardSprite(item) {
  const m = APP.manifest;
  if (!m || !item) return null;
  const t = item.type;
  const name = item.name;
  if (t === "joker" || (!t && name && m.jokers[name])) return { kind: "joker", pos: m.jokers[name] };
  if (t === "tarot" || (!t && name && m.tarots[name])) return { kind: "consumable", pos: m.tarots[name] };
  if (t === "planet" || (!t && name && m.planets[name])) return { kind: "consumable", pos: m.planets[name] };
  if (t === "spectral" || (!t && name && m.spectrals[name])) return { kind: "consumable", pos: m.spectrals[name] };
  if (t === "voucher" && name && m.vouchers[name]) return { kind: "voucher", pos: m.vouchers[name] };
  if (t === "booster_pack") {
    const key = `${item.size} ${item.booster_type}`;
    return { kind: "pack", pos: m.boosters[key] };
  }
  if (t === "playing_card" || item.rank !== undefined) return { kind: "playing_card", card: item };
  return null;
}

function renderJokerCardHtml(j, { cls = "", cornerBadge = "", isGhost = false } = {}) {
  const m = APP.manifest;
  const pos = m.jokers[j.name] || m.jokers["Joker"];
  const w = 68, h = 91;
  let layers = "";
  if (pos && pos.soul) {
    layers += `<div class="sprite-layer" style="${spriteStyle(pos.soul, w, h)}"></div>`;
  }
  if (j.eternal && m.stickers.eternal) layers += `<div class="sprite-layer" style="${spriteStyle(m.stickers.eternal, w, h)}"></div>`;
  if (j.perishable && m.stickers.perishable) layers += `<div class="sprite-layer" style="${spriteStyle(m.stickers.perishable, w, h)}"></div>`;
  if (j.rental && m.stickers.rental) layers += `<div class="sprite-layer" style="${spriteStyle(m.stickers.rental, w, h)}"></div>`;
  if (j.debuffed) layers += `<div class="debuff-x-overlay" title="Debuffed"></div>`;
  const ed = (j.edition || "none").toLowerCase();
  const edCls = ed !== "none" ? `ed-${ed}` : "";
  const stateEntries = Object.entries(j.state || {});
  const stateStr = stateEntries.length ? stateEntries.map(([k, v]) => `${k}:${v}`).join(" ") : "";
  const perishBadge = j.perishable ? `<span class="badge badge-purple">${j.perish_tally}/5</span>` : "";
  const edBadge = ed !== "none" ? `<span class="badge badge-blue">${ed}</span>` : "";
  const tooltip = esc(JSON.stringify(j));
  return `<div class="vis-card ${cls} ${isGhost ? "ghost-removed" : ""}" title="${tooltip}">
    ${cornerBadge}
    ${renderSprite(pos, w, h, edCls, layers)}
    <div class="card-caption" title="${esc(j.name)}">${esc(j.name)}</div>
    <div class="card-sub">
      <span style="color:#fbbf24;">$${j.sell_value ?? "?"}</span>
      ${edBadge}${perishBadge}
      ${stateStr ? `<span class="badge badge-gold">${esc(stateStr)}</span>` : ""}
    </div>
  </div>`;
}

function renderConsumableOrShopHtml(item, { cls = "", cornerBadge = "", showBuyCost = false, isGhost = false } = {}) {
  const m = APP.manifest;
  const w = 64, h = 86;
  const spec = lookupCardSprite(item);
  let spriteHtml = "";
  if (spec && spec.kind === "playing_card") {
    spriteHtml = renderPlayingCardSprite(item, w, h);
  } else if (spec && spec.pos) {
    const ed = (item.edition || "none").toLowerCase();
    let layers = "";
    if (spec.pos.soul) layers += `<div class="sprite-layer" style="${spriteStyle(spec.pos.soul, w, h)}"></div>`;
    if (item.eternal && m.stickers.eternal) layers += `<div class="sprite-layer" style="${spriteStyle(m.stickers.eternal, w, h)}"></div>`;
    if (item.perishable && m.stickers.perishable) layers += `<div class="sprite-layer" style="${spriteStyle(m.stickers.perishable, w, h)}"></div>`;
    if (item.rental && m.stickers.rental) layers += `<div class="sprite-layer" style="${spriteStyle(m.stickers.rental, w, h)}"></div>`;
    if (item.debuffed) layers += `<div class="debuff-x-overlay" title="Debuffed"></div>`;
    spriteHtml = renderSprite(spec.pos, w, h, ed !== "none" ? `ed-${ed}` : "", layers);
  } else {
    spriteHtml = `<div style="width:${w}px;height:${h}px;background:#1e293b;display:flex;align-items:center;justify-content:center;font-size:10px;padding:4px;text-align:center;">${esc(item.name || item.type)}</div>`;
  }
  const label = item.name || (item.booster_type ? `${item.size} ${item.booster_type}` : `${item.rank || ""}${item.suit || ""}`);
  const costVal = showBuyCost ? item.buy_cost : item.sell_value;
  const ed = (item.edition || "none").toLowerCase();
  const perishBadge = item.perishable ? `<span class="badge badge-purple">${item.perish_tally ?? 5}/5</span>` : "";
  const eternalBadge = item.eternal ? `<span class="badge badge-red">ETERNAL</span>` : "";
  const rentalBadge = item.rental ? `<span class="badge badge-gold">RENTAL</span>` : "";
  return `<div class="vis-card ${cls} ${isGhost ? "ghost-removed" : ""}" title="${esc(JSON.stringify(item))}">
    ${cornerBadge}
    ${spriteHtml}
    <div class="card-caption" title="${esc(label)}">${esc(label)}</div>
    <div class="card-sub">
      ${costVal !== undefined ? `<span style="color:#fbbf24;">$${costVal}</span>` : ""}
      ${ed !== "none" ? `<span class="badge badge-blue">${esc(ed)}</span>` : ""}
      ${eternalBadge}${perishBadge}${rentalBadge}
      ${item.couponed ? `<span class="badge badge-green">FREE</span>` : ""}
    </div>
  </div>`;
}

function renderHandCardGhostHtml(card, cornerText, badgeCls = "badge-red") {
  const c = card || {};
  const w = 70, h = 94;
  const spriteHtml = renderPlayingCardSprite(c, w, h);
  const label = c.enhancement === "stone" ? "Stone" :
    `${c.rank || "?"}${({spades:"♠",hearts:"♥",clubs:"♣",diamonds:"♦"})[c.suit] || ""}`;
  return `<div class="vis-card ghost-removed" title="${esc(JSON.stringify(c))}">
    <span class="tag-corner ${badgeCls}">${esc(cornerText)}</span>
    ${spriteHtml}
    <div class="card-caption">${esc(label)}</div>
  </div>`;
}

function renderHandCardHtml(hc, { cls = "", cornerBadge = "" } = {}) {
  const c = hc.card || {};
  const w = 70, h = 94;
  const spriteHtml = renderPlayingCardSprite(c, w, h, { facingBack: hc.facing_back });
  const label = hc.facing_back ? `Hidden (${hc.candidate_count ?? "?"})` :
    (c.enhancement === "stone" ? "Stone" : `${c.rank || "?"}${({spades:"♠",hearts:"♥",clubs:"♣",diamonds:"♦"})[c.suit] || ""}`);
  const badges = [];
  if (c.enhancement && c.enhancement !== "none" && c.enhancement !== "stone") badges.push(`<span class="badge badge-gold">${esc(c.enhancement)}</span>`);
  if (c.edition && c.edition !== "none") badges.push(`<span class="badge badge-blue">${esc(c.edition)}</span>`);
  if (c.seal && c.seal !== "none") badges.push(`<span class="badge badge-purple">${esc(c.seal)}</span>`);
  if (hc.forced_selection) badges.push(`<span class="badge badge-red">FORCED</span>`);
  const chipHtml = c.bonus_chips ? `<span class="chip-pill">+${c.bonus_chips}</span>` : "";
  return `<div class="vis-card ${hc.selected ? "selected" : ""} ${cls}" title="${esc(JSON.stringify(hc))}">
    ${cornerBadge}
    ${chipHtml}
    ${spriteHtml}
    <div class="card-caption">${esc(label)}</div>
    <div class="card-sub">${badges.join("")}</div>
  </div>`;
}

function renderSmallDeckCardHtml(pc, { cls = "", cornerBadge = "", extraSub = "", onHoverData = "" } = {}) {
  const w = 62, h = 83;
  const spriteHtml = renderPlayingCardSprite(pc, w, h);
  const label = pc.enhancement === "stone" ? "Stone" :
    `${pc.rank}${({spades:"♠",hearts:"♥",clubs:"♣",diamonds:"♦"})[pc.suit] || ""}`;
  const chipHtml = pc.bonus_chips ? `<span class="chip-pill">+${pc.bonus_chips}</span>` : "";
  return `<div class="vis-card ${cls}" ${onHoverData}>
    ${cornerBadge}${chipHtml}
    ${spriteHtml}
    <div class="card-caption">${esc(label)} <span style="color:var(--muted);font-weight:400;">#${pc.id || "?"}</span></div>
    ${extraSub}
  </div>`;
}

function getErrorStepIndices() {
  return APP.payload?.comparison?.summary?.error_step_indices || [];
}

function getStepDiscrepancies(stepIdx) {
  return APP.payload?.comparison?.steps_by_actual_idx?.[String(stepIdx)] || [];
}

function setActiveTab(tabName) {
  APP.activeTab = tabName;
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.toggle("active", b.dataset.tab === tabName));
  const st = APP.payload?.steps?.[APP.stepIdx];
  if (st) renderActiveTab(st);
}

function renderComparisonChrome() {
  const comp = APP.payload?.comparison;
  const pill = document.getElementById("compareStatusPill");
  const prevErrBtn = document.getElementById("btnPrevErr");
  const nextErrBtn = document.getElementById("btnNextErr");
  const tabErrBtn = document.getElementById("tabBtnErrors");
  const legendErr = document.getElementById("legendErrItem");

  if (!comp) {
    pill.style.display = "none";
    prevErrBtn.style.display = "none";
    nextErrBtn.style.display = "none";
    legendErr.style.display = "none";
    tabErrBtn.textContent = "Errors";
    tabErrBtn.classList.remove("has-errors");
    return;
  }

  const sm = comp.summary || {};
  const errCount = sm.error_step_count || 0;
  pill.style.display = "inline-flex";
  if (errCount === 0) {
    pill.className = "meta-pill err-pill-ok";
    pill.textContent = "✓ 0 Errors";
    prevErrBtn.style.display = "none";
    nextErrBtn.style.display = "none";
    legendErr.style.display = "none";
    tabErrBtn.textContent = "Errors (0)";
    tabErrBtn.classList.remove("has-errors");
  } else {
    pill.className = "meta-pill err-pill-bad";
    pill.textContent = `✕ ${errCount} Error Step${errCount === 1 ? "" : "s"}`;
    prevErrBtn.style.display = "inline-block";
    nextErrBtn.style.display = "inline-block";
    legendErr.style.display = "inline";
    tabErrBtn.textContent = `Errors (${errCount})`;
    tabErrBtn.classList.add("has-errors");
  }
}

function stepErrorRelative(delta) {
  const errs = getErrorStepIndices();
  if (!errs.length) return;
  if (delta > 0) {
    const next = errs.find(idx => idx > APP.stepIdx);
    goToStep(next !== undefined ? next : errs[0]);
  } else {
    const prev = [...errs].reverse().find(idx => idx < APP.stepIdx);
    goToStep(prev !== undefined ? prev : errs[errs.length - 1]);
  }
}

function getFilteredSteps() {
  const steps = APP.payload?.steps || [];
  const mode = document.getElementById("filterSelect").value;
  if (mode === "all") return steps.map((_, i) => i);
  const errSet = new Set(getErrorStepIndices());
  return steps.reduce((acc, st, i) => {
    const t = st.action?.type || "";
    const d = st.diff || {};
    if (mode === "test_errors" && errSet.has(i)) acc.push(i);
    else if (mode === "changed" && d.has_changes) acc.push(i);
    else if (mode === "play_discard" && (t === "Play Hand" || t === "Discard")) acc.push(i);
    else if (mode === "shop_pack" && ["shop", "pack"].includes(st.state.location)) acc.push(i);
    else if (mode === "blind_round" && (["Select Blind", "Skip Blind", "End Round", "Cash Out", "Next Round"].includes(t))) acc.push(i);
    else if (mode === "deck_mod" && (d.full_deck?.added_ids?.length || d.full_deck?.removed?.length || Object.keys(d.full_deck?.modified || {}).length)) acc.push(i);
    return acc;
  }, []);
}

function renderAll() {
  const steps = APP.payload?.steps || [];
  if (!steps.length) return;
  APP.stepIdx = Math.max(0, Math.min(APP.stepIdx, steps.length - 1));
  const step = steps[APP.stepIdx];
  const st = step.state;
  const diff = step.diff;
  const m = APP.manifest;

  document.getElementById("stepInput").value = APP.stepIdx;
  document.getElementById("stepTotal").textContent = Math.max(0, steps.length - 1);
  document.getElementById("btnDeckModal").textContent = `Deck (${st.full_deck_count})`;

  // Top Run Badges (Deck Back sprite + Stake Chip + Seed + Location)
  const deckBack = m.backs[st.deck] || m.backs["Red Deck"];
  const stakeChip = m.stakes[st.stake] || m.stakes["White Stake"];
  document.getElementById("runMetaBadges").innerHTML = `
    <span class="meta-pill">${renderSprite(deckBack, 16, 21)} ${esc(st.deck)}</span>
    <span class="meta-pill">${renderSprite(stakeChip, 18, 18)} ${esc(st.stake)}</span>
    <span class="meta-pill seed-btn" onclick="navigator.clipboard?.writeText('${esc(st.seed)}')" title="Click to copy seed">Seed: ${esc(st.seed || "—")}</span>
  `;

  // LEFT HUD
  const cf = diff.changed_fields || {};
  const dp = (field, prefix = "") => cf[field] ? `<span class="delta-pill">${esc(prefix)}${esc(cf[field].prev)} &rarr; ${esc(prefix)}${esc(cf[field].curr)}</span>` : "";
  const blindName = st.current_blind === "Small Blind" || st.current_blind === "Big Blind"
    ? st.current_blind : (st.boss_blind || st.current_blind);
  const blindPos = m.blinds[blindName] || m.blinds["Small Blind"];
  const pct = st.target_chips ? Math.min(100, Math.round((st.current_chips / st.target_chips) * 100)) : 0;

  document.getElementById("hudPanel").innerHTML = `
    <div class="section-title">
      <span>Run HUD</span>
      <span class="badge ${cf.location ? "badge-gold" : "badge-blue"}">${esc(st.location)}${dp("location")}</span>
    </div>
    <div class="blind-header">
      ${renderSprite(blindPos, 38, 38)}
      <div style="min-width:0; flex:1;">
        <div style="font-weight:700; font-size:13.5px; color:${blindPos?.color || "#fff"};">${esc(blindName)}</div>
        <div style="font-size:11px; color:var(--muted);">
          Ante <strong class="${cf.ante ? "changed-glow" : ""}">${st.ante}</strong> ${dp("ante")} &bull;
          Round <strong class="${cf.round ? "changed-glow" : ""}">${st.round}</strong> ${dp("round")}
          ${st.boss_disabled ? `<span class="badge badge-red">DISABLED</span>` : ""}
          ${st.round_won ? `<span class="badge badge-green">WON</span>` : ""}
        </div>
      </div>
    </div>
    <div class="${cf.current_chips || cf.target_chips ? "changed-glow" : ""}" style="padding:3px 5px; border-radius:4px; border:1px solid transparent;">
      <div style="display:flex; justify-content:space-between; font-size:12px;">
        <span>Chips: <strong>${fmtNum(st.current_chips)}</strong></span>
        <span style="color:var(--muted);">/ ${fmtNum(st.target_chips)} (${pct}%)</span>
      </div>
      ${cf.current_chips ? `<div>${dp("current_chips")}</div>` : ""}
      <div class="score-bar-track"><div class="score-bar-fill" style="width:${pct}%;"></div></div>
    </div>
    <div class="stat-grid">
      <div class="stat-cell ${cf.money ? "changed-glow" : ""}">
        <span class="stat-label">Money ${dp("money", "$")}</span>
        <span class="stat-val val-money">$${st.money ?? "—"}</span>
      </div>
      <div class="stat-cell ${cf.reroll_cost ? "changed-glow" : ""}">
        <span class="stat-label">Reroll ${dp("reroll_cost", "$")}</span>
        <span class="stat-val">$${st.reroll_cost ?? "—"}${st.free_rerolls ? ` <span class="badge badge-green">${st.free_rerolls} FREE</span>` : ""}</span>
      </div>
      <div class="stat-cell ${cf.hands_left ? "changed-glow" : ""}">
        <span class="stat-label">Hands ${dp("hands_left")}</span>
        <span class="stat-val val-hands">${st.hands_left ?? "—"}</span>
      </div>
      <div class="stat-cell ${cf.discards_left ? "changed-glow" : ""}">
        <span class="stat-label">Discards ${dp("discards_left")}</span>
        <span class="stat-val val-discards">${st.discards_left ?? "—"}</span>
      </div>
    </div>
  `;

  // DYNAMIC TARGETS
  const targetRows = [
    ["Idol", `${st.idol_card?.[0]} of ${st.idol_card?.[1]}`, "idol_card"],
    ["Mail", st.mail_card_rank, "mail_card_rank"],
    ["Ancient", st.ancient_suit, "ancient_suit"],
    ["Castle", st.castle_suit, "castle_suit"],
    ["To Do", st.to_do_poker_hand, "to_do_poker_hand"],
  ];
  document.getElementById("targetsPanel").innerHTML = `
    <div class="section-title">Round Joker Targets</div>
    ${targetRows.map(([k, val, f]) => `
      <div class="kv-mini ${cf[f] ? "changed-glow" : ""}">
        <span style="color:var(--muted);">${k}</span>
        <strong>${esc(val)}</strong>
      </div>
    `).join("")}
  `;

  // ACTIVE TAGS & VOUCHERS
  const tagsHtml = st.active_tags.length ? st.active_tags.map(t => `
    <span class="meta-pill ${diff.active_tags?.added?.includes(t) ? "added-glow" : ""}" title="${esc(t)}">
      ${renderSprite(m.tags[t], 20, 20)} <span>${esc(t.replace(" Tag", ""))}</span>
    </span>`).join("") : `<span style="color:var(--muted);font-size:11px;">No active tags</span>`;

  const vouchersHtml = st.vouchers_redeemed.length ? st.vouchers_redeemed.map(v => `
    <span class="meta-pill ${diff.vouchers_redeemed?.added?.includes(v) ? "added-glow" : ""}" title="${esc(v)}">
      ${renderSprite(m.vouchers[v], 18, 24)} <span>${esc(v)}</span>
    </span>`).join("") : `<span style="color:var(--muted);font-size:11px;">No vouchers redeemed</span>`;

  document.getElementById("tagsVouchersPanel").innerHTML = `
    <div class="section-title">Tags (${st.active_tags.length}) &amp; Vouchers (${st.vouchers_redeemed.length})</div>
    <div style="display:flex; flex-wrap:wrap; gap:4px; margin-bottom:5px;">${tagsHtml}</div>
    <div style="display:flex; flex-wrap:wrap; gap:4px;">${vouchersHtml}</div>
  `;

  // STEP CHANGE FEED
  const summaryHtml = diff.summary_items?.length
    ? diff.summary_items.map(s => `<div class="kv-mini" style="color:#fde68a;">&bull; ${esc(s)}</div>`).join("")
    : `<div style="color:var(--muted);font-size:11px;">${APP.stepIdx === 0 ? "Initial run state." : "No scalar or deck delta detected."}</div>`;
  document.getElementById("changeFeedPanel").innerHTML = `
    <div class="section-title">Changed Since Step ${Math.max(0, APP.stepIdx - 1)}</div>
    <div style="display:flex;flex-wrap:wrap;gap:3px;margin-bottom:5px;">
      ${(diff.badges || []).map(b => `<span class="badge badge-gold">${esc(b)}</span>`).join("")}
    </div>
    ${summaryHtml}
  `;

  // CENTER: JOKERS
  const jd = diff.jokers || {};
  const addedJ = new Set(jd.added_ids || []);
  const updJ = jd.updated || {};
  let jokersBody = st.jokers.map(j => {
    let cls = ""; let corner = "";
    if (addedJ.has(j.id)) { cls = "added-glow"; corner = `<span class="tag-corner badge-green">NEW</span>`; }
    else if (updJ[String(j.id)]) { cls = "changed-glow"; corner = `<span class="tag-corner badge-gold">UPD</span>`; }
    return renderJokerCardHtml(j, { cls, cornerBadge: corner });
  }).join("");
  if (jd.removed?.length) {
    jokersBody += jd.removed.map(rj => renderJokerCardHtml(rj, {
      isGhost: true,
      cornerBadge: `<span class="tag-corner badge-red">LOST</span>`,
    })).join("");
  }
  document.getElementById("jokersPanel").innerHTML = `
    <div class="section-title">
      <span>Jokers (${st.jokers.length} / ${st.max_jokers}) ${jd.reordered ? `<span class="badge badge-blue">REORDERED</span>` : ""}</span>
      ${cf.joker_slots || cf.max_jokers ? `<span class="delta-pill">Slots ${st.joker_slots}/${st.max_jokers}</span>` : ""}
    </div>
    <div class="card-tray">${jokersBody || `<span style="color:var(--muted);">No Jokers</span>`}</div>
  `;

  // CENTER: CONSUMABLES
  const cd = diff.consumables || {};
  const addedC = new Set(cd.added_indices || []);
  const updC = cd.updated || {};
  let consBody = st.consumables.map((c, idx) => {
    let cls = ""; let corner = "";
    if (addedC.has(idx)) { cls = "added-glow"; corner = `<span class="tag-corner badge-green">NEW</span>`; }
    else if (updC[String(idx)]) { cls = "changed-glow"; corner = `<span class="tag-corner badge-gold">UPD</span>`; }
    return renderConsumableOrShopHtml(c, { cls, cornerBadge: corner });
  }).join("");
  if (cd.removed?.length) {
    consBody += cd.removed.map(rc => renderConsumableOrShopHtml(rc, {
      isGhost: true, cornerBadge: `<span class="tag-corner badge-red">USED/SOLD</span>`,
    })).join("");
  }
  document.getElementById("consumablesPanel").innerHTML = `
    <div class="section-title"><span>Consumables (${st.consumables.length} / ${st.max_consumables})</span></div>
    <div class="card-tray">${consBody || `<span style="color:var(--muted);">No Consumables</span>`}</div>
  `;

  // CENTER: MIDDLE STAGE (Location adaptive)
  renderMiddleStage(st, diff, step.action);

  // CENTER: HAND & DECK CORNER
  const hd = diff.hand || {};
  const addedH = new Set(hd.added_indices || []);
  const updH = hd.updated || {};
  const playedCards = hd.played_cards || [];
  const discardedCards = hd.discarded_cards || [];
  const isHookRound = step.action?.type === "Play Hand" && st.current_blind === "The Hook" && !st.boss_disabled;
  const discardBadge = isHookRound ? "HOOK DISCARD" : "DISCARDED";
  const handHtml = st.hand.map((hc, idx) => {
    let cls = ""; let corner = "";
    if (addedH.has(idx)) { cls = "added-glow"; corner = `<span class="tag-corner badge-green">DRAWN</span>`; }
    else if (updH[String(idx)]) { cls = "mod-glow"; corner = `<span class="tag-corner badge-blue">CHG</span>`; }
    return renderHandCardHtml(hc, { cls, cornerBadge: corner });
  }).join("")
    + playedCards.map(c => renderHandCardGhostHtml(c, "PLAYED", "badge-blue")).join("")
    + discardedCards.map(c => renderHandCardGhostHtml(c, discardBadge, "badge-red")).join("");

  const exitSummaryPills = [];
  if (playedCards.length) exitSummaryPills.push(`<span class="badge badge-blue">-${playedCards.length} played</span>`);
  if (discardedCards.length) exitSummaryPills.push(`<span class="badge badge-red">-${discardedCards.length} ${isHookRound ? "Hook discarded" : "discarded"}</span>`);
  if (!exitSummaryPills.length && hd.removed?.length) exitSummaryPills.push(`<span class="badge badge-red">-${hd.removed.length} left hand</span>`);

  document.getElementById("handPanel").innerHTML = `
    <div class="section-title">
      <span>Hand After Action (${st.hand.length} / ${st.hand_max_size}) ${hd.reordered ? `<span class="badge badge-blue">SORTED</span>` : ""}</span>
      <span style="display:flex;gap:4px;align-items:center;">${exitSummaryPills.join("")}</span>
    </div>
    <div class="hand-tray">${handHtml || `<span style="color:var(--muted);">Hand empty</span>`}</div>
  `;

  document.getElementById("deckCornerPanel").innerHTML = `
    <div class="vis-card ${cf.draw_pile_count || cf.full_deck_count ? "changed-glow" : ""}" onclick="openDeckModal()" title="Open Interactive Deck Visualizer (D)">
      ${renderSprite(deckBack, 64, 86)}
      <div class="card-caption">${st.draw_pile_count} / ${st.full_deck_count} Draw</div>
      <div class="card-sub">
        <span>Disc: ${st.discard_pile_count}</span>
        <span>Play: ${st.played_round_count}</span>
      </div>
    </div>
  `;

  // RIGHT: VIDEO, ACTION, TABS
  renderComparisonChrome();
  renderVideoFrame(st.frame_idx, st.timestamp);
  renderActionPanel(step.action, APP.stepIdx);
  renderActiveTab(step);
  renderScrubber();

  if (document.getElementById("deckModal").classList.contains("open")) {
    renderDeckModal();
  }
}

function _consumedFromObserved(obsList, currList, keyFn) {
  if (!obsList || !obsList.length) return [];
  const used = new Array(obsList.length).fill(false);
  for (const cur of currList || []) {
    const ck = keyFn(cur);
    const idx = obsList.findIndex((o, i) => !used[i] && keyFn(o) === ck);
    if (idx !== -1) used[idx] = true;
  }
  return obsList.filter((_, i) => !used[i]);
}

function renderMiddleStage(st, diff, act) {
  const panel = document.getElementById("middleStagePanel");
  const m = APP.manifest;
  const shopItemKey = c => {
    if (c.booster_type || c.type === "booster_pack") {
      return `pack|${c.size || ""}|${c.booster_type || ""}`;
    }
    if (c.type === "voucher" || (!c.type && c.name && m?.vouchers?.[c.name])) {
      return `voucher|${c.name || ""}`;
    }
    const ed = (c.edition && String(c.edition).toLowerCase() !== "none") ? String(c.edition).toLowerCase() : "";
    return `card|${c.name || ""}|${ed}|${c.rank || ""}|${c.suit || ""}|${!!c.eternal}|${!!c.perishable}|${!!c.rental}`;
  };

  const renderObservedPackSection = () => {
    const obsPack = act?.observed_pack;
    if (!obsPack || !obsPack.length) return "";
    const pickedIdx = act?.type === "Use In Pack" ? act?.target_idx : null;
    const packHtml = obsPack.map((c, idx) => renderConsumableOrShopHtml(c, {
      isGhost: pickedIdx === idx,
      cornerBadge: pickedIdx === idx ? `<span class="tag-corner badge-red">PICKED</span>` : "",
    })).join("");
    return `
      <div style="margin-bottom:8px; padding-bottom:8px; border-bottom:1px solid var(--border);">
        <div class="section-title">
          <span>Pack Contents (${esc(act?.type || "Pack")})</span>
          <span class="badge badge-purple">${obsPack.length} cards seen</span>
        </div>
        <div class="card-tray" style="min-height:92px;">${packHtml}</div>
      </div>
    `;
  };

  if (st.location === "shop") {
    const scd = diff.shop_cards || {};
    const addedSC = new Set(scd.added_indices || []);
    if (
      act?.type === "Cash Out" &&
      !st.shop_cards.length &&
      !st.shop_vouchers.length &&
      !st.shop_packs.length &&
      !(scd.removed?.length)
    ) {
      panel.innerHTML = `
        <div class="section-title">
          <span>Cash Out &rarr; Entering Shop</span>
          <span class="badge badge-muted">Shelves not visible on Cash Out frame</span>
        </div>
        <div style="font-size:11px; color:var(--muted); padding:6px 0;">
          Cash Out clicked on the round reward screen; Shop offerings are populated on the first shop action frame.
        </div>
      `;
      return;
    }

    const removedCards = scd.removed?.length
      ? scd.removed
      : _consumedFromObserved(act?.observed_shop?.cards, st.shop_cards, shopItemKey);
    const cardGoneLabel = (act?.type === "Buy Card" || act?.type === "Buy and Use") ? "BOUGHT" : "GONE";
    const cardsHtml = st.shop_cards.map((c, idx) => renderConsumableOrShopHtml(c, {
      showBuyCost: true,
      cls: addedSC.has(idx) ? "added-glow" : "",
      cornerBadge: addedSC.has(idx) ? `<span class="tag-corner badge-green">NEW</span>` : "",
    })).join("") + removedCards.map(c => renderConsumableOrShopHtml(c, {
      isGhost: true, cornerBadge: `<span class="tag-corner badge-red">${cardGoneLabel}</span>`,
    })).join("");

    const removedVouchers = (diff.shop_vouchers?.removed?.length)
      ? diff.shop_vouchers.removed
      : _consumedFromObserved(act?.observed_shop?.vouchers, st.shop_vouchers, shopItemKey);
    const removedPacks = (diff.shop_packs?.removed?.length)
      ? diff.shop_packs.removed
      : _consumedFromObserved(act?.observed_shop?.packs, st.shop_packs, shopItemKey);
    const vHtml = st.shop_vouchers.map(v => renderConsumableOrShopHtml({ type: "voucher", ...v }, { showBuyCost: true })).join("")
      + removedVouchers.map(v => renderConsumableOrShopHtml({ type: "voucher", ...v }, {
        isGhost: true, cornerBadge: `<span class="tag-corner badge-red">REDEEMED</span>`,
      })).join("");
    const pHtml = st.shop_packs.map(p => renderConsumableOrShopHtml({ type: "booster_pack", ...p }, { showBuyCost: true })).join("")
      + removedPacks.map(p => renderConsumableOrShopHtml({ type: "booster_pack", ...p }, {
        isGhost: true, cornerBadge: `<span class="tag-corner badge-red">OPENED</span>`,
      })).join("");

    panel.innerHTML = `
      ${renderObservedPackSection()}
      <div class="section-title">
        <span>Shop Offerings ${st.coupon_shop_active ? `<span class="badge badge-green">COUPON ACTIVE</span>` : ""}</span>
        <span>Cards: ${st.shop_cards.length} &bull; Vouchers: ${st.shop_vouchers.length} &bull; Packs: ${st.shop_packs.length}</span>
      </div>
      <div style="display:flex; gap:14px; flex-wrap:wrap; align-items:flex-end;">
        <div class="card-tray" style="min-height:92px;">${cardsHtml || `<span style="color:var(--muted);">No cards</span>`}</div>
        <div style="border-left:1px solid var(--border); padding-left:10px; display:flex; gap:8px; flex-wrap:wrap;">
          ${vHtml}${pHtml}
        </div>
      </div>
    `;
    return;
  }

  if (st.location === "pack" || st.active_pack) {
    const pk = st.active_pack;
    const pcd = diff.pack_cards || {};
    const addedP = new Set(pcd.added_indices || []);
    const dispPackCards = st.pack_cards.length ? st.pack_cards : (act?.observed_pack || []);
    const packCardsHtml = dispPackCards.map((c, idx) => renderConsumableOrShopHtml(c, {
      cls: addedP.has(idx) ? "added-glow" : "",
    })).join("") + (pcd.removed || []).map(c => renderConsumableOrShopHtml(c, {
      isGhost: true, cornerBadge: `<span class="tag-corner badge-red">PICKED</span>`,
    })).join("");

    panel.innerHTML = `
      ${renderObservedPackSection()}
      <div class="section-title">
        <span>Booster Pack Open: ${pk ? `${esc(pk.size)} ${esc(pk.booster_type)} Pack` : "Pack"}</span>
        <span class="badge badge-purple">Choices Left: ${st.pack_choices_remaining}</span>
      </div>
      <div style="display:flex; gap:12px; align-items:center; flex-wrap:wrap;">
        ${pk ? renderConsumableOrShopHtml({ type: "booster_pack", ...pk }) : ""}
        <div class="card-tray" style="min-height:92px;">${packCardsHtml || `<span style="color:var(--muted);">No pack cards</span>`}</div>
      </div>
    `;
    return;
  }

  if (st.location === "select") {
    const preExitShop = act?.observed_shop || (
      (st.shop_cards.length || st.shop_vouchers.length || st.shop_packs.length)
        ? { cards: st.shop_cards, vouchers: st.shop_vouchers, packs: st.shop_packs }
        : null
    );
    let preExitHtml = "";
    if (preExitShop && (preExitShop.cards?.length || preExitShop.vouchers?.length || preExitShop.packs?.length)) {
      const scHtml = (preExitShop.cards || []).map(c => renderConsumableOrShopHtml(c, { showBuyCost: true })).join("");
      const svHtml = (preExitShop.vouchers || []).map(v => renderConsumableOrShopHtml({ type: "voucher", ...v }, { showBuyCost: true })).join("");
      const spHtml = (preExitShop.packs || []).map(p => renderConsumableOrShopHtml({ type: "booster_pack", ...p }, { showBuyCost: true })).join("");
      preExitHtml = `
        <div style="margin-bottom:8px; padding-bottom:8px; border-bottom:1px solid var(--border);">
          <div class="section-title">
            <span>Shop Observed Before Leaving (${esc(act?.type || "Left Shop")})</span>
            <span class="badge badge-muted">Snapshot prior to exit</span>
          </div>
          <div style="display:flex; gap:14px; flex-wrap:wrap; align-items:flex-end;">
            <div class="card-tray">${scHtml || `<span style="color:var(--muted);">No cards</span>`}</div>
            <div style="border-left:1px solid var(--border); padding-left:10px; display:flex; gap:8px; flex-wrap:wrap;">
              ${svHtml}${spHtml}
            </div>
          </div>
        </div>
      `;
    }
    const blinds = [
      ["Small Blind", st.small_blind_tag, st.current_blind === "Small Blind"],
      ["Big Blind", st.big_blind_tag, st.current_blind === "Big Blind"],
      [st.boss_blind || "Boss Blind", null, st.current_blind !== "Small Blind" && st.current_blind !== "Big Blind"],
    ];
    panel.innerHTML = `
      ${renderObservedPackSection()}
      ${preExitHtml}
      <div class="section-title">Blind Select — Ante ${st.ante}</div>
      <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:8px;">
        ${blinds.map(([bName, tag, isCur]) => {
          const bPos = m.blinds[bName] || m.blinds["Small Blind"];
          const tPos = tag ? m.tags[tag] : null;
          return `<div style="padding:8px; border-radius:6px; border:1.5px solid ${isCur ? "var(--gold)" : "var(--border)"}; background:${isCur ? "rgba(245,158,11,0.1)" : "var(--panel-alt)"}; display:flex; align-items:center; gap:8px;">
            ${renderSprite(bPos, 36, 36)}
            <div>
              <div style="font-weight:700;">${esc(bName)}</div>
              ${tag ? `<div style="display:flex; align-items:center; gap:4px; font-size:11px; color:#93c5fd;">${renderSprite(tPos, 18, 18)} Skip: ${esc(tag)}</div>` : `<div style="font-size:11px;color:var(--muted);">${isCur ? "Current Blind" : "Upcoming"}</div>`}
            </div>
          </div>`;
        }).join("")}
      </div>
    `;
    return;
  }

  // Blind Play Stage
  panel.innerHTML = `
    <div class="section-title">
      <span>In Blind Play &mdash; ${esc(st.boss_blind || "No Boss")}</span>
      <span>Played Round: ${st.played_round_count} &bull; Discarded: ${st.discard_pile_count} &bull; Played Ante: ${st.played_ante_count}</span>
    </div>
    <div style="display:flex; gap:12px; flex-wrap:wrap; align-items:center; font-size:12px;">
      <span>Hands Played This Round: <strong>${st.hands_played_this_round}</strong></span>
      <span>Discards Used: <strong>${st.discards_used_this_round}</strong></span>
      ${st.locked_hand_type ? `<span class="badge badge-red">Locked Hand (The Mouth): ${esc(st.locked_hand_type)}</span>` : ""}
      ${st.ox_target_hand ? `<span class="badge badge-gold">Ox Hand: ${esc(st.ox_target_hand)}</span>` : ""}
      ${st.round_won ? `<span class="badge badge-green">Blind Defeated &mdash; Ready to Cash Out</span>` : ""}
    </div>
  `;
}

function computeStepVideoTime(baseFrameIdx, timestamp) {
  const vMeta = APP.payload?.video;
  const targetFrame = Math.max(0, (baseFrameIdx || 0) + APP.frameOffset);
  const fps = vMeta?.fps || 60;
  if (fps > 0 && targetFrame > 0) return targetFrame / fps;
  return timestamp || 0;
}

function toggleVideoMode() {
  APP.videoMode = APP.videoMode === "video" ? "frame" : "video";
  const btn = document.getElementById("btnVideoMode");
  btn.textContent = APP.videoMode === "video" ? "Frame View" : "Video Player";
  btn.classList.toggle("active-toggle", APP.videoMode === "video");
  const st = APP.payload?.steps?.[APP.stepIdx]?.state;
  renderVideoFrame(st?.frame_idx || 0, st?.timestamp || 0);
}

function seekVideoPlayerToStep() {
  const videoEl = document.getElementById("videoPlayer");
  const st = APP.payload?.steps?.[APP.stepIdx]?.state;
  if (!st || !APP.payload?.video?.available) return;
  const syncTime = computeStepVideoTime(st.frame_idx, st.timestamp);
  videoEl.dataset.pendingSeek = String(syncTime);
  if (videoEl.readyState >= 1) {
    try { videoEl.currentTime = syncTime; } catch (e) {}
  }
}

function renderVideoFrame(baseFrameIdx, timestamp) {
  const vMeta = APP.payload?.video;
  const stageEl = document.getElementById("videoStage");
  const imgEl = document.getElementById("videoImg");
  const videoEl = document.getElementById("videoPlayer");
  const phEl = document.getElementById("videoPlaceholder");
  const nudgeEl = document.getElementById("frameNudgeGroup");
  const syncBtn = document.getElementById("btnSyncVideo");
  const targetFrame = Math.max(0, (baseFrameIdx || 0) + APP.frameOffset);
  const offLabel = APP.frameOffset ? ` (${APP.frameOffset > 0 ? "+" : ""}${APP.frameOffset}f)` : "";
  const syncTime = computeStepVideoTime(baseFrameIdx, timestamp);

  if (!vMeta || !vMeta.available) {
    document.getElementById("videoCaption").textContent =
      `Frame ${targetFrame}${offLabel}  |  t = ${(timestamp || 0).toFixed(2)}s`;
    imgEl.style.display = "none";
    if (!videoEl.paused) videoEl.pause();
    videoEl.style.display = "none";
    phEl.style.display = "block";
    phEl.textContent = "No matched video file";
    return;
  }

  phEl.style.display = "none";
  const vKey = vMeta.cache_key || vMeta.filename || "active";
  if (APP.activeVideoKey !== vKey) {
    APP.activeVideoKey = vKey;
    imgEl.dataset.srcUrl = "";
    imgEl.removeAttribute("src");
    document.getElementById("videoModalImg").removeAttribute("src");
  }
  if (APP.videoMode === "video") {
    stageEl.classList.add("video-mode");
    stageEl.title = "";
    imgEl.style.display = "none";
    nudgeEl.style.display = "none";
    syncBtn.style.display = "inline-block";
    videoEl.style.display = "block";
    document.getElementById("videoCaption").textContent =
      `Video @ ${syncTime.toFixed(2)}s (Frame ${targetFrame})`;
    const videoUrl = `/api/video?traj=${encodeURIComponent(APP.payload?.trajectory_name || "active")}&v=${encodeURIComponent(vKey)}`;
    videoEl.dataset.pendingSeek = String(syncTime);
    if (videoEl.dataset.srcUrl !== videoUrl) {
      videoEl.dataset.srcUrl = videoUrl;
      videoEl.src = videoUrl;
    } else if (videoEl.readyState >= 1) {
      try { videoEl.currentTime = syncTime; } catch (e) {}
    }
    return;
  }

  stageEl.classList.remove("video-mode");
  stageEl.title = "Click to enlarge video frame";
  if (!videoEl.paused) videoEl.pause();
  videoEl.style.display = "none";
  nudgeEl.style.display = "flex";
  syncBtn.style.display = "none";
  document.getElementById("videoCaption").textContent =
    `Frame ${targetFrame}${offLabel} (click moment)  |  t = ${(timestamp || 0).toFixed(2)}s`;

  const url = `/api/frame/${targetFrame}?v=${encodeURIComponent(vKey)}`;
  if (imgEl.dataset.srcUrl !== url) {
    imgEl.dataset.srcUrl = url;
    imgEl.style.opacity = "0.55";
    imgEl.src = url;
  }
  imgEl.style.display = "block";
  if (document.getElementById("videoModal").classList.contains("open")) {
    document.getElementById("videoModalImg").src = url;
    document.getElementById("videoModalCaption").textContent =
      `${vMeta.filename || "Video"} — Frame ${targetFrame}${offLabel}`;
  }
}

function renderActionPanel(act, stepIdx) {
  const el = document.getElementById("actionPanel");
  const discs = getStepDiscrepancies(stepIdx);
  const discHtml = discs.length
    ? `<div class="err-banner">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px;">
          <strong>${discs.length} Discrepanc${discs.length === 1 ? "y" : "ies"} on Step ${stepIdx}</strong>
          <button style="padding:1px 6px;font-size:10px;" onclick="setActiveTab('errors')">Details</button>
        </div>
        ${discs.slice(0, 3).map(d => `<div style="margin-top:2px;">&bull; <span class="badge badge-red">${esc(d.category)}</span> ${esc(d.message)}</div>`).join("")}
        ${discs.length > 3 ? `<div style="margin-top:2px;color:#fca5a5;">+ ${discs.length - 3} more in Errors tab</div>` : ""}
      </div>`
    : "";

  if (!act) {
    el.innerHTML = `<div class="section-title">Action</div><div style="color:var(--muted);">Initial step (no action)</div>${discHtml}`;
    return;
  }
  const ti = act.target_item;
  let targetHtml = "";
  if (ti) {
    let thumb = "";
    if (ti.item) {
      const spec = lookupCardSprite(ti.item);
      if (spec?.kind === "playing_card") thumb = renderPlayingCardSprite(ti.item, 36, 48);
      else if (spec?.pos) thumb = renderSprite(spec.pos, 36, 48);
    } else if (ti.blind && APP.manifest.blinds[ti.blind]) {
      thumb = renderSprite(APP.manifest.blinds[ti.blind], 34, 34);
    }
    targetHtml = `<div style="display:flex;align-items:center;gap:8px;margin-top:5px;padding:4px 6px;background:var(--panel-alt);border-radius:5px;border:1px solid var(--border);">
      ${thumb}
      <div style="min-width:0;">
        <div style="font-size:10px;color:var(--muted);text-transform:uppercase;">Target (${esc(ti.area)}${ti.idx !== undefined ? ` #${ti.idx}` : ""})</div>
        <div style="font-weight:700;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${esc(ti.label)}</div>
      </div>
    </div>`;
  }

  const orderNotes = [];
  if (act.joker_order?.length) orderNotes.push(`joker_order: [${act.joker_order.join(",")}]`);
  if (act.hand_order?.length) orderNotes.push(`hand_order: [${act.hand_order.join(",")}]`);

  el.innerHTML = `
    <div class="section-title">
      <span>Action Executed</span>
      <span class="badge badge-blue">${esc(act.type)}</span>
    </div>
    ${targetHtml}
    ${orderNotes.length ? `<div style="font-size:11px;color:var(--muted);margin-top:4px;">${esc(orderNotes.join("  |  "))}</div>` : ""}
    ${act.metadata?.hud_ocr ? `<div style="font-size:11px;color:#94a3b8;margin-top:4px;">OCR: ${esc(JSON.stringify(act.metadata.hud_ocr))}</div>` : ""}
    ${discHtml}
  `;
}

function renderActiveTab(step) {
  const body = document.getElementById("tabBody");
  const st = step.state;
  const diff = step.diff;

  if (APP.activeTab === "diff") {
    const rows = [];
    for (const [k, v] of Object.entries(diff.changed_fields || {})) {
      rows.push(`<tr class="row-changed"><td><strong>${esc(k)}</strong></td><td>${esc(JSON.stringify(v.prev))}</td><td>${esc(JSON.stringify(v.curr))}</td></tr>`);
    }
    for (const [hname, hd] of Object.entries(diff.poker_hands || {})) {
      rows.push(`<tr class="row-changed"><td><strong>Hand: ${esc(hname)}</strong></td><td>${esc(JSON.stringify(Object.fromEntries(Object.entries(hd).map(([k, v]) => [k, v.prev]))))}</td><td>${esc(JSON.stringify(Object.fromEntries(Object.entries(hd).map(([k, v]) => [k, v.curr]))))}</td></tr>`);
    }
    const fd = diff.full_deck || {};
    if (fd.added_ids?.length) rows.push(`<tr class="row-changed"><td>Full Deck Added</td><td>—</td><td>IDs: ${esc(fd.added_ids.join(", "))}</td></tr>`);
    if (fd.removed?.length) rows.push(`<tr class="row-changed"><td>Full Deck Removed</td><td>${fd.removed.length} card(s)</td><td>—</td></tr>`);
    for (const [cid, mod] of Object.entries(fd.modified || {})) {
      rows.push(`<tr class="row-changed"><td>Deck Card #${esc(cid)}</td><td>${esc(JSON.stringify(Object.fromEntries(Object.entries(mod).map(([k, v]) => [k, v.prev]))))}</td><td>${esc(JSON.stringify(Object.fromEntries(Object.entries(mod).map(([k, v]) => [k, v.curr]))))}</td></tr>`);
    }
    for (const [cid, zc] of Object.entries(fd.zone_changed || {})) {
      rows.push(`<tr><td>Zone #${esc(cid)}</td><td>${esc(zc.prev)}</td><td>${esc(zc.curr)}</td></tr>`);
    }
    body.innerHTML = rows.length
      ? `<table class="compact"><thead><tr><th>Field</th><th>Previous</th><th>Current</th></tr></thead><tbody>${rows.join("")}</tbody></table>`
      : `<div style="color:var(--muted);padding:8px;">No changes recorded at this step.</div>`;
    return;
  }

  if (APP.activeTab === "hands") {
    const phd = diff.poker_hands || {};
    body.innerHTML = `<table class="compact">
      <thead><tr><th>Hand</th><th>Lvl</th><th>Chips x Mult</th><th>Played</th></tr></thead>
      <tbody>
        ${st.poker_hands.map(h => {
          const d = phd[h.hand];
          return `<tr class="${d ? "row-changed" : ""}">
            <td>${esc(h.hand)} ${h.played_this_round ? `<span class="badge badge-blue">ROUND</span>` : ""}</td>
            <td>${h.level}${d?.level ? ` <span class="delta-pill">${d.level.prev}&rarr;${d.level.curr}</span>` : ""}</td>
            <td><span style="color:#60a5fa;font-weight:700;">${h.chips}</span> &times; <span style="color:#f87171;font-weight:700;">${h.mult}</span></td>
            <td>${h.play_count}${d?.play_count ? ` <span class="delta-pill">${d.play_count.prev}&rarr;${d.play_count.curr}</span>` : ""}</td>
          </tr>`;
        }).join("")}
      </tbody>
    </table>`;
    return;
  }

  if (APP.activeTab === "piles") {
    const mkList = (title, arr) => `
      <div style="margin-bottom:8px;">
        <div class="section-title" style="margin-bottom:3px;">${title} (${arr.length})</div>
        <div style="display:flex;flex-wrap:wrap;gap:4px;">
          ${arr.length ? arr.map(c => `<span class="meta-pill">${esc(c.enhancement === "stone" ? "Stone" : `${c.rank}${({spades:"♠",hearts:"♥",clubs:"♣",diamonds:"♦"})[c.suit] || ""}`)} <span style="color:var(--muted);font-size:10px;">#${c.id}</span></span>`).join("")
            : `<span style="color:var(--muted);font-size:11px;">Empty</span>`}
        </div>
      </div>`;
    body.innerHTML =
      mkList("Draw Pile (Remaining)", st.draw_pile) +
      mkList("Played This Round", st.played_cards_this_round) +
      mkList("Discard Pile", st.discard_pile) +
      mkList("Played This Ante", st.played_cards_this_ante);
    return;
  }

  if (APP.activeTab === "raw") {
    const mods = {
      rates: { tarot: st.tarot_rate, planet: st.planet_rate, spectral: st.spectral_rate, playing: st.playing_card_rate, edition: st.edition_rate },
      stake_flags: { eternal: st.enable_eternals_in_shop, perishable: st.enable_perishables_in_shop, rental: st.enable_rentals_in_shop, no_interest: st.no_interest, ante_scaling: st.ante_scaling },
      misc: { tarots_used: st.tarots_used, unused_discards: st.unused_discards, ectoplasm_minus: st.ectoplasm_minus, gros_michel_extinct: st.gros_michel_extinct, last_tarot_planet: st.last_tarot_planet, unique_planets_used: st.unique_planets_used, pending_cv_checks: st.pending_cv_checks },
    };
    body.innerHTML = `<pre style="margin:0;font-size:11px;white-space:pre-wrap;color:#cbd5e1;">${esc(JSON.stringify(mods, null, 2))}</pre>`;
    return;
  }

  if (APP.activeTab === "errors") {
    const comp = APP.payload?.comparison;
    if (!comp) {
      body.innerHTML = `<div style="color:var(--muted);padding:10px;line-height:1.5;">
        No ground-truth comparison loaded.<br/>
        Choose a telemetry <code>.jsonl</code> or GT <code>_trajectory.json</code> in the topbar <strong>Compare</strong> selector, or launch with <code>--telemetry path/to/run_telemetry.jsonl</code>.
      </div>`;
      return;
    }
    const sm = comp.summary || {};
    const warnHtml = (comp.warnings || []).map(w => `<div class="err-banner" style="margin-top:0;margin-bottom:6px;background:rgba(245,158,11,0.14);border-color:rgba(245,158,11,0.5);color:#fde68a;">${esc(w)}</div>`).join("");
    const statsHtml = `<div style="display:flex;flex-wrap:wrap;gap:5px;margin-bottom:7px;">
      <span class="badge ${sm.error_step_count ? "badge-red" : "badge-green"}">Err Steps: ${sm.error_step_count || 0}</span>
      <span class="badge badge-muted">Diffs: ${sm.total_discrepancies || 0}</span>
      <span class="badge badge-blue">Aligned: ${sm.aligned_pairs || 0}</span>
      <span class="badge badge-muted">Miss GT: ${sm.missing_actions || 0}</span>
      <span class="badge badge-muted">Extra: ${sm.extra_actions || 0}</span>
      <span class="badge badge-muted">Suppressed: ${sm.suppressed_cascades || 0}</span>
    </div>`;

    const cats = Object.entries(sm.by_category || {});
    const chipsHtml = cats.length
      ? `<div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:7px;">
          <button class="err-chip ${APP.errorCategoryFilter === "all" ? "active" : ""}" onclick="APP.errorCategoryFilter='all'; renderActiveTab(APP.payload.steps[APP.stepIdx]);">All (${sm.error_step_count || 0})</button>
          ${cats.map(([cat, cnt]) => `<button class="err-chip ${APP.errorCategoryFilter === cat ? "active" : ""}" onclick="APP.errorCategoryFilter='${esc(cat)}'; renderActiveTab(APP.payload.steps[APP.stepIdx]);">${esc(cat)} (${cnt})</button>`).join("")}
        </div>`
      : "";

    const curDiscs = getStepDiscrepancies(APP.stepIdx);
    const curHtml = curDiscs.length
      ? `<div style="margin-bottom:8px;">
          <div class="section-title" style="color:#fca5a5;">Current Step ${APP.stepIdx} Discrepancies</div>
          <table class="compact">
            <thead><tr><th>Cat</th><th>Expected (GT)</th><th>Actual</th><th>Message</th></tr></thead>
            <tbody>
              ${curDiscs.map(d => `<tr class="row-changed">
                <td><span class="badge badge-red">${esc(d.category)}</span></td>
                <td><code>${esc(d.expected === null || d.expected === undefined ? "—" : (typeof d.expected === "string" ? d.expected : JSON.stringify(d.expected)))}</code></td>
                <td><code>${esc(d.actual === null || d.actual === undefined ? "—" : (typeof d.actual === "string" ? d.actual : JSON.stringify(d.actual)))}</code></td>
                <td>${esc(d.message)}</td>
              </tr>`).join("")}
            </tbody>
          </table>
        </div>`
      : "";

    const allErrs = (comp.errors || []).filter(e =>
      APP.errorCategoryFilter === "all" || (e.categories || []).includes(APP.errorCategoryFilter)
    );
    const listHtml = allErrs.length
      ? allErrs.map(e => {
          const isCur = e.actual_step_idx === APP.stepIdx;
          const catBadges = (e.categories || []).map(c => `<span class="badge badge-red">${esc(c)}</span>`).join(" ");
          const firstMsg = e.discrepancies?.[0]?.message || "";
          const moreCount = (e.discrepancies?.length || 0) - 1;
          return `<div class="err-row ${isCur ? "active" : ""}" onclick="goToStep(${e.actual_step_idx})">
            <div style="display:flex;justify-content:space-between;align-items:center;gap:6px;margin-bottom:3px;">
              <strong>Step ${e.actual_step_idx}${e.expected_step_idx !== null && e.expected_step_idx !== undefined ? ` <span style="color:var(--muted);font-weight:400;">(GT ${e.expected_step_idx})</span>` : ""}</strong>
              <span style="font-size:10.5px;color:var(--muted);">f=${e.frame_idx} &bull; A${e.ante}R${e.round} &bull; ${esc(e.action_type)}</span>
            </div>
            <div style="display:flex;flex-wrap:wrap;gap:3px;margin-bottom:3px;">${catBadges}</div>
            <div style="font-size:11px;color:#e2e8f0;">${esc(firstMsg)}${moreCount > 0 ? ` <span style="color:var(--muted);">(+${moreCount} more)</span>` : ""}</div>
          </div>`;
        }).join("")
      : `<div style="color:#86efac;padding:8px;">No discrepancies${APP.errorCategoryFilter !== "all" ? ` in category "${esc(APP.errorCategoryFilter)}"` : ""} — trajectory matches ground truth!</div>`;

    body.innerHTML = `${warnHtml}${statsHtml}${chipsHtml}${curHtml}<div class="section-title">All Error Steps (${allErrs.length})</div>${listHtml}`;
  }
}

function renderScrubber() {
  const stats = APP.payload?.run_stats || [];
  const track = document.getElementById("scrubberTrack");
  const diagTrack = document.getElementById("diagTrack");
  const legendWarn = document.getElementById("legendWarnItem");
  const colors = { blind: "#3b82f6", shop: "#f59e0b", pack: "#c084fc", select: "#22c55e" };
  const errSet = new Set(getErrorStepIndices());
  let html = "";
  let diagHtml = "";
  let hasAnyWarn = false;
  let lastAnte = null;
  for (let i = 0; i < stats.length; i++) {
    const r = stats[i];
    const hPct = r.has_changes ? "85%" : "42%";
    const anteBorder = (lastAnte !== null && r.ante !== lastAnte) ? "border-left:2px solid #fff;" : "";
    lastAnte = r.ante;
    const isErr = errSet.has(i);
    const reasons = r.warning_reasons || [];
    const hasDirectWarn = reasons.some(x => x !== "downstream_cascade_quarantine") || (!!r.corrupted && reasons.length === 0);
    const hasCascadeOnly = !hasDirectWarn && reasons.includes("downstream_cascade_quarantine");
    if (hasDirectWarn || hasCascadeOnly) hasAnyWarn = true;

    const errMark = isErr ? `<div class="scrub-err-mark"></div>` : "";
    html += `<div class="scrub-tick" data-step="${i}" style="${anteBorder}">
      ${errMark}
      <div class="scrub-tick-bar" style="height:${hPct};background:${colors[r.location] || "#64748b"};opacity:${i === APP.stepIdx ? 1 : 0.72};"></div>
    </div>`;

    let diagSegs = "";
    if (isErr) diagSegs += `<div class="diag-seg-err"></div>`;
    if (hasDirectWarn) diagSegs += `<div class="diag-seg-warn"></div>`;
    else if (hasCascadeOnly) diagSegs += `<div class="diag-seg-cascade"></div>`;
    diagHtml += `<div class="diag-tick" data-step="${i}" style="${anteBorder}">${diagSegs}</div>`;
  }
  const pctLeft = stats.length > 1 ? (APP.stepIdx / (stats.length - 1)) * 100 : 0;
  const playheadHtml = `<div class="scrub-playhead" style="left:calc(${pctLeft.toFixed(3)}% - 1.5px);"></div>`;
  track.innerHTML = html + playheadHtml;
  if (diagTrack) diagTrack.innerHTML = diagHtml + playheadHtml;
  if (legendWarn) legendWarn.style.display = hasAnyWarn ? "inline" : "none";
}

function openDeckModal() {
  document.getElementById("deckModal").classList.add("open");
  renderDeckModal();
}
function closeDeckModal() {
  document.getElementById("deckModal").classList.remove("open");
}

function renderDeckModal() {
  const step = APP.payload?.steps?.[APP.stepIdx];
  if (!step) return;
  const st = step.state;
  const fd = step.diff?.full_deck || {};
  const zoneFilter = document.getElementById("deckZoneFilter").value;
  const groupBy = document.getElementById("deckGroupBy").value;
  const search = document.getElementById("deckSearch").value.trim().toLowerCase();

  const addedIds = new Set(fd.added_ids || []);
  const modIds = fd.modified || {};
  const zoneChanged = fd.zone_changed || {};

  // Stats strip across full deck
  const suitsCount = { spades: 0, hearts: 0, clubs: 0, diamonds: 0, stone: 0 };
  let enhCount = 0, sealCount = 0, edCount = 0;
  for (const c of st.full_deck) {
    const enh = c.enhancement || "none";
    const seal = c.seal || "none";
    const ed = c.edition || "none";
    if (enh === "stone") suitsCount.stone++;
    else if (suitsCount[c.suit] !== undefined) suitsCount[c.suit]++;
    if (enh !== "none") enhCount++;
    if (seal !== "none") sealCount++;
    if (ed !== "none") edCount++;
  }
  document.getElementById("deckModalSummary").textContent =
    `${st.full_deck.length} Cards — Draw ${st.draw_pile_count}, Hand ${st.hand.length}, Played ${st.played_round_count}, Discarded ${st.discard_pile_count}`;
  document.getElementById("deckStatsStrip").innerHTML = `
    <span class="meta-pill">♠ ${suitsCount.spades}</span>
    <span class="meta-pill" style="color:#fca5a5;">♥ ${suitsCount.hearts}</span>
    <span class="meta-pill" style="color:#86efac;">♣ ${suitsCount.clubs}</span>
    <span class="meta-pill" style="color:#7dd3fc;">♦ ${suitsCount.diamonds}</span>
    ${suitsCount.stone ? `<span class="meta-pill">Stone ${suitsCount.stone}</span>` : ""}
    <span class="badge badge-gold">Enhanced: ${enhCount}</span>
    <span class="badge badge-purple">Seals: ${sealCount}</span>
    <span class="badge badge-blue">Editions: ${edCount}</span>
  `;

  const cards = st.full_deck.filter(c => {
    const enh = c.enhancement || "none";
    const seal = c.seal || "none";
    const ed = c.edition || "none";
    if (zoneFilter === "modified_cards") {
      if (enh === "none" && seal === "none" && ed === "none" && !c.bonus_chips) return false;
    } else if (zoneFilter === "step_changed") {
      const idStr = String(c.id);
      if (!addedIds.has(c.id) && !modIds[idStr] && !zoneChanged[idStr]) return false;
    } else if (zoneFilter !== "all" && c.zone !== zoneFilter) {
      return false;
    }
    if (search) {
      const hay = `${c.rank} ${c.suit} ${enh} ${seal} ${ed} ${c.zone} #${c.id}`.toLowerCase();
      if (!hay.includes(search)) return false;
    }
    return true;
  });

  const renderCardItem = (c) => {
    const idStr = String(c.id);
    const enh = c.enhancement || "none";
    const seal = c.seal || "none";
    const ed = c.edition || "none";
    let cls = ""; let corner = "";
    if (addedIds.has(c.id)) { cls = "added-glow"; corner = `<span class="tag-corner badge-green">NEW</span>`; }
    else if (modIds[idStr]) { cls = "mod-glow"; corner = `<span class="tag-corner badge-blue">MOD</span>`; }
    else if (zoneChanged[idStr]) { cls = "changed-glow"; corner = `<span class="tag-corner badge-gold">MOVED</span>`; }
    const zoneLabel = { in_draw_pile: "Draw", in_hand: "Hand", played_round: "Played", discarded: "Discard" }[c.zone] || c.zone;
    const modBadges = [
      enh !== "none" ? `<span class="badge badge-gold">${esc(enh)}</span>` : "",
      ed !== "none" ? `<span class="badge badge-blue">${esc(ed)}</span>` : "",
      seal !== "none" ? `<span class="badge badge-purple">${esc(seal)}</span>` : "",
    ].join("");
    const extraSub = `<div class="card-sub"><span class="zone-dot zone-${esc(c.zone)}">${esc(zoneLabel)}</span>${modBadges}</div>`;
    const hoverInfo = `data-cardinfo="${esc(JSON.stringify({ card: c, mod: modIds[idStr] || null, zone_diff: zoneChanged[idStr] || null }))}"`;
    return renderSmallDeckCardHtml(c, { cls, cornerBadge: corner, extraSub, onHoverData: hoverInfo });
  };

  let bodyHtml = "";
  if (fd.removed?.length) {
    bodyHtml += `<div class="deck-group-title" style="color:#fca5a5;">Removed From Deck This Step (${fd.removed.length})</div>
      <div class="deck-grid" style="margin-bottom:12px;">${fd.removed.map(c => renderSmallDeckCardHtml(c, { cls: "ghost-removed", cornerBadge: `<span class="tag-corner badge-red">LOST</span>` })).join("")}</div>`;
  }

  if (groupBy === "none") {
    bodyHtml += `<div class="deck-grid">${cards.map(renderCardItem).join("")}</div>`;
  } else {
    const groups = new Map();
    const order = groupBy === "suit" ? ["spades", "hearts", "clubs", "diamonds", "stone", "none"]
      : groupBy === "zone" ? ["in_hand", "played_round", "discarded", "in_draw_pile"] : [];
    for (const key of order) groups.set(key, []);
    for (const c of cards) {
      const enh = c.enhancement || "none";
      const gk = groupBy === "suit" ? (enh === "stone" ? "stone" : c.suit)
        : groupBy === "zone" ? c.zone : enh;
      if (!groups.has(gk)) groups.set(gk, []);
      groups.get(gk).push(c);
    }
    for (const [gk, list] of groups.entries()) {
      if (!list.length) continue;
      bodyHtml += `<div class="deck-group-title"><span>${esc(gk)}</span><span>${list.length} cards</span></div>
        <div class="deck-grid">${list.map(renderCardItem).join("")}</div>`;
    }
  }
  document.getElementById("deckModalBody").innerHTML = bodyHtml || `<div style="color:var(--muted);">No cards match current filter.</div>`;
}

function goToStep(newIdx) {
  const n = APP.payload?.steps?.length || 0;
  if (!n) return;
  APP.stepIdx = Math.max(0, Math.min(newIdx, n - 1));
  APP.frameOffset = 0;
  renderAll();
}

function stepRelative(delta) {
  const allowed = getFilteredSteps();
  if (!allowed.length) return;
  if (delta > 0) {
    const next = allowed.find(idx => idx > APP.stepIdx);
    if (next !== undefined) goToStep(next);
  } else if (delta < 0) {
    const prev = [...allowed].reverse().find(idx => idx < APP.stepIdx);
    if (prev !== undefined) goToStep(prev);
  }
}

function togglePlay() {
  const btn = document.getElementById("btnPlay");
  if (APP.playTimer) {
    clearInterval(APP.playTimer);
    APP.playTimer = null;
    btn.textContent = "Play";
    return;
  }
  const speed = Number(document.getElementById("speedSelect").value) || 650;
  btn.textContent = "Pause";
  APP.playTimer = setInterval(() => {
    const allowed = getFilteredSteps();
    const next = allowed.find(idx => idx > APP.stepIdx);
    if (next === undefined) { togglePlay(); return; }
    goToStep(next);
  }, speed);
}

function stopPreloadPolling() {
  if (APP.preloadTimer) {
    clearInterval(APP.preloadTimer);
    APP.preloadTimer = null;
  }
}

async function pollPreloadStatusOnce() {
  const badge = document.getElementById("preloadBadge");
  try {
    const resp = await fetch("/api/preload-status");
    if (!resp.ok) return;
    const s = await resp.json();
    if (!s.available) {
      badge.style.display = "none";
      stopPreloadPolling();
      return;
    }
    badge.style.display = "inline-block";
    if (s.priority_total > 0 && s.priority_cached < s.priority_total) {
      badge.className = "preload-pill";
      badge.textContent = `Err frames: ${s.priority_cached}/${s.priority_total}`;
    } else if (!s.done) {
      badge.className = "preload-pill";
      badge.textContent = s.priority_total > 0
        ? `✓ Err cached (${s.priority_cached}/${s.priority_total}) · ${s.total_cached}/${s.total_targets}`
        : `Caching: ${s.total_cached}/${s.total_targets}`;
    } else {
      badge.className = "preload-pill ready";
      badge.textContent = s.priority_total > 0
        ? `✓ ${s.priority_cached}/${s.priority_total} err + ${s.total_cached} frames`
        : `✓ ${s.total_cached} frames`;
      stopPreloadPolling();
    }
  } catch (e) {}
}

function startPreloadPolling() {
  stopPreloadPolling();
  pollPreloadStatusOnce();
  APP.preloadTimer = setInterval(pollPreloadStatusOnce, 750);
}

async function runComparisonBySource(sourcePath) {
  const q = sourcePath ? `?source=${encodeURIComponent(sourcePath)}` : "?source=none";
  const resp = await fetch(`/api/compare${q}`);
  const data = await resp.json();
  if (!resp.ok) {
    alert(`Comparison failed: ${data.error || resp.status}`);
    return;
  }
  APP.payload.comparison = data.comparison || null;
  APP.errorCategoryFilter = "all";
  if ((data.comparison?.summary?.error_step_count || 0) > 0) {
    APP.activeTab = "errors";
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.toggle("active", b.dataset.tab === "errors"));
  }
  renderAll();
  startPreloadPolling();
}

async function loadTrajectoryByName(name) {
  const videoEl = document.getElementById("videoPlayer");
  if (!videoEl.paused) videoEl.pause();
  videoEl.dataset.srcUrl = "";
  const imgEl = document.getElementById("videoImg");
  imgEl.dataset.srcUrl = "";
  imgEl.removeAttribute("src");
  document.getElementById("videoModalImg").removeAttribute("src");
  APP.activeVideoKey = "";
  const q = name ? `?name=${encodeURIComponent(name)}` : "";
  const resp = await fetch(`/api/trajectory${q}`);
  APP.payload = await resp.json();
  APP.stepIdx = 0;
  APP.frameOffset = 0;
  APP.errorCategoryFilter = "all";
  document.getElementById("telemetrySelect").value = "";
  renderAll();
  startPreloadPolling();
}

async function init() {
  const [manifestResp, listResp, trajResp, telResp] = await Promise.all([
    fetch("/api/manifest"),
    fetch("/api/trajectories"),
    fetch("/api/trajectory"),
    fetch("/api/telemetry-sources"),
  ]);
  APP.manifest = await manifestResp.json();
  APP.trajectories = await listResp.json();
  APP.payload = await trajResp.json();
  APP.telemetrySources = await telResp.json();

  const sel = document.getElementById("trajSelect");
  const items = APP.trajectories.items || [];
  sel.innerHTML = items.length
    ? items.map(it => `<option value="${esc(it.name)}" ${it.name === APP.trajectories.active_name ? "selected" : ""}>${esc(it.name)} (${it.size_kb} KB${it.video ? " + MP4" : ""})</option>`).join("")
    : `<option value="">Current In-Memory Trajectory</option>`;

  const telSel = document.getElementById("telemetrySelect");
  const telItems = APP.telemetrySources.items || [];
  const activeTel = APP.telemetrySources.active_source || APP.payload?.comparison?.source_path || "";
  telSel.innerHTML = `<option value="">Compare: None</option>`
    + telItems.map(it => `<option value="${esc(it.path)}" ${it.path === activeTel ? "selected" : ""}>${esc(it.label)} (${it.size_kb} KB)</option>`).join("");

  const imgEl = document.getElementById("videoImg");
  imgEl.onload = () => { imgEl.style.opacity = "1"; };
  imgEl.onerror = () => { imgEl.style.opacity = "1"; };

  const videoEl = document.getElementById("videoPlayer");
  videoEl.addEventListener("loadedmetadata", () => {
    const t = Number(videoEl.dataset.pendingSeek);
    if (Number.isFinite(t)) {
      try { videoEl.currentTime = t; } catch (e) {}
    }
  });

  sel.addEventListener("change", () => { if (sel.value) loadTrajectoryByName(sel.value); });
  telSel.addEventListener("change", () => runComparisonBySource(telSel.value));
  document.getElementById("compareStatusPill").addEventListener("click", () => setActiveTab("errors"));
  document.getElementById("btnPrevErr").addEventListener("click", () => stepErrorRelative(-1));
  document.getElementById("btnNextErr").addEventListener("click", () => stepErrorRelative(1));
  document.getElementById("btnVideoMode").addEventListener("click", toggleVideoMode);
  document.getElementById("btnSyncVideo").addEventListener("click", seekVideoPlayerToStep);

  document.getElementById("btnFirst").addEventListener("click", () => goToStep(0));
  document.getElementById("btnLast").addEventListener("click", () => goToStep((APP.payload?.steps?.length || 1) - 1));
  document.getElementById("btnPrev").addEventListener("click", () => stepRelative(-1));
  document.getElementById("btnNext").addEventListener("click", () => stepRelative(1));
  document.getElementById("btnPlay").addEventListener("click", togglePlay);
  document.getElementById("stepInput").addEventListener("change", e => goToStep(Number(e.target.value) || 0));
  document.getElementById("filterSelect").addEventListener("change", () => {
    const allowed = getFilteredSteps();
    if (allowed.length && !allowed.includes(APP.stepIdx)) goToStep(allowed[0]);
  });

  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => setActiveTab(btn.dataset.tab));
  });

  document.querySelectorAll("[data-nudge]").forEach(btn => {
    btn.addEventListener("click", () => {
      const n = Number(btn.dataset.nudge);
      APP.frameOffset = n === 0 ? 0 : APP.frameOffset + n;
      const st = APP.payload?.steps?.[APP.stepIdx]?.state;
      if (st) renderVideoFrame(st.frame_idx, st.timestamp);
    });
  });

  // Timeline click/drag scrub (main bar + stacked diagnostic bar)
  const track = document.getElementById("scrubberTrack");
  const diagTrackEl = document.getElementById("diagTrack");
  let activeDragTrack = null;
  const scrubFromClientX = (cx, el) => {
    const targetEl = el || track;
    const rect = targetEl.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (cx - rect.left) / Math.max(1, rect.width)));
    const total = APP.payload?.steps?.length || 1;
    goToStep(Math.round(frac * (total - 1)));
  };
  track.addEventListener("mousedown", e => { activeDragTrack = track; scrubFromClientX(e.clientX, track); });
  if (diagTrackEl) {
    diagTrackEl.addEventListener("mousedown", e => { activeDragTrack = diagTrackEl; scrubFromClientX(e.clientX, diagTrackEl); });
  }
  window.addEventListener("mousemove", e => {
    if (activeDragTrack) scrubFromClientX(e.clientX, activeDragTrack);
    const tick = e.target?.closest?.(".scrub-tick, .diag-tick");
    if (tick) {
      const sIdx = Number(tick.dataset.step);
      const r = APP.payload?.run_stats?.[sIdx];
      if (r) {
        const discs = getStepDiscrepancies(sIdx);
        const errText = discs.length ? `  |  ✕ ${discs.length} error(s): ${discs[0].message}` : "";
        const warnReasons = r.warning_reasons || [];
        const warnText = warnReasons.length
          ? `  |  ⚠ Warning: ${warnReasons.join(", ")}`
          : (r.corrupted ? "  |  ⚠ Warning: corrupted" : "");
        document.getElementById("timelineHoverInfo").textContent =
          `Step ${r.step_idx} | Frame ${r.frame_idx} | ${r.action_type} | Ante ${r.ante} R${r.round} | $${r.money ?? "—"}${errText}${warnText}`;
      }
    }
  });
  window.addEventListener("mouseup", () => { activeDragTrack = null; });

  // Deck Modal & Hover Inspector
  document.getElementById("btnDeckModal").addEventListener("click", openDeckModal);
  document.getElementById("btnCloseDeckModal").addEventListener("click", closeDeckModal);
  ["deckZoneFilter", "deckGroupBy", "deckSearch"].forEach(id => {
    document.getElementById(id).addEventListener("input", renderDeckModal);
  });
  document.getElementById("deckModalBody").addEventListener("mouseover", e => {
    const cardEl = e.target?.closest?.("[data-cardinfo]");
    if (!cardEl) return;
    const info = JSON.parse(cardEl.dataset.cardinfo);
    const c = info.card;
    const modStr = info.mod ? ` | Modified: ${JSON.stringify(info.mod)}` : "";
    const zoneStr = info.zone_diff ? ` | Zone: ${info.zone_diff.prev} -> ${info.zone_diff.curr}` : "";
    document.getElementById("deckInspectorBar").textContent =
      `Card #${c.id}: ${c.rank} of ${c.suit} | Zone: ${c.zone} | Enh: ${c.enhancement || "none"} | Seal: ${c.seal || "none"} | Ed: ${c.edition || "none"} | +Chips: ${c.bonus_chips || 0}${modStr}${zoneStr}`;
  });

  // Video Zoom Modal (Frame mode only so native video controls stay clickable)
  document.getElementById("videoStage").addEventListener("click", () => {
    if (APP.videoMode !== "frame") return;
    if (!APP.payload?.video?.available) return;
    document.getElementById("videoModal").classList.add("open");
    const st = APP.payload.steps[APP.stepIdx].state;
    renderVideoFrame(st.frame_idx, st.timestamp);
  });
  document.getElementById("btnCloseVideoModal").addEventListener("click", () => {
    document.getElementById("videoModal").classList.remove("open");
  });

  // Keyboard shortcuts
  window.addEventListener("keydown", e => {
    if (["INPUT", "SELECT", "TEXTAREA"].includes(e.target?.tagName)) return;
    if (e.key === "ArrowRight") { e.preventDefault(); stepRelative(1); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); stepRelative(-1); }
    else if (e.key === "]") { e.preventDefault(); stepErrorRelative(1); }
    else if (e.key === "[") { e.preventDefault(); stepErrorRelative(-1); }
    else if (e.key === "v" || e.key === "V") { e.preventDefault(); toggleVideoMode(); }
    else if (e.key === "Home") { e.preventDefault(); goToStep(0); }
    else if (e.key === "End") { e.preventDefault(); goToStep((APP.payload?.steps?.length || 1) - 1); }
    else if (e.key === " ") { e.preventDefault(); togglePlay(); }
    else if (e.key === "d" || e.key === "D") {
      e.preventDefault();
      document.getElementById("deckModal").classList.contains("open") ? closeDeckModal() : openDeckModal();
    } else if (e.key === "Escape") {
      closeDeckModal();
      document.getElementById("videoModal").classList.remove("open");
    } else if (["1", "2", "3", "4", "5"].includes(e.key)) {
      const tabs = ["diff", "hands", "piles", "raw", "errors"];
      setActiveTab(tabs[Number(e.key) - 1]);
    }
  });

  if ((APP.payload?.comparison?.summary?.error_step_count || 0) > 0) {
    setActiveTab("errors");
  }
  renderAll();
  startPreloadPolling();
}

init();
</script>
</body>
</html>
"""


def _resolve_default_trajectory(traj_dir: Path) -> Path | None:
    items = discover_trajectories(traj_dir)
    if not items:
        return None
    return Path(items[0]["path"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive Balatro Run Trajectory Visualizer"
    )
    parser.add_argument(
        "--trajectory",
        "-t",
        type=Path,
        default=None,
        help="Path to *_trajectory.json (defaults to newest JSON in output/trajectories/)",
    )
    parser.add_argument(
        "--telemetry",
        "--gt",
        dest="telemetry",
        type=Path,
        default=None,
        help="Optional Lua run_telemetry.jsonl or GT *_trajectory.json for non-cascading error checks",
    )
    parser.add_argument(
        "--telemetry-seed",
        type=str,
        default=None,
        help="Optional seed filter when selecting a run from --telemetry .jsonl",
    )
    parser.add_argument(
        "--telemetry-run-index",
        type=int,
        default=None,
        help="Optional 0-based run segment index when loading --telemetry .jsonl",
    )
    parser.add_argument(
        "--video",
        "-v",
        type=Path,
        default=None,
        help="Optional MP4 path (auto-discovered from assets/sample_inputs/videos/ if omitted)",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="HTTP bind host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=8765, help="HTTP port (default: 8765)"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically launch a web browser on startup",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log HTTP requests to stderr",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    traj_path = args.trajectory or _resolve_default_trajectory(DEFAULT_TRAJECTORIES_DIR)
    if traj_path is not None and not traj_path.is_file():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")
    if args.telemetry is not None and not args.telemetry.is_file():
        raise FileNotFoundError(f"Telemetry/GT file not found: {args.telemetry}")

    server: TrajectoryVisualizerServer | None = None
    candidate_ports = (
        [args.port] if args.port == 0 else list(range(args.port, args.port + 21))
    )
    last_err: OSError | None = None
    for port_try in candidate_ports:
        try:
            server = TrajectoryVisualizerServer(
                trajectory_path=traj_path,
                video_path=args.video,
                telemetry_path=args.telemetry,
                telemetry_seed=args.telemetry_seed,
                telemetry_run_index=args.telemetry_run_index,
                host=args.host,
                port=port_try,
                verbose=args.verbose,
            )
            if args.port != 0 and port_try != args.port:
                print(
                    f"[TrajectoryVisualizer] Port {args.port} is already in use; serving on port {server.port} instead."
                )
            break
        except OSError as exc:
            last_err = exc
    if server is None:
        assert last_err is not None
        raise last_err
    print(f"[TrajectoryVisualizer] Serving at {server.url}")
    if traj_path:
        print(f"[TrajectoryVisualizer] Loaded trajectory: {traj_path}")
    if server.app_state.frame_cache and server.app_state.frame_cache.meta.get(
        "available"
    ):
        print(
            f"[TrajectoryVisualizer] Matched video: {server.app_state.frame_cache.meta['filename']}"
        )
    if server.app_state.comparison is not None:
        sm = server.app_state.comparison["summary"]
        print(
            f"[TrajectoryVisualizer] Ground-truth test: {sm['error_step_count']} error step(s), "
            f"{sm['total_discrepancies']} discrepancy(ies) ({args.telemetry})"
        )

    if not args.no_browser:
        threading.Timer(0.25, lambda: webbrowser.open(server.url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[TrajectoryVisualizer] Shutting down...")
        server.close()


if __name__ == "__main__":
    main()
