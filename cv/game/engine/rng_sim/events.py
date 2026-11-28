"""Event Mechanics & Roll Solvers for Balatro Simulation (`game.engine.rng_sim.events`).

Uses the canonical models and enums from `game.balatro_models` (`PlayingCard`, `Tarot`,
`TarotName`, `Voucher`, `VoucherName`, `SkipTag`, `BlindName`, `PokerHand`, `Suit`,
`Rank`, `Enhancement`, `Edition`, `Seal`) and verified against `balatro_source`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from game.balatro_models import CardType, Planet, PlanetName
from game.engine.rng_sim.rng import BalatroRNG, Tw223
from game.engine.rng_sim.state import (
    DEFAULT_VISIBLE_POKER_HANDS,
    BlindName,
    Card,
    Deck,
    Edition,
    Enhancement,
    GameState,
    Joker,
    JokerName,
    PlayingCard,
    PokerHand,
    Rank,
    Seal,
    SimCard,
    SkipTag,
    Spectral,
    SpectralName,
    Suit,
    Tarot,
    TarotName,
    Voucher,
    VoucherName,
    card_is_rank,
    order_poker_hands_luajit,
    parse_suit,
    seal_balatro_name,
)

# ---------------------------------------------------------------------------
# 1. G.P_CARDS Table (52 Standard Playing Card Fronts in Lexicographical Key Order)
# ---------------------------------------------------------------------------

P_CARDS_KEYS: list[str] = [
    "C_2",
    "C_3",
    "C_4",
    "C_5",
    "C_6",
    "C_7",
    "C_8",
    "C_9",
    "C_A",
    "C_J",
    "C_K",
    "C_Q",
    "C_T",
    "D_2",
    "D_3",
    "D_4",
    "D_5",
    "D_6",
    "D_7",
    "D_8",
    "D_9",
    "D_A",
    "D_J",
    "D_K",
    "D_Q",
    "D_T",
    "H_2",
    "H_3",
    "H_4",
    "H_5",
    "H_6",
    "H_7",
    "H_8",
    "H_9",
    "H_A",
    "H_J",
    "H_K",
    "H_Q",
    "H_T",
    "S_2",
    "S_3",
    "S_4",
    "S_5",
    "S_6",
    "S_7",
    "S_8",
    "S_9",
    "S_A",
    "S_J",
    "S_K",
    "S_Q",
    "S_T",
]

_SUIT_LOOKUP: dict[str, Suit] = {
    "C": Suit.CLUBS,
    "D": Suit.DIAMONDS,
    "H": Suit.HEARTS,
    "S": Suit.SPADES,
}

_RANK_LOOKUP: dict[str, Rank] = {
    "2": Rank.TWO,
    "3": Rank.THREE,
    "4": Rank.FOUR,
    "5": Rank.FIVE,
    "6": Rank.SIX,
    "7": Rank.SEVEN,
    "8": Rank.EIGHT,
    "9": Rank.NINE,
    "T": Rank.TEN,
    "J": Rank.JACK,
    "Q": Rank.QUEEN,
    "K": Rank.KING,
    "A": Rank.ACE,
}

P_SEALS_KEYS: list[Seal] = [
    Seal.BLUE,
    Seal.GOLD,
    Seal.PURPLE,
    Seal.RED,
]


def parse_card_key(key: str) -> tuple[Suit, Rank]:
    """Parse Balatro card front key into `(Suit, Rank)`, e.g. `'C_Q'` -> `(Suit.CLUBS, Rank.QUEEN)`."""
    parts = key.split("_")
    suit = _SUIT_LOOKUP[parts[0]]
    rank = _RANK_LOOKUP[parts[1]]
    return suit, rank


# ---------------------------------------------------------------------------
# 2. G.P_CENTER_POOLS['Tarot'] (22 Tarots mapped to `balatro_models.TarotName`)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TarotProto:
    key: str
    tarot_name: TarotName
    order: int

    @property
    def name(self) -> str:
        return self.tarot_name.value


P_TAROTS: list[TarotProto] = [
    TarotProto("c_fool", TarotName.FOOL, 1),
    TarotProto("c_magician", TarotName.MAGICIAN, 2),
    TarotProto("c_high_priestess", TarotName.HIGH_PRIESTESS, 3),
    TarotProto("c_empress", TarotName.EMPRESS, 4),
    TarotProto("c_emperor", TarotName.EMPEROR, 5),
    TarotProto("c_heirophant", TarotName.HIEROPHANT, 6),
    TarotProto("c_lovers", TarotName.LOVERS, 7),
    TarotProto("c_chariot", TarotName.CHARIOT, 8),
    TarotProto("c_justice", TarotName.JUSTICE, 9),
    TarotProto("c_hermit", TarotName.HERMIT, 10),
    TarotProto("c_wheel_of_fortune", TarotName.WHEEL_OF_FORTUNE, 11),
    TarotProto("c_strength", TarotName.STRENGTH, 12),
    TarotProto("c_hanged_man", TarotName.HANGED_MAN, 13),
    TarotProto("c_death", TarotName.DEATH, 14),
    TarotProto("c_temperance", TarotName.TEMPERANCE, 15),
    TarotProto("c_devil", TarotName.DEVIL, 16),
    TarotProto("c_tower", TarotName.TOWER, 17),
    TarotProto("c_star", TarotName.STAR, 18),
    TarotProto("c_moon", TarotName.MOON, 19),
    TarotProto("c_sun", TarotName.SUN, 20),
    TarotProto("c_judgement", TarotName.JUDGEMENT, 21),
    TarotProto("c_world", TarotName.WORLD, 22),
]

TAROT_BY_KEY: dict[str, TarotProto] = {t.key: t for t in P_TAROTS}
TAROT_BY_NAME: dict[str, TarotProto] = {t.name.lower(): t for t in P_TAROTS}


def normalize_tarot_key(key_or_name: TarotName | str) -> str:
    """Convert a `TarotName`, key (`'c_fool'`), or string (`'The Fool'`) to its `'c_...'` key."""
    if isinstance(key_or_name, TarotName):
        return TAROT_BY_NAME[key_or_name.value.lower()].key
    clean = str(key_or_name).strip()
    if clean in TAROT_BY_KEY:
        return clean
    lower = clean.lower()
    if lower in TAROT_BY_NAME:
        return TAROT_BY_NAME[lower].key
    if f"the {lower}" in TAROT_BY_NAME:
        return TAROT_BY_NAME[f"the {lower}"].key
    return clean


# ---------------------------------------------------------------------------
# 3. G.P_CENTER_POOLS['Voucher'] (32 Vouchers mapped to `balatro_models.VoucherName`)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VoucherProto:
    key: str
    voucher_name: VoucherName
    order: int
    requires: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return self.voucher_name.value


P_VOUCHERS: list[VoucherProto] = [
    VoucherProto("v_overstock_norm", VoucherName.OVERSTOCK, 1),
    VoucherProto(
        "v_overstock_plus", VoucherName.OVERSTOCK_PLUS, 2, ("v_overstock_norm",)
    ),
    VoucherProto("v_clearance_sale", VoucherName.CLEARANCE_SALE, 3),
    VoucherProto("v_liquidation", VoucherName.LIQUIDATION, 4, ("v_clearance_sale",)),
    VoucherProto("v_hone", VoucherName.HONE, 5),
    VoucherProto("v_glow_up", VoucherName.GLOW_UP, 6, ("v_hone",)),
    VoucherProto("v_reroll_surplus", VoucherName.REROLL_SURPLUS, 7),
    VoucherProto("v_reroll_glut", VoucherName.REROLL_GLUT, 8, ("v_reroll_surplus",)),
    VoucherProto("v_crystal_ball", VoucherName.CRYSTAL_BALL, 9),
    VoucherProto("v_omen_globe", VoucherName.OMEN_GLOBE, 10, ("v_crystal_ball",)),
    VoucherProto("v_telescope", VoucherName.TELESCOPE, 11),
    VoucherProto("v_observatory", VoucherName.OBSERVATORY, 12, ("v_telescope",)),
    VoucherProto("v_grabber", VoucherName.GRABBER, 13),
    VoucherProto("v_nacho_tong", VoucherName.NACHO_TONG, 14, ("v_grabber",)),
    VoucherProto("v_wasteful", VoucherName.WASTEFUL, 15),
    VoucherProto("v_recyclomancy", VoucherName.RECYCLOMANCY, 16, ("v_wasteful",)),
    VoucherProto("v_tarot_merchant", VoucherName.TAROT_MERCHANT, 17),
    VoucherProto("v_tarot_tycoon", VoucherName.TAROT_TYCOON, 18, ("v_tarot_merchant",)),
    VoucherProto("v_planet_merchant", VoucherName.PLANET_MERCHANT, 19),
    VoucherProto(
        "v_planet_tycoon", VoucherName.PLANET_TYCOON, 20, ("v_planet_merchant",)
    ),
    VoucherProto("v_seed_money", VoucherName.SEED_MONEY, 21),
    VoucherProto("v_money_tree", VoucherName.MONEY_TREE, 22, ("v_seed_money",)),
    VoucherProto("v_blank", VoucherName.BLANK, 23),
    VoucherProto("v_antimatter", VoucherName.ANTIMATTER, 24, ("v_blank",)),
    VoucherProto("v_magic_trick", VoucherName.MAGIC_TRICK, 25),
    VoucherProto("v_illusion", VoucherName.ILLUSION, 26, ("v_magic_trick",)),
    VoucherProto("v_hieroglyph", VoucherName.HIEROGLYPH, 27),
    VoucherProto("v_petroglyph", VoucherName.PETROGLYPH, 28, ("v_hieroglyph",)),
    VoucherProto("v_directors_cut", VoucherName.DIRECTORS_CUT, 29),
    VoucherProto("v_retcon", VoucherName.RETCON, 30, ("v_directors_cut",)),
    VoucherProto("v_paint_brush", VoucherName.PAINT_BRUSH, 31),
    VoucherProto("v_palette", VoucherName.PALETTE, 32, ("v_paint_brush",)),
]

VOUCHER_BY_KEY: dict[str, VoucherProto] = {v.key: v for v in P_VOUCHERS}
VOUCHER_BY_NAME: dict[str, VoucherProto] = {v.name.lower(): v for v in P_VOUCHERS}

DECK_STARTING_VOUCHERS: dict[str, tuple[str, ...]] = {
    "magic deck": ("v_crystal_ball",),
    "b_magic": ("v_crystal_ball",),
    "nebula deck": ("v_telescope",),
    "b_nebula": ("v_telescope",),
    "zodiac deck": ("v_tarot_merchant", "v_planet_merchant", "v_overstock_norm"),
    "b_zodiac": ("v_tarot_merchant", "v_planet_merchant", "v_overstock_norm"),
}


def normalize_voucher_key(key_or_name: VoucherName | str) -> str:
    """Convert a `VoucherName`, key (`'v_overstock_norm'`), or string (`'Overstock'`) to its `'v_...'` key."""
    if isinstance(key_or_name, VoucherName):
        return VOUCHER_BY_NAME[key_or_name.value.lower()].key
    clean = str(key_or_name).strip()
    if clean in VOUCHER_BY_KEY:
        return clean
    lower = clean.lower()
    if lower in VOUCHER_BY_NAME:
        return VOUCHER_BY_NAME[lower].key
    return clean


# ---------------------------------------------------------------------------
# 4. G.P_CENTER_POOLS['Tag'] (24 Skip Tags mapped to `balatro_models.SkipTag`)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TagProto:
    key: str
    skip_tag: SkipTag
    order: int
    min_ante: int | None = None
    requires: str | None = None

    @property
    def name(self) -> str:
        return self.skip_tag.value


P_TAGS: list[TagProto] = [
    TagProto("tag_uncommon", SkipTag.UNCOMMON_TAG, 1, None),
    TagProto("tag_rare", SkipTag.RARE_TAG, 2, None, "j_blueprint"),
    TagProto("tag_negative", SkipTag.NEGATIVE_TAG, 3, 2, "e_negative"),
    TagProto("tag_foil", SkipTag.FOIL_TAG, 4, None, "e_foil"),
    TagProto("tag_holo", SkipTag.HOLOGRAPHIC_TAG, 5, None, "e_holo"),
    TagProto("tag_polychrome", SkipTag.POLYCHROME_TAG, 6, None, "e_polychrome"),
    TagProto("tag_investment", SkipTag.INVESTMENT_TAG, 7, None),
    TagProto("tag_voucher", SkipTag.VOUCHER_TAG, 8, None),
    TagProto("tag_boss", SkipTag.BOSS_TAG, 9, None),
    TagProto("tag_standard", SkipTag.STANDARD_TAG, 10, 2),
    TagProto("tag_charm", SkipTag.CHARM_TAG, 11, None),
    TagProto("tag_meteor", SkipTag.METEOR_TAG, 12, 2),
    TagProto("tag_buffoon", SkipTag.BUFFOON_TAG, 13, 2),
    TagProto("tag_handy", SkipTag.HANDY_TAG, 14, 2),
    TagProto("tag_garbage", SkipTag.GARBAGE_TAG, 15, 2),
    TagProto("tag_ethereal", SkipTag.ETHEREAL_TAG, 16, 2),
    TagProto("tag_coupon", SkipTag.COUPON_TAG, 17, None),
    TagProto("tag_double", SkipTag.DOUBLE_TAG, 18, None),
    TagProto("tag_juggle", SkipTag.JUGGLE_TAG, 19, None),
    TagProto("tag_d_six", SkipTag.D6_TAG, 20, None),
    TagProto("tag_top_up", SkipTag.TOP_UP_TAG, 21, 2),
    TagProto("tag_skip", SkipTag.SPEED_TAG, 22, None),
    TagProto("tag_orbital", SkipTag.ORBITAL_TAG, 23, 2),
    TagProto("tag_economy", SkipTag.ECONOMY_TAG, 24, None),
]

TAG_BY_KEY: dict[str, TagProto] = {t.key: t for t in P_TAGS}
TAG_BY_NAME: dict[str, TagProto] = {t.name.lower(): t for t in P_TAGS}
TAG_BY_NAME["skip tag"] = TAG_BY_KEY["tag_skip"]


def normalize_tag_key(key_or_name: SkipTag | str) -> str:
    """Convert a `SkipTag`, key (`'tag_orbital'`), or string (`'Orbital Tag'`) to its `'tag_...'` key."""
    if isinstance(key_or_name, SkipTag):
        return TAG_BY_NAME[key_or_name.value.lower()].key
    clean = str(key_or_name).strip()
    if clean in TAG_BY_KEY:
        return clean
    lower = clean.lower()
    if lower in TAG_BY_NAME:
        return TAG_BY_NAME[lower].key
    if f"{lower} tag" in TAG_BY_NAME:
        return TAG_BY_NAME[f"{lower} tag"].key
    return clean


# ---------------------------------------------------------------------------
# 5. G.P_BLINDS Boss Blinds (28 Bosses mapped to `balatro_models.BlindName`)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BossBlindProto:
    key: str
    blind_name: BlindName
    order: int
    min_ante: int
    showdown: bool = False

    @property
    def name(self) -> str:
        return self.blind_name.value


P_BOSS_BLINDS: list[BossBlindProto] = [
    BossBlindProto("bl_arm", BlindName.THE_ARM, 8, 2, False),
    BossBlindProto("bl_club", BlindName.THE_CLUB, 9, 1, False),
    BossBlindProto("bl_eye", BlindName.THE_EYE, 16, 3, False),
    BossBlindProto("bl_final_acorn", BlindName.AMBER_ACORN, 26, 10, True),
    BossBlindProto("bl_final_bell", BlindName.CERULEAN_BELL, 30, 10, True),
    BossBlindProto("bl_final_heart", BlindName.CRIMSON_HEART, 29, 10, True),
    BossBlindProto("bl_final_leaf", BlindName.VERDANT_LEAF, 27, 10, True),
    BossBlindProto("bl_final_vessel", BlindName.VIOLET_VESSEL, 28, 10, True),
    BossBlindProto("bl_fish", BlindName.THE_FISH, 10, 2, False),
    BossBlindProto("bl_flint", BlindName.THE_FLINT, 24, 2, False),
    BossBlindProto("bl_goad", BlindName.THE_GOAD, 12, 1, False),
    BossBlindProto("bl_head", BlindName.THE_HEAD, 22, 1, False),
    BossBlindProto("bl_hook", BlindName.THE_HOOK, 3, 1, False),
    BossBlindProto("bl_house", BlindName.THE_HOUSE, 5, 2, False),
    BossBlindProto("bl_manacle", BlindName.THE_MANACLE, 15, 1, False),
    BossBlindProto("bl_mark", BlindName.THE_MARK, 25, 2, False),
    BossBlindProto("bl_mouth", BlindName.THE_MOUTH, 17, 2, False),
    BossBlindProto("bl_needle", BlindName.THE_NEEDLE, 21, 2, False),
    BossBlindProto("bl_ox", BlindName.THE_OX, 4, 6, False),
    BossBlindProto("bl_pillar", BlindName.THE_PILLAR, 20, 1, False),
    BossBlindProto("bl_plant", BlindName.THE_PLANT, 18, 4, False),
    BossBlindProto("bl_psychic", BlindName.THE_PSYCHIC, 11, 1, False),
    BossBlindProto("bl_serpent", BlindName.THE_SERPENT, 19, 5, False),
    BossBlindProto("bl_tooth", BlindName.THE_TOOTH, 23, 3, False),
    BossBlindProto("bl_wall", BlindName.THE_WALL, 6, 2, False),
    BossBlindProto("bl_water", BlindName.THE_WATER, 13, 2, False),
    BossBlindProto("bl_wheel", BlindName.THE_WHEEL, 7, 2, False),
    BossBlindProto("bl_window", BlindName.THE_WINDOW, 14, 1, False),
]

BOSS_BY_KEY: dict[str, BossBlindProto] = {b.key: b for b in P_BOSS_BLINDS}
BOSS_BY_NAME: dict[str, BossBlindProto] = {b.name.lower(): b for b in P_BOSS_BLINDS}


def get_boss_display_name(proto: BossBlindProto) -> str:
    return proto.name


def normalize_boss_key(key_or_name: BlindName | str) -> str:
    """Convert a `BlindName`, key (`'bl_hook'`), or string (`'The Hook'`) to its `'bl_...'` key."""
    if isinstance(key_or_name, BlindName):
        return BOSS_BY_NAME[key_or_name.value.lower()].key
    clean = str(key_or_name).strip()
    if clean in BOSS_BY_KEY:
        return clean
    lower = clean.lower()
    if lower in BOSS_BY_NAME:
        return BOSS_BY_NAME[lower].key
    if f"the {lower}" in BOSS_BY_NAME:
        return BOSS_BY_NAME[f"the {lower}"].key
    return clean


# ---------------------------------------------------------------------------
# 6. Castle, Ancient Joker, The Idol, Mail-in Rebate, To Do List
# ---------------------------------------------------------------------------


def get_castle_suit(
    rng: BalatroRNG,
    deck: Deck,
    ante: int = 1,
    round_num: int | None = None,
    state: GameState | None = None,
    stream_key: str | None = None,
) -> Suit:
    """Determine the suit Castle selects for the round (`reset_castle_card`, common_events.lua:2312-2324).

    Mid-Ante Deck Mutation Note:
    - If `round_num` (1=Small, 2=Big, 3=Boss) is passed, `rng.ensure_call_count('cas{ante}', round_num - 1)`
      automatically advances the `'cas{ante}'` LCG stream to `round_num - 1` (if not already there)
      without evaluating intermediate decks, and then evaluates `round_num` directly against the current `deck`!
    """
    valid_cards = deck.get_non_stone_cards()
    if not valid_cards:
        suit = Suit.SPADES
    else:
        key = stream_key if stream_key is not None else f"cas{ante}"
        if round_num is not None and round_num >= 1:
            rng.ensure_call_count(key, round_num - 1)
        chosen_card, _ = rng.choice(key, valid_cards)
        suit = chosen_card.suit

    if state is not None:
        state.castle_suit = suit
    return suit


def get_ancient_joker_suit(
    rng: BalatroRNG,
    ante: int = 1,
    current_suit: Suit | str | None = None,
    state: GameState | None = None,
    stream_key: str | None = None,
) -> Suit:
    """Determine the suit Ancient Joker selects for the round (`reset_ancient_card`, common_events.lua:2303-2310)."""
    if current_suit is not None:
        prev_suit: Suit | None = parse_suit(current_suit)
    elif state is not None:
        prev_suit = state.ancient_suit
    else:
        prev_suit = None

    base_order = [Suit.SPADES, Suit.HEARTS, Suit.CLUBS, Suit.DIAMONDS]
    ancient_suits = [s for s in base_order if s != prev_suit]

    key = stream_key if stream_key is not None else f"anc{ante}"
    chosen_suit, _ = rng.choice(key, ancient_suits)

    if state is not None:
        state.ancient_suit = chosen_suit
    return chosen_suit


@dataclass
class IdolTarget:
    """Target card (`rank` and `suit`) selected by The Idol for the round."""

    rank: Rank
    suit: Suit
    card: SimCard | None = None
    ante: int = 1

    @property
    def card_id(self) -> int | None:
        return self.card.id if self.card is not None else None

    def __repr__(self) -> str:
        return f"IdolTarget({self.rank.value} of {self.suit.value}, ante={self.ante})"


def get_idol_card(
    rng: BalatroRNG,
    deck: Deck,
    ante: int = 1,
    round_num: int | None = None,
    state: GameState | None = None,
    stream_key: str | None = None,
) -> IdolTarget:
    """Determine the card (`rank` + `suit`) The Idol selects for the round (`reset_idol_card`, common_events.lua:2271-2286).

    Mid-Ante Deck Mutation Note:
    - If `round_num` (1=Small, 2=Big, 3=Boss) is passed, `rng.ensure_call_count('idol{ante}', round_num - 1)`
      automatically advances `'idol{ante}'` to `round_num - 1` (if not already there) and evaluates
      `round_num` directly against the current `deck`.
    """
    valid_cards = deck.get_non_stone_cards()
    if not valid_cards:
        target = IdolTarget(rank=Rank.ACE, suit=Suit.SPADES, card=None, ante=ante)
    else:
        key = stream_key if stream_key is not None else f"idol{ante}"
        if round_num is not None and round_num >= 1:
            rng.ensure_call_count(key, round_num - 1)
        chosen_card, _ = rng.choice(key, valid_cards)
        target = IdolTarget(
            rank=chosen_card.rank,
            suit=chosen_card.suit,
            card=chosen_card,
            ante=ante,
        )

    if state is not None:
        state.idol_card = (target.rank, target.suit)
    return target


@dataclass
class RebateEvaluation:
    """Outcome of Mail-in Rebate discard evaluation."""

    target_rank: Rank
    matching_cards: list[PlayingCard]
    matches: int
    payout: int
    payout_per_card: int = 5


def get_mail_in_rebate_target(
    rng: BalatroRNG,
    deck: Deck,
    ante: int | None = None,
    round_num: int | None = None,
    state: GameState | None = None,
    stream_key: str | None = None,
) -> Rank:
    """Dynamically determine Mail-in Rebate target rank for the round (`reset_mail_rank`, common_events.lua:2288-2301)."""
    valid_cards = deck.get_non_stone_cards()
    if not valid_cards:
        rank = Rank.ACE
    else:
        key = (
            stream_key
            if stream_key is not None
            else (f"mail{ante}" if ante is not None else "mail1")
        )
        if round_num is not None and round_num >= 1:
            rng.ensure_call_count(key, round_num - 1)
        chosen_card, _ = rng.choice(key, valid_cards)
        rank = chosen_card.rank

    if state is not None:
        state.mail_rank = rank
    return rank


def evaluate_mail_in_rebate_discard(
    target_rank: Rank,
    discarded_cards: Sequence[PlayingCard],
    payout_per_card: int = 5,
) -> RebateEvaluation:
    """Evaluate payout for discarding `PlayingCard`s when Mail-in Rebate is active."""
    matches = [c for c in discarded_cards if card_is_rank(c, target_rank)]
    count = len(matches)
    total_payout = count * payout_per_card
    return RebateEvaluation(
        target_rank=target_rank,
        matching_cards=matches,
        matches=count,
        payout=total_payout,
        payout_per_card=payout_per_card,
    )


def get_to_do_list_hand(
    rng: BalatroRNG,
    current_hand: PokerHand | str | None = None,
    visible_hands: Sequence[PokerHand | str] | None = None,
    state: GameState | None = None,
    is_initial_spawn: bool = False,
    stream_key: str = "to_do",
) -> PokerHand:
    """Determine the `PokerHand` To Do List selects (`card.lua:311-323`, `2975-2984`)."""
    raw_pool = (
        list(visible_hands)
        if visible_hands is not None
        else (
            list(state.visible_hands)
            if state is not None
            else list(DEFAULT_VISIBLE_POKER_HANDS)
        )
    )
    hands_pool: list[PokerHand] = order_poker_hands_luajit(raw_pool)

    raw_prev = (
        current_hand
        if current_hand is not None
        else (state.to_do_hand if state is not None else None)
    )
    prev_hand: PokerHand | None = (
        (raw_prev if isinstance(raw_prev, PokerHand) else PokerHand(str(raw_prev)))
        if raw_prev is not None
        else None
    )

    if is_initial_spawn or prev_hand is None:
        while True:
            chosen, _ = rng.choice(stream_key, hands_pool)
            if chosen != prev_hand or len(hands_pool) <= 1:
                break
    else:
        filtered_pool = [h for h in hands_pool if h != prev_hand]
        if not filtered_pool:
            filtered_pool = hands_pool
        chosen, _ = rng.choice(stream_key, filtered_pool)

    if state is not None:
        state.to_do_hand = chosen
    return chosen


# ---------------------------------------------------------------------------
# 7. Cartomancer & Vagabond Tarot Generation (`get_current_pool('Tarot', ...)`)
# ---------------------------------------------------------------------------


@dataclass
class TarotPrediction:
    """Result of a deterministic Tarot card generation (Cartomancer, Vagabond, etc.)."""

    key: str
    tarot_name: TarotName
    ante: int
    source: str
    stream_key: str
    resample_count: int = 0

    @property
    def name(self) -> str:
        return self.tarot_name.value

    def to_tarot_card(self) -> Tarot:
        """Instantiate a `balatro_models.Tarot` card object."""
        return Tarot(name=self.tarot_name)

    def to_model(self) -> Tarot:
        """Alias for `to_tarot_card()`."""
        return self.to_tarot_card()

    def __repr__(self) -> str:
        res = f", resamples={self.resample_count}" if self.resample_count else ""
        return f"TarotPrediction({self.name!r} [{self.key}], ante={self.ante}, stream={self.stream_key!r}{res})"


def get_blocked_card_names(state: Any, card_type: CardType) -> set[Any]:
    """Collects card names currently in inventory, external shop, or open pack.

    Matches Balatro's `G.GAME.used_jokers` blocking across consumables/jokers,
    `shop_cards`, and `pack_cards` so pools cannot roll duplicates while those
    cards remain in any active area (unless Showman is owned).
    """
    if state is None:
        return set()
    blocked: set[Any] = set()
    inv = getattr(state, "inventory", None)
    if inv is not None:
        if card_type == CardType.JOKER:
            for j in getattr(inv, "jokers", ()) or ():
                if getattr(j, "name", None) is not None:
                    blocked.add(j.name)
        else:
            for c in getattr(inv, "consumables", ()) or ():
                if getattr(c, "type", None) == card_type and getattr(c, "name", None) is not None:
                    blocked.add(c.name)
    for region_attr in ("shop_cards", "pack_cards"):
        for item in getattr(state, region_attr, ()) or ():
            if getattr(item, "type", None) == card_type and getattr(item, "name", None) is not None:
                blocked.add(item.name)
    return blocked


def state_has_showman(state: Any) -> bool:
    """Returns True if state holds an active (undebuffed) Showman."""
    inv = getattr(state, "inventory", None) if state is not None else None
    if not inv:
        return False
    return any(
        getattr(j, "name", None) == JokerName.SHOWMAN and not getattr(j, "debuffed", False)
        for j in (getattr(inv, "jokers", ()) or ())
    )


def poll_tarot_pool(
    rng: BalatroRNG,
    ante: int = 1,
    key_append: str = "",
    state: GameState | None = None,
    active_tarots: set[TarotName | str] | Sequence[TarotName | str] | None = None,
    banned_keys: set[str] | None = None,
    has_showman: bool = False,
    source_label: str = "Tarot",
) -> TarotPrediction:
    """Poll `G.P_CENTER_POOLS['Tarot']` matching `get_current_pool('Tarot', nil, nil, key_append)`."""
    eff_ante = max(1, ante)
    pool_key = f"Tarot{key_append}{eff_ante}"

    used = set()
    if state is not None:
        used.update(normalize_tarot_key(x) for x in state.active_tarots)
    if active_tarots is not None:
        used.update(normalize_tarot_key(x) for x in active_tarots)

    banned = set()
    if state is not None:
        banned.update(normalize_tarot_key(x) for x in state.banned_keys)
    if banned_keys is not None:
        banned.update(normalize_tarot_key(x) for x in banned_keys)

    showman = has_showman or (state.has_showman if state is not None else False)

    pool: list[str] = []
    pool_size = 0
    for t in P_TAROTS:
        is_avail = True
        if t.key in banned:
            is_avail = False
        elif not showman and t.key in used:
            is_avail = False

        if is_avail:
            pool.append(t.key)
            pool_size += 1
        else:
            pool.append("UNAVAILABLE")

    if pool_size == 0:
        pool = ["c_strength"]

    center, _ = rng.choice(pool_key, pool)
    it = 1
    resamples = 0
    while center == "UNAVAILABLE":
        it += 1
        resamples += 1
        center, _ = rng.choice(f"{pool_key}_resample{it}", pool)

    proto = TAROT_BY_KEY[center]
    return TarotPrediction(
        key=proto.key,
        tarot_name=proto.tarot_name,
        ante=ante,
        source=source_label,
        stream_key=pool_key,
        resample_count=resamples,
    )


def predict_cartomancer_tarot(
    rng: BalatroRNG,
    ante: int = 1,
    state: GameState | None = None,
    active_tarots: set[TarotName | str] | Sequence[TarotName | str] | None = None,
    has_showman: bool = False,
) -> TarotPrediction:
    """Predict which `TarotName` Cartomancer gives when a blind is selected (`card.lua:2551`)."""
    return poll_tarot_pool(
        rng=rng,
        ante=ante,
        key_append="car",
        state=state,
        active_tarots=active_tarots,
        has_showman=has_showman,
        source_label="Cartomancer",
    )


def predict_vagabond_tarot(
    rng: BalatroRNG,
    ante: int = 1,
    state: GameState | None = None,
    active_tarots: set[TarotName | str] | Sequence[TarotName | str] | None = None,
    has_showman: bool = False,
) -> TarotPrediction:
    """Predict which `TarotName` Vagabond gives when a hand is played with <= $4 (`card.lua:3750`)."""
    return poll_tarot_pool(
        rng=rng,
        ante=ante,
        key_append="vag",
        state=state,
        active_tarots=active_tarots,
        has_showman=has_showman,
        source_label="Vagabond",
    )


# ---------------------------------------------------------------------------
# 8. Marble Joker / Stone Joker Card Generation (`card.lua:2580-2601`)
# ---------------------------------------------------------------------------


@dataclass
class StoneCardResult:
    """Result of Marble Joker ("Stone Joker") adding a Stone `PlayingCard` on blind select."""

    card: SimCard
    front_key: str

    @property
    def hidden_suit(self) -> Suit:
        """The underlying `Suit` revealed if the Stone Card is given a different enhancement or vampired."""
        return self.card.suit

    @property
    def hidden_rank(self) -> Rank:
        """The underlying `Rank` revealed if the Stone Card is given a different enhancement or vampired."""
        return self.card.rank

    @property
    def suit(self) -> Suit:
        return self.card.suit

    @property
    def rank(self) -> Rank:
        return self.card.rank

    def __repr__(self) -> str:
        return f"StoneCardResult(hidden={self.hidden_rank.value} of {self.hidden_suit.value} [{self.front_key}], sort_id={self.card.id})"


def predict_stone_joker_card(
    rng: BalatroRNG,
    state: GameState | None = None,
    add_to_deck: bool = True,
    stream_key: str = "marb_fr",
) -> StoneCardResult:
    """Predict the `PlayingCard` Marble Joker ("Stone Joker") adds to the deck on blind select (`card.lua:2580-2601`)."""
    chosen_front_key, _ = rng.choice(stream_key, P_CARDS_KEYS)
    suit, rank = parse_card_key(chosen_front_key)

    stone_card = SimCard(
        suit=suit,
        rank=rank,
        enhancement=Enhancement.STONE,
    )

    if state is not None and add_to_deck:
        state.deck.add_card(stone_card)

    return StoneCardResult(
        card=stone_card,
        front_key=chosen_front_key,
    )


predict_marble_joker_card = predict_stone_joker_card


# ---------------------------------------------------------------------------
# 9. Order of Boss Blinds (`get_new_boss`, common_events.lua:2338-2383)
# ---------------------------------------------------------------------------


@dataclass
class BossBlindPrediction:
    """Result of rolling a Boss Blind via `get_new_boss()`."""

    key: str
    blind_name: BlindName
    name: str
    ante: int
    is_showdown: bool
    is_reroll: bool = False
    eligible_pool_size: int = 0
    min_use_tier: int = 0

    def __repr__(self) -> str:
        kind = "Showdown" if self.is_showdown else "Normal"
        rr = ", reroll=True" if self.is_reroll else ""
        return f"BossBlindPrediction({self.name!r} [{self.key}], ante={self.ante}, {kind}{rr})"


def get_next_boss_blind(
    rng: BalatroRNG,
    ante: int = 1,
    state: GameState | None = None,
    bosses_used: dict[str, int] | None = None,
    banned_keys: set[str] | None = None,
    win_ante: int = 8,
    is_reroll: bool = False,
    stream_key: str = "boss",
) -> BossBlindPrediction:
    """Roll the next Boss Blind matching `get_new_boss()` (`common_events.lua:2338-2383`)."""
    if state is not None:
        used_map = state.bosses_used
        banned = state.banned_keys
        eff_win_ante = state.win_ante
    else:
        used_map = bosses_used if bosses_used is not None else {}
        banned = banned_keys if banned_keys is not None else set()
        eff_win_ante = win_ante

    for b in P_BOSS_BLINDS:
        if b.key not in used_map:
            used_map[b.key] = 0

    eff_ante = max(1, ante)
    eligible: list[BossBlindProto] = []
    for b in P_BOSS_BLINDS:
        if b.key in banned:
            continue
        if not b.showdown:
            if b.min_ante <= eff_ante and (eff_ante % eff_win_ante != 0 or ante < 2):
                eligible.append(b)
        else:
            if ante % eff_win_ante == 0 and ante >= 2:
                eligible.append(b)

    if not eligible:
        raise ValueError(f"No eligible boss blinds found for ante={ante}")

    min_use = min(used_map[b.key] for b in eligible)
    filtered = [b for b in eligible if used_map[b.key] <= min_use]
    filtered.sort(key=lambda b: b.key)

    chosen_proto, _ = rng.choice(stream_key, filtered)
    used_map[chosen_proto.key] += 1

    if state is not None:
        state.current_boss = chosen_proto.key

    return BossBlindPrediction(
        key=chosen_proto.key,
        blind_name=chosen_proto.blind_name,
        name=get_boss_display_name(chosen_proto),
        ante=ante,
        is_showdown=chosen_proto.showdown,
        is_reroll=is_reroll,
        eligible_pool_size=len(filtered),
        min_use_tier=min_use,
    )


# ---------------------------------------------------------------------------
# 10. Order of Skip Tags (`get_next_tag_key`, common_events.lua:1914-1925, 1982-1986)
# ---------------------------------------------------------------------------


@dataclass
class TagPrediction:
    """Result of rolling a Skip Tag via `get_next_tag_key()`."""

    key: str
    skip_tag: SkipTag
    ante: int
    blind_type: str  # "Small" or "Big"
    stream_key: str
    resample_count: int = 0
    orbital_hand: PokerHand | None = None

    @property
    def name(self) -> str:
        return self.skip_tag.value

    def __repr__(self) -> str:
        orb = f", orbital_hand={self.orbital_hand.value!r}" if self.orbital_hand else ""
        res = f", resamples={self.resample_count}" if self.resample_count else ""
        return f"TagPrediction({self.name!r} [{self.key}], ante={self.ante}, blind={self.blind_type!r}{orb}{res})"


def get_next_tag(
    rng: BalatroRNG,
    ante: int = 1,
    blind_type: str = "Small",
    state: GameState | None = None,
    banned_keys: set[str] | None = None,
    roll_orbital_choice: bool = False,
) -> TagPrediction:
    """Roll the next Skip Tag matching `get_next_tag_key()` (`common_events.lua:1914-1925`, `1982-1986`)."""
    eff_ante = max(1, ante)
    pool_key = f"Tag{eff_ante}"

    banned = set()
    if state is not None:
        banned.update(normalize_tag_key(x) for x in state.banned_keys)
    if banned_keys is not None:
        banned.update(normalize_tag_key(x) for x in banned_keys)

    pool: list[str] = []
    pool_size = 0
    for t in P_TAGS:
        is_avail = True
        if t.key in banned:
            is_avail = False
        elif t.min_ante is not None and t.min_ante > ante:
            is_avail = False

        if is_avail:
            pool.append(t.key)
            pool_size += 1
        else:
            pool.append("UNAVAILABLE")

    if pool_size == 0:
        pool = ["tag_handy"]

    center, _ = rng.choice(pool_key, pool)
    it = 1
    resamples = 0
    while center == "UNAVAILABLE":
        it += 1
        resamples += 1
        center, _ = rng.choice(f"{pool_key}_resample{it}", pool)

    orbital_hand: PokerHand | None = None
    if roll_orbital_choice:
        hands = (
            state.visible_hands if state is not None else DEFAULT_VISIBLE_POKER_HANDS
        )
        orbital_hand, _ = rng.choice("orbital", hands)

    proto = TAG_BY_KEY[center]
    return TagPrediction(
        key=proto.key,
        skip_tag=proto.skip_tag,
        ante=ante,
        blind_type=blind_type,
        stream_key=pool_key,
        resample_count=resamples,
        orbital_hand=orbital_hand,
    )


def get_orbital_tag_hands_for_ante(
    rng: BalatroRNG,
    visible_hands: Sequence[PokerHand | str] | None = None,
    state: GameState | None = None,
) -> dict[str, PokerHand]:
    """Roll `G.GAME.orbital_choices[ante]` for `'Small'`, `'Big'`, and `'Boss'` (`UI_definitions.lua:1506-1516`)."""
    raw_hands = (
        list(visible_hands)
        if visible_hands is not None
        else (
            list(state.visible_hands)
            if state is not None
            else list(DEFAULT_VISIBLE_POKER_HANDS)
        )
    )
    hands = [h if isinstance(h, PokerHand) else PokerHand(str(h)) for h in raw_hands]
    small_hand, _ = rng.choice("orbital", hands)
    big_hand, _ = rng.choice("orbital", hands)
    boss_hand, _ = rng.choice("orbital", hands)
    return {
        "Small": small_hand,
        "Big": big_hand,
        "Boss": boss_hand,
    }


# ---------------------------------------------------------------------------
# 11. Order of Vouchers (`get_next_voucher_key`, common_events.lua:1901-1912, 1989-2007)
# ---------------------------------------------------------------------------


@dataclass
class VoucherPrediction:
    """Result of rolling a Voucher via `get_next_voucher_key(_from_tag)`."""

    key: str
    voucher_name: VoucherName
    ante: int
    from_tag: bool
    stream_key: str
    resample_count: int = 0

    @property
    def name(self) -> str:
        return self.voucher_name.value

    def to_voucher(self) -> Voucher:
        """Instantiate a `balatro_models.Voucher` object."""
        return Voucher(name=self.voucher_name)

    def to_model(self) -> Voucher:
        """Alias for `to_voucher()`."""
        return self.to_voucher()

    def __repr__(self) -> str:
        src = "VoucherTag" if self.from_tag else f"Ante {self.ante} Shop"
        res = f", resamples={self.resample_count}" if self.resample_count else ""
        return f"VoucherPrediction({self.name!r} [{self.key}], {src}, stream={self.stream_key!r}{res})"


def get_next_voucher(
    rng: BalatroRNG,
    ante: int = 1,
    from_tag: bool = False,
    state: GameState | None = None,
    used_vouchers: set[VoucherName | str] | None = None,
    shop_vouchers: Sequence[VoucherName | str] | None = None,
    banned_keys: set[str] | None = None,
    add_to_shop_vouchers: bool = False,
) -> VoucherPrediction:
    """Roll the next Voucher matching `get_next_voucher_key(_from_tag)` (`common_events.lua:1901-1912`, `1989-2007`)."""
    eff_ante = max(1, ante)
    pool_key = "Voucher_fromtag" if from_tag else f"Voucher{eff_ante}"

    redeemed: set[str] = set()
    if state is not None:
        redeemed.update(normalize_voucher_key(x) for x in state.used_vouchers)
    if used_vouchers is not None:
        redeemed.update(normalize_voucher_key(x) for x in used_vouchers)

    in_shop: set[str] = set()
    if state is not None:
        in_shop.update(normalize_voucher_key(x) for x in state.shop_vouchers)
    if shop_vouchers is not None:
        in_shop.update(normalize_voucher_key(x) for x in shop_vouchers)

    banned: set[str] = set()
    if state is not None:
        banned.update(normalize_voucher_key(x) for x in state.banned_keys)
    if banned_keys is not None:
        banned.update(normalize_voucher_key(x) for x in banned_keys)

    pool: list[str] = []
    pool_size = 0
    for v in P_VOUCHERS:
        is_avail = True
        if v.key in redeemed or v.key in in_shop or v.key in banned:
            is_avail = False
        elif v.requires:
            for req in v.requires:
                if req not in redeemed:
                    is_avail = False
                    break

        if is_avail:
            pool.append(v.key)
            pool_size += 1
        else:
            pool.append("UNAVAILABLE")

    if pool_size == 0:
        pool = ["v_blank"]

    center, _ = rng.choice(pool_key, pool)
    it = 1
    resamples = 0
    while center == "UNAVAILABLE":
        it += 1
        resamples += 1
        center, _ = rng.choice(f"{pool_key}_resample{it}", pool)

    if state is not None and add_to_shop_vouchers:
        if center not in state.shop_vouchers:
            state.shop_vouchers.append(center)

    proto = VOUCHER_BY_KEY[center]
    return VoucherPrediction(
        key=proto.key,
        voucher_name=proto.voucher_name,
        ante=ante,
        from_tag=from_tag,
        stream_key=pool_key,
        resample_count=resamples,
    )


# ---------------------------------------------------------------------------
# 12. Certificate Event (`card.lua:2463-2475`)
# ---------------------------------------------------------------------------


@dataclass
class CertificateResult:
    """Result of Certificate joker start-of-round card generation."""

    card: SimCard
    front_key: str
    seal_key: str
    ante: int

    @property
    def suit(self) -> Suit:
        return self.card.suit

    @property
    def rank(self) -> Rank:
        return self.card.rank

    @property
    def seal(self) -> Seal:
        return self.card.seal


def predict_certificate(
    rng: BalatroRNG,
    ante: int = 1,
    state: GameState | None = None,
    add_to_state: bool = True,
    ante_scoped: bool = True,
) -> CertificateResult:
    """Predict and generate Certificate joker `PlayingCard`."""
    if ante_scoped:
        key_fr = f"cert_fr{ante}"
        key_sl = f"cert_sl{ante}"
        chosen_front_key, _ = rng.choice(key_fr, P_CARDS_KEYS)
        suit, rank = parse_card_key(chosen_front_key)
        chosen_seal, _ = rng.choice(key_sl, P_SEALS_KEYS)
        seal_label = seal_balatro_name(chosen_seal)
    else:
        chosen_front_key, _ = rng.choice("cert_fr", P_CARDS_KEYS)
        suit, rank = parse_card_key(chosen_front_key)
        seal_roll = rng.random("certsl")
        if seal_roll > 0.75:
            chosen_seal = Seal.RED
        elif seal_roll > 0.5:
            chosen_seal = Seal.BLUE
        elif seal_roll > 0.25:
            chosen_seal = Seal.GOLD
        else:
            chosen_seal = Seal.PURPLE
        seal_label = seal_balatro_name(chosen_seal)

    generated_card = SimCard(suit=suit, rank=rank, seal=chosen_seal)

    if state is not None and add_to_state:
        state.deck.add_card(generated_card)
        state.hand.append(generated_card)

    return CertificateResult(
        card=generated_card,
        front_key=chosen_front_key,
        seal_key=seal_label,
        ante=ante,
    )


# ---------------------------------------------------------------------------
# 13. Lucky Card Evaluation (`card.lua:1292-1299`)
# ---------------------------------------------------------------------------


@dataclass
class LuckyHitResult:
    """Outcome of Lucky Card probability checks."""

    mult: bool  # 1 in 5 base (+20 Mult)
    money: bool  # 1 in 15 base (+$20 cash)
    mult_roll: float
    money_roll: float
    oops_multiplier: float

    @property
    def any_hit(self) -> bool:
        return self.mult or self.money

    def to_dict(self) -> dict[str, bool]:
        return {"mult": self.mult, "money": self.money}


def evaluate_lucky_hit(
    rng: BalatroRNG,
    oops_count: int = 0,
    probability_modifier: float | None = None,
) -> LuckyHitResult:
    """Evaluate deterministic mid-round triggers for a scored Lucky Card (`card.lua:1292-1299`)."""
    modifier = (
        probability_modifier
        if probability_modifier is not None
        else float(2**oops_count)
    )

    roll_mult = rng.random("lucky_mult")
    roll_money = rng.random("lucky_money")

    mult_success = roll_mult < ((1.0 * modifier) / 5.0)
    money_success = roll_money < ((1.0 * modifier) / 15.0)

    return LuckyHitResult(
        mult=mult_success,
        money=money_success,
        mult_roll=roll_mult,
        money_roll=roll_money,
        oops_multiplier=modifier,
    )


# ---------------------------------------------------------------------------
# 14. 1-in-N Outcome Resolvers
# ---------------------------------------------------------------------------


class EventResolver:
    """Deterministic roll solvers for probability-based Balatro jokers and cards."""

    def __init__(self, rng: BalatroRNG) -> None:
        self.rng = rng

    def check_odds(
        self,
        stream_key: str,
        numerator: int,
        denominator: int,
        oops_multiplier: float = 1.0,
    ) -> bool:
        roll = self.rng.random(stream_key)
        threshold = (numerator * oops_multiplier) / float(denominator)
        return roll < threshold

    def wheel_of_fortune(self, oops_multiplier: float = 1.0) -> bool:
        return self.check_odds("wheel_of_fortune", 1, 4, oops_multiplier)

    def space_joker(
        self, oops_multiplier: float = 1.0, stream_key: str = "space_joker"
    ) -> bool:
        return self.check_odds(stream_key, 1, 4, oops_multiplier)

    def bloodstone(
        self, oops_multiplier: float = 1.0, base_denominator: int = 2
    ) -> bool:
        return self.check_odds("bloodstone", 1, base_denominator, oops_multiplier)

    def glass_card_shatter(self, oops_multiplier: float = 1.0) -> bool:
        return self.check_odds("glass", 1, 4, oops_multiplier)

    def business_card(
        self, oops_multiplier: float = 1.0, stream_key: str = "business_card"
    ) -> bool:
        return self.check_odds(stream_key, 1, 2, oops_multiplier)

    def gros_michel_extinct(self, oops_multiplier: float = 1.0) -> bool:
        return self.check_odds("gros_michel", 1, 6, oops_multiplier)

    def cavendish_extinct(self, oops_multiplier: float = 1.0) -> bool:
        return self.check_odds("cavendish", 1, 1000, oops_multiplier)

    def reserved_parking(self, oops_multiplier: float = 1.0) -> bool:
        return self.check_odds("parking", 1, 2, oops_multiplier)

    def eight_ball(self, oops_multiplier: float = 1.0) -> bool:
        return self.check_odds("8ball", 1, 4, oops_multiplier)

    def hallucination(self, oops_multiplier: float = 1.0, ante: int = 1) -> bool:
        return self.check_odds(f"halu{ante}", 1, 2, oops_multiplier)


# ---------------------------------------------------------------------------
# 15. Pools and Deterministic Generators (Common Jokers, Spectrals, Edition Polling)
# ---------------------------------------------------------------------------

P_COMMON_JOKERS: list[tuple[str, JokerName]] = [
    ("j_joker", JokerName.JOKER),
    ("j_greedy_joker", JokerName.GREEDY_JOKER),
    ("j_lusty_joker", JokerName.LUSTY_JOKER),
    ("j_wrathful_joker", JokerName.WRATHFUL_JOKER),
    ("j_gluttenous_joker", JokerName.GLUTTONOUS_JOKER),
    ("j_jolly", JokerName.JOLLY_JOKER),
    ("j_zany", JokerName.ZANY_JOKER),
    ("j_mad", JokerName.MAD_JOKER),
    ("j_crazy", JokerName.CRAZY_JOKER),
    ("j_droll", JokerName.DROLL_JOKER),
    ("j_sly", JokerName.SLY_JOKER),
    ("j_wily", JokerName.WILY_JOKER),
    ("j_clever", JokerName.CLEVER_JOKER),
    ("j_devious", JokerName.DEVIOUS_JOKER),
    ("j_crafty", JokerName.CRAFTY_JOKER),
    ("j_half", JokerName.HALF_JOKER),
    ("j_credit_card", JokerName.CREDIT_CARD),
    ("j_banner", JokerName.BANNER),
    ("j_mystic_summit", JokerName.MYSTIC_SUMMIT),
    ("j_8_ball", JokerName.EIGHT_BALL),
    ("j_misprint", JokerName.MISPRINT),
    ("j_raised_fist", JokerName.RAISED_FIST),
    ("j_chaos", JokerName.CHAOS_THE_CLOWN),
    ("j_scary_face", JokerName.SCARY_FACE),
    ("j_abstract", JokerName.ABSTRACT_JOKER),
    ("j_delayed_grat", JokerName.DELAYED_GRATIFICATION),
    ("j_gros_michel", JokerName.GROS_MICHEL),
    ("j_even_steven", JokerName.EVEN_STEVEN),
    ("j_odd_todd", JokerName.ODD_TODD),
    ("j_scholar", JokerName.SCHOLAR),
    ("j_business", JokerName.BUSINESS_CARD),
    ("j_supernova", JokerName.SUPERNOVA),
    ("j_ride_the_bus", JokerName.RIDE_THE_BUS),
    ("j_egg", JokerName.EGG),
    ("j_runner", JokerName.RUNNER),
    ("j_ice_cream", JokerName.ICE_CREAM),
    ("j_splash", JokerName.SPLASH),
    ("j_blue_joker", JokerName.BLUE_JOKER),
    ("j_faceless", JokerName.FACELESS_JOKER),
    ("j_green_joker", JokerName.GREEN_JOKER),
    ("j_superposition", JokerName.SUPERPOSITION),
    ("j_todo_list", JokerName.TO_DO_LIST),
    ("j_cavendish", JokerName.CAVENDISH),
    ("j_red_card", JokerName.RED_CARD),
    ("j_square", JokerName.SQUARE_JOKER),
    ("j_riff_raff", JokerName.RIFF_RAFF),
    ("j_photograph", JokerName.PHOTOGRAPH),
    ("j_reserved_parking", JokerName.RESERVED_PARKING),
    ("j_mail", JokerName.MAIL_IN_REBATE),
    ("j_hallucination", JokerName.HALLUCINATION),
    ("j_fortune_teller", JokerName.FORTUNE_TELLER),
    ("j_juggler", JokerName.JUGGLER),
    ("j_drunkard", JokerName.DRUNKARD),
    ("j_golden", JokerName.GOLDEN_JOKER),
    ("j_popcorn", JokerName.POPCORN),
    ("j_walkie_talkie", JokerName.WALKIE_TALKIE),
    ("j_smiley", JokerName.SMILEY_FACE),
    ("j_ticket", JokerName.GOLDEN_TICKET),
    ("j_swashbuckler", JokerName.SWASHBUCKLER),
    ("j_hanging_chad", JokerName.HANGING_CHAD),
    ("j_shoot_the_moon", JokerName.SHOOT_THE_MOON),
]

P_SPECTRALS: list[tuple[str, SpectralName]] = [
    ("c_familiar", SpectralName.FAMILIAR),
    ("c_grim", SpectralName.GRIM),
    ("c_incantation", SpectralName.INCANTATION),
    ("c_talisman", SpectralName.TALISMAN),
    ("c_aura", SpectralName.AURA),
    ("c_wraith", SpectralName.WRAITH),
    ("c_sigil", SpectralName.SIGIL),
    ("c_ouija", SpectralName.OUIJA),
    ("c_ectoplasm", SpectralName.ECTOPLASM),
    ("c_immolate", SpectralName.IMMOLATE),
    ("c_ankh", SpectralName.ANKH),
    ("c_deja_vu", SpectralName.DEJA_VU),
    ("c_hex", SpectralName.HEX),
    ("c_trance", SpectralName.TRANCE),
    ("c_medium", SpectralName.MEDIUM),
    ("c_cryptid", SpectralName.CRYPTID),
]


def poll_edition(
    rng: BalatroRNG,
    stream_key: str,
    mod: float = 1.0,
    no_neg: bool = False,
    guaranteed: bool = False,
) -> Edition:
    """Replicates Balatro's `poll_edition(_key, _mod, _no_neg, _guaranteed)` (common_events.lua:2055-2076)."""
    poll = rng.random(stream_key)
    if guaranteed:
        if not no_neg and poll > 1.0 - (0.003 * 25):
            return Edition.NEGATIVE
        elif poll > 1.0 - (0.006 * 25):
            return Edition.POLYCHROME
        elif poll > 1.0 - (0.02 * 25):
            return Edition.HOLOGRAPHIC
        elif poll > 1.0 - (0.04 * 25):
            return Edition.FOIL
        return Edition.FOIL
    if not no_neg and poll > 1.0 - (0.003 * mod):
        return Edition.NEGATIVE
    elif poll > 1.0 - (0.006 * mod):
        return Edition.POLYCHROME
    elif poll > 1.0 - (0.02 * mod):
        return Edition.HOLOGRAPHIC
    elif poll > 1.0 - (0.04 * mod):
        return Edition.FOIL
    return Edition.NONE


def predict_riff_raff_jokers(
    rng: BalatroRNG,
    ante: int = 1,
    count: int = 2,
    owned_jokers: set[JokerName] | Sequence[JokerName] | None = None,
    has_showman: bool = False,
) -> list[Joker]:
    """Generates 1 or 2 Common Jokers for Riff-Raff (card.lua:2534-2541)."""
    used = set(owned_jokers or ())
    jokers: list[Joker] = []
    for _ in range(count):
        pool: list[JokerName | str] = []
        pool_size = 0
        for j in P_COMMON_JOKERS:
            if has_showman or j[1] not in used:
                pool.append(j[1])
                pool_size += 1
            else:
                pool.append("UNAVAILABLE")
        if pool_size == 0:
            pool = [JokerName.JOKER]
        chosen, _ = rng.choice("Joker1rif", pool)
        it = 1
        while chosen == "UNAVAILABLE":
            it += 1
            chosen, _ = rng.choice(f"Joker1rif_resample{it}", pool)
        if not has_showman:
            used.add(chosen)
        ed = poll_edition(rng, f"edirif{ante}")
        jokers.append(Joker(name=chosen, edition=ed))
    return jokers


def poll_spectral_pool(
    rng: BalatroRNG,
    ante: int = 1,
    key_append: str = "",
    active_spectrals: set[SpectralName | str] | Sequence[SpectralName | str] | None = None,
    banned_keys: set[str] | None = None,
    has_showman: bool = False,
) -> Spectral:
    """Polls G.P_CENTER_POOLS['Spectral'] matching `get_current_pool('Spectral', nil, nil, key_append)`."""
    eff_ante = max(1, ante)
    pool_key = f"Spectral{key_append}{eff_ante}"
    used = set(active_spectrals or ())
    banned = banned_keys or set()
    pool: list[SpectralName | str] = []
    pool_size = 0
    for key, s_name in P_SPECTRALS:
        is_avail = key not in banned and s_name not in banned
        if is_avail and not has_showman and (s_name in used or key in used):
            is_avail = False
        if is_avail:
            pool.append(s_name)
            pool_size += 1
        else:
            pool.append("UNAVAILABLE")
    if pool_size == 0:
        pool = [SpectralName.INCANTATION]
    chosen, _ = rng.choice(pool_key, pool)
    it = 1
    while chosen == "UNAVAILABLE":
        it += 1
        chosen, _ = rng.choice(f"{pool_key}_resample{it}", pool)
    return Spectral(name=chosen)


def predict_sixth_sense_spectral(
    rng: BalatroRNG,
    ante: int = 1,
    active_spectrals: set[SpectralName | str] | Sequence[SpectralName | str] | None = None,
    has_showman: bool = False,
) -> Spectral:
    return poll_spectral_pool(
        rng,
        ante=ante,
        key_append="sixth",
        active_spectrals=active_spectrals,
        has_showman=has_showman,
    )


def predict_seance_spectral(
    rng: BalatroRNG,
    ante: int = 1,
    active_spectrals: set[SpectralName | str] | Sequence[SpectralName | str] | None = None,
    has_showman: bool = False,
) -> Spectral:
    return poll_spectral_pool(
        rng,
        ante=ante,
        key_append="sea",
        active_spectrals=active_spectrals,
        has_showman=has_showman,
    )


def predict_superposition_tarot(
    rng: BalatroRNG,
    ante: int = 1,
    active_tarots: set[TarotName | str] | Sequence[TarotName | str] | None = None,
    has_showman: bool = False,
) -> TarotPrediction:
    return poll_tarot_pool(
        rng,
        ante=ante,
        key_append="sup",
        active_tarots=active_tarots,
        has_showman=has_showman,
        source_label="Superposition",
    )


def predict_purple_seal_tarot(
    rng: BalatroRNG,
    ante: int = 1,
    active_tarots: set[TarotName | str] | Sequence[TarotName | str] | None = None,
    has_showman: bool = False,
) -> TarotPrediction:
    return poll_tarot_pool(
        rng,
        ante=ante,
        # balatro_source/card.lua:2260 uses key_append='8ba' for Purple Seal too
        key_append="8ba",
        active_tarots=active_tarots,
        has_showman=has_showman,
        source_label="Purple Seal",
    )


def predict_eight_ball_tarot(
    rng: BalatroRNG,
    ante: int = 1,
    active_tarots: set[TarotName | str] | Sequence[TarotName | str] | None = None,
    has_showman: bool = False,
) -> TarotPrediction:
    return poll_tarot_pool(
        rng,
        ante=ante,
        key_append="8ba",
        active_tarots=active_tarots,
        has_showman=has_showman,
        source_label="8 Ball",
    )


# ---------------------------------------------------------------------------
# 11. Spectral Destruction & Card Generation Mechanics
# ---------------------------------------------------------------------------

_SPECTRAL_SUITS: list[Suit] = [Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS]
_ENHANCED_CENTERS: list[Enhancement] = [
    Enhancement.BONUS,
    Enhancement.MULT,
    Enhancement.WILD,
    Enhancement.GLASS,
    Enhancement.STEEL,
    Enhancement.GOLD,
    Enhancement.LUCKY,
]


def predict_random_destroy(rng: BalatroRNG, hand_cards: list[Any]) -> Any | None:
    """Predicts which card in hand is destroyed by Familiar, Grim, or Incantation.

    Replicates:
    `destroyed_cards[#destroyed_cards+1] = pseudorandom_element(G.hand.cards, pseudoseed('random_destroy'))`
    (card.lua:1293)
    """
    if not hand_cards:
        return None
    pseed = rng.step_seed("random_destroy")
    tw = Tw223(pseed)
    idx = tw.random_range(1, len(hand_cards)) - 1
    return hand_cards[idx]


def predict_immolate_destructions(
    rng: BalatroRNG, hand_cards: list[Any], count: int = 5
) -> list[Any]:
    """Predicts which cards in hand are destroyed by Immolate.

    Replicates:
    `table.sort(temp_hand, function (a, b) return not a.playing_card or not b.playing_card or a.playing_card < b.playing_card end)`
    `pseudoshuffle(temp_hand, pseudoseed('immolate'))`
    (card.lua:1341-1346)
    """
    if not hand_cards:
        return []
    temp = list(hand_cards)
    temp.sort(key=lambda c: getattr(c, "id", getattr(c, "sort_id", 0)))
    pseed = rng.step_seed("immolate")
    tw = Tw223(pseed)
    for i in range(len(temp) - 1, 0, -1):
        j = tw.random_range(1, i + 1) - 1
        temp[i], temp[j] = temp[j], temp[i]
    return temp[:count]


def predict_familiar_spawns(rng: BalatroRNG, count: int = 3) -> list[PlayingCard]:
    """Generates 3 enhanced face cards for Familiar (card.lua:1319-1336)."""
    cards: list[PlayingCard] = []
    for _ in range(count):
        r, _ = rng.choice("familiar_create", [Rank.JACK, Rank.QUEEN, Rank.KING])
        s, _ = rng.choice("familiar_create", _SPECTRAL_SUITS)
        enh, _ = rng.choice("spe_card", _ENHANCED_CENTERS)
        cards.append(PlayingCard(rank=r, suit=s, enhancement=enh))
    return cards


def predict_grim_spawns(rng: BalatroRNG, count: int = 2) -> list[PlayingCard]:
    """Generates 2 enhanced Aces for Grim (card.lua:1323-1336)."""
    cards: list[PlayingCard] = []
    for _ in range(count):
        s, _ = rng.choice("grim_create", _SPECTRAL_SUITS)
        enh, _ = rng.choice("spe_card", _ENHANCED_CENTERS)
        cards.append(PlayingCard(rank=Rank.ACE, suit=s, enhancement=enh))
    return cards


def predict_incantation_spawns(rng: BalatroRNG, count: int = 4) -> list[PlayingCard]:
    """Generates 4 enhanced numbered cards for Incantation (card.lua:1326-1336)."""
    ranks = [
        Rank.TWO,
        Rank.THREE,
        Rank.FOUR,
        Rank.FIVE,
        Rank.SIX,
        Rank.SEVEN,
        Rank.EIGHT,
        Rank.NINE,
        Rank.TEN,
    ]
    cards: list[PlayingCard] = []
    for _ in range(count):
        r, _ = rng.choice("incantation_create", ranks)
        s, _ = rng.choice("incantation_create", _SPECTRAL_SUITS)
        enh, _ = rng.choice("spe_card", _ENHANCED_CENTERS)
        cards.append(PlayingCard(rank=r, suit=s, enhancement=enh))
    return cards


P_UNCOMMON_JOKERS: list[tuple[str, JokerName]] = [
    ("j_stencil", JokerName.JOKER_STENCIL),
    ("j_four_fingers", JokerName.FOUR_FINGERS),
    ("j_mime", JokerName.MIME),
    ("j_ceremonial", JokerName.CEREMONIAL_DAGGER),
    ("j_marble", JokerName.MARBLE_JOKER),
    ("j_loyalty_card", JokerName.LOYALTY_CARD),
    ("j_dusk", JokerName.DUSK),
    ("j_fibonacci", JokerName.FIBONACCI),
    ("j_steel_joker", JokerName.STEEL_JOKER),
    ("j_hack", JokerName.HACK),
    ("j_pareidolia", JokerName.PAREIDOLIA),
    ("j_space", JokerName.SPACE_JOKER),
    ("j_burglar", JokerName.BURGLAR),
    ("j_blackboard", JokerName.BLACKBOARD),
    ("j_sixth_sense", JokerName.SIXTH_SENSE),
    ("j_constellation", JokerName.CONSTELLATION),
    ("j_hiker", JokerName.HIKER),
    ("j_card_sharp", JokerName.CARD_SHARP),
    ("j_madness", JokerName.MADNESS),
    ("j_seance", JokerName.SEANCE),
    ("j_vampire", JokerName.VAMPIRE),
    ("j_shortcut", JokerName.SHORTCUT),
    ("j_hologram", JokerName.HOLOGRAM),
    ("j_cloud_9", JokerName.CLOUD_9),
    ("j_rocket", JokerName.ROCKET),
    ("j_midas_mask", JokerName.MIDAS_MASK),
    ("j_luchador", JokerName.LUCHADOR),
    ("j_gift", JokerName.GIFT_CARD),
    ("j_turtle_bean", JokerName.TURTLE_BEAN),
    ("j_erosion", JokerName.EROSION),
    ("j_to_the_moon", JokerName.TO_THE_MOON),
    ("j_stone", JokerName.STONE_JOKER),
    ("j_lucky_cat", JokerName.LUCKY_CAT),
    ("j_bull", JokerName.BULL),
    ("j_diet_cola", JokerName.DIET_COLA),
    ("j_trading", JokerName.TRADING_CARD),
    ("j_flash", JokerName.FLASH_CARD),
    ("j_trousers", JokerName.SPARE_TROUSERS),
    ("j_ramen", JokerName.RAMEN),
    ("j_selzer", JokerName.SELTZER),
    ("j_castle", JokerName.CASTLE),
    ("j_mr_bones", JokerName.MR_BONES),
    ("j_acrobat", JokerName.ACROBAT),
    ("j_sock_and_buskin", JokerName.SOCK_AND_BUSKIN),
    ("j_troubadour", JokerName.TROUBADOUR),
    ("j_certificate", JokerName.CERTIFICATE),
    ("j_smeared", JokerName.SMEARED_JOKER),
    ("j_throwback", JokerName.THROWBACK),
    ("j_rough_gem", JokerName.ROUGH_GEM),
    ("j_bloodstone", JokerName.BLOODSTONE),
    ("j_arrowhead", JokerName.ARROWHEAD),
    ("j_onyx_agate", JokerName.ONYX_AGATE),
    ("j_glass", JokerName.GLASS_JOKER),
    ("j_ring_master", JokerName.SHOWMAN),
    ("j_flower_pot", JokerName.FLOWER_POT),
    ("j_merry_andy", JokerName.MERRY_ANDY),
    ("j_oops", JokerName.OOPS_ALL_6S),
    ("j_idol", JokerName.THE_IDOL),
    ("j_seeing_double", JokerName.SEEING_DOUBLE),
    ("j_matador", JokerName.MATADOR),
    ("j_satellite", JokerName.SATELLITE),
    ("j_cartomancer", JokerName.CARTOMANCER),
    ("j_astronomer", JokerName.ASTRONOMER),
    ("j_bootstraps", JokerName.BOOTSTRAPS),
]

P_RARE_JOKERS: list[tuple[str, JokerName]] = [
    ("j_dna", JokerName.DNA),
    ("j_vagabond", JokerName.VAGABOND),
    ("j_baron", JokerName.BARON),
    ("j_obelisk", JokerName.OBELISK),
    ("j_baseball", JokerName.BASEBALL_CARD),
    ("j_ancient", JokerName.ANCIENT_JOKER),
    ("j_campfire", JokerName.CAMPFIRE),
    ("j_blueprint", JokerName.BLUEPRINT),
    ("j_wee", JokerName.WEE_JOKER),
    ("j_hit_the_road", JokerName.HIT_THE_ROAD),
    ("j_duo", JokerName.THE_DUO),
    ("j_trio", JokerName.THE_TRIO),
    ("j_family", JokerName.THE_FAMILY),
    ("j_order", JokerName.THE_ORDER),
    ("j_tribe", JokerName.THE_TRIBE),
    ("j_stuntman", JokerName.STUNTMAN),
    ("j_invisible", JokerName.INVISIBLE_JOKER),
    ("j_brainstorm", JokerName.BRAINSTORM),
    ("j_drivers_license", JokerName.DRIVER),
    ("j_burnt", JokerName.BURNT_JOKER),
]

P_LEGENDARY_JOKERS: list[tuple[str, JokerName]] = [
    ("j_caino", JokerName.CANIO),
    ("j_triboulet", JokerName.TRIBOULET),
    ("j_yorick", JokerName.YORICK),
    ("j_chicot", JokerName.CHICOT),
    ("j_perkeo", JokerName.PERKEO),
]

# (key, PlanetName, hand_type_required_if_softlocked)
P_PLANETS: list[tuple[str, PlanetName, PokerHand | None]] = [
    ("c_mercury", PlanetName.MERCURY, None),
    ("c_venus", PlanetName.VENUS, None),
    ("c_earth", PlanetName.EARTH, None),
    ("c_mars", PlanetName.MARS, None),
    ("c_jupiter", PlanetName.JUPITER, None),
    ("c_saturn", PlanetName.SATURN, None),
    ("c_uranus", PlanetName.URANUS, None),
    ("c_neptune", PlanetName.NEPTUNE, None),
    ("c_pluto", PlanetName.PLUTO, None),
    ("c_planet_x", PlanetName.PLANET_X, PokerHand.FIVE_OF_A_KIND),
    ("c_ceres", PlanetName.CERES, PokerHand.FLUSH_HOUSE),
    ("c_eris", PlanetName.ERIS, PokerHand.FLUSH_FIVE),
]

_ALL_JOKER_BY_KEY: dict[str, JokerName] = dict(
    P_COMMON_JOKERS + P_UNCOMMON_JOKERS + P_RARE_JOKERS + P_LEGENDARY_JOKERS
)
_PLANET_BY_KEY: dict[str, PlanetName] = {k: name for k, name, _ in P_PLANETS}


def poll_planet_pool(
    rng: BalatroRNG,
    ante: int = 1,
    key_append: str = "",
    hand_play_counts: dict[PokerHand, int] | None = None,
    active_planets: set[PlanetName] | None = None,
    banned_keys: set[str] | None = None,
    has_showman: bool = False,
) -> Planet:
    """Polls `G.P_CENTER_POOLS['Planet']` matching `get_current_pool('Planet', ...)`."""
    eff_ante = max(1, ante)
    pool_key = f"Planet{key_append}{eff_ante}"
    counts = hand_play_counts or {}
    used = active_planets or set()
    banned = banned_keys or set()

    pool: list[str] = []
    pool_size = 0
    for key, p_name, req_hand in P_PLANETS:
        is_avail = True
        if key in banned:
            is_avail = False
        elif not has_showman and p_name in used:
            is_avail = False
        elif req_hand is not None and counts.get(req_hand, 0) <= 0:
            is_avail = False

        if is_avail:
            pool.append(key)
            pool_size += 1
        else:
            pool.append("UNAVAILABLE")

    if pool_size == 0:
        pool = ["c_pluto"]

    chosen, _ = rng.choice(pool_key, pool)
    it = 1
    while chosen == "UNAVAILABLE":
        it += 1
        chosen, _ = rng.choice(f"{pool_key}_resample{it}", pool)

    return Planet(name=_PLANET_BY_KEY.get(chosen, PlanetName.PLUTO))


def poll_joker_pool(
    rng: BalatroRNG,
    ante: int = 1,
    rarity_tier: int | None = None,
    rarity_roll: float | None = None,
    legendary: bool = False,
    key_append: str = "",
    owned_jokers: set[JokerName] | None = None,
    gros_michel_extinct: bool = False,
    banned_keys: set[str] | None = None,
    has_showman: bool = False,
    include_edition: bool = True,
    edition_mod: float = 1.0,
) -> Joker:
    """Polls `G.P_JOKER_RARITY_POOLS` matching `create_card('Joker', ..., legendary, rarity, ..., append)`."""
    eff_ante = max(1, ante)
    if legendary:
        tier = 4
        # common_events.lua:1971 omits _append when _legendary is true
        pool_key = f"Joker4{eff_ante}"
        raw_pool = P_LEGENDARY_JOKERS
    else:
        if rarity_tier is not None:
            tier = max(1, min(3, rarity_tier))
        else:
            roll = (
                float(rarity_roll)
                if rarity_roll is not None
                else rng.random(f"rarity{eff_ante}{key_append}")
            )
            tier = 3 if roll > 0.95 else (2 if roll > 0.7 else 1)
        pool_key = f"Joker{tier}{key_append}{eff_ante}"
        raw_pool = (
            P_RARE_JOKERS
            if tier == 3
            else (P_UNCOMMON_JOKERS if tier == 2 else P_COMMON_JOKERS)
        )

    owned = owned_jokers or set()
    banned = banned_keys or set()
    pool: list[str] = []
    pool_size = 0
    for key, j_name in raw_pool:
        is_avail = True
        if key in banned:
            is_avail = False
        elif not has_showman and j_name in owned:
            is_avail = False
        elif j_name == JokerName.GROS_MICHEL and gros_michel_extinct:
            is_avail = False
        elif j_name == JokerName.CAVENDISH and not gros_michel_extinct:
            is_avail = False

        if is_avail:
            pool.append(key)
            pool_size += 1
        else:
            pool.append("UNAVAILABLE")

    if pool_size == 0:
        pool = ["j_joker"]

    chosen, _ = rng.choice(pool_key, pool)
    it = 1
    while chosen == "UNAVAILABLE":
        it += 1
        chosen, _ = rng.choice(f"{pool_key}_resample{it}", pool)

    j_name = _ALL_JOKER_BY_KEY.get(chosen, JokerName.JOKER)
    ed = (
        poll_edition(rng, f"edi{key_append}{eff_ante}", mod=edition_mod)
        if include_edition
        else Edition.NONE
    )
    return Joker(name=j_name, edition=ed)


def predict_ouija_rank(rng: BalatroRNG) -> Rank:
    """Predicts target rank chosen by Ouija Spectral (`card.lua:1247`)."""
    ranks = [
        Rank.TWO,
        Rank.THREE,
        Rank.FOUR,
        Rank.FIVE,
        Rank.SIX,
        Rank.SEVEN,
        Rank.EIGHT,
        Rank.NINE,
        Rank.TEN,
        Rank.JACK,
        Rank.QUEEN,
        Rank.KING,
        Rank.ACE,
    ]
    chosen, _ = rng.choice("ouija", ranks)
    return chosen


def predict_ankh_choice(rng: BalatroRNG, jokers: Sequence[Joker]) -> Joker | None:
    """Predicts surviving Joker chosen by Ankh (`card.lua:1434`)."""
    if not jokers:
        return None
    indexed = list(enumerate(jokers))
    if all(getattr(j, "sort_id", 0) or getattr(j, "id", 0) for _, j in indexed):
        indexed.sort(
            key=lambda pair: getattr(pair[1], "sort_id", 0) or getattr(pair[1], "id", 0)
        )
    chosen_pair, _ = rng.choice("ankh_choice", indexed)
    return chosen_pair[1]


def predict_hook_discards(
    rng: BalatroRNG | None,
    hand_cards: Sequence[Any],
    count: int = 2,
) -> list[Any]:
    """Selects up to `count` remaining hand cards discarded by The Hook (`blind.lua:464-487`)."""
    remaining = list(hand_cards)
    if not remaining:
        return []
    to_pick = min(count, len(remaining))
    if rng is None:
        return remaining[:to_pick]

    picked: list[Any] = []
    # Match pseudorandom_element(sort_id) ordering when card ids are populated
    indexed = list(enumerate(remaining))
    if all(getattr(getattr(item, "card", item), "id", 0) for _, item in indexed):
        indexed.sort(
            key=lambda pair: getattr(getattr(pair[1], "card", pair[1]), "id", 0)
        )

    pool = list(indexed)
    for _ in range(to_pick):
        if not pool:
            break
        chosen_pair, _ = rng.choice("hook", pool)
        pool.remove(chosen_pair)
        picked.append(chosen_pair[1])
    return picked


def generate_erratic_deck(rng: BalatroRNG | None = None) -> list[PlayingCard]:
    """Generates a 52-card Erratic Deck matching `game.lua:2342, 2367-2373`."""
    chosen_keys: list[str] = []
    for i in range(len(P_CARDS_KEYS)):
        if rng is not None:
            k, _ = rng.choice("erratic", P_CARDS_KEYS)
        else:
            k = P_CARDS_KEYS[i]
        chosen_keys.append(k)
    # table.sort on (suit_char .. rank_char)
    chosen_keys.sort(key=lambda k: k.replace("_", ""))
    cards: list[PlayingCard] = []
    for idx, key in enumerate(chosen_keys, start=1):
        suit, rank = parse_card_key(key)
        cards.append(PlayingCard(rank=rank, suit=suit, id=idx))
    return cards
