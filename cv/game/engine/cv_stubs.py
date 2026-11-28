"""
CV Model Query Stubs & Verification Interface:
Acts as the computer vision observation interface for visible on-screen elements
(cards in player hand, shop offerings, joker/consumable bars, booster packs).

Enforces strict GameState location checks:
- cv_detect_hand_cards: Raises InvalidLocationError if called when hand cards are not visible
  (e.g. in shop outside of a pack, in celestial/buffoon/standard packs, selecting a blind, cash out screen).
- cv_detect_shop_cards: Raises InvalidLocationError if called when not in the main shop
  (e.g. selecting a blind, in a blind, inside a booster pack, cash out screen).
- cv_detect_pack_cards: Raises InvalidLocationError if not inside an open booster pack
  (or if currently on the cash out screen).
- cv_detect_joker_bar: Raises InvalidLocationError if not in an active gameplay screen
  (select, blind/cash-out, shop, or pack).

Deterministic non-visible state, RNG rolls, and Boss Blinds are resolved via PRNG (`game.engine.rng_sim`).
`cv_detect_boss_blind` has been removed in favor of `get_boss_blind_from_seed` and `state.roll_boss_blind()`.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

try:
    from game.balatro_models import (
        BlindName,
        BoosterPack,
        BoosterSize,
        BoosterType,
        Card,
        CardType,
        Edition,
        Enhancement,
        HandCard,
        Joker,
        JokerName,
        Location,
        Planet,
        PlanetName,
        PlayingCard,
        Rank,
        Seal,
        Spectral,
        SpectralName,
        Suit,
        Tarot,
        TarotName,
        Voucher,
        VoucherName,
    )
except ImportError:
    from balatro_models import (
        BlindName,
        BoosterPack,
        BoosterSize,
        BoosterType,
        Card,
        CardType,
        Edition,
        Enhancement,
        HandCard,
        Joker,
        JokerName,
        Location,
        Planet,
        PlanetName,
        PlayingCard,
        Rank,
        Seal,
        Spectral,
        SpectralName,
        Suit,
        Tarot,
        TarotName,
        Voucher,
        VoucherName,
    )


class InvalidLocationError(ValueError):
    """Raised when CV card detection is invoked in an invalid game location or UI state."""


# =========================================================================
# ATTRIBUTE MAPPERS
# =========================================================================

_RANK_MAP: dict[str, Rank] = {
    "2": Rank.TWO,
    "3": Rank.THREE,
    "4": Rank.FOUR,
    "5": Rank.FIVE,
    "6": Rank.SIX,
    "7": Rank.SEVEN,
    "8": Rank.EIGHT,
    "9": Rank.NINE,
    "10": Rank.TEN,
    "Jack": Rank.JACK,
    "Queen": Rank.QUEEN,
    "King": Rank.KING,
    "Ace": Rank.ACE,
    "Stone": Rank.NONE,
    "none": Rank.NONE,
    "None": Rank.NONE,
}

_SUIT_MAP: dict[str, Suit] = {
    "spades": Suit.SPADES,
    "hearts": Suit.HEARTS,
    "clubs": Suit.CLUBS,
    "diamonds": Suit.DIAMONDS,
    "none": Suit.NONE,
    "": Suit.NONE,
}

_EDITION_MAP: dict[str, Edition] = {
    "none": Edition.NONE,
    "None": Edition.NONE,
    "foil": Edition.FOIL,
    "holo": Edition.HOLOGRAPHIC,
    "holographic": Edition.HOLOGRAPHIC,
    "poly": Edition.POLYCHROME,
    "polychrome": Edition.POLYCHROME,
    "neg": Edition.NEGATIVE,
    "negative": Edition.NEGATIVE,
}

_ENHANCEMENT_MAP: dict[str, Enhancement] = {
    "none": Enhancement.NONE,
    "None": Enhancement.NONE,
    "bonus": Enhancement.BONUS,
    "mult": Enhancement.MULT,
    "wild": Enhancement.WILD,
    "glass": Enhancement.GLASS,
    "steel": Enhancement.STEEL,
    "stone": Enhancement.STONE,
    "gold": Enhancement.GOLD,
    "lucky": Enhancement.LUCKY,
}

_SEAL_MAP: dict[str, Seal] = {
    "none": Seal.NONE,
    "None": Seal.NONE,
    "red": Seal.RED,
    "gold": Seal.GOLD,
    "blue": Seal.BLUE,
    "purple": Seal.PURPLE,
}


def _clean_str(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", s).lower()


_JOKER_LOOKUP: dict[str, JokerName] = {}
for j in JokerName:
    _JOKER_LOOKUP[_clean_str(j.value)] = j
    _JOKER_LOOKUP[_clean_str(j.name)] = j

_JOKER_ALIASES: dict[str, JokerName] = {
    "abstract": JokerName.ABSTRACT_JOKER,
    "ancient": JokerName.ANCIENT_JOKER,
    "baseball": JokerName.BASEBALL_CARD,
    "burnt": JokerName.BURNT_JOKER,
    "business": JokerName.BUSINESS_CARD,
    "caino": JokerName.CANIO,
    "ceremonial": JokerName.CEREMONIAL_DAGGER,
    "chaos": JokerName.CHAOS_THE_CLOWN,
    "clever": JokerName.CLEVER_JOKER,
    "crafty": JokerName.CRAFTY_JOKER,
    "crazy": JokerName.CRAZY_JOKER,
    "delayedgrat": JokerName.DELAYED_GRATIFICATION,
    "devious": JokerName.DEVIOUS_JOKER,
    "driverslicense": JokerName.DRIVER,
    "droll": JokerName.DROLL_JOKER,
    "duo": JokerName.THE_DUO,
    "faceless": JokerName.FACELESS_JOKER,
    "family": JokerName.THE_FAMILY,
    "flash": JokerName.FLASH_CARD,
    "gift": JokerName.GIFT_CARD,
    "glass": JokerName.GLASS_JOKER,
    "gluttenousjoker": JokerName.GLUTTONOUS_JOKER,
    "golden": JokerName.GOLDEN_JOKER,
    "half": JokerName.HALF_JOKER,
    "idol": JokerName.THE_IDOL,
    "invisible": JokerName.INVISIBLE_JOKER,
    "jolly": JokerName.JOLLY_JOKER,
    "mad": JokerName.MAD_JOKER,
    "mail": JokerName.MAIL_IN_REBATE,
    "marble": JokerName.MARBLE_JOKER,
    "oops": JokerName.OOPS_ALL_6S,
    "order": JokerName.THE_ORDER,
    "ringmaster": JokerName.SHOWMAN,
    "selzer": JokerName.SELTZER,
    "sly": JokerName.SLY_JOKER,
    "smeared": JokerName.SMEARED_JOKER,
    "smiley": JokerName.SMILEY_FACE,
    "space": JokerName.SPACE_JOKER,
    "square": JokerName.SQUARE_JOKER,
    "stencil": JokerName.JOKER_STENCIL,
    "stone": JokerName.STONE_JOKER,
    "ticket": JokerName.GOLDEN_TICKET,
    "trading": JokerName.TRADING_CARD,
    "tribe": JokerName.THE_TRIBE,
    "trio": JokerName.THE_TRIO,
    "trousers": JokerName.SPARE_TROUSERS,
    "wee": JokerName.WEE_JOKER,
    "wily": JokerName.WILY_JOKER,
    "zany": JokerName.ZANY_JOKER,
}
_JOKER_LOOKUP.update(_JOKER_ALIASES)

_TAROT_LOOKUP: dict[str, TarotName] = {}
for t in TarotName:
    _TAROT_LOOKUP[_clean_str(t.value)] = t
    _TAROT_LOOKUP[_clean_str(t.name)] = t
    if t.value.startswith("The "):
        _TAROT_LOOKUP[_clean_str(t.value[4:])] = t
_TAROT_LOOKUP["heirophant"] = TarotName.HIEROPHANT

_PLANET_LOOKUP: dict[str, PlanetName] = {}
for p in PlanetName:
    _PLANET_LOOKUP[_clean_str(p.value)] = p
    _PLANET_LOOKUP[_clean_str(p.name)] = p

_SPECTRAL_LOOKUP: dict[str, SpectralName] = {}
for sp in SpectralName:
    _SPECTRAL_LOOKUP[_clean_str(sp.value)] = sp
    _SPECTRAL_LOOKUP[_clean_str(sp.name)] = sp
    if sp.value.startswith("The "):
        _SPECTRAL_LOOKUP[_clean_str(sp.value[4:])] = sp
_SPECTRAL_LOOKUP["soul"] = SpectralName.THE_SOUL

_VOUCHER_LOOKUP: dict[str, VoucherName] = {}
for v in VoucherName:
    _VOUCHER_LOOKUP[_clean_str(v.value)] = v
    _VOUCHER_LOOKUP[_clean_str(v.name)] = v
_VOUCHER_LOOKUP["overstocknorm"] = VoucherName.OVERSTOCK
_VOUCHER_LOOKUP["overstockplus"] = VoucherName.OVERSTOCK_PLUS

_BOOSTER_TYPE_MAP: dict[str, BoosterType] = {
    "arcana": BoosterType.ARCANA,
    "celestial": BoosterType.CELESTIAL,
    "standard": BoosterType.STANDARD,
    "buffoon": BoosterType.BUFFOON,
    "spectral": BoosterType.SPECTRAL,
}

_BOOSTER_SIZE_MAP: dict[str, BoosterSize] = {
    "normal": BoosterSize.REGULAR,
    "regular": BoosterSize.REGULAR,
    "jumbo": BoosterSize.JUMBO,
    "mega": BoosterSize.MEGA,
}


class ShopCardsResult(list):
    """List of shop shelf cards (Jokers, Consumables, PlayingCards) with attached voucher and booster shelves."""

    def __init__(
        self,
        cards: list[Any],
        vouchers: list[Voucher] | None = None,
        packs: list[BoosterPack] | None = None,
        joker_bar: list[Joker] | None = None,
        consumables: list[Any] | None = None,
        pack_cards: list[Any] | None = None,
        raw_output: Any = None,
    ) -> None:
        super().__init__(cards)
        self.shop_cards = cards
        self.vouchers = vouchers or []
        self.packs = packs or []
        self.joker_bar = joker_bar or []
        self.consumables = consumables or []
        self.pack_cards = pack_cards or []
        self.raw_output = raw_output


# =========================================================================
# LOCATION VALIDATION
# =========================================================================


def _resolve_state_and_frame(first: Any, second: Any = None) -> tuple[Any, Any]:
    """Helper to extract (state, frame) flexibly from either cv_detect(state, frame) or cv_detect(frame, state)."""
    if hasattr(first, "location"):
        return first, second
    if hasattr(second, "location"):
        return second, first
    return first, second


def validate_hand_cards_location(state: Any) -> None:
    """Validates that the current GameState allows observing cards in the player's hand.

    Raises:
        InvalidLocationError: If hand cards are not visible (e.g. in main shop outside a pack,
                             in celestial/buffoon/standard packs, selecting a blind, or cash out screen).
    """
    if state is None:
        raise ValueError("GameState must be provided to validate location.")

    loc = getattr(state, "location", None)
    round_won = getattr(state, "round_won", False)
    active_pack = getattr(state, "active_pack", None)

    # 1. Cash out screen (post_processing uses Location.CASH_OUT internally;
    #    GameState keeps Location.BLIND with round_won=True)
    if round_won:
        raise InvalidLocationError(
            f"Cannot detect hand cards: round is won / currently in cash out screen (location='{loc}')."
        )

    # 2. Blind selection screen
    if loc in (Location.SELECT, "select"):
        raise InvalidLocationError(
            "Cannot detect hand cards: currently selecting a blind (location='select'). Hand cards are not visible."
        )

    # 3. In the shop outside of a booster pack
    if loc in (Location.SHOP, "shop") and active_pack is None:
        raise InvalidLocationError(
            "Cannot detect hand cards: currently in the shop (location='shop') and not inside a pack. Hand cards are not visible."
        )

    # 4. Inside a pack
    if loc in (Location.PACK, "pack") or active_pack is not None:
        if active_pack is None:
            raise InvalidLocationError(
                "Cannot detect hand cards: gamestate indicates pack location but active_pack is None."
            )
        b_type = getattr(active_pack, "booster_type", None)
        if b_type not in (BoosterType.ARCANA, BoosterType.SPECTRAL):
            raise InvalidLocationError(
                f"Cannot detect hand cards: inside a {b_type} booster pack where hand cards are not visible. "
                "Hand cards are only visible in Arcana and Spectral packs."
            )
        return

    # 5. During active blind play
    if loc in (Location.BLIND, "blind"):
        return

    raise InvalidLocationError(
        f"Cannot detect hand cards: unrecognized or invalid game location '{loc}'."
    )


def validate_shop_cards_location(state: Any) -> None:
    """Validates that the current GameState allows observing shop cards.

    Raises:
        InvalidLocationError: If not in the main shop (e.g. selecting a blind, in a blind,
                             inside a pack in shop or skip menu, or in cash out screen).
    """
    if state is None:
        raise ValueError("GameState must be provided to validate location.")

    loc = getattr(state, "location", None)
    round_won = getattr(state, "round_won", False)
    active_pack = getattr(state, "active_pack", None)

    # 1. Cash out screen (post_processing uses Location.CASH_OUT internally;
    #    GameState keeps Location.BLIND with round_won=True)
    if round_won:
        raise InvalidLocationError(
            f"Cannot detect shop cards: currently in the cash out screen (location='{loc}')."
        )

    # 2. In a blind
    if loc in (Location.BLIND, "blind"):
        raise InvalidLocationError(
            "Cannot detect shop cards: currently in a blind (location='blind')."
        )

    # 3. Selecting a blind
    if loc in (Location.SELECT, "select"):
        raise InvalidLocationError(
            "Cannot detect shop cards: currently selecting a blind (location='select')."
        )

    # 4. Inside a booster pack (whether in shop or opened from skip tag)
    if loc in (Location.PACK, "pack") or active_pack is not None:
        pack_type = (
            getattr(active_pack, "booster_type", "unknown")
            if active_pack
            else "unknown"
        )
        raise InvalidLocationError(
            f"Cannot detect shop cards: currently inside a booster pack ({pack_type}) (location='pack')."
        )

    # 5. Must be in Location.SHOP
    if loc in (Location.SHOP, "shop"):
        return

    raise InvalidLocationError(
        f"Cannot detect shop cards: unrecognized or invalid game location '{loc}'. Expected location='shop'."
    )


def validate_pack_cards_location(state: Any) -> None:
    """Validates that the current GameState allows observing open booster pack cards.

    Raises:
        InvalidLocationError: If not currently inside an open booster pack (e.g. cash out screen,
                              selecting a blind, active blind play, or main shop with no active pack).
    """
    if state is None:
        raise ValueError("GameState must be provided to validate location.")

    loc = getattr(state, "location", None)
    round_won = getattr(state, "round_won", False)
    active_pack = getattr(state, "active_pack", None)

    if round_won or loc in (Location.CASH_OUT, "cash out", "cash_out"):
        raise InvalidLocationError(
            f"Cannot detect pack cards: currently in the cash out screen (location='{loc}')."
        )

    if loc in (Location.PACK, "pack") or active_pack is not None:
        return

    raise InvalidLocationError(
        f"Cannot detect pack cards: not inside a booster pack (location='{loc}')."
    )


def validate_joker_bar_location(state: Any) -> None:
    """Validates that the current GameState corresponds to an in-run screen showing the top Joker bar.

    Raises:
        InvalidLocationError: If GameState has no valid in-run screen location.
    """
    if state is None:
        raise ValueError("GameState must be provided to validate location.")

    loc = getattr(state, "location", None)
    valid_locations = (
        Location.SELECT,
        "select",
        Location.BLIND,
        "blind",
        Location.SHOP,
        "shop",
        Location.PACK,
        "pack",
        Location.CASH_OUT,
        "cash out",
        "cash_out",
    )
    if loc in valid_locations:
        return

    raise InvalidLocationError(
        f"Cannot detect joker bar: unrecognized or invalid game location '{loc}'."
    )


# =========================================================================
# ITEM DESERIALIZERS
# =========================================================================


def _convert_card_result_to_hand_card(card_res: Any) -> HandCard:
    """Converts a CardResult dataclass from card recognition pipeline into HandCard."""
    r_str = getattr(card_res, "rank", "none")
    s_str = getattr(card_res, "suit", "none")
    ed_str = getattr(card_res, "edition", "none")
    enh_str = getattr(card_res, "enhancement", "none")
    seal_str = getattr(card_res, "seal", "none")
    selected = bool(getattr(card_res, "selected", False))
    debuffed = bool(getattr(card_res, "debuffed", False))
    face_down = bool(getattr(card_res, "face_down", False))

    rank = _RANK_MAP.get(r_str, Rank.NONE)
    suit = _SUIT_MAP.get(s_str.lower(), Suit.NONE)
    enh = _ENHANCEMENT_MAP.get(enh_str.lower(), Enhancement.NONE)
    ed = _EDITION_MAP.get(ed_str.lower(), Edition.NONE)
    seal = _SEAL_MAP.get(seal_str.lower(), Seal.NONE)

    if r_str == "Stone" or enh == Enhancement.STONE:
        rank = Rank.NONE
        suit = Suit.NONE
        enh = Enhancement.STONE

    playing_card = PlayingCard(
        rank=rank,
        suit=suit,
        enhancement=enh,
        edition=ed,
        seal=seal,
        debuffed=debuffed,
    )
    return HandCard(
        card=playing_card,
        selected=selected,
        facing_back=face_down,
    )


def _convert_shop_item_result(item: Any) -> Any:
    """Converts a ShopItemResult dataclass into a Joker, Consumable, Voucher, or BoosterPack."""
    key = getattr(item, "key", "")
    card_type = getattr(item, "card_type", "")
    edition_str = getattr(item, "edition", "none")
    ed = _EDITION_MAP.get(edition_str.lower(), Edition.NONE)
    eternal = bool(getattr(item, "eternal", False))
    perishable = bool(getattr(item, "perishable", False))
    rental = bool(getattr(item, "rental", False))
    debuffed = bool(getattr(item, "debuffed", False))

    clean_key = _clean_str(key[2:] if len(key) > 2 and key[1] == "_" else key)

    # 1. Booster Pack
    if key.startswith("p_") or card_type == "Booster":
        tokens = key.split("_")
        b_type = _BOOSTER_TYPE_MAP.get(
            tokens[1] if len(tokens) > 1 else "", BoosterType.ARCANA
        )
        b_size = _BOOSTER_SIZE_MAP.get(
            tokens[2] if len(tokens) > 2 else "", BoosterSize.REGULAR
        )
        cost = (
            4
            if b_size == BoosterSize.REGULAR
            else (6 if b_size == BoosterSize.JUMBO else 8)
        )
        choose = 2 if b_size == BoosterSize.MEGA else 1
        extra = 5 if b_size in (BoosterSize.JUMBO, BoosterSize.MEGA) else 3
        if b_type in (BoosterType.BUFFOON, BoosterType.SPECTRAL):
            extra = 4 if b_size in (BoosterSize.JUMBO, BoosterSize.MEGA) else 2
        return BoosterPack(
            booster_type=b_type,
            size=b_size,
            cost=cost,
            choose=choose,
            extra=extra,
        )

    # 2. Voucher
    if key.startswith("v_") or card_type == "Voucher":
        v_name = _VOUCHER_LOOKUP.get(clean_key, VoucherName.OVERSTOCK)
        return Voucher(name=v_name)

    # 3. Joker
    if key.startswith("j_") or card_type == "Joker":
        j_name = _JOKER_LOOKUP.get(clean_key, JokerName.JOKER)
        return Joker(
            name=j_name,
            edition=ed,
            eternal=eternal,
            perishable=perishable,
            rental=rental,
            debuffed=debuffed,
        )

    # 4. Consumables
    if key.startswith("c_") or card_type in ("Tarot", "Planet", "Spectral"):
        if clean_key in _TAROT_LOOKUP or card_type == "Tarot":
            t_name = _TAROT_LOOKUP.get(clean_key, TarotName.FOOL)
            return Tarot(name=t_name, edition=ed)
        elif clean_key in _PLANET_LOOKUP or card_type == "Planet":
            p_name = _PLANET_LOOKUP.get(clean_key, PlanetName.MERCURY)
            return Planet(name=p_name, edition=ed)
        elif clean_key in _SPECTRAL_LOOKUP or card_type == "Spectral":
            sp_name = _SPECTRAL_LOOKUP.get(clean_key, SpectralName.FAMILIAR)
            return Spectral(name=sp_name, edition=ed)

    # 5. Playing Card (from Standard pack or Shop Magic Trick / Illusion voucher)
    if card_type in ("PlayingCard", "playing_card") or getattr(item, "card_details", None) is not None:
        details = getattr(item, "card_details", None)
        if details is not None:
            r_str = getattr(details, "rank", "None")
            s_str = getattr(details, "suit", "None")
            enh_str = getattr(details, "enhancement", "None")
            ed_str = getattr(details, "edition", edition_str)
            seal_str = getattr(details, "seal", "None")
            p_debuffed = bool(getattr(details, "debuffed", debuffed))
        else:
            r_str = "None"
            s_str = "None"
            enh_str = "None"
            ed_str = edition_str
            seal_str = "None"
            p_debuffed = debuffed

        rank = _RANK_MAP.get(r_str, Rank.NONE)
        suit = _SUIT_MAP.get(s_str.lower(), Suit.NONE)
        enh = _ENHANCEMENT_MAP.get(enh_str.lower(), Enhancement.NONE)
        ed = _EDITION_MAP.get(ed_str.lower(), Edition.NONE)
        seal = _SEAL_MAP.get(seal_str.lower(), Seal.NONE)

        if r_str == "Stone" or enh == Enhancement.STONE:
            rank = Rank.NONE
            suit = Suit.NONE
            enh = Enhancement.STONE

        return PlayingCard(
            rank=rank,
            suit=suit,
            enhancement=enh,
            edition=ed,
            seal=seal,
            debuffed=p_debuffed,
        )

    # Fallback to generic Joker
    return Joker(name=JokerName.JOKER, edition=ed)


# =========================================================================
# PUBLIC CV DETECTION APIs
# =========================================================================


def _import_classifier_fn(name: str) -> Any:
    """Imports a recognition entrypoint from `classifiers.inference`, adding repo root if needed."""
    try:
        from classifiers import inference as cls_inf
    except ImportError:
        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parent.parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from classifiers import inference as cls_inf

    return getattr(cls_inf, name)


def _convert_items_of_type(
    items: Sequence[Any] | None,
    expected_types: type | tuple[type, ...] | None = None,
) -> list[Any]:
    """Converts ShopItemResult objects once and optionally filters to expected domain classes."""
    converted: list[Any] = []
    for item in items or []:
        obj = _convert_shop_item_result(item)
        if expected_types is None or isinstance(obj, expected_types):
            converted.append(obj)
    return converted


def cv_detect_hand_cards(
    state_or_frame: Any,
    frame_or_state: Any = None,
    conf_threshold: float = 0.50,
    device: str | None = None,
) -> list[HandCard]:
    """Evaluates the player's card hand ROI from the video frame.

    Args:
        state_or_frame: Current GameState (or video frame if called as cv_detect_hand_cards(frame, state)).
        frame_or_state: Video frame (numpy ndarray, PIL image, or VideoFrame) or GameState.
        conf_threshold: Detection confidence threshold.
        device: Torch device (e.g. 'cuda:0' or 'cpu').

    Returns:
        List of HandCard objects sorted left-to-right.

    Raises:
        InvalidLocationError: If the GameState indicates hand cards are not currently visible
                             (e.g. in shop outside pack, celestial/buffoon/standard packs, selecting blind, cash out).
        ValueError: If frame is None or invalid.
    """
    state, frame = _resolve_state_and_frame(state_or_frame, frame_or_state)
    validate_hand_cards_location(state)

    if frame is None:
        raise ValueError(
            "A valid video frame must be provided for cv_detect_hand_cards."
        )

    recognize_hand_cards = _import_classifier_fn("recognize_hand_cards")
    rec_output = recognize_hand_cards(
        frame, conf_threshold=conf_threshold, device=device
    )
    return [_convert_card_result_to_hand_card(c) for c in rec_output.cards]


def cv_detect_shop_cards(
    state_or_frame: Any,
    frame_or_state: Any = None,
    conf_threshold: float = 0.80,
    device: str | None = None,
) -> ShopCardsResult:
    """Evaluates the shop ROI for cards, vouchers, and booster packs from the video frame.

    Args:
        state_or_frame: Current GameState (or video frame if called as cv_detect_shop_cards(frame, state)).
        frame_or_state: Video frame (numpy ndarray, PIL image, or VideoFrame) or GameState.
        conf_threshold: Detection confidence threshold.
        device: Torch device.

    Returns:
        ShopCardsResult (subclass of list) containing the shop shelf cards, with attached
        .vouchers, .packs, .joker_bar, and .consumables attributes.

    Raises:
        InvalidLocationError: If GameState is not in the main shop (e.g. selecting a blind, in a blind,
                             inside a pack, or in the cash out screen).
        ValueError: If frame is None or invalid.
    """
    state, frame = _resolve_state_and_frame(state_or_frame, frame_or_state)
    validate_shop_cards_location(state)

    if frame is None:
        raise ValueError(
            "A valid video frame must be provided for cv_detect_shop_cards."
        )

    recognize_shop = _import_classifier_fn("recognize_shop")
    rec_output = recognize_shop(
        frame, mode="shop", conf_threshold=conf_threshold, device=device
    )

    shop_cards = _convert_items_of_type(getattr(rec_output, "shop_items", None))
    vouchers = _convert_items_of_type(getattr(rec_output, "vouchers", None), Voucher)
    packs = _convert_items_of_type(getattr(rec_output, "boosters", None), BoosterPack)
    jokers = _convert_items_of_type(getattr(rec_output, "joker_bar", None), Joker)
    consumables = _convert_items_of_type(
        getattr(rec_output, "consumeables", None), (Tarot, Planet, Spectral)
    )
    pack_items = _convert_items_of_type(getattr(rec_output, "pack_items", None))

    return ShopCardsResult(
        cards=shop_cards,
        vouchers=vouchers,
        packs=packs,
        joker_bar=jokers,
        consumables=consumables,
        pack_cards=pack_items,
        raw_output=rec_output,
    )


def cv_detect_pack_cards(
    state_or_frame: Any,
    frame_or_state: Any = None,
    conf_threshold: float = 0.80,
    device: str | None = None,
) -> list[Any]:
    """Evaluates the open booster pack ROI for pack offerings from the video frame.

    Args:
        state_or_frame: Current GameState (or video frame if called as cv_detect_pack_cards(frame, state)).
        frame_or_state: Video frame (numpy ndarray, PIL image, or VideoFrame) or GameState.
        conf_threshold: Detection confidence threshold.
        device: Torch device.

    Returns:
        List of converted pack item objects (Joker, Tarot, Planet, Spectral, or PlayingCard)
        sorted left-to-right.

    Raises:
        InvalidLocationError: If GameState is not inside an open booster pack.
        ValueError: If frame is None or invalid.
    """
    state, frame = _resolve_state_and_frame(state_or_frame, frame_or_state)
    validate_pack_cards_location(state)

    if frame is None:
        raise ValueError(
            "A valid video frame must be provided for cv_detect_pack_cards."
        )

    recognize_shop = _import_classifier_fn("recognize_shop")
    rec_output = recognize_shop(
        frame, mode="pack_only", conf_threshold=conf_threshold, device=device
    )
    return _convert_items_of_type(
        getattr(rec_output, "pack_items", None),
        (Joker, Tarot, Planet, Spectral, PlayingCard),
    )


def cv_detect_joker_bar(
    state_or_frame: Any,
    frame_or_state: Any = None,
    conf_threshold: float = 0.80,
    device: str | None = None,
) -> list[Joker]:
    """Evaluates the top Joker bar ROI from the video frame on any active gameplay screen.

    Args:
        state_or_frame: Current GameState (or video frame if called as cv_detect_joker_bar(frame, state)).
        frame_or_state: Video frame (numpy ndarray, PIL image, or VideoFrame) or GameState.
        conf_threshold: Detection confidence threshold.
        device: Torch device.

    Returns:
        List of Joker objects sorted left-to-right (with `id=0`; call `match_cv_joker_order`
        or `reconcile_cv_jokers` to preserve persistent Joker IDs).

    Raises:
        InvalidLocationError: If GameState does not correspond to a valid gameplay screen.
        ValueError: If frame is None or invalid.
    """
    state, frame = _resolve_state_and_frame(state_or_frame, frame_or_state)
    validate_joker_bar_location(state)

    if frame is None:
        raise ValueError(
            "A valid video frame must be provided for cv_detect_joker_bar."
        )

    recognize_shop = _import_classifier_fn("recognize_shop")
    rec_output = recognize_shop(
        frame, mode="joker_only", conf_threshold=conf_threshold, device=device
    )
    return _convert_items_of_type(getattr(rec_output, "joker_bar", None), Joker)


def match_cv_joker_order(state: Any, observed_jokers: Sequence[Joker]) -> list[int]:
    """Matches CV-observed Jokers (id=0) to tracked `state.inventory.jokers` persistent IDs.

    Visually identical copies (e.g. duplicate Blueprints or same-edition Showman duplicates)
    are matched arbitrarily in order so `Action(joker_order=...)` works without Lua IDs.
    """
    if state is None or not hasattr(state, "match_cv_joker_order"):
        return []
    return state.match_cv_joker_order(observed_jokers)


def reconcile_cv_jokers(
    state: Any,
    observed_jokers: Sequence[Joker],
    *,
    trigger_added_hooks: bool = False,
) -> list[int]:
    """Reconciles `state.inventory.jokers` against CV observations while preserving persistent IDs."""
    if state is None or not hasattr(state, "reconcile_cv_jokers"):
        return []
    return state.reconcile_cv_jokers(
        observed_jokers, trigger_added_hooks=trigger_added_hooks
    )


# =========================================================================
# PRNG BOSS BLIND RESOLVER (Replaces cv_detect_boss_blind)
# =========================================================================


def get_boss_blind_from_seed(
    seed: str,
    ante: int = 1,
    is_reroll: bool = False,
    bosses_used: dict[str, int] | None = None,
) -> BlindName:
    """Predicts or rolls the deterministic Boss Blind for the given seed and ante using Balatro PRNG.

    Replaces former `cv_detect_boss_blind()` queries.
    """
    try:
        from game.engine.rng_sim.events import get_next_boss_blind
        from game.engine.rng_sim.rng import BalatroRNG
    except ImportError:
        from engine.rng_sim.events import get_next_boss_blind
        from engine.rng_sim.rng import BalatroRNG

    clean_seed = str(seed).strip().upper()
    rng = BalatroRNG(clean_seed)
    used = bosses_used if bosses_used is not None else {}
    pred = get_next_boss_blind(
        rng,
        ante=ante,
        bosses_used=used,
        is_reroll=is_reroll,
    )
    return pred.blind_name
