"""Game State & Deck Tracking for Balatro RNG Simulation (`game.engine.rng_sim.state`).

Uses the canonical models and enums from `game.balatro_models`:
- `Suit`, `Rank`, `Enhancement`, `Edition`, `Seal`, `Stake`, `PokerHand`
- `TarotName`, `PlanetName`, `SpectralName`, `VoucherName`, `SkipTag`, `BlindName`, `JokerName`
- `Deck as DeckType` (the 16 Balatro deck backs)
- `PlayingCard`, `Joker`, `Tarot`, `Voucher`

Provides:
- `SimCard(PlayingCard)` (aliased as `Card`): a `PlayingCard` subclass (`isinstance(c, PlayingCard) == True`)
  that tracks Balatro's internal `sort_id` (`id`) and accepts both keyword (`PlayingCard(rank=..., suit=...)`)
  and positional (`Card(suit, rank, ...)`) construction.
- `SimDeck` (aliased as `Deck`): manages `G.playing_cards` (`list[PlayingCard]`) in exact `sort_id` order
  and supports direct synchronization with `game.gamestate.GameState.full_deck`.
- `SimGameState` (aliased as `GameState`): tracks PRNG-relevant run state and supports
  one-call synchronization from `game.gamestate.GameState` via `sync_from_gamestate()`.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

try:
    from game.balatro_models import (
        JOKER_REGISTRY,
        BlindName,
        BlindType,
        Card as BaseModelCard,
        CardType,
        Deck as DeckType,
        Edition,
        Enhancement,
        Joker,
        JokerName,
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
except ImportError:
    from balatro_models import (  # type: ignore[no-redef]
        JOKER_REGISTRY,
        BlindName,
        BlindType,
        Card as BaseModelCard,
        CardType,
        Deck as DeckType,
        Edition,
        Enhancement,
        Joker,
        JokerName,
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

if TYPE_CHECKING:
    from game.engine.rng_sim.rng import BalatroRNG


# ---------------------------------------------------------------------------
# Enum Conversion & Short-Code Helpers for `balatro_models` Enums
# ---------------------------------------------------------------------------

SUIT_SHORT_CODES: dict[Suit, str] = {
    Suit.SPADES: "S",
    Suit.HEARTS: "H",
    Suit.CLUBS: "C",
    Suit.DIAMONDS: "D",
    Suit.NONE: "N",
}

RANK_SHORT_CODES: dict[Rank, str] = {
    Rank.TWO: "2",
    Rank.THREE: "3",
    Rank.FOUR: "4",
    Rank.FIVE: "5",
    Rank.SIX: "6",
    Rank.SEVEN: "7",
    Rank.EIGHT: "8",
    Rank.NINE: "9",
    Rank.TEN: "T",
    Rank.JACK: "J",
    Rank.QUEEN: "Q",
    Rank.KING: "K",
    Rank.ACE: "A",
    Rank.NONE: "N",
}


def suit_short_code(suit: Suit) -> str:
    return SUIT_SHORT_CODES.get(suit, "N")


def rank_short_code(rank: Rank) -> str:
    return RANK_SHORT_CODES.get(rank, "N")


def parse_suit(s: Suit | str) -> Suit:
    if isinstance(s, Suit):
        return s
    val = str(s).strip().lower()
    if val in ("s", "spades", "spade"):
        return Suit.SPADES
    if val in ("h", "hearts", "heart"):
        return Suit.HEARTS
    if val in ("c", "clubs", "club"):
        return Suit.CLUBS
    if val in ("d", "diamonds", "diamond"):
        return Suit.DIAMONDS
    return Suit.NONE


def parse_rank(r: Rank | str) -> Rank:
    if isinstance(r, Rank):
        return r
    val = str(r).strip().upper()
    mapping = {
        "2": Rank.TWO,
        "3": Rank.THREE,
        "4": Rank.FOUR,
        "5": Rank.FIVE,
        "6": Rank.SIX,
        "7": Rank.SEVEN,
        "8": Rank.EIGHT,
        "9": Rank.NINE,
        "10": Rank.TEN,
        "T": Rank.TEN,
        "J": Rank.JACK,
        "JACK": Rank.JACK,
        "Q": Rank.QUEEN,
        "QUEEN": Rank.QUEEN,
        "K": Rank.KING,
        "KING": Rank.KING,
        "A": Rank.ACE,
        "ACE": Rank.ACE,
    }
    return mapping.get(val, Rank.NONE)


def parse_seal(s: Seal | str) -> Seal:
    if isinstance(s, Seal):
        return s
    val = str(s).strip().lower()
    if "red" in val:
        return Seal.RED
    if "blue" in val:
        return Seal.BLUE
    if "gold" in val:
        return Seal.GOLD
    if "purple" in val:
        return Seal.PURPLE
    return Seal.NONE


def seal_balatro_name(seal: Seal) -> str:
    match seal:
        case Seal.RED:
            return "Red"
        case Seal.BLUE:
            return "Blue"
        case Seal.GOLD:
            return "Gold"
        case Seal.PURPLE:
            return "Purple"
        case _:
            return "None"


def parse_joker_name(name: JokerName | str) -> JokerName | str:
    """Resolve a JokerName enum or string to `JokerName` if registered in `balatro_models`."""
    if isinstance(name, JokerName):
        return name
    clean = str(name).strip()
    lower = clean.lower()
    for jn in JokerName:
        if jn.value.lower() == lower or jn.name.lower() == lower:
            return jn
    if lower in ("ring master", "ring_master", "j_ring_master"):
        return JokerName.SHOWMAN
    if "oops" in lower:
        return JokerName.OOPS_ALL_6S
    if "mail" in lower and "rebate" in lower:
        return JokerName.MAIL_IN_REBATE
    if "driver" in lower:
        return JokerName.DRIVER
    if "caino" in lower or "canio" in lower:
        return JokerName.CANIO
    return clean


# Default poker hands order in G.GAME.hands (game.lua:2001-2014) using `balatro_models.PokerHand`
ALL_POKER_HANDS_ORDER: list[PokerHand] = [
    PokerHand.FLUSH_FIVE,       # order = 1, visible = False by default
    PokerHand.FLUSH_HOUSE,      # order = 2, visible = False by default
    PokerHand.FIVE_OF_A_KIND,   # order = 3, visible = False by default
    PokerHand.STRAIGHT_FLUSH,   # order = 4, visible = True
    PokerHand.FOUR_OF_A_KIND,   # order = 5, visible = True
    PokerHand.FULL_HOUSE,       # order = 6, visible = True
    PokerHand.FLUSH,            # order = 7, visible = True
    PokerHand.STRAIGHT,         # order = 8, visible = True
    PokerHand.THREE_OF_A_KIND,  # order = 9, visible = True
    PokerHand.TWO_PAIR,         # order = 10, visible = True
    PokerHand.PAIR,             # order = 11, visible = True
    PokerHand.HIGH_CARD,        # order = 12, visible = True
]

DEFAULT_VISIBLE_POKER_HANDS: list[PokerHand] = [
    PokerHand.STRAIGHT_FLUSH,
    PokerHand.FOUR_OF_A_KIND,
    PokerHand.FULL_HOUSE,
    PokerHand.FLUSH,
    PokerHand.STRAIGHT,
    PokerHand.THREE_OF_A_KIND,
    PokerHand.TWO_PAIR,
    PokerHand.PAIR,
    PokerHand.HIGH_CARD,
]

# LuaJIT 2.1 hash-table iteration order for `pairs(G.GAME.hands)` (`game.lua:2001-2014`)
# when initialized from those 12 literal keys. `card.lua:313` & `card.lua:2977` iterate
# `pairs(G.GAME.hands)` directly before indexing with `pseudorandom_element`.
LUAJIT_POKER_HANDS_ORDER: list[PokerHand] = [
    PokerHand.FLUSH_HOUSE,      # secret
    PokerHand.FULL_HOUSE,
    PokerHand.FLUSH,
    PokerHand.PAIR,
    PokerHand.HIGH_CARD,
    PokerHand.STRAIGHT_FLUSH,
    PokerHand.STRAIGHT,
    PokerHand.TWO_PAIR,
    PokerHand.FLUSH_FIVE,       # secret
    PokerHand.FIVE_OF_A_KIND,   # secret
    PokerHand.THREE_OF_A_KIND,
    PokerHand.FOUR_OF_A_KIND,
]


def order_poker_hands_luajit(hands: Sequence[PokerHand | str]) -> list[PokerHand]:
    """Orders a sequence of poker hands in LuaJIT 2.1 `pairs(G.GAME.hands)` iteration order."""
    parsed = [h if isinstance(h, PokerHand) else PokerHand(str(h)) for h in hands]
    hand_set = set(parsed)
    ordered = [h for h in LUAJIT_POKER_HANDS_ORDER if h in hand_set]
    known = set(ordered)
    for h in parsed:
        if h not in known:
            ordered.append(h)
            known.add(h)
    return ordered

# Lexicographical rank order used by Balatro's card_protos sort (`a.s .. a.r`):
# '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'J', 'K', 'Q', 'T'
LEXICOGRAPHICAL_RANKS: list[Rank] = [
    Rank.TWO,
    Rank.THREE,
    Rank.FOUR,
    Rank.FIVE,
    Rank.SIX,
    Rank.SEVEN,
    Rank.EIGHT,
    Rank.NINE,
    Rank.ACE,
    Rank.JACK,
    Rank.KING,
    Rank.QUEEN,
    Rank.TEN,
]

# Lexicographical suit order used by Balatro's card_protos sort (`a.s .. a.r`):
# 'C', 'D', 'H', 'S'
LEXICOGRAPHICAL_SUITS: list[Suit] = [
    Suit.CLUBS,
    Suit.DIAMONDS,
    Suit.HEARTS,
    Suit.SPADES,
]


# ---------------------------------------------------------------------------
# PlayingCard Subclass / Helpers (`SimCard` / `Card`)
# ---------------------------------------------------------------------------


def get_card_key(card: PlayingCard) -> str:
    """Return Balatro card front key like `'C_Q'` or `'S_A'` for any `PlayingCard`."""
    return f"{suit_short_code(card.suit)}_{rank_short_code(card.rank)}"


def card_is_suit(card: PlayingCard, suit: Suit | str) -> bool:
    """Check whether `card` matches `suit` (accounting for Wild and Stone cards)."""
    if card.enhancement == Enhancement.STONE:
        return False
    if card.enhancement == Enhancement.WILD:
        return True
    target = parse_suit(suit)
    return card.suit == target


def card_is_rank(card: PlayingCard, rank: Rank | str) -> bool:
    """Check whether `card` matches `rank` (Stone card masks rank)."""
    if card.enhancement == Enhancement.STONE:
        return False
    target = parse_rank(rank)
    return card.rank == target


class SimCard(PlayingCard):
    """A `game.balatro_models.PlayingCard` subclass with `sort_id` (`id`) and convenience methods.
    
    Every `SimCard` is a genuine `PlayingCard` (`isinstance(c, PlayingCard) == True`),
    so it can be passed directly into `game.gamestate.GameState`, `poker_hands.py`, and `jokers.py`,
    while also supporting positional `Card(suit, rank, ...)` or keyword `PlayingCard(rank=..., suit=...)`.
    """

    id: int = 0

    def __init__(
        self,
        suit: Suit | str = Suit.NONE,
        rank: Rank | str = Rank.NONE,
        enhancement: Enhancement | str = Enhancement.NONE,
        edition: Edition | str = Edition.NONE,
        seal: Seal | str = Seal.NONE,
        debuffed: bool = False,
        bonus_chips: int = 0,
        id: int = 0,
        destroyed: bool = False,
        extra_value: int = 0,
        type: CardType = CardType.PLAYING_CARD,
    ) -> None:
        resolved_suit = parse_suit(suit)
        resolved_rank = parse_rank(rank)
        resolved_enh = (
            Enhancement(enhancement)
            if isinstance(enhancement, str)
            else enhancement
        )
        resolved_ed = Edition(edition) if isinstance(edition, str) else edition
        resolved_seal = parse_seal(seal)
        super().__init__(
            rank=resolved_rank,
            suit=resolved_suit,
            enhancement=resolved_enh,
            edition=resolved_ed,
            seal=resolved_seal,
            type=type,
            bonus_chips=bonus_chips,
            debuffed=debuffed,
            destroyed=destroyed,
            extra_value=extra_value,
        )
        self.id = id

    @classmethod
    def from_playing_card(cls, card: PlayingCard, sort_id: int = 0) -> SimCard:
        """Wrap or clone an existing `balatro_models.PlayingCard` into a `SimCard`."""
        existing_id = getattr(card, "id", 0) or getattr(card, "sort_id", 0) or sort_id
        return cls(
            suit=card.suit,
            rank=card.rank,
            enhancement=card.enhancement,
            edition=card.edition,
            seal=card.seal,
            debuffed=card.debuffed,
            bonus_chips=card.bonus_chips,
            id=existing_id,
            destroyed=card.destroyed,
            extra_value=card.extra_value,
        )

    @property
    def sort_id(self) -> int:
        return self.id

    @sort_id.setter
    def sort_id(self, value: int) -> None:
        self.id = value

    @property
    def card_key(self) -> str:
        """Balatro key representation like `'C_Q'` or `'S_A'`."""
        return get_card_key(self)

    def is_suit(self, suit: Suit | str) -> bool:
        """Check card suit, taking Wild Card and Stone Card enhancements into account."""
        return card_is_suit(self, suit)

    def is_rank(self, rank: Rank | str) -> bool:
        """Check card rank (Stone card masks rank)."""
        return card_is_rank(self, rank)

    def clone(self) -> SimCard:
        return copy.copy(self)


# Alias `Card` to `SimCard` so both `PlayingCard` and `Card` work seamlessly
Card = SimCard


# ---------------------------------------------------------------------------
# Joker Wrapper / Helper (`Joker` from `balatro_models.py`)
# ---------------------------------------------------------------------------


def make_joker(
    name: JokerName | str,
    position: int = 0,
    edition: Edition | str = Edition.NONE,
    eternal: bool = False,
    perishable: bool = False,
    rental: bool = False,
    state: dict[str, Any] | None = None,
) -> Joker:
    """Create a `balatro_models.Joker` instance with an attached `.position` attribute."""
    resolved_name = parse_joker_name(name)
    ed = Edition(edition) if isinstance(edition, str) else edition
    joker = Joker(
        name=resolved_name,  # type: ignore[arg-type]
        edition=ed,
        eternal=eternal,
        perishable=perishable,
        rental=rental,
        state=copy.deepcopy(state) if state is not None else {},
    )
    joker.position = position  # type: ignore[attr-defined]
    return joker


# Alias `JokerInstance` to `balatro_models.Joker`
JokerInstance = Joker


# ---------------------------------------------------------------------------
# Deck Tracking (`SimDeck`, aliased as `Deck`)
# ---------------------------------------------------------------------------


class SimDeck:
    """Active deck representation (`G.playing_cards`) storing `PlayingCard` objects in `sort_id` order."""

    def __init__(self, cards: Sequence[PlayingCard] | None = None) -> None:
        self.cards: list[SimCard] = []
        self._next_id: int = 1
        if cards is not None:
            self._init_cards(cards)

    def _init_cards(self, cards: Sequence[PlayingCard]) -> None:
        """Initialize deck from a sequence of `PlayingCard`s.
        
        If the input cards are a standard un-ID'd deck (e.g., from `gamestate.create_standard_deck()`),
        assigns `sort_id` in Balatro's exact `a.s .. a.r` lexicographical order (`C_2..C_T, D_2..D_T, H_2..H_T, S_2..S_T`)
        when there are 40 or 52 un-ID'd base cards; otherwise preserves their sequence order.
        """
        all_unassigned = all(
            (getattr(c, "id", 0) == 0 and getattr(c, "sort_id", 0) == 0)
            for c in cards
        )
        if all_unassigned and len(cards) in (40, 52):
            # Sort by Balatro's `a.s .. a.r .. a.e .. a.d .. a.g` lexicographical key (game.lua:2367-2369)
            sorted_cards = sorted(
                cards,
                key=lambda c: (
                    suit_short_code(c.suit),
                    rank_short_code(c.rank),
                    c.enhancement.value,
                    c.edition.value,
                    c.seal.value,
                ),
            )
            for c in sorted_cards:
                sc = (
                    c
                    if isinstance(c, SimCard)
                    else SimCard.from_playing_card(c)
                )
                sc.id = self._next_id
                self._next_id += 1
                self.cards.append(sc)
        else:
            for c in cards:
                self.add_card(c)

    @classmethod
    def create_standard_52(cls) -> SimDeck:
        """Create a standard 52-card deck of `PlayingCard`s in Balatro's exact `sort_id` order (`1..52`)."""
        cards: list[SimCard] = []
        card_id = 1
        for s in LEXICOGRAPHICAL_SUITS:
            for r in LEXICOGRAPHICAL_RANKS:
                cards.append(SimCard(suit=s, rank=r, id=card_id))
                card_id += 1
        return cls(cards)

    @classmethod
    def create_abandoned_40(cls) -> SimDeck:
        """Create an Abandoned Deck (40 `PlayingCard`s, no Jack/Queen/King) in exact `sort_id` order."""
        cards: list[SimCard] = []
        card_id = 1
        for s in LEXICOGRAPHICAL_SUITS:
            for r in LEXICOGRAPHICAL_RANKS:
                if r not in (Rank.JACK, Rank.QUEEN, Rank.KING):
                    cards.append(SimCard(suit=s, rank=r, id=card_id))
                    card_id += 1
        return cls(cards)

    @classmethod
    def create_checkered_52(cls) -> SimDeck:
        """Create a Checkered Deck (26 Spades, 26 Hearts) in exact `sort_id` order (`back.lua:239-253`)."""
        deck = cls.create_standard_52()
        for c in deck.cards:
            if c.suit == Suit.CLUBS:
                c.suit = Suit.SPADES
            elif c.suit == Suit.DIAMONDS:
                c.suit = Suit.HEARTS
        return deck

    @classmethod
    def create_erratic_52(cls, rng: BalatroRNG) -> SimDeck:
        """Create an Erratic Deck (52 randomized `PlayingCard`s using stream `'erratic'`) in exact `sort_id` order."""
        from game.engine.rng_sim.events import P_CARDS_KEYS, parse_card_key

        protos: list[str] = []
        for _ in range(52):
            k, _ = rng.choice("erratic", P_CARDS_KEYS)
            protos.append(k)
        protos.sort(key=lambda key: key.replace("_", ""))
        cards: list[SimCard] = []
        for idx, k in enumerate(protos, start=1):
            suit, rank = parse_card_key(k)
            cards.append(SimCard(suit=suit, rank=rank, id=idx))
        return cls(cards)

    def add_card(self, card: PlayingCard) -> SimCard:
        """Add a `PlayingCard` to the deck, assigning the next `sort_id` if unset."""
        sc = card if isinstance(card, SimCard) else SimCard.from_playing_card(card)
        if sc.id <= 0:
            sc.id = self._next_id
            self._next_id += 1
        else:
            if sc.id >= self._next_id:
                self._next_id = sc.id + 1
        # Also mirror `id` onto the original PlayingCard if it was wrapped
        if card is not sc:
            card.id = sc.id  # type: ignore[attr-defined]
        self.cards.append(sc)
        return sc

    def remove_card(
        self,
        card_or_id: PlayingCard | int | None = None,
        *,
        suit: Suit | str | None = None,
        rank: Rank | str | None = None,
    ) -> SimCard | None:
        """Remove a card from the deck by `PlayingCard` instance, `sort_id` (`int`), or `(suit, rank)`."""
        if isinstance(card_or_id, int):
            for i, c in enumerate(self.cards):
                if c.id == card_or_id:
                    return self.cards.pop(i)
            return None

        if isinstance(card_or_id, PlayingCard):
            target_id = getattr(card_or_id, "id", 0) or getattr(card_or_id, "sort_id", 0)
            if target_id > 0:
                for i, c in enumerate(self.cards):
                    if c.id == target_id:
                        return self.cards.pop(i)
            # Fallback match by identity or card attributes
            for i, c in enumerate(self.cards):
                if c is card_or_id or (
                    c.rank == card_or_id.rank
                    and c.suit == card_or_id.suit
                    and c.enhancement == card_or_id.enhancement
                    and c.edition == card_or_id.edition
                    and c.seal == card_or_id.seal
                ):
                    return self.cards.pop(i)
            return None

        if suit is not None or rank is not None:
            target_s = parse_suit(suit) if suit is not None else None
            target_r = parse_rank(rank) if rank is not None else None
            for i, c in enumerate(self.cards):
                if (target_s is None or c.suit == target_s) and (
                    target_r is None or c.rank == target_r
                ):
                    return self.cards.pop(i)
        return None

    def destroy_card(
        self,
        card_or_id: PlayingCard | int | None = None,
        *,
        suit: Suit | str | None = None,
        rank: Rank | str | None = None,
    ) -> SimCard | None:
        """Destroy a card (alias for `remove_card`)."""
        return self.remove_card(card_or_id, suit=suit, rank=rank)

    def convert_card(
        self,
        card_or_id: PlayingCard | int,
        new_suit: Suit | str | None = None,
        new_rank: Rank | str | None = None,
        new_enhancement: Enhancement | str | None = None,
        new_edition: Edition | str | None = None,
        new_seal: Seal | str | None = None,
    ) -> SimCard | None:
        """Mutate an existing card's attributes in place (preserving its `sort_id`)."""
        target: SimCard | None = None
        if isinstance(card_or_id, int):
            target = next((c for c in self.cards if c.id == card_or_id), None)
        else:
            tid = getattr(card_or_id, "id", 0) or getattr(card_or_id, "sort_id", 0)
            if tid > 0:
                target = next((c for c in self.cards if c.id == tid), None)
            if target is None:
                target = next(
                    (
                        c
                        for c in self.cards
                        if c is card_or_id
                        or (
                            c.rank == card_or_id.rank
                            and c.suit == card_or_id.suit
                            and c.enhancement == card_or_id.enhancement
                        )
                    ),
                    None,
                )
        if target is None:
            return None
        if new_suit is not None:
            target.suit = parse_suit(new_suit)
        if new_rank is not None:
            target.rank = parse_rank(new_rank)
        if new_enhancement is not None:
            target.enhancement = (
                Enhancement(new_enhancement)
                if isinstance(new_enhancement, str)
                else new_enhancement
            )
        if new_edition is not None:
            target.edition = (
                Edition(new_edition) if isinstance(new_edition, str) else new_edition
            )
        if new_seal is not None:
            target.seal = parse_seal(new_seal)
        return target

    def sync_from_playing_cards(
        self,
        new_cards: Sequence[PlayingCard],
        preserve_existing_ids: bool = True,
    ) -> None:
        """Synchronize this deck with an external `list[PlayingCard]` (such as `gamestate.full_deck`)
        while preserving existing `sort_id`s for remaining cards and assigning new `sort_id`s to added cards.
        """
        if not preserve_existing_ids:
            rebuilt = SimDeck.from_playing_cards(new_cards)
            self.cards = rebuilt.cards
            self._next_id = rebuilt._next_id
            return

        # Build lookup pool of existing cards by their `id` or `(suit, rank, enhancement, edition, seal)`
        by_id: dict[int, SimCard] = {c.id: c for c in self.cards if c.id > 0}
        unmatched_existing: list[SimCard] = list(self.cards)

        updated_deck: list[SimCard] = []
        newly_added: list[PlayingCard] = []

        for nc in new_cards:
            if getattr(nc, "destroyed", False):
                continue
            nc_id = getattr(nc, "id", 0) or getattr(nc, "sort_id", 0)
            if nc_id > 0 and nc_id in by_id:
                matched = by_id.pop(nc_id)
                if matched in unmatched_existing:
                    unmatched_existing.remove(matched)
                matched.suit = nc.suit
                matched.rank = nc.rank
                matched.enhancement = nc.enhancement
                matched.edition = nc.edition
                matched.seal = nc.seal
                matched.bonus_chips = nc.bonus_chips
                matched.debuffed = nc.debuffed
                nc.id = matched.id  # type: ignore[attr-defined]
                updated_deck.append(matched)
            else:
                # Match by exact attributes against remaining existing cards (preserving earliest sort_id)
                found_idx: int | None = None
                for idx, ex in enumerate(unmatched_existing):
                    if (
                        ex.suit == nc.suit
                        and ex.rank == nc.rank
                        and ex.enhancement == nc.enhancement
                        and ex.edition == nc.edition
                        and ex.seal == nc.seal
                    ):
                        found_idx = idx
                        break
                if found_idx is not None:
                    matched = unmatched_existing.pop(found_idx)
                    by_id.pop(matched.id, None)
                    nc.id = matched.id  # type: ignore[attr-defined]
                    updated_deck.append(matched)
                else:
                    newly_added.append(nc)

        self.cards = updated_deck
        for nc in newly_added:
            self.add_card(nc)
        self.cards.sort(key=lambda c: c.id)

    def get_non_stone_cards(self) -> list[SimCard]:
        """Return all cards where `v.ability.effect ~= 'Stone Card'`, sorted by `sort_id`."""
        valid = [
            c
            for c in self.cards
            if not c.destroyed
            and c.enhancement != Enhancement.STONE
            and c.rank != Rank.NONE
            and c.suit != Suit.NONE
        ]
        valid.sort(key=lambda c: c.id)
        return valid

    def get_valid_rank_cards(self) -> list[SimCard]:
        """Alias for `get_non_stone_cards()`."""
        return self.get_non_stone_cards()

    def get_rank_histogram(self) -> dict[Rank, int]:
        """Return counts of each rank in current deck (excluding Stone cards)."""
        hist: Counter[Rank] = Counter()
        for c in self.get_non_stone_cards():
            hist[c.rank] += 1
        return dict(hist)

    def get_suit_histogram(self) -> dict[Suit, int]:
        """Return counts of each base suit in current deck (excluding Stone cards)."""
        hist: Counter[Suit] = Counter()
        for c in self.get_non_stone_cards():
            hist[c.suit] += 1
        return dict(hist)

    def clone(self) -> SimDeck:
        cloned_cards = [c.clone() for c in self.cards]
        deck = SimDeck(cloned_cards)
        deck._next_id = self._next_id
        return deck

    def __len__(self) -> int:
        return len(self.cards)

    def __iter__(self) -> Iterator[SimCard]:
        return iter(self.cards)

    def __getitem__(self, index: int) -> SimCard:
        return self.cards[index]

    def __contains__(self, item: object) -> bool:
        return item in self.cards


# Alias `Deck` to `SimDeck` (`DeckType` is `balatro_models.Deck`)
Deck = SimDeck


# ---------------------------------------------------------------------------
# GameState (`SimGameState`, aliased as `GameState`)
# ---------------------------------------------------------------------------


@dataclass
class SimGameState:
    """Active simulation state tracking PRNG-relevant run parameters and `balatro_models` objects."""

    seed: str
    ante: int = 1
    round_within_ante: int = 1  # 1=Small Blind, 2=Big Blind, 3=Boss Blind
    total_rounds: int = 0
    deck: SimDeck = field(default_factory=SimDeck.create_standard_52)
    deck_type: DeckType | str = DeckType.RED_DECK
    owned_jokers: list[Joker] = field(default_factory=list)
    stake: Stake = Stake.WHITE
    hand: list[SimCard] = field(default_factory=list)
    discards_made_this_round: int = 0
    hands_played_this_round: int = 0
    money: int = 4
    round_active: bool = True
    win_ante: int = 8

    # Dynamic round targets (G.GAME.current_round)
    ancient_suit: Suit | None = None
    castle_suit: Suit = Suit.SPADES
    idol_card: tuple[Rank, Suit] = (Rank.ACE, Suit.SPADES)
    mail_rank: Rank = Rank.ACE
    to_do_hand: PokerHand | str | None = None

    # Global pool / progression tracking (G.GAME)
    used_vouchers: set[str] = field(default_factory=set)
    shop_vouchers: list[str] = field(default_factory=list)
    bosses_used: dict[str, int] = field(default_factory=dict)
    current_boss: str | None = None
    banned_keys: set[str] = field(default_factory=set)
    active_tarots: set[str] = field(default_factory=set)
    visible_hands: list[PokerHand] = field(
        default_factory=lambda: list(DEFAULT_VISIBLE_POKER_HANDS)
    )
    pool_flags: dict[str, bool] = field(default_factory=dict)

    start_of_round_hooks: list[Callable[[SimGameState], None]] = field(
        default_factory=list
    )

    @property
    def round(self) -> int:
        return self.round_within_ante

    @round.setter
    def round(self, value: int) -> None:
        self.round_within_ante = value

    @property
    def total_rounds_played(self) -> int:
        return self.total_rounds

    @total_rounds_played.setter
    def total_rounds_played(self, value: int) -> None:
        self.total_rounds = value

    @property
    def oops_count(self) -> int:
        """Count how many active `Oops! All 6s` jokers are owned."""
        return sum(
            1
            for j in self.owned_jokers
            if j.name == JokerName.OOPS_ALL_6S or "oops" in str(j.name).lower()
        )

    @property
    def has_showman(self) -> bool:
        """Check whether `Showman` is currently owned."""
        return any(
            j.name == JokerName.SHOWMAN
            or "showman" in str(j.name).lower()
            or "ring_master" in str(j.name).lower()
            for j in self.owned_jokers
        )

    @property
    def probability_modifier(self) -> float:
        """Probability multiplier from `Oops! All 6s` (doubles per joker)."""
        return float(2 ** self.oops_count)

    def unlock_secret_hand(self, hand_name: PokerHand | str) -> None:
        """Make a secret poker hand (`Flush Five`, `Flush House`, `Five of a Kind`) visible."""
        resolved = (
            hand_name
            if isinstance(hand_name, PokerHand)
            else PokerHand(str(hand_name).strip())
        )
        if resolved not in self.visible_hands:
            new_visible_set = set(self.visible_hands) | {resolved}
            self.visible_hands = [
                h for h in ALL_POKER_HANDS_ORDER if h in new_visible_set
            ]

    def add_joker(
        self,
        name_or_joker: Joker | JokerName | str,
        edition: Edition | str = Edition.NONE,
        eternal: bool = False,
        perishable: bool = False,
        rental: bool = False,
        state: dict[str, Any] | None = None,
    ) -> Joker:
        """Add a `balatro_models.Joker` to the owned jokers list."""
        pos = len(self.owned_jokers)
        if isinstance(name_or_joker, Joker):
            joker = copy.deepcopy(name_or_joker)
            joker.position = pos  # type: ignore[attr-defined]
        else:
            joker = make_joker(
                name=name_or_joker,
                position=pos,
                edition=edition,
                eternal=eternal,
                perishable=perishable,
                rental=rental,
                state=state,
            )
        self.owned_jokers.append(joker)
        return joker

    def remove_joker(self, target: int | JokerName | str | Joker) -> Joker | None:
        """Remove a joker by index, `Joker` instance, `JokerName`, or string name."""
        if isinstance(target, int):
            if 0 <= target < len(self.owned_jokers):
                removed = self.owned_jokers.pop(target)
                for idx, j in enumerate(self.owned_jokers):
                    j.position = idx  # type: ignore[attr-defined]
                return removed
            return None
        if isinstance(target, Joker):
            for idx, j in enumerate(self.owned_jokers):
                if j is target or j.name == target.name:
                    removed = self.owned_jokers.pop(idx)
                    for i, remaining in enumerate(self.owned_jokers):
                        remaining.position = i  # type: ignore[attr-defined]
                    return removed
            return None
        target_name = str(parse_joker_name(target)).lower()
        for idx, j in enumerate(self.owned_jokers):
            if str(j.name).lower() == target_name:
                removed = self.owned_jokers.pop(idx)
                for i, remaining in enumerate(self.owned_jokers):
                    remaining.position = i  # type: ignore[attr-defined]
                return removed
        return None

    def find_joker(self, name: JokerName | str) -> list[Joker]:
        """Find active jokers matching `name`."""
        target = str(parse_joker_name(name)).lower()
        return [
            j
            for j in self.owned_jokers
            if target in str(j.name).lower()
        ]

    def has_joker(self, name: JokerName | str) -> bool:
        """Check whether any active joker matches `name`."""
        return len(self.find_joker(name)) > 0

    def sync_from_gamestate(self, gs: Any) -> None:
        """Synchronize directly from a `game.gamestate.GameState` instance.
        
        Updates:
        - `ante`, `round_within_ante`, `stake`, `deck_type`, `money`
        - `deck` (preserving `sort_id`s of existing `PlayingCard`s in `gs.full_deck`)
        - `owned_jokers` (from `gs.inventory.jokers`)
        - `active_tarots` (from `gs.inventory.consumables`, `gs.shop_cards`, and `gs.pack_cards`)
        - `used_vouchers` (from `gs.vouchers_redeemed`)
        - `shop_vouchers` (from `gs.shop_vouchers`)
        - `visible_hands` (unlocking any secret hand with `gs.hand_play_counts[h] > 0`)
        """
        from game.engine.rng_sim.events import normalize_tarot_key, normalize_voucher_key

        self.ante = getattr(gs, "ante", self.ante)
        self.round_within_ante = getattr(gs, "round", self.round_within_ante)
        self.stake = getattr(gs, "stake", self.stake)
        self.deck_type = getattr(gs, "deck", self.deck_type)
        self.money = getattr(gs, "money", self.money)
        self.hands_played_this_round = getattr(
            gs, "hands_played_this_round", self.hands_played_this_round
        )
        self.discards_made_this_round = getattr(
            gs, "discards_used_this_round", self.discards_made_this_round
        )

        full_deck = getattr(gs, "full_deck", None)
        if full_deck is not None:
            self.deck.sync_from_playing_cards(full_deck)

        inv = getattr(gs, "inventory", None)
        if inv is not None:
            self.owned_jokers = []
            for idx, j in enumerate(getattr(inv, "jokers", [])):
                cloned_j = copy.deepcopy(j)
                cloned_j.position = idx  # type: ignore[attr-defined]
                self.owned_jokers.append(cloned_j)

            self.active_tarots.clear()
            for c in getattr(inv, "consumables", []):
                if getattr(c, "type", None) == CardType.TAROT or isinstance(c, Tarot):
                    self.active_tarots.add(normalize_tarot_key(str(c.name)))

        vouchers_redeemed = getattr(gs, "vouchers_redeemed", None)
        if vouchers_redeemed is not None:
            for v in vouchers_redeemed:
                self.used_vouchers.add(normalize_voucher_key(str(v)))

        shop_vouchers = getattr(gs, "shop_vouchers", None)
        if shop_vouchers is not None:
            self.shop_vouchers = [
                normalize_voucher_key(str(v.name))
                for v in shop_vouchers
            ]

        for c in getattr(gs, "shop_cards", []):
            if getattr(c, "type", None) == CardType.TAROT or isinstance(c, Tarot):
                self.active_tarots.add(normalize_tarot_key(str(c.name)))

        for c in getattr(gs, "pack_cards", []):
            if getattr(c, "type", None) == CardType.TAROT or isinstance(c, Tarot):
                self.active_tarots.add(normalize_tarot_key(str(c.name)))

        hand_play_counts = getattr(gs, "hand_play_counts", {}) or {}
        for secret in (
            PokerHand.FLUSH_FIVE,
            PokerHand.FLUSH_HOUSE,
            PokerHand.FIVE_OF_A_KIND,
        ):
            if hand_play_counts.get(secret, 0) > 0:
                self.unlock_secret_hand(secret)

    # -----------------------------------------------------------------------
    # Lifecycle Transitions
    # -----------------------------------------------------------------------

    def start_ante(self, ante_num: int) -> None:
        self.ante = ante_num
        self.round_within_ante = 0

    def start_round(self, round_num: int | None = None) -> None:
        if round_num is not None:
            self.round_within_ante = round_num
        else:
            self.round_within_ante += 1
        self.total_rounds += 1
        self.discards_made_this_round = 0
        self.hands_played_this_round = 0
        self.round_active = True
        self.trigger_start_of_round_effects()

    def trigger_start_of_round_effects(self) -> None:
        for hook in list(self.start_of_round_hooks):
            hook(self)

    def end_round(self) -> None:
        self.round_active = False

    # -----------------------------------------------------------------------
    # Checkpointing & Forking
    # -----------------------------------------------------------------------

    def checkpoint(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "ante": self.ante,
            "round_within_ante": self.round_within_ante,
            "total_rounds": self.total_rounds,
            "deck_type": str(self.deck_type),
            "stake": self.stake.value,
            "money": self.money,
            "round_active": self.round_active,
            "win_ante": self.win_ante,
            "discards_made_this_round": self.discards_made_this_round,
            "hands_played_this_round": self.hands_played_this_round,
            "ancient_suit": self.ancient_suit.value if self.ancient_suit else None,
            "castle_suit": self.castle_suit.value,
            "idol_card": [self.idol_card[0].value, self.idol_card[1].value],
            "mail_rank": self.mail_rank.value,
            "to_do_hand": str(self.to_do_hand) if self.to_do_hand else None,
            "used_vouchers": sorted(self.used_vouchers),
            "shop_vouchers": list(self.shop_vouchers),
            "bosses_used": dict(self.bosses_used),
            "current_boss": self.current_boss,
            "banned_keys": sorted(self.banned_keys),
            "active_tarots": sorted(self.active_tarots),
            "visible_hands": [h.value for h in self.visible_hands],
            "pool_flags": dict(self.pool_flags),
            "deck": [
                {
                    "suit": c.suit.value,
                    "rank": c.rank.value,
                    "enhancement": c.enhancement.value,
                    "edition": c.edition.value,
                    "seal": c.seal.value,
                    "debuffed": c.debuffed,
                    "bonus_chips": c.bonus_chips,
                    "id": c.id,
                    "destroyed": c.destroyed,
                    "extra_value": c.extra_value,
                }
                for c in self.deck.cards
            ],
            "owned_jokers": [
                {
                    "name": str(j.name),
                    "position": getattr(j, "position", idx),
                    "edition": j.edition.value,
                    "eternal": j.eternal,
                    "perishable": j.perishable,
                    "rental": j.rental,
                    "extra_value": j.extra_value,
                    "state": copy.deepcopy(j.state),
                }
                for idx, j in enumerate(self.owned_jokers)
            ],
        }

    def restore(self, snap: dict[str, Any]) -> None:
        self.seed = snap["seed"]
        self.ante = snap["ante"]
        self.round_within_ante = snap["round_within_ante"]
        self.total_rounds = snap["total_rounds"]
        dt_str = snap.get("deck_type", DeckType.RED_DECK.value)
        try:
            self.deck_type = DeckType(dt_str)
        except ValueError:
            self.deck_type = dt_str
        self.stake = Stake(snap["stake"])
        self.money = snap["money"]
        self.round_active = snap["round_active"]
        self.win_ante = snap.get("win_ante", 8)
        self.discards_made_this_round = snap.get("discards_made_this_round", 0)
        self.hands_played_this_round = snap.get("hands_played_this_round", 0)

        anc = snap.get("ancient_suit")
        self.ancient_suit = parse_suit(anc) if anc else None
        self.castle_suit = parse_suit(snap.get("castle_suit", Suit.SPADES.value))
        idol_raw = snap.get("idol_card", [Rank.ACE.value, Suit.SPADES.value])
        self.idol_card = (parse_rank(idol_raw[0]), parse_suit(idol_raw[1]))
        self.mail_rank = parse_rank(snap.get("mail_rank", Rank.ACE.value))
        td = snap.get("to_do_hand")
        self.to_do_hand = PokerHand(td) if td else None

        self.used_vouchers = set(snap.get("used_vouchers", []))
        self.shop_vouchers = list(snap.get("shop_vouchers", []))
        self.bosses_used = dict(snap.get("bosses_used", {}))
        self.current_boss = snap.get("current_boss")
        self.banned_keys = set(snap.get("banned_keys", []))
        self.active_tarots = set(snap.get("active_tarots", []))
        self.visible_hands = [
            PokerHand(h) for h in snap.get("visible_hands", DEFAULT_VISIBLE_POKER_HANDS)
        ]
        self.pool_flags = dict(snap.get("pool_flags", {}))

        restored_cards = [
            SimCard(
                suit=Suit(c["suit"]),
                rank=Rank(c["rank"]),
                enhancement=Enhancement(c["enhancement"]),
                edition=Edition(c["edition"]),
                seal=Seal(c["seal"]),
                debuffed=c.get("debuffed", False),
                bonus_chips=c.get("bonus_chips", 0),
                id=c.get("id", 0),
                destroyed=c.get("destroyed", False),
                extra_value=c.get("extra_value", 0),
            )
            for c in snap["deck"]
        ]
        self.deck = SimDeck(restored_cards)

        self.owned_jokers = []
        for j in snap["owned_jokers"]:
            joker = make_joker(
                name=j["name"],
                position=j.get("position", len(self.owned_jokers)),
                edition=Edition(j["edition"]),
                eternal=j.get("eternal", False),
                perishable=j.get("perishable", False),
                rental=j.get("rental", False),
                state=copy.deepcopy(j.get("state", {})),
            )
            joker.extra_value = j.get("extra_value", 0)
            self.owned_jokers.append(joker)

    def clone(self) -> SimGameState:
        cloned = SimGameState(seed=self.seed)
        cloned.restore(self.checkpoint())
        cloned.hand = [c.clone() for c in self.hand]
        cloned.start_of_round_hooks = list(self.start_of_round_hooks)
        return cloned


GameState = SimGameState
