import copy
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

try:
    from game.balatro_models import (
        Action,
        ActionType,
        BlindName,
        BoosterPack,
        Card,
        CardBounds,
        Deck,
        Edition,
        Enhancement,
        Hand,
        HandCard,
        Inventory,
        Joker,
        JokerName,
        Location,
        PlanetName,
        PlayingCard,
        PokerHand,
        PokerHandLevel,
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
    from game.engine.rules import on_action
except ImportError:
    from balatro_models import (
        Action,
        ActionType,
        BlindName,
        BoosterPack,
        Card,
        CardBounds,
        Deck,
        Edition,
        Enhancement,
        Hand,
        HandCard,
        Inventory,
        Joker,
        JokerName,
        Location,
        PlanetName,
        PlayingCard,
        PokerHand,
        PokerHandLevel,
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
    from engine import on_action


def create_standard_deck() -> list[PlayingCard]:
    """Generates a standard 52-card deck of PlayingCard objects in Balatro's exact sort_id order (1..52)."""
    suits = [Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS, Suit.SPADES]
    ranks = [
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
    deck = []
    card_id = 1
    for s in suits:
        for r in ranks:
            deck.append(PlayingCard(rank=r, suit=s, id=card_id))
            card_id += 1
    return deck


# Base blind score thresholds for Antes 1..8 by Stake scaling tier (misc_functions.lua:919-954)
ANTE_CHIPS_TABLE: dict[int, tuple[int, ...]] = {
    1: (300, 800, 2000, 5000, 11000, 20000, 35000, 50000),  # White, Red Stakes
    2: (300, 900, 2600, 8000, 20000, 36000, 60000, 100000),  # Green, Black, Blue Stakes
    3: (
        300,
        1000,
        3200,
        9000,
        25000,
        60000,
        110000,
        200000,
    ),  # Purple, Orange, Gold Stakes
}

STAKE_SCALING_TIER: dict[Stake, int] = {
    Stake.WHITE: 1,
    Stake.RED: 1,
    Stake.GREEN: 2,
    Stake.BLACK: 2,
    Stake.BLUE: 2,
    Stake.PURPLE: 3,
    Stake.ORANGE: 3,
    Stake.GOLD: 3,
}

STAKE_ORDER: tuple[Stake, ...] = (
    Stake.WHITE,
    Stake.RED,
    Stake.GREEN,
    Stake.BLACK,
    Stake.BLUE,
    Stake.PURPLE,
    Stake.ORANGE,
    Stake.GOLD,
)

BLIND_SCORE_MULTIPLIERS: dict[BlindName, float] = {
    BlindName.SMALL: 1.0,
    BlindName.BIG: 1.5,
    BlindName.THE_WALL: 4.0,
    BlindName.VIOLET_VESSEL: 6.0,
    BlindName.THE_NEEDLE: 1.0,
}


def get_stake_level(stake: Stake) -> int:
    """Returns 1..8 stake index matching Balatro's G.GAME.stake."""
    try:
        return STAKE_ORDER.index(stake) + 1
    except ValueError:
        return 1


def get_blind_amount(ante: int, stake: Stake = Stake.WHITE) -> int:
    """Returns base chip requirement for `ante` and `stake` (misc_functions.lua:919-954)."""
    if ante < 1:
        return 100
    scaling = STAKE_SCALING_TIER.get(stake, 1)
    amounts = ANTE_CHIPS_TABLE[scaling]
    if ante <= 8:
        return amounts[ante - 1]
    k = 0.75
    a = amounts[-1]
    b = 1.6
    c = ante - 8
    d = 1.0 + 0.2 * c
    amount = math.floor(a * (b + (k * c) ** d) ** c)
    scale = 10 ** math.floor(math.log10(amount) - 1)
    return int(amount - (amount % scale))


def calculate_target_chips(
    ante: int = 1,
    round: int = 1,
    stake: Stake = Stake.WHITE,
    blind: BlindName | None = None,
    deck: Deck = Deck.RED_DECK,
) -> int:
    """Computes blind target_chips from ante, round (or explicit blind), stake, and deck."""
    base = get_blind_amount(ante, stake)
    if blind is not None:
        mult = BLIND_SCORE_MULTIPLIERS.get(blind, 2.0)
    else:
        round_slot = (max(1, round) - 1) % 3
        mult = (1.0, 1.5, 2.0)[round_slot]
    deck_scaling = 2 if deck == Deck.PLASMA_DECK else 1
    return int(base * mult * deck_scaling)


def _cards_match(a: PlayingCard, b: PlayingCard) -> bool:
    """Checks whether two PlayingCard instances represent the same card instance or identical card."""
    if a is b:
        return True
    if a.id and b.id and a.id == b.id:
        return True
    if (
        a.enhancement == Enhancement.STONE
        and b.enhancement == Enhancement.STONE
        and (
            a.rank == Rank.NONE
            or a.suit == Suit.NONE
            or b.rank == Rank.NONE
            or b.suit == Suit.NONE
        )
    ):
        return a.edition == b.edition and a.seal == b.seal
    return (
        a.rank == b.rank
        and a.suit == b.suit
        and a.enhancement == b.enhancement
        and a.edition == b.edition
        and a.seal == b.seal
    )


def get_rank_sort_nominal(card: PlayingCard) -> float:
    """Calculates Balatro's exact nominal value for 'Sort by Rank' (descending).

    Nominals:
      Ace: 11.4, King: 10.3, Queen: 10.2, Jack: 10.1, 10: 10.0, 9..2: 9.0..2.0
      Suit nominals: Spades: +0.04, Hearts: +0.03, Clubs: +0.02, Diamonds: +0.01
      Stone card: -1000.0 (always sorts to rightmost end)
    """
    if card.enhancement == Enhancement.STONE:
        return -1000.0

    rank_nominals = {
        Rank.TWO: (2.0, 0.0),
        Rank.THREE: (3.0, 0.0),
        Rank.FOUR: (4.0, 0.0),
        Rank.FIVE: (5.0, 0.0),
        Rank.SIX: (6.0, 0.0),
        Rank.SEVEN: (7.0, 0.0),
        Rank.EIGHT: (8.0, 0.0),
        Rank.NINE: (9.0, 0.0),
        Rank.TEN: (10.0, 0.0),
        Rank.JACK: (10.0, 0.1),
        Rank.QUEEN: (10.0, 0.2),
        Rank.KING: (10.0, 0.3),
        Rank.ACE: (11.0, 0.4),
    }
    suit_nominals = {
        Suit.DIAMONDS: 0.01,
        Suit.CLUBS: 0.02,
        Suit.HEARTS: 0.03,
        Suit.SPADES: 0.04,
    }
    base_nom, face_nom = rank_nominals.get(card.rank, (0.0, 0.0))
    suit_nom = suit_nominals.get(card.suit, 0.0)
    tie_break = (card.id * 1e-6) if card.id else 0.0
    return base_nom + face_nom + suit_nom + tie_break


def get_suit_sort_nominal(card: PlayingCard) -> float:
    """Calculates Balatro's exact nominal value for 'Sort by Suit' (suit desc).

    Nominals:
      Spades block: 40.0 + rank_nominal
      Hearts block: 30.0 + rank_nominal
      Clubs block: 20.0 + rank_nominal
      Diamonds block: 10.0 + rank_nominal
      Stone card: -1000.0
    """
    if card.enhancement == Enhancement.STONE:
        return -1000.0

    rank_nominals = {
        Rank.TWO: (2.0, 0.0),
        Rank.THREE: (3.0, 0.0),
        Rank.FOUR: (4.0, 0.0),
        Rank.FIVE: (5.0, 0.0),
        Rank.SIX: (6.0, 0.0),
        Rank.SEVEN: (7.0, 0.0),
        Rank.EIGHT: (8.0, 0.0),
        Rank.NINE: (9.0, 0.0),
        Rank.TEN: (10.0, 0.0),
        Rank.JACK: (10.0, 0.1),
        Rank.QUEEN: (10.0, 0.2),
        Rank.KING: (10.0, 0.3),
        Rank.ACE: (11.0, 0.4),
    }
    suit_nominals = {
        Suit.DIAMONDS: 0.01,
        Suit.CLUBS: 0.02,
        Suit.HEARTS: 0.03,
        Suit.SPADES: 0.04,
    }
    base_nom, face_nom = rank_nominals.get(card.rank, (0.0, 0.0))
    suit_nom = suit_nominals.get(card.suit, 0.0)
    tie_break = (card.id * 1e-6) if card.id else 0.0
    return base_nom + face_nom + (suit_nom * 1000.0) + tie_break


@dataclass(kw_only=True)
class GameState:
    """Full snapshot of the game state at a specific frame/decision boundary.

    Scope notes:
    - Starting mid-run is out of scope; simulations always start from Ante 1 / run start.
    - Challenge Decks (`Deck.CHALLENGE_DECK`) are out of scope.
    - Exact chip/mult score calculation is intentionally omitted (`current_chips` / round
      scores are captured externally via OCR).
    """

    step_idx: int = 0
    frame_idx: int = 0
    timestamp: float = 0.0
    location: Location = Location.BLIND
    stake: Stake = Stake.WHITE
    deck: Deck = Deck.RED_DECK
    ante: int = 1
    round: int = 1
    current_blind: BlindName = BlindName.SMALL
    target_chips: int | None = None
    current_chips: int = 0

    # Economy & Round Counters (formerly RunStats)
    money: int | None = None
    hands_left: int | None = None
    discards_left: int | None = None
    hands_played_this_round: int = 0
    discards_used_this_round: int = 0
    reroll_cost: int | None = None
    free_rerolls: int = 0
    inflation: int = 0
    bankrupt_at: int = 0
    base_hands: int = 4
    base_discards: int = 3
    base_reroll_cost: int = 5

    # Shop Rates, Limits, and Stake Flags (G.GAME / G.GAME.modifiers)
    shop_size: int = 2
    tarot_rate: float = 4.0
    planet_rate: float = 4.0
    playing_card_rate: float = 0.0
    spectral_rate: float = 0.0
    edition_rate: float = 1.0
    no_interest: bool = False
    ante_scaling: float = 1.0
    enable_eternals_in_shop: bool = False
    enable_perishables_in_shop: bool = False
    enable_rentals_in_shop: bool = False
    coupon_shop_active: bool = False

    # Inventory & Hand
    inventory: Inventory = field(default_factory=Inventory)
    hand: Hand = field(default_factory=Hand)

    # Shop Offerings (formerly ShopState)
    shop_cards: list[Card] = field(default_factory=list)
    shop_vouchers: list[Voucher] = field(default_factory=list)
    shop_packs: list[BoosterPack] = field(default_factory=list)

    # Booster Pack Opening State (Location.PACK)
    active_pack: BoosterPack | None = None
    pack_cards: list[Card] = field(default_factory=list)
    pack_choices_remaining: int = 0
    pre_pack_location: Location | None = None
    pending_packs: list[BoosterPack] = field(default_factory=list)

    # Skip Tags & Skip History
    small_blind_tag: SkipTag | None = None
    big_blind_tag: SkipTag | None = None
    small_blind_orbital_hand: PokerHand | None = None
    big_blind_orbital_hand: PokerHand | None = None
    active_tags: list[SkipTag] = field(default_factory=list)
    blinds_skipped: int = 0
    temp_hand_size_bonus: int = 0
    round_advanced: bool = False
    pending_blind_advance: bool = False

    # Vouchers, Consumable Memory & Scaling Modifiers
    vouchers_redeemed: list[VoucherName] = field(default_factory=list)
    last_tarot_planet: TarotName | PlanetName | None = None
    ectoplasm_minus: int = 1
    unique_planets_used: set[str] = field(default_factory=set)
    tarots_used: int = 0
    unused_discards: int = 0
    gros_michel_extinct: bool = False
    pending_cv_checks: set[str] = field(default_factory=set)

    # Per-Round / Per-Ante Dynamic Targets (G.GAME.current_round)
    idol_card: tuple[Rank, Suit] = (Rank.ACE, Suit.SPADES)
    mail_card_rank: Rank = Rank.ACE
    ancient_suit: Suit = Suit.SPADES
    castle_suit: Suit = Suit.SPADES
    to_do_poker_hand: PokerHand = PokerHand.HIGH_CARD

    # Boss Blind & Hand/Card Constraint State
    boss_blind: BlindName | None = None
    boss_disabled: bool = False
    round_won: bool = False
    hand_levels: dict[PokerHand, int] = field(default_factory=dict)
    hand_play_counts: dict[PokerHand, int] = field(default_factory=dict)
    hands_played_this_round_by_type: set[PokerHand] = field(default_factory=set)
    locked_hand_type: PokerHand | None = None
    ox_target_hand: PokerHand | None = None
    fish_prepped: bool = False
    crimson_heart_prepped: bool = False
    _manacle_active: bool = False
    _water_discards_sub: int = 0
    _needle_hands_sub: int = 0

    # Deck & Pile Tracking
    deck_cards_count: int | None = None
    starting_deck_size: int | None = None
    full_deck: list[PlayingCard] | None = None
    discard_pile: list[PlayingCard] = field(default_factory=list)
    played_cards_this_round: list[PlayingCard] = field(default_factory=list)
    played_cards_this_ante: list[PlayingCard] = field(default_factory=list)

    # PRNG Engine State
    seed: str = ""
    rng: Any = None
    resolver: Any = None
    bosses_used: dict[str, int] = field(default_factory=dict)
    apply_deck_stake_on_init: bool = field(default=True, repr=False, compare=False)
    _next_slot_uid: int = field(default=1, repr=False, compare=False)
    _next_joker_id: int = field(default=1, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.seed and self.rng is None:
            self.set_seed(self.seed)

        self._init_stake_and_deck_rules()
        self.ensure_joker_ids()

        if self.seed and self.boss_blind is None:
            self.roll_boss_blind(is_reroll=False)

        if self.seed and self.rng is not None and self.apply_deck_stake_on_init and self.full_deck:
            try:
                from game.engine.rng_sim.events import (
                    Deck as SimDeck,
                    get_ancient_joker_suit,
                    get_castle_suit,
                    get_idol_card,
                    get_mail_in_rebate_target,
                )
            except ImportError:
                from engine.rng_sim.events import (
                    Deck as SimDeck,
                    get_ancient_joker_suit,
                    get_castle_suit,
                    get_idol_card,
                    get_mail_in_rebate_target,
                )
            sdeck = SimDeck(self.full_deck)
            idol_target = get_idol_card(self.rng, sdeck, ante=self.ante)
            self.idol_card = (idol_target.rank, idol_target.suit)
            self.mail_card_rank = get_mail_in_rebate_target(
                self.rng, sdeck, ante=self.ante
            )
            # game.lua:2387 sets ancient_card.suit = nil before reset_ancient_card() on run start
            self.ancient_suit = get_ancient_joker_suit(
                self.rng, ante=self.ante, current_suit=None
            )
            self.castle_suit = get_castle_suit(self.rng, sdeck, ante=self.ante)

        if self.target_chips is None:
            self.target_chips = self.compute_target_chips()

    def _init_stake_and_deck_rules(self) -> None:
        """Applies Stake (White..Gold) and Deck modifiers onto initial run state.

        Note: Starting mid-run and Challenge Decks (`Deck.CHALLENGE_DECK`) are out of scope.
        """
        stake_level = get_stake_level(self.stake)
        if stake_level >= 4:
            self.enable_eternals_in_shop = True
        if stake_level >= 7:
            self.enable_perishables_in_shop = True
        if stake_level >= 8:
            self.enable_rentals_in_shop = True

        self.base_discards = 3 - (1 if stake_level >= 5 else 0)
        self.base_hands = 4
        bonus_dollars = 0

        match self.deck:
            case Deck.RED_DECK:
                self.base_discards += 1
            case Deck.BLUE_DECK:
                self.base_hands += 1
            case Deck.YELLOW_DECK:
                bonus_dollars += 10
            case Deck.GREEN_DECK:
                self.no_interest = True
            case Deck.BLACK_DECK:
                self.base_hands = max(1, self.base_hands - 1)
                if self.apply_deck_stake_on_init:
                    self.inventory.joker_slots += 1
            case Deck.MAGIC_DECK:
                if self.apply_deck_stake_on_init:
                    self.apply_voucher(
                        VoucherName.CRYSTAL_BALL, adjust_round_counters=False
                    )
                    if not self.inventory.consumables:
                        self.inventory.consumables.extend(
                            [Tarot(name=TarotName.FOOL), Tarot(name=TarotName.FOOL)]
                        )
            case Deck.NEBULA_DECK:
                if self.apply_deck_stake_on_init:
                    self.apply_voucher(
                        VoucherName.TELESCOPE, adjust_round_counters=False
                    )
                    self.inventory.consumable_slots = max(
                        0, self.inventory.consumable_slots - 1
                    )
            case Deck.GHOST_DECK:
                self.spectral_rate = 2.0
                if self.apply_deck_stake_on_init and not self.inventory.consumables:
                    self.inventory.consumables.append(Spectral(name=SpectralName.HEX))
            case Deck.ZODIAC_DECK:
                if self.apply_deck_stake_on_init:
                    self.apply_voucher(
                        VoucherName.TAROT_MERCHANT, adjust_round_counters=False
                    )
                    self.apply_voucher(
                        VoucherName.PLANET_MERCHANT, adjust_round_counters=False
                    )
                    self.apply_voucher(
                        VoucherName.OVERSTOCK, adjust_round_counters=False
                    )
            case Deck.PAINTED_DECK:
                if self.apply_deck_stake_on_init:
                    self.hand.max_size += 2
                    self.inventory.joker_slots = max(0, self.inventory.joker_slots - 1)
            case Deck.PLASMA_DECK:
                self.ante_scaling = 2.0
            case _:
                pass

        if self.full_deck is None:
            self.full_deck = self._build_initial_deck()
        if self.deck_cards_count is None:
            self.deck_cards_count = len(self.full_deck)
        if self.starting_deck_size is None:
            self.starting_deck_size = len(self.full_deck)

        if self.apply_deck_stake_on_init:
            if self.money is None:
                self.money = 4 + bonus_dollars
            if self.hands_left is None:
                self.hands_left = self.base_hands
            if self.discards_left is None:
                self.discards_left = max(0, self.base_discards)
            if self.reroll_cost is None:
                self.reroll_cost = self.base_reroll_cost
        else:
            if self.money is None:
                self.money = 4
            if self.hands_left is None:
                self.hands_left = 4
            if self.discards_left is None:
                self.discards_left = 3
            if self.reroll_cost is None:
                self.reroll_cost = self.base_reroll_cost

    def _build_initial_deck(self) -> list[PlayingCard]:
        """Builds the initial deck corresponding to self.deck."""
        if self.deck == Deck.ABANDONED_DECK:
            cards = [
                c
                for c in create_standard_deck()
                if c.rank not in (Rank.JACK, Rank.QUEEN, Rank.KING)
            ]
            for idx, c in enumerate(cards, start=1):
                c.id = idx
            return cards
        if self.deck == Deck.CHECKERED_DECK:
            cards = create_standard_deck()
            for c in cards:
                if c.suit == Suit.CLUBS:
                    c.suit = Suit.SPADES
                elif c.suit == Suit.DIAMONDS:
                    c.suit = Suit.HEARTS
            return cards
        if self.deck == Deck.ERRATIC_DECK:
            try:
                from game.engine.rng_sim.events import generate_erratic_deck
            except ImportError:
                from engine.rng_sim.events import generate_erratic_deck
            return generate_erratic_deck(self.rng)
        return create_standard_deck()

    def compute_target_chips(self, blind: BlindName | None = None) -> int:
        """Computes blind score requirement for the current/given blind."""
        target_blind = blind if blind is not None else self.current_blind
        return calculate_target_chips(
            ante=self.ante,
            round=self.round,
            stake=self.stake,
            blind=target_blind,
            deck=self.deck,
        )

    def has_joker(self, joker_name: JokerName) -> bool:
        """Returns True if an undebuffed Joker of the given name is in inventory."""
        if not self.inventory:
            return False
        return any(
            j.name == joker_name and not j.debuffed for j in self.inventory.jokers
        )

    def apply_voucher(
        self, voucher_name: VoucherName, adjust_round_counters: bool = True
    ) -> bool:
        """Applies a Voucher effect onto GameState idempotently; returns False if already redeemed."""
        if voucher_name in self.vouchers_redeemed:
            return False
        self.vouchers_redeemed.append(voucher_name)

        match voucher_name:
            case VoucherName.OVERSTOCK | VoucherName.OVERSTOCK_PLUS:
                self.shop_size += 1
            case VoucherName.TAROT_MERCHANT:
                self.tarot_rate = 4.0 * 2.4
            case VoucherName.TAROT_TYCOON:
                self.tarot_rate = 4.0 * 8.0
            case VoucherName.PLANET_MERCHANT:
                self.planet_rate = 4.0 * 2.4
            case VoucherName.PLANET_TYCOON:
                self.planet_rate = 4.0 * 8.0
            case VoucherName.HONE:
                self.edition_rate = max(self.edition_rate, 2.0)
            case VoucherName.GLOW_UP:
                self.edition_rate = max(self.edition_rate, 4.0)
            case VoucherName.MAGIC_TRICK | VoucherName.ILLUSION:
                self.playing_card_rate = 4.0
            case VoucherName.CRYSTAL_BALL:
                self.inventory.consumable_slots += 1
            case VoucherName.ANTIMATTER:
                self.inventory.joker_slots += 1
            case VoucherName.PAINT_BRUSH | VoucherName.PALETTE:
                self.hand.max_size += 1
            case VoucherName.REROLL_SURPLUS | VoucherName.REROLL_GLUT:
                self.base_reroll_cost = max(0, self.base_reroll_cost - 2)
                if adjust_round_counters:
                    self.reroll_cost = max(0, self.reroll_cost - 2)
            case VoucherName.GRABBER | VoucherName.NACHO_TONG:
                if adjust_round_counters:
                    self.hands_left += 1
            case VoucherName.WASTEFUL | VoucherName.RECYCLOMANCY:
                if adjust_round_counters:
                    self.discards_left += 1
            case VoucherName.HIEROGLYPH:
                self.ante = max(1, self.ante - 1)
                if adjust_round_counters:
                    self.hands_left = max(1, self.hands_left - 1)
            case VoucherName.PETROGLYPH:
                self.ante = max(1, self.ante - 1)
                if adjust_round_counters:
                    self.discards_left = max(0, self.discards_left - 1)
            case _:
                # BLANK, CLEARANCE_SALE, LIQUIDATION, SEED_MONEY, MONEY_TREE,
                # TELESCOPE, OBSERVATORY, OMEN_GLOBE, DIRECTORS_CUT, RETCON
                pass
        return True

    def get_observatory_x_mult(self, poker_hand: PokerHand) -> float:
        """Returns Observatory voucher X-mult (1.5^n for matching held Planet cards)."""
        if VoucherName.OBSERVATORY not in self.vouchers_redeemed or not self.inventory:
            return 1.0
        target_planet = {
            PokerHand.PAIR: PlanetName.MERCURY,
            PokerHand.THREE_OF_A_KIND: PlanetName.VENUS,
            PokerHand.TWO_PAIR: PlanetName.URANUS,
            PokerHand.FULL_HOUSE: PlanetName.EARTH,
            PokerHand.FLUSH: PlanetName.JUPITER,
            PokerHand.STRAIGHT: PlanetName.SATURN,
            PokerHand.FOUR_OF_A_KIND: PlanetName.MARS,
            PokerHand.STRAIGHT_FLUSH: PlanetName.NEPTUNE,
            PokerHand.FIVE_OF_A_KIND: PlanetName.PLANET_X,
            PokerHand.FLUSH_HOUSE: PlanetName.CERES,
            PokerHand.FLUSH_FIVE: PlanetName.ERIS,
            PokerHand.HIGH_CARD: PlanetName.PLUTO,
        }.get(poker_hand)
        if target_planet is None:
            return 1.0
        matches = sum(1 for c in self.inventory.consumables if c.name == target_planet)
        return float(1.5**matches)

    def balance_plasma_score(self, chips: float, mult: float) -> tuple[float, float]:
        """Balances Chips and Mult on Plasma Deck (back.lua:125-128)."""
        if self.deck != Deck.PLASMA_DECK:
            return chips, mult
        balanced = math.floor((chips + mult) / 2.0)
        return balanced, balanced

    def roll_boss_blind(self, is_reroll: bool = False) -> BlindName | None:
        """Deterministically evaluates or updates the Boss Blind using Balatro PRNG based on seed."""
        if not self.seed and self.rng is None:
            return self.boss_blind
        if self.rng is None and self.seed:
            self.set_seed(self.seed)
        try:
            from game.engine.rng_sim.events import get_next_boss_blind
        except ImportError:
            from engine.rng_sim.events import get_next_boss_blind
        if not hasattr(self, "bosses_used") or self.bosses_used is None:
            self.bosses_used = {}
        pred = get_next_boss_blind(
            self.rng,
            ante=self.ante,
            bosses_used=self.bosses_used,
            is_reroll=is_reroll,
        )
        self.boss_blind = pred.blind_name
        return self.boss_blind

    def roll_blind_tags(self) -> tuple[SkipTag | None, SkipTag | None]:
        """Deterministically populates Small & Big Blind Skip Tags + Orbital choices via PRNG."""
        if not self.seed and self.rng is None:
            return self.small_blind_tag, self.big_blind_tag
        if self.rng is None and self.seed:
            self.set_seed(self.seed)
        try:
            from game.engine.rng_sim.events import (
                get_next_tag,
                get_orbital_tag_hands_for_ante,
            )
        except ImportError:
            from engine.rng_sim.events import (
                get_next_tag,
                get_orbital_tag_hands_for_ante,
            )

        small_pred = get_next_tag(
            self.rng,
            ante=self.ante,
            blind_type="Small",
            state=None,
            roll_orbital_choice=False,
        )
        big_pred = get_next_tag(
            self.rng,
            ante=self.ante,
            blind_type="Big",
            state=None,
            roll_orbital_choice=False,
        )
        self.small_blind_tag = small_pred.skip_tag
        self.big_blind_tag = big_pred.skip_tag

        visible_hands = [
            h
            for h in PokerHand
            if h
            not in (
                PokerHand.FIVE_OF_A_KIND,
                PokerHand.FLUSH_HOUSE,
                PokerHand.FLUSH_FIVE,
            )
            or self.hand_play_counts.get(h, 0) > 0
        ]
        orb = get_orbital_tag_hands_for_ante(self.rng, visible_hands=visible_hands)
        self.small_blind_orbital_hand = orb["Small"]
        self.big_blind_orbital_hand = orb["Big"]
        return self.small_blind_tag, self.big_blind_tag

    def _is_card_face_for_blind(self, card: PlayingCard) -> bool:
        """Checks face status ignoring debuff state (matching Balatro `card:is_face(true)`)."""
        if self.has_joker(JokerName.PAREIDOLIA):
            return True
        if card.enhancement == Enhancement.STONE or card.rank == Rank.NONE:
            return False
        return card.is_face

    def _card_matches_suit_for_blind(
        self, card: PlayingCard, target_suit: Suit
    ) -> bool:
        """Checks suit match ignoring debuff state (matching Balatro `card:is_suit(suit, true)`)."""
        if card.enhancement == Enhancement.STONE or card.rank == Rank.NONE:
            return False
        if card.enhancement == Enhancement.WILD:
            return True
        if self.has_joker(JokerName.SMEARED_JOKER):
            if target_suit in (Suit.HEARTS, Suit.DIAMONDS):
                return card.suit in (Suit.HEARTS, Suit.DIAMONDS)
            if target_suit in (Suit.SPADES, Suit.CLUBS):
                return card.suit in (Suit.SPADES, Suit.CLUBS)
        return card.suit == target_suit

    def _was_played_this_ante(self, card: PlayingCard) -> bool:
        for prev in self.played_cards_this_ante:
            if card.id and prev.id:
                if card.id == prev.id:
                    return True
            elif _cards_match(card, prev):
                return True
        return False

    def card_is_debuffed_by_boss(self, card: PlayingCard) -> bool:
        """Evaluates whether `card` should be debuffed by the active Boss Blind."""
        if (
            self.boss_disabled
            or self.location != Location.BLIND
            or self.current_blind in (BlindName.SMALL, BlindName.BIG)
        ):
            return False
        suit_bosses = {
            BlindName.THE_CLUB: Suit.CLUBS,
            BlindName.THE_GOAD: Suit.SPADES,
            BlindName.THE_HEAD: Suit.HEARTS,
            BlindName.THE_WINDOW: Suit.DIAMONDS,
        }
        target_suit = suit_bosses.get(self.current_blind)
        if target_suit is not None:
            return self._card_matches_suit_for_blind(card, target_suit)
        if self.current_blind == BlindName.THE_PLANT:
            return self._is_card_face_for_blind(card)
        if self.current_blind == BlindName.THE_PILLAR:
            return self._was_played_this_ante(card)
        if self.current_blind == BlindName.VERDANT_LEAF:
            return True
        return False

    def refresh_debuffs(self) -> None:
        """Recomputes debuff status on all playing cards in hand and full_deck."""
        for hc in self.hand.cards:
            hc.card.debuffed = self.card_is_debuffed_by_boss(hc.card)
        if self.full_deck:
            for c in self.full_deck:
                c.debuffed = self.card_is_debuffed_by_boss(c)

    def update_ox_target_hand(self) -> None:
        """Sets `ox_target_hand` to the most played poker hand (tie-broken by highest hand rank)."""
        # Matches state_events.lua:130-137 (highest hand order: Flush Five..High Card)
        hand_order = (
            PokerHand.FLUSH_FIVE,
            PokerHand.FLUSH_HOUSE,
            PokerHand.FIVE_OF_A_KIND,
            PokerHand.STRAIGHT_FLUSH,
            PokerHand.FOUR_OF_A_KIND,
            PokerHand.FULL_HOUSE,
            PokerHand.FLUSH,
            PokerHand.STRAIGHT,
            PokerHand.THREE_OF_A_KIND,
            PokerHand.TWO_PAIR,
            PokerHand.PAIR,
            PokerHand.HIGH_CARD,
        )
        best_hand = PokerHand.HIGH_CARD
        best_count = -1
        for h in hand_order:
            count = self.hand_play_counts.get(h, 0)
            if count > best_count:
                best_count = count
                best_hand = h
        self.ox_target_hand = best_hand

    def apply_crimson_heart(self) -> None:
        """Disables 1 random Joker for the current hand under Crimson Heart."""
        self.crimson_heart_prepped = False
        if (
            self.boss_disabled
            or self.current_blind != BlindName.CRIMSON_HEART
            or not self.inventory
            or not self.inventory.jokers
        ):
            return
        jokers = self.inventory.jokers
        candidates = [
            j
            for j in jokers
            if (not j.debuffed or len(jokers) < 2)
            and not (j.perishable and j.perish_tally <= 0)
        ]
        for j in jokers:
            if not (j.perishable and j.perish_tally <= 0):
                j.debuffed = False
        if not candidates:
            return
        if self.rng is not None:
            chosen, _ = self.rng.choice("crimson_heart", candidates)
            chosen.debuffed = True
        else:
            self.pending_cv_checks.add("jokers")

    def apply_cerulean_bell(self) -> None:
        """Forces 1 card in hand to stay selected under Cerulean Bell."""
        if (
            self.boss_disabled
            or self.current_blind != BlindName.CERULEAN_BELL
            or not self.hand.cards
        ):
            return
        if any(hc.forced_selection for hc in self.hand.cards):
            return
        for hc in self.hand.cards:
            hc.selected = False
            hc.forced_selection = False
        if self.rng is not None:
            chosen, _ = self.rng.choice("cerulean_bell", self.hand.cards)
            chosen.forced_selection = True
            chosen.selected = True

    def clear_boss_round_modifiers(self) -> None:
        """Restores round-scoped Boss Blind mutations at end of round or defeat."""
        if self._manacle_active:
            self.hand.max_size += 1
            self._manacle_active = False
        self.fish_prepped = False
        self.crimson_heart_prepped = False
        if self.inventory:
            for j in self.inventory.jokers:
                j.facing_back = False
                if not (j.perishable and j.perish_tally <= 0):
                    j.debuffed = False
        for hc in self.hand.cards:
            hc.forced_selection = False
            hc.card.debuffed = False
            if hc.facing_back and hc._real_card:
                hc.resolve_face_down(hc._real_card, lift_facing_back=True)
                self.sync_card_to_deck(hc.card, rebind_hand_card=hc)
            else:
                hc.facing_back = False
        if self.full_deck:
            for c in self.full_deck:
                c.debuffed = False

    def disable_active_boss(self) -> None:
        """Disables the currently active Boss Blind (`Blind:disable`, Chicot / Luchador / Verdant Leaf)."""
        if self.current_blind in (BlindName.SMALL, BlindName.BIG):
            return
        self.boss_disabled = True
        if self._water_discards_sub:
            if self.discards_left is not None:
                self.discards_left += self._water_discards_sub
            self._water_discards_sub = 0
        if self._needle_hands_sub:
            if self.hands_left is not None:
                self.hands_left += self._needle_hands_sub
            self._needle_hands_sub = 0
        if self.current_blind in (BlindName.THE_WALL, BlindName.VIOLET_VESSEL):
            base = get_blind_amount(self.ante, self.stake)
            deck_scaling = 2 if self.deck == Deck.PLASMA_DECK else 1
            self.target_chips = int(base * 2.0 * deck_scaling)
        self.clear_boss_round_modifiers()

    def set_seed(self, seed: str) -> None:
        """Initializes or resets the deterministic BalatroRNG and EventResolver on this GameState."""
        self.seed = str(seed).strip().upper()
        try:
            from game.engine.rng_sim.rng import BalatroRNG
            from game.engine.rng_sim.events import EventResolver
        except ImportError:
            from engine.rng_sim.rng import BalatroRNG
            from engine.rng_sim.events import EventResolver
        self.rng = BalatroRNG(self.seed)
        self.resolver = EventResolver(self.rng)
        if self.boss_blind is None and self.seed:
            self.roll_boss_blind(is_reroll=False)
        if self.small_blind_tag is None and self.big_blind_tag is None and self.seed:
            self.roll_blind_tags()

    @property
    def discount_percent(self) -> int:
        """Shop discount percentage from Clearance Sale (25%) or Liquidation (50%)."""
        if VoucherName.LIQUIDATION in self.vouchers_redeemed:
            return 50
        if VoucherName.CLEARANCE_SALE in self.vouchers_redeemed:
            return 25
        return 0

    def sync_card_to_deck(
        self,
        card: PlayingCard | None,
        *,
        rebind_hand_card: HandCard | None = None,
    ) -> PlayingCard | None:
        """Propagates in-place mutations on a hand/played card back to its matching `full_deck` entry by `id`."""
        if card is None or card.id == 0 or not self.full_deck:
            return None
        for dc in self.full_deck:
            if dc.id == card.id:
                if dc is not card:
                    dc.rank = card.rank
                    dc.suit = card.suit
                    dc.enhancement = card.enhancement
                    dc.edition = card.edition
                    dc.seal = card.seal
                    dc.bonus_chips = card.bonus_chips
                    dc.debuffed = card.debuffed
                if rebind_hand_card is not None and not rebind_hand_card.facing_back:
                    rebind_hand_card.card = dc
                return dc
        return None

    def add_card_to_deck(self, card: PlayingCard, add_to_hand: bool = False) -> None:
        """Adds a card to the run's full deck (and optionally hand) and triggers Hologram."""
        existing_ids = {c.id for c in self.full_deck}
        if card.id == 0 or card.id in existing_ids:
            card.id = max(existing_ids, default=0) + 1
        self.full_deck.append(card)
        self.deck_cards_count = len(self.full_deck)
        if add_to_hand:
            self.hand.cards.append(HandCard(card=card, selected=False))
        if self.inventory:
            for j in self.inventory.jokers:
                if j.name == JokerName.HOLOGRAM and not j.debuffed:
                    j.state["x_mult"] = round(j.state.get("x_mult", 1.0) + 0.25, 2)

    def remove_card_from_deck(self, card: PlayingCard, shattered: bool = False) -> bool:
        """Removes a single matching card from full deck, discard pile, and hand when destroyed."""
        removed = False
        popped_card: PlayingCard | None = None
        if card.id != 0:
            for i, c in enumerate(self.full_deck):
                if c.id == card.id:
                    popped_card = self.full_deck.pop(i)
                    popped_card.destroyed = True
                    removed = True
                    break
        if not removed:
            for i, c in enumerate(self.full_deck):
                if _cards_match(c, card):
                    popped_card = self.full_deck.pop(i)
                    popped_card.destroyed = True
                    removed = True
                    break
        if not removed:
            for i, c in enumerate(self.full_deck):
                if c.rank == card.rank and c.suit == card.suit:
                    popped_card = self.full_deck.pop(i)
                    popped_card.destroyed = True
                    removed = True
                    break
        if not removed and self.full_deck:
            popped_card = self.full_deck.pop()
            popped_card.destroyed = True
            removed = True
        for i, c in enumerate(self.discard_pile):
            if (card.id != 0 and c.id == card.id) or _cards_match(c, card):
                self.discard_pile.pop(i)
                break
        hand_removed = False
        if card.id != 0:
            for i, hc in enumerate(self.hand.cards):
                hc_id = hc._real_card.id if (hc.facing_back and hc._real_card) else hc.card.id
                if hc_id == card.id:
                    self.hand.cards.pop(i)
                    hand_removed = True
                    break
        if not hand_removed:
            for i, hc in enumerate(self.hand.cards):
                if _cards_match(hc.card, card):
                    self.hand.cards.pop(i)
                    break
        card.destroyed = True
        destroyed_id = (
            popped_card.id if popped_card and popped_card.id else card.id
        ) or 0
        for hc in self.hand.cards:
            if hc.bounds:
                hc.bounds.candidate_cards = [
                    cc
                    for cc in hc.bounds.candidate_cards
                    if not getattr(cc, "destroyed", False)
                    and (destroyed_id == 0 or cc.id != destroyed_id)
                ]
                if destroyed_id:
                    hc.bounds.candidate_card_ids.discard(destroyed_id)
        if removed:
            self.deck_cards_count = len(self.full_deck)
        else:
            self.deck_cards_count = max(0, self.deck_cards_count - 1)
        if self.inventory:
            has_pareidolia = any(
                j.name == JokerName.PAREIDOLIA and not j.debuffed
                for j in self.inventory.jokers
            )
            is_face = (
                card.enhancement != Enhancement.STONE
                and card.rank != Rank.NONE
                and not card.debuffed
                and (card.is_face or has_pareidolia)
            )
            for j in self.inventory.jokers:
                if j.debuffed:
                    continue
                if is_face and j.name == JokerName.CANIO:
                    j.state["x_mult"] = round(j.state.get("x_mult", 1.0) + 1.0, 2)
                elif (
                    card.enhancement == Enhancement.GLASS
                    and shattered
                    and j.name == JokerName.GLASS_JOKER
                ):
                    j.state["x_mult"] = round(j.state.get("x_mult", 1.0) + 0.75, 2)
        return removed

    @property
    def remaining_deck(self) -> list[PlayingCard]:
        """Cards remaining in draw pile: full_deck minus hand, and minus discard/played during an active blind."""
        drawn_or_used = [hc.card for hc in self.hand.cards]
        if self.location == Location.BLIND and not self.round_won:
            # Played non-destroyed cards also sit in discard_pile; deduplicate so they are only subtracted once.
            discard_ids = {c.id for c in self.discard_pile if c.id}
            discard_ptrs = {id(c) for c in self.discard_pile}
            extra_played = [
                c
                for c in self.played_cards_this_round
                if id(c) not in discard_ptrs and (not c.id or c.id not in discard_ids)
            ]
            drawn_or_used = drawn_or_used + self.discard_pile + extra_played
        remaining = list(self.full_deck)
        for card in drawn_or_used:
            # Prefer exact persistent card.id match before attribute fallback so
            # identical clones (e.g. DNA / Death) subtract the exact drawn/discarded copy.
            matched_idx: int | None = None
            if card.id:
                for i, rem_card in enumerate(remaining):
                    if rem_card.id == card.id:
                        matched_idx = i
                        break
            if matched_idx is None:
                for i, rem_card in enumerate(remaining):
                    if _cards_match(rem_card, card):
                        matched_idx = i
                        break
            if matched_idx is not None:
                remaining.pop(matched_idx)
        return remaining

    def _get_next_slot_uid(self) -> int:
        uid = self._next_slot_uid
        self._next_slot_uid += 1
        return uid

    def assign_joker_id(self, joker: Joker, *, force_new: bool = False) -> int:
        """Assigns a unique persistent non-zero ID to `joker` (skipping existing IDs)."""
        existing = self.inventory.jokers if self.inventory else []
        used_ids = {j.id for j in existing if j is not joker and j.id != 0}
        if not force_new and joker.id != 0 and joker.id not in used_ids:
            self._next_joker_id = max(self._next_joker_id, joker.id + 1)
            return joker.id

        uid = max(self._next_joker_id, max(used_ids, default=0) + 1)
        while uid in used_ids:
            uid += 1
        joker.id = uid
        self._next_joker_id = uid + 1
        return uid

    def ensure_joker_ids(self) -> None:
        """Ensures every Joker in inventory has a distinct non-zero persistent ID."""
        if not self.inventory or not self.inventory.jokers:
            return
        seen: set[int] = set()
        for j in self.inventory.jokers:
            if j.id != 0 and j.id not in seen:
                seen.add(j.id)
                self._next_joker_id = max(self._next_joker_id, j.id + 1)
            else:
                self.assign_joker_id(j, force_new=True)
                seen.add(j.id)

    def _match_observed_jokers(
        self, observed_jokers: Sequence[Joker]
    ) -> list[Joker | None]:
        """Matches visually observed CV Jokers against tracked inventory Jokers.

        Prioritizes exact attribute matches before edition/name fallbacks, and resolves
        visually indistinguishable copies (e.g. duplicate Blueprints or Showman copies)
        arbitrarily in pool order so persistent IDs remain valid across drag reordering.
        """
        self.ensure_joker_ids()
        existing = list(self.inventory.jokers) if self.inventory else []
        available = list(range(len(existing)))
        matched: list[Joker | None] = [None] * len(observed_jokers)

        predicates: tuple[Callable[[Joker, Joker], bool], ...] = (
            lambda e, o: (
                e.name == o.name
                and e.edition == o.edition
                and e.eternal == o.eternal
                and e.perishable == o.perishable
                and e.rental == o.rental
                and e.debuffed == o.debuffed
                and e.facing_back == o.facing_back
            ),
            lambda e, o: e.name == o.name and e.edition == o.edition,
            lambda e, o: e.name == o.name,
        )

        for pred in predicates:
            for obs_idx, obs in enumerate(observed_jokers):
                if matched[obs_idx] is not None:
                    continue
                for pos, ex_idx in enumerate(available):
                    if pred(existing[ex_idx], obs):
                        matched[obs_idx] = existing[ex_idx]
                        available.pop(pos)
                        break

        return matched

    def match_cv_joker_order(self, observed_jokers: Sequence[Joker]) -> list[int]:
        """Returns persistent Joker IDs matching observed left-to-right CV order."""
        if not self.inventory or len(observed_jokers) != len(self.inventory.jokers):
            return []
        matched = self._match_observed_jokers(observed_jokers)
        if any(m is None for m in matched):
            return []
        return [m.id for m in matched if m is not None]

    def reconcile_cv_jokers(
        self,
        observed_jokers: Sequence[Joker],
        *,
        trigger_added_hooks: bool = False,
    ) -> list[int]:
        """Reconciles `inventory.jokers` with CV observations, preserving IDs & scaled stats."""
        if not self.inventory:
            return []
        matched = self._match_observed_jokers(observed_jokers)
        try:
            from game.engine.jokers import on_joker_added, sync_dynamic_joker_states
        except ImportError:
            from engine.jokers import on_joker_added, sync_dynamic_joker_states

        reconciled: list[Joker] = [
            match_j if match_j is not None else copy.deepcopy(obs)
            for obs, match_j in zip(observed_jokers, matched)
        ]
        self.inventory.jokers = reconciled
        for j, match_j in zip(reconciled, matched):
            if match_j is None:
                self.assign_joker_id(j)
                if trigger_added_hooks:
                    on_joker_added(self, j)
        sync_dynamic_joker_states(self)
        return [j.id for j in self.inventory.jokers]

    def ensure_slot_uids(self) -> None:
        """Assigns persistent slot UIDs and initializes CardBounds for any unassigned hand cards."""
        for hc in self.hand.cards:
            if hc.slot_uid == 0:
                hc.slot_uid = self._get_next_slot_uid()
            if hc.facing_back:
                if hc.bounds is None:
                    hc.bounds = CardBounds(is_face_down=True)
                if not hc.bounds.candidate_card_ids:
                    candidates = self.get_face_down_candidate_pool(hc)
                    hc.bounds.set_candidates(candidates)

    def get_face_down_candidate_pool(
        self, target_hc: HandCard | None = None
    ) -> list[PlayingCard]:
        """Gathers available candidate PlayingCard objects that could occupy a face-down slot."""
        resolved_in_hand_ids = {
            c.card.id
            for c in self.hand.cards
            if c is not target_hc
            and (not c.facing_back or (c.bounds and c.bounds.is_resolved))
            and c.card.id != 0
        }
        unavailable_ids = set(resolved_in_hand_ids)
        if self.location == Location.BLIND and not self.round_won:
            unavailable_ids |= {c.id for c in self.discard_pile if c.id}
            unavailable_ids |= {c.id for c in self.played_cards_this_round if c.id}

        base_pool = list(self.remaining_deck)
        if not base_pool and self.full_deck:
            base_pool = [c for c in self.full_deck if c.id not in unavailable_ids]

        pool: list[PlayingCard] = []
        for c in base_pool:
            if c.id not in unavailable_ids:
                pool.append(c)

        if target_hc is not None:
            if target_hc._real_card and not any(
                _cards_match(c, target_hc._real_card) for c in pool
            ):
                if target_hc._real_card.id not in unavailable_ids:
                    pool.append(target_hc._real_card)
            elif target_hc.bounds and target_hc.bounds.candidate_cards:
                for c in target_hc.bounds.candidate_cards:
                    if c.id not in unavailable_ids and not any(
                        _cards_match(p, c) for p in pool
                    ):
                        pool.append(c)

        return pool

    def apply_sort_observation(
        self, sort_type: str = "rank", selected_slot_uid: int | None = None
    ) -> None:
        """Updates hand order and narrows CardBounds for all face-down cards from sort order."""
        self.ensure_slot_uids()
        norm_type = sort_type.lower()
        is_suit_sort = "suit" in norm_type
        nominal_fn = get_suit_sort_nominal if is_suit_sort else get_rank_sort_nominal

        # 1. Deterministic sort if simulation or cards have true values
        can_simulate_sort = all(
            (
                c._real_card is not None
                or c.card.rank != Rank.NONE
                or c.card.enhancement == Enhancement.STONE
            )
            for c in self.hand.cards
        )
        if can_simulate_sort:
            self.hand.cards.sort(
                key=lambda hc: nominal_fn(hc._real_card if hc._real_card else hc.card),
                reverse=True,
            )

        # 2. Iterate deduction passes until fixpoint is reached
        changed = True
        while changed:
            changed = False
            for i, hc in enumerate(self.hand.cards):
                if (hc.bounds and hc.bounds.is_resolved) or (
                    not hc.facing_back and hc.bounds is None
                ):
                    continue

                if hc.bounds is None:
                    hc.bounds = CardBounds(is_face_down=True)

                # Determine upper nominal bound from nearest known face-up card to the left
                upper_nominal = float("inf")
                for left_i in range(i - 1, -1, -1):
                    left_card = self.hand.cards[left_i]
                    if not left_card.facing_back and left_card.card.rank != Rank.NONE:
                        upper_nominal = nominal_fn(left_card.card)
                        break

                # Determine lower nominal bound from nearest known face-up card to the right
                lower_nominal = float("-inf")
                for right_i in range(i + 1, len(self.hand.cards)):
                    right_card = self.hand.cards[right_i]
                    if not right_card.facing_back and right_card.card.rank != Rank.NONE:
                        lower_nominal = nominal_fn(right_card.card)
                        break

                # Filter candidates against bounds
                pool = self.get_face_down_candidate_pool(hc)
                valid_candidates: list[PlayingCard] = []
                for cand in pool:
                    cand_nom = nominal_fn(cand)
                    if (
                        cand_nom > upper_nominal + 1e-5
                        or cand_nom < lower_nominal - 1e-5
                    ):
                        continue
                    if cand.rank not in hc.bounds.possible_ranks:
                        continue
                    if cand.suit not in hc.bounds.possible_suits:
                        continue
                    if cand.enhancement not in hc.bounds.possible_enhancements:
                        continue
                    if cand.edition not in hc.bounds.possible_editions:
                        continue
                    if cand.seal not in hc.bounds.possible_seals:
                        continue
                    valid_candidates.append(cand)

                # Update bounds
                prev_ids = set(hc.bounds.candidate_card_ids)
                hc.bounds.set_candidates(valid_candidates)

                if prev_ids != hc.bounds.candidate_card_ids:
                    changed = True

                # Check if exactly 1 candidate remains -> resolve!
                if hc.bounds.is_resolved and valid_candidates:
                    resolved = valid_candidates[0]
                    hc.resolve_face_down(resolved)
                    changed = True
                    break  # Restart loop with new resolved card as face-up boundary

        self.reconcile_face_down_adjacencies()

    def reconcile_face_down_adjacencies(self) -> None:
        """Enforces visual adjacency constraints and human memory bounds for face-down cards.

        In Balatro's UI, cards in hand are displayed in a horizontal sequence:
        1. Isolated face-down cards (separated from other face-down cards by at least one
           face-up card) remain visually distinct landmarks and retain all specific
           information known about them (e.g. from Tarot flip-reveals or deductions).
        2. Contiguous adjacent face-down cards without face-up dividers that share the same
           visual signature (selection highlight, seal sticker, or edition sheen on the back)
           are visually indistinguishable to human perception. They lose individual positional
           certainty and conflate their candidate pools to the union of their possibilities,
           forgetting specific single-card identity.
        """
        cards = self.hand.cards
        if len(cards) <= 1:
            return

        # 1. Partition hand into contiguous segments of face-down cards
        segments: list[list[int]] = []
        current_segment: list[int] = []

        for idx, hc in enumerate(cards):
            is_fd = hc.facing_back or (hc.bounds is not None and hc.bounds.is_face_down)
            if is_fd:
                current_segment.append(idx)
            else:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
        if current_segment:
            segments.append(current_segment)

        # 2. Process each contiguous segment
        for segment in segments:
            # An isolated face-down card (length 1) has visual landmarks on both sides -> retains identity!
            if len(segment) < 2:
                continue

            # Group adjacent cards by visual appearance on the card back:
            # - selected state (raised vs resting)
            # - visible seal (e.g. Red, Blue, Gold, Purple seal sticker on back)
            # - visible edition (e.g. Foil, Holographic, Polychrome sheen on back)
            vis_groups: dict[tuple, list[int]] = {}
            for idx in segment:
                hc = cards[idx]
                seal_vis = hc.card.seal
                if (
                    seal_vis == Seal.NONE
                    and hc._real_card
                    and hc._real_card.seal != Seal.NONE
                ):
                    seal_vis = hc._real_card.seal
                ed_vis = hc.card.edition
                if (
                    ed_vis == Edition.NONE
                    and hc._real_card
                    and hc._real_card.edition != Edition.NONE
                ):
                    ed_vis = hc._real_card.edition

                vis_sig = (hc.selected, seal_vis, ed_vis)
                vis_groups.setdefault(vis_sig, []).append(idx)

            for vis_sig, group_indices in vis_groups.items():
                if len(group_indices) >= 2:
                    pooled_candidates: list[PlayingCard] = []
                    seen_ids: set[int] = set()

                    for idx in group_indices:
                        hc = cards[idx]
                        if hc.bounds and hc.bounds.candidate_cards:
                            cands = hc.bounds.candidate_cards
                        else:
                            cands = self.get_face_down_candidate_pool(hc)

                        for c in cands:
                            cid = c.id if c.id else id(c)
                            if cid not in seen_ids:
                                seen_ids.add(cid)
                                pooled_candidates.append(c)

                    for idx in group_indices:
                        hc = cards[idx]
                        if hc._real_card:
                            cid = (
                                hc._real_card.id
                                if hc._real_card.id
                                else id(hc._real_card)
                            )
                            if cid not in seen_ids:
                                seen_ids.add(cid)
                                pooled_candidates.append(hc._real_card)

                    if len(pooled_candidates) >= 2:
                        for idx in group_indices:
                            hc = cards[idx]
                            hc.blur_candidates(pooled_candidates)

    def update_face_down_candidates(self) -> None:
        """Prunes candidate pools of unresolved face-down cards against remaining deck."""
        changed = True
        while changed:
            changed = False
            for hc in self.hand.cards:
                if (hc.bounds and hc.bounds.is_resolved) or (
                    not hc.facing_back and hc.bounds is None
                ):
                    continue
                if hc.bounds is None:
                    hc.bounds = CardBounds(is_face_down=True)
                pool = self.get_face_down_candidate_pool(hc)
                if not hc.bounds.candidate_card_ids:
                    valid = pool
                else:
                    valid = [c for c in pool if c.id in hc.bounds.candidate_card_ids]
                prev_len = len(hc.bounds.candidate_card_ids)
                hc.bounds.set_candidates(valid)
                if len(hc.bounds.candidate_card_ids) != prev_len:
                    changed = True
                if hc.bounds.is_resolved and valid:
                    hc.resolve_face_down(valid[0])
                    changed = True
                    break

    def draw_cards(
        self, count: int | None = None, face_down: bool = False
    ) -> list[HandCard]:
        """Draws cards from remaining deck into hand, appending them strictly to the right."""
        self.ensure_slot_uids()
        needed = (self.hand.max_size - len(self.hand.cards)) if count is None else count
        if (
            count is None
            and self.current_blind == BlindName.THE_SERPENT
            and not self.boss_disabled
            and (self.hands_played_this_round > 0 or self.discards_used_this_round > 0)
        ):
            needed = 3
        if needed <= 0:
            return []

        rem = self.remaining_deck
        drawn_playing_cards = rem[:needed]
        new_hand_cards: list[HandCard] = []

        oops_mod = 1.0
        if self.inventory:
            oops_count = sum(
                1
                for j in self.inventory.jokers
                if j.name == JokerName.OOPS_ALL_6S and not j.debuffed
            )
            oops_mod = float(2**oops_count)

        for c in drawn_playing_cards:
            c.debuffed = self.card_is_debuffed_by_boss(c)
            is_flipped = face_down
            if (
                not is_flipped
                and not self.boss_disabled
                and self.location == Location.BLIND
            ):
                if (
                    self.current_blind == BlindName.THE_HOUSE
                    and self.hands_played_this_round == 0
                    and self.discards_used_this_round == 0
                ):
                    is_flipped = True
                elif (
                    self.current_blind == BlindName.THE_MARK
                    and self._is_card_face_for_blind(c)
                ):
                    is_flipped = True
                elif self.current_blind == BlindName.THE_FISH and self.fish_prepped:
                    is_flipped = True
                elif self.current_blind == BlindName.THE_WHEEL and self.rng is not None:
                    if self.rng.poll_odds("wheel", 1, 7, oops_mod):
                        is_flipped = True

            uid = self._get_next_slot_uid()
            if is_flipped:
                hc = HandCard(card=c, facing_back=True, slot_uid=uid)
                hc.bounds = CardBounds(is_face_down=True)
                hc.bounds.set_candidates(self.remaining_deck)
            else:
                hc = HandCard(card=c, facing_back=False, slot_uid=uid)
            self.hand.cards.append(hc)
            new_hand_cards.append(hc)

        self.reconcile_face_down_adjacencies()
        if self.crimson_heart_prepped:
            self.apply_crimson_heart()
        if self.current_blind == BlindName.CERULEAN_BELL and not self.boss_disabled:
            self.apply_cerulean_bell()
        return new_hand_cards

    def copy(self) -> "GameState":
        return copy.deepcopy(self)


# Scalar/collection fields on GameState that are diffed directly by name in scalar_diffs
_TRACKED_STATE_FIELDS: tuple[str, ...] = (
    "location",
    "stake",
    "deck",
    "ante",
    "round",
    "current_blind",
    "target_chips",
    "current_chips",
    "money",
    "hands_left",
    "discards_left",
    "hands_played_this_round",
    "discards_used_this_round",
    "reroll_cost",
    "free_rerolls",
    "inflation",
    "bankrupt_at",
    "shop_cards",
    "shop_vouchers",
    "shop_packs",
    "active_pack",
    "pack_cards",
    "pack_choices_remaining",
    "small_blind_tag",
    "big_blind_tag",
    "active_tags",
    "blinds_skipped",
    "vouchers_redeemed",
    "last_tarot_planet",
    "ectoplasm_minus",
    "unique_planets_used",
    "tarots_used",
    "unused_discards",
    "gros_michel_extinct",
    "pending_cv_checks",
    "idol_card",
    "mail_card_rank",
    "ancient_suit",
    "castle_suit",
    "to_do_poker_hand",
    "boss_blind",
    "boss_disabled",
    "round_won",
    "hand_levels",
    "hand_play_counts",
    "hands_played_this_round_by_type",
    "locked_hand_type",
    "ox_target_hand",
    "deck_cards_count",
    "starting_deck_size",
    "full_deck",
    "discard_pile",
    "played_cards_this_round",
    "played_cards_this_ante",
    "base_hands",
    "base_discards",
    "base_reroll_cost",
    "shop_size",
    "tarot_rate",
    "planet_rate",
    "playing_card_rate",
    "spectral_rate",
    "edition_rate",
    "no_interest",
    "ante_scaling",
    "enable_eternals_in_shop",
    "enable_perishables_in_shop",
    "enable_rentals_in_shop",
    "coupon_shop_active",
    "pre_pack_location",
    "pending_packs",
    "small_blind_orbital_hand",
    "big_blind_orbital_hand",
    "temp_hand_size_bonus",
    "round_advanced",
    "pending_blind_advance",
    "fish_prepped",
    "crimson_heart_prepped",
    "_manacle_active",
    "_water_discards_sub",
    "_needle_hands_sub",
    "seed",
)


@dataclass(kw_only=True)
class GameStateDelta:
    """Sparse delta between two GameStates (P-frame). None fields indicate no change."""

    step_idx: int
    frame_idx: int
    timestamp: float
    action: Action | None = None

    location: Location | None = None
    ante: int | None = None
    round: int | None = None
    current_blind: BlindName | None = None
    boss_blind: BlindName | None = None
    target_chips: int | None = None
    current_chips: int | None = None
    deck_cards_count: int | None = None
    full_deck_diff: list[PlayingCard] | None = None
    discard_pile_diff: list[PlayingCard] | None = None
    played_cards_diff: list[PlayingCard] | None = None

    field_diffs: dict[str, Any] | None = None
    jokers_diff: list[Joker] | None = None
    consumables_diff: list[Any] | None = None
    joker_slots_diff: int | None = None
    consumable_slots_diff: int | None = None
    hand_diff: list[HandCard] | None = None
    hand_max_size_diff: int | None = None
    vouchers_diff: list[VoucherName] | None = None
    hand_levels_diff: dict[PokerHand, int] | None = None
    hand_play_counts_diff: dict[PokerHand, int] | None = None
    unique_planets_diff: set[str] | None = None


def diff_states(
    prev: GameState, curr: GameState, action: Action | None = None
) -> GameStateDelta:
    """Generates the minimal GameStateDelta between prev and curr."""
    delta = GameStateDelta(
        step_idx=curr.step_idx,
        frame_idx=curr.frame_idx,
        timestamp=curr.timestamp,
        action=action,
    )

    if prev.location != curr.location:
        delta.location = curr.location
    if prev.ante != curr.ante:
        delta.ante = curr.ante
    if prev.round != curr.round:
        delta.round = curr.round
    if prev.current_blind != curr.current_blind:
        delta.current_blind = curr.current_blind
    if prev.boss_blind != curr.boss_blind:
        delta.boss_blind = curr.boss_blind
    if prev.target_chips != curr.target_chips:
        delta.target_chips = curr.target_chips
    if prev.current_chips != curr.current_chips:
        delta.current_chips = curr.current_chips

    field_diffs: dict[str, Any] = {}
    for attr in _TRACKED_STATE_FIELDS:
        prev_val = getattr(prev, attr)
        curr_val = getattr(curr, attr)
        if prev_val != curr_val:
            field_diffs[attr] = copy.deepcopy(curr_val)
    if field_diffs:
        delta.field_diffs = field_diffs

    if prev.inventory.jokers != curr.inventory.jokers:
        delta.jokers_diff = copy.deepcopy(curr.inventory.jokers)
    if prev.inventory.consumables != curr.inventory.consumables:
        delta.consumables_diff = copy.deepcopy(curr.inventory.consumables)
    if prev.inventory.joker_slots != curr.inventory.joker_slots:
        delta.joker_slots_diff = curr.inventory.joker_slots
    if prev.inventory.consumable_slots != curr.inventory.consumable_slots:
        delta.consumable_slots_diff = curr.inventory.consumable_slots

    if prev.hand.cards != curr.hand.cards:
        delta.hand_diff = copy.deepcopy(curr.hand.cards)
    if prev.hand.max_size != curr.hand.max_size:
        delta.hand_max_size_diff = curr.hand.max_size

    if prev.vouchers_redeemed != curr.vouchers_redeemed:
        delta.vouchers_diff = list(curr.vouchers_redeemed)

    if prev.deck_cards_count != curr.deck_cards_count:
        delta.deck_cards_count = curr.deck_cards_count
    if prev.full_deck != curr.full_deck or (
        prev.full_deck is not None
        and curr.full_deck is not None
        and any(
            p.debuffed != c.debuffed for p, c in zip(prev.full_deck, curr.full_deck)
        )
    ):
        delta.full_deck_diff = copy.deepcopy(curr.full_deck)
    if prev.discard_pile != curr.discard_pile:
        delta.discard_pile_diff = copy.deepcopy(curr.discard_pile)
    if prev.played_cards_this_round != curr.played_cards_this_round:
        delta.played_cards_diff = copy.deepcopy(curr.played_cards_this_round)
    if prev.hand_levels != curr.hand_levels:
        delta.hand_levels_diff = dict(curr.hand_levels)
    if prev.hand_play_counts != curr.hand_play_counts:
        delta.hand_play_counts_diff = dict(curr.hand_play_counts)
    if prev.unique_planets_used != curr.unique_planets_used:
        delta.unique_planets_diff = set(curr.unique_planets_used)

    return delta


def apply_delta(base: GameState, delta: GameStateDelta) -> GameState:
    """Applies a GameStateDelta onto a base GameState to produce the updated GameState."""
    next_state = base.copy()
    next_state.step_idx = delta.step_idx
    next_state.frame_idx = delta.frame_idx
    next_state.timestamp = delta.timestamp

    if delta.field_diffs:
        for k, v in delta.field_diffs.items():
            setattr(next_state, k, copy.deepcopy(v))

    if delta.location is not None:
        next_state.location = delta.location
    if delta.ante is not None:
        next_state.ante = delta.ante
    if delta.round is not None:
        next_state.round = delta.round
    if delta.current_blind is not None:
        next_state.current_blind = delta.current_blind
    if delta.boss_blind is not None:
        next_state.boss_blind = delta.boss_blind
    if delta.target_chips is not None:
        next_state.target_chips = delta.target_chips
    if delta.current_chips is not None:
        next_state.current_chips = delta.current_chips
    if delta.deck_cards_count is not None:
        next_state.deck_cards_count = delta.deck_cards_count
    if delta.full_deck_diff is not None:
        next_state.full_deck = copy.deepcopy(delta.full_deck_diff)
    if delta.discard_pile_diff is not None:
        next_state.discard_pile = copy.deepcopy(delta.discard_pile_diff)
    if delta.played_cards_diff is not None:
        next_state.played_cards_this_round = copy.deepcopy(delta.played_cards_diff)

    if delta.jokers_diff is not None:
        next_state.inventory.jokers = copy.deepcopy(delta.jokers_diff)
    if delta.consumables_diff is not None:
        next_state.inventory.consumables = copy.deepcopy(delta.consumables_diff)
    if delta.joker_slots_diff is not None:
        next_state.inventory.joker_slots = delta.joker_slots_diff
    if delta.consumable_slots_diff is not None:
        next_state.inventory.consumable_slots = delta.consumable_slots_diff

    if delta.hand_diff is not None:
        next_state.hand.cards = copy.deepcopy(delta.hand_diff)
    if delta.hand_max_size_diff is not None:
        next_state.hand.max_size = delta.hand_max_size_diff

    if delta.vouchers_diff is not None:
        next_state.vouchers_redeemed = list(delta.vouchers_diff)

    if delta.hand_levels_diff is not None:
        next_state.hand_levels = dict(delta.hand_levels_diff)

    if delta.hand_play_counts_diff is not None:
        next_state.hand_play_counts = dict(delta.hand_play_counts_diff)

    if delta.unique_planets_diff is not None:
        next_state.unique_planets_used = set(delta.unique_planets_diff)

    return next_state


@dataclass(kw_only=True)
class RunTrajectory:
    """Sequence of (GameState, Action) pairs stored as Keyframes + Deltas."""

    keyframe_interval: int = 20
    keyframes: dict[int, GameState] = field(default_factory=dict)
    deltas: list[GameStateDelta] = field(default_factory=list)

    def record_step(
        self, state: GameState, action: Action | None = None
    ) -> GameStateDelta:
        """Records a new state step into the trajectory."""
        step = len(self.deltas)
        state.step_idx = step

        if step % self.keyframe_interval == 0:
            self.keyframes[step] = state.copy()

        prev_state = self.get_state(step - 1) if step > 0 else state
        delta = diff_states(prev_state, state, action)
        self.deltas.append(delta)
        return delta

    def get_state(self, step: int) -> GameState:
        """Reconstructs state at step N by seeking to nearest keyframe and replaying deltas."""
        if step < 0 or step >= len(self.deltas):
            raise IndexError(
                f"Step {step} out of range (total steps: {len(self.deltas)})"
            )

        kf_step = max(k for k in self.keyframes if k <= step)
        curr = self.keyframes[kf_step].copy()

        for delta in self.deltas[kf_step + 1 : step + 1]:
            curr = apply_delta(curr, delta)

        return curr

    def impute_retroactive_boss_blinds(self):
        """Backward-propagates discovered boss blinds per contiguous ante segment,
        using PRNG simulation from seed when available without clobbering Hieroglyph revisits."""
        if not self.deltas:
            return

        states: list[GameState] = [self.get_state(0)]
        for s in range(1, len(self.deltas)):
            states.append(apply_delta(states[-1].copy(), self.deltas[s]))

        # Partition steps into contiguous ante-visit segments (so Hieroglyph ante -1 keeps separate segments).
        segments: list[tuple[int, int, int]] = []
        seg_start = 0
        for s in range(1, len(states)):
            if states[s].ante != states[s - 1].ante:
                segments.append((seg_start, s - 1, states[seg_start].ante))
                seg_start = s
        segments.append((seg_start, len(states) - 1, states[seg_start].ante))

        # PRNG fallback based on seed for first visits without any observed boss blind.
        prng_bosses: dict[int, BlindName] = {}
        seed = getattr(states[0], "seed", "")
        if seed:
            try:
                from game.engine.rng_sim.api import BalatroRun

                run = BalatroRun(seed)
                max_ante = max((st.ante for st in states), default=1)
                bosses = run.predict_boss_blinds(max_ante=max(8, max_ante))
                for a_idx, b_pred in enumerate(bosses, start=1):
                    prng_bosses[a_idx] = b_pred.blind_name
            except Exception:
                pass

        desired: list[BlindName | None] = [st.boss_blind for st in states]
        visited_antes: set[int] = set()

        for s_start, s_end, ante_num in segments:
            reroll_idx = s_end + 1
            for s in range(s_start, s_end + 1):
                act = self.deltas[s].action
                if act is not None and act.type == ActionType.REROLL_BOSS:
                    reroll_idx = s
                    break

            seg_boss: BlindName | None = None
            # Priority 1: actual Boss Blind round fought in this segment.
            for s in range(s_start, s_end + 1):
                st = states[s]
                if (
                    st.location == Location.BLIND
                    and st.current_blind not in (BlindName.SMALL, BlindName.BIG)
                    and st.boss_blind is not None
                ):
                    seg_boss = st.boss_blind
                    break
            # Priority 2: latest non-None boss observed in Small/Big Blind or Blind Select (before post-boss shop).
            if seg_boss is None:
                for s in range(s_start, s_end + 1):
                    st = states[s]
                    if (
                        st.location in (Location.SELECT, Location.BLIND)
                        and st.current_blind in (BlindName.SMALL, BlindName.BIG)
                        and st.boss_blind is not None
                    ):
                        seg_boss = st.boss_blind
            # Priority 3: any non-None boss in segment, or PRNG fallback on first visit to this ante.
            if seg_boss is None:
                for s in range(s_start, s_end + 1):
                    if states[s].boss_blind is not None:
                        seg_boss = states[s].boss_blind
                        break
            if seg_boss is None and ante_num not in visited_antes:
                seg_boss = prng_bosses.get(ante_num)
            visited_antes.add(ante_num)

            pre_boss: BlindName | None = None
            if reroll_idx <= s_end:
                for s in range(s_start, reroll_idx):
                    if states[s].boss_blind is not None:
                        pre_boss = states[s].boss_blind
                        break
            target_pre = pre_boss if pre_boss is not None else seg_boss

            for s in range(s_start, s_end + 1):
                st = states[s]
                fill_boss = target_pre if s < reroll_idx else seg_boss
                if fill_boss is None:
                    continue
                if st.boss_blind is None or (
                    s < reroll_idx
                    and st.current_blind in (BlindName.SMALL, BlindName.BIG)
                ):
                    desired[s] = fill_boss

        for step, kf in self.keyframes.items():
            if 0 <= step < len(desired) and desired[step] is not None:
                kf.boss_blind = desired[step]

        for s, delta in enumerate(self.deltas):
            target_boss = desired[s]
            if target_boss is None:
                continue
            if s == 0 or delta.ante is not None or desired[s - 1] != target_boss:
                delta.boss_blind = target_boss
                if delta.field_diffs is not None and "boss_blind" in delta.field_diffs:
                    delta.field_diffs["boss_blind"] = target_boss
            else:
                delta.boss_blind = None
                if delta.field_diffs:
                    delta.field_diffs.pop("boss_blind", None)


def step_simulation(
    state: GameState, action: Action
) -> tuple[GameState, GameStateDelta]:
    """Advances state using deterministic rules & trigger dispatcher, returning new state + delta."""
    prev = state.copy()
    on_action(state, action)
    state.step_idx += 1
    state.frame_idx = action.frame_idx
    state.timestamp = action.timestamp
    delta = diff_states(prev, state, action)
    return state, delta
