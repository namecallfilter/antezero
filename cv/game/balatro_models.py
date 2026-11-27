import copy
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

# ---------------------------------------------------------
# Core Game Enums
# ---------------------------------------------------------


class Location(StrEnum):
    BLIND = "blind"
    SHOP = "shop"
    SELECT = "select"
    PACK = "pack"
    # Internal to post_processing.py only; GameState models the cash-out screen as
    # Location.BLIND with round_won=True (do not pass Location.CASH_OUT into GameState).
    CASH_OUT = "cash out"


class PokerHand(StrEnum):
    HIGH_CARD = "High Card"
    PAIR = "Pair"
    TWO_PAIR = "Two Pair"
    THREE_OF_A_KIND = "Three of a Kind"
    FLUSH = "Flush"
    STRAIGHT = "Straight"
    FULL_HOUSE = "Full House"
    FOUR_OF_A_KIND = "Four of a Kind"
    STRAIGHT_FLUSH = "Straight Flush"
    FIVE_OF_A_KIND = "Five of a Kind"
    FLUSH_HOUSE = "Flush House"
    FLUSH_FIVE = "Flush Five"


class Edition(StrEnum):
    NONE = "none"
    FOIL = "foil"
    HOLOGRAPHIC = "holographic"
    POLYCHROME = "polychrome"
    NEGATIVE = "negative"


class Seal(StrEnum):
    NONE = "none"
    RED = "red"
    GOLD = "gold"
    BLUE = "blue"
    PURPLE = "purple"


class Enhancement(StrEnum):
    NONE = "none"
    LUCKY = "lucky"
    MULT = "mult"
    BONUS = "bonus"
    GOLD = "gold"
    STEEL = "steel"
    STONE = "stone"
    WILD = "wild"
    GLASS = "glass"


class Suit(StrEnum):
    NONE = "none"
    SPADES = "spades"
    CLUBS = "clubs"
    HEARTS = "hearts"
    DIAMONDS = "diamonds"


class Rank(StrEnum):
    NONE = "none"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "10"
    JACK = "Jack"
    QUEEN = "Queen"
    KING = "King"
    ACE = "Ace"


class CardType(StrEnum):
    PLAYING_CARD = "playing_card"
    JOKER = "joker"
    TAROT = "tarot"
    PLANET = "planet"
    SPECTRAL = "spectral"


class JokerRarity(StrEnum):
    COMMON = "common"
    UNCOMMON = "uncommon"
    RARE = "rare"
    LEGENDARY = "legendary"


class TarotName(StrEnum):
    CHARIOT = "The Chariot"
    DEATH = "Death"
    DEVIL = "The Devil"
    EMPEROR = "The Emperor"
    EMPRESS = "The Empress"
    FOOL = "The Fool"
    HANGED_MAN = "The Hanged Man"
    HIEROPHANT = "The Hierophant"
    HERMIT = "The Hermit"
    HIGH_PRIESTESS = "The High Priestess"
    JUDGEMENT = "Judgement"
    JUSTICE = "Justice"
    LOVERS = "The Lovers"
    MAGICIAN = "The Magician"
    MOON = "The Moon"
    STAR = "The Star"
    STRENGTH = "Strength"
    SUN = "The Sun"
    TEMPERANCE = "Temperance"
    TOWER = "The Tower"
    WHEEL_OF_FORTUNE = "The Wheel of Fortune"
    WORLD = "The World"


class PlanetName(StrEnum):
    CERES = "Ceres"
    EARTH = "Earth"
    ERIS = "Eris"
    JUPITER = "Jupiter"
    MARS = "Mars"
    MERCURY = "Mercury"
    NEPTUNE = "Neptune"
    PLANET_X = "Planet X"
    PLUTO = "Pluto"
    SATURN = "Saturn"
    URANUS = "Uranus"
    VENUS = "Venus"


class SpectralName(StrEnum):
    ANKH = "Ankh"
    AURA = "Aura"
    BLACK_HOLE = "Black Hole"
    CRYPTID = "Cryptid"
    DEJA_VU = "Deja Vu"
    ECTOPLASM = "Ectoplasm"
    FAMILIAR = "Familiar"
    GRIM = "Grim"
    HEX = "Hex"
    IMMOLATE = "Immolate"
    INCANTATION = "Incantation"
    MEDIUM = "Medium"
    OUIJA = "Ouija"
    SIGIL = "Sigil"
    THE_SOUL = "The Soul"
    TALISMAN = "Talisman"
    TRANCE = "Trance"
    WRAITH = "Wraith"


class VoucherName(StrEnum):
    ANTIMATTER = "Antimatter"
    BLANK = "Blank"
    CLEARANCE_SALE = "Clearance Sale"
    CRYSTAL_BALL = "Crystal Ball"
    DIRECTORS_CUT = "Director's Cut"
    GLOW_UP = "Glow Up"
    GRABBER = "Grabber"
    HIEROGLYPH = "Hieroglyph"
    HONE = "Hone"
    ILLUSION = "Illusion"
    LIQUIDATION = "Liquidation"
    MAGIC_TRICK = "Magic Trick"
    MONEY_TREE = "Money Tree"
    NACHO_TONG = "Nacho Tong"
    OBSERVATORY = "Observatory"
    OMEN_GLOBE = "Omen Globe"
    OVERSTOCK = "Overstock"
    OVERSTOCK_PLUS = "Overstock Plus"
    PAINT_BRUSH = "Paint Brush"
    PALETTE = "Palette"
    PETROGLYPH = "Petroglyph"
    PLANET_MERCHANT = "Planet Merchant"
    PLANET_TYCOON = "Planet Tycoon"
    RECYCLOMANCY = "Recyclomancy"
    REROLL_GLUT = "Reroll Glut"
    REROLL_SURPLUS = "Reroll Surplus"
    RETCON = "Retcon"
    SEED_MONEY = "Seed Money"
    TAROT_MERCHANT = "Tarot Merchant"
    TAROT_TYCOON = "Tarot Tycoon"
    TELESCOPE = "Telescope"
    WASTEFUL = "Wasteful"


class Deck(StrEnum):
    ABANDONED_DECK = "Abandoned Deck"
    ANAGLYPH_DECK = "Anaglyph Deck"
    BLACK_DECK = "Black Deck"
    BLUE_DECK = "Blue Deck"
    # Challenge Decks are out of scope for this engine.
    CHALLENGE_DECK = "Challenge Deck"
    CHECKERED_DECK = "Checkered Deck"
    ERRATIC_DECK = "Erratic Deck"
    GHOST_DECK = "Ghost Deck"
    GREEN_DECK = "Green Deck"
    MAGIC_DECK = "Magic Deck"
    NEBULA_DECK = "Nebula Deck"
    PAINTED_DECK = "Painted Deck"
    PLASMA_DECK = "Plasma Deck"
    RED_DECK = "Red Deck"
    YELLOW_DECK = "Yellow Deck"
    ZODIAC_DECK = "Zodiac Deck"


class SkipTag(StrEnum):
    BOSS_TAG = "Boss Tag"
    BUFFOON_TAG = "Buffoon Tag"
    CHARM_TAG = "Charm Tag"
    COUPON_TAG = "Coupon Tag"
    D6_TAG = "D6 Tag"
    DOUBLE_TAG = "Double Tag"
    ECONOMY_TAG = "Economy Tag"
    ETHEREAL_TAG = "Ethereal Tag"
    FOIL_TAG = "Foil Tag"
    GARBAGE_TAG = "Garbage Tag"
    HANDY_TAG = "Handy Tag"
    HOLOGRAPHIC_TAG = "Holographic Tag"
    INVESTMENT_TAG = "Investment Tag"
    JUGGLE_TAG = "Juggle Tag"
    METEOR_TAG = "Meteor Tag"
    NEGATIVE_TAG = "Negative Tag"
    ORBITAL_TAG = "Orbital Tag"
    POLYCHROME_TAG = "Polychrome Tag"
    RARE_TAG = "Rare Tag"
    SPEED_TAG = "Speed Tag"
    SKIP_TAG = "Speed Tag"
    STANDARD_TAG = "Standard Tag"
    TOP_UP_TAG = "Top-up Tag"
    UNCOMMON_TAG = "Uncommon Tag"
    VOUCHER_TAG = "Voucher Tag"


class BoosterType(StrEnum):
    ARCANA = "Arcana"
    CELESTIAL = "Celestial"
    STANDARD = "Standard"
    BUFFOON = "Buffoon"
    SPECTRAL = "Spectral"


class BoosterSize(StrEnum):
    REGULAR = "Regular"
    JUMBO = "Jumbo"
    MEGA = "Mega"


class BlindType(StrEnum):
    SMALL = "Small Blind"
    BIG = "Big Blind"
    BOSS = "Boss Blind"


class BlindName(StrEnum):
    SMALL = "Small Blind"
    BIG = "Big Blind"
    THE_HOOK = "The Hook"
    THE_OX = "The Ox"
    THE_HOUSE = "The House"
    THE_WALL = "The Wall"
    THE_WHEEL = "The Wheel"
    THE_ARM = "The Arm"
    THE_CLUB = "The Club"
    THE_FISH = "The Fish"
    THE_PSYCHIC = "The Psychic"
    THE_GOAD = "The Goad"
    THE_WATER = "The Water"
    THE_WINDOW = "The Window"
    THE_MANACLE = "The Manacle"
    THE_EYE = "The Eye"
    THE_MOUTH = "The Mouth"
    THE_PLANT = "The Plant"
    THE_SERPENT = "The Serpent"
    THE_PILLAR = "The Pillar"
    THE_NEEDLE = "The Needle"
    THE_HEAD = "The Head"
    THE_TOOTH = "The Tooth"
    THE_FLINT = "The Flint"
    THE_MARK = "The Mark"
    AMBER_ACORN = "Amber Acorn"
    VERDANT_LEAF = "Verdant Leaf"
    VIOLET_VESSEL = "Violet Vessel"
    CRIMSON_HEART = "Crimson Heart"
    CERULEAN_BELL = "Cerulean Bell"


class Stake(StrEnum):
    WHITE = "White Stake"
    RED = "Red Stake"
    GREEN = "Green Stake"
    BLACK = "Black Stake"
    BLUE = "Blue Stake"
    PURPLE = "Purple Stake"
    ORANGE = "Orange Stake"
    GOLD = "Gold Stake"


class ActionType(StrEnum):
    BUY_AND_USE = "Buy and Use"
    BUY_CARD = "Buy Card"
    CASH_OUT = "Cash Out"
    DISCARD = "Discard"
    NEXT_ROUND = "Next Round"
    OPEN_PACK = "Open Pack"
    OPEN_OPTIONS = "Open Options Menu"
    PLAY_HAND = "Play Hand"
    REDEEM_VOUCHER = "Redeem Voucher"
    REROLL_BOSS = "Reroll Boss"
    REROLL_SHOP = "Reroll Shop"
    RUN_INFO = "Open Run Info Menu"
    SELECT_BLIND = "Select Blind"
    SELL_CARD = "Sell Card"
    SKIP_BLIND = "Skip Blind"
    SKIP_PACK = "Skip Pack"
    SORT_RANK = "Sort by Rank"
    SORT_SUIT = "Sort by Suit"
    USE_CONSUMABLE = "Use Consumable"
    USE_IN_PACK = "Use In Pack"
    PLAY_GAME = "Play Game"
    END_ROUND = "End Round"
    SETUP_TEST_SUITE = "Setup Test Suite"


@dataclass(kw_only=True)
class Action:
    type: ActionType
    frame_idx: int = 0
    timestamp: float = 0.0
    target_idx: int | None = None
    selected_cards: list["PlayingCard"] = field(default_factory=list)
    joker_order: list[int] = field(default_factory=list)
    hand_order: list[int] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------
# Card Models & Dynamic Pricing Hierarchy
# ---------------------------------------------------------


@dataclass(kw_only=True)
class Card:
    type: CardType
    edition: Edition = Edition.NONE
    extra_value: int = 0  # Dynamic additions (Gift Card, Egg, etc.)
    couponed: bool = False
    rental: bool = False

    @property
    def base_cost(self) -> int:
        return 0

    def _get_pre_coupon_cost(
        self,
        discount_percent: int = 0,
        inflation: int = 0,
        astronomer: bool = False,
    ) -> int:
        """Computes Card buy cost before Coupon Tag zeroing (matching Card:set_cost)."""
        edition_extra = {
            Edition.NONE: 0,
            Edition.FOIL: 2,
            Edition.HOLOGRAPHIC: 3,
            Edition.POLYCHROME: 5,
            Edition.NEGATIVE: 5,
        }.get(self.edition, 0)

        extra_cost = inflation + edition_extra
        raw = (self.base_cost + extra_cost + 0.5) * (100 - discount_percent) / 100.0
        cost = max(1, math.floor(raw))
        if astronomer and self.type == CardType.PLANET:
            cost = 0
        if getattr(self, "rental", False):
            cost = 1
        return cost

    def get_cost(
        self,
        discount_percent: int = 0,
        inflation: int = 0,
        astronomer: bool = False,
    ) -> int:
        """Computes buy cost matching Balatro's Card:set_cost()."""
        if getattr(self, "couponed", False):
            return 0
        return self._get_pre_coupon_cost(
            discount_percent=discount_percent,
            inflation=inflation,
            astronomer=astronomer,
        )

    def get_sell_value(
        self,
        discount_percent: int = 0,
        inflation: int = 0,
        astronomer: bool = False,
    ) -> int:
        """Sell value is half the pre-coupon cost plus any accrued extra_value (Gift Card)."""
        cost = self._get_pre_coupon_cost(
            discount_percent=discount_percent,
            inflation=inflation,
            astronomer=astronomer,
        )
        return max(1, math.floor(cost / 2)) + self.extra_value

    @property
    def sell_value(self) -> int:
        """Default sell value at baseline (0% discount, 0 inflation)."""
        return self.get_sell_value()


@dataclass(kw_only=True)
class PlayingCard(Card):
    rank: Rank = Rank.NONE  # None for Stone cards
    suit: Suit = Suit.NONE  # None for Stone cards
    enhancement: Enhancement = Enhancement.NONE
    seal: Seal = Seal.NONE
    type: CardType = CardType.PLAYING_CARD
    bonus_chips: int = 0  # Permanent chips gained (e.g. from Hiker)
    id: int = field(default=0, compare=False)  # Balatro sort_id (G.playing_card)
    debuffed: bool = field(default=False, compare=False)
    destroyed: bool = field(default=False, compare=False)

    @property
    def sort_id(self) -> int:
        return self.id

    @sort_id.setter
    def sort_id(self, value: int) -> None:
        self.id = value

    @property
    def card_key(self) -> str:
        """Balatro `G.P_CARDS` key (e.g., `'C_2'`, `'D_T'`, `'H_Q'`, `'S_A'`)."""
        suit_code = {
            Suit.SPADES: "S",
            Suit.HEARTS: "H",
            Suit.CLUBS: "C",
            Suit.DIAMONDS: "D",
        }.get(self.suit, "S")
        rank_code = {
            Rank.TEN: "T",
            Rank.JACK: "J",
            Rank.QUEEN: "Q",
            Rank.KING: "K",
            Rank.ACE: "A",
            Rank.NONE: "A",
        }.get(self.rank, self.rank.value)
        return f"{suit_code}_{rank_code}"

    def is_suit(self, suit: Suit | str) -> bool:
        """Check whether this card matches `suit` (Stone cards have no suit; Wild cards match all suits)."""
        if self.enhancement == Enhancement.STONE or self.suit == Suit.NONE:
            return False
        if self.enhancement == Enhancement.WILD:
            return True
        target = suit if isinstance(suit, Suit) else Suit(str(suit).strip().lower())
        return self.suit == target

    def is_rank(self, rank: Rank | str) -> bool:
        """Check whether this card matches `rank` (Stone cards have no rank)."""
        if self.enhancement == Enhancement.STONE or self.rank == Rank.NONE:
            return False
        target = rank if isinstance(rank, Rank) else Rank(str(rank).strip())
        return self.rank == target

    @property
    def base_cost(self) -> int:
        return 2  # standard shop cost when Magic Trick is active

    @property
    def is_face(self) -> bool:
        """Pareidolia makes all cards face cards; Stone cards are not face cards."""
        if self.enhancement == Enhancement.STONE or self.rank == Rank.NONE:
            return False
        return self.rank in (Rank.JACK, Rank.QUEEN, Rank.KING)

    @property
    def base_nominal_chips(self) -> int:
        """Base chip value before enhancements or jokers."""
        if self.enhancement == Enhancement.STONE:
            return 50
        if self.rank == Rank.NONE:
            return 0
        match self.rank:
            case Rank.ACE:
                return 11
            case Rank.KING | Rank.QUEEN | Rank.JACK | Rank.TEN:
                return 10
            case _:
                return int(self.rank.value)

    @property
    def chips(self) -> int:
        """Total base chip yield when scored (including Bonus enhancement and Hiker)."""
        extra = 30 if self.enhancement == Enhancement.BONUS else 0
        return self.base_nominal_chips + extra + self.bonus_chips


@dataclass
class CardBounds:
    """Represents candidate constraints and marginal distributions for partially-observed or face-down cards."""

    is_face_down: bool = True
    possible_ranks: set[Rank] = field(default_factory=lambda: set(Rank) - {Rank.NONE})
    possible_suits: set[Suit] = field(default_factory=lambda: set(Suit) - {Suit.NONE})
    possible_enhancements: set[Enhancement] = field(
        default_factory=lambda: set(Enhancement)
    )
    possible_editions: set[Edition] = field(default_factory=lambda: set(Edition))
    possible_seals: set[Seal] = field(default_factory=lambda: set(Seal))
    candidate_card_ids: set[int] = field(default_factory=set)
    candidate_cards: list["PlayingCard"] = field(default_factory=list)

    @property
    def is_resolved(self) -> bool:
        return len(self.candidate_card_ids) == 1

    def get_resolved_card(self) -> "PlayingCard | None":
        if self.is_resolved and self.candidate_cards:
            return self.candidate_cards[0]
        return None

    def set_candidates(self, cards: Sequence["PlayingCard"]) -> None:
        """Sets active candidate PlayingCards and synchronizes possible rank/suit/enhancement/seal sets."""
        self.candidate_cards = list(cards)
        self.candidate_card_ids = {c.id for c in cards}
        if cards:
            c_ranks = {c.rank for c in cards}
            if self.possible_ranks:
                inter_r = self.possible_ranks & c_ranks
                if inter_r:
                    self.possible_ranks = inter_r
            else:
                self.possible_ranks = c_ranks

            c_suits = {c.suit for c in cards}
            if self.possible_suits:
                inter_s = self.possible_suits & c_suits
                if inter_s:
                    self.possible_suits = inter_s
            else:
                self.possible_suits = c_suits

            c_enh = {c.enhancement for c in cards}
            if self.possible_enhancements:
                inter_e = self.possible_enhancements & c_enh
                if inter_e:
                    self.possible_enhancements = inter_e
            else:
                self.possible_enhancements = c_enh

            c_ed = {c.edition for c in cards}
            if self.possible_editions:
                inter_ed = self.possible_editions & c_ed
                if inter_ed:
                    self.possible_editions = inter_ed
            else:
                self.possible_editions = c_ed

            c_seals = {c.seal for c in cards}
            if self.possible_seals:
                inter_sl = self.possible_seals & c_seals
                if inter_sl:
                    self.possible_seals = inter_sl
            else:
                self.possible_seals = c_seals

    def get_rank_probabilities(
        self, candidate_pool: Sequence["PlayingCard"] | None = None
    ) -> dict[Rank, float]:
        """Returns marginal probability P(Rank) for each of the 13 ranks."""
        ranks = [r for r in Rank if r != Rank.NONE]
        pool = candidate_pool if candidate_pool is not None else self.candidate_cards
        if pool:
            valid_pool = [
                c
                for c in pool
                if not self.candidate_card_ids or c.id in self.candidate_card_ids
            ]
            if valid_pool:
                total = len(valid_pool)
                counts = {r: 0 for r in ranks}
                for c in valid_pool:
                    if c.rank in counts:
                        counts[c.rank] += 1
                return {r: counts[r] / total for r in ranks}
        total = len(self.possible_ranks)
        if total == 0:
            return {r: 0.0 for r in ranks}
        return {r: (1.0 / total if r in self.possible_ranks else 0.0) for r in ranks}

    def get_suit_probabilities(
        self, candidate_pool: Sequence["PlayingCard"] | None = None
    ) -> dict[Suit, float]:
        """Returns marginal probability P(Suit) for each of the 4 suits."""
        suits = [s for s in Suit if s != Suit.NONE]
        pool = candidate_pool if candidate_pool is not None else self.candidate_cards
        if pool:
            valid_pool = [
                c
                for c in pool
                if not self.candidate_card_ids or c.id in self.candidate_card_ids
            ]
            if valid_pool:
                total = len(valid_pool)
                counts = {s: 0 for s in suits}
                for c in valid_pool:
                    if c.suit in counts:
                        counts[c.suit] += 1
                return {s: counts[s] / total for s in suits}
        total = len(self.possible_suits)
        if total == 0:
            return {s: 0.0 for s in suits}
        return {s: (1.0 / total if s in self.possible_suits else 0.0) for s in suits}

    def get_seal_probabilities(
        self, candidate_pool: Sequence["PlayingCard"] | None = None
    ) -> dict[Seal, float]:
        """Returns marginal probability P(Seal) for each of the 5 seals."""
        seals = list(Seal)
        pool = candidate_pool if candidate_pool is not None else self.candidate_cards
        if pool:
            valid_pool = [
                c
                for c in pool
                if not self.candidate_card_ids or c.id in self.candidate_card_ids
            ]
            if valid_pool:
                total = len(valid_pool)
                counts = {s: 0 for s in seals}
                for c in valid_pool:
                    if c.seal in counts:
                        counts[c.seal] += 1
                return {s: counts[s] / total for s in seals}
        total = len(self.possible_seals)
        if total == 0:
            return {s: 0.0 for s in seals}
        return {s: (1.0 / total if s in self.possible_seals else 0.0) for s in seals}

    def get_rank_vector(
        self, candidate_pool: Sequence["PlayingCard"] | None = None
    ) -> list[float]:
        """Fixed 13-dim probability vector in order: 2, 3, 4, 5, 6, 7, 8, 9, 10, Jack, Queen, King, Ace."""
        ordered_ranks = [
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
        probs = self.get_rank_probabilities(candidate_pool)
        return [probs.get(r, 0.0) for r in ordered_ranks]

    def get_suit_vector(
        self, candidate_pool: Sequence["PlayingCard"] | None = None
    ) -> list[float]:
        """Fixed 4-dim probability vector in order: Spades, Hearts, Clubs, Diamonds."""
        ordered_suits = [Suit.SPADES, Suit.HEARTS, Suit.CLUBS, Suit.DIAMONDS]
        probs = self.get_suit_probabilities(candidate_pool)
        return [probs.get(s, 0.0) for s in ordered_suits]

    def get_seal_vector(
        self, candidate_pool: Sequence["PlayingCard"] | None = None
    ) -> list[float]:
        """Fixed 5-dim probability vector in order: None, Red, Gold, Blue, Purple."""
        ordered_seals = [Seal.NONE, Seal.RED, Seal.GOLD, Seal.BLUE, Seal.PURPLE]
        probs = self.get_seal_probabilities(candidate_pool)
        return [probs.get(s, 0.0) for s in ordered_seals]


@dataclass(kw_only=True)
class HandCard:
    card: PlayingCard
    selected: bool = False
    facing_back: bool = False
    forced_selection: bool = False
    bounds: CardBounds | None = None
    slot_uid: int = 0
    _real_card: PlayingCard | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.facing_back:
            if self._real_card is None and (
                self.card.rank != Rank.NONE
                or self.card.suit != Suit.NONE
                or self.card.id != 0
            ):
                self._real_card = copy.deepcopy(self.card)
            if self.bounds is None:
                self.bounds = CardBounds(is_face_down=True)
            if not self.bounds.is_resolved:
                self.card = copy.deepcopy(self.card)
                self.card.rank = Rank.NONE
                self.card.suit = Suit.NONE
                self.card.id = 0

    def resolve_face_down(
        self,
        resolved_card: PlayingCard,
        lift_facing_back: bool = False,
        preserve_id: bool = False,
    ) -> None:
        """Resolves a face-down card to an exact PlayingCard, lifting hidden-state masks."""
        if lift_facing_back:
            self.facing_back = False
            if self.bounds:
                self.bounds.is_face_down = False
        preserved_id = self.card.id if preserve_id else resolved_card.id
        self._real_card = copy.deepcopy(resolved_card)
        self._real_card.id = preserved_id
        self.card.rank = resolved_card.rank
        self.card.suit = resolved_card.suit
        self.card.enhancement = resolved_card.enhancement
        self.card.edition = resolved_card.edition
        self.card.seal = resolved_card.seal
        self.card.id = preserved_id
        self.card.bonus_chips = resolved_card.bonus_chips
        if self.bounds is None:
            self.bounds = CardBounds(is_face_down=self.facing_back)
        self.bounds.is_face_down = self.facing_back
        self.bounds.possible_ranks = {resolved_card.rank}
        self.bounds.possible_suits = {resolved_card.suit}
        self.bounds.possible_enhancements = {resolved_card.enhancement}
        self.bounds.possible_editions = {resolved_card.edition}
        self.bounds.possible_seals = {resolved_card.seal}
        self.bounds.candidate_card_ids = {resolved_card.id}
        self.bounds.candidate_cards = [resolved_card]

    def reveal_temporarily(self, observed_card: PlayingCard | None = None) -> None:
        """Called when a face-down card is temporarily flipped face-up (e.g. by a Tarot or Spectral card).

        The player observes the true rank, suit, seal, edition, and enhancement during the flip animation.
        The card flips back face-down (facing_back remains True), but its bounds and observable card attributes
        are resolved to the exact card observed.
        """
        source = observed_card or self._real_card or self.card
        self._real_card = copy.deepcopy(source)
        if self.bounds is None:
            self.bounds = CardBounds(is_face_down=True)
        self.bounds.is_face_down = True
        self.bounds.possible_ranks = {source.rank}
        self.bounds.possible_suits = {source.suit}
        self.bounds.possible_enhancements = {source.enhancement}
        self.bounds.possible_editions = {source.edition}
        self.bounds.possible_seals = {source.seal}
        source_id = source.id if source.id else getattr(source, "sort_id", 0)
        self.bounds.candidate_card_ids = {source_id} if source_id else {id(source)}
        self.bounds.candidate_cards = [copy.deepcopy(source)]
        self.card.rank = source.rank
        self.card.suit = source.suit
        self.card.enhancement = source.enhancement
        self.card.edition = source.edition
        self.card.seal = source.seal
        self.card.id = source.id
        self.card.bonus_chips = source.bonus_chips
        self.facing_back = True

    def blur_candidates(self, candidate_cards: Sequence[PlayingCard]) -> None:
        """Called when positional certainty is lost (e.g. adjacent face-down cards drawn or clustered).

        Expands candidate bounds to the union of indistinguishable cards, masking card rank/suit/id
        if not uniquely resolved.
        """
        if self.bounds is None:
            self.bounds = CardBounds(is_face_down=True)
        cards_list = list(candidate_cards)
        self.bounds.candidate_cards = cards_list
        self.bounds.candidate_card_ids = {c.id for c in cards_list}
        self.bounds.possible_ranks = {c.rank for c in cards_list if c.rank != Rank.NONE}
        self.bounds.possible_suits = {c.suit for c in cards_list if c.suit != Suit.NONE}
        self.bounds.possible_enhancements = {c.enhancement for c in cards_list}
        self.bounds.possible_editions = {c.edition for c in cards_list}
        self.bounds.possible_seals = {c.seal for c in cards_list}
        self.bounds.is_face_down = True
        if not self.bounds.is_resolved:
            self.card.rank = Rank.NONE
            self.card.suit = Suit.NONE
            self.card.id = 0

    def get_rank_vector(self) -> list[float]:
        if not self.facing_back and self.card.rank != Rank.NONE:
            ordered_ranks = [
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
            return [1.0 if r == self.card.rank else 0.0 for r in ordered_ranks]
        if self.bounds:
            return self.bounds.get_rank_vector()
        return [0.0] * 13

    def get_suit_vector(self) -> list[float]:
        if not self.facing_back and self.card.suit != Suit.NONE:
            ordered_suits = [Suit.SPADES, Suit.HEARTS, Suit.CLUBS, Suit.DIAMONDS]
            return [1.0 if s == self.card.suit else 0.0 for s in ordered_suits]
        if self.bounds:
            return self.bounds.get_suit_vector()
        return [0.0] * 4

    def get_seal_vector(self) -> list[float]:
        if not self.facing_back:
            ordered_seals = [Seal.NONE, Seal.RED, Seal.GOLD, Seal.BLUE, Seal.PURPLE]
            return [1.0 if s == self.card.seal else 0.0 for s in ordered_seals]
        if self.bounds:
            return self.bounds.get_seal_vector()
        return [0.0] * 5


@dataclass(kw_only=True)
class Hand:
    cards: list[HandCard] = field(default_factory=list)
    max_size: int = 8

    @property
    def selected_cards(self) -> list[PlayingCard]:
        return [c.card for c in self.cards if c.selected]

    @property
    def unselected_cards(self) -> list[PlayingCard]:
        return [c.card for c in self.cards if not c.selected]

    def get_scoring_cards(
        self, jokers: Sequence["Joker"] | None = None
    ) -> list[PlayingCard]:
        """Returns only the cards in the current selection that actively score."""
        try:
            from game.engine.poker_hands import get_scoring_cards
        except ImportError:
            from engine.poker_hands import get_scoring_cards
        return get_scoring_cards(self.selected_cards, jokers)

    @property
    def can_play_or_discard(self) -> bool:
        """Balatro allows playing or discarding 1 to 5 selected cards."""
        return 1 <= len(self.selected_cards) <= 5

    def select(self, index: int) -> None:
        if 0 <= index < len(self.cards):
            hc = self.cards[index]
            if hc.selected and hc.forced_selection:
                return
            # Only allow selecting if fewer than 5 are already selected
            if not hc.selected and len(self.selected_cards) >= 5:
                return
            hc.selected = not hc.selected

    def clear_selection(self) -> None:
        for c in self.cards:
            if not c.forced_selection:
                c.selected = False


@dataclass(kw_only=True)
class Consumable(Card):
    @property
    def base_cost(self) -> int:
        return 3


@dataclass(kw_only=True)
class Tarot(Consumable):
    name: TarotName
    type: CardType = CardType.TAROT


@dataclass(kw_only=True)
class Planet(Consumable):
    name: PlanetName
    type: CardType = CardType.PLANET


@dataclass(kw_only=True)
class Spectral(Consumable):
    name: SpectralName
    type: CardType = CardType.SPECTRAL

    @property
    def base_cost(self) -> int:
        return 4


@dataclass(frozen=True, slots=True)
class JokerInfo:
    """Static metadata for a Joker."""

    name: "JokerName"
    rarity: JokerRarity
    base_cost: int
    base_sell_value: int
    blueprint_compat: bool = True


@dataclass(kw_only=True)
class Joker(Card):
    """Runtime instance of a Joker in a run."""

    name: "JokerName" = field(default=None)  # type: ignore
    type: CardType = CardType.JOKER
    edition: Edition = Edition.NONE
    eternal: bool = False
    perishable: bool = False
    perish_tally: int = 5
    rental: bool = False
    debuffed: bool = False
    facing_back: bool = False
    extra_value: int = 0  # e.g., Gift Card bonuses
    state: dict = field(default_factory=dict)  # e.g. {"chips": 45} for Runner
    id: int = field(default=0, compare=False)  # Balatro sort_id

    @property
    def info(self) -> JokerInfo:
        return JOKER_REGISTRY[self.name]

    @property
    def base_cost(self) -> int:
        return self.info.base_cost

    @property
    def blueprint_compat(self) -> bool:
        return self.info.blueprint_compat


@dataclass(kw_only=True)
class Voucher:
    name: VoucherName
    price: int = 10

    def get_cost(self, discount_percent: int = 0, inflation: int = 0) -> int:
        """Computes voucher buy cost matching Balatro's Card:set_cost()."""
        raw = (self.price + inflation + 0.5) * (100 - discount_percent) / 100.0
        return max(1, math.floor(raw))


@dataclass(kw_only=True)
class BoosterPack:
    booster_type: BoosterType
    size: BoosterSize
    cost: int
    choose: int
    extra: int  # total options shown (e.g. choose 1 of 3 -> extra=3)
    couponed: bool = False

    def get_cost(
        self,
        discount_percent: int = 0,
        inflation: int = 0,
        astronomer: bool = False,
    ) -> int:
        """Computes booster pack buy cost matching Balatro's Card:set_cost()."""
        if self.couponed:
            return 0
        if astronomer and self.booster_type == BoosterType.CELESTIAL:
            return 0
        raw = (self.cost + inflation + 0.5) * (100 - discount_percent) / 100.0
        return max(1, math.floor(raw))


@dataclass(kw_only=True)
class PokerHandLevel:
    hand: PokerHand
    level: int = 1
    base_chips: int
    base_mult: int
    l_chips: int
    l_mult: int

    @property
    def chips(self) -> int:
        return self.base_chips + (self.level - 1) * self.l_chips

    @property
    def mult(self) -> int:
        return self.base_mult + (self.level - 1) * self.l_mult


POKER_HAND_DEFAULTS: dict[PokerHand, PokerHandLevel] = {
    PokerHand.HIGH_CARD: PokerHandLevel(
        hand=PokerHand.HIGH_CARD, base_chips=5, base_mult=1, l_chips=10, l_mult=1
    ),
    PokerHand.PAIR: PokerHandLevel(
        hand=PokerHand.PAIR, base_chips=10, base_mult=2, l_chips=15, l_mult=1
    ),
    PokerHand.TWO_PAIR: PokerHandLevel(
        hand=PokerHand.TWO_PAIR, base_chips=20, base_mult=2, l_chips=20, l_mult=1
    ),
    PokerHand.THREE_OF_A_KIND: PokerHandLevel(
        hand=PokerHand.THREE_OF_A_KIND, base_chips=30, base_mult=3, l_chips=20, l_mult=2
    ),
    PokerHand.STRAIGHT: PokerHandLevel(
        hand=PokerHand.STRAIGHT, base_chips=30, base_mult=4, l_chips=30, l_mult=3
    ),
    PokerHand.FLUSH: PokerHandLevel(
        hand=PokerHand.FLUSH, base_chips=35, base_mult=4, l_chips=15, l_mult=2
    ),
    PokerHand.FULL_HOUSE: PokerHandLevel(
        hand=PokerHand.FULL_HOUSE, base_chips=40, base_mult=4, l_chips=25, l_mult=2
    ),
    PokerHand.FOUR_OF_A_KIND: PokerHandLevel(
        hand=PokerHand.FOUR_OF_A_KIND, base_chips=60, base_mult=7, l_chips=30, l_mult=3
    ),
    PokerHand.STRAIGHT_FLUSH: PokerHandLevel(
        hand=PokerHand.STRAIGHT_FLUSH, base_chips=100, base_mult=8, l_chips=40, l_mult=4
    ),
    PokerHand.FIVE_OF_A_KIND: PokerHandLevel(
        hand=PokerHand.FIVE_OF_A_KIND,
        base_chips=120,
        base_mult=12,
        l_chips=35,
        l_mult=3,
    ),
    PokerHand.FLUSH_HOUSE: PokerHandLevel(
        hand=PokerHand.FLUSH_HOUSE, base_chips=140, base_mult=14, l_chips=40, l_mult=4
    ),
    PokerHand.FLUSH_FIVE: PokerHandLevel(
        hand=PokerHand.FLUSH_FIVE, base_chips=160, base_mult=16, l_chips=50, l_mult=3
    ),
}


@dataclass(kw_only=True)
class Inventory:
    jokers: list[Joker] = field(default_factory=list)
    consumables: list[Consumable] = field(default_factory=list)
    joker_slots: int = 5
    consumable_slots: int = 2

    @property
    def max_jokers(self) -> int:
        negative_count = sum(1 for j in self.jokers if j.edition == Edition.NEGATIVE)
        return self.joker_slots + negative_count

    @property
    def max_consumables(self) -> int:
        negative_count = sum(
            1 for c in self.consumables if c.edition == Edition.NEGATIVE
        )
        return self.consumable_slots + negative_count

    @property
    def can_add_joker(self) -> bool:
        return len(self.jokers) < self.max_jokers

    @property
    def can_add_consumable(self) -> bool:
        return len(self.consumables) < self.max_consumables


# @dataclass(frozen=True, slots=True)
# class JokerFunctions:
#     """
#     Joker functionality; each field is a potential in-game trigger.
#     2 types of possible functions:
#         1) Directly update a game value triggered by a certain action
#            OR event  (action ~ player input; event ~ glass breaking);
#            i.e., playing a hand increases green joker. Closed-loop.

#         2) Some value is updated, but we need to queue a CV query to
#            determine what the result is (usually due to randomness).
#            i.e., cartomancer/perkeo creates a consumable card -> CV
#            query to re-document consumables
#            NOTE: If this is triggered by an action, that action was
#                  made without the updated value already existing, so
#                  we can queue the CV re-check for the next action.
#                  For instance: Clicking Select Blind with Cartomancer
#                  is done without the resultant tarot card in
#                  inventory. However, we don't care about what that
#                  tarot card is until the NEXT action, so add "re-index
#                  consumables" to the list of things to scan at the
#                  time of next action.

#                  If the value is triggered by a random event (i.e.
#                  glass breaking), same thing. Queue CV check for the
#                  *next* action.
#     We don't care about jokers that statically update the score, like
#     The Trio, since they don't modify the game state more than playing
#     a hand / performing any action without them would.
#         ex: With/without The Trio, when we receive the "Play Hand"
#         action, we need to:
#             - Check how many hands are remaining
#             - Check the current blind score
#             - Check how much money we have
#             ...etc (nothing changes by having The Trio)


#     """

#     discard: list[
#         Callable
#     ]  # castle, ramen, trading card, green joker, yorick, burnt joker, hit the road,

#     # ride the bus, lucky cat, seltzer, midas mask, obelisk, vagabond, vampire, seance, square joker,
#     # superposition, green joker, hiker, sixth sense, DNA, runner, matador, space joker, 8 ball, loyalty card,
#     play_hand: list[Callable]

#     next_round: list[Callable]  # perkeo

#     # madness, riff-raff, burglar, cartomancer, certificate, ceremonial dagger, marble joker
#     select_blind: list[Callable]
#     use_planet: list[Callable]  # satellite, constellation,
#     sell_card: list[Callable]  # campfire, cola, luchador, invisible joker,
#     end_round: list[Callable]  # gift card, egg, hit the road
#     end_ante: list[Callable]  # campfire, rocket
#     add_card: list[Callable]  # hologram
#     glass_broken: list[Callable]  # glass joker
#     reroll: list[Callable]  # flashcard
#     open_pack: list[Callable]  # hallucation
#     skip_pack: list[Callable]  # red card
#     remove_card: list[Callable]  # canio
#     deck_change: list[Callable]  # stone joker, steel joker


# ---------------------------------------------------------
# Static Data Tables & Registries (Keep at bottom)
# ---------------------------------------------------------


class JokerName(StrEnum):
    JOKER = "Joker"
    GREEDY_JOKER = "Greedy Joker"
    LUSTY_JOKER = "Lusty Joker"
    WRATHFUL_JOKER = "Wrathful Joker"
    GLUTTONOUS_JOKER = "Gluttonous Joker"
    JOLLY_JOKER = "Jolly Joker"
    ZANY_JOKER = "Zany Joker"
    MAD_JOKER = "Mad Joker"
    CRAZY_JOKER = "Crazy Joker"
    DROLL_JOKER = "Droll Joker"
    SLY_JOKER = "Sly Joker"
    WILY_JOKER = "Wily Joker"
    CLEVER_JOKER = "Clever Joker"
    DEVIOUS_JOKER = "Devious Joker"
    CRAFTY_JOKER = "Crafty Joker"
    HALF_JOKER = "Half Joker"
    JOKER_STENCIL = "Joker Stencil"
    FOUR_FINGERS = "Four Fingers"
    MIME = "Mime"
    CREDIT_CARD = "Credit Card"
    CEREMONIAL_DAGGER = "Ceremonial Dagger"
    BANNER = "Banner"
    MYSTIC_SUMMIT = "Mystic Summit"
    MARBLE_JOKER = "Marble Joker"
    LOYALTY_CARD = "Loyalty Card"
    EIGHT_BALL = "8 Ball"
    MISPRINT = "Misprint"
    DUSK = "Dusk"
    RAISED_FIST = "Raised Fist"
    CHAOS_THE_CLOWN = "Chaos the Clown"
    FIBONACCI = "Fibonacci"
    STEEL_JOKER = "Steel Joker"
    SCARY_FACE = "Scary Face"
    ABSTRACT_JOKER = "Abstract Joker"
    DELAYED_GRATIFICATION = "Delayed Gratification"
    HACK = "Hack"
    PAREIDOLIA = "Pareidolia"
    GROS_MICHEL = "Gros Michel"
    EVEN_STEVEN = "Even Steven"
    ODD_TODD = "Odd Todd"
    SCHOLAR = "Scholar"
    BUSINESS_CARD = "Business Card"
    SUPERNOVA = "Supernova"
    RIDE_THE_BUS = "Ride the Bus"
    SPACE_JOKER = "Space Joker"
    EGG = "Egg"
    BURGLAR = "Burglar"
    BLACKBOARD = "Blackboard"
    RUNNER = "Runner"
    ICE_CREAM = "Ice Cream"
    DNA = "DNA"
    SPLASH = "Splash"
    BLUE_JOKER = "Blue Joker"
    SIXTH_SENSE = "Sixth Sense"
    CONSTELLATION = "Constellation"
    HIKER = "Hiker"
    FACELESS_JOKER = "Faceless Joker"
    GREEN_JOKER = "Green Joker"
    SUPERPOSITION = "Superposition"
    TO_DO_LIST = "To Do List"
    CAVENDISH = "Cavendish"
    CARD_SHARP = "Card Sharp"
    RED_CARD = "Red Card"
    MADNESS = "Madness"
    SQUARE_JOKER = "Square Joker"
    SEANCE = "Séance"
    RIFF_RAFF = "Riff-Raff"
    VAMPIRE = "Vampire"
    SHORTCUT = "Shortcut"
    HOLOGRAM = "Hologram"
    VAGABOND = "Vagabond"
    BARON = "Baron"
    CLOUD_9 = "Cloud 9"
    ROCKET = "Rocket"
    OBELISK = "Obelisk"
    MIDAS_MASK = "Midas Mask"
    LUCHADOR = "Luchador"
    PHOTOGRAPH = "Photograph"
    GIFT_CARD = "Gift Card"
    TURTLE_BEAN = "Turtle Bean"
    EROSION = "Erosion"
    RESERVED_PARKING = "Reserved Parking"
    MAIL_IN_REBATE = "Mail-In Rebate"
    TO_THE_MOON = "To the Moon"
    HALLUCINATION = "Hallucination"
    FORTUNE_TELLER = "Fortune Teller"
    JUGGLER = "Juggler"
    DRUNKARD = "Drunkard"
    STONE_JOKER = "Stone Joker"
    GOLDEN_JOKER = "Golden Joker"
    LUCKY_CAT = "Lucky Cat"
    BASEBALL_CARD = "Baseball Card"
    BULL = "Bull"
    DIET_COLA = "Diet Cola"
    TRADING_CARD = "Trading Card"
    FLASH_CARD = "Flash Card"
    POPCORN = "Popcorn"
    SPARE_TROUSERS = "Spare Trousers"
    ANCIENT_JOKER = "Ancient Joker"
    RAMEN = "Ramen"
    WALKIE_TALKIE = "Walkie Talkie"
    SELTZER = "Seltzer"
    CASTLE = "Castle"
    SMILEY_FACE = "Smiley Face"
    CAMPFIRE = "Campfire"
    GOLDEN_TICKET = "Golden Ticket"
    MR_BONES = "Mr. Bones"
    ACROBAT = "Acrobat"
    SOCK_AND_BUSKIN = "Sock and Buskin"
    SWASHBUCKLER = "Swashbuckler"
    TROUBADOUR = "Troubadour"
    CERTIFICATE = "Certificate"
    SMEARED_JOKER = "Smeared Joker"
    THROWBACK = "Throwback"
    HANGING_CHAD = "Hanging Chad"
    ROUGH_GEM = "Rough Gem"
    BLOODSTONE = "Bloodstone"
    ARROWHEAD = "Arrowhead"
    ONYX_AGATE = "Onyx Agate"
    GLASS_JOKER = "Glass Joker"
    SHOWMAN = "Showman"
    FLOWER_POT = "Flower Pot"
    BLUEPRINT = "Blueprint"
    WEE_JOKER = "Wee Joker"
    MERRY_ANDY = "Merry Andy"
    OOPS_ALL_6S = "Oops! All 6s"
    THE_IDOL = "The Idol"
    SEEING_DOUBLE = "Seeing Double"
    MATADOR = "Matador"
    HIT_THE_ROAD = "Hit the Road"
    THE_DUO = "The Duo"
    THE_TRIO = "The Trio"
    THE_FAMILY = "The Family"
    THE_ORDER = "The Order"
    THE_TRIBE = "The Tribe"
    STUNTMAN = "Stuntman"
    INVISIBLE_JOKER = "Invisible Joker"
    BRAINSTORM = "Brainstorm"
    SATELLITE = "Satellite"
    SHOOT_THE_MOON = "Shoot the Moon"
    DRIVER = "Driver"
    CARTOMANCER = "Cartomancer"
    ASTRONOMER = "Astronomer"
    BURNT_JOKER = "Burnt Joker"
    BOOTSTRAPS = "Bootstraps"
    CANIO = "Canio"
    TRIBOULET = "Triboulet"
    YORICK = "Yorick"
    CHICOT = "Chicot"
    PERKEO = "Perkeo"


JOKER_REGISTRY: dict[JokerName, JokerInfo] = {
    JokerName.JOKER: JokerInfo(JokerName.JOKER, JokerRarity.COMMON, 2, 1),
    JokerName.GREEDY_JOKER: JokerInfo(JokerName.GREEDY_JOKER, JokerRarity.COMMON, 5, 2),
    JokerName.LUSTY_JOKER: JokerInfo(JokerName.LUSTY_JOKER, JokerRarity.COMMON, 5, 2),
    JokerName.WRATHFUL_JOKER: JokerInfo(
        JokerName.WRATHFUL_JOKER, JokerRarity.COMMON, 5, 2
    ),
    JokerName.GLUTTONOUS_JOKER: JokerInfo(
        JokerName.GLUTTONOUS_JOKER, JokerRarity.COMMON, 5, 2
    ),
    JokerName.JOLLY_JOKER: JokerInfo(JokerName.JOLLY_JOKER, JokerRarity.COMMON, 3, 1),
    JokerName.ZANY_JOKER: JokerInfo(JokerName.ZANY_JOKER, JokerRarity.COMMON, 4, 2),
    JokerName.MAD_JOKER: JokerInfo(JokerName.MAD_JOKER, JokerRarity.COMMON, 4, 2),
    JokerName.CRAZY_JOKER: JokerInfo(JokerName.CRAZY_JOKER, JokerRarity.COMMON, 4, 2),
    JokerName.DROLL_JOKER: JokerInfo(JokerName.DROLL_JOKER, JokerRarity.COMMON, 4, 2),
    JokerName.SLY_JOKER: JokerInfo(JokerName.SLY_JOKER, JokerRarity.COMMON, 3, 1),
    JokerName.WILY_JOKER: JokerInfo(JokerName.WILY_JOKER, JokerRarity.COMMON, 4, 2),
    JokerName.CLEVER_JOKER: JokerInfo(JokerName.CLEVER_JOKER, JokerRarity.COMMON, 4, 2),
    JokerName.DEVIOUS_JOKER: JokerInfo(
        JokerName.DEVIOUS_JOKER, JokerRarity.COMMON, 4, 2
    ),
    JokerName.CRAFTY_JOKER: JokerInfo(JokerName.CRAFTY_JOKER, JokerRarity.COMMON, 4, 2),
    JokerName.HALF_JOKER: JokerInfo(JokerName.HALF_JOKER, JokerRarity.COMMON, 5, 2),
    JokerName.JOKER_STENCIL: JokerInfo(
        JokerName.JOKER_STENCIL, JokerRarity.UNCOMMON, 8, 4
    ),
    JokerName.FOUR_FINGERS: JokerInfo(
        JokerName.FOUR_FINGERS, JokerRarity.UNCOMMON, 7, 3, blueprint_compat=False
    ),
    JokerName.MIME: JokerInfo(JokerName.MIME, JokerRarity.UNCOMMON, 5, 2),
    JokerName.CREDIT_CARD: JokerInfo(
        JokerName.CREDIT_CARD, JokerRarity.COMMON, 1, 1, blueprint_compat=False
    ),
    JokerName.CEREMONIAL_DAGGER: JokerInfo(
        JokerName.CEREMONIAL_DAGGER, JokerRarity.UNCOMMON, 6, 3
    ),
    JokerName.BANNER: JokerInfo(JokerName.BANNER, JokerRarity.COMMON, 5, 2),
    JokerName.MYSTIC_SUMMIT: JokerInfo(
        JokerName.MYSTIC_SUMMIT, JokerRarity.COMMON, 5, 2
    ),
    JokerName.MARBLE_JOKER: JokerInfo(
        JokerName.MARBLE_JOKER, JokerRarity.UNCOMMON, 6, 3
    ),
    JokerName.LOYALTY_CARD: JokerInfo(
        JokerName.LOYALTY_CARD, JokerRarity.UNCOMMON, 5, 2
    ),
    JokerName.EIGHT_BALL: JokerInfo(JokerName.EIGHT_BALL, JokerRarity.COMMON, 5, 2),
    JokerName.MISPRINT: JokerInfo(JokerName.MISPRINT, JokerRarity.COMMON, 4, 2),
    JokerName.DUSK: JokerInfo(JokerName.DUSK, JokerRarity.UNCOMMON, 5, 2),
    JokerName.RAISED_FIST: JokerInfo(JokerName.RAISED_FIST, JokerRarity.COMMON, 5, 2),
    JokerName.CHAOS_THE_CLOWN: JokerInfo(
        JokerName.CHAOS_THE_CLOWN, JokerRarity.COMMON, 4, 2, blueprint_compat=False
    ),
    JokerName.FIBONACCI: JokerInfo(JokerName.FIBONACCI, JokerRarity.UNCOMMON, 8, 4),
    JokerName.STEEL_JOKER: JokerInfo(JokerName.STEEL_JOKER, JokerRarity.UNCOMMON, 7, 3),
    JokerName.SCARY_FACE: JokerInfo(JokerName.SCARY_FACE, JokerRarity.COMMON, 4, 2),
    JokerName.ABSTRACT_JOKER: JokerInfo(
        JokerName.ABSTRACT_JOKER, JokerRarity.COMMON, 4, 2
    ),
    JokerName.DELAYED_GRATIFICATION: JokerInfo(
        JokerName.DELAYED_GRATIFICATION,
        JokerRarity.COMMON,
        4,
        2,
        blueprint_compat=False,
    ),
    JokerName.HACK: JokerInfo(JokerName.HACK, JokerRarity.UNCOMMON, 6, 3),
    JokerName.PAREIDOLIA: JokerInfo(
        JokerName.PAREIDOLIA, JokerRarity.UNCOMMON, 5, 2, blueprint_compat=False
    ),
    JokerName.GROS_MICHEL: JokerInfo(JokerName.GROS_MICHEL, JokerRarity.COMMON, 5, 2),
    JokerName.EVEN_STEVEN: JokerInfo(JokerName.EVEN_STEVEN, JokerRarity.COMMON, 4, 2),
    JokerName.ODD_TODD: JokerInfo(JokerName.ODD_TODD, JokerRarity.COMMON, 4, 2),
    JokerName.SCHOLAR: JokerInfo(JokerName.SCHOLAR, JokerRarity.COMMON, 4, 2),
    JokerName.BUSINESS_CARD: JokerInfo(
        JokerName.BUSINESS_CARD, JokerRarity.COMMON, 4, 2
    ),
    JokerName.SUPERNOVA: JokerInfo(JokerName.SUPERNOVA, JokerRarity.COMMON, 5, 2),
    JokerName.RIDE_THE_BUS: JokerInfo(JokerName.RIDE_THE_BUS, JokerRarity.COMMON, 6, 3),
    JokerName.SPACE_JOKER: JokerInfo(JokerName.SPACE_JOKER, JokerRarity.UNCOMMON, 5, 2),
    JokerName.EGG: JokerInfo(
        JokerName.EGG, JokerRarity.COMMON, 4, 2, blueprint_compat=False
    ),
    JokerName.BURGLAR: JokerInfo(JokerName.BURGLAR, JokerRarity.UNCOMMON, 6, 3),
    JokerName.BLACKBOARD: JokerInfo(JokerName.BLACKBOARD, JokerRarity.UNCOMMON, 6, 3),
    JokerName.RUNNER: JokerInfo(JokerName.RUNNER, JokerRarity.COMMON, 5, 2),
    JokerName.ICE_CREAM: JokerInfo(JokerName.ICE_CREAM, JokerRarity.COMMON, 5, 2),
    JokerName.DNA: JokerInfo(JokerName.DNA, JokerRarity.RARE, 8, 4),
    JokerName.SPLASH: JokerInfo(
        JokerName.SPLASH, JokerRarity.COMMON, 3, 1, blueprint_compat=False
    ),
    JokerName.BLUE_JOKER: JokerInfo(JokerName.BLUE_JOKER, JokerRarity.COMMON, 5, 2),
    JokerName.SIXTH_SENSE: JokerInfo(
        JokerName.SIXTH_SENSE, JokerRarity.UNCOMMON, 6, 3, blueprint_compat=False
    ),
    JokerName.CONSTELLATION: JokerInfo(
        JokerName.CONSTELLATION, JokerRarity.UNCOMMON, 6, 3
    ),
    JokerName.HIKER: JokerInfo(JokerName.HIKER, JokerRarity.UNCOMMON, 5, 2),
    JokerName.FACELESS_JOKER: JokerInfo(
        JokerName.FACELESS_JOKER, JokerRarity.COMMON, 4, 2
    ),
    JokerName.GREEN_JOKER: JokerInfo(JokerName.GREEN_JOKER, JokerRarity.COMMON, 4, 2),
    JokerName.SUPERPOSITION: JokerInfo(
        JokerName.SUPERPOSITION, JokerRarity.COMMON, 4, 2
    ),
    JokerName.TO_DO_LIST: JokerInfo(JokerName.TO_DO_LIST, JokerRarity.COMMON, 4, 2),
    JokerName.CAVENDISH: JokerInfo(JokerName.CAVENDISH, JokerRarity.COMMON, 4, 2),
    JokerName.CARD_SHARP: JokerInfo(JokerName.CARD_SHARP, JokerRarity.UNCOMMON, 6, 3),
    JokerName.RED_CARD: JokerInfo(JokerName.RED_CARD, JokerRarity.COMMON, 5, 2),
    JokerName.MADNESS: JokerInfo(JokerName.MADNESS, JokerRarity.UNCOMMON, 7, 3),
    JokerName.SQUARE_JOKER: JokerInfo(JokerName.SQUARE_JOKER, JokerRarity.COMMON, 4, 2),
    JokerName.SEANCE: JokerInfo(JokerName.SEANCE, JokerRarity.UNCOMMON, 6, 3),
    JokerName.RIFF_RAFF: JokerInfo(JokerName.RIFF_RAFF, JokerRarity.COMMON, 6, 3),
    JokerName.VAMPIRE: JokerInfo(JokerName.VAMPIRE, JokerRarity.UNCOMMON, 7, 3),
    JokerName.SHORTCUT: JokerInfo(
        JokerName.SHORTCUT, JokerRarity.UNCOMMON, 7, 3, blueprint_compat=False
    ),
    JokerName.HOLOGRAM: JokerInfo(JokerName.HOLOGRAM, JokerRarity.UNCOMMON, 7, 3),
    JokerName.VAGABOND: JokerInfo(JokerName.VAGABOND, JokerRarity.RARE, 8, 4),
    JokerName.BARON: JokerInfo(JokerName.BARON, JokerRarity.RARE, 8, 4),
    JokerName.CLOUD_9: JokerInfo(
        JokerName.CLOUD_9, JokerRarity.UNCOMMON, 7, 3, blueprint_compat=False
    ),
    JokerName.ROCKET: JokerInfo(
        JokerName.ROCKET, JokerRarity.UNCOMMON, 6, 3, blueprint_compat=False
    ),
    JokerName.OBELISK: JokerInfo(JokerName.OBELISK, JokerRarity.RARE, 8, 4),
    JokerName.MIDAS_MASK: JokerInfo(
        JokerName.MIDAS_MASK, JokerRarity.UNCOMMON, 7, 3, blueprint_compat=False
    ),
    JokerName.LUCHADOR: JokerInfo(JokerName.LUCHADOR, JokerRarity.UNCOMMON, 5, 2),
    JokerName.PHOTOGRAPH: JokerInfo(JokerName.PHOTOGRAPH, JokerRarity.COMMON, 5, 2),
    JokerName.GIFT_CARD: JokerInfo(
        JokerName.GIFT_CARD, JokerRarity.UNCOMMON, 6, 3, blueprint_compat=False
    ),
    JokerName.TURTLE_BEAN: JokerInfo(
        JokerName.TURTLE_BEAN, JokerRarity.UNCOMMON, 6, 3, blueprint_compat=False
    ),
    JokerName.EROSION: JokerInfo(JokerName.EROSION, JokerRarity.UNCOMMON, 6, 3),
    JokerName.RESERVED_PARKING: JokerInfo(
        JokerName.RESERVED_PARKING, JokerRarity.COMMON, 6, 3
    ),
    JokerName.MAIL_IN_REBATE: JokerInfo(
        JokerName.MAIL_IN_REBATE, JokerRarity.COMMON, 4, 2
    ),
    JokerName.TO_THE_MOON: JokerInfo(
        JokerName.TO_THE_MOON, JokerRarity.UNCOMMON, 5, 2, blueprint_compat=False
    ),
    JokerName.HALLUCINATION: JokerInfo(
        JokerName.HALLUCINATION, JokerRarity.COMMON, 4, 2
    ),
    JokerName.FORTUNE_TELLER: JokerInfo(
        JokerName.FORTUNE_TELLER, JokerRarity.COMMON, 6, 3
    ),
    JokerName.JUGGLER: JokerInfo(
        JokerName.JUGGLER, JokerRarity.COMMON, 4, 2, blueprint_compat=False
    ),
    JokerName.DRUNKARD: JokerInfo(
        JokerName.DRUNKARD, JokerRarity.COMMON, 4, 2, blueprint_compat=False
    ),
    JokerName.STONE_JOKER: JokerInfo(JokerName.STONE_JOKER, JokerRarity.UNCOMMON, 6, 3),
    JokerName.GOLDEN_JOKER: JokerInfo(
        JokerName.GOLDEN_JOKER, JokerRarity.COMMON, 6, 3, blueprint_compat=False
    ),
    JokerName.LUCKY_CAT: JokerInfo(JokerName.LUCKY_CAT, JokerRarity.UNCOMMON, 6, 3),
    JokerName.BASEBALL_CARD: JokerInfo(JokerName.BASEBALL_CARD, JokerRarity.RARE, 8, 4),
    JokerName.BULL: JokerInfo(JokerName.BULL, JokerRarity.UNCOMMON, 6, 3),
    JokerName.DIET_COLA: JokerInfo(JokerName.DIET_COLA, JokerRarity.UNCOMMON, 6, 3),
    JokerName.TRADING_CARD: JokerInfo(
        JokerName.TRADING_CARD, JokerRarity.UNCOMMON, 6, 3, blueprint_compat=False
    ),
    JokerName.FLASH_CARD: JokerInfo(JokerName.FLASH_CARD, JokerRarity.UNCOMMON, 5, 2),
    JokerName.POPCORN: JokerInfo(JokerName.POPCORN, JokerRarity.COMMON, 5, 2),
    JokerName.SPARE_TROUSERS: JokerInfo(
        JokerName.SPARE_TROUSERS, JokerRarity.UNCOMMON, 6, 3
    ),
    JokerName.ANCIENT_JOKER: JokerInfo(JokerName.ANCIENT_JOKER, JokerRarity.RARE, 8, 4),
    JokerName.RAMEN: JokerInfo(JokerName.RAMEN, JokerRarity.UNCOMMON, 6, 3),
    JokerName.WALKIE_TALKIE: JokerInfo(
        JokerName.WALKIE_TALKIE, JokerRarity.COMMON, 4, 2
    ),
    JokerName.SELTZER: JokerInfo(JokerName.SELTZER, JokerRarity.UNCOMMON, 6, 3),
    JokerName.CASTLE: JokerInfo(JokerName.CASTLE, JokerRarity.UNCOMMON, 6, 3),
    JokerName.SMILEY_FACE: JokerInfo(JokerName.SMILEY_FACE, JokerRarity.COMMON, 4, 2),
    JokerName.CAMPFIRE: JokerInfo(JokerName.CAMPFIRE, JokerRarity.RARE, 9, 4),
    JokerName.GOLDEN_TICKET: JokerInfo(
        JokerName.GOLDEN_TICKET, JokerRarity.COMMON, 5, 2
    ),
    JokerName.MR_BONES: JokerInfo(
        JokerName.MR_BONES, JokerRarity.UNCOMMON, 5, 2, blueprint_compat=False
    ),
    JokerName.ACROBAT: JokerInfo(JokerName.ACROBAT, JokerRarity.UNCOMMON, 6, 3),
    JokerName.SOCK_AND_BUSKIN: JokerInfo(
        JokerName.SOCK_AND_BUSKIN, JokerRarity.UNCOMMON, 6, 3
    ),
    JokerName.SWASHBUCKLER: JokerInfo(JokerName.SWASHBUCKLER, JokerRarity.COMMON, 4, 2),
    JokerName.TROUBADOUR: JokerInfo(
        JokerName.TROUBADOUR, JokerRarity.UNCOMMON, 6, 3, blueprint_compat=False
    ),
    JokerName.CERTIFICATE: JokerInfo(JokerName.CERTIFICATE, JokerRarity.UNCOMMON, 6, 3),
    JokerName.SMEARED_JOKER: JokerInfo(
        JokerName.SMEARED_JOKER, JokerRarity.UNCOMMON, 7, 3, blueprint_compat=False
    ),
    JokerName.THROWBACK: JokerInfo(JokerName.THROWBACK, JokerRarity.UNCOMMON, 6, 3),
    JokerName.HANGING_CHAD: JokerInfo(JokerName.HANGING_CHAD, JokerRarity.COMMON, 4, 2),
    JokerName.ROUGH_GEM: JokerInfo(JokerName.ROUGH_GEM, JokerRarity.UNCOMMON, 7, 3),
    JokerName.BLOODSTONE: JokerInfo(JokerName.BLOODSTONE, JokerRarity.UNCOMMON, 7, 3),
    JokerName.ARROWHEAD: JokerInfo(JokerName.ARROWHEAD, JokerRarity.UNCOMMON, 7, 3),
    JokerName.ONYX_AGATE: JokerInfo(JokerName.ONYX_AGATE, JokerRarity.UNCOMMON, 7, 3),
    JokerName.GLASS_JOKER: JokerInfo(JokerName.GLASS_JOKER, JokerRarity.UNCOMMON, 6, 3),
    JokerName.SHOWMAN: JokerInfo(
        JokerName.SHOWMAN, JokerRarity.UNCOMMON, 5, 2, blueprint_compat=False
    ),
    JokerName.FLOWER_POT: JokerInfo(JokerName.FLOWER_POT, JokerRarity.UNCOMMON, 6, 3),
    JokerName.BLUEPRINT: JokerInfo(JokerName.BLUEPRINT, JokerRarity.RARE, 10, 5),
    JokerName.WEE_JOKER: JokerInfo(JokerName.WEE_JOKER, JokerRarity.RARE, 8, 4),
    JokerName.MERRY_ANDY: JokerInfo(
        JokerName.MERRY_ANDY, JokerRarity.UNCOMMON, 7, 3, blueprint_compat=False
    ),
    JokerName.OOPS_ALL_6S: JokerInfo(
        JokerName.OOPS_ALL_6S, JokerRarity.UNCOMMON, 4, 2, blueprint_compat=False
    ),
    JokerName.THE_IDOL: JokerInfo(JokerName.THE_IDOL, JokerRarity.UNCOMMON, 6, 3),
    JokerName.SEEING_DOUBLE: JokerInfo(
        JokerName.SEEING_DOUBLE, JokerRarity.UNCOMMON, 6, 3
    ),
    JokerName.MATADOR: JokerInfo(JokerName.MATADOR, JokerRarity.UNCOMMON, 7, 3),
    JokerName.HIT_THE_ROAD: JokerInfo(JokerName.HIT_THE_ROAD, JokerRarity.RARE, 8, 4),
    JokerName.THE_DUO: JokerInfo(JokerName.THE_DUO, JokerRarity.RARE, 8, 4),
    JokerName.THE_TRIO: JokerInfo(JokerName.THE_TRIO, JokerRarity.RARE, 8, 4),
    JokerName.THE_FAMILY: JokerInfo(JokerName.THE_FAMILY, JokerRarity.RARE, 8, 4),
    JokerName.THE_ORDER: JokerInfo(JokerName.THE_ORDER, JokerRarity.RARE, 8, 4),
    JokerName.THE_TRIBE: JokerInfo(JokerName.THE_TRIBE, JokerRarity.RARE, 8, 4),
    JokerName.STUNTMAN: JokerInfo(JokerName.STUNTMAN, JokerRarity.RARE, 7, 3),
    JokerName.INVISIBLE_JOKER: JokerInfo(
        JokerName.INVISIBLE_JOKER, JokerRarity.RARE, 8, 4, blueprint_compat=False
    ),
    JokerName.BRAINSTORM: JokerInfo(JokerName.BRAINSTORM, JokerRarity.RARE, 10, 5),
    JokerName.SATELLITE: JokerInfo(
        JokerName.SATELLITE, JokerRarity.UNCOMMON, 6, 3, blueprint_compat=False
    ),
    JokerName.SHOOT_THE_MOON: JokerInfo(
        JokerName.SHOOT_THE_MOON, JokerRarity.COMMON, 5, 2
    ),
    JokerName.DRIVER: JokerInfo(JokerName.DRIVER, JokerRarity.RARE, 7, 3),
    JokerName.CARTOMANCER: JokerInfo(JokerName.CARTOMANCER, JokerRarity.UNCOMMON, 6, 3),
    JokerName.ASTRONOMER: JokerInfo(
        JokerName.ASTRONOMER, JokerRarity.UNCOMMON, 8, 4, blueprint_compat=False
    ),
    JokerName.BURNT_JOKER: JokerInfo(JokerName.BURNT_JOKER, JokerRarity.RARE, 8, 4),
    JokerName.BOOTSTRAPS: JokerInfo(JokerName.BOOTSTRAPS, JokerRarity.UNCOMMON, 7, 3),
    JokerName.CANIO: JokerInfo(JokerName.CANIO, JokerRarity.LEGENDARY, 20, 10),
    JokerName.TRIBOULET: JokerInfo(JokerName.TRIBOULET, JokerRarity.LEGENDARY, 20, 10),
    JokerName.YORICK: JokerInfo(JokerName.YORICK, JokerRarity.LEGENDARY, 20, 10),
    JokerName.CHICOT: JokerInfo(
        JokerName.CHICOT, JokerRarity.LEGENDARY, 20, 10, blueprint_compat=False
    ),
    JokerName.PERKEO: JokerInfo(JokerName.PERKEO, JokerRarity.LEGENDARY, 20, 10),
}
