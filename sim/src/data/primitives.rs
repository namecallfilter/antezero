//! Core card and poker-hand primitives: suits, ranks, hand types,
//! enhancements, editions, seals and pack kinds.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Suit {
	Spades = 0,
	Hearts = 1,
	Clubs = 2,
	Diamonds = 3,
}

impl Suit {
	/// Iteration order of `get_flush` (and Ancient Joker's suit list).
	pub const ALL: [Suit; 4] = [Suit::Spades, Suit::Hearts, Suit::Clubs, Suit::Diamonds];

	#[must_use]
	pub fn name(self) -> &'static str {
		match self {
			Suit::Spades => "Spades",
			Suit::Hearts => "Hearts",
			Suit::Clubs => "Clubs",
			Suit::Diamonds => "Diamonds",
		}
	}

	#[must_use]
	pub fn letter(self) -> u8 {
		match self {
			Suit::Spades => b'S',
			Suit::Hearts => b'H',
			Suit::Clubs => b'C',
			Suit::Diamonds => b'D',
		}
	}

	#[must_use]
	pub fn from_name(s: &str) -> Option<Suit> {
		Suit::ALL.into_iter().find(|suit| suit.name() == s)
	}

	/// `base.suit_nominal`.
	#[inline]
	#[must_use]
	pub fn nominal(self) -> f64 {
		match self {
			Suit::Diamonds => 0.01,
			Suit::Clubs => 0.02,
			Suit::Hearts => 0.03,
			Suit::Spades => 0.04,
		}
	}

	/// `base.suit_nominal_original`.
	#[inline]
	#[must_use]
	pub fn nominal_original(self) -> f64 {
		match self {
			Suit::Diamonds => 0.001,
			Suit::Clubs => 0.002,
			Suit::Hearts => 0.003,
			Suit::Spades => 0.004,
		}
	}

	#[inline]
	#[must_use]
	pub fn is_red(self) -> bool {
		matches!(self, Suit::Hearts | Suit::Diamonds)
	}
}

/// A card rank; the discriminant is the game's `base.id`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Rank {
	Two = 2,
	Three,
	Four,
	Five,
	Six,
	Seven,
	Eight,
	Nine,
	Ten,
	Jack,
	Queen,
	King,
	Ace,
}

impl Rank {
	/// Every rank, from the 2 up to the Ace: the list the game's rank rolls
	/// index.
	pub const ALL: [Rank; 13] = [
		Rank::Two,
		Rank::Three,
		Rank::Four,
		Rank::Five,
		Rank::Six,
		Rank::Seven,
		Rank::Eight,
		Rank::Nine,
		Rank::Ten,
		Rank::Jack,
		Rank::Queen,
		Rank::King,
		Rank::Ace,
	];
	/// The face ranks, in [`Rank::ALL`] order.
	pub const FACES: [Rank; 3] = [Rank::Jack, Rank::Queen, Rank::King];
	/// The numbered ranks, from the 2 up to the 10, in [`Rank::ALL`] order.
	pub const NUMBERED: [Rank; 9] = [
		Rank::Two,
		Rank::Three,
		Rank::Four,
		Rank::Five,
		Rank::Six,
		Rank::Seven,
		Rank::Eight,
		Rank::Nine,
		Rank::Ten,
	];

	/// `base.id`: 2 to 10, then Jack 11, Queen 12, King 13 and Ace 14.
	#[inline]
	#[must_use]
	pub const fn id(self) -> u8 {
		self as u8
	}

	/// `base.value`: `"2"` to `"10"`, then `"Jack"`, `"Queen"`, `"King"` and
	/// `"Ace"`.
	#[must_use]
	pub fn value(self) -> &'static str {
		match self {
			Rank::Two => "2",
			Rank::Three => "3",
			Rank::Four => "4",
			Rank::Five => "5",
			Rank::Six => "6",
			Rank::Seven => "7",
			Rank::Eight => "8",
			Rank::Nine => "9",
			Rank::Ten => "10",
			Rank::Jack => "Jack",
			Rank::Queen => "Queen",
			Rank::King => "King",
			Rank::Ace => "Ace",
		}
	}

	/// The rank's position in [`Rank::ALL`].
	#[inline]
	#[must_use]
	pub const fn index(self) -> usize {
		self as usize - Rank::Two as usize
	}

	/// The rank with the `base.id` `id`.
	#[must_use]
	pub fn from_id(id: u8) -> Option<Rank> {
		Rank::ALL.get(usize::from(id).checked_sub(2)?).copied()
	}

	#[inline]
	#[must_use]
	pub fn is_face(self) -> bool {
		Rank::FACES.contains(&self)
	}

	/// `base.nominal` (chip value).
	#[inline]
	#[must_use]
	pub fn nominal(self) -> f64 {
		match self {
			Rank::Jack | Rank::Queen | Rank::King => 10.0,
			Rank::Ace => 11.0,
			pip => f64::from(pip.id()),
		}
	}

	/// `base.face_nominal`.
	#[inline]
	#[must_use]
	pub fn face_nominal(self) -> f64 {
		match self {
			Rank::Jack => 0.1,
			Rank::Queen => 0.2,
			Rank::King => 0.3,
			Rank::Ace => 0.4,
			_ => 0.0,
		}
	}

	/// The `P_CARDS` key suffix character.
	#[must_use]
	pub fn letter(self) -> u8 {
		match self {
			Rank::Ten => b'T',
			Rank::Jack => b'J',
			Rank::Queen => b'Q',
			Rank::King => b'K',
			Rank::Ace => b'A',
			pip => b'0' + pip.id(),
		}
	}

	/// Strength's rank: one higher, and a 2 for an Ace.
	#[must_use]
	pub fn raised(self) -> Rank {
		Rank::ALL[(usize::from(self.id()) - 1) % Rank::ALL.len()]
	}
}

/// `G.P_CARDS` sorted by key (`"C_2" < ... < "S_T"`), the order
/// `pseudorandom_element(G.P_CARDS, ...)` samples from.
pub const P_CARDS_SORTED: [(Suit, Rank); Suit::ALL.len() * Rank::ALL.len()] = {
	const RANKS: [Rank; Rank::ALL.len()] = [
		Rank::Two,
		Rank::Three,
		Rank::Four,
		Rank::Five,
		Rank::Six,
		Rank::Seven,
		Rank::Eight,
		Rank::Nine,
		Rank::Ace,
		Rank::Jack,
		Rank::King,
		Rank::Queen,
		Rank::Ten,
	];
	const SUITS: [Suit; 4] = [Suit::Clubs, Suit::Diamonds, Suit::Hearts, Suit::Spades];
	let mut out = [(Suit::Clubs, Rank::Two); Suit::ALL.len() * Rank::ALL.len()];
	let mut i = 0;
	while i < out.len() {
		out[i] = (SUITS[i / RANKS.len()], RANKS[i % RANKS.len()]);
		i += 1;
	}
	out
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum HandType {
	FlushFive = 0,
	FlushHouse = 1,
	FiveOfAKind = 2,
	StraightFlush = 3,
	FourOfAKind = 4,
	FullHouse = 5,
	Flush = 6,
	Straight = 7,
	ThreeOfAKind = 8,
	TwoPair = 9,
	Pair = 10,
	HighCard = 11,
}

impl HandType {
	/// `G.handlist` order (also the `order` field).
	pub const ALL: [HandType; 12] = [
		HandType::FlushFive,
		HandType::FlushHouse,
		HandType::FiveOfAKind,
		HandType::StraightFlush,
		HandType::FourOfAKind,
		HandType::FullHouse,
		HandType::Flush,
		HandType::Straight,
		HandType::ThreeOfAKind,
		HandType::TwoPair,
		HandType::Pair,
		HandType::HighCard,
	];
	/// Iteration order of `pairs(G.GAME.hands)` in the game's `LuaJIT` build
	/// (hash order; verified against the real lua51.dll).
	pub const PAIRS_ORDER: [HandType; 12] = [
		HandType::FlushHouse,
		HandType::FullHouse,
		HandType::Flush,
		HandType::Pair,
		HandType::HighCard,
		HandType::StraightFlush,
		HandType::Straight,
		HandType::TwoPair,
		HandType::FlushFive,
		HandType::FiveOfAKind,
		HandType::ThreeOfAKind,
		HandType::FourOfAKind,
	];

	#[must_use]
	pub fn name(self) -> &'static str {
		match self {
			HandType::FlushFive => "Flush Five",
			HandType::FlushHouse => "Flush House",
			HandType::FiveOfAKind => "Five of a Kind",
			HandType::StraightFlush => "Straight Flush",
			HandType::FourOfAKind => "Four of a Kind",
			HandType::FullHouse => "Full House",
			HandType::Flush => "Flush",
			HandType::Straight => "Straight",
			HandType::ThreeOfAKind => "Three of a Kind",
			HandType::TwoPair => "Two Pair",
			HandType::Pair => "Pair",
			HandType::HighCard => "High Card",
		}
	}

	#[must_use]
	pub fn from_name(s: &str) -> Option<HandType> {
		HandType::ALL.iter().copied().find(|h| h.name() == s)
	}

	/// The hand's base values (`G.GAME.hands[...]` at the start of a run).
	#[must_use]
	pub fn base(self) -> HandBase {
		let (chips, mult, level_chips, level_mult, visible) = match self {
			HandType::FlushFive => (160.0, 16.0, 50.0, 3.0, false),
			HandType::FlushHouse => (140.0, 14.0, 40.0, 4.0, false),
			HandType::FiveOfAKind => (120.0, 12.0, 35.0, 3.0, false),
			HandType::StraightFlush => (100.0, 8.0, 40.0, 4.0, true),
			HandType::FourOfAKind => (60.0, 7.0, 30.0, 3.0, true),
			HandType::FullHouse => (40.0, 4.0, 25.0, 2.0, true),
			HandType::Flush => (35.0, 4.0, 15.0, 2.0, true),
			HandType::Straight => (30.0, 4.0, 30.0, 3.0, true),
			HandType::ThreeOfAKind => (30.0, 3.0, 20.0, 2.0, true),
			HandType::TwoPair => (20.0, 2.0, 20.0, 1.0, true),
			HandType::Pair => (10.0, 2.0, 15.0, 1.0, true),
			HandType::HighCard => (5.0, 1.0, 10.0, 1.0, true),
		};
		HandBase {
			chips,
			mult,
			level_chips,
			level_mult,
			visible,
		}
	}

	/// The `order` field (1-based position in `G.handlist`).
	#[inline]
	#[must_use]
	pub fn order(self) -> u8 {
		self as u8 + 1
	}
}

/// A poker hand's base values: `s_chips`, `s_mult`, the `l_chips` and
/// `l_mult` each level adds, and whether it is visible from the start.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HandBase {
	pub chips: f64,
	pub mult: f64,
	pub level_chips: f64,
	pub level_mult: f64,
	pub visible: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
#[repr(u8)]
pub enum Enhancement {
	#[default]
	Base = 0,
	Bonus = 1,
	Mult = 2,
	Wild = 3,
	Glass = 4,
	Steel = 5,
	Stone = 6,
	Gold = 7,
	Lucky = 8,
}

impl Enhancement {
	/// Every enhancement, by discriminant.
	pub const ALL: [Enhancement; 9] = [
		Enhancement::Base,
		Enhancement::Bonus,
		Enhancement::Mult,
		Enhancement::Wild,
		Enhancement::Glass,
		Enhancement::Steel,
		Enhancement::Stone,
		Enhancement::Gold,
		Enhancement::Lucky,
	];
	/// `G.P_CENTER_POOLS.Enhanced` (sorted by `order`).
	pub const POOL: [Enhancement; 8] = [
		Enhancement::Bonus,
		Enhancement::Mult,
		Enhancement::Wild,
		Enhancement::Glass,
		Enhancement::Steel,
		Enhancement::Stone,
		Enhancement::Gold,
		Enhancement::Lucky,
	];

	#[must_use]
	pub fn key(self) -> &'static str {
		match self {
			Enhancement::Base => "c_base",
			Enhancement::Bonus => "m_bonus",
			Enhancement::Mult => "m_mult",
			Enhancement::Wild => "m_wild",
			Enhancement::Glass => "m_glass",
			Enhancement::Steel => "m_steel",
			Enhancement::Stone => "m_stone",
			Enhancement::Gold => "m_gold",
			Enhancement::Lucky => "m_lucky",
		}
	}

	#[must_use]
	pub fn from_key(k: &str) -> Option<Enhancement> {
		Enhancement::ALL.into_iter().find(|e| e.key() == k)
	}

	/// `config.bonus` (chips added by the enhancement).
	#[inline]
	#[must_use]
	pub fn bonus(self) -> f64 {
		match self {
			Enhancement::Bonus => 30.0,
			Enhancement::Stone => 50.0,
			_ => 0.0,
		}
	}

	/// `config.mult`.
	#[inline]
	#[must_use]
	pub fn mult(self) -> f64 {
		match self {
			Enhancement::Mult => 4.0,
			Enhancement::Lucky => 20.0,
			_ => 0.0,
		}
	}

	/// `config.Xmult`.
	#[inline]
	#[must_use]
	pub fn x_mult(self) -> f64 {
		if self == Enhancement::Glass { 2.0 } else { 1.0 }
	}

	/// `config.h_x_mult` (held in hand).
	#[inline]
	#[must_use]
	pub fn h_x_mult(self) -> f64 {
		if self == Enhancement::Steel { 1.5 } else { 0.0 }
	}

	/// `config.h_dollars` (held at end of round).
	#[inline]
	#[must_use]
	pub fn h_dollars(self) -> f64 {
		if self == Enhancement::Gold { 3.0 } else { 0.0 }
	}

	/// `config.p_dollars` (Lucky card payout).
	#[inline]
	#[must_use]
	pub fn p_dollars(self) -> f64 {
		if self == Enhancement::Lucky {
			20.0
		} else {
			0.0
		}
	}
}

/// A card's edition (`card.edition`, which a card without one has `nil`
/// for: `None` here).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Edition {
	Foil = 1,
	Holo = 2,
	Polychrome = 3,
	Negative = 4,
}

impl Edition {
	/// Every edition, in discriminant order.
	pub const ALL: [Edition; 4] = [
		Edition::Foil,
		Edition::Holo,
		Edition::Polychrome,
		Edition::Negative,
	];
	/// How many numbers [`Edition::id`] gives: one for no edition and one for
	/// each edition.
	pub const ID_COUNT: usize = Edition::ALL.len() + 1;

	/// The number of a card's edition in codes and observations: 0 for none,
	/// else the discriminant (1 to 4).
	#[inline]
	#[must_use]
	pub fn id(edition: Option<Edition>) -> u8 {
		edition.map_or(0, |e| e as u8)
	}

	/// The card edition numbered `id` by [`Edition::id`], or `None` if no
	/// edition has that number.
	#[must_use]
	pub fn from_id(id: u8) -> Option<Option<Edition>> {
		if id == 0 {
			return Some(None);
		}
		Edition::ALL.into_iter().find(|&e| e as u8 == id).map(Some)
	}

	#[must_use]
	pub fn name(self) -> &'static str {
		match self {
			Edition::Foil => "foil",
			Edition::Holo => "holo",
			Edition::Polychrome => "polychrome",
			Edition::Negative => "negative",
		}
	}

	/// `edition.chips`: what Foil adds to the chips.
	#[inline]
	#[must_use]
	pub fn chips(self) -> Option<f64> {
		(self == Edition::Foil).then_some(50.0)
	}

	/// `edition.mult`: what Holographic adds to the mult.
	#[inline]
	#[must_use]
	pub fn mult(self) -> Option<f64> {
		(self == Edition::Holo).then_some(10.0)
	}

	/// `edition.x_mult`: what Polychrome multiplies the mult by.
	#[inline]
	#[must_use]
	pub fn x_mult(self) -> Option<f64> {
		(self == Edition::Polychrome).then_some(1.5)
	}

	/// Shop price surcharge in `Card:set_cost`.
	#[inline]
	#[must_use]
	pub fn extra_cost(self) -> f64 {
		match self {
			Edition::Foil => 2.0,
			Edition::Holo => 3.0,
			Edition::Polychrome | Edition::Negative => 5.0,
		}
	}
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Seal {
	Gold = 0,
	Red = 1,
	Blue = 2,
	Purple = 3,
}

impl Seal {
	/// Every seal, by discriminant.
	pub const ALL: [Seal; 4] = [Seal::Gold, Seal::Red, Seal::Blue, Seal::Purple];

	#[must_use]
	pub fn name(self) -> &'static str {
		match self {
			Seal::Gold => "Gold",
			Seal::Red => "Red",
			Seal::Blue => "Blue",
			Seal::Purple => "Purple",
		}
	}
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum PackKind {
	Arcana,
	Celestial,
	Spectral,
	Standard,
	Buffoon,
}
