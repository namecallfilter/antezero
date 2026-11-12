//! Static game data: enums, prototype definitions and the orderings the game
//! uses when sampling. Large tables live in the generated `data_gen.rs`.

pub use crate::data_gen::*;

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
		Some(match s {
			"Spades" => Suit::Spades,
			"Hearts" => Suit::Hearts,
			"Clubs" => Suit::Clubs,
			"Diamonds" => Suit::Diamonds,
			_ => return None,
		})
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

/// Card ranks use the game's `base.id` values: 2..=10, J=11, Q=12, K=13, A=14.
pub type Rank = u8;
pub const JACK: Rank = 11;
pub const QUEEN: Rank = 12;
pub const KING: Rank = 13;
pub const ACE: Rank = 14;

/// `base.nominal` (chip value) for a rank.
#[inline]
#[must_use]
pub fn rank_nominal(r: Rank) -> f64 {
	match r {
		2..=10 => f64::from(r),
		JACK | QUEEN | KING => 10.0,
		_ => 11.0,
	}
}

/// `base.face_nominal`.
#[inline]
#[must_use]
pub fn rank_face_nominal(r: Rank) -> f64 {
	match r {
		JACK => 0.1,
		QUEEN => 0.2,
		KING => 0.3,
		ACE => 0.4,
		_ => 0.0,
	}
}

/// The `P_CARDS` key suffix character for a rank.
#[must_use]
pub fn rank_char(r: Rank) -> u8 {
	match r {
		2..=9 => b'0' + r,
		10 => b'T',
		JACK => b'J',
		QUEEN => b'Q',
		KING => b'K',
		_ => b'A',
	}
}

/// `base.value` strings (used by the `Pillar/cards_played` bookkeeping and the oracle).
#[must_use]
pub fn rank_value(r: Rank) -> &'static str {
	match r {
		2 => "2",
		3 => "3",
		4 => "4",
		5 => "5",
		6 => "6",
		7 => "7",
		8 => "8",
		9 => "9",
		10 => "10",
		JACK => "Jack",
		QUEEN => "Queen",
		KING => "King",
		_ => "Ace",
	}
}

/// `G.P_CARDS` sorted by key (`"C_2" < ... < "S_T"`), the order
/// `pseudorandom_element(G.P_CARDS, ...)` samples from.
pub const P_CARDS_SORTED: [(Suit, Rank); 52] = {
	const RANKS: [Rank; 13] = [2, 3, 4, 5, 6, 7, 8, 9, ACE, JACK, KING, QUEEN, 10];
	const SUITS: [Suit; 4] = [Suit::Clubs, Suit::Diamonds, Suit::Hearts, Suit::Spades];
	let mut out = [(Suit::Clubs, 2u8); 52];
	let mut i = 0;
	while i < 52 {
		out[i] = (SUITS[i / 13], RANKS[i % 13]);
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
		Some(match k {
			"c_base" => Enhancement::Base,
			"m_bonus" => Enhancement::Bonus,
			"m_mult" => Enhancement::Mult,
			"m_wild" => Enhancement::Wild,
			"m_glass" => Enhancement::Glass,
			"m_steel" => Enhancement::Steel,
			"m_stone" => Enhancement::Stone,
			"m_gold" => Enhancement::Gold,
			"m_lucky" => Enhancement::Lucky,
			_ => return None,
		})
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
#[repr(u8)]
pub enum Edition {
	#[default]
	None = 0,
	Foil = 1,
	Holo = 2,
	Polychrome = 3,
	Negative = 4,
}

impl Edition {
	/// Every edition, by discriminant.
	pub const ALL: [Edition; 5] = [
		Edition::None,
		Edition::Foil,
		Edition::Holo,
		Edition::Polychrome,
		Edition::Negative,
	];

	#[must_use]
	pub fn name(self) -> Option<&'static str> {
		match self {
			Edition::None => None,
			Edition::Foil => Some("foil"),
			Edition::Holo => Some("holo"),
			Edition::Polychrome => Some("polychrome"),
			Edition::Negative => Some("negative"),
		}
	}

	/// Shop price surcharge in `Card:set_cost`.
	#[inline]
	#[must_use]
	pub fn extra_cost(self) -> f64 {
		match self {
			Edition::None => 0.0,
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

#[derive(Clone, Copy, Debug)]
pub struct JokerDef {
	pub key: &'static str,
	pub name: &'static str,
	pub rarity: u8,
	pub cost: u32,
	pub perishable_compat: bool,
	pub eternal_compat: bool,
	pub enhancement_gate: Option<Enhancement>,
	pub no_pool_flag: Option<&'static str>,
	pub yes_pool_flag: Option<&'static str>,
}

#[derive(Clone, Copy, Debug)]
pub struct ConsumableDef {
	pub key: &'static str,
	pub name: &'static str,
	pub cost: u32,
	pub hand_type: Option<HandType>,
	pub max_highlighted: u8,
	pub min_highlighted: u8,
}

#[derive(Clone, Copy, Debug)]
pub struct VoucherDef {
	pub key: &'static str,
	pub name: &'static str,
	pub cost: u32,
	pub requires: Option<Voucher>,
	pub extra: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct BoosterDef {
	pub key: &'static str,
	pub name: &'static str,
	pub kind: PackKind,
	pub cost: u32,
	pub weight: f64,
	pub extra: u8,
	pub choose: u8,
}

#[derive(Clone, Copy, Debug)]
pub struct TagDef {
	pub key: &'static str,
	pub name: &'static str,
	pub min_ante: i32,
}

#[derive(Clone, Copy, Debug)]
pub struct BlindDef {
	pub key: &'static str,
	pub name: &'static str,
	pub dollars: i32,
	pub mult: f64,
	pub boss: bool,
	pub showdown: bool,
	pub min_ante: i32,
}

#[derive(Clone, Copy, Debug)]
pub struct BackDef {
	pub key: &'static str,
	pub name: &'static str,
}

macro_rules! def_accessors {
	($ty:ident, $all:ident, $defs:ident, $def:ty) => {
		impl $ty {
			#[inline]
			pub fn def(self) -> &'static $def {
				&$defs[self as usize]
			}

			#[inline]
			pub fn key(self) -> &'static str {
				self.def().key
			}

			#[inline]
			pub fn name(self) -> &'static str {
				self.def().name
			}

			pub fn from_key(k: &str) -> Option<$ty> {
				$all.iter().copied().find(|x| x.key() == k)
			}
		}
	};
}

def_accessors!(Joker, ALL_JOKERS, JOKER_DEFS, JokerDef);
def_accessors!(Tarot, ALL_TAROTS, TAROT_DEFS, ConsumableDef);
def_accessors!(Planet, ALL_PLANETS, PLANET_DEFS, ConsumableDef);
def_accessors!(Spectral, ALL_SPECTRALS, SPECTRAL_DEFS, ConsumableDef);
def_accessors!(Voucher, ALL_VOUCHERS, VOUCHER_DEFS, VoucherDef);
def_accessors!(Booster, ALL_BOOSTERS, BOOSTER_DEFS, BoosterDef);
def_accessors!(TagKind, ALL_TAGS, TAG_DEFS, TagDef);
def_accessors!(BlindKind, ALL_BLINDS, BLIND_DEFS, BlindDef);
def_accessors!(Back, ALL_BACKS, BACK_DEFS, BackDef);

impl Joker {
	#[inline]
	#[must_use]
	pub fn rarity(self) -> u8 {
		self.def().rarity
	}
}
impl Planet {
	#[inline]
	#[must_use]
	#[expect(
		clippy::missing_panics_doc,
		reason = "the generated table gives every planet a hand"
	)]
	pub fn hand(self) -> HandType {
		self.def()
			.hand_type
			.expect("every planet levels up one poker hand")
	}

	#[must_use]
	#[expect(
		clippy::missing_panics_doc,
		reason = "the generated table has a planet for every hand"
	)]
	pub fn for_hand(h: HandType) -> Planet {
		*ALL_PLANETS
			.iter()
			.find(|p| p.hand() == h)
			.expect("every poker hand has its planet")
	}
}
impl Booster {
	#[inline]
	#[must_use]
	pub fn kind(self) -> PackKind {
		self.def().kind
	}
}
impl BlindKind {
	#[inline]
	#[must_use]
	pub fn is_boss(self) -> bool {
		self.def().boss
	}
}

/// Identity of a `G.P_CENTERS` entry as far as `G.GAME.used_jokers` is
/// concerned (the "no duplicates" bookkeeping covers every created card).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Center {
	Joker(Joker),
	Tarot(Tarot),
	Planet(Planet),
	Spectral(Spectral),
	Voucher(Voucher),
	Booster(Booster),
	Enhanced(Enhancement),
}

// Offsets of each center kind in the dense `Center::index` numbering.
const TAROT_BASE: usize = ALL_JOKERS.len();
const PLANET_BASE: usize = TAROT_BASE + ALL_TAROTS.len();
const SPECTRAL_BASE: usize = PLANET_BASE + ALL_PLANETS.len();
const VOUCHER_BASE: usize = SPECTRAL_BASE + ALL_SPECTRALS.len();
const BOOSTER_BASE: usize = VOUCHER_BASE + ALL_VOUCHERS.len();
const ENHANCED_BASE: usize = BOOSTER_BASE + ALL_BOOSTERS.len();
pub const CENTER_COUNT: usize = ENHANCED_BASE + 1 + Enhancement::POOL.len();

impl Center {
	#[inline]
	#[must_use]
	pub fn index(self) -> usize {
		match self {
			Center::Joker(j) => j as usize,
			Center::Tarot(t) => TAROT_BASE + t as usize,
			Center::Planet(p) => PLANET_BASE + p as usize,
			Center::Spectral(s) => SPECTRAL_BASE + s as usize,
			Center::Voucher(v) => VOUCHER_BASE + v as usize,
			Center::Booster(b) => BOOSTER_BASE + b as usize,
			Center::Enhanced(e) => ENHANCED_BASE + e as usize,
		}
	}

	/// Every center except playing-card enhancements, in `Center::index` order.
	pub fn all_cards() -> impl Iterator<Item = Center> {
		ALL_JOKERS
			.iter()
			.map(|&j| Center::Joker(j))
			.chain(ALL_TAROTS.iter().map(|&t| Center::Tarot(t)))
			.chain(ALL_PLANETS.iter().map(|&p| Center::Planet(p)))
			.chain(ALL_SPECTRALS.iter().map(|&s| Center::Spectral(s)))
			.chain(ALL_VOUCHERS.iter().map(|&v| Center::Voucher(v)))
			.chain(ALL_BOOSTERS.iter().map(|&b| Center::Booster(b)))
	}

	#[must_use]
	pub fn key(self) -> &'static str {
		match self {
			Center::Joker(j) => j.key(),
			Center::Tarot(t) => t.key(),
			Center::Planet(p) => p.key(),
			Center::Spectral(s) => s.key(),
			Center::Voucher(v) => v.key(),
			Center::Booster(b) => b.key(),
			Center::Enhanced(e) => e.key(),
		}
	}
}

/// Fixed-size bitset over [`Center`]s.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CenterSet([u64; CENTER_COUNT.div_ceil(64)]);

impl CenterSet {
	#[inline]
	#[must_use]
	pub fn contains(&self, c: Center) -> bool {
		let i = c.index();
		self.0[i >> 6] >> (i & 63) & 1 == 1
	}

	#[inline]
	pub fn insert(&mut self, c: Center) {
		let i = c.index();
		self.0[i >> 6] |= 1 << (i & 63);
	}

	#[inline]
	pub fn remove(&mut self, c: Center) {
		let i = c.index();
		self.0[i >> 6] &= !(1 << (i & 63));
	}
}

/// Base chips required for an ante (`get_blind_amount`).
#[must_use]
#[expect(
	clippy::many_single_char_names,
	reason = "`a`, `b`, `c`, `d` and `k` are the names in `get_blind_amount`"
)]
pub fn blind_amount(ante: i32, scaling: u8) -> f64 {
	const S1: [f64; 8] = [
		300.0, 800.0, 2000.0, 5000.0, 11000.0, 20000.0, 35000.0, 50000.0,
	];
	const S2: [f64; 8] = [
		300.0, 900.0, 2600.0, 8000.0, 20000.0, 36000.0, 60000.0, 100_000.0,
	];
	const S3: [f64; 8] = [
		300.0, 1000.0, 3200.0, 9000.0, 25000.0, 60000.0, 110_000.0, 200_000.0,
	];
	let amounts = match scaling {
		2 => &S2,
		3 => &S3,
		_ => &S1,
	};
	if ante < 1 {
		return 100.0;
	}
	if let Some(&amount) = usize::try_from(ante - 1).ok().and_then(|i| amounts.get(i)) {
		return amount;
	}
	let k = 0.75f64;
	let (a, b, c, d) = (
		amounts[7],
		1.6f64,
		f64::from(ante - 8),
		1.0 + 0.2 * f64::from(ante - 8),
	);
	let mut amount = (a * (b + (k * c).powf(d)).powf(c)).floor();
	amount -= crate::rng::lua_mod(amount, 10f64.powf((amount.log10() - 1.0).floor()));
	amount
}
