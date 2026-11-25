//! Prototype definitions, compiled lookup tables and [`Center`] indexing over
//! the generated tables in `data_gen.rs`.

use super::primitives::{Enhancement, HandType, PackKind};
use crate::data_gen::{
	ALL_BACKS, ALL_BLINDS, ALL_BOOSTERS, ALL_JOKERS, ALL_PLANETS, ALL_SPECTRALS, ALL_TAGS,
	ALL_TAROTS, ALL_VOUCHERS, BACK_DEFS, BLIND_DEFS, BOOSTER_DEFS, Back, BlindKind, Booster,
	JOKER_DEFS, Joker, PLANET_DEFS, Planet, SPECTRAL_DEFS, Spectral, TAG_DEFS, TAROT_DEFS, TagKind,
	Tarot, VOUCHER_DEFS, Voucher,
};

/// A joker's prototype (`G.P_CENTERS.j_*`).
#[derive(Clone, Copy, Debug)]
pub struct JokerDef {
	pub key: &'static str,
	pub name: &'static str,
	/// The game's rarity code, 1 to 4 (see [`Joker::rarity`]).
	pub rarity: u8,
	pub cost: u32,
	pub perishable_compat: bool,
	pub eternal_compat: bool,
	/// The joker is offered only while a playing card has this enhancement.
	pub enhancement_gate: Option<Enhancement>,
	/// `no_pool_flag` and `yes_pool_flag` (see [`Joker::pool_flag`]).
	pub no_pool_flag: Option<&'static str>,
	pub yes_pool_flag: Option<&'static str>,
}

/// A tarot's, planet's or spectral card's prototype.
#[derive(Clone, Copy, Debug)]
pub struct ConsumableDef {
	pub key: &'static str,
	pub name: &'static str,
	pub cost: u32,
	/// A planet's poker hand (`config.hand_type`; see [`Planet::hand`]).
	pub hand_type: Option<HandType>,
	/// `config.max_highlighted` and `config.min_highlighted`, 0 where the
	/// game has none.
	pub max_highlighted: u8,
	pub min_highlighted: u8,
}

/// A voucher's prototype.
#[derive(Clone, Copy, Debug)]
pub struct VoucherDef {
	pub key: &'static str,
	pub name: &'static str,
	pub cost: u32,
	/// The voucher that must be redeemed first.
	pub requires: Option<Voucher>,
	/// `config.extra`: the voucher's amount (a discount percent, a rate, a
	/// number of slots, ...).
	pub extra: f64,
}

/// A booster pack's prototype.
#[derive(Clone, Copy, Debug)]
pub struct BoosterDef {
	pub key: &'static str,
	pub name: &'static str,
	pub kind: PackKind,
	pub cost: u32,
	/// The pack's weight in the shop's booster roll.
	pub weight: f64,
	/// `config.extra`: the cards the pack shows.
	pub extra: u8,
	/// `config.choose`: the cards one may take from it.
	pub choose: u8,
}

/// A tag's prototype.
#[derive(Clone, Copy, Debug)]
pub struct TagDef {
	pub key: &'static str,
	pub name: &'static str,
	pub min_ante: i32,
}

/// A blind's prototype.
#[derive(Clone, Copy, Debug)]
pub struct BlindDef {
	pub key: &'static str,
	pub name: &'static str,
	/// The reward for beating it.
	pub dollars: i32,
	/// Its chips as a multiple of the ante's base amount (`get_blind_amount`).
	pub mult: f64,
	/// `boss` and `boss.showdown` (see [`BlindKind::boss`]).
	pub boss: bool,
	pub showdown: bool,
	pub min_ante: i32,
}

/// A deck's prototype.
#[derive(Clone, Copy, Debug)]
pub struct BackDef {
	pub key: &'static str,
	pub name: &'static str,
}

macro_rules! def_accessors {
	($ty:ident, $all:ident, $defs:ident, $def:ty $(, { $($extra:item)* })?) => {
		impl $ty {
			/// The prototype.
			#[inline]
			#[must_use]
			pub fn def(self) -> &'static $def {
				&$defs[self as usize]
			}

			/// The prototype's key, e.g. `j_joker`.
			#[inline]
			#[must_use]
			pub fn key(self) -> &'static str {
				self.def().key
			}

			/// The name the game's code looks it up by.
			#[inline]
			#[must_use]
			pub fn name(self) -> &'static str {
				self.def().name
			}

			/// The variant whose key is `k`.
			#[must_use]
			pub fn from_key(k: &str) -> Option<$ty> {
				$all.iter().copied().find(|x| x.key() == k)
			}

			$($($extra)*)?
		}
	};
}

def_accessors!(Joker, ALL_JOKERS, JOKER_DEFS, JokerDef, {
	#[inline]
	#[must_use]
	pub fn rarity(self) -> Rarity {
		JOKER_RARITIES[self as usize]
	}

	/// What the joker's place in the shop's pool depends on beyond its
	/// rarity, if anything.
	#[inline]
	#[must_use]
	pub fn pool_flag(self) -> Option<PoolFlag> {
		JOKER_POOL_FLAGS[self as usize]
	}
});
def_accessors!(Tarot, ALL_TAROTS, TAROT_DEFS, ConsumableDef);
def_accessors!(Planet, ALL_PLANETS, PLANET_DEFS, ConsumableDef, {
	/// The poker hand the planet levels up.
	#[inline]
	#[must_use]
	pub fn hand(self) -> HandType {
		PLANET_HANDS[self as usize]
	}

	#[inline]
	#[must_use]
	pub fn for_hand(h: HandType) -> Planet {
		HAND_PLANETS[h as usize]
	}
});
def_accessors!(Spectral, ALL_SPECTRALS, SPECTRAL_DEFS, ConsumableDef);
def_accessors!(Voucher, ALL_VOUCHERS, VOUCHER_DEFS, VoucherDef);
def_accessors!(Booster, ALL_BOOSTERS, BOOSTER_DEFS, BoosterDef, {
	#[inline]
	#[must_use]
	pub fn kind(self) -> PackKind {
		self.def().kind
	}
});
def_accessors!(TagKind, ALL_TAGS, TAG_DEFS, TagDef, {
	/// The first ante the tag can come up in, if it has one (`min_ante`,
	/// which the generated table writes as 0 for `nil`).
	#[inline]
	#[must_use]
	pub fn min_ante(self) -> Option<i32> {
		match self.def().min_ante {
			0 => None,
			ante => Some(ante),
		}
	}
});
def_accessors!(BlindKind, ALL_BLINDS, BLIND_DEFS, BlindDef, {
	/// Which antes the blind can be the boss of; `None` for the small and the
	/// big blind.
	#[inline]
	#[must_use]
	pub fn boss(self) -> Option<Boss> {
		BLIND_BOSSES[self as usize]
	}

	#[inline]
	#[must_use]
	pub fn is_boss(self) -> bool {
		self.boss().is_some()
	}
});
def_accessors!(Back, ALL_BACKS, BACK_DEFS, BackDef, {
	/// Whether the simulator plays this deck: all but the Challenge Deck,
	/// which needs a challenge's rules.
	#[inline]
	#[must_use]
	pub fn is_simulated(self) -> bool {
		self != Back::Challenge
	}

	/// The decks the simulator plays, in the game's order.
	pub fn simulated() -> impl Iterator<Item = Back> {
		ALL_BACKS.into_iter().filter(|b| b.is_simulated())
	}
});

/// A joker's rarity (1 to 4 in the game's data).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Rarity {
	Common = 1,
	Uncommon = 2,
	Rare = 3,
	Legendary = 4,
}

impl Rarity {
	/// The game's rarity number.
	#[must_use]
	pub const fn id(self) -> u8 {
		self as u8
	}
}

/// Each joker's rarity, by joker. Built when the crate compiles: a rarity
/// code the generated table should not have does not build.
const JOKER_RARITIES: [Rarity; ALL_JOKERS.len()] = {
	let mut rarities = [Rarity::Common; ALL_JOKERS.len()];
	let mut i = 0;
	while i < rarities.len() {
		rarities[i] = match JOKER_DEFS[i].rarity {
			1 => Rarity::Common,
			2 => Rarity::Uncommon,
			3 => Rarity::Rare,
			4 => Rarity::Legendary,
			_ => panic!("a joker in the generated table has a rarity outside 1 to 4"),
		};
		i += 1;
	}
	rarities
};

/// A joker's pool flag: the game's only one, `gros_michel_extinct`, which
/// Gros Michel sets when it goes extinct.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolFlag {
	/// Offered only while Gros Michel is not extinct (`no_pool_flag`).
	GrosMichelAlive,
	/// Offered only once Gros Michel is extinct (`yes_pool_flag`).
	GrosMichelExtinct,
}

/// Each joker's pool flag, by joker. Built when the crate compiles: a flag
/// other than `gros_michel_extinct` in the generated table does not build.
const JOKER_POOL_FLAGS: [Option<PoolFlag>; ALL_JOKERS.len()] = {
	const fn is_gros_michel_extinct(flag: &str) -> bool {
		let (flag, known) = (flag.as_bytes(), b"gros_michel_extinct".as_slice());
		if flag.len() != known.len() {
			return false;
		}
		let mut i = 0;
		while i < flag.len() {
			if flag[i] != known[i] {
				return false;
			}
			i += 1;
		}
		true
	}
	let mut flags = [None; ALL_JOKERS.len()];
	let mut i = 0;
	while i < flags.len() {
		flags[i] = match (JOKER_DEFS[i].no_pool_flag, JOKER_DEFS[i].yes_pool_flag) {
			(None, None) => None,
			(Some(flag), None) if is_gros_michel_extinct(flag) => Some(PoolFlag::GrosMichelAlive),
			(None, Some(flag)) if is_gros_michel_extinct(flag) => Some(PoolFlag::GrosMichelExtinct),
			_ => {
				panic!("a joker in the generated table has a pool flag the simulator does not know")
			}
		};
		i += 1;
	}
	flags
};

/// Each planet's poker hand (`config.hand_type`), by planet. Built when
/// the crate compiles: a planet without a hand in the generated table does
/// not build.
const PLANET_HANDS: [HandType; ALL_PLANETS.len()] = {
	let mut hands = [HandType::HighCard; ALL_PLANETS.len()];
	let mut i = 0;
	while i < hands.len() {
		assert!(
			ALL_PLANETS[i] as usize == i,
			"`ALL_PLANETS` is in discriminant order"
		);
		hands[i] = match PLANET_DEFS[i].hand_type {
			Some(hand) => hand,
			None => panic!("a planet in the generated table has no poker hand"),
		};
		i += 1;
	}
	hands
};

/// Each poker hand's planet, by hand: the inverse of [`PLANET_HANDS`].
const HAND_PLANETS: [Planet; HandType::ALL.len()] = {
	let mut found = [None; HandType::ALL.len()];
	let mut i = 0;
	while i < ALL_PLANETS.len() {
		found[PLANET_HANDS[i] as usize] = Some(ALL_PLANETS[i]);
		i += 1;
	}
	let mut planets = [Planet::Pluto; HandType::ALL.len()];
	let mut i = 0;
	while i < planets.len() {
		planets[i] = match found[i] {
			Some(planet) => planet,
			None => panic!("a poker hand has no planet in the generated table"),
		};
		i += 1;
	}
	planets
};

/// The antes a boss blind can come up in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Boss {
	/// The antes that are not a multiple of the winning ante.
	Regular,
	/// The final boss of every multiple of the winning ante (`boss.showdown`).
	Showdown,
}

/// Each blind's boss kind, by blind. Built when the crate compiles: a
/// showdown blind that is not a boss in the generated table does not build.
const BLIND_BOSSES: [Option<Boss>; ALL_BLINDS.len()] = {
	let mut bosses = [None; ALL_BLINDS.len()];
	let mut i = 0;
	while i < bosses.len() {
		bosses[i] = match (BLIND_DEFS[i].boss, BLIND_DEFS[i].showdown) {
			(false, false) => None,
			(true, false) => Some(Boss::Regular),
			(true, true) => Some(Boss::Showdown),
			(false, true) => panic!("a blind in the generated table is a showdown but no boss"),
		};
		i += 1;
	}
	bosses
};

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
/// The centers of cards other than playing cards come first: this many.
pub(crate) const ENHANCED_BASE: usize = BOOSTER_BASE + ALL_BOOSTERS.len();
pub const CENTER_COUNT: usize = ENHANCED_BASE + Enhancement::ALL.len();

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
