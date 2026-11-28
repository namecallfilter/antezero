//! Playing cards (`c_base` or an `m_*` enhancement on a front).

use super::item::CardCore;
use crate::data::{Center, Edition, Enhancement, Rank, Seal, Suit};

/// How `G.hand` sorts (`G.hand.config.sort`: 'desc' by rank or 'suit desc').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HandSort {
	Rank,
	Suit,
}

/// Which of `Card:is_suit`'s rule sets applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SuitTest {
	/// A scoring effect asks: a debuffed card has no suit.
	Scoring,
	/// The debuff is looked past (`bypass_debuff`), as the blind's own test does.
	IgnoreDebuff,
	/// Flush detection (`flush_calc`): a debuffed Wild card has only its own suit.
	Flush,
}

/// Cerulean Bell's hold on a hand card: `ability.forced_selection`, and for
/// a card that has it, whether the card is highlighted. In the order of how
/// firmly the card is held, which `min` and `max` on it rely on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Forced {
	#[default]
	No,
	/// The card has the flag but is not highlighted: a copy of a forced card
	/// (Cryptid) that no selection has held since. The game does not ask
	/// for it.
	Flagged,
	/// The card has the flag and is highlighted. The game refuses to
	/// unhighlight it (`CardArea:remove_from_highlighted`), so every
	/// selection holds it.
	Locked,
}

impl Forced {
	/// Whether the card has the flag.
	#[must_use]
	pub fn is_flagged(self) -> bool {
		self != Forced::No
	}

	/// Whether every selection must hold the card.
	#[must_use]
	pub fn is_locked(self) -> bool {
		self == Forced::Locked
	}
}

/// A playing card.
#[derive(Clone, Debug, PartialEq)]
pub struct PlayingCard {
	pub core: CardCore,
	/// The base suit and rank (`card.base`).
	pub suit: Suit,
	pub rank: Rank,
	/// Suit the card had when its base was first set (`suit_nominal_original`).
	pub orig_suit: Suit,
	pub enhancement: Enhancement,
	pub seal: Option<Seal>,
	/// `ability.perma_bonus`: permanent chip bonus (Hiker).
	pub perma_bonus: f64,
	pub forced: Forced,
	pub played_this_ante: bool,
	pub wheel_flipped: bool,
	pub times_played: u32,
	pub shattered: bool,
	pub destroyed: bool,
	pub lucky_trigger: bool,
	/// Drained by a Vampire this hand (until its juice event runs).
	pub vampired: bool,
}

impl PlayingCard {
	/// A plain card of `suit` and `rank` (`Card:init` with a front and
	/// `c_base`), at no cost.
	#[must_use]
	pub fn new(id: u32, suit: Suit, rank: Rank) -> PlayingCard {
		PlayingCard {
			core: CardCore::new(id),
			suit,
			rank,
			orig_suit: suit,
			enhancement: Enhancement::Base,
			seal: None,
			perma_bonus: 0.0,
			forced: Forced::No,
			played_this_ante: false,
			wheel_flipped: false,
			times_played: 0,
			shattered: false,
			destroyed: false,
			lucky_trigger: false,
			vampired: false,
		}
	}

	/// `Card:set_base(card)` on an existing playing card (rebuilds `base`,
	/// which also resets `times_played`). The blind's debuff is the caller's
	/// to re-evaluate (`round::CardDebuff::apply`).
	pub(crate) fn set_base(&mut self, suit: Suit, rank: Rank) {
		self.suit = suit;
		self.rank = rank;
		self.times_played = 0;
	}

	#[inline]
	#[must_use]
	pub fn is_stone(&self) -> bool {
		self.enhancement == Enhancement::Stone
	}

	/// The center this card counts as.
	#[must_use]
	pub fn center(&self) -> Center {
		Center::Enhanced(self.enhancement)
	}

	/// `config.center_key`.
	#[must_use]
	pub fn key(&self) -> &'static str {
		self.enhancement.key()
	}

	/// `Card:get_id()`, the rank the game's rank tests see; `None` for Stone
	/// cards (which get a fresh random negative id on every call in the
	/// game, so never compare equal).
	#[inline]
	#[must_use]
	pub fn effective_rank(&self) -> Option<Rank> {
		if self.is_stone() {
			None
		} else {
			Some(self.rank)
		}
	}

	/// `Card:get_id() == r`.
	#[inline]
	#[must_use]
	pub fn has_rank(&self, r: Rank) -> bool {
		self.effective_rank() == Some(r)
	}

	/// `Card:is_face()`: a debuffed card is no face card.
	#[inline]
	#[must_use]
	pub fn is_face(&self, pareidolia: bool) -> bool {
		!self.core.debuff && self.is_face_from_boss(pareidolia)
	}

	/// `Card:is_face(from_boss)` with `from_boss` set: the blind's own test,
	/// which looks past the debuff.
	#[inline]
	#[must_use]
	pub fn is_face_from_boss(&self, pareidolia: bool) -> bool {
		pareidolia || self.effective_rank().is_some_and(Rank::is_face)
	}

	/// `Card:is_suit(suit, bypass_debuff, flush_calc)`.
	#[inline]
	#[must_use]
	pub fn is_suit(&self, suit: Suit, test: SuitTest, smeared: bool) -> bool {
		if test == SuitTest::Flush {
			if self.is_stone() {
				return false;
			}
			if self.enhancement == Enhancement::Wild && !self.core.debuff {
				return true;
			}
		} else {
			if self.core.debuff && test != SuitTest::IgnoreDebuff {
				return false;
			}
			if self.is_stone() {
				return false;
			}
			if self.enhancement == Enhancement::Wild {
				return true;
			}
		}
		if smeared && self.suit.is_red() == suit.is_red() {
			return true;
		}
		self.suit == suit
	}

	/// Blackboard's test: a Club or a Spade, as a flush counts suits.
	#[must_use]
	pub(crate) fn is_black(&self, smeared: bool) -> bool {
		self.is_suit(Suit::Clubs, SuitTest::Flush, smeared)
			|| self.is_suit(Suit::Spades, SuitTest::Flush, smeared)
	}

	/// Stand-in for the UI node id behind `unique_val`; see
	/// [`PlayingCard::nominal`].
	#[inline]
	#[must_use]
	pub fn unique_val(&self) -> f64 {
		1.0 - (f64::from(self.core.id) * UNIQUE_ID_SCALE) / 1_603_301.0
	}

	/// `Card:get_nominal(mod)`.
	///
	/// The final tie-breaker (`unique_val`) derives from the global UI node
	/// counter in the real game, which depends on UI activity (hovering etc.)
	/// and is therefore not reproducible. We model it as monotone in creation
	/// order (the harness applies the same model to the oracle).
	#[inline]
	#[must_use]
	pub fn nominal(&self, sort: HandSort) -> f64 {
		self.sort_key(sort) + 0.000_001 * self.unique_val()
	}

	/// What `sort` orders the hand by of what the player sees of the card:
	/// not the suit it was dealt with (a tie-break), nor anything of a Stone
	/// card, whose rank and suit do not show (Stone cards sort last).
	pub(crate) fn shown_sort_key(&self, sort: HandSort) -> f64 {
		if self.is_stone() {
			return f64::NEG_INFINITY;
		}
		let mult = if sort == HandSort::Suit { 1000.0 } else { 1.0 };
		self.rank.nominal() + self.suit.nominal() * mult + self.rank.face_nominal()
	}

	/// What `sort` orders the hand by, without the tie-break between cards
	/// that look the same.
	fn sort_key(&self, sort: HandSort) -> f64 {
		let mut mult = if sort == HandSort::Suit { 1000.0 } else { 1.0 };
		if self.is_stone() {
			mult = -1000.0;
		}
		self.rank.nominal()
			+ self.suit.nominal() * mult
			+ self.orig_suit.nominal_original() * 0.0001 * mult
			+ self.rank.face_nominal()
	}

	/// `Card:get_chip_bonus()`.
	#[inline]
	#[must_use]
	pub fn chip_bonus(&self) -> f64 {
		if self.core.debuff {
			return 0.0;
		}
		if self.is_stone() {
			return self.enhancement.bonus() + self.perma_bonus;
		}
		self.rank.nominal() + self.enhancement.bonus() + self.perma_bonus
	}

	/// The card's `P_CARDS` key, e.g. `H_Q`.
	#[must_use]
	pub fn card_key(&self) -> String {
		format!(
			"{}_{}",
			char::from(self.suit.letter()),
			char::from(self.rank.letter())
		)
	}
}

/// Node ids grow ~this much per created card in a typical session.
pub const UNIQUE_ID_SCALE: f64 = 50.0;

/// What a player sees of a playing card's face (never its id).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct CardFace {
	/// 0 for a Stone card, which shows no suit or rank.
	pub(crate) suit: u8,
	/// The rank id (0 for a Stone card).
	pub(crate) rank: u8,
	pub(crate) enhancement: u8,
	/// [`Edition::id`].
	pub(crate) edition: u8,
	/// 0 for none, else 1 + the seal.
	pub(crate) seal: u8,
	/// The bits of Hiker's bonus.
	pub(crate) perma_bonus: u64,
}

impl CardFace {
	pub(crate) fn of(c: &PlayingCard) -> CardFace {
		let (suit, rank) = if c.is_stone() {
			(0, 0)
		} else {
			(c.suit as u8, c.rank.id())
		};
		CardFace {
			suit,
			rank,
			enhancement: c.enhancement as u8,
			edition: Edition::id(c.core.edition),
			seal: c.seal.map_or(0, |s| s as u8 + 1),
			perma_bonus: c.perma_bonus.to_bits(),
		}
	}
}

/// The visible content of a playing card: the sort key of the canonical
/// pool order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct CanonKey {
	pub(crate) face: CardFace,
	pub(crate) played_this_ante: bool,
	pub(crate) times_played: u32,
}

pub(crate) fn canon_key(c: &PlayingCard) -> CanonKey {
	CanonKey {
		face: CardFace::of(c),
		played_this_ante: c.played_this_ante,
		times_played: c.times_played,
	}
}
