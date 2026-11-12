//! The universal card object (playing cards, jokers, consumables, vouchers and
//! booster packs are all `Card`s in Balatro).

use crate::data::{
	Booster, Center, Edition, Enhancement, HandType, JACK, Joker, KING, Planet, QUEEN, Rank, Seal,
	Spectral, Suit, Tarot, Voucher, rank_char, rank_face_nominal, rank_nominal,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
	/// A playing card (`c_base` or an `m_*` enhancement, see [`Card::enhancement`]).
	Playing,
	Joker(Joker),
	Tarot(Tarot),
	Planet(Planet),
	Spectral(Spectral),
	Voucher(Voucher),
	Booster(Booster),
}

impl Kind {
	#[inline]
	#[must_use]
	pub fn is_consumable(self) -> bool {
		matches!(self, Kind::Tarot(_) | Kind::Planet(_) | Kind::Spectral(_))
	}

	#[inline]
	#[must_use]
	pub fn joker(self) -> Option<Joker> {
		if let Kind::Joker(j) = self {
			Some(j)
		} else {
			None
		}
	}

	/// The center this kind counts as (playing cards: their enhancement).
	#[must_use]
	pub fn center(self, enhancement: Enhancement) -> Center {
		match self {
			Kind::Playing => Center::Enhanced(enhancement),
			Kind::Joker(j) => Center::Joker(j),
			Kind::Tarot(t) => Center::Tarot(t),
			Kind::Planet(p) => Center::Planet(p),
			Kind::Spectral(s) => Center::Spectral(s),
			Kind::Voucher(v) => Center::Voucher(v),
			Kind::Booster(b) => Center::Booster(b),
		}
	}

	/// `config.center_key` (playing cards: `c_base`).
	#[must_use]
	pub fn key(self) -> &'static str {
		self.center(Enhancement::Base).key()
	}
}

/// Per-card mutable state mirroring the relevant parts of Lua's `card.ability`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[expect(
	clippy::struct_excessive_bools,
	reason = "independent flags, one per field of the game's `card.ability`"
)]
pub struct Ability {
	pub mult: f64,
	pub x_mult: f64,
	pub extra_value: f64,
	/// Playing cards: permanent chip bonus (Hiker).
	pub perma_bonus: f64,
	/// `extra.chips` (Runner, Ice Cream, Square, Castle, Wee Joker).
	pub chips: f64,
	/// `extra.dollars` (Rocket).
	pub dollars: f64,
	/// Scalar `extra` that changes during play (Seltzer count) or
	/// `extra.h_size` (Turtle Bean).
	pub remaining: f64,
	pub invis_rounds: f64,
	pub yorick_discards: f64,
	pub caino_xmult: f64,
	pub loyalty_remaining: f64,
	pub to_do: Option<HandType>,
	pub hands_played_at_create: f64,
	pub eternal: bool,
	pub perishable: bool,
	pub perish_tally: f64,
	pub rental: bool,
	pub couponed: bool,
	// Values the game recomputes every frame in `Card:update`.
	pub steel_tally: f64,
	pub stone_tally: f64,
	pub nine_tally: f64,
	pub driver_tally: f64,
	/// Temperance's `ability.money`.
	pub money: f64,
	// Playing card flags.
	pub played_this_ante: bool,
	pub discarded: bool,
	pub forced_selection: bool,
	pub wheel_flipped: bool,
}

impl Default for Ability {
	fn default() -> Self {
		Ability {
			mult: 0.0,
			x_mult: 1.0,
			extra_value: 0.0,
			perma_bonus: 0.0,
			chips: 0.0,
			dollars: 0.0,
			remaining: 0.0,
			invis_rounds: 0.0,
			yorick_discards: 0.0,
			caino_xmult: 0.0,
			loyalty_remaining: 0.0,
			to_do: None,
			hands_played_at_create: 0.0,
			eternal: false,
			perishable: false,
			perish_tally: 0.0,
			rental: false,
			couponed: false,
			steel_tally: 0.0,
			stone_tally: 0.0,
			nine_tally: 0.0,
			driver_tally: 0.0,
			money: 0.0,
			played_this_ante: false,
			discarded: false,
			forced_selection: false,
			wheel_flipped: false,
		}
	}
}

#[derive(Clone, Debug, PartialEq)]
#[expect(
	clippy::struct_excessive_bools,
	reason = "independent flags, one per field of the game's `Card`"
)]
pub struct Card {
	/// `sort_id`: creation counter, the game's tie-breaker for sampling.
	pub id: u32,
	pub kind: Kind,
	// Playing card base.
	pub suit: Suit,
	pub rank: Rank,
	/// Suit the card had when its base was first set (`suit_nominal_original`).
	pub orig_suit: Option<Suit>,
	pub enhancement: Enhancement,
	pub seal: Option<Seal>,
	pub edition: Edition,
	pub debuff: bool,
	pub face_down: bool,
	pub ab: Ability,
	pub base_cost: f64,
	pub extra_cost: f64,
	pub cost: f64,
	pub sell_cost: f64,
	pub times_played: u32,
	// Transient flags used during resolution.
	pub getting_sliced: bool,
	pub shattered: bool,
	pub destroyed: bool,
	pub lucky_trigger: bool,
	/// Drained by a Vampire this hand (until its juice event runs).
	pub vampired: bool,
	pub temp_edition: bool,
	pub shop_voucher: bool,
	/// Booster cards created by the shop remember their slot (`booster_pos`).
	pub booster_pos: u8,
	/// Set while a card is "added to deck" (passive effects applied).
	pub added_to_deck: bool,
	pub queue_negative_removal: bool,
}

impl Card {
	/// `Card:set_eternal` (only for jokers that allow it and are not
	/// perishable).
	pub(crate) fn set_eternal(&mut self, v: bool) {
		self.ab.eternal = false;
		if let Some(j) = self.joker()
			&& j.def().eternal_compat
			&& !self.ab.perishable
		{
			self.ab.eternal = v;
		}
	}

	/// `Card:set_base(card)` on an existing playing card (rebuilds `base`,
	/// which also resets `times_played`). The blind's debuff is the caller's
	/// to re-evaluate (`Game::debuff_hand_card`).
	pub(crate) fn set_base(&mut self, suit: Suit, rank: Rank) {
		self.suit = suit;
		self.rank = rank;
		self.times_played = 0;
		if self.orig_suit.is_none() {
			self.orig_suit = Some(suit);
		}
	}

	#[must_use]
	pub fn new(id: u32, kind: Kind) -> Card {
		Card {
			id,
			kind,
			suit: Suit::Spades,
			rank: 0,
			orig_suit: None,
			enhancement: Enhancement::Base,
			seal: None,
			edition: Edition::None,
			debuff: false,
			face_down: false,
			ab: Ability::default(),
			base_cost: 0.0,
			extra_cost: 0.0,
			cost: 0.0,
			sell_cost: 0.0,
			times_played: 0,
			getting_sliced: false,
			shattered: false,
			destroyed: false,
			lucky_trigger: false,
			vampired: false,
			temp_edition: false,
			shop_voucher: false,
			booster_pos: 0,
			added_to_deck: false,
			queue_negative_removal: false,
		}
	}

	#[inline]
	#[must_use]
	pub fn is_playing(&self) -> bool {
		matches!(self.kind, Kind::Playing)
	}

	#[inline]
	#[must_use]
	pub fn joker(&self) -> Option<Joker> {
		self.kind.joker()
	}

	#[inline]
	#[must_use]
	pub fn is_joker(&self, j: Joker) -> bool {
		self.kind == Kind::Joker(j)
	}

	#[inline]
	#[must_use]
	pub fn is_stone(&self) -> bool {
		self.is_playing() && self.enhancement == Enhancement::Stone
	}

	/// The center this card counts as for `used_jokers` bookkeeping.
	#[must_use]
	pub fn center(&self) -> Center {
		self.kind.center(self.enhancement)
	}

	/// `config.center_key`.
	#[must_use]
	pub fn key(&self) -> &'static str {
		self.center().key()
	}

	/// `ability.name` (used by name-based lookups such as `find_joker`).
	#[must_use]
	pub fn name(&self) -> &'static str {
		match self.kind {
			Kind::Playing => match self.enhancement {
				Enhancement::Base => "Default Base",
				Enhancement::Bonus => "Bonus",
				Enhancement::Mult => "Mult",
				Enhancement::Wild => "Wild Card",
				Enhancement::Glass => "Glass Card",
				Enhancement::Steel => "Steel Card",
				Enhancement::Stone => "Stone Card",
				Enhancement::Gold => "Gold Card",
				Enhancement::Lucky => "Lucky Card",
			},
			Kind::Joker(j) => j.name(),
			Kind::Tarot(t) => t.name(),
			Kind::Planet(p) => p.name(),
			Kind::Spectral(s) => s.name(),
			Kind::Voucher(v) => v.name(),
			Kind::Booster(b) => b.name(),
		}
	}

	// ---------------------------------------------------------------
	// Playing-card queries (card.lua)
	// ---------------------------------------------------------------

	/// `Card:get_id()`; `None` for Stone cards (which get a fresh random
	/// negative id on every call in the game, so never compare equal).
	#[inline]
	#[must_use]
	pub fn get_id(&self) -> Option<Rank> {
		if self.is_stone() {
			None
		} else {
			Some(self.rank)
		}
	}

	#[inline]
	#[must_use]
	pub fn id_is(&self, r: Rank) -> bool {
		self.get_id() == Some(r)
	}

	/// `Card:is_face(from_boss)`.
	#[inline]
	#[must_use]
	pub fn is_face(&self, from_boss: bool, pareidolia: bool) -> bool {
		if self.debuff && !from_boss {
			return false;
		}
		match self.get_id() {
			Some(JACK | QUEEN | KING) => true,
			_ => pareidolia,
		}
	}

	/// `Card:is_suit(suit, bypass_debuff, flush_calc)`.
	#[inline]
	#[must_use]
	pub fn is_suit(
		&self, suit: Suit, bypass_debuff: bool, flush_calc: bool, smeared: bool,
	) -> bool {
		if flush_calc {
			if self.is_stone() {
				return false;
			}
			if self.enhancement == Enhancement::Wild && !self.debuff {
				return true;
			}
		} else {
			if self.debuff && !bypass_debuff {
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

	/// Stand-in for the UI node id behind `unique_val`; see [`Card::nominal`].
	#[inline]
	#[must_use]
	pub fn unique_val(&self) -> f64 {
		1.0 - (f64::from(self.id) * UNIQUE_ID_SCALE) / 1_603_301.0
	}

	/// `Card:get_nominal(mod)`.
	///
	/// The final tie-breaker (`unique_val`) derives from the global UI node
	/// counter in the real game, which depends on UI activity (hovering etc.)
	/// and is therefore not reproducible. We model it as monotone in creation
	/// order (the harness applies the same model to the oracle).
	#[inline]
	#[must_use]
	pub fn nominal(&self, suit_mode: bool) -> f64 {
		let mut mult = if suit_mode { 1000.0 } else { 1.0 };
		if self.is_stone() {
			mult = -1000.0;
		}
		let orig = self.orig_suit.map_or(0.0, Suit::nominal_original);
		rank_nominal(self.rank)
			+ self.suit.nominal() * mult
			+ orig * 0.0001 * mult
			+ rank_face_nominal(self.rank)
			+ 0.000_001 * self.unique_val()
	}

	/// `Card:get_chip_bonus()`.
	#[inline]
	#[must_use]
	pub fn chip_bonus(&self) -> f64 {
		if self.debuff {
			return 0.0;
		}
		if self.is_stone() {
			return self.enhancement.bonus() + self.ab.perma_bonus;
		}
		rank_nominal(self.rank) + self.enhancement.bonus() + self.ab.perma_bonus
	}

	/// The playing card's `P_CARDS` key, e.g. `H_Q`.
	#[must_use]
	pub fn card_key(&self) -> String {
		format!(
			"{}_{}",
			self.suit.letter() as char,
			rank_char(self.rank) as char
		)
	}
}

/// Node ids grow ~this much per created card in a typical session.
pub const UNIQUE_ID_SCALE: f64 = 50.0;
