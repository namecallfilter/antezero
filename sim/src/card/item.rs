//! Non-playing cards (jokers, consumables, vouchers and booster packs) and the
//! fields every card shares ([`CardCore`]).

use crate::data::{Booster, Center, Edition, HandType, Joker, Planet, Spectral, Tarot, Voucher};

/// What a card that is not a playing card is: its center.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
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

	#[must_use]
	pub fn center(self) -> Center {
		match self {
			Kind::Joker(j) => Center::Joker(j),
			Kind::Tarot(t) => Center::Tarot(t),
			Kind::Planet(p) => Center::Planet(p),
			Kind::Spectral(s) => Center::Spectral(s),
			Kind::Voucher(v) => Center::Voucher(v),
			Kind::Booster(b) => Center::Booster(b),
		}
	}

	/// `config.center_key`.
	#[must_use]
	pub fn key(self) -> &'static str {
		self.center().key()
	}
}

/// What every card has, playing cards included: the parts of Lua's `Card`
/// that the game's functions on any card read and write.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CardCore {
	/// `sort_id`: creation counter, the game's tie-breaker for sampling.
	pub id: u32,
	pub edition: Option<Edition>,
	pub debuff: bool,
	pub face_down: bool,
	pub base_cost: f64,
	pub extra_cost: f64,
	pub cost: f64,
	pub sell_cost: f64,
	/// `ability.couponed`: free in the shop.
	pub couponed: bool,
	/// `ability.hands_played_at_create`.
	pub hands_played_at_create: f64,
	/// Set while a card is "added to deck" (passive effects applied).
	pub added_to_deck: bool,
}

impl CardCore {
	#[must_use]
	pub fn new(id: u32) -> CardCore {
		CardCore {
			id,
			edition: None,
			debuff: false,
			face_down: false,
			base_cost: 0.0,
			extra_cost: 0.0,
			cost: 0.0,
			sell_cost: 0.0,
			couponed: false,
			hands_played_at_create: 0.0,
			added_to_deck: false,
		}
	}
}

/// The mutable state of a joker, consumable, voucher or booster mirroring
/// the relevant parts of Lua's `card.ability`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ability {
	pub mult: f64,
	pub x_mult: f64,
	pub extra_value: f64,
	/// `extra.chips` (Runner, Ice Cream, Square, Castle, Wee Joker).
	pub chips: f64,
	/// `extra.dollars` (Rocket).
	pub dollars: f64,
	/// The `extra` that changes during play: Seltzer's scalar `extra` (hands
	/// left) or Turtle Bean's `extra.h_size`.
	pub extra: f64,
	pub invis_rounds: f64,
	pub yorick_discards: f64,
	pub caino_xmult: f64,
	pub loyalty_remaining: f64,
	pub to_do: Option<HandType>,
	pub sticker: Option<Sticker>,
	pub rental: bool,
	// Values the game recomputes every frame in `Card:update`.
	pub steel_tally: f64,
	pub stone_tally: f64,
	pub nine_tally: f64,
	pub driver_tally: f64,
	/// Temperance's `ability.money`.
	pub money: f64,
}

impl Default for Ability {
	fn default() -> Self {
		Ability {
			mult: 0.0,
			x_mult: 1.0,
			extra_value: 0.0,
			chips: 0.0,
			dollars: 0.0,
			extra: 0.0,
			invis_rounds: 0.0,
			yorick_discards: 0.0,
			caino_xmult: 0.0,
			loyalty_remaining: 0.0,
			to_do: None,
			sticker: None,
			rental: false,
			steel_tally: 0.0,
			stone_tally: 0.0,
			nine_tally: 0.0,
			driver_tally: 0.0,
			money: 0.0,
		}
	}
}

impl Ability {
	/// `ability.eternal`.
	#[inline]
	#[must_use]
	pub fn eternal(&self) -> bool {
		self.sticker == Some(Sticker::Eternal)
	}

	/// `ability.perishable`.
	#[inline]
	#[must_use]
	pub fn perishable(&self) -> bool {
		matches!(self.sticker, Some(Sticker::Perishable { .. }))
	}

	/// `ability.perish_tally`: the rounds a perishable card has left, 0 for
	/// any other.
	#[inline]
	#[must_use]
	pub fn perish_tally(&self) -> f64 {
		match self.sticker {
			Some(Sticker::Perishable { tally }) => tally,
			_ => 0.0,
		}
	}
}

/// The two stake stickers a joker can't have together: the game keeps them in
/// two flags, `ability.eternal` and `ability.perishable`, which
/// `Card:set_eternal` and `Card:set_perishable` never set together. Rental
/// combines with either and is [`Ability::rental`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sticker {
	/// It cannot be sold or destroyed.
	Eternal,
	/// It is debuffed once `tally` (`ability.perish_tally`) reaches 0, a
	/// round at a time.
	Perishable { tally: f64 },
}

/// A joker, consumable, voucher or booster pack: every card that is not a
/// playing card.
#[derive(Clone, Debug, PartialEq)]
pub struct Card {
	pub core: CardCore,
	pub kind: Kind,
	pub ab: Ability,
	pub getting_sliced: bool,
	pub temp_edition: bool,
	/// The shop slot of a booster the shop offers (`booster_pos`, from 0).
	pub booster_slot: Option<u8>,
	pub queue_negative_removal: bool,
	/// Marked when an Ankh is used: a joker that is not eternal and not the
	/// one copied (the game's `deletable_jokers` less `chosen_joker`), which
	/// the Ankh's destroy event dissolves.
	pub ankh_deletable: bool,
}

impl Card {
	/// A fresh card of `kind` with the `sort_id` `id`: no edition, no
	/// stickers, no cost yet.
	#[must_use]
	pub fn new(id: u32, kind: Kind) -> Card {
		Card {
			core: CardCore::new(id),
			kind,
			ab: Ability::default(),
			getting_sliced: false,
			temp_edition: false,
			booster_slot: None,
			queue_negative_removal: false,
			ankh_deletable: false,
		}
	}

	/// `Card:set_eternal(true)`: only for a joker that allows it and is not
	/// perishable.
	pub(crate) fn make_eternal(&mut self) {
		if !self.ab.perishable() {
			let compat = self.joker().is_some_and(|j| j.def().eternal_compat);
			self.ab.sticker = compat.then_some(Sticker::Eternal);
		}
	}

	/// `Card:set_perishable()` with `rounds` to go: only for a joker that
	/// allows it and is not eternal.
	pub(crate) fn make_perishable(&mut self, rounds: f64) {
		if !self.ab.eternal() {
			let compat = self.joker().is_some_and(|j| j.def().perishable_compat);
			self.ab.sticker = compat.then_some(Sticker::Perishable { tally: rounds });
		}
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

	/// Whether this is a non-debuffed `j` (what `find_joker` finds).
	#[inline]
	#[must_use]
	pub fn is_active_joker(&self, j: Joker) -> bool {
		self.is_joker(j) && !self.core.debuff
	}

	/// The center this card counts as for `used_jokers` bookkeeping.
	#[must_use]
	pub fn center(&self) -> Center {
		self.kind.center()
	}

	/// `config.center_key`.
	#[must_use]
	pub fn key(&self) -> &'static str {
		self.kind.key()
	}

	/// `ability.name` (used by name-based lookups such as `find_joker`).
	#[must_use]
	pub fn name(&self) -> &'static str {
		match self.kind {
			Kind::Joker(j) => j.name(),
			Kind::Tarot(t) => t.name(),
			Kind::Planet(p) => p.name(),
			Kind::Spectral(s) => s.name(),
			Kind::Voucher(v) => v.name(),
			Kind::Booster(b) => b.name(),
		}
	}
}

/// Whether a non-debuffed `j` is among `jokers` (`next(find_joker(name))`).
#[inline]
pub(crate) fn has_joker(jokers: &[Card], j: Joker) -> bool {
	jokers.iter().any(|c| c.is_active_joker(j))
}
