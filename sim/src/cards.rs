//! Card construction (`Card:init` / `set_ability` / `set_base`), costs,
//! editions, deck membership (`add_to_deck` / `remove_from_deck` /
//! `remove`), debuffs and resource easing.

use crate::blind;
use crate::card::{Ability, Card, Kind};
use crate::data::{
	ALL_BOOSTERS, Center, Edition, Enhancement, HandType, Joker, PackKind, Rank, Suit,
};
use crate::game::{Area, Game, Limit};
use crate::rng::Key;

// -----------------------------------------------------------------
// Costs
// -----------------------------------------------------------------

/// `Card:set_cost()`; `area` is the card's current area (for coupons).
pub(crate) fn set_cost(g: &Game, c: &mut Card, area: Option<Area>) {
	price(c, area, g.discount_percent, g.has_joker(Joker::Astronomer));
}

/// `Card:set_cost()` on the `i`-th card of `area`.
pub(crate) fn reprice(g: &mut Game, area: Area, i: usize) {
	let (discount_percent, astronomer) = (g.discount_percent, g.has_joker(Joker::Astronomer));
	price(
		&mut g.area_mut(area)[i],
		Some(area),
		discount_percent,
		astronomer,
	);
}

/// Re-runs `set_cost` on every card in existence (the `G.I.CARD` loops).
pub(crate) fn set_cost_all(g: &mut Game) {
	let (discount_percent, astronomer) = (g.discount_percent, g.has_joker(Joker::Astronomer));
	let areas = [
		(&mut g.jokers, Area::Jokers),
		(&mut g.consumables, Area::Consumables),
		(&mut g.shop_jokers, Area::ShopJokers),
		(&mut g.shop_vouchers, Area::ShopVouchers),
		(&mut g.shop_boosters, Area::ShopBoosters),
		(&mut g.pack_cards, Area::PackCards),
		(&mut g.deck, Area::Deck),
		(&mut g.hand, Area::Hand),
		(&mut g.play, Area::Play),
		(&mut g.discard, Area::Discard),
	];
	for (cards, area) in areas {
		for c in cards {
			price(c, Some(area), discount_percent, astronomer);
		}
	}
}

// -----------------------------------------------------------------
// Card construction (Card:init / set_ability / set_base)
// -----------------------------------------------------------------

/// The part of `Card:set_ability` that is the same for every center: a fresh
/// ability table, which keeps only `perma_bonus` and `forced_selection` (so
/// e.g. `played_this_ante` is lost), and the center's base cost.
fn reset_ability(c: &mut Card, hands_played: f64) {
	c.ab = Ability {
		hands_played_at_create: hands_played,
		perma_bonus: c.ab.perma_bonus,
		forced_selection: c.ab.forced_selection,
		..Ability::default()
	};
	c.base_cost = match c.kind {
		Kind::Playing => 1.0,
		Kind::Joker(j) => f64::from(j.def().cost),
		Kind::Tarot(t) => f64::from(t.def().cost),
		Kind::Planet(p) => f64::from(p.def().cost),
		Kind::Spectral(s) => f64::from(s.def().cost),
		Kind::Voucher(v) => f64::from(v.def().cost),
		Kind::Booster(b) => f64::from(b.def().cost),
	};
}

/// Initial ability values for a center (`Card:set_ability`).
fn init_ability(g: &mut Game, c: &mut Card) {
	reset_ability(c, g.hands_played);
	if let Kind::Joker(j) = c.kind {
		use Joker as J;
		let ab = &mut c.ab;
		match j {
			J::Joker => ab.mult = 4.0,
			J::Popcorn => ab.mult = 20.0,
			J::Swashbuckler => ab.mult = 1.0,
			J::Constellation | J::Vampire | J::Hologram | J::Obelisk | J::LuckyCat | J::Glass => {
				ab.x_mult = 1.0;
			}
			J::Ramen | J::Duo | J::Tribe => ab.x_mult = 2.0,
			J::Trio | J::Order => ab.x_mult = 3.0,
			J::Family => ab.x_mult = 4.0,
			J::IceCream => ab.chips = 100.0,
			J::Selzer => ab.remaining = 10.0,
			J::TurtleBean => ab.remaining = 5.0,
			J::Rocket => ab.dollars = 1.0,
			J::LoyaltyCard => ab.loyalty_remaining = 5.0,
			J::Caino => ab.caino_xmult = 1.0,
			J::Yorick => ab.yorick_discards = 23.0,
			_ => {}
		}
		if j == J::TodoList {
			// pseudorandom_element over visible hands (pairs order).
			let (hands, n) = g.visible_hands_pairs(None);
			let key = Key::from("to_do");
			let i = g.rng.element(&key, n);
			ab.to_do = Some(hands[i]);
		}
	}
	// G.GAME.used_jokers[key] = true for every center with this name.
	mark_used(g, c);
}

pub(crate) fn mark_used(g: &mut Game, c: &Card) {
	match c.kind {
		Kind::Playing => {}
		Kind::Booster(b) => {
			// Names are shared between the numbered variants of a pack.
			let name = b.name();
			for o in ALL_BOOSTERS {
				if o.name() == name {
					g.used_jokers.insert(Center::Booster(o));
				}
			}
		}
		_ => g.used_jokers.insert(c.center()),
	}
}

/// Clears `used_jokers` when a card leaves the game (`Card:remove`).
pub(crate) fn unmark_used(g: &mut Game, c: &Card) {
	let name = c.name();
	let held = g
		.jokers
		.iter()
		.chain(g.consumables.iter())
		.any(|o| o.name() == name);
	if held {
		return;
	}
	match c.kind {
		Kind::Playing => {}
		Kind::Booster(b) => {
			for o in ALL_BOOSTERS {
				if o.name() == b.name() {
					g.used_jokers.remove(Center::Booster(o));
				}
			}
		}
		_ => g.used_jokers.remove(c.center()),
	}
}

/// A brand new non-playing card of the given kind (Card:init).
pub(crate) fn make_card(g: &mut Game, kind: Kind) -> Card {
	let mut c = g.new_card(kind);
	init_ability(g, &mut c);
	set_cost(g, &mut c, None);
	c
}

/// A new playing card (`Card:init` with a front and a center).
pub(crate) fn make_playing_card(g: &mut Game, suit: Suit, rank: Rank, enh: Enhancement) -> Card {
	let mut c = g.new_card(Kind::Playing);
	c.enhancement = enh;
	init_ability(g, &mut c);
	c.suit = suit;
	c.rank = rank;
	c.orig_suit = Some(suit);
	set_cost(g, &mut c, None);
	c
}

/// `Card:set_ability(center)` on the `i`-th playing card of `area`.
pub(crate) fn set_enhancement(g: &mut Game, area: Area, i: usize, e: Enhancement) {
	let hands_played = g.hands_played;
	let c = &mut g.area_mut(area)[i];
	c.enhancement = e;
	reset_ability(c, hands_played);
	let debuffed = blind::debuffs(&g.blind, g.smeared(), g.pareidolia(), &g.area(area)[i]);
	set_debuff_playing(&mut g.area_mut(area)[i], debuffed);
}

/// Gives the card the edition. True if that adds a slot to the card's area
/// (a card in play turning Negative).
fn apply_edition(c: &mut Card, e: Edition) -> bool {
	let had = c.edition != Edition::None;
	c.edition = e;
	e == Edition::Negative && !had && c.added_to_deck
}

/// `Card:set_edition(edition)` on a card that is in no area yet (`area` is
/// where it will be sold from, for coupons).
pub(crate) fn set_edition(g: &mut Game, c: &mut Card, e: Edition, area: Option<Area>) {
	if apply_edition(c, e) {
		negative_limit(g, c.kind).limit += 1.0;
	}
	set_cost(g, c, area);
}

/// `Card:set_edition(edition)` on the `i`-th card of `area`.
pub(crate) fn set_edition_at(g: &mut Game, area: Area, i: usize, e: Edition) {
	let c = &mut g.area_mut(area)[i];
	let kind = c.kind;
	if apply_edition(c, e) {
		negative_limit(g, kind).limit += 1.0;
	}
	reprice(g, area, i);
}

/// `Card:set_perishable`.
pub(crate) fn set_perishable(g: &Game, c: &mut Card) {
	c.ab.perishable = false;
	if let Some(j) = c.joker()
		&& j.def().perishable_compat
		&& !c.ab.eternal
	{
		c.ab.perishable = true;
		c.ab.perish_tally = g.perishable_rounds;
	}
}

// -----------------------------------------------------------------
// Deck membership (Card:add_to_deck / remove_from_deck / remove)
// -----------------------------------------------------------------

/// The card area slot count a Negative `kind` adds.
fn negative_limit(g: &mut Game, kind: Kind) -> &mut Limit {
	if kind.is_consumable() {
		&mut g.consumable_limit
	} else {
		&mut g.joker_limit
	}
}

/// A card entering or leaving play: what its effect on the game depends on.
#[derive(Clone, Copy)]
struct DeckChange {
	kind: Kind,
	/// Turtle Bean's hand size (`ability.extra.h_size`).
	remaining: f64,
	/// The card is Negative and its extra slot comes or goes with it (a
	/// debuff keeps the slot).
	negative_slot: bool,
}

/// `Card:add_to_deck(from_debuff)`.
pub(crate) fn add_to_deck(g: &mut Game, c: &mut Card, from_debuff: bool) {
	if let Some(change) = mark_added(c, from_debuff) {
		apply_added(g, change);
	}
}

/// The card's half of [`add_to_deck`]; `None` if the game is unaffected.
fn mark_added(c: &mut Card, from_debuff: bool) -> Option<DeckChange> {
	if c.added_to_deck {
		return None;
	}
	c.added_to_deck = true;
	if c.is_playing() {
		return None;
	}
	let negative = c.edition == Edition::Negative;
	if negative && from_debuff {
		c.queue_negative_removal = false;
	}
	Some(DeckChange {
		kind: c.kind,
		remaining: c.ab.remaining,
		negative_slot: negative && !from_debuff,
	})
}

/// The game's half of [`add_to_deck`].
fn apply_added(g: &mut Game, change: DeckChange) {
	if let Kind::Joker(j) = change.kind {
		use Joker as J;
		let h_size = joker_hand_size(j);
		if h_size != 0.0 {
			change_hand_size(g, h_size);
		}
		let d_size = joker_discards(j);
		if d_size > 0.0 {
			g.rr_discards += d_size;
			g.events.push(crate::ev::Ev::EaseDiscard(d_size));
		}
		match j {
			J::CreditCard => g.bankrupt_at -= 20.0,
			J::Chicot => {
				if g.blind.boss && !g.blind.disabled {
					blind::disable_blind(g);
					g.events.status_extra();
				}
			}
			J::Chaos => {
				g.free_rerolls += 1.0;
				calculate_reroll_cost(g, true);
			}
			J::TurtleBean => change_hand_size(g, change.remaining),
			J::Oops => g.probabilities *= 2.0,
			J::ToTheMoon => g.interest_amount += 1.0,
			J::Astronomer => g.events.push(crate::ev::Ev::SetCostAll),
			J::Troubadour => {
				change_hand_size(g, 2.0);
				g.rr_hands += -1.0;
			}
			J::Stuntman => change_hand_size(g, -2.0),
			_ => {}
		}
	}
	if change.negative_slot {
		negative_limit(g, change.kind).limit += 1.0;
	}
	g.events.push(crate::ev::Ev::RefreshDebuffs);
}

/// `Card:remove_from_deck(from_debuff)`.
pub(crate) fn remove_from_deck(g: &mut Game, c: &mut Card, from_debuff: bool) {
	if let Some(change) = mark_removed(c, from_debuff) {
		apply_removed(g, change);
	}
}

/// The card's half of [`remove_from_deck`]; `None` if the game is unaffected.
fn mark_removed(c: &mut Card, from_debuff: bool) -> Option<DeckChange> {
	if !c.added_to_deck {
		return None;
	}
	c.added_to_deck = false;
	let negative = c.edition == Edition::Negative;
	if negative && from_debuff {
		c.queue_negative_removal = true;
	}
	Some(DeckChange {
		kind: c.kind,
		remaining: c.ab.remaining,
		negative_slot: negative && !from_debuff,
	})
}

/// The game's half of [`remove_from_deck`].
fn apply_removed(g: &mut Game, change: DeckChange) {
	if let Kind::Joker(j) = change.kind {
		use Joker as J;
		let h_size = joker_hand_size(j);
		if h_size != 0.0 {
			change_hand_size(g, -h_size);
		}
		let d_size = joker_discards(j);
		if d_size > 0.0 {
			g.rr_discards -= d_size;
			g.events.push(crate::ev::Ev::EaseDiscard(-d_size));
		}
		match j {
			J::CreditCard => g.bankrupt_at += 20.0,
			J::Chaos => {
				g.free_rerolls -= 1.0;
				calculate_reroll_cost(g, true);
			}
			J::TurtleBean => change_hand_size(g, -change.remaining),
			J::Oops => g.probabilities /= 2.0,
			J::ToTheMoon => g.interest_amount -= 1.0,
			J::Astronomer => g.events.push(crate::ev::Ev::SetCostAll),
			J::Troubadour => {
				change_hand_size(g, -2.0);
				g.rr_hands -= -1.0;
			}
			J::Stuntman => change_hand_size(g, 2.0),
			_ => {}
		}
	}
	if change.negative_slot {
		negative_limit(g, change.kind).limit -= 1.0;
	}
	g.events.push(crate::ev::Ev::RefreshDebuffs);
}

/// `Card:remove()` for a card already taken out of its area.
pub(crate) fn destroy_card(g: &mut Game, mut c: Card) {
	remove_from_deck(g, &mut c, false);
	if c.queue_negative_removal {
		negative_limit(g, c.kind).limit -= 1.0;
	}
	unmark_used(g, &c);
}

/// `Card:remove()` for a card the player does not own (shop or pack).
#[expect(
	clippy::needless_pass_by_value,
	reason = "the card leaves the game here, so the caller gives it up"
)]
pub(crate) fn drop_unowned(g: &mut Game, c: Card) {
	unmark_used(g, &c);
}

/// `Card:remove()` for an owned card (timeline-traced).
pub(crate) fn remove_owned(g: &mut Game, c: Card) {
	if c.added_to_deck || c.is_playing() {
		crate::game::trace!(g, "rm {}", c.id);
	}
	destroy_card(g, c);
}

/// `Card:set_debuff(should_debuff)` for a joker in the joker area.
pub(crate) fn set_debuff_joker(g: &mut Game, idx: usize, should: bool) {
	let c = &mut g.jokers[idx];
	if c.ab.perishable && c.ab.perish_tally <= 0.0 {
		if !c.debuff {
			c.debuff = true;
			if let Some(change) = mark_removed(c, true) {
				apply_removed(g, change);
			}
		}
		return;
	}
	if should == c.debuff {
		return;
	}
	if should {
		if let Some(change) = mark_removed(c, true) {
			apply_removed(g, change);
		}
	} else if let Some(change) = mark_added(c, true) {
		apply_added(g, change);
	}
	g.jokers[idx].debuff = should;
}

/// `Card:set_debuff` for a playing card (no deck side effects).
#[inline]
pub(crate) fn set_debuff_playing(c: &mut Card, should: bool) {
	if c.ab.perishable && c.ab.perish_tally <= 0.0 {
		c.debuff = true;
		return;
	}
	c.debuff = should;
}

// -----------------------------------------------------------------
// Resource easing
// -----------------------------------------------------------------

#[inline]
pub(crate) fn ease_dollars(g: &mut Game, d: f64) {
	g.dollars += d;
}

/// `ease_discard(mod)` (applied immediately).
pub(crate) fn ease_discard(g: &mut Game, m: f64) {
	if g.discards_left.max(m).abs() == 0.0 {
		return;
	}
	let m = (-g.discards_left).max(m);
	g.discards_left += m;
}

#[inline]
pub(crate) fn ease_hands_played(g: &mut Game, m: f64) {
	g.hands_left += m;
}

/// `G.hand:change_size(delta)`.
pub(crate) fn change_hand_size(g: &mut Game, delta: f64) {
	if delta != 0.0 {
		g.events.push(crate::ev::Ev::HandSize(delta));
	}
}

/// `calculate_reroll_cost(skip_increment)`.
pub(crate) fn calculate_reroll_cost(g: &mut Game, skip_increment: bool) {
	if g.free_rerolls < 0.0 {
		g.free_rerolls = 0.0;
	}
	if g.free_rerolls > 0.0 {
		g.reroll_cost = 0.0;
		return;
	}
	if !skip_increment {
		g.reroll_cost_increase += 1.0;
	}
	g.reroll_cost = g.temp_reroll_cost.unwrap_or(g.rr_reroll_cost) + g.reroll_cost_increase;
}

/// `level_up_hand(card, hand, instant, amount)`.
pub(crate) fn level_up_hand(g: &mut Game, h: HandType, amount: f64) {
	let base = h.base();
	let hl = &mut g.hands[h as usize];
	hl.level = (hl.level + amount).max(0.0);
	hl.mult = (base.mult + base.level_mult * (hl.level - 1.0)).max(1.0);
	hl.chips = (base.chips + base.level_chips * (hl.level - 1.0)).max(0.0);
}

/// `Card:set_cost()` given the shop discount and whether Astronomer is in
/// effect; `area` is the card's current area (for coupons).
fn price(c: &mut Card, area: Option<Area>, discount_percent: f64, astronomer: bool) {
	// `G.GAME.inflation` only grows in a challenge.
	c.extra_cost = c.edition.extra_cost();
	c.cost =
		1f64.max(((c.base_cost + c.extra_cost + 0.5) * (100.0 - discount_percent) / 100.0).floor());
	let celestial = matches!(c.kind, Kind::Booster(b) if b.kind() == PackKind::Celestial);
	if (matches!(c.kind, Kind::Planet(_)) || celestial) && astronomer {
		c.cost = 0.0;
	}
	if c.ab.rental {
		c.cost = 1.0;
	}
	c.sell_cost = 1f64.max((c.cost / 2.0).floor()) + c.ab.extra_value;
	if c.ab.couponed && matches!(area, Some(Area::ShopJokers | Area::ShopBoosters)) {
		c.cost = 0.0;
	}
}

/// Hand size a joker adds while owned (`ability.h_size`).
fn joker_hand_size(j: Joker) -> f64 {
	match j {
		Joker::Juggler => 1.0,
		Joker::MerryAndy => -1.0,
		_ => 0.0,
	}
}

/// Discards per round a joker adds while owned (`ability.d_size`).
fn joker_discards(j: Joker) -> f64 {
	match j {
		Joker::Drunkard => 1.0,
		Joker::MerryAndy => 3.0,
		_ => 0.0,
	}
}
