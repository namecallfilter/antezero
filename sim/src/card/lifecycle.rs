//! Card construction (`Card:init` / `set_ability` / `set_base`), costs,
//! editions, deck membership (`add_to_deck` / `remove_from_deck` /
//! `remove`), destruction and debuffs.

use super::any::CardLike;
use super::item::{Ability, Card, CardCore, Kind, Sticker, has_joker};
use super::playing_card::PlayingCard;
use crate::data::{
	ALL_BOOSTERS, Center, CenterSet, Edition, Enhancement, Joker, PackKind, Rank, Suit,
};
use crate::event::{Event, EventQueue};
use crate::game::{Area, Areas, Game, HandLevels, Limit, Limits};
use crate::rng::{PseudoRandom, trace};
use crate::round::{CardDebuff, Round};
use crate::run::{History, Run};
use crate::shop::Shop;
use crate::{flow, round, shop};

/// Starts destroying a playing card: `Card:shatter()` (which marks it
/// shattered) for a Glass card, else `Card:start_dissolve()`.
pub(crate) fn shatter_or_dissolve(c: &mut PlayingCard, events: &mut EventQueue) {
	if c.enhancement == Enhancement::Glass {
		c.shattered = true;
		events.shatter(c.core.id);
	} else {
		events.start_dissolve(c.core.id);
	}
}

/// What `Card:set_cost()` reads from the game: taken first, so that the
/// card can be one of the game's own.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Pricing {
	discount_percent: f64,
	astronomer: bool,
}

impl Pricing {
	pub(crate) fn new(discount_percent: f64, jokers: &[Card]) -> Pricing {
		Pricing {
			discount_percent,
			astronomer: has_joker(jokers, Joker::Astronomer),
		}
	}

	/// `Card:set_cost()`; `area` is the card's current area (for coupons).
	pub(crate) fn set_cost(self, c: &mut (impl CardLike + ?Sized), area: Option<Area>) {
		// What only the cards that are not playing cards have: a cost their
		// kind (Astronomer) or a rental sticker fixes, and `ability.extra_value`.
		let mut fixed_cost = None;
		let mut extra_value = 0.0;
		if let Some(card) = c.as_card() {
			let celestial =
				matches!(card.kind, Kind::Booster(b) if b.kind() == PackKind::Celestial);
			if (matches!(card.kind, Kind::Planet(_)) || celestial) && self.astronomer {
				fixed_cost = Some(0.0);
			}
			if card.ab.rental {
				fixed_cost = Some(1.0);
			}
			extra_value = card.ab.extra_value;
		}
		let c = c.core_mut();
		// `G.GAME.inflation` only grows in a challenge.
		c.extra_cost = c.edition.map_or(0.0, Edition::extra_cost);
		c.cost = 1f64.max(
			((c.base_cost + c.extra_cost + 0.5) * (100.0 - self.discount_percent) / 100.0).floor(),
		);
		if let Some(cost) = fixed_cost {
			c.cost = cost;
		}
		c.sell_cost = 1f64.max((c.cost / 2.0).floor()) + extra_value;
		if c.couponed && matches!(area, Some(Area::ShopJokers | Area::ShopBoosters)) {
			c.cost = 0.0;
		}
	}
}

/// Re-runs `set_cost` on every card in existence (the `G.I.CARD` loops).
pub(crate) fn set_cost_all(areas: &mut Areas, discount_percent: f64) {
	let pricing = Pricing::new(discount_percent, &areas.jokers);
	for area in Area::ALL {
		areas
			.cards_mut(area)
			.for_each(|c| pricing.set_cost(c, Some(area)));
	}
}

/// The part of `Card:set_ability(center)` that every card has: a fresh
/// ability table (without a coupon) and the center's base cost.
fn reset_core(c: &mut CardCore, hands_played: f64, base_cost: f64) {
	c.couponed = false;
	c.hands_played_at_create = hands_played;
	c.base_cost = base_cost;
}

/// The discards each of Yorick's ×Mult gains takes (`extra.discards`).
pub(crate) const YORICK_DISCARDS: f64 = 23.0;

/// `Card:set_ability(center)`'s fresh ability table for a card that is not
/// a playing card.
fn reset_ability(c: &mut Card, hands_played: f64) {
	c.ab = Ability::default();
	let base_cost = match c.kind {
		Kind::Joker(j) => j.def().cost,
		Kind::Tarot(t) => t.def().cost,
		Kind::Planet(p) => p.def().cost,
		Kind::Spectral(s) => s.def().cost,
		Kind::Voucher(v) => v.def().cost,
		Kind::Booster(b) => b.def().cost,
	};
	reset_core(&mut c.core, hands_played, f64::from(base_cost));
}

/// `Card:set_ability(center)` on a playing card: the fresh ability table
/// keeps only `perma_bonus` and `forced_selection` (so e.g.
/// `played_this_ante` is lost).
fn reset_playing_ability(c: &mut PlayingCard, hands_played: f64) {
	c.played_this_ante = false;
	c.wheel_flipped = false;
	reset_core(&mut c.core, hands_played, 1.0);
}

/// Initial ability values for a center (`Card:set_ability`).
fn init_ability(
	hands_played: f64, history: &mut History, hands: &HandLevels, rng: &mut PseudoRandom,
	c: &mut Card,
) {
	reset_ability(c, hands_played);
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
			J::Selzer => ab.extra = 10.0,
			J::TurtleBean => ab.extra = 5.0,
			J::Rocket => ab.dollars = 1.0,
			J::LoyaltyCard => ab.loyalty_remaining = 5.0,
			J::Caino => ab.caino_xmult = 1.0,
			J::Yorick => ab.yorick_discards = YORICK_DISCARDS,
			_ => {}
		}
		if j == J::TodoList {
			ab.to_do = Some(hands.random_visible(rng, "to_do", None));
		}
	}
	// G.GAME.used_jokers[key] = true for every center with this name.
	mark_used(&mut history.used_jokers, c);
}

pub(crate) fn mark_used(used: &mut CenterSet, c: &Card) {
	match c.kind {
		Kind::Booster(b) => {
			// Names are shared between the numbered variants of a pack.
			let name = b.name();
			for o in ALL_BOOSTERS {
				if o.name() == name {
					used.insert(Center::Booster(o));
				}
			}
		}
		_ => used.insert(c.center()),
	}
}

/// Clears `used_jokers` when a card leaves the game: all there is to
/// `Card:remove()` for a card the player does not own (shop or pack).
pub(crate) fn unmark_used(areas: &Areas, history: &mut History, c: &impl CardLike) {
	let Some(c) = c.as_card() else {
		return;
	};
	let name = c.name();
	let held = areas
		.jokers
		.iter()
		.chain(areas.consumables.iter())
		.any(|o| o.name() == name);
	if held {
		return;
	}
	match c.kind {
		Kind::Booster(b) => {
			for o in ALL_BOOSTERS {
				if o.name() == b.name() {
					history.used_jokers.remove(Center::Booster(o));
				}
			}
		}
		_ => history.used_jokers.remove(c.center()),
	}
}

/// A brand new non-playing card of the given kind (Card:init).
pub(crate) fn make_card(g: &mut Game, kind: Kind) -> Card {
	let mut c = Card::new(g.areas.next_id(), kind);
	init_ability(
		g.run.hands_played,
		&mut g.history,
		&g.hands,
		&mut g.rng,
		&mut c,
	);
	Pricing::new(g.shop.discount_percent, &g.areas.jokers).set_cost(&mut c, None);
	c
}

/// A new playing card (`Card:init` with a front and a center).
pub(crate) fn make_playing_card(
	areas: &mut Areas, hands_played: f64, discount_percent: f64, suit: Suit, rank: Rank,
	enh: Enhancement,
) -> PlayingCard {
	let mut c = PlayingCard::new(areas.next_id(), suit, rank);
	c.enhancement = enh;
	reset_playing_ability(&mut c, hands_played);
	Pricing::new(discount_percent, &areas.jokers).set_cost(&mut c, None);
	c
}

/// `Card:set_ability(center)` with the enhancement `e` on a playing card.
pub(crate) fn set_enhancement(
	c: &mut PlayingCard, hands_played: f64, debuff: CardDebuff, e: Enhancement,
) {
	c.enhancement = e;
	reset_playing_ability(c, hands_played);
	debuff.apply(c);
}

/// `Card:set_edition(edition)`: the edition and the cost that goes with it
/// (`area` is the card's area, or where it will be sold from, for coupons),
/// and the slot a card in play adds when it turns Negative.
pub(crate) fn set_edition(
	c: &mut (impl CardLike + ?Sized), e: Option<Edition>, area: Option<Area>, pricing: Pricing,
	limits: &mut Limits,
) {
	let had = c.core().edition.is_some();
	c.core_mut().edition = e;
	pricing.set_cost(c, area);
	if e == Some(Edition::Negative) && !had && c.core().added_to_deck {
		negative_limit(limits, c.is_consumable()).limit += 1.0;
	}
}

/// The card area slot count a Negative card adds: a consumable's to the
/// consumables, any other card's to the jokers (playing cards included, as in
/// the game).
fn negative_limit(limits: &mut Limits, consumable: bool) -> &mut Limit {
	if consumable {
		&mut limits.consumables
	} else {
		&mut limits.jokers
	}
}

/// A card entering or leaving play: what its effect on the game depends on.
#[derive(Clone, Copy)]
struct DeckChange {
	direction: Direction,
	/// The joker whose effect comes or goes.
	joker: Option<Joker>,
	/// Turtle Bean's hand size (`ability.extra.h_size`).
	hand_size: f64,
	/// The card is Negative and its extra slot comes or goes with it (a
	/// debuff keeps the slot).
	negative_slot: bool,
	consumable: bool,
}

/// Whether a card's effect comes or goes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Direction {
	Added,
	Removed,
}

impl Direction {
	/// The sign of what the effect adds.
	fn sign(self) -> f64 {
		match self {
			Direction::Added => 1.0,
			Direction::Removed => -1.0,
		}
	}
}

/// What takes a card into or out of play (`from_debuff`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Cause {
	Moved,
	Debuff,
}

/// `Card:add_to_deck()`.
pub(crate) fn add_to_deck(g: &mut Game, c: &mut impl CardLike) {
	if let Some(change) = mark_added(c, Cause::Moved) {
		apply_added(g, change);
	}
}

/// The card's half of [`add_to_deck`]; `None` if the game is unaffected
/// (a playing card is only marked).
fn mark_added(c: &mut (impl CardLike + ?Sized), cause: Cause) -> Option<DeckChange> {
	let core = c.core_mut();
	if core.added_to_deck {
		return None;
	}
	core.added_to_deck = true;
	let negative = core.edition == Some(Edition::Negative);
	let c = c.as_card_mut()?;
	if negative && cause == Cause::Debuff {
		c.queue_negative_removal = false;
	}
	Some(DeckChange {
		direction: Direction::Added,
		joker: c.joker(),
		hand_size: c.ab.extra,
		negative_slot: negative && cause == Cause::Moved,
		consumable: c.kind.is_consumable(),
	})
}

/// The game's half of [`add_to_deck`].
fn apply_added(g: &mut Game, change: DeckChange) {
	joker_effects(&mut g.run, &mut g.round, &mut g.shop, &mut g.events, change);
	if change.joker == Some(Joker::Chicot) && g.blind.is_boss() && !g.blind.disabled {
		round::disable_blind(g);
		g.events.status_extra();
	}
	slot_effects(&mut g.areas.limits, &mut g.events, change);
}

/// `Card:remove_from_deck()`.
pub(crate) fn remove_from_deck(
	limits: &mut Limits, run: &mut Run, round: &mut Round, shop: &mut Shop,
	events: &mut EventQueue, c: &mut impl CardLike,
) {
	if let Some(change) = mark_removed(c, Cause::Moved) {
		apply_removed(limits, run, round, shop, events, change);
	}
}

/// The card's half of [`remove_from_deck`]; `None` if the game is unaffected.
fn mark_removed(c: &mut (impl CardLike + ?Sized), cause: Cause) -> Option<DeckChange> {
	let core = c.core_mut();
	if !core.added_to_deck {
		return None;
	}
	core.added_to_deck = false;
	let negative = core.edition == Some(Edition::Negative);
	let mut change = DeckChange {
		direction: Direction::Removed,
		joker: None,
		hand_size: 0.0,
		negative_slot: negative && cause == Cause::Moved,
		consumable: false,
	};
	if let Some(c) = c.as_card_mut() {
		if negative && cause == Cause::Debuff {
			c.queue_negative_removal = true;
		}
		change.joker = c.joker();
		change.hand_size = c.ab.extra;
		change.consumable = c.kind.is_consumable();
	}
	Some(change)
}

/// The game's half of [`remove_from_deck`].
fn apply_removed(
	limits: &mut Limits, run: &mut Run, round: &mut Round, shop: &mut Shop,
	events: &mut EventQueue, change: DeckChange,
) {
	joker_effects(run, round, shop, events, change);
	slot_effects(limits, events, change);
}

/// What owning the changed joker does, coming or going: everything but
/// Chicot's disabling of the boss, which only coming does.
fn joker_effects(
	run: &mut Run, round: &mut Round, shop: &mut Shop, events: &mut EventQueue, change: DeckChange,
) {
	use Joker as J;
	let Some(j) = change.joker else {
		return;
	};
	let sign = change.direction.sign();
	let h_size = joker_hand_size(j);
	if h_size != 0.0 {
		flow::change_hand_size(events, sign * h_size);
	}
	let d_size = joker_discards(j);
	if d_size > 0.0 {
		round.base_discards += sign * d_size;
		events.push(Event::EaseDiscard(sign * d_size));
	}
	match j {
		J::CreditCard => run.bankrupt_at -= sign * 20.0,
		J::Chaos => {
			shop.free_rerolls += sign;
			shop::calculate_reroll_cost(shop);
		}
		J::TurtleBean => flow::change_hand_size(events, sign * change.hand_size),
		J::Oops => match change.direction {
			Direction::Added => run.probabilities *= 2.0,
			Direction::Removed => run.probabilities /= 2.0,
		},
		J::ToTheMoon => run.interest_amount += sign,
		J::Astronomer => events.push(Event::SetCostAll),
		J::Troubadour => {
			flow::change_hand_size(events, sign * 2.0);
			round.base_hands -= sign;
		}
		J::Stuntman => flow::change_hand_size(events, sign * -2.0),
		_ => {}
	}
}

/// The Negative card's slot coming or going, and the debuffs to refresh.
fn slot_effects(limits: &mut Limits, events: &mut EventQueue, change: DeckChange) {
	if change.negative_slot {
		negative_limit(limits, change.consumable).limit += change.direction.sign();
	}
	events.push(Event::RefreshDebuffs);
}

/// `Card:remove()` for a card already taken out of its area.
pub(crate) fn destroy_card(g: &mut Game, mut c: impl CardLike) {
	remove_from_deck(
		&mut g.areas.limits,
		&mut g.run,
		&mut g.round,
		&mut g.shop,
		&mut g.events,
		&mut c,
	);
	if let Some(card) = c.as_card()
		&& card.queue_negative_removal
	{
		negative_limit(&mut g.areas.limits, card.kind.is_consumable()).limit -= 1.0;
	}
	unmark_used(&g.areas, &mut g.history, &c);
}

/// `Card:remove()` for an owned card (timeline-traced).
pub(crate) fn remove_owned(g: &mut Game, c: impl CardLike) {
	if c.core().added_to_deck || c.is_playing() {
		trace!(g.rng, "rm {}", c.core().id);
	}
	destroy_card(g, c);
}

/// `Card:set_debuff(should_debuff)` for a joker in the joker area.
pub(crate) fn set_debuff_joker(g: &mut Game, i: usize, should: bool) {
	let c = &mut g.areas.jokers[i];
	let perished = matches!(c.ab.sticker, Some(Sticker::Perishable { tally }) if tally <= 0.0);
	let should = should || perished;
	if should == c.core.debuff {
		return;
	}
	c.core.debuff = should;
	if should {
		if let Some(change) = mark_removed(c, Cause::Debuff) {
			apply_removed(
				&mut g.areas.limits,
				&mut g.run,
				&mut g.round,
				&mut g.shop,
				&mut g.events,
				change,
			);
		}
	} else if let Some(change) = mark_added(c, Cause::Debuff) {
		apply_added(g, change);
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
