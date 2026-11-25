//! Card shuffling, drawing, removal, copying, debuff refreshes, and stat
//! easing helpers.

use crate::areas::{self, Pile};
use crate::card::{AnyCard, Card, CardLike, PlayingCard, Pricing};
use crate::data::{BlindKind, Joker};
use crate::event::{Draw, Event, EventQueue};
use crate::game::{Area, Areas, Blind, Flow, Game, HandLevels, State};
use crate::rng::{Key, PseudoRandom, shuffle, trace};
use crate::round::{CardDebuff, Round};
use crate::run::{History, Run};
use crate::{card, num, round};

/// `CardArea:shuffle(seed)`: sort by `sort_id`, then Fisher–Yates.
pub(crate) fn shuffle_area<C: CardLike>(cards: &mut [C], rng: &mut PseudoRandom, key: &Key) {
	cards.sort_unstable_by_key(|c| c.core().id);
	let seed = rng.seed_for(key);
	shuffle(cards, seed);
}

#[inline]
pub(crate) fn ease_dollars(run: &mut Run, d: f64) {
	run.dollars += d;
}

/// `ease_discard(mod)` (applied immediately).
pub(crate) fn ease_discard(round: &mut Round, m: f64) {
	if round.discards_left.max(m).abs() == 0.0 {
		return;
	}
	let m = (-round.discards_left).max(m);
	round.discards_left += m;
}

#[inline]
pub(crate) fn ease_hands_played(round: &mut Round, m: f64) {
	round.hands_left += m;
}

/// `G.hand:change_size(delta)`.
pub(crate) fn change_hand_size(events: &mut EventQueue, delta: f64) {
	if delta != 0.0 {
		events.push(Event::HandSize(delta));
	}
}

/// `CardArea:emplace` for piles (deck inserts at the front).
fn emplace(areas: &mut Areas, to: Pile, mut c: PlayingCard, stay_flipped: bool) {
	match to {
		Pile::Deck => {
			c.core.face_down = true;
			areas.deck.insert(0, c);
		}
		Pile::Discard => {
			c.core.face_down = true;
			areas.discard.push(c);
		}
		Pile::Hand | Pile::Play => {
			if c.core.face_down && !stay_flipped {
				c.core.face_down = false;
				c.wheel_flipped = false;
			}
			if to == Pile::Hand && stay_flipped {
				c.wheel_flipped = true;
				c.core.face_down = true;
			}
			if to == Pile::Hand {
				areas.hand.cards.push(c);
			} else {
				areas.play.push(c);
			}
		}
	}
}

/// The body of `draw_card`'s event.
pub(crate) fn draw_card(
	areas: &mut Areas, probabilities: f64, round: &Round, blind: &Blind, rng: &mut PseudoRandom,
	draw: Draw,
) {
	let Draw {
		from,
		to,
		card,
		sort,
	} = draw;
	let taken = if let Some(id) = card {
		let src = areas.pile_mut(from);
		areas::position_of(src, id).map(|i| src.remove(i))
	} else {
		// CardArea:draw_card_from → area:remove_card()
		let src_is_stack = matches!(from, Pile::Deck | Pile::Discard);
		let src = areas.pile_mut(from);
		if src.is_empty() {
			None
		} else if src_is_stack {
			src.pop()
		} else {
			Some(src.remove(0))
		}
	};
	let Some(c) = taken else { return };
	let stay = round::stay_flipped(areas, probabilities, round, blind, rng, to, &c);
	trace!(rng, "draw {} {}", to.name(), c.core.id);
	emplace(areas, to, c, stay);
	if sort && to == Pile::Hand {
		areas.hand.sort();
	}
}

/// `G.FUNCS.draw_from_deck_to_hand(e)`: draws `count` cards, or with `None`
/// enough to fill the hand; returns true on the game-over path.
pub(crate) fn draw_from_deck_to_hand(
	flow: &mut Flow, areas: &Areas, round: &Round, blind: &Blind, events: &mut EventQueue,
	count: Option<usize>,
) -> bool {
	if !matches!(flow.state, State::TarotPack | State::SpectralPack)
		&& areas.hand.limit.limit <= 0.0
		&& areas.hand.cards.is_empty()
	{
		flow.state = State::GameOver;
		flow.state_complete = false;
		return true;
	}
	let mut hand_space: usize = match count {
		Some(n) => n,
		None => areas
			.deck
			.len()
			.min(num::trunc(areas.hand.limit.limit).saturating_sub(areas.hand.cards.len())),
	};
	if blind.active(BlindKind::Serpent) && (round.hands_played > 0.0 || round.discards_used > 0.0) {
		hand_space = areas.deck.len().min(3);
	}
	events.delay(0.3);
	for _ in 0..hand_space {
		events.draw(
			Draw {
				sort: true,
				..Draw::next(Pile::Deck, Pile::Hand)
			},
			0.1,
		);
	}
	false
}

/// Finds and removes a card anywhere (`Card:remove`).
pub(crate) fn remove_card_by_id(g: &mut Game, id: u32) {
	let (c, owned) = if let Some((area, c)) = g.areas.take(id) {
		let owned = match area {
			Area::Jokers | Area::Consumables => c.core().added_to_deck,
			Area::Hand | Area::Play | Area::Deck | Area::Discard => true,
			Area::ShopJokers | Area::ShopVouchers | Area::ShopBoosters | Area::PackCards => false,
		};
		(c, owned)
	} else if let Some(i) = areas::position_of(&g.areas.consumed, id) {
		let c = g.areas.consumed.remove(i);
		let owned = c.core.added_to_deck;
		(AnyCard::from(c), owned)
	} else {
		return;
	};
	if owned {
		card::remove_owned(g, c);
	} else {
		card::destroy_card(g, c);
	}
}

/// `copy_card(other, nil, nil, nil, strip_edition)` for a joker or a
/// consumable.
///
/// `c` is the caller's clone of the card to copy; it becomes the copy.
pub(crate) fn copy_card(
	areas: &mut Areas, history: &mut History, discount_percent: f64, hands: &HandLevels,
	rng: &mut PseudoRandom, mut c: Card, strip_edition: bool,
) -> Card {
	// set_ability(center) marks the center as used; ability is then copied.
	card::mark_used(&mut history.used_jokers, &c);
	c.queue_negative_removal = false;
	c.getting_sliced = false;
	c.ankh_deletable = false;
	c.temp_edition = false;
	// set_ability(center) rolls a To Do List hand before the copied
	// ability overwrites it: only the stream's advance matters.
	if c.is_joker(Joker::TodoList) {
		let _rolled = hands.random_visible(rng, "to_do", None);
	}
	copy_core(areas, discount_percent, c, strip_edition)
}

/// `copy_card(other, nil, nil, playing_card)` (Cryptid, DNA).
///
/// `c` is the caller's clone of the card to copy; it becomes the copy.
pub(crate) fn copy_playing_card(
	areas: &mut Areas, discount_percent: f64, mut c: PlayingCard,
) -> PlayingCard {
	c.shattered = false;
	c.destroyed = false;
	c.lucky_trigger = false;
	c.vampired = false;
	// set_base with a fresh card: times_played reset, suit_nominal_original
	// from the current suit.
	c.times_played = 0;
	c.orig_suit = c.suit;
	copy_core(areas, discount_percent, c, /* strip_edition= */ false)
}

/// The part of `copy_card` every card has: a new id, out of the deck, face
/// up, the edition (unless stripped) and the cost that goes with it, and
/// the debuff kept.
fn copy_core<C: CardLike>(
	areas: &mut Areas, discount_percent: f64, mut c: C, strip_edition: bool,
) -> C {
	let core = c.core_mut();
	core.id = areas.next_id();
	core.added_to_deck = false;
	// A new Card starts face up.
	core.face_down = false;
	if strip_edition {
		core.edition = None;
	}
	Pricing::new(discount_percent, &areas.jokers).set_cost(&mut c, None);
	c
}

/// `Blind:set_blind(nil, true)`: re-evaluate playing-card debuffs.
pub(crate) fn refresh_playing_debuffs(areas: &mut Areas, blind: &Blind) {
	let debuff = CardDebuff::new(blind, &areas.jokers);
	for c in areas.playing_cards_mut() {
		debuff.apply(c);
	}
}
