//! The card areas: `G.deck`, `G.hand`, `G.play`, `G.discard`, `G.jokers`,
//! `G.consumeables`, the shop's areas and `G.pack_cards`.

use crate::card::{AnyCard, Card, CardLike, Kind};
use crate::data::Joker;
use crate::hand_eval::EvalFlags;
use crate::playing_card::PlayingCard;
use crate::rng::PseudoRandom;
use crate::{calculate_joker, num};

/// A card area (`G.jokers`, `G.hand`, ...).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Area {
	Jokers,
	Consumables,
	Hand,
	Deck,
	Play,
	Discard,
	ShopJokers,
	ShopVouchers,
	ShopBoosters,
	PackCards,
}

impl Area {
	/// Every area, in the order a card is looked for by its id.
	pub const ALL: [Area; 10] = [
		Area::Jokers,
		Area::Consumables,
		Area::Hand,
		Area::Play,
		Area::Deck,
		Area::Discard,
		Area::ShopJokers,
		Area::ShopVouchers,
		Area::ShopBoosters,
		Area::PackCards,
	];
}

/// A card area's `config.card_limit`, with `real_card_limit` bookkeeping.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Limit {
	pub limit: f64,
	pub real: f64,
}

impl Limit {
	#[must_use]
	pub fn new(l: f64) -> Limit {
		Limit { limit: l, real: l }
	}

	/// `CardArea:change_size(delta)` (the deferred part).
	#[inline]
	pub fn change(&mut self, delta: f64) {
		self.real += delta;
		self.limit = self.real.max(0.0);
	}
}

/// The limits of the joker and the consumable area, the areas a Negative
/// card adds a slot to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Limits {
	pub jokers: Limit,
	pub consumables: Limit,
}

/// The areas playing cards are drawn between.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Pile {
	Deck,
	Hand,
	Play,
	Discard,
}

impl Pile {
	/// Timeline spelling (`trace-keys` builds).
	#[cfg(feature = "trace-keys")]
	pub(crate) fn name(self) -> &'static str {
		match self {
			Pile::Deck => "deck",
			Pile::Hand => "hand",
			Pile::Play => "play",
			Pile::Discard => "discard",
		}
	}
}

/// How `G.hand` sorts (`G.hand.config.sort`: 'desc' by rank or 'suit desc').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HandSort {
	Rank,
	Suit,
}

/// `G.hand`: the cards held, the hand size and the sort order.
#[derive(Debug, PartialEq)]
pub struct Hand {
	pub cards: Vec<PlayingCard>,
	pub limit: Limit,
	pub order: HandSort,
}

impl Clone for Hand {
	fn clone(&self) -> Hand {
		Hand {
			cards: self.cards.clone(),
			..*self
		}
	}

	fn clone_from(&mut self, source: &Hand) {
		let Hand {
			cards,
			limit,
			order,
		} = self;
		cards.clone_from(&source.cards);
		*limit = source.limit;
		*order = source.order;
	}
}

impl Hand {
	/// The most cards a selection may hold (`G.hand.config.highlighted_limit`).
	pub const HIGHLIGHT_LIMIT: usize = 5;

	/// `G.hand:sort()`.
	pub(crate) fn sort(&mut self) {
		let order = self.order;
		self.cards
			.sort_by(|a, b| b.nominal(order).total_cmp(&a.nominal(order)));
	}
}

/// The cards of one area.
#[derive(Clone, Copy)]
pub(crate) enum Cards<'a> {
	Playing(&'a [PlayingCard]),
	Other(&'a [Card]),
	Any(&'a [AnyCard]),
}

impl<'a> Cards<'a> {
	/// The `i`-th card; panics if there is none.
	fn at(self, i: usize) -> &'a dyn CardLike {
		match self {
			Cards::Playing(cards) => &cards[i],
			Cards::Other(cards) => &cards[i],
			Cards::Any(cards) => &cards[i],
		}
	}

	/// Where the card `id` is.
	fn position(self, id: u32) -> Option<usize> {
		match self {
			Cards::Playing(cards) => position_of(cards, id),
			Cards::Other(cards) => position_of(cards, id),
			Cards::Any(cards) => position_of(cards, id),
		}
	}
}

/// The cards of one area, to change.
pub(crate) enum CardsMut<'a> {
	Playing(&'a mut Vec<PlayingCard>),
	Other(&'a mut Vec<Card>),
	Any(&'a mut Vec<AnyCard>),
}

impl<'a> CardsMut<'a> {
	/// The `i`-th card; panics if there is none.
	pub(crate) fn at_mut(self, i: usize) -> &'a mut dyn CardLike {
		match self {
			CardsMut::Playing(cards) => &mut cards[i],
			CardsMut::Other(cards) => &mut cards[i],
			CardsMut::Any(cards) => &mut cards[i],
		}
	}

	fn remove(self, i: usize) -> AnyCard {
		match self {
			CardsMut::Playing(cards) => cards.remove(i).into(),
			CardsMut::Other(cards) => cards.remove(i).into(),
			CardsMut::Any(cards) => cards.remove(i),
		}
	}

	pub(crate) fn for_each(self, mut f: impl FnMut(&mut dyn CardLike)) {
		match self {
			CardsMut::Playing(cards) => {
				for c in cards {
					f(c);
				}
			}
			CardsMut::Other(cards) => {
				for c in cards {
					f(c);
				}
			}
			CardsMut::Any(cards) => {
				for c in cards {
					f(c);
				}
			}
		}
	}
}

/// Every card area, and the cards on their way into one.
#[derive(Debug, PartialEq)]
pub struct Areas {
	pub deck: Vec<PlayingCard>,
	pub hand: Hand,
	pub play: Vec<PlayingCard>,
	pub discard: Vec<PlayingCard>,
	pub jokers: Vec<Card>,
	pub consumables: Vec<Card>,
	pub limits: Limits,
	/// The shop's card row: jokers, consumables and, with Magic Trick or
	/// Illusion, playing cards.
	pub shop_jokers: Vec<AnyCard>,
	pub shop_vouchers: Vec<Card>,
	pub shop_boosters: Vec<Card>,
	/// An open pack's cards: playing cards in a Standard pack.
	pub pack_cards: Vec<AnyCard>,
	/// Jokers and consumables created but not yet in their area
	/// (`G.GAME.joker_buffer`, `G.GAME.consumeable_buffer`).
	pub joker_buffer: f64,
	pub consumable_buffer: f64,
	/// Cards outside any area (used consumables, opened boosters) until
	/// their dissolve animation removes them.
	pub(crate) consumed: Vec<Card>,
	pub next_sort_id: u32,
}

impl Default for Areas {
	fn default() -> Areas {
		Areas {
			deck: Vec::with_capacity(64),
			hand: Hand {
				cards: Vec::with_capacity(16),
				limit: Limit::new(8.0),
				order: HandSort::Rank,
			},
			play: Vec::with_capacity(8),
			discard: Vec::with_capacity(64),
			jokers: Vec::with_capacity(8),
			consumables: Vec::with_capacity(4),
			limits: Limits {
				jokers: Limit::new(5.0),
				consumables: Limit::new(2.0),
			},
			shop_jokers: Vec::with_capacity(4),
			shop_vouchers: Vec::with_capacity(2),
			shop_boosters: Vec::with_capacity(2),
			pack_cards: Vec::with_capacity(5),
			joker_buffer: 0.0,
			consumable_buffer: 0.0,
			consumed: Vec::new(),
			next_sort_id: 0,
		}
	}
}

impl Clone for Areas {
	fn clone(&self) -> Areas {
		Areas {
			deck: self.deck.clone(),
			hand: self.hand.clone(),
			play: self.play.clone(),
			discard: self.discard.clone(),
			jokers: self.jokers.clone(),
			consumables: self.consumables.clone(),
			shop_jokers: self.shop_jokers.clone(),
			shop_vouchers: self.shop_vouchers.clone(),
			shop_boosters: self.shop_boosters.clone(),
			pack_cards: self.pack_cards.clone(),
			consumed: self.consumed.clone(),
			..*self
		}
	}

	fn clone_from(&mut self, source: &Areas) {
		let Areas {
			deck,
			hand,
			play,
			discard,
			jokers,
			consumables,
			limits,
			shop_jokers,
			shop_vouchers,
			shop_boosters,
			pack_cards,
			joker_buffer,
			consumable_buffer,
			consumed,
			next_sort_id,
		} = self;
		deck.clone_from(&source.deck);
		hand.clone_from(&source.hand);
		play.clone_from(&source.play);
		discard.clone_from(&source.discard);
		jokers.clone_from(&source.jokers);
		consumables.clone_from(&source.consumables);
		*limits = source.limits;
		shop_jokers.clone_from(&source.shop_jokers);
		shop_vouchers.clone_from(&source.shop_vouchers);
		shop_boosters.clone_from(&source.shop_boosters);
		pack_cards.clone_from(&source.pack_cards);
		*joker_buffer = source.joker_buffer;
		*consumable_buffer = source.consumable_buffer;
		consumed.clone_from(&source.consumed);
		*next_sort_id = source.next_sort_id;
	}
}

impl Areas {
	/// The cards of `a`.
	pub(crate) fn cards(&self, a: Area) -> Cards<'_> {
		match a {
			Area::Jokers => Cards::Other(&self.jokers),
			Area::Consumables => Cards::Other(&self.consumables),
			Area::Hand => Cards::Playing(&self.hand.cards),
			Area::Deck => Cards::Playing(&self.deck),
			Area::Play => Cards::Playing(&self.play),
			Area::Discard => Cards::Playing(&self.discard),
			Area::ShopJokers => Cards::Any(&self.shop_jokers),
			Area::ShopVouchers => Cards::Other(&self.shop_vouchers),
			Area::ShopBoosters => Cards::Other(&self.shop_boosters),
			Area::PackCards => Cards::Any(&self.pack_cards),
		}
	}

	/// The cards of `a`, and the limits (for a card that turns Negative).
	pub(crate) fn cards_and_limits_mut(&mut self, a: Area) -> (CardsMut<'_>, &mut Limits) {
		let cards = match a {
			Area::Jokers => CardsMut::Other(&mut self.jokers),
			Area::Consumables => CardsMut::Other(&mut self.consumables),
			Area::Hand => CardsMut::Playing(&mut self.hand.cards),
			Area::Deck => CardsMut::Playing(&mut self.deck),
			Area::Play => CardsMut::Playing(&mut self.play),
			Area::Discard => CardsMut::Playing(&mut self.discard),
			Area::ShopJokers => CardsMut::Any(&mut self.shop_jokers),
			Area::ShopVouchers => CardsMut::Other(&mut self.shop_vouchers),
			Area::ShopBoosters => CardsMut::Other(&mut self.shop_boosters),
			Area::PackCards => CardsMut::Any(&mut self.pack_cards),
		};
		(cards, &mut self.limits)
	}

	pub(crate) fn cards_mut(&mut self, a: Area) -> CardsMut<'_> {
		self.cards_and_limits_mut(a).0
	}

	/// The cards of the pile `p`.
	pub(crate) fn pile_mut(&mut self, p: Pile) -> &mut Vec<PlayingCard> {
		match p {
			Pile::Deck => &mut self.deck,
			Pile::Hand => &mut self.hand.cards,
			Pile::Play => &mut self.play,
			Pile::Discard => &mut self.discard,
		}
	}

	/// The `i`-th card of `a`; panics if there is none.
	pub(crate) fn card_at(&self, a: Area, i: usize) -> &dyn CardLike {
		self.cards(a).at(i)
	}

	/// The `i`-th card of `a`; panics if there is none.
	pub(crate) fn card_at_mut(&mut self, a: Area, i: usize) -> &mut dyn CardLike {
		self.cards_mut(a).at_mut(i)
	}

	/// Where the card `id` is in `a`.
	pub(crate) fn position(&self, a: Area, id: u32) -> Option<usize> {
		self.cards(a).position(id)
	}

	/// Takes the `i`-th card out of `a`.
	pub(crate) fn remove(&mut self, a: Area, i: usize) -> AnyCard {
		self.cards_mut(a).remove(i)
	}

	/// Takes the card `id` out of its area (looked for in [`Area::ALL`]
	/// order).
	pub(crate) fn take(&mut self, id: u32) -> Option<(Area, AnyCard)> {
		let (area, i) = Area::ALL
			.into_iter()
			.find_map(|a| Some((a, self.position(a, id)?)))?;
		Some((area, self.remove(area, i)))
	}

	/// The next card's `sort_id`.
	#[inline]
	pub(crate) fn next_id(&mut self) -> u32 {
		self.next_sort_id += 1;
		self.next_sort_id
	}

	/// Whether the joker area has a free slot.
	#[inline]
	#[must_use]
	pub fn joker_room(&self) -> bool {
		num::float(self.jokers.len()) < self.limits.jokers.limit
	}

	/// Whether the consumable area has a free slot.
	#[inline]
	#[must_use]
	pub fn consumable_room(&self) -> bool {
		num::float(self.consumables.len()) < self.limits.consumables.limit
	}

	/// Whether the consumable area has a free slot once the cards already
	/// promised by queued creations (`consumable_buffer`) arrive: what jokers
	/// and seals test before they queue one.
	#[inline]
	pub(crate) fn consumable_room_after_queued(&self) -> bool {
		(num::float(self.consumables.len()) + self.consumable_buffer)
			< self.limits.consumables.limit
	}

	/// A non-debuffed `j` in the joker area (`next(find_joker(name))`).
	#[inline]
	#[must_use]
	pub fn has_joker(&self, j: Joker) -> bool {
		calculate_joker::has_joker(&self.jokers, j)
	}

	/// Whether a `j` is in the joker area, debuffed or not.
	#[inline]
	#[must_use]
	pub(crate) fn owns_joker(&self, j: Joker) -> bool {
		self.jokers.iter().any(|c| c.kind == Kind::Joker(j))
	}

	/// The non-debuffed `j`s in the joker area (`#find_joker(name)`).
	#[inline]
	#[must_use]
	pub fn count_joker(&self, j: Joker) -> usize {
		self.jokers.iter().filter(|c| c.is_active_joker(j)).count()
	}

	/// The jokers' sell value (Temperance's money; Swashbuckler's mult is
	/// the others').
	#[must_use]
	pub(crate) fn joker_sell_total(&self) -> f64 {
		let mut total = 0.0;
		for c in &self.jokers {
			if matches!(c.kind, Kind::Joker(_)) {
				total += c.core.sell_cost;
			}
		}
		total
	}

	#[inline]
	pub(crate) fn pareidolia(&self) -> bool {
		self.has_joker(Joker::Pareidolia)
	}

	#[inline]
	pub(crate) fn smeared(&self) -> bool {
		self.has_joker(Joker::Smeared)
	}

	#[must_use]
	pub fn eval_flags(&self) -> EvalFlags {
		EvalFlags {
			four_fingers: self.has_joker(Joker::FourFingers),
			shortcut: self.has_joker(Joker::Shortcut),
			smeared: self.smeared(),
		}
	}

	/// All playing cards (`G.playing_cards`) in their areas.
	pub(crate) fn playing_cards(&self) -> impl Iterator<Item = &PlayingCard> {
		self.deck
			.iter()
			.chain(self.hand.cards.iter())
			.chain(self.play.iter())
			.chain(self.discard.iter())
	}

	pub(crate) fn playing_cards_mut(&mut self) -> impl Iterator<Item = &mut PlayingCard> {
		self.deck
			.iter_mut()
			.chain(self.hand.cards.iter_mut())
			.chain(self.play.iter_mut())
			.chain(self.discard.iter_mut())
	}

	#[must_use]
	pub fn playing_card_count(&self) -> usize {
		self.deck.len() + self.hand.cards.len() + self.play.len() + self.discard.len()
	}

	/// The valid (non-stone) playing cards in `sort_id` order, for the
	/// `reset_*_card` samplers.
	pub(crate) fn non_stone_sorted(&self) -> Vec<&PlayingCard> {
		let mut cards: Vec<&PlayingCard> = self.playing_cards().filter(|c| !c.is_stone()).collect();
		cards.sort_unstable_by_key(|c| c.core.id);
		cards
	}
}

/// The index of the card `id` in `cards`, if it is there.
pub(crate) fn position_of(cards: &[impl CardLike], id: u32) -> Option<usize> {
	cards.iter().position(|c| c.core().id == id)
}

/// Deals the cards the player has not seen again from the unseen pool (the
/// deck, the face-down hand cards and the cards discarded unseen, which keep
/// `wheel_flipped` until the round ends): the pool is put in the order of
/// `key`, shuffled by `shuffle` and dealt into the face-down hand slots and
/// then the unseen discard slots, and the rest becomes the deck. A slot
/// keeps its flags, and a card moved to the deck takes the deck's.
pub(crate) fn redeal_unseen<K: Ord>(
	areas: &mut Areas, key: impl FnMut(&PlayingCard) -> K, shuffle: impl FnOnce(&mut [PlayingCard]),
) {
	let in_hand = |c: &PlayingCard| c.core.face_down;
	let discarded = |c: &PlayingCard| c.wheel_flipped;
	let n = areas.hand.cards.iter().filter(|c| in_hand(c)).count()
		+ areas.discard.iter().filter(|c| discarded(c)).count();
	if n == 0 {
		return;
	}
	let deck_flags = areas
		.deck
		.first()
		.map_or((false, false), |c| (c.core.face_down, c.wheel_flipped));
	// The pool, in the deck's buffer: the face-down hand cards in hand order,
	// the unseen discards in pile order, then the deck.
	let unseen: Vec<PlayingCard> = areas
		.hand
		.cards
		.iter()
		.filter(|c| in_hand(c))
		.chain(areas.discard.iter().filter(|c| discarded(c)))
		.map(|c| {
			let mut c = c.clone();
			(c.core.face_down, c.wheel_flipped) = deck_flags;
			c
		})
		.collect();
	areas.deck.extend(unseen);
	areas.deck.rotate_right(n);
	areas.deck.sort_by_key(key);
	shuffle(&mut areas.deck);
	let mut dealt = areas.deck.drain(..n);
	let slots = areas
		.hand
		.cards
		.iter_mut()
		.filter(|c| in_hand(c))
		.chain(areas.discard.iter_mut().filter(|c| discarded(c)));
	for slot in slots {
		let mut c = dealt.next().expect("as many cards dealt as slots");
		(c.core.face_down, c.wheel_flipped) = (slot.core.face_down, slot.wheel_flipped);
		*slot = c;
	}
}

/// `pseudorandom_element(cards, pseudoseed(key))` over the cards of `cards`
/// that `keep` accepts (by index and card), which the game samples in
/// `sort_id` order: the chosen card's index, or `None` (and no draw) if
/// `keep` accepts none.
pub(crate) fn pick_card<C: CardLike>(
	rng: &mut PseudoRandom, key: &str, cards: &[C], keep: impl Fn(usize, &C) -> bool,
) -> Option<usize> {
	let mut candidates: Vec<usize> = cards
		.iter()
		.enumerate()
		.filter(|&(i, c)| keep(i, c))
		.map(|(i, _)| i)
		.collect();
	if candidates.is_empty() {
		return None;
	}
	candidates.sort_by_key(|&i| cards[i].core().id);
	Some(candidates[rng.element_str(key, candidates.len())])
}
