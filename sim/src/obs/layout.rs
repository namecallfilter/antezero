use super::preview::PREVIEW_TOK;
use super::{Zone, center};
use crate::data::{
	ALL_BACKS, ALL_BLINDS, ALL_TAGS, Edition, Enhancement, HandType, Rank, Seal, Suit,
};
use crate::game::{Areas, BlindSlot, Phase};
use crate::playing_card::PlayingCard;
use crate::tags::Tags;
use crate::{STAKES, num};

/// Maximum tokens per observation. Tokens past it are dropped (the deck and
/// discard tokens come last), as are the ops on them.
pub const MAX_TOKENS: usize = 200;
/// Categorical features per token: zone, center, suit, rank, enhancement,
/// edition, seal, slot, aux.
pub const TOK_INT: usize = 9;
/// Continuous features per token: 26 card features, then the item's build
/// preview (`TOK_PREVIEW..`, see `BuildPreview::item_feats`).
pub const TOK_FLOAT: usize = TOK_PREVIEW + PREVIEW_TOK;
pub const TOK_PREVIEW: usize = 26;
/// Categorical global features: phase, stake, back, current blind, boss of
/// the ante, small-blind tag, big-blind tag, last hand played, last Tarot /
/// Planet used (The Fool's target, a center id), the ante's voucher once a
/// shop has shown it.
pub const GLOB_INT: usize = 10;
/// Continuous global features: 48 scalars, 32 voucher bits, then a
/// histogram of the remaining deck (52 suit x rank counts, enhancement,
/// edition and seal counts, size, the build preview's capacities) and of
/// the discard pile (52 counts).
pub const GLOB_FLOAT: usize = 216;
/// Extended global floats when `obs_v2` is enabled: appends
/// `last_hand_score`, `last_hand_chips`, `last_hand_mult`, and
/// `last_blind_share`.
pub const GLOB_FLOAT_V2: usize = 220;
pub const MAX_OPS: usize = 96;
/// Maximum scored play candidates per decision (the bindings reject larger
/// requests; [`ops`](fn@super::ops) also never lets them crowd out other ops).
pub const MAX_CANDIDATES: usize = MAX_OPS / 2;
/// Integers per op: kind, target token (-1 = none), card min, card max,
/// allowed card mask, required card mask, destination count, target slot.
pub const OP_INT: usize = 8;
/// Continuous features per op (scored play candidates: log of the mean
/// score, mean score / chips still needed, estimated probability of clearing
/// the blind, log of the lowest and highest sampled score, cards / 5; joker
/// moves: [`OP_ORDER_GAIN`]).
pub const OP_FLOAT: usize = 7;
/// The `OP_FLOAT` column of a joker-move op's gain (see
/// [`order_gain`](super::order_gain)).
pub const OP_ORDER_GAIN: usize = 6;
/// Pointer slots for hand cards (hand tokens come first, so a hand index is
/// also its token index).
pub const MAX_HAND: usize = 32;

/// Slot ids a token can have (later tokens of a long pile share the last).
pub(super) const SLOT_VOCAB: usize = 256;

pub const TOK_VOCAB: [i32; TOK_INT] = TokInts {
	zone: Zone::COUNT,
	center: center::COUNT,
	suit: num::id(Suit::ALL.len()) + 1,
	rank: Rank::Ace as i32 + 1,
	enhancement: num::id(Enhancement::ALL.len()),
	edition: num::id(Edition::ID_COUNT),
	seal: num::id(Seal::ALL.len()) + 1,
	slot: num::id(SLOT_VOCAB),
	aux: 32,
}
.to_array();
pub const GLOB_VOCAB: [i32; GLOB_INT] = [
	num::id(Phase::ALL.len()),
	*STAKES.end() as i32 + 1,
	num::id(ALL_BACKS.len()),
	num::id(ALL_BLINDS.len()) + 1,
	num::id(ALL_BLINDS.len()) + 1,
	num::id(ALL_TAGS.len()) + 1,
	num::id(ALL_TAGS.len()) + 1,
	num::id(HandType::ALL.len()) + 1,
	center::COUNT,
	center::COUNT,
];

/// The id of a phase in the observation: its place in [`Phase::ALL`] (a
/// test checks the two agree).
pub(super) const fn phase_id(p: Phase) -> i32 {
	match p {
		Phase::BlindSelect => 0,
		Phase::Round => 1,
		Phase::Shop => 2,
		Phase::Pack => 3,
		Phase::RoundEval => 4,
		Phase::GameOver => 5,
	}
}

/// A token's `TOK_INT` columns. In `center`, `suit`, `rank`, `enhancement`,
/// `edition`, `seal` and `aux`, 0 is none, and zone 0 is [`Zone::Pad`];
/// `slot` is an index from 0.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct TokInts {
	pub zone: i32,
	pub center: i32,
	pub suit: i32,
	/// A rank id; blinds put a poker hand + 1 here.
	pub rank: i32,
	pub enhancement: i32,
	pub edition: i32,
	pub seal: i32,
	/// The token's index within its area.
	pub slot: i32,
	/// A poker hand + 1 (To Do List, Orbital Tag) or a tag + 1 (blinds).
	pub aux: i32,
}

impl TokInts {
	/// The columns of a token of `zone` at `slot`, the rest none.
	pub(super) fn new(zone: Zone, slot: usize) -> TokInts {
		TokInts {
			zone: zone.id(),
			slot: num::id(slot),
			..TokInts::default()
		}
	}

	/// The columns of a token row (see [`TokInts::to_array`]).
	#[must_use]
	pub fn from_array(columns: [i32; TOK_INT]) -> TokInts {
		let [
			zone,
			center,
			suit,
			rank,
			enhancement,
			edition,
			seal,
			slot,
			aux,
		] = columns;
		TokInts {
			zone,
			center,
			suit,
			rank,
			enhancement,
			edition,
			seal,
			slot,
			aux,
		}
	}

	/// The token row, in the column order of [`TOK_VOCAB`].
	#[must_use]
	pub const fn to_array(self) -> [i32; TOK_INT] {
		[
			self.zone,
			self.center,
			self.suit,
			self.rank,
			self.enhancement,
			self.edition,
			self.seal,
			self.slot,
			self.aux,
		]
	}
}

/// The hand cards that get tokens and pointer slots: the first
/// [`MAX_HAND`].
pub(super) fn shown_hand(cards: &[PlayingCard]) -> &[PlayingCard] {
	&cards[..cards.len().min(MAX_HAND)]
}

/// Observation buffers for one state (flat, row-major).
pub struct ObsMut<'a> {
	/// `[MAX_TOKENS * TOK_INT]`.
	pub tok_i: &'a mut [i32],
	/// `[MAX_TOKENS * TOK_FLOAT]`.
	pub tok_f: &'a mut [f32],
	/// `[GLOB_INT]`.
	pub glob_i: &'a mut [i32],
	/// `[GLOB_FLOAT]`.
	pub glob_f: &'a mut [f32],
}

/// Owned observation buffers for one state.
#[derive(Clone, Debug, PartialEq)]
pub struct ObsBuf {
	pub tok_i: Vec<i32>,
	pub tok_f: Vec<f32>,
	pub glob_i: Vec<i32>,
	pub glob_f: Vec<f32>,
}

impl ObsBuf {
	/// Zeroed buffers of the sizes an observation takes.
	#[must_use]
	pub fn new() -> ObsBuf {
		ObsBuf {
			tok_i: vec![0; MAX_TOKENS * TOK_INT],
			tok_f: vec![0.0; MAX_TOKENS * TOK_FLOAT],
			glob_i: vec![0; GLOB_INT],
			glob_f: vec![0.0; GLOB_FLOAT],
		}
	}

	/// Zeroed buffers sized for `obs_v2` (`GLOB_FLOAT_V2` global floats).
	#[must_use]
	pub fn new_v2() -> ObsBuf {
		ObsBuf {
			tok_i: vec![0; MAX_TOKENS * TOK_INT],
			tok_f: vec![0.0; MAX_TOKENS * TOK_FLOAT],
			glob_i: vec![0; GLOB_INT],
			glob_f: vec![0.0; GLOB_FLOAT_V2],
		}
	}

	/// The buffers, to encode into.
	pub fn as_mut(&mut self) -> ObsMut<'_> {
		ObsMut {
			tok_i: &mut self.tok_i,
			tok_f: &mut self.tok_f,
			glob_i: &mut self.glob_i,
			glob_f: &mut self.glob_f,
		}
	}
}

impl Default for ObsBuf {
	fn default() -> ObsBuf {
		ObsBuf::new()
	}
}

/// Token index layout of an encoded observation (needed to decode ops):
/// the number of hand tokens, the first token of each other area, and the
/// number of tokens.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Layout {
	pub hand: usize,
	pub jokers: usize,
	pub consumables: usize,
	pub shop_cards: usize,
	pub shop_vouchers: usize,
	pub shop_boosters: usize,
	pub pack: usize,
	pub n_tokens: usize,
}

impl Layout {
	/// The layout [`encode`](fn@super::encode) produces for a game with these
	/// areas and tags, without encoding it. Deck and discard tokens (which no
	/// op refers to) are not counted in `n_tokens`.
	#[must_use]
	pub fn of(areas: &Areas, tags: &Tags) -> Layout {
		let mut n = 0;
		let mut area = |len: usize| {
			n = (n + len).min(MAX_TOKENS);
			n
		};
		let hand = area(shown_hand(&areas.hand.cards).len());
		let consumables = area(areas.jokers.len());
		let shop_cards = area(areas.consumables.len());
		let shop_vouchers = area(areas.shop_jokers.len());
		let shop_boosters = area(areas.shop_vouchers.len());
		let pack = area(areas.shop_boosters.len());
		// pack cards, tags, the blinds, the poker hands
		let n_tokens = area(
			areas.pack_cards.len() + tags.held.len() + BlindSlot::ALL.len() + HandType::ALL.len(),
		);
		Layout {
			hand,
			jokers: hand,
			consumables,
			shop_cards,
			shop_vouchers,
			shop_boosters,
			pack,
			n_tokens,
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn phase_ids_are_places_in_phase_all() {
		for (i, p) in Phase::ALL.into_iter().enumerate() {
			assert_eq!(phase_id(p), num::id(i), "the id of {p:?}");
		}
	}
}
