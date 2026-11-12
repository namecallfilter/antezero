//! The sizes of the observation and action tensors, and the token layout of
//! one encoded state.

use super::preview::PREVIEW_TOK;
use super::{center, zone};
use crate::data::{
	ACE, ALL_BACKS, ALL_BLINDS, ALL_TAGS, Edition, Enhancement, HandType, Seal, Suit,
};
use crate::game::Game;
use crate::num;

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
/// Maximum legal ops per decision.
pub const MAX_OPS: usize = 96;
/// Maximum scored play candidates per decision (the bindings reject larger
/// requests; [`ops`] also never lets them crowd out other ops).
pub const MAX_CANDIDATES: usize = MAX_OPS / 2;
/// Integers per op: kind, target token (-1 = none), card min, card max,
/// allowed card mask, required card mask, destination count, target slot.
pub const OP_INT: usize = 8;
/// Continuous features per op (scored play candidates: log of the mean
/// score, mean score / chips still needed, estimated probability of clearing
/// the blind, log of the lowest and highest sampled score, cards / 5).
pub const OP_FLOAT: usize = 6;
/// Pointer slots for hand cards (hand tokens come first, so a hand index is
/// also its token index).
pub const MAX_HAND: usize = 32;

/// Vocabulary size of each `TOK_INT` column (for embeddings).
pub const TOK_VOCAB: [i32; TOK_INT] = [
	zone::COUNT,
	center::COUNT,
	num::id(Suit::ALL.len()) + 1,
	ACE as i32 + 1, // ranks; blinds put a poker hand + 1 here
	num::id(Enhancement::ALL.len()),
	num::id(Edition::ALL.len()),
	num::id(Seal::ALL.len()) + 1,
	256, // slot
	32,  // aux: a poker hand + 1 (To Do List, Orbital Tag) or a tag + 1 (blinds)
];
/// Vocabulary size of each `GLOB_INT` column.
pub const GLOB_VOCAB: [i32; GLOB_INT] = [
	6, // phases
	9, // stakes 1..=8
	num::id(ALL_BACKS.len()),
	num::id(ALL_BLINDS.len()) + 1,
	num::id(ALL_BLINDS.len()) + 1,
	num::id(ALL_TAGS.len()) + 1,
	num::id(ALL_TAGS.len()) + 1,
	num::id(HandType::ALL.len()) + 1,
	center::COUNT,
	center::COUNT,
];

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
	/// The layout [`encode`] produces for `g`, without encoding it. Deck and
	/// discard tokens (which no op refers to) are not counted in `n_tokens`.
	#[must_use]
	pub fn of(g: &Game) -> Layout {
		let mut n = 0;
		let mut area = |len: usize| {
			n = (n + len).min(MAX_TOKENS);
			n
		};
		let hand = area(g.hand.len().min(MAX_HAND));
		let consumables = area(g.jokers.len());
		let shop_cards = area(g.consumables.len());
		let shop_vouchers = area(g.shop_jokers.len());
		let shop_boosters = area(g.shop_vouchers.len());
		let pack = area(g.shop_boosters.len());
		// pack cards, tags, the three blinds, the poker hands
		let n_tokens = area(g.pack_cards.len() + g.tags.len() + 3 + HandType::ALL.len());
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
