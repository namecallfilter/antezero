//! Port of `evaluate_poker_hand` and helpers from `misc_functions.lua`.
//!
//! Groups are represented as bitmasks over the positions of the evaluated
//! cards; the game always re-sorts the scoring hand by table position, so
//! membership is all that matters downstream.

use crate::card::{HandSort, PlayingCard, SuitTest};
use crate::data::{HandSet, HandType, Rank, Suit};

/// The most cards [`evaluate`] takes: one bit of a `u8` mask per card.
pub const MAX_EVAL_CARDS: usize = 8;

/// Slots of a table indexed by rank id (`Card:get_id()`, 2 to the Ace's 14).
const RANK_IDS: usize = Rank::Ace.id() as usize + 1;

/// The best hand found and its cards (`poker_hands[top][1]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TopHand {
	pub hand: HandType,
	/// The hand's cards, one bit per position.
	pub mask: u8,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PokerHands {
	/// The hands whose `poker_hands[h]` is non-empty.
	pub present: HandSet,
	/// The best of them; `None` only for no cards.
	pub top: Option<TopHand>,
}

impl PokerHands {
	#[inline]
	#[must_use]
	pub fn has(&self, h: HandType) -> bool {
		self.present.contains(h)
	}
}

#[derive(Clone, Copy, Debug)]
pub struct EvalFlags {
	pub four_fingers: bool,
	pub shortcut: bool,
	pub smeared: bool,
}

/// The cards of each rank id, one bit per position.
fn rank_masks(ids: &[Option<Rank>]) -> [u8; RANK_IDS] {
	let mut by_rank = [0u8; RANK_IDS];
	for (i, id) in ids.iter().enumerate() {
		if let Some(id) = *id {
			by_rank[usize::from(id.id())] |= 1 << i;
		}
	}
	by_rank
}

/// The most groups of two or more cards that [`MAX_EVAL_CARDS`] cards make.
const MAX_GROUPS: usize = MAX_EVAL_CARDS / 2;

/// The smallest and largest same-rank groups the poker hands are made of.
const SMALLEST_GROUP: usize = 2;
const LARGEST_GROUP: usize = 5;
const GROUP_SIZE_COUNT: usize = LARGEST_GROUP - SMALLEST_GROUP + 1;

/// Groups of same-rank cards of each size from [`SMALLEST_GROUP`] to
/// [`LARGEST_GROUP`], highest rank first.
struct Groups {
	/// `masks[size - SMALLEST_GROUP]`: the groups of exactly `size` cards.
	masks: [[u8; MAX_GROUPS]; GROUP_SIZE_COUNT],
	counts: [usize; GROUP_SIZE_COUNT],
}

impl Groups {
	/// How many groups of exactly `size` cards there are.
	fn count(&self, size: usize) -> usize {
		self.counts[size - SMALLEST_GROUP]
	}

	/// The `k`-th group of `size` cards (0 if there is none).
	fn nth(&self, size: usize, k: usize) -> u8 {
		self.masks[size - SMALLEST_GROUP][k]
	}

	/// The highest-ranked group of `size` cards (0 if there is none).
	fn first(&self, size: usize) -> u8 {
		self.nth(size, 0)
	}
}

/// The groups of `by_rank` in one pass: this runs for every subset of a hand
/// that gets ranked.
fn same_rank(by_rank: &[u8; RANK_IDS]) -> Groups {
	let mut groups = Groups {
		masks: [[0; MAX_GROUPS]; GROUP_SIZE_COUNT],
		counts: [0; GROUP_SIZE_COUNT],
	};
	for &mask in by_rank.iter().rev() {
		let size = mask.count_ones() as usize;
		if (SMALLEST_GROUP..=LARGEST_GROUP).contains(&size) {
			let s = size - SMALLEST_GROUP;
			groups.masks[s][groups.counts[s]] = mask;
			groups.counts[s] += 1;
		}
	}
	groups
}

fn flush(cards: &[&PlayingCard], flags: EvalFlags) -> u8 {
	let need = if flags.four_fingers { 4 } else { 5 };
	if cards.len() > 5 || cards.len() < need {
		return 0;
	}
	for suit in Suit::ALL {
		let mut mask = 0u8;
		let mut count = 0;
		for (i, c) in cards.iter().enumerate() {
			if c.is_suit(suit, SuitTest::Flush, flags.smeared) {
				count += 1;
				mask |= 1 << i;
			}
		}
		if count >= need {
			return mask;
		}
	}
	0
}

/// The straight among `n` cards whose ranks are `by_rank`.
fn straight(by_rank: &[u8; RANK_IDS], n: usize, flags: EvalFlags) -> u8 {
	const ACE: usize = RANK_IDS - 1;
	let need = if flags.four_fingers { 4 } else { 5 };
	if n > 5 || n < need {
		return 0;
	}
	let mut t = 0u8;
	let mut length = 0;
	let mut found = false;
	let mut skipped = false;
	// The Ace also counts as 1, below the 2.
	for j in 1..=ACE {
		let r = if j == 1 { ACE } else { j };
		if by_rank[r] != 0 {
			length += 1;
			skipped = false;
			t |= by_rank[r];
		} else if flags.shortcut && !skipped && j != ACE {
			skipped = true;
		} else {
			length = 0;
			skipped = false;
			if !found {
				t = 0;
			}
			if found {
				break;
			}
		}
		if length >= need {
			found = true;
		}
	}
	if found { t } else { 0 }
}

fn highest(cards: &[&PlayingCard]) -> u8 {
	let mut best: Option<(usize, f64)> = None;
	for (i, c) in cards.iter().enumerate() {
		let v = c.nominal(HandSort::Rank);
		match best {
			None => best = Some((i, v)),
			Some((_, bv)) if v > bv => best = Some((i, v)),
			_ => {}
		}
	}
	best.map_or(0, |(i, _)| 1u8 << i)
}

/// The hands found so far, with their cards (`poker_hands`).
struct Found {
	present: HandSet,
	masks: [u8; HandType::ALL.len()],
}

impl Found {
	fn set(&mut self, h: HandType, mask: u8) {
		self.present.insert(h);
		self.masks[h as usize] = mask;
	}

	fn mask(&self, h: HandType) -> u8 {
		self.masks[h as usize]
	}
}

/// `evaluate_poker_hand(hand)` + the hand selection of `get_poker_hand_info`.
///
/// # Panics
///
/// Panics if `cards` holds more than [`MAX_EVAL_CARDS`] cards.
#[must_use]
pub fn evaluate(cards: &[&PlayingCard], flags: EvalFlags) -> PokerHands {
	assert!(
		cards.len() <= MAX_EVAL_CARDS,
		"a hand mask has room for {MAX_EVAL_CARDS} cards, got {}",
		cards.len()
	);
	let mut ids = [None; MAX_EVAL_CARDS];
	for (i, c) in cards.iter().enumerate() {
		ids[i] = c.effective_rank();
	}
	let by_rank = rank_masks(&ids[..cards.len()]);
	let groups = same_rank(&by_rank);
	let flush_mask = flush(cards, flags);
	let straight_mask = straight(&by_rank, cards.len(), flags);
	let high_card = highest(cards);

	let mut found = Found {
		present: HandSet::default(),
		masks: [0; HandType::ALL.len()],
	};
	if groups.count(5) > 0 && flush_mask != 0 {
		found.set(HandType::FlushFive, groups.first(5));
	}
	if groups.count(3) > 0 && groups.count(2) > 0 && flush_mask != 0 {
		found.set(HandType::FlushHouse, groups.first(3) | groups.first(2));
	}
	if groups.count(5) > 0 {
		found.set(HandType::FiveOfAKind, groups.first(5));
	}
	if flush_mask != 0 && straight_mask != 0 {
		found.set(HandType::StraightFlush, flush_mask | straight_mask);
	}
	if groups.count(4) > 0 {
		found.set(HandType::FourOfAKind, groups.first(4));
	}
	if groups.count(3) > 0 && groups.count(2) > 0 {
		found.set(HandType::FullHouse, groups.first(3) | groups.first(2));
	}
	if flush_mask != 0 {
		found.set(HandType::Flush, flush_mask);
	}
	if straight_mask != 0 {
		found.set(HandType::Straight, straight_mask);
	}
	if groups.count(3) > 0 {
		found.set(HandType::ThreeOfAKind, groups.first(3));
	}
	if groups.count(2) == 2 || (groups.count(3) == 1 && groups.count(2) == 1) {
		let second = if groups.count(2) >= 2 {
			groups.nth(2, 1)
		} else {
			groups.first(3)
		};
		found.set(HandType::TwoPair, groups.first(2) | second);
	}
	if groups.count(2) > 0 {
		found.set(HandType::Pair, groups.first(2));
	}
	if high_card != 0 {
		found.set(HandType::HighCard, high_card);
	}
	// Containment fix-ups at the end of evaluate_poker_hand.
	if groups.count(5) > 0 {
		found.set(HandType::FourOfAKind, groups.first(5));
	}
	if found.present.contains(HandType::FourOfAKind) {
		found.set(HandType::ThreeOfAKind, found.mask(HandType::FourOfAKind));
	}
	if found.present.contains(HandType::ThreeOfAKind) {
		found.set(HandType::Pair, found.mask(HandType::ThreeOfAKind));
	}

	let top = HandType::ALL
		.iter()
		.copied()
		.find(|&h| found.present.contains(h))
		.map(|hand| TopHand {
			hand,
			mask: found.mask(hand),
		});
	PokerHands {
		present: found.present,
		top,
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::num;
	use crate::rng::Xorshift;

	/// The game's own grouping (`get_X_same`): one scan of the hand per card, per group size.
	fn x_same(size: usize, ids: &[Option<Rank>]) -> ([u8; MAX_GROUPS], usize) {
		let mut vals = [0u8; RANK_IDS];
		for i in (0..ids.len()).rev() {
			let Some(id) = ids[i] else { continue };
			let mut mask = 1u8 << i;
			let mut count = 1;
			for (j, other) in ids.iter().enumerate() {
				if j != i && *other == Some(id) {
					mask |= 1 << j;
					count += 1;
				}
			}
			if count == size {
				vals[usize::from(id.id())] = mask;
			}
		}
		let mut out = [0u8; MAX_GROUPS];
		let mut k = 0;
		for &mask in vals.iter().rev() {
			if mask != 0 {
				out[k] = mask;
				k += 1;
			}
		}
		(out, k)
	}

	#[test]
	fn same_rank_groups_as_the_game_does() {
		let mut rng = Xorshift::new(0x9e37_79b9_7f4a_7c15);
		for _ in 0..400_000 {
			let mut ids = [None; MAX_EVAL_CARDS];
			let x = rng.next_u64();
			let n = num::pick(x, MAX_EVAL_CARDS + 1);
			// Few ranks, so groups of every size (and several of a size) are common; some cards have no rank (Stone).
			let span = 2 + (x >> 8) % 6;
			for (k, id) in ids.iter_mut().enumerate().take(n) {
				let v = (x >> (16 + 5 * k)) % (span + 1);
				*id = (v < span).then(|| Rank::ALL[num::pick(v * 13 / span, Rank::ALL.len())]);
			}
			let groups = same_rank(&rank_masks(&ids[..n]));
			for size in SMALLEST_GROUP..=LARGEST_GROUP {
				assert_eq!(
					(groups.masks[size - SMALLEST_GROUP], groups.count(size)),
					x_same(size, &ids[..n]),
					"groups of {size} in {:?}",
					&ids[..n]
				);
			}
		}
	}
}
