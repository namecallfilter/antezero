//! A set of hand positions: the cards an action selects.

use std::fmt;
use std::num::ParseIntError;
use std::ops::{BitAnd, BitOr, Not};
use std::str::FromStr;

use crate::num;

/// Hand cards by position: bit `i` selects the `i`-th card of the hand.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CardMask(u32);

impl CardMask {
	/// The most hand positions a mask holds.
	pub const CAPACITY: usize = u32::BITS as usize;
	pub const EMPTY: CardMask = CardMask(0);

	#[must_use]
	pub const fn from_bits(bits: u32) -> CardMask {
		CardMask(bits)
	}

	#[must_use]
	pub const fn bits(self) -> u32 {
		self.0
	}

	/// Every position of a hand of `n` cards.
	#[must_use]
	pub fn first(n: usize) -> CardMask {
		CardMask(if n >= Self::CAPACITY {
			u32::MAX
		} else {
			(1 << n) - 1
		})
	}

	#[inline]
	#[must_use]
	pub fn contains(self, i: usize) -> bool {
		i < Self::CAPACITY && self.0 >> i & 1 == 1
	}

	/// Whether every position of `other` is in `self`.
	#[inline]
	#[must_use]
	pub fn contains_all(self, other: CardMask) -> bool {
		self.0 & other.0 == other.0
	}

	/// # Panics
	///
	/// Panics if `i` is not a position a mask holds.
	#[inline]
	pub fn insert(&mut self, i: usize) {
		assert!(
			i < Self::CAPACITY,
			"position {i} is past the {} a mask holds",
			Self::CAPACITY
		);
		self.0 |= 1 << i;
	}

	#[inline]
	#[must_use]
	pub fn len(self) -> usize {
		self.0.count_ones() as usize
	}

	#[inline]
	#[must_use]
	pub fn is_empty(self) -> bool {
		self.0 == 0
	}

	/// The lowest position.
	#[must_use]
	pub fn lowest(self) -> Option<usize> {
		(!self.is_empty()).then(|| self.0.trailing_zeros() as usize)
	}

	/// The positions, in ascending order.
	#[inline]
	#[must_use]
	pub fn indices(self) -> Indices {
		Indices(self.0)
	}

	/// This mask with the lowest positions of `from` added until it holds `n`
	/// (or `from` runs out).
	#[must_use]
	pub fn fill_lowest(self, from: CardMask, n: usize) -> CardMask {
		let mut m = self;
		for i in from.indices() {
			if m.len() >= n {
				break;
			}
			m.insert(i);
		}
		m
	}

	/// Whether every position is in a hand of `n` cards.
	#[inline]
	#[must_use]
	pub fn fits(self, n: usize) -> bool {
		self.0.checked_shr(num::index32(n)).unwrap_or(0) == 0
	}

	/// Calls `f` for every non-empty subset of `n` positions with at most
	/// `max` members, in increasing mask order.
	///
	/// # Panics
	///
	/// Panics if `n` is more than the positions a mask holds.
	#[inline]
	pub(crate) fn for_each_subset(n: usize, max: usize, mut f: impl FnMut(CardMask)) {
		if n == 0 {
			return;
		}
		assert!(
			n <= Self::CAPACITY,
			"a card mask holds {} cards, not {n}",
			Self::CAPACITY
		);
		let all = Self::first(n).0;
		let mut mask: u32 = 1;
		loop {
			if mask.count_ones() as usize > max {
				// Skip ahead to the next mask with fewer bits set.
				match mask.checked_add(mask.isolate_lowest_one()) {
					Some(next) if next <= all => mask = next,
					_ => return,
				}
				continue;
			}
			f(CardMask(mask));
			if mask == all {
				return;
			}
			mask += 1;
		}
	}
}

/// The positions of a [`CardMask`], in ascending order.
#[derive(Clone, Debug)]
pub struct Indices(u32);

impl Iterator for Indices {
	type Item = usize;

	#[inline]
	fn next(&mut self) -> Option<usize> {
		if self.0 == 0 {
			return None;
		}
		let i = self.0.trailing_zeros() as usize;
		self.0 &= self.0 - 1;
		Some(i)
	}

	fn size_hint(&self) -> (usize, Option<usize>) {
		let n = self.0.count_ones() as usize;
		(n, Some(n))
	}
}

impl ExactSizeIterator for Indices {}

impl FromIterator<usize> for CardMask {
	/// The mask of the given positions.
	///
	/// # Panics
	///
	/// Panics on a position past the [`CardMask::CAPACITY`] a mask holds.
	fn from_iter<I: IntoIterator<Item = usize>>(positions: I) -> CardMask {
		let mut m = CardMask::EMPTY;
		for i in positions {
			m.insert(i);
		}
		m
	}
}

impl BitAnd for CardMask {
	type Output = CardMask;

	fn bitand(self, other: CardMask) -> CardMask {
		CardMask(self.0 & other.0)
	}
}

impl BitOr for CardMask {
	type Output = CardMask;

	fn bitor(self, other: CardMask) -> CardMask {
		CardMask(self.0 | other.0)
	}
}

impl Not for CardMask {
	type Output = CardMask;

	fn not(self) -> CardMask {
		CardMask(!self.0)
	}
}

/// The bits in decimal, as the action codes write them.
impl fmt::Display for CardMask {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		self.0.fmt(f)
	}
}

impl FromStr for CardMask {
	type Err = ParseIntError;

	fn from_str(s: &str) -> Result<CardMask, ParseIntError> {
		s.parse().map(CardMask)
	}
}
