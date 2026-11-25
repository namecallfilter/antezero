//! Bit sets of the game's enums: centers (`G.GAME.used_jokers`), poker
//! hands, planets, vouchers, suits and tags.

use super::defs::{CENTER_COUNT, Center};
use super::primitives::{HandType, Suit};
use crate::data_gen::{Planet, TagKind, Voucher};

/// A set of centers, one bit each (`G.GAME.used_jokers`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CenterSet([u64; CENTER_COUNT.div_ceil(64)]);

impl CenterSet {
	#[inline]
	#[must_use]
	pub fn contains(&self, c: Center) -> bool {
		let i = c.index();
		self.0[i >> 6] >> (i & 63) & 1 == 1
	}

	#[inline]
	pub fn insert(&mut self, c: Center) {
		let i = c.index();
		self.0[i >> 6] |= 1 << (i & 63);
	}

	#[inline]
	pub fn remove(&mut self, c: Center) {
		let i = c.index();
		self.0[i >> 6] &= !(1 << (i & 63));
	}
}

/// A set of the variants of a fieldless enum, one bit each.
macro_rules! variant_set {
	($(#[$doc:meta])* $name:ident($bits:ty) of $item:ty) => {
		$(#[$doc])*
		#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
		pub struct $name($bits);

		impl $name {
			#[inline]
			#[must_use]
			pub fn contains(self, item: $item) -> bool {
				self.0 >> (item as u32) & 1 == 1
			}

			#[inline]
			pub fn insert(&mut self, item: $item) {
				self.0 |= 1 << (item as u32);
			}

			#[must_use]
			pub fn count(self) -> u32 {
				self.0.count_ones()
			}
		}
	};
}

variant_set!(
	/// A set of poker hands.
	HandSet(u16) of HandType
);
variant_set!(
	/// A set of planets.
	PlanetSet(u16) of Planet
);
variant_set!(
	/// A set of vouchers (`G.GAME.used_vouchers`).
	VoucherSet(u32) of Voucher
);
variant_set!(
	/// A set of suits.
	SuitSet(u8) of Suit
);
variant_set!(
	/// A set of skip tags.
	TagSet(u32) of TagKind
);
