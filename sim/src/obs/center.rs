//! Center id vocabulary (0 = none).

use crate::card::Kind;
use crate::data::{ALL_BLINDS, ALL_TAGS, Center, ENHANCED_BASE, HandType};
use crate::num;

pub const PLAYING: i32 = 1;
/// The first card center: the other cards (jokers, consumables, vouchers,
/// boosters) follow in [`Center::index`](crate::data::Center::index) order.
pub const JOKER0: i32 = 2;
pub const TAG0: i32 = JOKER0 + num::id(ENHANCED_BASE);
pub const BLIND0: i32 = TAG0 + num::id(ALL_TAGS.len());
pub const HAND0: i32 = BLIND0 + num::id(ALL_BLINDS.len());
/// A face-down card or joker.
pub const HIDDEN: i32 = HAND0 + num::id(HandType::ALL.len());
pub const COUNT: i32 = HIDDEN + 1;
const _: () = assert!(
	COUNT == 335,
	"the vocabulary is the size of the trained checkpoints' center embedding"
);

/// The center id of a card that is not a playing card.
pub(super) fn of_kind(k: Kind) -> i32 {
	of(k.center())
}

/// The center id of a card center.
#[must_use]
pub fn of(c: Center) -> i32 {
	JOKER0 + num::id(c.index())
}
