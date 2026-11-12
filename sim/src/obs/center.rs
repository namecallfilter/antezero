//! Center id vocabulary (0 = none).

use crate::data::{
	ALL_BLINDS, ALL_BOOSTERS, ALL_JOKERS, ALL_PLANETS, ALL_SPECTRALS, ALL_TAGS, ALL_TAROTS,
	ALL_VOUCHERS, HandType,
};
use crate::num;

pub const PLAYING: i32 = 1;
pub const JOKER0: i32 = 2;
pub const TAROT0: i32 = JOKER0 + num::id(ALL_JOKERS.len());
pub const PLANET0: i32 = TAROT0 + num::id(ALL_TAROTS.len());
pub const SPECTRAL0: i32 = PLANET0 + num::id(ALL_PLANETS.len());
pub const VOUCHER0: i32 = SPECTRAL0 + num::id(ALL_SPECTRALS.len());
pub const BOOSTER0: i32 = VOUCHER0 + num::id(ALL_VOUCHERS.len());
pub const TAG0: i32 = BOOSTER0 + num::id(ALL_BOOSTERS.len());
pub const BLIND0: i32 = TAG0 + num::id(ALL_TAGS.len());
pub const HAND0: i32 = BLIND0 + num::id(ALL_BLINDS.len());
/// A face-down card or joker.
pub const HIDDEN: i32 = HAND0 + num::id(HandType::ALL.len());
pub const COUNT: i32 = HIDDEN + 1;
const _: () = assert!(COUNT == 335);
