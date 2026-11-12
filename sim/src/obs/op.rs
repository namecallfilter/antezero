//! Op kinds.

pub const SELECT_BLIND: i32 = 0;
pub const SKIP_BLIND: i32 = 1;
pub const REROLL_BOSS: i32 = 2;
pub const PLAY: i32 = 3;
pub const DISCARD: i32 = 4;
pub const USE: i32 = 5;
pub const SELL_JOKER: i32 = 6;
pub const SELL_CONSUMABLE: i32 = 7;
pub const BUY: i32 = 8;
pub const BUY_AND_USE: i32 = 9;
pub const REDEEM: i32 = 10;
pub const OPEN: i32 = 11;
pub const REROLL: i32 = 12;
pub const END_SHOP: i32 = 13;
pub const CASH_OUT: i32 = 14;
pub const PICK: i32 = 15;
pub const SKIP_PACK: i32 = 16;
pub const MOVE_JOKER: i32 = 17;
pub const SORT_RANK: i32 = 18;
pub const SORT_SUIT: i32 = 19;
pub const MOVE_HAND: i32 = 20;
/// One of the top-scoring plays (cards fixed; see `ops`).
pub const PLAY_TOP: i32 = 21;
pub const COUNT: i32 = 22;
