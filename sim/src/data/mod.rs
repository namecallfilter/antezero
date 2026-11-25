//! Static game data: enums, prototype definitions and the orderings the game
//! uses when sampling. Large tables live in the generated `data_gen.rs`.

mod defs;
mod primitives;
mod sets;

pub(crate) use defs::ENHANCED_BASE;
pub use defs::{
	BackDef, BlindDef, BoosterDef, Boss, CENTER_COUNT, Center, ConsumableDef, JokerDef, PoolFlag,
	Rarity, TagDef, VoucherDef,
};
pub use primitives::{
	Edition, Enhancement, HandBase, HandType, P_CARDS_SORTED, PackKind, Rank, Seal, Suit,
};
pub use sets::{CenterSet, HandSet, PlanetSet, VoucherSet};
pub(crate) use sets::{SuitSet, TagSet};

pub use crate::data_gen::{
	ALL_BACKS, ALL_BLINDS, ALL_BOOSTERS, ALL_JOKERS, ALL_PLANETS, ALL_SPECTRALS, ALL_TAGS,
	ALL_TAROTS, ALL_VOUCHERS, BACK_DEFS, BLIND_DEFS, BOOSTER_DEFS, Back, BlindKind, Booster,
	JOKER_COUNT, JOKER_DEFS, Joker, PLANET_DEFS, Planet, RARITY_POOL_1, RARITY_POOL_2,
	RARITY_POOL_3, RARITY_POOL_4, SPECTRAL_DEFS, Spectral, TAG_DEFS, TAROT_DEFS, TagKind, Tarot,
	VOUCHER_DEFS, Voucher,
};
