//! Cards: non-playing cards ([`Card`]), playing cards ([`PlayingCard`]),
//! card masks ([`CardMask`]), and card lifecycle operations.

mod any;
mod item;
mod lifecycle;
pub(crate) mod mask;
mod playing_card;

pub use any::{AnyCard, CardLike};
pub(crate) use item::has_joker;
pub use item::{Ability, Card, CardCore, Kind, Sticker};
pub(crate) use lifecycle::{
	Pricing, YORICK_DISCARDS, add_to_deck, destroy_card, make_card, make_playing_card, mark_used,
	remove_from_deck, remove_owned, set_cost_all, set_debuff_joker, set_edition, set_enhancement,
	shatter_or_dissolve, unmark_used,
};
pub(crate) use mask::CardMask;
pub(crate) use playing_card::{CanonKey, CardFace, canon_key};
pub use playing_card::{Forced, HandSort, PlayingCard, SuitTest, UNIQUE_ID_SCALE};
