//! Tarots, planets, and spectral cards: usability checks, queueing, and
//! deferred consumable events.

mod event;
mod use_card;

pub(crate) use event::{ConsumableEvent, run_event};
pub(crate) use use_card::{can_use_consumable, targets, use_consumable};
