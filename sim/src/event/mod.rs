//! Emulation of `G.E_MANAGER` and `Game:update`'s per-state handlers: event
//! definitions, FIFO queue with frame timing, and the event runner.

mod kind;
mod queue;
mod runner;

pub(crate) use kind::{Draw, Event};
pub(crate) use queue::{Blocks, EditionTiming, EventQueue, GAMESPEED, Trigger};
pub(crate) use runner::{
	begin_action, level_up_hand, run_queue, run_until_played, run_until_scored,
};
