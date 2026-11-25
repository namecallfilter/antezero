//! Emulation of `G.E_MANAGER` and of `Game:update`'s per-state handlers:
//! the event definitions, the queue and the event runner.
//!
//! The game's flows are chains of events on a FIFO queue; events created while
//! an event runs are appended at the tail, and the per-state `update_*`
//! handlers run after the event that changed the state. The queue is stepped
//! frame by frame (`dt = 1/60`, `G.TIMERS.TOTAL` rebased to 0 per action)
//! with the game's `trigger` / `delay` / `blocking` / `blockable` semantics,
//! so event order and frame timing match the real game. Purely visual events
//! are kept as timing-only placeholders (`Event::Wait`).

mod kind;
mod queue;
mod runner;

pub(crate) use kind::{Draw, Event};
pub(crate) use queue::{Blocks, EditionTiming, EventQueue, GAMESPEED, Trigger};
pub(crate) use runner::{
	begin_action, level_up_hand, run_queue, run_until_played, run_until_scored,
};
