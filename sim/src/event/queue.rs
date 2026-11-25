//! `G.E_MANAGER`'s base FIFO queue, trigger/blocking flags, and frame timers.

use std::collections::VecDeque;
use std::sync::OnceLock;

use super::kind::{Draw, Event};
use crate::data::Edition;
use crate::game::State;

/// Game speed (`G.SETTINGS.GAMESPEED`) the timing model assumes (the
/// game's default).
pub(crate) const GAMESPEED: f64 = 1.0;
/// Real seconds per frame (60 fps).
pub(super) const DT: f64 = 1.0 / 60.0;

/// The frames [`totals`] covers; an action that runs longer steps the rest
/// frame by frame.
const TOTALS_LEN: usize = 1 << 16;

/// `G.TIMERS.TOTAL` after `n` frames at speed factor 1, accumulated frame by
/// frame exactly like the game does.
pub(super) fn totals() -> &'static [f64] {
	static T: OnceLock<Box<[f64]>> = OnceLock::new();
	T.get_or_init(|| {
		let mut v = Vec::with_capacity(TOTALS_LEN);
		let mut t = 0.0f64;
		v.push(t);
		for _ in 1..TOTALS_LEN {
			t += DT;
			v.push(t);
		}
		v.into_boxed_slice()
	})
}

/// An event's `trigger` and `delay` (in `G.TIMERS.TOTAL` seconds).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Trigger {
	/// `'immediate'`.
	Immediate,
	/// `'after'`: runs once `delay` has elapsed since first handled.
	After(f64),
	/// `'before'`: runs when first handled, completes after `delay`.
	Before(f64),
	/// `'ease'`: completes once strictly more than `delay` has elapsed.
	Ease(f64),
	/// An immediate event polling `TOTAL - start > delay` (`Card:explode`).
	Until { start: f64, delay: f64 },
	/// An immediate event polling a UI condition that holds `n` frames
	/// after its first call (measured; UI motion is not simulated).
	Frames(u32),
}

/// An event's `blocking` and `blockable` flags.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Blocks {
	/// Blocking and blockable (an `Event`'s defaults): runs in queue order.
	InOrder,
	/// Blocking but not blockable: runs even behind a blocking event, and
	/// holds back every blockable event queued after it.
	Barrier,
	/// Blockable but not blocking: later events do not wait for it.
	Background,
	/// Neither blocking nor blockable.
	Detached,
}

/// An entry of `G.E_MANAGER.queues.base`.
#[derive(Clone, Copy, Debug)]
pub(super) struct Queued {
	pub(super) event: Event,
	pub(super) trigger: Trigger,
	pub(super) blocking: bool,
	pub(super) blockable: bool,
	/// `Event.time`: when the event was first handled, once it has been.
	pub(super) time: Option<f64>,
	/// `Event.complete`.
	pub(super) done: bool,
}

/// When a new edition shows (`immediate` of `Card:set_edition`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EditionTiming {
	Immediate,
	Delayed,
}

/// Room for the events one action queues at a time.
pub(super) const QUEUE_ROOM: usize = 64;

/// `G.E_MANAGER`'s base queue and the timers it runs on.
#[derive(Debug)]
pub(crate) struct EventQueue {
	pub(super) queue: VecDeque<Queued>,
	/// Number of non-blockable events in `queue`.
	pub(super) unblockable: u32,
	/// `G.TIMERS.TOTAL`.
	pub(super) total: f64,
	/// Frames since the current action started.
	pub(super) frames: u32,
	/// `total == totals()[frames]` (no speed-up so far in this action).
	pub(super) on_table: bool,
	/// `G.ACC` / `G.ACC_state`.
	pub(super) acc: f64,
	pub(super) acc_state: State,
}

impl EventQueue {
	pub(crate) fn new() -> EventQueue {
		EventQueue {
			queue: VecDeque::with_capacity(QUEUE_ROOM),
			unblockable: 0,
			total: 0.0,
			frames: 0,
			on_table: true,
			acc: 0.0,
			acc_state: State::BlindSelect,
		}
	}

	pub(crate) fn clear(&mut self) {
		self.queue.clear();
		self.unblockable = 0;
	}

	pub(crate) fn take_pending(&mut self) -> Vec<Event> {
		self.unblockable = 0;
		self.queue.drain(..).map(|q| q.event).collect()
	}

	/// `G.E_MANAGER:add_event(Event({func = ...}))` (immediate, blocking).
	#[inline]
	pub(crate) fn push(&mut self, e: Event) {
		self.push_event(e, Trigger::Immediate, Blocks::InOrder);
	}

	#[inline]
	pub(crate) fn push_event(&mut self, event: Event, trigger: Trigger, blocks: Blocks) {
		let (blocking, blockable) = match blocks {
			Blocks::InOrder => (true, true),
			Blocks::Barrier => (true, false),
			Blocks::Background => (false, true),
			Blocks::Detached => (false, false),
		};
		if !blockable {
			self.unblockable += 1;
		}
		self.queue.push_back(Queued {
			event,
			trigger,
			blocking,
			blockable,
			time: None,
			done: false,
		});
	}

	/// `draw_card(from, to, ..., delay)`: a `'before'` event.
	#[inline]
	pub(crate) fn draw(&mut self, draw: Draw, delay: f64) {
		self.push_before(delay, Event::DrawCard(draw));
	}

	#[inline]
	pub(crate) fn push_after(&mut self, d: f64, e: Event) {
		self.push_event(e, Trigger::After(d), Blocks::InOrder);
	}

	#[inline]
	pub(crate) fn push_before(&mut self, d: f64, e: Event) {
		self.push_event(e, Trigger::Before(d), Blocks::InOrder);
	}

	/// `delay(t)`, and any purely visual `'after'` event (kept for its
	/// timing).
	#[inline]
	pub(crate) fn delay(&mut self, d: f64) {
		self.push_event(Event::Wait, Trigger::After(d), Blocks::InOrder);
	}

	/// A blocking event with no game effect (`trigger`, `delay`).
	#[inline]
	pub(crate) fn wait(&mut self, t: Trigger) {
		self.push_event(Event::Wait, t, Blocks::InOrder);
	}

	/// A non-blockable (but blocking) event with no game effect: holds back
	/// every event queued after it.
	#[inline]
	pub(crate) fn barrier(&mut self, t: Trigger) {
		self.push_event(Event::Wait, t, Blocks::Barrier);
	}

	/// `update_hand_text({delay = d}, ...)`.
	#[inline]
	pub(crate) fn hand_text(&mut self, d: f64) {
		self.wait(Trigger::Before(d));
	}

	/// `update_hand_text({immediate = true, delay = d}, ...)` (non-blockable).
	#[inline]
	pub(crate) fn hand_text_immediate(&mut self, d: f64) {
		self.barrier(Trigger::Before(d));
	}

	/// `highlight_card(card, ...)`.
	#[inline]
	pub(crate) fn highlight(&mut self) {
		self.wait(Trigger::Before(0.1));
	}

	/// The events `Card:set_edition(edition, immediate)` queues (not silent).
	pub(crate) fn edition_fx(&mut self, e: Option<Edition>, timing: EditionTiming) {
		if e.is_none() {
			return;
		}
		match timing {
			EditionTiming::Immediate => self.barrier(Trigger::After(0.0)),
			EditionTiming::Delayed => self.delay(0.2),
		}
		self.delay(0.1);
	}

	/// `card_eval_status_text(card, ...)` with an explicit base `delay`: a
	/// `'before'` event of `delay*1.25`.
	#[inline]
	pub(crate) fn status_text(&mut self, delay: f64) {
		self.wait(Trigger::Before(delay * 1.25));
	}

	/// `card_eval_status_text(card, 'extra', ...)` without `extra.delay`.
	#[inline]
	pub(crate) fn status_extra(&mut self) {
		self.status_text(0.75);
	}

	/// `Card:start_dissolve(colours, silent, dissolve_time_fac)`.
	pub(crate) fn start_dissolve_scaled(&mut self, id: u32, time_factor: f64) {
		let dt = 0.7 * time_factor;
		self.push_event(
			Event::RemoveCard { id },
			Trigger::After(1.05 * dt),
			Blocks::Barrier,
		);
		self.barrier(Trigger::After(1.051 * dt));
	}

	#[inline]
	pub(crate) fn start_dissolve(&mut self, id: u32) {
		self.start_dissolve_scaled(id, 1.0);
	}

	/// `Card:shatter()`.
	pub(crate) fn shatter(&mut self, id: u32) {
		let dt = 0.7;
		self.push_event(
			Event::RemoveCard { id },
			Trigger::After(0.55 * dt),
			Blocks::Barrier,
		);
		self.barrier(Trigger::After(0.51 * dt));
	}

	/// `Card:explode()`.
	pub(crate) fn explode(&mut self, id: u32) {
		let et = 1.3 * GAMESPEED.sqrt();
		self.barrier(Trigger::Until {
			start: self.total,
			delay: 1.5 * et,
		});
		self.push_event(
			Event::RemoveCard { id },
			Trigger::After(1.5 * et),
			Blocks::Barrier,
		);
	}

	/// `Card:start_materialize(colours, silent, timefac)`: only its timing.
	pub(crate) fn materialize(&mut self, time_factor: f64) {
		let dt = 0.6 * time_factor;
		self.barrier(Trigger::After(0.5 * dt));
		self.barrier(Trigger::Ease(dt));
		self.barrier(Trigger::After(1.05 * dt));
	}

	/// Earliest `TOTAL` at which a waiting event can progress.
	pub(super) fn next_due(&self) -> f64 {
		let mut due = f64::INFINITY;
		for q in &self.queue {
			let Some(time) = q.time else { continue };
			let t = match q.trigger {
				Trigger::Immediate => continue,
				Trigger::After(d) | Trigger::Before(d) | Trigger::Ease(d) => time + d,
				Trigger::Until { start, delay } => start + delay,
				Trigger::Frames(_) => self.total,
			};
			due = due.min(t);
		}
		due
	}
}

impl Clone for EventQueue {
	fn clone(&self) -> EventQueue {
		EventQueue {
			queue: self.queue.clone(),
			..*self
		}
	}

	fn clone_from(&mut self, source: &EventQueue) {
		let EventQueue {
			queue,
			unblockable,
			total,
			frames,
			on_table,
			acc,
			acc_state,
		} = self;
		queue.clone_from(&source.queue);
		*unblockable = source.unblockable;
		*total = source.total;
		*frames = source.frames;
		*on_table = source.on_table;
		*acc = source.acc;
		*acc_state = source.acc_state;
	}
}
