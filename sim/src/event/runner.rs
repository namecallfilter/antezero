//! Frame-by-frame event loop (`G.E_MANAGER:update`) and `Game:update` state
//! dispatch.

use super::kind::{Draw, Event};
use super::queue::{Blocks, DT, EventQueue, GAMESPEED, QUEUE_ROOM, Queued, Trigger, totals};
use crate::areas::Pile;
use crate::data::HandType;
use crate::game::{Flow, Game, Hand, HandLevels, State};
use crate::pools::OwnedArea;
use crate::rng::PseudoRandom;
use crate::{card, consumables, flow, joker, num, pools, round, run, shop};

/// `level_up_hand(card, hand, nil, amount)` including its events.
pub(crate) fn level_up_hand(
	hands: &mut HandLevels, events: &mut EventQueue, h: HandType, amount: f64,
) {
	hands.level_up(h, amount);
	events.delay(0.2);
	events.hand_text(0.0);
	events.delay(0.9);
	events.hand_text(0.0);
	events.delay(0.9);
	events.hand_text(0.0);
	events.delay(1.3);
	events.push(Event::Wait);
}

/// What an event's `func` has returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Status {
	/// False, or it has not run: the event stays queued.
	Pending,
	/// True.
	Done,
	/// True, and the game paused (`G.SETTINGS.paused`): the rest of this
	/// frame's pass is skipped.
	DoneAndPaused,
}

/// What `Event:handle` reports about an event this frame.
struct Handled {
	/// What its `func` has returned (now or on an earlier frame).
	status: Status,
	/// Its delay has elapsed.
	time_done: bool,
	/// Something happened that a later pass could build on.
	changed: bool,
}

impl Handled {
	/// The event's `func` ran now, with its delay elapsed.
	fn ran(status: Status) -> Handled {
		Handled {
			status,
			time_done: true,
			changed: true,
		}
	}
}

/// `Event:handle`: runs the event's `func` if its trigger allows; `time` is
/// when the event was first handled.
#[inline]
fn handle(g: &mut Game, q: &Queued, time: f64) -> Handled {
	let waiting = Handled {
		status: Status::Pending,
		time_done: false,
		changed: false,
	};
	match q.trigger {
		Trigger::Immediate => Handled::ran(run_event(g, q.event)),
		Trigger::After(d) => {
			if time + d <= g.events.total {
				Handled::ran(run_event(g, q.event))
			} else {
				waiting
			}
		}
		Trigger::Before(d) => Handled {
			status: if q.done {
				Status::Done
			} else {
				run_event(g, q.event)
			},
			time_done: time + d <= g.events.total,
			changed: !q.done,
		},
		Trigger::Ease(d) => {
			if time + d >= g.events.total {
				waiting
			} else {
				Handled::ran(run_event(g, q.event))
			}
		}
		Trigger::Until { start, delay: d } => {
			if g.events.total - start > d {
				Handled::ran(run_event(g, q.event))
			} else {
				Handled {
					time_done: true,
					..waiting
				}
			}
		}
		Trigger::Frames(n) => {
			if f64::from(g.events.frames) - time >= f64::from(n) {
				Handled::ran(run_event(g, q.event))
			} else {
				Handled {
					time_done: true,
					changed: true,
					..waiting
				}
			}
		}
	}
}

/// One frame of `Game:update` after the timers advanced:
/// `G.E_MANAGER:update`, then the state handlers. Returns whether
/// anything happened.
fn pass(g: &mut Game) -> bool {
	let mut changed = false;
	let mut blocked = false;
	let mut seen_unblockable = 0u32;
	let mut i = 0;
	while i < g.events.queue.len() {
		let q = g.events.queue[i];
		if blocked && q.blockable {
			// Only non-blockable events can still run this frame.
			if seen_unblockable == g.events.unblockable {
				break;
			}
			i += 1;
			continue;
		}
		if !q.blockable {
			seen_unblockable += 1;
		}
		let time = if let Some(time) = q.time {
			time
		} else {
			changed = true;
			let now = if matches!(q.trigger, Trigger::Frames(_)) {
				f64::from(g.events.frames)
			} else {
				g.events.total
			};
			g.events.queue[i].time = Some(now);
			now
		};
		let handled = handle(g, &q, time);
		changed |= handled.changed;
		let completed = handled.status != Status::Pending;
		let time_done = handled.time_done;
		if q.blocking {
			blocked = true;
		}
		if handled.status == Status::DoneAndPaused {
			if time_done {
				g.events.queue.remove(i);
				if !q.blockable {
					g.events.unblockable -= 1;
				}
			}
			break;
		}
		if completed && time_done {
			#[cfg(feature = "trace-keys")]
			g.rng.trace.log_event(format_args!(
				"{} {:?} {:?}{}{}",
				g.events.frames,
				q.event,
				q.trigger,
				if q.blocking { "" } else { " nb" },
				if q.blockable { "" } else { " nbl" }
			));
			g.events.queue.remove(i);
			if !q.blockable {
				g.events.unblockable -= 1;
				seen_unblockable -= 1;
			}
			changed = true;
		} else {
			if completed {
				g.events.queue[i].done = true;
			}
			i += 1;
		}
	}
	let before = (g.flow.state, g.flow.state_complete, g.events.queue.len());
	tick_state(g);
	changed || before != (g.flow.state, g.flow.state_complete, g.events.queue.len())
}

/// `G.TIMERS.TOTAL` advance (with the `G.ACC` speed-up of long hands).
#[inline]
fn advance_time(state: State, rng: &mut PseudoRandom, events: &mut EventQueue) {
	events.frames += 1;
	rng.trace.set_frame(events.frames);
	if state != events.acc_state {
		events.acc = 0.0;
	}
	events.acc_state = state;
	if matches!(state, State::HandPlayed | State::NewRound) {
		events.acc = (events.acc + DT * 0.2 * GAMESPEED).min(16.0);
	} else {
		events.acc = 0.0;
	}
	let sf = GAMESPEED + (events.acc.abs() - 2.0).max(0.0);
	if sf != 1.0 {
		events.on_table = false;
	}
	events.total += DT * sf;
}

/// Advances idle frames until `TOTAL >= due`. While every frame of the
/// action advanced `TOTAL` by exactly `DT` (no `G.ACC` speed-up), `TOTAL`
/// after `n` frames is `totals()[n]`, so the skip is a table lookup with
/// bit-identical results.
fn skip_until(state: State, rng: &mut PseudoRandom, events: &mut EventQueue, due: f64) {
	if events.on_table && !matches!(state, State::HandPlayed | State::NewRound) {
		let t = totals();
		let cur = num::index(events.frames);
		if cur < t.len() && events.total == t[cur] {
			// First frame n > cur with t[n] >= due.
			let mut n = num::trunc(due / DT).max(cur + 1);
			while n > cur + 1 && t.get(n - 1).is_some_and(|&v| v >= due) {
				n -= 1;
			}
			while n < t.len() && t[n] < due {
				n += 1;
			}
			if n < t.len() {
				events.frames = num::index32(n);
				events.total = t[n];
				events.acc = 0.0;
				events.acc_state = state;
				rng.trace.set_frame(events.frames);
				return;
			}
		}
	}
	while events.total < due {
		advance_time(state, rng, events);
	}
}

/// Passes of the event loop after which an action must be stuck (a
/// simulator bug); no action comes near it.
const STUCK_AFTER: u32 = 100_000_000;

/// Every action starts at `G.TIMERS.TOTAL = 0` (the oracle rebases its
/// timers the same way), so timer arithmetic is reproducible.
pub(crate) fn begin_action(rng: &mut PseudoRandom, events: &mut EventQueue) {
	// A cloned game's queue has no spare room: make it once, not by doubling.
	events.queue.reserve(QUEUE_ROOM);
	events.total = 0.0;
	events.frames = 0;
	events.on_table = true;
	rng.trace.clear();
}

/// Runs frames until `done` holds after one (true), or until nothing is
/// left to happen before the next input (false).
///
/// # Panics
///
/// Panics if the event queue stalls, or if neither happens (a simulator
/// bug).
fn run_frames(g: &mut Game, done: impl Fn(&Flow) -> bool) -> bool {
	let mut guard = 0u32;
	loop {
		advance_time(g.flow.state, &mut g.rng, &mut g.events);
		if !pass(g) && !done(&g.flow) {
			if g.events.queue.is_empty() {
				return false;
			}
			// Nothing can happen before a timer expires: skip idle
			// frames (their passes and state handlers are no-ops).
			let due = g.events.next_due();
			assert!(due.is_finite(), "event queue stalled");
			skip_until(g.flow.state, &mut g.rng, &mut g.events, due);
			pass(g);
		}
		if done(&g.flow) {
			return true;
		}
		guard += 1;
		assert!(guard < STUCK_AFTER, "event loop did not terminate");
	}
}

/// Runs frames until `evaluate_play` has run (the rest of the chain is
/// abandoned: the game is left mid-animation and must be discarded).
///
/// # Panics
///
/// Panics if the events run out or stall before the hand is scored (a
/// simulator bug).
pub(crate) fn run_until_scored(g: &mut Game) {
	g.flow.scored = false;
	let scored = run_frames(g, |flow| flow.scored);
	assert!(
		scored,
		"the play's events ran out before the hand was scored"
	);
}

/// Runs frames until the played hand's own chain has run: its score, money
/// and created or destroyed cards are applied and the played cards are
/// discarded; nothing is drawn yet and the round has not ended. The game is
/// left mid-animation and must be discarded.
///
/// # Panics
///
/// Panics if the events run out or stall before the hand is played out (a
/// simulator bug).
pub(crate) fn run_until_played(g: &mut Game) {
	g.flow.scored = false;
	let played = run_frames(g, |flow| flow.scored && flow.state != State::HandPlayed);
	assert!(
		played,
		"the play's events ran out before the hand was played out"
	);
}

/// Runs frames until the game waits for input.
///
/// # Panics
///
/// Panics if the event queue stalls, never empties, or empties in a state
/// that is not a decision point (a simulator bug).
pub(crate) fn run_queue(g: &mut Game) {
	// The game-over screen pauses the game, and events queued before a pause
	// never run (a lost round's rent, say): the simulator stops there too.
	run_frames(g, |flow| flow.state == State::GameOver);
	flow::sync_phase(&mut g.flow);
}

fn run_event(g: &mut Game, e: Event) -> Status {
	match e {
		Event::Wait => {}
		Event::PushHandTextImmediate => g.events.hand_text_immediate(0.0),
		Event::Materialize => g.events.materialize(1.0),
		Event::SetChips(c) => g.round.chips = c,
		Event::WinPoll => {
			if g.flow.state != State::RoundEval {
				return Status::Pending;
			}
			g.run.won = true;
			g.events.push(Event::WinPause);
			// round_text = 'Endless Round ' once unpaused
			g.events.push(Event::Wait);
		}
		Event::WinPause => return Status::DoneAndPaused,
		Event::EaseChips0 => {
			g.events
				.push_event(Event::SetChips(0.0), Trigger::Ease(0.3), Blocks::Barrier);
		}
		Event::StatusLater(d) => g.events.status_text(d),
		Event::DollarsStatus { dollars, delay } => {
			g.events.push(Event::EaseDollars(dollars));
			g.events.status_text(delay);
		}
		Event::EaseDollars(d) => flow::ease_dollars(&mut g.run, d),
		Event::EaseDiscard(d) => flow::ease_discard(&mut g.round, d),
		Event::EaseHands(d) => flow::ease_hands_played(&mut g.round, d),
		Event::EaseRound => g.run.round += 1,
		Event::EaseAnte(d) => g.run.ante += d,
		Event::DollarBufferReset => g.run.dollar_buffer = 0.0,
		Event::HandSize(d) => apply_hand_size(g.flow.state, &mut g.areas.hand, &mut g.events, d),
		Event::SortHand => g.areas.hand.sort(),
		Event::RefreshDebuffs => flow::refresh_playing_debuffs(&mut g.areas, &g.blind),
		Event::SetCostAll => card::set_cost_all(&mut g.areas, g.shop.discount_percent),
		Event::CheckeredDeck => run::checkered_deck(&mut g.areas, &g.blind),
		Event::Create(spec) => {
			pools::create_owned(g, spec);
			match spec.area {
				OwnedArea::Jokers => g.areas.joker_buffer = 0.0,
				OwnedArea::Consumables => g.areas.consumable_buffer = 0.0,
			}
		}
		Event::DissolvePlaying { id } => {
			let card = g
				.areas
				.playing_cards_mut()
				.find(|c| c.core.id == id)
				.expect("a destroyed card stays in its area until this event dissolves it");
			card::shatter_or_dissolve(card, &mut g.events);
		}
		Event::MarkPlayed { id } => {
			if let Some(c) = g.areas.playing_cards_mut().find(|c| c.core.id == id) {
				c.played_this_ante = true;
			}
		}
		Event::RemoveCard { id } => flow::remove_card_by_id(g, id),
		Event::BlindReset => round::set_blind(g, None),
		Event::DrawCard(draw) => {
			flow::draw_card(
				&mut g.areas,
				g.run.probabilities,
				&g.round,
				&g.blind,
				&mut g.rng,
				draw,
			);
		}
		Event::Flow(e) => flow::run_event(g, e),
		Event::Joker(e) => joker::run_event(g, e),
		Event::Shop(e) => shop::run_event(g, e),
		Event::Pack(e) => shop::run_pack_event(g, e),
		Event::Tag(e) => run::run_tag_event(g, e),
		Event::Consumable(e) => consumables::run_event(g, e),
	}
	Status::Done
}

/// `Game:update`'s state dispatch (after each event-queue pass).
fn tick_state(g: &mut Game) {
	// A handler can move to another state; keep dispatching until stable
	// (no transition chain is longer than a few states).
	for _ in 0..8 {
		let before = (g.flow.state, g.flow.state_complete);
		match g.flow.state {
			State::SelectingHand => {
				if g.areas.hand.cards.is_empty() && !g.areas.deck.is_empty() {
					g.flow.state = State::DrawToHand;
					g.flow.state_complete = false;
				} else {
					flow::update_selecting_hand(&mut g.flow, &g.areas, &mut g.events);
				}
			}
			State::Shop => shop::update_shop(&mut g.flow, &mut g.areas, &mut g.shop, &mut g.events),
			State::HandPlayed => flow::update_hand_played(&mut g.flow, &mut g.events),
			State::DrawToHand => {
				flow::update_draw_to_hand(&mut g.flow, &mut g.round, &mut g.tags, &mut g.events);
			}
			State::NewRound => flow::update_new_round(&mut g.flow, &mut g.events),
			State::BlindSelect => flow::update_blind_select(&mut g.flow, &mut g.events),
			State::RoundEval => flow::update_round_eval(&mut g.flow, &mut g.events),
			State::TarotPack
			| State::PlanetPack
			| State::SpectralPack
			| State::StandardPack
			| State::BuffoonPack => shop::update_pack(&mut g.flow, &mut g.events),
			State::GameOver => g.flow.state_complete = true,
			State::PlayTarot => {}
		}
		if (g.flow.state, g.flow.state_complete) == before {
			break;
		}
	}
}

/// Body of the `change_size` event.
fn apply_hand_size(state: State, hand: &mut Hand, events: &mut EventQueue, delta: f64) {
	hand.limit.change(delta);
	if delta > 0.0
		&& hand.limit.real > 1.0
		&& !hand.cards.is_empty()
		&& matches!(state, State::DrawToHand | State::SelectingHand)
	{
		for _ in 0..num::trunc(delta.abs()) {
			events.draw(Draw::next(Pile::Deck, Pile::Hand), 0.07);
			events.push(Event::SortHand);
		}
	}
}
