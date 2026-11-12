//! Emulation of `G.E_MANAGER` and of `Game:update`'s per-state handlers.
//!
//! The game's flows are chains of events on a FIFO queue; events created while
//! an event runs are appended at the tail, and the per-state `update_*`
//! handlers run after the event that changed the state. The queue is stepped
//! frame by frame (`dt = 1/60`, `G.TIMERS.TOTAL` rebased to 0 per action)
//! with the game's `trigger` / `delay` / `blocking` / `blockable` semantics,
//! so event order and frame timing match the real game. Purely visual events
//! are kept as timing-only placeholders (`Ev::Wait`).

use std::collections::VecDeque;

use crate::data::{Edition, Enhancement, HandType};
use crate::ev::{Ev, Pile};
use crate::game::{Area, Game};
use crate::{blind, cards, consumables, flow, num, packs, pools, shop, tags};

/// `G.STATE`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum St {
	SelectingHand,
	HandPlayed,
	DrawToHand,
	GameOver,
	Shop,
	PlayTarot,
	BlindSelect,
	RoundEval,
	TarotPack,
	PlanetPack,
	SpectralPack,
	StandardPack,
	BuffoonPack,
	NewRound,
}

impl St {
	#[must_use]
	pub fn is_pack(self) -> bool {
		matches!(
			self,
			St::TarotPack | St::PlanetPack | St::SpectralPack | St::StandardPack | St::BuffoonPack
		)
	}

	#[must_use]
	pub fn name(self) -> &'static str {
		match self {
			St::SelectingHand => "SELECTING_HAND",
			St::HandPlayed => "HAND_PLAYED",
			St::DrawToHand => "DRAW_TO_HAND",
			St::GameOver => "GAME_OVER",
			St::Shop => "SHOP",
			St::PlayTarot => "PLAY_TAROT",
			St::BlindSelect => "BLIND_SELECT",
			St::RoundEval => "ROUND_EVAL",
			St::TarotPack => "TAROT_PACK",
			St::PlanetPack => "PLANET_PACK",
			St::SpectralPack => "SPECTRAL_PACK",
			St::StandardPack => "STANDARD_PACK",
			St::BuffoonPack => "BUFFOON_PACK",
			St::NewRound => "NEW_ROUND",
		}
	}
}

/// Game speed (`G.SETTINGS.GAMESPEED`) the timing model assumes (the
/// game's default).
pub(crate) const GAMESPEED: f64 = 1.0;
/// Real seconds per frame (60 fps).
pub(crate) const DT: f64 = 1.0 / 60.0;

/// `G.TIMERS.TOTAL` after `n` frames at speed factor 1, accumulated frame by
/// frame exactly like the game does.
fn totals() -> &'static [f64] {
	static T: std::sync::OnceLock<Box<[f64]>> = std::sync::OnceLock::new();
	T.get_or_init(|| {
		let mut v = Vec::with_capacity(1 << 16);
		let mut t = 0.0f64;
		v.push(t);
		for _ in 1..(1 << 16) {
			t += DT;
			v.push(t);
		}
		v.into_boxed_slice()
	})
}

/// An event's `trigger` and `delay` (in `G.TIMERS.TOTAL` seconds).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Trig {
	/// `'immediate'`.
	Imm,
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

/// An entry of `G.E_MANAGER.queues.base`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Queued {
	pub ev: Ev,
	pub trig: Trig,
	pub blocking: bool,
	pub blockable: bool,
	/// `Event.time` (set when first handled; NaN before).
	pub time: f64,
	/// `Event.complete`.
	pub done: bool,
}

/// `level_up_hand(card, hand, instant, amount)` including its events.
pub(crate) fn level_up_hand_ev(g: &mut Game, h: HandType, amount: f64, instant: bool) {
	cards::level_up_hand(g, h, amount);
	if !instant {
		g.events.push_after(0.2, Ev::Wait);
		g.events.hand_text(0.0);
		g.events.push_after(0.9, Ev::Wait);
		g.events.hand_text(0.0);
		g.events.push_after(0.9, Ev::Wait);
		g.events.hand_text(0.0);
		g.events.delay(1.3);
	}
	g.events.push(Ev::Wait);
}

/// Glass cards shatter, everything else dissolves.
pub(crate) fn destroy_playing(g: &mut Game, id: u32) {
	let glass = g
		.playing_cards()
		.any(|c| c.id == id && c.enhancement == Enhancement::Glass);
	if glass {
		g.events.shatter(id);
	} else {
		g.events.start_dissolve(id);
	}
}

/// One frame of `Game:update` after the timers advanced:
/// `G.E_MANAGER:update`, then the state handlers. Returns whether
/// anything happened.
#[expect(
	clippy::too_many_lines,
	reason = "one loop, kept in one piece to follow `EventManager:update`"
)]
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
		let time = if q.time.is_nan() {
			changed = true;
			let t = if matches!(q.trig, Trig::Frames(_)) {
				f64::from(g.events.frames)
			} else {
				g.events.total
			};
			g.events.queue[i].time = t;
			t
		} else {
			q.time
		};
		let (completed, time_done) = match q.trig {
			Trig::Imm => {
				changed = true;
				(run_event(g, q.ev), true)
			}
			Trig::After(d) => {
				if time + d <= g.events.total {
					changed = true;
					(run_event(g, q.ev), true)
				} else {
					(false, false)
				}
			}
			Trig::Before(d) => {
				let ran = if q.done {
					true
				} else {
					changed = true;
					run_event(g, q.ev)
				};
				(ran, time + d <= g.events.total)
			}
			Trig::Ease(d) => {
				if time + d >= g.events.total {
					(false, false)
				} else {
					changed = true;
					(run_event(g, q.ev), true)
				}
			}
			Trig::Until { start, delay: d } => {
				if g.events.total - start > d {
					changed = true;
					(run_event(g, q.ev), true)
				} else {
					(false, true)
				}
			}
			Trig::Frames(n) => {
				changed = true;
				if f64::from(g.events.frames) - time >= f64::from(n) {
					(run_event(g, q.ev), true)
				} else {
					(false, true)
				}
			}
		};
		if q.blocking {
			blocked = true;
		}
		if g.pause_rest {
			// `G.SETTINGS.paused`: the rest of this pass is skipped.
			g.pause_rest = false;
			if completed && time_done {
				g.events.queue.remove(i);
				if !q.blockable {
					g.events.unblockable -= 1;
				}
			}
			break;
		}
		if completed && time_done {
			#[cfg(feature = "trace-keys")]
			if let Some(log) = g.rng.trace.evlog.as_mut() {
				log.push(format!(
					"{} {:?} {:?}{}{}",
					g.events.frames,
					q.ev,
					q.trig,
					if q.blocking { "" } else { " nb" },
					if q.blockable { "" } else { " nbl" }
				));
			}
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
	let before = (g.state, g.state_complete, g.events.queue.len());
	tick_state(g);
	changed || before != (g.state, g.state_complete, g.events.queue.len())
}

/// `G.TIMERS.TOTAL` advance (with the `G.ACC` speed-up of long hands).
#[inline]
#[expect(
	clippy::float_cmp,
	reason = "bit-exact timer arithmetic: the speed factor is exactly 1 or it is not"
)]
fn advance_time(g: &mut Game) {
	g.events.frames += 1;
	#[cfg(feature = "trace-keys")]
	{
		g.rng.trace.frame = g.events.frames;
	}
	if g.state != g.events.acc_state {
		g.events.acc = 0.0;
	}
	g.events.acc_state = g.state;
	if matches!(g.state, St::HandPlayed | St::NewRound) {
		g.events.acc = (g.events.acc + DT * 0.2 * GAMESPEED).min(16.0);
	} else {
		g.events.acc = 0.0;
	}
	let sf = GAMESPEED + (g.events.acc.abs() - 2.0).max(0.0);
	if sf != 1.0 {
		g.events.on_table = false;
	}
	g.events.total += DT * sf;
}

/// Advances idle frames until `TOTAL >= due`. While every frame of the
/// action advanced `TOTAL` by exactly `DT` (no `G.ACC` speed-up), `TOTAL`
/// after `n` frames is `totals()[n]`, so the skip is a table lookup with
/// bit-identical results.
#[expect(
	clippy::float_cmp,
	reason = "bit-exact timer arithmetic: the table applies only on an exact match"
)]
fn skip_until(g: &mut Game, due: f64) {
	if g.events.on_table && !matches!(g.state, St::HandPlayed | St::NewRound) {
		let t = totals();
		let cur = g.events.frames as usize;
		if cur < t.len() && g.events.total == t[cur] {
			// First frame n > cur with t[n] >= due.
			let mut n = num::trunc(due / DT).max(cur + 1);
			while n > cur + 1 && t.get(n - 1).is_some_and(|&v| v >= due) {
				n -= 1;
			}
			while n < t.len() && t[n] < due {
				n += 1;
			}
			if n < t.len() {
				g.events.frames = num::index32(n);
				g.events.total = t[n];
				g.events.acc = 0.0;
				g.events.acc_state = g.state;
				#[cfg(feature = "trace-keys")]
				{
					g.rng.trace.frame = g.events.frames;
				}
				return;
			}
		}
	}
	while g.events.total < due {
		advance_time(g);
	}
}

/// Every action starts at `G.TIMERS.TOTAL = 0` (the oracle rebases its
/// timers the same way), so timer arithmetic is reproducible.
pub(crate) fn begin_action(g: &mut Game) {
	// A cloned game's queue has no spare room: make it once, not by doubling.
	g.events.queue.reserve(QUEUE_ROOM);
	g.events.total = 0.0;
	g.events.frames = 0;
	g.events.on_table = true;
	#[cfg(feature = "trace-keys")]
	{
		let t = &mut g.rng.trace;
		t.frame = 0;
		t.entries.clear();
		if let Some(l) = t.evlog.as_mut() {
			l.clear();
		}
	}
}

/// Runs frames until `evaluate_play` has run (the rest of the chain is
/// abandoned: the game is left mid-animation and must be discarded).
pub(crate) fn run_until_scored(g: &mut Game) {
	g.scored = false;
	let mut guard = 0u32;
	while !g.scored {
		advance_time(g);
		if !pass(g) && !g.scored {
			if g.events.queue.is_empty() {
				break;
			}
			let due = g.events.next_due();
			if !due.is_finite() {
				break;
			}
			skip_until(g, due);
			pass(g);
		}
		guard += 1;
		if guard > 1_000_000 {
			break;
		}
	}
}

/// Runs frames until the game waits for input.
pub(crate) fn run_queue(g: &mut Game) {
	let mut guard = 0u32;
	loop {
		advance_time(g);
		if !pass(g) {
			if g.events.queue.is_empty() {
				break;
			}
			// Nothing can happen before a timer expires: skip idle
			// frames (their passes and state handlers are no-ops).
			let due = g.events.next_due();
			assert!(due.is_finite(), "event queue stalled");
			skip_until(g, due);
			pass(g);
		}
		guard += 1;
		assert!(guard < 100_000_000, "event loop did not terminate");
	}
	flow::sync_phase(g);
}

/// Runs an event's `func`; returns its result (false = retry next frame).
#[expect(clippy::too_many_lines, reason = "one arm per event")]
pub(crate) fn run_event(g: &mut Game, e: Ev) -> bool {
	match e {
		Ev::EaseDollars(d) => cards::ease_dollars(g, d),
		Ev::EaseDiscard(d) => cards::ease_discard(g, d),
		Ev::EaseHands(d) => cards::ease_hands_played(g, d),
		Ev::EaseRound => g.round += 1,
		Ev::EaseAnte(d) => g.ante += d,
		Ev::DollarBufferReset => g.dollar_buffer = 0.0,
		Ev::HandSize(d) => apply_hand_size(g, d),
		Ev::RefreshDebuffs => flow::refresh_playing_debuffs(g),
		Ev::SetCostAll => cards::set_cost_all(g),
		Ev::Create {
			set,
			area,
			forced,
			append,
			rarity,
			legendary,
		} => {
			let mut c = pools::create_card(g, set, area, legendary, rarity, false, forced, append);
			cards::add_to_deck(g, &mut c, false);
			if area == Area::Jokers {
				g.jokers.push(c);
				g.joker_buffer = 0.0;
			} else {
				g.consumables.push(c);
				g.consumeable_buffer = 0.0;
			}
		}
		Ev::Wait => {}
		Ev::PushHandTextImm => g.events.hand_text_imm(0.0),
		Ev::Materialize => g.events.materialize(1.0),
		Ev::SetChips(c) => g.chips = c,
		Ev::WinPoll => {
			if g.state != St::RoundEval {
				return false;
			}
			g.won = true;
			g.events.push(Ev::WinPause);
			// round_text = 'Endless Round ' once unpaused
			g.events.push(Ev::Wait);
		}
		Ev::WinPause => g.pause_rest = true,
		Ev::EaseChips0 => g
			.events
			.push_ev(Ev::SetChips(0.0), Trig::Ease(0.3), true, false),
		Ev::RerollOuter => g.events.push(Ev::RerollShopJokers),
		Ev::StatusLater(d) => g.events.status_text(d),
		Ev::DollarsStatus { d, delay } => {
			g.events.push(Ev::EaseDollars(d));
			g.events.status_text(delay);
		}
		Ev::PackRemove => {
			let cards = std::mem::take(&mut g.pack_cards);
			for c in cards {
				cards::drop_unowned(g, c);
			}
		}
		Ev::EatenOuter { id } => {
			g.events
				.push_ev(Ev::RemoveJokerLater { id }, Trig::After(0.3), true, false);
		}
		Ev::RemoveCard { id } => flow::remove_card_by_id(g, id),
		Ev::DissolvePlaying { id } => destroy_playing(g, id),
		Ev::BlindReset => blind::set_blind(g, None),
		Ev::RemoveJokerLater { id } => {
			if let Some(i) = g.jokers.iter().position(|c| c.id == id) {
				let c = g.jokers.remove(i);
				cards::remove_owned(g, c);
			}
		}
		Ev::DissolveJoker { id, fac } => g.events.start_dissolve_f(id, fac),
		Ev::DrawCard {
			from,
			to,
			card,
			sort,
			discarded_only,
		} => flow::draw_card(g, from, to, card, sort, discarded_only),
		Ev::AddTag { kind, orbital } => tags::add_tag(g, kind, orbital),
		Ev::Consumable(c) => consumables::run_consumable_event(g, c),
		Ev::LastTarotPlanetOuter { kind } => g.events.push(Ev::LastTarotPlanetInner { kind }),
		Ev::LastTarotPlanetInner { kind } => g.last_tarot_planet = Some(kind),
		Ev::Unvampire { id } => {
			if let Some(c) = g.playing_cards_mut().find(|c| c.id == id) {
				c.vampired = false;
			}
		}
		Ev::TagFunc { .. } | Ev::TagRemove { .. } => tags::run_tag_event(g, e),
		Ev::BlindSelectInit
		| Ev::BlindSelectUi
		| Ev::SelectBlind { .. }
		| Ev::NewRound
		| Ev::NewRoundBody
		| Ev::NewRoundDraw
		| Ev::BlindSetHud
		| Ev::DrawToHand
		| Ev::EnterSelecting
		| Ev::PlayStart
		| Ev::PlayChain
		| Ev::EvaluatePlay
		| Ev::AfterPlay
		| Ev::StateCompleteFalse
		| Ev::HandPlayedCheck
		| Ev::HookDiscard
		| Ev::ToothOuter
		| Ev::EndRound
		| Ev::EndRoundFinal
		| Ev::RoundEvalInit
		| Ev::RoundEvalRun
		| Ev::CashOutFinish
		| Ev::BlindDisableCheck
		| Ev::AcornOuter
		| Ev::AcornShuffle
		| Ev::SkipBlindRun
		| Ev::RerollBossRun
		| Ev::ChicotOuter
		| Ev::ChicotDisable
		| Ev::Burglar
		| Ev::RiffRaff { .. }
		| Ev::CartomancerOuter
		| Ev::DaggerSlice { .. }
		| Ev::MarbleStone
		| Ev::CertificateCard
		| Ev::GlassJokerOuter { .. }
		| Ev::GlassJokerInner { .. }
		| Ev::PerkeoCopy => flow::run_flow_event(g, e),
		Ev::ShopOuter
		| Ev::ShopPopulate { .. }
		| Ev::StoreModify { .. }
		| Ev::RerollShopRun
		| Ev::RerollShopJokers
		| Ev::BuyRun { .. }
		| Ev::SellRun { .. }
		| Ev::VerdantDisable
		| Ev::ToggleShopRun
		| Ev::OpenPack { .. }
		| Ev::PackCreate
		| Ev::PackStateUi
		| Ev::PackDrawHand
		| Ev::UseFinishOuter { .. }
		| Ev::UseFinishInner { .. }
		| Ev::EndConsumeable { .. }
		| Ev::EndConsumeableInner => shop::run_shop_event(g, e),
	}
	true
}

/// `Game:update`'s state dispatch (after each event-queue pass).
pub(crate) fn tick_state(g: &mut Game) {
	// A handler can move to another state; keep dispatching until stable
	// (no transition chain is longer than a few states).
	for _ in 0..8 {
		let before = (g.state, g.state_complete);
		match g.state {
			St::SelectingHand => {
				if g.hand.is_empty() && !g.deck.is_empty() {
					g.state = St::DrawToHand;
					g.state_complete = false;
				} else {
					flow::update_selecting_hand(g);
				}
			}
			St::Shop => shop::update_shop(g),
			St::HandPlayed => flow::update_hand_played(g),
			St::DrawToHand => flow::update_draw_to_hand(g),
			St::NewRound => flow::update_new_round(g),
			St::BlindSelect => flow::update_blind_select(g),
			St::RoundEval => flow::update_round_eval(g),
			St::TarotPack
			| St::PlanetPack
			| St::SpectralPack
			| St::StandardPack
			| St::BuffoonPack => packs::update_pack(g),
			St::GameOver => g.state_complete = true,
			St::PlayTarot => {}
		}
		if (g.state, g.state_complete) == before {
			break;
		}
	}
}

/// Body of the `change_size` event.
pub(crate) fn apply_hand_size(g: &mut Game, delta: f64) {
	g.hand_limit.change(delta);
	if delta > 0.0
		&& g.hand_limit.real > 1.0
		&& !g.hand.is_empty()
		&& matches!(g.state, St::DrawToHand | St::SelectingHand)
	{
		for _ in 0..num::trunc(delta.abs()) {
			g.events
				.draw_ev(Pile::Deck, Pile::Hand, None, false, false, 0.07);
			g.events
				.push(Ev::Consumable(crate::consumables::CEv::SortHand));
		}
	}
}

/// Room for the events one action queues at a time.
const QUEUE_ROOM: usize = 64;

/// `G.E_MANAGER`'s base queue and the timers it runs on.
#[derive(Clone, Debug)]
pub(crate) struct EventQueue {
	queue: VecDeque<Queued>,
	/// Number of non-blockable events in `queue`.
	unblockable: u32,
	/// `G.TIMERS.TOTAL`.
	total: f64,
	/// Frames since the current action started.
	frames: u32,
	/// `total == totals()[frames]` (no speed-up so far in this action).
	on_table: bool,
	/// `G.ACC` / `G.ACC_state`.
	acc: f64,
	acc_state: St,
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
			acc_state: St::BlindSelect,
		}
	}

	/// Drops every pending event.
	pub(crate) fn clear(&mut self) {
		self.queue.clear();
		self.unblockable = 0;
	}

	/// Takes every pending event out, in queue order.
	pub(crate) fn take_pending(&mut self) -> Vec<Ev> {
		self.unblockable = 0;
		self.queue.drain(..).map(|q| q.ev).collect()
	}

	/// `G.E_MANAGER:add_event(Event({func = ...}))` (immediate, blocking).
	#[inline]
	pub(crate) fn push(&mut self, e: Ev) {
		self.push_ev(e, Trig::Imm, true, true);
	}

	#[inline]
	pub(crate) fn push_ev(&mut self, ev: Ev, trig: Trig, blocking: bool, blockable: bool) {
		if !blockable {
			self.unblockable += 1;
		}
		self.queue.push_back(Queued {
			ev,
			trig,
			blocking,
			blockable,
			time: f64::NAN,
			done: false,
		});
	}

	/// `draw_card(from, to, ..., delay)`: a `'before'` event.
	#[inline]
	pub(crate) fn draw_ev(
		&mut self, from: Pile, to: Pile, card: Option<u32>, sort: bool, discarded_only: bool,
		delay: f64,
	) {
		self.push_before(delay, Ev::DrawCard {
			from,
			to,
			card,
			sort,
			discarded_only,
		});
	}

	#[inline]
	pub(crate) fn push_after(&mut self, d: f64, e: Ev) {
		self.push_ev(e, Trig::After(d), true, true);
	}

	#[inline]
	pub(crate) fn push_before(&mut self, d: f64, e: Ev) {
		self.push_ev(e, Trig::Before(d), true, true);
	}

	/// `delay(t)`.
	#[inline]
	pub(crate) fn delay(&mut self, d: f64) {
		self.push_ev(Ev::Wait, Trig::After(d), true, true);
	}

	/// A blocking event with no game effect (`trigger`, `delay`).
	#[inline]
	pub(crate) fn wait(&mut self, t: Trig) {
		self.push_ev(Ev::Wait, t, true, true);
	}

	/// A non-blockable (but blocking) event with no game effect: holds back
	/// every event queued after it.
	#[inline]
	pub(crate) fn barrier(&mut self, t: Trig) {
		self.push_ev(Ev::Wait, t, true, false);
	}

	/// `update_hand_text({delay = d}, ...)`.
	#[inline]
	pub(crate) fn hand_text(&mut self, d: f64) {
		self.wait(Trig::Before(d));
	}

	/// `update_hand_text({immediate = true, delay = d}, ...)` (non-blockable).
	#[inline]
	pub(crate) fn hand_text_imm(&mut self, d: f64) {
		self.barrier(Trig::Before(d));
	}

	/// `highlight_card(card, ...)`.
	#[inline]
	pub(crate) fn highlight_ev(&mut self) {
		self.wait(Trig::Before(0.1));
	}

	/// The events `Card:set_edition(edition, immediate)` queues (not silent).
	pub(crate) fn edition_fx(&mut self, e: Edition, immediate: bool) {
		if e == Edition::None {
			return;
		}
		if immediate {
			self.push_ev(Ev::Wait, Trig::After(0.0), true, false);
		} else {
			self.push_after(0.2, Ev::Wait);
		}
		self.push_after(0.1, Ev::Wait);
	}

	/// `card_eval_status_text(card, ...)` with an explicit base `delay`: a
	/// `'before'` event of `delay*1.25`.
	#[inline]
	pub(crate) fn status_text(&mut self, delay: f64) {
		self.wait(Trig::Before(delay * 1.25));
	}

	/// `card_eval_status_text(card, 'extra', ...)` without `extra.delay`.
	#[inline]
	pub(crate) fn status_extra(&mut self) {
		self.status_text(0.75);
	}

	/// `Card:start_dissolve(colours, silent, dissolve_time_fac)`.
	pub(crate) fn start_dissolve_f(&mut self, id: u32, fac: f64) {
		let dt = 0.7 * fac;
		self.push_ev(Ev::RemoveCard { id }, Trig::After(1.05 * dt), true, false);
		self.barrier(Trig::After(1.051 * dt));
	}

	#[inline]
	pub(crate) fn start_dissolve(&mut self, id: u32) {
		self.start_dissolve_f(id, 1.0);
	}

	/// `Card:shatter()`.
	pub(crate) fn shatter(&mut self, id: u32) {
		let dt = 0.7;
		self.push_ev(Ev::RemoveCard { id }, Trig::After(0.55 * dt), true, false);
		self.barrier(Trig::After(0.51 * dt));
	}

	/// `Card:explode()`.
	pub(crate) fn explode(&mut self, id: u32) {
		let et = 1.3 * GAMESPEED.sqrt();
		self.barrier(Trig::Until {
			start: self.total,
			delay: 1.5 * et,
		});
		self.push_ev(Ev::RemoveCard { id }, Trig::After(1.5 * et), true, false);
	}

	/// `Card:start_materialize(colours, silent, timefac)`: only its timing.
	pub(crate) fn materialize(&mut self, timefac: f64) {
		let dt = 0.6 * timefac;
		self.barrier(Trig::After(0.5 * dt));
		self.barrier(Trig::Ease(dt));
		self.barrier(Trig::After(1.05 * dt));
	}

	/// Earliest `TOTAL` at which a waiting event can progress.
	fn next_due(&self) -> f64 {
		let mut due = f64::INFINITY;
		for q in &self.queue {
			if q.time.is_nan() {
				continue;
			}
			let t = match q.trig {
				Trig::Imm => continue,
				Trig::After(d) | Trig::Before(d) | Trig::Ease(d) => q.time + d,
				Trig::Until { start, delay } => start + delay,
				Trig::Frames(_) => self.total,
			};
			due = due.min(t);
		}
		due
	}
}
