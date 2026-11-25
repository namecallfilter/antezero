//! Per-state update handlers (`Game:update_*`), phase synchronization, and
//! [`FlowEvent`] execution.

use super::actions::{FlowEvent, hook_discard, new_round};
use super::cards::{draw_from_deck_to_hand, shuffle_area};
use crate::areas::Pile;
use crate::card::PlayingCard;
use crate::data::{BlindKind, HandType};
use crate::event::{Blocks, Draw, Event, EventQueue, Trigger};
use crate::game::{
	Areas, BlindSelect, BlindSlot, BlindStatus, Flow, Game, HandLevels, PerBlind, Phase, State,
};
use crate::joker::LifecycleCtx;
use crate::rng::{Key, PseudoRandom};
use crate::round::Round;
use crate::run::{OrbitalChoices, Run, Tags};
use crate::shop::Shop;
use crate::{joker, pools, round, run};

/// `Game:update_selecting_hand`. With hand, deck and play all empty the
/// Lua calls `end_round()` on every frame until its first event resolves,
/// queueing several end-of-round passes; the simulator queues one. The
/// extra passes only matter after the run is lost (or saved by Mr. Bones).
pub(crate) fn update_selecting_hand(flow: &mut Flow, areas: &Areas, events: &mut EventQueue) {
	let out = areas.hand.cards.is_empty() && areas.deck.is_empty();
	if out && areas.play.is_empty() && !flow.end_round_pending {
		flow.end_round_pending = true;
		events.push_after(0.2, Event::Flow(FlowEvent::EndRound));
	}
	if !flow.state_complete {
		flow.state_complete = true;
		if out && !flow.end_round_pending {
			flow.end_round_pending = true;
			events.push_after(0.2, Event::Flow(FlowEvent::EndRound));
		}
	}
}

pub(crate) fn update_hand_played(flow: &mut Flow, events: &mut EventQueue) {
	if !flow.state_complete {
		flow.state_complete = true;
		events.push(Event::Flow(FlowEvent::HandPlayedCheck));
	}
}

pub(crate) fn update_draw_to_hand(
	flow: &mut Flow, round: &mut Round, tags: &mut Tags, events: &mut EventQueue,
) {
	if !flow.state_complete {
		flow.state_complete = true;
		run::apply_round_start_bonus(round, tags, events);
		events.push(Event::Flow(FlowEvent::DrawToHand));
	}
}

pub(crate) fn update_new_round(flow: &mut Flow, events: &mut EventQueue) {
	if !flow.state_complete {
		flow.state_complete = true;
		events.push_after(0.2, Event::Flow(FlowEvent::EndRound));
	}
}

pub(crate) fn update_blind_select(flow: &mut Flow, events: &mut EventQueue) {
	if !flow.state_complete {
		flow.state_complete = true;
		// save_run
		events.push(Event::Wait);
		events.push(Event::Flow(FlowEvent::BlindSelectInit));
	}
}

pub(crate) fn update_round_eval(flow: &mut Flow, events: &mut EventQueue) {
	if !flow.state_complete {
		flow.state_complete = true;
		events.push(Event::Flow(FlowEvent::RoundEvalInit));
	}
}

/// Derives the agent-facing phase from `G.STATE` once the queue is empty.
///
/// # Panics
///
/// Panics if the state is one the game only passes through (a simulator
/// bug).
pub(crate) fn sync_phase(flow: &mut Flow) {
	flow.phase = match flow.state {
		State::SelectingHand => Phase::Round,
		State::Shop => Phase::Shop,
		State::BlindSelect => Phase::BlindSelect,
		State::RoundEval => Phase::RoundEval,
		State::GameOver => Phase::GameOver,
		State::TarotPack
		| State::PlanetPack
		| State::SpectralPack
		| State::StandardPack
		| State::BuffoonPack => Phase::Pack,
		transient
		@ (State::HandPlayed | State::DrawToHand | State::NewRound | State::PlayTarot) => {
			panic!("queue drained in transient state {transient:?}")
		}
	};
}

pub(crate) fn run_event(g: &mut Game, e: FlowEvent) {
	use FlowEvent as F;
	match e {
		F::BlindSelectInit => g.events.push(Event::Flow(F::BlindSelectUi)),
		F::BlindSelectUi => blind_select_ui(g),
		F::SelectBlind { blind } => select_blind(&mut g.blind_select, &mut g.events, blind),
		// new_round()
		F::NewRound => {
			g.events.delay(0.4);
			g.events.push(Event::Flow(F::NewRoundBody));
		}
		F::NewRoundBody => new_round(g),
		F::BlindSetHud => g.events.barrier(Trigger::After(0.15)),
		F::NewRoundDraw => {
			g.flow.state = State::DrawToHand;
			let key = g.run.key_ante("nr");
			shuffle_area(&mut g.areas.deck, &mut g.rng, &key);
			g.flow.state_complete = false;
		}
		F::DrawToHand => draw_to_hand(g),
		F::EnterSelecting => {
			g.flow.state = State::SelectingHand;
			g.flow.state_complete = false;
			round::drawn_to_hand(g);
		}
		F::PlayStart => {
			g.flow.state = State::HandPlayed;
			g.flow.state_complete = true;
		}
		F::PlayChain => {
			g.events.push(Event::Flow(F::EvaluatePlay));
			g.events.push_after(0.1, Event::Flow(F::AfterPlay));
			g.events.push(Event::Flow(F::StateCompleteFalse));
		}
		F::EvaluatePlay => round::evaluate_play(g),
		F::AfterPlay => after_play(&g.areas.play, &mut g.run, &mut g.round, &mut g.events),
		F::StateCompleteFalse => g.flow.state_complete = false,
		F::HandPlayedCheck => {
			g.flow.state = if g.round.beats(g.blind.chips) || g.round.hands_left < 1.0 {
				State::NewRound
			} else {
				State::DrawToHand
			};
			g.flow.state_complete = false;
		}
		F::HookDiscard => hook_discard(g),
		F::ToothOuter => {
			for _ in 0..g.areas.play.len() {
				g.events.push(Event::Wait);
				g.events.push(Event::EaseDollars(-1.0));
				g.events.delay(0.23);
			}
		}
		F::EndRound => round::end_round(g),
		F::EndRoundFinal => round::end_round_final(g),
		F::DiscardToDeck => {
			for _ in 0..g.areas.discard.len() {
				g.events.draw(Draw::next(Pile::Discard, Pile::Deck), 0.005);
			}
		}
		F::BlueSealPlanet => round::blue_seal_planet(g),
		F::RoundEvalInit => {
			g.round.facing_blind = false;
			// Polls until the round-eval UI has slid in.
			g.events.push_event(
				Event::Flow(F::RoundEvalRun),
				Trigger::Frames(21),
				Blocks::InOrder,
			);
		}
		F::RoundEvalRun => {
			g.events.delay(0.1);
			round::evaluate_round(g);
		}
		F::BlindDefeat => round::defeat(&mut g.areas.jokers, &g.blind, &mut g.events),
		F::CashOutFinish => enter_shop(&mut g.flow, &mut g.round, &mut g.shop),
		F::BlindDisableCheck => {
			if g.blind.is_boss() && g.round.beats(g.blind.chips) {
				g.flow.state = State::NewRound;
				g.flow.state_complete = false;
			}
		}
		F::AcornOuter => {
			g.events.push(Event::Flow(F::AcornShuffle));
			g.events.delay(0.15);
			g.events.push(Event::Flow(F::AcornShuffle));
			g.events.delay(0.15);
			g.events.push(Event::Flow(F::AcornShuffle));
			g.events.delay(0.5);
		}
		F::AcornShuffle => shuffle_area(&mut g.areas.jokers, &mut g.rng, &Key::from("aajk")),
		F::SkipBlindRun => skip_blind_run(g),
		F::RerollBossRun => reroll_boss_run(g),
	}
}

/// The on-deck blind becoming the current one.
fn select_blind(blind_select: &mut BlindSelect, events: &mut EventQueue, blind: BlindKind) {
	events.push(Event::EaseRound);
	blind_select.selected = Some(blind);
	if let Some(s) = blind_select.on_deck {
		blind_select.states[s] = BlindStatus::Current;
	}
	events.delay(0.2);
}

/// The end of cashing out: hands and discards reset, on to the shop.
fn enter_shop(flow: &mut Flow, round: &mut Round, shop: &mut Shop) {
	round.reset_hands_and_discards();
	flow.state = State::Shop;
	shop.free = false;
	shop.d6ed = false;
	flow.state_complete = false;
}

/// A new boss for the ante, with its Orbital hand and the waiting tags.
fn reroll_boss_run(g: &mut Game) {
	g.blind_select.boss = pools::get_new_boss(
		g.setup.win_ante,
		g.run.ante,
		&mut g.blind_select,
		&mut g.rng,
	);
	if let Some(s) = g.blind_select.on_deck {
		g.blind_select.states[s] = BlindStatus::Select;
	}
	ensure_orbital(
		g.run.ante,
		&g.hands,
		&mut g.tags,
		&mut g.rng,
		BlindSlot::Boss,
	);
	run::apply_new_blind_choice(&mut g.tags, &mut g.events);
}

/// The hand drawn at the start of a round or after a play or discard.
fn draw_to_hand(g: &mut Game) {
	if draw_from_deck_to_hand(
		&mut g.flow,
		&g.areas,
		&g.round,
		&g.blind,
		&mut g.events,
		None,
	) {
		return;
	}
	if g.round.hands_played == 0.0 && g.round.discards_used == 0.0 && g.round.facing_blind {
		joker::calc_jokers(g, LifecycleCtx::FirstHandDrawn);
	}
	g.events.push(Event::Flow(FlowEvent::EnterSelecting));
}

/// The played cards going to the discard pile, and the hand counters.
fn after_play(play: &[PlayingCard], run: &mut Run, round: &mut Round, events: &mut EventQueue) {
	for c in play {
		if !c.shattered && !c.destroyed {
			events.draw(Draw::card(Pile::Play, Pile::Discard, c.core.id), 0.1);
		}
	}
	run.hands_played += 1.0;
	round.hands_played += 1.0;
}

/// The jokers and tags reacting to a skipped blind.
fn skip_blind_run(g: &mut Game) {
	g.events.delay(0.3);
	joker::calc_jokers(g, LifecycleCtx::SkipBlind);
	run::apply_immediate(&g.run, &mut g.hands, &mut g.tags, &mut g.events);
	run::apply_new_blind_choice(&mut g.tags, &mut g.events);
	// The blind-choice UI notices the new on-deck blind this frame
	// (blind_choice_handler) and queues a focus event.
	g.events.push(Event::Wait);
}

fn orbital_slot(ante: i32, tags: &mut Tags) -> &mut PerBlind<Option<HandType>> {
	if let Some(i) = tags.orbital_choices.iter().position(|o| o.ante == ante) {
		&mut tags.orbital_choices[i].hands
	} else {
		tags.orbital_choices.push(OrbitalChoices {
			ante,
			hands: PerBlind::default(),
		});
		let n = tags.orbital_choices.len() - 1;
		&mut tags.orbital_choices[n].hands
	}
}

fn ensure_orbital(
	ante: i32, hands: &HandLevels, tags: &mut Tags, rng: &mut PseudoRandom, slot: BlindSlot,
) {
	let choice = &mut orbital_slot(ante, tags)[slot];
	if choice.is_none() {
		*choice = Some(hands.random_visible(rng, "orbital", None));
	}
}

/// `create_UIBox_blind_select` + the tag passes of `update_blind_select`.
fn blind_select_ui(g: &mut Game) {
	let s = &g.blind_select.states;
	let on_deck = if !matches!(
		s[BlindSlot::Small],
		BlindStatus::Defeated | BlindStatus::Skipped
	) {
		BlindSlot::Small
	} else if !matches!(
		s[BlindSlot::Big],
		BlindStatus::Defeated | BlindStatus::Skipped
	) {
		BlindSlot::Big
	} else {
		BlindSlot::Boss
	};
	g.blind_select.on_deck = Some(on_deck);
	g.blind_select.states[on_deck] = BlindStatus::Select;
	for slot in BlindSlot::ALL {
		ensure_orbital(g.run.ante, &g.hands, &mut g.tags, &mut g.rng, slot);
	}
	run::apply_immediate(&g.run, &mut g.hands, &mut g.tags, &mut g.events);
	run::apply_new_blind_choice(&mut g.tags, &mut g.events);
}
