//! Round and blind-select action handlers (`select_blind`, `skip_blind`,
//! `reroll_boss`, `new_round`, `play`, `discard`).

use crate::areas::Pile;
use crate::card::{CardMask, Forced, Kind, PlayingCard};
use crate::data::{BlindKind, HandType, Joker, Seal, TagKind};
use crate::event::{Draw, Event, EventQueue, Trigger};
use crate::game::{
	Areas, Blind, BlindSelect, BlindSlot, BlindStatus, CardSet, Flow, Game, HandLevels, State,
};
use crate::joker::{Discarded, LifecycleCtx};
use crate::rng::Key;
use crate::round::Round;
use crate::run::{Run, Tags};
use crate::shop::{BOOSTER_SLOTS, PackSlot};
use crate::{card, hand_eval, joker, num, round, run, shop};

/// The events of the round flow: blind select, the round's start, plays
/// and the end of the round.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum FlowEvent {
	BlindSelectInit,
	BlindSelectUi,
	SelectBlind {
		blind: BlindKind,
	},
	NewRound,
	NewRoundBody,
	NewRoundDraw,
	/// `Blind:set_blind`'s HUD event (non-blockable, queues a second one).
	BlindSetHud,
	DrawToHand,
	EnterSelecting,
	PlayStart,
	PlayChain,
	EvaluatePlay,
	AfterPlay,
	StateCompleteFalse,
	HandPlayedCheck,
	HookDiscard,
	ToothOuter,
	EndRound,
	EndRoundFinal,
	/// The discard pile going back to the deck after a round.
	DiscardToDeck,
	/// A Blue Seal's planet for the last hand played.
	BlueSealPlanet,
	RoundEvalInit,
	RoundEvalRun,
	/// `Blind:defeat()` at the start of the round evaluation.
	BlindDefeat,
	CashOutFinish,
	BlindDisableCheck,
	AcornOuter,
	AcornShuffle,
	SkipBlindRun,
	RerollBossRun,
}

pub(crate) fn orbital_for(ante: i32, tags: &Tags, slot: BlindSlot) -> Option<HandType> {
	tags.orbital_choices
		.iter()
		.find(|o| o.ante == ante)
		.and_then(|o| o.hands[slot])
}

/// `G.FUNCS.select_blind` for the blind on deck.
///
/// # Panics
///
/// Panics if no blind is on deck (the blind-select phase always has one).
pub(crate) fn do_select_blind(
	blind_select: &BlindSelect, round: &mut Round, events: &mut EventQueue,
) {
	let b = blind_select
		.on_deck_kind()
		.expect("`blind_select_ui` puts a blind on deck before the phase is offered");
	round.facing_blind = true;
	events.wait(Trigger::Before(0.2));
	events.push(Event::Flow(FlowEvent::SelectBlind { blind: b }));
	events.push(Event::Flow(FlowEvent::NewRound));
}

/// `G.FUNCS.skip_blind` for the blind on deck.
///
/// # Panics
///
/// Panics if the boss or no blind is on deck, or the ante's skip tags have
/// not been rolled.
pub(crate) fn do_skip_blind(
	blind_select: &mut BlindSelect, run: &mut Run, tags: &mut Tags, events: &mut EventQueue,
) {
	let slot = blind_select
		.on_deck
		.filter(|&slot| slot != BlindSlot::Boss)
		.expect("only the small and the big blind can be skipped");
	let kind = blind_select
		.skip_tags
		.of(slot)
		.expect("the skip tags are rolled at the start of every ante");
	run.skips += 1.0;
	run.blinds_cleared += 1;
	let orbital = if kind == TagKind::Orbital {
		orbital_for(run.ante, tags, slot)
	} else {
		None
	};
	run::add_tag(tags, events, kind, orbital);
	let to = if slot == BlindSlot::Small {
		BlindSlot::Big
	} else {
		BlindSlot::Boss
	};
	blind_select.states[slot] = BlindStatus::Skipped;
	blind_select.states[to] = BlindStatus::Select;
	blind_select.on_deck = Some(to);
	events.push(Event::Flow(FlowEvent::SkipBlindRun));
}

/// What the player pays to reroll the boss (the Boss Tag rerolls for free).
pub(crate) const BOSS_REROLL_COST: f64 = 10.0;

/// `G.FUNCS.reroll_boss`, after the payment.
pub(crate) fn do_reroll_boss(blind_select: &mut BlindSelect, events: &mut EventQueue) {
	blind_select.boss_rerolled = true;
	events.push(Event::Wait);
	events.push_after(0.3, Event::Flow(FlowEvent::RerollBossRun));
}

/// The counters `new_round` resets: the round's hands and discards, and
/// each poker hand's plays this round.
pub(crate) fn reset_round_counters(round: &mut Round, hands: &mut HandLevels) {
	round.reset_hands_and_discards();
	round.hands_played = 0.0;
	round.discards_used = 0.0;
	for h in hands {
		h.played_this_round = 0.0;
	}
}

pub(super) fn new_round(g: &mut Game) {
	reset_round_counters(&mut g.round, &mut g.hands);
	g.shop.reroll_cost_increase = 0.0;
	g.shop.booster_slots = [PackSlot::Unrolled; BOOSTER_SLOTS];
	for c in g.areas.playing_cards_mut() {
		c.wheel_flipped = false;
	}
	g.shop.free_rerolls = num::float(g.areas.count_joker(Joker::Chaos));
	shop::calculate_reroll_cost(&mut g.shop);
	g.round.bonus_next_hands = 0.0;
	g.round.bonus_discards = 0.0;
	let blind = g
		.blind_select
		.selected
		.expect("new_round without a selected blind");
	g.blind_select.states[BlindSlot::of(blind)] = BlindStatus::Current;
	round::set_blind(g, Some(blind));
	let boss = blind.is_boss();
	joker::calc_jokers(g, LifecycleCtx::SettingBlind { boss });
	g.flow.end_round_pending = false;
	g.events.delay(0.4);
	g.events.push(Event::Flow(FlowEvent::NewRoundDraw));
}

/// `G.FUNCS.play_cards_from_highlighted` with the hand cards at the
/// positions `sel`; under Steamodded, which marks a played card for The
/// Pillar only once the hand is evaluated ([`Event::MarkPlayed`]).
///
/// # Panics
///
/// Panics if `sel` holds a position past the end of the hand.
pub(crate) fn do_play(
	flow: &mut Flow, areas: &mut Areas, blind: &mut Blind, events: &mut EventQueue, sel: CardMask,
	steamodded: bool,
) {
	if !areas.play.is_empty() {
		return;
	}
	blind.triggered = false;
	for c in areas.playing_cards_mut() {
		c.forced = Forced::No;
	}
	areas.play.reserve(sel.len());
	flow.hand_text_cleared = false;
	events.push(Event::Flow(FlowEvent::PlayStart));
	events.push(Event::EaseHands(-1.0));
	events.delay(0.4);
	for i in sel.indices() {
		let c = &mut areas.hand.cards[i];
		c.times_played += 1;
		if !steamodded {
			c.played_this_ante = true;
		}
		let id = c.core.id;
		events.draw(Draw::card(Pile::Hand, Pile::Play, id), 0.1);
	}
	if round::press_play(&areas.jokers, blind, events) {
		events.push(Event::Wait);
		events.delay(0.4);
	}
	events.push(Event::Flow(FlowEvent::PlayChain));
}

pub(super) fn hook_discard(g: &mut Game) {
	// pseudorandom_element picks from the remaining cards sorted by sort_id.
	let mut cards: Vec<usize> = (0..g.areas.hand.cards.len()).collect();
	cards.sort_by_key(|&i| g.areas.hand.cards[i].core.id);
	let mut chosen = [0; 2];
	let n = cards.len().min(chosen.len());
	for slot in &mut chosen[..n] {
		let k = Key::from("hook");
		let pick = g.rng.element(&k, cards.len());
		*slot = cards.remove(pick);
	}
	if n > 0 {
		do_discard(g, &mut chosen[..n], /* hook= */ true);
	}
}

/// `G.discard.config.card_limit`.
const DISCARD_LIMIT: usize = 500;

/// `G.FUNCS.discard_cards_from_highlighted(e, hook)` with the hand cards at
/// the positions `sel`.
///
/// # Panics
///
/// Panics if `sel` holds a position past the end of the hand.
pub(crate) fn do_discard(g: &mut Game, sel: &mut [usize], hook: bool) {
	for c in g.areas.playing_cards_mut() {
		c.forced = Forced::No;
	}
	sel.sort_unstable();
	let count = sel
		.len()
		.min(DISCARD_LIMIT.saturating_sub(g.areas.play.len()));
	if count == 0 {
		return;
	}
	let sel = &sel[..count];
	// update_hand_text({immediate = true, delay = 0}, {handname = ''})
	g.events.hand_text_immediate(0.0);
	g.flow.hand_text_cleared = true;
	let burnt_hand = if g
		.areas
		.jokers
		.iter()
		.any(|c| c.kind == Kind::Joker(Joker::Burnt))
	{
		let refs: Vec<&PlayingCard> = sel.iter().map(|&i| &g.areas.hand.cards[i]).collect();
		hand_eval::evaluate(&refs, g.areas.eval_flags())
			.top
			.map(|top| top.hand)
	} else {
		None
	};
	joker::calc_jokers(g, LifecycleCtx::PreDiscard { hook, burnt_hand });
	let par = g.areas.pareidolia();
	let full_faces = sel
		.iter()
		.filter(|&&i| g.areas.hand.cards[i].is_face(par))
		.count();
	let last = sel[count - 1];
	let mut destroyed: Vec<usize> = Vec::new();
	for &i in sel {
		let card = &g.areas.hand.cards[i];
		let purple = !card.core.debuff && card.seal == Some(Seal::Purple);
		let id = card.core.id;
		// Card:calculate_seal({discard = true})
		if purple && g.areas.consumable_room_after_queued() {
			joker::queue_consumable(&mut g.areas, &mut g.events, CardSet::Tarot, "8ba");
			g.events.status_extra();
		}
		let mut removed = false;
		for j in 0..g.areas.jokers.len() {
			if let Some(e) = joker::calc_joker(
				g,
				j,
				LifecycleCtx::Discard(Discarded {
					other: i,
					last,
					n_full: count,
					full_faces,
				}),
			) {
				if e.remove {
					removed = true;
				}
				joker::effect_status(&mut g.events, &e);
			}
		}
		if removed {
			destroyed.push(i);
			card::shatter_or_dissolve(&mut g.areas.hand.cards[i], &mut g.events);
		} else {
			g.events
				.draw(Draw::card(Pile::Hand, Pile::Discard, id), 0.1);
		}
	}
	if !destroyed.is_empty() {
		let faces = destroyed
			.iter()
			.filter(|&&i| g.areas.hand.cards[i].is_face(par))
			.count();
		let glass = destroyed
			.iter()
			.filter(|&&i| g.areas.hand.cards[i].shattered)
			.count();
		joker::calc_jokers(g, LifecycleCtx::RemovePlayingCards { faces, glass });
	}
	if !hook {
		g.events.push(Event::EaseDiscard(-1.0));
		g.round.discards_used += 1.0;
		g.flow.state = State::DrawToHand;
		g.events.push(Event::Flow(FlowEvent::StateCompleteFalse));
	}
}
