//! End of round, round evaluation and cash out (`state_events.lua`
//! `end_round`, `button_callbacks.lua` `evaluate_round` / `cash_out`).

use super::blind::{BlindSelect, BlindSlot, BlindStatus, PerBlind};
use super::scoring;
use super::state::{JokerTargets, Round};
use crate::areas::Pile;
use crate::card::{Ability, Card, Forced, Kind, Sticker};
use crate::data::{Back, HandType, Joker, Planet, PlanetSet, Rank, Seal, TagKind};
use crate::event::{Blocks, Draw, Event, EventQueue, Trigger};
use crate::flow::FlowEvent;
use crate::game::{Areas, Game, HandLevels, State};
use crate::joker::LifecycleCtx;
use crate::pools::{NewCard, OwnedArea, VoucherRoll};
use crate::rng::{PseudoRandom, trace};
use crate::run::TagEvent;
use crate::{card, flow, joker, num, pools, run, shop};

pub(crate) fn end_round(g: &mut Game) {
	let mut game_over = !g.round.beats(g.blind.chips);
	for i in 0..g.areas.jokers.len() {
		if let Some(e) = joker::calc_joker(g, i, LifecycleCtx::EndOfRound { game_over }) {
			if e.saved {
				game_over = false;
			}
			joker::effect_status(&mut g.events, &e);
		}
		let ab = &mut g.areas.jokers[i].ab;
		calculate_rental(ab, &mut g.events, g.run.rental_rate);
		if calculate_perishable(ab, &mut g.events) {
			// The game calls `set_debuff()` without an argument here, which
			// debuffs a perished joker all the same.
			card::set_debuff_joker(g, i, /* should= */ true);
		}
	}
	let is_boss = g.blind.is_boss();
	if g.run.ante == g.setup.win_ante && is_boss {
		g.run.won = true;
	}
	if game_over {
		g.flow.state = State::GameOver;
		g.flow.state_complete = false;
		return;
	}
	g.run.blinds_cleared += 1;
	if g.run.ante == g.setup.win_ante && is_boss && !g.run.win_notified {
		g.run.win_notified = true;
		g.events
			.push_event(Event::WinPoll, Trigger::Immediate, Blocks::Detached);
	}
	g.run.unused_discards += g.round.discards_left;
	if is_boss {
		g.round.most_played_hand = most_played_hand(&g.hands);
	}
	let hn = g.areas.hand.cards.len();
	for i in 0..hn {
		let mut reps = 1;
		let mut j = 0;
		while j < reps {
			if j > 0 {
				// card_eval_status_text for the repetition source
				g.events.status_extra();
			}
			let (eff_nonempty, h_dollars) = end_of_round_effect(&mut g.areas, &mut g.events, i);
			if j == 0 {
				let c = &g.areas.hand.cards[i];
				let red_seal = !c.core.debuff && c.seal == Some(Seal::Red) && eff_nonempty;
				let ctx = LifecycleCtx::EndOfRoundRepetition {
					effects: eff_nonempty,
				};
				reps = scoring::repetitions(g.areas.jokers.len(), red_seal, |k| {
					joker::calc_joker(g, k, ctx)
				});
			}
			if h_dollars > 0.0 {
				// `effects[1].card` → juice
				g.events.push(Event::Wait);
				g.events.push(Event::EaseDollars(h_dollars));
				g.events.status_text(0.65);
			}
			j += 1;
		}
	}
	g.events.delay(0.3);
	for _ in 0..g.areas.hand.cards.len() {
		g.events.draw(Draw::next(Pile::Hand, Pile::Discard), 0.07);
	}
	if is_boss {
		g.events.delay(0.4);
		g.events.push(Event::EaseAnte(1));
		g.events.delay(0.4);
	}
	g.events.push(Event::Flow(FlowEvent::DiscardToDeck));
	g.events
		.push_after(0.3, Event::Flow(FlowEvent::EndRoundFinal));
}

/// `Card:calculate_rental()`: a rental joker charges its rent.
fn calculate_rental(ab: &Ability, events: &mut EventQueue, rental_rate: f64) {
	if ab.rental {
		events.push(Event::EaseDollars(-rental_rate));
		events.status_text(0.65);
	}
}

/// `Card:calculate_perishable()`: a perishable joker counts down a round;
/// returns whether it has just perished.
fn calculate_perishable(ab: &mut Ability, events: &mut EventQueue) -> bool {
	let Some(Sticker::Perishable { tally }) = &mut ab.sticker else {
		return false;
	};
	if *tally <= 0.0 {
		return false;
	}
	*tally -= 1.0;
	events.status_text(0.45);
	*tally == 0.0
}

/// The hand The Ox names: the most played, in `pairs(G.GAME.hands)` order.
/// The game's tie-break compares with an `_order` it never updates, so ties
/// go to the later hand, which is the one `max_by` keeps.
fn most_played_hand(hands: &HandLevels) -> HandType {
	HandType::PAIRS_ORDER
		.into_iter()
		.max_by(|&a, &b| hands[a].played.total_cmp(&hands[b].played))
		.expect("PAIRS_ORDER is not empty")
}

/// `Card:get_end_of_round_effect()` for a held card; returns whether the
/// effect table is non-empty and its `h_dollars`. A Blue Seal also queues
/// its planet when there is room, once per repetition.
fn end_of_round_effect(areas: &mut Areas, events: &mut EventQueue, i: usize) -> (bool, f64) {
	let c = &areas.hand.cards[i];
	if c.core.debuff {
		return (false, 0.0);
	}
	let h = c.enhancement.h_dollars();
	let mut nonempty = h > 0.0;
	if c.seal == Some(Seal::Blue) && areas.consumable_room_after_queued() {
		areas.consumable_buffer += 1.0;
		events.push_before(0.0, Event::Flow(FlowEvent::BlueSealPlanet));
		events.status_extra();
		nonempty = true;
	}
	(nonempty, h)
}

/// The planet a Blue Seal gives for the last hand played.
pub(crate) fn blue_seal_planet(g: &mut Game) {
	if let Some(h) = g.history.last_hand_played {
		let p = Planet::for_hand(h);
		pools::create_owned(
			g,
			NewCard::forced(Kind::Planet(p), OwnedArea::Consumables, "blusl"),
		);
		g.areas.consumable_buffer = 0.0;
	}
}

pub(crate) fn end_round_final(g: &mut Game) {
	g.flow.state = State::RoundEval;
	g.flow.state_complete = false;
	let selected = g
		.blind_select
		.selected
		.expect("a round is only started on a selected blind");
	let slot = BlindSlot::of(selected);
	g.blind_select.states[slot] = BlindStatus::Defeated;
	if slot == BlindSlot::Boss {
		g.shop.voucher = Some(pools::get_next_voucher(
			&g.areas,
			&g.run,
			&g.history,
			&mut g.rng,
			VoucherRoll::Ante,
		));
		g.shop.voucher_seen = false;
		for c in g.areas.playing_cards_mut() {
			c.played_this_ante = false;
		}
	}
	if let Some(t) = g.round.temp_handsize.take() {
		flow::change_hand_size(&mut g.events, -t);
	}
	if g.shop.temp_reroll_cost.is_some() {
		g.shop.temp_reroll_cost = None;
		shop::calculate_reroll_cost(&mut g.shop);
	}
	g.round
		.reset_joker_targets(&g.areas, &g.run, &mut g.rng, JokerTargets::RoundEnd);
	for c in g.areas.playing_cards_mut() {
		c.forced = Forced::No;
	}
}

/// `G.FUNCS.evaluate_round()`.
pub(crate) fn evaluate_round(g: &mut Game) {
	trace!(g.rng, "eval_round");
	let mut dollars = 0.0;
	if g.round.beats(g.blind.chips) {
		dollars += g.blind.dollars;
	}
	g.events.push(Event::Flow(FlowEvent::BlindDefeat));
	// Back:trigger_effect({context = 'eval'})
	if g.setup.back == Back::Anaglyph && g.round.last_blind_boss {
		g.events.push(Event::Tag(TagEvent::Add {
			kind: TagKind::Double,
			orbital: None,
		}));
	}
	if g.round.hands_left > 0.0 {
		dollars += g.round.hands_left * g.setup.modifiers.money_per_hand.unwrap_or(1.0);
	}
	if g.round.discards_left > 0.0
		&& let Some(m) = g.setup.modifiers.money_per_discard
	{
		dollars += g.round.discards_left * m;
	}
	for c in &g.areas.jokers {
		if let Some(d) = dollar_bonus(c, &g.areas, g.history.planets_used, &g.round) {
			dollars += d;
		}
	}
	dollars += run::apply_eval(&g.round, &mut g.tags, &mut g.events);
	if g.run.dollars >= 5.0 && !g.setup.modifiers.no_interest {
		dollars +=
			g.run.interest_amount * (g.run.dollars / 5.0).floor().min(g.run.interest_cap / 5.0);
	}
	g.round.cash_out = dollars;
}

/// `Card:calculate_dollar_bonus()` for the joker `c` (`areas` for Cloud 9's
/// count of Nines).
fn dollar_bonus(c: &Card, areas: &Areas, planets_used: PlanetSet, round: &Round) -> Option<f64> {
	if c.core.debuff {
		return None;
	}
	let j = c.joker()?;
	match j {
		Joker::Golden => Some(4.0),
		Joker::Cloud9 => {
			let count = num::float(
				areas
					.playing_cards()
					.filter(|c| c.has_rank(Rank::Nine))
					.count(),
			);
			if count > 0.0 { Some(count) } else { None }
		}
		Joker::Rocket => Some(c.ab.dollars),
		Joker::Satellite => {
			let count = f64::from(planets_used.count());
			if count == 0.0 { None } else { Some(count) }
		}
		Joker::DelayedGrat if round.discards_used == 0.0 && round.discards_left > 0.0 => {
			Some(round.discards_left * 2.0)
		}
		_ => None,
	}
}

/// `G.FUNCS.cash_out`.
pub(crate) fn do_cash_out(g: &mut Game) {
	let k = g.run.key_ante("cashout");
	flow::shuffle_area(&mut g.areas.deck, &mut g.rng, &k);
	g.events.delay(0.3);
	g.events.push(Event::Flow(FlowEvent::CashOutFinish));
	let d = g.round.cash_out;
	g.events.push(Event::EaseDollars(d));
	// previous_round.dollars bookkeeping
	g.events.push(Event::Wait);
	g.events.push(Event::EaseChips0);
	if g.blind_select.states[BlindSlot::Boss] == BlindStatus::Defeated {
		g.blind_select.skip_tags.small = Some(pools::get_next_tag(&g.run, &g.history, &mut g.rng));
		g.blind_select.skip_tags.big = Some(pools::get_next_tag(&g.run, &g.history, &mut g.rng));
	}
	reset_blinds(
		g.setup.win_ante,
		g.run.ante,
		&mut g.blind_select,
		&mut g.rng,
	);
	g.events.delay(0.6);
}

/// `reset_blinds()`.
pub(crate) fn reset_blinds(
	win_ante: i32, ante: i32, blind_select: &mut BlindSelect, rng: &mut PseudoRandom,
) {
	if blind_select.states[BlindSlot::Boss] == BlindStatus::Defeated {
		blind_select.states = PerBlind::all(BlindStatus::Upcoming);
		blind_select.on_deck = Some(BlindSlot::Small);
		blind_select.boss = pools::get_new_boss(win_ante, ante, blind_select, rng);
		blind_select.boss_rerolled = false;
	}
}
