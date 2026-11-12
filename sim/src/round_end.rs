//! End of round, round evaluation and cash out (`state_events.lua`
//! `end_round`, `button_callbacks.lua` `evaluate_round` / `cash_out`).

use crate::data::{Back, BlindKind, Edition, HandType, Joker, Seal, TagKind};
use crate::ev::{Ev, Pile};
use crate::events::St;
use crate::game::{BlindSlot, BlindStatus, Game};
use crate::jokers::Ctx;
use crate::rng::Key;
use crate::{cards, flow, jokers, num, pools, tags};

// -----------------------------------------------------------------
// End of round
// -----------------------------------------------------------------

#[expect(
	clippy::float_cmp,
	reason = "the game compares these whole-number counters exactly"
)]
pub(crate) fn end_round(g: &mut Game) {
	let mut game_over = g.chips - g.blind.chips < 0.0;
	let n = g.jokers.len();
	for i in 0..n {
		if i >= g.jokers.len() {
			break;
		}
		if let Some(e) = jokers::calc_joker(g, i, &Ctx::EndOfRound { game_over }, None) {
			if e.saved {
				game_over = false;
			}
			jokers::effect_status(g, &e);
		}
		// calculate_rental / calculate_perishable
		if g.jokers[i].ab.rental {
			g.events.push(Ev::EaseDollars(-g.rental_rate));
			g.events.status_text(0.65);
		}
		let ab = g.jokers[i].ab;
		if ab.perishable && ab.perish_tally > 0.0 {
			if ab.perish_tally == 1.0 {
				g.jokers[i].ab.perish_tally = 0.0;
				g.events.status_text(0.45);
				cards::set_debuff_joker(g, i, false);
			} else {
				g.jokers[i].ab.perish_tally -= 1.0;
				g.events.status_text(0.45);
			}
		}
	}
	let is_boss = g.blind.slot() == Some(BlindSlot::Boss);
	if g.ante == g.win_ante && is_boss {
		g.won = true;
	}
	if game_over {
		g.state = St::GameOver;
		g.state_complete = false;
		return;
	}
	g.blinds_cleared += 1;
	if g.ante == g.win_ante && is_boss && !g.win_notified {
		g.win_notified = true;
		g.events
			.push_ev(Ev::WinPoll, crate::events::Trig::Imm, false, false);
	}
	g.unused_discards += g.discards_left;
	if is_boss {
		g.most_played_hand = most_played_hand(g);
	}
	// Held-card end of round effects.
	let hn = g.hand.len();
	for i in 0..hn {
		let mut reps = 1usize;
		let mut j = 0usize;
		while j < reps {
			if j > 0 {
				// card_eval_status_text for the repetition source
				g.events.status_extra();
			}
			let (eff_nonempty, h_dollars) = end_of_round_effect(g, i);
			if j == 0 {
				let c = &g.hand[i];
				if !c.debuff && c.seal == Some(Seal::Red) && eff_nonempty {
					reps += 1;
				}
				for k in 0..g.jokers.len() {
					if let Some(e) = jokers::calc_joker(
						g,
						k,
						&Ctx::EndOfRoundRepetition {
							effects: eff_nonempty,
						},
						None,
					) {
						reps += e.repetitions as usize;
					}
				}
			}
			if h_dollars > 0.0 {
				// `effects[1].card` → juice
				g.events.push(Ev::Wait);
				g.events.push(Ev::EaseDollars(h_dollars));
				g.events.status_text(0.65);
			}
			j += 1;
		}
	}
	g.events.delay(0.3);
	for _ in 0..g.hand.len() {
		g.events
			.draw_ev(Pile::Hand, Pile::Discard, None, false, false, 0.07);
	}
	if is_boss {
		g.events.delay(0.4);
		g.events.push(Ev::EaseAnte(1));
		g.events.delay(0.4);
	}
	g.events
		.push(Ev::Consumable(crate::consumables::CEv::DiscardToDeckOuter));
	g.events.push_after(0.3, Ev::EndRoundFinal);
}

/// The hand The Ox names: the most played, as `end_round` picks it.
#[expect(
	clippy::float_cmp,
	reason = "the game compares these whole-number counters exactly"
)]
fn most_played_hand(g: &Game) -> HandType {
	let mut name = HandType::HighCard;
	let mut played = -1.0;
	for h in HandType::PAIRS_ORDER {
		let v = g.hands[h as usize].played;
		// `_order` is never updated, so ties go to the later key.
		if v > played || (v == played && 100 > h.order()) {
			played = v;
			name = h;
		}
	}
	name
}

/// `Card:get_end_of_round_effect()` for a held card; returns whether the
/// effect table is non-empty and its `h_dollars`.
fn end_of_round_effect(g: &mut Game, i: usize) -> (bool, f64) {
	let c = &g.hand[i];
	if c.debuff {
		return (false, 0.0);
	}
	let h = c.enhancement.h_dollars();
	let mut nonempty = h > 0.0;
	if c.seal == Some(Seal::Blue) && jokers::consumable_room(g) {
		g.consumeable_buffer += 1.0;
		g.events
			.push_before(0.0, Ev::Consumable(crate::consumables::CEv::BlueSealPlanet));
		g.events.status_extra();
		nonempty = true;
	}
	(nonempty, h)
}

pub(crate) fn end_round_final(g: &mut Game) {
	g.state = St::RoundEval;
	g.state_complete = false;
	match g.selected_blind {
		Some(BlindKind::Small) => g.blind_states[0] = BlindStatus::Defeated,
		Some(BlindKind::Big) => g.blind_states[1] = BlindStatus::Defeated,
		_ => {
			g.current_voucher = Some(pools::get_next_voucher(g, false));
			g.voucher_seen = false;
			g.blind_states[2] = BlindStatus::Defeated;
			for c in g.playing_cards_mut() {
				c.ab.played_this_ante = false;
			}
		}
	}
	if let Some(t) = g.temp_handsize.take() {
		cards::change_hand_size(g, -t);
	}
	if g.temp_reroll_cost.is_some() {
		g.temp_reroll_cost = None;
		cards::calculate_reroll_cost(g, true);
	}
	pools::reset_idol_card(g);
	pools::reset_mail_rank(g);
	pools::reset_ancient_card(g);
	pools::reset_castle_card(g);
	for c in g.playing_cards_mut() {
		c.ab.discarded = false;
		c.ab.forced_selection = false;
	}
}

/// `G.FUNCS.evaluate_round()`.
pub(crate) fn evaluate_round(g: &mut Game) {
	crate::game::trace!(g, "eval_round");
	let mut dollars = 0.0;
	if g.chips - g.blind.chips >= 0.0 {
		dollars += g.blind.dollars;
	}
	g.events
		.push(Ev::Consumable(crate::consumables::CEv::BlindDefeat));
	// Back:trigger_effect({context = 'eval'})
	if g.back == Back::Anaglyph && g.last_blind_boss {
		g.events.push(Ev::AddTag {
			kind: TagKind::Double,
			orbital: None,
		});
	}
	if g.hands_left > 0.0 {
		dollars += g.hands_left * g.modifiers.money_per_hand.unwrap_or(1.0);
	}
	if g.discards_left > 0.0
		&& let Some(m) = g.modifiers.money_per_discard
	{
		dollars += g.discards_left * m;
	}
	for i in 0..g.jokers.len() {
		if let Some(d) = dollar_bonus(g, i) {
			dollars += d;
		}
	}
	for i in 0..g.tags.len() {
		if let Some(d) = tags::apply_tag_eval(g, i) {
			dollars += d;
		}
	}
	if g.dollars >= 5.0 && !g.modifiers.no_interest {
		dollars += g.interest_amount * (g.dollars / 5.0).floor().min(g.interest_cap / 5.0);
	}
	g.round_dollars = dollars;
}

/// `Card:calculate_dollar_bonus()`.
fn dollar_bonus(g: &Game, i: usize) -> Option<f64> {
	let c = &g.jokers[i];
	if c.debuff {
		return None;
	}
	let j = c.joker()?;
	match j {
		Joker::Golden => Some(4.0),
		Joker::Cloud9 => {
			let count = num::float(g.playing_cards().filter(|c| c.id_is(9)).count());
			if count > 0.0 { Some(count) } else { None }
		}
		Joker::Rocket => Some(c.ab.dollars),
		Joker::Satellite => {
			let count = f64::from(g.planets_used.count_ones());
			if count == 0.0 { None } else { Some(count) }
		}
		Joker::DelayedGrat if g.discards_used == 0.0 && g.discards_left > 0.0 => {
			Some(g.discards_left * 2.0)
		}
		_ => None,
	}
}

/// `G.FUNCS.cash_out`.
pub(crate) fn do_cash_out(g: &mut Game) {
	let k = g.key_ante("cashout");
	flow::shuffle_deck(g, &k);
	g.events.delay(0.3);
	g.events.push(Ev::CashOutFinish);
	let d = g.round_dollars;
	g.events.push(Ev::EaseDollars(d));
	// previous_round.dollars bookkeeping
	g.events.push(Ev::Wait);
	g.events.push(Ev::EaseChips0);
	if g.blind_states[2] == BlindStatus::Defeated {
		g.blind_ante = g.ante;
		g.blind_tags[0] = Some(pools::get_next_tag(g, ""));
		g.blind_tags[1] = Some(pools::get_next_tag(g, ""));
	}
	reset_blinds(g);
	g.events.delay(0.6);
}

/// `reset_blinds()`.
pub(crate) fn reset_blinds(g: &mut Game) {
	if g.blind_states[2] == BlindStatus::Defeated {
		g.blind_states = [BlindStatus::Upcoming; 3];
		g.blind_on_deck = Some(BlindSlot::Small);
		g.boss_choice = pools::get_new_boss(g);
		g.boss_rerolled = false;
	}
}

/// Body of Perkeo's ending-shop event.
pub(crate) fn perkeo_copy(g: &mut Game) {
	if g.consumables.is_empty() {
		return;
	}
	let mut order: Vec<usize> = (0..g.consumables.len()).collect();
	order.sort_by_key(|&i| g.consumables[i].id);
	let k = Key::from("perkeo");
	let pick = order[g.rng.element(&k, order.len())];
	let mut c = flow::copy_card(g, g.consumables[pick].clone(), false);
	cards::set_edition(g, &mut c, Edition::Negative, None);
	g.events.edition_fx(Edition::Negative, true);
	cards::add_to_deck(g, &mut c, false);
	g.consumables.push(c);
}
