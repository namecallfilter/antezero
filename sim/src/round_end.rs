//! End of round, round evaluation and cash out (state_events.lua
//! `end_round`, button_callbacks.lua `evaluate_round` / `cash_out`).

use crate::data::*;
use crate::events::{Ev, Pile, St};
use crate::game::{BlindSlot, BlindStatus, Game};
use crate::jokers::Ctx;
use crate::rng::Key;

impl Game {
    // -----------------------------------------------------------------
    // End of round
    // -----------------------------------------------------------------

    pub(crate) fn end_round(&mut self) {
        let mut game_over = self.chips - self.blind.chips < 0.0;
        let n = self.jokers.len();
        for i in 0..n {
            if i >= self.jokers.len() {
                break;
            }
            if let Some(e) = self.calc_joker(i, &Ctx::EndOfRound { game_over }, None) {
                if e.saved {
                    game_over = false;
                }
                self.effect_status(&e);
            }
            // calculate_rental / calculate_perishable
            if self.jokers[i].ab.rental {
                self.push(Ev::EaseDollars(-self.rental_rate));
                self.status_text(0.65);
            }
            let ab = self.jokers[i].ab;
            if ab.perishable && ab.perish_tally > 0.0 {
                if ab.perish_tally == 1.0 {
                    self.jokers[i].ab.perish_tally = 0.0;
                    self.status_text(0.45);
                    self.set_debuff_joker(i, false);
                } else {
                    self.jokers[i].ab.perish_tally -= 1.0;
                    self.status_text(0.45);
                }
            }
        }
        let is_boss = self.blind.slot() == Some(BlindSlot::Boss);
        if self.ante == self.win_ante && is_boss {
            self.won = true;
        }
        if game_over {
            self.state = St::GameOver;
            self.state_complete = false;
            return;
        }
        self.blinds_cleared += 1;
        if self.ante == self.win_ante && is_boss && !self.win_notified {
            self.win_notified = true;
            self.push_ev(Ev::WinPoll, crate::events::Trig::Imm, false, false);
        }
        self.unused_discards += self.discards_left;
        if is_boss {
            let mut name = HandType::HighCard;
            let mut played = -1.0;
            for h in HandType::PAIRS_ORDER {
                let v = self.hands[h as usize].played;
                // `_order` is never updated, so ties go to the later key.
                if v > played || (v == played && 100 > h.order()) {
                    played = v;
                    name = h;
                }
            }
            self.most_played_hand = name;
        }
        // Held-card end of round effects.
        let hn = self.hand.len();
        for i in 0..hn {
            let mut reps = 1usize;
            let mut j = 0usize;
            while j < reps {
                if j > 0 {
                    // card_eval_status_text for the repetition source
                    self.status_extra();
                }
                let (eff_nonempty, h_dollars) = self.end_of_round_effect(i);
                if j == 0 {
                    let c = &self.hand[i];
                    if !c.debuff && c.seal == Some(Seal::Red) && eff_nonempty {
                        reps += 1;
                    }
                    for k in 0..self.jokers.len() {
                        if let Some(e) = self.calc_joker(k, &Ctx::EndOfRoundRepetition { effects: eff_nonempty }, None) {
                            reps += e.repetitions as usize;
                        }
                    }
                }
                if h_dollars > 0.0 {
                    // `effects[1].card` → juice
                    self.push(Ev::Wait);
                    self.push(Ev::EaseDollars(h_dollars));
                    self.status_text(0.65);
                }
                j += 1;
            }
        }
        self.delay(0.3);
        for _ in 0..self.hand.len() {
            self.draw_ev(Pile::Hand, Pile::Discard, None, false, false, 0.07);
        }
        if is_boss {
            self.delay(0.4);
            self.push(Ev::EaseAnte(1));
            self.delay(0.4);
        }
        self.push(Ev::Consumable(crate::consumables::CEv::DiscardToDeckOuter));
        self.push_after(0.3, Ev::EndRoundFinal);
    }

    /// `Card:get_end_of_round_effect()` for a held card; returns whether the
    /// effect table is non-empty and its `h_dollars`.
    fn end_of_round_effect(&mut self, i: usize) -> (bool, f64) {
        let c = &self.hand[i];
        if c.debuff {
            return (false, 0.0);
        }
        let h = c.enhancement.h_dollars();
        let mut nonempty = h > 0.0;
        if c.seal == Some(Seal::Blue) && self.consumable_room() {
            self.consumeable_buffer += 1.0;
            self.push_before(0.0, Ev::Consumable(crate::consumables::CEv::BlueSealPlanet));
            self.status_extra();
            nonempty = true;
        }
        (nonempty, h)
    }

    pub(crate) fn end_round_final(&mut self) {
        self.state = St::RoundEval;
        self.state_complete = false;
        match self.selected_blind {
            Some(BlindKind::Small) => self.blind_states[0] = BlindStatus::Defeated,
            Some(BlindKind::Big) => self.blind_states[1] = BlindStatus::Defeated,
            _ => {
                self.current_voucher = Some(self.get_next_voucher(false));
                self.voucher_seen = false;
                self.blind_states[2] = BlindStatus::Defeated;
                for c in self.playing_cards_mut() {
                    c.ab.played_this_ante = false;
                }
            }
        }
        if let Some(t) = self.temp_handsize.take() {
            self.change_hand_size(-t);
        }
        if self.temp_reroll_cost.is_some() {
            self.temp_reroll_cost = None;
            self.calculate_reroll_cost(true);
        }
        self.reset_idol_card();
        self.reset_mail_rank();
        self.reset_ancient_card();
        self.reset_castle_card();
        for c in self.playing_cards_mut() {
            c.ab.discarded = false;
            c.ab.forced_selection = false;
        }
    }

    /// `G.FUNCS.evaluate_round()`.
    pub(crate) fn evaluate_round(&mut self) {
        crate::game::trace!(self, "eval_round");
        let mut dollars = 0.0;
        if self.chips - self.blind.chips >= 0.0 {
            dollars += self.blind.dollars;
        }
        self.push(Ev::Consumable(crate::consumables::CEv::BlindDefeat));
        // Back:trigger_effect({context = 'eval'})
        if self.back == Back::Anaglyph && self.last_blind_boss {
            self.push(Ev::AddTag { kind: TagKind::Double, orbital: None });
        }
        if self.hands_left > 0.0 {
            dollars += self.hands_left * self.modifiers.money_per_hand.unwrap_or(1.0);
        }
        if self.discards_left > 0.0 && let Some(m) = self.modifiers.money_per_discard {
            dollars += self.discards_left * m;
        }
        for i in 0..self.jokers.len() {
            if let Some(d) = self.dollar_bonus(i) {
                dollars += d;
            }
        }
        for i in 0..self.tags.len() {
            if let Some(d) = self.apply_tag_eval(i) {
                dollars += d;
            }
        }
        if self.dollars >= 5.0 && !self.modifiers.no_interest {
            dollars += self.interest_amount * (self.dollars / 5.0).floor().min(self.interest_cap / 5.0);
        }
        self.round_dollars = dollars;
    }

    /// `Card:calculate_dollar_bonus()`.
    fn dollar_bonus(&self, i: usize) -> Option<f64> {
        let c = &self.jokers[i];
        if c.debuff {
            return None;
        }
        let j = c.joker()?;
        match j {
            Joker::Golden => Some(4.0),
            Joker::Cloud9 => {
                let n = self.playing_cards().filter(|c| c.id_is(9)).count() as f64;
                if n > 0.0 { Some(n) } else { None }
            }
            Joker::Rocket => Some(c.ab.dollars),
            Joker::Satellite => {
                let n = self.planets_used.count_ones() as f64;
                if n == 0.0 { None } else { Some(n) }
            }
            Joker::DelayedGrat if self.discards_used == 0.0 && self.discards_left > 0.0 => Some(self.discards_left * 2.0),
            _ => None,
        }
    }

    /// `G.FUNCS.cash_out`.
    pub(crate) fn do_cash_out(&mut self) {
        let k = self.key_ante("cashout");
        self.shuffle_deck(&k);
        self.delay(0.3);
        self.push(Ev::CashOutFinish);
        let d = self.round_dollars;
        self.push(Ev::EaseDollars(d));
        // previous_round.dollars bookkeeping
        self.push(Ev::Wait);
        self.push(Ev::EaseChips0);
        if self.blind_states[2] == BlindStatus::Defeated {
            self.blind_ante = self.ante;
            self.blind_tags[0] = Some(self.get_next_tag(""));
            self.blind_tags[1] = Some(self.get_next_tag(""));
        }
        self.reset_blinds();
        self.delay(0.6);
    }

    /// `reset_blinds()`.
    pub(crate) fn reset_blinds(&mut self) {
        if self.blind_states[2] == BlindStatus::Defeated {
            self.blind_states = [BlindStatus::Upcoming; 3];
            self.blind_on_deck = Some(BlindSlot::Small);
            self.boss_choice = self.get_new_boss();
            self.boss_rerolled = false;
        }
    }

    /// Body of Perkeo's ending-shop event.
    pub(crate) fn perkeo_copy(&mut self) {
        if self.consumables.is_empty() {
            return;
        }
        let mut order: Vec<usize> = (0..self.consumables.len()).collect();
        order.sort_by_key(|&i| self.consumables[i].id);
        let k = Key::from("perkeo");
        let pick = order[self.rng.element(&k, order.len())];
        let src = self.consumables[pick].clone();
        let mut c = self.copy_card(&src, false);
        self.set_edition(&mut c, Edition::Negative, None);
        self.edition_fx(Edition::Negative, true);
        self.add_to_deck(&mut c, false);
        self.consumables.push(c);
    }
}
