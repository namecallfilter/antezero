//! Port of `blind.lua`.

use crate::card::Card;
use crate::data::*;
use crate::rng::Key;
use crate::events::{Ev, Pile};
use crate::game::{Blind, Game};

impl Game {
    /// `Blind:set_blind(blind, reset=false)`.
    pub(crate) fn set_blind(&mut self, blind: Option<BlindKind>) {
        let mut b = Blind::empty();
        b.kind = blind;
        if let Some(k) = blind {
            let d = k.def();
            b.dollars = d.dollars as f64;
            b.mult = d.mult;
            b.boss = d.boss;
            if self.modifiers.no_blind_reward[k.slot() as usize] {
                b.dollars = 0.0;
            }
            b.chips = blind_amount(self.ante, self.modifiers.scaling) * b.mult * self.starting_params.ante_scaling;
        }
        b.prepped = true;
        self.last_blind_boss = b.boss;
        self.blind = b;
        if blind.is_some() {
            self.push_ev(Ev::BlindSetHud, crate::events::Trig::After(0.05), true, false);
        }
        match blind {
            Some(BlindKind::Fish) => self.blind.prepped = false,
            Some(BlindKind::Water) => {
                self.blind.discards_sub = self.discards_left;
                let d = self.blind.discards_sub;
                self.push(Ev::EaseDiscard(-d));
            }
            Some(BlindKind::Needle) => {
                self.blind.hands_sub = self.rr_hands - 1.0;
                let h = self.blind.hands_sub;
                self.push(Ev::EaseHands(-h));
            }
            Some(BlindKind::Manacle) => self.change_hand_size(-1.0),
            Some(BlindKind::FinalAcorn) if !self.jokers.is_empty() => {
                for c in self.jokers.iter_mut() {
                    c.face_down = !c.face_down;
                }
                if self.jokers.len() > 1 {
                    self.push_after(0.2, Ev::AcornOuter);
                }
            }
            _ => {}
        }
        self.refresh_playing_debuffs();
        for i in 0..self.jokers.len() {
            self.debuff_joker(i);
        }
    }

    /// `Blind:debuff_card(card)` for a card outside the joker area.
    pub(crate) fn debuff_card(&self, c: &mut Card) {
        let b = &self.blind;
        if !b.disabled {
            let smeared = self.smeared();
            let suit = match b.kind {
                Some(BlindKind::Club) => Some(Suit::Clubs),
                Some(BlindKind::Goad) => Some(Suit::Spades),
                Some(BlindKind::Head) => Some(Suit::Hearts),
                Some(BlindKind::Window) => Some(Suit::Diamonds),
                _ => None,
            };
            if let Some(s) = suit && c.is_suit(s, true, false, smeared) {
                Game::set_debuff_playing(c, true);
                return;
            }
            if b.kind == Some(BlindKind::Plant) && c.is_face(true, self.pareidolia()) {
                Game::set_debuff_playing(c, true);
                return;
            }
            if b.kind == Some(BlindKind::Pillar) && c.ab.played_this_ante {
                Game::set_debuff_playing(c, true);
                return;
            }
            if b.kind == Some(BlindKind::FinalLeaf) {
                Game::set_debuff_playing(c, true);
                return;
            }
        }
        Game::set_debuff_playing(c, false);
    }

    /// `Blind:debuff_card(joker)`.
    pub(crate) fn debuff_joker(&mut self, i: usize) {
        if self.blind.kind == Some(BlindKind::FinalHeart) && !self.blind.disabled {
            return;
        }
        self.set_debuff_joker(i, false);
    }

    /// `Blind:disable()`.
    pub(crate) fn disable_blind(&mut self) {
        self.blind.disabled = true;
        for c in self.jokers.iter_mut() {
            c.face_down = false;
        }
        match self.blind.kind {
            Some(BlindKind::Water) => {
                let d = self.blind.discards_sub;
                self.push(Ev::EaseDiscard(d));
            }
            Some(BlindKind::Wheel | BlindKind::House | BlindKind::Mark | BlindKind::Fish) => {
                for c in self.hand.iter_mut() {
                    c.face_down = false;
                }
                for c in self.playing_cards_mut() {
                    c.ab.wheel_flipped = false;
                }
            }
            Some(BlindKind::Needle) => {
                let h = self.blind.hands_sub;
                self.push(Ev::EaseHands(h));
            }
            Some(BlindKind::Wall) => self.blind.chips /= 2.0,
            Some(BlindKind::FinalBell) => {
                for c in self.playing_cards_mut() {
                    c.ab.forced_selection = false;
                }
            }
            Some(BlindKind::Manacle) => {
                self.change_hand_size(1.0);
                self.draw_from_deck_to_hand(Some(1));
            }
            Some(BlindKind::FinalVessel) => self.blind.chips /= 3.0,
            _ => {}
        }
        self.push(Ev::BlindDisableCheck);
        self.refresh_playing_debuffs();
        for i in 0..self.jokers.len() {
            self.debuff_joker(i);
        }
    }

    /// `Blind:press_play()`; returns its (truthy) result.
    pub(crate) fn blind_press_play(&mut self) -> bool {
        if self.blind.disabled {
            return false;
        }
        match self.blind.kind {
            Some(BlindKind::Hook) => {
                self.push(Ev::HookDiscard);
                self.blind.triggered = true;
                self.delay(0.7);
                return true;
            }
            Some(BlindKind::FinalHeart) => {
                if !self.jokers.is_empty() {
                    self.blind.triggered = true;
                    self.blind.prepped = true;
                }
            }
            Some(BlindKind::Fish) => self.blind.prepped = true,
            Some(BlindKind::Tooth) => {
                self.push_after(0.2, Ev::ToothOuter);
                self.blind.triggered = true;
                return true;
            }
            _ => {}
        }
        false
    }

    /// `Blind:modify_hand` → `(mult, chips)`.
    pub(crate) fn blind_modify_hand(&mut self, mult: f64, chips: f64) -> (f64, f64) {
        if self.blind.disabled {
            return (mult, chips);
        }
        if self.blind.kind == Some(BlindKind::Flint) {
            self.blind.triggered = true;
            return ((mult * 0.5 + 0.5).floor().max(1.0), (chips * 0.5 + 0.5).floor().max(0.0));
        }
        (mult, chips)
    }

    /// `Blind:debuff_hand(cards, hand, handname, check)`.
    pub(crate) fn blind_debuff_hand(&mut self, n_cards: usize, name: HandType, check: bool) -> bool {
        if self.blind.disabled {
            return false;
        }
        self.blind.triggered = false;
        match self.blind.kind {
            Some(BlindKind::Psychic) if n_cards < 5 => {
                self.blind.triggered = true;
                return true;
            }
            Some(BlindKind::Eye) => {
                if self.blind.eye_hands >> (name as u16) & 1 == 1 {
                    self.blind.triggered = true;
                    return true;
                }
                if !check {
                    self.blind.eye_hands |= 1 << (name as u16);
                }
            }
            Some(BlindKind::Mouth) => {
                if let Some(only) = self.blind.only_hand && only != name {
                    self.blind.triggered = true;
                    return true;
                }
                if !check {
                    self.blind.only_hand = Some(name);
                }
            }
            _ => {}
        }
        if self.blind.kind == Some(BlindKind::Arm) {
            self.blind.triggered = false;
            if self.hands[name as usize].level > 1.0 {
                self.blind.triggered = true;
                if !check {
                    self.level_up_hand_ev(name, -1.0, false);
                }
            }
        }
        if self.blind.kind == Some(BlindKind::Ox) {
            self.blind.triggered = false;
            if name == self.most_played_hand {
                self.blind.triggered = true;
                if !check {
                    let d = self.dollars;
                    self.ease_dollars(-d);
                }
            }
        }
        false
    }

    /// `Blind:drawn_to_hand()`.
    pub(crate) fn blind_drawn_to_hand(&mut self) {
        if !self.blind.disabled {
            if self.blind.kind == Some(BlindKind::FinalBell) && !self.hand.is_empty() {
                let any = self.hand.iter().any(|c| c.ab.forced_selection);
                if !any {
                    let mut order: Vec<usize> = (0..self.hand.len()).collect();
                    order.sort_by_key(|&i| self.hand[i].id);
                    let k = Key::from("cerulean_bell");
                    let pick = order[self.rng.element(&k, order.len())];
                    self.hand[pick].ab.forced_selection = true;
                }
            }
            if self.blind.kind == Some(BlindKind::FinalHeart) && self.blind.prepped && !self.jokers.is_empty() {
                let n = self.jokers.len();
                let mut cands: Vec<usize> = Vec::new();
                for i in 0..n {
                    if !self.jokers[i].debuff || n < 2 {
                        cands.push(i);
                    }
                    self.set_debuff_joker(i, false);
                }
                if !cands.is_empty() {
                    cands.sort_by_key(|&i| self.jokers[i].id);
                    let k = Key::from("crimson_heart");
                    let pick = cands[self.rng.element(&k, cands.len())];
                    self.set_debuff_joker(pick, true);
                }
            }
        }
        self.blind.prepped = false;
    }

    /// `Blind:stay_flipped(area, card)`.
    pub(crate) fn blind_stay_flipped(&mut self, to: Pile, c: &Card) -> bool {
        if self.blind.disabled || to != Pile::Hand {
            return false;
        }
        match self.blind.kind {
            Some(BlindKind::Wheel) => {
                let k = Key::from("wheel");
                self.rng.random(&k) < self.probabilities / 7.0
            }
            Some(BlindKind::House) => self.hands_played_round == 0.0 && self.discards_used == 0.0,
            Some(BlindKind::Mark) => c.is_face(true, self.pareidolia()),
            Some(BlindKind::Fish) => self.blind.prepped,
            _ => false,
        }
    }

    /// `Blind:defeat()` (the parts with game effects).
    pub(crate) fn blind_defeat(&mut self) {
        for c in self.jokers.iter_mut() {
            c.face_down = false;
        }
        if self.blind.kind == Some(BlindKind::Manacle) && !self.blind.disabled {
            self.change_hand_size(1.0);
        }
        // `set_blind(nil)` runs at the end of the defeat animation.
        self.push(Ev::BlindReset);
    }
}
