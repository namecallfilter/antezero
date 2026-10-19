//! Port of `G.FUNCS.evaluate_play` (state_events.lua).

use crate::card::Kind;
use crate::data::*;
use crate::rng::Key;
use crate::events::Ev;
use crate::game::{Area, CardSet, Game};
use crate::hand_eval;
use crate::jokers::{CardArea, Ctx, Effect, Scoring};

/// A playing card's own `eval_card` result in the play area.
#[derive(Default, Clone, Copy)]
struct CardEval {
    chips: Option<f64>,
    mult: Option<f64>,
    x_mult: Option<f64>,
    p_dollars: Option<f64>,
    ed_chips: Option<f64>,
    ed_mult: Option<f64>,
    ed_xmult: Option<f64>,
    has_edition: bool,
}

/// Inputs to `Card:update`'s per-frame joker values.
#[derive(Default)]
struct Derived {
    steel: f64,
    stone: f64,
    driver: f64,
    nine: f64,
    n_jokers: f64,
    stencils: f64,
    limit: f64,
    skips: f64,
}

impl Derived {
    fn apply(&self, c: &mut crate::card::Card, sells: &[f64], me: Option<usize>) {
        match c.kind {
            Kind::Joker(Joker::Throwback) => c.ab.x_mult = 1.0 + self.skips * 0.25,
            Kind::Joker(Joker::DriversLicense) => c.ab.driver_tally = self.driver,
            Kind::Joker(Joker::SteelJoker) => c.ab.steel_tally = self.steel,
            Kind::Joker(Joker::Cloud9) => c.ab.nine_tally = self.nine,
            Kind::Joker(Joker::Stone) => c.ab.stone_tally = self.stone,
            Kind::Joker(Joker::Stencil) => c.ab.x_mult = (self.limit - self.n_jokers) + self.stencils,
            Kind::Joker(Joker::Swashbuckler) => {
                let mut s = 0.0;
                for (k, v) in sells.iter().enumerate() {
                    if Some(k) != me {
                        s += *v;
                    }
                }
                c.ab.mult = s;
            }
            _ => {}
        }
    }
}

impl Game {
    /// Recomputes the joker values the game's `Card:update` refreshes every
    /// frame (Steel / Stone tallies, Stencil, Swashbuckler, Temperance, ...).
    /// Observers call it before reading a state.
    pub fn update_derived(&mut self) {
        let mut d = Derived::default();
        for c in self.playing_cards() {
            match c.enhancement {
                Enhancement::Steel => d.steel += 1.0,
                Enhancement::Stone => d.stone += 1.0,
                _ => {}
            }
            if c.enhancement != Enhancement::Base {
                d.driver += 1.0;
            }
            if c.id_is(9) {
                d.nine += 1.0;
            }
        }
        d.n_jokers = self.jokers.len() as f64;
        d.stencils = self.jokers.iter().filter(|c| c.is_joker(Joker::Stencil)).count() as f64;
        d.limit = self.joker_limit.limit;
        d.skips = self.skips;
        let sells: Vec<f64> = self.jokers.iter().map(|c| c.sell_cost).collect();
        let temperance = self.joker_sell_total();
        for (i, c) in self.jokers.iter_mut().enumerate() {
            d.apply(c, &sells, Some(i));
        }
        for c in self.shop_jokers.iter_mut().chain(self.pack_cards.iter_mut()) {
            d.apply(c, &sells, None);
        }
        for c in self.consumables.iter_mut().chain(self.shop_jokers.iter_mut()).chain(self.pack_cards.iter_mut()) {
            if c.kind == Kind::Tarot(Tarot::Temperance) {
                c.ab.money = temperance.min(50.0);
            }
        }
    }

    /// `eval_card(playing_card, {cardarea = G.play})` (RNG for Lucky cards).
    fn eval_play_card(&mut self, i: usize) -> CardEval {
        let prob = self.probabilities;
        let mut r = CardEval::default();
        let c = &self.play[i];
        let chips = c.chip_bonus();
        if chips > 0.0 {
            r.chips = Some(chips);
        }
        // get_chip_mult
        if !c.debuff {
            let m = if c.enhancement == Enhancement::Lucky {
                let k = Key::from("lucky_mult");
                if self.rng.random(&k) < prob / 5.0 {
                    self.play[i].lucky_trigger = true;
                    20.0
                } else {
                    0.0
                }
            } else {
                c.enhancement.mult()
            };
            if m > 0.0 {
                r.mult = Some(m);
            }
        }
        let c = &self.play[i];
        if !c.debuff && c.enhancement.x_mult() > 1.0 {
            r.x_mult = Some(c.enhancement.x_mult());
        }
        // get_p_dollars
        if !c.debuff {
            let mut ret = 0.0;
            if c.seal == Some(Seal::Gold) {
                ret += 3.0;
            }
            if c.enhancement.p_dollars() > 0.0 && c.enhancement == Enhancement::Lucky {
                let k = Key::from("lucky_money");
                if self.rng.random(&k) < prob / 15.0 {
                    self.play[i].lucky_trigger = true;
                    ret += 20.0;
                }
            }
            if ret > 0.0 {
                self.buffer_dollars(ret);
                r.p_dollars = Some(ret);
            }
        }
        let c = &self.play[i];
        if !c.debuff {
            match c.edition {
                Edition::None => {}
                Edition::Foil => {
                    r.has_edition = true;
                    r.ed_chips = Some(50.0);
                }
                Edition::Holo => {
                    r.has_edition = true;
                    r.ed_mult = Some(10.0);
                }
                Edition::Polychrome => {
                    r.has_edition = true;
                    r.ed_xmult = Some(1.5);
                }
                Edition::Negative => r.has_edition = true,
            }
        }
        r
    }

    /// Applies one joker effect table in `evaluate_play`'s fixed field order
    /// (with its juice / hand text / status text events).
    fn apply_play_effect(&mut self, chips: &mut f64, mult: &mut f64, e: &Effect) {
        // Every field queues the juice of the effect's card first.
        if let Some(c) = e.chips {
            self.push(Ev::Wait);
            *chips += c;
            self.hand_text(0.0);
            self.status_text(0.6);
        }
        if let Some(m) = e.mult {
            self.push(Ev::Wait);
            *mult += m;
            self.hand_text(0.0);
            self.status_text(0.65);
        }
        if let Some(d) = e.dollars {
            self.push(Ev::Wait);
            self.push(Ev::EaseDollars(d));
            self.status_text(0.65);
        }
        if e.extra {
            self.push(Ev::Wait);
            if e.eight_ball {
                self.push_before(0.0, Ev::Create { set: CardSet::Tarot, area: Area::Consumables, forced: None, append: "8ba", rarity: None, legendary: false });
            }
            self.hand_text(0.0);
            self.status_text(e.delay.unwrap_or(0.75));
        }
        if let Some(x) = e.x_mult {
            self.push(Ev::Wait);
            *mult *= x;
            self.hand_text(0.0);
            self.status_text(0.65);
        }
    }

    pub(crate) fn evaluate_play(&mut self) {
        self.update_derived();
        let flags = self.eval_flags();
        let ph = {
            let refs: Vec<&crate::card::Card> = self.play.iter().collect();
            hand_eval::evaluate(&refs, &flags)
        };
        let text = ph.top.expect("evaluate_play with no cards");
        {
            let h = &mut self.hands[text as usize];
            h.played += 1.0;
            h.played_this_round += 1.0;
            h.visible = true;
        }
        self.last_hand_played = Some(text);
        let mut mask = ph.top_mask;
        if self.has_joker(Joker::Splash) {
            mask = ((1u16 << self.play.len()) - 1) as u8;
        } else {
            for (i, c) in self.play.iter().enumerate() {
                if c.is_stone() {
                    mask |= 1 << i;
                }
            }
        }
        let mut sc = Scoring { name: text, hands: ph, scoring: [0; 8], n_scoring: 0 };
        for i in 0..self.play.len() {
            if mask >> i & 1 == 1 {
                sc.scoring[sc.n_scoring as usize] = i as u8;
                sc.n_scoring += 1;
            }
        }
        self.last_scoring = sc.scoring()[..].iter().map(|&i| self.play[i as usize].id).collect();

        // Selection/highlight visuals. The hand text shows the selection's
        // name unless a selected card was face down ('????').
        let renamed = self.hand_text_cleared || self.play.iter().any(|c| c.face_down);
        self.delay(0.2);
        for _ in 0..sc.n_scoring {
            self.highlight_ev();
        }
        if renamed {
            self.delay(0.3);
        }
        self.hand_text_imm(if renamed { 0.4 } else { 0.0 });

        let mut hand_chips;
        let mut mult;
        let n_play = self.play.len();
        if !self.blind_debuff_hand(n_play, text, false) {
            self.delay(0.4);
            // before
            let nj = self.jokers.len();
            for i in 0..nj {
                if i >= self.jokers.len() {
                    break;
                }
                if let Some(e) = self.calc_joker(i, &Ctx::Before, Some(&sc)) {
                    self.effect_status(&e);
                    if e.playing_cards_created {
                        self.playing_card_joker_effects(1);
                    }
                    if e.level_up {
                        self.level_up_hand_ev(text, 1.0, false);
                    }
                }
            }
            mult = self.hands[text as usize].mult;
            hand_chips = self.hands[text as usize].chips;
            let flint = self.blind.active(BlindKind::Flint);
            let (m, c) = self.blind_modify_hand(mult, hand_chips);
            mult = m;
            hand_chips = c;
            if flint {
                self.hand_text(0.8);
            }

            self.score_played_cards(&sc, &mut hand_chips, &mut mult);
            self.score_held_cards(&sc, &mut mult);
            self.score_joker_area(&sc, &mut hand_chips, &mut mult);

            // Back:trigger_effect('final_scoring_step')
            if self.back == Back::Plasma {
                let tot = hand_chips + mult;
                hand_chips = (tot / 2.0).floor();
                mult = (tot / 2.0).floor();
                self.hand_text(0.0);
                self.push(Ev::Wait);
                self.delay(0.6);
            }

            self.destroy_scored_cards(&sc, n_play);
        } else {
            mult = 0.0;
            hand_chips = 0.0;
            self.push(Ev::Wait);
            // play_area_status_text("Not Allowed!")
            self.wait(crate::events::Trig::Before(0.6));
            for i in 0..self.jokers.len() {
                if let Some(e) = self.calc_joker(i, &Ctx::DebuffedHand, Some(&sc)) {
                    self.effect_status(&e);
                }
            }
        }
        let score = (hand_chips * mult).floor();
        self.last_hand_score = score;
        self.scored = true;
        self.push_after(0.4, Ev::PushHandTextImm);
        if hand_chips * mult > 0.0 {
            self.delay(0.8);
            self.push(Ev::Wait);
        }
        let target = self.chips + score;
        self.push_ev(Ev::SetChips(target), crate::events::Trig::Ease(0.5), false, true);
        self.wait(crate::events::Trig::Ease(0.5));
        self.push(Ev::Wait);
        self.delay(0.3);
        for i in 0..self.jokers.len() {
            if let Some(e) = self.calc_joker(i, &Ctx::After, Some(&sc)) {
                self.effect_status(&e);
            }
        }
        self.push(Ev::Wait);
    }

    /// The scoring cards of the played hand (with repetitions): their own
    /// chips / mult / money / editions, then the joker effects on each.
    fn score_played_cards(&mut self, sc: &Scoring, hand_chips: &mut f64, mult: &mut f64) {
        for si in 0..sc.n_scoring as usize {
            let pi = sc.scoring[si] as usize;
            if self.play[pi].debuff {
                self.blind.triggered = true;
                self.push(Ev::Wait);
                self.status_text(0.65);
                continue;
            }
            let mut reps = 1u32;
            if self.play[pi].seal == Some(Seal::Red) {
                reps += 1;
            }
            for j in 0..self.jokers.len() {
                if let Some(e) = self.calc_joker(j, &Ctx::Repetition { area: CardArea::Play, other: pi, effects: false }, Some(sc)) {
                    reps += e.repetitions;
                }
            }
            for r in 0..reps {
                if r > 0 {
                    self.status_text(0.75);
                }
                let own = self.eval_play_card(pi);
                let mut jeffs: Vec<Effect> = Vec::new();
                for k in 0..self.jokers.len() {
                    if let Some(e) = self.calc_joker(k, &Ctx::Individual { area: CardArea::Play, other: pi }, Some(sc)) {
                        jeffs.push(e);
                    }
                }
                self.play[pi].lucky_trigger = false;
                // effects[1]: the card itself
                if let Some(c) = own.chips {
                    *hand_chips += c;
                    self.hand_text(0.0);
                    self.status_text(0.6);
                }
                if let Some(m) = own.mult {
                    *mult += m;
                    self.hand_text(0.0);
                    self.status_text(0.65);
                }
                if let Some(d) = own.p_dollars {
                    self.push(Ev::EaseDollars(d));
                    self.status_text(0.65);
                }
                if let Some(x) = own.x_mult {
                    *mult *= x;
                    self.hand_text(0.0);
                    self.status_text(0.65);
                }
                if own.has_edition {
                    *hand_chips += own.ed_chips.unwrap_or(0.0);
                    *mult += own.ed_mult.unwrap_or(0.0);
                    *mult *= own.ed_xmult.unwrap_or(1.0);
                    self.hand_text(0.0);
                    self.status_text(0.75);
                }
                for e in &jeffs {
                    self.apply_play_effect(hand_chips, mult, e);
                }
            }
        }
    }

    /// The cards held in hand (with repetitions): Steel, Baron, Shoot the Moon, ...
    fn score_held_cards(&mut self, sc: &Scoring, mult: &mut f64) {
        self.delay(0.3);
        let nh = self.hand.len();
        for hi in 0..nh {
            if hi >= self.hand.len() {
                break;
            }
            let mut reps = 1usize;
            let mut j = 0usize;
            while j < reps {
                if j > 0 {
                    self.status_text(0.75);
                }
                let c = &self.hand[hi];
                let own_x = if !c.debuff && c.enhancement.h_x_mult() > 0.0 { Some(c.enhancement.h_x_mult()) } else { None };
                let mut jeffs: Vec<Effect> = Vec::new();
                for k in 0..self.jokers.len() {
                    if let Some(e) = self.calc_joker(k, &Ctx::Individual { area: CardArea::Hand, other: hi }, Some(sc)) {
                        jeffs.push(e);
                    }
                }
                if j == 0 {
                    let nonempty = own_x.is_some() || !jeffs.is_empty();
                    let c = &self.hand[hi];
                    if !c.debuff && c.seal == Some(Seal::Red) && nonempty {
                        reps += 1;
                    }
                    for k in 0..self.jokers.len() {
                        if let Some(e) = self.calc_joker(k, &Ctx::Repetition { area: CardArea::Hand, other: hi, effects: nonempty }, Some(sc)) {
                            reps += e.repetitions as usize;
                        }
                    }
                }
                if let Some(x) = own_x {
                    *mult *= x;
                    self.hand_text(0.0);
                    self.status_text(0.65);
                }
                for e in &jeffs {
                    self.push(Ev::Wait);
                    if let Some(d) = e.dollars {
                        self.push(Ev::EaseDollars(d));
                        self.status_text(0.65);
                    }
                    if let Some(h) = e.h_mult {
                        *mult += h;
                        self.hand_text(0.0);
                        self.status_text(0.65);
                    }
                    if let Some(x) = e.x_mult {
                        *mult *= x;
                        self.hand_text(0.0);
                        self.status_text(0.65);
                    }
                    if e.extra {
                        // `message` (debuffed)
                        self.hand_text(0.0);
                        self.status_text(e.delay.unwrap_or(0.75));
                    }
                }
                j += 1;
            }
        }
    }

    /// The joker (and consumable) area: editions, each joker's main effect
    /// and the jokers reacting to it (Baseball Card).
    fn score_joker_area(&mut self, sc: &Scoring, hand_chips: &mut f64, mult: &mut f64) {
        let text = sc.name;
        let nj = self.jokers.len();
        let nc = self.consumables.len();
        for i in 0..nj + nc {
            let is_cons = i >= nj;
            let (ed_chips, ed_mult, ed_x) = {
                let c = if is_cons { self.consumables.get(i - nj) } else { self.jokers.get(i) };
                let Some(c) = c else { continue };
                if c.debuff {
                    (None, None, None)
                } else {
                    match c.edition {
                        Edition::Foil => (Some(50.0), None, None),
                        Edition::Holo => (None, Some(10.0), None),
                        Edition::Polychrome => (None, None, Some(1.5)),
                        _ => (None, None, None),
                    }
                }
            };
            if let Some(c) = ed_chips {
                *hand_chips += c;
                self.hand_text(0.0);
                self.status_text(0.75);
            }
            if let Some(m) = ed_mult {
                *mult += m;
                self.hand_text(0.0);
                self.status_text(0.75);
            }
            let eff = if is_cons {
                let c = &self.consumables[i - nj];
                match c.kind {
                    Kind::Planet(p) if self.voucher_used(Voucher::Observatory) && p.hand() == text => {
                        Some(Effect { xmult_mod: Some(1.5), ..Effect::default() })
                    }
                    _ => None,
                }
            } else {
                self.calc_joker(i, &Ctx::Main, Some(sc))
            };
            if let Some(e) = eff {
                if let Some(m) = e.mult_mod {
                    *mult += m;
                }
                if let Some(c) = e.chip_mod {
                    *hand_chips += c;
                }
                if let Some(x) = e.xmult_mod {
                    *mult *= x;
                }
                self.hand_text(0.0);
                self.effect_status(&e);
            }
            for v in 0..self.jokers.len() {
                let ctx = Ctx::OtherJoker { other: (!is_cons).then_some(i) };
                if let Some(e) = self.calc_joker(v, &ctx, Some(sc)) {
                    let mut any = false;
                    if let Some(m) = e.mult_mod {
                        *mult += m;
                        any = true;
                    }
                    if let Some(c) = e.chip_mod {
                        *hand_chips += c;
                        any = true;
                    }
                    if let Some(x) = e.xmult_mod {
                        *mult *= x;
                        any = true;
                    }
                    if any {
                        self.hand_text(0.0);
                        self.effect_status(&e);
                    }
                }
            }
            if let Some(x) = ed_x {
                *mult *= x;
                self.hand_text(0.0);
                self.status_text(0.75);
            }
        }
    }

    /// Scored cards destroyed by jokers (Sixth Sense) or breaking glass.
    fn destroy_scored_cards(&mut self, sc: &Scoring, n_play: usize) {
        let mut destroyed_ids: Vec<u32> = Vec::new();
        let mut faces = 0;
        let mut glass = 0;
        let par = self.pareidolia();
        for si in 0..sc.n_scoring as usize {
            let pi = sc.scoring[si] as usize;
            let mut destroyed = false;
            self.highlight_ev();
            for j in 0..self.jokers.len() {
                if let Some(e) = self.calc_joker(j, &Ctx::DestroyingCard { play_idx: pi, full_len: n_play }, Some(sc)) && e.remove {
                    destroyed = true;
                    break;
                }
            }
            let c = &self.play[pi];
            if c.enhancement == Enhancement::Glass && !c.debuff {
                let k = Key::from("glass");
                if self.rng.random(&k) < self.probabilities / 4.0 {
                    destroyed = true;
                }
            }
            if destroyed {
                let c = &mut self.play[pi];
                if c.enhancement == Enhancement::Glass {
                    c.shattered = true;
                    glass += 1;
                } else {
                    c.destroyed = true;
                }
                if c.is_face(false, par) {
                    faces += 1;
                }
                destroyed_ids.push(c.id);
            }
        }
        for j in 0..self.jokers.len() {
            self.calc_joker(j, &Ctx::RemovePlayingCards { faces, glass }, Some(sc));
        }
        for id in destroyed_ids {
            self.push(Ev::DissolvePlaying { id });
        }
    }
}
