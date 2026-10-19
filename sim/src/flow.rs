//! Round flow: blind select, new round, drawing, play / discard and the
//! per-state updates (state_events.lua, button_callbacks.lua, game.lua
//! `update_*`). The end of the round is in `round_end`.

use crate::card::{Card, Kind};
use crate::data::*;
use crate::events::{Ev, Pile, St};
use crate::game::{Area, BlindSlot, BlindStatus, CardSet, Game, Phase};
use crate::hand_eval::{self, EvalFlags};
use crate::jokers::Ctx;
use crate::rng::{Key, shuffle};

impl Game {
    // -----------------------------------------------------------------
    // Piles
    // -----------------------------------------------------------------

    pub(crate) fn pile(&mut self, p: Pile) -> &mut Vec<Card> {
        match p {
            Pile::Deck => &mut self.deck,
            Pile::Hand => &mut self.hand,
            Pile::Play => &mut self.play,
            Pile::Discard => &mut self.discard,
        }
    }

    /// `CardArea:shuffle(seed)`: sort by `sort_id`, then Fisher–Yates.
    pub(crate) fn shuffle_deck(&mut self, key: &Key) {
        self.deck.sort_unstable_by_key(|c| c.id);
        let seed = self.rng.seed_for(key);
        shuffle(&mut self.deck, seed);
    }

    /// `G.hand:sort()` with the area's current sort mode.
    pub(crate) fn sort_hand(&mut self) {
        let by_suit = self.hand_sort_by_suit;
        self.hand.sort_by(|a, b| {
            let (x, y) = (a.nominal(by_suit), b.nominal(by_suit));
            y.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// `CardArea:emplace` for piles (deck inserts at the front).
    fn emplace(&mut self, to: Pile, mut c: Card, stay_flipped: bool) {
        match to {
            Pile::Deck => {
                c.face_down = true;
                self.deck.insert(0, c);
            }
            Pile::Discard => {
                c.face_down = true;
                self.discard.push(c);
            }
            Pile::Hand | Pile::Play => {
                if c.face_down && !stay_flipped {
                    c.face_down = false;
                    c.ab.wheel_flipped = false;
                }
                if to == Pile::Hand && stay_flipped {
                    c.ab.wheel_flipped = true;
                    c.face_down = true;
                }
                if to == Pile::Hand {
                    self.hand.push(c);
                } else {
                    self.play.push(c);
                }
            }
        }
    }

    /// The body of `draw_card`'s event.
    pub(crate) fn draw_card(&mut self, from: Pile, to: Pile, card: Option<u32>, sort: bool, discarded_only: bool) {
        let taken = match card {
            Some(id) => {
                let src = self.pile(from);
                src.iter().position(|c| c.id == id).map(|i| src.remove(i))
            }
            None => {
                // CardArea:draw_card_from → area:remove_card(nil, discarded_only)
                let src_is_stack = matches!(from, Pile::Deck | Pile::Discard);
                let src = self.pile(from);
                if discarded_only {
                    let pick = if src_is_stack {
                        src.iter().rposition(|c| c.ab.discarded)
                    } else {
                        src.iter().position(|c| c.ab.discarded)
                    };
                    pick.map(|i| src.remove(i))
                } else if src.is_empty() {
                    None
                } else if src_is_stack {
                    src.pop()
                } else {
                    Some(src.remove(0))
                }
            }
        };
        let Some(c) = taken else { return };
        let stay = self.blind_stay_flipped(to, &c);
        crate::game::trace!(self, "draw {} {}", to.name(), c.id);
        self.emplace(to, c, stay);
        if sort && to == Pile::Hand {
            self.sort_hand();
        }
    }

    /// `G.FUNCS.draw_from_deck_to_hand(e)`; returns true on the game-over path.
    pub(crate) fn draw_from_deck_to_hand(&mut self, e: Option<usize>) -> bool {
        if !matches!(self.state, St::TarotPack | St::SpectralPack) && self.hand_limit.limit <= 0.0 && self.hand.is_empty() {
            self.state = St::GameOver;
            self.state_complete = false;
            return true;
        }
        let mut hand_space: i64 = match e {
            Some(n) => n as i64,
            None => (self.deck.len() as i64).min(self.hand_limit.limit as i64 - self.hand.len() as i64),
        };
        if self.blind.active(BlindKind::Serpent) && (self.hands_played_round > 0.0 || self.discards_used > 0.0) {
            hand_space = (self.deck.len() as i64).min(3);
        }
        self.delay(0.3);
        for _ in 0..hand_space.max(0) {
            self.draw_ev(Pile::Deck, Pile::Hand, None, true, false, 0.1);
        }
        false
    }

    /// Finds and removes a card anywhere (`Card:remove`).
    pub(crate) fn remove_card_by_id(&mut self, id: u32) {
        macro_rules! try_area {
            ($v:ident, $owned:expr) => {
                if let Some(i) = self.$v.iter().position(|c| c.id == id) {
                    let c = self.$v.remove(i);
                    if $owned(&c) {
                        self.remove_owned(c);
                    } else {
                        self.destroy_card(c);
                    }
                    return;
                }
            };
        }
        let yes = |_: &Card| true;
        let no = |_: &Card| false;
        let added = |c: &Card| c.added_to_deck;
        try_area!(jokers, added);
        try_area!(consumables, added);
        try_area!(hand, yes);
        try_area!(play, yes);
        try_area!(deck, yes);
        try_area!(discard, yes);
        try_area!(shop_jokers, no);
        try_area!(shop_vouchers, no);
        try_area!(shop_boosters, no);
        try_area!(pack_cards, no);
        try_area!(consumed, |c: &Card| c.added_to_deck);
    }

    /// `copy_card(other, nil, nil, playing_card, strip_edition)`.
    pub(crate) fn copy_card(&mut self, other: &Card, strip_edition: bool) -> Card {
        let mut c = self.new_card(other.kind);
        let id = c.id;
        c = other.clone();
        c.id = id;
        // set_ability(center) marks the center as used; ability is then copied.
        self.mark_used(&c);
        c.added_to_deck = false;
        c.queue_negative_removal = false;
        c.getting_sliced = false;
        c.shattered = false;
        c.destroyed = false;
        c.lucky_trigger = false;
        c.vampired = false;
        c.temp_edition = false;
        // A new Card starts face up; set_base resets times_played.
        c.face_down = false;
        c.times_played = 0;
        // set_ability(center) rolls a To Do List hand before the copied
        // ability overwrites it: the stream still advances.
        if c.is_joker(Joker::TodoList) {
            let (hands, n) = self.visible_hands_pairs(None);
            let k = Key::from("to_do");
            let _ = hands[self.rng.element(&k, n)];
        }
        if c.is_playing() {
            // set_base with a fresh card: suit_nominal_original from the current suit.
            c.orig_suit = Some(c.suit);
        }
        c.edition = Edition::None;
        if !strip_edition {
            c.edition = other.edition;
        }
        self.set_cost(&mut c, None);
        c.debuff = other.debuff;
        c
    }

    /// `Blind:set_blind(nil, true)`: re-evaluate playing-card debuffs.
    pub(crate) fn refresh_playing_debuffs(&mut self) {
        for p in [Pile::Deck, Pile::Hand, Pile::Play, Pile::Discard] {
            let mut v = std::mem::take(self.pile(p));
            for c in v.iter_mut() {
                self.debuff_card(c);
            }
            *self.pile(p) = v;
        }
    }

    // -----------------------------------------------------------------
    // State handlers (Game:update_*)
    // -----------------------------------------------------------------

    /// `Game:update_selecting_hand`. With hand, deck and play all empty the
    /// Lua calls `end_round()` on every frame until its first event resolves,
    /// queueing several end-of-round passes; the simulator queues one. The
    /// extra passes only matter after the run is lost (or saved by Mr. Bones).
    pub(crate) fn update_selecting_hand(&mut self) {
        let out = self.hand.is_empty() && self.deck.is_empty();
        if out && self.play.is_empty() && !self.end_round_pending {
            self.end_round_pending = true;
            self.push_after(0.2, Ev::EndRound);
        }
        if !self.state_complete {
            self.state_complete = true;
            if out && !self.end_round_pending {
                self.end_round_pending = true;
                self.push_after(0.2, Ev::EndRound);
            }
        }
    }

    pub(crate) fn update_hand_played(&mut self) {
        if !self.state_complete {
            self.state_complete = true;
            self.push(Ev::HandPlayedCheck);
        }
    }

    pub(crate) fn update_draw_to_hand(&mut self) {
        if !self.state_complete {
            self.state_complete = true;
            for i in 0..self.tags.len() {
                self.apply_tag(i, crate::tags::TagCtx::RoundStartBonus);
            }
            self.push(Ev::DrawToHand);
        }
    }

    pub(crate) fn update_new_round(&mut self) {
        if !self.state_complete {
            self.state_complete = true;
            self.push_after(0.2, Ev::EndRound);
        }
    }

    pub(crate) fn update_blind_select(&mut self) {
        if !self.state_complete {
            self.state_complete = true;
            // save_run
            self.push(Ev::Wait);
            self.push(Ev::BlindSelectInit);
        }
    }

    pub(crate) fn update_round_eval(&mut self) {
        if !self.state_complete {
            self.state_complete = true;
            self.push(Ev::RoundEvalInit);
        }
    }

    /// Derives the agent-facing phase from `G.STATE` once the queue is empty.
    pub(crate) fn sync_phase(&mut self) {
        self.phase = match self.state {
            St::SelectingHand => Phase::Round,
            St::Shop => Phase::Shop,
            St::BlindSelect => Phase::BlindSelect,
            St::RoundEval => Phase::RoundEval,
            St::GameOver => Phase::GameOver,
            s if s.is_pack() => Phase::Pack,
            other => panic!("queue drained in transient state {:?}", other),
        };
    }

    // -----------------------------------------------------------------
    // Flow events
    // -----------------------------------------------------------------

    /// Runs a round-flow event (routed by `run_event`).
    pub(crate) fn run_flow_event(&mut self, e: Ev) {
        match e {
            Ev::BlindSelectInit => self.push(Ev::BlindSelectUi),
            Ev::BlindSelectUi => self.blind_select_ui(),
            Ev::SelectBlind { blind } => {
                self.push(Ev::EaseRound);
                self.selected_blind = Some(blind);
                if let Some(s) = self.blind_on_deck {
                    self.blind_states[s as usize] = BlindStatus::Current;
                }
                self.delay(0.2);
            }
            // new_round()
            Ev::NewRound => {
                self.delay(0.4);
                self.push(Ev::NewRoundBody);
            }
            Ev::NewRoundBody => self.new_round(),
            Ev::BlindSetHud => self.push_ev(Ev::Wait, crate::events::Trig::After(0.15), true, false),
            Ev::NewRoundDraw => {
                self.state = St::DrawToHand;
                let k = self.key_ante("nr");
                self.shuffle_deck(&k);
                self.state_complete = false;
            }
            Ev::DrawToHand => {
                if self.draw_from_deck_to_hand(None) {
                    return;
                }
                if self.hands_played_round == 0.0 && self.discards_used == 0.0 && self.facing_blind {
                    for i in 0..self.jokers.len() {
                        self.calc_joker(i, &Ctx::FirstHandDrawn, None);
                    }
                }
                self.push(Ev::EnterSelecting);
            }
            Ev::EnterSelecting => {
                self.state = St::SelectingHand;
                self.state_complete = false;
                self.blind_drawn_to_hand();
            }
            Ev::PlayStart => {
                self.state = St::HandPlayed;
                self.state_complete = true;
            }
            Ev::PlayChain => {
                self.push(Ev::EvaluatePlay);
                self.push_after(0.1, Ev::AfterPlay);
                self.push(Ev::StateCompleteFalse);
            }
            Ev::EvaluatePlay => self.evaluate_play(),
            Ev::AfterPlay => {
                let ids: Vec<u32> = self.play.iter().filter(|c| !c.shattered && !c.destroyed).map(|c| c.id).collect();
                for id in ids {
                    self.draw_ev(Pile::Play, Pile::Discard, Some(id), false, false, 0.1);
                }
                self.hands_played += 1.0;
                self.hands_played_round += 1.0;
            }
            Ev::StateCompleteFalse => self.state_complete = false,
            Ev::HandPlayedCheck => {
                self.state = if self.chips - self.blind.chips >= 0.0 || self.hands_left < 1.0 { St::NewRound } else { St::DrawToHand };
                self.state_complete = false;
            }
            Ev::HookDiscard => self.hook_discard(),
            Ev::ToothOuter => {
                for _ in 0..self.play.len() {
                    self.push(Ev::Wait);
                    self.push(Ev::EaseDollars(-1.0));
                    self.delay(0.23);
                }
            }
            Ev::EndRound => self.end_round(),
            Ev::EndRoundFinal => self.end_round_final(),
            Ev::RoundEvalInit => {
                self.facing_blind = false;
                // Polls until the round-eval UI has slid in.
                self.push_ev(Ev::RoundEvalRun, crate::events::Trig::Frames(21), true, true);
            }
            Ev::RoundEvalRun => {
                self.delay(0.1);
                self.evaluate_round();
            }
            Ev::CashOutFinish => {
                self.discards_left = 0f64.max(self.rr_discards + self.round_bonus_discards);
                self.hands_left = 1f64.max(self.rr_hands + self.round_bonus_next_hands);
                self.state = St::Shop;
                self.shop_free = false;
                self.shop_d6ed = false;
                self.state_complete = false;
            }
            Ev::BlindDisableCheck => {
                if self.blind.boss && self.chips - self.blind.chips >= 0.0 {
                    self.state = St::NewRound;
                    self.state_complete = false;
                }
            }
            Ev::AcornOuter => {
                self.push(Ev::AcornShuffle);
                self.delay(0.15);
                self.push(Ev::AcornShuffle);
                self.delay(0.15);
                self.push(Ev::AcornShuffle);
                self.delay(0.5);
            }
            Ev::AcornShuffle => {
                self.jokers.sort_unstable_by_key(|c| c.id);
                let k = Key::from("aajk");
                let seed = self.rng.seed_for(&k);
                shuffle(&mut self.jokers, seed);
            }
            Ev::SkipBlindRun => {
                self.delay(0.3);
                for i in 0..self.jokers.len() {
                    self.calc_joker(i, &Ctx::SkipBlind, None);
                }
                for i in 0..self.tags.len() {
                    self.apply_tag(i, crate::tags::TagCtx::Immediate);
                }
                self.apply_new_blind_choice_tags();
                // The blind-choice UI notices the new on-deck blind this frame
                // (blind_choice_handler) and queues a focus event.
                self.push(Ev::Wait);
            }
            Ev::RerollBossRun => {
                self.boss_choice = self.get_new_boss();
                if let Some(s) = self.blind_on_deck {
                    self.blind_states[s as usize] = BlindStatus::Select;
                }
                self.ensure_orbital(BlindSlot::Boss);
                self.apply_new_blind_choice_tags();
            }
            Ev::ChicotOuter => {
                self.push(Ev::ChicotDisable);
                self.status_extra();
            }
            Ev::ChicotDisable => {
                self.disable_blind();
                self.delay(0.4);
            }
            Ev::Burglar => {
                let d = self.discards_left;
                self.push(Ev::EaseDiscard(-d));
                self.push(Ev::EaseHands(3.0));
                self.status_extra();
            }
            Ev::RiffRaff { n } => {
                for _ in 0..n {
                    let mut c = self.create_card(CardSet::Joker, Area::Jokers, false, Some(0.0), false, None, "rif");
                    self.add_to_deck(&mut c, false);
                    self.jokers.push(c);
                    self.materialize(1.0);
                    self.joker_buffer = 0.0;
                }
            }
            Ev::CartomancerOuter => {
                self.push(Ev::Create { set: CardSet::Tarot, area: Area::Consumables, forced: None, append: "car", rarity: None, legendary: false });
                self.status_extra();
            }
            Ev::DaggerSlice { dagger, victim } => {
                self.joker_buffer = 0.0;
                let sell = self.jokers.iter().find(|c| c.id == victim).map(|c| c.sell_cost).unwrap_or(0.0);
                if let Some(d) = self.jokers.iter_mut().find(|c| c.id == dagger) {
                    d.ab.mult += sell * 2.0;
                }
                self.start_dissolve_f(victim, 1.6);
            }
            Ev::MarbleStone => {
                let k = Key::from("marb_fr");
                let i = self.rng.element(&k, 52);
                let (suit, rank) = P_CARDS_SORTED[i];
                let c = self.make_playing_card(suit, rank, Enhancement::Stone);
                self.materialize(1.0);
                self.play.push(c);
            }
            Ev::CertificateCard => {
                let k = Key::from("cert_fr");
                let i = self.rng.element(&k, 52);
                let (suit, rank) = P_CARDS_SORTED[i];
                let mut c = self.make_playing_card(suit, rank, Enhancement::Base);
                let k = Key::from("certsl");
                let st = self.rng.random(&k);
                c.seal = Some(if st > 0.75 {
                    Seal::Red
                } else if st > 0.5 {
                    Seal::Blue
                } else if st > 0.25 {
                    Seal::Gold
                } else {
                    Seal::Purple
                });
                self.set_cost(&mut c, None);
                self.materialize(1.0);
                self.debuff_card(&mut c);
                self.hand.push(c);
                self.sort_hand();
            }
            Ev::GlassJokerOuter { id, n } => self.push(Ev::GlassJokerInner { id, n }),
            Ev::GlassJokerInner { id, n } => {
                if let Some(c) = self.jokers.iter_mut().find(|c| c.id == id) {
                    c.ab.x_mult += 0.75 * n as f64;
                }
            }
            Ev::PerkeoCopy => self.perkeo_copy(),
            other => unreachable!("not a flow event: {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // Blind select
    // -----------------------------------------------------------------

    fn orbital_slot(&mut self) -> &mut [Option<HandType>; 3] {
        let ante = self.ante;
        if let Some(i) = self.orbital_choices.iter().position(|(a, _)| *a == ante) {
            &mut self.orbital_choices[i].1
        } else {
            self.orbital_choices.push((ante, [None; 3]));
            let n = self.orbital_choices.len() - 1;
            &mut self.orbital_choices[n].1
        }
    }

    pub(crate) fn orbital_for(&self, slot: BlindSlot) -> Option<HandType> {
        self.orbital_choices.iter().find(|(a, _)| *a == self.ante).and_then(|(_, v)| v[slot as usize])
    }

    pub(crate) fn ensure_orbital(&mut self, slot: BlindSlot) {
        if self.orbital_slot()[slot as usize].is_none() {
            let (hands, n) = self.visible_hands_pairs(None);
            let k = Key::from("orbital");
            let i = self.rng.element(&k, n);
            self.orbital_slot()[slot as usize] = Some(hands[i]);
        }
    }

    /// `create_UIBox_blind_select` + the tag passes of `update_blind_select`.
    fn blind_select_ui(&mut self) {
        let s = &self.blind_states;
        let on_deck = if !matches!(s[0], BlindStatus::Defeated | BlindStatus::Skipped) {
            BlindSlot::Small
        } else if !matches!(s[1], BlindStatus::Defeated | BlindStatus::Skipped) {
            BlindSlot::Big
        } else {
            BlindSlot::Boss
        };
        self.blind_on_deck = Some(on_deck);
        self.blind_states[on_deck as usize] = BlindStatus::Select;
        for slot in [BlindSlot::Small, BlindSlot::Big, BlindSlot::Boss] {
            self.ensure_orbital(slot);
        }
        if self.blind_ante == i32::MIN {
            self.blind_ante = self.ante;
        }
        for i in 0..self.tags.len() {
            self.apply_tag(i, crate::tags::TagCtx::Immediate);
        }
        self.apply_new_blind_choice_tags();
    }

    pub(crate) fn apply_new_blind_choice_tags(&mut self) {
        for i in 0..self.tags.len() {
            if self.apply_tag(i, crate::tags::TagCtx::NewBlindChoice) {
                break;
            }
        }
    }

    /// The blind the "select" button refers to.
    pub fn blind_on_deck_kind(&self) -> Option<BlindKind> {
        self.blind_on_deck.map(|s| match s {
            BlindSlot::Small => BlindKind::Small,
            BlindSlot::Big => BlindKind::Big,
            BlindSlot::Boss => self.boss_choice,
        })
    }

    pub(crate) fn do_select_blind(&mut self) {
        let Some(b) = self.blind_on_deck_kind() else { return };
        self.facing_blind = true;
        self.wait(crate::events::Trig::Before(0.2));
        self.push(Ev::SelectBlind { blind: b });
        self.push(Ev::NewRound);
    }

    pub(crate) fn do_skip_blind(&mut self) {
        let Some(slot) = self.blind_on_deck else { return };
        self.skips += 1.0;
        if slot == BlindSlot::Boss {
            return;
        }
        if let Some(kind) = self.blind_tags[slot as usize] {
            self.blinds_cleared += 1;
            let orbital = if kind == TagKind::Orbital { self.orbital_for(slot) } else { None };
            self.add_tag(kind, orbital);
            let to = if slot == BlindSlot::Small { BlindSlot::Big } else { BlindSlot::Boss };
            self.blind_states[slot as usize] = BlindStatus::Skipped;
            self.blind_states[to as usize] = BlindStatus::Select;
            self.blind_on_deck = Some(to);
            self.push(Ev::SkipBlindRun);
        }
    }

    pub(crate) fn do_reroll_boss(&mut self, from_tag: bool) {
        self.boss_rerolled = true;
        if !from_tag {
            self.push(Ev::EaseDollars(-10.0));
        }
        self.push(Ev::Wait);
        self.push_after(0.3, Ev::RerollBossRun);
    }

    // -----------------------------------------------------------------
    // Round start
    // -----------------------------------------------------------------

    fn new_round(&mut self) {
        self.discards_left = 0f64.max(self.rr_discards + self.round_bonus_discards);
        self.hands_left = 1f64.max(self.rr_hands + self.round_bonus_next_hands);
        self.hands_played_round = 0.0;
        self.discards_used = 0.0;
        self.reroll_cost_increase = 0.0;
        self.used_packs = [None, None];
        for h in self.hands.iter_mut() {
            h.played_this_round = 0.0;
        }
        for c in self.playing_cards_mut() {
            c.ab.wheel_flipped = false;
        }
        self.free_rerolls = self.count_joker(Joker::Chaos) as f64;
        self.calculate_reroll_cost(true);
        self.round_bonus_next_hands = 0.0;
        self.round_bonus_discards = 0.0;
        let blind = self.selected_blind.expect("new_round without a selected blind");
        self.blind_states[blind.slot() as usize] = BlindStatus::Current;
        self.set_blind(Some(blind));
        let boss = blind.is_boss();
        for i in 0..self.jokers.len() {
            self.calc_joker(i, &Ctx::SettingBlind { boss }, None);
        }
        self.end_round_pending = false;
        self.delay(0.4);
        self.push(Ev::NewRoundDraw);
    }

    // -----------------------------------------------------------------
    // Play / discard
    // -----------------------------------------------------------------

    /// `G.FUNCS.play_cards_from_highlighted` with the given hand indices.
    pub(crate) fn do_play(&mut self, mut sel: Vec<usize>) {
        if !self.play.is_empty() {
            return;
        }
        self.blind.triggered = false;
        for c in self.playing_cards_mut() {
            c.ab.forced_selection = false;
        }
        sel.sort_unstable();
        self.hand_text_cleared = false;
        self.push(Ev::PlayStart);
        self.push(Ev::EaseHands(-1.0));
        self.delay(0.4);
        for &i in &sel {
            let c = &mut self.hand[i];
            c.times_played += 1;
            c.ab.played_this_ante = true;
            let id = c.id;
            self.draw_ev(Pile::Hand, Pile::Play, Some(id), false, false, 0.1);
        }
        if self.blind_press_play() {
            self.push(Ev::Wait);
            self.delay(0.4);
        }
        self.push(Ev::PlayChain);
    }

    fn hook_discard(&mut self) {
        // pseudorandom_element picks from the remaining cards sorted by sort_id.
        let mut cards: Vec<usize> = (0..self.hand.len()).collect();
        cards.sort_by_key(|&i| self.hand[i].id);
        let mut chosen = Vec::new();
        for _ in 0..cards.len().min(2) {
            let k = Key::from("hook");
            let pick = self.rng.element(&k, cards.len());
            chosen.push(cards.remove(pick));
        }
        if !chosen.is_empty() {
            self.do_discard(chosen, true);
        }
    }

    /// `G.FUNCS.discard_cards_from_highlighted(e, hook)`.
    pub(crate) fn do_discard(&mut self, mut sel: Vec<usize>, hook: bool) {
        for c in self.playing_cards_mut() {
            c.ab.forced_selection = false;
        }
        sel.sort_unstable();
        let count = sel.len().min(500usize.saturating_sub(self.play.len()));
        if count == 0 {
            return;
        }
        let sel = &sel[..count];
        // update_hand_text({immediate = true, delay = 0}, {handname = ''})
        self.push_ev(Ev::Wait, crate::events::Trig::Before(0.0), true, false);
        self.hand_text_cleared = true;
        // Burnt Joker evaluates the highlighted cards' poker hand (only it
        // reads the result, in the `pre_discard` pass below).
        self.burnt_hand = if self.jokers.iter().any(|c| c.kind == Kind::Joker(Joker::Burnt)) {
            let refs: Vec<&Card> = sel.iter().map(|&i| &self.hand[i]).collect();
            hand_eval::evaluate(&refs, &self.eval_flags()).top
        } else {
            None
        };
        for j in 0..self.jokers.len() {
            self.calc_joker(j, &Ctx::PreDiscard { hook }, None);
        }
        let par = self.pareidolia();
        let full_faces = sel.iter().filter(|&&i| self.hand[i].is_face(false, par)).count();
        let last = *sel.last().unwrap();
        let mut destroyed: Vec<usize> = Vec::new();
        for &i in sel {
            // Card:calculate_seal({discard = true})
            if !self.hand[i].debuff && self.hand[i].seal == Some(Seal::Purple)
                && self.consumable_room()
            {
                self.consumeable_buffer += 1.0;
                self.push_before(0.0, Ev::Create { set: CardSet::Tarot, area: Area::Consumables, forced: None, append: "8ba", rarity: None, legendary: false });
                self.status_extra();
            }
            let mut removed = false;
            for j in 0..self.jokers.len() {
                if let Some(e) = self.calc_joker(j, &Ctx::Discard { other: i, last, n_full: count, full_faces }, None) {
                    if e.remove {
                        removed = true;
                    }
                    self.effect_status(&e);
                }
            }
            if removed {
                destroyed.push(i);
                let c = &mut self.hand[i];
                let id = c.id;
                if c.enhancement == Enhancement::Glass {
                    c.shattered = true;
                    self.shatter(id);
                } else {
                    self.start_dissolve(id);
                }
            } else {
                self.hand[i].ab.discarded = true;
                let id = self.hand[i].id;
                self.draw_ev(Pile::Hand, Pile::Discard, Some(id), false, false, 0.1);
            }
        }
        if !destroyed.is_empty() {
            let faces = destroyed.iter().filter(|&&i| self.hand[i].is_face(false, par)).count();
            let glass = destroyed.iter().filter(|&&i| self.hand[i].shattered).count();
            for j in 0..self.jokers.len() {
                self.calc_joker(j, &Ctx::RemovePlayingCards { faces, glass }, None);
            }
        }
        if !hook {
            self.push(Ev::EaseDiscard(-1.0));
            self.discards_used += 1.0;
            self.state = St::DrawToHand;
            self.push(Ev::StateCompleteFalse);
        }
    }

    /// Jokers that change how poker hands are evaluated.
    pub fn eval_flags(&self) -> EvalFlags {
        EvalFlags {
            four_fingers: self.has_joker(Joker::FourFingers),
            shortcut: self.has_joker(Joker::Shortcut),
            smeared: self.smeared(),
        }
    }
}
