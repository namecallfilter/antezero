//! Booster packs: opening (`Card:open`), contents, picking and skipping
//! (`G.FUNCS.skip_booster` / `end_consumeable`).

use crate::card::{Card, Kind};
use crate::data::*;
use crate::events::{Ev, St};
use crate::game::{Area, CardSet, Game, PackState};
use crate::jokers::Ctx;
use crate::rng::Key;

impl Game {
    // -----------------------------------------------------------------
    // Booster packs
    // -----------------------------------------------------------------

    /// `Card:open()`.
    pub(crate) fn open_booster(&mut self, card: Card) {
        let Kind::Booster(b) = card.kind else { return };
        let d = b.def();
        self.state_complete = false;
        self.state = match d.kind {
            PackKind::Arcana => St::TarotPack,
            PackKind::Celestial => St::PlanetPack,
            PackKind::Spectral => St::SpectralPack,
            PackKind::Standard => St::StandardPack,
            PackKind::Buffoon => St::BuffoonPack,
        };
        self.pack_size = d.extra;
        self.pack_choices = d.choose;
        self.pack = Some(PackState { kind: d.kind });
        if card.cost > 0.0 {
            self.push_after(0.2, Ev::Wait);
            self.push(Ev::EaseDollars(-card.cost));
        } else {
            self.delay(0.2);
        }
        self.consumed.push(card.clone());
        self.push_after(0.4, Ev::OpenPack { card: card.id });
    }

    pub(crate) fn open_pack_run(&mut self, booster_id: u32) {
        self.explode(booster_id);
        // Contents are created by a timed non-blocking, non-blockable event.
        let t = 1.3 * crate::events::GAMESPEED.sqrt();
        self.push_ev(Ev::PackCreate, crate::events::Trig::After(t), false, false);
        for j in 0..self.jokers.len() {
            self.calc_joker(j, &Ctx::OpenBooster, None);
        }
    }

    pub(crate) fn pack_create(&mut self) {
        let Some(p) = self.pack else { return };
        let size = self.pack_size as usize;
        for i in 0..size {
            let c = match p.kind {
                PackKind::Arcana => {
                    let omen = self.voucher_used(Voucher::OmenGlobe) && {
                        let k = Key::from("omen_globe");
                        self.rng.random(&k) > 0.8
                    };
                    if omen {
                        self.create_card(CardSet::Spectral, Area::PackCards, false, None, true, None, "ar2")
                    } else {
                        self.create_card(CardSet::Tarot, Area::PackCards, false, None, true, None, "ar1")
                    }
                }
                PackKind::Celestial => {
                    let mut forced = None;
                    if self.voucher_used(Voucher::Telescope) && i == 0 {
                        let mut best: Option<HandType> = None;
                        let mut tally = 0.0;
                        for h in HandType::ALL {
                            let hl = &self.hands[h as usize];
                            if hl.visible && hl.played > tally {
                                best = Some(h);
                                tally = hl.played;
                            }
                        }
                        forced = best.map(|h| Kind::Planet(Planet::for_hand(h)));
                    }
                    self.create_card(CardSet::Planet, Area::PackCards, false, None, true, forced, "pl1")
                }
                PackKind::Spectral => self.create_card(CardSet::Spectral, Area::PackCards, false, None, true, None, "spe"),
                PackKind::Standard => {
                    let k = self.key_ante("stdset");
                    let set = if self.rng.random(&k) > 0.6 { CardSet::Enhanced } else { CardSet::Base };
                    let mut c = self.create_card(set, Area::PackCards, false, None, true, None, "sta");
                    let k = self.key_ante("standard_edition");
                    let e = self.poll_edition(&k, 2.0, true, false);
                    self.set_edition(&mut c, e, None);
                    self.edition_fx(e, false);
                    let k = self.key_ante("stdseal");
                    if self.rng.random(&k) > 1.0 - 0.02 * 10.0 {
                        let k = self.key_ante("stdsealtype");
                        let t = self.rng.random(&k);
                        c.seal = Some(if t > 0.75 {
                            Seal::Red
                        } else if t > 0.5 {
                            Seal::Blue
                        } else if t > 0.25 {
                            Seal::Gold
                        } else {
                            Seal::Purple
                        });
                        self.set_cost(&mut c, None);
                    }
                    c
                }
                PackKind::Buffoon => self.create_card(CardSet::Joker, Area::PackCards, false, None, true, None, "buf"),
            };
            self.materialize(1.5 * crate::events::GAMESPEED);
            self.pack_cards.push(c);
        }
    }

    /// Opens a pack granted by a tag (cost 0, `from_tag`).
    pub(crate) fn open_tag_pack(&mut self, b: Booster) {
        let mut c = self.make_card(Kind::Booster(b));
        c.cost = 0.0;
        self.use_card(c, None, &[]);
        self.materialize(1.0);
    }

    pub(crate) fn update_pack(&mut self) {
        if !self.state_complete {
            self.state_complete = true;
            self.push(Ev::PackStateUi);
        }
    }

    /// `G.FUNCS.skip_booster`.
    pub(crate) fn do_skip_pack(&mut self) {
        for j in 0..self.jokers.len() {
            self.calc_joker(j, &Ctx::SkippingBooster, None);
        }
        self.end_consumeable(1.0);
    }

    /// `G.FUNCS.end_consumeable(e, delayfac)`.
    pub(crate) fn end_consumeable(&mut self, df: f64) {
        // The pack UI (and with it the unchosen cards) is removed by a
        // timed non-blocking event.
        self.push_ev(Ev::PackRemove, crate::events::Trig::After(0.2 * df), false, false);
        self.delay(0.2 * df);
        self.push_after(0.2 * df, Ev::EndConsumeable { df });
    }
}
