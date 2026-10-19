//! Joker contexts outside hand scoring: shop, packs, selling, blind setup,
//! discards and the end of the round.

use super::*;
use crate::rng::Key;

impl Game {
    pub(super) fn joker_lifecycle(&mut self, ji: usize, j: Joker, ctx: &Ctx, blueprinted: bool, actor: usize) -> Option<Effect> {
        use Joker as J;
        let ab = self.jokers[ji].ab;
        let prob = self.probabilities;
        match *ctx {
            Ctx::OpenBooster => {
                if j == J::Hallucination && self.consumable_room() {
                    let k = self.key_ante("halu");
                    if self.rng.random(&k) < prob / 2.0 {
                        self.consumeable_buffer += 1.0;
                        self.push_before(0.0, Ev::Create { set: CardSet::Tarot, area: Area::Consumables, forced: None, append: "hal", rarity: None, legendary: false });
                        self.status_extra();
                    }
                }
                None
            }
            Ctx::SellingSelf => {
                if j == J::Luchador && self.blind.kind.is_some() && !self.blind.disabled && self.blind.slot() == Some(crate::game::BlindSlot::Boss) {
                    self.status_extra();
                    self.disable_blind();
                }
                if j == J::DietCola {
                    self.push(Ev::AddTag { kind: TagKind::Double, orbital: None });
                }
                if j == J::Invisible && ab.invis_rounds >= 2.0 && !blueprinted {
                    let mut sorted: Vec<usize> = (0..self.jokers.len()).filter(|&i| i != ji).collect();
                    self.status_extra();
                    if !sorted.is_empty() && self.jokers.len() as f64 <= self.joker_limit.limit {
                        // pseudorandom_element(jokers) sorts by sort_id.
                        sorted.sort_by_key(|&i| self.jokers[i].id);
                        let k = Key::from("invisible");
                        let pick = sorted[self.rng.element(&k, sorted.len())];
                        let src = self.jokers[pick].clone();
                        let strip = src.edition == Edition::Negative;
                        let mut c = self.copy_card(&src, strip);
                        if c.is_joker(J::Invisible) {
                            c.ab.invis_rounds = 0.0;
                        }
                        self.add_to_deck(&mut c, false);
                        self.jokers.push(c);
                    }
                }
                None
            }
            Ctx::SellingCard => {
                if j == J::Campfire && !blueprinted {
                    self.jab(ji).x_mult += 0.25;
                    self.push(Ev::StatusLater(0.75));
                }
                None
            }
            Ctx::RerollShop => {
                if j == J::Flash && !blueprinted {
                    self.jab(ji).mult += 2.0;
                    self.push(Ev::StatusLater(0.75));
                }
                None
            }
            Ctx::EndingShop => {
                if j == J::Perkeo && !self.consumables.is_empty() {
                    self.push(Ev::PerkeoCopy);
                    self.status_extra();
                }
                None
            }
            Ctx::SkipBlind => {
                if j == J::Throwback && !blueprinted {
                    self.push(Ev::StatusLater(0.75));
                }
                None
            }
            Ctx::SkippingBooster => {
                if j == J::RedCard && !blueprinted {
                    self.jab(ji).mult += 3.0;
                    self.push(Ev::StatusLater(0.45));
                }
                None
            }
            Ctx::PlayingCardAdded { n } => {
                if self.jokers[ji].getting_sliced {
                    return None;
                }
                if j == J::Hologram && !blueprinted && n > 0 {
                    self.jab(ji).x_mult += n as f64 * 0.25;
                    self.status_extra();
                }
                None
            }
            Ctx::FirstHandDrawn => {
                if j == J::Certificate {
                    self.push(Ev::CertificateCard);
                    self.playing_card_joker_effects(1);
                }
                None
            }
            Ctx::SettingBlind { boss } => {
                if self.jokers[ji].getting_sliced {
                    return None;
                }
                if j == J::Chicot && !blueprinted && boss {
                    self.push(Ev::ChicotOuter);
                }
                if j == J::Madness && !blueprinted && !boss {
                    self.jab(ji).x_mult += 0.5;
                    let mut dest: Vec<usize> = (0..self.jokers.len())
                        .filter(|&i| i != ji && !self.jokers[i].ab.eternal && !self.jokers[i].getting_sliced)
                        .collect();
                    if !dest.is_empty() {
                        dest.sort_by_key(|&i| self.jokers[i].id);
                        let k = Key::from("madness");
                        let pick = dest[self.rng.element(&k, dest.len())];
                        if !self.jokers[actor].getting_sliced {
                            self.jokers[pick].getting_sliced = true;
                            let id = self.jokers[pick].id;
                            self.push(Ev::DissolveJoker { id, fac: 1.6 });
                        }
                    }
                    if !self.jokers[actor].getting_sliced {
                        self.status_extra();
                    }
                }
                if j == J::Burglar && !self.jokers[actor].getting_sliced {
                    self.push(Ev::Burglar);
                }
                if j == J::RiffRaff
                    && !self.jokers[actor].getting_sliced
                    && (self.jokers.len() as f64 + self.joker_buffer) < self.joker_limit.limit
                {
                    let n = 2f64.min(self.joker_limit.limit - (self.jokers.len() as f64 + self.joker_buffer));
                    self.joker_buffer += n;
                    self.push(Ev::RiffRaff { n: n as u32 });
                    self.status_extra();
                }
                if j == J::Cartomancer
                    && !self.jokers[actor].getting_sliced
                    && self.consumable_room()
                {
                    self.consumeable_buffer += 1.0;
                    self.push(Ev::CartomancerOuter);
                }
                if j == J::Ceremonial
                    && !blueprinted
                    && ji + 1 < self.jokers.len()
                    && !self.jokers[ji + 1].ab.eternal
                    && !self.jokers[ji + 1].getting_sliced
                {
                    self.jokers[ji + 1].getting_sliced = true;
                    self.joker_buffer -= 1.0;
                    let me = self.jokers[ji].id;
                    let victim = self.jokers[ji + 1].id;
                    self.push(Ev::DaggerSlice { dagger: me, victim });
                    self.status_extra();
                }
                if j == J::Marble && !self.jokers[actor].getting_sliced {
                    self.push(Ev::MarbleStone);
                    self.status_extra();
                    // G.deck.config.card_limit + 1 (no game effect here)
                    self.push(Ev::Wait);
                    self.draw_ev(crate::events::Pile::Play, crate::events::Pile::Deck, None, false, false, 0.1);
                    self.playing_card_joker_effects(1);
                }
                None
            }
            Ctx::RemovePlayingCards { faces, glass } => {
                if j == J::Caino && !blueprinted {
                    if faces > 0 {
                        self.jab(ji).caino_xmult += faces as f64 * 1.0;
                        self.push(Ev::StatusLater(0.75));
                    }
                    return None;
                }
                if j == J::Glass && !blueprinted {
                    if glass > 0 {
                        self.push(Ev::GlassJokerOuter { id: self.jokers[ji].id, n: glass as u32 });
                    }
                    return None;
                }
                None
            }
            Ctx::UsingConsumeable { kind, hanged_glass } => {
                if j == J::Glass && !blueprinted && kind == Kind::Tarot(Tarot::HangedMan) {
                    if hanged_glass > 0 {
                        self.jab(ji).x_mult += 0.75 * hanged_glass as f64;
                        self.push(Ev::StatusLater(0.75));
                    }
                    return None;
                }
                if j == J::FortuneTeller && !blueprinted && matches!(kind, Kind::Tarot(_)) {
                    self.push(Ev::StatusLater(0.75));
                }
                if j == J::Constellation && !blueprinted && matches!(kind, Kind::Planet(_)) {
                    self.jab(ji).x_mult += 0.1;
                    self.push(Ev::StatusLater(0.75));
                    return None;
                }
                None
            }
            Ctx::PreDiscard { hook } => {
                if j == J::Burnt && self.discards_used <= 0.0 && !hook && let Some(h) = self.burnt_hand {
                    self.status_extra();
                    self.hand_text(0.3);
                    self.level_up_hand_ev(h, 1.0, false);
                    self.hand_text(0.0);
                }
                None
            }
            Ctx::Discard { other, last, n_full, full_faces } => {
                let oc = &self.hand[other];
                if j == J::Ramen && !blueprinted {
                    if ab.x_mult - 0.01 <= 1.0 {
                        self.schedule_self_destruct(ji);
                        ret!(extra: true);
                    } else {
                        self.jab(ji).x_mult -= 0.01;
                        ret!(extra: true, delay: Some(0.2));
                    }
                }
                if j == J::Yorick && !blueprinted {
                    if ab.yorick_discards <= 1.0 {
                        self.jab(ji).yorick_discards = 23.0;
                        self.jab(ji).x_mult += 1.0;
                        ret!(extra: true, delay: Some(0.2));
                    } else {
                        self.jab(ji).yorick_discards -= 1.0;
                    }
                    return None;
                }
                if j == J::Trading && !blueprinted && self.discards_used <= 0.0 && n_full == 1 {
                    self.push(Ev::EaseDollars(3.0));
                    ret!(remove: true, delay: Some(0.45));
                }
                let smeared = self.smeared();
                if j == J::Castle && !oc.debuff && oc.is_suit(self.castle_suit, false, false, smeared) && !blueprinted {
                    self.jab(ji).chips += 3.0;
                    ret!(extra: true);
                }
                let oc = &self.hand[other];
                if j == J::Mail && !oc.debuff && oc.get_id() == Some(self.mail_rank.0) {
                    self.push(Ev::EaseDollars(5.0));
                    ret!(dollars: None);
                }
                let oc = &self.hand[other];
                if j == J::HitTheRoad && !oc.debuff && oc.id_is(JACK) && !blueprinted {
                    self.jab(ji).x_mult += 0.5;
                    ret!(extra: true, delay: Some(0.45));
                }
                if j == J::GreenJoker && !blueprinted && other == last {
                    let prev = ab.mult;
                    let m = (ab.mult - 1.0).max(0.0);
                    self.jab(ji).mult = m;
                    if m != prev {
                        ret!(extra: true);
                    }
                }
                if j == J::Faceless && other == last && full_faces >= 3 {
                    self.push(Ev::DollarsStatus { d: 5.0, delay: 0.45 });
                    return None;
                }
                None
            }
            Ctx::EndOfRound { game_over } => {
                if blueprinted {
                    return None;
                }
                let boss = self.blind.boss;
                if j == J::Campfire && boss && ab.x_mult > 1.0 {
                    self.jab(ji).x_mult = 1.0;
                    ret!(extra: true);
                }
                if j == J::Rocket && boss {
                    self.jab(ji).dollars += 2.0;
                    ret!(extra: true);
                }
                if j == J::TurtleBean {
                    if ab.remaining - 1.0 <= 0.0 {
                        self.schedule_self_destruct(ji);
                        ret!(extra: true);
                    } else {
                        self.jab(ji).remaining -= 1.0;
                        self.change_hand_size(-1.0);
                        ret!(extra: true);
                    }
                }
                if j == J::Invisible {
                    self.jab(ji).invis_rounds += 1.0;
                    ret!(extra: true);
                }
                if j == J::Popcorn {
                    if ab.mult - 4.0 <= 0.0 {
                        self.schedule_self_destruct(ji);
                        ret!(extra: true);
                    } else {
                        self.jab(ji).mult -= 4.0;
                        ret!(extra: true);
                    }
                }
                if j == J::TodoList {
                    let (hands, n) = self.visible_hands_pairs(ab.to_do);
                    let k = Key::from("to_do");
                    let i = self.rng.element(&k, n);
                    self.jab(ji).to_do = Some(hands[i]);
                    ret!(extra: true);
                }
                if j == J::Egg {
                    self.jab(ji).extra_value += 3.0;
                    self.reprice(Area::Jokers, ji);
                    ret!(extra: true);
                }
                if j == J::Gift {
                    for area in [Area::Jokers, Area::Consumables] {
                        for i in 0..self.area(area).len() {
                            self.area_mut(area)[i].ab.extra_value += 1.0;
                            self.reprice(area, i);
                        }
                    }
                    ret!(extra: true);
                }
                if j == J::HitTheRoad && ab.x_mult > 1.0 {
                    self.jab(ji).x_mult = 1.0;
                    ret!(extra: true);
                }
                if j == J::GrosMichel || j == J::Cavendish {
                    let (key, odds) = if j == J::Cavendish { ("cavendish", 1000.0) } else { ("gros_michel", 6.0) };
                    let k = Key::from(key);
                    if self.rng.random(&k) < prob / odds {
                        self.schedule_self_destruct(ji);
                        if j == J::GrosMichel {
                            self.gros_michel_extinct = true;
                        }
                    }
                    ret!(extra: true);
                }
                if j == J::MrBones && game_over && self.chips / self.blind.chips >= 0.25 {
                    let id = self.jokers[ji].id;
                    self.push(Ev::DissolveJoker { id, fac: 1.0 });
                    ret!(saved: true);
                }
                None
            }
            Ctx::EndOfRoundRepetition { effects } => {
                if j == J::Mime && effects {
                    ret!(repetitions: 1);
                }
                None
            }
            _ => unreachable!("scoring context {ctx:?}"),
        }
    }
}
