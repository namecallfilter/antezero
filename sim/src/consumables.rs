//! Port of `Card:use_consumeable` / `Card:can_use_consumeable` (card.lua) plus
//! a few small deferred effects (voucher application events etc.).

use crate::card::{Card, Kind};
use crate::data::*;
use crate::rng::Key;
use crate::events::{Ev, Pile, St};
use crate::game::{Area, CardSet, Game};
use crate::jokers::Ctx;

/// Deferred pieces of consumable resolution and other small events.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum CEv {
    SortHand,
    DiscardToDeckOuter,
    BlueSealPlanet,
    BlindDefeat,
    ShopSize(f64),
    SetRate { which: Rate, v: f64 },
    ConsumableSlot,
    Discount(f64),
    RerollDiscount(f64),
    JokerSlot,
    // Tarots
    Death { target: u32, source: u32 },
    Strength { target: u32 },
    SuitConv { target: u32, suit: Suit },
    Enhance { target: u32, e: Enhancement },
    UnhighlightAll,
    SetSeal { target: u32, seal: Seal },
    Aura { target: u32 },
    Cryptid { source: u32 },
    SigilSet { target: u32, suit: Suit },
    OuijaSet { target: u32, rank: Rank },
    DestroyCards { n: u8, ids: [u32; 8] },
    CreateSpectralCards { which: Spectral, n: u8 },
    Fool,
    Hermit,
    Temperance,
    EmperorPriestess { planet: bool },
    JudgementSoul { soul: bool },
    AnkhDestroy { chosen: u32 },
    AnkhCopy { chosen: u32 },
    Wraith,
    WheelEctoHex { which: EditionGift },
    /// Checkered Deck: Clubs → Spades, Diamonds → Hearts.
    Checkered,
}

/// A `G.GAME` rate a voucher sets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Rate {
    Tarot,
    Planet,
    Edition,
    PlayingCard,
    InterestCap,
}

/// The consumables that give a random editionless joker an edition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EditionGift {
    Wheel,
    Ectoplasm,
    Hex,
}

impl Game {
    fn hand_pos(&self, id: u32) -> Option<usize> {
        self.hand.iter().position(|c| c.id == id)
    }

    /// Editionless jokers (`eligible_strength_jokers` / `eligible_editionless_jokers`).
    fn editionless_jokers(&self) -> Vec<usize> {
        (0..self.jokers.len()).filter(|&i| self.jokers[i].edition == Edition::None).collect()
    }

    fn has_editionless_joker(&self) -> bool {
        self.jokers.iter().any(|c| c.edition == Edition::None)
    }

    /// Total sell value of the owned jokers (Temperance, Swashbuckler).
    pub(crate) fn joker_sell_total(&self) -> f64 {
        let mut m = 0.0;
        for c in &self.jokers {
            if matches!(c.kind, Kind::Joker(_)) {
                m += c.sell_cost;
            }
        }
        m
    }

    /// `Card:can_use_consumeable()` for the consumable `c` located in `area`
    /// with the given highlighted hand cards.
    pub fn can_use_consumable(&self, c: &Card, area: Option<Area>, n_highlighted: usize, highlighted_first_editionless: bool) -> bool {
        if !self.play.is_empty() {
            return false;
        }
        let st = self.state;
        if matches!(st, St::HandPlayed | St::DrawToHand | St::PlayTarot) {
            return false;
        }
        let in_cons = area == Some(Area::Consumables);
        let cons_room = (self.consumables.len() as f64) < self.consumable_limit.limit || in_cons;
        match c.kind {
            Kind::Planet(_) => return true,
            Kind::Tarot(Tarot::Hermit) | Kind::Tarot(Tarot::Temperance) | Kind::Spectral(Spectral::BlackHole) => return true,
            Kind::Tarot(Tarot::WheelOfFortune) => {
                if self.has_editionless_joker() {
                    return true;
                }
            }
            Kind::Spectral(Spectral::Ankh) => {
                if !self.jokers.is_empty() && self.joker_limit.limit > 1.0 {
                    return true;
                }
            }
            Kind::Spectral(Spectral::Aura) => {
                if n_highlighted == 1 && highlighted_first_editionless {
                    return true;
                }
            }
            Kind::Spectral(Spectral::Ectoplasm | Spectral::Hex) => {
                if self.has_editionless_joker() {
                    return true;
                }
            }
            Kind::Tarot(Tarot::Emperor | Tarot::HighPriestess) => {
                if cons_room {
                    return true;
                }
            }
            Kind::Tarot(Tarot::Fool) => {
                if cons_room && self.last_tarot_planet.is_some() && self.last_tarot_planet != Some(Kind::Tarot(Tarot::Fool)) {
                    return true;
                }
            }
            Kind::Tarot(Tarot::Judgement) | Kind::Spectral(Spectral::Soul | Spectral::Wraith) => {
                return (self.jokers.len() as f64) < self.joker_limit.limit;
            }
            _ => {}
        }
        if matches!(st, St::SelectingHand | St::TarotPack | St::SpectralPack | St::PlanetPack) {
            let (max_h, min_h) = consumable_highlight(c.kind);
            if max_h > 0 {
                let mod_num = max_h.min(5);
                if mod_num as usize >= n_highlighted && n_highlighted >= min_h.max(1) as usize {
                    return true;
                }
            }
            if matches!(
                c.kind,
                Kind::Spectral(Spectral::Familiar | Spectral::Grim | Spectral::Incantation | Spectral::Immolate | Spectral::Sigil | Spectral::Ouija)
            ) && self.hand.len() > 1
            {
                return true;
            }
        }
        false
    }

    /// `set_consumeable_usage(card)`.
    fn set_consumeable_usage(&mut self, c: &Card) {
        match c.kind {
            Kind::Tarot(_) => self.consumable_usage_tarot += 1.0,
            Kind::Planet(p) => {
                self.consumable_usage_planet += 1.0;
                self.planets_used |= 1 << (p as u16);
            }
            Kind::Spectral(_) => self.consumable_usage_spectral += 1.0,
            _ => {}
        }
        if matches!(c.kind, Kind::Tarot(_) | Kind::Planet(_)) {
            self.push(Ev::LastTarotPlanetOuter { kind: c.kind });
        }
    }

    /// `Card:use_consumeable(area)`; `hl` are the highlighted hand indices.
    pub(crate) fn use_consumeable(&mut self, card: &Card, hl: &[usize]) {
        self.set_consumeable_usage(card);
        if card.debuff {
            return;
        }
        let mut hl: Vec<usize> = hl.to_vec();
        hl.sort_unstable();
        let hl_ids: Vec<u32> = hl.iter().map(|&i| self.hand[i].id).collect();
        let k = card.kind;
        use Spectral as S;
        use Tarot as T;

        // Enhancement / suit / rank conversions.
        let mod_conv = match k {
            Kind::Tarot(T::Magician) => Some(Enhancement::Lucky),
            Kind::Tarot(T::Empress) => Some(Enhancement::Mult),
            Kind::Tarot(T::Heirophant) => Some(Enhancement::Bonus),
            Kind::Tarot(T::Lovers) => Some(Enhancement::Wild),
            Kind::Tarot(T::Chariot) => Some(Enhancement::Steel),
            Kind::Tarot(T::Justice) => Some(Enhancement::Glass),
            Kind::Tarot(T::Devil) => Some(Enhancement::Gold),
            Kind::Tarot(T::Tower) => Some(Enhancement::Stone),
            _ => None,
        };
        let suit_conv = match k {
            Kind::Tarot(T::Star) => Some(Suit::Diamonds),
            Kind::Tarot(T::Moon) => Some(Suit::Clubs),
            Kind::Tarot(T::Sun) => Some(Suit::Hearts),
            Kind::Tarot(T::World) => Some(Suit::Spades),
            _ => None,
        };
        let is_conv = mod_conv.is_some() || suit_conv.is_some() || matches!(k, Kind::Tarot(T::Death | T::Strength));
        if consumable_highlight(k).0 > 0 {
            self.hand_text_imm(0.0);
        }
        if is_conv {
            self.push_after(0.4, Ev::Wait);
            for _ in &hl_ids {
                self.push_after(0.15, Ev::Wait);
            }
            self.delay(0.2);
            if k == Kind::Tarot(T::Death) {
                // Rightmost highlighted card is the template.
                let src = *hl_ids.last().unwrap();
                for &t in &hl_ids {
                    self.push_after(0.1, Ev::Consumable(CEv::Death { target: t, source: src }));
                }
            } else if k == Kind::Tarot(T::Strength) {
                for &t in &hl_ids {
                    self.push_after(0.1, Ev::Consumable(CEv::Strength { target: t }));
                }
            } else if let Some(s) = suit_conv {
                for &t in &hl_ids {
                    self.push_after(0.1, Ev::Consumable(CEv::SuitConv { target: t, suit: s }));
                }
            } else if let Some(e) = mod_conv {
                for &t in &hl_ids {
                    self.push_after(0.1, Ev::Consumable(CEv::Enhance { target: t, e }));
                }
            }
            for _ in &hl_ids {
                self.push_after(0.15, Ev::Wait);
            }
            self.push_after(0.2, Ev::Consumable(CEv::UnhighlightAll));
            self.delay(0.5);
        }
        if k == Kind::Spectral(S::BlackHole) {
            self.hand_text(0.3);
            self.push_after(0.2, Ev::Wait);
            self.hand_text(0.0);
            self.push_after(0.9, Ev::Wait);
            self.hand_text(0.0);
            self.push_after(0.9, Ev::Wait);
            self.hand_text(0.0);
            self.delay(1.3);
            for h in HandType::PAIRS_ORDER {
                self.level_up_hand_ev(h, 1.0, true);
            }
            self.hand_text(0.0);
        }
        let seal = match k {
            Kind::Spectral(S::Talisman) => Some(Seal::Gold),
            Kind::Spectral(S::DejaVu) => Some(Seal::Red),
            Kind::Spectral(S::Trance) => Some(Seal::Blue),
            Kind::Spectral(S::Medium) => Some(Seal::Purple),
            _ => None,
        };
        if let Some(seal) = seal {
            let t = hl_ids[0];
            self.push(Ev::Wait);
            self.push_after(0.1, Ev::Consumable(CEv::SetSeal { target: t, seal }));
            self.delay(0.5);
            self.push_after(0.2, Ev::Consumable(CEv::UnhighlightAll));
        }
        if k == Kind::Spectral(S::Aura) {
            self.push_after(0.4, Ev::Consumable(CEv::Aura { target: hl_ids[0] }));
        }
        if k == Kind::Spectral(S::Cryptid) {
            self.push(Ev::Consumable(CEv::Cryptid { source: hl_ids[0] }));
        }
        if matches!(k, Kind::Spectral(S::Sigil | S::Ouija)) {
            let n = self.hand.len();
            self.push_after(0.4, Ev::Wait);
            for _ in 0..n {
                self.push_after(0.15, Ev::Wait);
            }
            self.delay(0.2);
            if k == Kind::Spectral(S::Sigil) {
                let idx = self.rng.element_str("sigil", 4);
                let suit = [Suit::Spades, Suit::Hearts, Suit::Diamonds, Suit::Clubs][idx];
                let ids: Vec<u32> = self.hand.iter().map(|c| c.id).collect();
                for t in ids {
                    self.push(Ev::Consumable(CEv::SigilSet { target: t, suit }));
                }
            } else {
                const R: [Rank; 13] = [2, 3, 4, 5, 6, 7, 8, 9, 10, JACK, QUEEN, KING, ACE];
                let idx = self.rng.element_str("ouija", 13);
                let rank = R[idx];
                let ids: Vec<u32> = self.hand.iter().map(|c| c.id).collect();
                for t in ids {
                    self.push(Ev::Consumable(CEv::OuijaSet { target: t, rank }));
                }
                self.change_hand_size(-1.0);
            }
            for _ in 0..n {
                self.push_after(0.15, Ev::Wait);
            }
            self.delay(0.5);
        }
        if let Kind::Planet(p) = k {
            self.hand_text(0.3);
            self.level_up_hand_ev(p.hand(), 1.0, false);
            self.hand_text(0.0);
        }
        // remove_card consumables.
        let removes = matches!(k, Kind::Tarot(T::HangedMan) | Kind::Spectral(S::Familiar | S::Grim | S::Incantation | S::Immolate));
        if removes {
            let mut destroyed: Vec<u32> = Vec::new();
            match k {
                Kind::Tarot(T::HangedMan) => {
                    for &id in hl_ids.iter().rev() {
                        destroyed.push(id);
                    }
                }
                Kind::Spectral(S::Familiar | S::Grim | S::Incantation) => {
                    let mut order: Vec<usize> = (0..self.hand.len()).collect();
                    order.sort_by_key(|&i| self.hand[i].id);
                    let i = order[self.rng.element_str("random_destroy", order.len())];
                    destroyed.push(self.hand[i].id);
                }
                Kind::Spectral(S::Immolate) => {
                    let mut tmp: Vec<u32> = self.hand.iter().map(|c| c.id).collect();
                    tmp.sort_unstable();
                    let seed = self.rng.seed_str("immolate");
                    crate::rng::shuffle(&mut tmp, seed);
                    for id in tmp.into_iter().take(5) {
                        destroyed.push(id);
                    }
                }
                _ => {}
            }
            // The destroy loops run back to front.
            let order: Vec<u32> = if k == Kind::Tarot(T::HangedMan) { destroyed.clone() } else { destroyed.iter().rev().copied().collect() };
            let mut ids = [0u32; 8];
            for (i, &id) in order.iter().enumerate().take(8) {
                ids[i] = id;
            }
            self.push_after(0.4, Ev::Wait);
            let d = if k == Kind::Tarot(T::HangedMan) { 0.2 } else { 0.1 };
            self.push_after(d, Ev::Consumable(CEv::DestroyCards { n: order.len() as u8, ids }));
            match k {
                Kind::Spectral(s @ (S::Familiar | S::Grim | S::Incantation)) => {
                    let n = match s {
                        S::Familiar => 3,
                        S::Grim => 2,
                        _ => 4,
                    };
                    self.push_after(0.7, Ev::Consumable(CEv::CreateSpectralCards { which: s, n }));
                }
                Kind::Spectral(S::Immolate) => {
                    self.delay(0.5);
                    self.push(Ev::EaseDollars(20.0));
                }
                _ => {}
            }
            self.delay(0.3);
            let par = self.pareidolia();
            let faces = destroyed
                .iter()
                .filter(|&&id| self.hand.iter().find(|c| c.id == id).is_some_and(|c| c.is_face(false, par)))
                .count();
            for j in 0..self.jokers.len() {
                self.calc_joker(j, &Ctx::RemovePlayingCards { faces, glass: 0 }, None);
            }
        }
        match k {
            Kind::Tarot(T::Fool) => {
                self.push_after(0.4, Ev::Consumable(CEv::Fool));
                self.delay(0.6);
            }
            Kind::Tarot(T::Hermit) => {
                self.push_after(0.4, Ev::Consumable(CEv::Hermit));
                self.delay(0.6);
            }
            Kind::Tarot(T::Temperance) => {
                self.push_after(0.4, Ev::Consumable(CEv::Temperance));
                self.delay(0.6);
            }
            Kind::Tarot(T::Emperor | T::HighPriestess) => {
                let n = 2f64.min(self.consumable_limit.limit - self.consumables.len() as f64);
                let planet = k == Kind::Tarot(T::HighPriestess);
                let mut i = 1.0;
                while i <= n {
                    self.push_after(0.4, Ev::Consumable(CEv::EmperorPriestess { planet }));
                    i += 1.0;
                }
                self.delay(0.6);
            }
            Kind::Tarot(T::Judgement) | Kind::Spectral(S::Soul) => {
                let soul = k == Kind::Spectral(S::Soul);
                self.push_after(0.4, Ev::Consumable(CEv::JudgementSoul { soul }));
                self.delay(0.6);
            }
            Kind::Spectral(S::Ankh) => {
                let mut order: Vec<usize> = (0..self.jokers.len()).collect();
                order.sort_by_key(|&i| self.jokers[i].id);
                let pick = order[self.rng.element_str("ankh_choice", order.len())];
                let chosen = self.jokers[pick].id;
                self.ankh_deletable = self.jokers.iter().filter(|c| !c.ab.eternal).map(|c| c.id).collect();
                self.push_before(0.75, Ev::Consumable(CEv::AnkhDestroy { chosen }));
                self.push_before(0.4, Ev::Consumable(CEv::AnkhCopy { chosen }));
            }
            Kind::Spectral(S::Wraith) => {
                self.push_after(0.4, Ev::Consumable(CEv::Wraith));
                self.delay(0.6);
            }
            Kind::Tarot(T::WheelOfFortune) => {
                let k = Key::from("wheel_of_fortune");
                if self.rng.random(&k) < self.probabilities / 4.0 {
                    self.push_after(0.4, Ev::Consumable(CEv::WheelEctoHex { which: EditionGift::Wheel }));
                } else {
                    self.push_after(0.4, Ev::Wait);
                }
                self.delay(0.6);
            }
            Kind::Spectral(S::Ectoplasm) => {
                self.push_after(0.4, Ev::Consumable(CEv::WheelEctoHex { which: EditionGift::Ectoplasm }));
                self.delay(0.6);
            }
            Kind::Spectral(S::Hex) => {
                self.push_after(0.4, Ev::Consumable(CEv::WheelEctoHex { which: EditionGift::Hex }));
                self.delay(0.6);
            }
            _ => {}
        }
    }

    pub(crate) fn run_consumable_event(&mut self, e: CEv) {
        match e {
            CEv::SortHand => self.sort_hand(),
            CEv::DiscardToDeckOuter => {
                for _ in 0..self.discard.len() {
                    self.draw_ev(Pile::Discard, Pile::Deck, None, false, false, 0.005);
                }
            }
            CEv::BlueSealPlanet => {
                if let Some(h) = self.last_hand_played {
                    let p = Planet::for_hand(h);
                    let mut c = self.create_card(CardSet::Planet, Area::Consumables, false, None, false, Some(Kind::Planet(p)), "blusl");
                    self.add_to_deck(&mut c, false);
                    self.consumables.push(c);
                    self.consumeable_buffer = 0.0;
                }
            }
            CEv::BlindDefeat => self.blind_defeat(),
            CEv::ShopSize(m) => self.change_shop_size(m),
            CEv::SetRate { which, v } => match which {
                Rate::Tarot => self.tarot_rate = v,
                Rate::Planet => self.planet_rate = v,
                Rate::Edition => self.edition_rate = v,
                Rate::PlayingCard => self.playing_card_rate = v,
                Rate::InterestCap => self.interest_cap = v,
            },
            CEv::ConsumableSlot => self.consumable_limit.limit += 1.0,
            CEv::Discount(d) => {
                self.discount_percent = d;
                self.set_cost_all();
            }
            CEv::RerollDiscount(d) => {
                self.rr_reroll_cost -= d;
                self.reroll_cost = (self.reroll_cost - d).max(0.0);
            }
            CEv::JokerSlot => self.joker_limit.limit += 1.0,
            CEv::Death { target, source } => {
                if target == source {
                    return;
                }
                let (Some(ti), Some(si)) = (self.hand_pos(target), self.hand_pos(source)) else { return };
                let src = self.hand[si].clone();
                let mut t = self.hand[ti].clone();
                self.copy_into(&src, &mut t);
                self.hand[ti] = t;
            }
            CEv::Strength { target } => {
                let Some(i) = self.hand_pos(target) else { return };
                let mut c = self.hand[i].clone();
                let r = if c.rank == ACE { 2 } else { (c.rank + 1).min(ACE) };
                let s = c.suit;
                self.set_base(&mut c, s, r);
                self.hand[i] = c;
            }
            CEv::SuitConv { target, suit } => {
                let Some(i) = self.hand_pos(target) else { return };
                let mut c = self.hand[i].clone();
                let r = c.rank;
                self.set_base(&mut c, suit, r);
                self.debuff_card(&mut c);
                self.hand[i] = c;
            }
            CEv::Enhance { target, e } => {
                let Some(i) = self.hand_pos(target) else { return };
                let mut c = self.hand[i].clone();
                self.set_enhancement(&mut c, e);
                self.hand[i] = c;
            }
            CEv::UnhighlightAll => {}
            CEv::SetSeal { target, seal } => {
                if let Some(i) = self.hand_pos(target) {
                    let mut c = self.hand[i].clone();
                    c.seal = Some(seal);
                    self.set_cost(&mut c, Some(Area::Hand));
                    self.hand[i] = c;
                }
            }
            CEv::Aura { target } => {
                let k = Key::from("aura");
                let e = self.poll_edition(&k, 1.0, true, true);
                if let Some(i) = self.hand_pos(target) {
                    let mut c = self.hand[i].clone();
                    self.set_edition(&mut c, e, Some(Area::Hand));
                    self.hand[i] = c;
                }
                self.edition_fx(e, true);
            }
            CEv::Cryptid { source } => {
                let Some(si) = self.hand_pos(source) else { return };
                let src = self.hand[si].clone();
                for _ in 0..2 {
                    let mut c = self.copy_card(&src, false);
                    self.add_to_deck(&mut c, false);
                    self.hand.push(c);
                    self.materialize(1.0);
                }
                self.playing_card_joker_effects(2);
            }
            CEv::SigilSet { target, suit } => {
                if let Some(i) = self.hand_pos(target) {
                    let mut c = self.hand[i].clone();
                    let r = c.rank;
                    self.set_base(&mut c, suit, r);
                    self.hand[i] = c;
                }
            }
            CEv::OuijaSet { target, rank } => {
                if let Some(i) = self.hand_pos(target) {
                    let mut c = self.hand[i].clone();
                    let s = c.suit;
                    self.set_base(&mut c, s, rank);
                    self.hand[i] = c;
                }
            }
            CEv::DestroyCards { n, ids } => {
                for &id in &ids[..n as usize] {
                    if let Some(i) = self.hand_pos(id) {
                        if self.hand[i].enhancement == Enhancement::Glass {
                            self.hand[i].shattered = true;
                            self.shatter(id);
                        } else {
                            self.start_dissolve(id);
                        }
                    }
                }
            }
            CEv::CreateSpectralCards { which, n } => {
                for _ in 0..n {
                    let (rank, suit) = match which {
                        Spectral::Familiar => {
                            let r = [JACK, QUEEN, KING][self.rng.element_str("familiar_create", 3)];
                            let s = [Suit::Spades, Suit::Hearts, Suit::Diamonds, Suit::Clubs][self.rng.element_str("familiar_create", 4)];
                            (r, s)
                        }
                        Spectral::Grim => {
                            let s = [Suit::Spades, Suit::Hearts, Suit::Diamonds, Suit::Clubs][self.rng.element_str("grim_create", 4)];
                            (ACE, s)
                        }
                        _ => {
                            let r = [2, 3, 4, 5, 6, 7, 8, 9, 10][self.rng.element_str("incantation_create", 9)];
                            let s = [Suit::Spades, Suit::Hearts, Suit::Diamonds, Suit::Clubs][self.rng.element_str("incantation_create", 4)];
                            (r, s)
                        }
                    };
                    // Enhanced pool without Stone, in pool order.
                    const POOL: [Enhancement; 7] = [
                        Enhancement::Bonus,
                        Enhancement::Mult,
                        Enhancement::Wild,
                        Enhancement::Glass,
                        Enhancement::Steel,
                        Enhancement::Gold,
                        Enhancement::Lucky,
                    ];
                    let e = POOL[self.rng.element_str("spe_card", 7)];
                    let mut c = self.make_playing_card(suit, rank, e);
                    self.add_to_deck(&mut c, false);
                    self.materialize(1.0);
                    self.hand.push(c);
                }
                self.playing_card_joker_effects(n as usize);
            }
            CEv::Fool => {
                if self.consumable_limit.limit > self.consumables.len() as f64 {
                    let forced = self.last_tarot_planet;
                    let mut c = self.create_card(CardSet::TarotPlanet, Area::Consumables, false, None, false, forced, "fool");
                    self.add_to_deck(&mut c, false);
                    self.consumables.push(c);
                }
            }
            CEv::Hermit => {
                let d = 0f64.max(self.dollars.min(20.0));
                self.ease_dollars(d);
            }
            CEv::Temperance => {
                let m = self.joker_sell_total().min(50.0);
                self.ease_dollars(m);
            }
            CEv::EmperorPriestess { planet } => {
                if self.consumable_limit.limit > self.consumables.len() as f64 {
                    let (set, app) = if planet { (CardSet::Planet, "pri") } else { (CardSet::Tarot, "emp") };
                    let mut c = self.create_card(set, Area::Consumables, false, None, false, None, app);
                    self.add_to_deck(&mut c, false);
                    self.consumables.push(c);
                }
            }
            CEv::JudgementSoul { soul } => {
                let mut c = self.create_card(CardSet::Joker, Area::Jokers, soul, None, false, None, if soul { "sou" } else { "jud" });
                self.add_to_deck(&mut c, false);
                self.jokers.push(c);
            }
            CEv::AnkhDestroy { chosen } => {
                let del = std::mem::take(&mut self.ankh_deletable);
                for id in del {
                    if id != chosen {
                        self.start_dissolve(id);
                    }
                }
            }
            CEv::AnkhCopy { chosen } => {
                let Some(i) = self.jokers.iter().position(|c| c.id == chosen) else { return };
                let src = self.jokers[i].clone();
                let strip = src.edition == Edition::Negative;
                let mut c = self.copy_card(&src, strip);
                self.materialize(1.0);
                self.add_to_deck(&mut c, false);
                self.jokers.push(c);
            }
            CEv::Wraith => {
                let mut c = self.create_card(CardSet::Joker, Area::Jokers, false, Some(0.99), false, None, "wra");
                self.add_to_deck(&mut c, false);
                self.jokers.push(c);
                if self.dollars != 0.0 {
                    let d = self.dollars;
                    self.ease_dollars(-d);
                }
            }
            CEv::Checkered => self.checkered(),
            CEv::WheelEctoHex { which } => {
                let mut pool = self.editionless_jokers();
                if pool.is_empty() {
                    return;
                }
                pool.sort_by_key(|&i| self.jokers[i].id);
                let key = match which {
                    EditionGift::Wheel => "wheel_of_fortune",
                    EditionGift::Ectoplasm => "ectoplasm",
                    EditionGift::Hex => "hex",
                };
                let pick = pool[self.rng.element_str(key, pool.len())];
                let ed = match which {
                    EditionGift::Ectoplasm => Edition::Negative,
                    EditionGift::Hex => Edition::Polychrome,
                    EditionGift::Wheel => {
                        let k = Key::from("wheel_of_fortune");
                        self.poll_edition(&k, 1.0, true, true)
                    }
                };
                let mut c = self.jokers[pick].clone();
                self.set_edition(&mut c, ed, Some(Area::Jokers));
                self.jokers[pick] = c;
                self.edition_fx(ed, true);
                if which == EditionGift::Hex {
                    let keep = self.jokers[pick].id;
                    let ids: Vec<u32> = self.jokers.iter().filter(|c| c.id != keep && !c.ab.eternal).map(|c| c.id).collect();
                    for id in ids {
                        self.start_dissolve(id);
                    }
                }
                if which == EditionGift::Ectoplasm {
                    let m = self.ecto_minus;
                    self.change_hand_size(-m);
                    self.ecto_minus += 1.0;
                }
            }
        }
    }

    fn checkered(&mut self) {
        let mut v = std::mem::take(&mut self.deck);
        for c in v.iter_mut() {
            let to = match c.suit {
                Suit::Clubs => Some(Suit::Spades),
                Suit::Diamonds => Some(Suit::Hearts),
                _ => None,
            };
            if let Some(s) = to {
                let r = c.rank;
                self.set_base(c, s, r);
            }
        }
        self.deck = v;
    }

    /// `copy_card(other, new_card)` into an existing playing card (Death).
    fn copy_into(&mut self, src: &Card, t: &mut Card) {
        let keep_id = t.id;
        let keep_orig = t.orig_suit;
        let keep_added = t.added_to_deck;
        let mut c = src.clone();
        c.id = keep_id;
        c.orig_suit = keep_orig;
        c.added_to_deck = keep_added;
        c.lucky_trigger = false;
        // Not part of what copy_card copies: the card keeps its facing and
        // area flags; set_base resets times_played.
        c.face_down = t.face_down;
        c.times_played = 0;
        // set_ability keeps the target's forced_selection; the copied
        // ability table only overrides it when the source has one.
        c.ab.forced_selection = src.ab.forced_selection || t.ab.forced_selection;
        c.shattered = t.shattered;
        c.destroyed = t.destroyed;
        c.getting_sliced = t.getting_sliced;
        self.mark_used(&c);
        self.set_cost(&mut c, Some(Area::Hand));
        *t = c;
    }
}

/// Whether a consumable acts on highlighted hand cards (Aura has no
/// `max_highlighted` but needs exactly one card).
pub(crate) fn uses_targets(k: Kind) -> bool {
    consumable_highlight(k).0 > 0 || k == Kind::Spectral(Spectral::Aura)
}

/// `(max_highlighted, min_highlighted)`.
pub(crate) fn consumable_highlight(k: Kind) -> (u8, u8) {
    match k {
        Kind::Tarot(t) => (t.def().max_highlighted, t.def().min_highlighted),
        Kind::Spectral(s) => (s.def().max_highlighted, s.def().min_highlighted),
        _ => (0, 0),
    }
}
