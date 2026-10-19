//! Shop and `G.FUNCS.use_card` (button_callbacks.lua, UI_definitions.lua
//! `create_card_for_shop`, card.lua `redeem`). Booster packs are in `packs`.

use crate::card::{Card, Kind};
use crate::consumables::{CEv, Rate};
use crate::data::*;
use crate::rng::Key;
use crate::events::{Ev, Pile, St};
use crate::game::{Area, CardSet, Game};
use crate::jokers::Ctx;
use crate::tags::TagCtx;

impl Game {
    pub(crate) fn locate_shop_card(&self, id: u32) -> Option<(Area, usize)> {
        if let Some(i) = self.shop_jokers.iter().position(|c| c.id == id) {
            return Some((Area::ShopJokers, i));
        }
        if let Some(i) = self.shop_boosters.iter().position(|c| c.id == id) {
            return Some((Area::ShopBoosters, i));
        }
        if let Some(i) = self.shop_vouchers.iter().position(|c| c.id == id) {
            return Some((Area::ShopVouchers, i));
        }
        None
    }

    pub(crate) fn find_shop_card(&self, id: u32) -> Option<&Card> {
        let (a, i) = self.locate_shop_card(id)?;
        Some(&self.area(a)[i])
    }

    pub(crate) fn with_shop_card(&mut self, id: u32, f: impl FnOnce(&mut Card)) {
        if let Some((a, i)) = self.locate_shop_card(id) {
            f(&mut self.area_mut(a)[i]);
        }
    }

    pub(crate) fn recost_shop_card(&mut self, id: u32) {
        if let Some((a, i)) = self.locate_shop_card(id) {
            self.reprice(a, i);
        }
    }

    // -----------------------------------------------------------------
    // Shop
    // -----------------------------------------------------------------

    pub(crate) fn update_shop(&mut self) {
        if !self.state_complete {
            let exists = self.shop_open;
            if !exists {
                self.shop_open = true;
                self.shop_jokers.clear();
                self.shop_vouchers.clear();
                self.shop_boosters.clear();
            }
            if !exists {
                // G.UIDEF.shop()'s shop-sign event
                self.push(Ev::Wait);
            }
            self.push(Ev::ShopOuter);
            self.state_complete = true;
            self.shop_exists_at_open = exists;
        }
    }

    fn shop_populate(&mut self, shop_exists: bool) {
        for i in 0..self.tags.len() {
            self.apply_tag(i, TagCtx::ShopStart);
        }
        if !shop_exists {
            self.fill_shop_jokers();
            if let Some(v) = self.current_voucher {
                self.voucher_seen = true;
                let mut c = self.make_card(Kind::Voucher(v));
                c.shop_voucher = true;
                self.materialize(1.0);
                // A 'shop'-type area: insertion order is kept (no sort).
                self.shop_vouchers.push(c);
            }
            for i in 0..2 {
                if self.used_packs[i].is_none() {
                    let b = self.get_pack("shop_pack", None);
                    self.used_packs[i] = Some(Some(b));
                }
                if let Some(Some(b)) = self.used_packs[i] {
                    let mut c = self.make_card(Kind::Booster(b));
                    c.booster_pos = i as u8 + 1;
                    self.materialize(1.0);
                    self.shop_boosters.push(c);
                }
            }
            for i in 0..self.tags.len() {
                self.apply_tag(i, TagCtx::VoucherAdd);
            }
            for i in 0..self.tags.len() {
                self.apply_tag(i, TagCtx::ShopFinalPass);
            }
        }
        crate::game::trace!(self, "shop_populate");
        // save_run
        self.push(Ev::Wait);
    }

    /// Fills the joker row up to `shop_joker_max` (the shop's
    /// `for i = 1, shop_joker_max - #G.shop_jokers.cards` loops).
    fn fill_shop_jokers(&mut self) {
        let n = (self.shop_joker_max as i64 - self.shop_jokers.len() as i64).max(0);
        for _ in 0..n {
            let c = self.create_card_for_shop();
            self.shop_jokers.push(c);
        }
    }

    /// `create_card_for_shop(G.shop_jokers)`.
    pub(crate) fn create_card_for_shop(&mut self) -> Card {
        for i in 0..self.tags.len() {
            let (_, card) = self.apply_tag_card(i, TagCtx::StoreJokerCreate { area: Area::ShopJokers });
            if let Some(mut c) = card {
                for k in 0..self.tags.len() {
                    if self.apply_tag_modify_card(k, &mut c) {
                        break;
                    }
                }
                return c;
            }
        }
        let total = self.joker_rate + self.tarot_rate + self.planet_rate + self.playing_card_rate + self.spectral_rate;
        let k = self.key_ante("cdt");
        let polled = self.rng.random(&k) * total;
        let illusion = self.voucher_used(Voucher::Illusion) && {
            let k = Key::from("illusion");
            self.rng.random(&k) > 0.6
        };
        let opts = [
            (CardSet::Joker, self.joker_rate),
            (CardSet::Tarot, self.tarot_rate),
            (CardSet::Planet, self.planet_rate),
            (if illusion { CardSet::Enhanced } else { CardSet::Base }, self.playing_card_rate),
            (CardSet::Spectral, self.spectral_rate),
        ];
        let mut check = 0.0;
        for (set, val) in opts {
            if polled > check && polled <= check + val {
                let mut c = self.create_card(set, Area::ShopJokers, false, None, false, None, "sho");
                let id = c.id;
                self.push(Ev::StoreModify { card: id });
                if matches!(set, CardSet::Base | CardSet::Enhanced) && self.voucher_used(Voucher::Illusion) {
                    let k = Key::from("illusion");
                    if self.rng.random(&k) > 0.8 {
                        let p = self.rng.random(&k);
                        let e = if p > 1.0 - 0.15 {
                            Edition::Polychrome
                        } else if p > 0.5 {
                            Edition::Holo
                        } else {
                            Edition::Foil
                        };
                        self.set_edition(&mut c, e, None);
                        self.edition_fx(e, false);
                    }
                }
                return c;
            }
            check += val;
        }
        unreachable!("create_card_for_shop: no card type selected (polled {polled})")
    }

    /// `store_joker_modify` on a card that is not yet in an area.
    fn apply_tag_modify_card(&mut self, idx: usize, c: &mut Card) -> bool {
        let t = self.tags[idx];
        let is_mod = matches!(t.kind, TagKind::Foil | TagKind::Holo | TagKind::Polychrome | TagKind::Negative);
        if t.triggered || !is_mod {
            return false;
        }
        if c.edition == Edition::None && !c.temp_edition && matches!(c.kind, Kind::Joker(_)) {
            let e = match t.kind {
                TagKind::Foil => Edition::Foil,
                TagKind::Holo => Edition::Holo,
                TagKind::Polychrome => Edition::Polychrome,
                _ => Edition::Negative,
            };
            c.temp_edition = true;
            self.tag_yep(idx, crate::events::TagFn::Edition { card: c.id, edition: e });
            self.tags[idx].triggered = true;
            return true;
        }
        false
    }

    /// `change_shop_size(mod)`.
    pub(crate) fn change_shop_size(&mut self, m: f64) {
        self.shop_joker_max += m;
        if self.shop_open {
            if m < 0.0 {
                while self.shop_jokers.len() as f64 > self.shop_joker_max {
                    let c = self.shop_jokers.pop().unwrap();
                    self.drop_unowned(c);
                }
            }
            if m > 0.0 {
                self.fill_shop_jokers();
            }
        }
    }

    pub(crate) fn do_reroll(&mut self) {
        if self.reroll_cost > 0.0 {
            self.push(Ev::EaseDollars(-self.reroll_cost));
        }
        self.push(Ev::RerollShopRun);
        self.push_after(0.3, Ev::RerollOuter);
        // save_run
        self.push(Ev::Wait);
    }

    /// `G.FUNCS.check_for_buy_space(card)`.
    pub fn has_buy_space(&self, c: &Card) -> bool {
        let neg = if c.edition == Edition::Negative { 1.0 } else { 0.0 };
        match c.kind {
            Kind::Voucher(_) | Kind::Playing => true,
            Kind::Joker(_) => (self.jokers.len() as f64) < self.joker_limit.limit + neg,
            k if k.is_consumable() => (self.consumables.len() as f64) < self.consumable_limit.limit + neg,
            _ => false,
        }
    }

    pub(crate) fn do_buy(&mut self, idx: usize, buy_and_use: bool, highlighted: &[usize]) {
        let c = &self.shop_jokers[idx];
        if !buy_and_use && !self.has_buy_space(c) {
            return;
        }
        let id = c.id;
        self.pending_highlight = highlighted.to_vec();
        self.push_after(0.1, Ev::BuyRun { card: id, buy_and_use });
    }

    fn buy_run(&mut self, id: u32, buy_and_use: bool) {
        let Some(i) = self.shop_jokers.iter().position(|c| c.id == id) else { return };
        let mut c = self.shop_jokers.remove(i);
        self.add_to_deck(&mut c, false);
        let cost = c.cost;
        if c.is_playing() {
            self.deck.insert(0, Card { face_down: true, ..c });
            self.playing_card_joker_effects(1);
        } else if !buy_and_use {
            if c.kind.is_consumable() {
                self.consumables.push(c);
            } else {
                self.jokers.push(c);
            }
            // calculate_joker({buying_card = true, card = self}) event
            self.push(Ev::Wait);
        } else {
            self.in_flight = Some(c);
        }
        if cost != 0.0 {
            self.push(Ev::EaseDollars(-cost));
        }
        if buy_and_use {
            let c = self.in_flight.take().expect("buy and use without card");
            let hl = std::mem::take(&mut self.pending_highlight);
            self.use_card(c, Some(Area::ShopJokers), &hl);
        }
    }

    pub(crate) fn do_sell(&mut self, area: Area, idx: usize) {
        let card = self.area(area)[idx].clone();
        let id = card.id;
        if area == Area::Jokers {
            self.calc_joker(idx, &Ctx::SellingSelf, None);
        }
        self.push_after(0.2, Ev::Wait);
        self.delay(0.2);
        self.push(Ev::SellRun { card: id, area });
        for j in 0..self.jokers.len() {
            if self.jokers[j].id != id {
                self.calc_joker(j, &Ctx::SellingCard, None);
            }
        }
    }

    pub(crate) fn do_end_shop(&mut self) {
        for j in 0..self.jokers.len() {
            self.calc_joker(j, &Ctx::EndingShop, None);
        }
        self.push(Ev::Wait);
        self.push_after(0.5, Ev::ToggleShopRun);
    }

    // -----------------------------------------------------------------
    // use_card
    // -----------------------------------------------------------------

    /// `G.FUNCS.use_card` for a card already removed from `area`.
    pub(crate) fn use_card(&mut self, card: Card, area: Option<Area>, highlighted: &[usize]) {
        let prev = self.state;
        if matches!(card.kind, Kind::Booster(_)) {
            self.pack_interrupt = Some(self.state);
        }
        if !self.state.is_pack() {
            self.state = St::PlayTarot;
        }
        let id = card.id;
        let mut dissolve = true;
        let mut delay_fac = 1.0;
        match card.kind {
            k if k.is_consumable() => {
                let hanged_glass = if k == Kind::Tarot(Tarot::HangedMan) {
                    highlighted.iter().filter(|&&i| self.hand[i].enhancement == Enhancement::Glass).count()
                } else {
                    0
                };
                if !matches!(prev, St::TarotPack | St::PlanetPack | St::SpectralPack) {
                    // draw_card(G.hand, G.play, ..., card)
                    self.wait(crate::events::Trig::Before(0.1));
                }
                self.delay(0.2);
                self.consumed.push(card.clone());
                self.use_consumeable(&card, highlighted);
                for j in 0..self.jokers.len() {
                    self.calc_joker(j, &Ctx::UsingConsumeable { kind: k, hanged_glass }, None);
                }
            }
            Kind::Playing => {
                let mut c = card;
                c.face_down = true;
                self.playing_card_joker_effects(1);
                self.add_to_deck(&mut c, false);
                self.deck.insert(0, c);
                dissolve = false;
                delay_fac = 0.2;
            }
            Kind::Joker(_) => {
                let mut c = card;
                self.add_to_deck(&mut c, false);
                self.jokers.push(c);
                dissolve = false;
                delay_fac = 0.2;
            }
            Kind::Booster(_) => {
                self.delay(0.1);
                if card.booster_pos > 0 {
                    self.used_packs[card.booster_pos as usize - 1] = Some(None);
                }
                self.wait(crate::events::Trig::Before(0.1));
                self.open_booster(card);
                return;
            }
            Kind::Voucher(_) => {
                self.delay(0.1);
                self.wait(crate::events::Trig::Before(0.1));
                self.consumed.push(card.clone());
                self.redeem(&card);
            }
            _ => unreachable!(),
        }
        self.push_after(0.2, Ev::UseFinishOuter { prev, area, card: id, dissolve, delay_fac });
    }

    /// `Card:redeem()`.
    fn redeem(&mut self, card: &Card) {
        let Kind::Voucher(v) = card.kind else { return };
        self.set_voucher_used(v);
        self.push_after(0.4, Ev::Wait);
        self.push(Ev::EaseDollars(-card.cost));
        self.current_voucher = None;
        self.apply_voucher(v, false);
        self.delay(0.6);
    }

    /// `Card:apply_to_run(center)`; `from_back` when applied by a deck.
    pub(crate) fn apply_voucher(&mut self, v: Voucher, from_back: bool) {
        use Voucher as V;
        let extra = v.def().extra;
        match v {
            V::OverstockNorm | V::OverstockPlus => self.push(Ev::Consumable(CEv::ShopSize(1.0))),
            V::TarotMerchant | V::TarotTycoon => self.push(Ev::Consumable(CEv::SetRate { which: Rate::Tarot, v: 4.0 * extra })),
            V::PlanetMerchant | V::PlanetTycoon => self.push(Ev::Consumable(CEv::SetRate { which: Rate::Planet, v: 4.0 * extra })),
            V::Hone | V::GlowUp => self.push(Ev::Consumable(CEv::SetRate { which: Rate::Edition, v: extra })),
            V::MagicTrick | V::Illusion => self.push(Ev::Consumable(CEv::SetRate { which: Rate::PlayingCard, v: extra })),
            V::CrystalBall => self.push(Ev::Consumable(CEv::ConsumableSlot)),
            V::ClearanceSale | V::Liquidation => self.push(Ev::Consumable(CEv::Discount(extra))),
            V::RerollSurplus | V::RerollGlut => {
                if !from_back {
                    self.push(Ev::Consumable(CEv::RerollDiscount(extra)));
                }
            }
            V::SeedMoney | V::MoneyTree => self.push(Ev::Consumable(CEv::SetRate { which: Rate::InterestCap, v: extra })),
            V::Grabber | V::NachoTong => {
                self.rr_hands += extra;
                self.push(Ev::EaseHands(extra));
            }
            V::PaintBrush | V::Palette => self.change_hand_size(1.0),
            V::Wasteful | V::Recyclomancy => {
                self.rr_discards += extra;
                self.push(Ev::EaseDiscard(extra));
            }
            V::Antimatter => self.push(Ev::Consumable(CEv::JokerSlot)),
            V::Hieroglyph | V::Petroglyph => {
                self.push(Ev::EaseAnte(-(extra as i32)));
                self.blind_ante -= extra as i32;
                if v == V::Hieroglyph {
                    self.rr_hands -= extra;
                    self.push(Ev::EaseHands(-extra));
                } else {
                    self.rr_discards -= extra;
                    self.push(Ev::EaseDiscard(-extra));
                }
            }
            _ => {}
        }
    }

    /// Runs a shop / pack / use-card event (routed by `run_event`).
    pub(crate) fn run_shop_event(&mut self, e: Ev) {
        match e {
            Ev::ShopOuter => {
                let ex = self.shop_exists_at_open;
                self.push_ev(Ev::ShopPopulate { shop_exists: ex }, crate::events::Trig::After(0.2), true, false);
            }
            Ev::ShopPopulate { shop_exists } => self.shop_populate(shop_exists),
            Ev::StoreModify { card } => {
                for i in 0..self.tags.len() {
                    if self.apply_tag(i, TagCtx::StoreJokerModify { card }) {
                        break;
                    }
                }
            }
            Ev::RerollShopRun => {
                let final_free = self.free_rerolls > 0.0;
                self.free_rerolls = (self.free_rerolls - 1.0).max(0.0);
                self.calculate_reroll_cost(final_free);
                while let Some(c) = self.shop_jokers.pop() {
                    self.drop_unowned(c);
                }
                self.fill_shop_jokers();
            }
            Ev::RerollShopJokers => {
                for j in 0..self.jokers.len() {
                    self.calc_joker(j, &Ctx::RerollShop, None);
                }
            }
            Ev::BuyRun { card, buy_and_use } => self.buy_run(card, buy_and_use),
            Ev::SellRun { card, area } => {
                if let Some(i) = self.area(area).iter().position(|c| c.id == card) {
                    let sc = self.area(area)[i].sell_cost;
                    self.push(Ev::EaseDollars(sc));
                    self.start_dissolve(card);
                    self.delay(0.3);
                    let is_joker = matches!(self.area(area)[i].kind, Kind::Joker(_));
                    if is_joker && self.blind.kind == Some(BlindKind::FinalLeaf) {
                        self.push(Ev::VerdantDisable);
                    }
                }
            }
            Ev::VerdantDisable => self.disable_blind(),
            Ev::ToggleShopRun => {
                self.shop_open = false;
                for v in [std::mem::take(&mut self.shop_jokers), std::mem::take(&mut self.shop_vouchers), std::mem::take(&mut self.shop_boosters)] {
                    for c in v {
                        self.drop_unowned(c);
                    }
                }
                self.state_complete = false;
                self.state = St::BlindSelect;
            }
            Ev::OpenPack { card } => self.open_pack_run(card),
            Ev::PackCreate => self.pack_create(),
            Ev::PackStateUi => self.push(Ev::PackDrawHand),
            Ev::PackDrawHand => match self.state {
                St::TarotPack | St::SpectralPack => {
                    self.draw_from_deck_to_hand(None);
                    self.push_after(0.5, Ev::Wait);
                }
                St::StandardPack | St::BuffoonPack => self.push_after(0.5, Ev::Wait),
                _ => {}
            },
            Ev::UseFinishOuter { prev, area, card, dissolve, delay_fac } => {
                if dissolve {
                    self.start_dissolve(card);
                }
                self.push_after(0.1, Ev::UseFinishInner { prev, area, delay_fac });
            }
            Ev::UseFinishInner { prev, area, delay_fac } => {
                self.state = prev;
                // Using a held consumable inside a pack does not count as a pick.
                if prev.is_pack() && self.pack.is_some() && area != Some(Area::Consumables) {
                    if self.pack_choices > 1 {
                        self.pack_choices -= 1;
                    } else {
                        self.end_consumeable(delay_fac);
                    }
                }
            }
            Ev::EndConsumeable { df } => {
                for _ in 0..self.hand.len() {
                    self.draw_ev(Pile::Hand, Pile::Deck, None, false, false, 0.08);
                }
                self.push_after(0.2 * df, Ev::EndConsumeableInner);
                self.apply_new_blind_choice_tags();
                // save_run
                self.push_after(0.2 * df, Ev::Wait);
            }
            Ev::EndConsumeableInner => {
                let back = self.pack_interrupt.take().unwrap_or(St::Shop);
                self.state = back;
                self.pack = None;
                self.push(Ev::Wait);
            }
            other => unreachable!("not a shop event: {other:?}"),
        }
    }
}
