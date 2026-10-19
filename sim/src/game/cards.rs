//! Card construction (`Card:init` / `set_ability` / `set_base`), costs,
//! editions, deck membership (`add_to_deck` / `remove_from_deck` /
//! `remove`), debuffs and resource easing.

use super::*;
use crate::card::{Ability, Card, Kind};

impl Game {
    // -----------------------------------------------------------------
    // Costs
    // -----------------------------------------------------------------

    /// `Card:set_cost()`; `area` is the card's current area (for coupons).
    pub(crate) fn set_cost(&self, c: &mut Card, area: Option<Area>) {
        // `G.GAME.inflation` only grows in a challenge.
        c.extra_cost = c.edition.extra_cost();
        c.cost = 1f64.max(((c.base_cost + c.extra_cost + 0.5) * (100.0 - self.discount_percent) / 100.0).floor());
        let celestial = matches!(c.kind, Kind::Booster(b) if b.kind() == PackKind::Celestial);
        if (matches!(c.kind, Kind::Planet(_)) || celestial) && self.has_joker(Joker::Astronomer) {
            c.cost = 0.0;
        }
        if c.ab.rental {
            c.cost = 1.0;
        }
        c.sell_cost = 1f64.max((c.cost / 2.0).floor()) + c.ab.extra_value;
        if c.ab.couponed && matches!(area, Some(Area::ShopJokers | Area::ShopBoosters)) {
            c.cost = 0.0;
        }
    }

    /// Re-runs `set_cost` on every card in existence (the `G.I.CARD` loops).
    pub(crate) fn set_cost_all(&mut self) {
        macro_rules! redo {
            ($f:ident, $a:expr) => {{
                let mut v = std::mem::take(&mut self.$f);
                for c in v.iter_mut() {
                    self.set_cost(c, $a);
                }
                self.$f = v;
            }};
        }
        redo!(jokers, Some(Area::Jokers));
        redo!(consumables, Some(Area::Consumables));
        redo!(shop_jokers, Some(Area::ShopJokers));
        redo!(shop_vouchers, Some(Area::ShopVouchers));
        redo!(shop_boosters, Some(Area::ShopBoosters));
        redo!(pack_cards, Some(Area::PackCards));
        redo!(deck, Some(Area::Deck));
        redo!(hand, Some(Area::Hand));
        redo!(play, Some(Area::Play));
        redo!(discard, Some(Area::Discard));
    }

    // -----------------------------------------------------------------
    // Card construction (Card:init / set_ability / set_base)
    // -----------------------------------------------------------------

    /// Initial ability values for a center (`Card:set_ability`).
    fn init_ability(&mut self, c: &mut Card) {
        let hp = self.hands_played;
        let mut ab = Ability { hands_played_at_create: hp, ..Ability::default() };
        // Playing-card fields carried over by set_ability.
        ab.perma_bonus = c.ab.perma_bonus;
        ab.forced_selection = c.ab.forced_selection;
        if let Kind::Joker(j) = c.kind {
            use Joker as J;
            match j {
                J::Joker => ab.mult = 4.0,
                J::Popcorn => ab.mult = 20.0,
                J::Swashbuckler => ab.mult = 1.0,
                J::Constellation | J::Vampire | J::Hologram | J::Obelisk | J::LuckyCat | J::Glass => ab.x_mult = 1.0,
                J::Ramen => ab.x_mult = 2.0,
                J::Duo | J::Tribe => ab.x_mult = 2.0,
                J::Trio | J::Order => ab.x_mult = 3.0,
                J::Family => ab.x_mult = 4.0,
                J::IceCream => ab.chips = 100.0,
                J::Selzer => ab.remaining = 10.0,
                J::TurtleBean => ab.remaining = 5.0,
                J::Rocket => ab.dollars = 1.0,
                J::LoyaltyCard => ab.loyalty_remaining = 5.0,
                J::Caino => ab.caino_xmult = 1.0,
                J::Yorick => ab.yorick_discards = 23.0,
                _ => {}
            }
            if j == J::TodoList {
                // pseudorandom_element over visible hands (pairs order).
                let (hands, n) = self.visible_hands_pairs(None);
                let k = Key::from("to_do");
                let i = self.rng.element(&k, n);
                ab.to_do = Some(hands[i]);
            }
        }
        c.ab = ab;
        c.base_cost = match c.kind {
            Kind::Playing => 1.0,
            Kind::Joker(j) => j.def().cost as f64,
            Kind::Tarot(t) => t.def().cost as f64,
            Kind::Planet(p) => p.def().cost as f64,
            Kind::Spectral(s) => s.def().cost as f64,
            Kind::Voucher(v) => v.def().cost as f64,
            Kind::Booster(b) => b.def().cost as f64,
        };
        // G.GAME.used_jokers[k] = true for every center with this name.
        self.mark_used(c);
    }

    pub(crate) fn mark_used(&mut self, c: &Card) {
        match c.kind {
            Kind::Playing => {}
            Kind::Booster(b) => {
                // Names are shared between the numbered variants of a pack.
                let name = b.name();
                for o in ALL_BOOSTERS {
                    if o.name() == name {
                        self.used_jokers.insert(Center::Booster(o));
                    }
                }
            }
            _ => self.used_jokers.insert(c.center()),
        }
    }

    /// Clears `used_jokers` when a card leaves the game (`Card:remove`).
    pub(crate) fn unmark_used(&mut self, c: &Card) {
        let name = c.name();
        let held = self.jokers.iter().chain(self.consumables.iter()).any(|o| o.name() == name);
        if held {
            return;
        }
        match c.kind {
            Kind::Playing => {}
            Kind::Booster(b) => {
                for o in ALL_BOOSTERS {
                    if o.name() == b.name() {
                        self.used_jokers.remove(Center::Booster(o));
                    }
                }
            }
            _ => self.used_jokers.remove(c.center()),
        }
    }

    /// A brand new non-playing card of the given kind (Card:init).
    pub(crate) fn make_card(&mut self, kind: Kind) -> Card {
        let mut c = self.new_card(kind);
        self.init_ability(&mut c);
        self.set_cost(&mut c, None);
        c
    }

    /// A new playing card (`Card:init` with a front and a center).
    pub(crate) fn make_playing_card(&mut self, suit: Suit, rank: Rank, enh: Enhancement) -> Card {
        let mut c = self.new_card(Kind::Playing);
        c.enhancement = enh;
        self.init_ability(&mut c);
        c.suit = suit;
        c.rank = rank;
        c.orig_suit = Some(suit);
        self.set_cost(&mut c, None);
        c
    }

    /// `Card:set_ability(center)` on an existing playing card.
    pub(crate) fn set_enhancement(&mut self, c: &mut Card, e: Enhancement) {
        // set_ability rebuilds `ability`, keeping only perma_bonus and
        // forced_selection (so e.g. `played_this_ante` is lost).
        c.enhancement = e;
        self.init_ability(c);
        self.debuff_card(c);
    }

    /// `Card:set_base(card)` on an existing playing card (rebuilds `base`,
    /// which also resets `times_played`).
    pub(crate) fn set_base(&mut self, c: &mut Card, suit: Suit, rank: Rank) {
        c.suit = suit;
        c.rank = rank;
        c.times_played = 0;
        if c.orig_suit.is_none() {
            c.orig_suit = Some(suit);
        }
        self.debuff_card(c);
    }

    /// `Card:set_edition(edition)`; `added` says whether the card is
    /// currently "added to deck" (adjusts Negative slot counts).
    pub(crate) fn set_edition(&mut self, c: &mut Card, e: Edition, area: Option<Area>) {
        let had = c.edition != Edition::None;
        c.edition = e;
        if e == Edition::Negative && !had && c.added_to_deck {
            self.negative_limit(c.kind).limit += 1.0;
        }
        self.set_cost(c, area);
    }


    /// `Card:set_perishable`.
    pub(crate) fn set_perishable(&self, c: &mut Card) {
        c.ab.perishable = false;
        if let Some(j) = c.joker() && j.def().perishable_compat && !c.ab.eternal {
            c.ab.perishable = true;
            c.ab.perish_tally = self.perishable_rounds;
        }
    }

    // -----------------------------------------------------------------
    // Deck membership (Card:add_to_deck / remove_from_deck / remove)
    // -----------------------------------------------------------------

    /// The card area slot count a Negative `kind` adds.
    fn negative_limit(&mut self, kind: Kind) -> &mut Limit {
        if kind.is_consumable() { &mut self.consumable_limit } else { &mut self.joker_limit }
    }

    /// `Card:add_to_deck(from_debuff)`.
    pub(crate) fn add_to_deck(&mut self, c: &mut Card, from_debuff: bool) {
        if c.added_to_deck {
            return;
        }
        c.added_to_deck = true;
        if c.is_playing() {
            return;
        }
        if let Kind::Joker(j) = c.kind {
            use Joker as J;
            let h_size = joker_hand_size(j);
            if h_size != 0.0 {
                self.change_hand_size(h_size);
            }
            let d_size = joker_discards(j);
            if d_size > 0.0 {
                self.rr_discards += d_size;
                self.push(crate::events::Ev::EaseDiscard(d_size));
            }
            match j {
                J::CreditCard => self.bankrupt_at -= 20.0,
                J::Chicot => {
                    if self.blind.boss && !self.blind.disabled {
                        self.disable_blind();
                        self.status_extra();
                    }
                }
                J::Chaos => {
                    self.free_rerolls += 1.0;
                    self.calculate_reroll_cost(true);
                }
                J::TurtleBean => self.change_hand_size(c.ab.remaining),
                J::Oops => self.probabilities *= 2.0,
                J::ToTheMoon => self.interest_amount += 1.0,
                J::Astronomer => self.push(crate::events::Ev::SetCostAll),
                J::Troubadour => {
                    self.change_hand_size(2.0);
                    self.rr_hands += -1.0;
                }
                J::Stuntman => self.change_hand_size(-2.0),
                _ => {}
            }
        }
        if c.edition == Edition::Negative {
            if from_debuff {
                c.queue_negative_removal = false;
            } else {
                self.negative_limit(c.kind).limit += 1.0;
            }
        }
        self.push(crate::events::Ev::RefreshDebuffs);
    }

    /// `Card:remove_from_deck(from_debuff)`.
    pub(crate) fn remove_from_deck(&mut self, c: &mut Card, from_debuff: bool) {
        if !c.added_to_deck {
            return;
        }
        c.added_to_deck = false;
        if let Kind::Joker(j) = c.kind {
            use Joker as J;
            let h_size = joker_hand_size(j);
            if h_size != 0.0 {
                self.change_hand_size(-h_size);
            }
            let d_size = joker_discards(j);
            if d_size > 0.0 {
                self.rr_discards -= d_size;
                self.push(crate::events::Ev::EaseDiscard(-d_size));
            }
            match j {
                J::CreditCard => self.bankrupt_at += 20.0,
                J::Chaos => {
                    self.free_rerolls -= 1.0;
                    self.calculate_reroll_cost(true);
                }
                J::TurtleBean => self.change_hand_size(-c.ab.remaining),
                J::Oops => self.probabilities /= 2.0,
                J::ToTheMoon => self.interest_amount -= 1.0,
                J::Astronomer => self.push(crate::events::Ev::SetCostAll),
                J::Troubadour => {
                    self.change_hand_size(-2.0);
                    self.rr_hands -= -1.0;
                }
                J::Stuntman => self.change_hand_size(2.0),
                _ => {}
            }
        }
        if c.edition == Edition::Negative {
            if from_debuff {
                c.queue_negative_removal = true;
            } else {
                self.negative_limit(c.kind).limit -= 1.0;
            }
        }
        self.push(crate::events::Ev::RefreshDebuffs);
    }

    /// `Card:remove()` for a card already taken out of its area.
    pub(crate) fn destroy_card(&mut self, mut c: Card) {
        self.remove_from_deck(&mut c, false);
        if c.queue_negative_removal {
            self.negative_limit(c.kind).limit -= 1.0;
        }
        self.unmark_used(&c);
    }

    /// `Card:remove()` for a card the player does not own (shop or pack).
    pub(crate) fn drop_unowned(&mut self, c: Card) {
        self.unmark_used(&c);
    }

    /// `Card:remove()` for an owned card (timeline-traced).
    pub(crate) fn remove_owned(&mut self, c: Card) {
        if c.added_to_deck || c.is_playing() {
            crate::game::trace!(self, "rm {}", c.id);
        }
        self.destroy_card(c);
    }

    /// `Card:set_debuff(should_debuff)` for a joker in the joker area.
    pub(crate) fn set_debuff_joker(&mut self, idx: usize, should: bool) {
        let mut c = self.jokers[idx].clone();
        if c.ab.perishable && c.ab.perish_tally <= 0.0 {
            if !c.debuff {
                c.debuff = true;
                self.remove_from_deck(&mut c, true);
            }
            self.jokers[idx] = c;
            return;
        }
        if should != c.debuff {
            if should {
                self.remove_from_deck(&mut c, true);
            } else {
                self.add_to_deck(&mut c, true);
            }
            c.debuff = should;
        }
        self.jokers[idx] = c;
    }

    /// `Card:set_debuff` for a playing card (no deck side effects).
    #[inline]
    pub(crate) fn set_debuff_playing(c: &mut Card, should: bool) {
        if c.ab.perishable && c.ab.perish_tally <= 0.0 {
            c.debuff = true;
            return;
        }
        c.debuff = should;
    }

    // -----------------------------------------------------------------
    // Resource easing
    // -----------------------------------------------------------------

    #[inline]
    pub(crate) fn ease_dollars(&mut self, d: f64) {
        self.dollars += d;
    }

    /// `ease_discard(mod)` (applied immediately).
    pub(crate) fn ease_discard(&mut self, m: f64) {
        if self.discards_left.max(m).abs() == 0.0 {
            return;
        }
        let m = (-self.discards_left).max(m);
        self.discards_left += m;
    }

    #[inline]
    pub(crate) fn ease_hands_played(&mut self, m: f64) {
        self.hands_left += m;
    }

    /// `G.hand:change_size(delta)`.
    pub(crate) fn change_hand_size(&mut self, delta: f64) {
        if delta != 0.0 {
            self.push(crate::events::Ev::HandSize(delta));
        }
    }

    /// `calculate_reroll_cost(skip_increment)`.
    pub(crate) fn calculate_reroll_cost(&mut self, skip_increment: bool) {
        if self.free_rerolls < 0.0 {
            self.free_rerolls = 0.0;
        }
        if self.free_rerolls > 0.0 {
            self.reroll_cost = 0.0;
            return;
        }
        if !skip_increment {
            self.reroll_cost_increase += 1.0;
        }
        self.reroll_cost = self.temp_reroll_cost.unwrap_or(self.rr_reroll_cost) + self.reroll_cost_increase;
    }

    /// `level_up_hand(card, hand, instant, amount)`.
    pub(crate) fn level_up_hand(&mut self, h: HandType, amount: f64) {
        let (s_chips, s_mult, l_chips, l_mult, _) = h.base();
        let hl = &mut self.hands[h as usize];
        hl.level = (hl.level + amount).max(0.0);
        hl.mult = (s_mult + l_mult * (hl.level - 1.0)).max(1.0);
        hl.chips = (s_chips + l_chips * (hl.level - 1.0)).max(0.0);
    }
}

/// Hand size a joker adds while owned (`ability.h_size`).
fn joker_hand_size(j: Joker) -> f64 {
    match j {
        Joker::Juggler => 1.0,
        Joker::MerryAndy => -1.0,
        _ => 0.0,
    }
}

/// Discards per round a joker adds while owned (`ability.d_size`).
fn joker_discards(j: Joker) -> f64 {
    match j {
        Joker::Drunkard => 1.0,
        Joker::MerryAndy => 3.0,
        _ => 0.0,
    }
}
