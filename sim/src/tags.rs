//! Port of `tag.lua` (`Tag:apply_to_run`) and `add_tag`.

use crate::card::Kind;
use crate::data::*;
use crate::events::{Ev, TagFn};
use crate::game::{Area, CardSet, Game, Tag};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum TagCtx {
    Immediate,
    NewBlindChoice,
    VoucherAdd,
    TagAdd { kind: TagKind, orbital: Option<HandType> },
    RoundStartBonus,
    StoreJokerCreate { area: Area },
    ShopStart,
    StoreJokerModify { card: u32 },
    ShopFinalPass,
}

/// `config.type` of each tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TagType {
    Eval,
    Immediate,
    NewBlindChoice,
    VoucherAdd,
    TagAdd,
    RoundStartBonus,
    StoreJokerCreate,
    ShopStart,
    StoreJokerModify,
    ShopFinalPass,
}

fn tag_type(k: TagKind) -> TagType {
    use TagKind as T;
    match k {
        T::Uncommon | T::Rare => TagType::StoreJokerCreate,
        T::Negative | T::Foil | T::Holo | T::Polychrome => TagType::StoreJokerModify,
        T::Investment => TagType::Eval,
        T::Voucher => TagType::VoucherAdd,
        T::Boss | T::Standard | T::Charm | T::Meteor | T::Buffoon | T::Ethereal => TagType::NewBlindChoice,
        T::Handy | T::Garbage | T::TopUp | T::Skip | T::Orbital | T::Economy => TagType::Immediate,
        T::Coupon => TagType::ShopFinalPass,
        T::Double => TagType::TagAdd,
        T::Juggle => TagType::RoundStartBonus,
        T::DSix => TagType::ShopStart,
    }
}

impl TagCtx {
    fn ty(&self) -> TagType {
        match self {
            TagCtx::Immediate => TagType::Immediate,
            TagCtx::NewBlindChoice => TagType::NewBlindChoice,
            TagCtx::VoucherAdd => TagType::VoucherAdd,
            TagCtx::TagAdd { .. } => TagType::TagAdd,
            TagCtx::RoundStartBonus => TagType::RoundStartBonus,
            TagCtx::StoreJokerCreate { .. } => TagType::StoreJokerCreate,
            TagCtx::ShopStart => TagType::ShopStart,
            TagCtx::StoreJokerModify { .. } => TagType::StoreJokerModify,
            TagCtx::ShopFinalPass => TagType::ShopFinalPass,
        }
    }
}

impl Game {
    /// `Tag:yep(..., func)`: the callback event, then removal.
    pub(crate) fn tag_yep(&mut self, idx: usize, which: TagFn) {
        let id = self.tags[idx].id;
        self.push_after(0.4, Ev::Wait);
        self.push(Ev::Wait);
        self.push(Ev::TagFunc { which });
        self.push_after(0.7, Ev::TagRemove { tag_id: id });
    }

    /// `Tag:nope()`.
    fn tag_nope(&mut self, idx: usize) {
        let id = self.tags[idx].id;
        self.push_after(0.2, Ev::Wait);
        self.push_after(0.1, Ev::Wait);
        self.push_after(0.5, Ev::TagRemove { tag_id: id });
    }

    /// `add_tag(tag)`.
    pub(crate) fn add_tag(&mut self, kind: TagKind, orbital: Option<HandType>) {
        for i in 0..self.tags.len() {
            self.apply_tag(i, TagCtx::TagAdd { kind, orbital });
        }
        self.next_tag_id += 1;
        let id = self.next_tag_id;
        self.tags.push(Tag { kind, triggered: false, orbital_hand: orbital, id });
    }

    /// `Tag:apply_to_run({type = 'eval'})` → dollars.
    pub(crate) fn apply_tag_eval(&mut self, idx: usize) -> Option<f64> {
        let t = self.tags[idx];
        if t.triggered || tag_type(t.kind) != TagType::Eval {
            return None;
        }
        if t.kind == TagKind::Investment && self.last_blind_boss {
            self.tag_yep(idx, TagFn::Noop);
            self.tags[idx].triggered = true;
            return Some(25.0);
        }
        None
    }

    /// `Tag:apply_to_run(context)`; returns the Lua truthiness of the result.
    pub(crate) fn apply_tag(&mut self, idx: usize, ctx: TagCtx) -> bool {
        self.apply_tag_card(idx, ctx).0
    }

    /// Like [`apply_tag`], also returning a created shop card (Rare/Uncommon).
    pub(crate) fn apply_tag_card(&mut self, idx: usize, ctx: TagCtx) -> (bool, Option<crate::card::Card>) {
        let t = self.tags[idx];
        if t.triggered || tag_type(t.kind) != ctx.ty() {
            return (false, None);
        }
        use TagKind as T;
        match ctx {
            TagCtx::Immediate => match t.kind {
                T::TopUp => {
                    self.tag_yep(idx, TagFn::TopUp);
                    self.tags[idx].triggered = true;
                    (true, None)
                }
                T::Skip => {
                    self.tag_yep(idx, TagFn::Noop);
                    self.push(Ev::EaseDollars(self.skips * 5.0));
                    self.tags[idx].triggered = true;
                    (true, None)
                }
                T::Garbage => {
                    self.tag_yep(idx, TagFn::Noop);
                    self.push(Ev::EaseDollars(self.unused_discards * 1.0));
                    self.tags[idx].triggered = true;
                    (true, None)
                }
                T::Handy => {
                    self.tag_yep(idx, TagFn::Noop);
                    self.push(Ev::EaseDollars(self.hands_played * 1.0));
                    self.tags[idx].triggered = true;
                    (true, None)
                }
                T::Economy => {
                    self.tag_yep(idx, TagFn::Noop);
                    self.push(Ev::TagFunc { which: TagFn::Economy });
                    self.tags[idx].triggered = true;
                    (true, None)
                }
                T::Orbital => {
                    if let Some(h) = t.orbital_hand {
                        self.hand_text(0.3);
                        self.level_up_hand_ev(h, 3.0, false);
                        self.hand_text(0.0);
                    }
                    self.tag_yep(idx, TagFn::Noop);
                    self.tags[idx].triggered = true;
                    (true, None)
                }
                _ => (false, None),
            },
            TagCtx::NewBlindChoice => {
                let which = match t.kind {
                    T::Charm => TagFn::Charm,
                    T::Meteor => TagFn::Meteor,
                    T::Ethereal => TagFn::Ethereal,
                    T::Standard => TagFn::Standard,
                    T::Buffoon => TagFn::Buffoon,
                    T::Boss => TagFn::Boss,
                    _ => return (false, None),
                };
                self.tag_yep(idx, which);
                self.tags[idx].triggered = true;
                (true, None)
            }
            TagCtx::VoucherAdd => {
                if t.kind == T::Voucher {
                    self.tag_yep(idx, TagFn::Voucher);
                    self.tags[idx].triggered = true;
                }
                (false, None)
            }
            TagCtx::TagAdd { kind, orbital } => {
                if t.kind == T::Double && kind != T::Double {
                    self.tag_yep(idx, TagFn::Double { kind, orbital });
                    self.tags[idx].triggered = true;
                }
                (false, None)
            }
            TagCtx::RoundStartBonus => {
                if t.kind == T::Juggle {
                    self.tag_yep(idx, TagFn::Noop);
                    self.change_hand_size(3.0);
                    self.temp_handsize = Some(self.temp_handsize.unwrap_or(0.0) + 3.0);
                    self.tags[idx].triggered = true;
                    return (true, None);
                }
                (false, None)
            }
            TagCtx::StoreJokerCreate { area } => {
                let mut card = None;
                if t.kind == T::Rare {
                    let mut seen: Vec<Joker> = Vec::new();
                    for c in &self.jokers {
                        if let Some(j) = c.joker() && j.rarity() == 3 && !seen.contains(&j) {
                            seen.push(j);
                        }
                    }
                    if RARITY_POOL_3.len() > seen.len() {
                        let c = self.create_card(CardSet::Joker, area, false, Some(1.0), false, None, "rta");
                        self.tag_yep(idx, TagFn::RareCard { card: c.id });
                        card = Some(c);
                    } else {
                        self.tag_nope(idx);
                    }
                } else if t.kind == T::Uncommon {
                    let c = self.create_card(CardSet::Joker, area, false, Some(0.9), false, None, "uta");
                    self.tag_yep(idx, TagFn::UncommonCard { card: c.id });
                    card = Some(c);
                }
                self.tags[idx].triggered = true;
                (card.is_some(), card)
            }
            TagCtx::ShopStart => {
                if t.kind == T::DSix && !self.shop_d6ed {
                    self.shop_d6ed = true;
                    self.tag_yep(idx, TagFn::D6);
                    self.tags[idx].triggered = true;
                    return (true, None);
                }
                (false, None)
            }
            TagCtx::StoreJokerModify { card } => {
                let Some(c) = self.find_shop_card(card) else { return (false, None) };
                let mut applied = false;
                if c.edition == Edition::None && !c.temp_edition && matches!(c.kind, Kind::Joker(_)) {
                    let ed = match t.kind {
                        T::Foil => Some(Edition::Foil),
                        T::Holo => Some(Edition::Holo),
                        T::Polychrome => Some(Edition::Polychrome),
                        T::Negative => Some(Edition::Negative),
                        _ => None,
                    };
                    if let Some(e) = ed {
                        self.with_shop_card(card, |c| c.temp_edition = true);
                        self.tag_yep(idx, TagFn::Edition { card, edition: e });
                        applied = true;
                    }
                    self.tags[idx].triggered = true;
                }
                (applied, None)
            }
            TagCtx::ShopFinalPass => {
                if t.kind == T::Coupon && self.shop_open && !self.shop_free {
                    self.shop_free = true;
                    self.tag_yep(idx, TagFn::Coupon);
                    self.tags[idx].triggered = true;
                    return (true, None);
                }
                (false, None)
            }
        }
    }

    pub(crate) fn run_tag_event(&mut self, e: Ev) {
        match e {
            Ev::TagRemove { tag_id } => {
                if let Some(i) = self.tags.iter().position(|t| t.id == tag_id) {
                    self.tags.remove(i);
                }
            }
            Ev::TagFunc { which } => match which {
                TagFn::Noop => {}
                TagFn::TopUp => {
                    for _ in 0..2 {
                        if (self.jokers.len() as f64) < self.joker_limit.limit {
                            let mut c = self.create_card(CardSet::Joker, Area::Jokers, false, Some(0.0), false, None, "top");
                            self.add_to_deck(&mut c, false);
                            self.jokers.push(c);
                        }
                    }
                }
                TagFn::Economy => {
                    let d = 40f64.min(self.dollars.max(0.0));
                    self.ease_dollars(d);
                }
                TagFn::Charm => self.open_tag_pack(Booster::ArcanaMega1),
                TagFn::Meteor => self.open_tag_pack(Booster::CelestialMega1),
                TagFn::Ethereal => self.open_tag_pack(Booster::SpectralNormal1),
                TagFn::Standard => self.open_tag_pack(Booster::StandardMega1),
                TagFn::Buffoon => self.open_tag_pack(Booster::BuffoonMega1),
                TagFn::Boss => self.do_reroll_boss(true),
                TagFn::Voucher => {
                    let v = self.get_next_voucher(true);
                    let c = self.make_card(Kind::Voucher(v));
                    self.materialize(1.0);
                    self.shop_vouchers.push(c);
                }
                TagFn::Double { kind, orbital } => self.add_tag(kind, orbital),
                TagFn::RareCard { card } | TagFn::UncommonCard { card } => {
                    self.materialize(1.0);
                    self.with_shop_card(card, |c| c.ab.couponed = true);
                    self.recost_shop_card(card);
                }
                TagFn::Edition { card, edition } => {
                    // Bought before the tag resolved: the edition still lands.
                    let at = self.locate_shop_card(card).or_else(|| self.jokers.iter().position(|c| c.id == card).map(|i| (Area::Jokers, i)));
                    if let Some((area, i)) = at {
                        let mut c = self.area(area)[i].clone();
                        self.set_edition(&mut c, edition, Some(area));
                        self.edition_fx(edition, true);
                        c.ab.couponed = true;
                        self.set_cost(&mut c, Some(area));
                        c.temp_edition = false;
                        self.area_mut(area)[i] = c;
                    }
                }
                TagFn::D6 => {
                    self.temp_reroll_cost = Some(0.0);
                    self.calculate_reroll_cost(true);
                }
                TagFn::Coupon => {
                    if self.shop_open {
                        for area in [Area::ShopJokers, Area::ShopBoosters] {
                            for i in 0..self.area(area).len() {
                                self.area_mut(area)[i].ab.couponed = true;
                                self.reprice(area, i);
                            }
                        }
                    }
                }
            },
            _ => unreachable!(),
        }
    }
}
