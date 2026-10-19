//! The factored action space: legal ops, decoding a choice into an
//! [`Action`], and a fallback.

use super::*;
use crate::actions::{Action, CardMask};
use crate::card::Card;
use crate::consumables::{consumable_highlight, uses_targets};
use crate::game::Game;

/// One legal op.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Op {
    pub kind: i32,
    /// Token index of the op's subject (-1 = none).
    pub target: i32,
    /// Index within its area (shop slot, joker index, ...).
    pub index: u8,
    /// Card-set argument: `card_max == 0` means none.
    pub card_min: u8,
    pub card_max: u8,
    /// Cards that may be chosen / must be included (forced selections).
    pub allowed: CardMask,
    pub required: CardMask,
    /// Destination slots (reordering ops), 0 = none.
    pub dest: u8,
    /// `OP_FLOAT` features (zero except for `PLAY_TOP`).
    pub feats: [f32; OP_FLOAT],
}

impl Op {
    fn simple(kind: i32) -> Op {
        Op { kind, target: -1, index: 0, card_min: 0, card_max: 0, allowed: 0, required: 0, dest: 0, feats: [0.0; OP_FLOAT] }
    }
    fn on(kind: i32, target: usize, index: usize) -> Op {
        Op { target: target as i32, index: index as u8, ..Op::simple(kind) }
    }
    /// The op as `OP_INT` integers (masks reinterpreted as `i32`).
    pub fn ints(&self) -> [i32; OP_INT] {
        [
            self.kind,
            self.target,
            self.card_min as i32,
            self.card_max as i32,
            self.allowed as i32,
            self.required as i32,
            self.dest as i32,
            self.index as i32,
        ]
    }
}

fn hand_mask(n: usize) -> CardMask {
    if n >= 32 { u32::MAX } else { (1u32 << n) - 1 }
}

/// Legal ops of `g` (given the layout of its encoded observation). In the
/// round phase up to `candidates` scored `PLAY_TOP` ops come first, in the
/// room the other ops leave. Ops on subjects past the token limit (which
/// the observation does not show) are dropped.
pub fn ops(g: &Game, lay: &Layout, candidates: usize, out: &mut Vec<Op>) {
    out.clear();
    base_ops(g, lay, out);
    out.retain(|o| o.target < lay.n_tokens as i32);
    out.truncate(MAX_OPS);
    if candidates > 0 && g.phase == Phase::Round {
        let need = (g.blind.chips - g.chips).max(1.0);
        let top = top_plays(g, candidates.min(MAX_OPS - out.len()));
        out.splice(
            0..0,
            top.into_iter().map(|c| {
                let n = c.mask.count_ones() as u8;
                Op {
                    card_min: n,
                    card_max: n,
                    allowed: c.mask,
                    required: c.mask,
                    feats: [sl(c.mean), (c.mean / need).min(4.0) as f32, c.p_clear as f32, sl(c.min), sl(c.max), n as f32 / 5.0],
                    ..Op::simple(op::PLAY_TOP)
                }
            }),
        );
    }
}

/// The ops other than scored candidates, in their fixed order.
fn base_ops(g: &Game, lay: &Layout, out: &mut Vec<Op>) {
    let nh = g.hand.len().min(MAX_HAND);
    let all = hand_mask(nh);
    let forced = g.forced_mask();
    let ok = |a: Action| g.check(&a).is_ok();
    // Card-set ops with targets: find the legal set sizes with
    // representative selections (legality only depends on the count).
    let targeted = |c: &Card, mk: &dyn Fn(CardMask) -> Action, base: Op, out: &mut Vec<Op>| {
        let (max_h, min_h) = consumable_highlight(c.kind);
        if !uses_targets(c.kind) {
            if ok(mk(0)) {
                out.push(base);
            }
            return;
        }
        let max = if max_h == 0 { 1 } else { max_h.min(5) };
        let allowed = if c.kind == Kind::Spectral(Spectral::Aura) {
            g.hand.iter().take(MAX_HAND).enumerate().filter(|(_, c)| c.edition == Edition::None).fold(0u32, |m, (i, _)| m | 1 << i)
        } else {
            all
        };
        let (mut lo, mut hi) = (0u8, 0u8);
        for k in min_h.max(1)..=max {
            // forced cards first, then the lowest allowed others
            let mut m = forced;
            let mut i = 0;
            while (m.count_ones() as u8) < k && i < nh {
                if allowed >> i & 1 == 1 {
                    m |= 1 << i;
                }
                i += 1;
            }
            if m.count_ones() as u8 == k && ok(mk(m)) {
                if lo == 0 {
                    lo = k;
                }
                hi = k;
            }
        }
        if lo > 0 {
            out.push(Op { card_min: lo, card_max: hi, allowed: allowed | forced, required: forced, ..base });
        }
    };
    match g.phase {
        Phase::GameOver => return,
        Phase::BlindSelect => {
            for (a, k) in [(Action::SelectBlind, op::SELECT_BLIND), (Action::SkipBlind, op::SKIP_BLIND), (Action::RerollBoss, op::REROLL_BOSS)] {
                if ok(a) {
                    out.push(Op::simple(k));
                }
            }
        }
        Phase::Round => {
            if nh > 0 && forced.count_ones() <= 5 {
                let base = Op { card_min: 1, card_max: 5, allowed: all, required: forced, ..Op::simple(op::PLAY) };
                out.push(base);
                if g.discards_left > 0.0 {
                    out.push(Op { kind: op::DISCARD, ..base });
                }
            }
        }
        Phase::Shop => {
            for i in 0..g.shop_jokers.len() {
                if ok(Action::BuyShopCard(i as u8)) {
                    out.push(Op::on(op::BUY, lay.shop_cards + i, i));
                }
                let c = &g.shop_jokers[i];
                if c.kind.is_consumable() {
                    targeted(c, &|t| Action::BuyAndUse { index: i as u8, targets: t }, Op::on(op::BUY_AND_USE, lay.shop_cards + i, i), out);
                }
            }
            for i in 0..g.shop_vouchers.len() {
                if ok(Action::RedeemVoucher(i as u8)) {
                    out.push(Op::on(op::REDEEM, lay.shop_vouchers + i, i));
                }
            }
            for i in 0..g.shop_boosters.len() {
                if ok(Action::OpenBooster(i as u8)) {
                    out.push(Op::on(op::OPEN, lay.shop_boosters + i, i));
                }
            }
            if ok(Action::Reroll) {
                out.push(Op::simple(op::REROLL));
            }
            out.push(Op::simple(op::END_SHOP));
        }
        Phase::Pack => {
            for i in 0..g.pack_cards.len() {
                let c = &g.pack_cards[i];
                let base = Op::on(op::PICK, lay.pack + i, i);
                if c.kind.is_consumable() {
                    targeted(c, &|t| Action::PickPackCard { index: i as u8, targets: t }, base, out);
                } else if ok(Action::PickPackCard { index: i as u8, targets: 0 }) {
                    out.push(base);
                }
            }
            if ok(Action::SkipPack) {
                out.push(Op::simple(op::SKIP_PACK));
            }
        }
        Phase::RoundEval => out.push(Op::simple(op::CASH_OUT)),
    }
    // Reordering the hand (in rounds, and in packs opened with it drawn).
    if matches!(g.phase, Phase::Round | Phase::Pack) && nh >= 2 {
        out.push(Op::simple(op::SORT_RANK));
        out.push(Op::simple(op::SORT_SUIT));
        out.push(Op { card_min: 1, card_max: 1, allowed: all, required: 0, dest: nh as u8, ..Op::simple(op::MOVE_HAND) });
    }
    for i in 0..g.consumables.len() {
        let c = &g.consumables[i];
        targeted(c, &|t| Action::UseConsumable { index: i as u8, targets: t }, Op::on(op::USE, lay.consumables + i, i), out);
    }
    for i in 0..g.jokers.len() {
        if ok(Action::SellJoker(i as u8)) {
            out.push(Op::on(op::SELL_JOKER, lay.jokers + i, i));
        }
    }
    for i in 0..g.consumables.len() {
        if ok(Action::SellConsumable(i as u8)) {
            out.push(Op::on(op::SELL_CONSUMABLE, lay.consumables + i, i));
        }
    }
    if g.jokers.len() >= 2 {
        for i in 0..g.jokers.len() {
            out.push(Op { dest: g.jokers.len() as u8, ..Op::on(op::MOVE_JOKER, lay.jokers + i, i) });
        }
    }
}

/// The [`Action`] for `op` with the chosen cards and destination, if legal.
/// `cards` may omit the op's required cards (they are always added).
pub fn decode(g: &Game, o: &Op, cards: CardMask, dest: u32) -> Option<Action> {
    let m = (cards & o.allowed) | o.required;
    let dest = || u8::try_from(dest).ok().filter(|&d| d < o.dest);
    let a = match o.kind {
        op::SELECT_BLIND => Action::SelectBlind,
        op::SKIP_BLIND => Action::SkipBlind,
        op::REROLL_BOSS => Action::RerollBoss,
        op::PLAY | op::PLAY_TOP => Action::Play(m),
        op::DISCARD => Action::Discard(m),
        op::USE => Action::UseConsumable { index: o.index, targets: if o.card_max > 0 { m } else { 0 } },
        op::SELL_JOKER => Action::SellJoker(o.index),
        op::SELL_CONSUMABLE => Action::SellConsumable(o.index),
        op::BUY => Action::BuyShopCard(o.index),
        op::BUY_AND_USE => Action::BuyAndUse { index: o.index, targets: if o.card_max > 0 { m } else { 0 } },
        op::REDEEM => Action::RedeemVoucher(o.index),
        op::OPEN => Action::OpenBooster(o.index),
        op::REROLL => Action::Reroll,
        op::END_SHOP => Action::EndShop,
        op::CASH_OUT => Action::CashOut,
        op::PICK => Action::PickPackCard { index: o.index, targets: if o.card_max > 0 { m } else { 0 } },
        op::SKIP_PACK => Action::SkipPack,
        op::MOVE_JOKER => Action::MoveJoker { from: o.index, to: dest()? },
        op::SORT_RANK => Action::SortHand { by_suit: false },
        op::SORT_SUIT => Action::SortHand { by_suit: true },
        op::MOVE_HAND => {
            if m.count_ones() != 1 {
                return None;
            }
            Action::MoveHand { from: m.trailing_zeros() as u8, to: dest()? }
        }
        _ => return None,
    };
    if o.card_max > 0 {
        let n = m.count_ones() as u8;
        if n < o.card_min || n > o.card_max {
            return None;
        }
    }
    g.check(&a).ok().map(|_| a)
}

/// A legal fallback action (the first op with its minimal card set).
pub fn fallback(g: &Game, ops: &[Op]) -> Option<Action> {
    for o in ops {
        let mut m = o.required;
        let mut i = 0;
        while (m.count_ones() as u8) < o.card_min && i < 32 {
            if o.allowed >> i & 1 == 1 {
                m |= 1 << i;
            }
            i += 1;
        }
        let dest = if o.dest > 0 { (o.index as u32 + 1) % o.dest as u32 } else { 0 };
        if let Some(a) = decode(g, o, m, dest) {
            return Some(a);
        }
    }
    let mut v = Vec::new();
    g.legal_actions(&mut v);
    v.first().copied()
}
