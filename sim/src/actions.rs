//! The agent-facing action interface: enumeration, validation and `step`.

use crate::card::Kind;
use crate::consumables::{consumable_highlight, uses_targets};
use crate::data::*;
use crate::events::St;
use crate::game::{Area, BlindSlot, Game, Phase};
use std::fmt;
use std::str::FromStr;

/// Hand-card selection as a bitmask over `Game::hand` indices.
pub type CardMask = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Action {
    SelectBlind,
    SkipBlind,
    RerollBoss,
    Play(CardMask),
    Discard(CardMask),
    /// Use consumable `index` with the given hand cards highlighted.
    UseConsumable { index: u8, targets: CardMask },
    SellJoker(u8),
    SellConsumable(u8),
    MoveJoker { from: u8, to: u8 },
    MoveHand { from: u8, to: u8 },
    SortHand { by_suit: bool },
    BuyShopCard(u8),
    BuyAndUse { index: u8, targets: CardMask },
    RedeemVoucher(u8),
    OpenBooster(u8),
    Reroll,
    EndShop,
    CashOut,
    PickPackCard { index: u8, targets: CardMask },
    SkipPack,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActionError {
    WrongPhase,
    BadIndex,
    BadSelection,
    CannotAfford,
    NoSpace,
    NotAllowed,
}

impl fmt::Display for ActionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ActionError::WrongPhase => "not possible in the current phase",
            ActionError::BadIndex => "no card at that index",
            ActionError::BadSelection => "invalid card selection",
            ActionError::CannotAfford => "not enough money",
            ActionError::NoSpace => "no room for the card",
            ActionError::NotAllowed => "not allowed now",
        })
    }
}
impl std::error::Error for ActionError {}

#[inline]
pub(crate) fn mask_indices(m: CardMask) -> Vec<usize> {
    (0..CardMask::BITS as usize).filter(|i| m >> i & 1 == 1).collect()
}

/// Whether `m` only selects cards of a hand of `n` cards.
#[inline]
pub(crate) fn mask_fits(m: CardMask, n: usize) -> bool {
    m.checked_shr(n as u32).unwrap_or(0) == 0
}

impl Game {
    #[inline]
    fn spendable(&self) -> f64 {
        self.dollars - self.bankrupt_at
    }

    /// Hand cards that must be part of any selection (Cerulean Bell).
    pub(crate) fn forced_mask(&self) -> CardMask {
        let mut m = 0;
        for (i, c) in self.hand.iter().take(CardMask::BITS as usize).enumerate() {
            if c.ab.forced_selection {
                m |= 1 << i;
            }
        }
        m
    }

    fn valid_selection(&self, m: CardMask, max: u32) -> bool {
        let n = m.count_ones();
        let forced = self.forced_mask();
        n >= 1 && n <= max && mask_fits(m, self.hand.len()) && (m & forced) == forced
    }

    fn consumable_usable(&self, c: &crate::card::Card, area: Area, targets: CardMask) -> bool {
        if !mask_fits(targets, self.hand.len()) {
            return false;
        }
        // Only consumables that use targets must include the forced card.
        let uses_targets = uses_targets(c.kind);
        let forced = self.forced_mask();
        if uses_targets && targets & forced != forced {
            return false;
        }
        let n = targets.count_ones() as usize;
        let first_editionless = targets != 0 && self.hand[targets.trailing_zeros() as usize].edition == Edition::None;
        if !self.can_use_consumable(c, Some(area), n, first_editionless) {
            return false;
        }
        // Ankh refuses to resolve with a full joker area (`Card:check_use`).
        if c.kind == Kind::Spectral(Spectral::Ankh) && self.jokers.len() as f64 >= self.joker_limit.limit {
            return false;
        }
        // Only send targets to consumables that use them (extra highlighted
        // cards are harmless in the game; the action space stays canonical).
        if !uses_targets && targets != 0 {
            return false;
        }
        true
    }

    /// Checks an action without applying it.
    pub fn check(&self, a: &Action) -> Result<(), ActionError> {
        use Action as A;
        let ph = self.phase;
        let need = |p: Phase| if ph == p { Ok(()) } else { Err(ActionError::WrongPhase) };
        let not_over = || if ph == Phase::GameOver { Err(ActionError::WrongPhase) } else { Ok(()) };
        match *a {
            A::SelectBlind => need(Phase::BlindSelect),
            A::SkipBlind => {
                need(Phase::BlindSelect)?;
                match self.blind_on_deck {
                    Some(BlindSlot::Small | BlindSlot::Big) => Ok(()),
                    _ => Err(ActionError::NotAllowed),
                }
            }
            A::RerollBoss => {
                need(Phase::BlindSelect)?;
                let ok = self.voucher_used(Voucher::Retcon) || (self.voucher_used(Voucher::DirectorsCut) && !self.boss_rerolled);
                if !ok {
                    return Err(ActionError::NotAllowed);
                }
                if self.spendable() - 10.0 < 0.0 {
                    return Err(ActionError::CannotAfford);
                }
                Ok(())
            }
            A::Play(m) => {
                need(Phase::Round)?;
                if !self.valid_selection(m, 5) {
                    return Err(ActionError::BadSelection);
                }
                Ok(())
            }
            A::Discard(m) => {
                need(Phase::Round)?;
                if self.discards_left <= 0.0 {
                    return Err(ActionError::NotAllowed);
                }
                if !self.valid_selection(m, 5) {
                    return Err(ActionError::BadSelection);
                }
                Ok(())
            }
            A::UseConsumable { index, targets } => {
                not_over()?;
                let c = self.consumables.get(index as usize).ok_or(ActionError::BadIndex)?;
                if !self.consumable_usable(c, Area::Consumables, targets) {
                    return Err(ActionError::NotAllowed);
                }
                Ok(())
            }
            A::SellJoker(i) => {
                not_over()?;
                let c = self.jokers.get(i as usize).ok_or(ActionError::BadIndex)?;
                if c.ab.eternal {
                    return Err(ActionError::NotAllowed);
                }
                Ok(())
            }
            A::SellConsumable(i) => {
                not_over()?;
                let c = self.consumables.get(i as usize).ok_or(ActionError::BadIndex)?;
                if c.ab.eternal {
                    return Err(ActionError::NotAllowed);
                }
                Ok(())
            }
            A::MoveJoker { from, to } => {
                not_over()?;
                if from as usize >= self.jokers.len() || to as usize >= self.jokers.len() || from == to {
                    return Err(ActionError::BadIndex);
                }
                Ok(())
            }
            A::MoveHand { from, to } => {
                if !matches!(ph, Phase::Round | Phase::Pack) {
                    return Err(ActionError::WrongPhase);
                }
                if from as usize >= self.hand.len() || to as usize >= self.hand.len() || from == to {
                    return Err(ActionError::BadIndex);
                }
                Ok(())
            }
            A::SortHand { .. } => {
                if !matches!(ph, Phase::Round | Phase::Pack) || self.hand.is_empty() {
                    return Err(ActionError::WrongPhase);
                }
                Ok(())
            }
            A::BuyShopCard(i) => {
                need(Phase::Shop)?;
                let c = self.shop_jokers.get(i as usize).ok_or(ActionError::BadIndex)?;
                if c.cost > self.spendable() && c.cost > 0.0 {
                    return Err(ActionError::CannotAfford);
                }
                if !self.has_buy_space(c) {
                    return Err(ActionError::NoSpace);
                }
                Ok(())
            }
            A::BuyAndUse { index, targets } => {
                need(Phase::Shop)?;
                let c = self.shop_jokers.get(index as usize).ok_or(ActionError::BadIndex)?;
                if !c.kind.is_consumable() {
                    return Err(ActionError::NotAllowed);
                }
                if c.cost > self.spendable() && c.cost > 0.0 {
                    return Err(ActionError::CannotAfford);
                }
                if !self.consumable_usable(c, Area::ShopJokers, targets) {
                    return Err(ActionError::NotAllowed);
                }
                Ok(())
            }
            A::RedeemVoucher(i) => {
                need(Phase::Shop)?;
                let c = self.shop_vouchers.get(i as usize).ok_or(ActionError::BadIndex)?;
                if c.cost > self.spendable() {
                    return Err(ActionError::CannotAfford);
                }
                Ok(())
            }
            A::OpenBooster(i) => {
                need(Phase::Shop)?;
                let c = self.shop_boosters.get(i as usize).ok_or(ActionError::BadIndex)?;
                if c.cost > 0.0 && c.cost > self.spendable() {
                    return Err(ActionError::CannotAfford);
                }
                Ok(())
            }
            A::Reroll => {
                need(Phase::Shop)?;
                if self.spendable() - self.reroll_cost < 0.0 && self.reroll_cost != 0.0 {
                    return Err(ActionError::CannotAfford);
                }
                Ok(())
            }
            A::EndShop => need(Phase::Shop),
            A::CashOut => need(Phase::RoundEval),
            A::PickPackCard { index, targets } => {
                need(Phase::Pack)?;
                let c = self.pack_cards.get(index as usize).ok_or(ActionError::BadIndex)?;
                match c.kind {
                    k if k.is_consumable() => {
                        if !self.consumable_usable(c, Area::PackCards, targets) {
                            return Err(ActionError::NotAllowed);
                        }
                    }
                    Kind::Joker(_) => {
                        if targets != 0 {
                            return Err(ActionError::BadSelection);
                        }
                        if c.edition != Edition::Negative && self.jokers.len() as f64 >= self.joker_limit.limit {
                            return Err(ActionError::NoSpace);
                        }
                    }
                    _ => {
                        if targets != 0 {
                            return Err(ActionError::BadSelection);
                        }
                    }
                }
                Ok(())
            }
            A::SkipPack => {
                need(Phase::Pack)?;
                let p = self.pack.as_ref().ok_or(ActionError::WrongPhase)?;
                let ok = !self.pack_cards.is_empty()
                    && (matches!(p.kind, PackKind::Celestial | PackKind::Standard | PackKind::Buffoon)
                        || !self.hand.is_empty()
                        || self.hand_limit.limit <= 0.0);
                if ok { Ok(()) } else { Err(ActionError::NotAllowed) }
            }
        }
    }

    /// Applies an action and runs the game until the next decision point.
    pub fn step(&mut self, a: Action) -> Result<(), ActionError> {
        self.check(&a)?;
        self.begin_action();
        use Action as A;
        match a {
            A::SelectBlind => self.do_select_blind(),
            A::SkipBlind => self.do_skip_blind(),
            A::RerollBoss => self.do_reroll_boss(false),
            A::Play(m) => self.do_play(mask_indices(m)),
            A::Discard(m) => self.do_discard(mask_indices(m), false),
            A::UseConsumable { index, targets } => {
                let c = self.consumables.remove(index as usize);
                self.use_card(c, Some(Area::Consumables), &mask_indices(targets));
            }
            A::SellJoker(i) => self.do_sell(Area::Jokers, i as usize),
            A::SellConsumable(i) => self.do_sell(Area::Consumables, i as usize),
            A::MoveJoker { from, to } => {
                let c = self.jokers.remove(from as usize);
                self.jokers.insert(to as usize, c);
            }
            A::MoveHand { from, to } => {
                let c = self.hand.remove(from as usize);
                self.hand.insert(to as usize, c);
            }
            A::SortHand { by_suit } => {
                self.hand_sort_by_suit = by_suit;
                self.sort_hand();
            }
            A::BuyShopCard(i) => self.do_buy(i as usize, false, &[]),
            A::BuyAndUse { index, targets } => self.do_buy(index as usize, true, &mask_indices(targets)),
            A::RedeemVoucher(i) => {
                let c = self.shop_vouchers.remove(i as usize);
                self.use_card(c, Some(Area::ShopVouchers), &[]);
            }
            A::OpenBooster(i) => {
                let c = self.shop_boosters.remove(i as usize);
                self.use_card(c, Some(Area::ShopBoosters), &[]);
            }
            A::Reroll => self.do_reroll(),
            A::EndShop => self.do_end_shop(),
            A::CashOut => self.do_cash_out(),
            A::PickPackCard { index, targets } => {
                let c = self.pack_cards.remove(index as usize);
                self.use_card(c, Some(Area::PackCards), &mask_indices(targets));
            }
            A::SkipPack => self.do_skip_pack(),
        }
        self.run_queue();
        Ok(())
    }

    /// The legal actions at the current decision point, in a stable order:
    /// card subsets for play / discard / consumable targets are enumerated
    /// exhaustively; reordering (`MoveJoker`, `MoveHand`, `SortHand`) is
    /// legal (see [`Game::check`]) but not enumerated.
    pub fn legal_actions(&self, out: &mut Vec<Action>) {
        out.clear();
        let push = |out: &mut Vec<Action>, g: &Game, a: Action| {
            if g.check(&a).is_ok() {
                out.push(a);
            }
        };
        let nh = self.hand.len().min(CardMask::BITS as usize);
        match self.phase {
            Phase::GameOver => return,
            Phase::BlindSelect => {
                push(out, self, Action::SelectBlind);
                push(out, self, Action::SkipBlind);
                push(out, self, Action::RerollBoss);
            }
            Phase::Round => {
                // Same order and rules as `check`, without re-validating.
                let forced = self.forced_mask();
                let can_discard = self.discards_left > 0.0;
                for_each_subset(nh, 5, |m| {
                    if m & forced == forced {
                        out.push(Action::Play(m));
                        if can_discard {
                            out.push(Action::Discard(m));
                        }
                    }
                });
            }
            Phase::Shop => {
                for i in 0..self.shop_jokers.len() {
                    push(out, self, Action::BuyShopCard(i as u8));
                    if self.shop_jokers[i].kind.is_consumable() {
                        self.push_targeted(out, &self.shop_jokers[i], |t| Action::BuyAndUse { index: i as u8, targets: t });
                    }
                }
                for i in 0..self.shop_vouchers.len() {
                    push(out, self, Action::RedeemVoucher(i as u8));
                }
                for i in 0..self.shop_boosters.len() {
                    push(out, self, Action::OpenBooster(i as u8));
                }
                push(out, self, Action::Reroll);
                push(out, self, Action::EndShop);
            }
            Phase::Pack => {
                for i in 0..self.pack_cards.len() {
                    let c = &self.pack_cards[i];
                    if c.kind.is_consumable() {
                        self.push_targeted(out, c, |t| Action::PickPackCard { index: i as u8, targets: t });
                    } else {
                        push(out, self, Action::PickPackCard { index: i as u8, targets: 0 });
                    }
                }
                push(out, self, Action::SkipPack);
            }
            Phase::RoundEval => push(out, self, Action::CashOut),
        }
        for i in 0..self.consumables.len() {
            let c = &self.consumables[i];
            self.push_targeted(out, c, |t| Action::UseConsumable { index: i as u8, targets: t });
        }
        for i in 0..self.jokers.len() {
            push(out, self, Action::SellJoker(i as u8));
        }
        for i in 0..self.consumables.len() {
            push(out, self, Action::SellConsumable(i as u8));
        }
    }

    fn push_targeted(&self, out: &mut Vec<Action>, c: &crate::card::Card, mk: impl Fn(CardMask) -> Action) {
        let (max_h, _) = consumable_highlight(c.kind);
        if !uses_targets(c.kind) {
            let a = mk(0);
            if self.check(&a).is_ok() {
                out.push(a);
            }
            return;
        }
        let max = if max_h == 0 { 1 } else { max_h.min(5) as u32 };
        let nh = self.hand.len().min(CardMask::BITS as usize);
        for_each_subset(nh, max, |m| {
            let a = mk(m);
            if self.check(&a).is_ok() {
                out.push(a);
            }
        });
    }

    /// Whether the run has ended: lost, or won (round evaluation of the
    /// final boss) unless `endless`. Further actions stay possible; stopping
    /// is the caller's decision.
    pub fn is_terminal(&self) -> bool {
        self.phase == Phase::GameOver || (self.run_won() && self.state == St::RoundEval && !self.endless)
    }
}

impl fmt::Display for Action {
    /// The text code used in histories and by the bindings, e.g. `play 11`
    /// (card masks in decimal); parsed back by [`FromStr`].
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Action as A;
        match *self {
            A::SelectBlind => f.write_str("select"),
            A::SkipBlind => f.write_str("skip"),
            A::RerollBoss => f.write_str("reroll_boss"),
            A::Play(m) => write!(f, "play {m}"),
            A::Discard(m) => write!(f, "discard {m}"),
            A::UseConsumable { index, targets } => write!(f, "use {index} {targets}"),
            A::SellJoker(i) => write!(f, "sell_joker {i}"),
            A::SellConsumable(i) => write!(f, "sell_consumable {i}"),
            A::MoveJoker { from, to } => write!(f, "move_joker {from} {to}"),
            A::MoveHand { from, to } => write!(f, "move_hand {from} {to}"),
            A::SortHand { by_suit } => write!(f, "sort {}", by_suit as u8),
            A::BuyShopCard(i) => write!(f, "buy {i}"),
            A::BuyAndUse { index, targets } => write!(f, "buy_use {index} {targets}"),
            A::RedeemVoucher(i) => write!(f, "redeem {i}"),
            A::OpenBooster(i) => write!(f, "open {i}"),
            A::Reroll => f.write_str("reroll"),
            A::EndShop => f.write_str("end_shop"),
            A::CashOut => f.write_str("cash_out"),
            A::PickPackCard { index, targets } => write!(f, "pick {index} {targets}"),
            A::SkipPack => f.write_str("skip_pack"),
        }
    }
}

/// A malformed action or injection code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError(pub String);

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bad code {:?}", self.0)
    }
}
impl std::error::Error for ParseError {}

/// The whitespace-separated operands of a text code, parsed strictly (the
/// right count, each in range).
pub(crate) struct Operands<'a> {
    it: std::str::SplitWhitespace<'a>,
    code: &'a str,
}

impl<'a> Operands<'a> {
    pub(crate) fn new(code: &'a str) -> (Option<&'a str>, Operands<'a>) {
        let mut it = code.split_whitespace();
        (it.next(), Operands { it, code })
    }

    pub(crate) fn err(&self) -> ParseError {
        ParseError(self.code.to_string())
    }

    pub(crate) fn next<T: FromStr>(&mut self) -> Result<T, ParseError> {
        self.it.next().and_then(|t| t.parse().ok()).ok_or_else(|| self.err())
    }

    /// Fails if operands are left over.
    pub(crate) fn end<T>(mut self, v: T) -> Result<T, ParseError> {
        if self.it.next().is_some() { Err(self.err()) } else { Ok(v) }
    }
}

impl FromStr for Action {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Action, ParseError> {
        use Action as A;
        let (op, mut n) = Operands::new(s);
        let a = match op.ok_or_else(|| n.err())? {
            "select" => A::SelectBlind,
            "skip" => A::SkipBlind,
            "reroll_boss" => A::RerollBoss,
            "play" => A::Play(n.next()?),
            "discard" => A::Discard(n.next()?),
            "use" => A::UseConsumable { index: n.next()?, targets: n.next()? },
            "sell_joker" => A::SellJoker(n.next()?),
            "sell_consumable" => A::SellConsumable(n.next()?),
            "move_joker" => A::MoveJoker { from: n.next()?, to: n.next()? },
            "move_hand" => A::MoveHand { from: n.next()?, to: n.next()? },
            "sort" => A::SortHand {
                by_suit: match n.next::<u8>()? {
                    0 => false,
                    1 => true,
                    _ => return Err(n.err()),
                },
            },
            "buy" => A::BuyShopCard(n.next()?),
            "buy_use" => A::BuyAndUse { index: n.next()?, targets: n.next()? },
            "redeem" => A::RedeemVoucher(n.next()?),
            "open" => A::OpenBooster(n.next()?),
            "reroll" => A::Reroll,
            "end_shop" => A::EndShop,
            "cash_out" => A::CashOut,
            "pick" => A::PickPackCard { index: n.next()?, targets: n.next()? },
            "skip_pack" => A::SkipPack,
            _ => return Err(n.err()),
        };
        n.end(a)
    }
}

/// Calls `f` for every non-empty subset of `n` items with at most `max`
/// members, in increasing mask order.
#[inline]
fn for_each_subset(n: usize, max: u32, mut f: impl FnMut(CardMask)) {
    if n == 0 {
        return;
    }
    let end: u64 = 1u64 << n;
    let mut m: u64 = 1;
    while m < end {
        if (m as u32).count_ones() > max {
            // Skip ahead to the next mask with fewer bits set.
            m += m.isolate_lowest_one();
            continue;
        }
        f(m as CardMask);
        m += 1;
    }
}
