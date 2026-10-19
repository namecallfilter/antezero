//! Port of `Card:calculate_joker(context)` (card.lua). The branch structure and
//! evaluation order mirror the Lua if-chain exactly, because the order of
//! RNG draws and of state mutations is observable. This module holds the
//! contexts, the effect table and the dispatch (with Blueprint / Brainstorm
//! copying); the contexts themselves are in `lifecycle` (shop, blind, discard,
//! end of round) and `scoring` (hand scoring).

use crate::card::Kind;
use crate::data::*;
use crate::events::Ev;
use crate::game::{Area, CardSet, Game};
use crate::hand_eval::PokerHands;

/// Shared data of the hand being scored.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Scoring {
    pub name: HandType,
    pub hands: PokerHands,
    /// Indices into `G.play` of the scoring hand, in table order.
    pub scoring: [u8; 8],
    pub n_scoring: u8,
}

impl Scoring {
    #[inline]
    pub fn scoring(&self) -> &[u8] {
        &self.scoring[..self.n_scoring as usize]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CardArea {
    Play,
    Hand,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Ctx {
    OpenBooster,
    SellingSelf,
    SellingCard,
    RerollShop,
    EndingShop,
    SkipBlind,
    SkippingBooster,
    PlayingCardAdded { n: usize },
    FirstHandDrawn,
    SettingBlind { boss: bool },
    DestroyingCard { play_idx: usize, full_len: usize },
    RemovePlayingCards { faces: usize, glass: usize },
    UsingConsumeable { kind: Kind, hanged_glass: usize },
    DebuffedHand,
    PreDiscard { hook: bool },
    /// `other` and `last` index `G.hand`; `n_full` / `full_faces` count the
    /// discarded (highlighted) cards and their face cards.
    Discard { other: usize, last: usize, n_full: usize, full_faces: usize },
    EndOfRound { game_over: bool },
    EndOfRoundRepetition { effects: bool },
    Individual { area: CardArea, other: usize },
    Repetition { area: CardArea, other: usize, effects: bool },
    /// `other`: the joker being scored (`None` for a consumable).
    OtherJoker { other: Option<usize> },
    Before,
    After,
    Main,
}

/// A joker's returned effect table.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Effect {
    pub chips: Option<f64>,
    pub mult: Option<f64>,
    pub x_mult: Option<f64>,
    pub h_mult: Option<f64>,
    pub dollars: Option<f64>,
    pub chip_mod: Option<f64>,
    pub mult_mod: Option<f64>,
    pub xmult_mod: Option<f64>,
    pub repetitions: u32,
    pub level_up: bool,
    pub saved: bool,
    pub remove: bool,
    /// `extra.func` that creates an 8 Ball tarot.
    pub eight_ball: bool,
    /// Has an `extra` table (individual-context bookkeeping).
    pub extra: bool,
    pub playing_cards_created: bool,
    /// `delay` of the returned table (status-text timing).
    pub delay: Option<f64>,
}

macro_rules! ret {
    ($($f:ident : $v:expr),* $(,)?) => {
        return Some(Effect { $($f: $v,)* ..Effect::default() })
    };
}

// After `ret!`, which the context handlers use.
mod lifecycle;
mod scoring;

impl Game {
    #[inline]
    fn jab(&mut self, ji: usize) -> &mut crate::card::Ability {
        &mut self.jokers[ji].ab
    }

    /// Schedules the "eaten/melted" self-destruction of a joker
    /// (`G.jokers:remove_card(self); self:remove()` in nested events).
    fn schedule_self_destruct(&mut self, ji: usize) {
        let id = self.jokers[ji].id;
        self.push(Ev::EatenOuter { id });
    }

    /// Whether the consumable area has room for one more card (counting
    /// cards already promised by queued creations).
    #[inline]
    pub(crate) fn consumable_room(&self) -> bool {
        (self.consumables.len() as f64 + self.consumeable_buffer) < self.consumable_limit.limit
    }

    /// Money promised by a scoring effect before its `ease_dollars` event
    /// runs (`G.GAME.dollar_buffer`).
    #[inline]
    pub(crate) fn buffer_dollars(&mut self, d: f64) {
        self.dollar_buffer += d;
        self.push(Ev::DollarBufferReset);
    }

    /// `Card:calculate_joker(context)` for the joker at `ji`.
    #[inline]
    pub(crate) fn calc_joker(&mut self, ji: usize, ctx: &Ctx, sc: Option<&Scoring>) -> Option<Effect> {
        self.calc_joker_copy(ji, ctx, sc, 0, None)
    }

    /// `calculate_joker` as reached through `bp` Blueprint / Brainstorm
    /// copies on behalf of the joker at `bp_card`.
    fn calc_joker_copy(&mut self, ji: usize, ctx: &Ctx, sc: Option<&Scoring>, bp: u32, bp_card: Option<usize>) -> Option<Effect> {
        let card = &self.jokers[ji];
        if card.debuff {
            return None;
        }
        let Kind::Joker(j) = card.kind else {
            return None;
        };
        use Joker as J;
        let blueprinted = bp > 0;
        let actor = bp_card.unwrap_or(ji);

        // Copycats come first and fall through to their own (empty) logic.
        if j == J::Blueprint || j == J::Brainstorm {
            let other = if j == J::Blueprint {
                if ji + 1 < self.jokers.len() { Some(ji + 1) } else { None }
            } else if !self.jokers.is_empty() {
                Some(0)
            } else {
                None
            };
            if let Some(o) = other && o != ji {
                let depth = bp + 1;
                let bcard = Some(bp_card.unwrap_or(ji));
                if depth as usize > self.jokers.len() + 1 {
                    return None;
                }
                if let Some(r) = self.calc_joker_copy(o, ctx, sc, depth, bcard) {
                    return Some(r);
                }
            }
        }

        match *ctx {
            Ctx::DestroyingCard { .. }
            | Ctx::DebuffedHand
            | Ctx::Individual { .. }
            | Ctx::Repetition { .. }
            | Ctx::OtherJoker { .. }
            | Ctx::Before
            | Ctx::After
            | Ctx::Main => self.joker_scoring(ji, j, ctx, sc, blueprinted),
            _ => self.joker_lifecycle(ji, j, ctx, blueprinted, actor),
        }
    }

    /// `card_eval_status_text(joker, 'jokers', nil, percent, nil, eval)`.
    #[inline]
    pub(crate) fn effect_status(&mut self, e: &Effect) {
        self.status_text(e.delay.unwrap_or(0.75));
    }

    /// `playing_card_joker_effects(cards)`.
    pub(crate) fn playing_card_joker_effects(&mut self, n: usize) {
        for i in 0..self.jokers.len() {
            self.calc_joker(i, &Ctx::PlayingCardAdded { n }, None);
        }
    }
}
