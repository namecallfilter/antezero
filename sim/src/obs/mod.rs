//! Fixed-size numeric observation and factored action space for learning
//! agents.
//!
//! **Observation**: a set of entity tokens (hand, jokers, consumables, shop,
//! pack, tags, blinds, poker-hand levels, remaining deck and discard pile)
//! plus a global feature vector. Only information visible to a player is
//! encoded: the deck is a sorted multiset (draw order is hidden), face-down
//! cards hide their identity, and no RNG state is exposed.
//!
//! **Actions** are factored into an *op* (what to do, optionally on a
//! target token), an optional set of hand cards (chosen by a pointer in
//! increasing index order, so each set has exactly one encoding) and an
//! optional destination slot (for reordering). [`ops`] lists the legal ops
//! with their card constraints; [`decode`] turns a choice back into an
//! [`Action`](crate::Action) and checks it.

use crate::card::Kind;
use crate::data::*;
use crate::game::{Game, Phase};

mod candidates;
mod encode;
mod ops;

pub use candidates::{Candidate, preview_score, top_plays};
pub use encode::{encode, encode_opts, hidden_scoring_state};
pub use ops::{Op, decode, fallback, ops};

/// Maximum tokens per observation. Tokens past it are dropped (the deck and
/// discard tokens come last), as are the ops on them.
pub const MAX_TOKENS: usize = 200;
/// Categorical features per token: zone, center, suit, rank, enhancement,
/// edition, seal, slot, aux.
pub const TOK_INT: usize = 9;
/// Continuous features per token.
pub const TOK_FLOAT: usize = 26;
/// Categorical global features: phase, stake, back, current blind, boss of
/// the ante, small-blind tag, big-blind tag, last hand played, last Tarot /
/// Planet used (The Fool's target, a center id), the ante's voucher once a
/// shop has shown it.
pub const GLOB_INT: usize = 10;
/// Continuous global features: 48 scalars, 32 voucher bits, then a
/// histogram of the remaining deck (52 suit x rank counts, enhancement,
/// edition and seal counts, size) and of the discard pile (52 counts).
pub const GLOB_FLOAT: usize = 216;
/// Maximum legal ops per decision.
pub const MAX_OPS: usize = 96;
/// Maximum scored play candidates per decision (the bindings reject larger
/// requests; [`ops`] also never lets them crowd out other ops).
pub const MAX_CANDIDATES: usize = MAX_OPS / 2;
/// Integers per op: kind, target token (-1 = none), card min, card max,
/// allowed card mask, required card mask, destination count, target slot.
pub const OP_INT: usize = 8;
/// Continuous features per op (scored play candidates: log of the mean
/// score, mean score / chips still needed, estimated probability of clearing
/// the blind, log of the lowest and highest sampled score, cards / 5).
pub const OP_FLOAT: usize = 6;
/// Plays previewed per decision for candidates (ranked by a cheap
/// joker-free estimate first).
pub const PREVIEW_BUDGET: usize = 64;
/// Chance scenarios per play whose score depends on luck (for the best
/// `PREVIEW_REFINE` of them by their first scenario; the others are dropped).
pub const PREVIEW_SAMPLES: usize = 6;
pub const PREVIEW_REFINE: usize = 16;
/// Pointer slots for hand cards (hand tokens come first, so a hand index is
/// also its token index).
pub const MAX_HAND: usize = 32;

/// Token zones.
pub mod zone {
    pub const PAD: i32 = 0;
    pub const HAND: i32 = 1;
    pub const JOKER: i32 = 2;
    pub const CONSUMABLE: i32 = 3;
    pub const SHOP_CARD: i32 = 4;
    pub const SHOP_VOUCHER: i32 = 5;
    pub const SHOP_BOOSTER: i32 = 6;
    pub const PACK: i32 = 7;
    pub const TAG: i32 = 8;
    pub const BLIND: i32 = 9;
    pub const HAND_TYPE: i32 = 10;
    pub const DECK: i32 = 11;
    pub const DISCARD: i32 = 12;
    pub const COUNT: i32 = 13;
}

/// Center id vocabulary (0 = none).
pub mod center {
    use crate::data::*;
    pub const PLAYING: i32 = 1;
    pub const JOKER0: i32 = 2;
    pub const TAROT0: i32 = JOKER0 + ALL_JOKERS.len() as i32;
    pub const PLANET0: i32 = TAROT0 + ALL_TAROTS.len() as i32;
    pub const SPECTRAL0: i32 = PLANET0 + ALL_PLANETS.len() as i32;
    pub const VOUCHER0: i32 = SPECTRAL0 + ALL_SPECTRALS.len() as i32;
    pub const BOOSTER0: i32 = VOUCHER0 + ALL_VOUCHERS.len() as i32;
    pub const TAG0: i32 = BOOSTER0 + ALL_BOOSTERS.len() as i32;
    pub const BLIND0: i32 = TAG0 + ALL_TAGS.len() as i32;
    pub const HAND0: i32 = BLIND0 + ALL_BLINDS.len() as i32;
    /// A face-down card or joker.
    pub const HIDDEN: i32 = HAND0 + HandType::ALL.len() as i32;
    pub const COUNT: i32 = HIDDEN + 1;
    const _: () = assert!(COUNT == 335);
}

/// Vocabulary size of each `TOK_INT` column (for embeddings).
pub const TOK_VOCAB: [i32; TOK_INT] = [
    zone::COUNT,
    center::COUNT,
    Suit::ALL.len() as i32 + 1,
    ACE as i32 + 1, // ranks; blinds put a poker hand + 1 here
    Enhancement::ALL.len() as i32,
    Edition::ALL.len() as i32,
    Seal::ALL.len() as i32 + 1,
    256, // slot
    32,  // aux: a poker hand + 1 (To Do List, Orbital Tag) or a tag + 1 (blinds)
];
/// Vocabulary size of each `GLOB_INT` column.
pub const GLOB_VOCAB: [i32; GLOB_INT] = [
    6, // phases
    9, // stakes 1..=8
    ALL_BACKS.len() as i32,
    ALL_BLINDS.len() as i32 + 1,
    ALL_BLINDS.len() as i32 + 1,
    ALL_TAGS.len() as i32 + 1,
    ALL_TAGS.len() as i32 + 1,
    HandType::ALL.len() as i32 + 1,
    center::COUNT,
    center::COUNT,
];

/// Op kinds.
pub mod op {
    pub const SELECT_BLIND: i32 = 0;
    pub const SKIP_BLIND: i32 = 1;
    pub const REROLL_BOSS: i32 = 2;
    pub const PLAY: i32 = 3;
    pub const DISCARD: i32 = 4;
    pub const USE: i32 = 5;
    pub const SELL_JOKER: i32 = 6;
    pub const SELL_CONSUMABLE: i32 = 7;
    pub const BUY: i32 = 8;
    pub const BUY_AND_USE: i32 = 9;
    pub const REDEEM: i32 = 10;
    pub const OPEN: i32 = 11;
    pub const REROLL: i32 = 12;
    pub const END_SHOP: i32 = 13;
    pub const CASH_OUT: i32 = 14;
    pub const PICK: i32 = 15;
    pub const SKIP_PACK: i32 = 16;
    pub const MOVE_JOKER: i32 = 17;
    pub const SORT_RANK: i32 = 18;
    pub const SORT_SUIT: i32 = 19;
    pub const MOVE_HAND: i32 = 20;
    /// One of the top-scoring plays (cards fixed; see `ops`).
    pub const PLAY_TOP: i32 = 21;
    pub const COUNT: i32 = 22;
}

/// Zones / phases / enums as small integers.
pub(super) fn phase_id(p: Phase) -> i32 {
    match p {
        Phase::BlindSelect => 0,
        Phase::Round => 1,
        Phase::Shop => 2,
        Phase::Pack => 3,
        Phase::RoundEval => 4,
        Phase::GameOver => 5,
    }
}

pub(super) fn kind_center(k: Kind) -> i32 {
    use center::*;
    match k {
        Kind::Playing => PLAYING,
        Kind::Joker(j) => JOKER0 + j as i32,
        Kind::Tarot(t) => TAROT0 + t as i32,
        Kind::Planet(p) => PLANET0 + p as i32,
        Kind::Spectral(s) => SPECTRAL0 + s as i32,
        Kind::Voucher(v) => VOUCHER0 + v as i32,
        Kind::Booster(b) => BOOSTER0 + b as i32,
    }
}

// Categorical ids (0 = none).
pub(super) fn suit_id(s: Suit) -> i32 {
    s as i32 + 1
}

pub(super) fn enh_id(e: Enhancement) -> i32 {
    e as i32
}

pub(super) fn edition_id(e: Edition) -> i32 {
    e as i32
}

pub(super) fn seal_id(s: Option<Seal>) -> i32 {
    s.map_or(0, |s| s as i32 + 1)
}

/// Signed log scaling for unbounded quantities.
#[inline]
pub(super) fn sl(x: f64) -> f32 {
    (x.signum() * x.abs().ln_1p()) as f32
}

#[inline]
pub(super) fn bf(b: bool) -> f32 {
    if b { 1.0 } else { 0.0 }
}

/// Observation buffers for one state (flat, row-major).
pub struct ObsMut<'a> {
    /// `[MAX_TOKENS * TOK_INT]`
    pub tok_i: &'a mut [i32],
    /// `[MAX_TOKENS * TOK_FLOAT]`
    pub tok_f: &'a mut [f32],
    /// `[GLOB_INT]`
    pub glob_i: &'a mut [i32],
    /// `[GLOB_FLOAT]`
    pub glob_f: &'a mut [f32],
}

/// Token index layout of an encoded observation (needed to decode ops):
/// the number of hand tokens, the first token of each other area, and the
/// number of tokens.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Layout {
    pub hand: usize,
    pub jokers: usize,
    pub consumables: usize,
    pub shop_cards: usize,
    pub shop_vouchers: usize,
    pub shop_boosters: usize,
    pub pack: usize,
    pub n_tokens: usize,
}

impl Layout {
    /// The layout [`encode`] produces for `g`, without encoding it. Deck and
    /// discard tokens (which no op refers to) are not counted in `n_tokens`.
    pub fn of(g: &Game) -> Layout {
        let mut n = 0;
        let mut area = |len: usize| {
            n = (n + len).min(MAX_TOKENS);
            n
        };
        let hand = area(g.hand.len().min(MAX_HAND));
        let consumables = area(g.jokers.len());
        let shop_cards = area(g.consumables.len());
        let shop_vouchers = area(g.shop_jokers.len());
        let shop_boosters = area(g.shop_vouchers.len());
        let pack = area(g.shop_boosters.len());
        // pack cards, tags, the three blinds, the poker hands
        let n_tokens = area(g.pack_cards.len() + g.tags.len() + 3 + HandType::ALL.len());
        Layout { hand, jokers: hand, consumables, shop_cards, shop_vouchers, shop_boosters, pack, n_tokens }
    }
}
