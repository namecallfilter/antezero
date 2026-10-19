//! Port of `evaluate_poker_hand` and helpers from `misc_functions.lua`.
//!
//! Groups are represented as bitmasks over the positions of the evaluated
//! cards; the game always re-sorts the scoring hand by table position, so
//! membership is all that matters downstream.

use crate::card::Card;
use crate::data::{HandType, Rank, Suit};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PokerHands {
    /// Bit `h as u16` set when `poker_hands[h]` is non-empty.
    pub present: u16,
    /// `poker_hands[top][1]` membership mask.
    pub top_mask: u8,
    pub top: Option<HandType>,
}

impl PokerHands {
    #[inline]
    pub fn has(&self, h: HandType) -> bool {
        self.present >> (h as u16) & 1 == 1
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EvalFlags {
    pub four_fingers: bool,
    pub shortcut: bool,
    pub smeared: bool,
}

/// Groups of exactly `num` same-id cards, highest rank first.
fn x_same(num: u32, ids: &[Option<Rank>]) -> ([u8; 5], usize) {
    let mut vals = [0u8; 15];
    let n = ids.len();
    for i in (0..n).rev() {
        let Some(id) = ids[i] else { continue };
        let mut mask = 1u8 << i;
        let mut cnt = 1;
        for (j, other) in ids.iter().enumerate() {
            if j != i && *other == Some(id) {
                mask |= 1 << j;
                cnt += 1;
            }
        }
        if cnt == num {
            vals[id as usize] = mask;
        }
    }
    let mut out = [0u8; 5];
    let mut k = 0;
    for r in (1..15).rev() {
        if vals[r] != 0 && k < 5 {
            out[k] = vals[r];
            k += 1;
        }
    }
    (out, k)
}

fn flush(cards: &[&Card], f: &EvalFlags) -> u8 {
    let need = if f.four_fingers { 4 } else { 5 };
    if cards.len() > 5 || cards.len() < need {
        return 0;
    }
    for suit in Suit::ALL {
        let mut mask = 0u8;
        let mut cnt = 0;
        for (i, c) in cards.iter().enumerate() {
            if c.is_suit(suit, false, true, f.smeared) {
                cnt += 1;
                mask |= 1 << i;
            }
        }
        if cnt >= need {
            return mask;
        }
    }
    0
}

fn straight(ids: &[Option<Rank>], f: &EvalFlags) -> u8 {
    let need = if f.four_fingers { 4 } else { 5 };
    if ids.len() > 5 || ids.len() < need {
        return 0;
    }
    let mut by_id = [0u8; 15];
    for (i, id) in ids.iter().enumerate() {
        if let Some(id) = *id && id > 1 && id < 15 {
            by_id[id as usize] |= 1 << i;
        }
    }
    let mut t = 0u8;
    let mut length = 0;
    let mut found = false;
    let mut skipped = false;
    for j in 1..=14usize {
        let r = if j == 1 { 14 } else { j };
        if by_id[r] != 0 {
            length += 1;
            skipped = false;
            t |= by_id[r];
        } else if f.shortcut && !skipped && j != 14 {
            skipped = true;
        } else {
            length = 0;
            skipped = false;
            if !found {
                t = 0;
            }
            if found {
                break;
            }
        }
        if length >= need {
            found = true;
        }
    }
    if found { t } else { 0 }
}

fn highest(cards: &[&Card]) -> u8 {
    let mut best: Option<(usize, f64)> = None;
    for (i, c) in cards.iter().enumerate() {
        let v = c.nominal(false);
        match best {
            None => best = Some((i, v)),
            Some((_, bv)) if v > bv => best = Some((i, v)),
            _ => {}
        }
    }
    best.map(|(i, _)| 1u8 << i).unwrap_or(0)
}

/// `evaluate_poker_hand(hand)` + the hand selection of `get_poker_hand_info`.
pub fn evaluate(cards: &[&Card], f: &EvalFlags) -> PokerHands {
    debug_assert!(cards.len() <= 8);
    let mut ids = [None; 8];
    for (i, c) in cards.iter().enumerate() {
        ids[i] = c.get_id();
    }
    let ids = &ids[..cards.len()];
    let (g5, n5) = x_same(5, ids);
    let (g4, n4) = x_same(4, ids);
    let (g3, n3) = x_same(3, ids);
    let (g2, n2) = x_same(2, ids);
    let fl = flush(cards, f);
    let st = straight(ids, f);
    let hi = if cards.is_empty() { 0 } else { highest(cards) };

    let mut present = 0u16;
    let mut masks = [0u8; 12];
    macro_rules! set {
        ($h:expr, $m:expr) => {{
            let m = $m;
            present |= 1 << ($h as u16);
            masks[$h as usize] = m;
        }};
    }
    if n5 > 0 && fl != 0 {
        set!(HandType::FlushFive, g5[0]);
    }
    if n3 > 0 && n2 > 0 && fl != 0 {
        set!(HandType::FlushHouse, g3[0] | g2[0]);
    }
    if n5 > 0 {
        set!(HandType::FiveOfAKind, g5[0]);
    }
    if fl != 0 && st != 0 {
        set!(HandType::StraightFlush, fl | st);
    }
    if n4 > 0 {
        set!(HandType::FourOfAKind, g4[0]);
    }
    if n3 > 0 && n2 > 0 {
        set!(HandType::FullHouse, g3[0] | g2[0]);
    }
    if fl != 0 {
        set!(HandType::Flush, fl);
    }
    if st != 0 {
        set!(HandType::Straight, st);
    }
    if n3 > 0 {
        set!(HandType::ThreeOfAKind, g3[0]);
    }
    if n2 == 2 || (n3 == 1 && n2 == 1) {
        let b = if n2 >= 2 { g2[1] } else { g3[0] };
        set!(HandType::TwoPair, g2[0] | b);
    }
    if n2 > 0 {
        set!(HandType::Pair, g2[0]);
    }
    if hi != 0 {
        set!(HandType::HighCard, hi);
    }
    // Containment fix-ups at the end of evaluate_poker_hand.
    if n5 > 0 {
        set!(HandType::FourOfAKind, g5[0]);
    }
    if present >> (HandType::FourOfAKind as u16) & 1 == 1 {
        let m = masks[HandType::FourOfAKind as usize];
        set!(HandType::ThreeOfAKind, m);
    }
    if present >> (HandType::ThreeOfAKind as u16) & 1 == 1 {
        let m = masks[HandType::ThreeOfAKind as usize];
        set!(HandType::Pair, m);
    }

    let top = HandType::ALL.iter().copied().find(|h| present >> (*h as u16) & 1 == 1);
    PokerHands { present, top_mask: top.map(|h| masks[h as usize]).unwrap_or(0), top }
}
