//! Scored play candidates: previews of the most promising plays over shared
//! chance scenarios.

use super::*;
use crate::actions::{Action, CardMask};
use crate::card::Card;
use crate::game::Game;

/// Score of playing `mask` now on a copy of the game that uses the chance
/// scenario `alt` (a reseeded RNG), and whether the score depended on chance
/// (any random draw happened). `None` if the play is not legal.
pub fn preview_score(g: &Game, mask: CardMask, alt: &crate::rng::PseudoRandom) -> Option<(f64, bool)> {
    let mut c = g.clone();
    c.rng = alt.clone();
    let before = c.rng.draws();
    let s = c.preview_play(mask).ok()?;
    Some((s, c.rng.draws() != before))
}

/// A scored play candidate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    pub mask: CardMask,
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    /// Fraction of scenarios that reach the chips still needed.
    pub p_clear: f64,
}

/// Cheap joker-free estimate used to pick which plays to preview: hand
/// level chips plus the scoring cards' chips, times the hand level mult.
fn rough_score(g: &Game, mask: CardMask, flags: &crate::hand_eval::EvalFlags) -> f64 {
    let cards: Vec<&Card> = (0..g.hand.len()).filter(|&i| mask >> i & 1 == 1).map(|i| &g.hand[i]).collect();
    let ph = crate::hand_eval::evaluate(&cards, flags);
    let Some(top) = ph.top else { return 0.0 };
    let l = &g.hands[top as usize];
    let mut chips = l.chips;
    for (k, c) in cards.iter().enumerate() {
        if ph.top_mask >> k & 1 == 1 || c.is_stone() {
            chips += c.chip_bonus();
        }
    }
    chips * l.mult
}

/// The `k` best plays (by probability of clearing the blind, then mean
/// score) over shared chance scenarios. Plays are first ranked by a cheap
/// estimate and only the best `PREVIEW_BUDGET` are simulated, once each;
/// the `PREVIEW_REFINE` best plays whose score depended on chance are then
/// simulated on `PREVIEW_SAMPLES` scenarios in total (the same scenarios for
/// every play, seeded from visible state only, so they differ between
/// decisions but not between candidates or under `resample_hidden`), and
/// the other chance plays are dropped (a single sample is not a fair
/// estimate). Nothing is previewed while scoring-relevant cards are face
/// down (the preview would reveal them).
pub fn top_plays(g: &Game, k: usize) -> Vec<Candidate> {
    if k == 0 || g.phase != Phase::Round || hidden_scoring_state(g) {
        return Vec::new();
    }
    let nh = g.hand.len().min(MAX_HAND);
    let flags = g.eval_flags();
    let mut acts = Vec::new();
    g.legal_actions(&mut acts);
    let mut plays: Vec<(f64, CardMask)> = acts
        .iter()
        .filter_map(|a| if let Action::Play(m) = *a { Some(m) } else { None })
        .filter(|&m| crate::actions::mask_fits(m, nh))
        .map(|m| (rough_score(g, m, &flags), m))
        .collect();
    plays.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
    plays.truncate(PREVIEW_BUDGET.max(k));
    // The hand as the player sees it (card ids are hidden state).
    let hand_id = g.hand.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, c| {
        let seen = [suit_id(c.suit) as u64, c.rank as u64, enh_id(c.enhancement) as u64, edition_id(c.edition) as u64, seal_id(c.seal) as u64, c.debuff as u64, c.ab.perma_bonus.to_bits()];
        seen.iter().fold(h, |h, &x| (h ^ x).wrapping_mul(0x100_0000_01B3))
    });
    let base = format!("PV{}~{}~{}~{}~{hand_id:x}", g.ante, g.round, g.hands_played_round, g.discards_used);
    let alts: Vec<crate::rng::PseudoRandom> = (0..PREVIEW_SAMPLES).map(|i| crate::rng::PseudoRandom::new_untraced(&format!("{base}~{i}"))).collect();
    let need = (g.blind.chips - g.chips).max(0.0);
    let mut first: Vec<(CardMask, f64, bool)> = plays.iter().filter_map(|&(_, m)| preview_score(g, m, &alts[0]).map(|(s, c)| (m, s, c))).collect();
    // Refine the most promising chance plays.
    let mut chance: Vec<usize> = (0..first.len()).filter(|&i| first[i].2).collect();
    chance.sort_by(|&a, &b| first[b].1.partial_cmp(&first[a].1).unwrap_or(std::cmp::Ordering::Equal).then(first[a].0.cmp(&first[b].0)));
    let refine: std::collections::HashSet<usize> = chance.into_iter().take(PREVIEW_REFINE).collect();
    first = first.into_iter().enumerate().filter(|(i, f)| !f.2 || refine.contains(i)).map(|(_, f)| f).collect();
    let mut out: Vec<Candidate> = Vec::new();
    for &(m, s0, chance) in &first {
        let mut v = vec![s0];
        if chance {
            for alt in &alts[1..] {
                if let Some((s, _)) = preview_score(g, m, alt) {
                    v.push(s);
                }
            }
        }
        let n = v.len() as f64;
        let mean = v.iter().sum::<f64>() / n;
        let min = v.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let p_clear = v.iter().filter(|&&s| s >= need).count() as f64 / n;
        out.push(Candidate { mask: m, mean, min, max, p_clear });
    }
    out.sort_by(|a, b| {
        b.p_clear
            .partial_cmp(&a.p_clear)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.mean.partial_cmp(&a.mean).unwrap_or(std::cmp::Ordering::Equal))
            .then(a.mask.count_ones().cmp(&b.mask.count_ones()))
            .then(a.mask.cmp(&b.mask))
    });
    out.truncate(k);
    out
}
