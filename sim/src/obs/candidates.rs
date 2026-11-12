//! Scored play candidates: previews of the most promising plays over shared
//! chance scenarios.

use super::MAX_HAND;
use super::encode::{edition_id, enh_id, seal_id, suit_id};
use super::preview::PREVIEW_PLAYS;
use crate::actions::{Action, CardMask};
use crate::card::{Card, Kind};
use crate::data::{HandType, Joker, Suit};
use crate::game::{Game, Phase};
use crate::{blind, num};

/// Plays previewed per decision for candidates (ranked by a cheap
/// joker-free estimate first).
pub const PREVIEW_BUDGET: usize = 64;
/// Chance scenarios per play whose score depends on luck (for the best
/// `PREVIEW_REFINE` of them by their first scenario; the others are dropped).
pub const PREVIEW_SAMPLES: usize = 6;
pub const PREVIEW_REFINE: usize = 16;

/// What a play scores in one chance scenario.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlayPreview {
	pub score: f64,
	/// Whether the score depended on chance (any random draw happened).
	pub chance: bool,
}

/// The preview of playing `mask` now on a copy of the game that uses the
/// chance scenario `alt` (a reseeded RNG). `None` if the play is not legal.
#[must_use]
pub fn preview_score(
	g: &Game, mask: CardMask, alt: &crate::rng::PseudoRandom,
) -> Option<PlayPreview> {
	let mut c = g.clone();
	c.rng = alt.clone();
	let before = c.rng.draws();
	let score = c.preview_play(mask).ok()?;
	Some(PlayPreview {
		score,
		chance: c.rng.draws() != before,
	})
}

/// A copy of `g` whose jokers are in a pseudo-random order derived from
/// `seed` and their ids only (not from their current, possibly hidden order).
fn shuffled_jokers(g: &Game, seed: u64) -> Game {
	let mut c = g.clone();
	c.jokers.sort_by_key(|j| j.id);
	super::digest::splitmix_shuffle(&mut c.jokers, seed);
	c
}

/// Redeals `v`'s face-down hand cards from the unseen pool (deck + face-down
/// hand cards) put in a canonical order (visible content only) and shuffled
/// with `seed`: a world consistent with what the player sees, independent of
/// the hidden assignment.
fn redeal_face_down(v: &mut Game, seed: u64) {
	let down: Vec<usize> = (0..v.hand.len()).filter(|&i| v.hand[i].face_down).collect();
	if down.is_empty() {
		return;
	}
	let mut pool: Vec<Card> = down.iter().map(|&i| v.hand[i].clone()).collect();
	pool.append(&mut v.deck);
	let key = |c: &Card| {
		(
			suit_id(c.suit),
			i32::from(c.rank),
			enh_id(c.enhancement),
			edition_id(c.edition),
			seal_id(c.seal),
			c.ab.perma_bonus.to_bits(),
			c.debuff,
		)
	};
	pool.sort_by_key(key);
	super::digest::splitmix_shuffle(&mut pool, seed);
	let rest = pool.split_off(down.len());
	for (mut c, &i) in pool.into_iter().zip(&down) {
		c.face_down = v.hand[i].face_down;
		c.ab.wheel_flipped = v.hand[i].ab.wheel_flipped;
		v.hand[i] = c;
	}
	v.deck = rest;
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

/// A play with its cheap estimate (see [`rough_score`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Rough {
	pub mask: CardMask,
	pub estimate: f64,
	/// The play's scoring core: its hand type and scoring cards. `None` for a
	/// play the blind makes score nothing.
	pub core: Option<(HandType, CardMask)>,
}

/// Cheap estimate used to pick which plays to preview (hand
/// level chips plus the scoring cards' chips, times the hand level mult),
/// and the play's scoring core: its hand type and scoring cards (plays with
/// the same core differ only in kickers, which add nothing to the estimate).
/// Plays the blind makes score nothing get (0, 0), behind every other play.
pub(super) fn rough_score(g: &Game, mask: CardMask, flags: crate::hand_eval::EvalFlags) -> Rough {
	// A play has at most 5 cards (stack arrays: this runs for every subset).
	let (mut idx, mut cards, mut n, mut m) = (
		[0usize; 5],
		[&g.hand[mask.trailing_zeros() as usize]; 5],
		0,
		mask,
	);
	while m != 0 && n < 5 {
		let i = m.trailing_zeros() as usize;
		(idx[n], cards[n]) = (i, &g.hand[i]);
		n += 1;
		m &= m - 1;
	}
	let ph = crate::hand_eval::evaluate(&cards[..n], &flags);
	let nothing = Rough {
		mask,
		estimate: 0.0,
		core: None,
	};
	let Some(top) = ph.top else { return nothing };
	if blind::debuffs_play(g, n, top) {
		return nothing;
	}
	let l = &g.hands[top as usize];
	let mut chips = l.chips;
	let mut core: CardMask = 0;
	let splash = g.has_joker(Joker::Splash);
	for (k, c) in cards[..n].iter().enumerate() {
		if splash || ph.top_mask >> k & 1 == 1 || c.is_stone() {
			chips += c.chip_bonus();
			core |= 1 << idx[k];
		}
	}
	Rough {
		mask,
		estimate: chips * l.mult,
		core: Some((top, core)),
	}
}

/// The `n` plays worth simulating, from `(estimate, play, core)` (see
/// `rough_score`): the smallest play of each scoring core first, best
/// estimate first, then the other plays by estimate. Ranked by the estimate
/// alone, kicker variants of one core fill the budget and crowd out the plays
/// that hold cards back (Steel cards, Baron, Mime, Shoot the Moon).
pub(super) fn shortlist(mut plays: Vec<Rough>, n: usize) -> Vec<CardMask> {
	plays.sort_by(|a, b| {
		b.estimate
			.partial_cmp(&a.estimate)
			.unwrap_or(std::cmp::Ordering::Equal)
			.then(a.mask.count_ones().cmp(&b.mask.count_ones()))
			.then(a.mask.cmp(&b.mask))
	});
	let mut seen = std::collections::HashSet::new();
	let (mut out, rest): (Vec<_>, Vec<_>) = plays.into_iter().partition(|p| seen.insert(p.core));
	out.extend(rest);
	out.truncate(n);
	out.into_iter().map(|p| p.mask).collect()
}

pub(super) fn has_kind(g: &Game, j: Joker) -> bool {
	g.jokers.iter().any(|c| c.kind == Kind::Joker(j))
}

/// Two jokers pay for what is no longer held, so their best play often
/// carries kickers that no scoring core has: Blackboard (every card held is
/// black) and Raised Fist (the lowest rank held). With them, the plays to
/// try beside `cores` (the best scoring cores of `v`'s hand, best first):
/// for Blackboard the best two that can take every card that is not black
/// along, for Raised Fist the best two with the lowest ranks added.
pub(super) fn kicker_plays(v: &Game, cores: &[CardMask], nh: usize) -> Vec<CardMask> {
	let (board, fist) = (
		has_kind(v, Joker::Blackboard),
		has_kind(v, Joker::RaisedFist),
	);
	if !(board || fist) {
		return Vec::new();
	}
	let smeared = v.eval_flags().smeared;
	let black = |c: &Card| {
		c.is_suit(Suit::Clubs, false, true, smeared)
			|| c.is_suit(Suit::Spades, false, true, smeared)
	};
	let red = (0..nh)
		.filter(|&i| !black(&v.hand[i]))
		.fold(0u32, |m, i| m | 1 << i);
	let mut low: Vec<usize> = (0..nh).collect();
	low.sort_by_key(|&i| (v.hand[i].is_stone(), v.hand[i].rank, i));
	let with_low = |m: CardMask| {
		low.iter()
			.filter(|&&i| m >> i & 1 == 0)
			.take(5usize.saturating_sub(m.count_ones() as usize))
			.fold(m, |m, &i| m | 1 << i)
	};
	let clears = cores
		.iter()
		.filter(|&&m| board && (red & !m).count_ones() + m.count_ones() <= 5)
		.take(2)
		.map(|&m| m | red);
	clears
		.chain(
			cores
				.iter()
				.take(if fist { 2 } else { 0 })
				.map(|&m| with_low(m)),
		)
		.collect()
}

/// The `k` best plays (by probability of clearing the blind, then mean
/// score) over shared chance scenarios. Plays are first ranked by a cheap
/// estimate and only the best `PREVIEW_BUDGET` are simulated, once each;
/// the `PREVIEW_REFINE` best plays whose score depended on chance are then
/// simulated on `PREVIEW_SAMPLES` scenarios in total (the same scenarios for
/// every play, seeded from visible state only, so they differ between
/// decisions but not between candidates or under `resample_hidden`), and
/// the other chance plays are dropped (a single sample is not a fair
/// estimate). Nothing is previewed while hand cards are face down (the
/// preview would reveal them). Face-down jokers (Amber Acorn) hide only
/// their order: each scenario plays them in its own shuffle of a canonical
/// order, so every play is a chance play and the hidden order stays hidden.
#[expect(
	clippy::too_many_lines,
	reason = "one pass from ranking to refined candidates"
)]
pub fn top_plays(g: &Game, k: usize) -> Vec<Candidate> {
	if k == 0 || g.phase != Phase::Round {
		return Vec::new();
	}
	let hidden = g.hand.iter().any(|c| c.face_down);
	let nh = g.hand.len().min(MAX_HAND);
	let flags = g.eval_flags();
	let mut legal = Vec::new();
	g.legal_actions(&mut legal);
	// The hand as the player sees it (card ids are hidden state).
	let hand_id = g.hand.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, c| {
		let seen = if c.face_down {
			[99, 0, 0, 0, 0, 0, 0]
		} else {
			[
				c.suit as u64 + 1,
				u64::from(c.rank),
				c.enhancement as u64,
				c.edition as u64,
				c.seal.map_or(0, |s| s as u64 + 1),
				u64::from(c.debuff),
				c.ab.perma_bonus.to_bits(),
			]
		};
		seen.iter()
			.fold(h, |h, &x| (h ^ x).wrapping_mul(0x100_0000_01b3))
	});
	let base = format!(
		"PV{}~{}~{}~{}~{hand_id:x}",
		g.ante, g.round, g.hands_played_round, g.discards_used
	);
	let alts: Vec<crate::rng::PseudoRandom> = (0..PREVIEW_SAMPLES)
		.map(|i| crate::rng::PseudoRandom::new_untraced(&format!("{base}~{i}")))
		.collect();
	let shuffled = g.jokers.iter().any(|c| c.face_down);
	let views: Vec<Game> = if shuffled || hidden {
		(0..PREVIEW_SAMPLES)
			.map(|i| {
				let mut v = if shuffled {
					shuffled_jokers(g, hand_id ^ i as u64)
				} else {
					g.clone()
				};
				if hidden {
					redeal_face_down(&mut v, hand_id.rotate_left(17) ^ (i as u64 + 1));
				}
				v
			})
			.collect()
	} else {
		Vec::new()
	};
	let view = |i: usize| views.get(i).unwrap_or(g);
	// Ranked on a view: face-down identities must not choose the previewed plays.
	let mut plays = shortlist(
		legal
			.iter()
			.filter_map(|a| {
				if let Action::Play(m) = *a {
					Some(m)
				} else {
					None
				}
			})
			.filter(|&m| crate::actions::mask_fits(m, nh))
			.map(|m| rough_score(view(0), m, flags))
			.collect(),
		PREVIEW_BUDGET.max(k),
	);
	// The cores come first in the shortlist; on a view, like the ranking.
	for m in kicker_plays(view(0), &plays[..plays.len().min(PREVIEW_PLAYS)], nh) {
		if !plays.contains(&m) && legal.contains(&Action::Play(m)) {
			plays.push(m);
		}
	}
	let need = (g.blind.chips - g.chips).max(0.0);
	let mut first: Vec<(CardMask, f64, bool)> = plays
		.iter()
		.filter_map(|&m| {
			preview_score(view(0), m, &alts[0])
				.map(|p| (m, p.score, p.chance || shuffled || hidden))
		})
		.collect();
	// Refine the most promising chance plays.
	let mut chance: Vec<usize> = (0..first.len()).filter(|&i| first[i].2).collect();
	chance.sort_by(|&a, &b| {
		first[b]
			.1
			.partial_cmp(&first[a].1)
			.unwrap_or(std::cmp::Ordering::Equal)
			.then(first[a].0.cmp(&first[b].0))
	});
	let refine: std::collections::HashSet<usize> =
		chance.into_iter().take(PREVIEW_REFINE).collect();
	first = first
		.into_iter()
		.enumerate()
		.filter(|(i, f)| !f.2 || refine.contains(i))
		.map(|(_, f)| f)
		.collect();
	let mut out: Vec<Candidate> = Vec::new();
	for &(m, s0, chance) in &first {
		let (mut scores, mut sampled) = ([s0; PREVIEW_SAMPLES], 1);
		if chance {
			for (i, alt) in alts.iter().enumerate().skip(1) {
				if let Some(p) = preview_score(view(i), m, alt) {
					scores[sampled] = p.score;
					sampled += 1;
				}
			}
		}
		let v = &scores[..sampled];
		let n = num::float(v.len());
		let mean = v.iter().sum::<f64>() / n;
		let min = v.iter().copied().fold(f64::INFINITY, f64::min);
		let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
		let p_clear = num::float(v.iter().filter(|&&s| s >= need).count()) / n;
		out.push(Candidate {
			mask: m,
			mean,
			min,
			max,
			p_clear,
		});
	}
	out.sort_by(|a, b| {
		b.p_clear
			.partial_cmp(&a.p_clear)
			.unwrap_or(std::cmp::Ordering::Equal)
			.then(
				b.mean
					.partial_cmp(&a.mean)
					.unwrap_or(std::cmp::Ordering::Equal),
			)
			.then(a.mask.count_ones().cmp(&b.mask.count_ones()))
			.then(a.mask.cmp(&b.mask))
	});
	out.truncate(k);
	out
}
