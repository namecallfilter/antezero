//! Play candidates: a cheap ranking of the legal plays, the shared chance
//! scenarios they are previewed in, and the [`top_plays`] summary.

use std::collections::HashSet;

use super::layout::shown_hand;
use crate::actions::Action;
use crate::card::{CardFace, CardMask};
use crate::data::{HandType, Joker};
use crate::game::{Areas, Blind, Game, Hand, HandLevels, Phase, Scratch};
use crate::hand_eval::EvalFlags;
use crate::rng::{self, PseudoRandom};
use crate::{areas, hand_eval, num, round};

/// Plays a build preview simulates per hand (the best by a cheap joker-free
/// estimate).
pub const PREVIEW_PLAYS: usize = 6;
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

/// The preview of playing `mask` now on a copy of the game (made in
/// `scratch`) that uses the chance scenario `alt` (a reseeded RNG). `None`
/// if the play is not legal.
#[must_use]
pub fn preview_score(
	scratch: &mut Scratch, g: &Game, mask: CardMask, alt: &PseudoRandom,
) -> Option<PlayPreview> {
	preview_in(scratch.copy(g, alt), mask)
}

/// The preview of playing `mask` in `copy`, which is spent on it.
pub(super) fn preview_in(copy: &mut Game, mask: CardMask) -> Option<PlayPreview> {
	let before = copy.rng.draws();
	let score = copy.preview_play(mask).ok()?;
	Some(PlayPreview {
		score,
		chance: copy.rng.draws() != before,
	})
}

/// A copy of `g` whose jokers are in a pseudo-random order derived from
/// `seed` and their ids only (not from their current, possibly hidden order).
fn shuffled_jokers(g: &Game, seed: u64) -> Game {
	let mut c = g.clone();
	c.areas.jokers.sort_by_key(|j| j.core.id);
	rng::splitmix_shuffle(&mut c.areas.jokers, seed);
	c
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
	pub mask: CardMask,
	pub mean: f64,
	pub min: f64,
	pub max: f64,
	/// Fraction of scenarios that reach the chips still needed.
	pub p_clear: f64,
}

impl Candidate {
	/// The summary of the play `mask` over the scores it sampled; `need` is
	/// the chips still needed to clear the blind.
	fn of(mask: CardMask, scores: &[f64], need: f64) -> Candidate {
		let n = num::float(scores.len());
		Candidate {
			mask,
			mean: scores.iter().sum::<f64>() / n,
			min: scores.iter().copied().fold(f64::INFINITY, f64::min),
			max: scores.iter().copied().fold(f64::NEG_INFINITY, f64::max),
			p_clear: num::float(scores.iter().filter(|&&s| s >= need).count()) / n,
		}
	}
}

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
/// A play the blind makes score nothing gets an estimate of 0 and no core,
/// behind every other play.
pub(super) fn rough_score(
	areas: &Areas, blind: &Blind, hands: &HandLevels, mask: CardMask, flags: EvalFlags,
) -> Rough {
	// Stack arrays as long as the longest play: this runs for every subset.
	let first = mask.lowest().expect("a play selects a card");
	let (mut idx, mut cards, mut n) = (
		[0usize; Hand::HIGHLIGHT_LIMIT],
		[&areas.hand.cards[first]; Hand::HIGHLIGHT_LIMIT],
		0,
	);
	for i in mask.indices().take(Hand::HIGHLIGHT_LIMIT) {
		(idx[n], cards[n]) = (i, &areas.hand.cards[i]);
		n += 1;
	}
	let top = hand_eval::evaluate(&cards[..n], flags)
		.top
		.expect("a play of at least one card is a hand");
	if round::debuffs_play(blind, n, top.hand) {
		return Rough {
			mask,
			estimate: 0.0,
			core: None,
		};
	}
	let l = &hands[top.hand];
	let mut chips = l.chips;
	let mut core = CardMask::EMPTY;
	let splash = areas.has_joker(Joker::Splash);
	for (k, c) in cards[..n].iter().enumerate() {
		if splash || top.mask >> k & 1 == 1 || c.is_stone() {
			chips += c.chip_bonus();
			core.insert(idx[k]);
		}
	}
	Rough {
		mask,
		estimate: chips * l.mult,
		core: Some((top.hand, core)),
	}
}

/// The `n` plays worth simulating, from their [`rough_score`]s: the smallest
/// play of each scoring core first, best estimate first, then the other
/// plays by estimate. Ranked by the estimate
/// alone, kicker variants of one core fill the budget and crowd out the plays
/// that hold cards back (Steel cards, Baron, Mime, Shoot the Moon).
pub(super) fn shortlist(mut plays: Vec<Rough>, n: usize) -> Vec<CardMask> {
	plays.sort_by(|a, b| {
		b.estimate
			.total_cmp(&a.estimate)
			.then(a.mask.len().cmp(&b.mask.len()))
			.then(a.mask.cmp(&b.mask))
	});
	let mut seen = HashSet::new();
	let (mut out, rest): (Vec<_>, Vec<_>) = plays.into_iter().partition(|p| seen.insert(p.core));
	out.extend(rest);
	out.truncate(n);
	out.into_iter().map(|p| p.mask).collect()
}

/// Two jokers pay for what is no longer held, so their best play often
/// carries kickers that no scoring core has: Blackboard (every card held is
/// black) and Raised Fist (the lowest rank held). With them, the plays to
/// try beside `cores` (the best scoring cores of the hand, best first):
/// for Blackboard the best two that can take every card that is not black
/// along, for Raised Fist the best two with the lowest ranks added.
pub(super) fn kicker_plays(areas: &Areas, cores: &[CardMask], nh: usize) -> Vec<CardMask> {
	let (board, fist) = (
		areas.owns_joker(Joker::Blackboard),
		areas.owns_joker(Joker::RaisedFist),
	);
	if !(board || fist) {
		return Vec::new();
	}
	let smeared = areas.smeared();
	let red: CardMask = (0..nh)
		.filter(|&i| !areas.hand.cards[i].is_black(smeared))
		.collect();
	let mut low: Vec<usize> = (0..nh).collect();
	low.sort_by_key(|&i| (areas.hand.cards[i].is_stone(), areas.hand.cards[i].rank, i));
	let with_low = |m: CardMask| {
		let add: CardMask = low
			.iter()
			.copied()
			.filter(|&i| !m.contains(i))
			.take(Hand::HIGHLIGHT_LIMIT.saturating_sub(m.len()))
			.collect();
		m | add
	};
	let clears = cores
		.iter()
		.filter(|&&m| board && (red & !m).len() + m.len() <= Hand::HIGHLIGHT_LIMIT)
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

/// The chance scenarios a decision's plays are previewed in: the same
/// scenarios for every play, seeded from visible state only, so they differ
/// between decisions but not between candidates or under `resample_hidden`.
#[derive(Debug)]
pub struct Scenarios<'game> {
	g: &'game Game,
	/// The reseeded RNG of each scenario.
	alts: Vec<PseudoRandom>,
	/// With something face down, each scenario's own world: face-down hand
	/// cards redealt from the unseen pool, face-down jokers (which hide
	/// only their order) in a shuffle of a canonical order. Else empty.
	views: Vec<Game>,
}

impl<'game> Scenarios<'game> {
	/// The first `samples` scenarios of `g`'s decision.
	#[must_use]
	pub fn new(g: &'game Game, samples: usize) -> Scenarios<'game> {
		// The hand as the player sees it (card ids are hidden state).
		let hand_id = g
			.areas
			.hand
			.cards
			.iter()
			.fold(rng::FNV_OFFSET_BASIS, |h, c| {
				let seen = if c.core.face_down {
					[99, 0, 0, 0, 0, 0, 0]
				} else {
					let face = CardFace::of(c);
					[
						u64::from(face.suit) + 1,
						u64::from(face.rank),
						u64::from(face.enhancement),
						u64::from(face.edition),
						u64::from(face.seal),
						u64::from(c.core.debuff),
						face.perma_bonus,
					]
				};
				seen.into_iter().fold(h, rng::fnv1a)
			});
		let base = format!(
			"PV{}~{}~{}~{}~{hand_id:x}",
			g.run.ante, g.run.round, g.round.hands_played, g.round.discards_used
		);
		let alts: Vec<PseudoRandom> = (0..samples)
			.map(|i| PseudoRandom::new_untraced(&format!("{base}~{i}")))
			.collect();
		let hidden_hand = g.areas.hand.cards.iter().any(|c| c.core.face_down);
		let shuffled = g.areas.jokers.iter().any(|c| c.core.face_down);
		let views: Vec<Game> = if shuffled || hidden_hand {
			(0..samples)
				.map(|i| {
					let mut v = if shuffled {
						shuffled_jokers(g, hand_id ^ num::index64(i))
					} else {
						g.clone()
					};
					if hidden_hand {
						// A world consistent with what the player sees, independent of the
						// hidden assignment.
						let seed = hand_id.rotate_left(17) ^ (num::index64(i) + 1);
						areas::redeal_unseen(
							&mut v.areas,
							|c| (CardFace::of(c), c.core.debuff),
							|pool| rng::splitmix_shuffle(pool, seed),
						);
					}
					v
				})
				.collect()
		} else {
			Vec::new()
		};
		Scenarios { g, alts, views }
	}

	/// Whether something is face down, so that every play is a chance play.
	fn hidden(&self) -> bool {
		!self.views.is_empty()
	}

	/// The game of scenario `i`.
	#[must_use]
	pub fn view(&self, i: usize) -> &Game {
		if self.hidden() {
			&self.views[i]
		} else {
			self.g
		}
	}

	/// The reseeded RNG of scenario `i`.
	#[must_use]
	pub fn alt(&self, i: usize) -> &PseudoRandom {
		&self.alts[i]
	}
}

/// A play and its preview in the first scenario.
#[derive(Clone, Copy)]
struct FirstPreview {
	mask: CardMask,
	preview: PlayPreview,
}

/// The plays to simulate: the `n` best by the cheap estimate (see
/// [`shortlist`]), and the kicker plays Blackboard and Raised Fist ask for.
/// Ranked on a view: face-down identities must not choose the previewed plays.
fn plays_to_preview(g: &Game, view: &Game, n: usize) -> Vec<CardMask> {
	let nh = shown_hand(&g.areas.hand.cards).len();
	let flags = g.areas.eval_flags();
	let mut legal = Vec::new();
	g.legal_actions(&mut legal);
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
			.filter(|&m| m.fits(nh))
			.map(|m| rough_score(&view.areas, &view.blind, &view.hands, m, flags))
			.collect(),
		n,
	);
	// The cores come first in the shortlist; on a view, like the ranking.
	for m in kicker_plays(&view.areas, &plays[..plays.len().min(PREVIEW_PLAYS)], nh) {
		if !plays.contains(&m) && legal.contains(&Action::Play(m)) {
			plays.push(m);
		}
	}
	plays
}

/// The `k` best plays (by probability of clearing the blind, then mean
/// score) over shared chance scenarios. Plays are first ranked by a cheap
/// estimate and only the best `PREVIEW_BUDGET` are simulated, once each;
/// the `PREVIEW_REFINE` best plays whose score depended on chance are then
/// simulated on `PREVIEW_SAMPLES` scenarios in total, and the other chance
/// plays are dropped (a single sample is not a fair estimate). With
/// face-down hand cards or jokers every play is a chance play, previewed in
/// worlds that do not reveal them (see [`Scenarios`]).
#[must_use]
pub fn top_plays(g: &Game, k: usize) -> Vec<Candidate> {
	if k == 0 || g.flow.phase != Phase::Round {
		return Vec::new();
	}
	let scenarios = Scenarios::new(g, PREVIEW_SAMPLES);
	let plays = plays_to_preview(g, scenarios.view(0), PREVIEW_BUDGET.max(k));
	let need = (g.blind.chips - g.round.chips).max(0.0);
	let mut scratch = Scratch::default();
	let mut first: Vec<FirstPreview> = plays
		.iter()
		.filter_map(|&mask| {
			let p = preview_in(scratch.copy(scenarios.view(0), &scenarios.alts[0]), mask)?;
			let preview = PlayPreview {
				chance: p.chance || scenarios.hidden(),
				..p
			};
			Some(FirstPreview { mask, preview })
		})
		.collect();
	let mut chance: Vec<&FirstPreview> = first.iter().filter(|f| f.preview.chance).collect();
	chance.sort_by(|a, b| {
		b.preview
			.score
			.total_cmp(&a.preview.score)
			.then(a.mask.cmp(&b.mask))
	});
	let refined: Vec<CardMask> = chance.iter().take(PREVIEW_REFINE).map(|f| f.mask).collect();
	first.retain(|f| !f.preview.chance || refined.contains(&f.mask));
	let mut out: Vec<Candidate> = first
		.iter()
		.map(|&FirstPreview { mask, preview }| {
			let (mut scores, mut sampled) = ([preview.score; PREVIEW_SAMPLES], 1);
			if preview.chance {
				for (i, alt) in scenarios.alts.iter().enumerate().skip(1) {
					if let Some(p) = preview_in(scratch.copy(scenarios.view(i), alt), mask) {
						scores[sampled] = p.score;
						sampled += 1;
					}
				}
			}
			Candidate::of(mask, &scores[..sampled], need)
		})
		.collect();
	out.sort_by(|a, b| {
		b.p_clear
			.total_cmp(&a.p_clear)
			.then(b.mean.total_cmp(&a.mean))
			.then(a.mask.len().cmp(&b.mask.len()))
			.then(a.mask.cmp(&b.mask))
	});
	out.truncate(k);
	out
}
