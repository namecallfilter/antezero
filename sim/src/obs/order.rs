//! Joker order: what the best single joker move would add to a play's score,
//! as an input on the joker-move ops ([`order_gain`]) and as labels at played
//! hands ([`order_label`]).
//!
//! A play is previewed in the current order and after every single move on
//! the decision's shared chance scenarios (see [`Scenarios`]), so nothing here
//! reads the deck order or the game's own RNG. Scores are capped at the chips
//! still needed: a score beyond the blind is worth nothing. A move that
//! changes what the play leaves besides its score (money, jokers,
//! consumables, cards in hand and deck) counts as no gain: where a joker
//! stands then matters beyond the score.

use super::candidates::{PlayPreview, Scenarios, preview_in};
use super::{MAX_REARRANGES, OP_ORDER_GAIN, Op, OpKind, PREVIEW_SAMPLES, hidden_scoring_state};
use crate::card::CardMask;
use crate::data::Joker;
use crate::game::{Areas, Flow, Game, Phase, Scratch};
use crate::num;

/// Most jokers whose single moves are scored.
pub const ORDER_JOKERS: usize = 6;
/// Scale of the gain input. A zero-initialised input weight moves about one
/// learning rate per optimizer step, so the scale sets how fast the input
/// can act.
pub const ORDER_GAIN_SCALE: f64 = 300.0;
/// The least ratio of capped scores (each plus one) that counts as a gain.
const MIN_GAIN: f64 = 1.01;
/// Moves within this share of the best capped score are labelled with it.
const TIE: f64 = 0.999;

/// A single joker move.
#[derive(Clone, Copy)]
struct Move {
	from: usize,
	to: usize,
}

/// Every single move of `slots` jokers, by source, then destination.
fn single_moves(slots: usize) -> impl Iterator<Item = Move> {
	(0..slots).flat_map(move |from| {
		(0..slots)
			.filter(move |&to| to != from)
			.map(move |to| Move { from, to })
	})
}

/// What a play leaves besides its score.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Side {
	/// The money's bits.
	dollars: u64,
	jokers: usize,
	consumables: usize,
	/// Cards in hand and deck.
	cards: usize,
}

/// One scenario's capped scores of a play.
struct Scores {
	/// In the current order.
	unmoved: f64,
	/// After the move `[from][to]`.
	moved: [[f64; ORDER_JOKERS]; ORDER_JOKERS],
	/// Whether a score involved a random draw.
	chance: bool,
}

/// A play in the shared scenarios, in any joker order.
struct Play<'game> {
	scenarios: Scenarios<'game>,
	scratch: Scratch,
	mask: CardMask,
	/// Per scenario: what the play leaves in the current order (made when
	/// first asked).
	sides: Vec<Option<Side>>,
}

impl Play<'_> {
	/// A copy of scenario `k` after the joker move `mv`.
	fn world(&mut self, k: usize, mv: Option<Move>) -> &mut Game {
		let copy = self
			.scratch
			.copy(self.scenarios.view(k), self.scenarios.alt(k));
		if let Some(Move { from, to }) = mv {
			let joker = copy.areas.jokers.remove(from);
			copy.areas.jokers.insert(to, joker);
		}
		copy
	}

	/// The play's preview in scenario `k` after `mv`. `None` if the play is
	/// not legal.
	fn preview(&mut self, k: usize, mv: Option<Move>) -> Option<PlayPreview> {
		let mask = self.mask;
		preview_in(self.world(k, mv), mask)
	}

	/// Scenario `k`'s capped scores of the play, whose preview in the current
	/// order is `unmoved`; `need` is the chips still needed.
	fn scores(&mut self, k: usize, unmoved: PlayPreview, need: f64) -> Scores {
		let slots = self.scenarios.view(k).areas.jokers.len();
		let mut chance = unmoved.chance;
		let mut moved = [[0.0; ORDER_JOKERS]; ORDER_JOKERS];
		for mv in single_moves(slots) {
			// One slot down is the order its neighbour makes moving one slot up.
			moved[mv.from][mv.to] = if mv.to + 1 == mv.from {
				moved[mv.to][mv.from]
			} else {
				let preview = self
					.preview(k, Some(mv))
					.expect("moving a joker keeps a play legal");
				chance |= preview.chance;
				preview.score.min(need)
			};
		}
		Scores {
			unmoved: unmoved.score.min(need),
			moved,
			chance,
		}
	}

	/// What the play leaves in scenario `k` after `mv`.
	fn side(&mut self, k: usize, mv: Option<Move>) -> Side {
		let mask = self.mask;
		let copy = self.world(k, mv);
		copy.preview_hand(mask)
			.expect("a play that was scored is legal");
		Side {
			dollars: copy.run.dollars.to_bits(),
			jokers: copy.areas.jokers.len(),
			consumables: copy.areas.consumables.len(),
			cards: copy.areas.hand.cards.len() + copy.areas.deck.len(),
		}
	}

	/// Whether `mv` leaves the same as the current order besides the score,
	/// in every scenario `sides` covers.
	fn clean(&mut self, mv: Move) -> bool {
		(0..self.sides.len()).all(|k| {
			let unmoved = if let Some(side) = self.sides[k] {
				side
			} else {
				let side = self.side(k, None);
				self.sides[k] = Some(side);
				side
			};
			self.side(k, Some(mv)) == unmoved
		})
	}
}

fn gains(capped: f64, base: f64) -> bool {
	(capped + 1.0) / (base + 1.0) >= MIN_GAIN
}

/// Whether single joker moves are scored: a round with 2 to `ORDER_JOKERS`
/// jokers and the reorder ops on offer (`can_reorder`), nothing face down
/// (the previews would reveal it), no Ceremonial Dagger (it destroys its
/// right neighbour when a blind is selected).
fn eligible(flow: Flow, areas: &Areas, can_reorder: bool) -> bool {
	flow.phase == Phase::Round
		&& (2..=ORDER_JOKERS).contains(&areas.jokers.len())
		&& can_reorder
		&& !hidden_scoring_state(areas)
		&& !areas.owns_joker(Joker::Ceremonial)
}

/// A move that gains on a play.
struct Gain {
	/// The play's capped score after the move (the mean over the scenarios
	/// used).
	capped: f64,
	mv: Move,
}

/// The single joker moves that would gain on a play.
struct Order<'game> {
	play: Play<'game>,
	/// The play's capped score in the current order (the mean over the
	/// scenarios used).
	base: f64,
	/// The moves whose capped score gains on `base` and is no lower than the
	/// current order's in any scenario, best first. What they leave besides
	/// the score is not checked yet (see [`Play::clean`]).
	moves: Vec<Gain>,
}

impl<'game> Order<'game> {
	/// For the play `mask` of `g`, scored on the first shared scenario; if a
	/// score involved a random draw and `samples` allows, on the first
	/// `samples` scenarios instead. `None` if moves are not scored in `g`
	/// (see [`eligible`]), the play is not legal or it already reaches the
	/// chips still needed.
	fn new(
		g: &'game Game, mask: CardMask, samples: usize, can_reorder: bool,
	) -> Option<Order<'game>> {
		if !eligible(g.flow, &g.areas, can_reorder) {
			return None;
		}
		let need = g.blind.chips - g.round.chips;
		let mut play = Play {
			scenarios: Scenarios::new(g, samples),
			scratch: Scratch::default(),
			mask,
			sides: Vec::new(),
		};
		// Nothing to gain: skipped before any move is scored.
		let first = play.preview(0, None)?;
		if first.score >= need && (samples == 1 || !first.chance) {
			return None;
		}
		let mut scored = vec![play.scores(0, first, need)];
		if scored[0].chance {
			for k in 1..samples {
				let unmoved = play.preview(k, None)?;
				scored.push(play.scores(k, unmoved, need));
			}
		}
		let count = num::float(scored.len());
		let base = scored.iter().map(|s| s.unmoved).sum::<f64>() / count;
		if base >= need {
			return None;
		}
		let mut moves = Vec::new();
		for mv in single_moves(g.areas.jokers.len()) {
			let after = |s: &Scores| s.moved[mv.from][mv.to];
			let capped = scored.iter().map(after).sum::<f64>() / count;
			if gains(capped, base) && scored.iter().all(|s| after(s) >= s.unmoved) {
				moves.push(Gain { capped, mv });
			}
		}
		moves.sort_by(|a, b| b.capped.total_cmp(&a.capped));
		play.sides = vec![None; scored.len()];
		Some(Order { play, base, moves })
	}
}

/// Fills the gain column (`OP_ORDER_GAIN`) of the joker-move ops of `ops`
/// (those of `g`, as [`ops`](fn@super::ops) lists them): `ORDER_GAIN_SCALE`
/// times the log gain, at most 2, of the top play candidate under the best
/// single move of that joker, on the first shared scenario. It stays zero
/// below a gain of 1%, without a top candidate or one that already reaches
/// the chips still needed, and outside rounds with 2 to `ORDER_JOKERS`
/// jokers, the reorder ops on offer, nothing face down and no Ceremonial
/// Dagger.
pub fn order_gain(g: &Game, ops: &mut [Op]) {
	if !ops.iter().any(|o| o.kind == OpKind::MoveJoker) {
		return;
	}
	let Some(Op {
		kind: OpKind::PlayTop,
		cards: Some(top),
		..
	}) = ops.first()
	else {
		return;
	};
	let Some(mut order) = Order::new(g, top.required, 1, /* can_reorder= */ true) else {
		return;
	};
	// Each joker's first move in the list that leaves nothing else changed is its best.
	let mut best = [order.base; ORDER_JOKERS];
	for &Gain { capped, mv } in &order.moves {
		if best[mv.from] <= order.base && order.play.clean(mv) {
			best[mv.from] = capped;
		}
	}
	for o in ops.iter_mut().filter(|o| o.kind == OpKind::MoveJoker) {
		let gain = ((best[usize::from(o.index)] + 1.0) / (order.base + 1.0)).ln();
		o.feats[OP_ORDER_GAIN] = num::feat(ORDER_GAIN_SCALE * gain.min(2.0));
	}
}

/// The label of playing `mask` in `g`: per joker, the destination slots (bit
/// `to` of entry `from`) whose move scores within 0.1% of the best single
/// move, when that gains at least 1% on the current order (all zero when no
/// move does). Scored on the first shared scenario, or on all
/// `PREVIEW_SAMPLES` (mean capped score, a move no worse than the current
/// order in each) if a score involved a random draw. `None` where
/// [`order_gain`] stays zero for want of an eligible state, and when the
/// play already reaches the chips still needed.
#[must_use]
pub fn order_label(g: &Game, mask: CardMask) -> Option<[u8; ORDER_JOKERS]> {
	let can_reorder = g.flow.rearranges < MAX_REARRANGES;
	let mut order = Order::new(g, mask, PREVIEW_SAMPLES, can_reorder)?;
	let mut dest = [0u8; ORDER_JOKERS];
	let mut best = None;
	for &Gain { capped, mv } in &order.moves {
		if best.is_some_and(|b| capped < TIE * b) {
			break;
		}
		if order.play.clean(mv) {
			best.get_or_insert(capped);
			dest[mv.from] |= 1 << mv.to;
		}
	}
	Some(dest)
}
