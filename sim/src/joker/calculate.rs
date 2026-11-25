//! Port of `Card:calculate_joker(context)` (card.lua) dispatch and Blueprint /
//! Brainstorm copying. What each joker does in a context is in
//! [`scoring`](super::scoring), [`card_scoring`](super::card_scoring), and
//! [`lifecycle`](super::lifecycle), in the Lua's evaluation order, because
//! the order of RNG draws and of state mutations is observable.

use super::effect::{Effect, LifecycleCtx, Scoring, ScoringCtx};
use super::{lifecycle, scoring};
use crate::card::Kind;
use crate::data::Joker;
use crate::game::Game;

/// A `calculate_joker` context of either kind.
#[derive(Clone, Copy)]
enum Ctx<'a> {
	Scoring(ScoringCtx, &'a Scoring),
	Lifecycle(LifecycleCtx),
}

/// `Card:calculate_joker(context)` for the joker at `ji`, outside hand
/// scoring.
#[inline]
pub(crate) fn calc_joker(g: &mut Game, ji: usize, ctx: LifecycleCtx) -> Option<Effect> {
	calc_joker_copy(g, ji, Ctx::Lifecycle(ctx), 0, None)
}

/// Every joker's `calculate_joker(context)`, in order, for a context whose
/// effects need no handling.
pub(crate) fn calc_jokers(g: &mut Game, ctx: LifecycleCtx) {
	for ji in 0..g.areas.jokers.len() {
		calc_joker(g, ji, ctx);
	}
}

/// `Card:calculate_joker(context)` for the joker at `ji`, while the hand
/// `sc` is scored.
#[inline]
pub(crate) fn calc_scoring(
	g: &mut Game, ji: usize, ctx: ScoringCtx, sc: &Scoring,
) -> Option<Effect> {
	calc_joker_copy(g, ji, Ctx::Scoring(ctx, sc), 0, None)
}

/// `calculate_joker` as reached through a chain of `depth` Blueprint /
/// Brainstorm copies, on behalf of the joker at `copier`.
fn calc_joker_copy(
	g: &mut Game, ji: usize, ctx: Ctx<'_>, depth: usize, copier: Option<usize>,
) -> Option<Effect> {
	use Joker as J;

	let card = &g.areas.jokers[ji];
	if card.core.debuff {
		return None;
	}
	let Kind::Joker(j) = card.kind else {
		return None;
	};

	// Copycats come first and fall through to their own (empty) logic.
	if j == J::Blueprint || j == J::Brainstorm {
		let other = if j == J::Blueprint {
			(ji + 1 < g.areas.jokers.len()).then_some(ji + 1)
		} else {
			Some(0)
		};
		if let Some(o) = other
			&& o != ji
		{
			let depth = depth + 1;
			if depth > g.areas.jokers.len() + 1 {
				return None;
			}
			if let Some(r) = calc_joker_copy(g, o, ctx, depth, Some(copier.unwrap_or(ji))) {
				return Some(r);
			}
		}
	}

	match ctx {
		Ctx::Scoring(ctx, sc) => scoring::calculate(g, ji, j, ctx, sc, copier),
		Ctx::Lifecycle(ctx) => lifecycle::calculate(g, ji, j, ctx, copier),
	}
}
