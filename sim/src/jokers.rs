//! Port of `Card:calculate_joker(context)` (card.lua). The branch structure and
//! evaluation order mirror the Lua if-chain exactly, because the order of
//! RNG draws and of state mutations is observable. This module holds the
//! contexts, the effect table and the dispatch (with Blueprint / Brainstorm
//! copying); the contexts themselves are in `lifecycle` (shop, blind, discard,
//! end of round) and `scoring` (hand scoring).

use crate::card::Kind;
use crate::data::{HandType, Joker};
use crate::ev::Ev;
use crate::game::Game;
use crate::hand_eval::PokerHands;
use crate::{joker_lifecycle, joker_scoring, num};

/// Shared data of the hand being scored.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Scoring {
	pub name: HandType,
	pub hands: PokerHands,
	/// Indices into `G.play` of the scoring hand, in table order.
	pub cards: [u8; 8],
	pub n_cards: u8,
}

impl Scoring {
	#[inline]
	pub fn scoring(&self) -> &[u8] {
		&self.cards[..self.n_cards as usize]
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
	PlayingCardAdded {
		n: usize,
	},
	FirstHandDrawn,
	SettingBlind {
		boss: bool,
	},
	DestroyingCard {
		play_idx: usize,
		full_len: usize,
	},
	RemovePlayingCards {
		faces: usize,
		glass: usize,
	},
	UsingConsumeable {
		kind: Kind,
		hanged_glass: usize,
	},
	DebuffedHand,
	PreDiscard {
		hook: bool,
	},
	/// `other` and `last` index `G.hand`; `n_full` / `full_faces` count the
	/// discarded (highlighted) cards and their face cards.
	Discard {
		other: usize,
		last: usize,
		n_full: usize,
		full_faces: usize,
	},
	EndOfRound {
		game_over: bool,
	},
	EndOfRoundRepetition {
		effects: bool,
	},
	Individual {
		area: CardArea,
		other: usize,
	},
	Repetition {
		area: CardArea,
		other: usize,
		effects: bool,
	},
	/// `other`: the joker being scored (`None` for a consumable).
	OtherJoker {
		other: Option<usize>,
	},
	Before,
	After,
	Main,
}

/// A joker's returned effect table.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[expect(
	clippy::struct_excessive_bools,
	reason = "independent flags, one per field of the Lua effect table"
)]
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

pub(crate) use ret;

#[inline]
pub(crate) fn jab(g: &mut Game, ji: usize) -> &mut crate::card::Ability {
	&mut g.jokers[ji].ab
}

/// Schedules the "eaten/melted" self-destruction of a joker
/// (`G.jokers:remove_card(self); self:remove()` in nested events).
pub(crate) fn schedule_self_destruct(g: &mut Game, ji: usize) {
	let id = g.jokers[ji].id;
	g.events.push(Ev::EatenOuter { id });
}

/// Whether the consumable area has room for one more card (counting
/// cards already promised by queued creations).
#[inline]
pub(crate) fn consumable_room(g: &Game) -> bool {
	(num::float(g.consumables.len()) + g.consumeable_buffer) < g.consumable_limit.limit
}

/// Money promised by a scoring effect before its `ease_dollars` event
/// runs (`G.GAME.dollar_buffer`).
#[inline]
pub(crate) fn buffer_dollars(g: &mut Game, d: f64) {
	g.dollar_buffer += d;
	g.events.push(Ev::DollarBufferReset);
}

/// `Card:calculate_joker(context)` for the joker at `ji`.
#[inline]
pub(crate) fn calc_joker(
	g: &mut Game, ji: usize, ctx: &Ctx, sc: Option<&Scoring>,
) -> Option<Effect> {
	calc_joker_copy(g, ji, ctx, sc, 0, None)
}

/// `calculate_joker` as reached through `bp` Blueprint / Brainstorm
/// copies on behalf of the joker at `bp_card`.
fn calc_joker_copy(
	g: &mut Game, ji: usize, ctx: &Ctx, sc: Option<&Scoring>, bp: u32, bp_card: Option<usize>,
) -> Option<Effect> {
	use Joker as J;

	let card = &g.jokers[ji];
	if card.debuff {
		return None;
	}
	let Kind::Joker(j) = card.kind else {
		return None;
	};
	let blueprinted = bp > 0;
	let actor = bp_card.unwrap_or(ji);

	// Copycats come first and fall through to their own (empty) logic.
	if j == J::Blueprint || j == J::Brainstorm {
		let other = if j == J::Blueprint {
			if ji + 1 < g.jokers.len() {
				Some(ji + 1)
			} else {
				None
			}
		} else if !g.jokers.is_empty() {
			Some(0)
		} else {
			None
		};
		if let Some(o) = other
			&& o != ji
		{
			let depth = bp + 1;
			let bcard = Some(bp_card.unwrap_or(ji));
			if depth as usize > g.jokers.len() + 1 {
				return None;
			}
			if let Some(r) = calc_joker_copy(g, o, ctx, sc, depth, bcard) {
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
		| Ctx::Main => joker_scoring::calculate(g, ji, j, ctx, sc, blueprinted),
		_ => joker_lifecycle::calculate(g, ji, j, ctx, blueprinted, actor),
	}
}

/// `card_eval_status_text(joker, 'jokers', nil, percent, nil, eval)`.
#[inline]
pub(crate) fn effect_status(g: &mut Game, e: &Effect) {
	g.events.status_text(e.delay.unwrap_or(0.75));
}

/// `playing_card_joker_effects(cards)`.
pub(crate) fn playing_card_joker_effects(g: &mut Game, n: usize) {
	for i in 0..g.jokers.len() {
		calc_joker(g, i, &Ctx::PlayingCardAdded { n }, None);
	}
}
