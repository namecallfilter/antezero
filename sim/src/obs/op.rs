use super::layout::shown_hand;
use super::tokens::{null_sort, signed_log};
use super::{Layout, MAX_OPS, OP_FLOAT, OP_INT, top_plays};
use crate::actions::Action;
use crate::card::{AnyCard, Card, CardMask, Kind};
use crate::data::Spectral;
use crate::game::{Game, Hand, HandSort, Phase};
use crate::{actions, consumables, num};

/// Consecutive free reorders (hand moves / sorts, joker moves) offered before
/// another action is required, so a policy cannot stall a game by
/// reordering.
// Two: a paired eval scored cap 2 = cap 5 (-0.1 pts) and cap 1 -1.0.
pub const MAX_REARRANGES: u8 = 2;

/// What an op does; the discriminant is the op's kind feature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum OpKind {
	SelectBlind,
	SkipBlind,
	RerollBoss,
	Play,
	Discard,
	Use,
	SellJoker,
	SellConsumable,
	Buy,
	BuyAndUse,
	Redeem,
	Open,
	Reroll,
	EndShop,
	CashOut,
	Pick,
	SkipPack,
	MoveJoker,
	SortRank,
	SortSuit,
	MoveHand,
	/// One of the top-scoring plays (cards fixed; see `ops`).
	PlayTop,
}

impl OpKind {
	pub const COUNT: i32 = OpKind::PlayTop.id() + 1;

	/// The kind feature.
	#[must_use]
	pub const fn id(self) -> i32 {
		self as i32
	}
}

/// The set of hand cards an op takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CardArg {
	/// The fewest cards the set holds.
	pub min: u8,
	/// The most cards the set holds.
	pub max: u8,
	/// Cards that may be chosen.
	pub allowed: CardMask,
	/// Cards that must be included (forced selections).
	pub required: CardMask,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Op {
	pub kind: OpKind,
	/// Token index of the op's subject.
	pub target: Option<usize>,
	/// Index within its area (shop slot, joker index, ...).
	pub index: u8,
	/// The hand cards the op takes, if any.
	pub cards: Option<CardArg>,
	/// Destination slots (reordering ops).
	pub dest: Option<u8>,
	/// `OP_FLOAT` features (zero except for [`OpKind::PlayTop`] and, once
	/// [`order_gain`](super::order_gain) has run, [`OpKind::MoveJoker`]).
	pub feats: [f32; OP_FLOAT],
}

impl Op {
	fn simple(kind: OpKind) -> Op {
		Op {
			kind,
			target: None,
			index: 0,
			cards: None,
			dest: None,
			feats: [0.0; OP_FLOAT],
		}
	}

	fn on(kind: OpKind, target: usize, index: usize) -> Op {
		Op {
			target: Some(target),
			index: num::byte(index),
			..Op::simple(kind)
		}
	}

	/// The smallest card set the op takes: the required cards, then the
	/// lowest allowed ones up to its minimum.
	#[must_use]
	pub fn minimal_cards(&self) -> CardMask {
		self.cards.map_or(CardMask::EMPTY, |c| {
			c.required.fill_lowest(c.allowed, usize::from(c.min))
		})
	}

	/// The `OP_INT` columns: an op without cards or destination has 0 there.
	#[must_use]
	pub fn ints(&self) -> [i32; OP_INT] {
		let cards = self.cards.unwrap_or(CardArg {
			min: 0,
			max: 0,
			allowed: CardMask::EMPTY,
			required: CardMask::EMPTY,
		});
		[
			self.kind.id(),
			self.target.map_or(-1, num::id),
			i32::from(cards.min),
			i32::from(cards.max),
			cards.allowed.bits().cast_signed(),
			cards.required.bits().cast_signed(),
			i32::from(self.dest.unwrap_or(0)),
			i32::from(self.index),
		]
	}
}

/// What [`ops`] offers besides the plain legal moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpOptions {
	/// Scored [`OpKind::PlayTop`] ops to put first in rounds.
	pub candidates: usize,
	/// Whether skipping a blind is offered (a training switch: the game
	/// itself always allows skips).
	pub allow_skip: bool,
	/// Whether a sort that would move no card the player can tell apart is
	/// offered (a training switch: a policy can pad its turns with it).
	pub null_sorts: bool,
	/// Maximum consecutive free reorders (hand moves / sorts, joker moves)
	/// offered before another action is required.
	pub max_rearranges: u8,
}

impl Default for OpOptions {
	fn default() -> OpOptions {
		OpOptions {
			candidates: 0,
			allow_skip: true,
			null_sorts: true,
			max_rearranges: MAX_REARRANGES,
		}
	}
}

/// Legal ops of `g` (given the layout of its encoded observation). In the
/// round phase up to `options.candidates` scored [`OpKind::PlayTop`] ops
/// come first, in the room the other ops leave. Ops on subjects past the
/// token limit (which the observation does not show) are dropped.
pub fn ops(g: &Game, lay: &Layout, options: OpOptions, out: &mut Vec<Op>) {
	out.clear();
	base_ops(g, lay, options, out);
	out.retain(|o| o.target.is_none_or(|token| token < lay.n_tokens));
	out.truncate(MAX_OPS);
	if options.candidates > 0 && g.flow.phase == Phase::Round {
		let need = (g.blind.chips - g.round.chips).max(1.0);
		let top = top_plays(g, options.candidates.min(MAX_OPS - out.len()));
		out.splice(
			0..0,
			top.into_iter().map(|c| {
				let n = num::byte(c.mask.len());
				Op {
					cards: Some(CardArg {
						min: n,
						max: n,
						allowed: c.mask,
						required: c.mask,
					}),
					feats: [
						signed_log(c.mean),
						num::feat((c.mean / need).min(4.0)),
						num::feat(c.p_clear),
						signed_log(c.min),
						signed_log(c.max),
						f32::from(n) / f32::from(num::byte(Hand::HIGHLIGHT_LIMIT)),
						0.0,
					],
					..Op::simple(OpKind::PlayTop)
				}
			}),
		);
	}
}

fn legal(g: &Game, a: Action) -> bool {
	g.check(a).is_ok()
}

/// The op `base` of `consumable`, used by the action `mk` makes of the
/// chosen cards: pushed if some selection is legal. For a consumable with
/// targets, the legal set sizes are found with representative selections
/// (legality only depends on the count).
fn targeted(
	g: &Game, consumable: &Card, mk: impl Fn(CardMask) -> Action, base: Op, out: &mut Vec<Op>,
) {
	let Some(targets) = consumables::targets(consumable.kind) else {
		if legal(g, mk(CardMask::EMPTY)) {
			out.push(base);
		}
		return;
	};
	let hand = shown_hand(&g.areas.hand.cards);
	let Some(forced) = actions::forced_mask(&g.areas.hand.cards) else {
		return;
	};
	let allowed = if consumable.kind == Kind::Spectral(Spectral::Aura) {
		hand.iter()
			.enumerate()
			.filter(|(_, card)| card.core.edition.is_none())
			.map(|(i, _)| i)
			.collect()
	} else {
		CardMask::first(hand.len())
	};
	let mut sizes: Option<(u8, u8)> = None;
	for k in targets.min..=targets.max {
		let m = forced.fill_lowest(allowed, k);
		if m.len() == k && legal(g, mk(m)) {
			let k = num::byte(k);
			sizes = Some((sizes.map_or(k, |(min, _)| min), k));
		}
	}
	if let Some((min, max)) = sizes {
		out.push(Op {
			cards: Some(CardArg {
				min,
				max,
				allowed: allowed | forced,
				required: forced,
			}),
			..base
		});
	}
}

fn shop_ops(g: &Game, lay: &Layout, out: &mut Vec<Op>) {
	for (i, c) in g.areas.shop_jokers.iter().enumerate() {
		if legal(g, Action::BuyShopCard(num::byte(i))) {
			out.push(Op::on(OpKind::Buy, lay.shop_cards + i, i));
		}
		if let AnyCard::Card(c) = c
			&& c.kind.is_consumable()
		{
			targeted(
				g,
				c,
				|t| Action::BuyAndUse {
					index: num::byte(i),
					targets: t,
				},
				Op::on(OpKind::BuyAndUse, lay.shop_cards + i, i),
				out,
			);
		}
	}
	for i in 0..g.areas.shop_vouchers.len() {
		if legal(g, Action::RedeemVoucher(num::byte(i))) {
			out.push(Op::on(OpKind::Redeem, lay.shop_vouchers + i, i));
		}
	}
	for i in 0..g.areas.shop_boosters.len() {
		if legal(g, Action::OpenBooster(num::byte(i))) {
			out.push(Op::on(OpKind::Open, lay.shop_boosters + i, i));
		}
	}
	if legal(g, Action::Reroll) {
		out.push(Op::simple(OpKind::Reroll));
	}
	out.push(Op::simple(OpKind::EndShop));
}

fn pack_ops(g: &Game, lay: &Layout, out: &mut Vec<Op>) {
	for (i, c) in g.areas.pack_cards.iter().enumerate() {
		let base = Op::on(OpKind::Pick, lay.pack + i, i);
		if let AnyCard::Card(c) = c
			&& c.kind.is_consumable()
		{
			targeted(
				g,
				c,
				|t| Action::PickPackCard {
					index: num::byte(i),
					targets: t,
				},
				base,
				out,
			);
		} else if legal(g, Action::PickPackCard {
			index: num::byte(i),
			targets: CardMask::EMPTY,
		}) {
			out.push(base);
		}
	}
	if legal(g, Action::SkipPack) {
		out.push(Op::simple(OpKind::SkipPack));
	}
}

/// Whether another free reorder is offered.
fn can_reorder(g: &Game, options: OpOptions) -> bool {
	g.flow.rearranges < options.max_rearranges
}

/// The ops that reorder the hand: the sorts and the move of a shown card
/// (in rounds, and in packs opened with the hand drawn), and the move of a
/// held card past the shown hand.
fn hand_reorder_ops(g: &Game, options: OpOptions, out: &mut Vec<Op>) {
	let hand = &g.areas.hand;
	let nh = shown_hand(&hand.cards).len();
	if can_reorder(g, options) && matches!(g.flow.phase, Phase::Round | Phase::Pack) && nh >= 2 {
		let null = !options.null_sorts && null_sort(hand);
		for (order, kind) in [
			(HandSort::Rank, OpKind::SortRank),
			(HandSort::Suit, OpKind::SortSuit),
		] {
			if !(null && hand.order == order) {
				out.push(Op::simple(kind));
			}
		}
		out.push(Op {
			cards: Some(CardArg {
				min: 1,
				max: 1,
				allowed: CardMask::first(nh),
				required: CardMask::EMPTY,
			}),
			dest: Some(num::byte(nh)),
			..Op::simple(OpKind::MoveHand)
		});
	}
	// A held card past the shown hand is in no selection, so none is legal,
	// and no op above names it: an op of its own moves it into the shown
	// hand, whatever the reorder allowance.
	for (i, c) in hand.cards.iter().enumerate().skip(nh) {
		if c.forced.is_locked() {
			out.push(Op {
				index: num::byte(i),
				dest: Some(num::byte(nh)),
				..Op::simple(OpKind::MoveHand)
			});
		}
	}
}

/// The ops of the current phase, then the ones open in every phase: hand
/// reordering, using and selling what is owned, joker reordering. The order
/// is part of the action space.
fn base_ops(g: &Game, lay: &Layout, options: OpOptions, out: &mut Vec<Op>) {
	let nh = shown_hand(&g.areas.hand.cards).len();
	let all = CardMask::first(nh);
	match g.flow.phase {
		Phase::GameOver => return,
		Phase::BlindSelect => {
			for (a, k) in [
				(Action::SelectBlind, OpKind::SelectBlind),
				(Action::SkipBlind, OpKind::SkipBlind),
				(Action::RerollBoss, OpKind::RerollBoss),
			] {
				if (k != OpKind::SkipBlind || options.allow_skip) && legal(g, a) {
					out.push(Op::simple(k));
				}
			}
		}
		Phase::Round => {
			if let Some(forced) = actions::forced_mask(&g.areas.hand.cards)
				&& nh > 0
				&& forced.len() <= Hand::HIGHLIGHT_LIMIT
			{
				let base = Op {
					cards: Some(CardArg {
						min: 1,
						max: num::byte(Hand::HIGHLIGHT_LIMIT),
						allowed: all,
						required: forced,
					}),
					..Op::simple(OpKind::Play)
				};
				out.push(base);
				if g.round.discards_left > 0.0 {
					out.push(Op {
						kind: OpKind::Discard,
						..base
					});
				}
			}
		}
		Phase::Shop => shop_ops(g, lay, out),
		Phase::Pack => pack_ops(g, lay, out),
		Phase::RoundEval => out.push(Op::simple(OpKind::CashOut)),
	}
	hand_reorder_ops(g, options, out);
	for (i, c) in g.areas.consumables.iter().enumerate() {
		targeted(
			g,
			c,
			|t| Action::UseConsumable {
				index: num::byte(i),
				targets: t,
			},
			Op::on(OpKind::Use, lay.consumables + i, i),
			out,
		);
	}
	for i in 0..g.areas.jokers.len() {
		if legal(g, Action::SellJoker(num::byte(i))) {
			out.push(Op::on(OpKind::SellJoker, lay.jokers + i, i));
		}
	}
	for i in 0..g.areas.consumables.len() {
		if legal(g, Action::SellConsumable(num::byte(i))) {
			out.push(Op::on(OpKind::SellConsumable, lay.consumables + i, i));
		}
	}
	if can_reorder(g, options) && g.areas.jokers.len() >= 2 {
		for i in 0..g.areas.jokers.len() {
			out.push(Op {
				dest: Some(num::byte(g.areas.jokers.len())),
				..Op::on(OpKind::MoveJoker, lay.jokers + i, i)
			});
		}
	}
}

/// The [`Action`] for `op` with the chosen cards and destination, if legal.
/// `cards` may omit the op's required cards (they are always added).
#[must_use]
pub fn decode(g: &Game, o: &Op, cards: CardMask, dest: Option<u8>) -> Option<Action> {
	let mask = o
		.cards
		.map_or(CardMask::EMPTY, |c| (cards & c.allowed) | c.required);
	let to_slot = || dest.filter(|&d| o.dest.is_some_and(|slots| d < slots));
	let action = match o.kind {
		OpKind::SelectBlind => Action::SelectBlind,
		OpKind::SkipBlind => Action::SkipBlind,
		OpKind::RerollBoss => Action::RerollBoss,
		OpKind::Play | OpKind::PlayTop => Action::Play(mask),
		OpKind::Discard => Action::Discard(mask),
		OpKind::Use => Action::UseConsumable {
			index: o.index,
			targets: mask,
		},
		OpKind::SellJoker => Action::SellJoker(o.index),
		OpKind::SellConsumable => Action::SellConsumable(o.index),
		OpKind::Buy => Action::BuyShopCard(o.index),
		OpKind::BuyAndUse => Action::BuyAndUse {
			index: o.index,
			targets: mask,
		},
		OpKind::Redeem => Action::RedeemVoucher(o.index),
		OpKind::Open => Action::OpenBooster(o.index),
		OpKind::Reroll => Action::Reroll,
		OpKind::EndShop => Action::EndShop,
		OpKind::CashOut => Action::CashOut,
		OpKind::Pick => Action::PickPackCard {
			index: o.index,
			targets: mask,
		},
		OpKind::SkipPack => Action::SkipPack,
		OpKind::MoveJoker => Action::MoveJoker {
			from: o.index,
			to: to_slot()?,
		},
		OpKind::SortRank => Action::SortHand(HandSort::Rank),
		OpKind::SortSuit => Action::SortHand(HandSort::Suit),
		OpKind::MoveHand => {
			let from = match o.cards {
				Some(_) => num::byte(mask.lowest().filter(|_| mask.len() == 1)?),
				// The op of a held card past the shown hand.
				None => o.index,
			};
			Action::MoveHand {
				from,
				to: to_slot()?,
			}
		}
	};
	if let Some(c) = o.cards {
		let count = mask.len();
		if count < usize::from(c.min) || count > usize::from(c.max) {
			return None;
		}
	}
	g.check(action).ok().map(|()| action)
}

/// A legal fallback action (the first op with its minimal card set).
#[must_use]
pub fn fallback(g: &Game, ops: &[Op]) -> Option<Action> {
	for o in ops {
		let m = o.minimal_cards();
		let dest = o
			.dest
			.map(|slots| num::byte((usize::from(o.index) + 1) % usize::from(slots)));
		if let Some(a) = decode(g, o, m, dest) {
			return Some(a);
		}
	}
	let mut v = Vec::new();
	g.legal_actions(&mut v);
	v.first().copied()
}
