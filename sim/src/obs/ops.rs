//! The factored action space: legal ops, decoding a choice into an
//! [`Action`], and a fallback.

use super::encode::sl;
use super::{Layout, MAX_HAND, MAX_OPS, OP_FLOAT, OP_INT, op, top_plays};
use crate::actions::{Action, CardMask};
use crate::card::{Card, Kind};
use crate::consumables::{consumable_highlight, uses_targets};
use crate::data::{Edition, Spectral};
use crate::game::{Game, Phase};
use crate::{actions, num};

/// Consecutive free reorders (hand moves / sorts, joker moves) offered before
/// another action is required, so a policy cannot stall a game by
/// reordering (a trained policy padded turns up to a cap of 10). Two: run13's
/// policy shuffled jokers up to the old cap of 5 (35 moves a game, 16% undone
/// at once), and a paired eval scored cap 2 = cap 5 (-0.1 pts) and cap 1 -1.0.
pub const MAX_REARRANGES: u8 = 2;

/// One legal op.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Op {
	pub kind: i32,
	/// Token index of the op's subject (-1 = none).
	pub target: i32,
	/// Index within its area (shop slot, joker index, ...).
	pub index: u8,
	/// Card-set argument: `card_max == 0` means none.
	pub card_min: u8,
	pub card_max: u8,
	/// Cards that may be chosen / must be included (forced selections).
	pub allowed: CardMask,
	pub required: CardMask,
	/// Destination slots (reordering ops), 0 = none.
	pub dest: u8,
	/// `OP_FLOAT` features (zero except for `PLAY_TOP`).
	pub feats: [f32; OP_FLOAT],
}

impl Op {
	fn simple(kind: i32) -> Op {
		Op {
			kind,
			target: -1,
			index: 0,
			card_min: 0,
			card_max: 0,
			allowed: 0,
			required: 0,
			dest: 0,
			feats: [0.0; OP_FLOAT],
		}
	}

	fn on(kind: i32, target: usize, index: usize) -> Op {
		Op {
			target: num::id(target),
			index: num::byte(index),
			..Op::simple(kind)
		}
	}

	/// The op as `OP_INT` integers (masks reinterpreted as `i32`).
	#[must_use]
	pub fn ints(&self) -> [i32; OP_INT] {
		[
			self.kind,
			self.target,
			i32::from(self.card_min),
			i32::from(self.card_max),
			self.allowed.cast_signed(),
			self.required.cast_signed(),
			i32::from(self.dest),
			i32::from(self.index),
		]
	}
}

/// A bit count or bit index of a [`CardMask`] as the `u8` that ops carry.
fn bits(n: u32) -> u8 {
	u8::try_from(n).expect("a card mask has 32 bits")
}

fn hand_mask(n: usize) -> CardMask {
	if n >= 32 { u32::MAX } else { (1u32 << n) - 1 }
}

/// Legal ops of `g` (given the layout of its encoded observation). In the
/// round phase up to `candidates` scored `PLAY_TOP` ops come first, in the
/// room the other ops leave. Ops on subjects past the token limit (which
/// the observation does not show) are dropped.
pub fn ops(g: &Game, lay: &Layout, candidates: usize, out: &mut Vec<Op>) {
	out.clear();
	base_ops(g, lay, out);
	out.retain(|o| o.target < num::id(lay.n_tokens));
	out.truncate(MAX_OPS);
	if candidates > 0 && g.phase == Phase::Round {
		let need = (g.blind.chips - g.chips).max(1.0);
		let top = top_plays(g, candidates.min(MAX_OPS - out.len()));
		out.splice(
			0..0,
			top.into_iter().map(|c| {
				let n = bits(c.mask.count_ones());
				Op {
					card_min: n,
					card_max: n,
					allowed: c.mask,
					required: c.mask,
					feats: [
						sl(c.mean),
						num::feat((c.mean / need).min(4.0)),
						num::feat(c.p_clear),
						sl(c.min),
						sl(c.max),
						f32::from(n) / 5.0,
					],
					..Op::simple(op::PLAY_TOP)
				}
			}),
		);
	}
}

/// The ops other than scored candidates, in their fixed order.
#[expect(
	clippy::too_many_lines,
	reason = "one arm per phase; the order of the ops is part of the action space"
)]
fn base_ops(g: &Game, lay: &Layout, out: &mut Vec<Op>) {
	let nh = g.hand.len().min(MAX_HAND);
	let all = hand_mask(nh);
	let forced = actions::forced_mask(g);
	let ok = |a: Action| g.check(&a).is_ok();
	// Card-set ops with targets: find the legal set sizes with
	// representative selections (legality only depends on the count).
	let targeted = |c: &Card, mk: &dyn Fn(CardMask) -> Action, base: Op, out: &mut Vec<Op>| {
		let (max_h, min_h) = consumable_highlight(c.kind);
		if !uses_targets(c.kind) {
			if ok(mk(0)) {
				out.push(base);
			}
			return;
		}
		let max = if max_h == 0 { 1 } else { max_h.min(5) };
		let allowed = if c.kind == Kind::Spectral(Spectral::Aura) {
			g.hand
				.iter()
				.take(MAX_HAND)
				.enumerate()
				.filter(|(_, c)| c.edition == Edition::None)
				.fold(0u32, |m, (i, _)| m | 1 << i)
		} else {
			all
		};
		let (mut lo, mut hi) = (0u8, 0u8);
		for k in min_h.max(1)..=max {
			// forced cards first, then the lowest allowed others
			let mut m = forced;
			let mut i = 0;
			while bits(m.count_ones()) < k && i < nh {
				if allowed >> i & 1 == 1 {
					m |= 1 << i;
				}
				i += 1;
			}
			if bits(m.count_ones()) == k && ok(mk(m)) {
				if lo == 0 {
					lo = k;
				}
				hi = k;
			}
		}
		if lo > 0 {
			out.push(Op {
				card_min: lo,
				card_max: hi,
				allowed: allowed | forced,
				required: forced,
				..base
			});
		}
	};
	match g.phase {
		Phase::GameOver => return,
		Phase::BlindSelect => {
			for (a, k) in [
				(Action::SelectBlind, op::SELECT_BLIND),
				(Action::SkipBlind, op::SKIP_BLIND),
				(Action::RerollBoss, op::REROLL_BOSS),
			] {
				if (k != op::SKIP_BLIND || g.allow_skip) && ok(a) {
					out.push(Op::simple(k));
				}
			}
		}
		Phase::Round => {
			if nh > 0 && forced.count_ones() <= 5 {
				let base = Op {
					card_min: 1,
					card_max: 5,
					allowed: all,
					required: forced,
					..Op::simple(op::PLAY)
				};
				out.push(base);
				if g.discards_left > 0.0 {
					out.push(Op {
						kind: op::DISCARD,
						..base
					});
				}
			}
		}
		Phase::Shop => {
			for i in 0..g.shop_jokers.len() {
				if ok(Action::BuyShopCard(num::byte(i))) {
					out.push(Op::on(op::BUY, lay.shop_cards + i, i));
				}
				let c = &g.shop_jokers[i];
				if c.kind.is_consumable() {
					targeted(
						c,
						&|t| Action::BuyAndUse {
							index: num::byte(i),
							targets: t,
						},
						Op::on(op::BUY_AND_USE, lay.shop_cards + i, i),
						out,
					);
				}
			}
			for i in 0..g.shop_vouchers.len() {
				if ok(Action::RedeemVoucher(num::byte(i))) {
					out.push(Op::on(op::REDEEM, lay.shop_vouchers + i, i));
				}
			}
			for i in 0..g.shop_boosters.len() {
				if ok(Action::OpenBooster(num::byte(i))) {
					out.push(Op::on(op::OPEN, lay.shop_boosters + i, i));
				}
			}
			if ok(Action::Reroll) {
				out.push(Op::simple(op::REROLL));
			}
			out.push(Op::simple(op::END_SHOP));
		}
		Phase::Pack => {
			for i in 0..g.pack_cards.len() {
				let c = &g.pack_cards[i];
				let base = Op::on(op::PICK, lay.pack + i, i);
				if c.kind.is_consumable() {
					targeted(
						c,
						&|t| Action::PickPackCard {
							index: num::byte(i),
							targets: t,
						},
						base,
						out,
					);
				} else if ok(Action::PickPackCard {
					index: num::byte(i),
					targets: 0,
				}) {
					out.push(base);
				}
			}
			if ok(Action::SkipPack) {
				out.push(Op::simple(op::SKIP_PACK));
			}
		}
		Phase::RoundEval => out.push(Op::simple(op::CASH_OUT)),
	}
	// Reordering the hand (in rounds, and in packs opened with it drawn).
	let can_reorder = g.rearranges < MAX_REARRANGES;
	if can_reorder && matches!(g.phase, Phase::Round | Phase::Pack) && nh >= 2 {
		out.push(Op::simple(op::SORT_RANK));
		out.push(Op::simple(op::SORT_SUIT));
		out.push(Op {
			card_min: 1,
			card_max: 1,
			allowed: all,
			required: 0,
			dest: num::byte(nh),
			..Op::simple(op::MOVE_HAND)
		});
	}
	for i in 0..g.consumables.len() {
		let c = &g.consumables[i];
		targeted(
			c,
			&|t| Action::UseConsumable {
				index: num::byte(i),
				targets: t,
			},
			Op::on(op::USE, lay.consumables + i, i),
			out,
		);
	}
	for i in 0..g.jokers.len() {
		if ok(Action::SellJoker(num::byte(i))) {
			out.push(Op::on(op::SELL_JOKER, lay.jokers + i, i));
		}
	}
	for i in 0..g.consumables.len() {
		if ok(Action::SellConsumable(num::byte(i))) {
			out.push(Op::on(op::SELL_CONSUMABLE, lay.consumables + i, i));
		}
	}
	if can_reorder && g.jokers.len() >= 2 {
		for i in 0..g.jokers.len() {
			out.push(Op {
				dest: num::byte(g.jokers.len()),
				..Op::on(op::MOVE_JOKER, lay.jokers + i, i)
			});
		}
	}
}

/// The [`Action`] for `op` with the chosen cards and destination, if legal.
/// `cards` may omit the op's required cards (they are always added).
#[must_use]
#[expect(
	clippy::many_single_char_names,
	reason = "the short names used across the obs modules: game, op, mask, action"
)]
pub fn decode(g: &Game, o: &Op, cards: CardMask, dest: u32) -> Option<Action> {
	let m = (cards & o.allowed) | o.required;
	let dest = || u8::try_from(dest).ok().filter(|&d| d < o.dest);
	let a = match o.kind {
		op::SELECT_BLIND => Action::SelectBlind,
		op::SKIP_BLIND => Action::SkipBlind,
		op::REROLL_BOSS => Action::RerollBoss,
		op::PLAY | op::PLAY_TOP => Action::Play(m),
		op::DISCARD => Action::Discard(m),
		op::USE => Action::UseConsumable {
			index: o.index,
			targets: if o.card_max > 0 { m } else { 0 },
		},
		op::SELL_JOKER => Action::SellJoker(o.index),
		op::SELL_CONSUMABLE => Action::SellConsumable(o.index),
		op::BUY => Action::BuyShopCard(o.index),
		op::BUY_AND_USE => Action::BuyAndUse {
			index: o.index,
			targets: if o.card_max > 0 { m } else { 0 },
		},
		op::REDEEM => Action::RedeemVoucher(o.index),
		op::OPEN => Action::OpenBooster(o.index),
		op::REROLL => Action::Reroll,
		op::END_SHOP => Action::EndShop,
		op::CASH_OUT => Action::CashOut,
		op::PICK => Action::PickPackCard {
			index: o.index,
			targets: if o.card_max > 0 { m } else { 0 },
		},
		op::SKIP_PACK => Action::SkipPack,
		op::MOVE_JOKER => Action::MoveJoker {
			from: o.index,
			to: dest()?,
		},
		op::SORT_RANK => Action::SortHand { by_suit: false },
		op::SORT_SUIT => Action::SortHand { by_suit: true },
		op::MOVE_HAND => {
			if m.count_ones() != 1 {
				return None;
			}
			Action::MoveHand {
				from: bits(m.trailing_zeros()),
				to: dest()?,
			}
		}
		_ => return None,
	};
	if o.card_max > 0 {
		let n = bits(m.count_ones());
		if n < o.card_min || n > o.card_max {
			return None;
		}
	}
	g.check(&a).ok().map(|()| a)
}

/// A legal fallback action (the first op with its minimal card set).
#[must_use]
pub fn fallback(g: &Game, ops: &[Op]) -> Option<Action> {
	for o in ops {
		let mut m = o.required;
		let mut i = 0;
		while bits(m.count_ones()) < o.card_min && i < 32 {
			if o.allowed >> i & 1 == 1 {
				m |= 1 << i;
			}
			i += 1;
		}
		let dest = if o.dest > 0 {
			(u32::from(o.index) + 1) % u32::from(o.dest)
		} else {
			0
		};
		if let Some(a) = decode(g, o, m, dest) {
			return Some(a);
		}
	}
	let mut v = Vec::new();
	g.legal_actions(&mut v);
	v.first().copied()
}
