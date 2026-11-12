//! The agent-facing action interface: enumeration, validation and `step`.

use std::fmt;
use std::str::FromStr;

use crate::card::Kind;
use crate::consumables::{consumable_highlight, uses_targets};
use crate::data::{Edition, PackKind, Spectral, Voucher};
use crate::game::{Area, BlindSlot, Game, Phase};
use crate::{consumables, events, flow, num, packs, round_end, shop};

/// Hand-card selection as a bitmask over `Game::hand` indices.
pub type CardMask = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Action {
	SelectBlind,
	SkipBlind,
	RerollBoss,
	Play(CardMask),
	Discard(CardMask),
	/// Use consumable `index` with the given hand cards highlighted.
	UseConsumable {
		index: u8,
		targets: CardMask,
	},
	SellJoker(u8),
	SellConsumable(u8),
	MoveJoker {
		from: u8,
		to: u8,
	},
	MoveHand {
		from: u8,
		to: u8,
	},
	SortHand {
		by_suit: bool,
	},
	BuyShopCard(u8),
	BuyAndUse {
		index: u8,
		targets: CardMask,
	},
	RedeemVoucher(u8),
	OpenBooster(u8),
	Reroll,
	EndShop,
	CashOut,
	PickPackCard {
		index: u8,
		targets: CardMask,
	},
	SkipPack,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActionError {
	WrongPhase,
	BadIndex,
	BadSelection,
	CannotAfford,
	NoSpace,
	NotAllowed,
}

impl fmt::Display for ActionError {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str(match self {
			ActionError::WrongPhase => "not possible in the current phase",
			ActionError::BadIndex => "no card at that index",
			ActionError::BadSelection => "invalid card selection",
			ActionError::CannotAfford => "not enough money",
			ActionError::NoSpace => "no room for the card",
			ActionError::NotAllowed => "not allowed now",
		})
	}
}
impl std::error::Error for ActionError {}

/// The hand positions `m` selects, in ascending order.
#[inline]
pub(crate) fn mask_bits(m: CardMask) -> impl Iterator<Item = usize> {
	(0..CardMask::BITS as usize).filter(move |i| m >> i & 1 == 1)
}

#[inline]
pub(crate) fn mask_indices(m: CardMask) -> Vec<usize> {
	mask_bits(m).collect()
}

/// Whether `m` only selects cards of a hand of `n` cards.
#[inline]
pub(crate) fn mask_fits(m: CardMask, n: usize) -> bool {
	m.checked_shr(num::index32(n)).unwrap_or(0) == 0
}

#[inline]
fn spendable(g: &Game) -> f64 {
	g.dollars - g.bankrupt_at
}

/// Hand cards that must be part of any selection (Cerulean Bell).
pub(crate) fn forced_mask(g: &Game) -> CardMask {
	let mut m = 0;
	for (i, c) in g.hand.iter().take(CardMask::BITS as usize).enumerate() {
		if c.ab.forced_selection {
			m |= 1 << i;
		}
	}
	m
}

fn valid_selection(g: &Game, m: CardMask, max: u32) -> bool {
	let n = m.count_ones();
	let forced = forced_mask(g);
	n >= 1 && n <= max && mask_fits(m, g.hand.len()) && (m & forced) == forced
}

fn consumable_usable(g: &Game, c: &crate::card::Card, area: Area, targets: CardMask) -> bool {
	if !mask_fits(targets, g.hand.len()) {
		return false;
	}
	// Only consumables that use targets must include the forced card.
	let uses_targets = uses_targets(c.kind);
	let forced = forced_mask(g);
	if uses_targets && targets & forced != forced {
		return false;
	}
	let n = targets.count_ones() as usize;
	let first_editionless =
		targets != 0 && g.hand[targets.trailing_zeros() as usize].edition == Edition::None;
	if !consumables::can_use_consumable(g, c, Some(area), n, first_editionless) {
		return false;
	}
	// Ankh refuses to resolve with a full joker area (`Card:check_use`).
	if c.kind == Kind::Spectral(Spectral::Ankh) && num::float(g.jokers.len()) >= g.joker_limit.limit
	{
		return false;
	}
	// Only send targets to consumables that use them (extra highlighted
	// cards are harmless in the game; the action space stays canonical).
	if !uses_targets && targets != 0 {
		return false;
	}
	true
}

/// See [`Game::check`].
#[expect(
	clippy::too_many_lines,
	reason = "one arm per action, each a short list of the game's own checks"
)]
pub(crate) fn check(g: &Game, a: Action) -> Result<(), ActionError> {
	use Action as A;
	let ph = g.phase;
	let need = |p: Phase| {
		if ph == p {
			Ok(())
		} else {
			Err(ActionError::WrongPhase)
		}
	};
	let not_over = || {
		if ph == Phase::GameOver {
			Err(ActionError::WrongPhase)
		} else {
			Ok(())
		}
	};
	match a {
		A::SelectBlind => need(Phase::BlindSelect),
		A::SkipBlind => {
			need(Phase::BlindSelect)?;
			match g.blind_on_deck {
				Some(BlindSlot::Small | BlindSlot::Big) => Ok(()),
				_ => Err(ActionError::NotAllowed),
			}
		}
		A::RerollBoss => {
			need(Phase::BlindSelect)?;
			let ok = g.voucher_used(Voucher::Retcon)
				|| (g.voucher_used(Voucher::DirectorsCut) && !g.boss_rerolled);
			if !ok {
				return Err(ActionError::NotAllowed);
			}
			if spendable(g) - 10.0 < 0.0 {
				return Err(ActionError::CannotAfford);
			}
			Ok(())
		}
		A::Play(m) => {
			need(Phase::Round)?;
			if !valid_selection(g, m, 5) {
				return Err(ActionError::BadSelection);
			}
			Ok(())
		}
		A::Discard(m) => {
			need(Phase::Round)?;
			if g.discards_left <= 0.0 {
				return Err(ActionError::NotAllowed);
			}
			if !valid_selection(g, m, 5) {
				return Err(ActionError::BadSelection);
			}
			Ok(())
		}
		A::UseConsumable { index, targets } => {
			not_over()?;
			let c = g
				.consumables
				.get(index as usize)
				.ok_or(ActionError::BadIndex)?;
			if !consumable_usable(g, c, Area::Consumables, targets) {
				return Err(ActionError::NotAllowed);
			}
			Ok(())
		}
		A::SellJoker(i) => {
			not_over()?;
			let c = g.jokers.get(i as usize).ok_or(ActionError::BadIndex)?;
			if c.ab.eternal {
				return Err(ActionError::NotAllowed);
			}
			Ok(())
		}
		A::SellConsumable(i) => {
			not_over()?;
			let c = g.consumables.get(i as usize).ok_or(ActionError::BadIndex)?;
			if c.ab.eternal {
				return Err(ActionError::NotAllowed);
			}
			Ok(())
		}
		A::MoveJoker { from, to } => {
			not_over()?;
			if from as usize >= g.jokers.len() || to as usize >= g.jokers.len() || from == to {
				return Err(ActionError::BadIndex);
			}
			Ok(())
		}
		A::MoveHand { from, to } => {
			if !matches!(ph, Phase::Round | Phase::Pack) {
				return Err(ActionError::WrongPhase);
			}
			if from as usize >= g.hand.len() || to as usize >= g.hand.len() || from == to {
				return Err(ActionError::BadIndex);
			}
			Ok(())
		}
		A::SortHand { .. } => {
			if !matches!(ph, Phase::Round | Phase::Pack) || g.hand.is_empty() {
				return Err(ActionError::WrongPhase);
			}
			Ok(())
		}
		A::BuyShopCard(i) => {
			need(Phase::Shop)?;
			let c = g.shop_jokers.get(i as usize).ok_or(ActionError::BadIndex)?;
			if c.cost > spendable(g) && c.cost > 0.0 {
				return Err(ActionError::CannotAfford);
			}
			if !shop::has_buy_space(g, c) {
				return Err(ActionError::NoSpace);
			}
			Ok(())
		}
		A::BuyAndUse { index, targets } => {
			need(Phase::Shop)?;
			let c = g
				.shop_jokers
				.get(index as usize)
				.ok_or(ActionError::BadIndex)?;
			if !c.kind.is_consumable() {
				return Err(ActionError::NotAllowed);
			}
			if c.cost > spendable(g) && c.cost > 0.0 {
				return Err(ActionError::CannotAfford);
			}
			if !consumable_usable(g, c, Area::ShopJokers, targets) {
				return Err(ActionError::NotAllowed);
			}
			Ok(())
		}
		A::RedeemVoucher(i) => {
			need(Phase::Shop)?;
			let c = g
				.shop_vouchers
				.get(i as usize)
				.ok_or(ActionError::BadIndex)?;
			if c.cost > spendable(g) {
				return Err(ActionError::CannotAfford);
			}
			Ok(())
		}
		A::OpenBooster(i) => {
			need(Phase::Shop)?;
			let c = g
				.shop_boosters
				.get(i as usize)
				.ok_or(ActionError::BadIndex)?;
			if c.cost > 0.0 && c.cost > spendable(g) {
				return Err(ActionError::CannotAfford);
			}
			Ok(())
		}
		A::Reroll => {
			need(Phase::Shop)?;
			if spendable(g) - g.reroll_cost < 0.0 && g.reroll_cost != 0.0 {
				return Err(ActionError::CannotAfford);
			}
			Ok(())
		}
		A::EndShop => need(Phase::Shop),
		A::CashOut => need(Phase::RoundEval),
		A::PickPackCard { index, targets } => {
			need(Phase::Pack)?;
			let c = g
				.pack_cards
				.get(index as usize)
				.ok_or(ActionError::BadIndex)?;
			match c.kind {
				k if k.is_consumable() => {
					if !consumable_usable(g, c, Area::PackCards, targets) {
						return Err(ActionError::NotAllowed);
					}
				}
				Kind::Joker(_) => {
					if targets != 0 {
						return Err(ActionError::BadSelection);
					}
					if c.edition != Edition::Negative
						&& num::float(g.jokers.len()) >= g.joker_limit.limit
					{
						return Err(ActionError::NoSpace);
					}
				}
				_ => {
					if targets != 0 {
						return Err(ActionError::BadSelection);
					}
				}
			}
			Ok(())
		}
		A::SkipPack => {
			need(Phase::Pack)?;
			let p = g.pack.as_ref().ok_or(ActionError::WrongPhase)?;
			let ok = !g.pack_cards.is_empty()
				&& (matches!(
					p.kind,
					PackKind::Celestial | PackKind::Standard | PackKind::Buffoon
				) || !g.hand.is_empty()
					|| g.hand_limit.limit <= 0.0);
			if ok {
				Ok(())
			} else {
				Err(ActionError::NotAllowed)
			}
		}
	}
}

/// See [`Game::step`].
pub(crate) fn step(g: &mut Game, a: Action) -> Result<(), ActionError> {
	use Action as A;

	g.check(&a)?;
	events::begin_action(g);
	let reorder = matches!(
		a,
		A::MoveJoker { .. } | A::MoveHand { .. } | A::SortHand { .. }
	);
	g.rearranges = if reorder {
		g.rearranges.saturating_add(1)
	} else {
		0
	};
	match a {
		A::SelectBlind => flow::do_select_blind(g),
		A::SkipBlind => flow::do_skip_blind(g),
		A::RerollBoss => flow::do_reroll_boss(g, false),
		A::Play(m) => flow::do_play(g, m),
		A::Discard(m) => flow::do_discard(g, mask_indices(m), false),
		A::UseConsumable { index, targets } => {
			let c = g.consumables.remove(index as usize);
			shop::use_card(g, c, Some(Area::Consumables), &mask_indices(targets));
		}
		A::SellJoker(i) => shop::do_sell(g, Area::Jokers, i as usize),
		A::SellConsumable(i) => shop::do_sell(g, Area::Consumables, i as usize),
		A::MoveJoker { from, to } => {
			let c = g.jokers.remove(from as usize);
			g.jokers.insert(to as usize, c);
		}
		A::MoveHand { from, to } => {
			let c = g.hand.remove(from as usize);
			g.hand.insert(to as usize, c);
		}
		A::SortHand { by_suit } => {
			g.hand_sort_by_suit = by_suit;
			flow::sort_hand(g);
		}
		A::BuyShopCard(i) => shop::do_buy(g, i as usize, false, &[]),
		A::BuyAndUse { index, targets } => {
			shop::do_buy(g, index as usize, true, &mask_indices(targets));
		}
		A::RedeemVoucher(i) => {
			let c = g.shop_vouchers.remove(i as usize);
			shop::use_card(g, c, Some(Area::ShopVouchers), &[]);
		}
		A::OpenBooster(i) => {
			let c = g.shop_boosters.remove(i as usize);
			shop::use_card(g, c, Some(Area::ShopBoosters), &[]);
		}
		A::Reroll => shop::do_reroll(g),
		A::EndShop => shop::do_end_shop(g),
		A::CashOut => round_end::do_cash_out(g),
		A::PickPackCard { index, targets } => {
			let c = g.pack_cards.remove(index as usize);
			shop::use_card(g, c, Some(Area::PackCards), &mask_indices(targets));
		}
		A::SkipPack => packs::do_skip_pack(g),
	}
	events::run_queue(g);
	Ok(())
}

/// See [`Game::legal_actions`].
pub(crate) fn legal_actions(g: &Game, out: &mut Vec<Action>) {
	out.clear();
	let push = |out: &mut Vec<Action>, g: &Game, a: Action| {
		if g.check(&a).is_ok() {
			out.push(a);
		}
	};
	let nh = g.hand.len().min(CardMask::BITS as usize);
	match g.phase {
		Phase::GameOver => return,
		Phase::BlindSelect => {
			push(out, g, Action::SelectBlind);
			push(out, g, Action::SkipBlind);
			push(out, g, Action::RerollBoss);
		}
		Phase::Round => {
			// Same order and rules as `check`, without re-validating.
			let forced = forced_mask(g);
			let can_discard = g.discards_left > 0.0;
			for_each_subset(nh, 5, |m| {
				if m & forced == forced {
					out.push(Action::Play(m));
					if can_discard {
						out.push(Action::Discard(m));
					}
				}
			});
		}
		Phase::Shop => {
			for i in 0..g.shop_jokers.len() {
				push(out, g, Action::BuyShopCard(num::byte(i)));
				if g.shop_jokers[i].kind.is_consumable() {
					push_targeted(g, out, &g.shop_jokers[i], |t| Action::BuyAndUse {
						index: num::byte(i),
						targets: t,
					});
				}
			}
			for i in 0..g.shop_vouchers.len() {
				push(out, g, Action::RedeemVoucher(num::byte(i)));
			}
			for i in 0..g.shop_boosters.len() {
				push(out, g, Action::OpenBooster(num::byte(i)));
			}
			push(out, g, Action::Reroll);
			push(out, g, Action::EndShop);
		}
		Phase::Pack => {
			for i in 0..g.pack_cards.len() {
				let c = &g.pack_cards[i];
				if c.kind.is_consumable() {
					push_targeted(g, out, c, |t| Action::PickPackCard {
						index: num::byte(i),
						targets: t,
					});
				} else {
					push(out, g, Action::PickPackCard {
						index: num::byte(i),
						targets: 0,
					});
				}
			}
			push(out, g, Action::SkipPack);
		}
		Phase::RoundEval => push(out, g, Action::CashOut),
	}
	for i in 0..g.consumables.len() {
		let c = &g.consumables[i];
		push_targeted(g, out, c, |t| Action::UseConsumable {
			index: num::byte(i),
			targets: t,
		});
	}
	for i in 0..g.jokers.len() {
		push(out, g, Action::SellJoker(num::byte(i)));
	}
	for i in 0..g.consumables.len() {
		push(out, g, Action::SellConsumable(num::byte(i)));
	}
}

fn push_targeted(
	g: &Game, out: &mut Vec<Action>, c: &crate::card::Card, mk: impl Fn(CardMask) -> Action,
) {
	let (max_h, _) = consumable_highlight(c.kind);
	if !uses_targets(c.kind) {
		let a = mk(0);
		if g.check(&a).is_ok() {
			out.push(a);
		}
		return;
	}
	let max = if max_h == 0 {
		1
	} else {
		u32::from(max_h.min(5))
	};
	let nh = g.hand.len().min(CardMask::BITS as usize);
	for_each_subset(nh, max, |m| {
		let a = mk(m);
		if g.check(&a).is_ok() {
			out.push(a);
		}
	});
}

impl fmt::Display for Action {
	/// The text code used in histories and by the bindings, e.g. `play 11`
	/// (card masks in decimal); parsed back by [`FromStr`].
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		use Action as A;
		match *self {
			A::SelectBlind => f.write_str("select"),
			A::SkipBlind => f.write_str("skip"),
			A::RerollBoss => f.write_str("reroll_boss"),
			A::Play(m) => write!(f, "play {m}"),
			A::Discard(m) => write!(f, "discard {m}"),
			A::UseConsumable { index, targets } => write!(f, "use {index} {targets}"),
			A::SellJoker(i) => write!(f, "sell_joker {i}"),
			A::SellConsumable(i) => write!(f, "sell_consumable {i}"),
			A::MoveJoker { from, to } => write!(f, "move_joker {from} {to}"),
			A::MoveHand { from, to } => write!(f, "move_hand {from} {to}"),
			A::SortHand { by_suit } => write!(f, "sort {}", u8::from(by_suit)),
			A::BuyShopCard(i) => write!(f, "buy {i}"),
			A::BuyAndUse { index, targets } => write!(f, "buy_use {index} {targets}"),
			A::RedeemVoucher(i) => write!(f, "redeem {i}"),
			A::OpenBooster(i) => write!(f, "open {i}"),
			A::Reroll => f.write_str("reroll"),
			A::EndShop => f.write_str("end_shop"),
			A::CashOut => f.write_str("cash_out"),
			A::PickPackCard { index, targets } => write!(f, "pick {index} {targets}"),
			A::SkipPack => f.write_str("skip_pack"),
		}
	}
}

/// A malformed action or injection code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError(pub String);

impl fmt::Display for ParseError {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "bad code {:?}", self.0)
	}
}
impl std::error::Error for ParseError {}

/// The whitespace-separated operands of a text code, parsed strictly (the
/// right count, each in range).
pub(crate) struct Operands<'a> {
	it: std::str::SplitWhitespace<'a>,
	code: &'a str,
}

impl<'a> Operands<'a> {
	pub(crate) fn new(code: &'a str) -> (Option<&'a str>, Operands<'a>) {
		let mut it = code.split_whitespace();
		(it.next(), Operands { it, code })
	}

	pub(crate) fn err(&self) -> ParseError {
		ParseError(self.code.to_string())
	}

	pub(crate) fn next<T: FromStr>(&mut self) -> Result<T, ParseError> {
		self.it
			.next()
			.and_then(|t| t.parse().ok())
			.ok_or_else(|| self.err())
	}

	/// Fails if operands are left over.
	pub(crate) fn end<T>(mut self, v: T) -> Result<T, ParseError> {
		if self.it.next().is_some() {
			Err(self.err())
		} else {
			Ok(v)
		}
	}
}

impl FromStr for Action {
	type Err = ParseError;

	fn from_str(s: &str) -> Result<Action, ParseError> {
		use Action as A;
		let (op, mut n) = Operands::new(s);
		let a = match op.ok_or_else(|| n.err())? {
			"select" => A::SelectBlind,
			"skip" => A::SkipBlind,
			"reroll_boss" => A::RerollBoss,
			"play" => A::Play(n.next()?),
			"discard" => A::Discard(n.next()?),
			"use" => A::UseConsumable {
				index: n.next()?,
				targets: n.next()?,
			},
			"sell_joker" => A::SellJoker(n.next()?),
			"sell_consumable" => A::SellConsumable(n.next()?),
			"move_joker" => A::MoveJoker {
				from: n.next()?,
				to: n.next()?,
			},
			"move_hand" => A::MoveHand {
				from: n.next()?,
				to: n.next()?,
			},
			"sort" => A::SortHand {
				by_suit: match n.next::<u8>()? {
					0 => false,
					1 => true,
					_ => return Err(n.err()),
				},
			},
			"buy" => A::BuyShopCard(n.next()?),
			"buy_use" => A::BuyAndUse {
				index: n.next()?,
				targets: n.next()?,
			},
			"redeem" => A::RedeemVoucher(n.next()?),
			"open" => A::OpenBooster(n.next()?),
			"reroll" => A::Reroll,
			"end_shop" => A::EndShop,
			"cash_out" => A::CashOut,
			"pick" => A::PickPackCard {
				index: n.next()?,
				targets: n.next()?,
			},
			"skip_pack" => A::SkipPack,
			_ => return Err(n.err()),
		};
		n.end(a)
	}
}

/// Calls `f` for every non-empty subset of `n` items with at most `max`
/// members, in increasing mask order.
#[inline]
pub(crate) fn for_each_subset(n: usize, max: u32, mut f: impl FnMut(CardMask)) {
	if n == 0 {
		return;
	}
	let end: u64 = 1u64 << n;
	let mut m: u64 = 1;
	while m < end {
		#[expect(
			clippy::cast_possible_truncation,
			reason = "`m < 1 << n`, and callers cap `n` at the mask's 32 bits"
		)]
		let mask = m as CardMask;
		if mask.count_ones() > max {
			// Skip ahead to the next mask with fewer bits set.
			m += m.isolate_lowest_one();
			continue;
		}
		f(mask);
		m += 1;
	}
}
