//! The agent-facing action interface.

use std::error::Error;
use std::fmt;
use std::str::{FromStr, SplitWhitespace};

use crate::card::{AnyCard, Card, CardLike, Kind};
use crate::card_mask::CardMask;
use crate::data::{Edition, PackKind, Spectral, Voucher};
use crate::event::Event;
use crate::game::{Area, Areas, BlindSlot, Flow, Game, Hand, HandSort, Phase, State};
use crate::packs::OpenPack;
use crate::playing_card::PlayingCard;
use crate::run::{History, Run};
use crate::shop::Shop;
use crate::{VoucherSet, consumables, event_queue, flow, num, packs, round_end, shop};

/// A decision of the player's, what [`Game::step`] applies. It displays as
/// the text code that histories and the bindings use, e.g. `play 11` (card
/// masks in decimal), which [`FromStr`] parses back.
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
	SortHand(HandSort),
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

/// Why an action is not legal now.
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
impl Error for ActionError {}

#[inline]
fn spendable(run: &Run) -> f64 {
	run.dollars - run.bankrupt_at
}

/// Whether the run can pay `cost` (a free card always can).
fn can_afford(run: &Run, cost: f64) -> bool {
	cost <= 0.0 || cost <= spendable(run)
}

/// The `i`-th of `cards`.
fn card_at<C>(cards: &[C], i: u8) -> Result<&C, ActionError> {
	cards.get(usize::from(i)).ok_or(ActionError::BadIndex)
}

/// Hand cards that must be part of any selection (Cerulean Bell).
#[must_use]
pub fn forced_mask(hand: &[PlayingCard]) -> CardMask {
	let cards = hand.iter().take(CardMask::CAPACITY).enumerate();
	cards
		.filter(|(_, c)| c.forced_selection)
		.map(|(i, _)| i)
		.collect()
}

/// Whether `m` is a selection to play or discard from a hand of `n_hand`
/// cards whose forced cards are `forced`.
fn valid_selection(m: CardMask, forced: CardMask, n_hand: usize) -> bool {
	(1..=Hand::HIGHLIGHT_LIMIT).contains(&m.len()) && m.fits(n_hand) && m.contains_all(forced)
}

fn can_discard(discards_left: f64) -> bool {
	discards_left > 0.0
}

fn consumable_usable(
	state: State, areas: &Areas, history: &History, kind: Kind, area: Area, targets: CardMask,
) -> bool {
	if !targets.fits(areas.hand.cards.len()) {
		return false;
	}
	// Only consumables that use targets must include the forced card.
	let uses_targets = consumables::targets(kind).is_some();
	if uses_targets && !targets.contains_all(forced_mask(&areas.hand.cards)) {
		return false;
	}
	if !consumables::can_use_consumable(state, areas, history, kind, area, targets) {
		return false;
	}
	// Ankh refuses to resolve with a full joker area (`Card:check_use`).
	if kind == Kind::Spectral(Spectral::Ankh) && !areas.joker_room() {
		return false;
	}
	// Only send targets to consumables that use them (extra highlighted
	// cards are harmless in the game; the action space stays canonical).
	if !uses_targets && !targets.is_empty() {
		return false;
	}
	true
}

fn in_phase(phase: Phase, wanted: Phase) -> Result<(), ActionError> {
	if phase == wanted {
		Ok(())
	} else {
		Err(ActionError::WrongPhase)
	}
}

fn not_over(phase: Phase) -> Result<(), ActionError> {
	if phase == Phase::GameOver {
		Err(ActionError::WrongPhase)
	} else {
		Ok(())
	}
}

fn check_skip_blind(phase: Phase, on_deck: Option<BlindSlot>) -> Result<(), ActionError> {
	in_phase(phase, Phase::BlindSelect)?;
	match on_deck {
		Some(BlindSlot::Small | BlindSlot::Big) => Ok(()),
		_ => Err(ActionError::NotAllowed),
	}
}

fn check_reroll_boss(
	phase: Phase, run: &Run, used_vouchers: VoucherSet, boss_rerolled: bool,
) -> Result<(), ActionError> {
	in_phase(phase, Phase::BlindSelect)?;
	let ok = used_vouchers.contains(Voucher::Retcon)
		|| (used_vouchers.contains(Voucher::DirectorsCut) && !boss_rerolled);
	if !ok {
		return Err(ActionError::NotAllowed);
	}
	if spendable(run) - flow::BOSS_REROLL_COST < 0.0 {
		return Err(ActionError::CannotAfford);
	}
	Ok(())
}

fn check_play(phase: Phase, hand: &[PlayingCard], m: CardMask) -> Result<(), ActionError> {
	in_phase(phase, Phase::Round)?;
	if !valid_selection(m, forced_mask(hand), hand.len()) {
		return Err(ActionError::BadSelection);
	}
	Ok(())
}

fn check_discard(
	phase: Phase, hand: &[PlayingCard], discards_left: f64, m: CardMask,
) -> Result<(), ActionError> {
	in_phase(phase, Phase::Round)?;
	if !can_discard(discards_left) {
		return Err(ActionError::NotAllowed);
	}
	if !valid_selection(m, forced_mask(hand), hand.len()) {
		return Err(ActionError::BadSelection);
	}
	Ok(())
}

fn check_use_consumable(
	flow: Flow, areas: &Areas, history: &History, index: u8, targets: CardMask,
) -> Result<(), ActionError> {
	not_over(flow.phase)?;
	let c = card_at(&areas.consumables, index)?;
	if !consumable_usable(
		flow.state,
		areas,
		history,
		c.kind,
		Area::Consumables,
		targets,
	) {
		return Err(ActionError::NotAllowed);
	}
	Ok(())
}

/// Selling the `i`-th card of `owned` (the jokers or the consumables).
fn check_sell(phase: Phase, owned: &[Card], i: u8) -> Result<(), ActionError> {
	not_over(phase)?;
	let c = card_at(owned, i)?;
	if c.ab.eternal() {
		return Err(ActionError::NotAllowed);
	}
	Ok(())
}

/// Moving a card of `cards` from one position to another.
fn check_move<C>(cards: &[C], from: u8, to: u8) -> Result<(), ActionError> {
	if usize::from(from) >= cards.len() || usize::from(to) >= cards.len() || from == to {
		return Err(ActionError::BadIndex);
	}
	Ok(())
}

fn check_buy(phase: Phase, areas: &Areas, run: &Run, i: u8) -> Result<(), ActionError> {
	in_phase(phase, Phase::Shop)?;
	let c = card_at(&areas.shop_jokers, i)?;
	if !can_afford(run, c.core().cost) {
		return Err(ActionError::CannotAfford);
	}
	if !shop::has_buy_space(areas, c) {
		return Err(ActionError::NoSpace);
	}
	Ok(())
}

fn check_buy_and_use(
	flow: Flow, areas: &Areas, run: &Run, history: &History, index: u8, targets: CardMask,
) -> Result<(), ActionError> {
	in_phase(flow.phase, Phase::Shop)?;
	let c = card_at(&areas.shop_jokers, index)?;
	let Some(c) = c.as_card().filter(|c| c.kind.is_consumable()) else {
		return Err(ActionError::NotAllowed);
	};
	if !can_afford(run, c.core.cost) {
		return Err(ActionError::CannotAfford);
	}
	if !consumable_usable(
		flow.state,
		areas,
		history,
		c.kind,
		Area::ShopJokers,
		targets,
	) {
		return Err(ActionError::NotAllowed);
	}
	Ok(())
}

fn check_redeem(phase: Phase, areas: &Areas, run: &Run, i: u8) -> Result<(), ActionError> {
	in_phase(phase, Phase::Shop)?;
	let c = card_at(&areas.shop_vouchers, i)?;
	if c.core.cost > spendable(run) {
		return Err(ActionError::CannotAfford);
	}
	Ok(())
}

fn check_open_booster(phase: Phase, areas: &Areas, run: &Run, i: u8) -> Result<(), ActionError> {
	in_phase(phase, Phase::Shop)?;
	let c = card_at(&areas.shop_boosters, i)?;
	if !can_afford(run, c.core.cost) {
		return Err(ActionError::CannotAfford);
	}
	Ok(())
}

fn check_reroll(phase: Phase, run: &Run, shop: &Shop) -> Result<(), ActionError> {
	in_phase(phase, Phase::Shop)?;
	if spendable(run) - shop.reroll_cost < 0.0 && shop.reroll_cost != 0.0 {
		return Err(ActionError::CannotAfford);
	}
	Ok(())
}

fn check_pick_pack_card(
	flow: Flow, areas: &Areas, history: &History, index: u8, targets: CardMask,
) -> Result<(), ActionError> {
	in_phase(flow.phase, Phase::Pack)?;
	match card_at(&areas.pack_cards, index)? {
		AnyCard::Card(c) if c.kind.is_consumable() => {
			if !consumable_usable(flow.state, areas, history, c.kind, Area::PackCards, targets) {
				return Err(ActionError::NotAllowed);
			}
		}
		AnyCard::Card(Card {
			kind: Kind::Joker(_),
			core,
			..
		}) => {
			if !targets.is_empty() {
				return Err(ActionError::BadSelection);
			}
			if core.edition != Some(Edition::Negative) && !areas.joker_room() {
				return Err(ActionError::NoSpace);
			}
		}
		AnyCard::Card(_) | AnyCard::Playing(_) => {
			if !targets.is_empty() {
				return Err(ActionError::BadSelection);
			}
		}
	}
	Ok(())
}

fn check_skip_pack(phase: Phase, areas: &Areas, open: Option<OpenPack>) -> Result<(), ActionError> {
	in_phase(phase, Phase::Pack)?;
	let p = open.as_ref().ok_or(ActionError::WrongPhase)?;
	let ok = !areas.pack_cards.is_empty()
		&& (matches!(
			p.kind,
			PackKind::Celestial | PackKind::Standard | PackKind::Buffoon
		) || !areas.hand.cards.is_empty()
			|| areas.hand.limit.limit <= 0.0);
	if ok {
		Ok(())
	} else {
		Err(ActionError::NotAllowed)
	}
}

/// See [`Game::check`].
pub(crate) fn check(g: &Game, a: Action) -> Result<(), ActionError> {
	use Action as A;
	match a {
		A::SelectBlind => in_phase(g.flow.phase, Phase::BlindSelect),
		A::SkipBlind => check_skip_blind(g.flow.phase, g.blind_select.on_deck),
		A::RerollBoss => check_reroll_boss(
			g.flow.phase,
			&g.run,
			g.history.used_vouchers,
			g.blind_select.boss_rerolled,
		),
		A::Play(m) => check_play(g.flow.phase, &g.areas.hand.cards, m),
		A::Discard(m) => check_discard(g.flow.phase, &g.areas.hand.cards, g.round.discards_left, m),
		A::UseConsumable { index, targets } => {
			check_use_consumable(g.flow, &g.areas, &g.history, index, targets)
		}
		A::SellJoker(i) => check_sell(g.flow.phase, &g.areas.jokers, i),
		A::SellConsumable(i) => check_sell(g.flow.phase, &g.areas.consumables, i),
		A::MoveJoker { from, to } => {
			not_over(g.flow.phase)?;
			check_move(&g.areas.jokers, from, to)
		}
		A::MoveHand { from, to } => {
			if !matches!(g.flow.phase, Phase::Round | Phase::Pack) {
				return Err(ActionError::WrongPhase);
			}
			check_move(&g.areas.hand.cards, from, to)
		}
		A::SortHand(_) => {
			if !matches!(g.flow.phase, Phase::Round | Phase::Pack) || g.areas.hand.cards.is_empty()
			{
				return Err(ActionError::WrongPhase);
			}
			Ok(())
		}
		A::BuyShopCard(i) => check_buy(g.flow.phase, &g.areas, &g.run, i),
		A::BuyAndUse { index, targets } => {
			check_buy_and_use(g.flow, &g.areas, &g.run, &g.history, index, targets)
		}
		A::RedeemVoucher(i) => check_redeem(g.flow.phase, &g.areas, &g.run, i),
		A::OpenBooster(i) => check_open_booster(g.flow.phase, &g.areas, &g.run, i),
		A::Reroll => check_reroll(g.flow.phase, &g.run, &g.shop),
		A::EndShop => in_phase(g.flow.phase, Phase::Shop),
		A::CashOut => in_phase(g.flow.phase, Phase::RoundEval),
		A::PickPackCard { index, targets } => {
			check_pick_pack_card(g.flow, &g.areas, &g.history, index, targets)
		}
		A::SkipPack => check_skip_pack(g.flow.phase, &g.areas, g.pack.open),
	}
}

/// See [`Game::step`].
pub(crate) fn step(g: &mut Game, a: Action) -> Result<(), ActionError> {
	use Action as A;

	g.check(a)?;
	event_queue::begin_action(&mut g.rng, &mut g.events);
	let reorder = matches!(a, A::MoveJoker { .. } | A::MoveHand { .. } | A::SortHand(_));
	g.flow.rearranges = if reorder {
		g.flow.rearranges.saturating_add(1)
	} else {
		0
	};
	match a {
		A::SelectBlind => flow::do_select_blind(&g.blind_select, &mut g.round, &mut g.events),
		A::SkipBlind => {
			flow::do_skip_blind(&mut g.blind_select, &mut g.run, &mut g.tags, &mut g.events);
		}
		A::RerollBoss => {
			g.events.push(Event::EaseDollars(-flow::BOSS_REROLL_COST));
			flow::do_reroll_boss(&mut g.blind_select, &mut g.events);
		}
		A::Play(m) => flow::do_play(
			&mut g.flow,
			&mut g.areas,
			&mut g.blind,
			&mut g.events,
			m,
			g.setup.steamodded,
		),
		A::Discard(m) => {
			let mut sel = [0; Hand::HIGHLIGHT_LIMIT];
			for (slot, i) in sel.iter_mut().zip(m.indices()) {
				*slot = i;
			}
			flow::do_discard(g, &mut sel[..m.len()], /* hook= */ false);
		}
		A::UseConsumable { index, targets } => use_from(g, Area::Consumables, index, targets),
		A::SellJoker(i) => shop::do_sell(g, Area::Jokers, usize::from(i)),
		A::SellConsumable(i) => shop::do_sell(g, Area::Consumables, usize::from(i)),
		A::MoveJoker { from, to } => {
			let c = g.areas.jokers.remove(usize::from(from));
			g.areas.jokers.insert(usize::from(to), c);
		}
		A::MoveHand { from, to } => {
			let c = g.areas.hand.cards.remove(usize::from(from));
			g.areas.hand.cards.insert(usize::from(to), c);
		}
		A::SortHand(order) => {
			g.areas.hand.order = order;
			g.areas.hand.sort();
		}
		A::BuyShopCard(i) => {
			shop::do_buy(&g.areas.shop_jokers, &mut g.events, usize::from(i), None);
		}
		A::BuyAndUse { index, targets } => {
			shop::do_buy(
				&g.areas.shop_jokers,
				&mut g.events,
				usize::from(index),
				Some(targets),
			);
		}
		A::RedeemVoucher(i) => use_from(g, Area::ShopVouchers, i, CardMask::EMPTY),
		A::OpenBooster(i) => use_from(g, Area::ShopBoosters, i, CardMask::EMPTY),
		A::Reroll => shop::do_reroll(g.shop.reroll_cost, &mut g.events),
		A::EndShop => shop::do_end_shop(g),
		A::CashOut => round_end::do_cash_out(g),
		A::PickPackCard { index, targets } => use_from(g, Area::PackCards, index, targets),
		A::SkipPack => packs::do_skip_pack(g),
	}
	event_queue::run_queue(g);
	Ok(())
}

/// See [`Game::preview_play`]: [`step`] for `Action::Play(mask)` up to the
/// scoring.
pub(crate) fn preview_play(g: &mut Game, mask: CardMask) -> Result<f64, ActionError> {
	g.check(Action::Play(mask))?;
	event_queue::begin_action(&mut g.rng, &mut g.events);
	flow::do_play(
		&mut g.flow,
		&mut g.areas,
		&mut g.blind,
		&mut g.events,
		mask,
		g.setup.steamodded,
	);
	event_queue::run_until_scored(g);
	Ok(g.run.last_hand_score)
}

/// See [`Game::preview_hand`]: [`step`] for `Action::Play(mask)` up to the
/// end of the hand's own chain.
pub(crate) fn preview_hand(g: &mut Game, mask: CardMask) -> Result<f64, ActionError> {
	g.check(Action::Play(mask))?;
	event_queue::begin_action(&mut g.rng, &mut g.events);
	flow::do_play(
		&mut g.flow,
		&mut g.areas,
		&mut g.blind,
		&mut g.events,
		mask,
		g.setup.steamodded,
	);
	event_queue::run_until_played(g);
	Ok(g.run.last_hand_score)
}

/// Uses the `i`-th card of `area` with the hand cards `targets`.
fn use_from(g: &mut Game, area: Area, i: u8, targets: CardMask) {
	let c = g.areas.remove(area, usize::from(i));
	shop::use_card(g, c, Some(area), targets);
}

fn push_if_legal(g: &Game, out: &mut Vec<Action>, a: Action) {
	if g.check(a).is_ok() {
		out.push(a);
	}
}

/// See [`Game::legal_actions`].
pub(crate) fn legal_actions(g: &Game, out: &mut Vec<Action>) {
	out.clear();
	let nh = g.areas.hand.cards.len().min(CardMask::CAPACITY);
	match g.flow.phase {
		Phase::GameOver => return,
		Phase::BlindSelect => {
			push_if_legal(g, out, Action::SelectBlind);
			push_if_legal(g, out, Action::SkipBlind);
			push_if_legal(g, out, Action::RerollBoss);
		}
		Phase::Round => {
			let hand = &g.areas.hand.cards;
			let forced = forced_mask(hand);
			let can_discard = can_discard(g.round.discards_left);
			// Every subset has 1..=HIGHLIGHT_LIMIT cards of the hand, so only the
			// forced cards are left to test.
			CardMask::for_each_subset(nh, Hand::HIGHLIGHT_LIMIT, |m| {
				if m.contains_all(forced) {
					out.push(Action::Play(m));
					if can_discard {
						out.push(Action::Discard(m));
					}
				}
			});
		}
		Phase::Shop => {
			for (i, c) in g.areas.shop_jokers.iter().enumerate() {
				push_if_legal(g, out, Action::BuyShopCard(num::byte(i)));
				if let AnyCard::Card(c) = c
					&& c.kind.is_consumable()
				{
					push_targeted(g, out, c, |t| Action::BuyAndUse {
						index: num::byte(i),
						targets: t,
					});
				}
			}
			for i in 0..g.areas.shop_vouchers.len() {
				push_if_legal(g, out, Action::RedeemVoucher(num::byte(i)));
			}
			for i in 0..g.areas.shop_boosters.len() {
				push_if_legal(g, out, Action::OpenBooster(num::byte(i)));
			}
			push_if_legal(g, out, Action::Reroll);
			push_if_legal(g, out, Action::EndShop);
		}
		Phase::Pack => {
			for (i, c) in g.areas.pack_cards.iter().enumerate() {
				if let AnyCard::Card(c) = c
					&& c.kind.is_consumable()
				{
					push_targeted(g, out, c, |t| Action::PickPackCard {
						index: num::byte(i),
						targets: t,
					});
				} else {
					push_if_legal(g, out, Action::PickPackCard {
						index: num::byte(i),
						targets: CardMask::EMPTY,
					});
				}
			}
			push_if_legal(g, out, Action::SkipPack);
		}
		Phase::RoundEval => push_if_legal(g, out, Action::CashOut),
	}
	for (i, c) in g.areas.consumables.iter().enumerate() {
		push_targeted(g, out, c, |t| Action::UseConsumable {
			index: num::byte(i),
			targets: t,
		});
	}
	for i in 0..g.areas.jokers.len() {
		push_if_legal(g, out, Action::SellJoker(num::byte(i)));
	}
	for i in 0..g.areas.consumables.len() {
		push_if_legal(g, out, Action::SellConsumable(num::byte(i)));
	}
}

/// The legal uses of the consumable `c`, with every target selection it
/// takes.
fn push_targeted(g: &Game, out: &mut Vec<Action>, c: &Card, mk: impl Fn(CardMask) -> Action) {
	let Some(targets) = consumables::targets(c.kind) else {
		push_if_legal(g, out, mk(CardMask::EMPTY));
		return;
	};
	let nh = g.areas.hand.cards.len().min(CardMask::CAPACITY);
	CardMask::for_each_subset(nh, targets.max, |m| push_if_legal(g, out, mk(m)));
}

impl fmt::Display for Action {
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
			A::SortHand(order) => write!(f, "sort {}", u8::from(order == HandSort::Suit)),
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
impl Error for ParseError {}

/// The whitespace-separated operands of a text code, parsed strictly (the
/// right count, each in range).
pub(crate) struct Operands<'a> {
	it: SplitWhitespace<'a>,
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
			"sort" => A::SortHand(match n.next::<u8>()? {
				0 => HandSort::Rank,
				1 => HandSort::Suit,
				_ => return Err(n.err()),
			}),
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
