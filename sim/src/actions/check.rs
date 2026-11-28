//! Action legality checks and legal-action enumeration.

use super::parse::{Action, ActionError};
use crate::card::{AnyCard, Card, CardLike, CardMask, Kind, PlayingCard};
use crate::data::{Edition, PackKind, Spectral, Voucher, VoucherSet};
use crate::game::{Area, Areas, BlindSlot, Flow, Game, Hand, Phase, State};
use crate::run::{History, Run};
use crate::shop::{OpenPack, Shop};
use crate::{consumables, flow, num, shop};

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

/// Hand cards that must be part of any selection: the ones Cerulean Bell
/// holds highlighted.
#[must_use]
pub fn forced_mask(hand: &[PlayingCard]) -> CardMask {
	let cards = hand.iter().take(CardMask::CAPACITY).enumerate();
	cards
		.filter(|(_, c)| c.forced.is_locked())
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
