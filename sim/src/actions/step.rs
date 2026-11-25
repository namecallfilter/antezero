//! Action execution and play previews.

use super::parse::{Action, ActionError};
use crate::card::CardMask;
use crate::event::{self, Event};
use crate::game::{Area, Game, Hand};
use crate::{flow, round, shop};

/// See [`Game::step`].
pub(crate) fn step(g: &mut Game, a: Action) -> Result<(), ActionError> {
	use Action as A;

	g.check(a)?;
	event::begin_action(&mut g.rng, &mut g.events);
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
		A::CashOut => round::do_cash_out(g),
		A::PickPackCard { index, targets } => use_from(g, Area::PackCards, index, targets),
		A::SkipPack => shop::do_skip_pack(g),
	}
	event::run_queue(g);
	Ok(())
}

/// See [`Game::preview_play`]: [`step`] for `Action::Play(mask)` up to the
/// scoring.
pub(crate) fn preview_play(g: &mut Game, mask: CardMask) -> Result<f64, ActionError> {
	g.check(Action::Play(mask))?;
	event::begin_action(&mut g.rng, &mut g.events);
	flow::do_play(
		&mut g.flow,
		&mut g.areas,
		&mut g.blind,
		&mut g.events,
		mask,
		g.setup.steamodded,
	);
	event::run_until_scored(g);
	Ok(g.run.last_hand_score)
}

/// See [`Game::preview_hand`]: [`step`] for `Action::Play(mask)` up to the
/// end of the hand's own chain.
pub(crate) fn preview_hand(g: &mut Game, mask: CardMask) -> Result<f64, ActionError> {
	g.check(Action::Play(mask))?;
	event::begin_action(&mut g.rng, &mut g.events);
	flow::do_play(
		&mut g.flow,
		&mut g.areas,
		&mut g.blind,
		&mut g.events,
		mask,
		g.setup.steamodded,
	);
	event::run_until_played(g);
	Ok(g.run.last_hand_score)
}

/// Uses the `i`-th card of `area` with the hand cards `targets`.
fn use_from(g: &mut Game, area: Area, i: u8, targets: CardMask) {
	let c = g.areas.remove(area, usize::from(i));
	shop::use_card(g, c, Some(area), targets);
}
