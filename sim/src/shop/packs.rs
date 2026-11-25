//! Booster packs: opening (`Card:open`), contents, picking and skipping
//! (`G.FUNCS.skip_booster` / `end_consumeable`).

use std::mem;

use super::state::Shop;
use crate::areas::Pile;
use crate::card::{AnyCard, Kind, PlayingCard, Pricing};
use crate::data::{Booster, HandType, PackKind, Planet, Seal, Voucher};
use crate::event::{Blocks, Draw, EditionTiming, Event, EventQueue, GAMESPEED, Trigger};
use crate::game::{Area, Areas, CardSet, Flow, Game, State};
use crate::joker::LifecycleCtx;
use crate::pools::{EditionPool, NewCard, PlayingSet};
use crate::rng::{Key, PseudoRandom};
use crate::run::Run;
use crate::{card, flow, joker, pools, run};

/// Booster packs: the open one, and `G.GAME.pack_size` / `pack_choices`,
/// which keep the values of the last pack opened.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PackState {
	pub open: Option<OpenPack>,
	pub size: u8,
	/// The picks left.
	pub choices: Option<u8>,
}

impl Default for PackState {
	fn default() -> PackState {
		PackState {
			open: None,
			size: 2,
			choices: None,
		}
	}
}

/// An open booster pack.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OpenPack {
	pub kind: PackKind,
	/// The state the pack was opened in, to return to.
	pub interrupted: State,
}

/// The events of an open booster pack.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum PackEvent {
	/// The booster `card` exploding open.
	Open {
		card: u32,
	},
	Create,
	StateUi,
	DrawHand,
	/// `G.FUNCS.end_consumeable`'s event.
	End {
		delay_factor: f64,
	},
	EndInner,
	/// The pack UI being removed (with the unchosen pack cards).
	Remove,
}

pub(crate) fn run_event(g: &mut Game, e: PackEvent) {
	match e {
		PackEvent::Open { card } => open_pack_run(g, card),
		PackEvent::Create => pack_create(g),
		PackEvent::StateUi => g.events.push(Event::Pack(PackEvent::DrawHand)),
		PackEvent::DrawHand => match g.flow.state {
			State::TarotPack | State::SpectralPack => {
				flow::draw_from_deck_to_hand(
					&mut g.flow,
					&g.areas,
					&g.round,
					&g.blind,
					&mut g.events,
					None,
				);
				g.events.delay(0.5);
			}
			State::StandardPack | State::BuffoonPack => g.events.delay(0.5),
			_ => {}
		},
		PackEvent::End { delay_factor } => {
			for _ in 0..g.areas.hand.cards.len() {
				g.events.draw(Draw::next(Pile::Hand, Pile::Deck), 0.08);
			}
			g.events
				.push_after(0.2 * delay_factor, Event::Pack(PackEvent::EndInner));
			run::apply_new_blind_choice(&mut g.tags, &mut g.events);
			// The game's event that calls `save_run`, which only takes time here.
			g.events.delay(0.2 * delay_factor);
		}
		PackEvent::EndInner => {
			let pack = g
				.pack
				.open
				.take()
				.expect("only an open pack queues its own end");
			g.flow.state = pack.interrupted;
			g.events.push(Event::Wait);
		}
		PackEvent::Remove => {
			let cards = mem::take(&mut g.areas.pack_cards);
			for c in cards {
				card::unmark_used(&g.areas, &mut g.history, &c);
			}
		}
	}
}

/// `Card:open()` for the booster `b` with card id `id`, bought at `cost`
/// in the state `interrupted`.
pub(super) fn open_booster(
	flow: &mut Flow, pack: &mut PackState, events: &mut EventQueue, b: Booster, id: u32, cost: f64,
	interrupted: State,
) {
	let d = b.def();
	flow.state_complete = false;
	flow.state = match d.kind {
		PackKind::Arcana => State::TarotPack,
		PackKind::Celestial => State::PlanetPack,
		PackKind::Spectral => State::SpectralPack,
		PackKind::Standard => State::StandardPack,
		PackKind::Buffoon => State::BuffoonPack,
	};
	pack.size = d.extra;
	pack.choices = Some(d.choose);
	pack.open = Some(OpenPack {
		kind: d.kind,
		interrupted,
	});
	if cost > 0.0 {
		events.delay(0.2);
		events.push(Event::EaseDollars(-cost));
	} else {
		events.delay(0.2);
	}
	events.push_after(0.4, Event::Pack(PackEvent::Open { card: id }));
}

fn open_pack_run(g: &mut Game, booster_id: u32) {
	g.events.explode(booster_id);
	// Contents are created by a timed non-blocking, non-blockable event.
	let t = 1.3 * GAMESPEED.sqrt();
	g.events.push_event(
		Event::Pack(PackEvent::Create),
		Trigger::After(t),
		Blocks::Detached,
	);
	joker::calc_jokers(g, LifecycleCtx::OpenBooster);
}

fn pack_create(g: &mut Game) {
	let pack = g
		.pack
		.open
		.expect("pack contents are created for an open pack");
	for i in 0..g.pack.size {
		let c: AnyCard = match pack.kind {
			PackKind::Arcana => {
				let omen = g.history.used_vouchers.contains(Voucher::OmenGlobe) && {
					let k = Key::from("omen_globe");
					g.rng.random(&k) > 0.8
				};
				let spec = if omen {
					NewCard::soulable(CardSet::Spectral, Area::PackCards, "ar2")
				} else {
					NewCard::soulable(CardSet::Tarot, Area::PackCards, "ar1")
				};
				pools::create_card(g, spec).into()
			}
			PackKind::Celestial => {
				let mut forced = None;
				if g.history.used_vouchers.contains(Voucher::Telescope) && i == 0 {
					let mut best: Option<HandType> = None;
					let mut tally = 0.0;
					for h in HandType::ALL {
						let hl = &g.hands[h];
						if hl.visible && hl.played > tally {
							best = Some(h);
							tally = hl.played;
						}
					}
					forced = best.map(|h| Kind::Planet(Planet::for_hand(h)));
				}
				let spec = match forced {
					Some(kind) => NewCard::forced(kind, Area::PackCards, "pl1"),
					None => NewCard::soulable(CardSet::Planet, Area::PackCards, "pl1"),
				};
				pools::create_card(g, spec).into()
			}
			PackKind::Spectral => {
				let spec = NewCard::soulable(CardSet::Spectral, Area::PackCards, "spe");
				pools::create_card(g, spec).into()
			}
			PackKind::Standard => {
				standard_card(&mut g.areas, &g.run, &g.shop, &mut g.rng, &mut g.events).into()
			}
			PackKind::Buffoon => {
				let spec = NewCard::soulable(CardSet::Joker, Area::PackCards, "buf");
				pools::create_card(g, spec).into()
			}
		};
		g.events.materialize(1.5 * GAMESPEED);
		g.areas.pack_cards.push(c);
	}
}

/// A Standard pack's card: a playing card with a polled edition and,
/// sometimes, a seal.
fn standard_card(
	areas: &mut Areas, run: &Run, shop: &Shop, rng: &mut PseudoRandom, events: &mut EventQueue,
) -> PlayingCard {
	let k = run.key_ante("stdset");
	let set = if rng.random(&k) > 0.6 {
		PlayingSet::Enhanced
	} else {
		PlayingSet::Base
	};
	let mut c = pools::create_playing_card(areas, run, shop.discount_percent, rng, set, "sta");
	let k = run.key_ante("standard_edition");
	let edition = pools::poll_edition(rng, shop.edition_rate, &k, 2.0, EditionPool::NoNegative);
	card::set_edition(
		&mut c,
		edition,
		None,
		Pricing::new(shop.discount_percent, &areas.jokers),
		&mut areas.limits,
	);
	events.edition_fx(edition, EditionTiming::Delayed);
	let k = run.key_ante("stdseal");
	if rng.random(&k) > 1.0 - 0.02 * 10.0 {
		let k = run.key_ante("stdsealtype");
		let roll = rng.random(&k);
		c.seal = Some(if roll > 0.75 {
			Seal::Red
		} else if roll > 0.5 {
			Seal::Blue
		} else if roll > 0.25 {
			Seal::Gold
		} else {
			Seal::Purple
		});
		Pricing::new(shop.discount_percent, &areas.jokers).set_cost(&mut c, None);
	}
	c
}

pub(crate) fn update_pack(flow: &mut Flow, events: &mut EventQueue) {
	if !flow.state_complete {
		flow.state_complete = true;
		events.push(Event::Pack(PackEvent::StateUi));
	}
}

/// `G.FUNCS.skip_booster`.
pub(crate) fn do_skip_pack(g: &mut Game) {
	joker::calc_jokers(g, LifecycleCtx::SkippingBooster);
	end_consumeable(&mut g.events, 1.0);
}

/// `G.FUNCS.end_consumeable(e, delayfac)`.
pub(super) fn end_consumeable(events: &mut EventQueue, delay_factor: f64) {
	// The pack UI (and with it the unchosen cards) is removed by a
	// timed non-blocking event.
	events.push_event(
		Event::Pack(PackEvent::Remove),
		Trigger::After(0.2 * delay_factor),
		Blocks::Detached,
	);
	events.delay(0.2 * delay_factor);
	events.push_after(
		0.2 * delay_factor,
		Event::Pack(PackEvent::End { delay_factor }),
	);
}
