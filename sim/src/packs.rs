//! Booster packs: opening (`Card:open`), contents, picking and skipping
//! (`G.FUNCS.skip_booster` / `end_consumeable`).

use crate::card::{Card, Kind};
use crate::data::{Booster, HandType, PackKind, Planet, Seal, Voucher};
use crate::ev::Ev;
use crate::events::St;
use crate::game::{Area, CardSet, Game, PackState};
use crate::jokers::Ctx;
use crate::rng::Key;
use crate::{cards, jokers, pools, shop};

// -----------------------------------------------------------------
// Booster packs
// -----------------------------------------------------------------

/// `Card:open()`.
pub(crate) fn open_booster(g: &mut Game, card: Card) {
	let Kind::Booster(b) = card.kind else { return };
	let d = b.def();
	g.state_complete = false;
	g.state = match d.kind {
		PackKind::Arcana => St::TarotPack,
		PackKind::Celestial => St::PlanetPack,
		PackKind::Spectral => St::SpectralPack,
		PackKind::Standard => St::StandardPack,
		PackKind::Buffoon => St::BuffoonPack,
	};
	g.pack_size = d.extra;
	g.pack_choices = d.choose;
	g.pack = Some(PackState { kind: d.kind });
	if card.cost > 0.0 {
		g.events.push_after(0.2, Ev::Wait);
		g.events.push(Ev::EaseDollars(-card.cost));
	} else {
		g.events.delay(0.2);
	}
	let id = card.id;
	g.consumed.push(card);
	g.events.push_after(0.4, Ev::OpenPack { card: id });
}

pub(crate) fn open_pack_run(g: &mut Game, booster_id: u32) {
	g.events.explode(booster_id);
	// Contents are created by a timed non-blocking, non-blockable event.
	let t = 1.3 * crate::events::GAMESPEED.sqrt();
	g.events
		.push_ev(Ev::PackCreate, crate::events::Trig::After(t), false, false);
	for j in 0..g.jokers.len() {
		jokers::calc_joker(g, j, &Ctx::OpenBooster, None);
	}
}

#[expect(clippy::too_many_lines, reason = "one arm per pack kind")]
pub(crate) fn pack_create(g: &mut Game) {
	let Some(pack) = g.pack else { return };
	let size = g.pack_size as usize;
	for i in 0..size {
		let c = match pack.kind {
			PackKind::Arcana => {
				let omen = g.voucher_used(Voucher::OmenGlobe) && {
					let k = Key::from("omen_globe");
					g.rng.random(&k) > 0.8
				};
				if omen {
					pools::create_card(
						g,
						CardSet::Spectral,
						Area::PackCards,
						false,
						None,
						true,
						None,
						"ar2",
					)
				} else {
					pools::create_card(
						g,
						CardSet::Tarot,
						Area::PackCards,
						false,
						None,
						true,
						None,
						"ar1",
					)
				}
			}
			PackKind::Celestial => {
				let mut forced = None;
				if g.voucher_used(Voucher::Telescope) && i == 0 {
					let mut best: Option<HandType> = None;
					let mut tally = 0.0;
					for h in HandType::ALL {
						let hl = &g.hands[h as usize];
						if hl.visible && hl.played > tally {
							best = Some(h);
							tally = hl.played;
						}
					}
					forced = best.map(|h| Kind::Planet(Planet::for_hand(h)));
				}
				pools::create_card(
					g,
					CardSet::Planet,
					Area::PackCards,
					false,
					None,
					true,
					forced,
					"pl1",
				)
			}
			PackKind::Spectral => pools::create_card(
				g,
				CardSet::Spectral,
				Area::PackCards,
				false,
				None,
				true,
				None,
				"spe",
			),
			PackKind::Standard => {
				let k = g.key_ante("stdset");
				let set = if g.rng.random(&k) > 0.6 {
					CardSet::Enhanced
				} else {
					CardSet::Base
				};
				let mut c =
					pools::create_card(g, set, Area::PackCards, false, None, true, None, "sta");
				let k = g.key_ante("standard_edition");
				let edition = pools::poll_edition(g, &k, 2.0, true, false);
				cards::set_edition(g, &mut c, edition, None);
				g.events.edition_fx(edition, false);
				let k = g.key_ante("stdseal");
				if g.rng.random(&k) > 1.0 - 0.02 * 10.0 {
					let k = g.key_ante("stdsealtype");
					let roll = g.rng.random(&k);
					c.seal = Some(if roll > 0.75 {
						Seal::Red
					} else if roll > 0.5 {
						Seal::Blue
					} else if roll > 0.25 {
						Seal::Gold
					} else {
						Seal::Purple
					});
					cards::set_cost(g, &mut c, None);
				}
				c
			}
			PackKind::Buffoon => pools::create_card(
				g,
				CardSet::Joker,
				Area::PackCards,
				false,
				None,
				true,
				None,
				"buf",
			),
		};
		g.events.materialize(1.5 * crate::events::GAMESPEED);
		g.pack_cards.push(c);
	}
}

/// Opens a pack granted by a tag (cost 0, `from_tag`).
pub(crate) fn open_tag_pack(g: &mut Game, b: Booster) {
	let mut c = cards::make_card(g, Kind::Booster(b));
	c.cost = 0.0;
	shop::use_card(g, c, None, &[]);
	g.events.materialize(1.0);
}

pub(crate) fn update_pack(g: &mut Game) {
	if !g.state_complete {
		g.state_complete = true;
		g.events.push(Ev::PackStateUi);
	}
}

/// `G.FUNCS.skip_booster`.
pub(crate) fn do_skip_pack(g: &mut Game) {
	for j in 0..g.jokers.len() {
		jokers::calc_joker(g, j, &Ctx::SkippingBooster, None);
	}
	end_consumeable(g, 1.0);
}

/// `G.FUNCS.end_consumeable(e, delayfac)`.
pub(crate) fn end_consumeable(g: &mut Game, df: f64) {
	// The pack UI (and with it the unchosen cards) is removed by a
	// timed non-blocking event.
	g.events.push_ev(
		Ev::PackRemove,
		crate::events::Trig::After(0.2 * df),
		false,
		false,
	);
	g.events.delay(0.2 * df);
	g.events.push_after(0.2 * df, Ev::EndConsumeable { df });
}
