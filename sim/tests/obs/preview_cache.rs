//! The build-preview cache gives what a fresh encode gives.

use sim::data::{Back, Edition, Joker, Planet, Voucher};
use sim::game::{Game, Phase};
use sim::inject::Inject;
use sim::obs::{self, ObsMut};
use sim::{Action, Kind, num};

use crate::{Buf, G_PREVIEW, obs_and_ops};

#[test]
#[expect(
	clippy::many_single_char_names,
	reason = "short names for the games and buffers being compared"
)]
fn build_preview_cache_matches_fresh_encode() {
	// The env reuses a cached preview while the joker multiset, money and
	// cards stay the same (joker moves, the blind select after a shop); it
	// must equal a fresh encode, also with order-sensitive jokers, and the
	// blind select must show the capacity the shop showed.
	let (mut after_moves, mut after_shops) = (0, 0);
	for s in 0..30u64 {
		let mut g = Game::new_run(&format!("CM{s}"), Back::Red, 1);
		let mut cache = Vec::new();
		let mut x: u64 = 0x9e37_79b9_7f4a_7c15 ^ s;
		let mut acts = Vec::new();
		let (mut injected, mut moved) = (false, false);
		let mut shop_capacity: Option<Vec<f32>> = None;
		for _ in 0..300 {
			if g.is_terminal() {
				break;
			}
			if g.phase == Phase::Shop && !injected {
				injected = true;
				for (j, e) in [
					(Joker::Joker, Edition::Polychrome),
					(Joker::Blueprint, Edition::None),
					(Joker::Cavendish, Edition::Holo),
					(Joker::Cavendish, Edition::None),
				] {
					g.inject(&Inject::Joker {
						joker: j,
						edition: e,
						eternal: false,
					});
				}
				if s % 2 == 0 {
					g.inject(&Inject::Joker {
						joker: Joker::Bull,
						edition: Edition::None,
						eternal: false,
					});
				}
			}
			let mut a = Buf::new();
			let has = obs::build_preview(&g, &mut cache);
			obs::encode_with(
				&g,
				&mut ObsMut {
					tok_i: &mut a.ti,
					tok_f: &mut a.tf,
					glob_i: &mut a.gi,
					glob_f: &mut a.gf,
				},
				true,
				if has { cache.first_mut() } else { None },
			);
			let (b, _) = obs_and_ops(&g);
			a.assert_same(
				&b,
				&format!("cached preview vs a fresh one in {:?}", g.phase),
			);
			after_moves += usize::from(has && moved);
			if let Some(c) = shop_capacity.take() {
				assert_eq!(
					&a.gf[G_PREVIEW..G_PREVIEW + 6],
					&c[..],
					"the build did not change, its capacity did (shop -> {:?})",
					g.phase
				);
				after_shops += 1;
			}
			x ^= x << 13;
			x ^= x >> 7;
			x ^= x << 17;
			let top = obs::top_plays(&g, 1);
			let act = match top.first() {
				Some(c) if g.phase == Phase::Round && !x.is_multiple_of(7) => Action::Play(c.mask),
				_ if g.phase == Phase::Shop && g.jokers.len() >= 2 && x.is_multiple_of(3) => {
					let n = num::byte(g.jokers.len());
					let [_, from, to, ..] = x.to_le_bytes();
					Action::MoveJoker {
						from: from % n,
						to: to % n,
					}
				}
				_ => {
					g.legal_actions(&mut acts);
					acts[(x >> 8) as usize % acts.len()]
				}
			};
			moved = matches!(act, Action::MoveJoker { .. });
			shop_capacity = (g.phase == Phase::Shop && act == Action::EndShop)
				.then(|| a.gf[G_PREVIEW..G_PREVIEW + 6].to_vec());
			if g.step(act).is_err() {
				moved = false;
				shop_capacity = None;
			}
		}
	}
	assert!(
		after_moves > 30 && after_shops > 20,
		"{after_moves} states after joker moves, {after_shops} after shops"
	);
}

#[test]
#[expect(
	clippy::too_many_lines,
	reason = "one scenario: the walk and its direct check share their setup"
)]
fn build_preview_cache_ignores_money_and_consumables_only_when_unread() {
	// The cache key leaves money and the held consumables out while no
	// preview reads them. Everything that does read them is put in play here
	// (Bull / Bootstraps owned, debuffed, or for sale; Observatory owned or
	// for sale; consumables with a scoring edition) while money and
	// consumables change under a warm cache: it must always equal a fresh one.
	let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
	let mut rnd = |n: usize| {
		x ^= x << 13;
		x ^= x >> 7;
		x ^= x << 17;
		num::pick(x >> 11, n)
	};
	let money_joker =
		|c: &sim::Card| matches!(c.kind, Kind::Joker(Joker::Bull | Joker::Bootstraps));
	let (mut cover, mut acts) = ([0usize; 6], Vec::new());
	for s in 0..40u64 {
		let mut g = Game::new_run(&format!("CK{s}"), Back::Red, 1);
		g.allow_skip = false;
		match s % 4 {
			0 => g.inject(&Inject::Joker {
				joker: Joker::Bull,
				edition: Edition::None,
				eternal: false,
			}),
			1 => g.inject(&Inject::Joker {
				joker: Joker::Bootstraps,
				edition: Edition::None,
				eternal: false,
			}),
			2 => g.inject(&Inject::Voucher(Voucher::Observatory)),
			_ => {}
		}
		if s % 3 == 0 {
			g.inject(&Inject::Joker {
				joker: Joker::Blueprint,
				edition: Edition::None,
				eternal: false,
			});
		}
		g.inject(&Inject::Joker {
			joker: Joker::Jolly,
			edition: Edition::None,
			eternal: false,
		});
		let (mut cache, mut last) = (Vec::new(), g.phase);
		for _ in 0..300 {
			if g.is_terminal() {
				break;
			}
			if g.phase == Phase::Shop && last != Phase::Shop {
				g.inject(&Inject::Dollars(25.0));
				if rnd(5) < 2 {
					let e = [
						Edition::None,
						Edition::Foil,
						Edition::Holo,
						Edition::Polychrome,
					][rnd(4)];
					g.inject(&Inject::Consumable {
						kind: Kind::Planet(
							[
								Planet::Pluto,
								Planet::Mercury,
								Planet::Uranus,
								Planet::Venus,
							][rnd(4)],
						),
						edition: e,
					});
				}
				if rnd(5) < 2
					&& !g.shop_vouchers.is_empty()
					&& !g.voucher_used(Voucher::Observatory)
				{
					g.shop_vouchers[0].kind = Kind::Voucher(Voucher::Observatory);
				}
				if rnd(10) < 3 && !g.shop_jokers.is_empty() {
					let mut d = g.clone();
					d.inject(&Inject::Joker {
						joker: if rnd(2) == 0 {
							Joker::Bull
						} else {
							Joker::Bootstraps
						},
						edition: Edition::None,
						eternal: false,
					});
					g.shop_jokers[0] = d.jokers.pop().unwrap();
				}
				if s % 8 == 0 {
					// The previewed blind clears this debuff: the joker still counts.
					for c in g.jokers.iter_mut().filter(|c| money_joker(c)) {
						c.debuff = true;
					}
				}
			}
			last = g.phase;
			if matches!(g.phase, Phase::Shop | Phase::Pack | Phase::BlindSelect) {
				let mut a = Buf::new();
				let has = obs::build_preview(&g, &mut cache);
				obs::encode_with(
					&g,
					&mut ObsMut {
						tok_i: &mut a.ti,
						tok_f: &mut a.tf,
						glob_i: &mut a.gi,
						glob_f: &mut a.gf,
					},
					true,
					if has { cache.first_mut() } else { None },
				);
				let (b, _) = obs_and_ops(&g);
				a.assert_same(
					&b,
					&format!("cached preview vs a fresh one in {:?} (game {s})", g.phase),
				);
				let hit = [
					g.jokers.iter().any(&money_joker),
					g.jokers.iter().any(|c| money_joker(c) && c.debuff),
					g.shop_jokers.iter().chain(&g.pack_cards).any(&money_joker),
					g.voucher_used(Voucher::Observatory),
					g.shop_vouchers
						.iter()
						.any(|c| c.kind == Kind::Voucher(Voucher::Observatory)),
					g.consumables.iter().any(|c| {
						matches!(
							c.edition,
							Edition::Foil | Edition::Holo | Edition::Polychrome
						)
					}),
				];
				for (k, f) in hit.into_iter().enumerate() {
					cover[k] += usize::from(f);
				}
				if rnd(4) == 0 {
					g.inject(&Inject::Dollars(if rnd(2) == 0 { 3.0 } else { -2.0 }));
				}
				if hit[4] && rnd(3) == 0 {
					// A planet arrives while the Observatory voucher's preview is cached.
					g.inject(&Inject::Consumable {
						kind: Kind::Planet(if rnd(2) == 0 {
							Planet::Pluto
						} else {
							Planet::Mercury
						}),
						edition: Edition::None,
					});
				}
			}
			let act = match obs::top_plays(&g, 1).first() {
				Some(c) if g.phase == Phase::Round => Action::Play(c.mask),
				_ => {
					g.legal_actions(&mut acts);
					let sell = rnd(7) == 0;
					let pool: Vec<Action> = acts
						.iter()
						.copied()
						.filter(|a| {
							!matches!(
								a,
								Action::EndShop
									| Action::MoveJoker { .. }
									| Action::MoveHand { .. }
									| Action::SortHand { .. }
							) && (sell || !matches!(a, Action::SellJoker(_)))
						})
						.collect();
					if pool.is_empty() || (g.phase == Phase::Shop && rnd(8) == 0) {
						acts.iter()
							.copied()
							.find(|a| matches!(a, Action::EndShop))
							.unwrap_or(acts[0])
					} else {
						pool[rnd(pool.len())]
					}
				}
			};
			if g.step(act).is_err() {
				break;
			}
		}
	}
	assert!(
		cover.iter().all(|&n| n > 20),
		"states with a money joker owned / debuffed / for sale, Observatory owned / for sale, an edition consumable: {cover:?}"
	);
	// Directly: a planet bought while the shop's Observatory voucher is previewed changes that preview.
	let mut changed = 0;
	for s in 0..12u64 {
		let mut g = Game::new_run(&format!("CK{s}"), Back::Red, 1);
		g.allow_skip = false;
		g.inject(&Inject::Joker {
			joker: Joker::Jolly,
			edition: Edition::None,
			eternal: false,
		});
		while !g.is_terminal() && (g.phase != Phase::Shop || g.shop_vouchers.is_empty()) {
			let act = match obs::top_plays(&g, 1).first() {
				Some(c) if g.phase == Phase::Round => Action::Play(c.mask),
				_ => {
					g.legal_actions(&mut acts);
					acts[0]
				}
			};
			g.step(act).unwrap();
		}
		if g.is_terminal() {
			continue;
		}
		g.shop_vouchers[0].kind = Kind::Voucher(Voucher::Observatory);
		let mut cache = Vec::new();
		assert!(obs::build_preview(&g, &mut cache));
		let before = cache[0].item_feats(&g, obs::zone::SHOP_VOUCHER, 0);
		for p in [Planet::Pluto, Planet::Mercury] {
			g.inject(&Inject::Consumable {
				kind: Kind::Planet(p),
				edition: Edition::None,
			});
		}
		assert!(obs::build_preview(&g, &mut cache));
		let warm = cache[0].item_feats(&g, obs::zone::SHOP_VOUCHER, 0);
		let mut fresh = Vec::new();
		assert!(obs::build_preview(&g, &mut fresh));
		assert_eq!(
			warm,
			fresh[0].item_feats(&g, obs::zone::SHOP_VOUCHER, 0),
			"stale Observatory preview (game {s})"
		);
		changed += usize::from(warm != before);
	}
	assert!(
		changed >= 3,
		"the held planets changed Observatory's preview in only {changed} games"
	);
}
