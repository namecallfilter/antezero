//! Cost of the build previews on states of real-looking games:
//! `cargo run --release --example preview_bench [games]`.
//!
//! Walks seeded games with a cheap policy (plays the best candidate in
//! rounds, otherwise a seeded random legal action), collects decision
//! states by phase, and times on them: a round decision's candidate
//! previews (the reference), the baseline build preview, and the item
//! features of the legal ops.

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::hint::black_box;
use std::time::Instant;

use sim::data::Back;
use sim::game::{Game, Phase};
use sim::obs::{self, ObsMut};
use sim::{Action, num};

#[expect(clippy::too_many_lines, reason = "a flat script of timed sections")]
#[expect(clippy::float_cmp, reason = "the previewed flag is exactly 1")]
fn main() {
	let games: usize = std::env::args()
		.nth(1)
		.and_then(|s| s.parse().ok())
		.unwrap_or(40);
	let mut states: Vec<Game> = Vec::new();
	let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
	let mut acts = Vec::new();
	let mut steps_by_phase = [0usize; 6];
	for s in 0..games {
		let mut g = Game::new_run(&format!("BENCH{s}"), Back::Red, 1);
		g.allow_skip = false;
		for _ in 0..1500 {
			if g.is_terminal() {
				break;
			}
			steps_by_phase[g.phase as usize] += 1;
			if states.len() < 20_000 {
				states.push(g.clone());
			}
			x ^= x << 13;
			x ^= x >> 7;
			x ^= x << 17;
			let a = match g.phase {
				Phase::Round => {
					let top = obs::top_plays(&g, 1);
					match top.first() {
						Some(c) if !x.is_multiple_of(7) || g.discards_left <= 0.0 => {
							Action::Play(c.mask)
						}
						_ => {
							g.legal_actions(&mut acts);
							acts[(x >> 8) as usize % acts.len()]
						}
					}
				}
				Phase::BlindSelect => Action::SelectBlind,
				Phase::RoundEval => Action::CashOut,
				// Build up: buy what is affordable (jokers first), open a pack now and then, else leave.
				Phase::Shop => {
					g.legal_actions(&mut acts);
					let buy = acts.iter().find(|a| matches!(a, Action::BuyShopCard(i) if matches!(g.shop_jokers[*i as usize].kind, sim::Kind::Joker(_))));
					let use_ = acts
						.iter()
						.find(|a| matches!(a, Action::BuyAndUse { .. } | Action::RedeemVoucher(_)));
					let open = acts.iter().find(|a| matches!(a, Action::OpenBooster(_)));
					match (buy, use_, open) {
						(Some(a), ..) => *a,
						(_, Some(a), _) if x.is_multiple_of(2) => *a,
						(_, _, Some(a)) if x.is_multiple_of(3) => *a,
						_ => Action::EndShop,
					}
				}
				Phase::Pack => {
					g.legal_actions(&mut acts);
					let pick = acts.iter().find(|a| matches!(a, Action::PickPackCard { index, .. } if matches!(g.pack_cards[*index as usize].kind, sim::Kind::Joker(_) | sim::Kind::Planet(_))));
					match pick {
						Some(a) => *a,
						None => acts
							.iter()
							.find(|a| matches!(a, Action::SkipPack))
							.copied()
							.unwrap_or(acts[(x >> 8) as usize % acts.len()]),
					}
				}
				Phase::GameOver => break,
			};
			g.step(a).expect("a legal action steps");
		}
	}
	println!(
		"{} states; steps by phase [blind_select, round, shop, pack, round_eval, over] = {:?}",
		states.len(),
		steps_by_phase
	);
	let max_ante = states.iter().map(|g| g.ante).max().unwrap_or(0);
	println!("max ante reached {max_ante}");
	let mut buf = (
		vec![0i32; obs::MAX_TOKENS * obs::TOK_INT],
		vec![0f32; obs::MAX_TOKENS * obs::TOK_FLOAT],
		vec![0i32; obs::GLOB_INT],
		vec![0f32; obs::GLOB_FLOAT],
	);
	let mut ops = Vec::new();
	for (name, phase) in [
		("round", Phase::Round),
		("shop", Phase::Shop),
		("pack", Phase::Pack),
		("blind_select", Phase::BlindSelect),
	] {
		let sel: Vec<&Game> = states
			.iter()
			.filter(|g| g.phase == phase)
			.take(400)
			.collect();
		if sel.is_empty() {
			continue;
		}
		// Full observation (encode + ops), as the env does it, no cache.
		let t = Instant::now();
		let mut n_item_ops = 0usize;
		for g in &sel {
			let mut g = (*g).clone();
			g.update_derived();
			let mut pv = Vec::new();
			obs::build_preview(&g, &mut pv);
			let lay = obs::encode_with(
				&g,
				&mut ObsMut {
					tok_i: &mut buf.0,
					tok_f: &mut buf.1,
					glob_i: &mut buf.2,
					glob_f: &mut buf.3,
				},
				false,
				pv.first_mut(),
			);
			obs::ops(&g, &lay, 8, &mut ops);
			n_item_ops += (0..lay.n_tokens)
				.filter(|&k| buf.1[k * obs::TOK_FLOAT + obs::TOK_FLOAT - 1] == 1.0)
				.count();
			black_box(&ops);
		}
		let full = t.elapsed().as_secs_f64() / num::float(sel.len());
		// Baseline preview only.
		let t = Instant::now();
		for g in &sel {
			let mut pv = Vec::new();
			obs::build_preview(g, &mut pv);
			black_box(pv);
		}
		let base = t.elapsed().as_secs_f64() / num::float(sel.len());
		// Observation without any preview (the old cost).
		let t = Instant::now();
		for g in &sel {
			let mut g = (*g).clone();
			g.update_derived();
			let lay = obs::encode_with(
				&g,
				&mut ObsMut {
					tok_i: &mut buf.0,
					tok_f: &mut buf.1,
					glob_i: &mut buf.2,
					glob_f: &mut buf.3,
				},
				false,
				None,
			);
			obs::ops(
				&g,
				&lay,
				if phase == Phase::Round { 8 } else { 0 },
				&mut ops,
			);
			black_box(&ops);
		}
		let old = t.elapsed().as_secs_f64() / num::float(sel.len());
		println!(
			"{name:>12}: {} states, old obs {:7.1} us, baseline preview {:7.1} us, full obs with previews {:7.1} us ({:.1} previewed items/state)",
			sel.len(),
			old * 1e6,
			base * 1e6,
			full * 1e6,
			num::float(n_item_ops) / num::float(sel.len())
		);
	}
	// Split: virtual round setup vs best-play search, on shop states.
	let shops: Vec<&Game> = states
		.iter()
		.filter(|g| g.phase == Phase::Shop)
		.take(300)
		.collect();
	if !shops.is_empty() {
		let t = Instant::now();
		let worlds: Vec<Game> = shops
			.iter()
			.map(|g| obs::virtual_round(g, g.boss_choice, 1))
			.collect();
		let vr = t.elapsed().as_secs_f64() / num::float(shops.len());
		let t = Instant::now();
		for v in &worlds {
			black_box(obs::best_play(v));
		}
		let bp = t.elapsed().as_secs_f64() / num::float(shops.len());
		let jokers = num::float(shops.iter().map(|g| g.jokers.len()).sum::<usize>())
			/ num::float(shops.len());
		let antes = shops.iter().map(|g| f64::from(g.ante)).sum::<f64>() / num::float(shops.len());
		println!(
			"shop states: mean ante {antes:.1}, mean jokers {jokers:.1}; virtual_round {:.1} us, best_play {:.1} us (PREVIEW_PLAYS={})",
			vr * 1e6,
			bp * 1e6,
			obs::PREVIEW_PLAYS
		);
	}
	// One preview_play, for scale.
	let rounds: Vec<&Game> = states
		.iter()
		.filter(|g| g.phase == Phase::Round)
		.take(200)
		.collect();
	let alt = sim::rng::PseudoRandom::new("X");
	let t = Instant::now();
	let mut n = 0;
	for g in &rounds {
		for m in [1u32, 3, 7, 15, 31] {
			if obs::preview_score(g, m, &alt).is_some() {
				n += 1;
			}
		}
	}
	println!(
		"preview_play: {:.1} us each ({n} plays)",
		t.elapsed().as_secs_f64() * 1e6 / f64::from(n.max(1))
	);
}
