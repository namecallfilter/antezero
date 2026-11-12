//! Throughput benchmark: `cargo run --release --example bench [seconds]`.
//!
//! * random: uniformly random legal actions (short games).
//! * greedy: plays the best-scoring hand found by stepping clones (the
//!   lookahead pattern MCTS uses), buys/skips randomly elsewhere.
//! * micro: clone, `legal_actions`, and step costs on mid-game states.

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::hint::black_box;
use std::time::Instant;

use sim::data::Back;
use sim::game::{Game, Phase};
use sim::{Action, num};

struct Xs(u64);
impl Xs {
	fn next(&mut self) -> u64 {
		let mut x = self.0;
		x ^= x << 13;
		x ^= x >> 7;
		x ^= x << 17;
		self.0 = x;
		x
	}

	fn below(&mut self, n: usize) -> usize {
		num::pick(self.next(), n)
	}
}

fn random_game(seed: u64, buf: &mut Vec<Action>) -> usize {
	let mut g = Game::new_run(&format!("R{seed}"), Back::Red, 1);
	let mut x = Xs((seed * 2_654_435_761) | 1);
	let mut n = 0;
	while !g.is_terminal() && n < 2000 {
		g.legal_actions(buf);
		let a = buf[x.below(buf.len())];
		g.step(a).expect("a legal action steps");
		n += 1;
	}
	n
}

/// Greedy-ish policy; returns (steps, plays evaluated by lookahead, final ante).
fn greedy_game(seed: u64, buf: &mut Vec<Action>) -> (usize, usize, i32) {
	let mut g = Game::new_run(&format!("G{seed}"), Back::Red, 1);
	let mut x = Xs((seed * 2_654_435_761) | 1);
	let (mut n, mut looked) = (0, 0);
	while !g.is_terminal() && n < 3000 {
		g.legal_actions(buf);
		let a = match g.phase {
			Phase::Round => {
				let mut best = None;
				let mut best_s = -1.0;
				for &a in buf.iter().filter(|a| matches!(a, Action::Play(_))) {
					let mut c = g.clone();
					c.step(a).expect("a legal play steps");
					looked += 1;
					if c.last_hand_score > best_s {
						best_s = c.last_hand_score;
						best = Some(a);
					}
				}
				best.expect("a round always has a legal play")
			}
			Phase::Shop => {
				let buys: Vec<Action> = buf
					.iter()
					.copied()
					.filter(|a| matches!(a, Action::BuyShopCard(_)))
					.collect();
				if !buys.is_empty() && x.below(2) == 0 {
					buys[x.below(buys.len())]
				} else {
					Action::EndShop
				}
			}
			Phase::BlindSelect => Action::SelectBlind,
			Phase::RoundEval => Action::CashOut,
			_ => buf[x.below(buf.len())],
		};
		g.step(a).expect("a legal action steps");
		n += 1;
	}
	(n, looked, g.ante)
}

#[expect(
	clippy::too_many_lines,
	clippy::many_single_char_names,
	reason = "a flat script of timed sections"
)]
#[expect(clippy::cast_precision_loss, reason = "averages for display")]
fn main() {
	let secs: f64 = std::env::args()
		.nth(1)
		.and_then(|s| s.parse().ok())
		.unwrap_or(3.0);
	let mut buf = Vec::with_capacity(512);

	let t = Instant::now();
	let (mut games, mut steps) = (0u64, 0usize);
	while t.elapsed().as_secs_f64() < secs {
		steps += random_game(games, &mut buf);
		games += 1;
	}
	let dt = t.elapsed().as_secs_f64();
	println!(
		"random : {games} games, {steps} steps, {:.0} steps/s, {:.1} us/step (incl. new_run)",
		num::float(steps) / dt,
		dt * 1e6 / num::float(steps)
	);

	let t = Instant::now();
	let (mut games, mut steps, mut looked, mut antes) = (0u64, 0usize, 0usize, 0i64);
	while t.elapsed().as_secs_f64() < secs {
		let (s, l, a) = greedy_game(games, &mut buf);
		steps += s;
		looked += l;
		antes += i64::from(a);
		games += 1;
	}
	let dt = t.elapsed().as_secs_f64();
	println!(
		"greedy : {games} games (avg ante {:.2}), {steps} steps + {looked} lookahead plays; {:.1} us per lookahead (clone+play)",
		antes as f64 / games as f64,
		dt * 1e6 / num::float(looked + steps)
	);

	// Micro: the first round, hand drawn.
	let mut g = Game::new_run("MICRO", Back::Red, 1);
	g.step(Action::SelectBlind)
		.expect("a new run starts at blind select");
	let n = 200_000;
	let t = Instant::now();
	for _ in 0..n {
		black_box(g.clone());
	}
	println!(
		"clone  : {:.0} ns",
		t.elapsed().as_nanos() as f64 / f64::from(n)
	);
	let t = Instant::now();
	for _ in 0..n {
		g.legal_actions(&mut buf);
		black_box(&buf);
	}
	println!(
		"legal  : {:.0} ns ({} actions)",
		t.elapsed().as_nanos() as f64 / f64::from(n),
		buf.len()
	);
	g.legal_actions(&mut buf);
	let play = *buf
		.iter()
		.find(|a| matches!(a, Action::Play(_)))
		.expect("the first hand can be played");
	let disc = *buf
		.iter()
		.find(|a| matches!(a, Action::Discard(_)))
		.expect("the first hand can be discarded");
	let m = 50_000;
	let t = Instant::now();
	for _ in 0..m {
		let mut c = g.clone();
		c.step(black_box(play)).expect("the play is legal");
		black_box(&c);
	}
	println!(
		"play   : {:.2} us (clone + one hand)",
		t.elapsed().as_secs_f64() * 1e6 / f64::from(m)
	);
	let t = Instant::now();
	for _ in 0..m {
		let mut c = g.clone();
		c.step(black_box(disc)).expect("the discard is legal");
		black_box(&c);
	}
	println!(
		"discard: {:.2} us (clone + one discard)",
		t.elapsed().as_secs_f64() * 1e6 / f64::from(m)
	);
	// Scored play candidates (the best PREVIEW_BUDGET plays by a cheap
	// estimate are previewed; chance-dependent ones over several scenarios).
	let t = Instant::now();
	let reps = 200;
	let mut tot = 0;
	for _ in 0..reps {
		tot += black_box(sim::obs::top_plays(&g, 8)).len();
	}
	println!(
		"top8   : {:.1} us per decision ({} legal plays, no jokers)",
		t.elapsed().as_secs_f64() * 1e6 / f64::from(reps),
		buf.iter().filter(|a| matches!(a, Action::Play(_))).count()
	);
	let mut gj = g.clone();
	for j in [
		sim::data::Joker::Misprint,
		sim::data::Joker::Bloodstone,
		sim::data::Joker::Blueprint,
		sim::data::Joker::Hiker,
		sim::data::Joker::Photograph,
	] {
		gj.inject(&sim::inject::Inject::Joker {
			joker: j,
			edition: sim::data::Edition::None,
			eternal: false,
		});
	}
	let t = Instant::now();
	for _ in 0..reps {
		tot += black_box(sim::obs::top_plays(&gj, 8)).len();
	}
	println!(
		"top8   : {:.1} us per decision (5 jokers incl. Misprint: every play is chance-dependent)",
		t.elapsed().as_secs_f64() * 1e6 / f64::from(reps)
	);
	black_box(tot);
	let alt = sim::rng::PseudoRandom::new("X");
	{
		let plays: Vec<_> = buf
			.iter()
			.filter_map(|a| {
				if let Action::Play(m) = *a {
					Some(m)
				} else {
					None
				}
			})
			.collect();
		let flags = g.eval_flags();
		let t = Instant::now();
		let mut k = 0usize;
		for _ in 0..reps {
			for &m in &plays {
				let cards: Vec<&sim::card::Card> = (0..g.hand.len())
					.filter(|&i| m >> i & 1 == 1)
					.map(|i| &g.hand[i])
					.collect();
				k += black_box(sim::hand_eval::evaluate(&cards, &flags)).top_mask as usize;
			}
		}
		println!(
			"eval   : {:.1} us for all {} plays",
			t.elapsed().as_secs_f64() * 1e6 / f64::from(reps),
			plays.len()
		);
		let t = Instant::now();
		for _ in 0..reps {
			for &m in plays.iter().take(64) {
				k += usize::from(black_box(sim::obs::preview_score(&g, m, &alt)).is_some());
			}
		}
		println!(
			"preview: {:.2} us each",
			t.elapsed().as_secs_f64() * 1e6 / f64::from(reps) / 64.0
		);
		black_box(k);
	}
	for (name, gg) in [("no jokers", &g), ("5 jokers", &gj)] {
		let plays: Vec<_> = buf
			.iter()
			.filter_map(|a| {
				if let Action::Play(m) = *a {
					Some(m)
				} else {
					None
				}
			})
			.collect();
		let chance = plays
			.iter()
			.filter(|&&m| sim::obs::preview_score(gg, m, &alt).is_some_and(|p| p.chance))
			.count();
		println!(
			"chance : {chance}/{} plays draw randomness ({name})",
			plays.len()
		);
	}
	let t = Instant::now();
	for i in 0..20_000u64 {
		black_box(Game::new_run(&format!("N{i}"), Back::Red, 1));
	}
	println!(
		"new_run: {:.2} us",
		t.elapsed().as_secs_f64() * 1e6 / 20_000.0
	);
}
