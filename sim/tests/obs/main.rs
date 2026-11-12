//! Tests of the learning interface (`sim::obs`): the shared helpers, with one module of tests per subject.

mod candidates;
mod capacity;
mod encoding;
mod preview_cache;
mod previews;

use sim::data::Back;
use sim::game::{Game, Phase};
use sim::obs::{self, ObsMut};
use sim::{Action, num};

struct Buf {
	ti: Vec<i32>,
	tf: Vec<f32>,
	gi: Vec<i32>,
	gf: Vec<f32>,
}

impl Buf {
	fn new() -> Buf {
		Buf {
			ti: vec![0; obs::MAX_TOKENS * obs::TOK_INT],
			tf: vec![0.0; obs::MAX_TOKENS * obs::TOK_FLOAT],
			gi: vec![0; obs::GLOB_INT],
			gf: vec![0.0; obs::GLOB_FLOAT],
		}
	}

	/// Panics with the first differing token / global if `self != o`.
	fn assert_same(&self, o: &Buf, what: &str) {
		self.first_diff("tok_i", &self.ti, &o.ti, obs::TOK_INT, what);
		self.first_diff("tok_f", &self.tf, &o.tf, obs::TOK_FLOAT, what);
		self.first_diff("glob_i", &self.gi, &o.gi, 1, what);
		self.first_diff("glob_f", &self.gf, &o.gf, 1, what);
	}

	/// Panics at the first index where `a` and `b` differ (`w` features per
	/// row; token rows are shown with their categorical features).
	fn first_diff<T: PartialEq + std::fmt::Debug>(
		&self, name: &str, a: &[T], b: &[T], w: usize, what: &str,
	) {
		if let Some(i) = (0..a.len()).find(|&i| a[i] != b[i]) {
			let tok = i / w;
			let row = if w > 1 {
				format!(
					" token {tok} ints {:?}",
					&self.ti[tok * obs::TOK_INT..(tok + 1) * obs::TOK_INT]
				)
			} else {
				String::new()
			};
			panic!(
				"{what}: {name}[{i}] (feature {}){row}: {:?} != {:?}",
				i % w,
				a[i],
				b[i]
			);
		}
	}

	fn encode(&mut self, g: &Game) -> obs::Layout {
		obs::encode_opts(
			g,
			&mut ObsMut {
				tok_i: &mut self.ti,
				tok_f: &mut self.tf,
				glob_i: &mut self.gi,
				glob_f: &mut self.gf,
			},
			true,
		)
	}
}

/// Plays a game with a simple seeded policy, calling `f` at every decision.
fn walk(seed: &str, steps: usize, mut f: impl FnMut(&Game)) {
	let mut g = Game::new_run(seed, Back::Red, 1);
	let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
	let mut acts = Vec::new();
	for _ in 0..steps {
		if g.is_terminal() {
			break;
		}
		f(&g);
		g.legal_actions(&mut acts);
		x ^= x << 13;
		x ^= x >> 7;
		x ^= x << 17;
		// Prefer progress: play more often than anything else in rounds.
		let plays: Vec<Action> = acts
			.iter()
			.copied()
			.filter(|a| matches!(a, Action::Play(_)))
			.collect();
		let a = if g.phase == Phase::Round && !x.is_multiple_of(4) && !plays.is_empty() {
			plays[(x >> 8) as usize % plays.len()]
		} else {
			acts[(x >> 8) as usize % acts.len()]
		};
		g.step(a).expect("a legal action steps");
	}
}

/// The token at `i` (ints).
fn tok(b: &Buf, i: usize) -> &[i32] {
	&b.ti[i * obs::TOK_INT..(i + 1) * obs::TOK_INT]
}

/// Like `walk`, but plays the best candidate in rounds (so the games
/// reach shops, packs and later antes) and acts at random elsewhere.
fn walk_build(seed: &str, steps: usize, mut f: impl FnMut(&Game)) {
	let mut g = Game::new_run(seed, Back::Red, 1);
	let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
	let mut acts = Vec::new();
	for _ in 0..steps {
		if g.is_terminal() {
			break;
		}
		f(&g);
		x ^= x << 13;
		x ^= x >> 7;
		x ^= x << 17;
		let top = obs::top_plays(&g, 1);
		let a = match top.first() {
			Some(c) if g.phase == Phase::Round && !x.is_multiple_of(7) => Action::Play(c.mask),
			_ => {
				g.legal_actions(&mut acts);
				acts[(x >> 8) as usize % acts.len()]
			}
		};
		g.step(a).expect("a legal action steps");
	}
}

/// Observation and ops of `g` (with 4 candidates and the build preview).
fn obs_and_ops(g: &Game) -> (Buf, Vec<obs::Op>) {
	let mut b = Buf::new();
	let lay = b.encode(g);
	let mut ops = Vec::new();
	obs::ops(g, &lay, 4, &mut ops);
	(b, ops)
}

const G_HAS_PREVIEW: usize = 33;

const G_PREVIEW: usize = 155;

fn hand_of(cards: &[(u8, sim::data::Suit)]) -> Vec<sim::Card> {
	cards
		.iter()
		.enumerate()
		.map(|(i, &(rank, suit))| {
			let mut c = sim::Card::new(100 + num::index32(i), sim::Kind::Playing);
			c.rank = rank;
			c.suit = suit;
			c
		})
		.collect()
}

/// The best score of any legal play (every play simulated).
fn best_legal_play(g: &Game) -> f64 {
	let mut acts = Vec::new();
	g.legal_actions(&mut acts);
	acts.iter()
		.filter_map(|a| {
			if let Action::Play(m) = *a {
				g.clone().preview_play(m).ok()
			} else {
				None
			}
		})
		.fold(0.0, f64::max)
}

fn best_candidate(g: &Game) -> f64 {
	obs::top_plays(g, 8)
		.iter()
		.map(|c| c.mean)
		.fold(0.0, f64::max)
}

/// A game at its first shop (best-candidate play gets there on most seeds).
fn first_shop(prefix: &str, from: u64) -> Game {
	for s in from.. {
		let mut g = Game::new_run(&format!("{prefix}{s}"), Back::Red, 1);
		g.allow_skip = false;
		g.step(Action::SelectBlind)
			.expect("a new run starts at blind select");
		while g.phase == Phase::Round || g.phase == Phase::RoundEval {
			let a = if g.phase == Phase::Round {
				Action::Play(obs::top_plays(&g, 1)[0].mask)
			} else {
				Action::CashOut
			};
			g.step(a)
				.expect("a candidate play and cashing out are legal");
		}
		if g.phase == Phase::Shop {
			return g;
		}
	}
	unreachable!()
}
