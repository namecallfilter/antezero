//! Golden trajectories: the simulator alone (no game DLL needed) plays a
//! fixed set of seeded games across every deck and stake, with random
//! scenario injections, and every step is fingerprinted: the canonical state
//! (including every RNG stream), the action's timeline (RNG calls, draws,
//! removals, with frame numbers) and the learning observation with its ops.
//! Any behaviour change shows up as a mismatch, so refactors can be checked
//! in seconds; the oracle difftest remains the ground truth for parity.
//!
//! The simulator part (state + timeline, `golden_sim.txt`) and the learning
//! interface part (observation + ops, `golden_obs.txt`) are recorded
//! separately, so an observation change cannot hide a simulator change.
//! Re-record after an intended change with `GOLDEN_BLESS=obs` (observation
//! only; the simulator must still match) or `GOLDEN_BLESS=all`.

use oracle::diff::{Xs, choose, random_inject};
use sim::data::{ALL_BACKS, Back};
use sim::game::{Game, Phase};
use sim::num;
use sim::obs::{self, ObsMut};

const GAMES: u64 = 200;
const STEPS: usize = 300;

struct Fnv(u64);

impl Fnv {
	fn new() -> Fnv {
		Fnv(0xcbf2_9ce4_8422_2325)
	}

	fn bytes(&mut self, b: &[u8]) {
		for &x in b {
			self.0 ^= u64::from(x);
			self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
		}
	}

	fn str(&mut self, s: &str) {
		self.bytes(s.as_bytes());
		self.bytes(&[0xff]);
	}
}

/// Fingerprints of the state after a step: (simulator, learning interface).
fn fingerprint(g: &mut Game, action: &str) -> (u64, u64) {
	let mut h = Fnv::new();
	h.str(action);
	for (k, v) in sim::canon::from_sim(g) {
		h.str(&k);
		h.str(&v);
	}
	for t in g.timeline() {
		h.str(t);
	}
	let sim = h.0;
	let mut h = Fnv::new();
	let mut ti = vec![0i32; obs::MAX_TOKENS * obs::TOK_INT];
	let mut tf = vec![0f32; obs::MAX_TOKENS * obs::TOK_FLOAT];
	let mut gi = vec![0i32; obs::GLOB_INT];
	let mut gf = vec![0f32; obs::GLOB_FLOAT];
	let lay = obs::encode_opts(
		g,
		&mut ObsMut {
			tok_i: &mut ti,
			tok_f: &mut tf,
			glob_i: &mut gi,
			glob_f: &mut gf,
		},
		true,
	);
	let of = obs::Layout::of(g);
	assert!(
		of == obs::Layout {
			n_tokens: of.n_tokens,
			..lay
		} && of.n_tokens <= lay.n_tokens,
		"Layout::of {of:?} vs encoded {lay:?}"
	);
	for x in ti.iter().chain(&gi) {
		h.bytes(&x.to_le_bytes());
	}
	for x in tf.iter().chain(&gf) {
		h.bytes(&x.to_bits().to_le_bytes());
	}
	let mut ops = Vec::new();
	obs::ops(g, &lay, 4, &mut ops);
	for o in &ops {
		for x in o.ints() {
			h.bytes(&x.to_le_bytes());
		}
		for x in o.feats {
			h.bytes(&x.to_bits().to_le_bytes());
		}
	}
	(sim, h.0)
}

fn play(n: u64) -> Vec<(u64, u64)> {
	let backs: Vec<Back> = ALL_BACKS
		.iter()
		.copied()
		.filter(|&b| b != Back::Challenge)
		.collect();
	let back = backs[num::pick(n, backs.len())];
	let stake = num::byte(num::pick(n, 8)) + 1;
	let mut g = Game::new_run(&format!("GOLD{n:04}"), back, stake);
	let mut rng = Xs(n.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
	let mut rerolls = 0;
	let mut out = vec![fingerprint(&mut g, "start")];
	for _ in 0..STEPS {
		if g.phase == Phase::GameOver {
			break;
		}
		if g.phase != Phase::Shop {
			rerolls = 0;
		}
		if rng.chance(0.1)
			&& let Some(inj) = random_inject(&g, &mut rng)
		{
			g.inject(&inj);
			out.push(fingerprint(&mut g, &inj.to_string()));
			continue;
		}
		let Some(a) = choose(&g, &mut rng, &mut rerolls) else {
			break;
		};
		g.step(a)
			.unwrap_or_else(|e| panic!("game {n}: {a:?} rejected: {e}"));
		out.push(fingerprint(&mut g, &a.to_string()));
	}
	out
}

/// Compares `games` (one fingerprint list per game) with the recording at
/// `file`, or re-records it; returns the diverging games.
fn check(file: &str, games: &[Vec<u64>], bless: bool) -> Vec<String> {
	use std::fmt::Write;

	let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
		.join("tests")
		.join(file);
	let mut text = String::new();
	for (i, v) in games.iter().enumerate() {
		let hashes: Vec<String> = v.iter().map(|h| format!("{h:016x}")).collect();
		writeln!(text, "{i} {}", hashes.join(" ")).expect("writing to a String cannot fail");
	}
	if bless {
		std::fs::write(&path, &text).expect("the golden file is writable");
		eprintln!(
			"recorded {file}: {} games, {} steps",
			games.len(),
			games.iter().map(std::vec::Vec::len).sum::<usize>()
		);
		return Vec::new();
	}
	let want = std::fs::read_to_string(&path)
		.unwrap_or_else(|_| panic!("tests/{file} missing; run with GOLDEN_BLESS=all"));
	let mut bad = Vec::new();
	for (i, (got, exp)) in text.lines().zip(want.lines()).enumerate() {
		if got != exp {
			let a: Vec<&str> = got.split(' ').skip(1).collect();
			let b: Vec<&str> = exp.split(' ').skip(1).collect();
			let step = (0..a.len().max(b.len()))
				.find(|&k| a.get(k) != b.get(k))
				.unwrap_or(0);
			bad.push(format!(
				"{file}: game {i} at step {step} (of {} vs {} recorded)",
				a.len(),
				b.len()
			));
		}
	}
	if text.lines().count() != want.lines().count() {
		bad.push(format!("{file}: number of games changed"));
	}
	bad
}

#[test]
fn golden_trajectories() {
	let bless = std::env::var("GOLDEN_BLESS").unwrap_or_default();
	assert!(
		matches!(bless.as_str(), "" | "obs" | "all"),
		"GOLDEN_BLESS must be `obs` or `all`"
	);
	let games: Vec<Vec<(u64, u64)>> = (0..GAMES).map(play).collect();
	let part = |f: fn(&(u64, u64)) -> u64| {
		games
			.iter()
			.map(|v| v.iter().map(f).collect())
			.collect::<Vec<Vec<u64>>>()
	};
	let mut bad = check("golden_sim.txt", &part(|h| h.0), bless == "all");
	bad.extend(check(
		"golden_obs.txt",
		&part(|h| h.1),
		bless == "obs" || bless == "all",
	));
	assert!(
		bad.is_empty(),
		"{} games diverge from the golden trajectories: {}",
		bad.len(),
		bad.join("; ")
	);
}
