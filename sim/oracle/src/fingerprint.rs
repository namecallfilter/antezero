//! Fingerprints of a simulator state: the canonical state with the last
//! action's timeline, and the learning observation with its ops (the golden
//! trajectories record them; the dump check compares observations).

use sim::game::Game;
use sim::obs::{self, ObsBuf};
use sim::{canon, rng};

struct Fnv(u64);

impl Fnv {
	fn new() -> Fnv {
		Fnv(rng::FNV_OFFSET_BASIS)
	}

	fn bytes(&mut self, b: &[u8]) {
		for &x in b {
			self.0 = rng::fnv1a(self.0, u64::from(x));
		}
	}

	fn str(&mut self, s: &str) {
		self.bytes(s.as_bytes());
		self.bytes(&[0xff]);
	}
}

/// Fingerprints of the state after a step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fingerprint {
	/// The simulator: canonical state and timeline.
	pub sim: u64,
	/// The learning interface: observation and ops.
	pub obs: u64,
}

/// The fingerprints of `g` after the action whose code is `action`.
///
/// # Panics
///
/// If `obs::Layout::of` disagrees with the layout the encoder produced.
pub fn fingerprint(g: &mut Game, action: &str) -> Fingerprint {
	let mut h = Fnv::new();
	h.str(action);
	for (k, v) in canon::from_sim(g) {
		h.str(&k);
		h.str(&v);
	}
	for t in g.timeline() {
		h.str(t);
	}
	Fingerprint {
		sim: h.0,
		obs: obs_fingerprint(g),
	}
}

/// The fingerprint of the observation of `g` (with the deck's tokens) and
/// of its ops (with four play candidates and the joker-order gains).
///
/// # Panics
///
/// If `obs::Layout::of` disagrees with the layout the encoder produced.
pub fn obs_fingerprint(g: &mut Game) -> u64 {
	obs_fingerprint_with(g, true, obs::OpOptions {
		candidates: 4,
		..Default::default()
	})
}

/// [`obs_fingerprint`] with the deck's tokens or not, and the ops of `options`.
///
/// # Panics
///
/// If `obs::Layout::of` disagrees with the layout the encoder produced.
pub fn obs_fingerprint_with(g: &mut Game, deck_tokens: bool, options: obs::OpOptions) -> u64 {
	let mut h = Fnv::new();
	let mut buf = ObsBuf::new();
	let lay = obs::encode_opts(g, &mut buf.as_mut(), deck_tokens);
	let of = obs::Layout::of(&g.areas, &g.tags);
	assert!(
		of == obs::Layout {
			n_tokens: of.n_tokens,
			..lay
		} && of.n_tokens <= lay.n_tokens,
		"Layout::of {of:?} vs encoded {lay:?}"
	);
	for x in buf.tok_i.iter().chain(&buf.glob_i) {
		h.bytes(&x.to_le_bytes());
	}
	for x in buf.tok_f.iter().chain(&buf.glob_f) {
		h.bytes(&x.to_bits().to_le_bytes());
	}
	let mut ops = Vec::new();
	obs::ops(g, &lay, options, &mut ops);
	obs::order_gain(g, &mut ops);
	for o in &ops {
		for x in o.ints() {
			h.bytes(&x.to_le_bytes());
		}
		for x in o.feats {
			h.bytes(&x.to_bits().to_le_bytes());
		}
	}
	h.0
}
