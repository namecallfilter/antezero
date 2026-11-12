//! Bit-exact reimplementation of Balatro's randomness.
//!
//! Balatro (LÖVE / `LuaJIT` 2.1) derives every random decision from named
//! streams: `pseudoseed(key)` advances a per-key float state and the result is
//! fed to `math.randomseed`, after which `math.random` draws from `LuaJIT`'s
//! Tausworthe (TW223) generator. Reproducing both layers exactly is what makes
//! seeds behave identically to the real game.
//!
//! Stream states live in one flat table, so snapshotting the RNG is a single
//! contiguous copy (see [`PseudoRandom`]).

use std::f64::consts::{E, PI};

use crate::num;

/// `LuaJIT`'s `math.random` generator state (four combined Tausworthe LFSRs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LuaRandom {
	state: [u64; 4],
}

impl LuaRandom {
	/// Equivalent of `math.randomseed(seed)`.
	#[inline]
	#[must_use]
	pub fn new(seed: f64) -> Self {
		let mut rs = LuaRandom { state: [0; 4] };
		// 64-k[i] as four 8 bit constants.
		let mut r: u32 = 0x1109_0601;
		let mut d = seed;
		for i in 0..4 {
			let m: u64 = 1u64 << (r & 255);
			r >>= 8;
			// LuaJIT's literals are exactly M_PI and M_E.
			d = d * PI + E;
			let mut u = d.to_bits();
			if u < m {
				u += m; // Ensure k[i] MSB of gen[i] are non-zero.
			}
			rs.state[i] = u;
		}
		for _ in 0..10 {
			rs.step();
		}
		rs
	}

	/// One PRNG step. Returns the raw bits of a double in `[1, 2)`.
	#[inline(always)]
	fn step(&mut self) -> u64 {
		#[inline(always)]
		#[expect(
			clippy::inline_always,
			reason = "kept from the tuned original: this runs for every random draw"
		)]
		fn tw(z: u64, k: u32, q: u32, s: u32) -> u64 {
			(((z << q) ^ z) >> (k - s)) ^ ((z & (u64::MAX << (64 - k))) << s)
		}
		let z0 = tw(self.state[0], 63, 31, 18);
		let z1 = tw(self.state[1], 58, 19, 28);
		let z2 = tw(self.state[2], 55, 24, 7);
		let z3 = tw(self.state[3], 47, 21, 8);
		self.state = [z0, z1, z2, z3];
		let r = z0 ^ z1 ^ z2 ^ z3;
		(r & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000
	}

	/// `math.random()` — a double in `[0, 1)`.
	#[inline]
	pub fn random(&mut self) -> f64 {
		f64::from_bits(self.step()) - 1.0
	}

	/// `math.random(m)` — an integer in `[1, m]`.
	#[inline]
	pub fn random_int(&mut self, m: f64) -> f64 {
		(self.random() * m).floor() + 1.0
	}

	/// `math.random(m, n)` — an integer in `[m, n]`.
	#[inline]
	pub fn random_range(&mut self, m: f64, n: f64) -> f64 {
		(self.random() * (n - m + 1.0)).floor() + m
	}

	/// `math.random(#t)` as a zero-based index.
	#[inline]
	pub fn index(&mut self, len: usize) -> usize {
		num::trunc(self.random_int(num::float(len))) - 1
	}
}

/// Lua's `a % b` for numbers: `a - floor(a/b)*b`.
#[inline(always)]
#[expect(
	clippy::inline_always,
	reason = "kept from the tuned original: this runs for every random draw"
)]
#[must_use]
pub fn lua_mod(a: f64, b: f64) -> f64 {
	a - (a / b).floor() * b
}

/// `pseudohash(str)` from `misc_functions.lua`.
#[inline]
#[must_use]
pub fn pseudohash(bytes: &[u8]) -> f64 {
	let mut num = 1.0f64;
	let mut i = bytes.len();
	while i > 0 {
		let b = f64::from(bytes[i - 1]);
		num = lua_mod((1.123_928_502_3 / num) * b * PI + PI * num::float(i), 1.0);
		i -= 1;
	}
	num
}

/// `tonumber(string.format("%.13f", x))` for `x` in `[0, 1]`, exactly.
///
/// The decimal value `n / 10^13` is recovered by rounding the *exact* product
/// `x * 10^13` to the nearest integer (an FMA recovers the product's rounding
/// error), then dividing — IEEE division of two exact integers is correctly
/// rounded, which is precisely what `tonumber` produces for that string.
#[inline]
#[must_use]
#[expect(
	clippy::float_cmp,
	reason = "exact ties decide the rounding, as in printf"
)]
pub fn round13(x: f64) -> f64 {
	const S: f64 = 1e13;
	let product = x * S;
	let err = x.mul_add(S, -product);
	let floor = product.floor();
	let frac = product - floor;
	let rounded = if frac > 0.5 || (frac == 0.5 && err > 0.0) {
		floor + 1.0
	} else if frac < 0.5 || err < 0.0 || (floor * 0.5).floor() * 2.0 == floor {
		// (the last case: an exact tie, e.g. x = 1/16384, rounds half to
		// even like printf)
		floor
	} else {
		floor + 1.0
	};
	rounded / S
}

/// A small, allocation-free byte buffer used to build stream keys such as
/// `"Joker1sho3"` exactly the way Lua's `..` operator would.
#[derive(Clone, Copy)]
pub struct Key {
	len: u8,
	buf: [u8; 63],
}

impl Key {
	#[inline]
	#[must_use]
	pub const fn new() -> Self {
		Key {
			len: 0,
			buf: [0; 63],
		}
	}

	#[inline]
	pub fn push(&mut self, s: &str) -> &mut Self {
		self.push_bytes(s.as_bytes())
	}

	/// # Panics
	///
	/// Panics if the key would grow past its 63 bytes.
	#[inline]
	pub fn push_bytes(&mut self, s: &[u8]) -> &mut Self {
		let l = self.len as usize;
		// A truncated key would silently be another stream than the game's.
		assert!(
			l + s.len() <= self.buf.len(),
			"RNG stream key longer than {} bytes",
			self.buf.len()
		);
		self.buf[l..l + s.len()].copy_from_slice(s);
		self.len = num::byte(l + s.len());
		self
	}

	/// Appends an integer the way Lua's `tostring` formats integral numbers.
	#[inline]
	pub fn push_int(&mut self, v: i64) -> &mut Self {
		let mut tmp = [0u8; 20];
		let mut i = tmp.len();
		let mut u = v.unsigned_abs();
		loop {
			i -= 1;
			tmp[i] = b'0' + (u % 10) as u8;
			u /= 10;
			if u == 0 {
				break;
			}
		}
		if v < 0 {
			i -= 1;
			tmp[i] = b'-';
		}
		self.push_bytes(&tmp[i..])
	}

	#[inline]
	#[must_use]
	pub fn as_bytes(&self) -> &[u8] {
		&self.buf[..self.len as usize]
	}

	/// 64-bit hash used to index stream state (never 0).
	#[inline]
	fn hash(&self) -> u64 {
		// FNV-1a followed by a murmur3 finalizer.
		let mut h: u64 = 0xcbf2_9ce4_8422_2325;
		for &b in self.as_bytes() {
			h ^= u64::from(b);
			h = h.wrapping_mul(0x0000_0100_0000_01b3);
		}
		h ^= h >> 33;
		h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
		h ^= h >> 33;
		h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
		h ^= h >> 33;
		if h == 0 { 1 } else { h }
	}
}

impl Default for Key {
	fn default() -> Self {
		Key::new()
	}
}

impl From<&str> for Key {
	#[inline]
	fn from(s: &str) -> Self {
		let mut k = Key::new();
		k.push(s);
		k
	}
}

impl std::fmt::Debug for Key {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "Key({:?})", String::from_utf8_lossy(self.as_bytes()))
	}
}

/// The diagnostic timeline of the current action (feature `trace-keys`,
/// used by the oracle differential tests): `"<frame> <what>"` entries for
/// every `pseudoseed` call, card draw and card removal, and optionally a log
/// of completed events. It lives with the RNG because every random draw is a
/// timeline entry.
#[cfg(feature = "trace-keys")]
#[derive(Clone, Debug)]
pub struct Trace {
	/// Whether entries are recorded (on by default in feature builds).
	pub enabled: bool,
	/// Frame of the current action, stamped on every entry.
	pub frame: u32,
	pub entries: Vec<String>,
	/// Completed-event log, when enabled.
	pub evlog: Option<Vec<String>>,
}

#[cfg(feature = "trace-keys")]
impl Default for Trace {
	fn default() -> Self {
		Trace {
			enabled: true,
			frame: 0,
			entries: Vec::new(),
			evlog: None,
		}
	}
}

#[cfg(feature = "trace-keys")]
impl Trace {
	#[inline]
	pub fn record(&mut self, what: std::fmt::Arguments<'_>) {
		if self.enabled {
			self.entries.push(format!("{} {what}", self.frame));
		}
	}
}

/// Names of the streams created in this process, by key hash. The stream
/// table stores only hashes; diagnostics (`canon::from_sim`) look names up
/// here. A thread-local set keeps the shared map off the hot path.
mod names {
	use std::cell::RefCell;
	use std::collections::{HashMap, HashSet};
	use std::sync::{OnceLock, RwLock};

	use super::Key;

	fn registry() -> &'static RwLock<HashMap<u64, Box<str>>> {
		static R: OnceLock<RwLock<HashMap<u64, Box<str>>>> = OnceLock::new();
		R.get_or_init(Default::default)
	}

	thread_local! {
		static SEEN: RefCell<HashSet<u64>> = RefCell::new(HashSet::new());
	}

	pub(super) fn note(hash: u64, key: &Key) {
		if SEEN.with(|s| s.borrow_mut().insert(hash)) {
			let name = String::from_utf8_lossy(key.as_bytes()).into();
			registry()
				.write()
				.expect("no writer panics while holding the lock")
				.entry(hash)
				.or_insert(name);
		}
	}

	pub(super) fn get(hash: u64) -> Option<String> {
		registry()
			.read()
			.expect("no writer panics while holding the lock")
			.get(&hash)
			.map(ToString::to_string)
	}
}

/// Per-key stream states (`G.GAME.pseudorandom`) plus the run seed.
///
/// Stored as an open-addressing table of `(hash, state)` pairs so cloning a
/// snapshot copies one contiguous buffer.
#[derive(Clone, Debug)]
pub struct PseudoRandom {
	seed: Key,
	hashed_seed: f64,
	len: usize,
	slots: Vec<(u64, f64)>,
	/// Number of `pseudoseed` calls so far (any random draw).
	draws: u64,
	#[cfg(feature = "trace-keys")]
	pub trace: Trace,
}

impl PseudoRandom {
	#[must_use]
	pub fn new(seed: &str) -> Self {
		let seed_key = Key::from(seed);
		PseudoRandom {
			seed: seed_key,
			hashed_seed: pseudohash(seed_key.as_bytes()),
			len: 0,
			slots: vec![(0, 0.0); 256],
			draws: 0,
			#[cfg(feature = "trace-keys")]
			trace: Trace::default(),
		}
	}

	/// A generator whose calls are never traced (scratch scenarios).
	#[must_use]
	pub fn new_untraced(seed: &str) -> Self {
		#[cfg_attr(
			not(feature = "trace-keys"),
			expect(unused_mut, reason = "only trace builds change it")
		)]
		let mut r = Self::new(seed);
		#[cfg(feature = "trace-keys")]
		{
			r.trace.enabled = false;
		}
		r
	}

	/// Slot of the stream with hash `h`: `Ok(index)` if present, else
	/// `Err(index)` of the empty slot where it would be inserted.
	#[inline]
	fn find(&self, h: u64) -> Result<usize, usize> {
		let mask = self.slots.len() - 1;
		#[expect(
			clippy::cast_possible_truncation,
			reason = "only the hash's low bits index the table"
		)]
		let mut i = (h as usize) & mask;
		loop {
			match self.slots[i].0 {
				s if s == h => return Ok(i),
				0 => return Err(i),
				_ => i = (i + 1) & mask,
			}
		}
	}

	/// Current state of the stream `key` (None if never used).
	#[must_use]
	pub fn stream_value(&self, key: &str) -> Option<f64> {
		self.find(Key::from(key).hash())
			.ok()
			.map(|i| self.slots[i].1)
	}

	/// Names of all streams used so far (in table order).
	#[must_use]
	pub fn stream_names(&self) -> Vec<String> {
		self.slots
			.iter()
			.filter(|s| s.0 != 0)
			.filter_map(|s| names::get(s.0))
			.collect()
	}

	#[must_use]
	pub fn seed(&self) -> &str {
		std::str::from_utf8(self.seed.as_bytes()).unwrap_or("")
	}

	#[must_use]
	pub fn hashed_seed(&self) -> f64 {
		self.hashed_seed
	}

	/// Number of random draws (`pseudoseed` calls) so far.
	#[must_use]
	pub fn draws(&self) -> u64 {
		self.draws
	}

	/// Number of streams touched so far.
	#[must_use]
	pub fn streams(&self) -> usize {
		self.len
	}

	#[inline]
	fn slot(&mut self, key: &Key) -> &mut f64 {
		let h = key.hash();
		let i = match self.find(h) {
			Ok(i) => return &mut self.slots[i].1,
			Err(i) => i,
		};
		// First use of this stream: state = pseudohash(key .. seed).
		if (self.len + 1) * 2 > self.slots.len() {
			self.grow();
			return self.slot(key);
		}
		let mut full = *key;
		full.push_bytes(self.seed.as_bytes());
		self.slots[i] = (h, pseudohash(full.as_bytes()));
		self.len += 1;
		names::note(h, key);
		&mut self.slots[i].1
	}

	fn grow(&mut self) {
		let cap = self.slots.len() * 2;
		let old = std::mem::replace(&mut self.slots, vec![(0, 0.0); cap]);
		let mask = self.slots.len() - 1;
		for (h, v) in old {
			if h != 0 {
				#[expect(
					clippy::cast_possible_truncation,
					reason = "only the hash's low bits index the table"
				)]
				let mut i = (h as usize) & mask;
				while self.slots[i].0 != 0 {
					i = (i + 1) & mask;
				}
				self.slots[i] = (h, v);
			}
		}
	}

	/// `pseudoseed(key)`.
	#[inline]
	pub fn seed_for(&mut self, key: &Key) -> f64 {
		#[cfg(feature = "trace-keys")]
		self.trace.record(format_args!(
			"rng {}",
			String::from_utf8_lossy(key.as_bytes())
		));
		self.draws += 1;
		let hashed = self.hashed_seed;
		let st = self.slot(key);
		*st = round13(lua_mod(2.134_453_429_141 + *st * 1.724_312_34, 1.0)).abs();
		#[expect(
			clippy::manual_midpoint,
			reason = "the game's own arithmetic, kept bit for bit"
		)]
		let seed = (*st + hashed) / 2.0;
		seed
	}

	/// `pseudoseed(str)` convenience.
	#[inline]
	pub fn seed_str(&mut self, key: &str) -> f64 {
		self.seed_for(&Key::from(key))
	}

	/// `pseudorandom(key)` → float in `[0, 1)`.
	#[inline]
	pub fn random(&mut self, key: &Key) -> f64 {
		LuaRandom::new(self.seed_for(key)).random()
	}

	/// `pseudorandom(key, min, max)`.
	#[inline]
	pub fn random_range(&mut self, key: &Key, min: f64, max: f64) -> f64 {
		LuaRandom::new(self.seed_for(key)).random_range(min, max)
	}

	/// `pseudorandom_element` over a list already ordered the way the game
	/// sorts it (by integer key or by `sort_id`); returns the chosen index.
	///
	/// # Panics
	///
	/// Panics if `len` is 0.
	#[inline]
	pub fn element(&mut self, key: &Key, len: usize) -> usize {
		assert!(len > 0, "pseudorandom_element of an empty list");
		LuaRandom::new(self.seed_for(key)).index(len)
	}

	#[inline]
	pub fn element_str(&mut self, key: &str, len: usize) -> usize {
		self.element(&Key::from(key), len)
	}
}

/// Fisher–Yates exactly as `pseudoshuffle` does it (after the caller sorted the
/// list by `sort_id`).
#[inline]
pub fn shuffle<T>(list: &mut [T], seed: f64) {
	let mut r = LuaRandom::new(seed);
	let mut i = list.len();
	while i >= 2 {
		let j = num::trunc(r.random_int(num::float(i)));
		list.swap(i - 1, j - 1);
		i -= 1;
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn key_int_formatting() {
		let mut k = Key::from("rarity");
		k.push_int(1).push("sho");
		assert_eq!(k.as_bytes(), b"rarity1sho");
		let mut k = Key::new();
		k.push_int(-12).push_int(0).push_int(1_234_567);
		assert_eq!(k.as_bytes(), b"-1201234567");
	}

	#[test]
	#[expect(clippy::float_cmp, reason = "the test compares exact values")]
	fn round13_basic() {
		assert_eq!(round13(0.5), 0.5);
		assert_eq!(round13(0.123_456_789_012_345_67), 0.123_456_789_012_3);
		assert_eq!(round13(0.999_999_999_999_99), 1.0);
		assert_eq!(round13(0.0), 0.0);
		// Exact decimal ties round half to even (x * 1e13 = 610351562.5 / 1831054687.5).
		assert_eq!(round13(1.0 / 16384.0), 0.000_061_035_156_2);
		assert_eq!(round13(3.0 / 16384.0), 0.000_183_105_468_8);
	}

	#[test]
	#[expect(clippy::float_cmp, reason = "the test compares exact values")]
	fn streams_are_independent_and_stable() {
		let mut a = PseudoRandom::new("ABCDEFGH");
		let mut b = PseudoRandom::new("ABCDEFGH");
		let x1 = a.seed_str("boss");
		let _ = a.seed_str("shuffle");
		let x2 = a.seed_str("boss");
		assert_eq!(x1, b.seed_str("boss"));
		assert_eq!(x2, b.seed_str("boss"));
		assert_ne!(x1, x2);
	}

	#[test]
	#[expect(clippy::float_cmp, reason = "the test compares exact values")]
	fn table_growth_preserves_state() {
		let key = |i: i64| {
			let mut k = Key::from("k");
			k.push_int(i);
			k
		};
		let mut a = PseudoRandom::new("SEED");
		let mut states = Vec::new();
		for i in 0..2000 {
			a.seed_for(&key(i));
			states.push(
				a.stream_value(std::str::from_utf8(key(i).as_bytes()).unwrap())
					.unwrap(),
			);
		}
		assert_eq!(a.streams(), 2000);
		// Every stream keeps its state across the table growths, and advancing
		// an early stream matches a table that only ever held that stream.
		for (i, &state) in (0..).zip(&states) {
			assert_eq!(
				a.stream_value(std::str::from_utf8(key(i).as_bytes()).unwrap()),
				Some(state)
			);
		}
		let mut b = PseudoRandom::new("SEED");
		b.seed_for(&key(3));
		assert_eq!(a.seed_for(&key(3)), b.seed_for(&key(3)));
	}
}
