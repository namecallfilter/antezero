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
use std::num::NonZeroU64;
use std::{fmt, mem, str};

use crate::num;

/// Records a timeline entry in a [`PseudoRandom`]'s trace (`trace-keys`
/// builds only).
macro_rules! trace {
    ($rng:expr, $($t:tt)*) => {
        #[cfg(feature = "trace-keys")]
        $rng.trace.record(format_args!($($t)*));
    };
}
pub(crate) use trace;

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
		for slot in &mut rs.state {
			let m: u64 = 1u64 << (r & 255);
			r >>= 8;
			// LuaJIT's literals are exactly M_PI and M_E.
			d = d * PI + E;
			let mut u = d.to_bits();
			if u < m {
				u += m; // Ensure k[i] MSB of gen[i] are non-zero.
			}
			*slot = u;
		}
		for _ in 0..10 {
			rs.step();
		}
		rs
	}

	/// One PRNG step. Returns the raw bits of a double in `[1, 2)`.
	#[inline]
	fn step(&mut self) -> u64 {
		#[inline]
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
#[inline]
#[must_use]
pub fn lua_mod(a: f64, b: f64) -> f64 {
	a - (a / b).floor() * b
}

/// The game's (`LuaJIT`'s) `base ^ exponent`: a whole exponent of at most 65536 either
/// way multiplies by repeated squaring (`lj_vm_powi`), which rounds large
/// powers otherwise than `pow`; any other exponent is `pow`'s.
#[must_use]
pub fn lua_pow(base: f64, exponent: f64) -> f64 {
	let Some(whole) = num::whole(exponent).filter(|k| k.abs() <= 65_536) else {
		return base.powf(exponent);
	};
	let mut x = if whole < 0 { 1.0 / base } else { base };
	let mut n = whole.unsigned_abs();
	if n == 0 {
		return 1.0;
	}
	while n & 1 == 0 {
		x *= x;
		n >>= 1;
	}
	let mut product = x;
	n >>= 1;
	if n != 0 {
		loop {
			x *= x;
			if n == 1 {
				break;
			}
			if n & 1 == 1 {
				product *= x;
			}
			n >>= 1;
		}
		product *= x;
	}
	product
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
	buf: [u8; Key::CAPACITY],
}

impl Key {
	/// The longest key, in bytes.
	pub const CAPACITY: usize = 63;

	#[inline]
	#[must_use]
	pub const fn new() -> Self {
		Key {
			len: 0,
			buf: [0; Key::CAPACITY],
		}
	}

	/// # Panics
	///
	/// Panics if the key would grow past [`Key::CAPACITY`] bytes.
	#[inline]
	pub fn push(&mut self, s: &str) -> &mut Self {
		self.push_bytes(s.as_bytes())
	}

	/// # Panics
	///
	/// Panics if the key would grow past [`Key::CAPACITY`] bytes.
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
	///
	/// # Panics
	///
	/// Panics if the key would grow past [`Key::CAPACITY`] bytes.
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
		&self.buf[..usize::from(self.len)]
	}

	/// 64-bit hash used to index stream state (a hash of 0 is taken as 1).
	#[inline]
	fn hash(&self) -> NonZeroU64 {
		// FNV-1a followed by a murmur3 finalizer.
		let mut h = FNV_OFFSET_BASIS;
		for &b in self.as_bytes() {
			h = fnv1a(h, u64::from(b));
		}
		h ^= h >> 33;
		h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
		h ^= h >> 33;
		h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
		h ^= h >> 33;
		NonZeroU64::new(h).unwrap_or(NonZeroU64::MIN)
	}
}

impl Default for Key {
	fn default() -> Self {
		Key::new()
	}
}

impl From<&str> for Key {
	/// # Panics
	///
	/// Panics if `s` is longer than [`Key::CAPACITY`] bytes.
	#[inline]
	fn from(s: &str) -> Self {
		let mut k = Key::new();
		k.push(s);
		k
	}
}

impl fmt::Debug for Key {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
pub(crate) struct Trace {
	/// Whether entries are recorded (on by default in feature builds).
	enabled: bool,
	/// Frame of the current action, stamped on every entry.
	frame: u32,
	entries: Vec<String>,
	/// Completed-event log, when enabled.
	event_log: Option<Vec<String>>,
}

#[cfg(feature = "trace-keys")]
impl Default for Trace {
	fn default() -> Self {
		Trace {
			enabled: true,
			frame: 0,
			entries: Vec::new(),
			event_log: None,
		}
	}
}

#[cfg(feature = "trace-keys")]
impl Trace {
	#[inline]
	pub(crate) fn record(&mut self, what: fmt::Arguments<'_>) {
		if self.enabled {
			self.entries.push(format!("{} {what}", self.frame));
		}
	}

	/// Stamps the entries that follow with `frame`.
	#[inline]
	pub(crate) fn set_frame(&mut self, frame: u32) {
		self.frame = frame;
	}

	/// Starts the timeline of a new action.
	pub(crate) fn clear(&mut self) {
		self.frame = 0;
		self.entries.clear();
		if let Some(l) = self.event_log.as_mut() {
			l.clear();
		}
	}

	/// Turns recording on or off; off also drops the timeline.
	pub(crate) fn set_enabled(&mut self, on: bool) {
		self.enabled = on;
		if !on {
			self.entries = Vec::new();
		}
	}

	pub(crate) fn entries(&self) -> &[String] {
		&self.entries
	}

	/// Starts logging the completed events of each action.
	pub(crate) fn enable_event_log(&mut self) {
		self.event_log = Some(Vec::new());
	}

	/// The completed events of the current action, if logging is on.
	pub(crate) fn event_log(&self) -> &[String] {
		self.event_log.as_deref().unwrap_or(&[])
	}

	/// Logs a completed event, if logging is on.
	pub(crate) fn log_event(&mut self, what: fmt::Arguments<'_>) {
		if let Some(log) = self.event_log.as_mut() {
			log.push(what.to_string());
		}
	}
}

/// Without `trace-keys` there is no timeline: an empty stand-in whose
/// methods do nothing.
#[cfg(not(feature = "trace-keys"))]
#[derive(Clone, Debug, Default)]
pub(crate) struct Trace {}

#[cfg(not(feature = "trace-keys"))]
#[expect(
	clippy::unused_self,
	reason = "the stand-in records nothing, so its methods have nothing in `self` to use"
)]
impl Trace {
	#[inline]
	pub(crate) fn set_frame(&mut self, _frame: u32) {}

	#[inline]
	pub(crate) fn clear(&mut self) {}

	pub(crate) fn set_enabled(&mut self, _on: bool) {}

	pub(crate) fn entries(&self) -> &[String] {
		&[]
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

/// The slot a stream with hash `h` starts probing at, in a table of
/// `mask + 1` slots.
#[inline]
#[expect(
	clippy::cast_possible_truncation,
	reason = "only the hash's low bits index the table"
)]
const fn home(h: NonZeroU64, mask: usize) -> usize {
	(h.get() as usize) & mask
}

/// A slot of the stream table: a stream's key hash and state, or empty.
#[derive(Clone, Copy, Debug)]
struct Slot {
	hash: Option<NonZeroU64>,
	state: f64,
}

impl Slot {
	const EMPTY: Slot = Slot {
		hash: None,
		state: 0.0,
	};
}

/// Per-key stream states (`G.GAME.pseudorandom`) plus the run seed.
///
/// Stored as an open-addressing table of `Slot`s so cloning a snapshot
/// copies one contiguous buffer.
#[derive(Debug)]
pub struct PseudoRandom {
	seed: Key,
	hashed_seed: f64,
	len: usize,
	slots: Vec<Slot>,
	/// Number of `pseudoseed` calls so far (any random draw).
	draws: u64,
	pub(crate) trace: Trace,
}

impl Clone for PseudoRandom {
	fn clone(&self) -> PseudoRandom {
		PseudoRandom {
			slots: self.slots.clone(),
			trace: self.trace.clone(),
			..*self
		}
	}

	fn clone_from(&mut self, source: &PseudoRandom) {
		let PseudoRandom {
			seed,
			hashed_seed,
			len,
			slots,
			draws,
			trace,
		} = self;
		*seed = source.seed;
		*hashed_seed = source.hashed_seed;
		*len = source.len;
		slots.clone_from(&source.slots);
		*draws = source.draws;
		trace.clone_from(&source.trace);
	}
}

impl PseudoRandom {
	/// # Panics
	///
	/// Panics if `seed` is longer than [`Key::CAPACITY`] bytes.
	#[must_use]
	pub fn new(seed: &str) -> Self {
		let seed_key = Key::from(seed);
		PseudoRandom {
			seed: seed_key,
			hashed_seed: pseudohash(seed_key.as_bytes()),
			len: 0,
			slots: vec![Slot::EMPTY; 256],
			draws: 0,
			trace: Trace::default(),
		}
	}

	/// Starts over from `seed`; a trace carries on.
	pub(crate) fn reseed(&mut self, seed: &str) {
		*self = PseudoRandom {
			trace: mem::take(&mut self.trace),
			..PseudoRandom::new(seed)
		};
	}

	/// A generator whose calls are never traced (scratch scenarios).
	///
	/// # Panics
	///
	/// Panics if `seed` is longer than [`Key::CAPACITY`] bytes.
	#[must_use]
	pub fn new_untraced(seed: &str) -> Self {
		let mut r = Self::new(seed);
		r.trace.set_enabled(false);
		r
	}

	/// Slot of the stream with hash `h`: `Ok(index)` if present, else
	/// `Err(index)` of the empty slot where it would be inserted.
	#[inline]
	fn find(&self, h: NonZeroU64) -> Result<usize, usize> {
		let mask = self.slots.len() - 1;
		let mut i = home(h, mask);
		loop {
			match self.slots[i].hash {
				Some(s) if s == h => return Ok(i),
				None => return Err(i),
				Some(_) => i = (i + 1) & mask,
			}
		}
	}

	/// Every stream used so far, by name, with its state (in table order).
	///
	/// # Panics
	///
	/// Panics on a stream whose name was never registered (a bug: every
	/// stream registers its name when it is created).
	pub fn named_streams(&self) -> impl Iterator<Item = (String, f64)> + '_ {
		self.slots.iter().filter_map(|slot| {
			let name = names::get(slot.hash?.get())
				.expect("every stream registers its name when it is created");
			Some((name, slot.state))
		})
	}

	/// The run's seed.
	///
	/// # Panics
	///
	/// Never: the key holds a whole copy of the `&str` the generator was
	/// seeded with.
	#[must_use]
	pub fn seed(&self) -> &str {
		str::from_utf8(self.seed.as_bytes()).expect("the seed key is a whole copy of a `&str`")
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

	/// The number of streams used so far.
	#[must_use]
	pub fn stream_count(&self) -> usize {
		self.len
	}

	#[inline]
	fn slot(&mut self, key: &Key) -> &mut f64 {
		let h = key.hash();
		let i = match self.find(h) {
			Ok(i) => return &mut self.slots[i].state,
			Err(i) => i,
		};
		// First use of this stream: state = pseudohash(key .. seed).
		if (self.len + 1) * 2 > self.slots.len() {
			self.grow();
			return self.slot(key);
		}
		let mut full = *key;
		full.push_bytes(self.seed.as_bytes());
		self.slots[i] = Slot {
			hash: Some(h),
			state: pseudohash(full.as_bytes()),
		};
		self.len += 1;
		names::note(h.get(), key);
		&mut self.slots[i].state
	}

	fn grow(&mut self) {
		let cap = self.slots.len() * 2;
		let old = mem::replace(&mut self.slots, vec![Slot::EMPTY; cap]);
		let mask = self.slots.len() - 1;
		for slot in old {
			if let Some(h) = slot.hash {
				let mut i = home(h, mask);
				while self.slots[i].hash.is_some() {
					i = (i + 1) & mask;
				}
				self.slots[i] = slot;
			}
		}
	}

	/// Sets the stream `key` to `state`, creating it if needed.
	pub(crate) fn set_state(&mut self, key: &Key, state: f64) {
		*self.slot(key) = state;
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
		// `midpoint` is `(a + b) / 2`, as the game computes it, for values this small.
		st.midpoint(hashed)
	}

	/// `pseudoseed(str)` convenience.
	///
	/// # Panics
	///
	/// Panics if `key` is longer than [`Key::CAPACITY`] bytes.
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

	/// [`PseudoRandom::element`] with the key given as a string.
	///
	/// # Panics
	///
	/// Panics if `len` is 0 or `key` is longer than [`Key::CAPACITY`] bytes.
	#[inline]
	pub fn element_str(&mut self, key: &str, len: usize) -> usize {
		self.element(&Key::from(key), len)
	}
}

/// FNV-1a's 64-bit offset basis, the hash of nothing.
pub const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

/// One FNV-1a step: `h` with `x` folded in.
#[inline]
#[must_use]
pub fn fnv1a(h: u64, x: u64) -> u64 {
	(h ^ x).wrapping_mul(0x0000_0100_0000_01b3)
}

/// `SplitMix64`'s increment, the odd integer nearest 2^64 over the golden ratio.
pub const SPLITMIX_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;

/// One `SplitMix64` output for the state `n`: a well spread hash of `n`, for
/// arbitrary, repeatable choices like [`Xorshift`]'s. Never the game's own
/// randomness.
#[must_use]
pub fn splitmix64(n: u64) -> u64 {
	let mut z = n.wrapping_add(SPLITMIX_GAMMA);
	z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
	z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
	z ^ (z >> 31)
}

/// Shuffles `v` (Fisher-Yates) with the `SplitMix64` outputs that follow
/// `seed`: an arbitrary, repeatable order, never the game's own randomness.
pub(crate) fn splitmix_shuffle<T>(v: &mut [T], mut seed: u64) {
	for i in (1..v.len()).rev() {
		let z = splitmix64(seed);
		seed = seed.wrapping_add(SPLITMIX_GAMMA);
		v.swap(i, num::pick(z, i + 1));
	}
}

/// A xorshift64 generator for arbitrary, repeatable choices: the policies
/// of tests, benchmarks and rollouts. Never the game's own randomness.
#[derive(Clone, Debug)]
pub struct Xorshift(u64);

impl Xorshift {
	/// A generator started from `seed` made odd: a zero state would stay
	/// zero, so every draw would be zero.
	#[must_use]
	pub fn new(seed: u64) -> Xorshift {
		Xorshift(seed | 1)
	}

	pub fn next_u64(&mut self) -> u64 {
		let mut x = self.0;
		x ^= x << 13;
		x ^= x >> 7;
		x ^= x << 17;
		self.0 = x;
		x
	}

	/// An index below `n`.
	///
	/// # Panics
	///
	/// If `n` is 0.
	pub fn below(&mut self, n: usize) -> usize {
		num::pick(self.next_u64(), n)
	}

	/// `true` with probability `p`.
	#[expect(
		clippy::cast_precision_loss,
		reason = "the draw is shifted below 2^53 and 2^53 is a power of two, so both casts are exact"
	)]
	pub fn chance(&mut self, p: f64) -> bool {
		((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64) < p
	}
}

/// Fisher–Yates exactly as `pseudoshuffle` does it (after the caller sorted the
/// list by `sort_id`).
#[inline]
pub fn shuffle<T>(list: &mut [T], seed: f64) {
	let mut r = LuaRandom::new(seed);
	let mut i = list.len();
	while i >= 2 {
		let j = r.index(i);
		list.swap(i - 1, j);
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
	fn table_growth_preserves_state() {
		let key = |i: i64| {
			let mut k = Key::from("k");
			k.push_int(i);
			k
		};
		let mut a = PseudoRandom::new("SEED");
		for i in 0..2000 {
			a.seed_for(&key(i));
		}
		assert_eq!(a.stream_count(), 2000);
		assert_eq!(a.named_streams().count(), 2000);
		// Every stream keeps its state across the table growths: advancing it
		// again matches a table that only ever held that stream.
		for i in 0..2000 {
			let mut b = PseudoRandom::new("SEED");
			b.seed_for(&key(i));
			assert_eq!(a.seed_for(&key(i)), b.seed_for(&key(i)), "stream {i}");
		}
	}
}
