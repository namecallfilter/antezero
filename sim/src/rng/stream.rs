//! Named `pseudoseed` stream table ([`PseudoRandom`]) and allocation-free
//! stream keys ([`Key`]).

use std::num::NonZeroU64;
use std::{fmt, mem, str};

use super::lua::{FNV_OFFSET_BASIS, LuaRandom, fnv1a, lua_mod, pseudohash, round13};
use super::trace::Trace;
use crate::num;

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
