//! `LuaJIT` PRNG and arithmetic primitives (`LuaRandom`, `pseudohash`,
//! `round13`, `shuffle`), plus non-game hashing and search RNGs.

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
	fn round13_basic() {
		assert_eq!(round13(0.5), 0.5);
		assert_eq!(round13(0.123_456_789_012_345_67), 0.123_456_789_012_3);
		assert_eq!(round13(0.999_999_999_999_99), 1.0);
		assert_eq!(round13(0.0), 0.0);
		// Exact decimal ties round half to even (x * 1e13 = 610351562.5 / 1831054687.5).
		assert_eq!(round13(1.0 / 16384.0), 0.000_061_035_156_2);
		assert_eq!(round13(3.0 / 16384.0), 0.000_183_105_468_8);
	}
}
