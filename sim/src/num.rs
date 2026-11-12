//! Conversions between the simulator's number types.
//!
//! The game keeps every quantity in a Lua number (an `f64`), Rust indexes with
//! `usize`, and the observation is made of `f32` and `i32`. Each function here
//! is the one place where its `as` conversion is justified. The conversions
//! that must be exact are checked in builds with debug assertions (the tests),
//! and cost nothing in release builds.

/// `n` as a Lua number.
#[inline]
#[must_use]
#[expect(
	clippy::cast_precision_loss,
	reason = "counts and indices stay far below 2^53, so the result is exact"
)]
pub const fn float(n: usize) -> f64 {
	debug_assert!(
		n < 1 << 53,
		"a count this large is not exact as a Lua number"
	);
	n as f64
}

/// The integer part of a Lua number as a count. Like `as`, it saturates:
/// negative numbers and NaN give 0.
#[inline]
#[must_use]
#[expect(
	clippy::cast_possible_truncation,
	clippy::cast_sign_loss,
	reason = "truncating and saturating is the point"
)]
pub const fn trunc(x: f64) -> usize {
	x as usize
}

/// `x` as an observation feature. Observations are `f32`, so this rounds.
#[inline]
#[must_use]
#[expect(
	clippy::cast_possible_truncation,
	reason = "observation features are f32; rounding is intended"
)]
pub const fn feat(x: f64) -> f32 {
	x as f32
}

/// A small index or table length as a categorical id.
#[inline]
#[must_use]
#[expect(
	clippy::cast_possible_truncation,
	clippy::cast_possible_wrap,
	reason = "checked below in test builds"
)]
pub const fn id(n: usize) -> i32 {
	debug_assert!(
		n <= i32::MAX as usize,
		"an index this large does not fit an id"
	);
	n as i32
}

/// An index or count as the `u32` of card ids and counters.
#[inline]
#[must_use]
#[expect(
	clippy::cast_possible_truncation,
	reason = "checked below in test builds"
)]
pub const fn index32(n: usize) -> u32 {
	debug_assert!(
		n <= u32::MAX as usize,
		"an index this large does not fit a u32"
	);
	n as u32
}

/// A slot index or card count as the `u8` that actions and ops carry.
#[inline]
#[must_use]
#[expect(
	clippy::cast_possible_truncation,
	reason = "checked below in test builds"
)]
pub const fn byte(n: usize) -> u8 {
	debug_assert!(
		n <= u8::MAX as usize,
		"an index this large does not fit a byte"
	);
	n as u8
}

/// `hash` reduced to an index below `len`.
///
/// # Panics
///
/// Panics if `len` is 0.
#[inline]
#[must_use]
#[expect(
	clippy::cast_possible_truncation,
	reason = "the remainder is below `len`, a usize"
)]
pub const fn pick(hash: u64, len: usize) -> usize {
	(hash % len as u64) as usize
}
