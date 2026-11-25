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

mod lua;
mod stream;
mod trace;

pub(crate) use lua::splitmix_shuffle;
pub use lua::{
	FNV_OFFSET_BASIS, LuaRandom, SPLITMIX_GAMMA, Xorshift, fnv1a, lua_mod, lua_pow, pseudohash,
	round13, shuffle, splitmix64,
};
pub use stream::{Key, PseudoRandom};
pub(crate) use trace::trace;
