//! Headless, deterministic, bit-exact simulator of Balatro (v1.0.1o) game
//! logic, built for reinforcement learning and tree search.
//!
//! * [`Game`] holds the complete state, including every RNG stream, so a
//!   snapshot is just `game.clone()` and restoring is assignment.
//! * Randomness reproduces the game's `LuaJIT` `math.random` and `pseudoseed`
//!   streams exactly, so a seed plays out like in the real game: the oracle
//!   tests compare the full state (every RNG stream) and the order of RNG
//!   calls, draws and removals after every action, and the event frames,
//!   which match in about 99% of steps (see `sim/oracle`).

pub mod actions;
mod areas;
pub mod canon;
pub mod card;
mod consumables;
pub mod data;
mod data_gen;
pub mod dump;
mod event;
mod flow;
pub mod game;
pub mod hand_eval;
pub mod inject;
mod joker;
pub mod json;
pub mod num;
pub mod obs;
mod pools;
pub mod rng;
mod round;
mod run;
mod shop;

pub use actions::{Action, ActionError, ParseError};
pub use card::mask::CardMask;
pub use card::{AnyCard, Card, CardLike, Kind, PlayingCard};
pub use data::sets::{CenterSet, HandSet, PlanetSet, VoucherSet};
pub use data::*;
pub use game::{Game, Phase, State};
pub use run::{InvalidStake, MAX_SEED_LEN, STAKES, Stake};
