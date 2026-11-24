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
pub mod balatro;
mod blind;
mod calculate_joker;
pub mod canon;
pub mod card;
mod card_lifecycle;
mod card_mask;
mod consumable_event;
mod consumables;
pub mod data;
mod data_gen;
pub mod dump;
mod enum_sets;
mod event;
mod event_queue;
mod flow;
pub mod game;
pub mod hand_eval;
mod hidden;
pub mod inject;
mod joker_lifecycle;
mod joker_scoring;
pub mod json;
pub mod num;
pub mod obs;
mod packs;
pub mod playing_card;
mod pools;
pub mod rng;
mod round;
mod round_end;
mod run;
mod scoring;
mod shop;
mod tags;

pub use actions::{Action, ActionError, ParseError};
pub use card::{AnyCard, Card, CardLike, Kind};
pub use card_mask::CardMask;
pub use data::*;
pub use enum_sets::{CenterSet, HandSet, PlanetSet, VoucherSet};
pub use game::{Game, Phase, State};
pub use playing_card::PlayingCard;
pub use run::{InvalidStake, MAX_SEED_LEN, STAKES, Stake};
