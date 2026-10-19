//! Headless, deterministic, bit-exact simulator of Balatro (v1.0.1o) game
//! logic, built for reinforcement learning and tree search.
//!
//! * [`Game`] holds the complete state, including every RNG stream, so a
//!   snapshot is just `game.clone()` and restoring is assignment.
//! * [`Game::step`] applies an [`Action`] and runs the game until the next
//!   decision point; [`Game::legal_actions`] enumerates valid actions.
//! * Randomness reproduces the game's LuaJIT `math.random` and `pseudoseed`
//!   streams exactly, so a seed plays out like in the real game: the oracle
//!   tests compare the full state (every RNG stream) and the order of RNG
//!   calls, draws and removals after every action, and the event frames,
//!   which match in about 99% of steps (see `sim/oracle`).
//!
//! # Layout
//!
//! * State: [`game`] (the state, card construction and costs, pools and
//!   seeded sampling), [`card`], [`data`] (enums and the tables generated
//!   from the game's prototypes), [`rng`].
//! * The game's logic, each module a port of the matching Lua: `run` (new
//!   runs, decks, stakes), `flow` (blind selection, rounds, play / discard),
//!   `scoring` (`evaluate_play`) with [`hand_eval`], `round_end` (end of
//!   round, cash out), `shop`, `packs`, `consumables`, `tags`, `blind` (boss
//!   blinds) and `jokers` (`Card:calculate_joker`). `events` emulates the
//!   event queue that sequences all of it frame by frame.
//! * Interfaces: [`actions`] (legality, `step`, text codes), [`obs`] (the
//!   learning observation and factored actions), [`canon`] (the comparison
//!   view shared with the oracle and the real-game bridge), [`inject`]
//!   (scenario injections); `search` adds determinization and play previews
//!   to [`Game`].

pub mod actions;
mod blind;
pub mod canon;
pub mod card;
mod consumables;
pub mod data;
mod data_gen;
mod events;
mod flow;
pub mod game;
pub mod hand_eval;
pub mod inject;
mod jokers;
pub mod obs;
mod packs;
pub mod rng;
mod round_end;
mod run;
mod scoring;
mod search;
mod shop;
mod tags;

pub use actions::{Action, ActionError, CardMask, ParseError};
pub use events::St;
pub use card::{Card, Kind};
pub use data::*;
pub use game::{Game, Phase};
pub use run::MAX_SEED_LEN;
