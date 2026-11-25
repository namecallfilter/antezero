//! The game's state dump (`H.state` of the oracle harness and `az_state` of
//! the mod, see `sim/oracle/lua/harness.lua`) as a simulator [`Game`], and a
//! simulator game as such a dump.
//!
//! A dump is taken at a decision point, where the simulator's event queue is
//! empty and the per-action flags and card marks are clear, so a game is the
//! dump's `G.GAME`, card areas and RNG streams. Two things the simulator
//! keeps are not the game's and are derived, [`Run::blinds_cleared`] and
//! [`Shop::voucher_seen`]; others only number or count what the simulator
//! does and start over: the tag ids, the RNG draws, the last previewed
//! hand's score and the free rearranges since the last other action (none,
//! the caller carries them).
//!
//! [`Game`]: crate::Game
//! [`Run::blinds_cleared`]: crate::game::Run::blinds_cleared
//! [`Shop::voucher_seen`]: crate::game::Shop::voucher_seen

mod parse;
mod read;
mod read_cards;
mod write;

pub use parse::DumpError;
pub(crate) use read::read;
pub(crate) use write::write;
