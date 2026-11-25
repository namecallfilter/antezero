//! The agent-facing action interface: action definitions, legality checks,
//! execution, and text-code parsing.

mod check;
mod parse;
mod step;

pub use check::forced_mask;
pub(crate) use check::{check, legal_actions};
pub(crate) use parse::Operands;
pub use parse::{Action, ActionError, ParseError};
pub(crate) use step::{preview_hand, preview_play, step};
