//! `antezero_sim`: Python bindings for the simulator.
//!
//! Two interfaces: a text one (action / inject codes, canonical `str -> str`
//! state dicts), so histories and comparisons work the same across the
//! simulator, the headless oracle and the real-game bridge; and the numeric
//! learning one (`Game.encode` / `step_factored`, and `VecEnv`, which steps
//! many games in parallel), see `sim::obs`.

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod game;
mod interval;
mod pool;
mod rows;
mod vecenv;

use pyo3::prelude::*;
use sim::obs::{self, OpKind, Zone};

#[pymodule]
fn antezero_sim(m: &Bound<'_, PyModule>) -> PyResult<()> {
	m.add_class::<game::Game>()?;
	m.add_class::<vecenv::VecEnv>()?;
	m.add_function(wrap_pyfunction!(game::backs, m)?)?;
	m.add_function(wrap_pyfunction!(game::centers, m)?)?;
	m.add("MAX_TOKENS", obs::MAX_TOKENS)?;
	m.add("TOK_INT", obs::TOK_INT)?;
	m.add("TOK_FLOAT", obs::TOK_FLOAT)?;
	m.add("GLOB_INT", obs::GLOB_INT)?;
	m.add("GLOB_FLOAT", obs::GLOB_FLOAT)?;
	m.add("GLOB_FLOAT_V2", obs::GLOB_FLOAT_V2)?;
	m.add("MAX_OPS", obs::MAX_OPS)?;
	m.add("OP_INT", obs::OP_INT)?;
	m.add("OP_FLOAT", obs::OP_FLOAT)?;
	m.add("OP_ORDER_GAIN", obs::OP_ORDER_GAIN)?;
	m.add("ORDER_JOKERS", obs::ORDER_JOKERS)?;
	m.add("MAX_CANDIDATES", obs::MAX_CANDIDATES)?;
	m.add("TOK_VOCAB", obs::TOK_VOCAB.to_vec())?;
	m.add("GLOB_VOCAB", obs::GLOB_VOCAB.to_vec())?;
	m.add("OP_PLAY_TOP", OpKind::PlayTop.id())?;
	m.add("MAX_HAND", obs::MAX_HAND)?;
	m.add("N_ZONES", Zone::COUNT)?;
	m.add("N_CENTERS", obs::center::COUNT)?;
	m.add("N_OPS_KINDS", OpKind::COUNT)?;
	m.add("OP_PLAY", OpKind::Play.id())?;
	m.add("OP_DISCARD", OpKind::Discard.id())?;
	m.add("OP_MOVE_HAND", OpKind::MoveHand.id())?;
	m.add("OP_MOVE_JOKER", OpKind::MoveJoker.id())?;
	m.add("PREVIEW_BUDGET", obs::PREVIEW_BUDGET)?;
	m.add("PREVIEW_SAMPLES", obs::PREVIEW_SAMPLES)?;
	m.add("PREVIEW_HANDS", obs::PREVIEW_HANDS)?;
	m.add("PREVIEW_PLAYS", obs::PREVIEW_PLAYS)?;
	m.add("PREVIEW_ITEM_HANDS", obs::PREVIEW_ITEM_HANDS)?;
	m.add("TOK_PREVIEW", obs::TOK_PREVIEW)?;
	m.add("PREVIEW_TOK", obs::PREVIEW_TOK)?;
	Ok(())
}
