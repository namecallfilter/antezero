//! One state's encoded observation and legal ops, written into
//! caller-provided buffers: numpy rows (`VecEnv`) or owned vectors
//! (`Game.encode`).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use sim::game::Game;
use sim::num;
use sim::obs::{self, BuildPreview, ObsBuf, ObsMut, Op, OpOptions};

// The shape of one state's part of each two-dimensional observation array.
// Each of the other arrays holds `obs::GLOB_INT` or `obs::GLOB_FLOAT` values
// a state, or one value a state.
pub(crate) const TOK_I_SHAPE: [usize; 2] = [obs::MAX_TOKENS, obs::TOK_INT];
pub(crate) const TOK_F_SHAPE: [usize; 2] = [obs::MAX_TOKENS, obs::TOK_FLOAT];
pub(crate) const OPS_SHAPE: [usize; 2] = [obs::MAX_OPS, obs::OP_INT];
pub(crate) const OPS_F_SHAPE: [usize; 2] = [obs::MAX_OPS, obs::OP_FLOAT];

/// The number of values in a row of the shape `row`.
pub(crate) const fn row_len(row: [usize; 2]) -> usize {
	row[0] * row[1]
}

/// Mutable views of one state's observation and ops (row-major; see
/// `sim::obs`).
pub(crate) struct Rows<'a> {
	pub obs: ObsMut<'a>,
	pub ops: &'a mut [i32],
	pub ops_f: &'a mut [f32],
	pub n_ops: &'a mut i32,
	pub n_tokens: &'a mut i32,
	pub n_hand: &'a mut i32,
}

/// `options`, checked.
///
/// # Errors
///
/// Returns a `ValueError` if `candidates` is more than
/// `obs::MAX_CANDIDATES`.
pub(crate) fn op_options(options: OpOptions) -> PyResult<OpOptions> {
	if options.candidates > obs::MAX_CANDIDATES {
		return Err(PyValueError::new_err(format!(
			"play_candidates must be at most {}, got {}",
			obs::MAX_CANDIDATES,
			options.candidates
		)));
	}
	Ok(options)
}

/// What an observation holds besides its fixed part.
#[derive(Clone, Copy)]
pub(crate) struct ObsOptions {
	pub ops: OpOptions,
	pub deck_tokens: bool,
	/// Build previews of shops, packs and blind select.
	pub previews: bool,
	/// The gain column of the joker-move ops (`obs::order_gain`).
	pub order_gain: bool,
	/// Extended `obs_v2` global features (`GLOB_FLOAT_V2`).
	pub obs_v2: bool,
}

impl ObsOptions {
	pub(crate) fn glob_float(self) -> usize {
		if self.obs_v2 {
			obs::GLOB_FLOAT_V2
		} else {
			obs::GLOB_FLOAT
		}
	}
}

impl Rows<'_> {
	/// Encodes `g` (after refreshing the joker values the game recomputes
	/// every frame) and its legal ops; `op_list` receives the ops, for
	/// decoding the policy's choice. With `options.previews`, `previews`
	/// caches the env's recent build previews between steps (see
	/// `obs::build_preview`).
	pub fn write(
		&mut self, g: &mut Game, options: ObsOptions, op_list: &mut Vec<Op>,
		previews: &mut Vec<BuildPreview>,
	) {
		g.update_derived();
		let preview = if options.previews {
			obs::build_preview(g, previews)
		} else {
			None
		};
		let lay = obs::encode_with(g, &mut self.obs, options.deck_tokens, preview);
		obs::ops(g, &lay, options.ops, op_list);
		if options.order_gain {
			obs::order_gain(g, op_list);
		}
		self.ops.fill(0);
		self.ops_f.fill(0.0);
		for (j, op) in op_list.iter().enumerate() {
			self.ops[j * obs::OP_INT..(j + 1) * obs::OP_INT].copy_from_slice(&op.ints());
			self.ops_f[j * obs::OP_FLOAT..(j + 1) * obs::OP_FLOAT].copy_from_slice(&op.feats);
		}
		*self.n_ops = num::id(op_list.len());
		*self.n_tokens = num::id(lay.n_tokens);
		*self.n_hand = num::id(lay.hand);
	}
}

/// One state's observation and ops in owned buffers.
pub(crate) struct ObsRow {
	pub obs: ObsBuf,
	pub ops: Vec<i32>,
	pub ops_f: Vec<f32>,
	pub n_ops: i32,
	pub n_tokens: i32,
	pub n_hand: i32,
}

impl Default for ObsRow {
	fn default() -> ObsRow {
		ObsRow {
			obs: ObsBuf::new(),
			ops: vec![0; row_len(OPS_SHAPE)],
			ops_f: vec![0.0; row_len(OPS_F_SHAPE)],
			n_ops: 0,
			n_tokens: 0,
			n_hand: 0,
		}
	}
}

impl ObsRow {
	pub fn new(obs_v2: bool) -> ObsRow {
		ObsRow {
			obs: if obs_v2 {
				ObsBuf::new_v2()
			} else {
				ObsBuf::new()
			},
			..ObsRow::default()
		}
	}

	pub fn rows(&mut self) -> Rows<'_> {
		Rows {
			obs: self.obs.as_mut(),
			ops: &mut self.ops,
			ops_f: &mut self.ops_f,
			n_ops: &mut self.n_ops,
			n_tokens: &mut self.n_tokens,
			n_hand: &mut self.n_hand,
		}
	}
}
