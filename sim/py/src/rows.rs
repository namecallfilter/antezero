//! One state's encoded observation and legal ops, written into
//! caller-provided buffers: numpy rows (`VecEnv`) or owned vectors
//! (`Game.encode`), plus the batched numpy observation arrays shared by
//! `VecEnv`.

use numpy::ndarray::Dimension;
use numpy::{
	Element, PyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadwriteArray,
	PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
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
const fn row_len(row: [usize; 2]) -> usize {
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

/// One set of observation arrays, a row per env (see [`Rows`]).
pub(crate) struct ObsArrays {
	pub(crate) tok_i: Py<PyArray3<i32>>,
	pub(crate) tok_f: Py<PyArray3<f32>>,
	pub(crate) glob_i: Py<PyArray2<i32>>,
	pub(crate) glob_f: Py<PyArray2<f32>>,
	pub(crate) ops: Py<PyArray3<i32>>,
	pub(crate) ops_f: Py<PyArray3<f32>>,
	pub(crate) n_ops: Py<PyArray1<i32>>,
	pub(crate) n_tokens: Py<PyArray1<i32>>,
	pub(crate) n_hand: Py<PyArray1<i32>>,
}

/// The shape of `n` envs' rows of the shape `row`.
fn rows_of(n: usize, row: [usize; 2]) -> [usize; 3] {
	[n, row[0], row[1]]
}

/// Exclusive access to an array (they are exposed to Python, which could
/// also have reshaped them).
pub(crate) fn borrow<'py, T: Element, D: Dimension>(
	a: &Py<PyArray<T, D>>, py: Python<'py>, shape: &[usize],
) -> PyResult<PyReadwriteArray<'py, T, D>> {
	let a = a.bind(py);
	if a.shape() != shape {
		return Err(PyValueError::new_err(format!(
			"observation array has shape {:?}, expected {shape:?}",
			a.shape()
		)));
	}
	a.try_readwrite()
		.map_err(|e| PyValueError::new_err(format!("observation array is in use: {e}")))
}

pub(crate) fn slice<'a, T: Element, D: Dimension>(
	a: &'a mut PyReadwriteArray<'_, T, D>,
) -> PyResult<&'a mut [T]> {
	a.as_slice_mut()
		.map_err(|e| PyValueError::new_err(format!("observation array is not contiguous: {e}")))
}

pub(crate) struct ObsGuards<'py> {
	tok_i: PyReadwriteArray<'py, i32, numpy::Ix3>,
	tok_f: PyReadwriteArray<'py, f32, numpy::Ix3>,
	glob_i: PyReadwriteArray<'py, i32, numpy::Ix2>,
	glob_f: PyReadwriteArray<'py, f32, numpy::Ix2>,
	ops: PyReadwriteArray<'py, i32, numpy::Ix3>,
	ops_f: PyReadwriteArray<'py, f32, numpy::Ix3>,
	n_ops: PyReadwriteArray<'py, i32, numpy::Ix1>,
	n_tokens: PyReadwriteArray<'py, i32, numpy::Ix1>,
	n_hand: PyReadwriteArray<'py, i32, numpy::Ix1>,
}

impl ObsArrays {
	pub(crate) fn zeros(py: Python<'_>, n: usize) -> ObsArrays {
		ObsArrays {
			tok_i: PyArray3::zeros(py, rows_of(n, TOK_I_SHAPE), /* is_fortran= */ false).unbind(),
			tok_f: PyArray3::zeros(py, rows_of(n, TOK_F_SHAPE), /* is_fortran= */ false).unbind(),
			glob_i: PyArray2::zeros(py, [n, obs::GLOB_INT], /* is_fortran= */ false).unbind(),
			glob_f: PyArray2::zeros(py, [n, obs::GLOB_FLOAT], /* is_fortran= */ false).unbind(),
			ops: PyArray3::zeros(py, rows_of(n, OPS_SHAPE), /* is_fortran= */ false).unbind(),
			ops_f: PyArray3::zeros(py, rows_of(n, OPS_F_SHAPE), /* is_fortran= */ false).unbind(),
			n_ops: PyArray1::zeros(py, [n], /* is_fortran= */ false).unbind(),
			n_tokens: PyArray1::zeros(py, [n], /* is_fortran= */ false).unbind(),
			n_hand: PyArray1::zeros(py, [n], /* is_fortran= */ false).unbind(),
		}
	}

	pub(crate) fn borrow<'py>(&self, py: Python<'py>, n: usize) -> PyResult<ObsGuards<'py>> {
		Ok(ObsGuards {
			tok_i: borrow(&self.tok_i, py, &rows_of(n, TOK_I_SHAPE))?,
			tok_f: borrow(&self.tok_f, py, &rows_of(n, TOK_F_SHAPE))?,
			glob_i: borrow(&self.glob_i, py, &[n, obs::GLOB_INT])?,
			glob_f: borrow(&self.glob_f, py, &[n, obs::GLOB_FLOAT])?,
			ops: borrow(&self.ops, py, &rows_of(n, OPS_SHAPE))?,
			ops_f: borrow(&self.ops_f, py, &rows_of(n, OPS_F_SHAPE))?,
			n_ops: borrow(&self.n_ops, py, &[n])?,
			n_tokens: borrow(&self.n_tokens, py, &[n])?,
			n_hand: borrow(&self.n_hand, py, &[n])?,
		})
	}
}

impl ObsGuards<'_> {
	/// The rows of every env (the shapes were checked, so there is one per
	/// env).
	pub(crate) fn rows(&mut self) -> PyResult<impl IndexedParallelIterator<Item = Rows<'_>>> {
		let rows = slice(&mut self.tok_i)?
			.par_chunks_mut(row_len(TOK_I_SHAPE))
			.zip(slice(&mut self.tok_f)?.par_chunks_mut(row_len(TOK_F_SHAPE)))
			.zip(slice(&mut self.glob_i)?.par_chunks_mut(obs::GLOB_INT))
			.zip(slice(&mut self.glob_f)?.par_chunks_mut(obs::GLOB_FLOAT))
			.zip(slice(&mut self.ops)?.par_chunks_mut(row_len(OPS_SHAPE)))
			.zip(slice(&mut self.ops_f)?.par_chunks_mut(row_len(OPS_F_SHAPE)))
			.zip(slice(&mut self.n_ops)?.par_iter_mut())
			.zip(slice(&mut self.n_tokens)?.par_iter_mut())
			.zip(slice(&mut self.n_hand)?.par_iter_mut())
			.map(
				|(
					(((((((tok_i, tok_f), glob_i), glob_f), ops), ops_f), n_ops), n_tokens),
					n_hand,
				)| {
					Rows {
						obs: obs::ObsMut {
							tok_i,
							tok_f,
							glob_i,
							glob_f,
						},
						ops,
						ops_f,
						n_ops,
						n_tokens,
						n_hand,
					}
				},
			);
		Ok(rows)
	}
}
