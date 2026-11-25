//! `VecEnv`: many auto-resetting games stepped in parallel. One env's
//! stepping, its reward shaping and the reservoirs of ante starts the
//! episodes restart from are in `slot`.

use numpy::{PyArray1, PyArray2, PyArray3, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use sim::data::Back;
use sim::obs::{self, OpOptions};
use sim::{Stake, num};

use crate::game::{Game, parse_back};
use crate::rows::{ObsArrays, ObsOptions, borrow, op_options, slice};
use crate::slot::{
	Cfg, Choices, End, Episode, PRACTICE_ANTES, Reservoirs, Reward, Slot, StepCtx, StepOut,
	make_game, run_number,
};

/// The probability that an ante start is offered to its reservoir.
const OFFER: f32 = 0.25;

/// An [`Episode`] as Python receives it: `(slot, seed, back, stake, ante,
/// won, blinds_cleared, steps, return, truncated, from_snapshot)`.
type EpisodeTuple = (
	usize,
	String,
	&'static str,
	u8,
	i32,
	bool,
	u32,
	u32,
	f32,
	bool,
	bool,
);

/// `step`'s result arrays: (reward, done, truncated).
type StepArrays<'py> = (
	Bound<'py, PyArray1<f32>>,
	Bound<'py, PyArray1<bool>>,
	Bound<'py, PyArray1<bool>>,
);

/// Parallel, auto-resetting environments.
///
/// The observation arrays (`tok_i`, `tok_f`, `glob_i`, `glob_f`, `ops`,
/// `ops_f`, `n_ops`, `n_tokens`, `n_hand`; one row per env) are rewritten
/// in place by `step`, `observe` and `set_game`; finished envs are reset
/// automatically. When an episode is truncated, the observation of its last
/// state is in the `final_*` arrays (same layout, valid for the truncated
/// rows of the last `step` only). `aux_kind` and `aux_dest` hold the
/// joker-order labels of the last `step`'s actions. Do not resize the arrays
/// or use them from other threads while a call is running.
#[pyclass(module = "antezero_sim")]
pub struct VecEnv {
	slots: Vec<Slot>,
	backs: Vec<Back>,
	stakes: Vec<Stake>,
	base_seed: u64,
	cfg: Cfg,
	pool: rayon::ThreadPool,
	episodes: Vec<Episode>,
	reservoirs: Reservoirs,
	/// Choices that were illegal and replaced by a fallback.
	invalid: u64,
	cur: ObsArrays,
	fin: ObsArrays,
	aux_kind: Py<PyArray1<i8>>,
	aux_dest: Py<PyArray2<u8>>,
}

/// Checks the sampling settings of a new `VecEnv`: the label rate and the
/// restart probabilities.
///
/// # Errors
///
/// Returns a `ValueError` if a share is outside [0, 1], the two restart
/// probabilities add up to more than 1, or late snapshots would start in the
/// practice antes.
fn check_rates(
	order_label_rate: f32, snapshot_prob: f32, snapshot_min_ante: i32, practice_prob: f32,
) -> PyResult<()> {
	let bad = |msg: String| Err(PyValueError::new_err(msg));
	if !(0.0..=1.0).contains(&snapshot_prob) || !(0.0..=1.0).contains(&order_label_rate) {
		return bad(format!(
			"snapshot_prob and order_label_rate must be in [0, 1], got {snapshot_prob} and {order_label_rate}"
		));
	}
	// Both are f32 thresholds: 0.8 + 0.2 may exceed 1 by a rounding step.
	if !(0.0..=1.0).contains(&practice_prob) || snapshot_prob + practice_prob > 1.0 + 1e-6 {
		return bad(format!(
			"practice_prob must be in [0, 1 - snapshot_prob], got {practice_prob}"
		));
	}
	if practice_prob > 0.0 && snapshot_prob > 0.0 && snapshot_min_ante <= *PRACTICE_ANTES.end() {
		return bad(format!(
			"practice_prob needs snapshot_min_ante above {}, got {snapshot_min_ante}",
			PRACTICE_ANTES.end()
		));
	}
	Ok(())
}

/// What `step_all` does to the envs before it writes their observations.
#[derive(Clone, Copy)]
enum Change<'a> {
	/// Nothing.
	Observe,
	/// Every env takes its choice.
	Step(Choices<'a>),
	/// The env `slot` starts over on a copy of `game`.
	SetGame { slot: usize, game: &'a Game },
}

/// Applies `change` to the envs and writes the observations and, of a step,
/// the labels. Every array is borrowed before any env changes, so a failed
/// borrow leaves the envs as they were.
fn step_all(env: &mut VecEnv, py: Python<'_>, change: Change<'_>) -> PyResult<Vec<StepOut>> {
	let n = env.slots.len();
	let mut cur = env.cur.borrow(py, n)?;
	let mut fin = env.fin.borrow(py, n)?;
	let mut aux_kind = borrow(&env.aux_kind, py, &[n])?;
	let mut aux_dest = borrow(&env.aux_dest, py, &[n, obs::ORDER_JOKERS])?;
	let rows = cur.rows()?;
	let final_rows = fin.rows()?;
	let (dests, _) = slice(&mut aux_dest)?.as_chunks_mut::<{ obs::ORDER_JOKERS }>();
	let labels = slice(&mut aux_kind)?.iter_mut().zip(dests);
	let cfg = env.cfg;
	let ctx = StepCtx {
		cfg,
		backs: &env.backs,
		stakes: &env.stakes,
		base_seed: env.base_seed,
		n_slots: num::index64(n),
		reservoirs: &env.reservoirs,
		p_offer: if cfg.snapshot_prob > 0.0 || cfg.practice_prob > 0.0 {
			OFFER
		} else {
			0.0
		},
	};
	let choices = match change {
		Change::Observe => None,
		Change::Step(choices) => Some(choices),
		Change::SetGame { slot, game } => {
			env.slots[slot].set_game(game.g.clone(), &cfg);
			None
		}
	};
	let slots = &mut env.slots;
	let pool = &env.pool;
	let mut out: Vec<StepOut> = py.detach(|| {
		pool.install(|| {
			slots
				.par_iter_mut()
				.zip(rows)
				.zip(final_rows)
				.enumerate()
				.map(|(i, ((s, mut row), mut final_row))| {
					let res = match choices {
						Some(c) => s.step(i, c.of(i), &ctx, &mut final_row),
						None => StepOut::default(),
					};
					s.observe(&mut row, cfg.obs);
					res
				})
				.collect()
		})
	});
	if choices.is_some() {
		for (r, (kind, dest)) in out.iter().zip(labels) {
			*kind = r
				.label
				.map_or(-1, |label| i8::from(label != [0; obs::ORDER_JOKERS]));
			*dest = r.label.unwrap_or_default();
		}
	}
	for r in &mut out {
		env.invalid += u64::from(r.invalid);
		if let Some(ep) = r.episode.take() {
			env.episodes.push(ep);
		}
		if let Some(snapshot) = r.snapshot.take() {
			env.reservoirs.offer(snapshot, env.base_seed);
		}
	}
	Ok(out)
}

#[pymethods]
impl VecEnv {
	/// `build_previews`: capacity previews in shops, packs and blind select
	/// (see `sim::obs::BuildPreview`). `order_gain`: the gain column of the
	/// joker-move ops (`sim::obs::order_gain`). `order_label_rate`: the share
	/// of played hands that get a joker-order label (see `aux_kind`).
	/// `practice_prob`: the share of episodes that restart from an ante start
	/// of antes 2-3 holding a gift (a joker or three levels of a hand type);
	/// late snapshots need `snapshot_min_ante` above those antes then.
	/// `practice_eternal`: a gifted joker is Eternal where the game allows the
	/// sticker, sellable otherwise. `null_sorts`: whether a hand sort that
	/// would change nothing is offered.
	///
	/// Episodes end when the run is lost, or won unless `endless`; they are
	/// truncated (not terminal: bootstrap from `final_*`) after `step_limit`
	/// steps or, in endless mode, beyond `max_ante`.
	#[expect(
		clippy::too_many_arguments,
		clippy::fn_params_excessive_bools,
		reason = "the Python constructor's keyword arguments"
	)]
	#[expect(
		clippy::needless_pass_by_value,
		reason = "pyo3 extracts `backs` as an owned `Vec<String>`, which is only read"
	)]
	#[new]
	#[pyo3(signature = (num_envs, backs = vec!["b_red".to_string()], stakes = vec![1], seed = 0, endless = false, max_ante = 16, step_limit = 5000, gamma = 0.999, r_win = 1.0, r_endless = 0.0, r_loss = 0.0, w_blind = 0.04, w_score = 0.02, threads = 0, play_candidates = 0, deck_tokens = false, snapshot_prob = 0.0, snapshot_min_ante = 2, allow_skip = true, build_previews = true, terminal_phi = true, order_gain = true, order_label_rate = 0.0, practice_prob = 0.0, practice_eternal = false, null_sorts = true))]
	fn new(
		py: Python<'_>, num_envs: usize, backs: Vec<String>, stakes: Vec<u8>, seed: u64,
		endless: bool, max_ante: i32, step_limit: u32, gamma: f32, r_win: f32, r_endless: f32,
		r_loss: f32, w_blind: f32, w_score: f32, threads: usize, play_candidates: usize,
		deck_tokens: bool, snapshot_prob: f32, snapshot_min_ante: i32, allow_skip: bool,
		build_previews: bool, terminal_phi: bool, order_gain: bool, order_label_rate: f32,
		practice_prob: f32, practice_eternal: bool, null_sorts: bool,
	) -> PyResult<VecEnv> {
		let bad = |msg: String| Err(PyValueError::new_err(msg));
		if num_envs == 0 {
			return bad("num_envs must be > 0".into());
		}
		let backs: Vec<Back> = backs
			.iter()
			.map(|b| parse_back(b))
			.collect::<PyResult<_>>()?;
		if backs.is_empty() || stakes.is_empty() {
			return bad("need at least one back and one stake".into());
		}
		let stakes: Vec<Stake> = stakes
			.into_iter()
			.map(Stake::try_from)
			.collect::<Result<_, _>>()
			.map_err(|e| PyValueError::new_err(e.to_string()))?;
		if step_limit == 0 || max_ante < 1 {
			return bad(format!(
				"step_limit and max_ante must be positive, got {step_limit} and {max_ante}"
			));
		}
		if !(gamma > 0.0 && gamma <= 1.0) {
			return bad(format!("gamma must be in (0, 1], got {gamma}"));
		}
		if ![r_win, r_endless, r_loss, w_blind, w_score]
			.iter()
			.all(|x| x.is_finite())
		{
			return bad("rewards and shaping weights must be finite".into());
		}
		check_rates(
			order_label_rate,
			snapshot_prob,
			snapshot_min_ante,
			practice_prob,
		)?;
		let reward = Reward {
			gamma,
			r_win,
			r_endless,
			r_loss,
			w_blind,
			w_score,
			terminal_phi,
		};
		let cfg = Cfg {
			endless,
			max_ante,
			step_limit,
			reward,
			obs: ObsOptions {
				ops: op_options(OpOptions {
					candidates: play_candidates,
					allow_skip,
					null_sorts,
				})?,
				deck_tokens,
				previews: build_previews,
				order_gain,
			},
			order_label_rate,
			snapshot_prob,
			snapshot_min_ante,
			practice_prob,
			practice_eternal,
		};
		let pool = rayon::ThreadPoolBuilder::new()
			.num_threads(threads)
			.build()
			.map_err(|e| PyValueError::new_err(e.to_string()))?;
		let n_slots = num::index64(num_envs);
		let slots: Vec<Slot> = py.detach(|| {
			pool.install(|| {
				(0..num_envs)
					.into_par_iter()
					.map(|i| {
						let g = make_game(run_number(seed, 0, n_slots, i), &backs, &stakes, &cfg);
						Slot::new(g, &reward)
					})
					.collect()
			})
		});
		let aux_shape = [num_envs, obs::ORDER_JOKERS];
		let mut env = VecEnv {
			slots,
			backs,
			stakes,
			base_seed: seed,
			cfg,
			pool,
			episodes: Vec::new(),
			reservoirs: Reservoirs::default(),
			invalid: 0,
			cur: ObsArrays::zeros(py, num_envs),
			fin: ObsArrays::zeros(py, num_envs),
			aux_kind: PyArray1::from_vec(py, vec![-1; num_envs]).unbind(),
			aux_dest: PyArray2::zeros(py, aux_shape, /* is_fortran= */ false).unbind(),
		};
		env.observe(py)?;
		Ok(env)
	}

	#[getter]
	fn num_envs(&self) -> usize {
		self.slots.len()
	}

	#[getter]
	fn gamma(&self) -> f32 {
		self.cfg.reward.gamma
	}

	/// Number of snapshots available for restarts.
	#[getter]
	fn reservoir_size(&self) -> usize {
		self.reservoirs.len()
	}

	/// Re-encodes every observation.
	fn observe(&mut self, py: Python<'_>) -> PyResult<()> {
		step_all(self, py, Change::Observe)?;
		Ok(())
	}

	/// Steps every env with `(op, cards, dest)`; returns `(reward, done,
	/// truncated)` arrays (`done` includes truncation). Rewards are shaped
	/// (see `Reward`). An illegal choice (counted by `invalid_actions`) is
	/// replaced by a legal fallback. Finished envs are reset (their new
	/// observation is already written); see `pop_episodes` and the
	/// `final_*` arrays.
	#[expect(
		clippy::needless_pass_by_value,
		reason = "pyo3 extracts a read-only array as an owned borrow guard"
	)]
	fn step<'py>(
		&mut self, py: Python<'py>, op: PyReadonlyArray1<'py, i32>,
		cards: PyReadonlyArray1<'py, u32>, dest: PyReadonlyArray1<'py, i32>,
	) -> PyResult<StepArrays<'py>> {
		let n = self.slots.len();
		let choices = Choices {
			op: op.as_slice()?,
			cards: cards.as_slice()?,
			dest: dest.as_slice()?,
		};
		if choices.op.len() != n || choices.cards.len() != n || choices.dest.len() != n {
			return Err(PyValueError::new_err(format!("expected {n} actions")));
		}
		let res = step_all(self, py, Change::Step(choices))?;
		let rew = PyArray1::from_vec(py, res.iter().map(|r| r.reward).collect());
		let done = PyArray1::from_vec(py, res.iter().map(|r| r.end.is_some()).collect());
		let truncated = res.iter().map(|r| r.end == Some(End::Truncated)).collect();
		Ok((rew, done, PyArray1::from_vec(py, truncated)))
	}

	/// Finished episodes since the last call: `(slot, seed, back, stake,
	/// ante, won, blinds_cleared, steps, return, truncated, from_snapshot)`.
	/// `won` is the run being won (not the game's `won` flag, which is also
	/// set when the final boss is lost); `return` is unshaped.
	fn pop_episodes(&mut self) -> Vec<EpisodeTuple> {
		let as_tuple = |e: Episode| {
			(
				e.slot,
				e.seed,
				e.back.key(),
				e.stake,
				e.ante,
				e.won,
				e.blinds_cleared,
				e.steps,
				e.ret,
				e.truncated,
				e.from_snapshot,
			)
		};
		self.episodes.drain(..).map(as_tuple).collect()
	}

	fn antes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
		PyArray1::from_vec(py, self.slots.iter().map(Slot::ante).collect())
	}

	/// Number of composite actions that were illegal and replaced by a
	/// fallback (should stay ~0 with correct masking).
	fn invalid_actions(&self) -> u64 {
		self.invalid
	}

	/// A snapshot of env `i` as a `Game` (with the env's `allow_skip` and
	/// `null_sorts`).
	fn game(&self, i: usize) -> PyResult<Game> {
		let s = self
			.slots
			.get(i)
			.ok_or_else(|| PyValueError::new_err("index out of range"))?;
		let mut game = Game::from(s.game_copy());
		game.allow_skip = self.cfg.obs.ops.allow_skip;
		game.null_sorts = self.cfg.obs.ops.null_sorts;
		Ok(game)
	}

	/// Replaces env `i`'s game (e.g. for evaluation on fixed seeds) with a
	/// copy of `game` that follows this env's `endless` and `allow_skip`
	/// settings. The game must not be over. If the call fails, the env keeps
	/// its game.
	fn set_game(&mut self, py: Python<'_>, i: usize, game: &Game) -> PyResult<()> {
		if game.g.is_terminal() {
			return Err(PyValueError::new_err(
				"set_game needs a game that is not over",
			));
		}
		if i >= self.slots.len() {
			return Err(PyValueError::new_err("index out of range"));
		}
		step_all(self, py, Change::SetGame { slot: i, game })?;
		Ok(())
	}

	#[getter]
	fn tok_i(&self, py: Python<'_>) -> Py<PyArray3<i32>> {
		self.cur.tok_i.clone_ref(py)
	}

	#[getter]
	fn tok_f(&self, py: Python<'_>) -> Py<PyArray3<f32>> {
		self.cur.tok_f.clone_ref(py)
	}

	#[getter]
	fn glob_i(&self, py: Python<'_>) -> Py<PyArray2<i32>> {
		self.cur.glob_i.clone_ref(py)
	}

	#[getter]
	fn glob_f(&self, py: Python<'_>) -> Py<PyArray2<f32>> {
		self.cur.glob_f.clone_ref(py)
	}

	#[getter]
	fn ops(&self, py: Python<'_>) -> Py<PyArray3<i32>> {
		self.cur.ops.clone_ref(py)
	}

	#[getter]
	fn ops_f(&self, py: Python<'_>) -> Py<PyArray3<f32>> {
		self.cur.ops_f.clone_ref(py)
	}

	#[getter]
	fn n_ops(&self, py: Python<'_>) -> Py<PyArray1<i32>> {
		self.cur.n_ops.clone_ref(py)
	}

	#[getter]
	fn n_tokens(&self, py: Python<'_>) -> Py<PyArray1<i32>> {
		self.cur.n_tokens.clone_ref(py)
	}

	#[getter]
	fn n_hand(&self, py: Python<'_>) -> Py<PyArray1<i32>> {
		self.cur.n_hand.clone_ref(py)
	}

	/// Per env, the joker-order label of the action of the last `step`
	/// (aligned with the observation it was chosen on, also when that step
	/// ended the episode): -1 = none (not a sampled play, or not a state
	/// `sim::obs::order_label` covers), 0 = no single joker move would have
	/// gained, 1 = one would (see `aux_dest`).
	#[getter]
	fn aux_kind(&self, py: Python<'_>) -> Py<PyArray1<i8>> {
		self.aux_kind.clone_ref(py)
	}

	/// `[num_envs, ORDER_JOKERS]`: for labels of kind 1, bit `to` of entry
	/// `from` is set when moving joker `from` to slot `to` first would have
	/// scored within 0.1% of the best single move.
	#[getter]
	fn aux_dest(&self, py: Python<'_>) -> Py<PyArray2<u8>> {
		self.aux_dest.clone_ref(py)
	}

	#[getter]
	fn final_tok_i(&self, py: Python<'_>) -> Py<PyArray3<i32>> {
		self.fin.tok_i.clone_ref(py)
	}

	#[getter]
	fn final_tok_f(&self, py: Python<'_>) -> Py<PyArray3<f32>> {
		self.fin.tok_f.clone_ref(py)
	}

	#[getter]
	fn final_glob_i(&self, py: Python<'_>) -> Py<PyArray2<i32>> {
		self.fin.glob_i.clone_ref(py)
	}

	#[getter]
	fn final_glob_f(&self, py: Python<'_>) -> Py<PyArray2<f32>> {
		self.fin.glob_f.clone_ref(py)
	}

	#[getter]
	fn final_ops(&self, py: Python<'_>) -> Py<PyArray3<i32>> {
		self.fin.ops.clone_ref(py)
	}

	#[getter]
	fn final_ops_f(&self, py: Python<'_>) -> Py<PyArray3<f32>> {
		self.fin.ops_f.clone_ref(py)
	}

	#[getter]
	fn final_n_ops(&self, py: Python<'_>) -> Py<PyArray1<i32>> {
		self.fin.n_ops.clone_ref(py)
	}

	#[getter]
	fn final_n_tokens(&self, py: Python<'_>) -> Py<PyArray1<i32>> {
		self.fin.n_tokens.clone_ref(py)
	}

	#[getter]
	fn final_n_hand(&self, py: Python<'_>) -> Py<PyArray1<i32>> {
		self.fin.n_hand.clone_ref(py)
	}
}
