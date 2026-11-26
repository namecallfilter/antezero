//! Python bindings for `sim::balatro`: `StartPool`, `BalatroCoreEnv`,
//! `BalatroCoreVecEnv` and the functions that describe their layout.

use std::sync::Arc;

use numpy::ndarray::Dimension;
use numpy::{
	Element, PyArray1, PyArray2, PyArrayDyn, PyArrayMethods, PyReadonlyArray, PyReadonlyArray1,
	PyReadonlyArray2, PyReadwriteArrayDyn, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use sim::balatro::{
	self, BalConfig, BalEnv, BalStepInfo, HS_CTX, MAX_PICKS, N_ACTIONS, N_SEL_VERBS, OBS_GROUPS,
	OBS_SIZE, SEL_F, StartPoint, load_starts,
};
use sim::data::{Back, HandType};
use sim::{MAX_SEED_LEN, Stake, num};

use crate::game::parse_back;
use crate::rows::{borrow, slice};

fn parse_deck(deck: &str, stake: u8) -> PyResult<(Back, Stake)> {
	let back = parse_back(deck)?;
	let stake = Stake::try_from(stake).map_err(|e| PyValueError::new_err(e.to_string()))?;
	Ok((back, stake))
}

#[expect(
	clippy::too_many_arguments,
	clippy::fn_params_excessive_bools,
	reason = "mirrors Python keyword options for BalatroEnv / BalatroVecEnv"
)]
fn make_config(
	deck: &str, stake: u8, max_steps: i32, reward_blind: f32, reward_win: f32,
	reward_win_log: bool, endless_ante: i32, reward_lose: f32, reward_lose_partial: bool,
	reward_money: bool, money_per_reward: f32, reward_chips: f32, reward_blue_seal: f32,
	reward_skip_deferred: bool, step_cost: f32, toggle_cost: f32, disable_softlocks: bool,
	obs_hide_face_down: bool, record: bool,
) -> PyResult<BalConfig> {
	let (back, stake) = parse_deck(deck, stake)?;
	if max_steps <= 0 {
		return Err(PyValueError::new_err(format!(
			"max_steps must be positive, got {max_steps}"
		)));
	}
	if reward_money && !(money_per_reward.is_finite() && money_per_reward > 0.0) {
		return Err(PyValueError::new_err(format!(
			"reward_money needs a positive, finite money_per_reward, got {money_per_reward}"
		)));
	}
	Ok(BalConfig {
		back,
		stake,
		max_steps,
		reward_blind,
		reward_win,
		reward_win_log,
		endless_ante,
		reward_lose,
		reward_lose_partial,
		reward_money,
		money_per_reward,
		reward_chips,
		reward_blue_seal,
		reward_skip_deferred,
		step_cost,
		toggle_cost,
		disable_softlocks,
		hide_face_down: obs_hide_face_down,
		record,
	})
}

/// The values of the input array `a`, named `name` in errors.
///
/// # Errors
///
/// Returns a `ValueError` if `a` does not have the shape `shape` or is not
/// contiguous.
fn input<'a, T: Element, D: Dimension>(
	a: &'a PyReadonlyArray<'_, T, D>, name: &str, shape: &[usize],
) -> PyResult<&'a [T]> {
	if a.shape() != shape {
		return Err(PyValueError::new_err(format!(
			"{name} has shape {:?}, expected {shape:?}",
			a.shape()
		)));
	}
	a.as_slice()
		.map_err(|e| PyValueError::new_err(format!("{name} is not contiguous: {e}")))
}

/// The five arrays an environment writes into (see
/// `BalEnv::write_all_obs`), borrowed for writing.
struct Outputs<'py> {
	obs: PyReadwriteArrayDyn<'py, f32>,
	sel: PyReadwriteArrayDyn<'py, f32>,
	mask: PyReadwriteArrayDyn<'py, u8>,
	specs: PyReadwriteArrayDyn<'py, i32>,
	hctx: PyReadwriteArrayDyn<'py, i32>,
}

impl<'py> Outputs<'py> {
	/// Borrows the arrays. Each must have the shape `lead` followed by its
	/// own: `lead` is empty for one environment and `[n]` for `n` of them.
	///
	/// # Errors
	///
	/// Returns a `ValueError` if an array has another shape or is in use.
	fn borrow(
		py: Python<'py>, lead: &[usize], obs: &Bound<'py, PyArrayDyn<f32>>,
		sel: &Bound<'py, PyArrayDyn<f32>>, mask: &Bound<'py, PyArrayDyn<u8>>,
		specs: &Bound<'py, PyArrayDyn<i32>>, hctx: &Bound<'py, PyArrayDyn<i32>>,
	) -> PyResult<Outputs<'py>> {
		let shape = |own: &[usize]| [lead, own].concat();
		Ok(Outputs {
			obs: borrow(obs.as_unbound(), py, &shape(&[OBS_SIZE]))?,
			sel: borrow(sel.as_unbound(), py, &shape(&[SEL_F]))?,
			mask: borrow(mask.as_unbound(), py, &shape(&[N_ACTIONS]))?,
			specs: borrow(specs.as_unbound(), py, &shape(&[N_SEL_VERBS, 3]))?,
			hctx: borrow(hctx.as_unbound(), py, &shape(&[HS_CTX]))?,
		})
	}

	/// # Errors
	///
	/// Returns a `ValueError` if an array is not contiguous.
	fn buffers(&mut self) -> PyResult<Buffers<'_>> {
		Ok(Buffers {
			obs: slice(&mut self.obs)?,
			sel: slice(&mut self.sel)?,
			mask: slice(&mut self.mask)?,
			specs: slice(&mut self.specs)?,
			hctx: slice(&mut self.hctx)?,
		})
	}
}

/// The values of the output arrays: all of them, or one environment's.
struct Buffers<'a> {
	obs: &'a mut [f32],
	sel: &'a mut [f32],
	mask: &'a mut [u8],
	specs: &'a mut [i32],
	hctx: &'a mut [i32],
}

impl<'a> Buffers<'a> {
	/// The buffers of each environment, for arrays whose shapes were checked
	/// to hold a row per environment.
	fn rows(self) -> impl IndexedParallelIterator<Item = Buffers<'a>> {
		self.obs
			.par_chunks_mut(OBS_SIZE)
			.zip(self.sel.par_chunks_mut(SEL_F))
			.zip(self.mask.par_chunks_mut(N_ACTIONS))
			.zip(self.specs.par_chunks_mut(N_SEL_VERBS * 3))
			.zip(self.hctx.par_chunks_mut(HS_CTX))
			.map(|((((obs, sel), mask), specs), hctx)| Buffers {
				obs,
				sel,
				mask,
				specs,
				hctx,
			})
	}

	fn write(self, env: &mut BalEnv) {
		env.write_all_obs(self.obs, self.sel, self.mask, self.specs, self.hctx);
	}
}

fn step_info_to_dict<'py>(py: Python<'py>, s: &BalStepInfo) -> PyResult<Bound<'py, PyDict>> {
	let d = PyDict::new(py);
	d.set_item("reward", s.reward)?;
	d.set_item("terminated", s.terminated != 0)?;
	d.set_item("truncated", s.truncated != 0)?;
	d.set_item("won", s.won != 0)?;
	d.set_item("illegal", s.illegal != 0)?;
	d.set_item("ante", s.ante)?;
	d.set_item("round", s.round)?;
	d.set_item("dollars", s.dollars)?;
	d.set_item("steps", s.steps)?;
	d.set_item("score", s.score)?;
	d.set_item("earned", s.earned)?;
	d.set_item("blinds", s.blinds)?;
	d.set_item("softlocks_avoided", s.softlocks_avoided)?;
	d.set_item("dead_end", s.dead_end != 0)?;
	d.set_item("hand_played", s.hand_played)?;
	d.set_item("blue_seals", s.blue_seals)?;
	d.set_item("played_cards", s.played_cards)?;
	d.set_item("played_scoring", s.played_scoring)?;
	d.set_item("blue_held", s.blue_held)?;
	d.set_item("blue_played", s.blue_played)?;
	d.set_item("hs_declared", s.hs_declared)?;
	d.set_item("hand_score", s.hand_score)?;
	Ok(d)
}

/// Start points for `BalatroCoreEnv.set_start` and
/// `BalatroCoreVecEnv.set_start`: games replayed from `.bal` scripts up to a
/// chosen action. Its ids count from 0 in the order the points were added.
#[pyclass(name = "StartPool", module = "antezero_sim")]
#[derive(Default)]
pub struct PyStartPool {
	pub(crate) pts: Vec<Arc<StartPoint>>,
	pub(crate) err: String,
}

#[pymethods]
impl PyStartPool {
	#[new]
	fn new() -> PyStartPool {
		PyStartPool::default()
	}

	/// The number of start points.
	fn __len__(&self) -> usize {
		self.pts.len()
	}

	/// Why the last `add` added no point, or `""` if its script replayed.
	#[getter]
	fn error(&self) -> &str {
		&self.err
	}

	/// Replays the `.bal` script and adds the game after each action count
	/// of `ks` as a start point. Returns their ids, `-1` where there is no
	/// start point: the run is over there, or the line is a skipped optional
	/// line. If the script does not replay through the largest count, no
	/// point is added, every id is `-1` and `error` says why.
	#[expect(
		clippy::needless_pass_by_value,
		reason = "PyO3 extracts a Python list argument as a Vec by value"
	)]
	fn add(&mut self, py: Python<'_>, script: &str, ks: Vec<i32>) -> PyResult<Vec<i32>> {
		let steps: Vec<usize> = ks
			.iter()
			.map(|&k| usize::try_from(k))
			.collect::<Result<_, _>>()
			.map_err(|_| PyValueError::new_err("ks must not hold a negative action count"))?;
		let mut ids = vec![-1i32; ks.len()];
		self.err.clear();
		match py.detach(|| load_starts(script, &steps)) {
			Ok(points) => {
				for (id, point) in ids.iter_mut().zip(points) {
					if let Some(mut point) = point {
						*id = num::id(self.pts.len());
						point.id = *id;
						self.pts.push(Arc::new(point));
					}
				}
			}
			Err(e) => self.err = e.to_string(),
		}
		Ok(ids)
	}

	/// The script that leads to start point `id`: the header line and the
	/// action lines that were applied. `""` for an id the pool does not
	/// hold.
	fn script(&self, id: i32) -> &str {
		usize::try_from(id)
			.ok()
			.and_then(|i| self.pts.get(i))
			.map_or("", |p| p.script.as_str())
	}

	/// The challenge of start point `id`. Always `""`: challenge runs are
	/// not simulated.
	fn challenge(&self, id: i32) -> &str {
		usize::try_from(id)
			.ok()
			.and_then(|i| self.pts.get(i))
			.map_or("", |p| p.challenge.as_str())
	}
}

/// What `set_start(pool, id, ...)` asks for: no start point (a new run) for
/// no pool or the id `-1`, and the pool's point `id` otherwise.
///
/// # Errors
///
/// Returns a `ValueError` for any other id the pool does not hold.
fn start_point(pool: Option<&PyStartPool>, id: i32) -> PyResult<Option<Arc<StartPoint>>> {
	let Some(pool) = pool.filter(|_| id != -1) else {
		return Ok(None);
	};
	let point = usize::try_from(id).ok().and_then(|i| pool.pts.get(i));
	match point {
		Some(point) => Ok(Some(Arc::clone(point))),
		None => Err(PyValueError::new_err(format!(
			"start id {id} is not in the pool, which holds {} points",
			pool.pts.len()
		))),
	}
}

/// One game of Balatro with `RLBalatro`'s lean actions and observation.
///
/// The 152 actions: 0 play, 1 discard, 2 select the blind, 3 skip it, 4
/// reroll the boss, 5 cash out (the env does that itself), 6 leave the shop,
/// 7 reroll it, 8..15 buy shop card `i`, 16..23 buy and use it, 24..27 redeem
/// voucher `i`, 28..31 open booster `i`, 32..39 take pack card `i`, 40 skip
/// the pack, 41..60 sell joker `i`, 61..70 sell consumable `i`, 71..80 use
/// consumable `i`, 140 reorder the jokers, and 141..151 play a declared
/// poker hand (`balatro_hands()` order, without High Card). Actions 81..139
/// are never legal: they are joker moves of `RLBalatro` that its lean
/// actions do not use, kept for the numbering.
///
/// 32 of the actions are selection verbs, which take cards: play, discard,
/// the ten uses, the eight pack picks, the reorder and the eleven declared
/// hands, numbered 0 to 31 in this order. `reset_into` and `step_into` write
/// five arrays, which must be C-contiguous and have these shapes and dtypes:
///
/// - `obs`, float32 `(8281,)`: the observation (`balatro_obs_layout`);
/// - `sel`, float32 `(44,)`: all 0 but entry 7, the chips the blind still
///   needs;
/// - `mask`, uint8 `(152,)`: 1 for each legal action (all 0 once the game is
///   over);
/// - `specs`, int32 `(32, 3)`: for each selection verb the fewest and the
///   most cards to pick (the most is `-1` if the verb is not legal or takes
///   no cards) and the bit set of the cards that can be picked first;
/// - `hctx`, int32 `(32,)`: the hand a policy computes the pick masks of
///   declared hands from.
///
/// Options: `deck` (a deck key such as `"b_red"`) and `stake` (1 to 8) of
/// new runs; `max_steps`, the steps after which an episode is truncated;
/// `reward_blind` for each blind beaten; `reward_win` for a won run, or with
/// `reward_win_log` the log10 of the chips of its last round; `endless_ante`
/// above 8 to play on after the win until that ante's boss is beaten;
/// `reward_lose` for a lost run, plus with `reward_lose_partial` the share of
/// the blind's chips that was scored; `reward_money` to pay the dollars
/// earned over `money_per_reward` in place of `reward_blind`; `reward_chips`
/// times the share of the blind's chips a played hand scores;
/// `reward_blue_seal` for each Blue Seal the deck gains;
/// `reward_skip_deferred` to pay a skipped blind when the next one is
/// beaten; `step_cost` taken from every step and `toggle_cost` from a joker
/// reorder; `disable_softlocks` to allow skipping a pack the game would be
/// stuck in; `obs_hide_face_down` to show nothing of face-down cards;
/// `record` to keep the game's script.
///
/// Raises `ValueError` for an unknown deck or stake, a `max_steps` below 1,
/// or `reward_money` with a `money_per_reward` that is not positive and
/// finite.
#[pyclass(name = "BalatroCoreEnv", module = "antezero_sim")]
pub struct PyBalatroEnv {
	env: BalEnv,
}

#[pymethods]
impl PyBalatroEnv {
	#[new]
	#[pyo3(signature = (
		deck = "b_red",
		stake = 1,
		max_steps = 5000,
		reward_blind = 1.0,
		reward_win = 10.0,
		reward_win_log = false,
		endless_ante = 0,
		reward_lose = 0.0,
		reward_lose_partial = true,
		reward_money = false,
		money_per_reward = 15.0,
		reward_chips = 0.0,
		reward_blue_seal = 0.0,
		reward_skip_deferred = false,
		step_cost = 0.0,
		toggle_cost = 0.0,
		disable_softlocks = false,
		obs_hide_face_down = false,
		record = false,
	))]
	#[expect(
		clippy::too_many_arguments,
		clippy::fn_params_excessive_bools,
		reason = "Python keyword constructor mirrors RLBalatro's BalatroEnv options"
	)]
	fn new(
		deck: &str, stake: u8, max_steps: i32, reward_blind: f32, reward_win: f32,
		reward_win_log: bool, endless_ante: i32, reward_lose: f32, reward_lose_partial: bool,
		reward_money: bool, money_per_reward: f32, reward_chips: f32, reward_blue_seal: f32,
		reward_skip_deferred: bool, step_cost: f32, toggle_cost: f32, disable_softlocks: bool,
		obs_hide_face_down: bool, record: bool,
	) -> PyResult<PyBalatroEnv> {
		let cfg = make_config(
			deck,
			stake,
			max_steps,
			reward_blind,
			reward_win,
			reward_win_log,
			endless_ante,
			reward_lose,
			reward_lose_partial,
			reward_money,
			money_per_reward,
			reward_chips,
			reward_blue_seal,
			reward_skip_deferred,
			step_cost,
			toggle_cost,
			disable_softlocks,
			obs_hide_face_down,
			record,
		)?;
		Ok(PyBalatroEnv {
			env: BalEnv::new(cfg),
		})
	}

	/// Starts a game and writes the five arrays: a new run with the seed
	/// `seed_str` if it is given, else with the seed derived from
	/// `seed_index` (index 1 if that is `None` too), or the start point set
	/// with `set_start`. Raises `ValueError`, and starts nothing, for a seed
	/// that a script header cannot hold (empty, longer than 32 bytes, or with
	/// whitespace or `#`) or an array of the wrong shape.
	#[expect(
		clippy::too_many_arguments,
		reason = "PyO3 method takes all 5 preallocated observation/mask arrays in one call"
	)]
	fn reset_into<'py>(
		&mut self, py: Python<'py>, seed_index: Option<u64>, seed_str: Option<&str>,
		obs: &Bound<'py, PyArrayDyn<f32>>, sel: &Bound<'py, PyArrayDyn<f32>>,
		mask: &Bound<'py, PyArrayDyn<u8>>, specs: &Bound<'py, PyArrayDyn<i32>>,
		hctx: &Bound<'py, PyArrayDyn<i32>>,
	) -> PyResult<()> {
		if seed_str.is_some_and(|seed| {
			seed.is_empty()
				|| seed.len() > MAX_SEED_LEN
				|| seed.contains(|c: char| c.is_whitespace() || c == '#')
		}) {
			return Err(PyValueError::new_err(format!(
				"a seed is 1 to {MAX_SEED_LEN} bytes without whitespace or '#'"
			)));
		}
		let mut out = Outputs::borrow(py, &[], obs, sel, mask, specs, hctx)?;
		let buffers = out.buffers()?;
		match seed_str {
			Some(seed) => self.env.reset(seed),
			None => self.env.reset_index(seed_index.unwrap_or(1)),
		}
		buffers.write(&mut self.env);
		Ok(())
	}

	/// Takes `action` with the cards `picks` and writes the five arrays.
	///
	/// A selection verb takes `picks`: hand positions in pick order (joker
	/// positions for the joker reorder), without the cards the boss forces
	/// into the selection. Other actions ignore `picks`. An action the mask
	/// does not allow, or cards it cannot take, only count as a step.
	///
	/// Returns a dict: `reward` (float); `terminated`, `truncated`, `won`,
	/// `illegal`, `dead_end` (bool); `ante`, `round`, `dollars` (negative in
	/// debt), `steps`, `blinds` (beaten this step), `softlocks_avoided`,
	/// `blue_seals` (in the deck) (int); `score` (on the last step, the
	/// chips of the last round) and `earned` (dollars this step) (float);
	/// and for a step that played a hand `hand_played` (the hand's position
	/// in `balatro_hands()`, else `-1`), `hs_declared` (the hand declared,
	/// 12 for action 0, else `-1`), `hand_score`, `played_cards`,
	/// `played_scoring`, `blue_held` and `blue_played`.
	///
	/// After a step that ends the game, call `reset_into`. Raises
	/// `ValueError`, and takes no step, for an array of the wrong shape.
	#[expect(
		clippy::too_many_arguments,
		clippy::needless_pass_by_value,
		reason = "PyO3 extracts picks Vec and all 5 preallocated observation arrays by value"
	)]
	fn step_into<'py>(
		&mut self, py: Python<'py>, action: i32, picks: Vec<i32>,
		obs: &Bound<'py, PyArrayDyn<f32>>, sel: &Bound<'py, PyArrayDyn<f32>>,
		mask: &Bound<'py, PyArrayDyn<u8>>, specs: &Bound<'py, PyArrayDyn<i32>>,
		hctx: &Bound<'py, PyArrayDyn<i32>>,
	) -> PyResult<Bound<'py, PyDict>> {
		let mut out = Outputs::borrow(py, &[], obs, sel, mask, specs, hctx)?;
		let buffers = out.buffers()?;
		let info = self.env.step_select(action, &picks);
		buffers.write(&mut self.env);
		step_info_to_dict(py, &info)
	}

	/// Sets the deck and the stake of the new runs that later resets start.
	/// Raises `ValueError`, and changes neither, if one of them is unknown.
	fn set_deck(&mut self, deck: &str, stake: u8) -> PyResult<()> {
		let (back, stake) = parse_deck(deck, stake)?;
		self.env.set_deck(back, stake);
		Ok(())
	}

	/// Makes later resets start from point `id` of `pool`, which keeps its
	/// own deck and stake. A `pool` of `None` or an `id` of `-1` goes back
	/// to new runs. Raises `ValueError` for any other id the pool does not
	/// hold.
	///
	/// With `reseed`, a reset restarts the game's random streams from its
	/// seed. Cards already dealt are not dealt again: a point inside a round
	/// keeps its hand and the order of its draw pile, so the cards drawn
	/// until the round ends are the same under every seed. Without it the
	/// game replays the randomness of the recorded run.
	fn set_start(&mut self, pool: Option<&PyStartPool>, id: i32, reseed: bool) -> PyResult<()> {
		self.env.set_start(start_point(pool, id)?, reseed);
		Ok(())
	}

	/// The id of the start point the game started from, `-1` for a new run.
	fn episode_start(&self) -> i32 {
		self.env.episode_start()
	}

	/// The `.bal` script of the game so far (`StartPool.add` replays it), or
	/// `""` if the env does not `record`.
	fn script(&self) -> &str {
		self.env.script()
	}
}

/// `num_envs` games stepped together on `threads` threads, each as a
/// `BalatroCoreEnv` with the same options. An env whose game ends starts
/// its next one in the same step.
///
/// The arrays of `reset_all` and `step_all` are the five of
/// `BalatroCoreEnv` with a leading axis of `num_envs`: `obs` float32
/// `(num_envs, 8281)`, `sel` float32 `(num_envs, 44)`, `masks` uint8
/// `(num_envs, 152)`, `specs` int32 `(num_envs, 32, 3)` and `hctx` int32
/// `(num_envs, 32)`, all C-contiguous.
///
/// Raises `ValueError` for what `BalatroCoreEnv` refuses and for a
/// `num_envs` of 0.
#[pyclass(name = "BalatroCoreVecEnv", module = "antezero_sim")]
pub struct PyBalatroVecEnv {
	envs: Vec<BalEnv>,
	pool: rayon::ThreadPool,
}

#[pymethods]
impl PyBalatroVecEnv {
	#[new]
	#[pyo3(signature = (
		num_envs,
		threads = 16,
		deck = "b_red",
		stake = 1,
		max_steps = 5000,
		reward_blind = 1.0,
		reward_win = 10.0,
		reward_win_log = false,
		endless_ante = 0,
		reward_lose = 0.0,
		reward_lose_partial = true,
		reward_money = false,
		money_per_reward = 15.0,
		reward_chips = 0.0,
		reward_blue_seal = 0.0,
		reward_skip_deferred = false,
		step_cost = 0.0,
		toggle_cost = 0.0,
		disable_softlocks = false,
		obs_hide_face_down = false,
		record = false,
	))]
	#[expect(
		clippy::too_many_arguments,
		clippy::fn_params_excessive_bools,
		reason = "Python keyword constructor mirrors RLBalatro's BalatroVecEnv options"
	)]
	fn new(
		num_envs: usize, threads: usize, deck: &str, stake: u8, max_steps: i32, reward_blind: f32,
		reward_win: f32, reward_win_log: bool, endless_ante: i32, reward_lose: f32,
		reward_lose_partial: bool, reward_money: bool, money_per_reward: f32, reward_chips: f32,
		reward_blue_seal: f32, reward_skip_deferred: bool, step_cost: f32, toggle_cost: f32,
		disable_softlocks: bool, obs_hide_face_down: bool, record: bool,
	) -> PyResult<PyBalatroVecEnv> {
		if num_envs == 0 {
			return Err(PyValueError::new_err("num_envs must be > 0"));
		}
		let cfg = make_config(
			deck,
			stake,
			max_steps,
			reward_blind,
			reward_win,
			reward_win_log,
			endless_ante,
			reward_lose,
			reward_lose_partial,
			reward_money,
			money_per_reward,
			reward_chips,
			reward_blue_seal,
			reward_skip_deferred,
			step_cost,
			toggle_cost,
			disable_softlocks,
			obs_hide_face_down,
			record,
		)?;
		let pool = rayon::ThreadPoolBuilder::new()
			.num_threads(threads.max(1))
			.build()
			.map_err(|e| PyValueError::new_err(format!("cannot build thread pool: {e}")))?;
		let envs = pool.install(|| {
			(0..num_envs)
				.into_par_iter()
				.map(|_| BalEnv::new(cfg.clone()))
				.collect()
		});
		Ok(PyBalatroVecEnv { envs, pool })
	}

	/// Starts a game in every env and writes the five arrays. `seeds` is a
	/// uint64 array `(num_envs,)` of seed indices, one for each env's new
	/// run. Raises `ValueError`, and starts nothing, for an array of the
	/// wrong shape.
	#[expect(
		clippy::too_many_arguments,
		clippy::needless_pass_by_value,
		reason = "PyO3 extracts numpy array views by value"
	)]
	fn reset_all<'py>(
		&mut self, py: Python<'py>, seeds: PyReadonlyArray1<'py, u64>,
		obs: &Bound<'py, PyArrayDyn<f32>>, sel: &Bound<'py, PyArrayDyn<f32>>,
		masks: &Bound<'py, PyArrayDyn<u8>>, specs: &Bound<'py, PyArrayDyn<i32>>,
		hctx: &Bound<'py, PyArrayDyn<i32>>,
	) -> PyResult<()> {
		let n = self.envs.len();
		let seeds = input(&seeds, "seeds", &[n])?;
		let mut out = Outputs::borrow(py, &[n], obs, sel, masks, specs, hctx)?;
		let rows = out.buffers()?.rows();
		let (pool, envs) = (&self.pool, &mut self.envs);
		py.detach(|| {
			pool.install(|| {
				envs.par_iter_mut()
					.zip(seeds)
					.zip(rows)
					.for_each(|((env, &seed), row)| {
						env.reset_index(seed);
						row.write(env);
					});
			});
		});
		Ok(())
	}

	/// Takes one step in every env and writes the five arrays.
	///
	/// `actions` is an int32 array `(num_envs,)`. `picks` is an int32 array
	/// `(num_envs, 5)`: each env's cards as for `BalatroCoreEnv.step_into`,
	/// ended by a negative entry if there are fewer than 5. `reset_seeds` is
	/// a uint64 array `(num_envs,)`: the seed index of the game an env
	/// starts if this step ends its game; its arrays then show the new game.
	///
	/// Returns a dict with the keys of `BalatroCoreEnv.step_into`, each an
	/// array `(num_envs,)`: float32 `reward` and `earned`, float64 `score`
	/// and `hand_score`, bool `terminated` and `truncated`, uint8 `won` and
	/// `illegal`, and int32 for the others.
	///
	/// Raises `ValueError`, and steps no env, for an array of the wrong
	/// shape.
	#[expect(
		clippy::too_many_arguments,
		clippy::needless_pass_by_value,
		reason = "PyO3 extracts numpy array views by value"
	)]
	fn step_all<'py>(
		&mut self, py: Python<'py>, actions: PyReadonlyArray1<'py, i32>,
		picks: PyReadonlyArray2<'py, i32>, reset_seeds: PyReadonlyArray1<'py, u64>,
		obs: &Bound<'py, PyArrayDyn<f32>>, sel: &Bound<'py, PyArrayDyn<f32>>,
		masks: &Bound<'py, PyArrayDyn<u8>>, specs: &Bound<'py, PyArrayDyn<i32>>,
		hctx: &Bound<'py, PyArrayDyn<i32>>,
	) -> PyResult<Bound<'py, PyDict>> {
		let n = self.envs.len();
		let actions = input(&actions, "actions", &[n])?;
		let picks = input(&picks, "picks", &[n, MAX_PICKS])?;
		let reset_seeds = input(&reset_seeds, "reset_seeds", &[n])?;
		let mut out = Outputs::borrow(py, &[n], obs, sel, masks, specs, hctx)?;
		let rows = out.buffers()?.rows();
		let (pool, envs) = (&self.pool, &mut self.envs);
		let infos: Vec<BalStepInfo> = py.detach(|| {
			pool.install(|| {
				envs.par_iter_mut()
					.zip(actions)
					.zip(picks.par_chunks(MAX_PICKS))
					.zip(reset_seeds)
					.zip(rows)
					.map(|((((env, &action), picks), &seed), row)| {
						let k = picks.iter().take_while(|&&c| c >= 0).count();
						let info = env.step_select(action, &picks[..k]);
						if info.terminated != 0 || info.truncated != 0 {
							env.reset_index(seed);
						}
						row.write(env);
						info
					})
					.collect()
			})
		});
		infos_vec_to_dict(py, &infos)
	}

	/// Sets the deck and the stake of the new runs env `i` starts from its
	/// next reset on. Raises `ValueError`, and changes nothing, for an
	/// unknown deck or stake or an `i` that is not an env.
	fn set_deck(&mut self, i: usize, deck: &str, stake: u8) -> PyResult<()> {
		let (back, stake) = parse_deck(deck, stake)?;
		self.envs
			.get_mut(i)
			.ok_or_else(|| PyValueError::new_err("index out of range"))?
			.set_deck(back, stake);
		Ok(())
	}

	/// `BalatroCoreEnv.set_start` for env `i`. Raises `ValueError` for an
	/// `i` that is not an env, too.
	fn set_start(
		&mut self, i: usize, pool: Option<&PyStartPool>, id: i32, reseed: bool,
	) -> PyResult<()> {
		let start = start_point(pool, id)?;
		self.envs
			.get_mut(i)
			.ok_or_else(|| PyValueError::new_err("index out of range"))?
			.set_start(start, reseed);
		Ok(())
	}

	/// The id of the start point the game of env `i` started from, `-1` for
	/// a new run. Raises `ValueError` for an `i` that is not an env.
	fn episode_start(&self, i: usize) -> PyResult<i32> {
		self.envs
			.get(i)
			.map(BalEnv::episode_start)
			.ok_or_else(|| PyValueError::new_err("index out of range"))
	}
}

fn infos_vec_to_dict<'py>(py: Python<'py>, infos: &[BalStepInfo]) -> PyResult<Bound<'py, PyDict>> {
	let d = PyDict::new(py);
	let f32_col = |f: fn(&BalStepInfo) -> f32| PyArray1::from_iter(py, infos.iter().map(f));
	let f64_col = |f: fn(&BalStepInfo) -> f64| PyArray1::from_iter(py, infos.iter().map(f));
	let i32_col = |f: fn(&BalStepInfo) -> i32| PyArray1::from_iter(py, infos.iter().map(f));
	let u8_col = |f: fn(&BalStepInfo) -> u8| PyArray1::from_iter(py, infos.iter().map(f));
	let bool_col = |f: fn(&BalStepInfo) -> bool| PyArray1::from_iter(py, infos.iter().map(f));

	d.set_item("reward", f32_col(|s| s.reward))?;
	d.set_item("terminated", bool_col(|s| s.terminated != 0))?;
	d.set_item("truncated", bool_col(|s| s.truncated != 0))?;
	d.set_item("won", u8_col(|s| s.won))?;
	d.set_item("illegal", u8_col(|s| s.illegal))?;
	d.set_item("ante", i32_col(|s| s.ante))?;
	d.set_item("round", i32_col(|s| s.round))?;
	d.set_item("dollars", i32_col(|s| s.dollars))?;
	d.set_item("steps", i32_col(|s| s.steps))?;
	d.set_item("score", f64_col(|s| s.score))?;
	d.set_item("earned", f32_col(|s| s.earned))?;
	d.set_item("blinds", i32_col(|s| s.blinds))?;
	d.set_item("softlocks_avoided", i32_col(|s| s.softlocks_avoided))?;
	d.set_item("dead_end", i32_col(|s| s.dead_end))?;
	d.set_item("hand_played", i32_col(|s| s.hand_played))?;
	d.set_item("blue_seals", i32_col(|s| s.blue_seals))?;
	d.set_item("played_cards", i32_col(|s| s.played_cards))?;
	d.set_item("played_scoring", i32_col(|s| s.played_scoring))?;
	d.set_item("blue_held", i32_col(|s| s.blue_held))?;
	d.set_item("blue_played", i32_col(|s| s.blue_played))?;
	d.set_item("hs_declared", i32_col(|s| s.hs_declared))?;
	d.set_item("hand_score", f64_col(|s| s.hand_score))?;
	Ok(d)
}

/// Which sets of ranks hold a straight, as an int32 array `(4, 8192)`. The
/// row is `2 * four_fingers + shortcut`, and the column is a set of ranks
/// with bit `r - 2` for rank id `r` (2 to 14, the Ace). An entry is 0 if the
/// set holds no straight; otherwise bit 15 is set and the low 13 bits are
/// the ranks of the straight.
#[pyfunction]
pub fn balatro_straight_lut(py: Python<'_>) -> PyResult<Bound<'_, PyArray2<i32>>> {
	let lut = balatro::straight_lut();
	let entries = lut.as_flattened().iter().map(|&entry| i32::from(entry));
	PyArray1::from_iter(py, entries).reshape([lut.len(), lut[0].len()])
}

/// The layout of the observation: `(groups, size)`, where `groups` lists
/// `(name, offset, rows, columns)` for each group in order (it holds
/// `rows * columns` floats from `offset` on) and `size` is the length of the
/// observation, 8281.
///
/// The groups `stickers`, `hand_target`, `bounty` and `target` belong to
/// training options of `RLBalatro` that are not implemented, and are always
/// 0.
#[pyfunction]
pub fn balatro_obs_layout() -> (Vec<(&'static str, usize, usize, usize)>, usize) {
	let mut groups = Vec::with_capacity(OBS_GROUPS.len());
	let mut off = 0usize;
	for &(name, r, c) in &OBS_GROUPS {
		groups.push((name, off, r, c));
		off += r * c;
	}
	(groups, off)
}

/// The keys of the 150 jokers in the order the observation numbers them
/// (alphabetical).
#[pyfunction]
pub fn balatro_joker_keys() -> Vec<&'static str> {
	balatro::jokers_by_key().iter().map(|j| j.key()).collect()
}

/// The keys of the 32 vouchers in the order of the owned-voucher flags of
/// the `round_ctx` group (alphabetical).
#[pyfunction]
pub fn balatro_voucher_keys() -> Vec<&'static str> {
	balatro::vouchers_by_key().iter().map(|v| v.key()).collect()
}

/// The names of the 12 poker hands, from "Flush Five" to "High Card": the
/// order of the `hand_levels` group and of the hands in a step's info.
#[pyfunction]
pub fn balatro_hands() -> Vec<&'static str> {
	HandType::ALL.iter().map(|h| h.name()).collect()
}
