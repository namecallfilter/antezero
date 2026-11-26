//! Python bindings for `sim::balatro` (`StartPool`, `BalatroCoreEnv`,
//! `BalatroCoreVecEnv`, `balatro_straight_lut`, `balatro_obs_layout`).

use std::sync::Arc;

use numpy::{
	PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1,
	PyReadwriteArray2, PyReadwriteArray3,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use sim::balatro::hand_eval::{HS_CTX, straight_lut};
use sim::balatro::ids::{N_JOKER_KEYS, N_VOUCHER_KEYS, tables};
use sim::balatro::obs::{OBS_GROUPS, OBS_SIZE, SEL_F};
use sim::balatro::{
	BalConfig, BalEnv, BalStepInfo, N_ACTIONS, N_SEL_VERBS, StartPoint, load_starts,
};
use sim::data::HandType;
use sim::{Stake, num};

use crate::game::parse_back;

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
	obs_hide_face_down: bool, sticker_penalty: f32, record: bool,
) -> PyResult<BalConfig> {
	let back = parse_back(deck)?;
	let stake = Stake::try_from(stake).map_err(|e| PyValueError::new_err(e.to_string()))?;
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
		sticker_penalty,
		joker_marks: [0; N_JOKER_KEYS],
		record,
	})
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
	d.set_item("stickered_held", s.stickered_held)?;
	d.set_item("stickered_lineup", s.stickered_lineup)?;
	d.set_item("hand_played", s.hand_played)?;
	d.set_item("hand_factor", s.hand_factor)?;
	d.set_item("blue_seals", s.blue_seals)?;
	d.set_item("bounty_drawn", s.bounty_drawn)?;
	d.set_item("bounty_deck", s.bounty_deck)?;
	d.set_item("target_hits", s.target_hits)?;
	d.set_item("target_held", s.target_held)?;
	d.set_item("played_cards", s.played_cards)?;
	d.set_item("played_scoring", s.played_scoring)?;
	d.set_item("blue_held", s.blue_held)?;
	d.set_item("blue_played", s.blue_played)?;
	d.set_item("hs_declared", s.hs_declared)?;
	d.set_item("hand_score", s.hand_score)?;
	Ok(d)
}

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

	fn __len__(&self) -> usize {
		self.pts.len()
	}

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

	fn script(&self, id: i32) -> &str {
		usize::try_from(id)
			.ok()
			.and_then(|i| self.pts.get(i))
			.map_or("", |p| p.script.as_str())
	}

	fn challenge(&self, id: i32) -> &str {
		usize::try_from(id)
			.ok()
			.and_then(|i| self.pts.get(i))
			.map_or("", |p| p.challenge.as_str())
	}
}

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
		sticker_penalty = 1.0,
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
		obs_hide_face_down: bool, sticker_penalty: f32, record: bool,
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
			sticker_penalty,
			record,
		)?;
		Ok(PyBalatroEnv {
			env: BalEnv::new(cfg),
		})
	}

	#[expect(
		clippy::too_many_arguments,
		reason = "PyO3 method takes all 5 preallocated observation/mask arrays in one call"
	)]
	fn reset_into(
		&mut self, seed_index: Option<u64>, seed_str: Option<&str>,
		mut obs: PyReadwriteArray1<'_, f32>, mut sel: PyReadwriteArray1<'_, f32>,
		mut mask: PyReadwriteArray1<'_, u8>, mut specs: PyReadwriteArray2<'_, i32>,
		mut hctx: PyReadwriteArray1<'_, i32>,
	) -> PyResult<()> {
		if let Some(s) = seed_str {
			self.env.reset(s);
		} else {
			self.env.reset_index(seed_index.unwrap_or(1));
		}
		self.env.write_all_obs(
			obs.as_slice_mut()?,
			sel.as_slice_mut()?,
			mask.as_slice_mut()?,
			specs.as_slice_mut()?,
			hctx.as_slice_mut()?,
		);
		Ok(())
	}

	#[expect(
		clippy::too_many_arguments,
		clippy::needless_pass_by_value,
		reason = "PyO3 extracts picks Vec and all 5 preallocated observation arrays by value"
	)]
	fn step_into<'py>(
		&mut self, py: Python<'py>, action: i32, picks: Vec<i32>,
		mut obs: PyReadwriteArray1<'py, f32>, mut sel: PyReadwriteArray1<'py, f32>,
		mut mask: PyReadwriteArray1<'py, u8>, mut specs: PyReadwriteArray2<'py, i32>,
		mut hctx: PyReadwriteArray1<'py, i32>,
	) -> PyResult<Bound<'py, PyDict>> {
		let info = self.env.step_select(action, &picks);
		self.env.write_all_obs(
			obs.as_slice_mut()?,
			sel.as_slice_mut()?,
			mask.as_slice_mut()?,
			specs.as_slice_mut()?,
			hctx.as_slice_mut()?,
		);
		step_info_to_dict(py, &info)
	}

	fn set_deck(&mut self, deck: &str, stake: u8) -> PyResult<()> {
		self.env.cfg.back = parse_back(deck)?;
		self.env.cfg.stake =
			Stake::try_from(stake).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(())
	}

	fn set_start(&mut self, pool: Option<&PyStartPool>, id: i32, reseed: bool) {
		self.env.start =
			pool.and_then(|p| usize::try_from(id).ok().and_then(|i| p.pts.get(i).cloned()));
		self.env.start_reseed = reseed;
	}

	fn episode_start(&self) -> i32 {
		self.env.episode_start
	}

	fn script(&self) -> &str {
		&self.env.script
	}
}

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
		sticker_penalty = 1.0,
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
		disable_softlocks: bool, obs_hide_face_down: bool, sticker_penalty: f32, record: bool,
	) -> PyResult<PyBalatroVecEnv> {
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
			sticker_penalty,
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

	#[expect(
		clippy::too_many_arguments,
		clippy::needless_pass_by_value,
		reason = "PyO3 extracts numpy array views by value"
	)]
	fn reset_all(
		&mut self, py: Python<'_>, seeds: PyReadonlyArray1<'_, u64>,
		mut obs: PyReadwriteArray2<'_, f32>, mut sel: PyReadwriteArray2<'_, f32>,
		mut masks: PyReadwriteArray2<'_, u8>, mut specs: PyReadwriteArray3<'_, i32>,
		mut hctx: PyReadwriteArray2<'_, i32>,
	) -> PyResult<()> {
		let seeds_s = seeds.as_slice()?;
		let obs_s = obs.as_slice_mut()?;
		let sel_s = sel.as_slice_mut()?;
		let masks_s = masks.as_slice_mut()?;
		let specs_s = specs.as_slice_mut()?;
		let hctx_s = hctx.as_slice_mut()?;
		let pool = &self.pool;
		let envs = &mut self.envs;
		py.detach(|| {
			pool.install(|| {
				envs.par_iter_mut()
					.zip(seeds_s.par_iter())
					.zip(obs_s.par_chunks_mut(OBS_SIZE))
					.zip(sel_s.par_chunks_mut(SEL_F))
					.zip(masks_s.par_chunks_mut(N_ACTIONS))
					.zip(specs_s.par_chunks_mut(N_SEL_VERBS * 3))
					.zip(hctx_s.par_chunks_mut(HS_CTX))
					.for_each(|((((((env, &seed), o), s), m), sp), hc)| {
						env.reset_index(seed);
						env.write_all_obs(o, s, m, sp, hc);
					});
			});
		});
		Ok(())
	}

	#[expect(
		clippy::too_many_arguments,
		clippy::needless_pass_by_value,
		reason = "PyO3 extracts numpy array views by value"
	)]
	fn step_all<'py>(
		&mut self, py: Python<'py>, actions: PyReadonlyArray1<'py, i32>,
		picks: PyReadonlyArray2<'py, i32>, reset_seeds: PyReadonlyArray1<'py, u64>,
		mut obs: PyReadwriteArray2<'py, f32>, mut sel: PyReadwriteArray2<'py, f32>,
		mut masks: PyReadwriteArray2<'py, u8>, mut specs: PyReadwriteArray3<'py, i32>,
		mut hctx: PyReadwriteArray2<'py, i32>,
	) -> PyResult<Bound<'py, PyDict>> {
		let acts_s = actions.as_slice()?;
		let picks_s = picks.as_slice()?;
		let rseeds_s = reset_seeds.as_slice()?;
		let obs_s = obs.as_slice_mut()?;
		let sel_s = sel.as_slice_mut()?;
		let masks_s = masks.as_slice_mut()?;
		let specs_s = specs.as_slice_mut()?;
		let hctx_s = hctx.as_slice_mut()?;
		let pool = &self.pool;
		let envs = &mut self.envs;
		let infos: Vec<BalStepInfo> = py.detach(|| {
			pool.install(|| {
				envs.par_iter_mut()
					.zip(acts_s.par_iter())
					.zip(picks_s.par_chunks(5))
					.zip(rseeds_s.par_iter())
					.zip(obs_s.par_chunks_mut(OBS_SIZE))
					.zip(sel_s.par_chunks_mut(SEL_F))
					.zip(masks_s.par_chunks_mut(N_ACTIONS))
					.zip(specs_s.par_chunks_mut(N_SEL_VERBS * 3))
					.zip(hctx_s.par_chunks_mut(HS_CTX))
					.map(|((((((((env, &act), pk), &rseed), o), s), m), sp), hc)| {
						let k = pk.iter().take_while(|&&c| c >= 0).count();
						let info = env.step_select(act, &pk[..k]);
						if info.terminated != 0 || info.truncated != 0 {
							env.reset_index(rseed);
						}
						env.write_all_obs(o, s, m, sp, hc);
						info
					})
					.collect()
			})
		});
		infos_vec_to_dict(py, &infos)
	}

	fn set_deck(&mut self, i: usize, deck: &str, stake: u8) -> PyResult<()> {
		let back = parse_back(deck)?;
		let stake = Stake::try_from(stake).map_err(|e| PyValueError::new_err(e.to_string()))?;
		if let Some(env) = self.envs.get_mut(i) {
			env.cfg.back = back;
			env.cfg.stake = stake;
		}
		Ok(())
	}

	fn set_start(&mut self, i: usize, pool: Option<&PyStartPool>, id: i32, reseed: bool) {
		if let Some(env) = self.envs.get_mut(i) {
			env.start = pool.and_then(|p| {
				usize::try_from(id)
					.ok()
					.and_then(|idx| p.pts.get(idx).cloned())
			});
			env.start_reseed = reseed;
		}
	}

	fn episode_start(&self, i: usize) -> i32 {
		self.envs.get(i).map_or(-1, |e| e.episode_start)
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
	d.set_item("stickered_held", i32_col(|s| s.stickered_held))?;
	d.set_item("stickered_lineup", i32_col(|s| s.stickered_lineup))?;
	d.set_item("hand_played", i32_col(|s| s.hand_played))?;
	d.set_item("hand_factor", f32_col(|s| s.hand_factor))?;
	d.set_item("blue_seals", i32_col(|s| s.blue_seals))?;
	d.set_item("bounty_drawn", i32_col(|s| s.bounty_drawn))?;
	d.set_item("bounty_deck", i32_col(|s| s.bounty_deck))?;
	d.set_item("target_hits", i32_col(|s| s.target_hits))?;
	d.set_item("target_held", i32_col(|s| s.target_held))?;
	d.set_item("played_cards", i32_col(|s| s.played_cards))?;
	d.set_item("played_scoring", i32_col(|s| s.played_scoring))?;
	d.set_item("blue_held", i32_col(|s| s.blue_held))?;
	d.set_item("blue_played", i32_col(|s| s.blue_played))?;
	d.set_item("hs_declared", i32_col(|s| s.hs_declared))?;
	d.set_item("hand_score", f64_col(|s| s.hand_score))?;
	Ok(d)
}

#[pyfunction]
pub fn balatro_straight_lut(py: Python<'_>) -> PyResult<Bound<'_, PyArray2<i32>>> {
	let lut = straight_lut();
	let mut flat = Vec::with_capacity(4 * 8192);
	for row in lut.run.iter().take(4) {
		for &val in row {
			flat.push(i32::from(val));
		}
	}
	PyArray1::from_vec(py, flat).reshape([4, 8192])
}

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

#[pyfunction]
pub fn balatro_joker_keys() -> Vec<&'static str> {
	tables().joker_by_alpha.iter().map(|j| j.key()).collect()
}

#[pyfunction]
pub fn balatro_voucher_keys() -> Vec<&'static str> {
	let t = tables();
	(0..N_VOUCHER_KEYS)
		.map(|i| t.voucher_by_alpha[i].key())
		.collect()
}

#[pyfunction]
pub fn balatro_hands() -> Vec<&'static str> {
	HandType::ALL.iter().map(|h| h.name()).collect()
}
