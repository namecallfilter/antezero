//! `VecEnv`: many simulator games stepped in parallel (rayon), writing
//! observations straight into preallocated numpy arrays.

use numpy::ndarray::Dimension;
use numpy::{
	Element, PyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1,
	PyReadwriteArray, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use sim::data::Back;
use sim::game::{BlindStatus, Game as SimGame, Phase};
use sim::num;
use sim::obs::{self, BuildPreview, Op};

use crate::rows::{ObsRow, Rows, check_candidates};

const SEED_CHARS: &[u8] = b"123456789ABCDEFGHIJKLMNPQRSTUVWXYZ";

/// Seed string `n` in the game's seed alphabet (8 characters).
fn seed_string(mut n: u64) -> String {
	// Spread consecutive indices over the seed space.
	n = n.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (n >> 29);
	(0..8)
		.map(|_| {
			let c = SEED_CHARS[num::pick(n, SEED_CHARS.len())] as char;
			n /= SEED_CHARS.len() as u64;
			c
		})
		.collect()
}

fn mix(n: u64) -> u64 {
	let mut z = n.wrapping_add(0x9e37_79b9_7f4a_7c15);
	z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
	z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
	z ^ (z >> 31)
}

/// Uniform in [0, 1) from a hash.
#[expect(
	clippy::cast_precision_loss,
	reason = "both operands have at most 25 bits, which an f32 holds exactly"
)]
fn unit(n: u64) -> f32 {
	(mix(n) >> 40) as f32 / (1u64 << 24) as f32
}

/// The shaping potential `w_blind * progress + w_score * fraction of the
/// current blind's chips scored`, with progress = the run position in blinds
/// (3 per finished ante plus this ante's defeated blinds). Counting blinds
/// cleared instead paid for skipping blinds and for Hieroglyph/Petroglyph
/// (which add an ante of blinds to beat), and policies learned both. The
/// lost blind's fraction is kept at game over: if losing dropped it,
/// delaying a loss would pay (stalling).
pub(crate) fn potential(g: &SimGame, w_blind: f32, w_score: f32) -> f32 {
	let in_round = matches!(g.phase, Phase::Round | Phase::GameOver);
	let round = if in_round && g.blind.chips > 0.0 {
		num::feat((g.chips / g.blind.chips).min(1.0))
	} else {
		0.0
	};
	let defeated = num::id(
		g.blind_states
			.iter()
			.filter(|&&s| s == BlindStatus::Defeated)
			.count(),
	);
	let progress = 3 * (g.ante - 1).max(0) + defeated;
	w_blind * num::feat(f64::from(progress)) + w_score * round
}

/// The reward. The true objective is `r_win` for winning the run (plus
/// `r_endless` per blind cleared after the win, for endless training, and
/// `-r_loss` for losing). Progress is rewarded through the shaping
/// `F = gamma * Phi(s') - Phi(s)` ([`potential`]). `Phi` is *not* zeroed at
/// terminal states: with `Phi(terminal) = 0` a lost game would hand back all
/// its progress, every loss would return the same, and a policy that never
/// wins would get no learning signal. So the episode return is the true
/// return plus `gamma^T * Phi(s_T)`, the (discounted) blinds cleared. With
/// `terminal_phi = false` Phi *is* zeroed there (classic shaping: exactly P(win)).
#[derive(Clone, Copy, Debug)]
struct Reward {
	gamma: f32,
	r_win: f32,
	r_endless: f32,
	r_loss: f32,
	w_blind: f32,
	w_score: f32,
	terminal_phi: bool,
}

impl Reward {
	fn potential(&self, g: &SimGame) -> f32 {
		potential(g, self.w_blind, self.w_score)
	}

	/// The true (unshaped) reward of a transition.
	fn true_reward(&self, blinds_before: u32, won_before: bool, g: &SimGame) -> f32 {
		let mut r = 0.0;
		if g.run_won() && !won_before {
			r += self.r_win;
		} else if won_before {
			r += self.r_endless
				* num::feat(f64::from(g.blinds_cleared.saturating_sub(blinds_before)));
		}
		if g.phase == Phase::GameOver || g.is_stuck() {
			r -= self.r_loss;
		}
		r
	}
}

#[derive(Clone, Copy)]
#[expect(
	clippy::struct_excessive_bools,
	reason = "independent switches of the Python constructor"
)]
struct Cfg {
	endless: bool,
	max_ante: i32,
	step_limit: u32,
	reward: Reward,
	candidates: usize,
	deck_tokens: bool,
	previews: bool,
	snapshot_prob: f32,
	snapshot_min_ante: i32,
	allow_skip: bool,
}

/// Snapshots kept for restarts. Ante starts are offered with probability
/// `OFFER` and replace a random entry once full, so the pool follows the
/// current policy's state distribution.
const RESERVOIR: usize = 4096;
const OFFER: f32 = 0.25;

struct Slot {
	g: SimGame,
	/// The ops of the current observation (to decode the next choice).
	ops: Vec<Op>,
	steps: u32,
	ret: f32,
	phi: f32,
	resets: u64,
	invalid: u64,
	from_snapshot: bool,
	/// The observation of the last state before an auto-reset, if the
	/// episode was truncated (to bootstrap from).
	fin: Option<ObsRow>,
	/// Recent build previews (reused while the build is unchanged).
	pv: Vec<BuildPreview>,
}

/// What the step of every env shares.
struct StepCtx<'a> {
	cfg: Cfg,
	backs: &'a [Back],
	stakes: &'a [u8],
	base_seed: u64,
	n_slots: u64,
	reservoir: &'a [SimGame],
	/// Probability of offering an ante start to the reservoir.
	p_offer: f32,
}

impl Slot {
	/// Applies the choice `(op, cards, dest)` of env `i`. If the episode ended, the env is reset; a truncated
	/// episode's last observation is written to `row` first and kept in `fin`.
	fn step(
		&mut self, i: usize, (op, cards, dest): (i32, u32, i32), ctx: &StepCtx<'_>,
		row: &mut Rows<'_>,
	) -> StepOut {
		let (cfg, rw) = (ctx.cfg, ctx.cfg.reward);
		let mut res = StepOut::default();
		let blinds_before = self.g.blinds_cleared;
		let won_before = self.g.run_won();
		// A negative destination is out of range (it only
		// matters for reordering ops).
		let to = u32::try_from(dest).unwrap_or(u32::MAX);
		let a = usize::try_from(op)
			.ok()
			.and_then(|k| self.ops.get(k))
			.and_then(|o| obs::decode(&self.g, o, cards, to));
		let a = if let Some(a) = a {
			Some(a)
		} else {
			self.invalid += 1;
			obs::fallback(&self.g, &self.ops)
		};
		// The known no-action state (the game's own pack softlock) is terminal via
		// `is_stuck`; any other one is a simulator bug: dump it and truncate the
		// episode rather than kill a long training run.
		let dead_end = a.is_none();
		match a {
			Some(a) => {
				self.g.step(a).expect("decoded action rejected");
				self.steps += 1;
			}
			None => eprintln!(
				"[vecenv] dead end: no legal action, episode truncated. ops: {:?}\n{:#?}",
				self.ops, self.g
			),
		}
		let g = &self.g;
		let r_true = rw.true_reward(blinds_before, won_before, g);
		let terminal = g.is_terminal();
		let trunc = !terminal
			&& (dead_end || self.steps >= cfg.step_limit || (cfg.endless && g.ante > cfg.max_ante));
		let phi = if terminal && !rw.terminal_phi {
			0.0
		} else {
			rw.potential(g)
		};
		res.reward = r_true + rw.gamma * phi - self.phi;
		self.phi = phi;
		self.ret += r_true;
		res.done = terminal || trunc;
		res.truncated = trunc;
		if !terminal
			&& !trunc
			&& ctx.p_offer > 0.0
			&& snapshot_point(g, &cfg)
			&& unit(mix(ctx.base_seed ^ i as u64) ^ (self.resets << 20) ^ u64::from(self.steps))
				< ctx.p_offer
		{
			res.snapshot = Some(Box::new(g.clone()));
		}
		if terminal || trunc {
			res.episode = Some(Episode {
				slot: i,
				seed: g.rng.seed().to_string(),
				back: g.back.key().to_string(),
				stake: g.stake,
				ante: g.ante,
				won: g.run_won(),
				blinds_cleared: g.blinds_cleared,
				steps: self.steps,
				ret: self.ret,
				truncated: trunc,
				from_snapshot: self.from_snapshot,
			});
			if trunc {
				row.write(
					&mut self.g,
					cfg.candidates,
					cfg.deck_tokens,
					cfg.previews,
					&mut self.ops,
					&mut self.pv,
				);
				self.fin = Some(ObsRow::of(row));
			}
			self.resets += 1;
			let run = ctx
				.base_seed
				.wrapping_mul(1_000_003)
				.wrapping_add(self.resets * ctx.n_slots + i as u64);
			let (g, snap) = start(run, ctx.backs, ctx.stakes, cfg, ctx.reservoir);
			self.g = g;
			self.from_snapshot = snap;
			self.steps = 0;
			self.ret = 0.0;
			self.phi = rw.potential(&self.g);
		}
		res
	}
}

/// A finished episode.
struct Episode {
	/// The env slot it ran in.
	slot: usize,
	seed: String,
	back: String,
	stake: u8,
	ante: i32,
	/// The run was won (not the game's `won` flag, which is also set when
	/// the final boss is lost).
	won: bool,
	blinds_cleared: u32,
	steps: u32,
	/// The true (unshaped) return.
	ret: f32,
	truncated: bool,
	from_snapshot: bool,
}

/// An [`Episode`] as Python receives it: `(slot, seed, back, stake, ante,
/// won, blinds_cleared, steps, return, truncated, from_snapshot)`.
type EpisodeTuple = (
	usize,
	String,
	String,
	u8,
	i32,
	bool,
	u32,
	u32,
	f32,
	bool,
	bool,
);

/// Per-slot result of a step.
#[derive(Default)]
struct StepOut {
	/// The shaped reward.
	reward: f32,
	/// The episode ended (terminal or truncated).
	done: bool,
	truncated: bool,
	/// The finished episode.
	episode: Option<Episode>,
	/// A snapshot to offer the reservoir.
	snapshot: Option<Box<SimGame>>,
}

/// `step`'s result arrays: (reward, done, truncated).
type StepArrays<'py> = (
	Bound<'py, PyArray1<f32>>,
	Bound<'py, PyArray1<bool>>,
	Bound<'py, PyArray1<bool>>,
);

/// One step's choices of every env: (op index, card mask, destination).
type Choices<'a> = (&'a [i32], &'a [u32], &'a [i32]);

/// One set of observation arrays, a row per env (see `Rows`).
struct ObsArrays {
	tok_i: Py<PyArray3<i32>>,
	tok_f: Py<PyArray3<f32>>,
	glob_i: Py<PyArray2<i32>>,
	glob_f: Py<PyArray2<f32>>,
	ops: Py<PyArray3<i32>>,
	ops_f: Py<PyArray3<f32>>,
	n_ops: Py<PyArray1<i32>>,
	n_tokens: Py<PyArray1<i32>>,
	n_hand: Py<PyArray1<i32>>,
}

/// Exclusive access to an array (they are exposed to Python, which could
/// also have reshaped them).
fn borrow<'py, T: Element, D: Dimension>(
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

fn slice<'a, T: Element, D: Dimension>(
	a: &'a mut PyReadwriteArray<'_, T, D>,
) -> PyResult<&'a mut [T]> {
	a.as_slice_mut()
		.map_err(|e| PyValueError::new_err(format!("observation array is not contiguous: {e}")))
}

/// Exclusive borrows of an `ObsArrays`.
struct ObsGuards<'py> {
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
	fn zeros(py: Python<'_>, n: usize) -> ObsArrays {
		ObsArrays {
			tok_i: PyArray3::zeros(py, [n, obs::MAX_TOKENS, obs::TOK_INT], false).unbind(),
			tok_f: PyArray3::zeros(py, [n, obs::MAX_TOKENS, obs::TOK_FLOAT], false).unbind(),
			glob_i: PyArray2::zeros(py, [n, obs::GLOB_INT], false).unbind(),
			glob_f: PyArray2::zeros(py, [n, obs::GLOB_FLOAT], false).unbind(),
			ops: PyArray3::zeros(py, [n, obs::MAX_OPS, obs::OP_INT], false).unbind(),
			ops_f: PyArray3::zeros(py, [n, obs::MAX_OPS, obs::OP_FLOAT], false).unbind(),
			n_ops: PyArray1::zeros(py, [n], false).unbind(),
			n_tokens: PyArray1::zeros(py, [n], false).unbind(),
			n_hand: PyArray1::zeros(py, [n], false).unbind(),
		}
	}

	fn borrow<'py>(&self, py: Python<'py>, n: usize) -> PyResult<ObsGuards<'py>> {
		Ok(ObsGuards {
			tok_i: borrow(&self.tok_i, py, &[n, obs::MAX_TOKENS, obs::TOK_INT])?,
			tok_f: borrow(&self.tok_f, py, &[n, obs::MAX_TOKENS, obs::TOK_FLOAT])?,
			glob_i: borrow(&self.glob_i, py, &[n, obs::GLOB_INT])?,
			glob_f: borrow(&self.glob_f, py, &[n, obs::GLOB_FLOAT])?,
			ops: borrow(&self.ops, py, &[n, obs::MAX_OPS, obs::OP_INT])?,
			ops_f: borrow(&self.ops_f, py, &[n, obs::MAX_OPS, obs::OP_FLOAT])?,
			n_ops: borrow(&self.n_ops, py, &[n])?,
			n_tokens: borrow(&self.n_tokens, py, &[n])?,
			n_hand: borrow(&self.n_hand, py, &[n])?,
		})
	}
}

impl ObsGuards<'_> {
	/// The rows of every env (the shapes were checked, so there is one per
	/// env).
	fn rows(&mut self) -> PyResult<Vec<Rows<'_>>> {
		let rows = slice(&mut self.tok_i)?
			.chunks_mut(obs::MAX_TOKENS * obs::TOK_INT)
			.zip(slice(&mut self.tok_f)?.chunks_mut(obs::MAX_TOKENS * obs::TOK_FLOAT))
			.zip(slice(&mut self.glob_i)?.chunks_mut(obs::GLOB_INT))
			.zip(slice(&mut self.glob_f)?.chunks_mut(obs::GLOB_FLOAT))
			.zip(slice(&mut self.ops)?.chunks_mut(obs::MAX_OPS * obs::OP_INT))
			.zip(slice(&mut self.ops_f)?.chunks_mut(obs::MAX_OPS * obs::OP_FLOAT))
			.zip(slice(&mut self.n_ops)?.iter_mut())
			.zip(slice(&mut self.n_tokens)?.iter_mut())
			.zip(slice(&mut self.n_hand)?.iter_mut())
			.map(
				|((((((((ti, tf), gi), gf), oa), of), n_ops), n_tokens), n_hand)| Rows {
					ti,
					tf,
					gi,
					gf,
					oa,
					of,
					n_ops,
					n_tokens,
					n_hand,
				},
			)
			.collect();
		Ok(rows)
	}
}

/// Parallel, auto-resetting environments.
///
/// The observation arrays (`tok_i`, `tok_f`, `glob_i`, `glob_f`, `ops`,
/// `ops_f`, `n_ops`, `n_tokens`, `n_hand`; one row per env) are rewritten
/// in place by `step`, `observe` and `set_game`; finished envs are reset
/// automatically. When an episode is truncated, the observation of its last
/// state is in the `final_*` arrays (same layout, valid for the truncated
/// rows of the last `step` only). Do not resize the arrays or use them from
/// other threads while a call is running.
#[pyclass(module = "antezero_sim")]
pub struct VecEnv {
	slots: Vec<Slot>,
	backs: Vec<Back>,
	stakes: Vec<u8>,
	base_seed: u64,
	cfg: Cfg,
	pool: rayon::ThreadPool,
	episodes: Vec<Episode>,
	reservoir: Vec<SimGame>,
	offered: u64,
	cur: ObsArrays,
	fin: ObsArrays,
}

fn make_game(n: u64, backs: &[Back], stakes: &[u8], cfg: Cfg) -> SimGame {
	let h = n.wrapping_mul(0xd6e8_feb8_6659_fd93);
	let back = backs[num::pick(h, backs.len())];
	let stake = stakes[num::pick(h >> 20, stakes.len())];
	let mut g = SimGame::new_run(&seed_string(n), back, stake);
	g.set_tracing(false);
	g.endless = cfg.endless;
	g.allow_skip = cfg.allow_skip;
	g
}

/// A fresh run, or (with probability `snapshot_prob`) a reservoir
/// snapshot whose hidden state is resampled.
fn start(
	n: u64, backs: &[Back], stakes: &[u8], cfg: Cfg, reservoir: &[SimGame],
) -> (SimGame, bool) {
	if cfg.snapshot_prob > 0.0 && !reservoir.is_empty() && unit(n ^ 0x5eed) < cfg.snapshot_prob {
		let mut g = reservoir[num::pick(mix(n ^ 0xa11), reservoir.len())].clone();
		g.resample_hidden(&seed_string(n));
		g.set_tracing(false);
		g.endless = cfg.endless;
		g.allow_skip = cfg.allow_skip;
		return (g, true);
	}
	(make_game(n, backs, stakes, cfg), false)
}

/// Whether `g` is the start of an ante worth restarting from.
fn snapshot_point(g: &SimGame, cfg: &Cfg) -> bool {
	g.phase == Phase::BlindSelect
		&& g.blind_states[0] == BlindStatus::Select
		&& g.ante >= cfg.snapshot_min_ante
		&& !g.is_terminal()
		&& (cfg.endless || !g.run_won())
}

/// Applies actions (if any) and re-encodes all observations in parallel.
fn run(env: &mut VecEnv, py: Python<'_>, acts: Option<Choices<'_>>) -> PyResult<Vec<StepOut>> {
	let mut guards = env.cur.borrow(py, env.slots.len())?;
	let mut rows = guards.rows()?;
	let cfg = env.cfg;
	let ctx = StepCtx {
		cfg,
		backs: &env.backs,
		stakes: &env.stakes,
		base_seed: env.base_seed,
		n_slots: env.slots.len() as u64,
		reservoir: &env.reservoir,
		p_offer: if cfg.snapshot_prob > 0.0 { OFFER } else { 0.0 },
	};
	let slots = &mut env.slots;
	let pool = &env.pool;
	let mut out: Vec<StepOut> = py.detach(|| {
		pool.install(|| {
			slots
				.par_iter_mut()
				.zip(rows.par_iter_mut())
				.enumerate()
				.map(|(i, (s, row))| {
					let res = match acts {
						Some((op, cards, dest)) => s.step(i, (op[i], cards[i], dest[i]), &ctx, row),
						None => StepOut::default(),
					};
					row.write(
						&mut s.g,
						cfg.candidates,
						cfg.deck_tokens,
						cfg.previews,
						&mut s.ops,
						&mut s.pv,
					);
					res
				})
				.collect()
		})
	});
	drop(rows);
	drop(guards);
	// Truncated episodes: final observations.
	if env.slots.iter().any(|s| s.fin.is_some()) {
		let mut guards = env.fin.borrow(py, env.slots.len())?;
		for (s, mut row) in env.slots.iter_mut().zip(guards.rows()?) {
			if let Some(f) = s.fin.take() {
				row.copy_from(&f);
			}
		}
	}
	for r in &mut out {
		if let Some(ep) = r.episode.take() {
			env.episodes.push(ep);
		}
		if let Some(g) = r.snapshot.take() {
			env.offered += 1;
			if env.reservoir.len() < RESERVOIR {
				env.reservoir.push(*g);
			} else {
				let k = num::pick(mix(env.offered ^ env.base_seed), RESERVOIR);
				env.reservoir[k] = *g;
			}
		}
	}
	Ok(out)
}

#[pymethods]
impl VecEnv {
	/// `VecEnv(num_envs, backs=["b_red"], stakes=[1], seed=0, endless=False,
	/// max_ante=16, step_limit=5000, gamma=0.999, r_win=1.0, r_endless=0.0,
	/// r_loss=0.0, w_blind=0.04, w_score=0.02, threads=0, play_candidates=0,
	/// deck_tokens=False, snapshot_prob=0.0, snapshot_min_ante=2, allow_skip=True,
	/// build_previews=True, terminal_phi=True)`. `build_previews`: capacity previews in shops,
	/// packs and blind select (see `sim::obs::BuildPreview`).
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
		reason = "pyo3 hands arguments over by value"
	)]
	#[new]
	#[pyo3(signature = (num_envs, backs = vec!["b_red".to_string()], stakes = vec![1], seed = 0, endless = false, max_ante = 16, step_limit = 5000, gamma = 0.999, r_win = 1.0, r_endless = 0.0, r_loss = 0.0, w_blind = 0.04, w_score = 0.02, threads = 0, play_candidates = 0, deck_tokens = false, snapshot_prob = 0.0, snapshot_min_ante = 2, allow_skip = true, build_previews = true, terminal_phi = true))]
	fn new(
		py: Python<'_>, num_envs: usize, backs: Vec<String>, stakes: Vec<u8>, seed: u64,
		endless: bool, max_ante: i32, step_limit: u32, gamma: f32, r_win: f32, r_endless: f32,
		r_loss: f32, w_blind: f32, w_score: f32, threads: usize, play_candidates: usize,
		deck_tokens: bool, snapshot_prob: f32, snapshot_min_ante: i32, allow_skip: bool,
		build_previews: bool, terminal_phi: bool,
	) -> PyResult<VecEnv> {
		let bad = |msg: String| Err(PyValueError::new_err(msg));
		if num_envs == 0 {
			return bad("num_envs must be > 0".into());
		}
		let backs: Vec<Back> = backs
			.iter()
			.map(|b| crate::game::parse_back(b))
			.collect::<PyResult<_>>()?;
		if backs.is_empty() || stakes.is_empty() || stakes.iter().any(|s| !(1..=8).contains(s)) {
			return bad("need at least one back and stakes in 1..=8".into());
		}
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
		if !(0.0..=1.0).contains(&snapshot_prob) {
			return bad(format!(
				"snapshot_prob must be in [0, 1], got {snapshot_prob}"
			));
		}
		check_candidates(play_candidates)?;
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
			candidates: play_candidates,
			deck_tokens,
			previews: build_previews,
			snapshot_prob,
			snapshot_min_ante,
			allow_skip,
		};
		let pool = rayon::ThreadPoolBuilder::new()
			.num_threads(threads)
			.build()
			.map_err(|e| PyValueError::new_err(e.to_string()))?;
		let slots: Vec<Slot> = py.detach(|| {
			pool.install(|| {
				(0..num_envs)
					.into_par_iter()
					.map(|i| {
						let n = seed.wrapping_mul(1_000_003).wrapping_add(i as u64);
						let g = make_game(n, &backs, &stakes, cfg);
						Slot {
							phi: reward.potential(&g),
							g,
							ops: Vec::with_capacity(obs::MAX_OPS),
							steps: 0,
							ret: 0.0,
							resets: 0,
							invalid: 0,
							from_snapshot: false,
							fin: None,
							pv: Vec::new(),
						}
					})
					.collect()
			})
		});
		let mut env = VecEnv {
			slots,
			backs,
			stakes,
			base_seed: seed,
			cfg,
			pool,
			episodes: Vec::new(),
			reservoir: Vec::new(),
			offered: 0,
			cur: ObsArrays::zeros(py, num_envs),
			fin: ObsArrays::zeros(py, num_envs),
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
		self.reservoir.len()
	}

	/// Re-encodes every observation.
	fn observe(&mut self, py: Python<'_>) -> PyResult<()> {
		run(self, py, None)?;
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
		reason = "pyo3 hands arguments over by value"
	)]
	fn step<'py>(
		&mut self, py: Python<'py>, op: PyReadonlyArray1<'py, i32>,
		cards: PyReadonlyArray1<'py, u32>, dest: PyReadonlyArray1<'py, i32>,
	) -> PyResult<StepArrays<'py>> {
		let n = self.slots.len();
		let (op, cards, dest) = (op.as_slice()?, cards.as_slice()?, dest.as_slice()?);
		if op.len() != n || cards.len() != n || dest.len() != n {
			return Err(PyValueError::new_err(format!("expected {n} actions")));
		}
		let res = run(self, py, Some((op, cards, dest)))?;
		let rew = PyArray1::from_vec(py, res.iter().map(|r| r.reward).collect());
		let done = PyArray1::from_vec(py, res.iter().map(|r| r.done).collect());
		let trunc = PyArray1::from_vec(py, res.iter().map(|r| r.truncated).collect());
		Ok((rew, done, trunc))
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
				e.back,
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

	/// Current antes of all envs.
	fn antes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
		PyArray1::from_vec(py, self.slots.iter().map(|s| s.g.ante).collect())
	}

	/// Number of composite actions that were illegal and replaced by a
	/// fallback (should stay ~0 with correct masking).
	fn invalid_actions(&self) -> u64 {
		self.slots.iter().map(|s| s.invalid).sum()
	}

	/// A snapshot of env `i` as a `Game`.
	fn game(&self, i: usize) -> PyResult<crate::game::Game> {
		let s = self
			.slots
			.get(i)
			.ok_or_else(|| PyValueError::new_err("index out of range"))?;
		Ok(crate::game::Game::from(s.g.clone()))
	}

	/// Replaces env `i`'s game (e.g. for evaluation on fixed seeds) with a
	/// copy of `game` that follows this env's `endless` setting. The game
	/// must not be over.
	#[expect(
		clippy::needless_pass_by_value,
		reason = "pyo3 hands arguments over by value"
	)]
	fn set_game(
		&mut self, py: Python<'_>, i: usize, game: PyRef<'_, crate::game::Game>,
	) -> PyResult<()> {
		if game.g.is_terminal() {
			return Err(PyValueError::new_err(
				"set_game needs a game that is not over",
			));
		}
		let s = self
			.slots
			.get_mut(i)
			.ok_or_else(|| PyValueError::new_err("index out of range"))?;
		s.g = game.g.clone();
		s.g.set_tracing(false);
		s.g.endless = self.cfg.endless;
		s.g.allow_skip = self.cfg.allow_skip;
		s.steps = 0;
		s.ret = 0.0;
		s.from_snapshot = false;
		s.phi = self.cfg.reward.potential(&s.g);
		self.observe(py)
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
