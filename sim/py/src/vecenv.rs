//! `VecEnv`: many auto-resetting games stepped in parallel, with the reward
//! shaping and the reservoirs of ante starts the episodes restart from.

use std::ops::{Index, IndexMut, RangeInclusive};

use numpy::ndarray::Dimension;
use numpy::{
	Element, PyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1,
	PyReadwriteArray, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use sim::data::{ALL_JOKERS, ALL_PLANETS, Back};
use sim::game::{BlindSlot, BlindStatus, Game as SimGame, Phase};
use sim::inject::Inject;
use sim::obs::{self, BuildPreview, Op, OpOptions};
use sim::rng::splitmix64;
use sim::{Action, CardMask, Stake, num};

use crate::game::{Game, parse_back};
use crate::rows::{
	OPS_F_SHAPE, OPS_SHAPE, ObsOptions, Rows, TOK_F_SHAPE, TOK_I_SHAPE, op_options, row_len,
};

const SEED_CHARS: &[u8] = b"123456789ABCDEFGHIJKLMNPQRSTUVWXYZ";

/// Seed string `n` in the game's seed alphabet (8 characters).
fn seed_string(n: u64) -> String {
	// Spread consecutive indices over the seed space.
	let mut n = n.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (n >> 29);
	let radix = num::index64(SEED_CHARS.len());
	let mut seed = String::with_capacity(8);
	for _ in 0..8 {
		seed.push(char::from(SEED_CHARS[num::pick(n, SEED_CHARS.len())]));
		n /= radix;
	}
	seed
}

/// Uniform in [0, 1) from a hash.
#[expect(
	clippy::cast_precision_loss,
	reason = "`splitmix64(n) >> 40` is below 2^24, which an f32 holds exactly"
)]
fn unit(n: u64) -> f32 {
	(splitmix64(n) >> 40) as f32 / 16_777_216.0
}

/// The shaping potential `w_blind * progress + w_score * fraction of the
/// current blind's chips scored`, with progress = the run position in blinds
/// (the blinds of every finished ante plus this ante's defeated ones). The
/// lost blind's fraction is kept at game over: if losing dropped it,
/// delaying a loss would pay (stalling).
// Counting blinds cleared instead would pay for skipping blinds and for
// Hieroglyph/Petroglyph (which add an ante of blinds to beat), and policies
// exploit both.
pub(crate) fn potential(g: &SimGame, w_blind: f32, w_score: f32) -> f32 {
	let in_round = matches!(g.flow.phase, Phase::Round | Phase::GameOver);
	let round = if in_round && g.blind.chips > 0.0 {
		num::feat((g.round.chips / g.blind.chips).min(1.0))
	} else {
		0.0
	};
	let defeated = num::id(
		g.blind_select
			.states
			.iter()
			.filter(|&&s| s == BlindStatus::Defeated)
			.count(),
	);
	let progress = num::id(BlindSlot::ALL.len()) * (g.run.ante - 1).max(0) + defeated;
	w_blind * num::feat(f64::from(progress)) + w_score * round
}

/// The reward. The true objective is `r_win` for winning the run (plus
/// `r_endless` per blind cleared after the win, for endless training, and
/// `-r_loss` for losing). Progress is rewarded through the shaping
/// `F = gamma * Phi(s') - Phi(s)` ([`potential`]). `Phi` is *not* zeroed at
/// terminal states: with `Phi(terminal) = 0` a lost game would hand back all
/// its progress, every loss would return the same, and a policy that never
/// wins would get no learning signal. So the episode return is the true
/// return plus `gamma^T * Phi(s_T)`, the discounted shaping potential of the
/// final state. With `terminal_phi = false` Phi *is* zeroed there (classic
/// shaping: exactly P(win)).
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

	fn true_reward(&self, blinds_before: u32, won_before: bool, g: &SimGame) -> f32 {
		let mut r = 0.0;
		if g.run_won() && !won_before {
			r += self.r_win;
		} else if won_before {
			r += self.r_endless
				* num::feat(f64::from(
					g.run.blinds_cleared.saturating_sub(blinds_before),
				));
		}
		if g.flow.phase == Phase::GameOver || g.is_stuck() {
			r -= self.r_loss;
		}
		r
	}
}

#[derive(Clone, Copy)]
struct Cfg {
	endless: bool,
	max_ante: i32,
	step_limit: u32,
	reward: Reward,
	obs: ObsOptions,
	/// The share of played hands that get a joker-order label.
	order_label_rate: f32,
	snapshot_prob: f32,
	snapshot_min_ante: i32,
	practice_prob: f32,
	practice_eternal: bool,
}

/// Applies the env's settings to a game it is about to run.
fn configure(g: &mut SimGame, cfg: &Cfg) {
	g.set_tracing(false);
	g.setup.endless = cfg.endless;
}

/// Snapshots kept for restarts, per pool (a pool follows the current
/// policy's state distribution).
const RESERVOIR: usize = 4096;
/// The probability that an ante start is offered to its reservoir, where it
/// replaces a random entry once the reservoir is full.
const OFFER: f32 = 0.25;
/// The antes whose starts the practice pool takes: later ante starts mostly
/// have no joker slot free for a gift.
const PRACTICE_ANTES: RangeInclusive<i32> = 2..=3;
/// The planets of the nine hand types every run can play.
const GIFT_PLANETS: usize = 9;
const GIFT_LEVELS: f64 = 3.0;
/// Keeps the label sampling apart from the snapshot offers (the same hash
/// otherwise).
const LABEL_SALT: u64 = 0x6f72_6465_725f_6c62;

/// Run `slot`'s seed index after `resets` resets of a `VecEnv` of `n_slots`
/// envs started from `base_seed`.
fn run_number(base_seed: u64, resets: u64, n_slots: u64, slot: usize) -> u64 {
	base_seed
		.wrapping_mul(1_000_003)
		.wrapping_add(resets * n_slots + num::index64(slot))
}

/// A pool of ante starts that episodes restart from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pool {
	/// Late ante starts (`snapshot_min_ante` on), restarted from as they are.
	Late,
	/// Early ante starts (`PRACTICE_ANTES`), restarted from with a gift.
	Practice,
}

/// The snapshots kept for restarts.
#[derive(Default)]
struct Reservoirs {
	late: Vec<SimGame>,
	practice: Vec<SimGame>,
}

impl Index<Pool> for Reservoirs {
	type Output = Vec<SimGame>;

	fn index(&self, pool: Pool) -> &Vec<SimGame> {
		match pool {
			Pool::Late => &self.late,
			Pool::Practice => &self.practice,
		}
	}
}

impl IndexMut<Pool> for Reservoirs {
	fn index_mut(&mut self, pool: Pool) -> &mut Vec<SimGame> {
		match pool {
			Pool::Late => &mut self.late,
			Pool::Practice => &mut self.practice,
		}
	}
}

/// Where an episode's game came from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Origin {
	/// A new run, or a game handed in by `set_game`.
	Fresh,
	/// A snapshot of the pool, with its hidden state resampled.
	Snapshot(Pool),
}

struct Slot {
	g: SimGame,
	/// The ops of the current observation (to decode the next choice).
	ops: Vec<Op>,
	steps: u32,
	ret: f32,
	phi: f32,
	resets: u64,
	invalid: u64,
	origin: Origin,
	/// Recent build previews (reused while the build is unchanged).
	previews: Vec<BuildPreview>,
}

struct StepCtx<'a> {
	cfg: Cfg,
	backs: &'a [Back],
	stakes: &'a [Stake],
	base_seed: u64,
	n_slots: u64,
	reservoirs: &'a Reservoirs,
	/// Probability of offering an ante start to its reservoir.
	p_offer: f32,
}

/// One env's choice: an op index, a card mask and a destination.
#[derive(Clone, Copy)]
struct Choice {
	op: i32,
	cards: u32,
	dest: i32,
}

/// One step's choices of every env.
#[derive(Clone, Copy)]
struct Choices<'a> {
	op: &'a [i32],
	cards: &'a [u32],
	dest: &'a [i32],
}

impl Choices<'_> {
	fn of(self, i: usize) -> Choice {
		Choice {
			op: self.op[i],
			cards: self.cards[i],
			dest: self.dest[i],
		}
	}
}

impl Slot {
	fn new(g: SimGame, reward: &Reward) -> Slot {
		Slot {
			phi: reward.potential(&g),
			g,
			ops: Vec::with_capacity(obs::MAX_OPS),
			steps: 0,
			ret: 0.0,
			resets: 0,
			invalid: 0,
			origin: Origin::Fresh,
			previews: Vec::new(),
		}
	}

	/// Starts the slot's next episode on `g`.
	fn install(&mut self, g: SimGame, origin: Origin, reward: &Reward) {
		self.g = g;
		self.origin = origin;
		self.steps = 0;
		self.ret = 0.0;
		self.phi = reward.potential(&self.g);
	}

	/// Uniform in [0, 1) from a hash of `seed`, the env `i`, its game count
	/// and its step count: no RNG is consumed.
	fn roll(&self, seed: u64, i: usize) -> f32 {
		let n = splitmix64(seed ^ num::index64(i));
		unit(n ^ (self.resets << 20) ^ u64::from(self.steps))
	}

	/// Applies the choice of env `i`. If the episode ended, the env is reset;
	/// a truncated episode's last observation is written to `final_row`
	/// first.
	fn step(
		&mut self, i: usize, choice: Choice, ctx: &StepCtx<'_>, final_row: &mut Rows<'_>,
	) -> StepOut {
		let (cfg, rw) = (ctx.cfg, ctx.cfg.reward);
		let mut res = StepOut::default();
		let blinds_before = self.g.run.blinds_cleared;
		let won_before = self.g.run_won();
		let was_ante_start = ante_start(&self.g, cfg.endless);
		let a = usize::try_from(choice.op)
			.ok()
			.and_then(|k| self.ops.get(k))
			.and_then(|o| {
				let cards = CardMask::from_bits(choice.cards);
				obs::decode(&self.g, o, cards, u8::try_from(choice.dest).ok())
			});
		if let Some(Action::Play(mask)) = a
			&& cfg.order_label_rate > 0.0
			&& self.roll(ctx.base_seed ^ LABEL_SALT, i) < cfg.order_label_rate
		{
			res.label = obs::order_label(&self.g, mask);
		}
		let a = if let Some(a) = a {
			Some(a)
		} else {
			self.invalid += 1;
			obs::fallback(&self.g, &self.ops)
		};
		// The known state without a legal action (the game's own pack softlock) is
		// terminal via `is_stuck`. Any other one is a simulator bug, as is an
		// offered action the game then rejects: dump the state and truncate the
		// episode rather than kill a long training run.
		let dead_end = if let Some(a) = a {
			match self.g.step(a) {
				Ok(()) => {
					self.steps += 1;
					false
				}
				Err(e) => {
					eprintln!(
						"[vecenv] {a:?} was offered and then rejected ({e}), episode truncated. ops: {:?}\n{:#?}",
						self.ops, self.g
					);
					true
				}
			}
		} else {
			eprintln!(
				"[vecenv] dead end: no legal action, episode truncated. ops: {:?}\n{:#?}",
				self.ops, self.g
			);
			true
		};
		let g = &self.g;
		let r_true = rw.true_reward(blinds_before, won_before, g);
		let terminal = g.is_terminal();
		let end = if terminal {
			Some(End::Terminal)
		} else if dead_end
			|| self.steps >= cfg.step_limit
			|| (cfg.endless && g.run.ante > cfg.max_ante)
		{
			Some(End::Truncated)
		} else {
			None
		};
		let phi = if terminal && !rw.terminal_phi {
			0.0
		} else {
			rw.potential(g)
		};
		res.reward = r_true + rw.gamma * phi - self.phi;
		self.phi = phi;
		self.ret += r_true;
		res.end = end;
		if end.is_none()
			&& ctx.p_offer > 0.0
			&& let Some(pool) = snapshot_pool(g, &cfg, self.origin, !was_ante_start)
			&& self.roll(ctx.base_seed, i) < ctx.p_offer
		{
			res.snapshot = Some(Snapshot {
				pool,
				game: Box::new(g.clone()),
			});
		}
		if let Some(end) = end {
			res.episode = Some(Episode {
				slot: i,
				seed: g.rng.seed().to_string(),
				back: g.setup.back,
				stake: g.setup.stake.into(),
				ante: g.run.ante,
				won: g.run_won(),
				blinds_cleared: g.run.blinds_cleared,
				steps: self.steps,
				ret: self.ret,
				truncated: end == End::Truncated,
				from_snapshot: self.origin != Origin::Fresh,
			});
			if end == End::Truncated {
				final_row.write(&mut self.g, cfg.obs, &mut self.ops, &mut self.previews);
			}
			self.resets += 1;
			let n = run_number(ctx.base_seed, self.resets, ctx.n_slots, i);
			let (g, origin) = start(n, ctx.backs, ctx.stakes, &cfg, ctx.reservoirs);
			self.install(g, origin, &rw);
		}
		res
	}
}

struct Episode {
	slot: usize,
	seed: String,
	back: Back,
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

/// How an episode ended.
#[derive(Clone, Copy, PartialEq, Eq)]
enum End {
	/// The run was won (outside endless mode) or lost.
	Terminal,
	/// It was cut off (step limit, ante limit or a dead end): bootstrap from
	/// the final state.
	Truncated,
}

/// An ante start to offer the reservoir of its pool.
struct Snapshot {
	pool: Pool,
	game: Box<SimGame>,
}

#[derive(Default)]
struct StepOut {
	/// The shaped reward.
	reward: f32,
	end: Option<End>,
	episode: Option<Episode>,
	snapshot: Option<Snapshot>,
	/// The joker-order label of the play taken (`obs::order_label`), if
	/// sampled.
	label: Option<[u8; obs::ORDER_JOKERS]>,
}

/// `step`'s result arrays: (reward, done, truncated).
type StepArrays<'py> = (
	Bound<'py, PyArray1<f32>>,
	Bound<'py, PyArray1<bool>>,
	Bound<'py, PyArray1<bool>>,
);

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

/// The shape of `n` envs' rows of the shape `row`.
fn rows_of(n: usize, row: [usize; 2]) -> [usize; 3] {
	[n, row[0], row[1]]
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
	fn zeros(py: Python<'_>, n: usize, glob_float: usize) -> ObsArrays {
		ObsArrays {
			tok_i: PyArray3::zeros(py, rows_of(n, TOK_I_SHAPE), /* is_fortran= */ false).unbind(),
			tok_f: PyArray3::zeros(py, rows_of(n, TOK_F_SHAPE), /* is_fortran= */ false).unbind(),
			glob_i: PyArray2::zeros(py, [n, obs::GLOB_INT], /* is_fortran= */ false).unbind(),
			glob_f: PyArray2::zeros(py, [n, glob_float], /* is_fortran= */ false).unbind(),
			ops: PyArray3::zeros(py, rows_of(n, OPS_SHAPE), /* is_fortran= */ false).unbind(),
			ops_f: PyArray3::zeros(py, rows_of(n, OPS_F_SHAPE), /* is_fortran= */ false).unbind(),
			n_ops: PyArray1::zeros(py, [n], /* is_fortran= */ false).unbind(),
			n_tokens: PyArray1::zeros(py, [n], /* is_fortran= */ false).unbind(),
			n_hand: PyArray1::zeros(py, [n], /* is_fortran= */ false).unbind(),
		}
	}

	fn borrow<'py>(
		&self, py: Python<'py>, n: usize, glob_float: usize,
	) -> PyResult<ObsGuards<'py>> {
		Ok(ObsGuards {
			tok_i: borrow(&self.tok_i, py, &rows_of(n, TOK_I_SHAPE))?,
			tok_f: borrow(&self.tok_f, py, &rows_of(n, TOK_F_SHAPE))?,
			glob_i: borrow(&self.glob_i, py, &[n, obs::GLOB_INT])?,
			glob_f: borrow(&self.glob_f, py, &[n, glob_float])?,
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
	fn rows(&mut self, glob_float: usize) -> PyResult<impl IndexedParallelIterator<Item = Rows<'_>>> {
		let rows = slice(&mut self.tok_i)?
			.par_chunks_mut(row_len(TOK_I_SHAPE))
			.zip(slice(&mut self.tok_f)?.par_chunks_mut(row_len(TOK_F_SHAPE)))
			.zip(slice(&mut self.glob_i)?.par_chunks_mut(obs::GLOB_INT))
			.zip(slice(&mut self.glob_f)?.par_chunks_mut(glob_float))
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
	offered: u64,
	cur: ObsArrays,
	fin: ObsArrays,
	aux_kind: Py<PyArray1<i8>>,
	aux_dest: Py<PyArray2<u8>>,
}

fn make_game(n: u64, backs: &[Back], stakes: &[Stake], cfg: &Cfg) -> SimGame {
	let h = n.wrapping_mul(0xd6e8_feb8_6659_fd93);
	let back = backs[num::pick(h, backs.len())];
	let stake = stakes[num::pick(h >> 20, stakes.len())];
	let mut g = SimGame::new_run(&seed_string(n), back, stake);
	configure(&mut g, cfg);
	g
}

/// A fresh run, or a reservoir snapshot whose hidden state is resampled: a
/// late ante start (with probability `snapshot_prob`) or an early one with a
/// gift (with probability `practice_prob`).
fn start(
	n: u64, backs: &[Back], stakes: &[Stake], cfg: &Cfg, reservoirs: &Reservoirs,
) -> (SimGame, Origin) {
	let u = unit(n ^ 0x5eed);
	let pool = if u < cfg.snapshot_prob {
		Some(Pool::Late)
	} else if u < cfg.snapshot_prob + cfg.practice_prob {
		Some(Pool::Practice)
	} else {
		None
	};
	if let Some(pool) = pool.filter(|&pool| !reservoirs[pool].is_empty()) {
		let from = &reservoirs[pool];
		let mut g = from[num::pick(splitmix64(n ^ 0xa11), from.len())].clone();
		g.resample_hidden(&seed_string(n));
		configure(&mut g, cfg);
		if pool == Pool::Practice {
			gift(&mut g, n, cfg.practice_eternal);
		}
		return (g, Origin::Snapshot(pool));
	}
	(make_game(n, backs, stakes, cfg), Origin::Fresh)
}

/// Something to practise with that the run did not choose: a joker (half
/// the time, if a slot is free and it holds none of the kind; Eternal if
/// `eternal` and the joker can be), or `GIFT_LEVELS` levels of one hand type.
fn gift(g: &mut SimGame, n: u64, eternal: bool) {
	let h = splitmix64(n ^ 0x61f7);
	let joker = ALL_JOKERS[num::pick(h >> 1, ALL_JOKERS.len())];
	if h & 1 == 0 && g.areas.joker_room() && !g.areas.jokers.iter().any(|c| c.is_joker(joker)) {
		g.inject(Inject::Joker {
			joker,
			edition: None,
			eternal,
		});
	} else {
		let planet = ALL_PLANETS[num::pick(splitmix64(h), GIFT_PLANETS)];
		g.hands.level_up(planet.hand(), GIFT_LEVELS);
	}
}

/// Whether `g` is at the start of an ante, before its first blind is chosen.
fn ante_start(g: &SimGame, endless: bool) -> bool {
	g.flow.phase == Phase::BlindSelect
		&& g.blind_select.states[BlindSlot::Small] == BlindStatus::Select
		&& !g.is_terminal()
		&& (endless || !g.run_won())
}

/// The pool the ante start `g` belongs to, if any. `origin`: where its
/// episode's game came from; `arrived`: the last action led here. Practice
/// episodes feed neither pool, and only fresh runs feed the practice pool,
/// once per ante: a practice start holds one gift, a late start none, and
/// what a run does before choosing its blind does not weigh the pool.
fn snapshot_pool(g: &SimGame, cfg: &Cfg, origin: Origin, arrived: bool) -> Option<Pool> {
	if !ante_start(g, cfg.endless) {
		None
	} else if cfg.snapshot_prob > 0.0 && g.run.ante >= cfg.snapshot_min_ante {
		(origin != Origin::Snapshot(Pool::Practice)).then_some(Pool::Late)
	} else if cfg.practice_prob > 0.0 && arrived && PRACTICE_ANTES.contains(&g.run.ante) {
		(origin == Origin::Fresh).then_some(Pool::Practice)
	} else {
		None
	}
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

/// Steps every env with its choice (or none, to only observe) and writes the
/// observations and, of a step, the labels. Every array is borrowed before
/// any env changes, so a failed borrow leaves the envs as they were.
fn step_all(
	env: &mut VecEnv, py: Python<'_>, choices: Option<Choices<'_>>,
) -> PyResult<Vec<StepOut>> {
	let n = env.slots.len();
	let glob_float = env.cfg.obs.glob_float();
	let mut cur = env.cur.borrow(py, n, glob_float)?;
	let mut fin = env.fin.borrow(py, n, glob_float)?;
	let mut aux_kind = borrow(&env.aux_kind, py, &[n])?;
	let mut aux_dest = borrow(&env.aux_dest, py, &[n, obs::ORDER_JOKERS])?;
	let rows = cur.rows(glob_float)?;
	let final_rows = fin.rows(glob_float)?;
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
					row.write(&mut s.g, cfg.obs, &mut s.ops, &mut s.previews);
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
		if let Some(ep) = r.episode.take() {
			env.episodes.push(ep);
		}
		if let Some(Snapshot { pool, game }) = r.snapshot.take() {
			env.offered += 1;
			let reservoir = &mut env.reservoirs[pool];
			if reservoir.len() < RESERVOIR {
				reservoir.push(*game);
			} else {
				let k = num::pick(splitmix64(env.offered ^ env.base_seed), RESERVOIR);
				reservoir[k] = *game;
			}
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
	#[pyo3(signature = (num_envs, backs = vec!["b_red".to_string()], stakes = vec![1], seed = 0, endless = false, max_ante = 16, step_limit = 5000, gamma = 0.999, r_win = 1.0, r_endless = 0.0, r_loss = 0.0, w_blind = 0.04, w_score = 0.02, threads = 0, play_candidates = 0, deck_tokens = false, snapshot_prob = 0.0, snapshot_min_ante = 2, allow_skip = true, build_previews = true, terminal_phi = true, order_gain = true, order_label_rate = 0.0, practice_prob = 0.0, practice_eternal = false, null_sorts = true, obs_v2 = false, max_rearranges = obs::MAX_REARRANGES))]
	fn new(
		py: Python<'_>, num_envs: usize, backs: Vec<String>, stakes: Vec<u8>, seed: u64,
		endless: bool, max_ante: i32, step_limit: u32, gamma: f32, r_win: f32, r_endless: f32,
		r_loss: f32, w_blind: f32, w_score: f32, threads: usize, play_candidates: usize,
		deck_tokens: bool, snapshot_prob: f32, snapshot_min_ante: i32, allow_skip: bool,
		build_previews: bool, terminal_phi: bool, order_gain: bool, order_label_rate: f32,
		practice_prob: f32, practice_eternal: bool, null_sorts: bool, obs_v2: bool,
		max_rearranges: u8,
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
					max_rearranges,
				})?,
				deck_tokens,
				previews: build_previews,
				order_gain,
				obs_v2,
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
			offered: 0,
			cur: ObsArrays::zeros(py, num_envs, cfg.obs.glob_float()),
			fin: ObsArrays::zeros(py, num_envs, cfg.obs.glob_float()),
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
		self.reservoirs.late.len() + self.reservoirs.practice.len()
	}

	/// Re-encodes every observation.
	fn observe(&mut self, py: Python<'_>) -> PyResult<()> {
		step_all(self, py, None)?;
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
		let res = step_all(self, py, Some(choices))?;
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
		PyArray1::from_vec(py, self.slots.iter().map(|s| s.g.run.ante).collect())
	}

	/// Number of composite actions that were illegal and replaced by a
	/// fallback (should stay ~0 with correct masking).
	fn invalid_actions(&self) -> u64 {
		self.slots.iter().map(|s| s.invalid).sum()
	}

	/// A snapshot of env `i` as a `Game` (with the env's `allow_skip` and
	/// `null_sorts`).
	fn game(&self, i: usize) -> PyResult<Game> {
		let s = self
			.slots
			.get(i)
			.ok_or_else(|| PyValueError::new_err("index out of range"))?;
		let mut game = Game::from(s.g.clone());
		game.allow_skip = self.cfg.obs.ops.allow_skip;
		game.null_sorts = self.cfg.obs.ops.null_sorts;
		Ok(game)
	}

	/// Replaces env `i`'s game (e.g. for evaluation on fixed seeds) with a
	/// copy of `game` that follows this env's `endless` and `allow_skip`
	/// settings. The game must not be over.
	fn set_game(&mut self, py: Python<'_>, i: usize, game: &Game) -> PyResult<()> {
		if game.g.is_terminal() {
			return Err(PyValueError::new_err(
				"set_game needs a game that is not over",
			));
		}
		let s = self
			.slots
			.get_mut(i)
			.ok_or_else(|| PyValueError::new_err("index out of range"))?;
		let mut g = game.g.clone();
		configure(&mut g, &self.cfg);
		s.install(g, Origin::Fresh, &self.cfg.reward);
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
