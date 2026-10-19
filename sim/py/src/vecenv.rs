//! `VecEnv`: many simulator games stepped in parallel (rayon), writing
//! observations straight into preallocated numpy arrays.

use crate::rows::{ObsRow, Rows, check_candidates};
use numpy::ndarray::Dimension;
use numpy::{Element, PyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadwriteArray, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use sim::data::Back;
use sim::game::{BlindStatus, Game as SimGame, Phase};
use sim::obs::{self, Op};

const SEED_CHARS: &[u8] = b"123456789ABCDEFGHIJKLMNPQRSTUVWXYZ";

/// Seed string `n` in the game's seed alphabet (8 characters).
fn seed_string(mut n: u64) -> String {
    // Spread consecutive indices over the seed space.
    n = n.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (n >> 29);
    (0..8)
        .map(|_| {
            let c = SEED_CHARS[(n % SEED_CHARS.len() as u64) as usize] as char;
            n /= SEED_CHARS.len() as u64;
            c
        })
        .collect()
}

fn mix(n: u64) -> u64 {
    let mut z = n.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Uniform in [0, 1) from a hash.
fn unit(n: u64) -> f32 {
    (mix(n) >> 40) as f32 / (1u64 << 24) as f32
}

/// The shaping potential `w_blind * blinds cleared + w_score * fraction of
/// the current blind's chips scored`.
pub(crate) fn potential(g: &SimGame, w_blind: f32, w_score: f32) -> f32 {
    let round = if g.phase == Phase::Round && g.blind.chips > 0.0 { (g.chips / g.blind.chips).min(1.0) as f32 } else { 0.0 };
    w_blind * g.blinds_cleared as f32 + w_score * round
}

/// The reward. The true objective is `r_win` for winning the run (plus
/// `r_endless` per blind cleared after the win, for endless training, and
/// `-r_loss` for losing). Progress is rewarded only through potential-based
/// shaping `F = gamma * Phi(s') - Phi(s)` with `Phi(terminal) = 0`, which
/// leaves the optimal policy unchanged (Ng et al., 1999); `Phi` is
/// [`potential`].
#[derive(Clone, Copy, Debug)]
struct Reward {
    gamma: f32,
    r_win: f32,
    r_endless: f32,
    r_loss: f32,
    w_blind: f32,
    w_score: f32,
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
            r += self.r_endless * g.blinds_cleared.saturating_sub(blinds_before) as f32;
        }
        if g.phase == Phase::GameOver {
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
    candidates: usize,
    deck_tokens: bool,
    snapshot_prob: f32,
    snapshot_min_ante: i32,
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
}

/// A finished episode: (env slot, seed, back, stake, ante, won, blinds
/// cleared, steps, true return, truncated, started from a snapshot).
type Episode = (usize, String, String, u8, i32, bool, u32, u32, f32, bool, bool);

/// Per-slot result of a step: (shaped reward, done, truncated, finished
/// episode, snapshot to offer the reservoir).
type StepOut = (f32, bool, bool, Option<Episode>, Option<Box<SimGame>>);

/// `step`'s result arrays: (reward, done, truncated).
type StepArrays<'py> = (Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<bool>>, Bound<'py, PyArray1<bool>>);

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
fn borrow<'py, T: Element, D: Dimension>(a: &Py<PyArray<T, D>>, py: Python<'py>, shape: &[usize]) -> PyResult<PyReadwriteArray<'py, T, D>> {
    let a = a.bind(py);
    if a.shape() != shape {
        return Err(PyValueError::new_err(format!("observation array has shape {:?}, expected {shape:?}", a.shape())));
    }
    a.try_readwrite().map_err(|e| PyValueError::new_err(format!("observation array is in use: {e}")))
}

fn slice<'a, T: Element, D: Dimension>(a: &'a mut PyReadwriteArray<'_, T, D>) -> PyResult<&'a mut [T]> {
    a.as_slice_mut().map_err(|e| PyValueError::new_err(format!("observation array is not contiguous: {e}")))
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
            .map(|((((((((ti, tf), gi), gf), oa), of), n_ops), n_tokens), n_hand)| Rows { ti, tf, gi, gf, oa, of, n_ops, n_tokens, n_hand })
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

impl VecEnv {
    fn make_game(n: u64, backs: &[Back], stakes: &[u8], cfg: Cfg) -> SimGame {
        let h = n.wrapping_mul(0xD6E8_FEB8_6659_FD93);
        let back = backs[(h % backs.len() as u64) as usize];
        let stake = stakes[((h >> 20) % stakes.len() as u64) as usize];
        let mut g = SimGame::new_run(&seed_string(n), back, stake);
        g.set_tracing(false);
        g.endless = cfg.endless;
        g
    }

    /// A fresh run, or (with probability `snapshot_prob`) a reservoir
    /// snapshot whose hidden state is resampled.
    fn start(n: u64, backs: &[Back], stakes: &[u8], cfg: Cfg, reservoir: &[SimGame]) -> (SimGame, bool) {
        if cfg.snapshot_prob > 0.0 && !reservoir.is_empty() && unit(n ^ 0x5EED) < cfg.snapshot_prob {
            let mut g = reservoir[(mix(n ^ 0xA11) % reservoir.len() as u64) as usize].clone();
            g.resample_hidden(&seed_string(n));
            g.set_tracing(false);
            g.endless = cfg.endless;
            return (g, true);
        }
        (VecEnv::make_game(n, backs, stakes, cfg), false)
    }

    /// Whether `g` is the start of an ante worth restarting from.
    fn snapshot_point(g: &SimGame, cfg: &Cfg) -> bool {
        g.phase == Phase::BlindSelect && g.blind_states[0] == BlindStatus::Select && g.ante >= cfg.snapshot_min_ante && !g.is_terminal() && (cfg.endless || !g.run_won())
    }
}

#[allow(clippy::too_many_arguments)]
#[pymethods]
impl VecEnv {
    /// `VecEnv(num_envs, backs=["b_red"], stakes=[1], seed=0, endless=False,
    /// max_ante=16, step_limit=5000, gamma=0.999, r_win=1.0, r_endless=0.0,
    /// r_loss=0.0, w_blind=0.04, w_score=0.02, threads=0, play_candidates=0,
    /// deck_tokens=False, snapshot_prob=0.0, snapshot_min_ante=2)`.
    ///
    /// Episodes end when the run is lost, or won unless `endless`; they are
    /// truncated (not terminal: bootstrap from `final_*`) after `step_limit`
    /// steps or, in endless mode, beyond `max_ante`.
    #[new]
    #[pyo3(signature = (num_envs, backs = vec!["b_red".to_string()], stakes = vec![1], seed = 0, endless = false, max_ante = 16, step_limit = 5000, gamma = 0.999, r_win = 1.0, r_endless = 0.0, r_loss = 0.0, w_blind = 0.04, w_score = 0.02, threads = 0, play_candidates = 0, deck_tokens = false, snapshot_prob = 0.0, snapshot_min_ante = 2))]
    fn new(
        py: Python<'_>,
        num_envs: usize,
        backs: Vec<String>,
        stakes: Vec<u8>,
        seed: u64,
        endless: bool,
        max_ante: i32,
        step_limit: u32,
        gamma: f32,
        r_win: f32,
        r_endless: f32,
        r_loss: f32,
        w_blind: f32,
        w_score: f32,
        threads: usize,
        play_candidates: usize,
        deck_tokens: bool,
        snapshot_prob: f32,
        snapshot_min_ante: i32,
    ) -> PyResult<VecEnv> {
        let bad = |msg: String| Err(PyValueError::new_err(msg));
        if num_envs == 0 {
            return bad("num_envs must be > 0".into());
        }
        let backs: Vec<Back> = backs.iter().map(|b| crate::parse_back(b)).collect::<PyResult<_>>()?;
        if backs.is_empty() || stakes.is_empty() || stakes.iter().any(|s| !(1..=8).contains(s)) {
            return bad("need at least one back and stakes in 1..=8".into());
        }
        if step_limit == 0 || max_ante < 1 {
            return bad(format!("step_limit and max_ante must be positive, got {step_limit} and {max_ante}"));
        }
        if !(gamma > 0.0 && gamma <= 1.0) {
            return bad(format!("gamma must be in (0, 1], got {gamma}"));
        }
        if ![r_win, r_endless, r_loss, w_blind, w_score].iter().all(|x| x.is_finite()) {
            return bad("rewards and shaping weights must be finite".into());
        }
        if !(0.0..=1.0).contains(&snapshot_prob) {
            return bad(format!("snapshot_prob must be in [0, 1], got {snapshot_prob}"));
        }
        check_candidates(play_candidates)?;
        let reward = Reward { gamma, r_win, r_endless, r_loss, w_blind, w_score };
        let cfg = Cfg { endless, max_ante, step_limit, reward, candidates: play_candidates, deck_tokens, snapshot_prob, snapshot_min_ante };
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().map_err(|e| PyValueError::new_err(e.to_string()))?;
        let slots: Vec<Slot> = py.detach(|| {
            pool.install(|| {
                (0..num_envs)
                    .into_par_iter()
                    .map(|i| {
                        let n = seed.wrapping_mul(1_000_003).wrapping_add(i as u64);
                        let g = VecEnv::make_game(n, &backs, &stakes, cfg);
                        Slot { phi: reward.potential(&g), g, ops: Vec::with_capacity(obs::MAX_OPS), steps: 0, ret: 0.0, resets: 0, invalid: 0, from_snapshot: false, fin: None }
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
        self.run(py, None)?;
        Ok(())
    }

    /// Steps every env with `(op, cards, dest)`; returns `(reward, done,
    /// truncated)` arrays (`done` includes truncation). Rewards are shaped
    /// (see `Reward`). An illegal choice (counted by `invalid_actions`) is
    /// replaced by a legal fallback. Finished envs are reset (their new
    /// observation is already written); see `pop_episodes` and the
    /// `final_*` arrays.
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        op: PyReadonlyArray1<'py, i32>,
        cards: PyReadonlyArray1<'py, u32>,
        dest: PyReadonlyArray1<'py, i32>,
    ) -> PyResult<StepArrays<'py>> {
        let n = self.slots.len();
        let (op, cards, dest) = (op.as_slice()?.to_vec(), cards.as_slice()?.to_vec(), dest.as_slice()?.to_vec());
        if op.len() != n || cards.len() != n || dest.len() != n {
            return Err(PyValueError::new_err(format!("expected {n} actions")));
        }
        let res = self.run(py, Some((&op, &cards, &dest)))?;
        let rew = PyArray1::from_vec(py, res.iter().map(|r| r.0).collect());
        let done = PyArray1::from_vec(py, res.iter().map(|r| r.1).collect());
        let trunc = PyArray1::from_vec(py, res.iter().map(|r| r.2).collect());
        Ok((rew, done, trunc))
    }

    /// Finished episodes since the last call: `(slot, seed, back, stake,
    /// ante, won, blinds_cleared, steps, return, truncated, from_snapshot)`.
    /// `won` is the run being won (not the game's `won` flag, which is also
    /// set when the final boss is lost); `return` is unshaped.
    fn pop_episodes(&mut self) -> Vec<Episode> {
        std::mem::take(&mut self.episodes)
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
    fn game(&self, i: usize) -> PyResult<crate::Game> {
        let s = self.slots.get(i).ok_or_else(|| PyValueError::new_err("index out of range"))?;
        Ok(crate::Game::from_sim(s.g.clone()))
    }

    /// Replaces env `i`'s game (e.g. for evaluation on fixed seeds) with a
    /// copy of `game` that follows this env's `endless` setting. The game
    /// must not be over.
    fn set_game(&mut self, py: Python<'_>, i: usize, game: PyRef<'_, crate::Game>) -> PyResult<()> {
        if game.inner().is_terminal() {
            return Err(PyValueError::new_err("set_game needs a game that is not over"));
        }
        let s = self.slots.get_mut(i).ok_or_else(|| PyValueError::new_err("index out of range"))?;
        s.g = game.inner().clone();
        s.g.set_tracing(false);
        s.g.endless = self.cfg.endless;
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

impl VecEnv {
    /// Applies actions (if any) and re-encodes all observations in parallel.
    fn run(&mut self, py: Python<'_>, acts: Option<Choices<'_>>) -> PyResult<Vec<StepOut>> {
        let mut guards = self.cur.borrow(py, self.slots.len())?;
        let mut rows = guards.rows()?;
        let cfg = self.cfg;
        let rw = cfg.reward;
        let backs = &self.backs;
        let stakes = &self.stakes;
        let base = self.base_seed;
        let nslots = self.slots.len() as u64;
        let slots = &mut self.slots;
        let pool = &self.pool;
        let reservoir = &self.reservoir;
        let p_offer = if cfg.snapshot_prob > 0.0 { OFFER } else { 0.0 };
        let mut out: Vec<StepOut> = py.detach(|| {
            pool.install(|| {
                slots
                    .par_iter_mut()
                    .zip(rows.par_iter_mut())
                    .enumerate()
                    .map(|(i, (s, row))| {
                        let mut res: StepOut = (0.0, false, false, None, None);
                        if let Some((op, cards, dest)) = acts {
                            let blinds_before = s.g.blinds_cleared;
                            let won_before = s.g.run_won();
                            // A negative destination is out of range (it only
                            // matters for reordering ops).
                            let d = u32::try_from(dest[i]).unwrap_or(u32::MAX);
                            let a = usize::try_from(op[i]).ok().and_then(|k| s.ops.get(k)).and_then(|o| obs::decode(&s.g, o, cards[i], d));
                            let a = match a {
                                Some(a) => a,
                                None => {
                                    s.invalid += 1;
                                    obs::fallback(&s.g, &s.ops).expect("no legal action")
                                }
                            };
                            s.g.step(a).expect("decoded action rejected");
                            s.steps += 1;
                            let g = &s.g;
                            let r_true = rw.true_reward(blinds_before, won_before, g);
                            let terminal = g.is_terminal();
                            let trunc = !terminal && (s.steps >= cfg.step_limit || (cfg.endless && g.ante > cfg.max_ante));
                            let phi = if terminal { 0.0 } else { rw.potential(g) };
                            let r = r_true + rw.gamma * phi - s.phi;
                            s.phi = phi;
                            s.ret += r_true;
                            res.0 = r;
                            res.1 = terminal || trunc;
                            res.2 = trunc;
                            if !terminal && !trunc && p_offer > 0.0 && VecEnv::snapshot_point(g, &cfg) && unit(mix(base ^ i as u64) ^ (s.resets << 20) ^ s.steps as u64) < p_offer {
                                res.4 = Some(Box::new(g.clone()));
                            }
                            if terminal || trunc {
                                let ep = (i, g.rng.seed().to_string(), g.back.key().to_string(), g.stake, g.ante, g.run_won(), g.blinds_cleared, s.steps, s.ret, trunc, s.from_snapshot);
                                res.3 = Some(ep);
                                if trunc {
                                    row.write(&mut s.g, cfg.candidates, cfg.deck_tokens, &mut s.ops);
                                    s.fin = Some(ObsRow::of(row));
                                }
                                s.resets += 1;
                                let n = base.wrapping_mul(1_000_003).wrapping_add(s.resets * nslots + i as u64);
                                let (g, snap) = VecEnv::start(n, backs, stakes, cfg, reservoir);
                                s.g = g;
                                s.from_snapshot = snap;
                                s.steps = 0;
                                s.ret = 0.0;
                                s.phi = rw.potential(&s.g);
                            }
                        }
                        row.write(&mut s.g, cfg.candidates, cfg.deck_tokens, &mut s.ops);
                        res
                    })
                    .collect()
            })
        });
        drop(rows);
        drop(guards);
        // Truncated episodes: final observations.
        if self.slots.iter().any(|s| s.fin.is_some()) {
            let mut guards = self.fin.borrow(py, self.slots.len())?;
            for (s, mut row) in self.slots.iter_mut().zip(guards.rows()?) {
                if let Some(f) = s.fin.take() {
                    row.copy_from(&f);
                }
            }
        }
        for r in &mut out {
            if let Some(ep) = r.3.take() {
                self.episodes.push(ep);
            }
            if let Some(g) = r.4.take() {
                self.offered += 1;
                if self.reservoir.len() < RESERVOIR {
                    self.reservoir.push(*g);
                } else {
                    let k = (mix(self.offered ^ self.base_seed) % RESERVOIR as u64) as usize;
                    self.reservoir[k] = *g;
                }
            }
        }
        Ok(out)
    }
}
