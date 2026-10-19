//! `antezero_sim`: Python bindings for the simulator.
//!
//! Two interfaces: a text one (action / inject codes, canonical `str -> str`
//! state dicts), so histories and comparisons work the same across the
//! simulator, the headless oracle and the real-game bridge; and the numeric
//! learning one (`Game.encode` / `step_factored`, and `VecEnv`, which steps
//! many games in parallel), see `sim::obs`.

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod rows;
mod vecenv;

use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rows::{ObsRow, check_candidates};
use sim::data::Back;
use sim::game::{Game as SimGame, Phase};
use sim::inject::Inject;
use sim::obs::{self, Layout, Op};
use sim::Action;

/// A deck that can start a run (`b_challenge` needs challenge rules, which
/// are not simulated).
pub(crate) fn parse_back(key: &str) -> PyResult<Back> {
    match Back::from_key(key) {
        Some(Back::Challenge) => Err(PyValueError::new_err("b_challenge needs challenge rules, which are not simulated")),
        Some(b) => Ok(b),
        None => Err(PyValueError::new_err(format!("unknown back {key:?}"))),
    }
}

fn phase_name(p: Phase) -> &'static str {
    match p {
        Phase::BlindSelect => "blind_select",
        Phase::Round => "round",
        Phase::Shop => "shop",
        Phase::Pack => "pack",
        Phase::RoundEval => "round_eval",
        Phase::GameOver => "game_over",
    }
}

/// A seeded run. `clone()` is a full, cheap snapshot (all RNG streams
/// included), suitable for lookahead and tree search.
#[pyclass(module = "antezero_sim")]
#[derive(Clone)]
struct Game {
    g: SimGame,
    buf: Vec<Action>,
}

impl Game {
    pub(crate) fn from_sim(mut g: SimGame) -> Game {
        g.set_tracing(false);
        Game { g, buf: Vec::with_capacity(256) }
    }

    pub(crate) fn inner(&self) -> &SimGame {
        &self.g
    }

    fn apply(&mut self, a: Action) -> PyResult<()> {
        self.g.step(a).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// The ops `encode(candidates)` lists for the current state.
    fn current_ops(&mut self, candidates: usize) -> PyResult<Vec<Op>> {
        check_candidates(candidates)?;
        self.g.update_derived();
        let mut v = Vec::new();
        obs::ops(&self.g, &Layout::of(&self.g), candidates, &mut v);
        Ok(v)
    }

    /// The legal action of a factored choice over `current_ops(candidates)`.
    fn decode_factored(&mut self, op: usize, cards: u32, dest: u32, candidates: usize) -> PyResult<Action> {
        let v = self.current_ops(candidates)?;
        let o = v.get(op).ok_or_else(|| PyValueError::new_err(format!("op {op} out of range ({})", v.len())))?;
        obs::decode(&self.g, o, cards, dest).ok_or_else(|| PyValueError::new_err("illegal factored action"))
    }
}

#[pymethods]
impl Game {
    /// `Game(seed, back="b_red", stake=1, endless=False)`.
    #[new]
    #[pyo3(signature = (seed, back = "b_red", stake = 1, endless = false))]
    fn new(py: Python<'_>, seed: &str, back: &str, stake: u8, endless: bool) -> PyResult<Game> {
        let b = parse_back(back)?;
        if seed.len() > sim::MAX_SEED_LEN {
            return Err(PyValueError::new_err(format!("seeds have at most {} bytes", sim::MAX_SEED_LEN)));
        }
        if !(1..=8).contains(&stake) {
            return Err(PyValueError::new_err(format!("stake must be 1..=8, got {stake}")));
        }
        let seed = seed.to_string();
        let mut g = Game::from_sim(py.detach(move || SimGame::new_run(&seed, b, stake)));
        g.g.endless = endless;
        Ok(g)
    }

    /// The learning observation of this state (see `sim::obs`): a dict with
    /// `tok_i [MAX_TOKENS, TOK_INT]`, `tok_f [MAX_TOKENS, TOK_FLOAT]`,
    /// `glob_i [GLOB_INT]`, `glob_f [GLOB_FLOAT]`, `ops [MAX_OPS, OP_INT]`,
    /// `ops_f [MAX_OPS, OP_FLOAT]`, `n_ops`, `n_tokens`, `n_hand`; with up to
    /// `play_candidates` (at most `MAX_CANDIDATES`) scored plays.
    #[pyo3(signature = (play_candidates = 0, deck_tokens = false))]
    fn encode<'py>(&mut self, py: Python<'py>, play_candidates: usize, deck_tokens: bool) -> PyResult<Bound<'py, PyDict>> {
        use sim::obs::*;
        check_candidates(play_candidates)?;
        let r = py.detach(|| {
            let mut r = ObsRow::new();
            r.rows().write(&mut self.g, play_candidates, deck_tokens, &mut Vec::new());
            r
        });
        let d = PyDict::new(py);
        d.set_item("tok_i", PyArray1::from_vec(py, r.ti).reshape([MAX_TOKENS, TOK_INT])?)?;
        d.set_item("tok_f", PyArray1::from_vec(py, r.tf).reshape([MAX_TOKENS, TOK_FLOAT])?)?;
        d.set_item("glob_i", PyArray1::from_vec(py, r.gi))?;
        d.set_item("glob_f", PyArray1::from_vec(py, r.gf))?;
        d.set_item("ops", PyArray1::from_vec(py, r.oa).reshape([MAX_OPS, OP_INT])?)?;
        d.set_item("ops_f", PyArray1::from_vec(py, r.of).reshape([MAX_OPS, OP_FLOAT])?)?;
        d.set_item("n_ops", r.n_ops)?;
        d.set_item("n_tokens", r.n_tokens)?;
        d.set_item("n_hand", r.n_hand)?;
        Ok(d)
    }

    /// Applies the factored choice `(op index, card mask, destination)` (as
    /// produced by a policy over `encode(play_candidates)`); `ValueError` if
    /// illegal.
    #[pyo3(signature = (op, cards, dest, play_candidates = 0))]
    fn step_factored(&mut self, py: Python<'_>, op: usize, cards: u32, dest: u32, play_candidates: usize) -> PyResult<()> {
        py.detach(|| {
            let a = self.decode_factored(op, cards, dest, play_candidates)?;
            self.apply(a)
        })
    }

    /// Resamples the hidden state (deck order, face-down cards, future
    /// random outcomes, an unseen voucher); the visible state is unchanged.
    /// For determinized lookahead.
    fn resample_hidden(&mut self, seed: &str) {
        self.g.resample_hidden(seed);
    }

    /// The action code of a factored choice (see `encode`), or `None` if
    /// it is not legal.
    #[pyo3(signature = (op, cards, dest, play_candidates = 0))]
    fn factored_to_code(&mut self, py: Python<'_>, op: usize, cards: u32, dest: u32, play_candidates: usize) -> PyResult<Option<String>> {
        check_candidates(play_candidates)?;
        Ok(py.detach(|| self.decode_factored(op, cards, dest, play_candidates)).ok().map(|a| a.to_string()))
    }

    /// Blinds defeated or skipped so far.
    #[getter]
    fn blinds_cleared(&self) -> u32 {
        self.g.blinds_cleared
    }

    /// Whether the run was won (the final boss beaten). Use this rather
    /// than `won`, which the game also sets when the final boss is lost.
    #[getter]
    fn run_won(&self) -> bool {
        self.g.run_won()
    }

    /// The shaping potential `w_blind * blinds_cleared + w_score * round
    /// progress` used by `VecEnv` (0 must be used for terminal states).
    #[pyo3(signature = (w_blind = 0.04, w_score = 0.02))]
    fn potential(&self, w_blind: f32, w_score: f32) -> f32 {
        vecenv::potential(&self.g, w_blind, w_score)
    }

    #[getter]
    fn endless(&self) -> bool {
        self.g.endless
    }

    #[setter]
    fn set_endless(&mut self, v: bool) {
        self.g.endless = v;
    }

    /// Applies an action code (see `legal_codes`). `ValueError` if the code
    /// does not parse, `RuntimeError` if the action is not legal now.
    fn step(&mut self, py: Python<'_>, code: &str) -> PyResult<()> {
        let a: Action = code.parse().map_err(|e: sim::ParseError| PyValueError::new_err(e.to_string()))?;
        py.detach(|| self.apply(a))
    }

    /// Applies the `i`-th action of `legal_codes()` (cheaper than codes).
    fn step_index(&mut self, py: Python<'_>, i: usize) -> PyResult<()> {
        self.g.legal_actions(&mut self.buf);
        let a = *self.buf.get(i).ok_or_else(|| PyValueError::new_err(format!("action index {i} out of range ({})", self.buf.len())))?;
        py.detach(|| self.apply(a))
    }

    /// Applies a scenario injection code (`inject_joker ...` etc.).
    fn inject(&mut self, py: Python<'_>, code: &str) -> PyResult<()> {
        let i: Inject = code.parse().map_err(|e: sim::ParseError| PyValueError::new_err(e.to_string()))?;
        py.detach(|| self.g.inject(&i));
        Ok(())
    }

    /// Codes of all legal actions, in a stable order.
    fn legal_codes(&mut self) -> Vec<String> {
        self.g.legal_actions(&mut self.buf);
        self.buf.iter().map(|a| a.to_string()).collect()
    }

    /// Number of legal actions.
    fn num_legal(&mut self) -> usize {
        self.g.legal_actions(&mut self.buf);
        self.buf.len()
    }

    /// Flat `key -> value` view of the full state (incl. RNG streams), the
    /// same shape the oracle and the real-game bridge produce.
    fn canonical<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let c = sim::canon::from_sim(&mut self.g);
        let d = PyDict::new(py);
        for (k, v) in c {
            d.set_item(k, v)?;
        }
        Ok(d)
    }

    fn is_terminal(&self) -> bool {
        self.g.is_terminal()
    }

    /// Plays uniformly random legal actions (seeded) until the game ends or
    /// `max_steps` actions were taken; returns the number of steps.
    #[pyo3(signature = (policy_seed, max_steps = 10_000))]
    fn random_rollout(&mut self, py: Python<'_>, policy_seed: u64, max_steps: usize) -> PyResult<usize> {
        py.detach(|| {
            let mut x = policy_seed | 1;
            let mut n = 0;
            while n < max_steps && !self.g.is_terminal() {
                self.g.legal_actions(&mut self.buf);
                if self.buf.is_empty() {
                    break;
                }
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                let a = self.buf[(x % self.buf.len() as u64) as usize];
                self.apply(a)?;
                n += 1;
            }
            Ok(n)
        })
    }

    fn clone(&self) -> Game {
        Clone::clone(self)
    }

    fn __copy__(&self) -> Game {
        Clone::clone(self)
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Game {
        Clone::clone(self)
    }

    #[getter]
    fn phase(&self) -> &'static str {
        phase_name(self.g.phase)
    }

    #[getter]
    fn ante(&self) -> i32 {
        self.g.ante
    }

    #[getter]
    fn round(&self) -> i32 {
        self.g.round
    }

    #[getter]
    fn dollars(&self) -> f64 {
        self.g.dollars
    }

    #[getter]
    fn chips(&self) -> f64 {
        self.g.chips
    }

    #[getter]
    fn blind_chips(&self) -> f64 {
        self.g.blind.chips
    }

    /// The game's `G.GAME.won` (also set when the final boss is lost; see
    /// `run_won`).
    #[getter]
    fn won(&self) -> bool {
        self.g.won
    }

    #[getter]
    fn seed(&self) -> String {
        self.g.rng.seed().to_string()
    }

    /// Deck (`Back`) key, e.g. `"b_red"`.
    #[getter]
    fn back(&self) -> &'static str {
        self.g.back.key()
    }

    #[getter]
    fn stake(&self) -> u8 {
        self.g.stake
    }

    fn __repr__(&self) -> String {
        format!(
            "Game(seed={:?}, back={:?}, stake={}, ante={}, phase={:?}, dollars={})",
            self.g.rng.seed(),
            self.g.back.key(),
            self.g.stake,
            self.g.ante,
            phase_name(self.g.phase),
            self.g.dollars
        )
    }
}

/// Deck (`Back`) keys accepted by `Game` and `VecEnv`.
#[pyfunction]
fn backs() -> Vec<&'static str> {
    sim::data::ALL_BACKS.iter().filter(|&&b| b != Back::Challenge).map(|b| b.key()).collect()
}

#[pymodule]
fn antezero_sim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Game>()?;
    m.add_class::<vecenv::VecEnv>()?;
    m.add_function(wrap_pyfunction!(backs, m)?)?;
    m.add("MAX_TOKENS", obs::MAX_TOKENS)?;
    m.add("TOK_INT", obs::TOK_INT)?;
    m.add("TOK_FLOAT", obs::TOK_FLOAT)?;
    m.add("GLOB_INT", obs::GLOB_INT)?;
    m.add("GLOB_FLOAT", obs::GLOB_FLOAT)?;
    m.add("MAX_OPS", obs::MAX_OPS)?;
    m.add("OP_INT", obs::OP_INT)?;
    m.add("OP_FLOAT", obs::OP_FLOAT)?;
    m.add("MAX_CANDIDATES", obs::MAX_CANDIDATES)?;
    m.add("TOK_VOCAB", obs::TOK_VOCAB.to_vec())?;
    m.add("GLOB_VOCAB", obs::GLOB_VOCAB.to_vec())?;
    m.add("OP_PLAY_TOP", obs::op::PLAY_TOP)?;
    m.add("MAX_HAND", obs::MAX_HAND)?;
    m.add("N_ZONES", obs::zone::COUNT)?;
    m.add("N_CENTERS", obs::center::COUNT)?;
    m.add("N_OPS_KINDS", obs::op::COUNT)?;
    m.add("OP_PLAY", obs::op::PLAY)?;
    m.add("OP_DISCARD", obs::op::DISCARD)?;
    m.add("OP_MOVE_HAND", obs::op::MOVE_HAND)?;
    m.add("OP_MOVE_JOKER", obs::op::MOVE_JOKER)?;
    m.add("PREVIEW_BUDGET", obs::PREVIEW_BUDGET)?;
    m.add("PREVIEW_SAMPLES", obs::PREVIEW_SAMPLES)?;
    Ok(())
}
