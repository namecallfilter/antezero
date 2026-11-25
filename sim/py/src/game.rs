use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use sim::data::{Back, Center};
use sim::game::Game as SimGame;
use sim::inject::Inject;
use sim::obs::{self, BuildPreview, Layout, Op, OpOptions};
use sim::rng::Xorshift;
use sim::{Action, CardMask, Stake, canon, json};

use crate::rows::{
	OPS_F_SHAPE, OPS_SHAPE, ObsOptions, ObsRow, TOK_F_SHAPE, TOK_I_SHAPE, op_options,
};
use crate::slot;

pub(crate) fn parse_back(key: &str) -> PyResult<Back> {
	match Back::from_key(key) {
		Some(b) if b.is_simulated() => Ok(b),
		Some(_) => Err(PyValueError::new_err(format!(
			"{key} needs challenge rules, which are not simulated"
		))),
		None => Err(PyValueError::new_err(format!("unknown back {key:?}"))),
	}
}

/// A seeded run. `clone()` is a full, cheap snapshot (all RNG streams
/// included), suitable for lookahead and tree search.
#[pyclass(module = "antezero_sim")]
pub(crate) struct Game {
	pub(crate) g: SimGame,
	/// Scratch for the legal actions.
	buf: Vec<Action>,
	/// Scratch for the current ops.
	ops: Vec<Op>,
	/// This game's recent build previews (see `sim::obs::build_preview`), kept between `encode` calls.
	previews: Vec<BuildPreview>,
	/// Whether `encode` and the factored choices offer skipping blinds.
	pub(crate) allow_skip: bool,
	/// Whether they offer sorts that change nothing.
	pub(crate) null_sorts: bool,
}

impl From<SimGame> for Game {
	fn from(mut g: SimGame) -> Game {
		g.set_tracing(false);
		Game {
			g,
			buf: Vec::with_capacity(256),
			ops: Vec::new(),
			previews: Vec::new(),
			allow_skip: true,
			null_sorts: true,
		}
	}
}

/// The ops `game` offers, with `candidates` scored plays.
fn options_of(game: &Game, candidates: usize) -> PyResult<OpOptions> {
	op_options(OpOptions {
		candidates,
		allow_skip: game.allow_skip,
		null_sorts: game.null_sorts,
	})
}

impl Clone for Game {
	/// A copy of the run and of its preview cache; the scratch buffers start
	/// empty.
	fn clone(&self) -> Game {
		Game {
			g: self.g.clone(),
			buf: Vec::new(),
			ops: Vec::new(),
			previews: self.previews.clone(),
			allow_skip: self.allow_skip,
			null_sorts: self.null_sorts,
		}
	}
}

fn apply(g: &mut SimGame, a: Action) -> PyResult<()> {
	g.step(a)
		.map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Fills `ops` with the ops `encode` lists for the current state.
fn current_ops(g: &mut SimGame, options: OpOptions, ops: &mut Vec<Op>) {
	g.update_derived();
	obs::ops(g, &Layout::of(&g.areas, &g.tags), options, ops);
}

/// The action of a factored choice, or `None` if it is not legal; `ops`
/// receives the current ops.
///
/// # Errors
///
/// Returns an error if the state has no op `op`.
fn decode_factored(
	g: &mut SimGame, ops: &mut Vec<Op>, op: usize, cards: u32, dest: u32, options: OpOptions,
) -> PyResult<Option<Action>> {
	current_ops(g, options, ops);
	let o = ops
		.get(op)
		.ok_or_else(|| PyValueError::new_err(format!("op {op} out of range ({})", ops.len())))?;
	let dest = u8::try_from(dest).ok();
	Ok(obs::decode(g, o, CardMask::from_bits(cards), dest))
}

#[pymethods]
impl Game {
	#[new]
	#[pyo3(signature = (seed, back = "b_red", stake = 1, endless = false))]
	fn new(py: Python<'_>, seed: &str, back: &str, stake: u8, endless: bool) -> PyResult<Game> {
		let b = parse_back(back)?;
		if seed.len() > sim::MAX_SEED_LEN {
			return Err(PyValueError::new_err(format!(
				"seeds have at most {} bytes",
				sim::MAX_SEED_LEN
			)));
		}
		let stake = Stake::try_from(stake).map_err(|e| PyValueError::new_err(e.to_string()))?;
		let mut g = Game::from(py.detach(|| SimGame::new_run(seed, b, stake)));
		g.g.setup.endless = endless;
		Ok(g)
	}

	/// The learning observation of this state (see `sim::obs`): a dict with
	/// `tok_i [MAX_TOKENS, TOK_INT]`, `tok_f [MAX_TOKENS, TOK_FLOAT]`,
	/// `glob_i [GLOB_INT]`, `glob_f [GLOB_FLOAT]`, `ops [MAX_OPS, OP_INT]`,
	/// `ops_f [MAX_OPS, OP_FLOAT]`, `n_ops`, `n_tokens`, `n_hand`; with up to
	/// `play_candidates` (at most `MAX_CANDIDATES`) scored plays, with
	/// `build_previews` the build previews of shops, packs and blind select,
	/// and with `order_gain` the gain column of the joker-move ops.
	#[pyo3(signature = (play_candidates = 0, deck_tokens = false, build_previews = true, order_gain = true))]
	#[expect(
		clippy::fn_params_excessive_bools,
		reason = "the Python API takes the flags, by position or by keyword; pyo3 maps each Python flag to a `bool`"
	)]
	fn encode<'py>(
		&mut self, py: Python<'py>, play_candidates: usize, deck_tokens: bool,
		build_previews: bool, order_gain: bool,
	) -> PyResult<Bound<'py, PyDict>> {
		let options = ObsOptions {
			ops: options_of(self, play_candidates)?,
			deck_tokens,
			previews: build_previews,
			order_gain,
		};
		let r = py.detach(|| {
			let mut r = ObsRow::default();
			r.rows()
				.write(&mut self.g, options, &mut self.ops, &mut self.previews);
			r
		});
		let d = PyDict::new(py);
		d.set_item(
			"tok_i",
			PyArray1::from_vec(py, r.obs.tok_i).reshape(TOK_I_SHAPE)?,
		)?;
		d.set_item(
			"tok_f",
			PyArray1::from_vec(py, r.obs.tok_f).reshape(TOK_F_SHAPE)?,
		)?;
		d.set_item("glob_i", PyArray1::from_vec(py, r.obs.glob_i))?;
		d.set_item("glob_f", PyArray1::from_vec(py, r.obs.glob_f))?;
		d.set_item("ops", PyArray1::from_vec(py, r.ops).reshape(OPS_SHAPE)?)?;
		d.set_item(
			"ops_f",
			PyArray1::from_vec(py, r.ops_f).reshape(OPS_F_SHAPE)?,
		)?;
		d.set_item("n_ops", r.n_ops)?;
		d.set_item("n_tokens", r.n_tokens)?;
		d.set_item("n_hand", r.n_hand)?;
		Ok(d)
	}

	/// Applies the factored choice `(op index, card mask, destination)` (as
	/// produced by a policy over `encode(play_candidates)`); `ValueError` if
	/// illegal.
	#[pyo3(signature = (op, cards, dest, play_candidates = 0))]
	fn step_factored(
		&mut self, py: Python<'_>, op: usize, cards: u32, dest: u32, play_candidates: usize,
	) -> PyResult<()> {
		let options = options_of(self, play_candidates)?;
		py.detach(|| {
			let a = decode_factored(&mut self.g, &mut self.ops, op, cards, dest, options)?
				.ok_or_else(|| PyValueError::new_err("illegal factored action"))?;
			apply(&mut self.g, a)
		})
	}

	/// Replaces the run with the one in `state`, a state dump of the real
	/// game (the mod's `az_state`, or `state_json`) taken at a decision
	/// point; this game's settings (`endless`, `steamodded`, `allow_skip`,
	/// `null_sorts`) stay, and no free rearranges have been made (see `rearranges`).
	/// `ValueError` if the dump is not one the simulator can take.
	fn load_state(&mut self, py: Python<'_>, state: &str) -> PyResult<()> {
		let mut g = py
			.detach(|| {
				let dump = json::parse(state).map_err(|e| format!("state dump: {e}"))?;
				SimGame::from_dump(&dump).map_err(|e| e.to_string())
			})
			.map_err(PyValueError::new_err)?;
		g.set_tracing(false);
		g.setup.endless = self.g.setup.endless;
		g.setup.steamodded = self.g.setup.steamodded;
		self.g = g;
		Ok(())
	}

	/// The state dump the real game would give in this state (JSON, the
	/// shape of the mod's `az_state`).
	fn state_json(&mut self) -> String {
		self.g.to_dump().to_string()
	}

	/// Free rearranges (hand and joker moves, sorts) since the last other
	/// action: `encode` stops offering them at `sim::obs::MAX_REARRANGES`.
	#[getter]
	fn rearranges(&self) -> u8 {
		self.g.flow.rearranges
	}

	#[setter]
	fn set_rearranges(&mut self, n: u8) {
		self.g.flow.rearranges = n;
	}

	/// Resamples the hidden state (deck order, face-down cards, future
	/// random outcomes, an unseen voucher); the visible state is unchanged.
	/// For determinized lookahead.
	fn resample_hidden(&mut self, seed: &str) {
		self.g.resample_hidden(seed);
	}

	/// The action code of a factored choice (see `encode`), or `None` if
	/// it is not legal. Raises `ValueError` if the state has no op `op`.
	#[pyo3(signature = (op, cards, dest, play_candidates = 0))]
	fn factored_to_code(
		&mut self, py: Python<'_>, op: usize, cards: u32, dest: u32, play_candidates: usize,
	) -> PyResult<Option<String>> {
		let options = options_of(self, play_candidates)?;
		let action =
			py.detach(|| decode_factored(&mut self.g, &mut self.ops, op, cards, dest, options))?;
		Ok(action.map(|a| a.to_string()))
	}

	/// Blinds defeated or skipped so far.
	#[getter]
	fn blinds_cleared(&self) -> u32 {
		self.g.run.blinds_cleared
	}

	/// Whether the run was won (the final boss beaten). Use this rather
	/// than `won`, which the game also sets when the final boss is lost.
	#[getter]
	fn run_won(&self) -> bool {
		self.g.run_won()
	}

	/// The shaping potential `w_blind * run position in blinds + w_score *
	/// round progress` used by `VecEnv` (also at terminal states, unless
	/// `terminal_phi=False`).
	#[pyo3(signature = (w_blind = 0.04, w_score = 0.02))]
	fn potential(&self, w_blind: f32, w_score: f32) -> f32 {
		slot::potential(&self.g, w_blind, w_score)
	}

	#[getter]
	fn endless(&self) -> bool {
		self.g.setup.endless
	}

	#[setter]
	fn set_endless(&mut self, v: bool) {
		self.g.setup.endless = v;
	}

	/// Whether `encode` and `step_factored` offer skipping blinds (observation
	/// only: the game itself always allows skips). `VecEnv.game` sets it from
	/// the env's `allow_skip`; a policy trained without skips must play with
	/// it off.
	#[getter]
	fn allow_skip(&self) -> bool {
		self.allow_skip
	}

	#[setter]
	fn set_allow_skip(&mut self, v: bool) {
		self.allow_skip = v;
	}

	/// Whether `encode` and `step_factored` offer a sort that would move no
	/// card the player can tell apart. Like `allow_skip`, part of the action
	/// set a policy was trained on.
	#[getter]
	fn null_sorts(&self) -> bool {
		self.null_sorts
	}

	#[setter]
	fn set_null_sorts(&mut self, v: bool) {
		self.null_sorts = v;
	}

	/// Whether ×Mult is rounded like Steamodded (`mult + mult * (x - 1)`),
	/// which the real game runs under the bridge, instead of like vanilla
	/// (`mult * x`). The two can differ by one floored chip.
	#[getter]
	fn steamodded(&self) -> bool {
		self.g.setup.steamodded
	}

	#[setter]
	fn set_steamodded(&mut self, v: bool) {
		self.g.setup.steamodded = v;
	}

	/// Applies an action code (see `legal_codes`). `ValueError` if the code
	/// does not parse, `RuntimeError` if the action is not legal now.
	fn step(&mut self, py: Python<'_>, code: &str) -> PyResult<()> {
		let a: Action = code
			.parse()
			.map_err(|e: sim::ParseError| PyValueError::new_err(e.to_string()))?;
		py.detach(|| apply(&mut self.g, a))
	}

	/// Applies the `i`-th action of `legal_codes()` (cheaper than codes).
	fn step_index(&mut self, py: Python<'_>, i: usize) -> PyResult<()> {
		self.g.legal_actions(&mut self.buf);
		let a = *self.buf.get(i).ok_or_else(|| {
			PyValueError::new_err(format!(
				"action index {i} out of range ({})",
				self.buf.len()
			))
		})?;
		py.detach(|| apply(&mut self.g, a))
	}

	/// Applies a scenario injection code (`inject_joker ...` etc.).
	fn inject(&mut self, py: Python<'_>, code: &str) -> PyResult<()> {
		let i: Inject = code
			.parse()
			.map_err(|e: sim::ParseError| PyValueError::new_err(e.to_string()))?;
		py.detach(|| self.g.inject(i));
		Ok(())
	}

	/// Codes of all legal actions, in a stable order.
	fn legal_codes(&mut self) -> Vec<String> {
		self.g.legal_actions(&mut self.buf);
		self.buf.iter().map(ToString::to_string).collect()
	}

	fn num_legal(&mut self) -> usize {
		self.g.legal_actions(&mut self.buf);
		self.buf.len()
	}

	/// Flat `key -> value` view of the full state (incl. RNG streams), the
	/// same shape the oracle and the real-game bridge produce.
	fn canonical<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
		let c = canon::from_sim(&mut self.g);
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
	fn random_rollout(
		&mut self, py: Python<'_>, policy_seed: u64, max_steps: usize,
	) -> PyResult<usize> {
		py.detach(|| {
			let mut rng = Xorshift::new(policy_seed);
			let mut n = 0;
			while n < max_steps && !self.g.is_terminal() {
				self.g.legal_actions(&mut self.buf);
				if self.buf.is_empty() {
					break;
				}
				let a = self.buf[rng.below(self.buf.len())];
				apply(&mut self.g, a)?;
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
		self.g.flow.phase.name()
	}

	#[getter]
	fn ante(&self) -> i32 {
		self.g.run.ante
	}

	#[getter]
	fn round(&self) -> i32 {
		self.g.run.round
	}

	#[getter]
	fn dollars(&self) -> f64 {
		self.g.run.dollars
	}

	#[getter]
	fn chips(&self) -> f64 {
		self.g.round.chips
	}

	#[getter]
	fn blind_chips(&self) -> f64 {
		self.g.blind.chips
	}

	/// The game's `G.GAME.won` (also set when the final boss is lost; see
	/// `run_won`).
	#[getter]
	fn won(&self) -> bool {
		self.g.run.won
	}

	#[getter]
	fn seed(&self) -> String {
		self.g.rng.seed().to_string()
	}

	/// Deck (`Back`) key, e.g. `"b_red"`.
	#[getter]
	fn back(&self) -> &'static str {
		self.g.setup.back.key()
	}

	#[getter]
	fn stake(&self) -> u8 {
		self.g.setup.stake.into()
	}

	fn __repr__(&self) -> String {
		format!(
			"Game(seed={:?}, back={:?}, stake={}, ante={}, phase={:?}, dollars={})",
			self.g.rng.seed(),
			self.g.setup.back.key(),
			u8::from(self.g.setup.stake),
			self.g.run.ante,
			self.g.flow.phase.name(),
			self.g.run.dollars
		)
	}
}

/// Deck (`Back`) keys accepted by `Game` and `VecEnv`.
#[pyfunction]
pub(crate) fn backs() -> Vec<&'static str> {
	Back::simulated().map(Back::key).collect()
}

/// The jokers, consumables, vouchers and boosters as `(key, center id,
/// rarity)`: the id a token's center column holds, and a joker's rarity
/// (1 common to 4 legendary; 0 for the other cards).
#[pyfunction]
pub(crate) fn centers() -> Vec<(&'static str, i32, u8)> {
	Center::all_cards()
		.map(|c| {
			let rarity = match c {
				Center::Joker(j) => j.rarity().id(),
				_ => 0,
			};
			(c.key(), obs::center::of(c), rarity)
		})
		.collect()
}
