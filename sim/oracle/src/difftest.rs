//! Randomized differential testing: play the same commands in the simulator
//! and in the real game and compare the canonical state (and the timeline of
//! RNG calls, draws and removals) after every step.

use std::env::{self, VarError};
use std::fmt;

use serde_json::Value;
use sim::actions::ActionError;
use sim::game::{Game, Phase};

use crate::Oracle;
use crate::canon::{self, Diff, FrameOffset, TimelineDiff};
use crate::command::{Command, Start};
use crate::policy::Policy;

/// What is compared besides the canonical state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Checks {
	/// Nothing else.
	StateOnly,
	/// The order of the timeline entries (RNG calls, draws, removals). An
	/// entry on another frame is only counted (see [`Stats`]).
	Timeline,
	/// The timeline's order, and every entry's frame.
	TimelineAndFrames,
}

/// The environment variable `name` as a flag (`0` or `1`), `None` if it is
/// not set.
///
/// # Errors
///
/// Returns the variable's bad value.
pub fn env_flag(name: &str) -> Result<Option<bool>, String> {
	match env::var(name) {
		Err(VarError::NotPresent) => Ok(None),
		Ok(v) if v == "1" => Ok(Some(true)),
		Ok(v) if v == "0" => Ok(Some(false)),
		Ok(v) => Err(format!("{name} must be 0 or 1, got {v:?}")),
		Err(VarError::NotUnicode(v)) => Err(format!("{name} must be 0 or 1, got {}", v.display())),
	}
}

impl Checks {
	/// From the environment: `ORACLE_TIMELINE=0` skips the timeline,
	/// `ORACLE_FRAMES=1` makes frame offsets failures.
	///
	/// # Errors
	///
	/// Returns which variable has a bad value, or that the two contradict.
	pub fn from_env() -> Result<Checks, String> {
		match (
			env_flag("ORACLE_TIMELINE")?.unwrap_or(true),
			env_flag("ORACLE_FRAMES")?.unwrap_or(false),
		) {
			(false, false) => Ok(Checks::StateOnly),
			(true, false) => Ok(Checks::Timeline),
			(true, true) => Ok(Checks::TimelineAndFrames),
			(false, true) => Err("ORACLE_FRAMES=1 needs the timeline (ORACLE_TIMELINE=1)".into()),
		}
	}
}

impl fmt::Display for Checks {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str(match self {
			Checks::StateOnly => "state only (ORACLE_TIMELINE=0)",
			Checks::Timeline => {
				"state + timeline order; frame offsets counted, not failed (ORACLE_FRAMES=1 to fail)"
			}
			Checks::TimelineAndFrames => "state + timeline order + frames",
		})
	}
}

/// How the engines disagree after a step.
#[derive(Clone, Debug, PartialEq)]
pub enum Mismatch {
	/// Canonical state differences.
	State(Vec<Diff>),
	Timeline(TimelineDiff),
}

impl fmt::Display for Mismatch {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		const SHOWN: usize = 40;
		match self {
			Mismatch::State(diffs) => {
				writeln!(f, "the state differs:")?;
				for d in diffs.iter().take(SHOWN) {
					writeln!(f, "  {d}")?;
				}
				if diffs.len() > SHOWN {
					writeln!(f, "  ... {} more", diffs.len() - SHOWN)?;
				}
				Ok(())
			}
			Mismatch::Timeline(TimelineDiff::Order(order)) => {
				writeln!(f, "the timeline order differs at {order}")
			}
			Mismatch::Timeline(TimelineDiff::Frame(offset)) => {
				writeln!(f, "the timeline frames differ: {offset}")
			}
		}
	}
}

/// How the engines compare after a step, under some [`Checks`].
#[derive(Clone, Debug, PartialEq)]
pub enum Comparison {
	/// Everything checked matches.
	Match,
	/// Everything matches but a timeline entry's frame, which these checks
	/// count without failing.
	FrameOffset(FrameOffset),
	Diverged(Mismatch),
}

/// Compares the simulator with the oracle's state dump after a step.
#[must_use]
pub fn compare(g: &mut Game, lua: &Value, checks: Checks) -> Comparison {
	let d = canon::diff(&canon::from_sim(g), &canon::from_lua(lua));
	if !d.is_empty() {
		return Comparison::Diverged(Mismatch::State(d));
	}
	if checks == Checks::StateOnly {
		return Comparison::Match;
	}
	match canon::timeline_diff(g, lua) {
		None => Comparison::Match,
		Some(TimelineDiff::Frame(offset)) if checks == Checks::Timeline => {
			Comparison::FrameOffset(offset)
		}
		Some(t) => Comparison::Diverged(Mismatch::Timeline(t)),
	}
}

/// Why a differential game stopped before its end.
#[derive(Debug)]
pub enum Cause {
	Diverged(Mismatch),
	/// The oracle failed to run a command.
	Oracle(String),
	/// The oracle's answer is not JSON.
	Json(String),
	/// The simulator rejected an action it listed as legal.
	SimRejected {
		command: Command,
		error: ActionError,
	},
}

impl fmt::Display for Cause {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Cause::Diverged(m) => fmt::Display::fmt(m, f),
			Cause::Oracle(e) => writeln!(f, "oracle: {e}"),
			Cause::Json(e) => writeln!(f, "json: {e}"),
			Cause::SimRejected { command, error } => {
				writeln!(f, "sim rejected {command:?}: {error}")
			}
		}
	}
}

/// A differential game that stopped before its end: where, why, and the
/// commands that led there.
#[derive(Debug)]
pub struct Failure {
	pub seed: String,
	pub step: usize,
	pub history: Vec<String>,
	/// Machine-readable history (the start's code, then each command's).
	pub codes: Vec<String>,
	pub cause: Cause,
	pub stats: Stats,
}

impl fmt::Display for Failure {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		writeln!(f, "seed {} diverged at step {}", self.seed, self.step)?;
		let n = self.history.len();
		for h in &self.history[n.saturating_sub(12)..] {
			writeln!(f, "  {h}")?;
		}
		fmt::Display::fmt(&self.cause, f)
	}
}

/// A step whose timeline had an entry on another frame.
#[derive(Clone, Debug)]
pub struct FrameExample {
	/// The step's code.
	pub code: String,
	pub offset: FrameOffset,
}

impl fmt::Display for FrameExample {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "{} :: {}", self.code, self.offset)
	}
}

/// What a differential game compared.
#[derive(Default, Debug, Clone)]
pub struct Stats {
	/// Commands compared (actions and injections).
	pub steps: usize,
	pub max_ante: i32,
	pub won: bool,
	/// Steps whose timeline order was compared and matched.
	pub timed_steps: usize,
	/// Of those, steps with an entry on another frame, and a few examples.
	pub frame_mismatches: usize,
	pub frame_examples: Vec<FrameExample>,
}

/// One differential game.
#[derive(Clone, Copy, Debug)]
pub struct GameSpec<'seed> {
	pub start: Start<'seed>,
	pub max_steps: usize,
	pub policy_seed: u64,
	/// Probability of a random scenario injection before each action.
	pub inject_rate: f64,
}

/// A game in progress in both engines.
struct Run<'oracle> {
	o: &'oracle mut Oracle,
	checks: Checks,
	g: Game,
	/// The step being compared (0 is the start).
	step: usize,
	history: Vec<String>,
	codes: Vec<String>,
	stats: Stats,
}

impl Run<'_> {
	/// Plays the game with `policy` for at most `max_steps` commands.
	fn play(
		&mut self, start: Start<'_>, max_steps: usize, mut policy: Policy,
	) -> Result<(), Cause> {
		let lua = start.to_lua();
		let code = start.to_string();
		let synced = self.sync(&lua, &code);
		self.history.push(lua);
		self.codes.push(code);
		synced?;
		for step in 1..=max_steps {
			self.step = step;
			let Some(command) = policy.next(&self.g) else {
				break;
			};
			let lua = command.to_lua();
			let code = command.to_string();
			self.history.push(format!("{lua}  -- {command:?}"));
			let synced = command
				.apply(&mut self.g)
				.map_err(|error| Cause::SimRejected { command, error })
				.and_then(|()| self.sync(&lua, &code));
			self.codes.push(code);
			synced?;
			if self.g.flow.phase == Phase::GameOver {
				break;
			}
		}
		Ok(())
	}

	/// Sends `lua`, the command whose code is `code`, to the oracle (the
	/// simulator has already performed it) and compares the engines.
	fn sync(&mut self, lua: &str, code: &str) -> Result<(), Cause> {
		let answer = self.o.command(lua).map_err(Cause::Oracle)?;
		let j: Value = serde_json::from_str(&answer).map_err(|e| Cause::Json(e.to_string()))?;
		match compare(&mut self.g, &j, self.checks) {
			Comparison::Diverged(m) => return Err(Cause::Diverged(m)),
			Comparison::FrameOffset(offset) => {
				self.stats.timed_steps += 1;
				self.stats.frame_mismatches += 1;
				if self.stats.frame_examples.len() < 400 {
					self.stats.frame_examples.push(FrameExample {
						code: code.to_string(),
						offset,
					});
				}
			}
			Comparison::Match if self.checks != Checks::StateOnly => self.stats.timed_steps += 1,
			Comparison::Match => {}
		}
		self.stats.steps = self.step;
		self.stats.max_ante = self.stats.max_ante.max(self.g.run.ante);
		self.stats.won |= self.g.run_won();
		Ok(())
	}
}

/// Plays one game in both engines with the test policy, comparing after
/// every action and injection.
///
/// # Errors
///
/// Returns the first divergence (or oracle failure) with the game's history.
pub fn run_game(
	o: &mut Oracle, spec: &GameSpec<'_>, checks: Checks,
) -> Result<Stats, Box<Failure>> {
	let GameSpec {
		start,
		max_steps,
		policy_seed,
		inject_rate,
	} = *spec;
	let g = start.new_game();
	let mut run = Run {
		o,
		checks,
		stats: Stats {
			max_ante: g.run.ante,
			..Stats::default()
		},
		g,
		step: 0,
		history: Vec::new(),
		codes: Vec::new(),
	};
	match run.play(start, max_steps, Policy::new(policy_seed, inject_rate)) {
		Ok(()) => Ok(run.stats),
		Err(cause) => Err(Box::new(Failure {
			seed: start.seed.to_string(),
			step: run.step,
			history: run.history,
			codes: run.codes,
			cause,
			stats: run.stats,
		})),
	}
}
