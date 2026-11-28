//! Replays a recorded history in both engines and reports the first
//! divergence, with the same checks as `difftest` (`ORACLE_TIMELINE`,
//! `ORACLE_FRAMES`). Exits with status 1 on a divergence.
//!
//! Usage: `replay <history file> [filter-prefix ...]`. With filter prefixes,
//! prints matching canonical fields of both engines at every step (handy for
//! watching e.g. `jokers.` or `dollars`). `REPLAY_FRAMES=1` prints both
//! timelines at every step; `REPLAY_STEP=<n>` with `REPLAY_LUA=<file>` runs
//! that Lua code in the oracle before step `n` and prints its `H.log` and the
//! simulator's event log after it.

use std::env::{self, VarError};
use std::fs;
use std::process::ExitCode;

use oracle::command::{Command, Start};
use oracle::difftest::{self, Checks, Comparison};
use oracle::{Oracle, canon};
use serde_json::Value;
use sim::game::Game;

/// A recorded run: its start's code, then one command code per line.
struct History<'text> {
	start: Start<'text>,
	/// Each line's text and what it parses to.
	steps: Vec<(&'text str, Command)>,
}

fn parse_history(text: &str) -> Result<History<'_>, String> {
	let mut lines = text.lines();
	let start = Start::parse(lines.next().ok_or("empty history")?)?;
	let steps = lines
		.enumerate()
		.map(|(i, line)| {
			line.parse::<Command>()
				.map(|command| (line, command))
				.map_err(|e| format!("line {}: {e}", i + 2))
		})
		.collect::<Result<_, _>>()?;
	Ok(History { start, steps })
}

/// Lua code to run in the oracle before one step (`REPLAY_STEP`,
/// `REPLAY_LUA`); both engines' event logs are printed after that step.
struct Hook {
	step: usize,
	code: String,
}

impl Hook {
	fn from_env() -> Result<Option<Hook>, String> {
		let step: usize = match env::var("REPLAY_STEP") {
			Err(VarError::NotPresent) => return Ok(None),
			Err(VarError::NotUnicode(s)) => {
				return Err(format!("bad REPLAY_STEP {}", s.display()));
			}
			Ok(s) => s.parse().map_err(|_| format!("bad REPLAY_STEP {s:?}"))?,
		};
		let file = env::var_os("REPLAY_LUA").ok_or("REPLAY_STEP needs REPLAY_LUA (a Lua file)")?;
		let code =
			fs::read_to_string(&file).map_err(|e| format!("{}: {e}", file.to_string_lossy()))?;
		Ok(Some(Hook { step, code }))
	}
}

/// Prints the canonical fields that either engine has whose path starts
/// with one of `filters`, marking the ones that differ.
fn print_fields(g: &mut Game, dump: &Value, filters: &[&str]) {
	let (sim, game) = (canon::from_sim(g), canon::from_lua(dump));
	let shown =
		canon::fields(&sim, &game).filter(|f| filters.iter().any(|p| f.path.starts_with(p)));
	for field in shown {
		let mark = if field.differs() { "!" } else { " " };
		println!("{mark} {field}");
	}
}

fn run() -> Result<bool, String> {
	let args: Vec<String> = env::args().collect();
	let path = args
		.get(1)
		.ok_or("usage: replay <history file> [filter-prefix ...]")?;
	let filters: Vec<&str> = args[2..].iter().map(String::as_str).collect();
	let text = fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
	let history = parse_history(&text)?;
	let checks = Checks::from_env()?;
	let hook = Hook::from_env()?;
	let show_frames = difftest::env_flag("REPLAY_FRAMES")?.unwrap_or(false);

	let mut oracle = Oracle::new()?;
	let mut g = history.start.new_game();
	let mut lua = oracle.command(&history.start.to_lua())?;
	let start = [("<start>", None)];
	let commands = start.into_iter().chain(
		history
			.steps
			.iter()
			.map(|&(text, command)| (text, Some(command))),
	);
	for (step, (text, command)) in commands.enumerate() {
		let hooked = hook.as_ref().filter(|h| h.step == step);
		if let Some(hook) = hooked {
			oracle.exec(&hook.code).map_err(|e| format!("hook: {e}"))?;
			g.enable_event_log();
		}
		if let Some(command) = command {
			if let Err(e) = command.apply(&mut g) {
				println!("DIVERGED at step {step}: sim rejected {command:?}: {e}");
				return Ok(false);
			}
			lua = oracle
				.command(&command.to_lua())
				.map_err(|e| format!("oracle: {e}"))?;
		}
		if hooked.is_some() {
			println!(
				"{}",
				oracle.eval("return table.concat(H.log or {}, string.char(10))")?
			);
			for entry in g.event_log() {
				println!("SIM {entry}");
			}
		}
		let dump: Value = serde_json::from_str(&lua).map_err(|e| format!("json: {e}"))?;
		if !filters.is_empty() {
			println!("--- step {step}: {text}");
			print_fields(&mut g, &dump, &filters);
		}
		if show_frames {
			println!("--- step {step}: {text}");
			for row in canon::timeline_frames(&g, &dump) {
				println!("{row}");
			}
		}
		match difftest::compare(&mut g, &dump, checks) {
			Comparison::Match => {}
			Comparison::FrameOffset(offset) => {
				println!("step {step}: frame offset (not failed): {offset}");
			}
			Comparison::Diverged(mismatch) => {
				print!("DIVERGED at step {step}: {text}\n{mismatch}");
				return Ok(false);
			}
		}
	}
	println!("no divergence ({checks})");
	Ok(true)
}

fn main() -> ExitCode {
	match run() {
		Ok(true) => ExitCode::SUCCESS,
		Ok(false) => ExitCode::FAILURE,
		Err(e) => {
			eprintln!("replay: {e}");
			ExitCode::from(2)
		}
	}
}
