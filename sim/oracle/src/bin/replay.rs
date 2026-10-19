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

use oracle::canon::{self, TimelineDiff};
use oracle::diff::{self, Checks, Inject, Mismatch, action_to_lua, inject_to_lua, start_to_lua};
use oracle::{Oracle, json};
use sim::actions::Action;
use sim::data::Back;
use sim::game::Game;
use std::process::ExitCode;

/// A history line: an action or an injection.
enum Line {
    Act(Action),
    Inject(Inject),
}

fn run() -> Result<bool, String> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).ok_or("usage: replay <history file> [filter-prefix ...]")?;
    let filters: Vec<&str> = args[2..].iter().map(|s| s.as_str()).collect();
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut lines = text.lines();
    let header = lines.next().ok_or("empty history")?;
    let (seed, back, stake) = match header.split_whitespace().collect::<Vec<_>>()[..] {
        ["start", seed, back, stake] => {
            let b = Back::from_key(back).ok_or_else(|| format!("unknown back {back:?}"))?;
            let s: u8 = stake.parse().ok().filter(|s| (1..=8).contains(s)).ok_or_else(|| format!("bad stake {stake:?}"))?;
            (seed.to_string(), b, s)
        }
        _ => return Err(format!("bad header {header:?} (expected `start <seed> <back> <stake>`)")),
    };
    let steps: Vec<(&str, Line)> = lines
        .enumerate()
        .map(|(i, l)| {
            let parsed = l.parse::<Action>().map(Line::Act).or_else(|_| l.parse::<Inject>().map(Line::Inject));
            parsed.map(|p| (l, p)).map_err(|e| format!("line {}: {e}", i + 2))
        })
        .collect::<Result<_, _>>()?;
    let checks = Checks::from_env()?;
    let hook_step: Option<usize> = match std::env::var("REPLAY_STEP") {
        Err(_) => None,
        Ok(s) => Some(s.parse().map_err(|_| format!("bad REPLAY_STEP {s:?}"))?),
    };
    let hook_code = match hook_step {
        Some(_) => {
            let f = std::env::var("REPLAY_LUA").map_err(|_| "REPLAY_STEP needs REPLAY_LUA (a Lua file)")?;
            Some(std::fs::read_to_string(&f).map_err(|e| format!("{f}: {e}"))?)
        }
        None => None,
    };
    let show_frames = std::env::var("REPLAY_FRAMES").is_ok_and(|v| v == "1");

    let o = Oracle::new()?;
    let mut g = Game::new_run(&seed, back, stake);
    let mut lua = o.call("oracle", &start_to_lua(&seed, back, stake))?;
    let start = [("<start>", None)];
    for (step, (line, cmd)) in start.into_iter().chain(steps.iter().map(|(l, p)| (*l, Some(p)))).enumerate() {
        if Some(step) == hook_step {
            o.exec(hook_code.as_deref().unwrap_or_default()).map_err(|e| format!("hook: {e}"))?;
            g.enable_evlog();
        }
        match cmd {
            None => {}
            Some(Line::Act(a)) => {
                if let Err(e) = g.step(*a) {
                    println!("DIVERGED at step {step}: sim rejected {a:?}: {e}");
                    return Ok(false);
                }
                lua = o.call("oracle", &action_to_lua(a)).map_err(|e| format!("oracle: {e}"))?;
            }
            Some(Line::Inject(i)) => {
                g.inject(i);
                lua = o.call("oracle", &inject_to_lua(i)).map_err(|e| format!("oracle: {e}"))?;
            }
        }
        if Some(step) == hook_step {
            println!("{}", o.eval("return table.concat(H.log or {}, string.char(10))")?);
            for l in g.evlog() {
                println!("SIM {l}");
            }
        }
        let j = json::parse(&lua).map_err(|e| format!("json: {e}"))?;
        if !filters.is_empty() {
            let (a, b) = (canon::from_sim(&mut g), canon::from_lua(&j));
            println!("--- step {step}: {line}");
            for (k, v) in &a {
                if filters.iter().any(|f| k.starts_with(f)) {
                    let w = b.get(k).map(|s| s.as_str()).unwrap_or("<missing>");
                    let mark = if w == v { " " } else { "!" };
                    println!("{mark} {k}: sim={v} game={w}");
                }
            }
        }
        if show_frames {
            println!("--- step {step}: {line}");
            for l in canon::timeline_frames(&g, &j) {
                println!("{l}");
            }
        }
        let diffs = match diff::compare(&mut g, &j, checks.timeline) {
            None => continue,
            Some(Mismatch::Timeline(TimelineDiff::Frame(m))) if !checks.frames => {
                println!("step {step}: frame offset (not failed): {m}");
                continue;
            }
            Some(Mismatch::State(d)) => d,
            Some(Mismatch::Timeline(TimelineDiff::Order(k, s, l))) => vec![(k, s, l)],
            Some(Mismatch::Timeline(TimelineDiff::Frame(m))) => vec![("timeline frames".into(), m, String::new())],
        };
        println!("DIVERGED at step {step}: {line}");
        for (k, s, l) in diffs {
            println!("  {k}: sim={s} game={l}");
        }
        return Ok(false);
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
