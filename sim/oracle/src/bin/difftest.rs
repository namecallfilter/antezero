//! Randomized differential test against the real game.
//!
//! Usage: `difftest [games=20] [first_seed=1] [max_steps=300] [back=b_red]
//! [stake=1] [inject_rate=0]`. Game `n` uses seed `T<first_seed + n>`. The
//! checks follow `ORACLE_TIMELINE` / `ORACLE_FRAMES` (see `diff::Checks`).
//! The history of every diverging game is written to
//! `.scratch/fail_<seed>_<back>_<stake>.txt` (replay it with `replay`).
//! Exits with status 1 if any game diverged.

use oracle::diff::{self, Checks, GameSpec};
use oracle::Oracle;
use sim::data::Back;
use std::process::ExitCode;

const USAGE: &str = "usage: difftest [games] [first_seed] [max_steps] [back] [stake] [inject_rate]";

fn arg<T: std::str::FromStr>(a: &[String], i: usize, name: &str, default: T) -> Result<T, String> {
    match a.get(i) {
        None => Ok(default),
        Some(s) => s.parse().map_err(|_| format!("bad {name} {s:?}")),
    }
}

fn run() -> Result<bool, String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() > 6 {
        return Err("too many arguments".into());
    }
    let games: usize = arg(&a, 0, "games", 20)?;
    let first: u64 = arg(&a, 1, "first_seed", 1)?;
    let max_steps: usize = arg(&a, 2, "max_steps", 300)?;
    let back = match a.get(3) {
        None => Back::Red,
        Some(k) => Back::from_key(k).filter(|&b| b != Back::Challenge).ok_or_else(|| format!("bad back {k:?}"))?,
    };
    let stake: u8 = arg(&a, 4, "stake", 1)?;
    let inject_rate: f64 = arg(&a, 5, "inject_rate", 0.0)?;
    if !(1..=8).contains(&stake) || !(0.0..=1.0).contains(&inject_rate) {
        return Err("stake must be 1..=8 and inject_rate in [0, 1]".into());
    }
    let checks = Checks::from_env()?;
    let o = Oracle::new()?;
    let out_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.scratch");
    println!("checks: {checks}");
    let (mut ok, mut steps, mut timed, mut frame_mismatches) = (0, 0, 0, 0);
    let mut examples = Vec::new();
    for n in 0..games as u64 {
        let id = first + n;
        let seed = format!("T{id:07}");
        let spec = GameSpec { seed: &seed, back, stake, max_steps, policy_seed: id * 7919, inject_rate };
        let stats = match diff::run_game(&o, &spec, checks) {
            Ok(s) => {
                ok += 1;
                println!("ok   {seed} steps={} ante={} won={}", s.steps, s.max_ante, s.won);
                s
            }
            Err(f) => {
                println!("FAIL {f}");
                let path = out_dir.join(format!("fail_{seed}_{}_{stake}.txt", back.key()));
                match std::fs::create_dir_all(&out_dir).and_then(|_| std::fs::write(&path, f.codes.join("\n"))) {
                    Ok(()) => println!("  history written to {}", path.display()),
                    Err(e) => println!("  could not write the history to {}: {e}", path.display()),
                }
                f.stats
            }
        };
        steps += stats.steps;
        timed += stats.timed_steps;
        frame_mismatches += stats.frame_mismatches;
        examples.extend(stats.frame_examples);
    }
    println!("{ok}/{games} games matched, {steps} steps compared");
    println!("frame-exact timelines: {}/{timed} steps", timed - frame_mismatches);
    let mut counts: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for e in &examples {
        let k = e.split(" :: ").next().unwrap_or("").split(['(', ' ', '{']).next().unwrap_or("");
        *counts.entry(k).or_default() += 1;
    }
    for (k, n) in counts {
        println!("  frame mismatch {k}: {n}");
    }
    for e in examples.iter().take(12) {
        println!("    {e}");
    }
    Ok(ok == games)
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("difftest: {e}\n{USAGE}");
            ExitCode::from(2)
        }
    }
}
