//! Tiny CLI for poking the oracle: each argument is a Lua expression
//! evaluated by the harness (e.g. `H.start('SEED','b_red',1)`); results go to
//! stdout and errors to stderr.
//!
//! With no arguments it reads one expression per line from stdin and answers
//! each with exactly one stdout line (the result, or `ERROR: <message>` with
//! newlines flattened), so other programs can drive it over a pipe.
//!
//! Exits with status 1 if any expression failed.

use std::io::BufRead;
use std::process::ExitCode;

fn main() -> ExitCode {
	let o = match oracle::Oracle::new() {
		Ok(o) => o,
		Err(e) => {
			eprintln!("oracle init failed: {e}");
			return ExitCode::from(2);
		}
	};
	let mut failed = false;
	let args: Vec<String> = std::env::args().skip(1).collect();
	if args.is_empty() {
		for line in std::io::stdin().lock().lines() {
			match line {
				Ok(l) if l.trim().is_empty() => {}
				Ok(l) => match o.call("oracle", &l) {
					Ok(s) => println!("{s}"),
					Err(e) => {
						println!("ERROR: {}", e.replace(['\r', '\n'], " | "));
						failed = true;
					}
				},
				Err(e) => {
					eprintln!("stdin: {e}");
					return ExitCode::from(2);
				}
			}
		}
	} else {
		for a in &args {
			match o.call("oracle", a) {
				Ok(s) => println!("{s}"),
				Err(e) => {
					eprintln!("ERROR: {e}");
					failed = true;
				}
			}
		}
	}
	if failed {
		ExitCode::FAILURE
	} else {
		ExitCode::SUCCESS
	}
}
