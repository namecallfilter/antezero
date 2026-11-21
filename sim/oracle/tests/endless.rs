//! The simulator in Endless Mode, past ante 8, against the real game. The
//! game's run is taken past its win (`won`, `win_notified`, as Endless Mode
//! leaves them) to a later ante, with every poker hand scoring enough to
//! beat that ante's blinds, so that it plays on through shops, packs, tags
//! and bosses; it is read into the simulator from its state dump, and then
//! both play the same commands, compared after every one (as in
//! `oracle::difftest`), and the simulator's observation must stay finite. It
//! needs the game's `lua51.dll` and scripts, so it
//! only runs when asked for:
//! `cargo test -p oracle --test endless -- --include-ignored`.

use oracle::command::Start;
use oracle::policy::Policy;
use oracle::{Oracle, canon};
use sim::data::Back;
use sim::game::Game;
use sim::obs::{self, ObsBuf};
use sim::{Stake, json, num};

/// The antes the runs start at: past the win, up to where the blinds' chips
/// overflow (the game ends a run whose score would have to be infinite).
const ANTES: [i32; 9] = [9, 10, 11, 13, 16, 20, 25, 30, 36];
const STEPS: usize = 400;

fn dump(o: &mut Oracle) -> json::Json {
	json::parse(&o.command("H.state()").expect("the harness dumps")).expect("its dump is JSON")
}

#[test]
#[ignore = "needs the game's lua51.dll; run with --include-ignored"]
fn endless_runs_are_the_game() {
	assert!(
		oracle::available(),
		"the oracle needs the game's lua51.dll and scripts (see oracle/src/lib.rs)"
	);
	let mut o = Oracle::new().expect("the harness loads (`oracle::available()` holds)");
	let backs: Vec<Back> = Back::simulated().collect();
	let (mut steps, mut max_ante) = (0, 0);
	for (n, ante) in (0u64..).zip(ANTES) {
		let seed = format!("E{n:07}");
		let start = Start {
			seed: &seed,
			back: backs[num::pick(n * 3, backs.len())],
			stake: Stake::ALL[num::pick(n, Stake::ALL.len())],
		};
		o.command(&start.to_lua()).expect("the run starts");
		o.exec(&format!(
			"G.GAME.won, G.GAME.win_notified = true, true
			G.GAME.round_resets.ante = {ante}
			for _, h in pairs(G.GAME.hands) do h.chips, h.mult = 1e100, 1e100 end"
		))
		.expect("the run goes on in Endless Mode");
		let mut j = dump(&mut o);
		let mut g = Game::from_dump(&j).unwrap_or_else(|e| panic!("{seed}: {e}"));
		g.setup.endless = true;
		let mut policy = Policy::new(n.wrapping_mul(7919), 0.0);
		let mut history = Vec::new();
		for step in 0..STEPS {
			let d = canon::diff(&canon::from_sim(&mut g), &canon::from_lua(&j));
			assert!(
				d.is_empty(),
				"{seed} from ante {ante}, step {step}: {}\nafter the commands {:?}",
				d.iter()
					.take(12)
					.map(ToString::to_string)
					.collect::<Vec<_>>()
					.join("; "),
				&history[history.len().saturating_sub(12)..]
			);
			let mut buf = ObsBuf::new();
			obs::encode_opts(&g, &mut buf.as_mut(), true);
			assert!(
				buf.tok_f.iter().chain(&buf.glob_f).all(|x| x.is_finite()),
				"{seed} from ante {ante}, step {step}: an observation that is not finite"
			);
			let Some(command) = policy.next(&g) else {
				break;
			};
			history.push(command.to_string());
			command
				.apply(&mut g)
				.unwrap_or_else(|e| panic!("{seed}: the simulator rejected {command}: {e}"));
			j = json::parse(&o.command(&command.to_lua()).expect("the harness acts"))
				.expect("JSON");
			steps += 1;
			max_ante = max_ante.max(g.run.ante);
		}
	}
	assert!(
		steps > 1000 && max_ante > 30,
		"{steps} steps, up to ante {max_ante}"
	);
}
