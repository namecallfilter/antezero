//! A state dump written by the simulator (`Game::to_dump`) reads back as the
//! same game (`Game::from_dump`), and reads as the game's own dump would
//! (`canon::from_lua`), at every decision point of seeded games over every
//! deck and stake. No game needed: `dumpcheck` checks the reader against the
//! real game's dumps.

use oracle::canon;
use oracle::dumpcheck::part_diffs;
use oracle::fingerprint::obs_fingerprint;
use oracle::policy::Policy;
use sim::data::Back;
use sim::game::Game;
use sim::json::Json;
use sim::{Stake, json, num};

const GAMES: u64 = 60;
const STEPS: usize = 250;

fn check(g: &mut Game, n: u64, step: usize) {
	let text = g.to_dump().to_string();
	let dump = json::parse(&text).expect("a written dump parses");
	let d = canon::diff(&canon::from_sim(g), &canon::from_lua(&dump));
	assert!(
		d.is_empty(),
		"game {n} step {step}: the dump reads as {d:?}"
	);
	let mut back = Game::from_dump(&dump).unwrap_or_else(|e| panic!("game {n} step {step}: {e}"));
	back.flow.rearranges = g.flow.rearranges;
	let parts = part_diffs(&mut back, g);
	assert!(parts.is_empty(), "game {n} step {step}: {parts:?}");
	assert_eq!(
		obs_fingerprint(&mut back),
		obs_fingerprint(g),
		"game {n} step {step}: the observation differs"
	);
}

#[test]
fn a_written_dump_reads_back_as_the_same_game() {
	let backs: Vec<Back> = Back::simulated().collect();
	for n in 0..GAMES {
		let back = backs[num::pick(n, backs.len())];
		let stake = Stake::ALL[num::pick(n / 3, Stake::ALL.len())];
		let mut g = Game::new_run(&format!("DUMP{n:04}"), back, stake);
		let mut policy = Policy::for_game(n, 0.1);
		check(&mut g, n, 0);
		for step in 1..=STEPS {
			let Some(command) = policy.next(&g) else {
				break;
			};
			command
				.apply(&mut g)
				.unwrap_or_else(|e| panic!("game {n}: {command:?} rejected: {e}"));
			check(&mut g, n, step);
		}
	}
}

/// The tables the dump copies whole from the game, where a key may be
/// absent (the game's nil): the RNG streams, the bosses seen, the
/// consumables used, the blinds' tags and a card's ability table.
const GAME_TABLES: [&str; 5] = [
	"rng.",
	"bosses_used.",
	"consumeable_usage_total.",
	"blind_tags.",
	".ability.",
];

/// Every key path of `j` (a list's first element, as index 0).
fn key_paths(j: &Json, prefix: &[String], out: &mut Vec<Vec<String>>) {
	let at = |k: &str| [prefix, &[k.to_string()]].concat();
	match j {
		Json::Obj(fields) => {
			for (k, v) in fields {
				out.push(at(k));
				key_paths(v, &at(k), out);
			}
		}
		Json::Arr(items) => {
			if let Some(v) = items.first() {
				key_paths(v, &at("0"), out);
			}
		}
		_ => {}
	}
}

/// `dump` without the key at `path`.
fn without(dump: &Json, path: &[String]) -> Json {
	let mut out = dump.clone();
	let (key, parent) = path.split_last().expect("a key");
	let mut at = &mut out;
	for seg in parent {
		at = match at {
			Json::Obj(fields) => fields.get_mut(seg).expect("a key of the dump"),
			Json::Arr(items) => &mut items[seg.parse::<usize>().expect("an index")],
			_ => panic!("{path:?}: no {seg}"),
		};
	}
	let Json::Obj(fields) = at else {
		panic!("{path:?}: not in an object");
	};
	fields.remove(key);
	out
}

/// The keys of `g`'s dump whose absence the importer reads as another game
/// (rather than refusing the dump or not reading the key); and how many it
/// tried.
fn read_without_a_key(g: &mut Game) -> (Vec<String>, usize) {
	let dump = g.to_dump();
	let mut whole = Game::from_dump(&dump).expect("a written dump reads back");
	let mut paths = Vec::new();
	key_paths(&dump, &[], &mut paths);
	paths.retain(|p| {
		let shown = format!("{}.", p.join("."));
		!GAME_TABLES
			.iter()
			.any(|t| shown.starts_with(t) || shown.contains(t))
	});
	let changed = paths
		.iter()
		.filter(|p| {
			Game::from_dump(&without(&dump, p)).is_ok_and(|mut other| {
				other.flow.rearranges = whole.flow.rearranges;
				!part_diffs(&mut other, &mut whole).is_empty()
			})
		})
		.map(|p| p.join("."))
		.collect();
	(changed, paths.len())
}

#[test]
fn a_dump_without_a_key_the_simulator_reads_is_refused() {
	let backs: Vec<Back> = Back::simulated().collect();
	let mut tried = 0;
	for n in 0..8 {
		let back = backs[num::pick(n, backs.len())];
		let stake = Stake::ALL[num::pick(n, Stake::ALL.len())];
		let mut g = Game::new_run(&format!("KEYS{n:04}"), back, stake);
		let mut policy = Policy::for_game(n, 0.1);
		for step in 0..STEPS {
			if step % 10 == 0 {
				let (changed, n_paths) = read_without_a_key(&mut g);
				assert!(
					changed.is_empty(),
					"game {n} step {step}: keys a dump can be without, read as another game: {changed:?}"
				);
				tried += n_paths;
			}
			let Some(command) = policy.next(&g) else {
				break;
			};
			command
				.apply(&mut g)
				.unwrap_or_else(|e| panic!("game {n}: {command:?} rejected: {e}"));
		}
	}
	assert!(tried > 10_000, "{tried} keys");
}
