//! A state dump written by the simulator (`Game::to_dump`) reads back as the
//! same game (`Game::from_dump`), and reads as the game's own dump would
//! (`canon::from_lua`), at every decision point of seeded games over every
//! deck and stake. No game needed: `dumpcheck` checks the reader against the
//! real game's dumps.

use std::collections::BTreeMap;

use oracle::canon;
use oracle::dumpcheck::part_diffs;
use oracle::fingerprint::obs_fingerprint;
use oracle::policy::Policy;
use sim::actions::Action;
use sim::data::Back;
use sim::game::Game;
use sim::json::Json;
use sim::{Forced, Stake, json, num};

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

/// A game in its first round whose first hand card Cerulean Bell holds and
/// whose second is a copy with the flag.
fn round_with_a_held_card_and_a_copy() -> Game {
	let mut g = Game::new_run("BELLDUMP", Back::Red, Stake::White);
	g.step(Action::SelectBlind)
		.expect("the first blind can be played");
	let hand = &mut g.areas.hand.cards;
	(hand[0].forced, hand[1].forced) = (Forced::Locked { place: 0 }, Forced::Flagged);
	g
}

/// The `i`-th hand card of `dump`.
fn hand_card(dump: &mut Json, i: usize) -> &mut BTreeMap<String, Json> {
	let Json::Obj(game) = dump else {
		panic!("the dump is an object");
	};
	let Some(Json::Arr(hand)) = game.get_mut("hand") else {
		panic!("the dump has a hand");
	};
	let Json::Obj(card) = &mut hand[i] else {
		panic!("a card is an object");
	};
	card
}

#[test]
fn a_dump_keeps_which_cards_with_the_forced_flag_are_highlighted() {
	check(&mut round_with_a_held_card_and_a_copy(), 0, 0);
}

#[test]
fn a_dump_keeps_the_order_the_held_cards_were_highlighted_in() {
	let mut g = round_with_a_held_card_and_a_copy();
	let hand = &mut g.areas.hand.cards;
	(hand[0].forced, hand[1].forced) = (Forced::Locked { place: 1 }, Forced::Locked { place: 0 });
	check(&mut g, 0, 0);
}

#[test]
fn a_highlighted_card_without_the_forced_flag_is_not_held() {
	let mut g = round_with_a_held_card_and_a_copy();
	let mut dump = g.to_dump();
	hand_card(&mut dump, 2).insert("highlighted".into(), Json::Num(2.0));
	let back = Game::from_dump(&dump).expect("the dump reads");
	assert_eq!(back.areas.hand.cards, g.areas.hand.cards);
}

#[test]
fn a_dump_of_a_mod_that_does_not_write_the_highlights_is_refused() {
	// Before version 0.3.0 the mod wrote no highlight, and 0.3.0 wrote it
	// without the card's place in the list.
	for old in [None, Some(Json::Bool(true))] {
		let mut dump = round_with_a_held_card_and_a_copy().to_dump();
		let card = hand_card(&mut dump, 0);
		match &old {
			None => card.remove("highlighted"),
			Some(highlighted) => card.insert("highlighted".into(), highlighted.clone()),
		};
		let refused = Game::from_dump(&dump).expect_err("no highlight place for a card");
		assert!(refused.to_string().contains("update the mod"), "{refused}");
		let forced = &canon::from_lua(&dump)["hand.0.forced"];
		assert_eq!(
			forced, "?",
			"the canonical state does not guess, with {old:?}"
		);
	}
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
