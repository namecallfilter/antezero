//! Recordings of the native environment (`sim::balatro`): what a recording
//! environment writes, [`load_starts`] replays to the game that was played.

use std::sync::Arc;

use sim::balatro::{
	ACT_DISCARD, ACT_PLAY, ACT_REORDER_JOKERS, ACT_SELL_JOKER, ACT_SKIP_BLIND, ACT_USE_CONS,
	BalConfig, BalEnv, HS_CTX, N_ACTIONS, N_SEL_VERBS, OBS_SIZE, SEL_F, ScriptError, load_starts,
	sel_verb_index,
};
use sim::data::Back;
use sim::game::HandSort;
use sim::rng::Xorshift;
use sim::{Action, Game, Stake, canon, num, obs};

/// The seed of the walks' random choices.
const WALK_SEED: u64 = 0x9e37_79b9_7f4a_7c15;

fn recording_env() -> BalEnv {
	BalEnv::new(BalConfig {
		record: true,
		..BalConfig::default()
	})
}

/// Cards for the selection verb `verb`, chosen at random among those the
/// environment allows next, until it allows stopping.
fn picks(env: &BalEnv, verb: usize, specs: &[i32], rng: &mut Xorshift) -> Vec<i32> {
	let (lo, hi) = (specs[verb * 3], specs[verb * 3 + 1]);
	let eligible = u32::from_ne_bytes(specs[verb * 3 + 2].to_ne_bytes());
	let mut picked = Vec::new();
	if hi < 0 {
		return picked;
	}
	let mut chosen = 0u32;
	if sel_verb_index(ACT_REORDER_JOKERS) != Some(verb) {
		for (i, card) in env.game().areas.hand.cards.iter().enumerate() {
			chosen |= u32::from(card.forced_selection) << i;
		}
	}
	loop {
		let (cards, stop) = env.pick_rule(verb, lo, hi, eligible, chosen, num::id(picked.len()));
		if cards == 0 || (stop && rng.chance(0.5)) {
			return picked;
		}
		let choices: Vec<i32> = (0..32).filter(|i| cards >> i & 1 != 0).collect();
		let card = choices[rng.below(choices.len())];
		chosen |= 1 << card;
		picked.push(card);
	}
}

/// Plays up to `turns` decisions of `env` with a seeded policy. In a round it
/// mostly plays the best-scoring cards, or discards the others while that
/// play scores too little, and it rarely skips a blind or sells, so that
/// games reach shops and packs; every other choice is random. Returns the
/// live game after each decision with the number of action lines recorded
/// up to it.
fn play(env: &mut BalEnv, rng: &mut Xorshift, turns: usize) -> Vec<(usize, Game)> {
	let mut obs = vec![0.0; OBS_SIZE];
	let (mut sel, mut hctx) = ([0.0; SEL_F], [0; HS_CTX]);
	let (mut mask, mut specs) = ([0; N_ACTIONS], [0; N_SEL_VERBS * 3]);
	let mut live = Vec::new();
	for _ in 0..turns {
		env.write_all_obs(&mut obs, &mut sel, &mut mask, &mut specs, &mut hctx);
		let keep = rng.chance(0.8);
		let gives_up =
			|a: usize| a == ACT_SKIP_BLIND || (ACT_SELL_JOKER..ACT_USE_CONS).contains(&a);
		let legal: Vec<usize> = (0..N_ACTIONS)
			.filter(|&a| mask[a] != 0 && !(keep && gives_up(a)))
			.collect();
		let best = (mask[ACT_PLAY] != 0 && rng.chance(0.85))
			.then(|| obs::top_plays(env.game(), 1))
			.and_then(|plays| plays.first().copied());
		let (action, cards) = if let Some(best) = best {
			let game = env.game();
			let needed = (game.blind.chips - game.round.chips) / game.round.hands_left;
			let hand = &game.areas.hand.cards;
			let free = |i: &usize| !hand[*i].forced_selection;
			let rest: Vec<i32> = (0..hand.len())
				.filter(|&i| !best.mask.contains(i))
				.filter(free)
				.take(5)
				.map(num::id)
				.collect();
			if best.mean < needed && mask[ACT_DISCARD] != 0 && !rest.is_empty() {
				(ACT_DISCARD, rest)
			} else {
				(
					ACT_PLAY,
					best.mask.indices().filter(free).map(num::id).collect(),
				)
			}
		} else {
			let action = legal[rng.below(legal.len())];
			let cards =
				sel_verb_index(action).map_or_else(Vec::new, |v| picks(env, v, &specs, rng));
			(action, cards)
		};
		let info = env.step_select(num::id(action), &cards);
		live.push((env.script().lines().count() - 1, env.game().clone()));
		if info.terminated != 0 || info.truncated != 0 {
			break;
		}
	}
	live
}

/// Panics with the first difference between the two games.
fn assert_same_game(replayed: &Game, live: &Game, what: &str) {
	let replayed_state = canon::from_sim(&mut replayed.clone());
	let live_state = canon::from_sim(&mut live.clone());
	for (key, value) in &live_state {
		assert_eq!(replayed_state.get(key), Some(value), "{what}: {key}");
	}
	assert_eq!(replayed_state.len(), live_state.len(), "{what}");
	assert_eq!(replayed.rng.seed(), live.rng.seed(), "{what}: seed");
	assert_eq!(
		replayed.flow.rearranges, live.flow.rearranges,
		"{what}: rearranges"
	);
}

/// Replays `script` to every step of `live` and compares the games; a step
/// at which the run is over is refused.
fn assert_replays(script: &str, live: &[(usize, Game)]) {
	let (over, alive): (Vec<_>, Vec<_>) = live.iter().partition(|(_, game)| game.is_terminal());
	let steps: Vec<usize> = alive.iter().map(|&&(step, _)| step).collect();
	let points = load_starts(script, &steps).expect("a recording replays");
	for (&&(step, ref game), point) in alive.iter().zip(points) {
		let what = format!("after action line {step} of\n{script}");
		assert_same_game(&point.game, game, &what);
	}
	for &(line, _) in over {
		assert_eq!(
			load_starts(script, &[line]).err(),
			Some(ScriptError::Over { line }),
			"a finished run is no start point"
		);
	}
}

/// The action lines of `scripts` that start with `verb` and name `cards`
/// hand positions (after the slot of a `use` or `pick`).
fn lines_with(scripts: &[String], verb: &str, cards: impl Fn(usize) -> bool) -> usize {
	let slot = usize::from(matches!(verb, "use" | "pick"));
	scripts
		.iter()
		.flat_map(|script| script.lines().skip(1))
		.filter(|line| {
			let mut words = line.split_whitespace();
			words.next() == Some(verb) && words.count().checked_sub(slot).is_some_and(&cards)
		})
		.count()
}

/// `script` as a recording of the real game has it: with a highlighted hand
/// card (position 0) listed on each `use` and `pick` line that lists none
/// while the hand holds cards. Also returns the number of such lines.
fn with_highlights(script: &str, live: &[(usize, Game)]) -> (String, usize) {
	let mut text: Vec<String> = script.lines().map(str::to_string).collect();
	let mut highlighted = 0;
	for pair in live.windows(2) {
		let (action, before) = (&mut text[pair[0].0 + 1], &pair[0].1);
		let words: Vec<&str> = action.split_whitespace().collect();
		if matches!(words[..], ["use" | "pick", _]) && !before.areas.hand.cards.is_empty() {
			action.push_str(" 0");
			highlighted += 1;
		}
	}
	(text.join("\n"), highlighted)
}

#[test]
fn a_recorded_game_replays_to_the_live_game_at_every_step() {
	let (mut scripts, mut highlighted) = (Vec::new(), 0);
	for i in 0..32 {
		let mut env = recording_env();
		env.reset_index(i);
		let live = play(&mut env, &mut Xorshift::new(WALK_SEED ^ i), 200);
		assert_replays(env.script(), &live);
		let (script, lines) = with_highlights(env.script(), &live);
		assert_replays(&script, &live);
		highlighted += lines;
		scripts.push(env.script().to_string());
	}
	assert!(highlighted > 0, "no card was used with a card highlighted");
	assert!(lines_with(&scripts, "play", |n| n == 1) > 0);
	assert!(lines_with(&scripts, "play", |n| n > 1) > 0);
	assert!(lines_with(&scripts, "discard", |n| n > 0) > 0);
	assert!(lines_with(&scripts, "use", |n| n > 0) > 0);
	assert!(lines_with(&scripts, "pick", |n| n > 0) > 0);
	assert!(lines_with(&scripts, "pick", |n| n == 0) > 0);
	for verb in ["move_hand", "buy", "open", "sell_joker", "sell_cons"] {
		assert!(lines_with(&scripts, verb, |_| true) > 0, "no {verb} line");
	}
}

#[test]
fn a_game_recorded_from_a_start_point_replays_to_the_live_game() {
	for (i, reseed) in [(2, true), (7, true), (30, true), (2, false)] {
		let mut env = recording_env();
		env.reset_index(i);
		let mut rng = Xorshift::new(WALK_SEED ^ i);
		let (step, before) = play(&mut env, &mut rng, 12)
			.pop()
			.expect("a played decision");
		let point = load_starts(env.script(), &[step])
			.expect("a recording replays")
			.pop()
			.expect("a start point for the step");
		assert_same_game(&point.game, &before, "the start point");

		env.set_start(Some(Arc::new(point)), reseed);
		env.reset("RESEED");
		assert_eq!(env.game().rng.seed() == "RESEED", reseed);
		assert_eq!(env.script().contains("\ndbg_reseed RESEED\n"), reseed);
		let mut live = vec![(env.script().lines().count() - 1, env.game().clone())];
		live.extend(play(&mut env, &mut rng, 200));
		assert!(live.len() > 3, "the game goes on from the start point");
		assert_replays(env.script(), &live);
	}
}

fn script(lines: &[&str]) -> String {
	format!("seed TEST deck b_red stake 1\n{}\n", lines.join("\n"))
}

#[test]
fn a_script_lists_hand_cards_by_position() {
	let mut game = Game::new_run("TEST", Back::Red, Stake::White);
	for action in [
		Action::SelectBlind,
		Action::SortHand(HandSort::Suit),
		Action::MoveHand { from: 0, to: 7 },
		Action::Discard([0, 5].into_iter().collect()),
		Action::Play([1, 2].into_iter().collect()),
	] {
		game.step(action).expect("a legal action");
	}
	let text = script(&[
		"select",
		"sort_suit  # by suit",
		"",
		"move_hand 0 7",
		"discard 0 5",
		"play 2 1",
	]);
	let points = load_starts(&text, &[5]).expect("the script replays");
	assert_same_game(&points[0].game, &game, "after the script");
}

#[test]
fn a_line_that_does_not_replay_fails_the_whole_script() {
	let illegal = |line, text: &str| {
		Some(ScriptError::Illegal {
			line,
			text: text.to_string(),
		})
	};
	let text = script(&["select", "cash_out"]);
	assert_eq!(load_starts(&text, &[1, 2]).err(), illegal(2, "cash_out"));
	assert!(
		load_starts(&text, &[1]).is_ok(),
		"lines after the largest step are not read"
	);
	for directive in ["endless", "dbg_skip_pack"] {
		let text = script(&["select", directive]);
		assert_eq!(load_starts(&text, &[2]).err(), illegal(2, directive));
	}
	let short = ScriptError::TooShort { lines: 1 };
	assert_eq!(load_starts(&script(&["select"]), &[3]).err(), Some(short));
}

#[test]
fn only_the_lines_a_recording_holds_are_lines_of_a_script() {
	for line in [
		"not_an_action 9",
		"? select",
		"sell 0",
		"sell_consumable 0",
		"sort 0",
		"select 1",
		"buy",
		"buy 0 1",
		"move_hand 0",
		"play 1 x",
		"use",
		"dbg_reseed",
		"dbg_reseed A B",
	] {
		let unparseable = ScriptError::Unparseable {
			line: 1,
			text: line.to_string(),
		};
		assert_eq!(load_starts(&script(&[line]), &[1]).err(), Some(unparseable));
	}
}

#[test]
fn a_header_is_a_seed_a_deck_and_a_stake() {
	let run_of = |header: &str| {
		load_starts(header, &[0]).map(|points| {
			let game = &points[0].game;
			(
				game.rng.seed().to_string(),
				game.setup.back,
				game.setup.stake,
			)
		})
	};
	assert_eq!(
		run_of("seed TEST deck b_blue stake 8"),
		Ok(("TEST".to_string(), Back::Blue, Stake::Gold))
	);
	for header in [
		"seed TEST",
		"seed TEST deck b_red",
		"seed TEST deck b_red stake",
		"stake 8 deck b_blue seed TEST",
		"seed TEST deck b_red stake 1 challenge c_omelette_1",
		"select",
	] {
		assert_eq!(run_of(header), Err(ScriptError::Header(header.to_string())));
	}
	for stake in ["garbage", "0", "9"] {
		assert_eq!(
			run_of(&format!("seed TEST deck b_red stake {stake}")),
			Err(ScriptError::Stake(stake.to_string()))
		);
	}
	assert_eq!(
		run_of("seed TEST deck b_nope stake 1"),
		Err(ScriptError::Deck("b_nope".to_string()))
	);
	let long_seed = "S".repeat(sim::MAX_SEED_LEN + 1);
	assert_eq!(
		run_of(&format!("seed {long_seed} deck b_red stake 1")),
		Err(ScriptError::Seed(long_seed))
	);
	assert_eq!(run_of("# nothing\n\n"), Err(ScriptError::Empty));
}
