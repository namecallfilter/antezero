//! The learning observation reads nothing a player cannot know. A game
//! must give the same observation and ops, play candidates and build
//! previews included, as the worlds the player cannot tell from it: its
//! hidden state resampled (`Game::resample_hidden`: deck order, face-down
//! cards, every future random outcome, an unseen voucher), and worlds built
//! apart from that resampler, from the state dump: a card the player has not
//! seen (face down in the hand, or discarded so) swapped with one of the
//! deck, and a face-up card dealt with another suit than it shows; and the
//! resampler must deal the swapped worlds alike. A game
//! read from the real game's state dump holds the game's true hidden state,
//! so the games are compared as the bot reads them (imported from their
//! dumps), with and without the deck's tokens, with the ops as training
//! offers them, as the released policy is offered them (eight candidates,
//! no skips) and without the null sorts.

use std::mem;

use oracle::fingerprint::obs_fingerprint_with;
use oracle::policy::Policy;
use sim::data::{Back, Edition, Suit};
use sim::game::Game;
use sim::json::Json;
use sim::obs::OpOptions;
use sim::{Stake, num};

const GAMES: u64 = 30;
const STEPS: usize = 200;

fn options() -> [OpOptions; 3] {
	[
		OpOptions {
			candidates: 4,
			..OpOptions::default()
		},
		OpOptions {
			candidates: 8,
			allow_skip: false,
			null_sorts: true,
		},
		OpOptions {
			candidates: 8,
			allow_skip: false,
			null_sorts: false,
		},
	]
}

fn imported(dump: &Json) -> Game {
	Game::from_dump(dump).expect("a dump the simulator wrote reads back")
}

fn fingerprints(g: &mut Game) -> Vec<u64> {
	let mut out = Vec::new();
	for deck_tokens in [false, true] {
		for o in options() {
			out.push(obs_fingerprint_with(g, deck_tokens, o));
		}
	}
	out
}

fn cards_mut<'d>(dump: &'d mut Json, area: &str) -> &'d mut Vec<Json> {
	let Json::Obj(fields) = dump else {
		panic!("a dump is an object");
	};
	let Some(Json::Arr(cards)) = fields.get_mut(area) else {
		panic!("the dump has the area {area}");
	};
	cards
}

fn field_mut<'c>(card: &'c mut Json, key: &str) -> &'c mut Json {
	let Json::Obj(fields) = card else {
		panic!("a card is an object");
	};
	fields.entry(key.to_string()).or_insert(Json::Null)
}

/// The places of the cards the player has not seen: face down in the hand,
/// or discarded face down this round (still `wheel_flipped`).
fn unseen(dump: &Json) -> Vec<(&'static str, usize)> {
	let hand = dump.get("hand").as_list().iter().enumerate();
	let discard = dump.get("discard").as_list().iter().enumerate();
	hand.filter(|(_, c)| c.get("facing").as_str() == Some("back"))
		.map(|(i, _)| ("hand", i))
		.chain(
			discard
				.filter(|(_, c)| c.get("ability").get("wheel_flipped").as_flag())
				.map(|(i, _)| ("discard", i)),
		)
		.collect()
}

/// `dump` with the card at `at` and the deck's card `d` changing places,
/// each place keeping how it shows the card.
fn swapped(dump: &Json, at: (&str, usize), d: usize) -> Json {
	let mut out = dump.clone();
	let mut a = cards_mut(&mut out, at.0)[at.1].clone();
	let mut b = cards_mut(&mut out, "deck")[d].clone();
	for key in ["facing", "debuff"] {
		mem::swap(field_mut(&mut a, key), field_mut(&mut b, key));
	}
	let (fa, fb) = (
		field_mut(field_mut(&mut a, "ability"), "wheel_flipped").clone(),
		field_mut(field_mut(&mut b, "ability"), "wheel_flipped").clone(),
	);
	*field_mut(field_mut(&mut a, "ability"), "wheel_flipped") = fb;
	*field_mut(field_mut(&mut b, "ability"), "wheel_flipped") = fa;
	cards_mut(&mut out, at.0)[at.1] = b;
	cards_mut(&mut out, "deck")[d] = a;
	out
}

/// `dump` with the hand's card `i` dealt with another suit than it was.
fn other_original_suit(dump: &Json, i: usize) -> Json {
	let mut out = dump.clone();
	let card = &mut cards_mut(&mut out, "hand")[i];
	let was = card.get("suit_nominal_original").as_f64();
	let other = Suit::ALL
		.into_iter()
		.map(Suit::nominal_original)
		.find(|&n| Some(n) != was)
		.expect("four suits");
	*field_mut(card, "suit_nominal_original") = Json::Num(other);
	out
}

/// The worlds of `dump` built from it: a card the player has not seen
/// swapped with a deck card, and a face-up hand card dealt with another
/// suit; with whether the card swapped places (so that resampling either
/// world deals the same one).
fn twins(dump: &Json, n: u64) -> Vec<(Json, bool)> {
	let mut out = Vec::new();
	let deck = dump.get("deck").as_list().len();
	if deck > 0 {
		for (k, at) in unseen(dump).into_iter().enumerate() {
			out.push((
				swapped(dump, at, num::pick(n + num::index64(k), deck)),
				true,
			));
		}
	}
	let hand = dump.get("hand").as_list();
	if let Some(i) = (0..hand.len()).find(|&i| hand[i].get("facing").as_str() == Some("front")) {
		out.push((other_original_suit(dump, i), false));
	}
	out
}

/// The dump of `g` with its hidden state resampled with `seed`.
fn resampled(g: &Game, seed: &str) -> Json {
	let mut w = g.clone();
	w.resample_hidden(seed);
	w.to_dump()
}

#[test]
fn the_observation_reads_no_hidden_state() {
	let backs: Vec<Back> = Back::simulated().collect();
	let (mut twinned, mut unseen_discards) = (0, 0);
	for n in 0..GAMES {
		let back = backs[num::pick(n, backs.len())];
		let stake = Stake::ALL[num::pick(n / 2, Stake::ALL.len())];
		let mut g = Game::new_run(&format!("HIDE{n:04}"), back, stake);
		let mut policy = Policy::for_game(n, 0.1);
		for step in 0..STEPS {
			let dump = g.to_dump();
			let seen = fingerprints(&mut imported(&dump));
			for world in 0..2 {
				let mut w = g.clone();
				w.resample_hidden(&format!("{n}_{step}_{world}"));
				assert_eq!(
					fingerprints(&mut imported(&w.to_dump())),
					seen,
					"game {n} step {step}: the observation changed with the hidden state"
				);
			}
			for (twin, places) in twins(&dump, n * 1000 + num::index64(step)) {
				twinned += 1;
				unseen_discards += usize::from(unseen(&twin).iter().any(|(a, _)| *a == "discard"));
				let mut other = imported(&twin);
				assert_eq!(
					fingerprints(&mut other),
					seen,
					"game {n} step {step}: the observation tells a world the player cannot tell apart"
				);
				let world = format!("{n}_{step}_twin");
				assert!(
					!places || resampled(&other, &world) == resampled(&imported(&dump), &world),
					"game {n} step {step}: resampling deals a world the player cannot tell apart otherwise"
				);
			}
			let Some(command) = policy.next(&g) else {
				break;
			};
			command
				.apply(&mut g)
				.unwrap_or_else(|e| panic!("game {n}: {command:?} rejected: {e}"));
		}
	}
	assert!(
		twinned > 1000 && unseen_discards > 0,
		"{twinned} worlds, {unseen_discards} with a card discarded unseen"
	);
}

/// A card discarded face down under The Wheel: the player never saw it.
#[test]
fn a_card_discarded_unseen_stays_unseen() {
	let mut g = Game::new_run("UNSEEN", Back::Red, Stake::White);
	let a: sim::Action = "select".parse().expect("an action");
	g.step(a).expect("the small blind");
	let mut dump = g.to_dump();
	let card = &mut cards_mut(&mut dump, "hand")[0];
	*field_mut(card, "facing") = Json::Str("back".into());
	*field_mut(field_mut(card, "ability"), "wheel_flipped") = Json::Bool(true);
	let mut g = imported(&dump);
	let a: sim::Action = "discard 1".parse().expect("an action");
	g.step(a).expect("a discard");
	let dump = g.to_dump();
	assert_eq!(unseen(&dump), [("discard", 0)]);
	let seen = fingerprints(&mut imported(&dump));
	let dealt = resampled(&imported(&dump), "unseen");
	for d in 0..dump.get("deck").as_list().len() {
		let other = imported(&swapped(&dump, ("discard", 0), d));
		assert_eq!(fingerprints(&mut other.clone()), seen, "deck card {d}");
		assert!(
			resampled(&other, "unseen") == dealt,
			"deck card {d}: resampled apart"
		);
	}
}

/// Two face-up Aces of Spades, one dealt a Heart: a sort orders them by the
/// suit they were dealt, which does not show.
#[test]
fn the_sort_ops_do_not_tell_the_suit_a_card_was_dealt() {
	let mut g = Game::new_run("SORTS", Back::Red, Stake::White);
	let a: sim::Action = "select".parse().expect("an action");
	g.step(a).expect("the small blind");
	let mut dump = g.to_dump();
	let hand = cards_mut(&mut dump, "hand");
	let ace = hand[0].clone();
	for (i, original, id) in [(0, Suit::Spades, 900.0), (1, Suit::Hearts, 901.0)] {
		hand[i] = ace.clone();
		for (key, v) in [
			("rank", Json::Num(14.0)),
			("suit", Json::Str("Spades".into())),
			("card", Json::Str("S_A".into())),
			("key", Json::Str("c_base".into())),
			(
				"suit_nominal_original",
				Json::Num(original.nominal_original()),
			),
			("sort_id", Json::Num(id)),
		] {
			*field_mut(&mut hand[i], key) = v;
		}
	}
	let mut other = dump.clone();
	for (i, original) in [(0, Suit::Hearts), (1, Suit::Spades)] {
		*field_mut(
			&mut cards_mut(&mut other, "hand")[i],
			"suit_nominal_original",
		) = Json::Num(original.nominal_original());
	}
	assert_eq!(
		fingerprints(&mut imported(&dump)),
		fingerprints(&mut imported(&other))
	);
}

/// `card` made a Stone card hiding the suit and rank it has.
fn stone(card: &mut Json, suit: Suit, rank: f64, edition: Option<Edition>) {
	for (key, v) in [
		("key", Json::Str("m_stone".into())),
		("suit", Json::Str(suit.name().into())),
		("rank", Json::Num(rank)),
		("suit_nominal_original", Json::Num(suit.nominal_original())),
		(
			"edition",
			edition.map_or(Json::Null, |e| Json::Str(e.name().into())),
		),
	] {
		*field_mut(card, key) = v;
	}
}

/// What shows of the hand and the deck, in order: a Stone card without the
/// suit and rank it hides.
fn shown(dump: &Json) -> Vec<String> {
	let cards = dump
		.get("hand")
		.as_list()
		.iter()
		.chain(dump.get("deck").as_list());
	cards
		.map(|c| {
			let key = c.get("key").as_str().unwrap_or("?");
			let face = if key == "m_stone" {
				String::new()
			} else {
				format!("{:?} {:?}", c.get("suit").as_str(), c.get("rank").as_f64())
			};
			format!(
				"{key} {:?} {:?} {face}",
				c.get("edition").as_str(),
				c.get("facing").as_str()
			)
		})
		.collect()
}

/// Stone cards show no suit or rank: one in the hand, and two in the deck
/// told apart only by their editions, read and resample alike whatever
/// suits and ranks they hide (with a card face down in the hand, so that the
/// play candidates deal unseen worlds too).
#[test]
fn a_stone_card_keeps_its_suit_and_rank_hidden() {
	let mut g = Game::new_run("STONES", Back::Red, Stake::White);
	let a: sim::Action = "select".parse().expect("an action");
	g.step(a).expect("the small blind");
	let mut dump = g.to_dump();
	*field_mut(&mut cards_mut(&mut dump, "hand")[1], "facing") = Json::Str("back".into());
	let mut other = dump.clone();
	for (d, hidden) in [
		(&mut dump, [
			(Suit::Spades, 14.0),
			(Suit::Spades, 14.0),
			(Suit::Hearts, 2.0),
		]),
		(&mut other, [
			(Suit::Clubs, 7.0),
			(Suit::Hearts, 2.0),
			(Suit::Spades, 14.0),
		]),
	] {
		stone(&mut cards_mut(d, "hand")[0], hidden[0].0, hidden[0].1, None);
		stone(
			&mut cards_mut(d, "deck")[0],
			hidden[1].0,
			hidden[1].1,
			Some(Edition::Foil),
		);
		stone(
			&mut cards_mut(d, "deck")[1],
			hidden[2].0,
			hidden[2].1,
			Some(Edition::Polychrome),
		);
	}
	let (a, b) = (imported(&dump), imported(&other));
	assert_eq!(
		fingerprints(&mut a.clone()),
		fingerprints(&mut b.clone()),
		"the observation"
	);
	assert_eq!(
		shown(&resampled(&a, "stones")),
		shown(&resampled(&b, "stones")),
		"the resampled worlds"
	);
}
