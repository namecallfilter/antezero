//! The native environment (`sim::balatro`) against the game it wraps: what
//! it lets a policy declare is what the game scores, what it hides stays
//! hidden, and what it reports is the game's state.

use std::mem;
use std::sync::Arc;

use sim::balatro::{
	ACT_DISCARD, ACT_PLAY_HAND, BalConfig, BalEnv, HS_CTX, N_ACTIONS, N_SEL_VERBS, OBS_SIZE, SEL_F,
	StartPoint, sel_verb_index,
};
use sim::data::{Back, HandType, Joker, Rank, Seal, Suit};
use sim::hand_eval::{self, EvalFlags};
use sim::inject::Inject;
use sim::rng::Xorshift;
use sim::{Action, Game, PlayingCard, Stake, num};

/// An environment whose episode starts in `game`.
fn env_at(game: Game, cfg: BalConfig) -> BalEnv {
	let mut env = BalEnv::new(cfg);
	let start = StartPoint {
		game,
		script: String::new(),
		id: 0,
	};
	env.set_start(Some(Arc::new(start)), false);
	env.reset("START");
	env
}

/// Everything an environment shows a policy.
#[derive(PartialEq)]
struct Seen {
	obs: Vec<f32>,
	sel: [f32; SEL_F],
	mask: [u8; N_ACTIONS],
	specs: [i32; N_SEL_VERBS * 3],
	hctx: [i32; HS_CTX],
}

fn see(env: &mut BalEnv) -> Seen {
	let mut seen = Seen {
		obs: vec![0.0; OBS_SIZE],
		sel: [0.0; SEL_F],
		mask: [0; N_ACTIONS],
		specs: [0; N_SEL_VERBS * 3],
		hctx: [0; HS_CTX],
	};
	env.write_all_obs(
		&mut seen.obs,
		&mut seen.sel,
		&mut seen.mask,
		&mut seen.specs,
		&mut seen.hctx,
	);
	seen
}

/// A game in its first round, holding `jokers` and exactly the hand `cards`.
fn round_with(jokers: &[Joker], cards: &[(Rank, Suit)]) -> Game {
	let mut game = Game::new_run("COREENV", Back::Red, Stake::White);
	for &joker in jokers {
		game.inject(Inject::joker(joker));
	}
	game.step(Action::SelectBlind)
		.expect("the first blind can be played");
	let hand = &mut game.areas.hand.cards;
	hand.truncate(cards.len());
	assert_eq!(hand.len(), cards.len(), "the hand is dealt enough cards");
	for (card, &(rank, suit)) in hand.iter_mut().zip(cards) {
		(card.rank, card.suit) = (rank, suit);
	}
	game
}

const STRAIGHT_FLUSH: usize = HandType::StraightFlush as usize;

/// A straight and a flush that Four Fingers finds in five cards without one
/// suit holding the straight.
const SPLIT_STRAIGHT_FLUSHES: [[(Rank, Suit); 5]; 2] = [
	[
		(Rank::Two, Suit::Hearts),
		(Rank::Three, Suit::Hearts),
		(Rank::Four, Suit::Hearts),
		(Rank::Five, Suit::Clubs),
		(Rank::King, Suit::Hearts),
	],
	[
		(Rank::Two, Suit::Hearts),
		(Rank::Three, Suit::Hearts),
		(Rank::Four, Suit::Hearts),
		(Rank::Seven, Suit::Hearts),
		(Rank::Five, Suit::Clubs),
	],
];

#[test]
fn a_straight_flush_four_fingers_makes_across_suits_can_be_declared_and_scores() {
	let declare = ACT_PLAY_HAND + STRAIGHT_FLUSH;
	for cards in SPLIT_STRAIGHT_FLUSHES {
		let mut env = env_at(
			round_with(&[Joker::FourFingers], &cards),
			BalConfig::default(),
		);
		assert_eq!(see(&mut env).mask[declare], 1, "{cards:?} can be declared");
		let info = env.step_select(num::id(declare), &[0, 1, 2, 3, 4]);
		assert_eq!(info.illegal, 0, "{cards:?} can be picked");
		assert_eq!(
			(info.hand_played, info.hs_declared),
			(num::id(STRAIGHT_FLUSH), num::id(STRAIGHT_FLUSH)),
			"{cards:?} is scored as declared"
		);

		let mut plain = env_at(round_with(&[], &cards), BalConfig::default());
		assert_eq!(see(&mut plain).mask[declare], 0, "not without Four Fingers");
	}
}

/// A hand of eight cards of few suits and ranks, so that straights and
/// flushes are common.
fn crowded_hand(rng: &mut Xorshift) -> Vec<(Rank, Suit)> {
	(0..8)
		.map(|_| {
			let suit = if rng.chance(0.7) {
				Suit::Hearts
			} else {
				Suit::Clubs
			};
			(Rank::ALL[rng.below(8)], suit)
		})
		.collect()
}

#[test]
fn the_picks_of_a_declared_straight_flush_are_the_ones_the_scorer_completes() {
	let verb = sel_verb_index(ACT_PLAY_HAND + STRAIGHT_FLUSH).expect("a selection verb");
	let rng = &mut Xorshift::new(0x5eed);
	let (mut stops, mut narrowed) = (0, 0);
	for round in 0..240 {
		let flags = EvalFlags {
			four_fingers: round % 4 != 0,
			shortcut: round % 3 == 0,
			smeared: false,
		};
		let mut jokers = Vec::new();
		if flags.four_fingers {
			jokers.push(Joker::FourFingers);
		}
		if flags.shortcut {
			jokers.push(Joker::Shortcut);
		}
		let cards = crowded_hand(rng);
		let env = env_at(round_with(&jokers, &cards), BalConfig::default());
		let hand = &env.game().areas.hand.cards;
		let selections: Vec<u32> = (0..1u32 << hand.len())
			.filter(|set| set.count_ones() <= 5)
			.collect();
		let scores = |set: u32| {
			let picked: Vec<&PlayingCard> = (0..hand.len())
				.filter(|i| set >> i & 1 != 0)
				.map(|i| &hand[i])
				.collect();
			hand_eval::evaluate(&picked, flags).has(HandType::StraightFlush)
		};
		let complete: Vec<u32> = selections.iter().copied().filter(|&s| scores(s)).collect();
		for &picked in &selections {
			let reaches = |card: usize| {
				let with = picked | 1 << card;
				with != picked && complete.iter().any(|&set| set & with == with)
			};
			let expected: u32 = (0..hand.len())
				.filter(|&card| reaches(card))
				.map(|card| 1 << card)
				.sum();
			let (allowed, stop) = env.pick_rule(verb, 1, 5, 0, picked, 0);
			let what = format!("{cards:?} with {flags:?} after picking {picked:#b}");
			assert_eq!(allowed, expected, "the next cards of {what}");
			assert_eq!(stop, complete.contains(&picked), "stopping in {what}");
			stops += u32::from(stop);
			narrowed += u32::from(allowed != 0 && allowed.count_ones() + picked.count_ones() < 8);
		}
	}
	assert!(stops > 100 && narrowed > 100, "{stops} {narrowed}");
}

#[test]
fn hiding_face_down_cards_shows_the_same_whichever_unseen_card_is_in_the_hand() {
	let mut game = Game::new_run("HIDDEN", Back::Red, Stake::White);
	game.step(Action::SelectBlind)
		.expect("the first blind can be played");
	let dealt = &mut game.areas.hand.cards[2];
	(dealt.core.face_down, dealt.wheel_flipped) = (true, true);
	let (rank, suit) = (dealt.rank, dealt.suit);
	let twin = &mut game.areas.deck[0];
	(twin.rank, twin.suit, twin.seal) = (rank, suit, Some(Seal::Red));

	let mut swapped = game.clone();
	let areas = &mut swapped.areas;
	let (in_hand, in_deck) = (&mut areas.hand.cards[2], &mut areas.deck[0]);
	mem::swap(in_hand, in_deck);
	mem::swap(&mut in_hand.core.face_down, &mut in_deck.core.face_down);
	mem::swap(&mut in_hand.wheel_flipped, &mut in_deck.wheel_flipped);

	let shown = |game: &Game, hide_face_down: bool| {
		let cfg = BalConfig {
			hide_face_down,
			..BalConfig::default()
		};
		see(&mut env_at(game.clone(), cfg))
	};
	assert!(
		shown(&game, false) != shown(&swapped, false),
		"the two games show different cards"
	);
	assert!(
		shown(&game, true) == shown(&swapped, true),
		"with face-down cards hidden they show the same"
	);
}

#[test]
fn a_step_reports_debt_as_negative_dollars() {
	let mut game = Game::new_run("DEBT", Back::Red, Stake::White);
	game.step(Action::SelectBlind)
		.expect("the first blind can be played");
	game.run.dollars = -3.0;
	let mut env = env_at(game, BalConfig::default());
	let info = env.step_select(num::id(ACT_DISCARD), &[0]);
	assert_eq!((info.illegal, info.dollars), (0, -3));
}
