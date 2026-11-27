//! The native environment (`sim::balatro`) against the game it wraps: what
//! it lets a policy declare is what the game scores, what it hides stays
//! hidden, and what it reports is the game's state.

use std::mem;
use std::sync::Arc;

use sim::balatro::{
	ACT_DISCARD, ACT_PLAY, ACT_PLAY_HAND, ACT_SELECT_BLIND, BalConfig, BalEnv, HS_CTX, N_ACTIONS,
	N_SEL_VERBS, OBS_GROUPS, OBS_SIZE, SEL_F, StartPoint, sel_verb_index,
};
use sim::data::{Back, BlindKind, HandType, Joker, Rank, Seal, Suit};
use sim::hand_eval::{self, EvalFlags};
use sim::inject::Inject;
use sim::rng::Xorshift;
use sim::{Action, Forced, Game, Kind, PlayingCard, Stake, num};

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

/// Where the observation's rows of the hand cards start, how many they are
/// and how long each is.
fn hand_rows() -> (usize, usize, usize) {
	let before = OBS_GROUPS.iter().take_while(|(name, ..)| *name != "hand");
	let start = before.map(|(_, rows, columns)| rows * columns).sum();
	let (_, rows, columns) = OBS_GROUPS
		.iter()
		.find(|(name, ..)| *name == "hand")
		.expect("the observation has hand rows");
	(start, *rows, *columns)
}

/// A round whose one joker is `joker`, face down, with hand cards that every
/// joker that marks cards marks, and the same game with a plain Joker there.
fn face_down_joker_and_plain(joker: Joker) -> (Game, Game) {
	let mut game = round_with(&[joker], &[(Rank::Two, Suit::Hearts); 3]);
	let round = game.round;
	let hand = &mut game.areas.hand.cards;
	(hand[0].rank, hand[0].suit) = (round.idol_card.rank, round.idol_card.suit);
	(hand[1].rank, hand[1].suit) = (round.mail_rank.rank, round.ancient_suit);
	hand[2].suit = round.castle_suit;
	game.areas.jokers[0].core.face_down = true;
	let mut plain = game.clone();
	plain.areas.jokers[0].kind = Kind::Joker(Joker::Joker);
	(game, plain)
}

fn shown(game: &Game, hide_face_down: bool) -> Seen {
	let cfg = BalConfig {
		hide_face_down,
		..BalConfig::default()
	};
	see(&mut env_at(game.clone(), cfg))
}

#[test]
fn hiding_face_down_cards_shows_the_same_whichever_joker_lies_face_down() {
	let (start, rows, columns) = hand_rows();
	let hand = start..start + rows * columns;
	for joker in [Joker::Idol, Joker::Mail, Joker::Ancient, Joker::Castle] {
		let (game, plain) = face_down_joker_and_plain(joker);
		assert!(
			shown(&game, false).obs[hand.clone()] != shown(&plain, false).obs[hand.clone()],
			"{joker:?} shows on the hand cards"
		);
		assert!(
			shown(&game, true) == shown(&plain, true),
			"a face-down {joker:?} shows nothing"
		);
	}
}

#[test]
fn a_face_down_joker_the_deck_view_counts_with_shows_unless_it_is_debuffed() {
	let (start, rows, columns) = hand_rows();
	let hand = start..start + rows * columns;
	for joker in [Joker::Smeared, Joker::Pareidolia] {
		let (mut game, mut plain) = face_down_joker_and_plain(joker);
		assert!(
			shown(&game, true).obs[hand.clone()] != shown(&plain, true).obs[hand.clone()],
			"a face-down {joker:?} shows on the hand cards"
		);
		game.areas.jokers[0].core.debuff = true;
		plain.areas.jokers[0].core.debuff = true;
		assert!(
			shown(&game, true) == shown(&plain, true),
			"a debuffed face-down {joker:?} shows nothing"
		);
	}
}

#[test]
fn a_forced_card_past_the_observed_hand_is_part_of_every_selection() {
	let (_, rows, _) = hand_rows();
	let mut game = Game::new_run("FORCED", Back::Red, Stake::White);
	game.step(Action::SelectBlind)
		.expect("the first blind can be played");
	let areas = &mut game.areas;
	areas.hand.cards.extend(areas.deck.drain(..20));
	let held = areas.hand.cards.len();
	let forced = &mut areas.hand.cards[held - 1];
	forced.forced = Forced::Locked;
	let forced = forced.core.id;

	for action in [ACT_PLAY, ACT_DISCARD] {
		let mut env = env_at(game.clone(), BalConfig::default());
		let seen = see(&mut env);
		let hand = &env.game().areas.hand.cards;
		let at = hand.iter().position(|card| card.core.id == forced);
		let at = at.expect("the forced card is in the hand");
		assert!(at < rows, "the forced card has a row");
		assert_eq!(seen.hctx[at] >> 9 & 1, 1, "and is shown as forced");
		let verb = sel_verb_index(action).expect("a selection verb");
		let pick = seen.specs[verb * 3 + 2].trailing_zeros().cast_signed();
		let info = env.step_select(num::id(action), &[pick]);
		assert_eq!(info.illegal, 0, "action {action} takes the forced card");
		let hand = &env.game().areas.hand.cards;
		assert_eq!(hand.len(), held - 2, "it takes two cards");
		assert!(
			hand.iter().all(|card| card.core.id != forced),
			"the forced card is one of them"
		);
	}
}

#[test]
fn a_card_forced_during_a_step_is_moved_into_the_observed_hand() {
	let (_, rows, _) = hand_rows();
	let mut game = Game::new_run("BELL", Back::Red, Stake::White);
	game.step(Action::SelectBlind)
		.expect("the first blind can be played");
	game.blind.kind = Some(BlindKind::FinalBell);
	game.round.discards_left = 8.0;
	let areas = &mut game.areas;
	areas.hand.cards.extend(areas.deck.drain(..30));
	let cfg = BalConfig {
		record: true,
		..BalConfig::default()
	};
	let mut env = env_at(game, cfg);
	assert!(!env.script().contains("move_hand"), "no card is forced yet");

	let verb = sel_verb_index(ACT_DISCARD).expect("a selection verb");
	for _ in 0..8 {
		let pick = see(&mut env).specs[verb * 3 + 2]
			.trailing_zeros()
			.cast_signed();
		let info = env.step_select(num::id(ACT_DISCARD), &[pick]);
		assert_eq!(info.illegal, 0, "the discard takes the forced card");
		let hand = &env.game().areas.hand.cards;
		let forced = hand.iter().rposition(|card| card.forced.is_locked());
		let forced = forced.expect("Cerulean Bell forces a card after every discard");
		assert!(forced < rows, "the forced card has a row");
	}
	assert!(
		env.script().contains("move_hand"),
		"a card past the rows was forced and moved"
	);
}

#[test]
fn a_round_that_starts_without_discards_is_not_a_dead_end() {
	let mut game = Game::new_run("NODISCARD", Back::Red, Stake::White);
	game.inject(Inject::Joker {
		joker: Joker::Burglar,
		edition: None,
		eternal: true,
	});
	let mut env = env_at(game, BalConfig::default());
	let info = env.step_select(num::id(ACT_SELECT_BLIND), &[]);
	assert_eq!(env.game().round.discards_left, 0.0, "Burglar took them");
	assert_eq!((info.illegal, info.terminated, info.dead_end), (0, 0, 0));
	assert_eq!(see(&mut env).mask[ACT_PLAY], 1, "a hand can be played");
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
