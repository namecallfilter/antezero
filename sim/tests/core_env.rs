//! The native environment (`sim::balatro`) against the game it wraps: what
//! it hides stays hidden, and what it reports is the game's state.

use std::mem;
use std::sync::Arc;

use sim::balatro::{
	ACT_DISCARD, BalConfig, BalEnv, HS_CTX, N_ACTIONS, N_SEL_VERBS, OBS_SIZE, SEL_F, StartPoint,
};
use sim::data::{Back, Seal};
use sim::{Action, Game, Stake, num};

/// An environment whose episode starts in `game`.
fn env_at(game: Game, cfg: BalConfig) -> BalEnv {
	let mut env = BalEnv::new(cfg);
	let start = StartPoint {
		game,
		script: String::new(),
		challenge: String::new(),
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
