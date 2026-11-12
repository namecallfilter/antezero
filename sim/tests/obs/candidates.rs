//! The scored play candidates.

use sim::Action;
use sim::data::{Back, BlindKind, Edition, Joker};
use sim::game::{Game, Phase};
use sim::inject::Inject;
use sim::obs::{self};

use crate::{Buf, best_candidate, best_legal_play, hand_of, walk};

#[test]
#[expect(clippy::float_cmp, reason = "the test compares exact values")]
fn preview_matches_full_step() {
	let alt = sim::rng::PseudoRandom::new("PREVIEW0");
	let mut checked = 0;
	for s in 0..6 {
		walk(&format!("PV{s}"), 150, |g| {
			if g.phase != Phase::Round {
				return;
			}
			let mut acts = Vec::new();
			g.legal_actions(&mut acts);
			for a in acts
				.iter()
				.filter(|a| matches!(a, Action::Play(_)))
				.take(25)
			{
				let Action::Play(m) = *a else { unreachable!() };
				let p = obs::preview_score(g, m, &alt).unwrap().score;
				let mut c = g.clone();
				c.rng = alt.clone();
				c.step(*a).unwrap();
				assert_eq!(p, c.last_hand_score, "mask {m:b}");
				checked += 1;
			}
		});
	}
	assert!(checked > 100);
}

#[test]
fn face_down_jokers_get_candidates_that_ignore_their_hidden_order() {
	let mut g = Game::new_run("ACORN", Back::Red, 1);
	// +mult before or after xmult changes the score, so the order matters.
	for j in [Joker::Joker, Joker::Cavendish, Joker::Misprint] {
		g.inject(&Inject::Joker {
			joker: j,
			edition: Edition::None,
			eternal: false,
		});
	}
	g.step(Action::SelectBlind).unwrap();
	for c in &mut g.jokers {
		c.face_down = true;
	}
	let top = obs::top_plays(&g, 8);
	assert!(
		!top.is_empty(),
		"face-down jokers must not disable the previews"
	);
	let mut r = g.clone();
	r.jokers.reverse();
	assert_eq!(
		top,
		obs::top_plays(&r, 8),
		"candidates reveal the hidden joker order"
	);
}

#[test]
fn face_down_candidates_do_not_reveal_identity() {
	for seed in ["FDC1", "FDC2", "FDC3", "FDC4"] {
		let mut g = Game::new_run(seed, Back::Red, 1);
		g.step(Action::SelectBlind).unwrap();
		g.hand[0].face_down = true;
		g.hand[3].face_down = true;
		let a = obs::top_plays(&g, 8);
		assert!(!a.is_empty());
		let k = g.deck.len() / 2;
		std::mem::swap(&mut g.hand[0], &mut g.deck[k]);
		g.hand[0].face_down = true;
		g.deck[k].face_down = false;
		g.deck.reverse();
		let b = obs::top_plays(&g, 8);
		assert_eq!(a, b, "candidates reveal a face-down card ({seed})");
	}
}

#[test]
fn candidates_are_consistent() {
	let mut n = 0;
	for s in 0..20 {
		walk(&format!("CD{s}"), 150, |g| {
			let a = obs::top_plays(g, 8);
			let b = obs::top_plays(g, 8);
			assert_eq!(a, b, "candidates must be deterministic");
			for c in &a {
				assert!(c.min <= c.mean + 1e-9 && c.mean <= c.max + 1e-9, "{c:?}");
				assert!((0.0..=1.0).contains(&c.p_clear));
				n += 1;
			}
			for w in a.windows(2) {
				assert!(w[0].p_clear >= w[1].p_clear);
			}
		});
	}
	assert!(n > 200, "only {n} candidates checked");
}

#[test]
fn candidates_never_crowd_out_other_ops() {
	let mut g = Game::new_run("ROOM", Back::Red, 1);
	g.step(Action::SelectBlind).unwrap();
	let mut buf = Buf::new();
	let lay = buf.encode(&g);
	let (mut few, mut many) = (Vec::new(), Vec::new());
	obs::ops(&g, &lay, 0, &mut few);
	obs::ops(&g, &lay, obs::MAX_OPS, &mut many);
	assert!(many.len() <= obs::MAX_OPS);
	let base: Vec<_> = many
		.iter()
		.filter(|o| o.kind != obs::op::PLAY_TOP)
		.copied()
		.collect();
	assert_eq!(base, few, "candidates replaced other ops");
	assert!(many.len() > few.len(), "no candidates");
}

#[test]
fn candidates_ignore_hidden_card_ids() {
	let mut n = 0;
	for s in 0..10 {
		walk(&format!("ID{s}"), 120, |g| {
			if g.phase != Phase::Round {
				return;
			}
			// Misprint makes every score depend on the chance scenarios.
			let mut a = g.clone();
			a.inject(&Inject::Joker {
				joker: Joker::Misprint,
				edition: Edition::None,
				eternal: false,
			});
			let mut r = a.clone();
			for c in &mut r.hand {
				c.id += 100_000;
			}
			let top = obs::top_plays(&a, 8);
			assert!(
				top.iter().all(|c| c.min < c.max),
				"Misprint plays should vary"
			);
			assert_eq!(top, obs::top_plays(&r, 8), "candidates depend on card ids");
			n += 1;
		});
	}
	assert!(n > 20, "only {n} rounds checked");
}

#[test]
#[expect(clippy::float_cmp, reason = "the test compares exact values")]
fn candidates_keep_plays_that_hold_cards_back() {
	// Baron (copied by Blueprint and Brainstorm) scores every King held in
	// hand: the best play is a lone Ace that holds all three Kings, which the
	// cheap estimate ranks below every play with a King in it.
	use sim::data::Suit;
	let mut g = Game::new_run("HOLD", Back::Red, 1);
	for j in [Joker::Blueprint, Joker::Baron, Joker::Brainstorm] {
		g.inject(&Inject::Joker {
			joker: j,
			edition: Edition::None,
			eternal: false,
		});
	}
	g.step(Action::SelectBlind).unwrap();
	g.hand = hand_of(&[
		(13, Suit::Spades),
		(13, Suit::Hearts),
		(13, Suit::Clubs),
		(14, Suit::Diamonds),
		(8, Suit::Clubs),
		(6, Suit::Spades),
		(4, Suit::Hearts),
		(2, Suit::Diamonds),
	]);
	let best = best_legal_play(&g);
	assert_eq!(best_candidate(&g), best, "scored candidates");
	assert_eq!(obs::best_play(&g), best, "build preview");
}

#[test]
#[expect(clippy::float_cmp, reason = "the test compares exact values")]
fn candidates_carry_the_kickers_blackboard_and_raised_fist_need() {
	// No pair, three red cards: with Blackboard the best play is the Ace with all three reds (x3: only black cards
	// stay held); with Raised Fist it is the Ace with the four lowest cards (a Jack is then the lowest card held).
	// Neither is a scoring core, and among the Ace's 98 kicker variants both fall outside the simulated budget.
	use sim::data::Suit;
	for j in [Joker::Blackboard, Joker::RaisedFist] {
		let mut g = Game::new_run("KICK", Back::Red, 1);
		g.inject(&Inject::Joker {
			joker: j,
			edition: Edition::None,
			eternal: false,
		});
		g.step(Action::SelectBlind).unwrap();
		g.hand = hand_of(&[
			(14, Suit::Spades),
			(13, Suit::Clubs),
			(12, Suit::Spades),
			(11, Suit::Clubs),
			(9, Suit::Spades),
			(7, Suit::Hearts),
			(4, Suit::Diamonds),
			(2, Suit::Hearts),
		]);
		let best = best_legal_play(&g);
		assert!(best > 3.0 * 16.0 - 0.5, "{j:?}: best legal play {best}");
		assert_eq!(best_candidate(&g), best, "{j:?}: scored candidates");
		assert_eq!(obs::best_play(&g), best, "{j:?}: build preview");
	}
}

#[test]
#[expect(clippy::float_cmp, reason = "the test compares exact values")]
fn candidates_follow_psychic_and_splash() {
	// No pair: every scoring core is a single card. The Psychic scores only
	// five-card plays; Splash scores every played card (kickers included).
	use sim::data::Suit;
	let mut g = Game::new_run("PSYCHIC", Back::Red, 1);
	g.step(Action::SelectBlind).unwrap();
	g.hand = hand_of(&[
		(14, Suit::Spades),
		(13, Suit::Hearts),
		(12, Suit::Clubs),
		(11, Suit::Diamonds),
		(9, Suit::Spades),
		(7, Suit::Hearts),
		(5, Suit::Clubs),
		(2, Suit::Diamonds),
	]);
	let mut psychic = g.clone();
	psychic.blind.kind = Some(BlindKind::Psychic);
	assert_eq!(
		best_candidate(&psychic),
		best_legal_play(&psychic),
		"The Psychic"
	);
	g.inject(&Inject::Joker {
		joker: Joker::Splash,
		edition: Edition::None,
		eternal: false,
	});
	let best = best_legal_play(&g);
	assert_eq!(best_candidate(&g), best, "Splash candidates");
	assert_eq!(obs::best_play(&g), best, "Splash build preview");
}
