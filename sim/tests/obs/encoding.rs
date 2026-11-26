//! What the observation shows and hides, and the ops it offers.

use std::mem;

use sim::data::{Back, BlindKind, HandType, Joker};
use sim::game::{BlindSlot, Game, HandSort, Phase};
use sim::inject::Inject;
use sim::obs::{self, ObsBuf};
use sim::{Action, AnyCard, Card, CardMask, Kind, Stake, canon, num};

use crate::{assert_same, encode_with_deck, first_shop, observe, tok, walk};

#[test]
fn deck_order_is_not_observable() {
	let mut g = Game::new_run("ORDER", Back::Red, Stake::White);
	g.step(Action::SelectBlind).unwrap();
	let a = observe(&g);
	g.areas.deck.reverse();
	assert_same(&a, &observe(&g), "reversed deck");
}

#[test]
fn every_op_has_a_legal_completion() {
	let mut ops = Vec::new();
	let mut buf = ObsBuf::new();
	let mut n = 0;
	for s in 0..40 {
		walk(&format!("OP{s}"), 300, |g| {
			let lay = encode_with_deck(g, &mut buf);
			obs::ops(
				g,
				&lay,
				obs::OpOptions {
					candidates: 4,
					..Default::default()
				},
				&mut ops,
			);
			assert!(!ops.is_empty(), "no ops in {:?}", g.flow.phase);
			for o in &ops {
				let m = o.minimal_cards();
				let dest = o.dest.map(|slots| match o.kind {
					obs::OpKind::MoveHand => {
						let from = m.lowest().expect("a hand move takes a card");
						num::byte((from + 1) % usize::from(slots))
					}
					_ => u8::from(o.index == 0),
				});
				assert!(
					obs::decode(g, o, m, dest).is_some(),
					"op {o:?} has no legal completion in {:?}",
					g.flow.phase
				);
				n += 1;
			}
		});
	}
	assert!(n > 500, "only {n} ops checked");
}

#[test]
fn reorders_are_capped_until_another_action() {
	let reorder = |o: &obs::Op| {
		matches!(
			o.kind,
			obs::OpKind::SortRank
				| obs::OpKind::SortSuit
				| obs::OpKind::MoveHand
				| obs::OpKind::MoveJoker
		)
	};
	let (mut buf, mut ops) = (ObsBuf::new(), Vec::new());
	let mut offered = |g: &Game| {
		let lay = encode_with_deck(g, &mut buf);
		obs::ops(
			g,
			&lay,
			obs::OpOptions {
				candidates: 4,
				..Default::default()
			},
			&mut ops,
		);
		ops.iter().any(reorder)
	};
	let mut g = Game::new_run("STALL", Back::Red, Stake::White);
	g.step(Action::SelectBlind).unwrap();
	assert_eq!(g.flow.phase, Phase::Round);
	for i in 0..obs::MAX_REARRANGES {
		assert!(
			offered(&g),
			"reorders not offered after {i} of the {} allowed",
			obs::MAX_REARRANGES
		);
		g.step(Action::SortHand(HandSort::Rank)).unwrap();
	}
	assert!(!offered(&g), "reorders still offered after the cap");
	g.step(Action::Discard(CardMask::from_bits(1))).unwrap();
	assert!(offered(&g), "another action must offer reorders again");
}

#[test]
fn skipping_can_be_switched_off() {
	let (mut buf, mut ops) = (ObsBuf::new(), Vec::new());
	let g = Game::new_run("NOSKIP", Back::Red, Stake::White);
	let lay = encode_with_deck(&g, &mut buf);
	obs::ops(
		&g,
		&lay,
		obs::OpOptions {
			candidates: 4,
			..Default::default()
		},
		&mut ops,
	);
	assert!(ops.iter().any(|o| o.kind == obs::OpKind::SkipBlind));
	obs::ops(
		&g,
		&lay,
		obs::OpOptions {
			candidates: 4,
			allow_skip: false,
			..Default::default()
		},
		&mut ops,
	);
	assert!(!ops.iter().any(|o| o.kind == obs::OpKind::SkipBlind));
	assert!(ops.iter().any(|o| o.kind == obs::OpKind::SelectBlind));
}

#[test]
fn face_down_cards_are_hidden() {
	let mut g = Game::new_run("HIDE", Back::Red, Stake::White);
	g.step(Action::SelectBlind).unwrap();
	for c in &mut g.areas.hand.cards {
		c.core.face_down = true;
	}
	let mut b = ObsBuf::new();
	let lay = encode_with_deck(&g, &mut b);
	for i in 0..lay.hand {
		let t = tok(&b, i);
		assert_eq!(t.center, obs::center::HIDDEN);
		assert_eq!(
			(t.suit, t.rank, t.enhancement, t.edition, t.seal),
			(0, 0, 0, 0, 0),
			"a face-down card shows its face"
		);
	}
	assert!(
		!obs::top_plays(&g, 8).is_empty(),
		"face-down hands get previews over redealt worlds"
	);
}

#[test]
fn face_down_identity_is_not_observable() {
	let mut g = Game::new_run("HIDE2", Back::Red, Stake::White);
	g.step(Action::SelectBlind).unwrap();
	g.areas.hand.cards[0].core.face_down = true;
	let a = observe(&g);
	// Swap the face-down card with an unseen deck card: same observation.
	let k = g.areas.deck.len() / 2;
	mem::swap(&mut g.areas.hand.cards[0], &mut g.areas.deck[k]);
	g.areas.hand.cards[0].core.face_down = true;
	g.areas.deck[k].core.face_down = false;
	assert_same(&a, &observe(&g), "swapped face-down card");
}

#[test]
fn resample_hidden_keeps_the_observation() {
	let mut checked = 0;
	for s in 0..40 {
		let states = every_other_state(&format!("RS{s}"), 200);
		for (k, mut g) in states.into_iter().enumerate() {
			if k % 2 == 0 {
				for card in g.areas.hand.cards.iter_mut().step_by(3) {
					card.core.face_down = true;
				}
			}
			let before = observe(&g);
			let mut resampled = g.clone();
			resampled.resample_hidden(&format!("X{k}"));
			let mut after = ObsBuf::new();
			let lay = encode_with_deck(&resampled, &mut after);
			assert_same(&before, &after, &format!("resample in {:?}", g.flow.phase));
			let (mut ops_before, mut ops_after) = (Vec::new(), Vec::new());
			obs::ops(
				&g,
				&lay,
				obs::OpOptions {
					candidates: 4,
					..Default::default()
				},
				&mut ops_before,
			);
			obs::ops(
				&resampled,
				&lay,
				obs::OpOptions {
					candidates: 4,
					..Default::default()
				},
				&mut ops_after,
			);
			obs::order_gain(&g, &mut ops_before);
			obs::order_gain(&resampled, &mut ops_after);
			assert_eq!(
				ops_before, ops_after,
				"ops differ after resample in {:?}",
				g.flow.phase
			);
			// The resampled game is still playable.
			let mut acts = Vec::new();
			resampled.legal_actions(&mut acts);
			let first = *acts
				.first()
				.expect("a resampled state that is not over has a legal action");
			resampled.step(first).unwrap();
			checked += 1;
		}
	}
	assert!(checked > 100, "only {checked} states checked");
}

#[test]
fn resample_hidden_ignores_the_hidden_arrangement() {
	// A world is a function of the visible state and the seed: reordering the
	// deck, swapping a face-down card with a deck card and reordering
	// face-down jokers (all hidden) must give the identical world.
	let (mut n, mut swaps, mut jokers) = (0, 0, 0);
	for s in 0..60 {
		let states = every_other_state(&format!("RI{s}"), 300);
		for (k, mut g) in states.into_iter().enumerate() {
			if k % 2 == 0 {
				for c in g.areas.hand.cards.iter_mut().step_by(3) {
					c.core.face_down = true;
				}
				for j in [Joker::Joker, Joker::Cavendish, Joker::Misprint] {
					g.inject(Inject::joker(j));
				}
				for c in &mut g.areas.jokers {
					c.core.face_down = true;
				}
			}
			let mut h = g.clone();
			h.areas.deck.reverse();
			if k % 2 == 0 && h.areas.jokers.len() > 1 {
				h.areas.jokers.reverse();
				jokers += 1;
			}
			if let Some(i) = h.areas.hand.cards.iter().position(|c| c.core.face_down)
				&& !h.areas.deck.is_empty()
			{
				let j = h.areas.deck.len() / 2;
				mem::swap(&mut h.areas.hand.cards[i], &mut h.areas.deck[j]);
				(
					h.areas.hand.cards[i].core.face_down,
					h.areas.hand.cards[i].wheel_flipped,
				) = (
					g.areas.hand.cards[i].core.face_down,
					g.areas.hand.cards[i].wheel_flipped,
				);
				(
					h.areas.deck[j].core.face_down,
					h.areas.deck[j].wheel_flipped,
				) = (
					g.areas.deck[j].core.face_down,
					g.areas.deck[j].wheel_flipped,
				);
				swaps += 1;
			}
			let phase = g.flow.phase;
			let (mut a, mut b) = (g, h);
			a.resample_hidden(&format!("W{k}"));
			b.resample_hidden(&format!("W{k}"));
			assert_eq!(
				canon::from_sim(&mut a),
				canon::from_sim(&mut b),
				"world depends on the hidden arrangement ({s}/{k}, {phase:?})"
			);
			n += 1;
		}
	}
	assert!(
		n > 150 && swaps > 40 && jokers > 80,
		"only {n} states, {swaps} face-down swaps, {jokers} joker reorders checked"
	);
}

#[test]
fn deck_token_ties_do_not_leak_order() {
	let mut g = Game::new_run("TIES", Back::Red, Stake::White);
	g.step(Action::SelectBlind).unwrap();
	// Two cards that agree on every categorical feature but not on a bonus.
	let last = g.areas.deck.len() - 1;
	let (suit, rank) = (g.areas.deck[0].suit, g.areas.deck[0].rank);
	g.areas.deck[last].suit = suit;
	g.areas.deck[last].rank = rank;
	g.areas.deck[last].perma_bonus = 10.0;
	for deck_tokens in [false, true] {
		let enc = |g: &Game| {
			let mut b = ObsBuf::new();
			obs::encode_opts(g, &mut b.as_mut(), deck_tokens);
			b
		};
		let a = enc(&g);
		let mut r = g.clone();
		r.areas.deck.reverse();
		assert_same(
			&a,
			&enc(&r),
			&format!("reversed deck (deck_tokens={deck_tokens})"),
		);
	}
}

#[test]
fn blind_tokens_show_the_hand_they_name() {
	let mut g = Game::new_run("NAMED", Back::Red, Stake::White);
	g.blind_select.boss = BlindKind::Ox;
	g.round.most_played_hand = HandType::Flush;
	let mut b = observe(&g);
	let boss = (0..obs::MAX_TOKENS)
		.find(|&i| {
			let t = tok(&b, i);
			t.zone == obs::Zone::Blind.id() && t.slot == BlindSlot::Boss as i32
		})
		.unwrap();
	// The rank column of a blind names its hand.
	assert_eq!(tok(&b, boss).rank, obs::hand_id(Some(HandType::Flush)));
	// The Mouth, once a hand has fixed it.
	g.blind_select.boss = BlindKind::Mouth;
	g.blind.kind = Some(BlindKind::Mouth);
	g.blind.only_hand = Some(HandType::Pair);
	encode_with_deck(&g, &mut b);
	assert_eq!(tok(&b, boss).rank, obs::hand_id(Some(HandType::Pair)));
}

#[test]
fn joker_targets_show_before_the_joker_is_owned() {
	for joker in [Joker::Idol, Joker::Mail, Joker::Ancient, Joker::Castle] {
		let mut g = first_shop("TARGET", 0);
		g.inject(Inject::joker(joker));
		let offered = AnyCard::Card(Card::new(999, Kind::Joker(joker)));
		g.areas.shop_jokers = vec![offered.clone()];
		g.areas.pack_cards = vec![offered];
		let b = observe(&g);
		let shown = |zone: obs::Zone| {
			let i = (0..obs::MAX_TOKENS)
				.rfind(|&i| tok(&b, i).zone == zone.id())
				.unwrap();
			[tok(&b, i).suit, tok(&b, i).rank]
		};
		let owned = shown(obs::Zone::Joker);
		assert_ne!(owned, [0, 0], "{joker:?}");
		assert_eq!(shown(obs::Zone::ShopCard), owned, "{joker:?} in the shop");
		assert_eq!(shown(obs::Zone::Pack), owned, "{joker:?} in a pack");
	}
}

#[test]
fn hand_can_be_reordered_in_packs() {
	let mut buf = ObsBuf::new();
	let mut ops = Vec::new();
	let mut packs = 0;
	for s in 0..60 {
		walk(&format!("PK{s}"), 300, |g| {
			if g.flow.phase == Phase::Pack && g.areas.hand.cards.len() >= 2 {
				let lay = encode_with_deck(g, &mut buf);
				obs::ops(
					g,
					&lay,
					obs::OpOptions {
						candidates: 0,
						..Default::default()
					},
					&mut ops,
				);
				for k in [
					obs::OpKind::SortRank,
					obs::OpKind::SortSuit,
					obs::OpKind::MoveHand,
				] {
					assert!(
						ops.iter().any(|o| o.kind == k),
						"op {k:?} missing in a pack with the hand drawn"
					);
				}
				packs += 1;
			}
		});
	}
	assert!(packs > 0, "no pack opened with the hand drawn");
}

#[test]
fn reorder_destinations_are_checked() {
	let mut g = Game::new_run("DEST", Back::Red, Stake::White);
	g.step(Action::SelectBlind).unwrap();
	let mut buf = ObsBuf::new();
	let lay = encode_with_deck(&g, &mut buf);
	let mut ops = Vec::new();
	obs::ops(
		&g,
		&lay,
		obs::OpOptions {
			candidates: 0,
			..Default::default()
		},
		&mut ops,
	);
	let mv = ops
		.iter()
		.find(|o| o.kind == obs::OpKind::MoveHand)
		.unwrap();
	assert_eq!(
		obs::decode(&g, mv, CardMask::from_bits(1), Some(2)),
		Some(Action::MoveHand { from: 0, to: 2 })
	);
	assert_eq!(obs::decode(&g, mv, CardMask::from_bits(1), None), None);
	assert_eq!(obs::decode(&g, mv, CardMask::from_bits(1), mv.dest), None);
}

/// The sorts offered in `g` with `null_sorts` off.
fn sorts(g: &Game) -> Vec<obs::OpKind> {
	let mut buf = ObsBuf::new();
	let lay = encode_with_deck(g, &mut buf);
	let mut ops = Vec::new();
	obs::ops(
		g,
		&lay,
		obs::OpOptions {
			null_sorts: false,
			..Default::default()
		},
		&mut ops,
	);
	ops.iter()
		.map(|o| o.kind)
		.filter(|k| matches!(k, obs::OpKind::SortRank | obs::OpKind::SortSuit))
		.collect()
}

#[test]
fn a_sort_that_changes_nothing_can_be_left_out() {
	use obs::OpKind::{SortRank, SortSuit};
	let mut g = Game::new_run("NULLSORT", Back::Red, Stake::White);
	g.step(Action::SelectBlind).unwrap();
	// A dealt hand is in rank order: only the other sort does anything.
	assert_eq!(sorts(&g), [SortSuit]);
	g.step(Action::SortHand(HandSort::Suit)).unwrap();
	assert_eq!(sorts(&g), [SortRank]);
	// Moved out of order, both sorts change the hand again.
	let last = num::byte(g.areas.hand.cards.len() - 1);
	g.step(Action::SortHand(HandSort::Rank)).unwrap();
	g.step(Action::Discard(CardMask::from_bits(1))).unwrap();
	g.step(Action::MoveHand { from: 0, to: last }).unwrap();
	assert_eq!(sorts(&g), [SortRank, SortSuit]);
	// Face-down cards hide whether the hand is in order.
	g.step(Action::SortHand(HandSort::Rank)).unwrap();
	g.step(Action::Discard(CardMask::from_bits(1))).unwrap();
	assert_eq!(sorts(&g), [SortSuit]);
	g.areas.hand.cards[0].core.face_down = true;
	assert_eq!(sorts(&g), [SortRank, SortSuit]);
}

/// Every other state of a [`walk`] from `seed` (the first, the third, ...).
fn every_other_state(seed: &str, steps: usize) -> Vec<Game> {
	let (mut states, mut keep) = (Vec::new(), true);
	walk(seed, steps, |g| {
		if keep {
			states.push(g.clone());
		}
		keep = !keep;
	});
	states
}

#[test]
fn obs_v2_encodes_last_hand_and_preserves_v1_prefix() {
	let mut g = Game::new_run("OBSV2", Back::Red, Stake::White);
	g.step(Action::SelectBlind).unwrap();
	g.step(Action::Play(CardMask::from_bits(1))).unwrap();
	assert!(g.run.last_hand_score > 0.0);
	assert!(g.run.last_hand_chips > 0.0);
	assert!(g.run.last_hand_mult > 0.0);

	let mut v1 = ObsBuf::new();
	obs::encode(&g, &mut v1.as_mut());
	let mut v2 = ObsBuf::new_v2();
	obs::encode(&g, &mut v2.as_mut());

	assert_eq!(v1.glob_f.len(), obs::GLOB_FLOAT);
	assert_eq!(v2.glob_f.len(), obs::GLOB_FLOAT_V2);
	assert_eq!(&v2.glob_f[..obs::GLOB_FLOAT], &v1.glob_f[..]);
	assert!(v2.glob_f[obs::GLOB_FLOAT] > 0.0);
	assert!(v2.glob_f[obs::GLOB_FLOAT + 1] > 0.0);
	assert!(v2.glob_f[obs::GLOB_FLOAT + 2] > 0.0);
	assert!(v2.glob_f[obs::GLOB_FLOAT + 3] > 0.0);
}

#[test]
fn obs_v2_keeps_the_blind_share_of_the_hand_that_beat_the_blind() {
	let mut g = Game::new_run("OBSV2", Back::Red, Stake::White);
	g.step(Action::SelectBlind).unwrap();
	g.inject(Inject::Joker {
		joker: Joker::Joker,
		edition: None,
		eternal: false,
	});
	let blind = g.blind.chips;
	while g.flow.phase == Phase::Round {
		g.step(Action::Play(CardMask::from_bits(0b1_1111))).unwrap();
	}
	assert_eq!(g.flow.phase, Phase::RoundEval);
	assert_eq!(g.blind.chips, 0.0);
	let share = g.run.last_hand_score / blind;
	assert!(share > 0.0 && share < 4.0);

	let mut v2 = ObsBuf::new_v2();
	obs::encode(&g, &mut v2.as_mut());
	assert_eq!(v2.glob_f[obs::GLOB_FLOAT + 3], num::feat(share));
}

#[test]
fn max_rearranges_is_configurable_in_op_options() {
	let mut g = Game::new_run("REARR", Back::Red, Stake::White);
	g.step(Action::SelectBlind).unwrap();
	let mut buf = ObsBuf::new();
	let lay = encode_with_deck(&g, &mut buf);
	let mut ops = Vec::new();
	for _ in 0..3 {
		g.step(Action::SortHand(HandSort::Rank)).unwrap();
	}
	obs::ops(
		&g,
		&lay,
		obs::OpOptions {
			max_rearranges: 2,
			..Default::default()
		},
		&mut ops,
	);
	assert!(!ops.iter().any(|o| matches!(
		o.kind,
		obs::OpKind::SortRank | obs::OpKind::SortSuit | obs::OpKind::MoveHand
	)));
	obs::ops(
		&g,
		&lay,
		obs::OpOptions {
			max_rearranges: 5,
			..Default::default()
		},
		&mut ops,
	);
	assert!(ops.iter().any(|o| matches!(
		o.kind,
		obs::OpKind::SortRank | obs::OpKind::SortSuit | obs::OpKind::MoveHand
	)));
}
