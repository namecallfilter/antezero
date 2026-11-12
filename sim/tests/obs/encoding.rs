//! What the observation shows and hides, and the ops it offers.

use sim::Action;
use sim::data::{Back, BlindKind, Edition, HandType, Joker};
use sim::game::{Game, Phase};
use sim::inject::Inject;
use sim::obs::{self, ObsMut};

use crate::{Buf, tok, walk};

#[test]
fn deck_order_is_not_observable() {
	let mut g = Game::new_run("ORDER", Back::Red, 1);
	g.step(Action::SelectBlind).unwrap();
	let mut a = Buf::new();
	a.encode(&g);
	g.deck.reverse();
	let mut b = Buf::new();
	b.encode(&g);
	assert_eq!(a.ti, b.ti);
	assert_eq!(a.tf, b.tf);
	assert_eq!(a.gf, b.gf);
}

#[test]
fn every_op_has_a_legal_completion() {
	let mut ops = Vec::new();
	let mut buf = Buf::new();
	let mut n = 0;
	for s in 0..40 {
		walk(&format!("OP{s}"), 300, |g| {
			let lay = buf.encode(g);
			obs::ops(g, &lay, 4, &mut ops);
			assert!(!ops.is_empty(), "no ops in {:?}", g.phase);
			for o in &ops {
				// minimal card set: required + lowest allowed cards
				let mut m = o.required;
				let mut i = 0;
				while m.count_ones() < u32::from(o.card_min) && i < 32 {
					if o.allowed >> i & 1 == 1 {
						m |= 1 << i;
					}
					i += 1;
				}
				let dest = if o.dest > 0 {
					if o.index == 0 && o.kind != obs::op::MOVE_HAND {
						1
					} else if o.kind == obs::op::MOVE_HAND {
						(m.trailing_zeros() + 1) % u32::from(o.dest)
					} else {
						0
					}
				} else {
					0
				};
				assert!(
					obs::decode(g, o, m, dest).is_some(),
					"op {o:?} has no legal completion in {:?}",
					g.phase
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
			obs::op::SORT_RANK | obs::op::SORT_SUIT | obs::op::MOVE_HAND | obs::op::MOVE_JOKER
		)
	};
	let (mut buf, mut ops) = (Buf::new(), Vec::new());
	let mut offered = |g: &Game| {
		let lay = buf.encode(g);
		obs::ops(g, &lay, 4, &mut ops);
		ops.iter().any(reorder)
	};
	let mut g = Game::new_run("STALL", Back::Red, 1);
	g.step(Action::SelectBlind).unwrap();
	assert_eq!(g.phase, Phase::Round);
	for _ in 0..obs::MAX_REARRANGES {
		assert!(offered(&g));
		g.step(Action::SortHand { by_suit: false }).unwrap();
	}
	assert!(!offered(&g), "reorders still offered after the cap");
	g.step(Action::Discard(1)).unwrap();
	assert!(offered(&g), "another action must offer reorders again");
}

#[test]
fn skipping_can_be_switched_off() {
	let (mut buf, mut ops) = (Buf::new(), Vec::new());
	let mut g = Game::new_run("NOSKIP", Back::Red, 1);
	let lay = buf.encode(&g);
	obs::ops(&g, &lay, 4, &mut ops);
	assert!(ops.iter().any(|o| o.kind == obs::op::SKIP_BLIND));
	g.allow_skip = false;
	obs::ops(&g, &lay, 4, &mut ops);
	assert!(!ops.iter().any(|o| o.kind == obs::op::SKIP_BLIND));
	assert!(ops.iter().any(|o| o.kind == obs::op::SELECT_BLIND));
}

#[test]
fn face_down_cards_are_hidden() {
	let mut g = Game::new_run("HIDE", Back::Red, 1);
	g.step(Action::SelectBlind).unwrap();
	for c in &mut g.hand {
		c.face_down = true;
	}
	let mut b = Buf::new();
	let lay = b.encode(&g);
	for i in 0..lay.hand {
		let t = &b.ti[i * obs::TOK_INT..(i + 1) * obs::TOK_INT];
		assert_eq!(t[1], obs::center::HIDDEN);
		assert_eq!(&t[2..7], &[0, 0, 0, 0, 0]);
	}
	assert!(
		!obs::top_plays(&g, 8).is_empty(),
		"face-down hands get previews over redealt worlds"
	);
}

#[test]
fn face_down_identity_is_not_observable() {
	let mut g = Game::new_run("HIDE2", Back::Red, 1);
	g.step(Action::SelectBlind).unwrap();
	g.hand[0].face_down = true;
	let mut a = Buf::new();
	a.encode(&g);
	// Swap the face-down card with an unseen deck card: same observation.
	let k = g.deck.len() / 2;
	std::mem::swap(&mut g.hand[0], &mut g.deck[k]);
	g.hand[0].face_down = true;
	g.deck[k].face_down = false;
	let mut b = Buf::new();
	b.encode(&g);
	a.assert_same(&b, "swapped face-down card");
}

#[test]
#[expect(
	clippy::many_single_char_names,
	reason = "short names for the games and buffers being compared"
)]
fn resample_hidden_keeps_the_observation() {
	let mut n = 0;
	for s in 0..40 {
		let mut states = Vec::new();
		walk(&format!("RS{s}"), 200, |g| states.push(g.clone()));
		// Every other state, alternately with face-down hand cards.
		for (k, g) in states.into_iter().step_by(2).enumerate() {
			let mut g = g;
			if k % 2 == 0 {
				for c in g.hand.iter_mut().step_by(3) {
					c.face_down = true;
				}
			}
			let mut a = Buf::new();
			a.encode(&g);
			let mut c = g.clone();
			c.resample_hidden(&format!("X{k}"));
			let mut b = Buf::new();
			let lay = b.encode(&c);
			a.assert_same(&b, &format!("resample in {:?}", g.phase));
			let (mut oa, mut ob) = (Vec::new(), Vec::new());
			obs::ops(&g, &lay, 4, &mut oa);
			obs::ops(&c, &lay, 4, &mut ob);
			assert_eq!(oa, ob, "ops differ after resample in {:?}", g.phase);
			// The resampled game is still playable.
			let mut acts = Vec::new();
			c.legal_actions(&mut acts);
			if let Some(&a0) = acts.first() {
				c.step(a0).unwrap();
			}
			n += 1;
		}
	}
	assert!(n > 100, "only {n} states checked");
}

#[test]
fn resample_hidden_ignores_the_hidden_arrangement() {
	// A world is a function of the visible state and the seed: reordering the
	// deck, swapping a face-down card with a deck card and reordering
	// face-down jokers (all hidden) must give the identical world.
	let (mut n, mut swaps, mut jokers) = (0, 0, 0);
	for s in 0..60 {
		let mut states = Vec::new();
		walk(&format!("RI{s}"), 300, |g| states.push(g.clone()));
		for (k, mut g) in states.into_iter().step_by(2).enumerate() {
			if k % 2 == 0 {
				for c in g.hand.iter_mut().step_by(3) {
					c.face_down = true;
				}
				for j in [Joker::Joker, Joker::Cavendish, Joker::Misprint] {
					g.inject(&Inject::Joker {
						joker: j,
						edition: Edition::None,
						eternal: false,
					});
				}
				for c in &mut g.jokers {
					c.face_down = true;
				}
			}
			let mut h = g.clone();
			h.deck.reverse();
			if k % 2 == 0 && h.jokers.len() > 1 {
				h.jokers.reverse();
				jokers += 1;
			}
			if let Some(i) = h.hand.iter().position(|c| c.face_down)
				&& !h.deck.is_empty()
			{
				let j = h.deck.len() / 2;
				std::mem::swap(&mut h.hand[i], &mut h.deck[j]);
				(h.hand[i].face_down, h.hand[i].ab.wheel_flipped) =
					(g.hand[i].face_down, g.hand[i].ab.wheel_flipped);
				(h.deck[j].face_down, h.deck[j].ab.wheel_flipped) =
					(g.deck[j].face_down, g.deck[j].ab.wheel_flipped);
				swaps += 1;
			}
			let (mut a, mut b) = (g.clone(), h);
			a.resample_hidden(&format!("W{k}"));
			b.resample_hidden(&format!("W{k}"));
			assert_eq!(
				sim::canon::from_sim(&mut a),
				sim::canon::from_sim(&mut b),
				"world depends on the hidden arrangement ({s}/{k}, {:?})",
				g.phase
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
	let mut g = Game::new_run("TIES", Back::Red, 1);
	g.step(Action::SelectBlind).unwrap();
	// Two cards that agree on every categorical feature but not on a bonus.
	let last = g.deck.len() - 1;
	let (suit, rank) = (g.deck[0].suit, g.deck[0].rank);
	g.deck[last].suit = suit;
	g.deck[last].rank = rank;
	g.deck[last].ab.perma_bonus = 10.0;
	for deck_tokens in [false, true] {
		let enc = |g: &Game| {
			let mut b = Buf::new();
			obs::encode_opts(
				g,
				&mut ObsMut {
					tok_i: &mut b.ti,
					tok_f: &mut b.tf,
					glob_i: &mut b.gi,
					glob_f: &mut b.gf,
				},
				deck_tokens,
			);
			b
		};
		let a = enc(&g);
		let mut r = g.clone();
		r.deck.reverse();
		a.assert_same(
			&enc(&r),
			&format!("reversed deck (deck_tokens={deck_tokens})"),
		);
	}
}

#[test]
fn blind_tokens_show_the_hand_they_name() {
	let mut g = Game::new_run("NAMED", Back::Red, 1);
	g.boss_choice = BlindKind::Ox;
	g.most_played_hand = HandType::Flush;
	let mut b = Buf::new();
	b.encode(&g);
	let boss = (0..obs::MAX_TOKENS)
		.find(|&i| tok(&b, i)[0] == obs::zone::BLIND && tok(&b, i)[7] == 2)
		.unwrap();
	assert_eq!(tok(&b, boss)[3], HandType::Flush as i32 + 1);
	// The Mouth, once a hand has fixed it.
	g.boss_choice = BlindKind::Mouth;
	g.blind.kind = Some(BlindKind::Mouth);
	g.blind.only_hand = Some(HandType::Pair);
	b.encode(&g);
	assert_eq!(tok(&b, boss)[3], HandType::Pair as i32 + 1);
}

#[test]
fn hand_can_be_reordered_in_packs() {
	let mut buf = Buf::new();
	let mut ops = Vec::new();
	let mut packs = 0;
	for s in 0..60 {
		walk(&format!("PK{s}"), 300, |g| {
			if g.phase == Phase::Pack && g.hand.len() >= 2 {
				let lay = buf.encode(g);
				obs::ops(g, &lay, 0, &mut ops);
				for k in [obs::op::SORT_RANK, obs::op::SORT_SUIT, obs::op::MOVE_HAND] {
					assert!(
						ops.iter().any(|o| o.kind == k),
						"op {k} missing in a pack with the hand drawn"
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
	let mut g = Game::new_run("DEST", Back::Red, 1);
	g.step(Action::SelectBlind).unwrap();
	let mut buf = Buf::new();
	let lay = buf.encode(&g);
	let mut ops = Vec::new();
	obs::ops(&g, &lay, 0, &mut ops);
	let mv = ops.iter().find(|o| o.kind == obs::op::MOVE_HAND).unwrap();
	assert_eq!(
		obs::decode(&g, mv, 1, 2),
		Some(Action::MoveHand { from: 0, to: 2 })
	);
	// 258 would wrap to 2 as a u8.
	assert_eq!(obs::decode(&g, mv, 1, 258), None);
	assert_eq!(obs::decode(&g, mv, 1, u32::from(mv.dest)), None);
}
