//! Agent support: determinization of hidden information and play previews
//! (used by search, candidate scoring and snapshot restarts).

use crate::actions::{Action, ActionError, CardMask};
use crate::card::Card;
use crate::game::Game;
use crate::obs::canon_key;
use crate::rng::{Key, PseudoRandom, shuffle};
use crate::{events, flow, pools};

/// See [`Game::resample_hidden`].
pub(crate) fn resample_hidden(g: &mut Game, seed: &str) {
	#[cfg(feature = "trace-keys")]
	let trace = std::mem::take(&mut g.rng.trace);
	g.rng = PseudoRandom::new(seed);
	#[cfg(feature = "trace-keys")]
	{
		g.rng.trace = trace;
	}
	let key = |s: &str| Key::from(s);
	let order = |c: &Card| (canon_key(c), c.id);
	// Face-down hand cards: shuffle them into the unseen pool and deal
	// the same slots again; their face-down state stays with the slot.
	let down: Vec<usize> = (0..g.hand.len()).filter(|&i| g.hand[i].face_down).collect();
	if !down.is_empty() {
		// Flags of a card lying in the deck, for hand cards moved there.
		let deck_flags = g
			.deck
			.first()
			.map_or((false, false), |c| (c.face_down, c.ab.wheel_flipped));
		let mut pool: Vec<(Card, bool)> = down.iter().map(|&i| (g.hand[i].clone(), true)).collect();
		pool.extend(g.deck.drain(..).map(|c| (c, false)));
		pool.sort_by_key(|(c, _)| order(c));
		let s = g.rng.seed_for(&key("resample_hand"));
		shuffle(&mut pool, s);
		let rest = pool.split_off(down.len());
		for ((mut c, _), &i) in pool.into_iter().zip(&down) {
			c.face_down = g.hand[i].face_down;
			c.ab.wheel_flipped = g.hand[i].ab.wheel_flipped;
			g.hand[i] = c;
		}
		g.deck = rest
			.into_iter()
			.map(|(mut c, from_hand)| {
				if from_hand {
					(c.face_down, c.ab.wheel_flipped) = deck_flags;
				}
				c
			})
			.collect();
	}
	g.deck.sort_by_key(order);
	let s = g.rng.seed_for(&key("resample"));
	shuffle(&mut g.deck, s);
	// Face-down jokers: permute among their slots.
	let jd: Vec<usize> = (0..g.jokers.len())
		.filter(|&i| g.jokers[i].face_down)
		.collect();
	if jd.len() > 1 {
		let mut js: Vec<Card> = jd.iter().map(|&i| g.jokers[i].clone()).collect();
		js.sort_by_key(|c| c.id);
		let s = g.rng.seed_for(&key("resample_jokers"));
		shuffle(&mut js, s);
		for (c, &i) in js.into_iter().zip(&jd) {
			g.jokers[i] = c;
		}
	}
	if !g.voucher_seen && g.current_voucher.is_some() && !g.shop_open {
		g.current_voucher = Some(pools::get_next_voucher(g, false));
	}
}

/// See [`Game::preview_play`].
pub(crate) fn preview_play(g: &mut Game, mask: CardMask) -> Result<f64, ActionError> {
	let a = Action::Play(mask);
	g.check(&a)?;
	events::begin_action(g);
	flow::do_play(g, mask);
	events::run_until_scored(g);
	assert!(g.scored, "preview_play: the hand was never scored");
	Ok(g.last_hand_score)
}
