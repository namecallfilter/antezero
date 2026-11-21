//! Determinization: resampling what a player cannot know (used by search,
//! candidate scoring and snapshot restarts).

use crate::areas::redeal_unseen;
use crate::card::Card;
use crate::game::Areas;
use crate::obs::canon_key;
use crate::playing_card::PlayingCard;
use crate::pools;
use crate::pools::VoucherRoll;
use crate::rng::{Key, PseudoRandom, shuffle};
use crate::run::{History, Run};
use crate::shop::Shop;

/// See [`Game::resample_hidden`](crate::Game::resample_hidden).
pub(crate) fn resample(
	areas: &mut Areas, run: &Run, history: &History, shop: &mut Shop, rng: &mut PseudoRandom,
	seed: &str,
) {
	rng.reseed(seed);
	let order = |c: &PlayingCard| (canon_key(c), c.core.id);
	redeal_unseen(areas, order, |pool| {
		shuffle(pool, rng.seed_for(&Key::from("resample_hand")));
	});
	areas.deck.sort_by_key(order);
	let s = rng.seed_for(&Key::from("resample"));
	shuffle(&mut areas.deck, s);
	permute_face_down_jokers(&mut areas.jokers, rng);
	if !shop.voucher_seen && shop.voucher.is_some() && !shop.open {
		shop.voucher = Some(pools::get_next_voucher(
			areas,
			run,
			history,
			rng,
			VoucherRoll::Ante,
		));
	}
}

/// Shuffles the face-down jokers among their slots (in id order first).
fn permute_face_down_jokers(jokers: &mut [Card], rng: &mut PseudoRandom) {
	let n_down = jokers.iter().filter(|c| c.core.face_down).count();
	if n_down < 2 {
		return;
	}
	let mut down: Vec<Card> = jokers
		.iter()
		.filter(|c| c.core.face_down)
		.cloned()
		.collect();
	down.sort_by_key(|c| c.core.id);
	shuffle(&mut down, rng.seed_for(&Key::from("resample_jokers")));
	let slots = jokers.iter_mut().filter(|c| c.core.face_down);
	for (slot, c) in slots.zip(down) {
		*slot = c;
	}
}
