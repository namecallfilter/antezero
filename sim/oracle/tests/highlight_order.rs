//! The order in which The Hanged Man destroys cards under Cerulean Bell: from
//! the end of the game's highlight list (`G.hand.highlighted`), in which a
//! held card stays where it was highlighted. Read from the simulator's
//! timeline, which this crate turns on. The oracle's scenarios check the
//! same against the game (`scenarios.rs`).

use sim::data::{Back, Edition, Spectral, Tarot};
use sim::inject::{Consumable, Inject};
use sim::{Action, Forced, Game, Stake, num};

const DEATH: Consumable = Consumable::Tarot(Tarot::Death);
const HANGED_MAN: Consumable = Consumable::Tarot(Tarot::HangedMan);

/// The hand position of the card the Bell holds in [`bell_round`].
const BELL: usize = 3;

/// A round in which the Bell holds the hand card [`BELL`], with
/// `consumables` in this order.
fn bell_round(consumables: &[Consumable]) -> Game {
	let mut g = Game::new_run("BELLLIST", Back::Red, Stake::White);
	for &card in consumables {
		g.inject(Inject::Consumable {
			card,
			edition: Some(Edition::Negative),
		});
	}
	g.step(Action::SelectBlind)
		.expect("the first blind can be played");
	g.areas.hand.cards[BELL].forced = Forced::Locked { place: 0 };
	g
}

/// The ids of the hand cards at `cards`.
fn ids<const N: usize>(g: &Game, cards: [usize; N]) -> [u32; N] {
	cards.map(|i| g.areas.hand.cards[i].core.id)
}

/// Uses the first consumable on the hand cards `cards`.
fn use_on(g: &mut Game, cards: &[usize]) {
	let action = Action::UseConsumable {
		index: 0,
		targets: cards.iter().copied().collect(),
	};
	g.step(action).expect("the consumable takes these cards");
}

fn move_hand_card(g: &mut Game, from: usize, to: usize) {
	let [from, to] = [from, to].map(num::byte);
	g.step(Action::MoveHand { from, to })
		.expect("a hand card can be moved");
}

/// Asserts that the last action removed the playing cards `cards` in this
/// order.
fn assert_removed(g: &Game, cards: [u32; 2]) {
	let ids = g.timeline().iter().filter_map(|entry| {
		let (_, id) = entry.split_once(" rm ")?;
		Some(id.parse().expect("a removal names the card's id"))
	});
	let removed: Vec<u32> = ids.filter(|id| cards.contains(id)).collect();
	assert_eq!(removed, cards, "in {:?}", g.timeline());
}

#[test]
fn the_hanged_man_destroys_the_held_card_after_the_card_the_action_adds() {
	for other in [0, BELL + 2] {
		let mut g = bell_round(&[HANGED_MAN]);
		let [other_id, bell] = ids(&g, [other, BELL]);
		use_on(&mut g, &[other, BELL]);
		assert_removed(&g, [other_id, bell]);
	}
}

#[test]
fn the_hanged_man_destroys_two_held_cards_from_the_one_highlighted_last() {
	let mut g = bell_round(&[
		Consumable::Spectral(Spectral::Cryptid),
		Consumable::Tarot(Tarot::Magician),
		HANGED_MAN,
	]);
	use_on(&mut g, &[BELL]);
	let copy = g.areas.hand.cards.len() - 1;
	let [bell_id, copy_id] = ids(&g, [BELL, copy]);
	use_on(&mut g, &[BELL, copy]);
	move_hand_card(&mut g, copy, 0);
	use_on(&mut g, &[0, BELL + 1]);
	assert_removed(&g, [copy_id, bell_id]);
}

#[test]
fn a_card_death_gives_the_hold_to_is_highlighted_after_the_held_card() {
	let mut g = bell_round(&[DEATH, HANGED_MAN]);
	let [left, bell] = ids(&g, [0, BELL]);
	use_on(&mut g, &[0, BELL]);
	use_on(&mut g, &[0, BELL]);
	assert_removed(&g, [left, bell]);
}

/// No card is held and a copy has the flag, as in a dump of a hand a player
/// left with a free card highlighted while the held card was destroyed.
#[test]
fn death_on_a_free_card_and_a_copy_highlights_them_by_hand_position() {
	let mut g = bell_round(&[DEATH, HANGED_MAN]);
	let hand = &mut g.areas.hand.cards;
	(hand[BELL].forced, hand[5].forced) = (Forced::No, Forced::Flagged);
	let [free, copy] = ids(&g, [4, 5]);
	use_on(&mut g, &[4, 5]);
	move_hand_card(&mut g, 4, 5);
	use_on(&mut g, &[4, 5]);
	assert_removed(&g, [copy, free]);
}
