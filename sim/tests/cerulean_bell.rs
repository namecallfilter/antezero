//! Cerulean Bell's forced card and its copies. A copy has the forced flag,
//! but the game only keeps a card with the flag in every selection once the
//! card is highlighted. The oracle's scenarios check the same rules against
//! the game (`sim/oracle/tests/scenarios.rs`).

use sim::actions::ActionError;
use sim::data::{Back, Edition, Spectral, Tarot};
use sim::inject::{Consumable, Inject};
use sim::{Action, CardMask, Forced, Game, Stake};

const CRYPTID: Consumable = Consumable::Spectral(Spectral::Cryptid);

/// The hand position of the card the Bell holds in [`bell_round`].
const BELL: usize = 2;

/// A round in which the Bell holds the hand card [`BELL`], with
/// `consumables` in this order.
fn bell_round(consumables: &[Consumable]) -> Game {
	let mut g = Game::new_run("BELLCOPY", Back::Red, Stake::White);
	for &card in consumables {
		g.inject(Inject::Consumable {
			card,
			edition: Some(Edition::Negative),
		});
	}
	g.step(Action::SelectBlind)
		.expect("the first blind can be played");
	g.areas.hand.cards[BELL].forced = Forced::Locked;
	g
}

fn mask(cards: &[usize]) -> CardMask {
	cards.iter().copied().collect()
}

/// The hand positions of the cards the Bell holds in a given way.
fn held(g: &Game, how: Forced) -> Vec<usize> {
	let hand = g.areas.hand.cards.iter().enumerate();
	hand.filter(|(_, c)| c.forced == how)
		.map(|(i, _)| i)
		.collect()
}

/// Uses the first consumable on the hand cards `cards`.
fn use_on(g: &mut Game, cards: &[usize]) {
	let action = Action::UseConsumable {
		index: 0,
		targets: mask(cards),
	};
	g.step(action).expect("the consumable takes these cards");
}

#[test]
fn a_copy_of_the_forced_card_is_not_part_of_every_selection() {
	let mut g = bell_round(&[CRYPTID, CRYPTID]);
	use_on(&mut g, &[BELL]);
	use_on(&mut g, &[BELL]);
	assert_eq!(held(&g, Forced::Locked), [BELL]);
	let copies = held(&g, Forced::Flagged);
	assert_eq!(copies.len(), 4, "the copies have the flag");

	assert_eq!(g.check(Action::Play(mask(&[BELL]))), Ok(()));
	assert_eq!(g.check(Action::Discard(mask(&[BELL, 0]))), Ok(()));
	assert_eq!(
		g.check(Action::Play(mask(&copies))),
		Err(ActionError::BadSelection),
		"the copies without the forced card"
	);
}

#[test]
fn a_copy_is_part_of_every_selection_once_a_consumable_is_used_on_it() {
	let mut g = bell_round(&[CRYPTID, Consumable::Tarot(Tarot::Magician)]);
	use_on(&mut g, &[BELL]);
	let copy = held(&g, Forced::Flagged)[0];
	use_on(&mut g, &[BELL, copy]);
	assert_eq!(held(&g, Forced::Locked), [BELL, copy]);
	assert_eq!(
		g.check(Action::Play(mask(&[BELL]))),
		Err(ActionError::BadSelection)
	);
	assert_eq!(g.check(Action::Play(mask(&[BELL, copy]))), Ok(()));
}

#[test]
fn the_first_copy_takes_over_when_the_forced_card_is_destroyed() {
	let mut g = bell_round(&[CRYPTID, Consumable::Tarot(Tarot::HangedMan)]);
	use_on(&mut g, &[BELL]);
	let copies = held(&g, Forced::Flagged);
	use_on(&mut g, &[BELL]);
	assert_eq!(held(&g, Forced::Locked), [copies[0] - 1]);
	assert_eq!(held(&g, Forced::Flagged), [copies[1] - 1]);
}

#[test]
fn death_keeps_the_hold_of_the_card_it_changes_and_passes_on_the_other_cards() {
	let death = Consumable::Tarot(Tarot::Death);
	let mut g = bell_round(&[death, death]);
	use_on(&mut g, &[BELL, BELL + 1]);
	assert_eq!(held(&g, Forced::Locked), [BELL], "the changed card is held");
	use_on(&mut g, &[BELL - 1, BELL]);
	assert_eq!(held(&g, Forced::Locked), [BELL - 1, BELL]);
}

/// A round in which two hand cards have the flag and none is held, as in a
/// dump of a hand a player left with a free card highlighted while the held
/// card was destroyed; with `consumable`.
fn round_with_free_copies(consumable: Consumable) -> Game {
	let mut g = bell_round(&[consumable]);
	let hand = &mut g.areas.hand.cards;
	(hand[BELL].forced, hand[4].forced, hand[5].forced) =
		(Forced::No, Forced::Flagged, Forced::Flagged);
	g
}

#[test]
fn no_copy_takes_over_while_the_game_keeps_a_card_highlighted() {
	let aura = Consumable::Spectral(Spectral::Aura);
	let mut sold = round_with_free_copies(aura);
	sold.step(Action::SellConsumable(0))
		.expect("a consumable can be sold");
	assert_eq!(held(&sold, Forced::Locked), [0; 0]);

	let mut used = round_with_free_copies(aura);
	use_on(&mut used, &[0]);
	assert_eq!(
		held(&used, Forced::Locked),
		[0; 0],
		"Aura's card stays highlighted"
	);
}

#[test]
fn the_first_copy_takes_over_when_the_highlights_run_out() {
	let cases: [(Consumable, &[usize], usize); 4] = [
		(Consumable::Tarot(Tarot::Hermit), &[], 4),
		(Consumable::Tarot(Tarot::Magician), &[0], 4),
		(Consumable::Spectral(Spectral::Talisman), &[0], 4),
		(Consumable::Tarot(Tarot::HangedMan), &[0], 3),
	];
	for (consumable, cards, first_copy) in cases {
		let mut g = round_with_free_copies(consumable);
		use_on(&mut g, cards);
		assert_eq!(held(&g, Forced::Locked), [first_copy], "{consumable:?}");
	}
}
