//! Cerulean Bell's forced card and its copies. A copy has the forced flag,
//! but the game only keeps a card with the flag in every selection once the
//! card is highlighted (checked against the game by the oracle's
//! `cerulean_bell_holds_only_the_highlighted_card` scenario).

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
