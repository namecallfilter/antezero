//! Cerulean Bell's forced card and its copies. A copy has the forced flag,
//! but the game only keeps a card with the flag in every selection once the
//! card is highlighted. The oracle's scenarios check the same rules against
//! the game (`sim/oracle/tests/scenarios.rs`).

use sim::actions::ActionError;
use sim::data::{Back, Edition, Spectral, Tarot};
use sim::inject::{Consumable, Inject};
use sim::obs::{self, Layout, OpKind, OpOptions};
use sim::{Action, CardMask, Forced, Game, Stake, num};

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
	g.areas.hand.cards[BELL].forced = Forced::Locked { place: 0 };
	g
}

fn mask(cards: &[usize]) -> CardMask {
	cards.iter().copied().collect()
}

/// The hand positions of the cards the Bell holds in a given way.
fn held(g: &Game, how: impl Fn(Forced) -> bool) -> Vec<usize> {
	let hand = g.areas.hand.cards.iter().enumerate();
	hand.filter(|(_, c)| how(c.forced))
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
	assert_eq!(held(&g, Forced::is_locked), [BELL]);
	let copies = held(&g, |f| f == Forced::Flagged);
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
	let copy = held(&g, |f| f == Forced::Flagged)[0];
	use_on(&mut g, &[BELL, copy]);
	assert_eq!(held(&g, Forced::is_locked), [BELL, copy]);
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
	let copies = held(&g, |f| f == Forced::Flagged);
	use_on(&mut g, &[BELL]);
	assert_eq!(held(&g, Forced::is_locked), [copies[0] - 1]);
	assert_eq!(held(&g, |f| f == Forced::Flagged), [copies[1] - 1]);
}

#[test]
fn death_keeps_the_hold_of_the_card_it_changes_and_passes_on_the_other_cards() {
	let death = Consumable::Tarot(Tarot::Death);
	let mut g = bell_round(&[death, death]);
	use_on(&mut g, &[BELL, BELL + 1]);
	assert_eq!(
		held(&g, Forced::is_locked),
		[BELL],
		"the changed card is held"
	);
	use_on(&mut g, &[BELL - 1, BELL]);
	assert_eq!(held(&g, Forced::is_locked), [BELL - 1, BELL]);
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
	assert_eq!(held(&sold, Forced::is_locked), [0; 0]);

	let mut used = round_with_free_copies(aura);
	use_on(&mut used, &[0]);
	assert_eq!(
		held(&used, Forced::is_locked),
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
		assert_eq!(held(&g, Forced::is_locked), [first_copy], "{consumable:?}");
	}
}

fn move_hand_card(g: &mut Game, from: usize, to: usize) {
	let [from, to] = [from, to].map(num::byte);
	g.step(Action::MoveHand { from, to })
		.expect("a hand card can be moved");
}

/// A round in which the Bell holds a hand card past the 32 positions a
/// selection can name, with a Cryptid and The Hermit left.
fn round_with_the_held_card_out_of_reach() -> Game {
	let mut cards = [CRYPTID; 15];
	cards[14] = Consumable::Tarot(Tarot::Hermit);
	let mut g = bell_round(&cards);
	for _ in 0..13 {
		use_on(&mut g, &[BELL]);
	}
	let last = g.areas.hand.cards.len() - 1;
	assert!(
		last >= CardMask::CAPACITY,
		"the hand has {} cards",
		last + 1
	);
	move_hand_card(&mut g, BELL, last);
	assert_eq!(held(&g, Forced::is_locked), [last]);
	g
}

#[test]
fn no_selection_is_legal_while_the_held_card_is_out_of_its_reach() {
	let mut g = round_with_the_held_card_out_of_reach();
	let first = mask(&[0]);
	let cryptid = |targets| Action::UseConsumable { index: 0, targets };
	assert_eq!(g.check(Action::Play(first)), Err(ActionError::BadSelection));
	assert_eq!(
		g.check(Action::Discard(first)),
		Err(ActionError::BadSelection)
	);
	assert_eq!(g.check(cryptid(first)), Err(ActionError::NotAllowed));
	let mut legal = Vec::new();
	g.legal_actions(&mut legal);
	let hermit = Action::UseConsumable {
		index: 1,
		targets: CardMask::EMPTY,
	};
	let sells = [Action::SellConsumable(0), Action::SellConsumable(1)];
	assert_eq!(legal, [hermit, sells[0], sells[1]], "what takes no cards");

	// In reach again, the card is part of every selection as before.
	let last = g.areas.hand.cards.len() - 1;
	move_hand_card(&mut g, last, CardMask::CAPACITY - 1);
	let held_card = mask(&[CardMask::CAPACITY - 1]);
	assert_eq!(g.check(Action::Play(first)), Err(ActionError::BadSelection));
	assert_eq!(g.check(Action::Play(first | held_card)), Ok(()));
	assert_eq!(g.check(cryptid(held_card)), Ok(()));
}

/// The ops of `g` with the default options.
fn ops(g: &Game) -> Vec<obs::Op> {
	let mut ops = Vec::new();
	let layout = Layout::of(&g.areas, &g.tags);
	obs::ops(g, &layout, OpOptions::default(), &mut ops);
	ops
}

#[test]
fn an_op_brings_a_held_card_out_of_reach_back_whatever_the_reorder_allowance() {
	let mut g = round_with_the_held_card_out_of_reach();
	let last = g.areas.hand.cards.len() - 1;
	while g.flow.rearranges < obs::MAX_REARRANGES {
		move_hand_card(&mut g, 0, 1);
	}
	let offered = ops(&g);
	let moves: Vec<&obs::Op> = offered
		.iter()
		.filter(|op| op.kind == OpKind::MoveHand)
		.collect();
	let [back] = moves[..] else {
		panic!("one move is offered, of the held card: {offered:?}");
	};
	assert_eq!((back.cards, usize::from(back.index)), (None, last));
	let selects = |op: &obs::Op| matches!(op.kind, OpKind::Play | OpKind::Discard);
	assert!(!offered.iter().any(selects), "no selection has the card");

	let action = obs::decode(&g, back, CardMask::EMPTY, Some(5));
	assert_eq!(
		action,
		Some(Action::MoveHand {
			from: num::byte(last),
			to: 5
		})
	);
	g.step(action.expect("the move is legal"))
		.expect("the move is legal");
	let offered = ops(&g);
	let play = offered.iter().find(|op| op.kind == OpKind::Play);
	let required = play.and_then(|op| op.cards).map(|cards| cards.required);
	assert_eq!(required, Some(mask(&[5])), "plays hold the card again");
}

/// The hand positions of the held cards with their places in the highlight
/// list.
fn places(g: &Game) -> Vec<(usize, u8)> {
	let hand = g.areas.hand.cards.iter().enumerate();
	hand.filter_map(|(i, c)| match c.forced {
		Forced::Locked { place } => Some((i, place)),
		Forced::No | Forced::Flagged => None,
	})
	.collect()
}

#[test]
fn the_held_cards_close_up_when_a_free_card_between_them_is_unhighlighted() {
	let mut g = bell_round(&[CRYPTID, Consumable::Tarot(Tarot::Sun)]);
	use_on(&mut g, &[BELL]);
	let copy = held(&g, |f| f == Forced::Flagged)[0];
	// The Bell's card is in the list, then the free card and the copy.
	use_on(&mut g, &[0, BELL, copy]);
	assert_eq!(places(&g), [(BELL, 0), (copy, 1)]);
}

#[test]
fn a_held_card_moves_up_when_the_one_before_it_is_destroyed() {
	// A run in which Familiar destroys the first of the two held cards.
	let (g, copy) = (0..200)
		.find_map(|n| {
			let mut g = Game::new_run(&format!("BELLGAP{n}"), Back::Red, Stake::White);
			for card in [
				CRYPTID,
				Consumable::Tarot(Tarot::Magician),
				Consumable::Spectral(Spectral::Familiar),
			] {
				g.inject(Inject::Consumable {
					card,
					edition: Some(Edition::Negative),
				});
			}
			g.step(Action::SelectBlind)
				.expect("the first blind can be played");
			g.areas.hand.cards[BELL].forced = Forced::Locked { place: 0 };
			use_on(&mut g, &[BELL]);
			let copy = held(&g, |f| f == Forced::Flagged)[0];
			use_on(&mut g, &[BELL, copy]);
			let ids = [BELL, copy].map(|i| g.areas.hand.cards[i].core.id);
			use_on(&mut g, &[]);
			let left = |id| g.areas.hand.cards.iter().any(|c| c.core.id == id);
			(!left(ids[0]) && left(ids[1])).then_some((g, copy - 1))
		})
		.expect("Familiar destroys the Bell's card in some run");
	assert_eq!(places(&g), [(copy, 0)]);
}

#[test]
fn the_held_cards_close_up_when_the_hand_is_unhighlighted_for_a_use_without_cards() {
	// Held cards of a dump in which free cards are highlighted before them.
	let mut g = bell_round(&[Consumable::Tarot(Tarot::Hermit)]);
	let hand = &mut g.areas.hand.cards;
	hand[BELL].forced = Forced::Locked { place: 1 };
	hand[5].forced = Forced::Locked { place: 3 };
	use_on(&mut g, &[]);
	assert_eq!(places(&g), [(BELL, 0), (5, 1)]);
}
