//! The real game's pack softlock (reproduced in the game's Lua): hand size 5
//! (Stuntman, Merry Andy), a Mega Spectral pack, Immolate destroys the whole
//! hand, nothing redraws and skipping needs hand cards.

use sim::actions::ActionError;
use sim::data::{Back, Edition, Joker, Spectral};
use sim::inject::Inject;
use sim::{Action, Game, Kind, Phase};

#[test]
fn immolate_emptying_the_pack_hand_is_a_terminal_softlock() {
	let mut g = Game::new_run("H28D9", Back::Red, 1);
	g.inject(&Inject::Joker {
		joker: Joker::Stuntman,
		edition: Edition::None,
		eternal: false,
	});
	g.inject(&Inject::Joker {
		joker: Joker::MerryAndy,
		edition: Edition::None,
		eternal: false,
	});
	for code in ["select", "play 12", "cash_out", "open 1"] {
		g.step(code.parse().unwrap()).unwrap();
	}
	assert_eq!((g.hand.len(), g.deck.len()), (5, 47));
	assert_eq!(g.pack_cards[0].kind, Kind::Spectral(Spectral::Immolate));
	g.step("pick 0 0".parse().unwrap()).unwrap();
	assert!(g.phase == Phase::Pack && g.hand.is_empty() && g.pack_choices == 1);
	assert_eq!(g.check(&Action::SkipPack), Err(ActionError::NotAllowed));
	let mut acts = Vec::new();
	g.legal_actions(&mut acts);
	assert_eq!(acts, [Action::SellJoker(0), Action::SellJoker(1)]);
	assert!(!g.is_terminal());
	for _ in 0..2 {
		g.step(Action::SellJoker(0)).unwrap();
	}
	g.legal_actions(&mut acts);
	assert!(acts.is_empty() && g.is_stuck() && g.is_terminal());
}
