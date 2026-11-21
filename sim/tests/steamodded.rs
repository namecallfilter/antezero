//! What the simulator's Steamodded mode plays as the real game (which runs
//! Steamodded) does, where it differs from vanilla.

use sim::data::{Back, Joker, Spectral};
use sim::inject::{Consumable, Inject};
use sim::json::Json;
use sim::{Action, Game, Stake};

fn act(g: &mut Game, code: &str) {
	let a: Action = code.parse().expect("an action code");
	g.step(a).expect("a legal action");
}

/// The Pillar's mark: Steamodded sets it once the hand is evaluated, so a
/// face card Midas Mask turns Gold while scoring keeps it; vanilla marks the
/// card when it is played, and the new enhancement clears the mark.
#[test]
fn a_card_changed_while_scoring_stays_marked_played() {
	for steamodded in [false, true] {
		let mut g = Game::new_run("PILLAR", Back::Red, Stake::White);
		g.setup.steamodded = steamodded;
		g.inject(Inject::joker(Joker::MidasMask));
		act(&mut g, "select");
		let at = g
			.areas
			.hand
			.cards
			.iter()
			.position(|c| c.is_face(false))
			.expect("a face card in the first hand");
		let id = g.areas.hand.cards[at].core.id;
		act(&mut g, &format!("play {}", 1u32 << at));
		let card = g
			.areas
			.discard
			.iter()
			.find(|c| c.core.id == id)
			.expect("the played card, discarded");
		assert_eq!(
			card.played_this_ante, steamodded,
			"steamodded: {steamodded}"
		);
	}
}

/// The Soul's legendary joker: vanilla rolls the rarity stream and ignores
/// the roll; Steamodded does not roll it.
#[test]
fn a_legendary_joker_takes_no_rarity_roll() {
	for steamodded in [false, true] {
		let mut g = Game::new_run("SOUL", Back::Red, Stake::White);
		g.setup.steamodded = steamodded;
		g.inject(Inject::Consumable {
			card: Consumable::Spectral(Spectral::Soul),
			edition: None,
		});
		act(&mut g, "use 0 0");
		assert_eq!(g.areas.jokers.len(), 1, "The Soul made a joker");
		let rolled = g.to_dump().get("rng").get("rarity1sou") != &Json::Null;
		assert_eq!(rolled, !steamodded, "steamodded: {steamodded}");
	}
}
