//! What the simulator's Steamodded mode plays as the real game (which runs
//! Steamodded) does, where it differs from vanilla.

use serde_json::Value;
use sim::data::{Back, Center, Joker, Spectral, Voucher};
use sim::inject::{Consumable, Inject};
use sim::{Action, Game, Phase, Stake};

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

/// DNA copies the first hand's one card while it scores: under Steamodded
/// the card is not marked played yet, so neither is its copy.
#[test]
fn a_card_copied_while_scoring_has_the_mark_its_original_had() {
	for steamodded in [false, true] {
		let mut g = Game::new_run("DNA", Back::Red, Stake::White);
		g.setup.steamodded = steamodded;
		g.inject(Inject::joker(Joker::Dna));
		act(&mut g, "select");
		let dealt = g.areas.deck.iter().chain(&g.areas.hand.cards);
		let last = dealt.map(|c| c.core.id).max();
		act(&mut g, "play 1");
		let copy = g
			.areas
			.hand
			.cards
			.iter()
			.find(|c| Some(c.core.id) > last)
			.expect("DNA's copy, drawn to the hand");
		assert_eq!(
			copy.played_this_ante, !steamodded,
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
		let rolled = !g.to_dump()["rng"]["rarity1sou"].is_null();
		assert_eq!(rolled, !steamodded, "steamodded: {steamodded}");
	}
}

/// Steamodded's display card of a redeemed voucher leaves the voucher in
/// `used_jokers`, which the simulator does not track: no pool it draws from
/// changes for it (a voucher's pool skips the redeemed ones anyway).
#[test]
fn a_redeemed_voucher_in_used_jokers_changes_no_draw() {
	let mut plain = Game::new_run("VOUCHER", Back::Red, Stake::White);
	plain.inject(Inject::Voucher(Voucher::OverstockNorm));
	plain.inject(Inject::Dollars(500.0));
	let mut marked = plain.clone();
	marked
		.history
		.used_jokers
		.insert(Center::Voucher(Voucher::OverstockNorm));
	let (mut acts, mut shops) = (Vec::new(), 0);
	for step in 0..400 {
		shops += usize::from(plain.flow.phase == Phase::Shop);
		plain.legal_actions(&mut acts);
		// the play that scores most, the shop's first few actions, and on
		let Some(a) = (if plain.flow.phase == Phase::Round {
			acts.iter()
				.copied()
				.filter(|a| matches!(a, Action::Play(_)))
				.max_by(|&x, &y| {
					let score = |a| {
						let mut g = plain.clone();
						g.step(a).expect("a legal action");
						g.round.chips
					};
					score(x).total_cmp(&score(y))
				})
		} else {
			acts.get((step * 7) % acts.len().max(1)).copied()
		}) else {
			break;
		};
		plain.step(a).expect("a legal action");
		marked.step(a).expect("the same action");
		let (mut x, mut y) = (plain.to_dump(), marked.to_dump());
		for d in [&mut x, &mut y] {
			let Value::Object(fields) = d else {
				panic!("a dump is an object");
			};
			fields.remove("used_jokers");
		}
		assert_eq!(x, y, "step {step}: {a:?}");
	}
	assert!(shops > 10, "{shops} steps in a shop");
}
