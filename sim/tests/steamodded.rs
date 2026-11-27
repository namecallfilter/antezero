//! What the simulator's Steamodded mode plays as the real game (which runs
//! Steamodded) does, where it differs from vanilla.

use sim::data::{Back, Center, Edition, Joker, Seal, Spectral, Voucher};
use sim::inject::{Consumable, Inject};
use sim::json::Json;
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
		let rolled = g.to_dump().get("rng").get("rarity1sou") != &Json::Null;
		assert_eq!(rolled, !steamodded, "steamodded: {steamodded}");
	}
}

/// The run the real game played under Steamodded 26.1002 (seed `MARBCERT`,
/// Red Deck, White Stake, these five jokers in this order), up to its first
/// hand.
fn first_hand_with_marble_joker_and_certificate(steamodded: bool) -> Game {
	let mut g = Game::new_run("MARBCERT", Back::Red, Stake::White);
	g.setup.steamodded = steamodded;
	for joker in [Joker::Cavendish, Joker::GrosMichel, Joker::Stuntman] {
		g.inject(Inject::Joker {
			joker,
			edition: Some(Edition::Negative),
			eternal: false,
		});
	}
	g.inject(Inject::joker(Joker::Marble));
	g.inject(Inject::joker(Joker::Certificate));
	act(&mut g, "select");
	g
}

/// Marble Joker's Stone card is in the deck when the blind shuffles it, under
/// Steamodded as in vanilla: in the real game it was the 24th of the 47 cards
/// left after the deal. (Steamodded builds before its "make marble joker
/// timing work like vanilla" change of May 2026 shuffled first and put the
/// card under the deck.)
#[test]
fn marble_jokers_stone_card_is_shuffled_into_the_deck() {
	for steamodded in [false, true] {
		let g = first_hand_with_marble_joker_and_certificate(steamodded);
		let stones: Vec<usize> = (0..g.areas.deck.len())
			.filter(|&at| g.areas.deck[at].is_stone())
			.collect();
		assert_eq!(g.areas.deck.len(), 47, "steamodded: {steamodded}");
		assert_eq!(stones, [23], "steamodded: {steamodded}");
	}
}

/// Certificate's seal takes one roll of `certsl` and none of the Standard
/// Pack's `stdseal<ante>`, under Steamodded as in vanilla: its
/// `SMODS.poll_seal` call is `guaranteed`, which skips that roll. In the real
/// game the card was a 10 of Diamonds with a Purple Seal.
#[test]
fn certificate_rolls_no_standard_pack_seal() {
	for steamodded in [false, true] {
		let mut g = first_hand_with_marble_joker_and_certificate(steamodded);
		let sealed: Vec<(String, Option<Seal>)> = g
			.areas
			.hand
			.cards
			.iter()
			.filter(|c| c.seal.is_some())
			.map(|c| (format!("{:?} of {:?}", c.rank, c.suit), c.seal))
			.collect();
		assert_eq!(
			sealed,
			[("Ten of Diamonds".to_string(), Some(Seal::Purple))],
			"steamodded: {steamodded}"
		);
		let rng = g.to_dump();
		let rng = rng.get("rng");
		assert_ne!(rng.get("certsl"), &Json::Null, "steamodded: {steamodded}");
		assert_eq!(rng.get("stdseal1"), &Json::Null, "steamodded: {steamodded}");
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
			let Json::Obj(fields) = d else {
				panic!("a dump is an object");
			};
			fields.remove("used_jokers");
		}
		assert_eq!(x, y, "step {step}: {a:?}");
	}
	assert!(shops > 10, "{shops} steps in a shop");
}
