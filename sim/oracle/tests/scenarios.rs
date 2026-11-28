//! Targeted differential scenarios against the real game. They need the
//! game's `lua51.dll` and scripts, so they only run when asked for:
//! `cargo test -p oracle --test scenarios -- --include-ignored`. Each
//! scenario drives a specific mechanic in both engines and compares, after
//! every command, the canonical state, the timeline order and the frame of
//! every timeline entry.

use oracle::Oracle;
use oracle::command::{Command, Start};
use oracle::difftest::{Checks, Comparison, compare};
use oracle::policy::{highest_scoring_play, unused_cards};
use sim::actions::Action;
use sim::data::{
	Back, BlindKind, Edition, Enhancement, Joker, Rank, Spectral, Suit, TagKind, Tarot,
};
use sim::game::{Game, Phase, Scratch};
use sim::inject::{Consumable, Inject};
use sim::{CardMask, Forced, Stake, json, num};

struct Both {
	o: Oracle,
	g: Game,
	log: Vec<String>,
	/// The frame offset this scenario is known to have: how many frames
	/// later the game has the first offset entry of a step. `None`: every
	/// timeline entry must be on the same frame.
	known_frame_offset: Option<i64>,
	/// Steps that had the known frame offset.
	frame_offsets: usize,
	scratch: Scratch,
}

impl Both {
	fn start(start: Start<'_>) -> Both {
		assert!(
			oracle::available(),
			"the oracle needs the game's lua51.dll and scripts (see oracle/src/lib.rs)"
		);
		let mut o = Oracle::new().expect("the harness loads (`oracle::available()` holds)");
		let lua = o
			.command(&start.to_lua())
			.expect("the oracle starts the run");
		let mut b = Both {
			o,
			g: start.new_game(),
			log: vec![start.to_string()],
			known_frame_offset: None,
			frame_offsets: 0,
			scratch: Scratch::default(),
		};
		b.check(&lua);
		b
	}

	fn check(&mut self, lua: &str) {
		let j = json::parse(lua).expect("the oracle returns JSON");
		let checks = if self.known_frame_offset.is_some() {
			Checks::Timeline
		} else {
			Checks::TimelineAndFrames
		};
		let recent = self.log[self.log.len().saturating_sub(6)..].join("\n  ");
		match compare(&mut self.g, &j, checks) {
			Comparison::Match => {}
			Comparison::FrameOffset(offset) => {
				assert_eq!(
					offset.frames(),
					self.known_frame_offset,
					"a frame offset other than the known one after:\n  {recent}\n  {offset}"
				);
				self.frame_offsets += 1;
			}
			Comparison::Diverged(mismatch) => {
				panic!("divergence after:\n  {recent}\n{mismatch}")
			}
		}
	}

	fn run(&mut self, command: Command) {
		self.log.push(command.to_string());
		command
			.apply(&mut self.g)
			.unwrap_or_else(|e| panic!("sim rejected {command:?}: {e}"));
		let lua = self
			.o
			.command(&command.to_lua())
			.expect("the oracle runs the command");
		self.check(&lua);
	}

	fn act(&mut self, a: Action) {
		self.run(Command::Act(a));
	}

	fn inject(&mut self, i: Inject) {
		self.run(Command::Inject(i));
	}

	/// Plays the current round with the best-scoring hands (discarding the
	/// rest when the best hand cannot win); returns whether it was won.
	fn win_round(&mut self) -> bool {
		let mut acts = Vec::new();
		while self.g.flow.phase == Phase::Round {
			self.g.legal_actions(&mut acts);
			let plays = acts.iter().filter_map(|a| match *a {
				Action::Play(m) => Some(m),
				_ => None,
			});
			let Some((best, score)) = highest_scoring_play(&mut self.scratch, &self.g, plays)
				.unwrap_or_else(|m| panic!("the simulator rejected its legal play {m:?}"))
			else {
				return false;
			};
			let need = self.g.blind.chips - self.g.round.chips;
			let mut action = Action::Play(best);
			if score < need && self.g.round.discards_left > 0.0 {
				let d = unused_cards(best, self.g.areas.hand.cards.len());
				if !d.is_empty() && self.g.check(Action::Discard(d)).is_ok() {
					action = Action::Discard(d);
				}
			}
			self.act(action);
		}
		self.g.flow.phase == Phase::RoundEval
	}
}

/// First seed (from a fixed list) whose new run satisfies `ok`.
fn find_seed(ok: impl Fn(&Game) -> bool) -> String {
	(0..20_000)
		.map(|i| format!("SC{i:05}"))
		.find(|s| ok(&Game::new_run(s, Back::Red, Stake::White)))
		.expect("some seed among the first 20,000 has the wanted run")
}

fn red_white(seed: &str) -> Start<'_> {
	Start {
		seed,
		back: Back::Red,
		stake: Stake::White,
	}
}

/// A shop joker created by an Uncommon / Rare tag (`Tag:yep` callback with
/// `start_materialize`).
#[test]
#[ignore = "needs the game's lua51.dll; run with --include-ignored"]
fn tag_created_shop_joker() {
	let seed = find_seed(|g| {
		matches!(
			g.blind_select.skip_tags.small,
			Some(TagKind::Uncommon | TagKind::Rare)
		)
	});
	let mut b = Both::start(red_white(&seed));
	for _ in 0..2 {
		b.inject(Inject::joker(Joker::GrosMichel));
	}
	b.act(Action::SkipBlind);
	b.act(Action::SelectBlind);
	assert!(b.win_round(), "big blind not won");
	b.act(Action::CashOut);
	assert_eq!(b.g.flow.phase, Phase::Shop);
	b.act(Action::EndShop);
}

/// An edition tag applied to a tag-created shop joker (`store_joker_modify`
/// on the forced card).
#[test]
#[ignore = "needs the game's lua51.dll; run with --include-ignored"]
fn edition_tag_on_tag_created_joker() {
	use sim::CardLike;

	let seed = find_seed(|g| {
		matches!(
			g.blind_select.skip_tags.small,
			Some(TagKind::Uncommon | TagKind::Rare)
		) && matches!(
			g.blind_select.skip_tags.big,
			Some(TagKind::Foil | TagKind::Holo | TagKind::Polychrome | TagKind::Negative)
		) && matches!(
			g.blind_select.boss,
			BlindKind::Club
				| BlindKind::Goad
				| BlindKind::Head
				| BlindKind::Window
				| BlindKind::Pillar
				| BlindKind::Manacle
		)
	});
	let mut b = Both::start(red_white(&seed));
	for _ in 0..3 {
		b.inject(Inject::joker(Joker::GrosMichel));
	}
	b.act(Action::SkipBlind);
	b.act(Action::SkipBlind);
	b.act(Action::SelectBlind);
	assert!(b.win_round(), "boss blind not won");
	b.act(Action::CashOut);
	assert_eq!(b.g.flow.phase, Phase::Shop);
	assert!(
		b.g.areas
			.shop_jokers
			.iter()
			.any(|c| c.core().edition.is_some()),
		"the edition tag did not apply"
	);
}

/// Two Vampires with Midas Mask between them: a card drained by the first
/// Vampire and gilded by Midas is not drained again (`vampired`).
#[test]
#[ignore = "needs the game's lua51.dll; run with --include-ignored"]
fn vampire_midas_vampire() {
	let mut b = Both::start(red_white("VAMPIRE1"));
	// The post-scoring draws of the play land 5 frames earlier in the sim
	// than in the game (state and event order match).
	b.known_frame_offset = Some(5);
	for j in [Joker::Vampire, Joker::MidasMask, Joker::Vampire] {
		b.inject(Inject::joker(j));
	}
	for s in Suit::ALL {
		for _ in 0..6 {
			b.inject(Inject::Card {
				suit: s,
				rank: Rank::King,
				enhancement: Enhancement::Steel,
				edition: None,
				seal: None,
			});
		}
	}
	b.act(Action::SelectBlind);
	let kings: CardMask =
		b.g.areas
			.hand
			.cards
			.iter()
			.enumerate()
			.filter(|(_, c)| c.rank == Rank::King)
			.map(|(i, _)| i)
			.take(5)
			.collect();
	assert!(!kings.is_empty(), "no king drawn");
	b.act(Action::Play(kings));
	assert_eq!(
		b.frame_offsets, 1,
		"the play's draws were to be the one step 5 frames early; if they now match, drop `known_frame_offset`"
	);
	let gold =
		b.g.areas
			.discard
			.iter()
			.chain(&b.g.areas.hand.cards)
			.chain(&b.g.areas.deck)
			.filter(|c| c.enhancement == Enhancement::Gold)
			.count();
	assert!(gold > 0, "Midas Mask gilded nothing");
}

/// The hand positions of the cards Cerulean Bell holds in a given way.
fn held(g: &Game, how: Forced) -> Vec<usize> {
	let hand = g.areas.hand.cards.iter().enumerate();
	hand.filter(|(_, c)| c.forced == how)
		.map(|(i, _)| i)
		.collect()
}

/// Whether skipping the run's first two blinds only gives tags that leave
/// the game at the next blind.
fn quiet_skips(g: &Game) -> bool {
	let quiet = |tag| {
		matches!(
			tag,
			Some(TagKind::Economy | TagKind::Handy | TagKind::Garbage | TagKind::Skip)
		)
	};
	quiet(g.blind_select.skip_tags.small) && quiet(g.blind_select.skip_tags.big)
}

/// Both engines in a first round against Cerulean Bell, which holds one
/// hand card, with `consumables` in this order and a Joker.
fn bell_round(seed: &str, consumables: &[Consumable]) -> Both {
	let mut b = Both::start(red_white(seed));
	b.g.blind_select.boss = BlindKind::FinalBell;
	b.o.exec("G.GAME.round_resets.blind_choices.Boss = 'bl_final_bell'")
		.expect("the game takes the boss");
	b.inject(Inject::joker(Joker::Joker));
	for &card in consumables {
		b.inject(Inject::Consumable {
			card,
			edition: Some(Edition::Negative),
		});
	}
	b.act(Action::SkipBlind);
	b.act(Action::SkipBlind);
	b.act(Action::SelectBlind);
	assert_eq!(b.g.blind.kind, Some(BlindKind::FinalBell));
	assert_eq!(held(&b.g, Forced::Locked).len(), 1, "the Bell holds a card");
	b
}

/// Using the first consumable on the hand cards `cards`.
fn use_on(cards: &[usize]) -> Action {
	Action::UseConsumable {
		index: 0,
		targets: cards.iter().copied().collect(),
	}
}

/// Cerulean Bell with copies of its forced card: the copies have the flag
/// (`copy_card` copies the ability table), but the game only keeps the
/// highlighted card in the selection. A copy is held once a consumable has
/// been used on it, and the first one is when no held card is left.
#[test]
#[ignore = "needs the game's lua51.dll; run with --include-ignored"]
fn cerulean_bell_holds_only_the_highlighted_card() {
	let mut b = bell_round(&find_seed(quiet_skips), &[
		Consumable::Spectral(Spectral::Cryptid),
		Consumable::Spectral(Spectral::Cryptid),
		Consumable::Tarot(Tarot::Magician),
		Consumable::Tarot(Tarot::HangedMan),
	]);
	let bell = held(&b.g, Forced::Locked);

	// Two Cryptids on the held card: the second needs the copies to be free.
	b.act(use_on(&bell));
	b.act(use_on(&bell));
	assert_eq!(held(&b.g, Forced::Locked), bell);
	let copies = held(&b.g, Forced::Flagged);
	assert_eq!(copies.len(), 4, "the copies have the flag");

	// The Magician on the held card and a copy: the copy is held from now on.
	b.act(use_on(&[bell[0], copies[0]]));
	assert_eq!(held(&b.g, Forced::Locked), [bell[0], copies[0]]);

	// The Hanged Man destroys the two held cards: the first copy left is held.
	b.act(use_on(&[bell[0], copies[0]]));
	let next = held(&b.g, Forced::Locked);
	assert_eq!(next.len(), 1, "one copy is held: {next:?}");
	assert_eq!(held(&b.g, Forced::Flagged).len(), 2);

	// A discard of the held card alone, then the Bell holds a new card.
	b.act(Action::Discard(next.iter().copied().collect()));
	assert_eq!(held(&b.g, Forced::Locked).len(), 1);
	assert_eq!(held(&b.g, Forced::Flagged), [0; 0], "no card has the flag");
	b.win_round();
}

/// Death under Cerulean Bell. The left card becomes the right one but keeps
/// its own forced flag (`Card:set_ability`), and takes the right card's.
#[test]
#[ignore = "needs the game's lua51.dll; run with --include-ignored"]
fn death_keeps_and_passes_on_the_bells_hold() {
	let death = Consumable::Tarot(Tarot::Death);
	let mut b = bell_round(&find_seed(quiet_skips), &[death, death]);
	let bell = held(&b.g, Forced::Locked)[0];
	if bell != 3 {
		b.act(Action::MoveHand {
			from: num::byte(bell),
			to: 3,
		});
	}

	// The held card becomes the free card to its right, and is still held.
	b.act(use_on(&[3, 5]));
	assert_eq!(held(&b.g, Forced::Locked), [3]);
	assert_eq!(held(&b.g, Forced::Flagged), [0; 0]);

	// The free card to its left becomes the held card, and is held too.
	b.act(use_on(&[0, 3]));
	assert_eq!(held(&b.g, Forced::Locked), [0, 3]);
}

/// What is done with the last consumable in
/// [`a_copy_takes_over_when_the_highlights_run_out`].
#[derive(Clone, Copy)]
enum Next {
	SellJoker,
	SellConsumable,
	UseOnAFreeCard,
	UseOnNoCard,
}

/// A hand a player left with a free card highlighted while the held card
/// was destroyed: the copies have the flag and none is held. The game then
/// holds the first copy when its highlights run out, and not before.
#[test]
#[ignore = "needs the game's lua51.dll; run with --include-ignored"]
fn a_copy_takes_over_when_the_highlights_run_out() {
	let cryptid = Consumable::Spectral(Spectral::Cryptid);
	let familiar = Consumable::Spectral(Spectral::Familiar);
	// A run in which Familiar destroys the card the Bell holds.
	let seed = find_seed(|g| {
		if !quiet_skips(g) {
			return false;
		}
		let mut g = g.clone();
		g.blind_select.boss = BlindKind::FinalBell;
		g.inject(Inject::joker(Joker::Joker));
		for card in [cryptid, familiar] {
			g.inject(Inject::Consumable {
				card,
				edition: Some(Edition::Negative),
			});
		}
		for action in [Action::SkipBlind, Action::SkipBlind, Action::SelectBlind] {
			g.step(action).expect("the run reaches the boss");
		}
		let bell = held(&g, Forced::Locked);
		let id = g.areas.hand.cards[bell[0]].core.id;
		for cards in [&bell[..], &[]] {
			g.step(use_on(cards)).expect("Cryptid, then Familiar");
		}
		g.areas.hand.cards.iter().all(|c| c.core.id != id)
	});
	let free = |g: &Game| held(g, Forced::No)[0];
	let aura = Consumable::Spectral(Spectral::Aura);
	let cases = [
		(aura, Next::SellJoker, 0),
		(aura, Next::SellConsumable, 0),
		(aura, Next::UseOnAFreeCard, 0),
		(Consumable::Tarot(Tarot::Hermit), Next::UseOnNoCard, 1),
		(Consumable::Tarot(Tarot::Magician), Next::UseOnAFreeCard, 1),
		(Consumable::Tarot(Tarot::HangedMan), Next::UseOnAFreeCard, 1),
	];
	for (last, next, held_after) in cases {
		let mut b = bell_round(&seed, &[cryptid, familiar, last]);
		let bell = held(&b.g, Forced::Locked);
		b.act(use_on(&bell));
		// The player highlights a free card and uses Familiar.
		let click = format!(
			"G.hand:add_to_highlighted(G.hand.cards[{}])
			G.FUNCS.use_card({{config = {{ref_table = G.consumeables.cards[1]}}}})
			H.settle()",
			free(&b.g) + 1
		);
		b.o.exec(&click).expect("the game takes the clicks");
		let state = b.o.command("H.state()").expect("the game dumps its state");
		let state = json::parse(&state).expect("the oracle returns JSON");
		b.g = Game::from_dump(&state).expect("the simulator reads the dump");
		assert_eq!(held(&b.g, Forced::Locked), [0; 0], "no card is held");
		assert_eq!(held(&b.g, Forced::Flagged).len(), 2);

		let action = match next {
			Next::SellJoker => Action::SellJoker(0),
			Next::SellConsumable => Action::SellConsumable(0),
			Next::UseOnAFreeCard => use_on(&[free(&b.g)]),
			Next::UseOnNoCard => use_on(&[]),
		};
		b.act(action);
		let now = held(&b.g, Forced::Locked).len();
		assert_eq!(now, held_after, "after {last:?} and {action:?}");
	}
}
