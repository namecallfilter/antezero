//! The random test policy that plays the differential games, the golden
//! trajectories and the bridge's legal-action test.

use sim::actions::{self, Action};
use sim::data::{
	ALL_JOKERS, ALL_PLANETS, ALL_SPECTRALS, ALL_TAROTS, ALL_VOUCHERS, Edition, Enhancement,
	P_CARDS_SORTED, Seal, Voucher,
};
use sim::game::{Game, Hand, HandSort, Phase, Scratch};
use sim::inject::{Consumable, Inject};
use sim::rng::{self, Xorshift};
use sim::{CardMask, num};

use crate::command::Command;

/// Plays a game with seeded random choices, and now and then a random
/// scenario injection.
#[derive(Debug)]
pub struct Policy {
	rng: Xorshift,
	/// Probability of an injection before each action.
	inject_rate: f64,
	rerolls_this_shop: u32,
	acts: Vec<Action>,
	scratch: Scratch,
}

impl Policy {
	#[must_use]
	pub fn new(seed: u64, inject_rate: f64) -> Policy {
		Policy {
			rng: Xorshift::new(seed),
			inject_rate,
			rerolls_this_shop: 0,
			acts: Vec::new(),
			scratch: Scratch::default(),
		}
	}

	/// The policy of game `n` of a numbered set of games.
	#[must_use]
	pub fn for_game(n: u64, inject_rate: f64) -> Policy {
		Policy::new(n.wrapping_mul(rng::SPLITMIX_GAMMA), inject_rate)
	}

	/// The next command in `g`, or `None` once the game is over (or if it
	/// finds nothing to do: no legal action, or a round without a play).
	pub fn next(&mut self, g: &Game) -> Option<Command> {
		if g.flow.phase == Phase::GameOver {
			return None;
		}
		if g.flow.phase != Phase::Shop {
			self.rerolls_this_shop = 0;
		}
		if self.inject_rate > 0.0
			&& self.rng.chance(self.inject_rate)
			&& let Some(inj) = random_inject(g, &mut self.rng)
		{
			return Some(Command::Inject(inj));
		}
		self.choose(g).map(Command::Act)
	}

	fn choose(&mut self, g: &Game) -> Option<Action> {
		let Policy {
			rng,
			acts,
			scratch,
			rerolls_this_shop,
			..
		} = self;
		g.legal_actions(acts);
		if acts.is_empty() {
			return None;
		}
		let is_use = |a: &Action| matches!(a, Action::UseConsumable { .. });
		let is_sell = |a: &Action| matches!(a, Action::SellJoker(_) | Action::SellConsumable(_));
		// Occasional joker reordering (not enumerated by legal_actions).
		if g.areas.jokers.len() >= 2 && rng.chance(0.03) {
			let (from, to) = random_move(rng, g.areas.jokers.len());
			return Some(Action::MoveJoker { from, to });
		}
		if acts.iter().any(is_use) && rng.chance(0.15) {
			return pick(rng, acts, is_use);
		}
		if acts.iter().any(is_sell) && rng.chance(0.03) {
			return pick(rng, acts, is_sell);
		}
		match g.flow.phase {
			Phase::BlindSelect => {
				if rng.chance(0.08)
					&& let Some(a) = pick(rng, acts, |a| matches!(a, Action::RerollBoss))
				{
					return Some(a);
				}
				if rng.chance(0.2)
					&& let Some(a) = pick(rng, acts, |a| matches!(a, Action::SkipBlind))
				{
					return Some(a);
				}
				Some(Action::SelectBlind)
			}
			Phase::Round => choose_in_round(g, rng, acts, scratch),
			Phase::Shop => choose_in_shop(rng, acts, rerolls_this_shop),
			Phase::Pack => {
				let is_pick = |a: &Action| matches!(a, Action::PickPackCard { .. });
				if acts.iter().any(is_pick) && rng.chance(0.85) {
					return pick(rng, acts, is_pick);
				}
				if acts.contains(&Action::SkipPack) {
					return Some(Action::SkipPack);
				}
				pick(rng, acts, |_| true)
			}
			Phase::RoundEval => Some(Action::CashOut),
			Phase::GameOver => None,
		}
	}
}

/// The play of `plays` that scores the most in `g` (the first of equal
/// ones) and its score, each tried on a copy of `g` in `scratch`.
///
/// # Errors
///
/// The first of `plays` the simulator rejects. They come from its legal
/// actions, so a rejected one is a simulator bug.
pub fn highest_scoring_play(
	scratch: &mut Scratch, g: &Game, plays: impl IntoIterator<Item = CardMask>,
) -> Result<Option<(CardMask, f64)>, CardMask> {
	let mut best: Option<(CardMask, f64)> = None;
	for m in plays {
		let c = scratch.copy(g, &g.rng);
		c.step(Action::Play(m)).map_err(|_| m)?;
		let score = c.run.last_hand_score;
		if best.is_none_or(|(_, b)| score > b) {
			best = Some((m, score));
		}
	}
	Ok(best)
}

/// The cards of a hand of `hand_len` that `play` leaves out, as many as can
/// be selected, from the right.
#[must_use]
pub fn unused_cards(play: CardMask, hand_len: usize) -> CardMask {
	let mut d = CardMask::EMPTY;
	for i in (0..hand_len).rev() {
		if !play.contains(i) && d.len() < Hand::HIGHLIGHT_LIMIT {
			d.insert(i);
		}
	}
	d
}

/// A random one of the actions in `acts` that `keep` accepts, if any.
fn pick(rng: &mut Xorshift, acts: &[Action], keep: impl Fn(&Action) -> bool) -> Option<Action> {
	let n = acts.iter().filter(|a| keep(a)).count();
	if n == 0 {
		return None;
	}
	acts.iter().copied().filter(|a| keep(a)).nth(rng.below(n))
}

/// A move of one of `n` cards to another position.
fn random_move(rng: &mut Xorshift, n: usize) -> (u8, u8) {
	let from = num::byte(rng.below(n));
	let mut to = num::byte(rng.below(n));
	if to == from {
		to = (to + 1) % num::byte(n);
	}
	(from, to)
}

fn choose_in_round(
	g: &Game, rng: &mut Xorshift, acts: &[Action], scratch: &mut Scratch,
) -> Option<Action> {
	let hand = &g.areas.hand.cards;
	if hand.len() >= 2 && rng.chance(0.03) {
		let (from, to) = random_move(rng, hand.len());
		return Some(Action::MoveHand { from, to });
	}
	if rng.chance(0.02) {
		return Some(Action::SortHand(if rng.chance(0.5) {
			HandSort::Suit
		} else {
			HandSort::Rank
		}));
	}
	let is_discard = |a: &Action| matches!(a, Action::Discard(_));
	let can_discard = acts.iter().any(is_discard);
	if can_discard && rng.chance(0.08) {
		return pick(rng, acts, is_discard);
	}
	if rng.chance(0.85) {
		// Look ahead on copies: play the highest-scoring hand, or discard
		// what it doesn't use when it can't win the round.
		let plays = acts.iter().filter_map(|a| match *a {
			Action::Play(m) => Some(m),
			_ => None,
		});
		// A rejected play goes to the real game, where the difftest reports
		// the simulator's rejection with the game's history.
		let best = match highest_scoring_play(scratch, g, plays) {
			Ok(best) => best,
			Err(rejected) => return Some(Action::Play(rejected)),
		};
		let needed = g.blind.chips - g.round.chips;
		if can_discard && best.is_none_or(|(_, score)| score < needed) && rng.chance(0.75) {
			if let Some((m, _)) = best {
				let forced = actions::forced_mask(hand)
					.expect("the simulator listed a play, so a mask holds every held card");
				let d = unused_cards(m, hand.len()) | forced;
				if acts.contains(&Action::Discard(d)) {
					return Some(Action::Discard(d));
				}
			}
			return pick(rng, acts, is_discard);
		}
		return best.map(|(m, _)| Action::Play(m));
	}
	pick(rng, acts, |a| matches!(a, Action::Play(_)))
}

fn choose_in_shop(
	rng: &mut Xorshift, acts: &[Action], rerolls_this_shop: &mut u32,
) -> Option<Action> {
	let is_buy = |a: &Action| matches!(a, Action::BuyShopCard(_) | Action::BuyAndUse { .. });
	let is_redeem = |a: &Action| matches!(a, Action::RedeemVoucher(_));
	let is_open = |a: &Action| matches!(a, Action::OpenBooster(_));
	if acts.iter().any(is_buy) && rng.chance(0.45) {
		return pick(rng, acts, is_buy);
	}
	if acts.iter().any(is_redeem) && rng.chance(0.4) {
		return pick(rng, acts, is_redeem);
	}
	if acts.iter().any(is_open) && rng.chance(0.4) {
		return pick(rng, acts, is_open);
	}
	if *rerolls_this_shop < 3 && rng.chance(0.25) && acts.contains(&Action::Reroll) {
		*rerolls_this_shop += 1;
		return Some(Action::Reroll);
	}
	Some(Action::EndShop)
}

fn random_edition(rng: &mut Xorshift) -> Option<Edition> {
	match rng.below(10) {
		0 => Some(Edition::Foil),
		1 => Some(Edition::Holo),
		2 => Some(Edition::Polychrome),
		3 => Some(Edition::Negative),
		_ => None,
	}
}

/// A random scenario injection that fits `g` (or `None`).
fn random_inject(g: &Game, rng: &mut Xorshift) -> Option<Inject> {
	let r = rng.below(100);
	if r < 45 {
		if g.areas.joker_room() {
			let j = ALL_JOKERS[rng.below(ALL_JOKERS.len())];
			let e = random_edition(rng);
			return Some(Inject::Joker {
				joker: j,
				edition: e,
				eternal: rng.chance(0.1),
			});
		}
		None
	} else if r < 70 {
		if g.areas.consumable_room() {
			let c = match rng.below(3) {
				0 => Consumable::Tarot(ALL_TAROTS[rng.below(ALL_TAROTS.len())]),
				1 => Consumable::Planet(ALL_PLANETS[rng.below(ALL_PLANETS.len())]),
				_ => Consumable::Spectral(ALL_SPECTRALS[rng.below(ALL_SPECTRALS.len())]),
			};
			let e = if rng.chance(0.1) {
				Some(Edition::Negative)
			} else {
				None
			};
			return Some(Inject::Consumable {
				card: c,
				edition: e,
			});
		}
		None
	} else if r < 88 {
		let (s, rk) = P_CARDS_SORTED[rng.below(P_CARDS_SORTED.len())];
		let enh = if rng.chance(0.7) {
			Enhancement::POOL[rng.below(Enhancement::POOL.len())]
		} else {
			Enhancement::Base
		};
		let e = match rng.below(8) {
			0 => Some(Edition::Foil),
			1 => Some(Edition::Holo),
			2 => Some(Edition::Polychrome),
			_ => None,
		};
		let seal = match rng.below(8) {
			0 => Some(Seal::Gold),
			1 => Some(Seal::Red),
			2 => Some(Seal::Blue),
			3 => Some(Seal::Purple),
			_ => None,
		};
		Some(Inject::Card {
			suit: s,
			rank: rk,
			enhancement: enh,
			edition: e,
			seal,
		})
	} else if r < 96 {
		Some(Inject::Dollars(num::float(5 + rng.below(30))))
	} else {
		let v = ALL_VOUCHERS[rng.below(ALL_VOUCHERS.len())];
		if g.history.used_vouchers.contains(v)
			|| matches!(v, Voucher::RerollSurplus | Voucher::RerollGlut)
		{
			return None;
		}
		Some(Inject::Voucher(v))
	}
}
