//! Scenario construction helpers: inject cards, money and vouchers into a
//! running game (mirroring the game's own `add_joker` /
//! `create_playing_card`). Useful for curricula, unit tests and the oracle
//! differential tests, which apply identical injections to the real game.

use std::fmt;
use std::str::FromStr;

use crate::actions::{Operands, ParseError};
use crate::card::Kind;
use crate::data::{
	ACE, ALL_JOKERS, ALL_PLANETS, ALL_SPECTRALS, ALL_TAROTS, ALL_VOUCHERS, Edition, Enhancement,
	Joker, Rank, Seal, Suit, Voucher,
};
use crate::game::{Area, Game};
use crate::{cards, events, shop};

/// A scenario injection applied identically to both engines.
#[derive(Clone, Copy, Debug)]
pub enum Inject {
	/// `add_joker(key, edition, silent, eternal)` for a joker.
	Joker {
		joker: Joker,
		edition: Edition,
		eternal: bool,
	},
	/// `add_joker(key, edition)` for a tarot, planet or spectral card.
	Consumable {
		kind: Kind,
		edition: Edition,
	},
	/// `create_playing_card({front, center}, G.deck)` + edition and seal.
	Card {
		suit: Suit,
		rank: Rank,
		enhancement: Enhancement,
		edition: Edition,
		seal: Option<Seal>,
	},
	Dollars(f64),
	/// Marks a voucher redeemed and applies it (like a starting deck voucher).
	Voucher(Voucher),
}

/// See [`Game::inject`].
///
/// # Panics
///
/// Panics if a [`Inject::Consumable`] is not a tarot, planet or spectral card.
pub(crate) fn inject(g: &mut Game, i: &Inject) {
	events::begin_action(g);
	match *i {
		Inject::Joker {
			joker,
			edition,
			eternal,
		} => {
			let mut c = cards::make_card(g, Kind::Joker(joker));
			g.events.materialize(1.0);
			cards::add_to_deck(g, &mut c, false);
			if edition != Edition::None {
				cards::set_edition(g, &mut c, edition, Some(Area::Jokers));
				g.events.edition_fx(edition, false);
			}
			if eternal {
				c.set_eternal(true);
			}
			g.jokers.push(c);
		}
		Inject::Consumable { kind, edition } => {
			assert!(
				kind.is_consumable(),
				"Inject::Consumable needs a consumable, got {kind:?}"
			);
			let mut c = cards::make_card(g, kind);
			g.events.materialize(1.0);
			cards::add_to_deck(g, &mut c, false);
			if edition != Edition::None {
				cards::set_edition(g, &mut c, edition, Some(Area::Consumables));
				g.events.edition_fx(edition, false);
			}
			g.consumables.push(c);
		}
		Inject::Card {
			suit,
			rank,
			enhancement,
			edition,
			seal,
		} => {
			let mut c = cards::make_playing_card(g, suit, rank, enhancement);
			c.face_down = true;
			if edition != Edition::None {
				cards::set_edition(g, &mut c, edition, Some(Area::Deck));
			}
			if seal.is_some() {
				c.seal = seal;
				cards::set_cost(g, &mut c, Some(Area::Deck));
			}
			g.deck.insert(0, c);
		}
		Inject::Dollars(amount) => {
			g.dollars += amount;
			// Money moves without an event: nothing to run.
			return;
		}
		Inject::Voucher(voucher) => {
			g.set_voucher_used(voucher);
			shop::apply_voucher(g, voucher, true);
		}
	}
	events::run_queue(g);
}

impl fmt::Display for Inject {
	/// The text code used in histories (enum discriminants as numbers).
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match *self {
			Inject::Joker {
				joker,
				edition,
				eternal,
			} => {
				write!(
					f,
					"inject_joker {} {} {}",
					joker as u8,
					edition as u8,
					u8::from(eternal)
				)
			}
			Inject::Consumable { kind, edition } => {
				let (set, index) = match kind {
					Kind::Tarot(tarot) => (0, tarot as u8),
					Kind::Planet(planet) => (1, planet as u8),
					Kind::Spectral(spectral) => (2, spectral as u8),
					_ => return Err(fmt::Error),
				};
				write!(f, "inject_cons {set} {index} {}", edition as u8)
			}
			Inject::Card {
				suit,
				rank,
				enhancement,
				edition,
				seal,
			} => {
				write!(
					f,
					"inject_card {} {rank} {} {} {}",
					suit as u8,
					enhancement as u8,
					edition as u8,
					seal.map_or(0, |seal| seal as u8 + 1)
				)
			}
			Inject::Dollars(amount) => write!(f, "inject_dollars {amount}"),
			Inject::Voucher(voucher) => write!(f, "inject_voucher {}", voucher as u8),
		}
	}
}

impl FromStr for Inject {
	type Err = ParseError;

	fn from_str(s: &str) -> Result<Inject, ParseError> {
		fn pick<T: Copy>(table: &[T], i: usize, n: &Operands) -> Result<T, ParseError> {
			table.get(i).copied().ok_or_else(|| n.err())
		}

		let (op, mut n) = Operands::new(s);
		let inject = match op.ok_or_else(|| n.err())? {
			"inject_joker" => {
				let joker = pick(&ALL_JOKERS, n.next()?, &n)?;
				let edition = pick(&Edition::ALL, n.next()?, &n)?;
				Inject::Joker {
					joker,
					edition,
					eternal: n.next::<u8>()? != 0,
				}
			}
			"inject_cons" => {
				let kind = match n.next::<u8>()? {
					0 => Kind::Tarot(pick(&ALL_TAROTS, n.next()?, &n)?),
					1 => Kind::Planet(pick(&ALL_PLANETS, n.next()?, &n)?),
					2 => Kind::Spectral(pick(&ALL_SPECTRALS, n.next()?, &n)?),
					_ => return Err(n.err()),
				};
				Inject::Consumable {
					kind,
					edition: pick(&Edition::ALL, n.next()?, &n)?,
				}
			}
			"inject_card" => {
				let suit = pick(&Suit::ALL, n.next()?, &n)?;
				let rank: Rank = n.next()?;
				if !(2..=ACE).contains(&rank) {
					return Err(n.err());
				}
				let enhancement = pick(&Enhancement::ALL, n.next()?, &n)?;
				let edition = pick(&Edition::ALL, n.next()?, &n)?;
				let seal = match n.next::<usize>()? {
					0 => None,
					index => Some(pick(&Seal::ALL, index - 1, &n)?),
				};
				Inject::Card {
					suit,
					rank,
					enhancement,
					edition,
					seal,
				}
			}
			"inject_dollars" => {
				let amount: f64 = n.next()?;
				if !amount.is_finite() {
					return Err(n.err());
				}
				Inject::Dollars(amount)
			}
			"inject_voucher" => Inject::Voucher(pick(&ALL_VOUCHERS, n.next()?, &n)?),
			_ => return Err(n.err()),
		};
		n.end(inject)
	}
}
