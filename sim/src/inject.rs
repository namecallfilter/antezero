//! Scenario construction helpers: inject cards, money and vouchers into a
//! running game (mirroring the game's own `add_joker` /
//! `create_playing_card`). Useful for curricula, unit tests and the oracle
//! differential tests, which apply identical injections to the real game.

use std::fmt;
use std::str::FromStr;

use crate::actions::{Operands, ParseError};
use crate::card::{Card, CardLike, Kind, Pricing};
use crate::data::{
	ALL_JOKERS, ALL_PLANETS, ALL_SPECTRALS, ALL_TAROTS, ALL_VOUCHERS, Edition, Enhancement, Joker,
	Planet, Rank, Seal, Spectral, Suit, Tarot, Voucher,
};
use crate::event::EditionTiming;
use crate::game::{Area, Areas, Game};
use crate::pools::OwnedArea;
use crate::shop::VoucherSource;
use crate::{card, event, shop};

/// A scenario injection applied identically to both engines.
///
/// Its text code, used in histories (written by `Display`, read back by
/// `FromStr`), gives every enum as its discriminant, an edition as
/// [`Edition::id`] and a seal as 1 + its discriminant (0 for none):
/// `inject_joker <joker> <edition> <eternal: 0 or 1>`,
/// `inject_cons <set: 0 tarot, 1 planet, 2 spectral> <card> <edition>`,
/// `inject_card <suit> <rank id> <enhancement> <edition> <seal>`,
/// `inject_dollars <amount>` and `inject_voucher <voucher>`; any eternal
/// byte but 0 reads as eternal. For example, `inject_cons 1 0 0` injects
/// Mercury with no edition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Inject {
	/// `add_joker(key, edition, silent, eternal)` for a joker.
	Joker {
		joker: Joker,
		edition: Option<Edition>,
		eternal: bool,
	},
	/// `add_joker(key, edition)` for a tarot, planet or spectral card.
	Consumable {
		card: Consumable,
		edition: Option<Edition>,
	},
	/// `create_playing_card({front, center}, G.deck)` + edition and seal.
	Card {
		suit: Suit,
		rank: Rank,
		enhancement: Enhancement,
		edition: Option<Edition>,
		seal: Option<Seal>,
	},
	Dollars(f64),
	/// Marks a voucher redeemed and applies it (like a starting deck voucher).
	Voucher(Voucher),
}

/// The card of an [`Inject::Consumable`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Consumable {
	Tarot(Tarot),
	Planet(Planet),
	Spectral(Spectral),
}

impl Consumable {
	/// The text code's set and card numbers.
	fn code(self) -> (u8, u8) {
		match self {
			Consumable::Tarot(tarot) => (0, tarot as u8),
			Consumable::Planet(planet) => (1, planet as u8),
			Consumable::Spectral(spectral) => (2, spectral as u8),
		}
	}

	/// The card a text code's set and card numbers name, if any.
	fn from_code(set: u8, card: u8) -> Option<Consumable> {
		let card = usize::from(card);
		match set {
			0 => ALL_TAROTS.get(card).copied().map(Consumable::Tarot),
			1 => ALL_PLANETS.get(card).copied().map(Consumable::Planet),
			2 => ALL_SPECTRALS.get(card).copied().map(Consumable::Spectral),
			_ => None,
		}
	}
}

impl From<Consumable> for Kind {
	fn from(card: Consumable) -> Kind {
		match card {
			Consumable::Tarot(tarot) => Kind::Tarot(tarot),
			Consumable::Planet(planet) => Kind::Planet(planet),
			Consumable::Spectral(spectral) => Kind::Spectral(spectral),
		}
	}
}

impl Inject {
	/// The joker `joker` with no edition, not eternal.
	#[must_use]
	pub fn joker(joker: Joker) -> Inject {
		Inject::Joker {
			joker,
			edition: None,
			eternal: false,
		}
	}
}

/// See [`Game::inject`].
pub(crate) fn inject(g: &mut Game, i: Inject) {
	event::begin_action(&mut g.rng, &mut g.events);
	match i {
		Inject::Joker {
			joker,
			edition,
			eternal,
		} => {
			let mut c = make_owned_card(g, Kind::Joker(joker), edition, OwnedArea::Jokers);
			if eternal {
				c.make_eternal();
			}
			g.areas.jokers.push(c);
		}
		Inject::Consumable { card, edition } => {
			let c = make_owned_card(g, card.into(), edition, OwnedArea::Consumables);
			g.areas.consumables.push(c);
		}
		Inject::Card {
			suit,
			rank,
			enhancement,
			edition,
			seal,
		} => {
			let mut c = card::make_playing_card(
				&mut g.areas,
				g.run.hands_played,
				g.shop.discount_percent,
				suit,
				rank,
				enhancement,
			);
			c.core.face_down = true;
			give_edition(
				&mut c,
				edition,
				Area::Deck,
				&mut g.areas,
				g.shop.discount_percent,
			);
			if seal.is_some() {
				c.seal = seal;
				Pricing::new(g.shop.discount_percent, &g.areas.jokers)
					.set_cost(&mut c, Some(Area::Deck));
			}
			g.areas.deck.insert(0, c);
		}
		Inject::Dollars(amount) => {
			g.run.dollars += amount;
			// Money moves without an event: nothing to run.
			return;
		}
		Inject::Voucher(voucher) => {
			g.history.used_vouchers.insert(voucher);
			shop::apply_voucher(&mut g.round, &mut g.events, voucher, VoucherSource::Deck);
		}
	}
	event::run_queue(g);
}

/// An injected joker or consumable: made, materialized and added to the
/// deck, with its edition, on its way into `area`.
fn make_owned_card(g: &mut Game, kind: Kind, edition: Option<Edition>, area: OwnedArea) -> Card {
	let mut c = card::make_card(g, kind);
	g.events.materialize(1.0);
	card::add_to_deck(g, &mut c);
	give_edition(
		&mut c,
		edition,
		area.into(),
		&mut g.areas,
		g.shop.discount_percent,
	);
	g.events.edition_fx(edition, EditionTiming::Delayed);
	c
}

/// `set_edition` for an injected card in `area`; one without an edition is
/// left as it was made.
fn give_edition(
	c: &mut impl CardLike, edition: Option<Edition>, area: Area, areas: &mut Areas,
	discount_percent: f64,
) {
	if edition.is_some() {
		let pricing = Pricing::new(discount_percent, &areas.jokers);
		card::set_edition(c, edition, Some(area), pricing, &mut areas.limits);
	}
}

impl fmt::Display for Inject {
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
					Edition::id(edition),
					u8::from(eternal)
				)
			}
			Inject::Consumable { card, edition } => {
				let (set, index) = card.code();
				write!(f, "inject_cons {set} {index} {}", Edition::id(edition))
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
					"inject_card {} {} {} {} {}",
					suit as u8,
					rank.id(),
					enhancement as u8,
					Edition::id(edition),
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
		/// An edition's number (see [`Edition::id`]).
		fn edition(n: &mut Operands) -> Result<Option<Edition>, ParseError> {
			Edition::from_id(n.next()?).ok_or_else(|| n.err())
		}

		let (op, mut n) = Operands::new(s);
		let inject = match op.ok_or_else(|| n.err())? {
			"inject_joker" => {
				let joker = pick(&ALL_JOKERS, n.next()?, &n)?;
				let edition = edition(&mut n)?;
				Inject::Joker {
					joker,
					edition,
					eternal: n.next::<u8>()? != 0,
				}
			}
			"inject_cons" => {
				let set = n.next()?;
				let card = Consumable::from_code(set, n.next()?).ok_or_else(|| n.err())?;
				Inject::Consumable {
					card,
					edition: edition(&mut n)?,
				}
			}
			"inject_card" => {
				let suit = pick(&Suit::ALL, n.next()?, &n)?;
				let rank = Rank::from_id(n.next()?).ok_or_else(|| n.err())?;
				let enhancement = pick(&Enhancement::ALL, n.next()?, &n)?;
				let edition = edition(&mut n)?;
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
