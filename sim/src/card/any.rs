//! Unified view of playing and non-playing cards ([`AnyCard`] and
//! [`CardLike`]).

use super::item::{Card, CardCore};
use super::playing_card::PlayingCard;

/// A card of either type, in the areas that can hold both: the shop's card
/// row (playing cards with Magic Trick or Illusion) and an open pack
/// (playing cards in a Standard pack).
#[derive(Clone, Debug, PartialEq)]
pub enum AnyCard {
	Playing(PlayingCard),
	Card(Card),
}

impl From<PlayingCard> for AnyCard {
	fn from(c: PlayingCard) -> AnyCard {
		AnyCard::Playing(c)
	}
}

impl From<Card> for AnyCard {
	fn from(c: Card) -> AnyCard {
		AnyCard::Card(c)
	}
}

/// A card of either type, for the game's functions that work on any card
/// (`Card:set_cost`, `Card:set_edition`, removal by id, ...).
pub trait CardLike {
	fn core(&self) -> &CardCore;

	fn core_mut(&mut self) -> &mut CardCore;

	/// The card, unless it is a playing card.
	fn as_card(&self) -> Option<&Card>;

	fn as_card_mut(&mut self) -> Option<&mut Card>;

	/// Whether it is a playing card (it has no center).
	fn is_playing(&self) -> bool {
		self.as_card().is_none()
	}

	/// Whether it is a tarot, planet or spectral card
	/// (`ability.consumeable`).
	fn is_consumable(&self) -> bool {
		self.as_card().is_some_and(|c| c.kind.is_consumable())
	}
}

impl CardLike for Card {
	fn core(&self) -> &CardCore {
		&self.core
	}

	fn core_mut(&mut self) -> &mut CardCore {
		&mut self.core
	}

	fn as_card(&self) -> Option<&Card> {
		Some(self)
	}

	fn as_card_mut(&mut self) -> Option<&mut Card> {
		Some(self)
	}
}

impl CardLike for PlayingCard {
	fn core(&self) -> &CardCore {
		&self.core
	}

	fn core_mut(&mut self) -> &mut CardCore {
		&mut self.core
	}

	fn as_card(&self) -> Option<&Card> {
		None
	}

	fn as_card_mut(&mut self) -> Option<&mut Card> {
		None
	}
}

impl CardLike for AnyCard {
	fn core(&self) -> &CardCore {
		match self {
			AnyCard::Playing(c) => &c.core,
			AnyCard::Card(c) => &c.core,
		}
	}

	fn core_mut(&mut self) -> &mut CardCore {
		match self {
			AnyCard::Playing(c) => &mut c.core,
			AnyCard::Card(c) => &mut c.core,
		}
	}

	fn as_card(&self) -> Option<&Card> {
		match self {
			AnyCard::Playing(_) => None,
			AnyCard::Card(c) => Some(c),
		}
	}

	fn as_card_mut(&mut self) -> Option<&mut Card> {
		match self {
			AnyCard::Playing(_) => None,
			AnyCard::Card(c) => Some(c),
		}
	}
}
