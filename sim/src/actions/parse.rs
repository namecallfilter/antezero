//! Action and error definitions, text-code formatting (`Display`), and
//! parsing (`FromStr`, [`Operands`]).

use std::error::Error;
use std::fmt;
use std::str::{FromStr, SplitWhitespace};

use crate::card::{CardMask, HandSort};

/// A decision of the player's, what [`Game::step`](crate::Game::step) applies.
/// It displays as the text code that histories and the bindings use, e.g.
/// `play 11` (card masks in decimal), which [`FromStr`] parses back.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Action {
	SelectBlind,
	SkipBlind,
	RerollBoss,
	Play(CardMask),
	Discard(CardMask),
	/// Use consumable `index` with the given hand cards highlighted.
	UseConsumable {
		index: u8,
		targets: CardMask,
	},
	SellJoker(u8),
	SellConsumable(u8),
	MoveJoker {
		from: u8,
		to: u8,
	},
	MoveHand {
		from: u8,
		to: u8,
	},
	SortHand(HandSort),
	BuyShopCard(u8),
	BuyAndUse {
		index: u8,
		targets: CardMask,
	},
	RedeemVoucher(u8),
	OpenBooster(u8),
	Reroll,
	EndShop,
	CashOut,
	PickPackCard {
		index: u8,
		targets: CardMask,
	},
	SkipPack,
}

/// Why an action is not legal now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActionError {
	WrongPhase,
	BadIndex,
	BadSelection,
	CannotAfford,
	NoSpace,
	NotAllowed,
}

impl fmt::Display for ActionError {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str(match self {
			ActionError::WrongPhase => "not possible in the current phase",
			ActionError::BadIndex => "no card at that index",
			ActionError::BadSelection => "invalid card selection",
			ActionError::CannotAfford => "not enough money",
			ActionError::NoSpace => "no room for the card",
			ActionError::NotAllowed => "not allowed now",
		})
	}
}
impl Error for ActionError {}

impl fmt::Display for Action {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		use Action as A;
		match *self {
			A::SelectBlind => f.write_str("select"),
			A::SkipBlind => f.write_str("skip"),
			A::RerollBoss => f.write_str("reroll_boss"),
			A::Play(m) => write!(f, "play {m}"),
			A::Discard(m) => write!(f, "discard {m}"),
			A::UseConsumable { index, targets } => write!(f, "use {index} {targets}"),
			A::SellJoker(i) => write!(f, "sell_joker {i}"),
			A::SellConsumable(i) => write!(f, "sell_consumable {i}"),
			A::MoveJoker { from, to } => write!(f, "move_joker {from} {to}"),
			A::MoveHand { from, to } => write!(f, "move_hand {from} {to}"),
			A::SortHand(order) => write!(f, "sort {}", u8::from(order == HandSort::Suit)),
			A::BuyShopCard(i) => write!(f, "buy {i}"),
			A::BuyAndUse { index, targets } => write!(f, "buy_use {index} {targets}"),
			A::RedeemVoucher(i) => write!(f, "redeem {i}"),
			A::OpenBooster(i) => write!(f, "open {i}"),
			A::Reroll => f.write_str("reroll"),
			A::EndShop => f.write_str("end_shop"),
			A::CashOut => f.write_str("cash_out"),
			A::PickPackCard { index, targets } => write!(f, "pick {index} {targets}"),
			A::SkipPack => f.write_str("skip_pack"),
		}
	}
}

/// A malformed action or injection code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError(pub String);

impl fmt::Display for ParseError {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "bad code {:?}", self.0)
	}
}
impl Error for ParseError {}

/// The whitespace-separated operands of a text code, parsed strictly (the
/// right count, each in range).
pub(crate) struct Operands<'a> {
	it: SplitWhitespace<'a>,
	code: &'a str,
}

impl<'a> Operands<'a> {
	pub(crate) fn new(code: &'a str) -> (Option<&'a str>, Operands<'a>) {
		let mut it = code.split_whitespace();
		(it.next(), Operands { it, code })
	}

	pub(crate) fn err(&self) -> ParseError {
		ParseError(self.code.to_string())
	}

	pub(crate) fn next<T: FromStr>(&mut self) -> Result<T, ParseError> {
		self.it
			.next()
			.and_then(|t| t.parse().ok())
			.ok_or_else(|| self.err())
	}

	/// Fails if operands are left over.
	pub(crate) fn end<T>(mut self, v: T) -> Result<T, ParseError> {
		if self.it.next().is_some() {
			Err(self.err())
		} else {
			Ok(v)
		}
	}
}

impl FromStr for Action {
	type Err = ParseError;

	fn from_str(s: &str) -> Result<Action, ParseError> {
		use Action as A;
		let (op, mut n) = Operands::new(s);
		let a = match op.ok_or_else(|| n.err())? {
			"select" => A::SelectBlind,
			"skip" => A::SkipBlind,
			"reroll_boss" => A::RerollBoss,
			"play" => A::Play(n.next()?),
			"discard" => A::Discard(n.next()?),
			"use" => A::UseConsumable {
				index: n.next()?,
				targets: n.next()?,
			},
			"sell_joker" => A::SellJoker(n.next()?),
			"sell_consumable" => A::SellConsumable(n.next()?),
			"move_joker" => A::MoveJoker {
				from: n.next()?,
				to: n.next()?,
			},
			"move_hand" => A::MoveHand {
				from: n.next()?,
				to: n.next()?,
			},
			"sort" => A::SortHand(match n.next::<u8>()? {
				0 => HandSort::Rank,
				1 => HandSort::Suit,
				_ => return Err(n.err()),
			}),
			"buy" => A::BuyShopCard(n.next()?),
			"buy_use" => A::BuyAndUse {
				index: n.next()?,
				targets: n.next()?,
			},
			"redeem" => A::RedeemVoucher(n.next()?),
			"open" => A::OpenBooster(n.next()?),
			"reroll" => A::Reroll,
			"end_shop" => A::EndShop,
			"cash_out" => A::CashOut,
			"pick" => A::PickPackCard {
				index: n.next()?,
				targets: n.next()?,
			},
			"skip_pack" => A::SkipPack,
			_ => return Err(n.err()),
		};
		n.end(a)
	}
}
