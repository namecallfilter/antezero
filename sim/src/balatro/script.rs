//! The `.bal` script format: what a recording [`BalEnv`](super::BalEnv)
//! writes and what [`load_starts`] replays.
//!
//! A script is the header line `seed S deck D stake N` followed by one line
//! per action. Cards of the hand are listed by position (`play 0 3 4`,
//! `use 1 2`). Text after a `#` is a comment, and blank lines do not count.

use std::error::Error;
use std::fmt;
use std::str::SplitWhitespace;

use crate::actions::Action;
use crate::card::{AnyCard, CardLike, CardMask, HandSort};
use crate::data::Back;
use crate::game::{Game, State};
use crate::{MAX_SEED_LEN, Stake, consumables, event, shop};

/// A mid-run state that episodes can start from, see [`load_starts`].
#[derive(Clone, Debug)]
pub struct StartPoint {
	/// The game at the start point.
	pub game: Game,
	/// The script that leads here: the header and the action lines. The
	/// recording of an episode that starts here begins with it.
	pub script: String,
	/// The caller's number for the point (`-1` until it gives one).
	pub id: i32,
}

/// Why a script does not replay.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScriptError {
	/// The script has no line.
	Empty,
	/// The first line is not `seed S deck D stake N`.
	Header(String),
	/// The header's seed is longer than [`MAX_SEED_LEN`] bytes.
	Seed(String),
	/// The header's deck is unknown or needs rules that are not simulated.
	Deck(String),
	/// The header's stake is not the number of a stake.
	Stake(String),
	/// Action line `line` (counted from 1 after the header) is not a line
	/// of the format.
	Unparseable { line: usize, text: String },
	/// Action line `line` is not legal in the game the lines before it lead
	/// to.
	Illegal { line: usize, text: String },
	/// A requested step is past the script's last action line, which is
	/// line `lines`.
	TooShort { lines: usize },
	/// The run is over after action line `line`, so no episode starts
	/// there.
	Over { line: usize },
}

impl fmt::Display for ScriptError {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			ScriptError::Empty => f.write_str("empty script"),
			ScriptError::Header(line) => {
				write!(f, "header '{line}' is not 'seed S deck D stake N'")
			}
			ScriptError::Seed(seed) => {
				write!(f, "seed {seed} is longer than {MAX_SEED_LEN} bytes")
			}
			ScriptError::Deck(deck) => write!(f, "unsupported deck {deck}"),
			ScriptError::Stake(stake) => write!(f, "invalid stake {stake}"),
			ScriptError::Unparseable { line, text } => {
				write!(f, "line {line} '{text}' unparseable")
			}
			ScriptError::Illegal { line, text } => write!(f, "line {line} '{text}' failed"),
			ScriptError::TooShort { lines } => write!(f, "script has only {lines} action lines"),
			ScriptError::Over { line } => write!(f, "the run is over after line {line}"),
		}
	}
}

impl Error for ScriptError {}

/// The first line of a script: the run its action lines play.
pub(super) struct Header<'a> {
	pub(super) seed: &'a str,
	pub(super) back: Back,
	pub(super) stake: Stake,
}

impl fmt::Display for Header<'_> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(
			f,
			"seed {} deck {} stake {}",
			self.seed,
			self.back.key(),
			u8::from(self.stake)
		)
	}
}

impl<'a> Header<'a> {
	fn parse(line: &'a str) -> Result<Header<'a>, ScriptError> {
		let words: Vec<&str> = line.split_whitespace().collect();
		let &["seed", seed, "deck", deck, "stake", stake] = words.as_slice() else {
			return Err(ScriptError::Header(line.to_string()));
		};
		if seed.len() > MAX_SEED_LEN {
			return Err(ScriptError::Seed(seed.to_string()));
		}
		let back = Back::from_key(deck)
			.filter(|back| back.is_simulated())
			.ok_or_else(|| ScriptError::Deck(deck.to_string()))?;
		let stake = stake
			.parse::<u8>()
			.ok()
			.and_then(|number| Stake::try_from(number).ok())
			.ok_or_else(|| ScriptError::Stake(stake.to_string()))?;
		Ok(Header { seed, back, stake })
	}
}

/// A line of a script after the header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Step<'a> {
	Act(Action),
	/// Keep playing a won run (Endless Mode); legal only right after the win.
	Endless,
	/// Restart every random stream from this seed. Cards already dealt stay.
	Reseed(&'a str),
	/// Skip a pack that the game does not let the player skip, see
	/// [`skip_refused`]; legal only there.
	ForcedSkip,
}

/// Hand positions as a script lists them: each one after a space.
struct Positions(CardMask);

impl fmt::Display for Positions {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		self.0.indices().try_for_each(|i| write!(f, " {i}"))
	}
}

impl fmt::Display for Step<'_> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		use Action as A;
		let act = match *self {
			Step::Act(act) => act,
			Step::Endless => return f.write_str("endless"),
			Step::Reseed(seed) => return write!(f, "dbg_reseed {seed}"),
			Step::ForcedSkip => return f.write_str("dbg_skip_pack"),
		};
		match act {
			A::Play(cards) => write!(f, "play{}", Positions(cards)),
			A::Discard(cards) => write!(f, "discard{}", Positions(cards)),
			A::UseConsumable { index, targets } => write!(f, "use {index}{}", Positions(targets)),
			A::BuyAndUse { index, targets } => write!(f, "buy_use {index}{}", Positions(targets)),
			A::PickPackCard { index, targets } => write!(f, "pick {index}{}", Positions(targets)),
			A::SellConsumable(i) => write!(f, "sell_cons {i}"),
			A::SortHand(HandSort::Rank) => f.write_str("sort_rank"),
			A::SortHand(HandSort::Suit) => f.write_str("sort_suit"),
			A::SelectBlind
			| A::SkipBlind
			| A::RerollBoss
			| A::SellJoker(_)
			| A::MoveJoker { .. }
			| A::MoveHand { .. }
			| A::BuyShopCard(_)
			| A::RedeemVoucher(_)
			| A::OpenBooster(_)
			| A::Reroll
			| A::EndShop
			| A::CashOut
			| A::SkipPack => act.fmt(f),
		}
	}
}

/// The operands of a line: the words after its verb.
struct Operands<'a>(SplitWhitespace<'a>);

impl<'a> Operands<'a> {
	/// The next operand, a slot number.
	fn slot(&mut self) -> Option<u8> {
		self.0.next()?.parse().ok()
	}

	/// The rest of the operands, hand positions.
	fn positions(self) -> Option<CardMask> {
		self.0
			.map(|word| {
				let i: usize = word.parse().ok()?;
				(i < CardMask::CAPACITY).then_some(i)
			})
			.collect()
	}

	/// `value` if no operand is left.
	fn end<T>(mut self, value: T) -> Option<T> {
		self.0.next().is_none().then_some(value)
	}

	/// The last operand, a seed.
	fn seed(mut self) -> Option<&'a str> {
		let seed = self.0.next().filter(|seed| seed.len() <= MAX_SEED_LEN)?;
		self.end(seed)
	}
}

impl<'a> Step<'a> {
	/// The line that [`Step`]'s `Display` writes as `line`, with any
	/// spacing and with the hand positions in any order.
	fn parse(line: &'a str) -> Option<Step<'a>> {
		use Action as A;
		let mut words = line.split_whitespace();
		let verb = words.next()?;
		let mut ops = Operands(words);
		let act = match verb {
			"play" => A::Play(ops.positions()?),
			"discard" => A::Discard(ops.positions()?),
			"use" => A::UseConsumable {
				index: ops.slot()?,
				targets: ops.positions()?,
			},
			"buy_use" => A::BuyAndUse {
				index: ops.slot()?,
				targets: ops.positions()?,
			},
			"pick" => A::PickPackCard {
				index: ops.slot()?,
				targets: ops.positions()?,
			},
			"move_joker" => {
				let (from, to) = (ops.slot()?, ops.slot()?);
				ops.end(A::MoveJoker { from, to })?
			}
			"move_hand" => {
				let (from, to) = (ops.slot()?, ops.slot()?);
				ops.end(A::MoveHand { from, to })?
			}
			"sell_joker" => {
				let slot = ops.slot()?;
				ops.end(A::SellJoker(slot))?
			}
			"sell_cons" => {
				let slot = ops.slot()?;
				ops.end(A::SellConsumable(slot))?
			}
			"buy" => {
				let slot = ops.slot()?;
				ops.end(A::BuyShopCard(slot))?
			}
			"redeem" => {
				let slot = ops.slot()?;
				ops.end(A::RedeemVoucher(slot))?
			}
			"open" => {
				let slot = ops.slot()?;
				ops.end(A::OpenBooster(slot))?
			}
			"select" => ops.end(A::SelectBlind)?,
			"skip" => ops.end(A::SkipBlind)?,
			"reroll_boss" => ops.end(A::RerollBoss)?,
			"reroll" => ops.end(A::Reroll)?,
			"end_shop" => ops.end(A::EndShop)?,
			"cash_out" => ops.end(A::CashOut)?,
			"skip_pack" => ops.end(A::SkipPack)?,
			"sort_rank" => ops.end(A::SortHand(HandSort::Rank))?,
			"sort_suit" => ops.end(A::SortHand(HandSort::Suit))?,
			"endless" => return ops.end(Step::Endless),
			"dbg_skip_pack" => return ops.end(Step::ForcedSkip),
			"dbg_reseed" => return ops.seed().map(Step::Reseed),
			_ => return None,
		};
		Some(Step::Act(act))
	}

	/// Applies the line to `g`; `false`, and `g` unchanged, if the game does
	/// not allow it.
	pub(super) fn apply(self, g: &mut Game) -> bool {
		match self {
			Step::Act(act) => g.step(without_ignored_targets(g, act)).is_ok(),
			Step::Endless => {
				let on_win_screen =
					g.run_won() && g.flow.state == State::RoundEval && !g.setup.endless;
				g.setup.endless |= on_win_screen;
				on_win_screen
			}
			Step::Reseed(seed) => {
				g.rng.reseed(seed);
				true
			}
			Step::ForcedSkip => {
				let refused = skip_refused(g);
				if refused {
					event::begin_action(&mut g.rng, &mut g.events);
					shop::do_skip_pack(g);
					event::run_queue(g);
				}
				refused
			}
		}
	}
}

/// Whether `g` is in an open pack that the game does not let the player
/// skip: an Arcana or Spectral pack with an empty hand.
pub(super) fn skip_refused(g: &Game) -> bool {
	g.flow.state.is_pack() && !g.areas.pack_cards.is_empty() && g.check(Action::SkipPack).is_err()
}

fn takes_targets(card: &AnyCard) -> bool {
	card.as_card()
		.is_some_and(|card| consumables::targets(card.kind).is_some())
}

/// `act` as [`Game::step`] takes it. A script lists the hand cards that were
/// highlighted when a card was used, bought and used or picked. The game
/// ignores them if the card takes no targets, where `Game::step` refuses
/// them, so the positions listed for such a card are dropped if they are
/// cards of the hand.
fn without_ignored_targets(g: &Game, act: Action) -> Action {
	use Action as A;
	let areas = &g.areas;
	let ignored = |takes_targets: Option<bool>, listed: CardMask| {
		takes_targets == Some(false) && listed.fits(areas.hand.cards.len())
	};
	let at = |cards: &[AnyCard], index: u8| cards.get(usize::from(index)).map(takes_targets);
	let none = CardMask::EMPTY;
	match act {
		A::UseConsumable { index, targets } => {
			let card = areas.consumables.get(usize::from(index));
			let takes = card.map(|card| consumables::targets(card.kind).is_some());
			let targets = if ignored(takes, targets) {
				none
			} else {
				targets
			};
			A::UseConsumable { index, targets }
		}
		A::BuyAndUse { index, targets } if ignored(at(&areas.shop_jokers, index), targets) => {
			A::BuyAndUse {
				index,
				targets: none,
			}
		}
		A::PickPackCard { index, targets } if ignored(at(&areas.pack_cards, index), targets) => {
			A::PickPackCard {
				index,
				targets: none,
			}
		}
		_ => act,
	}
}

/// Replays `script` and returns, for each entry of `steps`, the game after
/// that many action lines (`0`: the new run of the header). Lines after the
/// largest step are not read.
///
/// # Errors
///
/// Returns the first reason there is no start point for every step: the
/// header or a line up to the largest step is not of the format, a line is
/// not legal in the game, the script is shorter than a step, or the run is
/// over at a step. No start point is returned then.
pub fn load_starts(script: &str, steps: &[usize]) -> Result<Vec<StartPoint>, ScriptError> {
	if steps.is_empty() {
		return Ok(Vec::new());
	}
	let mut lines = script
		.lines()
		.map(|raw| raw.split_once('#').map_or(raw, |(code, _)| code).trim())
		.filter(|line| !line.is_empty());
	let header_line = lines.next().ok_or(ScriptError::Empty)?;
	let header = Header::parse(header_line)?;
	let mut game = Game::new_run(header.seed, header.back, header.stake);
	game.set_tracing(false);
	let mut text = format!("{header_line}\n");

	let mut wanted: Vec<usize> = (0..steps.len()).collect();
	wanted.sort_unstable_by_key(|&i| steps[i]);
	let mut wanted = wanted.into_iter().peekable();
	let mut points: Vec<(usize, StartPoint)> = Vec::with_capacity(steps.len());
	let mut line = 0;
	loop {
		while let Some(i) = wanted.next_if(|&i| steps[i] == line) {
			if game.is_terminal() {
				return Err(ScriptError::Over { line });
			}
			let point = StartPoint {
				game: game.clone(),
				script: text.clone(),
				id: -1,
			};
			points.push((i, point));
		}
		if wanted.peek().is_none() {
			break;
		}
		let code = lines.next().ok_or(ScriptError::TooShort { lines: line })?;
		line += 1;
		let step = Step::parse(code).ok_or_else(|| ScriptError::Unparseable {
			line,
			text: code.to_string(),
		})?;
		if !step.apply(&mut game) {
			return Err(ScriptError::Illegal {
				line,
				text: code.to_string(),
			});
		}
		text.push_str(code);
		text.push('\n');
	}
	points.sort_unstable_by_key(|&(i, _)| i);
	Ok(points.into_iter().map(|(_, point)| point).collect())
}

#[cfg(test)]
mod tests {
	use std::sync::Arc;

	use super::super::{ACT_SKIP_PACK, BalConfig, BalEnv};
	use super::*;
	use crate::data::Joker;
	use crate::inject::Inject;
	use crate::{canon, num};

	/// An open Mega Spectral pack with Immolate, and a hand of five (Stuntman
	/// and Merry Andy).
	fn spectral_pack() -> Game {
		let mut game = Game::new_run("H28D9", Back::Red, Stake::White);
		game.inject(Inject::joker(Joker::Stuntman));
		game.inject(Inject::joker(Joker::MerryAndy));
		play(&mut game, &["select", "play 2 3", "cash_out", "open 1"]);
		game
	}

	/// The pack of [`spectral_pack`] after Immolate has destroyed the whole
	/// hand: a pick is left and the game refuses the skip.
	fn stuck_pack() -> Game {
		let mut game = spectral_pack();
		play(&mut game, &["pick 0 0"]);
		assert!(game.areas.hand.cards.is_empty() && skip_refused(&game));
		game
	}

	fn play(game: &mut Game, lines: &[&str]) {
		for line in lines {
			let step = Step::parse(line).expect("a line of the format");
			assert!(step.apply(game), "{line} is legal");
		}
	}

	#[test]
	fn a_forced_pack_skip_is_recorded_as_its_own_line_and_replays() {
		let stuck = stuck_pack();
		let mut env = BalEnv::new(BalConfig {
			disable_softlocks: true,
			record: true,
			..BalConfig::default()
		});
		let start = StartPoint {
			game: stuck.clone(),
			script: String::new(),
			id: 0,
		};
		env.set_start(Some(Arc::new(start)), false);
		env.reset("START");
		let info = env.step_select(num::id(ACT_SKIP_PACK), &[]);
		assert_eq!((info.illegal, info.softlocks_avoided), (0, 1));
		assert_eq!(env.script(), "dbg_skip_pack\n");

		let mut replayed = stuck.clone();
		assert!(Step::parse("dbg_skip_pack").is_some_and(|step| step.apply(&mut replayed)));
		assert_eq!(
			canon::from_sim(&mut replayed),
			canon::from_sim(&mut env.game().clone())
		);
		assert_ne!(replayed.flow.state, stuck.flow.state, "the pack is left");
	}

	#[test]
	fn a_forced_pack_skip_is_legal_only_where_the_game_refuses_the_skip() {
		let mut stuck = stuck_pack();
		assert!(!Step::Act(Action::SkipPack).apply(&mut stuck));

		let mut open = spectral_pack();
		let before = canon::from_sim(&mut open.clone());
		assert!(!Step::ForcedSkip.apply(&mut open));
		assert_eq!(canon::from_sim(&mut open), before);
		assert!(Step::Act(Action::SkipPack).apply(&mut open));
	}
}
