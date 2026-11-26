//! The `.bal` script format: what a recording [`BalEnv`](super::BalEnv)
//! writes and what [`load_starts`] replays.
//!
//! A script is a header line (`seed S deck D stake N`) followed by one line
//! per action. Cards of the hand are listed by position (`play 0 3 4`,
//! `use 1 2`). Text after a `#` is a comment, and a line that starts with `?`
//! is optional: it is skipped when it cannot be applied.

use std::error::Error;
use std::fmt;
use std::str::SplitWhitespace;

use crate::actions::Action;
use crate::card::{CardMask, HandSort};
use crate::data::Back;
use crate::game::{Game, State};
use crate::{MAX_SEED_LEN, Stake};

/// A mid-run state that episodes can start from, see [`load_starts`].
#[derive(Clone, Debug)]
pub struct StartPoint {
	/// The game at the start point.
	pub game: Game,
	/// The script that leads here: the header and the applied action lines.
	/// The recording of an episode that starts here begins with it.
	pub script: String,
	/// The run's challenge. Always empty: challenge runs are not simulated.
	pub challenge: String,
	/// The caller's number for the point (`-1` until it gives one).
	pub id: i32,
}

/// Why a script does not replay.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScriptError {
	/// The script has no header line.
	Empty,
	/// The first line names no seed.
	NoSeed,
	/// The header's seed is longer than [`MAX_SEED_LEN`] bytes.
	Seed(String),
	/// The header's deck is unknown or needs rules that are not simulated.
	Deck(String),
	/// The header's stake is not the number of a stake.
	Stake(String),
	/// The header names a challenge, and challenge runs are not simulated.
	Challenge(String),
	/// Action line `line` (counted from 1 after the header) is not a line
	/// of the format.
	Unparseable { line: usize, text: String },
	/// Action line `line` is not legal in the game the lines before it lead
	/// to.
	Illegal { line: usize, text: String },
	/// A requested step is past the script's last action line, which is
	/// line `lines`.
	TooShort { lines: usize },
}

impl fmt::Display for ScriptError {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			ScriptError::Empty => f.write_str("empty script"),
			ScriptError::NoSeed => f.write_str("no header"),
			ScriptError::Seed(seed) => {
				write!(f, "seed {seed} is longer than {MAX_SEED_LEN} bytes")
			}
			ScriptError::Deck(deck) => write!(f, "unsupported deck {deck}"),
			ScriptError::Stake(stake) => write!(f, "invalid stake {stake}"),
			ScriptError::Challenge(challenge) => write!(f, "challenge {challenge} not supported"),
			ScriptError::Unparseable { line, text } => {
				write!(f, "line {line} '{text}' unparseable")
			}
			ScriptError::Illegal { line, text } => write!(f, "line {line} '{text}' failed"),
			ScriptError::TooShort { lines } => write!(f, "script has only {lines} action lines"),
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
	/// Reads `name value` pairs in any order. The deck defaults to the Red
	/// Deck and the stake to White; other words are ignored.
	fn parse(line: &'a str) -> Result<Header<'a>, ScriptError> {
		let mut words = line.split_whitespace();
		let (mut seed, mut back, mut stake) = (None, Back::Red, Stake::White);
		while let Some(word) = words.next() {
			match word {
				"seed" => seed = words.next(),
				"deck" => {
					let key = words.next().unwrap_or_default();
					back = Back::from_key(key)
						.filter(|b| b.is_simulated())
						.ok_or_else(|| ScriptError::Deck(key.to_string()))?;
				}
				"stake" => {
					let number = words.next().unwrap_or_default();
					stake = number
						.parse::<u8>()
						.ok()
						.and_then(|n| Stake::try_from(n).ok())
						.ok_or_else(|| ScriptError::Stake(number.to_string()))?;
				}
				"challenge" => {
					if let Some(challenge) = words.next() {
						return Err(ScriptError::Challenge(challenge.to_string()));
					}
				}
				_ => {}
			}
		}
		let seed = seed.ok_or(ScriptError::NoSeed)?;
		if seed.len() > MAX_SEED_LEN {
			return Err(ScriptError::Seed(seed.to_string()));
		}
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

fn positions<'a>(words: impl Iterator<Item = &'a str>) -> Option<CardMask> {
	words
		.map(|word| {
			let i: usize = word.parse().ok()?;
			(i < CardMask::CAPACITY).then_some(i)
		})
		.collect()
}

impl<'a> Step<'a> {
	fn parse(line: &'a str) -> Option<Step<'a>> {
		let mut words = line.split_whitespace();
		let index = |words: &mut SplitWhitespace<'_>| -> Option<u8> { words.next()?.parse().ok() };
		let act = match words.next()? {
			"play" => Action::Play(positions(words)?),
			"discard" => Action::Discard(positions(words)?),
			"use" => Action::UseConsumable {
				index: index(&mut words)?,
				targets: positions(words)?,
			},
			"buy_use" => Action::BuyAndUse {
				index: index(&mut words)?,
				targets: positions(words)?,
			},
			"pick" => Action::PickPackCard {
				index: index(&mut words)?,
				targets: positions(words)?,
			},
			"sell" => Action::SellJoker(index(&mut words)?),
			"sell_cons" => Action::SellConsumable(index(&mut words)?),
			"sort_rank" => Action::SortHand(HandSort::Rank),
			"sort_suit" => Action::SortHand(HandSort::Suit),
			"endless" => return Some(Step::Endless),
			"dbg_reseed" => {
				let seed = words.next().filter(|seed| seed.len() <= MAX_SEED_LEN)?;
				return words.next().is_none().then_some(Step::Reseed(seed));
			}
			_ => line.parse().ok()?,
		};
		Some(Step::Act(act))
	}

	/// Applies the line to `g`; `false` if the game does not allow it.
	fn apply(self, g: &mut Game) -> bool {
		match self {
			Step::Act(act) => {
				g.step(act).is_ok() || untargeted(act).is_some_and(|a| g.step(a).is_ok())
			}
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
		}
	}
}

/// `act` without its hand positions, if it has some. A recording lists the
/// cards highlighted when a card was used, bought or picked, and the real
/// game ignores them where the card takes no targets; [`Game::step`] refuses
/// them there.
fn untargeted(act: Action) -> Option<Action> {
	let none = CardMask::EMPTY;
	match act {
		Action::UseConsumable { index, targets } if targets != none => {
			Some(Action::UseConsumable {
				index,
				targets: none,
			})
		}
		Action::BuyAndUse { index, targets } if targets != none => Some(Action::BuyAndUse {
			index,
			targets: none,
		}),
		Action::PickPackCard { index, targets } if targets != none => Some(Action::PickPackCard {
			index,
			targets: none,
		}),
		_ => None,
	}
}

/// Replays `script` and returns, for each entry of `steps`, the game after
/// that many action lines (`0`: the new run of the header).
///
/// An entry is `None` where the run is over at that step and where the
/// step's line is an optional line that was skipped (the script of a start
/// point leaves skipped lines out). Lines after the largest step are not
/// read.
///
/// # Errors
///
/// Returns the first reason the script does not replay through the largest
/// step. No start point is returned then, including those of earlier steps:
/// a script that stops replaying is not the game that was recorded.
pub fn load_starts(script: &str, steps: &[usize]) -> Result<Vec<Option<StartPoint>>, ScriptError> {
	let mut out = vec![None; steps.len()];
	let Some(&last) = steps.iter().max() else {
		return Ok(out);
	};
	let mut lines = script
		.lines()
		.map(|raw| raw.split_once('#').map_or(raw, |(code, _)| code).trim())
		.filter(|line| !line.is_empty());
	let header_line = lines.next().ok_or(ScriptError::Empty)?;
	let header = Header::parse(header_line)?;
	let mut game = Game::new_run(header.seed, header.back, header.stake);
	game.set_tracing(false);
	let mut text = format!("{header_line}\n");
	let mut capture = |game: &Game, text: &str, step: usize| {
		if game.is_terminal() {
			return;
		}
		for (point, _) in out.iter_mut().zip(steps).filter(|&(_, &s)| s == step) {
			*point = Some(StartPoint {
				game: game.clone(),
				script: text.to_string(),
				challenge: String::new(),
				id: -1,
			});
		}
	};
	capture(&game, &text, 0);
	let mut line = 0;
	for raw in lines.take(last) {
		line += 1;
		let optional = raw.starts_with('?');
		let code = raw.trim_start_matches('?').trim_start();
		let Some(step) = Step::parse(code) else {
			if optional {
				continue;
			}
			return Err(ScriptError::Unparseable {
				line,
				text: code.to_string(),
			});
		};
		if step.apply(&mut game) {
			text.push_str(code);
			text.push('\n');
			capture(&game, &text, line);
		} else if !optional {
			return Err(ScriptError::Illegal {
				line,
				text: code.to_string(),
			});
		}
	}
	if line < last {
		return Err(ScriptError::TooShort { lines: line });
	}
	Ok(out)
}
