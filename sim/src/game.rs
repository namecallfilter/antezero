//! Game state: the card areas and the parts of `G.GAME`, each its own type.
//! The game's logic lives in the modules that port it.

use std::ops::{Index, IndexMut};
use std::slice;

use crate::actions::{self, Action, ActionError};
pub use crate::areas::{Area, Areas, Hand, Limit, Limits};
use crate::card::CardMask;
pub use crate::card::HandSort;
use crate::data::{Back, HandType};
use crate::dump::{self, DumpError};
use crate::event::EventQueue;
use crate::inject::{self, Inject};
use crate::json::Json;
use crate::rng::PseudoRandom;
pub use crate::round::{
	Blind, BlindScaling, BlindSelect, BlindSlot, BlindStatus, BossesUsed, IdolCard, MailRank,
	PerBlind, Round, SkipTags,
};
pub use crate::run::{
	History, Modifiers, OrbitalChoices, Run, Setup, Stake, StartingParams, Tag, Tags,
};
pub use crate::shop::{BOOSTER_SLOTS, OpenPack, PackSlot, PackState, Shop};
use crate::{round, run};

/// Decision points exposed to the agent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Phase {
	BlindSelect,
	Round,
	Shop,
	Pack,
	RoundEval,
	GameOver,
}

impl Phase {
	/// Every phase, in the order above.
	pub const ALL: [Phase; 6] = [
		Phase::BlindSelect,
		Phase::Round,
		Phase::Shop,
		Phase::Pack,
		Phase::RoundEval,
		Phase::GameOver,
	];

	/// The phase's name, as the Python bindings show it.
	#[must_use]
	pub fn name(self) -> &'static str {
		match self {
			Phase::BlindSelect => "blind_select",
			Phase::Round => "round",
			Phase::Shop => "shop",
			Phase::Pack => "pack",
			Phase::RoundEval => "round_eval",
			Phase::GameOver => "game_over",
		}
	}
}

/// `G.STATE`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum State {
	SelectingHand,
	HandPlayed,
	DrawToHand,
	GameOver,
	Shop,
	PlayTarot,
	BlindSelect,
	RoundEval,
	TarotPack,
	PlanetPack,
	SpectralPack,
	StandardPack,
	BuffoonPack,
	NewRound,
}

impl State {
	/// Every state, in the order above.
	pub const ALL: [State; 14] = [
		State::SelectingHand,
		State::HandPlayed,
		State::DrawToHand,
		State::GameOver,
		State::Shop,
		State::PlayTarot,
		State::BlindSelect,
		State::RoundEval,
		State::TarotPack,
		State::PlanetPack,
		State::SpectralPack,
		State::StandardPack,
		State::BuffoonPack,
		State::NewRound,
	];

	#[must_use]
	pub fn is_pack(self) -> bool {
		matches!(
			self,
			State::TarotPack
				| State::PlanetPack
				| State::SpectralPack
				| State::StandardPack
				| State::BuffoonPack
		)
	}

	#[must_use]
	pub fn name(self) -> &'static str {
		match self {
			State::SelectingHand => "SELECTING_HAND",
			State::HandPlayed => "HAND_PLAYED",
			State::DrawToHand => "DRAW_TO_HAND",
			State::GameOver => "GAME_OVER",
			State::Shop => "SHOP",
			State::PlayTarot => "PLAY_TAROT",
			State::BlindSelect => "BLIND_SELECT",
			State::RoundEval => "ROUND_EVAL",
			State::TarotPack => "TAROT_PACK",
			State::PlanetPack => "PLANET_PACK",
			State::SpectralPack => "SPECTRAL_PACK",
			State::StandardPack => "STANDARD_PACK",
			State::BuffoonPack => "BUFFOON_PACK",
			State::NewRound => "NEW_ROUND",
		}
	}
}

/// `G.GAME.hands`: every poker hand's level and play counts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HandLevels(pub(crate) [HandLevel; HandType::ALL.len()]);

impl HandLevels {
	/// The levels, in [`HandType::ALL`] order.
	pub fn iter(&self) -> slice::Iter<'_, HandLevel> {
		self.0.iter()
	}

	pub fn iter_mut(&mut self) -> slice::IterMut<'_, HandLevel> {
		self.0.iter_mut()
	}

	/// `pseudorandom_element` over the visible hands other than `exclude`,
	/// which the game lists in `pairs(G.GAME.hands)` order.
	pub(crate) fn random_visible(
		&self, rng: &mut PseudoRandom, key: &str, exclude: Option<HandType>,
	) -> HandType {
		let mut visible = [HandType::HighCard; HandType::ALL.len()];
		let mut n = 0;
		for h in HandType::PAIRS_ORDER {
			if self[h].visible && Some(h) != exclude {
				visible[n] = h;
				n += 1;
			}
		}
		visible[rng.element_str(key, n)]
	}

	/// `level_up_hand(card, hand, instant, amount)`: raises `hand` by `amount`
	/// levels with no planet used and no events.
	pub fn level_up(&mut self, hand: HandType, amount: f64) {
		let base = hand.base();
		let level = &mut self[hand];
		level.level = (level.level + amount).max(0.0);
		level.mult = (base.mult + base.level_mult * (level.level - 1.0)).max(1.0);
		level.chips = (base.chips + base.level_chips * (level.level - 1.0)).max(0.0);
	}
}

impl<'table> IntoIterator for &'table HandLevels {
	type IntoIter = slice::Iter<'table, HandLevel>;
	type Item = &'table HandLevel;

	fn into_iter(self) -> slice::Iter<'table, HandLevel> {
		self.iter()
	}
}

impl<'table> IntoIterator for &'table mut HandLevels {
	type IntoIter = slice::IterMut<'table, HandLevel>;
	type Item = &'table mut HandLevel;

	fn into_iter(self) -> slice::IterMut<'table, HandLevel> {
		self.iter_mut()
	}
}

impl Index<HandType> for HandLevels {
	type Output = HandLevel;

	#[inline]
	fn index(&self, hand: HandType) -> &HandLevel {
		&self.0[hand as usize]
	}
}

impl IndexMut<HandType> for HandLevels {
	#[inline]
	fn index_mut(&mut self, hand: HandType) -> &mut HandLevel {
		&mut self.0[hand as usize]
	}
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HandLevel {
	pub level: f64,
	pub chips: f64,
	pub mult: f64,
	pub played: f64,
	pub played_this_round: f64,
	pub visible: bool,
}

/// The sets of the cards other than playing cards that `create_card` can
/// sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CardSet {
	Joker,
	Tarot,
	Planet,
	Spectral,
}

impl CardSet {
	pub(crate) fn name(self) -> &'static str {
		match self {
			CardSet::Joker => "Joker",
			CardSet::Tarot => "Tarot",
			CardSet::Planet => "Planet",
			CardSet::Spectral => "Spectral",
		}
	}
}

/// Where the run is: the decision point, `G.STATE`, and flags that last for
/// one action.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Flow {
	pub phase: Phase,
	pub state: State,
	/// `G.STATE_COMPLETE`.
	pub state_complete: bool,
	/// Set by `evaluate_play` (used by `preview_play` to stop early).
	pub(crate) scored: bool,
	/// The hand-name text was cleared since the cards were selected (The
	/// Hook's discard), so scoring re-announces the hand.
	pub(crate) hand_text_cleared: bool,
	pub(crate) end_round_pending: bool,
	/// Free reorders (hand moves / sorts, joker moves) since the last other
	/// action (observation only): [`crate::obs`] stops offering them at a
	/// cap, else a policy can stall a game forever.
	pub rearranges: u8,
}

impl Default for Flow {
	fn default() -> Flow {
		Flow {
			phase: Phase::BlindSelect,
			state: State::BlindSelect,
			state_complete: false,
			scored: false,
			hand_text_cleared: false,
			end_round_pending: false,
			rearranges: 0,
		}
	}
}

/// A run.
#[derive(Debug)]
pub struct Game {
	pub rng: PseudoRandom,
	/// `G.E_MANAGER`'s queue and the timers it runs on.
	pub(crate) events: EventQueue,
	pub setup: Setup,
	pub flow: Flow,
	pub areas: Areas,
	pub run: Run,
	pub history: History,
	pub round: Round,
	pub blind_select: BlindSelect,
	/// The active blind (`G.GAME.blind`).
	pub blind: Blind,
	pub shop: Shop,
	pub pack: PackState,
	/// `G.GAME.hands`.
	pub hands: HandLevels,
	pub tags: Tags,
}

impl Game {
	/// Starts a new seeded run (fully unlocked profile, no challenge):
	/// `seed` of at most [`MAX_SEED_LEN`](crate::MAX_SEED_LEN) bytes.
	///
	/// # Panics
	///
	/// Panics if `seed` is longer than that.
	#[must_use]
	pub fn new_run(seed: &str, back: Back, stake: Stake) -> Game {
		run::new_run(seed, back, stake)
	}

	/// The game in a state dump of the real game or the oracle (see
	/// [`crate::dump`]), taken at a decision point.
	///
	/// # Errors
	///
	/// Returns the first part of the dump the simulator cannot take.
	pub fn from_dump(dump: &Json) -> Result<Game, DumpError> {
		dump::read(dump)
	}

	/// The state dump the game would give in this state, a decision point
	/// (see [`crate::dump`]); refreshes the per-frame derived joker values
	/// first, like the game does every frame.
	#[must_use]
	pub fn to_dump(&mut self) -> Json {
		self.update_derived();
		dump::write(self)
	}

	/// Checks an action without applying it.
	///
	/// # Errors
	///
	/// Returns why `a` is not legal now.
	pub fn check(&self, a: Action) -> Result<(), ActionError> {
		actions::check(self, a)
	}

	/// Applies an action and runs the game until the next decision point.
	///
	/// # Errors
	///
	/// Returns why `a` is not legal now; the game is then unchanged.
	pub fn step(&mut self, a: Action) -> Result<(), ActionError> {
		actions::step(self, a)
	}

	/// The legal actions at the current decision point, in a stable order:
	/// card subsets for play / discard / consumable targets are enumerated
	/// exhaustively; reordering (`MoveJoker`, `MoveHand`, `SortHand`) is
	/// legal (see [`Game::check`]) but not enumerated.
	pub fn legal_actions(&self, out: &mut Vec<Action>) {
		actions::legal_actions(self, out);
	}

	/// Whether the run has ended: lost (including [`Game::is_stuck`]), or won
	/// (round evaluation of the final boss) unless `endless`. After a win
	/// further actions stay possible; stopping is the caller's decision.
	#[must_use]
	pub fn is_terminal(&self) -> bool {
		self.flow.phase == Phase::GameOver
			|| (self.run_won() && self.flow.state == State::RoundEval && !self.setup.endless)
			|| self.is_stuck()
	}

	/// The real game's softlock: no legal action left. Only an Arcana or
	/// Spectral pack with an empty hand gets here (e.g. Immolate with hand
	/// size 5): `can_skip_booster` needs hand cards and nothing redraws.
	/// The run cannot continue, so it counts as lost.
	#[must_use]
	pub fn is_stuck(&self) -> bool {
		self.flow.phase == Phase::Pack
			&& self.areas.hand.cards.is_empty()
			&& self.check(Action::SkipPack).is_err()
			&& {
				let mut v = Vec::new();
				self.legal_actions(&mut v);
				v.is_empty()
			}
	}

	/// Applies a scenario injection (see [`crate::inject`]).
	pub fn inject(&mut self, i: Inject) {
		inject::inject(self, i);
	}

	/// Recomputes the joker values the game's `Card:update` refreshes every
	/// frame (Steel / Stone tallies, Stencil, Swashbuckler, Temperance, ...).
	/// Observers call it before reading a state.
	pub fn update_derived(&mut self) {
		round::update_derived(&mut self.areas, self.run.skips);
	}

	/// Replaces everything a player cannot know with a fresh sample, keeping
	/// the visible state: every future random outcome (shops, packs, bosses,
	/// tags, chance effects), the draw order of the deck, the identities of
	/// face-down hand cards (drawn again from the unseen cards), the order of
	/// face-down jokers, and the ante's voucher if no shop has shown it yet.
	/// Cards are put in a canonical order (visible content, then id) before
	/// each shuffle, so the result depends on the visible state and `seed`
	/// only, not on the hidden arrangement it replaces.
	/// For determinized search and for restarting from snapshots.
	pub fn resample_hidden(&mut self, seed: &str) {
		run::resample(
			&mut self.areas,
			&self.run,
			&self.history,
			&mut self.shop,
			&mut self.rng,
			seed,
		);
	}

	/// Score `Play(mask)` would make, computed on `self`, which is left
	/// mid-hand and must be discarded (call it on a clone).
	///
	/// # Errors
	///
	/// Returns why the play is not legal now.
	///
	/// # Panics
	///
	/// Panics if the event queue stalls before the hand is scored (a simulator bug).
	pub fn preview_play(&mut self, mask: CardMask) -> Result<f64, ActionError> {
		actions::preview_play(self, mask)
	}

	/// Like [`Game::preview_play`], but the hand's own effects are applied
	/// to `self` as well (money, created and destroyed cards, the played
	/// cards discarded), before anything is drawn or the round ends.
	///
	/// # Errors
	///
	/// Returns why the play is not legal now.
	///
	/// # Panics
	///
	/// Panics if the event queue stalls before the hand is played out (a simulator bug).
	pub fn preview_hand(&mut self, mask: CardMask) -> Result<f64, ActionError> {
		actions::preview_hand(self, mask)
	}

	/// Makes this game a copy of `source` that draws from a copy of `rng`,
	/// in the buffers it has: what a search makes for each candidate in place
	/// of a fresh clone.
	pub(crate) fn copy_from(&mut self, source: &Game, rng: &PseudoRandom) {
		let Game {
			rng: own_rng,
			events,
			setup,
			flow,
			areas,
			run,
			history,
			round,
			blind_select,
			blind,
			shop,
			pack,
			hands,
			tags,
		} = self;
		own_rng.clone_from(rng);
		events.clone_from(&source.events);
		areas.clone_from(&source.areas);
		tags.clone_from(&source.tags);
		*setup = source.setup;
		*run = source.run;
		*flow = source.flow;
		*history = source.history;
		*round = source.round;
		*blind_select = source.blind_select;
		*blind = source.blind;
		*shop = source.shop;
		*pack = source.pack;
		*hands = source.hands;
	}

	/// Turns timeline recording on or off (a no-op without `trace-keys`).
	pub fn set_tracing(&mut self, on: bool) {
		self.rng.trace.set_enabled(on);
	}

	/// Starts logging the completed events of each action (see
	/// [`Game::event_log`]).
	#[cfg(feature = "trace-keys")]
	pub fn enable_event_log(&mut self) {
		self.rng.trace.enable_event_log();
	}

	/// Completed events of the current action (`trace-keys` builds).
	#[cfg(feature = "trace-keys")]
	#[must_use]
	pub fn event_log(&self) -> &[String] {
		self.rng.trace.event_log()
	}

	/// The current action's timeline (empty without `trace-keys`).
	#[must_use]
	pub fn timeline(&self) -> &[String] {
		self.rng.trace.entries()
	}

	/// Whether the run was won (the final boss was beaten). Unlike `won`,
	/// false when the run ended on the final boss.
	#[must_use]
	pub fn run_won(&self) -> bool {
		self.run.win_notified
	}
}

// Written out, like the `Clone` of the parts with buffers, so that
// `clone_from` reuses the buffers: a search refreshes one copy per candidate.
impl Clone for Game {
	fn clone(&self) -> Game {
		Game {
			rng: self.rng.clone(),
			events: self.events.clone(),
			areas: self.areas.clone(),
			tags: self.tags.clone(),
			..*self
		}
	}

	fn clone_from(&mut self, source: &Game) {
		self.copy_from(source, &source.rng);
	}
}

/// A game to try actions on: each try gets a copy of the real game, built in
/// the buffers of the copy before it.
#[derive(Debug, Default)]
pub struct Scratch(Option<Game>);

impl Scratch {
	/// A copy of `source` that draws from a copy of `rng`, made in the
	/// buffers of the previous copy.
	pub fn copy(&mut self, source: &Game, rng: &PseudoRandom) -> &mut Game {
		if let Some(ref mut copy) = self.0 {
			copy.copy_from(source, rng);
			copy
		} else {
			let mut first = source.clone();
			first.rng.clone_from(rng);
			self.0.insert(first)
		}
	}
}
