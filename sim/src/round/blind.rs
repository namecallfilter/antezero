//! Active blind state, ante blind selection, and port of `blind.lua`.

use std::ops::{Index, IndexMut};
use std::slice;

use super::state::Round;
use crate::areas::Pile;
use crate::card::{Card, Forced, PlayingCard, SuitTest};
use crate::data::{ALL_BLINDS, BlindKind, HandSet, HandType, Joker, Suit, TagKind};
use crate::event::{Blocks, Event, EventQueue, Trigger};
use crate::flow::FlowEvent;
use crate::game::{Areas, Game, HandLevels};
use crate::rng::{Key, PseudoRandom};
use crate::run::Run;
use crate::{areas, card, event, flow, num, rng};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlindStatus {
	Select,
	Upcoming,
	Current,
	Defeated,
	Skipped,
}

impl BlindStatus {
	/// Every status, in the order above.
	pub const ALL: [BlindStatus; 5] = [
		BlindStatus::Select,
		BlindStatus::Upcoming,
		BlindStatus::Current,
		BlindStatus::Defeated,
		BlindStatus::Skipped,
	];

	#[must_use]
	pub fn name(self) -> &'static str {
		match self {
			BlindStatus::Select => "Select",
			BlindStatus::Upcoming => "Upcoming",
			BlindStatus::Current => "Current",
			BlindStatus::Defeated => "Defeated",
			BlindStatus::Skipped => "Skipped",
		}
	}
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlindSlot {
	Small = 0,
	Big = 1,
	Boss = 2,
}

impl BlindSlot {
	pub const ALL: [BlindSlot; 3] = [BlindSlot::Small, BlindSlot::Big, BlindSlot::Boss];

	#[must_use]
	pub fn of(kind: BlindKind) -> BlindSlot {
		match kind {
			BlindKind::Small => BlindSlot::Small,
			BlindKind::Big => BlindSlot::Big,
			_ => BlindSlot::Boss,
		}
	}

	#[must_use]
	pub fn name(self) -> &'static str {
		match self {
			BlindSlot::Small => "Small",
			BlindSlot::Big => "Big",
			BlindSlot::Boss => "Boss",
		}
	}
}

/// One value for each blind of an ante.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct PerBlind<T>(pub(crate) [T; BlindSlot::ALL.len()]);

impl<T: Copy> PerBlind<T> {
	/// `value` for every blind.
	#[must_use]
	pub fn all(value: T) -> PerBlind<T> {
		PerBlind([value; BlindSlot::ALL.len()])
	}
}

impl<T> PerBlind<T> {
	/// The values, from the small blind to the boss.
	pub fn iter(&self) -> slice::Iter<'_, T> {
		self.0.iter()
	}
}

impl<'table, T> IntoIterator for &'table PerBlind<T> {
	type IntoIter = slice::Iter<'table, T>;
	type Item = &'table T;

	fn into_iter(self) -> slice::Iter<'table, T> {
		self.iter()
	}
}

impl<T> Index<BlindSlot> for PerBlind<T> {
	type Output = T;

	#[inline]
	fn index(&self, slot: BlindSlot) -> &T {
		&self.0[slot as usize]
	}
}

impl<T> IndexMut<BlindSlot> for PerBlind<T> {
	#[inline]
	fn index_mut(&mut self, slot: BlindSlot) -> &mut T {
		&mut self.0[slot as usize]
	}
}

/// The tags for skipping the ante's small and big blind
/// (`G.GAME.round_resets.blind_tags`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SkipTags {
	pub small: Option<TagKind>,
	pub big: Option<TagKind>,
}

impl SkipTags {
	/// The tag for skipping the blind at `slot`: the boss has none.
	#[must_use]
	pub fn of(self, slot: BlindSlot) -> Option<TagKind> {
		match slot {
			BlindSlot::Small => self.small,
			BlindSlot::Big => self.big,
			BlindSlot::Boss => None,
		}
	}
}

/// The active blind (`G.GAME.blind`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Blind {
	pub kind: Option<BlindKind>,
	pub chips: f64,
	pub dollars: f64,
	pub mult: f64,
	pub disabled: bool,
	pub triggered: bool,
	pub prepped: bool,
	/// The Eye: hands already played this round.
	pub eye_hands: HandSet,
	/// The Mouth: the only allowed hand.
	pub only_hand: Option<HandType>,
	pub discards_sub: f64,
	pub hands_sub: f64,
}

impl Default for Blind {
	/// No blind (`G.GAME.blind` before one is set).
	fn default() -> Blind {
		Blind {
			kind: None,
			chips: 0.0,
			dollars: 0.0,
			mult: 0.0,
			disabled: false,
			triggered: false,
			prepped: true,
			eye_hands: HandSet::default(),
			only_hand: None,
			discards_sub: 0.0,
			hands_sub: 0.0,
		}
	}
}

impl Blind {
	/// Whether the blind is a boss (`G.GAME.blind.boss`).
	#[inline]
	#[must_use]
	pub fn is_boss(&self) -> bool {
		self.kind.is_some_and(BlindKind::is_boss)
	}

	#[inline]
	#[must_use]
	pub fn active(&self, k: BlindKind) -> bool {
		self.kind == Some(k) && !self.disabled
	}
}

/// How often each blind has been the boss (`G.GAME.bosses_used`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BossesUsed([u32; ALL_BLINDS.len()]);

impl Index<BlindKind> for BossesUsed {
	type Output = u32;

	#[inline]
	fn index(&self, blind: BlindKind) -> &u32 {
		&self.0[blind as usize]
	}
}

impl IndexMut<BlindKind> for BossesUsed {
	#[inline]
	fn index_mut(&mut self, blind: BlindKind) -> &mut u32 {
		&mut self.0[blind as usize]
	}
}

/// The ante's blinds (`G.GAME.round_resets`' blind fields).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlindSelect {
	pub states: PerBlind<BlindStatus>,
	/// The ante's boss.
	pub boss: BlindKind,
	pub skip_tags: SkipTags,
	pub boss_rerolled: bool,
	pub selected: Option<BlindKind>,
	pub on_deck: Option<BlindSlot>,
	pub bosses_used: BossesUsed,
}

impl Default for BlindSelect {
	fn default() -> BlindSelect {
		BlindSelect {
			states: PerBlind([
				BlindStatus::Select,
				BlindStatus::Upcoming,
				BlindStatus::Upcoming,
			]),
			boss: BlindKind::Small,
			skip_tags: SkipTags::default(),
			boss_rerolled: false,
			selected: None,
			on_deck: None,
			bosses_used: BossesUsed::default(),
		}
	}
}

impl BlindSelect {
	/// The ante's blind in `slot`.
	#[must_use]
	pub fn kind(&self, slot: BlindSlot) -> BlindKind {
		match slot {
			BlindSlot::Small => BlindKind::Small,
			BlindSlot::Big => BlindKind::Big,
			BlindSlot::Boss => self.boss,
		}
	}

	/// The blind the "select" button refers to.
	#[must_use]
	pub fn on_deck_kind(&self) -> Option<BlindKind> {
		self.on_deck.map(|slot| self.kind(slot))
	}
}

/// What decides whether the blind debuffs a playing card
/// (`Blind:debuff_card`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct CardDebuff {
	/// The blind, unless it is disabled.
	blind: Option<BlindKind>,
	smeared: bool,
	pareidolia: bool,
}

impl CardDebuff {
	pub(crate) fn new(blind: &Blind, jokers: &[Card]) -> CardDebuff {
		CardDebuff {
			blind: if blind.disabled { None } else { blind.kind },
			smeared: card::has_joker(jokers, Joker::Smeared),
			pareidolia: card::has_joker(jokers, Joker::Pareidolia),
		}
	}

	fn debuffs(self, c: &PlayingCard) -> bool {
		let suit = match self.blind {
			Some(BlindKind::Club) => Some(Suit::Clubs),
			Some(BlindKind::Goad) => Some(Suit::Spades),
			Some(BlindKind::Head) => Some(Suit::Hearts),
			Some(BlindKind::Window) => Some(Suit::Diamonds),
			_ => None,
		};
		if let Some(s) = suit
			&& c.is_suit(s, SuitTest::IgnoreDebuff, self.smeared)
		{
			return true;
		}
		match self.blind {
			Some(BlindKind::Plant) => c.is_face_from_boss(self.pareidolia),
			Some(BlindKind::Pillar) => c.played_this_ante,
			Some(BlindKind::FinalLeaf) => true,
			_ => false,
		}
	}

	/// `Blind:debuff_card(card)` for a playing card (`Card:set_debuff`, which
	/// has no side effects for one).
	pub(crate) fn apply(self, c: &mut PlayingCard) {
		c.core.debuff = self.debuffs(c);
	}
}

/// `Blind:set_blind(blind, reset=false)`.
pub(crate) fn set_blind(g: &mut Game, blind: Option<BlindKind>) {
	let mut b = Blind {
		kind: blind,
		..Blind::default()
	};
	if let Some(k) = blind {
		let d = k.def();
		b.dollars = f64::from(d.dollars);
		b.mult = d.mult;
		if g.setup.modifiers.no_blind_reward[BlindSlot::of(k)] {
			b.dollars = 0.0;
		}
		b.chips = blind_amount(g.run.ante, g.setup.modifiers.scaling)
			* b.mult
			* g.setup.starting_params.ante_scaling;
	}
	b.prepped = true;
	g.round.last_blind_boss = b.is_boss();
	g.blind = b;
	if blind.is_some() {
		g.events.push_event(
			Event::Flow(FlowEvent::BlindSetHud),
			Trigger::After(0.05),
			Blocks::Barrier,
		);
	}
	match blind {
		Some(BlindKind::Fish) => g.blind.prepped = false,
		Some(BlindKind::Water) => {
			g.blind.discards_sub = g.round.discards_left;
			g.events.push(Event::EaseDiscard(-g.blind.discards_sub));
		}
		Some(BlindKind::Needle) => {
			g.blind.hands_sub = g.round.base_hands - 1.0;
			g.events.push(Event::EaseHands(-g.blind.hands_sub));
		}
		Some(BlindKind::Manacle) => flow::change_hand_size(&mut g.events, -1.0),
		Some(BlindKind::FinalAcorn) if !g.areas.jokers.is_empty() => {
			for c in &mut g.areas.jokers {
				c.core.face_down = !c.core.face_down;
			}
			if g.areas.jokers.len() > 1 {
				g.events.push_after(0.2, Event::Flow(FlowEvent::AcornOuter));
			}
		}
		_ => {}
	}
	refresh_debuffs(g);
}

/// The blind's debuffs on every playing card and joker, once it changed
/// (the loops of `Blind:set_blind` and `Blind:disable`).
fn refresh_debuffs(g: &mut Game) {
	flow::refresh_playing_debuffs(&mut g.areas, &g.blind);
	// By index: lifting a joker's debuff can bring Chicot back, which
	// disables the blind (the whole game).
	for i in 0..g.areas.jokers.len() {
		lift_joker_debuff(g, i);
	}
}

/// `Blind:debuff_card(joker)`, which lifts the joker's debuff unless The
/// Crimson Heart is active (it picks the debuffed joker itself).
fn lift_joker_debuff(g: &mut Game, i: usize) {
	if g.blind.kind == Some(BlindKind::FinalHeart) && !g.blind.disabled {
		return;
	}
	card::set_debuff_joker(g, i, false);
}

/// `Blind:disable()`.
pub(crate) fn disable_blind(g: &mut Game) {
	g.blind.disabled = true;
	for c in &mut g.areas.jokers {
		c.core.face_down = false;
	}
	match g.blind.kind {
		Some(BlindKind::Water) => g.events.push(Event::EaseDiscard(g.blind.discards_sub)),
		Some(BlindKind::Wheel | BlindKind::House | BlindKind::Mark | BlindKind::Fish) => {
			for c in &mut g.areas.hand.cards {
				c.core.face_down = false;
			}
			for c in g.areas.playing_cards_mut() {
				c.wheel_flipped = false;
			}
		}
		Some(BlindKind::Needle) => g.events.push(Event::EaseHands(g.blind.hands_sub)),
		Some(BlindKind::Wall) => g.blind.chips /= 2.0,
		Some(BlindKind::FinalBell) => {
			for c in g.areas.playing_cards_mut() {
				c.forced = Forced::No;
			}
		}
		Some(BlindKind::Manacle) => {
			flow::change_hand_size(&mut g.events, 1.0);
			flow::draw_from_deck_to_hand(
				&mut g.flow,
				&g.areas,
				&g.round,
				&g.blind,
				&mut g.events,
				Some(1),
			);
		}
		Some(BlindKind::FinalVessel) => g.blind.chips /= 3.0,
		_ => {}
	}
	g.events.push(Event::Flow(FlowEvent::BlindDisableCheck));
	refresh_debuffs(g);
}

/// `Blind:press_play()`; returns its (truthy) result.
pub(crate) fn press_play(jokers: &[Card], blind: &mut Blind, events: &mut EventQueue) -> bool {
	if blind.disabled {
		return false;
	}
	match blind.kind {
		Some(BlindKind::Hook) => {
			events.push(Event::Flow(FlowEvent::HookDiscard));
			blind.triggered = true;
			events.delay(0.7);
			return true;
		}
		Some(BlindKind::FinalHeart) => {
			if !jokers.is_empty() {
				blind.triggered = true;
				blind.prepped = true;
			}
		}
		Some(BlindKind::Fish) => blind.prepped = true,
		Some(BlindKind::Tooth) => {
			events.push_after(0.2, Event::Flow(FlowEvent::ToothOuter));
			blind.triggered = true;
			return true;
		}
		_ => {}
	}
	false
}

/// `Blind:modify_hand` → `(mult, chips)`.
pub(super) fn modify_hand(blind: &mut Blind, mult: f64, chips: f64) -> (f64, f64) {
	if blind.disabled {
		return (mult, chips);
	}
	if blind.kind == Some(BlindKind::Flint) {
		blind.triggered = true;
		return (
			(mult * 0.5 + 0.5).floor().max(1.0),
			(chips * 0.5 + 0.5).floor().max(0.0),
		);
	}
	(mult, chips)
}

/// Whether The Psychic, The Eye or The Mouth makes a play of `n_cards`
/// cards of hand `name` score nothing (`debuff_hand` without its
/// side effects).
pub(crate) fn debuffs_play(blind: &Blind, n_cards: usize, name: HandType) -> bool {
	!blind.disabled
		&& match blind.kind {
			Some(BlindKind::Psychic) => n_cards < 5,
			Some(BlindKind::Eye) => blind.eye_hands.contains(name),
			Some(BlindKind::Mouth) => blind.only_hand.is_some_and(|only| only != name),
			_ => false,
		}
}

/// `Blind:debuff_hand(cards, hand, handname)` for a hand that is played.
pub(super) fn debuff_hand(
	run: &mut Run, most_played_hand: HandType, blind: &mut Blind, hands: &mut HandLevels,
	events: &mut EventQueue, n_cards: usize, name: HandType,
) -> bool {
	if blind.disabled {
		return false;
	}
	blind.triggered = false;
	if debuffs_play(blind, n_cards, name) {
		blind.triggered = true;
		return true;
	}
	match blind.kind {
		Some(BlindKind::Eye) => blind.eye_hands.insert(name),
		Some(BlindKind::Mouth) => blind.only_hand = Some(name),
		_ => {}
	}
	if blind.kind == Some(BlindKind::Arm) {
		blind.triggered = false;
		if hands[name].level > 1.0 {
			blind.triggered = true;
			event::level_up_hand(hands, events, name, -1.0);
		}
	}
	if blind.kind == Some(BlindKind::Ox) {
		blind.triggered = false;
		if name == most_played_hand {
			blind.triggered = true;
			let d = run.dollars;
			flow::ease_dollars(run, -d);
		}
	}
	false
}

/// `Blind:drawn_to_hand()`.
pub(crate) fn drawn_to_hand(g: &mut Game) {
	if !g.blind.disabled {
		if g.blind.kind == Some(BlindKind::FinalBell) && !g.areas.hand.cards.is_empty() {
			let any = g.areas.hand.cards.iter().any(|c| c.forced.is_flagged());
			if !any
				&& let Some(pick) =
					areas::pick_card(&mut g.rng, "cerulean_bell", &g.areas.hand.cards, |_, _| {
						true
					}) {
				g.areas.hand.cards[pick].forced = Forced::Locked;
			}
		}
		if g.blind.kind == Some(BlindKind::FinalHeart)
			&& g.blind.prepped
			&& !g.areas.jokers.is_empty()
		{
			let n = g.areas.jokers.len();
			// Whether each joker was debuffed is read just before its own
			// debuff is lifted: lifting an earlier one can lift the others.
			let mut eligible = vec![false; n];
			for (i, is_eligible) in eligible.iter_mut().enumerate() {
				*is_eligible = !g.areas.jokers[i].core.debuff || n < 2;
				card::set_debuff_joker(g, i, false);
			}
			let picked = areas::pick_card(&mut g.rng, "crimson_heart", &g.areas.jokers, |i, _| {
				eligible[i]
			});
			if let Some(pick) = picked {
				card::set_debuff_joker(g, pick, true);
			}
		}
	}
	g.blind.prepped = false;
}

/// `Blind:stay_flipped(area, card)`.
pub(crate) fn stay_flipped(
	areas: &Areas, probabilities: f64, round: &Round, blind: &Blind, rng: &mut PseudoRandom,
	to: Pile, c: &PlayingCard,
) -> bool {
	if blind.disabled || to != Pile::Hand {
		return false;
	}
	match blind.kind {
		Some(BlindKind::Wheel) => {
			let k = Key::from("wheel");
			rng.random(&k) < probabilities / 7.0
		}
		Some(BlindKind::House) => round.hands_played == 0.0 && round.discards_used == 0.0,
		Some(BlindKind::Mark) => c.is_face_from_boss(areas.pareidolia()),
		Some(BlindKind::Fish) => blind.prepped,
		_ => false,
	}
}

/// `Blind:defeat()` (the parts with game effects).
pub(crate) fn defeat(jokers: &mut [Card], blind: &Blind, events: &mut EventQueue) {
	for c in jokers {
		c.core.face_down = false;
	}
	if blind.kind == Some(BlindKind::Manacle) && !blind.disabled {
		flow::change_hand_size(events, 1.0);
	}
	// `set_blind(nil)` runs at the end of the defeat animation.
	events.push(Event::BlindReset);
}

/// Which table of blind sizes a stake uses (`G.GAME.modifiers.scaling`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BlindScaling {
	#[default]
	Base,
	/// Green Stake and above.
	Faster,
	/// Purple Stake and above.
	Fastest,
}

/// Base chips required for an ante (`get_blind_amount`).
#[must_use]
pub(crate) fn blind_amount(ante: i32, scaling: BlindScaling) -> f64 {
	const BASE: [f64; 8] = [
		300.0, 800.0, 2000.0, 5000.0, 11000.0, 20000.0, 35000.0, 50000.0,
	];
	const FASTER: [f64; 8] = [
		300.0, 900.0, 2600.0, 8000.0, 20000.0, 36000.0, 60000.0, 100_000.0,
	];
	const FASTEST: [f64; 8] = [
		300.0, 1000.0, 3200.0, 9000.0, 25000.0, 60000.0, 110_000.0, 200_000.0,
	];
	let amounts = match scaling {
		BlindScaling::Base => &BASE,
		BlindScaling::Faster => &FASTER,
		BlindScaling::Fastest => &FASTEST,
	};
	let Ok(index) = usize::try_from(ante - 1) else {
		return 100.0;
	};
	if let Some(&amount) = amounts.get(index) {
		return amount;
	}
	// `a * (b + (k * c) ^ d) ^ c` in the game's names: `a` the last listed
	// amount, `c` the antes past the table.
	let past = f64::from(ante - num::id(amounts.len()));
	let power = 1.0 + 0.2 * past;
	let last = amounts[amounts.len() - 1];
	let mut amount = (last * rng::lua_pow(1.6 + rng::lua_pow(0.75 * past, power), past)).floor();
	amount -= rng::lua_mod(amount, rng::lua_pow(10.0, (amount.log10() - 1.0).floor()));
	amount
}

#[cfg(test)]
mod tests {
	use super::*;

	/// The game's blind sizes past ante 8, as its `LuaJIT` computes
	/// `get_blind_amount`: whole powers there are repeated squarings, which
	/// round otherwise than `pow`.
	#[test]
	fn blind_sizes_past_ante_8_are_the_games() {
		for (ante, scaling, chips) in [
			(30, BlindScaling::Base, 2.100_000_000_000_000_6e149),
			(25, BlindScaling::Faster, 4.900_000_000_000_001e87),
			(30, BlindScaling::Faster, 4.300_000_000_000_001e149),
			(25, BlindScaling::Fastest, 9.800_000_000_000_001e87),
			(30, BlindScaling::Fastest, 8.600_000_000_000_002e149),
		] {
			assert_eq!(blind_amount(ante, scaling), chips, "ante {ante}");
		}
	}
}
