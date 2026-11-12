//! Game state (`G.GAME` + card areas), the small queries every module
//! shares and the public entry points. The game's logic lives in the modules
//! that port it (see the crate docs): each is a set of functions over a
//! [`Game`].

/// Appends a timeline entry (`trace-keys` builds; see [`crate::rng::Trace`]).
macro_rules! trace {
    ($g:expr, $($t:tt)*) => {
        #[cfg(feature = "trace-keys")]
        $g.rng.trace.record(format_args!($($t)*));
    };
}
pub(crate) use trace;

use crate::actions::{self, Action, ActionError, CardMask};
use crate::card::{Card, Kind};
use crate::data::{
	Back, BlindKind, Booster, CenterSet, HandType, Joker, PackKind, Rank, Suit, TagKind, Voucher,
};
use crate::events::{EventQueue, St};
use crate::hand_eval::EvalFlags;
use crate::inject::{self, Inject};
use crate::rng::{Key, PseudoRandom};
use crate::{run, scoring, search};

/// Decision points exposed to the agent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Phase {
	/// Choosing (or skipping) the next blind.
	BlindSelect,
	/// Selecting cards to play/discard.
	Round,
	/// Between rounds.
	Shop,
	/// A booster pack is open (see [`Game::pack`]).
	Pack,
	/// Round won; waiting for "Cash Out".
	RoundEval,
	/// Run lost.
	GameOver,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlindStatus {
	Select,
	Upcoming,
	Current,
	Defeated,
	Skipped,
}

impl BlindStatus {
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
	/// The slot `kind` occupies in an ante (every boss is `Boss`).
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HandLevel {
	pub level: f64,
	pub chips: f64,
	pub mult: f64,
	pub played: f64,
	pub played_this_round: f64,
	pub visible: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tag {
	pub kind: TagKind,
	pub triggered: bool,
	pub orbital_hand: Option<HandType>,
	pub id: u32,
}

/// The active blind (`G.GAME.blind`).
#[derive(Clone, Copy, Debug, PartialEq)]
#[expect(
	clippy::struct_excessive_bools,
	reason = "independent flags, one per field of the game's `Blind`"
)]
pub struct Blind {
	pub kind: Option<BlindKind>,
	pub chips: f64,
	pub dollars: f64,
	pub mult: f64,
	pub disabled: bool,
	pub triggered: bool,
	pub prepped: bool,
	pub boss: bool,
	/// The Eye: hands already played this round.
	pub eye_hands: u16,
	/// The Mouth: the only allowed hand.
	pub only_hand: Option<HandType>,
	pub discards_sub: f64,
	pub hands_sub: f64,
}

impl Blind {
	#[must_use]
	pub fn empty() -> Blind {
		Blind {
			kind: None,
			chips: 0.0,
			dollars: 0.0,
			mult: 0.0,
			disabled: false,
			triggered: false,
			prepped: true,
			boss: false,
			eye_hands: 0,
			only_hand: None,
			discards_sub: 0.0,
			hands_sub: 0.0,
		}
	}

	#[inline]
	#[must_use]
	pub fn is(&self, k: BlindKind) -> bool {
		self.kind == Some(k)
	}

	/// `Blind:get_type()`.
	pub fn slot(&self) -> Option<BlindSlot> {
		self.kind.map(BlindSlot::of)
	}

	/// The blind is a boss whose effect is active.
	#[inline]
	#[must_use]
	pub fn active(&self, k: BlindKind) -> bool {
		self.kind == Some(k) && !self.disabled
	}
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
#[expect(
	clippy::struct_excessive_bools,
	reason = "independent flags, one per `G.GAME.modifiers` field"
)]
pub struct Modifiers {
	pub no_blind_reward: [bool; 3],
	pub scaling: u8,
	pub enable_eternals_in_shop: bool,
	pub enable_perishables_in_shop: bool,
	pub enable_rentals_in_shop: bool,
	pub no_interest: bool,
	pub money_per_hand: Option<f64>,
	pub money_per_discard: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StartingParams {
	pub dollars: f64,
	pub hand_size: f64,
	pub discards: f64,
	pub hands: f64,
	pub reroll_cost: f64,
	pub joker_slots: f64,
	pub ante_scaling: f64,
	pub consumable_slots: f64,
	pub no_faces: bool,
	pub erratic_suits_and_ranks: bool,
}

impl Default for StartingParams {
	fn default() -> Self {
		StartingParams {
			dollars: 4.0,
			hand_size: 8.0,
			discards: 3.0,
			hands: 4.0,
			reroll_cost: 5.0,
			joker_slots: 5.0,
			ante_scaling: 1.0,
			consumable_slots: 2.0,
			no_faces: false,
			erratic_suits_and_ranks: false,
		}
	}
}

/// One of the shop's two booster slots this round (`G.GAME.current_round.used_packs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackSlot {
	/// No pack has been rolled for the slot yet.
	Unrolled,
	/// The slot offers this pack.
	Offered(Booster),
	/// The slot's pack was opened.
	Opened,
}

/// The Idol's card (`G.GAME.current_round.idol_card`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IdolCard {
	pub rank: Rank,
	pub suit: Suit,
	/// The card's id, which keeps its last valid value like the game's.
	pub id: Option<Rank>,
}

/// Mail-In Rebate's rank (`G.GAME.current_round.mail_card`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MailRank {
	pub rank: Rank,
	/// The rank's id, which keeps its last valid value like the game's.
	pub id: Option<Rank>,
}

/// The Orbital Tag hands rolled for an ante's three blinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OrbitalChoices {
	pub ante: i32,
	/// By [`BlindSlot`].
	pub hands: [Option<HandType>; 3],
}

/// An open booster pack (the state to return to is `pack_interrupt`, the
/// remaining picks are `Game::pack_choices`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PackState {
	pub kind: PackKind,
}

/// A card area with its `config.card_limit` (+ `real_card_limit` bookkeeping).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Limit {
	pub limit: f64,
	pub real: f64,
}

impl Limit {
	#[must_use]
	pub fn new(l: f64) -> Limit {
		Limit { limit: l, real: l }
	}

	/// `CardArea:change_size(delta)` (the deferred part).
	#[inline]
	pub fn change(&mut self, delta: f64) {
		self.real += delta;
		self.limit = self.real.max(0.0);
	}
}

/// Which `Card:open`/area a newly created card goes into (affects polls).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Area {
	Jokers,
	Consumables,
	Hand,
	Deck,
	Play,
	Discard,
	ShopJokers,
	ShopVouchers,
	ShopBoosters,
	PackCards,
}

/// Card sets that `create_card` can sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CardSet {
	Joker,
	Tarot,
	Planet,
	Spectral,
	TarotPlanet,
	Base,
	Enhanced,
}

impl CardSet {
	pub(crate) fn name(self) -> &'static str {
		match self {
			CardSet::Joker => "Joker",
			CardSet::Tarot => "Tarot",
			CardSet::Planet => "Planet",
			CardSet::Spectral => "Spectral",
			CardSet::TarotPlanet => "Tarot_Planet",
			CardSet::Base => "Base",
			CardSet::Enhanced => "Enhanced",
		}
	}
}

#[derive(Clone, Debug)]
#[expect(
	clippy::struct_excessive_bools,
	reason = "independent flags, one per `G.GAME` field"
)]
pub struct Game {
	pub rng: PseudoRandom,
	pub back: Back,
	pub stake: u8,
	pub modifiers: Modifiers,
	pub starting_params: StartingParams,
	pub phase: Phase,
	pub won: bool,
	/// Keep playing after winning ("Endless mode"); otherwise the run is
	/// terminal once the winning round is evaluated.
	pub endless: bool,
	/// Blinds defeated or skipped so far (progress measure for agents).
	pub blinds_cleared: u32,
	/// `G.GAME.win_notified`: the run was won (`won` is also set when the
	/// final boss is lost, like the game's `G.GAME.won`).
	pub(crate) win_notified: bool,
	/// The win overlay paused the game for the rest of the current frame.
	pub(crate) pause_rest: bool,
	/// Set by `evaluate_play` (used by `preview_play` to stop early).
	pub(crate) scored: bool,
	/// The hand-name text was cleared since the cards were selected (The
	/// Hook's discard), so scoring re-announces the hand.
	pub(crate) hand_text_cleared: bool,

	// Counters.
	pub next_sort_id: u32,

	// Areas.
	pub deck: Vec<Card>,
	pub hand: Vec<Card>,
	pub play: Vec<Card>,
	pub discard: Vec<Card>,
	pub jokers: Vec<Card>,
	pub consumables: Vec<Card>,
	pub shop_jokers: Vec<Card>,
	pub shop_vouchers: Vec<Card>,
	pub shop_boosters: Vec<Card>,
	pub pack_cards: Vec<Card>,
	pub joker_limit: Limit,
	pub consumable_limit: Limit,
	pub hand_limit: Limit,
	/// `G.hand.config.sort` ('desc' by rank or 'suit desc').
	pub hand_sort_by_suit: bool,
	pub pack: Option<PackState>,

	// G.GAME
	pub ante: i32,
	pub blind_ante: i32,
	pub win_ante: i32,
	pub round: i32,
	pub dollars: f64,
	pub chips: f64,
	pub hands_left: f64,
	pub discards_left: f64,
	pub hands_played_round: f64,
	pub discards_used: f64,
	pub hands_played: f64,
	pub unused_discards: f64,
	pub skips: f64,
	pub reroll_cost: f64,
	pub reroll_cost_increase: f64,
	pub free_rerolls: f64,
	pub used_packs: [PackSlot; 2],
	pub current_voucher: Option<Voucher>,
	/// The ante's voucher has been shown in a shop (observation only).
	pub voucher_seen: bool,
	/// Free reorders (hand moves / sorts, joker moves) since the last other
	/// action (observation only): [`crate::obs::ops`] stops offering them at
	/// a cap, else a policy can stall a game forever.
	pub rearranges: u8,
	/// Whether [`crate::obs::ops`] offers skipping blinds (observation only:
	/// a training switch; the game itself always allows skips).
	pub allow_skip: bool,
	/// Round ×Mult like Steamodded, which the real game runs under the bridge
	/// (see `Game::x_mult`); off, like vanilla and the oracle.
	pub steamodded: bool,
	pub idol_card: IdolCard,
	pub mail_rank: MailRank,
	pub ancient_suit: Option<Suit>,
	pub castle_suit: Suit,
	pub most_played_hand: HandType,
	pub round_dollars: f64,

	pub rr_hands: f64,
	pub rr_discards: f64,
	pub rr_reroll_cost: f64,
	pub temp_reroll_cost: Option<f64>,
	pub temp_handsize: Option<f64>,
	pub blind_states: [BlindStatus; 3],
	pub boss_choice: BlindKind,
	pub blind_tags: [Option<TagKind>; 2],
	pub boss_rerolled: bool,
	pub selected_blind: Option<BlindKind>,
	pub blind_on_deck: Option<BlindSlot>,
	pub round_bonus_next_hands: f64,
	pub round_bonus_discards: f64,

	pub hands: [HandLevel; 12],
	pub probabilities: f64,
	pub bosses_used: [u32; 30],
	pub used_jokers: CenterSet,
	pub used_vouchers: u32,
	pub banned: CenterSet,
	pub banned_tags: u32,
	pub banned_blinds: u32,
	pub gros_michel_extinct: bool,
	pub planets_used: u16,
	pub consumable_usage_tarot: f64,
	pub consumable_usage_planet: f64,
	pub consumable_usage_spectral: f64,
	pub last_tarot_planet: Option<Kind>,
	pub last_hand_played: Option<HandType>,
	pub tags: Vec<Tag>,
	pub next_tag_id: u32,
	pub shop_joker_max: f64,
	pub edition_rate: f64,
	pub joker_rate: f64,
	pub tarot_rate: f64,
	pub planet_rate: f64,
	pub spectral_rate: f64,
	pub playing_card_rate: f64,
	pub consumeable_buffer: f64,
	pub joker_buffer: f64,
	pub dollar_buffer: f64,
	pub discount_percent: f64,
	pub interest_cap: f64,
	pub interest_amount: f64,
	pub bankrupt_at: f64,
	pub perishable_rounds: f64,
	pub rental_rate: f64,
	pub ecto_minus: f64,
	pub pack_size: u8,
	pub pack_choices: u8,
	pub first_shop_buffoon: bool,
	pub orbital_choices: Vec<OrbitalChoices>,
	pub shop_free: bool,
	pub shop_d6ed: bool,
	pub starting_deck_size: f64,
	pub last_blind_boss: bool,
	pub facing_blind: bool,
	pub blind: Blind,

	/// `G.STATE` / `G.STATE_COMPLETE`.
	pub state: St,
	pub state_complete: bool,
	/// `G.E_MANAGER`'s queue and the timers it runs on.
	pub(crate) events: EventQueue,
	/// Cards outside any area (used consumables, opened boosters) until
	/// their dissolve animation removes them.
	pub(crate) consumed: Vec<Card>,
	pub(crate) in_flight: Option<Card>,
	pub(crate) pending_highlight: Vec<usize>,
	pub(crate) pack_interrupt: Option<St>,
	pub(crate) end_round_pending: bool,
	pub(crate) burnt_hand: Option<HandType>,
	/// `G.shop` exists.
	pub shop_open: bool,
	pub(crate) shop_exists_at_open: bool,
	pub(crate) ankh_deletable: Vec<u32>,
	/// Ids of the cards that scored in the last evaluated hand.
	pub last_scoring: Vec<u32>,
	/// Score of the last evaluated hand.
	pub last_hand_score: f64,
}

impl Game {
	// -----------------------------------------------------------------
	// Entry points
	// -----------------------------------------------------------------

	/// Starts a new seeded run (fully unlocked profile, no challenge):
	/// `seed` of at most [`MAX_SEED_LEN`](crate::MAX_SEED_LEN) bytes, `stake`
	/// 1..=8.
	///
	/// # Panics
	///
	/// Panics if `seed` is longer than that or `stake` is not in `1..=8`.
	#[must_use]
	pub fn new_run(seed: &str, back: Back, stake: u8) -> Game {
		run::new_run(seed, back, stake)
	}

	/// Checks an action without applying it.
	///
	/// # Errors
	///
	/// Returns why `a` is not legal now.
	pub fn check(&self, a: &Action) -> Result<(), ActionError> {
		actions::check(self, *a)
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
		self.phase == Phase::GameOver
			|| (self.run_won() && self.state == St::RoundEval && !self.endless)
			|| self.is_stuck()
	}

	/// The real game's softlock: no legal action left. Only an Arcana or
	/// Spectral pack with an empty hand gets here (e.g. Immolate with hand
	/// size 5): `can_skip_booster` needs hand cards and nothing redraws.
	/// The run cannot continue, so it counts as lost.
	#[must_use]
	pub fn is_stuck(&self) -> bool {
		self.phase == Phase::Pack
			&& self.hand.is_empty()
			&& self.check(&Action::SkipPack).is_err()
			&& {
				let mut v = Vec::new();
				self.legal_actions(&mut v);
				v.is_empty()
			}
	}

	/// Applies a scenario injection (see [`crate::inject`]).
	///
	/// # Panics
	///
	/// Panics if an [`Inject::Consumable`] is not a tarot, planet or spectral card.
	pub fn inject(&mut self, i: &Inject) {
		inject::inject(self, i);
	}

	/// Recomputes the joker values the game's `Card:update` refreshes every
	/// frame (Steel / Stone tallies, Stencil, Swashbuckler, Temperance, ...).
	/// Observers call it before reading a state.
	pub fn update_derived(&mut self) {
		scoring::update_derived(self);
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
		search::resample_hidden(self, seed);
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
		search::preview_play(self, mask)
	}

	/// The blind the "select" button refers to.
	#[must_use]
	pub fn blind_on_deck_kind(&self) -> Option<BlindKind> {
		self.blind_on_deck.map(|s| match s {
			BlindSlot::Small => BlindKind::Small,
			BlindSlot::Big => BlindKind::Big,
			BlindSlot::Boss => self.boss_choice,
		})
	}

	/// Jokers that change how poker hands are evaluated.
	#[must_use]
	pub fn eval_flags(&self) -> EvalFlags {
		EvalFlags {
			four_fingers: self.has_joker(Joker::FourFingers),
			shortcut: self.has_joker(Joker::Shortcut),
			smeared: self.smeared(),
		}
	}

	/// Turns timeline recording on or off (a no-op without `trace-keys`).
	pub fn set_tracing(&mut self, on: bool) {
		#[cfg(feature = "trace-keys")]
		{
			self.rng.trace.enabled = on;
			if !on {
				self.rng.trace.entries = Vec::new();
			}
		}
		#[cfg(not(feature = "trace-keys"))]
		let _ = on;
	}

	/// Enables the completed-event log (`trace-keys` builds).
	#[cfg(feature = "trace-keys")]
	pub fn enable_evlog(&mut self) {
		self.rng.trace.evlog = Some(Vec::new());
	}

	/// Completed events of the current action (`trace-keys` builds).
	#[cfg(feature = "trace-keys")]
	#[must_use]
	pub fn evlog(&self) -> &[String] {
		self.rng.trace.evlog.as_deref().unwrap_or(&[])
	}

	/// The current action's timeline (empty without `trace-keys`).
	#[must_use]
	pub fn timeline(&self) -> &[String] {
		#[cfg(feature = "trace-keys")]
		{
			&self.rng.trace.entries
		}
		#[cfg(not(feature = "trace-keys"))]
		{
			&[]
		}
	}

	// -----------------------------------------------------------------
	// Small helpers
	// -----------------------------------------------------------------

	/// Whether the run was won (the final boss was beaten). Unlike `won`,
	/// false when the run ended on the final boss.
	#[must_use]
	pub fn run_won(&self) -> bool {
		self.win_notified
	}

	/// `base .. ante`.
	#[inline]
	pub(crate) fn key_ante(&self, base: &str) -> Key {
		let mut k = Key::from(base);
		k.push_int(i64::from(self.ante));
		k
	}

	#[inline]
	pub(crate) fn new_card(&mut self, kind: Kind) -> Card {
		self.next_sort_id += 1;
		Card::new(self.next_sort_id, kind)
	}

	/// A non-debuffed `j` in the joker area (`next(find_joker(name))`).
	#[inline]
	#[must_use]
	pub fn has_joker(&self, j: Joker) -> bool {
		self.jokers
			.iter()
			.any(|c| c.kind == Kind::Joker(j) && !c.debuff)
	}

	#[inline]
	#[must_use]
	pub fn count_joker(&self, j: Joker) -> usize {
		self.jokers
			.iter()
			.filter(|c| c.kind == Kind::Joker(j) && !c.debuff)
			.count()
	}

	#[inline]
	pub(crate) fn pareidolia(&self) -> bool {
		self.has_joker(Joker::Pareidolia)
	}

	#[inline]
	pub(crate) fn smeared(&self) -> bool {
		self.has_joker(Joker::Smeared)
	}

	#[inline]
	#[must_use]
	pub fn voucher_used(&self, v: Voucher) -> bool {
		self.used_vouchers >> (v as u32) & 1 == 1
	}

	#[inline]
	pub(crate) fn set_voucher_used(&mut self, v: Voucher) {
		self.used_vouchers |= 1 << (v as u32);
	}

	pub(crate) fn area(&self, a: Area) -> &Vec<Card> {
		match a {
			Area::Jokers => &self.jokers,
			Area::Consumables => &self.consumables,
			Area::Hand => &self.hand,
			Area::Deck => &self.deck,
			Area::Play => &self.play,
			Area::Discard => &self.discard,
			Area::ShopJokers => &self.shop_jokers,
			Area::ShopVouchers => &self.shop_vouchers,
			Area::ShopBoosters => &self.shop_boosters,
			Area::PackCards => &self.pack_cards,
		}
	}

	pub(crate) fn area_mut(&mut self, a: Area) -> &mut Vec<Card> {
		match a {
			Area::Jokers => &mut self.jokers,
			Area::Consumables => &mut self.consumables,
			Area::Hand => &mut self.hand,
			Area::Deck => &mut self.deck,
			Area::Play => &mut self.play,
			Area::Discard => &mut self.discard,
			Area::ShopJokers => &mut self.shop_jokers,
			Area::ShopVouchers => &mut self.shop_vouchers,
			Area::ShopBoosters => &mut self.shop_boosters,
			Area::PackCards => &mut self.pack_cards,
		}
	}

	/// Visible hands in `pairs(G.GAME.hands)` order.
	pub(crate) fn visible_hands_pairs(&self, exclude: Option<HandType>) -> ([HandType; 12], usize) {
		let mut out = [HandType::HighCard; 12];
		let mut n = 0;
		for h in HandType::PAIRS_ORDER {
			if self.hands[h as usize].visible && Some(h) != exclude {
				out[n] = h;
				n += 1;
			}
		}
		(out, n)
	}

	/// All playing cards (`G.playing_cards`) in their areas.
	pub(crate) fn playing_cards(&self) -> impl Iterator<Item = &Card> {
		self.deck
			.iter()
			.chain(self.hand.iter())
			.chain(self.play.iter())
			.chain(self.discard.iter())
	}

	pub(crate) fn playing_cards_mut(&mut self) -> impl Iterator<Item = &mut Card> {
		self.deck
			.iter_mut()
			.chain(self.hand.iter_mut())
			.chain(self.play.iter_mut())
			.chain(self.discard.iter_mut())
	}

	#[must_use]
	pub fn playing_card_count(&self) -> usize {
		self.deck.len() + self.hand.len() + self.play.len() + self.discard.len()
	}

	/// The `(rank, suit)` of the valid (non-stone) playing cards in `sort_id`
	/// order, for the `reset_*_card` samplers.
	pub(crate) fn non_stone_sorted(&self) -> Vec<(Rank, Suit)> {
		let mut cards: Vec<&Card> = self.playing_cards().filter(|c| !c.is_stone()).collect();
		cards.sort_unstable_by_key(|c| c.id);
		cards.into_iter().map(|c| (c.rank, c.suit)).collect()
	}
}
