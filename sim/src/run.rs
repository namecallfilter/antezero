//! `Game:start_run` / `init_game_object` / `Back:apply_to_run`.

use std::error::Error;
use std::fmt;
use std::ops::RangeInclusive;

use crate::blind::{BlindScaling, CardDebuff};
use crate::card::Kind;
use crate::data::{
	Back, Enhancement, HandType, P_CARDS_SORTED, Rank, Spectral, Suit, Tarot, Voucher,
};
use crate::enum_sets::{CenterSet, PlanetSet, TagSet, VoucherSet};
use crate::event::Event;
use crate::event_queue::EventQueue;
use crate::game::{
	Areas, Blind, BlindSelect, BlindSlot, Flow, Game, HandLevel, HandLevels, Limit, PackState,
	PerBlind, Round, Shop, State, Tags,
};
use crate::pools::{NewCard, OwnedArea, VoucherRoll};
use crate::rng::{Key, PseudoRandom};
use crate::round::JokerTargets;
use crate::shop::VoucherSource;
use crate::{blind, card_lifecycle, event_queue, flow, num, pools, round_end, shop};

/// Longest supported seed (the game's own seeds have 8 characters; stream
/// keys are the stream name followed by the seed).
pub const MAX_SEED_LEN: usize = 32;

/// The numbers of the stakes, from White (1) to Gold (8).
pub const STAKES: RangeInclusive<u8> = Stake::White as u8..=Stake::Gold as u8;

/// A stake. Each has the rules of the ones below it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Stake {
	White = 1,
	Red = 2,
	Green = 3,
	Black = 4,
	Blue = 5,
	Purple = 6,
	Orange = 7,
	Gold = 8,
}

impl Stake {
	/// Every stake, from White to Gold.
	pub const ALL: [Stake; 8] = [
		Stake::White,
		Stake::Red,
		Stake::Green,
		Stake::Black,
		Stake::Blue,
		Stake::Purple,
		Stake::Orange,
		Stake::Gold,
	];
}

impl From<Stake> for u8 {
	/// The stake's number, from 1 (White) to 8 (Gold).
	fn from(stake: Stake) -> u8 {
		stake as u8
	}
}

impl TryFrom<u8> for Stake {
	type Error = InvalidStake;

	/// The stake numbered `n`.
	///
	/// # Errors
	///
	/// Returns [`InvalidStake`] if `n` is not in [`STAKES`].
	fn try_from(n: u8) -> Result<Stake, InvalidStake> {
		Stake::ALL
			.into_iter()
			.find(|&s| u8::from(s) == n)
			.ok_or(InvalidStake(n))
	}
}

/// A stake number outside [`STAKES`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidStake(pub u8);

impl fmt::Display for InvalidStake {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "stake {} is not in {STAKES:?}", self.0)
	}
}

impl Error for InvalidStake {}

/// See [`Game::new_run`].
pub(crate) fn new_run(seed: &str, back: Back, stake: Stake) -> Game {
	assert!(
		seed.len() <= MAX_SEED_LEN,
		"seed is {} bytes, longer than {MAX_SEED_LEN}",
		seed.len()
	);
	let mut g = blank(seed, back, stake);
	event_queue::begin_action(&mut g.rng, &mut g.events);
	start_run(&mut g);
	event_queue::run_queue(&mut g);
	g
}

/// The poker hands at level 1 (`G.GAME.hands` of a new run).
fn starting_hands() -> HandLevels {
	HandLevels(HandType::ALL.map(|h| {
		let base = h.base();
		HandLevel {
			level: 1.0,
			chips: base.chips,
			mult: base.mult,
			played: 0.0,
			played_this_round: 0.0,
			visible: base.visible,
		}
	}))
}

/// The stake's and the deck's changes to the rules.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Modifiers {
	pub no_blind_reward: PerBlind<bool>,
	pub scaling: BlindScaling,
	pub enable_eternals_in_shop: bool,
	pub enable_perishables_in_shop: bool,
	pub enable_rentals_in_shop: bool,
	pub no_interest: bool,
	pub money_per_hand: Option<f64>,
	pub money_per_discard: Option<f64>,
}

/// What the deck starts the run with (`G.GAME.starting_params`).
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

/// What the run was started with (deck, stake and the stake's modifiers)
/// and the switches the caller sets.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Setup {
	pub back: Back,
	pub stake: Stake,
	pub modifiers: Modifiers,
	pub starting_params: StartingParams,
	pub win_ante: i32,
	/// Keep playing after winning ("Endless mode"); otherwise the run is
	/// terminal once the winning round is evaluated.
	pub endless: bool,
	/// Round ×Mult like Steamodded, which the real game runs under the bridge
	/// (see `scoring::apply_x_mult`); off, like vanilla and the oracle.
	pub steamodded: bool,
}

impl Setup {
	/// The rules of a run on `back` at `stake`: the stake's modifiers
	/// (`Game:start_run`) and the deck's (`Back:apply_to_run`).
	pub(crate) fn new(back: Back, stake: Stake) -> Setup {
		let mut modifiers = Modifiers::default();
		let mut params = StartingParams::default();
		if stake >= Stake::Red {
			modifiers.no_blind_reward[BlindSlot::Small] = true;
		}
		if stake >= Stake::Green {
			modifiers.scaling = BlindScaling::Faster;
		}
		if stake >= Stake::Black {
			modifiers.enable_eternals_in_shop = true;
		}
		if stake >= Stake::Blue {
			params.discards -= 1.0;
		}
		if stake >= Stake::Purple {
			modifiers.scaling = BlindScaling::Fastest;
		}
		if stake >= Stake::Orange {
			modifiers.enable_perishables_in_shop = true;
		}
		if stake >= Stake::Gold {
			modifiers.enable_rentals_in_shop = true;
		}
		match back {
			Back::Red => params.discards += 1.0,
			Back::Blue => params.hands += 1.0,
			Back::Yellow => params.dollars += 10.0,
			Back::Green => {
				modifiers.no_interest = true;
				modifiers.money_per_hand = Some(2.0);
				modifiers.money_per_discard = Some(1.0);
			}
			Back::Black => {
				params.hands -= 1.0;
				params.joker_slots += 1.0;
			}
			Back::Nebula => params.consumable_slots -= 1.0,
			Back::Abandoned => params.no_faces = true,
			Back::Painted => {
				params.joker_slots -= 1.0;
				params.hand_size += 2.0;
			}
			Back::Plasma => params.ante_scaling = 2.0,
			Back::Erratic => params.erratic_suits_and_ranks = true,
			Back::Magic
			| Back::Ghost
			| Back::Checkered
			| Back::Zodiac
			| Back::Anaglyph
			| Back::Challenge => {}
		}
		Setup {
			back,
			stake,
			modifiers,
			starting_params: params,
			win_ante: 8,
			endless: false,
			steamodded: false,
		}
	}
}

/// The run's progress and economy (`G.GAME`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Run {
	pub ante: i32,
	pub round: i32,
	pub dollars: f64,
	/// Dollars a joker has earned but not paid yet (`G.GAME.dollar_buffer`).
	pub dollar_buffer: f64,
	pub interest_cap: f64,
	pub interest_amount: f64,
	pub bankrupt_at: f64,
	pub rental_rate: f64,
	pub perishable_rounds: f64,
	pub ecto_minus: f64,
	pub probabilities: f64,
	pub hands_played: f64,
	pub unused_discards: f64,
	pub skips: f64,
	pub starting_deck_size: f64,
	/// Blinds defeated or skipped so far (progress measure for agents).
	pub blinds_cleared: u32,
	/// `G.GAME.won`: set at the end of the final boss's round, whether it was
	/// won or lost; [`Game::run_won`] tells whether the run was won.
	pub won: bool,
	/// `G.GAME.win_notified`: the run was won.
	pub(crate) win_notified: bool,
	pub last_hand_score: f64,
	pub last_hand_chips: f64,
	pub last_hand_mult: f64,
	pub earned: f64,
}

impl Default for Run {
	fn default() -> Run {
		Run {
			ante: 1,
			round: 0,
			dollars: 0.0,
			dollar_buffer: 0.0,
			interest_cap: 25.0,
			interest_amount: 1.0,
			bankrupt_at: 0.0,
			rental_rate: 3.0,
			perishable_rounds: 5.0,
			ecto_minus: 1.0,
			probabilities: 1.0,
			hands_played: 0.0,
			unused_discards: 0.0,
			skips: 0.0,
			starting_deck_size: 52.0,
			blinds_cleared: 0,
			won: false,
			win_notified: false,
			last_hand_score: 0.0,
			last_hand_chips: 0.0,
			last_hand_mult: 0.0,
			earned: 0.0,
		}
	}
}

impl Run {
	/// `base .. ante`.
	#[inline]
	pub(crate) fn key_ante(&self, base: &str) -> Key {
		let mut k = Key::from(base);
		k.push_int(i64::from(self.ante));
		k
	}

	/// `base .. append .. ante`.
	#[inline]
	pub(crate) fn key_append_ante(&self, base: &str, append: &str) -> Key {
		let mut k = Key::from(base);
		k.push(append).push_int(i64::from(self.ante));
		k
	}
}

/// What the run has used, for the pools and the jokers that count it, and
/// what the player's profile keeps out of the pools.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct History {
	/// The cards the profile has not unlocked: in no pool, but for legendary
	/// jokers (`get_current_pool`).
	pub locked: CenterSet,
	/// The skip tags whose prerequisite the profile has not discovered.
	pub locked_tags: TagSet,
	pub used_jokers: CenterSet,
	pub used_vouchers: VoucherSet,
	pub planets_used: PlanetSet,
	pub gros_michel_extinct: bool,
	pub consumable_usage_tarot: f64,
	pub consumable_usage_planet: f64,
	pub consumable_usage_spectral: f64,
	pub last_tarot_planet: Option<Kind>,
	pub last_hand_played: Option<HandType>,
}

fn blank(seed: &str, back: Back, stake: Stake) -> Game {
	Game {
		rng: PseudoRandom::new(seed),
		events: EventQueue::new(),
		setup: Setup::new(back, stake),
		flow: Flow::default(),
		areas: Areas::default(),
		run: Run::default(),
		history: History::default(),
		round: Round::default(),
		blind_select: BlindSelect::default(),
		blind: Blind::default(),
		shop: Shop::default(),
		pack: PackState::default(),
		hands: starting_hands(),
		tags: Tags::default(),
	}
}

fn start_run(g: &mut Game) {
	apply_back(
		g.setup.back,
		&mut g.history,
		&mut g.round,
		&mut g.shop,
		&mut g.events,
	);

	let sp = g.setup.starting_params;
	g.round.base_hands = sp.hands;
	g.round.base_discards = sp.discards;
	g.run.dollars = sp.dollars;
	g.shop.base_reroll_cost = sp.reroll_cost;
	g.shop.reroll_cost = sp.reroll_cost;

	g.blind_select.boss = pools::get_new_boss(
		g.setup.win_ante,
		g.run.ante,
		&mut g.blind_select,
		&mut g.rng,
	);
	g.shop.voucher = Some(pools::get_next_voucher(
		&g.areas,
		&g.run,
		&g.history,
		&mut g.rng,
		VoucherRoll::Ante,
	));
	g.blind_select.skip_tags.small = Some(pools::get_next_tag(&g.run, &g.history, &mut g.rng));
	g.blind_select.skip_tags.big = Some(pools::get_next_tag(&g.run, &g.history, &mut g.rng));

	g.areas.limits.consumables = Limit::new(sp.consumable_slots);
	g.areas.limits.jokers = Limit::new(sp.joker_slots);
	g.areas.hand.limit = Limit::new(sp.hand_size);

	// Deck: card prototypes sorted by `P_CARDS` key ("C_2" < ... < "S_T").
	let mut protos: Vec<(Suit, Rank)> = Vec::with_capacity(P_CARDS_SORTED.len());
	for &(s, r) in &P_CARDS_SORTED {
		let (s, r) = if sp.erratic_suits_and_ranks {
			pools::random_front(&mut g.rng, &Key::from("erratic"))
		} else {
			(s, r)
		};
		if sp.no_faces && r.is_face() {
			continue;
		}
		protos.push((s, r));
	}
	protos.sort_by_key(|&(s, r)| (s.letter(), r.letter()));
	for (s, r) in protos {
		let mut c = card_lifecycle::make_playing_card(
			&mut g.areas,
			g.run.hands_played,
			g.shop.discount_percent,
			s,
			r,
			Enhancement::Base,
		);
		c.core.face_down = true;
		g.areas.deck.insert(0, c);
	}
	g.run.starting_deck_size = num::float(g.areas.playing_card_count());
	g.events.delay(0.5);

	g.round.discards_left = g.round.base_discards;
	g.round.hands_left = g.round.base_hands;
	let k = Key::from("shuffle");
	flow::shuffle_area(&mut g.areas.deck, &mut g.rng, &k);
	g.round
		.reset_joker_targets(&g.areas, &g.run, &mut g.rng, JokerTargets::RunStart);
	blind::set_blind(g, None);
	round_end::reset_blinds(
		g.setup.win_ante,
		g.run.ante,
		&mut g.blind_select,
		&mut g.rng,
	);
	// unlock_notify
	g.events.push(Event::Wait);

	g.flow.state = State::BlindSelect;
	g.flow.state_complete = false;
}

/// A voucher the deck starts with.
fn deck_voucher(history: &mut History, round: &mut Round, events: &mut EventQueue, v: Voucher) {
	history.used_vouchers.insert(v);
	shop::apply_voucher(round, events, v, VoucherSource::Deck);
}

/// What `Back:apply_to_run()` does to the run beyond its rules (see
/// [`Setup::new`]): the deck's vouchers and cards.
fn apply_back(
	back: Back, history: &mut History, round: &mut Round, shop: &mut Shop, events: &mut EventQueue,
) {
	let deck_card =
		|kind: Kind| Event::Create(NewCard::forced(kind, OwnedArea::Consumables, "deck"));
	match back {
		Back::Magic => {
			deck_voucher(history, round, events, Voucher::CrystalBall);
			for _ in 0..2 {
				events.push(deck_card(Kind::Tarot(Tarot::Fool)));
			}
		}
		Back::Nebula => deck_voucher(history, round, events, Voucher::Telescope),
		Back::Ghost => {
			events.push(deck_card(Kind::Spectral(Spectral::Hex)));
			shop.spectral_rate = 2.0;
		}
		Back::Checkered => events.push(Event::CheckeredDeck),
		Back::Zodiac => {
			for v in [
				Voucher::TarotMerchant,
				Voucher::PlanetMerchant,
				Voucher::OverstockNorm,
			] {
				deck_voucher(history, round, events, v);
			}
		}
		Back::Red
		| Back::Blue
		| Back::Yellow
		| Back::Green
		| Back::Black
		| Back::Abandoned
		| Back::Painted
		| Back::Plasma
		| Back::Erratic
		| Back::Anaglyph
		| Back::Challenge => {}
	}
}

/// Checkered Deck: Clubs become Spades and Diamonds become Hearts.
pub(crate) fn checkered_deck(areas: &mut Areas, blind: &Blind) {
	let debuff = CardDebuff::new(blind, &areas.jokers);
	for c in &mut areas.deck {
		let to = match c.suit {
			Suit::Clubs => Some(Suit::Spades),
			Suit::Diamonds => Some(Suit::Hearts),
			_ => None,
		};
		if let Some(s) = to {
			c.set_base(s, c.rank);
			debuff.apply(c);
		}
	}
}
