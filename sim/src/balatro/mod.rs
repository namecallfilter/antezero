//! A native implementation of `RLBalatro`'s lean-actions environment: 152
//! discrete actions, 32 of them selection verbs that take cards, plays that
//! declare their poker hand, steps without a decision taken by the
//! environment itself, episodes that start from recorded games, and its
//! observation of 8281 floats.

mod hand_eval;
mod ids;
mod obs;
mod script;

use std::fmt::Write as _;
use std::sync::Arc;

use self::hand_eval::{CoreCtx, HS_ANY, MAX_HAND, N_HANDS, core_ctx, core_rule, sel_ctx, sel_eval};
pub use self::hand_eval::{HS_CTX, straight_lut};
pub use self::ids::{jokers_by_key, vouchers_by_key};
use self::obs::{
	MAX_BOOST, MAX_CONS, MAX_JOKERS, MAX_PACK, MAX_SHOP, MAX_VOUCH, write_hand_ctx, write_obs,
	write_obs_selection,
};
pub use self::obs::{OBS_GROUPS, OBS_SIZE, SEL_F};
use self::script::{Header, Step, skip_refused};
pub use self::script::{ScriptError, StartPoint, load_starts};
use crate::actions::Action;
use crate::card::{AnyCard, Card, CardLike, CardMask, Kind, PlayingCard};
use crate::data::{Back, Edition, Seal, Spectral, Tarot};
use crate::game::{Area, BlindSlot, Game, Phase, State};
use crate::{SEED_CHARS, Stake, consumables, num};

/// The number of actions: the length of the mask that
/// [`BalEnv::write_all_obs`] writes.
pub const N_ACTIONS: usize = 152;
/// The number of selection verbs, the actions that take cards (see
/// [`sel_verb_index`]).
pub const N_SEL_VERBS: usize = 32;
/// The most cards an action takes.
pub const MAX_PICKS: usize = 5;

/// Play the picked hand cards, whatever hand they make.
pub const ACT_PLAY: usize = 0;
/// Discard the picked hand cards.
pub const ACT_DISCARD: usize = 1;
/// Play the blind on offer.
pub const ACT_SELECT_BLIND: usize = 2;
/// Skip the blind on offer for its tag.
pub const ACT_SKIP_BLIND: usize = 3;
/// Reroll the boss blind.
pub const ACT_REROLL_BOSS: usize = 4;
/// Leave the round evaluation with its payout. The environment takes this
/// action itself, so it is never left to the caller.
pub const ACT_CASH_OUT: usize = 5;
/// Leave the shop.
pub const ACT_END_SHOP: usize = 6;
/// Reroll the shop.
pub const ACT_REROLL: usize = 7;
/// `ACT_BUY + i` buys shop card `i`.
pub const ACT_BUY: usize = 8;
/// `ACT_BUY_USE + i` buys shop card `i`, a consumable, and uses it at once.
pub const ACT_BUY_USE: usize = ACT_BUY + MAX_SHOP;
/// `ACT_REDEEM + i` redeems shop voucher `i`.
pub const ACT_REDEEM: usize = ACT_BUY_USE + MAX_SHOP;
/// `ACT_OPEN + i` opens shop booster pack `i`.
pub const ACT_OPEN: usize = ACT_REDEEM + MAX_VOUCH;
/// `ACT_PICK + i` takes card `i` of the open pack; a card that acts on hand
/// cards takes the picked ones.
pub const ACT_PICK: usize = ACT_OPEN + MAX_BOOST;
/// Skip the rest of the open pack.
pub const ACT_SKIP_PACK: usize = ACT_PICK + MAX_PACK;
/// `ACT_SELL_JOKER + i` sells joker `i`.
pub const ACT_SELL_JOKER: usize = ACT_SKIP_PACK + 1;
/// `ACT_SELL_CONS + i` sells consumable `i`.
pub const ACT_SELL_CONS: usize = ACT_SELL_JOKER + MAX_JOKERS;
/// `ACT_USE_CONS + i` uses consumable `i`; one that acts on hand cards takes
/// the picked ones.
pub const ACT_USE_CONS: usize = ACT_SELL_CONS + MAX_CONS;
/// 19 ids that are never legal. In `RLBalatro` they swap joker `i` with its
/// right neighbour, but only in its mode with one action per card toggle;
/// its lean-actions mode, the one implemented here, orders jokers with
/// [`ACT_REORDER_JOKERS`]. The ids stay so that the later ones keep their
/// numbers, which trained checkpoints depend on.
pub const ACT_JOKER_RIGHT: usize = ACT_USE_CONS + MAX_CONS;
/// 20 ids that are never legal, here as in `RLBalatro`, which names them for
/// moving joker `i` to the front. Kept like [`ACT_JOKER_RIGHT`].
pub const ACT_JOKER_FRONT: usize = ACT_JOKER_RIGHT + MAX_JOKERS - 1;
/// 20 ids that are never legal, here as in `RLBalatro`, which names them for
/// moving joker `i` to the back. Kept like [`ACT_JOKER_RIGHT`].
pub const ACT_JOKER_BACK: usize = ACT_JOKER_FRONT + MAX_JOKERS;
/// Move the picked jokers to the front, in pick order. Legal with two jokers
/// or more, once until another action changes the game.
pub const ACT_REORDER_JOKERS: usize = ACT_JOKER_BACK + MAX_JOKERS;
/// `ACT_PLAY_HAND + h` plays the picked hand cards as poker hand `h`, counted
/// in `HandType::ALL` order from Flush Five to Pair (High Card has no id:
/// [`ACT_PLAY`] plays any cards). A card can be picked only while the picks
/// can still be completed to that hand, and the picks can stop once they
/// hold it; cards picked after that are free.
pub const ACT_PLAY_HAND: usize = ACT_REORDER_JOKERS + 1;
const _: () = assert!(ACT_PLAY_HAND + (N_HANDS - 1) == N_ACTIONS);

const SEL_VERB_PLAY: usize = 0;
const SEL_VERB_DISCARD: usize = 1;
const SEL_VERB_USE_CONS: usize = 2;
const SEL_VERB_PICK: usize = SEL_VERB_USE_CONS + MAX_CONS;
const SEL_VERB_REORDER_JOKERS: usize = SEL_VERB_PICK + MAX_PACK;
const SEL_VERB_PLAY_HAND: usize = SEL_VERB_REORDER_JOKERS + 1;
const _: () = assert!(SEL_VERB_PLAY_HAND + (N_HANDS - 1) == N_SEL_VERBS);

fn seed_from_index(i: u64) -> String {
	let mut x = i
		.wrapping_mul(0x9e37_79b9_7f4a_7c15)
		.wrapping_add(0x632b_e59b_d9b4_e019);
	let mut s = String::with_capacity(8);
	for _ in 0..8 {
		x ^= x >> 29;
		x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
		s.push(char::from(SEED_CHARS[num::pick(x >> 32, SEED_CHARS.len())]));
	}
	s
}

/// The number of action `a` among the selection verbs, the actions that take
/// cards: [`ACT_PLAY`], [`ACT_DISCARD`], the ten [`ACT_USE_CONS`] and the
/// eight [`ACT_PICK`] actions, [`ACT_REORDER_JOKERS`] and the eleven
/// [`ACT_PLAY_HAND`] actions, in this order. `None` for any other action.
#[inline]
#[must_use]
pub fn sel_verb_index(a: usize) -> Option<usize> {
	if a == ACT_PLAY {
		Some(SEL_VERB_PLAY)
	} else if a == ACT_DISCARD {
		Some(SEL_VERB_DISCARD)
	} else if (ACT_USE_CONS..ACT_USE_CONS + MAX_CONS).contains(&a) {
		Some(SEL_VERB_USE_CONS + (a - ACT_USE_CONS))
	} else if (ACT_PICK..ACT_PICK + MAX_PACK).contains(&a) {
		Some(SEL_VERB_PICK + (a - ACT_PICK))
	} else if a == ACT_REORDER_JOKERS {
		Some(SEL_VERB_REORDER_JOKERS)
	} else if (ACT_PLAY_HAND..ACT_PLAY_HAND + N_HANDS - 1).contains(&a) {
		Some(SEL_VERB_PLAY_HAND + (a - ACT_PLAY_HAND))
	} else {
		None
	}
}

#[inline]
fn sel_verb_action(v: usize) -> usize {
	if v == SEL_VERB_PLAY {
		ACT_PLAY
	} else if v == SEL_VERB_DISCARD {
		ACT_DISCARD
	} else if (SEL_VERB_USE_CONS..SEL_VERB_USE_CONS + MAX_CONS).contains(&v) {
		ACT_USE_CONS + (v - SEL_VERB_USE_CONS)
	} else if (SEL_VERB_PICK..SEL_VERB_PICK + MAX_PACK).contains(&v) {
		ACT_PICK + (v - SEL_VERB_PICK)
	} else if v == SEL_VERB_REORDER_JOKERS {
		ACT_REORDER_JOKERS
	} else {
		ACT_PLAY_HAND + (v - SEL_VERB_PLAY_HAND)
	}
}

/// The hand that selection verb `v` declares, if it is a play.
#[inline]
fn hand_verb_target(v: usize) -> Option<usize> {
	if v == SEL_VERB_PLAY {
		Some(HS_ANY)
	} else if (SEL_VERB_PLAY_HAND..SEL_VERB_PLAY_HAND + N_HANDS - 1).contains(&v) {
		Some(v - SEL_VERB_PLAY_HAND)
	} else {
		None
	}
}

/// The options of a [`BalEnv`].
#[derive(Clone, Debug)]
pub struct BalConfig {
	/// The deck of a new run.
	pub back: Back,
	/// The stake of a new run.
	pub stake: Stake,
	/// The number of steps after which an episode is truncated.
	pub max_steps: i32,
	/// The reward for a blind beaten. Not paid with `reward_money`.
	pub reward_blind: f32,
	/// The reward for winning the run.
	pub reward_win: f32,
	/// Pay `log10` of the chips scored in the winning round in place of
	/// `reward_win`.
	pub reward_win_log: bool,
	/// Above 8, a won run goes on in Endless Mode and the episode ends when
	/// the boss of this ante is beaten, which pays the win reward again.
	/// Otherwise the episode ends with the win.
	pub endless_ante: i32,
	/// Added to the reward of a lost run.
	pub reward_lose: f32,
	/// On a loss, also pay the share of the failed blind's chips that was
	/// scored.
	pub reward_lose_partial: bool,
	/// Pay the dollars a step earns, divided by `money_per_reward`, in place
	/// of `reward_blind`. The partial credit of a loss is then multiplied by
	/// the failed blind's payout over `money_per_reward`.
	pub reward_money: bool,
	/// The dollars that pay a reward of 1 with `reward_money`. The rewards
	/// are only finite if it is positive and finite.
	pub money_per_reward: f32,
	/// Pay this times the share of the blind's chips that a played hand
	/// scores, unless the hand beats the blind.
	pub reward_chips: f32,
	/// Paid for each Blue Seal the deck gains in a step, and taken back for
	/// each one it loses.
	pub reward_blue_seal: f32,
	/// Pay the reward of a skipped blind when the next blind is beaten.
	pub reward_skip_deferred: bool,
	/// Subtracted from the reward of every step.
	pub step_cost: f32,
	/// Subtracted from the reward of a joker reorder.
	pub toggle_cost: f32,
	/// Let a pack be skipped where the game refuses the skip (an Arcana or
	/// Spectral pack with an empty hand), which can leave no action at all:
	/// a softlock in the real game. Without it, a run that gets stuck there
	/// ends as a loss.
	pub disable_softlocks: bool,
	/// Show nothing of the cards that lie face down in the hand or among the
	/// jokers but that they are there, and treat such hand cards as having
	/// no rank and no suit in the rule for declared hands. The deck rows of
	/// the observation then hold the draw pile together with the face-down
	/// hand cards and the cards discarded face down.
	///
	/// A face-down joker still takes effect, and these effects stay visible:
	///
	/// - The suits of a Smeared Joker and the face cards of a Pareidolia in
	///   the card rows, unless the joker is debuffed. The game's deck view
	///   counts them.
	/// - Four Fingers, Shortcut and Smeared Joker in the rule for declared
	///   hands and in the hand context, from which the pick masks are
	///   computed. The game names the hand of the selected cards with them,
	///   so it only tells them when a selection differs by them; the hand
	///   context always does.
	/// - The numbers of the run that a joker changed when it was bought. The
	///   observation copies them from the game: the hand size, the hands,
	///   the discards and the joker slots, which the game shows during a
	///   round, and the odds, the debt limit, the interest, the free rerolls,
	///   the reroll cost and the price of a held Planet under Astronomer,
	///   which it does not.
	/// - What a face-up joker's own row counts of the other jokers (Joker
	///   Stencil, Swashbuckler), as on its card in the game.
	/// - In the mask, what the game's buttons tell: whether a face-down
	///   joker can be sold, whether a face-down hand card can take Aura,
	///   whether a joker is left without an edition (The Wheel of Fortune,
	///   Ectoplasm, Hex) and whether a joker slot is free.
	///
	/// A face-down hand card still sorts into the hand by its rank or suit,
	/// as in the game.
	pub hide_face_down: bool,
	/// Keep the script of the episode, see [`BalEnv::script`].
	pub record: bool,
}

impl Default for BalConfig {
	fn default() -> Self {
		BalConfig {
			back: Back::Red,
			stake: Stake::White,
			max_steps: 5000,
			reward_blind: 1.0,
			reward_win: 10.0,
			reward_win_log: false,
			endless_ante: 0,
			reward_lose: 0.0,
			reward_lose_partial: true,
			reward_money: false,
			money_per_reward: 15.0,
			reward_chips: 0.0,
			reward_blue_seal: 0.0,
			reward_skip_deferred: false,
			step_cost: 0.0,
			toggle_cost: 0.0,
			disable_softlocks: false,
			hide_face_down: false,
			record: false,
		}
	}
}

/// What a step did, see [`BalEnv::step_select`]. The flags are 0 or 1.
#[derive(Clone, Copy, Debug, Default)]
pub struct BalStepInfo {
	/// The reward of the step, as the [`BalConfig`] rewards add up.
	pub reward: f32,
	/// The episode is over: the run was won, lost or got stuck.
	pub terminated: u8,
	/// The episode reached [`BalConfig::max_steps`] without ending.
	pub truncated: u8,
	/// The run was won (on a terminal step).
	pub won: u8,
	/// The action or its cards were not allowed, and the game did not
	/// change.
	pub illegal: u8,
	/// The ante after the step.
	pub ante: i32,
	/// The round after the step.
	pub round: i32,
	/// The money after the step, in whole dollars. Negative in debt.
	pub dollars: i32,
	/// The steps of the episode so far, illegal ones included.
	pub steps: i32,
	/// On a terminal step, the chips scored in the run's last round (in the
	/// round that won it, if the run was then lost in Endless Mode); 0 on
	/// any other step.
	pub score: f64,
	/// The dollars the step earned. Spending does not count, and a sale
	/// counts only what the card's sell value had gained.
	pub earned: f32,
	/// The blinds the step beat.
	pub blinds: i32,
	/// The pack skips of the step that only
	/// [`BalConfig::disable_softlocks`] allowed.
	pub softlocks_avoided: i32,
	/// The run ended because no action was left, or none that leaves a pack.
	pub dead_end: i32,
	/// The poker hand the step played, as the game named it (its position in
	/// `HandType::ALL`); `-1` if it played none.
	pub hand_played: i32,
	/// The Blue Seals in the deck after the step.
	pub blue_seals: i32,
	/// The number of cards the step played; 0 if it played no hand, like the
	/// three counts below.
	pub played_cards: i32,
	/// How many of the played cards score.
	pub played_scoring: i32,
	/// The Blue Seal cards that were in the hand before the play.
	pub blue_held: i32,
	/// How many of those were played.
	pub blue_played: i32,
	/// The hand a play declared (12 for [`ACT_PLAY`]); `-1` on a step that
	/// played none.
	pub hs_declared: i32,
	/// The score of the hand the step played; 0 if it played none.
	pub hand_score: f64,
}

/// One environment: a game, and the rewards and the episode around it.
pub struct BalEnv {
	cfg: BalConfig,
	g: Game,
	steps: i32,
	endless: bool,
	win_score: f64,
	joker_ordered_in_phase: bool,
	skips_pending: i32,
	blue_seals: i32,
	earned: f64,
	softlocks: i32,
	start: Option<Arc<StartPoint>>,
	start_reseed: bool,
	episode_start: i32,
	script: String,
	core_cache: CoreCtx,
}

/// A selection: its verb, and its cards in pick order after the forced ones.
type Selection<'a> = (usize, &'a [usize]);

fn needs_target(card: &Card) -> bool {
	card.kind.is_consumable() && consumables::targets(card.kind).is_some()
}

fn mask_of_slice(order: &[usize]) -> CardMask {
	let mut m = CardMask::EMPTY;
	for &i in order {
		m.insert(i);
	}
	m
}

fn simple_action(a: usize) -> Option<Action> {
	match a {
		ACT_SELECT_BLIND => Some(Action::SelectBlind),
		ACT_SKIP_BLIND => Some(Action::SkipBlind),
		ACT_REROLL_BOSS => Some(Action::RerollBoss),
		ACT_CASH_OUT => Some(Action::CashOut),
		ACT_END_SHOP => Some(Action::EndShop),
		ACT_REROLL => Some(Action::Reroll),
		ACT_SKIP_PACK => Some(Action::SkipPack),
		_ => None,
	}
}

fn decode_non_select_action(a: usize) -> Option<Action> {
	if let Some(act) = simple_action(a) {
		return Some(act);
	}
	if (ACT_BUY..ACT_BUY + MAX_SHOP).contains(&a) {
		Some(Action::BuyShopCard(num::byte(a - ACT_BUY)))
	} else if (ACT_BUY_USE..ACT_BUY_USE + MAX_SHOP).contains(&a) {
		Some(Action::BuyAndUse {
			index: num::byte(a - ACT_BUY_USE),
			targets: CardMask::EMPTY,
		})
	} else if (ACT_REDEEM..ACT_REDEEM + MAX_VOUCH).contains(&a) {
		Some(Action::RedeemVoucher(num::byte(a - ACT_REDEEM)))
	} else if (ACT_OPEN..ACT_OPEN + MAX_BOOST).contains(&a) {
		Some(Action::OpenBooster(num::byte(a - ACT_OPEN)))
	} else if (ACT_PICK..ACT_PICK + MAX_PACK).contains(&a) {
		Some(Action::PickPackCard {
			index: num::byte(a - ACT_PICK),
			targets: CardMask::EMPTY,
		})
	} else if (ACT_SELL_JOKER..ACT_SELL_JOKER + MAX_JOKERS).contains(&a) {
		Some(Action::SellJoker(num::byte(a - ACT_SELL_JOKER)))
	} else if (ACT_SELL_CONS..ACT_SELL_CONS + MAX_CONS).contains(&a) {
		Some(Action::SellConsumable(num::byte(a - ACT_SELL_CONS)))
	} else if (ACT_USE_CONS..ACT_USE_CONS + MAX_CONS).contains(&a) {
		Some(Action::UseConsumable {
			index: num::byte(a - ACT_USE_CONS),
			targets: CardMask::EMPTY,
		})
	} else {
		None
	}
}

impl BalEnv {
	/// An environment with the options `cfg`. Its game is a placeholder
	/// until the first [`reset`](Self::reset).
	#[must_use]
	pub fn new(cfg: BalConfig) -> BalEnv {
		let mut g = Game::new_run("11111111", cfg.back, cfg.stake);
		g.set_tracing(false);
		g.setup.endless = cfg.endless_ante > 8;
		let mut env = BalEnv {
			cfg,
			g,
			steps: 0,
			endless: false,
			win_score: 0.0,
			joker_ordered_in_phase: false,
			skips_pending: 0,
			blue_seals: 0,
			earned: 0.0,
			softlocks: 0,
			start: None,
			start_reseed: true,
			episode_start: -1,
			script: String::new(),
			core_cache: CoreCtx::default(),
		};
		env.refresh_core();
		env
	}

	/// [`reset`](Self::reset) with the seed that `RLBalatro` derives from
	/// `seed_index`: eight seed characters.
	pub fn reset_index(&mut self, seed_index: u64) {
		self.reset(&seed_from_index(seed_index));
	}

	/// Starts an episode: a new run of the configured deck and stake with
	/// the seed `seed`, or a copy of the start point's game (see
	/// [`set_start`](Self::set_start)). The environment then takes the steps
	/// that need no decision.
	///
	/// # Panics
	///
	/// Panics if the game takes `seed` and it is longer than
	/// [`MAX_SEED_LEN`](crate::MAX_SEED_LEN) bytes.
	pub fn reset(&mut self, seed: &str) {
		if let Some(ref sp) = self.start {
			self.g.clone_from(&sp.game);
			if self.start_reseed {
				self.g.rng.reseed(seed);
			}
			self.episode_start = sp.id;
		} else {
			self.g = Game::new_run(seed, self.cfg.back, self.cfg.stake);
			self.episode_start = -1;
		}
		self.g.set_tracing(false);
		self.g.setup.endless = self.cfg.endless_ante > 8;
		self.steps = 0;
		self.endless = self.start.is_some() && self.g.run_won() && self.cfg.endless_ante > 8;
		self.joker_ordered_in_phase = false;
		self.win_score = 0.0;
		self.blue_seals = self.count_blue_seals();
		self.skips_pending = 0;
		self.script.clear();
		if let Some(ref sp) = self.start {
			if self.cfg.record {
				self.script.clone_from(&sp.script);
			}
			if self.start_reseed {
				self.record(Step::Reseed(seed));
			}
		} else if self.cfg.record {
			let header = Header {
				seed,
				back: self.cfg.back,
				stake: self.cfg.stake,
			};
			writeln!(self.script, "{header}").expect("a String takes every write");
		}
		let mut unused = BalStepInfo::default();
		self.auto_advance(&mut unused);
		self.show_forced_cards();
		self.refresh_core();
	}

	fn refresh_core(&mut self) {
		self.core_cache = core_ctx(&self.g, self.cfg.hide_face_down);
	}

	/// The deck and the stake of the new runs that later resets start.
	pub fn set_deck(&mut self, back: Back, stake: Stake) {
		self.cfg.back = back;
		self.cfg.stake = stake;
	}

	/// Makes later resets start from a copy of `start`'s game, which keeps
	/// its own deck and stake; `None` goes back to new runs.
	///
	/// With `reseed`, a reset then restarts the game's random streams from
	/// its seed. Nothing already dealt is dealt again: a start point inside
	/// a round keeps its hand and the order of its draw pile, so the cards
	/// drawn until the round ends are the recorded ones under every seed.
	/// Without `reseed`, the episode replays the recorded run's randomness.
	pub fn set_start(&mut self, start: Option<Arc<StartPoint>>, reseed: bool) {
		self.start = start;
		self.start_reseed = reseed;
	}

	/// The [`StartPoint::id`] the episode started from, or `-1` for a new
	/// run.
	#[must_use]
	pub fn episode_start(&self) -> i32 {
		self.episode_start
	}

	/// The script of the episode so far if [`BalConfig::record`] is set, and
	/// the empty string otherwise: [`load_starts`] replays it to this game.
	#[must_use]
	pub fn script(&self) -> &str {
		&self.script
	}

	/// The game being played.
	#[must_use]
	pub fn game(&self) -> &Game {
		&self.g
	}

	fn count_blue_seals(&self) -> i32 {
		num::id(
			self.g
				.areas
				.playing_cards()
				.filter(|c| c.seal == Some(Seal::Blue))
				.count(),
		)
	}

	fn hands_available(&self) -> u32 {
		let mut av = 0u32;
		for h in 0..=HS_ANY {
			let (cm, stop) = core_rule(&self.core_cache, h, self.core_cache.forced);
			if cm != 0 || stop {
				av |= 1u32 << h;
			}
		}
		av
	}

	fn mask(&self, m: &mut [u8]) {
		m[..N_ACTIONS].fill(0);
		if self.is_env_terminal() {
			return;
		}
		let st = self.g.flow.state;
		let nh = self.g.areas.hand.cards.len().min(MAX_HAND);
		let hand_state = st == State::SelectingHand || st.is_pack();
		if st == State::SelectingHand {
			let av = if nh > 0 { self.hands_available() } else { 0 };
			m[ACT_PLAY] = u8::from((av >> HS_ANY) & 1 != 0);
			for h in 0..N_HANDS - 1 {
				m[ACT_PLAY_HAND + h] = u8::from((av >> h) & 1 != 0);
			}
			m[ACT_DISCARD] = u8::from(nh > 0 && self.g.round.discards_left > 0.0);
		}
		if st == State::BlindSelect {
			m[ACT_SELECT_BLIND] = 1;
			let can_skip = matches!(
				self.g.blind_select.on_deck,
				Some(BlindSlot::Small | BlindSlot::Big)
			) && self
				.g
				.blind_select
				.on_deck
				.and_then(|s| self.g.blind_select.skip_tags.of(s))
				.is_some();
			m[ACT_SKIP_BLIND] = u8::from(can_skip);
			m[ACT_REROLL_BOSS] = u8::from(self.g.check(Action::RerollBoss).is_ok());
		}
		if st == State::RoundEval {
			m[ACT_CASH_OUT] = u8::from(self.g.check(Action::CashOut).is_ok());
		}
		if st == State::Shop {
			self.mask_shop(m);
		}
		if st.is_pack() {
			self.mask_pack(st, m);
		}
		let nj = self.g.areas.jokers.len().min(MAX_JOKERS);
		for i in 0..nj {
			m[ACT_SELL_JOKER + i] = u8::from(self.g.check(Action::SellJoker(num::byte(i))).is_ok());
		}
		let can_reorder_state =
			matches!(st, State::SelectingHand | State::Shop | State::BlindSelect) || st.is_pack();
		m[ACT_REORDER_JOKERS] =
			u8::from(!self.joker_ordered_in_phase && nj >= 2 && can_reorder_state);
		for (i, c) in self.g.areas.consumables.iter().take(MAX_CONS).enumerate() {
			m[ACT_SELL_CONS + i] =
				u8::from(self.g.check(Action::SellConsumable(num::byte(i))).is_ok());
			let usable = if hand_state && needs_target(c) {
				self.target_feasible(c)
			} else {
				self.g
					.check(Action::UseConsumable {
						index: num::byte(i),
						targets: CardMask::EMPTY,
					})
					.is_ok()
			};
			m[ACT_USE_CONS + i] = u8::from(usable);
		}
	}

	fn mask_shop(&self, m: &mut [u8]) {
		m[ACT_END_SHOP] = 1;
		m[ACT_REROLL] = u8::from(self.g.check(Action::Reroll).is_ok());
		for i in 0..self.g.areas.shop_jokers.len().min(MAX_SHOP) {
			let idx = num::byte(i);
			m[ACT_BUY + i] = u8::from(self.g.check(Action::BuyShopCard(idx)).is_ok());
			m[ACT_BUY_USE + i] = u8::from(
				self.g
					.check(Action::BuyAndUse {
						index: idx,
						targets: CardMask::EMPTY,
					})
					.is_ok(),
			);
		}
		for i in 0..self.g.areas.shop_vouchers.len().min(MAX_VOUCH) {
			m[ACT_REDEEM + i] = u8::from(self.g.check(Action::RedeemVoucher(num::byte(i))).is_ok());
		}
		for i in 0..self.g.areas.shop_boosters.len().min(MAX_BOOST) {
			m[ACT_OPEN + i] = u8::from(self.g.check(Action::OpenBooster(num::byte(i))).is_ok());
		}
	}

	fn mask_pack(&self, st: State, m: &mut [u8]) {
		for (i, c) in self.g.areas.pack_cards.iter().take(MAX_PACK).enumerate() {
			let ok = match c {
				AnyCard::Card(cc) if needs_target(cc) => self.target_feasible(cc),
				AnyCard::Card(cc) if cc.kind.is_consumable() => {
					consumables::can_use_consumable(
						st,
						&self.g.areas,
						&self.g.history,
						cc.kind,
						Area::PackCards,
						CardMask::EMPTY,
					) && (cc.kind != Kind::Spectral(Spectral::Ankh) || self.g.areas.joker_room())
				}
				AnyCard::Card(Card {
					kind: Kind::Joker(_),
					core,
					..
				}) => core.edition == Some(Edition::Negative) || self.g.areas.joker_room(),
				AnyCard::Card(_) | AnyCard::Playing(_) => true,
			};
			m[ACT_PICK + i] = u8::from(ok);
		}
		let can_skip = self.g.check(Action::SkipPack).is_ok()
			|| (self.cfg.disable_softlocks && skip_refused(&self.g));
		m[ACT_SKIP_PACK] = u8::from(can_skip);
	}

	fn target_ok(&self, card: &Card, hand_idx: usize) -> bool {
		card.kind != Kind::Spectral(Spectral::Aura)
			|| self.g.areas.hand.cards[hand_idx].core.edition.is_none()
	}

	fn target_feasible(&self, card: &Card) -> bool {
		let Some(hl) = consumables::targets(card.kind) else {
			return false;
		};
		let nh = self.g.areas.hand.cards.len().min(MAX_HAND);
		let (lo, hi) = (hl.min, hl.max);
		let (mut ok, mut forced) = (0usize, 0usize);
		for i in 0..nh {
			let f = self.g.areas.hand.cards[i].forced.is_locked();
			if f && !self.target_ok(card, i) {
				return false;
			}
			forced += usize::from(f);
			ok += usize::from(self.target_ok(card, i));
		}
		ok >= lo && forced <= hi && lo <= hi
	}

	/// For selection verb `v` under the mask `m`: the fewest and the most
	/// cards to pick, and the cards that can be picked first.
	fn select_spec(&self, v: usize, m: &[u8]) -> (i32, i32, u32) {
		if v >= N_SEL_VERBS || self.is_env_terminal() {
			return (0, -1, 0);
		}
		let a = sel_verb_action(v);
		if m[a] == 0 {
			return (0, -1, 0);
		}
		if v == SEL_VERB_REORDER_JOKERS {
			let nj = self.g.areas.jokers.len().min(MAX_JOKERS);
			let k = num::id(nj.min(MAX_PICKS));
			let elig = if nj >= 32 { u32::MAX } else { (1u32 << nj) - 1 };
			return (k, k, elig);
		}
		let hc = &self.g.areas.hand.cards;
		let nh = hc.len().min(MAX_HAND);
		let mut tgt_card: Option<&Card> = None;
		let (mut lo, mut hi): (i32, i32) =
			if (SEL_VERB_USE_CONS..SEL_VERB_PICK + MAX_PACK).contains(&v) {
				let card_opt = if v < SEL_VERB_PICK {
					self.g.areas.consumables.get(v - SEL_VERB_USE_CONS)
				} else {
					self.g
						.areas
						.pack_cards
						.get(v - SEL_VERB_PICK)
						.and_then(AnyCard::as_card)
				};
				let Some(c) = card_opt else {
					return (0, -1, 0);
				};
				let Some(hl) = consumables::targets(c.kind) else {
					return (0, -1, 0);
				};
				tgt_card = Some(c);
				(num::id(hl.min), num::id(hl.max))
			} else {
				(1, num::id(nh.min(MAX_PICKS)))
			};
		let mut forced = 0i32;
		let mut fbits = 0u32;
		let mut eligible = 0u32;
		for (i, pc) in hc.iter().take(nh).enumerate() {
			if pc.forced.is_locked() {
				forced += 1;
				fbits |= 1u32 << i;
			} else if tgt_card.is_none_or(|c| self.target_ok(c, i)) {
				eligible |= 1u32 << i;
			}
		}
		if let Some(ht) = hand_verb_target(v) {
			let (cm, _) = core_rule(&self.core_cache, ht, fbits);
			eligible = cm;
		}
		lo = (lo - forced).max(0);
		let elig_cnt = eligible.count_ones().cast_signed();
		hi = (hi - forced).min(elig_cnt);
		if hi < lo {
			(0, -1, 0)
		} else {
			(lo, hi, eligible)
		}
	}

	fn sel_specs(&self, m: &[u8], out: &mut [i32]) {
		for v in 0..N_SEL_VERBS {
			let (lo, hi, el) = self.select_spec(v, m);
			out[v * 3] = lo;
			out[v * 3 + 1] = hi;
			out[v * 3 + 2] = i32::from_ne_bytes(el.to_ne_bytes());
		}
	}

	/// The rule that [`step_select`](Self::step_select) checks every pick
	/// of selection verb `v` with: the cards that may be picked next, as a
	/// bit set, and whether the picks may stop here.
	///
	/// `lo`, `hi` and `elig` are the verb's row of the selection specs (see
	/// [`write_all_obs`](Self::write_all_obs)), `prefix` is the bit set of
	/// the forced cards and the cards picked so far, and `npicks` is the
	/// number of cards picked so far.
	#[must_use]
	pub fn pick_rule(
		&self, v: usize, lo: i32, hi: i32, elig: u32, prefix: u32, npicks: i32,
	) -> (u32, bool) {
		if let Some(ht) = hand_verb_target(v) {
			return core_rule(&self.core_cache, ht, prefix);
		}
		let cards = if npicks < hi { elig & !prefix } else { 0 };
		let stop = npicks >= lo;
		(cards, stop)
	}

	/// Writes what a policy reads of the game:
	///
	/// - `obs_out`: the observation, [`OBS_SIZE`] floats in the groups of
	///   [`OBS_GROUPS`];
	/// - `sel_out`: the selection observation, [`SEL_F`] floats;
	/// - `mask_out`: for each of the [`N_ACTIONS`] actions, 1 if it is legal
	///   and 0 if not (all 0 once the episode is over);
	/// - `specs_out`: three ints for each of the [`N_SEL_VERBS`] selection
	///   verbs (see [`sel_verb_index`]): the fewest and the most cards to
	///   pick, and the bit set of the cards that can be picked first (hand
	///   positions, or joker positions for [`ACT_REORDER_JOKERS`]). The most
	///   is `-1` for a verb that is not legal or takes no cards. Forced
	///   cards are always part of a selection and are not picked;
	/// - `hctx_out`: the hand context, [`HS_CTX`] ints.
	///
	/// # Panics
	///
	/// Panics if a slice is shorter than that.
	pub fn write_all_obs(
		&mut self, obs_out: &mut [f32], sel_out: &mut [f32], mask_out: &mut [u8],
		specs_out: &mut [i32], hctx_out: &mut [i32],
	) {
		write_obs(&mut self.g, self.cfg.hide_face_down, obs_out);
		write_obs_selection(&self.g, sel_out);
		self.mask(mask_out);
		self.sel_specs(mask_out, specs_out);
		write_hand_ctx(&self.g, &self.core_cache, hctx_out);
	}

	#[must_use]
	fn is_env_terminal(&self) -> bool {
		if self.g.flow.phase == Phase::GameOver {
			return true;
		}
		if self.g.run_won() && self.g.flow.state == State::RoundEval && self.cfg.endless_ante <= 8 {
			return true;
		}
		if !self.cfg.disable_softlocks && self.g.is_stuck() {
			return true;
		}
		false
	}

	fn dead_end(&self) -> bool {
		if self.is_env_terminal() {
			return false;
		}
		let mut m = [0u8; N_ACTIONS];
		self.mask(&mut m);
		let any = m.iter().any(|&x| x != 0);
		let pick = m[ACT_PICK..ACT_PICK + MAX_PACK].iter().any(|&x| x != 0);
		!any || (self.g.flow.state.is_pack() && !pick && m[ACT_SKIP_PACK] == 0)
	}

	fn record(&mut self, step: Step<'_>) {
		if self.cfg.record {
			writeln!(self.script, "{step}").expect("a String takes every write");
		}
	}

	fn exec_action(&mut self, act: Action) -> bool {
		let mut base_sale = 0.0;
		match act {
			Action::SellJoker(i) => {
				if let Some(c) = self.g.areas.jokers.get(usize::from(i)) {
					base_sale = (c.core.sell_cost - c.ab.extra_value).max(0.0);
				}
			}
			Action::SellConsumable(i) => {
				if let Some(c) = self.g.areas.consumables.get(usize::from(i)) {
					base_sale = (c.core.sell_cost - c.ab.extra_value).max(0.0);
				}
			}
			_ => {}
		}
		let before = self.g.run.earned;
		let forced = act == Action::SkipPack && self.cfg.disable_softlocks && skip_refused(&self.g);
		let step = if forced {
			Step::ForcedSkip
		} else {
			Step::Act(act)
		};
		let ok = step.apply(&mut self.g);
		if ok {
			self.record(step);
			self.refresh_core();
		}
		if ok && act == Action::SkipBlind {
			self.skips_pending += 1;
		}
		if ok && forced {
			self.softlocks += 1;
		}
		let gain = self.g.run.earned - before;
		self.earned += gain - gain.min(base_sale);
		ok
	}

	fn blind_beaten(&mut self, info: &mut BalStepInfo) {
		info.blinds += 1;
		let extra_skips = if self.cfg.reward_skip_deferred {
			self.skips_pending
		} else {
			0
		};
		let paid = self.cfg.reward_blind * num::feat(f64::from(1 + extra_skips));
		self.skips_pending = 0;
		if !self.cfg.reward_money {
			info.reward += paid;
		}
	}

	fn auto_advance(&mut self, info: &mut BalStepInfo) {
		for _ in 0..64 {
			if self.is_env_terminal() {
				return;
			}
			if self.g.run_won() && self.g.flow.state == State::RoundEval && !self.endless {
				return;
			}
			let mut m = [0u8; N_ACTIONS];
			self.mask(&mut m);
			let next_act = if self.g.flow.state == State::RoundEval && m[ACT_CASH_OUT] != 0 {
				Some(Action::CashOut)
			} else if self.g.flow.state == State::BlindSelect
				&& m[ACT_SELECT_BLIND] != 0
				&& m[ACT_SKIP_BLIND] == 0
				&& m[ACT_REROLL_BOSS] == 0
			{
				Some(Action::SelectBlind)
			} else {
				let mut only = None;
				let mut n = 0usize;
				for (a, &ok) in m.iter().enumerate() {
					if ok != 0 {
						only = Some(a);
						n += 1;
					}
				}
				if n == 1
					&& let Some(a) = only
					&& sel_verb_index(a).is_none()
				{
					simple_action(a)
				} else {
					None
				}
			};
			let Some(act) = next_act else {
				return;
			};
			let prev_state = self.g.flow.state;
			if !self.exec_action(act) {
				info.illegal = 1;
				return;
			}
			if self.g.flow.state != prev_state {
				self.joker_ordered_in_phase = false;
			}
			if self.g.flow.state == State::RoundEval && prev_state != State::RoundEval {
				self.blind_beaten(info);
			}
		}
	}

	/// Takes `action` and then the steps that need no decision.
	///
	/// A selection verb (see [`sel_verb_index`]) takes `cards`: hand
	/// positions in pick order, or joker positions for
	/// [`ACT_REORDER_JOKERS`], each allowed by
	/// [`pick_rule`](Self::pick_rule) after the ones before it. The forced
	/// cards are always included and are not listed. Any other action
	/// ignores `cards`.
	///
	/// An action that the mask does not allow, or cards that the rule does
	/// not allow, change nothing but the step count: the step is `illegal`.
	/// After a step that ends the episode, call [`reset`](Self::reset).
	pub fn step_select(&mut self, action: i32, cards: &[i32]) -> BalStepInfo {
		let mut m = [0u8; N_ACTIONS];
		self.mask(&mut m);
		let Some(a) = usize::try_from(action)
			.ok()
			.filter(|&a| a < N_ACTIONS && m[a] != 0)
		else {
			return self.finish_step(None);
		};
		let Some(v) = sel_verb_index(a) else {
			return self.finish_step(Some((a, None)));
		};
		let (lo, hi, elig) = self.select_spec(v, &m);
		if hi < 0 {
			return self.finish_step(Some((a, None)));
		}
		let mut order: Vec<usize> = Vec::with_capacity(MAX_PICKS);
		let hc_len = self.g.areas.hand.cards.len().min(MAX_HAND);
		if a != ACT_REORDER_JOKERS {
			for (i, pc) in self.g.areas.hand.cards.iter().enumerate().take(hc_len) {
				if pc.forced.is_locked() {
					order.push(i);
				}
			}
		}
		let mut prefix = 0u32;
		for &c in &order {
			prefix |= 1u32 << c;
		}
		let mut ok = cards.len() <= MAX_PICKS;
		for (k, &c_i32) in cards.iter().enumerate() {
			if !ok {
				break;
			}
			let (cm, _) = self.pick_rule(v, lo, hi, elig, prefix, num::id(k));
			if let Ok(c) = usize::try_from(c_i32)
				&& c < MAX_HAND
				&& ((cm >> c) & 1) != 0
			{
				order.push(c);
				prefix |= 1u32 << c;
			} else {
				ok = false;
			}
		}
		if ok {
			let (_, stop) = self.pick_rule(v, lo, hi, elig, prefix, num::id(cards.len()));
			ok &= stop;
		}
		if !ok {
			return self.finish_step(None);
		}
		self.finish_step(Some((a, Some((v, &order)))))
	}

	/// The position in the hand of the card `id`, which the hand holds.
	fn hand_position(&self, id: u32) -> usize {
		let hand = &self.g.areas.hand.cards;
		hand.iter()
			.position(|c| c.core.id == id)
			.expect("moving hand cards keeps every card in the hand")
	}

	/// Takes a move between two cards the environment has just found, which
	/// the game always allows.
	fn reorder(&mut self, act: Action) {
		assert!(
			self.exec_action(act),
			"the game refused {act:?} between two cards it holds"
		);
	}

	/// Moves the forced cards past the first [`MAX_HAND`] hand cards to the
	/// front of the hand. Every selection must hold the forced cards, and
	/// the observation and the pick rules only cover those first cards.
	fn show_forced_cards(&mut self) {
		if self.g.flow.phase != Phase::Round {
			return;
		}
		let unseen = |hand: &[PlayingCard]| {
			let mut past = hand.iter().skip(MAX_HAND);
			Some(MAX_HAND + past.position(|card| card.forced.is_locked())?)
		};
		while let Some(from) = unseen(&self.g.areas.hand.cards) {
			self.reorder(Action::MoveHand {
				from: num::byte(from),
				to: 0,
			});
		}
	}

	fn hs_order_play(&mut self, order: &[usize]) -> CardMask {
		let ids: Vec<u32> = order
			.iter()
			.map(|&i| self.g.areas.hand.cards[i].core.id)
			.collect();
		for j in 1..ids.len() {
			let (p, q) = (self.hand_position(ids[j]), self.hand_position(ids[j - 1]));
			if p < q {
				self.reorder(Action::MoveHand {
					from: num::byte(p),
					to: num::byte(q),
				});
			}
		}
		ids.iter().map(|&id| self.hand_position(id)).collect()
	}

	fn reorder_jokers(&mut self, order: &[usize]) {
		let nj = self.g.areas.jokers.len();
		let mut target = Vec::with_capacity(nj);
		let mut used = vec![false; nj];
		for &idx in order {
			target.push(self.g.areas.jokers[idx].core.id);
			used[idx] = true;
		}
		for (i, &u) in used.iter().enumerate() {
			if !u {
				target.push(self.g.areas.jokers[i].core.id);
			}
		}
		for (k, &tid) in target.iter().enumerate() {
			if self.g.areas.jokers[k].core.id == tid {
				continue;
			}
			let offset = self.g.areas.jokers[k + 1..]
				.iter()
				.position(|c| c.core.id == tid)
				.expect("a joker that is not in place yet is further right");
			self.reorder(Action::MoveJoker {
				from: num::byte(k + 1 + offset),
				to: num::byte(k),
			});
		}
	}

	fn apply_death_reorder(&mut self, order: &[usize]) -> CardMask {
		if order.len() == 2 && order[0] < order[1] {
			let (src, dst) = (order[0], order[1]);
			for k in (src..dst).rev() {
				self.reorder(Action::MoveHand {
					from: num::byte(k),
					to: num::byte(k + 1),
				});
			}
			let mut mask = CardMask::EMPTY;
			mask.insert(src);
			mask.insert(src + 1);
			mask
		} else {
			mask_of_slice(order)
		}
	}

	fn record_play_info(&self, mask: CardMask, ht: usize, info: &mut BalStepInfo) {
		let x = sel_ctx(&self.g);
		let bits = mask.bits();
		let n_played = mask.len();
		info.played_cards = num::id(n_played);
		let scoring = sel_eval(&x, bits, n_played);
		info.played_scoring = scoring.count_ones().cast_signed();
		for (i, pc) in self.g.areas.hand.cards.iter().take(x.nh).enumerate() {
			if pc.seal == Some(Seal::Blue) {
				info.blue_held += 1;
				info.blue_played += i32::from(mask.contains(i));
			}
		}
		info.hs_declared = num::id(ht);
	}

	/// Counts a step and takes it: `legal` is the action, with the verb and
	/// the cards of a selection, or `None` for a step that is not allowed.
	fn finish_step(&mut self, legal: Option<(usize, Option<Selection<'_>>)>) -> BalStepInfo {
		let mut info = BalStepInfo {
			hand_played: -1,
			hs_declared: -1,
			..BalStepInfo::default()
		};
		let earned0 = self.earned;
		let softlocks0 = self.softlocks;
		self.steps += 1;
		info.reward -= self.cfg.step_cost;
		match legal {
			None => info.illegal = 1,
			Some((ACT_REORDER_JOKERS, Some((_, order)))) => {
				self.joker_ordered_in_phase = true;
				info.reward -= self.cfg.toggle_cost;
				self.reorder_jokers(order);
				self.auto_advance(&mut info);
			}
			Some((a, sel_order)) => self.step_valid_action(a, sel_order, &mut info),
		}
		self.show_forced_cards();
		self.finalize_step_info(earned0, softlocks0, &mut info);
		info
	}

	fn step_valid_action(
		&mut self, a: usize, sel_order: Option<Selection<'_>>, info: &mut BalStepInfo,
	) {
		let prev_state = self.g.flow.state;
		let prev_chips = self.g.round.chips;
		let hands0 = self.g.run.hands_played;
		let mut is_hand_play = false;

		let sim_act = if let Some((v, order)) = sel_order {
			if let Some(ht) = hand_verb_target(v) {
				is_hand_play = true;
				let mask = self.hs_order_play(order);
				if self.g.flow.state == State::SelectingHand {
					self.record_play_info(mask, ht, info);
				}
				Some(Action::Play(mask))
			} else if v == SEL_VERB_DISCARD {
				Some(Action::Discard(mask_of_slice(order)))
			} else if v < SEL_VERB_PICK {
				let slot = v - SEL_VERB_USE_CONS;
				let is_death = self
					.g
					.areas
					.consumables
					.get(slot)
					.is_some_and(|c| c.kind == Kind::Tarot(Tarot::Death));
				let mask = if is_death {
					self.apply_death_reorder(order)
				} else {
					mask_of_slice(order)
				};
				Some(Action::UseConsumable {
					index: num::byte(slot),
					targets: mask,
				})
			} else {
				let slot = v - SEL_VERB_PICK;
				let is_death = self
					.g
					.areas
					.pack_cards
					.get(slot)
					.and_then(AnyCard::as_card)
					.is_some_and(|c| c.kind == Kind::Tarot(Tarot::Death));
				let mask = if is_death {
					self.apply_death_reorder(order)
				} else {
					mask_of_slice(order)
				};
				Some(Action::PickPackCard {
					index: num::byte(slot),
					targets: mask,
				})
			}
		} else {
			decode_non_select_action(a)
		};

		if let Some(act) = sim_act {
			if !self.exec_action(act) {
				info.illegal = 1;
			}
		} else {
			info.illegal = 1;
		}

		if (self.g.run.hands_played - hands0).abs() > f64::EPSILON {
			let hand = self.g.history.last_hand_played;
			info.hand_played = i32::from(hand.expect("the game names a played hand") as u8);
			info.hand_score = self.g.run.last_hand_score.floor();
		}
		if self.g.flow.state == State::RoundEval && prev_state != State::RoundEval {
			self.blind_beaten(info);
		}
		if self.cfg.reward_chips != 0.0
			&& (a == ACT_PLAY || is_hand_play)
			&& self.g.blind.chips > 0.0
		{
			info.reward += self.cfg.reward_chips
				* num::feat((self.g.round.chips - prev_chips) / self.g.blind.chips.max(1.0));
		}
		self.update_phase_order_gate(prev_state, a, is_hand_play);
		self.auto_advance(info);
	}

	fn update_phase_order_gate(&mut self, prev_state: State, a: usize, hand_play: bool) {
		if self.g.flow.state != prev_state {
			self.joker_ordered_in_phase = false;
		} else if prev_state == State::SelectingHand {
			if a == ACT_PLAY
				|| hand_play
				|| a == ACT_DISCARD
				|| (ACT_USE_CONS..ACT_USE_CONS + MAX_CONS).contains(&a)
			{
				self.joker_ordered_in_phase = false;
			}
		} else if prev_state == State::Shop {
			if (ACT_BUY..ACT_BUY_USE + MAX_SHOP).contains(&a)
				|| (ACT_SELL_JOKER..ACT_SELL_JOKER + MAX_JOKERS).contains(&a)
				|| (ACT_SELL_CONS..ACT_SELL_CONS + MAX_CONS).contains(&a)
				|| (ACT_OPEN..ACT_OPEN + MAX_BOOST).contains(&a)
				|| a == ACT_REROLL
				|| a == ACT_END_SHOP
			{
				self.joker_ordered_in_phase = false;
			}
		} else if (self.g.flow.state.is_pack() || prev_state.is_pack())
			&& (ACT_PICK..ACT_PICK + MAX_PACK).contains(&a)
		{
			self.joker_ordered_in_phase = false;
		}
	}

	fn finalize_step_info(&mut self, earned0: f64, softlocks0: i32, info: &mut BalStepInfo) {
		let win_reward = |env: &Self, chips: f64| -> f32 {
			if env.cfg.reward_win_log {
				num::feat(chips.max(1.0).log10())
			} else {
				env.cfg.reward_win
			}
		};
		let lose_partial = |env: &Self| -> f32 {
			let f = num::feat((env.g.round.chips / env.g.blind.chips).clamp(0.0, 1.0));
			if env.cfg.reward_money {
				f * num::feat(env.g.blind.dollars) / env.cfg.money_per_reward
			} else {
				f
			}
		};

		if self.cfg.endless_ante > 8
			&& !self.endless
			&& self.g.run_won()
			&& self.g.flow.state == State::RoundEval
		{
			self.endless = true;
			self.win_score = self.g.round.chips;
			info.reward += win_reward(self, self.g.round.chips);
			self.record(Step::Endless);
			self.auto_advance(info);
		}

		if self.endless && (self.is_env_terminal() || self.g.run.ante > self.cfg.endless_ante) {
			info.terminated = 1;
			info.won = 1;
			let beat = self.g.flow.state != State::GameOver && !self.is_env_terminal();
			info.score = if beat {
				self.g.round.chips
			} else {
				self.win_score
			};
			if beat {
				info.reward += win_reward(self, self.g.round.chips);
			} else if self.cfg.reward_lose_partial && self.g.blind.chips > 0.0 {
				info.reward += lose_partial(self);
			}
		} else if self.is_env_terminal() {
			info.terminated = 1;
			let won = self.g.run_won() && self.g.flow.state != State::GameOver;
			info.won = u8::from(won);
			info.score = self.g.round.chips;
			if won {
				info.reward += win_reward(self, self.g.round.chips);
			} else {
				info.reward += self.cfg.reward_lose;
				if self.cfg.reward_lose_partial && self.g.blind.chips > 0.0 {
					info.reward += lose_partial(self);
				}
			}
		} else if self.dead_end() {
			info.terminated = 1;
			info.dead_end = 1;
			info.score = self.g.round.chips;
			info.reward += self.cfg.reward_lose;
		} else if self.steps >= self.cfg.max_steps {
			info.truncated = 1;
		}

		info.earned = num::feat(self.earned - earned0);
		info.softlocks_avoided = self.softlocks - softlocks0;
		let blue = self.count_blue_seals();
		if self.cfg.reward_blue_seal != 0.0 {
			info.reward += self.cfg.reward_blue_seal * num::feat(f64::from(blue - self.blue_seals));
		}
		self.blue_seals = blue;
		info.blue_seals = blue;
		if self.cfg.reward_money {
			info.reward += info.earned / self.cfg.money_per_reward;
		}
		info.ante = self.g.run.ante;
		info.round = self.g.run.round;
		#[expect(
			clippy::cast_possible_truncation,
			reason = "the game's money is a whole number of dollars; a balance beyond an i32 saturates"
		)]
		let dollars = self.g.run.dollars as i32;
		info.dollars = dollars;
		info.steps = self.steps;
	}
}
