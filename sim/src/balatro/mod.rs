//! Native Rust implementation of `RLBalatro`'s lean-actions environment
//! (`152` discrete actions, `32` selection verbs, hand-first declared-hand
//! plays, `auto_advance`, `StartPoint` replay, and exact `8281 + 44` obs).

pub mod hand_eval;
pub mod ids;
pub mod obs;
mod script;

use std::fmt::Write as _;
use std::sync::Arc;

use self::hand_eval::{CoreCtx, HS_ANY, N_HANDS, core_ctx, core_rule, sel_ctx, sel_eval};
use self::ids::{N_JOKER_KEYS, joker_alpha};
use self::obs::{
	MAX_BOOST, MAX_CONS, MAX_HAND, MAX_JOKERS, MAX_PACK, MAX_SHOP, MAX_VOUCH, write_hand_ctx,
	write_obs, write_obs_selection,
};
use self::script::{Header, Step};
pub use self::script::{ScriptError, StartPoint, load_starts};
use crate::actions::Action;
use crate::card::{AnyCard, Card, CardLike, CardMask, Kind};
use crate::data::{Back, Edition, Seal, Spectral, Tarot};
use crate::game::{Area, BlindSlot, Game, Phase, State};
use crate::{SEED_CHARS, Stake, consumables, event, num, shop};

pub const N_ACTIONS: usize = 152;
pub const N_SEL_VERBS: usize = 32;

pub const ACT_PLAY: usize = 0;
pub const ACT_DISCARD: usize = 1;
pub const ACT_SELECT_BLIND: usize = 2;
pub const ACT_SKIP_BLIND: usize = 3;
pub const ACT_REROLL_BOSS: usize = 4;
pub const ACT_CASH_OUT: usize = 5;
pub const ACT_END_SHOP: usize = 6;
pub const ACT_REROLL: usize = 7;
pub const ACT_BUY: usize = 8;
pub const ACT_BUY_USE: usize = ACT_BUY + MAX_SHOP;
pub const ACT_REDEEM: usize = ACT_BUY_USE + MAX_SHOP;
pub const ACT_OPEN: usize = ACT_REDEEM + MAX_VOUCH;
pub const ACT_PICK: usize = ACT_OPEN + MAX_BOOST;
pub const ACT_SKIP_PACK: usize = ACT_PICK + MAX_PACK;
pub const ACT_SELL_JOKER: usize = ACT_SKIP_PACK + 1;
pub const ACT_SELL_CONS: usize = ACT_SELL_JOKER + MAX_JOKERS;
pub const ACT_USE_CONS: usize = ACT_SELL_CONS + MAX_CONS;
pub const ACT_JOKER_RIGHT: usize = ACT_USE_CONS + MAX_CONS;
pub const ACT_JOKER_FRONT: usize = ACT_JOKER_RIGHT + MAX_JOKERS - 1;
pub const ACT_JOKER_BACK: usize = ACT_JOKER_FRONT + MAX_JOKERS;
pub const ACT_REORDER_JOKERS: usize = ACT_JOKER_BACK + MAX_JOKERS;
pub const ACT_PLAY_HAND: usize = ACT_REORDER_JOKERS + 1;
const _: () = assert!(ACT_PLAY_HAND + (N_HANDS - 1) == N_ACTIONS);

pub const SEL_VERB_PLAY: usize = 0;
pub const SEL_VERB_DISCARD: usize = 1;
pub const SEL_VERB_USE_CONS: usize = 2;
pub const SEL_VERB_PICK: usize = SEL_VERB_USE_CONS + MAX_CONS;
pub const SEL_VERB_REORDER_JOKERS: usize = SEL_VERB_PICK + MAX_PACK;
pub const SEL_VERB_PLAY_HAND: usize = SEL_VERB_REORDER_JOKERS + 1;
const _: () = assert!(SEL_VERB_PLAY_HAND + (N_HANDS - 1) == N_SEL_VERBS);

#[must_use]
pub fn seed_from_index(i: u64) -> String {
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
#[must_use]
pub fn sel_verb_action(v: usize) -> usize {
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

#[inline]
#[must_use]
pub fn hand_verb_target(v: usize) -> Option<usize> {
	if v == SEL_VERB_PLAY {
		Some(HS_ANY)
	} else if (SEL_VERB_PLAY_HAND..SEL_VERB_PLAY_HAND + N_HANDS - 1).contains(&v) {
		Some(v - SEL_VERB_PLAY_HAND)
	} else {
		None
	}
}

#[derive(Clone, Debug)]
pub struct BalConfig {
	pub back: Back,
	pub stake: Stake,
	pub max_steps: i32,
	pub reward_blind: f32,
	pub reward_win: f32,
	pub reward_win_log: bool,
	pub endless_ante: i32,
	pub reward_lose: f32,
	pub reward_lose_partial: bool,
	pub reward_money: bool,
	pub money_per_reward: f32,
	pub reward_chips: f32,
	pub reward_blue_seal: f32,
	pub reward_skip_deferred: bool,
	pub step_cost: f32,
	pub toggle_cost: f32,
	pub disable_softlocks: bool,
	pub hide_face_down: bool,
	pub sticker_penalty: f32,
	pub joker_marks: [u8; N_JOKER_KEYS],
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
			sticker_penalty: 1.0,
			joker_marks: [0; N_JOKER_KEYS],
			record: false,
		}
	}
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BalStepInfo {
	pub reward: f32,
	pub terminated: u8,
	pub truncated: u8,
	pub won: u8,
	pub illegal: u8,
	pub ante: i32,
	pub round: i32,
	pub dollars: i32,
	pub steps: i32,
	pub score: f64,
	pub earned: f32,
	pub blinds: i32,
	pub softlocks_avoided: i32,
	pub dead_end: i32,
	pub stickered_held: i32,
	pub stickered_lineup: i32,
	pub hand_played: i32,
	pub hand_factor: f32,
	pub blue_seals: i32,
	pub bounty_drawn: i32,
	pub bounty_deck: i32,
	pub target_hits: i32,
	pub target_held: i32,
	pub played_cards: i32,
	pub played_scoring: i32,
	pub blue_held: i32,
	pub blue_played: i32,
	pub hs_declared: i32,
	pub hand_score: f64,
}

pub struct BalEnv {
	pub cfg: BalConfig,
	pub g: Game,
	pub steps: i32,
	pub endless: bool,
	pub win_score: f64,
	pub last_play_blind: f64,
	pub joker_ordered_in_phase: bool,
	pub marks: [u8; N_JOKER_KEYS],
	pub n_marked: usize,
	pub blind_factor: f32,
	pub skips_pending: i32,
	pub blue_seals: i32,
	pub earned: f64,
	pub earned_scaled: f64,
	pub softlocks: i32,
	pub start: Option<Arc<StartPoint>>,
	pub start_reseed: bool,
	pub episode_start: i32,
	pub script: String,
	core_cache: CoreCtx,
}

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
			last_play_blind: 0.0,
			joker_ordered_in_phase: false,
			marks: [0; N_JOKER_KEYS],
			n_marked: 0,
			blind_factor: 1.0,
			skips_pending: 0,
			blue_seals: 0,
			earned: 0.0,
			earned_scaled: 0.0,
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

	pub fn reset_index(&mut self, seed_index: u64) {
		self.reset(&seed_from_index(seed_index));
	}

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
		self.last_play_blind = 0.0;
		self.marks = self.cfg.joker_marks;
		self.n_marked = self.marks.iter().filter(|&&m| m != 0).count();
		self.blind_factor = 1.0;
		self.earned_scaled = self.earned;
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
			let _ = writeln!(self.script, "{header}");
		}
		let mut unused = BalStepInfo::default();
		self.auto_advance(&mut unused);
		self.refresh_core();
	}

	fn refresh_core(&mut self) {
		self.core_cache = core_ctx(&self.g, self.cfg.hide_face_down);
	}

	#[must_use]
	pub fn count_blue_seals(&self) -> i32 {
		num::id(
			self.g
				.areas
				.playing_cards()
				.filter(|c| c.seal == Some(Seal::Blue))
				.count(),
		)
	}

	#[must_use]
	pub fn stickered_in_lineup(&self) -> i32 {
		if self.n_marked == 0 {
			return 0;
		}
		let mut n = 0i32;
		for c in &self.g.areas.jokers {
			if let Kind::Joker(j) = c.kind
				&& (self.marks[joker_alpha(j)] & 1) != 0
			{
				n += 1;
			}
		}
		n
	}

	#[must_use]
	pub fn sticker_factor(&self, n: i32) -> f32 {
		if n == 0 {
			1.0
		} else {
			let slots = num::feat(self.g.areas.limits.jokers.limit.max(1.0));
			(1.0 - self.cfg.sticker_penalty * num::feat(f64::from(n)) / slots).max(0.0)
		}
	}

	fn hands_available(&self) -> u32 {
		let mut av = 0u32;
		for h in 0..=HS_ANY {
			let (cm, stop) = core_rule(&self.core_cache, h, true, self.core_cache.forced);
			if cm != 0 || stop {
				av |= 1u32 << h;
			}
		}
		av
	}

	pub fn mask(&self, m: &mut [u8]) {
		m[..N_ACTIONS].fill(0);
		if self.is_env_terminal() {
			return;
		}
		let st = self.g.flow.state;
		let nh = self.g.areas.hand.cards.len().min(MAX_HAND);
		let hand_state = st == State::SelectingHand || st.is_pack();
		if st == State::SelectingHand {
			let av = if nh > 0 { self.hands_available() } else { 0 };
			m[ACT_PLAY] = u8::try_from((av >> HS_ANY) & 1).unwrap_or(0);
			for h in 0..N_HANDS - 1 {
				m[ACT_PLAY_HAND + h] = u8::try_from((av >> h) & 1).unwrap_or(0);
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
			|| (self.cfg.disable_softlocks && !self.g.areas.pack_cards.is_empty());
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
			let f = self.g.areas.hand.cards[i].forced_selection;
			if f && !self.target_ok(card, i) {
				return false;
			}
			forced += usize::from(f);
			ok += usize::from(self.target_ok(card, i));
		}
		ok >= lo && forced <= hi && lo <= hi
	}

	#[must_use]
	pub fn select_spec(&self, v: usize, m: &[u8]) -> (i32, i32, u32) {
		if v >= N_SEL_VERBS || self.is_env_terminal() {
			return (0, -1, 0);
		}
		let a = sel_verb_action(v);
		if m[a] == 0 {
			return (0, -1, 0);
		}
		if v == SEL_VERB_REORDER_JOKERS {
			let nj = self.g.areas.jokers.len().min(MAX_JOKERS);
			let k = num::id(nj.min(5));
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
				(1, num::id(nh.min(5)))
			};
		let mut forced = 0i32;
		let mut fbits = 0u32;
		let mut eligible = 0u32;
		for (i, pc) in hc.iter().take(nh).enumerate() {
			if pc.forced_selection {
				forced += 1;
				fbits |= 1u32 << i;
			} else if tgt_card.is_none_or(|c| self.target_ok(c, i)) {
				eligible |= 1u32 << i;
			}
		}
		if let Some(ht) = hand_verb_target(v) {
			let (cm, _) = core_rule(&self.core_cache, ht, true, fbits);
			eligible = cm;
		}
		lo = (lo - forced).max(0);
		let elig_cnt = i32::try_from(eligible.count_ones()).unwrap_or(0);
		hi = (hi - forced).min(elig_cnt);
		if hi < lo {
			(0, -1, 0)
		} else {
			(lo, hi, eligible)
		}
	}

	pub fn sel_specs(&self, m: &[u8], out: &mut [i32]) {
		for v in 0..N_SEL_VERBS {
			let (lo, hi, el) = self.select_spec(v, m);
			out[v * 3] = lo;
			out[v * 3 + 1] = hi;
			out[v * 3 + 2] = i32::from_ne_bytes(el.to_ne_bytes());
		}
	}

	#[must_use]
	pub fn pick_rule(
		&self, v: usize, lo: i32, hi: i32, elig: u32, prefix: u32, npicks: i32,
	) -> (u32, bool) {
		if let Some(ht) = hand_verb_target(v) {
			return core_rule(&self.core_cache, ht, true, prefix);
		}
		let cards = if npicks < hi { elig & !prefix } else { 0 };
		let stop = npicks >= lo;
		(cards, stop)
	}

	pub fn write_all_obs(
		&mut self, obs_out: &mut [f32], sel_out: &mut [f32], mask_out: &mut [u8],
		specs_out: &mut [i32], hctx_out: &mut [i32],
	) {
		self.refresh_core();
		write_obs(
			&mut self.g,
			self.cfg.hide_face_down,
			self.last_play_blind,
			&self.marks,
			self.n_marked,
			obs_out,
		);
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
			let _ = writeln!(self.script, "{step}");
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
		let is_softlock_skip = act == Action::SkipPack
			&& self.g.flow.state.is_pack()
			&& self.g.check(Action::SkipPack).is_err()
			&& self.cfg.disable_softlocks
			&& !self.g.areas.pack_cards.is_empty();
		let ok = if is_softlock_skip {
			event::begin_action(&mut self.g.rng, &mut self.g.events);
			shop::do_skip_pack(&mut self.g);
			event::run_queue(&mut self.g);
			true
		} else {
			self.g.step(act).is_ok()
		};
		if ok {
			self.record(Step::Act(act));
		}
		if ok && act == Action::SkipBlind {
			self.skips_pending += 1;
		}
		if ok && is_softlock_skip {
			self.softlocks += 1;
		}
		let gain = self.g.run.earned - before;
		let counted = gain - gain.min(base_sale);
		self.earned += counted;
		if self.n_marked > 0 && counted != 0.0 {
			let factor = if act == Action::CashOut {
				self.blind_factor
			} else {
				self.sticker_factor(self.stickered_in_lineup())
			};
			self.earned_scaled += counted * f64::from(factor);
		}
		ok
	}

	fn blind_beaten(&mut self, info: &mut BalStepInfo) {
		info.blinds += 1;
		let n = if self.n_marked > 0 {
			self.stickered_in_lineup()
		} else {
			0
		};
		self.blind_factor = self.sticker_factor(n);
		info.stickered_held += n;
		let extra_skips = if self.cfg.reward_skip_deferred {
			self.skips_pending
		} else {
			0
		};
		let paid = self.cfg.reward_blind * num::feat(f64::from(1 + extra_skips));
		self.skips_pending = 0;
		if !self.cfg.reward_money {
			info.reward += if n > 0 {
				paid * self.blind_factor
			} else {
				paid
			};
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

	pub fn step_select(&mut self, action: i32, cards: &[i32]) -> BalStepInfo {
		self.refresh_core();
		let mut m = [0u8; N_ACTIONS];
		self.mask(&mut m);
		let Ok(a) = usize::try_from(action) else {
			return self.finish_step(action, None, &m, true);
		};
		if a >= N_ACTIONS || m[a] == 0 {
			return self.finish_step(action, None, &m, true);
		}
		let Some(v) = sel_verb_index(a) else {
			return self.finish_step(action, None, &m, false);
		};
		let (lo, hi, elig) = self.select_spec(v, &m);
		if hi < 0 {
			return self.finish_step(action, None, &m, false);
		}
		let mut order: Vec<usize> = Vec::with_capacity(5);
		let hc_len = self.g.areas.hand.cards.len().min(MAX_HAND);
		if a != ACT_REORDER_JOKERS {
			for (i, pc) in self.g.areas.hand.cards.iter().enumerate().take(hc_len) {
				if pc.forced_selection {
					order.push(i);
				}
			}
		}
		let mut prefix = 0u32;
		for &c in &order {
			prefix |= 1u32 << c;
		}
		let mut ok = cards.len() <= 5;
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
			return self.finish_step(action, None, &m, true);
		}
		self.finish_step(action, Some((v, &order)), &m, false)
	}

	fn hs_order_play(&mut self, order: &[usize]) -> CardMask {
		let ids: Vec<u32> = order
			.iter()
			.map(|&i| self.g.areas.hand.cards[i].core.id)
			.collect();
		for j in 1..ids.len() {
			let p = self
				.g
				.areas
				.hand
				.cards
				.iter()
				.position(|c| c.core.id == ids[j])
				.unwrap_or(0);
			let q = self
				.g
				.areas
				.hand
				.cards
				.iter()
				.position(|c| c.core.id == ids[j - 1])
				.unwrap_or(0);
			if p > q {
				continue;
			}
			let _ = self.exec_action(Action::MoveHand {
				from: num::byte(p),
				to: num::byte(q),
			});
		}
		let mut mask = CardMask::EMPTY;
		for id in ids {
			if let Some(pos) = self.g.areas.hand.cards.iter().position(|c| c.core.id == id) {
				mask.insert(pos);
			}
		}
		mask
	}

	fn reorder_jokers(&mut self, order: &[usize]) {
		let nj = self.g.areas.jokers.len();
		if nj < 2 {
			return;
		}
		let mut target = Vec::with_capacity(nj);
		let mut used = vec![false; nj];
		for &idx in order {
			if idx < nj && !used[idx] {
				target.push(self.g.areas.jokers[idx].core.id);
				used[idx] = true;
			}
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
			if let Some(offset) = self.g.areas.jokers[k + 1..]
				.iter()
				.position(|c| c.core.id == tid)
			{
				let curr_idx = k + 1 + offset;
				let _ = self.exec_action(Action::MoveJoker {
					from: num::byte(curr_idx),
					to: num::byte(k),
				});
			}
		}
	}

	fn apply_death_reorder(&mut self, order: &[usize]) -> CardMask {
		if order.len() == 2 && order[0] < order[1] {
			let (src, dst) = (order[0], order[1]);
			for k in (src..dst).rev() {
				let _ = self.exec_action(Action::MoveHand {
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
		let n_played = usize::try_from(bits.count_ones()).unwrap_or(0);
		info.played_cards = num::id(n_played);
		let ev = sel_eval(&x, bits, n_played);
		info.played_scoring = i32::try_from(ev.scoring.count_ones()).unwrap_or(0);
		for (i, pc) in self.g.areas.hand.cards.iter().take(x.nh).enumerate() {
			if pc.seal == Some(Seal::Blue) {
				info.blue_held += 1;
				info.blue_played += i32::try_from((bits >> i) & 1).unwrap_or(0);
			}
		}
		info.hs_declared = num::id(ht);
	}

	fn finish_step(
		&mut self, action: i32, sel_order: Option<(usize, &[usize])>, m: &[u8], mut illegal: bool,
	) -> BalStepInfo {
		let mut info = BalStepInfo {
			hand_played: -1,
			hs_declared: -1,
			..BalStepInfo::default()
		};
		let earned0 = self.earned;
		let scaled0 = self.earned_scaled;
		let softlocks0 = self.softlocks;
		self.steps += 1;
		info.reward -= self.cfg.step_cost;
		let a = usize::try_from(action).unwrap_or(usize::MAX);
		if a >= N_ACTIONS || m[a] == 0 {
			illegal = true;
		}
		if illegal {
			info.illegal = 1;
		} else if a == ACT_REORDER_JOKERS {
			self.joker_ordered_in_phase = true;
			info.reward -= self.cfg.toggle_cost;
			if let Some((_, order)) = sel_order {
				self.reorder_jokers(order);
			}
			self.auto_advance(&mut info);
		} else {
			self.step_valid_action(a, sel_order, &mut info);
		}
		self.finalize_step_info(earned0, scaled0, softlocks0, &mut info);
		info
	}

	fn step_valid_action(
		&mut self, a: usize, sel_order: Option<(usize, &[usize])>, info: &mut BalStepInfo,
	) {
		let prev_state = self.g.flow.state;
		let prev_chips = self.g.round.chips;
		let hands0 = self.g.run.hands_played;
		let blind0 = self.g.blind.chips;
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
			info.hand_played = self
				.g
				.history
				.last_hand_played
				.map_or(-1, |h| i32::from(h as u8));
			info.hand_score = self.g.run.last_hand_score.floor();
			self.last_play_blind = blind0;
		}
		if self.g.flow.state == State::RoundEval && prev_state != State::RoundEval {
			self.blind_beaten(info);
		}
		if self.cfg.reward_chips != 0.0
			&& (a == ACT_PLAY || is_hand_play)
			&& self.g.blind.chips > 0.0
		{
			let rc = self.cfg.reward_chips
				* num::feat((self.g.round.chips - prev_chips) / self.g.blind.chips.max(1.0));
			let n = if self.n_marked > 0 {
				self.stickered_in_lineup()
			} else {
				0
			};
			info.reward += if n > 0 {
				rc * self.sticker_factor(n)
			} else {
				rc
			};
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

	fn finalize_step_info(
		&mut self, earned0: f64, scaled0: f64, softlocks0: i32, info: &mut BalStepInfo,
	) {
		let n_stk = if self.n_marked > 0 {
			self.stickered_in_lineup()
		} else {
			0
		};
		let scaled = |env: &Self, r: f32| -> f32 {
			if n_stk > 0 {
				r * env.sticker_factor(n_stk)
			} else {
				r
			}
		};
		let win_reward = |env: &Self, chips: f64| -> f32 {
			let base = if env.cfg.reward_win_log {
				num::feat(chips.max(1.0).log10())
			} else {
				env.cfg.reward_win
			};
			scaled(env, base)
		};
		let lose_partial = |env: &Self| -> f32 {
			let f = num::feat((env.g.round.chips / env.g.blind.chips).clamp(0.0, 1.0));
			let base = if env.cfg.reward_money {
				f * num::feat(env.g.blind.dollars) / env.cfg.money_per_reward
			} else {
				f
			};
			scaled(env, base)
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
			let paid = if self.n_marked > 0 {
				num::feat(self.earned_scaled - scaled0)
			} else {
				info.earned
			};
			info.reward += paid / self.cfg.money_per_reward;
		}
		info.stickered_lineup = n_stk;
		info.ante = self.g.run.ante;
		info.round = self.g.run.round;
		info.dollars = num::id(num::trunc(self.g.run.dollars));
		info.steps = self.steps;
	}
}
