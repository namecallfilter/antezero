//! Per-env episode stepping, reward shaping, seeding, and ante-start snapshot
//! reservoirs used by `VecEnv`.

use std::ops::RangeInclusive;

use rayon::prelude::*;
use sim::data::{ALL_JOKERS, ALL_PLANETS, Back};
use sim::game::{BlindSlot, BlindStatus, Game as SimGame, Phase};
use sim::inject::Inject;
use sim::obs::{self, BuildPreview, Op};
use sim::rng::splitmix64;
use sim::{Action, CardMask, SEED_CHARS, Stake, num};

use crate::interval::{ConsEvent, EpisodeStatsOut, HandCounts, JokerSet, consumable_event};
use crate::rows::{ObsOptions, Rows};

/// Seed string `n` in the game's seed alphabet (8 characters).
fn seed_string(n: u64) -> String {
	// Spread consecutive indices over the seed space.
	let mut n = n.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (n >> 29);
	let radix = num::index64(SEED_CHARS.len());
	let mut seed = String::with_capacity(8);
	for _ in 0..8 {
		seed.push(char::from(SEED_CHARS[num::pick(n, SEED_CHARS.len())]));
		n /= radix;
	}
	seed
}

/// Uniform in [0, 1) from a hash.
#[expect(
	clippy::cast_precision_loss,
	reason = "`splitmix64(n) >> 40` is below 2^24, which an f32 holds exactly"
)]
fn unit(n: u64) -> f32 {
	(splitmix64(n) >> 40) as f32 / 16_777_216.0
}

/// The shaping potential `w_blind * progress + w_score * fraction of the
/// current blind's chips scored`, with progress = the run position in blinds
/// (the blinds of every finished ante plus this ante's defeated ones). The
/// lost blind's fraction is kept at game over: if losing dropped it,
/// delaying a loss would pay (stalling).
// Counting blinds cleared instead would pay for skipping blinds and for
// Hieroglyph/Petroglyph (which add an ante of blinds to beat), and policies
// exploit both.
pub(crate) fn potential(g: &SimGame, w_blind: f32, w_score: f32) -> f32 {
	let in_round = matches!(g.flow.phase, Phase::Round | Phase::GameOver);
	let round = if in_round && g.blind.chips > 0.0 {
		num::feat((g.round.chips / g.blind.chips).min(1.0))
	} else {
		0.0
	};
	let defeated = num::id(
		g.blind_select
			.states
			.iter()
			.filter(|&&s| s == BlindStatus::Defeated)
			.count(),
	);
	let progress = num::id(BlindSlot::ALL.len()) * (g.run.ante - 1).max(0) + defeated;
	w_blind * num::feat(f64::from(progress)) + w_score * round
}

/// The reward. The true objective is `r_win` for winning the run (plus
/// `r_endless` per blind cleared after the win, for endless training, and
/// `-r_loss` for losing). Progress is rewarded through the shaping
/// `F = gamma * Phi(s') - Phi(s)` ([`potential`]). `Phi` is *not* zeroed at
/// terminal states: with `Phi(terminal) = 0` a lost game would hand back all
/// its progress, every loss would return the same, and a policy that never
/// wins would get no learning signal. So the episode return is the true
/// return plus `gamma^T * Phi(s_T)`, the discounted shaping potential of the
/// final state. With `terminal_phi = false` Phi *is* zeroed there (classic
/// shaping: exactly P(win)).
///
/// With [`RewardMode::Blinds`], there is no shaping potential: each blind
/// beaten pays 1 (skipped blinds are paid with the next beaten one), a win
/// pays `log10` of the final round's chips, and a loss pays the fraction of
/// the lost blind's chips scored.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Reward {
	pub(crate) mode: RewardMode,
	pub(crate) gamma: f32,
	pub(crate) r_win: f32,
	pub(crate) r_endless: f32,
	pub(crate) r_loss: f32,
	pub(crate) w_blind: f32,
	pub(crate) w_score: f32,
	pub(crate) terminal_phi: bool,
}

/// Which reward [`Reward`] pays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RewardMode {
	/// The true objective plus potential shaping.
	Shaping,
	/// Blinds beaten, with no shaping.
	Blinds,
}

impl Reward {
	fn potential(&self, g: &SimGame) -> f32 {
		match self.mode {
			RewardMode::Shaping => potential(g, self.w_blind, self.w_score),
			RewardMode::Blinds => 0.0,
		}
	}

	fn true_reward(&self, blinds_before: u32, won_before: bool, g: &SimGame) -> f32 {
		let mut r = 0.0;
		if g.run_won() && !won_before {
			r += self.r_win;
		} else if won_before {
			r += self.r_endless
				* num::feat(f64::from(
					g.run.blinds_cleared.saturating_sub(blinds_before),
				));
		}
		if g.flow.phase == Phase::GameOver || g.is_stuck() {
			r -= self.r_loss;
		}
		r
	}

	fn blinds_reward(
		blinds_before: u32, skips_before: f64, won_before: bool, deferred_skips: &mut u32,
		g: &SimGame,
	) -> f32 {
		let delta_cleared = g
			.run
			.blinds_cleared
			.checked_sub(blinds_before)
			.expect("the count of cleared blinds only grows");
		let delta_skips = num::whole(g.run.skips - skips_before)
			.and_then(|skips| u32::try_from(skips).ok())
			.expect("the count of skipped blinds is whole and only grows");
		let delta_beaten = delta_cleared
			.checked_sub(delta_skips)
			.expect("a skipped blind counts as cleared");
		let mut r = 0.0f32;
		if delta_skips > 0 {
			*deferred_skips += delta_skips;
		}
		if delta_beaten > 0 {
			r += num::feat(f64::from(delta_beaten + *deferred_skips));
			*deferred_skips = 0;
		}
		if g.run_won() && !won_before {
			r += num::feat(g.round.chips.max(1.0).log10());
		}
		if (g.flow.phase == Phase::GameOver || g.is_stuck()) && g.blind.chips > 0.0 {
			r += num::feat((g.round.chips / g.blind.chips).clamp(0.0, 1.0));
		}
		r
	}
}

#[derive(Clone, Copy)]
pub(crate) struct Cfg {
	pub(crate) endless: bool,
	pub(crate) max_ante: i32,
	pub(crate) step_limit: u32,
	pub(crate) reward: Reward,
	pub(crate) obs: ObsOptions,
	/// The share of played hands that get a joker-order label.
	pub(crate) order_label_rate: f32,
	pub(crate) snapshot_prob: f32,
	pub(crate) snapshot_min_ante: i32,
	pub(crate) practice_prob: f32,
	pub(crate) practice_eternal: bool,
	/// The share of episodes that start from the external start pool.
	pub(crate) start_frac: f32,
}

/// Applies the env's settings to a game it is about to run.
fn configure(g: &mut SimGame, cfg: &Cfg) {
	g.set_tracing(false);
	g.setup.endless = cfg.endless;
}

/// The antes whose starts the practice pool takes: later ante starts mostly
/// have no joker slot free for a gift.
pub(crate) const PRACTICE_ANTES: RangeInclusive<i32> = 2..=3;
/// The planets of the nine hand types every run can play.
const GIFT_PLANETS: usize = 9;
const GIFT_LEVELS: f64 = 3.0;
/// Keeps the label sampling apart from the snapshot offers (the same hash
/// otherwise).
const LABEL_SALT: u64 = 0x6f72_6465_725f_6c62;
/// Snapshots kept for restarts, per pool (a pool follows the current
/// policy's state distribution).
const RESERVOIR: usize = 4096;

/// Run `slot`'s seed index after `resets` resets of a `VecEnv` of `n_slots`
/// envs started from `base_seed`.
pub(crate) fn run_number(base_seed: u64, resets: u64, n_slots: u64, slot: usize) -> u64 {
	base_seed
		.wrapping_mul(1_000_003)
		.wrapping_add(resets * n_slots + num::index64(slot))
}

/// A pool of ante starts that episodes restart from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pool {
	/// Late ante starts (`snapshot_min_ante` on), restarted from as they are.
	Late,
	/// Early ante starts (`PRACTICE_ANTES`), restarted from with a gift.
	Practice,
}

/// The snapshots kept for restarts.
#[derive(Default)]
pub(crate) struct Reservoirs {
	late: Vec<SimGame>,
	practice: Vec<SimGame>,
	/// Snapshots offered so far.
	offered: u64,
}

impl Reservoirs {
	/// Number of snapshots kept, over both pools.
	pub(crate) fn len(&self) -> usize {
		self.late.len() + self.practice.len()
	}

	/// Keeps `snapshot` in the reservoir of its pool, where it replaces a
	/// random entry once the reservoir is full.
	pub(crate) fn offer(&mut self, snapshot: Snapshot, base_seed: u64) {
		self.offered += 1;
		let reservoir = match snapshot.pool {
			Pool::Late => &mut self.late,
			Pool::Practice => &mut self.practice,
		};
		if reservoir.len() < RESERVOIR {
			reservoir.push(*snapshot.game);
		} else {
			let k = num::pick(splitmix64(self.offered ^ base_seed), RESERVOIR);
			reservoir[k] = *snapshot.game;
		}
	}

	fn pool(&self, pool: Pool) -> &[SimGame] {
		match pool {
			Pool::Late => &self.late,
			Pool::Practice => &self.practice,
		}
	}
}

/// Where an episode's game came from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Origin {
	/// A new run, or a game handed in by `set_game`.
	Fresh,
	/// A snapshot of the pool, with its hidden state resampled.
	Snapshot(Pool),
	/// A state from the external `start_pool`, with its hidden state resampled.
	StartPool,
}

pub(crate) struct Slot {
	g: SimGame,
	/// The ops of the current observation (to decode the next choice).
	ops: Vec<Op>,
	steps: u32,
	ret: f32,
	phi: f32,
	/// Skipped blinds not yet paid (`RewardMode::Blinds`).
	deferred_skips: u32,
	resets: u64,
	origin: Origin,
	/// Recent build previews (reused while the build is unchanged).
	previews: Vec<BuildPreview>,
	jokers_seen: JokerSet,
	hand_plays: HandCounts<u32>,
	best_hand_score: f64,
}

pub(crate) struct StepCtx<'a> {
	pub(crate) cfg: Cfg,
	pub(crate) backs: &'a [Back],
	pub(crate) stakes: &'a [Stake],
	pub(crate) base_seed: u64,
	pub(crate) n_slots: u64,
	pub(crate) reservoirs: &'a Reservoirs,
	pub(crate) start_pool: &'a [SimGame],
	/// Probability of offering an ante start to its reservoir.
	pub(crate) p_offer: f32,
}

/// One env's choice: an op index, a card mask and a destination.
#[derive(Clone, Copy)]
pub(crate) struct Choice {
	op: i32,
	cards: u32,
	dest: i32,
}

/// One step's choices of every env.
#[derive(Clone, Copy)]
pub(crate) struct Choices<'a> {
	pub(crate) op: &'a [i32],
	pub(crate) cards: &'a [u32],
	pub(crate) dest: &'a [i32],
}

impl Choices<'_> {
	pub(crate) fn of(self, i: usize) -> Choice {
		Choice {
			op: self.op[i],
			cards: self.cards[i],
			dest: self.dest[i],
		}
	}
}

impl Slot {
	fn new(g: SimGame, origin: Origin, reward: &Reward) -> Slot {
		let mut jokers_seen = JokerSet::default();
		jokers_seen.mark(&g);
		Slot {
			phi: reward.potential(&g),
			g,
			ops: Vec::with_capacity(obs::MAX_OPS),
			steps: 0,
			ret: 0.0,
			deferred_skips: 0,
			resets: 0,
			origin,
			previews: Vec::new(),
			jokers_seen,
			hand_plays: HandCounts::default(),
			best_hand_score: 0.0,
		}
	}

	/// Starts the slot's next episode on `g`.
	fn install(&mut self, g: SimGame, origin: Origin, reward: &Reward) {
		self.g = g;
		self.origin = origin;
		self.steps = 0;
		self.ret = 0.0;
		self.phi = reward.potential(&self.g);
		self.deferred_skips = 0;
		self.jokers_seen = JokerSet::default();
		self.jokers_seen.mark(&self.g);
		self.hand_plays = HandCounts::default();
		self.best_hand_score = 0.0;
	}

	/// Starts the slot's next episode on `g`, a game handed in from outside,
	/// under the env's settings.
	pub(crate) fn set_game(&mut self, mut g: SimGame, cfg: &Cfg) {
		configure(&mut g, cfg);
		self.install(g, Origin::Fresh, &cfg.reward);
	}

	/// The ante the game is in.
	pub(crate) fn ante(&self) -> i32 {
		self.g.run.ante
	}

	/// A copy of the game as it stands.
	pub(crate) fn game_copy(&self) -> SimGame {
		self.g.clone()
	}

	/// Writes the observation of the game to `row`, and keeps its ops to
	/// decode the next choice with.
	pub(crate) fn observe(&mut self, row: &mut Rows<'_>, options: ObsOptions) {
		row.write(&mut self.g, options, &mut self.ops, &mut self.previews);
	}

	/// Uniform in [0, 1) from a hash of `seed`, the env `i`, its game count
	/// and its step count: no RNG is consumed.
	fn roll(&self, seed: u64, i: usize) -> f32 {
		let n = splitmix64(seed ^ num::index64(i));
		unit(n ^ (self.resets << 20) ^ u64::from(self.steps))
	}

	/// Applies the choice of env `i`. If the episode ended, the env is reset;
	/// a truncated episode's last observation is written to `final_row`
	/// first.
	pub(crate) fn step(
		&mut self, i: usize, choice: Choice, ctx: &StepCtx<'_>, final_row: &mut Rows<'_>,
	) -> StepOut {
		let (cfg, rw) = (ctx.cfg, ctx.cfg.reward);
		let mut res = StepOut::default();
		let blinds_before = self.g.run.blinds_cleared;
		let skips_before = self.g.run.skips;
		let won_before = self.g.run_won();
		let was_ante_start = ante_start(&self.g, cfg.endless);
		let a = usize::try_from(choice.op)
			.ok()
			.and_then(|k| self.ops.get(k))
			.and_then(|o| {
				let cards = CardMask::from_bits(choice.cards);
				obs::decode(&self.g, o, cards, u8::try_from(choice.dest).ok())
			});
		if let Some(Action::Play(mask)) = a
			&& cfg.order_label_rate > 0.0
			&& self.roll(ctx.base_seed ^ LABEL_SALT, i) < cfg.order_label_rate
		{
			res.label = obs::order_label(&self.g, mask, cfg.obs.ops.max_rearranges);
		}
		let a = if let Some(a) = a {
			Some(a)
		} else {
			res.invalid = true;
			obs::fallback(&self.g, &self.ops)
		};
		let pending_cons = a.and_then(|act| consumable_event(&self.g, act));
		// The known state without a legal action (the game's own pack softlock) is
		// terminal via `is_stuck`. Any other one is a simulator bug, as is an
		// offered action the game then rejects: dump the state and truncate the
		// episode rather than kill a long training run.
		let dead_end = if let Some(act) = a {
			match self.g.step(act) {
				Ok(()) => {
					self.steps += 1;
					res.cons_event = pending_cons;
					self.record_action(act);
					false
				}
				Err(e) => {
					eprintln!(
						"[vecenv] {act:?} was offered and then rejected ({e}), episode truncated. ops: {:?}\n{:#?}",
						self.ops, self.g
					);
					true
				}
			}
		} else {
			eprintln!(
				"[vecenv] dead end: no legal action, episode truncated. ops: {:?}\n{:#?}",
				self.ops, self.g
			);
			true
		};
		let g = &self.g;
		let r_true = match rw.mode {
			RewardMode::Shaping => rw.true_reward(blinds_before, won_before, g),
			RewardMode::Blinds => Reward::blinds_reward(
				blinds_before,
				skips_before,
				won_before,
				&mut self.deferred_skips,
				g,
			),
		};
		let terminal = g.is_terminal();
		let end = if terminal {
			Some(End::Terminal)
		} else if dead_end
			|| self.steps >= cfg.step_limit
			|| (cfg.endless && g.run.ante > cfg.max_ante)
		{
			Some(End::Truncated)
		} else {
			None
		};
		let phi = if terminal && !rw.terminal_phi {
			0.0
		} else {
			rw.potential(g)
		};
		res.reward = r_true + rw.gamma * phi - self.phi;
		self.phi = phi;
		self.ret += r_true;
		res.end = end;
		if end.is_none()
			&& ctx.p_offer > 0.0
			&& let Some(pool) = snapshot_pool(g, &cfg, self.origin, !was_ante_start)
			&& self.roll(ctx.base_seed, i) < ctx.p_offer
		{
			res.snapshot = Some(Snapshot {
				pool,
				game: Box::new(g.clone()),
			});
		}
		if let Some(end) = end {
			self.finish_episode(i, end, ctx, final_row, &mut res);
		}
		res
	}

	fn record_action(&mut self, act: Action) {
		if matches!(act, Action::Play(_)) {
			let played = self.g.history.last_hand_played;
			self.hand_plays[played.expect("the game names a played hand")] += 1;
			if self.g.run.last_hand_score > self.best_hand_score {
				self.best_hand_score = self.g.run.last_hand_score;
			}
		}
		self.jokers_seen.mark(&self.g);
	}

	/// Reports the episode that just ended and starts the slot's next one.
	fn finish_episode(
		&mut self, i: usize, end: End, ctx: &StepCtx<'_>, final_row: &mut Rows<'_>,
		res: &mut StepOut,
	) {
		let g = &self.g;
		res.ep_stats = Some(EpisodeStatsOut {
			back: g.setup.back,
			stake: g.setup.stake.into(),
			ante: g.run.ante,
			won: g.run_won(),
			truncated: end == End::Truncated,
			blinds_cleared: g.run.blinds_cleared,
			jokers_seen: self.jokers_seen,
			hand_plays: self.hand_plays,
			best_hand_score: self.best_hand_score,
		});
		res.episode = Some(Episode {
			slot: i,
			seed: g.rng.seed().to_string(),
			back: g.setup.back,
			stake: g.setup.stake.into(),
			ante: g.run.ante,
			won: g.run_won(),
			blinds_cleared: g.run.blinds_cleared,
			steps: self.steps,
			ret: self.ret,
			truncated: end == End::Truncated,
			from_snapshot: self.origin != Origin::Fresh,
		});
		if end == End::Truncated {
			self.observe(final_row, ctx.cfg.obs);
		}
		self.resets += 1;
		let n = run_number(ctx.base_seed, self.resets, ctx.n_slots, i);
		let (g, origin) = start(
			n,
			ctx.backs,
			ctx.stakes,
			&ctx.cfg,
			ctx.reservoirs,
			ctx.start_pool,
		);
		self.install(g, origin, &ctx.cfg.reward);
	}
}

pub(crate) struct Episode {
	pub(crate) slot: usize,
	pub(crate) seed: String,
	pub(crate) back: Back,
	pub(crate) stake: u8,
	pub(crate) ante: i32,
	/// The run was won (not the game's `won` flag, which is also set when
	/// the final boss is lost).
	pub(crate) won: bool,
	pub(crate) blinds_cleared: u32,
	pub(crate) steps: u32,
	/// The true (unshaped) return.
	pub(crate) ret: f32,
	pub(crate) truncated: bool,
	pub(crate) from_snapshot: bool,
}

/// How an episode ended.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum End {
	/// The run was won (outside endless mode) or lost.
	Terminal,
	/// It was cut off (step limit, ante limit or a dead end): bootstrap from
	/// the final state.
	Truncated,
}

/// An ante start to offer the reservoir of its pool.
pub(crate) struct Snapshot {
	pool: Pool,
	game: Box<SimGame>,
}

#[derive(Default)]
pub(crate) struct StepOut {
	/// The shaped reward.
	pub(crate) reward: f32,
	pub(crate) end: Option<End>,
	/// The choice was illegal and replaced by a fallback.
	pub(crate) invalid: bool,
	pub(crate) episode: Option<Episode>,
	pub(crate) ep_stats: Option<EpisodeStatsOut>,
	pub(crate) cons_event: Option<((&'static str, &'static str), ConsEvent)>,
	pub(crate) snapshot: Option<Snapshot>,
	/// The joker-order label of the play taken (`obs::order_label`), if
	/// sampled.
	pub(crate) label: Option<[u8; obs::ORDER_JOKERS]>,
}

fn make_game(n: u64, backs: &[Back], stakes: &[Stake], cfg: &Cfg) -> SimGame {
	let h = n.wrapping_mul(0xd6e8_feb8_6659_fd93);
	let back = backs[num::pick(h, backs.len())];
	let stake = stakes[num::pick(h >> 20, stakes.len())];
	let mut g = SimGame::new_run(&seed_string(n), back, stake);
	configure(&mut g, cfg);
	g
}

/// A fresh run, a reservoir snapshot whose hidden state is resampled (a
/// late ante start with probability `snapshot_prob`, or an early one with a
/// gift with probability `practice_prob`), or a state from the external
/// `start_pool` (with probability `start_frac`), its hidden state resampled.
fn start(
	n: u64, backs: &[Back], stakes: &[Stake], cfg: &Cfg, reservoirs: &Reservoirs,
	start_pool: &[SimGame],
) -> (SimGame, Origin) {
	let u = unit(n ^ 0x5eed);
	let pool = if u < cfg.snapshot_prob {
		Some(Pool::Late)
	} else if u < cfg.snapshot_prob + cfg.practice_prob {
		Some(Pool::Practice)
	} else {
		None
	};
	if let Some(pool) = pool.filter(|&pool| !reservoirs.pool(pool).is_empty()) {
		let from = reservoirs.pool(pool);
		let mut g = from[num::pick(splitmix64(n ^ 0xa11), from.len())].clone();
		g.resample_hidden(&seed_string(n));
		configure(&mut g, cfg);
		if pool == Pool::Practice {
			gift(&mut g, n, cfg.practice_eternal);
		}
		return (g, Origin::Snapshot(pool));
	}
	if u >= cfg.snapshot_prob + cfg.practice_prob
		&& u < cfg.snapshot_prob + cfg.practice_prob + cfg.start_frac
	{
		let mut g = start_pool[num::pick(splitmix64(n ^ 0xa11), start_pool.len())].clone();
		g.resample_hidden(&seed_string(n));
		configure(&mut g, cfg);
		return (g, Origin::StartPool);
	}
	(make_game(n, backs, stakes, cfg), Origin::Fresh)
}

/// The first games of a new `VecEnv`'s `num_envs` slots.
pub(crate) fn make_slots(
	num_envs: usize, seed: u64, backs: &[Back], stakes: &[Stake], cfg: &Cfg, start_pool: &[SimGame],
) -> Vec<Slot> {
	let empty_res = Reservoirs::default();
	let n_slots = num::index64(num_envs);
	(0..num_envs)
		.into_par_iter()
		.map(|i| {
			let (g, origin) = start(
				run_number(seed, 0, n_slots, i),
				backs,
				stakes,
				cfg,
				&empty_res,
				start_pool,
			);
			Slot::new(g, origin, &cfg.reward)
		})
		.collect()
}

/// Something to practise with that the run did not choose: a joker (half
/// the time, if a slot is free and it holds none of the kind; Eternal if
/// `eternal` and the joker can be), or `GIFT_LEVELS` levels of one hand type.
fn gift(g: &mut SimGame, n: u64, eternal: bool) {
	let h = splitmix64(n ^ 0x61f7);
	let joker = ALL_JOKERS[num::pick(h >> 1, ALL_JOKERS.len())];
	if h & 1 == 0 && g.areas.joker_room() && !g.areas.jokers.iter().any(|c| c.is_joker(joker)) {
		g.inject(Inject::Joker {
			joker,
			edition: None,
			eternal,
		});
	} else {
		let planet = ALL_PLANETS[num::pick(splitmix64(h), GIFT_PLANETS)];
		g.hands.level_up(planet.hand(), GIFT_LEVELS);
	}
}

/// Whether `g` is at the start of an ante, before its first blind is chosen.
fn ante_start(g: &SimGame, endless: bool) -> bool {
	g.flow.phase == Phase::BlindSelect
		&& g.blind_select.states[BlindSlot::Small] == BlindStatus::Select
		&& !g.is_terminal()
		&& (endless || !g.run_won())
}

/// The pool the ante start `g` belongs to, if any. `origin`: where its
/// episode's game came from; `arrived`: the last action led here. Practice
/// episodes feed neither pool, and only fresh runs feed the practice pool,
/// once per ante: a practice start holds one gift, a late start none, and
/// what a run does before choosing its blind does not weigh the pool.
fn snapshot_pool(g: &SimGame, cfg: &Cfg, origin: Origin, arrived: bool) -> Option<Pool> {
	if !ante_start(g, cfg.endless) {
		None
	} else if cfg.snapshot_prob > 0.0 && g.run.ante >= cfg.snapshot_min_ante {
		(origin != Origin::Snapshot(Pool::Practice)).then_some(Pool::Late)
	} else if cfg.practice_prob > 0.0 && arrived && PRACTICE_ANTES.contains(&g.run.ante) {
		(origin == Origin::Fresh).then_some(Pool::Practice)
	} else {
		None
	}
}
