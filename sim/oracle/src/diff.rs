//! Randomized differential testing: play the same actions in the simulator
//! and in the real game and compare the canonical state (and the timeline of
//! RNG calls, draws and removals) after every step.

use sim::actions::Action;
use sim::card::{Card, Kind};
use sim::data::{
	ALL_JOKERS, ALL_PLANETS, ALL_SPECTRALS, ALL_TAROTS, ALL_VOUCHERS, Back, Edition, Enhancement,
	P_CARDS_SORTED, Seal, Voucher, rank_char,
};
use sim::game::{Game, Phase};
pub use sim::inject::Inject;
use sim::num;

use crate::canon::{self, Diff, TimelineDiff};
use crate::json::{self, Json};
use crate::{Oracle, lua_str};

/// Small deterministic RNG for the test policy (xorshift64).
pub struct Xs(pub u64);

impl Xs {
	pub fn next_u64(&mut self) -> u64 {
		let mut x = self.0;
		x ^= x << 13;
		x ^= x >> 7;
		x ^= x << 17;
		self.0 = x;
		x
	}

	pub fn below(&mut self, n: usize) -> usize {
		num::pick(self.next_u64(), n)
	}

	#[expect(
		clippy::cast_precision_loss,
		reason = "both operands are below 2^53, so the casts are exact"
	)]
	pub fn chance(&mut self, p: f64) -> bool {
		((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64) < p
	}
}

fn cards_list(m: u32) -> String {
	let v: Vec<String> = (0..32)
		.filter(|i| m >> i & 1 == 1)
		.map(|i| (i + 1).to_string())
		.collect();
	format!("{{{}}}", v.join(","))
}

/// The harness command starting a run.
#[must_use]
pub fn start_to_lua(seed: &str, back: Back, stake: u8) -> String {
	format!("H.start({},{},{stake})", lua_str(seed), lua_str(back.key()))
}

/// The harness command performing `a`.
#[must_use]
pub fn action_to_lua(a: &Action) -> String {
	use Action as A;
	let body = match *a {
		A::SelectBlind => "type='select_blind'".to_string(),
		A::SkipBlind => "type='skip_blind'".to_string(),
		A::RerollBoss => "type='reroll_boss'".to_string(),
		A::Play(m) => format!("type='play', cards={}", cards_list(m)),
		A::Discard(m) => format!("type='discard', cards={}", cards_list(m)),
		A::UseConsumable { index, targets } => format!(
			"type='use', index={}, cards={}",
			index + 1,
			cards_list(targets)
		),
		A::SellJoker(i) => format!("type='sell_joker', index={}", i + 1),
		A::SellConsumable(i) => format!("type='sell_consumable', index={}", i + 1),
		A::MoveJoker { from, to } => format!("type='move_joker', from={}, to={}", from + 1, to + 1),
		A::MoveHand { from, to } => format!("type='move_hand', from={}, to={}", from + 1, to + 1),
		A::SortHand { by_suit } => format!(
			"type='sort_hand', by='{}'",
			if by_suit { "suit" } else { "value" }
		),
		A::BuyShopCard(i) => format!("type='buy', index={}", i + 1),
		A::BuyAndUse { index, targets } => format!(
			"type='buy_and_use', index={}, cards={}",
			index + 1,
			cards_list(targets)
		),
		A::RedeemVoucher(i) => format!("type='redeem', index={}", i + 1),
		A::OpenBooster(i) => format!("type='open', index={}", i + 1),
		A::Reroll => "type='reroll'".to_string(),
		A::EndShop => "type='end_shop'".to_string(),
		A::CashOut => "type='cash_out'".to_string(),
		A::PickPackCard { index, targets } => format!(
			"type='pick', index={}, cards={}",
			index + 1,
			cards_list(targets)
		),
		A::SkipPack => "type='skip_pack'".to_string(),
	};
	format!("H.act({{{body}}})")
}

fn ed_lua(e: Edition) -> String {
	e.name().map_or("nil".into(), |n| format!("'{n}'"))
}

/// The harness command performing the injection `i`.
#[must_use]
pub fn inject_to_lua(i: &Inject) -> String {
	match *i {
		Inject::Joker {
			joker: j,
			edition: e,
			eternal: et,
		} => format!(
			"H.inject({{kind='joker', key='{}', edition={}, eternal={}}})",
			j.key(),
			ed_lua(e),
			et
		),
		Inject::Consumable {
			kind: k,
			edition: e,
		} => format!(
			"H.inject({{kind='consumable', key='{}', edition={}}})",
			Card::new(0, k).key(),
			ed_lua(e)
		),
		Inject::Card {
			suit: s,
			rank: r,
			enhancement: enh,
			edition: e,
			seal,
		} => format!(
			"H.inject({{kind='card', key='{}_{}', center='{}', edition={}, seal={}}})",
			s.letter() as char,
			rank_char(r) as char,
			enh.key(),
			ed_lua(e),
			seal.map_or("nil".into(), |x| format!("'{}'", x.name()))
		),
		Inject::Dollars(d) => format!("H.inject({{kind='dollars', amount={d}}})"),
		Inject::Voucher(v) => format!("H.inject({{kind='voucher', key='{}'}})", v.key()),
	}
}

/// Highest-scoring play found by stepping clones of the game.
fn best_play(g: &Game, plays: &[Action]) -> (Option<Action>, f64) {
	let mut best = None;
	let mut best_score = -1.0;
	for a in plays {
		let mut c = g.clone();
		if c.step(*a).is_ok() {
			let s = c.last_hand_score;
			if s > best_score {
				best_score = s;
				best = Some(*a);
			}
		}
	}
	(best, best_score)
}

/// Picks an action for the test policy.
#[expect(clippy::too_many_lines, reason = "the test policy, one arm per phase")]
pub fn choose(g: &Game, rng: &mut Xs, rerolls_this_shop: &mut u32) -> Option<Action> {
	let mut acts = Vec::new();
	g.legal_actions(&mut acts);
	if acts.is_empty() {
		return None;
	}
	let pick = |rng: &mut Xs, v: &[Action]| -> Option<Action> {
		if v.is_empty() {
			None
		} else {
			Some(v[rng.below(v.len())])
		}
	};
	let of = |f: &dyn Fn(&Action) -> bool| -> Vec<Action> {
		acts.iter().copied().filter(|a| f(a)).collect()
	};
	let uses = of(&|a| matches!(a, Action::UseConsumable { .. }));
	let sells = of(&|a| matches!(a, Action::SellJoker(_) | Action::SellConsumable(_)));
	// Occasional joker/hand reordering (not enumerated by legal_actions).
	if g.jokers.len() >= 2 && rng.chance(0.03) {
		let from = num::byte(rng.below(g.jokers.len()));
		let mut to = num::byte(rng.below(g.jokers.len()));
		if to == from {
			to = (to + 1) % num::byte(g.jokers.len());
		}
		return Some(Action::MoveJoker { from, to });
	}
	if !uses.is_empty() && rng.chance(0.15) {
		return pick(rng, &uses);
	}
	if !sells.is_empty() && rng.chance(0.03) {
		return pick(rng, &sells);
	}
	match g.phase {
		Phase::BlindSelect => {
			if rng.chance(0.08)
				&& let Some(a) = pick(rng, &of(&|a| matches!(a, Action::RerollBoss)))
			{
				return Some(a);
			}
			if rng.chance(0.2)
				&& let Some(a) = pick(rng, &of(&|a| matches!(a, Action::SkipBlind)))
			{
				return Some(a);
			}
			Some(Action::SelectBlind)
		}
		Phase::Round => {
			if g.hand.len() >= 2 && rng.chance(0.03) {
				let from = num::byte(rng.below(g.hand.len()));
				let mut to = num::byte(rng.below(g.hand.len()));
				if to == from {
					to = (to + 1) % num::byte(g.hand.len());
				}
				return Some(Action::MoveHand { from, to });
			}
			if rng.chance(0.02) {
				return Some(Action::SortHand {
					by_suit: rng.chance(0.5),
				});
			}
			let discards = of(&|a| matches!(a, Action::Discard(_)));
			if !discards.is_empty() && rng.chance(0.08) {
				return pick(rng, &discards);
			}
			let plays = of(&|a| matches!(a, Action::Play(_)));
			if rng.chance(0.85) {
				// Look ahead on snapshots: play the highest-scoring hand, or
				// discard what it doesn't use when it can't win the round.
				let (best, best_score) = best_play(g, &plays);
				let needed = g.blind.chips - g.chips;
				if !discards.is_empty() && best_score < needed && rng.chance(0.75) {
					if let Some(Action::Play(m)) = best {
						let forced: u32 = (0..g.hand.len())
							.filter(|&i| g.hand[i].ab.forced_selection)
							.fold(0, |a, i| a | 1 << i);
						let mut d = 0u32;
						for i in (0..g.hand.len()).rev() {
							if (m >> i & 1 == 0) && d.count_ones() < 5 {
								d |= 1 << i;
							}
						}
						d |= forced;
						if discards.contains(&Action::Discard(d)) {
							return Some(Action::Discard(d));
						}
					}
					return pick(rng, &discards);
				}
				return best;
			}
			pick(rng, &plays)
		}
		Phase::Shop => {
			let buys = of(&|a| matches!(a, Action::BuyShopCard(_) | Action::BuyAndUse { .. }));
			let reds = of(&|a| matches!(a, Action::RedeemVoucher(_)));
			let opens = of(&|a| matches!(a, Action::OpenBooster(_)));
			if !buys.is_empty() && rng.chance(0.45) {
				return pick(rng, &buys);
			}
			if !reds.is_empty() && rng.chance(0.4) {
				return pick(rng, &reds);
			}
			if !opens.is_empty() && rng.chance(0.4) {
				return pick(rng, &opens);
			}
			if *rerolls_this_shop < 3 && rng.chance(0.25) && acts.contains(&Action::Reroll) {
				*rerolls_this_shop += 1;
				return Some(Action::Reroll);
			}
			Some(Action::EndShop)
		}
		Phase::Pack => {
			let picks = of(&|a| matches!(a, Action::PickPackCard { .. }));
			if !picks.is_empty() && rng.chance(0.85) {
				return pick(rng, &picks);
			}
			if acts.contains(&Action::SkipPack) {
				return Some(Action::SkipPack);
			}
			pick(rng, &acts)
		}
		Phase::RoundEval => Some(Action::CashOut),
		Phase::GameOver => None,
	}
}

fn random_edition(rng: &mut Xs) -> Edition {
	match rng.below(10) {
		0 => Edition::Foil,
		1 => Edition::Holo,
		2 => Edition::Polychrome,
		3 => Edition::Negative,
		_ => Edition::None,
	}
}

/// A random scenario injection that fits `g` (or `None`).
pub fn random_inject(g: &Game, rng: &mut Xs) -> Option<Inject> {
	let r = rng.below(100);
	if r < 45 {
		if num::float(g.jokers.len()) < g.joker_limit.limit {
			let j = ALL_JOKERS[rng.below(ALL_JOKERS.len())];
			let e = random_edition(rng);
			return Some(Inject::Joker {
				joker: j,
				edition: e,
				eternal: rng.chance(0.1),
			});
		}
		None
	} else if r < 70 {
		if num::float(g.consumables.len()) < g.consumable_limit.limit {
			let k = match rng.below(3) {
				0 => Kind::Tarot(ALL_TAROTS[rng.below(22)]),
				1 => Kind::Planet(ALL_PLANETS[rng.below(12)]),
				_ => Kind::Spectral(ALL_SPECTRALS[rng.below(18)]),
			};
			let e = if rng.chance(0.1) {
				Edition::Negative
			} else {
				Edition::None
			};
			return Some(Inject::Consumable {
				kind: k,
				edition: e,
			});
		}
		None
	} else if r < 88 {
		let (s, rk) = P_CARDS_SORTED[rng.below(52)];
		let enh = if rng.chance(0.7) {
			Enhancement::POOL[rng.below(8)]
		} else {
			Enhancement::Base
		};
		let e = match rng.below(8) {
			0 => Edition::Foil,
			1 => Edition::Holo,
			2 => Edition::Polychrome,
			_ => Edition::None,
		};
		let seal = match rng.below(8) {
			0 => Some(Seal::Gold),
			1 => Some(Seal::Red),
			2 => Some(Seal::Blue),
			3 => Some(Seal::Purple),
			_ => None,
		};
		Some(Inject::Card {
			suit: s,
			rank: rk,
			enhancement: enh,
			edition: e,
			seal,
		})
	} else if r < 96 {
		Some(Inject::Dollars(num::float(5 + rng.below(30))))
	} else {
		let v = ALL_VOUCHERS[rng.below(32)];
		if g.voucher_used(v) || matches!(v, Voucher::RerollSurplus | Voucher::RerollGlut) {
			return None;
		}
		Some(Inject::Voucher(v))
	}
}

/// What is compared besides the canonical state.
#[derive(Clone, Copy, Debug)]
pub struct Checks {
	/// The order of the timeline entries (RNG calls, draws, removals).
	pub timeline: bool,
	/// Also fail when a timeline entry happens on another frame (otherwise
	/// such steps are only counted, see [`Stats`]).
	pub frames: bool,
}

impl Checks {
	/// From the environment: `ORACLE_TIMELINE=0` skips the timeline,
	/// `ORACLE_FRAMES=1` makes frame offsets failures.
	///
	/// # Errors
	///
	/// Returns which variable has a bad value, or that the two contradict.
	pub fn from_env() -> Result<Checks, String> {
		let flag = |name: &str, default: bool| match std::env::var(name).as_deref() {
			Err(_) => Ok(default),
			Ok("1") => Ok(true),
			Ok("0") => Ok(false),
			Ok(v) => Err(format!("{name} must be 0 or 1, got {v:?}")),
		};
		let c = Checks {
			timeline: flag("ORACLE_TIMELINE", true)?,
			frames: flag("ORACLE_FRAMES", false)?,
		};
		if c.frames && !c.timeline {
			return Err("ORACLE_FRAMES=1 needs the timeline (ORACLE_TIMELINE=1)".into());
		}
		Ok(c)
	}
}

impl std::fmt::Display for Checks {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match (self.timeline, self.frames) {
			(false, _) => write!(f, "state only (ORACLE_TIMELINE=0)"),
			(true, false) => write!(
				f,
				"state + timeline order; frame offsets counted, not failed (ORACLE_FRAMES=1 to fail)"
			),
			(true, true) => write!(f, "state + timeline order + frames"),
		}
	}
}

/// How the engines disagree after a step.
pub enum Mismatch {
	/// Canonical state differences `(path, sim, game)`.
	State(Vec<Diff>),
	Timeline(TimelineDiff),
}

/// Compares the simulator with the oracle's state dump after a step.
pub fn compare(g: &mut Game, lua: &Json, timeline: bool) -> Option<Mismatch> {
	let d = canon::diff(&canon::from_sim(g), &canon::from_lua(lua));
	if !d.is_empty() {
		return Some(Mismatch::State(d));
	}
	if timeline {
		canon::timeline_diff(g, lua).map(Mismatch::Timeline)
	} else {
		None
	}
}

pub struct Failure {
	pub seed: String,
	pub step: usize,
	pub history: Vec<String>,
	/// Machine-readable history (`start <seed> <back> <stake>` then step codes).
	pub codes: Vec<String>,
	pub diffs: Vec<Diff>,
	pub error: Option<String>,
	/// Statistics up to the failure.
	pub stats: Stats,
}

impl std::fmt::Display for Failure {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		writeln!(f, "seed {} diverged at step {}", self.seed, self.step)?;
		if let Some(e) = &self.error {
			writeln!(f, "error: {e}")?;
		}
		let n = self.history.len();
		for h in &self.history[n.saturating_sub(12)..] {
			writeln!(f, "  {h}")?;
		}
		for d in self.diffs.iter().take(40) {
			writeln!(f, "  {d}")?;
		}
		if self.diffs.len() > 40 {
			writeln!(f, "  ... {} more", self.diffs.len() - 40)?;
		}
		Ok(())
	}
}

#[derive(Default, Debug, Clone)]
pub struct Stats {
	/// Commands compared (actions and injections).
	pub steps: usize,
	pub max_ante: i32,
	pub won: bool,
	/// Steps whose timeline order was compared and matched.
	pub timed_steps: usize,
	/// Of those, steps with an entry on another frame, and a few examples
	/// (`command :: first differing entry`).
	pub frame_mismatches: usize,
	pub frame_examples: Vec<String>,
}

/// One differential game.
pub struct GameSpec<'a> {
	pub seed: &'a str,
	pub back: Back,
	pub stake: u8,
	pub max_steps: usize,
	/// Seed of the test policy.
	pub policy_seed: u64,
	/// Probability of a random scenario injection before each action.
	pub inject_rate: f64,
}

/// A game in progress in both engines.
struct Run<'a> {
	o: &'a Oracle,
	checks: Checks,
	seed: &'a str,
	history: Vec<String>,
	codes: Vec<String>,
	stats: Stats,
}

impl Run<'_> {
	fn fail(&self, step: usize, diffs: Vec<Diff>, error: Option<String>) -> Box<Failure> {
		Box::new(Failure {
			seed: self.seed.to_string(),
			step,
			history: self.history.clone(),
			codes: self.codes.clone(),
			diffs,
			error,
			stats: self.stats.clone(),
		})
	}

	/// Sends `cmd` to the oracle (the simulator has already performed it)
	/// and compares the engines.
	fn sync(
		&mut self, g: &mut Game, step: usize, cmd: &str, what: &str,
	) -> Result<(), Box<Failure>> {
		let lua = self
			.o
			.call("oracle", cmd)
			.map_err(|e| self.fail(step, vec![], Some(format!("oracle: {e}"))))?;
		let j =
			json::parse(&lua).map_err(|e| self.fail(step, vec![], Some(format!("json: {e}"))))?;
		match compare(g, &j, self.checks.timeline) {
			Some(Mismatch::State(d)) => return Err(self.fail(step, d, None)),
			Some(Mismatch::Timeline(TimelineDiff::Order(d))) => {
				return Err(self.fail(step, vec![d], Some("timeline order differs".into())));
			}
			Some(Mismatch::Timeline(TimelineDiff::Frame(m))) => {
				self.stats.timed_steps += 1;
				self.stats.frame_mismatches += 1;
				if self.stats.frame_examples.len() < 400 {
					self.stats.frame_examples.push(format!("{what} :: {m}"));
				}
				if self.checks.frames {
					return Err(self.fail(
						step,
						vec![Diff {
							path: "timeline frames".into(),
							sim: m,
							game: String::new(),
						}],
						Some("timeline frames differ".into()),
					));
				}
			}
			None if self.checks.timeline => self.stats.timed_steps += 1,
			None => {}
		}
		self.stats.steps = step;
		self.stats.max_ante = self.stats.max_ante.max(g.ante);
		self.stats.won |= g.run_won();
		Ok(())
	}
}

/// Plays one game in both engines with the test policy, comparing after
/// every action and injection.
///
/// # Errors
///
/// Returns the first divergence (or oracle failure) with the game's history.
pub fn run_game(o: &Oracle, spec: &GameSpec<'_>, checks: Checks) -> Result<Stats, Box<Failure>> {
	let &GameSpec {
		seed,
		back,
		stake,
		max_steps,
		policy_seed,
		inject_rate,
	} = spec;
	let mut g = Game::new_run(seed, back, stake);
	let start = start_to_lua(seed, back, stake);
	let mut run = Run {
		o,
		checks,
		seed,
		history: vec![start.clone()],
		codes: vec![format!("start {seed} {} {stake}", back.key())],
		stats: Stats {
			max_ante: g.ante,
			..Stats::default()
		},
	};
	run.sync(&mut g, 0, &start, "start")?;
	let mut rng = Xs(policy_seed | 1);
	let mut rerolls = 0;
	for step in 1..=max_steps {
		if g.phase != Phase::Shop {
			rerolls = 0;
		}
		if inject_rate > 0.0
			&& g.phase != Phase::GameOver
			&& rng.chance(inject_rate)
			&& let Some(inj) = random_inject(&g, &mut rng)
		{
			let cmd = inject_to_lua(&inj);
			let what = format!("{inj:?}");
			run.history.push(format!("{cmd}  -- {what}"));
			run.codes.push(inj.to_string());
			g.inject(&inj);
			run.sync(&mut g, step, &cmd, &what)?;
			continue;
		}
		let Some(a) = choose(&g, &mut rng, &mut rerolls) else {
			break;
		};
		let cmd = action_to_lua(&a);
		let what = format!("{a:?}");
		run.history.push(format!("{cmd}  -- {what}"));
		run.codes.push(a.to_string());
		if let Err(e) = g.step(a) {
			return Err(run.fail(step, vec![], Some(format!("sim rejected {a:?}: {e}"))));
		}
		run.sync(&mut g, step, &cmd, &what)?;
		if g.phase == Phase::GameOver {
			break;
		}
	}
	Ok(run.stats)
}
