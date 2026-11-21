//! The state dump importer (`Game::from_dump`) checked against the real
//! game. A differential game is played in both engines (as in
//! [`crate::difftest`]); at random decision points the game's state dump is
//! imported, and the imported game must
//!
//! * show the dump's canonical state,
//! * equal the simulator that reached the point in lockstep, part by part
//!   (see [`part_diffs`]), and give its learning observation,
//! * stay equal to the game for the rest of the game under the same
//!   commands.
//!
//! There the game is also dumped by the mod's own writer (`mod/src/state.lua`,
//! which the bot imports in the real game), and both dumps must be the same;
//! and the simulator's own dump of the same point must read as the same
//! canonical state.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;
use std::path::Path;

use sim::data::Booster;
use sim::game::{Areas, Flow, Game, PackSlot, Phase, State};
use sim::json::{self, Json};
use sim::rng::Xorshift;
use sim::{AnyCard, CardLike, Kind};

use crate::canon::{self, Diff};
use crate::difftest::GameSpec;
use crate::fingerprint::obs_fingerprint;
use crate::policy::Policy;
use crate::{Oracle, lua_path};

/// What a checked game covered.
#[derive(Clone, Debug, Default)]
pub struct DumpStats {
	/// Commands played.
	pub steps: usize,
	/// Decision points imported, by phase.
	pub imports: BTreeMap<&'static str, usize>,
	/// Steps an imported game was compared with the game after its import.
	pub followed: usize,
	pub max_ante: i32,
}

/// Where `imported` differs from `lockstep` (the lines of their `Debug`
/// output that differ, by part), but for what the simulator keeps that a
/// dump cannot hold and that does not matter at a decision point (see
/// [`excused`]), and the per-action flags of [`Flow`] that the next action
/// resets before reading them. The RNG streams are the canonical state's.
pub fn part_diffs(imported: &mut Game, lockstep: &mut Game) -> Vec<String> {
	fn flow(f: Flow) -> (Phase, State, bool) {
		(f.phase, f.state, f.state_complete)
	}
	imported.update_derived();
	lockstep.update_derived();
	let (a, b) = (excused(imported), excused(lockstep));
	let mut out = Vec::new();
	part(&mut out, "setup", &a.setup, &b.setup);
	part(&mut out, "flow", &flow(a.flow), &flow(b.flow));
	part(
		&mut out,
		"areas",
		&without_cards(&a.areas),
		&without_cards(&b.areas),
	);
	let (x, y) = (&a.areas, &b.areas);
	cards(&mut out, "deck", &x.deck, &y.deck);
	cards(&mut out, "hand", &x.hand.cards, &y.hand.cards);
	cards(&mut out, "discard", &x.discard, &y.discard);
	cards(&mut out, "jokers", &x.jokers, &y.jokers);
	cards(&mut out, "consumables", &x.consumables, &y.consumables);
	cards(&mut out, "shop_jokers", &x.shop_jokers, &y.shop_jokers);
	cards(
		&mut out,
		"shop_vouchers",
		&x.shop_vouchers,
		&y.shop_vouchers,
	);
	cards(
		&mut out,
		"shop_boosters",
		&x.shop_boosters,
		&y.shop_boosters,
	);
	cards(&mut out, "pack_cards", &x.pack_cards, &y.pack_cards);
	part(&mut out, "run", &a.run, &b.run);
	part(&mut out, "history", &a.history, &b.history);
	part(&mut out, "round", &a.round, &b.round);
	part(&mut out, "blind_select", &a.blind_select, &b.blind_select);
	part(&mut out, "blind", &a.blind, &b.blind);
	part(&mut out, "shop", &a.shop, &b.shop);
	part(&mut out, "pack", &a.pack, &b.pack);
	part(&mut out, "hands", &a.hands, &b.hands);
	part(&mut out, "tags", &a.tags, &b.tags);
	out
}

/// A copy of `g` without what a dump cannot hold and does not matter at a
/// decision point:
///
/// * whether a playing card was added to the deck: the engines mark it on
///   different paths, and only a Negative edition's slot reads it, an
///   edition no playing card gets in a run;
/// * the joker and consumable areas' `real` limits: only the hand changes
///   size, the game sets their limits directly;
/// * whether the blind triggered: the game's value is its warning for the
///   highlighted cards, and the next play resets it before anything reads
///   it;
/// * the last hand's score, a preview's result that every preview sets;
/// * the tag ids (the held tags' order is what counts) and the order of the
///   Orbital Tag's hands (looked up by ante).
///
/// The free rearranges are the caller's to carry.
fn excused(g: &Game) -> Game {
	let mut g = g.clone();
	let a = &mut g.areas;
	for c in a
		.deck
		.iter_mut()
		.chain(&mut a.hand.cards)
		.chain(&mut a.discard)
	{
		c.core.added_to_deck = false;
	}
	for c in a.shop_jokers.iter_mut().chain(&mut a.pack_cards) {
		if let AnyCard::Playing(c) = c {
			c.core.added_to_deck = false;
		}
	}
	g.areas.limits.jokers.real = 0.0;
	g.areas.limits.consumables.real = 0.0;
	g.blind.triggered = false;
	g.run.last_hand_score = 0.0;
	for tag in &mut g.tags.held {
		tag.id = 0;
	}
	g.tags.next_id = 0;
	g.tags.orbital_choices.sort_by_key(|c| c.ante);
	g
}

/// The areas but their cards (compared card by card).
fn without_cards(a: &Areas) -> Areas {
	let mut a = a.clone();
	for v in [&mut a.deck, &mut a.hand.cards, &mut a.discard] {
		v.clear();
	}
	for v in [
		&mut a.jokers,
		&mut a.consumables,
		&mut a.shop_vouchers,
		&mut a.shop_boosters,
	] {
		v.clear();
	}
	a.shop_jokers.clear();
	a.pack_cards.clear();
	a
}

/// The lines of the `Debug` output of `a` and `b` that differ, as `name`'s.
fn part<T: Debug + PartialEq>(out: &mut Vec<String>, name: &str, a: &T, b: &T) {
	if a == b {
		return;
	}
	let (a, b) = (format!("{a:#?}"), format!("{b:#?}"));
	let (a, b): (Vec<&str>, Vec<&str>) = (a.lines().collect(), b.lines().collect());
	for (x, y) in a.iter().zip(&b) {
		if x != y {
			out.push(format!("{name}: {} vs {}", x.trim(), y.trim()));
		}
	}
	if a.len() != b.len() {
		out.push(format!("{name}: {} vs {} lines", a.len(), b.len()));
	}
}

/// The differences between the cards of the area `name`, card by card.
fn cards<C: CardLike + Debug + PartialEq>(out: &mut Vec<String>, name: &str, a: &[C], b: &[C]) {
	if a.len() != b.len() {
		out.push(format!("{name}: {} vs {} cards", a.len(), b.len()));
		return;
	}
	for (i, (x, y)) in a.iter().zip(b).enumerate() {
		part(out, &format!("{name}[{i}] (id {})", x.core().id), x, y);
	}
}

/// A game in both engines, the imported games following it, and the
/// commands so far.
struct Check<'oracle> {
	o: &'oracle mut Oracle,
	g: Game,
	/// The imported games, with the step each was imported at.
	followers: Vec<(usize, Game)>,
	history: Vec<String>,
	stats: DumpStats,
}

impl Check<'_> {
	fn fail(&self, step: usize, what: &str) -> String {
		let n = self.history.len();
		format!(
			"step {step}: {what}\n  last commands: {}",
			self.history[n.saturating_sub(8)..].join(" | ")
		)
	}

	/// Runs `lua` in the oracle; the simulator in lockstep must agree.
	fn sync(&mut self, step: usize, lua: &str) -> Result<Json, String> {
		let answer = self.o.command(lua).map_err(|e| self.fail(step, &e))?;
		let j = json::parse(&answer).map_err(|e| self.fail(step, &e))?;
		let d = canon::diff(&canon::from_sim(&mut self.g), &canon::from_lua(&j));
		if !d.is_empty() {
			return Err(self.fail(step, &format!("lockstep diverged: {}", shown(&d))));
		}
		Ok(j)
	}

	/// Imports the dump `j` of step `step` and checks the imported game.
	fn import(&mut self, step: usize, j: &Json) -> Result<(), String> {
		let theirs = self
			.o
			.eval("return AZB.state()")
			.map_err(|e| self.fail(step, &e))?;
		let theirs = json::parse(&theirs).map_err(|e| self.fail(step, &e))?;
		let differ = writer_diffs(j, &theirs);
		if !differ.is_empty() {
			return Err(self.fail(
				step,
				&format!(
					"the mod's dump differs from the harness's in {}",
					differ.join(", ")
				),
			));
		}
		// The simulator's own writer against the game's (`dump_roundtrip` holds
		// it to its own reader only).
		let d = canon::diff(&canon::from_lua(&self.g.to_dump()), &canon::from_lua(j));
		if !d.is_empty() {
			return Err(self.fail(
				step,
				&format!(
					"the simulator's dump differs from the game's: {}",
					shown(&d)
				),
			));
		}
		let mut imp = Game::from_dump(j).map_err(|e| self.fail(step, &e.to_string()))?;
		imp.flow.rearranges = self.g.flow.rearranges;
		let d = canon::diff(&canon::from_sim(&mut imp), &canon::from_lua(j));
		if !d.is_empty() {
			return Err(self.fail(
				step,
				&format!("import differs from its dump: {}", shown(&d)),
			));
		}
		// At GAME_OVER the run is over: its canonical state is all there is to
		// compare (above), and no decision follows.
		if imp.flow.phase == Phase::GameOver {
			if !imp.is_terminal() || self.g.flow.phase != Phase::GameOver {
				return Err(self.fail(step, "the imported game over is not the game's"));
			}
			*self
				.stats
				.imports
				.entry(Phase::GameOver.name())
				.or_default() += 1;
			return Ok(());
		}
		let (mut a, mut b) = (first_variants(&imp), first_variants(&self.g));
		let parts = part_diffs(&mut a, &mut b);
		if !parts.is_empty() {
			return Err(self.fail(
				step,
				&format!("import differs from lockstep: {}", parts.join("; ")),
			));
		}
		if obs_fingerprint(&mut a) != obs_fingerprint(&mut b) {
			return Err(self.fail(step, "the import's observation differs from lockstep's"));
		}
		*self
			.stats
			.imports
			.entry(self.g.flow.phase.name())
			.or_default() += 1;
		self.followers.push((step, imp));
		Ok(())
	}

	/// Compares every imported game with the game's dump `j` of step `step`.
	fn follow(&mut self, step: usize, j: &Json) -> Result<(), String> {
		let lua = canon::from_lua(j);
		for (since, f) in &mut self.followers {
			let d = canon::diff(&canon::from_sim(f), &lua);
			if !d.is_empty() {
				let since = *since;
				return Err(self.fail(
					step,
					&format!("the game imported at step {since} diverged: {}", shown(&d)),
				));
			}
			self.stats.followed += 1;
		}
		Ok(())
	}
}

/// Loads the mod's state writer into the oracle as `AZB.state()`, with what
/// it takes from the rest of the mod as the mod has it in a game without
/// Steamodded.
fn load_mod_writer(o: &mut Oracle) -> Result<(), String> {
	let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../mod/src");
	o.exec(&format!(
		"SMODS = {{}}
		AZB = {{json = dofile({json})}}
		function AZB.state_name(s) for k, v in pairs(G.STATES) do if v == s then return k end end end
		function AZB.vanilla_state_name() return AZB.state_name(G.STATE) end
		function AZB.is_settled() return true end
		function AZB.is_bridge() return false end
		function AZB.smods_only_rng() return false end
		dofile({state})",
		json = lua_path(&dir.join("json.lua")),
		state = lua_path(&dir.join("state.lua")),
	))
}

/// The keys of the harness's dump `ours` and the mod's `theirs` whose values
/// differ, but for what only one of them writes (the harness's frame count
/// and trace, the mod's frame count and `az`).
fn writer_diffs(ours: &Json, theirs: &Json) -> Vec<String> {
	let (Some(ours), Some(theirs)) = (ours.as_object(), theirs.as_object()) else {
		return vec!["the whole dump".into()];
	};
	let own = |k: &str| matches!(k, "frame" | "trace" | "az");
	let keys: BTreeSet<&String> = ours.keys().chain(theirs.keys()).collect();
	keys.into_iter()
		.filter(|k| !own(k) && ours.get(*k) != theirs.get(*k))
		.cloned()
		.collect()
}

/// A copy of `g` whose first shop's Buffoon Pack is the first of its two
/// art variants: the game picks the variant with the unseeded `math.random`,
/// the simulator always deals the first.
fn first_variants(g: &Game) -> Game {
	let mut g = g.clone();
	// the first shop is the one after the first blind played
	if g.run.round != 1 {
		return g;
	}
	for c in &mut g.areas.shop_boosters {
		if c.booster_slot == Some(0) && c.kind == Kind::Booster(Booster::BuffoonNormal2) {
			c.kind = Kind::Booster(Booster::BuffoonNormal1);
		}
	}
	if let Some(slot @ PackSlot::Offered(Booster::BuffoonNormal2)) =
		g.shop.booster_slots.first_mut()
	{
		*slot = PackSlot::Offered(Booster::BuffoonNormal1);
	}
	g
}

fn shown(d: &[Diff]) -> String {
	let v: Vec<String> = d.iter().take(12).map(ToString::to_string).collect();
	format!(
		"{}{}",
		v.join("; "),
		if d.len() > 12 { "; ..." } else { "" }
	)
}

/// Plays one differential game with the test policy and imports the game's
/// state at each decision point with probability `import_rate`.
///
/// # Errors
///
/// Returns the first failed check, with the step and the last commands.
pub fn check_game(
	o: &mut Oracle, spec: &GameSpec<'_>, import_rate: f64,
) -> Result<DumpStats, String> {
	let mut c = Check {
		o,
		g: spec.start.new_game(),
		followers: Vec::new(),
		history: vec![spec.start.to_string()],
		stats: DumpStats::default(),
	};
	load_mod_writer(c.o)?;
	let mut policy = Policy::new(spec.policy_seed, spec.inject_rate);
	let mut picks = Xorshift::new(spec.policy_seed ^ 0x5eed_d0d0);
	let mut j = c.sync(0, &spec.start.to_lua())?;
	for step in 0.. {
		c.follow(step, &j)?;
		if picks.chance(import_rate) {
			c.import(step, &j)?;
		}
		let command = match policy.next(&c.g) {
			Some(command) if step < spec.max_steps => command,
			_ => break,
		};
		c.history.push(command.to_string());
		command
			.apply(&mut c.g)
			.map_err(|e| c.fail(step + 1, &format!("the simulator rejected {command}: {e}")))?;
		for (since, f) in &mut c.followers {
			let since = *since;
			command.apply(f).map_err(|e| {
				format!(
					"step {}: the game imported at step {since} rejected {command}: {e}",
					step + 1
				)
			})?;
		}
		j = c.sync(step + 1, &command.to_lua())?;
		c.stats.steps = step + 1;
		c.stats.max_ante = c.stats.max_ante.max(c.g.run.ante);
	}
	Ok(c.stats)
}
