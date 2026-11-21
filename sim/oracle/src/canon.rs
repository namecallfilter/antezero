//! Canonical state from the oracle's JSON dump, and diffing against the
//! simulator's (see `sim::canon`).

use std::collections::BTreeSet;
use std::fmt;

pub use sim::canon::{Canon, from_sim};
use sim::canon::{CanonArea, JokerField, canon_number, normalize_key, placeholder_center};
use sim::data::{HandType, Joker};
use sim::game::Game;
use sim::json::Json;

/// The game's value at `path` of a joker's `ability` table, `absent` if it
/// has none.
fn lua_ability(ab: &Json, path: &str, absent: &str) -> String {
	let v = if let Some(rest) = path.strip_prefix("extra.") {
		ab.get("extra").get(rest)
	} else {
		ab.get(path)
	};
	match v {
		Json::Null => absent.into(),
		Json::Bool(x) => x.to_string(),
		Json::Num(n) => canon_number(*n),
		Json::Str(s) => s.clone(),
		_ => "?".into(),
	}
}

fn lua_card(out: &mut Canon, p: &str, c: &Json, area: CanonArea) {
	let ab = c.get("ability");
	let number = |field: &Json| field.as_f64().map_or("nil".into(), canon_number);
	out.insert(format!("{p}.id"), number(c.get("sort_id")));
	out.insert(
		format!("{p}.key"),
		normalize_key(c.get("key").as_str().unwrap_or("?")).to_string(),
	);
	out.insert(
		format!("{p}.edition"),
		c.get("edition").as_str().unwrap_or("nil").to_string(),
	);
	if area.shows_cost() {
		out.insert(format!("{p}.cost"), number(c.get("cost")));
		out.insert(format!("{p}.sell_cost"), number(c.get("sell_cost")));
	}
	let set = ab.get("set").as_str().unwrap_or("");
	if set == "Default" || set == "Enhanced" {
		out.insert(
			format!("{p}.card"),
			c.get("card").as_str().unwrap_or("?").to_string(),
		);
		out.insert(
			format!("{p}.seal"),
			c.get("seal").as_str().unwrap_or("nil").to_string(),
		);
		out.insert(format!("{p}.perma_bonus"), number(ab.get("perma_bonus")));
		if area.shows_playing_debuff() {
			out.insert(format!("{p}.debuff"), c.get("debuff").as_flag().to_string());
		}
		if area.shows_facing() {
			out.insert(
				format!("{p}.facing"),
				c.get("facing").as_str().unwrap_or("?").to_string(),
			);
			out.insert(
				format!("{p}.forced"),
				ab.get("forced_selection").as_flag().to_string(),
			);
		}
		out.insert(
			format!("{p}.played_this_ante"),
			ab.get("played_this_ante").as_flag().to_string(),
		);
	}
	if set == "Joker" {
		out.insert(format!("{p}.debuff"), c.get("debuff").as_flag().to_string());
		if area.shows_facing() {
			out.insert(
				format!("{p}.facing"),
				c.get("facing").as_str().unwrap_or("?").to_string(),
			);
		}
		let key = c.get("key").as_str().unwrap_or("");
		let fields: &[JokerField] = Joker::from_key(key).map_or(&[], JokerField::of);
		for field in JokerField::COMMON.iter().chain(fields) {
			let path = field.path();
			let absent = if field.is_flag() { "false" } else { "nil" };
			out.insert(format!("{p}.ab.{path}"), lua_ability(ab, path, absent));
		}
		if ab.get("perishable").as_flag() {
			out.insert(
				format!("{p}.ab.perish_tally"),
				lua_ability(ab, "perish_tally", "nil"),
			);
		}
	}
	if set == "Tarot" || set == "Planet" || set == "Spectral" {
		out.insert(format!("{p}.ab.extra_value"), number(ab.get("extra_value")));
	}
}

/// The dump's value at a dotted `path` (`Null` if any part is missing).
fn at<'a>(j: &'a Json, path: &str) -> &'a Json {
	path.split('.').fold(j, |j, key| j.get(key))
}

/// Number fields, compared at their own path (`nil` when absent).
const NUMBERS: [&str; 39] = [
	"ante",
	"round",
	"dollars",
	"chips",
	"hands_left",
	"discards_left",
	"reroll_cost",
	"free_rerolls",
	"rr_hands",
	"rr_discards",
	"rr_reroll_cost",
	"skips",
	"interest_cap",
	"interest_amount",
	"discount_percent",
	"bankrupt_at",
	"probabilities",
	"shop_joker_max",
	"tarot_rate",
	"planet_rate",
	"edition_rate",
	"playing_card_rate",
	"spectral_rate",
	"joker_limit",
	"consumable_limit",
	"hand_limit",
	"hands_played",
	"hands_played_round",
	"discards_used",
	"unused_discards",
	"ecto_minus",
	"starting_deck_size",
	"round_dollars",
	"pack_size",
	"pack_choices",
	"blind.chips",
	"blind.dollars",
	"idol.rank",
	"mail",
];

/// String fields, compared at their own path (`nil` when absent).
const STRINGS: [&str; 14] = [
	"blind_on_deck",
	"blind_states.Small",
	"blind_states.Big",
	"blind_states.Boss",
	"blind_choices.Boss",
	"blind_tags.Small",
	"blind_tags.Big",
	"voucher",
	"last_hand_played",
	"last_tarot_planet",
	"most_played",
	"idol.suit",
	"ancient",
	"castle",
];

/// Flags, compared at their own path (false when absent).
const FLAGS: [&str; 4] = [
	"blind.disabled",
	"won",
	"boss_rerolled",
	"gros_michel_extinct",
];

/// Canonical view of the oracle's JSON dump.
pub fn from_lua(j: &Json) -> Canon {
	let mut o = Canon::new();
	for path in NUMBERS {
		o.insert(
			path.into(),
			at(j, path).as_f64().map_or("nil".into(), canon_number),
		);
	}
	for path in STRINGS {
		o.insert(path.into(), at(j, path).as_str().unwrap_or("nil").into());
	}
	for path in FLAGS {
		o.insert(path.into(), at(j, path).as_flag().to_string());
	}
	o.insert(
		"state".into(),
		j.get("state").as_str().unwrap_or("?").into(),
	);
	o.insert(
		"blind.name".into(),
		at(j, "blind.name").as_str().unwrap_or("").into(),
	);
	o.insert(
		"sort_mode".into(),
		j.get("sort_mode").as_str().unwrap_or("nil").into(),
	);
	for h in HandType::ALL {
		let n = h.name();
		let l = j.get("hands").get(n);
		for k in ["level", "chips", "mult", "played", "played_this_round"] {
			o.insert(
				format!("hands.{n}.{k}"),
				l.get(k).as_f64().map_or("nil".into(), canon_number),
			);
		}
		o.insert(
			format!("hands.{n}.visible"),
			l.get("visible").as_flag().to_string(),
		);
	}
	for (i, t) in j.get("tag_info").as_list().iter().enumerate() {
		let key = t.get("key").as_str().unwrap_or("?");
		o.insert(format!("tags.{i}.key"), key.into());
		o.insert(
			format!("tags.{i}.triggered"),
			t.get("triggered").as_flag().to_string(),
		);
		if key == "tag_orbital" {
			o.insert(
				format!("tags.{i}.orbital"),
				t.get("orbital").as_str().unwrap_or("nil").into(),
			);
		}
	}
	let uv: Vec<&str> = j
		.get("used_vouchers")
		.as_list()
		.iter()
		.filter_map(|v| v.as_str())
		.collect();
	o.insert("used_vouchers".into(), uv.join(","));
	let mut uj: Vec<&str> = j
		.get("used_jokers")
		.as_list()
		.iter()
		.filter_map(|v| v.as_str())
		.filter(|k| {
			(k.starts_with("j_")
				|| k.starts_with("c_")
				|| k.starts_with("v_")
				|| k.starts_with("p_"))
				&& *k != "c_base"
				&& !placeholder_center(k)
		})
		.collect();
	uj.sort_unstable();
	o.insert("used_jokers".into(), uj.join(","));
	for area in CanonArea::ALL {
		// The dump's key for the area: the canonical name but for the pack.
		let key = match area {
			CanonArea::Pack => "pack_cards",
			_ => area.name(),
		};
		let v = j.get(key).as_list();
		let name = area.name();
		o.insert(format!("{name}.len"), v.len().to_string());
		for (i, c) in v.iter().enumerate() {
			lua_card(&mut o, &format!("{name}.{i}"), c, area);
		}
	}
	if let Some(m) = j.get("rng").as_object() {
		for (k, v) in m {
			if k == "seed" || k == "hashed_seed" {
				continue;
			}
			if let Some(n) = v.as_f64() {
				o.insert(format!("rng.{k}"), canon_number(n));
			}
		}
	}
	o
}

/// One canonical field as both engines hold it (`None` where an engine has
/// no such field).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Field<'a> {
	pub path: &'a str,
	pub sim: Option<&'a str>,
	pub game: Option<&'a str>,
}

impl Field<'_> {
	#[must_use]
	pub fn differs(&self) -> bool {
		self.sim != self.game
	}

	#[must_use]
	pub fn to_diff(&self) -> Diff {
		Diff {
			path: self.path.to_string(),
			sim: self.sim.map(str::to_string),
			game: self.game.map(str::to_string),
		}
	}
}

impl fmt::Display for Field<'_> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		const MISSING: &str = "<missing>";
		write!(
			f,
			"{}: sim={} game={}",
			self.path,
			self.sim.unwrap_or(MISSING),
			self.game.unwrap_or(MISSING)
		)
	}
}

/// Every field of either canonical state, in path order.
pub fn fields<'a>(sim: &'a Canon, lua: &'a Canon) -> impl Iterator<Item = Field<'a>> {
	let paths: BTreeSet<&str> = sim.keys().chain(lua.keys()).map(String::as_str).collect();
	paths.into_iter().map(|path| Field {
		path,
		sim: sim.get(path).map(String::as_str),
		game: lua.get(path).map(String::as_str),
	})
}

/// A field the engines disagree on (see [`Field`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diff {
	pub path: String,
	pub sim: Option<String>,
	pub game: Option<String>,
}

impl fmt::Display for Diff {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		let field = Field {
			path: &self.path,
			sim: self.sim.as_deref(),
			game: self.game.as_deref(),
		};
		fmt::Display::fmt(&field, f)
	}
}

/// The differences between the two canonical states.
#[must_use]
pub fn diff(sim: &Canon, lua: &Canon) -> Vec<Diff> {
	// The real game pauses on GAME_OVER (pending events never run); only the
	// outcome itself is comparable.
	let state = Field {
		path: "state",
		sim: sim.get("state").map(String::as_str),
		game: lua.get("state").map(String::as_str),
	};
	if state.sim == Some("GAME_OVER") || state.game == Some("GAME_OVER") {
		return if state.differs() {
			vec![state.to_diff()]
		} else {
			Vec::new()
		};
	}
	fields(sim, lua)
		.filter(Field::differs)
		.map(|field| field.to_diff())
		.collect()
}

/// How the current action's timeline differs between the engines.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TimelineDiff {
	/// The order of `pseudoseed` calls, card draws or card removals differs.
	Order(OrderDiff),
	/// Same order, but an entry happened on another frame.
	Frame(FrameOffset),
}

/// The first entry at which the timelines' order differs, with each
/// engine's entries around it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrderDiff {
	pub index: usize,
	pub sim: String,
	pub game: String,
}

impl fmt::Display for OrderDiff {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(
			f,
			"timeline[{}]\n  sim:  {}\n  game: {}",
			self.index, self.sim, self.game
		)
	}
}

/// The first timeline entry that each engine has on a different frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameOffset {
	pub sim: String,
	pub game: String,
}

impl FrameOffset {
	/// How many frames after the simulator's entry the game's happened
	/// (`None` if an entry has no frame number).
	#[must_use]
	pub fn frames(&self) -> Option<i64> {
		let frame = |entry: &str| entry_frame(entry)?.parse::<i64>().ok();
		Some(frame(&self.game)? - frame(&self.sim)?)
	}
}

impl fmt::Display for FrameOffset {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "sim `{}` game `{}`", self.sim, self.game)
	}
}

fn lua_trace(lua: &Json) -> Vec<&str> {
	lua.get("trace")
		.as_list()
		.iter()
		.filter_map(|j| j.as_str())
		.collect()
}

/// Compares the current action's timeline (`<frame> <entry>`) in both
/// engines: first the order of the entries, then their frames. Nothing is
/// compared once the game is over (the game pauses there).
#[must_use]
pub fn timeline_diff(g: &Game, lua: &Json) -> Option<TimelineDiff> {
	if lua.get("state").as_str() == Some("GAME_OVER") {
		return None;
	}
	let sim = g.timeline();
	let game = lua_trace(lua);
	let n = sim.len().max(game.len());
	for i in 0..n {
		let a = sim.get(i).map(String::as_str);
		let b = game.get(i).copied();
		if a.map(entry_body) != b.map(entry_body) {
			return Some(TimelineDiff::Order(OrderDiff {
				index: i,
				sim: around(i, n, |k| sim.get(k).map(String::as_str)),
				game: around(i, n, |k| game.get(k).copied()),
			}));
		}
	}
	sim.iter()
		.zip(&game)
		.find(|(a, b)| a.as_str() != **b)
		.map(|(a, b)| {
			TimelineDiff::Frame(FrameOffset {
				sim: a.clone(),
				game: (*b).to_string(),
			})
		})
}

/// The entries around the `i`-th of a timeline of `n` (`entry(k)` is the
/// `k`-th, if there is one), numbered.
fn around<'a>(i: usize, n: usize, entry: impl Fn(usize) -> Option<&'a str>) -> String {
	(i.saturating_sub(3)..(i + 3).min(n))
		.filter_map(|k| entry(k).map(|s| format!("[{k}] {s}")))
		.collect::<Vec<_>>()
		.join(" | ")
}

/// What a timeline entry (`<frame> <what happened>`) says happened: all of
/// it if it has no frame.
fn entry_body(entry: &str) -> &str {
	entry.split_once(' ').map_or(entry, |(_, body)| body)
}

fn entry_frame(entry: &str) -> Option<&str> {
	entry.split_once(' ').map(|(frame, _)| frame)
}

/// Both timelines side by side, `!` marking entries on different frames
/// (diagnostics).
#[must_use]
pub fn timeline_frames(g: &Game, lua: &Json) -> Vec<String> {
	let sim = g.timeline();
	let game = lua_trace(lua);
	(0..sim.len().max(game.len()))
		.map(|i| {
			let a = sim.get(i).map_or("-", String::as_str);
			let b = game.get(i).copied().unwrap_or("-");
			let mark = if entry_frame(a) == entry_frame(b) {
				" "
			} else {
				"!"
			};
			format!("{mark} sim {a:<40} game {b}")
		})
		.collect()
}
