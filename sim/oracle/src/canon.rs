//! Canonical state from the oracle's JSON dump, and diffing against the
//! simulator's (see `sim::canon`).

use crate::json::Json;
use sim::data::*;
use sim::game::Game;
pub use sim::canon::{Canon, from_sim};
use sim::canon::{JOKER_COMMON, b, f, joker_fields, normalize_key};

fn lua_field(ab: &Json, name: &str) -> String {
    let v = if let Some(rest) = name.strip_prefix("extra.") { ab.get("extra").get(rest) } else { ab.get(name) };
    match v {
        Json::Null => {
            if matches!(name, "eternal" | "rental" | "couponed" | "perishable") {
                "false".into()
            } else {
                "nil".into()
            }
        }
        Json::Bool(x) => b(*x),
        Json::Num(n) => f(*n),
        Json::Str(s) => s.clone(),
        _ => "?".into(),
    }
}

fn lua_card(out: &mut Canon, p: &str, c: &Json, area: &str) {
    let ab = c.get("ability");
    out.insert(format!("{p}.id"), f(c.get("sort_id").num().unwrap_or(-1.0)));
    out.insert(format!("{p}.key"), normalize_key(c.get("key").str().unwrap_or("?")).to_string());
    out.insert(format!("{p}.edition"), c.get("edition").str().unwrap_or("nil").to_string());
    if !matches!(area, "deck" | "discard") {
        out.insert(format!("{p}.cost"), f(c.get("cost").num().unwrap_or(-1.0)));
        out.insert(format!("{p}.sell_cost"), f(c.get("sell_cost").num().unwrap_or(-1.0)));
    }
    let set = ab.get("set").str().unwrap_or("");
    if set == "Default" || set == "Enhanced" {
        out.insert(format!("{p}.card"), c.get("card").str().unwrap_or("?").to_string());
        out.insert(format!("{p}.seal"), c.get("seal").str().unwrap_or("nil").to_string());
        out.insert(format!("{p}.perma_bonus"), f(ab.get("perma_bonus").num().unwrap_or(0.0)));
        if area != "pack" {
            out.insert(format!("{p}.debuff"), b(c.get("debuff").bool()));
        }
        if area == "hand" {
            out.insert(format!("{p}.facing"), c.get("facing").str().unwrap_or("?").to_string());
            out.insert(format!("{p}.forced"), b(ab.get("forced_selection").bool()));
        }
        out.insert(format!("{p}.played_this_ante"), b(ab.get("played_this_ante").bool()));
    }
    if set == "Joker" {
        out.insert(format!("{p}.debuff"), b(c.get("debuff").bool()));
        if area == "jokers" {
            out.insert(format!("{p}.facing"), c.get("facing").str().unwrap_or("?").to_string());
        }
        let key = c.get("key").str().unwrap_or("");
        let fields = Joker::from_key(key).map(joker_fields).unwrap_or(&[]);
        for name in JOKER_COMMON.iter().chain(fields) {
            out.insert(format!("{p}.ab.{name}"), lua_field(ab, name));
        }
        if ab.get("perishable").bool() {
            out.insert(format!("{p}.ab.perish_tally"), lua_field(ab, "perish_tally"));
        }
    }
    if set == "Tarot" || set == "Planet" || set == "Spectral" {
        out.insert(format!("{p}.ab.extra_value"), f(ab.get("extra_value").num().unwrap_or(0.0)));
    }
}

/// Canonical view of the oracle's JSON dump.
pub fn from_lua(j: &Json) -> Canon {
    let mut o = Canon::new();
    o.insert("state".into(), j.get("state").str().unwrap_or("?").into());
    let nums = [
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
    ];
    for k in nums {
        o.insert(k.into(), j.get(k).num().map(f).unwrap_or("nil".into()));
    }
    o.insert("pack_choices".into(), j.get("pack_choices").num().map(f).unwrap_or("nil".into()));
    let bl = j.get("blind");
    o.insert("blind.chips".into(), bl.get("chips").num().map(f).unwrap_or("nil".into()));
    o.insert("blind.dollars".into(), bl.get("dollars").num().map(f).unwrap_or("nil".into()));
    o.insert("blind.name".into(), bl.get("name").str().unwrap_or("").into());
    o.insert("blind.disabled".into(), b(bl.get("disabled").bool()));
    o.insert("blind_on_deck".into(), j.get("blind_on_deck").str().unwrap_or("nil").into());
    for n in ["Small", "Big", "Boss"] {
        o.insert(format!("blind_states.{n}"), j.get("blind_states").get(n).str().unwrap_or("nil").into());
    }
    o.insert("blind_choices.Boss".into(), j.get("blind_choices").get("Boss").str().unwrap_or("nil").into());
    o.insert("blind_tags.Small".into(), j.get("blind_tags").get("Small").str().unwrap_or("nil").into());
    o.insert("blind_tags.Big".into(), j.get("blind_tags").get("Big").str().unwrap_or("nil").into());
    o.insert("voucher".into(), j.get("voucher").str().unwrap_or("nil").into());
    o.insert("won".into(), b(j.get("won").bool()));
    o.insert("boss_rerolled".into(), b(j.get("boss_rerolled").bool()));
    o.insert("gros_michel_extinct".into(), b(j.get("gros_michel_extinct").bool()));
    o.insert("last_hand_played".into(), j.get("last_hand_played").str().unwrap_or("nil").into());
    o.insert("last_tarot_planet".into(), j.get("last_tarot_planet").str().unwrap_or("nil").into());
    o.insert("most_played".into(), j.get("most_played").str().unwrap_or("nil").into());
    o.insert("idol.rank".into(), j.get("idol").get("rank").num().map(f).unwrap_or("nil".into()));
    o.insert("idol.suit".into(), j.get("idol").get("suit").str().unwrap_or("nil").into());
    o.insert("mail".into(), j.get("mail").num().map(f).unwrap_or("nil".into()));
    o.insert("ancient".into(), j.get("ancient").str().unwrap_or("nil").into());
    o.insert("castle".into(), j.get("castle").str().unwrap_or("nil").into());
    o.insert("sort_mode".into(), j.get("sort_mode").str().unwrap_or("desc").into());
    for h in HandType::ALL {
        let n = h.name();
        let l = j.get("hands").get(n);
        for k in ["level", "chips", "mult", "played", "played_this_round"] {
            o.insert(format!("hands.{n}.{k}"), l.get(k).num().map(f).unwrap_or("nil".into()));
        }
        o.insert(format!("hands.{n}.visible"), b(l.get("visible").bool()));
    }
    for (i, t) in j.get("tag_info").arr().iter().enumerate() {
        let key = t.get("key").str().unwrap_or("?");
        o.insert(format!("tags.{i}.key"), key.into());
        o.insert(format!("tags.{i}.triggered"), b(t.get("triggered").bool()));
        if key == "tag_orbital" {
            o.insert(format!("tags.{i}.orbital"), t.get("orbital").str().unwrap_or("nil").into());
        }
    }
    let uv: Vec<&str> = j.get("used_vouchers").arr().iter().filter_map(|v| v.str()).collect();
    o.insert("used_vouchers".into(), uv.join(","));
    let mut uj: Vec<&str> = j
        .get("used_jokers")
        .arr()
        .iter()
        .filter_map(|v| v.str())
        .filter(|k| {
            (k.starts_with("j_") || k.starts_with("c_") || k.starts_with("v_") || k.starts_with("p_")) && *k != "c_base"
        })
        .collect();
    uj.sort();
    o.insert("used_jokers".into(), uj.join(","));
    let areas = [
        ("hand", "hand"),
        ("deck", "deck"),
        ("discard", "discard"),
        ("jokers", "jokers"),
        ("consumables", "consumables"),
        ("shop_jokers", "shop_jokers"),
        ("shop_vouchers", "shop_vouchers"),
        ("shop_booster", "shop_booster"),
        ("pack", "pack_cards"),
    ];
    for (name, key) in areas {
        let v = j.get(key).arr();
        o.insert(format!("{name}.len"), v.len().to_string());
        for (i, c) in v.iter().enumerate() {
            lua_card(&mut o, &format!("{name}.{i}"), c, name);
        }
    }
    if let Some(m) = j.get("rng").obj() {
        for (k, v) in m {
            if k == "seed" || k == "hashed_seed" {
                continue;
            }
            if let Some(n) = v.num() {
                o.insert(format!("rng.{k}"), f(n));
            }
        }
    }
    o
}

/// Differences `(path, sim, oracle)`.
pub fn diff(sim: &Canon, lua: &Canon) -> Vec<(String, String, String)> {
    let mut out = Vec::new();
    // The real game pauses on GAME_OVER (pending events never run); only the
    // outcome itself is comparable.
    if sim.get("state").map(|s| s.as_str()) == Some("GAME_OVER") || lua.get("state").map(|s| s.as_str()) == Some("GAME_OVER") {
        if sim.get("state") != lua.get("state") {
            out.push(("state".into(), sim["state"].clone(), lua["state"].clone()));
        }
        return out;
    }
    for (k, v) in sim {
        match lua.get(k) {
            Some(w) if w == v => {}
            Some(w) => out.push((k.clone(), v.clone(), w.clone())),
            None => out.push((k.clone(), v.clone(), "<missing>".into())),
        }
    }
    for (k, w) in lua {
        if !sim.contains_key(k) {
            out.push((k.clone(), "<missing>".into(), w.clone()));
        }
    }
    out
}

/// How the current action's timeline differs between the engines.
#[derive(Clone, Debug, PartialEq)]
pub enum TimelineDiff {
    /// The order of `pseudoseed` calls, card draws or card removals differs:
    /// `(timeline[i], sim context, game context)` around the first mismatch.
    Order(String, String, String),
    /// Same order, but an entry happened on another frame.
    Frame(String),
}

/// The oracle's timeline entries for the last command.
fn lua_trace(lua: &Json) -> Vec<&str> {
    lua.get("trace").arr().iter().filter_map(|j| j.str()).collect()
}

/// Compares the current action's timeline (`<frame> <entry>`) in both
/// engines: first the order of the entries, then their frames. Nothing is
/// compared once the game is over (the game pauses there).
pub fn timeline_diff(g: &Game, lua: &Json) -> Option<TimelineDiff> {
    if lua.get("state").str() == Some("GAME_OVER") {
        return None;
    }
    let sim = g.timeline();
    let game = lua_trace(lua);
    let body = |s: &str| s.split_once(' ').map(|(_, b)| b.to_string()).unwrap_or_default();
    let n = sim.len().max(game.len());
    for i in 0..n {
        let a = sim.get(i).map(|s| s.as_str());
        let b = game.get(i).copied();
        if a.map(body) != b.map(body) {
            let ctx = |v: &dyn Fn(usize) -> Option<String>| {
                (i.saturating_sub(3)..(i + 3).min(n)).filter_map(|k| v(k).map(|s| format!("[{k}] {s}"))).collect::<Vec<_>>().join(" | ")
            };
            let sa = ctx(&|k| sim.get(k).cloned());
            let sb = ctx(&|k| game.get(k).map(|s| s.to_string()));
            return Some(TimelineDiff::Order(format!("timeline[{i}]"), sa, sb));
        }
    }
    sim.iter().zip(&game).find(|(a, b)| a.as_str() != **b).map(|(a, b)| TimelineDiff::Frame(format!("sim `{a}` game `{b}`")))
}

/// Both timelines side by side, `!` marking entries on different frames
/// (diagnostics).
pub fn timeline_frames(g: &Game, lua: &Json) -> Vec<String> {
    let sim = g.timeline();
    let game = lua_trace(lua);
    let frame = |s: &str| s.split_once(' ').map(|x| x.0.to_string());
    (0..sim.len().max(game.len()))
        .map(|i| {
            let a = sim.get(i).map(|s| s.as_str()).unwrap_or("-");
            let b = game.get(i).copied().unwrap_or("-");
            let mark = if frame(a) == frame(b) { " " } else { "!" };
            format!("{mark} sim {a:<40} game {b}")
        })
        .collect()
}
