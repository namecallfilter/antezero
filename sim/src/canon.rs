//! A flattened, canonical view of the game state (`key -> value` strings),
//! built identically by the oracle from the real game's state dump so the
//! two can be diffed. Includes every RNG stream's state.

use std::collections::BTreeMap;

use crate::card::{Card, Kind};
use crate::data::{ALL_VOUCHERS, BlindKind, Center, HandType, Joker, Seal, Suit, TagKind, Voucher};
use crate::game::{BlindSlot, Game};

pub type Canon = BTreeMap<String, String>;

#[must_use]
pub fn f(v: f64) -> String {
	if v.is_nan() {
		"nan".into()
	} else {
		format!("{v}")
	}
}

#[must_use]
pub fn b(v: bool) -> String {
	if v { "true".into() } else { "false".into() }
}

/// Joker ability fields compared beyond the common ones.
#[must_use]
pub fn joker_fields(j: Joker) -> &'static [&'static str] {
	use Joker as J;
	match j {
		J::Runner | J::IceCream | J::Square | J::Castle | J::Wee => &["extra.chips"],
		J::Rocket => &["extra.dollars"],
		J::Selzer => &["extra"],
		J::TurtleBean => &["extra.h_size"],
		J::Invisible => &["invis_rounds"],
		J::Yorick => &["yorick_discards"],
		J::Caino => &["caino_xmult"],
		J::LoyaltyCard => &["loyalty_remaining"],
		J::TodoList => &["to_do_poker_hand"],
		J::SteelJoker => &["steel_tally"],
		J::Stone => &["stone_tally"],
		J::DriversLicense => &["driver_tally"],
		J::Cloud9 => &["nine_tally"],
		_ => &[],
	}
}

pub const JOKER_COMMON: [&str; 6] = [
	"mult",
	"x_mult",
	"extra_value",
	"eternal",
	"rental",
	"couponed",
];

fn sim_joker_field(c: &Card, name: &str) -> String {
	let a = &c.ab;
	match name {
		"mult" => f(a.mult),
		"x_mult" => f(a.x_mult),
		"extra_value" => f(a.extra_value),
		"eternal" => b(a.eternal),
		"rental" => b(a.rental),
		"couponed" => b(a.couponed),
		"extra.chips" => f(a.chips),
		"extra.dollars" => f(a.dollars),
		"extra" | "extra.h_size" => f(a.remaining),
		"invis_rounds" => f(a.invis_rounds),
		"yorick_discards" => f(a.yorick_discards),
		"caino_xmult" => f(a.caino_xmult),
		"loyalty_remaining" => f(a.loyalty_remaining),
		"to_do_poker_hand" => a
			.to_do
			.map_or_else(|| "nil".into(), |h| h.name().to_string()),
		"steel_tally" => f(a.steel_tally),
		"stone_tally" => f(a.stone_tally),
		"driver_tally" => f(a.driver_tally),
		"nine_tally" => f(a.nine_tally),
		_ => unreachable!(),
	}
}

#[must_use]
pub fn normalize_key(k: &str) -> &str {
	match k {
		"p_buffoon_normal_2" => "p_buffoon_normal_1",
		"p_arcana_mega_2" => "p_arcana_mega_1",
		"p_celestial_mega_2" => "p_celestial_mega_1",
		other => other,
	}
}

fn sim_card(out: &mut Canon, p: &str, c: &Card, area: &str) {
	out.insert(format!("{p}.id"), c.id.to_string());
	out.insert(format!("{p}.key"), normalize_key(c.key()).to_string());
	out.insert(
		format!("{p}.edition"),
		c.edition.name().unwrap_or("nil").to_string(),
	);
	if !matches!(area, "deck" | "discard") {
		out.insert(format!("{p}.cost"), f(c.cost));
		out.insert(format!("{p}.sell_cost"), f(c.sell_cost));
	}
	if c.is_playing() {
		out.insert(format!("{p}.card"), c.card_key());
		out.insert(
			format!("{p}.seal"),
			c.seal.map_or("nil", Seal::name).to_string(),
		);
		out.insert(format!("{p}.perma_bonus"), f(c.ab.perma_bonus));
		if area != "pack" {
			out.insert(format!("{p}.debuff"), b(c.debuff));
		}
		if area == "hand" {
			out.insert(
				format!("{p}.facing"),
				(if c.face_down { "back" } else { "front" }).into(),
			);
			out.insert(format!("{p}.forced"), b(c.ab.forced_selection));
		}
		out.insert(format!("{p}.played_this_ante"), b(c.ab.played_this_ante));
	}
	if let Kind::Joker(j) = c.kind {
		out.insert(format!("{p}.debuff"), b(c.debuff));
		if area == "jokers" {
			out.insert(
				format!("{p}.facing"),
				(if c.face_down { "back" } else { "front" }).into(),
			);
		}
		for name in JOKER_COMMON.iter().chain(joker_fields(j)) {
			out.insert(format!("{p}.ab.{name}"), sim_joker_field(c, name));
		}
		if c.ab.perishable {
			out.insert(format!("{p}.ab.perish_tally"), f(c.ab.perish_tally));
		}
	}
	if c.kind.is_consumable() {
		out.insert(format!("{p}.ab.extra_value"), f(c.ab.extra_value));
	}
}

/// Canonical view of the simulator's state (refreshes the per-frame derived
/// joker values first, like the game does every frame).
#[expect(clippy::too_many_lines, reason = "one line per compared field")]
pub fn from_sim(g: &mut Game) -> Canon {
	g.update_derived();
	let mut o = Canon::new();
	o.insert("state".into(), g.state.name().into());
	let nums: [(&str, f64); 36] = [
		("ante", f64::from(g.ante)),
		("round", f64::from(g.round)),
		("dollars", g.dollars),
		("chips", g.chips),
		("hands_left", g.hands_left),
		("discards_left", g.discards_left),
		("reroll_cost", g.reroll_cost),
		("free_rerolls", g.free_rerolls),
		("rr_hands", g.rr_hands),
		("rr_discards", g.rr_discards),
		("rr_reroll_cost", g.rr_reroll_cost),
		("skips", g.skips),
		("interest_cap", g.interest_cap),
		("interest_amount", g.interest_amount),
		("discount_percent", g.discount_percent),
		("bankrupt_at", g.bankrupt_at),
		("probabilities", g.probabilities),
		("shop_joker_max", g.shop_joker_max),
		("tarot_rate", g.tarot_rate),
		("planet_rate", g.planet_rate),
		("edition_rate", g.edition_rate),
		("playing_card_rate", g.playing_card_rate),
		("spectral_rate", g.spectral_rate),
		("joker_limit", g.joker_limit.limit),
		("consumable_limit", g.consumable_limit.limit),
		("hand_limit", g.hand_limit.limit),
		("hands_played", g.hands_played),
		("hands_played_round", g.hands_played_round),
		("discards_used", g.discards_used),
		("unused_discards", g.unused_discards),
		("ecto_minus", g.ecto_minus),
		("starting_deck_size", g.starting_deck_size),
		("round_dollars", g.round_dollars),
		("blind.chips", g.blind.chips),
		("blind.dollars", g.blind.dollars),
		("pack_size", f64::from(g.pack_size)),
	];
	for (k, v) in nums {
		o.insert(k.into(), f(v));
	}
	o.insert(
		"pack_choices".into(),
		if g.pack_choices == 0 {
			"nil".into()
		} else {
			f(f64::from(g.pack_choices))
		},
	);
	o.insert(
		"blind.name".into(),
		g.blind.kind.map_or("", BlindKind::name).into(),
	);
	o.insert("blind.disabled".into(), b(g.blind.disabled));
	o.insert(
		"blind_on_deck".into(),
		g.blind_on_deck.map_or("nil", BlindSlot::name).into(),
	);
	for (i, n) in ["Small", "Big", "Boss"].iter().enumerate() {
		o.insert(format!("blind_states.{n}"), g.blind_states[i].name().into());
	}
	o.insert("blind_choices.Boss".into(), g.boss_choice.key().into());
	o.insert(
		"blind_tags.Small".into(),
		g.blind_tags[0].map_or("nil", TagKind::key).into(),
	);
	o.insert(
		"blind_tags.Big".into(),
		g.blind_tags[1].map_or("nil", TagKind::key).into(),
	);
	o.insert(
		"voucher".into(),
		g.current_voucher.map_or("nil", Voucher::key).into(),
	);
	o.insert("won".into(), b(g.won));
	o.insert("boss_rerolled".into(), b(g.boss_rerolled));
	o.insert("gros_michel_extinct".into(), b(g.gros_michel_extinct));
	o.insert(
		"last_hand_played".into(),
		g.last_hand_played.map_or("nil", HandType::name).into(),
	);
	o.insert(
		"last_tarot_planet".into(),
		g.last_tarot_planet.map_or("nil", Kind::key).into(),
	);
	o.insert("most_played".into(), g.most_played_hand.name().into());
	o.insert(
		"idol.rank".into(),
		g.idol_card
			.id
			.map_or_else(|| "nil".into(), |r| f(f64::from(r))),
	);
	o.insert("idol.suit".into(), g.idol_card.suit.name().into());
	o.insert(
		"mail".into(),
		g.mail_rank
			.id
			.map_or_else(|| "nil".into(), |r| f(f64::from(r))),
	);
	o.insert(
		"ancient".into(),
		g.ancient_suit.map_or("nil", Suit::name).into(),
	);
	o.insert("castle".into(), g.castle_suit.name().into());
	o.insert(
		"sort_mode".into(),
		(if g.hand_sort_by_suit {
			"suit desc"
		} else {
			"desc"
		})
		.into(),
	);
	for h in HandType::ALL {
		let l = g.hands[h as usize];
		let n = h.name();
		o.insert(format!("hands.{n}.level"), f(l.level));
		o.insert(format!("hands.{n}.chips"), f(l.chips));
		o.insert(format!("hands.{n}.mult"), f(l.mult));
		o.insert(format!("hands.{n}.played"), f(l.played));
		o.insert(
			format!("hands.{n}.played_this_round"),
			f(l.played_this_round),
		);
		o.insert(format!("hands.{n}.visible"), b(l.visible));
	}
	for (i, t) in g.tags.iter().enumerate() {
		o.insert(format!("tags.{i}.key"), t.kind.key().into());
		o.insert(format!("tags.{i}.triggered"), b(t.triggered));
		if t.kind == TagKind::Orbital {
			o.insert(
				format!("tags.{i}.orbital"),
				t.orbital_hand.map_or("nil", HandType::name).into(),
			);
		}
	}
	let mut uv: Vec<&str> = ALL_VOUCHERS
		.iter()
		.filter(|v| g.voucher_used(**v))
		.map(|v| v.key())
		.collect();
	uv.sort_unstable();
	o.insert("used_vouchers".into(), uv.join(","));
	let mut uj: Vec<&str> = Center::all_cards()
		.filter(|&c| g.used_jokers.contains(c))
		.map(Center::key)
		.collect();
	uj.sort_unstable();
	o.insert("used_jokers".into(), uj.join(","));
	let areas: [(&str, &Vec<Card>); 9] = [
		("hand", &g.hand),
		("deck", &g.deck),
		("discard", &g.discard),
		("jokers", &g.jokers),
		("consumables", &g.consumables),
		("shop_jokers", &g.shop_jokers),
		("shop_vouchers", &g.shop_vouchers),
		("shop_booster", &g.shop_boosters),
		("pack", &g.pack_cards),
	];
	for (name, v) in areas {
		o.insert(format!("{name}.len"), v.len().to_string());
		for (i, c) in v.iter().enumerate() {
			sim_card(&mut o, &format!("{name}.{i}"), c, name);
		}
	}
	for k in g.rng.stream_names() {
		if let Some(v) = g.rng.stream_value(&k) {
			o.insert(format!("rng.{k}"), f(v));
		}
	}
	o
}
