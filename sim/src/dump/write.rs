//! A [`Game`] written as a state dump, in the shape of the game's.

use serde_json::{Map, Value};

use crate::canon::JokerField;
use crate::card::{AnyCard, Card, CardCore, Kind, PlayingCard, Sticker};
use crate::data::{
	ALL_BLINDS, ALL_PLANETS, ALL_TAGS, ALL_VOUCHERS, BlindKind, Center, Edition, Enhancement,
	HandType, Rank, Seal, TagKind, Voucher,
};
use crate::game::{
	Blind, BlindSelect, BlindSlot, Game, HandLevels, HandSort, PackSlot, Round, Shop, Tags,
};
use crate::json;
use crate::round::BlindScaling;
use crate::run::{History, Modifiers, Run, StartingParams};

fn object<'key>(fields: impl IntoIterator<Item = (&'key str, Value)>) -> Value {
	Value::Object(
		fields
			.into_iter()
			.map(|(k, v)| (k.to_string(), v))
			.collect(),
	)
}

fn array(items: impl IntoIterator<Item = Value>) -> Value {
	Value::Array(items.into_iter().collect())
}

/// The state dump of `g`: what the game's would hold in the same state.
pub(crate) fn write(g: &Game) -> Value {
	let mut fields = vec![
		("state", g.flow.state.name().into()),
		("back", g.setup.back.key().into()),
		("stake", json::number(f64::from(u8::from(g.setup.stake)))),
		("win_ante", json::number(f64::from(g.setup.win_ante))),
		("modifiers", Value::Object(modifiers(&g.setup.modifiers))),
		("challenge", Value::Null),
		("vouchers_in_shop", Value::Null),
		("boosters_in_shop", Value::Null),
		("starting_params", starting_params(&g.setup.starting_params)),
		("rng", rng(g)),
		("blind", blind(&g.blind)),
		("hands", hands(&g.hands)),
		(
			"pack_interrupt",
			g.pack.open.map(|p| p.interrupted.name()).into(),
		),
		("pack_size", json::number(f64::from(g.pack.size))),
		("pack_choices", json::number(g.pack.choices.map(f64::from))),
	];
	areas(g, &mut fields);
	run(&g.run, &mut fields);
	history(&g.history, &mut fields);
	round(&g.round, &mut fields);
	blind_select(&g.blind_select, &mut fields);
	shop(&g.shop, &mut fields);
	tags(&g.tags, &mut fields);
	object(fields)
}

/// What of `G.GAME.starting_params` holds after the run's start, as the
/// dump writes it (without Steamodded's play and discard limits).
fn starting_params(p: &StartingParams) -> Value {
	object(vec![
		("ante_scaling", json::number(p.ante_scaling)),
		("no_faces", p.no_faces.into()),
		("erratic_suits_and_ranks", p.erratic_suits_and_ranks.into()),
		("play_limit", Value::Null),
		("discard_limit", Value::Null),
	])
}

/// `G.GAME.modifiers` as the dump writes it (a nested table's entries as
/// `table.key`): what the game sets, and only that.
pub(super) fn modifiers(m: &Modifiers) -> Map<String, Value> {
	let mut out = Map::new();
	for slot in BlindSlot::ALL {
		if m.no_blind_reward[slot] {
			out.insert(format!("no_blind_reward.{}", slot.name()), true.into());
		}
	}
	match m.scaling {
		BlindScaling::Base => {}
		BlindScaling::Faster => _ = out.insert("scaling".into(), json::number(2.0)),
		BlindScaling::Fastest => _ = out.insert("scaling".into(), json::number(3.0)),
	}
	for (key, on) in [
		("enable_eternals_in_shop", m.enable_eternals_in_shop),
		("enable_perishables_in_shop", m.enable_perishables_in_shop),
		("enable_rentals_in_shop", m.enable_rentals_in_shop),
		("no_interest", m.no_interest),
	] {
		if on {
			out.insert(key.into(), true.into());
		}
	}
	for (key, dollars) in [
		("money_per_hand", m.money_per_hand),
		("money_per_discard", m.money_per_discard),
	] {
		if let Some(d) = dollars {
			out.insert(key.into(), json::number(d));
		}
	}
	out
}

fn rng(g: &Game) -> Value {
	let mut streams: Map<String, Value> = g
		.rng
		.named_streams()
		.map(|(k, v)| (k, json::number(v)))
		.collect();
	streams.insert("hashed_seed".into(), json::number(g.rng.hashed_seed()));
	streams.insert("seed".into(), g.rng.seed().into());
	Value::Object(streams)
}

fn blind(b: &Blind) -> Value {
	let eye = array(
		HandType::ALL
			.into_iter()
			.filter(|&h| b.eye_hands.contains(h))
			.map(|h| h.name().into()),
	);
	object([
		("name", b.kind.map_or("", BlindKind::name).into()),
		("chips", json::number(b.chips)),
		("dollars", json::number(b.dollars)),
		("mult", json::number(b.mult)),
		("disabled", b.disabled.into()),
		("triggered", b.triggered.into()),
		("prepped", b.prepped.into()),
		("hands", eye),
		("only_hand", b.only_hand.map(HandType::name).into()),
		("discards_sub", json::number(b.discards_sub)),
		("hands_sub", json::number(b.hands_sub)),
	])
}

fn hands(levels: &HandLevels) -> Value {
	object(HandType::ALL.into_iter().map(|h| {
		let l = levels[h];
		let level = object([
			("level", json::number(l.level)),
			("chips", json::number(l.chips)),
			("mult", json::number(l.mult)),
			("played", json::number(l.played)),
			("played_this_round", json::number(l.played_this_round)),
			("visible", l.visible.into()),
		]);
		(h.name(), level)
	}))
}

fn areas(g: &Game, out: &mut Vec<(&str, Value)>) {
	let a = &g.areas;
	let pack_cards = if g.flow.state.is_pack() {
		array(a.pack_cards.iter().map(any_card))
	} else {
		Value::Null
	};
	out.extend([
		("deck", array(a.deck.iter().map(playing_card))),
		("hand", array(a.hand.cards.iter().map(playing_card))),
		("discard", array(a.discard.iter().map(playing_card))),
		("jokers", array(a.jokers.iter().map(card))),
		("consumables", array(a.consumables.iter().map(card))),
		("shop_jokers", array(a.shop_jokers.iter().map(any_card))),
		("shop_vouchers", array(a.shop_vouchers.iter().map(card))),
		("shop_booster", array(a.shop_boosters.iter().map(card))),
		("pack_cards", pack_cards),
		("hand_limit", json::number(a.hand.limit.limit)),
		("hand_real_limit", json::number(a.hand.limit.real)),
		("joker_limit", json::number(a.limits.jokers.limit)),
		("joker_real_limit", json::number(a.limits.jokers.real)),
		("consumable_limit", json::number(a.limits.consumables.limit)),
		(
			"consumable_real_limit",
			json::number(a.limits.consumables.real),
		),
		(
			"sort_mode",
			match a.hand.order {
				HandSort::Rank => "desc",
				HandSort::Suit => "suit desc",
			}
			.into(),
		),
		("joker_buffer", json::number(a.joker_buffer)),
		("consumeable_buffer", json::number(a.consumable_buffer)),
		("sort_id", json::number(f64::from(a.next_sort_id))),
	]);
}

fn run(r: &Run, out: &mut Vec<(&str, Value)>) {
	out.extend([
		("ante", json::number(f64::from(r.ante))),
		("round", json::number(f64::from(r.round))),
		("dollars", json::number(r.dollars)),
		("dollar_buffer", json::number(r.dollar_buffer)),
		("interest_cap", json::number(r.interest_cap)),
		("interest_amount", json::number(r.interest_amount)),
		("bankrupt_at", json::number(r.bankrupt_at)),
		("rental_rate", json::number(r.rental_rate)),
		("perishable_rounds", json::number(r.perishable_rounds)),
		("ecto_minus", json::number(r.ecto_minus)),
		("probabilities", json::number(r.probabilities)),
		("hands_played", json::number(r.hands_played)),
		("unused_discards", json::number(r.unused_discards)),
		("skips", json::number(r.skips)),
		("starting_deck_size", json::number(r.starting_deck_size)),
		("won", r.won.into()),
		("win_notified", r.win_notified.into()),
	]);
}

fn history(h: &History, out: &mut Vec<(&str, Value)>) {
	let mut used_jokers: Vec<&str> = Center::all_cards()
		.filter(|&c| h.used_jokers.contains(c))
		.map(Center::key)
		.collect();
	used_jokers.sort_unstable();
	let mut used_vouchers: Vec<&str> = ALL_VOUCHERS
		.iter()
		.copied()
		.filter(|&v| h.used_vouchers.contains(v))
		.map(Voucher::key)
		.collect();
	used_vouchers.sort_unstable();
	let mut planets: Vec<&str> = ALL_PLANETS
		.iter()
		.filter(|&&p| h.planets_used.contains(p))
		.map(|p| p.key())
		.collect();
	planets.sort_unstable();
	let mut locked: Vec<&str> = Center::all_cards()
		.filter(|&c| h.locked.contains(c))
		.map(Center::key)
		.collect();
	locked.sort_unstable();
	let mut tags_locked: Vec<&str> = ALL_TAGS
		.iter()
		.filter(|&&t| h.locked_tags.contains(t))
		.map(|t| t.key())
		.collect();
	tags_locked.sort_unstable();
	let usage = object([
		("tarot", json::number(h.consumable_usage_tarot)),
		("planet", json::number(h.consumable_usage_planet)),
		("spectral", json::number(h.consumable_usage_spectral)),
	]);
	out.extend([
		(
			"used_jokers",
			array(used_jokers.into_iter().map(Value::from)),
		),
		(
			"used_vouchers",
			array(used_vouchers.into_iter().map(Value::from)),
		),
		("planets_used", array(planets.into_iter().map(Value::from))),
		("locked", array(locked.into_iter().map(Value::from))),
		(
			"tags_locked",
			array(tags_locked.into_iter().map(Value::from)),
		),
		("consumeable_usage_total", usage),
		("gros_michel_extinct", h.gros_michel_extinct.into()),
		(
			"last_tarot_planet",
			h.last_tarot_planet.map(Kind::key).into(),
		),
		(
			"last_hand_played",
			h.last_hand_played.map(HandType::name).into(),
		),
	]);
}

fn round(r: &Round, out: &mut Vec<(&str, Value)>) {
	let id = |rank: Option<Rank>| rank.map(|r| f64::from(r.id()));
	out.extend([
		("chips", json::number(r.chips)),
		("hands_left", json::number(r.hands_left)),
		("discards_left", json::number(r.discards_left)),
		("hands_played_round", json::number(r.hands_played)),
		("discards_used", json::number(r.discards_used)),
		("round_dollars", json::number(r.cash_out)),
		("rr_hands", json::number(r.base_hands)),
		("rr_discards", json::number(r.base_discards)),
		("temp_handsize", json::number(r.temp_handsize)),
		(
			"round_bonus",
			object([
				("next_hands", json::number(r.bonus_next_hands)),
				("discards", json::number(r.bonus_discards)),
			]),
		),
		(
			"idol",
			object([
				("rank", json::number(id(r.idol_card.id))),
				("suit", r.idol_card.suit.name().into()),
			]),
		),
		(
			"idol_card",
			object([
				("rank", r.idol_card.rank.value().into()),
				("suit", r.idol_card.suit.name().into()),
				("id", json::number(id(r.idol_card.id))),
			]),
		),
		("mail", json::number(id(r.mail_rank.id))),
		(
			"mail_card",
			object([
				("rank", r.mail_rank.rank.value().into()),
				("id", json::number(id(r.mail_rank.id))),
			]),
		),
		("ancient", r.ancient_suit.name().into()),
		("castle", r.castle_suit.name().into()),
		("most_played", r.most_played_hand.name().into()),
		("facing_blind", r.facing_blind.into()),
		("last_blind_boss", r.last_blind_boss.into()),
	]);
}

fn blind_select(b: &BlindSelect, out: &mut Vec<(&str, Value)>) {
	fn per_slot(value: impl Fn(BlindSlot) -> Value) -> Value {
		object(
			BlindSlot::ALL
				.into_iter()
				.map(|slot| (slot.name(), value(slot))),
		)
	}
	let bosses_used = object(
		ALL_BLINDS
			.iter()
			.filter(|k| k.is_boss())
			.map(|&k| (k.key(), json::number(f64::from(b.bosses_used[k])))),
	);
	out.extend([
		(
			"blind_states",
			per_slot(|slot| b.states[slot].name().into()),
		),
		("blind_choices", per_slot(|slot| b.kind(slot).key().into())),
		(
			"blind_tags",
			object(
				[BlindSlot::Small, BlindSlot::Big]
					.into_iter()
					.map(|slot| (slot.name(), b.skip_tags.of(slot).map(TagKind::key).into())),
			),
		),
		("boss_rerolled", b.boss_rerolled.into()),
		("blind_selected", b.selected.map(BlindKind::name).into()),
		("blind_on_deck", b.on_deck.map(BlindSlot::name).into()),
		("bosses_used", bosses_used),
	]);
}

fn shop(s: &Shop, out: &mut Vec<(&str, Value)>) {
	let mut used_packs: Vec<Value> = s
		.booster_slots
		.iter()
		.map(|slot| match slot {
			PackSlot::Unrolled => Value::Null,
			PackSlot::Offered(b) => b.key().into(),
			PackSlot::Opened => "USED".into(),
		})
		.collect();
	while used_packs.last() == Some(&Value::Null) {
		used_packs.pop();
	}
	out.extend([
		("shop_open", s.open.into()),
		("shop_free", s.free.into()),
		("shop_d6ed", s.d6ed.into()),
		("reroll_cost", json::number(s.reroll_cost)),
		("reroll_cost_increase", json::number(s.reroll_cost_increase)),
		("free_rerolls", json::number(s.free_rerolls)),
		("rr_reroll_cost", json::number(s.base_reroll_cost)),
		("temp_reroll_cost", json::number(s.temp_reroll_cost)),
		("used_packs", Value::Array(used_packs)),
		("voucher", s.voucher.map(Voucher::key).into()),
		("shop_joker_max", json::number(s.joker_max)),
		("discount_percent", json::number(s.discount_percent)),
		("edition_rate", json::number(s.edition_rate)),
		("joker_rate", json::number(s.joker_rate)),
		("tarot_rate", json::number(s.tarot_rate)),
		("planet_rate", json::number(s.planet_rate)),
		("spectral_rate", json::number(s.spectral_rate)),
		("playing_card_rate", json::number(s.playing_card_rate)),
		("first_shop_buffoon", s.first_buffoon.into()),
	]);
}

fn tags(t: &Tags, out: &mut Vec<(&str, Value)>) {
	let held = array(t.held.iter().map(|tag| {
		object([
			("key", tag.kind.key().into()),
			("triggered", tag.triggered.into()),
			("orbital", tag.orbital_hand.map(HandType::name).into()),
		])
	}));
	let orbital_choices = array(t.orbital_choices.iter().map(|c| {
		let mut fields = vec![("ante", json::number(f64::from(c.ante)))];
		for slot in BlindSlot::ALL {
			fields.push((slot.name(), c.hands[slot].map(HandType::name).into()));
		}
		object(fields)
	}));
	out.extend([("tag_info", held), ("orbital_choices", orbital_choices)]);
}

/// What every card has, with the fields of its `ability` table in `ability`.
fn core(
	c: &CardCore, key: &str, mut fields: Vec<(&str, Value)>, mut ability: Vec<(&str, Value)>,
) -> Value {
	ability.push(("couponed", c.couponed.into()));
	ability.push((
		"hands_played_at_create",
		json::number(c.hands_played_at_create),
	));
	fields.extend([
		("key", key.into()),
		("sort_id", json::number(f64::from(c.id))),
		("edition", c.edition.map(Edition::name).into()),
		("debuff", c.debuff.into()),
		(
			"facing",
			(if c.face_down { "back" } else { "front" }).into(),
		),
		("base_cost", json::number(c.base_cost)),
		("extra_cost", json::number(c.extra_cost)),
		("cost", json::number(c.cost)),
		("sell_cost", json::number(c.sell_cost)),
		("added_to_deck", c.added_to_deck.into()),
		("ability", object(ability)),
	]);
	object(fields)
}

fn playing_card(c: &PlayingCard) -> Value {
	let set = if c.enhancement == Enhancement::Base {
		"Default"
	} else {
		"Enhanced"
	};
	core(
		&c.core,
		c.key(),
		vec![
			("card", c.card_key().into()),
			("highlighted", c.forced.is_locked().into()),
			("rank", json::number(f64::from(c.rank.id()))),
			("suit", c.suit.name().into()),
			("seal", c.seal.map(Seal::name).into()),
			("perma_bonus", json::number(c.perma_bonus)),
			("times_played", json::number(f64::from(c.times_played))),
			(
				"suit_nominal_original",
				json::number(c.orig_suit.nominal_original()),
			),
		],
		vec![
			("set", set.into()),
			("perma_bonus", json::number(c.perma_bonus)),
			("forced_selection", c.forced.is_flagged().into()),
			("played_this_ante", c.played_this_ante.into()),
			("wheel_flipped", c.wheel_flipped.into()),
		],
	)
}

fn card(c: &Card) -> Value {
	let set = match c.kind {
		Kind::Joker(_) => "Joker",
		Kind::Tarot(_) => "Tarot",
		Kind::Planet(_) => "Planet",
		Kind::Spectral(_) => "Spectral",
		Kind::Voucher(_) => "Voucher",
		Kind::Booster(_) => "Booster",
	};
	let mut ability = vec![
		("set", set.into()),
		("queue_negative_removal", c.queue_negative_removal.into()),
	];
	match c.kind {
		Kind::Joker(j) => {
			let mut extra = Vec::new();
			// `core` writes `couponed`, a field of every card.
			let fields = JokerField::COMMON
				.iter()
				.chain(JokerField::of(j))
				.filter(|&&f| f != JokerField::Couponed);
			for &field in fields {
				let v = joker_field(field, c);
				match field.path().strip_prefix("extra.") {
					Some(key) => extra.push((key, v)),
					None => ability.push((field.path(), v)),
				}
			}
			if !extra.is_empty() {
				ability.push(("extra", object(extra)));
			}
			if let Some(Sticker::Perishable { tally }) = c.ab.sticker {
				ability.push(("perishable", true.into()));
				ability.push(("perish_tally", json::number(tally)));
			}
		}
		Kind::Tarot(_) | Kind::Planet(_) | Kind::Spectral(_) => {
			ability.push(("extra_value", json::number(c.ab.extra_value)));
		}
		Kind::Booster(_) => {
			ability.push((
				"booster_pos",
				json::number(c.booster_slot.map(|slot| f64::from(slot) + 1.0)),
			));
		}
		Kind::Voucher(_) => {}
	}
	core(&c.core, c.key(), Vec::new(), ability)
}

fn any_card(c: &AnyCard) -> Value {
	match c {
		AnyCard::Playing(c) => playing_card(c),
		AnyCard::Card(c) => card(c),
	}
}

/// The value of `field` of the joker `c` in its `ability` table.
fn joker_field(field: JokerField, c: &Card) -> Value {
	let a = &c.ab;
	match field {
		JokerField::Mult => json::number(a.mult),
		JokerField::XMult => json::number(a.x_mult),
		JokerField::ExtraValue => json::number(a.extra_value),
		JokerField::Eternal => a.eternal().into(),
		JokerField::Rental => a.rental.into(),
		JokerField::Couponed => c.core.couponed.into(),
		JokerField::ExtraChips => json::number(a.chips),
		JokerField::ExtraDollars => json::number(a.dollars),
		JokerField::Extra | JokerField::ExtraHandSize => json::number(a.extra),
		JokerField::InvisRounds => json::number(a.invis_rounds),
		JokerField::YorickDiscards => json::number(a.yorick_discards),
		JokerField::CainoXmult => json::number(a.caino_xmult),
		JokerField::LoyaltyRemaining => json::number(a.loyalty_remaining),
		JokerField::ToDoPokerHand => a.to_do.map(HandType::name).into(),
		JokerField::SteelTally => json::number(a.steel_tally),
		JokerField::StoneTally => json::number(a.stone_tally),
		JokerField::DriverTally => json::number(a.driver_tally),
		JokerField::NineTally => json::number(a.nine_tally),
	}
}
