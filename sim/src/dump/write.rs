//! A [`Game`] written as a state dump, in the shape of the game's.

use std::collections::BTreeMap;

use crate::canon::JokerField;
use crate::card::{AnyCard, Card, CardCore, Kind, PlayingCard, Sticker};
use crate::data::{
	ALL_BLINDS, ALL_PLANETS, ALL_TAGS, ALL_VOUCHERS, BlindKind, Center, Edition, Enhancement,
	HandType, Rank, Seal, TagKind, Voucher,
};
use crate::game::{
	Blind, BlindSelect, BlindSlot, Game, HandLevels, HandSort, PackSlot, Round, Shop, Tags,
};
use crate::json::Json;
use crate::round::BlindScaling;
use crate::run::{History, Modifiers, Run, StartingParams};

fn object<'key>(fields: impl IntoIterator<Item = (&'key str, Json)>) -> Json {
	Json::Obj(
		fields
			.into_iter()
			.map(|(k, v)| (k.to_string(), v))
			.collect(),
	)
}

fn array(items: impl IntoIterator<Item = Json>) -> Json {
	Json::Arr(items.into_iter().collect())
}

/// The state dump of `g`: what the game's would hold in the same state.
pub(crate) fn write(g: &Game) -> Json {
	let mut fields = vec![
		("state", g.flow.state.name().into()),
		("back", g.setup.back.key().into()),
		("stake", f64::from(u8::from(g.setup.stake)).into()),
		("win_ante", f64::from(g.setup.win_ante).into()),
		("modifiers", Json::Obj(modifiers(&g.setup.modifiers))),
		("challenge", Json::Null),
		("vouchers_in_shop", Json::Null),
		("boosters_in_shop", Json::Null),
		("starting_params", starting_params(&g.setup.starting_params)),
		("rng", rng(g)),
		("blind", blind(&g.blind)),
		("hands", hands(&g.hands)),
		(
			"pack_interrupt",
			g.pack.open.map(|p| p.interrupted.name()).into(),
		),
		("pack_size", f64::from(g.pack.size).into()),
		("pack_choices", g.pack.choices.map(f64::from).into()),
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
fn starting_params(p: &StartingParams) -> Json {
	object(vec![
		("ante_scaling", p.ante_scaling.into()),
		("no_faces", p.no_faces.into()),
		("erratic_suits_and_ranks", p.erratic_suits_and_ranks.into()),
		("play_limit", Json::Null),
		("discard_limit", Json::Null),
	])
}

/// `G.GAME.modifiers` as the dump writes it (a nested table's entries as
/// `table.key`): what the game sets, and only that.
pub(super) fn modifiers(m: &Modifiers) -> BTreeMap<String, Json> {
	let mut out = BTreeMap::new();
	for slot in BlindSlot::ALL {
		if m.no_blind_reward[slot] {
			out.insert(format!("no_blind_reward.{}", slot.name()), true.into());
		}
	}
	match m.scaling {
		BlindScaling::Base => {}
		BlindScaling::Faster => _ = out.insert("scaling".into(), 2.0.into()),
		BlindScaling::Fastest => _ = out.insert("scaling".into(), 3.0.into()),
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
			out.insert(key.into(), d.into());
		}
	}
	out
}

fn rng(g: &Game) -> Json {
	let mut streams: BTreeMap<String, Json> = g
		.rng
		.named_streams()
		.map(|(k, v)| (k, Json::Num(v)))
		.collect();
	streams.insert("hashed_seed".into(), g.rng.hashed_seed().into());
	streams.insert("seed".into(), g.rng.seed().into());
	Json::Obj(streams)
}

fn blind(b: &Blind) -> Json {
	let eye = array(
		HandType::ALL
			.into_iter()
			.filter(|&h| b.eye_hands.contains(h))
			.map(|h| h.name().into()),
	);
	object([
		("name", b.kind.map_or("", BlindKind::name).into()),
		("chips", b.chips.into()),
		("dollars", b.dollars.into()),
		("mult", b.mult.into()),
		("disabled", b.disabled.into()),
		("triggered", b.triggered.into()),
		("prepped", b.prepped.into()),
		("hands", eye),
		("only_hand", b.only_hand.map(HandType::name).into()),
		("discards_sub", b.discards_sub.into()),
		("hands_sub", b.hands_sub.into()),
	])
}

fn hands(levels: &HandLevels) -> Json {
	object(HandType::ALL.into_iter().map(|h| {
		let l = levels[h];
		let level = object([
			("level", l.level.into()),
			("chips", l.chips.into()),
			("mult", l.mult.into()),
			("played", l.played.into()),
			("played_this_round", l.played_this_round.into()),
			("visible", l.visible.into()),
		]);
		(h.name(), level)
	}))
}

fn areas(g: &Game, out: &mut Vec<(&str, Json)>) {
	let a = &g.areas;
	let pack_cards = if g.flow.state.is_pack() {
		array(a.pack_cards.iter().map(any_card))
	} else {
		Json::Null
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
		("hand_limit", a.hand.limit.limit.into()),
		("hand_real_limit", a.hand.limit.real.into()),
		("joker_limit", a.limits.jokers.limit.into()),
		("joker_real_limit", a.limits.jokers.real.into()),
		("consumable_limit", a.limits.consumables.limit.into()),
		("consumable_real_limit", a.limits.consumables.real.into()),
		(
			"sort_mode",
			match a.hand.order {
				HandSort::Rank => "desc",
				HandSort::Suit => "suit desc",
			}
			.into(),
		),
		("joker_buffer", a.joker_buffer.into()),
		("consumeable_buffer", a.consumable_buffer.into()),
		("sort_id", f64::from(a.next_sort_id).into()),
	]);
}

fn run(r: &Run, out: &mut Vec<(&str, Json)>) {
	out.extend([
		("ante", f64::from(r.ante).into()),
		("round", f64::from(r.round).into()),
		("dollars", r.dollars.into()),
		("dollar_buffer", r.dollar_buffer.into()),
		("interest_cap", r.interest_cap.into()),
		("interest_amount", r.interest_amount.into()),
		("bankrupt_at", r.bankrupt_at.into()),
		("rental_rate", r.rental_rate.into()),
		("perishable_rounds", r.perishable_rounds.into()),
		("ecto_minus", r.ecto_minus.into()),
		("probabilities", r.probabilities.into()),
		("hands_played", r.hands_played.into()),
		("unused_discards", r.unused_discards.into()),
		("skips", r.skips.into()),
		("starting_deck_size", r.starting_deck_size.into()),
		("won", r.won.into()),
		("win_notified", r.win_notified.into()),
	]);
}

fn history(h: &History, out: &mut Vec<(&str, Json)>) {
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
		("tarot", h.consumable_usage_tarot.into()),
		("planet", h.consumable_usage_planet.into()),
		("spectral", h.consumable_usage_spectral.into()),
	]);
	out.extend([
		(
			"used_jokers",
			array(used_jokers.into_iter().map(Json::from)),
		),
		(
			"used_vouchers",
			array(used_vouchers.into_iter().map(Json::from)),
		),
		("planets_used", array(planets.into_iter().map(Json::from))),
		("locked", array(locked.into_iter().map(Json::from))),
		(
			"tags_locked",
			array(tags_locked.into_iter().map(Json::from)),
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

fn round(r: &Round, out: &mut Vec<(&str, Json)>) {
	let id = |rank: Option<Rank>| rank.map(|r| f64::from(r.id()));
	out.extend([
		("chips", r.chips.into()),
		("hands_left", r.hands_left.into()),
		("discards_left", r.discards_left.into()),
		("hands_played_round", r.hands_played.into()),
		("discards_used", r.discards_used.into()),
		("round_dollars", r.cash_out.into()),
		("rr_hands", r.base_hands.into()),
		("rr_discards", r.base_discards.into()),
		("temp_handsize", r.temp_handsize.into()),
		(
			"round_bonus",
			object([
				("next_hands", r.bonus_next_hands.into()),
				("discards", r.bonus_discards.into()),
			]),
		),
		(
			"idol",
			object([
				("rank", id(r.idol_card.id).into()),
				("suit", r.idol_card.suit.name().into()),
			]),
		),
		(
			"idol_card",
			object([
				("rank", r.idol_card.rank.value().into()),
				("suit", r.idol_card.suit.name().into()),
				("id", id(r.idol_card.id).into()),
			]),
		),
		("mail", id(r.mail_rank.id).into()),
		(
			"mail_card",
			object([
				("rank", r.mail_rank.rank.value().into()),
				("id", id(r.mail_rank.id).into()),
			]),
		),
		("ancient", r.ancient_suit.name().into()),
		("castle", r.castle_suit.name().into()),
		("most_played", r.most_played_hand.name().into()),
		("facing_blind", r.facing_blind.into()),
		("last_blind_boss", r.last_blind_boss.into()),
	]);
}

fn blind_select(b: &BlindSelect, out: &mut Vec<(&str, Json)>) {
	fn per_slot(value: impl Fn(BlindSlot) -> Json) -> Json {
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
			.map(|&k| (k.key(), f64::from(b.bosses_used[k]).into())),
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

fn shop(s: &Shop, out: &mut Vec<(&str, Json)>) {
	let mut used_packs: Vec<Json> = s
		.booster_slots
		.iter()
		.map(|slot| match slot {
			PackSlot::Unrolled => Json::Null,
			PackSlot::Offered(b) => b.key().into(),
			PackSlot::Opened => "USED".into(),
		})
		.collect();
	while used_packs.last() == Some(&Json::Null) {
		used_packs.pop();
	}
	out.extend([
		("shop_open", s.open.into()),
		("shop_free", s.free.into()),
		("shop_d6ed", s.d6ed.into()),
		("reroll_cost", s.reroll_cost.into()),
		("reroll_cost_increase", s.reroll_cost_increase.into()),
		("free_rerolls", s.free_rerolls.into()),
		("rr_reroll_cost", s.base_reroll_cost.into()),
		("temp_reroll_cost", s.temp_reroll_cost.into()),
		("used_packs", Json::Arr(used_packs)),
		("voucher", s.voucher.map(Voucher::key).into()),
		("shop_joker_max", s.joker_max.into()),
		("discount_percent", s.discount_percent.into()),
		("edition_rate", s.edition_rate.into()),
		("joker_rate", s.joker_rate.into()),
		("tarot_rate", s.tarot_rate.into()),
		("planet_rate", s.planet_rate.into()),
		("spectral_rate", s.spectral_rate.into()),
		("playing_card_rate", s.playing_card_rate.into()),
		("first_shop_buffoon", s.first_buffoon.into()),
	]);
}

fn tags(t: &Tags, out: &mut Vec<(&str, Json)>) {
	let held = array(t.held.iter().map(|tag| {
		object([
			("key", tag.kind.key().into()),
			("triggered", tag.triggered.into()),
			("orbital", tag.orbital_hand.map(HandType::name).into()),
		])
	}));
	let orbital_choices = array(t.orbital_choices.iter().map(|c| {
		let mut fields = vec![("ante", Json::Num(f64::from(c.ante)))];
		for slot in BlindSlot::ALL {
			fields.push((slot.name(), c.hands[slot].map(HandType::name).into()));
		}
		object(fields)
	}));
	out.extend([("tag_info", held), ("orbital_choices", orbital_choices)]);
}

/// What every card has, with the fields of its `ability` table in `ability`.
fn core(
	c: &CardCore, key: &str, mut fields: Vec<(&str, Json)>, mut ability: Vec<(&str, Json)>,
) -> Json {
	ability.push(("couponed", c.couponed.into()));
	ability.push(("hands_played_at_create", c.hands_played_at_create.into()));
	fields.extend([
		("key", key.into()),
		("sort_id", f64::from(c.id).into()),
		("edition", c.edition.map(Edition::name).into()),
		("debuff", c.debuff.into()),
		(
			"facing",
			(if c.face_down { "back" } else { "front" }).into(),
		),
		("base_cost", c.base_cost.into()),
		("extra_cost", c.extra_cost.into()),
		("cost", c.cost.into()),
		("sell_cost", c.sell_cost.into()),
		("added_to_deck", c.added_to_deck.into()),
		("ability", object(ability)),
	]);
	object(fields)
}

fn playing_card(c: &PlayingCard) -> Json {
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
			("rank", f64::from(c.rank.id()).into()),
			("suit", c.suit.name().into()),
			("seal", c.seal.map(Seal::name).into()),
			("perma_bonus", c.perma_bonus.into()),
			("times_played", f64::from(c.times_played).into()),
			(
				"suit_nominal_original",
				c.orig_suit.nominal_original().into(),
			),
		],
		vec![
			("set", set.into()),
			("perma_bonus", c.perma_bonus.into()),
			("forced_selection", c.forced.is_flagged().into()),
			("played_this_ante", c.played_this_ante.into()),
			("wheel_flipped", c.wheel_flipped.into()),
		],
	)
}

fn card(c: &Card) -> Json {
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
				ability.push(("perish_tally", tally.into()));
			}
		}
		Kind::Tarot(_) | Kind::Planet(_) | Kind::Spectral(_) => {
			ability.push(("extra_value", c.ab.extra_value.into()));
		}
		Kind::Booster(_) => {
			ability.push((
				"booster_pos",
				c.booster_slot.map(|slot| f64::from(slot) + 1.0).into(),
			));
		}
		Kind::Voucher(_) => {}
	}
	core(&c.core, c.key(), Vec::new(), ability)
}

fn any_card(c: &AnyCard) -> Json {
	match c {
		AnyCard::Playing(c) => playing_card(c),
		AnyCard::Card(c) => card(c),
	}
}

/// The value of `field` of the joker `c` in its `ability` table.
fn joker_field(field: JokerField, c: &Card) -> Json {
	let a = &c.ab;
	match field {
		JokerField::Mult => a.mult.into(),
		JokerField::XMult => a.x_mult.into(),
		JokerField::ExtraValue => a.extra_value.into(),
		JokerField::Eternal => a.eternal().into(),
		JokerField::Rental => a.rental.into(),
		JokerField::Couponed => c.core.couponed.into(),
		JokerField::ExtraChips => a.chips.into(),
		JokerField::ExtraDollars => a.dollars.into(),
		JokerField::Extra | JokerField::ExtraHandSize => a.extra.into(),
		JokerField::InvisRounds => a.invis_rounds.into(),
		JokerField::YorickDiscards => a.yorick_discards.into(),
		JokerField::CainoXmult => a.caino_xmult.into(),
		JokerField::LoyaltyRemaining => a.loyalty_remaining.into(),
		JokerField::ToDoPokerHand => a.to_do.map(HandType::name).into(),
		JokerField::SteelTally => a.steel_tally.into(),
		JokerField::StoneTally => a.stone_tally.into(),
		JokerField::DriverTally => a.driver_tally.into(),
		JokerField::NineTally => a.nine_tally.into(),
	}
}
