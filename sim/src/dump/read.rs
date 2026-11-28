//! A state dump read into a [`Game`].

use serde_json::Value;

use super::parse::{
	DumpError, at, bad, blind_of_name, by_name, entries, flag, int, kind_of, list, named, number,
	object, opt_flag, opt_int, opt_named, opt_number, opt_text, rank_of_value, raw_named,
	state_of_name, text, whole,
};
use super::read_cards::{any_cards, cards, playing_cards};
use crate::areas::{Hand, Limit, Limits};
use crate::card::{HandSort, Kind};
use crate::data::{
	Back, BlindKind, CenterSet, Enhancement, HandSet, HandType, PackKind, Planet, PlanetSet, Rank,
	Suit, TagKind, TagSet, Voucher, VoucherSet,
};
use crate::event::EventQueue;
use crate::game::{
	Areas, Blind, BlindSelect, BlindSlot, BlindStatus, BossesUsed, Flow, Game, HandLevel,
	HandLevels, IdolCard, MailRank, OpenPack, OrbitalChoices, PackSlot, PackState, PerBlind, Round,
	Setup, Shop, SkipTags, Stake, State, Tag, Tags,
};
use crate::rng::{Key, PseudoRandom};
use crate::run::{History, Run};
use crate::shop::BOOSTER_SLOTS;
use crate::{MAX_SEED_LEN, canon, flow, num};

/// The game in the state dump `j`.
///
/// # Errors
///
/// Returns the first field the simulator cannot take: a missing or
/// malformed value, an unknown key or name, a deck that needs challenge
/// rules, or a state that is not a decision point.
pub(crate) fn read(j: &Value) -> Result<Game, DumpError> {
	let state = named(j, "state", state_of_name)?;
	if matches!(
		state,
		State::HandPlayed | State::DrawToHand | State::NewRound | State::PlayTarot
	) {
		return Err(DumpError(format!(
			"{} is not a decision point",
			state.name()
		)));
	}
	let mut flow = Flow {
		state,
		state_complete: true,
		..Flow::default()
	};
	flow::sync_phase(&mut flow);
	let run = read_run(j, state)?;
	let blind_select = read_blind_select(j)?;
	let mut shop = read_shop(j)?;
	shop.voucher_seen = voucher_seen(state, run.round, blind_select.states[BlindSlot::Boss]);
	let mut g = Game {
		rng: read_rng(object(j, "rng")?)?,
		events: EventQueue::new(),
		setup: read_setup(j)?,
		flow,
		areas: read_areas(j)?,
		run,
		history: read_history(j)?,
		round: read_round(j)?,
		blind_select,
		blind: read_blind(object(j, "blind")?)?,
		shop,
		pack: read_pack(j, state)?,
		hands: read_hands(object(j, "hands")?)?,
		tags: read_tags(j)?,
	};
	g.update_derived();
	Ok(g)
}

/// Whether the ante's voucher has been shown in a shop: by the first shop
/// of the run, until the boss is beaten and the next ante's voucher rolled
/// (it shows in the shop after the boss).
fn voucher_seen(state: State, round: i32, boss: BlindStatus) -> bool {
	let rolled_now = state == State::RoundEval && boss == BlindStatus::Defeated;
	let shop_shown = round >= 2
		|| (round == 1
			&& !matches!(
				state,
				State::SelectingHand | State::RoundEval | State::GameOver
			));
	shop_shown && !rolled_now
}

fn read_setup(j: &Value) -> Result<Setup, DumpError> {
	let back = named(j, "back", |k| {
		Back::from_key(k).filter(|b| b.is_simulated())
	})?;
	let stake = Stake::try_from(int::<u8>(j, "stake")?).map_err(|e| DumpError(e.to_string()))?;
	let setup = Setup::new(back, stake);
	if let Some(c) = opt_text(j, "challenge")? {
		return Err(DumpError(format!("a challenge run ({c})")));
	}
	// Steamodded's shop sizes (vanilla has neither).
	for (key, vanilla) in [("vouchers_in_shop", 1.0), ("boosters_in_shop", 2.0)] {
		if opt_number(j, key)?.is_some_and(|n| n != vanilla) {
			return Err(bad(key, at(j, key)?));
		}
	}
	// What of the starting rules holds after the run's start: the deck's, and
	// Steamodded's card limits (vanilla has none) as the simulator's.
	let p = object(j, "starting_params")?;
	let sp = &setup.starting_params;
	let mut limits = Vec::new();
	for key in ["play_limit", "discard_limit"] {
		limits.extend(opt_number(p, key)?);
	}
	if number(p, "ante_scaling")? != sp.ante_scaling
		|| flag(p, "no_faces")? != sp.no_faces
		|| flag(p, "erratic_suits_and_ranks")? != sp.erratic_suits_and_ranks
		|| limits
			.into_iter()
			.any(|n| n != num::float(Hand::HIGHLIGHT_LIMIT))
	{
		return Err(bad("starting_params", p));
	}
	let rules = entries(j, "modifiers")?;
	let want = super::write::modifiers(&setup.modifiers);
	let want: Vec<(&str, &Value)> = want.iter().map(|(k, v)| (k.as_str(), v)).collect();
	if rules != want {
		return Err(DumpError(format!(
			"rules the simulator does not have: {rules:?} (the deck and stake make {want:?})"
		)));
	}
	Ok(Setup {
		win_ante: int(j, "win_ante")?,
		..setup
	})
}

fn read_rng(j: &Value) -> Result<PseudoRandom, DumpError> {
	use crate::json::ValueExt;

	let seed = text(j, "seed")?;
	if seed.len() > MAX_SEED_LEN {
		return Err(bad("seed", &j["seed"]));
	}
	let mut rng = PseudoRandom::new(seed);
	if number(j, "hashed_seed")? != rng.hashed_seed() {
		return Err(bad("hashed_seed", &j["hashed_seed"]));
	}
	let streams = j.as_object().ok_or_else(|| bad("rng", j))?;
	for (k, v) in streams {
		if k == "seed" || k == "hashed_seed" {
			continue;
		}
		// A stream's state starts from its key followed by the seed.
		if k.len() + seed.len() > Key::CAPACITY {
			return Err(DumpError(format!("RNG stream {k:?} is too long")));
		}
		let state = v.number().ok_or_else(|| bad(k, v))?;
		rng.set_state(&Key::from(k.as_str()), state);
	}
	Ok(rng)
}

fn read_run(j: &Value, state: State) -> Result<Run, DumpError> {
	let round = int(j, "round")?;
	let skips = number(j, "skips")?;
	// Blinds beaten or skipped: every round started was beaten but one being
	// played or lost (none before the first).
	let unfinished = u32::from(matches!(state, State::SelectingHand | State::GameOver));
	let blinds_cleared = u32::try_from(round)
		.ok()
		.map(|r| r.saturating_sub(unfinished))
		.and_then(|beaten| beaten.checked_add(whole(j, "skips", skips).ok()?))
		.ok_or_else(|| bad("round", &j["round"]))?;
	Ok(Run {
		ante: int(j, "ante")?,
		round,
		dollars: number(j, "dollars")?,
		dollar_buffer: opt_number(j, "dollar_buffer")?.unwrap_or(0.0),
		interest_cap: number(j, "interest_cap")?,
		interest_amount: number(j, "interest_amount")?,
		bankrupt_at: number(j, "bankrupt_at")?,
		rental_rate: number(j, "rental_rate")?,
		perishable_rounds: number(j, "perishable_rounds")?,
		ecto_minus: number(j, "ecto_minus")?,
		probabilities: number(j, "probabilities")?,
		hands_played: number(j, "hands_played")?,
		unused_discards: number(j, "unused_discards")?,
		skips,
		starting_deck_size: number(j, "starting_deck_size")?,
		blinds_cleared,
		won: flag(j, "won")?,
		win_notified: flag(j, "win_notified")?,
		last_hand_score: 0.0,
		last_hand_chips: 0.0,
		last_hand_mult: 0.0,
		last_hand_blind: 0.0,
		earned: 0.0,
	})
}

fn read_history(j: &Value) -> Result<History, DumpError> {
	use crate::json::ValueExt;

	let mut used_jokers = CenterSet::default();
	for k in list(j, "used_jokers")? {
		let k = k.as_str().ok_or_else(|| bad("used_jokers", k))?;
		// The game also marks centers that are in no pool: the playing
		// cards' (`c_base`, the enhancements) and its placeholders.
		if k == "c_base" || Enhancement::from_key(k).is_some() || canon::placeholder_center(k) {
			continue;
		}
		let kind = kind_of(k).ok_or_else(|| bad("used_jokers", &Value::String(k.into())))?;
		used_jokers.insert(kind.center());
	}
	let mut used_vouchers = VoucherSet::default();
	for k in list(j, "used_vouchers")? {
		let v = k.as_str().and_then(Voucher::from_key);
		used_vouchers.insert(v.ok_or_else(|| bad("used_vouchers", k))?);
	}
	let mut planets_used = PlanetSet::default();
	for k in list(j, "planets_used")? {
		let p = k.as_str().and_then(Planet::from_key);
		planets_used.insert(p.ok_or_else(|| bad("planets_used", k))?);
	}
	// nil until a consumable is used
	let usage = at(j, "consumeable_usage_total")?;
	if !matches!(usage, Value::Null | Value::Object(_)) {
		return Err(bad("consumeable_usage_total", usage));
	}
	let used = |set: &str| match &usage[set] {
		Value::Null => Ok(0.0),
		v => v.number().ok_or_else(|| bad(set, v)),
	};
	// Another mod's cards are in no pool of the simulator's anyway.
	let mut locked = CenterSet::default();
	for k in list(j, "locked")? {
		let k = k.as_str().ok_or_else(|| bad("locked", k))?;
		if let Some(kind) = kind_of(k) {
			locked.insert(kind.center());
		}
	}
	let mut locked_tags = TagSet::default();
	for k in list(j, "tags_locked")? {
		let k = k.as_str().ok_or_else(|| bad("tags_locked", k))?;
		if let Some(t) = TagKind::from_key(k) {
			locked_tags.insert(t);
		}
	}
	Ok(History {
		locked,
		locked_tags,
		used_jokers,
		used_vouchers,
		planets_used,
		gros_michel_extinct: flag(j, "gros_michel_extinct")?,
		consumable_usage_tarot: used("tarot")?,
		consumable_usage_planet: used("planet")?,
		consumable_usage_spectral: used("spectral")?,
		last_tarot_planet: opt_named(j, "last_tarot_planet", |k| {
			kind_of(k).filter(|k| matches!(k, Kind::Tarot(_) | Kind::Planet(_)))
		})?,
		last_hand_played: opt_named(j, "last_hand_played", HandType::from_name)?,
	})
}

fn read_round(j: &Value) -> Result<Round, DumpError> {
	// what the canonical state reads of the cards of The Idol and Mail-In
	// Rebate (the simulator reads `idol_card` and `mail_card`)
	at(j, "idol")?;
	at(j, "mail")?;
	let bonus = object(j, "round_bonus")?;
	let idol = object(j, "idol_card")?;
	let mail = object(j, "mail_card")?;
	let rank_id = |c: &Value| {
		opt_int::<u8>(c, "id")?
			.map(|id| Rank::from_id(id).ok_or_else(|| bad("id", &c["id"])))
			.transpose()
	};
	Ok(Round {
		chips: number(j, "chips")?,
		hands_left: number(j, "hands_left")?,
		discards_left: number(j, "discards_left")?,
		hands_played: number(j, "hands_played_round")?,
		discards_used: number(j, "discards_used")?,
		cash_out: number(j, "round_dollars")?,
		base_hands: number(j, "rr_hands")?,
		base_discards: number(j, "rr_discards")?,
		temp_handsize: opt_number(j, "temp_handsize")?,
		bonus_next_hands: number(bonus, "next_hands")?,
		bonus_discards: number(bonus, "discards")?,
		idol_card: IdolCard {
			rank: named(idol, "rank", rank_of_value)?,
			suit: named(idol, "suit", Suit::from_name)?,
			id: rank_id(idol)?,
		},
		mail_rank: MailRank {
			rank: named(mail, "rank", rank_of_value)?,
			id: rank_id(mail)?,
		},
		ancient_suit: named(j, "ancient", Suit::from_name)?,
		castle_suit: named(j, "castle", Suit::from_name)?,
		most_played_hand: named(j, "most_played", HandType::from_name)?,
		facing_blind: flag(j, "facing_blind")?,
		last_blind_boss: flag(j, "last_blind_boss")?,
	})
}

fn read_blind_select(j: &Value) -> Result<BlindSelect, DumpError> {
	let states = object(j, "blind_states")?;
	let tags = object(j, "blind_tags")?;
	let mut out = BlindSelect {
		boss: named(object(j, "blind_choices")?, "Boss", BlindKind::from_key)?,
		skip_tags: SkipTags {
			small: raw_named(tags, "Small", TagKind::from_key)?,
			big: raw_named(tags, "Big", TagKind::from_key)?,
		},
		boss_rerolled: flag(j, "boss_rerolled")?,
		selected: opt_named(j, "blind_selected", blind_of_name)?,
		on_deck: opt_named(j, "blind_on_deck", |s| {
			by_name(&BlindSlot::ALL, BlindSlot::name, s)
		})?,
		states: PerBlind::all(BlindStatus::Upcoming),
		bosses_used: BossesUsed::default(),
	};
	for slot in BlindSlot::ALL {
		out.states[slot] = named(states, slot.name(), |s| {
			by_name(&BlindStatus::ALL, BlindStatus::name, s)
		})?;
	}
	let used = at(j, "bosses_used")?;
	for (k, _) in entries(j, "bosses_used")? {
		let boss = BlindKind::from_key(k).ok_or_else(|| bad("bosses_used", used))?;
		out.bosses_used[boss] = int(used, k)?;
	}
	Ok(out)
}

fn read_blind(j: &Value) -> Result<Blind, DumpError> {
	let kind = match opt_text(j, "name")? {
		None | Some("") => None,
		Some(name) => Some(blind_of_name(name).ok_or_else(|| bad("blind.name", &j["name"]))?),
	};
	// The game's blind keeps The Eye's and The Mouth's hands until it is that
	// blind again; the simulator's blind has them only while it is.
	let mut eye_hands = HandSet::default();
	if kind == Some(BlindKind::Eye) {
		for h in list(j, "hands")? {
			let hand = h.as_str().and_then(HandType::from_name);
			eye_hands.insert(hand.ok_or_else(|| bad("blind.hands", h))?);
		}
	}
	let only_hand = if kind == Some(BlindKind::Mouth) {
		opt_named(j, "only_hand", HandType::from_name)?
	} else {
		None
	};
	Ok(Blind {
		kind,
		chips: number(j, "chips")?,
		dollars: number(j, "dollars")?,
		mult: number(j, "mult")?,
		disabled: opt_flag(j, "disabled")?,
		triggered: flag(j, "triggered")?,
		prepped: flag(j, "prepped")?,
		eye_hands,
		only_hand,
		discards_sub: opt_number(j, "discards_sub")?.unwrap_or(0.0),
		hands_sub: opt_number(j, "hands_sub")?.unwrap_or(0.0),
	})
}

fn read_shop(j: &Value) -> Result<Shop, DumpError> {
	let used_packs = list(j, "used_packs")?;
	if used_packs.len() > BOOSTER_SLOTS {
		return Err(bad("used_packs", &j["used_packs"]));
	}
	let mut booster_slots = [PackSlot::Unrolled; BOOSTER_SLOTS];
	for (slot, p) in booster_slots.iter_mut().zip(used_packs) {
		*slot = match p.as_str() {
			Some("USED") => PackSlot::Opened,
			Some(k) => match kind_of(k) {
				Some(Kind::Booster(b)) => PackSlot::Offered(b),
				_ => return Err(bad("used_packs", p)),
			},
			None if *p == Value::Null => PackSlot::Unrolled,
			None => return Err(bad("used_packs", p)),
		};
	}
	Ok(Shop {
		open: flag(j, "shop_open")?,
		free: flag(j, "shop_free")?,
		d6ed: flag(j, "shop_d6ed")?,
		reroll_cost: number(j, "reroll_cost")?,
		reroll_cost_increase: number(j, "reroll_cost_increase")?,
		free_rerolls: number(j, "free_rerolls")?,
		base_reroll_cost: number(j, "rr_reroll_cost")?,
		temp_reroll_cost: opt_number(j, "temp_reroll_cost")?,
		booster_slots,
		voucher: opt_named(j, "voucher", Voucher::from_key)?,
		voucher_seen: false,
		joker_max: number(j, "shop_joker_max")?,
		discount_percent: number(j, "discount_percent")?,
		edition_rate: number(j, "edition_rate")?,
		joker_rate: number(j, "joker_rate")?,
		tarot_rate: number(j, "tarot_rate")?,
		planet_rate: number(j, "planet_rate")?,
		spectral_rate: number(j, "spectral_rate")?,
		playing_card_rate: number(j, "playing_card_rate")?,
		first_buffoon: flag(j, "first_shop_buffoon")?,
	})
}

fn read_pack(j: &Value, state: State) -> Result<PackState, DumpError> {
	at(j, "pack_interrupt")?;
	let open = if state.is_pack() {
		let kind = match state {
			State::TarotPack => PackKind::Arcana,
			State::PlanetPack => PackKind::Celestial,
			State::SpectralPack => PackKind::Spectral,
			State::StandardPack => PackKind::Standard,
			_ => PackKind::Buffoon,
		};
		Some(OpenPack {
			kind,
			interrupted: named(j, "pack_interrupt", state_of_name)?,
		})
	} else {
		None
	};
	Ok(PackState {
		open,
		size: int(j, "pack_size")?,
		choices: opt_int(j, "pack_choices")?,
	})
}

fn read_hands(j: &Value) -> Result<HandLevels, DumpError> {
	let mut out = HandLevels(
		[HandLevel {
			level: 0.0,
			chips: 0.0,
			mult: 0.0,
			played: 0.0,
			played_this_round: 0.0,
			visible: false,
		}; HandType::ALL.len()],
	);
	for h in HandType::ALL {
		let l = at(j, h.name())?;
		out[h] = HandLevel {
			level: number(l, "level")?,
			chips: number(l, "chips")?,
			mult: number(l, "mult")?,
			played: number(l, "played")?,
			played_this_round: number(l, "played_this_round")?,
			visible: flag(l, "visible")?,
		};
	}
	Ok(out)
}

fn read_tags(j: &Value) -> Result<Tags, DumpError> {
	let mut held = Vec::new();
	let numbered = list(j, "tag_info")?.iter().zip(1..);
	for (t, id) in numbered {
		let kind = named(t, "key", TagKind::from_key)?;
		// Every tag has an `orbital_hand`, a placeholder but the Orbital Tag's.
		let orbital_hand = if kind == TagKind::Orbital {
			opt_named(t, "orbital", HandType::from_name)?
		} else {
			None
		};
		held.push(Tag {
			kind,
			triggered: flag(t, "triggered")?,
			orbital_hand,
			id,
		});
	}
	let mut orbital_choices = Vec::new();
	for c in list(j, "orbital_choices")? {
		let mut hands = PerBlind::all(None);
		for slot in BlindSlot::ALL {
			hands[slot] = opt_named(c, slot.name(), HandType::from_name)?;
		}
		orbital_choices.push(OrbitalChoices {
			ante: int(c, "ante")?,
			hands,
		});
	}
	Ok(Tags {
		next_id: num::index32(held.len()),
		held,
		orbital_choices,
	})
}

fn read_areas(j: &Value) -> Result<Areas, DumpError> {
	let limit = |limit: &str, real: &str| -> Result<Limit, DumpError> {
		let limit = number(j, limit)?;
		Ok(Limit {
			limit,
			real: opt_number(j, real)?.unwrap_or(limit),
		})
	};
	let order = match text(j, "sort_mode")? {
		"desc" => HandSort::Rank,
		"suit desc" => HandSort::Suit,
		_ => return Err(bad("sort_mode", &j["sort_mode"])),
	};
	Ok(Areas {
		deck: playing_cards(j, "deck")?,
		hand: Hand {
			cards: playing_cards(j, "hand")?,
			limit: limit("hand_limit", "hand_real_limit")?,
			order,
		},
		discard: playing_cards(j, "discard")?,
		jokers: cards(j, "jokers")?,
		consumables: cards(j, "consumables")?,
		limits: Limits {
			jokers: limit("joker_limit", "joker_real_limit")?,
			consumables: limit("consumable_limit", "consumable_real_limit")?,
		},
		shop_jokers: any_cards(j, "shop_jokers")?,
		shop_vouchers: cards(j, "shop_vouchers")?,
		shop_boosters: cards(j, "shop_booster")?,
		// The game has the area only while a pack is open.
		pack_cards: match at(j, "pack_cards")? {
			Value::Null => Vec::new(),
			_ => any_cards(j, "pack_cards")?,
		},
		joker_buffer: number(j, "joker_buffer")?,
		consumable_buffer: number(j, "consumeable_buffer")?,
		next_sort_id: int(j, "sort_id")?,
		..Areas::default()
	})
}

#[cfg(test)]
mod tests {
	use serde_json::Map;

	use super::*;
	use crate::data::{Center, Joker};
	use crate::inject::Inject;
	use crate::json::ValueExt;

	/// A dump of a new run with `change` made to its top-level fields.
	fn changed(change: impl FnOnce(&mut Map<String, Value>)) -> Value {
		let mut dump = Game::new_run("TESTSEED", Back::Red, Stake::White).to_dump();
		let Value::Object(fields) = &mut dump else {
			panic!("a dump is an object");
		};
		change(fields);
		dump
	}

	#[test]
	fn a_new_run_reads_back() {
		assert!(read(&changed(|_| {})).is_ok());
	}

	#[test]
	fn a_stream_key_too_long_with_the_seed_is_refused() {
		let dump = changed(|f| {
			let Some(Value::Object(rng)) = f.get_mut("rng") else {
				panic!("a dump has its RNG streams");
			};
			rng.insert("k".repeat(Key::CAPACITY - 4), Value::from(0.5));
		});
		assert!(read(&dump).is_err());
	}

	/// Removes the key `path` (`key`, or `area.key` of the area's first card).
	fn without(dump: &Value, path: &str) -> Value {
		let mut dump = dump.clone();
		let Value::Object(fields) = &mut dump else {
			panic!("a dump is an object");
		};
		match path.split_once('.') {
			None => fields.remove(path),
			Some((area, key)) => {
				let Some(Value::Array(cards)) = fields.get_mut(area) else {
					panic!("the dump has the area {area}");
				};
				let Some(Value::Object(card)) = cards.first_mut() else {
					panic!("the area {area} has a card");
				};
				card.remove(key)
			}
		};
		dump
	}

	#[test]
	fn a_dump_without_any_key_it_is_written_with_is_refused() {
		let mut g = Game::new_run("TESTSEED", Back::Red, Stake::White);
		g.inject(Inject::joker(Joker::Joker));
		let dump = g.to_dump();
		let mut paths: Vec<String> = dump
			.as_object()
			.expect("a dump is an object")
			.keys()
			.cloned()
			.collect();
		for area in ["deck", "jokers"] {
			let card = dump[area].list().first().expect("the area has a card");
			let keys = card.as_object().expect("a card is an object").keys();
			paths.extend(keys.map(|k| format!("{area}.{k}")));
		}
		let accepted: Vec<&str> = paths
			.iter()
			.map(String::as_str)
			.filter(|p| read(&without(&dump, p)).is_ok())
			.collect();
		assert!(
			accepted.is_empty(),
			"keys a dump can be without: {accepted:?}"
		);
	}

	#[test]
	fn rules_the_simulator_does_not_have_are_refused() {
		let set = |key: &str, v: Value| changed(|f| _ = f.insert(key.into(), v));
		let start = |key: &str, v: Value| {
			changed(|f| {
				let Some(Value::Object(p)) = f.get_mut("starting_params") else {
					panic!("a dump has its starting rules");
				};
				p.insert(key.into(), v);
			})
		};
		let mut rules = Map::new();
		rules.insert("no_interest".to_string(), Value::Bool(true)); // a challenge's, not the Red Deck's
		for dump in [
			set("modifiers", Value::Object(rules)),
			set("challenge", Value::String("c_omelette_1".into())),
			set("vouchers_in_shop", Value::from(2.0)),
			start("ante_scaling", Value::from(2.0)), // the Plasma Deck's, not the Red Deck's
			start("play_limit", Value::from(6.0)),
			set(
				"voucher",
				Value::Array(vec![
					Value::String("v_grabber".into()),
					Value::String("v_wasteful".into()),
				]),
			),
		] {
			assert!(read(&dump).is_err(), "{:?}", [
				&dump["modifiers"],
				&dump["challenge"]
			]);
		}
		// Steamodded's shop sizes as vanilla's, and no rules: a White Stake Red Deck run
		let plain = changed(|f| {
			f.insert("vouchers_in_shop".into(), Value::from(1.0));
			f.insert("boosters_in_shop".into(), Value::from(2.0));
			let Some(Value::Object(p)) = f.get_mut("starting_params") else {
				panic!("a dump has its starting rules");
			};
			p.insert("play_limit".into(), Value::from(5.0));
			p.insert("discard_limit".into(), Value::from(5.0));
		});
		assert!(read(&plain).is_ok());
		let plasma = Game::new_run("TESTSEED", Back::Plasma, Stake::White).to_dump();
		assert!(
			read(&plasma).is_ok(),
			"the Plasma Deck's blinds scale twice as fast"
		);
		let red = Game::new_run("TESTSEED", Back::Red, Stake::Red).to_dump();
		assert!(
			read(&red).is_ok(),
			"a Red Stake run has the rule of no reward for the Small Blind"
		);
	}

	#[test]
	fn a_loss_before_the_first_blind_reads() {
		let dump = changed(|f| {
			f.insert("state".into(), Value::String("GAME_OVER".into()));
			f.insert("round".into(), Value::from(0.0));
		});
		let g = read(&dump).expect("a game over at round 0");
		assert!(g.is_terminal() && g.run.blinds_cleared == 0);
	}

	#[test]
	fn the_profiles_locks_are_read() {
		let dump = changed(|f| {
			f.insert(
				"locked".into(),
				Value::Array(vec![
					Value::String("j_blueprint".into()),
					Value::String("j_mod_card".into()),
				]),
			);
			f.insert(
				"tags_locked".into(),
				Value::Array(vec![Value::String("tag_rare".into())]),
			);
		});
		let mut g = read(&dump).expect("another mod's locked card is in no pool anyway");
		assert!(g.history.locked.contains(Center::Joker(Joker::Blueprint)));
		assert!(g.history.locked_tags.contains(TagKind::Rare));
		let back = read(&g.to_dump()).expect("written back");
		assert_eq!(back.history, g.history);
	}

	#[test]
	fn used_jokers_holds_the_cards_the_simulator_has() {
		let used = |keys: &[&str]| {
			changed(|f| {
				let keys = keys.iter().map(|&k| Value::String(k.into())).collect();
				f.insert("used_jokers".into(), Value::Array(keys));
			})
		};
		let g = read(&used(&["c_base", "m_glass", "j_locked", "j_joker"]))
			.expect("the game marks a placeholder in an unlock alert");
		let mut want = CenterSet::default();
		want.insert(Center::Joker(Joker::Joker));
		assert_eq!(g.history.used_jokers, want);
		assert!(read(&used(&["j_mod_card"])).is_err(), "another mod's card");
	}

	#[test]
	fn a_negative_round_is_refused() {
		let dump = changed(|f| {
			f.insert("round".into(), Value::from(-1.0));
		});
		assert!(read(&dump).is_err());
	}
}
