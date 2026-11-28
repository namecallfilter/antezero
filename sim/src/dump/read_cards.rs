//! Reading playing cards, jokers, consumables, vouchers, and boosters from a
//! state dump.

use serde_json::Value;

use super::parse::{
	DumpError, at, bad, by_name, flag, int, kind_of, list, named, number, opt_named, raw_flag,
	raw_named, text, whole,
};
use crate::canon::JokerField;
use crate::card::{AnyCard, Card, CardCore, Forced, Kind, PlayingCard, Sticker};
use crate::data::{Edition, Enhancement, HandType, Rank, Seal, Suit};

pub(super) fn playing_cards(j: &Value, area: &str) -> Result<Vec<PlayingCard>, DumpError> {
	list(j, area)?.iter().map(playing_card).collect()
}

pub(super) fn cards(j: &Value, area: &str) -> Result<Vec<Card>, DumpError> {
	list(j, area)?.iter().map(card).collect()
}

/// The cards of an area that holds both kinds (the shop's card row, a pack).
pub(super) fn any_cards(j: &Value, area: &str) -> Result<Vec<AnyCard>, DumpError> {
	list(j, area)?
		.iter()
		.map(|c| {
			let set = c["ability"]["set"].as_str();
			Ok(if matches!(set, Some("Default" | "Enhanced")) {
				playing_card(c)?.into()
			} else {
				card(c)?.into()
			})
		})
		.collect()
}

/// What every card has.
fn core(c: &Value) -> Result<CardCore, DumpError> {
	let ab = &c["ability"];
	Ok(CardCore {
		id: int(c, "sort_id")?,
		edition: opt_named(c, "edition", |s| by_name(&Edition::ALL, Edition::name, s))?,
		debuff: flag(c, "debuff")?,
		face_down: match text(c, "facing")? {
			"front" => false,
			"back" => true,
			_ => return Err(bad("facing", &c["facing"])),
		},
		base_cost: number(c, "base_cost")?,
		extra_cost: number(c, "extra_cost")?,
		cost: number(c, "cost")?,
		sell_cost: number(c, "sell_cost")?,
		couponed: raw_flag(ab, "couponed")?,
		hands_played_at_create: number(ab, "hands_played_at_create")?,
		added_to_deck: flag(c, "added_to_deck")?,
	})
}

/// The playing card's place in its area's highlight list (from 0), `None`
/// if it is not highlighted. The mod writes it since version 0.3.1.
fn highlighted(c: &Value) -> Result<Option<u8>, DumpError> {
	match c.get("highlighted") {
		None | Some(Value::Bool(true)) => Err(DumpError(
			"no highlighted place: the dump is of an AnteZero mod before 0.3.1, update the mod"
				.into(),
		)),
		Some(Value::Bool(false)) => Ok(None),
		Some(v) => {
			let nth: u16 = int(c, "highlighted")?;
			let place = nth
				.checked_sub(1)
				.and_then(|place| u8::try_from(place).ok());
			place.map(Some).ok_or_else(|| bad("highlighted", v))
		}
	}
}

fn playing_card(c: &Value) -> Result<PlayingCard, DumpError> {
	let ab = &c["ability"];
	// what the canonical state reads (the simulator has them from `rank`,
	// `suit` and the ability)
	at(c, "card")?;
	at(c, "perma_bonus")?;
	let original = number(c, "suit_nominal_original")?;
	Ok(PlayingCard {
		core: core(c)?,
		suit: named(c, "suit", Suit::from_name)?,
		rank: Rank::from_id(int(c, "rank")?).ok_or_else(|| bad("rank", &c["rank"]))?,
		orig_suit: Suit::ALL
			.into_iter()
			.find(|s| s.nominal_original() == original)
			.ok_or_else(|| bad("suit_nominal_original", &c["suit_nominal_original"]))?,
		enhancement: named(c, "key", Enhancement::from_key)?,
		seal: opt_named(c, "seal", |s| by_name(&Seal::ALL, Seal::name, s))?,
		perma_bonus: number(ab, "perma_bonus")?,
		forced: match (raw_flag(ab, "forced_selection")?, highlighted(c)?) {
			(false, _) => Forced::No,
			(true, None) => Forced::Flagged,
			(true, Some(place)) => Forced::Locked { place },
		},
		played_this_ante: raw_flag(ab, "played_this_ante")?,
		wheel_flipped: raw_flag(ab, "wheel_flipped")?,
		times_played: int(c, "times_played")?,
		shattered: false,
		destroyed: false,
		lucky_trigger: false,
		vampired: false,
	})
}

/// A card that is not a playing card.
fn card(c: &Value) -> Result<Card, DumpError> {
	use crate::json::ValueExt;

	let ab = &c["ability"];
	let kind = named(c, "key", kind_of)?;
	let mut card = Card::new(0, kind);
	card.core = core(c)?;
	card.queue_negative_removal = raw_flag(ab, "queue_negative_removal")?;
	match kind {
		Kind::Joker(j) => {
			for &field in JokerField::COMMON.iter().chain(JokerField::of(j)) {
				set_joker_field(&mut card, field, ab)?;
			}
			if raw_flag(ab, "perishable")? {
				card.ab.sticker = Some(Sticker::Perishable {
					tally: number(ab, "perish_tally")?,
				});
			}
		}
		Kind::Tarot(_) | Kind::Planet(_) | Kind::Spectral(_) => {
			card.ab.extra_value = number(ab, "extra_value")?;
		}
		Kind::Booster(_) => {
			let pos = match &ab["booster_pos"] {
				Value::Null => None,
				v => {
					let x = v.number().ok_or_else(|| bad("booster_pos", v))?;
					let pos = whole::<u8>(ab, "booster_pos", x)?;
					Some(pos.checked_sub(1).ok_or_else(|| bad("booster_pos", v))?)
				}
			};
			card.booster_slot = pos;
		}
		Kind::Voucher(_) => {}
	}
	Ok(card)
}

/// Sets `field` of the joker `c` from its `ability` table (the inverse of
/// what the canonical state reads, see [`JokerField::path`]).
fn set_joker_field(c: &mut Card, field: JokerField, ab: &Value) -> Result<(), DumpError> {
	let path = field.path();
	let (table, key) = match path.strip_prefix("extra.") {
		Some(rest) => (&ab["extra"], rest),
		None => (ab, path),
	};
	let a = &mut c.ab;
	match field {
		JokerField::Mult => a.mult = number(table, key)?,
		JokerField::XMult => a.x_mult = number(table, key)?,
		JokerField::ExtraValue => a.extra_value = number(table, key)?,
		JokerField::Eternal => {
			if raw_flag(table, key)? {
				a.sticker = Some(Sticker::Eternal);
			}
		}
		JokerField::Rental => a.rental = raw_flag(table, key)?,
		JokerField::Couponed => c.core.couponed = raw_flag(table, key)?,
		JokerField::ExtraChips => a.chips = number(table, key)?,
		JokerField::ExtraDollars => a.dollars = number(table, key)?,
		JokerField::Extra | JokerField::ExtraHandSize => a.extra = number(table, key)?,
		JokerField::InvisRounds => a.invis_rounds = number(table, key)?,
		JokerField::YorickDiscards => a.yorick_discards = number(table, key)?,
		JokerField::CainoXmult => a.caino_xmult = number(table, key)?,
		JokerField::LoyaltyRemaining => a.loyalty_remaining = number(table, key)?,
		JokerField::ToDoPokerHand => a.to_do = raw_named(table, key, HandType::from_name)?,
		JokerField::SteelTally => a.steel_tally = number(table, key)?,
		JokerField::StoneTally => a.stone_tally = number(table, key)?,
		JokerField::DriverTally => a.driver_tally = number(table, key)?,
		JokerField::NineTally => a.nine_tally = number(table, key)?,
	}
	Ok(())
}
