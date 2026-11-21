//! A flattened, canonical view of the game state (`key -> value` strings),
//! built identically by the oracle from the real game's state dump so the
//! two can be diffed. Includes every RNG stream's state.

use std::collections::BTreeMap;

use crate::areas::Cards;
use crate::card::{AnyCard, Card, CardCore, Kind, Sticker};
use crate::data::{
	ALL_VOUCHERS, BlindKind, Center, Edition, HandType, Joker, Seal, TagKind, Voucher,
};
use crate::game::{
	Area, Areas, Blind, BlindSelect, BlindSlot, Game, Hand, HandLevels, HandSort, Tag,
};
use crate::packs::PackState;
use crate::playing_card::PlayingCard;
use crate::round::Round;
use crate::run::{History, Run};
use crate::shop::Shop;

/// The canonical state: every compared value by its dotted path, as text.
pub type Canon = BTreeMap<String, String>;

/// A number as the canonical state writes it (both engines' numbers go
/// through here, so they compare as text): Rust's shortest round-trip form,
/// and NaN as `nan`, as Lua writes it.
#[must_use]
pub fn canon_number(v: f64) -> String {
	if v.is_nan() {
		"nan".into()
	} else {
		format!("{v}")
	}
}

/// A card area of the canonical state, and what it shows of its cards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CanonArea {
	Hand,
	Deck,
	Discard,
	Jokers,
	Consumables,
	ShopJokers,
	ShopVouchers,
	ShopBoosters,
	Pack,
}

impl CanonArea {
	/// Every area the canonical state lists.
	pub const ALL: [CanonArea; 9] = [
		CanonArea::Hand,
		CanonArea::Deck,
		CanonArea::Discard,
		CanonArea::Jokers,
		CanonArea::Consumables,
		CanonArea::ShopJokers,
		CanonArea::ShopVouchers,
		CanonArea::ShopBoosters,
		CanonArea::Pack,
	];

	/// The simulator's area.
	fn area(self) -> Area {
		match self {
			CanonArea::Hand => Area::Hand,
			CanonArea::Deck => Area::Deck,
			CanonArea::Discard => Area::Discard,
			CanonArea::Jokers => Area::Jokers,
			CanonArea::Consumables => Area::Consumables,
			CanonArea::ShopJokers => Area::ShopJokers,
			CanonArea::ShopVouchers => Area::ShopVouchers,
			CanonArea::ShopBoosters => Area::ShopBoosters,
			CanonArea::Pack => Area::PackCards,
		}
	}

	/// The area's name in the canonical paths (`hand.0.id`, ...).
	#[must_use]
	pub fn name(self) -> &'static str {
		match self {
			CanonArea::Hand => "hand",
			CanonArea::Deck => "deck",
			CanonArea::Discard => "discard",
			CanonArea::Jokers => "jokers",
			CanonArea::Consumables => "consumables",
			CanonArea::ShopJokers => "shop_jokers",
			CanonArea::ShopVouchers => "shop_vouchers",
			CanonArea::ShopBoosters => "shop_booster",
			CanonArea::Pack => "pack",
		}
	}

	/// Whether its cards show their cost and sell cost (all but the deck's
	/// and the discard pile's).
	#[must_use]
	pub fn shows_cost(self) -> bool {
		!matches!(self, CanonArea::Deck | CanonArea::Discard)
	}

	/// Whether its playing cards show their debuff (all but an open
	/// pack's).
	#[must_use]
	pub fn shows_playing_debuff(self) -> bool {
		self != CanonArea::Pack
	}

	/// Whether its cards show their facing: the hand's playing cards (with
	/// their forced selection) and the jokers.
	#[must_use]
	pub fn shows_facing(self) -> bool {
		matches!(self, CanonArea::Hand | CanonArea::Jokers)
	}
}

/// A field of a joker's `ability` table that the canonical state compares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JokerField {
	Mult,
	XMult,
	ExtraValue,
	Eternal,
	Rental,
	Couponed,
	ExtraChips,
	ExtraDollars,
	/// Seltzer's scalar `extra`.
	Extra,
	ExtraHandSize,
	InvisRounds,
	YorickDiscards,
	CainoXmult,
	LoyaltyRemaining,
	ToDoPokerHand,
	SteelTally,
	StoneTally,
	DriverTally,
	NineTally,
}

impl JokerField {
	/// The fields compared for every joker.
	pub const COMMON: [JokerField; 6] = [
		JokerField::Mult,
		JokerField::XMult,
		JokerField::ExtraValue,
		JokerField::Eternal,
		JokerField::Rental,
		JokerField::Couponed,
	];

	/// The fields compared for `j` beyond the common ones.
	#[must_use]
	pub fn of(j: Joker) -> &'static [JokerField] {
		use Joker as J;
		match j {
			J::Runner | J::IceCream | J::Square | J::Castle | J::Wee => &[JokerField::ExtraChips],
			J::Rocket => &[JokerField::ExtraDollars],
			J::Selzer => &[JokerField::Extra],
			J::TurtleBean => &[JokerField::ExtraHandSize],
			J::Invisible => &[JokerField::InvisRounds],
			J::Yorick => &[JokerField::YorickDiscards],
			J::Caino => &[JokerField::CainoXmult],
			J::LoyaltyCard => &[JokerField::LoyaltyRemaining],
			J::TodoList => &[JokerField::ToDoPokerHand],
			J::SteelJoker => &[JokerField::SteelTally],
			J::Stone => &[JokerField::StoneTally],
			J::DriversLicense => &[JokerField::DriverTally],
			J::Cloud9 => &[JokerField::NineTally],
			_ => &[],
		}
	}

	/// The field's path in the game's `ability` table.
	#[must_use]
	pub fn path(self) -> &'static str {
		match self {
			JokerField::Mult => "mult",
			JokerField::XMult => "x_mult",
			JokerField::ExtraValue => "extra_value",
			JokerField::Eternal => "eternal",
			JokerField::Rental => "rental",
			JokerField::Couponed => "couponed",
			JokerField::ExtraChips => "extra.chips",
			JokerField::ExtraDollars => "extra.dollars",
			JokerField::Extra => "extra",
			JokerField::ExtraHandSize => "extra.h_size",
			JokerField::InvisRounds => "invis_rounds",
			JokerField::YorickDiscards => "yorick_discards",
			JokerField::CainoXmult => "caino_xmult",
			JokerField::LoyaltyRemaining => "loyalty_remaining",
			JokerField::ToDoPokerHand => "to_do_poker_hand",
			JokerField::SteelTally => "steel_tally",
			JokerField::StoneTally => "stone_tally",
			JokerField::DriverTally => "driver_tally",
			JokerField::NineTally => "nine_tally",
		}
	}

	/// Whether the field is a flag, which the game leaves `nil` while it is
	/// false.
	#[must_use]
	pub fn is_flag(self) -> bool {
		matches!(
			self,
			JokerField::Eternal | JokerField::Rental | JokerField::Couponed
		)
	}

	/// The simulator's value of the field for the joker `c`.
	fn value(self, c: &Card) -> String {
		let a = &c.ab;
		match self {
			JokerField::Mult => canon_number(a.mult),
			JokerField::XMult => canon_number(a.x_mult),
			JokerField::ExtraValue => canon_number(a.extra_value),
			JokerField::Eternal => a.eternal().to_string(),
			JokerField::Rental => a.rental.to_string(),
			JokerField::Couponed => c.core.couponed.to_string(),
			JokerField::ExtraChips => canon_number(a.chips),
			JokerField::ExtraDollars => canon_number(a.dollars),
			JokerField::Extra | JokerField::ExtraHandSize => canon_number(a.extra),
			JokerField::InvisRounds => canon_number(a.invis_rounds),
			JokerField::YorickDiscards => canon_number(a.yorick_discards),
			JokerField::CainoXmult => canon_number(a.caino_xmult),
			JokerField::LoyaltyRemaining => canon_number(a.loyalty_remaining),
			JokerField::ToDoPokerHand => a
				.to_do
				.map_or_else(|| "nil".into(), |h| h.name().to_string()),
			JokerField::SteelTally => canon_number(a.steel_tally),
			JokerField::StoneTally => canon_number(a.stone_tally),
			JokerField::DriverTally => canon_number(a.driver_tally),
			JokerField::NineTally => canon_number(a.nine_tally),
		}
	}
}

/// Whether `k` is one of the game's placeholder centers, the faces it shows
/// for a card not unlocked or not discovered (`j_locked`, `v_undiscovered`,
/// ...): Steamodded marks one in `used_jokers` when an unlock alert shows it.
/// They are in no pool.
#[must_use]
pub fn placeholder_center(k: &str) -> bool {
	k.ends_with("_locked") || k.ends_with("_undiscovered")
}

/// The key a card shows in the canonical state: three packs have a second
/// variant that matches the first in name, cost, weight and contents, and
/// both show as the first.
#[must_use]
pub fn normalize_key(k: &str) -> &str {
	match k {
		"p_buffoon_normal_2" => "p_buffoon_normal_1",
		"p_arcana_mega_2" => "p_arcana_mega_1",
		"p_celestial_mega_2" => "p_celestial_mega_1",
		other => other,
	}
}

/// The canonical `facing` of a card.
fn facing(face_down: bool) -> String {
	(if face_down { "back" } else { "front" }).into()
}

/// What every card shows, `key` being its center's.
fn sim_core(out: &mut Canon, p: &str, c: &CardCore, key: &str, area: CanonArea) {
	out.insert(format!("{p}.id"), c.id.to_string());
	out.insert(format!("{p}.key"), normalize_key(key).to_string());
	out.insert(
		format!("{p}.edition"),
		c.edition.map_or("nil", Edition::name).to_string(),
	);
	if area.shows_cost() {
		out.insert(format!("{p}.cost"), canon_number(c.cost));
		out.insert(format!("{p}.sell_cost"), canon_number(c.sell_cost));
	}
}

fn sim_playing_card(out: &mut Canon, p: &str, c: &PlayingCard, area: CanonArea) {
	sim_core(out, p, &c.core, c.key(), area);
	out.insert(format!("{p}.card"), c.card_key());
	out.insert(
		format!("{p}.seal"),
		c.seal.map_or("nil", Seal::name).to_string(),
	);
	out.insert(format!("{p}.perma_bonus"), canon_number(c.perma_bonus));
	if area.shows_playing_debuff() {
		out.insert(format!("{p}.debuff"), c.core.debuff.to_string());
	}
	if area.shows_facing() {
		out.insert(format!("{p}.facing"), facing(c.core.face_down));
		out.insert(format!("{p}.forced"), c.forced_selection.to_string());
	}
	out.insert(
		format!("{p}.played_this_ante"),
		c.played_this_ante.to_string(),
	);
}

fn sim_card(out: &mut Canon, p: &str, c: &Card, area: CanonArea) {
	sim_core(out, p, &c.core, c.key(), area);
	if let Kind::Joker(j) = c.kind {
		out.insert(format!("{p}.debuff"), c.core.debuff.to_string());
		if area.shows_facing() {
			out.insert(format!("{p}.facing"), facing(c.core.face_down));
		}
		for field in JokerField::COMMON.iter().chain(JokerField::of(j)) {
			out.insert(format!("{p}.ab.{}", field.path()), field.value(c));
		}
		if let Some(Sticker::Perishable { tally }) = c.ab.sticker {
			out.insert(format!("{p}.ab.perish_tally"), canon_number(tally));
		}
	}
	if c.kind.is_consumable() {
		out.insert(
			format!("{p}.ab.extra_value"),
			canon_number(c.ab.extra_value),
		);
	}
}

fn sim_any_card(out: &mut Canon, p: &str, c: &AnyCard, area: CanonArea) {
	match c {
		AnyCard::Playing(c) => sim_playing_card(out, p, c, area),
		AnyCard::Card(c) => sim_card(out, p, c, area),
	}
}

/// The cards of `area`.
fn sim_area<C>(
	out: &mut Canon, area: CanonArea, cards: &[C], card: impl Fn(&mut Canon, &str, &C, CanonArea),
) {
	let name = area.name();
	out.insert(format!("{name}.len"), cards.len().to_string());
	for (i, c) in cards.iter().enumerate() {
		card(out, &format!("{name}.{i}"), c, area);
	}
}

/// Canonical view of the simulator's state (refreshes the per-frame derived
/// joker values first, like the game does every frame).
#[must_use]
pub fn from_sim(g: &mut Game) -> Canon {
	g.update_derived();
	let mut o = Canon::new();
	o.insert("state".into(), g.flow.state.name().into());
	game_numbers(&g.areas, &g.run, &g.round, &g.shop, g.pack, &mut o);
	run_flags(
		&g.areas.hand,
		g.run.won,
		&g.history,
		g.round.most_played_hand,
		&g.blind_select,
		&mut o,
	);
	current_blind(&g.blind, &mut o);
	ante_blinds(&g.blind_select, g.shop.voucher, &mut o);
	joker_targets(&g.round, &mut o);
	hand_levels(&g.hands, &mut o);
	tags(&g.tags.held, &mut o);
	used_keys(&g.history, &mut o);
	for area in CanonArea::ALL {
		match g.areas.cards(area.area()) {
			Cards::Playing(cards) => sim_area(&mut o, area, cards, sim_playing_card),
			Cards::Other(cards) => sim_area(&mut o, area, cards, sim_card),
			Cards::Any(cards) => sim_area(&mut o, area, cards, sim_any_card),
		}
	}
	for (name, state) in g.rng.named_streams() {
		o.insert(format!("rng.{name}"), canon_number(state));
	}
	o
}

/// The plain numbers of `G.GAME` and the card areas' limits.
fn game_numbers(
	areas: &Areas, run: &Run, round: &Round, shop: &Shop, pack: PackState, o: &mut Canon,
) {
	let nums: [(&str, f64); 34] = [
		("ante", f64::from(run.ante)),
		("round", f64::from(run.round)),
		("dollars", run.dollars),
		("chips", round.chips),
		("hands_left", round.hands_left),
		("discards_left", round.discards_left),
		("reroll_cost", shop.reroll_cost),
		("free_rerolls", shop.free_rerolls),
		("rr_hands", round.base_hands),
		("rr_discards", round.base_discards),
		("rr_reroll_cost", shop.base_reroll_cost),
		("skips", run.skips),
		("interest_cap", run.interest_cap),
		("interest_amount", run.interest_amount),
		("discount_percent", shop.discount_percent),
		("bankrupt_at", run.bankrupt_at),
		("probabilities", run.probabilities),
		("shop_joker_max", shop.joker_max),
		("tarot_rate", shop.tarot_rate),
		("planet_rate", shop.planet_rate),
		("edition_rate", shop.edition_rate),
		("playing_card_rate", shop.playing_card_rate),
		("spectral_rate", shop.spectral_rate),
		("joker_limit", areas.limits.jokers.limit),
		("consumable_limit", areas.limits.consumables.limit),
		("hand_limit", areas.hand.limit.limit),
		("hands_played", run.hands_played),
		("hands_played_round", round.hands_played),
		("discards_used", round.discards_used),
		("unused_discards", run.unused_discards),
		("ecto_minus", run.ecto_minus),
		("starting_deck_size", run.starting_deck_size),
		("round_dollars", round.cash_out),
		("pack_size", f64::from(pack.size)),
	];
	for (k, v) in nums {
		o.insert(k.into(), canon_number(v));
	}
	o.insert(
		"pack_choices".into(),
		pack.choices
			.map_or_else(|| "nil".into(), |n| canon_number(f64::from(n))),
	);
}

/// Flags and last-used facts of the run, and the hand's sort mode.
fn run_flags(
	hand: &Hand, run_won: bool, history: &History, most_played_hand: HandType,
	blind_select: &BlindSelect, o: &mut Canon,
) {
	o.insert("won".into(), run_won.to_string());
	o.insert(
		"boss_rerolled".into(),
		blind_select.boss_rerolled.to_string(),
	);
	o.insert(
		"gros_michel_extinct".into(),
		history.gros_michel_extinct.to_string(),
	);
	o.insert(
		"last_hand_played".into(),
		history
			.last_hand_played
			.map_or("nil", HandType::name)
			.into(),
	);
	o.insert(
		"last_tarot_planet".into(),
		history.last_tarot_planet.map_or("nil", Kind::key).into(),
	);
	o.insert("most_played".into(), most_played_hand.name().into());
	o.insert(
		"sort_mode".into(),
		(if hand.order == HandSort::Suit {
			"suit desc"
		} else {
			"desc"
		})
		.into(),
	);
}

/// `G.GAME.blind`.
fn current_blind(blind: &Blind, o: &mut Canon) {
	o.insert("blind.chips".into(), canon_number(blind.chips));
	o.insert("blind.dollars".into(), canon_number(blind.dollars));
	o.insert(
		"blind.name".into(),
		blind.kind.map_or("", BlindKind::name).into(),
	);
	o.insert("blind.disabled".into(), blind.disabled.to_string());
}

/// The ante's three blinds, their tags and its voucher.
fn ante_blinds(blind_select: &BlindSelect, shop_voucher: Option<Voucher>, o: &mut Canon) {
	o.insert(
		"blind_on_deck".into(),
		blind_select.on_deck.map_or("nil", BlindSlot::name).into(),
	);
	for slot in BlindSlot::ALL {
		o.insert(
			format!("blind_states.{}", slot.name()),
			blind_select.states[slot].name().into(),
		);
	}
	o.insert(
		format!("blind_choices.{}", BlindSlot::Boss.name()),
		blind_select.boss.key().into(),
	);
	for slot in [BlindSlot::Small, BlindSlot::Big] {
		o.insert(
			format!("blind_tags.{}", slot.name()),
			blind_select
				.skip_tags
				.of(slot)
				.map_or("nil", TagKind::key)
				.into(),
		);
	}
	o.insert(
		"voucher".into(),
		shop_voucher.map_or("nil", Voucher::key).into(),
	);
}

/// The cards the round's jokers name: The Idol, Mail-In Rebate, Ancient
/// Joker and Castle.
fn joker_targets(round: &Round, o: &mut Canon) {
	o.insert(
		"idol.rank".into(),
		round
			.idol_card
			.id
			.map_or_else(|| "nil".into(), |r| canon_number(f64::from(r.id()))),
	);
	o.insert("idol.suit".into(), round.idol_card.suit.name().into());
	o.insert(
		"mail".into(),
		round
			.mail_rank
			.id
			.map_or_else(|| "nil".into(), |r| canon_number(f64::from(r.id()))),
	);
	o.insert("ancient".into(), round.ancient_suit.name().into());
	o.insert("castle".into(), round.castle_suit.name().into());
}

/// `G.GAME.hands`.
fn hand_levels(hands: &HandLevels, o: &mut Canon) {
	for h in HandType::ALL {
		let l = hands[h];
		let n = h.name();
		o.insert(format!("hands.{n}.level"), canon_number(l.level));
		o.insert(format!("hands.{n}.chips"), canon_number(l.chips));
		o.insert(format!("hands.{n}.mult"), canon_number(l.mult));
		o.insert(format!("hands.{n}.played"), canon_number(l.played));
		o.insert(
			format!("hands.{n}.played_this_round"),
			canon_number(l.played_this_round),
		);
		o.insert(format!("hands.{n}.visible"), l.visible.to_string());
	}
}

/// `G.GAME.tags`.
fn tags(tags: &[Tag], o: &mut Canon) {
	for (i, t) in tags.iter().enumerate() {
		o.insert(format!("tags.{i}.key"), t.kind.key().into());
		o.insert(format!("tags.{i}.triggered"), t.triggered.to_string());
		if t.kind == TagKind::Orbital {
			o.insert(
				format!("tags.{i}.orbital"),
				t.orbital_hand.map_or("nil", HandType::name).into(),
			);
		}
	}
}

/// `used_vouchers` and `used_jokers` as sorted key lists.
fn used_keys(history: &History, o: &mut Canon) {
	let mut uv: Vec<&str> = ALL_VOUCHERS
		.iter()
		.filter(|v| history.used_vouchers.contains(**v))
		.map(|v| v.key())
		.collect();
	uv.sort_unstable();
	o.insert("used_vouchers".into(), uv.join(","));
	let mut uj: Vec<&str> = Center::all_cards()
		.filter(|&c| history.used_jokers.contains(c))
		.map(Center::key)
		.collect();
	uj.sort_unstable();
	o.insert("used_jokers".into(), uj.join(","));
}
