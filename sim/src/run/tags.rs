//! Port of `tag.lua` (`Tag:apply_to_run`) and `add_tag`.

use super::setup::Run;
use crate::card::{Card, CardLike, Kind, Pricing};
use crate::data::{Booster, Edition, HandType, Joker, RARITY_POOL_3, Rarity, TagKind};
use crate::event::{EditionTiming, Event, EventQueue};
use crate::game::{Area, Areas, Game, HandLevels, PerBlind};
use crate::pools::{NewCard, OwnedArea, RarityRoll, VoucherRoll};
use crate::round::Round;
use crate::shop::Shop;
use crate::{card, event, flow, pools, shop};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tag {
	pub kind: TagKind,
	pub triggered: bool,
	pub orbital_hand: Option<HandType>,
	pub id: u32,
}

/// The Orbital Tag hands rolled for an ante's three blinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OrbitalChoices {
	pub ante: i32,
	pub hands: PerBlind<Option<HandType>>,
}

/// The tags the player holds (`G.GAME.tags`) and the Orbital Tag's hands.
#[derive(Debug, Default, PartialEq)]
pub struct Tags {
	pub held: Vec<Tag>,
	pub next_id: u32,
	pub orbital_choices: Vec<OrbitalChoices>,
}

impl Clone for Tags {
	fn clone(&self) -> Tags {
		Tags {
			held: self.held.clone(),
			next_id: self.next_id,
			orbital_choices: self.orbital_choices.clone(),
		}
	}

	fn clone_from(&mut self, source: &Tags) {
		let Tags {
			held,
			next_id,
			orbital_choices,
		} = self;
		held.clone_from(&source.held);
		*next_id = source.next_id;
		orbital_choices.clone_from(&source.orbital_choices);
	}
}

/// The events of tags: `add_tag`, and a triggered tag's callback and
/// removal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum TagEvent {
	Add {
		kind: TagKind,
		orbital: Option<HandType>,
	},
	Func {
		which: TagFn,
	},
	Remove {
		tag_id: u32,
	},
}

/// The body of a tag's `yep(..., func)` callback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TagFn {
	TopUp,
	Charm,
	Meteor,
	Ethereal,
	Standard,
	Buffoon,
	Boss,
	Voucher,
	Double {
		kind: TagKind,
		orbital: Option<HandType>,
	},
	/// The free joker of a Rare or Uncommon Tag.
	FreeCard {
		card: u32,
	},
	Edition {
		card: u32,
		edition: Edition,
	},
	D6,
	Coupon,
	Economy,
	Noop,
}

/// What a tag does (its `config.type`), with what tells apart the kinds that
/// share a type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TagEffect {
	Eval,
	Immediate(Immediate),
	/// Opens a pack or rerolls the boss: the callback.
	NewBlindChoice(TagFn),
	VoucherAdd,
	TagAdd,
	RoundStartBonus,
	StoreJokerCreate(FreeJoker),
	ShopStart,
	StoreJokerModify(Edition),
	ShopFinalPass,
}

/// The `immediate` tags.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Immediate {
	TopUp,
	Skip,
	Garbage,
	Handy,
	Economy,
	Orbital,
}

/// The free joker a Rare or Uncommon Tag puts in the shop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FreeJoker {
	Rare,
	Uncommon,
}

fn effect(kind: TagKind) -> TagEffect {
	use TagKind as T;
	match kind {
		T::Uncommon => TagEffect::StoreJokerCreate(FreeJoker::Uncommon),
		T::Rare => TagEffect::StoreJokerCreate(FreeJoker::Rare),
		T::Negative => TagEffect::StoreJokerModify(Edition::Negative),
		T::Foil => TagEffect::StoreJokerModify(Edition::Foil),
		T::Holo => TagEffect::StoreJokerModify(Edition::Holo),
		T::Polychrome => TagEffect::StoreJokerModify(Edition::Polychrome),
		T::Investment => TagEffect::Eval,
		T::Voucher => TagEffect::VoucherAdd,
		T::Boss => TagEffect::NewBlindChoice(TagFn::Boss),
		T::Standard => TagEffect::NewBlindChoice(TagFn::Standard),
		T::Charm => TagEffect::NewBlindChoice(TagFn::Charm),
		T::Meteor => TagEffect::NewBlindChoice(TagFn::Meteor),
		T::Buffoon => TagEffect::NewBlindChoice(TagFn::Buffoon),
		T::Ethereal => TagEffect::NewBlindChoice(TagFn::Ethereal),
		T::Handy => TagEffect::Immediate(Immediate::Handy),
		T::Garbage => TagEffect::Immediate(Immediate::Garbage),
		T::TopUp => TagEffect::Immediate(Immediate::TopUp),
		T::Skip => TagEffect::Immediate(Immediate::Skip),
		T::Orbital => TagEffect::Immediate(Immediate::Orbital),
		T::Economy => TagEffect::Immediate(Immediate::Economy),
		T::Coupon => TagEffect::ShopFinalPass,
		T::Double => TagEffect::TagAdd,
		T::Juggle => TagEffect::RoundStartBonus,
		T::DSix => TagEffect::ShopStart,
	}
}

/// The tags of `held` that have not triggered, each with what it does, in
/// order (`Tag:apply_to_run` does nothing for a triggered tag).
fn pending(held: &mut [Tag]) -> impl Iterator<Item = (&mut Tag, TagEffect)> {
	held.iter_mut()
		.filter_map(|t| pending_effect(*t).map(|effect| (t, effect)))
}

/// What `tag` does, unless it has triggered.
fn pending_effect(tag: Tag) -> Option<TagEffect> {
	(!tag.triggered).then(|| effect(tag.kind))
}

/// `Tag:yep(..., func)`: `tag` triggers; its callback event, then its
/// removal.
fn yep(tag: &mut Tag, events: &mut EventQueue, which: TagFn) {
	tag.triggered = true;
	events.delay(0.4);
	events.push(Event::Wait);
	events.push(Event::Tag(TagEvent::Func { which }));
	events.push_after(0.7, Event::Tag(TagEvent::Remove { tag_id: tag.id }));
}

/// `Tag:nope()`: `tag` triggers for nothing and is removed.
fn nope(tag: &mut Tag, events: &mut EventQueue) {
	tag.triggered = true;
	events.delay(0.2);
	events.delay(0.1);
	events.push_after(0.5, Event::Tag(TagEvent::Remove { tag_id: tag.id }));
}

/// `add_tag(tag)`: the held Double Tags copy it (`tag_add`), then it is held.
pub(crate) fn add_tag(
	tags: &mut Tags, events: &mut EventQueue, kind: TagKind, orbital: Option<HandType>,
) {
	if kind != TagKind::Double {
		for (t, effect) in pending(&mut tags.held) {
			if effect == TagEffect::TagAdd {
				yep(t, events, TagFn::Double { kind, orbital });
			}
		}
	}
	tags.next_id += 1;
	tags.held.push(Tag {
		kind,
		triggered: false,
		orbital_hand: orbital,
		id: tags.next_id,
	});
}

/// `Tag:apply_to_run({type = 'eval'})` for every held tag: the dollars
/// Investment Tags pay after a boss blind.
pub(crate) fn apply_eval(round: &Round, tags: &mut Tags, events: &mut EventQueue) -> f64 {
	let mut dollars = 0.0;
	for (t, effect) in pending(&mut tags.held) {
		if effect == TagEffect::Eval && round.last_blind_boss {
			yep(t, events, TagFn::Noop);
			dollars += 25.0;
		}
	}
	dollars
}

/// `Tag:apply_to_run({type = 'immediate'})` for every held tag: money,
/// Top-up's jokers and Orbital's levels.
pub(crate) fn apply_immediate(
	run: &Run, hands: &mut HandLevels, tags: &mut Tags, events: &mut EventQueue,
) {
	for (t, effect) in pending(&mut tags.held) {
		let TagEffect::Immediate(immediate) = effect else {
			continue;
		};
		match immediate {
			Immediate::TopUp => yep(t, events, TagFn::TopUp),
			Immediate::Skip => {
				yep(t, events, TagFn::Noop);
				events.push(Event::EaseDollars(run.skips * 5.0));
			}
			Immediate::Garbage => {
				yep(t, events, TagFn::Noop);
				events.push(Event::EaseDollars(run.unused_discards * 1.0));
			}
			Immediate::Handy => {
				yep(t, events, TagFn::Noop);
				events.push(Event::EaseDollars(run.hands_played * 1.0));
			}
			Immediate::Economy => {
				yep(t, events, TagFn::Noop);
				events.push(Event::Tag(TagEvent::Func {
					which: TagFn::Economy,
				}));
			}
			Immediate::Orbital => {
				if let Some(h) = t.orbital_hand {
					events.hand_text(0.3);
					event::level_up_hand(hands, events, h, 3.0);
					events.hand_text(0.0);
				}
				yep(t, events, TagFn::Noop);
			}
		}
	}
}

/// `Tag:apply_to_run({type = 'new_blind_choice'})` up to the first tag that
/// triggers: it opens its pack or rerolls the boss.
pub(crate) fn apply_new_blind_choice(tags: &mut Tags, events: &mut EventQueue) {
	let first = pending(&mut tags.held).find_map(|(t, effect)| match effect {
		TagEffect::NewBlindChoice(which) => Some((t, which)),
		_ => None,
	});
	if let Some((t, which)) = first {
		yep(t, events, which);
	}
}

/// `Tag:apply_to_run({type = 'voucher_add'})` for every held tag: Voucher
/// Tag adds a voucher to the shop.
pub(crate) fn apply_voucher_add(tags: &mut Tags, events: &mut EventQueue) {
	for (t, effect) in pending(&mut tags.held) {
		if effect == TagEffect::VoucherAdd {
			yep(t, events, TagFn::Voucher);
		}
	}
}

/// `Tag:apply_to_run({type = 'round_start_bonus'})` for every held tag:
/// Juggle Tag's hand size for the round.
pub(crate) fn apply_round_start_bonus(round: &mut Round, tags: &mut Tags, events: &mut EventQueue) {
	for (t, effect) in pending(&mut tags.held) {
		if effect == TagEffect::RoundStartBonus {
			yep(t, events, TagFn::Noop);
			flow::change_hand_size(events, 3.0);
			round.temp_handsize = Some(round.temp_handsize.unwrap_or(0.0) + 3.0);
		}
	}
}

/// `Tag:apply_to_run({type = 'shop_start'})` for every held tag: D6 Tag's
/// free rerolls.
pub(crate) fn apply_shop_start(shop: &mut Shop, tags: &mut Tags, events: &mut EventQueue) {
	for (t, effect) in pending(&mut tags.held) {
		if effect == TagEffect::ShopStart && !shop.d6ed {
			shop.d6ed = true;
			yep(t, events, TagFn::D6);
		}
	}
}

/// `Tag:apply_to_run({type = 'shop_final_pass'})` for every held tag:
/// Coupon Tag makes the shop free.
pub(crate) fn apply_shop_final_pass(shop: &mut Shop, tags: &mut Tags, events: &mut EventQueue) {
	for (t, effect) in pending(&mut tags.held) {
		if effect == TagEffect::ShopFinalPass && shop.open && !shop.free {
			shop.free = true;
			yep(t, events, TagFn::Coupon);
		}
	}
}

/// `Tag:apply_to_run({type = 'store_joker_create'})` up to the first tag
/// that gives a card: a Rare or Uncommon Tag's free joker for the shop.
pub(crate) fn create_store_joker(g: &mut Game) -> Option<Card> {
	// The tag is found first and triggered after `create_card`, which needs
	// the whole game: its index is all that is kept in between.
	let mut giver = None;
	for (i, t) in g.tags.held.iter_mut().enumerate() {
		let Some(TagEffect::StoreJokerCreate(free)) = pending_effect(*t) else {
			continue;
		};
		if free == FreeJoker::Rare && !rare_joker_left(&g.areas.jokers) {
			nope(t, &mut g.events);
		} else {
			giver = Some((i, free));
			break;
		}
	}
	let (i, free) = giver?;
	let (rarity, append) = match free {
		FreeJoker::Rare => (1.0, "rta"),
		FreeJoker::Uncommon => (0.9, "uta"),
	};
	let c = pools::create_card(
		g,
		NewCard::joker(RarityRoll::Fixed(rarity), Area::ShopJokers, append),
	);
	yep(&mut g.tags.held[i], &mut g.events, TagFn::FreeCard {
		card: c.core.id,
	});
	Some(c)
}

/// Whether a rare joker is left that `jokers` do not hold (Rare Tag counts
/// the rares in possession).
fn rare_joker_left(jokers: &[Card]) -> bool {
	let mut seen: Vec<Joker> = Vec::new();
	for c in jokers {
		if let Some(j) = c.joker()
			&& j.rarity() == Rarity::Rare
			&& !seen.contains(&j)
		{
			seen.push(j);
		}
	}
	RARITY_POOL_3.len() > seen.len()
}

/// `Tag:apply_to_run({type = 'store_joker_modify', card = c})` up to the
/// first tag that triggers: an edition tag gives the shop card `c` its
/// edition if it is a joker without one.
pub(crate) fn modify_store_joker(tags: &mut Tags, events: &mut EventQueue, c: &mut Card) {
	if c.core.edition.is_some() || c.temp_edition || !matches!(c.kind, Kind::Joker(_)) {
		return;
	}
	let first = pending(&mut tags.held).find_map(|(t, effect)| match effect {
		TagEffect::StoreJokerModify(edition) => Some((t, edition)),
		_ => None,
	});
	if let Some((t, edition)) = first {
		c.temp_edition = true;
		yep(t, events, TagFn::Edition {
			card: c.core.id,
			edition,
		});
	}
}

pub(crate) fn run_event(g: &mut Game, e: TagEvent) {
	match e {
		TagEvent::Add { kind, orbital } => add_tag(&mut g.tags, &mut g.events, kind, orbital),
		TagEvent::Remove { tag_id } => {
			if let Some(i) = g.tags.held.iter().position(|t| t.id == tag_id) {
				g.tags.held.remove(i);
			}
		}
		TagEvent::Func { which } => match which {
			TagFn::Noop => {}
			TagFn::TopUp => {
				for _ in 0..2 {
					if g.areas.joker_room() {
						pools::create_owned(
							g,
							NewCard::joker(RarityRoll::Fixed(0.0), OwnedArea::Jokers, "top"),
						);
					}
				}
			}
			TagFn::Economy => {
				let d = 40f64.min(g.run.dollars.max(0.0));
				flow::ease_dollars(&mut g.run, d);
			}
			TagFn::Charm => shop::open_tag_pack(g, Booster::ArcanaMega1),
			TagFn::Meteor => shop::open_tag_pack(g, Booster::CelestialMega1),
			TagFn::Ethereal => shop::open_tag_pack(g, Booster::SpectralNormal1),
			TagFn::Standard => shop::open_tag_pack(g, Booster::StandardMega1),
			TagFn::Buffoon => shop::open_tag_pack(g, Booster::BuffoonMega1),
			TagFn::Boss => flow::do_reroll_boss(&mut g.blind_select, &mut g.events),
			TagFn::Voucher => {
				let v = pools::get_next_voucher(
					&g.areas,
					&g.run,
					&g.history,
					&mut g.rng,
					VoucherRoll::Tag,
				);
				let c = shop::make_shop_card(g, Kind::Voucher(v));
				g.areas.shop_vouchers.push(c);
			}
			TagFn::Double { kind, orbital } => add_tag(&mut g.tags, &mut g.events, kind, orbital),
			TagFn::FreeCard { card } => {
				g.events.materialize(1.0);
				if let Some((area, i)) = locate_tag_card(&g.areas, card) {
					let pricing = Pricing::new(g.shop.discount_percent, &g.areas.jokers);
					make_free(pricing, g.areas.card_at_mut(area, i), area);
				}
			}
			TagFn::Edition { card, edition } => {
				if let Some((area, i)) = locate_tag_card(&g.areas, card) {
					let pricing = Pricing::new(g.shop.discount_percent, &g.areas.jokers);
					let (cards, limits) = g.areas.cards_and_limits_mut(area);
					let c = cards.at_mut(i);
					card::set_edition(c, Some(edition), Some(area), pricing, limits);
					g.events.edition_fx(Some(edition), EditionTiming::Immediate);
					let c = g.areas.card_at_mut(area, i);
					if let Some(c) = c.as_card_mut() {
						c.temp_edition = false;
					}
					make_free(pricing, c, area);
				}
			}
			TagFn::D6 => {
				g.shop.temp_reroll_cost = Some(0.0);
				shop::calculate_reroll_cost(&mut g.shop);
			}
			TagFn::Coupon => {
				if g.shop.open {
					let pricing = Pricing::new(g.shop.discount_percent, &g.areas.jokers);
					for c in &mut g.areas.shop_jokers {
						make_free(pricing, c, Area::ShopJokers);
					}
					for c in &mut g.areas.shop_boosters {
						make_free(pricing, c, Area::ShopBoosters);
					}
				}
			}
		},
	}
}

/// Where the card `id` a tag's callback changes is: in the shop, or among
/// the jokers if it was bought first (the game's callback holds the card
/// itself, wherever it went).
fn locate_tag_card(areas: &Areas, id: u32) -> Option<(Area, usize)> {
	shop::locate_shop_card(areas, id)
		.or_else(|| Some((Area::Jokers, areas.position(Area::Jokers, id)?)))
}

/// A tag makes the card `c` of `area` free: `couponed`, then `set_cost`.
fn make_free(pricing: Pricing, c: &mut (impl CardLike + ?Sized), area: Area) {
	c.core_mut().couponed = true;
	pricing.set_cost(c, Some(area));
}
