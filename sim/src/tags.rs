//! Port of `tag.lua` (`Tag:apply_to_run`) and `add_tag`.

use crate::card::Kind;
use crate::data::{Booster, Edition, HandType, Joker, RARITY_POOL_3, TagKind};
use crate::ev::{Ev, TagFn};
use crate::game::{Area, CardSet, Game, Tag};
use crate::{cards, events, flow, num, packs, pools, shop};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum TagCtx {
	Immediate,
	NewBlindChoice,
	VoucherAdd,
	TagAdd {
		kind: TagKind,
		orbital: Option<HandType>,
	},
	RoundStartBonus,
	StoreJokerCreate {
		area: Area,
	},
	ShopStart,
	StoreJokerModify {
		card: u32,
	},
	ShopFinalPass,
}

/// `config.type` of each tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TagType {
	Eval,
	Immediate,
	NewBlindChoice,
	VoucherAdd,
	TagAdd,
	RoundStartBonus,
	StoreJokerCreate,
	ShopStart,
	StoreJokerModify,
	ShopFinalPass,
}

fn tag_type(k: TagKind) -> TagType {
	use TagKind as T;
	match k {
		T::Uncommon | T::Rare => TagType::StoreJokerCreate,
		T::Negative | T::Foil | T::Holo | T::Polychrome => TagType::StoreJokerModify,
		T::Investment => TagType::Eval,
		T::Voucher => TagType::VoucherAdd,
		T::Boss | T::Standard | T::Charm | T::Meteor | T::Buffoon | T::Ethereal => {
			TagType::NewBlindChoice
		}
		T::Handy | T::Garbage | T::TopUp | T::Skip | T::Orbital | T::Economy => TagType::Immediate,
		T::Coupon => TagType::ShopFinalPass,
		T::Double => TagType::TagAdd,
		T::Juggle => TagType::RoundStartBonus,
		T::DSix => TagType::ShopStart,
	}
}

impl TagCtx {
	fn ty(self) -> TagType {
		match self {
			TagCtx::Immediate => TagType::Immediate,
			TagCtx::NewBlindChoice => TagType::NewBlindChoice,
			TagCtx::VoucherAdd => TagType::VoucherAdd,
			TagCtx::TagAdd { .. } => TagType::TagAdd,
			TagCtx::RoundStartBonus => TagType::RoundStartBonus,
			TagCtx::StoreJokerCreate { .. } => TagType::StoreJokerCreate,
			TagCtx::ShopStart => TagType::ShopStart,
			TagCtx::StoreJokerModify { .. } => TagType::StoreJokerModify,
			TagCtx::ShopFinalPass => TagType::ShopFinalPass,
		}
	}
}

/// `Tag:yep(..., func)`: the callback event, then removal.
pub(crate) fn yep(g: &mut Game, idx: usize, which: TagFn) {
	let id = g.tags[idx].id;
	g.events.push_after(0.4, Ev::Wait);
	g.events.push(Ev::Wait);
	g.events.push(Ev::TagFunc { which });
	g.events.push_after(0.7, Ev::TagRemove { tag_id: id });
}

/// `Tag:nope()`.
fn nope(g: &mut Game, idx: usize) {
	let id = g.tags[idx].id;
	g.events.push_after(0.2, Ev::Wait);
	g.events.push_after(0.1, Ev::Wait);
	g.events.push_after(0.5, Ev::TagRemove { tag_id: id });
}

/// `add_tag(tag)`.
pub(crate) fn add_tag(g: &mut Game, kind: TagKind, orbital: Option<HandType>) {
	for i in 0..g.tags.len() {
		apply_tag(g, i, TagCtx::TagAdd { kind, orbital });
	}
	g.next_tag_id += 1;
	let id = g.next_tag_id;
	g.tags.push(Tag {
		kind,
		triggered: false,
		orbital_hand: orbital,
		id,
	});
}

/// `Tag:apply_to_run({type = 'eval'})` → dollars.
pub(crate) fn apply_tag_eval(g: &mut Game, idx: usize) -> Option<f64> {
	let t = g.tags[idx];
	if t.triggered || tag_type(t.kind) != TagType::Eval {
		return None;
	}
	if t.kind == TagKind::Investment && g.last_blind_boss {
		yep(g, idx, TagFn::Noop);
		g.tags[idx].triggered = true;
		return Some(25.0);
	}
	None
}

/// `Tag:apply_to_run(context)`; returns the Lua truthiness of the result.
pub(crate) fn apply_tag(g: &mut Game, idx: usize, ctx: TagCtx) -> bool {
	apply_tag_card(g, idx, ctx).0
}

/// Like [`apply_tag`], also returning a created shop card (Rare/Uncommon).
#[expect(
	clippy::too_many_lines,
	reason = "one arm per tag context, as in `Tag:apply_to_run`"
)]
pub(crate) fn apply_tag_card(
	g: &mut Game, idx: usize, ctx: TagCtx,
) -> (bool, Option<crate::card::Card>) {
	use TagKind as T;

	let t = g.tags[idx];
	if t.triggered || tag_type(t.kind) != ctx.ty() {
		return (false, None);
	}
	match ctx {
		TagCtx::Immediate => match t.kind {
			T::TopUp => {
				yep(g, idx, TagFn::TopUp);
				g.tags[idx].triggered = true;
				(true, None)
			}
			T::Skip => {
				yep(g, idx, TagFn::Noop);
				g.events.push(Ev::EaseDollars(g.skips * 5.0));
				g.tags[idx].triggered = true;
				(true, None)
			}
			T::Garbage => {
				yep(g, idx, TagFn::Noop);
				g.events.push(Ev::EaseDollars(g.unused_discards * 1.0));
				g.tags[idx].triggered = true;
				(true, None)
			}
			T::Handy => {
				yep(g, idx, TagFn::Noop);
				g.events.push(Ev::EaseDollars(g.hands_played * 1.0));
				g.tags[idx].triggered = true;
				(true, None)
			}
			T::Economy => {
				yep(g, idx, TagFn::Noop);
				g.events.push(Ev::TagFunc {
					which: TagFn::Economy,
				});
				g.tags[idx].triggered = true;
				(true, None)
			}
			T::Orbital => {
				if let Some(h) = t.orbital_hand {
					g.events.hand_text(0.3);
					events::level_up_hand_ev(g, h, 3.0, false);
					g.events.hand_text(0.0);
				}
				yep(g, idx, TagFn::Noop);
				g.tags[idx].triggered = true;
				(true, None)
			}
			_ => (false, None),
		},
		TagCtx::NewBlindChoice => {
			let which = match t.kind {
				T::Charm => TagFn::Charm,
				T::Meteor => TagFn::Meteor,
				T::Ethereal => TagFn::Ethereal,
				T::Standard => TagFn::Standard,
				T::Buffoon => TagFn::Buffoon,
				T::Boss => TagFn::Boss,
				_ => return (false, None),
			};
			yep(g, idx, which);
			g.tags[idx].triggered = true;
			(true, None)
		}
		TagCtx::VoucherAdd => {
			if t.kind == T::Voucher {
				yep(g, idx, TagFn::Voucher);
				g.tags[idx].triggered = true;
			}
			(false, None)
		}
		TagCtx::TagAdd { kind, orbital } => {
			if t.kind == T::Double && kind != T::Double {
				yep(g, idx, TagFn::Double { kind, orbital });
				g.tags[idx].triggered = true;
			}
			(false, None)
		}
		TagCtx::RoundStartBonus => {
			if t.kind == T::Juggle {
				yep(g, idx, TagFn::Noop);
				cards::change_hand_size(g, 3.0);
				g.temp_handsize = Some(g.temp_handsize.unwrap_or(0.0) + 3.0);
				g.tags[idx].triggered = true;
				return (true, None);
			}
			(false, None)
		}
		TagCtx::StoreJokerCreate { area } => {
			let mut card = None;
			if t.kind == T::Rare {
				let mut seen: Vec<Joker> = Vec::new();
				for c in &g.jokers {
					if let Some(j) = c.joker()
						&& j.rarity() == 3
						&& !seen.contains(&j)
					{
						seen.push(j);
					}
				}
				if RARITY_POOL_3.len() > seen.len() {
					let c = pools::create_card(
						g,
						CardSet::Joker,
						area,
						false,
						Some(1.0),
						false,
						None,
						"rta",
					);
					yep(g, idx, TagFn::RareCard { card: c.id });
					card = Some(c);
				} else {
					nope(g, idx);
				}
			} else if t.kind == T::Uncommon {
				let c = pools::create_card(
					g,
					CardSet::Joker,
					area,
					false,
					Some(0.9),
					false,
					None,
					"uta",
				);
				yep(g, idx, TagFn::UncommonCard { card: c.id });
				card = Some(c);
			}
			g.tags[idx].triggered = true;
			(card.is_some(), card)
		}
		TagCtx::ShopStart => {
			if t.kind == T::DSix && !g.shop_d6ed {
				g.shop_d6ed = true;
				yep(g, idx, TagFn::D6);
				g.tags[idx].triggered = true;
				return (true, None);
			}
			(false, None)
		}
		TagCtx::StoreJokerModify { card } => {
			let Some(c) = shop::find_shop_card(g, card) else {
				return (false, None);
			};
			let mut applied = false;
			if c.edition == Edition::None && !c.temp_edition && matches!(c.kind, Kind::Joker(_)) {
				let ed = match t.kind {
					T::Foil => Some(Edition::Foil),
					T::Holo => Some(Edition::Holo),
					T::Polychrome => Some(Edition::Polychrome),
					T::Negative => Some(Edition::Negative),
					_ => None,
				};
				if let Some(e) = ed {
					shop::with_shop_card(g, card, |c| c.temp_edition = true);
					yep(g, idx, TagFn::Edition { card, edition: e });
					applied = true;
				}
				g.tags[idx].triggered = true;
			}
			(applied, None)
		}
		TagCtx::ShopFinalPass => {
			if t.kind == T::Coupon && g.shop_open && !g.shop_free {
				g.shop_free = true;
				yep(g, idx, TagFn::Coupon);
				g.tags[idx].triggered = true;
				return (true, None);
			}
			(false, None)
		}
	}
}

pub(crate) fn run_tag_event(g: &mut Game, e: Ev) {
	match e {
		Ev::TagRemove { tag_id } => {
			if let Some(i) = g.tags.iter().position(|t| t.id == tag_id) {
				g.tags.remove(i);
			}
		}
		Ev::TagFunc { which } => match which {
			TagFn::Noop => {}
			TagFn::TopUp => {
				for _ in 0..2 {
					if num::float(g.jokers.len()) < g.joker_limit.limit {
						let mut c = pools::create_card(
							g,
							CardSet::Joker,
							Area::Jokers,
							false,
							Some(0.0),
							false,
							None,
							"top",
						);
						cards::add_to_deck(g, &mut c, false);
						g.jokers.push(c);
					}
				}
			}
			TagFn::Economy => {
				let d = 40f64.min(g.dollars.max(0.0));
				cards::ease_dollars(g, d);
			}
			TagFn::Charm => packs::open_tag_pack(g, Booster::ArcanaMega1),
			TagFn::Meteor => packs::open_tag_pack(g, Booster::CelestialMega1),
			TagFn::Ethereal => packs::open_tag_pack(g, Booster::SpectralNormal1),
			TagFn::Standard => packs::open_tag_pack(g, Booster::StandardMega1),
			TagFn::Buffoon => packs::open_tag_pack(g, Booster::BuffoonMega1),
			TagFn::Boss => flow::do_reroll_boss(g, true),
			TagFn::Voucher => {
				let v = pools::get_next_voucher(g, true);
				let c = cards::make_card(g, Kind::Voucher(v));
				g.events.materialize(1.0);
				g.shop_vouchers.push(c);
			}
			TagFn::Double { kind, orbital } => add_tag(g, kind, orbital),
			TagFn::RareCard { card } | TagFn::UncommonCard { card } => {
				g.events.materialize(1.0);
				shop::with_shop_card(g, card, |c| c.ab.couponed = true);
				shop::recost_shop_card(g, card);
			}
			TagFn::Edition { card, edition } => {
				// Bought before the tag resolved: the edition still lands.
				let at = shop::locate_shop_card(g, card).or_else(|| {
					g.jokers
						.iter()
						.position(|c| c.id == card)
						.map(|i| (Area::Jokers, i))
				});
				if let Some((area, i)) = at {
					cards::set_edition_at(g, area, i, edition);
					g.events.edition_fx(edition, true);
					let c = &mut g.area_mut(area)[i];
					c.ab.couponed = true;
					c.temp_edition = false;
					cards::reprice(g, area, i);
				}
			}
			TagFn::D6 => {
				g.temp_reroll_cost = Some(0.0);
				cards::calculate_reroll_cost(g, true);
			}
			TagFn::Coupon => {
				if g.shop_open {
					for area in [Area::ShopJokers, Area::ShopBoosters] {
						for i in 0..g.area(area).len() {
							g.area_mut(area)[i].ab.couponed = true;
							cards::reprice(g, area, i);
						}
					}
				}
			}
		},
		_ => unreachable!(),
	}
}
