//! Shop and `G.FUNCS.use_card` (`button_callbacks.lua`, `UI_definitions.lua`
//! `create_card_for_shop`, card.lua `redeem`). Booster packs are in `packs`.

use crate::card::{Card, Kind};
use crate::consumables::{CEv, Rate};
use crate::data::{BlindKind, Edition, Enhancement, TagKind, Tarot, Voucher};
use crate::ev::{Ev, Pile};
use crate::events::St;
use crate::game::{Area, CardSet, Game, PackSlot};
use crate::jokers::Ctx;
use crate::rng::Key;
use crate::tags::TagCtx;
use crate::{blind, cards, consumables, flow, jokers, num, packs, pools, tags};

pub(crate) fn locate_shop_card(g: &Game, id: u32) -> Option<(Area, usize)> {
	if let Some(i) = g.shop_jokers.iter().position(|c| c.id == id) {
		return Some((Area::ShopJokers, i));
	}
	if let Some(i) = g.shop_boosters.iter().position(|c| c.id == id) {
		return Some((Area::ShopBoosters, i));
	}
	if let Some(i) = g.shop_vouchers.iter().position(|c| c.id == id) {
		return Some((Area::ShopVouchers, i));
	}
	None
}

pub(crate) fn find_shop_card(g: &Game, id: u32) -> Option<&Card> {
	let (a, i) = locate_shop_card(g, id)?;
	Some(&g.area(a)[i])
}

pub(crate) fn with_shop_card(g: &mut Game, id: u32, f: impl FnOnce(&mut Card)) {
	if let Some((a, i)) = locate_shop_card(g, id) {
		f(&mut g.area_mut(a)[i]);
	}
}

pub(crate) fn recost_shop_card(g: &mut Game, id: u32) {
	if let Some((a, i)) = locate_shop_card(g, id) {
		cards::reprice(g, a, i);
	}
}

// -----------------------------------------------------------------
// Shop
// -----------------------------------------------------------------

pub(crate) fn update_shop(g: &mut Game) {
	if !g.state_complete {
		let exists = g.shop_open;
		if !exists {
			g.shop_open = true;
			g.shop_jokers.clear();
			g.shop_vouchers.clear();
			g.shop_boosters.clear();
		}
		if !exists {
			// G.UIDEF.shop()'s shop-sign event
			g.events.push(Ev::Wait);
		}
		g.events.push(Ev::ShopOuter);
		g.state_complete = true;
		g.shop_exists_at_open = exists;
	}
}

fn shop_populate(g: &mut Game, shop_exists: bool) {
	for i in 0..g.tags.len() {
		tags::apply_tag(g, i, TagCtx::ShopStart);
	}
	if !shop_exists {
		fill_shop_jokers(g);
		if let Some(v) = g.current_voucher {
			g.voucher_seen = true;
			let mut c = cards::make_card(g, Kind::Voucher(v));
			c.shop_voucher = true;
			g.events.materialize(1.0);
			// A 'shop'-type area: insertion order is kept (no sort).
			g.shop_vouchers.push(c);
		}
		for i in 0..2 {
			if g.used_packs[i] == PackSlot::Unrolled {
				let b = pools::get_pack(g, "shop_pack", None);
				g.used_packs[i] = PackSlot::Offered(b);
			}
			if let PackSlot::Offered(b) = g.used_packs[i] {
				let mut c = cards::make_card(g, Kind::Booster(b));
				c.booster_pos = num::byte(i) + 1;
				g.events.materialize(1.0);
				g.shop_boosters.push(c);
			}
		}
		for i in 0..g.tags.len() {
			tags::apply_tag(g, i, TagCtx::VoucherAdd);
		}
		for i in 0..g.tags.len() {
			tags::apply_tag(g, i, TagCtx::ShopFinalPass);
		}
	}
	crate::game::trace!(g, "shop_populate");
	// save_run
	g.events.push(Ev::Wait);
}

/// Fills the joker row up to `shop_joker_max` (the shop's
/// `for i = 1, shop_joker_max - #G.shop_jokers.cards` loops).
fn fill_shop_jokers(g: &mut Game) {
	let n = num::trunc(g.shop_joker_max).saturating_sub(g.shop_jokers.len());
	for _ in 0..n {
		let c = create_card_for_shop(g);
		g.shop_jokers.push(c);
	}
}

/// `create_card_for_shop(G.shop_jokers)`.
pub(crate) fn create_card_for_shop(g: &mut Game) -> Card {
	for i in 0..g.tags.len() {
		let (_, card) = tags::apply_tag_card(g, i, TagCtx::StoreJokerCreate {
			area: Area::ShopJokers,
		});
		if let Some(mut c) = card {
			for key in 0..g.tags.len() {
				if apply_tag_modify_card(g, key, &mut c) {
					break;
				}
			}
			return c;
		}
	}
	let total = g.joker_rate + g.tarot_rate + g.planet_rate + g.playing_card_rate + g.spectral_rate;
	let key = g.key_ante("cdt");
	let polled = g.rng.random(&key) * total;
	let illusion = g.voucher_used(Voucher::Illusion) && {
		let key = Key::from("illusion");
		g.rng.random(&key) > 0.6
	};
	let opts = [
		(CardSet::Joker, g.joker_rate),
		(CardSet::Tarot, g.tarot_rate),
		(CardSet::Planet, g.planet_rate),
		(
			if illusion {
				CardSet::Enhanced
			} else {
				CardSet::Base
			},
			g.playing_card_rate,
		),
		(CardSet::Spectral, g.spectral_rate),
	];
	let mut check = 0.0;
	for (set, val) in opts {
		if polled > check && polled <= check + val {
			let mut c =
				pools::create_card(g, set, Area::ShopJokers, false, None, false, None, "sho");
			let id = c.id;
			g.events.push(Ev::StoreModify { card: id });
			if matches!(set, CardSet::Base | CardSet::Enhanced) && g.voucher_used(Voucher::Illusion)
			{
				let key = Key::from("illusion");
				if g.rng.random(&key) > 0.8 {
					let p = g.rng.random(&key);
					let e = if p > 1.0 - 0.15 {
						Edition::Polychrome
					} else if p > 0.5 {
						Edition::Holo
					} else {
						Edition::Foil
					};
					cards::set_edition(g, &mut c, e, None);
					g.events.edition_fx(e, false);
				}
			}
			return c;
		}
		check += val;
	}
	unreachable!("create_card_for_shop: no card type selected (polled {polled})")
}

/// `store_joker_modify` on a card that is not yet in an area.
fn apply_tag_modify_card(g: &mut Game, idx: usize, c: &mut Card) -> bool {
	let t = g.tags[idx];
	let is_mod = matches!(
		t.kind,
		TagKind::Foil | TagKind::Holo | TagKind::Polychrome | TagKind::Negative
	);
	if t.triggered || !is_mod {
		return false;
	}
	if c.edition == Edition::None && !c.temp_edition && matches!(c.kind, Kind::Joker(_)) {
		let e = match t.kind {
			TagKind::Foil => Edition::Foil,
			TagKind::Holo => Edition::Holo,
			TagKind::Polychrome => Edition::Polychrome,
			_ => Edition::Negative,
		};
		c.temp_edition = true;
		tags::yep(g, idx, crate::ev::TagFn::Edition {
			card: c.id,
			edition: e,
		});
		g.tags[idx].triggered = true;
		return true;
	}
	false
}

/// `change_shop_size(mod)`.
pub(crate) fn change_shop_size(g: &mut Game, m: f64) {
	g.shop_joker_max += m;
	if g.shop_open {
		if m < 0.0 {
			while num::float(g.shop_jokers.len()) > g.shop_joker_max {
				let Some(c) = g.shop_jokers.pop() else {
					break;
				};
				cards::drop_unowned(g, c);
			}
		}
		if m > 0.0 {
			fill_shop_jokers(g);
		}
	}
}

pub(crate) fn do_reroll(g: &mut Game) {
	if g.reroll_cost > 0.0 {
		g.events.push(Ev::EaseDollars(-g.reroll_cost));
	}
	g.events.push(Ev::RerollShopRun);
	g.events.push_after(0.3, Ev::RerollOuter);
	// save_run
	g.events.push(Ev::Wait);
}

/// `G.FUNCS.check_for_buy_space(card)`.
#[must_use]
pub(crate) fn has_buy_space(g: &Game, c: &Card) -> bool {
	let neg = if c.edition == Edition::Negative {
		1.0
	} else {
		0.0
	};
	match c.kind {
		Kind::Voucher(_) | Kind::Playing => true,
		Kind::Joker(_) => num::float(g.jokers.len()) < g.joker_limit.limit + neg,
		k if k.is_consumable() => num::float(g.consumables.len()) < g.consumable_limit.limit + neg,
		_ => false,
	}
}

pub(crate) fn do_buy(g: &mut Game, idx: usize, buy_and_use: bool, highlighted: &[usize]) {
	let c = &g.shop_jokers[idx];
	if !buy_and_use && !has_buy_space(g, c) {
		return;
	}
	let id = c.id;
	g.pending_highlight = highlighted.to_vec();
	g.events.push_after(0.1, Ev::BuyRun {
		card: id,
		buy_and_use,
	});
}

fn buy_run(g: &mut Game, id: u32, buy_and_use: bool) {
	let Some(i) = g.shop_jokers.iter().position(|c| c.id == id) else {
		return;
	};
	let mut c = g.shop_jokers.remove(i);
	cards::add_to_deck(g, &mut c, false);
	let cost = c.cost;
	if c.is_playing() {
		g.deck.insert(0, Card {
			face_down: true,
			..c
		});
		jokers::playing_card_joker_effects(g, 1);
	} else if !buy_and_use {
		if c.kind.is_consumable() {
			g.consumables.push(c);
		} else {
			g.jokers.push(c);
		}
		// calculate_joker({buying_card = true, card = g}) event
		g.events.push(Ev::Wait);
	} else {
		g.in_flight = Some(c);
	}
	if cost != 0.0 {
		g.events.push(Ev::EaseDollars(-cost));
	}
	if buy_and_use {
		let c = g.in_flight.take().expect("buy and use without card");
		let hl = std::mem::take(&mut g.pending_highlight);
		use_card(g, c, Some(Area::ShopJokers), &hl);
	}
}

pub(crate) fn do_sell(g: &mut Game, area: Area, idx: usize) {
	let id = g.area(area)[idx].id;
	if area == Area::Jokers {
		jokers::calc_joker(g, idx, &Ctx::SellingSelf, None);
	}
	g.events.push_after(0.2, Ev::Wait);
	g.events.delay(0.2);
	g.events.push(Ev::SellRun { card: id, area });
	for j in 0..g.jokers.len() {
		if g.jokers[j].id != id {
			jokers::calc_joker(g, j, &Ctx::SellingCard, None);
		}
	}
}

pub(crate) fn do_end_shop(g: &mut Game) {
	for j in 0..g.jokers.len() {
		jokers::calc_joker(g, j, &Ctx::EndingShop, None);
	}
	g.events.push(Ev::Wait);
	g.events.push_after(0.5, Ev::ToggleShopRun);
}

// -----------------------------------------------------------------
// use_card
// -----------------------------------------------------------------

/// `G.FUNCS.use_card` for a card already removed from `area`.
pub(crate) fn use_card(g: &mut Game, card: Card, area: Option<Area>, highlighted: &[usize]) {
	let prev = g.state;
	if matches!(card.kind, Kind::Booster(_)) {
		g.pack_interrupt = Some(g.state);
	}
	if !g.state.is_pack() {
		g.state = St::PlayTarot;
	}
	let id = card.id;
	let mut dissolve = true;
	let mut delay_fac = 1.0;
	match card.kind {
		k if k.is_consumable() => {
			let hanged_glass = if k == Kind::Tarot(Tarot::HangedMan) {
				highlighted
					.iter()
					.filter(|&&i| g.hand[i].enhancement == Enhancement::Glass)
					.count()
			} else {
				0
			};
			if !matches!(prev, St::TarotPack | St::PlanetPack | St::SpectralPack) {
				// draw_card(G.hand, G.play, ..., card)
				g.events.wait(crate::events::Trig::Before(0.1));
			}
			g.events.delay(0.2);
			g.consumed.push(card.clone());
			consumables::use_consumeable(g, &card, highlighted);
			for j in 0..g.jokers.len() {
				jokers::calc_joker(
					g,
					j,
					&Ctx::UsingConsumeable {
						kind: k,
						hanged_glass,
					},
					None,
				);
			}
		}
		Kind::Playing => {
			let mut c = card;
			c.face_down = true;
			jokers::playing_card_joker_effects(g, 1);
			cards::add_to_deck(g, &mut c, false);
			g.deck.insert(0, c);
			dissolve = false;
			delay_fac = 0.2;
		}
		Kind::Joker(_) => {
			let mut c = card;
			cards::add_to_deck(g, &mut c, false);
			g.jokers.push(c);
			dissolve = false;
			delay_fac = 0.2;
		}
		Kind::Booster(_) => {
			g.events.delay(0.1);
			if card.booster_pos > 0 {
				g.used_packs[card.booster_pos as usize - 1] = PackSlot::Opened;
			}
			g.events.wait(crate::events::Trig::Before(0.1));
			packs::open_booster(g, card);
			return;
		}
		Kind::Voucher(_) => {
			g.events.delay(0.1);
			g.events.wait(crate::events::Trig::Before(0.1));
			g.consumed.push(card.clone());
			redeem(g, &card);
		}
		_ => unreachable!(),
	}
	g.events.push_after(0.2, Ev::UseFinishOuter {
		prev,
		area,
		card: id,
		dissolve,
		delay_fac,
	});
}

/// `Card:redeem()`.
fn redeem(g: &mut Game, card: &Card) {
	let Kind::Voucher(v) = card.kind else { return };
	g.set_voucher_used(v);
	g.events.push_after(0.4, Ev::Wait);
	g.events.push(Ev::EaseDollars(-card.cost));
	g.current_voucher = None;
	apply_voucher(g, v, false);
	g.events.delay(0.6);
}

/// `Card:apply_to_run(center)`; `from_back` when applied by a deck.
pub(crate) fn apply_voucher(g: &mut Game, v: Voucher, from_back: bool) {
	use Voucher as V;
	let extra = v.def().extra;
	match v {
		V::OverstockNorm | V::OverstockPlus => g.events.push(Ev::Consumable(CEv::ShopSize(1.0))),
		V::TarotMerchant | V::TarotTycoon => g.events.push(Ev::Consumable(CEv::SetRate {
			which: Rate::Tarot,
			v: 4.0 * extra,
		})),
		V::PlanetMerchant | V::PlanetTycoon => g.events.push(Ev::Consumable(CEv::SetRate {
			which: Rate::Planet,
			v: 4.0 * extra,
		})),
		V::Hone | V::GlowUp => g.events.push(Ev::Consumable(CEv::SetRate {
			which: Rate::Edition,
			v: extra,
		})),
		V::MagicTrick | V::Illusion => g.events.push(Ev::Consumable(CEv::SetRate {
			which: Rate::PlayingCard,
			v: extra,
		})),
		V::CrystalBall => g.events.push(Ev::Consumable(CEv::ConsumableSlot)),
		V::ClearanceSale | V::Liquidation => g.events.push(Ev::Consumable(CEv::Discount(extra))),
		V::RerollSurplus | V::RerollGlut => {
			if !from_back {
				g.events.push(Ev::Consumable(CEv::RerollDiscount(extra)));
			}
		}
		V::SeedMoney | V::MoneyTree => g.events.push(Ev::Consumable(CEv::SetRate {
			which: Rate::InterestCap,
			v: extra,
		})),
		V::Grabber | V::NachoTong => {
			g.rr_hands += extra;
			g.events.push(Ev::EaseHands(extra));
		}
		V::PaintBrush | V::Palette => cards::change_hand_size(g, 1.0),
		V::Wasteful | V::Recyclomancy => {
			g.rr_discards += extra;
			g.events.push(Ev::EaseDiscard(extra));
		}
		V::Antimatter => g.events.push(Ev::Consumable(CEv::JokerSlot)),
		V::Hieroglyph | V::Petroglyph => {
			#[expect(
				clippy::cast_possible_truncation,
				reason = "the voucher's `extra` is a whole number of antes"
			)]
			let antes = extra as i32;
			g.events.push(Ev::EaseAnte(-antes));
			g.blind_ante -= antes;
			if v == V::Hieroglyph {
				g.rr_hands -= extra;
				g.events.push(Ev::EaseHands(-extra));
			} else {
				g.rr_discards -= extra;
				g.events.push(Ev::EaseDiscard(-extra));
			}
		}
		_ => {}
	}
}

/// Runs a shop / pack / use-card event (routed by `run_event`).
#[expect(clippy::too_many_lines, reason = "one arm per event")]
pub(crate) fn run_shop_event(g: &mut Game, e: Ev) {
	match e {
		Ev::ShopOuter => {
			let ex = g.shop_exists_at_open;
			g.events.push_ev(
				Ev::ShopPopulate { shop_exists: ex },
				crate::events::Trig::After(0.2),
				true,
				false,
			);
		}
		Ev::ShopPopulate { shop_exists } => shop_populate(g, shop_exists),
		Ev::StoreModify { card } => {
			for i in 0..g.tags.len() {
				if tags::apply_tag(g, i, TagCtx::StoreJokerModify { card }) {
					break;
				}
			}
		}
		Ev::RerollShopRun => {
			let final_free = g.free_rerolls > 0.0;
			g.free_rerolls = (g.free_rerolls - 1.0).max(0.0);
			cards::calculate_reroll_cost(g, final_free);
			while let Some(c) = g.shop_jokers.pop() {
				cards::drop_unowned(g, c);
			}
			fill_shop_jokers(g);
		}
		Ev::RerollShopJokers => {
			for j in 0..g.jokers.len() {
				jokers::calc_joker(g, j, &Ctx::RerollShop, None);
			}
		}
		Ev::BuyRun { card, buy_and_use } => buy_run(g, card, buy_and_use),
		Ev::SellRun { card, area } => {
			if let Some(i) = g.area(area).iter().position(|c| c.id == card) {
				let sc = g.area(area)[i].sell_cost;
				g.events.push(Ev::EaseDollars(sc));
				g.events.start_dissolve(card);
				g.events.delay(0.3);
				let is_joker = matches!(g.area(area)[i].kind, Kind::Joker(_));
				if is_joker && g.blind.kind == Some(BlindKind::FinalLeaf) {
					g.events.push(Ev::VerdantDisable);
				}
			}
		}
		Ev::VerdantDisable => blind::disable_blind(g),
		Ev::ToggleShopRun => {
			g.shop_open = false;
			for v in [
				std::mem::take(&mut g.shop_jokers),
				std::mem::take(&mut g.shop_vouchers),
				std::mem::take(&mut g.shop_boosters),
			] {
				for c in v {
					cards::drop_unowned(g, c);
				}
			}
			g.state_complete = false;
			g.state = St::BlindSelect;
		}
		Ev::OpenPack { card } => packs::open_pack_run(g, card),
		Ev::PackCreate => packs::pack_create(g),
		Ev::PackStateUi => g.events.push(Ev::PackDrawHand),
		Ev::PackDrawHand => match g.state {
			St::TarotPack | St::SpectralPack => {
				flow::draw_from_deck_to_hand(g, None);
				g.events.push_after(0.5, Ev::Wait);
			}
			St::StandardPack | St::BuffoonPack => g.events.push_after(0.5, Ev::Wait),
			_ => {}
		},
		Ev::UseFinishOuter {
			prev,
			area,
			card,
			dissolve,
			delay_fac,
		} => {
			if dissolve {
				g.events.start_dissolve(card);
			}
			g.events.push_after(0.1, Ev::UseFinishInner {
				prev,
				area,
				delay_fac,
			});
		}
		Ev::UseFinishInner {
			prev,
			area,
			delay_fac,
		} => {
			g.state = prev;
			// Using a held consumable inside a pack does not count as a pick.
			if prev.is_pack() && g.pack.is_some() && area != Some(Area::Consumables) {
				if g.pack_choices > 1 {
					g.pack_choices -= 1;
				} else {
					packs::end_consumeable(g, delay_fac);
				}
			}
		}
		Ev::EndConsumeable { df } => {
			for _ in 0..g.hand.len() {
				g.events
					.draw_ev(Pile::Hand, Pile::Deck, None, false, false, 0.08);
			}
			g.events.push_after(0.2 * df, Ev::EndConsumeableInner);
			flow::apply_new_blind_choice_tags(g);
			// save_run
			g.events.push_after(0.2 * df, Ev::Wait);
		}
		Ev::EndConsumeableInner => {
			let back = g.pack_interrupt.take().unwrap_or(St::Shop);
			g.state = back;
			g.pack = None;
			g.events.push(Ev::Wait);
		}
		other => unreachable!("not a shop event: {other:?}"),
	}
}
