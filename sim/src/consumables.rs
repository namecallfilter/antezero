//! Port of `Card:use_consumeable` / `Card:can_use_consumeable` (card.lua) plus
//! a few small deferred effects (voucher application events etc.).

use crate::card::{Card, Kind};
use crate::data::{
	ACE, Edition, Enhancement, HandType, JACK, KING, Planet, QUEEN, Rank, Seal, Spectral, Suit,
	Tarot,
};
use crate::ev::{Ev, Pile};
use crate::events::St;
use crate::game::{Area, CardSet, Game};
use crate::jokers::Ctx;
use crate::rng::Key;
use crate::{blind, cards, events, flow, jokers, num, pools, shop};

/// Deferred pieces of consumable resolution and other small events.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum CEv {
	SortHand,
	DiscardToDeckOuter,
	BlueSealPlanet,
	BlindDefeat,
	ShopSize(f64),
	SetRate {
		which: Rate,
		v: f64,
	},
	ConsumableSlot,
	Discount(f64),
	RerollDiscount(f64),
	JokerSlot,
	// Tarots
	Death {
		target: u32,
		source: u32,
	},
	Strength {
		target: u32,
	},
	SuitConv {
		target: u32,
		suit: Suit,
	},
	Enhance {
		target: u32,
		e: Enhancement,
	},
	UnhighlightAll,
	SetSeal {
		target: u32,
		seal: Seal,
	},
	Aura {
		target: u32,
	},
	Cryptid {
		source: u32,
	},
	SigilSet {
		target: u32,
		suit: Suit,
	},
	OuijaSet {
		target: u32,
		rank: Rank,
	},
	DestroyCards {
		n: u8,
		ids: [u32; 8],
	},
	CreateSpectralCards {
		which: Spectral,
		n: u8,
	},
	Fool,
	Hermit,
	Temperance,
	EmperorPriestess {
		planet: bool,
	},
	JudgementSoul {
		soul: bool,
	},
	AnkhDestroy {
		chosen: u32,
	},
	AnkhCopy {
		chosen: u32,
	},
	Wraith,
	WheelEctoHex {
		which: EditionGift,
	},
	/// Checkered Deck: Clubs → Spades, Diamonds → Hearts.
	Checkered,
}

/// A `G.GAME` rate a voucher sets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Rate {
	Tarot,
	Planet,
	Edition,
	PlayingCard,
	InterestCap,
}

/// The consumables that give a random editionless joker an edition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EditionGift {
	Wheel,
	Ectoplasm,
	Hex,
}

fn hand_pos(g: &Game, id: u32) -> Option<usize> {
	g.hand.iter().position(|c| c.id == id)
}

/// Editionless jokers (`eligible_strength_jokers` / `eligible_editionless_jokers`).
fn editionless_jokers(g: &Game) -> Vec<usize> {
	(0..g.jokers.len())
		.filter(|&i| g.jokers[i].edition == Edition::None)
		.collect()
}

fn has_editionless_joker(g: &Game) -> bool {
	g.jokers.iter().any(|c| c.edition == Edition::None)
}

/// Total sell value of the owned jokers (Temperance, Swashbuckler).
pub(crate) fn joker_sell_total(g: &Game) -> f64 {
	let mut m = 0.0;
	for c in &g.jokers {
		if matches!(c.kind, Kind::Joker(_)) {
			m += c.sell_cost;
		}
	}
	m
}

/// `Card:can_use_consumeable()` for the consumable `c` located in `area`
/// with the given highlighted hand cards.
#[must_use]
pub(crate) fn can_use_consumable(
	g: &Game, c: &Card, area: Option<Area>, n_highlighted: usize,
	highlighted_first_editionless: bool,
) -> bool {
	if !g.play.is_empty() {
		return false;
	}
	let st = g.state;
	if matches!(st, St::HandPlayed | St::DrawToHand | St::PlayTarot) {
		return false;
	}
	let in_cons = area == Some(Area::Consumables);
	let cons_room = num::float(g.consumables.len()) < g.consumable_limit.limit || in_cons;
	match c.kind {
		Kind::Planet(_)
		| Kind::Tarot(Tarot::Hermit | Tarot::Temperance)
		| Kind::Spectral(Spectral::BlackHole) => return true,
		Kind::Tarot(Tarot::WheelOfFortune)
		| Kind::Spectral(Spectral::Ectoplasm | Spectral::Hex) => {
			if has_editionless_joker(g) {
				return true;
			}
		}
		Kind::Spectral(Spectral::Ankh) => {
			if !g.jokers.is_empty() && g.joker_limit.limit > 1.0 {
				return true;
			}
		}
		Kind::Spectral(Spectral::Aura) => {
			if n_highlighted == 1 && highlighted_first_editionless {
				return true;
			}
		}
		Kind::Tarot(Tarot::Emperor | Tarot::HighPriestess) => {
			if cons_room {
				return true;
			}
		}
		Kind::Tarot(Tarot::Fool) => {
			if cons_room
				&& g.last_tarot_planet.is_some()
				&& g.last_tarot_planet != Some(Kind::Tarot(Tarot::Fool))
			{
				return true;
			}
		}
		Kind::Tarot(Tarot::Judgement) | Kind::Spectral(Spectral::Soul | Spectral::Wraith) => {
			return num::float(g.jokers.len()) < g.joker_limit.limit;
		}
		_ => {}
	}
	if matches!(
		st,
		St::SelectingHand | St::TarotPack | St::SpectralPack | St::PlanetPack
	) {
		let (max_h, min_h) = consumable_highlight(c.kind);
		if max_h > 0 {
			let mod_num = max_h.min(5);
			if mod_num as usize >= n_highlighted && n_highlighted >= min_h.max(1) as usize {
				return true;
			}
		}
		if matches!(
			c.kind,
			Kind::Spectral(
				Spectral::Familiar
					| Spectral::Grim
					| Spectral::Incantation
					| Spectral::Immolate
					| Spectral::Sigil
					| Spectral::Ouija
			)
		) && g.hand.len() > 1
		{
			return true;
		}
	}
	false
}

/// `set_consumeable_usage(card)`.
fn set_consumeable_usage(g: &mut Game, c: &Card) {
	match c.kind {
		Kind::Tarot(_) => g.consumable_usage_tarot += 1.0,
		Kind::Planet(p) => {
			g.consumable_usage_planet += 1.0;
			g.planets_used |= 1 << (p as u16);
		}
		Kind::Spectral(_) => g.consumable_usage_spectral += 1.0,
		_ => {}
	}
	if matches!(c.kind, Kind::Tarot(_) | Kind::Planet(_)) {
		g.events.push(Ev::LastTarotPlanetOuter { kind: c.kind });
	}
}

/// `Card:use_consumeable(area)`; `hl` are the highlighted hand indices.
#[expect(
	clippy::too_many_lines,
	reason = "one block per consumable, in the order of `Card:use_consumeable`"
)]
pub(crate) fn use_consumeable(g: &mut Game, card: &Card, hl: &[usize]) {
	use Spectral as S;
	use Tarot as T;

	set_consumeable_usage(g, card);
	if card.debuff {
		return;
	}
	let mut hl: Vec<usize> = hl.to_vec();
	hl.sort_unstable();
	let hl_ids: Vec<u32> = hl.iter().map(|&i| g.hand[i].id).collect();
	let k = card.kind;

	// Enhancement / suit / rank conversions.
	let mod_conv = match k {
		Kind::Tarot(T::Magician) => Some(Enhancement::Lucky),
		Kind::Tarot(T::Empress) => Some(Enhancement::Mult),
		Kind::Tarot(T::Heirophant) => Some(Enhancement::Bonus),
		Kind::Tarot(T::Lovers) => Some(Enhancement::Wild),
		Kind::Tarot(T::Chariot) => Some(Enhancement::Steel),
		Kind::Tarot(T::Justice) => Some(Enhancement::Glass),
		Kind::Tarot(T::Devil) => Some(Enhancement::Gold),
		Kind::Tarot(T::Tower) => Some(Enhancement::Stone),
		_ => None,
	};
	let suit_conv = match k {
		Kind::Tarot(T::Star) => Some(Suit::Diamonds),
		Kind::Tarot(T::Moon) => Some(Suit::Clubs),
		Kind::Tarot(T::Sun) => Some(Suit::Hearts),
		Kind::Tarot(T::World) => Some(Suit::Spades),
		_ => None,
	};
	let is_conv = mod_conv.is_some()
		|| suit_conv.is_some()
		|| matches!(k, Kind::Tarot(T::Death | T::Strength));
	if consumable_highlight(k).0 > 0 {
		g.events.hand_text_imm(0.0);
	}
	if is_conv {
		g.events.push_after(0.4, Ev::Wait);
		for _ in &hl_ids {
			g.events.push_after(0.15, Ev::Wait);
		}
		g.events.delay(0.2);
		if k == Kind::Tarot(T::Death) {
			// Rightmost highlighted card is the template.
			if let Some(&src) = hl_ids.last() {
				for &t in &hl_ids {
					g.events.push_after(
						0.1,
						Ev::Consumable(CEv::Death {
							target: t,
							source: src,
						}),
					);
				}
			}
		} else if k == Kind::Tarot(T::Strength) {
			for &t in &hl_ids {
				g.events
					.push_after(0.1, Ev::Consumable(CEv::Strength { target: t }));
			}
		} else if let Some(s) = suit_conv {
			for &t in &hl_ids {
				g.events
					.push_after(0.1, Ev::Consumable(CEv::SuitConv { target: t, suit: s }));
			}
		} else if let Some(e) = mod_conv {
			for &t in &hl_ids {
				g.events
					.push_after(0.1, Ev::Consumable(CEv::Enhance { target: t, e }));
			}
		}
		for _ in &hl_ids {
			g.events.push_after(0.15, Ev::Wait);
		}
		g.events
			.push_after(0.2, Ev::Consumable(CEv::UnhighlightAll));
		g.events.delay(0.5);
	}
	if k == Kind::Spectral(S::BlackHole) {
		g.events.hand_text(0.3);
		g.events.push_after(0.2, Ev::Wait);
		g.events.hand_text(0.0);
		g.events.push_after(0.9, Ev::Wait);
		g.events.hand_text(0.0);
		g.events.push_after(0.9, Ev::Wait);
		g.events.hand_text(0.0);
		g.events.delay(1.3);
		for h in HandType::PAIRS_ORDER {
			events::level_up_hand_ev(g, h, 1.0, true);
		}
		g.events.hand_text(0.0);
	}
	let seal = match k {
		Kind::Spectral(S::Talisman) => Some(Seal::Gold),
		Kind::Spectral(S::DejaVu) => Some(Seal::Red),
		Kind::Spectral(S::Trance) => Some(Seal::Blue),
		Kind::Spectral(S::Medium) => Some(Seal::Purple),
		_ => None,
	};
	if let Some(seal) = seal {
		let t = hl_ids[0];
		g.events.push(Ev::Wait);
		g.events
			.push_after(0.1, Ev::Consumable(CEv::SetSeal { target: t, seal }));
		g.events.delay(0.5);
		g.events
			.push_after(0.2, Ev::Consumable(CEv::UnhighlightAll));
	}
	if k == Kind::Spectral(S::Aura) {
		g.events
			.push_after(0.4, Ev::Consumable(CEv::Aura { target: hl_ids[0] }));
	}
	if k == Kind::Spectral(S::Cryptid) {
		g.events
			.push(Ev::Consumable(CEv::Cryptid { source: hl_ids[0] }));
	}
	if matches!(k, Kind::Spectral(S::Sigil | S::Ouija)) {
		let n = g.hand.len();
		g.events.push_after(0.4, Ev::Wait);
		for _ in 0..n {
			g.events.push_after(0.15, Ev::Wait);
		}
		g.events.delay(0.2);
		if k == Kind::Spectral(S::Sigil) {
			let idx = g.rng.element_str("sigil", 4);
			let suit = [Suit::Spades, Suit::Hearts, Suit::Diamonds, Suit::Clubs][idx];
			for i in 0..n {
				let target = g.hand[i].id;
				g.events
					.push(Ev::Consumable(CEv::SigilSet { target, suit }));
			}
		} else {
			const R: [Rank; 13] = [2, 3, 4, 5, 6, 7, 8, 9, 10, JACK, QUEEN, KING, ACE];
			let idx = g.rng.element_str("ouija", 13);
			let rank = R[idx];
			for i in 0..n {
				let target = g.hand[i].id;
				g.events
					.push(Ev::Consumable(CEv::OuijaSet { target, rank }));
			}
			cards::change_hand_size(g, -1.0);
		}
		for _ in 0..n {
			g.events.push_after(0.15, Ev::Wait);
		}
		g.events.delay(0.5);
	}
	if let Kind::Planet(p) = k {
		g.events.hand_text(0.3);
		events::level_up_hand_ev(g, p.hand(), 1.0, false);
		g.events.hand_text(0.0);
	}
	// remove_card consumables.
	let removes = matches!(
		k,
		Kind::Tarot(T::HangedMan)
			| Kind::Spectral(S::Familiar | S::Grim | S::Incantation | S::Immolate)
	);
	if removes {
		let mut destroyed: Vec<u32> = Vec::new();
		match k {
			Kind::Tarot(T::HangedMan) => {
				for &id in hl_ids.iter().rev() {
					destroyed.push(id);
				}
			}
			Kind::Spectral(S::Familiar | S::Grim | S::Incantation) => {
				let mut order: Vec<usize> = (0..g.hand.len()).collect();
				order.sort_by_key(|&i| g.hand[i].id);
				let i = order[g.rng.element_str("random_destroy", order.len())];
				destroyed.push(g.hand[i].id);
			}
			Kind::Spectral(S::Immolate) => {
				let mut tmp: Vec<u32> = g.hand.iter().map(|c| c.id).collect();
				tmp.sort_unstable();
				let seed = g.rng.seed_str("immolate");
				crate::rng::shuffle(&mut tmp, seed);
				for id in tmp.into_iter().take(5) {
					destroyed.push(id);
				}
			}
			_ => {}
		}
		// The destroy loops run back to front.
		let order: Vec<u32> = if k == Kind::Tarot(T::HangedMan) {
			destroyed.clone()
		} else {
			destroyed.iter().rev().copied().collect()
		};
		let mut ids = [0u32; 8];
		for (i, &id) in order.iter().enumerate().take(8) {
			ids[i] = id;
		}
		g.events.push_after(0.4, Ev::Wait);
		let delay = if k == Kind::Tarot(T::HangedMan) {
			0.2
		} else {
			0.1
		};
		g.events.push_after(
			delay,
			Ev::Consumable(CEv::DestroyCards {
				n: num::byte(order.len()),
				ids,
			}),
		);
		match k {
			Kind::Spectral(s @ (S::Familiar | S::Grim | S::Incantation)) => {
				let n = match s {
					S::Familiar => 3,
					S::Grim => 2,
					_ => 4,
				};
				g.events.push_after(
					0.7,
					Ev::Consumable(CEv::CreateSpectralCards { which: s, n }),
				);
			}
			Kind::Spectral(S::Immolate) => {
				g.events.delay(0.5);
				g.events.push(Ev::EaseDollars(20.0));
			}
			_ => {}
		}
		g.events.delay(0.3);
		let par = g.pareidolia();
		let faces = destroyed
			.iter()
			.filter(|&&id| {
				g.hand
					.iter()
					.find(|c| c.id == id)
					.is_some_and(|c| c.is_face(false, par))
			})
			.count();
		for j in 0..g.jokers.len() {
			jokers::calc_joker(g, j, &Ctx::RemovePlayingCards { faces, glass: 0 }, None);
		}
	}
	match k {
		Kind::Tarot(T::Fool) => {
			g.events.push_after(0.4, Ev::Consumable(CEv::Fool));
			g.events.delay(0.6);
		}
		Kind::Tarot(T::Hermit) => {
			g.events.push_after(0.4, Ev::Consumable(CEv::Hermit));
			g.events.delay(0.6);
		}
		Kind::Tarot(T::Temperance) => {
			g.events.push_after(0.4, Ev::Consumable(CEv::Temperance));
			g.events.delay(0.6);
		}
		Kind::Tarot(T::Emperor | T::HighPriestess) => {
			let n = 2f64.min(g.consumable_limit.limit - num::float(g.consumables.len()));
			let planet = k == Kind::Tarot(T::HighPriestess);
			let mut i = 1.0;
			while i <= n {
				g.events
					.push_after(0.4, Ev::Consumable(CEv::EmperorPriestess { planet }));
				i += 1.0;
			}
			g.events.delay(0.6);
		}
		Kind::Tarot(T::Judgement) | Kind::Spectral(S::Soul) => {
			let soul = k == Kind::Spectral(S::Soul);
			g.events
				.push_after(0.4, Ev::Consumable(CEv::JudgementSoul { soul }));
			g.events.delay(0.6);
		}
		Kind::Spectral(S::Ankh) => {
			let mut order: Vec<usize> = (0..g.jokers.len()).collect();
			order.sort_by_key(|&i| g.jokers[i].id);
			let pick = order[g.rng.element_str("ankh_choice", order.len())];
			let chosen = g.jokers[pick].id;
			g.ankh_deletable = g
				.jokers
				.iter()
				.filter(|c| !c.ab.eternal)
				.map(|c| c.id)
				.collect();
			g.events
				.push_before(0.75, Ev::Consumable(CEv::AnkhDestroy { chosen }));
			g.events
				.push_before(0.4, Ev::Consumable(CEv::AnkhCopy { chosen }));
		}
		Kind::Spectral(S::Wraith) => {
			g.events.push_after(0.4, Ev::Consumable(CEv::Wraith));
			g.events.delay(0.6);
		}
		Kind::Tarot(T::WheelOfFortune) => {
			let k = Key::from("wheel_of_fortune");
			if g.rng.random(&k) < g.probabilities / 4.0 {
				g.events.push_after(
					0.4,
					Ev::Consumable(CEv::WheelEctoHex {
						which: EditionGift::Wheel,
					}),
				);
			} else {
				g.events.push_after(0.4, Ev::Wait);
			}
			g.events.delay(0.6);
		}
		Kind::Spectral(S::Ectoplasm) => {
			g.events.push_after(
				0.4,
				Ev::Consumable(CEv::WheelEctoHex {
					which: EditionGift::Ectoplasm,
				}),
			);
			g.events.delay(0.6);
		}
		Kind::Spectral(S::Hex) => {
			g.events.push_after(
				0.4,
				Ev::Consumable(CEv::WheelEctoHex {
					which: EditionGift::Hex,
				}),
			);
			g.events.delay(0.6);
		}
		_ => {}
	}
}

#[expect(clippy::too_many_lines, reason = "one arm per event")]
pub(crate) fn run_consumable_event(g: &mut Game, e: CEv) {
	match e {
		CEv::SortHand => flow::sort_hand(g),
		CEv::DiscardToDeckOuter => {
			for _ in 0..g.discard.len() {
				g.events
					.draw_ev(Pile::Discard, Pile::Deck, None, false, false, 0.005);
			}
		}
		CEv::BlueSealPlanet => {
			if let Some(h) = g.last_hand_played {
				let p = Planet::for_hand(h);
				let mut c = pools::create_card(
					g,
					CardSet::Planet,
					Area::Consumables,
					false,
					None,
					false,
					Some(Kind::Planet(p)),
					"blusl",
				);
				cards::add_to_deck(g, &mut c, false);
				g.consumables.push(c);
				g.consumeable_buffer = 0.0;
			}
		}
		CEv::BlindDefeat => blind::defeat(g),
		CEv::ShopSize(m) => shop::change_shop_size(g, m),
		CEv::SetRate { which, v } => match which {
			Rate::Tarot => g.tarot_rate = v,
			Rate::Planet => g.planet_rate = v,
			Rate::Edition => g.edition_rate = v,
			Rate::PlayingCard => g.playing_card_rate = v,
			Rate::InterestCap => g.interest_cap = v,
		},
		CEv::ConsumableSlot => g.consumable_limit.limit += 1.0,
		CEv::Discount(d) => {
			g.discount_percent = d;
			cards::set_cost_all(g);
		}
		CEv::RerollDiscount(d) => {
			g.rr_reroll_cost -= d;
			g.reroll_cost = (g.reroll_cost - d).max(0.0);
		}
		CEv::JokerSlot => g.joker_limit.limit += 1.0,
		CEv::Death { target, source } => {
			if target == source {
				return;
			}
			let (Some(ti), Some(si)) = (hand_pos(g, target), hand_pos(g, source)) else {
				return;
			};
			copy_into(g, si, ti);
		}
		CEv::Strength { target } => {
			let Some(i) = hand_pos(g, target) else {
				return;
			};
			let c = &mut g.hand[i];
			let rank = if c.rank == ACE {
				2
			} else {
				(c.rank + 1).min(ACE)
			};
			c.set_base(c.suit, rank);
			blind::debuff_hand_card(g, i);
		}
		CEv::SuitConv { target, suit } => {
			let Some(i) = hand_pos(g, target) else {
				return;
			};
			let c = &mut g.hand[i];
			c.set_base(suit, c.rank);
			blind::debuff_hand_card(g, i);
		}
		CEv::Enhance { target, e } => {
			let Some(i) = hand_pos(g, target) else {
				return;
			};
			cards::set_enhancement(g, Area::Hand, i, e);
		}
		CEv::UnhighlightAll => {}
		CEv::SetSeal { target, seal } => {
			if let Some(i) = hand_pos(g, target) {
				g.hand[i].seal = Some(seal);
				cards::reprice(g, Area::Hand, i);
			}
		}
		CEv::Aura { target } => {
			let k = Key::from("aura");
			let e = pools::poll_edition(g, &k, 1.0, true, true);
			if let Some(i) = hand_pos(g, target) {
				cards::set_edition_at(g, Area::Hand, i, e);
			}
			g.events.edition_fx(e, true);
		}
		CEv::Cryptid { source } => {
			let Some(si) = hand_pos(g, source) else {
				return;
			};
			for _ in 0..2 {
				let mut c = flow::copy_card(g, g.hand[si].clone(), false);
				cards::add_to_deck(g, &mut c, false);
				g.hand.push(c);
				g.events.materialize(1.0);
			}
			jokers::playing_card_joker_effects(g, 2);
		}
		CEv::SigilSet { target, suit } => {
			if let Some(i) = hand_pos(g, target) {
				let c = &mut g.hand[i];
				c.set_base(suit, c.rank);
				blind::debuff_hand_card(g, i);
			}
		}
		CEv::OuijaSet { target, rank } => {
			if let Some(i) = hand_pos(g, target) {
				let c = &mut g.hand[i];
				c.set_base(c.suit, rank);
				blind::debuff_hand_card(g, i);
			}
		}
		CEv::DestroyCards { n, ids } => {
			for &id in &ids[..n as usize] {
				if let Some(i) = hand_pos(g, id) {
					if g.hand[i].enhancement == Enhancement::Glass {
						g.hand[i].shattered = true;
						g.events.shatter(id);
					} else {
						g.events.start_dissolve(id);
					}
				}
			}
		}
		CEv::CreateSpectralCards { which, n } => {
			// Enhanced pool without Stone, in pool order.
			const POOL: [Enhancement; 7] = [
				Enhancement::Bonus,
				Enhancement::Mult,
				Enhancement::Wild,
				Enhancement::Glass,
				Enhancement::Steel,
				Enhancement::Gold,
				Enhancement::Lucky,
			];
			for _ in 0..n {
				let (rank, suit) = match which {
					Spectral::Familiar => {
						let rank = [JACK, QUEEN, KING][g.rng.element_str("familiar_create", 3)];
						let s = [Suit::Spades, Suit::Hearts, Suit::Diamonds, Suit::Clubs]
							[g.rng.element_str("familiar_create", 4)];
						(rank, s)
					}
					Spectral::Grim => {
						let s = [Suit::Spades, Suit::Hearts, Suit::Diamonds, Suit::Clubs]
							[g.rng.element_str("grim_create", 4)];
						(ACE, s)
					}
					_ => {
						let rank = [2, 3, 4, 5, 6, 7, 8, 9, 10]
							[g.rng.element_str("incantation_create", 9)];
						let s = [Suit::Spades, Suit::Hearts, Suit::Diamonds, Suit::Clubs]
							[g.rng.element_str("incantation_create", 4)];
						(rank, s)
					}
				};
				let e = POOL[g.rng.element_str("spe_card", 7)];
				let mut c = cards::make_playing_card(g, suit, rank, e);
				cards::add_to_deck(g, &mut c, false);
				g.events.materialize(1.0);
				g.hand.push(c);
			}
			jokers::playing_card_joker_effects(g, n as usize);
		}
		CEv::Fool => {
			if g.consumable_limit.limit > num::float(g.consumables.len()) {
				let forced = g.last_tarot_planet;
				let mut c = pools::create_card(
					g,
					CardSet::TarotPlanet,
					Area::Consumables,
					false,
					None,
					false,
					forced,
					"fool",
				);
				cards::add_to_deck(g, &mut c, false);
				g.consumables.push(c);
			}
		}
		CEv::Hermit => {
			let d = 0f64.max(g.dollars.min(20.0));
			cards::ease_dollars(g, d);
		}
		CEv::Temperance => {
			let m = joker_sell_total(g).min(50.0);
			cards::ease_dollars(g, m);
		}
		CEv::EmperorPriestess { planet } => {
			if g.consumable_limit.limit > num::float(g.consumables.len()) {
				let (set, app) = if planet {
					(CardSet::Planet, "pri")
				} else {
					(CardSet::Tarot, "emp")
				};
				let mut c =
					pools::create_card(g, set, Area::Consumables, false, None, false, None, app);
				cards::add_to_deck(g, &mut c, false);
				g.consumables.push(c);
			}
		}
		CEv::JudgementSoul { soul } => {
			let mut c = pools::create_card(
				g,
				CardSet::Joker,
				Area::Jokers,
				soul,
				None,
				false,
				None,
				if soul { "sou" } else { "jud" },
			);
			cards::add_to_deck(g, &mut c, false);
			g.jokers.push(c);
		}
		CEv::AnkhDestroy { chosen } => {
			let del = std::mem::take(&mut g.ankh_deletable);
			for id in del {
				if id != chosen {
					g.events.start_dissolve(id);
				}
			}
		}
		CEv::AnkhCopy { chosen } => {
			let Some(i) = g.jokers.iter().position(|c| c.id == chosen) else {
				return;
			};
			let strip = g.jokers[i].edition == Edition::Negative;
			let mut c = flow::copy_card(g, g.jokers[i].clone(), strip);
			g.events.materialize(1.0);
			cards::add_to_deck(g, &mut c, false);
			g.jokers.push(c);
		}
		CEv::Wraith => {
			let mut c = pools::create_card(
				g,
				CardSet::Joker,
				Area::Jokers,
				false,
				Some(0.99),
				false,
				None,
				"wra",
			);
			cards::add_to_deck(g, &mut c, false);
			g.jokers.push(c);
			if g.dollars != 0.0 {
				let d = g.dollars;
				cards::ease_dollars(g, -d);
			}
		}
		CEv::Checkered => checkered(g),
		CEv::WheelEctoHex { which } => {
			let mut pool = editionless_jokers(g);
			if pool.is_empty() {
				return;
			}
			pool.sort_by_key(|&i| g.jokers[i].id);
			let key = match which {
				EditionGift::Wheel => "wheel_of_fortune",
				EditionGift::Ectoplasm => "ectoplasm",
				EditionGift::Hex => "hex",
			};
			let pick = pool[g.rng.element_str(key, pool.len())];
			let ed = match which {
				EditionGift::Ectoplasm => Edition::Negative,
				EditionGift::Hex => Edition::Polychrome,
				EditionGift::Wheel => {
					let k = Key::from("wheel_of_fortune");
					pools::poll_edition(g, &k, 1.0, true, true)
				}
			};
			cards::set_edition_at(g, Area::Jokers, pick, ed);
			g.events.edition_fx(ed, true);
			if which == EditionGift::Hex {
				for i in 0..g.jokers.len() {
					let c = &g.jokers[i];
					if i != pick && !c.ab.eternal {
						let id = c.id;
						g.events.start_dissolve(id);
					}
				}
			}
			if which == EditionGift::Ectoplasm {
				let m = g.ecto_minus;
				cards::change_hand_size(g, -m);
				g.ecto_minus += 1.0;
			}
		}
	}
}

fn checkered(g: &mut Game) {
	let (blind, smeared, pareidolia) = (g.blind, g.smeared(), g.pareidolia());
	for c in &mut g.deck {
		let to = match c.suit {
			Suit::Clubs => Some(Suit::Spades),
			Suit::Diamonds => Some(Suit::Hearts),
			_ => None,
		};
		if let Some(s) = to {
			c.set_base(s, c.rank);
			let debuffed = crate::blind::debuffs(&blind, smeared, pareidolia, c);
			cards::set_debuff_playing(c, debuffed);
		}
	}
}

/// `copy_card(other, new_card)` from the `si`-th hand card into the `ti`-th
/// (Death).
fn copy_into(g: &mut Game, si: usize, ti: usize) {
	let mut c = g.hand[si].clone();
	let t = &g.hand[ti];
	c.id = t.id;
	c.orig_suit = t.orig_suit;
	c.added_to_deck = t.added_to_deck;
	c.lucky_trigger = false;
	// Not part of what copy_card copies: the card keeps its facing and
	// area flags; set_base resets times_played.
	c.face_down = t.face_down;
	c.times_played = 0;
	// set_ability keeps the target's forced_selection; the copied
	// ability table only overrides it when the source has one.
	c.ab.forced_selection |= t.ab.forced_selection;
	c.shattered = t.shattered;
	c.destroyed = t.destroyed;
	c.getting_sliced = t.getting_sliced;
	cards::mark_used(g, &c);
	cards::set_cost(g, &mut c, Some(Area::Hand));
	g.hand[ti] = c;
}

/// Whether a consumable acts on highlighted hand cards (Aura has no
/// `max_highlighted` but needs exactly one card).
pub(crate) fn uses_targets(k: Kind) -> bool {
	consumable_highlight(k).0 > 0 || k == Kind::Spectral(Spectral::Aura)
}

/// `(max_highlighted, min_highlighted)`.
pub(crate) fn consumable_highlight(k: Kind) -> (u8, u8) {
	match k {
		Kind::Tarot(t) => (t.def().max_highlighted, t.def().min_highlighted),
		Kind::Spectral(s) => (s.def().max_highlighted, s.def().min_highlighted),
		_ => (0, 0),
	}
}
