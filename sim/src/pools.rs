//! Pools and seeded sampling (`common_events.lua`): editions, card pools,
//! `create_card`, round-reset samplers, bosses, vouchers, tags and packs.

use crate::card::{Card, Kind};
use crate::cards;
use crate::data::{
	ACE, ALL_BLINDS, ALL_BOOSTERS, ALL_PLANETS, ALL_SPECTRALS, ALL_TAGS, ALL_TAROTS, ALL_VOUCHERS,
	BlindKind, Booster, Center, Edition, Enhancement, Joker, P_CARDS_SORTED, PackKind, Planet,
	RARITY_POOL_1, RARITY_POOL_2, RARITY_POOL_3, RARITY_POOL_4, Spectral, Suit, TagKind, Tarot,
	Voucher,
};
use crate::game::{Area, CardSet, Game, IdolCard, MailRank};
use crate::rng::Key;

// -----------------------------------------------------------------
// Pools and sampling (common_events.lua)
// -----------------------------------------------------------------

/// `poll_edition(key, mod, no_neg, guaranteed)`.
pub(crate) fn poll_edition(
	g: &mut Game, key: &Key, m: f64, no_neg: bool, guaranteed: bool,
) -> Edition {
	let p = g.rng.random(key);
	if guaranteed {
		if p > 1.0 - 0.003 * 25.0 && !no_neg {
			Edition::Negative
		} else if p > 1.0 - 0.006 * 25.0 {
			Edition::Polychrome
		} else if p > 1.0 - 0.02 * 25.0 {
			Edition::Holo
		} else if p > 0.0 {
			// (the game compares with 1 - 0.04*25, which is exactly 0)
			Edition::Foil
		} else {
			Edition::None
		}
	} else if p > 1.0 - 0.003 * m && !no_neg {
		Edition::Negative
	} else if p > 1.0 - 0.006 * g.edition_rate * m {
		Edition::Polychrome
	} else if p > 1.0 - 0.02 * g.edition_rate * m {
		Edition::Holo
	} else if p > 1.0 - 0.04 * g.edition_rate * m {
		Edition::Foil
	} else {
		Edition::None
	}
}

#[inline]
fn showman(g: &Game) -> bool {
	g.has_joker(Joker::RingMaster)
}

#[inline]
fn used_blocks(g: &Game, c: Center) -> bool {
	g.used_jokers.contains(c) && !showman(g)
}

/// Builds the pool for `_type` and samples it, including the resample
/// loop (`get_current_pool` + `pseudorandom_element`).
#[expect(
	clippy::too_many_lines,
	reason = "one arm per card set, as in `get_current_pool`"
)]
pub(crate) fn sample_pool(
	g: &mut Game, set: CardSet, rarity: Option<f64>, legendary: bool, append: &str,
) -> Kind {
	let mut pool: [Option<Kind>; 160] = [None; 160];
	let mut n = 0usize;
	let mut avail = 0usize;
	let mut key = Key::new();
	match set {
		CardSet::Joker => {
			let r = if let Some(r) = rarity {
				r
			} else {
				let mut k = Key::from("rarity");
				k.push_int(i64::from(g.ante)).push(append);
				g.rng.random(&k)
			};
			let rarity = if legendary {
				4
			} else if r > 0.95 {
				3
			} else if r > 0.7 {
				2
			} else {
				1
			};
			let src: &[Joker] = match rarity {
				1 => &RARITY_POOL_1,
				2 => &RARITY_POOL_2,
				3 => &RARITY_POOL_3,
				_ => &RARITY_POOL_4,
			};
			key.push("Joker").push_int(i64::from(rarity));
			if !legendary {
				key.push(append);
			}
			for &j in src {
				let mut add = false;
				if !used_blocks(g, Center::Joker(j)) {
					let d = j.def();
					add = match d.enhancement_gate {
						Some(gate) => g.playing_cards().any(|c| c.enhancement == gate),
						None => true,
					};
					if d.no_pool_flag.is_some() && g.gros_michel_extinct {
						add = false;
					}
					if d.yes_pool_flag.is_some() && !g.gros_michel_extinct {
						add = false;
					}
				}
				if add && !g.banned.contains(Center::Joker(j)) {
					pool[n] = Some(Kind::Joker(j));
					avail += 1;
				}
				n += 1;
			}
		}
		CardSet::Tarot | CardSet::TarotPlanet => {
			key.push(set.name()).push(append);
			if set == CardSet::Tarot {
				for &t in &ALL_TAROTS {
					let c = Center::Tarot(t);
					if !used_blocks(g, c) && !g.banned.contains(c) {
						pool[n] = Some(Kind::Tarot(t));
						avail += 1;
					}
					n += 1;
				}
			} else {
				unimplemented!("Tarot_Planet pool sampling is never reached in the base game");
			}
		}
		CardSet::Planet => {
			key.push("Planet").push(append);
			for &p in &ALL_PLANETS {
				let c = Center::Planet(p);
				let mut add = false;
				if !used_blocks(g, c) {
					let softlock = matches!(p, Planet::PlanetX | Planet::Ceres | Planet::Eris);
					add = !softlock || g.hands[p.hand() as usize].played > 0.0;
				}
				if add && !g.banned.contains(c) {
					pool[n] = Some(Kind::Planet(p));
					avail += 1;
				}
				n += 1;
			}
		}
		CardSet::Spectral => {
			key.push("Spectral").push(append);
			for &s in &ALL_SPECTRALS {
				let c = Center::Spectral(s);
				let mut add = !used_blocks(g, c);
				if matches!(s, Spectral::Soul | Spectral::BlackHole) {
					add = false;
				}
				if add && !g.banned.contains(c) {
					pool[n] = Some(Kind::Spectral(s));
					avail += 1;
				}
				n += 1;
			}
		}
		CardSet::Enhanced => unreachable!("Enhanced cards are sampled by sample_enhancement"),
		CardSet::Base => unreachable!("Base cards are forced to c_base"),
	}
	if !legendary {
		key.push_int(i64::from(g.ante));
	}
	if avail == 0 {
		// The game samples a one-entry fallback pool with the same seed.
		g.rng.element(&key, 1);
		return match set {
			CardSet::Tarot | CardSet::TarotPlanet => Kind::Tarot(Tarot::Strength),
			CardSet::Planet => Kind::Planet(Planet::Pluto),
			CardSet::Spectral => Kind::Spectral(Spectral::Incantation),
			_ => Kind::Joker(Joker::Joker),
		};
	}
	let mut i = g.rng.element(&key, n);
	let mut it = 1;
	loop {
		if let Some(kind) = pool[i] {
			return kind;
		}
		it += 1;
		let mut k = key;
		k.push("_resample").push_int(it);
		i = g.rng.element(&k, n);
	}
}

/// The enhancement chosen by sampling the `Enhanced` pool.
pub(crate) fn sample_enhancement(g: &mut Game, append: &str) -> Enhancement {
	let mut key = Key::from("Enhanced");
	key.push(append).push_int(i64::from(g.ante));
	// The Enhanced pool is never filtered by usage, only by bans.
	let n = Enhancement::POOL.len();
	let mut i = g.rng.element(&key, n);
	let mut it = 1;
	while g.banned.contains(Center::Enhanced(Enhancement::POOL[i])) {
		it += 1;
		let mut k = key;
		k.push("_resample").push_int(it);
		i = g.rng.element(&k, n);
	}
	Enhancement::POOL[i]
}

/// `create_card(_type, area, legendary, _rarity, skip_materialize, soulable, forced_key, key_append)`.
#[expect(clippy::too_many_arguments, reason = "the game's signature")]
#[expect(clippy::too_many_lines, reason = "follows `create_card` step by step")]
pub(crate) fn create_card(
	g: &mut Game, set: CardSet, area: Area, legendary: bool, rarity: Option<f64>, soulable: bool,
	forced: Option<Kind>, append: &str,
) -> Card {
	let mut forced = forced;
	if forced.is_none() && soulable && !g.banned.contains(Center::Spectral(Spectral::Soul)) {
		if matches!(
			set,
			CardSet::Tarot | CardSet::Spectral | CardSet::TarotPlanet
		) && !used_blocks(g, Center::Spectral(Spectral::Soul))
		{
			let mut k = Key::from("soul_");
			k.push(set.name()).push_int(i64::from(g.ante));
			if g.rng.random(&k) > 0.997 {
				forced = Some(Kind::Spectral(Spectral::Soul));
			}
		}
		if matches!(set, CardSet::Planet | CardSet::Spectral)
			&& !used_blocks(g, Center::Spectral(Spectral::BlackHole))
		{
			let mut k = Key::from("soul_");
			k.push(set.name()).push_int(i64::from(g.ante));
			if g.rng.random(&k) > 0.997 {
				forced = Some(Kind::Spectral(Spectral::BlackHole));
			}
		}
	}
	let mut set = set;
	if set == CardSet::Base {
		forced = Some(Kind::Playing);
	}
	let banned_forced = match forced {
		Some(Kind::Playing) | None => false,
		Some(k) => g.banned.contains(k.center(Enhancement::Base)),
	};
	let mut enh = Enhancement::Base;
	let kind = match forced {
		Some(k) if !banned_forced => {
			// `_type = center.set ~= 'Default' and center.set or _type`
			set = match k {
				Kind::Joker(_) => CardSet::Joker,
				Kind::Tarot(_) => CardSet::Tarot,
				Kind::Planet(_) => CardSet::Planet,
				Kind::Spectral(_) => CardSet::Spectral,
				_ => set,
			};
			k
		}
		_ => {
			if set == CardSet::Enhanced {
				enh = sample_enhancement(g, append);
				Kind::Playing
			} else {
				sample_pool(g, set, rarity, legendary, append)
			}
		}
	};
	let mut card = if kind == Kind::Playing {
		// front = pseudorandom_element(G.P_CARDS, pseudoseed('front'..append..ante))
		let mut k = Key::from("front");
		k.push(append).push_int(i64::from(g.ante));
		let i = g.rng.element(&k, 52);
		let (suit, rank) = P_CARDS_SORTED[i];
		cards::make_playing_card(g, suit, rank, enh)
	} else {
		cards::make_card(g, kind)
	};
	if card.kind.is_consumable() && area != Area::PackCards {
		g.events.materialize(1.0);
	}
	if set == CardSet::Joker {
		if matches!(area, Area::ShopJokers | Area::PackCards) {
			let mut k = Key::from(if area == Area::PackCards {
				"packetper"
			} else {
				"etperpoll"
			});
			k.push_int(i64::from(g.ante));
			let poll = g.rng.random(&k);
			if g.modifiers.enable_eternals_in_shop && poll > 0.7 {
				card.set_eternal(true);
			} else if g.modifiers.enable_perishables_in_shop && poll > 0.4 && poll <= 0.7 {
				cards::set_perishable(g, &mut card);
			}
			if g.modifiers.enable_rentals_in_shop {
				let mut k = Key::from(if area == Area::PackCards {
					"packssjr"
				} else {
					"ssjr"
				});
				k.push_int(i64::from(g.ante));
				if g.rng.random(&k) > 0.7 {
					card.ab.rental = true;
					cards::set_cost(g, &mut card, None);
				}
			}
		}
		let mut k = Key::from("edi");
		k.push(append).push_int(i64::from(g.ante));
		let e = poll_edition(g, &k, 1.0, false, false);
		if e == Edition::None {
			cards::set_cost(g, &mut card, None);
		} else {
			cards::set_edition(g, &mut card, e, None);
			g.events.edition_fx(e, false);
		}
	}
	card
}

// -----------------------------------------------------------------
// Round-reset samplers
// -----------------------------------------------------------------

pub(crate) fn reset_idol_card(g: &mut Game) {
	g.idol_card.rank = ACE;
	g.idol_card.suit = Suit::Spades;
	let v = g.non_stone_sorted();
	if !v.is_empty() {
		let k = g.key_ante("idol");
		let (rank, suit) = v[g.rng.element(&k, v.len())];
		g.idol_card = IdolCard {
			rank,
			suit,
			id: Some(rank),
		};
	}
}

pub(crate) fn reset_mail_rank(g: &mut Game) {
	g.mail_rank.rank = ACE;
	let v = g.non_stone_sorted();
	if !v.is_empty() {
		let k = g.key_ante("mail");
		let (rank, _) = v[g.rng.element(&k, v.len())];
		g.mail_rank = MailRank {
			rank,
			id: Some(rank),
		};
	}
}

pub(crate) fn reset_ancient_card(g: &mut Game) {
	let mut suits = [Suit::Spades; 4];
	let mut n = 0;
	for s in Suit::ALL {
		if Some(s) != g.ancient_suit {
			suits[n] = s;
			n += 1;
		}
	}
	let k = g.key_ante("anc");
	let i = g.rng.element(&k, n);
	g.ancient_suit = Some(suits[i]);
}

pub(crate) fn reset_castle_card(g: &mut Game) {
	g.castle_suit = Suit::Spades;
	let v = g.non_stone_sorted();
	if !v.is_empty() {
		let k = g.key_ante("cas");
		(_, g.castle_suit) = v[g.rng.element(&k, v.len())];
	}
}

/// `get_new_boss()`.
pub(crate) fn get_new_boss(g: &mut Game) -> BlindKind {
	let ante = g.ante;
	let a1 = ante.max(1);
	let mut eligible = [false; 30];
	for (i, b) in ALL_BLINDS.iter().enumerate() {
		let d = b.def();
		if !d.boss {
			continue;
		}
		if !d.showdown {
			if d.min_ante <= a1 && (a1 % g.win_ante != 0 || ante < 2) {
				eligible[i] = true;
			}
		} else if ante % g.win_ante == 0 && ante >= 2 {
			eligible[i] = true;
		}
	}
	for (i, e) in eligible.iter_mut().enumerate() {
		if g.banned_blinds >> i & 1 == 1 {
			*e = false;
		}
	}
	let uses = || {
		eligible
			.iter()
			.zip(g.bosses_used)
			.filter(|(e, _)| **e)
			.map(|(_, u)| u)
	};
	let min_use = uses().fold(100, u32::min);
	let mut list = [BlindKind::Small; 30];
	let mut n = 0;
	for (i, (&e, &u)) in eligible.iter().zip(&g.bosses_used).enumerate() {
		if e && u <= min_use {
			list[n] = ALL_BLINDS[i];
			n += 1;
		}
	}
	let k = Key::from("boss");
	let i = g.rng.element(&k, n);
	let boss = list[i];
	g.bosses_used[boss as usize] += 1;
	boss
}

/// `get_next_voucher_key(from_tag)`.
pub(crate) fn get_next_voucher(g: &mut Game, from_tag: bool) -> Voucher {
	let mut pool = [None; 32];
	let mut avail = 0;
	for (i, &v) in ALL_VOUCHERS.iter().enumerate() {
		let c = Center::Voucher(v);
		let mut add = false;
		if !used_blocks(g, c) && !g.voucher_used(v) {
			let mut include = true;
			if let Some(req) = v.def().requires
				&& !g.voucher_used(req)
			{
				include = false;
			}
			if g.shop_vouchers.iter().any(|s| s.kind == Kind::Voucher(v)) {
				include = false;
			}
			add = include;
		}
		if add && !g.banned.contains(c) {
			pool[i] = Some(v);
			avail += 1;
		}
	}
	let key = if from_tag {
		Key::from("Voucher_fromtag")
	} else {
		g.key_ante("Voucher")
	};
	if avail == 0 {
		// A one-entry fallback pool, still sampled.
		g.rng.element(&key, 1);
		return Voucher::Blank;
	}
	let mut i = g.rng.element(&key, 32);
	let mut it = 1;
	loop {
		if let Some(voucher) = pool[i] {
			return voucher;
		}
		it += 1;
		let mut k = key;
		k.push("_resample").push_int(it);
		i = g.rng.element(&k, 32);
	}
}

/// `get_next_tag_key(append)`.
pub(crate) fn get_next_tag(g: &mut Game, append: &str) -> TagKind {
	let mut pool = [None; 24];
	let mut avail = 0;
	for (i, &t) in ALL_TAGS.iter().enumerate() {
		let d = t.def();
		// With a fully discovered profile every `requires` is satisfied.
		let add = d.min_ante == 0 || d.min_ante <= g.ante;
		if add && g.banned_tags >> i & 1 == 0 {
			pool[i] = Some(t);
			avail += 1;
		}
	}
	let mut key = Key::from("Tag");
	key.push(append).push_int(i64::from(g.ante));
	if avail == 0 {
		// A one-entry fallback pool, still sampled.
		g.rng.element(&key, 1);
		return TagKind::Handy;
	}
	let mut i = g.rng.element(&key, 24);
	let mut it = 1;
	loop {
		if let Some(tag) = pool[i] {
			return tag;
		}
		it += 1;
		let mut k = key;
		k.push("_resample").push_int(it);
		i = g.rng.element(&k, 24);
	}
}

/// `get_pack(key, type)`.
pub(crate) fn get_pack(g: &mut Game, base: &str, kind: Option<PackKind>) -> Booster {
	if !g.first_shop_buffoon && !g.banned.contains(Center::Booster(Booster::BuffoonNormal1)) {
		g.first_shop_buffoon = true;
		// `math.random(1, 2)` on the unseeded global generator; both
		// variants are identical apart from art.
		return Booster::BuffoonNormal1;
	}
	let mut cume = 0.0;
	for &b in &ALL_BOOSTERS {
		if (kind.is_none() || kind == Some(b.kind())) && !g.banned.contains(Center::Booster(b)) {
			cume += b.def().weight;
		}
	}
	let k = g.key_ante(base);
	let poll = g.rng.random(&k) * cume;
	let mut it = 0.0;
	for &b in &ALL_BOOSTERS {
		if !g.banned.contains(Center::Booster(b)) {
			if kind.is_none() || kind == Some(b.kind()) {
				it += b.def().weight;
			}
			if it >= poll && it - b.def().weight <= poll {
				return b;
			}
		}
	}
	unreachable!("get_pack found no booster")
}
