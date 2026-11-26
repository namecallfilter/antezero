//! `RLBalatro` observation builder (`OBS_SIZE = 8281` floats, `SEL_F = 44`
//! floats, `HS_CTX = 32` ints), ported directly from `RLBalatro/env/src/env.cpp`.

use super::hand_eval::{CoreCtx, HS_CTX};
use super::ids::{
	CARD_BACK_CENTER_F, N_JOKER_KEYS, N_VOUCHER_KEYS, back_id, blind_id, card_extras, center_id,
	joker_alpha, joker_live, tag_id, voucher_alpha,
};
use crate::card::{AnyCard, Card, Kind, PlayingCard, SuitTest};
use crate::data::{ALL_VOUCHERS, BlindKind, Center, Edition, Enhancement, HandType, Joker, Suit};
use crate::game::{BlindSlot, Game, State};
use crate::num;
use crate::round::blind_amount;

pub const MAX_HAND: usize = 24;
pub const MAX_JOKERS: usize = 20;
pub const MAX_CONS: usize = 10;
pub const MAX_SHOP: usize = 8;
pub const MAX_VOUCH: usize = 4;
pub const MAX_BOOST: usize = 4;
pub const MAX_PACK: usize = 8;
pub const MAX_DECK: usize = 120;
pub const MAX_TAGS: usize = 12;

pub const SCALAR_F: usize = 48;
pub const HAND_LEVEL_F: usize = 5;
pub const CARD_F: usize = 39;
pub const TAG_F: usize = 2;
pub const HAND_TARGET_F: usize = 14;
pub const BOUNTY_F: usize = 19;
pub const TARGET_F: usize = N_JOKER_KEYS + 3;
pub const ROUND_CTX_F: usize = 46 + N_VOUCHER_KEYS;
pub const SEL_F: usize = 8 + 12 + MAX_HAND;

pub const OBS_GROUPS: [(&str, usize, usize); 18] = [
	("scalars", 1, SCALAR_F),
	("hand_levels", 12, HAND_LEVEL_F),
	("hand", MAX_HAND, CARD_F),
	("jokers", MAX_JOKERS, CARD_F),
	("consumables", MAX_CONS, CARD_F),
	("shop", MAX_SHOP, CARD_F),
	("vouchers", MAX_VOUCH, CARD_F),
	("boosters", MAX_BOOST, CARD_F),
	("pack", MAX_PACK, CARD_F),
	("deck", MAX_DECK, CARD_F),
	("tags", MAX_TAGS, TAG_F),
	("counts", 1, 10),
	("blind_tags", 1, 3),
	("stickers", 1, N_JOKER_KEYS),
	("hand_target", 1, HAND_TARGET_F),
	("bounty", 1, BOUNTY_F),
	("target", 1, TARGET_F),
	("round_ctx", 1, ROUND_CTX_F),
];

pub const OBS_SIZE: usize = {
	let mut total = 0;
	let mut i = 0;
	while i < OBS_GROUPS.len() {
		total += OBS_GROUPS[i].1 * OBS_GROUPS[i].2;
		i += 1;
	}
	total
};
const _: () = assert!(OBS_SIZE == 8281);

#[derive(Clone, Copy, PartialEq, Eq)]
enum PlayingZone {
	HandOrShop,
	Deck,
	Pack,
}

#[derive(Clone, Copy)]
struct ObsCtx {
	hide: bool,
	pareidolia: bool,
	smeared: bool,
	idol: bool,
	mail: bool,
	ancient: bool,
	castle: bool,
	idol_id: u8,
	idol_suit: Suit,
	mail_id: u8,
	ancient_suit: Suit,
	castle_suit: Suit,
}

fn obs_ctx(g: &Game, hide: bool) -> ObsCtx {
	let (mut pareidolia, mut smeared, mut idol, mut mail, mut ancient, mut castle) =
		(false, false, false, false, false, false);
	for c in &g.areas.jokers {
		match c.kind {
			Kind::Joker(Joker::Pareidolia) => pareidolia = true,
			Kind::Joker(Joker::Smeared) => smeared = true,
			Kind::Joker(Joker::Idol) => idol = true,
			Kind::Joker(Joker::Mail) => mail = true,
			Kind::Joker(Joker::Ancient) => ancient = true,
			Kind::Joker(Joker::Castle) => castle = true,
			_ => {}
		}
	}
	ObsCtx {
		hide,
		pareidolia,
		smeared,
		idol,
		mail,
		ancient,
		castle,
		idol_id: g.round.idol_card.id.unwrap_or(g.round.idol_card.rank).id(),
		idol_suit: g.round.idol_card.suit,
		mail_id: g.round.mail_rank.id.unwrap_or(g.round.mail_rank.rank).id(),
		ancient_suit: g.round.ancient_suit,
		castle_suit: g.round.castle_suit,
	}
}

fn playing_card_features(c: &PlayingCard, f: &mut [f32], x: &ObsCtx, zone: PlayingZone) {
	let is_deck = zone == PlayingZone::Deck;
	let pack_area = zone == PlayingZone::Pack;
	if x.hide && c.core.face_down && !is_deck {
		f[0] = 1.0;
		f[1] = CARD_BACK_CENTER_F;
		f[7] = 1.0;
		return;
	}
	let cid = center_id(c.center());
	f[0] = 1.0;
	f[1] = f32::from(cid + 1);
	f[2] = f32::from(c.rank.id());
	f[3] = f32::from(c.suit as u8 + 1);
	f[4] = f32::from(Edition::id(c.core.edition));
	f[5] = f32::from(c.seal.map_or(0, |s| s as u8 + 1));
	f[6] = if c.core.debuff && !pack_area {
		1.0
	} else {
		0.0
	};
	f[7] = if c.core.face_down && !is_deck {
		1.0
	} else {
		0.0
	};
	f[9] = if is_deck { 1.0 } else { num::feat(c.core.cost) };
	f[10] = if is_deck {
		1.0
	} else {
		num::feat(c.core.sell_cost)
	};
	f[14] = num::feat(c.enhancement.mult());
	f[15] = num::feat(c.enhancement.x_mult());
	f[16] = num::feat(c.rank.nominal() + c.enhancement.bonus() + c.perma_bonus);
	if c.enhancement == Enhancement::Glass {
		f[17] = 4.0;
	}
	f[19] = num::feat(f64::from(c.times_played));
	let stone = c.is_stone();
	let ranked = !stone || c.vampired;
	let id = if ranked { c.rank.id() } else { 0 };
	f[24] = if !c.core.debuff && (matches!(id, 11..=13) || x.pareidolia) {
		1.0
	} else {
		0.0
	};
	let mut su = [false; 4];
	for (q, s) in Suit::ALL.into_iter().enumerate() {
		su[q] = c.is_suit(s, SuitTest::Scoring, x.smeared);
		f[25 + q] = if su[q] { 1.0 } else { 0.0 };
	}
	f[29] = if ranked { 1.0 } else { 0.0 };
	f[30] = if c.forced_selection { 1.0 } else { 0.0 };
	f[31] = if matches!(id, 14 | 2 | 3 | 5 | 8) {
		1.0
	} else {
		0.0
	};
	f[32] = if (2..=10).contains(&id) && id % 2 == 0 {
		1.0
	} else {
		0.0
	};
	f[33] = if id == 14 || ((3..=9).contains(&id) && id % 2 == 1) {
		1.0
	} else {
		0.0
	};
	f[34] = if (2..=5).contains(&id) { 1.0 } else { 0.0 };
	f[35] = if x.idol && id == x.idol_id && su[x.idol_suit as usize] {
		1.0
	} else {
		0.0
	};
	f[36] = if x.mail && id == x.mail_id { 1.0 } else { 0.0 };
	f[37] = if x.ancient && su[x.ancient_suit as usize] {
		1.0
	} else {
		0.0
	};
	f[38] = if x.castle && su[x.castle_suit as usize] {
		1.0
	} else {
		0.0
	};
}

fn non_playing_card_features(
	g: &Game, c: &Card, f: &mut [f32], x: &ObsCtx, marks: &[u8; N_JOKER_KEYS], n_marked: usize,
) {
	if x.hide && c.core.face_down {
		f[0] = 1.0;
		f[1] = CARD_BACK_CENTER_F;
		f[7] = 1.0;
		return;
	}
	let cid = center_id(c.center());
	f[0] = 1.0;
	f[1] = f32::from(cid + 1);
	f[4] = f32::from(Edition::id(c.core.edition));
	f[6] = if c.core.debuff { 1.0 } else { 0.0 };
	f[7] = if c.core.face_down { 1.0 } else { 0.0 };
	f[9] = num::feat(c.core.cost);
	f[10] = num::feat(c.core.sell_cost);
	f[11] = if c.ab.eternal() { 1.0 } else { 0.0 };
	f[12] = if c.ab.perishable() {
		num::feat(c.ab.perish_tally() + 1.0)
	} else {
		0.0
	};
	f[13] = if c.ab.rental { 1.0 } else { 0.0 };
	f[14] = num::feat(c.ab.mult);
	f[15] = num::feat(c.ab.x_mult);
	let (e0, e1) = card_extras(c);
	f[17] = e0;
	f[18] = e1;
	if let Kind::Joker(j) = c.kind {
		if n_marked > 0 && (marks[joker_alpha(j)] & 1) != 0 {
			f[20] = 1.0;
		}
		match j {
			Joker::Mail => f[2] = f32::from(x.mail_id),
			Joker::Idol => {
				f[2] = f32::from(x.idol_id);
				f[3] = f32::from(x.idol_suit as u8 + 1);
			}
			Joker::Ancient => f[3] = f32::from(x.ancient_suit as u8 + 1),
			Joker::Castle => f[3] = f32::from(x.castle_suit as u8 + 1),
			_ => {}
		}
		let (jv, jc) = joker_live(g, c);
		f[22] = jv;
		f[23] = jc;
	}
}

fn write_scalars(g: &Game, s: &mut [f32]) {
	const STATES: [State; 10] = [
		State::BlindSelect,
		State::SelectingHand,
		State::RoundEval,
		State::Shop,
		State::TarotPack,
		State::PlanetPack,
		State::SpectralPack,
		State::StandardPack,
		State::BuffoonPack,
		State::GameOver,
	];
	for (i, &st) in STATES.iter().enumerate() {
		s[i] = if g.flow.state == st { 1.0 } else { 0.0 };
	}
	s[10] = num::feat(f64::from(g.run.ante));
	s[11] = num::feat(f64::from(g.run.round));
	s[12] = num::feat(g.run.dollars);
	s[13] = num::feat(g.round.hands_left);
	s[14] = num::feat(g.round.discards_left);
	s[15] = num::feat(g.round.chips);
	s[16] = num::feat(g.blind.chips);
	s[17] = g.blind.kind.map_or(0.0, |k| f32::from(blind_id(k) + 1));
	s[18] = if g.blind.is_boss() { 1.0 } else { 0.0 };
	s[19] = if g.blind.disabled { 1.0 } else { 0.0 };
	s[20] = num::feat(g.areas.hand.limit.limit);
	s[21] = num::feat(g.areas.limits.jokers.limit);
	s[22] = num::feat(g.areas.limits.consumables.limit);
	s[23] = num::feat(g.shop.reroll_cost);
	s[24] = num::feat(g.run.interest_cap);
	s[25] = 0.0;
	s[26] = g
		.blind_select
		.on_deck
		.map_or(0.0, |b| f32::from(b as u8 + 1));
	s[27] = f32::from(blind_id(g.blind_select.boss) + 1);
	s[28] = num::feat(num::float(g.areas.deck.len()));
	s[29] = num::feat(num::float(g.areas.playing_card_count()));
	s[30] = g.pack.choices.map_or(0.0, f32::from);
	s[31] = num::feat(g.run.bankrupt_at);
	s[32] = 0.0;
	s[33] = num::feat(g.run.skips);
	s[34] = num::feat(g.round.hands_played);
	s[35] = num::feat(g.round.discards_used);
	s[36] = num::feat(g.run.probabilities);
	s[37] = f32::from(u8::from(g.setup.stake));
	s[38] = f32::from(back_id(g.setup.back) + 1);
	s[39] = num::feat(g.shop.free_rerolls);
	s[40] = num::feat(g.shop.discount_percent);
	s[41] = num::feat(g.run.interest_amount);
	s[42] = f32::from(g.round.most_played_hand as u8);
	s[43] = g
		.shop
		.voucher
		.map_or(0.0, |v| f32::from(center_id(Center::Voucher(v)) + 1));
	s[44] = num::feat(g.run.hands_played);
	s[45] = if g.run.won { 1.0 } else { 0.0 };
	s[46] = 0.0;
	s[47] = num::feat(g.blind.mult);
}

fn write_round_ctx(g: &Game, last_play_blind: f64, p: &mut [f32]) {
	if let Some(lh) = g.history.last_hand_played {
		let score = g.run.last_hand_score.floor();
		p[0] = num::feat(score.max(0.0).ln_1p());
		p[1] = num::feat(g.run.last_hand_chips.max(0.0).ln_1p());
		p[2] = num::feat(g.run.last_hand_mult.max(0.0).ln_1p());
		let den = if g.blind.chips > 0.0 {
			g.blind.chips
		} else {
			last_play_blind
		};
		if den > 0.0 {
			p[3] = num::feat(score / den);
		}
		p[4 + lh as usize] = 1.0;
	}
	let eye = g.blind.active(BlindKind::Eye);
	let mouth = g.blind.active(BlindKind::Mouth);
	for h in HandType::ALL {
		let k = h as usize;
		p[16 + k] = num::feat(g.hands[h].played_this_round);
		let forbid = (eye && g.blind.eye_hands.contains(h))
			|| (mouth && g.blind.only_hand.is_some_and(|only| only != h));
		p[28 + k] = if forbid { 1.0 } else { 0.0 };
	}
	let base = blind_amount(g.run.ante, g.setup.modifiers.scaling);
	let scale = g.setup.starting_params.ante_scaling;
	for (t, slot) in BlindSlot::ALL.into_iter().enumerate() {
		let bk = g.blind_select.kind(slot);
		let bd = bk.def();
		p[40 + t] = num::feat((base * bd.mult * scale).max(0.0).ln_1p());
		p[43 + t] = if g.setup.modifiers.no_blind_reward[slot] {
			0.0
		} else {
			num::feat(f64::from(bd.dollars))
		};
	}
	for v in ALL_VOUCHERS {
		if g.history.used_vouchers.contains(v) {
			p[46 + voucher_alpha(v)] = 1.0;
		}
	}
}

pub fn write_obs(
	g: &mut Game, hide_face_down: bool, last_play_blind: f64, marks: &[u8; N_JOKER_KEYS],
	n_marked: usize, out: &mut [f32],
) {
	g.update_derived();
	out[..OBS_SIZE].fill(0.0);
	let mut off = 0usize;
	write_scalars(g, &mut out[off..off + SCALAR_F]);
	off += SCALAR_F;

	let ox = obs_ctx(g, hide_face_down);
	let mut ox_open = ox;
	ox_open.hide = false;

	for h in HandType::ALL {
		let hl = &g.hands[h];
		let p = &mut out[off..off + HAND_LEVEL_F];
		p[0] = num::feat(hl.level);
		p[1] = num::feat(hl.chips);
		p[2] = num::feat(hl.mult);
		p[3] = num::feat(hl.played);
		p[4] = if hl.visible { 1.0 } else { 0.0 };
		off += HAND_LEVEL_F;
	}

	let mut counts = [0usize; 10];
	off += write_active_areas(
		g,
		&ox,
		&ox_open,
		marks,
		n_marked,
		&mut counts,
		&mut out[off..],
	);
	write_deck_and_tail(
		g,
		&ox_open,
		last_play_blind,
		marks,
		n_marked,
		&mut counts,
		&mut out[off..OBS_SIZE],
	);
}

fn write_active_areas(
	g: &Game, ox: &ObsCtx, ox_open: &ObsCtx, marks: &[u8; N_JOKER_KEYS], n_marked: usize,
	counts: &mut [usize; 10], out: &mut [f32],
) -> usize {
	let mut off = 0usize;
	counts[0] = g.areas.hand.cards.len().min(MAX_HAND);
	for (i, c) in g.areas.hand.cards.iter().take(counts[0]).enumerate() {
		playing_card_features(
			c,
			&mut out[off + i * CARD_F..off + (i + 1) * CARD_F],
			ox,
			PlayingZone::HandOrShop,
		);
	}
	off += MAX_HAND * CARD_F;

	counts[1] = g.areas.jokers.len().min(MAX_JOKERS);
	for (i, c) in g.areas.jokers.iter().take(counts[1]).enumerate() {
		non_playing_card_features(
			g,
			c,
			&mut out[off + i * CARD_F..off + (i + 1) * CARD_F],
			ox,
			marks,
			n_marked,
		);
	}
	off += MAX_JOKERS * CARD_F;

	counts[2] = g.areas.consumables.len().min(MAX_CONS);
	for (i, c) in g.areas.consumables.iter().take(counts[2]).enumerate() {
		non_playing_card_features(
			g,
			c,
			&mut out[off + i * CARD_F..off + (i + 1) * CARD_F],
			ox_open,
			marks,
			n_marked,
		);
	}
	off += MAX_CONS * CARD_F;

	counts[3] = if g.shop.open {
		g.areas.shop_jokers.len().min(MAX_SHOP)
	} else {
		0
	};
	for (i, c) in g.areas.shop_jokers.iter().take(counts[3]).enumerate() {
		let slice = &mut out[off + i * CARD_F..off + (i + 1) * CARD_F];
		match c {
			AnyCard::Playing(pc) => {
				playing_card_features(pc, slice, ox_open, PlayingZone::HandOrShop);
			}
			AnyCard::Card(cc) => non_playing_card_features(g, cc, slice, ox_open, marks, n_marked),
		}
	}
	off += MAX_SHOP * CARD_F;

	counts[4] = if g.shop.open {
		g.areas.shop_vouchers.len().min(MAX_VOUCH)
	} else {
		0
	};
	for (i, c) in g.areas.shop_vouchers.iter().take(counts[4]).enumerate() {
		non_playing_card_features(
			g,
			c,
			&mut out[off + i * CARD_F..off + (i + 1) * CARD_F],
			ox_open,
			marks,
			n_marked,
		);
	}
	off += MAX_VOUCH * CARD_F;

	counts[5] = if g.shop.open {
		g.areas.shop_boosters.len().min(MAX_BOOST)
	} else {
		0
	};
	for (i, c) in g.areas.shop_boosters.iter().take(counts[5]).enumerate() {
		non_playing_card_features(
			g,
			c,
			&mut out[off + i * CARD_F..off + (i + 1) * CARD_F],
			ox_open,
			marks,
			n_marked,
		);
	}
	off += MAX_BOOST * CARD_F;

	counts[6] = if g.pack.open.is_some() {
		g.areas.pack_cards.len().min(MAX_PACK)
	} else {
		0
	};
	for (i, c) in g.areas.pack_cards.iter().take(counts[6]).enumerate() {
		let slice = &mut out[off + i * CARD_F..off + (i + 1) * CARD_F];
		match c {
			AnyCard::Playing(pc) => playing_card_features(pc, slice, ox_open, PlayingZone::Pack),
			AnyCard::Card(cc) => non_playing_card_features(g, cc, slice, ox_open, marks, n_marked),
		}
	}
	off + MAX_PACK * CARD_F
}

fn write_deck_and_tail(
	g: &Game, ox_open: &ObsCtx, last_play_blind: f64, marks: &[u8; N_JOKER_KEYS], n_marked: usize,
	counts: &mut [usize; 10], out: &mut [f32],
) {
	let mut off = 0usize;
	let mut deck_order: Vec<(u32, usize)> = g
		.areas
		.deck
		.iter()
		.enumerate()
		.map(|(idx, c)| {
			let key = (u32::from(c.rank.id() + 1) << 24)
				| (u32::from(c.suit as u8 + 1) << 16)
				| u32::from(center_id(c.center()) + 1);
			(key, idx)
		})
		.collect();
	deck_order.sort_by_key(|&(k, _)| k);
	let nd = deck_order.len().min(MAX_DECK);
	counts[7] = nd;
	for (i, &(_, idx)) in deck_order.iter().take(nd).enumerate() {
		playing_card_features(
			&g.areas.deck[idx],
			&mut out[off + i * CARD_F..off + (i + 1) * CARD_F],
			ox_open,
			PlayingZone::Deck,
		);
	}
	off += MAX_DECK * CARD_F;

	let nt = g.tags.held.len().min(MAX_TAGS);
	counts[8] = nt;
	for (i, tg) in g.tags.held.iter().take(nt).enumerate() {
		out[off + i * TAG_F] = f32::from(tag_id(tg.kind) + 1);
		out[off + i * TAG_F + 1] = tg.orbital_hand.map_or(0.0, |h| f32::from(h as u8 + 1));
	}
	off += MAX_TAGS * TAG_F;

	counts[9] = g.areas.discard.len();
	for (i, &c) in counts.iter().enumerate() {
		out[off + i] = num::feat(num::float(c));
	}
	off += 10;

	out[off] = g
		.blind_select
		.skip_tags
		.small
		.map_or(0.0, |t| f32::from(tag_id(t) + 1));
	out[off + 1] = g
		.blind_select
		.skip_tags
		.big
		.map_or(0.0, |t| f32::from(tag_id(t) + 1));
	out[off + 2] = f32::from(blind_id(g.blind_select.boss) + 1);
	off += 3;

	if n_marked > 0 {
		for (k, &m) in marks.iter().enumerate() {
			out[off + k] = if (m & 1) != 0 { 1.0 } else { 0.0 };
		}
	}
	off += N_JOKER_KEYS + HAND_TARGET_F + BOUNTY_F + TARGET_F;
	write_round_ctx(g, last_play_blind, &mut out[off..off + ROUND_CTX_F]);
}

pub fn write_obs_selection(g: &Game, out: &mut [f32]) {
	out[..SEL_F].fill(0.0);
	out[7] = num::feat((g.blind.chips - g.round.chips).max(0.0));
}

pub fn write_hand_ctx(g: &Game, core: &CoreCtx, out: &mut [i32]) {
	out[..HS_CTX].fill(0);
	if g.flow.state != State::SelectingHand || g.is_terminal() {
		return;
	}
	for (i, slot) in out.iter_mut().enumerate().take(core.nh) {
		let rank = if core.rank[i] >= 0 {
			i32::from(core.rank[i]) + 2
		} else {
			0
		};
		let stone = i32::from(core.rank[i] < 0);
		let forced = i32::try_from((core.forced >> i) & 1).unwrap_or(0);
		*slot = rank | (i32::from(core.suit[i]) << 4) | (stone << 8) | (forced << 9) | (1 << 10);
	}
	out[24] = num::id(core.nh);
	out[25] = i32::from(core.ff) | (i32::from(core.sc) << 1) | (i32::from(core.block) << 3);
	out[28] = num::id(core.lo);
	out[29] = num::id(core.hi);
}
