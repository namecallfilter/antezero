//! Keys and digests of what a player can see: the canonical order of cards,
//! the cache keys of the build previews and the seeded shuffles of their
//! sampled worlds.

use std::hash::{Hash, Hasher};

use super::candidates::has_kind;
use crate::card::{Card, Kind};
use crate::data::{Edition, Joker, Voucher};
use crate::game::{BlindStatus, Game};
use crate::num;

pub(super) fn mix(mut z: u64) -> u64 {
	z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
	z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
	z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
	z ^ (z >> 31)
}

/// Fisher-Yates with splitmix64 (the shuffle the candidate worlds use).
pub(super) fn splitmix_shuffle<T>(v: &mut [T], mut seed: u64) {
	for i in (1..v.len()).rev() {
		seed = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
		let mut z = seed;
		z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
		z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
		v.swap(i, num::pick(z ^ (z >> 31), i + 1));
	}
}

/// Which blinds of the ante are done (defeated or skipped). Unlike the raw
/// states, this does not change between a shop and the next blind select.
pub(super) fn blinds_done(g: &Game) -> [bool; 3] {
	g.blind_states
		.map(|s| matches!(s, BlindStatus::Defeated | BlindStatus::Skipped))
}

/// Visible content of a playing card (never its id): the sort key of the
/// canonical pool order.
pub(crate) fn canon_key(c: &Card) -> (u8, u8, u8, u8, u8, u64, bool, u32) {
	(
		c.suit as u8,
		c.rank,
		c.enhancement as u8,
		c.edition as u8,
		c.seal.map_or(0, |s| s as u8 + 1),
		c.ab.perma_bonus.to_bits(),
		c.ab.played_this_ante,
		c.times_played,
	)
}

/// Hashes the visible state of a card that can affect scoring.
pub(super) fn hash_card(c: &Card, h: &mut impl Hasher) {
	c.kind.hash(h);
	canon_key(c).hash(h);
	(
		c.debuff,
		c.face_down,
		c.sell_cost.to_bits(),
		c.cost.to_bits(),
	)
		.hash(h);
	let a = &c.ab;
	for x in [
		a.mult,
		a.x_mult,
		a.extra_value,
		a.chips,
		a.dollars,
		a.remaining,
		a.invis_rounds,
		a.yorick_discards,
		a.caino_xmult,
		a.loyalty_remaining,
		a.hands_played_at_create,
		a.perish_tally,
		a.money,
	] {
		x.to_bits().hash(h);
	}
	(a.to_do, a.eternal, a.perishable, a.rental).hash(h);
}

pub(super) fn card_hash(c: &Card) -> u64 {
	let mut h = std::hash::DefaultHasher::new();
	hash_card(c, &mut h);
	h.finish()
}

/// Order-free digest of a set of cards (a sum of card hashes).
pub(super) fn multiset_hash<'a>(cards: impl Iterator<Item = &'a Card>) -> u64 {
	cards.fold(0u64, |s, c| s.wrapping_add(card_hash(c)))
}

/// A digest of everything the previews read (so a cached preview can be
/// reused while a shop visit does not change the build). Joker order is
/// left out: the previews use the canonical order. Money and the held
/// consumables are left out while no preview can read them, so rerolls, pack
/// openings and held tarots keep the cached preview: money scores only
/// through Bull and Bootstraps (by kind: the previewed blind clears joker
/// debuffs; as items they are keyed in `item_feats`), consumables only
/// through their editions and Observatory (owned, or previewed as the shop's
/// voucher).
pub(super) fn state_key(g: &Game) -> u64 {
	let mut h = std::hash::DefaultHasher::new();
	let money = g.jokers.iter().any(reads_money);
	let cons = g.voucher_used(Voucher::Observatory)
		|| g.shop_vouchers
			.iter()
			.any(|c| c.kind == Kind::Voucher(Voucher::Observatory))
		|| g.consumables.iter().any(|c| {
			matches!(
				c.edition,
				Edition::Foil | Edition::Holo | Edition::Polychrome
			)
		});
	(
		g.ante,
		g.round,
		g.boss_choice,
		blinds_done(g),
		g.hand_sort_by_suit,
		g.steamodded,
	)
		.hash(&mut h);
	(
		g.hand_limit.limit.to_bits(),
		g.hand_limit.real.to_bits(),
		g.rr_hands.to_bits(),
		g.rr_discards.to_bits(),
		g.round_bonus_next_hands.to_bits(),
		g.round_bonus_discards.to_bits(),
	)
		.hash(&mut h);
	(
		g.probabilities.to_bits(),
		money.then(|| g.dollars.to_bits()),
		g.used_vouchers,
		g.joker_limit.limit.to_bits(),
		g.skips.to_bits(),
		g.bankrupt_at.to_bits(),
		g.consumable_usage_tarot.to_bits(),
	)
		.hash(&mut h);
	(
		g.idol_card.rank,
		g.idol_card.suit,
		g.mail_rank.rank,
		g.ancient_suit,
		g.castle_suit,
		g.most_played_hand,
	)
		.hash(&mut h);
	for l in &g.hands {
		(
			l.level.to_bits(),
			l.chips.to_bits(),
			l.mult.to_bits(),
			l.played.to_bits(),
			l.visible,
		)
			.hash(&mut h);
	}
	g.jokers.len().hash(&mut h);
	multiset_hash(g.jokers.iter()).hash(&mut h);
	if cons {
		for c in &g.consumables {
			hash_card(c, &mut h);
		}
	}
	multiset_hash(g.playing_cards()).hash(&mut h);
	h.finish()
}

/// The jokers whose scoring reads the money held.
pub(super) fn reads_money(c: &Card) -> bool {
	matches!(c.kind, Kind::Joker(Joker::Bull | Joker::Bootstraps))
}

/// What `rank_plays` reads: the hand (including debuffs), hand levels, eval
/// flags, Splash and the blind's hand debuffs (see `rough_score`).
pub(super) fn hand_digest(v: &Game) -> u64 {
	let mut h = std::hash::DefaultHasher::new();
	v.hand.len().hash(&mut h);
	for c in &v.hand {
		canon_key(c).hash(&mut h);
		(c.debuff, c.ab.forced_selection).hash(&mut h);
	}
	for l in &v.hands {
		(l.chips.to_bits(), l.mult.to_bits()).hash(&mut h);
	}
	let f = v.eval_flags();
	(
		f.four_fingers,
		f.shortcut,
		f.smeared,
		v.has_joker(Joker::Splash),
		has_kind(v, Joker::Blackboard),
		has_kind(v, Joker::RaisedFist),
	)
		.hash(&mut h);
	(
		v.blind.disabled,
		v.blind.kind,
		v.blind.eye_hands,
		v.blind.only_hand,
	)
		.hash(&mut h);
	h.finish()
}
