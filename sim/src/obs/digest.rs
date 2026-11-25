//! Keys and digests of what a player can see.

use std::hash::{Hash, Hasher};

use crate::card::{Ability, CanonKey, Card, CardCore, CardFace, Kind, PlayingCard, canon_key};
use crate::data::{Edition, Enhancement, Joker, Suit, Voucher};
use crate::game::{Areas, Blind, BlindSlot, BlindStatus, Game, HandLevels, HandSort, PerBlind};
use crate::num;

/// SipHash-1-3 with a zero key over the bytes written: what
/// `DefaultHasher::new()` computes today. The digests seed the previews'
/// sampled hands and order their jokers, which reach the observation, and
/// std leaves `DefaultHasher`'s algorithm unspecified, so it is fixed here.
#[derive(Clone, Debug)]
pub(super) struct SipHasher13 {
	v: [u64; 4],
	/// Bytes written but not yet compressed, little-endian.
	tail: u64,
	tail_len: usize,
	/// Bytes written in all.
	len: usize,
}

impl SipHasher13 {
	pub(super) fn new() -> SipHasher13 {
		SipHasher13 {
			v: [
				0x736f_6d65_7073_6575,
				0x646f_7261_6e64_6f6d,
				0x6c79_6765_6e65_7261,
				0x7465_6462_7974_6573,
			],
			tail: 0,
			tail_len: 0,
			len: 0,
		}
	}

	fn round(&mut self) {
		let [v0, v1, v2, v3] = &mut self.v;
		*v0 = v0.wrapping_add(*v1);
		*v1 = v1.rotate_left(13) ^ *v0;
		*v0 = v0.rotate_left(32);
		*v2 = v2.wrapping_add(*v3);
		*v3 = v3.rotate_left(16) ^ *v2;
		*v0 = v0.wrapping_add(*v3);
		*v3 = v3.rotate_left(21) ^ *v0;
		*v2 = v2.wrapping_add(*v1);
		*v1 = v1.rotate_left(17) ^ *v2;
		*v2 = v2.rotate_left(32);
	}

	fn compress(&mut self, m: u64) {
		self.v[3] ^= m;
		self.round();
		self.v[0] ^= m;
	}

	fn push_byte(&mut self, b: u8) {
		self.tail |= u64::from(b) << (8 * self.tail_len);
		self.tail_len += 1;
		if self.tail_len == 8 {
			self.compress(self.tail);
			self.tail = 0;
			self.tail_len = 0;
		}
	}
}

impl Hasher for SipHasher13 {
	fn write(&mut self, bytes: &[u8]) {
		self.len += bytes.len();
		let fill = bytes.len().min((8 - self.tail_len) % 8);
		let (head, rest) = bytes.split_at(fill);
		for &b in head {
			self.push_byte(b);
		}
		let (words, remainder) = rest.as_chunks::<8>();
		for &word in words {
			self.compress(u64::from_le_bytes(word));
		}
		for &b in remainder {
			self.push_byte(b);
		}
	}

	fn finish(&self) -> u64 {
		let mut s = self.clone();
		let last = (num::index64(self.len) & 0xff) << 56 | self.tail;
		s.compress(last);
		s.v[2] ^= 0xff;
		for _ in 0..3 {
			s.round();
		}
		s.v[0] ^ s.v[1] ^ s.v[2] ^ s.v[3]
	}
}

/// Which blinds of the ante are done (defeated or skipped). Unlike the raw
/// states, this does not change between a shop and the next blind select.
pub(super) fn blinds_done(states: PerBlind<BlindStatus>) -> [bool; BlindSlot::ALL.len()] {
	BlindSlot::ALL.map(|slot| matches!(states[slot], BlindStatus::Defeated | BlindStatus::Skipped))
}

/// [`crate::card::canon_key`]'s values for a card that is not a playing card: its
/// edition, and constants for the rest (Spades, rank id 0, no enhancement,
/// no seal, no bonus, never played).
fn card_canon_key(c: &Card) -> CanonKey {
	CanonKey {
		face: CardFace {
			suit: Suit::Spades as u8,
			rank: 0,
			enhancement: Enhancement::Base as u8,
			edition: Edition::id(c.core.edition),
			seal: 0,
			perma_bonus: 0f64.to_bits(),
		},
		played_this_ante: false,
		times_played: 0,
	}
}

/// Hashes the visible state of a card that can affect scoring: `card` is the
/// center and ability of a card that is not a playing card (`None` for a
/// playing card, which hashes the values of an empty ability).
fn hash_parts(card: Option<(Kind, &Ability)>, key: CanonKey, core: &CardCore, h: &mut impl Hasher) {
	// Playing cards are numbered 0 and the kinds from 1, written the way
	// derived `Hash` writes an enum (the variant's index as an `isize`, then
	// its payload). The values reach the observation: `card_hash` orders the
	// jokers of a virtual round.
	let blank;
	let a = match card {
		None => {
			0isize.hash(h);
			blank = Ability::default();
			&blank
		}
		Some((kind, a)) => {
			match kind {
				Kind::Joker(j) => (1isize, j).hash(h),
				Kind::Tarot(t) => (2isize, t).hash(h),
				Kind::Planet(p) => (3isize, p).hash(h),
				Kind::Spectral(s) => (4isize, s).hash(h),
				Kind::Voucher(v) => (5isize, v).hash(h),
				Kind::Booster(b) => (6isize, b).hash(h),
			}
			a
		}
	};
	key.hash(h);
	(
		core.debuff,
		core.face_down,
		core.sell_cost.to_bits(),
		core.cost.to_bits(),
	)
		.hash(h);
	for x in [
		a.mult,
		a.x_mult,
		a.extra_value,
		a.chips,
		a.dollars,
		a.extra,
		a.invis_rounds,
		a.yorick_discards,
		a.caino_xmult,
		a.loyalty_remaining,
		core.hands_played_at_create,
		a.perish_tally(),
		a.money,
	] {
		x.to_bits().hash(h);
	}
	(a.to_do, a.eternal(), a.perishable(), a.rental).hash(h);
}

fn hash_card(c: &Card, h: &mut impl Hasher) {
	hash_parts(Some((c.kind, &c.ab)), card_canon_key(c), &c.core, h);
}

pub(super) fn card_hash(c: &Card) -> u64 {
	let mut h = SipHasher13::new();
	hash_card(c, &mut h);
	h.finish()
}

fn playing_card_hash(c: &PlayingCard) -> u64 {
	let mut h = SipHasher13::new();
	hash_parts(None, canon_key(c), &c.core, &mut h);
	h.finish()
}

/// The sum of `hashes`, whatever their order.
fn multiset_hash(hashes: impl Iterator<Item = u64>) -> u64 {
	hashes.fold(0u64, u64::wrapping_add)
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
	let mut h = SipHasher13::new();
	let money = g.areas.jokers.iter().any(reads_money);
	let cons = g.history.used_vouchers.contains(Voucher::Observatory)
		|| g.areas
			.shop_vouchers
			.iter()
			.any(|c| c.kind == Kind::Voucher(Voucher::Observatory))
		|| g.areas.consumables.iter().any(|c| {
			matches!(
				c.core.edition,
				Some(Edition::Foil | Edition::Holo | Edition::Polychrome)
			)
		});
	(
		g.run.ante,
		g.run.round,
		g.blind_select.boss,
		blinds_done(g.blind_select.states),
		g.areas.hand.order == HandSort::Suit,
		g.setup.steamodded,
	)
		.hash(&mut h);
	(
		g.areas.hand.limit.limit.to_bits(),
		g.areas.hand.limit.real.to_bits(),
		g.round.base_hands.to_bits(),
		g.round.base_discards.to_bits(),
		g.round.bonus_next_hands.to_bits(),
		g.round.bonus_discards.to_bits(),
	)
		.hash(&mut h);
	(
		g.run.probabilities.to_bits(),
		money.then(|| g.run.dollars.to_bits()),
		g.history.used_vouchers,
		g.areas.limits.jokers.limit.to_bits(),
		g.run.skips.to_bits(),
		g.run.bankrupt_at.to_bits(),
		g.history.consumable_usage_tarot.to_bits(),
	)
		.hash(&mut h);
	(
		g.round.idol_card.rank.id(),
		g.round.idol_card.suit,
		g.round.mail_rank.rank.id(),
		// Hashed as an `Option<Suit>`: the layout the trained models' build
		// previews were seeded with.
		Some(g.round.ancient_suit),
		g.round.castle_suit,
		g.round.most_played_hand,
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
	g.areas.jokers.len().hash(&mut h);
	multiset_hash(g.areas.jokers.iter().map(card_hash)).hash(&mut h);
	if cons {
		for c in &g.areas.consumables {
			hash_card(c, &mut h);
		}
	}
	multiset_hash(g.areas.playing_cards().map(playing_card_hash)).hash(&mut h);
	h.finish()
}

pub(super) fn reads_money(c: &Card) -> bool {
	matches!(c.kind, Kind::Joker(Joker::Bull | Joker::Bootstraps))
}

/// What `rank_plays` reads: the hand (including debuffs), hand levels, eval
/// flags, Splash and the blind's hand debuffs (see `rough_score`).
pub(super) fn hand_digest(areas: &Areas, blind: &Blind, hands: &HandLevels) -> u64 {
	let mut h = SipHasher13::new();
	areas.hand.cards.len().hash(&mut h);
	for c in &areas.hand.cards {
		canon_key(c).hash(&mut h);
		(c.core.debuff, c.forced_selection).hash(&mut h);
	}
	for l in hands {
		(l.chips.to_bits(), l.mult.to_bits()).hash(&mut h);
	}
	let f = areas.eval_flags();
	(
		f.four_fingers,
		f.shortcut,
		f.smeared,
		areas.has_joker(Joker::Splash),
		areas.owns_joker(Joker::Blackboard),
		areas.owns_joker(Joker::RaisedFist),
	)
		.hash(&mut h);
	(blind.disabled, blind.kind, blind.eye_hands, blind.only_hand).hash(&mut h);
	h.finish()
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn sip_hasher_matches_siphash_1_3() {
		// Values of `DefaultHasher::new()` (SipHash-1-3, zero key) on Rust 1.98.
		let digest = |f: &dyn Fn(&mut SipHasher13)| {
			let mut h = SipHasher13::new();
			f(&mut h);
			h.finish()
		};
		assert_eq!(digest(&|_| {}), 0xd1fb_a762_150c_532c);
		assert_eq!(
			digest(&|h| h.write(b"0123456789abcdefg")),
			0x3323_a4f8_b8d9_776b
		);
		assert_eq!(
			digest(&|h| (
				3u8,
				0x0123_4567_89ab_cdefu64,
				true,
				[1u32, 2, 3],
				Some(7i32)
			)
				.hash(h)),
			0xd79d_3e57_9866_57c7
		);
		assert_eq!(
			digest(&|h| {
				for i in 0..100u64 {
					h.write_u64(i * 0x9e37_79b9);
					h.write_u8(u8::try_from(i).unwrap());
				}
			}),
			0x58c8_4a74_0d7d_c5cb
		);
	}
}
