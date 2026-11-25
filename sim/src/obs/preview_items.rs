//! Cached build previews and per-item deltas (buying, picking, redeeming, or
//! selling an item on the same sampled hands as the base build).

use std::hash::{Hash, Hasher};
use std::{array, iter};

use super::digest::{SipHasher13, blinds_done, card_hash, hand_digest, reads_money, state_key};
use super::layout::PREVIEW_TOK;
use super::preview::{
	Capacity, PREVIEW_HANDS, PREVIEW_LOG_MAX, Sample, best_of, capacity, capacity_samples,
	clamped_feat, rank_plays, settle, virtual_round,
};
use super::{ItemZone, Zone};
use crate::card::{AnyCard, Card, Kind};
use crate::data::{BlindKind, HandType, Joker, Spectral};
use crate::game::{Areas, BlindSelect, BlindSlot, BlindStatus, Game, Phase, Scratch};
use crate::joker::LifecycleCtx;
use crate::shop::VoucherSource;
use crate::{card, joker, num, rng, shop};

/// Recent build previews kept per env (a shop visit returns to the same
/// build after joker moves, rerolls and packs).
pub const PREVIEW_CACHE: usize = 4;
/// Hands (of `PREVIEW_HANDS`) an item is previewed on (the first ones; the
/// next blind is previewed on as many).
pub const PREVIEW_ITEM_HANDS: usize = 1;

fn next_blind(blind_select: &BlindSelect) -> BlindKind {
	BlindSlot::ALL
		.into_iter()
		.find(|&slot| {
			!matches!(
				blind_select.states[slot],
				BlindStatus::Defeated | BlindStatus::Skipped
			)
		})
		.map_or(blind_select.boss, |slot| blind_select.kind(slot))
}

/// An item's build preview: the build with the item acquired (a joker:
/// sold) against the boss, on the same hands as without it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ItemPreview {
	/// Log of the capacity with the item over without, within
	/// [`PREVIEW_LOG_MAX`].
	pub gain: f32,
	/// The capacity ratio with the item ([`Capacity::ratio_feat`]).
	pub ratio: f32,
	/// The clear fraction with the item.
	pub p_clear: f32,
}

impl ItemPreview {
	/// Its token columns: gain, ratio, clear fraction and a 1 marking a
	/// previewed item (an item without a preview has zeros there).
	#[must_use]
	pub fn to_columns(self) -> [f32; PREVIEW_TOK] {
		[self.gain, self.ratio, self.p_clear, 1.0]
	}

	/// The preview a token's preview columns hold, or `None` if they do not
	/// mark one (see [`ItemPreview::to_columns`]).
	#[must_use]
	pub fn from_columns(columns: [f32; PREVIEW_TOK]) -> Option<ItemPreview> {
		let [gain, ratio, p_clear, mark] = columns;
		(mark == 1.0).then_some(ItemPreview {
			gain,
			ratio,
			p_clear,
		})
	}
}

/// The capacity of a build against this ante's boss and against the next
/// blind to play (the same blind when the boss is next), plus what is
/// needed to preview items on the same sampled hands and a cache of the
/// item previews already computed for this build.
#[derive(Clone, Debug, PartialEq)]
pub struct BuildPreview {
	key: u64,
	seeds: [u64; PREVIEW_HANDS],
	pub boss: BlindKind,
	pub next: BlindKind,
	pub cap_boss: Capacity,
	pub cap_next: Capacity,
	samples: Vec<Sample>,
	/// What a hand is worth in a round against the boss over its best play
	/// alone (below 1 against The Eye and The Mouth): item previews, made on
	/// single hands, are put on the same scale.
	round_share: f64,
	items: Vec<(u64, Option<ItemPreview>)>,
}

/// Makes the preview of `g` the front of `cache` (this env's recent
/// previews): a cached preview of the same build is moved there and reused
/// (with its item previews), otherwise a new one is computed and inserted
/// (the cache keeps `PREVIEW_CACHE` entries). Returns that preview, or
/// `None` outside shops, packs and blind select (nothing is previewed
/// there).
pub fn build_preview<'cache>(
	g: &Game, cache: &'cache mut Vec<BuildPreview>,
) -> Option<&'cache mut BuildPreview> {
	if !matches!(g.flow.phase, Phase::Shop | Phase::Pack | Phase::BlindSelect) {
		return None;
	}
	let key = state_key(g);
	if let Some(i) = cache.iter().position(|p| p.key == key) {
		cache[..=i].rotate_right(1);
		return cache.first_mut();
	}
	// The boss and the hands played so far (both visible, both fixed while a
	// shop is open) make the sampled hands differ between games: seeded from
	// the ante and round alone, every game with the same cards previewed its
	// items on the same hand.
	let base = {
		let mut h = SipHasher13::new();
		(
			g.run.ante,
			g.run.round,
			blinds_done(g.blind_select.states),
			g.blind_select.boss,
		)
			.hash(&mut h);
		for l in &g.hands {
			l.played.to_bits().hash(&mut h);
		}
		h.finish()
	};
	let seeds: [u64; PREVIEW_HANDS] =
		array::from_fn(|k| rng::splitmix64(base ^ (num::index64(k) + 1)));
	let boss = g.blind_select.boss;
	let next = next_blind(&g.blind_select);
	let (cap_boss, samples) = capacity_samples(g, boss, &seeds);
	let cap_next = if next == boss {
		cap_boss
	} else {
		capacity(g, next, &seeds[..PREVIEW_ITEM_HANDS])
	};
	let free = samples.iter().map(|s| s.best).sum::<f64>() / num::float(samples.len());
	let round_share = if free > 0.0 {
		cap_boss.mean / free
	} else {
		1.0
	};
	cache.insert(0, BuildPreview {
		key,
		seeds,
		boss,
		next,
		cap_boss,
		cap_next,
		samples,
		round_share,
		items: Vec::new(),
	});
	cache.truncate(PREVIEW_CACHE);
	cache.first_mut()
}

/// The reaction of every `j` owned to what a preview just did: the growth
/// that comes with using a planet (Constellation) or selling a card (Campfire).
fn react(v: &mut Game, j: Joker, ctx: LifecycleCtx) {
	for i in 0..v.areas.jokers.len() {
		if v.areas.jokers[i].kind == Kind::Joker(j) {
			joker::calc_joker(v, i, ctx);
		}
	}
}

/// Adds a bought, picked or redeemed card to the build: jokers join the
/// joker area, planets and Black Hole are used, vouchers are redeemed. Other
/// consumables are not previewed (`None`).
fn acquire(v: &mut Game, mut c: Card) -> Option<()> {
	match c.kind {
		Kind::Joker(_) => {
			card::add_to_deck(v, &mut c);
			v.areas.jokers.push(c);
		}
		Kind::Planet(p) => {
			v.hands.level_up(p.hand(), 1.0);
			react(v, Joker::Constellation, LifecycleCtx::UsingConsumeable {
				kind: c.kind,
				hanged_glass: 0,
			});
		}
		Kind::Spectral(Spectral::BlackHole) => {
			for h in HandType::ALL {
				v.hands.level_up(h, 1.0);
			}
		}
		Kind::Voucher(vc) => {
			v.history.used_vouchers.insert(vc);
			shop::apply_voucher(&mut v.round, &mut v.events, vc, VoucherSource::Redeemed);
		}
		_ => return None,
	}
	Some(())
}

/// The hash of the item at `index` of `zone`, and whether its preview reads
/// the money; `None` if there is no item or it is a playing card, which is
/// never previewed.
fn item_digest(areas: &Areas, zone: ItemZone, index: usize) -> Option<(u64, bool)> {
	let of_card = |c: &Card| (card_hash(c), reads_money(c));
	let of_any = |c: &AnyCard| match c {
		AnyCard::Playing(_) => None,
		AnyCard::Card(c) => Some(of_card(c)),
	};
	match zone {
		ItemZone::ShopCard => areas.shop_jokers.get(index).and_then(of_any),
		ItemZone::ShopVoucher => areas.shop_vouchers.get(index).map(of_card),
		ItemZone::Pack => areas.pack_cards.get(index).and_then(of_any),
		ItemZone::Joker => areas.jokers.get(index).map(of_card),
		ItemZone::Consumable => areas.consumables.get(index).map(of_card),
	}
}

/// A copy of `g` with the item at `index` of `zone` acquired (shop card:
/// bought, consumables used at once; pack card: picked; voucher: redeemed;
/// consumable: used; joker: sold), money effects included, or `None` if the
/// item is not previewed.
fn with_item(g: &Game, zone: ItemZone, index: usize) -> Option<Game> {
	let mut v = g.clone();
	match zone {
		ItemZone::ShopCard => {
			let AnyCard::Card(c) = v.areas.shop_jokers.remove(index) else {
				return None;
			};
			v.run.dollars -= c.core.cost;
			acquire(&mut v, c)?;
		}
		ItemZone::Pack => {
			let AnyCard::Card(c) = v.areas.pack_cards.remove(index) else {
				return None;
			};
			acquire(&mut v, c)?;
		}
		ItemZone::Consumable => {
			let c = v.areas.consumables.remove(index);
			acquire(&mut v, c)?;
		}
		ItemZone::ShopVoucher => {
			let c = v.areas.shop_vouchers.remove(index);
			v.run.dollars -= c.core.cost;
			acquire(&mut v, c)?;
		}
		ItemZone::Joker => {
			let mut c = v.areas.jokers.remove(index);
			card::remove_from_deck(
				&mut v.areas.limits,
				&mut v.run,
				&mut v.round,
				&mut v.shop,
				&mut v.events,
				&mut c,
			);
			v.run.dollars += c.core.sell_cost;
			react(&mut v, Joker::Campfire, LifecycleCtx::SellingCard);
		}
	}
	settle(&mut v);
	Some(v)
}

impl BuildPreview {
	/// Token features of the item at `index` of `zone` (shop cards, pack
	/// cards, vouchers, consumables, jokers): log of the capacity with the
	/// item acquired (jokers: sold) over without, against the boss on the
	/// same hands; the capacity ratio and clear fraction with it; and a 1
	/// marking a previewed item. `None` for items without a preview (their
	/// columns stay zero). Results are cached per item.
	pub fn item_feats(&mut self, g: &Game, zone: ItemZone, index: usize) -> Option<ItemPreview> {
		debug_assert_eq!(
			state_key(g),
			self.key,
			"item_feats needs a state of the build this preview was made for"
		);
		let (card_hash, money) = item_digest(&g.areas, zone, index)?;
		let item_key = {
			let mut hasher = SipHasher13::new();
			// A money joker's own preview reads the money (`state_key` leaves it out).
			(
				Zone::from(zone).id(),
				index,
				card_hash,
				money.then(|| g.run.dollars.to_bits()),
			)
				.hash(&mut hasher);
			hasher.finish()
		};
		if let Some((_, feats)) = self.items.iter().find(|(k, _)| *k == item_key) {
			return *feats;
		}
		let Some(after) = with_item(g, zone, index) else {
			self.items.push((item_key, None));
			return None;
		};
		let changes_ante = after.run.ante != g.run.ante;
		let n = PREVIEW_ITEM_HANDS.min(self.samples.len());
		let (mut sum, mut base, mut clear, mut hands, mut chips, mut share) = (
			0.0,
			0.0,
			0usize,
			self.cap_boss.hands,
			self.cap_boss.chips,
			1.0,
		);
		let mut scratch = Scratch::default();
		// The item's game for each hand, moved into the last one.
		let worlds = iter::repeat_n(after, n).zip(self.seeds.iter().zip(&self.samples));
		for (game, (&seed, sample)) in worlds {
			let world = virtual_round(game, self.boss, seed);
			let best = if hand_digest(&world.areas, &world.blind, &world.hands) == sample.digest {
				best_of(&mut scratch, &world, &sample.plays)
			} else {
				best_of(&mut scratch, &world, &rank_plays(&world))
			};
			// The boss's rule on the round, unless the item disables the boss.
			share = if world.blind.disabled {
				1.0
			} else {
				self.round_share
			};
			hands = world.round.hands_left;
			chips = world.blind.chips;
			sum += best;
			base += sample.best;
			clear += usize::from(hands * best * share >= chips);
		}
		let n = num::float(n.max(1));
		let with = Capacity {
			mean: sum / n * share,
			hands,
			chips,
			p_clear: num::float(clear) / n,
		};
		// The change of capacity (hands x score), so Grabber and Troubadour count; over the target when the item
		// moves this boss's own target (Chicot). Hieroglyph and Petroglyph lower the ante instead: the run's last
		// boss is as far as before, so their gain is what they cost (the ratio column shows the nearer target).
		let target = if changes_ante {
			1.0
		} else {
			self.cap_boss.chips.max(1.0) / chips.max(1.0)
		};
		let feats = ItemPreview {
			gain: clamped_feat(
				((with.hands * sum / n + 1.0) / (self.cap_boss.hands * base / n + 1.0) * target)
					.ln(),
				-PREVIEW_LOG_MAX,
				PREVIEW_LOG_MAX,
			),
			ratio: with.ratio_feat(),
			p_clear: num::feat(with.p_clear),
		};
		self.items.push((item_key, Some(feats)));
		Some(feats)
	}
}
