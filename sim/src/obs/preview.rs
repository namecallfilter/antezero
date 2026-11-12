//! Build previews: outside rounds (shops, packs, blind select), the scoring
//! capacity of the current build against the coming blinds, and what a shop
//! or pack item would add to it.
//!
//! Like the play candidates, everything is estimated on worlds derived from
//! visible state only: the playing cards the player knows (deck, hand and
//! discard pile as one pool) are put in a canonical order and shuffled with a
//! seed taken from the ante and round, a hand is dealt against the blind
//! (its boss effects included) and scored with a reseeded RNG. The best play
//! of each of `PREVIEW_HANDS` sampled hands (over the `PREVIEW_PLAYS` plays
//! with the highest joker-free estimate) gives the capacity; an item's
//! preview repeats this with the item applied to a copy, on the first
//! `PREVIEW_ITEM_HANDS` of the same hands (paired with the base).
//!
//! Jokers are previewed in a canonical order (additive effects before
//! multiplicative ones, then by content: see `virtual_round`), so the
//! preview measures the build, not its current arrangement, and moving
//! jokers around does not change (or recompute) it.

use std::hash::{Hash, Hasher};

use super::candidates::{kicker_plays, rough_score, shortlist};
use super::digest::{
	blinds_done, canon_key, card_hash, hand_digest, mix, reads_money, splitmix_shuffle, state_key,
};
use super::{MAX_HAND, zone};
use crate::actions::{CardMask, for_each_subset, mask_fits};
use crate::card::{Card, Kind};
use crate::data::{BlindKind, Edition, HandType, Joker, Spectral};
use crate::ev::Ev;
use crate::events::St;
use crate::game::{BlindSlot, BlindStatus, Game, Phase};
use crate::rng::{Key, PseudoRandom, shuffle};
use crate::{actions, blind, cards, flow, jokers, num, shop};

/// Build previews (shops, packs, blind select): hands sampled per blind,
/// and plays simulated per hand (the best by a cheap joker-free estimate).
pub const PREVIEW_HANDS: usize = 4;
pub const PREVIEW_PLAYS: usize = 6;
/// Recent build previews kept per env (a shop visit returns to the same
/// build after joker moves, rerolls and packs).
pub const PREVIEW_CACHE: usize = 4;

/// Hands (of `PREVIEW_HANDS`) an item is previewed on (the first ones; the
/// next blind is previewed on as many).
pub const PREVIEW_ITEM_HANDS: usize = 1;
/// Token float columns of an item's preview (see [`BuildPreview::item_feats`]).
pub const PREVIEW_TOK: usize = 4;

/// Scoring capacity against one blind.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Capacity {
	/// Mean over the sampled hands of the best play's score.
	pub mean: f64,
	/// Hands the round would start with (after the blind's effect).
	pub hands: f64,
	pub chips: f64,
	/// Fraction of sampled hands whose best play times `hands` reaches `chips`.
	pub p_clear: f64,
}

impl Capacity {
	/// `hands * mean / chips`: how many times over the blind is covered.
	#[must_use]
	pub fn ratio(&self) -> f64 {
		self.hands * self.mean / self.chips.max(1.0)
	}

	/// Log of the ratio (with a +1 floor on the score), clamped to +-4.
	#[must_use]
	pub fn log_ratio(&self) -> f32 {
		feat(
			((self.hands * self.mean + 1.0) / self.chips.max(1.0)).ln(),
			-4.0,
			4.0,
		)
	}
}

/// One sampled hand against the boss: its best score, the plays ranked for
/// it (reused by item previews that deal the same hand) and a digest of its
/// cards.
#[derive(Clone, Debug, PartialEq)]
struct Sample {
	best: f64,
	plays: Vec<CardMask>,
	digest: u64,
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
	items: Vec<(u64, [f32; PREVIEW_TOK])>,
}

/// The next blind to play: the first slot neither defeated nor skipped.
fn next_blind(g: &Game) -> BlindKind {
	for (i, k) in [BlindKind::Small, BlindKind::Big, g.boss_choice]
		.into_iter()
		.enumerate()
	{
		if !matches!(
			g.blind_states[i],
			BlindStatus::Defeated | BlindStatus::Skipped
		) {
			return k;
		}
	}
	g.boss_choice
}

/// `x` as a feature: clamped to `[lo, hi]`, 0 if undefined (inf / inf).
fn feat(x: f64, lo: f64, hi: f64) -> f32 {
	if x.is_nan() {
		0.0
	} else {
		num::feat(x.clamp(lo, hi))
	}
}

/// Jokers whose scoring effect multiplies (they go last in the canonical
/// order, where they are worth most).
fn multiplies(c: &Card) -> bool {
	use Joker as J;
	c.ab.x_mult > 1.0
		|| c.ab.caino_xmult > 0.0
		|| matches!(
			c.kind,
			Kind::Joker(
				J::Stencil
					| J::LoyaltyCard
					| J::SteelJoker
					| J::Blackboard
					| J::Constellation
					| J::Cavendish
					| J::CardSharp
					| J::Madness
					| J::Vampire
					| J::Hologram
					| J::Baron
					| J::Obelisk
					| J::Photograph
					| J::LuckyCat
					| J::Baseball
					| J::Ancient
					| J::Ramen
					| J::Campfire
					| J::Acrobat
					| J::Throwback
					| J::Bloodstone
					| J::Glass
					| J::FlowerPot
					| J::Idol
					| J::SeeingDouble
					| J::HitTheRoad
					| J::Duo
					| J::Trio
					| J::Family
					| J::Order
					| J::Tribe
					| J::DriversLicense
					| J::Caino
					| J::Triboulet
					| J::Yorick
			)
		)
}

/// Applies the numeric effects of the queued events (hands, discards, hand
/// size, ante) and drops the rest: a copy that previews a round never runs
/// its animations.
fn settle(v: &mut Game) {
	loop {
		let pending = v.events.take_pending();
		if pending.is_empty() {
			break;
		}
		for e in pending {
			match e {
				Ev::EaseHands(d) => v.hands_left += d,
				Ev::EaseDiscard(d) => v.discards_left += d,
				Ev::HandSize(d) => v.hand_limit.change(d),
				Ev::EaseAnte(d) => v.ante += d,
				Ev::AcornOuter => {
					v.jokers.sort_unstable_by_key(|c| c.id);
					let s = v.rng.seed_for(&Key::from("aajk"));
					shuffle(&mut v.jokers, s);
				}
				// Round-start jokers (see `virtual_round`); disabling a blind queues its own effects.
				Ev::Burglar => {
					v.hands_left += 3.0;
					v.discards_left = 0.0;
				}
				Ev::ChicotOuter => blind::disable_blind(v),
				_ => {}
			}
		}
	}
}

/// Gathers every playing card of `v` into its deck, face down, canonically
/// ordered (ids renumbered in that order) and shuffled by the visible seed.
fn pool_into_deck(v: &mut Game, seed: u64) {
	let mut pool: Vec<Card> = Vec::with_capacity(v.playing_card_count());
	pool.append(&mut v.deck);
	pool.append(&mut v.hand);
	pool.append(&mut v.play);
	pool.append(&mut v.discard);
	for c in &mut pool {
		c.face_down = true;
		c.ab.wheel_flipped = false;
		c.ab.discarded = false;
		c.ab.forced_selection = false;
	}
	pool.sort_by_key(canon_key);
	for (i, c) in pool.iter_mut().enumerate() {
		c.id = 1_000_000 + num::index32(i);
	}
	splitmix_shuffle(&mut pool, seed);
	v.deck = pool;
}

/// A copy of `g` at the first decision of a round against `blind`: the
/// pool of playing cards in a canonical order shuffled with `seed`, the
/// hand dealt (face up: it is a sample), the jokers in the canonical order
/// with ids renumbered (the game breaks ties by id), the blind set with its
/// effects (hands, discards, hand size, debuffs, flipped jokers, Cerulean
/// Bell, Crimson Heart), the RNG reseeded from `seed`. Depends on visible
/// state only.
#[must_use]
pub fn virtual_round(g: &Game, blind: BlindKind, seed: u64) -> Game {
	let mut v = g.clone();
	v.rng = PseudoRandom::new_untraced(&format!("BP{seed:X}"));
	// Leave the shop or pack and its pending animation behind.
	v.events.clear();
	v.consumed.clear();
	v.in_flight = None;
	v.pending_highlight.clear();
	v.pack = None;
	v.pack_interrupt = None;
	v.pack_cards.clear();
	v.shop_open = false;
	v.state = St::SelectingHand;
	v.state_complete = true;
	v.phase = Phase::Round;
	v.end_round_pending = false;
	v.scored = false;
	v.pause_rest = false;
	// The round's counters (`new_round`).
	v.discards_left = 0f64.max(v.rr_discards + v.round_bonus_discards);
	v.hands_left = 1f64.max(v.rr_hands + v.round_bonus_next_hands);
	v.hands_played_round = 0.0;
	v.discards_used = 0.0;
	for h in &mut v.hands {
		h.played_this_round = 0.0;
	}
	pool_into_deck(&mut v, seed);
	// Jokers in a canonical order that ignores the player's (cached previews
	// are keyed by the joker multiset): additive before multiplicative, a
	// Polychrome last among the additive and a Holo first among the
	// multiplicative (where their editions are worth most), then by content.
	v.jokers.sort_by_cached_key(|c| {
		let m = multiplies(c);
		(
			m,
			if m {
				c.edition != Edition::Holo
			} else {
				c.edition == Edition::Polychrome
			},
			card_hash(c),
		)
	});
	for (i, c) in v.jokers.iter_mut().enumerate() {
		c.id = 2_000_000 + num::index32(i);
	}
	// The blind and its effects.
	v.blind_states[BlindSlot::of(blind) as usize] = BlindStatus::Current;
	v.selected_blind = Some(blind);
	v.facing_blind = true;
	blind::set_blind(&mut v, Some(blind));
	settle(&mut v);
	// The round-start joker effects that are certain: Burglar trades the
	// discards for three hands, Chicot disables the boss. (The others
	// destroy or create cards at random and are left out.) A copy joker
	// repeats Burglar when it stands by it as the blind is selected, which
	// is how a player has them then; Chicot cannot be copied.
	for i in 0..v.jokers.len() {
		if matches!(
			v.jokers[i].kind,
			Kind::Joker(Joker::Burglar | Joker::Chicot)
		) {
			jokers::calc_joker(
				&mut v,
				i,
				&crate::jokers::Ctx::SettingBlind {
					boss: blind.is_boss(),
				},
				None,
			);
		}
	}
	settle(&mut v);
	if v.jokers
		.iter()
		.any(|c| c.kind == Kind::Joker(Joker::Burglar) && !c.debuff)
	{
		v.hands_left += 3.0
			* num::float(
				v.jokers
					.iter()
					.filter(|c| {
						matches!(c.kind, Kind::Joker(Joker::Blueprint | Joker::Brainstorm))
							&& !c.debuff
					})
					.count(),
			);
	}
	// Deal.
	let n = num::trunc(v.hand_limit.limit.max(0.0)).min(v.deck.len());
	for _ in 0..n {
		let Some(mut c) = v.deck.pop() else { break };
		c.face_down = false;
		v.hand.push(c);
	}
	flow::sort_hand(&mut v);
	blind::drawn_to_hand(&mut v);
	place_copies(&mut v);
	v
}

/// Moves the copy jokers of a dealt round to where its hand scores most (a
/// player arranges them before playing; by content alone they copied a
/// neighbour picked by a hash): Brainstorm gets the best joker to lead the
/// row (and stays put or goes last), then each Blueprint the best joker to
/// stand before, each tried on
/// the hand's two best plays. Face-down jokers stay as dealt (their order is
/// hidden).
fn place_copies(v: &mut Game) {
	let is_copy = |c: &Card| matches!(c.kind, Kind::Joker(Joker::Blueprint | Joker::Brainstorm));
	if !v.jokers.iter().any(is_copy) || v.jokers.len() < 2 || v.jokers.iter().any(|c| c.face_down) {
		return;
	}
	// Rows are compared on the two plays that score most as dealt.
	let mut plays: Vec<(f64, CardMask)> = rank_plays(v)
		.into_iter()
		.filter_map(|m| v.clone().preview_play(m).ok().map(|s| (s, m)))
		.collect();
	plays.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
	let plays: Vec<CardMask> = plays.iter().take(2).map(|p| p.1).collect();
	// Ids name the jokers while they move; the game breaks ties by id, so a row is scored (and left) renumbered.
	let ids: Vec<(u32, Kind)> = v.jokers.iter().map(|c| (c.id, c.kind)).collect();
	let score = |w: &Game| {
		let mut w = w.clone();
		for (i, c) in w.jokers.iter_mut().enumerate() {
			c.id = 3_000_000 + num::index32(i);
		}
		best_of(&w, &plays)
	};
	let pos = |w: &Game, id: u32| w.jokers.iter().position(|c| c.id == id).expect("joker");
	let mut best = score(v);
	let mut keep = |w: Game, v: &mut Game| {
		let s = score(&w);
		if s > best {
			(best, *v) = (s, w);
		}
	};
	if ids.iter().any(|x| x.1 == Kind::Joker(Joker::Brainstorm)) {
		for &(t, k) in &ids {
			if !matches!(k, Kind::Joker(Joker::Blueprint | Joker::Brainstorm)) {
				// Its own slot matters too: a copied multiplier is worth most at the end of the row.
				for last in [false, true] {
					let mut w = v.clone();
					let c = w.jokers.remove(pos(&w, t));
					w.jokers.insert(0, c);
					if last {
						let i = w
							.jokers
							.iter()
							.position(|c| c.kind == Kind::Joker(Joker::Brainstorm))
							.expect("Brainstorm");
						let c = w.jokers.remove(i);
						w.jokers.push(c);
					}
					keep(w, v);
				}
			}
		}
	}
	for &(b, _) in ids.iter().filter(|x| x.1 == Kind::Joker(Joker::Blueprint)) {
		for &(t, _) in ids.iter().filter(|x| x.0 != b) {
			let mut w = v.clone();
			let c = w.jokers.remove(pos(&w, b));
			let at = pos(&w, t);
			w.jokers.insert(at, c);
			keep(w, v);
		}
	}
	for (i, c) in v.jokers.iter_mut().enumerate() {
		c.id = 2_000_000 + num::index32(i);
	}
}

/// The `PREVIEW_PLAYS` legal plays of `v`'s hand worth simulating (see
/// `shortlist`), which are the smallest play of each scoring core, plus the
/// plays with kickers some jokers need (see `kicker_plays`).
fn rank_plays(v: &Game) -> Vec<CardMask> {
	let flags = v.eval_flags();
	let nh = v.hand.len().min(MAX_HAND);
	let forced = actions::forced_mask(v);
	// The Psychic scores any play of fewer than five cards 0 (skipped, unless
	// the hand has fewer than five).
	let min = if v.blind.active(BlindKind::Psychic) && nh >= 5 {
		5
	} else {
		1
	};
	let mut plays = Vec::new();
	for_each_subset(nh, 5, |m| {
		if m & forced == forced && mask_fits(m, nh) && m.count_ones() >= min {
			plays.push(rough_score(v, m, flags));
		}
	});
	let mut out = shortlist(plays, PREVIEW_PLAYS);
	for m in kicker_plays(v, &out, nh) {
		if !out.contains(&m) && v.check(&crate::actions::Action::Play(m)).is_ok() {
			out.push(m);
		}
	}
	out
}

/// Score of the best of `plays` (those legal in `v`), simulated once each
/// on `v`'s RNG.
fn best_of(v: &Game, plays: &[CardMask]) -> f64 {
	plays
		.iter()
		.filter_map(|&m| v.clone().preview_play(m).ok())
		.fold(0.0, f64::max)
}

/// Score of the best play of `v`'s hand (see `rank_plays`, `best_of`).
#[must_use]
pub fn best_play(v: &Game) -> f64 {
	best_of(v, &rank_plays(v))
}

/// Capacity of `g` against `blind` over the hands dealt with `seeds`.
#[must_use]
pub fn capacity(g: &Game, blind: BlindKind, seeds: &[u64]) -> Capacity {
	capacity_samples(g, blind, seeds).0
}

fn capacity_samples(g: &Game, blind: BlindKind, seeds: &[u64]) -> (Capacity, Vec<Sample>) {
	let (mut hands, mut chips) = (0.0, 0.0);
	let (mut samples, mut rounds) = (
		Vec::with_capacity(seeds.len()),
		Vec::with_capacity(seeds.len()),
	);
	for &s in seeds {
		let v = virtual_round(g, blind, s);
		let plays = rank_plays(&v);
		let best = best_of(&v, &plays);
		hands = v.hands_left;
		chips = v.blind.chips;
		samples.push(Sample {
			best,
			plays,
			digest: hand_digest(&v),
		});
		rounds.push(v);
	}
	let n = num::float(seeds.len().max(1));
	let per_round = num::trunc(hands.max(1.0));
	if !(matches!(blind, BlindKind::Eye | BlindKind::Mouth)
		&& per_round > 1
		&& seeds.len() > 1
		&& !rounds[0].blind.disabled)
	{
		let clear = samples.iter().filter(|s| hands * s.best >= chips).count();
		return (
			Capacity {
				mean: samples.iter().map(|s| s.best).sum::<f64>() / n,
				hands,
				chips,
				p_clear: num::float(clear) / n,
			},
			samples,
		);
	}
	// Against The Eye and The Mouth a hand is not worth its best play: the
	// sampled hands are played as rounds of the round's own number of hands
	// (dealt again from the first when it has more hands than were sampled),
	// and a round clears or not as a whole.
	let mut deals: Vec<Game> = (0..per_round.max(rounds.len()))
		.map(|i| rounds[i % rounds.len()].clone())
		.collect();
	let (mut total, mut played, mut clear, mut n_rounds) = (0.0, 0usize, 0usize, 0usize);
	for round in deals.chunks_mut(per_round) {
		let scores = if blind == BlindKind::Eye {
			eye_round(round)
		} else {
			mouth_round(round)
		};
		let sum: f64 = scores.iter().sum();
		total += sum;
		played += scores.len();
		n_rounds += 1;
		clear += usize::from(sum * hands / num::float(scores.len()) >= chips); // a short last round counts at full length
	}
	(
		Capacity {
			mean: total / num::float(played),
			hands,
			chips,
			p_clear: num::float(clear) / num::float(n_rounds),
		},
		samples,
	)
}

/// The best of `v`'s shortlisted plays (as the blind stands) and its hand type.
fn best_typed(v: &Game) -> (f64, Option<HandType>) {
	let flags = v.eval_flags();
	let mut best = (0.0, None);
	for m in rank_plays(v) {
		if let Ok(s) = v.clone().preview_play(m)
			&& s > best.0
		{
			best = (s, rough_score(v, m, flags).core.map(|(hand, _)| hand));
		}
	}
	best
}

/// The Eye: no hand type scores twice in a round, so each sampled hand
/// plays its best type that the earlier ones left.
fn eye_round(rounds: &mut [Game]) -> Vec<f64> {
	let mut used = 0u16;
	rounds
		.iter_mut()
		.map(|v| {
			v.blind.eye_hands = used;
			let (s, t) = best_typed(v);
			used |= t.map_or(0, |t| 1 << (t as u16));
			s
		})
		.collect()
}

/// The Mouth: only the first hand's type scores all round. Each sampled
/// hand's best type is tried as that type on every hand; the best round counts.
fn mouth_round(rounds: &mut [Game]) -> Vec<f64> {
	let mut types: Vec<HandType> = rounds.iter().filter_map(|v| best_typed(v).1).collect();
	types.sort_unstable();
	types.dedup();
	let mut best: Vec<f64> = vec![0.0; rounds.len()];
	for t in types {
		let scores: Vec<f64> = rounds
			.iter_mut()
			.map(|v| {
				v.blind.only_hand = Some(t);
				best_typed(v).0
			})
			.collect();
		if scores.iter().sum::<f64>() > best.iter().sum::<f64>() {
			best = scores;
		}
	}
	best
}

/// Makes the preview of `g` the front of `cache` (this env's recent
/// previews): a cached preview of the same build is moved there and reused
/// (with its item previews), otherwise a new one is computed and inserted
/// (the cache keeps `PREVIEW_CACHE` entries). Returns false outside shops,
/// packs and blind select (nothing is previewed there).
pub fn build_preview(g: &Game, cache: &mut Vec<BuildPreview>) -> bool {
	if !matches!(g.phase, Phase::Shop | Phase::Pack | Phase::BlindSelect) {
		return false;
	}
	let key = state_key(g);
	if let Some(i) = cache.iter().position(|p| p.key == key) {
		cache[..=i].rotate_right(1);
		return true;
	}
	// The boss and the hands played so far (both visible, both fixed while a
	// shop is open) make the sampled hands differ between games: seeded from
	// the ante and round alone, every game with the same cards previewed its
	// items on the same hand.
	let base = {
		let mut h = std::hash::DefaultHasher::new();
		(g.ante, g.round, blinds_done(g), g.boss_choice).hash(&mut h);
		for l in &g.hands {
			l.played.to_bits().hash(&mut h);
		}
		h.finish()
	};
	let seeds: [u64; PREVIEW_HANDS] = std::array::from_fn(|k| mix(base ^ (k as u64 + 1)));
	let boss = g.boss_choice;
	let next = next_blind(g);
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
	true
}

/// The reaction of every `j` owned to what a preview just did: the growth
/// that comes with using a planet (Constellation) or selling a card (Campfire).
fn react(v: &mut Game, j: Joker, ctx: &crate::jokers::Ctx) {
	for i in 0..v.jokers.len() {
		if v.jokers[i].kind == Kind::Joker(j) {
			jokers::calc_joker(v, i, ctx, None);
		}
	}
}

/// Adds a bought or picked card to the build: jokers join the joker area;
/// planets and Black Hole are used. Playing cards and other consumables are
/// not previewed (`None`).
fn acquire(v: &mut Game, mut c: Card) -> Option<()> {
	match c.kind {
		Kind::Joker(_) => {
			cards::add_to_deck(v, &mut c, false);
			v.jokers.push(c);
		}
		Kind::Planet(p) => {
			cards::level_up_hand(v, p.hand(), 1.0);
			react(
				v,
				Joker::Constellation,
				&crate::jokers::Ctx::UsingConsumeable {
					kind: c.kind,
					hanged_glass: 0,
				},
			);
		}
		Kind::Spectral(Spectral::BlackHole) => {
			for h in HandType::ALL {
				cards::level_up_hand(v, h, 1.0);
			}
		}
		_ => return None,
	}
	Some(())
}

/// The card at `index` of the item area `zone`.
fn item_card(g: &Game, zone: i32, index: usize) -> Option<&Card> {
	match zone {
		zone::SHOP_CARD => g.shop_jokers.get(index),
		zone::SHOP_VOUCHER => g.shop_vouchers.get(index),
		zone::PACK => g.pack_cards.get(index),
		zone::JOKER => g.jokers.get(index),
		zone::CONSUMABLE => g.consumables.get(index),
		_ => None,
	}
}

/// A copy of `g` with the item at `index` of `zone` acquired (shop card:
/// bought, consumables used at once; pack card: picked; voucher: redeemed;
/// consumable: used; joker: sold), money effects included, or `None` if the
/// item is not previewed.
fn with_item(g: &Game, zone: i32, index: usize) -> Option<Game> {
	let mut v = g.clone();
	match zone {
		zone::SHOP_CARD => {
			let c = v.shop_jokers.remove(index);
			v.dollars -= c.cost;
			acquire(&mut v, c)?;
		}
		zone::PACK => {
			let c = v.pack_cards.remove(index);
			acquire(&mut v, c)?;
		}
		zone::CONSUMABLE => {
			let c = v.consumables.remove(index);
			acquire(&mut v, c)?;
		}
		zone::SHOP_VOUCHER => {
			let c = v.shop_vouchers.remove(index);
			let Kind::Voucher(vc) = c.kind else {
				return None;
			};
			v.dollars -= c.cost;
			v.set_voucher_used(vc);
			shop::apply_voucher(&mut v, vc, false);
		}
		zone::JOKER => {
			let mut c = v.jokers.remove(index);
			cards::remove_from_deck(&mut v, &mut c, false);
			v.dollars += c.sell_cost;
			react(&mut v, Joker::Campfire, &crate::jokers::Ctx::SellingCard);
		}
		_ => return None,
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
	#[expect(
		clippy::many_single_char_names,
		reason = "the short names used across the obs modules"
	)]
	pub fn item_feats(&mut self, g: &Game, zone: i32, index: usize) -> Option<[f32; PREVIEW_TOK]> {
		let c = item_card(g, zone, index)?;
		let ik = {
			let mut h = std::hash::DefaultHasher::new();
			// A money joker's own preview reads the money (`state_key` leaves it out).
			(
				zone,
				index,
				card_hash(c),
				reads_money(c).then(|| g.dollars.to_bits()),
			)
				.hash(&mut h);
			h.finish()
		};
		if let Some((_, f)) = self.items.iter().find(|(k, _)| *k == ik) {
			return Some(*f);
		}
		let v = with_item(g, zone, index)?;
		let n = PREVIEW_ITEM_HANDS.min(self.samples.len());
		let (mut sum, mut base, mut clear, mut hands, mut chips, mut share) = (
			0.0,
			0.0,
			0usize,
			self.cap_boss.hands,
			self.cap_boss.chips,
			1.0,
		);
		for k in 0..n {
			let w = virtual_round(&v, self.boss, self.seeds[k]);
			let s = &self.samples[k];
			let best = if hand_digest(&w) == s.digest {
				best_of(&w, &s.plays)
			} else {
				best_play(&w)
			};
			// The boss's rule on the round, unless the item disables the boss.
			share = if w.blind.disabled {
				1.0
			} else {
				self.round_share
			};
			hands = w.hands_left;
			chips = w.blind.chips;
			sum += best;
			base += s.best;
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
		let target = if v.ante == g.ante {
			self.cap_boss.chips.max(1.0) / chips.max(1.0)
		} else {
			1.0
		};
		let gain = feat(
			((with.hands * sum / n + 1.0) / (self.cap_boss.hands * base / n + 1.0) * target).ln(),
			-4.0,
			4.0,
		);
		let f = [
			gain,
			num::feat(with.ratio().min(4.0)),
			num::feat(with.p_clear),
			1.0,
		];
		self.items.push((ik, f));
		Some(f)
	}
}
