//! Build previews: outside rounds (shops, packs, blind select), the scoring
//! capacity of the current build against the coming blinds.
//!
//! Like the play candidates, everything is estimated on worlds derived from
//! visible state only: the playing cards the player knows (deck, hand and
//! discard pile as one pool) are put in a canonical order and shuffled with a
//! seed taken from the ante and round, a hand is dealt against the blind
//! (its boss effects included) and scored with a reseeded RNG. The best play
//! of each of `PREVIEW_HANDS` sampled hands (over the `PREVIEW_PLAYS` plays
//! with the highest joker-free estimate) gives the capacity; see
//! [`super::preview_items`] for item deltas on the same hands.
//!
//! Jokers are previewed in a canonical order (additive effects before
//! multiplicative ones, then by content: see `virtual_round`), so the
//! preview measures the build, not its current arrangement, and moving
//! jokers around does not change (or recompute) it.

use super::candidates::{PREVIEW_PLAYS, kicker_plays, rough_score, shortlist};
use super::digest::{card_hash, hand_digest};
use super::layout::shown_hand;
use crate::actions::Action;
use crate::card::{Card, CardLike, CardMask, Kind, canon_key};
use crate::data::{BlindKind, Edition, HandSet, HandType, Joker};
use crate::event::Event;
use crate::flow::FlowEvent;
use crate::game::{Areas, BlindSlot, BlindStatus, Game, Hand, Phase, Scratch, State};
use crate::joker::{JokerEvent, LifecycleCtx};
use crate::rng::{self, Key, PseudoRandom};
use crate::{actions, areas, flow, joker, num, round};

/// Hands a build preview (shops, packs, blind select) samples per blind.
pub const PREVIEW_HANDS: usize = 4;
/// The bound of the previews' log-ratio features (a capacity's
/// [`Capacity::log_ratio`], an item's [`ItemPreview::gain`](super::ItemPreview::gain)): they lie in
/// `-PREVIEW_LOG_MAX..=PREVIEW_LOG_MAX`.
pub const PREVIEW_LOG_MAX: f64 = 4.0;
/// The cap of the previews' capacity-ratio features ([`Capacity::ratio_feat`]).
pub const PREVIEW_RATIO_MAX: f64 = 4.0;

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
	/// How many times over the blind is covered.
	#[must_use]
	pub fn ratio(&self) -> f64 {
		self.hands * self.mean / self.chips.max(1.0)
	}

	#[must_use]
	pub fn log_ratio(&self) -> f32 {
		clamped_feat(
			((self.hands * self.mean + 1.0) / self.chips.max(1.0)).ln(),
			-PREVIEW_LOG_MAX,
			PREVIEW_LOG_MAX,
		)
	}

	/// [`Capacity::ratio`] as a feature, capped at [`PREVIEW_RATIO_MAX`].
	#[must_use]
	pub fn ratio_feat(&self) -> f32 {
		num::feat(self.ratio().min(PREVIEW_RATIO_MAX))
	}
}

/// One sampled hand against the boss: its best score, the plays ranked for
/// it (reused by item previews that deal the same hand) and a digest of its
/// cards.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Sample {
	pub(super) best: f64,
	pub(super) plays: Vec<CardMask>,
	pub(super) digest: u64,
}

// A preview renumbers the cards it orders (the game breaks ties by id) into
// ranges above any real card's.
/// The ids of a virtual round's pool of playing cards.
const PREVIEW_CARD_IDS: u32 = 1_000_000;
/// The ids of a virtual round's jokers, in their canonical order.
const PREVIEW_JOKER_IDS: u32 = 2_000_000;
/// The ids of the joker rows `place_copies` tries.
const TRIAL_JOKER_IDS: u32 = 3_000_000;

/// Numbers `cards` in order from `base`.
fn renumber(cards: &mut [impl CardLike], base: u32) {
	for (i, c) in cards.iter_mut().enumerate() {
		c.core_mut().id = base + num::index32(i);
	}
}

/// `x` clamped to `lo..=hi` as a feature; NaN, which a ratio of two
/// overflowed (infinite) scores in endless mode gives, is 0.
pub(super) fn clamped_feat(x: f64, lo: f64, hi: f64) -> f32 {
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
pub(super) fn settle(v: &mut Game) {
	loop {
		let pending = v.events.take_pending();
		if pending.is_empty() {
			break;
		}
		for e in pending {
			match e {
				Event::EaseHands(d) => v.round.hands_left += d,
				Event::EaseDiscard(d) => v.round.discards_left += d,
				Event::HandSize(d) => v.areas.hand.limit.change(d),
				Event::EaseAnte(d) => v.run.ante += d,
				Event::Flow(FlowEvent::AcornOuter) => {
					flow::shuffle_area(&mut v.areas.jokers, &mut v.rng, &Key::from("aajk"));
				}
				// Round-start jokers (see `virtual_round`); disabling a blind queues its own effects.
				Event::Joker(JokerEvent::Burglar) => {
					v.round.hands_left += 3.0;
					v.round.discards_left = 0.0;
				}
				Event::Joker(JokerEvent::ChicotOuter) => round::disable_blind(v),
				_ => {}
			}
		}
	}
}

/// Gathers every playing card into the deck, face down, in the canonical
/// order shuffled with `seed`.
fn pool_into_deck(areas: &mut Areas, seed: u64) {
	areas.deck.append(&mut areas.hand.cards);
	areas.deck.append(&mut areas.play);
	areas.deck.append(&mut areas.discard);
	for c in &mut areas.deck {
		c.core.face_down = true;
		c.wheel_flipped = false;
		c.forced_selection = false;
	}
	areas.deck.sort_by_key(canon_key);
	renumber(&mut areas.deck, PREVIEW_CARD_IDS);
	rng::splitmix_shuffle(&mut areas.deck, seed);
}

/// `v` (a copy of a game) at the first decision of a round against `blind`: the
/// pool of playing cards in a canonical order shuffled with `seed`, the
/// hand dealt (face up: it is a sample), the jokers in the canonical order
/// with ids renumbered (the game breaks ties by id), the blind set with its
/// effects (hands, discards, hand size, debuffs, flipped jokers, Cerulean
/// Bell, Crimson Heart), the RNG reseeded from `seed`. Depends on visible
/// state only.
#[must_use]
pub fn virtual_round(mut v: Game, blind: BlindKind, seed: u64) -> Game {
	v.rng = PseudoRandom::new_untraced(&format!("BP{seed:X}"));
	// Leave the shop or pack and its pending animation behind.
	v.events.clear();
	v.areas.consumed.clear();
	v.pack.open = None;
	v.areas.pack_cards.clear();
	v.shop.open = false;
	v.flow.state = State::SelectingHand;
	v.flow.state_complete = true;
	v.flow.phase = Phase::Round;
	v.flow.end_round_pending = false;
	v.flow.scored = false;
	flow::reset_round_counters(&mut v.round, &mut v.hands);
	pool_into_deck(&mut v.areas, seed);
	// Jokers in a canonical order that ignores the player's (cached previews
	// are keyed by the joker multiset): additive before multiplicative, a
	// Polychrome last among the additive and a Holo first among the
	// multiplicative (where their editions are worth most), then by content.
	v.areas.jokers.sort_by_cached_key(|c| {
		let m = multiplies(c);
		(
			m,
			if m {
				c.core.edition != Some(Edition::Holo)
			} else {
				c.core.edition == Some(Edition::Polychrome)
			},
			card_hash(c),
		)
	});
	renumber(&mut v.areas.jokers, PREVIEW_JOKER_IDS);
	v.blind_select.states[BlindSlot::of(blind)] = BlindStatus::Current;
	v.blind_select.selected = Some(blind);
	v.round.facing_blind = true;
	round::set_blind(&mut v, Some(blind));
	settle(&mut v);
	// The round-start joker effects that are certain: Burglar trades the
	// discards for three hands, Chicot disables the boss. (The others
	// destroy or create cards at random and are left out.) A copy joker
	// repeats Burglar when it stands by it as the blind is selected, which
	// is how a player has them then; Chicot cannot be copied.
	for i in 0..v.areas.jokers.len() {
		if matches!(
			v.areas.jokers[i].kind,
			Kind::Joker(Joker::Burglar | Joker::Chicot)
		) {
			joker::calc_joker(&mut v, i, LifecycleCtx::SettingBlind {
				boss: blind.is_boss(),
			});
		}
	}
	settle(&mut v);
	if v.areas
		.jokers
		.iter()
		.any(|c| c.kind == Kind::Joker(Joker::Burglar) && !c.core.debuff)
	{
		v.round.hands_left += 3.0
			* num::float(
				v.areas
					.jokers
					.iter()
					.filter(|c| {
						matches!(c.kind, Kind::Joker(Joker::Blueprint | Joker::Brainstorm))
							&& !c.core.debuff
					})
					.count(),
			);
	}
	let n = num::trunc(v.areas.hand.limit.limit.max(0.0)).min(v.areas.deck.len());
	let top = v.areas.deck.len() - n;
	for mut c in v.areas.deck.drain(top..).rev() {
		c.core.face_down = false;
		v.areas.hand.cards.push(c);
	}
	v.areas.hand.sort();
	round::drawn_to_hand(&mut v);
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
	if !v.areas.jokers.iter().any(is_copy)
		|| v.areas.jokers.len() < 2
		|| v.areas.jokers.iter().any(|c| c.core.face_down)
	{
		return;
	}
	let mut scratch = Scratch::default();
	let mut plays: Vec<(f64, CardMask)> = rank_plays(v)
		.into_iter()
		.filter_map(|m| {
			let score = scratch.copy(v, &v.rng).preview_play(m).ok()?;
			Some((score, m))
		})
		.collect();
	plays.sort_by(|a, b| b.0.total_cmp(&a.0));
	plays.truncate(2);
	// Ids name the jokers while they move; the game breaks ties by id, so a row is scored (and left) renumbered.
	let ids: Vec<(u32, Kind)> = v.areas.jokers.iter().map(|c| (c.core.id, c.kind)).collect();
	let score = |scratch: &mut Scratch, v: &Game, row: &[Card]| {
		plays
			.iter()
			.filter_map(|&(_, m)| {
				let copy = scratch.copy(v, &v.rng);
				copy.areas.jokers.clone_from_slice(row);
				renumber(&mut copy.areas.jokers, TRIAL_JOKER_IDS);
				copy.preview_play(m).ok()
			})
			.fold(0.0, f64::max)
	};
	let pos = |row: &[Card], id: u32| {
		areas::position_of(row, id).expect("every order holds the same jokers")
	};
	let mut best = score(&mut scratch, v, &v.areas.jokers);
	let mut keep = |row: &[Card], v: &mut Game| {
		let s = score(&mut scratch, v, row);
		if s > best {
			best = s;
			v.areas.jokers.clone_from_slice(row);
		}
	};
	// The row being tried, refreshed from the current one each time.
	let mut row: Vec<Card> = Vec::with_capacity(v.areas.jokers.len());
	if ids.iter().any(|x| x.1 == Kind::Joker(Joker::Brainstorm)) {
		for &(t, k) in &ids {
			if !matches!(k, Kind::Joker(Joker::Blueprint | Joker::Brainstorm)) {
				// Its own slot matters too: a copied multiplier is worth most at the end of the row.
				for last in [false, true] {
					row.clone_from(&v.areas.jokers);
					let c = row.remove(pos(&row, t));
					row.insert(0, c);
					if last {
						let i = row
							.iter()
							.position(|c| c.kind == Kind::Joker(Joker::Brainstorm))
							.expect("this branch runs only when a Brainstorm is owned");
						let c = row.remove(i);
						row.push(c);
					}
					keep(&row, v);
				}
			}
		}
	}
	for &(b, _) in ids.iter().filter(|x| x.1 == Kind::Joker(Joker::Blueprint)) {
		for &(t, _) in ids.iter().filter(|x| x.0 != b) {
			row.clone_from(&v.areas.jokers);
			let c = row.remove(pos(&row, b));
			let at = pos(&row, t);
			row.insert(at, c);
			keep(&row, v);
		}
	}
	renumber(&mut v.areas.jokers, PREVIEW_JOKER_IDS);
}

pub(super) fn rank_plays(v: &Game) -> Vec<CardMask> {
	let flags = v.areas.eval_flags();
	let nh = shown_hand(&v.areas.hand.cards).len();
	let forced = actions::forced_mask(&v.areas.hand.cards);
	// The Psychic scores any play of fewer than five cards 0 (skipped, unless
	// the hand has fewer than five).
	let min = if v.blind.active(BlindKind::Psychic) && nh >= 5 {
		5
	} else {
		1
	};
	let mut plays = Vec::new();
	CardMask::for_each_subset(nh, Hand::HIGHLIGHT_LIMIT, |m| {
		if m.contains_all(forced) && m.fits(nh) && m.len() >= min {
			plays.push(rough_score(&v.areas, &v.blind, &v.hands, m, flags));
		}
	});
	let mut out = shortlist(plays, PREVIEW_PLAYS);
	for m in kicker_plays(&v.areas, &out, nh) {
		if !out.contains(&m) && v.check(Action::Play(m)).is_ok() {
			out.push(m);
		}
	}
	out
}

pub(super) fn best_of(scratch: &mut Scratch, v: &Game, plays: &[CardMask]) -> f64 {
	plays
		.iter()
		.filter_map(|&m| scratch.copy(v, &v.rng).preview_play(m).ok())
		.fold(0.0, f64::max)
}

#[must_use]
pub fn best_play(v: &Game) -> f64 {
	best_of(&mut Scratch::default(), v, &rank_plays(v))
}

#[must_use]
pub fn capacity(g: &Game, blind: BlindKind, seeds: &[u64]) -> Capacity {
	capacity_samples(g, blind, seeds).0
}

pub(super) fn capacity_samples(
	g: &Game, blind: BlindKind, seeds: &[u64],
) -> (Capacity, Vec<Sample>) {
	let (mut hands, mut chips) = (0.0, 0.0);
	// The Eye and The Mouth are previewed on whole rounds of the sampled hands,
	// which are kept; otherwise one copy of the game is refreshed for each.
	let whole_rounds = matches!(blind, BlindKind::Eye | BlindKind::Mouth);
	let mut samples = Vec::with_capacity(seeds.len());
	let mut rounds = Vec::with_capacity(if whole_rounds { seeds.len() } else { 0 });
	let mut spare: Option<Game> = None;
	let mut scratch = Scratch::default();
	for &s in seeds {
		let copy = if let Some(mut v) = spare.take() {
			v.copy_from(g, &g.rng);
			v
		} else {
			g.clone()
		};
		let v = virtual_round(copy, blind, s);
		let plays = rank_plays(&v);
		let best = best_of(&mut scratch, &v, &plays);
		hands = v.round.hands_left;
		chips = v.blind.chips;
		samples.push(Sample {
			best,
			plays,
			digest: hand_digest(&v.areas, &v.blind, &v.hands),
		});
		if whole_rounds {
			rounds.push(v);
		} else {
			spare = Some(v);
		}
	}
	let n = num::float(seeds.len().max(1));
	let per_round = num::trunc(hands.max(1.0));
	if !(whole_rounds && per_round > 1 && seeds.len() > 1 && !rounds[0].blind.disabled) {
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
	let n_sampled = rounds.len();
	let mut deals = rounds;
	for i in n_sampled..per_round {
		let again = deals[i % n_sampled].clone();
		deals.push(again);
	}
	let (mut total, mut played, mut clear, mut n_rounds) = (0.0, 0usize, 0usize, 0usize);
	for round in deals.chunks_mut(per_round) {
		let sum = if blind == BlindKind::Eye {
			eye_round(&mut scratch, round)
		} else {
			mouth_round(&mut scratch, round)
		};
		total += sum;
		played += round.len();
		n_rounds += 1;
		clear += usize::from(sum * hands / num::float(round.len()) >= chips); // a short last round counts at full length
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

fn best_typed(scratch: &mut Scratch, v: &Game) -> (f64, Option<HandType>) {
	let flags = v.areas.eval_flags();
	let mut best = (0.0, None);
	for m in rank_plays(v) {
		if let Ok(s) = scratch.copy(v, &v.rng).preview_play(m)
			&& s > best.0
		{
			best = (
				s,
				rough_score(&v.areas, &v.blind, &v.hands, m, flags)
					.core
					.map(|(hand, _)| hand),
			);
		}
	}
	best
}

/// The Eye: no hand type scores twice in a round, so each sampled hand
/// plays its best type that the earlier ones left. Returns the round's score.
fn eye_round(scratch: &mut Scratch, rounds: &mut [Game]) -> f64 {
	let mut used = HandSet::default();
	let mut total = 0.0;
	for v in rounds {
		v.blind.eye_hands = used;
		let (s, t) = best_typed(scratch, v);
		if let Some(t) = t {
			used.insert(t);
		}
		total += s;
	}
	total
}

/// The Mouth: only the first hand's type scores all round. Each sampled
/// hand's best type is tried as that type on every hand; returns the best
/// round's score.
fn mouth_round(scratch: &mut Scratch, rounds: &mut [Game]) -> f64 {
	let mut types: Vec<HandType> = rounds
		.iter()
		.filter_map(|v| best_typed(scratch, v).1)
		.collect();
	types.sort_unstable();
	types.dedup();
	let mut best = 0.0;
	for t in types {
		let mut total = 0.0;
		for v in rounds.iter_mut() {
			v.blind.only_hand = Some(t);
			total += best_typed(scratch, v).0;
		}
		if total > best {
			best = total;
		}
	}
	best
}
