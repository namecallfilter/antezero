//! The observation tensors: entity tokens and global features.

use super::{
	BuildPreview, GLOB_FLOAT, Layout, MAX_HAND, MAX_TOKENS, ObsMut, TOK_FLOAT, TOK_INT,
	TOK_PREVIEW, build_preview, center, zone,
};
use crate::card::{Card, Kind};
use crate::data::{
	ALL_VOUCHERS, BlindKind, Edition, Enhancement, HandType, Joker, Seal, Suit, TagKind,
	blind_amount,
};
use crate::game::{BlindSlot, BlindStatus, Game, Phase};
use crate::{flow, num};

/// `TOK_INT` column of a token's slot (its index within its area).
const SLOT: usize = 7;

// Layout of the continuous globals.
const G_SCALARS: usize = 48;
/// Redeemed vouchers, one bit each.
const G_VOUCHERS: usize = G_SCALARS;
/// Histogram of the unseen pool: suit x rank counts, then enhancement,
/// edition and seal counts, size, stone / debuffed / face-down counts.
const G_DECK: usize = G_VOUCHERS + ALL_VOUCHERS.len();
const G_DECK_ENH: usize = G_DECK + 52;
const G_DECK_EDITION: usize = G_DECK_ENH + Enhancement::ALL.len();
const G_DECK_SEAL: usize = G_DECK_EDITION + Edition::ALL.len();
const G_DECK_SIZE: usize = G_DECK_SEAL + Seal::ALL.len() + 1;
const G_DECK_STONE: usize = G_DECK_SIZE + 1;
const G_DECK_DEBUFFED: usize = G_DECK_SIZE + 2;
const G_FACE_DOWN: usize = G_DECK_SIZE + 3;
/// Build preview (shops, packs, blind select; else zero): against the boss
/// and against the next blind, the log capacity ratio, the ratio (capped at
/// 4) and the clear fraction over the sampled hands. `G_HAS_PREVIEW` (a
/// reserved scalar) is 1 when they are present.
const G_PREVIEW: usize = G_DECK_SIZE + 4;
const G_HAS_PREVIEW: usize = 33;
/// Suit x rank counts of the discard pile.
const G_DISCARD: usize = G_DECK_SIZE + 10;
const _: () =
	assert!(G_DECK_SIZE == 151 && G_PREVIEW + 6 == G_DISCARD && G_DISCARD + 52 <= GLOB_FLOAT);

struct Toks<'b, 'a> {
	o: &'b mut ObsMut<'a>,
	n: usize,
}

impl Toks<'_, '_> {
	fn push(&mut self, ints: [i32; TOK_INT], floats: &[f32]) -> Option<usize> {
		if self.n >= MAX_TOKENS {
			return None;
		}
		let i = self.n;
		self.o.tok_i[i * TOK_INT..(i + 1) * TOK_INT].copy_from_slice(&ints);
		let f = &mut self.o.tok_f[i * TOK_FLOAT..(i + 1) * TOK_FLOAT];
		f.fill(0.0);
		f[..floats.len()].copy_from_slice(floats);
		self.n += 1;
		Some(i)
	}

	fn card(
		&mut self, z: i32, slot: usize, c: &Card, hide: bool, target: Option<(i32, i32)>,
	) -> Option<usize> {
		let (ints, floats) = card_token(z, slot, c, hide, target);
		self.push(ints, &floats)
	}
}

/// The token of a card. `target`: a (suit, rank) shown on the card's text
/// (The Idol, Mail-In Rebate, Ancient Joker, Castle), encoded in the suit /
/// rank columns.
fn card_token(
	z: i32, slot: usize, c: &Card, hide: bool, target: Option<(i32, i32)>,
) -> ([i32; TOK_INT], [f32; TOK_FLOAT]) {
	if hide {
		let mut f = [0.0f32; TOK_FLOAT];
		f[1] = 1.0;
		return ([z, center::HIDDEN, 0, 0, 0, 0, 0, num::id(slot), 0], f);
	}
	let a = &c.ab;
	let playing = c.is_playing();
	let (suit, rank) = if playing && !c.is_stone() {
		(suit_id(c.suit), i32::from(c.rank))
	} else {
		target.unwrap_or((0, 0))
	};
	let aux = match c.kind {
		Kind::Joker(Joker::TodoList) => a.to_do.map_or(0, |h| h as i32 + 1),
		_ => 0,
	};
	let f: [f32; TOK_FLOAT] = [
		bf(c.debuff),
		// Deck-view cards show face up.
		bf(c.face_down && z != zone::DECK),
		num::feat(c.cost / 10.0),
		num::feat(c.sell_cost / 10.0),
		sl(a.mult),
		sl(a.x_mult - 1.0),
		sl(a.chips),
		num::feat(a.dollars / 10.0),
		num::feat(a.extra_value / 10.0),
		sl(a.perma_bonus),
		num::feat(a.remaining / 10.0),
		num::feat(a.invis_rounds / 3.0),
		num::feat(a.yorick_discards / 23.0),
		sl(a.caino_xmult),
		num::feat(a.loyalty_remaining / 5.0),
		bf(a.eternal),
		bf(a.perishable),
		num::feat(a.perish_tally / 5.0),
		bf(a.rental),
		bf(a.forced_selection),
		bf(a.played_this_ante),
		bf(a.couponed),
		(num::feat(f64::from(c.times_played)) / 10.0).min(5.0),
		if playing {
			num::feat(c.enhancement.x_mult()) - 1.0
		} else {
			0.0
		},
		// The deck count the card's text shows (Steel Joker, Stone Joker,
		// Cloud 9, Driver's License; at most one is non-zero).
		num::feat((a.steel_tally + a.stone_tally + a.nine_tally + a.driver_tally) / 10.0),
		// Temperance's current payout.
		num::feat(a.money / 10.0),
		// The item's build preview (`encode_with`).
		0.0,
		0.0,
		0.0,
		0.0,
	];
	let ints = [
		z,
		kind_center(c.kind),
		suit,
		rank,
		if playing { enh_id(c.enhancement) } else { 0 },
		edition_id(c.edition),
		seal_id(c.seal),
		num::id(slot),
		aux,
	];
	(ints, f)
}

/// The cards a player has not seen placed: the draw pile plus face-down hand
/// cards (what the game's deck view lists as remaining).
fn unseen_pool(g: &Game) -> impl Iterator<Item = &Card> {
	g.deck.iter().chain(g.hand.iter().filter(|c| c.face_down))
}

/// Whether anything that affects scoring is face down (hand cards or
/// jokers); previews would then reveal hidden information.
#[must_use]
pub fn hidden_scoring_state(g: &Game) -> bool {
	g.hand.iter().any(|c| c.face_down) || g.jokers.iter().any(|c| c.face_down)
}

/// Writes the observation of `g` (deck and discard pile as histograms
/// only); returns the token layout.
pub fn encode(g: &Game, o: &mut ObsMut<'_>) -> Layout {
	encode_opts(g, o, false)
}

/// Like [`encode`]; with `deck_tokens` the remaining deck and the discard
/// pile are also tokenized card by card (sorted, so order stays hidden).
/// Computes the build preview itself (see [`encode_with`]).
pub fn encode_opts(g: &Game, o: &mut ObsMut<'_>, deck_tokens: bool) -> Layout {
	let mut pv = Vec::new();
	build_preview(g, &mut pv);
	encode_with(g, o, deck_tokens, pv.first_mut())
}

/// [`encode_opts`] with the build preview of `g` (see [`build_preview`]),
/// or none: it fills the preview globals and the preview columns of the
/// item tokens (jokers, consumables, shop cards, vouchers, pack cards).
#[expect(
	clippy::too_many_lines,
	reason = "writes the observation block by block, in tensor order"
)]
#[expect(
	clippy::many_single_char_names,
	reason = "the encoder's short names: game, observation, tokens, card"
)]
pub fn encode_with(
	g: &Game, o: &mut ObsMut<'_>, deck_tokens: bool, mut pv: Option<&mut BuildPreview>,
) -> Layout {
	let mut lay = Layout::of(g);
	let mut t = Toks { o, n: 0 };
	// Hand first: hand index == token index (pointer slots).
	for (i, c) in g.hand.iter().take(MAX_HAND).enumerate() {
		t.card(zone::HAND, i, c, c.face_down, None);
	}
	// An item token with its preview columns.
	let mut item = |t: &mut Toks, z: i32, i: usize, c: &Card, target: Option<(i32, i32)>| {
		let Some(k) = t.card(z, i, c, c.face_down, target) else {
			return;
		};
		if c.kind == Kind::Joker(Joker::Satellite) && !c.face_down {
			// Its payout, a dollar per distinct planet used this run, is on its card text but not in its
			// abilities: shown in the payout column.
			t.o.tok_f[k * TOK_FLOAT + 7] = num::feat(f64::from(g.planets_used.count_ones())) / 10.0;
		}
		if let Some(p) = pv.as_deref_mut()
			&& !c.face_down
			&& let Some(f) = p.item_feats(g, z, i)
		{
			t.o.tok_f[k * TOK_FLOAT + TOK_PREVIEW..(k + 1) * TOK_FLOAT].copy_from_slice(&f);
		}
	};
	for (i, c) in g.jokers.iter().enumerate() {
		let target = match c.kind {
			Kind::Joker(Joker::Idol) => {
				Some((suit_id(g.idol_card.suit), i32::from(g.idol_card.rank)))
			}
			Kind::Joker(Joker::Mail) => Some((0, i32::from(g.mail_rank.rank))),
			Kind::Joker(Joker::Ancient) => Some((g.ancient_suit.map_or(0, suit_id), 0)),
			Kind::Joker(Joker::Castle) => Some((suit_id(g.castle_suit), 0)),
			_ => None,
		};
		item(&mut t, zone::JOKER, i, c, target);
	}
	for (i, c) in g.consumables.iter().enumerate() {
		item(&mut t, zone::CONSUMABLE, i, c, None);
	}
	for (i, c) in g.shop_jokers.iter().enumerate() {
		item(&mut t, zone::SHOP_CARD, i, c, None);
	}
	for (i, c) in g.shop_vouchers.iter().enumerate() {
		item(&mut t, zone::SHOP_VOUCHER, i, c, None);
	}
	for (i, c) in g.shop_boosters.iter().enumerate() {
		t.card(zone::SHOP_BOOSTER, i, c, false, None);
	}
	for (i, c) in g.pack_cards.iter().enumerate() {
		item(&mut t, zone::PACK, i, c, None);
	}
	for (i, tag) in g.tags.iter().enumerate() {
		let f = [bf(tag.triggered)];
		t.push(
			[
				zone::TAG,
				center::TAG0 + tag.kind as i32,
				0,
				0,
				0,
				0,
				0,
				num::id(i),
				tag.orbital_hand.map_or(0, |h| h as i32 + 1),
			],
			&f,
		);
	}
	// The three blinds of the ante.
	for (i, slot) in [BlindSlot::Small, BlindSlot::Big, BlindSlot::Boss]
		.into_iter()
		.enumerate()
	{
		let kind = match slot {
			BlindSlot::Small => BlindKind::Small,
			BlindSlot::Big => BlindKind::Big,
			BlindSlot::Boss => g.boss_choice,
		};
		let d = kind.def();
		let chips =
			blind_amount(g.ante, g.modifiers.scaling) * d.mult * g.starting_params.ante_scaling;
		let dollars = if g.modifiers.no_blind_reward[slot as usize] {
			0.0
		} else {
			f64::from(d.dollars)
		};
		let st = g.blind_states[slot as usize];
		let tag = if i < 2 {
			g.blind_tags[i].map_or(0, |t| t as i32 + 1)
		} else {
			0
		};
		// The poker hand the blind's text names (rank column): an Orbital
		// Tag's hand, The Ox's most played hand, the hand The Mouth allows.
		let hand = match slot {
			BlindSlot::Small | BlindSlot::Big if g.blind_tags[i] == Some(TagKind::Orbital) => {
				flow::orbital_for(g, slot)
			}
			BlindSlot::Boss if kind == BlindKind::Ox => Some(g.most_played_hand),
			BlindSlot::Boss
				if kind == BlindKind::Mouth && g.blind.kind == Some(BlindKind::Mouth) =>
			{
				g.blind.only_hand
			}
			_ => None,
		};
		let f = [
			sl(chips),
			num::feat(dollars / 5.0),
			bf(st == BlindStatus::Select),
			bf(st == BlindStatus::Upcoming),
			bf(st == BlindStatus::Current),
			bf(st == BlindStatus::Defeated),
			bf(st == BlindStatus::Skipped),
			bf(g.blind_on_deck == Some(slot)),
		];
		t.push(
			[
				zone::BLIND,
				center::BLIND0 + kind as i32,
				0,
				hand.map_or(0, |h| h as i32 + 1),
				0,
				0,
				0,
				num::id(i),
				tag,
			],
			&f,
		);
	}
	for h in HandType::ALL {
		let l = &g.hands[h as usize];
		let f = [
			num::feat(l.level / 10.0),
			sl(l.chips),
			sl(l.mult),
			sl(l.played),
			num::feat(l.played_this_round / 3.0),
			bf(l.visible),
		];
		t.push(
			[
				zone::HAND_TYPE,
				center::HAND0 + h as i32,
				0,
				0,
				0,
				0,
				0,
				h as i32,
				0,
			],
			&f,
		);
	}
	debug_assert_eq!(t.n, lay.n_tokens, "Layout::of disagrees with the encoding");
	// Remaining deck and discard pile as sorted multisets. Like the game's
	// deck view, the "remaining" pool includes face-down hand cards, so their
	// identities cannot be deduced. Tokens are ordered by their whole
	// content, so neither draw order nor hidden state breaks ties.
	if deck_tokens {
		let piles: [(i32, Vec<&Card>); 2] = [
			(zone::DECK, unseen_pool(g).collect()),
			(zone::DISCARD, g.discard.iter().collect()),
		];
		for (z, pile) in piles {
			let mut toks: Vec<_> = pile
				.into_iter()
				.map(|c| card_token(z, 0, c, false, None))
				.collect();
			toks.sort_by(|a, b| {
				a.0.cmp(&b.0).then_with(|| {
					a.1.iter()
						.zip(&b.1)
						.map(|(x, y)| x.total_cmp(y))
						.find(|o| o.is_ne())
						.unwrap_or(std::cmp::Ordering::Equal)
				})
			});
			for (k, (mut ints, floats)) in toks.into_iter().enumerate() {
				ints[SLOT] = num::id(k.min(255));
				if t.push(ints, &floats).is_none() {
					break;
				}
			}
		}
	}
	lay.n_tokens = t.n;
	// Padding.
	for i in t.n..MAX_TOKENS {
		t.o.tok_i[i * TOK_INT..(i + 1) * TOK_INT].fill(0);
		t.o.tok_f[i * TOK_FLOAT..(i + 1) * TOK_FLOAT].fill(0.0);
	}
	let o = t.o;

	// Globals.
	let cur_blind = g.blind.kind.map_or(0, |k| k as i32 + 1);
	o.glob_i.copy_from_slice(&[
		phase_id(g.phase),
		i32::from(g.stake),
		g.back as i32,
		cur_blind,
		g.boss_choice as i32 + 1,
		g.blind_tags[0].map_or(0, |t| t as i32 + 1),
		g.blind_tags[1].map_or(0, |t| t as i32 + 1),
		g.last_hand_played.map_or(0, |h| h as i32 + 1),
		g.last_tarot_planet.map_or(0, kind_center),
		g.current_voucher
			.filter(|_| g.voucher_seen)
			.map_or(0, |v| kind_center(Kind::Voucher(v))),
	]);
	let gf = &mut *o.glob_f;
	gf.fill(0.0);
	let target = g.blind.chips.max(1.0);
	let vals: [f32; G_SCALARS] = [
		num::feat(f64::from(g.ante)) / 8.0,
		num::feat(f64::from(g.round)) / 24.0,
		sl(g.blind.chips),
		sl(g.chips),
		num::feat((g.chips / target).min(4.0)),
		num::feat(g.hands_left) / 5.0,
		num::feat(g.discards_left) / 5.0,
		num::feat(g.dollars / 25.0),
		sl(g.dollars),
		num::feat(g.hand_limit.limit) / 10.0,
		num::feat(g.joker_limit.limit) / 5.0,
		num::feat(num::float(g.jokers.len())) / 5.0,
		num::feat(g.consumable_limit.limit) / 2.0,
		num::feat(num::float(g.consumables.len())) / 2.0,
		num::feat(g.reroll_cost / 5.0),
		num::feat(g.free_rerolls),
		// Green Deck pays no interest whatever the cap says.
		if g.modifiers.no_interest {
			0.0
		} else {
			num::feat(g.interest_cap / 25.0)
		},
		num::feat(g.skips) / 5.0,
		num::feat(num::float(g.deck.len())) / 52.0,
		num::feat(num::float(g.discard.len())) / 52.0,
		num::feat(num::float(g.playing_cards().count())) / 52.0,
		num::feat(g.hands_played_round) / 4.0,
		num::feat(g.discards_used) / 4.0,
		f32::from(g.pack_choices),
		num::feat(g.round_dollars / 10.0),
		bf(g.run_won()),
		bf(g.endless),
		bf(g.boss_rerolled),
		bf(g.blind.disabled),
		bf(g.blind.boss),
		num::feat(g.probabilities) / 2.0,
		num::feat(g.bankrupt_at / 20.0),
		num::feat(g.discount_percent / 50.0),
		0.0, // G_HAS_PREVIEW (was G.GAME.inflation, which only challenges change)
		num::feat(g.unused_discards / 20.0),
		num::feat(g.hands_played) / 50.0,
		if g.modifiers.no_interest {
			0.0
		} else {
			num::feat(g.interest_amount)
		},
		num::feat(g.rr_hands / 4.0),
		num::feat(g.rr_discards / 3.0),
		num::feat(g.ecto_minus) / 3.0,
		bf(g.hand_sort_by_suit),
		num::feat(g.shop_joker_max / 2.0),
		bf(g.gros_michel_extinct),
		f32::from(g.pack_size) / 5.0,
		num::feat(g.consumable_usage_tarot / 10.0),
		num::feat(g.consumable_usage_planet / 10.0),
		num::feat(g.consumable_usage_spectral / 5.0),
		bf(g.shop_free),
	];
	gf[..G_SCALARS].copy_from_slice(&vals);
	for (i, v) in ALL_VOUCHERS.iter().enumerate() {
		gf[G_VOUCHERS + i] = bf(g.voucher_used(*v));
	}
	let card_slot = |c: &Card| -> Option<usize> {
		if c.is_stone() || !c.is_playing() {
			return None;
		}
		Some(c.suit as usize * 13 + usize::from(c.rank).saturating_sub(2).min(12))
	};
	let (mut size, mut stone, mut debuffed) = (0, 0, 0);
	for c in unseen_pool(g) {
		if let Some(i) = card_slot(c) {
			gf[G_DECK + i] += 0.5;
		}
		gf[G_DECK_ENH + c.enhancement as usize] += 0.1;
		gf[G_DECK_EDITION + c.edition as usize] += 0.1;
		gf[G_DECK_SEAL + c.seal.map_or(0, |s| s as usize + 1)] += 0.1;
		size += 1;
		stone += usize::from(c.is_stone());
		debuffed += usize::from(c.debuff);
	}
	gf[G_DECK_SIZE] = num::feat(f64::from(size)) / 52.0;
	gf[G_DECK_STONE] = num::feat(num::float(stone)) / 10.0;
	gf[G_DECK_DEBUFFED] = num::feat(num::float(debuffed)) / 10.0;
	gf[G_FACE_DOWN] = num::feat(num::float(g.hand.iter().filter(|c| c.face_down).count())) / 8.0;
	if let Some(p) = pv.as_deref() {
		for (k, c) in [p.cap_boss, p.cap_next].iter().enumerate() {
			gf[G_PREVIEW + 3 * k] = c.log_ratio();
			gf[G_PREVIEW + 3 * k + 1] = num::feat(c.ratio().min(4.0));
			gf[G_PREVIEW + 3 * k + 2] = num::feat(c.p_clear);
		}
		gf[G_HAS_PREVIEW] = 1.0;
	}
	for c in &g.discard {
		if let Some(i) = card_slot(c) {
			gf[G_DISCARD + i] += 0.5;
		}
	}
	lay
}

/// Zones / phases / enums as small integers.
pub(super) fn phase_id(p: Phase) -> i32 {
	match p {
		Phase::BlindSelect => 0,
		Phase::Round => 1,
		Phase::Shop => 2,
		Phase::Pack => 3,
		Phase::RoundEval => 4,
		Phase::GameOver => 5,
	}
}

pub(super) fn kind_center(k: Kind) -> i32 {
	use center::{BOOSTER0, JOKER0, PLANET0, PLAYING, SPECTRAL0, TAROT0, VOUCHER0};
	match k {
		Kind::Playing => PLAYING,
		Kind::Joker(j) => JOKER0 + j as i32,
		Kind::Tarot(t) => TAROT0 + t as i32,
		Kind::Planet(p) => PLANET0 + p as i32,
		Kind::Spectral(s) => SPECTRAL0 + s as i32,
		Kind::Voucher(v) => VOUCHER0 + v as i32,
		Kind::Booster(b) => BOOSTER0 + b as i32,
	}
}

// Categorical ids (0 = none).
pub(super) fn suit_id(s: Suit) -> i32 {
	s as i32 + 1
}

pub(super) fn enh_id(e: Enhancement) -> i32 {
	e as i32
}

pub(super) fn edition_id(e: Edition) -> i32 {
	e as i32
}

pub(super) fn seal_id(s: Option<Seal>) -> i32 {
	s.map_or(0, |s| s as i32 + 1)
}

/// Signed log scaling for unbounded quantities.
#[inline]
pub(super) fn sl(x: f64) -> f32 {
	num::feat(x.signum() * x.abs().ln_1p())
}

#[inline]
pub(super) fn bf(b: bool) -> f32 {
	if b { 1.0 } else { 0.0 }
}
