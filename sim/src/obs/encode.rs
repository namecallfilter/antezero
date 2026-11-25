//! Observation encoding: top-level entry points and global feature vectors.

use std::ops::Range;

use super::layout::phase_id;
use super::tokens::{
	TokenWriter, blind_tokens, card_tokens, flag, hand_id, hand_type_tokens, pile_tokens,
	seen_discards, signed_log, tag_id, tag_tokens, unseen_pool,
};
use super::{
	BuildPreview, GLOB_FLOAT, Layout, MAX_TOKENS, ObsMut, TOK_FLOAT, TOK_INT, build_preview, center,
};
use crate::card::{Kind, PlayingCard};
use crate::data::{ALL_VOUCHERS, Edition, Enhancement, Rank, Seal};
use crate::game::{Areas, Game, HandSort};
use crate::num;

const G_SCALARS: usize = 48;
/// Redeemed vouchers, one bit each.
const G_VOUCHERS: usize = G_SCALARS;
/// Histogram of the unseen pool: suit x rank counts, then enhancement,
/// edition and seal counts, size, stone / debuffed / face-down counts.
const G_DECK: usize = G_VOUCHERS + ALL_VOUCHERS.len();
const G_DECK_ENH: usize = G_DECK + 52;
const G_DECK_EDITION: usize = G_DECK_ENH + Enhancement::ALL.len();
const G_DECK_SEAL: usize = G_DECK_EDITION + Edition::ID_COUNT;
const G_DECK_SIZE: usize = G_DECK_SEAL + Seal::ALL.len() + 1;
const G_DECK_STONE: usize = G_DECK_SIZE + 1;
const G_DECK_DEBUFFED: usize = G_DECK_SIZE + 2;
const G_FACE_DOWN: usize = G_DECK_SIZE + 3;
/// The `GLOB_FLOAT` columns of the build preview (shops, packs, blind
/// select; else zero): [`CAPACITY_FLOATS`] columns against the boss, then
/// as many against the next blind.
pub const GLOB_PREVIEW: Range<usize> = G_FACE_DOWN + 1..G_FACE_DOWN + 1 + 2 * CAPACITY_FLOATS;
/// Columns per previewed capacity: the log capacity ratio, the ratio
/// (capped at 4) and the clear fraction over the sampled hands.
pub const CAPACITY_FLOATS: usize = 3;
/// The `GLOB_FLOAT` column that is 1 when the build preview is present.
pub const GLOB_HAS_PREVIEW: usize = 33;
/// Suit x rank counts of the discard pile.
const G_DISCARD: usize = GLOB_PREVIEW.end;
const _: () = assert!(G_DECK_SIZE == 151 && G_DISCARD + 52 <= GLOB_FLOAT);

/// Whether anything that affects scoring is face down (hand cards or
/// jokers); previews would then reveal hidden information.
#[must_use]
pub fn hidden_scoring_state(areas: &Areas) -> bool {
	areas.hand.cards.iter().any(|c| c.core.face_down)
		|| areas.jokers.iter().any(|c| c.core.face_down)
}

/// Writes the observation of `g` (deck and discard pile as histograms
/// only); returns the token layout.
pub fn encode(g: &Game, o: &mut ObsMut<'_>) -> Layout {
	encode_opts(g, o, /* deck_tokens= */ false)
}

/// Like [`encode`]; with `deck_tokens` the remaining deck and the discard
/// pile are also tokenized card by card (sorted, so order stays hidden).
/// Computes the build preview itself (see [`encode_with`]).
pub fn encode_opts(g: &Game, o: &mut ObsMut<'_>, deck_tokens: bool) -> Layout {
	let mut cache = Vec::new();
	let preview = build_preview(g, &mut cache);
	encode_with(g, o, deck_tokens, preview)
}

/// [`encode_opts`] with the build preview of `g` (see [`build_preview`]),
/// or none: it fills the preview globals and the preview columns of the
/// item tokens (jokers, consumables, shop cards, vouchers, pack cards).
pub fn encode_with(
	g: &Game, o: &mut ObsMut<'_>, deck_tokens: bool, mut pv: Option<&mut BuildPreview>,
) -> Layout {
	let mut lay = Layout::of(&g.areas, &g.tags);
	let mut toks = TokenWriter { obs: o, len: 0 };
	card_tokens(&mut toks, g, &lay, pv.as_deref_mut());
	tag_tokens(&mut toks, &g.tags);
	blind_tokens(&mut toks, g);
	hand_type_tokens(&mut toks, &g.hands);
	debug_assert_eq!(
		toks.len, lay.n_tokens,
		"Layout::of disagrees with the encoding"
	);
	if deck_tokens {
		pile_tokens(&mut toks, &g.areas);
	}
	lay.n_tokens = toks.len;
	for i in toks.len..MAX_TOKENS {
		toks.obs.tok_i[i * TOK_INT..(i + 1) * TOK_INT].fill(0);
		toks.obs.tok_f[i * TOK_FLOAT..(i + 1) * TOK_FLOAT].fill(0.0);
	}
	let o = toks.obs;
	global_ints(g, o.glob_i);
	global_floats(g, o.glob_f, pv.as_deref());
	lay
}

fn global_ints(g: &Game, glob_i: &mut [i32]) {
	glob_i.copy_from_slice(&[
		phase_id(g.flow.phase),
		i32::from(u8::from(g.setup.stake)),
		g.setup.back as i32,
		g.blind.kind.map_or(0, |k| k as i32 + 1),
		g.blind_select.boss as i32 + 1,
		tag_id(g.blind_select.skip_tags.small),
		tag_id(g.blind_select.skip_tags.big),
		hand_id(g.history.last_hand_played),
		g.history.last_tarot_planet.map_or(0, center::of_kind),
		g.shop
			.voucher
			.filter(|_| g.shop.voucher_seen)
			.map_or(0, |v| center::of_kind(Kind::Voucher(v))),
	]);
}

/// The scalar global floats before `GLOB_HAS_PREVIEW` and after it (to
/// `G_SCALARS`).
fn global_scalars(
	g: &Game,
) -> (
	[f32; GLOB_HAS_PREVIEW],
	[f32; G_SCALARS - GLOB_HAS_PREVIEW - 1],
) {
	let target = g.blind.chips.max(1.0);
	let before = [
		num::feat(f64::from(g.run.ante)) / 8.0,
		num::feat(f64::from(g.run.round)) / 24.0,
		signed_log(g.blind.chips),
		signed_log(g.round.chips),
		num::feat((g.round.chips / target).min(4.0)),
		num::feat(g.round.hands_left) / 5.0,
		num::feat(g.round.discards_left) / 5.0,
		num::feat(g.run.dollars / 25.0),
		signed_log(g.run.dollars),
		num::feat(g.areas.hand.limit.limit) / 10.0,
		num::feat(g.areas.limits.jokers.limit) / 5.0,
		num::feat(num::float(g.areas.jokers.len())) / 5.0,
		num::feat(g.areas.limits.consumables.limit) / 2.0,
		num::feat(num::float(g.areas.consumables.len())) / 2.0,
		num::feat(g.shop.reroll_cost / 5.0),
		num::feat(g.shop.free_rerolls),
		// Green Deck pays no interest whatever the cap says.
		if g.setup.modifiers.no_interest {
			0.0
		} else {
			num::feat(g.run.interest_cap / 25.0)
		},
		num::feat(g.run.skips) / 5.0,
		num::feat(num::float(g.areas.deck.len())) / 52.0,
		num::feat(num::float(g.areas.discard.len())) / 52.0,
		num::feat(num::float(g.areas.playing_cards().count())) / 52.0,
		num::feat(g.round.hands_played) / 4.0,
		num::feat(g.round.discards_used) / 4.0,
		f32::from(g.pack.choices.unwrap_or(0)),
		num::feat(g.round.cash_out / 10.0),
		flag(g.run_won()),
		flag(g.setup.endless),
		flag(g.blind_select.boss_rerolled),
		flag(g.blind.disabled),
		flag(g.blind.is_boss()),
		num::feat(g.run.probabilities) / 2.0,
		num::feat(g.run.bankrupt_at / 20.0),
		num::feat(g.shop.discount_percent / 50.0),
	];
	let after = [
		num::feat(g.run.unused_discards / 20.0),
		num::feat(g.run.hands_played) / 50.0,
		if g.setup.modifiers.no_interest {
			0.0
		} else {
			num::feat(g.run.interest_amount)
		},
		num::feat(g.round.base_hands / 4.0),
		num::feat(g.round.base_discards / 3.0),
		num::feat(g.run.ecto_minus) / 3.0,
		flag(g.areas.hand.order == HandSort::Suit),
		num::feat(g.shop.joker_max / 2.0),
		flag(g.history.gros_michel_extinct),
		f32::from(g.pack.size) / 5.0,
		num::feat(g.history.consumable_usage_tarot / 10.0),
		num::feat(g.history.consumable_usage_planet / 10.0),
		num::feat(g.history.consumable_usage_spectral / 5.0),
		flag(g.shop.free),
	];
	(before, after)
}

/// The column of a card in a suit x rank histogram (a Stone card has none).
fn card_index(card: &PlayingCard) -> Option<usize> {
	(!card.is_stone()).then(|| card.suit as usize * Rank::ALL.len() + card.rank.index())
}

/// The global floats: scalars, voucher bits, the histograms of the unseen
/// pool and the discard pile, and the build preview's capacities.
fn global_floats(g: &Game, gf: &mut [f32], pv: Option<&BuildPreview>) {
	gf.fill(0.0);
	let (before, after) = global_scalars(g);
	gf[..GLOB_HAS_PREVIEW].copy_from_slice(&before);
	gf[GLOB_HAS_PREVIEW] = flag(pv.is_some());
	gf[GLOB_HAS_PREVIEW + 1..G_SCALARS].copy_from_slice(&after);
	for (i, v) in ALL_VOUCHERS.iter().enumerate() {
		gf[G_VOUCHERS + i] = flag(g.history.used_vouchers.contains(*v));
	}
	let (mut size, mut stone, mut debuffed): (usize, usize, usize) = (0, 0, 0);
	for card in unseen_pool(&g.areas) {
		if let Some(i) = card_index(card) {
			gf[G_DECK + i] += 0.5;
		}
		gf[G_DECK_ENH + card.enhancement as usize] += 0.1;
		gf[G_DECK_EDITION + usize::from(Edition::id(card.core.edition))] += 0.1;
		gf[G_DECK_SEAL + card.seal.map_or(0, |s| s as usize + 1)] += 0.1;
		size += 1;
		stone += usize::from(card.is_stone());
		debuffed += usize::from(card.core.debuff);
	}
	gf[G_DECK_SIZE] = num::feat(num::float(size)) / 52.0;
	gf[G_DECK_STONE] = num::feat(num::float(stone)) / 10.0;
	gf[G_DECK_DEBUFFED] = num::feat(num::float(debuffed)) / 10.0;
	gf[G_FACE_DOWN] = num::feat(num::float(
		g.areas
			.hand
			.cards
			.iter()
			.filter(|c| c.core.face_down)
			.count(),
	)) / 8.0;
	if let Some(preview) = pv {
		for (k, capacity) in [preview.cap_boss, preview.cap_next].iter().enumerate() {
			let at = GLOB_PREVIEW.start + CAPACITY_FLOATS * k;
			gf[at..at + CAPACITY_FLOATS].copy_from_slice(&[
				capacity.log_ratio(),
				capacity.ratio_feat(),
				num::feat(capacity.p_clear),
			]);
		}
	}
	for card in seen_discards(&g.areas) {
		if let Some(i) = card_index(card) {
			gf[G_DISCARD + i] += 0.5;
		}
	}
}
