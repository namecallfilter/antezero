use std::cmp::Ordering;
use std::ops::Range;

use super::layout::{SLOT_VOCAB, TokInts, phase_id, shown_hand};
use super::{
	BuildPreview, GLOB_FLOAT, ItemZone, Layout, MAX_TOKENS, ObsMut, TOK_FLOAT, TOK_INT,
	TOK_PREVIEW, Zone, build_preview, center,
};
use crate::blind::blind_amount;
use crate::card::{Ability, AnyCard, Card, CardCore, CardLike, Kind};
use crate::data::{
	ALL_VOUCHERS, BlindKind, Edition, Enhancement, HandType, Joker, Rank, Seal, Suit, TagKind,
};
use crate::game::{Areas, BlindSlot, BlindStatus, Game, Hand, HandLevels, HandSort};
use crate::playing_card::PlayingCard;
use crate::round::Round;
use crate::tags::Tags;
use crate::{card_lifecycle, flow, num};

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

/// Writes tokens into the observation, in order.
struct TokenWriter<'buf, 'obs> {
	obs: &'buf mut ObsMut<'obs>,
	/// Tokens written so far.
	len: usize,
}

impl TokenWriter<'_, '_> {
	fn push(&mut self, ints: TokInts, floats: &[f32]) -> Option<usize> {
		if self.len >= MAX_TOKENS {
			return None;
		}
		let i = self.len;
		self.obs.tok_i[i * TOK_INT..(i + 1) * TOK_INT].copy_from_slice(&ints.to_array());
		let f = &mut self.obs.tok_f[i * TOK_FLOAT..(i + 1) * TOK_FLOAT];
		f.fill(0.0);
		f[..floats.len()].copy_from_slice(floats);
		self.len += 1;
		Some(i)
	}

	fn token(&mut self, t: &Token) -> Option<usize> {
		self.push(t.ints, &t.floats.to_array())
	}
}

/// The `TOK_FLOAT` columns of a card token before its build preview
/// (`TOK_PREVIEW..`, see `encode_with`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct CardFeats {
	debuff: f32,
	face_down: f32,
	cost: f32,
	sell_cost: f32,
	mult: f32,
	x_mult: f32,
	chips: f32,
	/// The payout the card shows.
	dollars: f32,
	extra_value: f32,
	perma_bonus: f32,
	extra: f32,
	invis_rounds: f32,
	yorick_discards: f32,
	caino_xmult: f32,
	loyalty_remaining: f32,
	eternal: f32,
	perishable: f32,
	perish_tally: f32,
	rental: f32,
	forced_selection: f32,
	played_this_ante: f32,
	couponed: f32,
	times_played: f32,
	/// The enhancement's `x_mult`, less 1.
	enhancement_x_mult: f32,
	/// The deck count the card's text shows (Steel Joker, Stone Joker,
	/// Cloud 9, Driver's License; at most one is non-zero).
	deck_count: f32,
	/// Temperance's current payout.
	money: f32,
}

impl CardFeats {
	fn to_array(self) -> [f32; TOK_PREVIEW] {
		[
			self.debuff,
			self.face_down,
			self.cost,
			self.sell_cost,
			self.mult,
			self.x_mult,
			self.chips,
			self.dollars,
			self.extra_value,
			self.perma_bonus,
			self.extra,
			self.invis_rounds,
			self.yorick_discards,
			self.caino_xmult,
			self.loyalty_remaining,
			self.eternal,
			self.perishable,
			self.perish_tally,
			self.rental,
			self.forced_selection,
			self.played_this_ante,
			self.couponed,
			self.times_played,
			self.enhancement_x_mult,
			self.deck_count,
			self.money,
		]
	}
}

/// A card's token.
#[derive(Clone, Copy, Debug)]
struct Token {
	ints: TokInts,
	floats: CardFeats,
}

/// The token columns that depend on the card's type (zero where it has
/// nothing to show).
#[derive(Default)]
struct Columns {
	center: i32,
	suit: i32,
	rank: i32,
	aux: i32,
	enhancement: i32,
	seal: i32,
	perma_bonus: f64,
	forced_selection: bool,
	played_this_ante: bool,
	times_played: u32,
	/// Already a feature: the enhancement's `x_mult` less 1 (0 for a card
	/// without one).
	enhancement_x_mult: f32,
}

/// The token of a card with the type's columns `c` and, unless it is a
/// playing card, the ability `ability` (a playing card's ability columns stay
/// zero). A face-down card shows only that it is one, except in the piles,
/// which list their cards as the deck view does.
fn token(z: Zone, slot: usize, core: &CardCore, ability: Option<&Ability>, c: &Columns) -> Token {
	if core.face_down && !z.is_pile() {
		return Token {
			ints: TokInts {
				center: center::HIDDEN,
				..TokInts::new(z, slot)
			},
			floats: CardFeats {
				face_down: 1.0,
				..CardFeats::default()
			},
		};
	}
	let mut floats = CardFeats {
		debuff: flag(core.debuff),
		// Deck-view cards show face up.
		face_down: flag(core.face_down && z != Zone::Deck),
		cost: num::feat(core.cost / 10.0),
		sell_cost: num::feat(core.sell_cost / 10.0),
		perma_bonus: signed_log(c.perma_bonus),
		forced_selection: flag(c.forced_selection),
		played_this_ante: flag(c.played_this_ante),
		couponed: flag(core.couponed),
		times_played: (num::feat(f64::from(c.times_played)) / 10.0).min(5.0),
		enhancement_x_mult: c.enhancement_x_mult,
		..CardFeats::default()
	};
	if let Some(a) = ability {
		floats.mult = signed_log(a.mult);
		floats.x_mult = signed_log(a.x_mult - 1.0);
		floats.chips = signed_log(a.chips);
		floats.dollars = num::feat(a.dollars / 10.0);
		floats.extra_value = num::feat(a.extra_value / 10.0);
		floats.extra = num::feat(a.extra / 10.0);
		floats.invis_rounds = num::feat(a.invis_rounds / 3.0);
		floats.yorick_discards = num::feat(a.yorick_discards / card_lifecycle::YORICK_DISCARDS);
		floats.caino_xmult = signed_log(a.caino_xmult);
		floats.loyalty_remaining = num::feat(a.loyalty_remaining / 5.0);
		floats.eternal = flag(a.eternal());
		floats.perishable = flag(a.perishable());
		floats.perish_tally = num::feat(a.perish_tally() / 5.0);
		floats.rental = flag(a.rental);
		floats.deck_count =
			num::feat((a.steel_tally + a.stone_tally + a.nine_tally + a.driver_tally) / 10.0);
		floats.money = num::feat(a.money / 10.0);
	}
	let ints = TokInts {
		center: c.center,
		suit: c.suit,
		rank: c.rank,
		enhancement: c.enhancement,
		edition: edition_id(core.edition),
		seal: c.seal,
		aux: c.aux,
		..TokInts::new(z, slot)
	};
	Token { ints, floats }
}

fn playing_token(z: Zone, slot: usize, c: &PlayingCard) -> Token {
	let (suit, rank) = if c.is_stone() {
		(0, 0)
	} else {
		(suit_id(c.suit), i32::from(c.rank.id()))
	};
	let columns = Columns {
		center: center::PLAYING,
		suit,
		rank,
		aux: 0,
		enhancement: enhancement_id(c.enhancement),
		seal: seal_id(c.seal),
		perma_bonus: c.perma_bonus,
		forced_selection: c.forced_selection,
		played_this_ante: c.played_this_ante,
		times_played: c.times_played,
		enhancement_x_mult: num::feat(c.enhancement.x_mult()) - 1.0,
	};
	token(z, slot, &c.core, None, &columns)
}

/// `target`: a (suit, rank) shown on the card's text
/// (The Idol, Mail-In Rebate, Ancient Joker, Castle), encoded in the suit /
/// rank columns.
fn card_token(z: Zone, slot: usize, c: &Card, target: Option<(i32, i32)>) -> Token {
	let (suit, rank) = target.unwrap_or((0, 0));
	let aux = match c.kind {
		Kind::Joker(Joker::TodoList) => hand_id(c.ab.to_do),
		_ => 0,
	};
	let columns = Columns {
		center: center::of_kind(c.kind),
		suit,
		rank,
		aux,
		..Columns::default()
	};
	token(z, slot, &c.core, Some(&c.ab), &columns)
}

/// The `target` of a joker's token (see [`card_token`]), owned or on offer.
fn joker_target(round: &Round, c: &Card) -> Option<(i32, i32)> {
	match c.kind {
		Kind::Joker(Joker::Idol) => Some((
			suit_id(round.idol_card.suit),
			i32::from(round.idol_card.rank.id()),
		)),
		Kind::Joker(Joker::Mail) => Some((0, i32::from(round.mail_rank.rank.id()))),
		Kind::Joker(Joker::Ancient) => Some((suit_id(round.ancient_suit), 0)),
		Kind::Joker(Joker::Castle) => Some((suit_id(round.castle_suit), 0)),
		_ => None,
	}
}

fn any_token(z: Zone, slot: usize, c: &AnyCard, round: &Round) -> Token {
	match c {
		AnyCard::Playing(c) => playing_token(z, slot, c),
		AnyCard::Card(c) => card_token(z, slot, c, joker_target(round, c)),
	}
}

/// The cards a player has not seen placed: the draw pile plus face-down hand
/// cards (what the game's deck view lists as remaining).
fn unseen_pool(areas: &Areas) -> impl Iterator<Item = &PlayingCard> {
	areas
		.deck
		.iter()
		.chain(areas.hand.cards.iter().filter(|c| c.core.face_down))
		.chain(areas.discard.iter().filter(|c| c.wheel_flipped))
}

/// The discard pile but for the cards discarded unseen (still
/// `ability.wheel_flipped`, which the game's deck view counts as remaining).
fn seen_discards(areas: &Areas) -> impl Iterator<Item = &PlayingCard> {
	areas.discard.iter().filter(|c| !c.wheel_flipped)
}

/// Whether sorting the hand by its current order would change nothing the
/// player sees: every card is face up, the cards stand in the order of what
/// the sort reads of them that shows, and the ones that order cannot tell
/// apart look the same (the sort orders those by what does not show: the
/// suit a card was dealt with, when it was made).
pub(crate) fn null_sort(hand: &Hand) -> bool {
	let order = hand.order;
	let looks = |c: &PlayingCard| {
		let t = playing_token(Zone::Hand, 0, c);
		(t.ints, t.floats)
	};
	hand.cards.iter().all(|c| !c.core.face_down)
		&& hand.cards.windows(2).all(|pair| {
			let (a, b) = (pair[0].shown_sort_key(order), pair[1].shown_sort_key(order));
			a > b || (a == b && looks(&pair[0]) == looks(&pair[1]))
		})
}

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

/// The tokens of the hand (first: a hand index is its token index) and of
/// the items: jokers, consumables, shop cards, vouchers, boosters and pack
/// cards, with their build previews from `pv`. Each area starts where `lay`
/// says.
fn card_tokens(toks: &mut TokenWriter, g: &Game, lay: &Layout, mut pv: Option<&mut BuildPreview>) {
	let at = |toks: &TokenWriter, start: usize| {
		debug_assert_eq!(toks.len, start, "Layout::of puts this area elsewhere");
	};
	for (i, card) in shown_hand(&g.areas.hand.cards).iter().enumerate() {
		toks.token(&playing_token(Zone::Hand, i, card));
	}
	let mut item =
		|toks: &mut TokenWriter, z: ItemZone, i: usize, card: &dyn CardLike, mut token: Token| {
			let face_down = card.core().face_down;
			if card.as_card().is_some_and(|c| c.is_joker(Joker::Satellite)) && !face_down {
				// Its payout, a dollar per distinct planet used this run, is on its card text but
				// not in its abilities: shown in the payout column.
				token.floats.dollars = num::feat(f64::from(g.history.planets_used.count())) / 10.0;
			}
			let Some(k) = toks.token(&token) else {
				return;
			};
			if let Some(preview) = pv.as_deref_mut()
				&& !face_down
				&& let Some(feats) = preview.item_feats(g, z, i)
			{
				toks.obs.tok_f[k * TOK_FLOAT + TOK_PREVIEW..(k + 1) * TOK_FLOAT]
					.copy_from_slice(&feats.to_columns());
			}
		};
	at(toks, lay.jokers);
	for (i, card) in g.areas.jokers.iter().enumerate() {
		let target = joker_target(&g.round, card);
		let z = ItemZone::Joker;
		item(toks, z, i, card, card_token(z.into(), i, card, target));
	}
	at(toks, lay.consumables);
	for (i, card) in g.areas.consumables.iter().enumerate() {
		let z = ItemZone::Consumable;
		item(toks, z, i, card, card_token(z.into(), i, card, None));
	}
	at(toks, lay.shop_cards);
	for (i, card) in g.areas.shop_jokers.iter().enumerate() {
		let z = ItemZone::ShopCard;
		item(toks, z, i, card, any_token(z.into(), i, card, &g.round));
	}
	at(toks, lay.shop_vouchers);
	for (i, card) in g.areas.shop_vouchers.iter().enumerate() {
		let z = ItemZone::ShopVoucher;
		item(toks, z, i, card, card_token(z.into(), i, card, None));
	}
	at(toks, lay.shop_boosters);
	for (i, card) in g.areas.shop_boosters.iter().enumerate() {
		toks.token(&card_token(Zone::ShopBooster, i, card, None));
	}
	at(toks, lay.pack);
	for (i, card) in g.areas.pack_cards.iter().enumerate() {
		let z = ItemZone::Pack;
		item(toks, z, i, card, any_token(z.into(), i, card, &g.round));
	}
}

fn tag_tokens(toks: &mut TokenWriter, tags: &Tags) {
	for (i, tag) in tags.held.iter().enumerate() {
		let ints = TokInts {
			center: center::TAG0 + tag.kind as i32,
			aux: hand_id(tag.orbital_hand),
			..TokInts::new(Zone::Tag, i)
		};
		toks.push(ints, &[flag(tag.triggered)]);
	}
}

/// The ante's blinds.
fn blind_tokens(toks: &mut TokenWriter, g: &Game) {
	for (i, slot) in BlindSlot::ALL.into_iter().enumerate() {
		let kind = g.blind_select.kind(slot);
		let def = kind.def();
		let chips = blind_amount(g.run.ante, g.setup.modifiers.scaling)
			* def.mult
			* g.setup.starting_params.ante_scaling;
		let dollars = if g.setup.modifiers.no_blind_reward[slot] {
			0.0
		} else {
			f64::from(def.dollars)
		};
		let status = g.blind_select.states[slot];
		// The poker hand the blind's text names (rank column): an Orbital
		// Tag's hand, The Ox's most played hand, the hand The Mouth allows.
		let hand = match slot {
			BlindSlot::Small | BlindSlot::Big
				if g.blind_select.skip_tags.of(slot) == Some(TagKind::Orbital) =>
			{
				flow::orbital_for(g.run.ante, &g.tags, slot)
			}
			BlindSlot::Boss if kind == BlindKind::Ox => Some(g.round.most_played_hand),
			BlindSlot::Boss
				if kind == BlindKind::Mouth && g.blind.kind == Some(BlindKind::Mouth) =>
			{
				g.blind.only_hand
			}
			_ => None,
		};
		let ints = TokInts {
			center: center::BLIND0 + kind as i32,
			rank: hand_id(hand),
			aux: tag_id(g.blind_select.skip_tags.of(slot)),
			..TokInts::new(Zone::Blind, i)
		};
		toks.push(ints, &[
			signed_log(chips),
			num::feat(dollars / 5.0),
			flag(status == BlindStatus::Select),
			flag(status == BlindStatus::Upcoming),
			flag(status == BlindStatus::Current),
			flag(status == BlindStatus::Defeated),
			flag(status == BlindStatus::Skipped),
			flag(g.blind_select.on_deck == Some(slot)),
		]);
	}
}

/// The poker hands and their levels.
fn hand_type_tokens(toks: &mut TokenWriter, hands: &HandLevels) {
	for hand in HandType::ALL {
		let level = &hands[hand];
		let ints = TokInts {
			center: center::HAND0 + hand as i32,
			..TokInts::new(Zone::HandType, hand as usize)
		};
		toks.push(ints, &[
			num::feat(level.level / 10.0),
			signed_log(level.chips),
			signed_log(level.mult),
			signed_log(level.played),
			num::feat(level.played_this_round / 3.0),
			flag(level.visible),
		]);
	}
}

/// The remaining deck and the discard pile as sorted multisets. Like the
/// game's deck view, the "remaining" pool includes the face-down hand cards
/// and the cards discarded unseen, so their identities cannot be deduced.
/// Tokens are ordered by their whole content, so neither draw order nor
/// hidden state breaks ties.
fn pile_tokens(toks: &mut TokenWriter, areas: &Areas) {
	pile(toks, Zone::Deck, unseen_pool(areas));
	pile(toks, Zone::Discard, seen_discards(areas));
}

fn pile<'cards>(toks: &mut TokenWriter, z: Zone, cards: impl Iterator<Item = &'cards PlayingCard>) {
	let mut tokens: Vec<Token> = cards.map(|card| playing_token(z, 0, card)).collect();
	tokens.sort_by(|a, b| {
		a.ints.cmp(&b.ints).then_with(|| {
			a.floats
				.to_array()
				.iter()
				.zip(&b.floats.to_array())
				.map(|(x, y)| x.total_cmp(y))
				.find(|order| order.is_ne())
				.unwrap_or(Ordering::Equal)
		})
	});
	for (k, mut token) in tokens.into_iter().enumerate() {
		token.ints.slot = num::id(k.min(SLOT_VOCAB - 1));
		if toks.token(&token).is_none() {
			break;
		}
	}
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

// Categorical ids (0 = none).
fn suit_id(s: Suit) -> i32 {
	s as i32 + 1
}

fn enhancement_id(e: Enhancement) -> i32 {
	e as i32
}

fn edition_id(e: Option<Edition>) -> i32 {
	i32::from(Edition::id(e))
}

fn seal_id(s: Option<Seal>) -> i32 {
	s.map_or(0, |s| s as i32 + 1)
}

/// The categorical id of a poker hand in a token or global column (0 is
/// none).
#[must_use]
pub fn hand_id(h: Option<HandType>) -> i32 {
	h.map_or(0, |h| h as i32 + 1)
}

fn tag_id(t: Option<TagKind>) -> i32 {
	t.map_or(0, |t| t as i32 + 1)
}

/// Signed log scaling for unbounded quantities.
#[inline]
pub(super) fn signed_log(x: f64) -> f32 {
	num::feat(x.signum() * x.abs().ln_1p())
}

/// A bool as a 0 / 1 feature.
#[inline]
fn flag(b: bool) -> f32 {
	if b { 1.0 } else { 0.0 }
}
