//! The observation tensors: entity tokens and global features.

use super::*;
use crate::card::Card;
use crate::game::{BlindSlot, BlindStatus, Game};

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
/// Suit x rank counts of the discard pile (after 6 unused slots).
const G_DISCARD: usize = G_DECK_SIZE + 10;
const _: () = assert!(G_DECK_SIZE == 151 && G_DISCARD + 52 <= GLOB_FLOAT);

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

    fn card(&mut self, z: i32, slot: usize, c: &Card, hide: bool, target: Option<(i32, i32)>) -> Option<usize> {
        let (ints, floats) = card_token(z, slot, c, hide, target);
        self.push(ints, &floats)
    }
}

/// The token of a card. `target`: a (suit, rank) shown on the card's text
/// (The Idol, Mail-In Rebate, Ancient Joker, Castle), encoded in the suit /
/// rank columns.
fn card_token(z: i32, slot: usize, c: &Card, hide: bool, target: Option<(i32, i32)>) -> ([i32; TOK_INT], [f32; TOK_FLOAT]) {
    if hide {
        let mut f = [0.0f32; TOK_FLOAT];
        f[1] = 1.0;
        return ([z, center::HIDDEN, 0, 0, 0, 0, 0, slot as i32, 0], f);
    }
    let a = &c.ab;
    let playing = c.is_playing();
    let (suit, rank) = if playing && !c.is_stone() { (suit_id(c.suit), c.rank as i32) } else { target.unwrap_or((0, 0)) };
    let aux = match c.kind {
        Kind::Joker(Joker::TodoList) => a.to_do.map(|h| h as i32 + 1).unwrap_or(0),
        _ => 0,
    };
    let f: [f32; TOK_FLOAT] = [
        bf(c.debuff),
        // Deck-view cards show face up.
        bf(c.face_down && z != zone::DECK),
        (c.cost / 10.0) as f32,
        (c.sell_cost / 10.0) as f32,
        sl(a.mult),
        sl(a.x_mult - 1.0),
        sl(a.chips),
        (a.dollars / 10.0) as f32,
        (a.extra_value / 10.0) as f32,
        sl(a.perma_bonus),
        (a.remaining / 10.0) as f32,
        (a.invis_rounds / 3.0) as f32,
        (a.yorick_discards / 23.0) as f32,
        sl(a.caino_xmult),
        (a.loyalty_remaining / 5.0) as f32,
        bf(a.eternal),
        bf(a.perishable),
        (a.perish_tally / 5.0) as f32,
        bf(a.rental),
        bf(a.forced_selection),
        bf(a.played_this_ante),
        bf(a.couponed),
        (c.times_played as f32 / 10.0).min(5.0),
        if playing { c.enhancement.x_mult() as f32 - 1.0 } else { 0.0 },
        // The deck count the card's text shows (Steel Joker, Stone Joker,
        // Cloud 9, Driver's License; at most one is non-zero).
        ((a.steel_tally + a.stone_tally + a.nine_tally + a.driver_tally) / 10.0) as f32,
        // Temperance's current payout.
        (a.money / 10.0) as f32,
    ];
    let ints = [z, kind_center(c.kind), suit, rank, if playing { enh_id(c.enhancement) } else { 0 }, edition_id(c.edition), seal_id(c.seal), slot as i32, aux];
    (ints, f)
}

/// The cards a player has not seen placed: the draw pile plus face-down hand
/// cards (what the game's deck view lists as remaining).
fn unseen_pool(g: &Game) -> impl Iterator<Item = &Card> {
    g.deck.iter().chain(g.hand.iter().filter(|c| c.face_down))
}

/// Whether anything that affects scoring is face down (hand cards or
/// jokers); previews would then reveal hidden information.
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
pub fn encode_opts(g: &Game, o: &mut ObsMut<'_>, deck_tokens: bool) -> Layout {
    let mut lay = Layout::of(g);
    let mut t = Toks { o, n: 0 };
    // Hand first: hand index == token index (pointer slots).
    for (i, c) in g.hand.iter().take(MAX_HAND).enumerate() {
        t.card(zone::HAND, i, c, c.face_down, None);
    }
    for (i, c) in g.jokers.iter().enumerate() {
        let target = match c.kind {
            Kind::Joker(Joker::Idol) => Some((suit_id(g.idol_card.1), g.idol_card.0 as i32)),
            Kind::Joker(Joker::Mail) => Some((0, g.mail_rank.0 as i32)),
            Kind::Joker(Joker::Ancient) => Some((g.ancient_suit.map(suit_id).unwrap_or(0), 0)),
            Kind::Joker(Joker::Castle) => Some((suit_id(g.castle_suit), 0)),
            _ => None,
        };
        t.card(zone::JOKER, i, c, c.face_down, target);
    }
    for (i, c) in g.consumables.iter().enumerate() {
        t.card(zone::CONSUMABLE, i, c, false, None);
    }
    for (i, c) in g.shop_jokers.iter().enumerate() {
        t.card(zone::SHOP_CARD, i, c, false, None);
    }
    for (i, c) in g.shop_vouchers.iter().enumerate() {
        t.card(zone::SHOP_VOUCHER, i, c, false, None);
    }
    for (i, c) in g.shop_boosters.iter().enumerate() {
        t.card(zone::SHOP_BOOSTER, i, c, false, None);
    }
    for (i, c) in g.pack_cards.iter().enumerate() {
        t.card(zone::PACK, i, c, false, None);
    }
    for (i, tag) in g.tags.iter().enumerate() {
        let f = [bf(tag.triggered)];
        t.push([zone::TAG, center::TAG0 + tag.kind as i32, 0, 0, 0, 0, 0, i as i32, tag.orbital_hand.map(|h| h as i32 + 1).unwrap_or(0)], &f);
    }
    // The three blinds of the ante.
    for (i, slot) in [BlindSlot::Small, BlindSlot::Big, BlindSlot::Boss].into_iter().enumerate() {
        let kind = match slot {
            BlindSlot::Small => BlindKind::Small,
            BlindSlot::Big => BlindKind::Big,
            BlindSlot::Boss => g.boss_choice,
        };
        let d = kind.def();
        let chips = blind_amount(g.ante, g.modifiers.scaling) * d.mult * g.starting_params.ante_scaling;
        let dollars = if g.modifiers.no_blind_reward[slot as usize] { 0.0 } else { d.dollars as f64 };
        let st = g.blind_states[slot as usize];
        let tag = if i < 2 { g.blind_tags[i].map(|t| t as i32 + 1).unwrap_or(0) } else { 0 };
        // The poker hand the blind's text names (rank column): an Orbital
        // Tag's hand, The Ox's most played hand, the hand The Mouth allows.
        let hand = match slot {
            BlindSlot::Small | BlindSlot::Big if g.blind_tags[i] == Some(TagKind::Orbital) => g.orbital_for(slot),
            BlindSlot::Boss if kind == BlindKind::Ox => Some(g.most_played_hand),
            BlindSlot::Boss if kind == BlindKind::Mouth && g.blind.kind == Some(BlindKind::Mouth) => g.blind.only_hand,
            _ => None,
        };
        let f = [
            sl(chips),
            (dollars / 5.0) as f32,
            bf(st == BlindStatus::Select),
            bf(st == BlindStatus::Upcoming),
            bf(st == BlindStatus::Current),
            bf(st == BlindStatus::Defeated),
            bf(st == BlindStatus::Skipped),
            bf(g.blind_on_deck == Some(slot)),
        ];
        t.push([zone::BLIND, center::BLIND0 + kind as i32, 0, hand.map_or(0, |h| h as i32 + 1), 0, 0, 0, i as i32, tag], &f);
    }
    for h in HandType::ALL {
        let l = &g.hands[h as usize];
        let f = [(l.level / 10.0) as f32, sl(l.chips), sl(l.mult), sl(l.played), (l.played_this_round / 3.0) as f32, bf(l.visible)];
        t.push([zone::HAND_TYPE, center::HAND0 + h as i32, 0, 0, 0, 0, 0, h as i32, 0], &f);
    }
    debug_assert_eq!(t.n, lay.n_tokens, "Layout::of disagrees with the encoding");
    // Remaining deck and discard pile as sorted multisets. Like the game's
    // deck view, the "remaining" pool includes face-down hand cards, so their
    // identities cannot be deduced. Tokens are ordered by their whole
    // content, so neither draw order nor hidden state breaks ties.
    if deck_tokens {
        let piles: [(i32, Vec<&Card>); 2] = [(zone::DECK, unseen_pool(g).collect()), (zone::DISCARD, g.discard.iter().collect())];
        for (z, pile) in piles {
            let mut toks: Vec<_> = pile.into_iter().map(|c| card_token(z, 0, c, false, None)).collect();
            toks.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.iter().zip(&b.1).map(|(x, y)| x.total_cmp(y)).find(|o| o.is_ne()).unwrap_or(std::cmp::Ordering::Equal)));
            for (k, (mut ints, floats)) in toks.into_iter().enumerate() {
                ints[SLOT] = k.min(255) as i32;
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
    let cur_blind = g.blind.kind.map(|k| k as i32 + 1).unwrap_or(0);
    o.glob_i.copy_from_slice(&[
        phase_id(g.phase),
        g.stake as i32,
        g.back as i32,
        cur_blind,
        g.boss_choice as i32 + 1,
        g.blind_tags[0].map(|t| t as i32 + 1).unwrap_or(0),
        g.blind_tags[1].map(|t| t as i32 + 1).unwrap_or(0),
        g.last_hand_played.map(|h| h as i32 + 1).unwrap_or(0),
        g.last_tarot_planet.map(kind_center).unwrap_or(0),
        g.current_voucher.filter(|_| g.voucher_seen).map(|v| kind_center(Kind::Voucher(v))).unwrap_or(0),
    ]);
    let gf = &mut *o.glob_f;
    gf.fill(0.0);
    let target = g.blind.chips.max(1.0);
    let vals: [f32; G_SCALARS] = [
        g.ante as f32 / 8.0,
        g.round as f32 / 24.0,
        sl(g.blind.chips),
        sl(g.chips),
        (g.chips / target).min(4.0) as f32,
        g.hands_left as f32 / 5.0,
        g.discards_left as f32 / 5.0,
        (g.dollars / 25.0) as f32,
        sl(g.dollars),
        g.hand_limit.limit as f32 / 10.0,
        g.joker_limit.limit as f32 / 5.0,
        g.jokers.len() as f32 / 5.0,
        g.consumable_limit.limit as f32 / 2.0,
        g.consumables.len() as f32 / 2.0,
        (g.reroll_cost / 5.0) as f32,
        g.free_rerolls as f32,
        (g.interest_cap / 25.0) as f32,
        g.skips as f32 / 5.0,
        g.deck.len() as f32 / 52.0,
        g.discard.len() as f32 / 52.0,
        g.playing_cards().count() as f32 / 52.0,
        g.hands_played_round as f32 / 4.0,
        g.discards_used as f32 / 4.0,
        g.pack_choices as f32,
        (g.round_dollars / 10.0) as f32,
        bf(g.run_won()),
        bf(g.endless),
        bf(g.boss_rerolled),
        bf(g.blind.disabled),
        bf(g.blind.boss),
        g.probabilities as f32 / 2.0,
        (g.bankrupt_at / 20.0) as f32,
        (g.discount_percent / 50.0) as f32,
        0.0, // reserved (was G.GAME.inflation, which only challenges change)
        (g.unused_discards / 20.0) as f32,
        g.hands_played as f32 / 50.0,
        (g.interest_amount) as f32,
        (g.rr_hands / 4.0) as f32,
        (g.rr_discards / 3.0) as f32,
        g.ecto_minus as f32 / 3.0,
        bf(g.hand_sort_by_suit),
        (g.shop_joker_max / 2.0) as f32,
        bf(g.gros_michel_extinct),
        g.pack_size as f32 / 5.0,
        (g.consumable_usage_tarot / 10.0) as f32,
        (g.consumable_usage_planet / 10.0) as f32,
        (g.consumable_usage_spectral / 5.0) as f32,
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
        Some((suit_id(c.suit) as usize - 1) * 13 + (c.rank as usize).saturating_sub(2).min(12))
    };
    let (mut size, mut stone, mut debuffed) = (0, 0, 0);
    for c in unseen_pool(g) {
        if let Some(i) = card_slot(c) {
            gf[G_DECK + i] += 0.5;
        }
        gf[G_DECK_ENH + enh_id(c.enhancement) as usize] += 0.1;
        gf[G_DECK_EDITION + edition_id(c.edition) as usize] += 0.1;
        gf[G_DECK_SEAL + seal_id(c.seal) as usize] += 0.1;
        size += 1;
        stone += c.is_stone() as usize;
        debuffed += c.debuff as usize;
    }
    gf[G_DECK_SIZE] = size as f32 / 52.0;
    gf[G_DECK_STONE] = stone as f32 / 10.0;
    gf[G_DECK_DEBUFFED] = debuffed as f32 / 10.0;
    gf[G_FACE_DOWN] = g.hand.iter().filter(|c| c.face_down).count() as f32 / 8.0;
    for c in &g.discard {
        if let Some(i) = card_slot(c) {
            gf[G_DISCARD + i] += 0.5;
        }
    }
    lay
}
