//! Port of `G.FUNCS.evaluate_play` (`state_events.lua`).

use crate::card::Kind;
use crate::data::{Back, BlindKind, Edition, Enhancement, Joker, Seal, Tarot, Voucher};
use crate::ev::Ev;
use crate::game::{Area, CardSet, Game};
use crate::jokers::{CardArea, Ctx, Effect, Scoring};
use crate::rng::Key;
use crate::{blind, consumables, events, hand_eval, jokers, num};

/// A playing card's own `eval_card` result in the play area.
#[derive(Default, Clone, Copy)]
struct CardEval {
	chips: Option<f64>,
	mult: Option<f64>,
	x_mult: Option<f64>,
	p_dollars: Option<f64>,
	ed_chips: Option<f64>,
	ed_mult: Option<f64>,
	ed_xmult: Option<f64>,
	has_edition: bool,
}

/// Inputs to `Card:update`'s per-frame joker values.
#[derive(Default)]
struct Derived {
	steel: f64,
	stone: f64,
	driver: f64,
	nine: f64,
	n_jokers: f64,
	stencils: f64,
	limit: f64,
	skips: f64,
}

impl Derived {
	fn apply(&self, c: &mut crate::card::Card, sells: &[f64], me: Option<usize>) {
		match c.kind {
			Kind::Joker(Joker::Throwback) => c.ab.x_mult = 1.0 + self.skips * 0.25,
			Kind::Joker(Joker::DriversLicense) => c.ab.driver_tally = self.driver,
			Kind::Joker(Joker::SteelJoker) => c.ab.steel_tally = self.steel,
			Kind::Joker(Joker::Cloud9) => c.ab.nine_tally = self.nine,
			Kind::Joker(Joker::Stone) => c.ab.stone_tally = self.stone,
			Kind::Joker(Joker::Stencil) => {
				c.ab.x_mult = (self.limit - self.n_jokers) + self.stencils;
			}
			Kind::Joker(Joker::Swashbuckler) => {
				let mut s = 0.0;
				for (k, v) in sells.iter().enumerate() {
					if Some(k) != me {
						s += *v;
					}
				}
				c.ab.mult = s;
			}
			_ => {}
		}
	}
}

/// See [`Game::update_derived`].
pub(crate) fn update_derived(g: &mut Game) {
	let mut d = Derived::default();
	for c in g.playing_cards() {
		match c.enhancement {
			Enhancement::Steel => d.steel += 1.0,
			Enhancement::Stone => d.stone += 1.0,
			_ => {}
		}
		if c.enhancement != Enhancement::Base {
			d.driver += 1.0;
		}
		if c.id_is(9) {
			d.nine += 1.0;
		}
	}
	d.n_jokers = num::float(g.jokers.len());
	d.stencils = num::float(
		g.jokers
			.iter()
			.filter(|c| c.is_joker(Joker::Stencil))
			.count(),
	);
	d.limit = g.joker_limit.limit;
	d.skips = g.skips;
	let sells: Vec<f64> = g.jokers.iter().map(|c| c.sell_cost).collect();
	let temperance = consumables::joker_sell_total(g);
	for (i, c) in g.jokers.iter_mut().enumerate() {
		d.apply(c, &sells, Some(i));
	}
	for c in g.shop_jokers.iter_mut().chain(g.pack_cards.iter_mut()) {
		d.apply(c, &sells, None);
	}
	for c in g
		.consumables
		.iter_mut()
		.chain(g.shop_jokers.iter_mut())
		.chain(g.pack_cards.iter_mut())
	{
		if c.kind == Kind::Tarot(Tarot::Temperance) {
			c.ab.money = temperance.min(50.0);
		}
	}
}

/// `eval_card(playing_card, {cardarea = G.play})` (RNG for Lucky cards).
fn eval_play_card(g: &mut Game, i: usize) -> CardEval {
	let prob = g.probabilities;
	let mut eval = CardEval::default();
	let c = &g.play[i];
	let chips = c.chip_bonus();
	if chips > 0.0 {
		eval.chips = Some(chips);
	}
	// get_chip_mult
	if !c.debuff {
		let m = if c.enhancement == Enhancement::Lucky {
			let key = Key::from("lucky_mult");
			if g.rng.random(&key) < prob / 5.0 {
				g.play[i].lucky_trigger = true;
				20.0
			} else {
				0.0
			}
		} else {
			c.enhancement.mult()
		};
		if m > 0.0 {
			eval.mult = Some(m);
		}
	}
	let c = &g.play[i];
	if !c.debuff && c.enhancement.x_mult() > 1.0 {
		eval.x_mult = Some(c.enhancement.x_mult());
	}
	// get_p_dollars
	if !c.debuff {
		let mut ret = 0.0;
		if c.seal == Some(Seal::Gold) {
			ret += 3.0;
		}
		if c.enhancement.p_dollars() > 0.0 && c.enhancement == Enhancement::Lucky {
			let key = Key::from("lucky_money");
			if g.rng.random(&key) < prob / 15.0 {
				g.play[i].lucky_trigger = true;
				ret += 20.0;
			}
		}
		if ret > 0.0 {
			jokers::buffer_dollars(g, ret);
			eval.p_dollars = Some(ret);
		}
	}
	let c = &g.play[i];
	if !c.debuff {
		match c.edition {
			Edition::None => {}
			Edition::Foil => {
				eval.has_edition = true;
				eval.ed_chips = Some(50.0);
			}
			Edition::Holo => {
				eval.has_edition = true;
				eval.ed_mult = Some(10.0);
			}
			Edition::Polychrome => {
				eval.has_edition = true;
				eval.ed_xmult = Some(1.5);
			}
			Edition::Negative => eval.has_edition = true,
		}
	}
	eval
}

/// Applies one joker effect table in `evaluate_play`'s fixed field order
/// (with its juice / hand text / status text events).
fn apply_play_effect(g: &mut Game, chips: &mut f64, mult: &mut f64, e: &Effect) {
	// Every field queues the juice of the effect's card first.
	if let Some(c) = e.chips {
		g.events.push(Ev::Wait);
		*chips += c;
		g.events.hand_text(0.0);
		g.events.status_text(0.6);
	}
	if let Some(m) = e.mult {
		g.events.push(Ev::Wait);
		*mult += m;
		g.events.hand_text(0.0);
		g.events.status_text(0.65);
	}
	if let Some(d) = e.dollars {
		g.events.push(Ev::Wait);
		g.events.push(Ev::EaseDollars(d));
		g.events.status_text(0.65);
	}
	if e.extra {
		g.events.push(Ev::Wait);
		if e.eight_ball {
			g.events.push_before(0.0, Ev::Create {
				set: CardSet::Tarot,
				area: Area::Consumables,
				forced: None,
				append: "8ba",
				rarity: None,
				legendary: false,
			});
		}
		g.events.hand_text(0.0);
		g.events.status_text(e.delay.unwrap_or(0.75));
	}
	if let Some(x) = e.x_mult {
		g.events.push(Ev::Wait);
		x_mult(g, mult, x);
		g.events.hand_text(0.0);
		g.events.status_text(0.65);
	}
}

/// `mult` times `x`. Steamodded applies a ×Mult as `mult + mult * (x - 1)`,
/// which can round one ulp away from vanilla's `mult * x` and so flip the
/// floored score (e.g. 150 × 22 × Ramen's 1.93: 6368 vs 6369).
#[expect(clippy::float_cmp, reason = "Steamodded's own exact test for a ×1")]
fn x_mult(g: &Game, mult: &mut f64, x: f64) {
	if !g.steamodded {
		*mult *= x;
	} else if x != 1.0 {
		*mult += *mult * (x - 1.0);
	}
}

#[expect(
	clippy::too_many_lines,
	reason = "follows `evaluate_play` step by step; its sections are already split out"
)]
pub(crate) fn evaluate_play(g: &mut Game) {
	g.update_derived();
	let flags = g.eval_flags();
	let ph = {
		// At most 8 cards are in play.
		let mut refs = [g.play.first().expect("evaluate_play with no cards"); 8];
		for (r, c) in refs.iter_mut().zip(&g.play) {
			*r = c;
		}
		hand_eval::evaluate(&refs[..g.play.len()], &flags)
	};
	let text = ph.top.expect("a play of at least one card is a hand");
	{
		let h = &mut g.hands[text as usize];
		h.played += 1.0;
		h.played_this_round += 1.0;
		h.visible = true;
	}
	g.last_hand_played = Some(text);
	let mut mask = ph.top_mask;
	if g.has_joker(Joker::Splash) {
		#[expect(
			clippy::cast_possible_truncation,
			reason = "at most 8 cards are in play, so the mask fits"
		)]
		let all = ((1u16 << g.play.len()) - 1) as u8;
		mask = all;
	} else {
		for (i, c) in g.play.iter().enumerate() {
			if c.is_stone() {
				mask |= 1 << i;
			}
		}
	}
	let mut sc = Scoring {
		name: text,
		hands: ph,
		cards: [0; 8],
		n_cards: 0,
	};
	for i in 0..g.play.len() {
		if mask >> i & 1 == 1 {
			sc.cards[sc.n_cards as usize] = num::byte(i);
			sc.n_cards += 1;
		}
	}
	g.last_scoring.clear();
	g.last_scoring
		.extend(sc.scoring().iter().map(|&i| g.play[i as usize].id));

	// Selection/highlight visuals. The hand text shows the selection's
	// name unless a selected card was face down ('????').
	let renamed = g.hand_text_cleared || g.play.iter().any(|c| c.face_down);
	g.events.delay(0.2);
	for _ in 0..sc.n_cards {
		g.events.highlight_ev();
	}
	if renamed {
		g.events.delay(0.3);
	}
	g.events.hand_text_imm(if renamed { 0.4 } else { 0.0 });

	let mut hand_chips;
	let mut mult;
	let n_play = g.play.len();
	if blind::debuff_hand(g, n_play, text, false) {
		mult = 0.0;
		hand_chips = 0.0;
		g.events.push(Ev::Wait);
		// play_area_status_text("Not Allowed!")
		g.events.wait(crate::events::Trig::Before(0.6));
		for i in 0..g.jokers.len() {
			if let Some(e) = jokers::calc_joker(g, i, &Ctx::DebuffedHand, Some(&sc)) {
				jokers::effect_status(g, &e);
			}
		}
	} else {
		g.events.delay(0.4);
		// before
		let nj = g.jokers.len();
		for i in 0..nj {
			if i >= g.jokers.len() {
				break;
			}
			if let Some(e) = jokers::calc_joker(g, i, &Ctx::Before, Some(&sc)) {
				jokers::effect_status(g, &e);
				if e.playing_cards_created {
					jokers::playing_card_joker_effects(g, 1);
				}
				if e.level_up {
					events::level_up_hand_ev(g, text, 1.0, false);
				}
			}
		}
		mult = g.hands[text as usize].mult;
		hand_chips = g.hands[text as usize].chips;
		let flint = g.blind.active(BlindKind::Flint);
		let (m, c) = blind::modify_hand(g, mult, hand_chips);
		mult = m;
		hand_chips = c;
		if flint {
			g.events.hand_text(0.8);
		}

		score_played_cards(g, &sc, &mut hand_chips, &mut mult);
		score_held_cards(g, &sc, &mut mult);
		score_joker_area(g, &sc, &mut hand_chips, &mut mult);

		// Back:trigger_effect('final_scoring_step')
		if g.back == Back::Plasma {
			let tot = hand_chips + mult;
			hand_chips = (tot / 2.0).floor();
			mult = (tot / 2.0).floor();
			g.events.hand_text(0.0);
			g.events.push(Ev::Wait);
			g.events.delay(0.6);
		}

		destroy_scored_cards(g, &sc, n_play);
	}
	let score = (hand_chips * mult).floor();
	g.last_hand_score = score;
	g.scored = true;
	g.events.push_after(0.4, Ev::PushHandTextImm);
	if hand_chips * mult > 0.0 {
		g.events.delay(0.8);
		g.events.push(Ev::Wait);
	}
	let target = g.chips + score;
	g.events.push_ev(
		Ev::SetChips(target),
		crate::events::Trig::Ease(0.5),
		false,
		true,
	);
	g.events.wait(crate::events::Trig::Ease(0.5));
	g.events.push(Ev::Wait);
	g.events.delay(0.3);
	for i in 0..g.jokers.len() {
		if let Some(e) = jokers::calc_joker(g, i, &Ctx::After, Some(&sc)) {
			jokers::effect_status(g, &e);
		}
	}
	g.events.push(Ev::Wait);
}

/// The scoring cards of the played hand (with repetitions): their own
/// chips / mult / money / editions, then the joker effects on each.
fn score_played_cards(g: &mut Game, sc: &Scoring, hand_chips: &mut f64, mult: &mut f64) {
	for si in 0..sc.n_cards as usize {
		let pi = sc.cards[si] as usize;
		if g.play[pi].debuff {
			g.blind.triggered = true;
			g.events.push(Ev::Wait);
			g.events.status_text(0.65);
			continue;
		}
		let mut reps = 1u32;
		if g.play[pi].seal == Some(Seal::Red) {
			reps += 1;
		}
		for j in 0..g.jokers.len() {
			if let Some(e) = jokers::calc_joker(
				g,
				j,
				&Ctx::Repetition {
					area: CardArea::Play,
					other: pi,
					effects: false,
				},
				Some(sc),
			) {
				reps += e.repetitions;
			}
		}
		for r in 0..reps {
			if r > 0 {
				g.events.status_text(0.75);
			}
			let own = eval_play_card(g, pi);
			let mut jeffs: Vec<Effect> = Vec::new();
			for k in 0..g.jokers.len() {
				if let Some(e) = jokers::calc_joker(
					g,
					k,
					&Ctx::Individual {
						area: CardArea::Play,
						other: pi,
					},
					Some(sc),
				) {
					jeffs.push(e);
				}
			}
			g.play[pi].lucky_trigger = false;
			// effects[1]: the card itself
			if let Some(c) = own.chips {
				*hand_chips += c;
				g.events.hand_text(0.0);
				g.events.status_text(0.6);
			}
			if let Some(m) = own.mult {
				*mult += m;
				g.events.hand_text(0.0);
				g.events.status_text(0.65);
			}
			if let Some(d) = own.p_dollars {
				g.events.push(Ev::EaseDollars(d));
				g.events.status_text(0.65);
			}
			if let Some(x) = own.x_mult {
				x_mult(g, mult, x);
				g.events.hand_text(0.0);
				g.events.status_text(0.65);
			}
			if own.has_edition {
				*hand_chips += own.ed_chips.unwrap_or(0.0);
				*mult += own.ed_mult.unwrap_or(0.0);
				x_mult(g, mult, own.ed_xmult.unwrap_or(1.0));
				g.events.hand_text(0.0);
				g.events.status_text(0.75);
			}
			for e in &jeffs {
				apply_play_effect(g, hand_chips, mult, e);
			}
		}
	}
}

/// The cards held in hand (with repetitions): Steel, Baron, Shoot the Moon, ...
fn score_held_cards(g: &mut Game, sc: &Scoring, mult: &mut f64) {
	g.events.delay(0.3);
	let nh = g.hand.len();
	for hi in 0..nh {
		if hi >= g.hand.len() {
			break;
		}
		let mut reps = 1usize;
		let mut j = 0usize;
		while j < reps {
			if j > 0 {
				g.events.status_text(0.75);
			}
			let c = &g.hand[hi];
			let own_x = if !c.debuff && c.enhancement.h_x_mult() > 0.0 {
				Some(c.enhancement.h_x_mult())
			} else {
				None
			};
			let mut jeffs: Vec<Effect> = Vec::new();
			for k in 0..g.jokers.len() {
				if let Some(e) = jokers::calc_joker(
					g,
					k,
					&Ctx::Individual {
						area: CardArea::Hand,
						other: hi,
					},
					Some(sc),
				) {
					jeffs.push(e);
				}
			}
			if j == 0 {
				let nonempty = own_x.is_some() || !jeffs.is_empty();
				let c = &g.hand[hi];
				if !c.debuff && c.seal == Some(Seal::Red) && nonempty {
					reps += 1;
				}
				for k in 0..g.jokers.len() {
					if let Some(e) = jokers::calc_joker(
						g,
						k,
						&Ctx::Repetition {
							area: CardArea::Hand,
							other: hi,
							effects: nonempty,
						},
						Some(sc),
					) {
						reps += e.repetitions as usize;
					}
				}
			}
			if let Some(x) = own_x {
				x_mult(g, mult, x);
				g.events.hand_text(0.0);
				g.events.status_text(0.65);
			}
			for e in &jeffs {
				g.events.push(Ev::Wait);
				if let Some(d) = e.dollars {
					g.events.push(Ev::EaseDollars(d));
					g.events.status_text(0.65);
				}
				if let Some(h) = e.h_mult {
					*mult += h;
					g.events.hand_text(0.0);
					g.events.status_text(0.65);
				}
				if let Some(x) = e.x_mult {
					x_mult(g, mult, x);
					g.events.hand_text(0.0);
					g.events.status_text(0.65);
				}
				if e.extra {
					// `message` (debuffed)
					g.events.hand_text(0.0);
					g.events.status_text(e.delay.unwrap_or(0.75));
				}
			}
			j += 1;
		}
	}
}

/// The joker (and consumable) area: editions, each joker's main effect
/// and the jokers reacting to it (Baseball Card).
fn score_joker_area(g: &mut Game, sc: &Scoring, hand_chips: &mut f64, mult: &mut f64) {
	let text = sc.name;
	let nj = g.jokers.len();
	let nc = g.consumables.len();
	for i in 0..nj + nc {
		let is_cons = i >= nj;
		let (ed_chips, ed_mult, ed_x) = {
			let c = if is_cons {
				g.consumables.get(i - nj)
			} else {
				g.jokers.get(i)
			};
			let Some(c) = c else { continue };
			if c.debuff {
				(None, None, None)
			} else {
				match c.edition {
					Edition::Foil => (Some(50.0), None, None),
					Edition::Holo => (None, Some(10.0), None),
					Edition::Polychrome => (None, None, Some(1.5)),
					_ => (None, None, None),
				}
			}
		};
		if let Some(c) = ed_chips {
			*hand_chips += c;
			g.events.hand_text(0.0);
			g.events.status_text(0.75);
		}
		if let Some(m) = ed_mult {
			*mult += m;
			g.events.hand_text(0.0);
			g.events.status_text(0.75);
		}
		let eff = if is_cons {
			let c = &g.consumables[i - nj];
			match c.kind {
				Kind::Planet(p) if g.voucher_used(Voucher::Observatory) && p.hand() == text => {
					Some(Effect {
						xmult_mod: Some(1.5),
						..Effect::default()
					})
				}
				_ => None,
			}
		} else {
			jokers::calc_joker(g, i, &Ctx::Main, Some(sc))
		};
		if let Some(e) = eff {
			if let Some(m) = e.mult_mod {
				*mult += m;
			}
			if let Some(c) = e.chip_mod {
				*hand_chips += c;
			}
			if let Some(x) = e.xmult_mod {
				x_mult(g, mult, x);
			}
			g.events.hand_text(0.0);
			jokers::effect_status(g, &e);
		}
		for v in 0..g.jokers.len() {
			let ctx = Ctx::OtherJoker {
				other: (!is_cons).then_some(i),
			};
			if let Some(e) = jokers::calc_joker(g, v, &ctx, Some(sc)) {
				let mut any = false;
				if let Some(m) = e.mult_mod {
					*mult += m;
					any = true;
				}
				if let Some(c) = e.chip_mod {
					*hand_chips += c;
					any = true;
				}
				if let Some(x) = e.xmult_mod {
					x_mult(g, mult, x);
					any = true;
				}
				if any {
					g.events.hand_text(0.0);
					jokers::effect_status(g, &e);
				}
			}
		}
		if let Some(x) = ed_x {
			x_mult(g, mult, x);
			g.events.hand_text(0.0);
			g.events.status_text(0.75);
		}
	}
}

/// Scored cards destroyed by jokers (Sixth Sense) or breaking glass.
fn destroy_scored_cards(g: &mut Game, sc: &Scoring, n_play: usize) {
	let mut destroyed_ids: Vec<u32> = Vec::new();
	let mut faces = 0;
	let mut glass = 0;
	let par = g.pareidolia();
	for si in 0..sc.n_cards as usize {
		let pi = sc.cards[si] as usize;
		let mut destroyed = false;
		g.events.highlight_ev();
		for j in 0..g.jokers.len() {
			if let Some(e) = jokers::calc_joker(
				g,
				j,
				&Ctx::DestroyingCard {
					play_idx: pi,
					full_len: n_play,
				},
				Some(sc),
			) && e.remove
			{
				destroyed = true;
				break;
			}
		}
		let c = &g.play[pi];
		if c.enhancement == Enhancement::Glass && !c.debuff {
			let k = Key::from("glass");
			if g.rng.random(&k) < g.probabilities / 4.0 {
				destroyed = true;
			}
		}
		if destroyed {
			let c = &mut g.play[pi];
			if c.enhancement == Enhancement::Glass {
				c.shattered = true;
				glass += 1;
			} else {
				c.destroyed = true;
			}
			if c.is_face(false, par) {
				faces += 1;
			}
			destroyed_ids.push(c.id);
		}
	}
	for j in 0..g.jokers.len() {
		jokers::calc_joker(g, j, &Ctx::RemovePlayingCards { faces, glass }, Some(sc));
	}
	for id in destroyed_ids {
		g.events.push(Ev::DissolvePlaying { id });
	}
}
