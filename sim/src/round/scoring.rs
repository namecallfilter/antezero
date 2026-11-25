//! Port of `G.FUNCS.evaluate_play` (`state_events.lua`).

use super::blind;
use crate::card::{AnyCard, Card, CardLike, CardMask, Kind, PlayingCard};
use crate::data::{Back, BlindKind, Edition, Enhancement, Joker, Rank, Seal, Tarot, Voucher};
use crate::event::{Blocks, Event, EventQueue, Trigger};
use crate::game::{Areas, CardSet, Game, HandLevels};
use crate::hand_eval::MAX_EVAL_CARDS;
use crate::joker::{CardArea, Effect, LifecycleCtx, Scoring, ScoringCtx};
use crate::pools::{NewCard, OwnedArea};
use crate::rng::{Key, PseudoRandom};
use crate::run::{History, Run};
use crate::{event, hand_eval, joker, num};

/// A playing card's own `eval_card` result in the play area.
#[derive(Default, Clone, Copy)]
struct CardEval {
	chips: Option<f64>,
	mult: Option<f64>,
	x_mult: Option<f64>,
	p_dollars: Option<f64>,
	/// The card's edition, unless it is debuffed (`ret.edition`).
	edition: Option<Edition>,
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
	/// Every derived value except Swashbuckler's, which needs the other
	/// jokers' sell values.
	fn apply(&self, c: &mut Card) {
		match c.kind {
			Kind::Joker(Joker::Throwback) => c.ab.x_mult = 1.0 + self.skips * 0.25,
			Kind::Joker(Joker::DriversLicense) => c.ab.driver_tally = self.driver,
			Kind::Joker(Joker::SteelJoker) => c.ab.steel_tally = self.steel,
			Kind::Joker(Joker::Cloud9) => c.ab.nine_tally = self.nine,
			Kind::Joker(Joker::Stone) => c.ab.stone_tally = self.stone,
			Kind::Joker(Joker::Stencil) => {
				c.ab.x_mult = (self.limit - self.n_jokers) + self.stencils;
			}
			_ => {}
		}
	}
}

/// See [`Game::update_derived`].
pub(crate) fn update_derived(areas: &mut Areas, skips: f64) {
	let mut d = Derived::default();
	for c in areas.playing_cards() {
		match c.enhancement {
			Enhancement::Steel => d.steel += 1.0,
			Enhancement::Stone => d.stone += 1.0,
			_ => {}
		}
		if c.enhancement != Enhancement::Base {
			d.driver += 1.0;
		}
		if c.has_rank(Rank::Nine) {
			d.nine += 1.0;
		}
	}
	d.n_jokers = num::float(areas.jokers.len());
	d.stencils = num::float(
		areas
			.jokers
			.iter()
			.filter(|c| c.is_joker(Joker::Stencil))
			.count(),
	);
	d.limit = areas.limits.jokers.limit;
	d.skips = skips;
	let sell_total = areas.joker_sell_total();
	for c in &mut areas.jokers {
		if c.is_joker(Joker::Swashbuckler) {
			// Swashbuckler's mult is the other jokers' sell value. Sell values
			// are whole numbers, so the difference is exactly their sum.
			c.ab.mult = sell_total - c.core.sell_cost;
		} else {
			d.apply(c);
		}
	}
	let offered = areas
		.shop_jokers
		.iter_mut()
		.chain(areas.pack_cards.iter_mut())
		.filter_map(AnyCard::as_card_mut);
	for c in offered {
		if c.is_joker(Joker::Swashbuckler) {
			c.ab.mult = sell_total;
		} else {
			d.apply(c);
		}
	}
	let offered = areas
		.shop_jokers
		.iter_mut()
		.chain(areas.pack_cards.iter_mut())
		.filter_map(AnyCard::as_card_mut);
	for c in areas.consumables.iter_mut().chain(offered) {
		if c.kind == Kind::Tarot(Tarot::Temperance) {
			c.ab.money = sell_total.min(50.0);
		}
	}
}

/// `eval_card(playing_card, {cardarea = G.play})` (RNG for Lucky cards).
fn eval_play_card(
	c: &mut PlayingCard, run: &mut Run, rng: &mut PseudoRandom, events: &mut EventQueue,
) -> CardEval {
	let prob = run.probabilities;
	let mut eval = CardEval::default();
	let chips = c.chip_bonus();
	if chips > 0.0 {
		eval.chips = Some(chips);
	}
	if c.core.debuff {
		return eval;
	}
	// get_chip_mult
	let m = if c.enhancement == Enhancement::Lucky {
		let key = Key::from("lucky_mult");
		if rng.random(&key) < prob / 5.0 {
			c.lucky_trigger = true;
			c.enhancement.mult()
		} else {
			0.0
		}
	} else {
		c.enhancement.mult()
	};
	if m > 0.0 {
		eval.mult = Some(m);
	}
	if c.enhancement.x_mult() > 1.0 {
		eval.x_mult = Some(c.enhancement.x_mult());
	}
	// get_p_dollars: Lucky is the only enhancement with `p_dollars`.
	let mut ret = 0.0;
	if c.seal == Some(Seal::Gold) {
		ret += 3.0;
	}
	if c.enhancement == Enhancement::Lucky {
		let key = Key::from("lucky_money");
		if rng.random(&key) < prob / 15.0 {
			c.lucky_trigger = true;
			ret += c.enhancement.p_dollars();
		}
	}
	if ret > 0.0 {
		joker::buffer_dollars(run, events, ret);
		eval.p_dollars = Some(ret);
	}
	eval.edition = c.core.edition;
	eval
}

/// Applies one joker effect table in `evaluate_play`'s fixed field order
/// (with its juice / hand text / status text events).
fn apply_play_effect(
	steamodded: bool, events: &mut EventQueue, chips: &mut f64, mult: &mut f64, e: &Effect,
) {
	// Every field queues the juice of the effect's card first.
	if let Some(c) = e.chips {
		events.push(Event::Wait);
		*chips += c;
		events.hand_text(0.0);
		events.status_text(0.6);
	}
	if let Some(m) = e.mult {
		events.push(Event::Wait);
		*mult += m;
		events.hand_text(0.0);
		events.status_text(0.65);
	}
	if let Some(d) = e.dollars {
		events.push(Event::Wait);
		events.push(Event::EaseDollars(d));
		events.status_text(0.65);
	}
	if e.extra {
		events.push(Event::Wait);
		if e.eight_ball {
			events.push_before(
				0.0,
				Event::Create(NewCard::new(CardSet::Tarot, OwnedArea::Consumables, "8ba")),
			);
		}
		events.hand_text(0.0);
		events.status_text(e.delay.unwrap_or(0.75));
	}
	if let Some(x) = e.x_mult {
		events.push(Event::Wait);
		apply_x_mult(steamodded, mult, x);
		events.hand_text(0.0);
		events.status_text(0.65);
	}
}

/// `mult` times `x`. Steamodded applies a ×Mult as `mult + mult * (x - 1)`,
/// which can round one ulp away from vanilla's `mult * x` and so flip the
/// floored score (e.g. 150 × 22 × Ramen's 1.93: 6368 vs 6369).
fn apply_x_mult(steamodded: bool, mult: &mut f64, x: f64) {
	if !steamodded {
		*mult *= x;
	} else if x != 1.0 {
		*mult += *mult * (x - 1.0);
	}
}

/// The played hand's poker hand and scoring cards (`G.FUNCS.get_poker_hand_info`
/// and the Splash / Stone additions), with the hand's play counters bumped.
fn scoring_hand(areas: &Areas, history: &mut History, hands: &mut HandLevels) -> Scoring {
	let flags = areas.eval_flags();
	let ph = {
		let first = areas
			.play
			.first()
			.expect("Action::Play is only legal with at least one selected card");
		let mut refs = [first; MAX_EVAL_CARDS];
		for (r, c) in refs.iter_mut().zip(&areas.play) {
			*r = c;
		}
		hand_eval::evaluate(&refs[..areas.play.len()], flags)
	};
	let top = ph.top.expect("a play of at least one card is a hand");
	let text = top.hand;
	{
		let h = &mut hands[text];
		h.played += 1.0;
		h.played_this_round += 1.0;
		h.visible = true;
	}
	history.last_hand_played = Some(text);
	let mut mask = top.mask;
	if areas.has_joker(Joker::Splash) {
		mask = u8::try_from((1u16 << areas.play.len()) - 1).expect("at most 8 cards are in play");
	} else {
		for (i, c) in areas.play.iter().enumerate() {
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
	let scoring = CardMask::from_bits(u32::from(mask));
	for (slot, i) in sc.cards.iter_mut().zip(scoring.indices()) {
		*slot = num::byte(i);
	}
	sc.n_cards = num::byte(scoring.len());
	sc
}

pub(crate) fn evaluate_play(g: &mut Game) {
	g.update_derived();
	let sc = scoring_hand(&g.areas, &mut g.history, &mut g.hands);

	// The hand text shows the selection's name unless a selected card was
	// face down ('????').
	let renamed = g.flow.hand_text_cleared || g.areas.play.iter().any(|c| c.core.face_down);
	g.events.delay(0.2);
	for _ in sc.scoring() {
		g.events.highlight();
	}
	if renamed {
		g.events.delay(0.3);
	}
	g.events
		.hand_text_immediate(if renamed { 0.4 } else { 0.0 });

	let n_play = g.areas.play.len();
	let (hand_chips, mult) = if blind::debuff_hand(
		&mut g.run,
		g.round.most_played_hand,
		&mut g.blind,
		&mut g.hands,
		&mut g.events,
		n_play,
		sc.name,
	) {
		g.events.push(Event::Wait);
		// play_area_status_text("Not Allowed!")
		g.events.wait(Trigger::Before(0.6));
		for i in 0..g.areas.jokers.len() {
			if let Some(e) = joker::calc_scoring(g, i, ScoringCtx::DebuffedHand, &sc) {
				joker::effect_status(&mut g.events, &e);
			}
		}
		(0.0, 0.0)
	} else {
		score_hand(g, &sc, n_play)
	};
	let score = (hand_chips * mult).floor();
	g.run.last_hand_score = score;
	g.flow.scored = true;
	g.events.push_after(0.4, Event::PushHandTextImmediate);
	if hand_chips * mult > 0.0 {
		g.events.delay(0.8);
		g.events.push(Event::Wait);
	}
	let target = g.round.chips + score;
	g.events.push_event(
		Event::SetChips(target),
		Trigger::Ease(0.5),
		Blocks::Background,
	);
	g.events.wait(Trigger::Ease(0.5));
	g.events.push(Event::Wait);
	g.events.delay(0.3);
	for i in 0..g.areas.jokers.len() {
		if let Some(e) = joker::calc_scoring(g, i, ScoringCtx::After, &sc) {
			joker::effect_status(&mut g.events, &e);
		}
	}
	if g.setup.steamodded {
		// Steamodded marks the cards for The Pillar here as well as when they
		// are played, so the mark stays on a card a joker changed while
		// scoring (Vampire, Midas Mask), which clears it.
		for i in 0..g.areas.play.len() {
			let id = g.areas.play[i].core.id;
			g.events.push(Event::MarkPlayed { id });
		}
	}
	g.events.push(Event::Wait);
}

/// The chips and mult of a hand the blind allows: the jokers' `before`
/// effects, the hand's level, then the played cards, the held cards and the
/// joker area, and last the scored cards that break.
fn score_hand(g: &mut Game, sc: &Scoring, n_play: usize) -> (f64, f64) {
	let text = sc.name;
	g.events.delay(0.4);
	// before
	for i in 0..g.areas.jokers.len() {
		if let Some(e) = joker::calc_scoring(g, i, ScoringCtx::Before, sc) {
			joker::effect_status(&mut g.events, &e);
			if e.playing_cards_created {
				joker::calc_jokers(g, LifecycleCtx::PlayingCardAdded { n: 1 });
			}
			if e.level_up {
				event::level_up_hand(&mut g.hands, &mut g.events, text, 1.0);
			}
		}
	}
	let flint = g.blind.active(BlindKind::Flint);
	let level = g.hands[text];
	let (mut mult, mut hand_chips) = blind::modify_hand(&mut g.blind, level.mult, level.chips);
	if flint {
		g.events.hand_text(0.8);
	}

	// The joker effects on one card, collected before any is applied.
	let mut effects: Vec<Effect> = Vec::new();
	score_played_cards(g, sc, &mut effects, &mut hand_chips, &mut mult);
	score_held_cards(g, sc, &mut effects, &mut mult);
	score_joker_area(g, sc, &mut hand_chips, &mut mult);

	// Back:trigger_effect('final_scoring_step')
	if g.setup.back == Back::Plasma {
		let tot = hand_chips + mult;
		hand_chips = (tot / 2.0).floor();
		mult = (tot / 2.0).floor();
		g.events.hand_text(0.0);
		g.events.push(Event::Wait);
		g.events.delay(0.6);
	}

	destroy_scored_cards(g, sc, n_play);
	(hand_chips, mult)
}

/// How often a card's effects run: once, once more for an active Red Seal,
/// and once more for each repetition a joker gives (`joker(j)` is the
/// effect of the `j`-th of `n_jokers`).
pub(super) fn repetitions(
	n_jokers: usize, red_seal: bool, mut joker: impl FnMut(usize) -> Option<Effect>,
) -> u32 {
	let mut reps = 1 + u32::from(red_seal);
	for j in 0..n_jokers {
		if let Some(e) = joker(j) {
			reps += e.repetitions;
		}
	}
	reps
}

/// The scoring cards of the played hand (with repetitions): their own
/// chips / mult / money / editions, then the joker effects on each.
fn score_played_cards(
	g: &mut Game, sc: &Scoring, effects: &mut Vec<Effect>, hand_chips: &mut f64, mult: &mut f64,
) {
	for pi in sc.scoring() {
		if g.areas.play[pi].core.debuff {
			g.blind.triggered = true;
			g.events.push(Event::Wait);
			g.events.status_text(0.65);
			continue;
		}
		let red_seal = g.areas.play[pi].seal == Some(Seal::Red);
		let ctx = ScoringCtx::Repetition {
			area: CardArea::Play,
			other: pi,
			effects: false,
		};
		let reps = repetitions(g.areas.jokers.len(), red_seal, |j| {
			joker::calc_scoring(g, j, ctx, sc)
		});
		for r in 0..reps {
			if r > 0 {
				g.events.status_text(0.75);
			}
			let own = eval_play_card(&mut g.areas.play[pi], &mut g.run, &mut g.rng, &mut g.events);
			effects.clear();
			for k in 0..g.areas.jokers.len() {
				if let Some(e) = joker::calc_scoring(
					g,
					k,
					ScoringCtx::Individual {
						area: CardArea::Play,
						other: pi,
					},
					sc,
				) {
					effects.push(e);
				}
			}
			g.areas.play[pi].lucky_trigger = false;
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
				g.events.push(Event::EaseDollars(d));
				g.events.status_text(0.65);
			}
			if let Some(x) = own.x_mult {
				apply_x_mult(g.setup.steamodded, mult, x);
				g.events.hand_text(0.0);
				g.events.status_text(0.65);
			}
			if let Some(edition) = own.edition {
				// `or 0` / `or 1` for the values the edition does not have.
				*hand_chips += edition.chips().unwrap_or(0.0);
				*mult += edition.mult().unwrap_or(0.0);
				apply_x_mult(g.setup.steamodded, mult, edition.x_mult().unwrap_or(1.0));
				g.events.hand_text(0.0);
				g.events.status_text(0.75);
			}
			for e in effects.iter() {
				apply_play_effect(g.setup.steamodded, &mut g.events, hand_chips, mult, e);
			}
		}
	}
}

/// The cards held in hand (with repetitions): Steel, Baron, Shoot the Moon, ...
fn score_held_cards(g: &mut Game, sc: &Scoring, effects: &mut Vec<Effect>, mult: &mut f64) {
	g.events.delay(0.3);
	for hi in 0..g.areas.hand.cards.len() {
		let mut reps = 1;
		let mut j = 0;
		while j < reps {
			if j > 0 {
				g.events.status_text(0.75);
			}
			let c = &g.areas.hand.cards[hi];
			let own_x = if !c.core.debuff && c.enhancement.h_x_mult() > 0.0 {
				Some(c.enhancement.h_x_mult())
			} else {
				None
			};
			effects.clear();
			for k in 0..g.areas.jokers.len() {
				if let Some(e) = joker::calc_scoring(
					g,
					k,
					ScoringCtx::Individual {
						area: CardArea::Hand,
						other: hi,
					},
					sc,
				) {
					effects.push(e);
				}
			}
			if j == 0 {
				let nonempty = own_x.is_some() || !effects.is_empty();
				let c = &g.areas.hand.cards[hi];
				let red_seal = !c.core.debuff && c.seal == Some(Seal::Red) && nonempty;
				let ctx = ScoringCtx::Repetition {
					area: CardArea::Hand,
					other: hi,
					effects: nonempty,
				};
				reps = repetitions(g.areas.jokers.len(), red_seal, |k| {
					joker::calc_scoring(g, k, ctx, sc)
				});
			}
			if let Some(x) = own_x {
				apply_x_mult(g.setup.steamodded, mult, x);
				g.events.hand_text(0.0);
				g.events.status_text(0.65);
			}
			for e in effects.iter() {
				g.events.push(Event::Wait);
				if let Some(d) = e.dollars {
					g.events.push(Event::EaseDollars(d));
					g.events.status_text(0.65);
				}
				if let Some(h) = e.h_mult {
					*mult += h;
					g.events.hand_text(0.0);
					g.events.status_text(0.65);
				}
				if let Some(x) = e.x_mult {
					apply_x_mult(g.setup.steamodded, mult, x);
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

/// A slot of the joker area as `evaluate_play` walks it: the jokers, then
/// the consumables.
#[derive(Clone, Copy)]
enum JokerAreaSlot {
	Joker(usize),
	Consumable(usize),
}

/// The joker (and consumable) area: editions, each joker's main effect
/// and the jokers reacting to it (Baseball Card).
fn score_joker_area(g: &mut Game, sc: &Scoring, hand_chips: &mut f64, mult: &mut f64) {
	let text = sc.name;
	let jokers = (0..g.areas.jokers.len()).map(JokerAreaSlot::Joker);
	let consumables = (0..g.areas.consumables.len()).map(JokerAreaSlot::Consumable);
	let slots = jokers.chain(consumables);
	for slot in slots {
		let c = match slot {
			JokerAreaSlot::Joker(i) => &g.areas.jokers[i],
			JokerAreaSlot::Consumable(i) => &g.areas.consumables[i],
		};
		let edition = if c.core.debuff { None } else { c.core.edition };
		if let Some(c) = edition.and_then(Edition::chips) {
			*hand_chips += c;
			g.events.hand_text(0.0);
			g.events.status_text(0.75);
		}
		if let Some(m) = edition.and_then(Edition::mult) {
			*mult += m;
			g.events.hand_text(0.0);
			g.events.status_text(0.75);
		}
		let eff = match slot {
			JokerAreaSlot::Joker(i) => joker::calc_scoring(g, i, ScoringCtx::Main, sc),
			JokerAreaSlot::Consumable(i) => match g.areas.consumables[i].kind {
				Kind::Planet(p)
					if g.history.used_vouchers.contains(Voucher::Observatory)
						&& p.hand() == text =>
				{
					Some(Effect::xmult_mod(1.5))
				}
				_ => None,
			},
		};
		if let Some(e) = eff {
			if let Some(m) = e.mult_mod {
				*mult += m;
			}
			if let Some(c) = e.chip_mod {
				*hand_chips += c;
			}
			if let Some(x) = e.xmult_mod {
				apply_x_mult(g.setup.steamodded, mult, x);
			}
			g.events.hand_text(0.0);
			joker::effect_status(&mut g.events, &e);
		}
		let other = match slot {
			JokerAreaSlot::Joker(i) => Some(i),
			JokerAreaSlot::Consumable(_) => None,
		};
		for v in 0..g.areas.jokers.len() {
			let ctx = ScoringCtx::OtherJoker { other };
			if let Some(e) = joker::calc_scoring(g, v, ctx, sc) {
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
					apply_x_mult(g.setup.steamodded, mult, x);
					any = true;
				}
				if any {
					g.events.hand_text(0.0);
					joker::effect_status(&mut g.events, &e);
				}
			}
		}
		if let Some(x) = edition.and_then(Edition::x_mult) {
			apply_x_mult(g.setup.steamodded, mult, x);
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
	let par = g.areas.pareidolia();
	for pi in sc.scoring() {
		let mut destroyed = false;
		g.events.highlight();
		for j in 0..g.areas.jokers.len() {
			if let Some(e) = joker::calc_scoring(
				g,
				j,
				ScoringCtx::DestroyingCard {
					play_idx: pi,
					full_len: n_play,
				},
				sc,
			) && e.remove
			{
				destroyed = true;
				break;
			}
		}
		let c = &g.areas.play[pi];
		if c.enhancement == Enhancement::Glass && !c.core.debuff {
			let k = Key::from("glass");
			if g.rng.random(&k) < g.run.probabilities / 4.0 {
				destroyed = true;
			}
		}
		if destroyed {
			let c = &mut g.areas.play[pi];
			if c.enhancement == Enhancement::Glass {
				c.shattered = true;
				glass += 1;
			} else {
				c.destroyed = true;
			}
			if c.is_face(par) {
				faces += 1;
			}
			destroyed_ids.push(c.core.id);
		}
	}
	joker::calc_jokers(g, LifecycleCtx::RemovePlayingCards { faces, glass });
	for id in destroyed_ids {
		g.events.push(Event::DissolvePlaying { id });
	}
}
