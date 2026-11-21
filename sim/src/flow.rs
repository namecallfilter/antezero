//! Round flow: blind select, new round, drawing, play / discard and the
//! per-state updates (`state_events.lua`, `button_callbacks.lua`, game.lua
//! `update_*`). The end of the round is in `round_end`.

use crate::areas::{self, Pile};
use crate::blind::CardDebuff;
use crate::calculate_joker::{Discarded, LifecycleCtx};
use crate::card::{AnyCard, Card, CardLike, Kind};
use crate::card_lifecycle::Pricing;
use crate::card_mask::CardMask;
use crate::data::{BlindKind, HandType, Joker, Seal, TagKind};
use crate::event::{Draw, Event};
use crate::event_queue::{Blocks, EventQueue, Trigger};
use crate::game::{
	Area, Areas, Blind, BlindSelect, BlindSlot, BlindStatus, CardSet, Flow, Game, HandLevels,
	PerBlind, Phase, State,
};
use crate::playing_card::PlayingCard;
use crate::rng::{Key, PseudoRandom, shuffle, trace};
use crate::round::Round;
use crate::run::{History, Run};
use crate::shop::{BOOSTER_SLOTS, PackSlot, Shop};
use crate::tags::{OrbitalChoices, Tags};
use crate::{
	blind, calculate_joker, card_lifecycle, hand_eval, num, pools, round_end, scoring, shop, tags,
};

/// `CardArea:shuffle(seed)`: sort by `sort_id`, then Fisher–Yates.
pub(crate) fn shuffle_area<C: CardLike>(cards: &mut [C], rng: &mut PseudoRandom, key: &Key) {
	cards.sort_unstable_by_key(|c| c.core().id);
	let seed = rng.seed_for(key);
	shuffle(cards, seed);
}

#[inline]
pub(crate) fn ease_dollars(run: &mut Run, d: f64) {
	run.dollars += d;
}

/// `ease_discard(mod)` (applied immediately).
pub(crate) fn ease_discard(round: &mut Round, m: f64) {
	if round.discards_left.max(m).abs() == 0.0 {
		return;
	}
	let m = (-round.discards_left).max(m);
	round.discards_left += m;
}

#[inline]
pub(crate) fn ease_hands_played(round: &mut Round, m: f64) {
	round.hands_left += m;
}

/// `G.hand:change_size(delta)`.
pub(crate) fn change_hand_size(events: &mut EventQueue, delta: f64) {
	if delta != 0.0 {
		events.push(Event::HandSize(delta));
	}
}

/// `CardArea:emplace` for piles (deck inserts at the front).
fn emplace(areas: &mut Areas, to: Pile, mut c: PlayingCard, stay_flipped: bool) {
	match to {
		Pile::Deck => {
			c.core.face_down = true;
			areas.deck.insert(0, c);
		}
		Pile::Discard => {
			c.core.face_down = true;
			areas.discard.push(c);
		}
		Pile::Hand | Pile::Play => {
			if c.core.face_down && !stay_flipped {
				c.core.face_down = false;
				c.wheel_flipped = false;
			}
			if to == Pile::Hand && stay_flipped {
				c.wheel_flipped = true;
				c.core.face_down = true;
			}
			if to == Pile::Hand {
				areas.hand.cards.push(c);
			} else {
				areas.play.push(c);
			}
		}
	}
}

/// The body of `draw_card`'s event.
pub(crate) fn draw_card(
	areas: &mut Areas, probabilities: f64, round: &Round, blind: &Blind, rng: &mut PseudoRandom,
	draw: Draw,
) {
	let Draw {
		from,
		to,
		card,
		sort,
	} = draw;
	let taken = if let Some(id) = card {
		let src = areas.pile_mut(from);
		areas::position_of(src, id).map(|i| src.remove(i))
	} else {
		// CardArea:draw_card_from → area:remove_card()
		let src_is_stack = matches!(from, Pile::Deck | Pile::Discard);
		let src = areas.pile_mut(from);
		if src.is_empty() {
			None
		} else if src_is_stack {
			src.pop()
		} else {
			Some(src.remove(0))
		}
	};
	let Some(c) = taken else { return };
	let stay = blind::stay_flipped(areas, probabilities, round, blind, rng, to, &c);
	trace!(rng, "draw {} {}", to.name(), c.core.id);
	emplace(areas, to, c, stay);
	if sort && to == Pile::Hand {
		areas.hand.sort();
	}
}

/// `G.FUNCS.draw_from_deck_to_hand(e)`: draws `count` cards, or with `None`
/// enough to fill the hand; returns true on the game-over path.
pub(crate) fn draw_from_deck_to_hand(
	flow: &mut Flow, areas: &Areas, round: &Round, blind: &Blind, events: &mut EventQueue,
	count: Option<usize>,
) -> bool {
	if !matches!(flow.state, State::TarotPack | State::SpectralPack)
		&& areas.hand.limit.limit <= 0.0
		&& areas.hand.cards.is_empty()
	{
		flow.state = State::GameOver;
		flow.state_complete = false;
		return true;
	}
	let mut hand_space: usize = match count {
		Some(n) => n,
		None => areas
			.deck
			.len()
			.min(num::trunc(areas.hand.limit.limit).saturating_sub(areas.hand.cards.len())),
	};
	if blind.active(BlindKind::Serpent) && (round.hands_played > 0.0 || round.discards_used > 0.0) {
		hand_space = areas.deck.len().min(3);
	}
	events.delay(0.3);
	for _ in 0..hand_space {
		events.draw(
			Draw {
				sort: true,
				..Draw::next(Pile::Deck, Pile::Hand)
			},
			0.1,
		);
	}
	false
}

/// Finds and removes a card anywhere (`Card:remove`).
pub(crate) fn remove_card_by_id(g: &mut Game, id: u32) {
	let (c, owned) = if let Some((area, c)) = g.areas.take(id) {
		let owned = match area {
			Area::Jokers | Area::Consumables => c.core().added_to_deck,
			Area::Hand | Area::Play | Area::Deck | Area::Discard => true,
			Area::ShopJokers | Area::ShopVouchers | Area::ShopBoosters | Area::PackCards => false,
		};
		(c, owned)
	} else if let Some(i) = areas::position_of(&g.areas.consumed, id) {
		let c = g.areas.consumed.remove(i);
		let owned = c.core.added_to_deck;
		(AnyCard::from(c), owned)
	} else {
		return;
	};
	if owned {
		card_lifecycle::remove_owned(g, c);
	} else {
		card_lifecycle::destroy_card(g, c);
	}
}

/// `copy_card(other, nil, nil, nil, strip_edition)` for a joker or a
/// consumable.
///
/// `c` is the caller's clone of the card to copy; it becomes the copy.
pub(crate) fn copy_card(
	areas: &mut Areas, history: &mut History, discount_percent: f64, hands: &HandLevels,
	rng: &mut PseudoRandom, mut c: Card, strip_edition: bool,
) -> Card {
	// set_ability(center) marks the center as used; ability is then copied.
	card_lifecycle::mark_used(&mut history.used_jokers, &c);
	c.queue_negative_removal = false;
	c.getting_sliced = false;
	c.ankh_deletable = false;
	c.temp_edition = false;
	// set_ability(center) rolls a To Do List hand before the copied
	// ability overwrites it: only the stream's advance matters.
	if c.is_joker(Joker::TodoList) {
		let _rolled = hands.random_visible(rng, "to_do", None);
	}
	copy_core(areas, discount_percent, c, strip_edition)
}

/// `copy_card(other, nil, nil, playing_card)` (Cryptid, DNA).
///
/// `c` is the caller's clone of the card to copy; it becomes the copy.
pub(crate) fn copy_playing_card(
	areas: &mut Areas, discount_percent: f64, mut c: PlayingCard,
) -> PlayingCard {
	c.shattered = false;
	c.destroyed = false;
	c.lucky_trigger = false;
	c.vampired = false;
	// set_base with a fresh card: times_played reset, suit_nominal_original
	// from the current suit.
	c.times_played = 0;
	c.orig_suit = c.suit;
	copy_core(areas, discount_percent, c, /* strip_edition= */ false)
}

/// The part of `copy_card` every card has: a new id, out of the deck, face
/// up, the edition (unless stripped) and the cost that goes with it, and
/// the debuff kept.
fn copy_core<C: CardLike>(
	areas: &mut Areas, discount_percent: f64, mut c: C, strip_edition: bool,
) -> C {
	let core = c.core_mut();
	core.id = areas.next_id();
	core.added_to_deck = false;
	// A new Card starts face up.
	core.face_down = false;
	if strip_edition {
		core.edition = None;
	}
	Pricing::new(discount_percent, &areas.jokers).set_cost(&mut c, None);
	c
}

/// `Blind:set_blind(nil, true)`: re-evaluate playing-card debuffs.
pub(crate) fn refresh_playing_debuffs(areas: &mut Areas, blind: &Blind) {
	let debuff = CardDebuff::new(blind, &areas.jokers);
	for c in areas.playing_cards_mut() {
		debuff.apply(c);
	}
}

/// `Game:update_selecting_hand`. With hand, deck and play all empty the
/// Lua calls `end_round()` on every frame until its first event resolves,
/// queueing several end-of-round passes; the simulator queues one. The
/// extra passes only matter after the run is lost (or saved by Mr. Bones).
pub(crate) fn update_selecting_hand(flow: &mut Flow, areas: &Areas, events: &mut EventQueue) {
	let out = areas.hand.cards.is_empty() && areas.deck.is_empty();
	if out && areas.play.is_empty() && !flow.end_round_pending {
		flow.end_round_pending = true;
		events.push_after(0.2, Event::Flow(FlowEvent::EndRound));
	}
	if !flow.state_complete {
		flow.state_complete = true;
		if out && !flow.end_round_pending {
			flow.end_round_pending = true;
			events.push_after(0.2, Event::Flow(FlowEvent::EndRound));
		}
	}
}

pub(crate) fn update_hand_played(flow: &mut Flow, events: &mut EventQueue) {
	if !flow.state_complete {
		flow.state_complete = true;
		events.push(Event::Flow(FlowEvent::HandPlayedCheck));
	}
}

pub(crate) fn update_draw_to_hand(
	flow: &mut Flow, round: &mut Round, tags: &mut Tags, events: &mut EventQueue,
) {
	if !flow.state_complete {
		flow.state_complete = true;
		tags::apply_round_start_bonus(round, tags, events);
		events.push(Event::Flow(FlowEvent::DrawToHand));
	}
}

pub(crate) fn update_new_round(flow: &mut Flow, events: &mut EventQueue) {
	if !flow.state_complete {
		flow.state_complete = true;
		events.push_after(0.2, Event::Flow(FlowEvent::EndRound));
	}
}

pub(crate) fn update_blind_select(flow: &mut Flow, events: &mut EventQueue) {
	if !flow.state_complete {
		flow.state_complete = true;
		// save_run
		events.push(Event::Wait);
		events.push(Event::Flow(FlowEvent::BlindSelectInit));
	}
}

pub(crate) fn update_round_eval(flow: &mut Flow, events: &mut EventQueue) {
	if !flow.state_complete {
		flow.state_complete = true;
		events.push(Event::Flow(FlowEvent::RoundEvalInit));
	}
}

/// Derives the agent-facing phase from `G.STATE` once the queue is empty.
///
/// # Panics
///
/// Panics if the state is one the game only passes through (a simulator
/// bug).
pub(crate) fn sync_phase(flow: &mut Flow) {
	flow.phase = match flow.state {
		State::SelectingHand => Phase::Round,
		State::Shop => Phase::Shop,
		State::BlindSelect => Phase::BlindSelect,
		State::RoundEval => Phase::RoundEval,
		State::GameOver => Phase::GameOver,
		State::TarotPack
		| State::PlanetPack
		| State::SpectralPack
		| State::StandardPack
		| State::BuffoonPack => Phase::Pack,
		transient
		@ (State::HandPlayed | State::DrawToHand | State::NewRound | State::PlayTarot) => {
			panic!("queue drained in transient state {transient:?}")
		}
	};
}

/// The events of the round flow: blind select, the round's start, plays
/// and the end of the round.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum FlowEvent {
	BlindSelectInit,
	BlindSelectUi,
	SelectBlind {
		blind: BlindKind,
	},
	NewRound,
	NewRoundBody,
	NewRoundDraw,
	/// `Blind:set_blind`'s HUD event (non-blockable, queues a second one).
	BlindSetHud,
	DrawToHand,
	EnterSelecting,
	PlayStart,
	PlayChain,
	EvaluatePlay,
	AfterPlay,
	StateCompleteFalse,
	HandPlayedCheck,
	HookDiscard,
	ToothOuter,
	EndRound,
	EndRoundFinal,
	/// The discard pile going back to the deck after a round.
	DiscardToDeck,
	/// A Blue Seal's planet for the last hand played.
	BlueSealPlanet,
	RoundEvalInit,
	RoundEvalRun,
	/// `Blind:defeat()` at the start of the round evaluation.
	BlindDefeat,
	CashOutFinish,
	BlindDisableCheck,
	AcornOuter,
	AcornShuffle,
	SkipBlindRun,
	RerollBossRun,
}

pub(crate) fn run_event(g: &mut Game, e: FlowEvent) {
	use FlowEvent as F;
	match e {
		F::BlindSelectInit => g.events.push(Event::Flow(F::BlindSelectUi)),
		F::BlindSelectUi => blind_select_ui(g),
		F::SelectBlind { blind } => select_blind(&mut g.blind_select, &mut g.events, blind),
		// new_round()
		F::NewRound => {
			g.events.delay(0.4);
			g.events.push(Event::Flow(F::NewRoundBody));
		}
		F::NewRoundBody => new_round(g),
		F::BlindSetHud => g.events.barrier(Trigger::After(0.15)),
		F::NewRoundDraw => {
			g.flow.state = State::DrawToHand;
			let key = g.run.key_ante("nr");
			shuffle_area(&mut g.areas.deck, &mut g.rng, &key);
			g.flow.state_complete = false;
		}
		F::DrawToHand => draw_to_hand(g),
		F::EnterSelecting => {
			g.flow.state = State::SelectingHand;
			g.flow.state_complete = false;
			blind::drawn_to_hand(g);
		}
		F::PlayStart => {
			g.flow.state = State::HandPlayed;
			g.flow.state_complete = true;
		}
		F::PlayChain => {
			g.events.push(Event::Flow(F::EvaluatePlay));
			g.events.push_after(0.1, Event::Flow(F::AfterPlay));
			g.events.push(Event::Flow(F::StateCompleteFalse));
		}
		F::EvaluatePlay => scoring::evaluate_play(g),
		F::AfterPlay => after_play(&g.areas.play, &mut g.run, &mut g.round, &mut g.events),
		F::StateCompleteFalse => g.flow.state_complete = false,
		F::HandPlayedCheck => {
			g.flow.state = if g.round.beats(g.blind.chips) || g.round.hands_left < 1.0 {
				State::NewRound
			} else {
				State::DrawToHand
			};
			g.flow.state_complete = false;
		}
		F::HookDiscard => hook_discard(g),
		F::ToothOuter => {
			for _ in 0..g.areas.play.len() {
				g.events.push(Event::Wait);
				g.events.push(Event::EaseDollars(-1.0));
				g.events.delay(0.23);
			}
		}
		F::EndRound => round_end::end_round(g),
		F::EndRoundFinal => round_end::end_round_final(g),
		F::DiscardToDeck => {
			for _ in 0..g.areas.discard.len() {
				g.events.draw(Draw::next(Pile::Discard, Pile::Deck), 0.005);
			}
		}
		F::BlueSealPlanet => round_end::blue_seal_planet(g),
		F::RoundEvalInit => {
			g.round.facing_blind = false;
			// Polls until the round-eval UI has slid in.
			g.events.push_event(
				Event::Flow(F::RoundEvalRun),
				Trigger::Frames(21),
				Blocks::InOrder,
			);
		}
		F::RoundEvalRun => {
			g.events.delay(0.1);
			round_end::evaluate_round(g);
		}
		F::BlindDefeat => blind::defeat(&mut g.areas.jokers, &g.blind, &mut g.events),
		F::CashOutFinish => enter_shop(&mut g.flow, &mut g.round, &mut g.shop),
		F::BlindDisableCheck => {
			if g.blind.is_boss() && g.round.beats(g.blind.chips) {
				g.flow.state = State::NewRound;
				g.flow.state_complete = false;
			}
		}
		F::AcornOuter => {
			g.events.push(Event::Flow(F::AcornShuffle));
			g.events.delay(0.15);
			g.events.push(Event::Flow(F::AcornShuffle));
			g.events.delay(0.15);
			g.events.push(Event::Flow(F::AcornShuffle));
			g.events.delay(0.5);
		}
		F::AcornShuffle => shuffle_area(&mut g.areas.jokers, &mut g.rng, &Key::from("aajk")),
		F::SkipBlindRun => skip_blind_run(g),
		F::RerollBossRun => reroll_boss_run(g),
	}
}

/// The on-deck blind becoming the current one.
fn select_blind(blind_select: &mut BlindSelect, events: &mut EventQueue, blind: BlindKind) {
	events.push(Event::EaseRound);
	blind_select.selected = Some(blind);
	if let Some(s) = blind_select.on_deck {
		blind_select.states[s] = BlindStatus::Current;
	}
	events.delay(0.2);
}

/// The end of cashing out: hands and discards reset, on to the shop.
fn enter_shop(flow: &mut Flow, round: &mut Round, shop: &mut Shop) {
	round.reset_hands_and_discards();
	flow.state = State::Shop;
	shop.free = false;
	shop.d6ed = false;
	flow.state_complete = false;
}

/// A new boss for the ante, with its Orbital hand and the waiting tags.
fn reroll_boss_run(g: &mut Game) {
	g.blind_select.boss = pools::get_new_boss(
		g.setup.win_ante,
		g.run.ante,
		&mut g.blind_select,
		&mut g.rng,
	);
	if let Some(s) = g.blind_select.on_deck {
		g.blind_select.states[s] = BlindStatus::Select;
	}
	ensure_orbital(
		g.run.ante,
		&g.hands,
		&mut g.tags,
		&mut g.rng,
		BlindSlot::Boss,
	);
	tags::apply_new_blind_choice(&mut g.tags, &mut g.events);
}

/// The hand drawn at the start of a round or after a play or discard.
fn draw_to_hand(g: &mut Game) {
	if draw_from_deck_to_hand(
		&mut g.flow,
		&g.areas,
		&g.round,
		&g.blind,
		&mut g.events,
		None,
	) {
		return;
	}
	if g.round.hands_played == 0.0 && g.round.discards_used == 0.0 && g.round.facing_blind {
		calculate_joker::calc_jokers(g, LifecycleCtx::FirstHandDrawn);
	}
	g.events.push(Event::Flow(FlowEvent::EnterSelecting));
}

/// The played cards going to the discard pile, and the hand counters.
fn after_play(play: &[PlayingCard], run: &mut Run, round: &mut Round, events: &mut EventQueue) {
	for c in play {
		if !c.shattered && !c.destroyed {
			events.draw(Draw::card(Pile::Play, Pile::Discard, c.core.id), 0.1);
		}
	}
	run.hands_played += 1.0;
	round.hands_played += 1.0;
}

/// The jokers and tags reacting to a skipped blind.
fn skip_blind_run(g: &mut Game) {
	g.events.delay(0.3);
	calculate_joker::calc_jokers(g, LifecycleCtx::SkipBlind);
	tags::apply_immediate(&g.run, &mut g.hands, &mut g.tags, &mut g.events);
	tags::apply_new_blind_choice(&mut g.tags, &mut g.events);
	// The blind-choice UI notices the new on-deck blind this frame
	// (blind_choice_handler) and queues a focus event.
	g.events.push(Event::Wait);
}

fn orbital_slot(ante: i32, tags: &mut Tags) -> &mut PerBlind<Option<HandType>> {
	if let Some(i) = tags.orbital_choices.iter().position(|o| o.ante == ante) {
		&mut tags.orbital_choices[i].hands
	} else {
		tags.orbital_choices.push(OrbitalChoices {
			ante,
			hands: PerBlind::default(),
		});
		let n = tags.orbital_choices.len() - 1;
		&mut tags.orbital_choices[n].hands
	}
}

pub(crate) fn orbital_for(ante: i32, tags: &Tags, slot: BlindSlot) -> Option<HandType> {
	tags.orbital_choices
		.iter()
		.find(|o| o.ante == ante)
		.and_then(|o| o.hands[slot])
}

pub(crate) fn ensure_orbital(
	ante: i32, hands: &HandLevels, tags: &mut Tags, rng: &mut PseudoRandom, slot: BlindSlot,
) {
	let choice = &mut orbital_slot(ante, tags)[slot];
	if choice.is_none() {
		*choice = Some(hands.random_visible(rng, "orbital", None));
	}
}

/// `create_UIBox_blind_select` + the tag passes of `update_blind_select`.
fn blind_select_ui(g: &mut Game) {
	let s = &g.blind_select.states;
	let on_deck = if !matches!(
		s[BlindSlot::Small],
		BlindStatus::Defeated | BlindStatus::Skipped
	) {
		BlindSlot::Small
	} else if !matches!(
		s[BlindSlot::Big],
		BlindStatus::Defeated | BlindStatus::Skipped
	) {
		BlindSlot::Big
	} else {
		BlindSlot::Boss
	};
	g.blind_select.on_deck = Some(on_deck);
	g.blind_select.states[on_deck] = BlindStatus::Select;
	for slot in BlindSlot::ALL {
		ensure_orbital(g.run.ante, &g.hands, &mut g.tags, &mut g.rng, slot);
	}
	tags::apply_immediate(&g.run, &mut g.hands, &mut g.tags, &mut g.events);
	tags::apply_new_blind_choice(&mut g.tags, &mut g.events);
}

/// `G.FUNCS.select_blind` for the blind on deck.
///
/// # Panics
///
/// Panics if no blind is on deck (the blind-select phase always has one).
pub(crate) fn do_select_blind(
	blind_select: &BlindSelect, round: &mut Round, events: &mut EventQueue,
) {
	let b = blind_select
		.on_deck_kind()
		.expect("`blind_select_ui` puts a blind on deck before the phase is offered");
	round.facing_blind = true;
	events.wait(Trigger::Before(0.2));
	events.push(Event::Flow(FlowEvent::SelectBlind { blind: b }));
	events.push(Event::Flow(FlowEvent::NewRound));
}

/// `G.FUNCS.skip_blind` for the blind on deck.
///
/// # Panics
///
/// Panics if the boss or no blind is on deck, or the ante's skip tags have
/// not been rolled.
pub(crate) fn do_skip_blind(
	blind_select: &mut BlindSelect, run: &mut Run, tags: &mut Tags, events: &mut EventQueue,
) {
	let slot = blind_select
		.on_deck
		.filter(|&slot| slot != BlindSlot::Boss)
		.expect("only the small and the big blind can be skipped");
	let kind = blind_select
		.skip_tags
		.of(slot)
		.expect("the skip tags are rolled at the start of every ante");
	run.skips += 1.0;
	run.blinds_cleared += 1;
	let orbital = if kind == TagKind::Orbital {
		orbital_for(run.ante, tags, slot)
	} else {
		None
	};
	tags::add_tag(tags, events, kind, orbital);
	let to = if slot == BlindSlot::Small {
		BlindSlot::Big
	} else {
		BlindSlot::Boss
	};
	blind_select.states[slot] = BlindStatus::Skipped;
	blind_select.states[to] = BlindStatus::Select;
	blind_select.on_deck = Some(to);
	events.push(Event::Flow(FlowEvent::SkipBlindRun));
}

/// What the player pays to reroll the boss (the Boss Tag rerolls for free).
pub(crate) const BOSS_REROLL_COST: f64 = 10.0;

/// `G.FUNCS.reroll_boss`, after the payment.
pub(crate) fn do_reroll_boss(blind_select: &mut BlindSelect, events: &mut EventQueue) {
	blind_select.boss_rerolled = true;
	events.push(Event::Wait);
	events.push_after(0.3, Event::Flow(FlowEvent::RerollBossRun));
}

/// The counters `new_round` resets: the round's hands and discards, and
/// each poker hand's plays this round.
pub(crate) fn reset_round_counters(round: &mut Round, hands: &mut HandLevels) {
	round.reset_hands_and_discards();
	round.hands_played = 0.0;
	round.discards_used = 0.0;
	for h in hands {
		h.played_this_round = 0.0;
	}
}

fn new_round(g: &mut Game) {
	reset_round_counters(&mut g.round, &mut g.hands);
	g.shop.reroll_cost_increase = 0.0;
	g.shop.booster_slots = [PackSlot::Unrolled; BOOSTER_SLOTS];
	for c in g.areas.playing_cards_mut() {
		c.wheel_flipped = false;
	}
	g.shop.free_rerolls = num::float(g.areas.count_joker(Joker::Chaos));
	shop::calculate_reroll_cost(&mut g.shop);
	g.round.bonus_next_hands = 0.0;
	g.round.bonus_discards = 0.0;
	let blind = g
		.blind_select
		.selected
		.expect("new_round without a selected blind");
	g.blind_select.states[BlindSlot::of(blind)] = BlindStatus::Current;
	blind::set_blind(g, Some(blind));
	let boss = blind.is_boss();
	calculate_joker::calc_jokers(g, LifecycleCtx::SettingBlind { boss });
	g.flow.end_round_pending = false;
	g.events.delay(0.4);
	g.events.push(Event::Flow(FlowEvent::NewRoundDraw));
}

/// `G.FUNCS.play_cards_from_highlighted` with the hand cards at the
/// positions `sel`; under Steamodded, which marks a played card for The
/// Pillar only once the hand is evaluated ([`Event::MarkPlayed`]).
///
/// # Panics
///
/// Panics if `sel` holds a position past the end of the hand.
pub(crate) fn do_play(
	flow: &mut Flow, areas: &mut Areas, blind: &mut Blind, events: &mut EventQueue, sel: CardMask,
	steamodded: bool,
) {
	if !areas.play.is_empty() {
		return;
	}
	blind.triggered = false;
	for c in areas.playing_cards_mut() {
		c.forced_selection = false;
	}
	areas.play.reserve(sel.len());
	flow.hand_text_cleared = false;
	events.push(Event::Flow(FlowEvent::PlayStart));
	events.push(Event::EaseHands(-1.0));
	events.delay(0.4);
	for i in sel.indices() {
		let c = &mut areas.hand.cards[i];
		c.times_played += 1;
		if !steamodded {
			c.played_this_ante = true;
		}
		let id = c.core.id;
		events.draw(Draw::card(Pile::Hand, Pile::Play, id), 0.1);
	}
	if blind::press_play(&areas.jokers, blind, events) {
		events.push(Event::Wait);
		events.delay(0.4);
	}
	events.push(Event::Flow(FlowEvent::PlayChain));
}

fn hook_discard(g: &mut Game) {
	// pseudorandom_element picks from the remaining cards sorted by sort_id.
	let mut cards: Vec<usize> = (0..g.areas.hand.cards.len()).collect();
	cards.sort_by_key(|&i| g.areas.hand.cards[i].core.id);
	let mut chosen = [0; 2];
	let n = cards.len().min(chosen.len());
	for slot in &mut chosen[..n] {
		let k = Key::from("hook");
		let pick = g.rng.element(&k, cards.len());
		*slot = cards.remove(pick);
	}
	if n > 0 {
		do_discard(g, &mut chosen[..n], /* hook= */ true);
	}
}

/// `G.discard.config.card_limit`.
const DISCARD_LIMIT: usize = 500;

/// `G.FUNCS.discard_cards_from_highlighted(e, hook)` with the hand cards at
/// the positions `sel`.
///
/// # Panics
///
/// Panics if `sel` holds a position past the end of the hand.
pub(crate) fn do_discard(g: &mut Game, sel: &mut [usize], hook: bool) {
	for c in g.areas.playing_cards_mut() {
		c.forced_selection = false;
	}
	sel.sort_unstable();
	let count = sel
		.len()
		.min(DISCARD_LIMIT.saturating_sub(g.areas.play.len()));
	if count == 0 {
		return;
	}
	let sel = &sel[..count];
	// update_hand_text({immediate = true, delay = 0}, {handname = ''})
	g.events.hand_text_immediate(0.0);
	g.flow.hand_text_cleared = true;
	let burnt_hand = if g
		.areas
		.jokers
		.iter()
		.any(|c| c.kind == Kind::Joker(Joker::Burnt))
	{
		let refs: Vec<&PlayingCard> = sel.iter().map(|&i| &g.areas.hand.cards[i]).collect();
		hand_eval::evaluate(&refs, g.areas.eval_flags())
			.top
			.map(|top| top.hand)
	} else {
		None
	};
	calculate_joker::calc_jokers(g, LifecycleCtx::PreDiscard { hook, burnt_hand });
	let par = g.areas.pareidolia();
	let full_faces = sel
		.iter()
		.filter(|&&i| g.areas.hand.cards[i].is_face(par))
		.count();
	let last = sel[count - 1];
	let mut destroyed: Vec<usize> = Vec::new();
	for &i in sel {
		let card = &g.areas.hand.cards[i];
		let purple = !card.core.debuff && card.seal == Some(Seal::Purple);
		let id = card.core.id;
		// Card:calculate_seal({discard = true})
		if purple && g.areas.consumable_room_after_queued() {
			calculate_joker::queue_consumable(&mut g.areas, &mut g.events, CardSet::Tarot, "8ba");
			g.events.status_extra();
		}
		let mut removed = false;
		for j in 0..g.areas.jokers.len() {
			if let Some(e) = calculate_joker::calc_joker(
				g,
				j,
				LifecycleCtx::Discard(Discarded {
					other: i,
					last,
					n_full: count,
					full_faces,
				}),
			) {
				if e.remove {
					removed = true;
				}
				calculate_joker::effect_status(&mut g.events, &e);
			}
		}
		if removed {
			destroyed.push(i);
			g.areas.hand.cards[i].shatter_or_dissolve(&mut g.events);
		} else {
			g.events
				.draw(Draw::card(Pile::Hand, Pile::Discard, id), 0.1);
		}
	}
	if !destroyed.is_empty() {
		let faces = destroyed
			.iter()
			.filter(|&&i| g.areas.hand.cards[i].is_face(par))
			.count();
		let glass = destroyed
			.iter()
			.filter(|&&i| g.areas.hand.cards[i].shattered)
			.count();
		calculate_joker::calc_jokers(g, LifecycleCtx::RemovePlayingCards { faces, glass });
	}
	if !hook {
		g.events.push(Event::EaseDiscard(-1.0));
		g.round.discards_used += 1.0;
		g.flow.state = State::DrawToHand;
		g.events.push(Event::Flow(FlowEvent::StateCompleteFalse));
	}
}
