//! Round flow: blind select, new round, drawing, play / discard and the
//! per-state updates (`state_events.lua`, `button_callbacks.lua`, game.lua
//! `update_*`). The end of the round is in `round_end`.

use crate::actions::{CardMask, mask_bits};
use crate::card::{Card, Kind};
use crate::data::{
	BlindKind, Edition, Enhancement, HandType, Joker, P_CARDS_SORTED, Seal, TagKind,
};
use crate::ev::{Ev, Pile};
use crate::events::St;
use crate::game::{Area, BlindSlot, BlindStatus, CardSet, Game, OrbitalChoices, PackSlot, Phase};
use crate::hand_eval::{self};
use crate::jokers::Ctx;
use crate::rng::{Key, shuffle};
use crate::{blind, cards, jokers, num, pools, round_end, scoring, tags};

// -----------------------------------------------------------------
// Piles
// -----------------------------------------------------------------

pub(crate) fn pile(g: &mut Game, p: Pile) -> &mut Vec<Card> {
	match p {
		Pile::Deck => &mut g.deck,
		Pile::Hand => &mut g.hand,
		Pile::Play => &mut g.play,
		Pile::Discard => &mut g.discard,
	}
}

/// `CardArea:shuffle(seed)`: sort by `sort_id`, then Fisher–Yates.
pub(crate) fn shuffle_deck(g: &mut Game, key: &Key) {
	g.deck.sort_unstable_by_key(|c| c.id);
	let seed = g.rng.seed_for(key);
	shuffle(&mut g.deck, seed);
}

/// `G.hand:sort()` with the area's current sort mode.
pub(crate) fn sort_hand(g: &mut Game) {
	let by_suit = g.hand_sort_by_suit;
	g.hand.sort_by(|a, b| {
		let (x, y) = (a.nominal(by_suit), b.nominal(by_suit));
		y.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Equal)
	});
}

/// `CardArea:emplace` for piles (deck inserts at the front).
fn emplace(g: &mut Game, to: Pile, mut c: Card, stay_flipped: bool) {
	match to {
		Pile::Deck => {
			c.face_down = true;
			g.deck.insert(0, c);
		}
		Pile::Discard => {
			c.face_down = true;
			g.discard.push(c);
		}
		Pile::Hand | Pile::Play => {
			if c.face_down && !stay_flipped {
				c.face_down = false;
				c.ab.wheel_flipped = false;
			}
			if to == Pile::Hand && stay_flipped {
				c.ab.wheel_flipped = true;
				c.face_down = true;
			}
			if to == Pile::Hand {
				g.hand.push(c);
			} else {
				g.play.push(c);
			}
		}
	}
}

/// The body of `draw_card`'s event.
pub(crate) fn draw_card(
	g: &mut Game, from: Pile, to: Pile, card: Option<u32>, sort: bool, discarded_only: bool,
) {
	let taken = if let Some(id) = card {
		let src = pile(g, from);
		src.iter().position(|c| c.id == id).map(|i| src.remove(i))
	} else {
		// CardArea:draw_card_from → area:remove_card(nil, discarded_only)
		let src_is_stack = matches!(from, Pile::Deck | Pile::Discard);
		let src = pile(g, from);
		if discarded_only {
			let pick = if src_is_stack {
				src.iter().rposition(|c| c.ab.discarded)
			} else {
				src.iter().position(|c| c.ab.discarded)
			};
			pick.map(|i| src.remove(i))
		} else if src.is_empty() {
			None
		} else if src_is_stack {
			src.pop()
		} else {
			Some(src.remove(0))
		}
	};
	let Some(c) = taken else { return };
	let stay = blind::stay_flipped(g, to, &c);
	crate::game::trace!(g, "draw {} {}", to.name(), c.id);
	emplace(g, to, c, stay);
	if sort && to == Pile::Hand {
		sort_hand(g);
	}
}

/// `G.FUNCS.draw_from_deck_to_hand(e)`; returns true on the game-over path.
pub(crate) fn draw_from_deck_to_hand(g: &mut Game, e: Option<usize>) -> bool {
	if !matches!(g.state, St::TarotPack | St::SpectralPack)
		&& g.hand_limit.limit <= 0.0
		&& g.hand.is_empty()
	{
		g.state = St::GameOver;
		g.state_complete = false;
		return true;
	}
	let mut hand_space: usize = match e {
		Some(n) => n,
		None => g
			.deck
			.len()
			.min(num::trunc(g.hand_limit.limit).saturating_sub(g.hand.len())),
	};
	if g.blind.active(BlindKind::Serpent) && (g.hands_played_round > 0.0 || g.discards_used > 0.0) {
		hand_space = g.deck.len().min(3);
	}
	g.events.delay(0.3);
	for _ in 0..hand_space {
		g.events
			.draw_ev(Pile::Deck, Pile::Hand, None, true, false, 0.1);
	}
	false
}

/// Finds and removes a card anywhere (`Card:remove`).
pub(crate) fn remove_card_by_id(g: &mut Game, id: u32) {
	macro_rules! try_area {
		($v:ident, $owned:expr) => {
			if let Some(i) = g.$v.iter().position(|c| c.id == id) {
				let c = g.$v.remove(i);
				if $owned(&c) {
					cards::remove_owned(g, c);
				} else {
					cards::destroy_card(g, c);
				}
				return;
			}
		};
	}
	let yes = |_: &Card| true;
	let no = |_: &Card| false;
	let added = |c: &Card| c.added_to_deck;
	try_area!(jokers, added);
	try_area!(consumables, added);
	try_area!(hand, yes);
	try_area!(play, yes);
	try_area!(deck, yes);
	try_area!(discard, yes);
	try_area!(shop_jokers, no);
	try_area!(shop_vouchers, no);
	try_area!(shop_boosters, no);
	try_area!(pack_cards, no);
	try_area!(consumed, |c: &Card| c.added_to_deck);
}

/// `copy_card(other, nil, nil, playing_card, strip_edition)`.
///
/// `c` is the caller's clone of the card to copy; it becomes the copy.
pub(crate) fn copy_card(g: &mut Game, mut c: Card, strip_edition: bool) -> Card {
	let (edition, debuff) = (c.edition, c.debuff);
	c.id = g.new_card(c.kind).id;
	// set_ability(center) marks the center as used; ability is then copied.
	cards::mark_used(g, &c);
	c.added_to_deck = false;
	c.queue_negative_removal = false;
	c.getting_sliced = false;
	c.shattered = false;
	c.destroyed = false;
	c.lucky_trigger = false;
	c.vampired = false;
	c.temp_edition = false;
	// A new Card starts face up; set_base resets times_played.
	c.face_down = false;
	c.times_played = 0;
	// set_ability(center) rolls a To Do List hand before the copied
	// ability overwrites it: the stream still advances.
	if c.is_joker(Joker::TodoList) {
		let (hands, n) = g.visible_hands_pairs(None);
		let k = Key::from("to_do");
		let _ = hands[g.rng.element(&k, n)];
	}
	if c.is_playing() {
		// set_base with a fresh card: suit_nominal_original from the current suit.
		c.orig_suit = Some(c.suit);
	}
	c.edition = Edition::None;
	if !strip_edition {
		c.edition = edition;
	}
	cards::set_cost(g, &mut c, None);
	c.debuff = debuff;
	c
}

/// `Blind:set_blind(nil, true)`: re-evaluate playing-card debuffs.
pub(crate) fn refresh_playing_debuffs(g: &mut Game) {
	let (blind, smeared, pareidolia) = (g.blind, g.smeared(), g.pareidolia());
	for c in g.playing_cards_mut() {
		let debuffed = crate::blind::debuffs(&blind, smeared, pareidolia, c);
		cards::set_debuff_playing(c, debuffed);
	}
}

// -----------------------------------------------------------------
// State handlers (Game:update_*)
// -----------------------------------------------------------------

/// `Game:update_selecting_hand`. With hand, deck and play all empty the
/// Lua calls `end_round()` on every frame until its first event resolves,
/// queueing several end-of-round passes; the simulator queues one. The
/// extra passes only matter after the run is lost (or saved by Mr. Bones).
pub(crate) fn update_selecting_hand(g: &mut Game) {
	let out = g.hand.is_empty() && g.deck.is_empty();
	if out && g.play.is_empty() && !g.end_round_pending {
		g.end_round_pending = true;
		g.events.push_after(0.2, Ev::EndRound);
	}
	if !g.state_complete {
		g.state_complete = true;
		if out && !g.end_round_pending {
			g.end_round_pending = true;
			g.events.push_after(0.2, Ev::EndRound);
		}
	}
}

pub(crate) fn update_hand_played(g: &mut Game) {
	if !g.state_complete {
		g.state_complete = true;
		g.events.push(Ev::HandPlayedCheck);
	}
}

pub(crate) fn update_draw_to_hand(g: &mut Game) {
	if !g.state_complete {
		g.state_complete = true;
		for i in 0..g.tags.len() {
			tags::apply_tag(g, i, crate::tags::TagCtx::RoundStartBonus);
		}
		g.events.push(Ev::DrawToHand);
	}
}

pub(crate) fn update_new_round(g: &mut Game) {
	if !g.state_complete {
		g.state_complete = true;
		g.events.push_after(0.2, Ev::EndRound);
	}
}

pub(crate) fn update_blind_select(g: &mut Game) {
	if !g.state_complete {
		g.state_complete = true;
		// save_run
		g.events.push(Ev::Wait);
		g.events.push(Ev::BlindSelectInit);
	}
}

pub(crate) fn update_round_eval(g: &mut Game) {
	if !g.state_complete {
		g.state_complete = true;
		g.events.push(Ev::RoundEvalInit);
	}
}

/// Derives the agent-facing phase from `G.STATE` once the queue is empty.
pub(crate) fn sync_phase(g: &mut Game) {
	g.phase = match g.state {
		St::SelectingHand => Phase::Round,
		St::Shop => Phase::Shop,
		St::BlindSelect => Phase::BlindSelect,
		St::RoundEval => Phase::RoundEval,
		St::GameOver => Phase::GameOver,
		s if s.is_pack() => Phase::Pack,
		other => panic!("queue drained in transient state {other:?}"),
	};
}

// -----------------------------------------------------------------
// Flow events
// -----------------------------------------------------------------

/// Runs a round-flow event (routed by `run_event`).
#[expect(clippy::too_many_lines, reason = "one arm per event")]
pub(crate) fn run_flow_event(g: &mut Game, e: Ev) {
	match e {
		Ev::BlindSelectInit => g.events.push(Ev::BlindSelectUi),
		Ev::BlindSelectUi => blind_select_ui(g),
		Ev::SelectBlind { blind } => {
			g.events.push(Ev::EaseRound);
			g.selected_blind = Some(blind);
			if let Some(s) = g.blind_on_deck {
				g.blind_states[s as usize] = BlindStatus::Current;
			}
			g.events.delay(0.2);
		}
		// new_round()
		Ev::NewRound => {
			g.events.delay(0.4);
			g.events.push(Ev::NewRoundBody);
		}
		Ev::NewRoundBody => new_round(g),
		Ev::BlindSetHud => {
			g.events
				.push_ev(Ev::Wait, crate::events::Trig::After(0.15), true, false);
		}
		Ev::NewRoundDraw => {
			g.state = St::DrawToHand;
			let key = g.key_ante("nr");
			shuffle_deck(g, &key);
			g.state_complete = false;
		}
		Ev::DrawToHand => {
			if draw_from_deck_to_hand(g, None) {
				return;
			}
			if g.hands_played_round == 0.0 && g.discards_used == 0.0 && g.facing_blind {
				for i in 0..g.jokers.len() {
					jokers::calc_joker(g, i, &Ctx::FirstHandDrawn, None);
				}
			}
			g.events.push(Ev::EnterSelecting);
		}
		Ev::EnterSelecting => {
			g.state = St::SelectingHand;
			g.state_complete = false;
			blind::drawn_to_hand(g);
		}
		Ev::PlayStart => {
			g.state = St::HandPlayed;
			g.state_complete = true;
		}
		Ev::PlayChain => {
			g.events.push(Ev::EvaluatePlay);
			g.events.push_after(0.1, Ev::AfterPlay);
			g.events.push(Ev::StateCompleteFalse);
		}
		Ev::EvaluatePlay => scoring::evaluate_play(g),
		Ev::AfterPlay => {
			for i in 0..g.play.len() {
				let c = &g.play[i];
				if !c.shattered && !c.destroyed {
					let id = c.id;
					g.events
						.draw_ev(Pile::Play, Pile::Discard, Some(id), false, false, 0.1);
				}
			}
			g.hands_played += 1.0;
			g.hands_played_round += 1.0;
		}
		Ev::StateCompleteFalse => g.state_complete = false,
		Ev::HandPlayedCheck => {
			g.state = if g.chips - g.blind.chips >= 0.0 || g.hands_left < 1.0 {
				St::NewRound
			} else {
				St::DrawToHand
			};
			g.state_complete = false;
		}
		Ev::HookDiscard => hook_discard(g),
		Ev::ToothOuter => {
			for _ in 0..g.play.len() {
				g.events.push(Ev::Wait);
				g.events.push(Ev::EaseDollars(-1.0));
				g.events.delay(0.23);
			}
		}
		Ev::EndRound => round_end::end_round(g),
		Ev::EndRoundFinal => round_end::end_round_final(g),
		Ev::RoundEvalInit => {
			g.facing_blind = false;
			// Polls until the round-eval UI has slid in.
			g.events.push_ev(
				Ev::RoundEvalRun,
				crate::events::Trig::Frames(21),
				true,
				true,
			);
		}
		Ev::RoundEvalRun => {
			g.events.delay(0.1);
			round_end::evaluate_round(g);
		}
		Ev::CashOutFinish => {
			g.discards_left = 0f64.max(g.rr_discards + g.round_bonus_discards);
			g.hands_left = 1f64.max(g.rr_hands + g.round_bonus_next_hands);
			g.state = St::Shop;
			g.shop_free = false;
			g.shop_d6ed = false;
			g.state_complete = false;
		}
		Ev::BlindDisableCheck => {
			if g.blind.boss && g.chips - g.blind.chips >= 0.0 {
				g.state = St::NewRound;
				g.state_complete = false;
			}
		}
		Ev::AcornOuter => {
			g.events.push(Ev::AcornShuffle);
			g.events.delay(0.15);
			g.events.push(Ev::AcornShuffle);
			g.events.delay(0.15);
			g.events.push(Ev::AcornShuffle);
			g.events.delay(0.5);
		}
		Ev::AcornShuffle => {
			g.jokers.sort_unstable_by_key(|c| c.id);
			let key = Key::from("aajk");
			let seed = g.rng.seed_for(&key);
			shuffle(&mut g.jokers, seed);
		}
		Ev::SkipBlindRun => {
			g.events.delay(0.3);
			for i in 0..g.jokers.len() {
				jokers::calc_joker(g, i, &Ctx::SkipBlind, None);
			}
			for i in 0..g.tags.len() {
				tags::apply_tag(g, i, crate::tags::TagCtx::Immediate);
			}
			apply_new_blind_choice_tags(g);
			// The blind-choice UI notices the new on-deck blind this frame
			// (blind_choice_handler) and queues a focus event.
			g.events.push(Ev::Wait);
		}
		Ev::RerollBossRun => {
			g.boss_choice = pools::get_new_boss(g);
			if let Some(s) = g.blind_on_deck {
				g.blind_states[s as usize] = BlindStatus::Select;
			}
			ensure_orbital(g, BlindSlot::Boss);
			apply_new_blind_choice_tags(g);
		}
		Ev::ChicotOuter => {
			g.events.push(Ev::ChicotDisable);
			g.events.status_extra();
		}
		Ev::ChicotDisable => {
			blind::disable_blind(g);
			g.events.delay(0.4);
		}
		Ev::Burglar => {
			let d = g.discards_left;
			g.events.push(Ev::EaseDiscard(-d));
			g.events.push(Ev::EaseHands(3.0));
			g.events.status_extra();
		}
		Ev::RiffRaff { n } => {
			for _ in 0..n {
				let mut c = pools::create_card(
					g,
					CardSet::Joker,
					Area::Jokers,
					false,
					Some(0.0),
					false,
					None,
					"rif",
				);
				cards::add_to_deck(g, &mut c, false);
				g.jokers.push(c);
				g.events.materialize(1.0);
				g.joker_buffer = 0.0;
			}
		}
		Ev::CartomancerOuter => {
			g.events.push(Ev::Create {
				set: CardSet::Tarot,
				area: Area::Consumables,
				forced: None,
				append: "car",
				rarity: None,
				legendary: false,
			});
			g.events.status_extra();
		}
		Ev::DaggerSlice { dagger, victim } => {
			g.joker_buffer = 0.0;
			let value = g
				.jokers
				.iter()
				.find(|c| c.id == victim)
				.map_or(0.0, |c| c.sell_cost);
			if let Some(d) = g.jokers.iter_mut().find(|c| c.id == dagger) {
				d.ab.mult += value * 2.0;
			}
			g.events.start_dissolve_f(victim, 1.6);
		}
		Ev::MarbleStone => {
			let key = Key::from("marb_fr");
			let i = g.rng.element(&key, 52);
			let (suit, rank) = P_CARDS_SORTED[i];
			let c = cards::make_playing_card(g, suit, rank, Enhancement::Stone);
			g.events.materialize(1.0);
			g.play.push(c);
		}
		Ev::CertificateCard => {
			let key = Key::from("cert_fr");
			let i = g.rng.element(&key, 52);
			let (suit, rank) = P_CARDS_SORTED[i];
			let mut c = cards::make_playing_card(g, suit, rank, Enhancement::Base);
			let key = Key::from("certsl");
			let st = g.rng.random(&key);
			c.seal = Some(if st > 0.75 {
				Seal::Red
			} else if st > 0.5 {
				Seal::Blue
			} else if st > 0.25 {
				Seal::Gold
			} else {
				Seal::Purple
			});
			cards::set_cost(g, &mut c, None);
			g.events.materialize(1.0);
			blind::debuff_card(g, &mut c);
			g.hand.push(c);
			sort_hand(g);
		}
		Ev::GlassJokerOuter { id, n } => g.events.push(Ev::GlassJokerInner { id, n }),
		Ev::GlassJokerInner { id, n } => {
			if let Some(c) = g.jokers.iter_mut().find(|c| c.id == id) {
				c.ab.x_mult += 0.75 * f64::from(n);
			}
		}
		Ev::PerkeoCopy => round_end::perkeo_copy(g),
		other => unreachable!("not a flow event: {other:?}"),
	}
}

// -----------------------------------------------------------------
// Blind select
// -----------------------------------------------------------------

fn orbital_slot(g: &mut Game) -> &mut [Option<HandType>; 3] {
	let ante = g.ante;
	if let Some(i) = g.orbital_choices.iter().position(|o| o.ante == ante) {
		&mut g.orbital_choices[i].hands
	} else {
		g.orbital_choices.push(OrbitalChoices {
			ante,
			hands: [None; 3],
		});
		let n = g.orbital_choices.len() - 1;
		&mut g.orbital_choices[n].hands
	}
}

pub(crate) fn orbital_for(g: &Game, slot: BlindSlot) -> Option<HandType> {
	g.orbital_choices
		.iter()
		.find(|o| o.ante == g.ante)
		.and_then(|o| o.hands[slot as usize])
}

pub(crate) fn ensure_orbital(g: &mut Game, slot: BlindSlot) {
	if orbital_slot(g)[slot as usize].is_none() {
		let (hands, n) = g.visible_hands_pairs(None);
		let k = Key::from("orbital");
		let i = g.rng.element(&k, n);
		orbital_slot(g)[slot as usize] = Some(hands[i]);
	}
}

/// `create_UIBox_blind_select` + the tag passes of `update_blind_select`.
fn blind_select_ui(g: &mut Game) {
	let s = &g.blind_states;
	let on_deck = if !matches!(s[0], BlindStatus::Defeated | BlindStatus::Skipped) {
		BlindSlot::Small
	} else if !matches!(s[1], BlindStatus::Defeated | BlindStatus::Skipped) {
		BlindSlot::Big
	} else {
		BlindSlot::Boss
	};
	g.blind_on_deck = Some(on_deck);
	g.blind_states[on_deck as usize] = BlindStatus::Select;
	for slot in [BlindSlot::Small, BlindSlot::Big, BlindSlot::Boss] {
		ensure_orbital(g, slot);
	}
	if g.blind_ante == i32::MIN {
		g.blind_ante = g.ante;
	}
	for i in 0..g.tags.len() {
		tags::apply_tag(g, i, crate::tags::TagCtx::Immediate);
	}
	apply_new_blind_choice_tags(g);
}

pub(crate) fn apply_new_blind_choice_tags(g: &mut Game) {
	for i in 0..g.tags.len() {
		if tags::apply_tag(g, i, crate::tags::TagCtx::NewBlindChoice) {
			break;
		}
	}
}

pub(crate) fn do_select_blind(g: &mut Game) {
	let Some(b) = g.blind_on_deck_kind() else {
		return;
	};
	g.facing_blind = true;
	g.events.wait(crate::events::Trig::Before(0.2));
	g.events.push(Ev::SelectBlind { blind: b });
	g.events.push(Ev::NewRound);
}

pub(crate) fn do_skip_blind(g: &mut Game) {
	let Some(slot) = g.blind_on_deck else {
		return;
	};
	g.skips += 1.0;
	if slot == BlindSlot::Boss {
		return;
	}
	if let Some(kind) = g.blind_tags[slot as usize] {
		g.blinds_cleared += 1;
		let orbital = if kind == TagKind::Orbital {
			orbital_for(g, slot)
		} else {
			None
		};
		tags::add_tag(g, kind, orbital);
		let to = if slot == BlindSlot::Small {
			BlindSlot::Big
		} else {
			BlindSlot::Boss
		};
		g.blind_states[slot as usize] = BlindStatus::Skipped;
		g.blind_states[to as usize] = BlindStatus::Select;
		g.blind_on_deck = Some(to);
		g.events.push(Ev::SkipBlindRun);
	}
}

pub(crate) fn do_reroll_boss(g: &mut Game, from_tag: bool) {
	g.boss_rerolled = true;
	if !from_tag {
		g.events.push(Ev::EaseDollars(-10.0));
	}
	g.events.push(Ev::Wait);
	g.events.push_after(0.3, Ev::RerollBossRun);
}

// -----------------------------------------------------------------
// Round start
// -----------------------------------------------------------------

fn new_round(g: &mut Game) {
	g.discards_left = 0f64.max(g.rr_discards + g.round_bonus_discards);
	g.hands_left = 1f64.max(g.rr_hands + g.round_bonus_next_hands);
	g.hands_played_round = 0.0;
	g.discards_used = 0.0;
	g.reroll_cost_increase = 0.0;
	g.used_packs = [PackSlot::Unrolled; 2];
	for h in &mut g.hands {
		h.played_this_round = 0.0;
	}
	for c in g.playing_cards_mut() {
		c.ab.wheel_flipped = false;
	}
	g.free_rerolls = num::float(g.count_joker(Joker::Chaos));
	cards::calculate_reroll_cost(g, true);
	g.round_bonus_next_hands = 0.0;
	g.round_bonus_discards = 0.0;
	let blind = g
		.selected_blind
		.expect("new_round without a selected blind");
	g.blind_states[BlindSlot::of(blind) as usize] = BlindStatus::Current;
	blind::set_blind(g, Some(blind));
	let boss = blind.is_boss();
	for i in 0..g.jokers.len() {
		jokers::calc_joker(g, i, &Ctx::SettingBlind { boss }, None);
	}
	g.end_round_pending = false;
	g.events.delay(0.4);
	g.events.push(Ev::NewRoundDraw);
}

// -----------------------------------------------------------------
// Play / discard
// -----------------------------------------------------------------

/// `G.FUNCS.play_cards_from_highlighted` with the given hand indices.
pub(crate) fn do_play(g: &mut Game, sel: CardMask) {
	if !g.play.is_empty() {
		return;
	}
	g.blind.triggered = false;
	for c in g.playing_cards_mut() {
		c.ab.forced_selection = false;
	}
	g.play.reserve(sel.count_ones() as usize);
	g.hand_text_cleared = false;
	g.events.push(Ev::PlayStart);
	g.events.push(Ev::EaseHands(-1.0));
	g.events.delay(0.4);
	for i in mask_bits(sel) {
		let c = &mut g.hand[i];
		c.times_played += 1;
		c.ab.played_this_ante = true;
		let id = c.id;
		g.events
			.draw_ev(Pile::Hand, Pile::Play, Some(id), false, false, 0.1);
	}
	if blind::press_play(g) {
		g.events.push(Ev::Wait);
		g.events.delay(0.4);
	}
	g.events.push(Ev::PlayChain);
}

fn hook_discard(g: &mut Game) {
	// pseudorandom_element picks from the remaining cards sorted by sort_id.
	let mut cards: Vec<usize> = (0..g.hand.len()).collect();
	cards.sort_by_key(|&i| g.hand[i].id);
	let mut chosen = Vec::new();
	for _ in 0..cards.len().min(2) {
		let k = Key::from("hook");
		let pick = g.rng.element(&k, cards.len());
		chosen.push(cards.remove(pick));
	}
	if !chosen.is_empty() {
		do_discard(g, chosen, true);
	}
}

/// `G.FUNCS.discard_cards_from_highlighted(e, hook)`.
pub(crate) fn do_discard(g: &mut Game, mut sel: Vec<usize>, hook: bool) {
	for c in g.playing_cards_mut() {
		c.ab.forced_selection = false;
	}
	sel.sort_unstable();
	let count = sel.len().min(500usize.saturating_sub(g.play.len()));
	if count == 0 {
		return;
	}
	let sel = &sel[..count];
	// update_hand_text({immediate = true, delay = 0}, {handname = ''})
	g.events
		.push_ev(Ev::Wait, crate::events::Trig::Before(0.0), true, false);
	g.hand_text_cleared = true;
	// Burnt Joker evaluates the highlighted cards' poker hand (only it
	// reads the result, in the `pre_discard` pass below).
	g.burnt_hand = if g.jokers.iter().any(|c| c.kind == Kind::Joker(Joker::Burnt)) {
		let refs: Vec<&Card> = sel.iter().map(|&i| &g.hand[i]).collect();
		hand_eval::evaluate(&refs, &g.eval_flags()).top
	} else {
		None
	};
	for j in 0..g.jokers.len() {
		jokers::calc_joker(g, j, &Ctx::PreDiscard { hook }, None);
	}
	let par = g.pareidolia();
	let full_faces = sel
		.iter()
		.filter(|&&i| g.hand[i].is_face(false, par))
		.count();
	let last = sel[count - 1];
	let mut destroyed: Vec<usize> = Vec::new();
	for &i in sel {
		// Card:calculate_seal({discard = true})
		if !g.hand[i].debuff && g.hand[i].seal == Some(Seal::Purple) && jokers::consumable_room(g) {
			g.consumeable_buffer += 1.0;
			g.events.push_before(0.0, Ev::Create {
				set: CardSet::Tarot,
				area: Area::Consumables,
				forced: None,
				append: "8ba",
				rarity: None,
				legendary: false,
			});
			g.events.status_extra();
		}
		let mut removed = false;
		for j in 0..g.jokers.len() {
			if let Some(e) = jokers::calc_joker(
				g,
				j,
				&Ctx::Discard {
					other: i,
					last,
					n_full: count,
					full_faces,
				},
				None,
			) {
				if e.remove {
					removed = true;
				}
				jokers::effect_status(g, &e);
			}
		}
		if removed {
			destroyed.push(i);
			let c = &mut g.hand[i];
			let id = c.id;
			if c.enhancement == Enhancement::Glass {
				c.shattered = true;
				g.events.shatter(id);
			} else {
				g.events.start_dissolve(id);
			}
		} else {
			g.hand[i].ab.discarded = true;
			let id = g.hand[i].id;
			g.events
				.draw_ev(Pile::Hand, Pile::Discard, Some(id), false, false, 0.1);
		}
	}
	if !destroyed.is_empty() {
		let faces = destroyed
			.iter()
			.filter(|&&i| g.hand[i].is_face(false, par))
			.count();
		let glass = destroyed.iter().filter(|&&i| g.hand[i].shattered).count();
		for j in 0..g.jokers.len() {
			jokers::calc_joker(g, j, &Ctx::RemovePlayingCards { faces, glass }, None);
		}
	}
	if !hook {
		g.events.push(Ev::EaseDiscard(-1.0));
		g.discards_used += 1.0;
		g.state = St::DrawToHand;
		g.events.push(Ev::StateCompleteFalse);
	}
}
