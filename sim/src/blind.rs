//! Port of `blind.lua`.

use crate::card::Card;
use crate::data::{BlindKind, HandType, Suit, blind_amount};
use crate::ev::{Ev, Pile};
use crate::game::{Blind, BlindSlot, Game};
use crate::rng::Key;
use crate::{cards, events, flow};

/// Whether `blind` debuffs the playing card `c` (`Blind:debuff_card`), given
/// whether Smeared Joker and Pareidolia are in effect.
pub(crate) fn debuffs(blind: &Blind, smeared: bool, pareidolia: bool, c: &Card) -> bool {
	if blind.disabled {
		return false;
	}
	let suit = match blind.kind {
		Some(BlindKind::Club) => Some(Suit::Clubs),
		Some(BlindKind::Goad) => Some(Suit::Spades),
		Some(BlindKind::Head) => Some(Suit::Hearts),
		Some(BlindKind::Window) => Some(Suit::Diamonds),
		_ => None,
	};
	if let Some(s) = suit
		&& c.is_suit(s, true, false, smeared)
	{
		return true;
	}
	match blind.kind {
		Some(BlindKind::Plant) => c.is_face(true, pareidolia),
		Some(BlindKind::Pillar) => c.ab.played_this_ante,
		Some(BlindKind::FinalLeaf) => true,
		_ => false,
	}
}

/// `Blind:set_blind(blind, reset=false)`.
pub(crate) fn set_blind(g: &mut Game, blind: Option<BlindKind>) {
	let mut b = Blind::empty();
	b.kind = blind;
	if let Some(k) = blind {
		let d = k.def();
		b.dollars = f64::from(d.dollars);
		b.mult = d.mult;
		b.boss = d.boss;
		if g.modifiers.no_blind_reward[BlindSlot::of(k) as usize] {
			b.dollars = 0.0;
		}
		b.chips =
			blind_amount(g.ante, g.modifiers.scaling) * b.mult * g.starting_params.ante_scaling;
	}
	b.prepped = true;
	g.last_blind_boss = b.boss;
	g.blind = b;
	if blind.is_some() {
		g.events.push_ev(
			Ev::BlindSetHud,
			crate::events::Trig::After(0.05),
			true,
			false,
		);
	}
	match blind {
		Some(BlindKind::Fish) => g.blind.prepped = false,
		Some(BlindKind::Water) => {
			g.blind.discards_sub = g.discards_left;
			let d = g.blind.discards_sub;
			g.events.push(Ev::EaseDiscard(-d));
		}
		Some(BlindKind::Needle) => {
			g.blind.hands_sub = g.rr_hands - 1.0;
			let h = g.blind.hands_sub;
			g.events.push(Ev::EaseHands(-h));
		}
		Some(BlindKind::Manacle) => cards::change_hand_size(g, -1.0),
		Some(BlindKind::FinalAcorn) if !g.jokers.is_empty() => {
			for c in &mut g.jokers {
				c.face_down = !c.face_down;
			}
			if g.jokers.len() > 1 {
				g.events.push_after(0.2, Ev::AcornOuter);
			}
		}
		_ => {}
	}
	flow::refresh_playing_debuffs(g);
	for i in 0..g.jokers.len() {
		debuff_joker(g, i);
	}
}

/// `Blind:debuff_card(card)` for a card outside the joker area.
pub(crate) fn debuff_card(g: &Game, c: &mut Card) {
	let debuffed = debuffs(&g.blind, g.smeared(), g.pareidolia(), c);
	cards::set_debuff_playing(c, debuffed);
}

/// [`Self::debuff_card`] on the `i`-th hand card.
pub(crate) fn debuff_hand_card(g: &mut Game, i: usize) {
	let debuffed = debuffs(&g.blind, g.smeared(), g.pareidolia(), &g.hand[i]);
	cards::set_debuff_playing(&mut g.hand[i], debuffed);
}

/// `Blind:debuff_card(joker)`.
pub(crate) fn debuff_joker(g: &mut Game, i: usize) {
	if g.blind.kind == Some(BlindKind::FinalHeart) && !g.blind.disabled {
		return;
	}
	cards::set_debuff_joker(g, i, false);
}

/// `Blind:disable()`.
pub(crate) fn disable_blind(g: &mut Game) {
	g.blind.disabled = true;
	for c in &mut g.jokers {
		c.face_down = false;
	}
	match g.blind.kind {
		Some(BlindKind::Water) => {
			let d = g.blind.discards_sub;
			g.events.push(Ev::EaseDiscard(d));
		}
		Some(BlindKind::Wheel | BlindKind::House | BlindKind::Mark | BlindKind::Fish) => {
			for c in &mut g.hand {
				c.face_down = false;
			}
			for c in g.playing_cards_mut() {
				c.ab.wheel_flipped = false;
			}
		}
		Some(BlindKind::Needle) => {
			let h = g.blind.hands_sub;
			g.events.push(Ev::EaseHands(h));
		}
		Some(BlindKind::Wall) => g.blind.chips /= 2.0,
		Some(BlindKind::FinalBell) => {
			for c in g.playing_cards_mut() {
				c.ab.forced_selection = false;
			}
		}
		Some(BlindKind::Manacle) => {
			cards::change_hand_size(g, 1.0);
			flow::draw_from_deck_to_hand(g, Some(1));
		}
		Some(BlindKind::FinalVessel) => g.blind.chips /= 3.0,
		_ => {}
	}
	g.events.push(Ev::BlindDisableCheck);
	flow::refresh_playing_debuffs(g);
	for i in 0..g.jokers.len() {
		debuff_joker(g, i);
	}
}

/// `Blind:press_play()`; returns its (truthy) result.
pub(crate) fn press_play(g: &mut Game) -> bool {
	if g.blind.disabled {
		return false;
	}
	match g.blind.kind {
		Some(BlindKind::Hook) => {
			g.events.push(Ev::HookDiscard);
			g.blind.triggered = true;
			g.events.delay(0.7);
			return true;
		}
		Some(BlindKind::FinalHeart) => {
			if !g.jokers.is_empty() {
				g.blind.triggered = true;
				g.blind.prepped = true;
			}
		}
		Some(BlindKind::Fish) => g.blind.prepped = true,
		Some(BlindKind::Tooth) => {
			g.events.push_after(0.2, Ev::ToothOuter);
			g.blind.triggered = true;
			return true;
		}
		_ => {}
	}
	false
}

/// `Blind:modify_hand` → `(mult, chips)`.
pub(crate) fn modify_hand(g: &mut Game, mult: f64, chips: f64) -> (f64, f64) {
	if g.blind.disabled {
		return (mult, chips);
	}
	if g.blind.kind == Some(BlindKind::Flint) {
		g.blind.triggered = true;
		return (
			(mult * 0.5 + 0.5).floor().max(1.0),
			(chips * 0.5 + 0.5).floor().max(0.0),
		);
	}
	(mult, chips)
}

/// Whether The Psychic, The Eye or The Mouth makes a play of `n_cards`
/// cards of hand `name` score nothing (`debuff_hand` without its
/// side effects).
pub(crate) fn debuffs_play(g: &Game, n_cards: usize, name: HandType) -> bool {
	!g.blind.disabled
		&& match g.blind.kind {
			Some(BlindKind::Psychic) => n_cards < 5,
			Some(BlindKind::Eye) => g.blind.eye_hands >> (name as u16) & 1 == 1,
			Some(BlindKind::Mouth) => g.blind.only_hand.is_some_and(|only| only != name),
			_ => false,
		}
}

/// `Blind:debuff_hand(cards, hand, handname, check)`.
pub(crate) fn debuff_hand(g: &mut Game, n_cards: usize, name: HandType, check: bool) -> bool {
	if g.blind.disabled {
		return false;
	}
	g.blind.triggered = false;
	if debuffs_play(g, n_cards, name) {
		g.blind.triggered = true;
		return true;
	}
	match g.blind.kind {
		Some(BlindKind::Eye) if !check => g.blind.eye_hands |= 1 << (name as u16),
		Some(BlindKind::Mouth) if !check => g.blind.only_hand = Some(name),
		_ => {}
	}
	if g.blind.kind == Some(BlindKind::Arm) {
		g.blind.triggered = false;
		if g.hands[name as usize].level > 1.0 {
			g.blind.triggered = true;
			if !check {
				events::level_up_hand_ev(g, name, -1.0, false);
			}
		}
	}
	if g.blind.kind == Some(BlindKind::Ox) {
		g.blind.triggered = false;
		if name == g.most_played_hand {
			g.blind.triggered = true;
			if !check {
				let d = g.dollars;
				cards::ease_dollars(g, -d);
			}
		}
	}
	false
}

/// `Blind:drawn_to_hand()`.
pub(crate) fn drawn_to_hand(g: &mut Game) {
	if !g.blind.disabled {
		if g.blind.kind == Some(BlindKind::FinalBell) && !g.hand.is_empty() {
			let any = g.hand.iter().any(|c| c.ab.forced_selection);
			if !any {
				let mut order: Vec<usize> = (0..g.hand.len()).collect();
				order.sort_by_key(|&i| g.hand[i].id);
				let k = Key::from("cerulean_bell");
				let pick = order[g.rng.element(&k, order.len())];
				g.hand[pick].ab.forced_selection = true;
			}
		}
		if g.blind.kind == Some(BlindKind::FinalHeart) && g.blind.prepped && !g.jokers.is_empty() {
			let n = g.jokers.len();
			let mut cands: Vec<usize> = Vec::new();
			for i in 0..n {
				if !g.jokers[i].debuff || n < 2 {
					cands.push(i);
				}
				cards::set_debuff_joker(g, i, false);
			}
			if !cands.is_empty() {
				cands.sort_by_key(|&i| g.jokers[i].id);
				let k = Key::from("crimson_heart");
				let pick = cands[g.rng.element(&k, cands.len())];
				cards::set_debuff_joker(g, pick, true);
			}
		}
	}
	g.blind.prepped = false;
}

/// `Blind:stay_flipped(area, card)`.
pub(crate) fn stay_flipped(g: &mut Game, to: Pile, c: &Card) -> bool {
	if g.blind.disabled || to != Pile::Hand {
		return false;
	}
	match g.blind.kind {
		Some(BlindKind::Wheel) => {
			let k = Key::from("wheel");
			g.rng.random(&k) < g.probabilities / 7.0
		}
		Some(BlindKind::House) => g.hands_played_round == 0.0 && g.discards_used == 0.0,
		Some(BlindKind::Mark) => c.is_face(true, g.pareidolia()),
		Some(BlindKind::Fish) => g.blind.prepped,
		_ => false,
	}
}

/// `Blind:defeat()` (the parts with game effects).
pub(crate) fn defeat(g: &mut Game) {
	for c in &mut g.jokers {
		c.face_down = false;
	}
	if g.blind.kind == Some(BlindKind::Manacle) && !g.blind.disabled {
		cards::change_hand_size(g, 1.0);
	}
	// `set_blind(nil)` runs at the end of the defeat animation.
	g.events.push(Ev::BlindReset);
}
