//! Port of `Card:use_consumeable` / `Card:can_use_consumeable` (card.lua):
//! whether a tarot, planet or spectral card can be used and what using it
//! queues. The queued steps are [`ConsumableEvent`]s.

use super::event::{
	ConsumableEvent, ConsumableMaker, DESTROY_CAPACITY, EditionGift, JokerMaker, SUITS,
	SpectralCreate,
};
use crate::areas::{self, Hand, position_of};
use crate::card::{Card, CardMask, Kind, PlayingCard};
use crate::data::{Enhancement, HandType, Rank, Seal, Spectral, Suit, Tarot};
use crate::event::{Event, EventQueue};
use crate::game::{Area, Areas, Game, HandLevels, State};
use crate::joker::LifecycleCtx;
use crate::rng::{Key, PseudoRandom};
use crate::run::History;
use crate::{event, flow, joker, num, rng};

/// The cards that destroy hand cards: The Hanged Man (the highlighted
/// ones) and the spectrals that destroy random ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HandDestroyer {
	HangedMan,
	Creator(SpectralCreate),
	Immolate,
}

/// The spectrals that change the whole hand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WholeHand {
	/// One suit for every card.
	Sigil,
	/// One rank for every card, and one less hand size.
	Ouija,
}

/// What a tarot does to the highlighted cards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Conversion {
	Enhance(Enhancement),
	Suit(Suit),
	/// Each becomes a copy of the rightmost one.
	Death,
	/// Each rank goes up by one.
	Strength,
}

/// How many hand cards a consumable acts on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Highlight {
	/// `config.max_highlighted`, capped at the hand's highlight limit
	/// (`mod_num`).
	pub(crate) max: usize,
	/// `config.min_highlighted`, at least 1.
	pub(crate) min: usize,
}

/// The consumable's `config.max_highlighted` and `min_highlighted`, if it
/// has them.
fn highlight(k: Kind) -> Option<Highlight> {
	let def = match k {
		Kind::Tarot(t) => t.def(),
		Kind::Spectral(s) => s.def(),
		_ => return None,
	};
	(def.max_highlighted > 0).then(|| Highlight {
		max: usize::from(def.max_highlighted).min(Hand::HIGHLIGHT_LIMIT),
		min: usize::from(def.min_highlighted.max(1)),
	})
}

/// The hand cards a consumable acts on, if any: its highlight range, or
/// exactly one card for Aura, which has none but needs one.
pub(crate) fn targets(k: Kind) -> Option<Highlight> {
	if k == Kind::Spectral(Spectral::Aura) {
		return Some(Highlight { max: 1, min: 1 });
	}
	highlight(k)
}

/// `Card:can_use_consumeable()` for a consumable of kind `kind` located in
/// `area` with the hand cards `highlighted`.
#[must_use]
pub(crate) fn can_use_consumable(
	state: State, areas: &Areas, history: &History, kind: Kind, area: Area, highlighted: CardMask,
) -> bool {
	if !areas.play.is_empty() {
		return false;
	}
	if matches!(
		state,
		State::HandPlayed | State::DrawToHand | State::PlayTarot
	) {
		return false;
	}
	let n_highlighted = highlighted.len();
	let cons_room = areas.consumable_room() || area == Area::Consumables;
	match kind {
		Kind::Planet(_)
		| Kind::Tarot(Tarot::Hermit | Tarot::Temperance)
		| Kind::Spectral(Spectral::BlackHole) => return true,
		Kind::Tarot(Tarot::WheelOfFortune)
		| Kind::Spectral(Spectral::Ectoplasm | Spectral::Hex) => {
			if areas.jokers.iter().any(|c| c.core.edition.is_none()) {
				return true;
			}
		}
		Kind::Spectral(Spectral::Ankh) => {
			if !areas.jokers.is_empty() && areas.limits.jokers.limit > 1.0 {
				return true;
			}
		}
		Kind::Spectral(Spectral::Aura) => {
			let first_editionless = highlighted
				.lowest()
				.is_some_and(|i| areas.hand.cards[i].core.edition.is_none());
			if n_highlighted == 1 && first_editionless {
				return true;
			}
		}
		Kind::Tarot(Tarot::Emperor | Tarot::HighPriestess) => {
			if cons_room {
				return true;
			}
		}
		Kind::Tarot(Tarot::Fool) => {
			if cons_room
				&& history.last_tarot_planet.is_some()
				&& history.last_tarot_planet != Some(Kind::Tarot(Tarot::Fool))
			{
				return true;
			}
		}
		Kind::Tarot(Tarot::Judgement) | Kind::Spectral(Spectral::Soul | Spectral::Wraith) => {
			return areas.joker_room();
		}
		_ => {}
	}
	if matches!(
		state,
		State::SelectingHand | State::TarotPack | State::SpectralPack | State::PlanetPack
	) {
		if let Some(h) = highlight(kind)
			&& h.max >= n_highlighted
			&& n_highlighted >= h.min
		{
			return true;
		}
		if matches!(
			kind,
			Kind::Spectral(
				Spectral::Familiar
					| Spectral::Grim
					| Spectral::Incantation
					| Spectral::Immolate
					| Spectral::Sigil
					| Spectral::Ouija
			)
		) && areas.hand.cards.len() > 1
		{
			return true;
		}
	}
	false
}

/// `set_consumeable_usage(card)`.
fn set_consumable_usage(history: &mut History, events: &mut EventQueue, kind: Kind) {
	match kind {
		Kind::Tarot(_) => history.consumable_usage_tarot += 1.0,
		Kind::Planet(p) => {
			history.consumable_usage_planet += 1.0;
			history.planets_used.insert(p);
		}
		Kind::Spectral(_) => history.consumable_usage_spectral += 1.0,
		_ => {}
	}
	if matches!(kind, Kind::Tarot(_) | Kind::Planet(_)) {
		events.push(Event::Consumable(ConsumableEvent::LastTarotPlanetOuter {
			kind,
		}));
	}
}

/// `Card:use_consumeable(area)` with the hand cards `highlighted`.
///
/// # Panics
///
/// Panics if `card` is used where the action check
/// (`actions::consumable_usable`) does not accept it: more highlighted cards
/// than the hand's highlight limit, none for a consumable that acts on one,
/// an empty hand for Familiar, Grim or Incantation, or no jokers for Ankh.
pub(crate) fn use_consumable(g: &mut Game, card: &Card, highlighted: CardMask) {
	set_consumable_usage(&mut g.history, &mut g.events, card.kind);
	if card.core.debuff {
		return;
	}
	let mut ids = [0; Hand::HIGHLIGHT_LIMIT];
	assert!(
		highlighted.len() <= ids.len(),
		"the action check accepts at most {} hand cards for a consumable, got {}",
		ids.len(),
		highlighted.len()
	);
	for (id, i) in ids.iter_mut().zip(highlighted.indices()) {
		*id = g.areas.hand.cards[i].core.id;
	}
	let highlighted_ids = &ids[..highlighted.len()];
	if highlight(card.kind).is_some() {
		g.events.hand_text_immediate(0.0);
	}
	match card.kind {
		Kind::Planet(p) => {
			g.events.hand_text(0.3);
			event::level_up_hand(&mut g.hands, &mut g.events, p.hand(), 1.0);
			g.events.hand_text(0.0);
		}
		Kind::Tarot(t) => use_tarot(g, t, highlighted_ids),
		Kind::Spectral(s) => use_spectral(g, s, highlighted_ids),
		// `use_card` hands over only consumables.
		Kind::Joker(_) | Kind::Voucher(_) | Kind::Booster(_) => {}
	}
}

fn use_tarot(g: &mut Game, tarot: Tarot, highlighted_ids: &[u32]) {
	use Tarot as T;

	let conversion = match tarot {
		T::HangedMan => {
			return destroy_hand_cards(g, HandDestroyer::HangedMan, highlighted_ids);
		}
		T::Emperor => {
			return emperor_priestess(&g.areas, &mut g.events, ConsumableMaker::Emperor);
		}
		T::HighPriestess => {
			return emperor_priestess(&g.areas, &mut g.events, ConsumableMaker::HighPriestess);
		}
		T::Fool => return delayed_effect(&mut g.events, ConsumableEvent::Fool),
		T::Hermit => return delayed_effect(&mut g.events, ConsumableEvent::Hermit),
		T::Temperance => return delayed_effect(&mut g.events, ConsumableEvent::Temperance),
		T::Judgement => {
			let which = JokerMaker::Judgement;
			return delayed_effect(&mut g.events, ConsumableEvent::JudgementSoul { which });
		}
		T::WheelOfFortune => {
			let key = Key::from("wheel_of_fortune");
			if g.rng.random(&key) < g.run.probabilities / 4.0 {
				let which = EditionGift::Wheel;
				delayed_effect(&mut g.events, ConsumableEvent::EditionGift { which });
			} else {
				g.events.push_after(0.4, Event::Wait);
				g.events.delay(0.6);
			}
			return;
		}
		T::Magician => Conversion::Enhance(Enhancement::Lucky),
		T::Empress => Conversion::Enhance(Enhancement::Mult),
		T::Heirophant => Conversion::Enhance(Enhancement::Bonus),
		T::Lovers => Conversion::Enhance(Enhancement::Wild),
		T::Chariot => Conversion::Enhance(Enhancement::Steel),
		T::Justice => Conversion::Enhance(Enhancement::Glass),
		T::Devil => Conversion::Enhance(Enhancement::Gold),
		T::Tower => Conversion::Enhance(Enhancement::Stone),
		T::Star => Conversion::Suit(Suit::Diamonds),
		T::Moon => Conversion::Suit(Suit::Clubs),
		T::Sun => Conversion::Suit(Suit::Hearts),
		T::World => Conversion::Suit(Suit::Spades),
		T::Death => Conversion::Death,
		T::Strength => Conversion::Strength,
	};
	convert_highlighted(&mut g.events, conversion, highlighted_ids);
}

fn use_spectral(g: &mut Game, spectral: Spectral, highlighted_ids: &[u32]) {
	use Spectral as S;

	let first = || {
		*highlighted_ids
			.first()
			.expect("the action check accepts a one-card spectral only with its card")
	};
	match spectral {
		S::BlackHole => black_hole(&mut g.hands, &mut g.events),
		S::Talisman => set_seal(&mut g.events, first(), Seal::Gold),
		S::DejaVu => set_seal(&mut g.events, first(), Seal::Red),
		S::Trance => set_seal(&mut g.events, first(), Seal::Blue),
		S::Medium => set_seal(&mut g.events, first(), Seal::Purple),
		S::Aura => {
			g.events.push_after(
				0.4,
				Event::Consumable(ConsumableEvent::Aura { target: first() }),
			);
		}
		S::Cryptid => {
			g.events.push(Event::Consumable(ConsumableEvent::Cryptid {
				source: first(),
			}));
		}
		S::Sigil => {
			convert_whole_hand(
				&g.areas.hand.cards,
				&mut g.rng,
				&mut g.events,
				WholeHand::Sigil,
			);
		}
		S::Ouija => {
			convert_whole_hand(
				&g.areas.hand.cards,
				&mut g.rng,
				&mut g.events,
				WholeHand::Ouija,
			);
		}
		S::Familiar => {
			let which = HandDestroyer::Creator(SpectralCreate::Familiar);
			destroy_hand_cards(g, which, highlighted_ids);
		}
		S::Grim => {
			let which = HandDestroyer::Creator(SpectralCreate::Grim);
			destroy_hand_cards(g, which, highlighted_ids);
		}
		S::Incantation => {
			let which = HandDestroyer::Creator(SpectralCreate::Incantation);
			destroy_hand_cards(g, which, highlighted_ids);
		}
		S::Immolate => destroy_hand_cards(g, HandDestroyer::Immolate, highlighted_ids),
		S::Ankh => ankh(&mut g.areas.jokers, &mut g.rng, &mut g.events),
		S::Soul => {
			let which = JokerMaker::Soul;
			delayed_effect(&mut g.events, ConsumableEvent::JudgementSoul { which });
		}
		S::Wraith => delayed_effect(&mut g.events, ConsumableEvent::Wraith),
		S::Ectoplasm => {
			let which = EditionGift::Ectoplasm;
			delayed_effect(&mut g.events, ConsumableEvent::EditionGift { which });
		}
		S::Hex => {
			let which = EditionGift::Hex;
			delayed_effect(&mut g.events, ConsumableEvent::EditionGift { which });
		}
	}
}

/// The spectrals that give the highlighted card a seal.
fn set_seal(events: &mut EventQueue, target: u32, seal: Seal) {
	events.push(Event::Wait);
	events.push_after(
		0.1,
		Event::Consumable(ConsumableEvent::SetSeal { target, seal }),
	);
	events.delay(0.5);
	events.push_after(0.2, Event::Consumable(ConsumableEvent::UnhighlightAll));
}

/// The tarots that change the highlighted cards.
fn convert_highlighted(events: &mut EventQueue, conversion: Conversion, highlighted_ids: &[u32]) {
	events.delay(0.4);
	for _ in highlighted_ids {
		events.delay(0.15);
	}
	events.delay(0.2);
	for &target in highlighted_ids {
		let step = match conversion {
			Conversion::Death => ConsumableEvent::Death {
				target,
				source: highlighted_ids[highlighted_ids.len() - 1],
			},
			Conversion::Strength => ConsumableEvent::Strength { target },
			Conversion::Suit(suit) => ConsumableEvent::SuitConversion { target, suit },
			Conversion::Enhance(enhancement) => ConsumableEvent::Enhance {
				target,
				enhancement,
			},
		};
		events.push_after(0.1, Event::Consumable(step));
	}
	for _ in highlighted_ids {
		events.delay(0.15);
	}
	events.push_after(0.2, Event::Consumable(ConsumableEvent::UnhighlightAll));
	events.delay(0.5);
}

fn black_hole(hands: &mut HandLevels, events: &mut EventQueue) {
	events.hand_text(0.3);
	events.delay(0.2);
	events.hand_text(0.0);
	events.delay(0.9);
	events.hand_text(0.0);
	events.delay(0.9);
	events.hand_text(0.0);
	events.delay(1.3);
	for h in HandType::PAIRS_ORDER {
		// `level_up_hand(card, hand, true)`: only its level-up event.
		hands.level_up(h, 1.0);
		events.push(Event::Wait);
	}
	events.hand_text(0.0);
}

/// Sigil or Ouija on the whole hand.
fn convert_whole_hand(
	hand: &[PlayingCard], rng: &mut PseudoRandom, events: &mut EventQueue, which: WholeHand,
) {
	events.delay(0.4);
	for _ in hand {
		events.delay(0.15);
	}
	events.delay(0.2);
	match which {
		WholeHand::Sigil => {
			let suit = SUITS[rng.element_str("sigil", SUITS.len())];
			for c in hand {
				events.push(Event::Consumable(ConsumableEvent::SigilSet {
					target: c.core.id,
					suit,
				}));
			}
		}
		WholeHand::Ouija => {
			let rank = Rank::ALL[rng.element_str("ouija", Rank::ALL.len())];
			for c in hand {
				events.push(Event::Consumable(ConsumableEvent::OuijaSet {
					target: c.core.id,
					rank,
				}));
			}
			flow::change_hand_size(events, -1.0);
		}
	}
	for _ in hand {
		events.delay(0.15);
	}
	events.delay(0.5);
}

/// Copies `ids` into the front of `out`; returns how many there are.
fn fill(out: &mut [u32], ids: impl IntoIterator<Item = u32>) -> usize {
	let mut n = 0;
	for id in ids {
		assert!(
			n < out.len(),
			"a consumable destroys at most {} hand cards",
			out.len()
		);
		out[n] = id;
		n += 1;
	}
	n
}

/// The Hanged Man and the spectrals that destroy random hand cards.
fn destroy_hand_cards(g: &mut Game, which: HandDestroyer, highlighted_ids: &[u32]) {
	// The ids in the order the destroy loops (which run back to front) reach
	// them.
	let mut ids = [0; DESTROY_CAPACITY];
	let n = match which {
		HandDestroyer::HangedMan => fill(&mut ids, highlighted_ids.iter().rev().copied()),
		HandDestroyer::Creator(_) => {
			let pick =
				areas::pick_card(&mut g.rng, "random_destroy", &g.areas.hand.cards, |_, _| {
					true
				})
				.expect(
					"the action check accepts Familiar, Grim and Incantation only with hand cards",
				);
			fill(&mut ids, [g.areas.hand.cards[pick].core.id])
		}
		HandDestroyer::Immolate => {
			let mut shuffled: Vec<u32> = g.areas.hand.cards.iter().map(|c| c.core.id).collect();
			shuffled.sort_unstable();
			let seed = g.rng.seed_str("immolate");
			rng::shuffle(&mut shuffled, seed);
			fill(&mut ids, shuffled.iter().take(5).rev().copied())
		}
	};
	g.events.delay(0.4);
	let delay = if which == HandDestroyer::HangedMan {
		0.2
	} else {
		0.1
	};
	g.events.push_after(
		delay,
		Event::Consumable(ConsumableEvent::DestroyCards {
			n: num::byte(n),
			ids,
		}),
	);
	match which {
		HandDestroyer::Creator(create) => {
			g.events.push_after(
				0.7,
				Event::Consumable(ConsumableEvent::CreateSpectralCards { which: create }),
			);
		}
		HandDestroyer::Immolate => {
			g.events.delay(0.5);
			g.events.push(Event::EaseDollars(20.0));
		}
		HandDestroyer::HangedMan => {}
	}
	g.events.delay(0.3);
	let par = g.areas.pareidolia();
	let hand = &g.areas.hand.cards;
	let faces = ids[..n]
		.iter()
		.filter(|&&id| position_of(hand, id).is_some_and(|i| hand[i].is_face(par)))
		.count();
	joker::calc_jokers(g, LifecycleCtx::RemovePlayingCards { faces, glass: 0 });
}

/// The Emperor or The High Priestess: a card for each free consumable slot,
/// at most two.
fn emperor_priestess(areas: &Areas, events: &mut EventQueue, which: ConsumableMaker) {
	let n = 2f64.min(areas.limits.consumables.limit - num::float(areas.consumables.len()));
	let mut i = 1.0;
	while i <= n {
		events.push_after(
			0.4,
			Event::Consumable(ConsumableEvent::EmperorPriestess { which }),
		);
		i += 1.0;
	}
	events.delay(0.6);
}

/// Ankh: picks the joker to copy, and marks every other joker that is not
/// eternal to go.
fn ankh(jokers: &mut [Card], rng: &mut PseudoRandom, events: &mut EventQueue) {
	let pick = areas::pick_card(rng, "ankh_choice", jokers, |_, _| true)
		.expect("the action check accepts Ankh only with jokers");
	let chosen = jokers[pick].core.id;
	for c in jokers.iter_mut() {
		if !c.ab.eternal() && c.core.id != chosen {
			c.ankh_deletable = true;
		}
	}
	events.push_before(0.75, Event::Consumable(ConsumableEvent::AnkhDestroy));
	events.push_before(0.4, Event::Consumable(ConsumableEvent::AnkhCopy { chosen }));
}

/// A consumable whose effect is one delayed event: `step` after 0.4 s, then
/// `delay(0.6)`.
fn delayed_effect(events: &mut EventQueue, step: ConsumableEvent) {
	events.push_after(0.4, Event::Consumable(step));
	events.delay(0.6);
}
