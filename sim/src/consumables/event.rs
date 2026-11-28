//! The steps of a consumable's effect: the [`ConsumableEvent`]s that
//! `Card:use_consumeable` (card.lua) queues, and what each does when it runs.

use crate::areas::{self, position_of};
use crate::card::{Forced, Kind, PlayingCard, Pricing};
use crate::data::{Edition, Enhancement, Rank, Seal, Suit};
use crate::event::{EditionTiming, Event, EventQueue};
use crate::game::{Area, Areas, Blind, CardSet, Game};
use crate::joker::LifecycleCtx;
use crate::pools::{NewCard, OwnedArea, RarityRoll};
use crate::rng::{Key, PseudoRandom};
use crate::round::CardDebuff;
use crate::run::Run;
use crate::{card, flow, joker, pools};

/// Room for the hand cards one consumable destroys (Immolate's five at most).
pub(super) const DESTROY_CAPACITY: usize = 8;

/// The deferred steps of a consumable's effect.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum ConsumableEvent {
	/// `set_consumeable_usage`'s event, which queues the next one.
	LastTarotPlanetOuter {
		kind: Kind,
	},
	LastTarotPlanetInner {
		kind: Kind,
	},
	Death {
		target: u32,
		source: u32,
		/// The target's place in the game's highlight list.
		place: u8,
	},
	Strength {
		target: u32,
	},
	SuitConversion {
		target: u32,
		suit: Suit,
	},
	Enhance {
		target: u32,
		enhancement: Enhancement,
	},
	UnhighlightAll,
	SetSeal {
		target: u32,
		seal: Seal,
	},
	Aura {
		target: u32,
	},
	Cryptid {
		source: u32,
	},
	SigilSet {
		target: u32,
		suit: Suit,
	},
	OuijaSet {
		target: u32,
		rank: Rank,
	},
	DestroyCards {
		n: u8,
		ids: [u32; DESTROY_CAPACITY],
	},
	CreateSpectralCards {
		which: SpectralCreate,
	},
	Fool,
	Hermit,
	Temperance,
	EmperorPriestess {
		which: ConsumableMaker,
	},
	JudgementSoul {
		which: JokerMaker,
	},
	AnkhDestroy,
	AnkhCopy {
		chosen: u32,
	},
	Wraith,
	EditionGift {
		which: EditionGift,
	},
}

/// The consumables that give a random editionless joker an edition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EditionGift {
	Wheel,
	Ectoplasm,
	Hex,
}

/// The tarots that make a consumable: The Emperor (a tarot) and The High
/// Priestess (a planet).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ConsumableMaker {
	Emperor,
	HighPriestess,
}

/// The cards that make a joker: Judgement (any) and The Soul (a legendary).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum JokerMaker {
	Judgement,
	Soul,
}

/// The spectrals that destroy a hand card and make enhanced ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SpectralCreate {
	/// Face cards.
	Familiar,
	/// Aces.
	Grim,
	/// Numbered cards.
	Incantation,
}

impl SpectralCreate {
	/// The cards it makes.
	fn count(self) -> u8 {
		match self {
			SpectralCreate::Familiar => 3,
			SpectralCreate::Grim => 2,
			SpectralCreate::Incantation => 4,
		}
	}
}

/// The suits in the order the spectral cards roll one.
pub(super) const SUITS: [Suit; 4] = [Suit::Spades, Suit::Hearts, Suit::Diamonds, Suit::Clubs];

pub(crate) fn run_event(g: &mut Game, e: ConsumableEvent) {
	use ConsumableEvent as E;
	match e {
		E::LastTarotPlanetOuter { kind } => {
			g.events
				.push(Event::Consumable(E::LastTarotPlanetInner { kind }));
		}
		E::LastTarotPlanetInner { kind } => g.history.last_tarot_planet = Some(kind),
		E::Death {
			target,
			source,
			place,
		} => death(&mut g.areas, g.shop.discount_percent, target, source, place),
		E::Strength { target } => rebase(&mut g.areas, &g.blind, target, |c| {
			(c.suit, c.rank.raised())
		}),
		E::SuitConversion { target, suit } | E::SigilSet { target, suit } => {
			rebase(&mut g.areas, &g.blind, target, |c| (suit, c.rank));
		}
		E::OuijaSet { target, rank } => rebase(&mut g.areas, &g.blind, target, |c| (c.suit, rank)),
		E::Enhance {
			target,
			enhancement,
		} => {
			if let Some(i) = position_of(&g.areas.hand.cards, target) {
				let debuff = CardDebuff::new(&g.blind, &g.areas.jokers);
				card::set_enhancement(
					&mut g.areas.hand.cards[i],
					g.run.hands_played,
					debuff,
					enhancement,
				);
			}
		}
		E::UnhighlightAll => g.areas.hand.highlight_forced(),
		E::SetSeal { target, seal } => {
			if let Some(i) = position_of(&g.areas.hand.cards, target) {
				let pricing = Pricing::new(g.shop.discount_percent, &g.areas.jokers);
				let card = &mut g.areas.hand.cards[i];
				card.seal = Some(seal);
				pricing.set_cost(card, Some(Area::Hand));
			}
		}
		E::Aura { target } => aura(
			&mut g.areas,
			g.shop.discount_percent,
			&mut g.rng,
			&mut g.events,
			target,
		),
		E::Cryptid { source } => cryptid(g, source),
		E::DestroyCards { n, ids } => destroy_cards(
			&mut g.areas.hand.cards,
			&mut g.events,
			&ids[..usize::from(n)],
		),
		E::CreateSpectralCards { which } => create_spectral_cards(g, which),
		E::Fool => fool(g),
		E::Hermit => {
			let d = 0f64.max(g.run.dollars.min(20.0));
			flow::ease_dollars(&mut g.run, d);
		}
		E::Temperance => {
			let m = g.areas.joker_sell_total().min(50.0);
			flow::ease_dollars(&mut g.run, m);
		}
		E::EmperorPriestess { which } => make_consumable(g, which),
		E::JudgementSoul { which } => make_joker(g, which),
		E::AnkhDestroy => {
			for c in &g.areas.jokers {
				if c.ankh_deletable {
					g.events.start_dissolve(c.core.id);
				}
			}
		}
		E::AnkhCopy { chosen } => ankh_copy(g, chosen),
		E::Wraith => wraith(g),
		E::EditionGift { which } => gift_edition(
			&mut g.areas,
			&mut g.run,
			g.shop.discount_percent,
			&mut g.rng,
			&mut g.events,
			which,
		),
	}
}

/// Death: the hand card `target`, at `place` of the game's highlight list,
/// becomes a copy of `source`.
fn death(areas: &mut Areas, discount_percent: f64, target: u32, source: u32, place: u8) {
	if target != source
		&& let (Some(ti), Some(si)) = (
			position_of(&areas.hand.cards, target),
			position_of(&areas.hand.cards, source),
		) {
		copy_into(areas, discount_percent, si, ti, place);
		// Both highlighted cards are flipped around the copy, as in `rebase`.
		areas.hand.cards[ti].wheel_flipped = false;
		areas.hand.cards[si].wheel_flipped = false;
	}
}

/// The Fool: a copy of the last tarot or planet used, if there is room.
fn fool(g: &mut Game) {
	if g.areas.consumable_room() {
		let last = g.history.last_tarot_planet.expect(
			"can_use_consumable allows The Fool only after a tarot or planet, and nothing clears it",
		);
		pools::create_owned(g, NewCard::forced(last, OwnedArea::Consumables, "fool"));
	}
}

/// The Emperor's tarot or The High Priestess's planet, if there is room.
fn make_consumable(g: &mut Game, which: ConsumableMaker) {
	if g.areas.consumable_room() {
		let (set, append) = match which {
			ConsumableMaker::Emperor => (CardSet::Tarot, "emp"),
			ConsumableMaker::HighPriestess => (CardSet::Planet, "pri"),
		};
		pools::create_owned(g, NewCard::new(set, OwnedArea::Consumables, append));
	}
}

/// Judgement's joker or The Soul's legendary joker.
fn make_joker(g: &mut Game, which: JokerMaker) {
	let (rarity, append) = match which {
		JokerMaker::Judgement => (RarityRoll::Seeded, "jud"),
		JokerMaker::Soul => (
			RarityRoll::Legendary {
				rolled: !g.setup.steamodded,
			},
			"sou",
		),
	};
	pools::create_owned(g, NewCard::joker(rarity, OwnedArea::Jokers, append));
}

/// Wraith: a rare joker, and the money goes.
fn wraith(g: &mut Game) {
	pools::create_owned(
		g,
		NewCard::joker(RarityRoll::Fixed(0.99), OwnedArea::Jokers, "wra"),
	);
	if g.run.dollars != 0.0 {
		let d = g.run.dollars;
		flow::ease_dollars(&mut g.run, -d);
	}
}

/// Aura: a rolled edition for the hand card `target` (the roll is drawn
/// even if the card is gone).
fn aura(
	areas: &mut Areas, discount_percent: f64, rng: &mut PseudoRandom, events: &mut EventQueue,
	target: u32,
) {
	let key = Key::from("aura");
	let e = pools::poll_guaranteed_edition(rng, &key);
	if let Some(i) = position_of(&areas.hand.cards, target) {
		let pricing = Pricing::new(discount_percent, &areas.jokers);
		let card = &mut areas.hand.cards[i];
		card::set_edition(card, e, Some(Area::Hand), pricing, &mut areas.limits);
	}
	events.edition_fx(e, EditionTiming::Immediate);
}

/// Ankh: a copy of the joker `chosen`, if it is still there (a Negative
/// one is copied without its edition).
fn ankh_copy(g: &mut Game, chosen: u32) {
	if let Some(i) = position_of(&g.areas.jokers, chosen) {
		let strip = g.areas.jokers[i].core.edition == Some(Edition::Negative);
		let original = g.areas.jokers[i].clone();
		let mut c = flow::copy_card(
			&mut g.areas,
			&mut g.history,
			g.shop.discount_percent,
			&g.hands,
			&mut g.rng,
			original,
			strip,
		);
		g.events.materialize(1.0);
		card::add_to_deck(g, &mut c);
		g.areas.jokers.push(c);
	}
}

/// Cryptid: two copies of the hand card `source`, if it is still there.
fn cryptid(g: &mut Game, source: u32) {
	if let Some(si) = position_of(&g.areas.hand.cards, source) {
		for _ in 0..2 {
			let original = g.areas.hand.cards[si].clone();
			let mut c = flow::copy_playing_card(&mut g.areas, g.shop.discount_percent, original);
			card::add_to_deck(g, &mut c);
			g.areas.hand.cards.push(c);
			g.events.materialize(1.0);
		}
		joker::calc_jokers(g, LifecycleCtx::PlayingCardAdded { n: 2 });
	}
}

/// Starts the destruction of the hand cards `ids` that are still there.
fn destroy_cards(hand: &mut [PlayingCard], events: &mut EventQueue, ids: &[u32]) {
	for &id in ids {
		if let Some(i) = position_of(hand, id) {
			card::shatter_or_dissolve(&mut hand[i], events);
		}
	}
}

/// `Card:set_base` on the hand card `target`, if it is still there, with
/// the suit and rank `change` gives for it.
fn rebase(
	areas: &mut Areas, blind: &Blind, target: u32,
	change: impl FnOnce(&PlayingCard) -> (Suit, Rank),
) {
	if let Some(i) = position_of(&areas.hand.cards, target) {
		let c = &mut areas.hand.cards[i];
		let (suit, rank) = change(c);
		c.set_base(suit, rank);
		// The game flips the card face up and back around the change, and a
		// card turning face up loses `ability.wheel_flipped`.
		c.wheel_flipped = false;
		CardDebuff::new(blind, &areas.jokers).apply(c);
	}
}

/// Familiar, Grim and Incantation: enhanced cards for the hand (face
/// cards, Aces or numbered cards).
fn create_spectral_cards(g: &mut Game, which: SpectralCreate) {
	// The enhanced pool without Stone, in pool order.
	const POOL: [Enhancement; Enhancement::POOL.len() - 1] = {
		let mut pool = [Enhancement::Base; Enhancement::POOL.len() - 1];
		let (mut i, mut n) = (0, 0);
		while i < Enhancement::POOL.len() {
			if !matches!(Enhancement::POOL[i], Enhancement::Stone) {
				pool[n] = Enhancement::POOL[i];
				n += 1;
			}
			i += 1;
		}
		pool
	};
	let n = which.count();
	for _ in 0..n {
		let (rank, suit) = match which {
			SpectralCreate::Familiar => {
				let rank = Rank::FACES[g.rng.element_str("familiar_create", Rank::FACES.len())];
				(
					rank,
					SUITS[g.rng.element_str("familiar_create", SUITS.len())],
				)
			}
			SpectralCreate::Grim => (
				Rank::Ace,
				SUITS[g.rng.element_str("grim_create", SUITS.len())],
			),
			SpectralCreate::Incantation => {
				let rank = Rank::NUMBERED[g
					.rng
					.element_str("incantation_create", Rank::NUMBERED.len())];
				(
					rank,
					SUITS[g.rng.element_str("incantation_create", SUITS.len())],
				)
			}
		};
		let e = POOL[g.rng.element_str("spe_card", POOL.len())];
		let mut c = card::make_playing_card(
			&mut g.areas,
			g.run.hands_played,
			g.shop.discount_percent,
			suit,
			rank,
			e,
		);
		card::add_to_deck(g, &mut c);
		g.events.materialize(1.0);
		g.areas.hand.cards.push(c);
	}
	joker::calc_jokers(g, LifecycleCtx::PlayingCardAdded { n: usize::from(n) });
}

/// The Wheel of Fortune, Ectoplasm and Hex: an edition for a random
/// editionless joker, with Hex's and Ectoplasm's price.
fn gift_edition(
	areas: &mut Areas, run: &mut Run, discount_percent: f64, rng: &mut PseudoRandom,
	events: &mut EventQueue, which: EditionGift,
) {
	let key = match which {
		EditionGift::Wheel => "wheel_of_fortune",
		EditionGift::Ectoplasm => "ectoplasm",
		EditionGift::Hex => "hex",
	};
	let pick = areas::pick_card(rng, key, &areas.jokers, |_, c| c.core.edition.is_none())
		.expect("can_use_consumable allows these only with an editionless joker, and nothing gives it an edition before this event");
	let ed = match which {
		EditionGift::Ectoplasm => Some(Edition::Negative),
		EditionGift::Hex => Some(Edition::Polychrome),
		EditionGift::Wheel => {
			let key = Key::from("wheel_of_fortune");
			pools::poll_guaranteed_edition(rng, &key)
		}
	};
	let pricing = Pricing::new(discount_percent, &areas.jokers);
	let joker = &mut areas.jokers[pick];
	card::set_edition(joker, ed, Some(Area::Jokers), pricing, &mut areas.limits);
	events.edition_fx(ed, EditionTiming::Immediate);
	if which == EditionGift::Hex {
		for (i, c) in areas.jokers.iter().enumerate() {
			if i != pick && !c.ab.eternal() {
				events.start_dissolve(c.core.id);
			}
		}
	}
	if which == EditionGift::Ectoplasm {
		let m = run.ecto_minus;
		flow::change_hand_size(events, -m);
		run.ecto_minus += 1.0;
	}
}

/// `copy_card(other, new_card)` from the `si`-th hand card into the `ti`-th,
/// which is at `place` of the game's highlight list (Death).
fn copy_into(areas: &mut Areas, discount_percent: f64, si: usize, ti: usize, place: u8) {
	let mut c = areas.hand.cards[si].clone();
	let t = &areas.hand.cards[ti];
	c.core.id = t.core.id;
	c.orig_suit = t.orig_suit;
	c.core.added_to_deck = t.core.added_to_deck;
	c.lucky_trigger = false;
	// Not part of what copy_card copies: the card keeps its facing and
	// area flags; set_base resets times_played.
	c.core.face_down = t.core.face_down;
	c.times_played = 0;
	// set_ability keeps the target's forced_selection; the copied
	// ability table only overrides it when the source has one. Both cards
	// are highlighted.
	c.forced = if c.forced.is_flagged() || t.forced.is_flagged() {
		Forced::Locked { place }
	} else {
		Forced::No
	};
	c.shattered = t.shattered;
	c.destroyed = t.destroyed;
	Pricing::new(discount_percent, &areas.jokers).set_cost(&mut c, Some(Area::Hand));
	areas.hand.cards[ti] = c;
}
