//! Joker contexts, [`Effect`], [`JokerEvent`], and shared payout/queue helpers.

use crate::card::Kind;
use crate::data::{HandType, Joker};
use crate::event::{Event, EventQueue};
use crate::game::{Areas, CardSet};
use crate::hand_eval::PokerHands;
use crate::pools::{NewCard, OwnedArea};
use crate::run::Run;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Scoring {
	pub name: HandType,
	pub hands: PokerHands,
	/// Indices into `G.play` of the scoring hand, in table order.
	pub cards: [u8; 8],
	pub n_cards: u8,
}

impl Scoring {
	/// The scoring cards, as indices into `G.play`.
	#[inline]
	pub fn scoring(&self) -> impl ExactSizeIterator<Item = usize> + '_ {
		self.cards[..usize::from(self.n_cards)]
			.iter()
			.map(|&i| usize::from(i))
	}
}

/// One `calculate_joker` call of hand scoring: the joker whose logic runs,
/// whether a copy joker runs it, and the hand being scored.
#[derive(Clone, Copy)]
pub(super) struct Call<'sc> {
	/// The joker's index in the joker area.
	pub(super) index: usize,
	pub(super) joker: Joker,
	/// A Blueprint or Brainstorm runs the logic of the joker it copies.
	pub(super) blueprinted: bool,
	/// The hand being scored.
	pub(super) scoring: &'sc Scoring,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CardArea {
	Play,
	Hand,
}

/// One card of a discard, as `context.discard` sees it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Discarded {
	/// The hand index of the card being discarded.
	pub other: usize,
	/// The hand index of the last card of the discard.
	pub last: usize,
	/// How many cards are discarded (highlighted).
	pub n_full: usize,
	/// How many of them are face cards.
	pub full_faces: usize,
}

/// The contexts of hand scoring (`joker::scoring`): each comes with the
/// [`Scoring`] of the hand.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ScoringCtx {
	DestroyingCard {
		play_idx: usize,
		full_len: usize,
	},
	DebuffedHand,
	Individual {
		area: CardArea,
		other: usize,
	},
	Repetition {
		area: CardArea,
		other: usize,
		effects: bool,
	},
	/// `other`: the joker being scored (`None` for a consumable).
	OtherJoker {
		other: Option<usize>,
	},
	Before,
	After,
	Main,
}

/// The contexts outside hand scoring (`joker::lifecycle`).
#[derive(Clone, Copy, Debug)]
pub(crate) enum LifecycleCtx {
	OpenBooster,
	SellingSelf,
	SellingCard,
	RerollShop,
	EndingShop,
	SkipBlind,
	SkippingBooster,
	PlayingCardAdded {
		n: usize,
	},
	FirstHandDrawn,
	SettingBlind {
		boss: bool,
	},
	RemovePlayingCards {
		faces: usize,
		glass: usize,
	},
	UsingConsumeable {
		kind: Kind,
		hanged_glass: usize,
	},
	/// `burnt_hand`: the poker hand of the discarded cards, when a Burnt
	/// Joker is there to read it.
	PreDiscard {
		hook: bool,
		burnt_hand: Option<HandType>,
	},
	Discard(Discarded),
	EndOfRound {
		game_over: bool,
	},
	EndOfRoundRepetition {
		effects: bool,
	},
}

/// A joker's returned effect table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Effect {
	pub chips: Option<f64>,
	pub mult: Option<f64>,
	pub x_mult: Option<f64>,
	pub h_mult: Option<f64>,
	pub dollars: Option<f64>,
	pub chip_mod: Option<f64>,
	pub mult_mod: Option<f64>,
	pub xmult_mod: Option<f64>,
	pub repetitions: u32,
	pub level_up: bool,
	pub saved: bool,
	pub remove: bool,
	/// `extra.func` that creates an 8 Ball tarot.
	pub eight_ball: bool,
	/// Has an `extra` table (individual-context bookkeeping).
	pub extra: bool,
	pub playing_cards_created: bool,
	/// `delay` of the returned table (status-text timing).
	pub delay: Option<f64>,
}

impl Effect {
	/// A table with only `extra`: the joker reacted without scoring.
	pub const EXTRA: Effect = Effect {
		extra: true,
		..Effect::NONE
	};
	/// The empty table, the base of the others
	/// (`Effect { mult: Some(3.0), ..Effect::NONE }`).
	pub const NONE: Effect = Effect {
		chips: None,
		mult: None,
		x_mult: None,
		h_mult: None,
		dollars: None,
		chip_mod: None,
		mult_mod: None,
		xmult_mod: None,
		repetitions: 0,
		level_up: false,
		saved: false,
		remove: false,
		eight_ball: false,
		extra: false,
		playing_cards_created: false,
		delay: None,
	};

	pub const fn chips(chips: f64) -> Effect {
		Effect {
			chips: Some(chips),
			..Effect::NONE
		}
	}

	pub const fn mult(mult: f64) -> Effect {
		Effect {
			mult: Some(mult),
			..Effect::NONE
		}
	}

	pub const fn x_mult(x_mult: f64) -> Effect {
		Effect {
			x_mult: Some(x_mult),
			..Effect::NONE
		}
	}

	pub const fn h_mult(h_mult: f64) -> Effect {
		Effect {
			h_mult: Some(h_mult),
			..Effect::NONE
		}
	}

	pub const fn chip_mod(chip_mod: f64) -> Effect {
		Effect {
			chip_mod: Some(chip_mod),
			..Effect::NONE
		}
	}

	pub const fn mult_mod(mult_mod: f64) -> Effect {
		Effect {
			mult_mod: Some(mult_mod),
			..Effect::NONE
		}
	}

	pub const fn xmult_mod(xmult_mod: f64) -> Effect {
		Effect {
			xmult_mod: Some(xmult_mod),
			..Effect::NONE
		}
	}
}

/// The events joker effects queue. An `...Outer` event is one whose body
/// queues the next event (its `...Inner` one, or the effect itself), as the
/// game's nested `add_event` calls do.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum JokerEvent {
	ChicotOuter,
	ChicotDisable,
	Burglar,
	RiffRaff {
		n: u32,
	},
	CartomancerOuter,
	DaggerSlice {
		dagger: u32,
		victim: u32,
	},
	MarbleStone,
	CertificateCard,
	GlassJokerOuter {
		id: u32,
		n: u32,
	},
	GlassJokerInner {
		id: u32,
		n: u32,
	},
	PerkeoCopy,
	/// A joker eating or melting itself: an event that queues the removal.
	EatenOuter {
		id: u32,
	},
	RemoveJokerLater {
		id: u32,
	},
	/// An event whose body calls `start_dissolve` on a joker.
	DissolveJoker {
		id: u32,
		time_factor: f64,
	},
	/// Vampire's juice event, which clears the card's `vampired` mark.
	Unvampire {
		id: u32,
	},
}

/// Schedules the "eaten/melted" self-destruction of the joker `id`
/// (`G.jokers:remove_card(self); self:remove()` in nested events).
pub(super) fn schedule_self_destruct(events: &mut EventQueue, id: u32) {
	events.push(Event::Joker(JokerEvent::EatenOuter { id }));
}

/// A consumable of `set` promised to the consumable area: created by a
/// queued event, and counted in the buffer until then.
pub(crate) fn queue_consumable(
	areas: &mut Areas, events: &mut EventQueue, set: CardSet, append: &'static str,
) {
	areas.consumable_buffer += 1.0;
	events.push_before(
		0.0,
		Event::Create(NewCard::new(set, OwnedArea::Consumables, append)),
	);
}

/// Money promised by a scoring effect before its `ease_dollars` event
/// runs (`G.GAME.dollar_buffer`).
#[inline]
pub(crate) fn buffer_dollars(run: &mut Run, events: &mut EventQueue, d: f64) {
	run.dollar_buffer += d;
	events.push(Event::DollarBufferReset);
}

/// A scoring effect that pays `d` dollars (buffered until paid).
pub(super) fn pay(run: &mut Run, events: &mut EventQueue, d: f64) -> Effect {
	buffer_dollars(run, events, d);
	Effect {
		dollars: Some(d),
		..Effect::NONE
	}
}

/// [`pay`] for a joker that queues its own `ease_dollars` first.
pub(super) fn ease_and_pay(run: &mut Run, events: &mut EventQueue, d: f64) -> Effect {
	events.push(Event::EaseDollars(d));
	pay(run, events, d)
}

/// `card_eval_status_text(joker, 'jokers', nil, percent, nil, eval)`.
#[inline]
pub(crate) fn effect_status(events: &mut EventQueue, e: &Effect) {
	events.status_text(e.delay.unwrap_or(0.75));
}
