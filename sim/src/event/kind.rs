//! The events of `G.E_MANAGER`'s queue: each is the body of one of the
//! game's `Event` functions. Events that belong to one part of the game
//! are that module's own enum, run by its `run_event`.

use crate::areas::Pile;
use crate::consumables::ConsumableEvent;
use crate::flow::FlowEvent;
use crate::joker::JokerEvent;
use crate::pools::{NewCard, OwnedArea};
use crate::run::TagEvent;
use crate::shop::{PackEvent, ShopEvent};

/// `draw_card(from, to, percent, dir, sort, card)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Draw {
	pub from: Pile,
	pub to: Pile,
	/// The card to move, by id; `None` takes the next one of `from`.
	pub card: Option<u32>,
	/// Sort the hand once the card is in it.
	pub sort: bool,
}

impl Draw {
	/// The next card of `from`, unsorted.
	pub(crate) fn next(from: Pile, to: Pile) -> Draw {
		Draw {
			from,
			to,
			card: None,
			sort: false,
		}
	}

	/// The card `id` of `from`, unsorted.
	pub(crate) fn card(from: Pile, to: Pile, id: u32) -> Draw {
		Draw {
			from,
			to,
			card: Some(id),
			sort: false,
		}
	}
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Event {
	/// A purely visual event (kept for its timing).
	Wait,
	/// An event whose body queues `update_hand_text({immediate = true, delay = 0})`.
	PushHandTextImmediate,
	/// An event whose body calls `start_materialize` (timing only).
	Materialize,
	/// `G.GAME.chips` reaching its eased target.
	SetChips(f64),
	/// `end_round`'s non-blocking poller calling `win_game()` in `ROUND_EVAL`.
	WinPoll,
	/// `win_game()`'s event: pauses the game behind the win overlay (the
	/// oracle, like a player choosing Endless, dismisses it at once).
	WinPause,
	/// `ease_chips(0)`'s outer event.
	EaseChips0,
	/// An event whose body calls `card_eval_status_text` (base delay).
	StatusLater(f64),
	/// An event whose body calls `ease_dollars(d)` + a status text.
	DollarsStatus {
		dollars: f64,
		delay: f64,
	},
	EaseDollars(f64),
	EaseDiscard(f64),
	EaseHands(f64),
	EaseRound,
	EaseAnte(i32),
	DollarBufferReset,
	HandSize(f64),
	/// `G.hand:sort()` after a hand-size change drew a card.
	SortHand,
	RefreshDebuffs,
	SetCostAll,
	/// The Checkered Deck's suits, once the deck is built.
	CheckeredDeck,
	/// `create_card` + `add_to_deck` + `emplace` (+ buffer reset).
	Create(NewCard<OwnedArea>),
	/// The event that destroys a scored card (`shatter`/`start_dissolve`).
	DissolvePlaying {
		id: u32,
	},
	/// Steamodded's `played_this_ante` mark on a card that was played (if it
	/// is still in the game).
	MarkPlayed {
		id: u32,
	},
	/// Removal at the end of a dissolve/shatter/explode animation. Those
	/// animations queue blocking-but-unblockable events that hold back every
	/// event queued after them, so removal is modelled as an event queued at
	/// the moment the animation starts.
	RemoveCard {
		id: u32,
	},
	/// `set_blind(nil)` at the end of `Blind:defeat`'s animation.
	BlindReset,
	DrawCard(Draw),
	Flow(FlowEvent),
	Joker(JokerEvent),
	Shop(ShopEvent),
	Pack(PackEvent),
	Tag(TagEvent),
	Consumable(ConsumableEvent),
}
