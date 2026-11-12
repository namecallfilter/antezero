//! The events of the queue (see `events`): what each one does when it runs.

use crate::card::Kind;
use crate::data::{BlindKind, Edition, HandType, TagKind};
use crate::events::St;
use crate::game::{Area, CardSet};

/// Card piles addressable by events.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Pile {
	Deck,
	Hand,
	Play,
	Discard,
}

impl Pile {
	/// Timeline spelling (`trace-keys` builds).
	#[cfg(feature = "trace-keys")]
	pub(crate) fn name(self) -> &'static str {
		match self {
			Pile::Deck => "deck",
			Pile::Hand => "hand",
			Pile::Play => "play",
			Pile::Discard => "discard",
		}
	}
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Ev {
	/// A purely visual event (kept for its timing).
	Wait,
	/// An event whose body queues `update_hand_text({immediate = true, delay = 0})`.
	PushHandTextImm,
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
	/// `G.FUNCS.reroll_shop`'s 'after' event queueing the joker pass.
	RerollOuter,
	/// The pack UI being removed (with the unchosen pack cards).
	PackRemove,
	/// An event whose body calls `card_eval_status_text` (base delay).
	StatusLater(f64),
	/// An event whose body calls `ease_dollars(d)` + a status text.
	DollarsStatus {
		d: f64,
		delay: f64,
	},
	// --- generic -----------------------------------------------------
	EaseDollars(f64),
	EaseDiscard(f64),
	EaseHands(f64),
	EaseRound,
	EaseAnte(i32),
	DollarBufferReset,
	HandSize(f64),
	RefreshDebuffs,
	SetCostAll,
	/// `create_card` + `add_to_deck` + `emplace` (+ buffer reset).
	Create {
		set: CardSet,
		area: Area,
		forced: Option<Kind>,
		append: &'static str,
		rarity: Option<f64>,
		legendary: bool,
	},
	/// Joker eats/melts itself: an event that queues the removal.
	EatenOuter {
		id: u32,
	},
	RemoveJokerLater {
		id: u32,
	},
	/// An event whose body calls `start_dissolve` on a joker.
	DissolveJoker {
		id: u32,
		fac: f64,
	},
	/// The event that destroys a scored card (`shatter`/`start_dissolve`).
	DissolvePlaying {
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
	DrawCard {
		from: Pile,
		to: Pile,
		card: Option<u32>,
		sort: bool,
		discarded_only: bool,
	},
	AddTag {
		kind: TagKind,
		orbital: Option<HandType>,
	},

	// --- blind select / round ----------------------------------------
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
	RoundEvalInit,
	RoundEvalRun,
	CashOutFinish,
	BlindDisableCheck,
	AcornOuter,
	AcornShuffle,
	SkipBlindRun,
	RerollBossRun,

	// --- jokers --------------------------------------------------------
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
	/// Vampire's juice event, which clears the card's `vampired` mark.
	Unvampire {
		id: u32,
	},

	// --- tags ----------------------------------------------------------
	TagFunc {
		which: TagFn,
	},
	TagRemove {
		tag_id: u32,
	},

	// --- shop ----------------------------------------------------------
	ShopOuter,
	ShopPopulate {
		shop_exists: bool,
	},
	StoreModify {
		card: u32,
	},
	RerollShopRun,
	RerollShopJokers,
	BuyRun {
		card: u32,
		buy_and_use: bool,
	},
	SellRun {
		card: u32,
		area: Area,
	},
	VerdantDisable,
	ToggleShopRun,

	// --- packs / consumables ---------------------------------------------
	OpenPack {
		card: u32,
	},
	PackCreate,
	PackStateUi,
	PackDrawHand,
	UseFinishOuter {
		prev: St,
		area: Option<Area>,
		card: u32,
		dissolve: bool,
		delay_fac: f64,
	},
	UseFinishInner {
		prev: St,
		area: Option<Area>,
		delay_fac: f64,
	},
	EndConsumeable {
		df: f64,
	},
	EndConsumeableInner,
	Consumable(crate::consumables::CEv),
	LastTarotPlanetOuter {
		kind: Kind,
	},
	LastTarotPlanetInner {
		kind: Kind,
	},
}

/// The body of a tag's `yep(..., func)` callback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TagFn {
	TopUp,
	Charm,
	Meteor,
	Ethereal,
	Standard,
	Buffoon,
	Boss,
	Voucher,
	Double {
		kind: TagKind,
		orbital: Option<HandType>,
	},
	RareCard {
		card: u32,
	},
	UncommonCard {
		card: u32,
	},
	Edition {
		card: u32,
		edition: Edition,
	},
	D6,
	Coupon,
	Economy,
	Noop,
}
