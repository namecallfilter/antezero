//! Shop state, booster-slot tracking, and [`ShopEvent`] definitions.

use crate::card::CardMask;
use crate::data::{Booster, Voucher};
use crate::game::{Area, State};

/// The boosters a shop offers.
pub const BOOSTER_SLOTS: usize = 2;

/// One of the shop's booster slots this round (`G.GAME.current_round.used_packs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackSlot {
	Unrolled,
	Offered(Booster),
	Opened,
}

/// The shop: whether it is open, what it offers this round, and the rules
/// that price and stock it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shop {
	/// `G.shop` exists.
	pub open: bool,
	/// `G.GAME.shop_free`.
	pub free: bool,
	/// `G.GAME.shop_d6ed`.
	pub d6ed: bool,
	pub reroll_cost: f64,
	pub reroll_cost_increase: f64,
	pub free_rerolls: f64,
	/// The reroll cost a round starts from (`G.GAME.round_resets.reroll_cost`).
	pub base_reroll_cost: f64,
	pub temp_reroll_cost: Option<f64>,
	/// `G.GAME.current_round.used_packs`.
	pub booster_slots: [PackSlot; BOOSTER_SLOTS],
	/// The ante's voucher (`G.GAME.current_round.voucher`).
	pub voucher: Option<Voucher>,
	/// The ante's voucher has been shown in a shop (observation only).
	pub voucher_seen: bool,
	/// `G.GAME.shop.joker_max`.
	pub joker_max: f64,
	pub discount_percent: f64,
	pub edition_rate: f64,
	pub joker_rate: f64,
	pub tarot_rate: f64,
	pub planet_rate: f64,
	pub spectral_rate: f64,
	pub playing_card_rate: f64,
	/// `G.GAME.first_shop_buffoon`.
	pub first_buffoon: bool,
}

impl Default for Shop {
	fn default() -> Shop {
		Shop {
			open: false,
			free: false,
			d6ed: false,
			reroll_cost: 5.0,
			reroll_cost_increase: 0.0,
			free_rerolls: 0.0,
			base_reroll_cost: 1.0,
			temp_reroll_cost: None,
			booster_slots: [PackSlot::Unrolled; BOOSTER_SLOTS],
			voucher: None,
			voucher_seen: false,
			joker_max: 2.0,
			discount_percent: 0.0,
			edition_rate: 1.0,
			joker_rate: 20.0,
			tarot_rate: 4.0,
			planet_rate: 4.0,
			spectral_rate: 0.0,
			playing_card_rate: 0.0,
			first_buffoon: false,
		}
	}
}

/// The events of the shop, of vouchers taking effect and of `use_card`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum ShopEvent {
	ShopOuter {
		shop_exists: bool,
	},
	ShopPopulate {
		shop_exists: bool,
	},
	StoreModify {
		card: u32,
	},
	RerollShopRun,
	/// `G.FUNCS.reroll_shop`'s 'after' event queueing the joker pass.
	RerollOuter,
	RerollShopJokers,
	BuyRun {
		card: u32,
		/// The hand cards to use it on at once, for a consumable.
		use_on: Option<CardMask>,
	},
	SellRun {
		card: u32,
		area: Area,
	},
	VerdantDisable,
	ToggleShopRun,
	UseFinishOuter {
		prev: State,
		area: Option<Area>,
		card: u32,
		dissolve: bool,
		delay_factor: f64,
	},
	UseFinishInner {
		prev: State,
		area: Option<Area>,
		delay_factor: f64,
	},
	ShopSize(f64),
	SetRate {
		which: Rate,
		v: f64,
	},
	/// `G.GAME.interest_cap`, which Seed Money and Money Tree raise.
	InterestCap(f64),
	ConsumableSlot,
	Discount(f64),
	RerollDiscount(f64),
	JokerSlot,
}

/// A `G.GAME` rate a voucher sets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Rate {
	Tarot,
	Planet,
	Edition,
	PlayingCard,
}
