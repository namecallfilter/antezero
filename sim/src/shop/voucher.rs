//! Voucher redemption (`Card:redeem`, `Card:apply_to_run`).

use super::state::{Rate, Shop, ShopEvent};
use crate::data::Voucher;
use crate::event::{Event, EventQueue};
use crate::round::Round;
use crate::run::History;
use crate::{flow, num};

/// `Card:redeem()` for the voucher `v`, bought at `cost`.
pub(super) fn redeem(
	history: &mut History, round: &mut Round, shop: &mut Shop, events: &mut EventQueue, v: Voucher,
	cost: f64,
) {
	history.used_vouchers.insert(v);
	events.delay(0.4);
	events.push(Event::EaseDollars(-cost));
	shop.voucher = None;
	apply_voucher(round, events, v, VoucherSource::Redeemed);
	events.delay(0.6);
}

/// What gives the run a voucher.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VoucherSource {
	Redeemed,
	/// A deck's starting voucher (`from_back`).
	Deck,
}

/// `Card:apply_to_run(center)`.
pub(crate) fn apply_voucher(
	round: &mut Round, events: &mut EventQueue, v: Voucher, source: VoucherSource,
) {
	use Voucher as V;
	let extra = v.def().extra;
	match v {
		V::OverstockNorm | V::OverstockPlus => events.push(Event::Shop(ShopEvent::ShopSize(1.0))),
		V::TarotMerchant | V::TarotTycoon => events.push(Event::Shop(ShopEvent::SetRate {
			which: Rate::Tarot,
			v: 4.0 * extra,
		})),
		V::PlanetMerchant | V::PlanetTycoon => events.push(Event::Shop(ShopEvent::SetRate {
			which: Rate::Planet,
			v: 4.0 * extra,
		})),
		V::Hone | V::GlowUp => events.push(Event::Shop(ShopEvent::SetRate {
			which: Rate::Edition,
			v: extra,
		})),
		V::MagicTrick | V::Illusion => events.push(Event::Shop(ShopEvent::SetRate {
			which: Rate::PlayingCard,
			v: extra,
		})),
		V::CrystalBall => events.push(Event::Shop(ShopEvent::ConsumableSlot)),
		V::ClearanceSale | V::Liquidation => events.push(Event::Shop(ShopEvent::Discount(extra))),
		V::RerollSurplus | V::RerollGlut => {
			if source == VoucherSource::Redeemed {
				events.push(Event::Shop(ShopEvent::RerollDiscount(extra)));
			}
		}
		V::SeedMoney | V::MoneyTree => events.push(Event::Shop(ShopEvent::InterestCap(extra))),
		V::Grabber | V::NachoTong => {
			round.base_hands += extra;
			events.push(Event::EaseHands(extra));
		}
		V::PaintBrush | V::Palette => flow::change_hand_size(events, 1.0),
		V::Wasteful | V::Recyclomancy => {
			round.base_discards += extra;
			events.push(Event::EaseDiscard(extra));
		}
		V::Antimatter => events.push(Event::Shop(ShopEvent::JokerSlot)),
		V::Hieroglyph => {
			events.push(Event::EaseAnte(-antes_back(extra)));
			round.base_hands -= extra;
			events.push(Event::EaseHands(-extra));
		}
		V::Petroglyph => {
			events.push(Event::EaseAnte(-antes_back(extra)));
			round.base_discards -= extra;
			events.push(Event::EaseDiscard(-extra));
		}
		// The game reads these from the redeemed vouchers where they apply.
		V::OmenGlobe | V::Telescope | V::Observatory | V::Blank | V::DirectorsCut | V::Retcon => {}
	}
}

/// The antes Hieroglyph and Petroglyph take back (their `extra`).
fn antes_back(extra: f64) -> i32 {
	i32::try_from(num::trunc(extra)).expect("a voucher lowers the ante by one")
}
