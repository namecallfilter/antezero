//! Per-card (`context.individual`) joker scoring effects for played cards
//! and cards held in hand.

use super::effect::{Call, Effect, pay};
use crate::card::{PlayingCard, SuitTest};
use crate::data::{Enhancement, Joker, Rank, Suit};
use crate::event::EventQueue;
use crate::game::Areas;
use crate::rng::{Key, PseudoRandom};
use crate::round::Round;
use crate::run::Run;

/// `context.individual` for the scored card `other` of the play area.
pub(super) fn individual_played(
	areas: &mut Areas, run: &mut Run, rng: &mut PseudoRandom, round: &Round,
	events: &mut EventQueue, call: Call<'_>, other: usize,
) -> Option<Effect> {
	use Joker as J;
	let Call {
		index: ji,
		joker: j,
		blueprinted,
		scoring: sc,
	} = call;
	let prob = run.probabilities;
	let oc = &areas.play[other];
	match j {
		J::Hiker => {
			areas.play[other].perma_bonus += 5.0;
			Some(Effect::EXTRA)
		}
		J::LuckyCat if oc.lucky_trigger && !blueprinted => {
			areas.jokers[ji].ab.x_mult += 0.25;
			Some(Effect::EXTRA)
		}
		J::Wee if oc.has_rank(Rank::Two) && !blueprinted => {
			areas.jokers[ji].ab.chips += 8.0;
			Some(Effect::EXTRA)
		}
		J::Photograph => {
			let par = areas.pareidolia();
			let first_face = sc.scoring().find(|&i| areas.play[i].is_face(par));
			(first_face == Some(other)).then_some(Effect::x_mult(2.0))
		}
		J::J8Ball if areas.consumable_room_after_queued() && oc.has_rank(Rank::Eight) => {
			let k = Key::from("8ball");
			if rng.random(&k) < prob / 4.0 {
				areas.consumable_buffer += 1.0;
				Some(Effect {
					extra: true,
					eight_ball: true,
					..Effect::NONE
				})
			} else {
				None
			}
		}
		J::Idol
			if oc.effective_rank() == Some(round.idol_card.rank)
				&& oc.is_suit(round.idol_card.suit, SuitTest::Scoring, areas.smeared()) =>
		{
			Some(Effect::x_mult(2.0))
		}
		J::ScaryFace if oc.is_face(areas.pareidolia()) => Some(Effect::chips(30.0)),
		J::Smiley if oc.is_face(areas.pareidolia()) => Some(Effect::mult(5.0)),
		J::Ticket if oc.enhancement == Enhancement::Gold => Some(pay(run, events, 4.0)),
		J::Business if oc.is_face(areas.pareidolia()) => {
			let k = Key::from("business");
			if rng.random(&k) < prob / 2.0 {
				Some(pay(run, events, 2.0))
			} else {
				None
			}
		}
		_ => rank_joker(oc, j)
			.or_else(|| suit_joker(oc, areas.smeared(), run, round.ancient_suit, rng, events, j)),
	}
}

/// The jokers that pay for the rank of the scored card `oc`.
fn rank_joker(oc: &PlayingCard, j: Joker) -> Option<Effect> {
	use Joker as J;
	match j {
		J::Scholar if oc.has_rank(Rank::Ace) => Some(Effect {
			chips: Some(20.0),
			mult: Some(4.0),
			..Effect::NONE
		}),
		J::WalkieTalkie if oc.has_rank(Rank::Ten) || oc.has_rank(Rank::Four) => Some(Effect {
			chips: Some(10.0),
			mult: Some(4.0),
			..Effect::NONE
		}),
		J::Fibonacci
			if matches!(
				oc.effective_rank(),
				Some(Rank::Two | Rank::Three | Rank::Five | Rank::Eight | Rank::Ace)
			) =>
		{
			Some(Effect::mult(8.0))
		}
		J::EvenSteven
			if oc
				.effective_rank()
				.is_some_and(|r| r <= Rank::Ten && r.id() % 2 == 0) =>
		{
			Some(Effect::mult(4.0))
		}
		J::OddTodd
			if oc
				.effective_rank()
				.is_some_and(|r| (r <= Rank::Ten && r.id() % 2 == 1) || r == Rank::Ace) =>
		{
			Some(Effect::chips(31.0))
		}
		J::Triboulet if oc.has_rank(Rank::Queen) || oc.has_rank(Rank::King) => {
			Some(Effect::x_mult(2.0))
		}
		_ => None,
	}
}

/// The jokers that pay for the suit of the scored card `oc`.
fn suit_joker(
	oc: &PlayingCard, smeared: bool, run: &mut Run, ancient_suit: Suit, rng: &mut PseudoRandom,
	events: &mut EventQueue, j: Joker,
) -> Option<Effect> {
	use Joker as J;
	let scores = |suit: Suit| oc.is_suit(suit, SuitTest::Scoring, smeared);
	match j {
		J::GreedyJoker if scores(Suit::Diamonds) => Some(Effect::mult(3.0)),
		J::LustyJoker if scores(Suit::Hearts) => Some(Effect::mult(3.0)),
		J::WrathfulJoker if scores(Suit::Spades) => Some(Effect::mult(3.0)),
		J::GluttenousJoker if scores(Suit::Clubs) => Some(Effect::mult(3.0)),
		J::RoughGem if scores(Suit::Diamonds) => Some(pay(run, events, 1.0)),
		J::OnyxAgate if scores(Suit::Clubs) => Some(Effect::mult(7.0)),
		J::Arrowhead if scores(Suit::Spades) => Some(Effect::chips(50.0)),
		J::Bloodstone if scores(Suit::Hearts) => {
			let k = Key::from("bloodstone");
			(rng.random(&k) < run.probabilities / 2.0).then_some(Effect::x_mult(1.5))
		}
		J::Ancient if scores(ancient_suit) => Some(Effect::x_mult(1.5)),
		_ => None,
	}
}

/// `context.individual` for the card `other` held in hand. A debuffed card
/// still triggers the jokers that pay for it, which then return only `extra`.
pub(super) fn individual_held(
	areas: &Areas, run: &mut Run, rng: &mut PseudoRandom, events: &mut EventQueue, j: Joker,
	other: usize,
) -> Option<Effect> {
	use Joker as J;
	let oc = &areas.hand.cards[other];
	let unless_debuffed = |e: Effect| Some(if oc.core.debuff { Effect::EXTRA } else { e });
	match j {
		J::ShootTheMoon if oc.has_rank(Rank::Queen) => unless_debuffed(Effect::h_mult(13.0)),
		J::Baron if oc.has_rank(Rank::King) => unless_debuffed(Effect::x_mult(1.5)),
		J::ReservedParking if oc.is_face(areas.pareidolia()) => {
			let k = Key::from("parking");
			if rng.random(&k) < run.probabilities / 2.0 {
				if oc.core.debuff {
					Some(Effect::EXTRA)
				} else {
					Some(pay(run, events, 1.0))
				}
			} else {
				None
			}
		}
		J::RaisedFist => {
			let mut lowest: Option<(usize, Rank)> = None;
			for (i, c) in areas.hand.cards.iter().enumerate() {
				if !c.is_stone() && lowest.is_none_or(|(_, low)| low >= c.rank) {
					lowest = Some((i, c.rank));
				}
			}
			match lowest {
				Some((i, rank)) if i == other => {
					unless_debuffed(Effect::h_mult(2.0 * rank.nominal()))
				}
				_ => None,
			}
		}
		_ => None,
	}
}
