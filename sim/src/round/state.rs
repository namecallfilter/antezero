//! The current round (`G.GAME.current_round`, `round_resets` and
//! `round_bonus`).

use crate::areas::Areas;
use crate::card::PlayingCard;
use crate::data::{HandType, Rank, Suit};
use crate::rng::PseudoRandom;
use crate::run::Run;

/// The Idol's card (`G.GAME.current_round.idol_card`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IdolCard {
	pub rank: Rank,
	pub suit: Suit,
	/// The card's id, which keeps its last valid value like the game's.
	pub id: Option<Rank>,
}

/// Mail-In Rebate's rank (`G.GAME.current_round.mail_card`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MailRank {
	pub rank: Rank,
	/// The rank's id, which keeps its last valid value like the game's.
	pub id: Option<Rank>,
}

/// The current round's counters and the cards its jokers target.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Round {
	/// The score so far (`G.GAME.chips`).
	pub chips: f64,
	pub hands_left: f64,
	pub discards_left: f64,
	pub hands_played: f64,
	pub discards_used: f64,
	/// The cash-out total (`G.GAME.current_round.dollars`).
	pub cash_out: f64,
	/// The hands and discards a round starts with
	/// (`G.GAME.round_resets.hands` / `.discards`).
	pub base_hands: f64,
	pub base_discards: f64,
	pub temp_handsize: Option<f64>,
	/// `G.GAME.round_bonus`: added to the next round's hands and discards.
	pub bonus_next_hands: f64,
	pub bonus_discards: f64,
	pub idol_card: IdolCard,
	pub mail_rank: MailRank,
	pub ancient_suit: Suit,
	pub castle_suit: Suit,
	pub most_played_hand: HandType,
	pub facing_blind: bool,
	pub last_blind_boss: bool,
}

impl Round {
	/// The hands and discards a round starts with: `round_resets` plus
	/// `round_bonus`.
	pub(crate) fn reset_hands_and_discards(&mut self) {
		self.discards_left = 0f64.max(self.base_discards + self.bonus_discards);
		self.hands_left = 1f64.max(self.base_hands + self.bonus_next_hands);
	}

	/// Whether the score has beaten a blind of `blind_chips`. The game tests
	/// `G.GAME.chips - G.GAME.blind.chips >= 0`, which an infinite score
	/// against an infinite blind fails (the difference is NaN).
	#[must_use]
	pub(crate) fn beats(&self, blind_chips: f64) -> bool {
		self.chips - blind_chips >= 0.0
	}

	/// `reset_idol_card`, `reset_mail_rank`, `reset_ancient_card` and
	/// `reset_castle_card`: new cards for The Idol, Mail-In Rebate, Ancient
	/// Joker and Castle, drawn in that order.
	pub(crate) fn reset_joker_targets(
		&mut self, areas: &Areas, run: &Run, rng: &mut PseudoRandom, when: JokerTargets,
	) {
		// At the start of a run the game clears the ancient suit first, so any
		// suit can come; at the end of a round it changes to another one.
		let old_ancient = (when == JokerTargets::RoundEnd).then_some(self.ancient_suit);
		let cards = areas.non_stone_sorted();
		// Without a card, The Idol and Mail-In Rebate keep their last valid id.
		if let Some(c) = pick_target(&cards, run, rng, "idol") {
			self.idol_card = IdolCard {
				rank: c.rank,
				suit: c.suit,
				id: Some(c.rank),
			};
		} else {
			self.idol_card.rank = Rank::Ace;
			self.idol_card.suit = Suit::Spades;
		}
		if let Some(c) = pick_target(&cards, run, rng, "mail") {
			self.mail_rank = MailRank {
				rank: c.rank,
				id: Some(c.rank),
			};
		} else {
			self.mail_rank.rank = Rank::Ace;
		}
		let mut suits = [Suit::Spades; Suit::ALL.len()];
		let mut n = 0;
		for s in Suit::ALL {
			if Some(s) != old_ancient {
				suits[n] = s;
				n += 1;
			}
		}
		self.ancient_suit = suits[rng.element(&run.key_ante("anc"), n)];
		self.castle_suit = pick_target(&cards, run, rng, "cas").map_or(Suit::Spades, |c| c.suit);
	}
}

/// When [`Round::reset_joker_targets`] runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum JokerTargets {
	RunStart,
	RoundEnd,
}

/// `pseudorandom_element(cards, pseudoseed(key .. ante))` of `cards` in
/// `sort_id` order, or `None` without a draw if there are no cards.
fn pick_target<'cards>(
	cards: &[&'cards PlayingCard], run: &Run, rng: &mut PseudoRandom, key: &str,
) -> Option<&'cards PlayingCard> {
	if cards.is_empty() {
		return None;
	}
	Some(cards[rng.element(&run.key_ante(key), cards.len())])
}

impl Default for Round {
	fn default() -> Round {
		Round {
			chips: 0.0,
			hands_left: 0.0,
			discards_left: 0.0,
			hands_played: 0.0,
			discards_used: 0.0,
			cash_out: 0.0,
			base_hands: 1.0,
			base_discards: 1.0,
			temp_handsize: None,
			bonus_next_hands: 0.0,
			bonus_discards: 0.0,
			idol_card: IdolCard {
				rank: Rank::Ace,
				suit: Suit::Spades,
				id: None,
			},
			mail_rank: MailRank {
				rank: Rank::Ace,
				id: None,
			},
			ancient_suit: Suit::Spades,
			castle_suit: Suit::Spades,
			most_played_hand: HandType::HighCard,
			facing_blind: false,
			last_blind_boss: false,
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn an_infinite_score_does_not_beat_an_infinite_blind() {
		let round = Round {
			chips: f64::INFINITY,
			..Round::default()
		};
		assert!(!round.beats(f64::INFINITY));
		assert!(round.beats(f64::MAX));
	}
}
