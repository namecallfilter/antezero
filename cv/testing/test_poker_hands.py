"""
Unit tests for deterministic poker hand evaluation logic in game/engine/poker_hands.py.

Usage:
    python -m unittest testing/test_poker_hands.py
    python testing/test_poker_hands.py
"""

from pathlib import Path
import sys
import unittest

# Ensure repo root and 'game' directory are in pythonpath
REPO_ROOT = Path(__file__).resolve().parent.parent
GAME_DIR = REPO_ROOT / "game"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(GAME_DIR) not in sys.path:
    sys.path.insert(0, str(GAME_DIR))

try:
    from game.balatro_models import (
        Enhancement,
        Joker,
        JokerName,
        PlayingCard,
        PokerHand,
        Rank,
        Suit,
    )
    from game.engine.poker_hands import evaluate_poker_hand, get_scoring_cards
except ImportError:
    from balatro_models import (
        Enhancement,
        Joker,
        JokerName,
        PlayingCard,
        PokerHand,
        Rank,
        Suit,
    )
    from engine.poker_hands import evaluate_poker_hand, get_scoring_cards


class TestPokerHands(unittest.TestCase):
    """Unit test suite for poker hand classifications and modifier jokers."""

    def test_high_card(self):
        cards = [
            PlayingCard(rank=Rank.ACE, suit=Suit.HEARTS),
            PlayingCard(rank=Rank.TEN, suit=Suit.CLUBS),
        ]
        self.assertEqual(evaluate_poker_hand(cards), PokerHand.HIGH_CARD)

    def test_pair(self):
        cards = [
            PlayingCard(rank=Rank.ACE, suit=Suit.HEARTS),
            PlayingCard(rank=Rank.ACE, suit=Suit.CLUBS),
            PlayingCard(rank=Rank.SEVEN, suit=Suit.DIAMONDS),
        ]
        self.assertEqual(evaluate_poker_hand(cards), PokerHand.PAIR)

    def test_two_pair(self):
        cards = [
            PlayingCard(rank=Rank.ACE, suit=Suit.HEARTS),
            PlayingCard(rank=Rank.ACE, suit=Suit.CLUBS),
            PlayingCard(rank=Rank.SEVEN, suit=Suit.DIAMONDS),
            PlayingCard(rank=Rank.SEVEN, suit=Suit.SPADES),
        ]
        self.assertEqual(evaluate_poker_hand(cards), PokerHand.TWO_PAIR)

    def test_three_of_a_kind(self):
        cards = [
            PlayingCard(rank=Rank.NINE, suit=Suit.HEARTS),
            PlayingCard(rank=Rank.NINE, suit=Suit.CLUBS),
            PlayingCard(rank=Rank.NINE, suit=Suit.SPADES),
            PlayingCard(rank=Rank.FOUR, suit=Suit.DIAMONDS),
        ]
        self.assertEqual(evaluate_poker_hand(cards), PokerHand.THREE_OF_A_KIND)

    def test_straight_broadway_and_wheel(self):
        # Broadway Straight (10-J-Q-K-A)
        c_broadway = [
            PlayingCard(rank=Rank.TEN, suit=Suit.HEARTS),
            PlayingCard(rank=Rank.JACK, suit=Suit.CLUBS),
            PlayingCard(rank=Rank.QUEEN, suit=Suit.SPADES),
            PlayingCard(rank=Rank.KING, suit=Suit.DIAMONDS),
            PlayingCard(rank=Rank.ACE, suit=Suit.HEARTS),
        ]
        self.assertEqual(evaluate_poker_hand(c_broadway), PokerHand.STRAIGHT)

        # Wheel Straight (A-2-3-4-5)
        c_wheel = [
            PlayingCard(rank=Rank.ACE, suit=Suit.SPADES),
            PlayingCard(rank=Rank.TWO, suit=Suit.HEARTS),
            PlayingCard(rank=Rank.THREE, suit=Suit.CLUBS),
            PlayingCard(rank=Rank.FOUR, suit=Suit.DIAMONDS),
            PlayingCard(rank=Rank.FIVE, suit=Suit.SPADES),
        ]
        self.assertEqual(evaluate_poker_hand(c_wheel), PokerHand.STRAIGHT)

    def test_flush(self):
        cards = [
            PlayingCard(rank=Rank.TWO, suit=Suit.HEARTS),
            PlayingCard(rank=Rank.FIVE, suit=Suit.HEARTS),
            PlayingCard(rank=Rank.SEVEN, suit=Suit.HEARTS),
            PlayingCard(rank=Rank.JACK, suit=Suit.HEARTS),
            PlayingCard(rank=Rank.KING, suit=Suit.HEARTS),
        ]
        self.assertEqual(evaluate_poker_hand(cards), PokerHand.FLUSH)

    def test_full_house(self):
        cards = [
            PlayingCard(rank=Rank.KING, suit=Suit.HEARTS),
            PlayingCard(rank=Rank.KING, suit=Suit.CLUBS),
            PlayingCard(rank=Rank.KING, suit=Suit.SPADES),
            PlayingCard(rank=Rank.THREE, suit=Suit.DIAMONDS),
            PlayingCard(rank=Rank.THREE, suit=Suit.HEARTS),
        ]
        self.assertEqual(evaluate_poker_hand(cards), PokerHand.FULL_HOUSE)

    def test_four_of_a_kind(self):
        cards = [
            PlayingCard(rank=Rank.EIGHT, suit=Suit.HEARTS),
            PlayingCard(rank=Rank.EIGHT, suit=Suit.CLUBS),
            PlayingCard(rank=Rank.EIGHT, suit=Suit.SPADES),
            PlayingCard(rank=Rank.EIGHT, suit=Suit.DIAMONDS),
            PlayingCard(rank=Rank.TWO, suit=Suit.HEARTS),
        ]
        self.assertEqual(evaluate_poker_hand(cards), PokerHand.FOUR_OF_A_KIND)

    def test_straight_flush(self):
        cards = [
            PlayingCard(rank=Rank.SIX, suit=Suit.CLUBS),
            PlayingCard(rank=Rank.SEVEN, suit=Suit.CLUBS),
            PlayingCard(rank=Rank.EIGHT, suit=Suit.CLUBS),
            PlayingCard(rank=Rank.NINE, suit=Suit.CLUBS),
            PlayingCard(rank=Rank.TEN, suit=Suit.CLUBS),
        ]
        self.assertEqual(evaluate_poker_hand(cards), PokerHand.STRAIGHT_FLUSH)

    def test_five_of_a_kind(self):
        cards = [
            PlayingCard(rank=Rank.JACK, suit=Suit.HEARTS),
            PlayingCard(rank=Rank.JACK, suit=Suit.CLUBS),
            PlayingCard(rank=Rank.JACK, suit=Suit.SPADES),
            PlayingCard(rank=Rank.JACK, suit=Suit.DIAMONDS),
            PlayingCard(rank=Rank.JACK, suit=Suit.HEARTS),
        ]
        self.assertEqual(evaluate_poker_hand(cards), PokerHand.FIVE_OF_A_KIND)

    def test_flush_house(self):
        cards = [
            PlayingCard(rank=Rank.QUEEN, suit=Suit.DIAMONDS),
            PlayingCard(rank=Rank.QUEEN, suit=Suit.DIAMONDS),
            PlayingCard(rank=Rank.QUEEN, suit=Suit.DIAMONDS),
            PlayingCard(rank=Rank.FIVE, suit=Suit.DIAMONDS),
            PlayingCard(rank=Rank.FIVE, suit=Suit.DIAMONDS),
        ]
        self.assertEqual(evaluate_poker_hand(cards), PokerHand.FLUSH_HOUSE)

    def test_flush_five(self):
        cards = [
            PlayingCard(rank=Rank.ACE, suit=Suit.SPADES),
            PlayingCard(rank=Rank.ACE, suit=Suit.SPADES),
            PlayingCard(rank=Rank.ACE, suit=Suit.SPADES),
            PlayingCard(rank=Rank.ACE, suit=Suit.SPADES),
            PlayingCard(rank=Rank.ACE, suit=Suit.SPADES),
        ]
        self.assertEqual(evaluate_poker_hand(cards), PokerHand.FLUSH_FIVE)

    def test_modifier_four_fingers(self):
        ff_joker = [Joker(name=JokerName.FOUR_FINGERS)]

        c_straight_4 = [
            PlayingCard(rank=Rank.TEN, suit=Suit.HEARTS),
            PlayingCard(rank=Rank.JACK, suit=Suit.CLUBS),
            PlayingCard(rank=Rank.QUEEN, suit=Suit.SPADES),
            PlayingCard(rank=Rank.KING, suit=Suit.DIAMONDS),
        ]
        self.assertEqual(evaluate_poker_hand(c_straight_4, ff_joker), PokerHand.STRAIGHT)

        c_flush_4 = [
            PlayingCard(rank=Rank.TWO, suit=Suit.HEARTS),
            PlayingCard(rank=Rank.FIVE, suit=Suit.HEARTS),
            PlayingCard(rank=Rank.SEVEN, suit=Suit.HEARTS),
            PlayingCard(rank=Rank.JACK, suit=Suit.HEARTS),
        ]
        self.assertEqual(evaluate_poker_hand(c_flush_4, ff_joker), PokerHand.FLUSH)

    def test_modifier_shortcut(self):
        sc_joker = [Joker(name=JokerName.SHORTCUT)]
        c_shortcut_straight = [
            PlayingCard(rank=Rank.TWO, suit=Suit.HEARTS),
            PlayingCard(rank=Rank.FOUR, suit=Suit.CLUBS),
            PlayingCard(rank=Rank.FIVE, suit=Suit.SPADES),
            PlayingCard(rank=Rank.SEVEN, suit=Suit.DIAMONDS),
            PlayingCard(rank=Rank.NINE, suit=Suit.HEARTS),
        ]
        self.assertEqual(evaluate_poker_hand(c_shortcut_straight, sc_joker), PokerHand.STRAIGHT)

    def test_modifier_smeared_joker(self):
        smeared_joker = [Joker(name=JokerName.SMEARED_JOKER)]
        c_smeared_flush = [
            PlayingCard(rank=Rank.TWO, suit=Suit.HEARTS),
            PlayingCard(rank=Rank.FIVE, suit=Suit.DIAMONDS),
            PlayingCard(rank=Rank.SEVEN, suit=Suit.HEARTS),
            PlayingCard(rank=Rank.JACK, suit=Suit.DIAMONDS),
            PlayingCard(rank=Rank.KING, suit=Suit.HEARTS),
        ]
        self.assertEqual(evaluate_poker_hand(c_smeared_flush, smeared_joker), PokerHand.FLUSH)

    def test_enhancement_wild_card(self):
        c_wild_flush = [
            PlayingCard(rank=Rank.TWO, suit=Suit.SPADES),
            PlayingCard(rank=Rank.FIVE, suit=Suit.SPADES),
            PlayingCard(rank=Rank.SEVEN, suit=Suit.SPADES),
            PlayingCard(rank=Rank.JACK, suit=Suit.SPADES),
            PlayingCard(rank=Rank.KING, suit=Suit.HEARTS, enhancement=Enhancement.WILD),
        ]
        self.assertEqual(evaluate_poker_hand(c_wild_flush), PokerHand.FLUSH)

    def test_enhancement_stone_card(self):
        c_stone_play = [
            PlayingCard(rank=Rank.ACE, suit=Suit.HEARTS),
            PlayingCard(rank=Rank.ACE, suit=Suit.CLUBS),
            PlayingCard(rank=Rank.ACE, suit=Suit.SPADES, enhancement=Enhancement.STONE),
        ]
        # Stone card has no rank/suit, so it's a Pair of Aces, not Three of a Kind
        self.assertEqual(evaluate_poker_hand(c_stone_play), PokerHand.PAIR)

    def test_empty_cards(self):
        self.assertEqual(evaluate_poker_hand([]), PokerHand.HIGH_CARD)

    def test_scoring_cards_high_card(self):
        # In High Card 8 with 6 and 5, ONLY the 8 scores
        c8 = PlayingCard(rank=Rank.EIGHT, suit=Suit.HEARTS)
        c6 = PlayingCard(rank=Rank.SIX, suit=Suit.CLUBS)
        c5 = PlayingCard(rank=Rank.FIVE, suit=Suit.SPADES)
        scoring = get_scoring_cards([c8, c6, c5])
        self.assertEqual(scoring, [c8])

    def test_scoring_cards_pair_excludes_kickers(self):
        # In Pair of 8s with King, Queen, Two, ONLY the two 8s score
        c8a = PlayingCard(rank=Rank.EIGHT, suit=Suit.HEARTS)
        c8b = PlayingCard(rank=Rank.EIGHT, suit=Suit.CLUBS)
        cK = PlayingCard(rank=Rank.KING, suit=Suit.DIAMONDS)
        cQ = PlayingCard(rank=Rank.QUEEN, suit=Suit.SPADES)
        c2 = PlayingCard(rank=Rank.TWO, suit=Suit.HEARTS)
        scoring = get_scoring_cards([c8a, c8b, cK, cQ, c2])
        self.assertEqual(scoring, [c8a, c8b])

    def test_scoring_cards_two_pair(self):
        cKa = PlayingCard(rank=Rank.KING, suit=Suit.HEARTS)
        cKb = PlayingCard(rank=Rank.KING, suit=Suit.CLUBS)
        c3a = PlayingCard(rank=Rank.THREE, suit=Suit.DIAMONDS)
        c3b = PlayingCard(rank=Rank.THREE, suit=Suit.SPADES)
        c2 = PlayingCard(rank=Rank.TWO, suit=Suit.HEARTS)
        scoring = get_scoring_cards([cKa, cKb, c3a, c3b, c2])
        self.assertEqual(scoring, [cKa, cKb, c3a, c3b])

    def test_scoring_cards_three_of_a_kind(self):
        c9a = PlayingCard(rank=Rank.NINE, suit=Suit.HEARTS)
        c9b = PlayingCard(rank=Rank.NINE, suit=Suit.CLUBS)
        c9c = PlayingCard(rank=Rank.NINE, suit=Suit.SPADES)
        c4 = PlayingCard(rank=Rank.FOUR, suit=Suit.DIAMONDS)
        c2 = PlayingCard(rank=Rank.TWO, suit=Suit.HEARTS)
        scoring = get_scoring_cards([c9a, c9b, c9c, c4, c2])
        self.assertEqual(scoring, [c9a, c9b, c9c])

    def test_scoring_cards_four_of_a_kind(self):
        c8a = PlayingCard(rank=Rank.EIGHT, suit=Suit.HEARTS)
        c8b = PlayingCard(rank=Rank.EIGHT, suit=Suit.CLUBS)
        c8c = PlayingCard(rank=Rank.EIGHT, suit=Suit.SPADES)
        c8d = PlayingCard(rank=Rank.EIGHT, suit=Suit.DIAMONDS)
        c2 = PlayingCard(rank=Rank.TWO, suit=Suit.HEARTS)
        scoring = get_scoring_cards([c8a, c8b, c8c, c8d, c2])
        self.assertEqual(scoring, [c8a, c8b, c8c, c8d])

    def test_scoring_cards_stone_card_always_scores(self):
        c8a = PlayingCard(rank=Rank.EIGHT, suit=Suit.HEARTS)
        c8b = PlayingCard(rank=Rank.EIGHT, suit=Suit.CLUBS)
        stone = PlayingCard(rank=Rank.ACE, suit=Suit.SPADES, enhancement=Enhancement.STONE)
        scoring = get_scoring_cards([c8a, c8b, stone])
        self.assertIn(c8a, scoring)
        self.assertIn(c8b, scoring)
        self.assertIn(stone, scoring)

    def test_scoring_cards_splash_joker_scores_all(self):
        splash = [Joker(name=JokerName.SPLASH)]
        c8a = PlayingCard(rank=Rank.EIGHT, suit=Suit.HEARTS)
        c8b = PlayingCard(rank=Rank.EIGHT, suit=Suit.CLUBS)
        cK = PlayingCard(rank=Rank.KING, suit=Suit.DIAMONDS)
        cQ = PlayingCard(rank=Rank.QUEEN, suit=Suit.SPADES)
        c2 = PlayingCard(rank=Rank.TWO, suit=Suit.HEARTS)
        all_played = [c8a, c8b, cK, cQ, c2]
        scoring = get_scoring_cards(all_played, jokers=splash)
        self.assertEqual(scoring, all_played)


def test_poker_hands() -> None:
    """Convenience function running all test cases directly."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPokerHands)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        raise AssertionError("One or more poker hand tests failed.")
    print("ALL POKER HAND TESTS PASSED PERFECTLY!")


if __name__ == "__main__":
    unittest.main()
