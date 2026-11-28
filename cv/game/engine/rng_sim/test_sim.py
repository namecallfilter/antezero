"""Comprehensive Unit Tests for Balatro Deterministic Simulation Engine (`game.engine.rng_sim`).

Verifies:
1. Exact bit-level parity of pseudohash, pseudoseed, Tw223, and LCG step against
   reference test vectors from Immolate and LuaJIT PRNG.
2. Reproducibility for seed 'HELLO' across multiple runs.
3. Step-dependent state advancement and PRNG channel consumption.
4. Certificate joker card & seal generation (both ante-scoped and `balatro_source` parity).
5. Lucky Cards (+20 Mult and +$20 Cash) probability mechanics and Oops! All 6s scaling.
6. Mail-in Rebate dynamic target selection and discard evaluations.
7. 1-in-N outcome resolvers (Wheel of Fortune, Space Joker, Bloodstone, Glass, Business Card).
8. Deck mutations and exact `sort_id` ordering (`C_2..C_T, D_2..D_T, H_2..H_T, S_2..S_T`).
9. Castle suit selection per round (`reset_castle_card`, deck composition & Stone Card filtering).
10. Ancient Joker suit selection per round (`reset_ancient_card`, 4-suit init & 3-suit non-repeat).
11. The Idol card selection per round (`reset_idol_card`, deck composition & Stone Card filtering).
12. To Do List poker hand selection (`'to_do'`, initial spawn vs end-of-round non-repeating reset, secret hand unlocks).
13. Cartomancer (`'Tarotcar{ante}'`) and Vagabond (`'Tarotvag{ante}'`) Tarot generation, active Tarot blocking, and Showman override.
14. Marble Joker / Stone Joker (`'marb_fr'`) hidden suit & rank generation and reveal upon re-enhancement.
15. Boss Blind queue (`'boss'`), including `min_ante` locks, Ante 8 Showdown bosses, boss rerolls, Hieroglyph/Petroglyph (`-1 ante` vouchers at Ante 1 and Ante 8), and `bosses_used` cycle wrap-around.
16. Skip Tag queue (`'Tag{ante}'`), including Ante 1 `min_ante = 2` resampling, `-1 ante` vouchers, and Orbital Tag (`'orbital'`) poker hand selection.
17. Voucher queue (`'Voucher{ante}'` vs `'Voucher_fromtag'`), including unbought voucher re-appearances, Tier-1 -> Tier-2 prerequisite unlocking, starting deck vouchers (Magic/Nebula/Zodiac), and Voucher Tag global stream behavior.
18. Mid-ante deck mutations: stateful round-by-round progression AND direct `round_num` jumping (`round_num=2|3`) without evaluating earlier rounds.
19. Direct integration with `game.balatro_models` (`PlayingCard`, `Joker`, `Tarot`, `Voucher`, enums) and `game.gamestate.GameState`.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from game.balatro_models import (
    BlindName,
    Deck as DeckType,
    Edition,
    Enhancement,
    Joker,
    JokerName,
    PlayingCard,
    PokerHand,
    Rank,
    Seal,
    SkipTag,
    Stake,
    Suit,
    Tarot,
    TarotName,
    Voucher,
    VoucherName,
)
from game.engine.rng_sim.api import BalatroRun, validate_seed
from game.engine.rng_sim.events import (
    P_BOSS_BLINDS,
    P_CARDS_KEYS,
    P_TAGS,
    P_TAROTS,
    P_VOUCHERS,
    BossBlindPrediction,
    CertificateResult,
    IdolTarget,
    LuckyHitResult,
    RebateEvaluation,
    StoneCardResult,
    TagPrediction,
    TarotPrediction,
    VoucherPrediction,
    evaluate_lucky_hit,
    evaluate_mail_in_rebate_discard,
    get_mail_in_rebate_target,
    predict_certificate,
)
from game.engine.rng_sim.rng import (
    BalatroRNG,
    Tw223,
    advance,
    double_to_u64,
    predict_seed,
    pseudohash,
    pseudoseed,
    truncate13,
    u64_to_double,
)
from game.engine.rng_sim.state import (
    Card,
    Deck,
    GameState,
    JokerInstance,
    SimCard,
    SimDeck,
    SimGameState,
)
from game.gamestate import GameState as EngineGameState


class TestRNGExactParity(unittest.TestCase):
    """Assert bit-exact parity of PRNG math against reference fixtures."""

    def test_pseudohash_reference_vectors(self) -> None:
        """Verify pseudohash reproduces exact 64-bit IEEE-754 floats."""
        cases = [
            ("", 0x3FF0000000000000),             # 1.0
            ("A", 0x3FE4DB26AE2C1A00),            # single char
            ("TESTSEED", 0x3FD46EF4279F6C00),     # standard test seed
            ("TUTORIAL", 0x3FDABFBA38E62000),     # tutorial seed
            ("A3K9NZ2B", 0x3FEE48C7B2F08D00),
            ("7LB2WVPK", 0x3FC6A501398D4000),
            ("shuffleTESTSEED", 0x3FE3C1F649809000),
            ("Joker1sho1TESTSEED", 0x3FE50C82E208F000),
        ]
        for s, expected_u64 in cases:
            h = pseudohash(s.encode("utf-8"))
            actual_u64 = double_to_u64(h)
            self.assertEqual(
                actual_u64,
                expected_u64,
                f"pseudohash('{s}') mismatch: 0x{actual_u64:016x} != 0x{expected_u64:016x}",
            )

        nan_hash = pseudohash("erratic7LB2WVPK".encode("utf-8"))
        self.assertTrue(math.isnan(nan_hash), "Expected NaN for 'erratic7LB2WVPK'")

    def test_predict_seed_reference_vectors(self) -> None:
        """Verify predict_seed produces exact channel initial seed values."""
        cases = [
            ("erratic", "P1HNLKJI", 0x3FE1E058575684B4),
            ("boss", "D5APLKJI", 0x3FDA8FD973A59632),
            ("shuffle", "293QLKJI", 0x3FD5E3161FC17942),
            ("Joker4", "QCVQLKJI", 0x3FDB31341C86132E),
            ("erratic", "EGNRLKJI", 0x3FE282D3B590D8E1),
        ]
        for key, seed, expected_u64 in cases:
            s_val = predict_seed(key, seed)
            actual_u64 = double_to_u64(s_val)
            self.assertEqual(
                actual_u64,
                expected_u64,
                f"predict_seed('{key}', '{seed}') mismatch: 0x{actual_u64:016x} != 0x{expected_u64:016x}",
            )

    def test_stream_interleaved_operations(self) -> None:
        """Verify multi-stream step progression against reference TESTSEED sequence."""
        rng = BalatroRNG("TESTSEED")

        r1 = rng.randint("cdt1", 1, 52)
        self.assertEqual(r1, 17)

        r2 = rng.randint("rarity1sho", 1, 52)
        self.assertEqual(r2, 7)

        f3 = rng.random("lucky_mult")
        self.assertEqual(double_to_u64(f3), 0x3FD496C83C669C70)

        r4 = rng.randint("rarity1sho", 1, 52)
        self.assertEqual(r4, 41)

        r5 = rng.randint("anc1", 1, 52)
        self.assertEqual(r5, 36)

    def test_tausworthe_tw223_seeded_draws(self) -> None:
        """Verify Tw223 seeded generator matches LuaJIT math.random exactly."""
        tw = Tw223(1.0)
        expected_hex_floats = [
            0x3FD4B94FEFE0FFCC,
            0x3F8BE15720593380,
            0x3FED35050026251A,
            0x3FD31DF64372CCA8,
        ]
        for i, exp in enumerate(expected_hex_floats):
            val = tw.random()
            self.assertEqual(
                double_to_u64(val),
                exp,
                f"Tw223(1.0) float[{i}] mismatch: 0x{double_to_u64(val):016x} != 0x{exp:016x}",
            )


class TestReproducibility(unittest.TestCase):
    """Assert 100% deterministic reproducibility for seed 'HELLO' across runs."""

    def test_reproducible_hello_sequences(self) -> None:
        run1 = BalatroRun(seed="HELLO")
        run2 = BalatroRun(seed="HELLO")

        streams = ["lucky_mult", "lucky_money", "cert_fr", "wheel_of_fortune", "boss"]
        for stream in streams:
            for _ in range(20):
                v1 = run1.rng.random(stream)
                v2 = run2.rng.random(stream)
                self.assertEqual(double_to_u64(v1), double_to_u64(v2))

    def test_reproducible_event_evaluations(self) -> None:
        run1 = BalatroRun(seed="HELLO")
        run2 = BalatroRun(seed="HELLO")

        c1 = run1.predict_certificate(ante=6, add_to_deck=False)
        c2 = run2.predict_certificate(ante=6, add_to_deck=False)
        self.assertEqual(c1.card, c2.card)
        self.assertEqual(c1.front_key, c2.front_key)
        self.assertEqual(c1.seal_key, c2.seal_key)

        for _ in range(5):
            h1 = run1.evaluate_lucky_hit()
            h2 = run2.evaluate_lucky_hit()
            self.assertEqual(h1, h2)

        for _ in range(5):
            w1 = run1.evaluate_wheel_of_fortune()
            w2 = run2.evaluate_wheel_of_fortune()
            self.assertEqual(w1, w2)


class TestStepDependentAdvancement(unittest.TestCase):
    """Verify that querying outcomes accurately advances PRNG channel states."""

    def test_lucky_hit_channel_consumption(self) -> None:
        run = BalatroRun(seed="HELLO")

        self.assertEqual(run.rng.get_channel_call_count("lucky_mult"), 0)
        self.assertEqual(run.rng.get_channel_call_count("lucky_money"), 0)

        run.evaluate_lucky_hit()
        self.assertEqual(run.rng.get_channel_call_count("lucky_mult"), 1)
        self.assertEqual(run.rng.get_channel_call_count("lucky_money"), 1)

        run.evaluate_lucky_hit()
        self.assertEqual(run.rng.get_channel_call_count("lucky_mult"), 2)
        self.assertEqual(run.rng.get_channel_call_count("lucky_money"), 2)

    def test_certificate_step_advancement(self) -> None:
        run = BalatroRun(seed="HELLO")

        run.predict_certificate(ante=1, add_to_deck=False)
        self.assertEqual(run.rng.get_channel_call_count("cert_fr1"), 1)
        self.assertEqual(run.rng.get_channel_call_count("cert_sl1"), 1)

        run.predict_certificate(ante=1, add_to_deck=False)
        self.assertEqual(run.rng.get_channel_call_count("cert_fr1"), 2)
        self.assertEqual(run.rng.get_channel_call_count("cert_sl1"), 2)

        self.assertEqual(run.rng.get_channel_call_count("cert_fr2"), 0)

    def test_round_transition_advancement(self) -> None:
        run = BalatroRun(seed="HELLO")
        run.set_state(ante=1, round_num=1)
        self.assertEqual(run.state.round, 1)

        run.end_round()
        run.start_round(2)
        self.assertEqual(run.state.round, 2)
        self.assertEqual(run.state.total_rounds_played, 1)


class TestCertificateMechanics(unittest.TestCase):
    """Test Certificate joker card and seal generation."""

    def test_certificate_seed_hello_ante_6(self) -> None:
        """Seed 'HELLO' at Ante 6 generates Queen of Clubs with Purple Seal."""
        run = BalatroRun(seed="HELLO")
        run.set_state(ante=6, round_num=2)

        initial_deck_len = len(run.state.deck)
        cert_result = run.predict_certificate()

        self.assertEqual(cert_result.suit, Suit.CLUBS)
        self.assertEqual(cert_result.rank, Rank.QUEEN)
        self.assertEqual(cert_result.seal, Seal.PURPLE)
        self.assertEqual(cert_result.front_key, "C_Q")
        self.assertEqual(cert_result.seal_key, "Purple")
        self.assertIsInstance(cert_result.card, PlayingCard)

        self.assertEqual(len(run.state.deck), initial_deck_len + 1)
        self.assertIn(cert_result.card, run.state.deck)
        self.assertIn(cert_result.card, run.state.hand)

    def test_certificate_source_parity_mode(self) -> None:
        """Verify `ante_scoped=False` uses `'cert_fr'` and `'certsl'` float thresholds."""
        run = BalatroRun(seed="HELLO")
        res = run.predict_certificate(ante_scoped=False, add_to_deck=False)
        self.assertEqual(run.get_stream_call_count("cert_fr"), 1)
        self.assertEqual(run.get_stream_call_count("certsl"), 1)
        self.assertIn(res.seal, [Seal.RED, Seal.BLUE, Seal.GOLD, Seal.PURPLE])

    def test_certificate_batch_prediction_no_mutation(self) -> None:
        run = BalatroRun(seed="HELLO")
        initial_deck_size = len(run.state.deck)

        batch = run.batch_predict_certificates(antes=[1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(len(batch), 8)
        self.assertEqual(len(run.state.deck), initial_deck_size)
        ante_6_result = batch[5]
        self.assertEqual(ante_6_result.ante, 6)
        self.assertEqual(ante_6_result.rank, Rank.QUEEN)
        self.assertEqual(ante_6_result.suit, Suit.CLUBS)
        self.assertEqual(ante_6_result.seal, Seal.PURPLE)


class TestLuckyCards(unittest.TestCase):
    """Test Lucky Card probability mechanics and Oops! All 6s scaling."""

    def test_lucky_hit_return_structure(self) -> None:
        run = BalatroRun(seed="HELLO")
        payout = run.evaluate_lucky_hit()
        self.assertIsInstance(payout, dict)
        self.assertIn("mult", payout)
        self.assertIn("money", payout)
        self.assertIsInstance(payout["mult"], bool)
        self.assertIsInstance(payout["money"], bool)

    def test_lucky_hit_oops_scaling(self) -> None:
        run = BalatroRun(seed="HELLO")
        self.assertEqual(run.state.oops_count, 0)
        self.assertEqual(run.state.probability_modifier, 1.0)

        run.add_joker(JokerName.OOPS_ALL_6S)
        self.assertEqual(run.state.oops_count, 1)
        self.assertEqual(run.state.probability_modifier, 2.0)

        run.add_joker("Oops! All 6s")
        self.assertEqual(run.state.oops_count, 2)
        self.assertEqual(run.state.probability_modifier, 4.0)

        hits = [run.evaluate_lucky_hit() for _ in range(50)]
        mult_hits = sum(1 for h in hits if h["mult"])
        money_hits = sum(1 for h in hits if h["money"])
        self.assertGreater(mult_hits, 25)
        self.assertGreater(money_hits, 0)


class TestMailInRebate(unittest.TestCase):
    """Test Mail-in Rebate target rank selection and discard payout."""

    def test_mail_in_rebate_target_from_standard_deck(self) -> None:
        run = BalatroRun(seed="HELLO")
        target = run.get_mail_in_rebate_target(ante=1)
        self.assertIsInstance(target, Rank)
        self.assertEqual(target, Rank.SIX)

    def test_mail_in_rebate_custom_deck_composition(self) -> None:
        custom_deck = Deck([
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.SPADES, Rank.ACE),
            Card(Suit.CLUBS, Rank.SEVEN),
        ])
        run = BalatroRun(seed="HELLO", deck=custom_deck)
        for ante in range(1, 21):
            target = run.get_mail_in_rebate_target(ante=ante)
            self.assertIn(target, [Rank.ACE, Rank.SEVEN])

    def test_mail_in_rebate_discard_evaluation(self) -> None:
        run = BalatroRun(seed="HELLO")
        target = Rank.FOUR

        discards = [
            PlayingCard(suit=Suit.SPADES, rank=Rank.FOUR),
            PlayingCard(suit=Suit.HEARTS, rank=Rank.FOUR),
            PlayingCard(suit=Suit.CLUBS, rank=Rank.KING),
            PlayingCard(suit=Suit.DIAMONDS, rank=Rank.ACE),
        ]
        result = run.evaluate_mail_in_rebate_discard(discards, target_rank=target)
        self.assertEqual(result.matches, 2)
        self.assertEqual(result.payout, 10)
        self.assertEqual(len(result.matching_cards), 2)


class TestNewRoundTargetJokers(unittest.TestCase):
    """Test Castle, Ancient Joker, The Idol, and To Do List."""

    def test_deck_sort_id_order_matches_balatro_source(self) -> None:
        """Standard 52-card deck must match `game.lua:2367-2373` lexicographical `s..r` order."""
        deck = Deck.create_standard_52()
        keys = [c.card_key for c in deck.cards]
        self.assertEqual(keys, P_CARDS_KEYS)
        self.assertEqual([c.id for c in deck.cards], list(range(1, 53)))

    def test_castle_suit_selection_and_deck_dependency(self) -> None:
        run = BalatroRun(seed="HELLO")
        ante1_suits = run.predict_castle_suits_for_ante(ante=1, rounds=3)
        self.assertEqual(ante1_suits, [Suit.DIAMONDS, Suit.SPADES, Suit.HEARTS])

        checkered_run = BalatroRun(seed="HELLO", deck=DeckType.CHECKERED_DECK)
        for ante in range(1, 9):
            for s in checkered_run.predict_castle_suits_for_ante(ante=ante, rounds=3):
                self.assertIn(s, [Suit.SPADES, Suit.HEARTS])

        all_stone_deck = Deck([
            Card(Suit.HEARTS, Rank.KING, enhancement=Enhancement.STONE),
            Card(Suit.DIAMONDS, Rank.ACE, enhancement=Enhancement.STONE),
        ])
        stone_run = BalatroRun(seed="HELLO", deck=all_stone_deck)
        self.assertEqual(stone_run.get_castle_suit(ante=1), Suit.SPADES)

    def test_ancient_joker_suit_selection_and_non_repeat(self) -> None:
        run = BalatroRun(seed="HELLO")
        suits_by_ante = run.predict_ancient_joker_suits(antes=range(1, 9), rounds_per_ante=3)

        flat_suits = [s for a in range(1, 9) for s in suits_by_ante[a]]
        self.assertEqual(len(flat_suits), 24)
        for i in range(1, len(flat_suits)):
            self.assertNotEqual(
                flat_suits[i],
                flat_suits[i - 1],
                f"Ancient Joker repeated {flat_suits[i]} at round index {i}",
            )

        self.assertEqual(suits_by_ante[1], [Suit.SPADES, Suit.HEARTS, Suit.SPADES])
        self.assertEqual(suits_by_ante[2], [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS])

        with_skip = run.predict_ancient_joker_suits(
            antes=[1, 2], rounds_per_ante=3, skips_per_ante={1: 1}
        )
        self.assertEqual(len(with_skip[1]), 2)
        self.assertEqual(with_skip[1], [Suit.SPADES, Suit.HEARTS])
        self.assertEqual(with_skip[2][0], Suit.SPADES)

    def test_idol_card_selection_and_deck_dependency(self) -> None:
        run = BalatroRun(seed="HELLO")
        ante1_idols = run.predict_idol_cards_for_ante(ante=1, rounds=3)
        self.assertEqual(len(ante1_idols), 3)
        self.assertEqual((ante1_idols[0].rank, ante1_idols[0].suit), (Rank.TWO, Suit.CLUBS))
        self.assertEqual((ante1_idols[1].rank, ante1_idols[1].suit), (Rank.NINE, Suit.SPADES))
        self.assertEqual((ante1_idols[2].rank, ante1_idols[2].suit), (Rank.FOUR, Suit.HEARTS))

        thin_deck = Deck([
            Card(Suit.SPADES, Rank.ACE),
            Card(Suit.HEARTS, Rank.TWO, enhancement=Enhancement.STONE),
            Card(Suit.CLUBS, Rank.KING, enhancement=Enhancement.STONE),
        ])
        thin_run = BalatroRun(seed="HELLO", deck=thin_deck)
        for _ in range(5):
            target = thin_run.get_idol_card(ante=1)
            self.assertEqual(target.rank, Rank.ACE)
            self.assertEqual(target.suit, Suit.SPADES)

    def test_to_do_list_hand_selection_and_secret_hands(self) -> None:
        run = BalatroRun(seed="HELLO")
        hands = run.predict_to_do_list_hands(rounds=10)
        self.assertEqual(len(hands), 10)
        self.assertIsInstance(hands[0], PokerHand)
        self.assertEqual(hands[0], PokerHand.STRAIGHT_FLUSH)
        self.assertEqual(
            hands[:4],
            [
                PokerHand.STRAIGHT_FLUSH,
                PokerHand.HIGH_CARD,
                PokerHand.STRAIGHT,
                PokerHand.FLUSH,
            ],
        )
        for i in range(1, len(hands)):
            self.assertNotEqual(hands[i], hands[i - 1])

        hands_with_secret = run.predict_to_do_list_hands(
            rounds=4,
            secret_hand_unlocks={2: [PokerHand.FIVE_OF_A_KIND, PokerHand.FLUSH_FIVE]},
        )
        self.assertEqual(
            hands_with_secret,
            [
                PokerHand.STRAIGHT_FLUSH,
                PokerHand.HIGH_CARD,
                PokerHand.FLUSH_FIVE,
                PokerHand.PAIR,
            ],
        )

    def test_to_do_list_matches_real_telemetry_seeds(self) -> None:
        """Verify LuaJIT pairs(G.GAME.hands) iteration order matches recorded Balatro runs."""
        # Seed LX6LFV2Q: initial Buffoon/Shop To Do List spawn rolls High Card
        lx_run = BalatroRun(seed="LX6LFV2Q")
        self.assertEqual(
            lx_run.get_to_do_list_hand(is_initial_spawn=True),
            PokerHand.HIGH_CARD,
        )

        # Seed DMF1SJ5Y: two concurrent To Do List instances sharing the `'to_do'` stream
        # across secret hand unlocks (Five of a Kind unlocked at round 3).
        dmf_run = BalatroRun(seed="DMF1SJ5Y")
        j1 = dmf_run.get_to_do_list_hand(is_initial_spawn=True)  # call 1: step 0
        self.assertEqual(j1, PokerHand.FLUSH)
        j1 = dmf_run.get_to_do_list_hand(current_hand=j1)  # call 2: round 1
        self.assertEqual(j1, PokerHand.HIGH_CARD)
        j1 = dmf_run.get_to_do_list_hand(current_hand=j1)  # call 3: round 2
        self.assertEqual(j1, PokerHand.TWO_PAIR)
        dmf_run.unlock_secret_hand(PokerHand.FIVE_OF_A_KIND)
        j1 = dmf_run.get_to_do_list_hand(current_hand=j1)  # call 4: round 3
        self.assertEqual(j1, PokerHand.PAIR)
        j2 = dmf_run.get_to_do_list_hand(is_initial_spawn=True)  # call 5: second To Do List
        self.assertEqual(j2, PokerHand.TWO_PAIR)
        j1 = dmf_run.get_to_do_list_hand(current_hand=j1)  # call 6: round 4
        self.assertEqual(j1, PokerHand.THREE_OF_A_KIND)
        j2 = dmf_run.get_to_do_list_hand(current_hand=j2)  # call 7: round 4
        self.assertEqual(j2, PokerHand.THREE_OF_A_KIND)
        j1 = dmf_run.get_to_do_list_hand(current_hand=j1)  # call 8: round 5
        self.assertEqual(j1, PokerHand.TWO_PAIR)
        j2 = dmf_run.get_to_do_list_hand(current_hand=j2)  # call 9: round 5
        self.assertEqual(j2, PokerHand.FOUR_OF_A_KIND)
        j1 = dmf_run.get_to_do_list_hand(current_hand=j1)  # call 10: round 6
        self.assertEqual(j1, PokerHand.FIVE_OF_A_KIND)
        j2 = dmf_run.get_to_do_list_hand(current_hand=j2)  # call 11: round 6
        self.assertEqual(j2, PokerHand.STRAIGHT)
        j1 = dmf_run.get_to_do_list_hand(current_hand=j1)  # call 12: round 7
        self.assertEqual(j1, PokerHand.HIGH_CARD)


class TestMidAnteDeckChanges(unittest.TestCase):
    """Test how the API handles mid-ante deck mutations without throwing away n-1 rounds."""

    def test_stateful_round_by_round_deck_mutation(self) -> None:
        """Calling `get_castle_suit` / `get_idol_card` / `get_mail_in_rebate_target` round-by-round
        advances the RNG stream one step at a time as the deck changes in-place."""
        run = BalatroRun(seed="HELLO")

        # Round 1 (Small Blind): standard 52-card deck
        r1_castle = run.get_castle_suit(ante=1)
        r1_idol = run.get_idol_card(ante=1)
        r1_mail = run.get_mail_in_rebate_target(ante=1)
        self.assertEqual(r1_castle, Suit.DIAMONDS)
        self.assertEqual((r1_idol.rank, r1_idol.suit), (Rank.TWO, Suit.CLUBS))
        self.assertEqual(r1_mail, Rank.SIX)
        self.assertEqual(run.get_stream_call_count("cas1"), 1)
        self.assertEqual(run.get_stream_call_count("idol1"), 1)
        self.assertEqual(run.get_stream_call_count("mail1"), 1)

        # In Round 1 shop, player converts all Spades into Hearts!
        for c in run.state.deck.cards:
            if c.suit == Suit.SPADES:
                c.suit = Suit.HEARTS

        # Round 2 (Big Blind): call `get_castle_suit(ante=1)` directly.
        # On an unmodified 52-card deck, Round 2 would have picked `Suit.SPADES` (9 of Spades).
        # Because those Spades are now Hearts and `'cas1'` is already at call 1, Round 2
        # directly evaluates call 2 against the mutated deck and picks `Suit.HEARTS`!
        r2_castle = run.get_castle_suit(ante=1)
        r2_idol = run.get_idol_card(ante=1)
        self.assertEqual(r2_castle, Suit.HEARTS)
        self.assertEqual((r2_idol.rank, r2_idol.suit), (Rank.NINE, Suit.HEARTS))
        self.assertEqual(run.get_stream_call_count("cas1"), 2)
        self.assertEqual(run.get_stream_call_count("idol1"), 2)

    def test_direct_round_num_jump_without_evaluating_previous_rounds(self) -> None:
        """Passing `round_num=2` or `round_num=3` jumps the LCG stream directly to that round
        and evaluates only once against the current deck—no need to evaluate rounds 1..n-1."""
        # Build a deck where all Spades were converted to Hearts
        mutated_deck = Deck.create_standard_52()
        for c in mutated_deck.cards:
            if c.suit == Suit.SPADES:
                c.suit = Suit.HEARTS

        # Create a fresh run and jump straight to Round 2 (`round_num=2`) without ever calling Round 1!
        fresh_run = BalatroRun(seed="HELLO", deck=mutated_deck)
        r2_castle_direct = fresh_run.get_castle_suit(ante=1, round_num=2)
        r2_idol_direct = fresh_run.get_idol_card(ante=1, round_num=2)
        r2_mail_direct = fresh_run.get_mail_in_rebate_target(ante=1, round_num=2)

        self.assertEqual(r2_castle_direct, Suit.HEARTS)
        self.assertEqual((r2_idol_direct.rank, r2_idol_direct.suit), (Rank.NINE, Suit.HEARTS))
        self.assertEqual(fresh_run.get_stream_call_count("cas1"), 2)
        self.assertEqual(fresh_run.get_stream_call_count("idol1"), 2)
        self.assertEqual(fresh_run.get_stream_call_count("mail1"), 2)
        self.assertIsInstance(r2_mail_direct, Rank)


class TestBalatroModelsAndGameStateIntegration(unittest.TestCase):
    """Verify seamless integration with `game.balatro_models` and `game.gamestate.GameState`."""

    def test_models_type_identity(self) -> None:
        run = BalatroRun(seed="HELLO")

        # Cards in deck are instances of `game.balatro_models.PlayingCard`
        self.assertTrue(all(isinstance(c, PlayingCard) for c in run.state.deck))

        # Jokers are instances of `game.balatro_models.Joker`
        j = run.add_joker(JokerName.BLUEPRINT, edition=Edition.NEGATIVE)
        self.assertIsInstance(j, Joker)
        self.assertEqual(j.name, JokerName.BLUEPRINT)
        self.assertEqual(j.edition, Edition.NEGATIVE)

        # TarotPrediction exposes `TarotName` and `to_model() -> Tarot`
        tarot_pred = run.predict_cartomancer_tarot(ante=1)
        self.assertEqual(tarot_pred.tarot_name, TarotName.MOON)
        self.assertIsInstance(tarot_pred.to_model(), Tarot)

        # VoucherPrediction exposes `VoucherName` and `to_model() -> Voucher`
        voucher_pred = run.get_next_voucher(ante=1)
        self.assertEqual(voucher_pred.voucher_name, VoucherName.PLANET_MERCHANT)
        self.assertIsInstance(voucher_pred.to_model(), Voucher)

        # TagPrediction exposes `SkipTag` (including `SkipTag.SKIP_TAG` alias)
        tag_pred = run.get_next_tag(ante=1)
        self.assertEqual(tag_pred.skip_tag, SkipTag.VOUCHER_TAG)
        self.assertEqual(SkipTag.SKIP_TAG, SkipTag.SPEED_TAG)

        # BossBlindPrediction exposes `BlindName` (including all 28 boss blinds like THE_WINDOW and THE_MANACLE)
        boss_pred = run.get_next_boss_blind(ante=1)
        self.assertEqual(boss_pred.blind_name, BlindName.THE_GOAD)
        self.assertEqual(BlindName.THE_WINDOW.value, "The Window")
        self.assertEqual(BlindName.THE_MANACLE.value, "The Manacle")

        # Native `PlayingCard` supports `id`, `sort_id`, `card_key`, `is_suit`, and `is_rank`
        pc = PlayingCard(suit=Suit.CLUBS, rank=Rank.TEN, id=9)
        self.assertEqual(pc.id, 9)
        self.assertEqual(pc.sort_id, 9)
        self.assertEqual(pc.card_key, "C_T")
        self.assertTrue(pc.is_suit(Suit.CLUBS))
        self.assertTrue(pc.is_rank(Rank.TEN))

    def test_sync_from_engine_gamestate(self) -> None:
        gs = EngineGameState(ante=2, round=1)
        # Engine GameState creates 52 standard PlayingCards ordered [SPADES, HEARTS, CLUBS, DIAMONDS]
        # Verify `from_gamestate` normalizes their `sort_id`s to Balatro's exact 1..52 order!
        run = BalatroRun.from_gamestate(seed="HELLO", gs=gs)
        self.assertEqual(run.state.ante, 2)
        self.assertEqual([c.card_key for c in run.state.deck.cards], P_CARDS_KEYS)

        # Mutate `gs.full_deck` mid-ante (e.g., remove all Clubs) and sync into `run`
        gs.full_deck = [c for c in gs.full_deck if c.suit != Suit.CLUBS]
        run.sync_from_gamestate(gs)
        self.assertEqual(len(run.state.deck), 39)
        for _ in range(5):
            self.assertNotEqual(run.get_castle_suit(ante=2), Suit.CLUBS)


class TestCartomancerVagabondAndStoneJoker(unittest.TestCase):
    """Test Cartomancer, Vagabond, and Marble/Stone Joker card predictions."""

    def test_cartomancer_and_vagabond_tarot_predictions(self) -> None:
        run = BalatroRun(seed="HELLO")

        c1 = run.predict_cartomancer_tarot(ante=1)
        self.assertEqual(c1.key, "c_moon")
        self.assertEqual(c1.name, "The Moon")
        self.assertEqual(c1.tarot_name, TarotName.MOON)
        self.assertEqual(c1.stream_key, "Tarotcar1")
        self.assertEqual(c1.resample_count, 0)

        v1 = run.predict_vagabond_tarot(ante=1)
        self.assertEqual(v1.key, "c_empress")
        self.assertEqual(v1.name, "The Empress")
        self.assertEqual(v1.tarot_name, TarotName.EMPRESS)
        self.assertEqual(v1.stream_key, "Tarotvag1")
        self.assertEqual(v1.resample_count, 0)

        run_blocked = BalatroRun(seed="HELLO")
        c1_blocked = run_blocked.predict_cartomancer_tarot(
            ante=1, active_tarots={TarotName.MOON}
        )
        self.assertNotEqual(c1_blocked.key, "c_moon")
        self.assertGreaterEqual(c1_blocked.resample_count, 1)
        self.assertEqual(run_blocked.get_stream_call_count("Tarotcar1_resample2"), 1)

        run_showman = BalatroRun(seed="HELLO")
        run_showman.add_joker(JokerName.SHOWMAN)
        c1_showman = run_showman.predict_cartomancer_tarot(
            ante=1, active_tarots={TarotName.MOON}
        )
        self.assertEqual(c1_showman.key, "c_moon")
        self.assertEqual(c1_showman.resample_count, 0)

        all_22 = {t.key for t in P_TAROTS}
        run_all_blocked = BalatroRun(seed="HELLO")
        fallback_t = run_all_blocked.predict_cartomancer_tarot(ante=1, active_tarots=all_22)
        self.assertEqual(fallback_t.key, "c_strength")
        self.assertEqual(fallback_t.tarot_name, TarotName.STRENGTH)

    def test_stone_joker_hidden_suit_and_rank(self) -> None:
        run = BalatroRun(seed="HELLO")
        initial_len = len(run.state.deck)

        stone_res = run.predict_stone_joker_card(add_to_deck=True)
        self.assertEqual(len(run.state.deck), initial_len + 1)
        self.assertEqual(stone_res.card.enhancement, Enhancement.STONE)
        self.assertEqual(stone_res.front_key, "S_6")
        self.assertEqual(stone_res.hidden_rank, Rank.SIX)
        self.assertEqual(stone_res.hidden_suit, Suit.SPADES)
        self.assertEqual(stone_res.card.id, 53)

        self.assertFalse(stone_res.card.is_suit(Suit.SPADES))
        self.assertFalse(stone_res.card.is_rank(Rank.SIX))
        self.assertNotIn(stone_res.card, run.state.deck.get_non_stone_cards())

        run.state.deck.convert_card(stone_res.card, new_enhancement=Enhancement.MULT)
        self.assertTrue(stone_res.card.is_suit(Suit.SPADES))
        self.assertTrue(stone_res.card.is_rank(Rank.SIX))
        self.assertIn(stone_res.card, run.state.deck.get_non_stone_cards())


class TestBossBlindsTagsAndVouchers(unittest.TestCase):
    """Test Boss Blinds, Skip Tags, and Vouchers queues and edge cases."""

    def test_boss_blind_sequence_and_edge_cases(self) -> None:
        run = BalatroRun(seed="HELLO")

        bosses = run.predict_boss_blinds(max_ante=8)
        self.assertEqual(len(bosses), 8)
        for b in bosses[:7]:
            self.assertFalse(b.is_showdown)
        self.assertTrue(bosses[7].is_showdown)
        self.assertEqual(bosses[0].key, "bl_goad")
        self.assertEqual(bosses[7].key, "bl_final_leaf")
        normal_keys = [b.key for b in bosses[:7]]
        self.assertEqual(len(normal_keys), len(set(normal_keys)))

        with_reroll = run.predict_boss_blinds(max_ante=8, rerolls_at_step={1: 1})
        self.assertEqual(len(with_reroll), 9)
        self.assertEqual(with_reroll[0].key, "bl_goad")
        self.assertTrue(with_reroll[1].is_reroll)
        self.assertEqual(with_reroll[1].ante, 1)
        self.assertNotEqual(with_reroll[1].key, "bl_goad")

        with_hiero_a1 = run.predict_boss_blinds(
            max_ante=8, minus_ante_vouchers_at_step={1: 1}
        )
        self.assertEqual(len(with_hiero_a1), 9)
        self.assertEqual(with_hiero_a1[0].ante, 1)
        self.assertEqual(with_hiero_a1[1].ante, 1)
        self.assertNotEqual(with_hiero_a1[0].key, with_hiero_a1[1].key)

        hiero_a8_reroll = run.predict_boss_blinds(
            max_ante=8,
            minus_ante_vouchers_at_step={8: 1},
            rerolls_after_voucher_at_step={8: 1},
        )
        self.assertEqual(hiero_a8_reroll[7].ante, 8)
        self.assertTrue(hiero_a8_reroll[7].is_showdown)
        self.assertEqual(hiero_a8_reroll[8].ante, 7)
        self.assertTrue(hiero_a8_reroll[8].is_reroll)
        self.assertFalse(hiero_a8_reroll[8].is_showdown)
        self.assertEqual(hiero_a8_reroll[9].ante, 8)
        self.assertTrue(hiero_a8_reroll[9].is_showdown)
        self.assertNotEqual(hiero_a8_reroll[9].key, hiero_a8_reroll[7].key)

        endless_bosses = run.predict_boss_blinds(max_ante=48)
        showdowns = [b for b in endless_bosses if b.is_showdown]
        self.assertEqual(len(showdowns), 6)
        first_five_keys = {b.key for b in showdowns[:5]}
        self.assertEqual(len(first_five_keys), 5)
        self.assertEqual(showdowns[5].min_use_tier, 1)

    def test_skip_tag_sequence_and_edge_cases(self) -> None:
        run = BalatroRun(seed="HELLO")
        tags_per_ante = run.predict_skip_tags(max_ante=8, include_orbital_hands=True)
        self.assertEqual(len(tags_per_ante), 8)

        min_ante_2_keys = {t.key for t in P_TAGS if t.min_ante == 2}
        for t in tags_per_ante[0]:
            self.assertNotIn(t.key, min_ante_2_keys)
        self.assertEqual(tags_per_ante[0][0].key, "tag_voucher")
        self.assertEqual(tags_per_ante[0][1].key, "tag_voucher")

        tags_with_hiero_a1 = run.predict_skip_tags(
            max_ante=2, minus_ante_vouchers_at_step={1: 1}
        )
        self.assertEqual(len(tags_with_hiero_a1), 3)
        for pair in tags_with_hiero_a1[:2]:
            for t in pair:
                self.assertEqual(t.ante, 1)
                self.assertNotIn(t.key, min_ante_2_keys)

        tags_with_hiero = run.predict_skip_tags(
            max_ante=3, minus_ante_vouchers_at_step={2: 1}
        )
        self.assertEqual(len(tags_with_hiero), 4)
        self.assertEqual(tags_with_hiero[1][0].ante, 2)
        self.assertEqual(tags_with_hiero[2][0].ante, 2)
        self.assertNotEqual(
            (tags_with_hiero[1][0].key, tags_with_hiero[1][1].key),
            (tags_with_hiero[2][0].key, tags_with_hiero[2][1].key),
        )

    def test_voucher_sequence_purchase_adaptation_and_voucher_tag_global_queue(self) -> None:
        run = BalatroRun(seed="HELLO")

        unbought_seq = run.predict_voucher_sequence(max_ante=8, purchase_plan=False)
        tier2_keys = {v.key for v in P_VOUCHERS if v.requires}
        for v in unbought_seq:
            self.assertNotIn(v.key, tier2_keys)
        self.assertEqual(unbought_seq[1].key, "v_grabber")
        self.assertEqual(unbought_seq[4].key, "v_grabber")

        bought_seq = run.predict_voucher_sequence(max_ante=8, purchase_plan=True)
        self.assertEqual(bought_seq[1].key, "v_grabber")
        self.assertEqual(bought_seq[4].key, "v_wasteful")
        self.assertGreaterEqual(bought_seq[4].resample_count, 1)
        bought_keys = [v.key for v in bought_seq]
        self.assertEqual(len(bought_keys), len(set(bought_keys)))

        seq_two_in_a1 = run.predict_voucher_sequence(
            max_ante=2, purchase_plan=False, voucher_tags_at_step={1: 2}
        )
        tag_vouchers_a1 = [v for v in seq_two_in_a1 if v.from_tag]
        self.assertEqual(len(tag_vouchers_a1), 2)
        self.assertEqual(tag_vouchers_a1[0].stream_key, "Voucher_fromtag")
        self.assertEqual(tag_vouchers_a1[1].stream_key, "Voucher_fromtag")

        seq_split_a1_a2 = run.predict_voucher_sequence(
            max_ante=2, purchase_plan=False, voucher_tags_at_step={1: 1, 2: 1}
        )
        tag_vouchers_split = [v for v in seq_split_a1_a2 if v.from_tag]
        self.assertEqual(tag_vouchers_a1[0].key, tag_vouchers_split[0].key)
        self.assertEqual(tag_vouchers_a1[1].key, tag_vouchers_split[1].key)

        zodiac_run = BalatroRun(seed="HELLO", deck=DeckType.ZODIAC_DECK)
        self.assertIn("v_overstock_norm", zodiac_run.state.used_vouchers)
        self.assertIn("v_tarot_merchant", zodiac_run.state.used_vouchers)
        self.assertIn("v_planet_merchant", zodiac_run.state.used_vouchers)
        zodiac_v1 = zodiac_run.get_next_voucher(ante=1)
        self.assertNotEqual(zodiac_v1.key, "v_planet_merchant")
        self.assertGreaterEqual(zodiac_v1.resample_count, 1)


class TestEventResolvers(unittest.TestCase):
    """Test 1-in-N outcome resolutions for probability jokers and cards."""

    def test_outcome_resolutions(self) -> None:
        run = BalatroRun(seed="HELLO")

        w = run.evaluate_wheel_of_fortune()
        self.assertIsInstance(w, bool)
        self.assertEqual(run.rng.get_channel_call_count("wheel_of_fortune"), 1)

        s = run.evaluate_space_joker()
        self.assertIsInstance(s, bool)
        self.assertEqual(run.rng.get_channel_call_count("space_joker"), 1)

        b = run.evaluate_bloodstone()
        self.assertIsInstance(b, bool)
        self.assertEqual(run.rng.get_channel_call_count("bloodstone"), 1)

        g = run.evaluate_glass_shatter()
        self.assertIsInstance(g, bool)
        self.assertEqual(run.rng.get_channel_call_count("glass"), 1)

        bc = run.evaluate_business_card()
        self.assertIsInstance(bc, bool)
        self.assertEqual(run.rng.get_channel_call_count("business_card"), 1)

    def test_oops_modifier_scaling(self) -> None:
        run = BalatroRun(seed="HELLO")
        for _ in range(3):
            run.add_joker(JokerName.OOPS_ALL_6S)
        self.assertEqual(run.state.probability_modifier, 8.0)

        self.assertTrue(run.evaluate_wheel_of_fortune())
        self.assertTrue(run.evaluate_space_joker())
        self.assertTrue(run.evaluate_bloodstone())


class TestDeckAndStateMutations(unittest.TestCase):
    """Test GameState transitions and deck card mutations."""

    def test_deck_mutations(self) -> None:
        deck = Deck.create_standard_52()
        self.assertEqual(len(deck), 52)

        custom_card = Card(
            suit=Suit.HEARTS,
            rank=Rank.ACE,
            enhancement=Enhancement.STEEL,
            edition=Edition.FOIL,
            seal=Seal.RED,
        )
        deck.add_card(custom_card)
        self.assertEqual(len(deck), 53)
        self.assertIn(custom_card, deck)

        destroyed = deck.destroy_card(custom_card)
        self.assertTrue(destroyed)
        self.assertEqual(len(deck), 52)
        self.assertNotIn(custom_card, deck)

        target_card = deck[0]
        converted = deck.convert_card(
            target_card,
            new_suit=Suit.SPADES,
            new_rank=Rank.KING,
            new_enhancement=Enhancement.GLASS,
            new_edition=Edition.POLYCHROME,
            new_seal=Seal.GOLD,
        )
        self.assertEqual(converted.suit, Suit.SPADES)
        self.assertEqual(converted.rank, Rank.KING)
        self.assertEqual(converted.enhancement, Enhancement.GLASS)
        self.assertEqual(converted.edition, Edition.POLYCHROME)
        self.assertEqual(converted.seal, Seal.GOLD)

    def test_joker_management(self) -> None:
        run = BalatroRun(seed="HELLO")
        run.add_joker(JokerName.BLUEPRINT)
        run.add_joker(JokerName.BRAINSTORM)
        self.assertTrue(run.has_joker(JokerName.BLUEPRINT))
        self.assertTrue(run.has_joker("Brainstorm"))
        self.assertEqual(len(run.state.owned_jokers), 2)

        removed = run.remove_joker(JokerName.BLUEPRINT)
        self.assertIsNotNone(removed)
        self.assertFalse(run.has_joker("Blueprint"))
        self.assertEqual(len(run.state.owned_jokers), 1)


class TestCheckpointAndFork(unittest.TestCase):
    """Test state checkpointing, restoring, and forking."""

    def test_checkpoint_and_restore(self) -> None:
        run = BalatroRun(seed="HELLO")
        run.set_state(ante=3, round_num=1)
        run.add_joker(JokerName.JOKER)

        cp = run.checkpoint()

        run.set_state(ante=7, round_num=4)
        run.predict_certificate()
        run.evaluate_lucky_hit()
        self.assertEqual(run.state.ante, 7)

        run.restore(cp)
        self.assertEqual(run.state.ante, 3)
        self.assertEqual(run.state.round, 1)
        self.assertTrue(run.has_joker(JokerName.JOKER))
        self.assertEqual(run.rng.get_channel_call_count("cert_fr7"), 0)

    def test_fork(self) -> None:
        run = BalatroRun(seed="HELLO")
        run.set_state(ante=4, round_num=2)

        forked = run.fork()
        self.assertEqual(forked.state.ante, 4)

        forked.set_state(ante=8)
        forked.predict_certificate()

        self.assertEqual(run.state.ante, 4)
        self.assertEqual(run.rng.get_channel_call_count("cert_fr8"), 0)


class TestPublicAPIAndValidation(unittest.TestCase):
    """Test top-level public API usage and seed validation."""

    def test_seed_validation(self) -> None:
        ok, clean = validate_seed("hello")
        self.assertTrue(ok)
        self.assertEqual(clean, "HELLO")

        ok_zero, _ = validate_seed("HELL0")
        self.assertFalse(ok_zero)

        ok_long, _ = validate_seed("ABCDEFGHI")
        self.assertFalse(ok_long)

        with self.assertRaises(ValueError):
            BalatroRun(seed="BAD_0_SEED")

    def test_prompt_api_snippet(self) -> None:
        run = BalatroRun(seed="HELLO")
        run.set_state(ante=6, round_num=2)

        cert_result = run.predict_certificate()
        self.assertEqual(cert_result.rank, Rank.QUEEN)
        self.assertEqual(cert_result.suit, Suit.CLUBS)
        self.assertEqual(cert_result.seal, Seal.PURPLE)

        lucky_payout = run.evaluate_lucky_hit()
        self.assertIn("mult", lucky_payout)
        self.assertIn("money", lucky_payout)

        rebate_target = run.get_mail_in_rebate_target()
        self.assertIsInstance(rebate_target, Rank)


if __name__ == "__main__":
    unittest.main(verbosity=2)
