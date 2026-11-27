"""
Unit tests for cv_detect_hand_cards, cv_detect_shop_cards, and PRNG Boss Blind resolution.
Validates location restrictions, domain model conversions, and deterministic boss predictions.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

import game.balatro_models as bm
import game.gamestate as gs
import game.engine.cv_stubs as cv_stubs
from game.engine.rules import on_action


class TestCVStubsLocationChecks(unittest.TestCase):
    """Verifies that cv_detect_hand_cards and cv_detect_shop_cards raise InvalidLocationError
    when gamestate is in the wrong location."""

    def setUp(self):
        self.state = gs.GameState(
            seed="TESTSEED",
            ante=1,
            round=1,
            location=bm.Location.BLIND,
        )
        # Dummy frame image (1080p BGR dummy)
        self.dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # -------------------------------------------------------------
    # cv_detect_hand_cards Location Tests
    # -------------------------------------------------------------

    def test_hand_cards_in_shop_raises(self):
        """Should raise InvalidLocationError if in shop outside a booster pack."""
        self.state.location = bm.Location.SHOP
        self.state.active_pack = None

        with self.assertRaises(cv_stubs.InvalidLocationError) as ctx:
            cv_stubs.cv_detect_hand_cards(self.state, self.dummy_frame)
        self.assertIn("shop", str(ctx.exception).lower())

    def test_hand_cards_selecting_blind_raises(self):
        """Should raise InvalidLocationError if selecting a blind."""
        self.state.location = bm.Location.SELECT

        with self.assertRaises(cv_stubs.InvalidLocationError) as ctx:
            cv_stubs.cv_detect_hand_cards(self.state, self.dummy_frame)
        self.assertIn("select", str(ctx.exception).lower())

    def test_hand_cards_in_celestial_pack_raises(self):
        """Should raise InvalidLocationError in Celestial packs where hand cards are not visible."""
        self.state.location = bm.Location.PACK
        self.state.active_pack = bm.BoosterPack(
            booster_type=bm.BoosterType.CELESTIAL,
            size=bm.BoosterSize.REGULAR,
            cost=4,
            choose=1,
            extra=3,
        )

        with self.assertRaises(cv_stubs.InvalidLocationError) as ctx:
            cv_stubs.cv_detect_hand_cards(self.state, self.dummy_frame)
        self.assertIn("celestial", str(ctx.exception).lower())

    def test_hand_cards_in_buffoon_pack_raises(self):
        """Should raise InvalidLocationError in Buffoon packs (jokers only)."""
        self.state.location = bm.Location.PACK
        self.state.active_pack = bm.BoosterPack(
            booster_type=bm.BoosterType.BUFFOON,
            size=bm.BoosterSize.REGULAR,
            cost=6,
            choose=1,
            extra=2,
        )

        with self.assertRaises(cv_stubs.InvalidLocationError):
            cv_stubs.cv_detect_hand_cards(self.state, self.dummy_frame)

    def test_hand_cards_in_standard_pack_raises(self):
        """Should raise InvalidLocationError in Standard packs (playing cards to add)."""
        self.state.location = bm.Location.PACK
        self.state.active_pack = bm.BoosterPack(
            booster_type=bm.BoosterType.STANDARD,
            size=bm.BoosterSize.REGULAR,
            cost=4,
            choose=1,
            extra=3,
        )

        with self.assertRaises(cv_stubs.InvalidLocationError):
            cv_stubs.cv_detect_hand_cards(self.state, self.dummy_frame)

    def test_hand_cards_in_cash_out_screen_raises(self):
        """Should raise InvalidLocationError if round is won / cash out screen."""
        self.state.location = bm.Location.BLIND
        self.state.round_won = True

        with self.assertRaises(cv_stubs.InvalidLocationError) as ctx:
            cv_stubs.cv_detect_hand_cards(self.state, self.dummy_frame)
        self.assertIn("cash out", str(ctx.exception).lower())

    def test_hand_cards_in_blind_valid(self):
        """Should succeed during active blind play."""
        self.state.location = bm.Location.BLIND
        self.state.round_won = False

        mock_card_result = MagicMock()
        mock_card_result.rank = "Ace"
        mock_card_result.suit = "Spades"
        mock_card_result.edition = "Foil"
        mock_card_result.enhancement = "Mult"
        mock_card_result.seal = "Red"
        mock_card_result.selected = True
        mock_card_result.debuffed = False
        mock_card_result.face_down = False

        mock_out = MagicMock()
        mock_out.cards = [mock_card_result]

        with patch("classifiers.inference.recognize_hand_cards", return_value=mock_out):
            hand = cv_stubs.cv_detect_hand_cards(self.state, self.dummy_frame)

        self.assertEqual(len(hand), 1)
        self.assertIsInstance(hand[0], bm.HandCard)
        self.assertEqual(hand[0].card.rank, bm.Rank.ACE)
        self.assertEqual(hand[0].card.suit, bm.Suit.SPADES)
        self.assertEqual(hand[0].card.edition, bm.Edition.FOIL)
        self.assertEqual(hand[0].card.enhancement, bm.Enhancement.MULT)
        self.assertEqual(hand[0].card.seal, bm.Seal.RED)
        self.assertTrue(hand[0].selected)
        self.assertFalse(hand[0].facing_back)

    def test_hand_cards_in_arcana_pack_valid(self):
        """Should succeed when inside an Arcana or Spectral pack where hand cards are visible."""
        self.state.location = bm.Location.PACK
        self.state.active_pack = bm.BoosterPack(
            booster_type=bm.BoosterType.ARCANA,
            size=bm.BoosterSize.MEGA,
            cost=8,
            choose=2,
            extra=5,
        )

        mock_card = MagicMock(
            rank="Stone",
            suit="none",
            edition="None",
            enhancement="Stone",
            seal="None",
            selected=False,
            debuffed=False,
            face_down=False,
        )
        mock_out = MagicMock(cards=[mock_card])

        with patch("classifiers.inference.recognize_hand_cards", return_value=mock_out):
            hand = cv_stubs.cv_detect_hand_cards(self.state, self.dummy_frame)

        self.assertEqual(len(hand), 1)
        self.assertEqual(hand[0].card.enhancement, bm.Enhancement.STONE)
        self.assertEqual(hand[0].card.rank, bm.Rank.NONE)

    def test_hand_cards_none_frame_raises(self):
        """Should raise ValueError if location is valid but frame is None."""
        self.state.location = bm.Location.BLIND
        with self.assertRaises(ValueError):
            cv_stubs.cv_detect_hand_cards(self.state, frame_or_state=None)

    # -------------------------------------------------------------
    # cv_detect_shop_cards Location Tests
    # -------------------------------------------------------------

    def test_shop_cards_selecting_blind_raises(self):
        """Should raise InvalidLocationError if selecting a blind."""
        self.state.location = bm.Location.SELECT

        with self.assertRaises(cv_stubs.InvalidLocationError) as ctx:
            cv_stubs.cv_detect_shop_cards(self.state, self.dummy_frame)
        self.assertIn("select", str(ctx.exception).lower())

    def test_shop_cards_in_blind_raises(self):
        """Should raise InvalidLocationError if in a blind."""
        self.state.location = bm.Location.BLIND

        with self.assertRaises(cv_stubs.InvalidLocationError) as ctx:
            cv_stubs.cv_detect_shop_cards(self.state, self.dummy_frame)
        self.assertIn("blind", str(ctx.exception).lower())

    def test_shop_cards_inside_pack_in_shop_raises(self):
        """Should raise InvalidLocationError if currently inside a pack in the shop."""
        self.state.location = bm.Location.PACK
        self.state.active_pack = bm.BoosterPack(
            booster_type=bm.BoosterType.BUFFOON,
            size=bm.BoosterSize.REGULAR,
            cost=6,
            choose=1,
            extra=2,
        )

        with self.assertRaises(cv_stubs.InvalidLocationError) as ctx:
            cv_stubs.cv_detect_shop_cards(self.state, self.dummy_frame)
        self.assertIn("pack", str(ctx.exception).lower())

    def test_shop_cards_inside_pack_from_skip_raises(self):
        """Should raise InvalidLocationError inside a skip tag booster pack."""
        self.state.location = bm.Location.PACK
        self.state.active_pack = bm.BoosterPack(
            booster_type=bm.BoosterType.STANDARD,
            size=bm.BoosterSize.MEGA,
            cost=0,
            choose=2,
            extra=5,
        )

        with self.assertRaises(cv_stubs.InvalidLocationError):
            cv_stubs.cv_detect_shop_cards(self.state, self.dummy_frame)

    def test_shop_cards_cash_out_screen_raises(self):
        """Should raise InvalidLocationError on cash out screen."""
        self.state.location = bm.Location.SHOP
        self.state.round_won = True

        with self.assertRaises(cv_stubs.InvalidLocationError) as ctx:
            cv_stubs.cv_detect_shop_cards(self.state, self.dummy_frame)
        self.assertIn("cash out", str(ctx.exception).lower())

    def test_shop_cards_in_main_shop_valid(self):
        """Should succeed in main shop and return ShopCardsResult with all shelves."""
        self.state.location = bm.Location.SHOP
        self.state.active_pack = None
        self.state.round_won = False

        # Mock shop recognition output
        joker_item = MagicMock(
            key="j_blueprint", card_type="Joker", edition="Holo",
            eternal=True, perishable=False, rental=False, debuffed=False,
        )
        cons_item = MagicMock(
            key="c_fool", card_type="Tarot", edition="Negative",
            eternal=False, perishable=False, rental=False, debuffed=False,
        )
        voucher_item = MagicMock(
            key="v_directors_cut", card_type="Voucher", edition="None",
            eternal=False, perishable=False, rental=False, debuffed=False,
        )
        pack_item = MagicMock(
            key="p_arcana_mega_1", card_type="Booster", edition="None",
            eternal=False, perishable=False, rental=False, debuffed=False,
        )

        mock_out = MagicMock()
        mock_out.shop_items = [joker_item, cons_item]
        mock_out.vouchers = [voucher_item]
        mock_out.boosters = [pack_item]
        mock_out.joker_bar = [joker_item]
        mock_out.consumeables = [cons_item]

        with patch("classifiers.inference.recognize_shop", return_value=mock_out):
            shop_res = cv_stubs.cv_detect_shop_cards(self.state, self.dummy_frame)

        self.assertIsInstance(shop_res, cv_stubs.ShopCardsResult)
        self.assertIsInstance(shop_res, list)
        self.assertEqual(len(shop_res), 2)

        # Shelf cards
        self.assertIsInstance(shop_res[0], bm.Joker)
        self.assertEqual(shop_res[0].name, bm.JokerName.BLUEPRINT)
        self.assertEqual(shop_res[0].edition, bm.Edition.HOLOGRAPHIC)
        self.assertTrue(shop_res[0].eternal)

        self.assertIsInstance(shop_res[1], bm.Tarot)
        self.assertEqual(shop_res[1].name, bm.TarotName.FOOL)
        self.assertEqual(shop_res[1].edition, bm.Edition.NEGATIVE)

        # Voucher shelf
        self.assertEqual(len(shop_res.vouchers), 1)
        self.assertEqual(shop_res.vouchers[0].name, bm.VoucherName.DIRECTORS_CUT)

        # Booster shelf
        self.assertEqual(len(shop_res.packs), 1)
        self.assertEqual(shop_res.packs[0].booster_type, bm.BoosterType.ARCANA)
        self.assertEqual(shop_res.packs[0].size, bm.BoosterSize.MEGA)
        self.assertEqual(shop_res.packs[0].choose, 2)

    # -------------------------------------------------------------
    # cv_detect_pack_cards Tests
    # -------------------------------------------------------------

    def test_pack_cards_in_blind_raises(self):
        self.state.location = bm.Location.BLIND
        with self.assertRaises(cv_stubs.InvalidLocationError):
            cv_stubs.cv_detect_pack_cards(self.state, self.dummy_frame)

    def test_pack_cards_in_select_raises(self):
        self.state.location = bm.Location.SELECT
        with self.assertRaises(cv_stubs.InvalidLocationError):
            cv_stubs.cv_detect_pack_cards(self.state, self.dummy_frame)

    def test_pack_cards_in_main_shop_without_pack_raises(self):
        self.state.location = bm.Location.SHOP
        self.state.active_pack = None
        with self.assertRaises(cv_stubs.InvalidLocationError):
            cv_stubs.cv_detect_pack_cards(self.state, self.dummy_frame)

    def test_pack_cards_in_cash_out_raises(self):
        self.state.location = bm.Location.PACK
        self.state.round_won = True
        with self.assertRaises(cv_stubs.InvalidLocationError):
            cv_stubs.cv_detect_pack_cards(self.state, self.dummy_frame)

    def test_pack_cards_none_frame_raises(self):
        self.state.location = bm.Location.PACK
        with self.assertRaises(ValueError):
            cv_stubs.cv_detect_pack_cards(self.state, None)

    def test_pack_cards_in_pack_valid(self):
        self.state.location = bm.Location.PACK
        self.state.active_pack = bm.BoosterPack(
            booster_type=bm.BoosterType.STANDARD,
            size=bm.BoosterSize.REGULAR,
            cost=4,
            choose=1,
            extra=3,
        )
        tarot_item = MagicMock(
            key="c_hermit", card_type="Tarot", edition="None",
            eternal=False, perishable=False, rental=False, debuffed=False,
        )
        card_item = MagicMock(
            key="", card_type="PlayingCard", edition="Foil",
            eternal=False, perishable=False, rental=False, debuffed=False,
            card_details=MagicMock(
                rank="King", suit="Hearts", enhancement="Glass",
                edition="Foil", seal="Gold", debuffed=False,
            ),
        )
        mock_out = MagicMock(pack_items=[tarot_item, card_item])

        with patch("classifiers.inference.recognize_shop", return_value=mock_out) as mock_rec:
            # Verify (frame, state) argument order works too
            pack_res = cv_stubs.cv_detect_pack_cards(self.dummy_frame, self.state)

        mock_rec.assert_called_once()
        self.assertEqual(mock_rec.call_args.kwargs.get("mode"), "pack_only")
        self.assertEqual(len(pack_res), 2)
        self.assertIsInstance(pack_res[0], bm.Tarot)
        self.assertEqual(pack_res[0].name, bm.TarotName.HERMIT)
        self.assertIsInstance(pack_res[1], bm.PlayingCard)
        self.assertEqual(pack_res[1].rank, bm.Rank.KING)
        self.assertEqual(pack_res[1].suit, bm.Suit.HEARTS)
        self.assertEqual(pack_res[1].enhancement, bm.Enhancement.GLASS)
        self.assertEqual(pack_res[1].seal, bm.Seal.GOLD)

    # -------------------------------------------------------------
    # cv_detect_joker_bar Tests
    # -------------------------------------------------------------

    def test_joker_bar_invalid_location_raises(self):
        self.state.location = "menu"
        with self.assertRaises(cv_stubs.InvalidLocationError):
            cv_stubs.cv_detect_joker_bar(self.state, self.dummy_frame)

    def test_joker_bar_none_frame_raises(self):
        self.state.location = bm.Location.BLIND
        with self.assertRaises(ValueError):
            cv_stubs.cv_detect_joker_bar(self.state, None)

    def test_joker_bar_valid_across_gameplay_screens(self):
        joker_item = MagicMock(
            key="j_brainstorm", card_type="Joker", edition="Polychrome",
            eternal=False, perishable=True, rental=True, debuffed=False,
        )
        stray_item = MagicMock(
            key="c_pluto", card_type="Planet", edition="None",
            eternal=False, perishable=False, rental=False, debuffed=False,
        )
        mock_out = MagicMock(joker_bar=[joker_item, stray_item])

        for loc, round_won in [
            (bm.Location.SELECT, False),
            (bm.Location.BLIND, False),
            (bm.Location.BLIND, True),
            (bm.Location.SHOP, False),
            (bm.Location.PACK, False),
        ]:
            self.state.location = loc
            self.state.round_won = round_won
            with patch("classifiers.inference.recognize_shop", return_value=mock_out) as mock_rec:
                jokers = cv_stubs.cv_detect_joker_bar(self.state, self.dummy_frame)
            self.assertEqual(mock_rec.call_args.kwargs.get("mode"), "joker_only")
            self.assertEqual(len(jokers), 1)
            self.assertIsInstance(jokers[0], bm.Joker)
            self.assertEqual(jokers[0].name, bm.JokerName.BRAINSTORM)
            self.assertEqual(jokers[0].edition, bm.Edition.POLYCHROME)
            self.assertTrue(jokers[0].perishable)
            self.assertTrue(jokers[0].rental)


class TestPRNGBossBlind(unittest.TestCase):
    """Verifies that cv_detect_boss_blind is removed and PRNG boss resolution works correctly."""

    def test_cv_detect_boss_blind_removed(self):
        """cv_detect_boss_blind must be removed from cv_stubs."""
        self.assertFalse(hasattr(cv_stubs, "cv_detect_boss_blind"))

    def test_get_boss_blind_from_seed(self):
        """get_boss_blind_from_seed deterministically rolls boss blinds matching game source."""
        # For seed DMF1SJ5Y, Ante 1 Boss is The Goad
        boss_ante1 = cv_stubs.get_boss_blind_from_seed("DMF1SJ5Y", ante=1)
        self.assertEqual(boss_ante1, bm.BlindName.THE_GOAD)

    def test_state_seed_initializes_boss_blind(self):
        """Setting seed on GameState should roll the Boss Blind for current ante."""
        state = gs.GameState(seed="DMF1SJ5Y", ante=1)
        self.assertEqual(state.boss_blind, bm.BlindName.THE_GOAD)

    def test_reroll_boss_action_uses_prng(self):
        """ActionType.REROLL_BOSS should roll a new boss blind via PRNG without pending CV checks."""
        state = gs.GameState(seed="DMF1SJ5Y", ante=1, money=20)
        initial_boss = state.boss_blind
        self.assertEqual(initial_boss, bm.BlindName.THE_GOAD)

        act = bm.Action(type=bm.ActionType.REROLL_BOSS)
        on_action(state, act)

        self.assertEqual(state.money, 10)
        self.assertNotIn("boss_blind", state.pending_cv_checks)
        self.assertIsNotNone(state.boss_blind)
        self.assertNotEqual(state.boss_blind, initial_boss)


class TestRealImageCVInference(unittest.TestCase):
    """End-to-end integration tests running real neural network inference on sample screenshot images."""

    def test_real_image_hand_detection(self):
        """Loads assets/sample_inputs/in_blind.jpg and runs actual card detector & classifier."""
        import cv2
        from pathlib import Path

        img_path = Path("assets/sample_inputs/in_blind.jpg")
        if not img_path.exists():
            self.skipTest(f"Sample image not found: {img_path}")

        img = cv2.imread(str(img_path))
        state = gs.GameState(location=bm.Location.BLIND)

        cards = cv_stubs.cv_detect_hand_cards(state, img)

        self.assertIsInstance(cards, list)
        self.assertGreaterEqual(len(cards), 8, "Expected at least a standard 8-card hand to be detected.")
        for hc in cards:
            self.assertIsInstance(hc, bm.HandCard)
            self.assertIsInstance(hc.card, bm.PlayingCard)
            self.assertNotEqual(hc.card.rank, bm.Rank.NONE)
            self.assertNotEqual(hc.card.suit, bm.Suit.NONE)

    def test_real_image_shop_detection(self):
        """Loads assets/sample_inputs/shop.jpg and runs actual shop detector & classifier."""
        import cv2
        from pathlib import Path

        img_path = Path("assets/sample_inputs/shop.jpg")
        if not img_path.exists():
            self.skipTest(f"Sample image not found: {img_path}")

        img = cv2.imread(str(img_path))
        state = gs.GameState(location=bm.Location.SHOP)

        res = cv_stubs.cv_detect_shop_cards(state, img)

        self.assertIsInstance(res, cv_stubs.ShopCardsResult)
        self.assertIsInstance(res, list)
        self.assertGreaterEqual(len(res), 1, "Expected at least 1 shop card to be detected.")
        for item in res:
            self.assertIsInstance(item, (bm.Card, bm.Joker, bm.Tarot, bm.Planet, bm.Spectral, bm.PlayingCard))

    def test_shop_classifier_masks_invalid_keys_before_softmax(self):
        """Verifies predict_attributes masks zero-sample INVALID_SHOP_KEYS logits to -inf before softmax."""
        import torch
        from classifiers.shop_classifier.stage2_classifier.config import INVALID_SHOP_KEYS, KEY_TO_IDX
        from classifiers.shop_classifier.stage2_classifier.model import MultiHeadItemClassifier

        model = MultiHeadItemClassifier(backbone_name="mobilenet_v3_small", pretrained=False)
        num_keys = model.head_key.out_features
        fake_logits = {
            "key": torch.full((1, num_keys), -50.0),
            "edition": torch.zeros((1, 5)),
            "eternal": torch.zeros((1,)),
            "perishable": torch.zeros((1,)),
            "rental": torch.zeros((1,)),
            "debuffed": torch.zeros((1,)),
        }
        # Force every masked invalid class to have a massive raw logit vs a modest valid key logit
        for invalid_k in INVALID_SHOP_KEYS:
            fake_logits["key"][0, KEY_TO_IDX[invalid_k]] = 100.0
        fake_logits["key"][0, KEY_TO_IDX["j_blueprint"]] = 1.0

        dummy_input = torch.zeros((1, 3, 192, 128))
        with patch.object(model, "forward", return_value=fake_logits):
            preds = model.predict_attributes(dummy_input)

        self.assertEqual(len(preds), 1)
        self.assertEqual(preds[0]["key"], "j_blueprint")
        self.assertNotIn(preds[0]["key"], INVALID_SHOP_KEYS)


if __name__ == "__main__":
    unittest.main()
