"""Unit tests for coarse screen-state template gating and Windows path handling."""

import json
from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from game.balatro_models import BlindName, BoosterPack, BoosterSize, BoosterType, Deck, Location
from game.gamestate import GameState
import main as main_module
from screen_state import (
    ALLOWED_TEMPLATES_BY_STATE,
    CoarseScreenState,
    ScreenStateTracker,
    is_action_clickable_in_gamestate,
    validate_templates_covered,
)


class TestScreenStateTracker(unittest.TestCase):
    def test_all_registered_templates_covered(self):
        templates_path = REPO_ROOT / "assets" / "templates" / "drspectred" / "templates.json"
        with open(templates_path, "r", encoding="utf-8") as f:
            registered = list(json.load(f).keys())
        self.assertEqual(validate_templates_covered(registered), [])

    def test_pre_game_to_blind_to_cash_out_to_shop(self):
        tracker = ScreenStateTracker()
        self.assertEqual(tracker.state, CoarseScreenState.PRE_GAME)
        pre_allowed = tracker.allowed_template_names()
        self.assertIn("Play Game", pre_allowed)
        self.assertIn("Select Blind", pre_allowed)
        self.assertIn("Skip Blind", pre_allowed)
        self.assertNotIn("Play Hand", pre_allowed)
        self.assertNotIn("Buy Card", pre_allowed)
        self.assertNotIn("Use In Pack", pre_allowed)

        # Main menu PLAY -> Run setup PLAY -> Blind Select
        self.assertEqual(tracker.on_template_clicked("Play Game"), CoarseScreenState.SELECT)
        self.assertIn("Play Game", tracker.allowed_template_names())
        self.assertEqual(tracker.on_template_clicked("Play Game"), CoarseScreenState.SELECT)
        self.assertIn("Select Blind", tracker.allowed_template_names())

        # Select Blind -> Blind (Play/Discard/Sort AND Cash Out for implicit round win!)
        self.assertEqual(tracker.on_template_clicked("Select Blind"), CoarseScreenState.BLIND)
        blind_allowed = tracker.allowed_template_names()
        self.assertNotIn("Play Game", blind_allowed)
        self.assertNotIn("Select Blind", blind_allowed)
        self.assertNotIn("Use In Pack", blind_allowed)
        for btn in ("Play Hand", "Discard", "Sort by Rank", "Sort by Suit", "Cash Out", "Sell Card", "Use Consumable"):
            self.assertIn(btn, blind_allowed)

        # Cash Out -> Shop
        self.assertEqual(tracker.on_template_clicked("Cash Out"), CoarseScreenState.SHOP)
        shop_allowed = tracker.allowed_template_names()
        self.assertIn("Buy Card", shop_allowed)
        self.assertIn("Open Pack", shop_allowed)
        self.assertIn("Next Round", shop_allowed)
        self.assertNotIn("Play Hand", shop_allowed)
        self.assertNotIn("Use In Pack", shop_allowed)

    def test_shop_pack_and_mega_pack_hybrid_states(self):
        tracker = ScreenStateTracker()
        tracker.on_template_clicked("Select Blind")
        tracker.on_template_clicked("Cash Out")

        # Open Pack from Shop -> PACK_OR_SHOP hybrid state so missed pack exit clicks don't deadlock SHOP
        self.assertEqual(tracker.on_template_clicked("Open Pack"), CoarseScreenState.PACK_OR_SHOP)
        pack_allowed = tracker.allowed_template_names()
        self.assertEqual(pack_allowed, ALLOWED_TEMPLATES_BY_STATE[CoarseScreenState.PACK_OR_SHOP])
        self.assertIn("Use In Pack", pack_allowed)
        self.assertIn("Select Card", pack_allowed)
        self.assertIn("Skip Pack", pack_allowed)
        self.assertNotIn("Play Hand", pack_allowed)
        self.assertNotIn("Cash Out", pack_allowed)
        self.assertNotIn("Select Blind", pack_allowed)

        # Choosing a card in pack enters PACK_OR_SHOP (supports both choose=1 exit & choose=2 Mega!)
        self.assertEqual(tracker.on_template_clicked("Use In Pack"), CoarseScreenState.PACK_OR_SHOP)
        hybrid_allowed = tracker.allowed_template_names()
        self.assertIn("Use In Pack", hybrid_allowed)
        self.assertIn("Skip Pack", hybrid_allowed)
        self.assertIn("Buy Card", hybrid_allowed)
        self.assertIn("Next Round", hybrid_allowed)
        self.assertNotIn("Play Hand", hybrid_allowed)

        # If it was Mega, second Skip Pack returns cleanly to SHOP
        self.assertEqual(tracker.on_template_clicked("Skip Pack"), CoarseScreenState.SHOP)

    def test_skip_blind_tag_pack_and_double_tag_caution(self):
        tracker = ScreenStateTracker()
        tracker.on_template_clicked("Play Game")
        # Skip Blind could stay on SELECT (economy/investment tag) OR open a pack (Charm/Meteor tag)
        self.assertEqual(tracker.on_template_clicked("Skip Blind"), CoarseScreenState.SELECT_OR_PACK)
        allowed = tracker.allowed_template_names()
        self.assertIn("Select Blind", allowed)
        self.assertIn("Skip Blind", allowed)
        self.assertIn("Use In Pack", allowed)
        self.assertIn("Skip Pack", allowed)
        self.assertNotIn("Play Hand", allowed)

        # Choosing in pack -> PACK_OR_SELECT
        self.assertEqual(tracker.on_template_clicked("Select Card"), CoarseScreenState.PACK_OR_SELECT)
        # Skip Pack from SELECT keeps SELECT_OR_PACK cleanly in case Double Tag queued another pack
        self.assertEqual(tracker.on_template_clicked("Skip Pack"), CoarseScreenState.SELECT_OR_PACK)
        # Select Blind resolves back to BLIND
        self.assertEqual(tracker.on_template_clicked("Select Blind"), CoarseScreenState.BLIND)

    def test_options_allows_new_run_play_game_and_end_round_narrows_to_cash_out(self):
        tracker = ScreenStateTracker()
        tracker.on_template_clicked("Select Blind")
        self.assertNotIn("Play Game", tracker.allowed_template_names())
        tracker.on_template_clicked("Open Options Menu")
        self.assertIn("Play Game", tracker.allowed_template_names())
        self.assertIn("Play Hand", tracker.allowed_template_names())

        # Normal gameplay resumes clearing allow_play_game
        tracker.on_template_clicked("Play Hand")
        self.assertNotIn("Play Game", tracker.allowed_template_names())

        # Explicit temporal END_ROUND narrows BLIND -> CASH_OUT
        self.assertEqual(tracker.on_end_round(), CoarseScreenState.CASH_OUT)
        self.assertNotIn("Play Hand", tracker.allowed_template_names())
        self.assertIn("Cash Out", tracker.allowed_template_names())

    def test_shop_pack_exit_triggers_next_round_cooldown(self):
        tracker = ScreenStateTracker()
        tracker.on_template_clicked("Select Blind", frame_idx=100)
        tracker.on_template_clicked("Cash Out", frame_idx=200)
        self.assertNotIn("Next Round", tracker.allowed_template_names(frame_idx=230))
        self.assertIn("Next Round", tracker.allowed_template_names(frame_idx=265))

        # Closing a booster pack slides the Shop back up and also suppresses premature Next Round
        tracker.on_template_clicked("Open Pack", frame_idx=400)
        tracker.on_template_clicked("Use In Pack", frame_idx=500)
        self.assertNotIn("Next Round", tracker.allowed_template_names(frame_idx=532))
        self.assertIn("Next Round", tracker.allowed_template_names(frame_idx=565))

        tracker.on_template_clicked("Open Pack", frame_idx=700)
        tracker.on_template_clicked("Skip Pack", frame_idx=800)
        self.assertNotIn("Next Round", tracker.allowed_template_names(frame_idx=850))
        self.assertIn("Next Round", tracker.allowed_template_names(frame_idx=865))

    def test_consistency_with_gamestate(self):
        s_blind = GameState(location=Location.BLIND, hands_left=4, discards_left=3)
        for tmpl in ("Play Hand", "Discard", "Sort by Rank", "Sort by Suit", "Sell Card", "Use Consumable"):
            self.assertTrue(is_action_clickable_in_gamestate(s_blind, tmpl))
            self.assertIn(tmpl, ALLOWED_TEMPLATES_BY_STATE[CoarseScreenState.BLIND])

        s_shop = GameState(location=Location.SHOP)
        for tmpl in ("Buy Card", "Buy and Use", "Open Pack", "Redeem Voucher", "Reroll Shop", "Next Round"):
            self.assertTrue(is_action_clickable_in_gamestate(s_shop, tmpl))
            self.assertIn(tmpl, ALLOWED_TEMPLATES_BY_STATE[CoarseScreenState.SHOP])

        s_pack = GameState(
            location=Location.PACK,
            active_pack=BoosterPack(
                booster_type=BoosterType.ARCANA, size=BoosterSize.REGULAR, cost=4, choose=1, extra=3
            ),
        )
        for tmpl in ("Use In Pack", "Select Card", "Skip Pack", "Sell Card", "Use Consumable"):
            self.assertTrue(is_action_clickable_in_gamestate(s_pack, tmpl))
            self.assertIn(tmpl, ALLOWED_TEMPLATES_BY_STATE[CoarseScreenState.PACK])
        for forbidden in ("Play Hand", "Sort by Rank", "Buy Card", "Select Blind"):
            self.assertFalse(is_action_clickable_in_gamestate(s_pack, forbidden))
            self.assertNotIn(forbidden, ALLOWED_TEMPLATES_BY_STATE[CoarseScreenState.PACK])

    def test_windows_and_posix_video_output_path(self):
        win_out = main_module.build_output_json_path(
            "outputs", r"C:\Users\thatp\Videos\balatro_run_1.mp4", timestamp=123456
        )
        self.assertEqual(Path(win_out), Path("outputs") / "balatro_run_1_123456.json")
        posix_out = main_module.build_output_json_path(
            "outputs", "/home/user/videos/balatro_run_2.mkv", timestamp=123456
        )
        self.assertEqual(Path(posix_out), Path("outputs") / "balatro_run_2_123456.json")

    def test_on_pack_possibly_opened_widens_shop_and_recovers(self):
        tracker = ScreenStateTracker()
        # No-op outside SHOP
        self.assertEqual(tracker.on_pack_possibly_opened(), CoarseScreenState.PRE_GAME)
        tracker.on_template_clicked("Select Blind")
        self.assertEqual(tracker.on_pack_possibly_opened(), CoarseScreenState.BLIND)

        # In SHOP -> widens to PACK_OR_SHOP with pre_pack_location=SHOP and Next Round cooldown preserved
        tracker.on_template_clicked("Cash Out", frame_idx=200)
        self.assertEqual(tracker.on_pack_possibly_opened(), CoarseScreenState.PACK_OR_SHOP)
        self.assertEqual(tracker.pre_pack_location, Location.SHOP)
        allowed = tracker.allowed_template_names(frame_idx=220)
        self.assertIn("Select Card", allowed)
        self.assertIn("Use In Pack", allowed)
        self.assertIn("Skip Pack", allowed)
        self.assertNotIn("Next Round", allowed)

        # Clicking a SHOP button narrows PACK_OR_SHOP cleanly back to SHOP
        self.assertEqual(tracker.on_template_clicked("Buy Card", frame_idx=240), CoarseScreenState.SHOP)

    def test_process_frame_luminance_guards_sell_gap_and_tracks_open_pack_present(self):
        import logging
        from types import SimpleNamespace
        from unittest.mock import patch
        import numpy as np

        from config import Config
        from eval_template import EvaluationResult
        from geometry import ImageSize, PixelROI
        from matcher import TemplateMatch

        vf = SimpleNamespace(
            index=100,
            timestamp_seconds=1.67,
            image=np.zeros((1080, 1920, 3), dtype=np.uint8),
        )
        cfg = Config(debug_plot=False, setup_mode=False, coarse_to_fine=False)
        tmpl_sell = SimpleNamespace(name="Sell Card", shadow_dy=0)
        tmpl_use = SimpleNamespace(name="Use Consumable", shadow_dy=0)
        tmpl_open = SimpleNamespace(name="Open Pack", shadow_dy=6)
        tkey = {"Sell Card": 0, "Use Consumable": 1, "Open Pack": 2}
        dummy_match = TemplateMatch(
            found=True,
            top_left=(1500, 100),
            confidence=0.96,
            score_map=None,
            roi=PixelROI(1, 1, 1920, 1080),
            template_size=ImageSize(75, 82),
        )

        def fake_eval(tmpl, img, gray, config, **kwargs):
            if tmpl.name == "Sell Card":
                return EvaluationResult(
                    template_name="Sell Card",
                    found=True,
                    match=dummy_match,
                    clicked=False,
                    box=(1500, 100, 1575, 182),
                    luminance_ratio=1.85,
                )
            if tmpl.name == "Use Consumable":
                return EvaluationResult(
                    template_name="Use Consumable",
                    found=True,
                    match=dummy_match,
                    clicked=False,
                    box=(1500, 183, 1575, 249),
                    luminance_ratio=1.70,
                )
            return EvaluationResult(
                template_name="Open Pack",
                found=True,
                match=dummy_match,
                clicked=False,
                box=(1200, 960, 1290, 1034),
                luminance_ratio=1.10,
            )

        with patch.object(main_module, "eval", side_effect=fake_eval):
            _, _, metrics, clicked, clicked_name = main_module._process_frame(
                vf,
                [tmpl_sell, tmpl_use, tmpl_open],
                tkey,
                cfg,
                logging.getLogger(__name__),
            )

        # Unhovered Sell Card (luminance_ratio=1.85) must NOT be flipped to True even though y1-y0 < 2
        self.assertIsNone(clicked)
        self.assertIsNone(clicked_name)
        self.assertTrue(metrics.get("open_pack_present"))

    def test_full_deck_template_allowed_everywhere_and_view_mode_metadata(self):
        import logging
        from pathlib import Path as _Path
        from types import SimpleNamespace
        from unittest.mock import patch
        import cv2
        import numpy as np

        from config import Config
        from eval_template import EvaluationResult, detect_view_deck_tab_mode
        from geometry import ImageSize, PixelROI
        from matcher import TemplateMatch

        tracker = ScreenStateTracker()
        # Full Deck is scannable on every coarse state without mutating state or flags
        for st in (
            CoarseScreenState.PRE_GAME,
            CoarseScreenState.SELECT,
            CoarseScreenState.BLIND,
            CoarseScreenState.CASH_OUT,
            CoarseScreenState.SHOP,
            CoarseScreenState.PACK,
        ):
            tracker.state = st
            self.assertIn("Full Deck", tracker.allowed_template_names())
            self.assertEqual(tracker.on_template_clicked("Full Deck", frame_idx=100), st)

        # Real screenshots confirm red triangle above Full Deck detects view_mode accurately
        img_full = cv2.imread(str(_Path(REPO_ROOT) / "assets" / "sample_inputs" / "full_deck_shop.jpg"))
        img_rem = cv2.imread(str(_Path(REPO_ROOT) / "assets" / "sample_inputs" / "full_deck_blind.jpg"))
        self.assertIsNotNone(img_full)
        self.assertIsNotNone(img_rem)
        self.assertEqual(detect_view_deck_tab_mode(img_full, (837, 105, 1041, 167)), "full_deck")
        self.assertEqual(detect_view_deck_tab_mode(img_rem, (948, 105, 1152, 167)), "remaining")

        # main._process_frame attaches metadata['view_mode'] when Full Deck matches
        cfg = Config(debug_plot=False, setup_mode=False, coarse_to_fine=False)
        tmpl_fd = SimpleNamespace(name="Full Deck", shadow_dy=0)
        dummy_match = TemplateMatch(
            found=True,
            top_left=(837, 105),
            confidence=0.992,
            score_map=None,
            roi=PixelROI(1, 1, 1920, 1080),
            template_size=ImageSize(204, 62),
        )
        vf = SimpleNamespace(index=50, timestamp_seconds=0.83, image=img_full)
        fd_res = EvaluationResult(
            template_name="Full Deck",
            found=True,
            match=dummy_match,
            clicked=True,
            box=(837, 105, 1041, 167),
        )
        with patch.object(main_module, "eval", return_value=fd_res):
            _, _, _, clicked, clicked_name = main_module._process_frame(
                vf, [tmpl_fd], {"Full Deck": 22}, cfg, logging.getLogger(__name__)
            )
        self.assertEqual(clicked_name, "Full Deck")
        self.assertEqual(clicked["metadata"]["view_mode"], "full_deck")

    def test_shop_idle_fallback_widens_to_select_blind_and_pack(self):
        tracker = ScreenStateTracker(state=CoarseScreenState.SHOP)
        self.assertEqual(tracker.on_shop_idle_fallback(), CoarseScreenState.SHOP_OR_SELECT_OR_PACK)
        allowed = tracker.allowed_template_names()
        for name in ("Select Blind", "Skip Blind", "Open Pack", "Buy Card", "Next Round"):
            self.assertIn(name, allowed)
        self.assertEqual(tracker.on_template_clicked("Select Blind"), CoarseScreenState.BLIND)

    def test_cash_out_cooldown_after_held_item_click_and_reset(self):
        tracker = ScreenStateTracker(state=CoarseScreenState.BLIND)
        self.assertIn("Cash Out", tracker.allowed_template_names(frame_idx=100))

        tracker.on_template_clicked("Use Consumable", frame_idx=100)
        self.assertEqual(tracker.last_held_item_action_frame, 100)
        self.assertNotIn("Cash Out", tracker.allowed_template_names(frame_idx=105))
        self.assertNotIn("Cash Out", tracker.allowed_template_names(frame_idx=129))
        self.assertIn("Cash Out", tracker.allowed_template_names(frame_idx=130))

        tracker.state = CoarseScreenState.CASH_OUT
        tracker.on_template_clicked("Sell Card", frame_idx=200)
        self.assertNotIn("Cash Out", tracker.allowed_template_names(frame_idx=210))

        tracker.on_template_clicked("Play Hand", frame_idx=215)
        self.assertIsNone(tracker.last_held_item_action_frame)
        self.assertIn("Cash Out", tracker.allowed_template_names(frame_idx=216))

    def test_blind_full_deck_tab_recovers_coarse_state_to_blind(self):
        tracker = ScreenStateTracker(state=CoarseScreenState.SHOP)
        self.assertNotIn("Play Hand", tracker.allowed_template_names())
        self.assertEqual(tracker.on_blind_view_deck_observed(), CoarseScreenState.BLIND)
        allowed = tracker.allowed_template_names()
        self.assertIn("Play Hand", allowed)
        self.assertIn("Discard", allowed)
        self.assertIn("Cash Out", allowed)


if __name__ == "__main__":
    unittest.main()
