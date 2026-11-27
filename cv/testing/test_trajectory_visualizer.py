"""Tests for `tools/visualize_trajectory.py` sprite manifest, diffs, zones, and HTTP endpoints."""

from __future__ import annotations

import http.client
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from game.balatro_models import (
    Action,
    ActionType,
    BlindName,
    BoosterPack,
    BoosterSize,
    BoosterType,
    Deck,
    Edition,
    Enhancement,
    Hand,
    HandCard,
    Inventory,
    Joker,
    JokerName,
    Location,
    Planet,
    PlanetName,
    PlayingCard,
    PokerHand,
    Rank,
    Seal,
    SkipTag,
    SpectralName,
    Stake,
    Suit,
    Tarot,
    TarotName,
    Voucher,
    VoucherName,
)
from game.gamestate import GameState, RunTrajectory
from game.trajectory_codec import serialize_trajectory
from tools.visualize_trajectory import (
    TEXTURES_2X_DIR,
    TrajectoryVisualizerServer,
    VideoFrameCache,
    _QuietThreadingHTTPServer,
    build_visualization_payload,
    find_matching_video,
    load_balatro_sprite_manifest,
)


def _sample_card(cid: int, rank: Rank, suit: Suit, **kwargs) -> PlayingCard:
    return PlayingCard(id=cid, rank=rank, suit=suit, **kwargs)


class TestSpriteManifest(unittest.TestCase):
    """Verifies Balatro `game.lua` sprite extraction covers all engine enums."""

    def test_manifest_covers_all_game_enums(self) -> None:
        manifest = load_balatro_sprite_manifest()

        missing: dict[str, list[str]] = {}
        enum_checks: list[tuple[str, list[str]]] = [
            ("jokers", [e.value for e in JokerName]),
            ("tarots", [e.value for e in TarotName]),
            ("planets", [e.value for e in PlanetName]),
            ("spectrals", [e.value for e in SpectralName]),
            ("vouchers", [e.value for e in VoucherName]),
            ("backs", [e.value for e in Deck]),
            ("tags", [e.value for e in SkipTag]),
            ("blinds", [e.value for e in BlindName]),
            ("stakes", [e.value for e in Stake]),
            ("enhancements", [e.value for e in Enhancement]),
            ("seals", [s.value for s in Seal if s != Seal.NONE]),
            ("boosters", [f"{sz.value} {bt.value}" for bt in BoosterType for sz in BoosterSize]),
        ]
        for bucket, names in enum_checks:
            miss = [n for n in names if n not in manifest[bucket]]
            if miss:
                missing[bucket] = miss

        self.assertEqual(missing, {})

        grid = manifest["playing_card_grid"]
        for rank in Rank:
            if rank != Rank.NONE:
                self.assertIn(rank.value, grid["ranks"])
        for suit in Suit:
            if suit != Suit.NONE:
                self.assertIn(suit.value, grid["suits"])

        # Soul overlay layers on Hologram & Legendaries
        for soul_joker in (
            JokerName.HOLOGRAM,
            JokerName.CANIO,
            JokerName.TRIBOULET,
            JokerName.YORICK,
            JokerName.CHICOT,
            JokerName.PERKEO,
        ):
            self.assertIn("soul", manifest["jokers"][soul_joker.value])

        self.assertIn("debuff", manifest["overlays"])
        for st_key in ("eternal", "perishable", "rental"):
            self.assertIn(st_key, manifest["stickers"])


class TestVisualizationPayload(unittest.TestCase):
    """Verifies step reconstruction, pre-action targets, deck zones, and diffs."""

    def _build_synthetic_trajectory_json(self) -> dict:
        c1 = _sample_card(1, Rank.ACE, Suit.SPADES)
        c2 = _sample_card(2, Rank.KING, Suit.SPADES)
        c3 = _sample_card(3, Rank.QUEEN, Suit.HEARTS)
        c4 = _sample_card(4, Rank.JACK, Suit.CLUBS)
        c5 = _sample_card(5, Rank.TEN, Suit.DIAMONDS)
        deck = [c1, c2, c3, c4, c5]

        # Step 0: Blind with 2 cards in hand, 1 in discard, 2 in draw pile
        state = GameState(
            step_idx=0,
            frame_idx=100,
            timestamp=1.5,
            location=Location.BLIND,
            deck=Deck.ANAGLYPH_DECK,
            stake=Stake.GOLD,
            seed="TESTSEED",
            ante=1,
            round=1,
            money=10,
            hands_left=4,
            discards_left=3,
            reroll_cost=5,
            target_chips=300,
            current_chips=0,
            full_deck=[PlayingCard(**vars(c)) for c in deck],
            hand=Hand(cards=[HandCard(card=_sample_card(1, Rank.ACE, Suit.SPADES), selected=True, slot_uid=101),
                             HandCard(card=_sample_card(2, Rank.KING, Suit.SPADES), slot_uid=102)]),
            discard_pile=[_sample_card(3, Rank.QUEEN, Suit.HEARTS)],
            played_cards_this_round=[],
            played_cards_this_ante=[],
            inventory=Inventory(
                jokers=[Joker(name=JokerName.JOKER, id=11)],
                consumables=[Tarot(name=TarotName.EMPRESS)],
            ),
            shop_cards=[Joker(name=JokerName.GREEDY_JOKER, id=12, edition=Edition.FOIL)],
            shop_vouchers=[Voucher(name=VoucherName.OVERSTOCK)],
            shop_packs=[BoosterPack(booster_type=BoosterType.ARCANA, size=BoosterSize.REGULAR, cost=4, choose=1, extra=3)],
            apply_deck_stake_on_init=False,
        )
        traj = RunTrajectory()
        traj.record_step(state)

        # Step 1: PLAY_HAND with Ace of Spades + PokerHand level & play count increment
        state = state.copy()
        state.frame_idx = 200
        state.timestamp = 3.2
        state.hands_left = 3
        state.current_chips = 150
        state.hand_levels[PokerHand.HIGH_CARD] = 2
        state.hand_play_counts[PokerHand.HIGH_CARD] = 1
        state.hands_played_this_round_by_type.add(PokerHand.HIGH_CARD)
        state.played_cards_this_round.append(_sample_card(1, Rank.ACE, Suit.SPADES))
        state.played_cards_this_ante.append(_sample_card(1, Rank.ACE, Suit.SPADES))
        state.hand.cards = [HandCard(card=_sample_card(2, Rank.KING, Suit.SPADES), slot_uid=102)]
        traj.record_step(
            state,
            Action(
                type=ActionType.PLAY_HAND,
                frame_idx=200,
                timestamp=3.2,
                selected_cards=[_sample_card(1, Rank.ACE, Suit.SPADES)],
            ),
        )

        # Step 2: BUY_CARD at shop index 0 (Foil Greedy Joker) & deck card modification + addition
        state = state.copy()
        state.frame_idx = 300
        state.timestamp = 5.0
        state.location = Location.SHOP
        state.money = 3
        bought = state.shop_cards.pop(0)
        state.inventory.jokers.append(bought)
        state.full_deck[1].enhancement = Enhancement.GLASS
        state.full_deck.append(_sample_card(6, Rank.TWO, Suit.HEARTS, seal=Seal.RED))
        traj.record_step(
            state,
            Action(type=ActionType.BUY_CARD, frame_idx=300, timestamp=5.0, target_idx=0),
        )

        # Step 3: SELL_CARD at index 2 (consumable slot 0: The Empress, after 2 jokers)
        state = state.copy()
        state.frame_idx = 400
        state.timestamp = 6.7
        state.money = 4
        state.inventory.consumables.pop(0)
        traj.record_step(
            state,
            Action(type=ActionType.SELL_CARD, frame_idx=400, timestamp=6.7, target_idx=2),
        )

        return serialize_trajectory(state, traj)

    def test_zones_targets_and_step_diffs(self) -> None:
        raw_json = self._build_synthetic_trajectory_json()
        payload = build_visualization_payload(raw_json)

        self.assertEqual(payload["num_steps"], 4)
        self.assertEqual(payload["run_meta"]["deck"], "Anaglyph Deck")
        self.assertEqual(payload["run_meta"]["stake"], "Gold Stake")

        # Step 0 zones: #1, #2 in_hand; #3 discarded; #4, #5 in_draw_pile
        s0 = payload["steps"][0]["state"]
        zones0 = {c["id"]: c["zone"] for c in s0["full_deck"]}
        self.assertEqual(
            zones0,
            {
                1: "in_hand",
                2: "in_hand",
                3: "discarded",
                4: "in_draw_pile",
                5: "in_draw_pile",
            },
        )
        self.assertFalse(payload["steps"][0]["diff"]["has_changes"])

        # Step 1 (PLAY_HAND): zone moves Ace♠ -> played_round; PokerHand High Card diffed
        step1 = payload["steps"][1]
        self.assertEqual(step1["action"]["target_item"]["area"], "hand")
        self.assertEqual(step1["action"]["target_item"]["label"], "Ace♠")
        zones1 = {c["id"]: c["zone"] for c in step1["state"]["full_deck"]}
        self.assertEqual(zones1[1], "played_round")
        self.assertEqual(
            step1["diff"]["full_deck"]["zone_changed"]["1"],
            {"prev": "in_hand", "curr": "played_round"},
        )
        self.assertEqual(
            step1["diff"]["poker_hands"]["High Card"]["level"],
            {"prev": 1, "curr": 2},
        )
        self.assertEqual(
            step1["diff"]["poker_hands"]["High Card"]["play_count"],
            {"prev": 0, "curr": 1},
        )

        # Step 2 (BUY_CARD): target resolved to pre-action shop_cards[0] (Greedy Joker),
        # joker added_ids=[12], full_deck card #6 added, card #2 modified -> glass
        step2 = payload["steps"][2]
        self.assertEqual(step2["action"]["target_item"]["area"], "shop_cards")
        self.assertEqual(step2["action"]["target_item"]["label"], "Greedy Joker")
        self.assertEqual(step2["diff"]["jokers"]["added_ids"], [12])
        self.assertEqual(step2["diff"]["full_deck"]["added_ids"], [6])
        self.assertEqual(
            step2["diff"]["full_deck"]["modified"]["2"]["enhancement"],
            {"prev": "none", "curr": "glass"},
        )

        # Step 3 (SELL_CARD idx=2 -> consumable #0 The Empress after 2 jokers)
        step3 = payload["steps"][3]
        self.assertEqual(step3["action"]["target_item"]["area"], "consumables")
        self.assertEqual(step3["action"]["target_item"]["label"], "The Empress")
        self.assertEqual(len(step3["diff"]["consumables"]["removed"]), 1)
        self.assertEqual(step3["diff"]["consumables"]["removed"][0]["name"], "The Empress")

    def test_anomaly_fixes_sell_reorder_shop_stickers_observed_shop_and_hook(self) -> None:
        from game.engine.rules import on_action
        from game.trajectory_codec import _serialize_card, _serialize_voucher

        deck = [
            _sample_card(1, Rank.ACE, Suit.SPADES),
            _sample_card(2, Rank.KING, Suit.SPADES),
            _sample_card(3, Rank.QUEEN, Suit.HEARTS),
            _sample_card(4, Rank.EIGHT, Suit.SPADES),
        ]
        state = GameState(
            seed="FIXTEST",
            deck=Deck.RED_DECK,
            stake=Stake.GOLD,
            location=Location.SHOP,
            money=20,
            current_blind=BlindName.THE_HOOK,
            full_deck=[PlayingCard(**vars(c)) for c in deck],
            inventory=Inventory(
                jokers=[
                    Joker(name=JokerName.RAMEN, id=1),
                    Joker(name=JokerName.JOKER, id=2, rental=True),
                ],
            ),
            apply_deck_stake_on_init=False,
        )
        traj = RunTrajectory()
        traj.record_step(state)

        # 1) BUY_CARD with observed_shop (Supernova + Perishable Jolly Joker + Overstock)
        supernova = Joker(name=JokerName.SUPERNOVA)
        jolly = Joker(name=JokerName.JOLLY_JOKER, perishable=True, perish_tally=5)
        overstock = Voucher(name=VoucherName.OVERSTOCK)
        state.shop_cards = [supernova, jolly]
        state.shop_vouchers = [overstock]
        buy_act = Action(
            type=ActionType.BUY_CARD,
            frame_idx=100,
            timestamp=1.5,
            target_idx=0,
            metadata={
                "target_card": "Supernova",
                "observed_shop": {
                    "cards": [_serialize_card(supernova), _serialize_card(jolly)],
                    "vouchers": [_serialize_voucher(overstock)],
                    "packs": [],
                },
            },
        )
        on_action(state, buy_act)
        traj.record_step(state, buy_act)

        # 2) NEXT_ROUND clears shop shelves cleanly (_leave_shop fix)
        leave_act = Action(type=ActionType.NEXT_ROUND, frame_idx=150, timestamp=2.5)
        on_action(state, leave_act)
        traj.record_step(state, leave_act)
        self.assertEqual(state.shop_cards, [])
        self.assertEqual(state.shop_vouchers, [])
        self.assertEqual(state.shop_packs, [])

        # 3) Enter Blind + SELL_CARD with joker_order reordering Joker(#2) ahead of Ramen(#1)
        state.location = Location.BLIND
        state.current_blind = BlindName.THE_HOOK
        state.hand = Hand(
            cards=[
                HandCard(card=_sample_card(1, Rank.ACE, Suit.SPADES), selected=True, slot_uid=101),
                HandCard(card=_sample_card(2, Rank.KING, Suit.SPADES), selected=False, slot_uid=102),
                HandCard(card=_sample_card(3, Rank.QUEEN, Suit.HEARTS), selected=False, slot_uid=103),
                HandCard(card=_sample_card(4, Rank.EIGHT, Suit.SPADES), selected=False, slot_uid=104),
            ]
        )
        sell_act = Action(
            type=ActionType.SELL_CARD,
            frame_idx=200,
            timestamp=3.3,
            target_idx=0,
            joker_order=[2, 1, 3],
            metadata={"target_card": "Joker"},
        )
        on_action(state, sell_act)
        traj.record_step(state, sell_act)

        # 4) PLAY_HAND against The Hook separates played card vs Hook forced discards
        play_act = Action(
            type=ActionType.PLAY_HAND,
            frame_idx=250,
            timestamp=4.1,
            selected_cards=[_sample_card(1, Rank.ACE, Suit.SPADES)],
        )
        on_action(state, play_act)
        traj.record_step(state, play_act)

        payload = build_visualization_payload(serialize_trajectory(state, traj))
        s1 = payload["steps"][1]
        self.assertEqual(s1["action"]["target_item"]["label"], "Supernova")
        self.assertEqual(len(s1["action"]["observed_shop"]["cards"]), 2)
        # Shop Jolly Joker keeps perishable metadata cleanly
        self.assertTrue(s1["state"]["shop_cards"][0]["perishable"])
        self.assertEqual(s1["state"]["shop_cards"][0]["perish_tally"], 5)

        s2 = payload["steps"][2]
        self.assertEqual(s2["state"]["shop_cards"], [])

        s3 = payload["steps"][3]
        self.assertEqual(s3["action"]["target_item"]["label"], "Joker")

        s4 = payload["steps"][4]
        self.assertEqual([c["rank"] for c in s4["diff"]["hand"]["played_cards"]], ["Ace"])
        self.assertEqual(len(s4["diff"]["hand"]["discarded_cards"]), 2)


class TestVisualizerServer(unittest.TestCase):
    """Tests HTTP API endpoints, static textures, security checks, and frame serving."""

    def test_http_endpoints_and_security(self) -> None:
        real_traj = (
            REPO_ROOT
            / "output"
            / "trajectories"
            / "bu_anaglyph_deck_11-13-26_1794699311_trajectory.json"
        )
        sample_video = (
            REPO_ROOT / "assets" / "sample_inputs" / "videos" / "bu_anaglyph_deck_11-13-26.mp4"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_traj = Path(tmpdir) / "mini_trajectory.json"
            if real_traj.is_file():
                traj_to_serve = real_traj
            else:
                state = GameState(apply_deck_stake_on_init=False)
                traj = RunTrajectory()
                traj.record_step(state)
                tmp_traj.write_text(json.dumps(serialize_trajectory(state, traj)), encoding="utf-8")
                traj_to_serve = tmp_traj

            matched = find_matching_video(traj_to_serve)
            if real_traj.is_file() and sample_video.is_file():
                self.assertEqual(matched, sample_video.resolve())

            server = TrajectoryVisualizerServer(
                trajectory_path=traj_to_serve,
                video_path=matched,
                host="127.0.0.1",
                port=0,
            )
            thread = threading.Thread(target=server.httpd.serve_forever, daemon=True)
            thread.start()
            try:
                conn = http.client.HTTPConnection(server.host, server.port, timeout=10)
                try:
                    # 1. HTML root
                    conn.request("GET", "/")
                    r_html = conn.getresponse()
                    html_body = r_html.read()
                    self.assertEqual(r_html.status, 200)
                    self.assertIn(b"BALATRO TRAJECTORY", html_body)
                    self.assertIn(b"item.eternal", html_body)
                    self.assertIn(b"item.perishable", html_body)
                    self.assertIn(b"HOOK DISCARD", html_body)

                    # 2. Manifest
                    conn.request("GET", "/api/manifest")
                    r_man = conn.getresponse()
                    man_data = json.loads(r_man.read().decode("utf-8"))
                    self.assertEqual(r_man.status, 200)
                    self.assertIn("jokers", man_data)

                    # 3. Trajectories list & active trajectory
                    conn.request("GET", "/api/trajectories")
                    r_list = conn.getresponse()
                    r_list.read()
                    self.assertEqual(r_list.status, 200)

                    conn.request("GET", "/api/trajectory")
                    r_traj = conn.getresponse()
                    traj_data = json.loads(r_traj.read().decode("utf-8"))
                    self.assertEqual(r_traj.status, 200)
                    self.assertGreaterEqual(traj_data["num_steps"], 1)

                    # 4. Texture asset + traversal guard
                    if (TEXTURES_2X_DIR / "Jokers.png").is_file():
                        conn.request("GET", "/textures/Jokers.png")
                        r_tex = conn.getresponse()
                        tex_bytes = r_tex.read()
                        self.assertEqual(r_tex.status, 200)
                        self.assertTrue(tex_bytes.startswith(b"\x89PNG"))

                    conn.request("GET", "/textures/../game.lua")
                    r_sec = conn.getresponse()
                    r_sec.read()
                    self.assertEqual(r_sec.status, 403)

                    # 5. Live MP4 frame endpoint (when video present)
                    if matched and matched.is_file():
                        conn.request("GET", "/api/frame/0")
                        r_frm = conn.getresponse()
                        jpg_bytes = r_frm.read()
                        self.assertEqual(r_frm.status, 200)
                        self.assertTrue(jpg_bytes.startswith(b"\xff\xd8"))
                finally:
                    conn.close()
            finally:
                server.close()
                thread.join(timeout=5)


def _create_synthetic_video(video_dir: Path, num_frames: int = 20) -> Path:
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    for ext, fourcc_str in ((".mp4", "mp4v"), (".avi", "MJPG")):
        cand = video_dir / f"test_video{ext}"
        writer = cv2.VideoWriter(
            str(cand),
            cv2.VideoWriter_fourcc(*fourcc_str),
            10.0,
            (64, 48),
        )
        if not writer.isOpened():
            writer.release()
            continue
        try:
            for idx in range(num_frames):
                frame = np.full((48, 64, 3), (idx * 10) % 255, dtype=np.uint8)
                writer.write(frame)
        finally:
            writer.release()
        if cand.is_file() and cand.stat().st_size > 2048:
            cap = cv2.VideoCapture(str(cand))
            ok = cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) >= num_frames
            cap.release()
            if ok:
                return cand
    raise RuntimeError("No working cv2.VideoWriter codec found for synthetic test video.")


class TestVisualizerComparisonAndVideo(unittest.TestCase):
    """Tests telemetry comparison endpoints, error-first preloading, /api/video Range, and HTML controls."""

    def _make_pair(self, tmp: Path) -> tuple[Path, Path]:
        def _build(*, inject_error: bool) -> dict:
            deck = [
                _sample_card(1, Rank.ACE, Suit.SPADES),
                _sample_card(2, Rank.KING, Suit.HEARTS),
                _sample_card(3, Rank.QUEEN, Suit.CLUBS),
            ]
            st = GameState(
                seed="VISCMP",
                deck=Deck.RED_DECK,
                stake=Stake.WHITE,
                location=Location.SELECT,
                money=10,
                full_deck=[PlayingCard(**vars(c)) for c in deck],
                apply_deck_stake_on_init=False,
            )
            traj = RunTrajectory()
            traj.record_step(st)  # step 0, f=0

            st = st.copy()
            st.frame_idx = 5
            st.location = Location.BLIND
            traj.record_step(st, Action(type=ActionType.SELECT_BLIND, frame_idx=5, timestamp=0.5))  # step 1

            st = st.copy()
            st.frame_idx = 15
            st.hands_left = 3
            if inject_error:
                st.full_deck.append(_sample_card(50, Rank.TWO, Suit.DIAMONDS))
            traj.record_step(
                st,
                Action(
                    type=ActionType.PLAY_HAND,
                    frame_idx=15,
                    timestamp=1.5,
                    selected_cards=[_sample_card(1, Rank.ACE, Suit.SPADES)],
                ),
            )  # step 2 (error when inject_error=True)

            st = st.copy()
            st.frame_idx = 18
            st.round = 2
            traj.record_step(st, Action(type=ActionType.END_ROUND, frame_idx=18, timestamp=1.8))  # step 3
            return serialize_trajectory(st, traj)

        actual_path = tmp / "run_actual_trajectory.json"
        gt_path = tmp / "run_VISCMP_gt_trajectory.json"
        actual_path.write_text(json.dumps(_build(inject_error=True)), encoding="utf-8")
        gt_path.write_text(json.dumps(_build(inject_error=False)), encoding="utf-8")
        return actual_path, gt_path

    def test_error_first_priority_preloader_and_pinning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            vid_path = _create_synthetic_video(Path(tmpdir), num_frames=20)
            cache = VideoFrameCache(vid_path, max_items=4)
            try:
                self.assertTrue(cache.meta.get("available"))
                cache.start_preload(priority_frames=[15, 18], secondary_frames=[0, 5, 10, 12])
                deadline = time.monotonic() + 5.0
                status = cache.preload_status()
                while not status["done"] and time.monotonic() < deadline:
                    time.sleep(0.05)
                    status = cache.preload_status()
                self.assertTrue(status["done"])
                self.assertEqual(status["priority_cached"], 2)
                self.assertEqual(status["priority_total"], 2)
                self.assertEqual(status["total_cached"], 6)
                # Priority error frames [15, 18] MUST be decoded strictly before secondary frames
                self.assertEqual(status["decoded_order"][:2], [15, 18])

                # Pin protection: even if max_items is capped artificially, pinned frames survive eviction
                with cache._lock:
                    cache.max_items = 4
                    cache._cache[9991] = b"x"
                    cache._evict_if_needed()
                    cache._cache[9992] = b"y"
                    cache._evict_if_needed()
                    self.assertIn(15, cache._cache)
                    self.assertIn(18, cache._cache)
            finally:
                cache.close()

    def test_compare_endpoints_preload_video_range_and_ui(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            actual_path, gt_path = self._make_pair(tmp)
            vid_path = _create_synthetic_video(tmp, num_frames=20)
            vid_bytes = vid_path.read_bytes()

            # Start server with startup telemetry_path=gt_path so preloader prioritizes error frame [15] first!
            server = TrajectoryVisualizerServer(
                trajectory_path=actual_path,
                video_path=vid_path,
                telemetry_path=gt_path,
                traj_dir=tmp,
                video_dir=tmp,
                host="127.0.0.1",
                port=0,
            )
            thread = threading.Thread(target=server.httpd.serve_forever, daemon=True)
            thread.start()
            try:
                # Wait for priority preloader to finish on 4 steps
                deadline = time.monotonic() + 5.0
                while not server.app_state.frame_cache.preload_status()["done"] and time.monotonic() < deadline:
                    time.sleep(0.05)
                order = server.app_state.frame_cache.preload_status()["decoded_order"]
                self.assertEqual(order[0], 15)  # Step 2 (error step) decoded FIRST!

                conn = http.client.HTTPConnection(server.host, server.port, timeout=10)
                try:
                    # 1. Rendered HTML smoke checks for video player toggle & error navigation
                    conn.request("GET", "/")
                    r_html = conn.getresponse()
                    html = r_html.read()
                    self.assertEqual(r_html.status, 200)
                    for needle in (
                        b'id="videoPlayer"',
                        b'id="btnVideoMode"',
                        b'id="telemetrySelect"',
                        b'id="btnPrevErr"',
                        b'id="btnNextErr"',
                        b'data-tab="errors"',
                        b"/api/video",
                        b"/api/compare",
                        b"/api/preload-status",
                    ):
                        self.assertIn(needle, html)

                    # 2. Startup trajectory payload contains non-cascading comparison summary
                    conn.request("GET", "/api/trajectory")
                    r_traj = conn.getresponse()
                    traj_payload = json.loads(r_traj.read().decode("utf-8"))
                    self.assertEqual(r_traj.status, 200)
                    self.assertEqual(traj_payload["comparison"]["summary"]["error_step_indices"], [2])

                    # 3. Telemetry sources & dynamic /api/compare + security traversal guard
                    conn.request("GET", "/api/telemetry-sources")
                    r_src = conn.getresponse()
                    src_data = json.loads(r_src.read().decode("utf-8"))
                    self.assertEqual(r_src.status, 200)
                    self.assertTrue(any(it["name"] == gt_path.name for it in src_data["items"]))

                    conn.request("GET", "/api/compare?source=none")
                    r_clr = conn.getresponse()
                    clr_data = json.loads(r_clr.read().decode("utf-8"))
                    self.assertEqual(r_clr.status, 200)
                    self.assertIsNone(clr_data["comparison"])

                    from urllib.parse import quote
                    conn.request("GET", f"/api/compare?source={quote(str(gt_path))}")
                    r_cmp = conn.getresponse()
                    cmp_data = json.loads(r_cmp.read().decode("utf-8"))
                    self.assertEqual(r_cmp.status, 200)
                    self.assertEqual(cmp_data["comparison"]["summary"]["error_step_indices"], [2])

                    conn.request("GET", "/api/compare?source=../../game.lua")
                    r_sec = conn.getresponse()
                    r_sec.read()
                    self.assertEqual(r_sec.status, 403)

                    # 4. Preload status endpoint
                    conn.request("GET", "/api/preload-status")
                    r_pre = conn.getresponse()
                    pre_data = json.loads(r_pre.read().decode("utf-8"))
                    self.assertEqual(r_pre.status, 200)
                    self.assertEqual(pre_data["priority_cached"], 1)

                    # 5. Byte-Range /api/video HEAD, full GET (200), Partial GET (206), Open-ended (206), 416
                    conn.request("HEAD", "/api/video")
                    r_head = conn.getresponse()
                    r_head.read()
                    self.assertEqual(r_head.status, 200)
                    self.assertEqual(r_head.getheader("Accept-Ranges"), "bytes")
                    self.assertEqual(int(r_head.getheader("Content-Length")), len(vid_bytes))

                    conn.request("GET", "/api/video")
                    r_full = conn.getresponse()
                    full_body = r_full.read()
                    self.assertEqual(r_full.status, 200)
                    self.assertEqual(r_full.getheader("Accept-Ranges"), "bytes")
                    self.assertEqual(full_body, vid_bytes)

                    conn.request("GET", "/api/video", headers={"Range": "bytes=0-1023"})
                    r_part = conn.getresponse()
                    part_body = r_part.read()
                    self.assertEqual(r_part.status, 206)
                    self.assertEqual(
                        r_part.getheader("Content-Range"),
                        f"bytes 0-1023/{len(vid_bytes)}",
                    )
                    self.assertEqual(part_body, vid_bytes[:1024])

                    conn.request("GET", "/api/video", headers={"Range": "bytes=1024-"})
                    r_tail = conn.getresponse()
                    tail_body = r_tail.read()
                    self.assertEqual(r_tail.status, 206)
                    self.assertEqual(
                        r_tail.getheader("Content-Range"),
                        f"bytes 1024-{len(vid_bytes) - 1}/{len(vid_bytes)}",
                    )
                    self.assertEqual(tail_body, vid_bytes[1024:])

                    conn.request("GET", "/api/video", headers={"Range": "bytes=999999999-"})
                    r_416 = conn.getresponse()
                    r_416.read()
                    self.assertEqual(r_416.status, 416)

                    # 6. Cache-key uniqueness, no-store on /api/frame, HTML ?v= cache-bust, Windows reuse guard
                    cache_a = VideoFrameCache(vid_path, max_items=2)
                    cache_b = VideoFrameCache(vid_path, max_items=2)
                    try:
                        key_a = cache_a.meta.get("cache_key")
                        key_b = cache_b.meta.get("cache_key")
                        self.assertTrue(key_a and key_b)
                        self.assertNotEqual(key_a, key_b)
                    finally:
                        cache_a.close()
                        cache_b.close()

                    conn.request("GET", f"/api/frame/0?v={server.app_state.frame_cache.meta['cache_key']}")
                    r_frame = conn.getresponse()
                    frame_body = r_frame.read()
                    self.assertEqual(r_frame.status, 200)
                    self.assertEqual(r_frame.getheader("Cache-Control"), "no-store")
                    self.assertGreater(len(frame_body), 100)

                    self.assertIn(b"/api/frame/${targetFrame}?v=", html)
                    self.assertIn(b'imgEl.removeAttribute("src")', html)
                    self.assertIn(b'id="diagTrack"', html)
                    self.assertIn(b'diag-seg-err', html)
                    self.assertIn(b'diag-seg-warn', html)
                    self.assertIn(b'warning_reasons', html)
                    self.assertEqual(_QuietThreadingHTTPServer.allow_reuse_address, os.name != "nt")
                finally:
                    conn.close()
            finally:
                server.close()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
