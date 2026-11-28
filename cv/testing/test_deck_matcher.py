"""Tests for `vod_pipeline.deck_matcher` left-slice deck template matching."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from game.balatro_models import Deck
from vod_pipeline.deck_matcher import (
    DEFAULT_ENHANCERS_PATH,
    PLAYABLE_DECK_TILE_COORDS,
    detect_deck_from_video,
    match_deck_in_frame,
)

_SCALE_1080P = 1.152
_TILE_W = 142
_TILE_H = 190


def _render_occluded_deck_frame(enh_sheet: np.ndarray, deck: Deck) -> np.ndarray:
    """Renders a 1080p frame with `deck` occluded on its right 58% and top 14%, JPEG-compressed."""
    canvas = np.full((1080, 1920, 3), (45, 78, 52), dtype=np.uint8)
    cx, cy = PLAYABLE_DECK_TILE_COORDS[deck]
    tile = enh_sheet[cy * _TILE_H : (cy + 1) * _TILE_H, cx * _TILE_W : (cx + 1) * _TILE_W]
    ys, xs = np.where(tile[:, :, 3] > 200)
    cropped = tile[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    nw = int(round(cropped.shape[1] * _SCALE_1080P))
    nh = int(round(cropped.shape[0] * _SCALE_1080P))
    scaled = cv2.resize(cropped, (nw, nh), interpolation=cv2.INTER_AREA)

    x0, y0 = 1580, 786
    alpha = scaled[:, :, 3:4].astype(np.float32) / 255.0
    fg = scaled[:, :, :3].astype(np.float32)
    bg = canvas[y0 : y0 + nh, x0 : x0 + nw].astype(np.float32)
    canvas[y0 : y0 + nh, x0 : x0 + nw] = np.clip(fg * alpha + bg * (1.0 - alpha), 0, 255).astype(np.uint8)

    # Simulate DrSpectred's bottom-right avatar covering right 58% of the deck
    occ_x = x0 + int(nw * 0.42)
    cv2.rectangle(canvas, (occ_x, y0 - 20), (1919, 1079), (40, 190, 235), thickness=-1)

    # Simulate plushie-on-deck avatar shoes covering top 14% of the deck
    occ_y = y0 + int(nh * 0.14)
    cv2.ellipse(canvas, (x0 + nw // 4, y0 + int(nh * 0.06)), (24, int(nh * 0.08)), 0, 0, 360, (28, 24, 34), -1)
    cv2.ellipse(canvas, (x0 + (3 * nw) // 4, y0 + int(nh * 0.06)), (24, int(nh * 0.08)), 0, 0, 360, (28, 24, 34), -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + nw, occ_y), (32, 28, 38), thickness=-1)

    ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    assert ok
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    assert decoded is not None
    return decoded


class TestDeckMatcher(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        enh = cv2.imread(str(DEFAULT_ENHANCERS_PATH), cv2.IMREAD_UNCHANGED)
        assert enh is not None, f"Missing Enhancers.png at {DEFAULT_ENHANCERS_PATH}"
        cls.enh_sheet = enh

    def test_synthetic_all_15_decks_with_heavy_occlusion_and_jpeg(self) -> None:
        self.assertEqual(len(PLAYABLE_DECK_TILE_COORDS), 15)
        for deck in PLAYABLE_DECK_TILE_COORDS:
            with self.subTest(deck=deck.value):
                frame = _render_occluded_deck_frame(self.enh_sheet, deck)
                res = match_deck_in_frame(frame)
                self.assertTrue(
                    res.accepted,
                    f"{deck.value} rejected: best={res.best_deck.value} runner_up={res.runner_up.value} "
                    f"score={res.score:.3f} sq={res.sq_score:.3f} cc={res.cc_score:.3f} margin={res.margin:.3f}",
                )
                self.assertEqual(res.deck, deck)
                self.assertGreaterEqual(res.margin, 0.08)

    def test_rejects_dimmed_overlay_and_blank_frames(self) -> None:
        blank = np.zeros((1080, 1920, 3), dtype=np.uint8)
        res_blank = match_deck_in_frame(blank)
        self.assertFalse(res_blank.accepted)
        self.assertIsNone(res_blank.deck)

        clean = _render_occluded_deck_frame(self.enh_sheet, Deck.ANAGLYPH_DECK)
        dimmed = np.clip(clean.astype(np.float32) * 0.45 + 30.0, 0, 255).astype(np.uint8)
        res_dim = match_deck_in_frame(dimmed)
        self.assertFalse(res_dim.accepted)
        self.assertIsNone(res_dim.deck)

    def test_detect_deck_from_video_synthetic(self) -> None:
        clean_zodiac = _render_occluded_deck_frame(self.enh_sheet, Deck.ZODIAC_DECK)
        blank = np.zeros((1080, 1920, 3), dtype=np.uint8)
        dimmed = np.clip(clean_zodiac.astype(np.float32) * 0.45 + 30.0, 0, 255).astype(np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            vid_path: Path | None = None
            for ext, fourcc in ((".mp4", "mp4v"), (".avi", "MJPG")):
                cand = Path(tmpdir) / f"synthetic_zodiac{ext}"
                writer = cv2.VideoWriter(str(cand), cv2.VideoWriter_fourcc(*fourcc), 10.0, (1920, 1080))
                if not writer.isOpened():
                    writer.release()
                    continue
                try:
                    for fr in [blank, dimmed, blank, *(clean_zodiac for _ in range(7))]:
                        writer.write(fr)
                finally:
                    writer.release()
                cap = cv2.VideoCapture(str(cand))
                ok = cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) >= 10
                cap.release()
                if ok:
                    vid_path = cand
                    break

            self.assertIsNotNone(vid_path, "No working cv2.VideoWriter codec available")
            assert vid_path is not None
            match = detect_deck_from_video(vid_path, num_samples=10)
            self.assertIsNotNone(match)
            assert match is not None
            self.assertEqual(match.deck, Deck.ZODIAC_DECK)
            self.assertGreaterEqual(match.matched_frames, 2)

    def test_real_local_vods_smoke(self) -> None:
        samples = [
            ("assets/sample_inputs/videos/bu_erratic_deck_11-5-26.mp4", Deck.ERRATIC_DECK),
            ("assets/sample_inputs/videos/bu_anaglyph_deck_11-7-26.mp4", Deck.ANAGLYPH_DECK),
            ("assets/sample_inputs/videos/bu_magic_deck_11-9-26.mp4", Deck.MAGIC_DECK),
            ("assets/sample_inputs/videos/bu_full.mp4", Deck.BLACK_DECK),
        ]
        for rel_path, expected_deck in samples:
            vp = REPO_ROOT / rel_path
            if not vp.is_file():
                continue
            with self.subTest(video=vp.name):
                match = detect_deck_from_video(vp, num_samples=8)
                self.assertIsNotNone(match, f"Expected match for {vp.name}")
                assert match is not None
                self.assertEqual(match.deck, expected_deck)
                self.assertGreaterEqual(match.matched_frames, 2)


if __name__ == "__main__":
    unittest.main()
