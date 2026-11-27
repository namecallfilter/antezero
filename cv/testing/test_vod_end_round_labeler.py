"""Unit tests for the bounded video scrubber and batch VOD end-round labeler."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classifiers.end_round_classifier.config import IGNORE_INDEX, TARGET_FPS
from classifiers.end_round_classifier.schema import ClipMeta, build_frame_targets
from config import Config
import main as main_module
from tools import assign_test_images
from tools.label_end_round_vods import (
    AnnotationManifest,
    EndRoundCandidate,
    discover_videos,
    extract_play_hand_cash_out_candidates,
    filter_pending_candidates,
    load_or_scan_video_actions,
    main as labeler_main,
    probe_video_metadata,
    save_real_vod_clip,
)
from tools.video_scrubber import (
    FrameRange,
    decode_scrubber_key,
    draw_hud_panel,
    draw_segment_timeline,
    format_video_timestamp,
)


def _write_synthetic_video(
    path: Path,
    num_frames: int = 120,
    fps: float = 30.0,
    width: int = 320,
    height: int = 180,
) -> Path:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")
    try:
        for i in range(num_frames):
            frame = np.full((height, width, 3), (i % 200, 40, 80), dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return path


class TestVideoScrubberPrimitives(unittest.TestCase):
    def test_frame_range_clamping_and_mapping(self):
        fr = FrameRange(2850, 3297)
        self.assertEqual(fr.span, 447)
        self.assertEqual(fr.num_frames, 448)
        self.assertEqual(fr.trackbar_max, 447)
        self.assertEqual(fr.to_rel(2800), 0)
        self.assertEqual(fr.to_rel(2850), 0)
        self.assertEqual(fr.to_rel(3000), 150)
        self.assertEqual(fr.to_rel(3400), 447)
        self.assertEqual(fr.to_abs(-5), 2850)
        self.assertEqual(fr.to_abs(150), 3000)
        self.assertEqual(fr.to_abs(999), 3297)

        single = FrameRange(100, 100)
        self.assertEqual(single.span, 0)
        self.assertEqual(single.trackbar_max, 1)

        with self.assertRaises(ValueError):
            FrameRange(200, 199)

    def test_decode_scrubber_key_and_timestamps(self):
        self.assertEqual(decode_scrubber_key(-1), "")
        self.assertEqual(decode_scrubber_key(13), "enter")
        self.assertEqual(decode_scrubber_key(10), "enter")
        self.assertEqual(decode_scrubber_key(27), "esc")
        self.assertEqual(decode_scrubber_key(32), "space")
        self.assertEqual(decode_scrubber_key(ord("A")), "a")
        self.assertEqual(decode_scrubber_key(2424832), "left")
        self.assertEqual(decode_scrubber_key(2555904), "right")
        self.assertEqual(decode_scrubber_key(2490368), "up")
        self.assertEqual(decode_scrubber_key(2621440), "down")

        self.assertEqual(format_video_timestamp(0, 60.0), "00:00.000")
        self.assertEqual(format_video_timestamp(3600, 60.0), "01:00.000")
        self.assertEqual(format_video_timestamp(216000, 60.0), "01:00:00.000")

    def test_overlay_helpers_mutate_canvas_cleanly(self):
        canvas = np.zeros((360, 640, 3), dtype=np.uint8)
        fr = FrameRange(100, 300)
        draw_hud_panel(canvas, [("Line 1", (0, 255, 0)), ("Line 2", (255, 255, 255))])
        draw_segment_timeline(
            canvas,
            fr,
            200,
            markers=[(100, (0, 255, 0), "PLAY"), (300, (0, 0, 255), "CASH")],
        )
        self.assertGreater(int(canvas.sum()), 0)

    def test_assign_test_images_module_imports_cleanly(self):
        self.assertTrue(callable(assign_test_images.main))


class TestCandidateExtractionAndCaching(unittest.TestCase):
    def test_pairs_final_play_hand_before_each_cash_out(self):
        # Mirrors bu_full sample structure: click bursts + multi-hand blind + unmatched clicks.
        actions_payload = {
            "key": {
                "Buy Card": 1,
                "Cash Out": 2,
                "Play Hand": 6,
                "Select Blind": 11,
            },
            "frames": {
                "100": 2,    # orphan Cash Out before any Play Hand
                "2472": 6,   # hand 1 click burst (lost / non-winning hand)
                "2475": 6,
                "2478": 6,
                "2847": 6,   # hand 2 click burst (blind-clearing hand)
                "2850": 6,
                "3294": 2,   # Cash Out click burst
                "3297": 2,
                "3400": 2,   # consecutive Cash Out with no intervening Play Hand
                "4440": 6,   # next round Play Hand
                "4443": 6,
                "4788": 2,   # next round Cash Out
                "4791": 2,
                "5100": 6,   # trailing Play Hand without Cash Out yet
            },
        }
        cands = extract_play_hand_cash_out_candidates(
            actions_payload,
            video_path=Path("assets/sample_inputs/videos/bu_full.mp4"),
            fps=60.0,
            total_frames=10000,
            width=1920,
            height=1080,
        )
        self.assertEqual(len(cands), 2)
        self.assertEqual((cands[0].play_hand_frame, cands[0].cash_out_frame), (2850, 3297))
        self.assertEqual((cands[0].range_start, cands[0].range_end), (2850, 3297))
        self.assertEqual((cands[1].play_hand_frame, cands[1].cash_out_frame), (4443, 4791))

    def test_padding_and_max_window_filter(self):
        actions_payload = {
            "key": {"Cash Out": 2, "Play Hand": 6},
            "frames": {
                "60": 6,
                "360": 2,     # 5.0s window -> keep with padding
                "1000": 6,
                "9000": 2,    # 133.3s window -> dropped by max_window_sec=60
            },
        }
        cands = extract_play_hand_cash_out_candidates(
            actions_payload,
            video_path=Path("vod_a.mp4"),
            fps=60.0,
            total_frames=10000,
            width=1920,
            height=1080,
            pad_before_sec=0.5,
            pad_after_sec=0.25,
            max_window_sec=60.0,
        )
        self.assertEqual(len(cands), 1)
        self.assertEqual(cands[0].range_start, 30)
        self.assertEqual(cands[0].range_end, 375)
        self.assertEqual(cands[0].play_hand_frame, 60)
        self.assertEqual(cands[0].cash_out_frame, 360)

    def test_discover_videos_and_action_cache_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vdir = root / "videos"
            cache_dir = root / "cache"
            vdir.mkdir()
            (vdir / "notes.txt").write_text("ignore", encoding="utf-8")
            vid_b = _write_synthetic_video(vdir / "b_vod.mp4", num_frames=40, fps=30.0)
            vid_a = _write_synthetic_video(vdir / "a_vod.MP4", num_frames=40, fps=30.0)

            discovered = discover_videos(vdir)
            self.assertEqual([p.name for p in discovered], ["a_vod.MP4", "b_vod.mp4"])

            cfg = Config(input_video_path=str(vid_a), start_timestamp=0.0, end_timestamp=None, skip_frames=2)
            fake_payload = {"key": {"Play Hand": 0, "Cash Out": 1}, "frames": {"5": 0, "30": 1}}

            def fake_detect(
                c,
                write_output=True,
                output_json_path=None,
                templates=None,
                extra_metadata=None,
                progress_desc="",
            ):
                payload = {**fake_payload, "meta": extra_metadata}
                if output_json_path:
                    p = Path(output_json_path)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(json.dumps(payload), encoding="utf-8")
                return payload, output_json_path

            with patch("tools.label_end_round_vods.detect_video_actions", side_effect=fake_detect) as mock_detect:

                # First call runs Pass 1 and populates cache
                res1 = load_or_scan_video_actions(
                    vid_a, actions_cache_dir=cache_dir, base_config=cfg
                )
                self.assertEqual(mock_detect.call_count, 1)
                self.assertEqual(res1["frames"], fake_payload["frames"])

                # Second call with identical config hits cache without calling detector
                res2 = load_or_scan_video_actions(
                    vid_a, actions_cache_dir=cache_dir, base_config=cfg
                )
                self.assertEqual(mock_detect.call_count, 1)
                self.assertEqual(res2["frames"], fake_payload["frames"])

                # Reprocess forces scan
                load_or_scan_video_actions(
                    vid_a, actions_cache_dir=cache_dir, base_config=cfg, reprocess=True
                )
                self.assertEqual(mock_detect.call_count, 2)

            # `--skip-scan` returns None for uncached video b_vod
            self.assertIsNone(
                load_or_scan_video_actions(
                    vid_b, actions_cache_dir=cache_dir, base_config=cfg, skip_scan=True
                )
            )


class TestClipSaveManifestAndCli(unittest.TestCase):
    def test_save_real_vod_clip_and_manifest_resume_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vdir = root / "videos"
            out_dir = root / "real_vod"
            vdir.mkdir()
            vid = _write_synthetic_video(vdir / "sample_run.mp4", num_frames=90, fps=30.0)

            fps, total_frames, width, height = probe_video_metadata(vid)
            self.assertAlmostEqual(fps, 30.0, places=1)
            self.assertEqual((width, height), (320, 180))
            self.assertGreaterEqual(total_frames, 85)

            cand1 = EndRoundCandidate(
                clip_id="sample_run_f0000010_0000050",
                video_path=vid,
                play_hand_frame=10,
                cash_out_frame=50,
                range_start=10,
                range_end=50,
                fps=fps,
                width=width,
                height=height,
            )
            cand2 = EndRoundCandidate(
                clip_id="sample_run_f0000060_0000080",
                video_path=vid,
                play_hand_frame=60,
                cash_out_frame=80,
                range_start=60,
                range_end=80,
                fps=fps,
                width=width,
                height=height,
            )

            meta_path = save_real_vod_clip(cand1, marked_abs_frame=35, output_dir=out_dir)
            loaded = ClipMeta.from_json(meta_path)
            self.assertEqual(loaded.source, "real_vod")
            self.assertEqual(loaded.frame_format, "video")
            self.assertFalse(loaded.already_cropped)
            self.assertEqual(loaded.source_start_frame, 10)
            self.assertEqual(loaded.source_end_frame, 51)
            self.assertEqual(loaded.num_frames, 41)
            self.assertEqual(loaded.events.play_hand_frame, 0)
            self.assertEqual(loaded.events.end_round_frame, 25)
            resolved_payload = (meta_path.parent / loaded.payload_file).resolve()
            self.assertEqual(resolved_payload, vid.resolve())

            # Verify target generation produces valid sparse real_vod labels & Gaussian peak
            resampled_len = 21
            states, boundary, mapped = build_frame_targets(
                num_frames=resampled_len,
                events=loaded.events,
                src_fps=loaded.fps,
                dst_fps=TARGET_FPS,
            )
            self.assertIn(IGNORE_INDEX, states)
            self.assertAlmostEqual(float(boundary[mapped["end_round_frame"]]), 1.0, places=4)

            # Verify disk recovery + skip behavior in AnnotationManifest
            manifest = AnnotationManifest(out_dir / "_annotation_manifest.json", out_dir)
            self.assertEqual(manifest.status_of(cand1.clip_id), "labeled")
            self.assertEqual(manifest.saved_frame_of(cand1.clip_id), 35)
            self.assertEqual(manifest.status_of(cand2.clip_id), "pending")

            pending = filter_pending_candidates([cand1, cand2], manifest)
            self.assertEqual([c.clip_id for c in pending], [cand2.clip_id])
            self.assertEqual(
                len(filter_pending_candidates([cand1, cand2], manifest, review_all=True)),
                2,
            )

            manifest.mark_skipped(cand1)
            self.assertFalse(meta_path.exists())
            self.assertEqual(manifest.status_of(cand1.clip_id), "skipped")

    def test_cli_stats_and_refactored_detect_video_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vid = _write_synthetic_video(
                root / "clip_1080p.mp4",
                num_frames=12,
                fps=30.0,
                width=1920,
                height=1080,
            )
            actions_json = root / "actions.json"
            actions_json.write_text(
                json.dumps(
                    {
                        "key": {"Play Hand": 0, "Cash Out": 1},
                        "frames": {"2": 0, "10": 1},
                    }
                ),
                encoding="utf-8",
            )
            out_dir = root / "real_vod"

            rc = labeler_main(
                [
                    "--video",
                    str(vid),
                    "--actions-json",
                    str(actions_json),
                    "--output-dir",
                    str(out_dir),
                    "--stats",
                ]
            )
            self.assertEqual(rc, 0)

            # Verify main.detect_video_actions returns schema-compatible payload cleanly
            cfg = Config(
                input_video_path=str(vid),
                output_dir=str(root / "main_out"),
                start_timestamp=0.0,
                end_timestamp=0.4,
                skip_frames=4,
                run_benchmark=False,
            )
            payload, out_path = main_module.detect_video_actions(cfg, write_output=True)
            self.assertIn("key", payload)
            self.assertIn("frames", payload)
            self.assertIsNotNone(out_path)
            self.assertTrue(Path(out_path).exists())

            # Verify resolution mismatch raises cleanly instead of dying silently in prefetch thread
            small_vid = _write_synthetic_video(
                root / "clip_small.mp4", num_frames=5, fps=30.0, width=320, height=180
            )
            bad_cfg = Config(
                input_video_path=str(small_vid),
                output_dir=str(root / "bad_out"),
                start_timestamp=0.0,
                end_timestamp=None,
                skip_frames=2,
                run_benchmark=False,
            )
            with self.assertRaises(ValueError):
                main_module.detect_video_actions(bad_cfg, write_output=False)


if __name__ == "__main__":
    unittest.main()
