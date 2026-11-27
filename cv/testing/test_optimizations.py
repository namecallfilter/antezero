from pathlib import Path
from types import SimpleNamespace
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import Config, _default_max_workers, get_config
import eval_template
import frame
from geometry import ImageSize, PixelROI
from matcher import find_element_coarse_to_fine
from template import template_loader


class TestCompiledSpecsAndCoarseToFine(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = Config()
        cls.config.debug_plot = False
        cls.config.setup_mode = False
        cls.templates = template_loader(
            cls.config.template_json_path,
            cls.config.template_img_dir,
            cls.config.resolution,
        )
        cls.template_dict = {t.name: t for t in cls.templates}
        with open(cls.config.template_json_path, "r", encoding="utf-8") as f:
            cls.templates_data = json.load(f)

    def test_compiled_specs_caching_and_bounds(self) -> None:
        tpl = self.templates[0]
        specs_a = tpl.get_compiled_specs(1920, 1080, self.config.expand_roi_factor)
        specs_b = tpl.get_compiled_specs(1920, 1080, self.config.expand_roi_factor)
        self.assertIs(specs_a, specs_b)
        self.assertGreater(len(specs_a), 0)

        for spec in specs_a:
            self.assertGreaterEqual(spec.x1, 0)
            self.assertGreaterEqual(spec.y1, 0)
            self.assertLessEqual(spec.x2, 1920)
            self.assertLessEqual(spec.y2, 1080)
            self.assertLess(spec.x1, spec.x2)
            self.assertLess(spec.y1, spec.y2)
            self.assertTrue(spec.image_bgr.flags["C_CONTIGUOUS"])
            self.assertTrue(spec.image_gray.flags["C_CONTIGUOUS"])
            self.assertTrue(spec.image_gray_half.flags["C_CONTIGUOUS"])
            if spec.can_coarse:
                self.assertEqual(spec.x1 % 2, 0)
                self.assertEqual(spec.y1 % 2, 0)

        specs_720 = tpl.get_compiled_specs(1280, 720, self.config.expand_roi_factor)
        self.assertIsNot(specs_a, specs_720)
        for spec in specs_720:
            self.assertLessEqual(spec.x2, 1280)
            self.assertLessEqual(spec.y2, 720)

    def test_coarse_to_fine_matches_full_res_on_annotated_frames(self) -> None:
        cfg_full = Config(coarse_to_fine=False, debug_plot=False, setup_mode=False)
        cfg_coarse = Config(coarse_to_fine=True, coarse_margin=0.04, debug_plot=False, setup_mode=False)

        checked = 0
        for template_name, data in self.templates_data.items():
            test_frames = data.get("test_frames", [])
            if not test_frames:
                continue
            tpl = self.template_dict[template_name]
            for tf in test_frames:
                img_path = ROOT_DIR / tf["path"]
                img = cv2.imread(str(img_path))
                self.assertIsNotNone(img, f"Failed to load {img_path}")
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_gray_half = cv2.resize(
                    img_gray, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_NEAREST
                )

                res_full = eval_template.eval(tpl, img, img_gray, cfg_full)
                res_coarse = eval_template.eval(
                    tpl, img, img_gray, cfg_coarse, img_gray_half=img_gray_half
                )
                self.assertEqual(
                    res_coarse.found,
                    res_full.found,
                    f"{template_name} ({tf['path']}) found mismatch",
                )
                self.assertEqual(
                    res_coarse.clicked,
                    res_full.clicked,
                    f"{template_name} ({tf['path']}) clicked mismatch",
                )
                checked += 1

        self.assertGreater(checked, 0)

    def test_find_element_coarse_to_fine_direct(self) -> None:
        rng = np.random.default_rng(42)
        bg = rng.integers(10, 60, size=(120, 160, 3), dtype=np.uint8)
        patch = rng.integers(150, 250, size=(32, 48, 3), dtype=np.uint8)
        bg[40:72, 52:100] = patch

        bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        bg_half = cv2.resize(bg_gray, (80, 60), interpolation=cv2.INTER_NEAREST)
        patch_half = cv2.resize(patch_gray, (24, 16), interpolation=cv2.INTER_NEAREST)

        roi = PixelROI(1, 1, 159, 119)
        tsize = ImageSize(48, 32)
        match, (dx, dy) = find_element_coarse_to_fine(
            bg_gray,
            bg,
            patch_gray,
            patch,
            None,
            bg_half,
            patch_half,
            None,
            tsize,
            cv2.TM_CCOEFF_NORMED,
            roi,
            0.85,
            coarse_margin=0.05,
        )
        self.assertTrue(match.found)
        self.assertGreater(match.confidence, 0.99)
        self.assertEqual((match.top_left[0] + dx, match.top_left[1] + dy), (52, 40))

        # Unrelated uniform image should fast-reject at coarse stage cleanly.
        blank = np.zeros_like(bg)
        blank_gray = np.zeros_like(bg_gray)
        blank_half = np.zeros_like(bg_half)
        miss, _ = find_element_coarse_to_fine(
            blank_gray,
            blank,
            patch_gray,
            patch,
            None,
            blank_half,
            patch_half,
            None,
            tsize,
            cv2.TM_CCOEFF_NORMED,
            roi,
            0.85,
        )
        self.assertFalse(miss.found)
        self.assertIsNone(miss.score_map)


class TestVideoReaderOptimizations(unittest.TestCase):
    def test_video_reader_nvdec_and_opencv_parity_on_bu_full(self) -> None:
        video_path = ROOT_DIR / "assets" / "sample_inputs" / "videos" / "bu_full.mp4"
        if not video_path.exists():
            self.skipTest(f"Sample video not present: {video_path}")

        res = ImageSize(1920, 1080)
        with frame.VideoReader(
            video_path,
            res,
            skip_frames=2,
            start_timestamp_seconds=14.0,
            end_timestamp_seconds=15.0,
            use_hwaccel=True,
        ) as hw_reader:
            self.assertIsNotNone(hw_reader._ffmpeg_proc)
            hw_frames = list(hw_reader)
        self.assertIsNone(hw_reader._ffmpeg_proc)

        with frame.VideoReader(
            video_path,
            res,
            skip_frames=2,
            start_timestamp_seconds=14.0,
            end_timestamp_seconds=15.0,
            use_hwaccel=False,
        ) as cv_reader:
            self.assertIsNone(cv_reader._ffmpeg_proc)
            cv_frames = list(cv_reader)

        self.assertEqual(len(hw_frames), 20)
        self.assertEqual([f.index for f in hw_frames], [f.index for f in cv_frames])
        mean_diff = float(np.mean(np.abs(hw_frames[0].image.astype(np.int16) - cv_frames[0].image.astype(np.int16))))
        self.assertLess(mean_diff, 5.0)

    def test_video_reader_hwaccel_first_frame_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            vid_path = Path(tmpdir) / "tiny.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(vid_path), fourcc, 30.0, (160, 120))
            self.assertTrue(writer.isOpened())
            try:
                for i in range(20):
                    img = np.full((120, 160, 3), i * 10, dtype=np.uint8)
                    writer.write(img)
            finally:
                writer.release()

            fake_proc = SimpleNamespace(
                stdout=SimpleNamespace(read=lambda _n: b"", close=lambda: None),
                poll=lambda: 1,
                terminate=lambda: None,
                kill=lambda: None,
                wait=lambda timeout=None: 1,
            )

            def _inject_failing_ffmpeg(self_reader, **_kwargs):
                self_reader._nv12_bytes = (160 * 120 * 3) // 2
                self_reader._ffmpeg_proc = fake_proc

            with patch.object(frame.VideoReader, "_start_ffmpeg_nvdec", _inject_failing_ffmpeg):
                frames = list(
                    frame.iter_video_frames(
                        vid_path,
                        ImageSize(160, 120),
                        skip_frames=0,
                        use_hwaccel=True,
                    )
                )

            self.assertEqual(len(frames), 20)
            self.assertEqual([f.index for f in frames], list(range(20)))

    def test_video_reader_rejects_wrong_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            vid_path = Path(tmpdir) / "wrong_res.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(vid_path), fourcc, 30.0, (160, 120))
            self.assertTrue(writer.isOpened())
            try:
                writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
            finally:
                writer.release()

            with self.assertRaises(ValueError):
                frame.VideoReader(vid_path, ImageSize(320, 240), use_hwaccel=True)


class TestConfigOptimizationFlags(unittest.TestCase):
    def test_config_cli_flags_and_default_workers(self) -> None:
        default_workers = _default_max_workers()
        self.assertGreaterEqual(default_workers, 1)
        self.assertLessEqual(default_workers, 8)

        argv = [
            "main.py",
            "--no-use-hwaccel",
            "--no-coarse-to-fine",
            "--coarse-margin",
            "0.06",
            "--max-workers",
            "4",
        ]
        with patch.object(sys, "argv", argv):
            cfg = get_config()

        self.assertFalse(cfg.use_hwaccel)
        self.assertFalse(cfg.coarse_to_fine)
        self.assertAlmostEqual(cfg.coarse_margin, 0.06)
        self.assertEqual(cfg.max_workers, 4)


if __name__ == "__main__":
    unittest.main()
