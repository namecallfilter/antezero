"""`matcher_gpu` against OpenCV: where it decides a template, the frame's result is OpenCV's."""

import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import cv2
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import Config
import eval_template
import main as stage1
from matcher_gpu import GpuMatcher
from template import template_loader

# Upstream's annotated frames, which are not in this repo.
FRAMES = ROOT_DIR / "assets" / "testing"


def prepared(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (image, *stage1._grayscale(image, True))


def outcome(result: eval_template.EvaluationResult) -> tuple:
    return (result.found, result.clicked, result.box, result.luminance_ratio)


class TestGpuMatcher(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = Config()
        cls.templates = [t for t in template_loader(cls.config.template_json_path, cls.config.template_img_dir, cls.config.resolution)
                         if t.px_rois]
        cls.matcher = GpuMatcher("cuda" if torch.cuda.is_available() else "cpu", cls.config)

    def compared(self, images: list[np.ndarray]) -> list[dict[str, eval_template.EvaluationResult | None]]:
        """OpenCV's result of every template on every image, after holding the matcher's to it; None where the matcher left it to OpenCV."""
        frames = [prepared(image) for image in images]
        out = []
        for (image, gray, half), matched in zip(frames, self.matcher.match(frames, self.templates)):
            results = {}
            for template in self.templates:
                opencv = eval_template.eval(template, image, gray, self.config, img_gray_half=half)
                results[template.name] = None
                if matched[template.name] is not None:
                    mine = eval_template.eval(template, image, gray, self.config, img_gray_half=half, matched=matched[template.name])
                    self.assertEqual(outcome(mine), outcome(opencv), template.name)
                    if opencv.found:
                        self.assertAlmostEqual(mine.match.confidence, opencv.match.confidence, delta=1e-6)
                    results[template.name] = opencv
            out.append(results)
        return out

    def test_a_button_drawn_on_a_screen_is_found_where_opencv_finds_it(self) -> None:
        rng = np.random.default_rng(7)
        screens = []
        for template in self.templates:
            screen = rng.integers(40, 90, size=(1080, 1920, 3), dtype=np.uint8)
            button, roi = template.images[0], template.px_rois[0]
            area = screen[roi.y_min:roi.y_min + button.shape[0], roi.x_min:roi.x_min + button.shape[1]]
            area[button[:, :, 3] > 0] = button[:, :, :3][button[:, :, 3] > 0]
            screens.append(screen)
        found = undecided = 0
        for template, results in zip(self.templates, self.compared(screens)):
            undecided += sum(result is None for result in results.values())
            own = results[template.name]
            if own is not None and own.found:
                found += 1
                self.assertEqual(own.box[:2], (template.px_rois[0].x_min, template.px_rois[0].y_min))
        self.assertGreater(found, len(self.templates) // 2)
        self.assertLess(undecided, len(self.templates))

    def test_a_flat_screen_is_left_to_opencv(self) -> None:
        """Every position of a flat area has one score, and which of them OpenCV takes is a matter of rounding."""
        results = self.compared([np.full((1080, 1920, 3), level, np.uint8) for level in (60, 128, 230)])
        self.assertTrue(any(result is None for screen in results for result in screen.values()))
        self.assertFalse(any(result.found for screen in results for result in screen.values() if result is not None))

    def test_a_black_screen_is_left_to_opencv(self) -> None:
        (results,) = self.compared([np.zeros((1080, 1920, 3), np.uint8)])
        self.assertEqual(set(results.values()), {None})

    @unittest.skipUnless(FRAMES.exists(), "upstream's annotated frames are not on this disk")
    def test_annotated_frames(self) -> None:
        with open(self.config.template_json_path, encoding="utf-8") as f:
            paths = sorted({frame["path"] for data in json.load(f).values() for frame in data.get("test_frames", [])
                            if frame["path"].startswith("assets/testing/")})
        images = [cv2.imread(str(ROOT_DIR / path)) for path in paths]
        images = [image for image in images if image is not None and image.shape == (1080, 1920, 3)]
        self.assertGreater(len(images), 50)
        decided = found = 0
        for start in range(0, len(images), 32):
            for results in self.compared(images[start:start + 32]):
                decided += sum(result is not None for result in results.values())
                found += sum(result is not None and result.found for result in results.values())
        self.assertGreater(decided, 0.8 * len(images) * len(self.templates))
        self.assertGreater(found, len(images))

    @unittest.skipUnless(FRAMES.exists(), "upstream's annotated frames are not on this disk")
    def test_a_frame_is_processed_as_without_the_matcher(self) -> None:
        key = {template.name: at for at, template in enumerate(self.templates)}
        logger = logging.getLogger(__name__)
        paths = sorted(FRAMES.glob("*/*_pressed_true.jpg"))
        images = [image for image in map(cv2.imread, map(str, paths)) if image is not None and image.shape == (1080, 1920, 3)]
        self.assertGreater(len(images), 20)
        frames = [prepared(image) for image in images[:32]]
        clicks = 0
        for at, (frame, matched) in enumerate(zip(frames, self.matcher.match(frames, self.templates))):
            video_frame = SimpleNamespace(image=frame[0], index=at, timestamp_seconds=0.0)
            alone = stage1._process_frame(video_frame, self.templates, key, self.config, logger)
            helped = stage1._process_frame(video_frame, self.templates, key, self.config, logger, (frame[1], frame[2], matched))
            self.assertEqual((alone[0], alone[3], alone[4]), (helped[0], helped[3], helped[4]))
            self.assertEqual({name: value for name, value in alone[2].items() if not name.endswith("_time")},
                             {name: value for name, value in helped[2].items() if not name.endswith("_time")})
            clicks += alone[3] is not None
        self.assertGreater(clicks, 10)


if __name__ == "__main__":
    unittest.main()
