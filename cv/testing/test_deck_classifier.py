"""Unit tests for View Deck dataset sync, preparation, sorting, matching, CardCropDataset, and auto_retrain."""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classifiers.card_classifier.stage2_classifier.dataset import (
    CardCropDataset,
    _select_deck_cards_for_frame,
)
from classifiers.card_classifier.stage2_classifier.model import (
    MultiHeadCardClassifier,
    infer_backbone_from_checkpoint as infer_card_backbone,
)
from classifiers.shop_classifier.stage2_classifier.model import (
    MultiHeadItemClassifier,
    infer_backbone_from_checkpoint as infer_shop_backbone,
)
from classifiers.augmentations import apply_gpu_batch_augmentations
from classifiers.common.dataset_sync import sync_mod_dataset
from classifiers.deck_classifier.config import resolve_view_deck_roi
from classifiers.deck_classifier.evaluate import compute_iou, match_predictions_to_gt
from classifiers.deck_classifier.pipeline import is_valid_deck_card_box, sort_deck_boxes
from classifiers.deck_classifier.stage1_detector.prepare_detector_data import prepare_detector_data
from tools.auto_retrain import resolve_images_directory


def _write_dummy_png(path: Path, width: int, height: int, fill_rgb: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), fill_rgb)
    img.save(path)


class TestDeckClassifier(unittest.TestCase):
    def test_sync_mod_dataset_nested_target_and_repo_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fake_appdata = tmp / "appdata"
            src_deck = fake_appdata / "Balatro" / "Mods" / "gamestate_dataset_cv" / "dataset_deck"
            (src_deck / "images").mkdir(parents=True, exist_ok=True)
            (src_deck / "labels").mkdir(parents=True, exist_ok=True)

            for i in range(3):
                _write_dummy_png(src_deck / "images" / f"frame_{i:05d}.png", 200, 120, (20 * i, 50, 80))
                (src_deck / "labels" / f"frame_{i:05d}.txt").write_text("0 0.5 0.5 0.1 0.2\n", encoding="utf-8")
                (src_deck / "labels" / f"frame_{i:05d}.json").write_text("[]\n", encoding="utf-8")

            nested_target = tmp / "classifiers" / "card_classifier" / "dataset" / "deck_frames"
            old_appdata = os.environ.get("APPDATA")
            os.environ["APPDATA"] = str(fake_appdata)
            try:
                copied = sync_mod_dataset(
                    source_subfolder="dataset_deck",
                    target_dataset_dir=nested_target,
                    labels_subdir_name="labels_deck",
                    clean=True,
                )
            finally:
                if old_appdata is None:
                    os.environ.pop("APPDATA", None)
                else:
                    os.environ["APPDATA"] = old_appdata

            self.assertEqual(copied, 3)
            self.assertEqual(len(list((nested_target / "images").glob("*.png"))), 3)
            self.assertEqual(len(list((nested_target / "labels_deck").glob("*.txt"))), 3)
            self.assertEqual(len(list((nested_target / "labels_json").glob("*.json"))), 3)

            # Verify repo_root inside dataset_sync points to the actual repository root
            import classifiers.common.dataset_sync as ds_mod

            computed_repo_root = Path(ds_mod.__file__).resolve().parent.parent.parent
            self.assertEqual(computed_repo_root, PROJECT_ROOT)
            self.assertTrue((computed_repo_root / "balatro_mod").exists())

    def test_prepare_detector_data_split_and_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ds_dir = tmp / "dataset"
            (ds_dir / "images").mkdir(parents=True, exist_ok=True)
            (ds_dir / "labels_deck").mkdir(parents=True, exist_ok=True)

            for idx in range(10):
                _write_dummy_png(ds_dir / "images" / f"frame_{idx:05d}.png", 300, 180, (10, 20, 30))
                lines = [f"0 {0.1 + 0.05 * k:.6f} 0.500000 0.030000 0.220000" for k in range((idx % 4) + 2)]
                (ds_dir / "labels_deck" / f"frame_{idx:05d}.txt").write_text("\n".join(lines), encoding="utf-8")

            split_dir = tmp / "data_split"
            yaml_path = tmp / "data.yaml"
            out_yaml = prepare_detector_data(
                val_ratio=0.20,
                seed=42,
                dataset_dir=ds_dir,
                processed_data_dir=split_dir,
                data_yaml_path=yaml_path,
            )
            self.assertEqual(out_yaml, yaml_path)
            self.assertTrue(yaml_path.exists())

            train_imgs = {p.name for p in (split_dir / "train" / "images").glob("*.png")}
            val_imgs = {p.name for p in (split_dir / "val" / "images").glob("*.png")}
            self.assertEqual(len(train_imgs), 8)
            self.assertEqual(len(val_imgs), 2)
            self.assertEqual(len(train_imgs & val_imgs), 0)

            cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            self.assertEqual(cfg["names"], {0: "card"})
            self.assertEqual(Path(cfg["train"]), split_dir / "train" / "images")
            self.assertEqual(Path(cfg["val"]), split_dir / "val" / "images")

    def test_sort_deck_boxes_100_cards_and_box_filter(self) -> None:
        self.assertFalse(is_valid_deck_card_box([0, 0, 11, 80]))
        self.assertFalse(is_valid_deck_card_box([0, 0, 25, 44]))
        self.assertTrue(is_valid_deck_card_box([0, 0, 12, 45]))
        self.assertTrue(is_valid_deck_card_box([100, 200, 128, 305]))

        # Build 4 rows of 25 cards = 100 boxes with mild vertical jitter, shuffle, then verify sort order
        expected_boxes: list[list[int]] = []
        rng = random.Random(123)
        for row_idx in range(4):
            y1_base = 40 + row_idx * 150
            for col_idx in range(25):
                x1 = 20 + col_idx * 34
                jitter_y = rng.randint(-4, 4)
                expected_boxes.append([x1, y1_base + jitter_y, x1 + 26, y1_base + jitter_y + 105])

        shuffled = list(expected_boxes)
        rng.shuffle(shuffled)
        sorted_boxes = sort_deck_boxes(shuffled)
        self.assertEqual(sorted_boxes, expected_boxes)
        self.assertEqual(sort_deck_boxes([]), [])

    def test_match_predictions_to_gt(self) -> None:
        gt_boxes = [
            [10, 20, 40, 120],
            [50, 20, 80, 120],
            [90, 20, 120, 120],
        ]
        pred_boxes = [
            [12, 22, 41, 119],   # matches GT 0
            [52, 21, 79, 121],   # matches GT 1
            [300, 20, 330, 120], # false positive
        ]
        self.assertGreater(compute_iou(pred_boxes[0], gt_boxes[0]), 0.70)
        matches, unmatched_p, unmatched_g = match_predictions_to_gt(pred_boxes, gt_boxes, iou_thresh=0.40)
        self.assertEqual(sorted((pi, gi) for pi, gi, _ in matches), [(0, 0), (1, 1)])
        self.assertEqual(unmatched_p, [2])
        self.assertEqual(unmatched_g, [2])

    def test_select_deck_cards_and_card_crop_dataset_mixed_sources(self) -> None:
        # 1. Verify _select_deck_cards_for_frame keeps rare variants before generic duplicates
        cards = []
        for i in range(20):
            cards.append({
                "box_pixel": [i * 20, 10, 20, 90],
                "rank": "King",
                "suit": "Hearts",
                "edition": "None",
                "enhancement": "None",
                "seal": "None",
                "debuffed": False,
                "greyed": False,
            })
        cards.append({
            "box_pixel": [420, 10, 20, 90],
            "rank": "Ace",
            "suit": "Spades",
            "edition": "Polychrome",
            "enhancement": "Glass",
            "seal": "Red",
            "debuffed": False,
            "greyed": True,
        })
        cards.append({
            "box_pixel": [440, 10, 20, 90],
            "rank": "Stone",
            "suit": "None",
            "edition": "None",
            "enhancement": "Stone",
            "seal": "None",
            "debuffed": True,
            "greyed": False,
        })
        picked = _select_deck_cards_for_frame(cards, max_crops=6, rng=random.Random(7))
        self.assertEqual(len(picked), 6)
        picked_ranks = {c["rank"] for c in picked}
        self.assertIn("Ace", picked_ranks)
        self.assertIn("Stone", picked_ranks)
        self.assertEqual(len(_select_deck_cards_for_frame(cards, max_crops=None, rng=random.Random(7))), len(cards))

        # 2. Verify CardCropDataset loads hand + shop_crops + deck_frames, keeps split disjoint,
        # crops exact pixels into RAM cache, and passes through MultiHeadCardClassifier.
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "dataset"
            (root / "images").mkdir(parents=True, exist_ok=True)
            (root / "labels_json").mkdir(parents=True, exist_ok=True)
            (root / "shop_crops" / "images").mkdir(parents=True, exist_ok=True)
            (root / "shop_crops" / "labels").mkdir(parents=True, exist_ok=True)
            (root / "deck_frames" / "images").mkdir(parents=True, exist_ok=True)
            (root / "deck_frames" / "labels_json").mkdir(parents=True, exist_ok=True)

            # Create 5 hand frames
            for i in range(5):
                _write_dummy_png(root / "images" / f"frame_{i:04d}.png", 300, 160, (40, 120, 40))
                (root / "labels_json" / f"frame_{i:04d}.json").write_text(
                    json.dumps([{
                        "box_pixel": [20, 20, 70, 100],
                        "rank": "Ace",
                        "suit": "Spades",
                        "edition": "None",
                        "enhancement": "None",
                        "seal": "None",
                        "debuffed": False,
                        "face_down": False,
                    }]),
                    encoding="utf-8",
                )

            # Create 5 shop crops
            for i in range(5):
                _write_dummy_png(root / "shop_crops" / "images" / f"crop_{i:04d}_shop_1.png", 71, 95, (150, 40, 40))
                (root / "shop_crops" / "labels" / f"crop_{i:04d}_shop_1.json").write_text(
                    json.dumps([{
                        "box_pixel": [0, 0, 71, 95],
                        "rank": "Queen",
                        "suit": "Hearts",
                        "edition": "Foil",
                        "enhancement": "Lucky",
                        "seal": "Gold",
                        "debuffed": False,
                        "face_down": False,
                    }]),
                    encoding="utf-8",
                )

            # Create 5 deck frames with distinct left-vs-right regions so we verify exact box_pixel crop
            for i in range(5):
                arr = np.zeros((200, 400, 3), dtype=np.uint8)
                arr[:, :100] = (10, 20, 200)   # card 0 region (blue)
                arr[:, 100:200] = (200, 200, 10)  # card 1 region (yellow)
                img_path = root / "deck_frames" / "images" / f"frame_{i:05d}.png"
                Image.fromarray(arr).save(img_path)

                deck_cards = [
                    {
                        "box_pixel": [5, 10, 80, 160],
                        "rank": "2",
                        "suit": "Clubs",
                        "edition": "None",
                        "enhancement": "None",
                        "seal": "None",
                        "debuffed": False,
                        "face_down": False,
                        "greyed": False,
                    },
                    {
                        "box_pixel": [105, 10, 80, 160],
                        "rank": "Jack",
                        "suit": "Diamonds",
                        "edition": "Holo",
                        "enhancement": "Steel",
                        "seal": "Blue",
                        "debuffed": True,
                        "face_down": False,
                        "greyed": True,
                    },
                ]
                for extra in range(18):
                    deck_cards.append({
                        "box_pixel": [210 + extra * 5, 10, 20, 100],
                        "rank": "2",
                        "suit": "Clubs",
                        "edition": "None",
                        "enhancement": "None",
                        "seal": "None",
                        "debuffed": False,
                        "face_down": False,
                        "greyed": False,
                    })
                (root / "deck_frames" / "labels_json" / f"frame_{i:05d}.json").write_text(
                    json.dumps(deck_cards), encoding="utf-8"
                )

            train_ds = CardCropDataset(root, split="train", val_ratio=0.20, seed=42, max_deck_crops_per_frame=8)
            val_ds = CardCropDataset(root, split="val", val_ratio=0.20, seed=42, max_deck_crops_per_frame=8)

            self.assertTrue(train_ds.uses_ram_cache)
            self.assertTrue(val_ds.uses_ram_cache)

            train_paths = {s["img_path"] for s in train_ds.samples}
            val_paths = {s["img_path"] for s in val_ds.samples}
            self.assertEqual(len(train_paths & val_paths), 0)

            self.assertIn("hand", set(train_ds.sample_sources))
            self.assertIn("shop", set(train_ds.sample_sources))
            self.assertIn("deck", set(train_ds.sample_sources))
            self.assertIn("deck", set(val_ds.sample_sources))

            # Verify exact pixel color of card 0 in deck crop (blue: R~10, G~20, B~200)
            deck_indices = [idx for idx, src in enumerate(val_ds.sample_sources) if src == "deck"]
            first_deck_img, _ = val_ds[deck_indices[0]]
            mean_rgb = first_deck_img.float().mean(dim=(1, 2)).tolist()
            self.assertGreater(mean_rgb[2], 150.0)
            self.assertLess(mean_rgb[0], 50.0)

            # Unlimited mode loads all 20 crops per frame
            unlimited_val = CardCropDataset(root, split="val", val_ratio=0.20, seed=42, max_deck_crops_per_frame=None)
            unlimited_deck_count = sum(1 for s in unlimited_val.sample_sources if s == "deck")
            self.assertEqual(unlimited_deck_count, 20)

            # Forward pass through MultiHeadCardClassifier
            model = MultiHeadCardClassifier(backbone_name="mobilenet_v3_small", pretrained=False)
            model.eval()
            batch = torch.stack([train_ds[0][0].float() / 255.0, first_deck_img.float() / 255.0], dim=0)
            out = model(batch)
            self.assertEqual(out["rank"].shape[0], 2)
            self.assertEqual(out["suit"].shape[0], 2)

    def test_auto_retrain_and_roi_config_resolution(self) -> None:
        images_dir, model_type, train_script = resolve_images_directory("deck")
        self.assertEqual(model_type, "deck")
        self.assertEqual(images_dir.parent.name, "dataset_deck")
        self.assertEqual(train_script, PROJECT_ROOT / "classifiers" / "deck_classifier" / "train.py")

        with tempfile.TemporaryDirectory() as tmpdir:
            boxes_lua = Path(tmpdir) / "boxes.lua"
            boxes_lua.write_text(
                "return {\n  rois = {\n    view_deck = {x=612.5, y=215.25, w=1105.0, h=642.5},\n  }\n}\n",
                encoding="utf-8",
            )
            roi = resolve_view_deck_roi(boxes_lua)
            self.assertEqual(roi, {"x": 612.5, "y": 215.25, "w": 1105.0, "h": 642.5})

    def test_backbone_auto_detection_and_padded_crop_jitter(self) -> None:
        # Both metadata-tagged and legacy unlabeled checkpoints resolve to exact backbone
        small_card = MultiHeadCardClassifier(backbone_name="mobilenet_v3_small", pretrained=False)
        large_card = MultiHeadCardClassifier(backbone_name="mobilenet_v3_large", pretrained=False)
        self.assertEqual(infer_card_backbone({"model": small_card.state_dict()}), "mobilenet_v3_small")
        self.assertEqual(
            infer_card_backbone({"model": large_card.state_dict(), "backbone": "mobilenet_v3_large"}),
            "mobilenet_v3_large",
        )
        self.assertEqual(infer_card_backbone({"model": large_card.state_dict()}), "mobilenet_v3_large")

        small_shop = MultiHeadItemClassifier(backbone_name="mobilenet_v3_small", pretrained=False)
        large_shop = MultiHeadItemClassifier(backbone_name="mobilenet_v3_large", pretrained=False)
        self.assertEqual(infer_shop_backbone({"model_state_dict": small_shop.state_dict()}), "mobilenet_v3_small")
        self.assertEqual(infer_shop_backbone({"model_state_dict": large_shop.state_dict()}), "mobilenet_v3_large")

        # Padded crop batch -> exact target (CROP_HEIGHT=192, CROP_WIDTH=128) in train and eval modes
        padded = torch.randint(0, 256, (4, 3, 223, 148), dtype=torch.uint8)
        out_train = apply_gpu_batch_augmentations(
            padded,
            is_train=True,
            compression_prob=0.0,
            cursor_prob=0.0,
            color_jitter=False,
            geom_prob=0.5,
            pad_crop_size=(192, 128),
        )
        out_val = apply_gpu_batch_augmentations(padded, is_train=False, pad_crop_size=(192, 128))
        self.assertEqual(tuple(out_train.shape), (4, 3, 192, 128))
        self.assertEqual(tuple(out_val.shape), (4, 3, 192, 128))


if __name__ == "__main__":
    unittest.main()
