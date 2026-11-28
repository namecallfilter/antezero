"""Unit tests for classifiers.seed_ocr (vocab, CTC decoders, synth, dataset, training, Copy localization)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import tempfile
import unittest

import cv2
import torch

from classifiers.hud_ocr.model import HudOcrCRNN
from classifiers.seed_ocr.config import (
    INPUT_HEIGHT,
    INPUT_WIDTH,
    SEED_ALPHABET,
    SEED_CHAR_TO_IDX,
    SEED_NUM_CLASSES,
)
from classifiers.seed_ocr.dataset import (
    ModSeedOcrDataset,
    SyntheticSeedOcrDataset,
    collate_ocr_batch,
    encode_seed_text,
)
from classifiers.seed_ocr.pipeline import (
    SeedOcrPipeline,
    crop_seed_from_frame,
    ctc_beam_decode_seed,
    ctc_greedy_decode_seed,
    detect_copy_button_box,
    is_valid_seed,
)
from classifiers.seed_ocr.synth import BalatroSeedSynthesizer
from classifiers.seed_ocr.train import train_seed_ocr


class TestSeedOcr(unittest.TestCase):
    def test_alphabet_decoders_and_duplicate_collapse(self) -> None:
        self.assertEqual(len(SEED_ALPHABET), 34)
        self.assertNotIn("0", SEED_ALPHABET)
        self.assertNotIn("O", SEED_ALPHABET)
        self.assertTrue(is_valid_seed("LX6LFV2Q"))
        self.assertFalse(is_valid_seed("LX6LFV20"))  # '0' not in Balatro seed alphabet
        self.assertFalse(is_valid_seed("LX6LFVOQ"))  # 'O' not in Balatro seed alphabet
        self.assertFalse(is_valid_seed("SHORT"))

        # Verify explicit CTC blank separation for repeated consecutive characters ("55SS11LL")
        # Sequence timesteps: 5, <blank>, 5, S, <blank>, S, 1, <blank>, 1, L, <blank>, L
        target = "55SS11LL"
        ids = encode_seed_text(target)
        log_probs = torch.full((1, 14, SEED_NUM_CLASSES), -20.0)
        seq = [
            ids[0], 0, ids[1],
            ids[2], 0, ids[3],
            ids[4], 0, ids[5],
            ids[6], 0, ids[7],
            0, 0,
        ]
        for t, cid in enumerate(seq):
            log_probs[0, t, cid] = -0.02

        greedy = ctc_greedy_decode_seed(log_probs)
        self.assertEqual(greedy[0][0], target)
        self.assertGreater(greedy[0][1], 0.95)

        beams = ctc_beam_decode_seed(log_probs, beam_width=6, top_k=3)
        self.assertGreaterEqual(len(beams[0]), 1)
        self.assertEqual(beams[0][0][0], target)

    def test_synth_mod_dataset_and_1epoch_roundtrip(self) -> None:
        synth = BalatroSeedSynthesizer()
        rng = random.Random(101)
        crop, txt = synth.render_sample(rng, augment=False)
        self.assertEqual(crop.shape, (INPUT_HEIGHT, INPUT_WIDTH, 3))
        self.assertTrue(is_valid_seed(txt))

        ds_synth = SyntheticSeedOcrDataset(num_samples=8, seed=7, fixed=True, augment=False)
        imgs, targets, target_lengths, texts = collate_ocr_batch([ds_synth[i] for i in range(4)])
        self.assertEqual(tuple(imgs.shape), (4, 3, INPUT_HEIGHT, INPUT_WIDTH))
        self.assertEqual(int(target_lengths.sum().item()), int(targets.numel()))
        self.assertEqual(len(texts), 4)

        model = HudOcrCRNN(num_classes=SEED_NUM_CLASSES)
        out = model(imgs)
        self.assertEqual(tuple(out.shape), (4, 48, SEED_NUM_CLASSES))

        with tempfile.TemporaryDirectory() as tmpdir:
            mod_dir = Path(tmpdir) / "dataset_seed_ocr"
            (mod_dir / "images").mkdir(parents=True, exist_ok=True)
            (mod_dir / "labels").mkdir(parents=True, exist_ok=True)
            for idx in range(6):
                lbl_seed = "" if idx == 5 else "Z1DCPJSE"
                sample_rgb, _ = synth.render_sample(
                    rng, text_override=(lbl_seed or ""), augment=False
                )
                cv2.imwrite(
                    str(mod_dir / "images" / f"frame_{idx:05d}_seed.png"),
                    cv2.cvtColor(sample_rgb, cv2.COLOR_RGB2BGR),
                )
                with (mod_dir / "labels" / f"frame_{idx:05d}.json").open(
                    "w", encoding="utf-8"
                ) as f:
                    json.dump(
                        {
                            "frame_index": idx,
                            "phase": "win" if idx % 2 == 0 else "game_over",
                            "seeded": False,
                            "seed": lbl_seed,
                        },
                        f,
                    )

            mod_train = ModSeedOcrDataset(mod_dir, split="train", seed=42, augment=True)
            mod_val = ModSeedOcrDataset(mod_dir, split="val", seed=42, augment=False)
            self.assertEqual(len(mod_train) + len(mod_val), 6)

            weights_path = Path(tmpdir) / "seed_ocr_test.pt"
            args = argparse.Namespace(
                epochs=1,
                batch_size=8,
                train_samples=16,
                val_samples=8,
                lr=1e-3,
                weight_decay=1e-4,
                rnn_hidden=64,
                rnn_layers=1,
                dropout=0.1,
                num_workers=0,
                seed=11,
                device="cpu",
                no_amp=True,
                font_path=None,
                dataset_dir=str(mod_dir),
                synth_only=False,
                synth_samples=8,
                weights_out=str(weights_path),
                runs_dir=str(Path(tmpdir) / "runs"),
            )
            saved = train_seed_ocr(args)
            self.assertTrue(saved.exists())

            pipe = SeedOcrPipeline(weights_path=saved, device="cpu", batch_size=4)
            preds = pipe.predict_crops([crop, crop], beam_top_k=3)
            self.assertEqual(len(preds), 2)
            self.assertIsInstance(preds[0].text, str)
            self.assertGreaterEqual(preds[0].confidence, 0.0)

    def test_copy_button_detection_on_real_vod_win_frame(self) -> None:
        video_path = Path("assets/sample_inputs/videos/bu_anaglyph_deck_11-7-26.mp4")
        if not video_path.exists():
            self.skipTest("Sample VOD video not available locally")

        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 140650)
        ok, frame_bgr = cap.read()
        cap.release()
        self.assertTrue(ok)

        copy_box = detect_copy_button_box(frame_bgr)
        self.assertIsNotNone(copy_box)
        bx, by, bw, bh = copy_box
        self.assertAlmostEqual(bx, 976, delta=25)
        self.assertAlmostEqual(by, 764, delta=25)
        self.assertGreater(bw, 140)
        self.assertLess(bh, 60)

        crop_rgb, seed_box, used_anchor = crop_seed_from_frame(frame_bgr)
        self.assertTrue(used_anchor)
        self.assertEqual(seed_box[2], 173)
        self.assertEqual(seed_box[3], 58)
        self.assertGreater((crop_rgb.mean(axis=2) > 200).sum(), 200)


if __name__ == "__main__":
    unittest.main()
