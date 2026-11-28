"""Datasets and CTC encoding helpers for Balatro Seed OCR."""

from __future__ import annotations

import json
from pathlib import Path
import random

import cv2
import torch
from torch.utils.data import Dataset

from classifiers.hud_ocr.dataset import collate_ocr_batch, crop_rgb_to_tensor
from classifiers.hud_ocr.synth import (
    apply_vod_crop_augmentations,
    corrupt_hud_crop_to_empty,
    resize_and_pad_crop,
)

try:
    from .config import INPUT_HEIGHT, INPUT_WIDTH, SEED_CHAR_TO_IDX
    from .synth import BalatroSeedSynthesizer
except ImportError:
    from config import INPUT_HEIGHT, INPUT_WIDTH, SEED_CHAR_TO_IDX  # type: ignore[no-redef]
    from synth import BalatroSeedSynthesizer  # type: ignore[no-redef]


def encode_seed_text(text: str) -> list[int]:
    """Encodes an uppercase Balatro seed string into Seed OCR CTC class indices."""
    encoded: list[int] = []
    for ch in text:
        if ch not in SEED_CHAR_TO_IDX:
            raise ValueError(f"Unsupported Seed OCR character {ch!r} in text {text!r}")
        encoded.append(SEED_CHAR_TO_IDX[ch])
    return encoded


class SyntheticSeedOcrDataset(Dataset):
    """On-the-fly or deterministic-val synthetic Balatro Seed OCR dataset."""

    def __init__(
        self,
        num_samples: int,
        seed: int = 42,
        fixed: bool = False,
        augment: bool = True,
        font_path: str | Path | None = None,
    ) -> None:
        self.num_samples = int(num_samples)
        self.base_seed = int(seed)
        self.fixed = bool(fixed)
        self.augment = bool(augment)
        self.font_path = font_path
        self._synth: BalatroSeedSynthesizer | None = None
        self._cached_val: list[tuple[torch.Tensor, torch.Tensor, str]] | None = None

        if self.fixed:
            synth = self._get_synth()
            cached: list[tuple[torch.Tensor, torch.Tensor, str]] = []
            for idx in range(self.num_samples):
                rng = random.Random(self.base_seed + idx * 9973)
                img, text = synth.render_sample(rng, augment=self.augment)
                cached.append(
                    (
                        crop_rgb_to_tensor(img),
                        torch.tensor(encode_seed_text(text), dtype=torch.long),
                        text,
                    )
                )
            self._cached_val = cached

    def _get_synth(self) -> BalatroSeedSynthesizer:
        if self._synth is None:
            self._synth = BalatroSeedSynthesizer(
                font_path=self.font_path,
                target_height=INPUT_HEIGHT,
                target_width=INPUT_WIDTH,
            )
        return self._synth

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        if self._cached_val is not None:
            return self._cached_val[idx]
        rng = random.Random(self.base_seed + idx * 7919 + random.randint(0, 1_000_000))
        img, text = self._get_synth().render_sample(rng, augment=self.augment)
        return (
            crop_rgb_to_tensor(img),
            torch.tensor(encode_seed_text(text), dtype=torch.long),
            text,
        )


class ModSeedOcrDataset(Dataset):
    """Dataset reading Balatro mod-captured Seed crops (`labels/frame_*.json` + `images/frame_*_seed.png`)."""

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str = "train",
        val_fraction: float = 0.15,
        seed: int = 42,
        augment: bool = True,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.labels_dir = self.dataset_dir / "labels"
        self.images_dir = self.dataset_dir / "images"
        self.split = split
        self.augment = bool(augment)

        if not self.labels_dir.exists():
            raise FileNotFoundError(
                f"Mod Seed OCR labels directory not found: {self.labels_dir}"
            )

        label_files = sorted(self.labels_dir.glob("frame_*.json"))
        if not label_files:
            raise FileNotFoundError(
                f"No frame_*.json files found in: {self.labels_dir}"
            )

        rng = random.Random(int(seed))
        shuffled = list(label_files)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_fraction))) if len(shuffled) > 1 else 0
        if split == "val":
            selected_files = sorted(shuffled[:n_val])
        elif split == "train":
            selected_files = (
                sorted(shuffled[n_val:]) if n_val < len(shuffled) else sorted(shuffled)
            )
        elif split == "all":
            selected_files = sorted(label_files)
        else:
            raise ValueError(
                f"Unsupported split {split!r}; expected 'train', 'val', or 'all'"
            )

        self.records: list[tuple[Path, str]] = []
        for lbl_path in selected_files:
            with lbl_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            frame_idx = int(meta["frame_index"])
            raw_seed = str(meta.get("seed") or "").strip().upper()
            if any(ch not in SEED_CHAR_TO_IDX for ch in raw_seed):
                continue
            img_path = self.images_dir / f"frame_{frame_idx:05d}_seed.png"
            if not img_path.exists():
                continue
            self.records.append((img_path, raw_seed))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        img_path, text = self.records[idx]
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Failed to read mod Seed OCR crop: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.augment:
            rng = random.Random(random.randint(0, 2**31 - 1))
            if text != "" and rng.random() < 0.08:
                rgb = corrupt_hud_crop_to_empty(rgb, rng)
                text = ""
            rgb = apply_vod_crop_augmentations(rgb, rng)
        if rgb.shape[:2] != (INPUT_HEIGHT, INPUT_WIDTH):
            rgb = resize_and_pad_crop(rgb, target_h=INPUT_HEIGHT, target_w=INPUT_WIDTH)
        return (
            crop_rgb_to_tensor(rgb),
            torch.tensor(encode_seed_text(text), dtype=torch.long),
            text,
        )


__all__ = [
    "ModSeedOcrDataset",
    "SyntheticSeedOcrDataset",
    "collate_ocr_batch",
    "encode_seed_text",
]
