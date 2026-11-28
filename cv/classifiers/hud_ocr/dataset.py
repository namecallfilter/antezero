"""Datasets and CTC batch collation for Balatro HUD OCR."""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .config import CHAR_TO_IDX, INPUT_HEIGHT, INPUT_WIDTH, NORM_MEAN, NORM_STD
    from .model import encode_text
    from .synth import (
        BalatroHudSynthesizer,
        apply_vod_crop_augmentations,
        corrupt_hud_crop_to_empty,
        resize_and_pad_crop,
    )
except ImportError:
    from config import CHAR_TO_IDX, INPUT_HEIGHT, INPUT_WIDTH, NORM_MEAN, NORM_STD  # type: ignore[no-redef]
    from model import encode_text  # type: ignore[no-redef]
    from synth import (  # type: ignore[no-redef]
        BalatroHudSynthesizer,
        apply_vod_crop_augmentations,
        corrupt_hud_crop_to_empty,
        resize_and_pad_crop,
    )


def crop_rgb_to_tensor(crop_rgb: np.ndarray) -> torch.Tensor:
    """Convert `(H, W, 3)` uint8 RGB image to normalized `(3, H, W)` float32 tensor."""
    arr = crop_rgb.astype(np.float32) / 255.0
    mean = np.array(NORM_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(NORM_STD, dtype=np.float32).reshape(1, 1, 3)
    arr = (arr - mean) / std
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))


class SyntheticHudOcrDataset(Dataset):
    """Deterministic-seed or random-epoch synthetic HUD OCR dataset."""

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
        self._synth: BalatroHudSynthesizer | None = None
        self._cached_val: list[tuple[torch.Tensor, torch.Tensor, str]] | None = None

        if self.fixed:
            synth = self._get_synth()
            cached: list[tuple[torch.Tensor, torch.Tensor, str]] = []
            for idx in range(self.num_samples):
                rng = random.Random(self.base_seed + idx * 9973)
                img, text, _ = synth.render_sample(rng, augment=self.augment)
                tensor_img = crop_rgb_to_tensor(img)
                target = torch.tensor(encode_text(text), dtype=torch.long)
                cached.append((tensor_img, target, text))
            self._cached_val = cached

    def _get_synth(self) -> BalatroHudSynthesizer:
        if self._synth is None:
            self._synth = BalatroHudSynthesizer(
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
        img, text, _ = self._get_synth().render_sample(rng, augment=self.augment)
        return crop_rgb_to_tensor(img), torch.tensor(encode_text(text), dtype=torch.long), text


class ManifestHudOcrDataset(Dataset):
    """Disk-backed OCR dataset reading `manifest.jsonl` (`{"image": ..., "text": ...}`)."""

    def __init__(self, dataset_dir: str | Path, split: str | None = None) -> None:
        self.dataset_dir = Path(dataset_dir)
        manifest_path = self.dataset_dir / "manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"OCR manifest not found: {manifest_path}")

        self.records: list[tuple[Path, str]] = []
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if split is not None and row.get("split") != split:
                    continue
                img_path = self.dataset_dir / row["image"]
                self.records.append((img_path, str(row["text"])))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        img_path, text = self.records[idx]
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Failed to read OCR sample image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (INPUT_HEIGHT, INPUT_WIDTH):
            rgb = resize_and_pad_crop(rgb, target_h=INPUT_HEIGHT, target_w=INPUT_WIDTH)
        return crop_rgb_to_tensor(rgb), torch.tensor(encode_text(text), dtype=torch.long), text


class ModHudOcrDataset(Dataset):
    """Dataset reading real mod-captured crops (`labels/frame_*.json` + `images/frame_*_<field>.png`).

    Splits strictly at the frame level so crops from the same frame never leak between
    training and validation. Applies VOD-realistic degradations when `augment=True`.
    """

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
            raise FileNotFoundError(f"Mod HUD OCR labels directory not found: {self.labels_dir}")

        label_files = sorted(self.labels_dir.glob("frame_*.json"))
        if not label_files:
            raise FileNotFoundError(f"No frame_*.json files found in: {self.labels_dir}")

        rng = random.Random(int(seed))
        shuffled = list(label_files)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_fraction))) if len(shuffled) > 1 else 0
        if split == "val":
            selected_files = sorted(shuffled[:n_val])
        elif split == "train":
            selected_files = sorted(shuffled[n_val:]) if n_val < len(shuffled) else sorted(shuffled)
        elif split == "all":
            selected_files = sorted(label_files)
        else:
            raise ValueError(f"Unsupported split {split!r}; expected 'train', 'val', or 'all'")

        self.records: list[tuple[Path, str, str]] = []
        for lbl_path in selected_files:
            with lbl_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            frame_idx = int(meta["frame_index"])
            fields = meta.get("fields", {})
            for field_name, raw_text in fields.items():
                img_path = self.images_dir / f"frame_{frame_idx:05d}_{field_name}.png"
                if not img_path.exists():
                    continue
                cleaned = str(raw_text or "").strip().replace("E", "e")
                if any(ch not in CHAR_TO_IDX for ch in cleaned):
                    continue
                self.records.append((img_path, cleaned, field_name))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        img_path, text, _field_name = self.records[idx]
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Failed to read mod HUD OCR crop: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.augment:
            rng = random.Random(random.randint(0, 2**31 - 1))
            if text != "" and rng.random() < 0.10:
                rgb = corrupt_hud_crop_to_empty(rgb, rng)
                text = ""
            rgb = apply_vod_crop_augmentations(rgb, rng)
        if rgb.shape[:2] != (INPUT_HEIGHT, INPUT_WIDTH):
            rgb = resize_and_pad_crop(rgb, target_h=INPUT_HEIGHT, target_w=INPUT_WIDTH)
        return crop_rgb_to_tensor(rgb), torch.tensor(encode_text(text), dtype=torch.long), text


def collate_ocr_batch(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor, str]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Collate `(image, target_ids, text)` items for `nn.CTCLoss`."""
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = torch.cat([item[1] for item in batch], dim=0)
    target_lengths = torch.tensor([int(item[1].numel()) for item in batch], dtype=torch.long)
    texts = [item[2] for item in batch]
    return images, targets, target_lengths, texts
