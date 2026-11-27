"""
Stage 2 Shop Item Crop Dataset Loader
=====================================
Extracts individual card/item crops from shop screenshots (joker bar, consumable area, shop, pack)
and provides multi-head target dictionaries for classification.
Splits data by frame ID to prevent data leakage between train and val splits.
"""

import os
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

try:
    from .config import (
        KEY_TO_IDX, EDITION_TO_IDX,
        CROP_WIDTH, CROP_HEIGHT
    )
except ImportError:
    from config import (
        KEY_TO_IDX, EDITION_TO_IDX,
        CROP_WIDTH, CROP_HEIGHT
    )

import sys

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# pyrefly: ignore [missing-import]
from classifiers.augmentations import BalatroStreamAugmentation

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(is_train: bool = True):
    if is_train:
        return T.Compose([
            T.Resize((CROP_HEIGHT, CROP_WIDTH), interpolation=T.InterpolationMode.BILINEAR),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            T.ToTensor(),
            BalatroStreamAugmentation(compression_prob=0.5, cursor_prob=0.3),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return T.Compose([
            T.Resize((CROP_HEIGHT, CROP_WIDTH), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


class ItemCropDataset(Dataset):
    def __init__(self, data_root, split="train", val_ratio=0.20, seed=42, transform=None, cache_in_memory=True, max_cache_gb=12.0, pad_ratio: float = 0.0):
        super().__init__()
        self.data_root = Path(data_root)
        self.images_dir = self.data_root / "images"
        self.labels_dir = self.data_root / "labels_json"
        self.is_train = (split == "train")
        self.raw_tensor_mode = (transform is None)
        self.transform = transform
        self.pad_ratio = max(0.0, float(pad_ratio))
        self.crop_w = int(round(CROP_WIDTH * (1.0 + 2.0 * self.pad_ratio))) if self.pad_ratio > 0 else CROP_WIDTH
        self.crop_h = int(round(CROP_HEIGHT * (1.0 + 2.0 * self.pad_ratio))) if self.pad_ratio > 0 else CROP_HEIGHT

        # 1. Discover all matching frame image & json pairs
        json_files = sorted(list(self.labels_dir.glob("*.json")))
        valid_frames = []
        for jf in json_files:
            img_path = self.images_dir / f"{jf.stem}.png"
            if not img_path.exists():
                img_path = self.images_dir / f"{jf.stem}.jpg"
            if img_path.exists():
                valid_frames.append((img_path, jf))

        # 2. Split by frame to prevent data leakage between train and val
        rng = random.Random(seed)
        shuffled = list(valid_frames)
        rng.shuffle(shuffled)
        val_count = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0

        if split == "train":
            selected_frames = shuffled[val_count:] if len(shuffled) > 1 else shuffled
        else:
            selected_frames = shuffled[:val_count] if len(shuffled) > 1 else shuffled

        # 3. Flatten into individual item crop samples
        self.samples = []
        for img_path, jf in selected_frames:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    items = json.load(f)
                for item in items:
                    self.samples.append({
                        "img_path": img_path,
                        "box_pixel": item.get("box_pixel", [0, 0, 50, 50]),
                        "key": item.get("key", "unknown"),
                        "edition": item.get("edition", "None"),
                        "eternal": item.get("eternal", False),
                        "perishable": item.get("perishable", False),
                        "rental": item.get("rental", False),
                        "debuffed": item.get("debuffed", False),
                    })
            except Exception:
                continue

        # Precompute contiguous target tensors
        n = len(self.samples)
        unknown_idx = KEY_TO_IDX.get("unknown", 0)
        self._targets = {
            "key": torch.tensor([KEY_TO_IDX.get(s["key"], unknown_idx) for s in self.samples], dtype=torch.long),
            "edition": torch.tensor([EDITION_TO_IDX.get(s["edition"], 0) for s in self.samples], dtype=torch.long),
            "eternal": torch.tensor([1.0 if s["eternal"] else 0.0 for s in self.samples], dtype=torch.float32),
            "perishable": torch.tensor([1.0 if s["perishable"] else 0.0 for s in self.samples], dtype=torch.float32),
            "rental": torch.tensor([1.0 if s["rental"] else 0.0 for s in self.samples], dtype=torch.float32),
            "debuffed": torch.tensor([1.0 if s["debuffed"] else 0.0 for s in self.samples], dtype=torch.float32),
        }

        self._cached_images = None
        self.uses_ram_cache = False
        est_bytes = n * 3 * self.crop_h * self.crop_w
        if cache_in_memory and self.raw_tensor_mode and n > 0 and est_bytes <= int(max_cache_gb * (1024 ** 3)):
            self._preload_crops_to_ram()

    def _extract_resized_crop(self, img_rgb: Image.Image, box: list | tuple) -> Image.Image:
        iw, ih = img_rgb.size
        x, y, w, h = box
        if w <= 0 or h <= 0:
            return img_rgb.resize((self.crop_w, self.crop_h), Image.Resampling.BILINEAR)
        if self.pad_ratio <= 0.0:
            x1 = max(0, int(round(x)))
            y1 = max(0, int(round(y)))
            x2 = min(iw, int(round(x + w)))
            y2 = min(ih, int(round(y + h)))
            crop = img_rgb if (x2 <= x1 or y2 <= y1) else img_rgb.crop((x1, y1, x2, y2))
            return crop.resize((self.crop_w, self.crop_h), Image.Resampling.BILINEAR)

        pad_x = max(1, int(round(w * self.pad_ratio)))
        pad_y = max(1, int(round(h * self.pad_ratio)))
        tx1 = int(round(x - pad_x))
        ty1 = int(round(y - pad_y))
        tx2 = int(round(x + w + pad_x))
        ty2 = int(round(y + h + pad_y))
        cx1 = max(0, min(iw - 1, tx1))
        cy1 = max(0, min(ih - 1, ty1))
        cx2 = max(cx1 + 1, min(iw, tx2))
        cy2 = max(cy1 + 1, min(ih, ty2))
        sub = img_rgb.crop((cx1, cy1, cx2, cy2))
        pl = max(0, cx1 - tx1)
        pt = max(0, cy1 - ty1)
        pr = max(0, tx2 - cx2)
        pb = max(0, ty2 - cy2)
        if pl or pt or pr or pb:
            arr = np.asarray(sub, dtype=np.uint8)
            arr = np.pad(arr, ((pt, pb), (pl, pr), (0, 0)), mode="edge")
            crop = Image.fromarray(arr)
        else:
            crop = sub
        return crop.resize((self.crop_w, self.crop_h), Image.Resampling.BILINEAR)

    def _preload_crops_to_ram(self):
        n = len(self.samples)
        buf = torch.empty((n, 3, self.crop_h, self.crop_w), dtype=torch.uint8)
        groups = defaultdict(list)
        for idx, s in enumerate(self.samples):
            groups[s["img_path"]].append((idx, s["box_pixel"]))

        def _process_frame(item):
            img_path, entries = item
            try:
                with Image.open(img_path) as full_img:
                    img_rgb = full_img.convert("RGB")
                    for idx, box in entries:
                        resized = self._extract_resized_crop(img_rgb, box)
                        arr = np.asarray(resized, dtype=np.uint8)
                        buf[idx].copy_(torch.from_numpy(arr).permute(2, 0, 1))
            except Exception:
                buf[entries[0][0]].fill_(128)
                for idx, _ in entries[1:]:
                    buf[idx].fill_(128)

        workers = min(16, max(4, os.cpu_count() or 8))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_process_frame, groups.items()))

        self._cached_images = buf
        self.uses_ram_cache = True

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.uses_ram_cache:
            crop_tensor = self._cached_images[idx]
        else:
            sample = self.samples[idx]
            img_path = sample["img_path"]
            box = sample["box_pixel"]

            with Image.open(img_path) as full_img:
                img_rgb = full_img.convert("RGB")
                if self.raw_tensor_mode:
                    resized = self._extract_resized_crop(img_rgb, box)
                    crop_tensor = torch.from_numpy(np.asarray(resized, dtype=np.uint8)).permute(2, 0, 1).contiguous()
                else:
                    iw, ih = img_rgb.size
                    x, y, w, h = box
                    x1 = max(0, int(round(x)))
                    y1 = max(0, int(round(y)))
                    x2 = min(iw, int(round(x + w)))
                    y2 = min(ih, int(round(y + h)))
                    crop = img_rgb if (x2 <= x1 or y2 <= y1) else img_rgb.crop((x1, y1, x2, y2))
                    crop_tensor = self.transform(crop)

        targets = {k: v[idx] for k, v in self._targets.items()}
        return crop_tensor, targets
