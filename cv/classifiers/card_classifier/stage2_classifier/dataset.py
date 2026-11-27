"""
Stage 2 Card Crop Dataset Loader
================================
Extracts individual card crops from full hand frames and provides
attribute target dictionaries for multi-task classification.
Splits data by frame ID so cards from the same screenshot remain in the same split.
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
        RANK_TO_IDX, SUIT_TO_IDX, EDITION_TO_IDX, ENHANCEMENT_TO_IDX, SEAL_TO_IDX,
        CROP_WIDTH, CROP_HEIGHT
    )
except ImportError:
    from config import (
        RANK_TO_IDX, SUIT_TO_IDX, EDITION_TO_IDX, ENHANCEMENT_TO_IDX, SEAL_TO_IDX,
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


def _select_deck_cards_for_frame(cards: list[dict], max_crops: int | None, rng: random.Random) -> list[dict]:
    """
    Subsamples dense View Deck screenshots while preserving rare attributes
    (editions, seals, enhancements, debuffed, greyed, Stone cards) and unique card combinations.
    """
    if max_crops is None or max_crops <= 0 or len(cards) <= max_crops:
        return list(cards)

    indexed = list(enumerate(cards))
    rng.shuffle(indexed)

    unique_by_sig: dict[tuple, tuple[int, dict]] = {}
    for orig_idx, card in indexed:
        sig = (
            card.get("rank", "None"),
            card.get("suit", "None"),
            card.get("edition", "None"),
            card.get("enhancement", "None"),
            card.get("seal", "None"),
            bool(card.get("debuffed", False)),
            bool(card.get("greyed", False)),
        )
        if sig not in unique_by_sig:
            unique_by_sig[sig] = (orig_idx, card)

    def _rarity_priority(entry: tuple[int, dict]) -> int:
        c = entry[1]
        score = 0
        if c.get("edition", "None") != "None":
            score += 8
        if c.get("seal", "None") != "None":
            score += 6
        if c.get("enhancement", "None") != "None":
            score += 3
        if bool(c.get("debuffed", False)):
            score += 4
        if bool(c.get("greyed", False)):
            score += 2
        if c.get("rank", "None") == "Stone":
            score += 3
        return -score

    unique_entries = sorted(unique_by_sig.values(), key=_rarity_priority)
    selected_entries = unique_entries[:max_crops]
    chosen_indices = {orig_idx for orig_idx, _ in selected_entries}

    if len(selected_entries) < max_crops:
        for orig_idx, card in indexed:
            if orig_idx not in chosen_indices:
                selected_entries.append((orig_idx, card))
                chosen_indices.add(orig_idx)
                if len(selected_entries) >= max_crops:
                    break

    selected_entries.sort(key=lambda item: item[0])
    return [card for _, card in selected_entries]


class CardCropDataset(Dataset):
    def __init__(
        self,
        data_root,
        split="train",
        val_ratio=0.20,
        seed=42,
        transform=None,
        cache_in_memory=True,
        max_cache_gb=12.0,
        max_deck_crops_per_frame: int | None = 16,
        pad_ratio: float = 0.0,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.images_dir = self.data_root / "images"
        self.labels_dir = self.data_root / "labels_json"
        self.is_train = (split == "train")
        self.raw_tensor_mode = (transform is None)
        self.transform = transform
        self.max_deck_crops_per_frame = max_deck_crops_per_frame
        self.pad_ratio = max(0.0, float(pad_ratio))
        self.crop_w = int(round(CROP_WIDTH * (1.0 + 2.0 * self.pad_ratio))) if self.pad_ratio > 0 else CROP_WIDTH
        self.crop_h = int(round(CROP_HEIGHT * (1.0 + 2.0 * self.pad_ratio))) if self.pad_ratio > 0 else CROP_HEIGHT

        # 1. Discover all matching hand frame image & json pairs
        valid_frames = []
        if self.labels_dir.exists() and self.images_dir.exists():
            json_files = sorted(list(self.labels_dir.glob("*.json")))
            for jf in json_files:
                img_path = self.images_dir / f"{jf.stem}.png"
                if not img_path.exists():
                    img_path = self.images_dir / f"{jf.stem}.jpg"
                if img_path.exists():
                    valid_frames.append((img_path, jf, "hand"))

        # Also discover shop & pack playing card crops (from self.data_root / "shop_crops")
        shop_crops_dir = self.data_root / "shop_crops"
        if shop_crops_dir.exists():
            shop_images = shop_crops_dir / "images"
            shop_labels = shop_crops_dir / "labels_json"
            if not shop_labels.exists():
                shop_labels = shop_crops_dir / "labels"
            if shop_labels.exists() and shop_images.exists():
                for jf in sorted(list(shop_labels.glob("*.json"))):
                    img_path = shop_images / f"{jf.stem}.png"
                    if not img_path.exists():
                        img_path = shop_images / f"{jf.stem}.jpg"
                    if img_path.exists():
                        valid_frames.append((img_path, jf, "shop"))

        # Discover View Deck frames (from self.data_root / "deck_frames" or fallback "deck_crops")
        deck_frames = []
        for deck_sub in ("deck_frames", "deck_crops"):
            deck_dir = self.data_root / deck_sub
            if deck_dir.exists():
                deck_images = deck_dir / "images"
                deck_labels = deck_dir / "labels_json"
                if not deck_labels.exists():
                    deck_labels = deck_dir / "labels"
                if deck_labels.exists() and deck_images.exists():
                    for jf in sorted(list(deck_labels.glob("*.json"))):
                        img_path = deck_images / f"{jf.stem}.png"
                        if not img_path.exists():
                            img_path = deck_images / f"{jf.stem}.jpg"
                        if img_path.exists():
                            deck_frames.append((img_path, jf, "deck"))

        # 2. Split by frame to prevent data leakage between train and val.
        # Hand+shop frames use the identical original RNG sequence so splits remain unchanged when deck_frames is absent.
        rng = random.Random(seed)
        shuffled = list(valid_frames)
        rng.shuffle(shuffled)
        val_count = max(1, int(len(shuffled) * val_ratio)) if shuffled else 0

        if split == "train":
            selected_frames = shuffled[val_count:]
        else:
            selected_frames = shuffled[:val_count]

        if deck_frames:
            deck_rng = random.Random(seed)
            deck_shuffled = list(deck_frames)
            deck_rng.shuffle(deck_shuffled)
            deck_val_count = max(1, int(len(deck_shuffled) * val_ratio))
            if split == "train":
                selected_frames.extend(deck_shuffled[deck_val_count:])
            else:
                selected_frames.extend(deck_shuffled[:deck_val_count])

        # 3. Flatten into individual card crop samples
        self.samples = []
        crop_rng = random.Random(seed + (0 if split == "train" else 10_000))
        source_counts = {"hand": 0, "shop": 0, "deck": 0}
        for img_path, jf, source in selected_frames:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    cards = json.load(f)
                if source == "deck":
                    cards = _select_deck_cards_for_frame(cards, self.max_deck_crops_per_frame, crop_rng)
                for card in cards:
                    self.samples.append({
                        "img_path": img_path,
                        "source": source,
                        "box_pixel": card["box_pixel"],
                        "rank": card.get("rank", "None"),
                        "suit": card.get("suit", "None"),
                        "edition": card.get("edition", "None"),
                        "enhancement": card.get("enhancement", "None"),
                        "seal": card.get("seal", "None"),
                        "selected": card.get("selected", False),
                        "debuffed": card.get("debuffed", False),
                        "face_down": card.get("face_down", False)
                    })
                    source_counts[source] = source_counts.get(source, 0) + 1
            except Exception:
                continue

        self.sample_sources = [s["source"] for s in self.samples]
        print(
            f"[{split.upper()}] Loaded {len(self.samples)} card crops across {len(selected_frames)} frames "
            f"(hand={source_counts['hand']}, shop={source_counts['shop']}, deck={source_counts['deck']})."
        )

        # Precompute contiguous target tensors
        n = len(self.samples)
        none_rank_idx = RANK_TO_IDX.get("None", 0)
        self._targets = {
            "rank": torch.tensor([RANK_TO_IDX.get(s["rank"], none_rank_idx) for s in self.samples], dtype=torch.long),
            "suit": torch.tensor([SUIT_TO_IDX.get(s["suit"], 4) for s in self.samples], dtype=torch.long),
            "edition": torch.tensor([EDITION_TO_IDX.get(s["edition"], 0) for s in self.samples], dtype=torch.long),
            "enhancement": torch.tensor([ENHANCEMENT_TO_IDX.get(s["enhancement"], 0) for s in self.samples], dtype=torch.long),
            "seal": torch.tensor([SEAL_TO_IDX.get(s["seal"], 0) for s in self.samples], dtype=torch.long),
            "debuffed": torch.tensor([1.0 if s["debuffed"] else 0.0 for s in self.samples], dtype=torch.float32),
            "face_down": torch.tensor([1.0 if s["face_down"] else 0.0 for s in self.samples], dtype=torch.float32),
        }

        self._cached_images = None
        self.uses_ram_cache = False
        est_bytes = n * 3 * self.crop_h * self.crop_w
        if cache_in_memory and self.raw_tensor_mode and n > 0 and est_bytes <= int(max_cache_gb * (1024 ** 3)):
            self._preload_crops_to_ram()

    def _extract_resized_crop(self, img_rgb: Image.Image, box: list | tuple) -> Image.Image:
        iw, ih = img_rgb.size
        px, py, pw, ph = box
        if self.pad_ratio <= 0.0:
            if px == 0 and py == 0 and abs(pw - iw) <= 2 and abs(ph - ih) <= 2:
                crop = img_rgb
            else:
                x1 = max(0, min(iw - 1, int(round(px))))
                y1 = max(0, min(ih - 1, int(round(py))))
                x2 = max(x1 + 1, min(iw, int(round(px + pw))))
                y2 = max(y1 + 1, min(ih, int(round(py + ph))))
                crop = img_rgb.crop((x1, y1, x2, y2))
            return crop.resize((self.crop_w, self.crop_h), Image.Resampling.BILINEAR)

        pad_x = max(1, int(round(pw * self.pad_ratio)))
        pad_y = max(1, int(round(ph * self.pad_ratio)))
        if px == 0 and py == 0 and abs(pw - iw) <= 2 and abs(ph - ih) <= 2:
            arr = np.asarray(img_rgb, dtype=np.uint8)
            arr = np.pad(arr, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode="edge")
            crop = Image.fromarray(arr)
        else:
            tx1 = int(round(px - pad_x))
            ty1 = int(round(py - pad_y))
            tx2 = int(round(px + pw + pad_x))
            ty2 = int(round(py + ph + pad_y))
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
                for idx, _ in entries:
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
            tensor_img = self._cached_images[idx]
        else:
            item = self.samples[idx]
            try:
                with Image.open(item["img_path"]) as full_img:
                    img_rgb = full_img.convert("RGB")
                    if self.raw_tensor_mode:
                        resized = self._extract_resized_crop(img_rgb, item["box_pixel"])
                        tensor_img = torch.from_numpy(np.asarray(resized, dtype=np.uint8)).permute(2, 0, 1).contiguous()
                    else:
                        iw, ih = img_rgb.size
                        px, py, pw, ph = item["box_pixel"]
                        if px == 0 and py == 0 and abs(pw - iw) <= 2 and abs(ph - ih) <= 2:
                            crop = img_rgb.copy()
                        else:
                            x1 = max(0, min(iw - 1, px))
                            y1 = max(0, min(ih - 1, py))
                            x2 = max(x1 + 1, min(iw, x1 + pw))
                            y2 = max(y1 + 1, min(ih, y1 + ph))
                            crop = img_rgb.crop((x1, y1, x2, y2))
                        tensor_img = self.transform(crop)
            except Exception:
                crop = Image.new("RGB", (self.crop_w, self.crop_h), (128, 128, 128))
                if self.raw_tensor_mode:
                    tensor_img = torch.from_numpy(np.asarray(crop, dtype=np.uint8)).permute(2, 0, 1).contiguous()
                else:
                    tensor_img = self.transform(crop)

        targets = {k: v[idx] for k, v in self._targets.items()}
        return tensor_img, targets
