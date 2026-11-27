"""PyTorch Dataset and variable-length collator for End-Round sequences."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .config import (
        BOUNDARY_SIGMA_FRAMES,
        DEFAULT_MAX_TRAIN_SEQ_LEN,
        DEFAULT_SEED,
        DEFAULT_VAL_RATIO,
        IGNORE_INDEX,
    )
    from .prepare_dataset import PREPARED_DIR
    from .temporal_augmentations import augment_sequence_length
except ImportError:
    from config import (  # type: ignore[no-redef]
        BOUNDARY_SIGMA_FRAMES,
        DEFAULT_MAX_TRAIN_SEQ_LEN,
        DEFAULT_SEED,
        DEFAULT_VAL_RATIO,
        IGNORE_INDEX,
    )
    from prepare_dataset import PREPARED_DIR  # type: ignore[no-redef]
    from temporal_augmentations import augment_sequence_length  # type: ignore[no-redef]


def load_prepared_manifest(dataset_dir: Path | str = PREPARED_DIR) -> tuple[Path, list[dict[str, Any]]]:
    """Load ``manifest.json`` created by ``prepare_dataset.py``."""
    root = Path(dataset_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Prepared manifest not found at {manifest_path}. "
            "Run prepare_dataset.py (or train.py --prepare) first."
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = list(payload.get("clips", []))
    return root, records


def split_manifest_records(
    records: list[dict[str, Any]],
    *,
    val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int = DEFAULT_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically split prepared clips into train/val grouped & stratified.

    Why:
      If multiple clips share a ``group_id`` or ``run_id`` in metadata (e.g., from
      the same Balatro run), keeping groups together prevents run-level leakage
      (same joker lineup / background seed appearing in both train and val).
      Stratifying by ``source`` ensures real VOD clips and synthetic mod clips
      appear proportionally in validation.
    """
    if len(records) <= 1 or val_ratio <= 0.0:
        return list(records), []

    rng = np.random.default_rng(seed)
    by_source: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        source = str(rec.get("source", "balatro_mod"))
        meta = rec.get("metadata") or {}
        group_key = str(meta.get("group_id") or meta.get("run_id") or rec["clip_id"])
        by_source[source][group_key].append(rec)

    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []

    for source in sorted(by_source):
        groups = sorted(by_source[source].items(), key=lambda kv: kv[0])
        group_keys = [kv[0] for kv in groups]
        rng.shuffle(group_keys)

        n_groups = len(group_keys)
        if n_groups == 1:
            train_records.extend(by_source[source][group_keys[0]])
            continue

        n_val = int(round(n_groups * float(val_ratio)))
        n_val = max(1, min(n_groups - 1, n_val))
        val_set = set(group_keys[:n_val])

        for gk in group_keys:
            if gk in val_set:
                val_records.extend(by_source[source][gk])
            else:
                train_records.extend(by_source[source][gk])

    # Fallback in case each source had only 1 clip
    if not val_records and len(train_records) >= 2:
        idx = int(rng.integers(0, len(train_records)))
        val_records.append(train_records.pop(idx))

    return train_records, val_records


class EndRoundClipDataset(Dataset):
    """Dataset returning variable-length ``(T, 3, H, W)`` clip sequences."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        dataset_dir: Path | str = PREPARED_DIR,
        *,
        is_train: bool = True,
        max_seq_len: int | None = DEFAULT_MAX_TRAIN_SEQ_LEN,
        sigma: float = BOUNDARY_SIGMA_FRAMES,
        cache_in_ram: bool = True,
        use_temporal_aug: bool = True,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self.records = list(records)
        self.dataset_dir = Path(dataset_dir)
        self.is_train = bool(is_train)
        self.max_seq_len = max_seq_len
        self.sigma = float(sigma)
        self.cache_in_ram = bool(cache_in_ram)
        self.use_temporal_aug = bool(use_temporal_aug and is_train)
        self.seed = int(seed)

        self._cache: dict[int, dict[str, Any]] = {}
        if self.cache_in_ram:
            for idx in range(len(self.records)):
                self._cache[idx] = self._load_record(idx)

    def _load_record(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        pt_path = self.dataset_dir / rec["pt_file"]
        if not pt_path.exists():
            alt = self.dataset_dir / "clips" / rec["pt_file"]
            if alt.exists():
                pt_path = alt
        return torch.load(pt_path, map_location="cpu", weights_only=True)

    def _get_item_rng(self, idx: int) -> np.random.Generator:
        if not self.is_train:
            det_seed = (self.seed ^ ((idx + 1) * 2654435761)) % (2**32)
            return np.random.default_rng(det_seed)
        # Mix in a live torch RNG draw so epochs vary even when num_workers=0
        draw = int(torch.randint(0, 2**31 - 1, (1,)).item())
        worker_seed = (int(torch.initial_seed() % (2**31 - 1)) ^ draw ^ ((idx + 1) * 2654435761)) % (2**32)
        return np.random.default_rng(worker_seed)

    def _crop_window_around_event(
        self,
        frames: torch.Tensor,
        state_labels: torch.Tensor,
        boundary_targets: torch.Tensor,
        end_round_frame: int | None,
        rng: np.random.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int | None]:
        t_len = int(frames.shape[0])
        if self.max_seq_len is None or t_len <= self.max_seq_len:
            return frames, state_labels, boundary_targets, end_round_frame

        win = int(self.max_seq_len)
        if end_round_frame is None:
            start = int(rng.integers(0, t_len - win + 1)) if self.is_train else 0
            end = start + win
            return (
                frames[start:end],
                state_labels[start:end],
                boundary_targets[start:end],
                None,
            )

        # Keep end_round_frame strictly inside the cropped window with >=4 frames margin when possible
        margin = min(8, max(2, win // 12))
        lo_start = max(0, int(end_round_frame) - (win - 1 - margin))
        hi_start = min(t_len - win, max(0, int(end_round_frame) - margin))
        if hi_start < lo_start:
            lo_start = max(0, int(end_round_frame) - win + 1)
            hi_start = min(t_len - win, int(end_round_frame))
            if hi_start < lo_start:
                lo_start = hi_start = max(0, min(t_len - win, int(end_round_frame) - win // 2))

        if self.is_train and hi_start > lo_start:
            start = int(rng.integers(lo_start, hi_start + 1))
        else:
            start = (lo_start + hi_start) // 2

        end = start + win
        return (
            frames[start:end],
            state_labels[start:end],
            boundary_targets[start:end],
            int(end_round_frame) - start,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        raw = self._cache[idx] if idx in self._cache else self._load_record(idx)

        frames = raw["frames"].clone()
        state_labels = raw["state_labels"].clone()
        boundary_targets = raw["boundary_targets"].clone()
        events = raw.get("events") or {}
        end_round_frame = events.get("end_round_frame")
        if end_round_frame is not None:
            end_round_frame = int(end_round_frame)

        rng = self._get_item_rng(idx)
        if self.use_temporal_aug:
            frames, state_labels, boundary_targets, end_round_frame = augment_sequence_length(
                frames,
                state_labels,
                boundary_targets,
                end_round_frame,
                rng=rng,
                sigma=self.sigma,
            )

        frames, state_labels, boundary_targets, end_round_frame = self._crop_window_around_event(
            frames,
            state_labels,
            boundary_targets,
            end_round_frame,
            rng,
        )

        return {
            "clip_id": str(raw["clip_id"]),
            "source": str(raw.get("source", "balatro_mod")),
            "fps": float(raw["fps"]),
            "frames": frames,
            "state_labels": state_labels,
            "boundary_targets": boundary_targets,
            "end_round_frame": end_round_frame,
        }


def collate_end_round_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate variable-length ``EndRoundClipDataset`` items into padded batch tensors."""
    bsz = len(batch)
    if bsz == 0:
        raise ValueError("Cannot collate an empty batch")

    lengths = torch.tensor([int(item["frames"].shape[0]) for item in batch], dtype=torch.long)
    max_t = int(lengths.max().item())
    _, c, h, w = batch[0]["frames"].shape

    frames = torch.zeros((bsz, max_t, c, h, w), dtype=torch.uint8)
    state_labels = torch.full((bsz, max_t), IGNORE_INDEX, dtype=torch.long)
    boundary_targets = torch.zeros((bsz, max_t), dtype=torch.float32)
    valid_mask = torch.zeros((bsz, max_t), dtype=torch.bool)

    for b, item in enumerate(batch):
        t_len = int(item["frames"].shape[0])
        frames[b, :t_len] = item["frames"]
        state_labels[b, :t_len] = item["state_labels"]
        boundary_targets[b, :t_len] = item["boundary_targets"]
        valid_mask[b, :t_len] = True

    return {
        "clip_ids": [item["clip_id"] for item in batch],
        "sources": [item["source"] for item in batch],
        "fps": [float(item["fps"]) for item in batch],
        "frames": frames,
        "state_labels": state_labels,
        "boundary_targets": boundary_targets,
        "valid_mask": valid_mask,
        "lengths": lengths,
        "end_round_frames": [item["end_round_frame"] for item in batch],
    }
