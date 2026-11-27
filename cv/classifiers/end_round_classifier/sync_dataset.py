"""
Sync Clip Dataset from Balatro Mod to End-Round Temporal Classifier
===================================================================
Copies clip directories produced by the future Balatro mod from:
  - %APPDATA%/Balatro/Mods/gamestate_dataset_cv/dataset_end_round/<clip_id>/
  - or balatro_mod/dataset_end_round/<clip_id>/
into:
  - classifiers/end_round_classifier/dataset/raw/<clip_id>/

Any hand-labeled or semi-automated real VOD clips placed in `dataset/real_vod/`
are preserved independently of `--clean`.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

try:
    from .config import MOD_SOURCE_SUBFOLDER
except ImportError:
    from config import MOD_SOURCE_SUBFOLDER

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
TARGET_RAW_DIR = CURRENT_DIR / "dataset" / "raw"


def find_mod_source_dir(source_subfolder: str = MOD_SOURCE_SUBFOLDER) -> Path | None:
    appdata_candidate = Path(
        os.path.expandvars(rf"%APPDATA%\Balatro\Mods\gamestate_dataset_cv\{source_subfolder}")
    )
    repo_candidate = REPO_ROOT / "balatro_mod" / source_subfolder

    for cand in (appdata_candidate, repo_candidate):
        if cand.exists() and any(p.is_dir() and (p / "meta.json").exists() for p in cand.iterdir()):
            return cand
    for cand in (appdata_candidate, repo_candidate):
        if cand.exists():
            return cand
    return None


def sync_dataset(
    clean: bool = False,
    overwrite: bool = False,
    *,
    src_dir: Path | None = None,
    target_raw_dir: Path = TARGET_RAW_DIR,
) -> int:
    resolved_src = src_dir if src_dir is not None else find_mod_source_dir()
    if resolved_src is None:
        print(
            "[Sync] No mod dataset_end_round directory found yet. "
            f"Expected %APPDATA%\\Balatro\\Mods\\gamestate_dataset_cv\\{MOD_SOURCE_SUBFOLDER} "
            f"or {REPO_ROOT / 'balatro_mod' / MOD_SOURCE_SUBFOLDER}."
        )
        return 0

    if clean and target_raw_dir.exists():
        print(f"[Sync] Cleaning {target_raw_dir}...")
        shutil.rmtree(target_raw_dir)
    target_raw_dir.mkdir(parents=True, exist_ok=True)

    clip_dirs = sorted(
        p for p in resolved_src.iterdir() if p.is_dir() and (p / "meta.json").exists()
    )
    print(f"[Sync] Source: {resolved_src} ({len(clip_dirs)} clip directories found)")

    copied = 0
    skipped = 0
    for clip_dir in clip_dirs:
        dst_clip = target_raw_dir / clip_dir.name
        if dst_clip.exists():
            if not overwrite:
                skipped += 1
                continue
            shutil.rmtree(dst_clip)
        shutil.copytree(clip_dir, dst_clip)
        copied += 1

    print(f"[Sync Complete] Copied {copied} clips, skipped {skipped} existing clips -> {target_raw_dir}")
    return copied


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync End-Round Clip Dataset from Balatro Mod")
    parser.add_argument("--clean", action="store_true", help="Clear dataset/raw before syncing")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing clip directories")
    args = parser.parse_args()
    sync_dataset(clean=args.clean, overwrite=args.overwrite)
