"""
Shared Dataset Sync Utility for Balatro Classifiers
"""

import os
import shutil
import sys
from pathlib import Path


def sync_mod_dataset(
    source_subfolder: str,
    target_dataset_dir: Path,
    labels_subdir_name: str = "labels",
    clean: bool = False
) -> int:
    """
    Copies images and labels from the Balatro Mod output into a classifier's dataset dir.
    Checks %APPDATA%\\Balatro\\Mods\\gamestate_dataset_cv\\<source_subfolder>
    and balatro_mod/<source_subfolder>.
    """
    appdata_candidate = Path(os.path.expandvars(rf"%APPDATA%\Balatro\Mods\gamestate_dataset_cv\{source_subfolder}"))
    repo_root = Path(__file__).resolve().parent.parent.parent
    repo_candidate = repo_root / "balatro_mod" / source_subfolder

    candidates = [appdata_candidate, repo_candidate]
    src_dir = None
    for c in candidates:
        if c.exists() and (c / "images").exists():
            img_count = len(list((c / "images").glob("*.png"))) + len(list((c / "images").glob("*.jpg")))
            if img_count > 0:
                src_dir = c
                break

    if src_dir is None:
        for c in candidates:
            if c.exists():
                src_dir = c
                break

    if src_dir is None:
        src_dir = candidates[0]

    src_images = src_dir / "images"
    src_labels = src_dir / "labels"

    print(f"[Sync] Source Directory: {src_dir}")
    if not src_images.exists() or not src_labels.exists():
        print(f"[Sync Error] Source directories missing in: {src_dir}")
        return 0

    target_images = target_dataset_dir / "images"
    target_labels_detect = target_dataset_dir / labels_subdir_name
    target_labels_json = target_dataset_dir / "labels_json"

    for d in [target_images, target_labels_detect, target_labels_json]:
        if clean and d.exists():
            print(f"[Sync] Cleaning {d.name}...")
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    img_files = list(src_images.glob("*.png")) + list(src_images.glob("*.jpg"))
    print(f"[Sync] Found {len(img_files)} images in mod directory.")

    synced_images = 0
    synced_txt = 0
    synced_json = 0

    for img_path in img_files:
        stem = img_path.stem
        txt_path = src_labels / f"{stem}.txt"
        json_path = src_labels / f"{stem}.json"

        shutil.copy2(img_path, target_images / img_path.name)
        synced_images += 1

        if txt_path.exists():
            shutil.copy2(txt_path, target_labels_detect / txt_path.name)
            synced_txt += 1

        if json_path.exists():
            shutil.copy2(json_path, target_labels_json / json_path.name)
            synced_json += 1

    print(f"[Sync Complete] Copied {synced_images} images, {synced_txt} detection txt files, {synced_json} JSON files.")
    return synced_images
