"""
Prepare Train/Val Split for Stage 1 YOLO Shop Item Detector
============================================================
Splits `dataset/images` and `dataset/labels_detect` into
`stage1_detector/data_split/images/{train,val}` and
`stage1_detector/data_split/labels/{train,val}`.
Also writes an updated `data.yaml` pointing to absolute directory paths.
"""

import random
import shutil
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
CLASSIFIER_ROOT = CURRENT_DIR.parent
DATASET_DIR = CLASSIFIER_ROOT / "dataset"
DATA_SPLIT_DIR = CURRENT_DIR / "data_split"
DATA_YAML = CURRENT_DIR / "data.yaml"


def prepare_detector_data(val_ratio=0.15, seed=42):
    images_dir = DATASET_DIR / "images"
    labels_dir = DATASET_DIR / "labels_detect"

    if not images_dir.exists() or not labels_dir.exists():
        print(f"[Error] Dataset directories missing: {images_dir}")
        return False

    img_files = sorted(list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")))
    print(f"[Stage 1] Found {len(img_files):,} images in {images_dir}")

    if len(img_files) == 0:
        print("[Warning] No images found to prepare.")
        return False

    # Filter to pairs that have matching labels
    valid_pairs = []
    for img in img_files:
        lbl = labels_dir / f"{img.stem}.txt"
        if lbl.exists():
            valid_pairs.append((img, lbl))

    print(f"[Stage 1] Found {len(valid_pairs):,} valid image/label pairs.")

    rng = random.Random(seed)
    shuffled = list(valid_pairs)
    rng.shuffle(shuffled)

    val_count = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
    val_pairs = shuffled[:val_count]
    train_pairs = shuffled[val_count:] if len(shuffled) > 1 else shuffled

    # Recreate clean target split folders
    if DATA_SPLIT_DIR.exists():
        shutil.rmtree(DATA_SPLIT_DIR)

    for split in ["train", "val"]:
        (DATA_SPLIT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATA_SPLIT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    def copy_pairs(pairs, split):
        for img, lbl in pairs:
            shutil.copy2(img, DATA_SPLIT_DIR / "images" / split / img.name)
            shutil.copy2(lbl, DATA_SPLIT_DIR / "labels" / split / lbl.name)

    copy_pairs(train_pairs, "train")
    copy_pairs(val_pairs, "val")
    print(f"[Stage 1] Prepared data split: {len(train_pairs):,} train, {len(val_pairs):,} val")

    # Generate data.yaml with absolute posix paths
    yaml_content = f"""path: {DATA_SPLIT_DIR.as_posix()}
train: images/train
val: images/val

names:
  0: Joker
  1: Tarot
  2: Spectral
  3: Planet
  4: Voucher
  5: Booster
  6: PlayingCard
"""
    with open(DATA_YAML, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"[Stage 1] Generated YOLO configuration: {DATA_YAML}")
    return True


if __name__ == "__main__":
    prepare_detector_data()
