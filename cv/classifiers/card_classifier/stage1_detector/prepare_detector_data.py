"""
Prepare Stage 1 Dataset for YOLO Card Detection
================================================
Splits dataset/images and dataset/labels_card into train and val (default 80/20),
and writes data.yaml pointing to the train and val splits with 1 class ("card").
"""

import random
import shutil
from pathlib import Path
import yaml

CURRENT_DIR = Path(__file__).resolve().parent
CARD_CLASSIFIER_DIR = CURRENT_DIR.parent
DATASET_DIR = CARD_CLASSIFIER_DIR / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels_card"
PROCESSED_DATA_DIR = CURRENT_DIR / "data_split"
DATA_YAML_PATH = CURRENT_DIR / "data.yaml"

def prepare_detector_data(val_ratio: float = 0.20, seed: int = 42):
    random.seed(seed)

    if not IMAGES_DIR.exists() or not LABELS_DIR.exists():
        raise FileNotFoundError(f"Images or card labels not found at {DATASET_DIR}")

    img_files = sorted(list(IMAGES_DIR.glob("*.png")) + list(IMAGES_DIR.glob("*.jpg")))
    valid_pairs = []

    for img_path in img_files:
        lbl_path = LABELS_DIR / f"{img_path.stem}.txt"
        if lbl_path.exists():
            valid_pairs.append((img_path, lbl_path))

    print(f"Found {len(valid_pairs)} valid image-label pairs for card detection.")
    if len(valid_pairs) == 0:
        raise ValueError("No valid image-label pairs found in dataset!")

    random.shuffle(valid_pairs)
    val_count = max(1, int(len(valid_pairs) * val_ratio))
    val_pairs = valid_pairs[:val_count]
    train_pairs = valid_pairs[val_count:]

    print(f"Split: {len(train_pairs)} training, {len(val_pairs)} validation.")

    # Recreate target directories
    for split in ["train", "val"]:
        for subdir in ["images", "labels"]:
            p = PROCESSED_DATA_DIR / split / subdir
            if p.exists():
                shutil.rmtree(p)
            p.mkdir(parents=True, exist_ok=True)

    # Copy files
    for split_name, pairs in [("train", train_pairs), ("val", val_pairs)]:
        target_img_dir = PROCESSED_DATA_DIR / split_name / "images"
        target_lbl_dir = PROCESSED_DATA_DIR / split_name / "labels"
        for img_path, lbl_path in pairs:
            shutil.copy2(img_path, target_img_dir / img_path.name)
            shutil.copy2(lbl_path, target_lbl_dir / lbl_path.name)

    # Write data.yaml using portable forward slashes
    train_images_path = (PROCESSED_DATA_DIR / "train" / "images").as_posix()
    val_images_path = (PROCESSED_DATA_DIR / "val" / "images").as_posix()

    yaml_data = {
        "path": PROCESSED_DATA_DIR.as_posix(),
        "train": train_images_path,
        "val": val_images_path,
        "names": {0: "card"}
    }

    with open(DATA_YAML_PATH, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=False)

    print(f"Detector data.yaml written to {DATA_YAML_PATH}")
    return DATA_YAML_PATH

if __name__ == "__main__":
    prepare_detector_data()
