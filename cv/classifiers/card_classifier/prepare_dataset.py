"""
Prepare Dataset for YOLO Card Classifier Training
=================================================
Reads dataset/card_classifier/images and dataset/card_classifier/labels,
verifies label/image pairs, reads dataset/card_classifier/classes.txt,
splits dataset into train and val (default 80/20),
and writes data.yaml pointing to the train and val splits.
"""

import os
import random
import shutil
from pathlib import Path

import yaml

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
DATASET_SRC = CURRENT_DIR / "dataset"
if not DATASET_SRC.exists() and (PROJECT_ROOT / "dataset" / "card_classifier").exists():
    DATASET_SRC = PROJECT_ROOT / "dataset" / "card_classifier"
PROCESSED_DATA_DIR = CURRENT_DIR / "data_split"
DATA_YAML_PATH = CURRENT_DIR / "data.yaml"

def prepare_dataset(val_ratio: float = 0.20, seed: int = 42):
    random.seed(seed)
    
    images_dir = DATASET_SRC / "images"
    labels_dir = DATASET_SRC / "labels"
    classes_file = DATASET_SRC / "classes.txt"

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"images or labels not found at {DATASET_SRC}")

    if not classes_file.exists():
        raise FileNotFoundError(f"classes.txt not found at {classes_file}")

    with open(classes_file, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]

    # Collect matched image and label files
    img_files = sorted(list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")))
    valid_pairs = []

    for img_path in img_files:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            valid_pairs.append((img_path, label_path))
        else:
            print(f"Warning: No label found for {img_path.name}")

    print(f"Found {len(valid_pairs)} valid image-label pairs in {DATASET_SRC}.")
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

    # Write data.yaml using forward slashes
    train_images_path = (PROCESSED_DATA_DIR / "train" / "images").as_posix()
    val_images_path = (PROCESSED_DATA_DIR / "val" / "images").as_posix()

    yaml_data = {
        "path": PROCESSED_DATA_DIR.as_posix(),
        "train": train_images_path,
        "val": val_images_path,
        "names": {i: name for i, name in enumerate(class_names)}
    }

    with open(DATA_YAML_PATH, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=False)

    print(f"data.yaml written to {DATA_YAML_PATH}")
    print(f"Classes ({len(class_names)}): {class_names[:5]} ... {class_names[-3:]}")
    return DATA_YAML_PATH

if __name__ == "__main__":
    prepare_dataset()
