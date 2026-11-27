"""
Prepare Stage 1 Dataset for YOLO View Deck Card Detection
=========================================================
Splits `dataset/images` and `dataset/labels_deck` into `train` and `val`
(default 80/20), writes `data.yaml` with 1 class (`card`), and logs deck size stats.
"""

import random
import shutil
import statistics
from pathlib import Path
import yaml

CURRENT_DIR = Path(__file__).resolve().parent
DECK_CLASSIFIER_DIR = CURRENT_DIR.parent
DATASET_DIR = DECK_CLASSIFIER_DIR / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels_deck"
PROCESSED_DATA_DIR = CURRENT_DIR / "data_split"
DATA_YAML_PATH = CURRENT_DIR / "data.yaml"


def _count_boxes(label_path: Path) -> int:
    with open(label_path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def prepare_detector_data(
    val_ratio: float = 0.20,
    seed: int = 42,
    dataset_dir: Path | None = None,
    processed_data_dir: Path | None = None,
    data_yaml_path: Path | None = None,
) -> Path:
    random.seed(seed)

    ds_dir = Path(dataset_dir) if dataset_dir is not None else DATASET_DIR
    images_dir = ds_dir / "images"
    labels_dir = ds_dir / "labels_deck"
    if not labels_dir.exists():
        labels_dir = ds_dir / "labels"

    out_split_dir = Path(processed_data_dir) if processed_data_dir is not None else PROCESSED_DATA_DIR
    out_yaml_path = Path(data_yaml_path) if data_yaml_path is not None else DATA_YAML_PATH

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Images or deck labels not found at {ds_dir}")

    img_files = sorted(list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")))
    valid_pairs = []
    cards_per_frame = []

    for img_path in img_files:
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            valid_pairs.append((img_path, lbl_path))
            cards_per_frame.append(_count_boxes(lbl_path))

    print(f"Found {len(valid_pairs)} valid image-label pairs for View Deck detection.")
    if len(valid_pairs) == 0:
        raise ValueError("No valid image-label pairs found in deck dataset!")

    total_boxes = sum(cards_per_frame)
    min_cards = min(cards_per_frame)
    med_cards = statistics.median(cards_per_frame)
    max_cards = max(cards_per_frame)
    print(
        f"Deck card counts: total_boxes={total_boxes}, "
        f"min={min_cards}, median={med_cards:.1f}, max={max_cards} cards/frame"
    )

    random.shuffle(valid_pairs)
    val_count = max(1, int(len(valid_pairs) * val_ratio)) if len(valid_pairs) > 1 else 0
    val_pairs = valid_pairs[:val_count]
    train_pairs = valid_pairs[val_count:] if len(valid_pairs) > 1 else valid_pairs

    print(f"Split: {len(train_pairs)} training, {len(val_pairs)} validation.")

    for split in ["train", "val"]:
        for subdir in ["images", "labels"]:
            p = out_split_dir / split / subdir
            if p.exists():
                shutil.rmtree(p)
            p.mkdir(parents=True, exist_ok=True)

    for split_name, pairs in [("train", train_pairs), ("val", val_pairs)]:
        target_img_dir = out_split_dir / split_name / "images"
        target_lbl_dir = out_split_dir / split_name / "labels"
        for img_path, lbl_path in pairs:
            shutil.copy2(img_path, target_img_dir / img_path.name)
            shutil.copy2(lbl_path, target_lbl_dir / lbl_path.name)

    train_images_path = (out_split_dir / "train" / "images").as_posix()
    val_images_path = (out_split_dir / "val" / "images").as_posix()

    yaml_data = {
        "path": out_split_dir.as_posix(),
        "train": train_images_path,
        "val": val_images_path,
        "names": {0: "card"},
    }

    with open(out_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=False)

    print(f"Deck detector data.yaml written to {out_yaml_path}")
    return out_yaml_path


if __name__ == "__main__":
    prepare_detector_data()
