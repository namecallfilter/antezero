"""
Prepare & Inspect Shop Classifier Dataset
=========================================
Validates synced images and JSON labels in `classifiers/shop_classifier/dataset/`,
verifies class distribution, and pre-generates Stage 1 YOLO data splits.
"""

import json
from pathlib import Path
from collections import Counter

CURRENT_DIR = Path(__file__).resolve().parent
DATASET_DIR = CURRENT_DIR / "dataset"

from stage1_detector.prepare_detector_data import prepare_detector_data
from sync_dataset import sync_dataset


def inspect_dataset():
    images_dir = DATASET_DIR / "images"
    labels_json = DATASET_DIR / "labels_json"
    labels_detect = DATASET_DIR / "labels_detect"

    if not images_dir.exists() or len(list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))) == 0:
        print("[Dataset] Empty local dataset, syncing from mod...")
        sync_dataset(clean=False)

    images = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
    json_files = list(labels_json.glob("*.json"))
    txt_files = list(labels_detect.glob("*.txt"))

    print("\n" + "=" * 60)
    print("SHOP DATASET INSPECTION & STATISTICS")
    print("=" * 60)
    print(f"Total Images      : {len(images):,}")
    print(f"Total YOLO Labels : {len(txt_files):,}")
    print(f"Total JSON Labels : {len(json_files):,}")

    type_counts = Counter()
    edition_counts = Counter()
    sticker_counts = Counter()
    total_items = 0

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                items = json.load(f)
            for item in items:
                total_items += 1
                type_counts[item.get("card_type", "Unknown")] += 1
                ed = item.get("edition", "None")
                if ed != "None":
                    edition_counts[ed] += 1
                if item.get("eternal"): sticker_counts["Eternal"] += 1
                if item.get("perishable"): sticker_counts["Perishable"] += 1
                if item.get("rental"): sticker_counts["Rental"] += 1
                if item.get("debuffed"): sticker_counts["Debuffed"] += 1
        except Exception:
            continue

    print(f"Total Item Crops  : {total_items:,}")
    print("\n[Card Types]")
    for k, v in type_counts.most_common():
        print(f"  {k:15s}: {v:,}")

    print("\n[Editions]")
    for k, v in edition_counts.most_common():
        print(f"  {k:15s}: {v:,}")

    print("\n[Stickers / Status]")
    for k, v in sticker_counts.most_common():
        print(f"  {k:15s}: {v:,}")
    print("=" * 60 + "\n")

    # Generate Stage 1 detector split
    print("[Stage 1] Pre-generating Stage 1 YOLO data split...")
    prepare_detector_data()


if __name__ == "__main__":
    inspect_dataset()
