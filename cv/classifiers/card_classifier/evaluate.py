"""
Evaluate Two-Stage Card Recognition Pipeline
============================================
Runs Stage 1 detector + Stage 2 classifier on a sample image or validation set.
Prints reconstructed cards and compares directly against ground truth JSON.
"""

import sys
import json
import random
import argparse
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
DATASET_DIR = CURRENT_DIR / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
JSON_LABELS_DIR = DATASET_DIR / "labels_json"

from pipeline import CardRecognitionPipeline


def evaluate_single_image(pipeline, image_path):
    image_path = Path(image_path)
    print(f"\nEvaluating on image: {image_path.name}")

    results = pipeline.predict(image_path)

    gt_file = JSON_LABELS_DIR / f"{image_path.stem}.json"
    gt_cards = []
    if gt_file.exists():
        with open(gt_file, "r", encoding="utf-8") as f:
            gt_cards = json.load(f)

    print("\n" + "=" * 70)
    print(f"RECONSTRUCTED HAND ({len(results)} cards detected left-to-right)")
    print("=" * 70)

    for i, c in enumerate(results):
        gt = gt_cards[i] if i < len(gt_cards) else None
        match_str = ""
        if gt:
            rank_match = (c.rank == gt.get("rank"))
            suit_match = (c.suit == gt.get("suit"))
            ed_match = (c.edition == gt.get("edition"))
            enh_match = (c.enhancement == gt.get("enhancement"))
            seal_match = (c.seal == gt.get("seal"))
            deb_match = (c.debuffed == gt.get("debuffed", False))
            fd_match = (c.face_down == gt.get("face_down", False))
            all_match = rank_match and suit_match and ed_match and enh_match and seal_match and deb_match and fd_match
            match_str = " [ALL CORRECT]" if all_match else " [MISMATCH]"

        print(f"Card {c.index:2d}: {c.display_name}{match_str}")
        print(f"   Rank       : {c.rank:<8} (conf: {c.confidences['rank']:.2f})")
        print(f"   Suit       : {c.suit:<8} (conf: {c.confidences['suit']:.2f})")
        print(f"   Edition    : {c.edition:<8} (conf: {c.confidences['edition']:.2f})")
        print(f"   Enhancement: {c.enhancement:<8} (conf: {c.confidences['enhancement']:.2f})")
        print(f"   Seal       : {c.seal:<8} (conf: {c.confidences['seal']:.2f})")
        print(f"   Debuffed   : {str(c.debuffed):<8} (conf: {c.confidences.get('debuffed', 1.0):.2f})")
        print(f"   Face Down  : {str(c.face_down):<8} (conf: {c.confidences.get('face_down', 1.0):.2f})")
        print(f"   Selected   : {c.selected}")
        if gt and not all_match:
            print(f"   --> Ground Truth: {gt.get('rank')} of {gt.get('suit')} | Ed: {gt.get('edition')} | Enh: {gt.get('enhancement')} | Seal: {gt.get('seal')} | Deb: {gt.get('debuffed', False)} | FD: {gt.get('face_down', False)}")
        print("-" * 50)


def evaluate_benchmark(pipeline, num_frames=None, split="val", conf=0.60, seed=42, val_ratio=0.20):
    json_files = sorted(list(JSON_LABELS_DIR.glob("*.json")))
    if not json_files:
        print("No JSON labels found in dataset/labels_json/!")
        return

    # Filter by existing image pairs
    valid_frames = []
    for jf in json_files:
        img_path = IMAGES_DIR / f"{jf.stem}.png"
        if not img_path.exists():
            img_path = IMAGES_DIR / f"{jf.stem}.jpg"
        if img_path.exists():
            valid_frames.append(jf)

    # Use the EXACT same split logic as training (seed 42, 80% train / 20% val)
    rng = random.Random(seed)
    shuffled = list(valid_frames)
    rng.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_ratio))

    if split == "val":
        target_jsons = shuffled[:val_count]
        split_desc = f"STRICT UNSEEN VALIDATION SET ({len(target_jsons)} frames, 20% split)"
    elif split == "train":
        target_jsons = shuffled[val_count:]
        split_desc = f"TRAINING SET ({len(target_jsons)} frames, 80% split)"
    else:
        target_jsons = valid_frames
        split_desc = f"ALL DATASET FRAMES ({len(target_jsons)} frames, train + val combined)"

    if num_frames and num_frames > 0:
        target_jsons = target_jsons[:num_frames]

    print("\n" + "=" * 70)
    print(f"EVALUATION BENCHMARK: {split_desc}")
    print(f"Testing {len(target_jsons)} frames at detection confidence={conf:.2f}")
    print("=" * 70)

    total_frames = len(target_jsons)
    frames_all_correct = 0
    frames_count_match = 0

    total_gt_cards = 0
    total_detected_cards = 0

    attr_correct = {
        "rank": 0,
        "suit": 0,
        "edition": 0,
        "enhancement": 0,
        "seal": 0,
        "debuffed": 0,
        "face_down": 0,
        "selected": 0,
        "exact_card": 0
    }
    evaluated_pairs = 0
    mismatches = []

    for idx, jf in enumerate(target_jsons, 1):
        img_path = IMAGES_DIR / f"{jf.stem}.png"
        if not img_path.exists():
            img_path = IMAGES_DIR / f"{jf.stem}.jpg"
        if not img_path.exists():
            continue

        with open(jf, "r", encoding="utf-8") as f:
            gt_cards = json.load(f)

        preds = pipeline.predict(img_path, conf=conf)

        n_gt = len(gt_cards)
        n_pred = len(preds)
        total_gt_cards += n_gt
        total_detected_cards += n_pred

        if n_gt == n_pred:
            frames_count_match += 1

        frame_perfect = (n_gt == n_pred)
        n_pairs = min(n_gt, n_pred)

        for i in range(n_pairs):
            c = preds[i]
            g = gt_cards[i]
            evaluated_pairs += 1

            r_m = (c.rank == g.get("rank"))
            s_m = (c.suit == g.get("suit"))
            e_m = (c.edition == g.get("edition"))
            en_m = (c.enhancement == g.get("enhancement"))
            se_m = (c.seal == g.get("seal"))
            deb_m = (c.debuffed == g.get("debuffed", False))
            fd_m = (c.face_down == g.get("face_down", False))
            sel_m = (c.selected == g.get("selected"))
            all_m = r_m and s_m and e_m and en_m and se_m and deb_m and fd_m

            if r_m: attr_correct["rank"] += 1
            if s_m: attr_correct["suit"] += 1
            if e_m: attr_correct["edition"] += 1
            if en_m: attr_correct["enhancement"] += 1
            if se_m: attr_correct["seal"] += 1
            if deb_m: attr_correct["debuffed"] += 1
            if fd_m: attr_correct["face_down"] += 1
            if sel_m: attr_correct["selected"] += 1
            if all_m: attr_correct["exact_card"] += 1

            if not all_m:
                frame_perfect = False
                mismatches.append({
                    "frame": jf.stem,
                    "card_idx": i + 1,
                    "pred": f"{c.rank} of {c.suit} | Ed:{c.edition} | Enh:{c.enhancement} | Seal:{c.seal} | Deb:{c.debuffed} | FD:{c.face_down}",
                    "gt": f"{g.get('rank')} of {g.get('suit')} | Ed:{g.get('edition')} | Enh:{g.get('enhancement')} | Seal:{g.get('seal')} | Deb:{g.get('debuffed', False)} | FD:{g.get('face_down', False)}"
                })

        if frame_perfect and n_pairs == n_gt:
            frames_all_correct += 1

        if idx % 50 == 0 or idx == total_frames:
            print(f"Progress: {idx}/{total_frames} frames processed...")

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Frames Evaluated           : {total_frames}")
    print(f"Total Ground Truth Cards   : {total_gt_cards}")
    print(f"Total Detected Cards       : {total_detected_cards}")
    print(f"Frames Card Count Matched  : {frames_count_match}/{total_frames} ({frames_count_match / total_frames * 100:.2f}%)")
    print(f"Entire Hand 100% Correct   : {frames_all_correct}/{total_frames} ({frames_all_correct / total_frames * 100:.2f}%)")
    print("-" * 70)

    if evaluated_pairs > 0:
        print(f"Exact Card Accuracy (All 7): {attr_correct['exact_card']}/{evaluated_pairs} ({attr_correct['exact_card'] / evaluated_pairs * 100:.2f}%)")
        print(f"   Rank Accuracy           : {attr_correct['rank']}/{evaluated_pairs} ({attr_correct['rank'] / evaluated_pairs * 100:.2f}%)")
        print(f"   Suit Accuracy           : {attr_correct['suit']}/{evaluated_pairs} ({attr_correct['suit'] / evaluated_pairs * 100:.2f}%)")
        print(f"   Edition Accuracy        : {attr_correct['edition']}/{evaluated_pairs} ({attr_correct['edition'] / evaluated_pairs * 100:.2f}%)")
        print(f"   Enhancement Accuracy    : {attr_correct['enhancement']}/{evaluated_pairs} ({attr_correct['enhancement'] / evaluated_pairs * 100:.2f}%)")
        print(f"   Seal Accuracy           : {attr_correct['seal']}/{evaluated_pairs} ({attr_correct['seal'] / evaluated_pairs * 100:.2f}%)")
        print(f"   Debuffed Accuracy       : {attr_correct['debuffed']}/{evaluated_pairs} ({attr_correct['debuffed'] / evaluated_pairs * 100:.2f}%)")
        print(f"   Face Down Accuracy      : {attr_correct['face_down']}/{evaluated_pairs} ({attr_correct['face_down'] / evaluated_pairs * 100:.2f}%)")
        print(f"   Selected (Raised) Acc   : {attr_correct['selected']}/{evaluated_pairs} ({attr_correct['selected'] / evaluated_pairs * 100:.2f}%)")

    if mismatches:
        print("-" * 70)
        print(f"Sample Mismatches (showing up to 5 of {len(mismatches)} total):")
        for m in mismatches[:5]:
            print(f"  [{m['frame']}] Card {m['card_idx']}:")
            print(f"     Pred: {m['pred']}")
            print(f"     GT  : {m['gt']}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Two-Stage Card Recognition Pipeline")
    parser.add_argument("--image", type=str, default=None, help="Path to single input image")
    parser.add_argument("--frames", type=int, default=None, help="Number of frames to evaluate (e.g. 500, 1000)")
    parser.add_argument("--split", type=str, default="val", choices=["val", "train", "all"], help="Which data split to benchmark: 'val' (unseen validation set, default), 'train', or 'all'")
    parser.add_argument("--conf", type=float, default=0.60, help="Detector confidence threshold (default: 0.60)")
    parser.add_argument("--detector", type=str, default=None, help="Path to detector weights")
    parser.add_argument("--classifier", type=str, default=None, help="Path to classifier weights")
    args = parser.parse_args()

    pipeline = CardRecognitionPipeline(
        detector_path=args.detector,
        classifier_path=args.classifier
    )

    if args.image:
        evaluate_single_image(pipeline, args.image)
    else:
        evaluate_benchmark(pipeline, num_frames=args.frames, split=args.split, conf=args.conf)
