"""
Evaluate Two-Stage View Deck Recognition Pipeline
=================================================
Runs Stage 1 deck YOLO detector + Stage 2 multi-head card classifier on a single
image or validation set, performs IoU-based 1-to-1 matching against ground-truth
deck labels, and reports detection, count, exact-card, and greyed breakdown metrics.
"""

import argparse
import json
import random
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
DATASET_DIR = CURRENT_DIR / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
JSON_LABELS_DIR = DATASET_DIR / "labels_json"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from classifiers.deck_classifier.pipeline import DeckRecognitionPipeline


def compute_iou(box_a: list[float] | tuple[float, ...], box_b: list[float] | tuple[float, ...]) -> float:
    """Computes intersection-over-union between two `[x1, y1, x2, y2]` boxes."""
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))
    union = area_a + area_b - inter
    return (inter / union) if union > 0.0 else 0.0


def match_predictions_to_gt(
    pred_boxes: list[list[float]],
    gt_boxes: list[list[float]],
    iou_thresh: float = 0.40,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """
    Greedy 1-to-1 matching sorted by descending IoU above `iou_thresh`.
    Returns `(matches, unmatched_pred_indices, unmatched_gt_indices)`.
    """
    candidates: list[tuple[float, int, int]] = []
    for pi, p_box in enumerate(pred_boxes):
        for gi, g_box in enumerate(gt_boxes):
            iou = compute_iou(p_box, g_box)
            if iou >= iou_thresh:
                candidates.append((iou, pi, gi))

    candidates.sort(key=lambda item: item[0], reverse=True)

    used_preds: set[int] = set()
    used_gts: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for iou, pi, gi in candidates:
        if pi in used_preds or gi in used_gts:
            continue
        used_preds.add(pi)
        used_gts.add(gi)
        matches.append((pi, gi, iou))

    unmatched_preds = [i for i in range(len(pred_boxes)) if i not in used_preds]
    unmatched_gts = [i for i in range(len(gt_boxes)) if i not in used_gts]
    return matches, unmatched_preds, unmatched_gts


def _gt_box_xyxy(gt_entry: dict) -> list[float]:
    px, py, pw, ph = gt_entry["box_pixel"]
    return [float(px), float(py), float(px) + float(pw), float(py) + float(ph)]


def evaluate_single_image(pipeline: DeckRecognitionPipeline, image_path: str | Path, conf: float = 0.45, iou: float = 0.55):
    image_path = Path(image_path)
    print(f"\nEvaluating on image: {image_path.name}")

    results = pipeline.predict(image_path, conf=conf, iou=iou)
    gt_file = JSON_LABELS_DIR / f"{image_path.stem}.json"
    gt_cards = []
    if gt_file.exists():
        with open(gt_file, "r", encoding="utf-8") as f:
            gt_cards = json.load(f)

    pred_boxes = [list(map(float, c.box)) for c in results]
    gt_boxes = [_gt_box_xyxy(g) for g in gt_cards]
    matches, unmatched_preds, unmatched_gts = match_predictions_to_gt(pred_boxes, gt_boxes) if gt_cards else ([], list(range(len(results))), [])
    gt_by_pred = {pi: (gi, m_iou) for pi, gi, m_iou in matches}

    print("\n" + "=" * 75)
    print(f"RECONSTRUCTED DECK ({len(results)} detected vs {len(gt_cards)} GT)")
    print("=" * 75)

    for pi, c in enumerate(results):
        match_info = gt_by_pred.get(pi)
        match_str = ""
        gt = None
        if match_info is not None:
            gi, m_iou = match_info
            gt = gt_cards[gi]
            all_match = (
                c.rank == gt.get("rank")
                and c.suit == gt.get("suit")
                and c.edition == gt.get("edition")
                and c.enhancement == gt.get("enhancement")
                and c.seal == gt.get("seal")
                and c.debuffed == gt.get("debuffed", False)
                and c.face_down == gt.get("face_down", False)
            )
            match_str = f" [ALL CORRECT | IoU={m_iou:.2f}]" if all_match else f" [MISMATCH | IoU={m_iou:.2f}]"
        elif gt_cards:
            match_str = " [FALSE POSITIVE]"

        print(f"Card {c.index:3d}: {c.display_name}{match_str}")
        if gt is not None and "[MISMATCH" in match_str:
            print(
                f"   --> GT: {gt.get('rank')} of {gt.get('suit')} | Ed: {gt.get('edition')} | "
                f"Enh: {gt.get('enhancement')} | Seal: {gt.get('seal')} | "
                f"Deb: {gt.get('debuffed', False)} | Greyed: {gt.get('greyed', False)}"
            )

    if unmatched_gts:
        print(f"\nMissed GT cards ({len(unmatched_gts)}):")
        for gi in unmatched_gts[:10]:
            gt = gt_cards[gi]
            print(f"  - GT #{gi + 1}: {gt.get('rank')} of {gt.get('suit')} (box={gt.get('box_pixel')})")
    if unmatched_preds:
        print(f"Unmatched predictions: {len(unmatched_preds)}")


def evaluate_benchmark(
    pipeline: DeckRecognitionPipeline,
    num_frames: int | None = None,
    split: str = "val",
    conf: float = 0.45,
    iou: float = 0.55,
    match_iou: float = 0.40,
    seed: int = 42,
    val_ratio: float = 0.20,
):
    json_files = sorted(list(JSON_LABELS_DIR.glob("*.json")))
    if not json_files:
        print("No JSON labels found in dataset/labels_json/!")
        return

    valid_frames = []
    for jf in json_files:
        img_path = IMAGES_DIR / f"{jf.stem}.png"
        if not img_path.exists():
            img_path = IMAGES_DIR / f"{jf.stem}.jpg"
        if img_path.exists():
            valid_frames.append((img_path, jf))

    rng = random.Random(seed)
    shuffled = list(valid_frames)
    rng.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0

    if split == "val":
        target_pairs = shuffled[:val_count]
        split_desc = f"STRICT UNSEEN VALIDATION SET ({len(target_pairs)} frames, {val_ratio * 100:.0f}% split)"
    elif split == "train":
        target_pairs = shuffled[val_count:]
        split_desc = f"TRAINING SET ({len(target_pairs)} frames)"
    else:
        target_pairs = valid_frames
        split_desc = f"ALL DECK FRAMES ({len(target_pairs)} frames)"

    if num_frames and num_frames > 0:
        target_pairs = target_pairs[:num_frames]

    print("\n" + "=" * 75)
    print(f"DECK EVALUATION BENCHMARK: {split_desc}")
    print(f"Testing {len(target_pairs)} frames (conf={conf:.2f}, nms_iou={iou:.2f}, match_iou={match_iou:.2f})")
    print("=" * 75)

    total_frames = len(target_pairs)
    frames_count_match = 0
    frames_all_correct = 0

    total_gt = 0
    total_pred = 0
    total_tp = 0

    attr_correct = {
        "rank": 0,
        "suit": 0,
        "edition": 0,
        "enhancement": 0,
        "seal": 0,
        "debuffed": 0,
        "face_down": 0,
        "exact_card": 0,
    }
    greyed_exact = 0
    greyed_total = 0
    non_greyed_exact = 0
    non_greyed_total = 0
    mismatches = []

    for idx, (img_path, jf) in enumerate(target_pairs, 1):
        with open(jf, "r", encoding="utf-8") as f:
            gt_cards = json.load(f)

        preds = pipeline.predict(img_path, conf=conf, iou=iou)
        pred_boxes = [list(map(float, c.box)) for c in preds]
        gt_boxes = [_gt_box_xyxy(g) for g in gt_cards]

        matches, unmatched_preds, unmatched_gts = match_predictions_to_gt(pred_boxes, gt_boxes, iou_thresh=match_iou)

        total_gt += len(gt_cards)
        total_pred += len(preds)
        total_tp += len(matches)

        if len(gt_cards) == len(preds):
            frames_count_match += 1

        frame_perfect = (len(unmatched_preds) == 0 and len(unmatched_gts) == 0)

        for pi, gi, _ in matches:
            c = preds[pi]
            g = gt_cards[gi]

            r_m = c.rank == g.get("rank")
            s_m = c.suit == g.get("suit")
            e_m = c.edition == g.get("edition")
            en_m = c.enhancement == g.get("enhancement")
            se_m = c.seal == g.get("seal")
            deb_m = c.debuffed == g.get("debuffed", False)
            fd_m = c.face_down == g.get("face_down", False)
            all_m = r_m and s_m and e_m and en_m and se_m and deb_m and fd_m

            if r_m: attr_correct["rank"] += 1
            if s_m: attr_correct["suit"] += 1
            if e_m: attr_correct["edition"] += 1
            if en_m: attr_correct["enhancement"] += 1
            if se_m: attr_correct["seal"] += 1
            if deb_m: attr_correct["debuffed"] += 1
            if fd_m: attr_correct["face_down"] += 1
            if all_m: attr_correct["exact_card"] += 1

            is_greyed = bool(g.get("greyed", False))
            if is_greyed:
                greyed_total += 1
                if all_m:
                    greyed_exact += 1
            else:
                non_greyed_total += 1
                if all_m:
                    non_greyed_exact += 1

            if not all_m:
                frame_perfect = False
                mismatches.append({
                    "frame": jf.stem,
                    "card_idx": gi + 1,
                    "pred": f"{c.rank} of {c.suit} | Ed:{c.edition} | Enh:{c.enhancement} | Seal:{c.seal} | Deb:{c.debuffed}",
                    "gt": f"{g.get('rank')} of {g.get('suit')} | Ed:{g.get('edition')} | Enh:{g.get('enhancement')} | Seal:{g.get('seal')} | Deb:{g.get('debuffed', False)} | Grey:{is_greyed}",
                })

        if frame_perfect:
            frames_all_correct += 1

        if idx % 50 == 0 or idx == total_frames:
            print(f"Progress: {idx}/{total_frames} frames processed...")

    precision = (total_tp / total_pred) if total_pred > 0 else 0.0
    recall = (total_tp / total_gt) if total_gt > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print("\n" + "=" * 75)
    print("VIEW DECK BENCHMARK RESULTS")
    print("=" * 75)
    print(f"Frames Evaluated           : {total_frames}")
    print(f"Total GT Cards             : {total_gt}")
    print(f"Total Detected Cards       : {total_pred}")
    print(f"Matched TP (IoU >= {match_iou:.2f})   : {total_tp}")
    print(f"Detection Precision / Rec  : {precision * 100:.2f}% / {recall * 100:.2f}% (F1: {f1 * 100:.2f}%)")
    if total_frames > 0:
        print(f"Frames Card Count Matched  : {frames_count_match}/{total_frames} ({frames_count_match / total_frames * 100:.2f}%)")
        print(f"Entire Deck 100% Correct   : {frames_all_correct}/{total_frames} ({frames_all_correct / total_frames * 100:.2f}%)")
    print("-" * 75)

    if total_tp > 0:
        print(f"Exact Card Accuracy (All 7): {attr_correct['exact_card']}/{total_tp} ({attr_correct['exact_card'] / total_tp * 100:.2f}%)")
        print(f"   Rank Accuracy           : {attr_correct['rank']}/{total_tp} ({attr_correct['rank'] / total_tp * 100:.2f}%)")
        print(f"   Suit Accuracy           : {attr_correct['suit']}/{total_tp} ({attr_correct['suit'] / total_tp * 100:.2f}%)")
        print(f"   Edition Accuracy        : {attr_correct['edition']}/{total_tp} ({attr_correct['edition'] / total_tp * 100:.2f}%)")
        print(f"   Enhancement Accuracy    : {attr_correct['enhancement']}/{total_tp} ({attr_correct['enhancement'] / total_tp * 100:.2f}%)")
        print(f"   Seal Accuracy           : {attr_correct['seal']}/{total_tp} ({attr_correct['seal'] / total_tp * 100:.2f}%)")
        print(f"   Debuffed Accuracy       : {attr_correct['debuffed']}/{total_tp} ({attr_correct['debuffed'] / total_tp * 100:.2f}%)")
        print(f"   Face Down Accuracy      : {attr_correct['face_down']}/{total_tp} ({attr_correct['face_down'] / total_tp * 100:.2f}%)")
        if non_greyed_total > 0:
            print(f"   Non-Greyed Exact Match  : {non_greyed_exact}/{non_greyed_total} ({non_greyed_exact / non_greyed_total * 100:.2f}%)")
        if greyed_total > 0:
            print(f"   Greyed (Played) Exact   : {greyed_exact}/{greyed_total} ({greyed_exact / greyed_total * 100:.2f}%)")

    if mismatches:
        print("-" * 75)
        print(f"Sample Attribute Mismatches (showing up to 5 of {len(mismatches)}):")
        for m in mismatches[:5]:
            print(f"  [{m['frame']}] Card {m['card_idx']}:")
            print(f"     Pred: {m['pred']}")
            print(f"     GT  : {m['gt']}")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Two-Stage View Deck Recognition Pipeline")
    parser.add_argument("--image", type=str, default=None, help="Path to single input image")
    parser.add_argument("--frames", type=int, default=None, help="Number of frames to evaluate")
    parser.add_argument("--split", type=str, default="val", choices=["val", "train", "all"], help="Data split to evaluate")
    parser.add_argument("--conf", type=float, default=0.45, help="Detector confidence threshold (default: 0.45)")
    parser.add_argument("--iou", type=float, default=0.55, help="Detector NMS IoU threshold (default: 0.55)")
    parser.add_argument("--match-iou", type=float, default=0.40, help="GT matching IoU threshold (default: 0.40)")
    parser.add_argument("--detector", type=str, default=None, help="Path to deck detector weights")
    parser.add_argument("--classifier", type=str, default=None, help="Path to Stage 2 card classifier weights")
    args = parser.parse_args()

    pipeline = DeckRecognitionPipeline(
        detector_path=args.detector,
        classifier_path=args.classifier,
    )

    if args.image:
        evaluate_single_image(pipeline, args.image, conf=args.conf, iou=args.iou)
    else:
        evaluate_benchmark(
            pipeline,
            num_frames=args.frames,
            split=args.split,
            conf=args.conf,
            iou=args.iou,
            match_iou=args.match_iou,
        )
