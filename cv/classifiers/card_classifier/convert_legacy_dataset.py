"""
Convert Legacy Multi-Box Labels to Two-Stage Dataset Format
===========================================================
Parses legacy Balatro label files where cards had multiple sub-boxes
(Identity, Edition, Enhancement, Seal, Card_Selected) into:
1. Stage 1 YOLO labels: frame_XXXX.txt with single class 0 (card)
2. Stage 2 Structured JSON labels: frame_XXXX.json with exact card attributes
   {rank, suit, edition, enhancement, seal, selected, box_yolo, box_pixel}
"""

import json
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
DATASET_DIR = CURRENT_DIR / "dataset"
LABELS_SRC = DATASET_DIR / "labels"
CLASSES_FILE = DATASET_DIR / "classes.txt"

def load_class_names():
    with open(CLASSES_FILE, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def convert_dataset(img_w=1102, img_h=330):
    class_names = load_class_names()
    label_files = sorted(list(LABELS_SRC.glob("frame_*.txt")))
    print(f"Converting {len(label_files)} legacy label files in {LABELS_SRC}...")

    # Directory for Stage 1 card-only YOLO labels
    card_labels_dir = DATASET_DIR / "labels_card"
    card_labels_dir.mkdir(parents=True, exist_ok=True)

    json_labels_dir = DATASET_DIR / "labels_json"
    json_labels_dir.mkdir(parents=True, exist_ok=True)

    converted_cards_count = 0

    for txt_path in label_files:
        stem = txt_path.stem
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = [l.strip().split() for l in f if l.strip()]

        identities = []
        editions = []
        enhancements = []
        seals = []
        selected_boxes = []

        for l in lines:
            c = int(l[0])
            if c >= len(class_names):
                continue
            x, y, w, h = [float(v) for v in l[1:5]]
            name = class_names[c]
            item = {
                "name": name, "x": x, "y": y, "w": w, "h": h,
                "x1": x - w / 2, "x2": x + w / 2,
                "y1": y - h / 2, "y2": y + h / 2
            }
            if name.startswith("Edition_"):
                editions.append(item)
            elif name.startswith("Enhancement_"):
                enhancements.append(item)
            elif name.startswith("Seal_"):
                seals.append(item)
            elif name == "Card_Selected":
                selected_boxes.append(item)
            else:
                identities.append(item)

        identities.sort(key=lambda it: it["x"])
        cards = []
        yolo_lines = []

        for k, id_item in enumerate(identities):
            card_x = id_item["x"]
            prev_x = identities[k - 1]["x"] if k > 0 else -1.0
            next_x = identities[k + 1]["x"] if k < len(identities) - 1 else 2.0
            left_bound = (prev_x + card_x) / 2.0
            right_bound = (card_x + next_x) / 2.0

            raw_name = id_item["name"]
            if raw_name == "Stone_Card":
                rank, suit = "Stone", "None"
            else:
                parts = raw_name.split("_")
                rank = parts[0]
                suit = parts[1] if len(parts) > 1 else "None"

            card_edition = "None"
            for ed in editions:
                if left_bound <= ed["x"] <= right_bound:
                    card_edition = ed["name"].replace("Edition_", "")
                    break

            card_enhancement = "None"
            for enh in enhancements:
                if left_bound <= enh["x"] <= right_bound:
                    card_enhancement = enh["name"].replace("Enhancement_", "").replace("Card", "")
                    break
            if raw_name == "Stone_Card":
                card_enhancement = "Stone"

            card_seal = "None"
            for s in seals:
                if left_bound <= s["x"] <= right_bound:
                    card_seal = s["name"].replace("Seal_", "")
                    break

            is_selected = False
            for sel in selected_boxes:
                if sel["x1"] <= card_x <= sel["x2"]:
                    is_selected = True
                    break

            # Card bounding box
            card_w = min(0.12, (next_x - card_x) * 1.05) if k < len(identities) - 1 else 0.12
            card_h = 0.68
            card_y = 0.58 if not is_selected else 0.38

            # Pixel bounding box
            px1 = max(0, int((card_x - card_w / 2) * img_w))
            py1 = max(0, int((card_y - card_h / 2) * img_h))
            pw = min(img_w - px1, int(card_w * img_w))
            ph = min(img_h - py1, int(card_h * img_h))

            cards.append({
                "card_index": k + 1,
                "box_yolo": [card_x, card_y, card_w, card_h],
                "box_pixel": [px1, py1, pw, ph],
                "rank": rank,
                "suit": suit,
                "edition": card_edition,
                "enhancement": card_enhancement,
                "seal": card_seal,
                "selected": is_selected,
                "skin": "default",
                "high_contrast": False
            })

            # Stage 1 YOLO label (class 0: card)
            yolo_lines.append(f"0 {card_x:.6f} {card_y:.6f} {card_w:.6f} {card_h:.6f}")
            converted_cards_count += 1

        # Write Stage 1 label
        with open(card_labels_dir / f"{stem}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

        # Write Stage 2 JSON label
        with open(json_labels_dir / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(cards, f, indent=2)

    print(f"Successfully converted {len(label_files)} frames ({converted_cards_count} total cards).")
    print(f"Stage 1 YOLO labels: {card_labels_dir}")
    print(f"Stage 2 JSON labels: {json_labels_dir}")

if __name__ == "__main__":
    convert_dataset()
