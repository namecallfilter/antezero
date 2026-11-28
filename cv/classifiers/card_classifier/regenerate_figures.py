"""
Two-Stage Validation Visualization Generator
============================================
Renders high-resolution (2880px) validation mosaic grids for the two-stage pipeline:
- GREEN (34, 177, 76): True Positive (all 5 attributes match ground truth).
- RED (220, 20, 20): Mismatch (shows exact predicted vs ground truth attribute).
- Smart anti-collision vertical label stacking to avoid overlapping text.
"""

import json
import math
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CURRENT_DIR = Path(__file__).resolve().parent
DATASET_DIR = CURRENT_DIR / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
JSON_LABELS_DIR = DATASET_DIR / "labels_json"
OUTPUT_FIG_DIR = CURRENT_DIR / "figures"

from pipeline import CardRecognitionPipeline

COLOR_CORRECT = (34, 177, 76)   # RGB Green
COLOR_WRONG   = (220, 20, 20)   # RGB Red
COLOR_TEXT    = (255, 255, 255) # RGB White


def render_card_on_image(draw, box, card, gt=None, font=None, drawn_labels=None):
    if drawn_labels is None:
        drawn_labels = []

    x1, y1, x2, y2 = box

    # Determine correctness against ground truth
    all_match = True
    mismatches = []
    if gt:
        if card.rank != gt.get("rank"):
            all_match = False
            mismatches.append(f"Rank: {card.rank} (True: {gt.get('rank')})")
        if card.suit != gt.get("suit"):
            all_match = False
            mismatches.append(f"Suit: {card.suit} (True: {gt.get('suit')})")
        if card.edition != gt.get("edition"):
            all_match = False
            mismatches.append(f"Ed: {card.edition} (True: {gt.get('edition')})")
        if card.enhancement != gt.get("enhancement"):
            all_match = False
            mismatches.append(f"Enh: {card.enhancement} (True: {gt.get('enhancement')})")
        if card.seal != gt.get("seal"):
            all_match = False
            mismatches.append(f"Seal: {card.seal} (True: {gt.get('seal')})")

    box_color = COLOR_CORRECT if all_match else COLOR_WRONG

    # 1. Draw bounding box outline
    draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)

    # 2. Build text tags
    tags = []
    if all_match:
        tags.append(f"{card.rank} of {card.suit}")
        if card.edition != "None":
            tags.append(f"Ed: {card.edition}")
        if card.enhancement != "None":
            tags.append(f"Enh: {card.enhancement}")
        if card.seal != "None":
            tags.append(f"Seal: {card.seal}")
    else:
        tags.extend(mismatches)

    # 3. Smart vertical collision avoidance stacking
    tag_x = x1 + 2
    tag_y = y1 + 2

    for tag in tags:
        # Approximate font bbox
        tw, th = 70, 14
        if font and hasattr(font, "getbbox"):
            bbox = font.getbbox(tag)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1] + 2

        # Shift down if colliding with existing tag
        shifted = True
        while shifted:
            shifted = False
            for (ox1, oy1, ox2, oy2) in drawn_labels:
                if not (tag_x + tw < ox1 or tag_x > ox2 or tag_y + th < oy1 or tag_y > oy2):
                    tag_y = oy2 + 2
                    shifted = True
                    break

        drawn_labels.append((tag_x, tag_y, tag_x + tw, tag_y + th))
        draw.rectangle([tag_x, tag_y, tag_x + tw + 2, tag_y + th + 2], fill=box_color)
        draw.text((tag_x + 1, tag_y), tag, fill=COLOR_TEXT, font=font)


def generate_validation_grid(
    pipeline,
    max_frames=16,
    grid_size=(4, 4),
    output_path=None,
    target_res=2880
):
    OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = OUTPUT_FIG_DIR / "val_twostage_pred.jpg"

    # Select validation frames
    json_files = sorted(list(JSON_LABELS_DIR.glob("*.json")))
    if not json_files:
        print("No JSON labels found in dataset/labels_json/!")
        return

    # Take evenly spaced samples across validation set
    step = max(1, len(json_files) // max_frames)
    selected_jsons = json_files[::step][:max_frames]

    cols, rows = grid_size
    # Hand screenshots are 1102x330 (approx 3.34:1 aspect ratio)
    cell_w = target_res // cols
    cell_h = int(cell_w * (330 / 1102))
    total_mosaic_h = cell_h * rows

    mosaic = Image.new("RGB", (target_res, total_mosaic_h), (30, 30, 30))
    draw_mosaic = ImageDraw.Draw(mosaic)

    total_cards = 0
    correct_cards = 0

    print(f"\nGenerating {target_res}x{total_mosaic_h}px Validation Mosaic ({len(selected_jsons)} frames)...")

    for i, jf in enumerate(selected_jsons):
        r = i // cols
        c = i % cols
        cell_x = c * cell_w
        cell_y = r * cell_h

        img_path = IMAGES_DIR / f"{jf.stem}.png"
        if not img_path.exists():
            img_path = IMAGES_DIR / f"{jf.stem}.jpg"
        if not img_path.exists():
            continue

        with Image.open(img_path) as orig_im:
            orig_im = orig_im.convert("RGB")
            ow, oh = orig_im.size
            
            # Predict
            results = pipeline.predict(orig_im)

            # Ground truth
            with open(jf, "r", encoding="utf-8") as f:
                gt_cards = json.load(f)

            # Draw predictions on cell image
            draw_cell = ImageDraw.Draw(orig_im)
            drawn_labels = []

            for k, card in enumerate(results):
                gt = gt_cards[k] if k < len(gt_cards) else None
                total_cards += 1
                if gt:
                    match = (
                        card.rank == gt.get("rank") and
                        card.suit == gt.get("suit") and
                        card.edition == gt.get("edition") and
                        card.enhancement == gt.get("enhancement") and
                        card.seal == gt.get("seal")
                    )
                    if match:
                        correct_cards += 1

                render_card_on_image(draw_cell, card.box, card, gt=gt, drawn_labels=drawn_labels)

            # Resize cell to fit mosaic slot
            resized_cell = orig_im.resize((cell_w, cell_h), Image.Resampling.BILINEAR)
            mosaic.paste(resized_cell, (cell_x, cell_y))
            draw_mosaic.rectangle([cell_x, cell_y, cell_x + cell_w, cell_y + cell_h], outline=(100, 100, 100), width=1)
            draw_mosaic.text((cell_x + 5, cell_y + 5), f"{jf.stem}", fill=(200, 200, 200))

    mosaic.save(output_path, quality=95)
    acc = (correct_cards / total_cards * 100) if total_cards > 0 else 0
    print(f"[Done] Validation grid saved to: {output_path}")
    print(f"Overall Exact Card Accuracy: {correct_cards}/{total_cards} ({acc:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Two-Stage Validation Figures")
    parser.add_argument("--detector", type=str, default=None, help="Detector weights")
    parser.add_argument("--classifier", type=str, default=None, help="Classifier weights")
    parser.add_argument("--out", type=str, default=None, help="Output image file")
    args = parser.parse_args()

    pipeline = CardRecognitionPipeline(detector_path=args.detector, classifier_path=args.classifier)
    generate_validation_grid(pipeline, output_path=args.out)
