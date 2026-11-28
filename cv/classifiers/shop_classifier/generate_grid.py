"""
Shop Recognition Visual Validation Mosaic Generator
===================================================
Renders high-resolution validation mosaic grids for the two-stage shop recognition pipeline:
- Green bounding box: Exact match (Key and Edition match ground truth)
- Red bounding box: Mismatch (shows predicted vs true attribute)
- Displays item name, edition, and stickers clearly on each detection.
"""

import json
import math
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

CURRENT_DIR = Path(__file__).resolve().parent
DATASET_DIR = CURRENT_DIR / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
JSON_LABELS_DIR = DATASET_DIR / "labels_json"
OUTPUT_FIG_DIR = CURRENT_DIR / "figures"

from pipeline import ShopRecognitionPipeline

COLOR_CORRECT = (34, 177, 76)   # RGB Green
COLOR_WRONG   = (220, 20, 20)   # RGB Red
COLOR_TEXT    = (255, 255, 255) # RGB White
COLOR_BG      = (20, 20, 24)    # Dark background


def compute_iou(box_a, box_b):
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])
    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h
    box_a_area = max(0.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    box_b_area = max(0.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    union = box_a_area + box_b_area - inter_area
    return inter_area / union if union > 0 else 0.0


def render_item_on_image(draw, box, item, gt=None, font=None, drawn_labels=None):
    if drawn_labels is None:
        drawn_labels = []

    x1, y1, x2, y2 = box

    all_match = True
    mismatches = []
    if gt:
        if item.key != gt.get("key"):
            all_match = False
            mismatches.append(f"Key: {item.key} (True: {gt.get('key')})")
        if item.edition != gt.get("edition", "None"):
            all_match = False
            mismatches.append(f"Ed: {item.edition} (True: {gt.get('edition')})")
        if item.eternal != gt.get("eternal", False):
            all_match = False
            mismatches.append(f"Et: {item.eternal}")
        if item.perishable != gt.get("perishable", False):
            all_match = False
            mismatches.append(f"Per: {item.perishable}")
        if item.rental != gt.get("rental", False):
            all_match = False
            mismatches.append(f"Rent: {item.rental}")

    box_color = COLOR_CORRECT if all_match else COLOR_WRONG
    draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)

    tags = []
    if all_match:
        tags.append(item.display_name)
    else:
        tags.extend(mismatches)

    tag_x = x1 + 2
    tag_y = max(2, y1 - 18)

    for tag in tags:
        tw, th = len(tag) * 7, 14
        if font and hasattr(font, "getbbox"):
            bbox = font.getbbox(tag)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1] + 2

        # Check collisions and stack
        shifted = True
        while shifted:
            shifted = False
            for (ox1, oy1, ox2, oy2) in drawn_labels:
                if not (tag_x + tw < ox1 or tag_x > ox2 or tag_y + th < oy1 or tag_y > oy2):
                    tag_y = oy2 + 2
                    shifted = True
                    break

        drawn_labels.append((tag_x, tag_y, tag_x + tw + 4, tag_y + th + 2))
        draw.rectangle([tag_x, tag_y, tag_x + tw + 4, tag_y + th + 2], fill=box_color)
        draw.text((tag_x + 2, tag_y), tag, fill=COLOR_TEXT, font=font)


def generate_shop_validation_grid(
    pipeline,
    max_frames=16,
    grid_size=(4, 4),
    output_path=None,
    target_width=2560
):
    OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = OUTPUT_FIG_DIR / "shop_val_examples.jpg"

    json_files = sorted(list(JSON_LABELS_DIR.glob("*.json")))
    if not json_files:
        print("[Error] No JSON labels found in dataset/labels_json/!")
        return

    # Take evenly spaced samples across validation set
    step = max(1, len(json_files) // max_frames)
    selected_jsons = json_files[::step][:max_frames]

    cols, rows = grid_size
    # Determine average aspect ratio from samples
    sample_img = None
    for jf in selected_jsons:
        p = IMAGES_DIR / f"{jf.stem}.png"
        if p.exists():
            sample_img = p
            break

    if not sample_img:
        print("[Error] No matching image files found.")
        return

    # Choose a standard cell height so cells never get squished
    cell_w = target_width // cols
    cell_h = int(cell_w * 0.60) # standard card crop aspect ratio
    total_h = cell_h * rows

    mosaic = Image.new("RGB", (target_width, total_h), COLOR_BG)
    draw_mosaic = ImageDraw.Draw(mosaic)

    print(f"\nGenerating {target_width}x{total_h}px Validation Grid ({len(selected_jsons)} frames)...")

    for i, jf in enumerate(selected_jsons):
        r = i // cols
        c = i % cols
        cell_x = c * cell_w
        cell_y = r * cell_h

        img_p = IMAGES_DIR / f"{jf.stem}.png"
        if not img_p.exists():
            continue

        with Image.open(img_p) as cell_img:
            cell_img = cell_img.convert("RGB")
            draw_cell = ImageDraw.Draw(cell_img)

            # Predict items
            preds = pipeline.predict(cell_img)

            # Ground truth items
            with open(jf, "r", encoding="utf-8") as f:
                gt_items = json.load(f)

            gt_boxes = []
            for g in gt_items:
                bp = g.get("box_pixel", [0, 0, 0, 0])
                gt_boxes.append([bp[0], bp[1], bp[0] + bp[2], bp[1] + bp[3]])

            drawn_labels = []
            # Match predictions to GT
            for p in preds:
                best_gt = None
                best_iou = 0.0
                for g_idx, g_box in enumerate(gt_boxes):
                    iou = compute_iou(p.box, g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gt_items[g_idx]

                gt_match = best_gt if best_iou >= 0.40 else None
                render_item_on_image(draw_cell, p.box, p, gt=gt_match, drawn_labels=drawn_labels)

            # Preserve exact aspect ratio via letterboxing / fitting
            cw_avail = cell_w - 6
            ch_avail = cell_h - 6
            im_w, im_h = cell_img.size
            scale = min(cw_avail / im_w, ch_avail / im_h)
            new_w = max(1, int(im_w * scale))
            new_h = max(1, int(im_h * scale))

            resized_cell = cell_img.resize((new_w, new_h), Image.BILINEAR)

            # Center inside cell
            offset_x = cell_x + 3 + (cw_avail - new_w) // 2
            offset_y = cell_y + 3 + (ch_avail - new_h) // 2
            mosaic.paste(resized_cell, (offset_x, offset_y))

            # Cell border
            draw_mosaic.rectangle([cell_x, cell_y, cell_x + cell_w, cell_y + cell_h], outline=(40, 40, 50), width=1)

    mosaic.save(str(output_path), quality=92)
    print(f"[Done] Validation grid saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate visual validation examples grid")
    parser.add_argument("--frames", type=int, default=16, help="Number of frames in grid (default: 16)")
    parser.add_argument("--cols", type=int, default=4, help="Grid columns (default: 4)")
    parser.add_argument("--rows", type=int, default=4, help="Grid rows (default: 4)")
    parser.add_argument("--output", type=str, default=None, help="Output image file path")
    args = parser.parse_args()

    pipeline = ShopRecognitionPipeline()
    generate_shop_validation_grid(
        pipeline,
        max_frames=args.frames,
        grid_size=(args.cols, args.rows),
        output_path=Path(args.output) if args.output else None
    )
