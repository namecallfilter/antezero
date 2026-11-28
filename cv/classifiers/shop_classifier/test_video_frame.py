"""
Test Shop Classifier on Video Frames and Real Screenshots
==========================================================
Extracts a frame from a video file at a specified timestamp (or loads a raw screenshot),
extracts and runs inference across all Balatro shop regions:
  - Top Joker Bar
  - Top Consumables Area
  - Active Shop Items / Open Pack
  - Shop Voucher Shelf
  - Shop Booster Shelf

Outputs predicted item attributes (Key, Type, Edition, Eternal/Perishable/Rental/Debuffed)
and saves an annotated visual diagnosis image highlighting all shop regions and detected cards.

Usage:
    # Test a video frame at a timestamp (e.g. MM:SS, HH:MM:SS, or seconds):
    python test_video_frame.py --video assets/sample_inputs/videos/bu_full.mp4 --time 02:45
    python test_video_frame.py --video assets/sample_inputs/videos/bu_full.mp4 --time 165.0

    # Test an existing full screenshot:
    python test_video_frame.py --image assets/sample_inputs/shop.jpg

    # Test an image that is already cropped to a single shop ROI:
    python test_video_frame.py --image path/to/roi_crop.png --already-cropped
"""

import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR / "pipeline.py").exists():
    CLASSIFIER_DIR = SCRIPT_DIR
    ROOT_DIR = SCRIPT_DIR.parent.parent
else:
    ROOT_DIR = SCRIPT_DIR
    CLASSIFIER_DIR = ROOT_DIR / "classifiers" / "shop_classifier"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
# pyrefly: ignore [missing-import]
from classifiers.shop_classifier.pipeline import ShopRecognitionPipeline

BASE_WIDTH = 1920.0
BASE_HEIGHT = 1080.0

# Shop ROIs in 1920x1080 Balatro space (from balatro_mod/dataset/boxes.lua)
SHOP_ROIS = {
    "joker_bar": {
        "label": "Joker Bar",
        "roi": {"x": 512.701385, "y": 20.409878, "w": 868.473724, "h": 259.967609},
        "color": (59, 130, 246)  # Blue
    },
    "consumeables": {
        "label": "Consumables",
        "roi": {"x": 1369.783812, "y": 30.836806, "w": 425.206356, "h": 270.406328},
        "color": (168, 85, 247)  # Purple
    },
    "shop": {
        "label": "Shop Items",
        "roi": {"x": 818.386186, "y": 369.181989, "w": 720.772049, "h": 280.979925},
        "color": (34, 197, 94)   # Green
    },
    "voucher": {
        "label": "Voucher Shelf",
        "roi": {"x": 656.059547, "y": 714.459732, "w": 360.743773, "h": 284.420985},
        "color": (234, 179, 8)   # Yellow
    },
    "booster": {
        "label": "Booster Shelf",
        "roi": {"x": 1071.358597, "y": 724.761198, "w": 427.707911, "h": 300.000000},
        "color": (249, 115, 22)  # Orange
    },
    "pack": {
        "label": "Booster Pack Open",
        "roi": {"x": 565.612571, "y": 588.005101, "w": 960.654887, "h": 307.239491},
        "color": (236, 72, 153)  # Pink
    }
}


# pyrefly: ignore [missing-import]
from classifiers.common import parse_timestamp, format_seconds, extract_video_frame, scale_roi_pixels


def get_scaled_roi(roi_dict: dict, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    return scale_roi_pixels(roi_dict, frame_w, frame_h, BASE_WIDTH, BASE_HEIGHT)


def save_diagnostic_image(
    full_frame: np.ndarray,
    region_results: dict,
    save_path: Path
):
    """
    Renders diagnostic visualization showing detected bounding boxes and class cards.
    """
    annotated = full_frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    total_detected = sum(len(res["items"]) for res in region_results.values())

    for reg_key, reg_info in region_results.items():
        rx, ry, rw, rh = reg_info["coords"]
        color = reg_info["color"]
        label = reg_info["label"]
        items = reg_info["items"]

        # Draw ROI outline
        cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), color, 2)
        cv2.putText(annotated, f"{label} ({len(items)})", (rx + 6, max(20, ry - 6)),
                    font, 0.55, color, 2, cv2.LINE_AA)

        # Draw detected items inside this ROI
        for item in items:
            # item.box is relative to the ROI crop
            ix1, iy1, ix2, iy2 = item.box
            gx1 = rx + ix1
            gy1 = ry + iy1
            gx2 = rx + ix2
            gy2 = ry + iy2

            box_color = (0, 255, 255)  # Bright Cyan for cards
            cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), box_color, 2)

            # Display badge
            disp_title = item.display_name
            tag_y = max(18, gy1 - 6)
            cv2.putText(annotated, disp_title, (gx1, tag_y), font, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(annotated, disp_title, (gx1, tag_y), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Header banner
    header = np.zeros((48, annotated.shape[1], 3), dtype=np.uint8)
    header[:] = (20, 20, 25)
    cv2.putText(
        header,
        f"BALATRO SHOP RECOGNITION DIAGNOSIS | Total Items Detected: {total_detected}",
        (16, 32),
        font,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    combined = np.vstack([header, annotated])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), combined)


def run_test(
    video_path: Path | None,
    image_path: Path | None,
    timestamp_str: str = "0",
    already_cropped: bool = False,
    detector_weights: Path | None = None,
    classifier_weights: Path | None = None,
    output_path: Path | None = None,
    conf_thresh: float = 0.35,
    mode: str = "shop"
):
    print("\n" + "=" * 70)
    print("BALATRO VIDEO & SCREENSHOT SHOP RECOGNITION TEST")
    print("=" * 70)

    # 1. Initialize Pipeline
    pipeline = ShopRecognitionPipeline(
        detector_path=detector_weights,
        classifier_path=classifier_weights
    )

    # 2. Extract or load image
    full_frame = None
    if video_path:
        if not video_path.exists() and (ROOT_DIR / video_path).exists():
            video_path = ROOT_DIR / video_path
        ts_sec = parse_timestamp(timestamp_str)
        print(f"Video Source      : {video_path}")
        print(f"Target Timestamp  : {format_seconds(ts_sec)} ({ts_sec:.2f}s)")

        full_frame, actual_sec, frame_idx = extract_video_frame(video_path, ts_sec)
        h, w = full_frame.shape[:2]
        print(f"Extracted Frame   : #{frame_idx} at {format_seconds(actual_sec)} ({w}x{h} px)")
    elif image_path:
        if not image_path.exists() and (ROOT_DIR / image_path).exists():
            image_path = ROOT_DIR / image_path
        print(f"Image Source      : {image_path}")
        full_frame = cv2.imread(str(image_path))
        if full_frame is None:
            raise FileNotFoundError(f"Could not load image file: {image_path}")
        h, w = full_frame.shape[:2]
        print(f"Image Resolution  : {w}x{h} px")
    else:
        raise ValueError("Either --video or --image must be provided.")

    frame_rgb = cv2.cvtColor(full_frame, cv2.COLOR_BGR2RGB)
    pil_full = Image.fromarray(frame_rgb)
    frame_h, frame_w = full_frame.shape[:2]

    # 3. Determine regions to evaluate
    regions_to_test = {}
    if already_cropped:
        regions_to_test["crop"] = {
            "label": "Direct Crop",
            "coords": (0, 0, frame_w, frame_h),
            "color": (0, 255, 0),
            "image": pil_full
        }
    else:
        active_keys = ["joker_bar", "consumeables"]
        if mode == "pack":
            active_keys.append("pack")
        else:
            active_keys.extend(["shop", "voucher", "booster"])

        for k in active_keys:
            rinfo = SHOP_ROIS[k]
            rx, ry, rw, rh = get_scaled_roi(rinfo["roi"], frame_w, frame_h)
            crop_img = pil_full.crop((rx, ry, rx + rw, ry + rh))
            regions_to_test[k] = {
                "label": rinfo["label"],
                "coords": (rx, ry, rw, rh),
                "color": rinfo["color"],
                "image": crop_img
            }

    # 4. Run Pipeline per Region
    region_results = {}
    total_found = 0

    print("-" * 70)
    for k, rdata in regions_to_test.items():
        rx, ry, rw, rh = rdata["coords"]
        crop_pil = rdata["image"]
        results = pipeline.predict(crop_pil, conf_threshold=conf_thresh)
        total_found += len(results)

        region_results[k] = {
            "label": rdata["label"],
            "coords": (rx, ry, rw, rh),
            "color": rdata["color"],
            "items": results
        }

        print(f"\n[{rdata['label'].upper()}] ({rw}x{rh}px) -> Found {len(results)} items:")
        if not results:
            print("   (None detected)")
        for idx, item in enumerate(results, start=1):
            key_conf = item.confidences.get("key", 1.0)
            det_conf = item.confidences.get("detection", 1.0)
            print(f"   {idx}. {item.display_name:<30} [Key: {item.key}] (Conf: Det={det_conf:.2f}, Key={key_conf:.2f})")

    # 5. Save Visual Diagnostic Image
    if output_path is None:
        if video_path:
            clean_name = video_path.stem
            ts_tag = timestamp_str.replace(":", "m") + "s"
            output_path = CLASSIFIER_DIR / "figures" / f"shop_test_{clean_name}_{ts_tag}.jpg"
        else:
            clean_name = image_path.stem
            output_path = CLASSIFIER_DIR / "figures" / f"shop_test_{clean_name}.jpg"

    save_diagnostic_image(full_frame, region_results, output_path)

    print("\n" + "=" * 70)
    print(f"Inference Complete! Total items recognized: {total_found}")
    print(f"Saved Annotated Diagnosis to: {output_path}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Test Shop Recognition Pipeline on Video Frame or Screenshot")
    parser.add_argument("--video", type=Path, default=None, help="Path to video file (e.g. .mp4)")
    parser.add_argument("--time", type=str, default="0", help="Timestamp in video (seconds or MM:SS or HH:MM:SS)")
    parser.add_argument("--image", type=Path, default=None, help="Path to screenshot image")
    parser.add_argument("--already-cropped", action="store_true", help="Set if image is already cropped to single shop shelf")
    parser.add_argument("--mode", type=str, default="shop", choices=["shop", "pack"], help="Shop screen mode (default: shop)")
    parser.add_argument("--conf", type=float, default=0.80, help="Detection confidence threshold")
    parser.add_argument("--detector-weights", type=Path, default=None, help="Custom detector .pt path")
    parser.add_argument("--classifier-weights", type=Path, default=None, help="Custom classifier .pt path")
    parser.add_argument("--output", type=Path, default=None, help="Output diagnosis image path")
    args = parser.parse_args()

    if not args.video and not args.image:
        # Default fallback to sample video if available
        sample_video = ROOT_DIR / "assets" / "sample_inputs" / "videos" / "bu_full.mp4"
        sample_shop = ROOT_DIR / "assets" / "sample_inputs" / "shop.jpg"
        if sample_shop.exists():
            print(f"[Notice] No input specified, defaulting to sample image: {sample_shop}")
            args.image = sample_shop
        elif sample_video.exists():
            print(f"[Notice] No input specified, defaulting to sample video: {sample_video} at 02:45")
            args.video = sample_video
            args.time = "02:45"
        else:
            parser.error("Please provide either --video or --image.")

    run_test(
        video_path=args.video,
        image_path=args.image,
        timestamp_str=args.time,
        already_cropped=args.already_cropped,
        detector_weights=args.detector_weights,
        classifier_weights=args.classifier_weights,
        output_path=args.output,
        conf_thresh=args.conf,
        mode=args.mode
    )


if __name__ == "__main__":
    main()
