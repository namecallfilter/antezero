"""
Test Card Classifier on Video Frames and Real Screenshots
==========================================================
Extracts a frame from a video file at a specified timestamp (or loads a raw screenshot),
crops the hand region using the mod's exact ROI coordinates, runs the two-stage
card recognition pipeline, prints predicted card attributes, and saves an
annotated visual diagnosis image.

Usage:
    # Test a video at a specific timestamp (seconds or MM:SS or HH:MM:SS):
    python test_video_frame.py --video assets/sample_inputs/videos/bu_full.mp4 --time 04:00
    python test_video_frame.py --video assets/sample_inputs/videos/bu_full.mp4 --time 240.5

    # Test an existing screenshot:
    python test_video_frame.py --image path/to/screenshot.png

    # If the input image is already cropped to the hand region:
    python test_video_frame.py --image path/to/hand_crop.png --already-cropped
"""

import sys
import argparse
import re
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Resolve project root and card classifier directory
SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR / "pipeline.py").exists():
    CLASSIFIER_DIR = SCRIPT_DIR
    ROOT_DIR = SCRIPT_DIR.parent.parent
else:
    ROOT_DIR = SCRIPT_DIR
    CLASSIFIER_DIR = ROOT_DIR / "classifiers" / "card_classifier"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
# pyrefly: ignore [missing-import]
from classifiers.card_classifier.pipeline import CardRecognitionPipeline

# Hand ROI in 1920x1080 Balatro space (from balatro_mod/dataset/boxes.lua)
BASE_WIDTH = 1920.0
BASE_HEIGHT = 1080.0
MOD_ROI = {
    "x": 508.319934,
    "y": 567.839093,
    "w": 1071.056521,
    "h": 322.734447,
}


# pyrefly: ignore [missing-import]
from classifiers.common import parse_timestamp, format_seconds, extract_video_frame, scale_roi_pixels


def get_hand_roi_pixels(frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    """Calculates pixel ROI coordinates scaled to the frame's actual resolution."""
    return scale_roi_pixels(MOD_ROI, frame_w, frame_h, BASE_WIDTH, BASE_HEIGHT)


def draw_annotated_results(
    full_frame: np.ndarray,
    hand_crop_bgr: np.ndarray,
    roi_coords: tuple[int, int, int, int],
    results: list,
    save_path: Path
):
    """
    Renders an informative side-by-side diagnosis image:
    - Top: Hand crop with precise detected card bounding boxes and identity badges.
    - Bottom: Full video frame showing the exact ROI cropping window.
    """
    rx, ry, rw, rh = roi_coords
    crop_h, crop_w = hand_crop_bgr.shape[:2]

    # Convert crop to PIL for clean antialiased text rendering
    crop_rgb = cv2.cvtColor(hand_crop_bgr, cv2.COLOR_BGR2RGB)
    pil_crop = Image.fromarray(crop_rgb)
    draw = ImageDraw.Draw(pil_crop)

    try:
        font = ImageFont.truetype("arial.ttf", 15)
        font_sm = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
        font_sm = ImageFont.load_default()

    # Palette
    COLOR_UNSELECTED = (34, 197, 94)    # Modern vibrant Green
    COLOR_SELECTED   = (249, 115, 22)   # Vibrant Orange
    COLOR_BOX_BG     = (15, 23, 42, 220)

    for c in results:
        x1, y1, x2, y2 = c.box
        color = COLOR_SELECTED if c.selected else COLOR_UNSELECTED

        # Bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Label badge
        if c.face_down:
            title = f"{c.index}. Face Down"
        elif c.rank == "Stone":
            title = f"{c.index}. Stone Card"
        else:
            title = f"{c.index}. {c.rank} of {c.suit}"

        extras = []
        if c.face_down:
            extras.append("[FACE DOWN]")
        elif c.debuffed:
            extras.append("[DEBUFFED]")
        if c.edition != "None": extras.append(c.edition)
        if c.enhancement != "None": extras.append(c.enhancement)
        if c.seal != "None": extras.append(f"{c.seal} Seal")
        if c.selected: extras.append("[SELECTED]")

        extra_str = " | ".join(extras) if extras else f"conf: {c.confidences['rank']:.2f}"

        # Draw text tag above card
        tag_y = max(4, y1 - 36 if y1 >= 38 else y2 + 4)
        
        # Background pill
        draw.rectangle([x1, tag_y, x1 + 130, tag_y + 32], fill=(20, 20, 20))
        draw.rectangle([x1, tag_y, x1 + 130, tag_y + 32], outline=color, width=1)
        draw.text((x1 + 4, tag_y + 2), title, fill=(255, 255, 255), font=font)
        draw.text((x1 + 4, tag_y + 17), extra_str, fill=color, font=font_sm)

    # Annotated full frame showing the ROI
    full_annotated = full_frame.copy()
    cv2.rectangle(full_annotated, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 3)
    cv2.putText(
        full_annotated,
        f"HAND CROP ROI ({rw}x{rh})",
        (rx + 8, max(24, ry - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Stack cropped hand view above resized full frame overview
    crop_bgr_annotated = cv2.cvtColor(np.array(pil_crop), cv2.COLOR_RGB2BGR)
    target_width = 1280
    
    # Scale crop to target width
    scale_crop = target_width / crop_w
    crop_resized = cv2.resize(crop_bgr_annotated, (target_width, int(round(crop_h * scale_crop))))

    # Scale full frame overview to target width
    scale_full = target_width / full_frame.shape[1]
    full_resized = cv2.resize(full_annotated, (target_width, int(round(full_frame.shape[0] * scale_full))))

    # Separator bar
    separator = np.zeros((20, target_width, 3), dtype=np.uint8)
    separator[:] = (40, 40, 40)

    combined = np.vstack([crop_resized, separator, full_resized])

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
    conf_thresh: float = 0.50
):
    print("\n" + "=" * 70)
    print("BALATRO VIDEO & SCREENSHOT CARD RECOGNITION TEST")
    print("=" * 70)

    # 1. Initialize Pipeline
    pipeline = CardRecognitionPipeline(
        detector_path=detector_weights,
        classifier_path=classifier_weights
    )

    # 2. Extract or load image (resolve relative to CWD or ROOT_DIR)
    full_frame = None
    if video_path:
        if not video_path.exists() and (ROOT_DIR / video_path).exists():
            video_path = ROOT_DIR / video_path
        ts_sec = parse_timestamp(timestamp_str)
        print(f"Video File        : {video_path}")
        print(f"Requested Time    : {format_seconds(ts_sec)} ({ts_sec:.2f}s)")
        full_frame, actual_sec, frame_idx = extract_video_frame(video_path, ts_sec)
        print(f"Actual Extracted  : {format_seconds(actual_sec)} (Frame #{frame_idx})")
    elif image_path:
        if not image_path.exists() and (ROOT_DIR / image_path).exists():
            image_path = ROOT_DIR / image_path
        print(f"Image File        : {image_path}")
        full_frame = cv2.imread(str(image_path))
        if full_frame is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
    else:
        raise ValueError("Must provide either --video or --image!")

    fh, fw = full_frame.shape[:2]
    print(f"Input Resolution  : {fw}x{fh}")

    # 3. Crop Hand Region
    if already_cropped:
        hand_crop_bgr = full_frame
        roi_coords = (0, 0, fw, fh)
        print("Crop Mode         : Input is ALREADY cropped to hand region.")
    else:
        rx, ry, rw, rh = get_hand_roi_pixels(fw, fh)
        roi_coords = (rx, ry, rw, rh)
        hand_crop_bgr = full_frame[ry : ry + rh, rx : rx + rw].copy()
        print(f"Hand ROI Crop     : x={rx}, y={ry}, w={rw}, h={rh} (from mod boxes.lua)")

    # 4. Run Pipeline Prediction with Precise GPU Timing
    crop_rgb = cv2.cvtColor(hand_crop_bgr, cv2.COLOR_BGR2RGB)

    # Warm-up passes to eliminate PyTorch / CuDNN cold-start JIT overhead
    for _ in range(3):
        _ = pipeline.predict(crop_rgb, conf=conf_thresh)

    # Steady-state benchmarked run
    results, timings = pipeline.predict(crop_rgb, conf=conf_thresh, return_timing=True)

    # 5. Print Detailed Output
    print("\n" + "=" * 70)
    print(f"RECOGNITION RESULTS ({len(results)} cards detected left-to-right)")
    print("=" * 70)

    if not results:
        print("[No cards detected in this frame. Check if the hand is currently dealt / visible.]")
    else:
        for c in results:
            sel_tag = " [SELECTED]" if c.selected else ""
            print(f"Card {c.index:2d}: {c.display_name}{sel_tag}")
            print(f"   Rank       : {c.rank:<10} (conf: {c.confidences['rank']:.3f})")
            print(f"   Suit       : {c.suit:<10} (conf: {c.confidences['suit']:.3f})")
            print(f"   Edition    : {c.edition:<10} (conf: {c.confidences['edition']:.3f})")
            print(f"   Enhancement: {c.enhancement:<10} (conf: {c.confidences['enhancement']:.3f})")
            print(f"   Seal       : {c.seal:<10} (conf: {c.confidences['seal']:.3f})")
            print(f"   Debuffed   : {str(c.debuffed):<10} (conf: {c.confidences.get('debuffed', 1.0):.3f})")
            print(f"   Face Down  : {str(c.face_down):<10} (conf: {c.confidences.get('face_down', 1.0):.3f})")
            print(f"   Box [x,y]  : [{c.box[0]}, {c.box[1]}, {c.box[2]}, {c.box[3]}] | Selected: {c.selected}")
            print("-" * 55)

    # GPU Inference Timing Benchmark
    s1_ms = timings["stage1_detector_ms"]
    s2_ms = timings["stage2_classifier_ms"]
    tot_ms = timings["total_inference_ms"]
    fps_equiv = (1000.0 / tot_ms) if tot_ms > 0 else 0.0

    print("\n" + "=" * 70)
    print("GPU INFERENCE BENCHMARK (Pure Model Forward Time)")
    print("=" * 70)
    print(f"Stage 1: YOLO Detector           : {s1_ms:6.2f} ms")
    print(f"Stage 2: Multi-Head Classifier   : {s2_ms:6.2f} ms  ({len(results)} cards batched)")
    print(f"----------------------------------------------------------------------")
    print(f"Total Model GPU Inference        : {tot_ms:6.2f} ms  (~{fps_equiv:,.0f} hands/sec)")
    print("=" * 70)

    # 6. Save Annotated Debug Image
    if output_path is None:
        output_dir = ROOT_DIR / "output"
        output_path = output_dir / "test_frame_annotated.jpg"

    draw_annotated_results(full_frame, hand_crop_bgr, roi_coords, results, output_path)
    print(f"\n[OK] Annotated diagnostic image saved to: {output_path.resolve()}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Balatro Card Classifier on Video Frames or Screenshots")
    parser.add_argument("--video", "-v", type=str, default=None, help="Path to video file (.mp4, etc.)")
    parser.add_argument("--image", "-i", type=str, default=None, help="Path to image file (.png, .jpg)")
    parser.add_argument("--time", "-t", type=str, default="0", help="Timestamp in video ('04:00', '1:20:15', or '240')")
    parser.add_argument("--already-cropped", action="store_true", help="Set this if the image is already cropped to the hand")
    parser.add_argument("--conf", type=float, default=0.50, help="Detector confidence threshold (default: 0.50)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output image file path (default: output/test_frame_annotated.jpg)")
    parser.add_argument("--detector", type=str, default=None, help="Custom detector weights path")
    parser.add_argument("--classifier", type=str, default=None, help="Custom classifier weights path")

    args = parser.parse_args()

    # Default to sample video if neither video nor image is specified
    default_vid = ROOT_DIR / "assets" / "sample_inputs" / "videos" / "bu_full.mp4"
    if not args.video and not args.image and default_vid.exists():
        print(f"No input specified. Defaulting to sample video: {default_vid}")
        args.video = str(default_vid)
        if args.time == "0":
            args.time = "04:00"  # default to a known hand timestamp

    run_test(
        video_path=Path(args.video) if args.video else None,
        image_path=Path(args.image) if args.image else None,
        timestamp_str=args.time,
        already_cropped=args.already_cropped,
        detector_weights=Path(args.detector) if args.detector else None,
        classifier_weights=Path(args.classifier) if args.classifier else None,
        output_path=Path(args.output) if args.output else None,
        conf_thresh=args.conf
    )
