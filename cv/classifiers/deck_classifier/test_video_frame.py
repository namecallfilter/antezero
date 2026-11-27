"""
Test View Deck Classifier on Video Frames and Real Screenshots
==============================================================
Extracts a frame from a video file at a specified timestamp (or loads a raw screenshot),
crops the View Deck region using `resolve_view_deck_roi()`, runs the two-stage deck
recognition pipeline, prints detected deck cards, and saves an annotated diagnosis image.
"""

import argparse
import sys
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
    CLASSIFIER_DIR = ROOT_DIR / "classifiers" / "deck_classifier"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classifiers.common import extract_video_frame, format_seconds, parse_timestamp, scale_roi_pixels
from classifiers.deck_classifier.config import BASE_HEIGHT, BASE_WIDTH, resolve_view_deck_roi
from classifiers.deck_classifier.pipeline import DeckRecognitionPipeline


def get_deck_roi_pixels(frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    """Calculates pixel View Deck ROI coordinates scaled to the frame's actual resolution."""
    roi = resolve_view_deck_roi()
    return scale_roi_pixels(roi, frame_w, frame_h, BASE_WIDTH, BASE_HEIGHT)


def draw_annotated_results(
    full_frame: np.ndarray,
    deck_crop_bgr: np.ndarray,
    roi_coords: tuple[int, int, int, int],
    results: list,
    save_path: Path,
):
    rx, ry, rw, rh = roi_coords
    crop_h, crop_w = deck_crop_bgr.shape[:2]

    crop_rgb = cv2.cvtColor(deck_crop_bgr, cv2.COLOR_BGR2RGB)
    pil_crop = Image.fromarray(crop_rgb)
    draw = ImageDraw.Draw(pil_crop)

    try:
        font_sm = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font_sm = ImageFont.load_default()

    color_ok = (34, 197, 94)
    for c in results:
        x1, y1, x2, y2 = c.box
        draw.rectangle([x1, y1, x2, y2], outline=color_ok, width=2)
        short_suit = c.suit[0] if c.suit != "None" else "?"
        short_rank = "St" if c.rank == "Stone" else ("10" if c.rank == "10" else c.rank[0])
        label = f"{short_rank}{short_suit}"
        draw.rectangle([x1, max(0, y1 - 14), x1 + 30, max(14, y1)], fill=(15, 23, 42))
        draw.text((x1 + 2, max(0, y1 - 13)), label, fill=(255, 255, 255), font=font_sm)

    full_annotated = full_frame.copy()
    cv2.rectangle(full_annotated, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 3)
    cv2.putText(
        full_annotated,
        f"VIEW DECK ROI ({rw}x{rh})",
        (rx + 8, max(24, ry - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    crop_bgr_annotated = cv2.cvtColor(np.array(pil_crop), cv2.COLOR_RGB2BGR)
    target_width = 1280
    scale_crop = target_width / max(1, crop_w)
    crop_resized = cv2.resize(crop_bgr_annotated, (target_width, max(1, int(round(crop_h * scale_crop)))))

    scale_full = target_width / max(1, full_frame.shape[1])
    full_resized = cv2.resize(full_annotated, (target_width, max(1, int(round(full_frame.shape[0] * scale_full)))))

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
    conf_thresh: float = 0.45,
    iou_thresh: float = 0.55,
):
    print("\n" + "=" * 70)
    print("BALATRO VIEW DECK RECOGNITION TEST")
    print("=" * 70)

    pipeline = DeckRecognitionPipeline(
        detector_path=detector_weights,
        classifier_path=classifier_weights,
    )

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

    if already_cropped:
        deck_crop_bgr = full_frame
        roi_coords = (0, 0, fw, fh)
        print("Crop Mode         : Input is ALREADY cropped to View Deck region.")
    else:
        rx, ry, rw, rh = get_deck_roi_pixels(fw, fh)
        roi_coords = (rx, ry, rw, rh)
        deck_crop_bgr = full_frame[ry : ry + rh, rx : rx + rw].copy()
        print(f"View Deck ROI     : x={rx}, y={ry}, w={rw}, h={rh}")

    crop_rgb = cv2.cvtColor(deck_crop_bgr, cv2.COLOR_BGR2RGB)
    for _ in range(3):
        _ = pipeline.predict(crop_rgb, conf=conf_thresh, iou=iou_thresh)
    results, timings = pipeline.predict(crop_rgb, conf=conf_thresh, iou=iou_thresh, return_timing=True)

    print("\n" + "=" * 70)
    print(f"DECK RECOGNITION RESULTS ({len(results)} cards detected)")
    print("=" * 70)
    for c in results:
        print(f"Card {c.index:3d}: {c.display_name} | box={c.box}")

    print("\n" + "=" * 70)
    print(f"Stage 1 (Deck YOLO)  : {timings['stage1_detector_ms']:6.2f} ms")
    print(f"Stage 2 (Classifier) : {timings['stage2_classifier_ms']:6.2f} ms ({len(results)} cards)")
    print(f"Total Inference      : {timings['total_inference_ms']:6.2f} ms")
    print("=" * 70)

    if output_path is None:
        output_path = ROOT_DIR / "output" / "test_deck_annotated.jpg"
    draw_annotated_results(full_frame, deck_crop_bgr, roi_coords, results, output_path)
    print(f"\n[OK] Annotated diagnostic image saved to: {output_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Balatro View Deck Classifier on Video Frames or Screenshots")
    parser.add_argument("--video", "-v", type=str, default=None, help="Path to video file (.mp4, etc.)")
    parser.add_argument("--image", "-i", type=str, default=None, help="Path to image file (.png, .jpg)")
    parser.add_argument("--time", "-t", type=str, default="0", help="Timestamp in video ('04:00', '1:20:15', or '240')")
    parser.add_argument("--already-cropped", action="store_true", help="Set if input image is already cropped to View Deck ROI")
    parser.add_argument("--conf", type=float, default=0.45, help="Detector confidence threshold (default: 0.45)")
    parser.add_argument("--iou", type=float, default=0.55, help="Detector NMS IoU threshold (default: 0.55)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output annotated image path")
    parser.add_argument("--detector", type=str, default=None, help="Custom deck detector weights path")
    parser.add_argument("--classifier", type=str, default=None, help="Custom Stage 2 classifier weights path")
    args = parser.parse_args()

    default_vid = ROOT_DIR / "assets" / "sample_inputs" / "videos" / "bu_full.mp4"
    if not args.video and not args.image and default_vid.exists():
        print(f"No input specified. Defaulting to sample video: {default_vid}")
        args.video = str(default_vid)

    run_test(
        video_path=Path(args.video) if args.video else None,
        image_path=Path(args.image) if args.image else None,
        timestamp_str=args.time,
        already_cropped=args.already_cropped,
        detector_weights=Path(args.detector) if args.detector else None,
        classifier_weights=Path(args.classifier) if args.classifier else None,
        output_path=Path(args.output) if args.output else None,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
    )
