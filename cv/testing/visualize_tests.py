import sys
from pathlib import Path
import cv2
import json
from itertools import product
import numpy as np

# Add project root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import Config
from template import template_loader
from matcher import find_element
from shadow_evaluator import analyze_shadow_presence
from geometry import ImageSize, PixelROI, crop_image_to_roi
from tools.debug_plotter import plot

# Set to True to only visualize/show failed tests. Set to False to view all tests.
SHOW_ONLY_FAILURES = True

def run_visual_tests(show_only_failures: bool = SHOW_ONLY_FAILURES):
    config = Config()
    templates = template_loader(config.template_json_path, config.template_img_dir, config.resolution)
    template_dict = {t.name: t for t in templates}
    
    with open(config.template_json_path, "r", encoding="utf-8") as f:
        templates_data = json.load(f)
        
    for template_name, data in templates_data.items():
        test_frames = data.get("test_frames", [])
        if not test_frames:
            continue
            
        template = template_dict.get(template_name)
        if not template:
            continue
            
        for tf in test_frames:
            frame_idx = tf.get("frame_idx", "unknown")
            expected_pressed = tf.get("pressed")
            
            # Construct path to annotated frame
            frame_path = Path(tf.get("path", ""))
            
            if not frame_path.exists():
                print(f"Missing frame: {frame_path}")
                continue
                
            img = cv2.imread(str(frame_path))
            if img is None:
                continue

            best_match = None
            best_confidence = -1.0
            best_top_left = None
            best_roi_bounds = None
            best_image = None
            
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # We bypass eval_template.eval here so we can keep the highest match 
            # even if it fails the threshold check (which eval_template throws away)
            for (image, image_gray, mask), px_roi in product(
                zip(template.images, template.images_gray, template.masks), template.px_rois
            ):
                h_inp, w_inp = image.shape[:2]
                img_h, img_w = img.shape[:2]
                expanded_x1, expanded_y1, expanded_x2, expanded_y2 = px_roi.expanded_bounds(config.expand_roi_factor)
                
                y1, y2 = max(0, expanded_y1), min(img_h, expanded_y2)
                x1, x2 = max(0, expanded_x1), min(img_w, expanded_x2)
                
                if (y2 - y1) < h_inp:
                    diff = h_inp - (y2 - y1)
                    y1 = max(0, y1 - diff // 2)
                    y2 = y1 + h_inp
                    if y2 > img_h:
                        y2 = img_h
                        y1 = max(0, y2 - h_inp)
                if (x2 - x1) < w_inp:
                    diff = w_inp - (x2 - x1)
                    x1 = max(0, x1 - diff // 2)
                    x2 = x1 + w_inp
                    if x2 > img_w:
                        x2 = img_w
                        x1 = max(0, x2 - w_inp)

                if (y2 - y1) < h_inp or (x2 - x1) < w_inp:
                    continue
                    
                roi_img_gray = img_gray[y1:y2, x1:x2]
                roi_img = img[y1:y2, x1:x2]
                match = find_element(
                    roi_img_gray,
                    image_gray,
                    mask,
                    roi_img,
                    image[:, :, :3],
                    ImageSize(width=w_inp, height=h_inp),
                    config.matching_method,
                    px_roi,
                    config.elem_conf_thresh,
                )
                abs_top_left = (match.top_left[0] + x1, match.top_left[1] + y1)
                
                if match.confidence > best_confidence:
                    best_confidence = match.confidence
                    best_match = match
                    best_top_left = abs_top_left
                    best_roi_bounds = (x1, y1, x2, y2)
                    best_image = image

            if best_match is None:
                print(f"Skipped {template_name} in frame {frame_idx} (ROI too small)")
                continue

            found_target = best_confidence > config.elem_conf_thresh
            
            # We ALWAYS compute the shadow properties here for visualization,
            # even if the confidence was too low and we technically "failed".
            h_inp, w_inp = best_image.shape[:2]
            dy = template.shadow_dy
            dw = template.shadow_dw
            dh = template.shadow_dh
            height = config.shadow_region_height
            bottom_left = (best_top_left[0], best_top_left[1] + h_inp)
            bottom_right = (best_top_left[0] + w_inp, best_top_left[1] + h_inp)
            
            shadow_region = PixelROI(
                x_min=bottom_left[0],
                y_min=bottom_left[1] + dy,
                x_max=bottom_right[0] + dw,
                y_max=bottom_right[1] + height + dy + dh,
            )
            shadow_img = crop_image_to_roi(img, shadow_region)
            has_shadow = analyze_shadow_presence(shadow_img, threshold=config.shadow_threshold)
            clicked = not has_shadow

            # Build detailed label
            if expected_pressed is False:
                if found_target and clicked:
                    status_text = "FAIL (False Positive. Expected Unpressed, but found as pressed)"
                elif not found_target:
                    status_text = "PASS (Not Found; fine for expected=False)"
                else:
                    status_text = "PASS (Found; correctly identified as Unpressed)"
            else:
                if not found_target:
                    status_text = f"FAIL (Not Found. {best_confidence:.4f} < {config.elem_conf_thresh})"
                elif expected_pressed is not None and clicked != expected_pressed:
                    status_text = f"FAIL (Shadow. Actual: {clicked}, Expected: {expected_pressed})"
                else:
                    status_text = f"PASS"
            
            is_failure = status_text.startswith("FAIL")
            if show_only_failures and not is_failure:
                continue

            label_text = f"Expected Pressed: {expected_pressed} | {status_text}"
            print(f"[{template_name}] Frame {frame_idx}: {status_text}", flush=True)
            
            # Draw using the existing debug_plotter. 
            if "--no-plot" not in sys.argv:
                plot(
                    img=img,
                    template=best_image,
                    roi=best_roi_bounds,
                    found_target=True,
                    match=best_match.score_map,
                    top_left=best_top_left,
                    max_val=best_confidence,
                    clicked=clicked,
                    label=label_text,
                    shadow_region=shadow_region,
                    shadow_threshold=config.shadow_threshold,
                    template_name=template.name,
                    template_json_path=config.template_json_path,
                    current_shadow_dy=template.shadow_dy,
                    current_shadow_dw=template.shadow_dw,
                    current_shadow_dh=template.shadow_dh,
                    shadow_height=config.shadow_region_height,
                    video_timestamp_str=f"Frame: {frame_idx}",
                )

if __name__ == "__main__":
    run_visual_tests()
