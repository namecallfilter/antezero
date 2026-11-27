"""
FULLY AI GENERATED
Interactive ROI selector for template matching.
Select templates by number, draw bounding boxes, and save normalized ROIs to JSON.
"""

import argparse
import json
import os

import cv2

DEFAULT_CONFIG_PATH = r"assets\templates\drspectred\templates.json"
DEFAULT_IMAGE_PATH = r"assets\sample_inputs\full_deck_shop.jpg"
EXPECTED_RES = (1920, 1080)  # (width, height)
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720


def load_templates(filepath: str) -> dict:
    """Loads templates config, normalizing flat string entries if needed."""
    if not os.path.exists(filepath):
        print(f"File '{filepath}' not found. Starting with empty config.")
        return {}

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    normalized = {}
    for key, val in data.items():
        if isinstance(val, str):
            normalized[key] = {
                "label": val,
                "filenames": [key] if key.lower().endswith(".png") else [],
                "rois": [],
            }
        elif isinstance(val, dict):
            normalized[key] = dict(val)
            normalized[key]["label"] = val.get("label", key)
            if "filenames" not in val:
                normalized[key]["filenames"] = (
                    [key] if key.lower().endswith(".png") else []
                )
            if "rois" not in val:
                normalized[key]["rois"] = []
        else:
            normalized[key] = {
                "label": key,
                "filenames": [key] if key.lower().endswith(".png") else [],
                "rois": [],
            }

    return normalized


def save_templates(filepath: str, data: dict):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved updates to '{filepath}'.")


def capture_single_roi(img, key_name: str) -> list[float] | None:
    """Opens a 16:9 window scaled to fit the display to draw an ROI."""
    h, w = img.shape[:2]
    window_title = f"Select ROI: {key_name} (SPACE/ENTER to confirm, 'c' to cancel)"

    # WINDOW_NORMAL allows fitting to screen; WINDOW_KEEPRATIO locks 16:9 aspect
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_title, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    # selectROI still records coordinates in original 1920x1080 pixel space
    rect = cv2.selectROI(window_title, img, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_title)

    x, y, box_w, box_h = rect

    # Cancelled or zero-area drag
    if box_w == 0 or box_h == 0:
        return None

    # Calculate normalized [x_min, y_min, x_max, y_max] rounded to 4 decimals
    return [
        round(x / w, 4),
        round(y / h, 4),
        round((x + box_w) / w, 4),
        round((y + box_h) / h, 4),
    ]


def main():
    parser = argparse.ArgumentParser(description="Tag ROIs for templates.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to templates JSON file",
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE_PATH,
        help="Path to reference screenshot",
    )
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Could not load screenshot at '{args.image}'")

    # Validate image resolution
    h, w = img.shape[:2]
    if (w, h) != EXPECTED_RES:
        raise ValueError(
            f"Image dimensions ({w}x{h}) do not match expected resolution "
            f"({EXPECTED_RES[0]}x{EXPECTED_RES[1]})."
        )

    templates = load_templates(args.config)
    if not templates:
        print("No templates found in configuration.")
        return

    while True:
        keys = list(templates.keys())
        print("\n" + "=" * 65)
        print("TEMPLATE SELECTOR")
        print("=" * 65)
        for idx, key in enumerate(keys, 1):
            entry = templates[key]
            label = entry.get("label", "")
            count = len(entry.get("rois", []))
            count_str = f"{count} ROI(s)" if count > 0 else "None"
            print(f"  [{idx:2d}] {key:<20} | {label:<22} | {count_str}")
        print("  [ q] Quit")

        user_input = input("\nSelect a number: ").strip()
        if user_input.lower() == "q":
            break

        if not user_input.isdigit() or not (1 <= int(user_input) <= len(keys)):
            print("Invalid selection. Enter a listed number.")
            continue

        selected_key = keys[int(user_input) - 1]
        entry = templates[selected_key]
        existing_rois = entry.get("rois", [])

        # Check existing ROIs and prompt for action
        action = "overwrite"
        if len(existing_rois) > 0:
            print(f"\n'{selected_key}' already has {len(existing_rois)} ROI(s):")
            for r_idx, r in enumerate(existing_rois, 1):
                print(f"    {r_idx}. {r}")

            while True:
                choice = (
                    input("\nAction: [O]verwrite, [A]ppend, or [S]witch template? ")
                    .strip()
                    .lower()
                )
                if choice in ("o", "overwrite"):
                    action = "overwrite"
                    break
                elif choice in ("a", "append"):
                    action = "append"
                    break
                elif choice in ("s", "switch"):
                    action = "switch"
                    break
                print("Please enter 'o', 'a', or 's'.")

            if action == "switch":
                continue

        # ROI collection loop
        new_rois = []
        while True:
            print("\nControls:")
            print("  • Drag with Left Mouse to select region")
            print("  • Press SPACE or ENTER to confirm box")
            print("  • Press 'c' to cancel")

            roi = capture_single_roi(img, selected_key)
            if roi is None:
                print("No box selected or selection canceled.")
                break

            new_rois.append(roi)
            print(f"Captured ROI: {roi}")

            more = input("Add another ROI for this template? (y/N): ").strip().lower()
            if more != "y":
                break

        if not new_rois:
            continue

        if action == "overwrite":
            entry["rois"] = new_rois
        elif action == "append":
            entry["rois"].extend(new_rois)

        save_templates(args.config, templates)


if __name__ == "__main__":
    main()
