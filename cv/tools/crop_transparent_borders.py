"""Crop fully transparent borders from template image files.

Examples:
    python tools/crop_transparent_borders.py
    python tools/crop_transparent_borders.py --in-place
    python tools/crop_transparent_borders.py --input-dir assets/templates --output-dir assets/templates
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".png", ".webp", ".tif", ".tiff"}


def alpha_bounds(image: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return the fully opaque bounding box as (x1, y1, x2, y2)."""
    if image.ndim != 3 or image.shape[2] != 4:
        return None

    alpha = image[:, :, 3]
    opaque_y, opaque_x = np.nonzero(alpha == 255)
    if len(opaque_x) == 0:
        return None

    return (
        int(opaque_x.min()),
        int(opaque_y.min()),
        int(opaque_x.max()) + 1,
        int(opaque_y.max()) + 1,
    )


def crop_transparent_border(image: np.ndarray) -> np.ndarray:
    """Zero partial alpha and crop to the fully opaque pixel bounds."""
    if image.ndim != 3 or image.shape[2] != 4:
        return image

    cleaned = image.copy()
    cleaned[:, :, 3][cleaned[:, :, 3] < 255] = 0
    bounds = alpha_bounds(cleaned)
    if bounds is None:
        return cleaned

    x1, y1, x2, y2 = bounds
    return cleaned[y1:y2, x1:x2]


def iter_images(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def process_image(input_path: Path, output_path: Path, *, dry_run: bool) -> bool:
    image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {input_path}")

    if image.ndim != 3 or image.shape[2] != 4:
        print(f"SKIP image without alpha channel: {input_path}")
        return False

    has_partial_alpha = bool(np.any(image[:, :, 3] < 255))
    cropped = crop_transparent_border(image)
    bounds = alpha_bounds(cropped)
    if bounds is None:
        print(f"SKIP image without fully opaque pixels: {input_path}")
        return False

    original_height, original_width = image.shape[:2]
    if not has_partial_alpha and cropped.shape[:2] == image.shape[:2]:
        print(f"SKIP already tight: {input_path}")
        return False

    print(
        f"CROP {input_path}: {original_width}x{original_height} -> "
        f"{cropped.shape[1]}x{cropped.shape[0]}"
    )
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), cropped):
            raise OSError(f"Could not write image: {output_path}")

    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop fully transparent borders from template images."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("assets/templates"),
        help="Directory to scan recursively.",
    )
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument(
        "--output-dir",
        type=Path,
        help="Write cropped files here, preserving the input directory layout.",
    )
    destination.add_argument(
        "--in-place",
        action="store_true",
        help="Replace source files with their cropped versions.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report changes without writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    image_paths = iter_images(input_dir)
    if not image_paths:
        print(f"No supported images found in {input_dir}")
        return

    if args.in_place:
        output_dir = input_dir
    else:
        output_dir = args.output_dir or input_dir.with_name(f"{input_dir.name}_cropped")

    changed = 0
    for input_path in image_paths:
        relative_path = input_path.relative_to(input_dir)
        output_path = output_dir / relative_path
        changed += process_image(
            input_path,
            output_path,
            dry_run=args.dry_run,
        )

    action = "would crop" if args.dry_run else "cropped"
    print(f"{action.capitalize()} {changed} of {len(image_paths)} image(s).")


if __name__ == "__main__":
    main()
