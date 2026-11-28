"""Matplotlib failure visualizer for real-image CV fixture tests.

Draws YOLO bounding boxes, ROI bounds, classifier predictions, and per-attribute
confidence scores for failed cases in `testing/test_cv_real_images.py`.

Can also be run directly from the CLI to inspect any fixture image:
    py testing/visualize_cv_failures.py --section joker_bar --image images/blind_1.png
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classifiers import inference as cls_inf


FIXTURES_DIR = Path(__file__).resolve().parent / "real_image_fixtures"
FAILURE_OUT_DIR = FIXTURES_DIR / "failures"

# Color palette (RGB 0-1 for matplotlib)
COLOR_OK = (0.13, 0.77, 0.37)       # Green
COLOR_WARN = (0.98, 0.45, 0.09)     # Orange (substitution / attr mismatch)
COLOR_FP = (0.94, 0.27, 0.27)       # Red (false positive)
COLOR_ROI = (0.02, 0.71, 0.83)      # Cyan (ROI boundary)


def _sanitize_filename(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_")


def _format_item_label(idx: int, item: Any) -> str:
    """Builds a compact multi-line label with classifier outputs and confidences."""
    confs: dict[str, float] = getattr(item, "confidences", None) or {}

    if hasattr(item, "rank") and hasattr(item, "suit"):
        # CardResult
        if getattr(item, "face_down", False):
            title = f"[{idx}] Face Down"
        else:
            title = f"[{idx}] {item.rank} of {item.suit}"
        conf_parts = []
        if "detection" in confs:
            conf_parts.append(f"yolo:{confs['detection']:.2f}")
        if "rank" in confs:
            conf_parts.append(f"rank:{confs['rank']:.2f}")
        if "suit" in confs:
            conf_parts.append(f"suit:{confs['suit']:.2f}")

        extras = []
        for attr in ("edition", "enhancement", "seal"):
            val = getattr(item, attr, "None")
            if val and str(val).lower() != "none":
                c = confs.get(attr)
                extras.append(f"{val}({c:.2f})" if c is not None else str(val))
        if getattr(item, "selected", False):
            extras.append("selected")
        if getattr(item, "debuffed", False):
            c = confs.get("debuffed")
            extras.append(f"debuffed({c:.2f})" if c is not None else "debuffed")
    else:
        # ShopItemResult
        key = getattr(item, "key", None) or getattr(item, "display_name", "Unknown")
        card_type = getattr(item, "card_type", "")
        title = f"[{idx}] {key}" + (f" ({card_type})" if card_type and card_type != "Joker" else "")

        conf_parts = []
        if "detection" in confs:
            conf_parts.append(f"yolo:{confs['detection']:.2f}")
        if "key" in confs:
            conf_parts.append(f"cls:{confs['key']:.2f}")

        extras = []
        edition = getattr(item, "edition", "None")
        if edition and str(edition).lower() != "none":
            c = confs.get("edition")
            extras.append(f"{edition}({c:.2f})" if c is not None else str(edition))
        for flag in ("eternal", "perishable", "rental", "debuffed"):
            if getattr(item, flag, False):
                c = confs.get(flag)
                extras.append(f"{flag}({c:.2f})" if c is not None else flag)

    lines = [title]
    if conf_parts:
        lines.append(" ".join(conf_parts))
    if extras:
        lines.append(" | ".join(extras))
    return "\n".join(lines)


def _compute_view_bounds(
    frame_shape: tuple[int, int, int],
    rois: list[tuple[str, tuple[int, int, int, int]]],
    items: list[Any],
    pad: int = 40,
) -> tuple[int, int, int, int]:
    """Computes a tight crop (x1, y1, x2, y2) covering all active ROIs and detected boxes."""
    h, w = frame_shape[:2]
    xs1, ys1, xs2, ys2 = [], [], [], []
    for _, (rx, ry, rw, rh) in rois:
        if rw > 0 and rh > 0:
            xs1.append(rx)
            ys1.append(ry)
            xs2.append(rx + rw)
            ys2.append(ry + rh)
    for item in items:
        box = getattr(item, "box", None)
        if box and len(box) == 4:
            xs1.append(int(box[0]))
            ys1.append(int(box[1]))
            xs2.append(int(box[2]))
            ys2.append(int(box[3]))

    if not xs1:
        return 0, 0, w, h

    return (
        max(0, min(xs1) - pad),
        max(0, min(ys1) - pad),
        min(w, max(xs2) + pad),
        min(h, max(ys2) + pad),
    )


def _get_raw_predictions(
    section: str,
    frame_bgr: np.ndarray,
    conf_threshold: float,
    shelf_key: str | None = None,
) -> tuple[list[Any], list[tuple[str, tuple[int, int, int, int]]]]:
    """Runs the raw classifier pipeline so YOLO boxes and confidences are available."""
    fh, fw = frame_bgr.shape[:2]

    def scaled_roi(name: str) -> tuple[str, tuple[int, int, int, int]]:
        roi = cls_inf.ROIS[name]
        rx = int(round(roi["x"] * (fw / cls_inf.BASE_WIDTH)))
        ry = int(round(roi["y"] * (fh / cls_inf.BASE_HEIGHT)))
        rw = int(round(roi["w"] * (fw / cls_inf.BASE_WIDTH)))
        rh = int(round(roi["h"] * (fh / cls_inf.BASE_HEIGHT)))
        return name, (rx, ry, rw, rh)

    if section == "hand":
        out = cls_inf.recognize_hand_cards(frame_bgr, conf_threshold=conf_threshold)
        return list(out.cards), [("card_hand", out.roi)]

    if section == "joker_bar":
        out = cls_inf.recognize_shop(frame_bgr, mode="joker_only", conf_threshold=conf_threshold)
        return list(out.joker_bar), [scaled_roi("joker_bar")]

    if section == "pack":
        out = cls_inf.recognize_shop(frame_bgr, mode="pack", conf_threshold=conf_threshold)
        return list(out.pack_items), [scaled_roi("pack")]

    if section == "shop":
        out = cls_inf.recognize_shop(frame_bgr, mode="shop", conf_threshold=conf_threshold)
        shelf_map = {
            "shop_cards": (out.shop_items, "shop"),
            "vouchers": (out.vouchers, "voucher"),
            "packs": (out.boosters, "booster"),
            "joker_bar": (out.joker_bar, "joker_bar"),
            "consumables": (out.consumeables, "consumeables"),
        }
        if shelf_key and shelf_key in shelf_map:
            items, roi_name = shelf_map[shelf_key]
            return list(items), [scaled_roi(roi_name)]
        rois = [scaled_roi(r) for r in ("joker_bar", "consumeables", "shop", "voucher", "booster")]
        return list(out.all_items), rois

    raise ValueError(f"Unsupported section for visualization: {section}")


def visualize_cv_failure(
    section: str,
    rel_path: str,
    frame_bgr: np.ndarray,
    conf_threshold: float,
    failure_summary: str,
    *,
    shelf_key: str | None = None,
    fp_indices: set[int] | None = None,
    warn_indices: set[int] | None = None,
    save_path: Path | None = None,
    show: bool | None = None,
) -> Path:
    """Renders a matplotlib figure of YOLO boxes, classifier labels, and confidences."""
    if show is None:
        show = os.environ.get("CV_TEST_SHOW", "1") != "0"

    raw_items, rois = _get_raw_predictions(
        section, frame_bgr, conf_threshold, shelf_key=shelf_key
    )
    fp_indices = fp_indices or set()
    warn_indices = warn_indices or set()

    vx1, vy1, vx2, vy2 = _compute_view_bounds(frame_bgr.shape, rois, raw_items, pad=60)
    crop_rgb = cv2.cvtColor(frame_bgr[vy1:vy2, vx1:vx2], cv2.COLOR_BGR2RGB)

    fig, (ax_img, ax_txt) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        gridspec_kw={"height_ratios": [3.2, 1.3]},
    )
    fig.patch.set_facecolor("#0f172a")

    ax_img.imshow(crop_rgb)
    ax_img.set_axis_off()
    title_suffix = f":{shelf_key}" if shelf_key else ""
    ax_img.set_title(
        f"CV Failure Diagnosis — {section}{title_suffix} | {rel_path} (conf >= {conf_threshold:.2f})",
        color="white",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )

    # Draw ROI outlines
    for roi_name, (rx, ry, rw, rh) in rois:
        if rw <= 0 or rh <= 0:
            continue
        rect = patches.Rectangle(
            (rx - vx1, ry - vy1),
            rw,
            rh,
            linewidth=1.5,
            linestyle="--",
            edgecolor=COLOR_ROI,
            facecolor="none",
        )
        ax_img.add_patch(rect)
        ax_img.text(
            rx - vx1 + 6,
            max(12, ry - vy1 + 16),
            f"ROI: {roi_name}",
            color=COLOR_ROI,
            fontsize=8,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f172a", edgecolor="none", alpha=0.75),
        )

    # Draw YOLO bounding boxes and classifier outputs + confidences
    for idx, item in enumerate(raw_items):
        box = getattr(item, "box", None)
        if not box or len(box) != 4:
            continue
        bx1, by1, bx2, by2 = [int(v) - off for v, off in zip(box, (vx1, vy1, vx1, vy1))]
        if idx in fp_indices:
            color = COLOR_FP
        elif idx in warn_indices:
            color = COLOR_WARN
        else:
            color = COLOR_OK

        rect = patches.Rectangle(
            (bx1, by1),
            max(1, bx2 - bx1),
            max(1, by2 - by1),
            linewidth=2.2,
            edgecolor=color,
            facecolor="none",
        )
        ax_img.add_patch(rect)

        label = _format_item_label(idx, item)
        # Alternate label placement slightly below/above box to reduce overlap
        text_y = by2 + 6 if by2 + 55 < (vy2 - vy1) else max(4, by1 - 8)
        va = "top" if text_y > by1 else "bottom"
        ax_img.text(
            bx1,
            text_y,
            label,
            color="white",
            fontsize=8,
            family="monospace",
            va=va,
            ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e293b", edgecolor=color, linewidth=1.2, alpha=0.92),
        )

    # Bottom panel: failure summary
    ax_txt.set_facecolor("#0f172a")
    ax_txt.set_axis_off()
    ax_txt.text(
        0.01,
        0.95,
        failure_summary.strip(),
        color="#e2e8f0",
        fontsize=9.5,
        family="monospace",
        va="top",
        ha="left",
        transform=ax_txt.transAxes,
    )

    plt.tight_layout()

    if save_path is None:
        FAILURE_OUT_DIR.mkdir(parents=True, exist_ok=True)
        fname = _sanitize_filename(f"{section}{title_suffix}_{Path(rel_path).stem}.png")
        save_path = FAILURE_OUT_DIR / fname
    else:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    if show:
        plt.show()
    plt.close(fig)
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize YOLO + classifier outputs for a fixture image.")
    parser.add_argument("--section", required=True, choices=["hand", "shop", "pack", "joker_bar"])
    parser.add_argument("--image", required=True, help="Relative path under real_image_fixtures/ (e.g. images/blind_1.png)")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold override")
    parser.add_argument("--no-show", action="store_true", help="Save PNG without opening interactive window")
    args = parser.parse_args()

    img_path = FIXTURES_DIR / args.image
    frame = cv2.imread(str(img_path))
    if frame is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")

    conf = args.conf if args.conf is not None else (0.50 if args.section == "hand" else 0.80)
    out_path = visualize_cv_failure(
        args.section,
        args.image,
        frame,
        conf,
        f"Manual inspection: {args.section} on {args.image}",
        show=not args.no_show,
    )
    print(f"Saved visualization to: {out_path}")


if __name__ == "__main__":
    main()
