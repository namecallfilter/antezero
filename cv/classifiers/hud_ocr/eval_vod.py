"""Evaluate trained HUD OCR weights on real gameplay VOD frames and render a contact sheet."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from classifiers.common.video_utils import (
    extract_video_frames,
    format_seconds,
    parse_timestamp,
    scale_roi_pixels,
)

try:
    from .config import CURRENT_DIR, HUD_ROIS, HUD_ROI_FIELDS
    from .pipeline import HudOcrPipeline, HudOcrResult
except ImportError:
    from config import CURRENT_DIR, HUD_ROIS, HUD_ROI_FIELDS  # type: ignore[no-redef]
    from pipeline import HudOcrPipeline, HudOcrResult  # type: ignore[no-redef]


def _pick_frame_indices(
    video_path: Path,
    num_frames: int,
    explicit_frames: Sequence[int] | None = None,
    explicit_timestamps: Sequence[str] | None = None,
) -> tuple[list[int], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    if total_frames <= 0:
        raise RuntimeError(f"Video has no readable frames: {video_path}")

    if explicit_frames:
        indices = sorted({max(0, min(total_frames - 1, int(f))) for f in explicit_frames})
        return indices, fps

    if explicit_timestamps:
        indices = sorted({
            max(0, min(total_frames - 1, int(round(parse_timestamp(ts) * fps))))
            for ts in explicit_timestamps
        })
        return indices, fps

    n = max(1, min(int(num_frames), total_frames))
    lo = int(total_frames * 0.05) if total_frames > 40 else 0
    hi = int(total_frames * 0.95) if total_frames > 40 else total_frames - 1
    indices = sorted({int(round(v)) for v in np.linspace(lo, max(lo, hi), n)})
    return indices, fps


def render_contact_sheet(
    rows: list[tuple[int, float, dict[str, tuple[np.ndarray, HudOcrResult]]]],
    fields: Sequence[str],
    out_path: Path,
) -> Path:
    """Save a visual contact sheet of cropped HUD ROIs and their OCR predictions."""
    cell_w, cell_img_h, label_h = 192, 48, 28
    pad = 8
    header_w = 135
    n_cols = len(fields)
    n_rows = len(rows)

    canvas_w = header_w + n_cols * (cell_w + pad) + pad
    canvas_h = 38 + n_rows * (cell_img_h + label_h + pad) + pad
    canvas = np.full((canvas_h, canvas_w, 3), 24, dtype=np.uint8)

    # Column headers
    for c_idx, field in enumerate(fields):
        x = header_w + pad + c_idx * (cell_w + pad)
        short = field.replace("hud_", "")
        cv2.putText(canvas, short, (x + 2, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 220, 140), 1, cv2.LINE_AA)

    for r_idx, (f_idx, sec, field_map) in enumerate(rows):
        y = 38 + r_idx * (cell_img_h + label_h + pad)
        cv2.putText(
            canvas,
            f"f={f_idx}",
            (10, y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            format_seconds(sec),
            (10, y + 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (160, 180, 200),
            1,
            cv2.LINE_AA,
        )

        for c_idx, field in enumerate(fields):
            x = header_w + pad + c_idx * (cell_w + pad)
            crop_rgb, res = field_map[field]
            crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
            ch, cw = crop_bgr.shape[:2]
            scale = min(cell_w / max(1, cw), cell_img_h / max(1, ch))
            nw = max(1, int(round(cw * scale)))
            nh = max(1, int(round(ch * scale)))
            resized = cv2.resize(crop_bgr, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
            y_off = y + (cell_img_h - nh) // 2
            x_off = x + (cell_w - nw) // 2
            canvas[y_off : y_off + nh, x_off : x_off + nw] = resized

            conf_pct = int(round(res.confidence * 100))
            txt = f"{res.text or '<blank>'} ({conf_pct}%)"
            if res.text == "":
                color = (200, 200, 140)
            elif res.parsed_value is not None:
                color = (120, 240, 140)
            else:
                color = (110, 130, 245)
            cv2.putText(
                canvas,
                txt[:26],
                (x + 2, y + cell_img_h + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    return out_path


def _crop_color_stats(crop_rgb: np.ndarray) -> tuple[tuple[int, int, int], int, tuple[int, int, int]]:
    """Return `(mean_rgb, green_dominance, mean_hsv)` for an RGB crop."""
    mean_rgb_f = crop_rgb.mean(axis=(0, 1))
    r, g, b = (int(round(mean_rgb_f[0])), int(round(mean_rgb_f[1])), int(round(mean_rgb_f[2])))
    g_dom = g - max(r, b)
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    mean_hsv_f = hsv.mean(axis=(0, 1))
    h, s, v = (int(round(mean_hsv_f[0])), int(round(mean_hsv_f[1])), int(round(mean_hsv_f[2])))
    return (r, g, b), g_dom, (h, s, v)


def render_single_field_grid(
    rows: list[tuple[int, float, dict[str, tuple[np.ndarray, HudOcrResult]]]],
    field: str,
    out_path: Path,
    cols: int = 12,
) -> Path:
    """Render a compact grid of single-field crops with OCR + average RGB/green-dominance labels."""
    cell_w, cell_img_h, label_h = 192, 48, 44
    pad = 8
    n_items = len(rows)
    n_cols = max(1, min(cols, n_items))
    n_rows = max(1, (n_items + n_cols - 1) // n_cols)

    canvas_w = pad + n_cols * (cell_w + pad)
    canvas_h = pad + n_rows * (cell_img_h + label_h + pad)
    canvas = np.full((canvas_h, canvas_w, 3), 24, dtype=np.uint8)

    for idx, (f_idx, _sec, fmap) in enumerate(rows):
        r_idx = idx // n_cols
        c_idx = idx % n_cols
        x = pad + c_idx * (cell_w + pad)
        y = pad + r_idx * (cell_img_h + label_h + pad)

        crop_rgb, res = fmap[field]
        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        ch, cw = crop_bgr.shape[:2]
        scale = min(cell_w / max(1, cw), cell_img_h / max(1, ch))
        nw = max(1, int(round(cw * scale)))
        nh = max(1, int(round(ch * scale)))
        resized = cv2.resize(crop_bgr, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        y_off = y + (cell_img_h - nh) // 2
        x_off = x + (cell_w - nw) // 2
        canvas[y_off : y_off + nh, x_off : x_off + nw] = resized

        (mr, mg, mb), g_dom, _ = _crop_color_stats(crop_rgb)
        conf_pct = int(round(res.confidence * 100))
        txt1 = f"f={f_idx} {res.text or '<blank>'} ({conf_pct}%)"
        txt2 = f"RGB({mr},{mg},{mb}) G+{g_dom}"
        color = (120, 240, 140) if g_dom >= 15 else ((200, 200, 140) if res.text == "" else (110, 130, 245))
        cv2.putText(canvas, txt1[:26], (x + 2, y + cell_img_h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)
        cv2.putText(canvas, txt2[:26], (x + 2, y + cell_img_h + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 215, 225), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    return out_path


def evaluate_vod(args: argparse.Namespace) -> Path:
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    reroll_mode = bool(getattr(args, "reroll_color_check", False))
    fields = ("hud_reroll_cost",) if reroll_mode else (tuple(args.fields) if args.fields else HUD_ROI_FIELDS)
    num_frames = args.num_frames if args.num_frames is not None else (144 if reroll_mode else 12)
    indices, fps = _pick_frame_indices(
        video_path,
        num_frames=num_frames,
        explicit_frames=args.frames,
        explicit_timestamps=args.timestamps,
    )
    frames_bgr = extract_video_frames(video_path, indices)
    pipe = HudOcrPipeline(weights_path=args.weights, device=args.device)

    all_crops: list[np.ndarray] = []
    all_fields: list[str] = []
    crop_keys: list[tuple[int, str]] = []

    for f_idx in indices:
        bgr = frames_bgr.get(f_idx)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        fh, fw = rgb.shape[:2]
        for field in fields:
            rx, ry, rw, rh = scale_roi_pixels(HUD_ROIS[field], fw, fh)
            crop = rgb[ry : ry + rh, rx : rx + rw]
            all_crops.append(crop)
            all_fields.append(field)
            crop_keys.append((f_idx, field))

    preds = pipe.predict_crops(all_crops, field_names=all_fields)

    by_frame: dict[int, dict[str, tuple[np.ndarray, HudOcrResult]]] = {}
    for (f_idx, field), crop, res in zip(crop_keys, all_crops, preds):
        by_frame.setdefault(f_idx, {})[field] = (crop, res)

    rows: list[tuple[int, float, dict[str, tuple[np.ndarray, HudOcrResult]]]] = []
    for f_idx in sorted(by_frame.keys()):
        sec = f_idx / fps
        fmap = by_frame[f_idx]
        rows.append((f_idx, sec, fmap))
        if reroll_mode:
            crop_rgb, res = fmap["hud_reroll_cost"]
            (mr, mg, mb), g_dom, (mh, ms, mv) = _crop_color_stats(crop_rgb)
            print(
                f"[{format_seconds(sec)} | f={f_idx}] reroll={res.text or '<blank>'} ({res.confidence:.0%}) | "
                f"RGB=({mr},{mg},{mb}) G-max(R,B)={g_dom:+d} | HSV=({mh},{ms},{mv})"
            )
        else:
            summary = " | ".join(
                f"{f.replace('hud_', '')}={fmap[f][1].text or '?'} ({fmap[f][1].confidence:.0%})"
                for f in fields
            )
            print(f"[{format_seconds(sec)} | f={f_idx}] {summary}")

    if reroll_mode:
        default_name = f"vod_reroll_color_{video_path.stem}.png"
        out_path = Path(args.out) if args.out else (CURRENT_DIR / "figures" / default_name)
        saved = render_single_field_grid(rows, "hud_reroll_cost", out_path, cols=12)
    else:
        out_path = Path(args.out) if args.out else (CURRENT_DIR / "figures" / f"vod_preview_{video_path.stem}.png")
        saved = render_contact_sheet(rows, fields, out_path)
    print(f"\nSaved visual contact sheet to: {saved}")
    return saved


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate HUD OCR on real VOD frames and save a contact sheet.")
    p.add_argument("--video", type=str, required=True, help="Path to a gameplay VOD (.mp4/.mkv/.avi)")
    p.add_argument("--num-frames", type=int, default=None, help="Number of evenly spaced frames to sample (default: 12, or 144 with --reroll-color-check)")
    p.add_argument("--frames", type=int, nargs="*", default=None, help="Specific frame indices to inspect")
    p.add_argument("--timestamps", type=str, nargs="*", default=None, help="Specific timestamps (e.g. 04:15 12:30)")
    p.add_argument("--fields", type=str, nargs="*", default=None, help="Subset of HUD fields to evaluate")
    p.add_argument("--reroll-color-check", action="store_true", help="Sample 144 hud_reroll_cost crops and print/render average RGB, green-dominance, and HSV")
    p.add_argument("--weights", type=str, default=None, help="Optional explicit checkpoint path")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default=None, help="Output contact-sheet PNG path")
    return p


if __name__ == "__main__":
    evaluate_vod(build_arg_parser().parse_args())
