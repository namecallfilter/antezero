"""Interactive helper to extract stratified real-VOD shelf crops and bootstrap
ground-truth labels for `classifiers/shop_classifier/evaluate.py --sweep`.

Usage examples:
  # Stratified walkthrough using an actions JSON (samples pack, shop, voucher, booster, bars)
  python tools/extract_vod_shop_val.py --video assets/sample_inputs/videos/bu_anaglyph_deck_11-13-26.mp4 --actions-json output/bu_anaglyph_deck_11-13-26_1794699311.json

  # Target specific frames & shelves (frame:shelf)
  python tools/extract_vod_shop_val.py --video assets/sample_inputs/videos/bu_anaglyph_deck_11-13-26.mp4 --frames 25573:pack,27022:pack,32420:shop

  # Then run the F1 / PR sweep:
  python classifiers/shop_classifier/evaluate.py --dataset-dir classifiers/shop_classifier/vod_val --sweep
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classifiers.inference import _crop_roi_to_pil, get_shop_pipeline

DEFAULT_OUT_DIR = ROOT / "classifiers" / "shop_classifier" / "vod_val"


def _infer_card_type(key: str) -> str:
    if key.startswith("j_"):
        return "Joker"
    if key.startswith("v_"):
        return "Voucher"
    if key.startswith("p_"):
        return "Booster"
    if key.startswith("c_"):
        return "Tarot"
    return "PlayingCard"


def _prompt_with_gui_pump(prompt: str, win_name: str, img: np.ndarray) -> str:
    """Reads a line from stdin while pumping OpenCV HighGUI events on the main thread."""
    print(prompt, end="", flush=True)
    result_q: queue.Queue[str] = queue.Queue()

    def _read_stdin() -> None:
        try:
            line = sys.stdin.readline()
        except EOFError:
            line = ""
        result_q.put(line)

    threading.Thread(target=_read_stdin, daemon=True).start()
    while True:
        cv2.imshow(win_name, img)
        cv2.waitKey(30)
        try:
            return result_q.get_nowait().strip()
        except queue.Empty:
            pass


def _sample_candidates_from_actions(
    actions_json: Path,
    *,
    per_bucket: int = 8,
) -> list[tuple[int, str, str]]:
    """Selects stratified (frame_idx, shelf_key, reason) candidates from an actions JSON."""
    with open(actions_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    inv_key = {v: k for k, v in data["key"].items()}
    buckets: dict[str, list[tuple[int, str, str]]] = {
        "pack": [],
        "shop": [],
        "voucher": [],
        "booster": [],
        "joker_bar": [],
        "consumeables": [],
    }
    for f_str, entry in sorted(data["frames"].items(), key=lambda kv: int(kv[0])):
        fid = int(f_str)
        act_id = entry["action"] if isinstance(entry, dict) else entry
        name = inv_key.get(act_id, "")
        target_f = max(0, fid - 2)
        if name in ("Use In Pack", "Skip Pack"):
            buckets["pack"].append((target_f, "pack", name))
        elif name in ("Buy Card", "Buy and Use", "Reroll Shop"):
            buckets["shop"].append((target_f, "shop", name))
        elif name == "Redeem Voucher":
            buckets["voucher"].append((target_f, "voucher", name))
        elif name == "Open Pack":
            buckets["booster"].append((target_f, "booster", name))
        elif name in ("Use Consumable", "Sell Card"):
            buckets["consumeables"].append((target_f, "consumeables", name))
            buckets["joker_bar"].append((target_f, "joker_bar", name))
        elif name == "Play Hand":
            buckets["joker_bar"].append((target_f, "joker_bar", name))

    selected: list[tuple[int, str, str]] = []
    for shelf, items in buckets.items():
        if not items:
            continue
        if len(items) <= per_bucket:
            selected.extend(items)
        else:
            step = len(items) / per_bucket
            selected.extend(items[int(i * step)] for i in range(per_bucket))
    return selected


def _parse_frame_specs(spec_str: str) -> list[tuple[int, str, str]]:
    out: list[tuple[int, str, str]] = []
    for tok in spec_str.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" in tok:
            f_s, shelf = tok.split(":", 1)
            out.append((int(f_s.strip()), shelf.strip(), "manual"))
        else:
            out.append((int(tok), "shop", "manual"))
    return out


def _draw_annotated_preview(crop_bgr: np.ndarray, items: list[dict[str, Any]]) -> np.ndarray:
    vis = crop_bgr.copy()
    for idx, it in enumerate(items):
        x, y, w, h = it["box_pixel"]
        det = it.get("_det", 1.0)
        key_c = it.get("_key_conf", 1.0)
        color = (0, 220, 0) if det >= 0.5 else (0, 180, 255)
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        label = f"[{idx}] {it['key']} ({det:.2f}/{key_c:.2f})"
        cv2.putText(
            vis, label, (x, max(16, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA
        )
    return vis


def main() -> None:
    p = argparse.ArgumentParser(description="Extract & bootstrap VOD shelf crops for F1 sweep.")
    p.add_argument("--video", type=Path, required=True, help="Path to source .mp4 VOD")
    p.add_argument("--actions-json", type=Path, default=None, help="Optional main.py actions JSON for stratified sampling")
    p.add_argument("--frames", type=str, default="", help="Comma-separated frame:shelf pairs, e.g. 25573:pack,27022:pack")
    p.add_argument("--per-bucket", type=int, default=6, help="Samples per shelf bucket when using --actions-json")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output dataset directory")
    p.add_argument("--bootstrap-conf", type=float, default=0.05, help="Stage 1 proposal conf for draft labels")
    p.add_argument("--auto", action="store_true", help="Save all draft crops/labels without interactive review")
    args = p.parse_args()

    candidates: list[tuple[int, str, str]] = []
    if args.frames:
        candidates.extend(_parse_frame_specs(args.frames))
    if args.actions_json:
        candidates.extend(_sample_candidates_from_actions(args.actions_json, per_bucket=args.per_bucket))
    if not candidates:
        raise SystemExit("Provide --actions-json and/or --frames.")

    images_dir = args.out_dir / "images"
    labels_dir = args.out_dir / "labels_json"
    previews_dir = args.out_dir / "previews"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    pipeline = get_shop_pipeline()
    saved = 0

    try:
        for i, (fid, shelf, reason) in enumerate(candidates, start=1):
            stem = f"vod_f{fid}_{shelf}"
            if (labels_dir / f"{stem}.json").exists():
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame_bgr = cap.read()
            if not ret:
                print(f"[Warn] Failed to read frame {fid}")
                continue

            crop_pil, _ = _crop_roi_to_pil(frame_bgr, shelf)
            crop_bgr = cv2.cvtColor(np.array(crop_pil), cv2.COLOR_RGB2BGR)
            preds = pipeline.predict(crop_pil, conf_threshold=args.bootstrap_conf)

            draft: list[dict[str, Any]] = []
            for pred in preds:
                x1, y1, x2, y2 = pred.box
                draft.append(
                    {
                        "key": pred.key,
                        "card_type": pred.card_type,
                        "edition": pred.edition,
                        "box_pixel": [x1, y1, max(1, x2 - x1), max(1, y2 - y1)],
                        "_det": float(pred.confidences.get("detection", 1.0)),
                        "_key_conf": float(pred.confidences.get("key", 1.0)),
                    }
                )

            preview = _draw_annotated_preview(crop_bgr, draft)
            cv2.imwrite(str(previews_dir / f"{stem}.jpg"), preview)

            if args.auto:
                clean = [{k: v for k, v in it.items() if not k.startswith("_")} for it in draft]
                crop_pil.save(images_dir / f"{stem}.png")
                with open(labels_dir / f"{stem}.json", "w", encoding="utf-8") as f:
                    json.dump(clean, f, indent=2)
                saved += 1
                print(f"[{i}/{len(candidates)}] Auto-saved {stem} ({len(clean)} items)")
                continue

            win_name = "VOD Val Review (see terminal)"
            while True:
                preview = _draw_annotated_preview(crop_bgr, draft)
                cv2.imshow(win_name, preview)
                cv2.waitKey(1)
                print(f"\n[{i}/{len(candidates)}] {stem} ({reason}) — {len(draft)} draft item(s):")
                for idx, it in enumerate(draft):
                    print(
                        f"  [{idx}] {it['key']:20s} det={it['_det']:.3f} key={it['_key_conf']:.3f} "
                        f"box={it['box_pixel']}"
                    )
                cmd = _prompt_with_gui_pump(
                    "Action [Enter=accept, d <idx,...>=delete, k <idx>=<key>, a=add box, s=skip, q=quit]: ",
                    win_name,
                    preview,
                )

                if cmd == "":
                    clean = [{k: v for k, v in it.items() if not k.startswith("_")} for it in draft]
                    crop_pil.save(images_dir / f"{stem}.png")
                    with open(labels_dir / f"{stem}.json", "w", encoding="utf-8") as f:
                        json.dump(clean, f, indent=2)
                    saved += 1
                    print(f"  -> Saved {stem} ({len(clean)} items)")
                    break
                if cmd.lower() == "s":
                    print("  -> Skipped")
                    break
                if cmd.lower() == "q":
                    cv2.destroyAllWindows()
                    print(f"\nSaved {saved} crops to {args.out_dir}.")
                    print(
                        f"Run sweep: python classifiers/shop_classifier/evaluate.py "
                        f"--dataset-dir {args.out_dir} --sweep"
                    )
                    return
                if cmd.lower().startswith("d "):
                    rm = {int(x.strip()) for x in cmd[2:].split(",") if x.strip().isdigit()}
                    draft = [it for j, it in enumerate(draft) if j not in rm]
                elif cmd.lower().startswith("k ") and "=" in cmd[2:]:
                    idx_s, new_key = cmd[2:].split("=", 1)
                    if idx_s.strip().isdigit() and int(idx_s.strip()) < len(draft):
                        j = int(idx_s.strip())
                        draft[j]["key"] = new_key.strip()
                        draft[j]["card_type"] = _infer_card_type(new_key.strip())
                elif cmd.lower() == "a":
                    roi = cv2.selectROI("Draw missing card box (Enter/Space to confirm)", crop_bgr, showCrosshair=True)
                    cv2.destroyWindow("Draw missing card box (Enter/Space to confirm)")
                    x, y, w, h = [int(v) for v in roi]
                    if w > 0 and h > 0:
                        new_key = _prompt_with_gui_pump(
                            "  Card key (e.g. c_judgement, j_ramen, v_overstock): ",
                            win_name,
                            preview,
                        )
                        if new_key:
                            draft.append(
                                {
                                    "key": new_key,
                                    "card_type": _infer_card_type(new_key),
                                    "edition": "None",
                                    "box_pixel": [x, y, w, h],
                                    "_det": 1.0,
                                    "_key_conf": 1.0,
                                }
                            )
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\nDone! Saved {saved} crops to {args.out_dir}.")
    print(
        f"Run sweep: python classifiers/shop_classifier/evaluate.py "
        f"--dataset-dir {args.out_dir} --sweep"
    )


if __name__ == "__main__":
    main()
