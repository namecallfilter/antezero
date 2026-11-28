"""Interactive CLI to tune shadow_dy / shadow_dw / shadow_dh for a single template on its test_frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config
from eval_template import eval as eval_tmpl
from template import template_loader


def _verify_all(tmpl, test_frames: list[dict], cfg: Config) -> list[tuple[dict, bool]]:
    failing: list[tuple[dict, bool]] = []
    print(f"\n=== Verification across {len(test_frames)} test_frames for '{tmpl.name}' ===")
    for entry in test_frames:
        img_path = ROOT / entry["path"]
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [SKIP] missing image: {img_path}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        res = eval_tmpl(tmpl, img, gray, cfg)
        expected = bool(entry["pressed"])
        ok = res.found and (res.clicked == expected)
        status = "PASS" if ok else "FAIL"
        print(
            f"  [{status}] {entry['path']} (frame={entry.get('frame_idx')}): "
            f"found={res.found}, clicked={res.clicked} (expected={expected})"
        )
        if not ok:
            failing.append((entry, res.clicked))
    return failing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune shadow region for a single template using its test_frames in templates.json."
    )
    parser.add_argument("template_name", help='Exact template label, e.g. "Select Card"')
    parser.add_argument(
        "--all",
        action="store_true",
        help="Open interactive tuner on all test_frames instead of only currently failing ones.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only print pass/fail across test_frames without opening interactive plot windows.",
    )
    args = parser.parse_args()

    cfg = Config(setup_mode=True)
    with open(cfg.template_json_path, "r", encoding="utf-8") as f:
        raw_json = json.load(f)

    if args.template_name not in raw_json:
        raise SystemExit(
            f"Template {args.template_name!r} not found in {cfg.template_json_path}"
        )

    test_frames = raw_json[args.template_name].get("test_frames", [])
    if not test_frames:
        raise SystemExit(f"No test_frames assigned to {args.template_name!r}.")

    tmpls = {
        t.name: t
        for t in template_loader(cfg.template_json_path, cfg.template_img_dir, cfg.resolution)
    }
    tmpl = tmpls[args.template_name]

    failing = _verify_all(tmpl, test_frames, Config(setup_mode=False))
    if args.verify_only:
        return

    targets = test_frames if args.all else [entry for entry, _ in failing]
    if not targets:
        print(f"\nAll {len(test_frames)} test_frames already pass for '{tmpl.name}'!")
        return

    print(f"\nOpening interactive tuner on {len(targets)} frame(s) for '{tmpl.name}'...")
    for entry in targets:
        img = cv2.imread(str(ROOT / entry["path"]))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tmpl.adjusted = False  # Force plot() UI even if already marked adjusted
        print(
            f"\n>>> Tuning '{tmpl.name}' on {entry['path']} "
            f"(expected pressed={entry['pressed']})..."
        )
        res = eval_tmpl(tmpl, img, gray, cfg, video_timestamp_str=f"f{entry.get('frame_idx', '?')}")
        print(
            f"  Result: found={res.found}, clicked={res.clicked}, "
            f"dy={tmpl.shadow_dy}, dw={tmpl.shadow_dw}, dh={tmpl.shadow_dh}, "
            f"thr={tmpl.shadow_threshold}"
        )

    # Reload from templates.json so saved adjustments are verified
    reloaded = {
        t.name: t
        for t in template_loader(cfg.template_json_path, cfg.template_img_dir, cfg.resolution)
    }[args.template_name]
    _verify_all(reloaded, test_frames, Config(setup_mode=False))


if __name__ == "__main__":
    main()
