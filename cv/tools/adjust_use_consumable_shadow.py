"""Interactive helper to manually tune 'Use Consumable' shadow ROI on failed VOD frames."""

from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config
from eval_template import eval as eval_tmpl
from template import template_loader

FRAMES_DIR = ROOT / "output" / "debug_use_consumable_frames"


def main() -> None:
    cfg = Config(setup_mode=True)
    tmpls = {
        t.name: t
        for t in template_loader(
            cfg.template_json_path, cfg.template_img_dir, cfg.resolution
        )
    }
    use_t = tmpls["Use Consumable"]
    use_t.adjusted = False  # Force interactive plot() UI even if marked adjusted

    pngs = sorted(FRAMES_DIR.glob("*.png"))
    if not pngs:
        raise FileNotFoundError(f"No PNG frames found in {FRAMES_DIR}")

    failed_depressed = [p for p in pngs if "depressed_failed" in p.name]
    print(
        f"Opening {len(failed_depressed)} failed DEPRESSED frames "
        "(these should all evaluate to clicked=True once shadow_dy/dw/dh are tuned)."
    )

    for path in failed_depressed:
        img = cv2.imread(str(path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f"\n>>> Tuning on {path.name} (expected: clicked=True)...")
        res = eval_tmpl(use_t, img, gray, cfg, video_timestamp_str=path.stem)
        print(
            f"  Result: found={res.found}, clicked={res.clicked}, "
            f"dy={use_t.shadow_dy}, dw={use_t.shadow_dw}, dh={use_t.shadow_dh}"
        )

    # Verify all extracted frames (unpressed -> False, depressed -> True)
    verify_cfg = Config(setup_mode=False)
    print("\n=== Verification across all extracted frames ===")
    for path in pngs:
        img = cv2.imread(str(path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        res = eval_tmpl(use_t, img, gray, verify_cfg)
        expected = "depressed" in path.name
        status = "PASS" if res.clicked == expected else "FAIL"
        print(f"  [{status}] {path.name}: clicked={res.clicked} (expected={expected})")


if __name__ == "__main__":
    main()
