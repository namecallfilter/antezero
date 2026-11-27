import json
import os
import subprocess
import tempfile
import time
from collections import defaultdict

import cv2
import numpy as np

from config import Config
from eval_template import eval as eval_tpl
from post_processing import remove_duplicates
from template import template_loader
from tools.compare_trajectories import compare_trajectories


def main() -> None:
    t0 = time.perf_counter()
    cfg = Config()
    tmpls = {
        t.name: t
        for t in template_loader(
            cfg.template_json_path, cfg.template_img_dir, cfg.resolution
        )
    }
    act_json = json.load(open("output/bu_anaglyph_deck_11-7-26_1795247075.json"))
    inv_key = {v: k for k, v in act_json["key"].items()}
    dedup = remove_duplicates(act_json["frames"])

    rep = compare_trajectories(
        "output/trajectories/bu_anaglyph_deck_11-7-26_1795247075_trajectory.json",
        "assets/local/ground_truth/bu_anaglyph_deck_11-7-26.jsonl",
    )
    extra_frames: set[int] = set()
    for err in rep.get("errors", []):
        for d in err.get("discrepancies", []):
            if d.get("kind") in ("extra_action", "action_type_mismatch"):
                extra_frames.add(err.get("frame_idx"))

    target_names = {
        "Next Round",
        "Buy Card",
        "Play Hand",
        "Open Pack",
        "Buy and Use",
        "Redeem Voucher",
        "Discard",
        "Cash Out",
    }
    by_frame_names: dict[int, list[str]] = defaultdict(list)
    for f_str, entry in dedup.items():
        f_idx = int(f_str)
        aid = entry["action"] if isinstance(entry, dict) else entry
        name = inv_key[aid]
        if name in target_names:
            by_frame_names[f_idx].append(name)

    needed = sorted(by_frame_names.keys())
    expr = "+".join(f"eq(n\\,{f})" for f in needed)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
        tf.write(f"select={expr}")
        fpath = tf.name

    print(f"Extracting {len(needed)} frames via ffmpeg select filter_script...", flush=True)
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-hwaccel",
        "cuda",
        "-c:v",
        "av1",
        "-i",
        "assets/sample_inputs/videos/bu_anaglyph_deck_11-7-26.mp4",
        "-filter_script:v",
        fpath,
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "nv12",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1920 * 1080 * 3)
    nv12_bytes = (1920 * 1080 * 3) // 2
    results_by_name: dict[str, list[tuple[int, float, int, bool | None]]] = defaultdict(list)
    idx = 0
    assert proc.stdout is not None
    while idx < len(needed):
        buf = proc.stdout.read(nv12_bytes)
        if len(buf) < nv12_bytes:
            break
        f_idx = needed[idx]
        idx += 1
        nv12 = np.frombuffer(buf, dtype=np.uint8).reshape(((1080 * 3) // 2, 1920))
        img = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for name in by_frame_names[f_idx]:
            res = eval_tpl(tmpls[name], img, gray, cfg)
            conf = res.match.confidence if res.match else 0.0
            y1 = res.box[1] if res.box else -1
            results_by_name[name].append((f_idx, round(conf, 4), y1, res.clicked))
    proc.wait()
    os.unlink(fpath)

    print(f"Decoded {idx}/{len(needed)} in {time.perf_counter()-t0:.1f}s")
    for name in [
        "Next Round",
        "Buy Card",
        "Play Hand",
        "Open Pack",
        "Buy and Use",
        "Redeem Voucher",
        "Discard",
        "Cash Out",
    ]:
        rows = sorted(results_by_name[name])
        tp = [r for r in rows if r[0] not in extra_frames]
        fp = [r for r in rows if r[0] in extra_frames]
        tp_confs = [r[1] for r in tp]
        tp_y1s = [r[2] for r in tp]
        print(f"=== {name}: total={len(rows)}, non_extra={len(tp)}, extra={len(fp)} ===")
        if tp_confs:
            print(
                f"  non_extra conf: min={min(tp_confs):.4f} p5={np.percentile(tp_confs,5):.4f} p10={np.percentile(tp_confs,10):.4f} med={np.median(tp_confs):.4f} max={max(tp_confs):.4f}"
            )
            print(
                f"  non_extra y1  : min={min(tp_y1s)} max={max(tp_y1s)} uniq={sorted(set(tp_y1s))[:15]}"
            )
        print(f"  lowest_8 all  : {sorted(rows, key=lambda x: x[1])[:8]}")
        if fp:
            print(f"  known extras  : {fp}")


if __name__ == "__main__":
    main()
