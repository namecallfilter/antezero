import json
from pathlib import Path
import sys

# Add parent directory to path so we can import config and shared tools helpers
sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from config import Config
from tools.video_scrubber import (
    FrameRange,
    ScrubberController,
    ScrubberVideoSession,
    decode_scrubber_key,
    draw_hud_panel,
)


def _overlay_template_preview(display_img: np.ndarray, template_img_path: Path) -> None:
    if not template_img_path.exists():
        return
    t_img = cv2.imread(str(template_img_path), cv2.IMREAD_UNCHANGED)
    if t_img is None:
        return
    if t_img.ndim == 3 and t_img.shape[2] == 4:
        alpha = t_img[:, :, 3:4] / 255.0
        t_img = (t_img[:, :, :3] * alpha).astype(np.uint8)

    th, tw = t_img.shape[:2]
    h, w = display_img.shape[:2]
    y_off, x_off = 50, w - tw - 50
    if x_off > 5 and y_off + th < h - 5:
        cv2.rectangle(display_img, (x_off - 5, y_off - 5), (x_off + tw + 5, y_off + th + 5), (0, 0, 0), -1)
        display_img[y_off : y_off + th, x_off : x_off + tw] = t_img


def main() -> None:
    config = Config()
    json_path = Path(config.template_json_path)

    if not json_path.exists():
        print(f"Error: {json_path} does not exist. Cannot run annotator.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        templates_data = json.load(f)

    for k in templates_data:
        templates_data[k].setdefault("test_frames", [])

    template_keys = list(templates_data.keys())
    if not template_keys:
        print("No templates found in JSON.")
        return

    try:
        session = ScrubberVideoSession(config.input_video_path, thumb_width=426, thumb_height=240)
    except Exception as exc:
        print(f"Error: Could not open video file {config.input_video_path} with decord. ({exc})")
        return

    testing_dir = Path("assets/testing")
    testing_dir.mkdir(parents=True, exist_ok=True)
    full_range = FrameRange(0, session.total_frames - 1)
    session.start_thumbnail_cache(frame_range=full_range, stride=15)

    win_name = "Annotator"
    scrubber = ScrubberController(win_name, window_width=1280, window_height=720, initial_range=full_range)
    state = "LANDING"
    current_template_idx = 0

    def save_json() -> None:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(templates_data, f, indent=4)

    try:
        while scrubber.is_window_open():
            if state == "LANDING":
                img = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(img, "Template Annotation Progress", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                y = 100
                start_i = max(0, current_template_idx - 10)
                end_i = min(len(template_keys), start_i + 20)
                for i in range(start_i, end_i):
                    k = template_keys[i]
                    count = len(templates_data[k]["test_frames"])
                    text = f"{k}: {count} frames"
                    color = (0, 255, 0) if i == current_template_idx else (200, 200, 200)
                    if i == current_template_idx:
                        text = "-> " + text
                    cv2.putText(img, text, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                    y += 25

                cv2.putText(img, "Press ENTER to start/continue with selected template.", (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(img, "W / S or Up/Down arrows to select template.", (600, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(img, f"Background Cache Loading: {session.caching_progress:.1f}%", (600, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img, "Press Q to quit.", (600, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow(win_name, img)
                key = decode_scrubber_key(cv2.waitKeyEx(30))
                if key == "enter":
                    state = "ANNOTATING"
                    scrubber.set_frame(scrubber.frame_idx, update_trackbar=True)
                elif key == "q":
                    break
                elif key in ("w", "up"):
                    current_template_idx = max(0, current_template_idx - 1)
                elif key in ("s", "down"):
                    current_template_idx = min(len(template_keys) - 1, current_template_idx + 1)
                continue

            prefer_thumb = scrubber.advance_playback(session.fps)
            ret = True
            try:
                frame, used_thumb = session.get_display_frame(
                    scrubber.frame_idx, prefer_thumbnail=prefer_thumb
                )
            except Exception:
                ret = False
                used_thumb = False
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(frame, "End of video / Error reading frame", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            display_img = frame.copy()
            k = template_keys[current_template_idx]
            count = len(templates_data[k]["test_frames"])
            _overlay_template_preview(
                display_img, Path(config.template_img_dir) / templates_data[k]["filenames"][0]
            )

            draw_hud_panel(
                display_img,
                [
                    (f"Template: {k} ({count} saved)", (0, 255, 0)),
                    ("SPACE: Play/Pause | A/Left: -1f | D/Right: +1f", (255, 255, 0)),
                    ("T: Tag True | F: Tag False | N: Skip/Next | M: Menu", (255, 255, 0)),
                    (", / . : -/+ 0.5s | J / L : -/+ 5.0s skips", (255, 255, 0)),
                ],
                origin=(30, 20),
                font_scale=0.62,
                line_height=30,
            )
            if used_thumb:
                h = display_img.shape[0]
                cv2.putText(display_img, "SCRUBBING", (50, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

            cv2.imshow(win_name, display_img)
            key = decode_scrubber_key(cv2.waitKeyEx(1 if scrubber.playing else 30))

            if key == "space":
                scrubber.toggle_play()
            elif key in ("a", "left"):
                scrubber.step_frames(-1)
            elif key in ("d", "right"):
                scrubber.step_frames(1)
            elif key == ",":
                scrubber.step_frames(-int(session.fps * 0.5), scrub_hint=True)
            elif key == ".":
                scrubber.step_frames(int(session.fps * 0.5), scrub_hint=True)
            elif key == "j":
                scrubber.step_frames(-int(session.fps * 5.0), scrub_hint=True)
            elif key == "l":
                scrubber.step_frames(int(session.fps * 5.0), scrub_hint=True)
            elif key in ("t", "f") and ret:
                save_frame = session.get_exact_frame(scrubber.frame_idx) if used_thumb else frame
                pressed = key == "t"
                safe_name = k.replace(" ", "_").lower()
                template_dir = testing_dir / safe_name
                template_dir.mkdir(exist_ok=True)

                out_path = template_dir / f"frame_{scrubber.frame_idx}_pressed_{str(pressed).lower()}.jpg"
                cv2.imwrite(str(out_path), save_frame)
                templates_data[k]["test_frames"].append(
                    {
                        "path": out_path.as_posix(),
                        "pressed": pressed,
                        "frame_idx": scrubber.frame_idx,
                    }
                )
                save_json()

                hh, ww = display_img.shape[:2]
                cv2.putText(display_img, "SAVED!", (ww // 2 - 100, hh // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                cv2.imshow(win_name, display_img)
                cv2.waitKey(100)
            elif key == "n":
                current_template_idx = min(len(template_keys) - 1, current_template_idx + 1)
            elif key in ("m", "esc"):
                state = "LANDING"
                scrubber.playing = False
            elif key == "q":
                break
    finally:
        session.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
