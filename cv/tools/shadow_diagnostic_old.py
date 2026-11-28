import cv2
import numpy as np

from tools.debug_plotter import plot
from geometry import PixelROI
from matcher import find_element

F_ELEM_CONF_THRESH = 0.95  # found element confidence threshold
PRESSED_CONF_THRESH = 0.9  # is the button pressed? confidence threshold

COLORS = {
    "sel_blind_gray": (64, 61, 44, 255),
    "sel_blind_drop_shadow": (36, 34, 21, 255),
}

# METHOD = cv2.TM_CCOEFF_NORMED
METHOD = cv2.TM_CCORR_NORMED


def get_avg_color_from_region(
    img: np.ndarray,
    y_start: int,
    y_end: int,
    x_start: int,
    x_end: int,
    anchor: tuple = (0, 0),
) -> tuple:
    anchor_x, anchor_y = anchor

    abs_y_start = y_start + anchor_y
    abs_y_end = anchor_y + y_end
    abs_x_start = anchor_x + x_start
    abs_x_end = anchor_x + x_end

    # region of interest
    roi = img[abs_y_start:abs_y_end, abs_x_start:abs_x_end]

    average_color_per_row = np.average(roi, axis=0)
    average_color = np.average(average_color_per_row, axis=0)

    # round colors & return
    return tuple(np.round(average_color).astype(int))


def evaluate_shadow(img, top_left, max_val, w_inp, h_inp) -> bool:
    """Reads the shadow under a region to determine if a button is clicked."""
    bottom_right = (top_left[0] + w_inp, top_left[1] + h_inp)

    # shadow_region: y_start, y_end, x_start, x_end
    sr = (h_inp - 1, h_inp + 2, 20, w_inp - 20)

    avg_bgr = get_avg_color_from_region(
        img=img,
        y_start=sr[0],
        y_end=sr[1],
        x_start=sr[2],
        x_end=sr[3],
        anchor=top_left,
    )

    abs_y_start = top_left[1] + sr[0]
    abs_y_end = top_left[1] + sr[1]
    abs_x_start = top_left[0] + sr[2]
    abs_x_end = top_left[0] + sr[3]

    shadow_img = img.copy()  # shadow_img shows shadow region
    cv2.rectangle(img, top_left, bottom_right, color=(0, 0, 255), thickness=10)
    cv2.rectangle(
        shadow_img,
        (abs_x_start, abs_y_start),
        (abs_x_end, abs_y_end),
        color=(0, 0, 255),
        thickness=1,
    )

    # is button pressed? based on color
    avg_bgr_np = np.array(avg_bgr)
    sel_blind_gray_np = np.array(COLORS["sel_blind_gray"])
    sel_blind_drop_shadow_np = np.array(COLORS["sel_blind_drop_shadow"])

    no_shadow_dist = np.average(abs(avg_bgr_np - sel_blind_gray_np))
    shadow_dist = np.average(abs(avg_bgr_np - sel_blind_drop_shadow_np))

    clicked = no_shadow_dist < shadow_dist
    return bool(clicked)


if __name__ == "__main__":
    bottom_right = None
    abs_x_start, abs_y_start, abs_x_end, abs_y_end = None, None, None, None

    img = cv2.imread("assets/sample_inputs/click_run_info.png", cv2.IMREAD_UNCHANGED)
    template = cv2.imread("assets/templates/run_info.png", cv2.IMREAD_UNCHANGED)

    if (template is None) or (img is None):
        raise ValueError(
            f"Got None for img or template: img={img}; template={template}"
        )

    h_inp, w_inp = template.shape[:2]
    original = img.copy()
    image_height, image_width = img.shape[:2]
    full_image_roi = PixelROI(1, 1, image_width, image_height)
    match = find_element(
        img,
        template,
        METHOD,
        full_image_roi,
        F_ELEM_CONF_THRESH,
    )
    found_target = match.found
    top_left = match.top_left
    max_val = match.confidence
    match_map = match.score_map
    if found_target:
        print(f"Found target at {top_left} with confidence of {max_val}.")

        bottom_right = (top_left[0] + w_inp, top_left[1] + h_inp)

        # shadow_region: y_start, y_end, x_start, x_end
        sr = (h_inp - 1, h_inp + 2, 20, w_inp - 20)

        avg_bgr = get_avg_color_from_region(
            img=img,
            y_start=sr[0],
            y_end=sr[1],
            x_start=sr[2],
            x_end=sr[3],
            anchor=top_left,
        )

        abs_y_start = top_left[1] + sr[0]
        abs_y_end = top_left[1] + sr[1]
        abs_x_start = top_left[0] + sr[2]
        abs_x_end = top_left[0] + sr[3]

        # is button pressed? based on color
        avg_bgr_np = np.array(avg_bgr)
        sel_blind_gray_np = np.array(COLORS["sel_blind_gray"])
        sel_blind_drop_shadow_np = np.array(COLORS["sel_blind_drop_shadow"])

        no_shadow_dist = np.average(abs(avg_bgr_np - sel_blind_gray_np))
        shadow_dist = np.average(abs(avg_bgr_np - sel_blind_drop_shadow_np))

        clicked = no_shadow_dist < shadow_dist

        print(f"Avg color of region={avg_bgr}")
        print(f"hover_dist={no_shadow_dist}; shadow_dist={shadow_dist}")
        print(f"Clicked={clicked}")

    else:
        print(f"Did not find target; highest confidence of {max_val}")
        clicked = False

    plot(
        img,
        template,
        found_target,
        clicked,
        top_left,
        bottom_right,
        abs_x_start,
        abs_y_start,
        abs_x_end,
        abs_y_end,
        match,
    )
