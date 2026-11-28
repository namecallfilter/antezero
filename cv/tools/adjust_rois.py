import cv2
import json


def noop(val):
    pass


def main():
    with open("assets/templates.json", "r") as f:
        data = json.load(f)

    img = cv2.imread("assets/sample_inputs/generic_blind_1080p.png")
    if img is None:
        print("Could not load image assets/sample_inputs/generic_blind_1080p.png")
        return

    cv2.namedWindow("Adjust ROIs", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Adjust ROIs", 1280, 720)

    # Trackbars only support 0 to Max, so we map 0-400 to -200 to +200
    cv2.createTrackbar("X Offset", "Adjust ROIs", 200, 400, noop)
    cv2.createTrackbar("Y Offset", "Adjust ROIs", 200, 400, noop)

    BASE_WIDTH, BASE_HEIGHT = 1920, 1080

    print("Use the sliders to adjust the ROIs globally.")
    print("Press 's' or ENTER to save the new ROIs.")
    print("Press 'q' or ESC to cancel.")

    x_off = 0
    y_off = 0

    while True:
        x_off = cv2.getTrackbarPos("X Offset", "Adjust ROIs") - 200
        y_off = cv2.getTrackbarPos("Y Offset", "Adjust ROIs") - 200

        display_img = img.copy()

        for key, val in data.items():
            rois = val.get("rois", [])
            label = val.get("label", key)
            for roi in rois:
                min_x_ratio, min_y_ratio, max_x_ratio, max_y_ratio = roi
                x1 = round(min_x_ratio * BASE_WIDTH) + x_off
                y1 = round(min_y_ratio * BASE_HEIGHT) + y_off
                x2 = round(max_x_ratio * BASE_WIDTH) + x_off
                y2 = round(max_y_ratio * BASE_HEIGHT) + y_off

                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    display_img,
                    label,
                    (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

        cv2.imshow("Adjust ROIs", display_img)

        key_press = cv2.waitKey(30) & 0xFF
        if key_press == ord("s") or key_press == 13:  # s or Enter
            break
        elif key_press == ord("q") or key_press == 27:  # q or Esc
            print("Cancelled. No changes made.")
            cv2.destroyAllWindows()
            return

    # Apply offsets
    for key, val in data.items():
        rois = val.get("rois", [])
        new_rois = []
        for roi in rois:
            min_x_ratio, min_y_ratio, max_x_ratio, max_y_ratio = roi

            # calculate new ratios based on the pixel offset
            new_min_x = round(((min_x_ratio * BASE_WIDTH) + x_off) / BASE_WIDTH, 4)
            new_min_y = round(((min_y_ratio * BASE_HEIGHT) + y_off) / BASE_HEIGHT, 4)
            new_max_x = round(((max_x_ratio * BASE_WIDTH) + x_off) / BASE_WIDTH, 4)
            new_max_y = round(((max_y_ratio * BASE_HEIGHT) + y_off) / BASE_HEIGHT, 4)

            new_rois.append([new_min_x, new_min_y, new_max_x, new_max_y])
        val["rois"] = new_rois

    cv2.destroyAllWindows()

    with open("assets/templates.json", "w") as f:
        json.dump(data, f, indent=4)

    print(f"\nApplied global offsets: X={x_off}px, Y={y_off}px")
    print("Saved new coordinates to assets/templates.json")


if __name__ == "__main__":
    main()
