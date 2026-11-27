import sys
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider, Button
from pathlib import Path

# Change this to the path of the image you want to crop
IMAGE_PATH = "assets/templates/drspectred/redeem.png"

# Project root directory (assumes script is inside tools/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(p: str) -> Path:
    path_obj = Path(p)
    if path_obj.is_absolute():
        return path_obj
    # Check relative to current working directory first
    if path_obj.exists():
        return path_obj.resolve()
    # Otherwise check relative to project root
    if (PROJECT_ROOT / path_obj).exists():
        return (PROJECT_ROOT / path_obj).resolve()
    return (PROJECT_ROOT / path_obj).resolve()


def main():
    target_path = resolve_path(IMAGE_PATH)
    if not target_path.exists():
        print(f"File not found: {IMAGE_PATH} (Resolved to: {target_path})")
        return

    # Read the image, keeping alpha channel if it exists
    img = cv2.imread(str(target_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Failed to load image from {target_path}")
        return

    # Convert for matplotlib display
    if img.ndim == 2:
        disp_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        disp_img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    else:
        disp_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(left=0.1, bottom=0.35)

    ax.imshow(disp_img)
    ax.set_title(f"Cropping {Path(IMAGE_PATH).name} ({w}x{h})")
    ax.axis("off")

    # Overlays for cropped regions
    color = "red"
    alpha = 0.5
    rect_top = patches.Rectangle(
        (0, 0), w, 0, linewidth=0, edgecolor="none", facecolor=color, alpha=alpha
    )
    rect_bottom = patches.Rectangle(
        (0, h), w, 0, linewidth=0, edgecolor="none", facecolor=color, alpha=alpha
    )
    rect_left = patches.Rectangle(
        (0, 0), 0, h, linewidth=0, edgecolor="none", facecolor=color, alpha=alpha
    )
    rect_right = patches.Rectangle(
        (w, 0), 0, h, linewidth=0, edgecolor="none", facecolor=color, alpha=alpha
    )

    ax.add_patch(rect_top)
    ax.add_patch(rect_bottom)
    ax.add_patch(rect_left)
    ax.add_patch(rect_right)

    # Sliders
    axcolor = "lightgoldenrodyellow"
    ax_top = plt.axes([0.2, 0.25, 0.65, 0.03], facecolor=axcolor)
    ax_bottom = plt.axes([0.2, 0.2, 0.65, 0.03], facecolor=axcolor)
    ax_left = plt.axes([0.2, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_right = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=axcolor)

    s_top = Slider(ax_top, "Top", 0, h - 1, valinit=0, valstep=1)
    s_bottom = Slider(ax_bottom, "Bottom", 0, h - 1, valinit=0, valstep=1)
    s_left = Slider(ax_left, "Left", 0, w - 1, valinit=0, valstep=1)
    s_right = Slider(ax_right, "Right", 0, w - 1, valinit=0, valstep=1)

    def update(val):
        t = int(s_top.val)
        b = int(s_bottom.val)
        l = int(s_left.val)
        r = int(s_right.val)

        # Prevent overlapping
        if t + b >= h:
            b = max(0, h - t - 1)
            s_bottom.set_val(b)
        if l + r >= w:
            r = max(0, w - l - 1)
            s_right.set_val(r)

        rect_top.set_height(t)

        rect_bottom.set_y(h - b)
        rect_bottom.set_height(b)

        # Left and right patches cover the middle section (avoids double overlapping which would make corners darker red)
        rect_left.set_y(t)
        rect_left.set_height(h - t - b)
        rect_left.set_width(l)

        rect_right.set_x(w - r)
        rect_right.set_y(t)
        rect_right.set_height(h - t - b)
        rect_right.set_width(r)

        fig.canvas.draw_idle()

    s_top.on_changed(update)
    s_bottom.on_changed(update)
    s_left.on_changed(update)
    s_right.on_changed(update)

    saveax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(saveax, "Save", color=axcolor, hovercolor="0.975")

    def save(event):
        t = int(s_top.val)
        b = int(s_bottom.val)
        l = int(s_left.val)
        r = int(s_right.val)

        y1 = t
        y2 = h - b
        x1 = l
        x2 = w - r

        if y2 <= y1 or x2 <= x1:
            print("Invalid crop bounds!")
            return

        cropped = img[y1:y2, x1:x2]
        cv2.imwrite(str(target_path), cropped)
        print(f"Saved cropped image to {target_path} (New Size: {x2 - x1}x{y2 - y1})")
        plt.close(fig)

    button.on_clicked(save)

    update(0)
    plt.show()


if __name__ == "__main__":
    main()
