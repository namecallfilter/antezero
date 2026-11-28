import json
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button

from geometry import PixelROI


def _to_rgb(image: np.ndarray) -> np.ndarray:
    """Helper to convert BGR/BGRA/Grayscale to RGB/RGBA for matplotlib."""
    if image is None:
        return image
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def plot(
    img: np.ndarray,
    template: np.ndarray,
    roi: tuple[int, int, int, int],
    found_target: bool,
    match: np.ndarray,
    top_left: tuple[int, int] | None = None,
    max_val: float | None = None,
    clicked: bool | None = None,
    label: str = "",
    shadow_region: PixelROI | None = None,
    shadow_threshold: float = 1000.0,
    template_name: str = "",
    template_json_path: str = "",
    current_shadow_dy: int = 0,
    current_shadow_dw: int = 0,
    current_shadow_dh: int = 0,
    shadow_height: int = 10,
    video_timestamp_str: str = "",
) -> None:
    """Plots visual debug data without relying on any global variables.

    Displays:
    - Full image with blue box for ROI and red box for detected element
    - Zoomed-in ROI with red box if found
    - cv2.matchTemplate response heatmap with colorbar
    - Template image
    - Cropped view of the detected element (if found)
    - Cropped view of shadow region with clicked status (if found)
    """
    x1, y1, x2, y2 = roi
    h_inp, w_inp = template.shape[:2]

    annotated_full = img.copy()
    # Blue box around ROI in BGR: (255, 0, 0)
    cv2.rectangle(annotated_full, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=3)

    bottom_right = None
    if found_target and top_left is not None:
        bottom_right = (top_left[0] + w_inp, top_left[1] + h_inp)
        # Red box around element in BGR: (0, 0, 255)
        cv2.rectangle(
            annotated_full, top_left, bottom_right, color=(0, 0, 255), thickness=3
        )

    # Zoomed ROI slice
    roi_crop = img[y1:y2, x1:x2].copy()
    if found_target and top_left is not None:
        rel_top_left = (top_left[0] - x1, top_left[1] - y1)
        rel_bottom_right = (rel_top_left[0] + w_inp, rel_top_left[1] + h_inp)
        cv2.rectangle(
            roi_crop, rel_top_left, rel_bottom_right, color=(0, 0, 255), thickness=2
        )

    fig, axs = plt.subplots(2, 4, figsize=(20, 8))
    # axs[0, 3].axis("off")  # Handled below

    # Overall title
    title_parts = []
    if video_timestamp_str:
        title_parts.append(f"[{video_timestamp_str}]")
    if label:
        title_parts.append(f"Label: {label}")
    title_parts.append(f"Found: {found_target}")
    if max_val is not None:
        title_parts.append(f"Confidence: {max_val:.4f}")
    if clicked is not None:
        title_parts.append(f"Clicked: {clicked}")
    fig.suptitle(" | ".join(title_parts), fontsize=13, fontweight="bold")

    # [0, 0] Full Frame
    axs[0, 0].imshow(_to_rgb(annotated_full))
    axs[0, 0].set_title("Full Frame (Blue: ROI, Red: Target)")

    # [0, 1] Zoomed ROI
    axs[0, 1].imshow(_to_rgb(roi_crop))
    axs[0, 1].set_title(f"ROI Zoom [({x1}, {y1}) -> ({x2}, {y2})]")

    # [0, 2] matchTemplate Score Map
    im = axs[0, 2].imshow(match, cmap="viridis")
    axs[0, 2].set_title("matchTemplate Score Map")
    fig.colorbar(im, ax=axs[0, 2], fraction=0.046, pad=0.04)

    # [0, 3] Diagnostic Overlay
    if found_target and top_left is not None and bottom_right is not None:
        detected_crop = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        
        if template.ndim == 3:
            if template.shape[2] == 4:
                gray_temp = cv2.cvtColor(template, cv2.COLOR_BGRA2GRAY)
            else:
                gray_temp = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            gray_temp = template.copy()
            
        edges_temp = cv2.Canny(gray_temp, 50, 150)
        
        edge_overlay = _to_rgb(detected_crop).copy()
        edge_overlay[edges_temp > 0] = [255, 0, 255]
        
        axs[0, 3].imshow(edge_overlay)
        axs[0, 3].set_title("Template Edges Overlay")
    else:
        axs[0, 3].axis("off")
        axs[0, 3].set_title("No Scale Diagnostic Needed")

    def save_and_continue(event):
        if template_json_path and template_name:
            try:
                with open(template_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if template_name in data:
                    data[template_name]["shadow_dy"] = int(shadow_slider.val)
                    data[template_name]["shadow_dw"] = int(dw_slider.val)
                    data[template_name]["shadow_dh"] = int(dh_slider.val)
                    data[template_name]["shadow_threshold"] = round(float(thresh_slider.val), 2)
                    data[template_name]["adjusted"] = True
                    with open(template_json_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                    print(
                        f"Saved dy={int(shadow_slider.val)} dw={int(dw_slider.val)} "
                        f"dh={int(dh_slider.val)} thr={float(thresh_slider.val):.2f} "
                        f"and marked adjusted for {template_name}"
                    )
            except Exception as e:
                print("Failed to save:", e)
        plt.close(fig)

    # [1, 0] Template Image
    axs[1, 0].imshow(_to_rgb(template))
    axs[1, 0].set_title(f"Template ({w_inp}x{h_inp})")

    # [1, 1] Detected Element Crop
    if found_target and top_left is not None and bottom_right is not None:
        pad = 6
        cy1 = max(0, top_left[1] - pad)
        cy2 = min(img.shape[0], bottom_right[1] + pad)
        cx1 = max(0, top_left[0] - pad)
        cx2 = min(img.shape[1], bottom_right[0] + pad)
        elem_crop = img[cy1:cy2, cx1:cx2]
        axs[1, 1].imshow(_to_rgb(elem_crop))
        axs[1, 1].set_title(f"Target Crop at {top_left}")
    else:
        axs[1, 1].text(
            0.5,
            0.5,
            "Target Not Found",
            ha="center",
            va="center",
            fontsize=11,
            color="gray",
        )
        axs[1, 1].set_title("Target Crop")

    # [1, 2] Shadow Region Crop & [1, 3] Edge Profile
    if found_target and top_left is not None:
        if shadow_region is not None:
            # Use provided shadow_region (PixelROI)
            abs_x_start = shadow_region.x_min
            abs_x_end = shadow_region.x_max
            abs_y_start = shadow_region.y_min
            abs_y_end = shadow_region.y_max
        else:
            # Fallback to dynamic logic using the slider
            # Will be overridden in the update function anyway, but we do initial setup here
            pass

        # Adjust layout for the sliders on the right
        plt.subplots_adjust(left=0.05, right=0.79)
        ax_dh_slider = plt.axes([0.82, 0.35, 0.02, 0.3])
        ax_dw_slider = plt.axes([0.86, 0.35, 0.02, 0.3])
        ax_dy_slider = plt.axes([0.90, 0.35, 0.02, 0.3])
        ax_thr_slider = plt.axes([0.94, 0.35, 0.02, 0.3])
        ax_button = plt.axes([0.85, 0.70, 0.08, 0.04])
        
        dh_slider = Slider(
            ax=ax_dh_slider, 
            label='Shadow dh', 
            valmin=-50, 
            valmax=50, 
            valinit=current_shadow_dh, 
            valfmt='%d',
            orientation='vertical'
        )
        dw_slider = Slider(
            ax=ax_dw_slider, 
            label='Shadow dw', 
            valmin=-100, 
            valmax=100, 
            valinit=current_shadow_dw, 
            valfmt='%d',
            orientation='vertical'
        )
        shadow_slider = Slider(
            ax=ax_dy_slider, 
            label='Shadow dy', 
            valmin=-100, 
            valmax=100, 
            valinit=current_shadow_dy, 
            valfmt='%d',
            orientation='vertical'
        )
        thresh_max = max(150.0, float(shadow_threshold) + 20.0)
        thresh_slider = Slider(
            ax=ax_thr_slider,
            label='Thr',
            valmin=5,
            valmax=thresh_max,
            valinit=float(shadow_threshold),
            valfmt='%d',
            orientation='vertical',
        )
        save_button = Button(ax_button, 'Save')
        save_button.on_clicked(save_and_continue)

        def update_shadow(val=None):
            dy = int(shadow_slider.val)
            dw = int(dw_slider.val)
            dh = int(dh_slider.val)
            cur_thresh = float(thresh_slider.val)
            # Re-calculate shadow region
            abs_x_start = top_left[0]
            abs_x_end = top_left[0] + w_inp + dw
            abs_y_start = top_left[1] + h_inp + dy
            abs_y_end = top_left[1] + h_inp + shadow_height + dy + dh
            
            axs[1, 2].clear()
            axs[1, 3].clear()

            if (
                abs_x_end > abs_x_start
                and abs_y_end > abs_y_start
                and abs_y_end <= img.shape[0]
                and abs_x_end <= img.shape[1]
                and abs_y_start >= 0
                and abs_x_start >= 0
            ):
                shadow_img_vis = img.copy()
                cv2.rectangle(
                    shadow_img_vis,
                    (abs_x_start, abs_y_start),
                    (abs_x_end, abs_y_end),
                    color=(0, 0, 255),
                    thickness=1,
                )
                pad_x, pad_y = 15, 15
                sy1 = max(0, abs_y_start - pad_y)
                sy2 = min(img.shape[0], abs_y_end + pad_y)
                sx1 = max(0, abs_x_start - pad_x)
                sx2 = min(img.shape[1], abs_x_end + pad_x)
                shadow_crop_padded = shadow_img_vis[sy1:sy2, sx1:sx2]
                axs[1, 2].imshow(_to_rgb(shadow_crop_padded))
                axs[1, 2].set_title(f"Shadow Region (dy={dy}, dw={dw})")

                # Plot Edge Profile
                from shadow_evaluator import detect_horizontal_edges
                from shadow_evaluator import analyze_shadow_presence
                
                actual_shadow_crop = img[abs_y_start:abs_y_end, abs_x_start:abs_x_end]
                row_sums = detect_horizontal_edges(actual_shadow_crop)
                
                has_shadow = analyze_shadow_presence(
                    actual_shadow_crop, threshold=cur_thresh
                )
                clicked = not has_shadow
                axs[1, 2].set_title(f"Shadow Region (Clicked={clicked}, thr={cur_thresh:.0f})")

                if len(row_sums) > 0:
                    y_positions = np.arange(len(row_sums))
                    axs[1, 3].plot(row_sums, y_positions, color="blue", linewidth=2)
                    axs[1, 3].invert_yaxis()
                    axs[1, 3].set_title("Shadow Y-Edge Profile (Row Sums)")
                    axs[1, 3].set_xlabel("Gradient Sum")
                    axs[1, 3].set_ylabel("Row Index")
                    axs[1, 3].axvline(
                        x=cur_thresh,
                        color="red",
                        linestyle="--",
                        label=f"Threshold ({cur_thresh:.0f})",
                    )
                    axs[1, 3].legend()
                else:
                    axs[1, 3].text(0.5, 0.5, "No edge data", ha="center", va="center", color="gray")
                    axs[1, 3].set_title("Shadow Edge Profile")
            else:
                for ax in (axs[1, 2], axs[1, 3]):
                    ax.text(0.5, 0.5, "Shadow bounds invalid", ha="center", va="center", color="gray")
                    ax.set_title("Shadow Analysis")
                    
            fig.canvas.draw_idle()

        dh_slider.on_changed(update_shadow)
        dw_slider.on_changed(update_shadow)
        shadow_slider.on_changed(update_shadow)
        thresh_slider.on_changed(update_shadow)
        update_shadow()
    else:
        for ax in (axs[1, 2], axs[1, 3]):
            ax.text(0.5, 0.5, "Target Not Found", ha="center", va="center", fontsize=11, color="gray")
            ax.set_title("Shadow Analysis")

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Maximize window to prevent drifting and ensure it's large enough
    try:
        manager = plt.get_current_fig_manager()
        backend = plt.get_backend()
        if backend == "TkAgg":
            manager.window.state("zoomed")
        elif backend == "wxAgg":
            manager.frame.Maximize(True)
        elif backend in ["Qt5Agg", "QtAgg", "Qt4Agg"]:
            manager.window.showMaximized()
    except Exception:  # noqa: BLE001, S110
        pass

    plt.show()
