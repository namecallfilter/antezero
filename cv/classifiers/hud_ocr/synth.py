"""Synthetic Balatro HUD text crop generator using m6x11plus.ttf and VOD augmentations."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from .config import (
        DEFAULT_FONTS_DIR,
        DEFAULT_FONT_PATH,
        HUD_ROI_FIELDS,
        INPUT_HEIGHT,
        INPUT_WIDTH,
    )
except ImportError:
    from config import (  # type: ignore[no-redef]
        DEFAULT_FONTS_DIR,
        DEFAULT_FONT_PATH,
        HUD_ROI_FIELDS,
        INPUT_HEIGHT,
        INPUT_WIDTH,
    )


# Balatro UI background panel colors (RGB)
PANEL_COLORS: tuple[tuple[int, int, int], ...] = (
    (30, 43, 45),    # Dark slate HUD panel
    (44, 54, 57),    # Mid slate HUD panel
    (55, 66, 68),    # Lighter slate
    (25, 34, 38),    # Deep shadow panel
    (45, 58, 62),    # Score box slate
    (184, 59, 53),   # Red button / discard badge
    (45, 109, 181),  # Blue hands badge
    (66, 158, 107),  # Green reroll button
    (194, 134, 43),  # Gold / cash accent
)

# Balatro HUD text colors (RGB)
TEXT_COLORS: dict[str, tuple[tuple[int, int, int], ...]] = {
    "hud_round_score": ((255, 255, 255), (245, 245, 245), (255, 105, 97)),
    "hud_money": ((243, 185, 88), (255, 200, 95), (255, 90, 80)),
    "hud_hands": ((77, 166, 255), (105, 185, 255), (255, 255, 255)),
    "hud_discards": ((255, 76, 64), (255, 105, 95), (255, 255, 255)),
    "hud_ante": ((243, 185, 88), (255, 255, 255)),
    "hud_round": ((243, 185, 88), (255, 255, 255)),
    "hud_reroll_cost": ((255, 255, 255), (243, 185, 88)),
    "hud_blind_target": ((255, 76, 64), (255, 255, 255)),
}

# Words near HUD numbers that can partially intrude if OBS/streamer framing drifts by a few pixels.
HUD_LABEL_DISTRACTORS: tuple[str, ...] = (
    "Hands",
    "Discards",
    "Ante",
    "Round",
    "Reroll",
    "Round score",
    "Score at least",
    "Play",
)


def resolve_font_path(font_path: str | Path | None = None) -> Path | None:
    """Find m6x11plus.ttf in explicit path, local fonts dir, or balatro_source."""
    candidates: list[Path] = []
    if font_path is not None:
        candidates.append(Path(font_path))
    candidates.append(DEFAULT_FONT_PATH)
    candidates.append(DEFAULT_FONTS_DIR / "m6x11plus.ttf")
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def sample_hud_text(rng: random.Random, field_name: str | None = None) -> tuple[str, str]:
    """Sample a realistic `(field_name, text)` pair for Balatro HUD counters."""
    if field_name is None:
        field_name = rng.choices(
            HUD_ROI_FIELDS,
            weights=(0.28, 0.22, 0.10, 0.10, 0.08, 0.07, 0.08, 0.07),
            k=1,
        )[0]

    if field_name in ("hud_round_score", "hud_blind_target"):
        kind = rng.random()
        if kind < 0.08:
            val = 0
            text = "0"
        elif kind < 0.35:
            val = rng.randint(1, 999)
            text = str(val)
        elif kind < 0.80:
            val = rng.randint(1_000, 99_999_999)
            text = f"{val:,}" if rng.random() < 0.92 else str(val)
        else:
            mantissa = rng.uniform(1.0, 9.999)
            dec = rng.choice((1, 2, 3))
            exp = rng.randint(11, 30)
            plus = "+" if rng.random() < 0.35 else ""
            text = f"{mantissa:.{dec}f}e{plus}{exp}"
        return field_name, text

    if field_name == "hud_money":
        if rng.random() < 0.18:
            val = rng.randint(1, 35)
            return field_name, f"-{val}"
        r_m = rng.random()
        if r_m < 0.45:
            val = rng.randint(0, 99)
        elif r_m < 0.85:
            val = rng.randint(100, 999)
        else:
            val = rng.randint(1000, 2999)
        return field_name, str(val)

    if field_name in ("hud_hands", "hud_discards"):
        val = rng.choices(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, rng.randint(9, 25)],
            weights=(1, 3, 3, 4, 4, 3, 2, 1, 1, 1),
            k=1,
        )[0]
        return field_name, str(val)

    if field_name == "hud_ante":
        ante = rng.randint(1, 16) if rng.random() < 0.9 else rng.randint(17, 39)
        return field_name, str(ante)

    if field_name == "hud_round":
        rnd = rng.randint(1, 35)
        return field_name, str(rnd)

    if field_name == "hud_reroll_cost":
        cost = rng.choices(
            [0, rng.randint(1, 12), rng.randint(13, 60)],
            weights=(1, 7, 2),
            k=1,
        )[0]
        return field_name, str(cost)

    return field_name, str(rng.randint(0, 999))


def resize_and_pad_crop(
    crop_rgb: np.ndarray,
    target_h: int = INPUT_HEIGHT,
    target_w: int = INPUT_WIDTH,
    pad_color: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Aspect-preserving resize into `(target_h, target_w, 3)` with edge padding."""
    if crop_rgb.ndim != 3 or crop_rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB crop, got shape {crop_rgb.shape}")

    h, w = crop_rgb.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid empty crop shape {crop_rgb.shape}")

    scale = min(target_w / float(w), target_h / float(h))
    new_w = max(1, min(target_w, int(round(w * scale))))
    new_h = max(1, min(target_h, int(round(h * scale))))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(crop_rgb, (new_w, new_h), interpolation=interp)

    if pad_color is None:
        # Use median border pixel so padding blends naturally with the HUD panel.
        border_pixels = np.concatenate(
            [resized[0, :, :], resized[-1, :, :], resized[:, 0, :], resized[:, -1, :]],
            axis=0,
        )
        pad_color_arr = np.median(border_pixels, axis=0).astype(np.uint8)
    else:
        pad_color_arr = np.array(pad_color, dtype=np.uint8)

    canvas = np.empty((target_h, target_w, 3), dtype=np.uint8)
    canvas[:, :] = pad_color_arr

    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def apply_vod_crop_augmentations(
    img_rgb: np.ndarray,
    rng: random.Random,
    bg_color: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Apply realistic OBS/YouTube VOD degradations (jitter, scaling blur, CRT scanlines, JPEG)."""
    h, w = img_rgb.shape[:2]
    if bg_color is None:
        bg_color = tuple(int(v) for v in np.median(img_rgb[0, :, :], axis=0))

    # 1. Mild shift / scale / rotation jitter (simulating Balatro juice/tilt & +/-3px OBS drift)
    if rng.random() < 0.65:
        angle = rng.uniform(-1.6, 1.6)
        scale = rng.uniform(0.94, 1.06)
        dx = rng.uniform(-3.0, 3.0)
        dy = rng.uniform(-3.0, 3.0)
        mat = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, scale)
        mat[0, 2] += dx
        mat[1, 2] += dy
        img_rgb = cv2.warpAffine(
            img_rgb,
            mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=bg_color,
        )

    # 2. Resolution downscale/upscale (simulating 720p/480p stream scaling)
    if rng.random() < 0.65:
        down_scale = rng.uniform(0.45, 0.88)
        dw = max(8, int(round(w * down_scale)))
        dh = max(6, int(round(h * down_scale)))
        down = cv2.resize(img_rgb, (dw, dh), interpolation=cv2.INTER_AREA)
        up_interp = rng.choice((cv2.INTER_LINEAR, cv2.INTER_CUBIC))
        img_rgb = cv2.resize(down, (w, h), interpolation=up_interp)

    # 3. Simulated CRT scanlines
    if rng.random() < 0.30:
        strength = rng.uniform(0.08, 0.25)
        period = rng.choice((2, 3))
        mask = np.ones((h, 1, 1), dtype=np.float32)
        mask[0::period, :, :] = 1.0 - strength
        img_rgb = np.clip(img_rgb.astype(np.float32) * mask, 0, 255).astype(np.uint8)

    # 4. Mild Gaussian blur
    if rng.random() < 0.25:
        img_rgb = cv2.GaussianBlur(img_rgb, (3, 3), sigmaX=rng.uniform(0.3, 0.8))

    # 5. Brightness / contrast / noise
    if rng.random() < 0.70:
        alpha = rng.uniform(0.85, 1.15)
        beta = rng.uniform(-14.0, 14.0)
        img_rgb = np.clip(img_rgb.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    if rng.random() < 0.45:
        sigma = rng.uniform(1.5, 6.5)
        noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0.0, sigma, size=img_rgb.shape)
        img_rgb = np.clip(img_rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 6. JPEG / H.264 compression artifacts
    if rng.random() < 0.70:
        quality = rng.randint(30, 88)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is not None:
                img_rgb = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)

    return img_rgb


# Colors used by Balatro attention_text / delta popups (+$5, -1, etc.)
DELTA_POPUP_COLORS: tuple[tuple[int, int, int], ...] = (
    (243, 185, 88),  # G.C.MONEY
    (235, 75, 65),   # G.C.RED
    (55, 130, 220),  # G.C.BLUE
    (66, 158, 107),  # G.C.GREEN
)


def corrupt_hud_crop_to_empty(
    img_rgb: np.ndarray,
    rng: random.Random,
    bg_color: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Apply severe cutoff/transition/occlusion corruption so the ground-truth text should be `""`."""
    h, w = img_rgb.shape[:2]
    if h <= 1 or w <= 1:
        return img_rgb.copy()

    if bg_color is None:
        border = np.concatenate([img_rgb[0, :, :], img_rgb[-1, :, :]], axis=0)
        bg_color = tuple(int(v) for v in np.median(border, axis=0))

    mode = rng.choice(("cutoff_shift", "delta_popup", "pop_transition"))
    out = img_rgb.copy()

    if mode == "cutoff_shift":
        if rng.random() < 0.72:
            dy = rng.choice((-1, 1)) * max(1, int(round(h * rng.uniform(0.45, 0.72))))
            dx = rng.randint(-max(1, w // 12), max(1, w // 12))
        else:
            dy = rng.randint(-max(1, h // 8), max(1, h // 8))
            dx = rng.choice((-1, 1)) * max(1, int(round(w * rng.uniform(0.48, 0.75))))
        mat = np.float32([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]])
        return cv2.warpAffine(
            out,
            mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=bg_color,
        )

    if mode == "delta_popup":
        pil = Image.fromarray(out)
        draw = ImageDraw.Draw(pil)
        pop_col = rng.choice(DELTA_POPUP_COLORS)
        pop_w = max(8, int(round(w * rng.uniform(0.50, 0.85))))
        pop_h = max(8, int(round(h * rng.uniform(0.55, 0.90))))
        cx = w // 2 + rng.randint(-max(1, w // 10), max(1, w // 10))
        cy = h // 2 + rng.randint(-max(1, h // 10), max(1, h // 10))
        x0 = max(0, cx - pop_w // 2)
        y0 = max(0, cy - pop_h // 2)
        x1 = min(w - 1, cx + pop_w // 2)
        y1 = min(h - 1, cy + pop_h // 2)
        draw.rectangle([x0, y0, x1, y1], fill=pop_col, outline=(25, 25, 25), width=1)
        # Draw a few horizontal/vertical bar strokes simulating large +/- delta popup glyphs
        stroke_col = (255, 255, 255)
        bw = max(2, (x1 - x0) // 6)
        bh = max(2, (y1 - y0) // 5)
        mx = (x0 + x1) // 2
        my = (y0 + y1) // 2
        draw.rectangle([max(x0 + 2, mx - bw), max(y0 + 2, my - bh), min(x1 - 2, mx + bw), min(y1 - 2, my + bh)], fill=stroke_col)
        arr = np.array(pil, dtype=np.uint8)
        angle = rng.uniform(-7.0, 7.0)
        mat = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, 1.0)
        return cv2.warpAffine(arr, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # pop_transition: mask central 55-85% band and blend a squashed/shifted ghost of previous text
    band_w = max(4, int(round(w * rng.uniform(0.55, 0.85))))
    band_h = max(4, int(round(h * rng.uniform(0.55, 0.88))))
    x0 = max(0, (w - band_w) // 2 + rng.randint(-max(1, w // 10), max(1, w // 10)))
    y0 = max(0, (h - band_h) // 2 + rng.randint(-max(1, h // 10), max(1, h // 10)))
    x1 = min(w, x0 + band_w)
    y1 = min(h, y0 + band_h)
    masked = out.copy()
    masked[y0:y1, x0:x1] = np.array(bg_color, dtype=np.uint8)
    ghost = cv2.resize(out, (w, max(4, int(round(h * rng.uniform(0.35, 0.65))))), interpolation=cv2.INTER_AREA)
    ghost = cv2.resize(ghost, (w, h), interpolation=cv2.INTER_LINEAR)
    shift_y = rng.choice((-1, 1)) * max(1, h // 4)
    mat = np.float32([[1.0, 0.0, 0.0], [0.0, 1.0, float(shift_y)]])
    ghost = cv2.warpAffine(ghost, mat, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color)
    alpha = rng.uniform(0.25, 0.50)
    blended = (1.0 - alpha) * masked.astype(np.float32) + alpha * ghost.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


class BalatroHudSynthesizer:
    """Renders synthetic Balatro HUD crops with realistic VOD degradations."""

    def __init__(
        self,
        font_path: str | Path | None = None,
        target_height: int = INPUT_HEIGHT,
        target_width: int = INPUT_WIDTH,
        font_sizes: Sequence[int] = (24, 28, 32, 36, 40, 44),
    ) -> None:
        self.target_height = int(target_height)
        self.target_width = int(target_width)
        self.font_path = resolve_font_path(font_path)
        self.fonts: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}
        for size in font_sizes:
            if self.font_path is not None:
                self.fonts[int(size)] = ImageFont.truetype(str(self.font_path), int(size))
            else:
                self.fonts[int(size)] = ImageFont.load_default()
        self.font_sizes = tuple(int(s) for s in font_sizes)

    def render_sample(
        self,
        rng: random.Random,
        field_name: str | None = None,
        text_override: str | None = None,
        augment: bool = True,
    ) -> tuple[np.ndarray, str, str]:
        """Generate `(rgb_uint8_HxWx3, target_text, field_name)`."""
        chosen_field, target_text = sample_hud_text(rng, field_name=field_name)
        if text_override is not None:
            target_text = text_override

        is_empty_sample = augment and text_override is None and rng.random() < 0.12
        if is_empty_sample:
            target_text = ""

        if chosen_field == "hud_reroll_cost" and not is_empty_sample:
            base_bg = (66, 158, 107)
        else:
            base_bg = rng.choice(PANEL_COLORS)
        bg = tuple(max(0, min(255, c + rng.randint(-12, 12))) for c in base_bg)
        palette = TEXT_COLORS.get(chosen_field, ((255, 255, 255),))
        fg_base = rng.choice(palette)
        fg = tuple(max(0, min(255, c + rng.randint(-10, 10))) for c in fg_base)

        # Visually render prefixes/suffixes ("$", "/ 8") even though target_text stays numeric-only.
        if is_empty_sample:
            display_text = rng.choice(("", "SHOP", "Reroll", "Blind", "Play Hand"))
        elif chosen_field == "hud_money":
            display_text = f"${target_text}"
        elif chosen_field == "hud_reroll_cost":
            display_text = f"${target_text}"
        else:
            display_text = target_text

        font_size = rng.choice(self.font_sizes)
        font = self.fonts[font_size]

        # Measure text on a scratch draw surface
        dummy = Image.new("RGB", (8, 8))
        d_draw = ImageDraw.Draw(dummy)
        if display_text:
            bbox = d_draw.textbbox((0, 0), display_text, font=font)
            tw = max(4, bbox[2] - bbox[0])
            th = max(8, bbox[3] - bbox[1])
            bbox_y0 = bbox[1]
        else:
            tw, th, bbox_y0 = 24, font_size, 0

        has_chip = (not is_empty_sample) and chosen_field in ("hud_round_score", "hud_blind_target")
        chip_space = int(th * 1.15) if has_chip else 0
        ante_suffix_space = int(th * 1.05) if (not is_empty_sample and chosen_field == "hud_ante") else 0

        pad_x = rng.randint(8, 28)
        pad_y = rng.randint(4, 14)
        canvas_w = max(tw + chip_space + ante_suffix_space + pad_x * 2, int(th * 2.2))
        canvas_h = th + pad_y * 2

        img = Image.new("RGB", (canvas_w, canvas_h), bg)
        draw = ImageDraw.Draw(img)

        # Optional subtle rounded inner panel or border edge
        if rng.random() < 0.45:
            border_col = tuple(max(0, min(255, c + rng.randint(-35, 35))) for c in bg)
            draw.rectangle([0, 0, canvas_w - 1, canvas_h - 1], outline=border_col, width=rng.randint(1, 2))

        total_w = tw + chip_space + ante_suffix_space
        align = rng.choice(("center", "left", "right"))
        if align == "left":
            start_x = rng.randint(4, max(4, pad_x))
        elif align == "right":
            start_x = max(2, canvas_w - total_w - rng.randint(4, max(4, pad_x)))
        else:
            start_x = (canvas_w - total_w) // 2 + rng.randint(-4, 4)

        tx = start_x + chip_space
        ty = (canvas_h - th) // 2 - bbox_y0 + rng.randint(-2, 2)

        # Balatro pixel drop shadow
        shadow_dx = rng.choice((1, 2, 3))
        shadow_dy = rng.choice((1, 2, 3))
        shadow_col = (
            max(0, bg[0] // 3),
            max(0, bg[1] // 3),
            max(0, bg[2] // 3),
        )

        # Draw circular poker chip art for round_score / blind_target
        if has_chip:
            r_chip = max(6, int(th * 0.46))
            cx = start_x + r_chip
            cy = canvas_h // 2
            chip_outer = rng.choice(((230, 235, 245), (225, 85, 85), (75, 150, 230), (195, 145, 240)))
            chip_inner = tuple(max(30, c - 60) for c in chip_outer)
            draw.ellipse([cx - r_chip + 1, cy - r_chip + 1, cx + r_chip + 1, cy + r_chip + 1], fill=shadow_col)
            draw.ellipse([cx - r_chip, cy - r_chip, cx + r_chip, cy + r_chip], fill=chip_outer)
            r_in = max(3, int(r_chip * 0.58))
            draw.ellipse([cx - r_in, cy - r_in, cx + r_in, cy + r_in], fill=chip_inner, outline=(255, 255, 255), width=1)

        if display_text:
            draw.text((tx + shadow_dx, ty + shadow_dy), display_text, font=font, fill=shadow_col)
            draw.text((tx, ty), display_text, font=font, fill=fg)

        # Draw Balatro's smaller "/ 8" suffix next to the big Ante number
        if not is_empty_sample and chosen_field == "hud_ante":
            small_font = self.fonts[self.font_sizes[0]]
            sx = tx + tw + rng.randint(3, 7)
            sy = ty + max(2, int(th * 0.25))
            draw.text((sx + 1, sy + 1), "/ 8", font=small_font, fill=shadow_col)
            draw.text((sx, sy), "/ 8", font=small_font, fill=(245, 245, 245))

        # Simulate 2-7px OBS/streamer vertical drift where adjacent label text ("Hands", "Reroll", etc.)
        # partially clips into the top or bottom border of the ROI crop.
        if augment and not is_empty_sample and rng.random() < 0.45:
            dist_word = rng.choice(HUD_LABEL_DISTRACTORS)
            dist_font = self.fonts[rng.choice(self.font_sizes[:3])]
            dist_bbox = d_draw.textbbox((0, 0), dist_word, font=dist_font)
            dist_h = max(8, dist_bbox[3] - dist_bbox[1])
            intrude_px = rng.randint(2, min(7, max(2, pad_y + 2)))
            dist_x = rng.randint(0, max(0, canvas_w // 3))
            if rng.random() < 0.7:
                dist_y = intrude_px - dist_h - dist_bbox[1]
            else:
                dist_y = canvas_h - intrude_px - dist_bbox[1]
            dist_col = rng.choice(((235, 235, 235), (200, 210, 215), (243, 185, 88)))
            draw.text((dist_x + 1, dist_y + 1), dist_word, font=dist_font, fill=shadow_col)
            draw.text((dist_x, dist_y), dist_word, font=dist_font, fill=dist_col)

        arr = np.array(img, dtype=np.uint8)
        if augment and text_override is None and not is_empty_sample and rng.random() < 0.12:
            arr = corrupt_hud_crop_to_empty(arr, rng, bg_color=bg)
            target_text = ""

        if augment:
            arr = self._apply_vod_augmentations(arr, rng, bg)

        out = resize_and_pad_crop(arr, target_h=self.target_height, target_w=self.target_width, pad_color=bg)
        return out, target_text, chosen_field

    def _apply_vod_augmentations(
        self,
        img_rgb: np.ndarray,
        rng: random.Random,
        bg_color: tuple[int, int, int],
    ) -> np.ndarray:
        return apply_vod_crop_augmentations(img_rgb, rng, bg_color=bg_color)
