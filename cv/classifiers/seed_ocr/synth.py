"""Synthetic end-of-run Seed pill generator matching Balatro m6x11plus UI + VOD degradations."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from classifiers.hud_ocr.synth import (
    apply_vod_crop_augmentations,
    corrupt_hud_crop_to_empty,
    resize_and_pad_crop,
    resolve_font_path,
)

try:
    from .config import INPUT_HEIGHT, INPUT_WIDTH, SEED_ALPHABET, SEED_LENGTH
except ImportError:
    from config import INPUT_HEIGHT, INPUT_WIDTH, SEED_ALPHABET, SEED_LENGTH  # type: ignore[no-redef]


CONFUSABLE_CHARS: str = "12568BILMSUVZWGC"

_SEED_PILL_COLORS_DARK: tuple[tuple[int, int, int], ...] = (
    (42, 50, 56),   # G.C.BLACK dark slate in round_scores pill
    (36, 44, 49),
    (52, 60, 66),
    (30, 38, 43),
)
_SEED_PILL_COLORS_SEEDED: tuple[tuple[int, int, int], ...] = (
    (225, 76, 68),  # G.C.RED for explicitly seeded runs
    (206, 66, 60),
)
_OUTER_ROW_GREY: tuple[int, int, int] = (148, 160, 168)


def sample_seed_text(rng: random.Random, length: int = SEED_LENGTH) -> str:
    """Samples an 8-char seed from Balatro's alphabet with confusable & repeat coverage."""
    use_confusable = rng.random() < 0.35
    pool = CONFUSABLE_CHARS if use_confusable else SEED_ALPHABET
    chars = [rng.choice(pool) for _ in range(length)]

    # Explicitly inject consecutive identical characters (~15%) for sharp CTC blank transitions
    if length >= 2 and rng.random() < 0.15:
        pos = rng.randint(0, length - 2)
        chars[pos + 1] = chars[pos]
    return "".join(chars)


class BalatroSeedSynthesizer:
    """Renders synthetic `Seed` pill crops with Balatro DynaText float, drop shadow, and VOD noise."""

    def __init__(
        self,
        font_path: str | Path | None = None,
        target_height: int = INPUT_HEIGHT,
        target_width: int = INPUT_WIDTH,
        font_sizes: Sequence[int] = (24, 28, 32, 36),
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
        text_override: str | None = None,
        augment: bool = True,
    ) -> tuple[np.ndarray, str]:
        """Generates `(rgb_uint8_HxWx3, target_seed_or_empty)`."""
        is_empty_negative = augment and text_override is None and rng.random() < 0.10
        target_text = (
            text_override.upper()
            if text_override is not None
            else ("" if is_empty_negative else sample_seed_text(rng))
        )

        is_seeded = rng.random() < 0.10
        base_bg = rng.choice(_SEED_PILL_COLORS_SEEDED if is_seeded else _SEED_PILL_COLORS_DARK)
        bg = tuple(max(0, min(255, c + rng.randint(-10, 10))) for c in base_bg)
        fg = tuple(255 - rng.randint(0, 12) for _ in range(3))

        display_text = (
            rng.choice(("", "Seed", "Copy", "New Run", "Endless Mode"))
            if is_empty_negative
            else target_text
        )

        font_size = rng.choice(self.font_sizes)
        font = self.fonts[font_size]
        dummy = Image.new("RGB", (8, 8))
        d_draw = ImageDraw.Draw(dummy)

        if display_text:
            bboxes = [d_draw.textbbox((0, 0), ch, font=font) for ch in display_text]
            advances = [max(4, int(round(d_draw.textlength(ch, font=font)))) for ch in display_text]
            tw = sum(advances)
            th = max(8, max(b[3] - b[1] for b in bboxes))
        else:
            advances, tw, th = [], 96, font_size

        pad_x = rng.randint(10, 22)
        pad_y = rng.randint(6, 12)
        outer_pad = rng.randint(3, 8)
        canvas_w = max(tw + pad_x * 2, int(th * 4.5))
        canvas_h = th + pad_y * 2

        outer_bg = tuple(max(0, min(255, c + rng.randint(-15, 15))) for c in _OUTER_ROW_GREY)
        img = Image.new("RGB", (canvas_w, canvas_h), outer_bg)
        draw = ImageDraw.Draw(img)

        # Rounded dark/red pill housing matching create_UIBox_round_scores_row
        pill_x0 = outer_pad if rng.random() < 0.85 else 0
        pill_x1 = canvas_w - 1 - (outer_pad if rng.random() < 0.85 else 0)
        pill_y0 = outer_pad // 2
        pill_y1 = canvas_h - 1 - outer_pad // 2
        emboss = tuple(max(0, c - 22) for c in bg)
        draw.rounded_rectangle(
            [pill_x0, pill_y0, pill_x1, pill_y1],
            radius=max(3, th // 5),
            fill=bg,
            outline=emboss,
            width=1,
        )

        shadow_dx = rng.choice((1, 2))
        shadow_dy = rng.choice((1, 2))
        shadow_col = tuple(max(0, c // 3) for c in bg)

        if display_text:
            start_x = (canvas_w - tw) // 2 + (rng.randint(-4, 4) if augment else 0)
            base_y = (canvas_h - th) // 2 + (rng.randint(-2, 2) if augment else 0)
            cx = start_x
            phase0 = rng.uniform(0.0, 6.28)
            for idx, (ch, adv) in enumerate(zip(display_text, advances)):
                # Replicate subtle DynaText sine float (+/-1.5px) across seed letters
                float_dy = int(round(1.5 * np.sin(phase0 + idx * 0.55))) if augment else 0
                bbox_ch = d_draw.textbbox((0, 0), ch, font=font)
                cy = base_y - bbox_ch[1] + float_dy
                draw.text((cx + shadow_dx, cy + shadow_dy), ch, font=font, fill=shadow_col)
                draw.text((cx, cy), ch, font=font, fill=fg)
                cx += adv

        # Occasional left-edge grey "Seed" label bleed if crop shifts left
        if augment and not is_empty_negative and rng.random() < 0.30:
            lbl_font = self.fonts[self.font_sizes[0]]
            draw.text((-rng.randint(8, 18), (canvas_h - th) // 2), "Seed", font=lbl_font, fill=(240, 240, 240))

        arr = np.array(img, dtype=np.uint8)
        if augment and text_override is None and not is_empty_negative and rng.random() < 0.08:
            arr = corrupt_hud_crop_to_empty(arr, rng, bg_color=bg)
            target_text = ""

        if augment:
            arr = apply_vod_crop_augmentations(arr, rng, bg_color=bg)

        out = resize_and_pad_crop(
            arr,
            target_h=self.target_height,
            target_w=self.target_width,
            pad_color=outer_bg if not is_empty_negative else bg,
        )
        return out, target_text
