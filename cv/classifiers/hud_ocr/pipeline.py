"""Production inference pipeline and value parser for Balatro HUD OCR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import torch

try:
    from .config import (
        DEFAULT_OCR_CONF_THRESHOLD,
        DEFAULT_WEIGHT_CANDIDATES,
        FIELD_ALLOWED_CHARS,
        INPUT_HEIGHT,
        INPUT_WIDTH,
        NORM_MEAN,
        NORM_STD,
        NUM_CLASSES,
        REROLL_MAX_RGB_DISTANCE,
        REROLL_MIN_GREEN_DOMINANCE,
        REROLL_TARGET_RGB,
    )
    from .model import HudOcrCRNN, ctc_greedy_decode
    from .synth import resize_and_pad_crop
except ImportError:
    from config import (  # type: ignore[no-redef]
        DEFAULT_OCR_CONF_THRESHOLD,
        DEFAULT_WEIGHT_CANDIDATES,
        FIELD_ALLOWED_CHARS,
        INPUT_HEIGHT,
        INPUT_WIDTH,
        NORM_MEAN,
        NORM_STD,
        NUM_CLASSES,
        REROLL_MAX_RGB_DISTANCE,
        REROLL_MIN_GREEN_DOMINANCE,
        REROLL_TARGET_RGB,
    )
    from model import HudOcrCRNN, ctc_greedy_decode  # type: ignore[no-redef]
    from synth import resize_and_pad_crop  # type: ignore[no-redef]


_INT_RE = re.compile(r"^[+-]?\d+$")
_SCI_RE = re.compile(r"^([+-]?\d+(?:\.\d+)?)[eE]\+?(\d+)$")
_SLASH_RE = re.compile(r"^(\d+)/(\d+)$")


@dataclass(slots=True)
class HudOcrResult:
    """Decoded HUD OCR prediction for a single ROI crop."""

    text: str
    parsed_value: int | float | None
    confidence: float
    field_name: str = ""


def resolve_weights_path(weights_path: str | Path | None = None) -> Path:
    """Resolve HUD OCR weights from explicit path, `prod_weights/hud_ocr.pt`, or training weights."""
    if weights_path is not None:
        p = Path(weights_path)
        if not p.exists():
            raise FileNotFoundError(f"HUD OCR weights not found at: {p}")
        return p

    for candidate in DEFAULT_WEIGHT_CANDIDATES:
        if candidate.exists():
            return candidate

    searched = ", ".join(str(c) for c in DEFAULT_WEIGHT_CANDIDATES)
    raise FileNotFoundError(f"Could not find HUD OCR weights. Searched: {searched}")


def parse_hud_value(field_name: str, raw_text: str) -> int | float | None:
    """Parse raw CTC text according to the expected Balatro HUD field format."""
    cleaned = raw_text.strip().replace(",", "").replace("$", "")
    if not cleaned:
        return None

    if field_name == "hud_ante":
        m_slash = _SLASH_RE.match(cleaned)
        if m_slash:
            return int(m_slash.group(1))

    if _INT_RE.match(cleaned):
        return int(cleaned)

    m_sci = _SCI_RE.match(cleaned)
    if m_sci:
        try:
            val = float(f"{m_sci.group(1)}e{m_sci.group(2)}")
            return int(val) if val.is_integer() and val < 10**15 else val
        except ValueError:
            return None

    return None


def is_reroll_button_color(
    crop_rgb: np.ndarray,
    *,
    target_rgb: tuple[int, int, int] = REROLL_TARGET_RGB,
    max_rgb_distance: float = REROLL_MAX_RGB_DISTANCE,
    min_green_dominance: float = REROLL_MIN_GREEN_DOMINANCE,
) -> bool:
    """Return True if the mean RGB of `crop_rgb` is within tolerance of Balatro's green reroll button."""
    if crop_rgb.size == 0:
        return False
    mean_rgb = crop_rgb.reshape(-1, crop_rgb.shape[-1]).mean(axis=0)
    r, g, b = float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2])
    g_dom = g - max(r, b)
    dist = float(
        ((r - target_rgb[0]) ** 2 + (g - target_rgb[1]) ** 2 + (b - target_rgb[2]) ** 2) ** 0.5
    )
    return dist <= max_rgb_distance and g_dom >= min_green_dominance


def preprocess_crops_to_tensor(crops_rgb: Sequence[np.ndarray]) -> torch.Tensor:
    """Resize/pad a list of RGB uint8 crops into a `(N, 3, H, W)` normalized tensor."""
    mean = np.array(NORM_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(NORM_STD, dtype=np.float32).reshape(1, 1, 3)
    batch = np.empty((len(crops_rgb), 3, INPUT_HEIGHT, INPUT_WIDTH), dtype=np.float32)
    for i, crop in enumerate(crops_rgb):
        resized = resize_and_pad_crop(crop, target_h=INPUT_HEIGHT, target_w=INPUT_WIDTH)
        normed = (resized.astype(np.float32) / 255.0 - mean) / std
        batch[i] = normed.transpose(2, 0, 1)
    return torch.from_numpy(batch)


class HudOcrPipeline:
    """High-throughput batched GPU/CPU inference wrapper for `HudOcrCRNN`."""

    def __init__(
        self,
        weights_path: str | Path | None = None,
        device: str | None = None,
        batch_size: int = 128,
        conf_threshold: float = DEFAULT_OCR_CONF_THRESHOLD,
        check_reroll_color: bool = True,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = max(1, int(batch_size))
        self.conf_threshold = float(conf_threshold)
        self.check_reroll_color = bool(check_reroll_color)
        self.weights_path = resolve_weights_path(weights_path)

        ckpt = torch.load(str(self.weights_path), map_location=self.device, weights_only=False)
        model_cfg = ckpt.get("model_config", {}) if isinstance(ckpt, dict) else {}
        self.model = HudOcrCRNN(
            num_classes=int(model_cfg.get("num_classes", NUM_CLASSES)),
            rnn_hidden=int(model_cfg.get("rnn_hidden", 128)),
            rnn_layers=int(model_cfg.get("rnn_layers", 1)),
            dropout=0.0,
        ).to(self.device)

        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.inference_mode()
    def predict_crops(
        self,
        crops_rgb: Sequence[np.ndarray],
        field_names: Sequence[str] | None = None,
        conf_threshold: float | None = None,
        check_reroll_color: bool | None = None,
    ) -> list[HudOcrResult]:
        """Run batched OCR inference on RGB uint8 crops."""
        if not crops_rgb:
            return []

        if field_names is not None and len(field_names) != len(crops_rgb):
            raise ValueError(
                f"field_names length ({len(field_names)}) must match crops_rgb length ({len(crops_rgb)})"
            )

        thresh = self.conf_threshold if conf_threshold is None else float(conf_threshold)
        use_reroll_check = self.check_reroll_color if check_reroll_color is None else bool(check_reroll_color)

        results: list[HudOcrResult] = []
        for start in range(0, len(crops_rgb), self.batch_size):
            chunk = crops_rgb[start : start + self.batch_size]
            chunk_fields = (
                field_names[start : start + len(chunk)]
                if field_names is not None
                else [""] * len(chunk)
            )

            chunk_results: list[HudOcrResult | None] = [None] * len(chunk)
            active_indices: list[int] = []
            active_crops: list[np.ndarray] = []
            active_allowed: list[str | None] = []

            for offset, (crop, fname) in enumerate(zip(chunk, chunk_fields)):
                if use_reroll_check and fname == "hud_reroll_cost" and not is_reroll_button_color(crop):
                    chunk_results[offset] = HudOcrResult(
                        text="",
                        parsed_value=None,
                        confidence=0.0,
                        field_name=fname,
                    )
                else:
                    active_indices.append(offset)
                    active_crops.append(crop)
                    active_allowed.append(FIELD_ALLOWED_CHARS.get(fname))

            if active_crops:
                x = preprocess_crops_to_tensor(active_crops).to(self.device, non_blocking=True)
                log_probs = self.model(x)
                decoded = ctc_greedy_decode(log_probs, allowed_chars_per_sample=active_allowed)
                for offset, (text, conf) in zip(active_indices, decoded):
                    fname = chunk_fields[offset]
                    parsed = parse_hud_value(fname, text) if conf >= thresh else None
                    chunk_results[offset] = HudOcrResult(
                        text=text,
                        parsed_value=parsed,
                        confidence=conf,
                        field_name=fname,
                    )

            results.extend(res for res in chunk_results if res is not None)
        return results

    def predict_crop(
        self,
        crop_rgb: np.ndarray,
        field_name: str = "",
        conf_threshold: float | None = None,
        check_reroll_color: bool | None = None,
    ) -> HudOcrResult:
        """Convenience single-crop OCR prediction."""
        return self.predict_crops(
            [crop_rgb],
            field_names=[field_name],
            conf_threshold=conf_threshold,
            check_reroll_color=check_reroll_color,
        )[0]
