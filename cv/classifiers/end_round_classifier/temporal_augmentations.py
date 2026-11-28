"""Clip-consistent temporal and spatial augmentations for End-Round recognition.

Unlike single-image augmentations in ``classifiers/augmentations.py`` (which
randomize independently per image), video sequence models require exposure,
tint, compression scale, slight ROI misalignment, and cursor trajectories to
evolve consistently across all ``T`` frames of a clip.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .config import (
        BOUNDARY_SIGMA_FRAMES,
        IGNORE_INDEX,
        STATE_TO_IDX,
    )
except ImportError:
    from config import (  # type: ignore[no-redef]
        BOUNDARY_SIGMA_FRAMES,
        IGNORE_INDEX,
        STATE_TO_IDX,
    )

_REPO_CLASSIFIERS_DIR = Path(__file__).resolve().parent.parent


def _load_shared_cursor_rgba() -> np.ndarray | None:
    """Reuse cursor asset loader from ``classifiers/augmentations.py``."""
    try:
        from classifiers.augmentations import load_cursor_asset  # type: ignore[import-not-found]
    except ImportError:
        parent_str = str(_REPO_CLASSIFIERS_DIR.parent)
        if parent_str not in sys.path:
            sys.path.insert(0, parent_str)
        try:
            from classifiers.augmentations import load_cursor_asset  # type: ignore[import-not-found]
        except ImportError:
            return None
    return load_cursor_asset()


def _recompute_gaussian_boundary(
    num_frames: int,
    end_round_frame: int | None,
    *,
    sigma: float = BOUNDARY_SIGMA_FRAMES,
) -> np.ndarray:
    """Recompute Gaussian target centered at ``end_round_frame``."""
    targets = np.zeros((num_frames,), dtype=np.float32)
    if end_round_frame is None or num_frames <= 0:
        return targets
    idx = np.arange(num_frames, dtype=np.float32)
    dist = idx - float(end_round_frame)
    targets = np.exp(-0.5 * (dist / max(1e-3, float(sigma))) ** 2).astype(np.float32)
    return targets


def augment_sequence_length(
    frames: torch.Tensor,
    state_labels: torch.Tensor,
    boundary_targets: torch.Tensor,
    end_round_frame: int | None,
    *,
    rng: np.random.Generator | None = None,
    p_trim_edges: float = 0.65,
    p_stretch_scoring: float = 0.50,
    p_stretch_post: float = 0.35,
    pre_transition_safe_margin: int = 8,
    post_transition_safe_margin: int = 4,
    sigma: float = BOUNDARY_SIGMA_FRAMES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int | None]:
    """Safely vary clip duration while keeping the END_ROUND transition intact.

    Why:
      Synthetic Balatro mod clips are short (typically 3-6 seconds), while real
      VOD segments can have 20-60+ second scoring chains or Cash Out stalls.
      This helper trims outer tails and inserts smooth ping-pong loops strictly
      outside the card-exit transition band ``[end_round - safe, end_round + safe]``.
    """
    if rng is None:
        rng = np.random.default_rng()

    t_len = int(frames.shape[0])
    if t_len < 12 or end_round_frame is None:
        return frames, state_labels, boundary_targets, end_round_frame

    labels_np = state_labels.cpu().numpy()
    scoring_idx = STATE_TO_IDX["SCORING"]
    round_eval_idx = STATE_TO_IDX["ROUND_EVAL"]

    # 1. Stretch an interior SCORING slice (strictly before the exit band) via ping-pong loop
    safe_scoring_end = int(end_round_frame) - pre_transition_safe_margin
    if safe_scoring_end > 6 and rng.random() < p_stretch_scoring:
        scoring_positions = np.flatnonzero(labels_np[:safe_scoring_end] == scoring_idx)
        if len(scoring_positions) >= 8:
            span_start = int(scoring_positions[0])
            span_end = int(scoring_positions[-1]) + 1
            if span_end - span_start >= 6:
                seg_len = int(rng.integers(3, min(14, span_end - span_start)))
                lo = int(rng.integers(span_start, span_end - seg_len + 1))
                hi = lo + seg_len
                # Forward + reverse ping-pong avoids an abrupt teleport cut
                fwd = torch.arange(lo, hi, dtype=torch.long)
                rev = torch.arange(hi - 2, lo, -1, dtype=torch.long) if seg_len >= 3 else fwd.flip(0)
                loop_idx = torch.cat([fwd, rev], dim=0)
                n_loops = int(rng.integers(1, 3))
                inserted = loop_idx.repeat(n_loops)

                prefix = torch.arange(0, hi, dtype=torch.long)
                suffix = torch.arange(hi, t_len, dtype=torch.long)
                gather_idx = torch.cat([prefix, inserted, suffix], dim=0)

                frames = frames.index_select(0, gather_idx)
                state_labels = state_labels.index_select(0, gather_idx)
                labels_np = state_labels.cpu().numpy()
                end_round_frame = int(end_round_frame) + int(inserted.numel())
                t_len = int(frames.shape[0])

    # 2. Hold/stretch settled ROUND_EVAL frames strictly after the transition band
    safe_post_start = int(end_round_frame) + post_transition_safe_margin
    if safe_post_start + 4 < t_len and rng.random() < p_stretch_post:
        post_positions = np.flatnonzero(labels_np[safe_post_start:] == round_eval_idx)
        if len(post_positions) >= 4:
            hold_pos = safe_post_start + int(post_positions[len(post_positions) // 2])
            hold_reps = int(rng.integers(4, 18))
            gather_idx = torch.cat(
                [
                    torch.arange(0, hold_pos + 1, dtype=torch.long),
                    torch.full((hold_reps,), hold_pos, dtype=torch.long),
                    torch.arange(hold_pos + 1, t_len, dtype=torch.long),
                ],
                dim=0,
            )
            frames = frames.index_select(0, gather_idx)
            state_labels = state_labels.index_select(0, gather_idx)
            t_len = int(frames.shape[0])

    # 3. Randomly trim PRE_PLAY prefix / far ROUND_EVAL tail while preserving transition context
    if rng.random() < p_trim_edges and t_len > 16:
        max_start = max(0, int(end_round_frame) - max(8, pre_transition_safe_margin))
        pre_positions = np.flatnonzero(labels_np[: max_start + 1] == STATE_TO_IDX["PRE_PLAY"])
        trim_start = 0
        if len(pre_positions) > 3:
            trim_start = int(rng.integers(0, min(max_start, int(pre_positions[-1])) + 1))

        min_end = min(t_len, int(end_round_frame) + post_transition_safe_margin + 2)
        trim_end = t_len
        if t_len - min_end > 4:
            trim_end = int(rng.integers(min_end, t_len + 1))

        if trim_end - trim_start >= 10:
            frames = frames[trim_start:trim_end]
            state_labels = state_labels[trim_start:trim_end]
            end_round_frame = int(end_round_frame) - trim_start
            t_len = int(frames.shape[0])

    # Recompute boundary targets cleanly to reflect any temporal shifts
    new_boundary = _recompute_gaussian_boundary(t_len, end_round_frame, sigma=sigma)
    # Preserve IGNORE_INDEX semantics where boundary wasn't valid? Actually boundary is
    # always well-defined when end_round_frame is known (even on partially labeled VODs).
    boundary_targets = torch.from_numpy(new_boundary)
    return frames, state_labels, boundary_targets, end_round_frame


class TemporalCursorOverlay:
    """Smooth Bezier cursor overlay across a video clip batch.

    Reuses ``cursor.png`` from ``classifiers/augmentations.py`` and animates a
    continuous pointer path so the temporal backbone learns to ignore mouse
    movement over ``G.play``.
    """

    def __init__(self, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        rgba = _load_shared_cursor_rgba()
        if rgba is None:
            self.rgb: torch.Tensor | None = None
            self.alpha: torch.Tensor | None = None
            self.aspect: float = 1.0
            return

        # Convert RGBA HxWx4 -> (1, 3, H, W) and (1, 1, H, W) in [0, 1]
        rgb_np = np.clip(rgba[:, :, :3] / 255.0, 0.0, 1.0).astype(np.float32)
        alpha_np = np.clip(rgba[:, :, 3:4], 0.0, 1.0).astype(np.float32)
        self.rgb = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
        self.alpha = torch.from_numpy(alpha_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
        self.aspect = float(rgba.shape[1]) / max(1.0, float(rgba.shape[0]))

    def to(self, device: torch.device | str) -> "TemporalCursorOverlay":
        target = torch.device(device)
        if self.rgb is not None and self.rgb.device != target:
            self.rgb = self.rgb.to(target)
        if self.alpha is not None and self.alpha.device != target:
            self.alpha = self.alpha.to(target)
        self.device = target
        return self

    def apply(
        self,
        frames: torch.Tensor,
        lengths: torch.Tensor,
        *,
        prob: float = 0.45,
    ) -> torch.Tensor:
        """Overlay smooth cursor trajectories on a ``(B, T, 3, H, W)`` float tensor."""
        if self.rgb is None or self.alpha is None or prob <= 0.0:
            return frames

        if self.rgb.device != frames.device:
            self.to(frames.device)

        bsz, max_t, _, height, width = frames.shape
        out = frames.clone()
        lengths_list = lengths.tolist()
        apply_flags = (torch.rand(bsz) < prob).tolist()

        for b in range(bsz):
            if not apply_flags[b]:
                continue
            valid_len = max(1, int(lengths_list[b]))

            # Scale cursor proportionally to the 72px-tall cropped G.play ROI (~12-22 px)
            cur_h = max(8, int(round(height * float(torch.empty(()).uniform_(0.16, 0.30).item()))))
            cur_w = max(8, int(round(cur_h * self.aspect)))
            if cur_h >= height or cur_w >= width:
                continue

            cur_rgb = F.interpolate(self.rgb, size=(cur_h, cur_w), mode="bilinear", align_corners=False)[0]
            cur_alpha = F.interpolate(self.alpha, size=(cur_h, cur_w), mode="bilinear", align_corners=False)[0]
            cur_rgb = cur_rgb.to(dtype=out.dtype)
            cur_alpha = cur_alpha.to(dtype=out.dtype)

            # Compute Bezier trajectory on CPU to avoid per-frame CUDA syncs (.item()).
            pts = torch.rand(6)
            max_x = float(max(1, width - cur_w))
            max_y = float(max(1, height - cur_h))
            p0x, p1x, p2x = (pts[0] * max_x), (pts[1] * max_x), (pts[2] * max_x)
            p0y, p1y, p2y = (pts[3] * max_y), (pts[4] * max_y), (pts[5] * max_y)

            # Optional mid-clip hover pause (mimics user moving to hover Cash Out / Jokers)
            t_param = torch.linspace(0.0, 1.0, valid_len)
            if float(torch.rand(()).item()) < 0.35:
                t_param = torch.clamp((t_param - 0.15) * 1.35, 0.0, 1.0)

            omt = 1.0 - t_param
            xs = (omt * omt * p0x + 2.0 * omt * t_param * p1x + t_param * t_param * p2x).round().long()
            ys = (omt * omt * p0y + 2.0 * omt * t_param * p1y + t_param * t_param * p2y).round().long()
            xs_list = xs.clamp(0, width - cur_w).tolist()
            ys_list = ys.clamp(0, height - cur_h).tolist()

            inv_alpha = 1.0 - cur_alpha
            weighted_rgb = cur_rgb * cur_alpha
            for t in range(valid_len):
                x = xs_list[t]
                y = ys_list[t]
                patch = out[b, t, :, y : y + cur_h, x : x + cur_w]
                out[b, t, :, y : y + cur_h, x : x + cur_w] = patch * inv_alpha + weighted_rgb

        return out


@torch.no_grad()
def apply_temporal_clip_augmentations(
    frames: torch.Tensor,
    lengths: torch.Tensor,
    *,
    cursor_overlay: TemporalCursorOverlay | None = None,
    p_spatial_jitter: float = 0.70,
    p_color_jitter: float = 0.85,
    p_compression: float = 0.65,
    p_blur: float = 0.25,
    p_noise: float = 0.40,
    p_cursor: float = 0.45,
) -> torch.Tensor:
    """Apply clip-consistent photometric, spatial, compression, and cursor noise.

    Args:
        frames: Float tensor ``(B, T, 3, H, W)`` in ``[0.0, 1.0]``.
        lengths: Int64 tensor ``(B,)`` of valid clip frame counts.

    Returns:
        Augmented float tensor ``(B, T, 3, H, W)`` clamped to ``[0.0, 1.0]``.
    """
    bsz, max_t, _, height, width = frames.shape
    device = frames.device
    dtype = frames.dtype
    out = frames

    # 1. Clip-consistent small ROI misalignment (translation +/-4% + slight zoom)
    if p_spatial_jitter > 0.0:
        apply_mask = (torch.rand(bsz, device=device) < p_spatial_jitter).to(dtype=dtype)
        # Identity affine matrix per clip perturbed identically for all T frames
        scale = torch.where(
            apply_mask > 0.5,
            torch.empty(bsz, device=device, dtype=dtype).uniform_(0.95, 1.05),
            torch.ones(bsz, device=device, dtype=dtype),
        )
        tx = apply_mask * torch.empty(bsz, device=device, dtype=dtype).uniform_(-0.04, 0.04)
        ty = apply_mask * torch.empty(bsz, device=device, dtype=dtype).uniform_(-0.05, 0.05)

        theta = torch.zeros((bsz, 2, 3), device=device, dtype=dtype)
        theta[:, 0, 0] = scale
        theta[:, 1, 1] = scale
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty
        # Expand clip affine transform across T without inter-frame drift
        theta_exp = theta.unsqueeze(1).expand(bsz, max_t, 2, 3).reshape(bsz * max_t, 2, 3)
        grid = F.affine_grid(theta_exp, size=(bsz * max_t, 3, height, width), align_corners=False)
        flat = F.grid_sample(
            out.reshape(bsz * max_t, 3, height, width),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        out = flat.reshape(bsz, max_t, 3, height, width)

    # 2. Clip-consistent color / capture-card exposure / tint jitter
    if p_color_jitter > 0.0:
        apply_c = (torch.rand((bsz, 1, 1, 1, 1), device=device) < p_color_jitter).to(dtype=dtype)
        brightness = torch.empty((bsz, 1, 1, 1, 1), device=device, dtype=dtype).uniform_(-0.09, 0.09)
        contrast = torch.empty((bsz, 1, 1, 1, 1), device=device, dtype=dtype).uniform_(0.84, 1.18)
        saturation = torch.empty((bsz, 1, 1, 1, 1), device=device, dtype=dtype).uniform_(0.80, 1.22)
        rgb_gain = torch.empty((bsz, 1, 3, 1, 1), device=device, dtype=dtype).uniform_(0.94, 1.06)

        # Blend with 1.0/0.0 when inactive so unaffected clips stay exact
        contrast = 1.0 + apply_c * (contrast - 1.0)
        saturation = 1.0 + apply_c * (saturation - 1.0)
        brightness = apply_c * brightness
        rgb_gain = 1.0 + apply_c * (rgb_gain - 1.0)

        out = (out - 0.5) * contrast + 0.5 + brightness
        luma = (
            0.299 * out[:, :, 0:1]
            + 0.587 * out[:, :, 1:2]
            + 0.114 * out[:, :, 2:3]
        )
        out = luma + (out - luma) * saturation
        out = (out * rgb_gain).clamp(0.0, 1.0)

    # 3. Cursor trajectory overlay (applied before compression/blur like OBS capture)
    if cursor_overlay is not None and p_cursor > 0.0:
        out = cursor_overlay.apply(out, lengths, prob=p_cursor)

    # 4. Clip-consistent stream compression (resolution downsample + slight quantization)
    if p_compression > 0.0:
        out = out.clone()
        lengths_list = lengths.tolist()
        comp_apply = (torch.rand(bsz) < p_compression).tolist()
        for b in range(bsz):
            if not comp_apply[b]:
                continue
            valid_len = max(1, int(lengths_list[b]))
            ratio = float(torch.empty(()).uniform_(0.50, 0.92).item())
            low_h = max(24, int(round(height * ratio)))
            low_w = max(64, int(round(width * ratio)))
            clip = out[b, :valid_len]
            down = F.interpolate(clip, size=(low_h, low_w), mode="bilinear", align_corners=False)
            up = F.interpolate(down, size=(height, width), mode="bilinear", align_corners=False)
            if float(torch.rand(()).item()) < 0.5:
                # Mild 5-bit/6-bit chroma quantization typical of aggressive H.264 presets
                levels = float(torch.empty(()).uniform_(28.0, 64.0).item())
                up = torch.round(up * levels) / levels
            out[b, :valid_len] = up

    # 5. Mild clip-level optical/stream blur
    if p_blur > 0.0:
        blur_mask = torch.rand(bsz, device=device) < p_blur
        if bool(blur_mask.any()):
            flat_blur = F.avg_pool2d(
                out[blur_mask].reshape(-1, 3, height, width),
                kernel_size=3,
                stride=1,
                padding=1,
            )
            out = out.clone()
            out[blur_mask] = flat_blur.reshape(-1, max_t, 3, height, width)

    # 6. Subtle temporally mixed grain (per-clip strength, fine frame variation)
    if p_noise > 0.0:
        noise_mask = (torch.rand((bsz, 1, 1, 1, 1), device=device) < p_noise).to(dtype=dtype)
        sigma = noise_mask * torch.empty((bsz, 1, 1, 1, 1), device=device, dtype=dtype).uniform_(0.005, 0.025)
        out = (out + torch.randn_like(out) * sigma).clamp(0.0, 1.0)

    # Ensure trailing padded frames stay exact zeros
    pad_mask = torch.arange(max_t, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
    if bool(pad_mask.any()):
        out = out.masked_fill(pad_mask.unsqueeze(2).unsqueeze(3).unsqueeze(4), 0.0)

    return out
