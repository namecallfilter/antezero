"""Temporal segmentation + boundary spotting model and Viterbi decoder."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    mobilenet_v3_large,
    mobilenet_v3_small,
    resnet18,
)

try:
    from .config import (
        DEFAULT_BACKBONE,
        DEFAULT_BOUNDARY_REFINE_ALPHA,
        DEFAULT_BOUNDARY_REFINE_RADIUS,
        DEFAULT_DROPOUT,
        DEFAULT_FRAME_FEAT_DIM,
        DEFAULT_MIN_POST_PLAY_FRAMES,
        DEFAULT_MIN_SCORING_FRAMES,
        DEFAULT_NUM_GRU_LAYERS,
        DEFAULT_SPATIAL_POOL_W,
        DEFAULT_TEMPORAL_HIDDEN_DIM,
        IMAGENET_MEAN,
        IMAGENET_STD,
        INPUT_HEIGHT,
        INPUT_WIDTH,
        NUM_STATES,
        STATE_TO_IDX,
    )
except ImportError:
    from config import (  # type: ignore[no-redef]
        DEFAULT_BACKBONE,
        DEFAULT_BOUNDARY_REFINE_ALPHA,
        DEFAULT_BOUNDARY_REFINE_RADIUS,
        DEFAULT_DROPOUT,
        DEFAULT_FRAME_FEAT_DIM,
        DEFAULT_MIN_POST_PLAY_FRAMES,
        DEFAULT_MIN_SCORING_FRAMES,
        DEFAULT_NUM_GRU_LAYERS,
        DEFAULT_SPATIAL_POOL_W,
        DEFAULT_TEMPORAL_HIDDEN_DIM,
        IMAGENET_MEAN,
        IMAGENET_STD,
        INPUT_HEIGHT,
        INPUT_WIDTH,
        NUM_STATES,
        STATE_TO_IDX,
    )

SUPPORTED_BACKBONES = ("mobilenet_v3_small", "mobilenet_v3_large", "resnet18")


def _build_spatial_backbone(name: str, pretrained: bool) -> nn.Module:
    """Build a convolutional trunk without its final global pool / classifier."""
    if name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        return mobilenet_v3_small(weights=weights).features
    if name == "mobilenet_v3_large":
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        return mobilenet_v3_large(weights=weights).features
    if name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        net = resnet18(weights=weights)
        return nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
            net.layer1,
            net.layer2,
            net.layer3,
            net.layer4,
        )
    raise ValueError(f"Unsupported backbone '{name}'. Choose from {SUPPORTED_BACKBONES}.")


class _ResidualTemporalConvBlock(nn.Module):
    """Masked residual dilated 1D conv block for short-horizon motion cues."""

    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.norm1 = nn.LayerNorm(channels)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, valid_mask_bt1: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C), valid_mask_bt1: (B, T, 1) float/bool
        y = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        y = F.gelu(self.norm1(y))
        y = self.dropout(y)
        y = self.conv2(y.transpose(1, 2)).transpose(1, 2)
        y = self.norm2(y)
        out = F.gelu(x + y)
        return out * valid_mask_bt1


class TemporalEndRoundNet(nn.Module):
    """Variable-length sequence model for END_ROUND segmentation & boundary spotting.

    Expected inputs:
      - ``frames``: ``(B, T, 3, H, W)`` float in ``[0, 1]`` (or ``uint8``).
      - ``lengths``: optional ``(B,)`` valid clip lengths. Only valid frames are
        evaluated through the CNN trunk so trailing zero padding never pollutes
        BatchNorm running statistics.
    """

    def __init__(
        self,
        backbone_name: str = DEFAULT_BACKBONE,
        *,
        pretrained: bool = True,
        spatial_pool_w: int = DEFAULT_SPATIAL_POOL_W,
        frame_feat_dim: int = DEFAULT_FRAME_FEAT_DIM,
        temporal_hidden_dim: int = DEFAULT_TEMPORAL_HIDDEN_DIM,
        num_gru_layers: int = DEFAULT_NUM_GRU_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
        num_states: int = NUM_STATES,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.spatial_pool_w = int(spatial_pool_w)
        self.frame_feat_dim = int(frame_feat_dim)
        self.temporal_hidden_dim = int(temporal_hidden_dim)
        self.num_gru_layers = int(num_gru_layers)
        self.num_states = int(num_states)

        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("img_mean", mean, persistent=False)
        self.register_buffer("img_std", std, persistent=False)

        self.backbone = _build_spatial_backbone(backbone_name, pretrained=pretrained)

        # Infer exact backbone output channel width safely in eval mode
        was_training = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            probe = torch.zeros((1, 3, INPUT_HEIGHT, INPUT_WIDTH), dtype=torch.float32)
            feat_c = int(self.backbone(probe).shape[1])
        self.backbone.train(was_training)

        # Keep horizontal (left -> right across G.play) structure via (1, W_bins) pooling
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, self.spatial_pool_w))
        self.frame_proj = nn.Sequential(
            nn.Linear(feat_c * self.spatial_pool_w, self.frame_feat_dim),
            nn.LayerNorm(self.frame_feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Short-horizon residual dilated conv stem (~15 frames receptive field)
        self.tcn_blocks = nn.ModuleList(
            [
                _ResidualTemporalConvBlock(self.frame_feat_dim, dilation=d, dropout=dropout)
                for d in (1, 2, 4)
            ]
        )

        # Long-horizon bidirectional context over full [PLAY_HAND, CASH_OUT] segments
        self.gru = nn.GRU(
            input_size=self.frame_feat_dim,
            hidden_size=self.temporal_hidden_dim,
            num_layers=self.num_gru_layers,
            batch_first=True,
            dropout=dropout if self.num_gru_layers > 1 else 0.0,
            bidirectional=True,
        )
        gru_out_dim = 2 * self.temporal_hidden_dim
        self.head_norm = nn.Sequential(
            nn.LayerNorm(gru_out_dim),
            nn.Dropout(dropout),
        )
        self.state_head = nn.Linear(gru_out_dim, self.num_states)
        self.boundary_head = nn.Linear(gru_out_dim, 1)

    def forward(
        self,
        frames: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if frames.ndim != 5:
            raise ValueError(f"Expected 5D tensor (B, T, 3, H, W), got {tuple(frames.shape)}")

        bsz, max_t, _, height, width = frames.shape
        device = frames.device

        if lengths is None:
            lengths = torch.full((bsz,), max_t, dtype=torch.long, device=device)
        else:
            lengths = lengths.to(device=device, dtype=torch.long)

        valid_mask = torch.arange(max_t, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        valid_frames = frames[valid_mask]
        if valid_frames.dtype == torch.uint8:
            valid_frames = valid_frames.float() / 255.0

        # Normalize using ImageNet statistics after any [0, 1] augmentations
        valid_norm = (valid_frames - self.img_mean.to(valid_frames.dtype)) / self.img_std.to(valid_frames.dtype)

        spatial_map = self.backbone(valid_norm)
        pooled = self.spatial_pool(spatial_map).flatten(1)
        frame_emb = self.frame_proj(pooled)

        seq_emb = torch.zeros(
            (bsz, max_t, self.frame_feat_dim),
            device=device,
            dtype=frame_emb.dtype,
        )
        seq_emb[valid_mask] = frame_emb

        mask_bt1 = valid_mask.unsqueeze(-1).to(dtype=seq_emb.dtype)
        tcn_out = seq_emb
        for block in self.tcn_blocks:
            tcn_out = block(tcn_out, mask_bt1)

        packed = nn.utils.rnn.pack_padded_sequence(
            tcn_out,
            lengths.clamp(min=1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.gru(packed)
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=max_t,
        )

        feats = self.head_norm(gru_out) * mask_bt1
        state_logits = self.state_head(feats)
        boundary_logits = self.boundary_head(feats).squeeze(-1)

        return {
            "state_logits": state_logits,
            "boundary_logits": boundary_logits,
            "valid_mask": valid_mask,
        }


def decode_monotonic_viterbi(
    state_log_probs: np.ndarray,
    *,
    min_scoring_frames: int = DEFAULT_MIN_SCORING_FRAMES,
    min_post_play_frames: int = DEFAULT_MIN_POST_PLAY_FRAMES,
    allow_start_in_scoring: bool = True,
    require_terminal_post_or_eval: bool = True,
) -> np.ndarray:
    """Decode an exact monotonic ``0 -> 1 -> 2 -> 3`` state path via Viterbi DP.

    Args:
        state_log_probs: Array of shape ``(T, 4)`` containing log probabilities
            for ``[PRE_PLAY, SCORING, POST_PLAY_EMPTY, ROUND_EVAL]``.

    Returns:
        Integer array of shape ``(T,)`` with decoded state indices in ``[0, 3]``.
    """
    t_len, num_states = state_log_probs.shape
    if t_len <= 0:
        return np.zeros((0,), dtype=np.int64)
    if num_states != 4:
        return np.argmax(state_log_probs, axis=-1).astype(np.int64)

    # Adaptive dwell clamps so tiny edge sequences remain feasible
    m_scoring = max(1, min(int(min_scoring_frames), max(1, t_len - 1)))
    m_post = max(1, min(int(min_post_play_frames), max(1, t_len - m_scoring)))
    if t_len < m_scoring + m_post:
        m_scoring = 1
        m_post = 1

    # Expanded monotonic sub-states enforce minimum dwell in SCORING and POST_PLAY_EMPTY
    # Index layout: [PRE(0), SCOR_1..SCOR_m1(1), POST_1..POST_m2(2), EVAL(3)]
    sub_to_state: list[int] = [0]
    scoring_sub_start = 1
    for _ in range(m_scoring):
        sub_to_state.append(1)
    post_sub_start = 1 + m_scoring
    for _ in range(m_post):
        sub_to_state.append(2)
    eval_sub = 1 + m_scoring + m_post
    sub_to_state.append(3)

    n_sub = len(sub_to_state)
    neg_inf = -1e9
    trans = np.full((n_sub, n_sub), neg_inf, dtype=np.float64)

    # PRE_PLAY self-loop & exit into SCORING
    trans[0, 0] = 0.0
    trans[0, scoring_sub_start] = -0.15

    # SCORING dwell chain -> final SCORING sub-state self-loop -> POST_PLAY_EMPTY
    for k in range(m_scoring - 1):
        trans[scoring_sub_start + k, scoring_sub_start + k + 1] = 0.0
    scoring_end = scoring_sub_start + m_scoring - 1
    trans[scoring_end, scoring_end] = 0.0
    trans[scoring_end, post_sub_start] = -0.15

    # POST_PLAY_EMPTY dwell chain -> final POST sub-state self-loop -> ROUND_EVAL
    for k in range(m_post - 1):
        trans[post_sub_start + k, post_sub_start + k + 1] = 0.0
    post_end = post_sub_start + m_post - 1
    trans[post_end, post_end] = 0.0
    trans[post_end, eval_sub] = -0.15

    # ROUND_EVAL absorbing self-loop
    trans[eval_sub, eval_sub] = 0.0

    sub_state_arr = np.asarray(sub_to_state, dtype=np.int64)
    dp = np.full((t_len, n_sub), neg_inf, dtype=np.float64)
    backptr = np.zeros((t_len, n_sub), dtype=np.int64)

    # Initial priors at t=0
    dp[0, 0] = float(state_log_probs[0, 0])
    if allow_start_in_scoring:
        dp[0, scoring_sub_start] = float(state_log_probs[0, 1]) - 0.50

    for t in range(1, t_len):
        # scores[u, v] = dp[t-1, u] + trans[u, v]
        scores = dp[t - 1, :, None] + trans
        best_prev = np.argmax(scores, axis=0)
        best_scores = scores[best_prev, np.arange(n_sub)]
        emit = state_log_probs[t, sub_state_arr]
        dp[t] = best_scores + emit
        backptr[t] = best_prev

    end_scores = dp[t_len - 1].copy()
    if require_terminal_post_or_eval:
        # Forced alignment over [PLAY_HAND, CASH_OUT]: segment must finish in POST_PLAY_EMPTY or ROUND_EVAL
        valid_tail = end_scores[post_sub_start:] > (neg_inf / 2)
        if bool(valid_tail.any()):
            end_scores[:post_sub_start] = neg_inf
            end_scores[eval_sub] += 0.25

    cur = int(np.argmax(end_scores))
    sub_path = np.empty((t_len,), dtype=np.int64)
    sub_path[t_len - 1] = cur
    for t in range(t_len - 1, 0, -1):
        cur = int(backptr[t, cur])
        sub_path[t - 1] = cur

    return sub_state_arr[sub_path]


def predict_end_round_index(
    state_logits: np.ndarray,
    boundary_logits: np.ndarray,
    *,
    min_scoring_frames: int = DEFAULT_MIN_SCORING_FRAMES,
    min_post_play_frames: int = DEFAULT_MIN_POST_PLAY_FRAMES,
    refine_radius: int = DEFAULT_BOUNDARY_REFINE_RADIUS,
    refine_alpha: float = DEFAULT_BOUNDARY_REFINE_ALPHA,
) -> dict[str, object]:
    """Decode state sequence + refined ``END_ROUND`` frame index from logits.

    Args:
        state_logits: Array of shape ``(T, 4)``.
        boundary_logits: Array of shape ``(T,)``.

    Returns:
        Dict with ``end_round_frame`` (int), ``end_round_float`` (float),
        ``viterbi_transition_frame`` (int), ``boundary_peak_frame`` (int),
        ``decoded_states`` (np.ndarray ``(T,)``), ``state_probs`` (``(T, 4)``),
        ``boundary_probs`` (``(T,)``), and ``confidence`` (float in ``[0, 1]``).
    """
    logits = np.asarray(state_logits, dtype=np.float64)
    b_logits = np.asarray(boundary_logits, dtype=np.float64)
    t_len = int(logits.shape[0])
    if t_len == 0:
        return {
            "end_round_frame": 0,
            "end_round_float": 0.0,
            "viterbi_transition_frame": 0,
            "boundary_peak_frame": 0,
            "decoded_states": np.zeros((0,), dtype=np.int64),
            "state_probs": np.zeros((0, NUM_STATES), dtype=np.float32),
            "boundary_probs": np.zeros((0,), dtype=np.float32),
            "confidence": 0.0,
        }

    # Numerically stable log-softmax & sigmoid
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    log_probs = shifted - log_sum_exp
    state_probs = np.exp(log_probs).astype(np.float32)

    b_probs = (1.0 / (1.0 + np.exp(-np.clip(b_logits, -40.0, 40.0)))).astype(np.float32)

    decoded = decode_monotonic_viterbi(
        log_probs,
        min_scoring_frames=min_scoring_frames,
        min_post_play_frames=min_post_play_frames,
    )

    post_idx = STATE_TO_IDX["POST_PLAY_EMPTY"]
    post_or_later = np.flatnonzero(decoded >= post_idx)
    global_b_peak = int(np.argmax(b_probs))
    if len(post_or_later) > 0:
        t_viterbi = int(post_or_later[0])
    else:
        t_viterbi = global_b_peak

    # Local boundary refinement around Viterbi 1 -> 2 transition
    lo = max(0, t_viterbi - int(refine_radius))
    hi = min(t_len, t_viterbi + int(refine_radius) + 1)
    local_b = b_probs[lo:hi]
    local_peak_offset = int(np.argmax(local_b))
    t_local_peak = lo + local_peak_offset
    local_peak_val = float(local_b[local_peak_offset])

    if local_peak_val >= 0.15:
        # Sub-frame weighted centroid in +-1 around the local peak for extra precision
        c_lo = max(lo, t_local_peak - 1)
        c_hi = min(hi, t_local_peak + 2)
        weights = b_probs[c_lo:c_hi].astype(np.float64)
        w_sum = float(np.sum(weights))
        if w_sum > 1e-6:
            coords = np.arange(c_lo, c_hi, dtype=np.float64)
            t_boundary_sub = float(np.sum(coords * weights) / w_sum)
        else:
            t_boundary_sub = float(t_local_peak)
        alpha = float(np.clip(refine_alpha, 0.0, 1.0))
        end_round_float = (1.0 - alpha) * float(t_viterbi) + alpha * t_boundary_sub
    else:
        end_round_float = float(t_viterbi)

    end_round_frame = int(np.clip(round(end_round_float), 0, t_len - 1))

    # Confidence heuristic combining local boundary peak & posterior margin around transition
    pre_prob = float(np.max(state_probs[max(0, end_round_frame - 2) : max(1, end_round_frame), 1])) if end_round_frame > 0 else 0.5
    post_prob = float(np.max(state_probs[end_round_frame : min(t_len, end_round_frame + 2), 2:]))
    confidence = float(np.clip(0.5 * local_peak_val + 0.25 * pre_prob + 0.25 * post_prob, 0.0, 1.0))

    return {
        "end_round_frame": end_round_frame,
        "end_round_float": float(end_round_float),
        "viterbi_transition_frame": t_viterbi,
        "boundary_peak_frame": global_b_peak,
        "decoded_states": decoded,
        "state_probs": state_probs,
        "boundary_probs": b_probs,
        "confidence": confidence,
    }
