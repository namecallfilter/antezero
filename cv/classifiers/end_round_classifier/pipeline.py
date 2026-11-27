"""Production inference pipeline for END_ROUND timestamp detection.

Designed to hook cleanly into ``post_processing.py`` between each round's final
``PLAY_HAND`` action and ``CASH_OUT`` action without touching unrelated modules
until the model is trained and verified.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch

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
        INPUT_HEIGHT,
        INPUT_WIDTH,
        NUM_STATES,
        PLAY_AREA_ROI,
        STATES,
        TARGET_FPS,
    )
    from .model import TemporalEndRoundNet, predict_end_round_index
    from .video_io import frames_sequence_to_tensor, read_video_segment_rgb
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
        INPUT_HEIGHT,
        INPUT_WIDTH,
        NUM_STATES,
        PLAY_AREA_ROI,
        STATES,
        TARGET_FPS,
    )
    from model import TemporalEndRoundNet, predict_end_round_index  # type: ignore[no-redef]
    from video_io import frames_sequence_to_tensor, read_video_segment_rgb  # type: ignore[no-redef]

CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHT_CANDIDATES = (
    CURRENT_DIR.parent / "prod_weights" / "end_round_classifier.pt",
    CURRENT_DIR / "weights" / "end_round_best.pt",
)


@dataclass(slots=True)
class EndRoundResult:
    """Prediction output for a single ``[PLAY_HAND, CASH_OUT]`` video segment."""

    end_round_frame: int
    end_round_timestamp: float
    confidence: float
    model_step_index: int
    model_step_float: float
    viterbi_transition_step: int
    boundary_peak_step: int
    src_fps: float
    num_sampled_frames: int
    decoded_states: np.ndarray | None = None
    state_probs: np.ndarray | None = None
    boundary_probs: np.ndarray | None = None
    sampled_video_frames: list[int] | None = None

    @property
    def state_names(self) -> list[str]:
        if self.decoded_states is None:
            return []
        return [STATES[int(idx)] for idx in self.decoded_states]


def resolve_weights_path(weights_path: Path | str | None = None) -> Path:
    """Locate trained checkpoint path or raise a descriptive error."""
    if weights_path is not None:
        p = Path(weights_path)
        if not p.exists():
            raise FileNotFoundError(f"END_ROUND checkpoint not found: {p}")
        return p
    for candidate in DEFAULT_WEIGHT_CANDIDATES:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(p) for p in DEFAULT_WEIGHT_CANDIDATES)
    raise FileNotFoundError(
        f"No END_ROUND checkpoint found. Searched: {searched}. "
        "Train via train.py or pass weights_path explicitly."
    )


def _probe_video_fps(video_path: Path | str) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 60.0)
        return fps if fps > 0.0 else 60.0
    finally:
        cap.release()


class EndRoundRecognitionPipeline:
    """End-to-end temporal inference wrapper for END_ROUND timestamp extraction."""

    def __init__(
        self,
        weights_path: Path | str | None = None,
        *,
        device: str | torch.device | None = None,
        use_amp: bool = True,
    ) -> None:
        self.weights_path = resolve_weights_path(weights_path)
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.amp_dtype = (
            torch.bfloat16
            if (self.use_amp and torch.cuda.is_bf16_supported())
            else (torch.float16 if self.use_amp else torch.float32)
        )

        ckpt = torch.load(self.weights_path, map_location="cpu", weights_only=False)
        cfg = dict(ckpt.get("model_config") or {})

        self.target_fps: float = float(cfg.get("target_fps", TARGET_FPS))
        self.input_width: int = int(cfg.get("input_width", INPUT_WIDTH))
        self.input_height: int = int(cfg.get("input_height", INPUT_HEIGHT))
        self.roi_base_1080p: dict[str, float] = dict(cfg.get("roi_base_1080p") or PLAY_AREA_ROI)

        v_cfg = dict(cfg.get("viterbi_params") or {})
        self.min_scoring_frames: int = int(v_cfg.get("min_scoring_frames", DEFAULT_MIN_SCORING_FRAMES))
        self.min_post_play_frames: int = int(v_cfg.get("min_post_play_frames", DEFAULT_MIN_POST_PLAY_FRAMES))
        self.refine_radius: int = int(v_cfg.get("refine_radius", DEFAULT_BOUNDARY_REFINE_RADIUS))
        self.refine_alpha: float = float(v_cfg.get("refine_alpha", DEFAULT_BOUNDARY_REFINE_ALPHA))

        self.model = TemporalEndRoundNet(
            backbone_name=str(cfg.get("backbone_name", DEFAULT_BACKBONE)),
            pretrained=False,
            spatial_pool_w=int(cfg.get("spatial_pool_w", DEFAULT_SPATIAL_POOL_W)),
            frame_feat_dim=int(cfg.get("frame_feat_dim", DEFAULT_FRAME_FEAT_DIM)),
            temporal_hidden_dim=int(cfg.get("temporal_hidden_dim", DEFAULT_TEMPORAL_HIDDEN_DIM)),
            num_gru_layers=int(cfg.get("num_gru_layers", DEFAULT_NUM_GRU_LAYERS)),
            dropout=float(cfg.get("dropout", DEFAULT_DROPOUT)),
            num_states=int(cfg.get("num_states", NUM_STATES)),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.model.eval()

    @torch.no_grad()
    def predict_batch(
        self,
        clips: torch.Tensor,
        lengths: torch.Tensor | None = None,
        *,
        fps_list: Sequence[float] | None = None,
        sampled_video_frames_list: Sequence[Sequence[int]] | None = None,
        return_profiles: bool = False,
    ) -> list[EndRoundResult]:
        """Run batched inference on ``(B, T, 3, H, W)`` clips (``uint8`` or ``[0, 1]`` float)."""
        if clips.ndim != 5:
            raise ValueError(f"Expected 5D clips tensor (B, T, 3, H, W), got {tuple(clips.shape)}")

        bsz, max_t, _, _, _ = clips.shape
        x = clips.to(self.device, non_blocking=True)
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        len_t = (
            torch.full((bsz,), max_t, dtype=torch.long, device=self.device)
            if lengths is None
            else lengths.to(self.device, dtype=torch.long)
        )

        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        ):
            outputs = self.model(x, lengths=len_t)

        state_logits_cpu = outputs["state_logits"].float().cpu().numpy()
        boundary_logits_cpu = outputs["boundary_logits"].float().cpu().numpy()
        lengths_cpu = len_t.cpu().tolist()

        results: list[EndRoundResult] = []
        for b in range(bsz):
            t_len = max(1, int(lengths_cpu[b]))
            decoded = predict_end_round_index(
                state_logits_cpu[b, :t_len],
                boundary_logits_cpu[b, :t_len],
                min_scoring_frames=self.min_scoring_frames,
                min_post_play_frames=self.min_post_play_frames,
                refine_radius=self.refine_radius,
                refine_alpha=self.refine_alpha,
            )

            step_idx = int(decoded["end_round_frame"])
            step_float = float(decoded["end_round_float"])
            src_fps = float(fps_list[b]) if fps_list is not None else self.target_fps
            src_fps = max(1e-3, src_fps)

            if sampled_video_frames_list is not None:
                vf_list = [int(v) for v in sampled_video_frames_list[b]]
                if vf_list:
                    vf_interp = float(
                        np.interp(step_float, np.arange(len(vf_list), dtype=np.float64), np.asarray(vf_list, dtype=np.float64))
                    )
                    video_frame = int(round(vf_interp))
                    timestamp_s = vf_interp / src_fps
                else:
                    vf_list = list(range(t_len))
                    video_frame = step_idx
                    timestamp_s = step_float / src_fps
            else:
                vf_list = None
                video_frame = step_idx
                timestamp_s = step_float / src_fps

            results.append(
                EndRoundResult(
                    end_round_frame=video_frame,
                    end_round_timestamp=timestamp_s,
                    confidence=float(decoded["confidence"]),
                    model_step_index=step_idx,
                    model_step_float=step_float,
                    viterbi_transition_step=int(decoded["viterbi_transition_frame"]),
                    boundary_peak_step=int(decoded["boundary_peak_frame"]),
                    src_fps=src_fps,
                    num_sampled_frames=t_len,
                    decoded_states=decoded["decoded_states"] if return_profiles else None,
                    state_probs=decoded["state_probs"] if return_profiles else None,
                    boundary_probs=decoded["boundary_probs"] if return_profiles else None,
                    sampled_video_frames=vf_list if return_profiles else None,
                )
            )
        return results

    def predict_tensor(
        self,
        clip_tchw: torch.Tensor,
        *,
        src_fps: float | None = None,
        sampled_video_frames: Sequence[int] | None = None,
        return_profiles: bool = False,
    ) -> EndRoundResult:
        """Predict END_ROUND on a single ``(T, 3, H, W)`` clip tensor."""
        results = self.predict_batch(
            clip_tchw.unsqueeze(0),
            lengths=torch.tensor([int(clip_tchw.shape[0])], dtype=torch.long),
            fps_list=[src_fps if src_fps is not None else self.target_fps],
            sampled_video_frames_list=[sampled_video_frames] if sampled_video_frames is not None else None,
            return_profiles=return_profiles,
        )
        return results[0]

    def predict_frames(
        self,
        frames: Sequence[np.ndarray],
        *,
        is_bgr: bool = True,
        already_cropped: bool = False,
        fps: float | None = None,
        return_profiles: bool = False,
    ) -> EndRoundResult:
        """Predict END_ROUND on in-memory decoded video/ROI frames."""
        clip_tchw = frames_sequence_to_tensor(
            frames,
            is_bgr=is_bgr,
            already_cropped=already_cropped,
            roi_base_1080p=self.roi_base_1080p,
            out_width=self.input_width,
            out_height=self.input_height,
        )
        return self.predict_tensor(
            clip_tchw,
            src_fps=fps if fps is not None else self.target_fps,
            sampled_video_frames=list(range(len(frames))),
            return_profiles=return_profiles,
        )

    def predict_video_segment(
        self,
        video_path: Path | str,
        *,
        start_frame: int | None = None,
        end_frame: int | None = None,
        start_ts: float | None = None,
        end_ts: float | None = None,
        already_cropped: bool = False,
        return_profiles: bool = False,
    ) -> EndRoundResult:
        """Decode ``[start_frame, end_frame)`` (or timestamps) from a video and predict END_ROUND.

        Returns exact source-video frame index and timestamp in ``EndRoundResult``.
        """
        if start_frame is None and start_ts is not None:
            native_fps = _probe_video_fps(video_path)
            s_frame = max(0, int(round(float(start_ts) * native_fps)))
        else:
            native_fps = None
            s_frame = max(0, int(start_frame or 0))

        if end_frame is None and end_ts is not None:
            if native_fps is None:
                native_fps = _probe_video_fps(video_path)
            e_frame: int | None = max(s_frame + 1, int(round(float(end_ts) * native_fps)))
        else:
            e_frame = end_frame

        clip_tchw, src_fps, sampled_indices = read_video_segment_rgb(
            video_path,
            start_frame=s_frame,
            end_frame=e_frame,
            target_fps=self.target_fps,
            already_cropped=already_cropped,
            roi_base_1080p=self.roi_base_1080p,
            out_width=self.input_width,
            out_height=self.input_height,
        )
        return self.predict_tensor(
            clip_tchw,
            src_fps=src_fps,
            sampled_video_frames=sampled_indices,
            return_profiles=return_profiles,
        )
