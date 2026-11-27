"""Inference, CTC decoding, Copy-button localization, and video-tail Seed extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

from classifiers.common.video_utils import scale_roi_pixels
from classifiers.hud_ocr.model import HudOcrCRNN
from classifiers.hud_ocr.pipeline import preprocess_crops_to_tensor

try:
    from .config import (
        COPY_BUTTON_TO_SEED_OFFSET_1080P,
        DEFAULT_SEED_CONF_THRESHOLD,
        DEFAULT_WEIGHT_CANDIDATES,
        SEED_BLANK_IDX,
        SEED_CHAR_TO_IDX,
        SEED_IDX_TO_CHAR,
        SEED_LENGTH,
        SEED_NUM_CLASSES,
        SEED_ROIS,
        SEED_VOCAB,
    )
except ImportError:
    from config import (  # type: ignore[no-redef]
        COPY_BUTTON_TO_SEED_OFFSET_1080P,
        DEFAULT_SEED_CONF_THRESHOLD,
        DEFAULT_WEIGHT_CANDIDATES,
        SEED_BLANK_IDX,
        SEED_CHAR_TO_IDX,
        SEED_IDX_TO_CHAR,
        SEED_LENGTH,
        SEED_NUM_CLASSES,
        SEED_ROIS,
        SEED_VOCAB,
    )


@dataclass(slots=True)
class SeedOcrResult:
    """Decoded prediction for a single cropped Seed ROI."""

    text: str
    is_valid_seed: bool
    confidence: float
    top_candidates: list[tuple[str, float]] = field(default_factory=list)


@dataclass(slots=True)
class SeedExtractionResult:
    """Consensus seed extracted from an end-of-run VOD overlay."""

    seed: str
    confidence: float
    matched_frames: int
    sampled_frames: int
    top_candidates: list[tuple[str, float]] = field(default_factory=list)


def is_valid_seed(text: str, length: int = SEED_LENGTH) -> bool:
    """Returns True if `text` is an exact 8-char Balatro random seed."""
    return len(text) == length and all(ch in SEED_CHAR_TO_IDX for ch in text)


def resolve_seed_weights_path(weights_path: str | Path | None = None) -> Path | None:
    """Returns Seed OCR checkpoint path if present on disk, else None."""
    if weights_path is not None:
        p = Path(weights_path)
        return p if p.exists() else None
    for candidate in DEFAULT_WEIGHT_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def ctc_greedy_decode_seed(
    log_probs: torch.Tensor,
    blank_idx: int = SEED_BLANK_IDX,
) -> list[tuple[str, float]]:
    """Greedy CTC decoder strictly using `SEED_IDX_TO_CHAR` (never `hud_ocr` numeric vocab)."""
    if log_probs.ndim != 3:
        raise ValueError(f"Expected (B, T, C) log_probs, got shape {tuple(log_probs.shape)}")
    probs = log_probs.exp()
    best_probs, best_ids = probs.max(dim=-1)
    best_ids_cpu = best_ids.detach().cpu().tolist()
    best_probs_cpu = best_probs.detach().cpu().tolist()

    out: list[tuple[str, float]] = []
    for seq_ids, seq_probs in zip(best_ids_cpu, best_probs_cpu):
        chars: list[str] = []
        confs: list[float] = []
        prev_id = blank_idx
        for idx, prob in zip(seq_ids, seq_probs):
            if idx != blank_idx and idx != prev_id:
                ch = SEED_IDX_TO_CHAR.get(int(idx), "")
                if ch:
                    chars.append(ch)
                    confs.append(float(prob))
            prev_id = int(idx)
        text = "".join(chars)
        confidence = (
            float(sum(confs) / len(confs))
            if confs
            else float(max(seq_probs) if seq_probs else 0.0)
        )
        out.append((text, confidence))
    return out


def _logsumexp2(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = a if a >= b else b
    return m + math.log1p(math.exp(-abs(a - b)))


def ctc_beam_decode_seed(
    log_probs: torch.Tensor,
    *,
    beam_width: int = 8,
    top_k: int = 4,
    blank_idx: int = SEED_BLANK_IDX,
) -> list[list[tuple[str, float]]]:
    """Prefix beam search over `(B, T, C)` log-probs returning top-K `(text, prob)` candidates."""
    if log_probs.ndim != 3:
        raise ValueError(f"Expected (B, T, C) log_probs, got shape {tuple(log_probs.shape)}")
    lp_cpu = log_probs.detach().cpu().tolist()
    batch_results: list[list[tuple[str, float]]] = []

    for sample_lp in lp_cpu:
        # prefix -> (log_p_blank, log_p_non_blank)
        beams: dict[str, tuple[float, float]] = {"": (0.0, -math.inf)}
        for t_lp in sample_lp:
            # Prune step tokens per timestep to keep Python beam search fast
            ranked_tokens = sorted(
                enumerate(t_lp), key=lambda kv: kv[1], reverse=True
            )[: max(6, beam_width)]
            if all(cid != blank_idx for cid, _ in ranked_tokens):
                ranked_tokens.append((blank_idx, t_lp[blank_idx]))

            nxt: dict[str, list[float]] = {}
            for prefix, (p_b, p_nb) in beams.items():
                p_tot = _logsumexp2(p_b, p_nb)
                for cid, tok_lp in ranked_tokens:
                    if cid == blank_idx:
                        entry = nxt.setdefault(prefix, [-math.inf, -math.inf])
                        entry[0] = _logsumexp2(entry[0], p_tot + tok_lp)
                        continue
                    ch = SEED_IDX_TO_CHAR.get(int(cid), "")
                    if not ch:
                        continue
                    if prefix and prefix[-1] == ch:
                        # Extending same char stays collapsed if previous path ended non-blank,
                        # or appends a second copy only via blank.
                        same_entry = nxt.setdefault(prefix, [-math.inf, -math.inf])
                        same_entry[1] = _logsumexp2(same_entry[1], p_nb + tok_lp)
                        ext_entry = nxt.setdefault(prefix + ch, [-math.inf, -math.inf])
                        ext_entry[1] = _logsumexp2(ext_entry[1], p_b + tok_lp)
                    else:
                        ext_entry = nxt.setdefault(prefix + ch, [-math.inf, -math.inf])
                        ext_entry[1] = _logsumexp2(ext_entry[1], p_tot + tok_lp)

            scored = [
                (pref, (vals[0], vals[1]), _logsumexp2(vals[0], vals[1]))
                for pref, vals in nxt.items()
            ]
            scored.sort(key=lambda item: item[2], reverse=True)
            beams = {pref: pair for pref, pair, _ in scored[: max(1, beam_width)]}

        final_scored = [
            (pref, float(math.exp(_logsumexp2(pb, pnb))))
            for pref, (pb, pnb) in beams.items()
        ]
        final_scored.sort(key=lambda item: item[1], reverse=True)
        batch_results.append(final_scored[: max(1, top_k)])

    return batch_results


class SeedOcrPipeline:
    """High-throughput batched inference wrapper for dedicated Balatro Seed OCR."""

    def __init__(
        self,
        weights_path: str | Path | None = None,
        device: str | None = None,
        batch_size: int = 64,
        conf_threshold: float = DEFAULT_SEED_CONF_THRESHOLD,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.batch_size = max(1, int(batch_size))
        self.conf_threshold = float(conf_threshold)

        resolved = resolve_seed_weights_path(weights_path)
        if resolved is None:
            searched = ", ".join(str(c) for c in DEFAULT_WEIGHT_CANDIDATES)
            raise FileNotFoundError(
                f"Could not find Seed OCR weights. Searched: {searched}"
            )
        self.weights_path = resolved

        ckpt = torch.load(
            str(self.weights_path), map_location=self.device, weights_only=False
        )
        model_cfg = ckpt.get("model_config", {}) if isinstance(ckpt, dict) else {}
        vocab = ckpt.get("vocab") if isinstance(ckpt, dict) else None
        if vocab is not None and tuple(vocab) != SEED_VOCAB:
            raise ValueError(
                f"Seed OCR checkpoint vocab mismatch (len={len(vocab)} vs expected {len(SEED_VOCAB)})"
            )
        self.model = HudOcrCRNN(
            num_classes=int(model_cfg.get("num_classes", SEED_NUM_CLASSES)),
            rnn_hidden=int(model_cfg.get("rnn_hidden", 128)),
            rnn_layers=int(model_cfg.get("rnn_layers", 1)),
            dropout=0.0,
        ).to(self.device)
        state_dict = (
            ckpt["model_state_dict"]
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt
            else ckpt
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.inference_mode()
    def predict_crops(
        self,
        crops_rgb: Sequence[np.ndarray],
        *,
        conf_threshold: float | None = None,
        beam_top_k: int = 4,
    ) -> list[SeedOcrResult]:
        if not crops_rgb:
            return []
        thresh = self.conf_threshold if conf_threshold is None else float(conf_threshold)
        results: list[SeedOcrResult] = []
        for start in range(0, len(crops_rgb), self.batch_size):
            chunk = crops_rgb[start : start + self.batch_size]
            x = preprocess_crops_to_tensor(chunk).to(self.device, non_blocking=True)
            log_probs = self.model(x)
            greedy = ctc_greedy_decode_seed(log_probs)
            beams = (
                ctc_beam_decode_seed(log_probs, top_k=beam_top_k)
                if beam_top_k > 1
                else [[(txt, conf)] for txt, conf in greedy]
            )
            for (text, conf), top_cands in zip(greedy, beams):
                valid = is_valid_seed(text) and conf >= thresh
                results.append(
                    SeedOcrResult(
                        text=text,
                        is_valid_seed=valid,
                        confidence=float(conf),
                        top_candidates=top_cands,
                    )
                )
        return results

    def predict_crop(
        self, crop_rgb: np.ndarray, *, conf_threshold: float | None = None, beam_top_k: int = 4
    ) -> SeedOcrResult:
        return self.predict_crops(
            [crop_rgb], conf_threshold=conf_threshold, beam_top_k=beam_top_k
        )[0]


def detect_copy_button_box(frame_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """Detects Balatro's blue `Copy` button beneath the Seed row on Win/Game Over modals."""
    if frame_bgr is None or frame_bgr.size == 0 or frame_bgr.ndim != 3:
        return None
    fh, fw = frame_bgr.shape[:2]
    if fw < 320 or fh < 180:
        return None
    sx = fw / 1920.0
    sy = fh / 1080.0

    # Restrict search to central modal lower region
    x0 = max(0, int(0.40 * fw))
    x1 = min(fw, int(0.72 * fw))
    y0 = max(0, int(0.55 * fh))
    y1 = min(fh, int(0.88 * fh))
    roi = frame_bgr[y0:y1, x0:x1]
    b = roi[:, :, 0].astype(np.int16)
    g = roi[:, :, 1].astype(np.int16)
    r = roi[:, :, 2].astype(np.int16)
    mask = ((b > 175) & (g > 90) & (g < 205) & (r < 85) & ((b - r) > 105)).astype(
        np.uint8
    )

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    best_box: tuple[int, int, int, int] | None = None
    best_score = float("inf")
    for idx in range(1, num_labels):
        bx, by, bw, bh, area = (int(v) for v in stats[idx])
        norm_w = bw / sx
        norm_h = bh / sy
        if not (115.0 <= norm_w <= 265.0 and 18.0 <= norm_h <= 58.0):
            continue
        aspect = norm_w / max(1e-6, norm_h)
        if not (3.2 <= aspect <= 8.0):
            continue
        fill = area / max(1, bw * bh)
        if fill < 0.58:
            continue
        # Score similarity to canonical 1080p Copy pill (w=191, h=34)
        size_err = abs(norm_w - 191.0) / 191.0 + abs(norm_h - 34.0) / 34.0
        if size_err < best_score:
            best_score = size_err
            best_box = (x0 + bx, y0 + by, bw, bh)
    return best_box


def crop_seed_from_frame(
    frame_bgr: np.ndarray, *, roi_fallback: str = "seed_win"
) -> tuple[np.ndarray, tuple[int, int, int, int], bool]:
    """Crops the RGB Seed pill image anchored via `Copy` button, or fallback ROI."""
    fh, fw = frame_bgr.shape[:2]
    copy_box = detect_copy_button_box(frame_bgr)
    used_copy_anchor = copy_box is not None
    if copy_box is not None:
        bx, by, _, _ = copy_box
        sx = fw / 1920.0
        sy = fh / 1080.0
        rx = int(round(bx + COPY_BUTTON_TO_SEED_OFFSET_1080P["dx"] * sx))
        ry = int(round(by + COPY_BUTTON_TO_SEED_OFFSET_1080P["dy"] * sy))
        rw = max(16, int(round(COPY_BUTTON_TO_SEED_OFFSET_1080P["w"] * sx)))
        rh = max(12, int(round(COPY_BUTTON_TO_SEED_OFFSET_1080P["h"] * sy)))
    else:
        roi_def = SEED_ROIS.get(roi_fallback, SEED_ROIS["seed_win"])
        rx, ry, rw, rh = scale_roi_pixels(roi_def, fw, fh, 1920.0, 1080.0)

    rx = max(0, min(fw - 1, rx))
    ry = max(0, min(fh - 1, ry))
    rx2 = max(rx + 1, min(fw, rx + rw))
    ry2 = max(ry + 1, min(fh, ry + rh))
    crop_bgr = frame_bgr[ry:ry2, rx:rx2]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return crop_rgb, (rx, ry, rx2 - rx, ry2 - ry), used_copy_anchor


def _aggregate_seed_predictions(
    preds: Sequence[SeedOcrResult],
) -> tuple[str, float, int, list[tuple[str, float]]]:
    """Combines multiple frame predictions via per-character confidence-weighted voting."""
    valid = [p for p in preds if is_valid_seed(p.text) and p.confidence > 0.0]
    if not valid:
        # Collect any partial beam candidates if all length-8 checks failed
        return "", 0.0, 0, []

    per_pos_votes: list[dict[str, float]] = [{} for _ in range(SEED_LENGTH)]
    seq_scores: dict[str, float] = {}
    total_w = 0.0
    for p in valid:
        w = max(1e-4, float(p.confidence))
        total_w += w
        seq_scores[p.text] = seq_scores.get(p.text, 0.0) + w
        for pos, ch in enumerate(p.text):
            per_pos_votes[pos][ch] = per_pos_votes[pos].get(ch, 0.0) + w

    consensus_chars: list[str] = []
    pos_margins: list[float] = []
    for pos in range(SEED_LENGTH):
        votes = per_pos_votes[pos]
        best_ch = max(votes, key=lambda c: votes[c])
        consensus_chars.append(best_ch)
        pos_margins.append(votes[best_ch] / max(1e-6, total_w))

    consensus_seed = "".join(consensus_chars)
    mean_conf = float(sum(p.confidence for p in valid) / len(valid)) * float(
        sum(pos_margins) / len(pos_margins)
    )
    ranked_seqs = sorted(
        ((s, w / max(1e-6, total_w)) for s, w in seq_scores.items()),
        key=lambda kv: kv[1],
        reverse=True,
    )[:4]
    return consensus_seed, mean_conf, len(valid), ranked_seqs


def extract_seed_from_video(
    video_path: str | Path,
    pipeline: SeedOcrPipeline | None = None,
    *,
    device: str | None = None,
    scan_tail_seconds: float = 420.0,
    sample_interval_sec: float = 1.0,
    burst_frames: int = 12,
    min_conf: float = DEFAULT_SEED_CONF_THRESHOLD,
) -> SeedExtractionResult | None:
    """Reverse-scans the tail of a VOD for the Win/Game Over `Copy` modal and decodes its Seed."""
    pipe = pipeline or SeedOcrPipeline(device=device)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 60.0)
        if total <= 0:
            return None
        step = max(1, int(round(fps * sample_interval_sec)))
        start_limit = max(0, total - int(round(fps * scan_tail_seconds)))

        anchor_frame: int | None = None
        for fid in range(total - 1, start_limit - 1, -step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, img = cap.read()
            if ok and detect_copy_button_box(img) is not None:
                anchor_frame = fid
                break
        if anchor_frame is None:
            return None

        # Sample a burst around the detected modal frame for consensus voting across DynaText float
        half_span = int(round(fps * 4.0))
        lo = max(0, anchor_frame - half_span)
        hi = min(total - 1, anchor_frame + half_span)
        stride = max(1, (hi - lo) // max(1, burst_frames - 1))
        probe_ids = list(range(lo, hi + 1, stride))[:burst_frames]

        crops: list[np.ndarray] = []
        for fid in probe_ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, img = cap.read()
            if not ok:
                continue
            crop_rgb, _, used_anchor = crop_seed_from_frame(img)
            if used_anchor:
                crops.append(crop_rgb)
    finally:
        cap.release()

    if not crops:
        return None

    preds = pipe.predict_crops(crops, conf_threshold=min_conf * 0.65)
    seed, conf, matched, top_cands = _aggregate_seed_predictions(preds)
    if not seed or conf < min_conf * 0.70:
        return None
    return SeedExtractionResult(
        seed=seed,
        confidence=conf,
        matched_frames=matched,
        sampled_frames=len(crops),
        top_candidates=top_cands,
    )
