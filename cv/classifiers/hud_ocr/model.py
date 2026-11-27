"""Lightweight CRNN (CNN + BiGRU + CTC) model for Balatro HUD OCR."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .config import BLANK_IDX, CHAR_TO_IDX, IDX_TO_CHAR, NUM_CLASSES
except ImportError:
    from config import BLANK_IDX, CHAR_TO_IDX, IDX_TO_CHAR, NUM_CLASSES


def encode_text(text: str) -> list[int]:
    """Encode a target string into CTC label indices."""
    encoded: list[int] = []
    for ch in text:
        if ch not in CHAR_TO_IDX:
            raise ValueError(f"Unsupported OCR character {ch!r} in text {text!r}")
        encoded.append(CHAR_TO_IDX[ch])
    return encoded


def ctc_greedy_decode(
    log_probs: torch.Tensor,
    blank_idx: int = BLANK_IDX,
    allowed_chars_per_sample: Sequence[str | None] | None = None,
) -> list[tuple[str, float]]:
    """Greedy CTC decoder for log-probabilities of shape (B, T, C).

    If `allowed_chars_per_sample` is provided, disallowed characters (other than `blank_idx`)
    are masked out per sample before greedy argmax so output stays strictly within each field's
    numeric character set.

    Returns a list of `(decoded_text, mean_token_confidence)` pairs.
    """
    if log_probs.ndim != 3:
        raise ValueError(f"Expected (B, T, C) log_probs, got shape {tuple(log_probs.shape)}")

    if allowed_chars_per_sample is not None:
        if len(allowed_chars_per_sample) != log_probs.shape[0]:
            raise ValueError(
                f"allowed_chars_per_sample length ({len(allowed_chars_per_sample)}) "
                f"must match batch size ({log_probs.shape[0]})"
            )
        masked = log_probs.clone()
        num_classes = log_probs.shape[2]
        for b_idx, allowed in enumerate(allowed_chars_per_sample):
            if not allowed:
                continue
            keep = torch.zeros(num_classes, dtype=torch.bool, device=log_probs.device)
            keep[blank_idx] = True
            for ch in allowed:
                cid = CHAR_TO_IDX.get(ch)
                if cid is not None:
                    keep[cid] = True
            masked[b_idx, :, ~keep] = float("-inf")
        probs = masked.exp()
    else:
        probs = log_probs.exp()
    best_probs, best_ids = probs.max(dim=-1)  # (B, T)
    best_ids_cpu = best_ids.detach().cpu().tolist()
    best_probs_cpu = best_probs.detach().cpu().tolist()

    results: list[tuple[str, float]] = []
    for seq_ids, seq_probs in zip(best_ids_cpu, best_probs_cpu):
        chars: list[str] = []
        confs: list[float] = []
        prev_id = blank_idx
        for idx, prob in zip(seq_ids, seq_probs):
            if idx != blank_idx:
                if idx != prev_id:
                    ch = IDX_TO_CHAR.get(int(idx), "")
                    if ch:
                        chars.append(ch)
                        confs.append(float(prob))
                elif confs:
                    confs[-1] = max(confs[-1], float(prob))
            prev_id = int(idx)
        text = "".join(chars)
        confidence = float(sum(confs) / len(confs)) if confs else float(max(seq_probs) if seq_probs else 0.0)
        results.append((text, confidence))
    return results


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: tuple[int, int] | int = 3, padding: tuple[int, int] | int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class HudOcrCRNN(nn.Module):
    """Compact CNN + BiGRU + CTC network for 40x192 HUD crops.

    Spatial schedule (H x W):
      (3, 40, 192)
      -> pool(2, 2): (32, 20, 96)
      -> pool(2, 2): (64, 10, 48)
      -> pool(2, 1): (128, 5, 48)
      -> pool(2, 1): (192, 2, 48)
      -> conv(2, 3), pad=(0, 1): (192, 1, 48)
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        rnn_hidden: int = 128,
        rnn_layers: int = 1,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.rnn_hidden = int(rnn_hidden)
        self.rnn_layers = int(rnn_layers)
        self.dropout_p = float(dropout)

        self.cnn = nn.Sequential(
            _ConvBlock(3, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            _ConvBlock(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            _ConvBlock(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            _ConvBlock(128, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            _ConvBlock(192, 192, kernel_size=(2, 3), padding=(0, 1)),
        )

        self.dropout = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()
        self.rnn = nn.GRU(
            input_size=192,
            hidden_size=self.rnn_hidden,
            num_layers=self.rnn_layers,
            bidirectional=True,
            batch_first=True,
            dropout=self.dropout_p if self.rnn_layers > 1 else 0.0,
        )
        self.head = nn.Linear(self.rnn_hidden * 2, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return log-probabilities of shape `(B, T, num_classes)`."""
        feats = self.cnn(x)  # (B, 192, 1, T)
        if feats.shape[2] != 1:
            feats = F.adaptive_avg_pool2d(feats, (1, feats.shape[3]))
        feats = feats.squeeze(2).permute(0, 2, 1).contiguous()  # (B, T, 192)
        feats = self.dropout(feats)
        seq, _ = self.rnn(feats)  # (B, T, 2 * rnn_hidden)
        seq = self.dropout(seq)
        logits = self.head(seq)  # (B, T, C)
        return F.log_softmax(logits, dim=-1)


def compute_cer(predictions: Sequence[str], targets: Sequence[str]) -> float:
    """Compute Character Error Rate (edit distance / total target characters)."""
    total_edits = 0
    total_chars = 0
    for pred, tgt in zip(predictions, targets):
        total_chars += max(1, len(tgt))
        # Standard Wagner-Fischer Levenshtein distance
        n, m = len(pred), len(tgt)
        dp = list(range(m + 1))
        for i in range(1, n + 1):
            prev = dp[0]
            dp[0] = i
            pi = pred[i - 1]
            for j in range(1, m + 1):
                cur = dp[j]
                cost = 0 if pi == tgt[j - 1] else 1
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
                prev = cur
        total_edits += dp[m]
    return float(total_edits / max(1, total_chars))
