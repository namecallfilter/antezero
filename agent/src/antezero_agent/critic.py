"""A decoupled critic for PPO (``ppo.critic``): its own copy of :class:`~.model.Net`'s token encoder with an
HL-Gauss value head (a categorical over return bins; Farebrother et al. 2024) and the P(win) head its checkpoints
were trained with (unused by PPO). ``value`` is the expected bin value, in the policy value head's units. Trained
offline (the training scripts are not part of this repository); PPO only reads it (frozen)."""

from __future__ import annotations

import antezero_sim as az
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .model import GLOB_VOCAB, TOK_VOCAB, Obs, mlp

V_MIN, V_MAX, N_BINS = -0.25, 2.0, 64  # returns span about -0.13..1.86


class Critic(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.tok_emb = nn.ModuleList(nn.Embedding(v, d) for v in TOK_VOCAB)
        self.tok_lin = nn.Linear(az.TOK_FLOAT, d)
        self.glob_emb = nn.ModuleList(nn.Embedding(v, d) for v in GLOB_VOCAB)
        self.glob_lin = nn.Linear(az.GLOB_FLOAT, d)
        self.in_norm = nn.LayerNorm(d)
        layer = nn.TransformerEncoderLayer(d, cfg.n_heads, cfg.ff_mult * d, 0.0, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d)
        self.v_head = mlp(d, d, N_BINS)
        self.w_head = mlp(d, d, 1)
        edges = torch.linspace(V_MIN, V_MAX, N_BINS + 1)
        self.register_buffer("centers", (edges[:-1] + edges[1:]) / 2, persistent=False)

    def _cls(self, o: Obs, T: int | None = None) -> torch.Tensor:
        """[B, d] state encodings. ``T``: the token slots to run (default: the longest list, one GPU sync). Float
        inputs are clipped to +-6e4 and rounded to float16, as the critic's training data was."""
        T = T or max(int(o.n_tokens.max().item()), 1)
        ti, tf = o.tok_i[:, :T], o.tok_f[:, :T].clamp(-6e4, 6e4).half().float()
        x = self.tok_lin(tf)
        for k, emb in enumerate(self.tok_emb):
            x = x + emb(ti[..., k].clamp(0, TOK_VOCAB[k] - 1))
        g = self.glob_lin(o.glob_f.clamp(-6e4, 6e4).half().float())
        for k, emb in enumerate(self.glob_emb):
            g = g + emb(o.glob_i[:, k].clamp(0, GLOB_VOCAB[k] - 1))
        h = self.in_norm(torch.cat([g.unsqueeze(1), x], 1))
        pad = torch.cat([torch.zeros_like(o.n_tokens[:, None], dtype=torch.bool), torch.arange(T, device=ti.device)[None] >= o.n_tokens[:, None]], 1)
        # The encoder may run in float16 (train.py on CUDA); the rest stays float32.
        enc = self.encoder(h.to(self.encoder.layers[0].linear1.weight.dtype), src_key_padding_mask=pad).float()
        return self.out_norm(enc)[:, 0]

    def value(self, o: Obs, T: int | None = None) -> torch.Tensor:
        """[B] values."""
        return (F.softmax(self.v_head(self._cls(o, T)), -1) * self.centers).sum(-1)

    def win_prob(self, o: Obs) -> torch.Tensor:
        """[B] estimated probabilities that the run is won (not used by PPO; shown by the in-game bot)."""
        return torch.sigmoid(self.w_head(self._cls(o))).squeeze(-1)


def critic_state(path) -> dict:
    """The critic weights of a checkpoint: a critic's own ({"model": ...}) or a training
    checkpoint that holds one ("critic")."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if "critic" in ck:
        return ck["critic"]
    sd = ck["model"]
    if "v_head.0.weight" not in sd:
        raise ValueError(f"{path} holds no critic")
    return sd
