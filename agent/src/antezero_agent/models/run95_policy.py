"""RLBalatro Run 95-style entity-transformer policy over ``antezero_sim`` observations.

Matches RLBalatro's ``Policy`` design (``rlbalatro/models/policy.py``):
- Separate learnable ``readout`` CLS token + dedicated ``glob_mlp`` global token
  before the entity tokens: ``[readout, global, tok_0, ..., tok_{T-1}]``.
- Zero-initialized output heads (``op_score``, ``ptr_q0``, ``ptr_stop``) for
  stable initial policy steps under Muon.
- Shares ``Net``'s exact factored action space (op head, increasing-index card
  pointer, destination pointer) so rollouts, GAE, and PPO work identically.
"""

from __future__ import annotations

import antezero_sim as az
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from ..model import GLOB_VOCAB, MAX_HAND, Net, Obs, TOK_VOCAB


class Run95Net(Net):
    """Run 95-style entity transformer over ``antezero_sim`` observations."""

    def __init__(self, cfg: ModelConfig, fx: torch.Tensor | None = None):
        super().__init__(cfg, fx)
        d = cfg.d_model
        glob_dim = az.GLOB_FLOAT_V2 if cfg.obs_v2 else az.GLOB_FLOAT
        self.readout = nn.Parameter(torch.zeros(d))
        self.glob_mlp = nn.Sequential(nn.Linear(glob_dim, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        if hasattr(self.action_head, "zero_init_outputs"):
            self.action_head.zero_init_outputs()

    def encode(self, o: Obs, T: int | None = None, O: int | None = None):
        T = T or max(int(o.n_tokens.max().item()), MAX_HAND, 1)
        O = O or o.ops.shape[1]
        ti, tf = o.tok_i[:, :T], o.tok_f[:, :T]
        x = self.tok_lin(tf)
        if self.cfg.effects:
            x = x + self.fx_lin(self.fx[ti[..., 1].clamp(0, az.N_CENTERS - 1)])
        for k, emb in enumerate(self.tok_emb):
            x = x + emb(ti[..., k].clamp(0, TOK_VOCAB[k] - 1))
        g = self.glob_mlp(o.glob_f)
        for k, emb in enumerate(self.glob_emb):
            g = g + emb(o.glob_i[:, k].clamp(0, GLOB_VOCAB[k] - 1))
        B = x.shape[0]
        ro = self.readout.view(1, 1, -1).expand(B, 1, -1)
        h = self.in_norm(torch.cat([ro, g.unsqueeze(1), x], 1))
        ar = torch.arange(T, device=ti.device)
        keep = torch.zeros((B, 2), dtype=torch.bool, device=ti.device)
        pad = torch.cat([keep, ar[None] >= o.n_tokens[:, None]], 1)
        with torch.autocast(h.device.type, dtype=torch.bfloat16, enabled=self.cfg.bf16_encoder and h.is_cuda):
            h = self.encoder(h, src_key_padding_mask=pad)
        h = self.out_norm(h.float())
        cls, tok = h[:, 0], F.pad(h[:, 2:], (0, 0, 0, max(0, MAX_HAND - T)))
        op_e, logits = self.action_head.score_ops(cls, tok, o, O)
        return cls, tok, op_e, logits
