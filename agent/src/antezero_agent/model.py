"""Entity-transformer policy/value network over ``antezero_sim`` observations.

State: every visible game object is a token (hand cards, jokers,
consumables, shop items, pack cards, tags, the three blinds, the twelve
poker-hand levels, the remaining deck and the discard pile) and the global
features form a CLS token. A pre-LN transformer encoder mixes them.

Action (factored, see ``sim/src/obs``):

1. **op head** - scores each legal op (an op kind plus, optionally, the
   token it acts on: "buy shop slot 2", "use consumable 0", "play", ...;
   scored play candidates also carry their cards, pooled from the hand
   tokens, and their score statistics).
2. **card pointer** - for ops that take hand cards (play, discard,
   targeted consumables, "move hand card"), picks cards one at a time in
   increasing hand index, or STOP. Each card set has exactly one encoding,
   so probabilities over sets are exact. Required (forced) cards are always
   included; count bounds are enforced by masking.
3. **destination pointer** - for reordering ops (move joker / hand card),
   picks the target slot.

The log-probability of a full action is the sum over the three parts; the
same code path computes it at acting time (sampling) and at training time
(teacher-forced), so PPO ratios are exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import antezero_sim as az
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import effects
from .config import ModelConfig
from .models.action_heads import (
    EXPLORE,
    GAIN,
    MAX_CARDS,
    MAX_DEST,
    MAX_HAND,
    NEG,
    OP_BUY,
    OP_BUY_AND_USE,
    OP_END_SHOP,
    OP_PICK,
    OP_REDEEM,
    OP_SKIP_PACK,
    ORDER_BAND,
    bits,
    build_action_head,
    mlp,
    sample,
)

# Categorical vocabularies (columns of tok_i / glob_i): see sim/src/obs.
TOK_VOCAB = list(az.TOK_VOCAB)
GLOB_VOCAB = list(az.GLOB_VOCAB)


@dataclass
class Obs:
    tok_i: torch.Tensor  # [B, T, TOK_INT] long
    tok_f: torch.Tensor  # [B, T, TOK_FLOAT] float
    glob_i: torch.Tensor  # [B, GLOB_INT] long
    glob_f: torch.Tensor  # [B, GLOB_FLOAT] float
    ops: torch.Tensor  # [B, O, OP_INT] long
    ops_f: torch.Tensor  # [B, O, OP_FLOAT] float (scored play candidates)
    n_ops: torch.Tensor  # [B] long
    n_tokens: torch.Tensor  # [B] long
    n_hand: torch.Tensor  # [B] long

    @property
    def batch(self) -> int:
        return self.tok_i.shape[0]


def obs_from_numpy(d, device="cpu", final: bool = False, rows=None) -> Obs:
    """From a ``VecEnv`` (attributes; ``final``: its ``final_*`` arrays,
    ``rows``: a subset of the envs) or a ``Game.encode()`` dict (batch 1).

    On the CPU, float features share memory with the source arrays, which a
    ``VecEnv`` overwrites on every step: copy an observation that must
    outlive the next step.
    """
    single = isinstance(d, dict)
    prefix = "final_" if final else ""
    get = (lambda k: d[k]) if single else (lambda k: getattr(d, prefix + k))

    def t(k, dtype):
        a = np.asarray(get(k))
        if single:
            a = a[None] if a.ndim else np.asarray([a])
        elif rows is not None:
            a = a[rows]
        return torch.as_tensor(a).to(device=device, dtype=dtype)

    return Obs(
        tok_i=t("tok_i", torch.long),
        tok_f=t("tok_f", torch.float32),
        glob_i=t("glob_i", torch.long),
        glob_f=t("glob_f", torch.float32),
        ops=t("ops", torch.long),
        ops_f=t("ops_f", torch.float32),
        n_ops=t("n_ops", torch.long),
        n_tokens=t("n_tokens", torch.long),
        n_hand=t("n_hand", torch.long),
    )


def noise_shapes(batch: int, *explore: float) -> list[tuple[tuple[int, ...], bool]]:
    """(shape, uniform) of :meth:`Net.act`'s random draws when sampling, in draw order: Exp(1)
    noise for the op, a U(0, 1) pair per exploration that is on (``explore``), then Exp(1)
    for the five card pointer steps and the destination."""
    s = [((batch, az.MAX_OPS), False)]
    for x in explore:
        s += [((batch,), True), ((batch, az.MAX_OPS), True)] if x else []
    return s + [((batch, MAX_HAND + 1), False)] * MAX_CARDS + [((batch, MAX_DEST), False)]


class Net(nn.Module):
    def __init__(self, cfg: ModelConfig, fx: torch.Tensor | None = None):
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        self.tok_emb = nn.ModuleList(nn.Embedding(v, d) for v in TOK_VOCAB)
        self.tok_lin = nn.Linear(az.TOK_FLOAT, d)
        self.glob_emb = nn.ModuleList(nn.Embedding(v, d) for v in GLOB_VOCAB)
        glob_dim = az.GLOB_FLOAT_V2 if cfg.obs_v2 else az.GLOB_FLOAT
        self.glob_lin = nn.Linear(glob_dim, d)
        self.in_norm = nn.LayerNorm(d)
        layer = nn.TransformerEncoderLayer(
            d, cfg.n_heads, cfg.ff_mult * d, cfg.dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d)
        head = build_action_head(cfg)
        if cfg.action_head == "factored":
            # Keep action-head submodules registered directly on `self` for state_dict compatibility,
            # while delegating all action-head logic to `self.action_head`.
            for k in (
                "op_kind", "op_tgt", "op_feat", "op_cards", "op_score",
                "ptr_q0", "ptr_k", "ptr_upd", "ptr_norm", "ptr_stop",
                "dst_q", "dst_k", "dst_pos",
            ):
                setattr(self, k, getattr(head, k))
            object.__setattr__(self, "action_head", head)
        else:
            self.action_head = head
        self.value = mlp(d, d, 1)
        self.scale = 1.0 / math.sqrt(d)
        # What each item does, added to its token: `fx` is a checkpoint's own table, else today's (effects.py).
        self.register_buffer("fx", effects.table(cfg.effects) if fx is None else fx)
        self.register_buffer("tryable", effects.tryable(), persistent=False)
        self.fx_lin = nn.Linear(self.fx.shape[1], d, bias=False)
        nn.init.zeros_(self.fx_lin.weight)

    def encode(self, o: Obs, T: int | None = None, O: int | None = None):
        """-> (cls [B,d], tokens [B,max(T,32),d], op embeddings [B,O,d], op logits [B,O]). T, O:
        the token and op slots to run (training passes each chunk's longest lists)."""
        T = T or max(int(o.n_tokens.max().item()), MAX_HAND, 1)
        O = O or o.ops.shape[1]
        ti, tf = o.tok_i[:, :T], o.tok_f[:, :T]
        x = self.tok_lin(tf) + self.fx_lin(self.fx[ti[..., 1].clamp(0, az.N_CENTERS - 1)])
        for k, emb in enumerate(self.tok_emb):
            x = x + emb(ti[..., k].clamp(0, TOK_VOCAB[k] - 1))
        g = self.glob_lin(o.glob_f)
        for k, emb in enumerate(self.glob_emb):
            g = g + emb(o.glob_i[:, k].clamp(0, GLOB_VOCAB[k] - 1))
        h = self.in_norm(torch.cat([g.unsqueeze(1), x], 1))
        ar = torch.arange(T, device=ti.device)
        pad = torch.cat([torch.zeros_like(o.n_tokens[:, None], dtype=torch.bool), ar[None] >= o.n_tokens[:, None]], 1)
        with torch.autocast(h.device.type, dtype=torch.bfloat16, enabled=self.cfg.bf16_encoder and h.is_cuda):
            h = self.encoder(h, src_key_padding_mask=pad)
        h = self.out_norm(h.float())
        # The pointers read 32 hand slots: zeros past T (slots past n_tokens are masked wherever read).
        cls, tok = h[:, 0], F.pad(h[:, 1:], (0, 0, 0, max(0, MAX_HAND - T)))
        op_e, logits = self.action_head.score_ops(cls, tok, o, O)
        return cls, tok, op_e, logits

    def _card_steps(self, cls, tok, op_e_a, op_row, n_hand, choose):
        return self.action_head.card_steps(cls, tok, op_e_a, op_row, n_hand, choose)

    def _dest_logits(self, cls, tok, op_e_a, gain, op_row, card_seq):
        return self.action_head.dest_logits(cls, tok, op_e_a, gain, op_row, card_seq)

    def _dest(self, cls, tok, op_e_a, gain, op_row, card_seq, choose):
        return self.action_head.dest(cls, tok, op_e_a, gain, op_row, card_seq, choose)

    @torch.no_grad()
    def act(self, o: Obs, deterministic: bool = False, explore: tuple[float, ...] = (0.0, 0.0), T: int | None = None, noise=None):
        """Sample (or argmax) a full action for each state."""
        cls, tok, op_e, logits = self.encode(o, T)
        out = self.action_head.act(
            cls, tok, op_e, logits, o, deterministic=deterministic, explore=explore, noise=noise, tryable=self.tryable
        )
        out["value"] = self.value(cls).squeeze(-1)
        return out

    def evaluate(self, o: Obs, op, card_seq, dest, T: int | None = None, O: int | None = None, enc=None):
        """Teacher-forced log-prob, entropy and value of stored actions."""
        cls, tok, op_e, logits = enc or self.encode(o, T, O)
        logp, ent = self.action_head.evaluate(cls, tok, op_e, logits, o, op, card_seq, dest)
        value = self.value(cls).squeeze(-1)
        return logp, ent, value

    def order_aux(self, o: Obs, dest_bits: torch.Tensor, T: int | None = None, O: int | None = None, enc=None):
        """The ordering loss (``ppo.aux_order``) of states with a joker-order label."""
        if enc is None:
            with torch.no_grad():
                enc = self.encode(o, T, O)
        return self.action_head.order_aux(enc[0], enc[1], enc[2], o, dest_bits)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
