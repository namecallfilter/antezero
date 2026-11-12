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

from .config import ModelConfig

MAX_CARDS = 5  # a card set has at most five cards
MAX_HAND = az.MAX_HAND
MAX_DEST = 32
NEG = -1e9
OP_REDEEM, OP_END_SHOP = 10, 13  # sim/src/obs/mod.rs op:: (not exported by the bindings)

# Categorical vocabularies (columns of tok_i / glob_i): see sim/src/obs.
TOK_VOCAB = list(az.TOK_VOCAB)
GLOB_VOCAB = list(az.GLOB_VOCAB)


@dataclass
class Obs:
    """A batch of observations as torch tensors (see :func:`obs_from_numpy`)."""

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


def bits(x: torch.Tensor, n: int = 32) -> torch.Tensor:
    """[...] int (u32 stored as i32) -> [..., n] bool."""
    x = x & 0xFFFFFFFF
    return ((x.unsqueeze(-1) >> torch.arange(n, device=x.device)) & 1).bool()


def mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


def sample(logits: torch.Tensor, q: torch.Tensor | None = None) -> torch.Tensor:
    """``Categorical(logits=logits).sample()``, bitwise, without its two host syncs per call
    (``multinomial`` validates its input on the host, then draws ``argmax(p / Exp(1))``).
    ``q``: that Exp(1) noise, drawn here when None (CUDA graphs pass noise drawn before replay)."""
    p = F.softmax(logits - logits.logsumexp(-1, keepdim=True), -1)
    if q is None:
        q = torch.empty_like(p).exponential_(1)
    return (p / q).argmax(-1)


def noise_shapes(batch: int, *explore: float) -> list[tuple[tuple[int, ...], bool]]:
    """(shape, uniform) of :meth:`Net.act`'s random draws when sampling, in draw order: Exp(1)
    noise for the op, a U(0, 1) pair per exploration that is on (``explore``), then Exp(1)
    for the five card pointer steps and the destination."""
    s = [((batch, az.MAX_OPS), False)]
    for x in explore:
        s += [((batch,), True), ((batch, az.MAX_OPS), True)] if x else []
    return s + [((batch, MAX_HAND + 1), False)] * MAX_CARDS + [((batch, MAX_DEST), False)]


class Net(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        self.tok_emb = nn.ModuleList(nn.Embedding(v, d) for v in TOK_VOCAB)
        self.tok_lin = nn.Linear(az.TOK_FLOAT, d)
        self.glob_emb = nn.ModuleList(nn.Embedding(v, d) for v in GLOB_VOCAB)
        self.glob_lin = nn.Linear(az.GLOB_FLOAT, d)
        self.in_norm = nn.LayerNorm(d)
        layer = nn.TransformerEncoderLayer(
            d, cfg.n_heads, cfg.ff_mult * d, cfg.dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d)
        # op head
        self.op_kind = nn.Embedding(az.N_OPS_KINDS, d)
        self.op_tgt = nn.Linear(d, d)
        self.op_feat = nn.Linear(5 + az.OP_FLOAT, d)
        self.op_cards = nn.Linear(d, d)
        self.op_score = mlp(2 * d, d, 1)
        # card pointer
        self.ptr_q0 = mlp(3 * d, d, d)
        self.ptr_k = nn.Linear(d, d)
        self.ptr_upd = nn.Linear(d, d)
        self.ptr_norm = nn.LayerNorm(d)
        self.ptr_stop = nn.Linear(d, 1)
        # destination pointer
        self.dst_q = mlp(3 * d, d, d)
        self.dst_k = nn.Linear(d, d)
        self.dst_pos = nn.Embedding(MAX_DEST, d)
        # value
        self.value = mlp(d, d, 1)
        self.scale = 1.0 / math.sqrt(d)

    # ------------------------------------------------------------------ encode
    def encode(self, o: Obs, T: int | None = None, O: int | None = None):
        """-> (cls [B,d], tokens [B,max(T,32),d], op embeddings [B,O,d], op logits [B,O]). T, O:
        the token and op slots to run (training passes each chunk's longest lists). The default,
        the longest token list but at least 32 and every op slot, keeps acting bitwise as before."""
        T = T or max(int(o.n_tokens.max().item()), MAX_HAND, 1)
        O = O or o.ops.shape[1]
        ti, tf = o.tok_i[:, :T], o.tok_f[:, :T]
        x = self.tok_lin(tf)
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
        T = tok.shape[1]
        # ops
        ops = o.ops[:, :O]
        kind = ops[..., 0].clamp(0, az.N_OPS_KINDS - 1)
        tgt = ops[..., 1]
        tgt_h = torch.gather(tok, 1, tgt.clamp(0, T - 1).unsqueeze(-1).expand(-1, -1, tok.shape[-1]))
        tgt_h = tgt_h * (tgt >= 0).unsqueeze(-1)
        feats = torch.stack(
            [
                ops[..., 2].float() / 5,
                ops[..., 3].float() / 5,
                ops[..., 6].float() / 10,
                bits(ops[..., 4]).sum(-1).float() / 10,
                bits(ops[..., 5]).sum(-1).float(),
            ],
            -1,
        )
        feats = torch.cat([feats, o.ops_f[:, :O]], -1)
        # Cards an op is fixed to (scored play candidates, forced cards):
        # mean of their hand-token embeddings.
        req = bits(ops[..., 5], MAX_HAND).to(tok.dtype)  # [B,O,32]
        cards_h = torch.einsum("boj,bjd->bod", req, tok[:, :MAX_HAND]) / req.sum(-1, keepdim=True).clamp(min=1)
        op_e = self.op_kind(kind) + self.op_tgt(tgt_h) + self.op_feat(feats) + self.op_cards(cards_h)
        logits = self.op_score(torch.cat([op_e, cls.unsqueeze(1).expand(-1, O, -1)], -1)).squeeze(-1)
        valid = torch.arange(O, device=ops.device)[None] < o.n_ops[:, None]
        logits = logits.masked_fill(~valid, NEG)
        return cls, tok, op_e, logits

    # ---------------------------------------------------------------- pointers
    def _card_steps(self, cls, tok, op_e_a, op_row, n_hand, choose):
        """Run the card pointer. ``choose(step, logits, mask) -> idx`` returns
        the choice per row (0..31 = card, 32 = STOP). Returns (mask u32 as
        long, sequence [B,5] with -1 = stop/none, logp [B], entropy [B])."""
        B, d = cls.shape
        dev = cls.device
        cmin, cmax = op_row[:, 2], op_row[:, 3]
        allowed, required = bits(op_row[:, 4]), bits(op_row[:, 5])
        hand_valid = torch.arange(MAX_HAND, device=dev)[None] < n_hand[:, None]
        cand = allowed & ~required & hand_valid  # [B,32]
        hand = tok[:, :MAX_HAND]
        keys = self.ptr_k(hand)  # [B,32,d]
        req_sum = (hand * required.unsqueeze(-1)).sum(1)
        q = self.ptr_q0(torch.cat([cls, op_e_a, req_sum], -1))
        count = required.sum(-1)
        last = torch.full((B,), -1, device=dev, dtype=torch.long)
        active = cmax > 0
        chosen = torch.zeros(B, MAX_HAND, dtype=torch.bool, device=dev)
        seq = torch.full((B, MAX_CARDS), -1, dtype=torch.long, device=dev)
        logp = torch.zeros(B, device=dev)
        ent = torch.zeros(B, device=dev)
        ar = torch.arange(MAX_HAND, device=dev)[None]
        # Number of candidate cards after each index (to keep `card_min`
        # reachable when cards are taken in increasing index order).
        after = cand.flip(-1).long().cumsum(-1).flip(-1) - cand.long()
        for s in range(MAX_CARDS):
            can_card = active & (count < cmax)
            feasible = (count[:, None] + 1 + after) >= cmin[:, None]
            cmask = cand & (ar > last[:, None]) & can_card[:, None] & feasible
            can_stop = active & (count >= cmin)
            # Rows without any option stop without a decision; rows with a
            # single option contribute log 1 = 0.
            decide = active & (cmask.any(-1) | can_stop)
            lc = torch.einsum("bd,bjd->bj", self.ptr_norm(q), keys) * self.scale
            ls = self.ptr_stop(self.ptr_norm(q)).squeeze(-1)
            lg = torch.cat([lc, ls.unsqueeze(-1)], -1)
            m = torch.cat([cmask, can_stop.unsqueeze(-1)], -1)
            lg = lg.masked_fill(~m, NEG)
            idx = choose(s, lg, m)
            lsm = F.log_softmax(lg, -1)
            p = lsm.exp()
            step_logp = lsm.gather(1, idx.clamp(0, MAX_HAND).unsqueeze(-1)).squeeze(-1)
            step_ent = -(p * lsm).masked_fill(~m, 0).sum(-1)
            logp = logp + torch.where(decide, step_logp, torch.zeros_like(step_logp))
            ent = ent + torch.where(decide, step_ent, torch.zeros_like(step_ent))
            is_card = active & (idx < MAX_HAND) & cmask.gather(1, idx.clamp(0, MAX_HAND - 1).unsqueeze(-1)).squeeze(-1)
            seq[:, s] = torch.where(is_card, idx, torch.full_like(idx, -1))
            pick = F.one_hot(idx.clamp(0, MAX_HAND - 1), MAX_HAND).bool() & is_card[:, None]
            chosen = chosen | pick
            upd = torch.einsum("bj,bjd->bd", pick.float(), hand)
            q = q + self.ptr_upd(upd)
            count = count + is_card.long()
            last = torch.where(is_card, idx, last)
            active = active & is_card
        mask = (chosen.long() << torch.arange(MAX_HAND, device=dev)).sum(-1)
        return mask, seq, logp, ent

    def _dest(self, cls, tok, op_e_a, op_row, card_seq, choose):
        B, d = cls.shape
        dev = cls.device
        kind, tgt, ndest, index = op_row[:, 0], op_row[:, 1], op_row[:, 6], op_row[:, 7]
        active = ndest > 0
        is_hand = kind == az.OP_MOVE_HAND
        start = torch.where(is_hand, torch.zeros_like(tgt), (tgt - index).clamp(min=0))
        frm = torch.where(is_hand, card_seq[:, 0].clamp(min=0), index)
        T = tok.shape[1]
        pos = torch.arange(MAX_DEST, device=dev)[None]
        gidx = (start[:, None] + pos).clamp(0, T - 1)
        keys = self.dst_k(torch.gather(tok, 1, gidx.unsqueeze(-1).expand(-1, -1, d))) + self.dst_pos(pos.expand(B, -1))
        src = torch.gather(tok, 1, (start + frm).clamp(0, T - 1)[:, None, None].expand(-1, 1, d)).squeeze(1)
        q = self.dst_q(torch.cat([cls, op_e_a, src], -1))
        lg = torch.einsum("bd,bjd->bj", q, keys) * self.scale
        m = (pos < ndest[:, None]) & (pos != frm[:, None]) & active[:, None]
        m = m | (~active[:, None] & (pos == 0))  # dummy choice for inactive rows
        lg = lg.masked_fill(~m, NEG)
        idx = choose(lg, m)
        lsm = F.log_softmax(lg, -1)
        logp = torch.where(active, lsm.gather(1, idx.unsqueeze(-1)).squeeze(-1), torch.zeros(B, device=dev))
        ent = torch.where(active, -(lsm.exp() * lsm).masked_fill(~m, 0).sum(-1), torch.zeros(B, device=dev))
        return torch.where(active, idx, torch.zeros_like(idx)), logp, ent

    # ----------------------------------------------------------------- public
    @torch.no_grad()
    def act(self, o: Obs, deterministic: bool = False, explore: float = 0.0, T: int | None = None, noise=None):
        """Sample (or argmax) a full action for each state. ``explore``
        (sampling only): the probability that a sampled shop exit with a
        legal voucher redeems it instead (``explored``; see
        ``PPOConfig.explore_redeem``). ``T``: :meth:`encode`'s token slots
        (the default saves nothing but a GPU sync). ``noise``: the random
        draws, made beforehand in this order and these shapes (see
        :func:`noise_shapes`); None draws them here (the same stream)."""
        nxt = iter(noise).__next__ if noise is not None else lambda: None
        cls, tok, op_e, logits = self.encode(o, T)
        if deterministic:
            op = logits.argmax(-1)
        else:
            op = sample(logits, nxt())
        explored = torch.zeros_like(op, dtype=torch.bool)
        if explore and not deterministic:
            r1 = torch.rand(op.shape, device=op.device) if noise is None else nxt()
            kinds = o.ops[..., 0]
            red = (kinds == OP_REDEEM) & (logits > NEG / 2)
            r2 = torch.rand(red.shape, device=op.device) if noise is None else nxt()
            explored = (kinds.gather(1, op[:, None]).squeeze(1) == OP_END_SHOP) & red.any(-1) & (r1 < explore)
            op = torch.where(explored, (r2 + red).argmax(-1), op)
        lsm = F.log_softmax(logits, -1)
        op_logp = lsm.gather(1, op[:, None]).squeeze(1)
        # How sure the state is, whatever gets sampled (ppo.learn_frac): the log-prob of its likeliest op that
        # takes no card or destination choice (fixed cards count as none).
        rows = o.ops[:, : logits.shape[1]]
        whole = ((rows[..., 3] == 0) | (rows[..., 4] == rows[..., 5])) & (rows[..., 6] == 0)
        sure = lsm.masked_fill(~whole, NEG).amax(-1)
        B = op.shape[0]
        ar = torch.arange(B, device=op.device)
        op_e_a = op_e[ar, op]
        op_row = o.ops[ar, op]

        def pick(_s, lg, m):
            if deterministic:
                return lg.argmax(-1)
            return sample(lg, nxt())

        mask, seq, c_logp, _ = self._card_steps(cls, tok, op_e_a, op_row, o.n_hand, pick)
        dest, d_logp, _ = self._dest(cls, tok, op_e_a, op_row, seq, lambda lg, m: pick(0, lg, m))
        value = self.value(cls).squeeze(-1)
        return {
            "op": op,
            "cards": mask,
            "card_seq": seq,
            "dest": dest,
            "logp": op_logp + c_logp + d_logp,
            "sure": sure,
            "value": value,
            "explored": explored,
        }

    def evaluate(self, o: Obs, op, card_seq, dest, T: int | None = None, O: int | None = None):
        """Teacher-forced log-prob, entropy and value of stored actions."""
        cls, tok, op_e, logits = self.encode(o, T, O)
        lsm = F.log_softmax(logits, -1)
        op_logp = lsm.gather(1, op[:, None]).squeeze(1)
        valid = logits > NEG / 2
        op_ent = -(lsm.exp() * lsm).masked_fill(~valid, 0).sum(-1)
        B = op.shape[0]
        ar = torch.arange(B, device=op.device)
        op_e_a = op_e[ar, op]
        op_row = o.ops[ar, op]

        def forced(s, lg, m):
            c = card_seq[:, s]
            return torch.where(c >= 0, c, torch.full_like(c, MAX_HAND))

        _, seq, c_logp, c_ent = self._card_steps(cls, tok, op_e_a, op_row, o.n_hand, forced)
        _, d_logp, d_ent = self._dest(cls, tok, op_e_a, op_row, card_seq, lambda lg, m: dest)
        value = self.value(cls).squeeze(-1)
        return op_logp + c_logp + d_logp, op_ent + c_ent + d_ent, value


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
