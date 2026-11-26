"""Pluggable action-head architectures for ``antezero_agent`` policies."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import antezero_sim as az
import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from ..config import ModelConfig

MAX_CARDS = 5
MAX_HAND = az.MAX_HAND
MAX_DEST = 32
NEG = -1e9
OP_BUY, OP_BUY_AND_USE, OP_REDEEM, OP_END_SHOP, OP_PICK, OP_SKIP_PACK = 8, 9, 10, 13, 15, 16
EXPLORE = (
    (False, [((OP_END_SHOP,), (OP_REDEEM,))]),
    (True, [((OP_END_SHOP,), (OP_BUY, OP_BUY_AND_USE)), ((OP_PICK, OP_SKIP_PACK), (OP_PICK,))]),
)
GAIN = 5 + az.OP_ORDER_GAIN
ORDER_BAND = 3.0


def bits(x: torch.Tensor, n: int = 32) -> torch.Tensor:
    """[...] int (u32 stored as i32) -> [..., n] bool."""
    x = x & 0xFFFFFFFF
    return ((x.unsqueeze(-1) >> torch.arange(n, device=x.device)) & 1).bool()


def mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


def sample(logits: torch.Tensor, q: torch.Tensor | None = None) -> torch.Tensor:
    """``Categorical(logits=logits).sample()``, bitwise, without host syncs."""
    p = F.softmax(logits - logits.logsumexp(-1, keepdim=True), -1)
    if q is None:
        q = torch.empty_like(p).exponential_(1)
    return (p / q).argmax(-1)


class ActionHead(nn.Module):
    """Base class for pluggable policy action heads."""

    def score_ops(self, cls: torch.Tensor, tok: torch.Tensor, o, O: int | None = None):
        raise NotImplementedError

    def act(
        self,
        cls: torch.Tensor,
        tok: torch.Tensor,
        op_e: torch.Tensor,
        logits: torch.Tensor,
        o,
        deterministic: bool = False,
        explore: tuple[float, ...] = (0.0, 0.0),
        noise=None,
        tryable: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def evaluate(
        self,
        cls: torch.Tensor,
        tok: torch.Tensor,
        op_e: torch.Tensor,
        logits: torch.Tensor,
        o,
        op: torch.Tensor,
        card_seq: torch.Tensor,
        dest: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class FactoredActionHead(ActionHead):
    """Antezero's 3-part factored action head (op, increasing-index card pointer, destination)."""

    def __init__(self, d_model: int):
        super().__init__()
        d = d_model
        self.op_kind = nn.Embedding(az.N_OPS_KINDS, d)
        self.op_tgt = nn.Linear(d, d)
        self.op_feat = nn.Linear(5 + az.OP_FLOAT, d)
        self.op_cards = nn.Linear(d, d)
        self.op_score = mlp(2 * d, d, 1)
        self.ptr_q0 = mlp(3 * d, d, d)
        self.ptr_k = nn.Linear(d, d)
        self.ptr_upd = nn.Linear(d, d)
        self.ptr_norm = nn.LayerNorm(d)
        self.ptr_stop = nn.Linear(d, 1)
        self.dst_q = mlp(3 * d, d, d)
        self.dst_k = nn.Linear(d, d)
        self.dst_pos = nn.Embedding(MAX_DEST, d)
        self.scale = 1.0 / math.sqrt(d)

    def zero_init_outputs(self) -> None:
        nn.init.zeros_(self.op_score[2].weight)
        nn.init.zeros_(self.ptr_q0[2].weight)
        nn.init.zeros_(self.ptr_stop.weight)

    def score_ops(self, cls: torch.Tensor, tok: torch.Tensor, o, O: int | None = None):
        O = O or o.ops.shape[1]
        T = tok.shape[1]
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
        req = bits(ops[..., 5], MAX_HAND).to(tok.dtype)
        cards_h = torch.einsum("boj,bjd->bod", req, tok[:, :MAX_HAND]) / req.sum(-1, keepdim=True).clamp(min=1)
        op_e = self.op_kind(kind) + self.op_tgt(tgt_h) + self.op_feat(feats) + self.op_cards(cards_h)
        logits = self.op_score(torch.cat([op_e, cls.unsqueeze(1).expand(-1, O, -1)], -1)).squeeze(-1)
        valid = torch.arange(O, device=ops.device)[None] < o.n_ops[:, None]
        logits = logits.masked_fill(~valid, NEG)
        return op_e, logits

    def card_steps(self, cls, tok, op_e_a, op_row, n_hand, choose):
        B, _d = cls.shape
        dev = cls.device
        cmin, cmax = op_row[:, 2], op_row[:, 3]
        allowed, required = bits(op_row[:, 4]), bits(op_row[:, 5])
        hand_valid = torch.arange(MAX_HAND, device=dev)[None] < n_hand[:, None]
        cand = allowed & ~required & hand_valid
        hand = tok[:, :MAX_HAND]
        keys = self.ptr_k(hand)
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
        after = cand.flip(-1).long().cumsum(-1).flip(-1) - cand.long()
        for s in range(MAX_CARDS):
            can_card = active & (count < cmax)
            feasible = (count[:, None] + 1 + after) >= cmin[:, None]
            cmask = cand & (ar > last[:, None]) & can_card[:, None] & feasible
            can_stop = active & (count >= cmin)
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

    def dest_logits(self, cls, tok, op_e_a, gain, op_row, card_seq):
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
        q = self.dst_q(torch.cat([cls, op_e_a - gain[:, None] * self.op_feat.weight[:, GAIN], src], -1))
        lg = torch.einsum("bd,bjd->bj", q, keys) * self.scale
        m = (pos < ndest[:, None]) & (pos != frm[:, None]) & active[:, None]
        m = m | (~active[:, None] & (pos == 0))
        return lg.masked_fill(~m, NEG), m, active

    def dest(self, cls, tok, op_e_a, gain, op_row, card_seq, choose):
        B, dev = cls.shape[0], cls.device
        lg, m, active = self.dest_logits(cls, tok, op_e_a, gain, op_row, card_seq)
        idx = choose(lg, m)
        lsm = F.log_softmax(lg, -1)
        logp = torch.where(active, lsm.gather(1, idx.unsqueeze(-1)).squeeze(-1), torch.zeros(B, device=dev))
        ent = torch.where(active, -(lsm.exp() * lsm).masked_fill(~m, 0).sum(-1), torch.zeros(B, device=dev))
        return torch.where(active, idx, torch.zeros_like(idx)), logp, ent

    def act(
        self,
        cls: torch.Tensor,
        tok: torch.Tensor,
        op_e: torch.Tensor,
        logits: torch.Tensor,
        o,
        deterministic: bool = False,
        explore: tuple[float, ...] = (0.0, 0.0),
        noise=None,
        tryable: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        nxt = iter(noise).__next__ if noise is not None else lambda: None
        op = logits.argmax(-1) if deterministic else sample(logits, nxt())
        explored = torch.zeros_like(op, dtype=torch.bool)
        if any(explore) and not deterministic:
            kinds, legal = o.ops[..., 0], logits > NEG / 2
            took = kinds.gather(1, op[:, None])
            among = lambda k, ks: torch.stack([k == x for x in ks]).any(0)  # noqa: E731
            for rate, (known, cases) in zip(explore, EXPLORE, strict=True):
                if not rate:
                    continue
                r1 = torch.rand(op.shape, device=op.device) if noise is None else nxt()
                r2 = torch.rand(legal.shape, device=op.device) if noise is None else nxt()
                cand = torch.stack([among(took, when) & among(kinds, tries) for when, tries in cases]).any(0) & legal
                if known and tryable is not None:
                    center = o.tok_i[..., 1].gather(1, o.ops[..., 1].clamp(min=0))
                    cand &= tryable[center.clamp(0, az.N_CENTERS - 1)]
                new = (r2 + cand).argmax(-1)
                x = cand.any(-1) & (r1 < rate) & (new != op) & ~explored
                op, explored = torch.where(x, new, op), explored | x
        lsm = F.log_softmax(logits, -1)
        op_logp = lsm.gather(1, op[:, None]).squeeze(1)
        rows = o.ops[:, : logits.shape[1]]
        whole = ((rows[..., 3] == 0) | (rows[..., 4] == rows[..., 5])) & (rows[..., 6] == 0)
        sure = lsm.masked_fill(~whole, NEG).amax(-1)
        B = op.shape[0]
        ar = torch.arange(B, device=op.device)
        op_e_a = op_e[ar, op]
        op_row = o.ops[ar, op]

        def pick(_s, lg, m):
            return lg.argmax(-1) if deterministic else sample(lg, nxt())

        mask, seq, c_logp, _ = self.card_steps(cls, tok, op_e_a, op_row, o.n_hand, pick)
        dest, d_logp, _ = self.dest(
            cls, tok, op_e_a, o.ops_f[ar, op, az.OP_ORDER_GAIN], op_row, seq, lambda lg, m: pick(0, lg, m)
        )
        return {
            "op": op,
            "cards": mask,
            "card_seq": seq,
            "dest": dest,
            "logp": op_logp + c_logp + d_logp,
            "sure": sure,
            "explored": explored,
        }

    def evaluate(self, cls, tok, op_e, logits, o, op, card_seq, dest):
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

        _, _seq, c_logp, c_ent = self.card_steps(cls, tok, op_e_a, op_row, o.n_hand, forced)
        _, d_logp, d_ent = self.dest(
            cls, tok, op_e_a, o.ops_f[ar, op, az.OP_ORDER_GAIN], op_row, card_seq, lambda lg, m: dest
        )
        return op_logp + c_logp + d_logp, op_ent + c_ent + d_ent

    def order_aux(self, cls, tok, op_e, o, dest_bits: torch.Tensor):
        cls, tok, op_e = cls.detach(), tok.detach(), op_e.detach()
        B, O = op_e.shape[:2]
        ops, gain = o.ops[:, :O], o.ops_f[:, :O, az.OP_ORDER_GAIN]
        w = self.op_feat.weight[:, GAIN]
        op_e = op_e - gain.unsqueeze(-1) * w.detach() + gain.unsqueeze(-1) * w
        logits = self.op_score(torch.cat([op_e, cls.unsqueeze(1).expand(-1, O, -1)], -1)).squeeze(-1)
        valid = torch.arange(O, device=ops.device)[None] < o.n_ops[:, None]
        move = valid & (ops[..., 0] == az.OP_MOVE_JOKER)
        S = move & (dest_bits.long().gather(1, ops[..., 7].clamp(0, az.ORDER_JOKERS - 1)) != 0)
        rest = logits.detach().masked_fill(~valid | move, NEG)
        lg = logits.masked_fill(~S, NEG)
        m, nm, top = lg.logsumexp(-1), rest.logsumexp(-1), rest.amax(-1)
        ar = torch.arange(B, device=ops.device)
        src = lg.argmax(-1)
        row = ops[ar, src]
        dl, dm, _ = self.dest_logits(cls, tok, op_e[ar, src], gain[ar, src], row, row.new_full((B, 1), -1))
        slots = bits(dest_bits.long().gather(1, row[:, 7:8].clamp(0, az.ORDER_JOKERS - 1)).squeeze(1), MAX_DEST) & dm
        mass = F.log_softmax(dl, -1).masked_fill(~slots, NEG).logsumexp(-1)
        loss = -F.logsigmoid(m - nm) + F.relu(m - top - ORDER_BAND) - mass
        full = logits.detach().masked_fill(~valid, NEG)
        hit = (full.argmax(-1) == src) & (lg.amax(-1) > NEG / 2) & slots[ar, dl.argmax(-1)]
        p_move = F.softmax(full, -1).masked_fill(~move, 0).sum(-1)
        return loss, (m - top).detach(), hit, p_move


class HandBuilderActionHead(FactoredActionHead):
    """RLBalatro-style hand-builder action head.

    Extends the op/card/destination interface with explicit hand-step count
    embeddings (``hb_count_emb``) and a residual step-context projection
    (``hb_step_ctx``) for autoregressive card selection.
    """

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.hb_count_emb = nn.Embedding(MAX_CARDS + 1, d_model)
        self.hb_step_ctx = mlp(2 * d_model, d_model, d_model)

    def card_steps(self, cls, tok, op_e_a, op_row, n_hand, choose):
        B, _d = cls.shape
        dev = cls.device
        cmin, cmax = op_row[:, 2], op_row[:, 3]
        allowed, required = bits(op_row[:, 4]), bits(op_row[:, 5])
        hand_valid = torch.arange(MAX_HAND, device=dev)[None] < n_hand[:, None]
        cand = allowed & ~required & hand_valid
        hand = tok[:, :MAX_HAND]
        keys = self.ptr_k(hand)
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
        after = cand.flip(-1).long().cumsum(-1).flip(-1) - cand.long()
        for s in range(MAX_CARDS):
            can_card = active & (count < cmax)
            feasible = (count[:, None] + 1 + after) >= cmin[:, None]
            cmask = cand & (ar > last[:, None]) & can_card[:, None] & feasible
            can_stop = active & (count >= cmin)
            decide = active & (cmask.any(-1) | can_stop)
            cnt_e = self.hb_count_emb(count.clamp(0, MAX_CARDS))
            q_step = self.ptr_norm(q + self.hb_step_ctx(torch.cat([q, cnt_e], -1)))
            lc = torch.einsum("bd,bjd->bj", q_step, keys) * self.scale
            ls = self.ptr_stop(q_step).squeeze(-1)
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


ACTION_HEADS: dict[str, type[ActionHead]] = {
    "factored": FactoredActionHead,
    "hand_builder": HandBuilderActionHead,
}


def build_action_head(cfg: ModelConfig) -> ActionHead:
    cls = ACTION_HEADS.get(cfg.action_head)
    if cls is None:
        raise ValueError(f"unknown model.action_head {cfg.action_head!r}")
    return cls(cfg.d_model)
