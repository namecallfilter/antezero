"""Greedy ensemble of checkpoints (load ``a.pt+b.pt+c.pt``). Every factored
choice (op, then each card-pointer step, then the destination) is the argmax
of the members' mean log-probabilities, with every member's pointer state
advanced by the chosen card. Sampling (the round search's proposals) and the
value are member 0's. The mean of three SWA windows (swa_e, swa_h, swa_f)
beat swa_e alone by +1.93 pts greedy over 8,192 paired games (p = 2e-4)."""

import torch
import torch.nn.functional as F

from .model import MAX_CARDS, MAX_HAND, NEG, bits


def _mean_lsm(lgs, m):
    lp = torch.stack([F.log_softmax(lg.masked_fill(~m, NEG), -1) for lg in lgs]).mean(0)
    return lp.masked_fill(~m, NEG)


class Ens(torch.nn.Module):
    def __init__(self, members):
        super().__init__()
        self.members = torch.nn.ModuleList(members)

    def value(self, cls):
        return self.members[0].value(cls)

    def encode(self, o):
        outs = [m.encode(o) for m in self.members]
        lg0 = outs[0][3]
        lp = _mean_lsm([x[3] for x in outs], lg0 > NEG / 2)
        cls, tok, op_e, _ = outs[0]
        return cls, tok, op_e, lp

    @torch.no_grad()
    def act(self, o, deterministic=False, explore=0.0):
        if not deterministic:
            return self.members[0].act(o, explore=explore)
        outs = [m.encode(o) for m in self.members]
        valid = outs[0][3] > NEG / 2
        op = _mean_lsm([x[3] for x in outs], valid).argmax(-1)
        B = op.shape[0]
        dev = op.device
        ar = torch.arange(B, device=dev)
        op_row = o.ops[ar, op]
        # card pointer in lockstep (Net._card_steps, greedy)
        cmin, cmax = op_row[:, 2], op_row[:, 3]
        allowed, required = bits(op_row[:, 4]), bits(op_row[:, 5])
        hand_valid = torch.arange(MAX_HAND, device=dev)[None] < o.n_hand[:, None]
        cand = allowed & ~required & hand_valid
        st = []
        for m, (cls, tok, op_e, _lg) in zip(self.members, outs):
            hand = tok[:, :MAX_HAND]
            req_sum = (hand * required.unsqueeze(-1)).sum(1)
            st.append([m, hand, m.ptr_k(hand), m.ptr_q0(torch.cat([cls, op_e[ar, op], req_sum], -1))])
        count = required.sum(-1)
        last = torch.full((B,), -1, device=dev, dtype=torch.long)
        active = cmax > 0
        chosen = torch.zeros(B, MAX_HAND, dtype=torch.bool, device=dev)
        seq = torch.full((B, MAX_CARDS), -1, dtype=torch.long, device=dev)
        arh = torch.arange(MAX_HAND, device=dev)[None]
        after = cand.flip(-1).long().cumsum(-1).flip(-1) - cand.long()
        for s in range(MAX_CARDS):
            can_card = active & (count < cmax)
            feasible = (count[:, None] + 1 + after) >= cmin[:, None]
            cmask = cand & (arh > last[:, None]) & can_card[:, None] & feasible
            can_stop = active & (count >= cmin)
            msk = torch.cat([cmask, can_stop.unsqueeze(-1)], -1)
            lgs = []
            for m, hand, keys, q in st:
                lc = torch.einsum("bd,bjd->bj", m.ptr_norm(q), keys) * m.scale
                ls = m.ptr_stop(m.ptr_norm(q)).squeeze(-1)
                lgs.append(torch.cat([lc, ls.unsqueeze(-1)], -1))
            idx = _mean_lsm(lgs, msk | ~msk.any(-1, keepdim=True)).argmax(-1)
            is_card = active & (idx < MAX_HAND) & cmask.gather(1, idx.clamp(0, MAX_HAND - 1).unsqueeze(-1)).squeeze(-1)
            seq[:, s] = torch.where(is_card, idx, torch.full_like(idx, -1))
            pick = F.one_hot(idx.clamp(0, MAX_HAND - 1), MAX_HAND).bool() & is_card[:, None]
            chosen = chosen | pick
            for e in st:
                e[3] = e[3] + e[0].ptr_upd(torch.einsum("bj,bjd->bd", pick.float(), e[1]))
            count = count + is_card.long()
            last = torch.where(is_card, idx, last)
            active = active & is_card
        mask = (chosen.long() << torch.arange(MAX_HAND, device=dev)).sum(-1)
        # destination: mean of the members' log-probs
        dl = []
        for m, (cls, tok, op_e, _lg) in zip(self.members, outs):
            cap = {}

            def grab(lg, msk, cap=cap):
                cap["lg"], cap["m"] = lg, msk
                return lg.argmax(-1)

            m._dest(cls, tok, op_e[ar, op], op_row, seq, grab)
            dl.append(cap)
        dest = _mean_lsm([c["lg"] for c in dl], dl[0]["m"]).argmax(-1)
        active_d = op_row[:, 6] > 0
        dest = torch.where(active_d, dest, torch.zeros_like(dest))
        m0 = self.members[0]
        return {"op": op, "cards": mask, "card_seq": seq, "dest": dest, "logp": torch.zeros(B, device=dev), "value": m0.value(outs[0][0]).squeeze(-1), "explored": torch.zeros_like(op, dtype=torch.bool)}
