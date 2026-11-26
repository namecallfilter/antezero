"""Card-pick masks of the plays, on the policy's device."""
import antezero_sim
import numpy as np
import torch

from ..rl_env import PLAY_HAND_ACTIONS

MAX_HAND = 24
HS_ANY = 12
STOP_BIT = 1 << MAX_HAND
INF = 64.0
_CONST = {}


def _const(dev):
    """Per-device constant tables: card bits, pack weights, rank powers, the straight minimal sets."""
    t = _CONST.get(dev)
    if t is None:
        cb = torch.ones(MAX_HAND, dtype=torch.long) << torch.arange(MAX_HAND)
        t = {"cb": cb.to(dev), "w": torch.cat([cb, torch.tensor([STOP_BIT])]).to(dev),
             "p2": (torch.ones(13, dtype=torch.long) << torch.arange(13)).to(dev),
             "r13": torch.arange(13, device=dev)}
        _CONST[dev] = t
    return t


_MTAB = {}


def straight_table(dev, modes=(0, 1, 2, 3)):
    """The minimal straight sets of these modes (Four Fingers * 2 + Shortcut) in one table: (sets (M,) long rank bits,
    valid (4, M) bool: the sets of each mode, bits (M, 13) float, bits.T (13, M)). Mode 0 alone: 10 sets; all: 204."""
    key = (dev, tuple(sorted(set(modes))))
    t = _MTAB.get(key)
    if t is None:
        mins = straight_mins()
        mu = sorted(set(x for m in key[1] for x in mins[m]))
        mt = torch.tensor(mu, dtype=torch.long)
        valid = torch.tensor([[x in set(m) for x in mu] for m in mins], dtype=torch.bool).reshape(4, len(mu))
        bits = ((mt[:, None] >> torch.arange(13)) & 1).float()
        t = (mt.to(dev), valid.to(dev), bits.to(dev), bits.T.contiguous().to(dev))
        _MTAB[key] = t
    return t


def card_bits(dev):
    return _const(dev)["cb"]


def pack_allow(allow):
    """(B, 25) bool allow mask of one pick step (24 cards, stop) -> (B,) int32 bits (bit 24: stop)."""
    return (allow.long() * _const(allow.device)["w"]).sum(-1).int()


def unpack_allow(bits):
    """(..., ) int32 packed allow masks -> (..., 25) bool."""
    sh = torch.arange(MAX_HAND + 1, device=bits.device)
    return ((bits.long()[..., None] >> sh) & 1).bool()


_LUT = None


def straight_lut():
    """The engine's straight table: (4, 8192) bool, mode = Four Fingers * 2 + Shortcut."""
    global _LUT
    if _LUT is None:
        out = np.asarray(antezero_sim.balatro_straight_lut(), dtype=np.uint16).reshape(4, 8192)
        _LUT = (out >> 15).astype(np.bool_)
    return _LUT


_MINS = None


def straight_mins():
    """Per mode, the minimal rank sets (bit r - 2) the straight table accepts (it is monotone: a rank set is a straight
    iff it contains one), in the engine's order."""
    global _MINS
    if _MINS is None:
        L = straight_lut()
        _MINS = []
        for m in range(4):
            s = L[m]
            _MINS.append([x for x in range(8192) if s[x] and not any(s[x & ~(1 << b)] for b in range(13) if x >> b & 1)])
    return _MINS


_HAND_SPEC = {0: (0, True, 5, 0), 1: (1, True, 3, 2), 2: (0, False, 5, 0), 3: (3, True, 0, 0), 4: (0, False, 4, 0),
              5: (1, False, 3, 2), 6: (2, False, 0, 0), 7: (3, False, 0, 0), 8: (0, False, 3, 0), 9: (1, False, 2, 2),
              10: (0, False, 2, 0), 11: (4, False, 0, 0), 12: (4, False, 0, 0)}
_SPEC_T = {}


def _spec(dev):
    t = _SPEC_T.get(dev)
    if t is None:
        rows = [_HAND_SPEC[h] for h in range(13)] + [(4, False, 0, 0)]
        f = torch.tensor([r[0] for r in rows])
        suited = torch.tensor([r[1] for r in rows])
        k = torch.tensor([float(r[2] if r[0] == 0 else 4 if r[0] == 1 and r[2] == 2 else 0) for r in rows])
        ka = torch.tensor([float(r[2] if r[0] == 1 else 0) for r in rows])
        kb = torch.tensor([float(r[3] if r[0] == 1 else 0) for r in rows])
        kind_on = (f == 0) | ((f == 1) & (ka == 2))
        t = tuple(x.to(dev) for x in (f, suited, k, ka, kb, kind_on))
        _SPEC_T[dev] = t
    return t


class HandMasker:
    """Per-pick legality masks of declared-hand verbs for a batch (rows without a declared hand are not used)."""

    def __init__(self, ctx, target, boss, ctx_host=None):
        dev = ctx.device
        C = _const(dev)
        c = ctx.long()
        Bn = c.shape[0]
        cards = c[:, :MAX_HAND]
        self.active = target >= 0
        tg = torch.where(self.active, target.long(), 13)
        f, suited, k, ka, kb, kind_on = (x[tg] for x in _spec(dev))
        self.k, self.ka, self.kb = k[:, None, None], ka[:, None, None], kb[:, None, None]
        self.any_on = f == 4
        present = ((cards >> 10) & 1).bool()
        rank = cards & 15
        has = present & (rank >= 2)
        self.ridx = torch.where(has, rank - 2, 13)
        sb = ((cards[..., None] >> torch.arange(4, device=dev) + 4) & 1).bool() & present[..., None]
        self.suit = sb
        self.inch = torch.cat([has[:, None, :], (sb & has[..., None]).transpose(1, 2)], 1)
        self.present = present
        self.forced = ((cards >> 9) & 1).bool() & present
        q4 = torch.arange(4, device=dev)
        slot = torch.cat([torch.where(self.inch, torch.arange(5, device=dev)[:, None] * 13 + self.ridx[:, None, :].clamp(max=12),
                                      69).transpose(1, 2), torch.where(sb, 65 + q4, 69)], 2)
        self.slot = slot.reshape(Bn, MAX_HAND * 9)
        tot = torch.zeros(Bn, 70, device=dev).scatter_add_(1, self.slot, present.float()[..., None].expand(-1, -1, 9)
                                                             .reshape(Bn, -1))
        self.tot, self.stot = tot[:, :65].view(Bn, 5, 13), tot[:, 65:69]
        chan = torch.zeros(Bn, 5, dtype=torch.bool, device=dev)
        chan[:, 0] = True
        chan = torch.where(suited[:, None], ~chan, chan)[..., None]
        flags = c[:, 25]
        ff, sc = flags & 1, (flags >> 1) & 1
        self.need = (5 - ff).float()[:, None]
        if boss:
            self.lo, self.budget = c[:, 28].float(), c[:, 29].float()
        else:
            self.lo, self.budget = torch.ones(Bn, device=dev), torch.full((Bn,), 5.0, device=dev)
        pen = lambda ok: torch.where(ok, 0.0, INF)   # noqa: E731
        tot = self.tot
        self.kpen = pen((tot >= self.k) & chan & kind_on[:, None, None])
        g_on = chan & (f == 1)[:, None, None]
        self.apen, self.bpen = pen((tot >= self.ka) & g_on), pen((tot >= self.kb) & g_on)
        self.spen = pen((self.stot >= self.need) & (f == 2)[:, None])
        self.anypen = pen(self.any_on)
        fl = None if ctx_host is None else np.asarray(ctx_host)[:, 25]
        modes = (0, 1, 2, 3) if fl is None else np.unique((fl & 1) * 2 + ((fl >> 1) & 1)).tolist()
        mt, mvalid, self.mbits, self.mbT = straight_table(dev, modes)
        have = ((self.tot > 0).long() * C["p2"]).sum(-1)
        mfeas = ((mt & ~have[..., None]) == 0) & mvalid[ff * 2 + sc][:, None, :] & chan & (f == 3)[:, None, None]
        self.mbase = self.mbits.sum(-1) + pen(mfeas)
        self.zero1 = torch.zeros(Bn, 1, dtype=torch.bool, device=dev)

    def card_stop(self, chosen):
        """(card_ok (B, 24), stop_ok (B,)) after picks chosen (B, 24) bool (forced cards excluded)."""
        dev = chosen.device
        Bn = chosen.shape[0]
        with torch.autocast(dev.type, enabled=False):
            P = chosen | self.forced
            Pf = P.float()
            cnt_all = torch.zeros(Bn, 70, device=dev).scatter_add_(1, self.slot, Pf[..., None].expand(-1, -1, 9)
                                                                   .reshape(Bn, -1))
            cnt = cnt_all[:, :65].view(Bn, 5, 13)
            n = Pf.sum(1)
            r1 = (self.budget - n)[:, None]
            r = r1[..., None]
            d = (self.k - cnt).clamp_(min=0).add_(self.kpen)
            cost = d.amin((1, 2))
            use = d <= r
            da = (self.ka - cnt).clamp_(min=0).add_(self.apen)
            db = (self.kb - cnt).clamp_(min=0).add_(self.bpen)
            va, ia = da.topk(2, -1, largest=False)
            vb, ib = db.topk(2, -1, largest=False)
            r13 = _const(dev)["r13"]
            oa = torch.where(r13 == ia[..., :1], va[..., 1:], va[..., :1])
            ob = torch.where(r13 == ib[..., :1], vb[..., 1:], vb[..., :1])
            cost = torch.minimum(cost, (da + ob).amin((1, 2)))
            use |= torch.minimum((da - 1).clamp_(min=0) + ob, (db - 1).clamp_(min=0) + oa) <= r - 1
            dm = self.mbase - ((cnt > 0).float() @ self.mbT)
            cost = torch.minimum(cost, dm.amin((1, 2)))
            use |= (((dm <= r).float() @ self.mbits) > 0) & (cnt == 0)
            ds = (self.need - cnt_all[:, 65:69]).clamp_(min=0).add_(self.spen)
            cost = torch.minimum(cost, ds.amin(1))
            cost = torch.minimum(cost, (n == 0).float() + self.anypen)
            us = torch.cat([use.view(Bn, 65), ds <= r1, self.zero1], 1)
            hit = us.gather(1, self.slot).view(Bn, MAX_HAND, 9).any(-1)
            rr = r1[:, 0]
            free = (cost <= rr - 1) | self.any_on
            card_ok = self.present & ~P & (rr >= 1)[:, None] & (free[:, None] | ((cost <= rr)[:, None] & hit))
            stop_ok = (cost == 0) & (n >= self.lo) & (rr >= 0)
        return card_ok, stop_ok


def verb_targets(sel_verbs):
    """Per selection verb (sel_verbs order): the declared hand (0..11, HS_ANY) or -1; and the verbs whose picks the
    picker is told the hand formed so far of (the plays and DISCARD)."""
    ht, he = [], []
    for name in sel_verbs:
        h = HS_ANY if name == "PLAY" else PLAY_HAND_ACTIONS.index(name) if name in PLAY_HAND_ACTIONS else -1
        ht.append(h)
        he.append(h >= 0 or name == "DISCARD")
    return ht, he
