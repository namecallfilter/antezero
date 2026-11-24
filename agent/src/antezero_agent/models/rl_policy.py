"""The policy: an entity Transformer over every card on the table, with pointer-style action heads."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import speed as SP
from ..rl_env import ACTIONS, IDS, SEL_FEATURES, SEL_VERBS
from . import hand_select as HS
from .layers import FastEncoder, FastEncoderLayer, ZeroLinear, pad_inputs, symlog

CARD_GROUPS = ["hand", "jokers", "consumables", "shop", "vouchers", "boosters", "pack"]
MAX_HAND = 24
CENTER = IDS["center"]
HAND_CENTERS = {k: CENTER[k] + 1 for k in ("m_wild", "m_stone", "j_four_fingers", "j_shortcut", "j_smeared")}
PLANET_HANDS = {"c_eris": 0, "c_ceres": 1, "c_planet_x": 2, "c_neptune": 3, "c_mars": 4, "c_earth": 5, "c_jupiter": 6,
                "c_saturn": 7, "c_venus": 8, "c_uranus": 9, "c_mercury": 10, "c_pluto": 11}
C_BASE = CENTER["c_base"] + 1
N_CENTER, N_BLIND, N_TAG = 300, 31, 25
_HAND_TABLES = {}


def _hand_tables(dev):
    t = _HAND_TABLES.get(dev)
    if t is not None:
        return t
    bits = torch.arange(8192)
    pres = torch.zeros((8192, 15), dtype=torch.bool)
    for r in range(2, 15):
        pres[:, r] = ((bits >> (r - 2)) & 1).bool()
    pres[:, 1] = pres[:, 14]
    straight_lut = torch.zeros((4, 8192), dtype=torch.long)
    for ff in (0, 1):
        need = 4 if ff else 5
        for sc in (0, 1):
            prev1 = prev2 = best = torch.zeros(8192, dtype=torch.long)
            for r in range(1, 15):
                c = pres[:, r].long() * (1 + torch.maximum(prev1, prev2 if sc else torch.zeros_like(prev1)))
                prev2, prev1 = prev1, c
                best = torch.maximum(best, c)
            straight_lut[ff * 2 + sc] = (best >= need).long()
    hand_lut = torch.full((2, 6, 6, 2, 2), 12, dtype=torch.long)
    for n_pos in (0, 1):
        for m1 in range(6):
            for m2 in range(6):
                for st in (0, 1):
                    for fl in (0, 1):
                        h = 11 if n_pos else 12
                        for cond, k in ((m1 >= 2, 10), ((m1 == 2) & (m2 == 2), 9), (m1 >= 3, 8), (st == 1, 7),
                                        (fl == 1, 6), ((m1 == 3) & (m2 == 2), 5), (m1 >= 4, 4), ((st == 1) & (fl == 1), 3),
                                        (m1 == 5, 2), ((m1 == 3) & (m2 == 2) & (fl == 1), 1), ((m1 == 5) & (fl == 1), 0)):
                            if cond:
                                h = k
                        hand_lut[n_pos, m1, m2, st, fl] = h
    rank_bit = torch.zeros(16, dtype=torch.long)
    for r in range(2, 15):
        rank_bit[r] = 1 << (r - 2)
    t = (straight_lut.to(dev), hand_lut.to(dev), rank_bit.to(dev))
    _HAND_TABLES[dev] = t
    return t


VERBS = [("BUY_USE_", "shop"), ("BUY_", "shop"), ("REDEEM_", "vouchers"), ("OPEN_", "boosters"), ("PICK_", "pack"),
         ("SELL_JOKER_", "jokers"), ("SELL_CONS_", "consumables"), ("USE_CONS_", "consumables"),
         ("JOKER_RIGHT_", "jokers"), ("JOKER_FRONT_", "jokers"), ("JOKER_BACK_", "jokers")]
JOKER_MOVES = ("JOKER_RIGHT_", "JOKER_FRONT_", "JOKER_BACK_")


def action_table(actions=ACTIONS):
    """Per action: a card-slot verb `(verb index >= 0, slot)`, scored from that card's token, or a global action
    `(-1, global head index)`, scored from the readout token. Returns (verb, slot, global action names)."""
    verb, slot, glob = [], [], []
    for a in actions:
        for v, (p, g) in enumerate(VERBS):
            if a.startswith(p) and a[len(p):].isdigit():
                verb.append(v), slot.append(int(a[len(p):]))
                break
        else:
            verb.append(-1), slot.append(len(glob))
            glob.append(a)
    return verb, slot, glob


class Policy(nn.Module):
    """See the module docstring. deck_emb: deck and stake id embeddings (for training on more than one deck / stake)."""

    def __init__(self, groups, d=256, layers=6, heads=8, deck_emb=False):
        super().__init__()
        self.d = d
        self.actions, self.sel_verbs = ACTIONS, SEL_VERBS
        self.deck_emb = nn.Embedding(20, d) if deck_emb else None
        self.stake_emb = nn.Embedding(10, d) if deck_emb else None
        self.g = {name: (off, r, c) for name, off, r, c in groups}
        self.n_sel = len(SEL_FEATURES)
        self.len_buckets, self.len_round = 1, 1
        self.rows_fn = None
        self.center = nn.Embedding(N_CENTER + 1, d)
        self.rank = nn.Embedding(16, d)
        self.suit = nn.Embedding(6, d)
        self.edition = nn.Embedding(6, d)
        self.seal = nn.Embedding(6, d)
        self.cf = self.g["hand"][2]
        self.n_num = self.cf - 6
        self.num = nn.Linear(14, d)
        pad_inputs(self.num, self.n_num)
        self.group = nn.Embedding(len(CARD_GROUPS) + 4, d)
        self.slot = nn.Embedding(24, d)
        n_glob = self.g["scalars"][2] + self.g["hand_levels"][1] * self.g["hand_levels"][2] + self.g["counts"][2] \
            + self.n_sel
        self.glob = nn.Sequential(nn.Linear(n_glob, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.blind = nn.Embedding(N_BLIND + 1, d)
        self.boss = nn.Embedding(N_BLIND + 1, d)
        self.tag = nn.Embedding(N_TAG + 1, d)
        n_hist = 16 * 6 + N_CENTER + 1 + 6 + 6 + 2
        self.deck = nn.Sequential(nn.Linear(n_hist, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.readout = nn.Parameter(torch.zeros(d))
        self.enc = FastEncoder(FastEncoderLayer(d, heads), layers)
        self.norm = nn.LayerNorm(d)
        verb, slot, glob = action_table(self.actions)
        self.register_buffer("verb", torch.tensor(verb), persistent=False)
        self.register_buffer("aslot", torch.tensor(slot), persistent=False)
        self.n_glob_act = len(glob)
        self.verb_head = nn.Linear(d, len(VERBS))
        self.glob_head = nn.Linear(d, self.n_glob_act)
        self.v = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.tok_off = {}
        off = 4
        for name in CARD_GROUPS:
            self.tok_off[name] = off
            off += self.g[name][1]
        idx = [self.tok_off[VERBS[v][1]] + s if v >= 0 else 0 for v, s in zip(verb, slot)]
        self.register_buffer("atok", torch.tensor(idx), persistent=False)
        nn.init.zeros_(self.verb_head.weight), nn.init.zeros_(self.glob_head.weight)
        self.sel_verb = nn.Embedding(len(self.sel_verbs), d)
        self.sel_q = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
        self.sel_k = nn.Linear(d, d)
        self.sel_cond = nn.Linear(d, d)
        self.sel_stop = nn.Parameter(torch.zeros(d))
        nn.init.zeros_(self.sel_q[2].weight)
        act = {a: i for i, a in enumerate(self.actions)}
        vmap = torch.full((len(self.actions),), -1, dtype=torch.long)
        vtok = torch.zeros(len(self.sel_verbs), dtype=torch.long)
        ht_, he_ = HS.verb_targets(self.sel_verbs)
        self.register_buffer("sel_ht", torch.tensor(ht_), persistent=False)
        for v, name in enumerate(self.sel_verbs):
            vmap[act[name]] = v
            if name.startswith("USE_CONS_"):
                vtok[v] = self.tok_off["consumables"] + int(name.split("_")[-1])
            elif name.startswith("PICK_"):
                vtok[v] = self.tok_off["pack"] + int(name.split("_")[-1])
        self.register_buffer("sel_vmap", vmap, persistent=False)
        self.register_buffer("sel_vtok", vtok, persistent=False)
        self.register_buffer("sel_he", torch.tensor(he_, dtype=torch.bool), persistent=False)
        self.sel_verb_reorder_jokers = self.sel_verbs.index("REORDER_JOKERS")
        self.register_buffer("jmove_mask", torch.tensor([a.startswith(JOKER_MOVES) for a in self.actions]),
                             persistent=False)
        self.count_emb = nn.Embedding(17, d)
        self.hand_emb = nn.Embedding(13, d)
        nn.init.zeros_(self.hand_emb.weight)
        self.jmove_gate = nn.Linear(d, 1)
        nn.init.zeros_(self.jmove_gate.weight), nn.init.zeros_(self.jmove_gate.bias)
        card_hand = torch.full((N_CENTER + 1,), -1, dtype=torch.long)
        for k, h in PLANET_HANDS.items():
            card_hand[CENTER[k] + 1] = h
        self.register_buffer("card_hand", card_hand, persistent=False)
        self.todo_feature = CENTER["j_todo_list"] + 1
        self.card_off = self.g[CARD_GROUPS[0]][0]
        self.card_rows = sum(self.g[name][1] for name in CARD_GROUPS)
        self.register_buffer("card_gi", torch.cat([torch.full((self.g[name][1],), gi, dtype=torch.long)
                                                   for gi, name in enumerate(CARD_GROUPS)]), persistent=False)
        self.register_buffer("card_si", torch.cat([torch.arange(self.g[name][1], dtype=torch.long)
                                                   for name in CARD_GROUPS]), persistent=False)
        self.register_buffer("mod_centers", torch.tensor([HAND_CENTERS[k] for k in ("j_four_fingers", "j_shortcut",
                                                                                    "j_smeared")]), persistent=False)
        self.n_ctx = self.g["round_ctx"][2]
        self.glob_ctx = ZeroLinear(self.n_ctx, 2 * d)

    def part(self, x, name):
        """Observation group `name` of x as (B, rows, cols)."""
        off, r, c = self.g[name]
        return x[:, off:off + r * c].reshape(-1, r, c)

    def deck_counts(self, f, keep=None):
        """Draw-pile cards f (B, 120, CARD_F) (those in keep) as a normalized rank x suit / center / edition / seal histogram."""
        ids, present = f[..., 1:6].long(), (f[..., 0] > 0).float()
        if keep is not None:
            present = present * keep.float()
        Bn = f.shape[0]
        k0 = ids[..., 1].clamp(0, 15) * 6 + ids[..., 2].clamp(0, 5)
        k1 = ids[..., 0].clamp(0, N_CENTER) + 96
        k2 = ids[..., 3].clamp(0, 5) + (96 + N_CENTER + 1)
        k3 = ids[..., 4].clamp(0, 5) + (96 + N_CENTER + 1 + 6)
        tot = 96 + (N_CENTER + 1) + 6 + 6
        hist = torch.zeros(Bn, tot + 2, device=f.device, dtype=f.dtype)
        hist[:, :tot].scatter_add_(1, torch.cat([k0, k1, k2, k3], 1), present.repeat(1, 4))
        hist[:, tot] = present.sum(1)
        hist[:, tot + 1] = (symlog(f[..., 16]) * present).sum(1)
        return hist / 13.0

    def row_lens(self, x):
        """(B,) each row's real tokens: 4 + present cards + tags + special draw-pile cards."""
        Bn = x.shape[0]
        f_cards = x[:, self.card_off:self.card_off + self.card_rows * self.cf].reshape(Bn, self.card_rows, self.cf)
        n = 4 + (f_cards[..., 0] > 0).sum(1) + (self.part(x, "tags")[..., 0].long() > 0).sum(1)
        fdeck = self.part(x, "deck")
        ids_d = fdeck.long()
        n = n + ((fdeck[..., 0] > 0) & ((ids_d[..., 1] != C_BASE) | (ids_d[..., 4] > 0) | (ids_d[..., 5] > 0))).sum(1)
        return n

    def n_slots(self):
        """Token slots before packing: 4 + card rows + tags + draw-pile rows."""
        return 4 + self.card_rows + self.g["tags"][1] + self.g["deck"][1]

    def forward(self, x, extra=False):
        if self.len_buckets <= 1 and self.len_round <= 1 and self.rows_fn is None:
            return self._rows(x, extra)
        rows = self._rows if self.rows_fn is None else self.rows_fn[self.training]
        S = self.n_slots()
        lens = self.row_lens(x)
        if self.len_buckets <= 1:
            ln = int(lens.max())
            return rows(x, extra, min(S, -(-ln // self.len_round) * self.len_round))
        plan = SP.bucket_plan(torch.bincount(lens, minlength=S + 1).tolist(), self.len_buckets, self.len_round, S)
        if len(plan) == 1:
            return rows(x, extra, plan[0][1])
        order = torch.argsort(lens, stable=True)
        inv = torch.empty_like(order)
        inv[order] = torch.arange(len(order), device=order.device)
        outs, s0 = [], 0
        for nb, L in plan:
            idx = order[s0:s0 + nb]
            outs.append(rows(x.index_select(0, idx), extra, L))
            s0 += nb

        def merge(parts):
            if parts[0] is None:
                return None
            if isinstance(parts[0], torch.Tensor):
                return torch.cat(parts, 0).index_select(0, inv)
            return type(parts[0])(merge([p[i] for p in parts]) for i in range(len(parts[0])))
        return merge(outs)

    def _rows(self, x, extra=False, L=None):
        """The forward of rows x (B, D), each independent, packed to L tokens (default: the longest row's)."""
        Bn = x.shape[0]
        sc = self.part(x, "scalars")[:, 0]
        parts = [sc, self.part(x, "hand_levels").reshape(Bn, -1), self.part(x, "counts")[:, 0], x[:, -self.n_sel:]]
        h0 = self.glob[0](symlog(torch.cat(parts, 1)))
        h0 = h0 + self.glob_ctx(self.part(x, "round_ctx")[:, 0])
        glob = self.glob[2](self.glob[1](h0))
        bt = self.part(x, "blind_tags")[:, 0].long()
        blind = (self.blind(sc[:, 17].long().clamp(0, N_BLIND)) + self.boss(sc[:, 27].long().clamp(0, N_BLIND))
                 + self.tag(bt[:, 0].clamp(0, N_TAG)) + self.tag(bt[:, 1].clamp(0, N_TAG))
                 + self.boss(bt[:, 2].clamp(0, N_BLIND)))
        if self.deck_emb is not None:
            blind = blind + self.deck_emb(sc[:, 38].long().clamp(0, 19)) + self.stake_emb(sc[:, 37].long().clamp(0, 9))
        fdeck = self.part(x, "deck")
        ids_d = fdeck.long()
        special = (fdeck[..., 0] > 0) & ((ids_d[..., 1] != C_BASE) | (ids_d[..., 4] > 0) | (ids_d[..., 5] > 0))
        deck = self.deck(self.deck_counts(fdeck, keep=~special))
        f_cards = x[:, self.card_off:self.card_off + self.card_rows * self.cf].reshape(Bn, self.card_rows, self.cf)
        cm = f_cards[..., 0] > 0
        tg = self.part(x, "tags")[..., 0].long()
        tm = tg > 0
        keep = torch.cat([torch.ones(Bn, 4, dtype=torch.bool, device=x.device), cm, tm, special], 1)
        all_cards = torch.cat([f_cards, fdeck], 1)
        S = keep.shape[1]
        if L is None:
            L = int(keep.sum(1).max())
        order = torch.argsort((~keep).to(torch.int8), dim=1, stable=True)[:, :L]
        keep = keep.gather(1, order)
        inv = torch.zeros(Bn, S, dtype=torch.long, device=x.device)
        inv.scatter_(1, order, torch.arange(L, device=x.device).expand(Bn, L))
        tok = inv[:, self.atok]
        sub = order[:, 4:] - 4
        is_tag = (sub >= self.card_rows) & (sub < self.card_rows + tg.shape[1])
        c_idx = torch.where(sub < self.card_rows, sub,
                            torch.where(sub >= self.card_rows + tg.shape[1], sub - tg.shape[1], 0))
        fc = all_cards.gather(1, c_idx[..., None].expand(-1, -1, self.cf))
        ids = fc.long()
        cid = ids[..., 1].clamp(0, N_CENTER)
        gi = torch.where(sub < self.card_rows, self.card_gi[c_idx.clamp(max=self.card_rows - 1)], len(CARD_GROUPS) + 2)
        ct = (self.center(cid) + self.rank(ids[..., 2].clamp(0, 15))
              + self.suit(ids[..., 3].clamp(0, 5)) + self.edition(ids[..., 4].clamp(0, 5))
              + self.seal(ids[..., 5].clamp(0, 5)) + self.num(symlog(fc[..., 6:6 + self.n_num]))
              + self.group(gi) + self.slot(self.card_si[c_idx.clamp(max=self.card_rows - 1)]))
        h_id = torch.where(cid == self.todo_feature, ids[..., 18] - 1, self.card_hand[cid])
        ct = ct + self.hand_emb(h_id.clamp(0, 12)) * (h_id >= 0)[..., None]
        t_idx = (sub - self.card_rows).clamp(0, tg.shape[1] - 1)
        tt = self.tag(tg.gather(1, t_idx).clamp(0, N_TAG)) + self.group.weight[len(CARD_GROUPS) + 1]
        ct = torch.where(is_tag[..., None], tt, ct)
        seq = torch.cat([self.readout.expand(Bn, -1)[:, None], glob[:, None], blind[:, None], deck[:, None], ct], 1)
        h = self.norm(self.enc(seq, src_key_padding_mask=~keep))
        ro = h[:, 0]
        verb_logits = self.verb_head(h)
        card = verb_logits[torch.arange(Bn, device=x.device)[:, None], tok, self.verb.clamp(min=0)]
        glob_l = self.glob_head(ro)[:, self.aslot.clamp(max=self.n_glob_act - 1)]
        logits = torch.where(self.verb >= 0, card, glob_l)
        if not extra:
            return logits, self.v(ro).squeeze(-1)
        off = self.tok_off["hand"]
        hpos = inv[:, off:off + MAX_HAND]
        hand_h = h.gather(1, hpos[..., None].expand(-1, -1, h.shape[2]))
        vpos = inv[:, self.sel_vtok]
        sel_h = h.gather(1, vpos[..., None].expand(-1, -1, h.shape[2]))
        joff = self.tok_off["jokers"]
        jpos = inv[:, joff:joff + 20]
        joker_h = h.gather(1, jpos[..., None].expand(-1, -1, h.shape[2]))
        jc = self.part(x, "jokers")[..., 1].long()
        mods = (jc[..., None] == self.mod_centers).any(1)
        return logits, self.v(ro).squeeze(-1), hand_h, sel_h, (self.part(x, "hand")[..., 1:4].long(), mods), joker_h

    def action_dist(self, logits, ro, mask):
        """Action log-probs (B, A) and entropy (B,) under the legality mask."""
        if ro.dim() == 3:
            ro = ro[:, 0]
        m_jm = mask & self.jmove_mask
        m_other = mask & ~self.jmove_mask
        any_jm = m_jm.any(1, keepdim=True)
        gate_l = self.jmove_gate(ro).float().masked_fill(~any_jm, -1e9)
        lp_top = F.log_softmax(torch.cat([logits.float().masked_fill(~m_other, -1e9), gate_l], 1), -1)
        lp_sub = F.log_softmax(logits.float().masked_fill(~m_jm, -1e9), -1)
        lp_all = torch.where(self.jmove_mask, lp_top[:, -1:] + lp_sub, lp_top[:, :-1]).masked_fill(~mask, -1e9)
        p_top, p_sub = lp_top.exp(), lp_sub.exp()
        ent = -(p_top * lp_top).sum(-1) + p_top[:, -1].detach() * (-(p_sub * lp_sub).sum(-1)) * any_jm.squeeze(1)
        return lp_all, ent

    @staticmethod
    def _prep_hand_info(hand_info):
        hf, mods = hand_info
        ff, shortcut, smeared = mods.unbind(1)
        wild = hf[..., 0] == HAND_CENTERS["m_wild"]
        stone = hf[..., 0] == HAND_CENTERS["m_stone"]
        r_idx = torch.where(stone, 0, hf[..., 1].clamp(0, 15))
        suit = hf[..., 2].clamp(0, 4)
        suit = torch.where(smeared[:, None], torch.where((suit == 2) | (suit == 4), 2, torch.where((suit == 1) | (suit == 3), 1, suit)), suit)
        s_idx = torch.where(stone | wild, 0, suit)
        need = torch.where(ff, 4, 5)
        ff_sc = ff.long() * 2 + shortcut.long()
        return r_idx, s_idx, wild, need, ff_sc

    @staticmethod
    def _eval_hand(chosen, prep):
        r_idx, s_idx, wild, need, ff_sc = prep
        straight_lut, hand_lut, rank_bit = _hand_tables(chosen.device)
        ch_l = chosen.long()
        cr = torch.zeros(chosen.shape[0], 16, dtype=torch.long, device=chosen.device)
        cr.scatter_add_(1, r_idx * ch_l, ch_l)
        cr[:, 0] = 0
        cs = torch.zeros(chosen.shape[0], 5, dtype=torch.long, device=chosen.device)
        cs.scatter_add_(1, s_idx * ch_l, ch_l)
        m = cr.topk(2, dim=1).values.clamp(max=5)
        flush = (cs[:, 1:5].max(1).values + (wild & chosen).sum(1)) >= need
        rmask = ((cr > 0).long() * rank_bit).sum(1)
        straight = straight_lut[ff_sc, rmask]
        return hand_lut[chosen.any(1).long(), m[:, 0], m[:, 1], straight, flush.long()]

    def pick_cards(self, hand_h, ro, verb, lo, hi, elig, picks=None, hand_info=None, masker=None, pmask=None,
                   record=False, joker_h=None):
        """Card picks for the selection actions: a pointer head over the hand cards (joker slots for REORDER_JOKERS)."""
        Bn, dev = hand_h.shape[0], hand_h.device
        active = (verb >= 0) & (hi >= 0)
        vh = ro[torch.arange(Bn, device=dev), verb.clamp(min=0)] if ro.dim() == 3 else ro
        q0 = self.sel_q(torch.cat([vh, self.sel_verb(verb.clamp(min=0))], 1))
        tgt = torch.where(verb >= 0, self.sel_ht[verb.clamp(min=0)], -1)
        q0 = q0 + self.hand_emb(tgt.clamp(0, 11)) * ((tgt >= 0) & (tgt < 12))[:, None]
        is_joker = (verb == self.sel_verb_reorder_jokers)[:, None, None]
        joker_h_24 = F.pad(joker_h, (0, 0, 0, MAX_HAND - joker_h.shape[1]))
        items_h = torch.where(is_joker, joker_h_24, hand_h)
        keys = self.sel_k(items_h)
        ok = ((elig[:, None] >> torch.arange(MAX_HAND, device=dev)) & 1).bool()
        use_he = (verb >= 0) & self.sel_he[verb.clamp(min=0)]
        scale = self.d ** -0.5
        if picks is not None:
            is_card = active[:, None] & (picks >= 0)
            live = torch.zeros(Bn, 5, dtype=torch.bool, device=dev)
            live[:, 0] = active
            live[:, 1:] = is_card[:, :-1]
            c = torch.where(picks >= 0, picks, torch.full_like(picks, MAX_HAND))
            cc = picks.clamp(min=0, max=MAX_HAND - 1)
            oh = F.one_hot(cc, MAX_HAND).bool() & is_card[..., None]
            chosen = torch.zeros(Bn, 5, MAX_HAND, dtype=torch.bool, device=dev)
            chosen[:, 1:] = oh[:, :-1].cumsum(1) > 0
            count = is_card.long().cumsum(1) - is_card.long()
            picked_h = items_h.gather(1, cc[..., None].expand(-1, -1, items_h.shape[2])) * is_card[..., None]
            cond = torch.zeros_like(picked_h)
            cond[:, 1:] = picked_h[:, :-1].cumsum(1)
            q = q0[:, None, :] + self.sel_cond(cond)
            if hand_info is not None:
                r_idx, s_idx, wild, need, ff_sc = self._prep_hand_info(hand_info)
                prep4 = (r_idx[:, None].expand(-1, 4, -1).reshape(Bn * 4, MAX_HAND),
                         s_idx[:, None].expand(-1, 4, -1).reshape(Bn * 4, MAX_HAND),
                         wild[:, None].expand(-1, 4, -1).reshape(Bn * 4, MAX_HAND),
                         need[:, None].expand(-1, 4).reshape(Bn * 4),
                         ff_sc[:, None].expand(-1, 4).reshape(Bn * 4))
                ht = torch.full((Bn, 5), 12, dtype=torch.long, device=dev)
                ht[:, 1:] = self._eval_hand(chosen[:, 1:].reshape(Bn * 4, MAX_HAND), prep4).reshape(Bn, 4)
                q = q + self.hand_emb(ht) * use_he[:, None, None]
            lg = torch.cat([(keys[:, None, :, :] * q[:, :, None, :]).sum(-1),
                            (q * self.sel_stop).sum(-1, keepdim=True)], -1).float() * scale
            if pmask is not None:
                al = HS.unpack_allow(pmask)
                allow_cards = al[..., :MAX_HAND]
                any_card = allow_cards.any(-1)
                allow_stop = al[..., MAX_HAND] | ~live
            else:
                allow_cards = ok[:, None, :] & ~chosen & (count < hi[:, None])[..., None]
                any_card = allow_cards.any(-1)
                allow_stop = (count >= lo[:, None]) | ~live | ~any_card
            allow = torch.cat([allow_cards, allow_stop[..., None]], -1)
            lp_all = F.log_softmax(lg.masked_fill(~allow, -1e9), -1)
            lp_sum = (lp_all.gather(2, c[..., None]).squeeze(2) * live).sum(1)
            p_all = lp_all.exp()
            p_stop = p_all[..., MAX_HAND].clamp(1e-7, 1.0 - 1e-7)
            p_cont = (1.0 - p_stop).clamp(1e-7, 1.0)
            ent_stop = -(p_stop * p_stop.log() + p_cont * p_cont.log())
            lp_cards = F.log_softmax(lg[..., :MAX_HAND].masked_fill(~allow_cards, -1e9), -1)
            ent_cards = -(lp_cards.exp() * lp_cards).sum(-1) * any_card
            ent_step = torch.where(allow_stop & any_card, ent_stop + p_cont.detach() * ent_cards, ent_cards)
            ent_mean = (ent_step * live).sum(1) / live.sum(1).clamp(min=1)
            return picks, lp_sum, ent_mean
        prep = self._prep_hand_info(hand_info) if hand_info is not None else None
        chosen = torch.zeros(Bn, MAX_HAND, dtype=torch.bool, device=dev)
        count = torch.zeros(Bn, dtype=torch.long, device=dev)
        cond = torch.zeros_like(q0)
        done = ~active
        outs = []
        lp_sum = torch.zeros(Bn, device=dev)
        ent_sum = torch.zeros(Bn, device=dev)
        n_live = torch.zeros(Bn, device=dev)
        rec = []
        for k in range(5):
            q = q0 + self.sel_cond(cond)
            if prep is not None:
                he = self.hand_emb.weight[12] if k == 0 else self.hand_emb(self._eval_hand(chosen, prep))
                q = q + he * use_he[:, None]
            lg = torch.cat([(keys * q[:, None, :]).sum(-1), (q * self.sel_stop).sum(-1, keepdim=True)], 1).float() * scale
            allow = torch.cat([ok & ~chosen & (count < hi)[:, None], (count >= lo)[:, None]], 1)
            if masker is not None:
                card_ok, stop_ok = masker.card_stop(chosen)
                allow = torch.where(masker.active[:, None], torch.cat([card_ok, stop_ok[:, None]], 1), allow)
            allow[:, MAX_HAND] |= done | ~allow[:, :MAX_HAND].any(1)
            if record:
                rec.append(HS.pack_allow(allow))
            lp_all = F.log_softmax(lg.masked_fill(~allow, -1e9), -1)
            probs = lp_all.exp()
            c = torch.multinomial(probs, 1, True).squeeze(1)
            live = ~done
            n_live = n_live + live.float()
            lp_sum = lp_sum + lp_all.gather(1, c[:, None]).squeeze(1) * live
            ent_sum = ent_sum - (probs * lp_all).sum(-1) * live
            card = live & (c < MAX_HAND)
            outs.append(torch.where(card, c, -1))
            if k < 4:
                cc = c.clamp(max=MAX_HAND - 1)
                chosen = chosen | (F.one_hot(cc, MAX_HAND).bool() & card[:, None])
                count = count + card.long()
                cond = cond + items_h.gather(1, cc[:, None, None].expand(-1, 1, items_h.shape[2])).squeeze(1) * card[:, None]
                done = done | (c == MAX_HAND)
        if record:
            return torch.stack(outs, 1), lp_sum, ent_sum / n_live.clamp(min=1), torch.stack(rec, 1)
        return torch.stack(outs, 1), lp_sum, ent_sum / n_live.clamp(min=1)

    @torch.no_grad()
    def act(self, x, mask, spec_t, hctx_t, hctx_np=None, amp=False, pick_fn=None):
        """Sample an action and its card picks."""
        pick_fn = pick_fn or self.pick_cards
        with torch.autocast("cuda", torch.bfloat16, enabled=amp):
            out = self(x, extra=True)
            logits, v = out[0], out[1]
            lp_a, _ = self.action_dist(logits, out[3], mask)
        a = torch.multinomial(lp_a.exp(), 1, True).squeeze(1)
        lp = lp_a.gather(1, a[:, None]).squeeze(1)
        vi = self.sel_vmap[a]
        lo_, hi_, el_ = spec_t.gather(1, vi.clamp(min=0)[:, None, None].expand(-1, 1, 3)).squeeze(1).unbind(1)
        hi_ = torch.where(vi >= 0, hi_, torch.full_like(hi_, -1))
        tgt = torch.where((vi >= 0) & (hi_ >= 0), self.sel_ht[vi.clamp(min=0)], -1)
        msk = HS.HandMasker(hctx_t, tgt, boss=True, ctx_host=hctx_np)
        with torch.autocast("cuda", torch.bfloat16, enabled=amp):
            pk, lp_pk, _, pm_ = pick_fn(out[2].float(), out[3].float(), vi, lo_, hi_, el_, hand_info=out[4],
                                        masker=msk, record=True, joker_h=out[5].float())
        lp = lp + lp_pk
        return a, lp, v.float(), pk, torch.stack([vi, lo_, hi_, el_], 1), pm_

    def evaluate_actions(self, xb, mb_mask, mb_a, mb_sp, mb_pk, mb_pm, amp=False, pick_fn=None):
        """Teacher-forced (log-prob (B,), value (B,), summed entropy) of stored actions and picks (PPO's update)."""
        pick_fn = pick_fn or self.pick_cards
        with torch.autocast("cuda", torch.bfloat16, enabled=amp):
            out = self(xb, extra=True)
            logits, v = out[0], out[1]
            lp_all, ent_a = self.action_dist(logits, out[3], mb_mask)
        v = v.float()
        lp = lp_all.gather(1, mb_a[:, None]).squeeze(1)
        ent_sum = ent_a.sum()
        with torch.autocast("cuda", torch.bfloat16, enabled=amp):
            _, lp_pk, ent_pk = pick_fn(out[2].float(), out[3].float(), mb_sp[:, 0], mb_sp[:, 1], mb_sp[:, 2],
                                       mb_sp[:, 3], picks=mb_pk, hand_info=out[4], pmask=mb_pm,
                                       joker_h=out[5].float())
        return lp + lp_pk, v, ent_sum + ent_pk.sum()
