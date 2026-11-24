"""PPO: the rollout buffer, GAE, the clipped-surrogate update, and the optimizer / learning-rate machinery."""
import math

import numpy as np
import torch
import torch.nn as nn

from . import speed as SP


class RolloutBuffer:
    """T rollout steps of E parallel envs."""

    def __init__(self, T, E, obs_dim, n_actions, dev, buf_dev):
        self.T, self.E, self.dev = T, E, dev
        self.x = torch.zeros(T, E, obs_dim, device=buf_dev, pin_memory=buf_dev.type == "cpu" and dev.type == "cuda")
        self.m = torch.zeros(T, E, n_actions, dtype=torch.bool, device=dev)
        self.a = torch.zeros(T, E, dtype=torch.long, device=dev)
        self.lp = torch.zeros(T, E, device=dev)
        self.v = torch.zeros(T, E, device=dev)
        self.r = torch.zeros(T, E, device=dev)
        self.d = torch.zeros(T, E, device=dev)
        self.pk = torch.full((T, E, 5), -1, dtype=torch.long, device=dev)
        self.sp = torch.zeros(T, E, 4, dtype=torch.long, device=dev)
        self.pm = torch.zeros(T, E, 5, dtype=torch.int32, device=dev)

    def store_step(self, t, x, mask, a, lp, v, pk, sp, pm):
        self.x[t], self.m[t], self.a[t], self.lp[t], self.v[t] = x, mask, a, lp, v
        self.pk[t], self.sp[t], self.pm[t] = pk, sp, pm

    def store_outcome(self, t, r, done):
        self.r[t] = torch.from_numpy(r).to(self.dev)
        self.d[t] = torch.from_numpy(done.astype("float32")).to(self.dev)

    def gae(self, last_v, gamma, lam):
        """Advantages and returns (T * E,) plus the flattened buffer."""
        T, E = self.T, self.E
        adv = torch.zeros_like(self.r)
        g = torch.zeros(E, device=self.dev)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else self.v[t + 1]
            nd = 1 - self.d[t]
            delta = self.r[t] + gamma * nv * nd - self.v[t]
            g = delta + gamma * lam * nd * g
            adv[t] = g
        ret = adv + self.v
        flat = lambda b: b.reshape(T * E, *b.shape[2:])   # noqa: E731
        return (flat(self.x), flat(self.m), flat(self.a), flat(self.lp), adv.reshape(-1), ret.reshape(-1),
                flat(self.pk), flat(self.sp), flat(self.pm))


class PPOTrainer:
    def __init__(self, model, opt, cfg, dev, buf_dev, obs_dim, n_actions, amp=False, pick_train=None, pgather=None):
        self.model, self.opt, self.cfg, self.dev, self.buf_dev = model, opt, cfg, dev, buf_dev
        self.mb = cfg.minibatch_size
        self.amp = amp
        self.pick_train = pick_train or model.pick_cards
        self.pgather = pgather
        self.buffer = RolloutBuffer(cfg.num_steps, cfg.num_envs, obs_dim, n_actions, dev, buf_dev)

    def update(self, last_x):
        """One PPO update on the full buffer. Returns the mean (policy loss, value loss, entropy, approx KL)."""
        c = self.cfg
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=self.amp):
            _, last_v = self.model(last_x)
        fx, fm, fa, flp, fadv, fret, fpk, fsp, fpm = self.buffer.gae(last_v.float(), c.gamma, c.gae_lambda)
        n = self.buffer.T * self.buffer.E
        self.model.train()
        stats = []
        for _ in range(c.epochs):
            perm = torch.randperm(n, device=self.dev)
            for i in range(0, n, self.mb):
                jb = perm[i:i + self.mb]
                a_mb = fadv[jb]
                a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                nb, cs = len(jb), c.microbatch_size or len(jb)
                self.opt.zero_grad()
                acc = torch.zeros(4, device=self.dev)
                for c0 in range(0, nb, cs):
                    j, a_ = jb[c0:c0 + cs], a_mb[c0:c0 + cs]
                    if self.buf_dev.type == self.dev.type:
                        xb = fx[j]
                    elif self.pgather is not None:
                        xb = self.pgather(fx, j)
                    else:
                        xb = fx[j.cpu()].to(self.dev, non_blocking=True)
                    lp, v, ent_sum = self.model.evaluate_actions(xb, fm[j], fa[j], fsp[j], fpk[j], fpm[j], amp=self.amp,
                                                                 pick_fn=self.pick_train)
                    ent = ent_sum / nb
                    ratio = (lp - flp[j]).exp()
                    pg = torch.max(-a_ * ratio, -a_ * ratio.clamp(1 - c.clip, 1 + c.clip)).sum() / nb
                    vl = 0.5 * ((v - fret[j]) ** 2).sum() / nb
                    loss = pg + c.vf_coef * vl - c.ent_coef * ent
                    loss.backward()
                    with torch.no_grad():
                        kl = ((ratio - 1) - (lp - flp[j])).sum() / nb
                        acc += torch.stack([pg.detach(), vl.detach(), ent.detach(), kl])
                nn.utils.clip_grad_norm_(self.model.parameters(), c.max_grad_norm)
                self.opt.step()
                stats.append(acc)
        return torch.stack(stats).mean(0).cpu().tolist() if stats else [float("nan")] * 4


class Optimizers:
    """Several optimizers stepped as one (param_groups, state_dict, step, zero_grad)."""

    def __init__(self, *opts):
        self.opts = opts

    @property
    def param_groups(self):
        return [g for o in self.opts for g in o.param_groups]

    def step(self):
        for o in self.opts:
            o.step()

    def zero_grad(self):
        for o in self.opts:
            o.zero_grad()

    def state_dict(self):
        return [o.state_dict() for o in self.opts]

    def load_state_dict(self, d):
        for o, s in zip(self.opts, d):
            o.load_state_dict(s)


def muon_params(model, cfg, emb_params=()):
    """The parameters Muon optimizes: the 2-D hidden matrices."""
    emb_ids = {id(q) for q in emb_params}
    table_ids = {id(m.weight) for m in model.modules() if isinstance(m, nn.Embedding)}
    heads = ("verb_head.", "glob_head.", "v.")
    return [p for n, p in model.named_parameters()
            if p.ndim == 2 and not n.startswith(heads) and "emb" not in n and id(p) not in emb_ids
            and not (cfg.embed_adam and id(p) in table_ids)]


def build_optimizer(model, cfg, dev, emb_params=()):
    """Muon for the hidden matrices + Adam for the rest (optimizer=muon), or Adam alone."""
    emb_ids = {id(q) for q in emb_params}
    fast = {"fused": True} if cfg.fast_opt and dev.type == "cuda" else {"foreach": True} if cfg.fast_opt else {}
    hidden = muon_params(model, cfg, emb_params) if cfg.optimizer == "muon" else []
    h_ids = {id(p) for p in hidden}
    groups = [{"params": [p for p in model.parameters() if id(p) not in emb_ids and id(p) not in h_ids]}]
    if emb_params:
        groups.append({"params": list(emb_params)})
    adam = torch.optim.Adam(groups, lr=cfg.lr, eps=1e-5, **fast)
    if not hidden:
        return Optimizers(adam), emb_ids
    muon = (SP.BatchedMuon if cfg.fast_opt else torch.optim.Muon)(
        hidden, lr=cfg.lr, weight_decay=cfg.muon_wd, adjust_lr_fn="match_rms_adamw")
    return Optimizers(muon, adam), emb_ids


def compute_lr(cfg, update, steps, kl_ctl):
    """The learning rate of update `update`."""
    if cfg.lr_schedule == "exp":
        tau = -cfg.lr_decay_updates / math.log(max(1e-6, cfg.lr_floor / max(1e-6, cfg.lr - cfg.lr_floor)))
        lr = cfg.lr_floor + (cfg.lr - cfg.lr_floor) * math.exp(-update / max(1.0, tau))
    else:
        lr = cfg.lr * (1 - min(1.0, steps / cfg.total_steps)) + 1e-6
    return lr * kl_ctl["m"]


def update_kl_controller(cfg, update, kl, kl_ctl):
    """From update kl_start on: adjust lr factor based on KL EMA."""
    if cfg.target_kl > 0 and not np.isnan(kl):
        kl_ctl["ema"] = kl if kl_ctl["ema"] is None else (1 - cfg.kl_ema) * kl_ctl["ema"] + cfg.kl_ema * kl
        if update >= cfg.kl_start:
            if kl_ctl["ema"] > cfg.kl_high * cfg.target_kl:
                kl_ctl["m"] *= 1 - cfg.kl_step
            elif kl_ctl["ema"] < cfg.kl_low * cfg.target_kl:
                kl_ctl["m"] = min(cfg.kl_max_factor, kl_ctl["m"] * (1 + cfg.kl_step))
