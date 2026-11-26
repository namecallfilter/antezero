"""What training writes besides checkpoints."""
import csv
import json
import math

import numpy as np
import torch

from .rl_env import HANDS, IDS, JOKER_INDEX, JOKER_KEYS

TAROTS = ("fool magician high_priestess empress emperor heirophant lovers chariot justice hermit wheel_of_fortune "
          "strength hanged_man death temperance devil tower star moon sun judgement world").split()
PLANETS = "mercury venus earth mars jupiter saturn uranus neptune pluto planet_x ceres eris".split()
SPECTRALS = ("familiar grim incantation talisman aura wraith sigil ouija ectoplasm immolate ankh deja_vu hex trance "
             "medium cryptid soul black_hole").split()
CONS_EVENTS = ("used", "buy_use", "pack_pick", "bought", "sold")
CENTER = IDS["center"]


class JokerTracker:
    """Per joker, like the game's own profile stats."""

    def __init__(self, n_env):
        self.names = {v + 1: k[2:] for k, v in CENTER.items() if k.startswith("j_")}
        n = max(self.names) + 1
        self.is_joker = np.zeros(n + 1, bool)
        self.is_joker[list(self.names)] = True
        self.held = np.zeros((n_env, n + 1), bool)
        self.last = np.zeros((n_env, 20), np.int32)
        self.rows = np.arange(n_env)[:, None]
        self.max_id = n
        self.stats = {}
        self.blinds = np.zeros(n + 1, np.int64)
        self.key_index = np.full(n + 2, -1, np.int64)
        for c, nm in self.names.items():
            self.key_index[c] = JOKER_INDEX["j_" + nm]

    def observe(self, o):
        cid = np.clip(o["jokers"][..., 1].astype(np.int32), 0, self.max_id)
        self.last = np.where(self.is_joker[cid], cid, 0)
        self.held[self.rows, self.last] = True

    def game_over(self, i, won, ante):
        end = set(self.last[i][self.last[i] > 0].tolist())
        for c in np.nonzero(self.held[i])[0]:
            if c in self.names:
                st = self.stats.setdefault(self.names[c], [0, 0, 0, 0])
                st[0] += 1
                st[3] += ante
                if c in end:
                    st[1 if won else 2] += 1
        self.held[i] = False

    def blinds_beaten(self, n):
        i = np.flatnonzero(n)
        if len(i):
            h = np.zeros((len(i), self.held.shape[1]), bool)
            h[np.arange(len(i))[:, None], self.last[i]] = True
            h[:, 0] = False
            self.blinds += (h * n[i, None]).sum(0)

    def flush(self):
        st, self.stats = self.stats, {}
        for c in np.nonzero(self.blinds)[0]:
            if c in self.names:
                st.setdefault(self.names[c], [0, 0, 0, 0])
        for v in st.values():
            v.append(0)
        for c in np.nonzero(self.blinds)[0]:
            if c in self.names:
                st[self.names[c]][4] = int(self.blinds[c])
        self.blinds[:] = 0
        return st


class CompletionTracker:
    """Per stake, how many won runs each joker was in the final lineup of."""

    def __init__(self, key_index):
        self.key_index = key_index
        self.wins = {}

    def won(self, stake, lineup_centers):
        j = self.key_index[lineup_centers[lineup_centers > 0]]
        w = self.wins.setdefault(int(stake), np.zeros(len(JOKER_KEYS), np.int64))
        np.add.at(w, np.unique(j[j >= 0]), 1)

    def coverage(self, stake, k):
        w = self.wins.get(int(stake))
        return 0 if w is None else int((w >= k).sum())

    def state(self):
        return {s: w.tolist() for s, w in self.wins.items()}

    def load(self, st):
        self.wins = {int(s): np.array(w, np.int64) for s, w in (st or {}).items()}

    def write_csv(self, path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["stake", "joker", "wins"])
            for s in sorted(self.wins):
                for k, nm in enumerate(JOKER_KEYS):
                    w.writerow([s, nm, int(self.wins[s][k])])


class HandTracker:
    """Poker hands played per game."""

    def __init__(self, n_env):
        self.last = np.zeros((n_env, len(HANDS)), np.float32)
        self.stats = np.zeros((len(HANDS), 2), np.int64)
        self.wins = 0

    def observe(self, o):
        np.copyto(self.last, o["hand_levels"][..., 3])

    def game_over(self, i, won):
        v = self.last[i].astype(np.int64)
        self.stats[:, 0] += v
        if won:
            self.stats[:, 1] += v
            self.wins += 1

    def flush(self):
        st, w = self.stats, self.wins
        self.stats, self.wins = np.zeros_like(st), 0
        return st, w


class ConsTracker:
    """Counts what the agent does with each tarot / planet / spectral."""

    def __init__(self, actions):
        self.names = {}
        for st, lst in (("tarot", TAROTS), ("planet", PLANETS), ("spectral", SPECTRALS)):
            for n in lst:
                self.names[CENTER["c_" + n] + 1] = (n, st)

        def slots(prefix):
            idx = [i for i, x in enumerate(actions) if x.startswith(prefix) and x[len(prefix):].isdigit()]
            return idx[0], len(idx)
        self.kinds = {"used": (*slots("USE_CONS_"), "consumables"), "buy_use": (*slots("BUY_USE_"), "shop"),
                      "pack_pick": (*slots("PICK_"), "pack"), "bought": (*slots("BUY_"), "shop"),
                      "sold": (*slots("SELL_CONS_"), "consumables")}
        self.counts = {}
        self.pending = []

    def before(self, o, a):
        self.pending = []
        for ev, (first, n, grp) in self.kinds.items():
            k = a - first
            hit = np.nonzero((k >= 0) & (k < n))[0]
            if len(hit):
                self.pending.append((ev, hit, o[grp][hit, k[hit], 1].astype(int)))

    def after(self, illegal):
        for ev, hit, cid in self.pending:
            for c in cid[illegal[hit] == 0]:
                if c in self.names:
                    key = (self.names[c], ev)
                    self.counts[key] = self.counts.get(key, 0) + 1

    def flush(self):
        c, self.counts = self.counts, {}
        return c


class PlayStats:
    """Per update: declared hand usage, kickers, blue seals kept, upgrades."""
    NAMES = HANDS + ["Any"]

    def __init__(self):
        self.reset()

    def reset(self):
        self.declared = np.zeros(13, np.int64)
        self.upgraded = np.zeros(13, np.int64)
        self.plays = self.kick = self.blue_held = self.blue_kept = 0

    def add(self, info, ok):
        hp = info["hand_played"]
        played = ok & (hp >= 0)
        if not played.any():
            return
        d = np.clip(info["hs_declared"][played], 0, 12)
        np.add.at(self.declared, d, 1)
        np.add.at(self.upgraded, d, (d < 12) & (hp[played] < d))
        self.plays += int(played.sum())
        self.kick += int((info["played_cards"][played] - info["played_scoring"][played]).sum())
        sf = played & np.isin(hp, (3, 6, 7))
        held, pl = info["blue_held"][sf], info["blue_played"][sf]
        self.blue_held += int(held.sum())
        self.blue_kept += int((held - pl).sum())

    def row(self):
        n = max(1, self.plays)
        nd = int(self.declared[:12].sum())
        nan = float("nan")
        return {"play_any_share": self.declared[12] / n if self.plays else nan,
                "play_kickers": self.kick / n if self.plays else nan,
                "play_blue_kept": self.blue_kept / self.blue_held if self.blue_held else nan,
                "play_upgrade": self.upgraded[:12].sum() / nd if nd else nan}

    def wandb(self):
        n = max(1, self.plays)
        out = {f"plays/declared_{nm.lower().replace(' ', '_')}": self.declared[k] / n
               for k, nm in enumerate(self.NAMES)}
        out.update({f"plays/upgraded_{nm.lower().replace(' ', '_')}": self.upgraded[k] / self.declared[k]
                    for k, nm in enumerate(self.NAMES[:12]) if self.declared[k]})
        return out


def _fmt(v):
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.6g}"
    return v


class RunLogger:
    """log.csv (+ W&B) rows of named values, and the interval tables written at each checkpoint."""

    def __init__(self, run_dir, cfg_dict, opt, resume):
        self.run = run_dir
        self.cfg = cfg_dict
        self.cols = None
        has_log = (run_dir / "log.csv").exists() and (run_dir / "log.csv").stat().st_size > 0
        has_ckpt = (run_dir / "ckpt.pt").exists()
        self.new = not (has_log and has_ckpt and resume)
        if not self.new:
            self.cols = open(run_dir / "log.csv").readline().strip().split(",")
        self.logf = open(run_dir / "log.csv", "w" if self.new else "a", newline="")
        self.log = csv.writer(self.logf)
        self.wb = None
        if cfg_dict.get("wandb"):
            import wandb
            name = cfg_dict["name"]
            self.wb = wandb.init(entity=cfg_dict.get("wandb_entity") or None, project=cfg_dict["wandb"],
                                 name=f"balatro-{name}", id=f"balatro-{name}", resume="allow", config=cfg_dict,
                                 dir=str(run_dir))
        self.w_groups = [("weights/rms_muon" if isinstance(o, torch.optim.Muon) else "weights/rms_adam",
                          [q for g in o.param_groups for q in g["params"]]) for o in opt.opts]

    def write(self, row, extra_wb=None):
        if self.cols is None:
            self.cols = list(row)
            self.log.writerow(self.cols)
        self.log.writerow([_fmt(row.get(k, float("nan"))) for k in self.cols])
        self.logf.flush()
        if self.wb:
            d = {f"train/{k}": float(v) for k, v in row.items() if k != "update" and not
                 (isinstance(v, float) and math.isnan(v))}
            with torch.no_grad():
                for k, ps in self.w_groups:
                    if ps:
                        d[k] = math.sqrt(float(torch.stack([q.float().square().sum() for q in ps]).sum())
                                         / sum(q.numel() for q in ps))
            self.wb.log({**d, **(extra_wb or {})}, step=row["update"])

    def _append(self, name, header, rows):
        new = not (self.run / name).exists()
        with open(self.run / name, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(header)
            w.writerows(rows)

    def interval_tables(self, update, steps, games, n_g, completion, cons, jok, hands, combos, best):
        head = [update, steps, games, n_g]
        completion.write_csv(self.run / "completion.csv")
        cc, js, (hs, h_wins) = cons.flush(), jok.flush(), hands.flush()
        self._append("hands.csv", ["update", "steps", "games", "interval_games", "interval_wins", "hand", "plays",
                                   "plays_won"], [[*head, h_wins, nm, *hs[k]] for k, nm in enumerate(HANDS)])
        self._append("jokers.csv", ["update", "steps", "games", "interval_games", "joker", "runs", "wins", "losses",
                                    "ante_sum", "blinds"], [[*head, nm, *js[nm]] for nm in sorted(js)])
        self._append("consumables.csv", ["update", "steps", "games", "interval_games", "card", "set", *CONS_EVENTS],
                     [[*head, nm, st, *[cc.get(((nm, st), e), 0) for e in CONS_EVENTS]]
                      for nm, st in sorted({k[0] for k in cc})])
        self._append("combos.csv", ["update", "steps", "games", "deck", "stake", "combo_games", "combo_wins",
                                    "recent_n", "recent_win", "recent_ante"],
                     [[update, steps, games, d, s, combos.games.get((d, s), 0), combos.wins.get((d, s), 0), len(q),
                       f"{np.mean([w for w, _ in q]):.4f}", f"{np.mean([a for _, a in q]):.2f}"]
                      for (d, s), q in sorted(combos.recent.items())])
        json.dump({"unlocked": sorted(combos.unlocked), "passed": sorted(combos.passed),
                   "games": {f"{d}/{s}": n for (d, s), n in combos.games.items()},
                   "wins": {f"{d}/{s}": n for (d, s), n in combos.wins.items()}},
                  open(self.run / "curriculum.json", "w"), indent=1)
        if self.wb:
            per = {f"combo/{d[2:] if d.startswith('b_') else d}_{s}": float(np.mean([w for w, _ in q]))
                   for (d, s), q in combos.recent.items()}
            for st in ("tarot", "planet", "spectral"):
                per[f"consumables/{st}_used_per_game"] = sum(
                    v for (k, e), v in cc.items() if k[1] == st and e in ("used", "buy_use", "pack_pick")) / max(1, n_g)
                per[f"consumables/{st}_sold_per_game"] = sum(
                    v for (k, e), v in cc.items() if k[1] == st and e == "sold") / max(1, n_g)
            for k, nm in enumerate(HANDS):
                per[f"hands/{nm.lower().replace(' ', '_')}_per_game"] = hs[k][0] / max(1, n_g)
            for deck, b in best.items():
                per[f"best/{deck[2:] if deck.startswith('b_') else deck}_stake"] = b["stake"]
            for st in completion.wins:
                for k in (1, 10, 100, 1000):
                    per[f"completion/stake{st}_ge{k}"] = completion.coverage(st, k)
            self.wb.log(per, step=update)

    def finish(self, stopped=False):
        if self.wb:
            self.wb.finish(exit_code=143 if stopped else 0)
