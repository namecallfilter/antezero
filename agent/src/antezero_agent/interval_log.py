"""Per-interval CSV & W&B logging (hands, jokers, consumables, combos, completion)."""

from __future__ import annotations

import csv
from collections import deque
from pathlib import Path

CONS_EVENTS = ("used", "buy_use", "pack_pick", "bought", "sold")


def merge_interval_stats(dicts: list[dict]) -> dict:
    """Merge multiple ``VecEnv.pop_interval_stats()`` dicts into one."""
    hands: dict[str, list[int]] = {}
    jokers: dict[str, list[int]] = {}
    consumables: dict[tuple[str, str], list[int]] = {}
    best: dict[str, list] = {}
    for d in dicts:
        for nm, plays, wins in d.get("hands", ()):
            cur = hands.setdefault(nm, [0, 0])
            cur[0] += plays
            cur[1] += wins
        for j, vals in d.get("jokers", {}).items():
            cur = jokers.setdefault(j, [0, 0, 0, 0, 0])
            for i, v in enumerate(vals):
                cur[i] += v
        for k, vals in d.get("consumables", {}).items():
            cur = consumables.setdefault(k, [0, 0, 0, 0, 0])
            for i, v in enumerate(vals):
                cur[i] += v
        for deck, (st, sc) in d.get("best", {}).items():
            cur = best.setdefault(deck, [0, 0.0])
            cur[0] = max(cur[0], int(st))
            cur[1] = max(cur[1], float(sc))
    return {
        "hands": [(nm, v[0], v[1]) for nm, v in hands.items()],
        "jokers": {j: tuple(v) for j, v in jokers.items()},
        "consumables": {k: tuple(v) for k, v in consumables.items()},
        "best": {deck: tuple(v) for deck, v in best.items()},
    }


class IntervalLogger:
    """Tracks per-interval episode/hand/joker/consumable stats and flushes CSVs + W&B."""

    def __init__(self, run_dir: Path, backs: list[str], stakes: list[int], recent_window: int = 200):
        self.run_dir = run_dir
        self.backs = list(backs)
        self.stakes = sorted(set(int(s) for s in stakes))
        self.recent_window = recent_window
        self.total_games = 0
        self.interval_games = 0
        self.interval_wins = 0
        self.combo_games: dict[tuple[str, int], int] = {}
        self.combo_wins: dict[tuple[str, int], int] = {}
        self.combo_recent: dict[tuple[str, int], deque[tuple[int, int]]] = {}
        self.best: dict[str, list] = {}  # deck -> [best_stake, best_score]
        self.last_flushed_update = 0

    def add_episodes(self, episodes) -> None:
        for ep in episodes:
            # (slot, seed, back, stake, ante, won, blinds_cleared, steps, ret, truncated, from_snapshot)
            if ep[10]:
                continue
            deck = str(ep[2])
            stake = int(ep[3])
            ante = int(ep[4])
            won = int(bool(ep[5]))
            self.total_games += 1
            self.interval_games += 1
            self.interval_wins += won
            key = (deck, stake)
            self.combo_games[key] = self.combo_games.get(key, 0) + 1
            self.combo_wins[key] = self.combo_wins.get(key, 0) + won
            q = self.combo_recent.get(key)
            if q is None:
                q = deque(maxlen=self.recent_window)
                self.combo_recent[key] = q
            q.append((won, ante))
            b = self.best.setdefault(deck, [0, 0.0])
            if won and stake > b[0]:
                b[0] = stake

    def _coverage(self, stake: int, threshold: int) -> float:
        n_decks = max(1, len(self.backs))
        hits = sum(1 for d in self.backs if self.combo_wins.get((d, stake), 0) >= threshold)
        return hits / n_decks

    def flush(self, update: int, steps: int, env, wb=None) -> dict[str, float]:
        if hasattr(env, "pop_interval_stats"):
            st = env.pop_interval_stats()
        else:
            st = {"hands": [], "jokers": {}, "consumables": {}, "best": {}}

        for deck, (bst, bsc) in st.get("best", {}).items():
            cur = self.best.setdefault(deck, [0, 0.0])
            if int(bst) > cur[0]:
                cur[0] = int(bst)
            if float(bsc) > cur[1]:
                cur[1] = float(bsc)

        n_g = self.interval_games
        h_wins = self.interval_wins
        hands_rows = st.get("hands", [])
        jokers_dict = st.get("jokers", {})
        cons_dict = st.get("consumables", {})

        hnew = not (self.run_dir / "hands.csv").exists()
        with open(self.run_dir / "hands.csv", "a", newline="") as hf:
            hw = csv.writer(hf)
            if hnew:
                hw.writerow(["update", "steps", "games", "interval_games", "interval_wins", "hand", "plays", "plays_won"])
            for nm, plays, pwins in hands_rows:
                hw.writerow([update, steps, self.total_games, n_g, h_wins, nm, plays, pwins])

        jnew = not (self.run_dir / "jokers.csv").exists()
        with open(self.run_dir / "jokers.csv", "a", newline="") as jf:
            jw = csv.writer(jf)
            if jnew:
                jw.writerow(["update", "steps", "games", "interval_games", "joker", "runs", "wins", "losses", "ante_sum", "blinds"])
            for nm in sorted(jokers_dict):
                jw.writerow([update, steps, self.total_games, n_g, nm, *jokers_dict[nm]])

        knew = not (self.run_dir / "consumables.csv").exists()
        with open(self.run_dir / "consumables.csv", "a", newline="") as kf:
            kw = csv.writer(kf)
            if knew:
                kw.writerow(["update", "steps", "games", "interval_games", "card", "set", *CONS_EVENTS])
            for card, cset in sorted(cons_dict):
                kw.writerow([update, steps, self.total_games, n_g, card, cset, *cons_dict[(card, cset)]])

        cnew = not (self.run_dir / "combos.csv").exists()
        with open(self.run_dir / "combos.csv", "a", newline="") as cf:
            cw = csv.writer(cf)
            if cnew:
                cw.writerow(["update", "steps", "games", "deck", "stake", "combo_games", "combo_wins", "recent_n", "recent_win", "recent_ante"])
            for (d, s_), q in sorted(self.combo_recent.items()):
                n_q = len(q)
                rw = sum(w for w, _ in q) / n_q if n_q else 0.0
                ra = sum(a for _, a in q) / n_q if n_q else 0.0
                cw.writerow([update, steps, self.total_games, d, s_, self.combo_games.get((d, s_), 0), self.combo_wins.get((d, s_), 0), n_q, f"{rw:.4f}", f"{ra:.2f}"])

        denom = max(1, n_g)
        wb_dict: dict[str, float] = {}
        for nm, plays, _pwins in hands_rows:
            wb_dict[f"hands/{nm.lower().replace(' ', '_')}_per_game"] = plays / denom
        for cset in ("tarot", "planet", "spectral"):
            wb_dict[f"consumables/{cset}_used_per_game"] = (
                sum(v[0] + v[1] + v[2] for (_card, st_name), v in cons_dict.items() if st_name == cset) / denom
            )
            wb_dict[f"consumables/{cset}_sold_per_game"] = (
                sum(v[4] for (_card, st_name), v in cons_dict.items() if st_name == cset) / denom
            )
        for (card, cset), v in cons_dict.items():
            if cset in ("tarot", "spectral"):
                wb_dict[f"{cset}/{card}_per100"] = 100.0 * (v[0] + v[1] + v[2]) / denom
        for deck, (bst, bsc) in self.best.items():
            d_short = deck[2:] if deck.startswith("b_") else deck
            wb_dict[f"best/{d_short}_stake"] = float(bst)
            wb_dict[f"best/{d_short}_score"] = float(bsc)
        stakes_seen = sorted(set(self.stakes) | {s_ for (_d, s_) in self.combo_wins})
        for st_ in stakes_seen:
            for k_ in (1, 10, 100, 1000):
                wb_dict[f"completion/stake{st_}_ge{k_}"] = self._coverage(st_, k_)

        if wb is not None and wb_dict:
            wb.log(wb_dict, step=update)

        self.interval_games = 0
        self.interval_wins = 0
        self.last_flushed_update = update
        return wb_dict

    def state_dict(self) -> dict:
        return {
            "total_games": self.total_games,
            "interval_games": self.interval_games,
            "interval_wins": self.interval_wins,
            "combo_games": [[d, s, v] for (d, s), v in self.combo_games.items()],
            "combo_wins": [[d, s, v] for (d, s), v in self.combo_wins.items()],
            "combo_recent": [[d, s, list(q)] for (d, s), q in self.combo_recent.items()],
            "best": {d: list(v) for d, v in self.best.items()},
            "last_flushed_update": self.last_flushed_update,
        }

    def load_state_dict(self, st: dict | None) -> None:
        if not st:
            return
        self.total_games = int(st.get("total_games", 0))
        self.interval_games = int(st.get("interval_games", 0))
        self.interval_wins = int(st.get("interval_wins", 0))
        self.combo_games = {(str(d), int(s)): int(v) for d, s, v in st.get("combo_games", ())}
        self.combo_wins = {(str(d), int(s)): int(v) for d, s, v in st.get("combo_wins", ())}
        self.combo_recent = {
            (str(d), int(s)): deque(((int(w), int(a)) for w, a in q), maxlen=self.recent_window)
            for d, s, q in st.get("combo_recent", ())
        }
        self.best = {str(d): [int(v[0]), float(v[1])] for d, v in st.get("best", {}).items()}
        self.last_flushed_update = int(st.get("last_flushed_update", 0))
