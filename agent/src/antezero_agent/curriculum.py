"""Which game each env plays next: its (deck, stake) combo, and whether it starts from a start-pool state."""
from collections import deque

import numpy as np

from .rl_env import DECKS, challenge_id
from .start_points import load_pool


class Combos:
    """Per env: the combo of the game being played (`cur`) and of the next one (`nxt`, already set in the env)."""

    def __init__(self, cfg, env, rng):
        self.cfg, self.env, self.rng = cfg, env, rng
        self.dynamic = bool(cfg.all_stakes or cfg.all_white or cfg.curriculum)
        self.all_white = [*DECKS, "jokerless"]
        self.all_combos = [(d, s) for d in DECKS for s in range(1, 9)] + [("jokerless", 1)]
        if cfg.stake_only:
            self.all_combos = [c for c in self.all_combos if c[1] == cfg.stake_only and c[0] != "jokerless"]
        if cfg.all_stakes:
            self.unlocked = set(self.all_combos)
        elif cfg.all_white or (cfg.curriculum and cfg.all_decks):
            self.unlocked = {(d, 1) for d in self.all_white}
        else:
            self.unlocked = {(cfg.deck, cfg.stake)}
        self.passed = set()
        self.games, self.wins = {}, {}
        self.recent = {}
        E = env.n
        self.cur = [(cfg.deck, cfg.stake)] * E
        self.nxt = [(cfg.deck, cfg.stake)] * E

    def state(self):
        return {"unlocked": sorted(self.unlocked), "passed": sorted(self.passed), "combo_games": list(self.games.items()),
                "combo_wins": list(self.wins.items()), "combo_recent": [(k, list(v)) for k, v in self.recent.items()]}

    def load(self, st):
        if not (self.cfg.all_stakes or self.cfg.all_white):
            self.unlocked = set(map(tuple, st["unlocked"]))
        self.passed = set(map(tuple, st["passed"]))
        self.games = {tuple(k): v for k, v in st["combo_games"]}
        self.wins = {tuple(k): v for k, v in st["combo_wins"]}
        self.recent = {tuple(k): deque(map(tuple, v), maxlen=500) for k, v in st["combo_recent"]}

    def _sample(self):
        c = self.cfg
        if c.all_stakes:
            return self.all_combos[self.rng.integers(len(self.all_combos))]
        if c.all_white:
            return (self.all_white[self.rng.integers(len(self.all_white))], 1)
        pool = sorted(self.unlocked)
        if not c.curriculum_uniform:
            frontier = [k for k in pool if (k not in self.passed if c.unlock_rate > 0 else not self.wins.get(k))]
            if frontier and self.rng.random() < 0.5:
                pool = frontier
        return pool[self.rng.integers(len(pool))]

    def assign(self, i):
        self.nxt[i] = d, s = self._sample()
        if d == "jokerless":
            self.env.set_deck(i, "b_challenge", 1)
            self.env.set_challenge(i, "jokerless")
        else:
            self.env.set_deck(i, d, s)
            self.env.set_challenge(i, None)

    def start(self):
        if self.dynamic:
            for i in range(self.env.n):
                self.assign(i)
                self.cur[i] = self.nxt[i]

    def game_over(self, i, won, ante, counted):
        c = self.cur[i]
        if counted:
            self.games[c] = self.games.get(c, 0) + 1
            q = self.recent.setdefault(c, deque(maxlen=500))
            q.append((int(won), int(ante)))
            if won:
                self.wins[c] = self.wins.get(c, 0) + 1
            cfg = self.cfg
            if cfg.unlock_rate > 0:
                w = list(q)[-cfg.unlock_window:]
                ok = len(q) >= cfg.unlock_window and sum(x for x, _ in w) >= cfg.unlock_rate * cfg.unlock_window
            else:
                ok = won
            if ok and c not in self.passed and self.dynamic:
                self.passed.add(c)
                if cfg.curriculum:
                    if not cfg.all_decks and c[0] in DECKS and DECKS.index(c[0]) + 1 < len(DECKS):
                        self.unlocked.add((DECKS[DECKS.index(c[0]) + 1], 1))
                    if c[0] != "jokerless" and c[1] < 8:
                        self.unlocked.add((c[0], c[1] + 1))
        if self.dynamic:
            self.cur[i] = self.nxt[i]
            self.assign(i)


class StartStates:
    """With probability start_frac, an env's next game starts from a random start-pool state."""

    def __init__(self, cfg, env, pool_dir, seed, update):
        self.cfg, self.env = cfg, env
        self.on = cfg.start_frac > 0
        self.cur = np.full(env.n, -1)
        self.nxt = np.full(env.n, -1)
        if not self.on:
            return
        self.pool, self.points = load_pool(pool_dir, cfg.start_filter, cfg.start_ante_min, cfg.start_ante_max,
                                           challenge=challenge_id(cfg.challenge))
        assert self.points, f"no start points in {pool_dir} pass the filter (challenge {cfg.challenge})"
        self.rng = np.random.default_rng([seed, update, 4])
        print(f"start pool: {cfg.start_frac:.0%} of games start from {len(self.points)} points of {pool_dir}"
              f" ({'reseeded' if cfg.start_reseed else 'recorded seeds'})", flush=True)

    def assign(self, i):
        k = int(self.rng.integers(len(self.points))) if self.rng.random() < self.cfg.start_frac else -1
        self.nxt[i] = k
        self.env.set_start(i, self.pool, self.points[k]["sid"] if k >= 0 else -1, self.cfg.start_reseed)

    def start(self):
        if self.on:
            for i in range(self.env.n):
                self.assign(i)
                self.cur[i] = self.nxt[i]

    def after_reset(self):
        if self.on:
            for i in range(self.env.n):
                self.assign(i)

    def game_over(self, i):
        if not self.on:
            return None
        k = self.cur[i]
        self.cur[i] = self.nxt[i]
        self.assign(i)
        return self.points[k]["tags"]["ante"] if k >= 0 else None
