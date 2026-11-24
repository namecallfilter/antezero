"""Balatro (1.0.1o) RLBalatro-compatible environments backed by `antezero_sim`."""
import json
from pathlib import Path

import antezero_sim
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    gym = None

DATA = Path(__file__).resolve().parents[2] / "data"
IDS = json.loads((DATA / "ids.json").read_text())

CARD_FEATURES = [
    "present", "center_id", "rank", "suit", "edition", "seal", "debuff", "face_down", "selected",
    "cost", "sell_cost", "eternal", "perishable", "rental", "mult", "x_mult", "chips", "extra0",
    "extra1", "times_played", "unused20", "unused21",
    "joker_value", "joker_count", "is_face", "suit_spades", "suit_hearts", "suit_clubs", "suit_diamonds",
    "has_rank", "forced", "fib", "even", "odd", "hack", "idol_match", "mail_match", "ancient_match",
    "castle_match",
]
CHALLENGES = {"jokerless": "c_jokerless_1"}
DECKS = [
    "b_red", "b_blue", "b_yellow", "b_green", "b_black", "b_magic", "b_nebula", "b_ghost",
    "b_abandoned", "b_checkered", "b_zodiac", "b_painted", "b_anaglyph", "b_plasma", "b_erratic",
]
COUNT_NAMES = ["hand", "jokers", "consumables", "shop", "vouchers", "boosters", "pack", "deck", "tags", "discard"]
N_HAND = 24
HS_CTX = 32


def challenge_id(name):
    if not name or name == "none":
        return ""
    cid = CHALLENGES.get(name, name)
    assert cid in CHALLENGES.values(), f"unknown challenge {name!r} (known: {', '.join(CHALLENGES)})"
    return cid


def _layout():
    return antezero_sim.balatro_obs_layout()


JOKER_KEYS = list(antezero_sim.balatro_joker_keys())
JOKER_INDEX = {k: i for i, k in enumerate(JOKER_KEYS)}
VOUCHER_KEYS = list(antezero_sim.balatro_voucher_keys())
HANDS = list(antezero_sim.balatro_hands())
HAND_INDEX = {h: k for k, h in enumerate(HANDS)}
ROUND_CTX_FEATURES = (
    ["last_score", "last_chips", "last_mult", "last_blind_share"]
    + [f"last_{h}" for h in HANDS]
    + [f"played_{h}" for h in HANDS]
    + [f"forbid_{h}" for h in HANDS]
    + ["chips_small", "chips_big", "chips_boss", "dollars_small", "dollars_big", "dollars_boss"]
    + [f"voucher_{v}" for v in VOUCHER_KEYS]
)
SEL_FEATURES = (
    ["n_selected", "poker_hand", "level", "chips", "mult", "n_scoring", "boss_disallows", "chips_needed"]
    + [f"contains_{i}" for i in range(12)]
    + [f"scoring_{i}" for i in range(N_HAND)]
)
PLAY_HAND_ACTIONS = ["PLAY_" + h.upper().replace(" ", "_") for h in HANDS if h != "High Card"]


def _action_names():
    J, C, S, V, B, P = 20, 10, 8, 4, 4, 8
    names = ["PLAY", "DISCARD", "SELECT_BLIND", "SKIP_BLIND", "REROLL_BOSS", "CASH_OUT", "END_SHOP", "REROLL"]
    names += [f"BUY_{i}" for i in range(S)] + [f"BUY_USE_{i}" for i in range(S)]
    names += [f"REDEEM_{i}" for i in range(V)] + [f"OPEN_{i}" for i in range(B)] + [f"PICK_{i}" for i in range(P)]
    names += ["SKIP_PACK"] + [f"SELL_JOKER_{i}" for i in range(J)] + [f"SELL_CONS_{i}" for i in range(C)]
    names += [f"USE_CONS_{i}" for i in range(C)] + [f"JOKER_RIGHT_{i}" for i in range(J - 1)]
    names += [f"JOKER_FRONT_{i}" for i in range(J)] + [f"JOKER_BACK_{i}" for i in range(J)]
    names += ["REORDER_JOKERS"] + PLAY_HAND_ACTIONS
    return names


ACTIONS = _action_names()
SEL_VERBS = (
    ["PLAY", "DISCARD"]
    + [f"USE_CONS_{i}" for i in range(10)]
    + [f"PICK_{i}" for i in range(8)]
    + ["REORDER_JOKERS"]
    + PLAY_HAND_ACTIONS
)
N_SEL_VERBS = len(SEL_VERBS)


def _split(buf, groups):
    out = {}
    for name, off, r, c in groups:
        a = buf[..., off : off + r * c]
        out[name] = a.reshape(buf.shape[:-1] + ((c,) if r == 1 else (r, c)))
    return out


class StartPool:
    """Mid-run states loaded from `.bal` scripts for starting episodes (`set_start`)."""

    def __init__(self):
        self._p = antezero_sim.StartPool()

    def __len__(self):
        return len(self._p)

    def error(self):
        return self._p.error

    def add(self, script, ks):
        ks_list = [int(x) for x in ([ks] if isinstance(ks, int) else ks)]
        ids = self._p.add(str(script), ks_list)
        return int(ids[0]) if isinstance(ks, int) else np.asarray(ids, dtype=np.int32)

    def script(self, point_id):
        return self._p.script(int(point_id))

    def challenge(self, point_id):
        return self._p.challenge(int(point_id))

    def close(self):
        self._p = None


class BalatroEnv(gym.Env if gym else object):
    """Single Balatro environment with RLBalatro's exact observation and action layout."""

    metadata = {"render_modes": []}

    def __init__(
        self, deck="b_red", stake=1, max_steps=5000, reward_blind=1.0, reward_win=10.0,
        reward_lose=0.0, reward_lose_partial=True, record=False, max_free_actions=0,
        reward_win_log=False, endless_ante=0, reward_money=False, money_per_reward=15.0,
        reward_best_score=False, reward_skip_deferred=False, disable_softlocks=False,
        challenge=None, obs_hide_face_down=False, reward_chips=0.0, reward_blue_seal=0.0,
        step_cost=0.0, toggle_cost=0.0, sticker_penalty=1.0,
    ):
        assert not (reward_best_score and reward_money), "reward_best_score and reward_money are exclusive"
        cid = challenge_id(challenge)
        if cid or deck == "b_challenge":
            deck = "b_red"
        assert deck in DECKS, f"unknown deck {deck!r}"
        self.deck, self.stake, self.max_steps = deck, int(stake), int(max_steps)
        self.groups, self.obs_size = _layout()
        self.actions = ACTIONS
        self.n_actions = len(self.actions)
        self.sel_verbs = SEL_VERBS
        self._core = antezero_sim.BalatroCoreEnv(
            deck=deck, stake=int(stake), max_steps=int(max_steps),
            reward_blind=float(reward_blind), reward_win=float(reward_win),
            reward_win_log=bool(reward_win_log), endless_ante=int(endless_ante),
            reward_lose=float(reward_lose), reward_lose_partial=bool(reward_lose_partial),
            reward_money=bool(reward_money), money_per_reward=float(money_per_reward),
            reward_chips=float(reward_chips), reward_blue_seal=float(reward_blue_seal),
            reward_skip_deferred=bool(reward_skip_deferred), step_cost=float(step_cost),
            toggle_cost=float(toggle_cost), disable_softlocks=bool(disable_softlocks),
            obs_hide_face_down=bool(obs_hide_face_down), sticker_penalty=float(sticker_penalty),
            record=bool(record),
        )
        self._buf = np.zeros(self.obs_size, dtype=np.float32)
        self._selbuf = np.zeros(len(SEL_FEATURES), dtype=np.float32)
        self._mask = np.zeros(self.n_actions, dtype=np.uint8)
        self._specs = np.zeros((len(self.sel_verbs), 3), dtype=np.int32)
        self._hctx = np.zeros(HS_CTX, dtype=np.int32)
        self._seed_counter = 0

    @property
    def obs_buffer(self):
        return self._buf

    def set_deck(self, deck, stake):
        if deck == "b_challenge":
            deck = "b_red"
        self.deck, self.stake = deck, int(stake)
        self._core.set_deck(deck, int(stake))

    def set_challenge(self, challenge=None):
        return None

    def set_start(self, pool=None, i=-1, reseed=True):
        self._pool = pool
        self._core.set_start(pool._p if pool is not None else None, int(i) if pool is not None else -1, bool(reseed))

    def episode_start(self):
        return int(self._core.episode_start())

    def action_masks(self):
        return self._mask.view(bool)

    def sel_specs(self):
        return self._specs.copy()

    def hand_ctx(self):
        return self._hctx.copy()

    def _obs(self):
        d = _split(self._buf.copy(), self.groups)
        d["selection"] = self._selbuf.copy()
        return d

    def reset(self, seed=None, options=None):
        if options and "seed_str" in options:
            self._core.reset_into(None, str(options["seed_str"]), self._buf, self._selbuf, self._mask, self._specs, self._hctx)
        elif isinstance(seed, str):
            self._core.reset_into(None, seed, self._buf, self._selbuf, self._mask, self._specs, self._hctx)
        else:
            if seed is None:
                seed = self._seed_counter
                self._seed_counter += 1
            self._core.reset_into(int(seed), None, self._buf, self._selbuf, self._mask, self._specs, self._hctx)
        return self._obs(), {"action_mask": self.action_masks()}

    def step(self, action):
        a, cards = action if isinstance(action, tuple) else (action, ())
        picks = [int(c) for c in cards if int(c) >= 0]
        info = dict(self._core.step_into(int(a), picks, self._buf, self._selbuf, self._mask, self._specs, self._hctx))
        info["action_mask"] = self.action_masks()
        for k in ("illegal", "won", "dead_end"):
            info[k] = bool(info[k])
        return self._obs(), float(info["reward"]), bool(info["terminated"]), bool(info["truncated"]), info

    def script(self):
        return self._core.script()

    def close(self):
        self._core = None


class BalatroVecEnv:
    """N environments stepped in parallel in Rust with Rayon and auto-reset on episode end."""

    def __init__(
        self, n, deck="b_red", stake=1, max_steps=5000, reward_blind=1.0, reward_win=10.0,
        reward_lose=0.0, reward_lose_partial=True, max_free_actions=0, reward_win_log=False,
        endless_ante=0, reward_money=False, money_per_reward=15.0, reward_best_score=False,
        reward_skip_deferred=False, threads=None, seed_start=0, disable_softlocks=False,
        challenge=None, obs_hide_face_down=False, reward_chips=0.0, reward_blue_seal=0.0,
        step_cost=0.0, toggle_cost=0.0, sticker_penalty=1.0,
    ):
        cid = challenge_id(challenge)
        if cid or deck == "b_challenge":
            deck = "b_red"
        self.n = int(n)
        self.num_envs = self.n
        self.threads = int(threads or 16)
        self.groups, self.obs_size = _layout()
        self.actions = ACTIONS
        self.n_actions = len(self.actions)
        self.sel_verbs = SEL_VERBS
        self._core = antezero_sim.BalatroCoreVecEnv(
            num_envs=self.n, threads=self.threads, deck=deck, stake=int(stake),
            max_steps=int(max_steps), reward_blind=float(reward_blind), reward_win=float(reward_win),
            reward_win_log=bool(reward_win_log), endless_ante=int(endless_ante),
            reward_lose=float(reward_lose), reward_lose_partial=bool(reward_lose_partial),
            reward_money=bool(reward_money), money_per_reward=float(money_per_reward),
            reward_chips=float(reward_chips), reward_blue_seal=float(reward_blue_seal),
            reward_skip_deferred=bool(reward_skip_deferred), step_cost=float(step_cost),
            toggle_cost=float(toggle_cost), disable_softlocks=bool(disable_softlocks),
            obs_hide_face_down=bool(obs_hide_face_down), sticker_penalty=float(sticker_penalty),
            record=False,
        )
        self._obs = np.zeros((self.n, self.obs_size), dtype=np.float32)
        self._sel = np.zeros((self.n, len(SEL_FEATURES)), dtype=np.float32)
        self._masks = np.zeros((self.n, self.n_actions), dtype=np.uint8)
        self._specs = np.zeros((self.n, len(self.sel_verbs), 3), dtype=np.int32)
        self._hctx = np.zeros((self.n, HS_CTX), dtype=np.int32)
        self._cards = np.full((self.n, 5), -1, dtype=np.int32)
        self.next_seed = int(seed_start)
        self.reset_seeds = np.zeros(self.n, dtype=np.uint64)
        self._seed_ar = np.arange(self.n, dtype=np.uint64)

    @property
    def obs_buffer(self):
        return self._obs

    def set_deck(self, i, deck, stake):
        if deck == "b_challenge":
            deck = "b_red"
        self._core.set_deck(int(i), deck, int(stake))

    def set_challenge(self, i, challenge=None):
        return None

    def set_start(self, i, pool=None, k=-1, reseed=True):
        self._core.set_start(int(i), pool._p if pool is not None else None, int(k) if pool is not None else -1, bool(reseed))

    def _obs_dict(self):
        obs = _split(self._obs, self.groups)
        obs["selection"] = self._sel
        return obs

    def _infos_now(self, infos):
        infos["action_mask"] = self._masks.view(bool)
        infos["sel_spec"] = self._specs
        infos["hand_ctx"] = self._hctx
        return infos

    def reset(self, seeds=None):
        if seeds is None:
            seeds = np.arange(self.next_seed, self.next_seed + self.n, dtype=np.uint64)
            self.next_seed += self.n
        seeds = np.ascontiguousarray(seeds, dtype=np.uint64)
        self._core.reset_all(seeds, self._obs, self._sel, self._masks, self._specs, self._hctx)
        return self._obs_dict(), self._infos_now({})

    def step(self, actions, cards=None, reset_seeds=None):
        if isinstance(actions, tuple):
            actions, cards = actions
        a = np.ascontiguousarray(actions, dtype=np.int32)
        if reset_seeds is None:
            np.add(self._seed_ar, self.next_seed, out=self.reset_seeds)
            self.next_seed += self.n
        else:
            self.reset_seeds[:] = reset_seeds
        c = self._cards if cards is None else np.ascontiguousarray(cards, dtype=np.int32)
        infos = dict(
            self._core.step_all(
                a, c, self.reset_seeds, self._obs, self._sel, self._masks, self._specs, self._hctx
            )
        )
        infos = self._infos_now(infos)
        return (
            self._obs_dict(),
            infos["reward"].astype(np.float32),
            infos["terminated"].astype(bool),
            infos["truncated"].astype(bool),
            infos,
        )

    def close(self):
        self._core = None
