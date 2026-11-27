"""Deterministic PRNG Pipeline Emulation for Balatro (`game.engine.rng_sim.rng`).

Replicates Balatro's exact deterministic PRNG pipeline in native Python using
IEEE-754 64-bit float math and 64-bit unsigned integer arithmetic, modeled after
`balatro_source` (`functions/misc_functions.lua`) and LuaJIT 2.0.5 / 2.1 (`lj_prng.c`).
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")

# 64-bit unsigned integer mask
MASK64 = 0xFFFFFFFFFFFFFFFF


def double_to_u64(d: float) -> int:
    """Return raw IEEE-754 64-bit float representation as a 64-bit unsigned integer."""
    return struct.unpack("<Q", struct.pack("<d", d))[0]


def u64_to_double(u: int) -> float:
    """Reinterpret a 64-bit unsigned integer bit pattern as an IEEE-754 64-bit float."""
    return struct.unpack("<d", struct.pack("<Q", u & MASK64))[0]


def truncate13(x: float) -> float:
    """Replicate Lua's `math.abs(tonumber(string.format('%.13f', x)))`.
    
    LuaJIT formats through C runtime `sprintf('%.13f')`, which rounds
    ties to nearest even and parses back to float64.
    """
    if math.isnan(x):
        return float("nan")
    return abs(float(f"{x:.13f}"))


def advance(state: float) -> float:
    """Perform one deterministic LCG step on a PRNG stream state.
    
    Formula: `math.abs(tonumber(string.format('%.13f', (2.134453429141 + s * 1.72431234) % 1)))`
    """
    if math.isnan(state):
        return float("nan")
    raw = (2.134453429141 + state * 1.72431234) % 1.0
    return truncate13(raw)


def pseudohash(s: str | bytes) -> float:
    """Replicate Balatro's byte-iteration hash backward over the string.
    
    Balatro Lua (`functions/misc_functions.lua`):
    ```lua
    function pseudohash(str)
        if not str then return nil end
        local num = 1
        for i = #str, 1, -1 do
            num = ((1.1239285023 / num) * string.byte(str, i) * math.pi + math.pi * i) % 1
        end
        return num
    end
    ```
    """
    s_bytes = s.encode("utf-8") if isinstance(s, str) else s
    num = 1.0
    for i in range(len(s_bytes), 0, -1):
        if math.isnan(num) or num == 0.0:
            return float("nan")
        byte_val = s_bytes[i - 1]
        term = (1.1239285023 / num) * byte_val * math.pi + math.pi * i
        num = term % 1.0
    return num


def predict_seed(key: str, predict: str) -> float:
    """One-shot pseudoseed prediction without mutating global state.
    
    Formula: `(advance(pseudohash(key .. predict)) + pseudohash(predict)) / 2.0`
    """
    pseed = advance(pseudohash(key + predict))
    return (pseed + pseudohash(predict)) / 2.0


def pseudoseed(
    key: str, seed: str, current_state: float | None = None
) -> tuple[float, float]:
    """Compute / advance stream state for `key` and `seed`.
    
    Returns `(next_stream_state, seed_float_for_tw223)`.
    If `current_state` is None, this is the first draw for the stream,
    so state is initialized via `pseudohash(key + seed)`.
    """
    hashed_seed = pseudohash(seed)
    if current_state is None:
        initial_hash = pseudohash(key + seed)
        st = advance(initial_hash)
    else:
        st = advance(current_state)
    seed_float = (st + hashed_seed) / 2.0
    return st, seed_float


# ---------------------------------------------------------------------------
# LuaJIT Tausworthe PRNG (Tw223 / lj_prng.c)
# ---------------------------------------------------------------------------


def _gen_word(z: int, k: int, q: int, s: int) -> int:
    return ((((z << q) & MASK64) ^ z) >> (k - s)) ^ (
        ((z & ((MASK64 << (64 - k)) & MASK64)) << s) & MASK64
    )


def _step_word(z: int, word: int) -> int:
    if word == 0:
        return _gen_word(z, 63, 31, 18)
    elif word == 1:
        return _gen_word(z, 58, 19, 28)
    elif word == 2:
        return _gen_word(z, 55, 24, 7)
    else:
        return _gen_word(z, 47, 21, 8)


class Tw223:
    """LuaJIT's combined Tausworthe LFSR pseudo-random number generator (`lj_prng.c`).
    
    Maintains a 4-word 64-bit state seeded from an IEEE-754 double precision float.
    Produces identical random float and integer sequences to LuaJIT's `math.random()`.
    """

    def __init__(self, seed: float) -> None:
        self.u: list[int] = [0, 0, 0, 0]
        r = 0x11090601
        d = seed
        for i in range(4):
            m = 1 << (r & 0xFF)
            r >>= 8
            d = d * math.pi + 2.7182818284590452354
            bits = double_to_u64(d)
            if bits < m:
                bits += m
            z = bits & MASK64
            for _ in range(10):
                z = _step_word(z, i)
            self.u[i] = z

    def random(self) -> float:
        """Draw a uniform random float in `[0.0, 1.0)`. Replicates `math.random()`."""
        r = 0
        for i in range(4):
            self.u[i] = _step_word(self.u[i], i)
            r ^= self.u[i]
        bits = (r & 0x000FFFFFFFFFFFFF) | 0x3FF0000000000000
        return u64_to_double(bits) - 1.0

    def random_n(self, m: int) -> int:
        """Draw an integer in `[1, m]`. Replicates `math.random(m)`."""
        return math.floor(self.random() * m) + 1

    def random_range(self, m: int, n: int) -> int:
        """Draw an integer in `[m, n]`. Replicates `math.random(m, n)`."""
        return math.floor(self.random() * (n - m + 1)) + m

    def randint(self, min_val: int, max_val: int) -> int:
        """Alias for `random_range(min_val, max_val)`."""
        return self.random_range(min_val, max_val)

    def element(self, items: Sequence[T]) -> tuple[T, int]:
        """Deterministic element selection matching Balatro's `pseudorandom_element`.
        
        Returns `(chosen_item, 0_based_index)`.
        """
        if not items:
            raise ValueError("Cannot select element from empty sequence")
        idx = self.random_n(len(items)) - 1
        return items[idx], idx

    def shuffle(self, items: list[Any]) -> None:
        """In-place Fisher-Yates shuffle matching Balatro's `pseudoshuffle`."""
        for i in range(len(items) - 1, 0, -1):
            j = self.random_range(1, i + 1) - 1
            items[i], items[j] = items[j], items[i]

    def get_state(self) -> list[int]:
        """Return a copy of the 4-word internal PRNG state."""
        return list(self.u)

    def set_state(self, state: Sequence[int]) -> None:
        """Set the 4-word internal PRNG state."""
        if len(state) != 4:
            raise ValueError("Tw223 state must contain exactly 4 64-bit words")
        self.u = [s & MASK64 for s in state]


# ---------------------------------------------------------------------------
# Named PRNG Stream Channel
# ---------------------------------------------------------------------------


@dataclass
class PRNGChannel:
    """Stateful RNG stream channel corresponding to one Balatro key (e.g. `'lucky_mult'`)."""

    key: str
    state: float  # Current raw LCG state
    call_count: int = 0
    history: list[float] = field(default_factory=list)

    def checkpoint(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "state": self.state,
            "call_count": self.call_count,
            "history": list(self.history),
        }

    @classmethod
    def from_checkpoint(cls, data: dict[str, Any]) -> PRNGChannel:
        return cls(
            key=data["key"],
            state=data["state"],
            call_count=data["call_count"],
            history=list(data.get("history", [])),
        )


# ---------------------------------------------------------------------------
# Master Balatro RNG Pipeline
# ---------------------------------------------------------------------------


class BalatroRNG:
    """Master deterministic RNG engine for a Balatro run.
    
    Maintains independent, named PRNG stream channels (`'cert_fr'`, `'certsl'`,
    `'lucky_mult'`, `'lucky_money'`, `'cas1'`, `'anc1'`, `'idol1'`, `'mail1'`, `'to_do'`,
    `'Tarotcar1'`, `'Tarotvag1'`, `'marb_fr'`, `'boss'`, `'Tag1'`, `'Voucher1'`, `'Voucher_fromtag'`, etc.)
    with bit-exact LCG stepping, state inspection, and checkpoint/fork capabilities.
    """

    def __init__(self, seed: str, hashed_seed: float | None = None) -> None:
        self.seed: str = str(seed).strip().upper()
        self.hashed_seed: float = (
            pseudohash(self.seed.encode("utf-8"))
            if hashed_seed is None
            else hashed_seed
        )
        self.channels: dict[str, PRNGChannel] = {}

    def _get_or_init_channel(self, key: str) -> PRNGChannel:
        """Retrieve existing channel or initialize with `pseudohash(key + seed)`."""
        if key not in self.channels:
            init_hash = pseudohash((key + self.seed).encode("utf-8"))
            st = advance(init_hash)
            ch = PRNGChannel(key=key, state=st, call_count=1, history=[st])
            self.channels[key] = ch
            return ch
        ch = self.channels[key]
        ch.state = advance(ch.state)
        ch.call_count += 1
        ch.history.append(ch.state)
        return ch

    def step_seed(self, key: str) -> float:
        """Advance stream channel `key` by one LCG step and return the seed float for Tw223."""
        ch = self._get_or_init_channel(key)
        return (ch.state + self.hashed_seed) / 2.0

    def ensure_call_count(self, key: str, target_calls_before_draw: int) -> None:
        """Advance channel `key` if its current `call_count` is less than `target_calls_before_draw`.
        
        Because Balatro's LCG stream step `st = advance(st)` does not depend on deck size or
        external game state, this allows jumping directly to round `r` within an ante without
        having to construct or evaluate intermediate rounds' decks!
        """
        current = self.get_channel_call_count(key)
        if current < target_calls_before_draw:
            self.advance(key, target_calls_before_draw - current)

    def random(self, key: str) -> float:
        """Advance stream channel `key` and return 1 uniform float in `[0.0, 1.0)`."""
        pseed = self.step_seed(key)
        tw = Tw223(pseed)
        return tw.random()

    def randint(self, key: str, min_val: int, max_val: int) -> int:
        """Advance stream channel `key` and return 1 integer in `[min_val, max_val]`."""
        pseed = self.step_seed(key)
        tw = Tw223(pseed)
        return tw.randint(min_val, max_val)

    def random_n(self, key: str, m: int) -> int:
        """Advance stream channel `key` and return 1 integer in `[1, m]`."""
        pseed = self.step_seed(key)
        tw = Tw223(pseed)
        return tw.random_n(m)

    def choice(self, key: str, items: Sequence[T]) -> tuple[T, int]:
        """Advance stream channel `key` and pick an element from `items`.
        
        Returns `(chosen_item, index)`.
        """
        pseed = self.step_seed(key)
        tw = Tw223(pseed)
        return tw.element(items)

    def poll_odds(
        self, key: str, numerator: int, denominator: int, modifier: float = 1.0
    ) -> bool:
        """Evaluate a 1-in-N or fractional probability check."""
        val = self.random(key)
        threshold = (numerator * modifier) / float(denominator)
        return val < threshold

    def shuffle(self, key: str, items: list[Any]) -> None:
        """In-place Fisher-Yates shuffle matching Balatro's `pseudoshuffle(items, pseudoseed(key))`."""
        pseed = self.step_seed(key)
        tw = Tw223(pseed)
        tw.shuffle(items)

    def advance(self, key: str, steps: int = 1) -> float:
        """Advance stream channel `key` by `steps` LCG calls without instantiating Tw223."""
        if steps <= 0:
            if key in self.channels:
                return self.channels[key].state
            return pseudohash((key + self.seed).encode("utf-8"))
        for _ in range(steps):
            self.step_seed(key)
        return self.channels[key].state

    def inspect_state(self, key: str) -> float | None:
        """Return the current raw LCG state for `key` if initialized, or None."""
        ch = self.channels.get(key)
        return ch.state if ch else None

    def get_channel_call_count(self, key: str) -> int:
        """Return how many times stream `key` has been stepped."""
        ch = self.channels.get(key)
        return ch.call_count if ch else 0

    def get_all_states(self) -> dict[str, float]:
        """Return a mapping of all active stream channel keys to their current LCG state."""
        return {k: ch.state for k, ch in self.channels.items()}

    def get_all_call_counts(self) -> dict[str, int]:
        """Return call counts for all active stream channels."""
        return {k: ch.call_count for k, ch in self.channels.items()}

    def checkpoint(self) -> dict[str, Any]:
        """Create a serializable checkpoint snapshot of this RNG pipeline."""
        return {
            "seed": self.seed,
            "hashed_seed": self.hashed_seed,
            "channels": {k: ch.checkpoint() for k, ch in self.channels.items()},
        }

    def restore(self, checkpoint: dict[str, Any]) -> None:
        """Restore pipeline state from a checkpoint dictionary."""
        self.seed = checkpoint["seed"]
        self.hashed_seed = checkpoint["hashed_seed"]
        self.channels = {
            k: PRNGChannel.from_checkpoint(v)
            for k, v in checkpoint["channels"].items()
        }

    def fork(self) -> BalatroRNG:
        """Create an independent clone / fork of this RNG pipeline."""
        forked = BalatroRNG(self.seed, self.hashed_seed)
        forked.channels = {
            k: PRNGChannel(
                key=ch.key,
                state=ch.state,
                call_count=ch.call_count,
                history=list(ch.history),
            )
            for k, ch in self.channels.items()
        }
        return forked

    def clone(self) -> BalatroRNG:
        """Alias for `fork()`."""
        return self.fork()

