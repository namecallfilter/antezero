"""``SimEnv``: the :class:`~antezero_bridge.env.Env` interface over the Rust
simulator's Python bindings (``antezero_sim``, built from ``sim/py``).

The bindings are text based like the bridge (action / inject codes, the
canonical ``str -> str`` state dict), so this adapter only maps the
interface: ``reset`` creates an ``antezero_sim.Game``, ``step`` / ``inject``
apply codes, ``canonical`` is key-for-key comparable with
:func:`antezero_bridge.canon.from_lua` of a real-game dump.
"""

from __future__ import annotations

from typing import Any

from .actions import Action, parse_action
from .canon import Canon
from .env import Obs, back_key, stake_num
from .injects import Inject, parse_inject


class SimEnv:
    def __init__(self, module: Any | None = None):
        if module is None:
            import antezero_sim as module  # type: ignore[import-not-found, no-redef]
        self._m = module
        self._g: Any = None

    def reset(self, seed: str, deck: str = "b_red", stake: int | str = 1) -> Obs:
        self._g = self._m.Game(seed, back_key(deck), stake_num(stake))
        return self.state()

    def step(self, action: Action | str) -> Obs:
        self._g.step(parse_action(action).code())
        return self.state()

    def inject(self, inj: Inject | str) -> Obs:
        self._g.inject(parse_inject(inj).code)
        return self.state()

    def legal_actions(self) -> list[Action]:
        return [parse_action(c) for c in self._g.legal_codes()]

    def state(self) -> Obs:
        """The canonical view (the simulator has no richer dump)."""
        return dict(self.canonical())

    def canonical(self) -> Canon:
        return dict(self._g.canonical())

    def is_terminal(self) -> bool:
        return bool(self._g.is_terminal())

    def close(self) -> None:
        self._g = None
