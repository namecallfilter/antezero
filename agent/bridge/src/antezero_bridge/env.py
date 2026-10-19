"""Backend-neutral environment interface.

Both backends speak the sim's action vocabulary (:mod:`antezero_bridge.actions`)
and expose the same flattened comparison view (``canonical()``: the
``dict[str, str]`` that ``sim/src/canon.rs`` and ``sim/oracle/src/canon.rs``
build), so an agent, a replay or a differential test can be written once and
run on either:

* :class:`antezero_bridge.game_env.GameEnv` - the real game (BalatroBot +
  companion mod).
* :class:`antezero_bridge.sim_env.SimEnv` - the Rust simulator
  (``antezero_sim``).

Observations are backend-native rich dicts (for ``GameEnv`` the oracle
``H.state`` JSON shape); use ``canonical()`` whenever two backends must be
compared.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from . import _codes
from .actions import Action
from .canon import Canon
from .injects import Inject

Obs = dict[str, Any]

# BalatroBot deck enum -> back key, for the decks a run can start with.
DECKS = {k.removeprefix("b_").upper(): k for k in _codes.BACKS if k != "b_challenge"}
STAKES = {"WHITE": 1, "RED": 2, "GREEN": 3, "BLACK": 4, "BLUE": 5, "PURPLE": 6, "ORANGE": 7, "GOLD": 8}


def back_key(deck: str) -> str:
    """Accept ``'b_red'``, ``'RED'`` or ``'red'`` (``ValueError`` otherwise)."""
    key = deck if deck.startswith("b_") else DECKS.get(deck.upper(), "")
    if key not in DECKS.values():
        raise ValueError(f"unknown or unsupported deck {deck!r}")
    return key


def stake_num(stake: int | str) -> int:
    """Accept ``1``..``8``, ``'3'`` or a stake name (``'GOLD'``)."""
    n = stake if isinstance(stake, int) else (int(stake) if str(stake).isdigit() else STAKES.get(str(stake).upper(), 0))
    if not 1 <= n <= 8:
        raise ValueError(f"unknown stake {stake!r}")
    return n


@runtime_checkable
class Env(Protocol):
    """What every backend implements."""

    def reset(self, seed: str, deck: str = "b_red", stake: int | str = 1) -> Obs:
        """Start a seeded run (``deck`` = back key or BalatroBot enum) and
        return the observation at the first decision point (BLIND_SELECT)."""
        ...

    def legal_actions(self) -> list[Action]:
        """All legal actions at the current decision point, as enumerated by
        ``Game::legal_actions`` (moves/sorts are legal but not enumerated)."""
        ...

    def step(self, action: Action | str) -> Obs:
        """Apply one action and run until the next decision point."""
        ...

    def inject(self, inj: Inject | str) -> Obs:
        """Apply a scenario injection (``inject_*`` history line)."""
        ...

    def state(self) -> Obs:
        """Backend-native rich state of the current decision point."""
        ...

    def canonical(self) -> Canon:
        """Flattened comparison view (canon.rs format)."""
        ...

    def is_terminal(self) -> bool:
        """Lost, or won (round eval after the winning ante) like ``Game::is_terminal``."""
        ...

    def close(self) -> None: ...
