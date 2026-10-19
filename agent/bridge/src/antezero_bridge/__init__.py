"""antezero bridge: drive the real Balatro game (BalatroBot + companion mod)
and the Rust simulator through one environment interface.

Quick start::

    from antezero_bridge import GameEnv, Action
    with GameEnv.launch() as env:            # starts Balatro with parity settings
        env.reset("TESTSEED", "b_red", 1)
        env.step("select")
        env.step(Action.play(0b11111))
        print(env.canonical()["dollars"], env.legal_actions()[:5])
"""

from .actions import Action
from .client import BridgeClient, RPCError, SettleTimeout, StateDump
from .env import Env
from .game_env import ActionRejected, GameEnv
from .injects import Inject
from .launch import FAST, PARITY, GameProcess, LaunchConfig, launch
from .sim_env import SimEnv

__all__ = [
    "Action",
    "ActionRejected",
    "BridgeClient",
    "Env",
    "FAST",
    "GameEnv",
    "GameProcess",
    "Inject",
    "LaunchConfig",
    "PARITY",
    "RPCError",
    "SettleTimeout",
    "SimEnv",
    "StateDump",
    "launch",
]
