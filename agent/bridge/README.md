# antezero bridge

Drives the real Balatro (Steam, Windows) through
[BalatroBot](https://github.com/coder/balatrobot) plus a companion mod
(`mod/antezero_bridge`), with the simulator's action vocabulary and the
oracle's state format, so real-game runs can be compared with the simulator
and the headless oracle step by step.

Stack: lovely (0.10+) + Steamodded + BalatroBot (1.5.x) + the companion mod,
which `launch` installs into the game's `Mods` folder. The companion blocks
every profile / settings / run save write while the bridge runs
(`AZB_ALLOW_SAVES=1` would allow them) and unlocks everything in memory only,
like the oracle; `replay` checks that the player's save files are unchanged.

```
uv run python -m antezero_bridge.serve --check     # what is installed where (no side effects)
uv run python -m antezero_bridge.serve --install   # copy the companion mod into Mods/
uv run python -m antezero_bridge.serve --serve     # start the game with parity settings
uv run python -m antezero_bridge.replay .scratch/fail_T0000004_b_red_1.txt --oracle-cli target/release/ocli.exe
uv run python -m antezero_bridge.compare game.jsonl oracle.jsonl
```

```python
from antezero_bridge import GameEnv, Action
with GameEnv.launch() as env:           # parity settings (dt 1/60, GAMESPEED 1)
    env.reset("TESTSEED", "b_red", 1)
    env.step("select")
    env.step(Action.play(0b11111))
    print(env.canonical()["dollars"], env.legal_actions()[:5])
```

| module | what |
|---|---|
| `actions`, `injects` | action / injection codes (the simulator's grammar), translation to the oracle's `H.act` / `H.inject` and to native BalatroBot calls |
| `game_env` | `GameEnv`: the real game; `route="harness"` (default) performs every action like the oracle harness via the companion, `"balatrobot"` uses native endpoints where they are reliable |
| `sim_env` | `SimEnv`: the same interface over the simulator (`antezero_sim`) |
| `canon`, `compare` | the canonical comparison view (port of `sim/oracle/src/canon.rs`), stream diffs |
| `client`, `launch` | JSON-RPC client, game process management and installation |
| `replay` | replay a simulator history in the game (optionally diffed against the oracle) |
| `mod/antezero_bridge` | the companion mod: harness-faithful actions, settle waiting, legal actions, `H.state`-shaped dumps |

Offline tests (`uv run --with pytest pytest agent/bridge/tests`) cover the
codecs against the simulator's parsers, the generated tables
(`scripts/gen_codes.py --check`), the canonicalizer against the simulator on
oracle dumps (with the oracle built), launch configuration and the client
protocol. The companion's legal-action enumeration is checked against the
simulator in the oracle (`cargo test -p oracle --test bridge_legal --
--include-ignored`). Anything else needs the game running.
