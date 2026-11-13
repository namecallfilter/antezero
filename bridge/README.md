# antezero bridge

The real Balatro (Steam, Windows) behind the simulator's action vocabulary
and the oracle's state format, through the `AnteZero` mod (`mod/` at the repository root).
Two uses:

- **The bot button** (for playing): a toggle on the New Run screen's Play
  button; a run started with it on is played by the agent. See the root
  README for the setup. The mod needs lovely (0.10+) and Steamodded only.
- **The developer tools** (below): real-game runs compared with the
  simulator and the headless oracle step by step.

## The bot button

`python -m antezero_agent.bot --install CKPT` copies the mod into the game's
`Mods` folder and stores the command that starts the agent in
`%APPDATA%/Balatro/antezero.json`. The mod starts that process the first time
a bot run begins and talks to it over JSON-RPC on 127.0.0.1:12346
(`AZB_PORT`).

A bot run is played in step with the simulator (same seed, deck, stake and
actions), so the mod gives it the oracle's conditions for its duration:
seeded, everything unlocked in memory, a fixed 1/60 s clock at game speed 1
(the bot's speed setting is the number of such updates per 1/60 s). Search
and that speed are the mod's settings (Mods > AnteZero > Config; 1x and
search on by default). None
of it reaches the player's files: save writes are blocked during the run and
the profile, unlocks and tutorial state are put back when it is left. While
the bot plays, input is limited to Esc, Run Info and Options, whose menus
pause it; a strip at the top shows its last action, how sure the policy was,
and the critic's win estimate. If the game and the simulator ever disagree
the bot stops and hands the run back.

## Developer tools

These start the game themselves, with
[BalatroBot](https://github.com/coder/balatrobot) (1.5.x) as the launcher and
RPC server. BalatroBot mutes the game and overrides its settings whenever it
loads, so its mod stays disabled (`Mods/balatrobot/.lovelyignore`) except
inside these launches. In such a session the mod blocks every profile /
settings / run save write from the start (`AZB_ALLOW_SAVES=1` would allow
them) and unlocks everything in memory only, like the oracle; `replay` checks
that the player's save files are unchanged.

```
uv run python -m antezero_bridge.serve --check     # what is installed where (no side effects)
uv run python -m antezero_bridge.serve --install   # copy the mod into Mods/
uv run python -m antezero_bridge.serve --serve     # start the game with parity settings
uv run python -m antezero_bridge.replay .scratch/fail_T0000004_b_red_1.txt --oracle-cli target/release/ocli.exe
uv run python -m antezero_bridge.compare game.jsonl oracle.jsonl
uv run python -m antezero_agent.play_real CKPT --seeds 20 --round-search --device cuda   # win rate in the real game
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
| `game_env` | `GameEnv`: the real game; `route="harness"` (default) performs every action like the oracle harness via the mod, `"balatrobot"` uses native endpoints where they are reliable |
| `sim_env` | `SimEnv`: the same interface over the simulator (`antezero_sim`) |
| `canon`, `compare` | the canonical comparison view (port of `sim/oracle/src/canon.rs`), stream diffs |
| `client`, `launch` | JSON-RPC client, game process management and installation |
| `replay` | replay a simulator history in the game (optionally diffed against the oracle) |
| `mod/` | the mod: the bot button, harness-faithful actions, settle waiting, legal actions, `H.state`-shaped dumps |

Offline tests (`uv run --with pytest pytest bridge/tests`) cover the
codecs against the simulator's parsers, the generated tables
(`scripts/gen_codes.py --check`), the canonicalizer against the simulator on
oracle dumps (with the oracle built), launch configuration and the client
protocol. The mod's legal-action enumeration is checked against the
simulator in the oracle (`cargo test -p oracle --test bridge_legal --
--include-ignored`). Anything else needs the game running.
