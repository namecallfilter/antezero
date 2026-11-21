# antezero bridge

The real Balatro (Steam, Windows) behind the simulator's action vocabulary
and the oracle's state format, through the `AnteZero` mod (`mod/` at the repository root).
Two uses:

- **The bot button** (for playing): a toggle in a run's options menu; with
  it on, the agent plays the run. See the root README for the setup. The mod
  needs lovely (0.10+) and Steamodded only.
- **The developer tools** (below): real-game runs compared with the
  simulator and the headless oracle step by step.

## The bot button

`python -m antezero_agent.bot --install CKPT` copies the mod into the game's
`Mods` folder and stores the command that starts the agent in
`%APPDATA%/Balatro/antezero.json`. The mod starts that process the first time
the bot takes over a run and talks to it over JSON-RPC on 127.0.0.1:12346
(`AZB_PORT`).

"Bot plays this run", in the options menu (Esc) of a run, hands the run to
the bot from where it is, a new run or a continued one; turned off, it hands
it back, and it is off at the start of every run. A challenge run is the
player's alone. The run stays an ordinary run on the player's profile, with
the cards it has unlocked and discovered: before every decision the agent
rebuilds its simulator from the game's state dump (`az_state`, `sim::dump`),
so whatever the game holds is what the bot plays. It checks that copy against
the dump as `canon` reads it, and against what the simulator expected its
last action to do (but for what the game draws from its pools, which a
profile's locked cards change), and stops on a difference; an action decided
before the player opened a menu is refused, and decided again. The bot plays
on a fixed 1/60 s clock at game speed 1 (the bot's speed setting is the
number of such updates per 1/60 s) and skips the tutorial, which is saved as
the player left it. At a win it waits on the win screen: Endless Mode lets it
play on, New Run and Main Menu end the run as usual.

With "Bot runs count" off (the default) the run stops counting when the bot
takes over, for the rest of it, even after it is handed back: the saved run
stays as it was then (Continue returns there), achievements are not awarded,
what the game saves meanwhile is the player's profile as it was then (with any
settings changed in Options and the choices made for the next run), and the
profile, unlocks and discoveries are put back when the run is left. With it on,
the run counts like one the player played: wins, stats, stake progress,
unlocks, discoveries and achievements are kept and saved. Search, the speed
and that switch are the mod's settings (Mods > AnteZero > Config; 1x, search
on, not counting by default). While the bot plays, input is limited to Esc,
Run Info and Options, whose menus pause it; a strip at the top shows its last
action, how sure the policy was, and the critic's win estimate.

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
