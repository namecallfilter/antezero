# antezero simulator

A bit-exact Balatro (v1.0.1o) simulator in Rust, verified against the real
game.

| path | what |
|---|---|
| `sim/` | the `sim` crate: game logic, frame-exact event timing, deterministic RNG streams, `clone()` snapshots, learning interface (`sim::obs`) |
| `sim/oracle/` | differential testing against the real game's Lua run headless in the game's own `lua51.dll` (`difftest`, `replay`, `ocli`), golden trajectories |
| `sim/py/` | PyO3 bindings: `antezero_sim.Game`, `antezero_sim.VecEnv` |

```
cargo test --release                                     # simulator tests + golden trajectories
cargo test --release -- --include-ignored                # also the oracle tests (need the game, see below)
cargo run --release -p sim --example bench               # simulator throughput
cargo run --release -p oracle --bin difftest -- 20 1 400 b_red 1 0.15
```

**Golden trajectories** (`sim/oracle/tests/golden.rs`, no game needed): 200
seeded games over every deck and stake are fingerprinted step by step, the
simulator state and timeline (`golden_sim.txt`) separately from the learning
observation (`golden_obs.txt`), so a refactor that changes behaviour fails in
seconds. After an intended change re-record with `GOLDEN_BLESS=obs` (the
simulator part must still match) or `GOLDEN_BLESS=all`.

**Oracle** (`sim/oracle`, Windows): runs the game's scripts headlessly in the
game's own LuaJIT. It needs `lua51.dll` from the Steam install (or
`BALATRO_LUA51`) and the extracted game scripts in `reference/balatro` (or
`BALATRO_SRC`); it never starts the game or touches its saves. `difftest
[games] [first_seed] [max_steps] [back] [stake] [inject_rate]` plays random
games in both engines and compares the canonical state (all RNG streams
included) and the timeline of RNG calls, draws and removals after every
step; frame offsets are counted, and fail the run with `ORACLE_FRAMES=1`. A
diverging game's history is written to `.scratch/` and can be replayed with
`replay <history>`. Both exit with status 1 on a divergence.

The simulator's tables (`sim/src/data_gen.rs`) are generated from the game's
prototypes (see `sim/oracle/lua/gen_data.lua` for the command).

Supported: all 15 regular decks, stakes 1-8, endless play past ante 8. Not
supported: challenges.
