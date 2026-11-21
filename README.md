# AnteZero

An AI that plays Balatro. It includes a fast Rust simulator matching the real
game (v1.0.1o), a neural network trained to play all 15 decks and 8 stakes,
and a mod so you can watch it play your copy of the game.

Playing purely on instinct (no lookahead search), it wins about 94% of White
Stake runs and 12% of Gold Stake runs.

Not affiliated with LocalThunk or Playstack. You need your own copy of the
game. Download the trained model from the
[releases page](https://github.com/namecallfilter/antezero/releases/latest).

## Watch it play

Needs [Balatro](https://store.steampowered.com/app/2379780/Balatro/) on
Windows (Steam) with
[lovely](https://github.com/ethangreen-dev/lovely-injector) and
[Steamodded](https://github.com/Steamodded/smods), [uv](https://docs.astral.sh/uv/),
[Rust](https://www.rust-lang.org/tools/install), and the trained model from the
releases page. An NVIDIA (CUDA) GPU is recommended.

```
uv sync
uv run python -m antezero_agent.bot --install antezero-v0.2.0.pt
```

Then start Balatro and, in a run, switch on "Bot plays this run" in the
options menu (Esc). The bot plays from there until you switch it off; it is
off at the start of every run. Esc, Run Info, and Options pause the bot. At a
win it waits, and plays on if you choose Endless Mode. By default, the bot
only uses items unlocked on your profile, and from when it takes over the run
won't save progress (your profile and saved run stay as they were). If you
want bot runs to count for wins, stats, unlocks, and achievements, turn on
"Bot runs count" under Mods > AnteZero > Config (where you can also adjust bot
speed and lookahead search).

## Train and evaluate

```
uv run az-train --preset base --name run1 --device cuda
uv run az-eval agent/runs/run1/last.pt --episodes 1024 --device cuda
```

## Test

```
cargo test --release
uv run --with pytest pytest agent/tests bridge/tests -q
```

## Win rates

Win rates for the released model with lookahead search turned off (~1,024 runs
per cell, 122,880 total; margin of error ~±3%). Rows are decks, columns are
stakes.

<table>
<tr><th rowspan="2">Deck</th><th colspan="8">Stake</th></tr>
<tr><th>White</th><th>Red</th><th>Green</th><th>Black</th><th>Blue</th><th>Purple</th><th>Orange</th><th>Gold</th></tr>
<tr><th>Red Deck</th><td align="right">96%</td><td align="right">90%</td><td align="right">71%</td><td align="right">63%</td><td align="right">59%</td><td align="right">31%</td><td align="right">20%</td><td align="right">12%</td></tr>
<tr><th>Blue Deck</th><td align="right">99%</td><td align="right">96%</td><td align="right">85%</td><td align="right">77%</td><td align="right">74%</td><td align="right">51%</td><td align="right">37%</td><td align="right">24%</td></tr>
<tr><th>Yellow Deck</th><td align="right">97%</td><td align="right">96%</td><td align="right">81%</td><td align="right">78%</td><td align="right">73%</td><td align="right">49%</td><td align="right">37%</td><td align="right">21%</td></tr>
<tr><th>Green Deck</th><td align="right">96%</td><td align="right">94%</td><td align="right">81%</td><td align="right">79%</td><td align="right">63%</td><td align="right">38%</td><td align="right">24%</td><td align="right">13%</td></tr>
<tr><th>Black Deck</th><td align="right">83%</td><td align="right">67%</td><td align="right">43%</td><td align="right">41%</td><td align="right">31%</td><td align="right">13%</td><td align="right">8%</td><td align="right">3%</td></tr>
<tr><th>Magic Deck</th><td align="right">95%</td><td align="right">90%</td><td align="right">74%</td><td align="right">69%</td><td align="right">59%</td><td align="right">36%</td><td align="right">23%</td><td align="right">15%</td></tr>
<tr><th>Nebula Deck</th><td align="right">93%</td><td align="right">88%</td><td align="right">69%</td><td align="right">57%</td><td align="right">50%</td><td align="right">29%</td><td align="right">16%</td><td align="right">10%</td></tr>
<tr><th>Ghost Deck</th><td align="right">93%</td><td align="right">91%</td><td align="right">69%</td><td align="right">63%</td><td align="right">56%</td><td align="right">35%</td><td align="right">21%</td><td align="right">12%</td></tr>
<tr><th>Abandoned Deck</th><td align="right">94%</td><td align="right">86%</td><td align="right">66%</td><td align="right">61%</td><td align="right">52%</td><td align="right">28%</td><td align="right">16%</td><td align="right">11%</td></tr>
<tr><th>Checkered Deck</th><td align="right">96%</td><td align="right">90%</td><td align="right">71%</td><td align="right">61%</td><td align="right">58%</td><td align="right">30%</td><td align="right">21%</td><td align="right">12%</td></tr>
<tr><th>Zodiac Deck</th><td align="right">98%</td><td align="right">92%</td><td align="right">78%</td><td align="right">74%</td><td align="right">64%</td><td align="right">42%</td><td align="right">29%</td><td align="right">17%</td></tr>
<tr><th>Painted Deck</th><td align="right">92%</td><td align="right">87%</td><td align="right">58%</td><td align="right">53%</td><td align="right">46%</td><td align="right">19%</td><td align="right">16%</td><td align="right">5%</td></tr>
<tr><th>Anaglyph Deck</th><td align="right">94%</td><td align="right">85%</td><td align="right">62%</td><td align="right">58%</td><td align="right">49%</td><td align="right">25%</td><td align="right">16%</td><td align="right">8%</td></tr>
<tr><th>Plasma Deck</th><td align="right">91%</td><td align="right">87%</td><td align="right">64%</td><td align="right">52%</td><td align="right">46%</td><td align="right">23%</td><td align="right">12%</td><td align="right">6%</td></tr>
<tr><th>Erratic Deck</th><td align="right">95%</td><td align="right">88%</td><td align="right">67%</td><td align="right">58%</td><td align="right">52%</td><td align="right">29%</td><td align="right">17%</td><td align="right">10%</td></tr>
<tr><th>All decks</th><td align="right">94%</td><td align="right">88%</td><td align="right">69%</td><td align="right">63%</td><td align="right">55%</td><td align="right">32%</td><td align="right">21%</td><td align="right">12%</td></tr>
</table>

## How it works

- **Simulator:** A bit-exact Rust reimplementation of Balatro v1.0.1o, verified in lockstep against the real game and exposed to Python as a vectorized environment (`antezero_sim.VecEnv`).
- **Model:** An entity transformer over variable-length game-state tokens, with card effects (jokers, tarots, planets) encoded directly as token features and factored action heads.
- **Training:** PPO over parallel Rust environments with a decoupled critic and a stake curriculum spanning all 15 decks and 8 stakes.
- **Search:** Optional test-time determinized lookahead and within-round search (both disabled in the win-rate table above).
- **Real-game bridge:** A Steamodded mod and local bridge that rebuild the simulator state from the live game at every decision.

## More

- [sim/README.md](sim/README.md): the simulator and how it is checked against the real game
- [agent/README.md](agent/README.md): the agent's design, training and evaluation
- [bridge/README.md](bridge/README.md): the mod (`mod/`) and the real-game tools

## License

[MIT](LICENSE): use and modify it freely, keeping the copyright notice that
credits this repository.
