# AnteZero

A Balatro agent: a bit-exact Rust simulator of the game (v1.0.1o), a
transformer policy trained on it with PPO, and a mod that lets it play your
real game.

It was trained on Red Deck, White Stake only, where it wins about 91% of
games without any search.

Not affiliated with LocalThunk or Playstack. You need your own copy of the
game. No trained checkpoint is included.

## Watch it play

Needs [Balatro](https://store.steampowered.com/app/2379780/Balatro/) on
Windows (Steam) with
[lovely](https://github.com/ethangreen-dev/lovely-injector) and
[Steamodded](https://github.com/Steamodded/smods), [uv](https://docs.astral.sh/uv/),
[Rust](https://www.rust-lang.org/tools/install), and a checkpoint. A CUDA GPU
is recommended.

```
uv sync
uv run python -m antezero_agent.bot --install path/to/checkpoint.pt
```

Then start Balatro, open New Run, switch on the bot toggle in the corner of
the Play button, and press Play. Esc, Run Info and Options pause the bot.
Bot runs never touch your saves, unlocks or stats. Search and the bot's speed
are under Mods > AnteZero > Config.

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

Games won by the best checkpoint, playing its own choice with no search,
over 1,024 games per cell (about ±3 points). Rows are decks, columns are
stakes. Only the bold cell (Red Deck, White Stake) was trained on; the agent
has never played any other deck or stake.

<table>
<tr><th rowspan="2">Deck</th><th colspan="8">Stake</th></tr>
<tr><th>White</th><th>Red</th><th>Green</th><th>Black</th><th>Blue</th><th>Purple</th><th>Orange</th><th>Gold</th></tr>
<tr><th>Red Deck</th><td align="right"><b>91%</b></td><td align="right">79%</td><td align="right">50%</td><td align="right">32%</td><td align="right">27%</td><td align="right">8%</td><td align="right">2%</td><td align="right">1%</td></tr>
<tr><th>Blue Deck</th><td align="right">96%</td><td align="right">89%</td><td align="right">66%</td><td align="right">44%</td><td align="right">39%</td><td align="right">16%</td><td align="right">5%</td><td align="right">2%</td></tr>
<tr><th>Yellow Deck</th><td align="right">92%</td><td align="right">82%</td><td align="right">57%</td><td align="right">44%</td><td align="right">31%</td><td align="right">10%</td><td align="right">3%</td><td align="right">1%</td></tr>
<tr><th>Green Deck</th><td align="right">92%</td><td align="right">88%</td><td align="right">59%</td><td align="right">39%</td><td align="right">29%</td><td align="right">7%</td><td align="right">2%</td><td align="right">1%</td></tr>
<tr><th>Black Deck</th><td align="right">63%</td><td align="right">41%</td><td align="right">15%</td><td align="right">11%</td><td align="right">7%</td><td align="right">2%</td><td align="right">1%</td><td align="right">0%</td></tr>
<tr><th>Magic Deck</th><td align="right">90%</td><td align="right">79%</td><td align="right">47%</td><td align="right">30%</td><td align="right">25%</td><td align="right">7%</td><td align="right">2%</td><td align="right">1%</td></tr>
<tr><th>Nebula Deck</th><td align="right">88%</td><td align="right">74%</td><td align="right">44%</td><td align="right">31%</td><td align="right">23%</td><td align="right">6%</td><td align="right">2%</td><td align="right">0%</td></tr>
<tr><th>Ghost Deck</th><td align="right">87%</td><td align="right">72%</td><td align="right">44%</td><td align="right">27%</td><td align="right">22%</td><td align="right">5%</td><td align="right">2%</td><td align="right">0%</td></tr>
<tr><th>Abandoned Deck</th><td align="right">86%</td><td align="right">71%</td><td align="right">40%</td><td align="right">26%</td><td align="right">21%</td><td align="right">7%</td><td align="right">2%</td><td align="right">0%</td></tr>
<tr><th>Checkered Deck</th><td align="right">88%</td><td align="right">76%</td><td align="right">41%</td><td align="right">27%</td><td align="right">22%</td><td align="right">6%</td><td align="right">2%</td><td align="right">0%</td></tr>
<tr><th>Zodiac Deck</th><td align="right">91%</td><td align="right">80%</td><td align="right">52%</td><td align="right">39%</td><td align="right">32%</td><td align="right">10%</td><td align="right">3%</td><td align="right">1%</td></tr>
<tr><th>Painted Deck</th><td align="right">77%</td><td align="right">60%</td><td align="right">24%</td><td align="right">14%</td><td align="right">12%</td><td align="right">2%</td><td align="right">0%</td><td align="right">0%</td></tr>
<tr><th>Anaglyph Deck</th><td align="right">86%</td><td align="right">61%</td><td align="right">35%</td><td align="right">25%</td><td align="right">15%</td><td align="right">5%</td><td align="right">1%</td><td align="right">0%</td></tr>
<tr><th>Plasma Deck</th><td align="right">72%</td><td align="right">55%</td><td align="right">21%</td><td align="right">13%</td><td align="right">10%</td><td align="right">2%</td><td align="right">0%</td><td align="right">0%</td></tr>
<tr><th>Erratic Deck</th><td align="right">85%</td><td align="right">74%</td><td align="right">41%</td><td align="right">29%</td><td align="right">21%</td><td align="right">4%</td><td align="right">1%</td><td align="right">0%</td></tr>
</table>

## More

- [sim/README.md](sim/README.md): the simulator and how it is checked against the real game
- [agent/README.md](agent/README.md): the agent's design, training and evaluation
- [bridge/README.md](bridge/README.md): the mod (`mod/`) and the real-game tools

## License

[MIT](LICENSE): use and modify it freely, keeping the copyright notice that
credits this repository.
