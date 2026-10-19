# antezero agent

Transformer policy/value agent for Balatro, trained with PPO on the Rust
simulator (`sim/`, exposed to Python as `antezero_sim`).

```
agent/
  src/antezero_agent/
    config.py      presets (smoke / cpu / base / large), all hyperparameters, validation
    model.py       entity transformer + factored action heads
    runtime.py     device, environment factory, checkpoints (shared by train / eval / play)
    train.py       PPO over antezero_sim.VecEnv (parallel Rust envs)
    eval.py        win rate (95% Wilson intervals) + endless survival, held-out seeds
    search.py      determinized lookahead (test-time policy improvement)
    bench.py       env-only and end-to-end PPO throughput on a device
    play_real.py   play the real game through the bridge, sim in lockstep
    stats.py       Wilson intervals, sample sizes
  bridge/          real-game control (BalatroBot + companion mod), see bridge/README.md
  tests/           legality, log-probs, shaping, GAE, truncation, overfit, config, training / resume
```

## Commands

```
uv run python -m antezero_agent.bench --preset base --device cuda
uv run python -m antezero_agent.train --preset base --name run1 --device cuda
uv run python -m antezero_agent.train --preset cpu --name dbg --set env.num_envs=128 ppo.lr=2e-4
uv run python -m antezero_agent.train --resume agent/runs/run1/last.pt
uv run python -m antezero_agent.eval agent/runs/run1/last.pt --episodes 10000
uv run python -m antezero_agent.eval CKPT --episodes 200 --search          # search ablation, same seeds
uv run python -m antezero_agent.play_real agent/runs/run1/last.pt --seeds 20 --search
uv run --with pytest pytest agent/tests -q
```

`play_real` needs the real-game stack (`uv sync --extra game` when the agent
is installed on its own; the uv workspace has it).

Logs go to `agent/runs/<name>/log.csv` (fixed columns: win rate with its 95%
interval, ante reached, losses, KL, optimizer steps, fps, logp drift,
illegal-action count, current stakes) and `eval.jsonl`. A fresh run refuses
an existing run directory (pick another `--name`, or `--resume`). Resuming
continues the model, optimizer, schedule, Torch RNG and stake curriculum;
the environments restart on fresh seeds, and the batch shape
(`env.num_envs`, `ppo.rollout_len`) cannot change. Checkpoints record the
observation schema and only load into a matching simulator. Configuration
errors (unknown fields, malformed `--set` values, a batch that does not
split into minibatches, nonzero dropout) are rejected before anything is
written.

## Design

**Observation** (`sim/src/obs/`): every visible object is a token: hand
cards, jokers (with the card a joker currently targets: The Idol, Mail-In
Rebate, Ancient Joker, Castle; To Do List's hand; the deck counts Steel
Joker, Stone Joker, Cloud 9 and Driver's License show), consumables
(Temperance's payout), shop cards / vouchers / boosters, pack cards, tags,
the three blinds of the ante (with targets, rewards, skip tags and the hand
an Orbital Tag levels, The Ox targets or The Mouth allows), the twelve
poker-hand levels. Globals (ante, chips vs. target, money, hands, discards,
slots, vouchers, stake, deck, The Fool's next card, the ante's voucher once a
shop has shown it, the endless flag, ...) form a CLS token and include
histograms of the unseen cards and of the discard pile (optionally tokenized
card by card with `env.deck_tokens=true`).

Only player-visible information is encoded, and tests check it
(`sim/tests/obs.rs`): the deck is a multiset (draw order hidden); face-down
cards and jokers hide their identity, and, as in the game's deck view, the
"unseen" pool includes face-down hand cards, so swapping a face-down card
with a deck card gives the identical observation; no RNG state is exposed;
`Game.resample_hidden` (search, snapshot restarts) redraws exactly the hidden
part (future randomness, deck order, face-down hand cards, the order of
face-down jokers, an unseen voucher) and leaves the observation unchanged.
Card tokens of the deck and discard pile are ordered by their whole content,
so no hidden order or id leaks through ties.

**Action** (factored, exact probabilities):
1. an *op head* scores the legal ops: select / skip / reroll boss, play,
   discard, use / sell / buy / buy-and-use / redeem / open / pick (each on its
   target token), reroll, end shop, cash out, skip pack, sort, move joker /
   hand card (in rounds and in packs opened with the hand drawn);
2. a *card pointer* picks hand cards in increasing index order (so every
   card set has a single encoding), with count bounds, forced cards (Cerulean
   Bell) and reachability of the minimum enforced by masking;
3. a *destination pointer* for reordering (joker order matters: Blueprint,
   Brainstorm, x-mult ordering).
The log-probability of a full action is the sum of the three parts, computed
by the same code when sampling and when training (teacher-forced): in float32
the two agree to 1e-4 (`tests/test_agent.py`). Under bf16 the rounding
depends on the batch, so after each rollout the behaviour log-probs are
recomputed with the training forward pass on minibatch-sized batches
(`ppo.recompute_logp`); the update's shuffled minibatches still differ, so
the first ratio is close to, not exactly, 1 (`logp_drift` in the log is the
largest correction, first-epoch `approx_kl` the residual).

**Scored play candidates** (`env.play_candidates`, default 8): in rounds the
env also offers the best plays as ops. Plays are ranked by a cheap
joker-free estimate, the best 64 are simulated, and plays whose score
depended on chance (Lucky cards, Misprint, Bloodstone, ...) are simulated on
6 chance scenarios that are shared by all plays and seeded from visible
state only (so they change from decision to decision, are never the real
future, and are identical across determinizations). Each candidate op
carries its cards (pooled hand-token embeddings) and features: log mean
score, mean / chips needed, estimated probability of clearing the blind, log
min / max over scenarios, cards / 5. Nothing is previewed while any hand
card or joker is face down. The free-form pointer keeps every other play
(setting up hands, levelling, Hiker, ...) available. Candidates never crowd
out the other ops (`MAX_CANDIDATES` = 48). Cost: ~0.2 ms per round decision
per core, ~0.8 ms when every play depends on chance.

**Network**: shared pre-LN transformer encoder (`base`: d=256, 6 layers, 8
heads, ~6M parameters; `large`: d=512, 10 layers, ~36M; both train on a
single device). Value head on CLS.

**Objective and reward** (`VecEnv`, mirrored in `search.py`). The true
reward is `r_win` for winning the run (the final boss beaten:
`Game.run_won`; the game's own `won` flag is also set when the final boss is
*lost*), optionally `r_endless` per blind cleared after the win and
`-r_loss` on a loss. Progress is rewarded only through potential-based
shaping `F = gamma * Phi(s') - Phi(s)` with `Phi = w_blind * blinds cleared
+ w_score * round progress` and `Phi(terminal) = 0`. This leaves the optimal
policy unchanged: over an episode the shaping telescopes, `sum_t gamma^t F_t
= gamma^T Phi(s_T) - Phi(s_0)`, which is `0` for a whole game (`Phi(start)
= 0`, `Phi(terminal) = 0`), so its discounted shaped return is the discounted
true return (tested for `gamma` 1 and 0.99); a truncated episode keeps
`gamma^T Phi(s_T)`, matching the bootstrap from `V(s_T)`. It acts like a
value prior that rewards clearing blinds early in training. Defaults: `r_win=1`, `w_blind=0.04` (~ r_win / 24
blinds), `w_score=0.02`.

**Episodes**: by default a run ends at the win or the loss (terminal). The
step limit and, in endless mode, the ante cap are *truncations*: the env
writes the last state's observation to `final_*` and the trainer adds
`gamma * V(final)` instead of treating it as a loss.

**Training**: PPO + GAE (gamma 0.999, lambda 0.95) on many parallel Rust
envs, AdamW with warmup + cosine decay, entropy annealing, KL early stop,
bf16 autocast, float32 observation storage, plain value MSE (value clipping
optional). Optional:
- *snapshot restarts* (`env.snapshot_prob`, e.g. 0.25): episodes restart from
  ante-start states of recent episodes (ante >= `snapshot_min_ante`) with the
  hidden state resampled, so the hard late antes are trained far more often
  than a weak policy reaches them. These episodes are excluded from the
  win-rate statistics;
- *stake curriculum* (`env.curriculum`): the next stake is added once the 95%
  Wilson lower bound of the win rate over the latest
  `curriculum_min_episodes` fresh episodes on the top stake reaches
  `curriculum_target`.

Suggested schedule: train the win objective first (`env.endless=false`), then
continue with `env.endless=true env.r_endless=0.04 env.snapshot_prob=0.25
env.max_ante=16` for endless depth; the endless flag is part of the
observation, the win reward stays in the objective, and the win rate is
still measured (`run_won`).

**Test-time search** (`search.py`): determinized one-step lookahead - for the
policy's top actions, average `r + gamma V(s')` over futures with resampled
hidden state (the same sampled worlds for every action), using the training
reward and shaping so values are on the scale they were trained on;
deterministic given its seed. Its benefit must be measured (`eval --search`
plays the same games as the plain eval).

## Scope

Supported and oracle-tested: all 15 regular decks (`backs()`), stakes 1-8,
endless play past ante 8. Not supported: the Challenge Deck / challenge
rules (`b_challenge` is rejected).

## Validation gates (before and during a long run)

1. Simulator fidelity: oracle difftests (`sim/oracle`, see the root README)
   pass for the decks / stakes being trained.
2. Legality: `invalid_actions` stays 0.
3. Exact log-probs: `tests/test_agent.py`; `logp_drift` small; first-epoch
   `approx_kl` ~ 0.
4. Learning capacity: the tiny-overfit test (memorizes op, card set and
   destination choices).
5. Throughput on the training machine: `antezero_agent.bench`.
6. Held-out evaluation per deck / stake with intervals: `eval --episodes
   10000` gives about +-0.4% at 95% win rate (`episodes_for_1pct` is
   printed); claim 95% only when the lower bound is >= 0.95. Runs cut off by
   the step limit count as not won (`unresolved` reports them).
7. Search ablation on the same seeds before relying on search.
8. Real game: `play_real` in lockstep (any sim divergence is reported; the
   win rate is over the games that finished in agreement).

## Compute

The model is compute-bound. On this laptop (Ryzen AI 9 CPU, fp32 ~170
GFLOP/s, no CUDA GPU) the `base` model runs ~260 states/s forward and ~120/s
forward+backward, i.e. far too slow to train to a high win rate. On a single
modern CUDA GPU the same code is roughly two to three orders of magnitude
faster, and the Rust envs supply 10^5 env-steps/s on 20 cores (round
decisions with candidates: 0.2-0.8 ms per core). Run `antezero_agent.bench`
on the training machine for real numbers. A 95%+ win rate likely needs
billions of env steps (an estimate, not a measurement); the first runs
should measure the learning curve before committing to that budget.
