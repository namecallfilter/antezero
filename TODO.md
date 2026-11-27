# Gold Win Rate To-Do

Untried ideas for Kyle's trainer (`run95-gcp`), in the order to run them. Checked against all 116 W&B runs and our older runs.

`az_run95` is at 26.4% sampled wins at 759M steps. At 500M it had 22.1%, against 12.2% for `0087` and 11.8% for `0093`. On the same host d384x8 runs about 1.65x slower than d256x6.

Sizes are unknown until tested. Most of these are worth a point or two at best, so compare on paired seeds.

## 1. Measure what we have (no training)

- [ ] **Greedy eval on fixed seeds, all 15 decks, 8,000+ games.** The W&B chart is sampled play. On `0095` greedy was 29.0% against 22.9% sampled (871 and 686 of 3,000). 4,096 games only resolves about 1 to 2 points.
- [ ] **Latest checkpoint against a short-window average, same seeds.** On ours averaging gave +3.7 on a plateau and lost 0.7 and 1.5 on climbing runs. Needs checkpoints kept more often than every 500 updates (`keep_ckpt_every`), or a running average of the weights.

## 2. Add logging (no behaviour change)

- [ ] **Rollout time and learning time, separately.** Only total steps/s since process start is logged. Every speed choice below depends on the split.
- [ ] **Fix `start_share`.** It reads 0.5 when the real pool share is 0.2, because it compares two 2,000-game windows (`train.py`).
- [ ] **KL after the last optimizer step, and KL of the first minibatch.** The learning-rate controller uses the mean of KLs measured before each step, which under-reads the real update. Needed before touching batch size or the KL target.
- [ ] **Value and policy gradient size before clipping.** `vf_coef` 0.5 and `max_grad_norm` 0.5 were never varied. This decides whether the value head needs changing.
- [ ] **Steps played after the ante-8 win, and timeouts.** Sizes the endless-cap test below.
- [ ] **Deck-fixing metrics at each round start: rank, suit and card ("idol") entropy, and unreachable cards.** Unreachable is the cards still undrawn at the last hand if every hand and discard used 5 cards (Red at Gold: 52 - 8 - 5 x 6 = 14). Neither agent deck-fixes: on 100 fixed seeds `az_run95` (update 10k) and `0098` gain about 7 cards by ante 8, and unreachable rises by about 5. Kyle's 47 Gold recordings lose 0.74 bits of card entropy and about 9.4 unreachable cards, and his CEF9MWJQ run won with 23 Jacks of Hearts (all three entropies 0). The measuring code is on the `deckfix-metrics` branch (`python -m antezero_agent.deckfix`). Its `deck_stats` needs porting to master's `StartPool`; log the per-ante means and the change from the deck's start.
- [ ] **Action counts per game.** A six-game probe of `0095` had joker reorders on 26% of steps. Unconfirmed in training.

## 3. Ten-minute speed benchmark (one card, after compile warm-up)

- [ ] **`len_buckets` 2 against 4 and 8.** Padding fell from 61% to 24% to 16% in a probe. Same math, one flag, never set above 3.
- [ ] **d256x6 against d384x8, each alone on the card.** The logs point to about 1.65x. It was never measured cleanly.
- [ ] **Rented 4090 against the L4.** The L4 does about 4,100 steps/s at $0.71/h on demand. A 4090 is an estimated 3x, unmeasured, and costs cash where the L4 uses credit.

## 4. One-flag tests (from the latest checkpoint, one at a time)

- [ ] **Stop at the ante-8 win: `endless_ante` 8 against 16.** After the win every blind still pays +1, and the win itself pays about 5.8. Changes the objective and saves steps.

## 5. Hyperparameters (forks of the latest checkpoint, paired seeds, about 1,000 to 1,500 updates each)

Judge each arm by greedy win rate on 8,000+ fixed-seed games across all decks, and by win-rate slope per 100M steps, not by the final number alone. Log the KL after the last optimizer step in every arm.

- [ ] **KL controller off.** Since about update 2,000 the exp schedule sits at its 5e-5 floor, so the controller is the only thing still lowering the learning rate: 2e-5 is a multiplier of 0.4. In RLBalatro's `0071` (same recipe at d256) it took the learning rate from 3.2e-5 to 1.2e-5 over 45k updates while measured KL sat on target, and entropy rose from 0.86 to 1.22. With 1 epoch, late KL is mostly minibatch noise, so the controller cuts on noise. Arms: A as now; B controller off at a fixed 2e-5 (separates the ongoing cut from the level already reached); C controller off at 5e-5. If C beats A, the controller costs slope. The controller comes from legged-robot PPO (Rudin et al. 2021, https://arxiv.org/abs/2109.11978), with targets 0.008 to 0.01, a 0.5x to 2x dead band and runs of hours; there is no study of it over billion-step runs.
- [ ] **KL target 0.0025 against 0.0012, measured after the last step.** 0.0012 is in every Gold run and its origin isn't recorded: the controller started at 0.0035 with 2,048 envs (64 steps per update) in `0050` and moved to 0.0012 with 512 envs in `0054`. The logged KL averages 16 readings taken before each step, so the real end-of-update KL is probably 2 to 3x the target. Also try the wider 0.5x to 2x dead band. Run after the KL logging.
- [ ] **Anneal the entropy bonus from 0.01 to 0.001-0.003 late.** It is 0.01 in all 110 PPO runs, and entropy rose through `0071`; check `az_run95`'s curve. OpenAI Five annealed entropy late along with a learning-rate cut and a longer horizon (Berner et al. 2019, https://arxiv.org/abs/1912.06680). Not to zero: on ours the voucher action died under low entropy. Needs a small schedule in `ppo.py`.
- [ ] **GAE lambda 0.99 against 0.95.** 0.95 in every run. Games are about 270 steps long and lambda 0.95 looks about 20 steps ahead.
- [ ] **Rollout shape at the same 65,536 steps: 512 envs x 128 steps (now) against 256 x 256 and 1,024 x 64.** Each step is already played by the policy of its own update, so carrying a game over an update isn't off-policy; the cost is the cut at the end of each segment, where advantages lean on the value estimate. With lambda 0.95 and gamma 1 that is roughly the last 20 steps, about 15% of samples at 128 steps and 8% at 256. Whole games per rollout (collect until N games end) is the extreme: no cut, but envs wait on the longest game (`max_steps` 3,000 against about 270 on average) and every batch starts in sync at ante 1. Keep minibatch 4,096; fewer envs also means less rollout parallelism, so log steps/s too.
- [ ] **More envs late, same minibatch 4,096.** The useful batch size (gradient noise scale) grows as play improves, Dota among the examples (McCandlish et al. 2018, https://arxiv.org/abs/1812.06162); OpenAI Five's batch grew from 61k to 197k samples (https://arxiv.org/abs/1912.06680). Scale the learning rate with the square root of the batch (Hilton et al. 2021, https://arxiv.org/abs/2110.00641), whose weight-average proximal policy also gave small, consistent gains. 1,024 or 2,048 envs x 128 steps.
- [ ] **Soft shrink-and-perturb, or L2 toward the initial weights.** In PPO only these "regenerative" fixes consistently stopped plasticity loss; ReDo, last-layer resets and plasticity injection were weak or worse than nothing, and shrink-and-perturb with LayerNorm was best (Juliani and Ash 2024, https://arxiv.org/abs/2405.19153; Ash and Adams 2019, https://arxiv.org/abs/1910.08475). Their runs are grid worlds and ProcGen, not billion-step. Shrink factor and noise size are the knobs.
- [ ] **Muon weight decay that doesn't shrink with the learning rate.** Decoupled decay is lr x wd, so the controller's cuts weaken it too, weight norms grow, and with normalised layers that lowers the effective learning rate further (Lyle et al. 2024, https://arxiv.org/abs/2407.01800). LayerNorm and weight decay together keep a network trainable where either alone can fail (Lyle et al. 2024, https://arxiv.org/abs/2402.18762). Log weight norms first.

## 6. Small code changes

- [ ] **Pick the start-pool deck first, then a point in it.** 581 of the 1,341 pool points are Blue, and Blue is already the best deck. About 10 lines in `curriculum.py`.
- [ ] **Deck weights for fresh games.** Black wins 7%, Plasma 14%, Blue 45%. On ours a larger Plasma share gave +5.4 on Plasma with other decks level.
- [ ] **Train the value head more than the policy.** `epochs` is 1, so the value head sees each sample once. Phasic Policy Gradient (Cobbe et al. 2020, https://arxiv.org/abs/2009.04416) found the value function tolerates much more sample reuse than the policy: it adds a phase of extra value epochs, with a KL term holding the policy still. Cheapest version: after the PPO step, one or two more epochs on the value loss alone. Compare value loss on held-out games, then wins. Run after the gradient-size logging.
- [ ] **Keep feature changes small (Proximal Feature Optimization).** In PPO, collapsing feature rank goes with the trust region failing; an extra loss on how much the features change per update reduced the collapse on Atari and MuJoCo (Moalla et al. 2024, https://arxiv.org/abs/2405.00662). Log feature rank first.
- [ ] **Rescale value targets as returns grow (PopArt).** With gamma 1 the returns' scale drifts as the agent improves, and the value loss is plain squared error on unscaled targets. PopArt rescales the targets and adjusts the last layer so outputs stay the same, so it fits the live run (van Hasselt et al. 2016, https://arxiv.org/abs/1602.07714). A categorical value head (HL-Gauss) scaled better and coped with moving targets (Farebrother et al. 2024, https://arxiv.org/abs/2403.03950); that one needs a fresh head.
- [ ] **Bootstrap games cut by the 3,000-step limit.** `train.py` treats `term | trunc` as game over, so a timed-out state learns a value of 0 instead of its own estimate. Rare, but wrong.
- [ ] **Constant win reward.** The bonus is log10 of the final hand's chips, so a bigger-scoring win pays more than a safe one.
- [ ] **Minibatch 8,192 or 16,384 late in training.** Never above 4,096 in any run. Kyle's 2,048-environment runs were all before Gold.
- [ ] **Less blind reward once wins pass 30%.** Blinds pay about four times what the win does over a full game. Needs `reward_blind` exposed in the config.
- [ ] **Pool: add Baron states, sweep the share.** Baron has 0 points in the pool. Only shares 0.1 and 0.2 were ever run.

- [ ] **Deck fixing as a second objective: does paying for it win more?** Two forms, each against the same checkpoint on paired seeds. (a) Shaping: pay for the drop in card entropy or unreachable cards between round starts, as a potential difference (gamma x phi(next) - phi(now)) so the best policy stays the same and only learning speeds up. Measure from the deck's own start, since Abandoned and Checkered begin lower. (b) An extra prediction head for next round's entropy, with no reward change. Check that wins go up, not just entropy: a thin deck of weak cards scores 0 bits too. Needs the logging above first.

## 7. Next fresh run

- [ ] **Value head: normalised targets or a categorical head.** Plain squared error on returns around 20, sharing the trunk. Choose after the gradient logging.
- [ ] **Muon on the transformer's hidden weights only.** Input layers and the card-picker heads go through Muon too. Gain unknown.
- [ ] **Squash the score-over-blind input.** It is fed raw and can reach the hundreds.
- [ ] **Show whether the joker reorder was used this phase.** It changes which actions are legal but is not in the observation.
- [ ] **Extra prediction heads: survive this ante, score margin.** Ours only ran short pilots, never a paired test.
- [ ] **Pace feature: hands left times typical score, over chips needed.** Proposed on both sides, never built.

## 8. Later (bigger jobs)

- [ ] **Port our round and endgame search to his action space.** His 8-simulation search got 33.0% against 29.0% greedy on 600 games.
- [ ] **Teach a new-size model from the best one, then PPO.** Ours matched its teacher only when it copied the teacher's layers. Untried across sizes.
- [ ] **Longer search-target training.** `0098_gumbel_D` ran 115 updates, too short to read.
- [ ] **Full current recipe at d256x6.** Never run from step 0. Only worth it if the benchmark shows d256 is much faster.
