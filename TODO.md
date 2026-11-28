# Gold Win Rate To-Do

> Baseline: `az_run95` u11,900 (**32.0% greedy / 26.8% sampled** on 8,010 fixed games); 500-update controls **31.9–32.4% greedy** (mean **32.2%**), **25.9–27.9% sampled** (mean **26.9%**); fixed-LR 2e-5 control **32.0% / 27.5%**.

## 1. Measure what we have

- [x] **Greedy vs sampled eval (all 15 decks, 8,010 fixed games)** — **Done:** 32.0% greedy vs 26.8% sampled (**+5.2 pts** ±0.5) at u11,900 (`python -m antezero_agent.rl_eval CKPT`, 5 min on 4060 Ti).
  - Greedy by deck: Blue 54%, Yellow 48%, Zodiac 40%, Ghost 38%, Checkered 35%, Magic 32%, Painted 32%, Red 31%, Green 31%, Abandoned 31%, Anaglyph 30%, Nebula 26%, Erratic 26%, Plasma 16%, Black 10%.
- [x] **Checkpoint averaging vs latest (same seeds)** — **Gain (+1.1 to +1.4 greedy, repeated):** avg of last 5 (or 9) checkpoints 25 updates apart = **33.8% vs 32.4%** (+1.4); repeated on 2nd control with 3 checkpoints 100 updates apart = **33.0% vs 31.9%** (+1.1). Does not stack with `1e-5` LR (35.4% vs 35.3%).

## 2. Add logging

- [x] **Separate rollout vs learning time** — **Done (`t_rollout`, `t_learn`):** 4090 = 1.6s rollout / 4.0s learn; 4060 Ti (no compile) = 9s / 18s (sim is 0.9s, rest is policy forward).
- [x] **Fix `start_share`** — **Done:** reads 0.196 at 0.2 pool share (was 0.5).
- [x] **End-of-update and first-minibatch KL** — **Done (`kl_first`, `kl_last`, `kl_end`):** `kl_end` is 0.0029 (~2.4× the 0.0012 controller reading); `kl_first` is 0.00001.
- [x] **Value vs policy gradient norms before clipping** — **Done (`grad_norm`, `grad_pg`, `grad_v`, `grad_cos`):** pre-clip norm ~8.6 (clipped ~17× at 0.5); value grad is 3–4× policy grad (4.4–5.2 vs 1.3–1.8), cosine -0.03.
- [x] **Post-ante-8 steps and timeouts** — **Done (`post_win_share`, `timeouts`):** 14% of steps are post-win; 1 timeout in 32.8M steps; max ante 16.
- [x] **Action counts per game** — **Done (`act_*`):** joker reorders are 24.7% of training steps (24.8% greedy); plays are 15.8%.
- [ ] **Deck-fixing metrics at round start (rank/suit/card entropy, unreachable cards)** — Waiting on `deckfix-metrics` branch on GitHub (port `deck_stats` to master's `StartPool`).

## 3. Speed benchmarks

- [x] **`len_buckets` 2 vs 4 vs 8** — **Faster at 4 on 4090 (+19%):** 5.08s vs 6.05s/update (8 buckets = 5.60s, +8% as rollout slows); learning padding 41% → 22% → 12%; peak VRAM 18.9 → 14.7 → 13.3 GB. Uncompiled 4060 Ti is fastest at 2 (26.6s vs 37.0s / 56.1s).
- [x] **`d256x6` vs `d384x8` speed** — **Done:** `d256x6` is **1.83× faster** on 4090 (19,900 vs 10,800 steps/s; **1.55×** with 4 buckets) and **1.25× faster** on uncompiled 4060 Ti (21.4s vs 26.6s/update).
- [x] **Rented 4090 / 3090 vs GCP L4** — **Faster & cheaper:** 4090 ($0.48/h) = 11,500 steps/s (**2.8× L4**, ~4× steps/$); 3090 ($0.19/h) = 6,040 steps/s (**1.5× L4**, best steps/$); L4 ($0.71/h) ≈ 4,100 steps/s.

## 4. One-flag tests

- [x] **Stop at ante-8 win (`endless_ante` 8 vs 16)** — **Slightly worse (-1.3 greedy, -2.4 sampled):** 30.9% greedy / 24.5% sampled at 500 updates; plays 19% more games/step (255 vs 286 steps/game), though value head may need >500 updates to adapt.

## 5. Hyperparameters

- [x] **KL controller off (fixed 2e-5 and 5e-5)** — **No change at 2e-5; worse at 5e-5:** fixed 2e-5 = 32.0% greedy / 27.5% sampled (-0.2 / +0.5); fixed 5e-5 = 28.1% / 23.4% (**-4.1 / -3.6**).
- [x] **KL target & learning rate sweep (0.0025, 1e-5, 5e-6)** — **Gain at `1e-5` (+1.5 to +1.9 greedy, repeated & holds at 1,000 updates):**
  - KL target 0.0025 (LR rose to 3.7e-5): **Worse** — 29.7% greedy / 25.1% sampled (-2.5 / -1.8).
  - Fixed `1e-5` (two seeds): **34.1% and 33.7% greedy** (+1.9, +1.5), **28.1% and 28.3% sampled** (+1.2, +1.4). At 1,000 updates: **35.3% vs 33.5% control** (+1.8), sampled 28.8% vs 28.1%.
  - Fixed `5e-6`: **33.4% greedy** (+1.2), 27.6% sampled (+0.7) — levels off around `1e-5`.
- [ ] **Entropy controller (`az_0099_d256_l6_ent075`)** — **Stopped at u4,246 of 20,000, no verdict** (from scratch at d256×6 on `run95-entctl`, targeting 0.75 entropy from u1,000): the controller held entropy at 0.74–0.76 by cutting the coefficient 0.01 → 0.004. Training win 13.3% vs 17.1% for d384×8 `az_run95` at the same update, but the gap was already there at u1,000 (2.9% vs 4.0%) and the model is smaller; still needs a matched d256×6 fixed-0.01 control.
- [x] **Anneal entropy bonus (0.01 → 0.003)** — **No greedy gain (+1.3 sampled):** 31.7% greedy (-0.5) / 28.2% sampled (+1.3); entropy 1.21 → 1.10, vouchers intact. Paired with fixed 5e-5: **Worse** — 25.2% greedy / 22.0% sampled (-7.0 / -5.0).
- [x] **GAE lambda 0.99 vs 0.95** — **Worse (-0.9 greedy, -2.7 sampled):** clean fixed-2e-5 rerun = 31.3% greedy / 24.3% sampled. Keep 0.95.
- [x] **Rollout shape (256×256 vs 512×128 vs 1,024×64, fixed 2e-5)** — **Gain at `1,024×64` (+1.2 to +1.7 greedy, repeated, same speed):**
  - `1,024×64` (two seeds): **33.4% and 33.9% greedy** (+1.2, +1.7), 27.1% and 27.8% sampled; 12,500 steps/s on 4090. Does not stack with `1e-5` (combo = 33.7% greedy / 28.2% sampled).
  - `256×256`: **No clear change, 14% slower** — 32.8% greedy (+0.6) / 27.9% sampled (+1.0), 10,700 vs 12,400 steps/s.
- [x] **More envs late with sqrt-scaled LR (1,024×128, fixed 2.8e-5)** — **Worse (-1.8 greedy):** 30.4% greedy / 25.9% sampled (`kl_end` doubled to 0.0062).
- [ ] **More envs late at unchanged/lower LR (1,024×128 or 2,048×128)** — Untested.
- [x] **Soft shrink-and-perturb (0.1% every 10 updates, fixed 2e-5)** — **Worse (-2.5 greedy):** 29.7% greedy / 24.6% sampled.
- [ ] **L2 toward initial weights** — Blocked (no checkpoint holds the initial weights).
- [x] **Stronger Muon weight decay (fixed 5e-5 vs 2e-5)** — **No gain (-1.0 greedy):** 31.2% greedy / 26.3% sampled (`weights/rms_muon` was already falling 0.0355 → 0.0328, not growing).

## 6. Small code changes

- [x] **Pick start-pool deck first, then point** — **No change (-0.3 greedy):** 31.9% greedy / 27.3% sampled (per-deck shifts within noise).
- [x] **Deck weights for fresh games (Black & Plasma 3×)** — **Likely gain (+1.2 greedy, 1 run):** **33.4% greedy** / 27.5% sampled (Black 10.6% → 14.0%, Plasma 16.9% → 20.0%, other decks within noise). Did not stack with `1e-5` in 500 updates (33.9% greedy); needs repeat / 1,000-update check.
- [x] **Extra value-only epoch after PPO step** — **Worse (-3.8 greedy, 2× learn time):** 28.4% greedy / 22.9% sampled (shared trunk pushed `kl_end` to 0.0081; needs PPG's KL anchor).
- [x] **Proximal Feature Optimization (feature-change loss)** — **No gain (-0.8 greedy):** feature rank shows no collapse (364/384 dims hold 99% SVs); clean fixed-2e-5 retry at coef 0.03 = 31.4% greedy / 26.4% sampled.
- [x] **PopArt value target rescaling** — **No greedy change (+1.0 sampled):** clean fixed-2e-5 rerun = 31.9% greedy / 28.0% sampled. Rebalanced value/policy grads (0.23 vs 2.0) harmlessly.
- [x] **Bootstrap games cut by 3,000-step limit** — **Done (in `todo-tests`, negligible impact):** only 1 timeout in 32.8M control steps (caused by a mask bug: 1,607 illegal actions in 1 game).
- [x] **Constant win reward (5.8 vs log10 chips)** — **No change (+0.4 greedy):** 32.6% greedy / 26.9% sampled.
- [x] **Minibatch 8,192 and 16,384** — **No gain, same speed:** 8,192 = 31.9% greedy / 27.3% sampled (-0.3); 16,384 = 31.2% / 25.0% (-1.1); learning still 3.6s/update.
- [x] **Less blind reward (0.5 vs 1.0)** — **No clear change (+0.8 greedy, borderline):** 33.0% greedy / 27.6% sampled (slightly confounded by KL controller drifting LR to 1.6e-5 late).
- [x] **Sweep start-pool share (0, 0.1, 0.2, 0.4 at fixed 2e-5)** — **Likely gain at 0–0.1 (~+1 greedy, 1 run each):** 0.1 = **33.4%** (+1.2), 0 = **33.0%** (+0.8, sampled 27.9%), 0.2 = 32.0–32.4%, 0.4 = **31.3%** (-0.9). Late training only; needs 1,000-update check.
- [ ] **Add Baron states to start pool** — Blocked (no human recordings with Baron).
- [ ] **Deck fixing as a second objective (potential-based shaping or auxiliary entropy head)** — Blocked on `deckfix-metrics` logging above.

## 7. Next fresh run

- [ ] **Value head: normalised targets or categorical (HL-Gauss) head.**
- [ ] **Muon on transformer hidden weights only** (exclude input layers and card-picker heads).
- [ ] **Squash the raw score-over-blind observation input.**
- [ ] **Add observation flag for whether joker reorder was used this phase.**
- [ ] **Extra prediction heads (survive this ante, score margin).**
- [ ] **Pace feature (`hands_left × typical_score / chips_needed`).**

## 8. Later (bigger jobs)

- [ ] **Port round and endgame search to `run95` action space.**
- [ ] **Distill / transfer from best model to a new size, then PPO.**
- [ ] **Longer search-target training** (`0098_gumbel_D` only ran 115 updates).
- [ ] **Full current recipe at `d256x6` from step 0.**
