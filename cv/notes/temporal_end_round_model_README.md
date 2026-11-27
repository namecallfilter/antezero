# Temporal `END_ROUND` Detection Model — Architectural Suggestions & Data Generation Estimates

> **Status:** Exploratory design notes and estimates (not scheduled commitments).  
> **Goal:** Estimate the timestamp/frame of the implicit `ActionType.END_ROUND` event between the final `ActionType.PLAY_HAND` click and the `ActionType.CASH_OUT` click in `post_processing.py`, using only the central played-card screen region (`G.play`) so streamer overlays (facecams, Twitch chat) on the right side of the screen are ignored.

---

## 1. Problem Context & Balatro Engine Mechanics

In `balatro_source/functions/state_events.lua`, the end-of-round sequence after the final hand beats the blind proceeds as follows:

1. **`play_cards_from_highlighted()` (`state_events.lua:470-538`)**:
   - Moves 1–5 selected cards from `G.hand` to `G.play` (`draw_card(G.hand, G.play, ...)`).
   - Runs `G.FUNCS.evaluate_play()`, scoring cards and jokers, optionally shattering/dissolving cards (e.g. Glass Cards at `state_events.lua:978-996`), and easing `chip_total` into `G.GAME.chips`.
   - Calls `G.FUNCS.draw_from_play_to_discard()` (`state_events.lua:522`, `1088-1097`), which moves any non-shattered, non-destroyed cards from `G.play` to `G.discard` (positioned off-screen to the right in `common_events.lua:18-19`).
2. **`Game:update_hand_played()` (`game.lua:3187-3206`) & `Game:update_new_round()` (`game.lua:3248-3256`)**:
   - Transitions `G.STATE` to `G.STATES.NEW_ROUND`, which calls `end_round()`.
3. **`end_round()` (`state_events.lua:87-270`)**:
   - Schedules an `Event` with `trigger = 'after', delay = 0.2`.
   - At `state_events.lua:99-110`, iterates `G.jokers.cards` and calls `calculate_joker({end_of_round = true, game_over = game_over})`. **This is the exact moment joker order is evaluated for end-of-round triggers** (and what `ActionType.END_ROUND` needs to timestamp so `joker_cv_queue` in `post_processing.py` captures the correct joker order).
   - Subsequently triggers `G.hand` end-of-round effects (Gold cards, Blue seals, Steel + Mime, etc.), calls `draw_from_hand_to_discard()` and `draw_from_discard_to_deck()`, and finally transitions to `G.STATE = G.STATES.ROUND_EVAL` (`state_events.lua:255`).
4. **`Game:update_round_eval()` (`game.lua:3302-3339`)**:
   - Slides `G.round_eval` (created by `create_UIBox_round_evaluation()` in `UI_definitions.lua:1612`) up from the bottom into the center of the screen, covering `G.play` and `G.hand` until the player clicks `Cash Out`.

### Key Timing Nuance (`SPEEDFACTOR` vs. `Moveable:move`)
In `balatro_source/game.lua:2489-2498` and `2617-2634`:
- `G.E_MANAGER` delays (`0.1`, `0.2`, `0.3`, etc.) advance via `G.TIMERS.TOTAL`, which scales by `self.SPEEDFACTOR` (`G.SETTINGS.GAMESPEED` plus `G.ACC`, which ramps up to `16` during long scoring chains).
- `Moveable:move_xy(move_dt)` uses `move_dt = math.min(1/20, self.real_dt)`, which is **not** multiplied by `GAMESPEED`.
- Consequently, at `1x` speed `end_round()` typically fires right after the last card visually clears `G.play`, whereas at `4x` speed or high `G.ACC`, `end_round()` can fire while the last card is still mid-exit across `G.play`. Because of this, labeling directly from an `end_round()` hook in Lua is preferable to a fixed "+0.2s after cards disappear" heuristic.

---

## 2. Suggested ROI & Why Scalar Clip Regression Should Be Avoided

### Suggested Screen Region (`play_area` ROI)
From `common_events.lua:6-7` (`G.play.T.x = G.hand.T.x + (G.hand.T.w - G.play.T.w)/2`, `G.play.T.y = G.hand.T.y - 3.6`) and `game.lua:2224-2227` (`play_W = 5.3*G.CARD_W`, `play_H = 0.95*G.CARD_H`), `G.play` sits directly in the vertical band between `joker_bar` and `card_hand` in `classifiers/inference.py:41-43`:
- `joker_bar`: `y = 20.41 .. 280.38`
- `card_hand`: `y = 567.84 .. 890.57`, `x = 508.32 .. 1579.38`
- **Calibrated `play_area` ROI (1920x1080 base)**: `x = 518.702717, y = 421.037276, w = 1034.463019, h = 287.461606` (~`1034 x 287`, ~3.60:1 aspect ratio, spanning `x ≈ 519..1553, y ≈ 421..708`), calibrated with the `balatro_mod` ROI tool and scaled via `scale_roi_pixels` (`classifiers/common/video_utils.py`).
- This keeps `G.hand` triggers below out of the crop, avoids the right-side overlay zone (`x > 1600`), retains enough headroom for scoring popups, and clearly observes `G.round_eval` sliding over the center when Cash Out appears.

### Why Whole-Clip Fractional Position Regression Is Fragile
A model that uniformly subsamples `[PLAY_HAND, CASH_OUT]` to fixed $N$ frames and regresses a single scalar $p \in [0, 1]$ would struggle with two independent sources of duration variance:
1. **Unbounded scoring time**: ~1s (4x High Card) vs. 60s+ (Mime/Baron/Blueprint retrigger chains). Uniform subsampling of a 60s clip to 32 frames yields ~1.9s/step—too coarse to pinpoint `END_ROUND`.
2. **Unbounded Cash Out pause**: Streamers frequently sit on `ROUND_EVAL` for 5–60s talking to chat before clicking `Cash Out`. Identical scoring sequences would map to wildly different fractional positions (`0.15` vs. `0.85`).

---

## 3. Suggested Model Architecture: Per-Frame Sequence Segmentation + Monotonic Viterbi

A **per-frame sequence segmentation + boundary spotting** model running at a fixed sampling rate (**15–20 fps**) handles arbitrary clip durations cleanly and integrates naturally with the existing PyTorch stack (`classifiers/card_classifier/stage2_classifier/model.py`, `classifiers/shop_classifier/stage2_classifier/model.py`).

### Input Preprocessing
- Extract frames across `[t_play, t_cash_out]` at **15 fps** (every 4th frame of 60fps video) or **20 fps** (every 3rd frame, matching `skip_frames=2` in `config.py:25`), using a single sequential pass (`cv2.VideoCapture` `grab()`/`read()`).
- Crop to `play_area` and resize to **`256 x 72`** (or `288 x 80`), preserving the ~3.7:1 aspect ratio (~18k–23k pixels/frame).

### Network Components
1. **Spatial Frame Encoder** (shared across $T$ frames, batched as `(B*T, 3, 72, 256)`):
   - Pretrained **`MobileNetV3-Small`** backbone (matching `MultiHeadCardClassifier` / `MultiHeadItemClassifier`), truncated before global `avgpool` (output `(B*T, 576, 3, 8)`).
   - **Horizontal Grid Pool**: Replace `AdaptiveAvgPool2d((1, 1))` with **`AdaptiveAvgPool2d((1, 6))`**, followed by `Conv2d(576, 64, kernel_size=1)` -> flatten (`384-d`) -> `Linear(384, 128) + LayerNorm + GELU + Dropout(0.2)`.
   - *Rationale*: Preserving 6 horizontal spatial bins lets the temporal network explicitly track cards clearing left-to-right off the playmat rather than collapsing spatial position.
2. **Temporal Sequence Encoder** (`(B, T, 128) -> (B, T, 256)`):
   - **Residual Dilated Conv1D stem** (e.g., 2 blocks, `kernel_size=5`, dilations `1, 2`) for sharp local motion sensitivity, followed by a **2-layer Bidirectional GRU** (`hidden_size=128`, bidirectional -> `256-d`, `dropout=0.2`).
   - *Rationale*: Handles arbitrary variable-length sequences ($T \in [15, 2000+]$) in $O(T)$ time without positional-encoding length caps, and leverages bidirectional context at post-processing time.
3. **Dual Output Heads**:
   - **4-Class Per-Frame State Head (`Linear(256, 4)`)**:
     - `0: PRE_PLAY` — empty `play_area` immediately after `Play Hand` before cards land.
     - `1: SCORING` — 1–5 cards present, triggering, shattering, or mid-exit.
     - `2: POST_PLAY_EMPTY` — played cards have cleared `play_area` (`end_round()` window), before `G.round_eval` enters.
     - `3: ROUND_EVAL` — dark Cash Out panel covering `play_area`.
   - **Boundary Spotting Head (`Linear(256, 1)`)**: Trained with BCE/focal loss against a narrow Gaussian target ($\sigma \approx 1.5$ frames) centered on the exact `END_ROUND` frame $t^*$.
4. **Monotonic Viterbi Decoding (Inference)**:
   - Run exact 4-state Viterbi DP over the per-frame log-probabilities allowing only non-decreasing transitions (`0 -> 0/1`, `1 -> 1/2`, `2 -> 2/3`, `3 -> 3`; allowing sequence start in `0` or `1` and requiring `1` before `2`).
   - Refine the decoded `1 -> 2` boundary frame using `log P(2|t) - log P(1|t) + alpha * log b_t` within a ±3 frame window.
   - *(Optional)*: If sub-50ms precision is desired, run a second pass at native 60fps over a ±0.5s window around the coarse estimate.

### Edge Cases to Cover
- **All-Glass-Shatter hands**: In `state_events.lua:1091-1095`, shattered/destroyed cards are skipped by `draw_from_play_to_discard()`, so no cards fly right—they explode/dissolve in place. State `1 -> 2` framing handles this naturally as long as training data includes all-shatter hands.
- **Face-down / Stone / Debuffed / Editions**: Boss blinds (The Wheel, The House, The Mark, The Fish) can play face-down cards; editions/seals/enhancements and shaders (`CRT`, reduced motion) alter visual texture.
- **Transient Occlusions**: Mouse cursor, dragged jokers dipping into the top of `play_area`, card-hover tooltips, and discarded cards from `G.hand` grazing the bottom-right corner during `POST_PLAY_EMPTY`.
- **Training-Time Temporal Augmentation**: Loop/stretch random subsegments inside `SCORING` and `ROUND_EVAL` during training so the model never overfits to short synthetic clip durations, and apply temporally consistent compression/brightness/cursor augmentations reusing `classifiers/augmentations.py`.

---

## 4. Dataset Size Estimates

Because the model only separates 4 coarse temporal states rather than 52+ card identities or 230+ shop keys:
- **Minimum Viable Baseline**: **300–500 mod clips** (~15k–25k frames at 15 fps) + **30–50 real VOD segments** for validation.
- **Recommended Production Target**: **1,500–2,500 mod clips** (~75k–150k frames at 15 fps) + **100–200 real streamer VOD segments** (for sim-to-real validation and light fine-tuning, following the pattern in `testing/real_image_fixtures/`).
- **Diminishing Returns**: Beyond ~3,000–4,000 clips unless targeting specific unhandled real-VOD failure modes.

---

## 5. Fast Dataset Generation Options (Target: ~15 min – 2 hours)

Instead of playing full runs in real time through Shop and Blind Select:

### A. In-Place Blind Reset Loop (Lua Mod)
- Keep the game permanently in `G.STATES.SELECTING_HAND`.
- Before each clip: randomize `G.SETTINGS.GAMESPEED` (`0.5, 1, 2, 4`), `G.SETTINGS.GRAPHICS.crt`, `G.SETTINGS.reduced_motion`, blind background colors, 0–8 jokers (including retrigger jokers like `j_mime`, `j_blueprint`, `j_brainstorm`, `j_hanging_chad`, `j_sock_and_buskin`, `j_hack`, `j_dusk` to vary `G.ACC`), and 1–5 selected cards in `G.hand` (including forced all-Glass break cases).
- Set `G.GAME.chips = 0` and `G.GAME.blind.chips = 1` so every played hand wins the round.
- Once `G.STATE == G.STATES.ROUND_EVAL` settles for a random `0.4s–1.5s` of game time, end the clip and immediately reset the board in-place (average simulated clip length: **~2.5–4.5 seconds**).
- **Exact Lua Ground-Truth Hooks**:
  - `t_play`: `G.FUNCS.play_cards_from_highlighted`
  - `t_scoring`: first frame a played card enters `play_area` (via `G.DATASET_MOD.captured_boxes`)
  - `t_end_round`: exact frame `end_round()` reaches `calculate_joker({end_of_round = true})` (`state_events.lua:101`)
  - `t_round_eval`: exact frame `G.round_eval` enters `play_area` (or `G.STATE == G.STATES.ROUND_EVAL`)

### B. Capture Speedup Options
1. **Option 1 — Fixed-Timestep Simulation + Cropped Readback (~15–60 min for 2,000 clips)**:
   - In `balatro_source/main.lua:33-84`, `love.run()` advances the entire engine via `love.update(dt_smooth)` and paces wall-clock time via `love.timer.sleep` / VSync. Nothing in gameplay animations or shaders reads `love.timer.getTime()`.
   - Disabling VSync (`love.window.setVSync(0)`), skipping `love.timer.sleep`, and stepping **4 substeps of `dt = 1/60`** per captured frame (`15 fps` output while keeping `move_dt <= 1/20` physics exact) eliminates real-time waiting.
   - Avoid full-1080p `love.graphics.captureScreenshot` (8 MB CPU readback + PNG encode per frame). Instead, render to an offscreen `Canvas` and read back only the `play_area` sub-rectangle (or run in a smaller window such as `960x540` / `640x360`) and dump raw/fast-encoded frames.
2. **Option 2 — Zero-Engine-Hack Real-Time Mod + FFmpeg Crop (~2 hours for 2,000 clips)**:
   - Run the ~3.5s/clip auto-reset mod at 1x wall-clock time (~1,000 clips/hour) while `ffmpeg` (`ddagrab`/`gdigrab` + NVENC) continuously records the `play_area` crop alongside the mod's JSONL event timestamps.
3. **Option 3 — Semi-Automated Real VOD Bootstrapping (~15 min for 100–200 real clips)**:
   - Run `main.py` on 3–5 real streamer VODs to slice `[PLAY_HAND, CASH_OUT]` segments, run a mod-trained model to propose `t_end_round`, and verify/nudge the boundary frame in a minimal thumbnail scrubber.

---

## 6. Suggested `post_processing.py` Integration Point

In `post_processing.py` (`plan_action_pipeline` / `build_run_trajectory`):
- For each round's `[last PLAY_HAND before CASH_OUT, CASH_OUT]` interval, run the temporal model to emit a synthetic `ScheduledAction` with `ActionType.END_ROUND` at `pred_frame`.
- Schedule `joker_cv_queue[pred_frame + FRAME_OFFSETS[ActionType.END_ROUND]]` (`FRAME_OFFSETS[END_ROUND] = -2` at `post_processing.py:87`) so `_sync_bars_into_state` records the exact pre-`end_round` joker order before `on_action(state, action)` executes `END_ROUND` rules.
