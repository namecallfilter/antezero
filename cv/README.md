# BalatroEngine

Computer vision pipeline and gamestate recognition system for **Balatro**, designed to automatically parse and extract gamestates and player actions from recorded video gameplay (e.g. YouTube runs by high-level players like Balatro University) to support imitation learning, move prediction, and Monte-Carlo Tree Search (MCTS).

---

## Overview

1. **Template Matching Action Classifier (`main.py`, `matcher.py`, `eval_template.py`, `screen_state.py`, `post_processing.py`)**: A high-speed UI state detector that locates interactive elements (buttons, blinds, shops, booster packs), tracks coarse screen state, evaluates user clicks via physical shadow displacement, and reconstructs full `GameState` trajectories.
2. **Synthetic Dataset Generation Mod (`balatro_mod/`)**: An in-engine Balatro mod (Steamodded / SMODS) that generates perfectly annotated screenshot datasets with pixel-accurate bounding boxes and card metadata, plus ground-truth run telemetry.
3. **Card Classifier Pipeline (`classifiers/card_classifier/`)**: A two-stage detection and classification pipeline that detects playing cards in the hand and predicts rank, suit, edition, enhancement, seal, debuff, face-down status, and collab deck skins.
4. **Shop & Item Classifier Pipeline (`classifiers/shop_classifier/`, `balatro_mod/src/shop_gen.lua`)**: Generates multi-region training data across the top Joker bar, shop slots, booster packs, and consumable slots, and trains a two-stage detector + multi-head classifier accounting for editions, stickers, debuffs, and YouTube mod variants.
5. **HUD OCR & End-of-Round Classifiers (`classifiers/hud_ocr/`, `classifiers/end_round_classifier/`)**: Reads HUD counters (money, hands, discards, ante, round, round score, blind target, reroll cost) from gameplay frames and detects blind-win `END_ROUND` transitions between `Play Hand` and `Cash Out` using a temporal video classifier.
6. **Balatro Game Engine & RNG Simulator (`game/`, `testing/`)**: A Python game-state, scoring, and deterministic LuaJIT PRNG simulation engine validated against in-game Lua telemetry logs.
7. **Frame Extraction & Calibration Tools (`tools/` & root utilities)**: Geometry, shadow evaluation, and ROI cropping utilities to extract gameplay frames and align detection areas across video resolutions.

---

## Main Stuff

### 1. Template Matching & Action Classifier
The primary pipeline for extracting player decisions directly from recorded gameplay videos.

- **High-Speed Template Matching (`main.py`, `matcher.py`)**: Scans video frames within targeted Region of Interest (ROI) bounding boxes using OpenCV template matching with normalized cross-correlation and masking.
- **Button Click Detection via Shadow Analysis (`shadow_evaluator.py`, `eval_template.py`)**: Balatro buttons render a distinct drop shadow beneath them when resting, which disappears when clicked/depressed. A Sobel horizontal edge filter and median gradient analyzer detect the presence or collapse of this shadow to register whether a button was actively clicked.
- **Interactive Calibration & Debug Plotter (`tools/debug_plotter.py`)**: An interactive OpenCV/matplotlib interface that renders match confidence heatmaps, ROI boundaries, and shadow analysis slices in real time, allowing hotkey-based alignment of shadow offsets (`shadow_dy`, `shadow_dw`, `shadow_dh`).
- **Batch Multiprocessing Engine (`main.py`)**: Parallelizes video decoding and frame evaluation using Python `concurrent.futures.ProcessPoolExecutor` with configurable frame skipping (`skip_frames`) for high throughput across multi-hour videos, and supports chaining directly into post-processing via CLI (`--deck`, `--stake`, `--seed`, `--full-video`, `--expressive`).
- **Coarse Screen-State Tracking (`screen_state.py`)**: Restricts active templates by UI screen state (`Blind`, `Shop`, `Select`, `Pack`, `Cash Out`) and applies a post-`Cash Out` cooldown to suppress premature `Next Round` false positives during the Shop slide-in animation.
- **Resolution-Independent Geometry (`geometry.py`)**: Normalized and pixel-level bounding box primitives (`NormalizedROI`, `PixelROI`) that scale UI coordinates seamlessly across 720p, 1080p, and 1440p video sources.
- **Action Post-Processing & Trajectory Reconstruction (`post_processing.py`)**: Deduplicates frame-level button press detections, injects missing `END_ROUND` transitions, extracts CV frames in parallel with `decord`, batches hand/shop CV and HUD OCR, resolves targeted items (`action.metadata["target_card"]`), synchronizes deterministic `GameState` simulation with HUD OCR drift recovery, reports diagnostics via `--expressive`, and saves compact trajectories to `output/trajectories/`.

---

### 2. In-Game Dataset Generator Mod (`balatro_mod/`)
A modular Steamodded mod installed into Balatro to capture synchronized, perfectly labeled CV datasets.

- **Operation Modes**: Configurable from the in-game mod settings menu between **Off**, **Card Classifier**, **Shop Classifier**, **End Round**, **HUD OCR**, **Seed OCR**, and **Deck Classifier**.
- **HUD, Seed & End-of-Round Generators (`src/hud_gen.lua`, `src/seed_gen.lua`, `src/end_round_gen.lua`, `src/end_round_capture.lua`)**: Synthesizes labeled HUD counter crops (money, hands, discards, ante, round, scores, reroll cost), end-of-run seed modals, and captures gameplay clips around round-ending vs non-ending hands for temporal classification.
- **View Deck Generator (`src/deck_gen.lua`)**: Randomizes `G.playing_cards` (15–100 cards concentrated near ~52, suit bias <= 30/suit, 2–20 duplicate card clusters with 2x trait odds, editions, seals, enhancements, and played/greyed states), opens Balatro's View Deck overlay, and exports left-edge YOLO bounding boxes plus card crop labels to `dataset_deck/`.
- **Hand Card Generator (`src/cards_gen.lua`)**: Generates randomized hands with configurable distributions for card ranks, suits, enhancements, editions (Foil/Holo/Poly/Negative), seals, face-down flips, and collab deck skins.
- **Shop & Booster Pack Generator (`src/shop_gen.lua`)**:
  - Cycles between **Main Shop** (2–4 slots) and **Booster Packs** (Buffoon, Arcana, Spectral, Celestial) across normal and Mega sizes.
  - Generates top **Joker Bar** setups (0–9 jokers) with combinations of Eternal, Perishable, Rental stickers, Editions, and Boss Blind debuffs.
  - Generates top **Consumable Slots** (Tarot/Planet/Spectral) with normal and **Perkeo duplicate modes** (supporting negative consumables).
  - Enforces vanilla item pools (Legendary jokers excluded from shop/buffoon packs; *The Soul* and *Black Hole* appropriately allowed in booster packs), and includes Joker stake stickers plus optional Joker Display value/tooltip randomization.
- **In-Engine Bounding Box Calibration (`src/ui.lua`)**: Provides an interactive on-screen editor with hotkeys (`Tab`, `Shift+Tab`, arrow keys, `+`/`-`, `[`/`]`) to align detection ROIs and card bounding boxes in real-time.
- **Automated Capture Engine (`src/capture.lua`)**: Hooks Love2D rendering to compute true bounding boxes from card transformation matrices, saving cropped ROI images alongside YOLO `.txt` labels and structured JSON metadata.
- **Run Telemetry Logger (`src/run_tracker.lua`)**: Records ground-truth `run_telemetry.jsonl` snapshots at each player action frame for closed-loop engine parity verification.

---

### 3. BUMod Art Integration
Balatro University's gameplay features custom card art from the [OceanRamen/BUMod](https://github.com/OceanRamen/BUMod).

- **Historical Texture Integration**: Textures across 2+ years of BUMod commit revisions (e.g. Showman, Scholar, Space Joker, Vampire, Rocket) are extracted into `balatro_mod/assets/bu_variants/`.
- **Dynamic In-Engine Swapping**: During dataset generation, the mod uniformly rolls between vanilla art and historical BUMod sprite revisions, ensuring editions, stickers, and debuff shaders render directly on top of all custom textures.
- **Update Utility (`tools/update_bumod_variants.py`)**: A standalone script that queries the BUMod repository, pulls newly updated Joker sprites into the mod, and regenerates `bu_variants_manifest.lua` for future mod releases.

---

### 4. Card Classifier (`classifiers/card_classifier/`)
A dual-stage neural network architecture optimized for precision on complex card overlaps.

- **Stage 1 (Card Detector)**: A YOLO-based object detector that locates bounding boxes for cards within the player's hand ROI.
- **Stage 2 (Multi-Task Classifier)**: A ResNet/CNN classifier that ingests cropped card images and predicts:
  - **Rank & Suit**: Identifies standard ranks (2–Ace, Stone) and suits, including High Contrast and collab deck variants.
  - **Modifiers**: Identifies card Enhancement (Bonus, Mult, Wild, Glass, Steel, Stone, Gold, Lucky), Edition (Foil, Holo, Polychrome, Negative), and Seal (Gold, Red, Blue, Purple).
  - **Status**: Identifies whether the card is Selected (highlighted upward), Debuffed, or Face-down.
- **End-to-End Pipeline (`pipeline.py`)**: Chains Stage 1 detection and Stage 2 classification into a single inference interface.
- **Automated Training Daemon (`tools/auto_retrain.py`)**: Watches the dataset folder for new captures, prepares training splits, trains both stages, and evaluates checkpoint metrics.

---

### 5. Shop & Item Classifier (`classifiers/shop_classifier/`) & Inference (`classifiers/inference.py`)
A companion two-stage pipeline for recognizing Jokers, consumables, vouchers, booster packs, and shop playing cards across shop and pack screens.

- **Stage 1 (Shop Item Detector)**: A YOLOv8 detector that localizes items across the Joker bar, consumables area, main shop, and booster pack ROIs.
- **Stage 2 (Multi-Head Item Classifier)**: Predicts item key (`all_keys.json`), edition, and sticker/debuff attributes (`eternal`, `perishable`, `rental`, `debuffed`), routing detected playing cards directly to `card_classifier`.
- **Stream Augmentations (`classifiers/augmentations.py`)**: GPU-accelerated training augmentations simulating YouTube video compression/bitrate artifacts, color jitter, and random cursor overlays (`assets/classifiers/cursor.png`).
- **Unified Production Inference (`classifiers/inference.py`, `classifiers/prod_weights/`)**: Provides single-frame (`recognize_hand_cards`, `recognize_shop`) and grouped-batch (`batch_recognize_hands`, `batch_recognize_shops`, `batch_recognize_hud`) recognition APIs.

---

### 6. HUD OCR & End-of-Round Temporal Classifier (`classifiers/hud_ocr/`, `classifiers/end_round_classifier/`)
- **HUD OCR (`classifiers/hud_ocr/`)**: Recognizes numeric HUD fields (`hud_money`, `hud_hands`, `hud_discards`, `hud_ante`, `hud_round`, `hud_round_score`, `hud_blind_target`, `hud_reroll_cost`) from mod-generated and gameplay crops with confidence thresholding and reroll-region color gating.
- **End-of-Round Classifier (`classifiers/end_round_classifier/`)**: Temporal video classifier that scans `[PLAY_HAND, CASH_OUT]` segments via parallel `decord` segment decoding to detect and inject missing `END_ROUND` transitions.

---

### 7. Balatro Game Engine & RNG Simulator (`game/`)
A Python implementation of Balatro's game state, scoring rules, and deterministic RNG for closed-loop state tracking and MCTS/imitation learning.

- **State & Models (`game/balatro_models.py`, `game/gamestate.py`)**: Typed models and state transitions covering hands, jokers, consumables, vouchers, blinds, shop slots, tags, and economy.
- **Trajectory Codec (`game/trajectory_codec.py`)**: Serializes compact `GameStateDelta` trajectories (`serialize_trajectory`) and reconstructs full `RunTrajectory` / `GameState` objects (`rebuild_trajectory`, `rebuild_gamestate`) from `trajectory.json`.
- **Rules, Scoring & Joker Engine (`game/engine/`)**: Implements poker hand evaluation (`poker_hands.py`), round and shop rules (`rules.py`), joker triggers and chip/mult scoring (`jokers.py`, `game/jokers.json`), execution context (`context.py`), and CV observation stubs (`cv_stubs.py`).
- **Deterministic PRNG Simulator (`game/engine/rng_sim/`)**: Python port of Balatro's LuaJIT PRNG pipeline (`rng.py`, `state.py`, `events.py`, `api.py`) for seeded run simulation (`BalatroRun`).

---

### 8. Video Scraping & Auxiliary CV Utilities (`tools/`)
- **`tools/scrape_frame.py`**: Extracts frames from local gameplay video files at specified timestamps.
- **`tools/adjust_rois.py` & `tools/grab_rois.py`**: Interactive visual tools to fine-tune region-of-interest coordinates across different video resolutions.
- **`tools/assign_test_images.py`**: Tooling to curate test evaluation splits from actual YouTube gameplay footage.
- **`tools/crop_transparent_borders.py` & `tools/manual_crop.py`**: Cleans and normalizes alpha channels on raw UI template crops and supports manual ROI cropping.
- **`tools/debug_plotter.py`**: Interactive OpenCV/matplotlib debug plotter for template matching and shadow calibration.
- **`tools/auto_retrain.py` & `tools/update_bumod_variants.py`**: Dataset-monitoring auto-retrainer and BUMod sprite sync utility.

---

### 9. Testing & Parity Verification (`testing/`)
- **Engine Parity Suite (`testing/test_engine_parity.py`)**: Replays archived in-game `run_telemetry.jsonl` logs (`testing/telemetry_data/`) to verify closed-loop `GameState` parity against real Balatro runs.
- **Unit & Visual Tests**: Covers engine transitions (`test_engine.py`), poker hand scoring (`test_poker_hands.py`), CV stubs (`test_cv_stubs.py`), Blueprint compatibility (`test_blueprint_compat.py`), template matching (`test_templates.py`, `visualize_tests.py`), screen-state gating (`test_screen_state.py`), HUD OCR (`test_hud_ocr.py`), post-processing (`test_post_processing.py`), and trajectory round-trip reconstruction (`test_trajectory_codec.py`).

---

## Usage Quickstart

### Running the Template Matching & Post-Processing Pipeline on Gameplay Video
```powershell
# Run full end-to-end action detection + post-processing trajectory reconstruction
python main.py --input-video path/to/video.mp4 --full-video --deck red --stake gold --seed SEED123 --expressive

# Run Stage 2 post-processing standalone from an existing actions JSON
python post_processing.py --actions-json output/vod_actions/run.json --video path/to/video.mp4 --deck red --stake gold --expressive

# Launch interactive calibration / debug plotting mode for template ROIs and shadows
python main.py --setup_mode --debug_plot
```

### Updating BUMod Textures
```powershell
python tools/update_bumod_variants.py
```

### Training Card Classifier
```powershell
# Prepare dataset splits from raw mod captures
python classifiers/card_classifier/prepare_dataset.py

# Run Stage 1 detector and Stage 2 multi-task classifier training
python classifiers/card_classifier/train.py

# Or launch continuous background retrainer:
python tools/auto_retrain.py --model card
```

### Training Shop & Item Classifier
```powershell
# Sync latest shop dataset from mod and inspect statistics
python classifiers/shop_classifier/prepare_dataset.py

# Run Stage 1 detector (YOLO) and Stage 2 multi-head item classifier
python classifiers/shop_classifier/train.py --sync --stage all

# Or launch continuous background retrainer (monitors dataset_shop and auto-trains):
python tools/auto_retrain.py --model shop --target 10000
```

### Training View Deck Classifier (Left-Edge YOLO + Shared Card Classifier Stage 2)
```powershell
# Sync View Deck dataset from mod, train Stage 1 left-edge YOLO detector,
# and retrain Stage 2 card classifier on hand + shop crops + View Deck crops:
python classifiers/deck_classifier/train.py --sync --stage all

# Or monitor dataset_deck generation and auto-train once target frame count is reached:
python tools/auto_retrain.py --model deck --target 2500
```

### Running Pipeline on Video Frames
```powershell
python classifiers/card_classifier/test_video_frame.py --input path/to/frame.png
python classifiers/shop_classifier/test_video_frame.py --input path/to/frame.png
python classifiers/deck_classifier/test_video_frame.py --input path/to/frame.png
```

### Running Game Engine & Parity Tests
```powershell
python -m unittest discover -s testing
python testing/test_engine_parity.py --all
```
