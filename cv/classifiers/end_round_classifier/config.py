"""
Configuration for Temporal End-Round Classifier
===============================================
Defines frame states, input resolution, play_area ROI coordinates, sampling rate,
loss/decoding hyperparameters, and dataset directory conventions.
"""

from __future__ import annotations

# Ordered monotonic sequence states between PLAY_HAND and CASH_OUT
STATES = [
    "PRE_PLAY",         # 0: Brief window after Play Hand before cards arrive in G.play
    "SCORING",          # 1: 1-5 cards present, triggering, shattering, or mid-exit in G.play
    "POST_PLAY_EMPTY",  # 2: Played cards have cleared / end_round() window, pre-ROUND_EVAL
    "ROUND_EVAL",       # 3: G.round_eval Cash Out panel covers play_area
]
STATE_TO_IDX = {name: idx for idx, name in enumerate(STATES)}
IDX_TO_STATE = {idx: name for idx, name in enumerate(STATES)}
NUM_STATES = len(STATES)
IGNORE_INDEX = -100

# Base 1920x1080 reference geometry (matches classifiers/inference.py).
# joker_bar spans y=20.41..280.38; card_hand spans y=567.84..890.57, x=508.32..1579.38.
# play_area occupies the horizontal center band between them, excluding right-side overlays.
# Note: calibrate exact bounds in balatro_mod/dataset/boxes.lua once mod support is added.
BASE_WIDTH = 1920.0
BASE_HEIGHT = 1080.0
PLAY_AREA_ROI = {
    "x": 518.702717,
    "y": 421.037276,
    "w": 1034.463019,
    "h": 287.461606,
}

# Resized network spatial input (preserves ~3.7:1 playmat aspect ratio)
INPUT_WIDTH = 256
INPUT_HEIGHT = 72

# Temporal sampling rate used during clip preparation, training, and inference
TARGET_FPS = 15.0

# Gaussian target width (in frames at TARGET_FPS) for the boundary spotting head
BOUNDARY_SIGMA_FRAMES = 1.5

# Standard ImageNet normalization (matches card_classifier & shop_classifier)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Dataset sync & storage defaults
MOD_SOURCE_SUBFOLDER = "dataset_end_round"
SCHEMA_VERSION = 1

# Model defaults
DEFAULT_BACKBONE = "mobilenet_v3_small"
DEFAULT_SPATIAL_POOL_W = 6
DEFAULT_FRAME_FEAT_DIM = 128
DEFAULT_TEMPORAL_HIDDEN_DIM = 128
DEFAULT_NUM_GRU_LAYERS = 2
DEFAULT_DROPOUT = 0.2

# Training defaults
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_MAX_TRAIN_SEQ_LEN = 256  # ~17s at 15 fps; validation runs full uncropped clips
DEFAULT_BOUNDARY_LOSS_WEIGHT = 2.0
DEFAULT_TRANSITION_SMOOTH_WEIGHT = 0.1
DEFAULT_VAL_RATIO = 0.20
DEFAULT_SEED = 42

# Inference / Viterbi decoding defaults
DEFAULT_MIN_SCORING_FRAMES = 2
DEFAULT_MIN_POST_PLAY_FRAMES = 1
DEFAULT_BOUNDARY_REFINE_RADIUS = 5
DEFAULT_BOUNDARY_REFINE_ALPHA = 0.5
