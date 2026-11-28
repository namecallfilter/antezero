"""
Stage 2 Multi-Head Shop & Item Classifier Configuration
=======================================================
Defines classes, indices, input dimensions, and loss weights for shop items:
- Item Key (152 Jokers + 77 Consumables + unknown = 230 classes)
- Edition (5 classes: None, Foil, Holo, Polychrome, Negative)
- Stickers / Status (Eternal, Perishable, Rental, Debuffed)
"""

import json
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
ALL_KEYS_FILE = CURRENT_DIR / "all_keys.json"

# 1. Item Keys (Jokers + Consumables)
if ALL_KEYS_FILE.exists():
    with open(ALL_KEYS_FILE, "r", encoding="utf-8") as f:
        ITEM_KEYS = json.load(f)
else:
    ITEM_KEYS = ["unknown", "j_joker"]

KEY_TO_IDX = {k: i for i, k in enumerate(ITEM_KEYS)}
IDX_TO_KEY = {i: k for i, k in enumerate(ITEM_KEYS)}

# Non-item / placeholder keys present in all_keys.json that have zero Stage 2 shop-item
# training samples (e.g., Balatro UI/run stat keys prefixed with c_, placeholders j_locked /
# j_undiscovered, and playing_card which Stage 1 routes to card_classifier). Masked to
# -inf before softmax during inference so Stage 2 never predicts non-existent items.
INVALID_SHOP_KEYS = frozenset({
    "c_blind_states",
    "c_cards_discarded",
    "c_cards_played",
    "c_cards_sold",
    "c_control",
    "c_dollars_earned",
    "c_face_cards_played",
    "c_hands_played",
    "c_jokers_sold",
    "c_locked",
    "c_losses",
    "c_planetarium_used",
    "c_planets_bought",
    "c_playing_cards_bought",
    "c_round_interest_cap_streak",
    "c_rounds",
    "c_shop_dollars_spent",
    "c_shop_rerolls",
    "c_single_hand_round_streak",
    "c_summit",
    "c_tarot_reading_used",
    "c_tarots_bought",
    "c_trick",
    "c_vouchers_bought",
    "c_wins",
    "j_locked",
    "j_undiscovered",
    "playing_card",
})

# 2. Editions (5 classes: None, Foil, Holo, Polychrome, Negative)
EDITIONS = ["None", "Foil", "Holo", "Polychrome", "Negative"]
EDITION_TO_IDX = {e: i for i, e in enumerate(EDITIONS)}
IDX_TO_EDITION = {i: e for i, e in enumerate(EDITIONS)}

# Standard Balatro card aspect ratio (71 x 95)
# Crop dimensions for classifier input (width, height)
CROP_WIDTH = 128
CROP_HEIGHT = 192

# Head configuration for MultiHeadItemClassifier
HEAD_CONFIG = {
    "key": len(ITEM_KEYS),
    "edition": len(EDITIONS),     # 5
    "eternal": 1,                 # binary logit (BCE)
    "perishable": 1,              # binary logit (BCE)
    "rental": 1,                  # binary logit (BCE)
    "debuffed": 1,                # binary logit (BCE)
}
