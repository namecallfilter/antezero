"""
Stage 2 Multi-Head Card Classifier Configuration
=================================================
Defines classes, indices, input dimensions, and loss weights for all card attributes.
"""

# 1. Card Ranks (15 classes: 13 standard ranks + Stone + None for face-down)
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace", "Stone", "None"]
RANK_TO_IDX = {r: i for i, r in enumerate(RANKS)}
IDX_TO_RANK = {i: r for i, r in enumerate(RANKS)}

# 2. Card Suits (5 classes: 4 suits + None for Stone/face-down)
SUITS = ["Hearts", "Diamonds", "Clubs", "Spades", "None"]
SUIT_TO_IDX = {s: i for i, s in enumerate(SUITS)}
IDX_TO_SUIT = {i: s for i, s in enumerate(SUITS)}

# 3. Editions (4 classes: None, Foil, Holo, Polychrome)
EDITIONS = ["None", "Foil", "Holo", "Polychrome"]
EDITION_TO_IDX = {e: i for i, e in enumerate(EDITIONS)}
IDX_TO_EDITION = {i: e for i, e in enumerate(EDITIONS)}

# 4. Enhancements (9 classes: None + 8 Balatro enhancements)
ENHANCEMENTS = ["None", "Bonus", "Mult", "Wild", "Glass", "Steel", "Stone", "Gold", "Lucky"]
ENHANCEMENT_TO_IDX = {e: i for i, e in enumerate(ENHANCEMENTS)}
IDX_TO_ENHANCEMENT = {i: e for i, e in enumerate(ENHANCEMENTS)}

# 5. Seals (5 classes: None + 4 seal colors)
SEALS = ["None", "Gold", "Red", "Blue", "Purple"]
SEAL_TO_IDX = {s: i for i, s in enumerate(SEALS)}
IDX_TO_SEAL = {i: s for i, s in enumerate(SEALS)}

# Standard card aspect ratio (Balatro card is roughly 71 x 95)
# Crop dimensions for classifier input (width, height)
CROP_WIDTH = 128
CROP_HEIGHT = 192

# Head configuration for MultiHeadCardClassifier
HEAD_CONFIG = {
    "rank": len(RANKS),               # 15
    "suit": len(SUITS),               # 5
    "edition": len(EDITIONS),         # 4
    "enhancement": len(ENHANCEMENTS), # 9
    "seal": len(SEALS),               # 5
    "debuffed": 1,                    # 1 binary logit (BCE)
    "face_down": 1,                   # 1 binary logit (BCE)
}
