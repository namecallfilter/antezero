"""
Multi-Head Card Attribute Classifier Model
==========================================
Architecture:
- Shared feature extraction backbone (MobileNetV3-Small / MobileNetV3-Large / ResNet18)
- 5 task-specific linear heads:
  1. Rank (14 classes: 2..A + Stone) -> Softmax
  2. Suit (5 classes: Hearts, Diamonds, Clubs, Spades, None) -> Softmax
  3. Edition (4 classes: None, Foil, Holo, Polychrome) -> Softmax
  4. Enhancement (9 classes: None, Bonus, Mult, Wild, Glass, Steel, Gold, Lucky, Stone) -> Softmax
  5. Seal (5 classes: None, Gold, Red, Blue, Purple) -> Softmax
"""

import torch
import torch.nn as nn
import torchvision.models as models

try:
    from .config import HEAD_CONFIG
except ImportError:
    from config import HEAD_CONFIG


class MultiHeadCardClassifier(nn.Module):
    def __init__(self, backbone_name="mobilenet_v3_small", pretrained=True, dropout=0.2):
        super().__init__()
        self.backbone_name = backbone_name

        if backbone_name == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            base = models.mobilenet_v3_small(weights=weights)
            self.features = base.features
            self.avgpool = base.avgpool
            self.fc_shared = nn.Sequential(
                base.classifier[0],  # Linear(576, 1024)
                base.classifier[1],  # Hardswish()
                nn.Dropout(p=dropout, inplace=True)
            )
            feat_dim = 1024
        elif backbone_name == "mobilenet_v3_large":
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            base = models.mobilenet_v3_large(weights=weights)
            self.features = base.features
            self.avgpool = base.avgpool
            self.fc_shared = nn.Sequential(
                base.classifier[0],  # Linear(960, 1280)
                base.classifier[1],  # Hardswish()
                nn.Dropout(p=dropout, inplace=True)
            )
            feat_dim = 1280
        elif backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base = models.resnet18(weights=weights)
            self.features = nn.Sequential(
                base.conv1, base.bn1, base.relu, base.maxpool,
                base.layer1, base.layer2, base.layer3, base.layer4
            )
            self.avgpool = base.avgpool
            self.fc_shared = nn.Sequential(
                nn.Dropout(p=dropout)
            )
            feat_dim = base.fc.in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Attribute-specific output classification heads
        self.head_rank = nn.Linear(feat_dim, HEAD_CONFIG["rank"])
        self.head_suit = nn.Linear(feat_dim, HEAD_CONFIG["suit"])
        self.head_edition = nn.Linear(feat_dim, HEAD_CONFIG["edition"])
        self.head_enhancement = nn.Linear(feat_dim, HEAD_CONFIG["enhancement"])
        self.head_seal = nn.Linear(feat_dim, HEAD_CONFIG["seal"])
        self.head_debuffed = nn.Linear(feat_dim, HEAD_CONFIG["debuffed"])
        self.head_face_down = nn.Linear(feat_dim, HEAD_CONFIG["face_down"])

    def forward(self, x):
        """
        Forward pass for a batch of card crop images.
        Args:
            x: Tensor of shape (B, 3, CROP_HEIGHT, CROP_WIDTH)
        Returns:
            Dict containing raw logits for each attribute head.
        """
        feats = self.features(x)
        pooled = self.avgpool(feats)
        flattened = torch.flatten(pooled, 1)
        shared = self.fc_shared(flattened)

        return {
            "rank": self.head_rank(shared),
            "suit": self.head_suit(shared),
            "edition": self.head_edition(shared),
            "enhancement": self.head_enhancement(shared),
            "seal": self.head_seal(shared),
            "debuffed": self.head_debuffed(shared).squeeze(-1),
            "face_down": self.head_face_down(shared).squeeze(-1),
        }

    def predict_attributes(self, x):
        """
        Runs inference and converts logits into class indices and confidence scores.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            
            p_rank = torch.softmax(logits["rank"], dim=-1)
            p_suit = torch.softmax(logits["suit"], dim=-1)
            p_edition = torch.softmax(logits["edition"], dim=-1)
            p_enhancement = torch.softmax(logits["enhancement"], dim=-1)
            p_seal = torch.softmax(logits["seal"], dim=-1)
            p_debuffed = torch.sigmoid(logits["debuffed"])
            p_face_down = torch.sigmoid(logits["face_down"])

            return {
                "rank_idx": p_rank.argmax(dim=-1),
                "rank_conf": p_rank.max(dim=-1).values,
                "rank_probs": p_rank,
                "suit_idx": p_suit.argmax(dim=-1),
                "suit_conf": p_suit.max(dim=-1).values,
                "suit_probs": p_suit,
                "edition_idx": p_edition.argmax(dim=-1),
                "edition_conf": p_edition.max(dim=-1).values,
                "edition_probs": p_edition,
                "enhancement_idx": p_enhancement.argmax(dim=-1),
                "enhancement_conf": p_enhancement.max(dim=-1).values,
                "enhancement_probs": p_enhancement,
                "seal_idx": p_seal.argmax(dim=-1),
                "seal_conf": p_seal.max(dim=-1).values,
                "seal_probs": p_seal,
                "debuffed": p_debuffed > 0.5,
                "debuffed_conf": p_debuffed,
                "face_down": p_face_down > 0.5,
                "face_down_conf": p_face_down,
            }


def infer_backbone_from_checkpoint(ckpt, default_backbone: str = "mobilenet_v3_small") -> str:
    """Detects classifier backbone from checkpoint metadata or linear layer input dims."""
    valid = {"mobilenet_v3_small", "mobilenet_v3_large", "resnet18"}
    if isinstance(ckpt, dict):
        meta_bb = ckpt.get("backbone")
        if isinstance(meta_bb, str) and meta_bb in valid:
            return meta_bb
        state_dict = ckpt.get("model_state_dict") or ckpt.get("model") or ckpt
    else:
        state_dict = ckpt
    if isinstance(state_dict, dict):
        fc_w = state_dict.get("fc_shared.0.weight")
        if isinstance(fc_w, torch.Tensor) and fc_w.ndim == 2:
            if fc_w.shape[1] == 960:
                return "mobilenet_v3_large"
            if fc_w.shape[1] == 576:
                return "mobilenet_v3_small"
        head_w = state_dict.get("head_rank.weight")
        if isinstance(head_w, torch.Tensor) and head_w.ndim == 2:
            if head_w.shape[1] == 1280:
                return "mobilenet_v3_large"
            if head_w.shape[1] == 1024:
                return "mobilenet_v3_small"
            if head_w.shape[1] == 512:
                return "resnet18"
    return default_backbone
