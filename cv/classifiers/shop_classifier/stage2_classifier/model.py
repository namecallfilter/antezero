"""
Multi-Head Shop Item Classifier Model
=====================================
Architecture:
- Shared feature extraction backbone (MobileNetV3-Small / MobileNetV3-Large / ResNet18)
- 6 task-specific output heads:
  1. Key (230 classes: 152 Jokers + 77 Consumables + unknown) -> Softmax
  2. Edition (5 classes: None, Foil, Holo, Polychrome, Negative) -> Softmax
  3. Eternal (binary logit) -> BCE
  4. Perishable (binary logit) -> BCE
  5. Rental (binary logit) -> BCE
  6. Debuffed (binary logit) -> BCE
"""

import torch
import torch.nn as nn
import torchvision.models as models

try:
    from .config import HEAD_CONFIG, IDX_TO_KEY, IDX_TO_EDITION, KEY_TO_IDX, INVALID_SHOP_KEYS
except ImportError:
    from config import HEAD_CONFIG, IDX_TO_KEY, IDX_TO_EDITION, KEY_TO_IDX, INVALID_SHOP_KEYS


class MultiHeadItemClassifier(nn.Module):
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
        self.head_key = nn.Linear(feat_dim, HEAD_CONFIG["key"])
        self.head_edition = nn.Linear(feat_dim, HEAD_CONFIG["edition"])
        self.head_eternal = nn.Linear(feat_dim, HEAD_CONFIG["eternal"])
        self.head_perishable = nn.Linear(feat_dim, HEAD_CONFIG["perishable"])
        self.head_rental = nn.Linear(feat_dim, HEAD_CONFIG["rental"])
        self.head_debuffed = nn.Linear(feat_dim, HEAD_CONFIG["debuffed"])

        invalid_key_mask = torch.zeros(HEAD_CONFIG["key"], dtype=torch.bool)
        for key_name in INVALID_SHOP_KEYS:
            idx = KEY_TO_IDX.get(key_name)
            if idx is not None and 0 <= idx < HEAD_CONFIG["key"]:
                invalid_key_mask[idx] = True
        self.register_buffer("_invalid_key_mask", invalid_key_mask, persistent=False)

    def forward(self, x):
        """
        Forward pass for a batch of cropped item images.
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
            "key": self.head_key(shared),
            "edition": self.head_edition(shared),
            "eternal": self.head_eternal(shared).squeeze(-1),
            "perishable": self.head_perishable(shared).squeeze(-1),
            "rental": self.head_rental(shared).squeeze(-1),
            "debuffed": self.head_debuffed(shared).squeeze(-1),
        }

    def predict_attributes(self, x):
        """
        Runs inference and converts logits into string names and confidence scores.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)

            # Softmax heads (mask zero-sample non-item / placeholder keys to -inf)
            key_logits = logits["key"].masked_fill(self._invalid_key_mask, float("-inf"))
            key_probs = torch.softmax(key_logits, dim=-1)
            key_confs, key_preds = torch.max(key_probs, dim=-1)
            top_k = min(5, key_probs.size(-1))
            top_vals, top_idxs = torch.topk(key_probs, k=top_k, dim=-1)

            ed_probs = torch.softmax(logits["edition"], dim=-1)
            ed_confs, ed_preds = torch.max(ed_probs, dim=-1)

            # Sigmoid binary heads
            eternal_probs = torch.sigmoid(logits["eternal"])
            perishable_probs = torch.sigmoid(logits["perishable"])
            rental_probs = torch.sigmoid(logits["rental"])
            debuffed_probs = torch.sigmoid(logits["debuffed"])

            results = []
            for i in range(x.size(0)):
                k_idx = key_preds[i].item()
                e_idx = ed_preds[i].item()
                top_keys = [
                    (
                        IDX_TO_KEY.get(int(top_idxs[i, j].item()), "unknown"),
                        float(top_vals[i, j].item()),
                    )
                    for j in range(top_k)
                ]
                results.append({
                    "key": IDX_TO_KEY.get(k_idx, "unknown"),
                    "key_conf": key_confs[i].item(),
                    "top_keys": top_keys,
                    "edition": IDX_TO_EDITION.get(e_idx, "None"),
                    "edition_conf": ed_confs[i].item(),
                    "eternal": eternal_probs[i].item() > 0.5,
                    "eternal_conf": eternal_probs[i].item(),
                    "perishable": perishable_probs[i].item() > 0.5,
                    "perishable_conf": perishable_probs[i].item(),
                    "rental": rental_probs[i].item() > 0.5,
                    "rental_conf": rental_probs[i].item(),
                    "debuffed": debuffed_probs[i].item() > 0.5,
                    "debuffed_conf": debuffed_probs[i].item(),
                })
            return results


def infer_backbone_from_checkpoint(ckpt, default_backbone: str = "mobilenet_v3_small") -> str:
    """Detects shop classifier backbone from checkpoint metadata or linear layer input dims."""
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
        head_w = state_dict.get("head_key.weight")
        if isinstance(head_w, torch.Tensor) and head_w.ndim == 2:
            if head_w.shape[1] == 1280:
                return "mobilenet_v3_large"
            if head_w.shape[1] == 1024:
                return "mobilenet_v3_small"
            if head_w.shape[1] == 512:
                return "resnet18"
    return default_backbone
