"""
model.py
--------
ConceptSegmentPredictor: spatial concept mask predictor on frozen DINO features.

Architecture:
    Frozen DINO ViT  →  patch tokens (B, N, D)
                     →  spatial grid (B, D, H_p, W_p)
    Shared conv neck →  refined features (B, neck_dim, H_p, W_p)
    Per-concept head →  concept logits  (B, num_concepts, H_p, W_p)

Design rationale:
- The backbone is frozen by default: we treat DINO as a rich feature extractor
  and only learn the lightweight neck + head. This keeps the probing baseline
  (raw DINO → linear probe) directly comparable to the full model.
- The per-concept head is a single Conv2d(neck_dim, num_concepts, 1), where
  each output channel corresponds to one concept. Concepts are kept independent
  at this stage — mixing happens downstream in the CBM attribution layer.
- predict_concept_masks() exposes a clean inference API that returns binary masks
  after sigmoid thresholding, which is what the PCBM-h post-hoc wrapper will use.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TrainConfig

logger = logging.getLogger(__name__)


class ConceptSegmentPredictor(nn.Module):
    """
    Predicts a spatial binary mask for each concept on top of frozen DINO features.

    Args:
        config: TrainConfig containing backbone name, dimensions, etc.
        num_concepts: Number of concepts to predict (must match training data).
    """

    def __init__(self, config: TrainConfig, num_concepts: int):
        super().__init__()
        self.patch_size = config.patch_size
        self.image_size = config.image_size
        self.num_patches = config.mask_size  # spatial side length (H_p = W_p)
        self.num_concepts = num_concepts

        # --- Backbone (frozen DINO ViT) ---
        logger.info(f"Loading backbone: {config.backbone}")
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            config.backbone,
            pretrained=True,
        )
        if config.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            logger.info("Backbone frozen — only neck + head will be trained.")

        # --- Shared convolutional neck ---
        # Two conv layers refine spatial features before concept prediction.
        # Shared across all concepts so common low-level features are reused.
        self.neck = nn.Sequential(
            nn.Conv2d(
                config.feat_dim, config.neck_dim, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(config.neck_dim),
            nn.GELU(),
            nn.Conv2d(
                config.neck_dim, config.neck_dim, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(config.neck_dim),
            nn.GELU(),
        )

        # --- Per-concept prediction head ---
        # 1x1 conv: each output channel is one concept's spatial logit map.
        # Equivalent to a linear probe per spatial location, extended to all concepts.
        self.concept_head = nn.Conv2d(
            in_channels=config.neck_dim,
            out_channels=num_concepts,
            kernel_size=1,
            bias=True,
        )

        self._init_weights()

    def _init_weights(self):
        for layer in self.neck:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out")
        nn.init.normal_(self.concept_head.weight, std=0.01)
        nn.init.zeros_(self.concept_head.bias)

    def extract_dino_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract patch-level features from the frozen DINO backbone and
        reshape to a spatial feature map.

        Args:
            x: Input images (B, 3, H, W)
        Returns:
            Spatial feature map (B, feat_dim, H_p, W_p)
        """
        with torch.no_grad():
            out = self.backbone.forward_features(x)
            # DINOv2 returns 'x_norm_patchtokens': (B, N_patches, D)
            patch_tokens = out["x_norm_patchtokens"]

        B, N, D = patch_tokens.shape
        H_p = W_p = self.num_patches
        assert N == H_p * W_p, (
            f"Patch count mismatch: expected {H_p * W_p}, got {N}. "
            "Check image_size / patch_size."
        )

        # (B, N, D) → (B, D, H_p, W_p)
        return patch_tokens.permute(0, 2, 1).reshape(B, D, H_p, W_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images (B, 3, H, W)
        Returns:
            concept_logits: (B, num_concepts, H_p, W_p)
                Raw (pre-sigmoid) logits. Apply sigmoid for probabilities.
        """
        features = self.extract_dino_features(x)  # (B, D, H_p, W_p)
        neck_out = self.neck(features)  # (B, neck_dim, H_p, W_p)
        concept_logits = self.concept_head(neck_out)  # (B, num_concepts, H_p, W_p)
        return concept_logits

    @torch.no_grad()
    def predict_concept_masks(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Inference API: returns hard binary concept masks.

        Args:
            x: Input images (B, 3, H, W)
            threshold: Sigmoid threshold for binarization.
        Returns:
            Binary masks (B, num_concepts, H_p, W_p) as float32.
        """
        logits = self.forward(x)
        return (torch.sigmoid(logits) > threshold).float()

    @torch.no_grad()
    def predict_concept_probs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference API: returns soft concept probability maps.

        Args:
            x: Input images (B, 3, H, W)
        Returns:
            Probability maps (B, num_concepts, H_p, W_p) in [0, 1].
        """
        return torch.sigmoid(self.forward(x))


def build_model(config: TrainConfig, num_concepts: int) -> ConceptSegmentPredictor:
    """
    Convenience factory: construct and log model statistics.
    """
    model = ConceptSegmentPredictor(config=config, num_concepts=num_concepts)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Model: {n_trainable:,} trainable / {n_total:,} total parameters "
        f"({100 * n_trainable / n_total:.1f}% trainable)"
    )
    return model
