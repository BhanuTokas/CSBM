"""
loss.py
-------
ConceptSegmentationLoss: combined loss for spatial concept mask prediction.

Three terms:
  1. BCE loss       — pixel-level concept presence supervision
  2. Orthogonality  — penalizes overlap between concept masks (disentanglement)
  3. Sparsity       — prevents trivially dense all-ones masks

The orthogonality term is the key research contribution here: it enforces
concept mask consistency (each head "owns" a distinct spatial region) and
is the spatial analogue of CBM disentanglement in classification settings.

Tuning guidance:
  - Start with ortho_weight=0.1, sparsity_weight=0.05.
  - If concept masks look redundant → increase ortho_weight.
  - If masks are too sparse and miss concept regions → decrease sparsity_weight.
  - pos_weight compensates for class imbalance (concept pixels << background).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TrainConfig


class ConceptSegmentationLoss(nn.Module):
    """
    Combined loss for concept mask prediction.

    Args:
        config: TrainConfig with loss weight hyperparameters.
    """

    def __init__(self, config: TrainConfig):
        super().__init__()
        self.bce_weight = config.bce_weight
        self.ortho_weight = config.ortho_weight
        self.sparsity_weight = config.sparsity_weight
        self.register_buffer("pos_weight", torch.tensor(config.pos_weight))

    # ------------------------------------------------------------------
    # Individual loss terms
    # ------------------------------------------------------------------

    def bce_loss(
        self,
        logits: torch.Tensor,  # (B, C, H, W)
        target: torch.Tensor,  # (B, C, H, W)
        valid: torch.Tensor,  # (B, C) bool
    ) -> torch.Tensor:
        """
        BCE only over (sample, concept) pairs that have supervision.
        valid[b, c] = True means sample b has a label for concept c.
        """
        # Per-element BCE, unreduced
        loss_map = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight, reduction="none"
        )  # (B, C, H, W)

        # Mask out unsupervised (sample, concept) pairs
        valid_4d = valid.unsqueeze(-1).unsqueeze(-1).float()  # (B, C, 1, 1)
        masked = loss_map * valid_4d

        # Mean over supervised entries only (avoid dividing by zero)
        n_valid = valid_4d.sum().clamp(min=1.0)
        return masked.sum() / (n_valid * logits.size(2) * logits.size(3))

    def orthogonality_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Penalize spatial overlap between concept masks (disentanglement).

        Computes the Frobenius norm of the off-diagonal entries of the
        normalized Gram matrix of concept masks. Each concept mask is
        treated as a vector over spatial locations.

        A high value means concepts are spatially correlated (bad).
        We minimize this to push concepts toward distinct spatial regions.

        Args:
            logits: (B, C, H, W)
        Returns:
            Scalar orthogonality loss.
        """
        probs = torch.sigmoid(logits)  # (B, C, H, W)
        B, C, H, W = probs.shape

        flat = probs.view(B, C, -1)  # (B, C, H*W)
        flat_norm = F.normalize(flat, dim=-1)  # unit-norm per concept per image
        gram = torch.bmm(flat_norm, flat_norm.transpose(1, 2))  # (B, C, C)

        # Zero out diagonal (self-similarity is always 1, not a penalty)
        eye = torch.eye(C, device=gram.device).unsqueeze(0)
        off_diag = gram * (1.0 - eye)

        return off_diag.pow(2).mean()

    def sparsity_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """
        L1 penalty on sigmoid activations to prevent trivially dense masks.

        Without this, a model can achieve low BCE by predicting everything
        as positive when positive pixels are dense. Sparsity encourages
        selective, concept-specific activations.

        Args:
            logits: (B, C, H, W)
        Returns:
            Scalar sparsity loss.
        """
        return torch.sigmoid(logits).mean()

    # ------------------------------------------------------------------
    # Combined forward
    # ------------------------------------------------------------------

    def forward(
        self,
        logits: torch.Tensor,  # (B, C, H, W)
        target: torch.Tensor,  # (B, C, H, W)
        valid: torch.Tensor,  # (B, C) bool
    ) -> dict:
        bce = self.bce_loss(logits, target, valid)
        ortho = self.orthogonality_loss(logits)
        sparsity = self.sparsity_loss(logits)

        total = (
            self.bce_weight * bce
            + self.ortho_weight * ortho
            + self.sparsity_weight * sparsity
        )

        return {
            "loss": total,
            "bce": bce.detach(),
            "ortho": ortho.detach(),
            "sparsity": sparsity.detach(),
        }
