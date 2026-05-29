"""
metrics.py
----------
Evaluation metrics and the linear probe baseline.

IoU (Intersection over Union) is the primary metric — borrowed directly
from NetDissect's methodology for measuring concept-spatial alignment.
Per-concept IoU is more informative than mean IoU alone, since different
concept types (region-like vs attribute-like) will show different alignment.

The linear probe baseline answers: "How much concept-spatial information
is already linearly decodable from raw DINO features?"
If the full model beats the probe by a meaningful margin, the learned
neck + head are doing real work. If not, the backbone alone is sufficient
and a simpler architecture is warranted.
"""

import logging

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------


def compute_batch_iou(
    logits: torch.Tensor,   # (B, C, H, W)
    target: torch.Tensor,   # (B, C, H, W)
    valid: torch.Tensor,    # (B, C) bool
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Mean IoU over all valid (sample, concept) pairs in the batch.
    """
    preds = (torch.sigmoid(logits) > threshold).float()  # (B, C, H, W)

    intersection = (preds * target).sum(dim=(2, 3))       # (B, C)
    union = (preds + target).clamp(max=1).sum(dim=(2, 3)) # (B, C)
    iou = intersection / (union + 1e-6)                   # (B, C)

    # Average only over supervised pairs
    valid_f = valid.float()
    return (iou * valid_f).sum() / valid_f.sum().clamp(min=1.0)


def compute_per_concept_iou(
    logits: torch.Tensor,   # (B, C, H, W)
    target: torch.Tensor,   # (B, C, H, W)
    valid: torch.Tensor,    # (B, C) bool
    num_concepts: int,
    threshold: float = 0.5,
) -> dict:
    """
    Per-concept IoU averaged over all supervised samples for each concept.
    """
    preds = (torch.sigmoid(logits) > threshold).float()

    intersection = (preds * target).sum(dim=(2, 3))       # (B, C)
    union = (preds + target).clamp(max=1).sum(dim=(2, 3)) # (B, C)
    iou = intersection / (union + 1e-6)                   # (B, C)

    per_concept = {}
    for c in range(num_concepts):
        valid_c = valid[:, c]           # (B,) bool
        if not valid_c.any():
            continue
        per_concept[c] = iou[valid_c, c].mean().item()

    return per_concept


# ---------------------------------------------------------------------------
# Linear probe baseline
# ---------------------------------------------------------------------------


def run_linear_probe(
    model,
    train_loader,
    val_loader,
    device: torch.device,
    num_concepts: int,
) -> dict:
    """
    Fit a logistic regression probe per concept directly on raw DINO patch features,
    bypassing the learned neck and head entirely.

    This answers: "Is concept-spatial information already linearly present in DINO?"

    Methodology mirrors NetDissect's unit-alignment evaluation:
    - Features: patch-level DINO tokens, flattened to (N_patches, feat_dim)
    - Target: binary concept presence at each patch location
    - Metric: patch-level IoU (same metric used for the full model)

    Args:
        model:        ConceptSegmentPredictor (used only for extract_dino_features)
        train_loader: Training DataLoader
        val_loader:   Validation DataLoader
        device:       torch device
        num_concepts: Total number of concepts

    Returns:
        Dict mapping concept_idx (int) → linear probe IoU (float).
        Also logs mean probe IoU for immediate comparison.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import jaccard_score
    except ImportError:
        raise ImportError(
            "scikit-learn is required for the linear probe. "
            "Install with: pip install scikit-learn"
        )

    logger.info("Running linear probe baseline on raw DINO features...")
    model.eval()
    probe_results = {}

    for concept_idx in range(num_concepts):
        X_train, y_train = _collect_features(model, train_loader, concept_idx, device)
        X_val, y_val = _collect_features(model, val_loader, concept_idx, device)

        if X_train is None or X_val is None:
            logger.debug(f"Concept {concept_idx}: no samples found, skipping.")
            continue

        clf = LogisticRegression(max_iter=300, C=1.0, verbose=0)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)
        iou = jaccard_score(y_val, preds, zero_division=0)
        probe_results[concept_idx] = iou
        logger.info(f"  Concept {concept_idx:3d} | linear probe IoU: {iou:.4f}")

    if probe_results:
        mean_iou = np.mean(list(probe_results.values()))
        logger.info(f"Linear probe mean IoU (all concepts): {mean_iou:.4f}")
    else:
        logger.warning("Linear probe: no results — check concept CSVs.")

    return probe_results


def _collect_features(
    model,
    loader,
    concept_idx: int,
    device: torch.device,
) -> tuple:
    X_list, y_list = [], []

    for batch in loader:
        valid = batch["valid"]          # (B, C) bool
        # Skip batches where no sample has this concept supervised
        if not valid[:, concept_idx].any():
            continue

        images = batch["image"].to(device)
        target = batch["target"]        # (B, C, H, W)

        with torch.no_grad():
            features = model.extract_dino_features(images)  # (B, D, H_p, W_p)

        B, D, H, W = features.shape

        # Only use samples where this concept is supervised
        valid_c = valid[:, concept_idx]  # (B,) bool

        feat_flat = (
            features[valid_c]
            .permute(0, 2, 3, 1)
            .reshape(-1, D)
            .cpu().numpy()
        )
        mask_flat = (
            target[valid_c, concept_idx]  # (n_valid, H, W)
            .reshape(-1)
            .numpy()
            .astype(int)
        )

        X_list.append(feat_flat)
        y_list.append(mask_flat)

    if not X_list:
        return None, None

    return np.concatenate(X_list), np.concatenate(y_list)