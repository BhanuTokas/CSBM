"""
trainer.py
----------
Training and validation loops, checkpoint management, and the main train() entry point.

Kept deliberately thin: all hyperparameters come from TrainConfig,
all model logic lives in model.py, all loss logic in loss.py.
This file only coordinates the epoch loop, logging, and checkpointing.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainConfig
from .loss import ConceptSegmentationLoss
from .metrics import compute_batch_iou, run_linear_probe
from .model import ConceptSegmentPredictor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single epoch
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: ConceptSegmentPredictor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: ConceptSegmentationLoss,
    config: TrainConfig,
    device: torch.device,
    epoch: int,
) -> dict:
    """
    Run one training epoch. Returns a dict of mean metric values.
    """
    model.train()
    totals = dict(loss=0.0, bce=0.0, ortho=0.0, sparsity=0.0, iou=0.0)

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} [train]", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        target = batch["target"].to(device)  # (B, num_concepts, H, W)
        valid = batch["valid"].to(device)  # (B, num_concepts) bool

        optimizer.zero_grad()
        logits = model(images)  # (B, num_concepts, H, W)
        loss_dict = criterion(logits, target, valid)
        loss_dict["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
        optimizer.step()

        iou = compute_batch_iou(logits.detach(), target, valid)
        totals["loss"] += loss_dict["loss"].item()
        totals["bce"] += loss_dict["bce"].item()
        totals["ortho"] += loss_dict["ortho"].item()
        totals["sparsity"] += loss_dict["sparsity"].item()
        totals["iou"] += iou.item()

        pbar.set_postfix(
            loss=f"{loss_dict['loss'].item():.4f}",
            iou=f"{iou.item():.4f}",
        )

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def validate(
    model: ConceptSegmentPredictor,
    loader: DataLoader,
    criterion: ConceptSegmentationLoss,
    device: torch.device,
    epoch: int,
) -> dict:
    """
    Run one validation epoch. Returns a dict of mean metric values.
    """
    model.eval()
    totals = dict(loss=0.0, iou=0.0)

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} [val]  ", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        target = batch["target"].to(device)
        valid = batch["valid"].to(device)

        logits = model(images)
        loss_dict = criterion(logits, target, valid)
        iou = compute_batch_iou(logits, target, valid)

        totals["loss"] += loss_dict["loss"].item()
        totals["iou"] += iou.item()

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_checkpoint(
    output_dir: Path,
    filename: str,
    model: ConceptSegmentPredictor,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_iou: float,
    concept_names: list,
    config: TrainConfig,
    history: list = None,
):
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_iou": val_iou,
        "concept_names": concept_names,
        "config": vars(config),
    }
    if history is not None:
        payload["history"] = history
    torch.save(payload, output_dir / filename)


def load_checkpoint(
    checkpoint_path: str,
    model: ConceptSegmentPredictor,
    optimizer: torch.optim.Optimizer = None,
) -> dict:
    """
    Load a checkpoint into model (and optionally optimizer).
    Returns the checkpoint dict for further inspection.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    logger.info(
        f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
        f"(val IoU: {ckpt.get('val_iou', '?'):.4f})"
    )
    return ckpt


# ---------------------------------------------------------------------------
# Main training orchestrator
# ---------------------------------------------------------------------------


def train(
    config: TrainConfig,
    model: ConceptSegmentPredictor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    concept_names: list,
    device: torch.device,
):
    """
    Full training loop: optional probe baseline, then epoch loop with
    checkpointing and history logging.

    Args:
        config:        TrainConfig with all hyperparameters.
        model:         Initialized ConceptSegmentPredictor on device.
        train_loader:  Training DataLoader.
        val_loader:    Validation DataLoader.
        concept_names: Ordered list of concept names (for checkpoint metadata).
        device:        torch device.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Optional linear probe baseline ---
    if config.run_probe:
        probe_results = run_linear_probe(
            model, train_loader, val_loader, device, len(concept_names)
        )
        probe_path = output_dir / "linear_probe_results.pt"
        torch.save(probe_results, probe_path)
        logger.info(f"Probe results saved → {probe_path}")

    # --- Loss, optimizer, scheduler ---
    criterion = ConceptSegmentationLoss(config).to(device)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)

    # --- Epoch loop ---
    best_val_iou = 0.0
    history = []

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, config, device, epoch
        )
        val_metrics = validate(model, val_loader, criterion, device, epoch)
        scheduler.step()

        logger.info(
            f"Epoch {epoch:3d}/{config.epochs} | "
            f"train loss: {train_metrics['loss']:.4f}  "
            f"train IoU: {train_metrics['iou']:.4f}  "
            f"ortho: {train_metrics['ortho']:.4f} | "
            f"val loss: {val_metrics['loss']:.4f}  "
            f"val IoU: {val_metrics['iou']:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
        )

        # Best checkpoint
        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            save_checkpoint(
                output_dir=output_dir,
                filename="best_concept_extractor.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_iou=best_val_iou,
                concept_names=concept_names,
                config=config,
            )
            logger.info(f"  ✓ Best val IoU: {best_val_iou:.4f} — checkpoint saved.")

        # Periodic checkpoint
        if epoch % config.save_every == 0:
            save_checkpoint(
                output_dir=output_dir,
                filename=f"epoch_{epoch:03d}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_iou=val_metrics["iou"],
                concept_names=concept_names,
                config=config,
                history=history,
            )

    # Save full training history
    history_path = output_dir / "training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    logger.info(f"Training complete. Best val IoU: {best_val_iou:.4f}")
    logger.info(f"History saved → {history_path}")
    logger.info(f"Checkpoints saved → {output_dir}")
