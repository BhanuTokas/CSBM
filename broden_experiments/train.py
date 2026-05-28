"""
train.py
--------
Entry point for training the concept extraction module on Broden.

Usage:
    python train.py --broden_path /path/to/broden \
                    --output_dir ./checkpoints \
                    --backbone dinov2_vitb14 \
                    --num_concepts 50 \
                    --run_probe

This script wires together the modules in concept_extraction/:
    config  →  dataset  →  model  →  trainer
"""

import logging
import numpy as np
import torch

from concept_extraction.config import parse_args
from concept_extraction.dataset import get_concept_names, build_dataloaders
from concept_extraction.model import build_model
from concept_extraction.trainer import train

# ---------------------------------------------------------------------------
# Logging (configured here so it covers all submodules)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    # --- Config ---
    config = parse_args()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # --- Concepts ---
    concept_names = get_concept_names(config.broden_path, config.num_concepts)
    num_concepts = len(concept_names)

    # --- Data ---
    train_loader, val_loader = build_dataloaders(config, concept_names)

    # --- Model ---
    model = build_model(config, num_concepts).to(device)

    # --- Train ---
    train(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        concept_names=concept_names,
        device=device,
    )


if __name__ == "__main__":
    main()
