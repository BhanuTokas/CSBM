"""
config.py
---------
Argument parsing and hyperparameter management.
All tuneable parameters live here — nothing is hardcoded elsewhere.
"""

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Feature dimensions for each supported DINO backbone variant
BACKBONE_FEAT_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
}


@dataclass
class TrainConfig:
    """
    Central configuration object. Constructed from argparse args or manually.
    Passed through to all modules so there is a single source of truth.
    """

    # Paths
    broden_path: str = ""
    output_dir: str = "./checkpoints"

    # Backbone
    backbone: str = "dinov2_vitb14"
    freeze_backbone: bool = True

    # Data
    num_concepts: int = 50
    image_size: int = 224
    patch_size: int = 14
    max_per_concept: int = 5000

    # Model
    neck_dim: int = 256

    # Training
    batch_size: int = 32
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    num_workers: int = 4
    seed: int = 42
    save_every: int = 5  # save a checkpoint every N epochs

    # Loss weights
    bce_weight: float = 1.0
    ortho_weight: float = 0.1
    sparsity_weight: float = 0.05
    pos_weight: float = 5.0  # BCE positive class weight

    # Extras
    run_probe: bool = False  # run linear probe baseline before training

    # Derived (set post-init)
    feat_dim: int = field(init=False)
    mask_size: int = field(init=False)

    def __post_init__(self):
        if self.backbone not in BACKBONE_FEAT_DIMS:
            raise ValueError(
                f"Unknown backbone '{self.backbone}'. "
                f"Choose from: {list(BACKBONE_FEAT_DIMS.keys())}"
            )
        self.feat_dim = BACKBONE_FEAT_DIMS[self.backbone]
        self.mask_size = self.image_size // self.patch_size

        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size ({self.image_size}) must be divisible by "
                f"patch_size ({self.patch_size})"
            )


def parse_args() -> TrainConfig:
    """Parse command-line arguments and return a TrainConfig."""
    parser = argparse.ArgumentParser(
        description="Train concept extraction module on Broden",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    parser.add_argument(
        "--broden_path",
        type=str,
        required=True,
        help="Path to Broden dataset root directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints and logs",
    )

    # Backbone
    parser.add_argument(
        "--backbone",
        type=str,
        default="dinov2_vitb14",
        choices=list(BACKBONE_FEAT_DIMS.keys()),
        help="DINO backbone variant",
    )
    parser.add_argument(
        "--no_freeze", action="store_true", help="Fine-tune backbone (default: frozen)"
    )

    # Data
    parser.add_argument(
        "--num_concepts",
        type=int,
        default=50,
        help="Number of Broden concepts to train on",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Input image size (must be divisible by patch_size)",
    )
    parser.add_argument("--patch_size", type=int, default=14, help="DINO patch size")
    parser.add_argument(
        "--max_per_concept",
        type=int,
        default=5000,
        help="Max training samples per concept (for balance)",
    )

    # Model
    parser.add_argument(
        "--neck_dim",
        type=int,
        default=256,
        help="Hidden dimension of the convolutional neck",
    )

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate (neck + head only)"
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save_every", type=int, default=5, help="Save a checkpoint every N epochs"
    )

    # Loss
    parser.add_argument("--bce_weight", type=float, default=1.0)
    parser.add_argument(
        "--ortho_weight",
        type=float,
        default=0.1,
        help="Weight for concept orthogonality (disentanglement) loss",
    )
    parser.add_argument(
        "--sparsity_weight",
        type=float,
        default=0.05,
        help="Weight for mask sparsity regularization",
    )
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=5.0,
        help="BCE positive class weight (concept pixels are rare)",
    )

    # Extras
    parser.add_argument(
        "--run_probe",
        action="store_true",
        help="Run linear probe baseline before training",
    )

    args = parser.parse_args()

    config = TrainConfig(
        broden_path=args.broden_path,
        output_dir=args.output_dir,
        backbone=args.backbone,
        freeze_backbone=not args.no_freeze,
        num_concepts=args.num_concepts,
        image_size=args.image_size,
        patch_size=args.patch_size,
        max_per_concept=args.max_per_concept,
        neck_dim=args.neck_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        num_workers=args.num_workers,
        seed=args.seed,
        save_every=args.save_every,
        bce_weight=args.bce_weight,
        ortho_weight=args.ortho_weight,
        sparsity_weight=args.sparsity_weight,
        pos_weight=args.pos_weight,
        run_probe=args.run_probe,
    )

    logger.info("Config:")
    for k, v in vars(config).items():
        logger.info(f"  {k}: {v}")

    return config
