from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchvision.transforms as T
from .config import TrainConfig

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MASK_CATEGORIES = {"color", "object", "material", "part"}
WEAK_CATEGORIES = {"scene", "texture"}
CATEGORY_ORDER = ["object", "part", "material", "color", "scene", "texture"]


# ---------------------------------------------------------------------------
# Concept discovery
# ---------------------------------------------------------------------------


def get_concept_names(broden_path: str, num_concepts: int) -> list[str]:
    """
    Returns up to num_concepts individual concept names (e.g. 'wall', 'sky',
    'wood') ordered by category priority then by frequency (most frequent first).
    """
    broden_path = Path(broden_path)
    all_concepts = []  # list of (name, category, code)

    for cat in CATEGORY_ORDER:
        csv_path = broden_path / f"c_{cat}.csv"
        if not csv_path.exists():
            log.warning(f"Missing {csv_path.name} — skipping category.")
            continue
        df = pd.read_csv(csv_path).sort_values("frequency", ascending=False)
        for _, row in df.iterrows():
            all_concepts.append((row["name"], cat, int(row["code"])))

    if not all_concepts:
        raise ValueError(f"No concept CSVs found in {broden_path}.")

    if num_concepts > len(all_concepts):
        log.warning(
            f"--num_concepts {num_concepts} exceeds available "
            f"{len(all_concepts)}; using all."
        )

    selected = all_concepts[:num_concepts]
    names = [n for n, _, _ in selected]
    log.info(
        f"Discovered {len(all_concepts)} concepts total; "
        f"using {len(selected)}: {names[:10]}"
        f"{'...' if len(selected) > 10 else ''}"
    )
    return names


def get_concept_metadata(broden_path: str, concept_names: list[str]) -> list[dict]:
    """
    For each concept name, return its category and pixel code.
    Used internally by BrodenConceptDataset.
    """
    broden_path = Path(broden_path)
    # Build lookup: name -> (category, code)
    lookup: dict[str, tuple[str, int]] = {}
    for cat in CATEGORY_ORDER:
        csv_path = broden_path / f"c_{cat}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            # First occurrence wins (concept may appear in multiple categories)
            if row["name"] not in lookup:
                lookup[row["name"]] = (cat, int(row["code"]))

    meta = []
    for name in concept_names:
        if name not in lookup:
            raise ValueError(
                f"Concept '{name}' not found in any c_*.csv. "
                f"Available: {list(lookup.keys())[:20]}"
            )
        cat, code = lookup[name]
        meta.append({"name": name, "category": cat, "code": code})
    return meta


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class BrodenConceptDataset(Dataset):
    def __init__(
        self,
        broden_path: str | Path,
        concept_names: list[str],
        split: str,
        config: TrainConfig,
        max_per_concept: int = 5000,
    ):
        self.broden_path = Path(broden_path)
        self.concept_names = concept_names
        self.split = split
        self.mask_size = config.mask_size

        # Metadata: category + pixel code for each concept
        self.concept_meta = get_concept_metadata(str(broden_path), concept_names)

        self.img_transform = T.Compose(
            [
                T.Resize((config.image_size, config.image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.samples = self._build_samples(max_per_concept)
        log.info(
            f"[{split}] {len(self.samples)} samples "
            f"across {len(concept_names)} concepts."
        )

    # ------------------------------------------------------------------

    def _build_samples(self, max_per_concept: int) -> list[dict]:
        idx = pd.read_csv(self.broden_path / "index.csv")
        idx = idx[idx["split"] == self.split].reset_index(drop=True)

        # Group concepts by category so we scan index.csv once per category
        from collections import defaultdict

        cat_to_concepts: dict[str, list[int]] = defaultdict(list)
        for ci, meta in enumerate(self.concept_meta):
            cat_to_concepts[meta["category"]].append(ci)

        # per_concept_count[ci] tracks how many samples we've collected
        per_concept_count = [0] * len(self.concept_names)

        # image -> {concept_idx: label_entry}
        image_labels: dict[str, dict[int, dict]] = defaultdict(dict)

        for cat, concept_indices in cat_to_concepts.items():
            if cat not in idx.columns:
                log.warning(f"Category '{cat}' not in index.csv — skipping.")
                continue

            cat_rows = idx[idx[cat].notna() & (idx[cat] != "")]

            for _, row in cat_rows.iterrows():
                cell = str(row[cat]).strip()
                img_key = row["image"]

                for ci in concept_indices:
                    if per_concept_count[ci] >= max_per_concept:
                        continue

                    meta = self.concept_meta[ci]

                    if cat in MASK_CATEGORIES:
                        # Cell may be "a/b.png;a/c.png"
                        paths = [
                            self.broden_path / "images" / p.strip()
                            for p in cell.split(";")
                        ]
                        image_labels[img_key][ci] = {
                            "type": "mask",
                            "paths": paths,
                            "code": meta["code"],
                        }
                    else:  # WEAK_CATEGORIES
                        label_ids = [int(float(x)) for x in cell.split(";")]
                        image_labels[img_key][ci] = {
                            "type": "weak",
                            "label_ids": label_ids,
                            "code": meta["code"],
                        }

                    per_concept_count[ci] += 1

        # Flatten to sample list
        samples = []
        for img_key, concept_labels in image_labels.items():
            samples.append(
                {
                    "img_path": self.broden_path / "images" / img_key,
                    "concept_labels": concept_labels,
                }
            )

        return samples

    # ------------------------------------------------------------------

    def _load_seg_mask(self, path: Path) -> np.ndarray:
        """Returns (H, W) int32 label map: pixel_value = R + 256*G."""
        seg = np.array(Image.open(path).convert("RGB"))
        return seg[:, :, 0].astype(np.int32) + 256 * seg[:, :, 1].astype(np.int32)

    def _mask_to_binary(self, paths: list[Path], code: int) -> torch.Tensor:
        """
        Load one or more seg masks, binarize by pixel code, OR together.
        Returns (1, mask_size, mask_size).
        """
        combined = None
        for path in paths:
            label_map = self._load_seg_mask(path)
            binary = (label_map == code).astype(np.float32)
            combined = binary if combined is None else np.maximum(combined, binary)

        tensor = torch.from_numpy(combined).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        return F.interpolate(
            tensor, size=(self.mask_size, self.mask_size), mode="nearest"
        ).squeeze(
            0
        )  # (1, mask_size, mask_size)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        image = Image.open(sample["img_path"]).convert("RGB")
        image = self.img_transform(image)

        target = torch.zeros(len(self.concept_names), self.mask_size, self.mask_size)
        valid_mask = torch.zeros(len(self.concept_names), dtype=torch.bool)

        for ci, entry in sample["concept_labels"].items():
            if entry["type"] == "mask":
                target[ci] = self._mask_to_binary(entry["paths"], entry["code"])
            else:  # weak — broadcast concept presence over full spatial map
                target[ci] = 1.0
            valid_mask[ci] = True

        return {"image": image, "target": target, "valid": valid_mask}


# ---------------------------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------------------------


def build_dataloaders(
    config: TrainConfig,
    concept_names: list[str],
) -> tuple[DataLoader, DataLoader]:
    train_dataset = BrodenConceptDataset(
        broden_path=config.broden_path,
        concept_names=concept_names,
        split="train",
        config=config,
        max_per_concept=config.max_per_concept,
    )
    val_dataset = BrodenConceptDataset(
        broden_path=config.broden_path,
        concept_names=concept_names,
        split="val",
        config=config,
        max_per_concept=config.max_per_concept,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
