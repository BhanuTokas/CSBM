from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F

log = logging.getLogger(__name__)

MASK_CONCEPTS = {"color", "object", "material", "part"}
WEAK_CONCEPTS = {"scene", "texture"}
ALL_CONCEPTS = ["color", "object", "material", "part", "scene", "texture"]


# ---------------------------------------------------------------------------
# Concept discovery
# ---------------------------------------------------------------------------

PREFERRED_ORDER = ["object", "part", "material", "color", "scene", "texture"]


def get_concept_names(broden_path: str, num_concepts: int) -> list[str]:
    broden_path = Path(broden_path)
    concept_csvs = sorted(broden_path.glob("c_*.csv"))
    all_concepts = [f.stem[2:] for f in concept_csvs]  # strip 'c_' prefix

    if not all_concepts:
        raise ValueError(
            f"No concept CSVs found in {broden_path}. "
            "Verify your Broden installation."
        )

    available = set(all_concepts)
    ordered = [c for c in PREFERRED_ORDER if c in available]
    ordered += sorted(c for c in all_concepts if c not in set(ordered))

    if num_concepts > len(ordered):
        log.warning(
            f"--num_concepts {num_concepts} exceeds available "
            f"{len(ordered)}; using all {len(ordered)}."
        )

    selected = ordered[:num_concepts]
    log.info(
        f"Discovered {len(all_concepts)} concepts; using {len(selected)}: {selected}"
    )
    return selected


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class BrodenConceptDataset(Dataset):
    def __init__(
        self,
        broden_path: str | Path,
        concept_names: list[str],
        split: Literal["train", "val"],
        image_size: int = 224,
        mask_size: int = 16,
        max_per_concept: int = 5000,
    ):
        self.broden_path = Path(broden_path)
        self.concept_names = concept_names
        self.split = split
        self.mask_size = mask_size
        self.max_per_concept = max_per_concept

        self.img_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Load dense per-concept label codes once
        self.concept_codes: dict[str, set[int]] = {}
        for c in concept_names:
            if c in MASK_CONCEPTS:
                csv = self.broden_path / f"c_{c}.csv"
                df = pd.read_csv(csv)
                self.concept_codes[c] = set(df["code"].tolist())

        self.samples = self._build_samples()
        log.info(
            f"[{split}] {len(self.samples)} samples "
            f"across {len(concept_names)} concepts"
        )

    # ------------------------------------------------------------------
    def _build_samples(self) -> list[dict]:
        idx = pd.read_csv(self.broden_path / "index.csv")
        idx = idx[idx["split"] == self.split].reset_index(drop=True)

        per_concept_count: dict[str, int] = {c: 0 for c in self.concept_names}
        samples = []

        for _, row in idx.iterrows():
            concept_labels: dict[str, dict] = {}

            for c in self.concept_names:
                cell = row.get(c, "")
                if pd.isna(cell) or str(cell).strip() == "":
                    continue
                if per_concept_count[c] >= self.max_per_concept:
                    continue

                cell = str(cell).strip()

                if c in MASK_CONCEPTS:
                    # Handle semicolon-separated multiple mask paths
                    paths = [
                        self.broden_path / "images" / p.strip() for p in cell.split(";")
                    ]
                    concept_labels[c] = {
                        "type": "mask",
                        "paths": paths,  # list, usually length 1
                    }
                else:  # WEAK_CONCEPTS
                    # May be semicolon-separated; take first (dominant) label
                    label_ids = [int(float(x)) for x in cell.split(";")]
                    concept_labels[c] = {
                        "type": "weak",
                        "label_ids": label_ids,
                    }

            if not concept_labels:
                continue

            for c in concept_labels:
                per_concept_count[c] += 1

            samples.append(
                {
                    "img_path": self.broden_path / "images" / row["image"],
                    "concept_labels": concept_labels,
                }
            )

        return samples

    # ------------------------------------------------------------------
    def _load_seg_mask(self, path: Path) -> np.ndarray:
        """Returns (H, W) int32 array of label codes (R + 256*G)."""
        seg = np.array(Image.open(path).convert("RGB"))
        return seg[:, :, 0].astype(np.int32) + 256 * seg[:, :, 1].astype(np.int32)

    def _mask_to_binary(self, paths: list[Path], concept: str) -> torch.Tensor:
        """Load one or more seg masks, binarize each, OR together → (1, mask_size, mask_size)."""
        codes = self.concept_codes[concept]
        combined = None

        for path in paths:
            label_map = self._load_seg_mask(path)
            binary = np.isin(label_map, list(codes)).astype(np.float32)
            if combined is None:
                combined = binary
            else:
                combined = np.maximum(combined, binary)  # OR

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
        image = self.img_transform(image)  # (3, H, W)

        # target: (num_concepts, mask_size, mask_size)  — zeros where no label
        target = torch.zeros(len(self.concept_names), self.mask_size, self.mask_size)
        valid_mask = torch.zeros(len(self.concept_names), dtype=torch.bool)

        for ci, c in enumerate(self.concept_names):
            entry = sample["concept_labels"].get(c)
            if entry is None:
                continue

            if entry["type"] == "mask":
                target[ci] = self._mask_to_binary(entry["paths"], c)

            else:  # weak
                # Broadcast: mark entire spatial map as positive
                target[ci] = 1.0

            valid_mask[ci] = True

        return {"image": image, "target": target, "valid": valid_mask}


# ---------------------------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------------------------


def build_dataloaders(config, concept_names: list[str]):
    common = dict(
        broden_path=config.broden_path,
        concept_names=concept_names,
        image_size=config.image_size,
        mask_size=config.mask_size,
        max_per_concept=config.max_per_concept,
    )
    train_dataset = BrodenConceptDataset(**common, split="train")
    val_dataset = BrodenConceptDataset(**common, split="val")

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
