"""Mask-conditioned cross-view CST for ranking nearby A/P cut candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from ..data import uniform_positions
from ..model import CSTEncoder2d
from .cut_model import VIEW_AXIS, best_cut_from_labels, make_view_set


OFFSETS = (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)


def make_candidate_mask_slices(
    union: torch.Tensor,
    cut_y: int,
    specification: Mapping[str, int],
    *,
    output_size: int = 32,
) -> torch.Tensor:
    """Sample candidate A/P masks in each view, preserving the supplied union."""
    if union.ndim != 3 or not 0 <= cut_y < union.shape[1] - 1:
        raise ValueError("union must be (X,Y,Z) with an interior coronal cut")
    y = torch.arange(union.shape[1], device=union.device)[None, :, None]
    anterior = union.bool() & (y > cut_y)
    posterior = union.bool() & (y <= cut_y)
    mask = torch.stack((anterior, posterior)).float()
    blocks = []
    for view, count in specification.items():
        axis = VIEW_AXIS[view] + 1
        positions = uniform_positions(union.shape[VIEW_AXIS[view]], count).to(union.device)
        selected = mask.index_select(axis, positions)
        if view == "sagittal":
            selected = selected.permute(1, 0, 2, 3)
        elif view == "coronal":
            selected = selected.permute(2, 0, 1, 3)
        else:
            selected = selected.permute(3, 0, 1, 2)
        if selected.shape[-2:] != (output_size, output_size):
            selected = F.interpolate(selected, size=(output_size, output_size), mode="nearest")
        blocks.append(selected)
    return torch.cat(blocks)


class CandidateSetDataset(Dataset):
    """Cache MRI view sets and ground-truth unions for synthetic cut ranking."""

    def __init__(self, base_dataset: Dataset, specification: Mapping[str, int], *, output_size: int = 32):
        self.specification = dict(specification)
        self.output_size = output_size
        self.items = []
        for index in range(len(base_dataset)):
            item = base_dataset[index]
            image = torch.as_tensor(item["image"], dtype=torch.float32)
            label = np.asarray(item["label"])[0].astype(np.int8)
            slabs, metadata = make_view_set(image, self.specification, output_size=output_size)
            cut, high, _ = best_cut_from_labels(label)
            if not high:
                raise ValueError("unexpected A/P orientation")
            self.items.append({
                "case_name": str(item["case_name"]),
                "image_slabs": slabs,
                "metadata": metadata,
                "union": torch.from_numpy(label > 0),
                "true_cut_y": cut,
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


def build_candidate_batch(batch: dict, candidate_cuts: np.ndarray, specification: Mapping[str, int], *, output_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand B cases into B×K MRI/mask sets with candidate-cut metadata."""
    candidate_cuts = np.asarray(candidate_cuts, dtype=np.int64)
    if candidate_cuts.ndim != 2 or candidate_cuts.shape[0] != len(batch["case_name"]):
        raise ValueError("candidate cuts must have shape (batch,candidates)")
    sets, metadata = [], []
    for patient in range(len(batch["case_name"])):
        union = batch["union"][patient]
        image_slabs = batch["image_slabs"][patient]
        view_metadata = batch["metadata"][patient]
        for cut in candidate_cuts[patient]:
            masks = make_candidate_mask_slices(union, int(cut), specification, output_size=output_size)
            if masks.shape[0] != image_slabs.shape[0]:
                raise ValueError("image and mask view tokens do not align")
            sets.append(torch.cat((image_slabs, masks), dim=1))
            normalized_cut = 2.0 * (float(cut) + 0.5) / union.shape[1] - 1.0
            metadata.append(torch.cat((view_metadata, torch.full((len(view_metadata), 1), normalized_cut)), dim=1))
    return torch.stack(sets), torch.stack(metadata)


class CSTCandidateRanker(nn.Module):
    """Shared-capacity set encoder; scalar anatomical plausibility per cut."""

    def __init__(self, channels: tuple[int, ...] = (16, 32, 64), heads: int = 4):
        super().__init__()
        self.encoder = CSTEncoder2d(5, metadata_dim=5, channels=channels, heads=heads)
        self.head = nn.Sequential(nn.Linear(channels[-1], channels[-1]), nn.GELU(), nn.Linear(channels[-1], 1))

    def forward(self, elements: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        _, embedding = self.encoder(elements, metadata)
        return self.head(embedding).squeeze(-1)


def candidate_cuts_around(centers: Sequence[int], y_size: int, offsets: Sequence[int] = OFFSETS) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.int64)
    if centers.ndim != 1:
        raise ValueError("centers must be one-dimensional")
    return np.clip(centers[:, None] + np.asarray(offsets)[None], 0, y_size - 2)
