"""Matched single-view and multi-view CST models for the MSD A/P cut plane."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from ..data import uniform_positions
from ..descriptors import labels_to_probabilities, soft_descriptors
from ..model import CSTEncoder2d


VIEW_AXIS = {"sagittal": 0, "coronal": 1, "axial": 2}
VIEW_INDEX = {"sagittal": 0, "coronal": 1, "axial": 2}
VIEW_SPECS: dict[str, dict[str, int]] = {
    "coronal32": {"coronal": 32},
    "sagittal32": {"sagittal": 32},
    "axial32": {"axial": 32},
    "triview32": {"sagittal": 11, "coronal": 11, "axial": 10},
    "triview96": {"sagittal": 32, "coronal": 32, "axial": 32},
}


def best_cut_from_labels(label: np.ndarray) -> tuple[int, bool, int]:
    """Return best coronal cut, anterior-high orientation, and wrong-side count."""
    label = np.asarray(label)
    if label.ndim != 3 or not np.isin(label, (0, 1, 2)).all():
        raise ValueError("expected an (X,Y,Z) label with values 0,1,2")
    anterior = (label == 1).sum(axis=(0, 2))
    posterior = (label == 2).sum(axis=(0, 2))
    if anterior.sum() == 0 or posterior.sum() == 0:
        raise ValueError("both A/P classes are required to define the cut")
    position = np.arange(label.shape[1])
    high = float((anterior * position).sum() / anterior.sum()) > float((posterior * position).sum() / posterior.sum())
    anterior_below = np.cumsum(anterior)
    posterior_below = np.cumsum(posterior)
    costs = (anterior_below + posterior.sum() - posterior_below) if high else (
        posterior_below + anterior.sum() - anterior_below
    )
    cut = int(np.argmin(costs[:-1]))
    return cut, high, int(costs[cut])


def extract_view_slabs(
    image: torch.Tensor,
    view: str,
    positions: torch.Tensor,
    *,
    slab_depth: int = 3,
    output_size: int = 32,
) -> torch.Tensor:
    """Return [N, C*depth, H, W] slabs along one anatomical axis."""
    if image.ndim != 4 or view not in VIEW_AXIS or slab_depth < 1 or slab_depth % 2 != 1:
        raise ValueError("expected (C,X,Y,Z), valid view, and odd positive slab depth")
    axis = VIEW_AXIS[view] + 1
    positions = torch.as_tensor(positions, dtype=torch.long, device=image.device)
    if positions.ndim != 1 or positions.min() < 0 or positions.max() >= image.shape[axis]:
        raise ValueError("slice positions are outside the chosen view axis")
    radius = slab_depth // 2
    slabs = []
    for position in positions.tolist():
        planes = [image.select(axis, max(0, min(image.shape[axis] - 1, position + offset)))
                  for offset in range(-radius, radius + 1)]
        slabs.append(torch.cat(planes, dim=0))
    result = torch.stack(slabs)
    if result.shape[-2:] != (output_size, output_size):
        result = F.interpolate(result, size=(output_size, output_size), mode="bilinear", align_corners=False)
    return result


def make_view_set(image: torch.Tensor, specification: Mapping[str, int], *, output_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    slabs, metadata = [], []
    for view, count in specification.items():
        axis_size = image.shape[VIEW_AXIS[view] + 1]
        positions = uniform_positions(axis_size, count)
        slabs.append(extract_view_slabs(image, view, positions, output_size=output_size))
        normalized = 2.0 * positions.float() / (axis_size - 1) - 1.0
        view_id = F.one_hot(torch.full((count,), VIEW_INDEX[view]), num_classes=3).float()
        metadata.append(torch.cat((normalized[:, None], view_id), dim=1))
    return torch.cat(slabs), torch.cat(metadata)


def make_view_profiles(label: torch.Tensor, specification: Mapping[str, int]) -> torch.Tensor:
    """A/P area profiles, normalized separately within each anatomical view."""
    if label.ndim != 4 or label.shape[0] != 1:
        raise ValueError("label must have shape (1,X,Y,Z)")
    probabilities = labels_to_probabilities(label.unsqueeze(0))[0]
    return make_probability_view_profiles(probabilities, specification)


def make_probability_view_profiles(probabilities: torch.Tensor, specification: Mapping[str, int]) -> torch.Tensor:
    """Per-view normalized A/P profiles for hard or soft 3-class volumes."""
    if probabilities.ndim != 4 or probabilities.shape[0] != 3:
        raise ValueError("probabilities must have shape (3,X,Y,Z)")
    blocks = []
    for view, count in specification.items():
        axis = VIEW_AXIS[view] + 1
        positions = uniform_positions(probabilities.shape[axis], count).to(probabilities.device)
        selected = probabilities[1:3].index_select(axis, positions)
        other_axes = tuple(index for index in (1, 2, 3) if index != axis)
        areas = selected.sum(dim=other_axes).T
        scale = areas.sum(dim=1).max().clamp_min(1e-6)
        blocks.append(areas / scale)
    return torch.cat(blocks)


class CutSetDataset(Dataset):
    """Cache deterministic view slabs and cut targets from preprocessed cases."""

    def __init__(self, base_dataset: Dataset, specification: Mapping[str, int], *, output_size: int = 32):
        self.specification = dict(specification)
        self.items = []
        for index in range(len(base_dataset)):
            item = base_dataset[index]
            image = torch.as_tensor(item["image"], dtype=torch.float32)
            label = np.asarray(item["label"])[0].astype(np.int8)
            if image.shape[1:] != label.shape:
                raise ValueError("image and label must share spatial shape")
            cut, high, wrong = best_cut_from_labels(label)
            if not high:
                raise ValueError("MSD cut target unexpectedly reverses A/P direction")
            slabs, metadata = make_view_set(image, self.specification, output_size=output_size)
            foreground_y = np.flatnonzero((label > 0).any(axis=(0, 2)))
            self.items.append({
                "case_name": str(item["case_name"]),
                "slabs": slabs,
                "metadata": metadata,
                "cut_target": torch.tensor((cut + 0.5) / label.shape[1], dtype=torch.float32),
                "cut_y": cut,
                "wrong_side_voxels": wrong,
                "foreground_y_min": int(foreground_y[0]),
                "foreground_y_max": int(foreground_y[-1]),
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


class ProfileSetDataset(Dataset):
    """Cache matched view sets, case descriptors, and per-view area profiles."""

    def __init__(self, base_dataset: Dataset, specification: Mapping[str, int], *, output_size: int = 32):
        self.specification = dict(specification)
        self.items = []
        for index in range(len(base_dataset)):
            item = base_dataset[index]
            image = torch.as_tensor(item["image"], dtype=torch.float32)
            label = torch.as_tensor(item["label"]).long()
            slabs, metadata = make_view_set(image, self.specification, output_size=output_size)
            probabilities = labels_to_probabilities(label.unsqueeze(0))
            self.items.append({
                "case_name": str(item["case_name"]),
                "slabs": slabs,
                "metadata": metadata,
                "descriptors": soft_descriptors(probabilities)[0],
                "profiles": make_view_profiles(label, self.specification),
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


class CSTCutRegressor(nn.Module):
    """One shared set encoder for all view variants; predict one coronal cut."""

    def __init__(self, channels: tuple[int, ...] = (16, 32, 64), heads: int = 4):
        super().__init__()
        self.encoder = CSTEncoder2d(3, metadata_dim=4, channels=channels, heads=heads)
        self.head = nn.Sequential(nn.Linear(channels[-1], channels[-1]), nn.GELU(), nn.Linear(channels[-1], 1))

    def forward(self, slabs: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        _, case_embedding = self.encoder(slabs, metadata)
        return self.head(case_embedding).squeeze(-1).sigmoid()
