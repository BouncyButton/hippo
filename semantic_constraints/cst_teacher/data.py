"""Set construction and synthetic corruptions for the CST teacher."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .descriptors import labels_to_probabilities, sampled_slice_profiles, soft_descriptors


def uniform_positions(axis_size: int, set_size: int) -> torch.Tensor:
    """Return unique, uniformly spaced integer locations on an axis."""

    if axis_size < 1 or set_size < 1:
        raise ValueError("axis_size and set_size must be positive")
    if set_size > axis_size:
        raise ValueError("set_size cannot exceed the sampled axis size")
    positions = torch.linspace(0, axis_size - 1, set_size).round().long()
    if positions.unique().numel() != set_size:
        raise AssertionError("uniform position rounding produced duplicates")
    return positions


def extract_coronal_slabs(
    image: torch.Tensor,
    positions: torch.Tensor | Sequence[int],
    *,
    slab_depth: int = 3,
    output_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Extract ``[N, C*slab_depth, X, Z]`` slabs from ``[C, X, Y, Z]``."""

    if image.ndim != 4:
        raise ValueError("image must have shape [C, X, Y, Z]")
    if slab_depth < 1 or slab_depth % 2 != 1:
        raise ValueError("slab_depth must be a positive odd number")
    positions = torch.as_tensor(positions, device=image.device, dtype=torch.long)
    if positions.ndim != 1 or positions.numel() == 0:
        raise ValueError("positions must be a non-empty vector")
    if positions.min() < 0 or positions.max() >= image.shape[2]:
        raise ValueError("position outside image Y axis")

    radius = slab_depth // 2
    padded = F.pad(image, (0, 0, radius, radius, 0, 0), mode="replicate")
    slabs = []
    for position in positions.tolist():
        planes = [padded[:, :, position + offset, :] for offset in range(slab_depth)]
        slabs.append(torch.cat(planes, dim=0))
    result = torch.stack(slabs, dim=0)
    if output_size is not None and result.shape[-2:] != output_size:
        result = F.interpolate(result, size=output_size, mode="bilinear", align_corners=False)
    return result


def extract_mask_slices(
    probabilities: torch.Tensor,
    positions: torch.Tensor | Sequence[int],
    *,
    output_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Extract anterior/posterior slices as ``[N, 2, X, Z]``."""

    if probabilities.ndim != 4 or probabilities.shape[0] < 3:
        raise ValueError("probabilities must have shape [>=3, X, Y, Z]")
    positions = torch.as_tensor(positions, device=probabilities.device, dtype=torch.long)
    result = probabilities[1:3].index_select(2, positions).permute(2, 0, 1, 3).contiguous()
    if output_size is not None and result.shape[-2:] != output_size:
        result = F.interpolate(result, size=output_size, mode="nearest")
    return result


def element_metadata(positions: torch.Tensor, axis_size: int) -> torch.Tensor:
    """Normalized physical-order metadata, shaped ``[N, 1]``."""

    if axis_size < 2:
        raise ValueError("axis_size must be at least two")
    return (2.0 * positions.float() / (axis_size - 1) - 1.0).unsqueeze(1)


class CSTSetDataset(Dataset):
    """Adapt the project's preprocessed MONAI cases to fixed-cardinality sets."""

    def __init__(
        self,
        base_dataset: Dataset,
        *,
        set_size: int = 12,
        slab_depth: int = 3,
        inplane_size: int | None = 32,
        cache: bool = False,
    ) -> None:
        self.base_dataset = base_dataset
        self.set_size = set_size
        self.slab_depth = slab_depth
        self.output_size = None if inplane_size is None else (inplane_size, inplane_size)
        self._cache = [self._build_item(index) for index in range(len(base_dataset))] if cache else None

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        if self._cache is not None:
            return self._cache[index]
        return self._build_item(index)

    def _build_item(self, index: int) -> dict[str, torch.Tensor | str]:
        item = self.base_dataset[index]
        image = torch.as_tensor(item["image"], dtype=torch.float32)
        label = torch.as_tensor(item["label"])
        if image.ndim != 4 or label.ndim != 4:
            raise ValueError("preprocessed cases must have shape [C, X, Y, Z]")
        positions = uniform_positions(image.shape[2], self.set_size)
        probabilities = labels_to_probabilities(label.unsqueeze(0))[0]
        return {
            "case_name": str(item.get("case_name", index)),
            "image_slabs": extract_coronal_slabs(
                image,
                positions,
                slab_depth=self.slab_depth,
                output_size=self.output_size,
            ),
            "mask_slices": extract_mask_slices(
                probabilities,
                positions,
                output_size=self.output_size,
            ),
            "metadata": element_metadata(positions, image.shape[2]),
            "positions": positions,
            "descriptors": soft_descriptors(probabilities.unsqueeze(0))[0],
            "slice_profiles": sampled_slice_profiles(probabilities.unsqueeze(0), positions)[0],
            "valid_elements": torch.ones(self.set_size, dtype=torch.bool),
        }


def random_valid_elements(
    batch_size: int,
    set_size: int,
    *,
    minimum: int = 4,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Combinatorial-training mask with a different cardinality per case."""

    if not 1 <= minimum <= set_size:
        raise ValueError("minimum must be in [1, set_size]")
    valid = torch.zeros(batch_size, set_size, dtype=torch.bool, device=device)
    for batch_index in range(batch_size):
        cardinality = int(torch.randint(minimum, set_size + 1, ()).item())
        selected = torch.randperm(set_size, device=device)[:cardinality]
        valid[batch_index, selected] = True
    return valid


def corrupt_mask_elements(
    masks: torch.Tensor,
    *,
    valid_elements: torch.Tensor | None = None,
    corruption_rate: float = 0.35,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create contextual anomalies in ``[B, N, 2, H, W]`` mask sets.

    Corruptions include A/P swaps, translations, local erasure, and isolated
    foreground islands.  The Boolean target is true exactly for altered set
    elements.  At least one valid element per case is corrupted.
    """

    if masks.ndim != 5 or masks.shape[2] != 2:
        raise ValueError("masks must have shape [B, N, 2, H, W]")
    if not 0.0 < corruption_rate <= 1.0:
        raise ValueError("corruption_rate must lie in (0, 1]")
    batch, set_size, _, height, width = masks.shape
    if valid_elements is None:
        valid_elements = torch.ones(batch, set_size, dtype=torch.bool, device=masks.device)
    if valid_elements.shape != (batch, set_size):
        raise ValueError("valid_elements must have shape [B, N]")

    corrupted = masks.clone()
    anomaly = torch.zeros(batch, set_size, dtype=torch.bool, device=masks.device)
    for batch_index in range(batch):
        eligible = torch.nonzero(valid_elements[batch_index], as_tuple=False).flatten()
        if eligible.numel() == 0:
            continue
        selected = eligible[torch.rand(eligible.numel(), device=masks.device) < corruption_rate]
        if selected.numel() == 0:
            selected = eligible[torch.randint(eligible.numel(), ())].reshape(1)
        for element_index in selected.tolist():
            operation = int(torch.randint(4, ()).item())
            original = corrupted[batch_index, element_index]
            current = original
            if operation == 0:
                current = current.flip(0)
            elif operation == 1:
                shift_x = int(torch.randint(-4, 5, ()).item())
                shift_y = int(torch.randint(-4, 5, ()).item())
                current = torch.roll(current, shifts=(shift_x, shift_y), dims=(-2, -1))
            elif operation == 2:
                patch_h = max(2, height // 8)
                patch_w = max(2, width // 8)
                start_h = int(torch.randint(max(1, height - patch_h + 1), ()).item())
                start_w = int(torch.randint(max(1, width - patch_w + 1), ()).item())
                current = current.clone()
                current[:, start_h : start_h + patch_h, start_w : start_w + patch_w] = 0
            else:
                current = current.clone()
                foreground_class = int(torch.randint(2, ()).item())
                center_h = int(torch.randint(height, ()).item())
                center_w = int(torch.randint(width, ()).item())
                radius = max(1, min(height, width) // 32)
                h0, h1 = max(0, center_h - radius), min(height, center_h + radius + 1)
                w0, w1 = max(0, center_w - radius), min(width, center_w + radius + 1)
                current[:, h0:h1, w0:w1] = 0
                current[foreground_class, h0:h1, w0:w1] = 1
            if torch.equal(current, original):
                current = current.clone()
                current[0, 0, 0] = torch.where(
                    current[0, 0, 0] < 0.5,
                    current.new_tensor(1.0),
                    current.new_tensor(0.0),
                )
            corrupted[batch_index, element_index] = current
            anomaly[batch_index, element_index] = True
    return corrupted.clamp(0.0, 1.0), anomaly
