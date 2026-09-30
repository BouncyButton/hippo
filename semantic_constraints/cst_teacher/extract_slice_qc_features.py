#!/usr/bin/env python3
"""Extract CST/Swin slice-QC features without using a reference mask.

The input MRI must already have passed the same 64-cube nonzero z-score and
center pad/crop preprocessing used by Swin and CST training. The probability
array must be the matching Swin softmax output in (C, X, Y, Z) order.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .data import element_metadata, extract_coronal_slabs, uniform_positions
from .descriptors import sampled_slice_profiles, soft_descriptors
from .evaluate_predictions import load_teachers
from .risk_probe import entropy_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--preprocessed-image-npy", type=Path, required=True)
    parser.add_argument("--swin-probabilities-npy", type=Path, required=True)
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


@torch.no_grad()
def extract_arrays(
    teacher,
    image: np.ndarray,
    probabilities: np.ndarray,
    *,
    set_size: int,
    slab_depth: int,
    inplane_size: int,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    image = np.asarray(image, dtype=np.float32)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if image.ndim == 3:
        image = image[None]
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("preprocessed MRI must have shape (1,X,Y,Z) or (X,Y,Z)")
    if probabilities.shape != (3, *image.shape[1:]):
        raise ValueError("Swin probabilities must have shape (3,X,Y,Z) matching the MRI")
    if not np.isfinite(image).all() or not np.isfinite(probabilities).all():
        raise ValueError("MRI and probabilities must be finite")
    if probabilities.min() < -1e-5 or probabilities.max() > 1 + 1e-5:
        raise ValueError("Swin probabilities must lie in [0,1]")
    if not np.allclose(probabilities.sum(axis=0), 1.0, atol=1e-3):
        raise ValueError("Swin class probabilities must sum to one at each voxel")
    if image.shape[2] < set_size:
        raise ValueError("coronal axis is shorter than CST set size")

    # CSTSetDataset constructs and bilinearly resizes slabs on CPU, then the
    # risk probe moves completed slabs to the model device. Keep that order:
    # CPU/GPU interpolation differences can perturb learned residual features.
    volume = torch.from_numpy(image)
    probabilities_tensor = torch.from_numpy(probabilities).unsqueeze(0).to(device)
    positions_cpu = uniform_positions(image.shape[2], set_size)
    metadata = element_metadata(positions_cpu, image.shape[2]).unsqueeze(0).to(device)
    slabs = extract_coronal_slabs(
        volume,
        positions_cpu,
        slab_depth=slab_depth,
        output_size=(inplane_size, inplane_size),
    ).unsqueeze(0).to(device)
    positions = positions_cpu.to(device)
    output = teacher(slabs, metadata, torch.ones(1, set_size, dtype=torch.bool, device=device))
    prediction_profiles = sampled_slice_profiles(probabilities_tensor, positions)
    residual = output.slice_profiles - prediction_profiles
    descriptors = soft_descriptors(probabilities_tensor)
    lower = output.descriptor_quantiles[..., 0]
    upper = output.descriptor_quantiles[..., 2]
    scale = ((upper - lower) / 2).clamp_min(0.02)
    descriptor_violation = (torch.relu(lower - descriptors) + torch.relu(descriptors - upper)) / scale
    case_uncertainty, slice_uncertainty = entropy_features(probabilities_tensor, positions)
    slice_uncertainty_block = torch.cat(
        (slice_uncertainty.unsqueeze(-1), prediction_profiles, metadata), dim=2
    )
    slice_relationship = torch.cat(
        (residual, residual.abs(), prediction_profiles, output.slice_profiles, metadata), dim=2
    )
    slice_combined = torch.cat(
        (slice_uncertainty_block, output.element_embeddings, slice_relationship), dim=2
    )
    case_relationship = torch.cat(
        (residual.flatten(start_dim=1), residual.abs().flatten(start_dim=1), descriptor_violation, descriptors),
        dim=1,
    )
    case_combined = torch.cat((case_uncertainty, output.case_embedding, case_relationship), dim=1)
    return {
        "slice_uncertainty": slice_uncertainty_block[0].cpu().numpy(),
        "slice_relationship_residuals": slice_relationship[0].cpu().numpy(),
        "slice_combined": slice_combined[0].cpu().numpy(),
        "case_uncertainty": case_uncertainty[0].cpu().numpy(),
        "case_relationship_residuals": case_relationship[0].cpu().numpy(),
        "case_combined": case_combined[0].cpu().numpy(),
        "positions": positions.cpu().numpy(),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    teacher, _, checkpoint = load_teachers(args.teacher_checkpoint, device)
    config = checkpoint["config"]
    image = np.load(args.preprocessed_image_npy, allow_pickle=False)
    probabilities = np.load(args.swin_probabilities_npy, allow_pickle=False)
    arrays = extract_arrays(
        teacher,
        image,
        probabilities,
        set_size=int(config["set_size"]),
        slab_depth=int(config["slab_depth"]),
        inplane_size=int(config.get("inplane_size", 32)),
        device=args.device,
    )
    slices = len(arrays["positions"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        case_names=np.asarray([args.case_name]),
        slice_groups=np.asarray([args.case_name] * slices),
        teacher_checkpoint=str(args.teacher_checkpoint),
        **arrays,
    )
    print(f"Extracted {slices} label-free CST/Swin slice features to {args.output}")


if __name__ == "__main__":
    main()
