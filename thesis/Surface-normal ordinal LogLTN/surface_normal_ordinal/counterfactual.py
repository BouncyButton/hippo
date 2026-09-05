"""Matched-magnitude frozen-logit repair and gradient diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from .geometry import SurfaceSamples
from .metrics import segmentation_metrics
from .objective import onecut_logltn_loss, ordinal_logltn_loss, semantic_field


@dataclass(frozen=True)
class CounterfactualResult:
    metrics_by_arm: dict[str, dict[str, float | int]]
    diagnostics: dict[str, float]


def soft_multiclass_dice(logits: torch.Tensor, labels: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """MONAI-like mean soft Dice over all three classes for one case."""

    probabilities = torch.softmax(logits.float(), dim=0)
    one_hot = F.one_hot(labels.long(), num_classes=3).movedim(-1, 0).float()
    spatial = (1, 2, 3)
    intersection = (probabilities * one_hot).sum(spatial)
    denominator = probabilities.sum(spatial) + one_hot.sum(spatial)
    return 1.0 - ((2.0 * intersection + epsilon) / (denominator + epsilon)).mean()


def grouped_boundary_bce(logits: torch.Tensor, labels: torch.Tensor, steps: int = 2) -> torch.Tensor:
    """Existing side-balanced two-step band BCE, used only as a comparator."""

    foreground = labels.detach().cpu().numpy() > 0
    structure = ndimage.generate_binary_structure(3, 1)
    eroded = ndimage.binary_erosion(foreground, structure=structure, iterations=steps)
    dilated = ndimage.binary_dilation(foreground, structure=structure, iterations=steps)
    inner = torch.as_tensor(foreground & ~eroded, device=logits.device)
    outer = torch.as_tensor(dilated & ~foreground, device=logits.device)
    if not bool(inner.any()) or not bool(outer.any()):
        raise ValueError("Both inner and outer comparison bands must be nonempty.")
    field = semantic_field(logits, "outer")
    inner_loss = F.softplus(-field[inner]).mean()
    outer_loss = F.softplus(field[outer]).mean()
    return 0.5 * (inner_loss + outer_loss)


def _rms(value: torch.Tensor) -> float:
    return float(value.float().square().mean().sqrt().detach().cpu())


def _cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    numerator = float((first.float() * second.float()).sum().detach().cpu())
    denominator = float(first.float().norm().detach().cpu()) * float(
        second.float().norm().detach().cpu()
    )
    return numerator / denominator if denominator > 0 else float("nan")


def _matched_update(logits: torch.Tensor, gradient: torch.Tensor, target_rms: float) -> torch.Tensor:
    gradient_rms = _rms(gradient)
    if gradient_rms <= 0 or not math.isfinite(gradient_rms):
        return logits.detach().clone()
    return (logits - target_rms * gradient / gradient_rms).detach()


def _pair_coordinates(
    samples: SurfaceSamples, delta_mm: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    coordinates = torch.as_tensor(
        samples.coordinates_at([-delta_mm, delta_mm]),
        dtype=torch.float32,
        device=device,
    )
    return coordinates[:, 0], coordinates[:, 1]


def _ray_coordinates(
    samples: SurfaceSamples, offsets_mm: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    coordinates = torch.as_tensor(
        samples.coordinates_at(offsets_mm), dtype=torch.float32, device=device
    )
    offsets = torch.as_tensor(offsets_mm, dtype=torch.float32, device=device)
    return coordinates, offsets


def run_counterfactual(
    logits: torch.Tensor,
    labels: np.ndarray,
    spacing: tuple[float, float, float],
    outer_samples: SurfaceSamples,
    *,
    ap_samples: SurfaceSamples | None = None,
    offsets_mm: np.ndarray | None = None,
    delta_mm: float = 1.0,
    cut_tolerance_mm: float = 1.0,
    margin: float = 0.0,
    temperature: float = 1.0,
    target_aux_gradient_ratio: float = 0.10,
    update_rms: float = 0.05,
) -> CounterfactualResult:
    """Compare equal-RMS local logit updates without changing model parameters."""

    if update_rms <= 0 or target_aux_gradient_ratio <= 0:
        raise ValueError("update_rms and target_aux_gradient_ratio must be positive.")
    z = logits.detach().float().clone().requires_grad_(True)
    labels_t = torch.as_tensor(labels, device=z.device, dtype=torch.long)
    losses: dict[str, torch.Tensor] = {"dice": soft_multiclass_dice(z, labels_t)}
    inside, outside = _pair_coordinates(outer_samples, delta_mm, z.device)
    losses["ordinal_outer"] = ordinal_logltn_loss(
        z,
        inside,
        outside,
        interface="outer",
        margin=margin,
        temperature=temperature,
    )[0]
    if offsets_mm is None:
        offsets_mm = np.arange(-3.0, 3.01, 0.5)
    outer_rays, offsets_t = _ray_coordinates(outer_samples, offsets_mm, z.device)
    losses["onecut_outer"] = onecut_logltn_loss(
        z,
        outer_rays,
        offsets_t,
        interface="outer",
        tolerance_mm=cut_tolerance_mm,
        margin=margin,
        temperature=temperature,
    )[0]
    losses["bands"] = grouped_boundary_bce(z, labels_t)
    if ap_samples is not None:
        ap_inside, ap_outside = _pair_coordinates(ap_samples, delta_mm, z.device)
        losses["ordinal_ap"] = ordinal_logltn_loss(
            z,
            ap_inside,
            ap_outside,
            interface="ap",
            margin=margin,
            temperature=temperature,
        )[0]
        ap_rays, _ = _ray_coordinates(ap_samples, offsets_mm, z.device)
        losses["onecut_ap"] = onecut_logltn_loss(
            z,
            ap_rays,
            offsets_t,
            interface="ap",
            tolerance_mm=cut_tolerance_mm,
            margin=margin,
            temperature=temperature,
        )[0]
    gradients = {
        name: torch.autograd.grad(loss, z, retain_graph=True)[0].float()
        for name, loss in losses.items()
    }
    dice_rms = _rms(gradients["dice"])
    diagnostics: dict[str, float] = {
        "dice_loss": float(losses["dice"].detach().cpu()),
        "dice_gradient_rms": dice_rms,
    }
    for name in losses:
        diagnostics[f"{name}_loss"] = float(losses[name].detach().cpu())
        diagnostics[f"{name}_gradient_rms"] = _rms(gradients[name])
        if name != "dice":
            diagnostics[f"{name}_dice_gradient_cosine"] = _cosine(
                gradients[name], gradients["dice"]
            )

    updated: dict[str, torch.Tensor] = {"baseline": z.detach()}
    updated["dice_only"] = _matched_update(z, gradients["dice"], update_rms)
    for name in losses:
        if name == "dice":
            continue
        aux_rms = _rms(gradients[name])
        weight = target_aux_gradient_ratio * dice_rms / aux_rms if aux_rms > 0 else 0.0
        diagnostics[f"{name}_matched_weight"] = float(weight)
        updated[f"{name}_only"] = _matched_update(z, gradients[name], update_rms)
        combined = gradients["dice"] + weight * gradients[name]
        updated[f"dice_plus_{name}"] = _matched_update(z, combined, update_rms)

    metrics_by_arm: dict[str, dict[str, float | int]] = {}
    for arm, repaired_logits in updated.items():
        prediction = repaired_logits.argmax(dim=0).cpu().numpy().astype(np.uint8)
        metrics_by_arm[arm] = segmentation_metrics(labels, prediction, spacing)
    return CounterfactualResult(metrics_by_arm, diagnostics)
