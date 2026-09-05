import numpy as np
import torch

from surface_normal_ordinal.counterfactual import run_counterfactual
from surface_normal_ordinal.geometry import Interface, build_surface_samples
from surface_normal_ordinal.metrics import binary_dice, segmentation_metrics


def _case() -> tuple[np.ndarray, torch.Tensor]:
    labels = np.zeros((15, 15, 15), dtype=np.uint8)
    labels[4:11, 4:11, 4:7] = 1
    labels[4:11, 4:11, 7:11] = 2
    logits = torch.zeros(3, *labels.shape)
    logits[0] = 1.0
    logits[1][torch.from_numpy(labels == 1)] = 2.0
    logits[2][torch.from_numpy(labels == 2)] = 2.0
    return labels, logits


def test_spacing_aware_metrics_are_exact_on_identical_masks() -> None:
    labels, _ = _case()
    metrics = segmentation_metrics(labels, labels, (1.0, 1.0, 1.0))
    assert metrics["union_dice"] == 1.0
    assert metrics["surface_dice_1mm"] == 1.0
    assert metrics["assd_mm"] == 0.0
    assert binary_dice(labels > 0, labels > 0) == 1.0


def test_counterfactual_produces_all_required_comparator_arms() -> None:
    labels, logits = _case()
    outer = build_surface_samples(
        labels, (1.0, 1.0, 1.0), interface=Interface.OUTER, radius_mm=2.0, max_points=128
    )
    ap = build_surface_samples(
        labels,
        (1.0, 1.0, 1.0),
        interface=Interface.ANTERIOR_POSTERIOR,
        radius_mm=2.0,
        max_points=128,
    )
    result = run_counterfactual(
        logits,
        labels,
        (1.0, 1.0, 1.0),
        outer,
        ap_samples=ap,
        delta_mm=1.0,
        update_rms=0.05,
    )
    assert {
        "baseline",
        "dice_only",
        "ordinal_outer_only",
        "dice_plus_ordinal_outer",
        "onecut_outer_only",
        "dice_plus_onecut_outer",
        "bands_only",
        "dice_plus_bands",
        "ordinal_ap_only",
        "dice_plus_ordinal_ap",
        "onecut_ap_only",
        "dice_plus_onecut_ap",
    }.issubset(result.metrics_by_arm)
    assert np.isfinite(result.diagnostics["ordinal_outer_dice_gradient_cosine"])
