"""Canonical weighted losses and training-only gradient-ratio calibration."""
from __future__ import annotations

import math
import numpy as np
import torch
from monai.losses import DiceLoss
from thesis.new_constraints.bands.outer_boundary import OuterBoundaryBandLoss
from thesis.new_constraints.constraint_result import differentiable_zero
from thesis.new_constraints.edge_consistency import BoundaryEdgeConsistencyLoss
from thesis.new_constraints.signed_distance import signed_distance_loss

ARMS = ("dice", "dice_bands", "dice_bands_edge", "dice_edge", "dice_sdf",
        "dice_bands_sdf", "dice_bands_edge_sdf", "dice_edge_sdf")


def training_loss(logits, labels, arm, weights, scale, target_sdf=None):
    auxiliary = differentiable_zero(logits.float())
    if "bands" in arm:
        auxiliary = auxiliary + weights["bands"] * OuterBoundaryBandLoss()(logits, labels).loss
    if "edge" in arm:
        auxiliary = auxiliary + weights["edge"] * BoundaryEdgeConsistencyLoss()(logits, labels)
    if "sdf" in arm:
        auxiliary = auxiliary + weights["sdf"] * signed_distance_loss(logits, labels, target_sdf)
    return DiceLoss(to_onehot_y=True, softmax=True)(logits, labels) + scale * auxiliary


def calibration_case(raw_logits, labels, arm, target_sdf=None):
    """Use detached logits: calibration never backpropagates through the model."""
    z = raw_logits.detach().float().requires_grad_(True)
    gd = torch.autograd.grad(DiceLoss(to_onehot_y=True, softmax=True)(z, labels), z)[0]
    dr = gd.square().mean().sqrt()
    result = {"dice_rms": float(dr)}
    if "bands" in arm:
        band = OuterBoundaryBandLoss()(z, labels)
        if not bool(band.details["valid"].all()):
            raise ValueError("Calibration requires valid inner and outer bands")
        gb = torch.autograd.grad(band.loss, z)[0]
        br = gb.square().mean().sqrt()
        result.update(bands_rms=float(br), bands_over_dice=float(br / dr))
        if "edge" in arm or "sdf" in arm:
            result["dice_bands_cross_mean"] = float((gd * gb).mean())
    if "edge" in arm:
        ge = torch.autograd.grad(BoundaryEdgeConsistencyLoss()(z, labels), z)[0]
        er = ge.square().mean().sqrt()
        result["edge_rms"] = float(er)
        if "bands" not in arm:
            result["edge_over_base"] = float(er / dr)
    if "sdf" in arm:
        gs = torch.autograd.grad(signed_distance_loss(z, labels, target_sdf), z)[0]
        result["sdf_rms"] = float(gs.square().mean().sqrt())
        if "edge" in arm:
            result["dice_edge_cross_mean"] = float((gd * ge).mean())
            if "bands" in arm:
                result["bands_edge_cross_mean"] = float((gb * ge).mean())
    if not all(math.isfinite(v) for v in result.values()) or any(
        v <= 0 for k, v in result.items() if not k.endswith("cross_mean")
    ):
        raise ValueError("Nonfinite or degenerate calibration gradients")
    return result


def calibrated_weight(ratios, target=0.1, cap=0.5):
    values = np.asarray(ratios, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Calibration ratios must be finite and positive")
    median, p95 = float(np.median(values)), float(np.quantile(values, .95))
    weight = min(target / median, cap / p95)
    return weight, {"median_raw_ratio": median, "p95_raw_ratio": p95,
                    "weighted_median_ratio": weight * median, "weighted_p95_ratio": weight * p95}


def select_weights(rows, arm):
    weights, statistics = {}, {}
    if "bands" in arm:
        weights["bands"], statistics["bands"] = calibrated_weight([r["bands_over_dice"] for r in rows])
    if "edge" in arm:
        if "bands" in arm:
            w = weights["bands"]
            for row in rows:
                # Exact squared norm identity for g_dice + w*g_bands.
                base_mse = row["dice_rms"] ** 2 + 2 * w * row["dice_bands_cross_mean"] + w ** 2 * row["bands_rms"] ** 2
                if not math.isfinite(base_mse) or base_mse <= 0:
                    raise ValueError("Degenerate combined Dice/bands gradient")
                row["edge_over_base"] = row["edge_rms"] / math.sqrt(base_mse)
        weights["edge"], statistics["edge"] = calibrated_weight([r["edge_over_base"] for r in rows])
    if "sdf" in arm:
        for row in rows:
            base_mse = row["dice_rms"] ** 2
            if "bands" in arm:
                w = weights["bands"]
                base_mse += 2 * w * row["dice_bands_cross_mean"] + w ** 2 * row["bands_rms"] ** 2
            if "edge" in arm:
                w = weights["edge"]
                base_mse += 2 * w * row["dice_edge_cross_mean"] + w ** 2 * row["edge_rms"] ** 2
                if "bands" in arm:
                    base_mse += 2 * weights["bands"] * w * row["bands_edge_cross_mean"]
            if not math.isfinite(base_mse) or base_mse <= 0:
                raise ValueError("Degenerate combined gradient for SDF calibration")
            row["sdf_over_base"] = row["sdf_rms"] / math.sqrt(base_mse)
        weights["sdf"], statistics["sdf"] = calibrated_weight([r["sdf_over_base"] for r in rows])
    return weights, statistics


def scheduled_costs(budget, baseline_cost, arm_cost, calibration_cost, prefix_updates):
    prefix = prefix_updates if calibration_cost else 0
    allowance = budget - prefix * baseline_cost - calibration_cost
    if allowance < arm_cost:
        raise ValueError("Budget cannot fit the calibration prefix and subsequent training")
    tail = allowance // arm_cost
    max_steps = prefix + tail
    consumed = prefix * baseline_cost + tail * arm_cost + calibration_cost
    assert consumed <= budget < consumed + arm_cost
    return {"max_steps": max_steps, "prefix_updates": prefix,
            "constraint_updates": tail, "calibration_flops": calibration_cost,
            "consumed_flops": consumed, "unused_flops": budget - consumed,
            "patient_pass_epochs": max_steps / 10}


def charge_for_step(step, schedule):
    return schedule["baseline_flops_per_update"] if step < schedule["prefix_updates"] else schedule["flops_per_update"]
