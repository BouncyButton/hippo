#!/usr/bin/env python3
"""Estimate canonical U-Net band/edge update costs using the baseline FLOP audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PROFILE_ROOT = ROOT / "reports/compute_budget_5pct_fold0_seed0_20261003"
sys.path[:0] = [str(ROOT), str(PROFILE_ROOT)]
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/hippo_bref_mpl")

import monai
import torch
from monai.losses import DiceLoss
from torch.utils.flop_counter import FlopCounterMode

import profile_matched_models as matched
import profile_reference as reference
from thesis.new_constraints.bands.outer_boundary import OuterBoundaryBandLoss
from thesis.new_constraints.constraint_result import differentiable_zero
from thesis.new_constraints.edge_consistency import BoundaryEdgeConsistencyLoss
from compute_constraint_objectives import ARMS, training_loss
from thesis.new_constraints.signed_distance import signed_distance_maps

BASE_SUPPLEMENT = reference.supplementary_flops


def constraint_supplement(func, args, kwargs, result):
    value, category = BASE_SUPPLEMENT(func, args, kwargs, result)
    if category != "no_supplement":
        return value, category
    name = str(func).split(".")[1] if str(func).startswith("aten.") else ""
    first = result[0] if isinstance(result, tuple) else result
    n = first.numel() if isinstance(first, torch.Tensor) and first.is_floating_point() else 0
    # Comparisons, gathers, copies and integer/boolean operations are excluded.
    # Count fused scalar arithmetic consistently with the reference's exp=1.
    formulas = {
        "logaddexp": 5,  # subtract, negate, exp, log1p, add; abs/comparison excluded
        "sigmoid": 4,  # negate, exp, add, reciprocal
        "sigmoid_backward": 3,  # (1-y)*y*upstream
        "log_sigmoid_forward": 5,
        "log_sigmoid_backward": 6,  # conservative fused scalar formula
        "log": 1,
        "log1p": 1,
    }
    if name in formulas:
        return formulas[name] * n, "constraint_scalar_estimate"
    if name in {"index_add", "index_add_"} and args[0].is_floating_point():
        return args[3].numel(), "scatter_add"
    if name == "binary_cross_entropy_with_logits":
        # Used only if not decomposed by this PyTorch dispatch version.
        return 9 * args[0].numel(), "constraint_bce_estimate"
    return value, category


def synthetic_target(geometry: str) -> torch.Tensor:
    coordinates = torch.arange(64)
    x, y, z = torch.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    if geometry == "ellipsoid":
        foreground = ((x - 31.5) / 15) ** 2 + ((y - 31.5) / 9) ** 2 + ((z - 31.5) / 7) ** 2 < 1
    else:
        foreground = (x >= 20) & (x < 44) & (y >= 22) & (y < 42) & (z >= 25) & (z < 39)
    return (foreground.long() * (1 + (x >= 32).long()))[None, None]


def profile_arm(arm: str, geometry: str) -> dict:
    torch.manual_seed(0)
    model, _ = matched.build_network("unet", "matched")
    model.train()
    criterion = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5, foreach=False)
    image = torch.randn(1, 1, 64, 64, 64)
    target = synthetic_target(geometry)
    sdf = signed_distance_maps(target) if "sdf" in arm else None
    audit = reference.OperatorAudit()
    stages = {}
    with audit, FlopCounterMode(display=False, custom_mapping={
        torch.ops.aten.convolution_backward: reference.conv_backward_flops,
    }) as counter:
        before = counter.get_total_flops()
        logits = model(image)
        stages["forward"] = counter.get_total_flops() - before
        audit.stage = "loss"
        before = counter.get_total_flops()
        # Call the production loss directly so the charged graph is identical.
        loss = training_loss(logits, target, arm, {"bands": .04, "edge": .1, "sdf": .1}, .2, sdf)
        stages["loss"] = counter.get_total_flops() - before
        audit.stage = "backward"
        before = counter.get_total_flops()
        loss.backward()
        stages["backward"] = counter.get_total_flops() - before
        audit.stage = "optimizer"
        before = counter.get_total_flops()
        optimizer.step()
        stages["optimizer"] = counter.get_total_flops() - before
    assert torch.isfinite(loss) and all(torch.isfinite(p).all() for p in model.parameters())
    return {
        "arm": arm, "geometry": geometry,
        "parameters": sum(p.numel() for p in model.parameters()),
        "flops_per_update": sum(stages.values()) + sum(audit.supplement.values()),
        "core_flops_per_stage": stages,
        "supplementary_estimate_per_stage": audit.supplement,
        "core_operator_flops": {str(k): int(v) for k, v in counter.flop_counts["Global"].items()},
        "operator_audit": audit.ops,
    }


def schedule(cost: int, budget: int) -> dict:
    updates = budget // cost
    epochs, remainder = divmod(updates, 10)
    assert updates * cost <= budget < (updates + 1) * cost
    return {
        "flops_per_update": cost, "gflops_per_update": cost / 1e9,
        "max_updates": updates, "patient_pass_epochs": updates / 10,
        "complete_epochs": epochs, "additional_updates": remainder,
        "loop_epochs": math.ceil(updates / 10),
        "lr_warmup_updates": max(1, math.ceil(0.05 * updates)),
        "consumed_flops": updates * cost, "unused_flops": budget - updates * cost,
        "next_update_exceeds_budget": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if monai.__version__ != "1.6.0" or not torch.__version__.startswith("2.2.2"):
        raise RuntimeError("Use the original profiling runtime: MONAI 1.6.0 / PyTorch 2.2.2")
    torch.set_num_threads(4)
    reference.supplementary_flops = constraint_supplement
    original = json.loads((ROOT / "experiments/compute_matched_baselines_20261003/protocol.json").read_text())
    baseline_cost = original["models"]["unet"]["flops_per_update"]
    budget = 4247 * baseline_cost
    args.output.mkdir(parents=True, exist_ok=True)
    profiles = []
    for arm in ARMS:
        for geometry in ("ellipsoid", "box"):
            row = profile_arm(arm, geometry)
            profiles.append(row)
            print(json.dumps({k: row[k] for k in ("arm", "geometry", "flops_per_update")}), flush=True)
    assert all(p["flops_per_update"] == baseline_cost for p in profiles if p["arm"] == "dice")
    schedules = {}
    for arm in ARMS:
        costs = {p["flops_per_update"] for p in profiles if p["arm"] == arm}
        assert len(costs) == 1, "Valid-mask geometry changed the arithmetic count"
        schedules[arm] = schedule(costs.pop(), budget)
        schedules[arm]["overhead_percent"] = 100 * (schedules[arm]["flops_per_update"] / baseline_cost - 1)
    source_paths = [Path(__file__), PROFILE_ROOT / "profile_reference.py",
                    PROFILE_ROOT / "profile_matched_models.py",
                    ROOT / "thesis/new_constraints/bands/outer_boundary.py",
                    ROOT / "thesis/new_constraints/edge_consistency.py",
                    ROOT / "thesis/new_constraints/signed_distance.py",
                    ROOT / "scripts/compute_constraint_objectives.py",
                    ROOT / "thesis/new_constraints/constraint_result.py"]
    result = {
        "official_budget_flops": budget, "official_budget_gflops": budget / 1e9,
        "budget_origin": "4247 times the frozen U-Net baseline update cost; pilot-derived",
        "runtime": {"torch": torch.__version__, "monai": monai.__version__, "device": "cpu"},
        "input_shape": [1, 1, 64, 64, 64], "training_patients": 10,
        "constraint_preset": "Original two-step balanced BCE bands, gamma=0, degree_alpha=0; original pooled edge consistency; weighted sum, one backward, no additional network forward",
        "calibration_included": False,
        "exclusions": ["validation", "preprocessing/augmentation", "checkpoint I/O",
                       "precision housekeeping", "integer/boolean/comparison and memory operations"],
        "notes": ["Declared arithmetic estimates, not GPU instruction measurements.",
                  "Counts match the original baseline exactly; dense MAC=2.",
                  "Morphology convolutions are charged as dense 3x3x3 kernels, including zero entries.",
                  "Bands and edge rebuild their masks independently, as in existing code; no unimplemented caching discount.",
                  "Two valid synthetic geometries verify fixed-shape counts; actual patient data are not used.",
                  "Positive weights and warmup scale do not affect arithmetic counts if losses are still computed.",
                  "If calibration is performed, deduct its separately measured compute before deriving the final run allowance.",
                  "Focal/degree bands, extra gradient probes, or multiple-backward methods require new profiling."],
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        "schedules": schedules, "profiles": profiles,
    }
    (args.output / "profiles.json").write_text(json.dumps(result, indent=2) + "\n")
    compact = {k: v for k, v in result.items() if k != "profiles"}
    (args.output / "training_schedules.json").write_text(json.dumps(compact, indent=2) + "\n")
    print(json.dumps(schedules, indent=2), flush=True)


if __name__ == "__main__":
    main()
