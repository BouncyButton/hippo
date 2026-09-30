#!/usr/bin/env python3
"""Frozen-logit screen for the directional outer-step constraint on MSD fold 0.

This is an exploratory local response test, not a model-training result. It
compares equal-RMS Dice-only and Dice-plus-step updates on saved probabilities.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from thesis.new_constraints.directional_steps import DirectionalStepSurfaceLoss  # noqa: E402
from thesis.new_constraints.supervised import build_supervised_loss  # noqa: E402


DEFAULT_INFERENCE = ROOT / "datasets/Dataset101_MSD/inference_fold0_val_600236"
DEFAULT_LABELS = ROOT / "datasets/Dataset101_MSD/labelsTr"


def _metrics(prediction: torch.Tensor, truth: torch.Tensor) -> dict[str, float | int]:
    pred = prediction.detach().cpu().numpy()
    gt = truth.detach().cpu().numpy()
    pred_fg, gt_fg = pred > 0, gt > 0
    intersection = np.count_nonzero(pred_fg & gt_fg)
    union_dice = 2 * intersection / max(1, np.count_nonzero(pred_fg) + np.count_nonzero(gt_fg))
    class_dice = []
    for class_id in (1, 2):
        a, b = pred == class_id, gt == class_id
        class_dice.append(2 * np.count_nonzero(a & b) / max(1, np.count_nonzero(a) + np.count_nonzero(b)))
    return {
        "union_dice": float(union_dice),
        "mean_class_dice": float(np.mean(class_dice)),
        "extra_voxels": int(np.count_nonzero(pred_fg & ~gt_fg)),
        "missed_voxels": int(np.count_nonzero(gt_fg & ~pred_fg)),
        "ap_swap_voxels": int(np.count_nonzero(pred_fg & gt_fg & (pred != gt))),
    }


def _rms(gradient: torch.Tensor) -> torch.Tensor:
    return gradient.square().mean().sqrt()


def _updated_metrics(
    logits: torch.Tensor,
    gradient: torch.Tensor,
    truth: torch.Tensor,
    step_rms: float,
) -> dict[str, float | int]:
    with torch.no_grad():
        update = step_rms * gradient / _rms(gradient).clamp_min(1e-20)
        return _metrics((logits - update).argmax(dim=1)[0], truth[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference", type=Path, default=DEFAULT_INFERENCE)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--step-rms", type=float, default=0.1)
    parser.add_argument("--relative-gradient-rms", type=float, default=0.1)
    args = parser.parse_args()
    if args.step_rms <= 0 or args.relative_gradient_rms <= 0:
        raise ValueError("step and relative gradient RMS must be positive")
    torch.set_num_threads(4)
    rows = list(csv.DictReader((args.inference / "prediction_index.csv").open()))
    if args.max_cases:
        rows = rows[:args.max_cases]
    dice_objective = build_supervised_loss("dice")
    step_objectives = {
        "directional_step": DirectionalStepSurfaceLoss(axes=(0, 1, 2)),
        "axis_steps_only": DirectionalStepSurfaceLoss(axes=(0, 1, 2), cross_weight=0.0),
    }
    cases = []
    for row in rows:
        index, name = int(row["prediction_index"]), row["case_name"]
        probability = np.load(args.inference / "predictions" / f"case_{index:04d}_prob.npy")
        saved_prediction = np.load(args.inference / "predictions" / f"case_{index:04d}_pred.npy")
        if probability.shape[0] != 3 or not np.array_equal(probability.argmax(axis=0), saved_prediction):
            raise ValueError(f"probability/prediction mismatch for {name}")
        native = np.asarray(nib.load(args.labels / f"{name}.nii.gz").dataobj)
        delta = np.asarray(probability.shape[1:]) - np.asarray(native.shape)
        if (delta < 0).any():
            raise ValueError(f"prediction grid smaller than native label for {name}")
        before = delta // 2
        padded = np.pad(native, tuple(zip(before, delta - before)))
        if padded.shape != saved_prediction.shape or not np.isin(padded, [0, 1, 2]).all():
            raise ValueError(f"invalid aligned label for {name}")
        logits = torch.from_numpy(np.log(np.maximum(probability, 1e-7))[None]).requires_grad_()
        truth = torch.from_numpy(padded[None].astype(np.int64))
        dice = dice_objective(logits, truth.unsqueeze(1))
        dice_gradient = torch.autograd.grad(dice, logits)[0]
        dice_rms = _rms(dice_gradient)
        if dice_rms <= 0:
            raise ValueError(f"zero Dice gradient for {name}")
        case: dict[str, object] = {
            "case": name,
            "dice_loss": float(dice.detach()),
            "before": _metrics(logits.detach().argmax(dim=1)[0], truth[0]),
            "dice_only": _updated_metrics(logits, dice_gradient, truth, args.step_rms),
            "arms": {},
        }
        for arm_name, objective in step_objectives.items():
            result = objective(logits, truth)
            auxiliary_gradient = torch.autograd.grad(result.loss, logits)[0]
            auxiliary_rms = _rms(auxiliary_gradient)
            if auxiliary_rms <= 0:
                raise ValueError(f"zero {arm_name} gradient for {name}")
            weight = args.relative_gradient_rms * dice_rms / auxiliary_rms
            combined = dice_gradient + weight * auxiliary_gradient
            cosine = torch.sum(dice_gradient * auxiliary_gradient) / (
                dice_rms * auxiliary_rms * dice_gradient.numel()
            )
            case["arms"][arm_name] = {
                "loss": float(result.loss.detach()),
                "axis_losses": [float(v) for v in result.value.detach()[0]],
                "longitudinal_order_loss": float(result.details["longitudinal_order_loss"].detach()[0]),
                "cross_axis_loss": float(result.details["cross_axis_loss"].detach()[0]),
                "calibrated_weight": float(weight.detach()),
                "dice_gradient_cosine": float(cosine.detach()),
                "after": _updated_metrics(logits, combined, truth, args.step_rms),
            }
        cases.append(case)
    summary = {}
    for arm_name in step_objectives:
        dice_deltas = np.asarray([
            case["arms"][arm_name]["after"]["union_dice"] - case["dice_only"]["union_dice"]
            for case in cases
        ])
        summary[arm_name] = {
            "mean_union_dice_delta_vs_dice_only": float(dice_deltas.mean()),
            "better_cases": int(np.count_nonzero(dice_deltas > 1e-12)),
            "worse_cases": int(np.count_nonzero(dice_deltas < -1e-12)),
            "same_cases": int(np.count_nonzero(np.abs(dice_deltas) <= 1e-12)),
            "mean_extra_voxel_delta_vs_dice_only": float(np.mean([
                case["arms"][arm_name]["after"]["extra_voxels"] - case["dice_only"]["extra_voxels"]
                for case in cases
            ])),
            "mean_missed_voxel_delta_vs_dice_only": float(np.mean([
                case["arms"][arm_name]["after"]["missed_voxels"] - case["dice_only"]["missed_voxels"]
                for case in cases
            ])),
            "mean_gradient_cosine": float(np.mean([case["arms"][arm_name]["dice_gradient_cosine"] for case in cases])),
            "median_calibrated_weight": float(np.median([case["arms"][arm_name]["calibrated_weight"] for case in cases])),
            "mean_auxiliary_loss": float(np.mean([case["arms"][arm_name]["loss"] for case in cases])),
            "mean_longitudinal_order_loss": float(np.mean([
                case["arms"][arm_name]["longitudinal_order_loss"] for case in cases
            ])),
        }
    report = {
        "schema": "hippo.directional_step_frozen_logit_audit.v2",
        "inference": str(args.inference),
        "cases_evaluated": len(cases),
        "step_rms": args.step_rms,
        "relative_gradient_rms": args.relative_gradient_rms,
        "warning": "Exploratory frozen-logit response on previously studied fold 0; no network retraining or independent test.",
        "summary": summary,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
