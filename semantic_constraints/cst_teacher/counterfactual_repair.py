#!/usr/bin/env python3
"""Falsify a CST descriptor clue by repairing frozen SwinUNETR predictions.

The repair never sees the validation label. It preserves each voxel's total
foreground probability and only redistributes probability between anterior
and posterior. Labels are used after optimization solely for measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from semantic_constraints.cst_teacher.descriptors import (  # noqa: E402
    DESCRIPTOR_NAMES,
    labels_to_probabilities,
    soft_descriptors,
)
from semantic_constraints.cst_teacher.evaluate_predictions import (  # noqa: E402
    foreground_dice,
    load_teachers,
    prediction_map,
    validation_label_map,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--inference-dir", type=Path, required=True)
    parser.add_argument(
        "--pkl",
        type=Path,
        default=REPO_ROOT / "datasets/Dataset101_MSD/msd_hippocampus_full.pkl",
    )
    parser.add_argument(
        "--splits-json",
        type=Path,
        default=REPO_ROOT / "datasets/Dataset101_MSD/splits_final.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("global_ap_bias", "voxel_ap_field"),
        default=("global_ap_bias", "voxel_ap_field"),
    )
    parser.add_argument("--gammas", type=float, nargs="+", default=(0.001, 0.01, 0.1, 1.0, 10.0))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--minimum-teacher-votes", type=int, default=2)
    parser.add_argument("--success-tolerance", type=float, default=0.01)
    parser.add_argument("--harm-tolerance", type=float, default=1e-6)
    parser.add_argument("--descriptor", choices=DESCRIPTOR_NAMES, default="anterior_volume_fraction")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=20260921)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if len(args.checkpoints) < args.minimum_teacher_votes:
        raise ValueError("minimum teacher votes cannot exceed the number of checkpoints")
    if args.steps < 1 or args.learning_rate <= 0:
        raise ValueError("steps and learning rate must be positive")
    if not args.gammas or any(gamma <= 0 for gamma in args.gammas):
        raise ValueError("all gammas must be positive")


def normalized_violation(
    value: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    minimum_scale: float = 0.02,
) -> torch.Tensor:
    scale = ((upper - lower) / 2.0).clamp_min(minimum_scale)
    return (F.relu(lower - value) + F.relu(value - upper)) / scale


def redistribute_ap(original: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Preserve P(foreground) while shifting its A/P conditional probability."""

    if original.shape[0] != 1 or original.shape[1] != 3:
        raise ValueError("original must have shape [1, 3, X, Y, Z]")
    foreground = original[:, 1:3].sum(dim=1, keepdim=True)
    anterior_given_foreground = (original[:, 1:2] / foreground.clamp_min(1e-7)).clamp(1e-5, 1 - 1e-5)
    conditional_logit = torch.logit(anterior_given_foreground)
    repaired_anterior = foreground * torch.sigmoid(conditional_logit + delta)
    repaired_posterior = foreground - repaired_anterior
    return torch.cat((original[:, 0:1], repaired_anterior, repaired_posterior), dim=1)


def mean_soft_foreground_dice(probabilities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    target = labels_to_probabilities(labels)
    intersection = (probabilities[:, 1:3] * target[:, 1:3]).sum(dim=(2, 3, 4))
    denominator = probabilities[:, 1:3].sum(dim=(2, 3, 4)) + target[:, 1:3].sum(dim=(2, 3, 4))
    return ((2.0 * intersection + 1e-8) / (denominator + 1e-8)).mean(dim=1)


def hard_change_fraction(first: torch.Tensor, second: torch.Tensor) -> float:
    return float((first.argmax(dim=1) != second.argmax(dim=1)).float().mean())


def repair_prediction(
    original: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    descriptor_index: int,
    mode: str,
    gamma: float,
    steps: int,
    learning_rate: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    spatial_shape = original.shape[2:]
    delta_shape = (1, 1, 1, 1, 1) if mode == "global_ap_bias" else (1, 1, *spatial_shape)
    delta = torch.zeros(delta_shape, device=original.device, dtype=original.dtype, requires_grad=True)
    optimizer = torch.optim.Adam((delta,), lr=learning_rate)

    initial_value = soft_descriptors(original)[0, descriptor_index].detach()
    initial_violation = normalized_violation(initial_value, lower, upper).detach()
    for _ in range(steps):
        repaired = redistribute_ap(original, delta)
        value = soft_descriptors(repaired)[0, descriptor_index]
        violation = normalized_violation(value, lower, upper)
        fidelity = F.mse_loss(repaired, original)
        loss = fidelity + gamma * violation.square()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        repaired = redistribute_ap(original, delta)
        final_value = soft_descriptors(repaired)[0, descriptor_index]
        final_violation = normalized_violation(final_value, lower, upper)
        diagnostics = {
            "initial_descriptor": float(initial_value),
            "final_descriptor": float(final_value),
            "initial_violation": float(initial_violation),
            "final_violation": float(final_violation),
            "fidelity_mse": float(F.mse_loss(repaired, original)),
            "delta_rms": float(delta.square().mean().sqrt()),
        }
    return repaired.detach(), diagnostics


def percentile_interval(values: list[float], rng: np.random.Generator, samples: int = 5000) -> list[float | None]:
    if not values:
        return [None, None]
    array = np.asarray(values, dtype=float)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def collect_cases(args: argparse.Namespace, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from semantic_constraints.cst_teacher.train_teacher import build_loaders

    teachers = []
    payloads = []
    for path in args.checkpoints:
        teacher, _, payload = load_teachers(path, device)
        teachers.append(teacher)
        payloads.append(payload)
    configs = [payload["config"] for payload in payloads]
    first = configs[0]
    architecture_fields = ("fold", "spatial_size", "set_size", "slab_depth", "inplane_size")
    for field in architecture_fields:
        if any(config.get(field) != first.get(field) for config in configs[1:]):
            raise ValueError(f"teacher checkpoints disagree on {field}")

    args.fold = int(first["fold"])
    args.spatial_size = tuple(first["spatial_size"])
    args.set_size = int(first["set_size"])
    args.slab_depth = int(first["slab_depth"])
    args.inplane_size = int(first.get("inplane_size", 32))
    args.cache_dataset = True
    args.max_train_cases = 0
    args.max_val_cases = 0
    _, loader = build_loaders(args)
    prediction_paths = prediction_map(args.inference_dir)
    labels = validation_label_map(loader)
    descriptor_index = DESCRIPTOR_NAMES.index(args.descriptor)

    cases: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            names = [str(name) for name in batch["case_name"]]
            image_slabs = batch["image_slabs"].to(device)
            metadata = batch["metadata"].to(device)
            valid = batch["valid_elements"].to(device)
            bounds = torch.stack(
                [teacher(image_slabs, metadata, valid).descriptor_quantiles for teacher in teachers]
            )[:, :, descriptor_index]
            probabilities = torch.stack(
                [torch.from_numpy(np.load(prediction_paths[name])).float() for name in names]
            ).to(device)
            values = soft_descriptors(probabilities)[:, descriptor_index]
            for index, name in enumerate(names):
                per_teacher = bounds[:, index]
                lower = per_teacher[:, 0]
                upper = per_teacher[:, 2]
                value = values[index]
                votes_low = int((value < lower).sum())
                votes_high = int((value > upper).sum())
                ensemble_lower = lower.median()
                ensemble_upper = upper.median()
                ensemble_violation = normalized_violation(value, ensemble_lower, ensemble_upper)
                flagged = (
                    float(ensemble_violation) > 0
                    and max(votes_low, votes_high) >= args.minimum_teacher_votes
                )
                cases.append(
                    {
                        "case_name": name,
                        "probabilities": probabilities[index : index + 1].detach().cpu(),
                        "label": labels[name].unsqueeze(0).detach().cpu(),
                        "lower": float(ensemble_lower),
                        "upper": float(ensemble_upper),
                        "teacher_lowers": [float(item) for item in lower],
                        "teacher_uppers": [float(item) for item in upper],
                        "votes_low": votes_low,
                        "votes_high": votes_high,
                        "flagged": flagged,
                    }
                )
    metadata = {
        "fold": args.fold,
        "descriptor": args.descriptor,
        "descriptor_index": descriptor_index,
        "checkpoint_configs": configs,
    }
    return cases, metadata


def run_configuration(
    cases: list[dict[str, Any]],
    *,
    descriptor_index: int,
    mode: str,
    gamma: float,
    args: argparse.Namespace,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, Any]:
    rows = []
    for case in cases:
        original = case["probabilities"].to(device)
        label = case["label"].to(device)
        lower = original.new_tensor(case["lower"])
        upper = original.new_tensor(case["upper"])
        target_descriptor = soft_descriptors(labels_to_probabilities(label))[0, descriptor_index]
        original_descriptor = soft_descriptors(original)[0, descriptor_index]
        original_hard_dice = foreground_dice(original, label)[0]
        original_soft_dice = mean_soft_foreground_dice(original, label)[0]

        if case["flagged"]:
            repaired, diagnostics = repair_prediction(
                original,
                lower,
                upper,
                descriptor_index=descriptor_index,
                mode=mode,
                gamma=gamma,
                steps=args.steps,
                learning_rate=args.learning_rate,
            )
        else:
            repaired = original
            value = soft_descriptors(original)[0, descriptor_index]
            violation = normalized_violation(value, lower, upper)
            diagnostics = {
                "initial_descriptor": float(value),
                "final_descriptor": float(value),
                "initial_violation": float(violation),
                "final_violation": float(violation),
                "fidelity_mse": 0.0,
                "delta_rms": 0.0,
            }

        repaired_hard_dice = foreground_dice(repaired, label)[0]
        repaired_soft_dice = mean_soft_foreground_dice(repaired, label)[0]
        if float(original_descriptor) < case["lower"]:
            required_direction = "increase"
            correct_direction = float(target_descriptor) > float(original_descriptor)
        elif float(original_descriptor) > case["upper"]:
            required_direction = "decrease"
            correct_direction = float(target_descriptor) < float(original_descriptor)
        else:
            required_direction = "inside"
            correct_direction = True
        rows.append(
            {
                "case_name": case["case_name"],
                "flagged": case["flagged"],
                "votes_low": case["votes_low"],
                "votes_high": case["votes_high"],
                "teacher_lower": case["lower"],
                "teacher_upper": case["upper"],
                "target_descriptor": float(target_descriptor),
                "required_direction": required_direction,
                "target_confirms_direction": bool(correct_direction),
                "original_hard_dice": float(original_hard_dice),
                "repaired_hard_dice": float(repaired_hard_dice),
                "hard_dice_delta": float(repaired_hard_dice - original_hard_dice),
                "original_soft_dice": float(original_soft_dice),
                "repaired_soft_dice": float(repaired_soft_dice),
                "soft_dice_delta": float(repaired_soft_dice - original_soft_dice),
                "descriptor_absolute_error_delta": float(
                    abs(diagnostics["final_descriptor"] - float(target_descriptor))
                    - abs(diagnostics["initial_descriptor"] - float(target_descriptor))
                ),
                "hard_changed_voxel_fraction": hard_change_fraction(original, repaired),
                **diagnostics,
            }
        )

    flagged = [row for row in rows if row["flagged"]]
    if not flagged:
        raise RuntimeError("the ensemble did not flag any cases")
    hard_deltas = [row["hard_dice_delta"] for row in flagged]
    soft_deltas = [row["soft_dice_delta"] for row in flagged]
    successful = [row["final_violation"] <= args.success_tolerance for row in flagged]
    summary = {
        "mode": mode,
        "gamma": gamma,
        "steps": args.steps,
        "flagged_cases": len(flagged),
        "flagged_case_names": [row["case_name"] for row in flagged],
        "direction_accuracy": float(np.mean([row["target_confirms_direction"] for row in flagged])),
        "repair_success_rate": float(np.mean(successful)),
        "mean_initial_violation": float(np.mean([row["initial_violation"] for row in flagged])),
        "mean_final_violation": float(np.mean([row["final_violation"] for row in flagged])),
        "mean_flagged_hard_dice_delta": float(np.mean(hard_deltas)),
        "mean_flagged_soft_dice_delta": float(np.mean(soft_deltas)),
        "mean_all_case_hard_dice_delta": float(np.mean([row["hard_dice_delta"] for row in rows])),
        "mean_all_case_soft_dice_delta": float(np.mean([row["soft_dice_delta"] for row in rows])),
        "hard_dice_delta_bootstrap_95ci": percentile_interval(hard_deltas, rng),
        "soft_dice_delta_bootstrap_95ci": percentile_interval(soft_deltas, rng),
        "hard_improvement_rate": float(np.mean([value > args.harm_tolerance for value in hard_deltas])),
        "hard_harm_rate": float(np.mean([value < -args.harm_tolerance for value in hard_deltas])),
        "mean_descriptor_absolute_error_delta": float(
            np.mean([row["descriptor_absolute_error_delta"] for row in flagged])
        ),
        "mean_hard_changed_voxel_fraction": float(
            np.mean([row["hard_changed_voxel_fraction"] for row in flagged])
        ),
        "mean_fidelity_mse": float(np.mean([row["fidelity_mse"] for row in flagged])),
    }
    return {"summary": summary, "per_case": rows}


def choose_configuration(configurations: list[dict[str, Any]]) -> int | None:
    eligible = [
        (index, config)
        for index, config in enumerate(configurations)
        if config["summary"]["repair_success_rate"] >= 0.75
        and config["summary"]["hard_harm_rate"] <= 0.25
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda item: (
            item[1]["summary"]["mean_flagged_soft_dice_delta"],
            item[1]["summary"]["mean_flagged_hard_dice_delta"],
            -item[1]["summary"]["mean_fidelity_mse"],
        ),
    )[0]


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CST counterfactual repair",
        "",
        f"Descriptor: `{report['descriptor']}`",
        f"Consensus flags: {', '.join(report['consensus_flagged_cases'])}",
        "",
        "| Mode | Gamma | Success | Hard Dice Δ (flagged) | Soft Dice Δ (flagged) | Harm | Changed voxels |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for config in report["configurations"]:
        summary = config["summary"]
        lines.append(
            "| {mode} | {gamma:g} | {repair_success_rate:.1%} | {mean_flagged_hard_dice_delta:+.6f} | "
            "{mean_flagged_soft_dice_delta:+.6f} | {hard_harm_rate:.1%} | "
            "{mean_hard_changed_voxel_fraction:.4%} |".format(**summary)
        )
    lines.extend(("", f"Selected configuration index: `{report['selected_configuration_index']}`"))
    return "\n".join(lines) + "\n"


def main() -> None:
    from semantic_constraints.cst_teacher.train_teacher import resolve_device, seed_everything

    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)
    cases, metadata = collect_cases(args, device)
    configurations = []
    for mode in args.modes:
        for gamma in args.gammas:
            print(json.dumps({"stage": "repair", "mode": mode, "gamma": gamma}), flush=True)
            configurations.append(
                run_configuration(
                    cases,
                    descriptor_index=metadata["descriptor_index"],
                    mode=mode,
                    gamma=gamma,
                    args=args,
                    device=device,
                    rng=rng,
                )
            )
            print(json.dumps(configurations[-1]["summary"], sort_keys=True), flush=True)

    selected = choose_configuration(configurations)
    report = {
        "schema": "semantic_constraints.cst_teacher.counterfactual_repair.v1",
        "descriptor": args.descriptor,
        "checkpoints": [str(path.resolve()) for path in args.checkpoints],
        "inference_dir": str(args.inference_dir.resolve()),
        "cases": len(cases),
        "minimum_teacher_votes": args.minimum_teacher_votes,
        "consensus_flagged_cases": [case["case_name"] for case in cases if case["flagged"]],
        "metadata": metadata,
        "configurations": configurations,
        "selected_configuration_index": selected,
        "warning": (
            "This is discovery-fold counterfactual evidence, not a holdout estimate. "
            "Ground truth was used only for post-repair measurement and configuration comparison."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = args.output.with_suffix(".md")
    markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "markdown": str(markdown), "selected": selected}))


if __name__ == "__main__":
    main()
