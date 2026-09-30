#!/usr/bin/env python3
"""Counterfactually test dense CST slice-profile constraints on frozen predictions."""

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
    labels_to_probabilities,
    sampled_slice_profiles,
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
    parser.add_argument("--output", type=Path, required=True)
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
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("slice_ap_bias", "slice_fg_ap_bias"),
        default=("slice_ap_bias", "slice_fg_ap_bias"),
    )
    parser.add_argument("--gammas", type=float, nargs="+", default=(0.001, 0.01, 0.1, 1.0))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--calibration-percentile", type=float, default=90.0)
    parser.add_argument("--minimum-teacher-votes", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=20260921)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if len(args.checkpoints) < args.minimum_teacher_votes:
        raise ValueError("minimum teacher votes cannot exceed checkpoint count")
    if not 0 < args.calibration_percentile < 100:
        raise ValueError("calibration percentile must lie in (0, 100)")
    if args.steps < 1 or args.learning_rate <= 0:
        raise ValueError("steps and learning rate must be positive")
    if any(gamma <= 0 for gamma in args.gammas):
        raise ValueError("gammas must be positive")


def redistribute_by_slice(
    original: torch.Tensor,
    foreground_delta: torch.Tensor,
    ap_delta: torch.Tensor,
) -> torch.Tensor:
    """Apply smooth class calibration parameters shared within each Y slice."""

    if original.shape[:2] != (1, 3):
        raise ValueError("original must have shape [1, 3, X, Y, Z]")
    expected = (1, 1, 1, original.shape[3], 1)
    if foreground_delta.shape != expected or ap_delta.shape != expected:
        raise ValueError(f"slice deltas must have shape {expected}")
    foreground = original[:, 1:3].sum(dim=1, keepdim=True).clamp(1e-6, 1 - 1e-6)
    anterior_given_foreground = (original[:, 1:2] / foreground).clamp(1e-5, 1 - 1e-5)
    repaired_foreground = torch.sigmoid(torch.logit(foreground) + foreground_delta)
    repaired_anterior_share = torch.sigmoid(torch.logit(anterior_given_foreground) + ap_delta)
    anterior = repaired_foreground * repaired_anterior_share
    posterior = repaired_foreground - anterior
    return torch.cat((1.0 - repaired_foreground, anterior, posterior), dim=1)


def soft_dice(probabilities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    target = labels_to_probabilities(labels)
    intersection = (probabilities[:, 1:3] * target[:, 1:3]).sum(dim=(2, 3, 4))
    denominator = probabilities[:, 1:3].sum(dim=(2, 3, 4)) + target[:, 1:3].sum(dim=(2, 3, 4))
    return ((2 * intersection + 1e-8) / (denominator + 1e-8)).mean(dim=1)


def profile_error(probabilities: torch.Tensor, positions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (sampled_slice_profiles(probabilities, positions) - target).abs().mean()


def repair(
    original: torch.Tensor,
    positions: torch.Tensor,
    target_profile: torch.Tensor,
    *,
    mode: str,
    gamma: float,
    steps: int,
    learning_rate: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    shape = (1, 1, 1, original.shape[3], 1)
    foreground_delta = torch.zeros(shape, device=original.device, requires_grad=mode == "slice_fg_ap_bias")
    ap_delta = torch.zeros(shape, device=original.device, requires_grad=True)
    parameters = [ap_delta] if mode == "slice_ap_bias" else [foreground_delta, ap_delta]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    initial_error = profile_error(original, positions, target_profile).detach()
    for _ in range(steps):
        repaired = redistribute_by_slice(original, foreground_delta, ap_delta)
        sequence_loss = profile_error(repaired, positions, target_profile)
        fidelity = F.mse_loss(repaired, original)
        loss = fidelity + gamma * sequence_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        repaired = redistribute_by_slice(original, foreground_delta, ap_delta)
        return repaired.detach(), {
            "initial_profile_error": float(initial_error),
            "final_profile_error": float(profile_error(repaired, positions, target_profile)),
            "fidelity_mse": float(F.mse_loss(repaired, original)),
        }


def bootstrap_interval(values: list[float], rng: np.random.Generator) -> list[float]:
    array = np.asarray(values, dtype=float)
    samples = array[rng.integers(0, len(array), size=(5000, len(array)))].mean(axis=1)
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def prepare(args: argparse.Namespace, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from semantic_constraints.cst_teacher.train_teacher import build_loaders

    teachers, payloads = [], []
    for checkpoint in args.checkpoints:
        teacher, _, payload = load_teachers(checkpoint, device)
        teachers.append(teacher)
        payloads.append(payload)
    first = payloads[0]["config"]
    for payload in payloads[1:]:
        for field in ("fold", "spatial_size", "set_size", "slab_depth", "inplane_size"):
            if payload["config"].get(field) != first.get(field):
                raise ValueError(f"checkpoints disagree on {field}")
    args.fold = int(first["fold"])
    args.spatial_size = tuple(first["spatial_size"])
    args.set_size = int(first["set_size"])
    args.slab_depth = int(first["slab_depth"])
    args.inplane_size = int(first.get("inplane_size", 32))
    args.cache_dataset = True
    args.max_train_cases = 0
    args.max_val_cases = 0
    train_loader, validation_loader = build_loaders(args)

    per_teacher_training_errors = [[] for _ in teachers]
    ensemble_training_errors = []
    with torch.no_grad():
        for batch in train_loader:
            image = batch["image_slabs"].to(device)
            metadata = batch["metadata"].to(device)
            valid = batch["valid_elements"].to(device)
            target = batch["slice_profiles"].to(device)
            outputs = torch.stack([teacher(image, metadata, valid).slice_profiles for teacher in teachers])
            errors = (outputs - target.unsqueeze(0)).abs().mean(dim=(2, 3))
            for index in range(len(teachers)):
                per_teacher_training_errors[index].extend(errors[index].cpu().tolist())
            ensemble = outputs.median(dim=0).values
            ensemble_training_errors.extend((ensemble - target).abs().mean(dim=(1, 2)).cpu().tolist())
    thresholds = [
        float(np.percentile(values, args.calibration_percentile)) for values in per_teacher_training_errors
    ]
    ensemble_threshold = float(np.percentile(ensemble_training_errors, args.calibration_percentile))

    paths = prediction_map(args.inference_dir)
    labels = validation_label_map(validation_loader)
    cases = []
    with torch.no_grad():
        for batch in validation_loader:
            names = [str(name) for name in batch["case_name"]]
            image = batch["image_slabs"].to(device)
            metadata = batch["metadata"].to(device)
            valid = batch["valid_elements"].to(device)
            positions = batch["positions"][0].to(device)
            outputs = torch.stack([teacher(image, metadata, valid).slice_profiles for teacher in teachers])
            ensemble = outputs.median(dim=0).values
            probabilities = torch.stack([torch.from_numpy(np.load(paths[name])).float() for name in names]).to(device)
            predicted = sampled_slice_profiles(probabilities, positions)
            errors = (outputs - predicted.unsqueeze(0)).abs().mean(dim=(2, 3))
            ensemble_errors = (ensemble - predicted).abs().mean(dim=(1, 2))
            for index, name in enumerate(names):
                votes = sum(float(errors[t, index]) > thresholds[t] for t in range(len(teachers)))
                difference = (ensemble[index] - predicted[index]).abs()
                flat_index = int(difference.flatten().argmax())
                element_index = flat_index // 2
                class_index = flat_index % 2
                cases.append(
                    {
                        "case_name": name,
                        "probabilities": probabilities[index : index + 1].cpu(),
                        "label": labels[name].unsqueeze(0).cpu(),
                        "teacher_profile": ensemble[index : index + 1].cpu(),
                        "positions": positions.cpu(),
                        "teacher_votes": votes,
                        "initial_profile_error": float(ensemble_errors[index]),
                        "flagged": votes >= args.minimum_teacher_votes
                        and float(ensemble_errors[index]) > ensemble_threshold,
                        "largest_mismatch_position": int(positions[element_index]),
                        "largest_mismatch_class": "anterior" if class_index == 0 else "posterior",
                    }
                )
    return cases, {
        "fold": args.fold,
        "set_size": args.set_size,
        "teacher_thresholds": thresholds,
        "ensemble_threshold": ensemble_threshold,
        "calibration_percentile": args.calibration_percentile,
    }


def run_configuration(
    cases: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
    mode: str,
    gamma: float,
    threshold: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    rows = []
    for case in cases:
        original = case["probabilities"].to(device)
        label = case["label"].to(device)
        target = case["teacher_profile"].to(device)
        positions = case["positions"].to(device)
        initial_hard = foreground_dice(original, label)[0]
        initial_soft = soft_dice(original, label)[0]
        if case["flagged"]:
            repaired, diagnostics = repair(
                original,
                positions,
                target,
                mode=mode,
                gamma=gamma,
                steps=args.steps,
                learning_rate=args.learning_rate,
            )
        else:
            repaired = original
            diagnostics = {
                "initial_profile_error": case["initial_profile_error"],
                "final_profile_error": case["initial_profile_error"],
                "fidelity_mse": 0.0,
            }
        final_hard = foreground_dice(repaired, label)[0]
        final_soft = soft_dice(repaired, label)[0]
        rows.append(
            {
                "case_name": case["case_name"],
                "flagged": case["flagged"],
                "teacher_votes": case["teacher_votes"],
                "largest_mismatch_position": case["largest_mismatch_position"],
                "largest_mismatch_class": case["largest_mismatch_class"],
                "hard_dice_delta": float(final_hard - initial_hard),
                "soft_dice_delta": float(final_soft - initial_soft),
                "hard_changed_voxel_fraction": float(
                    (repaired.argmax(dim=1) != original.argmax(dim=1)).float().mean()
                ),
                **diagnostics,
            }
        )
    flagged = [row for row in rows if row["flagged"]]
    if not flagged:
        raise RuntimeError("calibrated ensemble did not flag validation cases")
    hard = [row["hard_dice_delta"] for row in flagged]
    soft = [row["soft_dice_delta"] for row in flagged]
    summary = {
        "mode": mode,
        "gamma": gamma,
        "flagged_cases": len(flagged),
        "flagged_case_names": [row["case_name"] for row in flagged],
        "repair_success_rate": float(np.mean([row["final_profile_error"] <= threshold for row in flagged])),
        "mean_initial_profile_error": float(np.mean([row["initial_profile_error"] for row in flagged])),
        "mean_final_profile_error": float(np.mean([row["final_profile_error"] for row in flagged])),
        "mean_flagged_hard_dice_delta": float(np.mean(hard)),
        "mean_flagged_soft_dice_delta": float(np.mean(soft)),
        "hard_dice_delta_bootstrap_95ci": bootstrap_interval(hard, rng),
        "soft_dice_delta_bootstrap_95ci": bootstrap_interval(soft, rng),
        "hard_improvement_rate": float(np.mean([value > 1e-6 for value in hard])),
        "hard_harm_rate": float(np.mean([value < -1e-6 for value in hard])),
        "mean_all_case_hard_dice_delta": float(np.mean([row["hard_dice_delta"] for row in rows])),
        "mean_all_case_soft_dice_delta": float(np.mean([row["soft_dice_delta"] for row in rows])),
        "mean_changed_voxel_fraction": float(
            np.mean([row["hard_changed_voxel_fraction"] for row in flagged])
        ),
    }
    return {"summary": summary, "per_case": rows}


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Dense CST profile counterfactual repair",
        "",
        f"Calibration threshold: `{report['calibration']['ensemble_threshold']:.6f}`",
        f"Flagged cases: {', '.join(report['flagged_cases'])}",
        "",
        "| Mode | Gamma | Success | Hard Dice Δ | Soft Dice Δ | Harm |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for config in report["configurations"]:
        item = config["summary"]
        lines.append(
            "| {mode} | {gamma:g} | {repair_success_rate:.1%} | {mean_flagged_hard_dice_delta:+.6f} | "
            "{mean_flagged_soft_dice_delta:+.6f} | {hard_harm_rate:.1%} |".format(**item)
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    from semantic_constraints.cst_teacher.train_teacher import resolve_device, seed_everything

    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)
    cases, calibration = prepare(args, device)
    ranked_cases = sorted(cases, key=lambda case: case["initial_profile_error"], reverse=True)
    print(
        json.dumps(
            {
                "stage": "calibration",
                "calibration": calibration,
                "flagged_cases": [case["case_name"] for case in cases if case["flagged"]],
                "top_disagreements": [
                    {
                        "case_name": case["case_name"],
                        "profile_error": case["initial_profile_error"],
                        "teacher_votes": case["teacher_votes"],
                    }
                    for case in ranked_cases[:10]
                ],
            }
        ),
        flush=True,
    )
    configurations = []
    for mode in args.modes:
        for gamma in args.gammas:
            print(json.dumps({"stage": "repair", "mode": mode, "gamma": gamma}), flush=True)
            result = run_configuration(
                cases,
                args,
                device,
                mode,
                gamma,
                calibration["ensemble_threshold"],
                rng,
            )
            configurations.append(result)
            print(json.dumps(result["summary"], sort_keys=True), flush=True)
    report = {
        "schema": "semantic_constraints.cst_teacher.profile_counterfactual.v1",
        "checkpoints": [str(path.resolve()) for path in args.checkpoints],
        "cases": len(cases),
        "calibration": calibration,
        "flagged_cases": [case["case_name"] for case in cases if case["flagged"]],
        "configurations": configurations,
        "warning": "Discovery-fold counterfactual evidence only; labels were used after repair for measurement.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
