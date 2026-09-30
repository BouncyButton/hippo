#!/usr/bin/env python3
"""Evaluate frozen CST clues against saved fold-0 SwinUNETR probabilities."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from semantic_constraints.cst_teacher.data import extract_mask_slices  # noqa: E402
from semantic_constraints.cst_teacher.descriptors import (  # noqa: E402
    DESCRIPTOR_NAMES,
    labels_to_probabilities,
    sampled_slice_profiles,
    soft_descriptors,
)
from semantic_constraints.cst_teacher.model import (  # noqa: E402
    CSTDescriptorTeacher,
    CSTMaskAnomalyTeacher,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--inference-dir",
        type=Path,
        default=REPO_ROOT / "datasets/Dataset101_MSD/inference_fold0_val_600236",
    )
    parser.add_argument("--pkl", type=Path, default=REPO_ROOT / "datasets/Dataset101_MSD/msd_hippocampus_full.pkl")
    parser.add_argument("--splits-json", type=Path, default=REPO_ROOT / "datasets/Dataset101_MSD/splits_final.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args()


def pearson(first: np.ndarray, second: np.ndarray) -> float | None:
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if len(first) < 3 or first.std() < 1e-12 or second.std() < 1e-12:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def foreground_dice(probabilities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    hard = labels_to_probabilities(probabilities.argmax(dim=1, keepdim=True))
    target = labels_to_probabilities(labels)
    intersection = (hard[:, 1:3] * target[:, 1:3]).sum(dim=(2, 3, 4))
    denominator = hard[:, 1:3].sum(dim=(2, 3, 4)) + target[:, 1:3].sum(dim=(2, 3, 4))
    return ((2.0 * intersection + 1e-8) / (denominator + 1e-8)).mean(dim=1)


def load_teachers(
    checkpoint: Path,
    device: torch.device,
) -> tuple[CSTDescriptorTeacher, CSTMaskAnomalyTeacher, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if payload.get("schema") != "semantic_constraints.cst_teacher.v1":
        raise ValueError("unsupported CST teacher checkpoint schema")
    if tuple(payload["descriptor_names"]) != DESCRIPTOR_NAMES:
        raise ValueError("descriptor schema mismatch")
    config = payload["config"]
    common = {
        "slab_channels": int(config["slab_depth"]),
        "channels": tuple(config["channels"]),
        "heads": int(config["heads"]),
    }
    descriptor_teacher = CSTDescriptorTeacher(**common).to(device)
    anomaly_teacher = CSTMaskAnomalyTeacher(**common).to(device)
    descriptor_teacher.load_state_dict(payload["descriptor_teacher_state_dict"])
    anomaly_teacher.load_state_dict(payload["anomaly_teacher_state_dict"])
    descriptor_teacher.eval().requires_grad_(False)
    anomaly_teacher.eval().requires_grad_(False)
    return descriptor_teacher, anomaly_teacher, payload


def prediction_map(inference_dir: Path) -> dict[str, Path]:
    with (inference_dir / "prediction_index.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        path = inference_dir / "predictions" / f"case_{int(row['prediction_index']):04d}_prob.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        result[row["case_name"]] = path
    return result


def validation_label_map(loader: Any) -> dict[str, torch.Tensor]:
    result = {}
    for index in range(len(loader.dataset.base_dataset)):
        item = loader.dataset.base_dataset[index]
        result[str(item["case_name"])] = torch.as_tensor(item["label"]).long()
    return result


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from semantic_constraints.cst_teacher.train_teacher import build_loaders, resolve_device

    device = resolve_device(args.device)
    descriptor_teacher, anomaly_teacher, checkpoint = load_teachers(args.checkpoint, device)
    config = checkpoint["config"]
    args.fold = int(config["fold"])
    args.spatial_size = tuple(config["spatial_size"])
    args.set_size = int(config["set_size"])
    args.slab_depth = int(config["slab_depth"])
    args.inplane_size = int(config.get("inplane_size", 64))
    args.cache_dataset = True
    args.max_train_cases = 0
    args.max_val_cases = 0
    _, validation_loader = build_loaders(args)
    predictions = prediction_map(args.inference_dir)
    labels = validation_label_map(validation_loader)

    rows: list[dict[str, Any]] = []
    for batch in validation_loader:
        names = [str(name) for name in batch["case_name"]]
        missing = set(names) - predictions.keys()
        if missing:
            raise ValueError(f"missing saved predictions for {sorted(missing)}")
        image_slabs = batch["image_slabs"].to(device)
        metadata = batch["metadata"].to(device)
        valid = batch["valid_elements"].to(device)
        positions = batch["positions"][0].to(device)
        teacher_output = descriptor_teacher(image_slabs, metadata, valid)

        probabilities = torch.stack(
            [torch.from_numpy(np.load(predictions[name])).float() for name in names]
        ).to(device)
        label_batch = torch.stack([labels[name] for name in names]).to(device)
        dice = foreground_dice(probabilities, label_batch)
        predicted_descriptors = soft_descriptors(probabilities)
        target_descriptors = batch["descriptors"].to(device)
        predicted_profiles = sampled_slice_profiles(probabilities, positions)
        target_profiles = batch["slice_profiles"].to(device)
        bounds = teacher_output.descriptor_quantiles
        lower, median, upper = bounds[..., 0], bounds[..., 1], bounds[..., 2]
        interval_scale = ((upper - lower) / 2.0).clamp_min(0.02)
        violation = (
            torch.relu(lower - predicted_descriptors)
            + torch.relu(predicted_descriptors - upper)
        ) / interval_scale

        probability_slices = torch.stack(
            [
                extract_mask_slices(
                    probabilities[index],
                    positions,
                    output_size=(args.inplane_size, args.inplane_size),
                )
                for index in range(len(names))
            ]
        )
        anomaly_logits, _ = anomaly_teacher(image_slabs, probability_slices, metadata, valid)
        anomaly_scores = (anomaly_logits.sigmoid() * valid).sum(dim=1) / valid.sum(dim=1)
        teacher_profile_error = (teacher_output.slice_profiles - target_profiles).abs().mean(dim=(1, 2))
        prediction_profile_error = (predicted_profiles - target_profiles).abs().mean(dim=(1, 2))
        profile_violation = (predicted_profiles - teacher_output.slice_profiles).abs().mean(dim=(1, 2))

        for index, name in enumerate(names):
            descriptor_data = {}
            for descriptor_index, descriptor_name in enumerate(DESCRIPTOR_NAMES):
                descriptor_data[descriptor_name] = {
                    "target": float(target_descriptors[index, descriptor_index]),
                    "teacher_lower": float(lower[index, descriptor_index]),
                    "teacher_median": float(median[index, descriptor_index]),
                    "teacher_upper": float(upper[index, descriptor_index]),
                    "prediction": float(predicted_descriptors[index, descriptor_index]),
                    "teacher_covers_target": bool(
                        lower[index, descriptor_index] <= target_descriptors[index, descriptor_index]
                        <= upper[index, descriptor_index]
                    ),
                    "prediction_normalized_violation": float(violation[index, descriptor_index]),
                    "prediction_absolute_error": float(
                        (predicted_descriptors[index, descriptor_index] - target_descriptors[index, descriptor_index]).abs()
                    ),
                }
            rows.append(
                {
                    "case_name": name,
                    "foreground_dice": float(dice[index]),
                    "dice_error": float(1.0 - dice[index]),
                    "anomaly_score": float(anomaly_scores[index]),
                    "teacher_profile_mae": float(teacher_profile_error[index]),
                    "prediction_profile_mae": float(prediction_profile_error[index]),
                    "profile_violation": float(profile_violation[index]),
                    "descriptors": descriptor_data,
                }
            )

    dice_error = np.asarray([row["dice_error"] for row in rows])
    anomaly = np.asarray([row["anomaly_score"] for row in rows])
    profile = np.asarray([row["profile_violation"] for row in rows])
    descriptor_summary = {}
    for name in DESCRIPTOR_NAMES:
        data = [row["descriptors"][name] for row in rows]
        violations = np.asarray([item["prediction_normalized_violation"] for item in data])
        errors = np.asarray([item["prediction_absolute_error"] for item in data])
        teacher_errors = np.asarray([abs(item["teacher_median"] - item["target"]) for item in data])
        descriptor_summary[name] = {
            "teacher_median_mae": float(teacher_errors.mean()),
            "teacher_interval_coverage": float(np.mean([item["teacher_covers_target"] for item in data])),
            "prediction_mae": float(errors.mean()),
            "prediction_violation_rate": float(np.mean(violations > 0)),
            "mean_normalized_violation": float(violations.mean()),
            "violation_vs_dice_error_pearson": pearson(violations, dice_error),
            "violation_vs_descriptor_error_pearson": pearson(violations, errors),
        }

    return {
        "schema": "semantic_constraints.cst_teacher.prediction_evaluation.v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "inference_dir": str(args.inference_dir.resolve()),
        "cases": len(rows),
        "mean_foreground_dice": float(np.mean([row["foreground_dice"] for row in rows])),
        "descriptor_summary": descriptor_summary,
        "profile_summary": {
            "teacher_profile_mae": float(np.mean([row["teacher_profile_mae"] for row in rows])),
            "prediction_profile_mae": float(np.mean([row["prediction_profile_mae"] for row in rows])),
            "mean_profile_violation": float(profile.mean()),
            "violation_vs_dice_error_pearson": pearson(profile, dice_error),
        },
        "anomaly_summary": {
            "mean_score": float(anomaly.mean()),
            "score_vs_dice_error_pearson": pearson(anomaly, dice_error),
        },
        "per_case": rows,
    }


def main() -> None:
    args = parse_args()
    report = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    compact = {key: value for key, value in report.items() if key != "per_case"}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()

