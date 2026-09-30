#!/usr/bin/env python3
"""Cross-fit case- and slice-level Swin error probes from frozen CST features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from semantic_constraints.cst_teacher.descriptors import (  # noqa: E402
    labels_to_probabilities,
    sampled_slice_profiles,
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
    parser.add_argument("--output-dir", type=Path, required=True)
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
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--alphas", type=float, nargs="+", default=(1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=20260921)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.outer_folds < 2 or args.inner_folds < 2 or args.repeats < 1:
        raise ValueError("fold counts must be at least two and repeats positive")
    if not args.alphas or any(alpha <= 0 for alpha in args.alphas):
        raise ValueError("ridge alphas must be positive")


def grouped_folds(groups: np.ndarray, folds: int, seed: int) -> list[np.ndarray]:
    unique = np.unique(groups)
    if folds > len(unique):
        raise ValueError("fold count exceeds unique group count")
    shuffled = np.random.default_rng(seed).permutation(unique)
    return [np.asarray(part) for part in np.array_split(shuffled, folds)]


def ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    mean_x = x_train.mean(axis=0)
    scale_x = x_train.std(axis=0)
    scale_x[scale_x < 1e-8] = 1.0
    train = (x_train - mean_x) / scale_x
    test = (x_test - mean_x) / scale_x
    mean_y = float(y_train.mean())
    centered_y = y_train - mean_y
    dimension = train.shape[1]
    if train.shape[0] >= dimension:
        weights = np.linalg.solve(train.T @ train + alpha * np.eye(dimension), train.T @ centered_y)
    else:
        weights = train.T @ np.linalg.solve(
            train @ train.T + alpha * np.eye(train.shape[0]),
            centered_y,
        )
    return test @ weights + mean_y


def choose_alpha(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alphas: list[float],
    folds: int,
    seed: int,
) -> float:
    partitions = grouped_folds(groups, min(folds, len(np.unique(groups))), seed)
    scores = []
    for alpha in alphas:
        losses = []
        for validation_groups in partitions:
            validation = np.isin(groups, validation_groups)
            training = ~validation
            predictions = ridge_predict(x[training], y[training], x[validation], alpha)
            losses.append(float(np.mean((predictions - y[validation]) ** 2)))
        scores.append(mean(losses))
    return float(alphas[int(np.argmin(scores))])


def repeated_nested_predictions(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    outer_folds: int,
    inner_folds: int,
    repeats: int,
    alphas: list[float],
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    predictions = np.zeros((repeats, len(y)), dtype=np.float64)
    selections = []
    for repeat in range(repeats):
        partitions = grouped_folds(groups, outer_folds, seed + 1009 * repeat)
        for fold_index, test_groups in enumerate(partitions):
            test = np.isin(groups, test_groups)
            training = ~test
            alpha = choose_alpha(
                x[training],
                y[training],
                groups[training],
                alphas,
                inner_folds,
                seed + 1009 * repeat + 37 * fold_index,
            )
            predictions[repeat, test] = ridge_predict(x[training], y[training], x[test], alpha)
            selections.append({"repeat": repeat, "fold": fold_index, "alpha": alpha})
    return predictions.mean(axis=0), selections


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return result


def correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    if len(first) < 3 or first.std() < 1e-12 or second.std() < 1e-12:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def regression_metrics(y: np.ndarray, prediction: np.ndarray, *, top_count: int) -> dict[str, Any]:
    residual = y - prediction
    denominator = float(np.sum((y - y.mean()) ** 2))
    true_top = set(np.argsort(y)[-top_count:])
    predicted_top = set(np.argsort(prediction)[-top_count:])
    return {
        "pearson": correlation(y, prediction),
        "spearman": correlation(ranks(y), ranks(prediction)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": None if denominator < 1e-12 else float(1.0 - np.sum(residual**2) / denominator),
        "top_error_recall": float(len(true_top & predicted_top) / top_count),
        "target_mean": float(y.mean()),
        "prediction_mean": float(prediction.mean()),
    }


def slice_dice_error(probabilities: torch.Tensor, labels: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    hard = labels_to_probabilities(probabilities.argmax(dim=1, keepdim=True))[:, 1:3]
    target = labels_to_probabilities(labels)[:, 1:3]
    hard = hard.index_select(3, positions)
    target = target.index_select(3, positions)
    intersection = (hard * target).sum(dim=(2, 4))
    denominator = hard.sum(dim=(2, 4)) + target.sum(dim=(2, 4))
    dice = (2.0 * intersection + 1e-8) / (denominator + 1e-8)
    return 1.0 - dice.mean(dim=1)


def entropy_features(probabilities: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    entropy = -(probabilities.clamp_min(1e-7) * probabilities.clamp_min(1e-7).log()).sum(dim=1)
    per_slice = entropy.index_select(2, positions).mean(dim=(1, 3))
    flat = entropy.flatten(start_dim=1)
    case = torch.stack(
        (
            flat.mean(dim=1),
            torch.quantile(flat, 0.95, dim=1),
            probabilities.amax(dim=1).flatten(start_dim=1).mean(dim=1),
        ),
        dim=1,
    )
    return case, per_slice


def extract_seed_features(
    checkpoint: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    from semantic_constraints.cst_teacher.train_teacher import build_loaders

    teacher, _, payload = load_teachers(checkpoint, device)
    config = payload["config"]
    args.fold = int(config["fold"])
    args.spatial_size = tuple(config["spatial_size"])
    args.set_size = int(config["set_size"])
    args.slab_depth = int(config["slab_depth"])
    args.inplane_size = int(config.get("inplane_size", 32))
    args.cache_dataset = True
    args.max_train_cases = 0
    args.max_val_cases = 0
    _, loader = build_loaders(args)
    paths = prediction_map(args.inference_dir)
    labels = validation_label_map(loader)

    case_names: list[str] = []
    case_targets, slice_targets = [], []
    slice_correction_targets, slice_teacher_corrections, slice_prediction_profiles = [], [], []
    case_feature_blocks: dict[str, list[np.ndarray]] = {
        "uncertainty": [],
        "cst_embedding": [],
        "relationship_residuals": [],
        "combined": [],
    }
    slice_feature_blocks: dict[str, list[np.ndarray]] = {
        "uncertainty": [],
        "cst_embedding": [],
        "relationship_residuals": [],
        "combined": [],
    }
    slice_groups = []
    with torch.no_grad():
        for batch in loader:
            names = [str(name) for name in batch["case_name"]]
            image = batch["image_slabs"].to(device)
            metadata = batch["metadata"].to(device)
            valid = batch["valid_elements"].to(device)
            positions = batch["positions"][0].to(device)
            output = teacher(image, metadata, valid)
            probabilities = torch.stack(
                [torch.from_numpy(np.load(paths[name])).float() for name in names]
            ).to(device)
            label_batch = torch.stack([labels[name] for name in names]).to(device)
            prediction_profiles = sampled_slice_profiles(probabilities, positions)
            target_profiles = batch["slice_profiles"].to(device)
            residual = output.slice_profiles - prediction_profiles
            descriptors = soft_descriptors(probabilities)
            lower = output.descriptor_quantiles[..., 0]
            upper = output.descriptor_quantiles[..., 2]
            scale = ((upper - lower) / 2).clamp_min(0.02)
            descriptor_violation = (
                torch.relu(lower - descriptors) + torch.relu(descriptors - upper)
            ) / scale
            case_uncertainty, slice_uncertainty = entropy_features(probabilities, positions)
            case_relationship = torch.cat(
                (
                    residual.flatten(start_dim=1),
                    residual.abs().flatten(start_dim=1),
                    descriptor_violation,
                    descriptors,
                ),
                dim=1,
            )
            position_feature = metadata
            slice_uncertainty_block = torch.cat(
                (
                    slice_uncertainty.unsqueeze(-1),
                    prediction_profiles,
                    position_feature,
                ),
                dim=2,
            )
            slice_relationship = torch.cat(
                (
                    residual,
                    residual.abs(),
                    prediction_profiles,
                    output.slice_profiles,
                    position_feature,
                ),
                dim=2,
            )
            case_combined = torch.cat((case_uncertainty, output.case_embedding, case_relationship), dim=1)
            slice_combined = torch.cat(
                (slice_uncertainty_block, output.element_embeddings, slice_relationship), dim=2
            )

            case_names.extend(names)
            case_targets.extend((1.0 - foreground_dice(probabilities, label_batch)).cpu().tolist())
            slice_targets.append(slice_dice_error(probabilities, label_batch, positions).cpu().numpy())
            slice_correction_targets.append((target_profiles - prediction_profiles).cpu().numpy())
            slice_teacher_corrections.append(residual.cpu().numpy())
            slice_prediction_profiles.append(prediction_profiles.cpu().numpy())
            case_feature_blocks["uncertainty"].append(case_uncertainty.cpu().numpy())
            case_feature_blocks["cst_embedding"].append(output.case_embedding.cpu().numpy())
            case_feature_blocks["relationship_residuals"].append(case_relationship.cpu().numpy())
            case_feature_blocks["combined"].append(case_combined.cpu().numpy())
            slice_feature_blocks["uncertainty"].append(slice_uncertainty_block.cpu().numpy())
            slice_feature_blocks["cst_embedding"].append(
                torch.cat((output.element_embeddings, position_feature), dim=2).cpu().numpy()
            )
            slice_feature_blocks["relationship_residuals"].append(slice_relationship.cpu().numpy())
            slice_feature_blocks["combined"].append(slice_combined.cpu().numpy())
            for name in names:
                slice_groups.extend([name] * len(positions))

    case_features = {name: np.concatenate(blocks, axis=0) for name, blocks in case_feature_blocks.items()}
    slice_features = {
        name: np.concatenate(blocks, axis=0).reshape(-1, blocks[0].shape[-1])
        for name, blocks in slice_feature_blocks.items()
    }
    return {
        "case_names": np.asarray(case_names),
        "case_target": np.asarray(case_targets, dtype=float),
        "case_features": case_features,
        "slice_groups": np.asarray(slice_groups),
        "slice_target": np.concatenate(slice_targets, axis=0).reshape(-1),
        "slice_correction_target": np.concatenate(slice_correction_targets, axis=0).reshape(-1, 2),
        "slice_teacher_correction": np.concatenate(slice_teacher_corrections, axis=0).reshape(-1, 2),
        "slice_prediction_profile": np.concatenate(slice_prediction_profiles, axis=0).reshape(-1, 2),
        "slice_features": slice_features,
        "positions_per_case": args.set_size,
    }


def evaluate_feature_set(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    top_count: int,
) -> dict[str, Any]:
    prediction, selections = repeated_nested_predictions(
        x,
        y,
        groups,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        repeats=args.repeats,
        alphas=list(args.alphas),
        seed=seed,
    )
    return {
        "feature_count": int(x.shape[1]),
        "metrics": regression_metrics(y, prediction, top_count=top_count),
        "alpha_histogram": {
            str(alpha): sum(item["alpha"] == alpha for item in selections) for alpha in args.alphas
        },
        "predictions": prediction.tolist(),
    }


def aggregate_seed_metrics(seed_reports: list[dict[str, Any]], level: str) -> dict[str, Any]:
    output = {}
    feature_names = seed_reports[0][level].keys()
    for feature in feature_names:
        metrics = [report[level][feature]["metrics"] for report in seed_reports]
        output[feature] = {}
        for name in ("pearson", "spearman", "mae", "rmse", "r2", "top_error_recall"):
            values = [item[name] for item in metrics if item[name] is not None]
            output[feature][name] = {
                "mean": None if not values else mean(values),
                "std": None if not values else pstdev(values),
                "per_seed": values,
            }
    return output


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Cross-fitted CST error-risk probe",
        "",
        f"Repeated nested grouped CV: {report['config']['repeats']} repeats, "
        f"{report['config']['outer_folds']} outer folds, {report['config']['inner_folds']} inner folds.",
    ]
    for level in ("case", "slice"):
        lines.extend(
            (
                "",
                f"## {level.title()} error",
                "",
                "| Features | Pearson r | Spearman rho | MAE | R2 | Top-error recall |",
                "|---|---:|---:|---:|---:|---:|",
            )
        )
        for feature, metrics in report["aggregate"][level].items():
            def show(name: str) -> str:
                value = metrics[name]["mean"]
                std = metrics[name]["std"]
                return "nan" if value is None else f"{value:.3f} ± {std:.3f}"

            lines.append(
                f"| {feature} | {show('pearson')} | {show('spearman')} | {show('mae')} | "
                f"{show('r2')} | {show('top_error_recall')} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    from semantic_constraints.cst_teacher.train_teacher import resolve_device, seed_everything

    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_reports = []
    reference_names = None
    for checkpoint_index, checkpoint in enumerate(args.checkpoints):
        print(json.dumps({"stage": "feature_extraction", "checkpoint": str(checkpoint)}), flush=True)
        features = extract_seed_features(checkpoint, args, device)
        if reference_names is None:
            reference_names = features["case_names"]
        elif not np.array_equal(reference_names, features["case_names"]):
            raise ValueError("checkpoint feature extraction produced inconsistent case order")
        np.savez_compressed(
            args.output_dir / f"risk_features_seed_{checkpoint_index}.npz",
            case_names=features["case_names"],
            case_target=features["case_target"],
            slice_groups=features["slice_groups"],
            slice_target=features["slice_target"],
            slice_correction_target=features["slice_correction_target"],
            slice_teacher_correction=features["slice_teacher_correction"],
            slice_prediction_profile=features["slice_prediction_profile"],
            **{f"case_{name}": value for name, value in features["case_features"].items()},
            **{f"slice_{name}": value for name, value in features["slice_features"].items()},
        )
        case_results, slice_results = {}, {}
        case_groups = features["case_names"]
        for feature_index, (name, values) in enumerate(features["case_features"].items()):
            case_results[name] = evaluate_feature_set(
                values,
                features["case_target"],
                case_groups,
                args,
                args.seed + 10000 * checkpoint_index + feature_index,
                top_count=10,
            )
            print(
                json.dumps(
                    {"seed": checkpoint_index, "level": "case", "features": name, **case_results[name]["metrics"]}
                ),
                flush=True,
            )
        for feature_index, (name, values) in enumerate(features["slice_features"].items()):
            slice_results[name] = evaluate_feature_set(
                values,
                features["slice_target"],
                features["slice_groups"],
                args,
                args.seed + 10000 * checkpoint_index + 100 + feature_index,
                top_count=max(1, len(features["slice_target"]) // 10),
            )
            print(
                json.dumps(
                    {"seed": checkpoint_index, "level": "slice", "features": name, **slice_results[name]["metrics"]}
                ),
                flush=True,
            )
        seed_reports.append(
            {
                "seed_index": checkpoint_index,
                "checkpoint": str(checkpoint.resolve()),
                "case_names": features["case_names"].tolist(),
                "case_targets": features["case_target"].tolist(),
                "case": case_results,
                "slice": slice_results,
            }
        )

    report = {
        "schema": "semantic_constraints.cst_teacher.risk_probe.v1",
        "config": {
            "outer_folds": args.outer_folds,
            "inner_folds": args.inner_folds,
            "repeats": args.repeats,
            "alphas": args.alphas,
            "grouping": "patient; no slices from a patient cross folds",
        },
        "seed_reports": seed_reports,
        "aggregate": {
            "case": aggregate_seed_metrics(seed_reports, "case"),
            "slice": aggregate_seed_metrics(seed_reports, "slice"),
        },
        "warning": (
            "Fold-0 discovery analysis only. Probe targets use validation labels under repeated nested grouped CV; "
            "freeze all choices before evaluating another fold."
        ),
    }
    json_path = args.output_dir / "risk_probe_report.json"
    md_path = args.output_dir / "risk_probe_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"output": str(json_path), "markdown": str(md_path)}))


if __name__ == "__main__":
    main()
