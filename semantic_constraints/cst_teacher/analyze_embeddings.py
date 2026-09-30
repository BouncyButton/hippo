#!/usr/bin/env python3
"""Discover stable, interpretable morphology groups in frozen CST embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from semantic_constraints.cst_teacher.descriptors import DESCRIPTOR_NAMES  # noqa: E402
from semantic_constraints.cst_teacher.model import CSTDescriptorTeacher  # noqa: E402


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = values.max(axis=axis, keepdims=True)
    return maximum + np.log(np.exp(values - maximum).sum(axis=axis, keepdims=True))


def _mixture_memberships(model: dict[str, np.ndarray | float], values: np.ndarray) -> np.ndarray:
    means = np.asarray(model["means"])
    variances = np.asarray(model["variances"])
    weights = np.asarray(model["weights"])
    differences = values[:, None, :] - means[None, :, :]
    log_probabilities = np.log(np.clip(weights, 1e-12, None))[None, :] - 0.5 * (
        np.log(2.0 * np.pi * variances)[None, :, :]
        + differences**2 / variances[None, :, :]
    ).sum(axis=2)
    normalizer = _logsumexp(log_probabilities, axis=1)
    return np.exp(log_probabilities - normalizer)


def _fit_diagonal_mixture(
    values: np.ndarray,
    components: int,
    *,
    seed: int,
    initializations: int = 8,
    regularization: float = 1e-5,
    max_iterations: int = 300,
) -> dict[str, np.ndarray | float]:
    """Small deterministic NumPy EM implementation, avoiding mixed OpenMP runtimes."""

    sample_count, dimensions = values.shape
    if not 1 < components < sample_count:
        raise ValueError("components must be between one and the sample count")
    best: dict[str, np.ndarray | float] | None = None
    rng = np.random.default_rng(seed)
    global_variance = values.var(axis=0) + regularization
    for _ in range(initializations):
        means = values[rng.choice(sample_count, components, replace=False)].copy()
        variances = np.broadcast_to(global_variance, (components, dimensions)).copy()
        weights = np.full(components, 1.0 / components)
        previous = -np.inf
        for _ in range(max_iterations):
            model = {"means": means, "variances": variances, "weights": weights}
            memberships = _mixture_memberships(model, values)
            counts = memberships.sum(axis=0).clip(min=1e-8)
            weights = counts / sample_count
            means = memberships.T @ values / counts[:, None]
            second_moment = memberships.T @ (values**2) / counts[:, None]
            variances = (second_moment - means**2).clip(min=regularization)

            differences = values[:, None, :] - means[None, :, :]
            log_probabilities = np.log(weights)[None, :] - 0.5 * (
                np.log(2.0 * np.pi * variances)[None, :, :]
                + differences**2 / variances[None, :, :]
            ).sum(axis=2)
            likelihood = float(_logsumexp(log_probabilities, axis=1).sum())
            if likelihood - previous < 1e-6:
                break
            previous = likelihood
        parameters = (components - 1) + 2 * components * dimensions
        bic = parameters * np.log(sample_count) - 2.0 * likelihood
        candidate = {
            "means": means,
            "variances": variances,
            "weights": weights,
            "log_likelihood": likelihood,
            "bic": float(bic),
        }
        if best is None or likelihood > float(best["log_likelihood"]):
            best = candidate
    if best is None:
        raise RuntimeError("mixture fitting produced no candidate")
    return best


def _adjusted_rand_index(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("cluster assignments must be equally shaped vectors")
    first_ids, first_inverse = np.unique(first, return_inverse=True)
    second_ids, second_inverse = np.unique(second, return_inverse=True)
    contingency = np.zeros((len(first_ids), len(second_ids)), dtype=np.int64)
    np.add.at(contingency, (first_inverse, second_inverse), 1)

    def pairs(counts: np.ndarray) -> float:
        return float((counts * (counts - 1) / 2).sum())

    observed = pairs(contingency)
    row_pairs = pairs(contingency.sum(axis=1))
    column_pairs = pairs(contingency.sum(axis=0))
    total_pairs = first.size * (first.size - 1) / 2
    if total_pairs == 0:
        return 1.0
    expected = row_pairs * column_pairs / total_pairs
    maximum = 0.5 * (row_pairs + column_pairs)
    denominator = maximum - expected
    return 1.0 if abs(denominator) < 1e-12 and observed == maximum else (observed - expected) / denominator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pkl", type=Path, default=REPO_ROOT / "datasets/Dataset101_MSD/msd_hippocampus_full.pkl")
    parser.add_argument("--splits-json", type=Path, default=REPO_ROOT / "datasets/Dataset101_MSD/splits_final.json")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "semantic_constraints/cst_teacher/runs/analysis")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-clusters", type=int, default=4)
    parser.add_argument("--minimum-cluster-size", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--max-train-cases", type=int, default=0)
    parser.add_argument("--max-val-cases", type=int, default=0)
    return parser.parse_args()


def load_descriptor_teacher(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[CSTDescriptorTeacher, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if payload.get("schema") != "semantic_constraints.cst_teacher.v1":
        raise ValueError("unsupported CST teacher checkpoint schema")
    config = payload["config"]
    if tuple(payload["descriptor_names"]) != DESCRIPTOR_NAMES:
        raise ValueError("checkpoint descriptor schema does not match this source tree")
    teacher = CSTDescriptorTeacher(
        slab_channels=int(config["slab_depth"]),
        channels=tuple(config["channels"]),
        heads=int(config["heads"]),
    ).to(device)
    teacher.load_state_dict(payload["descriptor_teacher_state_dict"])
    teacher.eval()
    return teacher, payload


@torch.no_grad()
def extract_embeddings(
    teacher: CSTDescriptorTeacher,
    loader: Any,
    device: torch.device,
) -> dict[str, np.ndarray]:
    names: list[str] = []
    embeddings, targets, quantiles = [], [], []
    target_profiles, predicted_profiles = [], []
    for batch in loader:
        image_slabs = batch["image_slabs"].to(device)
        metadata = batch["metadata"].to(device)
        valid = batch["valid_elements"].to(device)
        output = teacher(image_slabs, metadata, valid)
        names.extend(str(name) for name in batch["case_name"])
        embeddings.append(output.case_embedding.cpu().numpy())
        targets.append(batch["descriptors"].numpy())
        quantiles.append(output.descriptor_quantiles.cpu().numpy())
        target_profiles.append(batch["slice_profiles"].numpy())
        predicted_profiles.append(output.slice_profiles.cpu().numpy())
    return {
        "case_names": np.asarray(names),
        "embeddings": np.concatenate(embeddings),
        "descriptor_targets": np.concatenate(targets),
        "descriptor_quantiles": np.concatenate(quantiles),
        "profile_targets": np.concatenate(target_profiles),
        "profile_predictions": np.concatenate(predicted_profiles),
    }


def _effect_rules(
    descriptors: np.ndarray,
    assignments: np.ndarray,
    cluster_count: int,
) -> list[dict[str, Any]]:
    global_mean = descriptors.mean(axis=0)
    global_scale = descriptors.std(axis=0)
    global_scale[global_scale < 1e-8] = 1.0
    rules = []
    for cluster in range(cluster_count):
        selected = descriptors[assignments == cluster]
        effects = (selected.mean(axis=0) - global_mean) / global_scale
        ordering = np.argsort(np.abs(effects))[::-1]
        distinctive = []
        for index in ordering[:3]:
            direction = "higher" if effects[index] >= 0 else "lower"
            distinctive.append(
                {
                    "descriptor": DESCRIPTOR_NAMES[index],
                    "direction_vs_training_mean": direction,
                    "standardized_effect": float(effects[index]),
                    "cluster_median": float(np.median(selected[:, index])),
                }
            )
        rules.append({"cluster": cluster, "distinctive_observations": distinctive})
    return rules


def discover_clusters(
    train_embeddings: np.ndarray,
    train_descriptors: np.ndarray,
    validation_embeddings: np.ndarray,
    validation_descriptors: np.ndarray,
    *,
    max_clusters: int = 4,
    minimum_cluster_size: int = 5,
    bootstrap_samples: int = 100,
    seed: int = 0,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Fit on training only; evaluate memberships and intervals on validation."""

    if train_embeddings.ndim != 2 or validation_embeddings.ndim != 2:
        raise ValueError("embeddings must be two-dimensional")
    if train_embeddings.shape[1] != validation_embeddings.shape[1]:
        raise ValueError("train and validation embedding dimensions differ")
    if train_descriptors.shape != (len(train_embeddings), len(DESCRIPTOR_NAMES)):
        raise ValueError("training descriptor schema mismatch")
    if validation_descriptors.shape != (len(validation_embeddings), len(DESCRIPTOR_NAMES)):
        raise ValueError("validation descriptor schema mismatch")
    if minimum_cluster_size < 1:
        raise ValueError("minimum_cluster_size must be positive")
    largest = min(max_clusters, len(train_embeddings) - 1)
    if largest < 2:
        raise ValueError("at least three training cases are required for clustering")

    scaler_mean = train_embeddings.mean(axis=0)
    scaler_scale = train_embeddings.std(axis=0)
    scaler_scale[scaler_scale < 1e-8] = 1.0
    train_scaled = (train_embeddings - scaler_mean) / scaler_scale
    validation_scaled = (validation_embeddings - scaler_mean) / scaler_scale
    candidates = []
    models: dict[int, dict[str, np.ndarray | float]] = {}
    for count in range(2, largest + 1):
        model = _fit_diagonal_mixture(train_scaled, count, seed=seed + count)
        models[count] = model
        assignments = _mixture_memberships(model, train_scaled).argmax(axis=1)
        smallest = min(int((assignments == cluster).sum()) for cluster in range(count))
        candidates.append(
            {
                "clusters": count,
                "bic": float(model["bic"]),
                "smallest_cluster": smallest,
                "eligible": smallest >= minimum_cluster_size,
            }
        )
    eligible_candidates = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible_candidates:
        raise ValueError(
            "no mixture satisfies minimum_cluster_size; reduce the maximum cluster count "
            "or explicitly lower the support requirement"
        )
    selected_count = min(eligible_candidates, key=lambda item: item["bic"])["clusters"]
    model = models[selected_count]
    train_assignments = _mixture_memberships(model, train_scaled).argmax(axis=1)
    validation_memberships = _mixture_memberships(model, validation_scaled)
    validation_assignments = validation_memberships.argmax(axis=1)

    rng = np.random.default_rng(seed)
    stability = []
    for bootstrap_index in range(bootstrap_samples):
        indices = rng.integers(0, len(train_scaled), size=len(train_scaled))
        try:
            bootstrap_model = _fit_diagonal_mixture(
                train_scaled[indices],
                selected_count,
                seed=seed + bootstrap_index + 1,
                initializations=3,
            )
            stability.append(
                _adjusted_rand_index(
                    train_assignments,
                    _mixture_memberships(bootstrap_model, train_scaled).argmax(axis=1),
                )
            )
        except ValueError:
            continue

    clusters = []
    validation_covered = np.zeros_like(validation_descriptors, dtype=bool)
    for cluster in range(selected_count):
        train_values = train_descriptors[train_assignments == cluster]
        lower, median, upper = np.quantile(train_values, (0.05, 0.5, 0.95), axis=0)
        val_mask = validation_assignments == cluster
        validation_covered[val_mask] = (
            (validation_descriptors[val_mask] >= lower)
            & (validation_descriptors[val_mask] <= upper)
        )
        clusters.append(
            {
                "cluster": cluster,
                "training_count": int(len(train_values)),
                "validation_count": int(val_mask.sum()),
                "descriptor_intervals": {
                    name: {
                        "q05": float(lower[index]),
                        "median": float(median[index]),
                        "q95": float(upper[index]),
                    }
                    for index, name in enumerate(DESCRIPTOR_NAMES)
                },
            }
        )

    report = {
        "selection": {
            "criterion": "minimum training BIC",
            "candidates": candidates,
            "selected_clusters": selected_count,
        },
        "bootstrap_stability": {
            "samples_requested": bootstrap_samples,
            "samples_completed": len(stability),
            "mean_adjusted_rand_index": float(np.mean(stability)) if stability else None,
            "q05_adjusted_rand_index": float(np.quantile(stability, 0.05)) if stability else None,
        },
        "clusters": clusters,
        "observational_rules": _effect_rules(train_descriptors, train_assignments, selected_count),
        "validation_cluster_interval_coverage": {
            name: float(validation_covered[:, index].mean())
            for index, name in enumerate(DESCRIPTOR_NAMES)
        },
        "warning": (
            "Clusters and rules are observational candidates, not training constraints. "
            "They require cross-fitting and counterfactual repair before use."
        ),
    }
    arrays = {
        "train_assignments": train_assignments,
        "validation_assignments": validation_assignments,
        "validation_memberships": validation_memberships,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "mixture_means": np.asarray(model["means"]),
        "mixture_variances": np.asarray(model["variances"]),
        "mixture_weights": np.asarray(model["weights"]),
    }
    return report, arrays


def main() -> None:
    # Loading the MONAI baseline stack is intentionally deferred so importing
    # the pure NumPy cluster-discovery functions stays lightweight and safe in
    # unit tests or notebooks.
    from semantic_constraints.cst_teacher.train_teacher import build_loaders, resolve_device

    args = parse_args()
    device = resolve_device(args.device)
    teacher, checkpoint = load_descriptor_teacher(args.checkpoint, device)
    config = checkpoint["config"]
    args.fold = int(config["fold"])
    args.spatial_size = tuple(config["spatial_size"])
    args.set_size = int(config["set_size"])
    args.slab_depth = int(config["slab_depth"])
    args.inplane_size = int(config.get("inplane_size", 64))
    args.cache_dataset = True
    train_loader, validation_loader = build_loaders(args)
    train = extract_embeddings(teacher, train_loader, device)
    validation = extract_embeddings(teacher, validation_loader, device)
    report, cluster_arrays = discover_clusters(
        train["embeddings"],
        train["descriptor_targets"],
        validation["embeddings"],
        validation["descriptor_targets"],
        max_clusters=args.max_clusters,
        minimum_cluster_size=args.minimum_cluster_size,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    report["checkpoint"] = str(args.checkpoint.resolve())
    report["fold"] = args.fold
    report["training_cases"] = len(train["case_names"])
    report["validation_cases"] = len(validation["case_names"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cluster_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.output_dir / "embeddings_and_clusters.npz",
        **{f"train_{key}": value for key, value in train.items()},
        **{f"validation_{key}": value for key, value in validation.items()},
        **cluster_arrays,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
