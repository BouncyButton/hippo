"""Discover local A/P-cut phenotypes around supervised training landmarks.

The annotated cut selects a local feature vector in each training case. The
clustering itself sees no anterior/posterior class identity. Outer validation
cuts remain hidden for deployable predictions; an explicitly named oracle
assignment is retained only to diagnose whether gating is the limiting step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

from .audit_cluster_phenotypes import (
    CASE_FEATURE_NAMES,
    Clusterer,
    bootstrap_stability,
    select_cluster_count,
)
from .audit_refined import (
    LoadedCase,
    _candidate_matrix,
    _fit_predict,
    _load_cases,
    _median_prediction,
    _metrics,
    _training_position_statistics,
)


ALPHA_GRID = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/Dataset101_MSD"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/experiments/uncal_cut_centered_phenotypes_20260921"),
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=40)
    return parser.parse_args()


def candidate_features(case: LoadedCase) -> np.ndarray:
    matrix, names = _candidate_matrix(
        case,
        "compact_geometry_image_plus_soft_position",
        position_statistics=(0.5, 0.1),
    )
    if tuple(names[:-1]) != CASE_FEATURE_NAMES or names[-1] != "position__soft_log_prior":
        raise ValueError("compact feature schema drift")
    return matrix[:, :-1]


def target_row(case: LoadedCase) -> int:
    return case.geometry.candidates.index(case.geometry.target_cut)


def relative_cut(case: LoadedCase) -> float:
    geometry = case.geometry
    return float(
        (geometry.target_cut - geometry.low) / max(geometry.high - geometry.low, 1)
    )


def fit_prototypes(
    train_names: list[str],
    features: dict[str, np.ndarray],
    cases: dict[str, LoadedCase],
    random_state: int,
) -> tuple[Clusterer, np.ndarray, int, dict[str, float], float]:
    target_matrix = np.stack(
        [features[name][target_row(cases[name])] for name in train_names]
    )
    n_clusters, silhouettes = select_cluster_count(target_matrix, random_state)
    clusterer = Clusterer(n_clusters, random_state).fit(target_matrix)
    labels = clusterer.predict(target_matrix)
    embedded = clusterer.transform(target_matrix)
    if clusterer.kmeans is None:
        raise RuntimeError("prototype model is not fitted")
    distances = np.square(
        embedded[:, None, :] - clusterer.kmeans.cluster_centers_[None, :, :]
    ).sum(axis=2)
    scale = float(max(np.median(distances.min(axis=1)), 1e-6))
    return clusterer, labels, n_clusters, silhouettes, scale


def prototype_prediction(
    name: str,
    train_names: list[str],
    clusterer: Clusterer,
    distance_scale: float,
    alpha: float,
    features: dict[str, np.ndarray],
    cases: dict[str, LoadedCase],
) -> int:
    if clusterer.kmeans is None:
        raise RuntimeError("prototype model is not fitted")
    embedded = clusterer.transform(features[name])
    squared_distance = np.square(
        embedded[:, None, :] - clusterer.kmeans.cluster_centers_[None, :, :]
    ).sum(axis=2)
    prototype_score = -squared_distance.min(axis=1) / distance_scale
    median, robust_scale = _training_position_statistics(train_names, cases)
    geometry = cases[name].geometry
    candidates = np.asarray(geometry.candidates)
    relative = (candidates - geometry.low) / max(geometry.high - geometry.low, 1)
    position_score = -0.5 * np.square((relative - median) / robust_scale)
    return int(candidates[np.argmax(prototype_score + alpha * position_score)])


def tune_alpha(
    outer_train: list[str],
    features: dict[str, np.ndarray],
    cases: dict[str, LoadedCase],
    random_state: int,
) -> tuple[float, dict[str, float]]:
    names = np.asarray(sorted(outer_train))
    splitter = KFold(n_splits=3, shuffle=True, random_state=random_state)
    errors = {alpha: [] for alpha in ALPHA_GRID}
    for inner_fold, (train_indices, validation_indices) in enumerate(splitter.split(names)):
        train_names = names[train_indices].tolist()
        validation_names = names[validation_indices].tolist()
        clusterer, _, _, _, distance_scale = fit_prototypes(
            train_names,
            features,
            cases,
            random_state * 10 + inner_fold,
        )
        for alpha in ALPHA_GRID:
            for name in validation_names:
                prediction = prototype_prediction(
                    name,
                    train_names,
                    clusterer,
                    distance_scale,
                    alpha,
                    features,
                    cases,
                )
                errors[alpha].append(abs(prediction - cases[name].geometry.target_cut))
    mean_errors = {str(alpha): float(np.mean(values)) for alpha, values in errors.items()}
    selected = min(ALPHA_GRID, key=lambda alpha: (mean_errors[str(alpha)], alpha))
    return float(selected), mean_errors


def assigned_cluster(
    name: str,
    cut: int,
    clusterer: Clusterer,
    features: dict[str, np.ndarray],
    cases: dict[str, LoadedCase],
) -> int:
    row = cases[name].geometry.candidates.index(cut)
    return int(clusterer.predict(features[name][row : row + 1])[0])


def expert_predictions(
    train_names: list[str],
    validation_names: list[str],
    train_labels: np.ndarray,
    validation_labels: dict[str, int],
    global_predictions: dict[str, int],
    cases: dict[str, LoadedCase],
    n_clusters: int,
) -> dict[str, int]:
    result = {}
    for label in range(n_clusters):
        cluster_train = [
            name
            for name, assigned in zip(train_names, train_labels, strict=True)
            if assigned == label
        ]
        cluster_validation = [
            name for name in validation_names if validation_labels[name] == label
        ]
        if not cluster_validation:
            continue
        if len(cluster_train) < 24:
            result.update({name: global_predictions[name] for name in cluster_validation})
            continue
        predictions, _, _ = _fit_predict(
            cluster_train,
            cluster_validation,
            cases,
            "compact_geometry_image_plus_soft_position",
            penalty="l2",
        )
        result.update(predictions)
    return result


def describe_clusters(
    train_names: list[str],
    target_matrix: np.ndarray,
    labels: np.ndarray,
    clusterer: Clusterer,
    cases: dict[str, LoadedCase],
) -> list[dict]:
    if clusterer.kmeans is None:
        raise RuntimeError("prototype model is not fitted")
    embedded = clusterer.transform(target_matrix)
    standardized = clusterer.scaler.transform(target_matrix)
    result = []
    for label in range(clusterer.n_clusters):
        indices = np.flatnonzero(labels == label)
        distances = np.linalg.norm(
            embedded[indices] - clusterer.kmeans.cluster_centers_[label], axis=1
        )
        representatives = indices[np.argsort(distances)[:4]]
        cuts = np.asarray([relative_cut(cases[train_names[index]]) for index in indices])
        shifts = standardized[indices].mean(axis=0)
        top = np.argsort(np.abs(shifts))[::-1][:5]
        result.append(
            {
                "cluster": label,
                "size": int(indices.size),
                "relative_cut_median": float(np.median(cuts)),
                "relative_cut_iqr": [
                    float(np.quantile(cuts, 0.25)),
                    float(np.quantile(cuts, 0.75)),
                ],
                "representative_cases": [train_names[index] for index in representatives],
                "largest_standardized_feature_shifts": [
                    {"feature": CASE_FEATURE_NAMES[index], "mean_shift": float(shifts[index])}
                    for index in top
                ],
            }
        )
    return result


def paired_comparison(
    reference: dict[str, int],
    candidate: dict[str, int],
    cases: dict[str, LoadedCase],
    seed: int,
) -> dict[str, float | int | list[float]]:
    names = sorted(cases)
    reference_errors = np.asarray(
        [abs(reference[name] - cases[name].geometry.target_cut) for name in names]
    )
    candidate_errors = np.asarray(
        [abs(candidate[name] - cases[name].geometry.target_cut) for name in names]
    )
    difference = candidate_errors - reference_errors
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray(
        [
            difference[rng.integers(0, len(difference), len(difference))].mean()
            for _ in range(10000)
        ]
    )
    return {
        "candidate_minus_reference_mae": float(difference.mean()),
        "bootstrap_95_ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "candidate_better_cases": int(np.count_nonzero(difference < 0)),
        "equal_cases": int(np.count_nonzero(difference == 0)),
        "candidate_worse_cases": int(np.count_nonzero(difference > 0)),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = _load_cases(args.dataset_root)
    splits = json.loads((args.dataset_root / "splits_final.json").read_text())
    features = {name: candidate_features(case) for name, case in cases.items()}
    predictions = {
        "global_median": {},
        "global_compact": {},
        "prototype_no_position": {},
        "prototype_tuned_position": {},
        "predicted_gate_cluster_expert": {},
        "oracle_gate_cluster_expert_diagnostic": {},
    }
    fold_reports = []
    assignments = {}

    for fold_index, split in enumerate(splits):
        train_names = list(split["train"])
        validation_names = list(split["val"])
        clusterer, train_labels, n_clusters, silhouettes, distance_scale = fit_prototypes(
            train_names, features, cases, fold_index
        )
        target_matrix = np.stack(
            [features[name][target_row(cases[name])] for name in train_names]
        )
        tuned_alpha, inner_alpha_mae = tune_alpha(
            train_names, features, cases, fold_index
        )
        global_predictions, _, _ = _fit_predict(
            train_names,
            validation_names,
            cases,
            "compact_geometry_image_plus_soft_position",
            penalty="l2",
        )
        predictions["global_compact"].update(global_predictions)
        for name in validation_names:
            predictions["global_median"][name] = _median_prediction(
                train_names, cases[name], cases
            )
            predictions["prototype_no_position"][name] = prototype_prediction(
                name,
                train_names,
                clusterer,
                distance_scale,
                0.0,
                features,
                cases,
            )
            predictions["prototype_tuned_position"][name] = prototype_prediction(
                name,
                train_names,
                clusterer,
                distance_scale,
                tuned_alpha,
                features,
                cases,
            )

        predicted_labels = {
            name: assigned_cluster(
                name, global_predictions[name], clusterer, features, cases
            )
            for name in validation_names
        }
        oracle_labels = {
            name: assigned_cluster(
                name, cases[name].geometry.target_cut, clusterer, features, cases
            )
            for name in validation_names
        }
        predictions["predicted_gate_cluster_expert"].update(
            expert_predictions(
                train_names,
                validation_names,
                train_labels,
                predicted_labels,
                global_predictions,
                cases,
                n_clusters,
            )
        )
        predictions["oracle_gate_cluster_expert_diagnostic"].update(
            expert_predictions(
                train_names,
                validation_names,
                train_labels,
                oracle_labels,
                global_predictions,
                cases,
                n_clusters,
            )
        )
        for name in validation_names:
            assignments[name] = {
                "fold": fold_index,
                "predicted_gate_cluster": predicted_labels[name],
                "oracle_target_cluster_diagnostic": oracle_labels[name],
            }

        embedded = clusterer.transform(target_matrix)
        fold_reports.append(
            {
                "fold": fold_index,
                "selected_clusters": n_clusters,
                "silhouette_by_count": silhouettes,
                "pca_explained_variance": float(
                    clusterer.pca.explained_variance_ratio_.sum()
                ),
                "bootstrap_stability": bootstrap_stability(
                    embedded,
                    train_labels,
                    n_clusters,
                    args.bootstrap_repeats,
                    10000 + fold_index * 1000,
                ),
                "selected_position_alpha": tuned_alpha,
                "inner_alpha_mae": inner_alpha_mae,
                "training_clusters": describe_clusters(
                    train_names,
                    target_matrix,
                    train_labels,
                    clusterer,
                    cases,
                ),
                "validation_predicted_gate_sizes": {
                    str(label): sum(value == label for value in predicted_labels.values())
                    for label in range(n_clusters)
                },
                "validation_gate_agreement_with_oracle": float(
                    np.mean(
                        [predicted_labels[name] == oracle_labels[name] for name in validation_names]
                    )
                ),
            }
        )

    if any(set(method) != set(cases) for method in predictions.values()):
        raise ValueError("five-fold predictions must cover all cases")
    per_case = {
        name: {
            "target": int(cases[name].geometry.target_cut),
            "predictions": {method: int(values[name]) for method, values in predictions.items()},
            "absolute_errors": {
                method: abs(int(values[name]) - int(cases[name].geometry.target_cut))
                for method, values in predictions.items()
            },
            **assignments[name],
        }
        for name in sorted(cases)
    }
    summary = {
        "schema": "uncal_cut_centered_phenotypes.v1",
        "case_count": len(cases),
        "method": {
            "training_alignment": "annotated A/P cut",
            "cluster_features": list(CASE_FEATURE_NAMES),
            "ap_class_identity_visible_to_clustering": False,
            "outer_validation_cut_visible_to_deployable_methods": False,
            "oracle_method_is_deployable": False,
        },
        "metrics": {name: _metrics(values, cases) for name, values in predictions.items()},
        "paired_vs_global_compact": {
            name: paired_comparison(
                predictions["global_compact"], values, cases, 20260921 + index
            )
            for index, (name, values) in enumerate(predictions.items())
            if name != "global_compact"
        },
        "folds": fold_reports,
        "cases": per_case,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                "metrics": summary["metrics"],
                "paired_vs_global_compact": summary["paired_vs_global_compact"],
                "folds": fold_reports,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
