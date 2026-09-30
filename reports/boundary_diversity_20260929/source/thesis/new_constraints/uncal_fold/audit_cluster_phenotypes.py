"""Leakage-controlled audit of unsupervised hippocampal cut phenotypes.

Clustering sees only the whole-hippocampus foreground union and local MRI
appearance.  Anterior/posterior labels are used afterward to describe clusters
and score global versus cluster-conditional cut predictors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from .audit_refined import (
    COMPACT_GEOMETRY_FEATURES,
    COMPACT_IMAGE_FEATURES,
    LoadedCase,
    _candidate_matrix,
    _fit_predict,
    _load_cases,
    _median_prediction,
    _metrics,
)


SEQUENCE_POINTS = 16
CASE_FEATURE_NAMES = (*COMPACT_GEOMETRY_FEATURES, *COMPACT_IMAGE_FEATURES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/Dataset101_MSD"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/experiments/uncal_cluster_phenotypes_20260921"),
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=40)
    return parser.parse_args()


def case_representation(case: LoadedCase) -> np.ndarray:
    """Resample within-case normalized transition sequences to a fixed grid."""
    matrix, names = _candidate_matrix(
        case,
        "compact_geometry_image_plus_soft_position",
        position_statistics=(0.5, 0.1),
    )
    if tuple(names[:-1]) != CASE_FEATURE_NAMES or names[-1] != "position__soft_log_prior":
        raise ValueError("compact feature schema drift")
    matrix = matrix[:, :-1]
    source = np.linspace(0.0, 1.0, matrix.shape[0])
    target = np.linspace(0.0, 1.0, SEQUENCE_POINTS)
    resampled = np.column_stack(
        [np.interp(target, source, matrix[:, column]) for column in range(matrix.shape[1])]
    )
    return resampled.ravel()


class Clusterer:
    def __init__(self, n_clusters: int, random_state: int):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca: PCA | None = None
        self.kmeans: KMeans | None = None

    def fit(self, matrix: np.ndarray) -> "Clusterer":
        scaled = self.scaler.fit_transform(matrix)
        n_components = min(12, matrix.shape[0] - 1, matrix.shape[1])
        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        embedded = self.pca.fit_transform(scaled)
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=50,
            random_state=self.random_state,
        ).fit(embedded)
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("clusterer is not fitted")
        return self.pca.transform(self.scaler.transform(matrix))

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        if self.kmeans is None:
            raise RuntimeError("clusterer is not fitted")
        return self.kmeans.predict(self.transform(matrix))


def select_cluster_count(matrix: np.ndarray, random_state: int) -> tuple[int, dict[str, float]]:
    scores = {}
    for count in range(2, 6):
        clusterer = Clusterer(count, random_state).fit(matrix)
        labels = clusterer.predict(matrix)
        scores[str(count)] = float(silhouette_score(clusterer.transform(matrix), labels))
    selected = max((int(count) for count in scores), key=lambda count: scores[str(count)])
    return selected, scores


def bootstrap_stability(
    embedded: np.ndarray,
    reference: np.ndarray,
    n_clusters: int,
    repeats: int,
    random_state: int,
) -> dict[str, float]:
    rng = np.random.default_rng(random_state)
    values = []
    sample_size = max(n_clusters * 4, int(round(0.8 * embedded.shape[0])))
    for repeat in range(repeats):
        chosen = rng.choice(embedded.shape[0], size=sample_size, replace=False)
        model = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state + repeat + 1)
        model.fit(embedded[chosen])
        values.append(adjusted_rand_score(reference, model.predict(embedded)))
    array = np.asarray(values)
    return {
        "mean_adjusted_rand": float(array.mean()),
        "median_adjusted_rand": float(np.median(array)),
        "p10_adjusted_rand": float(np.quantile(array, 0.1)),
    }


def relative_cut(case: LoadedCase) -> float:
    geometry = case.geometry
    return float(
        (geometry.target_cut - geometry.low) / max(geometry.high - geometry.low, 1)
    )


def cluster_description(
    names: list[str],
    labels: np.ndarray,
    clusterer: Clusterer,
    matrix: np.ndarray,
    cases: dict[str, LoadedCase],
) -> list[dict]:
    if clusterer.kmeans is None:
        raise RuntimeError("clusterer is not fitted")
    embedded = clusterer.transform(matrix)
    rows = []
    for label in range(clusterer.n_clusters):
        indices = np.flatnonzero(labels == label)
        distances = np.linalg.norm(
            embedded[indices] - clusterer.kmeans.cluster_centers_[label], axis=1
        )
        representative_indices = indices[np.argsort(distances)[:3]]
        cuts = np.asarray([relative_cut(cases[names[index]]) for index in indices])
        rows.append(
            {
                "cluster": label,
                "size": int(indices.size),
                "relative_cut_median": float(np.median(cuts)),
                "relative_cut_iqr": [
                    float(np.quantile(cuts, 0.25)),
                    float(np.quantile(cuts, 0.75)),
                ],
                "representative_cases": [names[index] for index in representative_indices],
            }
        )
    return rows


def cluster_median_prediction(
    train_names: list[str],
    validation_names: list[str],
    train_labels: np.ndarray,
    validation_labels: np.ndarray,
    cases: dict[str, LoadedCase],
) -> dict[str, int]:
    predictions = {}
    for label in np.unique(train_labels):
        members = [
            name for name, assigned in zip(train_names, train_labels, strict=True)
            if assigned == label
        ]
        median = float(np.median([relative_cut(cases[name]) for name in members]))
        for name, assigned in zip(validation_names, validation_labels, strict=True):
            if assigned != label:
                continue
            geometry = cases[name].geometry
            candidates = np.asarray(geometry.candidates)
            positions = (candidates - geometry.low) / max(geometry.high - geometry.low, 1)
            predictions[name] = int(candidates[np.argmin(np.abs(positions - median))])
    return predictions


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = _load_cases(args.dataset_root)
    splits = json.loads((args.dataset_root / "splits_final.json").read_text())
    representations = {name: case_representation(case) for name, case in cases.items()}

    predictions = {
        "global_median": {},
        "cluster_median": {},
        "global_compact": {},
        "cluster_compact": {},
    }
    fold_reports = []
    assignments = {}

    for fold_index, split in enumerate(splits):
        train_names = list(split["train"])
        validation_names = list(split["val"])
        train_matrix = np.stack([representations[name] for name in train_names])
        validation_matrix = np.stack([representations[name] for name in validation_names])
        n_clusters, silhouettes = select_cluster_count(train_matrix, fold_index)
        clusterer = Clusterer(n_clusters, fold_index).fit(train_matrix)
        train_labels = clusterer.predict(train_matrix)
        validation_labels = clusterer.predict(validation_matrix)

        for name, label in zip(validation_names, validation_labels, strict=True):
            assignments[name] = {"validation_fold": fold_index, "cluster": int(label)}
            predictions["global_median"][name] = _median_prediction(
                train_names, cases[name], cases
            )
        predictions["cluster_median"].update(
            cluster_median_prediction(
                train_names,
                validation_names,
                train_labels,
                validation_labels,
                cases,
            )
        )
        global_predictions, _, _ = _fit_predict(
            train_names,
            validation_names,
            cases,
            "compact_geometry_image_plus_soft_position",
            penalty="l2",
        )
        predictions["global_compact"].update(global_predictions)
        for label in range(n_clusters):
            cluster_train = [
                name for name, assigned in zip(train_names, train_labels, strict=True)
                if assigned == label
            ]
            cluster_validation = [
                name for name, assigned in zip(validation_names, validation_labels, strict=True)
                if assigned == label
            ]
            if not cluster_validation:
                continue
            if len(cluster_train) < 24:
                predictions["cluster_compact"].update(
                    {name: global_predictions[name] for name in cluster_validation}
                )
                continue
            cluster_predictions, _, _ = _fit_predict(
                cluster_train,
                cluster_validation,
                cases,
                "compact_geometry_image_plus_soft_position",
                penalty="l2",
            )
            predictions["cluster_compact"].update(cluster_predictions)

        train_embedded = clusterer.transform(train_matrix)
        fold_reports.append(
            {
                "fold": fold_index,
                "selected_clusters": n_clusters,
                "silhouette_by_count": silhouettes,
                "pca_explained_variance": float(clusterer.pca.explained_variance_ratio_.sum()),
                "bootstrap_stability": bootstrap_stability(
                    train_embedded,
                    train_labels,
                    n_clusters,
                    args.bootstrap_repeats,
                    fold_index * 1000,
                ),
                "training_clusters": cluster_description(
                    train_names, train_labels, clusterer, train_matrix, cases
                ),
                "validation_cluster_sizes": {
                    str(label): int(np.count_nonzero(validation_labels == label))
                    for label in range(n_clusters)
                },
            }
        )

    if any(set(values) != set(cases) for values in predictions.values()):
        raise ValueError("cross-validated predictions must cover every case")
    held_out_predictions = {
        name: {
            method: int(values[name])
            for method, values in predictions.items()
        }
        | {
            "target": int(cases[name].geometry.target_cut),
            "absolute_errors": {
                method: abs(int(values[name]) - int(cases[name].geometry.target_cut))
                for method, values in predictions.items()
            },
        }
        for name in sorted(cases)
    }
    global_errors = np.asarray(
        [held_out_predictions[name]["absolute_errors"]["global_compact"] for name in sorted(cases)]
    )
    cluster_errors = np.asarray(
        [held_out_predictions[name]["absolute_errors"]["cluster_compact"] for name in sorted(cases)]
    )
    difference = cluster_errors - global_errors
    rng = np.random.default_rng(20260921)
    bootstrap = np.asarray(
        [
            difference[rng.integers(0, len(difference), len(difference))].mean()
            for _ in range(10000)
        ]
    )
    paired_comparison = {
        "cluster_minus_global_mae": float(difference.mean()),
        "bootstrap_95_ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "cluster_better_cases": int(np.count_nonzero(difference < 0)),
        "equal_cases": int(np.count_nonzero(difference == 0)),
        "cluster_worse_cases": int(np.count_nonzero(difference > 0)),
    }
    summary = {
        "schema": "uncal_cluster_phenotypes.v1",
        "case_count": len(cases),
        "cluster_input": {
            "foreground": "ground-truth union of anterior and posterior labels",
            "image": "local native T1 features within and around foreground",
            "ap_labels_visible_to_clustering": False,
            "sequence_points": SEQUENCE_POINTS,
            "case_feature_names": list(CASE_FEATURE_NAMES),
        },
        "selection": "2-5 clusters selected independently per outer fold by training silhouette",
        "metrics": {name: _metrics(values, cases) for name, values in predictions.items()},
        "paired_compact_comparison": paired_comparison,
        "folds": fold_reports,
        "held_out_assignments": assignments,
        "held_out_predictions": held_out_predictions,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"metrics": summary["metrics"], "folds": fold_reports}, indent=2))


if __name__ == "__main__":
    main()
