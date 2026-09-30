"""Five-fold audit of interpretable uncal-fold descriptors on MSD Task04.

The audit is intentionally upstream of any LTN implementation.  It asks whether
the undivided hippocampal mask contains held-out information about the annotated
A/P cut.  Run from the repository root:

    .venv/bin/python -m thesis.new_constraints.uncal_fold.audit_foldedness
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .foldedness import (
    BASE_FEATURE_NAMES,
    CaseFeatures,
    extract_case_features,
    handcrafted_fold_scores,
    within_case_zscores,
)


RULE_MODES = ("max_current", "min_current", "max_delta", "min_delta", "max_abs_delta")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/Dataset101_MSD"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/experiments/uncal_foldedness_audit_20260921"),
    )
    return parser.parse_args()


def _load_cases(dataset_root: Path) -> dict[str, CaseFeatures]:
    cases: dict[str, CaseFeatures] = {}
    paths = sorted((dataset_root / "labelsTr").glob("hippocampus_*.nii.gz"))
    if not paths:
        raise FileNotFoundError(f"No Task04 labels found below {dataset_root}")
    for path in paths:
        image = nib.load(str(path))
        if nib.aff2axcodes(image.affine) != ("R", "A", "S"):
            raise ValueError(f"{path.name}: expected RAS storage orientation")
        labels = np.asarray(image.dataobj, dtype=np.uint8)
        cases[path.name.removesuffix(".nii.gz")] = extract_case_features(
            path.name.removesuffix(".nii.gz"), labels
        )
    return cases


def _candidate_features(case: CaseFeatures) -> tuple[list[int], np.ndarray, list[str]]:
    normalized = within_case_zscores(case)
    rows: list[list[float]] = []
    feature_names: list[str] = []
    if not feature_names:
        for prefix in ("current", "delta", "next_delta", "local_contrast"):
            feature_names.extend(f"{prefix}__{name}" for name in BASE_FEATURE_NAMES)

    for cut in case.candidates:
        previous = normalized[cut - 1]
        current = normalized[cut]
        following = normalized[min(cut + 1, case.high)]
        row = []
        for name in BASE_FEATURE_NAMES:
            row.append(current[name])
        for name in BASE_FEATURE_NAMES:
            row.append(current[name] - previous[name])
        for name in BASE_FEATURE_NAMES:
            row.append(following[name] - current[name])
        for name in BASE_FEATURE_NAMES:
            row.append(current[name] - 0.5 * (previous[name] + following[name]))
        rows.append(row)
    return case.candidates, np.asarray(rows, dtype=np.float64), feature_names


def _coordinate_features(case: CaseFeatures) -> np.ndarray:
    span = max(case.high - case.low, 1)
    relative = np.asarray([(cut - case.low) / span for cut in case.candidates])
    return np.column_stack((relative, relative**2, np.full_like(relative, span)))


def _model_rows(
    names: Iterable[str],
    cases: dict[str, CaseFeatures],
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, int]]]:
    feature_rows, targets, weights, keys = [], [], [], []
    for name in names:
        case = cases[name]
        candidates, shape, _ = _candidate_features(case)
        coordinate = _coordinate_features(case)
        if mode == "shape":
            matrix = shape
        elif mode == "coordinate":
            matrix = coordinate
        elif mode == "combined":
            matrix = np.column_stack((shape, coordinate))
        else:
            raise ValueError(mode)
        count = len(candidates)
        for index, cut in enumerate(candidates):
            feature_rows.append(matrix[index])
            targets.append(int(cut == case.target_cut))
            weights.append(1.0 / count)
            keys.append((name, cut))
    return (
        np.asarray(feature_rows),
        np.asarray(targets),
        np.asarray(weights),
        keys,
    )


def _predict_model(
    train_names: list[str],
    validation_names: list[str],
    cases: dict[str, CaseFeatures],
    mode: str,
    relative_gate: tuple[float, float] | None = None,
) -> tuple[dict[str, int], np.ndarray | None, list[str]]:
    x_train, y_train, weights, _ = _model_rows(train_names, cases, mode)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.25, class_weight="balanced", max_iter=4000),
    )
    model.fit(x_train, y_train, logisticregression__sample_weight=weights)
    predictions: dict[str, int] = {}
    for name in validation_names:
        case = cases[name]
        candidates, shape, shape_names = _candidate_features(case)
        coordinate = _coordinate_features(case)
        matrix = {"shape": shape, "coordinate": coordinate, "combined": np.column_stack((shape, coordinate))}[mode]
        scores = model.predict_proba(matrix)[:, 1]
        if relative_gate is not None:
            relative = coordinate[:, 0]
            allowed = (relative >= relative_gate[0]) & (relative <= relative_gate[1])
            if not allowed.any():
                centre = 0.5 * (relative_gate[0] + relative_gate[1])
                allowed[int(np.argmin(np.abs(relative - centre)))] = True
            scores = np.where(allowed, scores, -np.inf)
        predictions[name] = int(candidates[int(np.argmax(scores))])

    coefficients = None
    names = []
    if mode in {"shape", "combined"}:
        _, _, shape_names = _candidate_features(cases[train_names[0]])
        names = list(shape_names)
        if mode == "combined":
            names.extend(("coordinate__relative", "coordinate__relative_squared", "coordinate__span"))
        coefficients = model.named_steps["logisticregression"].coef_[0]
    return predictions, coefficients, names


def _rule_score(case: CaseFeatures, feature: str, mode: str) -> tuple[list[int], np.ndarray]:
    candidates = case.candidates
    normalized = within_case_zscores(case)
    current = np.asarray([normalized[cut][feature] for cut in candidates])
    previous = np.asarray([normalized[cut - 1][feature] for cut in candidates])
    if mode == "max_current":
        score = current
    elif mode == "min_current":
        score = -current
    elif mode == "max_delta":
        score = current - previous
    elif mode == "min_delta":
        score = previous - current
    elif mode == "max_abs_delta":
        score = np.abs(current - previous)
    else:
        raise ValueError(mode)
    return candidates, score


def _rule_prediction(
    case: CaseFeatures,
    rule: tuple[str, str],
    relative_gate: tuple[float, float] | None = None,
) -> int:
    candidates, score = _rule_score(case, *rule)
    if relative_gate is not None:
        relative = (np.asarray(candidates) - case.low) / max(case.high - case.low, 1)
        allowed = (relative >= relative_gate[0]) & (relative <= relative_gate[1])
        if not allowed.any():
            centre = 0.5 * (relative_gate[0] + relative_gate[1])
            allowed[int(np.argmin(np.abs(relative - centre)))] = True
        score = np.where(allowed, score, -np.inf)
    return int(candidates[int(np.argmax(score))])


def _select_rule(
    train_names: list[str],
    cases: dict[str, CaseFeatures],
    relative_gate: tuple[float, float] | None = None,
) -> tuple[str, str]:
    choices = [(feature, mode) for feature in BASE_FEATURE_NAMES for mode in RULE_MODES]
    ranked = []
    for rule in choices:
        errors = np.asarray(
            [
                abs(
                    _rule_prediction(cases[name], rule, relative_gate)
                    - cases[name].target_cut
                )
                for name in train_names
            ]
        )
        ranked.append((float(errors.mean()), -float((errors <= 1).mean()), rule))
    return min(ranked)[2]


def _handcrafted_prediction(
    case: CaseFeatures,
    relative_gate: tuple[float, float] | None = None,
) -> int:
    scores = handcrafted_fold_scores(case)
    candidates = case.candidates
    transition = np.asarray([scores[cut] - scores[cut - 1] for cut in candidates])
    if relative_gate is not None:
        relative = (np.asarray(candidates) - case.low) / max(case.high - case.low, 1)
        allowed = (relative >= relative_gate[0]) & (relative <= relative_gate[1])
        if not allowed.any():
            centre = 0.5 * (relative_gate[0] + relative_gate[1])
            allowed[int(np.argmin(np.abs(relative - centre)))] = True
        transition = np.where(allowed, transition, -np.inf)
    return int(candidates[int(np.argmax(transition))])


def _training_relative_gate(
    train_names: list[str],
    cases: dict[str, CaseFeatures],
) -> tuple[float, float]:
    """Broad 1st--99th percentile support learned from training subjects only."""

    positions = np.asarray(
        [
            (cases[name].target_cut - cases[name].low)
            / max(cases[name].high - cases[name].low, 1)
            for name in train_names
        ]
    )
    return float(np.quantile(positions, 0.01)), float(np.quantile(positions, 0.99))


def _median_relative_prediction(train: list[str], case: CaseFeatures, cases: dict[str, CaseFeatures]) -> int:
    positions = [
        (cases[name].target_cut - cases[name].low) / max(cases[name].high - cases[name].low, 1)
        for name in train
    ]
    expected = float(np.median(positions))
    candidates = np.asarray(case.candidates)
    relative = (candidates - case.low) / max(case.high - case.low, 1)
    return int(candidates[int(np.argmin(np.abs(relative - expected)))])


def _metrics(predictions: dict[str, int], cases: dict[str, CaseFeatures]) -> dict[str, float]:
    errors = np.asarray([abs(predictions[name] - cases[name].target_cut) for name in predictions])
    return {
        "count": int(errors.size),
        "mae_slices": float(errors.mean()),
        "median_absolute_error": float(np.median(errors)),
        "exact_fraction": float((errors == 0).mean()),
        "within_1_fraction": float((errors <= 1).mean()),
        "within_2_fraction": float((errors <= 2).mean()),
        "p90_absolute_error": float(np.quantile(errors, 0.9)),
    }


def _plot_case(case: CaseFeatures, predictions: dict[str, int], output: Path) -> None:
    slices = np.arange(case.low, case.high + 1)
    normalized = within_case_zscores(case)
    fold = handcrafted_fold_scores(case)
    figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(slices, [normalized[y]["area"] for y in slices], marker="o", label="area")
    axes[0].plot(slices, [normalized[y]["aspect_height_over_width"] for y in slices], marker="o", label="height/width")
    axes[0].legend(loc="best")
    axes[0].set_ylabel("within-case z")
    axes[1].plot(slices, [normalized[y]["top_asymmetry_abs"] for y in slices], marker="o", label="top asymmetry")
    axes[1].plot(slices, [normalized[y]["top_notch_max"] for y in slices], marker="o", label="top notch")
    axes[1].plot(slices, [normalized[y]["top_roughness"] for y in slices], marker="o", label="top roughness")
    axes[1].legend(loc="best")
    axes[1].set_ylabel("within-case z")
    axes[2].plot(slices, [fold[y] for y in slices], marker="o", label="fixed foldedness")
    axes[2].bar(
        case.candidates,
        [fold[y] - fold[y - 1] for y in case.candidates],
        alpha=0.3,
        label="foldedness transition",
    )
    axes[2].set_ylabel("score")
    axes[2].set_xlabel("native RAS coronal y (higher = anterior)")
    axes[2].legend(loc="best")
    for axis in axes:
        axis.axvline(case.target_cut, color="black", linewidth=2, label="target")
        for index, (name, cut) in enumerate(predictions.items()):
            axis.axvline(cut, linestyle="--", alpha=0.65, color=f"C{index + 3}")
        axis.grid(alpha=0.2)
    figure.suptitle(
        f"{case.case_name}: target {case.target_cut}; "
        + ", ".join(f"{name}={cut}" for name, cut in predictions.items())
    )
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _write_csv(
    path: Path,
    predictions: dict[str, dict[str, int]],
    cases: dict[str, CaseFeatures],
    validation_fold: dict[str, int],
    atypicality: dict[str, float],
    gate_target_covered: dict[str, bool],
) -> None:
    methods = sorted(predictions)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_name",
                "validation_fold",
                "target_cut",
                "target_cost",
                "second_best_gap",
                "relative_cut_atypicality",
                "target_inside_training_gate",
                *methods,
            ],
        )
        writer.writeheader()
        for name in sorted(cases):
            case = cases[name]
            row = {
                "case_name": name,
                "validation_fold": validation_fold[name],
                "target_cut": case.target_cut,
                "target_cost": case.target_cost,
                "second_best_gap": case.second_best_gap,
                "relative_cut_atypicality": atypicality[name],
                "target_inside_training_gate": int(gate_target_covered[name]),
            }
            row.update({method: predictions[method][name] for method in methods})
            writer.writerow(row)


def _write_readme(
    path: Path,
    metrics: dict[str, dict[str, float]],
    selected_rules: list[tuple[str, str]],
    cases: dict[str, CaseFeatures],
    coefficient_summary: list[dict[str, float | str]],
    atypical_metrics: dict[str, dict[str, float]],
    gate_target_covered: dict[str, bool],
    paired_comparison: dict[str, int],
) -> None:
    ordered = sorted(metrics, key=lambda name: metrics[name]["mae_slices"])
    lines = [
        "# Uncal-foldedness discovery audit",
        "",
        "This is a five-fold, subject-held-out audit on all 260 MSD Task04 labels. Every",
        "shape feature is calculated from the undivided foreground union `(label > 0)`.",
        "Class 1/2 labels are used only to derive the best-fitting first-anterior slice",
        "against which predictions are scored. No network predictions enter this audit.",
        "",
        "## Results",
        "",
        "| method | MAE (slices) | exact | within 1 | within 2 | p90 error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ordered:
        row = metrics[name]
        lines.append(
            f"| {name} | {row['mae_slices']:.3f} | {row['exact_fraction']:.1%} | "
            f"{row['within_1_fraction']:.1%} | {row['within_2_fraction']:.1%} | "
            f"{row['p90_absolute_error']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Most atypical third",
            "",
            "Atypicality is the absolute distance between a validation subject's relative",
            "cut and the median relative cut of that fold's training subjects.",
            "",
            "| method | MAE | exact | within 1 | p90 error |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in ("median_relative", "logistic_shape_only", "logistic_shape_only_gated"):
        row = atypical_metrics[name]
        lines.append(
            f"| {name} | {row['mae_slices']:.3f} | {row['exact_fraction']:.1%} | "
            f"{row['within_1_fraction']:.1%} | {row['p90_absolute_error']:.1f} |"
        )
    lines.extend(
        [
            "",
            "`median_relative` is the no-shape control. `handcrafted_fold_transition` is a",
            "fixed mean of superior-contour asymmetry, height asymmetry, notch depth,",
            "roughness and multi-run evidence. `selected_single_descriptor` chooses one",
            "human-readable descriptor/rule using only the training subjects of each fold.",
            "The logistic models are diagnostics: they test whether the descriptor vector",
            "contains recoverable held-out signal, not whether it is already an LTN rule.",
            "Methods ending in `_gated` restrict candidates to the 1st--99th percentile",
            "of relative cut positions measured on that fold's training subjects. This",
            "broad, training-only guard tests whether catastrophic end-slice extrema were",
            "hiding useful local shape evidence; it is not a foldedness measurement.",
            "",
            f"The shape-only gated model has lower error than the median baseline in "
            f"{paired_comparison['better']} cases, equal error in {paired_comparison['equal']}, "
            f"and higher error in {paired_comparison['worse']}. The training-derived gate",
            f"contains the true validation cut in {sum(gate_target_covered.values())}/"
            f"{len(gate_target_covered)} cases ({np.mean(list(gate_target_covered.values())):.1%}).",
            "It must therefore remain a soft/support diagnostic rather than a hard anatomical",
            "axiom. In particular, `hippocampus_164` lies outside that gate.",
            "",
            "## Fold-wise selected single rules",
            "",
        ]
    )
    for fold, rule in enumerate(selected_rules):
        lines.append(f"- Fold {fold}: `{rule[1]}({rule[0]})`")
    lines.extend(
        [
            "",
            "## Strongest standardized logistic coefficients",
            "",
            "These are averaged across folds and are descriptive only.",
            "",
            "| feature | mean coefficient | mean absolute coefficient |",
            "|---|---:|---:|",
        ]
    )
    for row in coefficient_summary[:12]:
        lines.append(
            f"| `{row['feature']}` | {row['mean']:.3f} | {row['mean_absolute']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "A foldedness formulation should move forward only if a shape-based method",
            "beats the held-out median-relative baseline, with particular attention to",
            "within-one-slice accuracy. A good logistic result with a poor fixed descriptor",
            "means that shape contains signal but the proposed hand equation is not yet the",
            "right grounding. Failure of both argues for examining MRI encoder features.",
            "",
            "## Reproduction",
            "",
            "```bash",
            ".venv/bin/python -m thesis.new_constraints.uncal_fold.audit_foldedness",
            "```",
            "",
            f"Cases: {len(cases)}. All inputs were verified as stored RAS.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cases = _load_cases(args.dataset_root)
    splits = json.loads((args.dataset_root / "splits_final.json").read_text())
    all_names = set(cases)
    predictions: dict[str, dict[str, int]] = {
        "median_relative": {},
        "handcrafted_fold_transition": {},
        "handcrafted_fold_transition_gated": {},
        "selected_single_descriptor": {},
        "selected_single_descriptor_gated": {},
        "logistic_coordinate_only": {},
        "logistic_shape_only": {},
        "logistic_shape_only_gated": {},
        "logistic_shape_plus_coordinate": {},
        "logistic_shape_plus_coordinate_gated": {},
    }
    selected_rules: list[tuple[str, str]] = []
    shape_coefficients: list[tuple[np.ndarray, list[str]]] = []
    validation_fold: dict[str, int] = {}
    atypicality: dict[str, float] = {}
    gate_target_covered: dict[str, bool] = {}

    for fold_index, split in enumerate(splits):
        train = list(split["train"])
        validation = list(split["val"])
        if not set(train + validation) <= all_names:
            missing = set(train + validation) - all_names
            raise ValueError(f"Fold {fold_index} refers to missing cases: {sorted(missing)}")

        rule = _select_rule(train, cases)
        relative_gate = _training_relative_gate(train, cases)
        training_positions = np.asarray(
            [
                (cases[name].target_cut - cases[name].low)
                / max(cases[name].high - cases[name].low, 1)
                for name in train
            ]
        )
        training_median = float(np.median(training_positions))
        gated_rule = _select_rule(train, cases, relative_gate)
        selected_rules.append(rule)
        for name in validation:
            case = cases[name]
            relative_target = (case.target_cut - case.low) / max(case.high - case.low, 1)
            validation_fold[name] = fold_index
            atypicality[name] = abs(relative_target - training_median)
            gate_target_covered[name] = relative_gate[0] <= relative_target <= relative_gate[1]
            predictions["median_relative"][name] = _median_relative_prediction(train, cases[name], cases)
            predictions["handcrafted_fold_transition"][name] = _handcrafted_prediction(cases[name])
            predictions["handcrafted_fold_transition_gated"][name] = _handcrafted_prediction(
                cases[name], relative_gate
            )
            predictions["selected_single_descriptor"][name] = _rule_prediction(cases[name], rule)
            predictions["selected_single_descriptor_gated"][name] = _rule_prediction(
                cases[name], gated_rule, relative_gate
            )

        for mode, method in (
            ("coordinate", "logistic_coordinate_only"),
            ("shape", "logistic_shape_only"),
            ("combined", "logistic_shape_plus_coordinate"),
        ):
            fold_predictions, coefficients, feature_names = _predict_model(train, validation, cases, mode)
            predictions[method].update(fold_predictions)
            if mode == "shape" and coefficients is not None:
                shape_coefficients.append((coefficients, feature_names))
            if mode in {"shape", "combined"}:
                gated_method = {
                    "shape": "logistic_shape_only_gated",
                    "combined": "logistic_shape_plus_coordinate_gated",
                }[mode]
                gated_predictions, _, _ = _predict_model(
                    train,
                    validation,
                    cases,
                    mode,
                    relative_gate=relative_gate,
                )
                predictions[gated_method].update(gated_predictions)

    for method, values in predictions.items():
        if set(values) != all_names:
            raise ValueError(f"{method}: five-fold predictions do not cover every case exactly once")

    metrics = {method: _metrics(values, cases) for method, values in predictions.items()}
    atypical_count = len(cases) // 3
    atypical_names = set(
        sorted(atypicality, key=atypicality.get, reverse=True)[:atypical_count]
    )
    atypical_metrics = {
        method: _metrics(
            {name: cut for name, cut in values.items() if name in atypical_names},
            cases,
        )
        for method, values in predictions.items()
    }
    median_error = {
        name: abs(predictions["median_relative"][name] - cases[name].target_cut)
        for name in cases
    }
    shape_error = {
        name: abs(predictions["logistic_shape_only_gated"][name] - cases[name].target_cut)
        for name in cases
    }
    paired_comparison = {
        "better": sum(shape_error[name] < median_error[name] for name in cases),
        "equal": sum(shape_error[name] == median_error[name] for name in cases),
        "worse": sum(shape_error[name] > median_error[name] for name in cases),
    }
    coefficient_arrays = np.stack([item[0] for item in shape_coefficients])
    coefficient_names = shape_coefficients[0][1]
    coefficient_summary = sorted(
        [
            {
                "feature": name,
                "mean": float(coefficient_arrays[:, index].mean()),
                "mean_absolute": float(np.abs(coefficient_arrays[:, index]).mean()),
            }
            for index, name in enumerate(coefficient_names)
        ],
        key=lambda row: row["mean_absolute"],
        reverse=True,
    )

    report = {
        "dataset_root": str(args.dataset_root.resolve()),
        "case_count": len(cases),
        "orientation": "RAS; stored axis 1 increases anteriorly",
        "target": "unique best-fitting first-anterior stored slice",
        "feature_source": "ground-truth foreground union only; A/P labels used only for target scoring",
        "metrics": metrics,
        "atypical_third_metrics": atypical_metrics,
        "shape_gated_vs_median_paired_error": paired_comparison,
        "training_gate_target_coverage": {
            "covered": sum(gate_target_covered.values()),
            "total": len(gate_target_covered),
            "fraction": float(np.mean(list(gate_target_covered.values()))),
            "excluded_cases": sorted(name for name, covered in gate_target_covered.items() if not covered),
        },
        "selected_rules": [list(rule) for rule in selected_rules],
        "shape_logistic_coefficients": coefficient_summary,
        "target_cost": {
            "nonzero_cases": sum(case.target_cost > 0 for case in cases.values()),
            "total_disagreeing_voxels": sum(case.target_cost for case in cases.values()),
            "median_second_best_gap": float(np.median([case.second_best_gap for case in cases.values()])),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    _write_csv(
        args.output_dir / "case_predictions.csv",
        predictions,
        cases,
        validation_fold,
        atypicality,
        gate_target_covered,
    )
    _write_readme(
        args.output_dir / "README.md",
        metrics,
        selected_rules,
        cases,
        coefficient_summary,
        atypical_metrics,
        gate_target_covered,
        paired_comparison,
    )

    for name in ("hippocampus_017", "hippocampus_164"):
        if name in cases:
            _plot_case(
                cases[name],
                {
                    "fixed": predictions["handcrafted_fold_transition"][name],
                    "shape-logistic": predictions["logistic_shape_only"][name],
                    "median": predictions["median_relative"][name],
                },
                args.output_dir / f"{name}_profiles.png",
            )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
