#!/usr/bin/env python3
"""Test the refined cut descriptor on fold-0 model-predicted foreground.

The compact locator is fitted once on the 208 fold-0 training cases using the
ground-truth foreground union.  Its coefficients are then frozen and applied
to three versions of each of the 52 validation cases: the ground-truth union,
the unaugmented baseline prediction, and the translation-augmented baseline
prediction.  Anterior/posterior labels are used only to define and score the
target cut.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.stats import wilcoxon

from .audit_refined import (
    LoadedCase,
    _fit_predict,
    _load_cases,
    _median_prediction,
)
from .foldedness import CaseFeatures
from .refined import extract_refined_case_features


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_ROOT = REPOSITORY_ROOT / "datasets" / "Dataset101_MSD"
DEFAULT_AUDIT_ROOT = (
    REPOSITORY_ROOT
    / "experiments"
    / "uncal_fold_early_stopping_20260921"
    / "voxel_audit"
)
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT
    / "docs"
    / "experiments"
    / "uncal_foldedness_predicted_foreground_early_stopping_20260921"
)
MODE = "compact_geometry_image_plus_soft_position"


@dataclass(frozen=True)
class PredictionSupportAudit:
    case: LoadedCase
    raw_component_count: int
    removed_island_voxels: int
    foreground_dice: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace artifacts in an existing output directory.",
    )
    return parser.parse_args()


def crop_model_volume_to_native(volume: np.ndarray, native_shape: tuple[int, ...]) -> np.ndarray:
    """Undo the symmetric MONAI padding used by the 64^3 baseline pipeline."""

    volume = np.asarray(volume)
    if volume.ndim != 3 or len(native_shape) != 3:
        raise ValueError("volume and native_shape must be three-dimensional")
    if any(native > model for native, model in zip(native_shape, volume.shape, strict=True)):
        raise ValueError("native volume cannot be larger than the model volume")
    starts = tuple(
        (model - native) // 2
        for native, model in zip(native_shape, volume.shape, strict=True)
    )
    slices = tuple(
        slice(start, start + size)
        for start, size in zip(starts, native_shape, strict=True)
    )
    return volume[slices]


def largest_foreground_component(mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Keep the largest 26-connected component and report discarded islands."""

    mask = np.asarray(mask, dtype=bool)
    structure = ndimage.generate_binary_structure(3, 3)
    components, count = ndimage.label(mask, structure=structure)
    if count == 0:
        raise ValueError("predicted foreground is empty")
    sizes = np.bincount(components.ravel())
    largest_label = int(np.argmax(sizes[1:]) + 1)
    largest = components == largest_label
    removed = int(mask.sum() - largest.sum())
    return largest, int(count), removed


def _foreground_dice(first: np.ndarray, second: np.ndarray) -> float:
    denominator = int(first.sum() + second.sum())
    return 2.0 * float(np.logical_and(first, second).sum()) / max(denominator, 1)


def load_predicted_case(
    *,
    name: str,
    error_map_path: Path,
    dataset_root: Path,
    ground_truth_case: LoadedCase,
) -> PredictionSupportAudit:
    label_path = dataset_root / "labelsTr" / f"{name}.nii.gz"
    image_path = dataset_root / "imagesTr" / f"{name}_0000.nii.gz"
    labels = np.asarray(nib.load(str(label_path)).dataobj, dtype=np.uint8)
    image = np.asarray(nib.load(str(image_path)).dataobj, dtype=np.float32)
    with np.load(error_map_path) as archive:
        stored_ground_truth = crop_model_volume_to_native(
            archive["ground_truth"], labels.shape
        )
        prediction = crop_model_volume_to_native(archive["prediction"], labels.shape)
    if not np.array_equal(stored_ground_truth, labels):
        raise ValueError(f"{name}: cached prediction and native ground truth misalign")

    raw_union = prediction != 0
    union, component_count, removed_voxels = largest_foreground_component(raw_union)
    occupied = np.flatnonzero(union.sum(axis=(0, 2)))
    if occupied.size < 3 or np.any(np.diff(occupied) != 1):
        raise ValueError(f"{name}: main predicted component lacks a contiguous AP extent")
    low, high = int(occupied[0]), int(occupied[-1])
    truth = ground_truth_case.geometry
    geometry = CaseFeatures(
        case_name=name,
        low=low,
        high=high,
        target_cut=truth.target_cut,
        target_cost=truth.target_cost,
        second_best_gap=truth.second_best_gap,
        features={},
    )
    refined = extract_refined_case_features(image, union, low, high)
    return PredictionSupportAudit(
        case=LoadedCase(geometry=geometry, refined=refined),
        raw_component_count=component_count,
        removed_island_voxels=removed_voxels,
        foreground_dice=_foreground_dice(labels != 0, union),
    )


def _cut_metrics(
    predictions: dict[str, int],
    ground_truth_cases: dict[str, LoadedCase],
) -> dict[str, float | int]:
    errors = np.asarray(
        [
            abs(predictions[name] - ground_truth_cases[name].geometry.target_cut)
            for name in sorted(predictions)
        ],
        dtype=np.float64,
    )
    return {
        "count": int(errors.size),
        "mae_slices": float(errors.mean()),
        "median_absolute_error": float(np.median(errors)),
        "exact_fraction": float((errors == 0).mean()),
        "within_1_fraction": float((errors <= 1).mean()),
        "within_2_fraction": float((errors <= 2).mean()),
        "p90_absolute_error": float(np.quantile(errors, 0.9)),
        "maximum_absolute_error": int(errors.max()),
    }


def _paired_summary(
    first: dict[str, int],
    second: dict[str, int],
    ground_truth_cases: dict[str, LoadedCase],
) -> dict[str, float | int]:
    names = sorted(first)
    first_error = np.asarray(
        [abs(first[name] - ground_truth_cases[name].geometry.target_cut) for name in names]
    )
    second_error = np.asarray(
        [abs(second[name] - ground_truth_cases[name].geometry.target_cut) for name in names]
    )
    differences = second_error - first_error
    if np.all(differences == 0):
        p_value = 1.0
    else:
        p_value = float(wilcoxon(second_error, first_error).pvalue)
    return {
        "second_better_cases": int(np.count_nonzero(second_error < first_error)),
        "equal_cases": int(np.count_nonzero(second_error == first_error)),
        "second_worse_cases": int(np.count_nonzero(second_error > first_error)),
        "mean_absolute_error_change_second_minus_first": float(differences.mean()),
        "wilcoxon_two_sided_p_value_exploratory": p_value,
        "cut_agreement_fraction": float(
            np.mean([first[name] == second[name] for name in names])
        ),
    }


def _write_report(path: Path, summary: dict[str, object]) -> None:
    methods = summary["cut_metrics"]
    medians = summary["relative_median_baseline_metrics"]
    paired = summary["paired_augmented_vs_unaugmented"]
    versus_median = summary["paired_descriptor_vs_relative_median"]
    support = summary["prediction_support"]
    lines = [
        "# Fold-0 predicted-foreground uncal-cut audit",
        "",
        "The compact geometry + T1-image + soft-position candidate was fitted on",
        "the 208 fold-0 training ground-truth unions. The fitted coefficients were",
        "frozen before application to the 52 validation cases. Validation A/P labels",
        "were used only to define and score the target cut.",
        "",
        "| foreground supplied to locator | MAE | exact | within 1 | within 2 | p90 | max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "ground_truth_union": "Ground-truth union (reference)",
        "unaugmented_prediction": "Unaugmented predicted union",
        "augmented_prediction": "Augmented predicted union",
    }
    for key in labels:
        row = methods[key]
        lines.append(
            f"| {labels[key]} | {row['mae_slices']:.3f} | {row['exact_fraction']:.1%} | "
            f"{row['within_1_fraction']:.1%} | {row['within_2_fraction']:.1%} | "
            f"{row['p90_absolute_error']:.1f} | {row['maximum_absolute_error']} |"
        )
    lines.extend(
        [
            "",
            "## Relative-position control",
            "",
            "| foreground supplied to locator | median-prior MAE | descriptor minus prior MAE | descriptor better/equal/worse | paired p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key in labels:
        median_row = medians[key]
        comparison = versus_median[key]
        lines.append(
            f"| {labels[key]} | {median_row['mae_slices']:.3f} | "
            f"{comparison['mean_absolute_error_change_second_minus_first']:+.3f} | "
            f"{comparison['second_better_cases']}/{comparison['equal_cases']}/"
            f"{comparison['second_worse_cases']} | "
            f"{comparison['wilcoxon_two_sided_p_value_exploratory']:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Paired augmented-versus-unaugmented result",
            "",
            f"The augmented foreground gives a smaller/equal/larger cut error in "
            f"`{paired['second_better_cases']}/{paired['equal_cases']}/{paired['second_worse_cases']}` "
            "of the 52 paired cases.",
            f"Mean error change (augmented minus unaugmented): "
            f"`{paired['mean_absolute_error_change_second_minus_first']:+.3f}` slices.",
            f"Exploratory paired Wilcoxon p-value: "
            f"`{paired['wilcoxon_two_sided_p_value_exploratory']:.4g}`.",
            "",
            "## Predicted-support quality and cleanup",
            "",
            "The locator receives the largest 26-connected foreground component. This",
            "is a prespecified hippocampus-support cleanup, not use of the A/P labels.",
            "",
            "| model | mean union Dice | cases with islands | discarded voxels |",
            "|---|---:|---:|---:|",
        ]
    )
    for key, display in (
        ("unaugmented_prediction", "Unaugmented"),
        ("augmented_prediction", "Augmented"),
    ):
        row = support[key]
        lines.append(
            f"| {display} | {row['mean_foreground_dice']:.4f} | "
            f"{row['cases_with_disconnected_islands']}/52 | {row['total_removed_island_voxels']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The ground-truth-union row measures the descriptor's localization ceiling on",
            "this exact fold. The two predicted-union rows measure transfer through the",
            "segmentation model's outer-boundary errors. A comparison between the augmented",
            "and unaugmented rows therefore answers whether augmentation produces a support",
            "on which this fixed anatomical descriptor can localize the annotated cut more",
            "reliably; it does not retrain or tune the locator on validation predictions.",
            "",
            "The descriptor improves materially over the relative-position control when its",
            "input is the clean ground-truth union. That advantage disappears on predicted",
            "foreground: it is slightly worse than the control for the unaugmented model and",
            "ties it in mean error for the augmented model. The current candidate is therefore",
            "appropriate as soft, quality-gated auxiliary evidence, but not as a hard or",
            "self-sufficient uncal-apex rule.",
            "",
            "## Artifacts",
            "",
            "- `summary.json`: aggregate and paired metrics.",
            "- `case_results.csv`: target, predicted cuts, errors, Dice, and cleanup per case.",
            "- `absolute_error_comparison.png`: paired per-case errors.",
            "",
            "## Reproduction",
            "",
            "```bash",
            ".venv/bin/python -m thesis.new_constraints.uncal_fold.audit_predicted_foreground",
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_plot(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(rows, key=lambda row: int(row["unaugmented_error"]), reverse=True)
    x = np.arange(len(ordered))
    figure, axis = plt.subplots(figsize=(14, 5))
    axis.plot(x, [row["unaugmented_error"] for row in ordered], "o-", label="Unaugmented")
    axis.plot(x, [row["augmented_error"] for row in ordered], "o-", label="Augmented")
    axis.plot(x, [row["ground_truth_union_error"] for row in ordered], "k.", label="GT-union reference")
    axis.set_xticks(
        x,
        [str(row["case_name"]).removeprefix("hippocampus_") for row in ordered],
        rotation=90,
        fontsize=7,
    )
    axis.set_ylabel("Absolute cut error (coronal slices)")
    axis.set_xlabel("Fold-0 validation case")
    axis.set_title("Fixed descriptor applied to model-predicted foreground")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"non-empty output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth_cases = _load_cases(args.dataset_root)
    splits = json.loads((args.dataset_root / "splits_final.json").read_text())
    train_names = list(splits[args.fold]["train"])
    validation_names = list(splits[args.fold]["val"])
    if set(train_names) & set(validation_names):
        raise ValueError("training and validation cases overlap")

    gt_predictions, coefficients, feature_names = _fit_predict(
        train_names,
        validation_names,
        ground_truth_cases,
        MODE,
        penalty="l2",
    )
    gt_median = {
        name: _median_prediction(train_names, ground_truth_cases[name], ground_truth_cases)
        for name in validation_names
    }

    variants = {
        "unaugmented_prediction": "baseline_seed0",
        "augmented_prediction": "augmentation_seed0",
    }
    predicted_cases: dict[str, dict[str, LoadedCase]] = {}
    support_audits: dict[str, dict[str, PredictionSupportAudit]] = {}
    descriptor_predictions: dict[str, dict[str, int]] = {
        "ground_truth_union": gt_predictions
    }
    median_predictions: dict[str, dict[str, int]] = {"ground_truth_union": gt_median}
    for variant, folder in variants.items():
        audits = {}
        for name in validation_names:
            audits[name] = load_predicted_case(
                name=name,
                error_map_path=args.audit_root / folder / "error_maps" / f"{name}.npz",
                dataset_root=args.dataset_root,
                ground_truth_case=ground_truth_cases[name],
            )
        support_audits[variant] = audits
        cases_for_transfer = dict(ground_truth_cases)
        cases_for_transfer.update({name: audit.case for name, audit in audits.items()})
        predicted_cases[variant] = cases_for_transfer
        predictions, variant_coefficients, variant_feature_names = _fit_predict(
            train_names,
            validation_names,
            cases_for_transfer,
            MODE,
            penalty="l2",
        )
        np.testing.assert_allclose(variant_coefficients, coefficients, atol=0.0, rtol=0.0)
        if variant_feature_names != feature_names:
            raise AssertionError("feature schema changed between transfer evaluations")
        descriptor_predictions[variant] = predictions
        median_predictions[variant] = {
            name: _median_prediction(train_names, cases_for_transfer[name], cases_for_transfer)
            for name in validation_names
        }

    cut_metrics = {
        variant: _cut_metrics(predictions, ground_truth_cases)
        for variant, predictions in descriptor_predictions.items()
    }
    median_metrics = {
        variant: _cut_metrics(predictions, ground_truth_cases)
        for variant, predictions in median_predictions.items()
    }
    support_summary = {}
    for variant, audits in support_audits.items():
        support_summary[variant] = {
            "mean_foreground_dice": float(
                np.mean([audit.foreground_dice for audit in audits.values()])
            ),
            "median_foreground_dice": float(
                np.median([audit.foreground_dice for audit in audits.values()])
            ),
            "cases_with_disconnected_islands": int(
                sum(audit.raw_component_count > 1 for audit in audits.values())
            ),
            "total_removed_island_voxels": int(
                sum(audit.removed_island_voxels for audit in audits.values())
            ),
            "maximum_removed_island_voxels_in_one_case": int(
                max(audit.removed_island_voxels for audit in audits.values())
            ),
        }

    paired = _paired_summary(
        descriptor_predictions["unaugmented_prediction"],
        descriptor_predictions["augmented_prediction"],
        ground_truth_cases,
    )
    versus_median = {
        variant: _paired_summary(
            median_predictions[variant],
            descriptor_predictions[variant],
            ground_truth_cases,
        )
        for variant in descriptor_predictions
    }
    coefficient_rows = [
        {"feature": name, "standardized_logistic_coefficient": float(value)}
        for name, value in zip(feature_names, coefficients, strict=True)
    ]
    summary = {
        "schema": "uncal_fold.predicted_foreground_audit.v1",
        "fold": args.fold,
        "seed": 0,
        "training_case_count": len(train_names),
        "validation_case_count": len(validation_names),
        "locator": MODE,
        "protocol": {
            "fit": "fold-0 training ground-truth union and native T1 only",
            "transfer": "frozen locator applied to largest 26-connected predicted foreground component",
            "target": "validation A/P labels used only for best-fit cut and scoring",
            "unaugmented_source": str(
                (args.audit_root / "baseline_seed0" / "error_maps").resolve()
            ),
            "augmented_source": str(
                (args.audit_root / "augmentation_seed0" / "error_maps").resolve()
            ),
        },
        "cut_metrics": cut_metrics,
        "relative_median_baseline_metrics": median_metrics,
        "prediction_support": support_summary,
        "paired_augmented_vs_unaugmented": paired,
        "paired_descriptor_vs_relative_median": versus_median,
        "frozen_locator_coefficients": coefficient_rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    rows = []
    for name in sorted(validation_names):
        target = ground_truth_cases[name].geometry.target_cut
        row: dict[str, object] = {
            "case_name": name,
            "target_cut": target,
            "ground_truth_union_cut": descriptor_predictions["ground_truth_union"][name],
            "ground_truth_union_median_cut": median_predictions["ground_truth_union"][name],
            "ground_truth_union_error": abs(
                descriptor_predictions["ground_truth_union"][name] - target
            ),
        }
        for variant, prefix in (
            ("unaugmented_prediction", "unaugmented"),
            ("augmented_prediction", "augmented"),
        ):
            audit = support_audits[variant][name]
            cut = descriptor_predictions[variant][name]
            row.update(
                {
                    f"{prefix}_cut": cut,
                    f"{prefix}_error": abs(cut - target),
                    f"{prefix}_median_cut": median_predictions[variant][name],
                    f"{prefix}_foreground_dice": audit.foreground_dice,
                    f"{prefix}_raw_component_count": audit.raw_component_count,
                    f"{prefix}_removed_island_voxels": audit.removed_island_voxels,
                    f"{prefix}_predicted_low": audit.case.geometry.low,
                    f"{prefix}_predicted_high": audit.case.geometry.high,
                }
            )
        rows.append(row)
    with (args.output_dir / "case_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    _make_plot(args.output_dir / "absolute_error_comparison.png", rows)
    _write_report(args.output_dir / "README.md", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
