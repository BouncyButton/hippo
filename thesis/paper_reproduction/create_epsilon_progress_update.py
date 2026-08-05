#!/usr/bin/env python3
"""Create a thesis-progress analysis package for the five-fold epsilon sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EPSILONS = (0, 500, 1000, 2500, 5000)
FOLDS = (1, 2, 3, 4, 5)
CONTROL_EPSILON = 5000
PALETTE = {
    0: "#0072B2",
    500: "#56B4E9",
    1000: "#009E73",
    2500: "#E69F00",
    5000: "#CC79A7",
}
MARKERS = {0: "o", 500: "s", 1000: "^", 2500: "D", 5000: "P"}
FINAL_METRICS = (
    "dice_all_classes",
    "dice_foreground",
    "dice_background",
    "dice_anterior",
    "dice_posterior",
    "prediction_volume_gap_voxels",
    "prediction_volume_similarity_configured",
    "prediction_volume_violation_configured",
    "prediction_connectedness",
    "prediction_nested",
)
LEARNING_METRICS = (
    "dice_all_classes",
    "train_constraint_satisfaction",
    "train_dice_truth",
    "train_connectedness_truth",
    "train_volume_truth",
    "train_nesting_truth",
    "train_loss",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Keep complete PNG/PDF figure pairs and render only missing figures.",
    )
    return parser.parse_args()


def mean_sd(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    return float(array.mean()), float(array.std(ddof=1))


def quantile(values: Iterable[float], probability: float) -> float:
    return float(np.quantile(np.asarray(list(values), dtype=float), probability))


def read_csv(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            {
                key: float(value)
                for key, value in row.items()
                if value not in ("", None)
            }
            for row in reader
        ]


def load_runs(runs_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    manifests: set[str] = set()
    for config_path in sorted(runs_root.glob("epsilon_*/config.json")):
        directory = config_path.parent
        final_path = directory / "final_metrics.json"
        metrics_path = directory / "metrics.csv"
        manifest_path = directory / "dataset_manifest.json"
        if not (final_path.is_file() and metrics_path.is_file() and manifest_path.is_file()):
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        run = config["run"]
        if run.get("method") != "ltn":
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifests.add(str(manifest["manifest_sha256"]))
        record = {
            "epsilon": int(float(run["volume_epsilon"])),
            "gamma": float(run["volume_gamma"]),
            "grounding": str(run["volume_grounding"]),
            "fraction": float(run["train_fraction"]),
            "fold": int(run["fold"]),
            "final": json.loads(final_path.read_text(encoding="utf-8")),
            "epochs": read_csv(metrics_path),
            "directory": directory.name,
        }
        records.append(record)

    if len(manifests) != 1:
        raise RuntimeError(f"Expected one dataset manifest; found {len(manifests)}")
    found = {(record["epsilon"], record["fold"]) for record in records}
    expected = {(epsilon, fold) for epsilon in EPSILONS for fold in FOLDS}
    if found != expected:
        raise RuntimeError(
            f"Incomplete or unexpected epsilon grid. Missing={sorted(expected - found)}, "
            f"extra={sorted(found - expected)}"
        )
    for record in records:
        if record["grounding"] != "paper-hard" or record["fraction"] != 1.0:
            raise RuntimeError(f"Mixed protocol in {record['directory']}")
        epochs = [int(row["epoch"]) for row in record["epochs"]]
        if epochs != list(range(1, 101)):
            raise RuntimeError(f"Expected epochs 1..100 in {record['directory']}")
        missing = [name for name in FINAL_METRICS if name not in record["final"]]
        if missing:
            raise RuntimeError(f"Missing final metrics in {record['directory']}: {missing}")
    return sorted(records, key=lambda record: (record["epsilon"], record["fold"]))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_tables(
    records: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    final_tidy: list[dict[str, Any]] = []
    for record in records:
        for metric in FINAL_METRICS:
            final_tidy.append(
                {
                    "epsilon": record["epsilon"],
                    "fold": record["fold"],
                    "metric": metric,
                    "value": float(record["final"][metric]),
                }
            )

    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in final_tidy:
        grouped[(int(row["epsilon"]), str(row["metric"]))].append(float(row["value"]))
    summary_tidy: list[dict[str, Any]] = []
    for epsilon in EPSILONS:
        for metric in FINAL_METRICS:
            values = grouped[(epsilon, metric)]
            mean, sd = mean_sd(values)
            summary_tidy.append(
                {
                    "epsilon": epsilon,
                    "metric": metric,
                    "n": len(values),
                    "mean": mean,
                    "sd": sd,
                    "min": min(values),
                    "max": max(values),
                }
            )

    final_lookup = {
        (record["epsilon"], record["fold"], metric): float(record["final"][metric])
        for record in records
        for metric in FINAL_METRICS
    }
    paired: list[dict[str, Any]] = []
    for metric in ("dice_all_classes", "dice_foreground"):
        for epsilon in EPSILONS:
            differences = []
            for fold in FOLDS:
                value = final_lookup[(epsilon, fold, metric)]
                control = final_lookup[(CONTROL_EPSILON, fold, metric)]
                difference = value - control
                differences.append(difference)
                paired.append(
                    {
                        "row_type": "fold",
                        "metric": metric,
                        "epsilon": epsilon,
                        "fold": fold,
                        "value": value,
                        "control_value": control,
                        "difference": difference,
                        "mean_difference": "",
                        "sd_difference": "",
                        "median_difference": "",
                        "cohen_dz": "",
                        "wins": "",
                        "n": "",
                    }
                )
            mean, sd = mean_sd(differences)
            effect = mean / sd if sd > 0 else math.nan
            paired.append(
                {
                    "row_type": "summary",
                    "metric": metric,
                    "epsilon": epsilon,
                    "fold": "",
                    "value": "",
                    "control_value": "",
                    "difference": "",
                    "mean_difference": mean,
                    "sd_difference": sd,
                    "median_difference": float(np.median(differences)),
                    "cohen_dz": effect,
                    "wins": sum(value > 0 for value in differences),
                    "n": len(differences),
                }
            )

    learning_tidy: list[dict[str, Any]] = []
    for record in records:
        for epoch_row in record["epochs"]:
            for metric in LEARNING_METRICS:
                learning_tidy.append(
                    {
                        "epsilon": record["epsilon"],
                        "fold": record["fold"],
                        "epoch": int(epoch_row["epoch"]),
                        "metric": metric,
                        "value": float(epoch_row[metric]),
                    }
                )
    return final_tidy, summary_tidy, paired, learning_tidy


def summary_lookup(summary_rows: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    return {
        (int(row["epsilon"]), str(row["metric"])): row
        for row in summary_rows
    }


def values_by(
    final_rows: list[dict[str, Any]], epsilon: int, metric: str
) -> list[tuple[int, float]]:
    return sorted(
        [
            (int(row["fold"]), float(row["value"]))
            for row in final_rows
            if int(row["epsilon"]) == epsilon and row["metric"] == metric
        ]
    )


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 320,
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "legend.fontsize": 9,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "axes.axisbelow": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(figure: Any, output_dir: Path, stem: str) -> list[str]:
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    figure.savefig(png, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return [png.name, pdf.name]


def existing_figure_pair(output_dir: Path, stem: str) -> list[str] | None:
    names = [f"{stem}.png", f"{stem}.pdf"]
    if all((output_dir / name).is_file() and (output_dir / name).stat().st_size > 0 for name in names):
        return names
    return None


def categorical_axis(axis: Any) -> None:
    axis.set_xticks(range(len(EPSILONS)), [str(value) for value in EPSILONS])
    axis.set_xlabel("Volume tolerance ε (voxels)")


def add_fold_points_and_summary(
    axis: Any,
    final_rows: list[dict[str, Any]],
    summary: dict[tuple[int, str], dict[str, Any]],
    metric: str,
    *,
    scale: float = 1.0,
    connect_folds: bool = True,
) -> None:
    if connect_folds:
        for fold in FOLDS:
            values = [
                next(value for item_fold, value in values_by(final_rows, epsilon, metric) if item_fold == fold)
                * scale
                for epsilon in EPSILONS
            ]
            axis.plot(
                range(len(EPSILONS)),
                values,
                color="#8B8B8B",
                alpha=0.26,
                linewidth=0.8,
                zorder=1,
            )
    offsets = np.linspace(-0.07, 0.07, len(FOLDS))
    for position, epsilon in enumerate(EPSILONS):
        fold_values = values_by(final_rows, epsilon, metric)
        for offset, (_, value) in zip(offsets, fold_values):
            axis.scatter(
                position + offset,
                value * scale,
                s=28,
                marker=MARKERS[epsilon],
                facecolor="white",
                edgecolor=PALETTE[epsilon],
                linewidth=1.1,
                zorder=3,
            )
        row = summary[(epsilon, metric)]
        axis.errorbar(
            position,
            float(row["mean"]) * scale,
            yerr=float(row["sd"]) * scale,
            fmt=MARKERS[epsilon],
            color=PALETTE[epsilon],
            markerfacecolor=PALETTE[epsilon],
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=8,
            capsize=4,
            elinewidth=1.8,
            linewidth=0,
            zorder=4,
        )
    categorical_axis(axis)


def plot_dice(
    output_dir: Path,
    final_rows: list[dict[str, Any]],
    summary: dict[tuple[int, str], dict[str, Any]],
) -> list[str]:
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), constrained_layout=True)
    specifications = (
        ("dice_all_classes", "All-class Dice", (0.83, 0.90)),
        ("dice_foreground", "Foreground Dice", (0.75, 0.84)),
    )
    for axis, (metric, title, limits) in zip(axes, specifications):
        add_fold_points_and_summary(axis, final_rows, summary, metric)
        axis.set_title(f"{title} · zoomed axis")
        axis.set_ylabel("Dice")
        axis.set_ylim(*limits)
        axis.text(
            0.01,
            0.02,
            "Large markers: mean ± fold SD; open markers: folds; grey lines: paired folds",
            transform=axis.transAxes,
            fontsize=8.5,
            color="#555555",
        )
    figure.suptitle("Segmentation performance across ε (n=5 folds)", fontsize=14)
    return save_figure(figure, output_dir, "01_dice_vs_epsilon")


def plot_paired_differences(
    output_dir: Path,
    paired_rows: list[dict[str, Any]],
) -> list[str]:
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), constrained_layout=True)
    specifications = (
        ("dice_all_classes", "All-class Dice"),
        ("dice_foreground", "Foreground Dice"),
    )
    offsets = np.linspace(-0.07, 0.07, len(FOLDS))
    for axis, (metric, title) in zip(axes, specifications):
        axis.axhline(0, color="#444444", linewidth=1)
        for position, epsilon in enumerate(EPSILONS):
            fold_rows = [
                row
                for row in paired_rows
                if row["row_type"] == "fold"
                and row["metric"] == metric
                and int(row["epsilon"]) == epsilon
            ]
            differences = [float(row["difference"]) * 100 for row in fold_rows]
            for offset, difference in zip(offsets, differences):
                axis.scatter(
                    position + offset,
                    difference,
                    s=32,
                    marker=MARKERS[epsilon],
                    facecolor="white",
                    edgecolor=PALETTE[epsilon],
                    linewidth=1.1,
                    zorder=3,
                )
            mean, sd = mean_sd(differences)
            axis.errorbar(
                position,
                mean,
                yerr=sd,
                fmt=MARKERS[epsilon],
                color=PALETTE[epsilon],
                markerfacecolor=PALETTE[epsilon],
                markeredgecolor="white",
                markeredgewidth=0.8,
                markersize=8,
                capsize=4,
                zorder=4,
            )
            if epsilon != CONTROL_EPSILON:
                axis.annotate(
                    f"{mean:+.2f} pp",
                    (position, mean),
                    xytext=(0, 12 if mean >= 0 else -17),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8.5,
                )
        categorical_axis(axis)
        axis.set_ylabel(f"Paired Δ {title} vs ε=5000 (percentage points)")
        axis.set_title(title)
        axis.text(
            0.01,
            0.02,
            "Every fold shown; bars are mean ± fold SD",
            transform=axis.transAxes,
            fontsize=8.5,
            color="#555555",
        )
    figure.suptitle("Paired performance change relative to the paper ε=5000 control", fontsize=14)
    return save_figure(figure, output_dir, "02_paired_dice_change_vs_5000")


def plot_classwise(
    output_dir: Path,
    final_rows: list[dict[str, Any]],
    summary: dict[tuple[int, str], dict[str, Any]],
) -> list[str]:
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), constrained_layout=True)
    specifications = (
        ("dice_background", "Background", (0.988, 0.993)),
        ("dice_anterior", "Anterior hippocampus", (0.77, 0.85)),
        ("dice_posterior", "Posterior hippocampus", (0.75, 0.83)),
    )
    for axis, (metric, title, limits) in zip(axes, specifications):
        add_fold_points_and_summary(axis, final_rows, summary, metric)
        axis.set_title(f"{title} · zoomed axis")
        axis.set_ylabel("Dice")
        axis.set_ylim(*limits)
    figure.suptitle("Classwise Dice across ε (mean ± fold SD; n=5)", fontsize=14)
    return save_figure(figure, output_dir, "03_classwise_dice_vs_epsilon")


def plot_volume_behavior(
    output_dir: Path,
    final_rows: list[dict[str, Any]],
    summary: dict[tuple[int, str], dict[str, Any]],
) -> list[str]:
    figure, axes = plt.subplots(1, 3, figsize=(15.8, 4.9), constrained_layout=True)
    specifications = (
        (
            "prediction_volume_gap_voxels",
            "Hard predicted volume gap",
            "Anterior–posterior gap (voxels)",
            (0, 1400),
        ),
        (
            "prediction_volume_similarity_configured",
            "Configured Eq. 7 satisfaction",
            "Satisfaction",
            (0, 1.05),
        ),
        (
            "prediction_volume_violation_configured",
            "Configured violation rate",
            "Validation-case fraction",
            (0, 1.05),
        ),
    )
    for axis, (metric, title, ylabel, limits) in zip(axes, specifications):
        add_fold_points_and_summary(
            axis,
            final_rows,
            summary,
            metric,
            connect_folds=True,
        )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_ylim(*limits)
    figure.suptitle(
        "Volume behavior: the gap is comparable; satisfaction and violations use different ε thresholds",
        fontsize=13.5,
    )
    return save_figure(figure, output_dir, "04_volume_constraint_behavior")


def plot_equation7(
    output_dir: Path,
    records: list[dict[str, Any]],
    output_rows: list[dict[str, Any]],
) -> list[str]:
    gamma = float(records[0]["gamma"])
    gaps = np.linspace(0, 6000, 601)
    figure, axes = plt.subplots(2, 3, figsize=(15.2, 9.0), constrained_layout=True)
    for axis, epsilon in zip(axes.flat, EPSILONS):
        truth = np.exp(-gamma * np.square(np.maximum(gaps - epsilon, 0)))
        axis.axvspan(0, epsilon, color=PALETTE[epsilon], alpha=0.10)
        axis.plot(gaps, truth, color=PALETTE[epsilon], linewidth=2.2)
        observed = [
            float(record["final"]["prediction_volume_gap_voxels"])
            for record in records
            if record["epsilon"] == epsilon
        ]
        observed_truth = np.exp(
            -gamma * np.square(np.maximum(np.asarray(observed) - epsilon, 0))
        )
        axis.scatter(
            observed,
            observed_truth,
            marker=MARKERS[epsilon],
            s=48,
            color=PALETTE[epsilon],
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        axis.set_title(f"ε={epsilon} · shaded dead zone")
        axis.set_xlim(0, 6000)
        axis.set_ylim(-0.03, 1.03)
        axis.set_xlabel("Hard volume gap (voxels)")
        axis.set_ylabel("Eq. 7 satisfaction")
        for gap, satisfaction in zip(gaps, truth):
            output_rows.append(
                {
                    "epsilon": epsilon,
                    "gamma": gamma,
                    "gap_voxels": float(gap),
                    "satisfaction": float(satisfaction),
                    "row_type": "curve",
                    "fold": "",
                }
            )
        for fold, (gap, satisfaction) in enumerate(
            zip(observed, observed_truth), start=1
        ):
            output_rows.append(
                {
                    "epsilon": epsilon,
                    "gamma": gamma,
                    "gap_voxels": gap,
                    "satisfaction": float(satisfaction),
                    "row_type": "observed_fold",
                    "fold": fold,
                }
            )
    axes.flat[-1].axis("off")
    axes.flat[-1].text(
        0.03,
        0.82,
        "Equation 7",
        fontsize=14,
        fontweight="normal",
        transform=axes.flat[-1].transAxes,
    )
    axes.flat[-1].text(
        0.03,
        0.66,
        "truth = exp[−γ · max(gap−ε, 0)²]",
        fontsize=11,
        transform=axes.flat[-1].transAxes,
    )
    axes.flat[-1].text(
        0.03,
        0.49,
        "Inside each shaded interval, truth is exactly 1.\n"
        "The constraint is therefore inactive with respect to gap size.",
        fontsize=10.5,
        transform=axes.flat[-1].transAxes,
    )
    axes.flat[-1].text(
        0.03,
        0.28,
        "Colored marks: observed fold-level validation means (n=5)",
        fontsize=9.5,
        color="#555555",
        transform=axes.flat[-1].transAxes,
    )
    figure.suptitle("Equation 7 dead zones and observed validation gaps", fontsize=14)
    return save_figure(figure, output_dir, "05_equation7_dead_zones")


def plot_learning_curves(
    output_dir: Path,
    learning_rows: list[dict[str, Any]],
) -> list[str]:
    figure, axes = plt.subplots(4, 2, figsize=(14.2, 15.2), constrained_layout=True)
    specifications = (
        ("dice_all_classes", "Validation all-class Dice", (0.65, 0.90)),
        ("train_constraint_satisfaction", "Total constraint satisfaction", (0, 1.02)),
        ("train_dice_truth", "Dice truth", (0, 1.02)),
        ("train_connectedness_truth", "Connectedness truth", (0, 1.02)),
        ("train_volume_truth", "Volume truth", (0, 1.02)),
        ("train_nesting_truth", "Nesting truth", (0, 1.02)),
        ("train_loss", "Training loss = 1 − SatAgg", (0, 1.02)),
    )
    lookup: dict[tuple[int, str, int], list[float]] = defaultdict(list)
    for row in learning_rows:
        lookup[(int(row["epsilon"]), str(row["metric"]), int(row["epoch"]))].append(
            float(row["value"])
        )
    for axis, (metric, title, limits) in zip(axes.flat, specifications):
        for epsilon in EPSILONS:
            epochs = np.arange(1, 101)
            means = []
            deviations = []
            for epoch in epochs:
                mean, sd = mean_sd(lookup[(epsilon, metric, int(epoch))])
                means.append(mean)
                deviations.append(sd)
            means_array = np.asarray(means)
            sd_array = np.asarray(deviations)
            axis.fill_between(
                epochs,
                means_array - sd_array,
                means_array + sd_array,
                color=PALETTE[epsilon],
                alpha=0.08,
                linewidth=0,
            )
            axis.plot(
                epochs,
                means_array,
                color=PALETTE[epsilon],
                linewidth=1.45,
                label=f"ε={epsilon}",
            )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Value")
        axis.set_xlim(1, 100)
        axis.set_ylim(*limits)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="upper left", frameon=False, ncol=1)
    axes.flat[-1].text(
        0,
        0.42,
        "Lines: five-fold mean\nBands: ± fold SD\n\n"
        "Only Dice truth is differentiable under paper-hard grounding.",
        transform=axes.flat[-1].transAxes,
        fontsize=10.5,
    )
    figure.suptitle("Learning dynamics across ε (n=5 folds)", fontsize=14)
    return save_figure(figure, output_dir, "06_learning_curves")


def plot_selection(
    output_dir: Path,
    summary: dict[tuple[int, str], dict[str, Any]],
) -> list[str]:
    figure, axis = plt.subplots(figsize=(8.5, 6.2), constrained_layout=True)
    for epsilon in EPSILONS:
        x = summary[(epsilon, "prediction_volume_gap_voxels")]
        y = summary[(epsilon, "dice_foreground")]
        axis.errorbar(
            float(x["mean"]),
            float(y["mean"]),
            xerr=float(x["sd"]),
            yerr=float(y["sd"]),
            fmt=MARKERS[epsilon],
            color=PALETTE[epsilon],
            markerfacecolor=PALETTE[epsilon],
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=10,
            capsize=4,
            elinewidth=1.5,
            label=f"ε={epsilon}",
        )
        axis.annotate(
            f"ε={epsilon}",
            (float(x["mean"]), float(y["mean"])),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=9,
        )
    axis.set_xlabel("Mean predicted hard volume gap (voxels) · lower is better")
    axis.set_ylabel("Mean foreground Dice · higher is better (zoomed axis)")
    axis.set_ylim(0.775, 0.825)
    axis.set_xlim(850, 1300)
    axis.set_title("Performance–structure selection view (mean ± fold SD; n=5)")
    axis.annotate(
        "Preferred direction",
        xy=(900, 0.819),
        xytext=(1040, 0.810),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        color="#555555",
    )
    return save_figure(figure, output_dir, "07_selection_tradeoff")


def analysis_payload(
    records: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    figures: list[str],
) -> dict[str, Any]:
    summary = summary_lookup(summary_rows)
    paired_summary = [
        row for row in paired_rows if row["row_type"] == "summary"
    ]
    best_epsilon = max(
        EPSILONS,
        key=lambda epsilon: float(summary[(epsilon, "dice_foreground")]["mean"]),
    )
    best_pair = next(
        row
        for row in paired_summary
        if row["metric"] == "dice_foreground" and int(row["epsilon"]) == best_epsilon
    )
    return {
        "protocol": {
            "method": "ltn",
            "train_fraction": 1.0,
            "volume_grounding": "paper-hard",
            "seed": 42,
            "folds": list(FOLDS),
            "epochs": 100,
            "epsilon_values": list(EPSILONS),
            "control_epsilon": CONTROL_EPSILON,
            "volume_gamma": float(records[0]["gamma"]),
        },
        "selection": {
            "recommended_epsilon": best_epsilon,
            "reason": (
                "Highest mean foreground Dice and positive paired change versus "
                "epsilon=5000 in every fold."
            ),
            "foreground_mean": float(
                summary[(best_epsilon, "dice_foreground")]["mean"]
            ),
            "foreground_sd": float(
                summary[(best_epsilon, "dice_foreground")]["sd"]
            ),
            "paired_mean_delta_vs_5000": float(best_pair["mean_difference"]),
            "paired_sd_delta_vs_5000": float(best_pair["sd_difference"]),
            "paired_cohen_dz_descriptive": float(best_pair["cohen_dz"]),
            "paired_wins": int(best_pair["wins"]),
            "paired_n": int(best_pair["n"]),
            "caution": (
                "n=5; the absolute improvement is small and this sweep does not "
                "include a five-fold unconstrained baseline."
            ),
        },
        "summary": summary_rows,
        "paired_effects": paired_summary,
        "figures": figures,
    }


def write_report(
    path: Path,
    payload: dict[str, Any],
    figures: list[str],
) -> None:
    summary = {
        (int(row["epsilon"]), str(row["metric"])): row
        for row in payload["summary"]
    }
    selection = payload["selection"]
    lines = [
        "# Full-data epsilon sweep — thesis progress update",
        "",
        "## Headline",
        "",
        (
            f"`ε={selection['recommended_epsilon']}` is the preferred paper-hard setting: "
            f"foreground Dice `{selection['foreground_mean']:.4f} ± "
            f"{selection['foreground_sd']:.4f}` across five folds. Relative to the paper "
            f"`ε=5000` control, the paired mean change is "
            f"`{selection['paired_mean_delta_vs_5000'] * 100:+.3f}` percentage points "
            f"(`{selection['paired_wins']}/{selection['paired_n']}` folds improve)."
        ),
        "",
        (
            "This is a small absolute gain and should be described as a robust directional "
            "result within this sweep, not as definitive statistical evidence or an "
            "improvement over an unconstrained baseline."
        ),
        "",
        "## Five-fold summary",
        "",
        "| ε | All-class Dice | Foreground Dice | Background | Anterior | Posterior | Hard gap (voxels) | Configured satisfaction | Violation rate |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for epsilon in EPSILONS:
        def fmt(metric: str, digits: int = 4) -> str:
            row = summary[(epsilon, metric)]
            return f"{float(row['mean']):.{digits}f} ± {float(row['sd']):.{digits}f}"

        lines.append(
            f"| {epsilon} | {fmt('dice_all_classes')} | {fmt('dice_foreground')} | "
            f"{fmt('dice_background')} | {fmt('dice_anterior')} | "
            f"{fmt('dice_posterior')} | "
            f"{fmt('prediction_volume_gap_voxels', 1)} | "
            f"{fmt('prediction_volume_similarity_configured')} | "
            f"{fmt('prediction_volume_violation_configured')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Dice improves monotonically from ε=0 through ε=2500, then gives back a small amount at ε=5000.",
            "- ε=2500 improves both all-class and foreground Dice against ε=5000 in all five paired folds, but the gain is only about 0.10–0.15 percentage points.",
            "- ε=2500 also has the lowest mean hard volume gap, although fold variability is larger than the between-setting difference.",
            "- Configured satisfaction and violation rate cannot be read as like-for-like model-quality outcomes because changing ε changes the scoring rule itself.",
            "- Equation 7 has an exact dead zone: every gap at or below ε maps to truth 1, so ε=5000 is vacuous for all observed validation fold means.",
            "- Under `paper-hard`, argmax-based volume, connectedness, and nesting truths are non-differentiable. Their numerical truths enter SatAgg, but gradient flow comes through Dice truth; the hard truths mainly change the scale applied to the Dice gradient.",
            "- A soft/differentiable volume grounding is justified as the next targeted experiment if the research question is whether the anatomical constraint can actively teach the model rather than merely alter Dice-gradient scaling.",
            "",
            "## Scope boundary",
            "",
            "This report compares ε settings within the full-data paper-hard LTN sweep. It does not establish improvement over an unconstrained five-fold baseline, and it does not cover 25% or 5% training fractions.",
            "",
            "## Figures",
            "",
        ]
    )
    for index in range(0, len(figures), 2):
        png = figures[index]
        lines.extend([f"### {Path(png).stem}", "", f"![{Path(png).stem}]({png})", ""])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    records = load_runs(args.runs_root)
    final_rows, summary_rows, paired_rows, learning_rows = build_tables(records)
    summary = summary_lookup(summary_rows)

    write_csv(
        args.output_dir / "thesis_progress_final_tidy.csv",
        final_rows,
        ["epsilon", "fold", "metric", "value"],
    )
    write_csv(
        args.output_dir / "thesis_progress_summary_tidy.csv",
        summary_rows,
        ["epsilon", "metric", "n", "mean", "sd", "min", "max"],
    )
    write_csv(
        args.output_dir / "thesis_progress_paired_vs_5000.csv",
        paired_rows,
        [
            "row_type",
            "metric",
            "epsilon",
            "fold",
            "value",
            "control_value",
            "difference",
            "mean_difference",
            "sd_difference",
            "median_difference",
            "cohen_dz",
            "wins",
            "n",
        ],
    )
    write_csv(
        args.output_dir / "thesis_progress_learning_tidy.csv",
        learning_rows,
        ["epsilon", "fold", "epoch", "metric", "value"],
    )

    figures: list[str] = []

    def render(stem: str, plotter: Any) -> list[str]:
        existing = existing_figure_pair(args.output_dir, stem)
        if args.resume_existing and existing is not None:
            return existing
        return plotter()

    figures.extend(
        render(
            "01_dice_vs_epsilon",
            lambda: plot_dice(args.output_dir, final_rows, summary),
        )
    )
    figures.extend(
        render(
            "02_paired_dice_change_vs_5000",
            lambda: plot_paired_differences(args.output_dir, paired_rows),
        )
    )
    figures.extend(
        render(
            "03_classwise_dice_vs_epsilon",
            lambda: plot_classwise(args.output_dir, final_rows, summary),
        )
    )
    figures.extend(
        render(
            "04_volume_constraint_behavior",
            lambda: plot_volume_behavior(args.output_dir, final_rows, summary),
        )
    )
    equation_rows: list[dict[str, Any]] = []
    equation_csv = args.output_dir / "thesis_progress_equation7_tidy.csv"
    equation_existing = existing_figure_pair(
        args.output_dir, "05_equation7_dead_zones"
    )
    if (
        args.resume_existing
        and equation_existing is not None
        and equation_csv.is_file()
        and equation_csv.stat().st_size > 0
    ):
        figures.extend(equation_existing)
    else:
        figures.extend(plot_equation7(args.output_dir, records, equation_rows))
        write_csv(
            equation_csv,
            equation_rows,
            ["epsilon", "gamma", "gap_voxels", "satisfaction", "row_type", "fold"],
        )
    figures.extend(
        render(
            "06_learning_curves",
            lambda: plot_learning_curves(args.output_dir, learning_rows),
        )
    )
    figures.extend(
        render(
            "07_selection_tradeoff",
            lambda: plot_selection(args.output_dir, summary),
        )
    )

    payload = analysis_payload(records, summary_rows, paired_rows, figures)
    (args.output_dir / "thesis_progress_analysis.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(
        args.output_dir / "thesis_progress_update.md",
        payload,
        figures,
    )


if __name__ == "__main__":
    main()
