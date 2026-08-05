#!/usr/bin/env python3
"""Aggregate LTN epsilon-sweep folds without mixing incompatible thresholds."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "mean": float(np.mean(values)),
        "std_population": float(np.std(values, ddof=0)),
        "n": len(values),
    }


def plot_epsilon_results(rows: list[dict[str, Any]], plots_dir: Path) -> list[Path]:
    """Create publication-ready comparisons from aggregated epsilon rows."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    series: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        series[(row["volume_grounding"], row["train_fraction"])].append(row)
    for values in series.values():
        values.sort(key=lambda row: row["volume_epsilon"])

    colors = plt.get_cmap("tab10")
    series_styles = {}
    for index, key in enumerate(sorted(series)):
        color = colors(index % 10)
        paper_hard = key[0] == "paper-hard"
        series_styles[key] = {
            "color": color,
            "marker": ("o", "s", "^", "D", "v")[index % 5],
            "markerfacecolor": "none" if paper_hard else color,
            "markeredgewidth": 1.8,
            "linestyle": "--" if paper_hard else "-",
            "zorder": 3 if paper_hard else 2,
            "label": f"{key[0]} · train={key[1]:g}",
        }

    def plot_metric(
        axis: Any,
        metric: str,
        title: str,
        ylabel: str,
        bounded: bool = False,
    ) -> None:
        observed = []
        for key, values in sorted(series.items()):
            x = [row["volume_epsilon"] for row in values]
            y = [row["metrics"][metric]["mean"] for row in values]
            yerr = [row["metrics"][metric]["std_population"] for row in values]
            observed.extend(
                value
                for mean, deviation in zip(y, yerr)
                for value in (mean - deviation, mean + deviation)
            )
            axis.errorbar(
                x,
                y,
                yerr=yerr,
                capsize=3,
                linewidth=1.8,
                markersize=5,
                **series_styles[key],
            )
        axis.set_title(title)
        axis.set_xlabel("Epsilon (voxels)")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25, linewidth=0.8)
        if bounded:
            axis.set_ylim(-0.02, 1.02)
        elif metric.startswith("dice_") and observed:
            lower = max(0.0, min(observed) - 0.02)
            upper = min(1.0, max(observed) + 0.02)
            if upper - lower < 0.05:
                middle = (upper + lower) / 2
                lower = max(0.0, middle - 0.025)
                upper = min(1.0, middle + 0.025)
            axis.set_ylim(lower, upper)

    dice_path = plots_dir / "dice-vs-epsilon.png"
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    plot_metric(axes[0], "dice_all_classes", "Dice across all classes", "Dice")
    plot_metric(axes[1], "dice_foreground", "Foreground Dice", "Dice")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=max(1, min(3, len(labels))),
    )
    figure.suptitle("Segmentation performance as epsilon changes", y=1.03)
    figure.savefig(dice_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    volume_path = plots_dir / "volume-metrics-vs-epsilon.png"
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    plot_metric(
        axes[0],
        "prediction_volume_gap_voxels",
        "Predicted hard-volume gap",
        "Absolute gap (voxels)",
    )
    plot_metric(
        axes[1],
        "prediction_volume_similarity_configured",
        "Configured volume satisfaction",
        "Satisfaction",
        bounded=True,
    )
    plot_metric(
        axes[2],
        "prediction_volume_violation_configured",
        "Configured violation rate",
        "Fraction of validation cases",
        bounded=True,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=max(1, min(3, len(labels))),
    )
    figure.suptitle("Anatomical volume behavior as epsilon changes", y=1.03)
    figure.savefig(volume_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    response_path = plots_dir / "epsilon-response-curves.png"
    epsilon_gamma_pairs = sorted(
        {(float(row["volume_epsilon"]), float(row["volume_gamma"])) for row in rows}
    )
    observed_ceiling = max(
        row["metrics"]["prediction_volume_gap_voxels"]["mean"]
        + 2 * row["metrics"]["prediction_volume_gap_voxels"]["std_population"]
        for row in rows
    )
    epsilon_ceiling = max(epsilon for epsilon, _ in epsilon_gamma_pairs)
    max_gap = max(1000.0, observed_ceiling * 1.1, epsilon_ceiling + 500.0)
    gaps = np.linspace(0.0, max_gap, 1000)
    figure, axis = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    for index, (epsilon, gamma) in enumerate(epsilon_gamma_pairs):
        excess = np.maximum(gaps - epsilon, 0.0)
        truth = np.exp(-gamma * np.square(excess))
        label = f"epsilon={epsilon:g}"
        if len({pair[1] for pair in epsilon_gamma_pairs}) > 1:
            label += f", gamma={gamma:g}"
        axis.plot(gaps, truth, color=colors(index % 10), linewidth=2, label=label)
    axis.set_title("Equation 7 response before training")
    axis.set_xlabel("Absolute anterior-posterior volume gap (voxels)")
    axis.set_ylabel("Volume satisfaction")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(True, alpha=0.25, linewidth=0.8)
    axis.legend(ncol=2)
    figure.savefig(response_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    return [dice_path, volume_path, response_path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Plot directory; defaults to <output-md-stem>_plots beside the Markdown report.",
    )
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    groups: dict[tuple[str, float, float], list[dict[str, Any]]] = defaultdict(list)
    manifests: set[str] = set()
    for config_path in sorted(args.runs_root.glob("*/config.json")):
        if not config_path.parent.name.startswith("epsilon_"):
            continue
        metrics_path = config_path.parent / "final_metrics.json"
        if not metrics_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        run = config["run"]
        if run.get("method") != "ltn" or "volume_epsilon" not in run:
            continue
        manifest_path = config_path.parent / "dataset_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing dataset manifest for {config_path.parent}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["manifest_sha256"]
        manifests.add(manifest)
        key = (
            str(run["volume_grounding"]),
            float(run["volume_epsilon"]),
            float(run["train_fraction"]),
        )
        groups[key].append(
            {
                "fold": int(run["fold"]),
                "config": config,
                "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
            }
        )

    if not groups:
        raise RuntimeError(f"No completed epsilon runs found under {args.runs_root}.")
    if len(manifests) != 1:
        raise RuntimeError("Refusing to aggregate runs made from different data manifests.")

    metric_names = (
        "dice_all_classes",
        "dice_foreground",
        "prediction_volume_gap_voxels",
        "prediction_volume_similarity",
        "prediction_volume_similarity_configured",
        "prediction_volume_violation_configured",
    )
    rows = []
    series_protocols: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for (grounding, epsilon, fraction), records in sorted(groups.items()):
        folds = sorted(record["fold"] for record in records)
        if len(folds) != len(set(folds)):
            raise RuntimeError(
                f"Duplicate folds for grounding={grounding}, epsilon={epsilon}, fraction={fraction}: {folds}"
            )
        if not args.allow_partial and folds != [1, 2, 3, 4, 5]:
            raise RuntimeError(
                f"Expected folds 1..5 for grounding={grounding}, epsilon={epsilon}, "
                f"fraction={fraction}; found {folds}"
            )
        protocols = []
        for record in records:
            run = record["config"]["run"]
            protocols.append(
                {
                    key: value
                    for key, value in run.items()
                    if key not in {"fold", "volume_epsilon"}
                }
            )
        if any(protocol != protocols[0] for protocol in protocols[1:]):
            raise RuntimeError(
                f"Mixed protocols for grounding={grounding}, epsilon={epsilon}, fraction={fraction}."
            )
        series_protocols[(grounding, fraction)].append(protocols[0])
        missing = [
            name
            for name in metric_names
            if any(name not in record["metrics"] for record in records)
        ]
        if missing:
            raise RuntimeError(
                f"Runs for grounding={grounding}, epsilon={epsilon}, fraction={fraction} "
                f"are missing metrics: {missing}"
            )
        rows.append(
            {
                "volume_grounding": grounding,
                "volume_epsilon": epsilon,
                "volume_gamma": float(records[0]["config"]["run"]["volume_gamma"]),
                "train_fraction": fraction,
                "folds": folds,
                "metrics": {
                    name: summarize(
                        [float(record["metrics"][name]) for record in records]
                    )
                    for name in metric_names
                },
            }
        )
    for (grounding, fraction), protocols in series_protocols.items():
        if any(protocol != protocols[0] for protocol in protocols[1:]):
            raise RuntimeError(
                f"Epsilon is not the only changing parameter for grounding={grounding}, "
                f"fraction={fraction}; refusing to draw a misleading series."
            )

    plots_dir = args.plots_dir or args.output_md.parent / f"{args.output_md.stem}_plots"
    plot_paths = plot_epsilon_results(rows, plots_dir)
    plot_paths_from_json = [
        Path(os.path.relpath(path, start=args.output_json.parent)).as_posix()
        for path in plot_paths
    ]
    payload = {
        "data_manifest_sha256": next(iter(manifests)),
        "plots": plot_paths_from_json,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Volume-epsilon sweep",
        "",
        (
            "| Grounding | Epsilon | Train fraction | Dice (all) | Dice (foreground) | "
            "Hard volume gap | Configured satisfaction | Violation rate | Folds |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        metrics = row["metrics"]
        lines.append(
            f"| {row['volume_grounding']} | {row['volume_epsilon']:g} | "
            f"{row['train_fraction']:g} | "
            f"{metrics['dice_all_classes']['mean']:.4f} ± "
            f"{metrics['dice_all_classes']['std_population']:.4f} | "
            f"{metrics['dice_foreground']['mean']:.4f} ± "
            f"{metrics['dice_foreground']['std_population']:.4f} | "
            f"{metrics['prediction_volume_gap_voxels']['mean']:.1f} ± "
            f"{metrics['prediction_volume_gap_voxels']['std_population']:.1f} | "
            f"{metrics['prediction_volume_similarity_configured']['mean']:.4f} ± "
            f"{metrics['prediction_volume_similarity_configured']['std_population']:.4f} | "
            f"{metrics['prediction_volume_violation_configured']['mean']:.4f} ± "
            f"{metrics['prediction_volume_violation_configured']['std_population']:.4f} | "
            f"{row['folds']} |"
        )
    lines.extend(["", "## Visualizations", ""])
    captions = (
        "Segmentation performance across epsilon values",
        "Volume behavior across epsilon values",
        "Equation 7 response curves",
    )
    for caption, path in zip(captions, plot_paths):
        relative_path = Path(
            os.path.relpath(path, start=args.output_md.parent)
        ).as_posix()
        lines.extend([f"### {caption}", "", f"![{caption}]({relative_path})", ""])
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
