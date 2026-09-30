"""Plots and descriptive paired summaries of the frozen graph landmark probe."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.probe_graph_landmark import ARMS, RULES, paired, design, fit
from evaluation.voxel_graph_anatomy import save_json, digest


def bootstrap_delta(delta):
    x = np.asarray(delta)
    samples = np.random.default_rng(20260924).choice(x, (10000, len(x)), replace=True).mean(1)
    return dict(mean=float(x.mean()), ci95=np.quantile(samples, [.025, .975]).tolist(),
                positive_cases=int((x > 1e-12).sum()), negative_cases=int((x < -1e-12).sum()),
                unchanged_cases=int((abs(x) <= 1e-12).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "docs/experiments/graph_landmark_20260924")
    parser.add_argument("--cache", type=Path, default=ROOT / "experiments/graph_landmark_20260924/features")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/graph_landmark_20260924")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    load = lambda f: json.loads((args.report / f).read_text())
    training, validation = load("training_summary.json"), load("validation_summary.json")
    oof, val_rows = load("training_oof.json"), load("validation_cases.json")
    extra = {"training_combined_vs_median_position": paired(oof["combined"], oof["median_position"]),
             "reference_validation_combined_vs_median_position": paired(val_rows["reference"]["combined"], val_rows["reference"]["median_position"])}
    for source in ("baseline_seed0", "augmentation_seed0"):
        rows = val_rows[source]
        extra[source] = {"combined_vs_original_cut": paired(rows["combined"], rows["original"]),
            "combined_vs_median_position": paired(rows["combined"], rows["median_position"]),
            "dice_delta_percentage_points_vs_original": bootstrap_delta([100*(a["ap_dice"]-b["ap_dice"]) for a,b in zip(rows["combined"],rows["original"])]),
            "dice_delta_percentage_points_vs_fitted_plane": bootstrap_delta([100*(a["ap_dice"]-b["ap_dice"]) for a,b in zip(rows["combined"],rows["fitted_plane"])])}
    save_json(args.report / "paired_details.json", extra)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fig.suptitle("Whole-graph landmark probe · a modest distributed signal", fontsize=18, weight="bold")
    methods = ("median_position", "shape", "graph", "combined", "shuffled_graph")
    labels = ("Position\nmedian", "Shape", "Graph", "Shape +\ngraph", "Shifted\ngraph control")
    for offset, data, name, color in ((-.18, training["methods"], "Training CV (208)", "#2b667d"),
                                      (.18, validation["reference"]["methods"], "Reference validation (52)", "#cf9953")):
        values = [data[m]["mae_mm"] for m in methods]
        bars = axs[0, 0].bar(np.arange(len(methods))+offset, values, .34, label=name, color=color)
        axs[0, 0].bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
    axs[0, 0].set_xticks(range(len(methods)), labels)
    axs[0, 0].set_ylim(0, 1.8)
    axs[0, 0].set_ylabel("Cut MAE (mm)")
    axs[0, 0].set_title("Aligned graph features add to ordinary shape")
    axs[0, 0].legend(frameon=False, fontsize=8)
    for method, title, color in (("combined", "Shape + graph", "#2b667d"), ("shape", "Shape", "#cf9953"),
                                  ("median_position", "Position median", "#8b979f")):
        errors = np.array([r["error_mm"] for r in val_rows["reference"][method]])
        x = np.arange(int(errors.max())+1)
        axs[0, 1].plot(x, [100*np.mean(errors<=e) for e in x], "o-", label=title, color=color)
    axs[0, 1].set_xlabel("Allowed cut error (mm)")
    axs[0, 1].set_ylabel("Validation cases within tolerance (%)")
    axs[0, 1].set_title("Exact boundary in 27/52; within 1 mm in 42/52")
    axs[0, 1].legend(frameon=False)
    ordered = sorted(RULES, key=lambda m: training["methods"][m]["mae_mm"])
    axs[1, 0].barh(range(len(ordered)), [validation["reference"]["methods"][m]["mae_mm"] for m in ordered], color="#6a939c")
    axs[1, 0].set_yticks(range(len(ordered)), [m.replace("_", " ") for m in ordered], fontsize=8)
    axs[1, 0].invert_yaxis()
    axs[1, 0].set_xlabel("Reference validation MAE (mm)")
    axs[1, 0].set_title("Single peaks are unreliable landmarks")
    for i, (source, title) in enumerate((("baseline_seed0", "Unaugmented"), ("augmentation_seed0", "Augmented"))):
        data = validation[source]["methods"]
        bars = axs[1, 1].bar(np.array([0, 1, 2])+i*4, [data[m]["mae_mm"] for m in ("original", "median_position", "combined")],
                            color=["#8b979f", "#cf9953", "#2b667d"])
        axs[1, 1].bar_label(bars, fmt="%.2f", fontsize=9, padding=3)
    axs[1, 1].set_xticks([0, 1, 2, 4, 5, 6], ["Network", "Position", "Graph"]*2)
    axs[1, 1].set_ylabel("Cut MAE on predicted foreground (mm)")
    axs[1, 1].set_title("No improvement over the existing network cuts")
    axs[1, 1].text(1, 1.23, "Unaugmented", ha="center", color="#37434b", weight="bold")
    axs[1, 1].text(5, 1.23, "Augmented", ha="center", color="#37434b", weight="bold")
    axs[1, 1].set_ylim(0, 1.35)
    fig.savefig(args.output / "results.png", dpi=180)
    plt.close(fig)

    # Illustrative cases selected AFTER evaluation, not a tuning subset.
    rows = val_rows["reference"]["combined"]
    selected = [next(r for r in rows if r["error_mm"] == e) for e in (0., 1., max(r["error_mm"] for r in rows))]
    split = json.loads((ROOT / "datasets/Dataset101_MSD/splits_final.json").read_text())[0]
    def cached(name):
        with np.load(args.cache / "reference" / f"{name}.npz", allow_pickle=False) as z:
            case = dict(z)
        case.update(name=str(case["name"]), target=int(case["target"]))
        return case
    with threadpool_limits(limits=1):
        model = fit([cached(n) for n in split["train"]], "combined")
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    fig.suptitle("Examples: no individual peak consistently marks the A/P boundary", fontsize=16, weight="bold")
    for ax, result in zip(axs, selected):
        case = cached(result["case"])
        signals = [("Readout score", model.decision_function(design(case, "combined")), "#2b667d", 2.4),
                   ("Thickness rise", case["rules"][:, 1], "#cb754e", 1.2),
                   ("Area rise", case["rules"][:, 0], "#509f89", 1.2)]
        for name, values, color, width in signals:
            scaled = (values-values.min()) / max(np.ptp(values), 1e-12)
            ax.plot(case["candidates"]-case["target"], scaled, label=name, color=color, linewidth=width)
        ax.axvline(0, color="black", linestyle="--", linewidth=1.1, label="Reference boundary")
        ax.axvline(result["cut"]-result["target"], color="#c55b6d", linestyle=":", linewidth=2, label="Selected cut")
        ax.set_title(f"{result['case']} · error {result['error_mm']:.0f} mm")
        ax.set_xlabel("Candidate offset from reference cut (mm)")
        ax.set_ylabel("Within-case scaled descriptor / score")
    axs[0].legend(frameon=False, fontsize=8)
    fig.savefig(args.output / "landmark_profiles.png", dpi=180)
    plt.close(fig)
    save_json(args.report / "figure_manifest.json", dict(script_sha256=digest(__file__),
        illustrative_cases=[r["case"] for r in selected], retrospective_example_selection=True))
    print(json.dumps(extra, indent=2))


if __name__ == "__main__":
    main()
