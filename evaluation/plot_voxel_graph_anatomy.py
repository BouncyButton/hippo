"""Reproducible figures for the voxel graph anatomy audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REGIONS = ("whole", "anterior", "posterior")
COLORS = ("#245c78", "#cc6546", "#25867c")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "docs/experiments/voxel_graph_anatomy_20260923")
    parser.add_argument("--graphs", type=Path, default=ROOT / "experiments/voxel_graph_anatomy_20260923/graphs")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/voxel_graph_anatomy_20260923")
    args = parser.parse_args()
    cases = json.loads((args.report / "cases.json").read_text())
    summary = json.loads((args.report / "summary.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    args.output.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    fig.suptitle("Hippocampus voxel graphs · 260 reference masks, 780 graphs", fontsize=19, weight="bold")
    for panel, (key, title) in enumerate((("disconnected_cases", "Disconnected six-face graphs"),
                                         ("cases_with_bridges", "Graphs containing bridge edges"))):
        ax = axs[0, panel]
        for offset, subset, color in ((-.18, "train", "#245c78"), (.18, "validation", "#d39b4a")):
            n = summary[subset]["cases"]
            nums = [summary[subset]["regions"][r][key] for r in REGIONS]
            bars = ax.bar(np.arange(3) + offset, np.array(nums) / n * 100, width=.34,
                          label=f"{subset.title()} (n={n})", color=color)
            ax.bar_label(bars, labels=[f"{v}/{n}" for v in nums], fontsize=9, padding=3)
        ax.set_xticks(range(3), [r.title() for r in REGIONS])
        ax.set_ylabel("Cases (%)")
        ax.set_ylim(0, 112 if panel == 1 else max(20, ax.get_ylim()[1] * 1.25))
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)
    ax = axs[0, 2]
    for i, r in enumerate(REGIONS):
        values = [c["regions"][r]["cubical"]["beta1"] for c in cases]
        bins = np.arange(max(values) + 1)
        counts = [values.count(int(b)) for b in bins]
        ax.plot(bins, counts, marker="o", label=r.title(), color=COLORS[i])
    ax.set_title("Actual tunnels in the closed voxel union")
    ax.set_xlabel("Cubical β₁ (not graph cycle rank)")
    ax.set_ylabel("Cases")
    ax.legend(frameon=False)
    ax = axs[1, 0]
    for subset, color in (("train", "#245c78"), ("validation", "#d39b4a")):
        vals = [c["interface"]["min_ncut_plane_error_mm"] for c in cases if c["split"] == subset]
        ax.hist(vals, bins=np.arange(-.5, max(vals) + 1.5), histtype="step", linewidth=2, label=subset.title(), color=color)
    ax.set_title("Minimum normalized cut can miss A/P location")
    ax.set_xlabel("Distance from reference best plane (mm)")
    ax.set_ylabel("Cases")
    ax.legend(frameon=False)
    ax = axs[1, 1]
    for i, r in enumerate(REGIONS):
        ax.scatter([c["regions"][r]["nodes"] for c in cases],
                   [c["regions"][r]["cycle_rank"] for c in cases], s=8, alpha=.4, label=r.title(), color=COLORS[i])
    ax.set_title("Graph cycles mostly track voxel count")
    ax.set_xlabel("Voxels / nodes")
    ax.set_ylabel("Independent graph cycles (E − V + C)")
    ax.legend(frameon=False)
    ax = axs[1, 2]
    if "predictions" in summary:
        arms = summary["predictions"]
        totals = [v["ap_swaps_total"] for v in arms.values()]
        passing = [v["reference_matching"]["all_three"]["ap_swaps"] for v in arms.values()]
        ax.bar(range(2), totals, color="#c7d8e0", label="All A/P swap voxels")
        ax.bar(range(2), passing, color="#245c78", label="In cases matching reference topology")
        for i, (n, total) in enumerate(zip(passing, totals)):
            ax.text(i, n / 2, f"{n:,}\n({n / total:.1%})", ha="center", va="center", color="white", weight="bold")
        ax.set_xticks(range(2), ("Unaugmented", "Augmented"))
        ax.set_title("Errors remain in structurally matching cases")
        ax.set_ylabel("A/P swap voxels")
        ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.12))
    fig.savefig(args.output / "topology_summary.png", dpi=180)
    plt.close(fig)

    # Three deterministic examples: typical graph, fragmented posterior, most tunnels.
    train = [c for c in cases if c["split"] == "train"]
    volume_median = np.median([c["regions"]["whole"]["nodes"] for c in train])
    selected = [min(train, key=lambda c: abs(c["regions"]["whole"]["nodes"] - volume_median)),
                max(train, key=lambda c: c["regions"]["posterior"]["island_voxels"]),
                max(train, key=lambda c: c["regions"]["whole"]["cubical"]["beta1"])]
    fig = plt.figure(figsize=(15, 12), constrained_layout=True)
    fig.suptitle("Exact face-adjacency graphs · every voxel and every edge retained", fontsize=18, weight="bold")
    for row, case in enumerate(selected):
        for col, region in enumerate(REGIONS):
            with np.load(args.graphs / case["case"] / f"{region}.npz", allow_pickle=False) as graph:
                coords, edges, labels, components = [graph[k] for k in ("coordinates_ijk", "edges", "node_labels", "component_ids")]
            ax = fig.add_subplot(3, 3, row * 3 + col + 1, projection="3d")
            coords = coords.astype(float) * case["spacing"]
            segments = coords[edges]
            colors = np.where(labels == 1, COLORS[1], COLORS[2])
            ax.add_collection3d(Line3DCollection(segments, colors="#607480", linewidths=.25, alpha=.12))
            ax.scatter(*coords.T, c=colors, s=1.6, alpha=.32, depthshade=False)
            sizes = np.bincount(components)
            islands = components != sizes.argmax()
            if islands.any():
                ax.scatter(*coords[islands].T, c="#c1255a", s=20, depthshade=False, label="Disconnected voxels")
            if region == "whole":
                cut = labels[edges[:, 0]] != labels[edges[:, 1]]
                ax.add_collection3d(Line3DCollection(segments[cut], colors="#e2b13f", linewidths=1.1, alpha=.95))
            stats = case["regions"][region]
            b = stats["cubical"]
            ax.set_title(f"{case['case']} · {region.title()}\n{stats['nodes']:,} nodes · {stats['components_6']} components · β=({b['beta0']},{b['beta1']},{b['beta2']})", fontsize=10)
            ax.set_xlabel("R (mm)", labelpad=-4)
            ax.set_ylabel("A (mm)", labelpad=-4)
            ax.set_zlabel("S (mm)", labelpad=-4)
            ax.tick_params(labelsize=7, pad=0)
            ax.set_box_aspect(np.maximum(np.ptp(coords, axis=0), 1))
            ax.view_init(elev=22, azim=-38)
    fig.savefig(args.output / "example_graphs.png", dpi=170)
    plt.close(fig)
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    fig.suptitle("A/P interfaces: topology has real reference exceptions", fontsize=16, weight="bold")
    lookup = {c["case"]: c for c in cases}
    examples = (("hippocampus_001", "Disk"), ("hippocampus_228", "Corner pinch"),
                ("hippocampus_164", "Annulus / hole"), ("hippocampus_338", "Hole and corner pinch"))
    for ax, (name, description) in zip(axs.ravel(), examples):
        case = lookup[name]
        with np.load(args.graphs / name / "whole.npz", allow_pickle=False) as graph:
            coords, edges, labels, axes = [graph[k] for k in ("coordinates_ijk", "edges", "node_labels", "edge_axis")]
        cross = labels[edges[:, 0]] != labels[edges[:, 1]]
        if not np.all(axes[cross] == 1):
            raise ValueError("These interface examples must be coronal")
        polygons = []
        for origin in coords[edges[cross, 0]][:, [0, 2]]:
            polygons.append(origin + np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) - .5)
        ax.add_collection(PolyCollection(polygons, facecolors="#7daab0", edgecolors="#245c78", linewidths=.7))
        ax.autoscale_view()
        ax.set_aspect("equal")
        stats = case["interface"]
        ax.set_title(f"{name} · {case['split']}\n{description}: β=({stats['beta0']},{stats['beta1']},{stats['beta2']})")
        ax.set_xlabel("R voxel coordinate")
        ax.set_ylabel("S voxel coordinate")
    fig.savefig(args.output / "interface_exceptions.png", dpi=180)
    plt.close(fig)

    probe_path = args.report / "compactness_probe.json"
    if probe_path.exists():
        probe = json.loads(probe_path.read_text())["summary"]
        fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        fig.suptitle("A consistent graph cue, but weak cut localization", fontsize=17, weight="bold")
        for subset, color in (("train", "#245c78"), ("validation", "#d39b4a")):
            selected = [c for c in cases if c["split"] == subset]
            ratio = lambda c, r: c["regions"][r]["degree_histogram"][6] / c["regions"][r]["nodes"]
            axs[0].scatter([ratio(c, "posterior") for c in selected], [ratio(c, "anterior") for c in selected],
                           color=color, s=20, alpha=.65, label=subset.title())
            offsets = sorted(map(int, probe[subset]))
            rates = [100 * probe[subset][str(o)]["anterior_more_compact_cases"] / probe[subset][str(o)]["cases"] for o in offsets]
            axs[1].plot(offsets, rates, "o-", color=color, label=subset.title())
        axs[0].plot([.4, .66], [.4, .66], "--", color="gray", linewidth=1)
        axs[0].set_xlabel("Posterior fraction of degree-six nodes")
        axs[0].set_ylabel("Anterior fraction of degree-six nodes")
        axs[0].set_title("Anterior > posterior in all 260 references")
        axs[0].legend(frameon=False)
        axs[1].set_xlabel("Artificial shift of reference best-fit cut (mm)")
        axs[1].set_ylabel("Cases satisfying anterior > posterior (%)")
        axs[1].set_ylim(0, 105)
        axs[1].set_title("Many displaced cuts still satisfy the same rule")
        axs[1].legend(frameon=False)
        fig.savefig(args.output / "compactness_probe.png", dpi=180)
        plt.close(fig)
    print(f"Saved figures to {args.output}")


if __name__ == "__main__":
    main()
