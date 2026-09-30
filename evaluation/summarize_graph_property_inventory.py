"""Pair descriptor inventory with references; audit numerical checklist claims."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.voxel_graph_anatomy import save_json, digest
from evaluation.audit_graph_property_inventory import describe


def best_threshold(field, anterior):
    """Exact threshold fit, permitting cuts only BETWEEN distinct field values."""
    order = np.argsort(field, kind="stable")
    x, a = field[order], anterior[order]
    allowed = np.r_[0, np.flatnonzero(np.diff(x) > 0) + 1, len(x)]
    cumulative_a = np.r_[0, a.cumsum()]
    costs = cumulative_a[allowed] + (len(a) - allowed) - (a.sum() - cumulative_a[allowed])
    k = int(allowed[np.argmin(costs)])
    threshold = float((x[k-1] + x[k]) / 2) if 0 < k < len(x) else float(x[0] - 1e-8 if k == 0 else x[-1] + 1e-8)
    return dict(threshold=threshold, errors=int(costs.min()), posterior_fraction=k / len(x))


def plot_results(results, cases, maps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(maps).parent
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.7), constrained_layout=True)
    fig.suptitle("Graph-property checks: what the measurements actually support", fontsize=16, weight="bold")
    for subset, color in (("train", "#245c78"), ("validation", "#d39b4a")):
        selected = [c for c in cases["reference"].values() if c["split"] == subset]
        axs[0].hist([c["regions"]["whole"]["diameter_path_boundary_fraction"] for c in selected],
                    bins=np.linspace(0, 1, 11), histtype="step", linewidth=2, color=color, label=subset.title())
    axs[0].set_title("A diameter path can hug the surface")
    axs[0].set_xlabel("Fraction of chosen path on boundary")
    axs[0].set_ylabel("Reference cases")
    axs[0].legend(frameon=False)
    for arm, color in (("baseline_seed0", "#245c78"), ("augmentation_seed0", "#d39b4a")):
        s = results["prediction_comparisons"][arm]["reference_node_error_strata"]
        names = ("degree_0_1", "degree_2_3", "degree_4_5", "degree_6")
        axs[1].plot(range(4), [100 * s[n]["missed_fraction"] for n in names], "o-", color=color,
                    label="Unaugmented" if arm == "baseline_seed0" else "Augmented")
    axs[1].set_xticks(range(4), ["0–1", "2–3", "4–5", "6"])
    axs[1].set_xlabel("Reference foreground degree")
    axs[1].set_ylabel("Missed foreground voxels (%)")
    axs[1].set_title("Boundary exposure marks missed voxels")
    axs[1].legend(frameon=False)
    names = ("fiedler01_lcc", "fiedler_volume_quantile_lcc", "harmonic_lcc")
    for shift, subset, color in ((-.18, "train", "#245c78"), (.18, "validation", "#d39b4a")):
        errors = [100 * results["coordinate_fit"][n][subset]["fixed_training_threshold_error_fraction"]["mean"] for n in names]
        bars = axs[2].bar(np.arange(3) + shift, errors, width=.34, color=color, label=subset.title())
        axs[2].bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
    axs[2].set_xticks(range(3), ["Fiedler", "Fiedler rank", "Harmonic"])
    axs[2].set_title("Fixed training threshold on reference support")
    axs[2].set_ylabel("A/P labeling error (% of LCC voxels)")
    axs[2].set_ylim(0, max(axs[2].get_ylim()[1] * 1.2, 10))
    axs[2].legend(frameon=False, fontsize=8, loc="upper left")
    fig.savefig(output / "property_checks.png", dpi=170)
    plt.close(fig)


def write_results(path, results):
    ref = results["regions"]["reference"]
    text = ["# Additional measurements from the property checklist", "",
            "Computed 1,092 region descriptor archives: 260 reference cases plus 52 cases from each of two existing prediction caches, each with whole/anterior/posterior graphs. Training and validation are fold 0 (208/52). No segmentation model was trained.", "",
            "## Exact distances and the centerline claim", "",
            "| Region | Median exact LCC diameter, train / val (hops) | Median boundary fraction of chosen diameter path, train / val | Branched coronal slice graphs, train / val |",
            "|---|---:|---:|---:|"]
    for r in ("whole", "anterior", "posterior"):
        a, b = ref["train"][r], ref["validation"][r]
        text.append(f"| {r.title()} | {a['diameter_largest_component']['median']:.1f} / {b['diameter_largest_component']['median']:.1f} | {a['diameter_path_boundary_fraction']['median']:.1%} / {b['diameter_path_boundary_fraction']['median']:.1%} | {a['cases_with_branched_slice_graph']}/208 / {b['cases_with_branched_slice_graph']}/52 |")
    text += ["", "The diameter is an exact longest-shortest-path distance within the largest component. Other components remain recorded. Each hop is a 1-mm face step, so the metric contains grid anisotropy. A deterministic shortest path between the selected diameter endpoints frequently runs on the boundary: **diameter does not specify a medial centerline**. Alternative equally short paths can exist. The slice graph is a coronal component-adjacency proxy, not an anatomical skeleton or a continuous Reeb graph.", "",
             "## New coordinates and cut-location probe", "",
             "| Coordinate | Training-median threshold | Best case-specific threshold error, train / val | Fixed training threshold error, train / val |",
             "|---|---:|---:|---:|"]
    for key, label in (("fiedler01_lcc", "Fiedler [0,1]"), ("fiedler_volume_quantile_lcc", "Fiedler volume rank"), ("harmonic_lcc", "Harmonic Y-end field")):
        r = results["coordinate_fit"][key]
        a, b = r["train"], r["validation"]
        text.append(f"| {label} | {r['training_median_threshold']:.4f} | {a['oracle_fit_error_fraction_per_case']['mean']:.2%} / {b['oracle_fit_error_fraction_per_case']['mean']:.2%} | {a['fixed_training_threshold_error_fraction']['mean']:.2%} / {b['fixed_training_threshold_error_fraction']['mean']:.2%} |")
    text += ["", "Errors are mean per-case A/P misclassification fractions on the **reference largest component**, not Dice and not deployable prediction results. The case-specific fits use GT labels and are oracle diagnostics. The common threshold uses training labels only, but evaluation still uses reference foreground. Fiedler is oriented by RAS Y, not a verified anatomical landmark. The harmonic field uses low/high 10% Y-extent anchors. Rank normalization preserves scalar ordering, so it cannot improve the optimal representable partition.", "",
             "## Which reference voxels are missed?", "",
             "| Reference whole-graph degree | Reference voxels across 52 cases | Missed foreground, unaugmented | Missed foreground, augmented |",
             "|---|---:|---:|---:|"]
    a = results["prediction_comparisons"]["baseline_seed0"]["reference_node_error_strata"]
    b = results["prediction_comparisons"]["augmentation_seed0"]["reference_node_error_strata"]
    for key, label in (("degree_0_1", "0–1"), ("degree_2_3", "2–3"), ("degree_4_5", "4–5"), ("degree_6", "6")):
        text.append(f"| {label} | {a[key]['voxels']:,} | {a[key]['missed_fraction']:.2%} | {b[key]['missed_fraction']:.2%} |")
    text += ["", "These are pooled voxel rates, not independent observations or proof of a useful loss. The strata use reference degree solely for scoring. A feature computed only on predicted foreground has no node at a missed voxel, which is why a full-grid depth/boundary auxiliary target or an explicitly extended correction field differs from adding a feature inside the current mask.", "",
             "## Surface and topology claims", "",
             "| Cached model | Mean per-case relative whole-surface change | Underestimated cases | Disconnected square interfaces | Good-minus-bad Dice association |",
             "|---|---:|---:|---:|---:|"]
    for arm, label in (("baseline_seed0", "Unaugmented"), ("augmentation_seed0", "Augmented")):
        r = results["prediction_comparisons"][arm]
        text.append(f"| {label} | {r['whole']['surface_mean_relative_change']:.2%} | {r['whole']['surface_underestimated_cases']}/52 | {r['square_interface_disconnected']['violating_cases']}/52 | {r['square_interface_disconnected']['observed_good_minus_bad_dice']:+.6f} |")
    angle_a, angle_b = ref["train"]["whole"]["coronal_normal_to_pca_axis_degrees"], ref["validation"]["whole"]["coronal_normal_to_pca_axis_degrees"]
    text += ["", "The Dice column compares different cases; it is **not an intervention gain**. Surface is exact exposed-voxel-face area. Other estimators and checkpoints can differ.", "",
        f"The mean whole-mask PCA-axis/coronal-normal angle is **{angle_a['mean']:.2f}° training / {angle_b['mean']:.2f}° validation**, with 5th–95th percentiles {angle_a['q05']:.2f}–{angle_a['q95']:.2f}° / {angle_b['q05']:.2f}–{angle_b['q95']:.2f}°. Thus approximately 24° is a cohort descriptor under this definition, not a fixed anatomical requirement.", "",
        "## Spectral checks and limitations", "",
        f"Maximum eigenpair residual: {results['numerical_checks']['max_eigen_residual']:.3g}; maximum orthogonality error: {results['numerical_checks']['max_eigen_orthogonality_error']:.3g}. Twelve combinatorial graph modes and four HKS times are retained per largest component, with per-case truncation bounds. These are volume-graph descriptors, not a registered anatomical frame or verified surface ShapeDNA. Full-graph λ₂=0 on disconnected cases; LCC λ₂ is separately labeled.", "",
        "![New property checks](../../../experiments/graph_property_inventory_20260923/property_checks.png)", "",
        "![Per-voxel descriptor example](../../../experiments/graph_property_inventory_20260923/descriptor_maps.png)", "",
        "See [the complete 44-property assessment](README.md), [machine-readable summary](summary.json), and [coordinate targets](coordinate_cases.json)."]
    path.write_text("\n".join(text) + "\n")


def plot_maps(maps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(maps).parent
    name = "hippocampus_025"
    with np.load(Path(maps) / "reference" / name / "whole.npz") as z:
        points, lcc = z["coordinates_ijk"], z["lcc_node_ids"]
        fields = {"Degree": z["degree"], "Depth (graph hops)": z["depth_hops"],
                  "Digital thickness (diameter)": z["digital_local_thickness_diameter"],
                  "Fiedler coordinate (LCC)": z["fiedler01_lcc"],
                  "Harmonic coordinate (LCC)": z["harmonic_lcc"],
                  "Betweenness (32 sampled sources)": z["betweenness_32_sources"]}
    fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    fig.suptitle(f"New per-voxel descriptors · {name} reference", fontsize=17, weight="bold")
    for index, (title, field) in enumerate(fields.items()):
        ax = fig.add_subplot(2, 3, index + 1, projection="3d")
        p = points[lcc] if "LCC" in title else points
        artist = ax.scatter(*p.T, c=field, s=4, cmap="viridis", alpha=.6)
        ax.set_title(title)
        ax.view_init(elev=25, azim=-40)
        ax.set_box_aspect(np.maximum(np.ptp(points, axis=0), 1))
        ax.set_xlabel("R")
        ax.set_ylabel("A")
        ax.set_zlabel("S")
        ax.tick_params(labelsize=7, pad=0)
        fig.colorbar(artist, ax=ax, shrink=.65, pad=.03, fraction=.035, orientation="horizontal")
    fig.savefig(output / "descriptor_maps.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiments/graph_property_inventory_20260923")
    args = parser.parse_args()
    manifest = json.loads((args.output / "manifest.json").read_text())
    cases = {}
    for job in manifest["jobs"]:
        path = args.output / "case_metrics" / job["source"] / f"{job['case']}.json"
        row = json.loads(path.read_text())
        assert row["source_sha256"] == job["sha256"]
        cases.setdefault(job["source"], {})[job["case"]] = row
    old = ROOT / "docs/experiments/voxel_graph_anatomy_20260923"
    old_manifest = json.loads((old / "manifest.json").read_text())
    if digest(old / "manifest.json") != manifest["prior_manifest_sha256"]:
        raise ValueError("Prior topology provenance changed")
    for job in manifest["jobs"]:
        if job["source"] == "reference":
            expected = old_manifest["inputs"][job["case"]]["source_sha256"]
        else:
            relative = Path("experiments/uncal_fold_early_stopping_20260921/voxel_audit") / job["source"] / "error_maps" / f"{job['case']}.npz"
            expected = old_manifest["prediction_hashes"][str(relative)]
        if job["sha256"] != expected:
            raise ValueError("Cannot pair changed predictions with prior topology results")
    refs = {c["case"]: c for c in json.loads((old / "cases.json").read_text())}
    predictions = json.loads((old / "prediction_cases.json").read_text())
    results = {"regions": {}, "prediction_comparisons": {}, "coordinate_fit": {}}
    for source, rows in cases.items():
        subsets = ("train", "validation") if source == "reference" else ("validation",)
        results["regions"][source] = {}
        for subset in subsets:
            selected = [c for c in rows.values() if c["split"] == subset]
            result = {}
            for region in ("whole", "anterior", "posterior"):
                records = [r["regions"][region] for r in selected]
                result[region] = dict(cases=len(records),
                    cases_with_degree_one_nodes=sum(r["leaves"] > 0 for r in records),
                    cases_with_branched_slice_graph=sum(r["slice_component_graph"]["branch_nodes"] > 0 for r in records),
                    cases_with_disconnected_graph=sum(r["components_6"] > 1 for r in records))
                for metric in ("surface_area_voxel_faces", "isoperimetric_ratio", "elongation_pca_std_ratio",
                               "coronal_normal_to_pca_axis_degrees", "diameter_largest_component",
                               "diameter_path_boundary_fraction", "lambda2_full_graph", "lambda2_lcc", "fiedler_y_spearman"):
                    result[region][metric] = describe([r[metric] for r in records])
                for metric in ("depth_hops", "edt_center_radius", "digital_local_thickness"):
                    result[region][metric + "_per_case_mean"] = describe([r[metric]["mean"] for r in records])
            results["regions"][source][subset] = result
    train = [r for r in cases["reference"].values() if r["split"] == "train"]
    bounds = {r: {m: [min(c["regions"][r][m] for c in train), max(c["regions"][r][m] for c in train)]
        for m in ("isoperimetric_ratio", "elongation_pca_std_ratio", "surface_area_voxel_faces", "lambda2_lcc")}
        for r in ("whole", "anterior", "posterior")}
    results["training_bounds"] = bounds
    for arm in ("baseline_seed0", "augmentation_seed0"):
        rows = cases[arm]
        stats = {}
        for region in ("whole", "anterior", "posterior"):
            ref = np.array([cases["reference"][n]["regions"][region]["surface_area_voxel_faces"] for n in rows])
            pred = np.array([c["regions"][region]["surface_area_voxel_faces"] for c in rows.values()])
            stats[region] = dict(surface_mean_relative_change=float(((pred - ref) / ref).mean()),
                surface_ratio_of_means_change=float(pred.mean() / ref.mean() - 1),
                surface_underestimated_cases=int((pred < ref).sum()),
                training_range_violations={metric: sum(not lo <= c["regions"][region][metric] <= hi for c in rows.values())
                    for metric, (lo, hi) in bounds[region].items()})
        for key, predicate in (
                ("square_interface_disconnected", lambda c: c["interface"]["beta0"] != 1),
                ("posterior_26_disconnected", lambda c: c["regions"]["posterior"]["cubical"]["beta0"] != 1),
                ("posterior_6_disconnected", lambda c: c["regions"]["posterior"]["components_6"] != 1)):
            good, bad = [], []
            for c in predictions[arm]:
                path = ROOT / "experiments/uncal_fold_early_stopping_20260921/voxel_audit" / arm / "error_maps" / f"{c['case']}.npz"
                with np.load(path) as z:
                    p, t = z["prediction"], z["ground_truth"]
                dice = np.mean([2 * ((p == cls) & (t == cls)).sum() / ((p == cls).sum() + (t == cls).sum()) for cls in (1, 2)])
                (bad if predicate(c) else good).append(dice)
            stats[key] = dict(violating_cases=len(bad),
                observed_good_minus_bad_dice=float(np.mean(good) - np.mean(bad)) if bad else None,
                note="Between-case association, NOT Dice gain from fixing topology")
        # Spatial coverage of candidate local targets: diagnostics on GT nodes,
        # never supplied as input to the predicted-mask descriptor calculation.
        strata = {k: dict(voxels=0, missed_foreground_voxels=0) for k in
                  ("degree_0_1", "degree_2_3", "degree_4_5", "degree_6", "depth_0", "depth_1", "depth_ge_2")}
        for name in rows:
            with np.load(Path(manifest["maps"]) / "reference" / name / "whole.npz") as z:
                points, degree, depth = z["coordinates_ijk"], z["degree"], z["depth_hops"]
            with np.load(ROOT / "experiments/voxel_graph_anatomy_20260923/graphs" / name / "whole.npz") as z:
                shape = z["shape"]
            path = ROOT / "experiments/uncal_fold_early_stopping_20260921/voxel_audit" / arm / "error_maps" / f"{name}.npz"
            with np.load(path) as z:
                pred = z["prediction"]
            offset = (np.array(pred.shape) - shape) // 2
            missing = pred[tuple((points + offset).T)] == 0
            selectors = dict(degree_0_1=degree <= 1, degree_2_3=(degree >= 2) & (degree <= 3),
                             degree_4_5=(degree >= 4) & (degree <= 5), degree_6=degree == 6,
                             depth_0=depth == 0, depth_1=depth == 1, depth_ge_2=depth >= 2)
            for key, selection in selectors.items():
                strata[key]["voxels"] += int(selection.sum())
                strata[key]["missed_foreground_voxels"] += int((selection & missing).sum())
        for record in strata.values():
            record["missed_fraction"] = record["missed_foreground_voxels"] / record["voxels"] if record["voxels"] else None
        stats["reference_node_error_strata"] = strata
        results["prediction_comparisons"][arm] = stats
    # GT supplies targets only here; whole-mask spectral/harmonic fields see no A/P labels.
    coordinate_rows = []
    for name, case in cases["reference"].items():
        with np.load(Path(manifest["maps"]) / "reference" / name / "whole.npz") as z:
            ids = z["lcc_node_ids"]
            fields = {k: z[k] for k in ("fiedler01_lcc", "fiedler_volume_quantile_lcc", "harmonic_lcc")}
        graph = ROOT / "experiments/voxel_graph_anatomy_20260923/graphs" / name / "whole.npz"
        with np.load(graph) as z:
            anterior = z["node_labels"][ids] == 1
        coordinate_rows.append(dict(case=name, split=case["split"], nodes=len(ids),
            fields={key: best_threshold(field, anterior) for key, field in fields.items()}))
    for key in ("fiedler01_lcc", "fiedler_volume_quantile_lcc", "harmonic_lcc"):
        threshold = float(np.median([c["fields"][key]["threshold"] for c in coordinate_rows if c["split"] == "train"]))
        record = dict(training_median_threshold=threshold)
        for subset in ("train", "validation"):
            selected = [c for c in coordinate_rows if c["split"] == subset]
            record[subset] = dict(oracle_fit_errors_total=sum(c["fields"][key]["errors"] for c in selected),
                oracle_fit_error_fraction_per_case=describe([c["fields"][key]["errors"] / c["nodes"] for c in selected]),
                oracle_threshold_distribution=describe([c["fields"][key]["threshold"] for c in selected]))
            errors = []
            for c in selected:
                with np.load(Path(manifest["maps"]) / "reference" / c["case"] / "whole.npz") as z:
                    ids, field = z["lcc_node_ids"], z[key]
                with np.load(ROOT / "experiments/voxel_graph_anatomy_20260923/graphs" / c["case"] / "whole.npz") as z:
                    anterior = z["node_labels"][ids] == 1
                errors.append(float(np.mean((field >= threshold) != anterior)))
            record[subset]["fixed_training_threshold_error_fraction"] = describe(errors)
        results["coordinate_fit"][key] = record
    results["numerical_checks"] = dict(graph_maps=len(manifest["jobs"]) * 3,
        max_eigen_residual=max(c["regions"][r]["spectral_diagnostics"]["residual_max"] for rows in cases.values() for c in rows.values() for r in ("whole", "anterior", "posterior")),
        max_eigen_orthogonality_error=max(c["regions"][r]["spectral_diagnostics"]["orthogonality_error"] for rows in cases.values() for c in rows.values() for r in ("whole", "anterior", "posterior")))
    maps_verified = 0
    for job in manifest["jobs"]:
        for region in ("whole", "anterior", "posterior"):
            metric = cases[job["source"]][job["case"]]["regions"][region]
            with np.load(Path(manifest["maps"]) / job["source"] / job["case"] / f"{region}.npz", allow_pickle=False) as z:
                n = len(z["coordinates_ijk"])
                assert n == metric["nodes"]
                for key in ("degree", "depth_hops", "edt_background_center_radius", "digital_local_thickness_diameter", "eccentricity_within_component"):
                    assert z[key].shape == (n,) and np.isfinite(z[key]).all()
                assert np.all(z["digital_local_thickness_diameter"] >= 2 * z["edt_background_center_radius"] - 1e-7)
                assert np.all((z["degree"] == 6) == (z["depth_hops"] > 0))
                ids = z["lcc_node_ids"]
                assert z["eigenvectors_lcc"].shape[0] == len(ids)
                assert z["hks_lcc"].shape == (len(ids), 4)
                assert np.isfinite(z["hks_lcc"]).all()
                assert np.isfinite(z["harmonic_lcc"]).all()
                assert len(z["diameter_path_node_ids"]) - 1 == metric["diameter_largest_component"]
            maps_verified += 1
    results["numerical_checks"]["maps_reopened_and_verified"] = maps_verified
    save_json(args.output / "summary.json", results)
    save_json(args.output / "coordinate_cases.json", coordinate_rows)
    save_json(args.output / "summary_provenance.json", dict(script_sha256=digest(__file__), manifest_sha256=digest(args.output / "manifest.json")))
    write_results(args.output / "RESULTS.md", results)
    plot_results(results, cases, manifest["maps"])
    plot_maps(manifest["maps"])
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
