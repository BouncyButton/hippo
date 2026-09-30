"""Post-primary descriptor ablation on the original TRAINING folds only."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.probe_graph_landmark import SEED, fit, predict, row, endpoints, paired, prior_values
from evaluation.voxel_graph_anatomy import save_json, digest


def group_for(name):
    if name.startswith(("harmonic_", "fiedler01_", "geodesic_u_", "intrinsic_valid_fraction_")):
        return "intrinsic_positions"
    if name.startswith(("components_", "branch_mass_", "geodesic_components_", "geodesic_branch_mass_")):
        return "branching"
    if name.startswith(("betweenness_", "eccentricity_")):
        return "traffic_eccentricity"
    if name.startswith("geodesic_"):
        return "geodesic_profiles"
    if name.startswith(("degree", "depth_", "thickness_")):
        return "degree_depth_thickness"
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiments/graph_landmark_20260924")
    parser.add_argument("--cache", type=Path, default=ROOT / "experiments/graph_landmark_20260924/features")
    args = parser.parse_args()
    split = json.loads((ROOT / "datasets/Dataset101_MSD/splits_final.json").read_text())[0]
    cases, hashes = [], {}
    for name in split["train"]:
        path = args.cache / "reference" / f"{name}.npz"
        with np.load(path, allow_pickle=False) as z:
            case = dict(z)
        case.update(name=str(case["name"]), target=int(case["target"]))
        cases.append(case)
        hashes[name] = digest(path)
    groups = np.array([group_for(n) for n in cases[0]["graph_names"]])
    selections = dict(full=np.ones(len(groups), bool), intrinsic_only=groups == "intrinsic_positions")
    selections.update({f"without_{g}": groups != g for g in sorted(set(groups))})
    results = {name: [] for name in selections}
    with threadpool_limits(limits=1):
        for fold, (train_ids, test_ids) in enumerate(KFold(4, shuffle=True, random_state=SEED).split(cases)):
            for name, keep in selections.items():
                subset = [dict(c, graph=c["graph"][:, keep]) for c in cases]
                model = fit([subset[i] for i in train_ids], "combined")
                results[name].extend(row(subset[i], predict(subset[i], model, "combined"), cv_fold=fold) for i in test_ids)
            print(f"Training-only ablation fold {fold+1}/4", flush=True)
        primary = json.loads((args.output / "training_oof.json").read_text())["combined"]
        assert [r["cut"] for r in results["full"]] == [r["cut"] for r in primary]
        model = fit(cases, "combined")
    names = ([f"position_{i}" for i in range(cases[0]["position"].shape[1])]
        + cases[0]["shape_names"].tolist() + cases[0]["graph_names"].tolist())
    coefficients = sorted([dict(feature=n, standardized_coefficient=float(w))
        for n, w in zip(names, model[-1].coef_[0])], key=lambda r: -abs(r["standardized_coefficient"]))
    summary = dict(methods={name: endpoints(rows) for name, rows in results.items()},
        removal_minus_full={name: paired(rows, results["full"]) for name, rows in results.items() if name != "full"},
        priors=prior_values(cases), groups={g: cases[0]["graph_names"][groups == g].tolist() for g in sorted(set(groups))},
        coefficients=coefficients,
        primary_combined_vs_median_position=paired(primary, json.loads((args.output / "training_oof.json").read_text())["median_position"]))
    save_json(args.output / "ablation_summary.json", summary)
    save_json(args.output / "ablation_cases.json", results)
    save_json(args.output / "ablation_manifest.json", dict(script_sha256=digest(__file__),
        protocol_sha256=digest(args.output / "ABLATION_PROTOCOL.md"),
        primary_selection_sha256=digest(args.output / "selection.json"), training_cache_hashes=hashes))
    print(json.dumps(summary["methods"], indent=2))


if __name__ == "__main__":
    main()
