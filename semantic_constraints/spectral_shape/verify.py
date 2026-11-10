"""Independent checks of saved masks and summary arithmetic; run with sbatch."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output
    protocol = json.loads((root / "protocol.json").read_text())
    result = json.loads((root / "results.json").read_text())
    rows = json.loads((root / "case_metrics.json").read_text())
    train_names, val_names = set(protocol["split"]["train"]), set(protocol["split"]["val"])
    assert len(train_names) == 208 and len(val_names) == 52 and not train_names & val_names
    methods = sorted({r["method"] for r in rows})
    verified_masks = 0
    for method in methods:
        items = [r for r in rows if r["method"] == method]
        assert len(items) == 52 and {r["case"] for r in items} == val_names
        summary = result["methods"][method]
        assert abs(np.mean([r["dice"] for r in items]) - summary["mean_dice"]) < 1e-10
        assert sum(r["outside_band_changes"] for r in items) == 0
        assert all(r["fixed"] - r["introduced"] == (
            next(b for b in rows if b["case"] == r["case"] and b["method"] == "identity")["fp"] +
            next(b for b in rows if b["case"] == r["case"] and b["method"] == "identity")["fn"] - r["fp"] - r["fn"])
                   for r in items)
    for name in sorted(val_names):
        data = np.load(root / "cases" / f"{name}.npz")
        baseline, truth = data["prediction"], data["truth"]
        support = ndi.binary_dilation(baseline, iterations=2) & ~ndi.binary_erosion(baseline, iterations=2)
        assert np.array_equal(support, data["support"])
        for method in methods:
            if method not in data:
                continue
            mask = data[method]
            assert np.array_equal(mask[~support], baseline[~support])
            row = next(r for r in rows if r["case"] == name and r["method"] == method)
            dice = 2 * np.count_nonzero(mask & truth) / (np.count_nonzero(mask) + np.count_nonzero(truth))
            assert abs(row["dice"] - dice) < 1e-8
            assert row["fp"] == int((mask & ~truth).sum())
            assert row["fn"] == int((~mask & truth).sum())
            verified_masks += 1
        ceiling = np.where(support, truth, baseline)
        ceiling_row = next(r for r in rows if r["case"] == name and r["method"] == "band_ceiling_oracle")
        ceiling_dice = 2 * (ceiling & truth).sum() / (ceiling.sum() + truth.sum())
        assert abs(ceiling_row["dice"] - ceiling_dice) < 1e-8
    # Known comparison from the completed spline pilot, same CUDA FP32 baseline.
    baseline_matches_prior = abs(result["methods"]["identity"]["mean_dice"] - .9009775183944705) < 1e-8
    verification = dict(split_disjoint=True, method_count=len(methods), rows=len(rows),
        checked_saved_masks=verified_masks, independently_verified_bands_and_band_ceiling_cases=52,
        exact_preservation_on_saved_masks=True, summary_arithmetic_verified=True,
        baseline_mean_dice_matches_prior_run=baseline_matches_prior)
    (root / "verification.json").write_text(json.dumps(verification, indent=2))
    print(json.dumps(verification), flush=True)


if __name__ == "__main__":
    main()
