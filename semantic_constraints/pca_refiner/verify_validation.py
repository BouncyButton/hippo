"""Recompute validation Dice and constraints from saved label volumes."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi


def main():
    root = Path(sys.argv[1])
    result = json.loads((root / "results.json").read_text())
    assert len(result["cases"]) == 52
    assert {r["case"] for r in result["cases"]} == set(result["split"]["val"])
    assert not set(result["refiner_training_cases"]) & set(result["split"]["val"])
    assert result["validation_fitting"] is False and result["oracle_fitting"] is False
    for row in result["cases"]:
        with np.load(root / "cases" / f"{row['case']}.npz") as data:
            original, refined, truth = data["prediction"], data["refined"], data["truth"] > 0
            p, q = original > 0, refined > 0
            support = ndi.binary_dilation(p, iterations=2) & ~ndi.binary_erosion(p, iterations=2)
            assert np.array_equal(support, data["support"])
            assert np.array_equal(original[~support], refined[~support])
            retained = p & q
            assert np.array_equal(original[retained], refined[retained])
            for key, mask in (("baseline", p), ("refined", q)):
                score = 2 * (mask & truth).sum() / (mask.sum() + truth.sum())
                assert abs(score - row[f"{key}_dice"]) < 1e-8
            assert row["fixed"] == int(((p != truth) & (q == truth)).sum())
            assert row["introduced"] == int(((p == truth) & (q != truth)).sum())
            assert row["changed"] == row["fixed"] + row["introduced"]
    for key in ("baseline_dice", "refined_dice"):
        assert abs(np.mean([r[key] for r in result["cases"]]) - result["summary"][key]) < 1e-10
    verification = dict(cases=52, train_validation_disjoint=True,
        independent_dice_and_error_counts=True, exact_outside_band_labels=True,
        exact_retained_ap_labels=True)
    (root / "verification.json").write_text(json.dumps(verification, indent=2))
    print(json.dumps(verification), flush=True)


if __name__ == "__main__":
    main()
