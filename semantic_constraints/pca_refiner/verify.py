"""Independently recompute metrics and preservation checks on training masks."""
import json
import sys
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi


def main():
    root = Path(sys.argv[1])
    result = json.loads((root / "results.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    count = protocol["regressor_subjects"]
    assert count in (8, 208) and len(protocol["selected"]) == count
    assert {r["case"] for r in result["cases"]} == set(protocol["selected"])
    if count == 208:
        assert set(protocol["selected"]) == set(protocol["split"]["train"])
    assert set(protocol["selected"]) <= set(protocol["split"]["train"])
    assert not set(protocol["selected"]) & set(protocol["split"]["val"])
    assert "hippocampus_164" not in protocol["selected"]
    for row in result["cases"]:
        data = np.load(root / "cases" / f"{row['case']}.npz")
        original, refined, truth = data["prediction"], data["refined"], data["truth"] > 0
        support = ndi.binary_dilation(original > 0, iterations=2) & ~ndi.binary_erosion(original > 0, iterations=2)
        assert np.array_equal(support, data["support"])
        assert np.array_equal(original[~support], refined[~support])
        retained = (original > 0) & (refined > 0)
        assert np.array_equal(original[retained], refined[retained])
        for name, mask in (("baseline", original > 0), ("learned", refined > 0), ("oracle", data["oracle"])):
            score = 2 * (mask & truth).sum() / (mask.sum() + truth.sum())
            assert abs(score - row[f"{name}_dice"]) < 1e-8
        assert row["fixed"] == int((((original > 0) != truth) & ((refined > 0) == truth)).sum())
        assert row["introduced"] == int((((original > 0) == truth) & ((refined > 0) != truth)).sum())
    for name in ("baseline", "learned", "oracle"):
        assert abs(np.mean([r[f"{name}_dice"] for r in result["cases"]]) - result["summary"][f"{name}_dice"]) < 1e-10
    verification = dict(training_cases=count, validation_overlap=0, case_164_excluded=True,
        independent_dice_and_error_counts=True, exact_outside_band_labels=True, exact_retained_ap_labels=True)
    (root / "verification.json").write_text(json.dumps(verification, indent=2))
    print(json.dumps(verification), flush=True)


if __name__ == "__main__":
    main()
