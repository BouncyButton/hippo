"""Independently check saved masks and render the completed pilot report."""
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
    results = json.loads((root / "results.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    rows = json.loads((root / "validation_cases.json").read_text())
    history = json.loads((root / "history.json").read_text())
    assert protocol["train_cases"] == 208 and protocol["val_cases"] == 52
    assert set(r["case"] for r in rows) == set(protocol["split"]["val"])
    assert not set(protocol["split"]["val"]) & set(protocol["split"]["train"])
    verified = []
    for row in rows:
        data = np.load(root / "validation_cases" / f"{row['case']}.npz")
        original, refined, truth = data["prediction"], data["refined"], data["truth"]
        a, b, g = original > 0, refined > 0, truth > 0
        support = ndi.binary_dilation(a, iterations=2) & ~ndi.binary_erosion(a, iterations=2)
        assert np.array_equal(support, data["support"])
        assert np.array_equal(original[~support], refined[~support])
        assert np.array_equal(original[a & b], refined[a & b])
        for key, pred in (("baseline", a), ("refined", b)):
            score = 2 * (pred & g).sum() / (pred.sum() + g.sum())
            assert abs(score - row[f"{key}_dice"]) < 1e-8
            assert int((pred & ~g).sum()) == row[f"{key}_fp"]
            assert int((~pred & g).sum()) == row[f"{key}_fn"]
        assert int(((a != g) & (b == g)).sum()) == row["fixed"]
        assert int(((a == g) & (b != g)).sum()) == row["introduced"]
        verified.append(row["case"])
    summary = results["validation"]
    assert summary["fixed"] - summary["introduced"] == (
        summary["baseline_fp"] + summary["baseline_fn"] - summary["refined_fp"] - summary["refined_fn"])
    verification = dict(verified_cases=verified, count=len(verified), exact_outside_band_preservation=True,
                        exact_retained_ap_preservation=True, independently_recomputed_dice_and_errors=True,
                        independently_recomputed_six_connected_bands=True)
    (root / "verification.json").write_text(json.dumps(verification, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].plot([r["epoch"] for r in history],
                    [r["validation"]["mean_refined_dice"] for r in history], label="Refined validation")
    axes[0, 0].axhline(summary["mean_baseline_dice"], color="gray", linestyle="--", label="Frozen baseline")
    axes[0, 0].axvline(results["best_epoch"], color="green", linestyle=":", label="Selected epoch")
    axes[0, 0].set(xlabel="Epoch", ylabel="Mean whole-hippocampus Dice", title="All 52 validation cases")
    axes[0, 0].legend(fontsize=8)
    deltas = np.sort([r["delta_dice"] * 100 for r in rows])
    axes[0, 1].bar(np.arange(len(deltas)), deltas, color=np.where(deltas >= 0, "seagreen", "indianred"))
    axes[0, 1].set(xlabel="Validation cases, sorted by gain", ylabel="Dice change (percentage points)",
                   title=f"{summary['improved']} improved / {summary['worsened']} worsened / {summary['unchanged']} unchanged")
    case = results["case_164"]
    if case is not None:
        data = np.load(root / "validation_cases" / f"{case['case']}.npz")
        truth = data["truth"] > 0
        z = int(np.argmax(((data["prediction"] > 0) != truth).sum((0, 1))))
        for ax, key, metric, color in ((axes[1, 0], "prediction", "baseline_dice", "deepskyblue"),
                                        (axes[1, 1], "refined", "refined_dice", "lime")):
            ax.imshow(data["image"][:, :, z].T, cmap="gray", origin="lower")
            ax.contour(truth[:, :, z].T, levels=[.5], colors=["tomato"], linewidths=1)
            ax.contour((data[key][:, :, z] > 0).T, levels=[.5], colors=[color], linewidths=1)
            ax.set_title(f"Case 164: {key}, Dice {case[metric]:.4f}")
            ax.set_xlabel(f"Slice {z}; annotation red, prediction {color}")
    fig.suptitle("Spline boundary refiner: fold 0 development pilot", fontsize=15)
    fig.tight_layout()
    fig.savefig(root / "overview.png", dpi=150)
    plt.close(fig)
    print(json.dumps(verification), flush=True)


if __name__ == "__main__":
    main()
