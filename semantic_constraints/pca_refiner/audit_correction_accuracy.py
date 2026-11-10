"""Measure saved correction directions and fixed amplification diagnostics."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
import torch

CONSTANT_MAGNITUDES = (.25, .6, .8, 1., 1.2, 1.6, 2.)


def sdf(mask):
    return np.clip(np.where(mask, -(ndi.distance_transform_edt(mask) - .5),
                           ndi.distance_transform_edt(~mask) - .5), -8, 8).astype(np.float32)


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def dice(mask, truth):
    return float(2 * (mask & truth).sum() / (mask.sum() + truth.sum()))


def mask_score(p, g, support, source, correction):
    mask = p.copy()
    mask[support] = source[support] + correction < 0
    assert np.array_equal(mask[~support], p[~support])
    changed = mask != p
    fixed = int((changed & (mask == g)).sum())
    introduced = int((changed & (mask != g)).sum())
    assert fixed + introduced == int(changed.sum())
    added, removed = mask & ~p, p & ~mask
    return dict(dice=dice(mask, g), gain=dice(mask, g)-dice(p, g),
        fixed=fixed, introduced=introduced, changed=int(changed.sum()),
        correct_additions=int((added & g).sum()), incorrect_additions=int((added & ~g).sum()),
        correct_removals=int((removed & ~g).sum()), incorrect_removals=int((removed & g).sum()))


def summarize(rows, magnitudes):
    methods = {}
    for method in rows[0]["methods"]:
        items = [r["methods"][method] for r in rows]
        gains = np.array([r["gain"] for r in items])
        fixed, introduced = sum(r["fixed"] for r in items), sum(r["introduced"] for r in items)
        bootstrap = np.random.default_rng(0).choice(gains, (5000, len(rows)), replace=True).mean(1)
        methods[method] = dict(mean_dice=float(np.mean([r["dice"] for r in items])),
            mean_gain=float(gains.mean()), paired_bootstrap_95_ci=np.quantile(bootstrap, [.025, .975]).tolist(),
            fixed=fixed, introduced=introduced, edit_precision=ratio(fixed, fixed+introduced),
            improved=int((gains > 1e-10).sum()), worsened=int((gains < -1e-10).sum()),
            unchanged=int((np.abs(gains) <= 1e-10).sum()))
        for key in ("correct_additions", "incorrect_additions", "correct_removals", "incorrect_removals"):
            methods[method][key] = sum(r[key] for r in items)
    counts = {key: sum(r[key] for r in rows) for key in (
        "total_errors", "band_errors", "band_false_positives", "band_false_negatives",
        "fp_direction_right", "fn_direction_right", "error_direction_zero", "band_voxels",
        "target_nonzero_voxels", "target_sign_agrees", "distance_error_reduced_voxels")}
    energy = sum(r["target_squared_norm"] for r in rows)
    predicted_energy = sum(r["predicted_squared_norm"] for r in rows)
    dot = sum(r["predicted_target_dot"] for r in rows)
    return dict(cases=len(rows), baseline_dice=float(np.mean([r["baseline_dice"] for r in rows])),
        **counts,
        right_direction_on_errors=ratio(counts["fp_direction_right"]+counts["fn_direction_right"], counts["band_errors"]),
        right_direction_on_fp=ratio(counts["fp_direction_right"], counts["band_false_positives"]),
        right_direction_on_fn=ratio(counts["fn_direction_right"], counts["band_false_negatives"]),
        signed_distance_direction_agreement=ratio(counts["target_sign_agrees"], counts["target_nonzero_voxels"]),
        baseline_band_sdf_mse=ratio(energy, counts["band_voxels"]),
        corrected_band_sdf_mse=ratio(sum(r["remaining_squared_norm"] for r in rows), counts["band_voxels"]),
        pooled_field_cosine=ratio(dot, np.sqrt(energy*predicted_energy)),
        absolute_correction_quantiles_mm=dict(zip(("minimum", "median", "p90", "p99", "maximum"),
            np.quantile(np.concatenate(magnitudes), [0, .5, .9, .99, 1]).tolist())),
        methods=methods)


def analyze(root, modes, offset):
    rows, magnitudes = [], []
    for path in sorted((root / "cases").glob("*.npz")):
        with np.load(path) as saved:
            p, g, support = saved["prediction"] > 0, saved["truth"] > 0, saved["support"]
            source, target = sdf(p), sdf(g)
            indices = np.flatnonzero(support)
            delta = saved["coefficients"] @ modes[:, indices] + offset[indices]
            reconstructed = p.copy()
            reconstructed[support] = source[support] + delta < 0
            assert np.array_equal(reconstructed, saved["refined"] > 0)
            assert np.isfinite(delta).all()
            residual = target[support] - source[support]
            pb, gb = p[support], g[support]
            fp, fn = pb & ~gb, ~pb & gb
            err = fp | fn
            needed = np.abs(residual) > 1e-6
            row = dict(case=path.stem, baseline_dice=dice(p, g),
                total_errors=int((p != g).sum()), band_errors=int(err.sum()),
                band_false_positives=int(fp.sum()), band_false_negatives=int(fn.sum()),
                fp_direction_right=int((fp & (delta > 0)).sum()),
                fn_direction_right=int((fn & (delta < 0)).sum()),
                error_direction_zero=int((err & (delta == 0)).sum()),
                band_voxels=int(support.sum()), target_nonzero_voxels=int(needed.sum()),
                target_sign_agrees=int((needed & (delta*residual > 0)).sum()),
                distance_error_reduced_voxels=int((np.abs(residual-delta) < np.abs(residual)).sum()),
                target_squared_norm=float(np.square(residual.astype(np.float64)).sum()),
                predicted_squared_norm=float(np.square(delta.astype(np.float64)).sum()),
                remaining_squared_norm=float(np.square((residual-delta).astype(np.float64)).sum()),
                predicted_target_dot=float(np.dot(delta.astype(np.float64), residual.astype(np.float64))),
                methods={})
            magnitudes.append(np.abs(delta))
            for gain in (1, 2, 4, 8):
                row["methods"][f"scale_{gain}"] = mask_score(p, g, support, source, gain*delta)
            for magnitude in CONSTANT_MAGNITUDES:
                row["methods"][f"sign_{magnitude:g}mm"] = mask_score(p, g, support, source, magnitude*np.sign(delta))
            row["methods"]["sign_mean_offset"] = mask_score(p, g, support, source, np.sign(offset[indices]))
            rows.append(row)
        if len(rows) % 40 == 0:
            print("AUDIT", root.name, len(rows), flush=True)
    return rows, summarize(rows, magnitudes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    destination = args.run_root / "correction_accuracy.json"
    assert not destination.exists()
    basis = torch.load(args.run_root / "results" / "fixed_basis.pt", map_location="cpu", weights_only=False)
    modes, offset = basis["modes"].numpy(), basis["error_mean"].numpy()
    train_rows, train = analyze(args.run_root / "results", modes, offset)
    assert len(train_rows) == 208
    # This rule is fixed before validation is read. The complete sweep remains
    # an exploratory diagnostic, not an independent test-set performance claim.
    candidates = ["scale_1", "scale_2", "scale_4", "scale_8"] + [f"sign_{c:g}mm" for c in CONSTANT_MAGNITUDES]
    selected = max(candidates,
                   key=lambda name: train["methods"][name]["mean_dice"])
    val_rows, validation = analyze(args.run_root / "validation", modes, offset)
    assert len(val_rows) == 52
    assert not {r["case"] for r in train_rows} & {r["case"] for r in val_rows}
    assert abs(train["baseline_dice"] - .9383332742805718) < 1e-9
    assert abs(validation["baseline_dice"] - .9009775183944705) < 1e-9
    assert validation["methods"]["scale_1"]["fixed"] == 0
    assert validation["methods"]["scale_1"]["introduced"] == 0
    result = dict(training=train, validation=validation, selected_using_training_only=selected,
        training_cases=train_rows, validation_cases=val_rows,
        definitions=dict(edit_precision="fixed / (fixed + introduced); undefined if no edits",
            right_direction_on_errors="positive correction on false positives; negative on false negatives; pooled within prediction bands",
            amplification="multiply complete predicted SDF correction including mean-error offset",
            sign_constants="replace complete correction by c times its sign; c in .25, .6, .8, 1, 1.2, 1.6, 2 mm; zero stays zero; avoid half-voxel constants that land exactly on zero for grid-aligned boundaries",
            sign_mean_offset="annotation-free-at-inference control using the fixed training mean-error direction alone",
            caution="Post-hoc descriptive analysis of previously examined validation data; no model retraining or validation-selected scale"))
    destination.write_text(json.dumps(result, indent=2, allow_nan=False))
    print("COMPLETE", json.dumps({key: result[key] for key in ("training", "validation", "selected_using_training_only")}), flush=True)


if __name__ == "__main__":
    main()
