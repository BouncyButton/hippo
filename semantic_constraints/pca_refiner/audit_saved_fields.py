"""Reconstruct validation band outputs from saved coefficients, without inference."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
import torch


def norm_summary(coefficients):
    norms = np.linalg.norm(np.stack(coefficients), axis=1)
    return dict(cases=len(norms), nonzero=int((norms > 0).sum()),
                minimum=float(norms.min()), median=float(np.median(norms)), maximum=float(norms.max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    train = args.run_root / "results"
    validation = args.run_root / "validation"
    basis = torch.load(train / "fixed_basis.pt", map_location="cpu", weights_only=False)
    modes = basis["modes"].numpy()
    offset = basis["error_mean"].numpy()
    train_coefficients = []
    for path in sorted((train / "cases").glob("*.npz")):
        with np.load(path) as saved:
            train_coefficients.append(saved["coefficients"])
    rows, coefficients = [], []
    for path in sorted((validation / "cases").glob("*.npz")):
        with np.load(path) as saved:
            p = saved["prediction"] > 0
            support = saved["support"]
            c = saved["coefficients"]
            coefficients.append(c)
            source = np.clip(np.where(p, -(ndi.distance_transform_edt(p) - .5),
                                      ndi.distance_transform_edt(~p) - .5), -8, 8).astype(np.float32)
            indices = np.flatnonzero(support)
            delta = c @ modes[:, indices] + offset[indices]
            assert np.isfinite(delta).all() and np.isfinite(c).all()
            corrected = p.copy()
            corrected[support] = source[support] + delta < 0
            assert np.array_equal(corrected, saved["refined"] > 0)
            signed_margin = np.where(p[support], -(source[support] + delta), source[support] + delta)
            rows.append(dict(case=path.stem, coefficient_norm=float(np.linalg.norm(c)),
                band_correction_min=float(delta.min()), band_correction_max=float(delta.max()),
                band_max_abs_correction=float(np.abs(delta).max()),
                minimum_margin_to_label_change=float(signed_margin.min()),
                reconstructed_changed_voxels=int((p != corrected).sum())))
    assert len(train_coefficients) == 208 and len(coefficients) == 52
    result = dict(training_coefficient_norms=norm_summary(train_coefficients),
        validation_coefficient_norms=norm_summary(coefficients),
        distinct_validation_coefficient_vectors=int(len(np.unique(np.stack(coefficients), axis=0))),
        maximum_absolute_validation_band_correction=float(max(r["band_max_abs_correction"] for r in rows)),
        smallest_validation_margin_to_label_change=float(min(r["minimum_margin_to_label_change"] for r in rows)),
        independently_reconstructed_masks_match_saved=True, cases=rows,
        interpretation="Corrections are additive signed-distance values in mm, not guaranteed physical displacements")
    (validation / "field_audit.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}), flush=True)


if __name__ == "__main__":
    main()
