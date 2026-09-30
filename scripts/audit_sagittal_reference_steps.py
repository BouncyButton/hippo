#!/usr/bin/env python3
"""Measure decreasing-step fit of upper/lower contours on sagittal sections."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from sklearn.isotonic import IsotonicRegression


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inference", type=Path,
        default=ROOT / "datasets/Dataset101_MSD/inference_fold0_val_600236",
    )
    parser.add_argument("--labels", type=Path, default=ROOT / "datasets/Dataset101_MSD/labelsTr")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader((args.inference / "prediction_index.csv").open()))
    residuals: list[float] = []
    endpoint_changes: list[float] = []
    section_count = 0
    disconnected_count = 0
    for row in rows:
        mask = np.asarray(nib.load(args.labels / f"{row['case_name']}.nii.gz").dataobj) > 0
        for section in mask:
            ys = np.flatnonzero(section.any(axis=1))
            if len(ys) < 3:
                continue
            section_count += 1
            disconnected_count += bool(np.any(np.diff(ys) > 1))
            lower = np.argmax(section, axis=1)
            upper = section.shape[1] - 1 - np.argmax(section[:, ::-1], axis=1)
            runs = np.split(ys, np.flatnonzero(np.diff(ys) > 1) + 1)
            for contour in (lower, upper):
                original = contour[ys].astype(float)
                fitted = np.empty_like(original)
                offset = 0
                for run in runs:
                    values = contour[run].astype(float)
                    if len(run) > 1:
                        values = IsotonicRegression(increasing=False).fit_transform(run, values)
                    fitted[offset:offset + len(run)] = values
                    offset += len(run)
                residuals.append(float(np.sqrt(np.mean((original - fitted) ** 2))))
                endpoint_changes.append(float(original[-1] - original[0]))
    report = {
        "schema": "hippo.sagittal_reference_step_fit.v1",
        "cases": len(rows),
        "sections_with_at_least_3_positions": section_count,
        "contour_functions": len(residuals),
        "fraction_sections_with_gaps": disconnected_count / section_count,
        "median_rmse_voxels": float(np.median(residuals)),
        "p90_rmse_voxels": float(np.quantile(residuals, 0.9)),
        "fraction_rmse_below_1_voxel": float(np.mean(np.asarray(residuals) < 1)),
        "fraction_endpoints_descend": float(np.mean(np.asarray(endpoint_changes) < 0)),
        "note": "Each contiguous y run in a sagittal section is fitted independently.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
