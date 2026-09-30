#!/usr/bin/env python3
"""Audit whether MSD anterior/posterior labels admit a coronal cut plane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def analyze_label(label: np.ndarray) -> dict[str, float | int | bool]:
    label = np.asarray(label)
    if label.ndim != 3 or not np.isin(label, (0, 1, 2)).all():
        raise ValueError("expected a 3-D MSD label with values 0, 1, 2")
    anterior_y = np.flatnonzero((label == 1).any(axis=(0, 2)))
    posterior_y = np.flatnonzero((label == 2).any(axis=(0, 2)))
    if not len(anterior_y) or not len(posterior_y):
        raise ValueError("both hippocampus classes are required")
    a_mean = float(np.where(label == 1)[1].mean())
    p_mean = float(np.where(label == 2)[1].mean())
    anterior_high = a_mean > p_mean
    foreground = label > 0
    y0, y1 = np.flatnonzero(foreground.any(axis=(0, 2)))[[0, -1]]
    contacts = ((label[:, :-1, :] == 1) & (label[:, 1:, :] == 2)) | (
        (label[:, :-1, :] == 2) & (label[:, 1:, :] == 1)
    )
    contact_y = np.flatnonzero(contacts.any(axis=(0, 2)))
    anterior_counts = (label == 1).sum(axis=(0, 2))
    posterior_counts = (label == 2).sum(axis=(0, 2))
    anterior_below = np.cumsum(anterior_counts)
    posterior_below = np.cumsum(posterior_counts)
    if anterior_high:
        wrong_per_cut = anterior_below + posterior_counts.sum() - posterior_below
    else:
        wrong_per_cut = posterior_below + anterior_counts.sum() - anterior_below
    cut = int(np.argmin(wrong_per_cut[:-1]))
    foreground_voxels = int(foreground.sum())
    return {
        "cut_y": cut,
        "cut_fraction_of_volume": float((cut + 0.5) / label.shape[1]),
        "cut_fraction_of_foreground_extent": float((cut + 0.5 - y0) / (y1 - y0 + 1)),
        "foreground_y_min": int(y0),
        "foreground_y_max": int(y1),
        "foreground_voxels": foreground_voxels,
        "wrong_side_voxels": int(wrong_per_cut[cut]),
        "wrong_side_fraction": float(wrong_per_cut[cut] / foreground_voxels),
        "anterior_is_high_y": anterior_high,
        "contact_voxels": int(contacts.sum()),
        "contact_y_positions": contact_y.tolist(),
        "contact_y_spread": int(np.ptp(contact_y)) if len(contact_y) else -1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.labels_dir.glob("*.nii.gz"))
    if not paths:
        raise ValueError("no label files found")
    cases = []
    for path in paths:
        label = np.asarray(nib.load(str(path)).dataobj, dtype=np.int8)
        cases.append({"case_name": path.name.removesuffix(".nii.gz"), **analyze_label(label)})
    wrong = np.asarray([case["wrong_side_fraction"] for case in cases])
    contact = np.asarray([case["contact_y_spread"] for case in cases])
    cut_fraction = np.asarray([case["cut_fraction_of_foreground_extent"] for case in cases])
    report = {
        "schema": "semantic_constraints.cst_teacher.ap_geometry.v1",
        "cases": cases,
        "summary": {
            "patients": len(cases),
            "planar_exact": int(np.sum(wrong == 0)),
            "planar_within_one_percent": int(np.sum(wrong <= 0.01)),
            "wrong_side_fraction_median": float(np.median(wrong)),
            "wrong_side_fraction_p95": float(np.quantile(wrong, 0.95)),
            "contact_y_spread_median": float(np.median(contact)),
            "contact_y_spread_p95": float(np.quantile(contact, 0.95)),
            "cut_fraction_of_foreground_extent_median": float(np.median(cut_fraction)),
            "cut_fraction_of_foreground_extent_p05": float(np.quantile(cut_fraction, 0.05)),
            "cut_fraction_of_foreground_extent_p95": float(np.quantile(cut_fraction, 0.95)),
            "anterior_high_y_count": int(sum(case["anterior_is_high_y"] for case in cases)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
