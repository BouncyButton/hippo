"""Audit consistency AND local coverage of the uncal-interval candidate.

Fit 156/calibrate 52 from the 208 training crop IDs. Freeze everything before
evaluating the 52 reused validation IDs on GT and predicted foreground support.
No validation-based filtering, interval widening or reliability selection.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from .audit_refined import _load_cases, _median_prediction
from .audit_predicted_foreground import (crop_model_volume_to_native,
    largest_foreground_component, load_predicted_case)
from .interval_teacher import FoldIntervalTeacher, calibration_radius, label_transition_interval

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/uncal_interval_constraint_20260921"


def consistency(labels, support, centre, radius, local_radius=6.):
    y = np.broadcast_to(np.arange(labels.shape[1])[None, :, None], labels.shape)
    local = np.abs(y - centre) <= local_radius
    anterior = support & local & (y > centre + radius)
    posterior = support & local & (y < centre - radius)
    active = anterior | posterior
    claimed_gt = active & (labels > 0)
    wrong = ((anterior & (labels == 2)) | (posterior & (labels == 1)))
    reference = label_transition_interval(labels)
    reference_centre = .5 * sum(reference)
    band = (np.abs(y - reference_centre) <= 2) & (labels > 0)
    claimed_band = active & band
    band_count = int(claimed_band.sum())
    count = int(claimed_gt.sum())
    return {
        "interval_covers_reference": bool(centre-radius <= reference[0] and centre+radius >= reference[1]),
        "active_gt_voxels": count, "wrong_gt_voxels": int(wrong.sum()),
        "label_agreement": 1 - int(wrong.sum()) / count if count else None,
        "active_gt_fraction": count / int((labels > 0).sum()),
        "band_active_fraction": band_count / max(int(band.sum()), 1),
        "band_active_voxels": band_count, "band_wrong_voxels": int((wrong & band).sum()),
        "band_label_agreement": 1 - int((wrong & band).sum()) / band_count if band_count else None,
        "unsupported_foreground_voxels": int((active & (labels == 0)).sum()),
        "has_active_near_cut": bool(band_count), "has_contradiction": bool(wrong.any()),
    }


def aggregate(rows):
    result = {"cases": len(rows)}
    for key in ("interval_covers_reference", "active_gt_fraction", "band_active_fraction", "has_active_near_cut", "has_contradiction"):
        result[key] = float(np.mean([r[key] for r in rows]))
    for key in ("label_agreement", "band_label_agreement"):
        values = [r[key] for r in rows if r[key] is not None]
        result[key] = float(np.mean(values)) if values else None
        result[key + "_evaluated_cases"] = len(values)
    for key in ("active_gt_voxels", "wrong_gt_voxels", "band_active_voxels", "band_wrong_voxels", "unsupported_foreground_voxels"):
        result[key] = int(sum(r[key] for r in rows))
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dataset = ROOT / "datasets/Dataset101_MSD"
    split = json.loads((dataset / "splits_final.json").read_text())[0]
    train = sorted(split["train"])
    val = sorted(split["val"])
    if len(train) != 208 or len(val) != 52 or set(train) & set(val):
        raise ValueError("Expected the existing disjoint 208/52 fold-0 case split")
    shuffled = np.random.default_rng(20260921).permutation(train).tolist()
    fit, calibration = shuffled[:156], shuffled[156:]
    print("Extracting union-shape and T1 features for 260 cases", flush=True)
    cases = _load_cases(dataset)
    labels = {n: np.asarray(nib.load(dataset / "labelsTr" / f"{n}.nii.gz").dataobj) for n in train + val}
    for n in train + val:
        nii = nib.load(dataset / "labelsTr" / f"{n}.nii.gz")
        image = nib.load(dataset / "imagesTr" / f"{n}_0000.nii.gz")
        if not np.allclose(nii.affine[:3, :3], np.eye(3)) or not np.allclose(nii.affine, image.affine):
            raise ValueError(f"{n}: requires aligned native 1-mm RAS")
    teacher = FoldIntervalTeacher().fit(fit, cases)
    restored = FoldIntervalTeacher.from_dict(json.loads(json.dumps(teacher.as_dict())))
    predicted_calibration = np.array([teacher.predict_cut(cases[n]) - .5 for n in calibration])
    reference_calibration = np.array([label_transition_interval(labels[n]) for n in calibration])
    radius = calibration_radius(predicted_calibration, reference_calibration)
    position_calibration = np.array([_median_prediction(fit, cases[n], cases) - .5 for n in calibration])
    position_radius = calibration_radius(position_calibration, reference_calibration)
    payload = {"schema": "uncal_interval.v1", "seed": 20260921, "alpha": .10,
               "fit_cases": fit, "calibration_cases": calibration, "validation_cases": val,
               "radius_mm": radius, "position_radius_mm": position_radius,
               "local_radius_mm": 6., "teacher": teacher.as_dict(),
               "warning": "Weak label-derived locator, not verified uncal anatomy. GT-union calibration does not guarantee predicted-support coverage or participant independence."}
    # Save before reading any validation predictions; never refit after calibration.
    (OUT / "teacher.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Frozen calibration radii: shape/MRI {radius:g} mm, positional control {position_radius:g} mm", flush=True)
    rows = []
    prediction_root = ROOT / "experiments/uncal_fold_early_stopping_20260921/voxel_audit"
    for variant, names in (("calibration_gt", calibration), ("validation_gt", val),
                           ("validation_unaugmented", val), ("validation_augmented", val)):
        for name in names:
            truth = labels[name]
            case, support = cases[name], truth > 0
            if variant in ("validation_unaugmented", "validation_augmented"):
                folder = "baseline_seed0" if variant == "validation_unaugmented" else "augmentation_seed0"
                path = prediction_root / folder / "error_maps" / f"{name}.npz"
                case = load_predicted_case(name=name, error_map_path=path, dataset_root=dataset,
                                           ground_truth_case=cases[name]).case
                with np.load(path) as archive:
                    pred = crop_model_volume_to_native(archive["prediction"], truth.shape)
                support = largest_foreground_component(pred > 0)[0]
            cut = teacher.predict_cut(case)
            if restored.predict_cut(case) != cut:
                raise AssertionError("Portable teacher round-trip changed prediction")
            position_cut = _median_prediction(fit, case, cases)
            for method, centre, r in (("fold_point", cut-.5, 0.), ("fold_interval", cut-.5, radius),
                                      ("position_interval", position_cut-.5, position_radius)):
                rows.append({"variant": variant, "case": name, "method": method,
                             "centre_mm": centre, "radius_mm": r,
                             **consistency(truth, support, centre, r)})
        print(f"Completed {variant}", flush=True)
    reference_rows = []
    for name in train + val:
        lower, upper = label_transition_interval(labels[name])
        reference_rows.append(consistency(labels[name], labels[name] > 0,
                                          (lower + upper) / 2, (upper - lower) / 2))
    summary = {"protocol": payload, "reference_grounding": aggregate(reference_rows), "results": {}}
    for variant in sorted({r["variant"] for r in rows}):
        summary["results"][variant] = {method: aggregate([r for r in rows if r["variant"] == variant and r["method"] == method])
                                         for method in ("fold_point", "fold_interval", "position_interval")}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (OUT / "per_case.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Uncal interval constraint: consistency and coverage", "",
             f"156 fit / 52 calibration / 52 reused validation crops. Nominal 90% calibration; radius {radius:g} mm (fold), {position_radius:g} mm (position).",
             "No guarantee under support shift or unknown participant grouping. Near-cut band: ±2 mm. Active loss region: ±6 mm around the predicted plane.", "",
             "| Support | Rule | Reference interval covered | Label agreement on active GT | GT foreground active | Near-cut GT active | Cases with any near-cut activity |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for variant, methods in summary["results"].items():
        for method, m in methods.items():
            agreement = "n/a" if m["label_agreement"] is None else f"{100*m['label_agreement']:.2f}%"
            lines.append(f"| {variant} | {method} | {100*m['interval_covers_reference']:.1f}% | {agreement} | {100*m['active_gt_fraction']:.1f}% | {100*m['band_active_fraction']:.1f}% | {100*m['has_active_near_cut']:.1f}% |")
    lines += ["", "Agreement excludes background and inactive cases; activity and false support are reported separately in JSON. Calibration agreement is not generalization performance."]
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
