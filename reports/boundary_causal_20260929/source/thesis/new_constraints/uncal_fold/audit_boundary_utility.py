"""Measure whether cached landmark cuts improve MSD A/P boundary segmentation.

Exploratory counterfactual, not a training or independent test experiment.
All methods retain the exact original predicted foreground, including islands.
Only the explicitly named oracle uses validation labels to choose a cut.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from .audit_predicted_foreground import crop_model_volume_to_native, largest_foreground_component
from .foldedness import best_fit_first_anterior_slice

ROOT = Path(__file__).resolve().parents[3]


def partition_foreground(prediction: np.ndarray, cut: int) -> np.ndarray:
    """In verified RAS arrays, label 1 starts at y=cut (inclusive)."""
    if prediction.ndim != 3 or not 0 < cut < prediction.shape[1]:
        raise ValueError("Expected 3-D prediction and an interior axis-1 cut")
    side = np.where(np.arange(prediction.shape[1])[None, :, None] >= cut, 1, 2)
    return np.where(prediction > 0, side, 0).astype(np.uint8)


def score_boundary(prediction: np.ndarray, truth: np.ndarray, cut: int,
                   spacing: float, band_mm: float) -> dict[str, float | int]:
    """Score a fixed GT-centred band; the plane lies between cut-1 and cut."""
    band = np.broadcast_to(
        (np.abs(np.arange(truth.shape[1]) - (cut - 0.5)) * spacing <= band_mm)[None, :, None],
        truth.shape,
    )
    if not (band & (truth > 0)).any():
        raise ValueError("Boundary band contains no ground-truth foreground")

    def dice(a: np.ndarray, b: np.ndarray) -> float:
        denominator = int(a.sum() + b.sum())
        return 2 * int((a & b).sum()) / denominator if denominator else 1.0

    intersection = band & (truth > 0) & (prediction > 0)
    swaps = int((intersection & (truth != prediction)).sum())
    gt_band = band & (truth > 0)
    return {
        "mean_ap_dice": float(np.mean([dice(prediction == c, truth == c) for c in (1, 2)])),
        "band_mean_ap_dice": float(np.mean([
            dice(band & (prediction == c), band & (truth == c)) for c in (1, 2)
        ])),
        "union_dice": dice(prediction > 0, truth > 0),
        "band_ap_swap_voxels": swaps,
        "band_shared_foreground_voxels": int(intersection.sum()),
        "band_gt_foreground_voxels": int(gt_band.sum()),
        "band_ap_swap_rate": swaps / int(intersection.sum()) if intersection.any() else float("nan"),
        "band_gt_error_rate": int((gt_band & (prediction != truth)).sum()) / int(gt_band.sum()),
        "band_false_positive_voxels": int((band & (truth == 0) & (prediction > 0)).sum()),
    }


def paired_delta(values: np.ndarray) -> dict[str, object]:
    """Case bootstrap is descriptive; participant grouping is unavailable."""
    rng = np.random.default_rng(20260921)
    samples = rng.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
    return {"mean": float(values.mean()), "case_bootstrap_95_interval": np.quantile(samples, [.025, .975]).tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/uncal_boundary_utility_20260921")
    parser.add_argument("--band-mm", type=float, default=2.0)
    args = parser.parse_args()
    if args.band_mm <= 0 or not np.isfinite(args.band_mm):
        parser.error("band-mm must be positive and finite")
    dataset = ROOT / "datasets/Dataset101_MSD"
    docs = ROOT / "docs/experiments"
    feature_path = docs / "uncal_feature_probe_20260921/per_case.csv"
    shape_path = docs / "uncal_foldedness_predicted_foreground_early_stopping_20260921/case_results.csv"
    with feature_path.open() as f:
        features = list(csv.DictReader(f))
    with shape_path.open() as f:
        shapes = {r["case_name"]: r for r in csv.DictReader(f)}
    split_path = dataset / "splits_final.json"
    split = json.loads(split_path.read_text())[0]
    validation = set(split["val"])
    if set(split["train"]) & validation:
        raise ValueError("Training and validation cases overlap")
    geometry_rows = []
    for path in sorted((dataset / "labelsTr").glob("*.nii.gz")):
        nii = nib.load(path)
        if not np.allclose(nii.affine[:3, :3], np.eye(3)):
            raise ValueError(f"{path}: this audit requires native axis-aligned 1-mm RAS")
        labels = np.asarray(nii.dataobj)
        cut, cost, gap = best_fit_first_anterior_slice(labels)
        mixed = ((labels == 1).any(axis=(0, 2)) & (labels == 2).any(axis=(0, 2))).sum()
        geometry_rows.append({"case": path.name.removesuffix(".nii.gz"), "cut": cut,
                              "plane_disagreement_voxels": cost, "second_best_gap": gap,
                              "mixed_slices": int(mixed)})
    all_rows, model_summaries = [], {}
    for model, folder in (("unaugmented", "baseline_seed0"), ("augmented", "augmentation_seed0")):
        model_features = [r for r in features if r["model"] == model]
        if len(model_features) != len(validation) or {r["case"] for r in model_features} != validation:
            raise ValueError("Feature rows must match the entire fold-0 validation split")
        model_rows = []
        for row in model_features:
            name = row["case"]
            native = np.asarray(nib.load(dataset / "labelsTr" / f"{name}.nii.gz").dataobj)
            path = ROOT / "experiments/uncal_fold_early_stopping_20260921/voxel_audit" / folder / "error_maps" / f"{name}.npz"
            with np.load(path) as archive:
                gt, pred = archive["ground_truth"], archive["prediction"]
            if not np.array_equal(crop_model_volume_to_native(gt, native.shape), native):
                raise ValueError(f"{name}: padding/native geometry mismatch")
            target, cost, _ = best_fit_first_anterior_slice(gt)
            offset = (gt.shape[1] - native.shape[1]) // 2
            if target != int(row["target_cut"]) or target != int(shapes[name]["target_cut"]) + offset:
                raise ValueError(f"{name}: cut coordinate mismatch")
            union, _, _ = largest_foreground_component(pred > 0)
            native_cut, _, _ = best_fit_first_anterior_slice(np.where(union, pred, 0))
            cuts = {"native": native_cut, "native_plane": native_cut,
                    "feature_plane": int(row["feature_only_cut"]),
                    "geometry_image_plane": int(shapes[name][f"{model}_cut"]) + offset,
                    "median_position_plane": int(shapes[name][f"{model}_median_cut"]) + offset,
                    "oracle_label_plane": target}
            for method, cut in cuts.items():
                corrected = pred if method == "native" else partition_foreground(pred, cut)
                if not np.array_equal(corrected > 0, pred > 0):
                    raise AssertionError("Foreground changed")
                model_rows.append({"model": model, "case": name, "method": method,
                                   "target_cut": target, "predicted_cut": cut,
                                   "cut_absolute_error_mm": abs(cut - target),
                                   "gt_plane_disagreement_voxels": cost,
                                   **score_boundary(corrected, gt, target, 1.0, args.band_mm)})
        summary = {}
        baseline = {r["case"]: r for r in model_rows if r["method"] == "native"}
        for method in cuts:
            selected = [r for r in model_rows if r["method"] == method]
            summary[method] = {"count": len(selected)}
            for metric in ("cut_absolute_error_mm", "mean_ap_dice", "band_mean_ap_dice", "union_dice", "band_ap_swap_rate", "band_gt_error_rate"):
                summary[method][metric] = float(np.nanmean([r[metric] for r in selected]))
            differences = np.array([r["band_mean_ap_dice"] - baseline[r["case"]]["band_mean_ap_dice"] for r in selected])
            summary[method]["band_dice_delta_vs_native"] = paired_delta(differences)
            summary[method]["better_equal_worse_band_dice"] = [int((differences > 1e-12).sum()), int((np.abs(differences) <= 1e-12).sum()), int((differences < -1e-12).sum())]
        model_summaries[model] = summary
        all_rows.extend(model_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_case.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    result = {"band_half_width_mm": args.band_mm,
              "protocol": "Exploratory cached validation audit; no training, selection, or threshold tuning. Original predicted foreground preserved. Oracle uses evaluation labels.",
              "bootstrap_unit": "MSD crop/case; participant mapping unavailable",
              "input_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (feature_path, shape_path, split_path)},
              "geometry": {"case_count": len(geometry_rows), "nonplanar_cases": sum(r["plane_disagreement_voxels"] > 0 for r in geometry_rows),
                           "plane_disagreement_voxels": sum(r["plane_disagreement_voxels"] for r in geometry_rows), "per_case": geometry_rows},
              "models": model_summaries}
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = ["# Landmark boundary utility audit", "", f"Fixed ±{args.band_mm:g} mm band around the label-derived plane. All metrics are case means.", "",
             "| Model | Method | Cut MAE mm | A/P Dice | Band A/P Dice | Band swap % | Band Dice delta [95% case bootstrap] |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for model, methods in model_summaries.items():
        for method, m in methods.items():
            d = m["band_dice_delta_vs_native"]
            lo, hi = d["case_bootstrap_95_interval"]
            lines.append(f"| {model} | {method} | {m['cut_absolute_error_mm']:.3f} | {m['mean_ap_dice']:.4f} | {m['band_mean_ap_dice']:.4f} | {100*m['band_ap_swap_rate']:.2f} | {d['mean']:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
    lines += ["", "Swap rate conditions on shared foreground; band Dice includes foreground misses and false positives.",
              "The oracle is a label-derived best plane, not an independent anatomical annotation or an achievable test result.",
              "Repeated use of this validation fold makes this exploratory. Case bootstrap does not establish participant-level independence."]
    (args.output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
