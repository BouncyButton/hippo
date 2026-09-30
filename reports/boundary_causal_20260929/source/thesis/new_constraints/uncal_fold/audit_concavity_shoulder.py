"""Fold-0 audit of a label-blind sagittal concavity shoulder.

See docs/experiments/concavity_shoulder_fold0_20260923/PROTOCOL.md.
All detector coordinates are native y voxel indices (verified 1 mm RAS).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from sklearn.model_selection import KFold

from .audit_predicted_foreground import largest_foreground_component
from .foldedness import best_fit_first_anterior_slice

ROOT = Path(__file__).resolve().parents[3]
SEED = 20260923
ALPHAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
METHODS = ("position", "silhouette", "sagittal", "shoulder_raw", "shoulder_offset", "hybrid")


@dataclass
class Shape:
    low: int
    high: int
    candidates: np.ndarray
    silhouette: np.ndarray
    sagittal: np.ndarray
    consensus: np.ndarray
    persistence: np.ndarray
    eligible_slices: np.ndarray
    profile_y: np.ndarray
    profile_z: np.ndarray
    removed_voxels: int


def smooth_profile(profile: np.ndarray) -> np.ndarray:
    """Smooth contiguous finite runs independently, never fill missing tissue."""
    out = np.full_like(profile, np.nan, dtype=float)
    runs, count = ndimage.label(np.isfinite(profile))
    for index in range(1, count + 1):
        ix = np.flatnonzero(runs == index)
        out[ix] = ndimage.gaussian_filter1d(profile[ix].astype(float), 1.0, mode="nearest")
    return out


def profile_shoulders(profile: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Positive steep-to-flat slope change across a separator at c-0.5."""
    out = np.full(len(candidates), np.nan)
    x = np.arange(4, dtype=float) - 1.5
    for i, c in enumerate(candidates):
        if c < 4 or c + 4 > len(profile):
            continue
        before, after = profile[c - 4:c], profile[c:c + 4]
        if not np.isfinite(np.r_[before, after]).all():
            continue
        posterior, anterior = before @ x / (x @ x), after @ x / (x @ x)
        # Clip at zero: a uniformly sloping or convex contour is not a shoulder.
        out[i] = max(0.0, anterior - posterior) if posterior < -1e-8 else 0.0
    return out


def extract_shape(mask: np.ndarray, origin_y: int = 0) -> Shape:
    """Binary-only input: never receives reference cut or A/P class identity."""
    if np.asarray(mask).ndim != 3 or np.asarray(mask).dtype != bool:
        raise ValueError("expected a 3-D Boolean foreground union")
    mask, _, removed = largest_foreground_component(mask)
    occupied = np.flatnonzero(mask.any(axis=(0, 2)))
    low, high = int(occupied[0]), int(occupied[-1])
    if high - low < 7:
        raise ValueError("foreground too short for two four-sample windows")
    candidates = np.arange(low + 4, high - 2)
    z = np.arange(mask.shape[2])
    tops = np.where(mask, z[None, None, :], -1).max(axis=2).astype(float)
    tops[tops < 0] = np.nan
    projected = np.where(mask.any(axis=0), z[None, :], -1).max(axis=1).astype(float)
    projected[projected < 0] = np.nan
    profile = smooth_profile(projected)
    silhouette = np.nan_to_num(profile_shoulders(profile, candidates), nan=0.0)
    slice_scores = np.stack([profile_shoulders(smooth_profile(p), candidates) for p in tops])
    eligible = np.isfinite(slice_scores).sum(axis=0)
    positive = (slice_scores > 1e-8).sum(axis=0)
    persistence = positive / np.maximum(eligible, 1)
    sagittal = np.nansum(slice_scores, axis=0) / np.maximum(eligible, 1) * persistence
    consensus = np.sqrt(silhouette * sagittal)
    return Shape(low + origin_y, high + origin_y, candidates + origin_y,
                 silhouette, sagittal, consensus, persistence, eligible,
                 np.arange(len(profile)) + origin_y, profile, removed)


def pick(candidates: np.ndarray, scores: np.ndarray) -> int:
    tied = np.flatnonzero(np.isclose(scores, np.max(scores), atol=1e-10, rtol=0))
    return int(candidates[tied[len(tied) // 2]])


def raw_cut(shape: Shape) -> int:
    return pick(shape.candidates, shape.consensus)


def fit_parameters(names: list[str], shapes: dict[str, Shape], targets: dict[str, int]) -> dict:
    positions = [(targets[n] - shapes[n].low) / (shapes[n].high - shapes[n].low) for n in names]
    median = float(np.median(positions))
    scale = max(0.075, float(1.4826 * np.median(np.abs(np.asarray(positions) - median))))
    offset = int(np.rint(np.median([targets[n] - raw_cut(shapes[n]) for n in names])))
    return {"relative_median": median, "relative_scale": scale, "offset_mm": offset}


def predict(shape: Shape, parameters: dict, alpha: float) -> dict[str, int]:
    full_candidates = np.arange(shape.low + 1, shape.high + 1)
    relative = (full_candidates - shape.low) / (shape.high - shape.low)
    position = int(full_candidates[np.argmin(np.abs(relative - parameters["relative_median"]))])
    raw = raw_cut(shape)
    # Offset is learned from training only and applied before the positional term.
    corrected = np.clip(shape.candidates + parameters["offset_mm"], shape.low + 1, shape.high)
    rel = (corrected - shape.low) / (shape.high - shape.low)
    standardized = (shape.consensus - shape.consensus.mean()) / max(float(shape.consensus.std()), 1e-8)
    prior = -0.5 * ((rel - parameters["relative_median"]) / parameters["relative_scale"]) ** 2
    return {"position": position,
            "silhouette": pick(shape.candidates, shape.silhouette),
            "sagittal": pick(shape.candidates, shape.sagittal),
            "shoulder_raw": raw,
            "shoulder_offset": int(np.clip(raw + parameters["offset_mm"], shape.low + 1, shape.high)),
            "hybrid": pick(corrected, standardized + alpha * prior)}


def select_alpha(names: list[str], shapes: dict[str, Shape], targets: dict[str, int], folds: int) -> tuple[float, dict]:
    errors = {a: [] for a in ALPHAS}
    for fit_ix, test_ix in KFold(folds, shuffle=True, random_state=SEED).split(names):
        params = fit_parameters([names[i] for i in fit_ix], shapes, targets)
        for i in test_ix:
            n = names[i]
            for alpha in ALPHAS:
                errors[alpha].append(abs(predict(shapes[n], params, alpha)["hybrid"] - targets[n]))
    means = {a: float(np.mean(e)) for a, e in errors.items()}
    return min(ALPHAS, key=lambda a: (means[a], a)), means


def error_metrics(predicted: list[int], target: list[int]) -> dict:
    e = np.abs(np.asarray(predicted) - np.asarray(target))
    return {"n": len(e), "mae_mm": float(e.mean()), "median_mm": float(np.median(e)),
            "exact": float(np.mean(e == 0)), "within_1": float(np.mean(e <= 1)),
            "within_2": float(np.mean(e <= 2)), "p90_mm": float(np.quantile(e, 0.9)),
            "max_mm": int(e.max()), "over_3": int(np.sum(e > 3))}


def paired(first: list[int], second: list[int], targets: list[int]) -> dict:
    a, b, t = np.asarray(first), np.asarray(second), np.asarray(targets)
    diff = np.abs(b - t) - np.abs(a - t)
    rng = np.random.default_rng(SEED)
    boot = diff[rng.integers(0, len(diff), (5000, len(diff)))].mean(axis=1)
    return {"second_minus_first_mae_mm": float(diff.mean()),
            "bootstrap95_mm": np.quantile(boot, [0.025, 0.975]).tolist(),
            "helped": int((diff < 0).sum()), "equal": int((diff == 0).sum()),
            "harmed": int((diff > 0).sum())}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets/Dataset101_MSD")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evaluation_output/concavity_shoulder_fold0_20260923")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    split_path = args.dataset_root / "splits_final.json"
    split = json.loads(split_path.read_text())[0]
    train, val = sorted(split["train"]), sorted(split["val"])
    assert not set(train) & set(val)
    assert len(train) == len(set(train)) and len(val) == len(set(val))
    shapes, targets, labels, target_info, manifest = {}, {}, {}, {}, []
    for n in train + val:
        path = args.dataset_root / "labelsTr" / f"{n}.nii.gz"
        nii = nib.load(path)
        if not np.allclose(nii.affine[:3, :3], np.eye(3)):
            raise ValueError(f"{n}: expected axis-aligned 1 mm RAS")
        label = np.asarray(nii.dataobj, dtype=np.uint8)
        cut, cost, gap = best_fit_first_anterior_slice(label)
        shapes[n], labels[n], targets[n] = extract_shape(label > 0), label, cut
        target_info[n] = {"nonplanar_voxels": cost, "second_best_gap": gap}
        manifest.append({"path": str(path), "sha256": sha(path)})
    rows, fold_models = [], []

    def append_row(n, subset, support, predictions, shape, **extra):
        idx = int(np.where(shape.candidates == raw_cut(shape))[0][0])
        rows.append({"case": n, "split": subset, "support": support, "target": targets[n],
                     **target_info[n], **predictions, "low": shape.low, "high": shape.high,
                     "raw_strength": float(shape.consensus[idx]),
                     "persistence": float(shape.persistence[idx]),
                     "eligible_slices": int(shape.eligible_slices[idx]),
                     "removed_voxels": shape.removed_voxels,
                     "target_in_detector_range": int(shape.candidates[0] <= targets[n] <= shape.candidates[-1]),
                     **extra})

    for fold, (fit_ix, test_ix) in enumerate(KFold(4, shuffle=True, random_state=SEED).split(train)):
        fitting, testing = [train[i] for i in fit_ix], [train[i] for i in test_ix]
        params = fit_parameters(fitting, shapes, targets)
        alpha, scores = select_alpha(fitting, shapes, targets, 3)
        fold_models.append({"outer_fold": fold, "fit_ids": fitting, "test_ids": testing,
                            "parameters": params, "alpha": alpha, "inner_mae": scores})
        for n in testing:
            append_row(n, "train_oof", "reference", predict(shapes[n], params, alpha), shapes[n], outer_fold=fold)
    params = fit_parameters(train, shapes, targets)
    alpha, cv_scores = select_alpha(train, shapes, targets, 4)
    frozen = {"parameters": params, "alpha": alpha, "training_cv_mae": cv_scores,
              "outer_training_models": fold_models, "train_ids": train, "val_ids": val}
    (out / "frozen_detector.json").write_text(json.dumps(frozen, indent=2) + "\n")
    print("Frozen training detector:", json.dumps({"parameters": params, "alpha": alpha}), flush=True)
    all_shapes = {("reference", n): shapes[n] for n in train + val}
    for n in val:
        append_row(n, "validation", "reference", predict(shapes[n], params, alpha), shapes[n])

    roots = {
        "viewer_baseline": ROOT / "experiments/augmentation_family_b_20260907/voxel_audit/baseline_seed0/error_maps",
        "early_baseline": ROOT / "experiments/uncal_fold_early_stopping_20260921/voxel_audit/baseline_seed0/error_maps",
        "early_augmented": ROOT / "experiments/uncal_fold_early_stopping_20260921/voxel_audit/augmentation_seed0/error_maps",
    }
    inventory = {}
    for support, folder in roots.items():
        available = {n: folder / f"{n}.npz" for n in train + val if (folder / f"{n}.npz").exists()}
        inventory[support] = {"train_available": len(set(available) & set(train)), "val_available": len(set(available) & set(val)), "path": str(folder)}
        for n in val:
            if n not in available:
                continue
            path = available[n]
            with np.load(path, allow_pickle=False) as cache:
                truth, pred = cache["ground_truth"], cache["prediction"]
            padding = [(int(s - k) // 2, int(s - k) - int(s - k) // 2) for s, k in zip(truth.shape, labels[n].shape)]
            if pred.shape != truth.shape or not np.array_equal(np.pad(labels[n], padding), truth):
                raise ValueError(f"{support}/{n}: cached labels fail full-volume alignment")
            shape = extract_shape(pred > 0, origin_y=-padding[1][0])
            native_cut, native_cost, native_gap = best_fit_first_anterior_slice(pred)
            baseline_cut = native_cut - padding[1][0]
            append_row(n, "validation", support, predict(shape, params, alpha), shape,
                       native_model=baseline_cut, model_cut_cost=native_cost, model_cut_gap=native_gap,
                       union_dice=float(2 * ((truth > 0) & (pred > 0)).sum() / ((truth > 0).sum() + (pred > 0).sum())))
            all_shapes[(support, n)] = shape
            manifest.append({"path": str(path), "sha256": sha(path)})
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with (out / "case_results.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["split"], r["support"], r["case"])))

    groups = {}
    radii = {}
    oof = [r for r in rows if r["split"] == "train_oof"]
    for method in ("shoulder_offset", "hybrid", "position"):
        errors = [abs(r[method] - r["target"]) for r in oof]
        radii[method] = float(np.quantile(errors, 0.9, method="higher"))
    for subset, support in sorted({(r["split"], r["support"]) for r in rows}):
        group = [r for r in rows if (r["split"], r["support"]) == (subset, support)]
        target = [r["target"] for r in group]
        methods = METHODS + (("native_model",) if "native_model" in group[0] else ())
        groups[f"{subset}/{support}"] = {
            "metrics": {m: error_metrics([r[m] for r in group], target) for m in methods},
            "paired_vs_position": {m: paired([r["position"] for r in group], [r[m] for r in group], target)
                                   for m in ("shoulder_offset", "hybrid")},
            "paired_vs_model": {m: paired([r["native_model"] for r in group], [r[m] for r in group], target)
                                for m in ("shoulder_offset", "hybrid")} if "native_model" in group[0] else {},
            "nonplanar_cases": sum(r["nonplanar_voxels"] > 0 for r in group),
            "reference_ties": sum(r["second_best_gap"] == 0 for r in group),
            "zero_shoulder_cases": sum(r["raw_strength"] <= 1e-8 for r in group),
            "target_outside_detector_range": sum(not r["target_in_detector_range"] for r in group),
            "interval_coverage": {m: float(np.mean([abs(r[m] - r["target"]) <= radii[m] for r in group])) for m in radii},
        }
    transfer = {}
    val_ref = {r["case"]: r for r in rows if r["split"] == "validation" and r["support"] == "reference"}
    for support in roots:
        group = [r for r in rows if r["support"] == support]
        if group:
            transfer[support] = {m: {"mean_absolute_shift_mm": float(np.mean([abs(r[m] - val_ref[r["case"]][m]) for r in group])),
                                      "within_1_fraction": float(np.mean([abs(r[m] - val_ref[r["case"]][m]) <= 1 for r in group]))}
                                 for m in ("shoulder_raw", "hybrid")}
    summary = {"schema": "concavity_shoulder.v1", "fold": 0, "train_n": len(train), "val_n": len(val),
               "splits_sha256": sha(split_path), "source_sha256": sha(Path(__file__)),
               "groups": groups, "interval_radius_mm": radii, "support_transfer": transfer,
               "prediction_inventory": inventory, "frozen_model": frozen,
               "warnings": ["Repeatedly examined development validation fold; not confirmatory.",
                            "Participant pairing unknown; all counts and bootstraps are crop-level.",
                            "Reference union is oracle support, not an independently detected anatomical landmark.",
                            "No segmentation model retrained; interval coverage is descriptive."]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Candidate scores permit independently checking peaks and reference-cut rank.
    score_rows = []
    for (support, n), s in all_shapes.items():
        for i, c in enumerate(s.candidates):
            score_rows.append({"case": n, "support": support, "candidate": int(c),
                               "silhouette": s.silhouette[i], "sagittal": s.sagittal[i],
                               "consensus": s.consensus[i], "persistence": s.persistence[i],
                               "eligible_slices": s.eligible_slices[i]})
    with (out / "candidate_scores.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(score_rows[0]))
        writer.writeheader(); writer.writerows(score_rows)
    from .report_concavity_shoulder import write_report
    write_report(out, summary, rows, all_shapes, labels, args.dataset_root)
    print(json.dumps({k: v["metrics"] for k, v in groups.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
