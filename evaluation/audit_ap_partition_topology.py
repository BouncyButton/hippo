"""Audit whether A/P partition defects justify another structural training loss.

Uses existing, paired NPZ hard predictions; never runs or changes a model.
Connectivity is explicitly digital. The interface statistic counts components
of a contact *band*, not the topology of a reconstructed surface. Coronal rays
are a diagnostic for this dataset's annotation axis, not hippocampal unfolding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import nibabel as nib
import numpy as np
import scipy
from scipy import ndimage as ndi


ROOT = Path(__file__).resolve().parents[1]


def validate_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 3 or min(labels.shape) == 0:
        raise ValueError("Expected a nonempty 3D label array.")
    if not np.isin(labels, (0, 1, 2)).all():
        raise ValueError("Expected background=0 and A/P=1/2 only.")
    return labels.astype(np.uint8, copy=False)


def components(mask: np.ndarray, connectivity: int) -> tuple[int, int]:
    """Return number of components and voxels outside the largest one."""
    ids, count = ndi.label(mask, ndi.generate_binary_structure(3, connectivity))
    sizes = np.bincount(ids.ravel())[1:]
    return int(count), int(sizes.sum() - sizes.max()) if count else 0


def partition_stats(labels: np.ndarray) -> dict:
    labels = validate_labels(labels)
    masks = {"a": labels == 1, "p": labels == 2, "union": labels > 0}
    faces = ndi.generate_binary_structure(3, 1)
    masks["contact_band"] = (
        masks["a"] & ndi.binary_dilation(masks["p"], structure=faces)
    ) | (masks["p"] & ndi.binary_dilation(masks["a"], structure=faces))
    result = {}
    for name, mask in masks.items():
        for connectivity, suffix in ((1, 6), (3, 26)):
            count, island_voxels = components(mask, connectivity)
            result[f"{name}_components_{suffix}"] = count
            result[f"{name}_island_voxels_{suffix}"] = island_voxels
    result["passes_connectivity_26"] = all(
        result[f"{name}_components_26"] == 1 for name in masks
    )
    return result


def single_switch(sequence: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Nearest A*P* or P*A* sequence; includes pure sequences and reports ties.

    Orientation is selected from the input alone. Ties use lower left-label,
    then lower cut index. Returned tie count counts distinct output sequences.
    """
    sequence = np.asarray(sequence)
    if sequence.ndim != 1 or not np.isin(sequence, (1, 2)).all():
        raise ValueError("Expected a foreground-only 1D sequence.")
    n = len(sequence)
    if not n:
        return sequence.copy(), 0, 1
    counts_a = np.r_[0, np.cumsum(sequence == 1)]
    counts_p = np.arange(n + 1) - counts_a
    costs = np.stack((
        counts_p + counts_a[-1] - counts_a,
        counts_a + counts_p[-1] - counts_p,
    ))
    orientation, cut = np.unravel_index(costs.argmin(), costs.shape)
    left = orientation + 1
    out = np.where(np.arange(n) < cut, left, 3 - left).astype(np.uint8)
    minimum = int(costs.min())
    # Pure outputs appear twice among the 2*(n+1) candidates.
    ties = int((costs == minimum).sum())
    ties -= int(costs[0, 0] == minimum) + int(costs[0, -1] == minimum)
    return out, minimum, ties


def ray_audit(labels: np.ndarray, axis: int = 1) -> tuple[dict, np.ndarray]:
    """Count/repair repeated transitions within contiguous foreground runs.

    Background gaps split runs; no adjacency is invented across a gap. Each
    run may choose its own orientation, so this is a deliberately weak prior.
    """
    labels = validate_labels(labels)
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    moved = np.moveaxis(labels, axis, -1)
    rays = moved.reshape(-1, moved.shape[-1])
    repaired = rays.copy()
    stats = {key: 0 for key in (
        "foreground_runs", "repeated_transition_runs", "excess_transitions",
        "repair_voxels", "ambiguous_repair_runs",
    )}
    for index, ray in enumerate(rays):
        foreground = ray > 0
        if not foreground.any():
            continue
        edges = np.diff(np.r_[False, foreground, False].astype(np.int8))
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            sequence = ray[start:end]
            stats["foreground_runs"] += 1
            transitions = int(np.count_nonzero(np.diff(sequence)))
            if transitions <= 1:
                continue
            fitted, cost, ties = single_switch(sequence)
            repaired[index, start:end] = fitted
            stats["repeated_transition_runs"] += 1
            stats["excess_transitions"] += transitions - 1
            stats["repair_voxels"] += cost
            stats["ambiguous_repair_runs"] += int(ties > 1)
    return stats, np.moveaxis(repaired.reshape(moved.shape), -1, axis)


def plane_projection(labels: np.ndarray, support: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Fit a coronal plane to labels, optionally render on a different support.

    Both orientations are considered; cuts must leave input foreground on both
    sides. With support=None this uses prediction only. Fitting ground truth
    and rendering on predicted support is an explicitly oracle diagnostic.
    """
    labels = validate_labels(labels)
    if support is None:
        support = labels > 0
    support = np.asarray(support, dtype=bool)
    if support.shape != labels.shape:
        raise ValueError("Support and label shapes must match.")
    occupied = np.flatnonzero((labels > 0).any(axis=(0, 2)))
    if len(occupied) < 2 or not (labels == 1).any() or not (labels == 2).any():
        return labels.copy(), {"valid": False, "cut": None, "disagreement": None, "ties": 0}
    a = (labels == 1).sum(axis=(0, 2))
    p = (labels == 2).sum(axis=(0, 2))
    ap, pp = np.r_[0, a.cumsum()], np.r_[0, p.cumsum()]
    cuts = np.arange(occupied[0] + 1, occupied[-1] + 1)
    costs = np.stack((pp[cuts] + ap[-1] - ap[cuts], ap[cuts] + pp[-1] - pp[cuts]))
    orientation, index = np.unravel_index(costs.argmin(), costs.shape)
    cut = int(cuts[index])
    left = int(orientation + 1)
    assigned = np.where(np.arange(labels.shape[1])[None, :, None] < cut, left, 3 - left)
    out = np.where(support, assigned, 0).astype(np.uint8)
    return out, {"valid": True, "cut": cut, "left_label": left,
                 "disagreement": int(costs.min()), "ties": int((costs == costs.min()).sum())}


def score(prediction: np.ndarray, truth: np.ndarray) -> dict:
    dice = []
    for cls in (1, 2):
        pred, gt = prediction == cls, truth == cls
        denominator = pred.sum() + gt.sum()
        dice.append(float(2 * (pred & gt).sum() / denominator) if denominator else 1.0)
    swaps = (prediction > 0) & (truth > 0) & (prediction != truth)
    return {"macro_dice": float(np.mean(dice)), "ap_swaps": int(swaps.sum())}


def compare_repair(prediction: np.ndarray, repaired: np.ndarray, truth: np.ndarray) -> dict:
    if not np.array_equal(prediction > 0, repaired > 0):
        raise ValueError("Partition repairs must preserve foreground exactly.")
    result = score(repaired, truth)
    result.update(
        changed=int((prediction != repaired).sum()),
        corrected=int(((prediction != truth) & (repaired == truth)).sum()),
        introduced=int(((prediction == truth) & (repaired != truth)).sum()),
    )
    return result


def describe_case(labels: np.ndarray) -> dict:
    rays, _ = ray_audit(labels)
    _, plane = plane_projection(labels)
    return {"connectivity": partition_stats(labels), "rays": rays, "plane": plane}


def summarize_cases(cases: list[dict]) -> dict:
    result = {"cases": len(cases)}
    for suffix in (6, 26):
        for name in ("a", "p", "union", "contact_band"):
            key = f"{name}_components_{suffix}"
            result[f"{key}_not_one"] = sum(c["connectivity"][key] != 1 for c in cases)
    result["passes_connectivity_26"] = sum(c["connectivity"]["passes_connectivity_26"] for c in cases)
    result["cases_with_repeated_transitions"] = sum(c["rays"]["repeated_transition_runs"] > 0 for c in cases)
    for key in ("foreground_runs", "repeated_transition_runs", "excess_transitions", "repair_voxels", "ambiguous_repair_runs"):
        result[key] = sum(c["rays"][key] for c in cases)
    return result


def summarize_predictions(cases: list[dict]) -> dict:
    result = summarize_cases([c["prediction"] for c in cases])
    result["ground_truth"] = summarize_cases([c["truth"] for c in cases])
    result["baseline_macro_dice"] = float(np.mean([c["baseline"]["macro_dice"] for c in cases]))
    result["baseline_ap_swaps"] = sum(c["baseline"]["ap_swaps"] for c in cases)
    clean = [c for c in cases if c["prediction"]["connectivity"]["passes_connectivity_26"]]
    result["swaps_in_connectivity_passing_cases"] = sum(c["baseline"]["ap_swaps"] for c in clean)
    ordered = [c for c in cases if c["prediction"]["rays"]["repeated_transition_runs"] == 0]
    result["swaps_in_cases_without_repeated_transitions"] = sum(c["baseline"]["ap_swaps"] for c in ordered)
    result["repairs"] = {}
    for name in ("ray", "plane", "oracle_plane"):
        deltas = np.array([c[name]["macro_dice"] - c["baseline"]["macro_dice"] for c in cases])
        result["repairs"][name] = {
            "macro_dice": float(np.mean([c[name]["macro_dice"] for c in cases])),
            "dice_delta": float(deltas.mean()),
            "better": int((deltas > 1e-12).sum()),
            "worse": int((deltas < -1e-12).sum()),
            **{key: sum(c[name][key] for c in cases) for key in ("ap_swaps", "changed", "corrected", "introduced")},
        }
    return result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets/Dataset101_MSD")
    parser.add_argument("--audit-root", type=Path, default=ROOT / "experiments/uncal_fold_early_stopping_20260921/voxel_audit")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    split_path = args.dataset_root / "splits_final.json"
    split = json.loads(split_path.read_text())[args.fold]
    if set(split["train"]) & set(split["val"]):
        raise ValueError("Training and development sets overlap.")
    manifest = {"script_sha256": sha256(Path(__file__)), "split_sha256": sha256(split_path),
                "fold": args.fold, "dataset_root": str(args.dataset_root.resolve()),
                "audit_root": str(args.audit_root.resolve()), "inputs": {}, "cache_configs": {},
                "versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "scipy": scipy.__version__, "nibabel": nib.__version__}}
    native = {}
    train_cases = []
    for name in split["train"] + split["val"]:
        path = args.dataset_root / "labelsTr" / f"{name}.nii.gz"
        image = nib.load(path)
        if nib.aff2axcodes(image.affine) != ("R", "A", "S"):
            raise ValueError(f"Expected RAS orientation: {path}")
        native[name] = validate_labels(np.asarray(image.dataobj))
        manifest["inputs"][str(path)] = sha256(path)
        if name in split["train"]:
            train_cases.append({"case": name, **describe_case(native[name])})
    arms = {}
    for arm in ("baseline_seed0", "augmentation_seed0"):
        directory = args.audit_root / arm / "error_maps"
        if {p.stem for p in directory.glob("*.npz")} != set(split["val"]):
            raise ValueError(f"Cache does not match requested validation fold: {directory}")
        cases = []
        for name in split["val"]:
            path = directory / f"{name}.npz"
            with np.load(path, allow_pickle=False) as archive:
                truth = validate_labels(archive["ground_truth"])
                prediction = validate_labels(archive["prediction"])
            if prediction.shape != truth.shape or any(a > b for a, b in zip(native[name].shape, truth.shape)):
                raise ValueError(f"Invalid cache shape: {path}")
            pads = [((b-a)//2, (b-a+1)//2) for a, b in zip(native[name].shape, truth.shape)]
            if not np.array_equal(np.pad(native[name], pads), truth):
                raise ValueError(f"Cached ground truth differs from symmetrically padded native label: {path}")
            manifest["inputs"][str(path)] = sha256(path)
            ray_stats, ray = ray_audit(prediction)
            plane, plane_fit = plane_projection(prediction)
            oracle, oracle_fit = plane_projection(truth, prediction > 0)
            if not plane_fit["valid"] or not oracle_fit["valid"]:
                raise ValueError(f"Plane diagnostic requires both classes and >=2 slices: {path}")
            cases.append({
                "case": name, "truth": describe_case(truth),
                "prediction": {"connectivity": partition_stats(prediction), "rays": ray_stats, "plane": plane_fit},
                "baseline": score(prediction, truth),
                "ray": compare_repair(prediction, ray, truth),
                "plane": compare_repair(prediction, plane, truth),
                "oracle_plane": compare_repair(prediction, oracle, truth),
                "cut_abs_error": abs(plane_fit["cut"] - oracle_fit["cut"]),
            })
        arms[arm] = cases
        source_summary_path = directory.parent / "summary.json"
        if source_summary_path.exists():
            source_summary = json.loads(source_summary_path.read_text())
            observed = summarize_predictions(cases)
            if not np.isclose(observed["baseline_macro_dice"], source_summary["mean_dice"], atol=1e-12, rtol=0):
                raise ValueError(f"Cached Dice does not reproduce source summary: {arm}")
            if observed["baseline_ap_swaps"] != source_summary["grouped_error_counts"]["anterior_posterior_swap"]:
                raise ValueError(f"Cached swap counts do not reproduce source summary: {arm}")
            manifest["inputs"][str(source_summary_path)] = sha256(source_summary_path)
            manifest["cache_configs"][arm] = source_summary["config"]
        print(f"Audited {len(cases)} paired cases for {arm}.", flush=True)
    summary = {"interpretation": f"Exploratory existing fold-{args.fold} development caches; no training, new inference, or hyperparameter selection. Oracle uses GT and is not deployable. Hard projections are mechanism checks, not training-loss experiments.",
               "train": summarize_cases(train_cases),
               "arms": {arm: summarize_predictions(cases) for arm, cases in arms.items()}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (("summary", summary), ("cases", {"train": train_cases, **arms}), ("manifest", manifest)):
        (args.output_dir / f"{name}.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
