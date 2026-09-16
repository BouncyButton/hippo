#!/usr/bin/env python3
"""Build an offline 3D MRI/mask/error viewer for a saved validation fold.

Run from the repository root: python utils/view_fold0_3d.py
Requires numpy, nibabel, scipy, and a browser with WebGL. No model inference,
web server, JavaScript downloads, or network connection is required.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
import webbrowser
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "datasets/Dataset101_MSD"
DEFAULT_PREDICTIONS = REPO_ROOT / (
    "experiments/augmentation_family_b_20260907/voxel_audit/"
    "baseline_seed0/error_maps"
)


def labels(array: np.ndarray, name: str) -> np.ndarray:
    """Reject invalid class IDs before converting a mask to uint8."""
    if array.ndim != 3 or not np.isin(array, [0, 1, 2]).all():
        raise ValueError(f"{name}: expected a 3D mask with class IDs 0, 1, 2")
    return array.astype(np.uint8)


def align_audit(image: np.ndarray, gt: np.ndarray, audit_gt: np.ndarray,
                prediction: np.ndarray) -> tuple:
    """Undo verified center padding, retaining any predictions outside the MRI.

    Returned origin maps viewer indices to original NIfTI voxel indices. Bounds
    delimit acquired MRI within the result; outside them there is no MRI signal.
    """
    audit_gt = labels(audit_gt, "audit ground_truth")
    prediction = labels(prediction, "audit prediction")
    if prediction.shape != audit_gt.shape:
        raise ValueError("Audit prediction and ground_truth shapes differ")
    delta = np.array(audit_gt.shape) - gt.shape
    if (delta < 0).any():
        raise ValueError("Audit grid is smaller than the native MRI; resized masks are unsupported")
    before = delta // 2
    expected = np.pad(gt, tuple(zip(before, delta-before)))
    if not np.array_equal(expected, audit_gt):
        raise ValueError("Audit ground_truth does not exactly match the native annotation after center padding")
    low, high = before.copy(), before + gt.shape
    occupied = np.argwhere(prediction > 0)
    if occupied.size:
        low = np.minimum(low, occupied.min(axis=0))
        high = np.maximum(high, occupied.max(axis=0)+1)
    crop = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
    native = tuple(slice(int(a), int(a+n)) for a, n in zip(before, image.shape))
    padded_image = np.zeros(audit_gt.shape, dtype=image.dtype)
    padded_image[native] = image
    origin = low-before
    bounds = [(-origin).tolist(), (np.array(image.shape)-origin).tolist()]
    return padded_image[crop], audit_gt[crop], prediction[crop], origin.tolist(), bounds


def rle(array: np.ndarray) -> list[int]:
    """Lossless value/count pairs in C order, decoded locally by the browser."""
    flat = array.ravel(order="C")
    starts = np.r_[0, np.flatnonzero(flat[1:] != flat[:-1])+1]
    lengths = np.diff(np.r_[starts, flat.size])
    return np.column_stack([flat[starts], lengths]).ravel().astype(int).tolist()


def mask_stats(gt: np.ndarray, pred: np.ndarray | None) -> dict:
    result = {"foreground": int((gt > 0).sum())}
    if pred is None:
        return result
    result.update(
        total=int((gt != pred).sum()),
        fp=int(((gt == 0) & (pred > 0)).sum()),
        fn=int(((gt > 0) & (pred == 0)).sum()),
        swaps=int(((gt > 0) & (pred > 0) & (gt != pred)).sum()),
        a_to_p=int(((gt == 1) & (pred == 2)).sum()),
        p_to_a=int(((gt == 2) & (pred == 1)).sum()),
    )
    for label_id, key in [(1, "dice_a"), (2, "dice_p"), (None, "dice_fg")]:
        a, b = (gt > 0, pred > 0) if label_id is None else (gt == label_id, pred == label_id)
        denominator = int(a.sum()+b.sum())
        result[key] = 2*int((a & b).sum())/denominator if denominator else None
    return result


def label_cut(gt: np.ndarray) -> float | None:
    """Return a cut only for an exactly planar, ordered A/P partition."""
    a = np.flatnonzero((gt == 1).any(axis=(0, 2)))
    p = np.flatnonzero((gt == 2).any(axis=(0, 2)))
    if a.size and p.size and p.max()+1 == a.min():
        return float(p.max()+0.5)
    return None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_case(dataset: Path, case: str, prediction_dir: Path | None) -> dict:
    image_path = dataset / "imagesTr" / f"{case}_0000.nii.gz"
    label_path = dataset / "labelsTr" / f"{case}.nii.gz"
    image_nii, gt_nii = nib.load(image_path), nib.load(label_path)
    image = np.asarray(image_nii.dataobj)
    gt = labels(np.asarray(gt_nii.dataobj), str(label_path))
    if image.shape != gt.shape or not np.allclose(image_nii.affine, gt_nii.affine):
        raise ValueError("MRI and annotation grids/affines differ")
    if image.ndim != 3 or not np.isfinite(image).all():
        raise ValueError("MRI must be a finite 3D array")
    matrix = image_nii.affine[:3, :3]
    if nib.aff2axcodes(image_nii.affine) != ("R", "A", "S") or not np.allclose(matrix, np.diag(np.diag(matrix)), atol=1e-5):
        raise ValueError("Expected an axis-aligned RAS grid, as in Dataset101_MSD; reorient/resample all inputs consistently first")
    native_shape = list(image.shape)
    # Window only the acquired MRI, never the synthetic padding.
    low, high = np.percentile(image, [1, 99])
    if high <= low:
        low, high = float(image.min()), float(image.max())
    image = np.rint(np.clip((image-low)/max(float(high-low), 1e-12), 0, 1)*255).astype(np.uint8)
    origin, bounds, pred = [0, 0, 0], [[0, 0, 0], native_shape], None
    sources = [image_path, label_path]
    alignment = "native annotation and MRI have matching shapes and affines"
    if prediction_dir is not None:
        candidates = [prediction_dir / f"{case}{ext}" for ext in [".npz", ".nii.gz", ".nii"]]
        candidates = [p for p in candidates if p.is_file()]
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one prediction {case}.npz/.nii.gz/.nii in {prediction_dir}; found {len(candidates)}")
        path = candidates[0]
        sources.append(path)
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                if not {"ground_truth", "prediction"}.issubset(archive.files):
                    raise ValueError("Audit NPZ must contain ground_truth and prediction for alignment verification")
                image, gt, pred, origin, bounds = align_audit(image, gt, archive["ground_truth"], archive["prediction"])
            alignment = "exact equality of full audit ground_truth and center-padded native annotation"
        else:
            prediction_nii = nib.load(path)
            pred = labels(np.asarray(prediction_nii.dataobj), str(path))
            if pred.shape != gt.shape or not np.allclose(prediction_nii.affine, gt_nii.affine):
                raise ValueError("Prediction NIfTI must match the MRI shape and affine exactly")
            alignment = "prediction NIfTI and MRI have matching shapes and affines"
    per_slice = []
    for y in range(gt.shape[1]):
        g = gt[:, y, :]
        stats = mask_stats(g, None if pred is None else pred[:, y, :])
        stats.update(anterior=int((g == 1).sum()), posterior=int((g == 2).sum()),
                     components=int(ndimage.label(g > 0)[1]))
        per_slice.append(stats)
    cut = label_cut(gt)
    foreground = (gt > 0) if pred is None else (gt > 0) | (pred > 0)
    occupied = np.argwhere(foreground)
    center = (occupied.min(0)+occupied.max(0))/2 if occupied.size else (np.array(gt.shape)-1)/2
    provenance = dict(native_shape=native_shape, native_affine=image_nii.affine.tolist(),
                      alignment=alignment, viewer_origin_in_native_voxels=origin,
                      sources=[dict(path=str(p.resolve()), sha256=sha256(p)) for p in sources])
    return dict(name=case, shape=list(gt.shape), spacing=list(map(float, gt_nii.header.get_zooms())),
                origin=origin, imageBounds=bounds, center=center.tolist(),
                image=base64.b64encode(image.tobytes(order="C")).decode("ascii"),
                gt=rle(gt), pred=None if pred is None else rle(pred), slices=per_slice,
                stats=mask_stats(gt, pred), cut=cut,
                start=int(np.ceil(cut)) if cut is not None else int(round(center[1])),
                window=[float(low), float(high)], provenance=provenance)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--splits-json", type=Path, help="Default: DATASET_DIR/splits_final.json")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--prediction-dir", type=Path, default=DEFAULT_PREDICTIONS,
                        help="Saved audit NPZs or native-grid NIfTI predictions (default: baseline_seed0/error_maps)")
    parser.add_argument("--prediction-name", help="Human-readable model/run label")
    parser.add_argument("--ground-truth-only", action="store_true", help="Build without predictions or error views")
    parser.add_argument("--initial-case", help="Initially selected validation case, e.g. hippocampus_164 or 164")
    parser.add_argument("--output", type=Path, help="Default: evaluation_output/foldN_3d/index.html")
    parser.add_argument("--no-open", action="store_true", help="Generate files without opening a browser")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = args.dataset_dir.resolve()
    splits_path = args.splits_json or dataset / "splits_final.json"
    output = (args.output or REPO_ROOT / f"evaluation_output/fold{args.fold}_3d/index.html").resolve()
    try:
        splits = json.loads(splits_path.read_text())
        if args.fold < 0 or args.fold >= len(splits):
            raise ValueError(f"Fold {args.fold} is unavailable in {splits_path}")
        names = splits[args.fold]["val"]
        if not names or len(names) != len(set(names)) or not all(isinstance(n, str) and re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in names):
            raise ValueError("Validation split must be a nonempty list of unique, safe case identifiers")
        pred_dir = None if args.ground_truth_only else args.prediction_dir.resolve()
        if pred_dir is not None and not pred_dir.is_dir():
            raise ValueError(f"Prediction directory not found: {pred_dir}. Use --prediction-dir PATH or --ground-truth-only")
        cases = []
        for i, name in enumerate(names, 1):
            try:
                cases.append(load_case(dataset, name, pred_dir))
            except (ValueError, OSError, KeyError) as error:
                raise ValueError(f"{name}: {error}") from error
            print(f"[{i:02d}/{len(names):02d}] {name}", flush=True)
        initial = args.initial_case
        if initial and initial.isdigit():
            initial = f"hippocampus_{int(initial):03d}"
        if initial is not None and initial not in names:
            raise ValueError(f"Initial case {initial} is not in fold {args.fold} validation")
        initial = initial or max(cases, key=lambda c: c["stats"].get("swaps", 0))["name"]
        prediction_name = args.prediction_name or (pred_dir.parent.name if pred_dir and pred_dir.name == "error_maps" else pred_dir.name if pred_dir else "No predictions")
        payload = dict(fold=args.fold, predictionName=prediction_name, initialCase=initial, cases=cases)
        template_path = Path(__file__).with_name("fold_3d_viewer.html")
        template = template_path.read_text()
        if template.count("__VIEWER_DATA__") != 1:
            raise ValueError("Viewer template must contain one data placeholder")
        script = template_path.with_suffix(".js").read_text()
        if template.count("__VIEWER_SCRIPT__") != 1:
            raise ValueError("Viewer template must contain one script placeholder")
        template = template.replace("__VIEWER_SCRIPT__", script)
        encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(template.replace("__VIEWER_DATA__", encoded), encoding="utf-8")
        manifest = dict(fold=args.fold, split="val", splits_json=str(splits_path.resolve()),
                        splits_sha256=sha256(splits_path), prediction_name=prediction_name,
                        cases=[dict(name=c["name"], stats=c["stats"], **c["provenance"]) for c in cases])
        output.with_suffix(".provenance.json").write_text(json.dumps(manifest, indent=2)+"\n")
        print(f"\nReady: {output}\n{len(cases)} validation cases; {output.stat().st_size/1e6:.1f} MB; works offline.")
        if not args.no_open:
            webbrowser.open(output.as_uri())
    except (ValueError, OSError, KeyError, IndexError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
