#!/usr/bin/env python3
"""Screen an adjacent-slice union constraint using frozen nnU-Net hard masks.

This is a diagnostic, not a soft-logit repair or training experiment. Fit the
overlap threshold on training labels and audit predictions from that split.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def adjacent_dice(mask: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    slices = np.moveaxis(mask, axis, 0)
    area = slices.sum(axis=(1, 2), dtype=np.int64)
    intersection = np.logical_and(slices[:-1], slices[1:]).sum(axis=(1, 2), dtype=np.int64)
    denominator = area[:-1] + area[1:]
    dice = np.divide(
        2 * intersection,
        denominator,
        out=np.zeros(len(denominator), dtype=np.float64),
        where=denominator > 0,
    )
    return dice, area


def read_mask(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = nib.load(str(path))
    values = np.asarray(image.dataobj)
    if not np.all(np.isin(values, (0, 1, 2))):
        raise ValueError(f"Unexpected labels: {path}")
    return values.astype(np.uint8, copy=False), image.affine


def physical_si_axis(affine: np.ndarray) -> int:
    codes = nib.aff2axcodes(affine)
    axes = [axis for axis, code in enumerate(codes) if code in ("S", "I")]
    if len(axes) != 1:
        raise ValueError(f"Cannot identify one superior-inferior axis: {codes}")
    return axes[0]


def pair_error(gt: np.ndarray, pred: np.ndarray, axis: int) -> np.ndarray:
    gt_slices = np.moveaxis(gt, axis, 0)
    pred_slices = np.moveaxis(pred, axis, 0)
    gt_area = gt_slices.sum(axis=(1, 2), dtype=np.int64)
    voxel_error = np.logical_xor(gt_slices, pred_slices).sum(axis=(1, 2), dtype=np.int64)
    return (voxel_error[:-1] + voxel_error[1:]) / np.maximum(
        gt_area[:-1] + gt_area[1:], 1
    )


def soft_adjacent_overlap(probability: np.ndarray, axis: int) -> np.ndarray:
    slices = np.moveaxis(probability, axis, 0)
    area = slices.sum(axis=(1, 2), dtype=np.float64)
    overlap = np.minimum(slices[:-1], slices[1:]).sum(axis=(1, 2), dtype=np.float64)
    return 2 * overlap / np.maximum(area[:-1] + area[1:], 1e-8)


def read_union_probability(path: Path, hard_labels: np.ndarray) -> np.ndarray:
    with np.load(path) as archive:
        probabilities = np.asarray(archive["probabilities"], dtype=np.float32)
    if probabilities.ndim != 4 or probabilities.shape[0] != 3:
        raise ValueError(f"Expected 3-class probabilities, got {probabilities.shape}: {path}")
    for permutation in itertools.permutations(range(1, 4)):
        candidate = probabilities.transpose((0, *permutation))
        if candidate.shape[1:] != hard_labels.shape:
            continue
        if np.array_equal(candidate.argmax(axis=0), hard_labels):
            return candidate[1] + candidate[2]
    raise ValueError(f"No probability orientation matches hard predictions: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--prob-dir", type=Path, help="nnU-Net exported probability NPZ files")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--min-area-fraction", type=float, default=0.1)
    parser.add_argument("--gt-quantile", type=float, default=0.01)
    args = parser.parse_args()

    if not 0 < args.min_area_fraction < 1 or not 0 < args.gt_quantile < 1:
        parser.error("Area fraction and GT quantile must lie strictly between 0 and 1")
    case_ids = json.loads(args.splits.read_text())[args.fold][args.split]
    records = []
    gt_values = []
    for case_id in case_ids:
        gt_labels, gt_affine = read_mask(args.gt_dir / f"{case_id}.nii.gz")
        pred_labels, pred_affine = read_mask(args.pred_dir / f"{case_id}.nii.gz")
        gt = gt_labels > 0
        pred = pred_labels > 0
        if gt.shape != pred.shape or not np.allclose(gt_affine, pred_affine, atol=1e-3):
            raise ValueError(f"Prediction/GT geometry mismatch for {case_id}")
        axis = physical_si_axis(gt_affine)
        gt_pair, gt_area = adjacent_dice(gt, axis)
        if args.prob_dir is None:
            pred_pair, _ = adjacent_dice(pred, axis)
        else:
            probability = read_union_probability(args.prob_dir / f"{case_id}.npz", pred_labels)
            pred_pair = soft_adjacent_overlap(probability, axis)
        eligible = (gt_area[:-1] >= args.min_area_fraction * gt_area.max()) & (
            gt_area[1:] >= args.min_area_fraction * gt_area.max()
        )
        if not eligible.any():
            raise ValueError(f"No eligible adjacent pairs for {case_id}")
        local_error = pair_error(gt, pred, axis)
        union_dice = 2 * np.logical_and(gt, pred).sum() / (gt.sum() + pred.sum())
        record = {
            "case_id": case_id,
            "union_dice": float(union_dice),
            "gt_pair_dice": gt_pair[eligible],
            "pred_pair_dice": pred_pair[eligible],
            "pair_error": local_error[eligible],
        }
        gt_values.extend(record["gt_pair_dice"].tolist())
        records.append(record)

    threshold = float(np.quantile(gt_values, args.gt_quantile))
    all_pred = np.concatenate([r["pred_pair_dice"] for r in records])
    all_gt = np.concatenate([r["gt_pair_dice"] for r in records])
    all_error = np.concatenate([r["pair_error"] for r in records])
    violations = all_pred < threshold
    gt_violations = all_gt < threshold
    case_rows = []
    for record in records:
        pred = record["pred_pair_dice"]
        gt = record["gt_pair_dice"]
        local = record["pair_error"]
        bad = pred < threshold
        case_rows.append(
            {
                "case_id": record["case_id"],
                "union_dice": record["union_dice"],
                "eligible_pairs": int(len(pred)),
                "pred_violations": int(bad.sum()),
                "gt_violations": int((gt < threshold).sum()),
                "mean_pair_error": float(local.mean()),
                "mean_violation_error": float(local[bad].mean()) if bad.any() else None,
            }
        )
    rates = np.asarray([r["pred_violations"] / r["eligible_pairs"] for r in case_rows])
    errors = np.asarray([1 - r["union_dice"] for r in case_rows])
    correlation = float(np.corrcoef(rates, errors)[0, 1]) if rates.std() and errors.std() else None
    result = {
        "protocol": {
            "fold": args.fold,
            "split": args.split,
            "case_count": len(records),
            "axis": "physical superior-inferior",
            "min_gt_slice_area_fraction_of_case_max": args.min_area_fraction,
            "threshold_source": "pooled eligible GT adjacent-pair Dice quantile",
            "gt_quantile": args.gt_quantile,
            "threshold": threshold,
            "prediction_type": (
                "hard labels; no causal repair" if args.prob_dir is None
                else "exported soft probabilities; no causal repair"
            ),
        },
        "summary": {
            "eligible_pairs": int(len(all_pred)),
            "gt_violations": int(gt_violations.sum()),
            "pred_violations": int(violations.sum()),
            "pred_only_violations": int(np.count_nonzero(violations & ~gt_violations)),
            "cases_with_pred_violations": int(sum(r["pred_violations"] > 0 for r in case_rows)),
            "median_case_violation_rate": float(np.median(rates)),
            "mean_train_union_dice": float(np.mean([r["union_dice"] for r in case_rows])),
            "case_violation_rate_vs_union_error_pearson": correlation,
            "mean_pair_error_when_violated": float(all_error[violations].mean()) if violations.any() else None,
            "mean_pair_error_when_satisfied": float(all_error[~violations].mean()) if (~violations).any() else None,
            "gt_adjacent_dice_quantiles": np.quantile(all_gt, [0.01, 0.05, 0.5, 0.95]).tolist(),
            "pred_adjacent_dice_quantiles": np.quantile(all_pred, [0.01, 0.05, 0.5, 0.95]).tolist(),
        },
        "per_case": case_rows,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
