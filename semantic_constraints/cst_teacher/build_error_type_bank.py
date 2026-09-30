#!/usr/bin/env python3
"""Build slice error-type targets from frozen Swin predictions and MSD labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import uniform_positions
from .error_types import TYPE_NAMES, audit_case
from .evaluate_predictions import prediction_map
from .slice_qc import load_patient_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", type=Path, required=True)
    parser.add_argument("--splits-json", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--inference-dir", type=Path, required=True)
    parser.add_argument("--reference-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from .train_teacher import build_loaders

    reference = load_patient_features(args.reference_features, require_target=True)
    build_args = argparse.Namespace(
        pkl=args.pkl,
        splits_json=args.splits_json,
        fold=args.fold,
        spatial_size=(64, 64, 64),
        set_size=reference.slices,
        slab_depth=3,
        inplane_size=32,
        cache_dataset=False,
        max_train_cases=0,
        max_val_cases=0,
        batch_size=16,
        num_workers=0,
    )
    _, loader = build_loaders(build_args)
    paths = prediction_map(args.inference_dir)
    actual_names = []
    arrays: dict[str, list[np.ndarray]] = {}
    for index in range(len(loader.dataset.base_dataset)):
        item = loader.dataset.base_dataset[index]
        name = str(item["case_name"])
        actual_names.append(name)
        labels = np.asarray(item["label"])[0].astype(np.int8)
        probabilities = np.load(paths[name], allow_pickle=False)
        if probabilities.shape != (3, *labels.shape):
            raise ValueError(f"Swin prediction shape mismatch for {name}")
        prediction = probabilities.argmax(axis=0).astype(np.int8)
        positions = uniform_positions(labels.shape[1], reference.slices).numpy()
        audited = audit_case(labels, prediction, positions)
        for key, value in audited.items():
            arrays.setdefault(key, []).append(value)
    if not np.array_equal(actual_names, reference.names):
        raise ValueError("validation cases differ in order from CST feature bank")
    stacked = {key: np.stack(blocks) for key, blocks in arrays.items()}
    if not np.array_equal(stacked["error_type"].shape, reference.target.shape):
        raise ValueError("error-type bank and slice-error targets are misaligned")
    distribution = dict(zip(TYPE_NAMES, np.bincount(stacked["error_type"].ravel(), minlength=len(TYPE_NAMES)).tolist()))
    summary = {
        "schema": "semantic_constraints.cst_teacher.error_type_bank.v1",
        "fold": args.fold,
        "patients": len(actual_names),
        "slices": int(stacked["error_type"].size),
        "distribution": distribution,
        "isolated_disappearance": int(stacked["isolated_disappearance"].sum()),
        "terminal_extension": int(stacked["terminal_extension"].sum()),
        "total_error_voxels": stacked["error_counts"].sum(axis=(0, 1)).tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, case_names=np.asarray(actual_names), **stacked)
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
