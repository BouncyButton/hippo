#!/usr/bin/env python3
"""Average matched teacher/head predictions for unseen patients."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .score_slice_qc import score_model
from .slice_qc import load_patient_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--features", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.models) != len(args.features) or len(args.models) < 2:
        raise ValueError("provide at least two matched model/feature pairs")
    names = None
    scores = []
    kinds = []
    for model_path, feature_path in zip(args.models, args.features, strict=True):
        data = load_patient_features(feature_path, require_target=False)
        if names is None:
            names = data.names
            slices = data.slices
        elif not np.array_equal(names, data.names) or slices != data.slices:
            raise ValueError("all teacher features must have identical patient and slice order")
        prediction, kind = score_model(model_path, data, args.device)
        scores.append(prediction)
        kinds.append(kind)
    if len(set(kinds)) != 1:
        raise ValueError("ensemble members must use the same feature kind")
    average = np.mean(scores, axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("case_name", "slice_index", "predicted_error", "ensemble_members", "feature_kind"))
        for patient, name in enumerate(names):
            for slice_index, score in enumerate(average[patient]):
                writer.writerow((name, slice_index, float(score), len(scores), kinds[0]))
    print(f"Scored {len(names)} patients with {len(scores)} matched heads to {args.output}")


if __name__ == "__main__":
    main()
