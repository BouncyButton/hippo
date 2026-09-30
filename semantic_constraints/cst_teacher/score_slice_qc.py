#!/usr/bin/env python3
"""Score patient slice-error risk from a trained CST quality head and feature NPZ."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from .slice_qc import load_patient_features, load_ridge, ridge_predict, temporal_predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def score_model(model_path: Path, data, device: str = "cpu"):
    """Score features without accessing targets; refuse in-sample patients."""
    if model_path.suffix == ".npz":
        kind, model, metadata = load_ridge(model_path)
        prediction = ridge_predict(model, data.features[kind])
    elif model_path.suffix == ".pt":
        artifact = torch.load(model_path, map_location="cpu", weights_only=True)
        kind = artifact["kind"]
        metadata = artifact
        prediction = temporal_predict(artifact, data.features[kind], device)
    else:
        raise ValueError("model must be a .npz ridge or .pt temporal artifact")
    training_names = set(metadata["training_patients"])
    overlap = training_names & set(data.names)
    if overlap:
        raise ValueError(f"refusing to score {len(overlap)} model-training patients")
    return prediction, kind


def main() -> None:
    args = parse_args()
    data = load_patient_features(args.features, require_target=False)
    prediction, kind = score_model(args.model, data, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("case_name", "slice_index", "predicted_error", "model", "feature_kind"))
        for patient, name in enumerate(data.names):
            for slice_index, score in enumerate(prediction[patient]):
                writer.writerow((name, slice_index, float(score), args.model.name, kind))
    print(f"Scored {data.count} patients and {data.count * data.slices} slices to {args.output}")


if __name__ == "__main__":
    main()
