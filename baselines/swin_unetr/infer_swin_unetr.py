#!/usr/bin/env python3
"""Run local-checkpoint SwinUNETR inference on a selected CV split."""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from monai.data import Dataset as MonaiDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation.evaluate import evaluate_swinunetr  # noqa: E402
from swin_unetr import (  # noqa: E402
    _build_monai_dataset_from_pkl,
    _infer_num_classes,
    _load_pkl_dataframe,
    _load_splits_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to model.pt")
    parser.add_argument("--pkl", required=True, help="Path to gzipped dataframe dataset")
    parser.add_argument("--dataset", required=True, choices=["MSD", "MNI", "ADNI", "COBRA"])
    parser.add_argument("--splits-json", required=True, help="CV split JSON used for training")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--spatial-size", type=int, nargs=3, default=[64, 64, 64])
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("No CUDA device is visible for inference.")

    df = _load_pkl_dataframe(args.pkl)
    num_classes = args.num_classes or _infer_num_classes(df, args.dataset)
    all_splits = _load_splits_json(args.splits_json)
    if args.fold < 0 or args.fold >= len(all_splits):
        raise ValueError(f"Fold must be in [0, {len(all_splits) - 1}]")

    split_cases = set(all_splits[args.fold][args.split])
    dataset = _build_monai_dataset_from_pkl(
        df,
        args.dataset,
        num_classes,
        spatial_size=tuple(args.spatial_size),
        do_resize=args.resize,
    )
    selected_items = [item for item in dataset.data if item["case_name"] in split_cases]
    if len(selected_items) != len(split_cases):
        found = {item["case_name"] for item in selected_items}
        missing = sorted(split_cases - found)
        raise RuntimeError(f"Split contains {len(missing)} cases absent from the pickle: {missing[:10]}")
    selected_dataset = MonaiDataset(data=selected_items, transform=dataset.transform)

    result = evaluate_swinunetr(
        checkpoint,
        selected_dataset,
        num_classes,
        device,
        args.batch_size,
        pred_dir=prediction_dir,
    )
    summary = {
        "checkpoint": str(checkpoint),
        "dataset": args.dataset,
        "fold": args.fold,
        "split": args.split,
        "num_cases": result.n_cases,
        "num_classes": num_classes,
        "spatial_size": args.spatial_size,
        "metrics": result.metrics,
    }
    with open(output_dir / "prediction_index.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["prediction_index", "case_name"])
        writer.writeheader()
        writer.writerows(
            {"prediction_index": index, "case_name": item["case_name"]}
            for index, item in enumerate(selected_items)
        )
    with open(output_dir / "metrics_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved {result.n_cases} predictions and probabilities to {prediction_dir}")


if __name__ == "__main__":
    main()
