#!/usr/bin/env python3
"""Export standardized logits/labels/spacing bundles without training a model."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for path in (HERE, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baselines.swin_unetr.swin_unetr import (  # noqa: E402
    _build_monai_dataset_from_pkl,
    _infer_num_classes,
    _load_pkl_dataframe,
    _load_splits_json,
)
from evaluation.fold0_voxel_errors import load_model, resolve_device  # noqa: E402
from surface_normal_ordinal.io import (  # noqa: E402
    probabilities_to_logits,
    save_case_bundle,
    sha256,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", type=Path, required=True)
    parser.add_argument("--splits-json", type=Path, required=True)
    parser.add_argument("--dataset", choices=("MSD", "MNI", "ADNI", "COBRA"), required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument(
        "--inference-dir",
        type=Path,
        help="Existing SwinUNETR inference folder with prediction_index.csv and predictions/.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spatial-size", type=int, nargs=3, default=(64, 64, 64))
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--labels-dir", type=Path, default=None, help="NIfTI labels used to read case spacing.")
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=None,
        help="Explicit transformed voxel spacing; required when --labels-dir is unavailable.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _case_spacing(
    case_name: str,
    labels_dir: Path | None,
    explicit: tuple[float, float, float] | None,
    *,
    resize: bool,
    transformed_shape: tuple[int, int, int],
) -> tuple[float, float, float]:
    if labels_dir is None:
        if explicit is None:
            raise ValueError("Pass --labels-dir or an explicit --spacing.")
        return tuple(float(v) for v in explicit)
    candidates = (labels_dir / f"{case_name}.nii.gz", labels_dir / f"{case_name}.nii")
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"No NIfTI label found for {case_name} in {labels_dir}.")
    image = nib.load(str(path))
    spacing = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
    if resize:
        spacing = spacing * np.asarray(image.shape[:3], dtype=np.float64) / np.asarray(
            transformed_shape, dtype=np.float64
        )
    return tuple(float(v) for v in spacing)


def _probability_index(inference_dir: Path) -> dict[str, Path]:
    index_path = inference_dir / "prediction_index.csv"
    predictions = inference_dir / "predictions"
    if not index_path.is_file() or not predictions.is_dir():
        raise FileNotFoundError("Inference directory must contain prediction_index.csv and predictions/.")
    output: dict[str, Path] = {}
    with index_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            case_name = row["case_name"]
            index = int(row["prediction_index"])
            path = predictions / f"case_{index:04d}_prob.npy"
            if not path.is_file():
                raise FileNotFoundError(path)
            output[case_name] = path
    return output


def main() -> None:
    args = parse_args()
    for path in (args.pkl, args.splits_json):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.checkpoint is not None and not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.amp and args.device == "cpu":
        raise ValueError("--amp is only supported on CUDA.")
    existing = sorted(args.output_dir.glob("*.npz")) if args.output_dir.exists() else []
    if existing and not args.overwrite:
        raise FileExistsError("Output already contains bundles; pass --overwrite to replace matching files.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = _load_pkl_dataframe(args.pkl)
    num_classes = _infer_num_classes(dataframe, args.dataset)
    if num_classes != 3:
        raise ValueError(f"The audit requires exactly three classes, found {num_classes}.")
    splits = _load_splits_json(args.splits_json)
    if not 0 <= args.fold < len(splits):
        raise ValueError("Fold index is out of range.")
    selected_names = set(splits[args.fold][args.split])
    dataset = _build_monai_dataset_from_pkl(
        dataframe,
        args.dataset,
        num_classes,
        spatial_size=tuple(args.spatial_size),
        do_resize=args.resize,
    )
    selected_items = [item for item in dataset.data if item["case_name"] in selected_names]
    if len(selected_items) != len(selected_names):
        found = {item["case_name"] for item in selected_items}
        raise ValueError(f"Cases missing from dataset pickle: {sorted(selected_names - found)[:10]}")
    if args.max_cases is not None:
        selected_items = selected_items[: args.max_cases]

    from monai.data import DataLoader, Dataset as MonaiDataset

    loader = DataLoader(MonaiDataset(selected_items, transform=dataset.transform), batch_size=1, shuffle=False)
    device = resolve_device(args.device)
    model = None
    probability_paths = None
    if args.checkpoint is not None:
        model = load_model(args.checkpoint, tuple(args.spatial_size), num_classes, device)
    else:
        probability_paths = _probability_index(args.inference_dir)

    records: list[dict[str, Any]] = []
    for item, batch in zip(selected_items, loader, strict=True):
        case_name = str(item["case_name"])
        actual = str(batch["case_name"][0])
        if actual != case_name:
            raise RuntimeError("Dataset order changed while exporting bundles.")
        labels = batch["label"][0, 0].cpu().numpy().astype(np.uint8)
        if model is not None:
            images = batch["image"].to(device)
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                enabled=args.amp,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            ):
                logits = model(images)[0].float().cpu().numpy()
            source_path = args.checkpoint
        else:
            source_path = probability_paths.get(case_name)
            if source_path is None:
                raise ValueError(f"No saved probabilities indexed for {case_name}.")
            probabilities = np.load(source_path, allow_pickle=False)
            logits = probabilities_to_logits(probabilities)
        if logits.shape[1:] != labels.shape:
            raise ValueError(
                f"Shape mismatch for {case_name}: logits {logits.shape}, labels {labels.shape}. "
                "Use the same --spatial-size/--resize settings as inference."
            )
        spacing = _case_spacing(
            case_name,
            args.labels_dir,
            tuple(args.spacing) if args.spacing is not None else None,
            resize=args.resize,
            transformed_shape=labels.shape,
        )
        output_path = args.output_dir / f"{case_name}.npz"
        save_case_bundle(
            output_path,
            case_name=case_name,
            logits=logits,
            labels=labels,
            spacing=spacing,
        )
        records.append(
            {
                "case_name": case_name,
                "bundle": str(output_path.resolve()),
                "bundle_sha256": sha256(output_path),
                "source": str(source_path.resolve()),
                "spacing": list(spacing),
            }
        )
    manifest = {
        "schema_version": 1,
        "dataset": args.dataset,
        "fold": args.fold,
        "split": args.split,
        "spatial_size": list(args.spatial_size),
        "resize": args.resize,
        "amp": args.amp,
        "pkl": str(args.pkl.resolve()),
        "pkl_sha256": sha256(args.pkl),
        "splits_json": str(args.splits_json.resolve()),
        "splits_json_sha256": sha256(args.splits_json),
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "checkpoint_sha256": sha256(args.checkpoint) if args.checkpoint else None,
        "inference_dir": str(args.inference_dir.resolve()) if args.inference_dir else None,
        "case_count": len(records),
        "cases": records,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"case_count": len(records), "output_dir": str(args.output_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
