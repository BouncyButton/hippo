"""Probe a frozen segmentation model with generic semantic descriptors.

This script is intentionally self-contained so it can be run independently from
the older experimental proposal files:

    python semantic_constraints/probe_model.py

It loads a SwinUNETR checkpoint, runs a selected fold split, compares predicted
soft masks against ground-truth masks, and writes descriptor diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PROPOSAL_ROOT = REPO_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from monai.apps import DecathlonDataset
from monai.data import DataLoader
from monai.networks.nets import SwinUNETR
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, Resized, Spacingd

from semantic_constraints.primitives import (
    adjacent,
    boundary_length,
    centroid,
    compactness,
    connectedness,
    contains,
    distance,
    entropy,
    overlap,
    volume,
)


DEFAULT_CHECKPOINT = PROPOSAL_ROOT / "model_state_dict_constraint-informed_fold1_tr=0.1_size=64x64x64.pth"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "probe_outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe a frozen segmentation model by comparing prediction and GT descriptors."
    )
    parser.add_argument("--load-weights", type=Path, default=DEFAULT_CHECKPOINT, help="Model state_dict path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for JSON/CSV outputs.")
    parser.add_argument("--root-dir", type=Path, default=PROPOSAL_ROOT / "tmp", help="MONAI dataset root directory.")
    parser.add_argument("--task", default="Task04_Hippocampus", help="Decathlon task identifier.")
    parser.add_argument("--download", action="store_true", help="Ask MONAI to download the dataset if missing.")
    parser.add_argument("--fold", type=int, default=1, help="Fold to probe, starting from 1.")
    parser.add_argument("--folds", type=int, default=5, help="Number of K-fold splits.")
    parser.add_argument("--train-fraction", type=float, default=0.1, help="Fraction used for the train subset.")
    parser.add_argument("--split", choices=("val", "train", "both"), default="val", help="Dataset split to probe.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--pixdim", type=float, nargs=3, default=(1.5, 0.5, 1.5), help="Resampling spacing.")
    parser.add_argument("--spatial-size", type=int, nargs=3, default=(64, 64, 64), help="Model input size.")
    parser.add_argument("--num-classes", type=int, default=3, help="Number of segmentation classes.")
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=(1, 2),
        help="Foreground class ids to probe. Defaults to hippocampus labels 1 and 2.",
    )
    parser.add_argument("--pair", type=int, nargs=2, default=(1, 2), help="Class pair for relation descriptors.")
    parser.add_argument("--connectedness-steps", type=int, default=32, help="Soft connectedness dilation steps.")
    parser.add_argument("--max-batches", type=int, default=0, help="Limit batches for a quick probe. 0 means all.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"), help="Inference device.")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def build_transforms(pixdim: tuple[float, ...], spatial_size: tuple[int, ...]) -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
            Resized(keys=["image", "label"], spatial_size=spatial_size, mode=("bilinear", "nearest")),
        ]
    )


def build_dataset(args: argparse.Namespace) -> DecathlonDataset:
    return DecathlonDataset(
        root_dir=str(args.root_dir),
        task=args.task,
        section="training",
        download=args.download,
        transform=build_transforms(tuple(args.pixdim), tuple(args.spatial_size)),
    )


def build_fold_subsets(
    dataset: DecathlonDataset,
    folds: int,
    fold_number: int,
    train_fraction: float,
) -> tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    if fold_number < 1 or fold_number > folds:
        raise ValueError(f"Fold {fold_number} is outside the available range 1..{folds}.")

    sample_count = len(dataset)
    indices = np.arange(sample_count)
    np.random.RandomState(42).shuffle(indices)

    fold_sizes = np.full(folds, sample_count // folds, dtype=int)
    fold_sizes[: sample_count % folds] += 1
    starts = np.cumsum(fold_sizes)
    stop = int(starts[fold_number - 1])
    start = int(stop - fold_sizes[fold_number - 1])

    val_idx = indices[start:stop]
    train_idx = np.concatenate((indices[:start], indices[stop:]))
    train_limit = max(1, int(train_fraction * len(train_idx)))
    return torch.utils.data.Subset(dataset, train_idx[:train_limit]), torch.utils.data.Subset(dataset, val_idx)


def build_model(checkpoint_path: Path, num_classes: int, device: torch.device) -> torch.nn.Module:
    model = SwinUNETR(in_channels=1, out_channels=num_classes, use_checkpoint=True).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    labels = labels.squeeze(1).long()
    return F.one_hot(labels, num_classes=num_classes).movedim(-1, 1).float()


def hard_one_hot(pred_labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(pred_labels.long(), num_classes=num_classes).movedim(-1, 1).float()


def binary_dice(pred_mask: torch.Tensor, gt_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    dims = tuple(range(1, pred_mask.ndim))
    intersection = (pred_mask * gt_mask).sum(dim=dims)
    denom = pred_mask.sum(dim=dims) + gt_mask.sum(dim=dims)
    return (2.0 * intersection + eps) / (denom + eps)


def scalar_list(values: torch.Tensor) -> list[float]:
    return [float(v) for v in values.detach().cpu().reshape(-1)]


def vector_list(values: torch.Tensor) -> list[list[float]]:
    return [[float(x) for x in row] for row in values.detach().cpu()]


def get_batch_paths(batch: dict[str, Any], batch_size: int) -> list[str]:
    meta = batch.get("image_meta_dict")
    if not meta:
        return [""] * batch_size

    filenames = meta.get("filename_or_obj") if isinstance(meta, dict) else None
    if filenames is None:
        return [""] * batch_size
    if isinstance(filenames, (str, Path)):
        return [str(filenames)] * batch_size
    return [str(item) for item in filenames]


def add_class_metrics(
    rows: list[dict[str, Any]],
    base: dict[str, Any],
    class_id: int,
    pred_soft: torch.Tensor,
    pred_hard: torch.Tensor,
    gt: torch.Tensor,
    spacing: tuple[float, ...],
    connectedness_steps: int,
) -> None:
    pred_soft_mask = pred_soft[:, class_id]
    pred_hard_mask = pred_hard[:, class_id]
    gt_mask = gt[:, class_id]

    gt_volume = scalar_list(volume(gt_mask, spacing=spacing))
    pred_volume = scalar_list(volume(pred_soft_mask, spacing=spacing))
    gt_centroid = vector_list(centroid(gt_mask, spacing=spacing))
    pred_centroid = vector_list(centroid(pred_soft_mask, spacing=spacing))
    centroid_gap = scalar_list(distance(pred_soft_mask, gt_mask, spacing=spacing))
    gt_boundary = scalar_list(boundary_length(gt_mask, spacing=spacing))
    pred_boundary = scalar_list(boundary_length(pred_soft_mask, spacing=spacing))
    gt_compactness = scalar_list(compactness(gt_mask, spacing=spacing))
    pred_compactness = scalar_list(compactness(pred_soft_mask, spacing=spacing))
    gt_connectedness = scalar_list(connectedness(gt_mask, steps=connectedness_steps))
    pred_connectedness = scalar_list(connectedness(pred_soft_mask, steps=connectedness_steps))
    pred_entropy = scalar_list(entropy(pred_soft_mask))
    dice = scalar_list(binary_dice(pred_hard_mask, gt_mask))

    for i in range(pred_soft.shape[0]):
        row = dict(base)
        row.update(
            {
                "metric_scope": "class",
                "class_id": class_id,
                "pair": "",
                "dice": dice[i],
                "gt_volume": gt_volume[i],
                "pred_volume": pred_volume[i],
                "volume_gap": pred_volume[i] - gt_volume[i],
                "volume_abs_gap": abs(pred_volume[i] - gt_volume[i]),
                "gt_centroid": gt_centroid[i],
                "pred_centroid": pred_centroid[i],
                "centroid_distance": centroid_gap[i],
                "gt_boundary_length": gt_boundary[i],
                "pred_boundary_length": pred_boundary[i],
                "boundary_abs_gap": abs(pred_boundary[i] - gt_boundary[i]),
                "gt_compactness": gt_compactness[i],
                "pred_compactness": pred_compactness[i],
                "compactness_abs_gap": abs(pred_compactness[i] - gt_compactness[i]),
                "gt_connectedness": gt_connectedness[i],
                "pred_connectedness": pred_connectedness[i],
                "connectedness_gap": pred_connectedness[i] - gt_connectedness[i],
                "pred_entropy": pred_entropy[i],
            }
        )
        rows.append(row)


def add_pair_metrics(
    rows: list[dict[str, Any]],
    base: dict[str, Any],
    pair: tuple[int, int],
    pred_soft: torch.Tensor,
    gt: torch.Tensor,
    spacing: tuple[float, ...],
) -> None:
    a, b = pair
    pred_a = pred_soft[:, a]
    pred_b = pred_soft[:, b]
    gt_a = gt[:, a]
    gt_b = gt[:, b]

    gt_distance = scalar_list(distance(gt_a, gt_b, spacing=spacing))
    pred_distance = scalar_list(distance(pred_a, pred_b, spacing=spacing))
    gt_overlap = scalar_list(overlap(gt_a, gt_b, mode="dice"))
    pred_overlap = scalar_list(overlap(pred_a, pred_b, mode="dice"))
    gt_adjacent = scalar_list(adjacent(gt_a, gt_b))
    pred_adjacent = scalar_list(adjacent(pred_a, pred_b))
    gt_contains_ab = scalar_list(contains(gt_a, gt_b))
    pred_contains_ab = scalar_list(contains(pred_a, pred_b))
    gt_contains_ba = scalar_list(contains(gt_b, gt_a))
    pred_contains_ba = scalar_list(contains(pred_b, pred_a))

    for i in range(pred_soft.shape[0]):
        row = dict(base)
        row.update(
            {
                "metric_scope": "pair",
                "class_id": "",
                "pair": f"{a}-{b}",
                "gt_pair_distance": gt_distance[i],
                "pred_pair_distance": pred_distance[i],
                "pair_distance_gap": pred_distance[i] - gt_distance[i],
                "pair_distance_abs_gap": abs(pred_distance[i] - gt_distance[i]),
                "gt_pair_overlap": gt_overlap[i],
                "pred_pair_overlap": pred_overlap[i],
                "pair_overlap_gap": pred_overlap[i] - gt_overlap[i],
                "gt_adjacent": gt_adjacent[i],
                "pred_adjacent": pred_adjacent[i],
                "adjacent_gap": pred_adjacent[i] - gt_adjacent[i],
                "gt_contains_a_b": gt_contains_ab[i],
                "pred_contains_a_b": pred_contains_ab[i],
                "contains_a_b_gap": pred_contains_ab[i] - gt_contains_ab[i],
                "gt_contains_b_a": gt_contains_ba[i],
                "pred_contains_b_a": pred_contains_ba[i],
                "contains_b_a_gap": pred_contains_ba[i] - gt_contains_ba[i],
            }
        )
        rows.append(row)


def probe_split(
    model: torch.nn.Module,
    data_loader: DataLoader,
    split_name: str,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample_offset = 0
    spacing = tuple(args.pixdim)
    pair = tuple(args.pair)

    from tqdm import tqdm
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Probing {split_name}"):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break

            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(images)
            pred_soft = torch.softmax(logits, dim=1)
            pred_labels = pred_soft.argmax(dim=1)
            pred_hard = hard_one_hot(pred_labels, args.num_classes)
            gt = one_hot(labels, args.num_classes)
            paths = get_batch_paths(batch, images.shape[0])

            for class_id in args.classes:
                add_class_metrics(
                    rows,
                    {
                        "split": split_name,
                        "batch_idx": batch_idx,
                        "batch_sample_idx": None,
                        "sample_idx": None,
                        "image": None,
                    },
                    class_id=class_id,
                    pred_soft=pred_soft,
                    pred_hard=pred_hard,
                    gt=gt,
                    spacing=spacing,
                    connectedness_steps=args.connectedness_steps,
                )
                for i in range(images.shape[0]):
                    rows[-images.shape[0] + i]["batch_sample_idx"] = i
                    rows[-images.shape[0] + i]["sample_idx"] = sample_offset + i
                    rows[-images.shape[0] + i]["image"] = paths[i] if i < len(paths) else ""

            add_pair_metrics(
                rows,
                {
                    "split": split_name,
                    "batch_idx": batch_idx,
                    "batch_sample_idx": None,
                    "sample_idx": None,
                    "image": None,
                },
                pair=pair,
                pred_soft=pred_soft,
                gt=gt,
                spacing=spacing,
            )
            for i in range(images.shape[0]):
                rows[-images.shape[0] + i]["batch_sample_idx"] = i
                rows[-images.shape[0] + i]["sample_idx"] = sample_offset + i
                rows[-images.shape[0] + i]["image"] = paths[i] if i < len(paths) else ""

            sample_offset += images.shape[0]

    return rows


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        prefix = f"{row['split']}.{row['metric_scope']}"
        if row["class_id"] != "":
            prefix += f".class_{row['class_id']}"
        if row["pair"] != "":
            prefix += f".pair_{row['pair']}"

        for key, value in row.items():
            if key in {"batch_idx", "batch_sample_idx", "sample_idx", "class_id"}:
                continue
            if is_number(value):
                grouped[f"{prefix}.{key}"].append(float(value))

    summary = {}
    for key, values in grouped.items():
        tensor = torch.tensor(values, dtype=torch.float32)
        summary[key] = {
            "mean": float(tensor.mean().item()),
            "std": float(tensor.std(unbiased=False).item()) if tensor.numel() > 1 else 0.0,
            "min": float(tensor.min().item()),
            "max": float(tensor.max().item()),
            "n": int(tensor.numel()),
        }

    summary["failure_examples"] = worst_samples(rows)
    summary["error_alignment"] = error_alignment(rows)
    return summary


def worst_samples(rows: list[dict[str, Any]], top_k: int = 8) -> list[dict[str, Any]]:
    dice_by_sample: dict[tuple[str, int], list[float]] = defaultdict(list)
    image_by_sample: dict[tuple[str, int], str] = {}
    for row in rows:
        if row["metric_scope"] != "class" or row.get("class_id") == "":
            continue
        if is_number(row.get("dice")):
            key = (row["split"], int(row["sample_idx"]))
            dice_by_sample[key].append(float(row["dice"]))
            image_by_sample[key] = row.get("image") or ""

    ranked = []
    for key, dice_values in dice_by_sample.items():
        mean_dice = sum(dice_values) / len(dice_values)
        ranked.append(
            {
                "split": key[0],
                "sample_idx": key[1],
                "mean_foreground_dice": mean_dice,
                "image": image_by_sample.get(key, ""),
            }
        )
    return sorted(ranked, key=lambda item: item["mean_foreground_dice"])[:top_k]


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32)
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if denom.item() == 0.0:
        return None
    return float(torch.dot(x, y).div(denom).item())


def error_alignment(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    dice_loss_by_sample: dict[tuple[str, int], list[float]] = defaultdict(list)
    descriptors: dict[str, dict[tuple[str, int], float]] = defaultdict(dict)

    for row in rows:
        key = (row["split"], int(row["sample_idx"]))
        if row["metric_scope"] == "class" and is_number(row.get("dice")):
            dice_loss_by_sample[key].append(1.0 - float(row["dice"]))
            for metric in ("volume_abs_gap", "centroid_distance", "boundary_abs_gap", "compactness_abs_gap"):
                if is_number(row.get(metric)):
                    descriptors[f"class_{row['class_id']}.{metric}"][key] = float(row[metric])
        elif row["metric_scope"] == "pair":
            for metric in ("pair_distance_abs_gap", "pair_overlap_gap", "adjacent_gap"):
                if is_number(row.get(metric)):
                    descriptors[f"pair_{row['pair']}.{metric}"][key] = abs(float(row[metric]))

    mean_loss = {key: sum(values) / len(values) for key, values in dice_loss_by_sample.items() if values}
    output = {}
    for name, values_by_sample in descriptors.items():
        common = sorted(set(mean_loss) & set(values_by_sample))
        output[f"pearson_{name}_vs_dice_loss"] = pearson(
            [values_by_sample[key] for key in common],
            [mean_loss[key] for key in common],
        )
    return output


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()})


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    dataset = build_dataset(args)
    train_subset, val_subset = build_fold_subsets(dataset, args.folds, args.fold, args.train_fraction)
    model = build_model(args.load_weights, args.num_classes, device)

    selected = ("train", "val") if args.split == "both" else (args.split,)
    subsets = {"train": train_subset, "val": val_subset}

    all_rows = []
    for split_name in selected:
        loader = DataLoader(
            subsets[split_name],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        split_rows = probe_split(model, loader, split_name, args, device)
        all_rows.extend(split_rows)

    summary = summarize_rows(all_rows)
    payload = {
        "config": vars(args),
        "num_rows": len(all_rows),
        "summary": summary,
        "rows": all_rows,
    }

    stem = f"probe_fold{args.fold}_{args.split}"
    json_path = args.output_dir / f"{stem}.json"
    csv_path = args.output_dir / f"{stem}.csv"
    write_json(json_path, payload)
    write_csv(csv_path, all_rows)

    print(f"loaded weights: {args.load_weights}")
    print(f"device: {device}")
    print(f"rows: {len(all_rows)}")
    print(f"wrote JSON: {json_path}")
    print(f"wrote CSV: {csv_path}")
    for item in summary["failure_examples"][:5]:
        print(
            "failure",
            f"split={item['split']}",
            f"sample={item['sample_idx']}",
            f"mean_fg_dice={item['mean_foreground_dice']:.4f}",
            item["image"],
        )


if __name__ == "__main__":
    main()
