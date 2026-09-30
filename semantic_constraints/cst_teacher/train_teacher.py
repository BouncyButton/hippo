#!/usr/bin/env python3
"""Train MRI-only descriptor and mask-anomaly CST teachers on an MSD fold."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.swin_unetr.swin_unetr import (  # noqa: E402
    _build_monai_dataset_from_pkl,
    _load_pkl_dataframe,
    _load_splits_json,
)
from semantic_constraints.cst_teacher.data import (  # noqa: E402
    CSTSetDataset,
    corrupt_mask_elements,
    random_valid_elements,
)
from semantic_constraints.cst_teacher.descriptors import DESCRIPTOR_NAMES  # noqa: E402
from semantic_constraints.cst_teacher.losses import (  # noqa: E402
    anomaly_detection_loss,
    descriptor_quantile_loss,
    slice_profile_loss,
)
from semantic_constraints.cst_teacher.model import (  # noqa: E402
    CSTDescriptorTeacher,
    CSTMaskAnomalyTeacher,
)


@dataclass(frozen=True)
class TrainConfig:
    fold: int
    spatial_size: tuple[int, int, int]
    set_size: int
    slab_depth: int
    inplane_size: int
    channels: tuple[int, ...]
    heads: int
    descriptor_weight: float
    profile_weight: float
    anomaly_weight: float
    corruption_rate: float
    minimum_set_size: int
    profile_loss: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", type=Path, default=REPO_ROOT / "datasets/Dataset101_MSD/msd_hippocampus_full.pkl")
    parser.add_argument("--splits-json", type=Path, default=REPO_ROOT / "datasets/Dataset101_MSD/splits_final.json")
    parser.add_argument("--fold", type=int, default=0, help="Discovery fold; zero-based.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "semantic_constraints/cst_teacher/runs/fold0")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--spatial-size", type=int, nargs=3, default=(64, 64, 64))
    parser.add_argument("--set-size", type=int, default=12)
    parser.add_argument("--minimum-set-size", type=int, default=4)
    parser.add_argument("--slab-depth", type=int, default=3)
    parser.add_argument("--inplane-size", type=int, default=32)
    parser.add_argument(
        "--cache-dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Materialize the small 2.5-D sets once instead of rebuilding them every epoch.",
    )
    parser.add_argument("--channels", type=int, nargs="+", default=(16, 32, 64))
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--descriptor-weight", type=float, default=1.0)
    parser.add_argument("--profile-weight", type=float, default=1.0)
    parser.add_argument("--profile-loss", choices=("smooth_l1", "l1"), default="smooth_l1")
    parser.add_argument("--anomaly-weight", type=float, default=0.5)
    parser.add_argument("--corruption-rate", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--minimum-improvement", type=float, default=1e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--max-train-cases", type=int, default=0)
    parser.add_argument("--max-val-cases", type=int, default=0)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def subset_dataframe(dataframe: Any, names: list[str], limit: int) -> Any:
    requested = names[:limit] if limit > 0 else names
    subset = dataframe[dataframe["subject_id"].isin(requested)].copy()
    found = set(subset["subject_id"].tolist())
    missing = set(requested) - found
    if missing:
        raise ValueError(f"split references {len(missing)} missing cases: {sorted(missing)[:5]}")
    order = {name: index for index, name in enumerate(requested)}
    return subset.sort_values("subject_id", key=lambda column: column.map(order)).reset_index(drop=True)


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    dataframe = _load_pkl_dataframe(args.pkl)
    splits = _load_splits_json(args.splits_json)
    if not 0 <= args.fold < len(splits):
        raise ValueError(f"fold must be in [0, {len(splits) - 1}]")
    fold = splits[args.fold]
    train_frame = subset_dataframe(dataframe, fold["train"], args.max_train_cases)
    val_frame = subset_dataframe(dataframe, fold["val"], args.max_val_cases)
    common = set(train_frame["subject_id"]) & set(val_frame["subject_id"])
    if common:
        raise ValueError(f"train/validation leakage detected: {sorted(common)[:5]}")

    def make_dataset(frame: Any) -> CSTSetDataset:
        base = _build_monai_dataset_from_pkl(
            frame,
            "MSD",
            3,
            spatial_size=tuple(args.spatial_size),
            do_resize=False,
        )
        return CSTSetDataset(
            base,
            set_size=args.set_size,
            slab_depth=args.slab_depth,
            inplane_size=getattr(args, "inplane_size", 32),
            cache=getattr(args, "cache_dataset", True),
        )

    train = DataLoader(
        make_dataset(train_frame),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val = DataLoader(
        make_dataset(val_frame),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return train, val


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def run_epoch(
    descriptor_teacher: CSTDescriptorTeacher,
    anomaly_teacher: CSTMaskAnomalyTeacher,
    loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, Any]:
    training = optimizer is not None
    descriptor_teacher.train(training)
    anomaly_teacher.train(training)
    totals = {
        "loss": 0.0,
        "descriptor_loss": 0.0,
        "profile_loss": 0.0,
        "anomaly_loss": 0.0,
        "profile_absolute_error": 0.0,
        "anomaly_correct": 0.0,
        "anomaly_count": 0.0,
        "case_count": 0.0,
    }
    descriptor_absolute_error = torch.zeros(len(DESCRIPTOR_NAMES), device=device)
    descriptor_coverage = torch.zeros(len(DESCRIPTOR_NAMES), device=device)

    context = torch.enable_grad if training else torch.no_grad
    with context():
        for raw_batch in loader:
            batch = batch_to_device(raw_batch, device)
            batch_size, set_size = batch["image_slabs"].shape[:2]
            if training:
                valid = random_valid_elements(
                    batch_size,
                    set_size,
                    minimum=config.minimum_set_size,
                    device=device,
                )
            else:
                valid = batch["valid_elements"]

            output = descriptor_teacher(batch["image_slabs"], batch["metadata"], valid)
            descriptor_loss = descriptor_quantile_loss(output.descriptor_quantiles, batch["descriptors"])
            profile_loss = slice_profile_loss(
                output.slice_profiles,
                batch["slice_profiles"],
                valid,
                loss_kind=config.profile_loss,
            )

            corrupted_masks, anomaly_targets = corrupt_mask_elements(
                batch["mask_slices"],
                valid_elements=valid,
                corruption_rate=config.corruption_rate,
            )
            anomaly_logits, _ = anomaly_teacher(
                batch["image_slabs"],
                corrupted_masks,
                batch["metadata"],
                valid,
            )
            anomaly_loss = anomaly_detection_loss(anomaly_logits, anomaly_targets, valid)
            loss = (
                config.descriptor_weight * descriptor_loss
                + config.profile_weight * profile_loss
                + config.anomaly_weight * anomaly_loss
            )

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(descriptor_teacher.parameters()) + list(anomaly_teacher.parameters()),
                    max_norm=5.0,
                )
                optimizer.step()

            case_weight = float(batch_size)
            totals["loss"] += float(loss.detach()) * case_weight
            totals["descriptor_loss"] += float(descriptor_loss.detach()) * case_weight
            totals["profile_loss"] += float(profile_loss.detach()) * case_weight
            totals["anomaly_loss"] += float(anomaly_loss.detach()) * case_weight
            totals["case_count"] += case_weight

            median = output.descriptor_quantiles[..., 1]
            descriptor_absolute_error += (median - batch["descriptors"]).abs().sum(dim=0)
            inside = (
                (batch["descriptors"] >= output.descriptor_quantiles[..., 0])
                & (batch["descriptors"] <= output.descriptor_quantiles[..., 2])
            )
            descriptor_coverage += inside.float().sum(dim=0)
            profile_error = (output.slice_profiles - batch["slice_profiles"]).abs().mean(dim=2)
            totals["profile_absolute_error"] += float(
                (profile_error * valid).sum() / valid.sum().clamp_min(1)
            ) * case_weight
            predicted_anomaly = anomaly_logits >= 0
            totals["anomaly_correct"] += float(((predicted_anomaly == anomaly_targets) & valid).sum())
            totals["anomaly_count"] += float(valid.sum())

    count = max(totals.pop("case_count"), 1.0)
    metrics: dict[str, Any] = {
        key: value / count
        for key, value in totals.items()
        if key not in {"anomaly_correct", "anomaly_count"}
    }
    metrics["anomaly_accuracy"] = totals["anomaly_correct"] / max(totals["anomaly_count"], 1.0)
    metrics["descriptor_mae"] = {
        name: float(value / count)
        for name, value in zip(DESCRIPTOR_NAMES, descriptor_absolute_error, strict=True)
    }
    metrics["descriptor_interval_coverage"] = {
        name: float(value / count)
        for name, value in zip(DESCRIPTOR_NAMES, descriptor_coverage, strict=True)
    }
    return metrics


def main() -> None:
    args = parse_args()
    if args.set_size > args.spatial_size[1]:
        raise ValueError("set size cannot exceed the coronal axis size")
    if not 1 <= args.minimum_set_size <= args.set_size:
        raise ValueError("minimum set size must be in [1, set size]")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    train_loader, val_loader = build_loaders(args)
    config = TrainConfig(
        fold=args.fold,
        spatial_size=tuple(args.spatial_size),
        set_size=args.set_size,
        slab_depth=args.slab_depth,
        inplane_size=args.inplane_size,
        channels=tuple(args.channels),
        heads=args.heads,
        descriptor_weight=args.descriptor_weight,
        profile_weight=args.profile_weight,
        anomaly_weight=args.anomaly_weight,
        corruption_rate=args.corruption_rate,
        minimum_set_size=args.minimum_set_size,
        profile_loss=args.profile_loss,
    )

    descriptor_teacher = CSTDescriptorTeacher(
        slab_channels=args.slab_depth,
        channels=tuple(args.channels),
        heads=args.heads,
    ).to(device)
    anomaly_teacher = CSTMaskAnomalyTeacher(
        slab_channels=args.slab_depth,
        channels=tuple(args.channels),
        heads=args.heads,
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(descriptor_teacher.parameters()) + list(anomaly_teacher.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            descriptor_teacher,
            anomaly_teacher,
            train_loader,
            config,
            device,
            optimizer,
        )
        val_metrics = run_epoch(
            descriptor_teacher,
            anomaly_teacher,
            val_loader,
            config,
            device,
            None,
        )
        record = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
        history.append(record)
        print(json.dumps(record, sort_keys=True))
        if val_metrics["loss"] < best_loss - args.minimum_improvement:
            best_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "schema": "semantic_constraints.cst_teacher.v1",
                    "config": asdict(config),
                    "descriptor_names": DESCRIPTOR_NAMES,
                    "descriptor_teacher_state_dict": descriptor_teacher.state_dict(),
                    "anomaly_teacher_state_dict": anomaly_teacher.state_dict(),
                    "epoch": epoch,
                    "validation": val_metrics,
                },
                args.output_dir / "best_teacher.pt",
            )
        else:
            epochs_without_improvement += 1
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(
                json.dumps(
                    {
                        "early_stopping": True,
                        "epoch": epoch,
                        "best_validation_loss": best_loss,
                        "patience": args.patience,
                    },
                    sort_keys=True,
                )
            )
            break


if __name__ == "__main__":
    main()
