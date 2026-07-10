"""Retrain a segmentation model with selected fuzzy semantic constraints.

This is the first end-to-end training script for the proposed pipeline:

    selected_constraints.json -> compiled fuzzy loss -> constrained retraining

It is intentionally standalone and minimal. It does not modify the older
proposal scripts and does not require W&B.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PROPOSAL_ROOT = REPO_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from monai.apps import DecathlonDataset
from monai.data import DataLoader
from monai.losses import DiceLoss
from monai.networks.nets import SwinUNETR
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, Resized, Spacingd
from monai.utils import set_determinism
from torch.optim import AdamW
from tqdm import tqdm

from semantic_constraints.compiled_losses import SemanticConstraintLoss


DEFAULT_CONSTRAINTS = Path(__file__).resolve().parent / "probe_outputs" / "selected_constraints.json"
DEFAULT_RUN_ROOT = Path(__file__).resolve().parent / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SwinUNETR with compiled semantic-constraint losses.")
    parser.add_argument("--constraints", type=Path, default=DEFAULT_CONSTRAINTS, help="selected_constraints.json path.")
    parser.add_argument(
        "--no-constraints",
        action="store_true",
        help="Fine-tune with segmentation loss only for a matched baseline run.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Run output directory.")
    parser.add_argument("--root-dir", type=Path, default=PROPOSAL_ROOT / "tmp", help="MONAI dataset root directory.")
    parser.add_argument("--task", default="Task04_Hippocampus", help="Decathlon task identifier.")
    parser.add_argument("--download", action="store_true", help="Ask MONAI to download the dataset if missing.")
    parser.add_argument("--fold", type=int, default=1, help="Fold to train, starting from 1.")
    parser.add_argument("--folds", type=int, default=5, help="Number of K-fold splits.")
    parser.add_argument(
        "--holdout-fold",
        type=int,
        default=0,
        help="Optional final holdout fold to exclude from training. 0 disables this.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.1, help="Fraction of train fold to use.")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="AdamW weight decay.")
    parser.add_argument("--constraint-multiplier", type=float, default=1.0, help="Global multiplier on compiled loss.")
    parser.add_argument("--num-classes", type=int, default=3, help="Number of segmentation classes.")
    parser.add_argument("--foreground-classes", type=int, nargs="+", default=(1, 2), help="Classes used for Dice reporting.")
    parser.add_argument("--pixdim", type=float, nargs=3, default=(1.5, 0.5, 1.5), help="Resampling spacing.")
    parser.add_argument("--spatial-size", type=int, nargs=3, default=(64, 64, 64), help="Model input size.")
    parser.add_argument("--init-weights", type=Path, default=None, help="Optional checkpoint to initialize from.")
    parser.add_argument("--max-train-batches", type=int, default=0, help="Limit train batches for smoke tests. 0 means all.")
    parser.add_argument("--max-val-batches", type=int, default=0, help="Limit val batches for smoke tests. 0 means all.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"), help="Training device.")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def default_output_dir(args: argparse.Namespace) -> Path:
    if args.no_constraints:
        return DEFAULT_RUN_ROOT / f"baseline_finetune_fold{args.fold}"
    mode = constraints_fuzzy_mode(args.constraints)
    return DEFAULT_RUN_ROOT / f"constrained_{mode}_fold{args.fold}"


def constraints_fuzzy_mode(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    constraints = payload.get("constraints", [])
    if not constraints:
        return "none"
    return str(constraints[0].get("fuzzy", {}).get("mode", "unknown"))


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
) -> tuple[torch.utils.data.Subset, torch.utils.data.Subset, Optional[torch.utils.data.Subset]]:
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
    return torch.utils.data.Subset(dataset, train_idx[:train_limit]), torch.utils.data.Subset(dataset, val_idx), None


def build_fold_subsets_with_holdout(
    dataset: DecathlonDataset,
    folds: int,
    discovery_fold: int,
    holdout_fold: int,
    train_fraction: float,
) -> tuple[torch.utils.data.Subset, torch.utils.data.Subset, Optional[torch.utils.data.Subset]]:
    if holdout_fold == 0:
        return build_fold_subsets(dataset, folds, discovery_fold, train_fraction)
    if holdout_fold < 1 or holdout_fold > folds:
        raise ValueError(f"Holdout fold {holdout_fold} is outside the available range 1..{folds}.")
    if holdout_fold == discovery_fold:
        raise ValueError("holdout-fold must differ from the discovery/validation fold.")

    sample_count = len(dataset)
    indices = np.arange(sample_count)
    np.random.RandomState(42).shuffle(indices)

    fold_sizes = np.full(folds, sample_count // folds, dtype=int)
    fold_sizes[: sample_count % folds] += 1
    starts = np.cumsum(fold_sizes)

    fold_indices = []
    previous_stop = 0
    for fold_size, stop in zip(fold_sizes, starts):
        start = int(previous_stop)
        stop = int(stop)
        fold_indices.append(indices[start:stop])
        previous_stop = stop

    discovery_idx = fold_indices[discovery_fold - 1]
    holdout_idx = fold_indices[holdout_fold - 1]
    train_idx = np.concatenate(
        [fold for idx, fold in enumerate(fold_indices, start=1) if idx not in {discovery_fold, holdout_fold}]
    )
    train_limit = max(1, int(train_fraction * len(train_idx)))
    return (
        torch.utils.data.Subset(dataset, train_idx[:train_limit]),
        torch.utils.data.Subset(dataset, discovery_idx),
        torch.utils.data.Subset(dataset, holdout_idx),
    )


def build_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    model = SwinUNETR(in_channels=1, out_channels=args.num_classes, use_checkpoint=True).to(device)
    if args.init_weights is not None:
        checkpoint = torch.load(args.init_weights, map_location=device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        model.load_state_dict(checkpoint)
    return model


def dice_per_class(
    pred_labels: torch.Tensor,
    labels: torch.Tensor,
    class_ids: list[int],
    eps: float = 1e-8,
) -> dict[int, torch.Tensor]:
    labels = labels.squeeze(1).long()
    output = {}
    for class_id in class_ids:
        pred_mask = (pred_labels == class_id).float()
        gt_mask = (labels == class_id).float()
        dims = tuple(range(1, pred_mask.ndim))
        intersection = (pred_mask * gt_mask).sum(dim=dims)
        denom = pred_mask.sum(dim=dims) + gt_mask.sum(dim=dims)
        output[class_id] = ((2.0 * intersection + eps) / (denom + eps)).mean()
    return output


def evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals = {class_id: 0.0 for class_id in args.foreground_classes}
    sample_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if args.max_val_batches > 0 and batch_idx >= args.max_val_batches:
                break
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            batch_dice = dice_per_class(preds, labels, list(args.foreground_classes))
            batch_size = images.shape[0]
            for class_id, value in batch_dice.items():
                totals[class_id] += float(value.detach().cpu()) * batch_size
            sample_count += batch_size

    if sample_count == 0:
        return {"val/foreground_dice": 0.0}

    metrics = {f"val/dice_class_{class_id}": total / sample_count for class_id, total in totals.items()}
    metrics["val/foreground_dice"] = sum(metrics.values()) / len(metrics)
    return metrics


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


@torch.no_grad()
def parameter_l2_squared(model: torch.nn.Module) -> float:
    """Return ||w||^2 over the trainable parameters decayed by AdamW."""
    total = torch.zeros((), device=next(model.parameters()).device)
    for parameter in model.parameters():
        if parameter.requires_grad:
            total += parameter.detach().float().square().sum()
    return scalar(total)


def accumulate_metric(totals: dict[str, float], key: str, value: float, batch_size: int) -> None:
    totals[key] = totals.get(key, 0.0) + value * batch_size


def train_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    seg_loss_fn: DiceLoss,
    constraint_loss: Optional[SemanticConstraintLoss],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    sample_count = 0

    progress = tqdm(train_loader, desc="train", leave=False)
    for batch_idx, batch in enumerate(progress):
        if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
            break

        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        batch_size = images.shape[0]

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        seg_loss = seg_loss_fn(logits, labels)
        preds = logits.argmax(dim=1)
        batch_dice = dice_per_class(preds, labels, list(args.foreground_classes))
        if constraint_loss is None:
            constraint_out = {
                "unweighted_loss": {},
                "truth": {},
                "violation": {},
                "value": {},
            }
            weighted_constraint_loss = logits.sum() * 0.0
        else:
            constraint_out = constraint_loss(logits)
            weighted_constraint_loss = args.constraint_multiplier * constraint_out["loss"]
        loss = seg_loss + weighted_constraint_loss
        loss.backward()
        optimizer.step()

        sample_count += batch_size
        accumulate_metric(totals, "train/loss", scalar(loss), batch_size)
        accumulate_metric(totals, "train/seg_loss", scalar(seg_loss), batch_size)
        accumulate_metric(totals, "train/constraint_loss", scalar(weighted_constraint_loss), batch_size)
        for class_id, value in batch_dice.items():
            accumulate_metric(totals, f"train/dice_class_{class_id}", scalar(value), batch_size)

        for name, value in constraint_out["unweighted_loss"].items():
            accumulate_metric(totals, f"constraint/unweighted_loss/{name}", scalar(value), batch_size)
        for name, value in constraint_out["truth"].items():
            accumulate_metric(totals, f"constraint/truth/{name}", scalar(value), batch_size)
        for name, value in constraint_out["violation"].items():
            accumulate_metric(totals, f"constraint/violation/{name}", scalar(value), batch_size)
        for name, value in constraint_out["value"].items():
            accumulate_metric(totals, f"constraint/value/{name}", scalar(value), batch_size)

        progress.set_postfix(
            loss=f"{scalar(loss):.4f}",
            seg=f"{scalar(seg_loss):.4f}",
            constraint=f"{scalar(weighted_constraint_loss):.4f}",
        )

    if sample_count == 0:
        return {}
    metrics = {key: value / sample_count for key, value in totals.items()}
    train_dice_keys = [f"train/dice_class_{class_id}" for class_id in args.foreground_classes]
    if train_dice_keys:
        metrics["train/foreground_dice"] = sum(metrics[key] for key in train_dice_keys) / len(train_dice_keys)
    return metrics


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


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_ready(payload)) + "\n")


def save_checkpoint(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir or default_output_dir(args)
    set_determinism(seed=args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    dataset = build_dataset(args)
    train_subset, val_subset, holdout_subset = build_fold_subsets_with_holdout(
        dataset,
        args.folds,
        args.fold,
        args.holdout_fold,
        args.train_fraction,
    )
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(args, device)
    seg_loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
    if args.no_constraints:
        constraint_loss = None
    else:
        constraint_loss = SemanticConstraintLoss.from_json(
            path=args.constraints,
            spacing=tuple(args.pixdim),
            input_is_logits=True,
        ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "config.json",
        {
            "args": vars(args),
            "device": str(device),
            "constraint_mode": "none" if args.no_constraints else "compiled",
            "train_samples": len(train_subset),
            "val_samples": len(val_subset),
            "holdout_samples": len(holdout_subset) if holdout_subset is not None else 0,
        },
    )

    best_val_dice = -1.0
    history = []
    for epoch in range(args.epochs):
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            seg_loss_fn=seg_loss_fn,
            constraint_loss=constraint_loss,
            args=args,
            device=device,
        )
        train_metrics["train/weight_l2_squared"] = parameter_l2_squared(model)
        val_metrics = evaluate(model, val_loader, args, device)
        epoch_metrics = {
            "epoch": epoch + 1,
            **train_metrics,
            **val_metrics,
        }
        history.append(epoch_metrics)
        append_jsonl(args.output_dir / "metrics.jsonl", epoch_metrics)
        write_json(args.output_dir / "metrics_history.json", {"history": history})
        save_checkpoint(model, args.output_dir / "latest_model_weights.pth")

        val_dice = epoch_metrics.get("val/foreground_dice", 0.0)
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            save_checkpoint(model, args.output_dir / "best_model_weights.pth")

        print(
            f"epoch {epoch + 1}/{args.epochs}",
            f"loss={epoch_metrics.get('train/loss', 0.0):.4f}",
            f"seg={epoch_metrics.get('train/seg_loss', 0.0):.4f}",
            f"constraint={epoch_metrics.get('train/constraint_loss', 0.0):.4f}",
            f"weight_l2_sq={epoch_metrics.get('train/weight_l2_squared', 0.0):.6e}",
            f"train_fg_dice={epoch_metrics.get('train/foreground_dice', 0.0):.4f}",
            f"val_fg_dice={val_dice:.4f}",
        )

    print(f"run directory: {args.output_dir}")
    print(f"best val foreground dice: {best_val_dice:.4f}")


if __name__ == "__main__":
    main()
