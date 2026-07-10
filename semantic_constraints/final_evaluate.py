"""Final holdout evaluation for baseline vs constrained fine-tuned models.

Use this only after constraint selection and training settings are fixed. The
holdout fold should be excluded from fine-tuning with
``train_with_constraints.py --holdout-fold``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PROPOSAL_ROOT = REPO_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from monai.apps import DecathlonDataset
from monai.data import DataLoader
from monai.networks.nets import SwinUNETR
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, Resized, Spacingd

from semantic_constraints.compiled_losses import SemanticConstraintLoss


DEFAULT_CONSTRAINTS = Path(__file__).resolve().parent / "probe_outputs" / "selected_constraints.json"
DEFAULT_OUTPUT_JSON = Path(__file__).resolve().parent / "probe_outputs" / "final_holdout_evaluation.json"
DEFAULT_OUTPUT_MD = Path(__file__).resolve().parent / "probe_outputs" / "final_holdout_evaluation.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on a reserved final holdout fold.")
    parser.add_argument(
        "--model",
        nargs=2,
        action="append",
        metavar=("NAME", "WEIGHTS"),
        required=True,
        help="Named checkpoint to evaluate. Can be repeated.",
    )
    parser.add_argument("--constraints", type=Path, default=DEFAULT_CONSTRAINTS)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--root-dir", type=Path, default=PROPOSAL_ROOT / "tmp")
    parser.add_argument("--task", default="Task04_Hippocampus")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--holdout-fold", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--foreground-classes", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--pixdim", type=float, nargs=3, default=(1.5, 0.5, 1.5))
    parser.add_argument("--spatial-size", type=int, nargs=3, default=(64, 64, 64))
    parser.add_argument("--max-batches", type=int, default=0, help="Limit holdout batches for smoke tests. 0 means all.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
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


def build_holdout_subset(dataset: DecathlonDataset, folds: int, holdout_fold: int) -> torch.utils.data.Subset:
    if holdout_fold < 1 or holdout_fold > folds:
        raise ValueError(f"Holdout fold {holdout_fold} is outside the available range 1..{folds}.")

    sample_count = len(dataset)
    indices = np.arange(sample_count)
    np.random.RandomState(42).shuffle(indices)

    fold_sizes = np.full(folds, sample_count // folds, dtype=int)
    fold_sizes[: sample_count % folds] += 1
    starts = np.cumsum(fold_sizes)
    stop = int(starts[holdout_fold - 1])
    start = int(stop - fold_sizes[holdout_fold - 1])
    return torch.utils.data.Subset(dataset, indices[start:stop])


def build_model(weights: Path, args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    model = SwinUNETR(in_channels=1, out_channels=args.num_classes, use_checkpoint=True).to(device)
    checkpoint = torch.load(weights, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint)
    model.eval()
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


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def accumulate(totals: dict[str, float], key: str, value: float, batch_size: int) -> None:
    totals[key] = totals.get(key, 0.0) + value * batch_size


def evaluate_checkpoint(
    name: str,
    weights: Path,
    loader: DataLoader,
    constraint_loss: SemanticConstraintLoss,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    model = build_model(weights, args, device)
    totals: dict[str, float] = {}
    sample_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            batch_size = images.shape[0]

            for class_id, value in dice_per_class(preds, labels, list(args.foreground_classes)).items():
                accumulate(totals, f"dice_class_{class_id}", scalar(value), batch_size)

            constraint_out = constraint_loss(logits)
            accumulate(totals, "constraint_loss", scalar(constraint_out["loss"]), batch_size)
            for constraint_name, value in constraint_out["truth"].items():
                accumulate(totals, f"constraint_truth/{constraint_name}", scalar(value), batch_size)
            for constraint_name, value in constraint_out["violation"].items():
                accumulate(totals, f"constraint_violation/{constraint_name}", scalar(value), batch_size)
            for constraint_name, value in constraint_out["value"].items():
                accumulate(totals, f"constraint_value/{constraint_name}", scalar(value), batch_size)

            sample_count += batch_size

    if sample_count == 0:
        raise RuntimeError(f"No samples evaluated for model {name}.")

    metrics = {key: value / sample_count for key, value in totals.items()}
    dice_values = [metrics[f"dice_class_{class_id}"] for class_id in args.foreground_classes]
    metrics["foreground_dice"] = sum(dice_values) / len(dice_values)
    return {
        "name": name,
        "weights": str(weights),
        "samples": sample_count,
        "metrics": metrics,
    }


def make_delta(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    names = list(results)
    if len(names) < 2:
        return {}
    first = results[names[0]]["metrics"]
    second = results[names[1]]["metrics"]
    common = sorted(set(first) & set(second))
    return {
        f"{names[1]}_minus_{names[0]}": {
            key: second[key] - first[key]
            for key in common
            if isinstance(first[key], (int, float)) and isinstance(second[key], (int, float))
        }
    }


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


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Final Holdout Evaluation",
        "",
        f"Holdout fold: `{payload['holdout_fold']}` / `{payload['folds']}`",
        f"Samples: `{payload['holdout_samples']}`",
        "",
    ]
    for name, result in payload["models"].items():
        metrics = result["metrics"]
        lines.extend(
            [
                f"## `{name}`",
                "",
                f"- weights: `{result['weights']}`",
                f"- foreground Dice: `{metrics['foreground_dice']:.4f}`",
                f"- constraint loss: `{metrics['constraint_loss']:.4f}`",
            ]
        )
        for key in sorted(metrics):
            if key.startswith("dice_class_"):
                lines.append(f"- {key}: `{metrics[key]:.4f}`")
        lines.append("")

    if payload["delta"]:
        lines.extend(["## Delta", ""])
        for delta_name, values in payload["delta"].items():
            lines.append(f"### `{delta_name}`")
            for key in ("foreground_dice", "constraint_loss"):
                if key in values:
                    lines.append(f"- {key}: `{values[key]:+.4f}`")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dataset = build_dataset(args)
    holdout_subset = build_holdout_subset(dataset, args.folds, args.holdout_fold)
    loader = DataLoader(
        holdout_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    constraint_loss = SemanticConstraintLoss.from_json(
        path=args.constraints,
        spacing=tuple(args.pixdim),
        input_is_logits=True,
    ).to(device)

    results = {}
    for model_name, weights in args.model:
        results[model_name] = evaluate_checkpoint(
            name=model_name,
            weights=Path(weights),
            loader=loader,
            constraint_loss=constraint_loss,
            args=args,
            device=device,
        )

    payload = {
        "schema_version": "semantic_constraints.final_eval.v1",
        "device": str(device),
        "folds": args.folds,
        "holdout_fold": args.holdout_fold,
        "holdout_samples": len(holdout_subset),
        "constraints": str(args.constraints),
        "models": results,
        "delta": make_delta(results),
        "method_note": "This holdout should be excluded during fine-tuning and not used for constraint/model selection.",
    }
    write_json(args.output_json, payload)
    write_markdown(args.output_md, payload)

    print(f"holdout samples: {len(holdout_subset)}")
    for name, result in results.items():
        metrics = result["metrics"]
        print(
            name,
            f"foreground_dice={metrics['foreground_dice']:.4f}",
            f"constraint_loss={metrics['constraint_loss']:.4f}",
        )
    print(f"wrote JSON: {args.output_json}")
    print(f"wrote Markdown: {args.output_md}")


if __name__ == "__main__":
    main()
