#!/usr/bin/env python3
"""Reproduce the camera-ready SwinUNETR/LTN hippocampus experiment.

This runner intentionally follows ``notebooks/segmentation-ltn-camera-ready.ipynb``:
five-fold shuffled KFold (seed 42), the first fraction of every fold's
training indices, 64-cube resampling, AdamW, and a 100-epoch warmup-cosine
schedule. It records the exact manifest and split indices so that results can
be audited even when the Decathlon dataset version differs from the paper.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import json
import platform
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from monai.apps import DecathlonDataset
from monai.data import DataLoader
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.optimizers import WarmupCosineSchedule
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, Resized, Spacingd
from sklearn.model_selection import KFold


PAPER_FRACTIONS = (1.0, 0.25, 0.05)
PAPER_VOLUME_EPSILON = 5000.0
PAPER_VOLUME_GAMMA = 0.0001
VOLUME_GROUNDINGS = ("paper-hard", "soft-probability")


@dataclass(frozen=True)
class RunSpec:
    method: str
    fold: int
    train_fraction: float
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    spacing: tuple[float, float, float]
    spatial_size: tuple[int, int, int]
    volume_epsilon: float
    volume_gamma: float
    volume_grounding: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Parent directory for MONAI Task04_Hippocampus.")
    parser.add_argument("--task", default="Task04_Hippocampus")
    parser.add_argument("--download", action="store_true", help="Download the official MONAI Decathlon task if absent.")
    parser.add_argument("--decathlon-val-frac", type=float, default=0.2,
                        help="DecathlonDataset internal split fraction; 0.2 matches the notebook default.")
    parser.add_argument("--method", choices=("baseline", "ltn"), required=True)
    parser.add_argument("--fold", type=int, required=True, help="One-based KFold index, matching the notebook.")
    parser.add_argument("--train-fraction", type=float, required=True, choices=PAPER_FRACTIONS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--total-scheduler-steps", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42, help="Model/data-loader seed; notebook did not report it.")
    parser.add_argument("--spacing", type=float, nargs=3, default=(1.5, 1.5, 1.5))
    parser.add_argument("--spatial-size", type=int, nargs=3, default=(64, 64, 64))
    parser.add_argument(
        "--volume-epsilon",
        type=float,
        default=PAPER_VOLUME_EPSILON,
        help="Allowed anterior/posterior volume gap in voxels; the paper uses 5000.",
    )
    parser.add_argument(
        "--volume-gamma",
        type=float,
        default=PAPER_VOLUME_GAMMA,
        help="Volume-similarity sharpness; the paper uses 0.0001.",
    )
    parser.add_argument(
        "--volume-grounding",
        choices=VOLUME_GROUNDINGS,
        default="paper-hard",
        help=(
            "paper-hard reproduces the notebook's non-differentiable argmax grounding; "
            "soft-probability uses differentiable expected class volumes."
        ),
    )
    parser.add_argument("--expected-samples", type=int, default=0,
                        help="Fail if Decathlon sample count differs; 0 records the count without asserting it.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint_latest.pt in --output-dir.")
    parser.add_argument("--wandb", action="store_true", help="Log configuration and epoch metrics to Weights & Biases.")
    parser.add_argument("--wandb-project", default="hippopotamus-project")
    parser.add_argument("--wandb-entity", default="focacciafilippo-bocconi-university")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(requested)


def build_transforms(spacing: tuple[float, float, float], spatial_size: tuple[int, int, int]) -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")),
            Resized(keys=["image", "label"], spatial_size=spatial_size, mode=("bilinear", "nearest")),
        ]
    )


def build_dataset(args: argparse.Namespace) -> DecathlonDataset:
    return DecathlonDataset(
        root_dir=str(args.data_root),
        task=args.task,
        section="training",
        download=args.download,
        val_frac=args.decathlon_val_frac,
        transform=build_transforms(tuple(args.spacing), tuple(args.spatial_size)),
    )


def build_fold_indices(sample_count: int, folds: int, split_seed: int, fold: int, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    if fold < 1 or fold > folds:
        raise ValueError(f"--fold must be in 1..{folds}, got {fold}.")
    splitter = KFold(n_splits=folds, shuffle=True, random_state=split_seed)
    splits = list(splitter.split(np.arange(sample_count)))
    train_indices, validation_indices = splits[fold - 1]
    train_limit = int(fraction * len(train_indices))
    if train_limit < 1:
        raise ValueError("The requested training fraction selects no samples.")
    # This intentionally mirrors the notebook's train_idx[:int(fraction * len(train_idx))].
    return train_indices[:train_limit], validation_indices


def one_hot(labels: torch.Tensor, classes: int = 3) -> torch.Tensor:
    if labels.ndim == 5:
        labels = labels.squeeze(1)
    return F.one_hot(labels.long(), num_classes=classes).movedim(-1, 1).float()


def build_model(
    device: torch.device,
    spatial_size: tuple[int, int, int] = (64, 64, 64),
) -> SwinUNETR:
    # feature_size, num_heads and depths are MONAI's paper/notebook defaults.
    kwargs: dict[str, Any] = {
        "in_channels": 1,
        "out_channels": 3,
        "use_checkpoint": True,
    }
    if "img_size" in inspect.signature(SwinUNETR).parameters:
        kwargs["img_size"] = spatial_size
    return SwinUNETR(**kwargs).to(device)


def hard_masks(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=1)


def volume_similarity(
    difference: torch.Tensor,
    gamma: float = PAPER_VOLUME_GAMMA,
    epsilon: float = PAPER_VOLUME_EPSILON,
) -> torch.Tensor:
    """Equation 7 evaluated from an absolute anterior/posterior volume gap."""
    excess = torch.clamp(difference - epsilon, min=0.0)
    return torch.exp(-gamma * excess.square())


def hard_volume_difference(hard: torch.Tensor) -> torch.Tensor:
    """Absolute class-volume gap for integer segmentation masks."""
    spatial_dimensions = tuple(range(1, hard.ndim))
    anterior = (hard == 1).sum(dim=spatial_dimensions)
    posterior = (hard == 2).sum(dim=spatial_dimensions)
    return (anterior - posterior).abs().float()


def paper_dimension(
    hard: torch.Tensor,
    gamma: float = PAPER_VOLUME_GAMMA,
    epsilon: float = PAPER_VOLUME_EPSILON,
) -> torch.Tensor:
    """Notebook's anterior/posterior volume-similarity grounding."""
    return volume_similarity(hard_volume_difference(hard), gamma=gamma, epsilon=epsilon)


def soft_volume_similarity(
    logits: torch.Tensor,
    gamma: float = PAPER_VOLUME_GAMMA,
    epsilon: float = PAPER_VOLUME_EPSILON,
) -> torch.Tensor:
    """Differentiable Equation 7 using expected class volumes from softmax."""
    return volume_similarity(soft_volume_difference(logits), gamma=gamma, epsilon=epsilon)


def soft_volume_difference(logits: torch.Tensor) -> torch.Tensor:
    """Absolute expected anterior/posterior volume gap from softmax probabilities."""
    probabilities = torch.softmax(logits, dim=1)
    # Class is dimension 1, so sum over all spatial dimensions after selecting it.
    spatial_dimensions = tuple(range(1, probabilities[:, 1].ndim))
    anterior = probabilities[:, 1].sum(dim=spatial_dimensions)
    posterior = probabilities[:, 2].sum(dim=spatial_dimensions)
    return (anterior - posterior).abs()


def directional_chamfer_minima(source: torch.Tensor, target: torch.Tensor, chunk_size: int = 1024) -> torch.Tensor:
    """Mean nearest-neighbour distance without materialising an N x M matrix."""
    minima = []
    for start in range(0, source.shape[0], chunk_size):
        distances = torch.cdist(source[start:start + chunk_size], target, p=2)
        minima.append(distances.min(dim=1).values)
    return torch.cat(minima).mean()


def paper_chamfer_distance(hard: torch.Tensor) -> torch.Tensor:
    """Notebook's Chamfer-distance grounding between the two hard masks."""
    values = []
    for index in range(hard.shape[0]):
        first = torch.nonzero(hard[index] == 1, as_tuple=False).float()
        second = torch.nonzero(hard[index] == 2, as_tuple=False).float()
        if first.numel() == 0 or second.numel() == 0:
            values.append(torch.tensor(float("inf"), device=hard.device))
            continue
        values.append(directional_chamfer_minima(first, second) + directional_chamfer_minima(second, first))
    return torch.stack(values)


def paper_nested(hard: torch.Tensor, pairs: int = 20, interpolation_points: int = 50) -> torch.Tensor:
    """Notebook's stochastic nesting/engulfment test for anterior into posterior."""
    values = torch.zeros(hard.shape[0], device=hard.device)
    for batch_index in range(hard.shape[0]):
        first = torch.nonzero(hard[batch_index] == 1, as_tuple=False)
        second = hard[batch_index] == 2
        if first.shape[0] < 2:
            continue
        selected = first[torch.randint(first.shape[0], (pairs, 2), device=hard.device)]
        for source, destination in selected:
            steps = torch.linspace(0, 1, interpolation_points, device=hard.device).unsqueeze(1)
            points = torch.round(source + steps * (destination - source)).long()
            points[:, 0].clamp_(0, hard.shape[1] - 1)
            points[:, 1].clamp_(0, hard.shape[2] - 1)
            points[:, 2].clamp_(0, hard.shape[3] - 1)
            if second[points[:, 0], points[:, 1], points[:, 2]].any():
                values[batch_index] = 1.0
                break
    return values


class PaperLTNObjective:
    """The notebook's LTN knowledge base, retained for protocol fidelity."""

    def __init__(
        self,
        model: SwinUNETR,
        volume_epsilon: float = PAPER_VOLUME_EPSILON,
        volume_gamma: float = PAPER_VOLUME_GAMMA,
        volume_grounding: str = "paper-hard",
    ) -> None:
        try:
            import ltn
        except ImportError as error:
            raise RuntimeError("LTNtorch is required for --method ltn. Install requirements.txt in the cluster environment.") from error
        if volume_grounding not in VOLUME_GROUNDINGS:
            raise ValueError(f"Unknown volume grounding: {volume_grounding}")

        self.ltn = ltn
        self.volume_grounding = volume_grounding
        self.segmentator = ltn.Function(model)
        self.forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
        self.sat_agg = ltn.fuzzy_ops.SatAgg()
        self.not_ = ltn.Connective(ltn.fuzzy_ops.NotStandard())
        dice_loss = DiceLoss(to_onehot_y=True, softmax=True, reduction="none")

        def dice_truth(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            per_class_loss = dice_loss(outputs, labels).reshape(outputs.shape[0], -1)
            return 1.0 - per_class_loss.mean(dim=1)

        def equality(left: torch.Tensor, right: torch.Tensor, alpha: float = 1e-3) -> torch.Tensor:
            return torch.exp(-alpha * torch.sqrt(torch.square(left - right)))

        self.dice_predicate = ltn.Predicate(func=dice_truth)
        self.equal_predicate = ltn.Predicate(func=equality)
        self.min_distance = ltn.Function(func=paper_chamfer_distance)
        volume_function = paper_dimension if volume_grounding == "paper-hard" else soft_volume_similarity

        def configured_volume(value: torch.Tensor) -> torch.Tensor:
            return volume_function(value, gamma=volume_gamma, epsilon=volume_epsilon)

        self.similar_volume = ltn.Function(func=configured_volume)
        self.nested = ltn.Function(func=paper_nested)

    def __call__(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        ltn = self.ltn
        x = ltn.Variable("x", images)
        y = ltn.Variable("y", labels)
        x, y = ltn.diag(x, y)
        outputs = self.segmentator(x)
        logits = outputs.value
        prediction = ltn.Variable("prediction", hard_masks(logits))
        soft_prediction = ltn.Variable("soft_prediction", logits)
        zero = ltn.Constant(torch.zeros((), device=images.device))
        volume_input = prediction if self.volume_grounding == "paper-hard" else soft_prediction
        formula_truths = {
            "dice_truth": self.forall([x, y], self.dice_predicate(outputs, y)).value,
            "connectedness_truth": self.forall(
                prediction, self.equal_predicate(self.min_distance(prediction), zero)
            ).value,
            "volume_truth": self.forall(volume_input, self.similar_volume(volume_input)).value,
            "nesting_truth": self.forall(prediction, self.not_(self.nested(prediction))).value,
        }
        satisfaction = self.sat_agg(*formula_truths.values())
        diagnostics = {
            "constraint_satisfaction": satisfaction,
            **formula_truths,
            "hard_volume_gap_voxels": hard_volume_difference(prediction.value).float().mean().detach(),
            "soft_volume_gap_voxels": soft_volume_difference(logits).mean().detach(),
        }
        return logits, 1.0 - satisfaction, diagnostics


def structural_metrics(
    hard: torch.Tensor,
    volume_epsilon: float = PAPER_VOLUME_EPSILON,
    volume_gamma: float = PAPER_VOLUME_GAMMA,
) -> dict[str, torch.Tensor]:
    distance = paper_chamfer_distance(hard)
    volume_gap = hard_volume_difference(hard)
    return {
        "connectedness": torch.exp(-0.001 * distance.square()),
        "nested": paper_nested(hard),
        "volume_similarity": paper_dimension(hard),
        "volume_similarity_configured": volume_similarity(
            volume_gap, gamma=volume_gamma, epsilon=volume_epsilon
        ),
        "volume_gap_voxels": volume_gap,
        "volume_violation_configured": (volume_gap > volume_epsilon).float(),
    }


def deterministic_structural_metrics(
    hard: torch.Tensor,
    seed: int,
    volume_epsilon: float = PAPER_VOLUME_EPSILON,
    volume_gamma: float = PAPER_VOLUME_GAMMA,
) -> dict[str, torch.Tensor]:
    """Isolate notebook nesting randomness from model-dependent predictions."""
    saved_rng = rng_state()
    try:
        seed_everything(seed)
        return structural_metrics(hard, volume_epsilon=volume_epsilon, volume_gamma=volume_gamma)
    finally:
        restore_rng_state(saved_rng)


def evaluate(
    model: SwinUNETR,
    loader: DataLoader,
    device: torch.device,
    include_structure: bool = False,
    volume_epsilon: float = PAPER_VOLUME_EPSILON,
    volume_gamma: float = PAPER_VOLUME_GAMMA,
) -> dict[str, float]:
    all_class_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=True)
    foreground_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=True)
    classwise_metric = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=True)
    structure_totals: dict[str, float] = {}
    sample_count = 0
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            prediction = hard_masks(model(images))
            prediction_oh = one_hot(prediction)
            labels_oh = one_hot(labels)
            all_class_metric(prediction_oh, labels_oh)
            foreground_metric(prediction_oh, labels_oh)
            classwise_metric(prediction_oh, labels_oh)
            if include_structure:
                for prefix, masks, seed in (
                    ("prediction", prediction, 20250722 + batch_index),
                    ("ground_truth", labels.squeeze(1), 30250722 + batch_index),
                ):
                    for name, values in deterministic_structural_metrics(
                        masks,
                        seed,
                        volume_epsilon=volume_epsilon,
                        volume_gamma=volume_gamma,
                    ).items():
                        key = f"{prefix}_{name}"
                        structure_totals[key] = structure_totals.get(key, 0.0) + float(values.sum().cpu())
                sample_count += images.shape[0]
    all_dice, _ = all_class_metric.aggregate()
    foreground_dice, _ = foreground_metric.aggregate()
    classwise_dice, _ = classwise_metric.aggregate()
    if classwise_dice.numel() != 3:
        raise RuntimeError(f"Expected three Dice classes, received shape {tuple(classwise_dice.shape)}.")
    result = {
        "dice_all_classes": float(all_dice.cpu()),
        "dice_foreground": float(foreground_dice.cpu()),
        "dice_background": float(classwise_dice[0].cpu()),
        "dice_anterior": float(classwise_dice[1].cpu()),
        "dice_posterior": float(classwise_dice[2].cpu()),
    }
    if include_structure:
        result.update({key: value / sample_count for key, value in structure_totals.items()})
    return result


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_manifest(dataset: DecathlonDataset, output_dir: Path, data_root: Path, task: str) -> None:
    records: list[dict[str, str]] = []
    for item in dataset.data:
        record = {"image": str(item.get("image", "")), "label": str(item.get("label", ""))}
        for key in ("image", "label"):
            value = record[key]
            file_path = Path(value)
            digest = hashlib.sha256()
            with file_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            record[f"{key}_sha256"] = digest.hexdigest()
        records.append(record)
    canonical = json.dumps(records, sort_keys=True).encode("utf-8")
    task_json = data_root / task / "dataset.json"
    task_json_hash = hashlib.sha256(task_json.read_bytes()).hexdigest() if task_json.is_file() else None
    write_json(
        output_dir / "dataset_manifest.json",
        {
            "effective_training_samples": len(records),
            "manifest_sha256": hashlib.sha256(canonical).hexdigest(),
            "task_dataset_json_sha256": task_json_hash,
            "records": records,
        },
    )


def source_dataset_count(data_root: Path, task: str) -> int | None:
    dataset_json = data_root / task / "dataset.json"
    if not dataset_json.is_file():
        return None
    return int(json.loads(dataset_json.read_text(encoding="utf-8")).get("numTraining", 0)) or None


def environment_metadata() -> dict[str, Any]:
    packages = {}
    for package in ("torch", "monai", "LTNtorch", "scikit-learn"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).resolve().parents[2]
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    return {
        "python": platform.python_version(),
        "packages": packages,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": git_commit,
    }


def rng_state() -> dict[str, Any]:
    payload: dict[str, Any] = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        payload["cuda"] = torch.cuda.get_rng_state_all()
    return payload


def restore_rng_state(payload: dict[str, Any]) -> None:
    # Checkpoints are loaded with map_location=device.  That also moves the CPU
    # RNG state to CUDA, but torch.set_rng_state requires a CPU ByteTensor.
    # Keep RNG restoration device-independent so an interrupted CUDA run can
    # resume deterministically.
    torch_state = payload["torch"]
    if not isinstance(torch_state, torch.Tensor):
        raise TypeError("Checkpoint CPU RNG state must be a torch.Tensor.")
    torch.set_rng_state(torch_state.detach().to(device="cpu", dtype=torch.uint8))
    np.random.set_state(payload["numpy"])
    random.setstate(payload["python"])
    if "cuda" in payload and torch.cuda.is_available():
        cuda_states = [
            state.detach().to(device="cpu", dtype=torch.uint8)
            if isinstance(state, torch.Tensor)
            else state
            for state in payload["cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)


def save_checkpoint(path: Path, epoch: int, model: SwinUNETR, optimizer: torch.optim.Optimizer, scheduler: Any, best_dice: float) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_dice": best_dice,
            "rng": rng_state(),
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.train_fraction not in PAPER_FRACTIONS:
        raise ValueError(f"Use one of the paper fractions: {PAPER_FRACTIONS}.")
    if args.volume_epsilon < 0:
        raise ValueError("--volume-epsilon must be non-negative.")
    if args.volume_gamma <= 0:
        raise ValueError("--volume-gamma must be positive.")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    output_dir = args.output_dir
    dataset = build_dataset(args)
    if args.expected_samples and len(dataset) != args.expected_samples:
        raise RuntimeError(f"Expected {args.expected_samples} Decathlon samples, found {len(dataset)}.")
    if args.resume:
        if not (output_dir / "checkpoint_latest.pt").is_file():
            raise FileNotFoundError(f"--resume requires {output_dir / 'checkpoint_latest.pt'}.")
    elif output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    else:
        output_dir.mkdir(parents=True)
    train_indices, validation_indices = build_fold_indices(
        len(dataset), args.folds, args.split_seed, args.fold, args.train_fraction
    )
    if not args.resume:
        np.save(output_dir / "train_indices.npy", train_indices)
        np.save(output_dir / "validation_indices.npy", validation_indices)
        write_manifest(dataset, output_dir, args.data_root, args.task)

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    validation_subset = torch.utils.data.Subset(dataset, validation_indices)
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    validation_loader = DataLoader(validation_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    spec = RunSpec(
        method=args.method,
        fold=args.fold,
        train_fraction=args.train_fraction,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        spacing=tuple(args.spacing),
        spatial_size=tuple(args.spatial_size),
        volume_epsilon=args.volume_epsilon,
        volume_gamma=args.volume_gamma,
        volume_grounding=args.volume_grounding,
    )
    if args.resume:
        previous_config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
        current_run = json.loads(json.dumps(asdict(spec)))
        if previous_config.get("run") != current_run:
            raise RuntimeError("Resume configuration does not match the existing run.")
        if not np.array_equal(np.load(output_dir / "train_indices.npy"), train_indices):
            raise RuntimeError("Resume training indices do not match the existing run.")
        if not np.array_equal(np.load(output_dir / "validation_indices.npy"), validation_indices):
            raise RuntimeError("Resume validation indices do not match the existing run.")
    else:
        write_json(
            output_dir / "config.json",
            {
                "run": asdict(spec),
                "args": vars(args) | {"data_root": str(args.data_root), "output_dir": str(args.output_dir)},
                "source_num_training": source_dataset_count(args.data_root, args.task),
                "effective_decathlon_training_samples": len(dataset),
                "train_samples": len(train_subset),
                "validation_samples": len(validation_subset),
                "device": str(device),
            },
        )
        write_json(output_dir / "environment.json", environment_metadata())

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or output_dir.name,
            config={
                "run": asdict(spec),
                "source_num_training": source_dataset_count(args.data_root, args.task),
                "effective_decathlon_training_samples": len(dataset),
                "train_samples": len(train_subset),
                "validation_samples": len(validation_subset),
            },
            resume="allow" if args.resume else None,
        )

    model = build_model(device, tuple(args.spatial_size))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.total_scheduler_steps)
    dice_loss = DiceLoss(to_onehot_y=True, softmax=True)
    ltn_objective = (
        PaperLTNObjective(
            model,
            volume_epsilon=args.volume_epsilon,
            volume_gamma=args.volume_gamma,
            volume_grounding=args.volume_grounding,
        )
        if args.method == "ltn"
        else None
    )
    start_epoch = 1
    best_dice = -float("inf")
    if args.resume:
        checkpoint = torch.load(output_dir / "checkpoint_latest.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        restore_rng_state(checkpoint["rng"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_dice = float(checkpoint["best_dice"])

    metrics_path = output_dir / "metrics.csv"
    mode = "a" if args.resume else "w"
    metric_fields = [
        "epoch",
        "train_loss",
        "learning_rate",
        "train_constraint_satisfaction",
        "train_dice_truth",
        "train_connectedness_truth",
        "train_volume_truth",
        "train_nesting_truth",
        "train_hard_volume_gap_voxels",
        "train_soft_volume_gap_voxels",
        "dice_all_classes",
        "dice_foreground",
        "dice_background",
        "dice_anterior",
        "dice_posterior",
    ]
    with metrics_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields)
        if not args.resume:
            writer.writeheader()
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            total_loss = 0.0
            component_totals: dict[str, float] = {}
            for batch in train_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                optimizer.zero_grad(set_to_none=True)
                if ltn_objective is None:
                    logits = model(images)
                    loss = dice_loss(logits, labels)
                    components = {"dice_truth": 1.0 - loss.detach()}
                else:
                    _, loss, components = ltn_objective(images, labels)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu())
                for name, value in components.items():
                    component_totals[name] = component_totals.get(name, 0.0) + float(value.detach().cpu())
            metrics = evaluate(model, validation_loader, device)
            row = {
                "epoch": epoch,
                "train_loss": total_loss / max(1, len(train_loader)),
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{
                    f"train_{name}": value / max(1, len(train_loader))
                    for name, value in component_totals.items()
                },
                **metrics,
            }
            writer.writerow(row)
            handle.flush()
            print(json.dumps(row), flush=True)
            if wandb_run is not None:
                wandb_run.log(row, step=epoch)
            scheduler.step()
            val_dice = row["dice_all_classes"]
            if val_dice > best_dice:
                best_dice = val_dice
                save_checkpoint(output_dir / "checkpoint_best.pt", epoch, model, optimizer, scheduler, best_dice)
            save_checkpoint(output_dir / "checkpoint_latest.pt", epoch, model, optimizer, scheduler, best_dice)

    torch.save(model.state_dict(), output_dir / "model_final.pt")
    final_metrics = evaluate(
        model,
        validation_loader,
        device,
        include_structure=True,
        volume_epsilon=args.volume_epsilon,
        volume_gamma=args.volume_gamma,
    )
    write_json(output_dir / "final_metrics.json", final_metrics)
    if wandb_run is not None:
        wandb_run.log({f"final/{name}": value for name, value in final_metrics.items()}, step=args.epochs)
        artifact = wandb.Artifact(name=f"paper-ltn-{args.method}-fraction-{args.train_fraction:g}-fold-{args.fold}", type="model")
        artifact.add_file(output_dir / "model_final.pt", name="model_final.pt")
        artifact.add_file(output_dir / "config.json", name="config.json")
        artifact.add_file(output_dir / "final_metrics.json", name="final_metrics.json")
        wandb_run.log_artifact(artifact)
        wandb_run.finish()
    print(json.dumps({"status": "complete", **final_metrics, "output_dir": str(output_dir)}), flush=True)


if __name__ == "__main__":
    main()
