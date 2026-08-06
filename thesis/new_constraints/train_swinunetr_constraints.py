#!/usr/bin/env python3
"""Train SwinUNETR with a selectable auxiliary constraint on an exact CV fold."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from monai.data import DataLoader, Dataset as MonaiDataset
from monai.losses import DiceLoss
from monai.networks.nets import SwinUNETR

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.swin_unetr.swin_unetr import (  # noqa: E402
    _build_monai_dataset_from_pkl,
    _load_pkl_dataframe,
    _load_splits_json,
    build_optimizer_and_scheduler,
)
from thesis.new_constraints import (  # noqa: E402
    NewConstraintConfig,
    NewConstraintObjective,
)
from thesis.new_constraints.constraint_result import ConstraintResult  # noqa: E402

CONSTRAINT_WEIGHTS = {"none": 0.0, "equivariance": 0.10, "translation": 0.10}
CONSTRAINT_CHOICES = ("none", "equivariance", "bands", "translation")
AGREEMENT_CONSTRAINT_NAMES = ("translation_equivariance",)
BAND_METRICS = (
    "raw_loss",
    "inner_loss",
    "outer_loss",
    "inner_voxels",
    "outer_voxels",
    "valid_patient",
    "skipped_patient",
    "edge_touching",
)


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest used to bind runs to exact input contents."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class FileSnapshot:
    """Private read-only copy whose digest describes the bytes later consumed."""

    original_path: Path
    path: Path
    sha256: str
    _temporary_directory: tempfile.TemporaryDirectory[str]

    def cleanup(self) -> None:
        self._temporary_directory.cleanup()


def snapshot_file(path: Path) -> FileSnapshot:
    """Copy a file once while hashing, then consume only the private copy."""

    original_path = path.expanduser().resolve()
    temporary_directory = tempfile.TemporaryDirectory(prefix="hippo-input-")
    snapshot_path = Path(temporary_directory.name) / original_path.name
    digest = hashlib.sha256()
    try:
        with original_path.open("rb") as source, snapshot_path.open("xb") as target:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
        snapshot_path.chmod(0o400)
    except Exception:
        temporary_directory.cleanup()
        raise
    return FileSnapshot(
        original_path=original_path,
        path=snapshot_path,
        sha256=digest.hexdigest(),
        _temporary_directory=temporary_directory,
    )


def collect_source_provenance() -> dict[str, Any]:
    """Describe the local source files that define training and calibration."""

    constraint_root = REPO_ROOT / "thesis" / "new_constraints"
    source_paths = [
        path
        for path in constraint_root.rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".sh"}
        and not path.name.startswith("test_")
        and "__pycache__" not in path.parts
    ]
    source_paths.append(REPO_ROOT / "baselines" / "swin_unetr" / "swin_unetr.py")
    manifest = {
        str(path.relative_to(REPO_ROOT)): file_sha256(path)
        for path in sorted(source_paths)
    }
    aggregate = hashlib.sha256()
    for relative_path, digest in manifest.items():
        aggregate.update(relative_path.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    try:
        git_status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--",
                *manifest.keys(),
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        git_dirty = bool(git_status.strip())
    except (OSError, subprocess.CalledProcessError):
        git_dirty = None
    return {
        "sha256": aggregate.hexdigest(),
        "files": manifest,
        "git_commit": commit,
        "git_dirty": git_dirty,
    }


def patient_id_from_case(case_name: str, dataset: str) -> str:
    """Return the patient group encoded in a dataset case name."""

    dataset = dataset.upper()
    if dataset == "MSD":
        return case_name
    patterns = {
        "MNI": r"s\d+",
        "ADNI": r"adni_\d+",
        "COBRA": r"cobra_\d+",
    }
    if dataset not in patterns:
        raise ValueError(f"Unknown dataset: {dataset}")
    match = re.search(patterns[dataset], case_name, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Could not derive a patient ID from case name: {case_name}")
    return match.group().lower()


def build_swinunetr(
    spatial_size: tuple[int, int, int],
    num_classes: int,
    device: torch.device,
) -> SwinUNETR:
    """Build SwinUNETR across MONAI versions with and without ``img_size``."""
    kwargs: dict[str, Any] = {
        "in_channels": 1,
        "out_channels": num_classes,
        "use_checkpoint": True,
    }
    if "img_size" in inspect.signature(SwinUNETR).parameters:
        kwargs["img_size"] = spatial_size
    return SwinUNETR(**kwargs).to(device)


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    fold: int
    seed: int
    translation_seed: int
    epochs: int
    batch_size: int
    spatial_size: tuple[int, int, int]
    resize: bool
    optimizer_mode: str
    learning_rate: float | None
    weight_decay: float | None
    step_size: int
    adamw_gamma: float
    constraint_set: str
    constraint_config: dict[str, Any]
    constraint_warmup_epochs: int
    constraint_eval_every: int
    amp: bool
    initial_checkpoint: str | None
    initial_checkpoint_sha256: str | None = None
    pkl_sha256: str = ""
    splits_json_sha256: str = ""
    source_sha256: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", type=Path, required=True)
    parser.add_argument("--dataset", choices=("MSD", "MNI", "ADNI", "COBRA"), default="MSD")
    parser.add_argument("--splits-json", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--spatial-size", type=int, nargs=3, default=(64, 64, 64))
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--optim-mode", choices=("adamw_0.01", "nnunetv2"), default="adamw_0.01")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--step-size", type=int, default=20)
    parser.add_argument("--adamw-gamma", type=float, default=0.5)

    parser.add_argument(
        "--constraint-set",
        choices=CONSTRAINT_CHOICES,
        default="equivariance",
        help="Constraint preset. 'translation' is a deprecated alias for 'equivariance'.",
    )
    parser.add_argument("--equivariance-weight", type=float, default=None)
    parser.add_argument("--translation-size", type=int, default=2)
    parser.add_argument("--equivariance-max-samples", type=int, default=1,
                        help="Maximum transformed samples per training batch; 0 uses the whole batch.")
    parser.add_argument(
        "--bands-weight",
        type=float,
        default=None,
        help="Positive calibrated weight required by --constraint-set bands.",
    )
    parser.add_argument(
        "--band-steps",
        type=int,
        default=2,
        help="Fixed at 2 for the canonical bands experiment.",
    )
    parser.add_argument("--foreground-class-ids", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--complement-class-ids", type=int, nargs="+", default=(0,))
    parser.add_argument("--constraint-warmup-epochs", type=int, default=5)
    parser.add_argument("--constraint-eval-every", type=int, default=5,
                        help="Evaluate validation constraint metrics every N epochs; 0 means final only.")

    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path, default=None,
                        help="Optional baseline model.pt used only to initialize a new run.")
    parser.add_argument("--wandb", action="store_true")
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


def build_experiment_generators(
    seed: int,
) -> tuple[torch.Generator, torch.Generator, int]:
    """Create independent streams for data order and translation sampling."""

    translation_seed = seed + 1
    data_generator = torch.Generator()
    data_generator.manual_seed(seed)
    translation_generator = torch.Generator()
    translation_generator.manual_seed(translation_seed)
    return data_generator, translation_generator, translation_seed


def rng_state(
    data_generator: torch.Generator,
    translation_generator: torch.Generator,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "data_generator": data_generator.get_state(),
        "translation_generator": translation_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(
    state: dict[str, Any],
    data_generator: torch.Generator,
    translation_generator: torch.Generator,
) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    data_generator.set_state(state["data_generator"])
    translation_generator.set_state(state["translation_generator"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(requested)


def resolve_constraint_config(args: argparse.Namespace) -> NewConstraintConfig:
    selected = "equivariance" if args.constraint_set == "translation" else args.constraint_set
    equivariance_override = getattr(args, "equivariance_weight", None)
    bands_override = getattr(args, "bands_weight", None)
    if selected == "none":
        if equivariance_override not in (None, 0.0) or bands_override not in (None, 0.0):
            raise ValueError(
                "--constraint-set none cannot be combined with a non-zero constraint weight."
            )
        equivariance_weight = 0.0
        bands_weight = 0.0
    elif selected == "equivariance":
        if bands_override not in (None, 0.0):
            raise ValueError(
                "--constraint-set equivariance cannot be combined with --bands-weight."
            )
        equivariance_weight = (
            CONSTRAINT_WEIGHTS["equivariance"]
            if equivariance_override is None
            else equivariance_override
        )
        if not math.isfinite(equivariance_weight) or equivariance_weight <= 0:
            raise ValueError(
                "--constraint-set equivariance requires a finite positive "
                "--equivariance-weight."
            )
        bands_weight = 0.0
    elif selected == "bands":
        if equivariance_override not in (None, 0.0):
            raise ValueError(
                "--constraint-set bands cannot be combined with --equivariance-weight."
            )
        if bands_override is None:
            raise ValueError(
                "--constraint-set bands requires --bands-weight from training-only "
                "gradient calibration."
            )
        bands_weight = bands_override
        if not math.isfinite(bands_weight) or bands_weight <= 0:
            raise ValueError(
                "--constraint-set bands requires a finite positive --bands-weight."
            )
        if getattr(args, "band_steps", 2) != 2:
            raise ValueError(
                "The canonical bands experiment requires exactly --band-steps 2."
            )
        equivariance_weight = 0.0
    else:
        raise ValueError(f"Unknown constraint set: {args.constraint_set}")
    return NewConstraintConfig(
        equivariance_weight=equivariance_weight,
        translation_size=getattr(args, "translation_size", 2),
        equivariance_max_samples=(
            None
            if getattr(args, "equivariance_max_samples", None) == 0
            else getattr(args, "equivariance_max_samples", None)
        ),
        bands_weight=bands_weight,
        band_steps=getattr(args, "band_steps", 2),
        foreground_class_ids=tuple(getattr(args, "foreground_class_ids", (1, 2))),
        complement_class_ids=tuple(getattr(args, "complement_class_ids", (0,))),
    )


def constraint_warmup_scale(epoch: int, warmup_epochs: int) -> float:
    """Return the multiplier applied to the already weighted constraint loss."""

    if epoch < 1:
        raise ValueError("epoch must be positive.")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative.")
    if warmup_epochs == 0:
        return 1.0
    return min(1.0, epoch / warmup_epochs)


def validate_three_class_labels(items: list[dict[str, Any]]) -> None:
    """Reject labels that cannot be interpreted exactly as classes 0, 1, and 2."""

    allowed = {0, 1, 2}
    for item in items:
        case_name = str(item.get("case_name", "<unknown>"))
        labels = np.asarray(item["label"])
        if labels.size == 0:
            raise ValueError(f"Case {case_name} has an empty label array.")
        if not np.issubdtype(labels.dtype, np.number):
            raise ValueError(f"Case {case_name} has non-numeric labels.")
        try:
            finite = bool(np.isfinite(labels).all())
        except TypeError as error:
            raise ValueError(f"Case {case_name} has invalid labels.") from error
        if not finite:
            raise ValueError(f"Case {case_name} has nonfinite labels.")
        if np.iscomplexobj(labels) or not np.equal(labels, np.rint(labels)).all():
            raise ValueError(f"Case {case_name} has non-integer labels.")
        values = {int(value) for value in np.unique(labels)}
        invalid = sorted(values - allowed)
        if invalid:
            preview = ", ".join(str(value) for value in invalid[:5])
            raise ValueError(
                f"Case {case_name} contains labels outside {{0,1,2}}: {preview}"
            )


def build_data(
    args: argparse.Namespace,
    train_generator: torch.Generator,
    *,
    pkl_path: Path | None = None,
    splits_path: Path | None = None,
) -> tuple[DataLoader, DataLoader, int, int, int]:
    dataframe = _load_pkl_dataframe(pkl_path or args.pkl)
    if args.num_classes not in (None, 3):
        raise ValueError(
            f"The new constraints expect exactly three classes, found {args.num_classes}."
        )
    num_classes = 3
    dataset = _build_monai_dataset_from_pkl(
        dataframe,
        args.dataset,
        num_classes,
        spatial_size=tuple(args.spatial_size),
        do_resize=args.resize,
    )
    splits = _load_splits_json(splits_path or args.splits_json)
    if args.fold < 0 or args.fold >= len(splits):
        raise ValueError(f"--fold must be in 0..{len(splits) - 1}.")
    split = splits[args.fold]
    train_list = list(split["train"])
    validation_list = list(split["val"])
    if len(train_list) != len(set(train_list)) or len(validation_list) != len(
        set(validation_list)
    ):
        raise ValueError("Split contains duplicate case names.")
    train_names = set(train_list)
    validation_names = set(validation_list)
    if train_names & validation_names:
        raise ValueError("Training and validation case names overlap.")
    train_patients = {
        patient_id_from_case(case_name, args.dataset) for case_name in train_names
    }
    validation_patients = {
        patient_id_from_case(case_name, args.dataset) for case_name in validation_names
    }
    patient_overlap = train_patients & validation_patients
    if patient_overlap:
        preview = ", ".join(sorted(patient_overlap)[:5])
        raise ValueError(
            f"Training and validation patient IDs overlap: {preview}"
        )

    available_names = {item["case_name"] for item in dataset.data}
    missing = (train_names | validation_names) - available_names
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise ValueError(f"The split references {len(missing)} missing cases: {preview}")

    train_items = [item for item in dataset.data if item["case_name"] in train_names]
    validation_items = [
        item for item in dataset.data if item["case_name"] in validation_names
    ]
    if len(train_items) != len(train_names) or len(validation_items) != len(validation_names):
        raise ValueError("Duplicate case names in the dataframe make the split ambiguous.")
    validate_three_class_labels(train_items + validation_items)

    train_dataset = MonaiDataset(data=train_items, transform=dataset.transform)
    validation_dataset = MonaiDataset(
        data=validation_items,
        transform=dataset.transform,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=train_generator,
        **loader_options,
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
    return (
        train_loader,
        validation_loader,
        num_classes,
        len(train_dataset),
        len(validation_dataset),
    )


def load_initial_weights(model: SwinUNETR, checkpoint_path: Path) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    model.load_state_dict(state_dict, strict=True)


def update_constraint_totals(
    totals: dict[str, dict[str, Any]],
    results: dict[str, ConstraintResult],
) -> None:
    for name, result in results.items():
        truth = result.truth.detach().float()
        confidence_weighted_agreement = (
            result.details["confidence_weighted_agreement"].detach().float()
        )
        confidence_adherent = result.details["confidence_adherent"].detach().float()
        entry = totals.setdefault(
            name,
            {
                "truth": 0.0,
                "confidence_weighted_agreement": 0.0,
                "confidence_adherent": 0.0,
                "count": 0.0,
                "metric_sums": {},
                "metric_counts": {},
            },
        )
        entry["truth"] += float(truth.sum().cpu())
        entry["confidence_weighted_agreement"] += float(
            confidence_weighted_agreement.sum().cpu()
        )
        entry["confidence_adherent"] += float(confidence_adherent.sum().cpu())
        entry["count"] += float(truth.numel())
        for metric_name, metric_value in result.details.get("metrics", {}).items():
            values = torch.as_tensor(metric_value).detach().float()
            entry["metric_sums"][metric_name] = entry["metric_sums"].get(
                metric_name, 0.0
            ) + float(values.sum().cpu())
            entry["metric_counts"][metric_name] = entry["metric_counts"].get(
                metric_name, 0.0
            ) + float(values.numel())


def averaged_constraint_metrics(
    totals: dict[str, dict[str, Any]],
    prefix: str,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, entry in totals.items():
        if name in AGREEMENT_CONSTRAINT_NAMES:
            count = max(entry["count"], 1.0)
            metrics[f"{prefix}_{name}_truth"] = entry["truth"] / count
            metrics[f"{prefix}_{name}_confidence_weighted_agreement"] = (
                entry["confidence_weighted_agreement"] / count
            )
            metrics[f"{prefix}_{name}_confidence_adherent"] = (
                entry["confidence_adherent"] / count
            )
        for metric_name, total in entry.get("metric_sums", {}).items():
            metric_count = max(entry["metric_counts"][metric_name], 1.0)
            metrics[f"{prefix}_{name}_{metric_name}"] = total / metric_count
    return metrics


def _update_validation_constraint_batch(
    model: nn.Module,
    objective: NewConstraintObjective,
    batch_index: int,
    images: torch.Tensor,
    labels: torch.Tensor,
    logits: torch.Tensor,
    case_names: list[str],
    totals: dict[str, dict[str, Any]],
    *,
    all_translation_shifts: bool,
    detail_rows: list[dict[str, Any]] | None,
) -> None:
    """Accumulate active constraints from an already computed base prediction."""

    if objective.config.bands_weight > 0:
        band_result = objective.bands(logits, labels)
        update_constraint_totals(totals, {"outer_boundary_band": band_result})
        if detail_rows is not None:
            details = band_result.details
            valid = details["valid"].detach().cpu()
            for sample_index, case_name in enumerate(case_names):
                is_valid = bool(valid[sample_index])
                detail_rows.append(
                    {
                        "constraint_name": "outer_boundary_band",
                        "case_name": case_name,
                        "band_valid": int(is_valid),
                        "band_loss": (
                            float(details["case_loss"][sample_index].detach().cpu())
                            if is_valid
                            else None
                        ),
                        "inner_loss": (
                            float(details["inner_loss"][sample_index].detach().cpu())
                            if is_valid
                            else None
                        ),
                        "outer_loss": (
                            float(details["outer_loss"][sample_index].detach().cpu())
                            if is_valid
                            else None
                        ),
                        "inner_voxels": int(
                            details["inner_voxels"][sample_index].detach().cpu()
                        ),
                        "outer_voxels": int(
                            details["outer_voxels"][sample_index].detach().cpu()
                        ),
                        "edge_touching": int(
                            details["edge_touching"][sample_index].detach().cpu()
                        ),
                    }
                )

    if objective.config.equivariance_weight <= 0:
        return
    shifts = objective.equivariance.shifts
    selected_shifts = (
        shifts if all_translation_shifts else (shifts[batch_index % len(shifts)],)
    )
    for shift in selected_shifts:
        translation_result = objective.equivariance(
            model,
            images,
            logits,
            shift=shift,
        )
        update_constraint_totals(
            totals,
            {"translation_equivariance": translation_result},
        )
        if detail_rows is None:
            continue
        optimization_values = translation_result.details[
            "optimization_classwise_dice"
        ].detach().float().cpu()
        linear_values = translation_result.value.detach().float().cpu()
        case_truth = translation_result.truth.detach().float().cpu()
        case_confidence_weighted_agreement = translation_result.details[
            "confidence_weighted_agreement"
        ].detach().float().cpu()
        case_confidence_adherent = translation_result.details[
            "confidence_adherent"
        ].detach().cpu()
        dx, dy, dz = shift
        for sample_index, case_name in enumerate(case_names):
            for class_index, class_id in enumerate(objective.equivariance.class_ids):
                linear_value = float(linear_values[sample_index, class_index])
                detail_rows.append(
                    {
                        "constraint_name": "translation_equivariance",
                        "case_name": case_name,
                        "shift_dx": dx,
                        "shift_dy": dy,
                        "shift_dz": dz,
                        "class_id": class_id,
                        "optimization_truth": float(
                            optimization_values[sample_index, class_index]
                        ),
                        "legacy_linear_value": linear_value,
                        "class_confidence_adherent": int(
                            linear_value
                            >= objective.equivariance.confidence_agreement_threshold
                        ),
                        "case_direction_truth": float(case_truth[sample_index]),
                        "case_direction_confidence_weighted_agreement": float(
                            case_confidence_weighted_agreement[sample_index]
                        ),
                        "case_direction_confidence_adherent": int(
                            case_confidence_adherent[sample_index]
                        ),
                    }
                )


@torch.no_grad()
def evaluate_constraint_metrics(
    model: SwinUNETR,
    loader: DataLoader,
    objective: NewConstraintObjective,
    device: torch.device,
    *,
    amp: bool,
    all_translation_shifts: bool,
    detail_rows: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    """Standalone constraint diagnostics; training uses the combined evaluator."""

    if (
        objective.config.equivariance_weight == 0
        and objective.config.bands_weight == 0
    ):
        return {}
    totals: dict[str, dict[str, Any]] = {}
    model.eval()
    for batch_index, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        case_names = [str(name) for name in batch["case_name"]]
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(images)
            _update_validation_constraint_batch(
                model,
                objective,
                batch_index,
                images,
                labels,
                logits,
                case_names,
                totals,
                all_translation_shifts=all_translation_shifts,
                detail_rows=detail_rows,
            )
    return averaged_constraint_metrics(totals, "val")


def _segmentation_dice_sums(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> tuple[float, float, int]:
    """Return foreground Dice sums and their valid sample/class count."""

    if num_classes < 2:
        raise ValueError("Segmentation Dice requires at least two classes.")
    if labels.ndim == logits.ndim and labels.shape[1] == 1:
        labels = labels.squeeze(1)
    probabilities = torch.softmax(logits.float(), dim=1)
    hard_labels = torch.argmax(probabilities, dim=1)
    targets = torch.nn.functional.one_hot(
        labels.long(),
        num_classes=num_classes,
    ).movedim(-1, 1).float()
    hard_predictions = torch.nn.functional.one_hot(
        hard_labels,
        num_classes=num_classes,
    ).movedim(-1, 1).float()
    spatial_dimensions = tuple(range(2, probabilities.ndim))
    foreground_targets = targets[:, 1:]
    foreground_probabilities = probabilities[:, 1:]
    foreground_hard = hard_predictions[:, 1:]
    soft_intersection = (
        foreground_probabilities * foreground_targets
    ).sum(spatial_dimensions)
    hard_intersection = (foreground_hard * foreground_targets).sum(
        spatial_dimensions
    )
    target_volume = foreground_targets.sum(spatial_dimensions)
    soft_denominator = foreground_probabilities.sum(spatial_dimensions) + target_volume
    hard_denominator = foreground_hard.sum(spatial_dimensions) + target_volume
    # Match MONAI's default ignore-empty convention: a class with no
    # ground-truth voxels does not enter the sample/class macro mean.
    valid_scores = target_volume > 0
    soft_scores = 2.0 * soft_intersection[valid_scores] / soft_denominator[
        valid_scores
    ]
    hard_scores = 2.0 * hard_intersection[valid_scores] / hard_denominator[
        valid_scores
    ]
    return (
        float(soft_scores.sum().cpu()),
        float(hard_scores.sum().cpu()),
        int(valid_scores.sum().item()),
    )


@torch.no_grad()
def evaluate_segmentation_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    num_classes: int,
) -> tuple[float, float]:
    """Return valid macro soft and hard Dice over foreground classes."""

    soft_total = 0.0
    hard_total = 0.0
    count = 0
    model.eval()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        batch_soft, batch_hard, batch_count = _segmentation_dice_sums(
            model(images), labels, num_classes
        )
        soft_total += batch_soft
        hard_total += batch_hard
        count += batch_count
    if count == 0:
        raise ValueError("The validation loader produced no foreground scores.")
    return soft_total / count, hard_total / count


@torch.no_grad()
def evaluate_validation_metrics(
    model: nn.Module,
    loader: DataLoader,
    objective: NewConstraintObjective,
    device: torch.device,
    *,
    num_classes: int,
    amp: bool,
    evaluate_constraints: bool,
    all_translation_shifts: bool,
    detail_rows: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    """Evaluate segmentation and active constraints from one base forward per batch."""

    soft_total = 0.0
    hard_total = 0.0
    count = 0
    constraint_totals: dict[str, dict[str, Any]] = {}
    has_active_constraints = (
        objective.config.equivariance_weight > 0 or objective.config.bands_weight > 0
    )
    model.eval()
    for batch_index, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        case_names = (
            [str(name) for name in batch["case_name"]]
            if evaluate_constraints and has_active_constraints
            else []
        )
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(images)
            if evaluate_constraints and has_active_constraints:
                _update_validation_constraint_batch(
                    model,
                    objective,
                    batch_index,
                    images,
                    labels,
                    logits,
                    case_names,
                    constraint_totals,
                    all_translation_shifts=all_translation_shifts,
                    detail_rows=detail_rows,
                )
        batch_soft, batch_hard, batch_count = _segmentation_dice_sums(
            logits, labels, num_classes
        )
        soft_total += batch_soft
        hard_total += batch_hard
        count += batch_count
    if count == 0:
        raise ValueError("The validation loader produced no foreground scores.")
    return {
        "val_dice_soft": soft_total / count,
        "val_dice_hard": hard_total / count,
        **averaged_constraint_metrics(constraint_totals, "val"),
    }


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: SwinUNETR,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.cuda.amp.GradScaler,
    best_hard_dice: float,
    best_epoch: int,
    data_generator: torch.Generator,
    translation_generator: torch.Generator,
    run_spec: RunSpec,
) -> None:
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": (
            scheduler.state_dict()
            if callable(getattr(scheduler, "state_dict", None))
            else None
        ),
        "scaler": scaler.state_dict(),
        "best_hard_dice": best_hard_dice,
        "best_epoch": best_epoch,
        "run": asdict(run_spec),
        "rng": rng_state(data_generator, translation_generator),
    }
    save_checkpoint_payload(path, payload)


def save_checkpoint_payload(path: Path, payload: dict[str, Any]) -> None:
    """Atomically publish a complete checkpoint payload."""

    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def metric_fieldnames() -> list[str]:
    fields = [
        "epoch",
        "train_loss",
        "train_supervised_loss",
        "train_constraint_loss",
        "constraint_scale",
        "learning_rate",
        "val_dice_soft",
        "val_dice_hard",
    ]
    for prefix in ("train", "val"):
        for name in AGREEMENT_CONSTRAINT_NAMES:
            for statistic in (
                "truth",
                "confidence_weighted_agreement",
                "confidence_adherent",
            ):
                fields.append(f"{prefix}_{name}_{statistic}")
        for statistic in BAND_METRICS:
            fields.append(f"{prefix}_outer_boundary_band_{statistic}")
    return fields


def resolve_resume_run_spec(run_spec: RunSpec, saved_run: dict[str, Any]) -> RunSpec:
    """Restore immutable initialization provenance before strict resume checks."""

    saved_initial_path = saved_run.get("initial_checkpoint")
    saved_initial_digest = saved_run.get("initial_checkpoint_sha256")
    if saved_initial_path is None:
        if saved_initial_digest is not None:
            raise ValueError("Resume initializer digest exists without a checkpoint path.")
    elif (
        not isinstance(saved_initial_path, str)
        or not saved_initial_path
        or not isinstance(saved_initial_digest, str)
        or len(saved_initial_digest) != 64
    ):
        raise ValueError("Resume initializer lacks immutable SHA-256 provenance.")
    current_run = json.loads(json.dumps(asdict(run_spec)))
    current_run["initial_checkpoint"] = saved_initial_path
    current_run["initial_checkpoint_sha256"] = saved_initial_digest
    if saved_run != current_run:
        raise ValueError("Resume configuration does not match config.json.")
    return replace(
        run_spec,
        initial_checkpoint=saved_initial_path,
        initial_checkpoint_sha256=saved_initial_digest,
    )


def validate_input_file_provenance(
    config: dict[str, Any],
    pkl_path: Path,
    splits_path: Path,
    *,
    pkl_digest: str | None = None,
    splits_digest: str | None = None,
) -> None:
    """Require both input paths and contents to match the recorded run."""

    expected = (
        ("pkl", "pkl_sha256", pkl_path, pkl_digest),
        ("splits_json", "splits_json_sha256", splits_path, splits_digest),
    )
    for path_key, digest_key, current_path, current_digest in expected:
        recorded_path = config.get(path_key)
        if not isinstance(recorded_path, str) or (
            Path(recorded_path).expanduser().resolve() != current_path.resolve()
        ):
            raise ValueError(f"Resume {path_key} path does not match config.json.")
        recorded_digest = config.get(digest_key)
        if not isinstance(recorded_digest, str) or len(recorded_digest) != 64:
            raise ValueError(
                f"config.json lacks valid {digest_key} provenance; start a fresh run."
            )
        actual_digest = current_digest or file_sha256(current_path)
        if actual_digest != recorded_digest:
            raise ValueError(f"Resume {path_key} contents changed after run creation.")


def validate_resume_checkpoint(
    checkpoint: dict[str, Any],
    run_spec: RunSpec,
) -> None:
    """Reject checkpoints that do not belong to the exact resumed run."""

    if not isinstance(checkpoint, dict):
        raise ValueError("Resume checkpoint payload is not a dictionary.")
    required = {
        "epoch",
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "best_hard_dice",
        "best_epoch",
        "rng",
        "run",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"Resume checkpoint is missing fields: {', '.join(missing)}")
    expected_run = json.loads(json.dumps(asdict(run_spec)))
    checkpoint_run = json.loads(json.dumps(checkpoint["run"]))
    if checkpoint_run != expected_run:
        raise ValueError("Resume checkpoint run provenance does not match config.json.")
    try:
        epoch = int(checkpoint["epoch"])
        best_epoch = int(checkpoint["best_epoch"])
        best_hard_dice = float(checkpoint["best_hard_dice"])
    except (TypeError, ValueError) as error:
        raise ValueError("Resume checkpoint chronology is malformed.") from error
    if epoch < 1 or epoch > run_spec.epochs:
        raise ValueError("Resume checkpoint epoch is outside the configured run.")
    if best_epoch < 0 or best_epoch > epoch:
        raise ValueError("Resume checkpoint best_epoch is inconsistent with epoch.")
    if not math.isfinite(best_hard_dice) or not 0.0 <= best_hard_dice <= 1.0:
        raise ValueError("Resume checkpoint best_hard_dice must be finite in [0, 1].")
    for field in ("model", "optimizer", "scaler", "rng"):
        if not isinstance(checkpoint[field], dict):
            raise ValueError(f"Resume checkpoint {field} state is malformed.")
    scheduler_state = checkpoint["scheduler"]
    if run_spec.optimizer_mode == "adamw_0.01":
        if not isinstance(scheduler_state, dict) or not scheduler_state:
            raise ValueError("AdamW resume requires a saved StepLR scheduler state.")
    elif scheduler_state is not None:
        raise ValueError("nnU-Net resume expects no serialized scheduler state.")


def reconcile_best_checkpoint(
    output_dir: Path,
    latest_checkpoint: dict[str, Any],
    run_spec: RunSpec,
) -> None:
    """Repair a best checkpoint interrupted after the durable latest save."""

    latest_epoch = int(latest_checkpoint["epoch"])
    expected_best_epoch = int(latest_checkpoint["best_epoch"])
    best_path = output_dir / "checkpoint_best.pt"
    if expected_best_epoch == latest_epoch:
        save_checkpoint_payload(best_path, latest_checkpoint)
        return
    if not best_path.is_file():
        raise FileNotFoundError("Resume run is missing checkpoint_best.pt.")
    best_snapshot = snapshot_file(best_path)
    best_checkpoint = torch.load(
        best_snapshot.path,
        map_location="cpu",
        weights_only=False,
    )
    best_snapshot.cleanup()
    validate_resume_checkpoint(best_checkpoint, run_spec)
    if int(best_checkpoint["epoch"]) != expected_best_epoch:
        raise ValueError("checkpoint_best.pt does not match the latest best_epoch.")


def reconcile_metrics_for_resume(
    path: Path,
    checkpoint_epoch: int,
    fieldnames: list[str],
) -> None:
    """Remove rows written after the last durable checkpoint before appending."""

    if not path.is_file():
        raise FileNotFoundError("--resume requires metrics.csv.")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fieldnames:
            raise ValueError("metrics.csv header does not match the current pipeline.")
        retained: list[dict[str, str]] = []
        seen_epochs: set[int] = set()
        for row in reader:
            try:
                epoch = int(row["epoch"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("metrics.csv contains an invalid epoch.") from error
            if epoch > checkpoint_epoch:
                continue
            if epoch in seen_epochs:
                raise ValueError(f"metrics.csv contains duplicate epoch {epoch}.")
            seen_epochs.add(epoch)
            retained.append(row)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(retained)
    temporary.replace(path)


def validate_optimizer_hyperparameters(
    *,
    step_size: int,
    learning_rate: float,
    weight_decay: float,
    adamw_gamma: float,
) -> None:
    """Reject optimizer values that would crash or poison a run after it starts."""

    if step_size <= 0:
        raise ValueError("--step-size must be positive.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("--learning-rate must be finite and positive.")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("--weight-decay must be finite and non-negative.")
    if not math.isfinite(adamw_gamma) or not 0 < adamw_gamma <= 1:
        raise ValueError("--adamw-gamma must be finite and in (0, 1].")


def validate_epoch_metrics(row: dict[str, Any]) -> None:
    """Stop before logging or checkpointing a numerically poisoned epoch."""

    for name, value in row.items():
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            raise FloatingPointError(f"Epoch metric {name} is not finite.")


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive.")
    if args.constraint_warmup_epochs < 0 or args.constraint_eval_every < 0:
        raise ValueError("Constraint warmup/evaluation intervals must be non-negative.")
    if args.equivariance_max_samples < 0:
        raise ValueError("--equivariance-max-samples must be non-negative.")
    if args.band_steps < 1:
        raise ValueError("--band-steps must be positive.")
    if args.resume and args.init_checkpoint is not None:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive.")
    if not args.pkl.is_file() or not args.splits_json.is_file():
        raise FileNotFoundError("The dataset pickle and split JSON must both exist.")
    if args.init_checkpoint is not None and not args.init_checkpoint.is_file():
        raise FileNotFoundError(args.init_checkpoint)
    validate_optimizer_hyperparameters(
        step_size=args.step_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adamw_gamma=args.adamw_gamma,
    )

    source_provenance = collect_source_provenance()
    pkl_snapshot = snapshot_file(args.pkl)
    splits_snapshot = snapshot_file(args.splits_json)
    initial_snapshot = (
        snapshot_file(args.init_checkpoint)
        if args.init_checkpoint is not None
        else None
    )
    pkl_digest = pkl_snapshot.sha256
    splits_digest = splits_snapshot.sha256

    seed_everything(args.seed)
    data_generator, translation_generator, translation_seed = (
        build_experiment_generators(args.seed)
    )
    device = resolve_device(args.device)
    if args.amp and device.type != "cuda":
        raise ValueError("--amp requires CUDA.")
    config = resolve_constraint_config(args)
    # Constructing the objective also validates all constraint hyperparameters.
    objective = NewConstraintObjective(config).to(device)
    evaluation_objective = NewConstraintObjective(
        replace(config, equivariance_max_samples=None)
    ).to(device)

    run_spec = RunSpec(
        dataset=args.dataset,
        fold=args.fold,
        seed=args.seed,
        translation_seed=translation_seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        spatial_size=tuple(args.spatial_size),
        resize=args.resize,
        optimizer_mode=args.optim_mode,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        step_size=args.step_size,
        adamw_gamma=args.adamw_gamma,
        constraint_set=(
            "equivariance" if args.constraint_set == "translation" else args.constraint_set
        ),
        constraint_config=asdict(config),
        constraint_warmup_epochs=args.constraint_warmup_epochs,
        constraint_eval_every=args.constraint_eval_every,
        amp=args.amp,
        initial_checkpoint=(
            str(initial_snapshot.original_path) if initial_snapshot is not None else None
        ),
        initial_checkpoint_sha256=(
            initial_snapshot.sha256 if initial_snapshot is not None else None
        ),
        pkl_sha256=pkl_digest,
        splits_json_sha256=splits_digest,
        source_sha256=source_provenance["sha256"],
    )

    output_dir = args.output_dir.resolve()
    previous: dict[str, Any] | None = None
    recover_incomplete_run = False
    if args.resume:
        if not (output_dir / "checkpoint_latest.pt").is_file():
            raise FileNotFoundError("--resume requires checkpoint_latest.pt.")
        if not (output_dir / "config.json").is_file():
            raise FileNotFoundError("--resume requires config.json.")
        previous = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
        validate_input_file_provenance(
            previous,
            args.pkl,
            args.splits_json,
            pkl_digest=pkl_digest,
            splits_digest=splits_digest,
        )
        run_spec = resolve_resume_run_spec(run_spec, previous["run"])
    elif output_dir.exists():
        config_path = output_dir / "config.json"
        if (
            not config_path.is_file()
            or (output_dir / "checkpoint_latest.pt").exists()
            or (output_dir / "checkpoint_best.pt").exists()
        ):
            raise FileExistsError(f"Refusing to overwrite existing directory: {output_dir}")
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        validate_input_file_provenance(
            previous,
            args.pkl,
            args.splits_json,
            pkl_digest=pkl_digest,
            splits_digest=splits_digest,
        )
        if previous.get("run") != json.loads(json.dumps(asdict(run_spec))):
            raise ValueError("Incomplete run configuration does not match this restart.")
        recover_incomplete_run = True

    if previous is not None:
        wandb_config = previous.get("wandb")
        if not isinstance(wandb_config, dict):
            raise ValueError("Existing run lacks persisted W&B provenance.")
        if bool(wandb_config.get("enabled")) != bool(args.wandb):
            raise ValueError("W&B enablement must match the existing run.")
        if args.wandb and (
            wandb_config.get("project") != args.wandb_project
            or wandb_config.get("entity") != args.wandb_entity
            or not isinstance(wandb_config.get("run_id"), str)
            or not wandb_config["run_id"]
        ):
            raise ValueError("Existing run has incompatible W&B provenance.")
    else:
        wandb_config = {
            "enabled": bool(args.wandb),
            "project": args.wandb_project if args.wandb else None,
            "entity": args.wandb_entity if args.wandb else None,
            "run_id": uuid.uuid4().hex if args.wandb else None,
        }

    train_loader, validation_loader, num_classes, train_count, validation_count = (
        build_data(
            args,
            data_generator,
            pkl_path=pkl_snapshot.path,
            splits_path=splits_snapshot.path,
        )
    )
    pkl_snapshot.cleanup()
    splits_snapshot.cleanup()

    model = build_swinunetr(run_spec.spatial_size, num_classes, device)
    if initial_snapshot is not None:
        load_initial_weights(model, initial_snapshot.path)
        initial_snapshot.cleanup()
    optimizer, scheduler = build_optimizer_and_scheduler(
        args.optim_mode,
        model,
        args.epochs,
        adamw_gamma=args.adamw_gamma,
        step_size=args.step_size if args.optim_mode == "adamw_0.01" else None,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    supervised_loss_function = DiceLoss(to_onehot_y=True, softmax=True)

    config_payload = {
        "run": asdict(run_spec),
        "train_samples": train_count,
        "validation_samples": validation_count,
        "pkl": str(args.pkl.resolve()),
        "pkl_sha256": pkl_digest,
        "splits_json": str(args.splits_json.resolve()),
        "splits_json_sha256": splits_digest,
        "source_provenance": source_provenance,
        "wandb": wandb_config,
        "device": str(device),
    }
    if not args.resume:
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        canonical_config_payload = json.loads(json.dumps(config_payload))
        if recover_incomplete_run and previous != canonical_config_payload:
            raise ValueError("Incomplete config.json changed before restart.")
        save_json(output_dir / "config.json", config_payload)

    start_epoch = 1
    best_hard_dice = -float("inf")
    best_epoch = 0
    if args.resume:
        checkpoint_snapshot = snapshot_file(output_dir / "checkpoint_latest.pt")
        checkpoint = torch.load(
            checkpoint_snapshot.path,
            # RNG states are CPU byte tensors; loading the full payload onto
            # CUDA makes torch.set_rng_state fail during resume.
            map_location="cpu",
            weights_only=False,
        )
        checkpoint_snapshot.cleanup()
        validate_resume_checkpoint(checkpoint, run_spec)
        reconcile_best_checkpoint(output_dir, checkpoint, run_spec)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint["scheduler"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng_state(
            checkpoint["rng"],
            data_generator,
            translation_generator,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        best_hard_dice = float(checkpoint["best_hard_dice"])
        best_epoch = int(checkpoint["best_epoch"])
        reconcile_metrics_for_resume(
            output_dir / "metrics.csv",
            int(checkpoint["epoch"]),
            metric_fieldnames(),
        )

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or output_dir.name,
            id=wandb_config["run_id"],
            config=asdict(run_spec),
            resume=(
                "must"
                if args.resume
                else "allow" if recover_incomplete_run else "never"
            ),
        )

    metrics_path = output_dir / "metrics.csv"
    metrics_mode = "a" if args.resume else "w"
    with metrics_path.open(metrics_mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fieldnames())
        if not args.resume:
            writer.writeheader()

        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            loss_totals = {"total": 0.0, "supervised": 0.0, "constraint": 0.0}
            constraint_totals: dict[str, dict[str, Any]] = {}
            constraint_scale = constraint_warmup_scale(
                epoch,
                args.constraint_warmup_epochs,
            )

            for batch in train_loader:
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=args.amp):
                    logits = model(images)
                    supervised_loss = supervised_loss_function(logits, labels)
                    constraint_output = objective(
                        model,
                        images,
                        logits,
                        labels,
                        generator=translation_generator,
                    )
                    constraint_loss = constraint_output["loss"]
                    loss = supervised_loss + constraint_scale * constraint_loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                loss_totals["total"] += float(loss.detach().cpu())
                loss_totals["supervised"] += float(supervised_loss.detach().cpu())
                loss_totals["constraint"] += float(constraint_loss.detach().cpu())
                update_constraint_totals(
                    constraint_totals,
                    constraint_output["results"],
                )

            batches = max(1, len(train_loader))
            learning_rate = optimizer.param_groups[0]["lr"]
            should_evaluate_constraints = (
                args.constraint_eval_every > 0
                and (
                    epoch % args.constraint_eval_every == 0
                    or epoch == 1
                    or epoch == args.epochs
                )
            )
            validation_metrics = evaluate_validation_metrics(
                model,
                validation_loader,
                evaluation_objective,
                device,
                num_classes=num_classes,
                amp=args.amp,
                evaluate_constraints=should_evaluate_constraints,
                all_translation_shifts=False,
            )
            soft_dice = validation_metrics["val_dice_soft"]
            hard_dice = validation_metrics["val_dice_hard"]
            row: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": loss_totals["total"] / batches,
                "train_supervised_loss": loss_totals["supervised"] / batches,
                "train_constraint_loss": loss_totals["constraint"] / batches,
                "constraint_scale": constraint_scale,
                "learning_rate": learning_rate,
                "val_dice_soft": soft_dice,
                "val_dice_hard": hard_dice,
                **averaged_constraint_metrics(constraint_totals, "train"),
                **{
                    key: value
                    for key, value in validation_metrics.items()
                    if key not in {"val_dice_soft", "val_dice_hard"}
                },
            }
            validate_epoch_metrics(row)
            writer.writerow(row)
            handle.flush()
            printable = {key: value for key, value in row.items() if value != ""}
            print(json.dumps(printable), flush=True)
            if wandb_run is not None:
                wandb_run.log(printable, step=epoch)

            if args.optim_mode == "nnunetv2":
                scheduler.step(epoch)
            else:
                scheduler.step()
            improved = hard_dice > best_hard_dice
            if improved:
                best_hard_dice = hard_dice
                best_epoch = epoch
            save_checkpoint(
                output_dir / "checkpoint_latest.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_hard_dice=best_hard_dice,
                best_epoch=best_epoch,
                data_generator=data_generator,
                translation_generator=translation_generator,
                run_spec=run_spec,
            )
            if improved:
                save_checkpoint(
                    output_dir / "checkpoint_best.pt",
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    best_hard_dice=best_hard_dice,
                    best_epoch=best_epoch,
                    data_generator=data_generator,
                    translation_generator=translation_generator,
                    run_spec=run_spec,
                )

    model_dir = output_dir / f"{args.dataset}_fold{args.fold}"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "model.pt")
    final_constraint_details: list[dict[str, Any]] = []
    final_validation_metrics = evaluate_validation_metrics(
        model,
        validation_loader,
        evaluation_objective,
        device,
        num_classes=num_classes,
        amp=args.amp,
        evaluate_constraints=True,
        all_translation_shifts=True,
        detail_rows=final_constraint_details,
    )
    final_metrics = {
        "epoch": args.epochs,
        "checkpoint": "final",
        "best_epoch": best_epoch,
        "best_val_dice_hard": best_hard_dice,
        **final_validation_metrics,
    }
    save_json(output_dir / "final_metrics.json", final_metrics)
    detail_path = output_dir / "validation_constraint_details.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as detail_handle:
        detail_writer = csv.DictWriter(
            detail_handle,
            fieldnames=[
                "constraint_name",
                "case_name",
                "shift_dx",
                "shift_dy",
                "shift_dz",
                "class_id",
                "optimization_truth",
                "legacy_linear_value",
                "class_confidence_adherent",
                "case_direction_truth",
                "case_direction_confidence_weighted_agreement",
                "case_direction_confidence_adherent",
                "band_valid",
                "band_loss",
                "inner_loss",
                "outer_loss",
                "inner_voxels",
                "outer_voxels",
                "edge_touching",
            ],
        )
        detail_writer.writeheader()
        detail_writer.writerows(final_constraint_details)
    if wandb_run is not None:
        wandb_run.log(
            {f"final/{name}": value for name, value in final_metrics.items()},
            step=args.epochs,
        )
        artifact = wandb.Artifact(
            name=f"swinunetr-new-constraints-{args.constraint_set}-{args.dataset}-fold{args.fold}",
            type="model",
        )
        artifact.add_file(str(model_dir / "model.pt"), name="model.pt")
        artifact.add_file(str(output_dir / "config.json"), name="config.json")
        artifact.add_file(
            str(output_dir / "final_metrics.json"),
            name="final_metrics.json",
        )
        artifact.add_file(
            str(detail_path),
            name="validation_constraint_details.csv",
        )
        wandb_run.log_artifact(artifact)
        wandb_run.finish()
    print(
        json.dumps(
            {"status": "complete", "output_dir": str(output_dir), **final_metrics}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
