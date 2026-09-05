#!/usr/bin/env python3
"""Calibrate the canonical outer-band weight from training-only gradients."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SOURCE_PROVENANCE_ENV = "HIPPO_EXPECTED_SOURCE_SHA256"
if __name__ == "__main__" and SOURCE_PROVENANCE_ENV not in os.environ:
    os.execve(
        sys.executable,
        [
            sys.executable,
            str(REPO_ROOT / "thesis" / "new_constraints" / "source_bootstrap.py"),
            str(Path(__file__).resolve()),
            *sys.argv[1:],
        ],
        os.environ.copy(),
    )

import numpy as np  # noqa: E402
import torch  # noqa: E402
from monai.data import DataLoader, Dataset as MonaiDataset  # noqa: E402
from monai.losses import DiceLoss  # noqa: E402

from baselines.swin_unetr.swin_unetr import (  # noqa: E402
    _build_monai_dataset_from_pkl,
    _load_pkl_dataframe,
    _load_splits_json,
)
from thesis.new_constraints.bands import (  # noqa: E402
    ClassAwareBoundaryTverskyLoss,
    OuterBoundaryBandLoss,
    build_boundary_bands,
)
from thesis.new_constraints.train_swinunetr_constraints import (  # noqa: E402
    build_swinunetr,
    canonical_sha256,
    collect_execution_provenance,
    collect_runtime_provenance,
    collect_source_provenance,
    patient_id_from_case,
    resolve_device,
    save_json,
    seed_everything,
    snapshot_file,
    validate_image_label_samples,
    validate_source_provenance_unchanged,
)

MAX_CALIBRATION_CASES = 32
CANONICAL_BAND_STEPS = 2
CALIBRATION_CHECKPOINT_EPOCH = 5
TARGET_GRADIENT_RATIO = 0.10
SAFETY_GRADIENT_RATIO = 0.50
EXPECTED_NUM_CLASSES = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", type=Path, required=True)
    parser.add_argument("--dataset", choices=("MSD", "MNI", "ADNI", "COBRA"), default="MSD")
    parser.add_argument("--splits-json", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-config",
        type=Path,
        default=None,
        help="Run config.json; defaults to the checkpoint directory/config.json.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=MAX_CALIBRATION_CASES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spatial-size", type=int, nargs=3, default=(64, 64, 64))
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--band-steps", type=int, default=CANONICAL_BAND_STEPS)
    parser.add_argument(
        "--bands-focal-gamma",
        type=float,
        default=0.0,
        help="Focal exponent for the calibrated band loss; 0 preserves BCE.",
    )
    parser.add_argument(
        "--bands-inner-focal-gamma",
        type=float,
        default=None,
        help="Optional inner-band exponent; defaults to --bands-focal-gamma.",
    )
    parser.add_argument(
        "--bands-outer-focal-gamma",
        type=float,
        default=None,
        help="Optional outer-band exponent; defaults to --bands-focal-gamma.",
    )
    parser.add_argument(
        "--bands-loss-type",
        choices=("focal_bce", "class_tversky"),
        default="focal_bce",
    )
    parser.add_argument("--tversky-fp-weight", type=float, default=0.60)
    parser.add_argument("--tversky-fn-weight", type=float, default=0.40)
    parser.add_argument("--foreground-class-ids", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--complement-class-ids", type=int, nargs="+", default=(0,))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match AMP rounding used by the source/bands run (default: enabled).",
    )
    return parser.parse_args()


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "nonfinite_count": 0,
            "mean": None,
            "median": None,
            "q25": None,
            "q75": None,
            "q95": None,
            "minimum": None,
            "maximum": None,
        }
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "count": int(array.size),
            "nonfinite_count": int(array.size),
            "mean": None,
            "median": None,
            "q25": None,
            "q75": None,
            "q95": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "count": int(array.size),
        "nonfinite_count": int(array.size - finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "q25": float(np.quantile(finite, 0.25)),
        "q75": float(np.quantile(finite, 0.75)),
        "q95": float(np.quantile(finite, 0.95)),
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
    }


def effective_focal_gammas(args: argparse.Namespace) -> tuple[float, float, float]:
    shared = float(getattr(args, "bands_focal_gamma", 0.0))
    inner_value = getattr(args, "bands_inner_focal_gamma", None)
    outer_value = getattr(args, "bands_outer_focal_gamma", None)
    inner = shared if inner_value is None else float(inner_value)
    outer = shared if outer_value is None else float(outer_value)
    for name, gamma in (
        ("--bands-focal-gamma", shared),
        ("--bands-inner-focal-gamma", inner),
        ("--bands-outer-focal-gamma", outer),
    ):
        if not math.isfinite(gamma) or gamma < 0:
            raise ValueError(f"{name} must be finite and non-negative.")
    return shared, inner, outer


def calibrated_weight(
    dice_rms: list[float],
    band_rms: list[float],
    *,
    target_ratio: float = TARGET_GRADIENT_RATIO,
    safety_ratio: float = SAFETY_GRADIENT_RATIO,
) -> tuple[float, float, float]:
    """Return final, target, and capped weights from robust RMS statistics."""

    if not dice_rms or len(dice_rms) != len(band_rms):
        raise ValueError("dice_rms and band_rms must be nonempty and equally sized.")
    if (
        not math.isfinite(target_ratio)
        or not math.isfinite(safety_ratio)
        or target_ratio <= 0
        or safety_ratio <= 0
    ):
        raise ValueError("target_ratio and safety_ratio must be finite and positive.")
    dice_array = np.asarray(dice_rms, dtype=np.float64)
    band_array = np.asarray(band_rms, dtype=np.float64)
    if not np.isfinite(dice_array).all() or not np.isfinite(band_array).all():
        raise ValueError("Calibration gradients contain NaN or infinity.")
    dice_median = float(np.median(dice_array))
    band_median = float(np.median(band_array))
    band_q95 = float(np.quantile(band_array, 0.95))
    if dice_median <= 0 or band_median <= 0 or band_q95 <= 0:
        raise ValueError("Calibration gradients must have positive robust magnitudes.")
    target = target_ratio * dice_median / band_median
    cap = safety_ratio * dice_median / band_q95
    return min(target, cap), target, cap


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def validate_calibration_protocol(max_cases: int, band_steps: int) -> None:
    if not 1 <= max_cases <= MAX_CALIBRATION_CASES:
        raise ValueError(f"--max-cases must be in 1..{MAX_CALIBRATION_CASES}.")
    if band_steps != CANONICAL_BAND_STEPS:
        raise ValueError(
            f"The canonical bands experiment requires --band-steps {CANONICAL_BAND_STEPS}."
        )


def validate_split_integrity(
    train_list: list[str],
    validation_list: list[str],
    dataset: str,
) -> tuple[set[str], set[str]]:
    """Reject duplicate, case-overlapping, and patient-overlapping splits."""

    if len(train_list) != len(set(train_list)) or len(validation_list) != len(
        set(validation_list)
    ):
        raise ValueError("Split contains duplicate case names.")
    train_names = set(train_list)
    validation_names = set(validation_list)
    if train_names & validation_names:
        raise ValueError("Training and validation case names overlap.")
    train_patients = {patient_id_from_case(name, dataset) for name in train_names}
    validation_patients = {
        patient_id_from_case(name, dataset) for name in validation_names
    }
    patient_overlap = train_patients & validation_patients
    if patient_overlap:
        preview = ", ".join(sorted(patient_overlap)[:5])
        raise ValueError(f"Training and validation patient IDs overlap: {preview}")
    return train_names, validation_names


def _resolved_from_json(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Calibration run config contains a missing path.")
    return Path(value).expanduser().resolve()


def _exact_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Calibration {field} must be an exact integer.")
    return value


def validate_checkpoint_provenance(
    args: argparse.Namespace,
    *,
    pkl_digest: str,
    splits_digest: str,
    checkpoint_path: Path,
    source_provenance: dict[str, Any],
    runtime_provenance: dict[str, Any] | None = None,
    execution_provenance: dict[str, Any] | None = None,
    config_read_path: Path | None = None,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Require an epoch-five checkpoint from the matching Dice-only run."""

    if args.checkpoint.name != "checkpoint_latest.pt":
        raise ValueError("Calibration requires checkpoint_latest.pt from the source run.")
    config_path = (
        args.checkpoint_config
        if args.checkpoint_config is not None
        else args.checkpoint.parent / "config.json"
    )
    if not config_path.is_file():
        raise ValueError(
            "Calibration requires the source run config.json; pass --checkpoint-config."
        )
    if config_path.parent.resolve() != args.checkpoint.parent.resolve():
        raise ValueError("Checkpoint and config.json must come from the same run directory.")
    try:
        run_config = json.loads(
            (config_read_path or config_path).read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"Could not read checkpoint provenance: {error}") from error
    run = run_config.get("run")
    if not isinstance(run, dict):
        raise ValueError("Checkpoint config does not contain a run specification.")
    if run.get("constraint_set") != "none":
        raise ValueError("Calibration checkpoint must come from --constraint-set none.")
    if run.get("supervised_loss", "dice") != "dice" or run.get("ce_weight", 0.0) != 0.0:
        raise ValueError("Legacy calibration requires a Dice-only supervised checkpoint.")
    if _exact_int(run.get("epochs"), "run epochs") != CALIBRATION_CHECKPOINT_EPOCH:
        raise ValueError(
            f"Calibration source must be a dedicated {CALIBRATION_CHECKPOINT_EPOCH}-epoch "
            "none run."
        )
    if "initial_checkpoint" not in run or run["initial_checkpoint"] is not None:
        raise ValueError(
            "Calibration source must be trained from scratch without an initial "
            "checkpoint."
        )
    constraint_config = run.get("constraint_config", {})
    if not isinstance(constraint_config, dict):
        raise ValueError("Checkpoint constraint configuration is malformed.")
    if any(
        float(constraint_config.get(name, 0.0)) != 0.0
        for name in ("equivariance_weight", "bands_weight", "onecut_weight", "teacher_weight", "ap_cut_weight")
    ):
        raise ValueError("Calibration checkpoint is not a Dice-only run.")
    if run.get("dataset") != args.dataset or _exact_int(
        run.get("fold"), "run fold"
    ) != args.fold:
        raise ValueError("Checkpoint dataset/fold does not match calibration arguments.")
    if tuple(run.get("spatial_size", ())) != tuple(args.spatial_size):
        raise ValueError("Checkpoint spatial size does not match calibration arguments.")
    if bool(run.get("resize", False)) != bool(args.resize):
        raise ValueError("Checkpoint resize mode does not match calibration arguments.")
    if bool(run.get("amp")) != bool(args.amp):
        raise ValueError("Checkpoint AMP mode does not match calibration arguments.")
    if _resolved_from_json(run_config.get("pkl")) != args.pkl.resolve():
        raise ValueError("Checkpoint dataset pickle does not match calibration input.")
    if _resolved_from_json(run_config.get("splits_json")) != args.splits_json.resolve():
        raise ValueError("Checkpoint split file does not match calibration input.")
    expected_pkl_digest = run_config.get("pkl_sha256")
    expected_splits_digest = run_config.get("splits_json_sha256")
    if not isinstance(expected_pkl_digest, str) or len(expected_pkl_digest) != 64:
        raise ValueError("Checkpoint config lacks dataset pickle content provenance.")
    if not isinstance(expected_splits_digest, str) or len(expected_splits_digest) != 64:
        raise ValueError("Checkpoint config lacks split-file content provenance.")
    if run.get("pkl_sha256") != expected_pkl_digest or run.get(
        "splits_json_sha256"
    ) != expected_splits_digest:
        raise ValueError("Checkpoint run metadata does not bind the recorded input hashes.")
    if pkl_digest != expected_pkl_digest:
        raise ValueError("Dataset pickle contents changed after checkpoint creation.")
    if splits_digest != expected_splits_digest:
        raise ValueError("Split-file contents changed after checkpoint creation.")
    recorded_source = run_config.get("source_provenance")
    checkpoint_source_sha256 = run.get("source_sha256")
    if (
        not isinstance(recorded_source, dict)
        or recorded_source.get("sha256") != checkpoint_source_sha256
        or not isinstance(recorded_source.get("files"), dict)
    ):
        raise ValueError("Checkpoint run metadata does not bind source provenance.")
    _, inner_gamma, outer_gamma = effective_focal_gammas(args)
    loss_type = str(getattr(args, "bands_loss_type", "focal_bce"))
    if loss_type == "focal_bce" and inner_gamma == 0.0 and outer_gamma == 0.0:
        if checkpoint_source_sha256 != source_provenance.get("sha256"):
            raise ValueError("Calibration source code differs from the checkpoint run.")
    else:
        model_source = "baselines/swin_unetr/swin_unetr.py"
        if recorded_source["files"].get(model_source) != source_provenance.get(
            "files", {}
        ).get(model_source):
            raise ValueError(
                "Focal calibration requires the checkpoint's exact model/data-pipeline source."
            )
    current_runtime = runtime_provenance or collect_runtime_provenance()
    recorded_runtime = run_config.get("runtime_provenance")
    if not isinstance(recorded_runtime, dict) or canonical_sha256(
        recorded_runtime
    ) != canonical_sha256(current_runtime):
        raise ValueError("Calibration runtime differs from the checkpoint run.")
    if run.get("runtime_sha256") != canonical_sha256(current_runtime):
        raise ValueError("Checkpoint run metadata does not bind runtime provenance.")
    current_execution = execution_provenance
    recorded_execution = run_config.get("execution_provenance")
    if current_execution is None:
        raise ValueError("Calibration execution provenance was not provided.")
    if not isinstance(recorded_execution, dict) or canonical_sha256(
        recorded_execution
    ) != canonical_sha256(current_execution):
        raise ValueError("Calibration compute device differs from the checkpoint run.")
    if run.get("execution_sha256") != canonical_sha256(current_execution):
        raise ValueError("Checkpoint run metadata does not bind execution provenance.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError("Calibration requires a full training checkpoint with model state.")
    checkpoint_epoch = _exact_int(checkpoint.get("epoch"), "checkpoint epoch")
    if checkpoint_epoch != CALIBRATION_CHECKPOINT_EPOCH:
        raise ValueError(
            f"Calibration requires the Dice-only epoch-{CALIBRATION_CHECKPOINT_EPOCH} "
            "checkpoint_latest.pt."
        )
    embedded_run = checkpoint.get("run")
    if embedded_run is None:
        raise ValueError(
            "Calibration checkpoint lacks embedded run provenance; create a fresh "
            "five-epoch none checkpoint with the current pipeline."
        )
    canonical_embedded_run = (
        json.loads(json.dumps(embedded_run)) if embedded_run is not None else None
    )
    if canonical_embedded_run is not None and canonical_embedded_run != run:
        raise ValueError("Checkpoint-embedded run metadata disagrees with config.json.")
    slim_checkpoint = {
        "epoch": checkpoint_epoch,
        "model": checkpoint["model"],
        "run": embedded_run,
    }
    del checkpoint
    return slim_checkpoint, config_path.resolve(), run_config


def _build_training_loader(
    args: argparse.Namespace,
    *,
    pkl_path: Path | None = None,
    splits_path: Path | None = None,
) -> tuple[DataLoader, list[dict[str, str]], int]:
    dataframe = _load_pkl_dataframe(pkl_path or args.pkl)
    num_classes = EXPECTED_NUM_CLASSES
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
    train_list = list(splits[args.fold]["train"])
    validation_list = list(splits[args.fold]["val"])
    train_names, _ = validate_split_integrity(
        train_list,
        validation_list,
        args.dataset,
    )

    available_names = [str(item["case_name"]) for item in dataset.data]
    if len(available_names) != len(set(available_names)):
        raise ValueError("Dataset contains duplicate case names.")
    available = {str(item["case_name"]): item for item in dataset.data}
    missing = train_names - set(available)
    if missing:
        raise ValueError(f"Training split contains {len(missing)} unavailable cases.")
    if not train_names:
        raise ValueError("Training split is empty.")

    rng = random.Random(args.seed)
    selected_names = sorted(
        rng.sample(sorted(train_names), min(args.max_cases, len(train_names)))
    )
    selected_cases = [
        {
            "case_name": name,
            "patient_id": patient_id_from_case(name, args.dataset),
        }
        for name in selected_names
    ]
    selected_items = [available[item["case_name"]] for item in selected_cases]
    validate_image_label_samples(selected_items)
    loader = DataLoader(
        MonaiDataset(data=selected_items, transform=dataset.transform),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    return loader, selected_cases, num_classes


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, allow_nan=False)
    save_json(path, payload)
    print(encoded)


def compute_logit_gradients(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    dice_loss: DiceLoss,
    band_loss: torch.nn.Module,
    *,
    amp: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
    """Measure logit gradients without retaining a model-forward graph."""

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp):
        model_logits = model(images)
    logits = model_logits.detach().float().requires_grad_(True)
    supervised = dice_loss(logits, labels)
    result = band_loss(logits, labels)
    dice_gradient = torch.autograd.grad(supervised, logits)[0].float()
    band_gradient = torch.autograd.grad(result.loss, logits)[0].float()
    return logits, dice_gradient, band_gradient, result


def finalize_calibration_report(
    output: Path,
    payload: dict[str, Any],
    dice_rms: list[float],
    band_rms: list[float],
) -> dict[str, Any]:
    """Always persist calibration diagnostics, including degenerate failures."""

    try:
        final_weight, target_weight, cap_weight = calibrated_weight(
            dice_rms,
            band_rms,
        )
    except ValueError as error:
        payload.update(
            {
                "status": "failed",
                "failure_reason": str(error),
                "target_weight": None,
                "cap_weight": None,
                "recommended_bands_weight": None,
            }
        )
        _write_report(output, payload)
        raise RuntimeError(
            f"Calibration failed; diagnostic report written to {output}."
        ) from error
    payload.update(
        {
            "target_weight": target_weight,
            "cap_weight": cap_weight,
            "recommended_bands_weight": final_weight,
        }
    )
    _write_report(output, payload)
    return payload


def main() -> None:
    args = parse_args()
    validate_calibration_protocol(args.max_cases, args.band_steps)
    shared_gamma, inner_gamma, outer_gamma = effective_focal_gammas(args)
    if args.bands_loss_type == "class_tversky" and (
        inner_gamma != 0.0 or outer_gamma != 0.0
    ):
        raise ValueError("Class-aware Tversky calibration cannot use focal exponents.")
    if not math.isclose(
        args.tversky_fp_weight + args.tversky_fn_weight,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or args.tversky_fp_weight <= 0 or args.tversky_fn_weight <= 0:
        raise ValueError("Tversky FP and FN weights must be positive and sum to 1.")
    if not args.pkl.is_file() or not args.splits_json.is_file():
        raise FileNotFoundError("The dataset pickle and split JSON must exist.")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    source_provenance = collect_source_provenance()
    runtime_provenance = collect_runtime_provenance()
    config_input_path = (
        args.checkpoint_config
        if args.checkpoint_config is not None
        else args.checkpoint.parent / "config.json"
    )
    with snapshot_file(args.pkl) as pkl_snapshot, snapshot_file(
        args.splits_json
    ) as splits_snapshot, snapshot_file(args.checkpoint) as checkpoint_snapshot, snapshot_file(
        config_input_path
    ) as config_snapshot:
        seed_everything(args.seed)
        device = resolve_device(args.device)
        execution_provenance = collect_execution_provenance(device)
        if args.amp and device.type != "cuda":
            raise ValueError("Calibration --amp requires CUDA; use --no-amp on CPU.")
        checkpoint, config_path, source_config = validate_checkpoint_provenance(
            args,
            pkl_digest=pkl_snapshot.sha256,
            splits_digest=splits_snapshot.sha256,
            checkpoint_path=checkpoint_snapshot.path,
            source_provenance=source_provenance,
            runtime_provenance=runtime_provenance,
            execution_provenance=execution_provenance,
            config_read_path=config_snapshot.path,
        )
        checkpoint_digest = checkpoint_snapshot.sha256
        checkpoint_config_digest = config_snapshot.sha256
        loader, selected_cases, num_classes = _build_training_loader(
            args,
            pkl_path=pkl_snapshot.path,
            splits_path=splits_snapshot.path,
        )
        pkl_digest = pkl_snapshot.sha256
        splits_digest = splits_snapshot.sha256
    validate_source_provenance_unchanged(source_provenance)
    model = build_swinunetr(tuple(args.spatial_size), num_classes, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    dice_loss = DiceLoss(to_onehot_y=True, softmax=True)
    if args.bands_loss_type == "class_tversky":
        band_loss = ClassAwareBoundaryTverskyLoss(
            foreground_class_ids=args.foreground_class_ids,
            complement_class_ids=args.complement_class_ids,
            steps=CANONICAL_BAND_STEPS,
            false_positive_weight=args.tversky_fp_weight,
            false_negative_weight=args.tversky_fn_weight,
        ).to(device)
    else:
        band_loss = OuterBoundaryBandLoss(
            foreground_class_ids=args.foreground_class_ids,
            complement_class_ids=args.complement_class_ids,
            steps=CANONICAL_BAND_STEPS,
            focal_gamma=shared_gamma,
            inner_focal_gamma=inner_gamma,
            outer_focal_gamma=outer_gamma,
        ).to(device)

    records: list[dict[str, Any]] = []
    dice_rms: list[float] = []
    band_rms: list[float] = []
    conditional_band_rms: list[float] = []
    for selected, batch in zip(selected_cases, loader):
        actual_case_name = str(batch["case_name"][0])
        if actual_case_name != selected["case_name"]:
            raise RuntimeError("Calibration loader case order changed unexpectedly.")
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        logits, dice_gradient, band_gradient, result = compute_logit_gradients(
            model,
            images,
            labels,
            dice_loss,
            band_loss,
            amp=args.amp,
        )
        current_dice_rms = float(dice_gradient.square().mean().sqrt().cpu())
        current_band_rms = float(band_gradient.square().mean().sqrt().cpu())

        label_map = labels[:, 0] if labels.ndim == logits.ndim else labels
        if args.bands_loss_type == "class_tversky":
            active_spatial = torch.zeros_like(label_map, dtype=torch.bool)
            for class_id in args.foreground_class_ids:
                inner, outer = build_boundary_bands(
                    (label_map.long() == class_id).unsqueeze(1),
                    steps=CANONICAL_BAND_STEPS,
                )
                active_spatial |= inner[:, 0] | outer[:, 0]
        else:
            foreground = torch.zeros_like(label_map, dtype=torch.bool)
            for class_id in args.foreground_class_ids:
                foreground |= label_map.long() == class_id
            inner, outer = build_boundary_bands(
                foreground.unsqueeze(1),
                steps=CANONICAL_BAND_STEPS,
            )
            active_spatial = inner[:, 0] | outer[:, 0]
        active = active_spatial.unsqueeze(1).expand(-1, logits.shape[1], -1, -1, -1)
        conditional = (
            float(band_gradient[active].square().mean().sqrt().cpu())
            if bool(active.any())
            else 0.0
        )
        valid = bool(result.details["valid"].item())
        if valid:
            dice_rms.append(current_dice_rms)
            band_rms.append(current_band_rms)
            conditional_band_rms.append(conditional)
        records.append(
            {
                **selected,
                "valid": valid,
                "dice_gradient_rms": _finite_or_none(current_dice_rms),
                "band_gradient_rms_unconditional": _finite_or_none(
                    current_band_rms
                ),
                "band_gradient_rms_conditional": _finite_or_none(conditional),
                "inner_voxels": int(result.details["inner_voxels"].item()),
                "outer_voxels": int(result.details["outer_voxels"].item()),
                "edge_touching": bool(result.details["edge_touching"].item()),
            }
        )

    base_payload: dict[str, Any] = {
        "constraint_set": "bands",
        "status": "complete",
        "fold": args.fold,
        "dataset": args.dataset,
        "seed": args.seed,
        "max_cases": args.max_cases,
        "amp": args.amp,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_config": str(config_path),
        "checkpoint_config_sha256": checkpoint_config_digest,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_constraint_set": source_config["run"]["constraint_set"],
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_source_provenance": source_config["source_provenance"],
        "checkpoint_source_sha256": source_config["run"]["source_sha256"],
        "pkl": str(args.pkl.resolve()),
        "pkl_sha256": pkl_digest,
        "splits_json": str(args.splits_json.resolve()),
        "splits_json_sha256": splits_digest,
        "source_provenance": source_provenance,
        "runtime_provenance": runtime_provenance,
        "execution_provenance": execution_provenance,
        "training_cases": selected_cases,
        "training_case_ids": [item["case_name"] for item in selected_cases],
        "training_patient_ids": sorted(
            {item["patient_id"] for item in selected_cases}
        ),
        "band_steps": CANONICAL_BAND_STEPS,
        "bands_loss_type": args.bands_loss_type,
        "tversky_false_positive_weight": args.tversky_fp_weight,
        "tversky_false_negative_weight": args.tversky_fn_weight,
        "bands_focal_gamma": shared_gamma,
        "bands_inner_focal_gamma": inner_gamma,
        "bands_outer_focal_gamma": outer_gamma,
        "foreground_class_ids": list(args.foreground_class_ids),
        "complement_class_ids": list(args.complement_class_ids),
        "target_ratio": TARGET_GRADIENT_RATIO,
        "safety_ratio": SAFETY_GRADIENT_RATIO,
        "valid_case_fraction": float(np.mean([row["valid"] for row in records])),
        "valid_case_count": len(dice_rms),
        "skipped_case_count": len(records) - len(dice_rms),
        "dice_gradient_rms": _summary(dice_rms),
        "band_gradient_rms_unconditional": _summary(band_rms),
        "band_gradient_rms_conditional": _summary(conditional_band_rms),
        "cases": records,
    }
    validate_source_provenance_unchanged(source_provenance)
    finalize_calibration_report(
        args.output,
        base_payload,
        dice_rms,
        band_rms,
    )


if __name__ == "__main__":
    main()
