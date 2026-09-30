#!/usr/bin/env python3
"""Calibrate outer one-cut weight from deterministic training-only logit gradients."""

from __future__ import annotations

import argparse
import json
import math
import os
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
from monai.losses import DiceLoss  # noqa: E402

from thesis.new_constraints.bands.calibrate_weight import (  # noqa: E402
    CALIBRATION_CHECKPOINT_EPOCH,
    MAX_CALIBRATION_CASES,
    SAFETY_GRADIENT_RATIO,
    TARGET_GRADIENT_RATIO,
    _build_training_loader,
    _summary,
    calibrated_weight,
    validate_checkpoint_provenance,
)
from thesis.new_constraints.onecut import OuterOneCutLogLTNLoss  # noqa: E402
from thesis.new_constraints.train_swinunetr_constraints import (  # noqa: E402
    build_swinunetr,
    collect_execution_provenance,
    collect_runtime_provenance,
    collect_source_provenance,
    resolve_device,
    save_json,
    seed_everything,
    snapshot_file,
    validate_source_provenance_unchanged,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", type=Path, required=True)
    parser.add_argument("--dataset", choices=("MSD", "MNI", "ADNI", "COBRA"), default="MSD")
    parser.add_argument("--splits-json", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-config", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=MAX_CALIBRATION_CASES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spatial-size", type=int, nargs=3, default=(64, 64, 64))
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--foreground-class-ids", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--complement-class-ids", type=int, nargs="+", default=(0,))
    parser.add_argument("--onecut-spacing", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    parser.add_argument("--onecut-radius-mm", type=float, default=3.0)
    parser.add_argument("--onecut-ray-step-mm", type=float, default=0.5)
    parser.add_argument("--onecut-tolerance-mm", type=float, default=1.0)
    parser.add_argument("--onecut-margin", type=float, default=0.0)
    parser.add_argument("--onecut-temperature", type=float, default=1.0)
    parser.add_argument("--onecut-max-surface-points", type=int, default=4096)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    # Compatibility fields consumed by the shared strict checkpoint validator.
    parser.set_defaults(
        bands_focal_gamma=0.0,
        bands_inner_focal_gamma=None,
        bands_outer_focal_gamma=None,
        bands_loss_type="focal_bce",
    )
    return parser.parse_args()


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _write(path: Path, payload: dict[str, Any]) -> None:
    save_json(path, payload)
    print(json.dumps(payload, indent=2, allow_nan=False))


def main() -> None:
    args = parse_args()
    if not 1 <= args.max_cases <= MAX_CALIBRATION_CASES:
        raise ValueError(f"--max-cases must be in 1..{MAX_CALIBRATION_CASES}.")
    for path in (args.pkl, args.splits_json, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_provenance = collect_source_provenance()
    runtime_provenance = collect_runtime_provenance()
    config_input = args.checkpoint_config or args.checkpoint.parent / "config.json"
    with snapshot_file(args.pkl) as pkl_snapshot, snapshot_file(
        args.splits_json
    ) as splits_snapshot, snapshot_file(args.checkpoint) as checkpoint_snapshot, snapshot_file(
        config_input
    ) as config_snapshot:
        seed_everything(args.seed)
        device = resolve_device(args.device)
        if args.amp and device.type != "cuda":
            raise ValueError("--amp requires CUDA; use --no-amp on CPU.")
        execution_provenance = collect_execution_provenance(device)
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
        if float(source_config["run"]["constraint_config"].get("onecut_weight", 0.0)) != 0.0:
            raise ValueError("Calibration source checkpoint is not Dice-only for one-cut.")
        loader, selected_cases, num_classes = _build_training_loader(
            args, pkl_path=pkl_snapshot.path, splits_path=splits_snapshot.path
        )
        digests = {
            "pkl": pkl_snapshot.sha256,
            "splits": splits_snapshot.sha256,
            "checkpoint": checkpoint_snapshot.sha256,
            "config": config_snapshot.sha256,
        }
    validate_source_provenance_unchanged(source_provenance)
    model = build_swinunetr(tuple(args.spatial_size), num_classes, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    dice_loss = DiceLoss(to_onehot_y=True, softmax=True)
    onecut = OuterOneCutLogLTNLoss(
        foreground_class_ids=args.foreground_class_ids,
        complement_class_ids=args.complement_class_ids,
        spacing=args.onecut_spacing,
        radius_mm=args.onecut_radius_mm,
        ray_step_mm=args.onecut_ray_step_mm,
        tolerance_mm=args.onecut_tolerance_mm,
        margin=args.onecut_margin,
        temperature=args.onecut_temperature,
        max_surface_points=args.onecut_max_surface_points,
        geometry_seed=args.seed,
    ).to(device)
    records: list[dict[str, Any]] = []
    dice_rms_values: list[float] = []
    onecut_rms_values: list[float] = []
    for selected, batch in zip(selected_cases, loader, strict=True):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.amp):
            model_logits = model(images)
        logits = model_logits.detach().float().requires_grad_(True)
        supervised = dice_loss(logits, labels)
        result = onecut(logits, labels)
        dice_gradient = torch.autograd.grad(supervised, logits, retain_graph=True)[0].float()
        onecut_gradient = torch.autograd.grad(result.loss, logits)[0].float()
        dice_rms = float(dice_gradient.square().mean().sqrt().cpu())
        onecut_rms = float(onecut_gradient.square().mean().sqrt().cpu())
        valid = bool(result.details["valid"].item())
        if valid:
            dice_rms_values.append(dice_rms)
            onecut_rms_values.append(onecut_rms)
        records.append(
            {
                **selected,
                "valid": valid,
                "dice_gradient_rms": _finite(dice_rms),
                "onecut_gradient_rms_unconditional": _finite(onecut_rms),
                "ray_count": int(result.details["ray_count"].item()),
                "onecut_loss": _finite(float(result.details["case_loss"].item())),
                "onecut_truth": _finite(float(result.details["case_truth"].item())),
                "allowed_cut_mass": _finite(
                    float(result.details["allowed_cut_mass"].item())
                ),
                "edge_touching": bool(result.details["edge_touching"].item()),
            }
        )
    base_payload: dict[str, Any] = {
        "constraint_set": "onecut",
        "status": "complete",
        "dataset": args.dataset,
        "fold": args.fold,
        "seed": args.seed,
        "max_cases": args.max_cases,
        "amp": args.amp,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_config": str(config_path),
        "checkpoint_sha256": digests["checkpoint"],
        "checkpoint_config_sha256": digests["config"],
        "checkpoint_epoch": CALIBRATION_CHECKPOINT_EPOCH,
        "checkpoint_constraint_set": "none",
        "checkpoint_source_provenance": source_config["source_provenance"],
        "checkpoint_source_sha256": source_config["run"]["source_sha256"],
        "pkl": str(args.pkl.resolve()),
        "pkl_sha256": digests["pkl"],
        "splits_json": str(args.splits_json.resolve()),
        "splits_json_sha256": digests["splits"],
        "source_provenance": source_provenance,
        "runtime_provenance": runtime_provenance,
        "execution_provenance": execution_provenance,
        "training_cases": selected_cases,
        "training_case_ids": [item["case_name"] for item in selected_cases],
        "training_patient_ids": sorted({item["patient_id"] for item in selected_cases}),
        "foreground_class_ids": list(args.foreground_class_ids),
        "complement_class_ids": list(args.complement_class_ids),
        "onecut_geometry": {
            "spacing": list(args.onecut_spacing),
            "radius_mm": args.onecut_radius_mm,
            "ray_step_mm": args.onecut_ray_step_mm,
            "tolerance_mm": args.onecut_tolerance_mm,
            "margin": args.onecut_margin,
            "temperature": args.onecut_temperature,
            "max_surface_points": args.onecut_max_surface_points,
            "geometry_seed": args.seed,
        },
        "target_ratio": TARGET_GRADIENT_RATIO,
        "safety_ratio": SAFETY_GRADIENT_RATIO,
        "valid_case_count": len(dice_rms_values),
        "skipped_case_count": len(records) - len(dice_rms_values),
        "dice_gradient_rms": _summary(dice_rms_values),
        "onecut_gradient_rms_unconditional": _summary(onecut_rms_values),
        "cases": records,
    }
    try:
        final_weight, target_weight, cap_weight = calibrated_weight(
            dice_rms_values,
            onecut_rms_values,
            target_ratio=TARGET_GRADIENT_RATIO,
            safety_ratio=SAFETY_GRADIENT_RATIO,
        )
    except ValueError as error:
        base_payload.update(
            {
                "status": "failed",
                "failure_reason": str(error),
                "target_weight": None,
                "cap_weight": None,
                "recommended_onecut_weight": None,
            }
        )
        _write(args.output, base_payload)
        raise RuntimeError("One-cut calibration failed after writing diagnostics.") from error
    base_payload.update(
        {
            "target_weight": target_weight,
            "cap_weight": cap_weight,
            "recommended_onecut_weight": final_weight,
        }
    )
    validate_source_provenance_unchanged(source_provenance)
    _write(args.output, base_payload)


if __name__ == "__main__":
    main()
