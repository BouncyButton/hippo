#!/usr/bin/env python3
"""Calibrate the existential A/P-plane weight on fold-training cases only."""

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

from thesis.new_constraints.ap_plane import (  # noqa: E402
    BestFitAPPlaneLocationLoss,
    ExistentialAPPlaneLoss,
    OriginalLabelAPConditionalCELoss,
)
from thesis.new_constraints.bands.calibrate_weight import (  # noqa: E402
    MAX_CALIBRATION_CASES,
    SAFETY_GRADIENT_RATIO,
    TARGET_GRADIENT_RATIO,
    _build_training_loader,
    _summary,
    calibrated_weight,
)
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
    parser.add_argument(
        "--dataset", choices=("MSD", "MNI", "ADNI", "COBRA"), default="MSD"
    )
    parser.add_argument("--splits-json", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=MAX_CALIBRATION_CASES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spatial-size", type=int, nargs=3, default=(64, 64, 64))
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--ap-plane-axis", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument(
        "--ap-plane-anterior-side", choices=("low", "high"), required=True
    )
    parser.add_argument("--ap-plane-margin", type=float, default=0.0)
    parser.add_argument(
        "--ap-plane-objective",
        choices=("existential", "location", "conditional_ce"),
        default="existential",
        help="Calibrate coherence-only or patient-specific cut-location supervision.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _validate_source_checkpoint(
    payload: Any,
    args: argparse.Namespace,
    *,
    pkl_sha256: str,
    splits_sha256: str,
) -> tuple[dict[str, torch.Tensor], int, dict[str, Any]]:
    """Accept the existing completed Dice seed-0 control as calibration source."""

    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError("Calibration requires a full training checkpoint.")
    run = payload.get("run")
    if not isinstance(run, dict):
        raise ValueError("Calibration checkpoint lacks embedded run provenance.")
    if run.get("constraint_set") != "none":
        raise ValueError("Calibration checkpoint must be an unconstrained control.")
    if run.get("supervised_loss", "dice") != "dice" or float(
        run.get("ce_weight", 0.0)
    ) != 0.0:
        raise ValueError("Calibration checkpoint must use Dice-only supervision.")
    if run.get("dataset") != args.dataset or int(run.get("fold", -1)) != args.fold:
        raise ValueError("Calibration checkpoint dataset/fold does not match.")
    if int(run.get("seed", -1)) != args.seed:
        raise ValueError("Calibration checkpoint seed does not match.")
    if tuple(run.get("spatial_size", ())) != tuple(args.spatial_size):
        raise ValueError("Calibration checkpoint spatial size does not match.")
    if bool(run.get("resize", False)) != bool(args.resize):
        raise ValueError("Calibration checkpoint resize mode does not match.")
    if run.get("pkl_sha256") != pkl_sha256:
        raise ValueError("Dataset pickle differs from the checkpoint input.")
    if run.get("splits_json_sha256") != splits_sha256:
        raise ValueError("Split file differs from the checkpoint input.")
    epoch = payload.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise ValueError("Calibration checkpoint epoch is invalid.")
    return payload["model"], epoch, run


def main() -> None:
    args = parse_args()
    if not 1 <= args.max_cases <= MAX_CALIBRATION_CASES:
        raise ValueError(f"--max-cases must be in 1..{MAX_CALIBRATION_CASES}.")
    if not math.isfinite(args.ap_plane_margin) or args.ap_plane_margin < 0:
        raise ValueError("--ap-plane-margin must be finite and non-negative.")
    for path in (args.pkl, args.splits_json, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_provenance = collect_source_provenance()
    runtime_provenance = collect_runtime_provenance()
    with snapshot_file(args.pkl) as pkl_snapshot, snapshot_file(
        args.splits_json
    ) as splits_snapshot, snapshot_file(args.checkpoint) as checkpoint_snapshot:
        seed_everything(args.seed)
        device = resolve_device(args.device)
        if args.amp and device.type != "cuda":
            raise ValueError("--amp requires CUDA; use --no-amp on CPU.")
        execution_provenance = collect_execution_provenance(device)
        checkpoint = torch.load(
            checkpoint_snapshot.path, map_location="cpu", weights_only=True
        )
        state_dict, checkpoint_epoch, checkpoint_run = _validate_source_checkpoint(
            checkpoint,
            args,
            pkl_sha256=pkl_snapshot.sha256,
            splits_sha256=splits_snapshot.sha256,
        )
        loader, selected_cases, num_classes = _build_training_loader(
            args, pkl_path=pkl_snapshot.path, splits_path=splits_snapshot.path
        )
        digests = {
            "pkl": pkl_snapshot.sha256,
            "splits": splits_snapshot.sha256,
            "checkpoint": checkpoint_snapshot.sha256,
        }

    validate_source_provenance_unchanged(source_provenance)
    model = build_swinunetr(tuple(args.spatial_size), num_classes, device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    dice_loss = DiceLoss(to_onehot_y=True, softmax=True)
    plane_loss_class = {
        "existential": ExistentialAPPlaneLoss,
        "location": BestFitAPPlaneLocationLoss,
        "conditional_ce": OriginalLabelAPConditionalCELoss,
    }[args.ap_plane_objective]
    plane_loss = plane_loss_class(
        axis=args.ap_plane_axis,
        anterior_high=args.ap_plane_anterior_side == "high",
        margin=args.ap_plane_margin,
        require_both=True,
    ).to(device)

    records: list[dict[str, Any]] = []
    dice_rms_values: list[float] = []
    plane_rms_values: list[float] = []
    for selected, batch in zip(selected_cases, loader, strict=True):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.amp):
            model_logits = model(images)
        logits = model_logits.detach().float().requires_grad_(True)
        supervised = dice_loss(logits, labels)
        result = plane_loss(logits, labels)
        dice_gradient = torch.autograd.grad(
            supervised, logits, retain_graph=True
        )[0].float()
        plane_gradient = torch.autograd.grad(result.loss, logits)[0].float()
        dice_rms = float(dice_gradient.square().mean().sqrt().cpu())
        plane_rms = float(plane_gradient.square().mean().sqrt().cpu())
        valid = bool(result.details["valid"].item())
        finite = math.isfinite(dice_rms) and math.isfinite(plane_rms)
        if valid and finite and dice_rms > 0 and plane_rms > 0:
            dice_rms_values.append(dice_rms)
            plane_rms_values.append(plane_rms)
        records.append(
            {
                **selected,
                "valid": valid,
                "finite_positive_gradients": finite
                and dice_rms > 0
                and plane_rms > 0,
                "dice_gradient_rms": _finite(dice_rms),
                "plane_gradient_rms": _finite(plane_rms),
                "plane_loss": _finite(float(result.details["case_loss"].item())),
                **(
                    {
                        "selected_cut": int(result.details["selected_cut"].item()),
                        "candidate_count": int(
                            result.details["candidate_count"].item()
                        ),
                        "target_cut": int(result.details["target_cut"].item()),
                        "cut_abs_error": float(
                            result.details["cut_abs_error"].item()
                        ),
                        "gt_plane_disagreement_fraction": float(
                            result.details["gt_plane_disagreement_fraction"].item()
                        ),
                        "target_cut_count": int(
                            result.details["target_cut_count"].item()
                        ),
                    }
                    if args.ap_plane_objective == "location"
                    else {
                        **(
                            {
                                "selected_cut": int(
                                    result.details["selected_cut"].item()
                                ),
                                "candidate_count": int(
                                    result.details["candidate_count"].item()
                                ),
                            }
                            if args.ap_plane_objective == "existential"
                            else {}
                        )
                    }
                ),
            }
        )

    payload: dict[str, Any] = {
        "constraint_set": {
            "existential": "ap_plane",
            "location": "ap_plane_location",
            "conditional_ce": "ap_plane_ce_control",
        }[args.ap_plane_objective],
        "status": "complete",
        "dataset": args.dataset,
        "fold": args.fold,
        "seed": args.seed,
        "spatial_size": list(args.spatial_size),
        "resize": bool(args.resize),
        "max_cases": args.max_cases,
        "amp": args.amp,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": digests["checkpoint"],
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_constraint_set": checkpoint_run["constraint_set"],
        "checkpoint_source_sha256": checkpoint_run.get("source_sha256"),
        "pkl": str(args.pkl.resolve()),
        "pkl_sha256": digests["pkl"],
        "splits_json": str(args.splits_json.resolve()),
        "splits_json_sha256": digests["splits"],
        "source_provenance": source_provenance,
        "runtime_provenance": runtime_provenance,
        "execution_provenance": execution_provenance,
        "training_cases": selected_cases,
        "training_case_ids": [item["case_name"] for item in selected_cases],
        "ap_plane": {
            "axis": args.ap_plane_axis,
            "anterior_side": args.ap_plane_anterior_side,
            "margin": args.ap_plane_margin,
            "require_both": True,
            **(
                {"objective": args.ap_plane_objective}
                if args.ap_plane_objective != "existential"
                else {}
            ),
        },
        "target_ratio": TARGET_GRADIENT_RATIO,
        "safety_ratio": SAFETY_GRADIENT_RATIO,
        "valid_case_count": len(dice_rms_values),
        "skipped_case_count": len(records) - len(dice_rms_values),
        "dice_gradient_rms": _summary(dice_rms_values),
        "plane_gradient_rms": _summary(plane_rms_values),
        "cases": records,
    }
    try:
        final_weight, target_weight, cap_weight = calibrated_weight(
            dice_rms_values,
            plane_rms_values,
            target_ratio=TARGET_GRADIENT_RATIO,
            safety_ratio=SAFETY_GRADIENT_RATIO,
        )
    except ValueError as error:
        payload.update(
            {
                "status": "failed",
                "failure_reason": str(error),
                "target_weight": None,
                "cap_weight": None,
                "recommended_ap_plane_weight": None,
            }
        )
        save_json(args.output, payload)
        raise RuntimeError("A/P-plane calibration failed after writing diagnostics.") from error
    payload.update(
        {
            "target_weight": target_weight,
            "cap_weight": cap_weight,
            "recommended_ap_plane_weight": final_weight,
        }
    )
    validate_source_provenance_unchanged(source_provenance)
    save_json(args.output, payload)
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
