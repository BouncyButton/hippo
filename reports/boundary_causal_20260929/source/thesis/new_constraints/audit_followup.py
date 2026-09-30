#!/usr/bin/env python3
"""Bounded diagnostic-only frozen-logit audits for the follow-up constraints.

Input: one NPZ per case, containing logits [3,D,H,W], labels [D,H,W],
and, for the teacher, teacher_logits [V,3,D,H,W] and shifts [V,3].
Teacher logits are in each translated image's coordinate frame. This command
never declares a training calibration valid and never starts training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import monai
import numpy as np
import torch

from .ap_cut import APCutPosteriorLoss
from .source_bootstrap import source_digest
from .supervised import CalibrationDiagnostics, build_supervised_loss, supervised_loss_config
from .teacher import TranslationTeacherKLLoss


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def segmentation_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Score one case: foreground macro, union Dice, and disjoint error counts."""

    if logits.ndim != 4 or logits.shape[0] != 3 or labels.shape != logits.shape[1:]:
        raise ValueError("Metrics require logits [3,D,H,W] and labels [D,H,W].")
    prediction = logits.detach().argmax(dim=0)
    valid = torch.ones_like(labels, dtype=torch.bool) if valid_mask is None else valid_mask.bool()
    if valid.shape != labels.shape or not bool(valid.any()):
        raise ValueError("Metric support must have label shape and contain a valid voxel.")
    prediction, labels = prediction[valid], labels[valid]
    classes = {}
    for class_id in (1, 2):
        truth = labels == class_id
        decoded = prediction == class_id
        tp = int((truth & decoded).sum())
        fp = int((~truth & decoded).sum())
        fn = int((truth & ~decoded).sum())
        denominator = 2 * tp + fp + fn
        classes[str(class_id)] = {"tp": tp, "fp": fp, "fn": fn, "dice": 2 * tp / denominator if denominator else None}
    dice = [entry["dice"] for entry in classes.values() if entry["dice"] is not None]
    true_foreground, predicted_foreground = labels != 0, prediction != 0
    union_tp = int((true_foreground & predicted_foreground).sum())
    union_fp = int((~true_foreground & predicted_foreground).sum())
    union_fn = int((true_foreground & ~predicted_foreground).sum())
    swaps = int((true_foreground & predicted_foreground & (prediction != labels)).sum())
    denominator = 2 * union_tp + union_fp + union_fn
    return {
        "valid_voxels": int(valid.sum()),
        "macro_dice": sum(dice) / len(dice) if dice else None,
        "union_dice": 2 * union_tp / denominator if denominator else 1.0,
        "union_tp": union_tp,
        "fp": union_fp,
        "fn": union_fn,
        "swaps": swaps,
        "total_errors": union_fp + union_fn + swaps,
        "classes": classes,
    }


def gradient_statistics(
    supervised: torch.Tensor,
    auxiliary: torch.Tensor,
) -> dict[str, float | None]:
    """Whole-logit RMS and cosine, with null ratios for degenerate gradients."""

    first = supervised.detach().double().reshape(-1)
    second = auxiliary.detach().double().reshape(-1)
    supervised_rms = float(first.square().mean().sqrt())
    auxiliary_rms = float(second.square().mean().sqrt())
    product = float(first.norm() * second.norm())
    return {
        "supervised_rms": supervised_rms,
        "auxiliary_rms": auxiliary_rms,
        "auxiliary_over_supervised_rms": auxiliary_rms / supervised_rms if supervised_rms > 0 else None,
        "cosine": float((first @ second) / product) if product > 0 else None,
        "auxiliary_maxabs": float(second.abs().max()),
    }


def diagnostic_weight(gradients: list[dict[str, float | None]]) -> dict[str, Any]:
    """Scale the median RMS ratio to 0.1, subject to a 0.5 p95-ratio cap."""

    ratios = [
        entry["auxiliary_over_supervised_rms"] for entry in gradients
        if entry["auxiliary_over_supervised_rms"] is not None
        and math.isfinite(entry["auxiliary_over_supervised_rms"])
        and entry["auxiliary_over_supervised_rms"] > 0
    ]
    if not ratios:
        return {"value": None, "valid_cases": 0, "reason": "no nonzero finite paired gradients", "calibration_valid": False}
    median = float(np.median(ratios))
    percentile95 = float(np.percentile(ratios, 95))
    median_weight = 0.1 / median
    cap_weight = 0.5 / percentile95
    return {
        "value": min(median_weight, cap_weight),
        "valid_cases": len(ratios),
        "median_auxiliary_over_supervised_rms": median,
        "p95_auxiliary_over_supervised_rms": percentile95,
        "median_target": 0.1,
        "p95_cap": 0.5,
        "median_target_weight": median_weight,
        "p95_cap_weight": cap_weight,
        "calibration_valid": False,
        "reason": "diagnostic cohort; no verified training-fold/checkpoint calibration protocol",
    }


def normalized_logit_step(logits: torch.Tensor, gradient: torch.Tensor, step: float) -> torch.Tensor:
    """Single fixed-gradient update whose largest absolute logit change is step."""

    if not math.isfinite(step) or step <= 0:
        raise ValueError("Repair step sizes must be finite and positive.")
    magnitude = float(gradient.detach().abs().max())
    if magnitude == 0:
        return logits.detach().clone()
    return logits.detach() - (gradient.detach() / magnitude) * step


def _gradient_strata(gradient: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    predicted = logits.detach().argmax(dim=0)
    wrong = predicted != labels
    confidence = torch.softmax(logits.detach(), dim=0).max(dim=0).values
    masks = {
        "whole": torch.ones_like(labels, dtype=torch.bool),
        "gt_foreground": labels != 0,
        "fp": (predicted != 0) & (labels == 0),
        "fn": (predicted == 0) & (labels != 0),
        "swaps": (predicted != 0) & (labels != 0) & wrong,
        "saturated_wrong": wrong & (confidence >= 0.99),
    }
    total_energy = float(gradient.detach().double().square().sum())
    result = {}
    for name, mask in masks.items():
        values = gradient.detach()[:, mask].double()
        result[name] = {
            "voxels": int(mask.sum()),
            "rms": float(values.square().mean().sqrt()) if values.numel() else None,
            "squared_gradient_fraction": float(values.square().sum()) / total_energy if total_energy else None,
        }
    return result


def _load_case(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as cached:
        if "logits" not in cached or "labels" not in cached:
            raise ValueError("NPZ must contain logits and labels.")
        logits_array = cached["logits"]
        label_array = cached["labels"]
        if logits_array.ndim != 4 or logits_array.shape[0] != 3 or min(logits_array.shape) < 1:
            raise ValueError("logits must have nonempty shape [3,D,H,W].")
        if label_array.shape != logits_array.shape[1:]:
            raise ValueError("labels must have shape [D,H,W] matching logits.")
        if not np.issubdtype(logits_array.dtype, np.floating) or not np.isfinite(logits_array).all():
            raise ValueError("logits must be finite floating point values.")
        if not np.isin(label_array, (0, 1, 2)).all():
            raise ValueError("labels must contain only integer classes 0, 1, and 2.")
        result = {
            "logits": torch.from_numpy(logits_array.astype(np.float32)),
            "labels": torch.from_numpy(label_array.astype(np.int64)),
            "source_dtype": str(logits_array.dtype),
            "teacher_logits": None,
            "shifts": None,
        }
        if "teacher_logits" in cached or "shifts" in cached:
            if "teacher_logits" not in cached or "shifts" not in cached:
                raise ValueError("Teacher caches require both teacher_logits and shifts.")
            teacher = cached["teacher_logits"]
            shifts = cached["shifts"]
            if teacher.ndim != 5 or teacher.shape[1:] != logits_array.shape or teacher.shape[0] < 1:
                raise ValueError("teacher_logits must have shape [V,3,D,H,W] matching logits.")
            if not np.issubdtype(teacher.dtype, np.floating) or not np.isfinite(teacher).all():
                raise ValueError("teacher_logits must be finite floating point values.")
            if shifts.shape != (teacher.shape[0], 3) or not np.issubdtype(shifts.dtype, np.integer):
                raise ValueError("shifts must have integer shape [V,3].")
            result["teacher_logits"] = torch.from_numpy(teacher.astype(np.float32))
            result["shifts"] = tuple(tuple(int(value) for value in shift) for shift in shifts)
            result["teacher_source_dtype"] = str(teacher.dtype)
    return result


def _case_audit(path: Path, args: argparse.Namespace, supervised_loss) -> tuple[dict[str, Any], dict[str, torch.Tensor] | None]:
    record: dict[str, Any] = {"case_id": path.stem, "path": str(path.resolve()), "sha256": file_sha256(path)}
    try:
        case = _load_case(path)
        device = torch.device(getattr(args, "device", "cpu"))
        logits = case["logits"].to(device).clone().requires_grad_()
        labels = case["labels"].to(device)
        record["shape"] = list(logits.shape)
        record["source_dtype"] = case["source_dtype"]
        record["baseline"] = segmentation_metrics(logits, labels)
        calibration = CalibrationDiagnostics(num_classes=3)
        record["calibration_diagnostics"] = calibration.update(logits[None], labels[None])[0]
        supervised = supervised_loss(logits[None], labels[None, None])
        supervised_gradient, = torch.autograd.grad(supervised, logits)
        teacher = None
        if args.constraint == "teacher":
            if case["teacher_logits"] is None:
                raise ValueError("Teacher audit requires teacher_logits and shifts in each NPZ.")
            constraint = TranslationTeacherKLLoss(temperature=args.temperature)
            teacher = constraint.build_from_cached(
                [view[None].to(device) for view in case["teacher_logits"]], case["shifts"]
            )
            result = constraint.loss_from_teacher(logits[None], teacher)
            record["teacher_source_dtype"] = case["teacher_source_dtype"]
            record["shifts"] = [list(shift) for shift in teacher.shifts]
            support = teacher.valid_mask[0, 0]
            record["teacher_same_support"] = {
                "baseline": segmentation_metrics(logits, labels, support),
                "teacher": segmentation_metrics(teacher.probabilities[0], labels, support),
            }
        else:
            constraint = APCutPosteriorLoss(
                axis=args.ap_axis,
                anterior_low=args.ap_anterior_side == "low",
                temperature=args.temperature,
                invalid_policy="skip",
            )
            result = constraint(logits[None], labels[None])
            if not bool(result.details["valid"].any()):
                record.update({"status": "skipped", "reason": result.details["invalid_reasons"][0]})
                return record, None
        auxiliary_gradient, = torch.autograd.grad(result.loss, logits)
        if not torch.isfinite(auxiliary_gradient).all() or not torch.isfinite(supervised_gradient).all():
            raise ValueError("Non-finite logit gradients.")
        record.update({
            "status": "valid",
            "supervised_loss": float(supervised.detach()),
            "auxiliary_loss": float(result.loss.detach()),
            "auxiliary_truth": result.truth.detach().float().cpu().tolist(),
            "auxiliary_metrics": {name: value.detach().float().cpu().tolist() for name, value in result.details.get("metrics", {}).items()},
            "gradients": gradient_statistics(supervised_gradient, auxiliary_gradient),
            "gradient_strata": {
                "supervised": _gradient_strata(supervised_gradient, logits, labels),
                "auxiliary": _gradient_strata(auxiliary_gradient, logits, labels),
            },
        })
        return record, {"logits": logits.detach(), "labels": labels, "supervised": supervised_gradient, "auxiliary": auxiliary_gradient}
    except (ValueError, TypeError, KeyError) as error:
        record.update({"status": "skipped", "reason": str(error)})
        return record, None


def _aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"case_count": 0}
    macro = [record["macro_dice"] for record in records if record["macro_dice"] is not None]
    pooled_class_dice = []
    for class_id in ("1", "2"):
        tp = sum(record["classes"][class_id]["tp"] for record in records)
        fp = sum(record["classes"][class_id]["fp"] for record in records)
        fn = sum(record["classes"][class_id]["fn"] for record in records)
        denominator = 2 * tp + fp + fn
        if denominator:
            pooled_class_dice.append(2 * tp / denominator)
    union_tp = sum(record["union_tp"] for record in records)
    totals = {name: sum(record[name] for record in records) for name in ("fp", "fn", "swaps", "total_errors")}
    denominator = 2 * union_tp + totals["fp"] + totals["fn"]
    return {
        "case_count": len(records),
        "per_case_mean_macro_dice": sum(macro) / len(macro) if macro else None,
        "per_case_mean_union_dice": sum(record["union_dice"] for record in records) / len(records),
        "voxel_pooled_macro_dice": sum(pooled_class_dice) / len(pooled_class_dice) if pooled_class_dice else None,
        "voxel_pooled_union_dice": 2 * union_tp / denominator if denominator else 1.0,
        **totals,
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    """Run a bounded diagnostic on the selected device and create its report."""

    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if args.purpose != "diagnostic":
        raise ValueError("This tool is diagnostic-only; it cannot produce a valid training calibration.")
    device = torch.device(getattr(args, "device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA audit requested but no GPU is available; no CPU fallback.")
    if args.max_cases < 1:
        raise ValueError("max_cases must be positive.")
    if args.constraint == "ap_cut" and (args.ap_axis is None or args.ap_anterior_side is None):
        raise ValueError("AP audit requires explicit --ap-axis and --ap-anterior-side.")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("temperature must be finite and positive.")
    if not args.step_sizes or any(not math.isfinite(step) or step <= 0 for step in args.step_sizes):
        raise ValueError("Repair step sizes must be finite and positive.")
    if len(set(args.step_sizes)) != len(args.step_sizes):
        raise ValueError("Repair step sizes must be distinct.")
    input_dir = Path(args.input_dir).expanduser().resolve()
    paths = sorted(input_dir.glob("*.npz"))[:args.max_cases]
    if not paths:
        raise ValueError("No case NPZ files found in input directory.")
    source_sha256 = source_digest()
    checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
    checkpoint_provenance = {"path": str(checkpoint), "sha256": file_sha256(checkpoint)} if checkpoint else None
    supervised_loss = build_supervised_loss(args.supervised_loss, args.ce_weight).to(device)
    records, states = [], []
    for path in paths:
        record, state = _case_audit(path, args, supervised_loss)
        records.append(record)
        states.append(state)
    valid_records = [record for record in records if record["status"] == "valid"]
    weight = diagnostic_weight([record["gradients"] for record in valid_records])
    for record, state in zip(records, states):
        if state is None:
            continue
        directions = {"supervised": state["supervised"], "auxiliary": state["auxiliary"]}
        if weight["value"] is not None:
            directions["combined_diagnostic_weight"] = state["supervised"] + weight["value"] * state["auxiliary"]
        repairs = {}
        for name, gradient in directions.items():
            repairs[name] = {}
            for step in args.step_sizes:
                changed = normalized_logit_step(state["logits"], gradient, step)
                metrics = segmentation_metrics(changed, state["labels"])
                baseline_macro = record["baseline"]["macro_dice"]
                metrics["delta_macro_dice"] = metrics["macro_dice"] - baseline_macro if metrics["macro_dice"] is not None and baseline_macro is not None else None
                metrics["delta_union_dice"] = metrics["union_dice"] - record["baseline"]["union_dice"]
                metrics["delta_total_errors"] = metrics["total_errors"] - record["baseline"]["total_errors"]
                metrics["actual_maxabs_logit_change"] = float((changed - state["logits"]).abs().max())
                repairs[name][str(step)] = metrics
        record["repairs"] = repairs
    repair_summary = {}
    if valid_records:
        for direction in valid_records[0]["repairs"]:
            repair_summary[direction] = {
                str(step): _aggregate_metrics([record["repairs"][direction][str(step)] for record in valid_records])
                for step in args.step_sizes
            }
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "diagnostic",
        "training_calibration_valid": False,
        "quality_claim_supported": False,
        "interpretation": "Frozen-logit reachability on this selected diagnostic cohort; not training performance or independent validation of a tuned coefficient.",
        "selection": {"rule": "lexicographically sorted NPZ filename stems, first max_cases", "case_ids": [path.stem for path in paths]},
        "args": {
            key: str(value) if isinstance(value, Path) else list(value) if isinstance(value, tuple) else value
            for key, value in vars(args).items()
        },
        "source_sha256": source_sha256,
        "checkpoint": checkpoint_provenance,
        "checkpoint_linkage": "user-supplied checkpoint hash; NPZ-to-checkpoint generation linkage is not independently verified" if checkpoint else "unknown",
        "runtime": {
            "python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "monai": monai.__version__,
            "device": str(device), "evaluation_dtype": "float32",
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "supervised_objective": supervised_loss_config(args.supervised_loss, args.ce_weight),
        "metric_definitions": {
            "macro": "mean class Dice over anterior=1 and posterior=2; empty-both class omitted",
            "union": "foreground-vs-background Dice; empty-both union is one",
            "fp_fn": "foreground-union errors",
            "swaps": "predicted and GT classes both foreground but different",
            "aggregation": "per-case means and voxel-pooled totals are separate fields",
            "repairs": "independent single steps from unchanged base logits, gradient/max(abs(gradient)); whole-field maxabs normalization, no sequential updates",
            "gradient_ratio": "auxiliary logit-gradient RMS / supervised logit-gradient RMS across all classes and voxels, per case",
        },
        "diagnostic_auxiliary_weight": weight,
        "valid_cases": len(valid_records),
        "skipped_cases": len(records) - len(valid_records),
        "baseline_all_readable_cases": _aggregate_metrics([record["baseline"] for record in records if "baseline" in record]),
        "baseline_valid_cases": _aggregate_metrics([record["baseline"] for record in valid_records]),
        "repair_summary_same_valid_cases": repair_summary,
        "cases": records,
    }
    if source_digest() != source_sha256:
        raise RuntimeError("Source files changed during the audit; rerun after edits finish.")
    for record in records:
        if file_sha256(Path(record["path"])) != record["sha256"]:
            raise RuntimeError(f"Input changed during the audit: {record['path']}")
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(encoded)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--purpose", required=True, choices=("diagnostic",))
    parser.add_argument("--constraint", required=True, choices=("teacher", "ap_cut"))
    parser.add_argument("--supervised-loss", required=True, choices=("dice", "dice_ce"))
    parser.add_argument("--ce-weight", required=True, type=float)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-cases", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--ap-axis", type=int, choices=(0, 1, 2))
    parser.add_argument("--ap-anterior-side", choices=("low", "high"))
    parser.add_argument("--step-sizes", type=float, nargs="+", default=(0.25, 1.0, 4.0))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.set_num_threads(1)
    report = run_audit(args)
    print(json.dumps({"output": str(Path(args.output).resolve()), "valid_cases": report["valid_cases"], "skipped_cases": report["skipped_cases"], "purpose": report["purpose"]}, allow_nan=False))


if __name__ == "__main__":
    main()
