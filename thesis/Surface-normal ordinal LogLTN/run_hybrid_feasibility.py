#!/usr/bin/env python3
"""Frozen-logit feasibility audit for band and one-cut hybrid objectives."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from surface_normal_ordinal.counterfactual import (  # noqa: E402
    grouped_boundary_bce,
    soft_multiclass_dice,
)
from surface_normal_ordinal.geometry import Interface, build_surface_samples  # noqa: E402
from surface_normal_ordinal.hybrid import tolerance_aware_components  # noqa: E402
from surface_normal_ordinal.io import load_case_bundle, sha256, write_json  # noqa: E402
from surface_normal_ordinal.metrics import segmentation_metrics  # noqa: E402
from surface_normal_ordinal.objective import onecut_logltn_loss  # noqa: E402


PRIMITIVES = ("dice", "band", "onecut", "anchor", "location")
CANDIDATES = ("band", "onecut", "naive_hybrid", "tolerance_hybrid")
FLOAT_METRICS = (
    "union_dice",
    "anterior_dice",
    "posterior_dice",
    "surface_dice_1mm",
    "surface_dice_2mm",
    "assd_mm",
    "hd95_mm",
    "soft_dice_score",
)
COUNT_METRICS = (
    "foreground_fp",
    "foreground_fn",
    "ap_swaps",
    "predicted_foreground_voxels",
    "predicted_components",
    "total_errors",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--radius-mm", type=float, default=3.0)
    parser.add_argument("--ray-step-mm", type=float, default=0.5)
    parser.add_argument("--tolerance-mm", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-surface-points", type=int, default=4096)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--target-aux-gradient-ratio", type=float, default=0.10)
    parser.add_argument("--high-gradient-cap-ratio", type=float, default=0.50)
    parser.add_argument(
        "--repair-update-rms", type=float, nargs="+", default=(0.02, 0.05, 0.10)
    )
    return parser.parse_args()


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(requested)


def _rms(value: torch.Tensor) -> float:
    return float(value.float().square().mean().sqrt().detach().cpu())


def _cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    numerator = float((first.float() * second.float()).sum().detach().cpu())
    denominator = float(first.float().norm().detach().cpu()) * float(
        second.float().norm().detach().cpu()
    )
    return numerator / denominator if denominator > 0 else float("nan")


def _residual_fraction(first: torch.Tensor, second: torch.Tensor) -> float:
    """Fraction of ``second`` left after projection onto ``first``."""

    first_flat = first.float().reshape(-1)
    second_flat = second.float().reshape(-1)
    denominator = first_flat.square().sum()
    if float(denominator.detach().cpu()) <= 0:
        return float("nan")
    projection = (first_flat @ second_flat) / denominator * first_flat
    second_norm = second_flat.norm()
    if float(second_norm.detach().cpu()) <= 0:
        return float("nan")
    return float(((second_flat - projection).norm() / second_norm).detach().cpu())


def _voxel_conflict(first: torch.Tensor, second: torch.Tensor) -> tuple[float, float, float]:
    first = first.float()
    second = second.float()
    dot = (first * second).sum(dim=0)
    active = (first.square().sum(dim=0) > 0) & (second.square().sum(dim=0) > 0)
    overlap = float(active.float().mean().detach().cpu())
    if not bool(active.any()):
        return overlap, float("nan"), float("nan")
    selected = dot[active]
    conflict_fraction = float((selected < 0).float().mean().detach().cpu())
    absolute_mass = selected.abs().sum()
    conflict_mass = (
        float((-selected[selected < 0]).sum().detach().cpu() / absolute_mass.detach().cpu())
        if float(absolute_mass.detach().cpu()) > 0
        else float("nan")
    )
    return overlap, conflict_fraction, conflict_mass


def _offsets(args: argparse.Namespace, device: torch.device) -> tuple[np.ndarray, torch.Tensor]:
    values = np.arange(
        -args.radius_mm,
        args.radius_mm + 0.5 * args.ray_step_mm,
        args.ray_step_mm,
    )
    values = np.unique(np.concatenate((values, [0.0])))
    return values, torch.as_tensor(values, dtype=torch.float32, device=device)


def _losses_and_gradients(
    path: Path,
    args: argparse.Namespace,
    device: torch.device,
    case_index: int,
) -> tuple[Any, dict[str, torch.Tensor], dict[str, float]]:
    case = load_case_bundle(path)
    logits = torch.from_numpy(case.logits).to(device).float().requires_grad_(True)
    labels = torch.as_tensor(case.labels, dtype=torch.long, device=device)
    samples = build_surface_samples(
        case.labels,
        case.spacing,
        interface=Interface.OUTER,
        radius_mm=args.radius_mm,
        max_points=args.max_surface_points,
        seed=args.seed + case_index,
    )
    offsets_np, offsets = _offsets(args, device)
    coordinates = torch.as_tensor(
        samples.coordinates_at(offsets_np), dtype=torch.float32, device=device
    )
    onecut, _, allowed_mass, _ = onecut_logltn_loss(
        logits,
        coordinates,
        offsets,
        interface="outer",
        tolerance_mm=args.tolerance_mm,
        margin=args.margin,
        temperature=args.temperature,
    )
    anchor, location, hybrid_details = tolerance_aware_components(
        logits,
        coordinates,
        offsets,
        tolerance_mm=args.tolerance_mm,
        margin=args.margin,
        temperature=args.temperature,
    )
    losses = {
        "dice": soft_multiclass_dice(logits, labels),
        "band": grouped_boundary_bce(logits, labels),
        "onecut": onecut,
        "anchor": anchor,
        "location": location,
    }
    gradients = {
        name: torch.autograd.grad(loss, logits, retain_graph=True)[0].float()
        for name, loss in losses.items()
    }
    details = {
        **{f"{name}_loss": float(loss.detach().cpu()) for name, loss in losses.items()},
        **{f"{name}_gradient_rms": _rms(gradient) for name, gradient in gradients.items()},
        "ray_count": float(coordinates.shape[0]),
        "sample_count": float(coordinates.shape[1]),
        "onecut_allowed_mass": float(allowed_mass.mean().detach().cpu()),
        "conditional_allowed_mass": float(
            hybrid_details["allowed_cut_mass"].mean().detach().cpu()
        ),
    }
    return case, gradients, details


def _primitive_row(
    case_name: str, gradients: dict[str, torch.Tensor], details: dict[str, float]
) -> dict[str, Any]:
    row: dict[str, Any] = {"case_name": case_name, **details}
    for name in PRIMITIVES[1:]:
        row[f"dice_{name}_cosine"] = _cosine(gradients["dice"], gradients[name])
    for first, second, label in (
        ("band", "onecut", "band_onecut"),
        ("anchor", "location", "anchor_location"),
    ):
        row[f"{label}_cosine"] = _cosine(gradients[first], gradients[second])
        row[f"{label}_second_residual_fraction"] = _residual_fraction(
            gradients[first], gradients[second]
        )
        overlap, conflict, mass = _voxel_conflict(
            gradients[first], gradients[second]
        )
        row[f"{label}_voxel_overlap_fraction"] = overlap
        row[f"{label}_conflict_fraction"] = conflict
        row[f"{label}_conflict_mass_fraction"] = mass
    return row


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Summary values must be finite and nonempty.")
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _combined_rms(
    first_rms: float,
    second_rms: float,
    cosine: float,
    first_scale: float,
    second_scale: float,
) -> float:
    squared = (
        (first_scale * first_rms) ** 2
        + (second_scale * second_rms) ** 2
        + 2.0
        * first_scale
        * second_scale
        * first_rms
        * second_rms
        * cosine
    )
    return math.sqrt(max(0.0, squared))


def _calibrate(
    dice_rms: list[float],
    candidate_rms: list[float],
    target_ratio: float,
    cap_ratio: float,
) -> dict[str, float]:
    dice_median = float(np.median(dice_rms))
    candidate_median = float(np.median(candidate_rms))
    candidate_q95 = float(np.quantile(candidate_rms, 0.95))
    if min(dice_median, candidate_median, candidate_q95) <= 0:
        raise ValueError("Gradient calibration requires positive gradient magnitudes.")
    target_weight = target_ratio * dice_median / candidate_median
    cap_weight = cap_ratio * dice_median / candidate_q95
    return {
        "median_gradient_rms": candidate_median,
        "q95_gradient_rms": candidate_q95,
        "target_weight": target_weight,
        "cap_weight": cap_weight,
        "selected_weight": min(target_weight, cap_weight),
    }


def _candidate_coefficients(
    rows: list[dict[str, Any]], args: argparse.Namespace
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    medians = {
        name: float(np.median([row[f"{name}_gradient_rms"] for row in rows]))
        for name in PRIMITIVES
    }
    dice_rms = [float(row["dice_gradient_rms"]) for row in rows]
    raw_rms: dict[str, list[float]] = {
        "band": [float(row["band_gradient_rms"]) for row in rows],
        "onecut": [float(row["onecut_gradient_rms"]) for row in rows],
        "naive_hybrid": [
            _combined_rms(
                float(row["band_gradient_rms"]),
                float(row["onecut_gradient_rms"]),
                float(row["band_onecut_cosine"]),
                0.5 / medians["band"],
                0.5 / medians["onecut"],
            )
            for row in rows
        ],
        "tolerance_hybrid": [
            _combined_rms(
                float(row["anchor_gradient_rms"]),
                float(row["location_gradient_rms"]),
                float(row["anchor_location_cosine"]),
                0.5 / medians["anchor"],
                0.5 / medians["location"],
            )
            for row in rows
        ],
    }
    calibration = {
        name: _calibrate(
            dice_rms,
            values,
            args.target_aux_gradient_ratio,
            args.high_gradient_cap_ratio,
        )
        for name, values in raw_rms.items()
    }
    coefficients = {
        "band": {"band": calibration["band"]["selected_weight"]},
        "onecut": {"onecut": calibration["onecut"]["selected_weight"]},
        "naive_hybrid": {
            "band": calibration["naive_hybrid"]["selected_weight"]
            * 0.5
            / medians["band"],
            "onecut": calibration["naive_hybrid"]["selected_weight"]
            * 0.5
            / medians["onecut"],
        },
        "tolerance_hybrid": {
            "anchor": calibration["tolerance_hybrid"]["selected_weight"]
            * 0.5
            / medians["anchor"],
            "location": calibration["tolerance_hybrid"]["selected_weight"]
            * 0.5
            / medians["location"],
        },
    }
    return calibration, coefficients


def _candidate_gradient(
    gradients: dict[str, torch.Tensor], coefficients: dict[str, float]
) -> torch.Tensor:
    output = torch.zeros_like(gradients["dice"])
    for name, coefficient in coefficients.items():
        output = output + coefficient * gradients[name]
    return output


def _matched_update(
    logits: torch.Tensor, gradient: torch.Tensor, target_rms: float
) -> torch.Tensor:
    gradient_rms = _rms(gradient)
    if gradient_rms <= 0 or not math.isfinite(gradient_rms):
        raise ValueError("Cannot form a matched update from a zero or invalid gradient.")
    return (logits.detach() - target_rms * gradient / gradient_rms).detach()


def _metrics(logits: torch.Tensor, labels: np.ndarray, spacing: tuple[float, ...]) -> dict[str, Any]:
    prediction = logits.argmax(dim=0).cpu().numpy().astype(np.uint8)
    values = segmentation_metrics(labels, prediction, spacing)
    labels_t = torch.as_tensor(labels, dtype=torch.long, device=logits.device)
    values["soft_dice_score"] = 1.0 - float(
        soft_multiclass_dice(logits, labels_t).detach().cpu()
    )
    values["total_errors"] = int(
        values["foreground_fp"] + values["foreground_fn"] + values["ap_swaps"]
    )
    return values


def _counterfactual_rows(
    paths: list[Path],
    args: argparse.Namespace,
    device: torch.device,
    coefficients: dict[str, dict[str, float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    for case_index, path in enumerate(paths):
        case, gradients, _ = _losses_and_gradients(
            path, args, device, case_index
        )
        logits = torch.from_numpy(case.logits).to(device).float()
        candidate_gradients = {
            name: _candidate_gradient(gradients, coefficients[name])
            for name in CANDIDATES
        }
        alignment_rows.append(
            {
                "case_name": case.case_name,
                "naive_band_alignment": _cosine(
                    candidate_gradients["naive_hybrid"], gradients["band"]
                ),
                "naive_onecut_alignment": _cosine(
                    candidate_gradients["naive_hybrid"], gradients["onecut"]
                ),
                "tolerance_anchor_alignment": _cosine(
                    candidate_gradients["tolerance_hybrid"], gradients["anchor"]
                ),
                "tolerance_location_alignment": _cosine(
                    candidate_gradients["tolerance_hybrid"], gradients["location"]
                ),
            }
        )
        baseline = _metrics(logits, case.labels, case.spacing)
        for update_rms in args.repair_update_rms:
            metric_rows.append(
                {
                    "case_name": case.case_name,
                    "update_rms": update_rms,
                    "arm": "baseline",
                    **baseline,
                }
            )
            dice_updated = _matched_update(logits, gradients["dice"], update_rms)
            metric_rows.append(
                {
                    "case_name": case.case_name,
                    "update_rms": update_rms,
                    "arm": "dice_only",
                    **_metrics(dice_updated, case.labels, case.spacing),
                }
            )
            for name, auxiliary in candidate_gradients.items():
                combined = gradients["dice"] + auxiliary
                updated = _matched_update(logits, combined, update_rms)
                metric_rows.append(
                    {
                        "case_name": case.case_name,
                        "update_rms": update_rms,
                        "arm": f"dice_plus_{name}",
                        **_metrics(updated, case.labels, case.spacing),
                    }
                )
    return metric_rows, alignment_rows


def _comparison(
    rows: list[dict[str, Any]], candidate: str, reference: str
) -> dict[str, Any]:
    candidate_rows = {
        (str(row["case_name"]), float(row["update_rms"])): row
        for row in rows
        if row["arm"] == candidate
    }
    reference_rows = {
        (str(row["case_name"]), float(row["update_rms"])): row
        for row in rows
        if row["arm"] == reference
    }
    shared = sorted(candidate_rows.keys() & reference_rows.keys())
    output: dict[str, Any] = {"observation_count": len(shared)}
    for metric in (*FLOAT_METRICS, *COUNT_METRICS):
        deltas = [
            float(candidate_rows[key][metric]) - float(reference_rows[key][metric])
            for key in shared
        ]
        output[metric] = {
            **_summary(deltas),
            "positive_fraction": float(np.mean(np.asarray(deltas) > 0)),
            "negative_fraction": float(np.mean(np.asarray(deltas) < 0)),
            "unchanged_fraction": float(np.mean(np.asarray(deltas) == 0)),
        }
    return output


def _counterfactual_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"by_update_rms": {}}
    for update_rms in sorted({float(row["update_rms"]) for row in rows}):
        selected = [row for row in rows if float(row["update_rms"]) == update_rms]
        arms = sorted({str(row["arm"]) for row in selected})
        block: dict[str, Any] = {"arms": {}}
        for arm in arms:
            arm_rows = [row for row in selected if row["arm"] == arm]
            block["arms"][arm] = {
                metric: _summary([float(row[metric]) for row in arm_rows])
                for metric in (*FLOAT_METRICS, *COUNT_METRICS)
            }
        block["comparisons"] = {
            "naive_vs_band": _comparison(
                selected, "dice_plus_naive_hybrid", "dice_plus_band"
            ),
            "tolerance_vs_band": _comparison(
                selected, "dice_plus_tolerance_hybrid", "dice_plus_band"
            ),
            "tolerance_vs_onecut": _comparison(
                selected, "dice_plus_tolerance_hybrid", "dice_plus_onecut"
            ),
        }
        output["by_update_rms"][str(update_rms)] = block
    return output


def _decision(summary: dict[str, Any], gradient: dict[str, Any]) -> dict[str, Any]:
    feasible = (
        gradient["valid_case_count"] >= 32
        and gradient["tolerance_component_alignment_positive_fraction"] >= 0.90
        and gradient["anchor_location_second_residual_fraction"]["median"] >= 0.20
    )
    successful_updates: list[float] = []
    for update, block in summary["by_update_rms"].items():
        comparison = block["comparisons"]["tolerance_vs_band"]
        if (
            comparison["surface_dice_1mm"]["mean"] > 0
            and comparison["union_dice"]["mean"] >= -0.0005
            and comparison["total_errors"]["mean"] <= 0
        ):
            successful_updates.append(float(update))
    if not feasible:
        verdict = "NO_GO"
        reason = "The tolerance-aware components are invalid, redundant, or mutually destructive."
    elif successful_updates:
        verdict = "GO_TO_FIVE_EPOCH_HYBRID_PILOT"
        reason = "The tolerance-aware hybrid beats the calibrated band direction under matched frozen-logit updates."
    else:
        changes = [
            abs(
                block["comparisons"]["tolerance_vs_band"]["surface_dice_1mm"][
                    "mean"
                ]
            )
            for block in summary["by_update_rms"].values()
        ]
        verdict = "INCONCLUSIVE" if max(changes, default=0.0) < 1e-8 else "NO_GO"
        reason = (
            "The components are computationally feasible, but their matched repair does not improve on BCE bands."
        )
    return {
        "verdict": verdict,
        "reason": reason,
        "successful_update_rms": successful_updates,
        "gates": {
            "at_least_32_valid_cases": gradient["valid_case_count"] >= 32,
            "simultaneous_component_descent_in_at_least_90_percent": gradient[
                "tolerance_component_alignment_positive_fraction"
            ]
            >= 0.90,
            "median_location_residual_at_least_20_percent": gradient[
                "anchor_location_second_residual_fraction"
            ]["median"]
            >= 0.20,
            "matched_counterfactual_improvement": bool(successful_updates),
        },
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _report(payload: dict[str, Any]) -> str:
    gradient = payload["gradient_summary"]
    decision = payload["decision"]
    lines = [
        "# BCE-band + one-cut frozen-logit feasibility audit",
        "",
        f"Decision: **{decision['verdict']}**",
        "",
        decision["reason"],
        "",
        "## Gradient feasibility",
        "",
        f"- valid cases: {gradient['valid_case_count']}",
        f"- median band/one-cut cosine: {gradient['band_onecut_cosine']['median']:+.4f}",
        f"- median one-cut residual after projection on bands: {gradient['band_onecut_second_residual_fraction']['median']:.3f}",
        f"- median anchor/location cosine: {gradient['anchor_location_cosine']['median']:+.4f}",
        f"- median location residual after projection on anchors: {gradient['anchor_location_second_residual_fraction']['median']:.3f}",
        f"- tolerance hybrid descends both components in: {gradient['tolerance_component_alignment_positive_fraction']:.1%} of cases",
        "",
        "## Matched counterfactual comparison versus BCE bands",
        "",
    ]
    for update, block in payload["counterfactual_summary"]["by_update_rms"].items():
        comparison = block["comparisons"]["tolerance_vs_band"]
        lines.extend(
            [
                f"### Update RMS {update}",
                "",
                f"- union Dice: {comparison['union_dice']['mean']:+.6f}",
                f"- 1-mm surface Dice: {comparison['surface_dice_1mm']['mean']:+.6f}",
                f"- ASSD: {comparison['assd_mm']['mean']:+.6f} mm",
                f"- HD95: {comparison['hd95_mm']['mean']:+.6f} mm",
                f"- total errors per case: {comparison['total_errors']['mean']:+.3f}",
                "",
            ]
        )
    lines.extend(["## Gates", ""])
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'}: `{name}`"
        for name, passed in decision["gates"].items()
    )
    lines.extend(
        [
            "",
            "This is an exploratory frozen-logit diagnostic. A positive result authorizes only a matched five-epoch pilot, not a full run.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.radius_mm <= args.tolerance_mm or args.ray_step_mm <= 0:
        raise ValueError("radius must exceed tolerance and ray step must be positive.")
    if args.target_aux_gradient_ratio <= 0 or args.high_gradient_cap_ratio <= 0:
        raise ValueError("Gradient calibration ratios must be positive.")
    if any(value <= 0 for value in args.repair_update_rms):
        raise ValueError("Every repair update RMS must be positive.")
    paths = sorted(args.input_dir.glob("*.npz"))
    if args.max_cases is not None:
        paths = paths[: args.max_cases]
    if not paths:
        raise FileNotFoundError(f"No case bundles found in {args.input_dir}.")
    device = _device(args.device)
    args.output_dir.mkdir(parents=True)
    primitive_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for case_index, path in enumerate(paths):
        try:
            case, gradients, details = _losses_and_gradients(
                path, args, device, case_index
            )
            row = _primitive_row(case.case_name, gradients, details)
            numeric = [float(value) for key, value in row.items() if key != "case_name"]
            if not all(math.isfinite(value) for value in numeric):
                raise ValueError("Non-finite gradient diagnostic.")
            primitive_rows.append(row)
        except (ValueError, RuntimeError) as error:
            skipped.append({"path": str(path), "reason": str(error)})
    if len(primitive_rows) < 1:
        raise RuntimeError("No valid cases were available for the gradient audit.")
    valid_names = {str(row["case_name"]) for row in primitive_rows}
    valid_paths = [
        path for path in paths if load_case_bundle(path).case_name in valid_names
    ]
    calibration, coefficients = _candidate_coefficients(primitive_rows, args)
    metric_rows, alignment_rows = _counterfactual_rows(
        valid_paths, args, device, coefficients
    )
    fields = [
        "band_onecut_cosine",
        "band_onecut_second_residual_fraction",
        "band_onecut_voxel_overlap_fraction",
        "band_onecut_conflict_fraction",
        "band_onecut_conflict_mass_fraction",
        "anchor_location_cosine",
        "anchor_location_second_residual_fraction",
        "anchor_location_voxel_overlap_fraction",
        "anchor_location_conflict_fraction",
        "anchor_location_conflict_mass_fraction",
    ]
    gradient_summary: dict[str, Any] = {
        "valid_case_count": len(primitive_rows),
        "skipped_case_count": len(skipped),
        **{
            field: _summary([float(row[field]) for row in primitive_rows])
            for field in fields
        },
    }
    simultaneous = [
        row["tolerance_anchor_alignment"] > 0
        and row["tolerance_location_alignment"] > 0
        for row in alignment_rows
    ]
    gradient_summary["tolerance_component_alignment_positive_fraction"] = float(
        np.mean(simultaneous)
    )
    counterfactual = _counterfactual_summary(metric_rows)
    decision = _decision(counterfactual, gradient_summary)
    manifest_path = args.input_dir / "manifest.json"
    payload = {
        "schema_version": 1,
        "decision": decision,
        "config": {
            key: str(value.resolve()) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
        },
        "source": {
            "files": {
                str(path.relative_to(PACKAGE_ROOT)): sha256(path)
                for path in (
                    Path(__file__).resolve(),
                    PACKAGE_ROOT / "surface_normal_ordinal" / "counterfactual.py",
                    PACKAGE_ROOT / "surface_normal_ordinal" / "geometry.py",
                    PACKAGE_ROOT / "surface_normal_ordinal" / "hybrid.py",
                    PACKAGE_ROOT / "surface_normal_ordinal" / "io.py",
                    PACKAGE_ROOT / "surface_normal_ordinal" / "metrics.py",
                    PACKAGE_ROOT / "surface_normal_ordinal" / "objective.py",
                )
            }
        },
        "input": {
            "directory": str(args.input_dir.resolve()),
            "manifest": str(manifest_path.resolve()) if manifest_path.exists() else None,
            "manifest_sha256": sha256(manifest_path) if manifest_path.exists() else None,
            "bundle_count": len(paths),
        },
        "skipped": skipped,
        "gradient_summary": gradient_summary,
        "calibration": calibration,
        "candidate_coefficients": coefficients,
        "counterfactual_summary": counterfactual,
    }
    _write_csv(args.output_dir / "gradient_diagnostics.csv", primitive_rows)
    _write_csv(args.output_dir / "component_alignment.csv", alignment_rows)
    _write_csv(args.output_dir / "counterfactual_metrics.csv", metric_rows)
    write_json(args.output_dir / "audit_summary.json", payload)
    (args.output_dir / "REPORT.md").write_text(_report(payload), encoding="utf-8")
    print(json.dumps({"decision": decision, "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
