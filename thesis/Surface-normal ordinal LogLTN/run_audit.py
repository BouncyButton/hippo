#!/usr/bin/env python3
"""Run the complete frozen-logit go/no-go audit before any new model training."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from surface_normal_ordinal.counterfactual import run_counterfactual  # noqa: E402
from surface_normal_ordinal.geometry import Interface, build_surface_samples  # noqa: E402
from surface_normal_ordinal.io import load_case_bundle, sha256, write_json  # noqa: E402
from surface_normal_ordinal.rays import CATEGORY_NAMES, analyse_rays  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory of case .npz bundles.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interfaces", choices=("outer", "outer+ap"), default="outer")
    parser.add_argument("--radius-mm", type=float, default=3.0)
    parser.add_argument("--ray-step-mm", type=float, default=0.5)
    parser.add_argument("--delta-mm", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--localization-tolerance-mm", type=float, default=1.0)
    parser.add_argument("--max-surface-points", type=int, default=4096)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--repair-update-rms", type=float, default=0.10)
    parser.add_argument("--target-aux-gradient-ratio", type=float, default=0.10)
    parser.add_argument("--min-error-coverage", type=float, default=0.30)
    parser.add_argument("--min-precision-lift", type=float, default=1.25)
    parser.add_argument("--min-positive-case-fraction", type=float, default=0.50)
    parser.add_argument("--onecut-false-positive-rate", type=float, default=0.05)
    parser.add_argument("--min-onecut-auc", type=float, default=0.70)
    return parser.parse_args()


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(requested)


def _finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_finite(rows))


def _aggregate_ray(rows: list[dict[str, Any]], interface: str) -> dict[str, Any]:
    selected = [row for row in rows if row["interface"] == interface]
    total = sum(int(row["ray_count"]) for row in selected)
    errors = sum(int(row["error_count"]) for row in selected)
    violations = sum(int(row["violation_count"]) for row in selected)
    joint = sum(int(row["error_violation_count"]) for row in selected)
    prevalence = errors / total if total else float("nan")
    precision = joint / violations if violations else float("nan")
    return {
        "case_count": len(selected),
        "ray_count": total,
        "error_count": errors,
        "violation_count": violations,
        "error_violation_count": joint,
        "error_prevalence": prevalence,
        "violation_rate": violations / total if total else float("nan"),
        "error_coverage": joint / errors if errors else float("nan"),
        "error_precision": precision,
        "precision_lift": precision / prevalence if prevalence > 0 and math.isfinite(precision) else float("nan"),
        "categories": dict(
            sum((Counter(row["categories"]) for row in selected), Counter())
        ),
    }


def _repair_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "union_dice",
        "surface_dice_1mm",
        "surface_dice_2mm",
        "assd_mm",
        "hd95_mm",
        "foreground_fp",
        "foreground_fn",
        "ap_swaps",
    )
    arms = sorted({str(row["arm"]) for row in rows})
    output: dict[str, Any] = {}
    for arm in arms:
        selected = [row for row in rows if row["arm"] == arm]
        output[arm] = {
            metric: {
                "mean": float(np.mean([float(row[metric]) for row in selected])),
                "median": float(np.median([float(row[metric]) for row in selected])),
            }
            for metric in metrics
        }
    comparisons = {}
    for candidate, reference in (
        ("ordinal_outer_only", "baseline"),
        ("dice_plus_ordinal_outer", "dice_only"),
        ("onecut_outer_only", "baseline"),
        ("dice_plus_onecut_outer", "dice_only"),
        ("bands_only", "baseline"),
        ("dice_plus_bands", "dice_only"),
        ("ordinal_ap_only", "baseline"),
        ("dice_plus_ordinal_ap", "dice_only"),
        ("onecut_ap_only", "baseline"),
        ("dice_plus_onecut_ap", "dice_only"),
        ("dice_plus_onecut_outer", "dice_plus_bands"),
        ("onecut_outer_only", "bands_only"),
    ):
        candidate_rows = {row["case_name"]: row for row in rows if row["arm"] == candidate}
        reference_rows = {row["case_name"]: row for row in rows if row["arm"] == reference}
        shared = sorted(candidate_rows.keys() & reference_rows.keys())
        if not shared:
            continue
        comparison: dict[str, Any] = {"case_count": len(shared)}
        for metric in metrics:
            deltas = np.asarray(
                [float(candidate_rows[name][metric]) - float(reference_rows[name][metric]) for name in shared]
            )
            comparison[metric] = {
                "mean_delta": float(deltas.mean()),
                "median_delta": float(np.median(deltas)),
                "positive_case_fraction": float(np.mean(deltas > 0)),
                "negative_case_fraction": float(np.mean(deltas < 0)),
                "unchanged_case_fraction": float(np.mean(deltas == 0)),
            }
        comparisons[f"{candidate}_vs_{reference}"] = comparison
    output["comparisons"] = comparisons
    return output


def _interface_recommendations(
    decision: dict[str, Any], repair: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {
        "outer": {
            "recommendation": decision["verdict"],
            "reason": "The primary screen applies to the outer H/background one-cut formula.",
        }
    }
    ap = repair.get("comparisons", {}).get("dice_plus_onecut_ap_vs_dice_only")
    if ap is None:
        output["ap"] = {
            "recommendation": "NOT_EVALUATED",
            "reason": "No valid A/P interface comparison was available.",
        }
        return output
    swap_delta = ap["ap_swaps"]["mean_delta"]
    union_delta = ap["union_dice"]["mean_delta"]
    if swap_delta < 0 and union_delta < 0:
        recommendation = "DEFER_SEPARATE_TRADEOFF_ABLATION"
        reason = (
            "The A/P one-cut gradient reduces swaps but slightly worsens union Dice; "
            "do not mix it into the first outer-surface pilot."
        )
    elif swap_delta < 0:
        recommendation = "GO_TO_SEPARATE_AP_PILOT"
        reason = "The A/P one-cut gradient reduces swaps without reducing union Dice."
    else:
        recommendation = "NO_GO"
        reason = "The A/P one-cut gradient does not reduce swaps in the matched repair."
    output["ap"] = {
        "recommendation": recommendation,
        "reason": reason,
        "mean_ap_swap_delta": swap_delta,
        "mean_union_dice_delta": union_delta,
    }
    return output


def _rank_auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Tie-aware ROC AUC without adding another runtime dependency."""

    from scipy.stats import rankdata

    positive = np.asarray(positive, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    count_positive = int(positive.sum())
    count_negative = int((~positive).sum())
    if count_positive == 0 or count_negative == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    numerator = float(ranks[positive].sum()) - count_positive * (count_positive + 1) / 2
    return numerator / (count_positive * count_negative)


def _onecut_discrimination(
    records: list[dict[str, Any]], interface: str, false_positive_rate: float
) -> dict[str, Any]:
    selected = [record for record in records if record["interface"] == interface]
    if not selected:
        return {}
    scores = np.concatenate([record["score"] for record in selected])
    errors = np.concatenate([record["error"] for record in selected]).astype(bool)
    correct_scores = scores[~errors]
    if len(correct_scores) == 0:
        threshold = float("nan")
        flagged = np.zeros_like(errors)
    else:
        threshold = float(np.quantile(correct_scores, 1.0 - false_positive_rate, method="higher"))
        flagged = scores >= threshold
    error_count = int(errors.sum())
    flagged_count = int(flagged.sum())
    joint = int(np.count_nonzero(errors & flagged))
    actual_fpr = float(np.mean(flagged[~errors])) if np.any(~errors) else float("nan")
    precision = joint / flagged_count if flagged_count else float("nan")
    prevalence = float(errors.mean())
    return {
        "case_count": len(selected),
        "ray_count": int(len(scores)),
        "error_count": error_count,
        "score": "literal_normalized_negative_log_truth",
        "threshold_from_correct_rays": threshold,
        "target_false_positive_rate": false_positive_rate,
        "actual_false_positive_rate": actual_fpr,
        "roc_auc": _rank_auc(scores, errors),
        "error_coverage": joint / error_count if error_count else float("nan"),
        "error_precision": precision,
        "precision_lift": precision / prevalence if prevalence > 0 and math.isfinite(precision) else float("nan"),
        "flagged_ray_count": flagged_count,
        "error_flagged_count": joint,
        "mean_error_score": float(scores[errors].mean()) if error_count else float("nan"),
        "mean_correct_score": float(scores[~errors].mean()) if np.any(~errors) else float("nan"),
    }


def _decision(
    onecut_outer: dict[str, Any], repair: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    reasons: list[str] = []
    failures: list[str] = []
    coverage = onecut_outer.get("error_coverage")
    lift = onecut_outer.get("precision_lift")
    auc = onecut_outer.get("roc_auc")
    if not isinstance(coverage, float) or not math.isfinite(coverage):
        failures.append("Error coverage is undefined; no usable erroneous rays were observed.")
    elif coverage < args.min_error_coverage:
        failures.append(
            f"One-cut error coverage {coverage:.3f} is below the pre-specified screening threshold "
            f"{args.min_error_coverage:.3f}."
        )
    else:
        reasons.append(f"High one-cut loss covers {coverage:.1%} of erroneous outer rays at the fixed FPR.")
    if not isinstance(auc, float) or not math.isfinite(auc):
        failures.append("One-cut ROC AUC is undefined.")
    elif auc < args.min_onecut_auc:
        failures.append(f"One-cut ROC AUC {auc:.3f} is below {args.min_onecut_auc:.3f}.")
    else:
        reasons.append(f"One-cut loss separates erroneous from correct rays with ROC AUC {auc:.3f}.")
    if isinstance(lift, float) and math.isfinite(lift):
        if lift < args.min_precision_lift:
            failures.append(
                f"One-cut precision lift {lift:.3f} is below {args.min_precision_lift:.3f}."
            )
        else:
            reasons.append(f"High one-cut loss is enriched for errors by {lift:.2f}x.")
    comparison = repair.get("comparisons", {}).get(
        "dice_plus_onecut_outer_vs_dice_only"
    )
    if comparison is None:
        failures.append("The matched Dice-plus-one-cut repair comparison is missing.")
    else:
        surface = comparison["surface_dice_1mm"]
        fraction = surface["positive_case_fraction"]
        mean_delta = surface["mean_delta"]
        if mean_delta < 0:
            failures.append(
                f"Matched one-cut repair reduces mean 1-mm surface Dice by {abs(mean_delta):.6f}."
            )
        elif fraction < args.min_positive_case_fraction:
            failures.append(
                f"Only {fraction:.1%} of cases improve in 1-mm surface Dice after matched one-cut repair."
            )
        else:
            reasons.append(
                f"Matched one-cut repair improves 1-mm surface Dice in {fraction:.1%} of cases."
            )
    if failures:
        # Absence of hard-label movement is not evidence of harm: call it
        # inconclusive unless observability or an actual metric is negative.
        unchanged_only = all("Only" in item or "missing" in item for item in failures)
        verdict = "INCONCLUSIVE" if unchanged_only else "NO_GO"
    else:
        verdict = "GO_TO_SHORT_PILOT"
    return {"verdict": verdict, "supporting_reasons": reasons, "failed_gates": failures}


def _report(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    outer = payload["ray_summary"]["outer"]
    onecut = payload["onecut_discrimination"]["outer"]
    repair = payload["repair_summary"]["comparisons"].get(
        "dice_plus_onecut_outer_vs_dice_only", {}
    )
    lines = [
        "# Surface-normal one-cut and ordinal LogLTN pre-training audit",
        "",
        f"**Decision: {decision['verdict']}**",
        "",
        "## Outer-ray observability",
        "",
        f"- rays: {outer['ray_count']}",
        f"- erroneous rays: {outer['error_count']}",
        f"- ordinal violation rate: {outer['violation_rate']:.3f}",
        f"- error coverage: {outer['error_coverage']:.3f}" if outer["error_coverage"] is not None else "- error coverage: undefined",
        f"- error precision: {outer['error_precision']:.3f}" if outer["error_precision"] is not None else "- error precision: undefined",
        f"- precision lift: {outer['precision_lift']:.3f}" if outer["precision_lift"] is not None else "- precision lift: undefined",
        "",
        "The critical quantity is error coverage: a steep displaced boundary may satisfy the ordinal relation and remain invisible to this loss.",
        "",
        "## One-cut observability",
        "",
        f"- ROC AUC: {onecut['roc_auc']:.3f}",
        f"- target/actual correct-ray false-positive rate: {onecut['target_false_positive_rate']:.3f} / {onecut['actual_false_positive_rate']:.3f}",
        f"- error coverage at that threshold: {onecut['error_coverage']:.3f}",
        f"- error precision: {onecut['error_precision']:.3f}",
        f"- precision lift: {onecut['precision_lift']:.3f}",
        "",
        "## Matched frozen-logit repair",
        "",
    ]
    if repair:
        surface = repair["surface_dice_1mm"]
        lines.extend(
            [
                f"- mean incremental 1-mm surface Dice from one-cut: {surface['mean_delta']:+.6f}",
                f"- positive-case fraction: {surface['positive_case_fraction']:.3f}",
                f"- unchanged-case fraction: {surface['unchanged_case_fraction']:.3f}",
            ]
        )
    else:
        lines.append("- comparison unavailable")
    lines.extend(["", "## Decision evidence", ""])
    lines.extend(f"- {reason}" for reason in decision["supporting_reasons"])
    lines.extend(f"- FAILED: {reason}" for reason in decision["failed_gates"])
    recommendations = payload["interface_recommendations"]
    lines.extend(
        [
            "",
            "## Interface-specific recommendation",
            "",
            f"- outer: {recommendations['outer']['recommendation']} — {recommendations['outer']['reason']}",
            f"- A/P: {recommendations['ap']['recommendation']} — {recommendations['ap']['reason']}",
        ]
    )
    versus_bands = payload["repair_summary"]["comparisons"].get(
        "dice_plus_onecut_outer_vs_dice_plus_bands"
    )
    if versus_bands:
        lines.extend(
            [
                "",
                "## One-cut versus the existing band comparator",
                "",
                f"- union-Dice mean difference: {versus_bands['union_dice']['mean_delta']:+.6f}",
                f"- 1-mm surface-Dice mean difference: {versus_bands['surface_dice_1mm']['mean_delta']:+.6f}",
                f"- ASSD mean difference (negative is better): {versus_bands['assd_mm']['mean_delta']:+.6f} mm",
            ]
        )
    lines.extend(
        [
            "",
            "This is a screening decision, not a performance claim. A GO authorizes only a matched short pilot; it does not authorize a full training campaign.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.radius_mm < args.delta_mm or args.ray_step_mm <= 0:
        raise ValueError("radius-mm must cover delta-mm and ray-step-mm must be positive.")
    if args.margin < 0 or not math.isfinite(args.margin):
        raise ValueError("margin must be finite and non-negative.")
    if not 0.0 < args.onecut_false_positive_rate < 1.0:
        raise ValueError("onecut-false-positive-rate must lie strictly between zero and one.")
    if not 0.0 <= args.min_onecut_auc <= 1.0:
        raise ValueError("min-onecut-auc must lie between zero and one.")
    paths = sorted(args.input_dir.glob("*.npz"))
    if args.max_cases is not None:
        paths = paths[: args.max_cases]
    if not paths:
        raise FileNotFoundError(f"No .npz case bundles found in {args.input_dir}.")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    ray_dir = output / "rays"
    ray_dir.mkdir(exist_ok=True)
    device = _device(args.device)
    offsets = np.arange(-args.radius_mm, args.radius_mm + args.ray_step_mm * 0.5, args.ray_step_mm)
    offsets = np.unique(np.concatenate((offsets, [-args.delta_mm, 0.0, args.delta_mm])))
    ray_rows: list[dict[str, Any]] = []
    onecut_records: list[dict[str, Any]] = []
    repair_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    bundle_records: list[dict[str, str]] = []
    for case_index, path in enumerate(paths):
        case = load_case_bundle(path)
        bundle_records.append({"case_name": case.case_name, "path": str(path.resolve()), "sha256": sha256(path)})
        logits = torch.from_numpy(case.logits).to(device)
        try:
            outer = build_surface_samples(
                case.labels,
                case.spacing,
                interface=Interface.OUTER,
                radius_mm=args.radius_mm,
                max_points=args.max_surface_points,
                seed=args.seed + case_index,
            )
            samples_by_interface = {"outer": outer}
            if args.interfaces == "outer+ap":
                try:
                    samples_by_interface["ap"] = build_surface_samples(
                        case.labels,
                        case.spacing,
                        interface=Interface.ANTERIOR_POSTERIOR,
                        radius_mm=args.radius_mm,
                        max_points=args.max_surface_points,
                        seed=args.seed + case_index,
                    )
                except ValueError as error:
                    skipped.append({"case_name": case.case_name, "stage": "ap_geometry", "reason": str(error)})
            for interface, samples in samples_by_interface.items():
                rays = analyse_rays(
                    logits,
                    samples,
                    offsets_mm=offsets,
                    delta_mm=args.delta_mm,
                    margin=args.margin,
                    temperature=args.temperature,
                    localization_tolerance_mm=args.localization_tolerance_mm,
                )
                summary = rays.summary()
                summary.update(
                    {
                        "case_name": case.case_name,
                        "interface": interface,
                        "error_violation_count": int(np.count_nonzero(rays.error & rays.violation)),
                    }
                )
                ray_rows.append(summary)
                onecut_records.append(
                    {"case_name": case.case_name, "interface": interface, "score": rays.onecut_loss, "error": rays.error}
                )
                np.savez_compressed(
                    ray_dir / f"{case.case_name}_{interface}.npz",
                    points_voxel=samples.points_voxel,
                    normals_physical=samples.normals_physical,
                    offsets_mm=rays.offsets_mm,
                    values=rays.values.astype(np.float32),
                    pair_margin=rays.pair_margin.astype(np.float32),
                    violation=rays.violation,
                    category_code=rays.category_code,
                    nearest_crossing_mm=rays.nearest_crossing_mm.astype(np.float32),
                    crossing_count=rays.crossing_count,
                    onecut_loss=rays.onecut_loss.astype(np.float32),
                    onecut_truth=rays.onecut_truth.astype(np.float32),
                    onecut_allowed_mass=rays.onecut_allowed_mass.astype(np.float32),
                    onecut_best_cut_mm=rays.onecut_best_cut_mm.astype(np.float32),
                    category_names=np.asarray(CATEGORY_NAMES),
                )
            counterfactual = run_counterfactual(
                logits,
                case.labels,
                case.spacing,
                outer,
                ap_samples=samples_by_interface.get("ap"),
                offsets_mm=offsets,
                delta_mm=args.delta_mm,
                cut_tolerance_mm=args.localization_tolerance_mm,
                margin=args.margin,
                temperature=args.temperature,
                target_aux_gradient_ratio=args.target_aux_gradient_ratio,
                update_rms=args.repair_update_rms,
            )
            for arm, metrics in counterfactual.metrics_by_arm.items():
                repair_rows.append({"case_name": case.case_name, "arm": arm, **metrics})
            gradient_rows.append({"case_name": case.case_name, **counterfactual.diagnostics})
        except ValueError as error:
            skipped.append({"case_name": case.case_name, "stage": "outer_audit", "reason": str(error)})

    if not any(row["interface"] == "outer" for row in ray_rows):
        raise RuntimeError("Every case was invalid; see the skipped cases in the partial output.")
    ray_summary = {interface: _aggregate_ray(ray_rows, interface) for interface in ("outer", "ap") if any(row["interface"] == interface for row in ray_rows)}
    onecut_summary = {
        interface: _onecut_discrimination(
            onecut_records, interface, args.onecut_false_positive_rate
        )
        for interface in ("outer", "ap")
        if any(record["interface"] == interface for record in onecut_records)
    }
    repair_summary = _repair_summary(repair_rows)
    decision = _decision(onecut_summary["outer"], repair_summary, args)
    recommendations = _interface_recommendations(decision, repair_summary)
    config = {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    payload = _finite(
        {
            "schema_version": 1,
            "config": config,
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "device": str(device),
            },
            "case_bundles": bundle_records,
            "valid_case_count": len({row["case_name"] for row in ray_rows if row["interface"] == "outer"}),
            "skipped": skipped,
            "ray_summary": ray_summary,
            "onecut_discrimination": onecut_summary,
            "repair_summary": repair_summary,
            "decision": decision,
            "interface_recommendations": recommendations,
        }
    )
    _write_csv(output / "ray_case_summary.csv", [{**row, "categories": json.dumps(row["categories"], sort_keys=True)} for row in ray_rows])
    _write_csv(output / "counterfactual_metrics.csv", repair_rows)
    _write_csv(output / "gradient_diagnostics.csv", gradient_rows)
    write_json(output / "audit_summary.json", payload)
    (output / "REPORT.md").write_text(_report(payload), encoding="utf-8")
    print(json.dumps({"decision": decision, "output_dir": str(output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
