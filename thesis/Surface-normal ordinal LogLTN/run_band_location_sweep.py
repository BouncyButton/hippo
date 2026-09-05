#!/usr/bin/env python3
"""Training-only frozen-logit sweep of a BCE-band plus cut-location residual."""

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

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_hybrid_feasibility as common  # noqa: E402
from surface_normal_ordinal.io import load_case_bundle, sha256, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--location-shares", type=float, nargs="+", default=(0.05, 0.10, 0.20, 0.30)
    )
    parser.add_argument("--radius-mm", type=float, default=3.0)
    parser.add_argument("--ray-step-mm", type=float, default=0.5)
    parser.add_argument("--tolerance-mm", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-surface-points", type=int, default=4096)
    parser.add_argument("--max-cases", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--target-aux-gradient-ratio", type=float, default=0.10)
    parser.add_argument("--high-gradient-cap-ratio", type=float, default=0.50)
    parser.add_argument(
        "--repair-update-rms", type=float, nargs="+", default=(0.02, 0.05, 0.10)
    )
    parser.add_argument("--max-union-dice-drop", type=float, default=0.0005)
    parser.add_argument("--minimum-passing-update-count", type=int, default=2)
    return parser.parse_args()


def _label(share: float) -> str:
    return f"rho_{int(round(100 * share)):02d}"


def _finite_row(row: dict[str, Any]) -> bool:
    return all(
        math.isfinite(float(value))
        for key, value in row.items()
        if key != "case_name"
    )


def _first_pass(
    paths: list[Path], args: argparse.Namespace, device: torch.device
) -> tuple[list[dict[str, Any]], list[Path], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    valid_paths: list[Path] = []
    skipped: list[dict[str, str]] = []
    for case_index, path in enumerate(paths):
        try:
            case, gradients, details = common._losses_and_gradients(
                path, args, device, case_index
            )
            overlap, conflict, conflict_mass = common._voxel_conflict(
                gradients["band"], gradients["location"]
            )
            row = {
                "case_name": case.case_name,
                **details,
                "band_location_cosine": common._cosine(
                    gradients["band"], gradients["location"]
                ),
                "location_residual_after_band": common._residual_fraction(
                    gradients["band"], gradients["location"]
                ),
                "band_location_voxel_overlap_fraction": overlap,
                "band_location_conflict_fraction": conflict,
                "band_location_conflict_mass_fraction": conflict_mass,
            }
            if not _finite_row(row):
                raise ValueError("Non-finite gradient diagnostic.")
            rows.append(row)
            valid_paths.append(path)
        except (ValueError, RuntimeError) as error:
            skipped.append({"path": str(path), "reason": str(error)})
    return rows, valid_paths, skipped


def _calibration_and_coefficients(
    rows: list[dict[str, Any]], args: argparse.Namespace
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    dice_rms = [float(row["dice_gradient_rms"]) for row in rows]
    band_rms = [float(row["band_gradient_rms"]) for row in rows]
    location_rms = [float(row["location_gradient_rms"]) for row in rows]
    band_median = float(np.median(band_rms))
    location_median = float(np.median(location_rms))
    band_calibration = common._calibrate(
        dice_rms,
        band_rms,
        args.target_aux_gradient_ratio,
        args.high_gradient_cap_ratio,
    )
    calibration: dict[str, Any] = {"band": band_calibration}
    coefficients: dict[str, dict[str, float]] = {
        "band": {"band": band_calibration["selected_weight"]}
    }
    for share in args.location_shares:
        band_scale = (1.0 - share) / band_median
        location_scale = share / location_median
        raw_rms = [
            common._combined_rms(
                float(row["band_gradient_rms"]),
                float(row["location_gradient_rms"]),
                float(row["band_location_cosine"]),
                band_scale,
                location_scale,
            )
            for row in rows
        ]
        name = _label(share)
        calibration[name] = common._calibrate(
            dice_rms,
            raw_rms,
            args.target_aux_gradient_ratio,
            args.high_gradient_cap_ratio,
        )
        scale = float(calibration[name]["selected_weight"])
        coefficients[name] = {
            "band": scale * band_scale,
            "location": scale * location_scale,
        }
    return calibration, coefficients


def _counterfactual(
    paths: list[Path],
    args: argparse.Namespace,
    device: torch.device,
    coefficients: dict[str, dict[str, float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    alignments: list[dict[str, Any]] = []
    for case_index, path in enumerate(paths):
        case, gradients, _ = common._losses_and_gradients(
            path, args, device, case_index
        )
        logits = torch.from_numpy(case.logits).to(device).float()
        candidate_gradients = {
            name: common._candidate_gradient(gradients, values)
            for name, values in coefficients.items()
        }
        alignment: dict[str, Any] = {"case_name": case.case_name}
        for share in args.location_shares:
            name = _label(share)
            alignment[f"{name}_band_alignment"] = common._cosine(
                candidate_gradients[name], gradients["band"]
            )
            alignment[f"{name}_location_alignment"] = common._cosine(
                candidate_gradients[name], gradients["location"]
            )
        alignments.append(alignment)
        baseline = common._metrics(logits, case.labels, case.spacing)
        for update_rms in args.repair_update_rms:
            rows.append(
                {
                    "case_name": case.case_name,
                    "update_rms": update_rms,
                    "arm": "baseline",
                    **baseline,
                }
            )
            dice_only = common._matched_update(
                logits, gradients["dice"], update_rms
            )
            rows.append(
                {
                    "case_name": case.case_name,
                    "update_rms": update_rms,
                    "arm": "dice_only",
                    **common._metrics(dice_only, case.labels, case.spacing),
                }
            )
            for name, auxiliary in candidate_gradients.items():
                updated = common._matched_update(
                    logits, gradients["dice"] + auxiliary, update_rms
                )
                rows.append(
                    {
                        "case_name": case.case_name,
                        "update_rms": update_rms,
                        "arm": f"dice_plus_{name}",
                        **common._metrics(updated, case.labels, case.spacing),
                    }
                )
    return rows, alignments


def _summarize(
    rows: list[dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for update_rms in sorted({float(row["update_rms"]) for row in rows}):
        selected = [row for row in rows if float(row["update_rms"]) == update_rms]
        comparisons = {
            _label(share): common._comparison(
                selected,
                f"dice_plus_{_label(share)}",
                "dice_plus_band",
            )
            for share in args.location_shares
        }
        output[str(update_rms)] = {"comparisons_vs_band": comparisons}
    return output


def _decision(summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    ratio_results: dict[str, Any] = {}
    passing: list[float] = []
    for share in args.location_shares:
        name = _label(share)
        update_gates: dict[str, Any] = {}
        for update, block in summary.items():
            comparison = block["comparisons_vs_band"][name]
            gates = {
                "surface_dice_1mm_improved": comparison["surface_dice_1mm"]["mean"] > 0,
                "union_dice_within_limit": comparison["union_dice"]["mean"]
                >= -args.max_union_dice_drop,
                "assd_not_worse": comparison["assd_mm"]["mean"] <= 0,
                "total_errors_not_worse": comparison["total_errors"]["mean"] <= 0,
            }
            update_gates[update] = {"passed": all(gates.values()), "gates": gates}
        passing_count = sum(value["passed"] for value in update_gates.values())
        ratio_results[name] = {
            "location_share": share,
            "passing_update_count": passing_count,
            "required_passing_update_count": args.minimum_passing_update_count,
            "updates": update_gates,
        }
        if passing_count >= args.minimum_passing_update_count:
            passing.append(share)
    selected = min(passing) if passing else None
    return {
        "verdict": "GO_TO_FIVE_EPOCH_PILOT" if selected is not None else "NO_GO",
        "selected_location_share": selected,
        "selection_rule": "smallest share passing at least the required number of update magnitudes",
        "ratio_results": ratio_results,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _report(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    gradient = payload["gradient_summary"]
    lines = [
        "# Band-dominant cut-location residual sweep",
        "",
        f"Decision: **{decision['verdict']}**",
        "",
        f"- training cases: {gradient['valid_case_count']}",
        f"- skipped cases: {gradient['skipped_case_count']}",
        f"- median band/location cosine: {gradient['band_location_cosine']['median']:+.4f}",
        f"- median location residual after projection on bands: {gradient['location_residual_after_band']['median']:.3f}",
        "",
    ]
    for share in payload["config"]["location_shares"]:
        name = _label(float(share))
        result = decision["ratio_results"][name]
        lines.extend(
            [
                f"## Location share {float(share):.0%}",
                "",
                f"Passing update magnitudes: {result['passing_update_count']}/{len(result['updates'])}",
                "",
            ]
        )
        for update, block in payload["counterfactual_summary"].items():
            comparison = block["comparisons_vs_band"][name]
            gates = result["updates"][update]
            lines.extend(
                [
                    f"- RMS {update} ({'PASS' if gates['passed'] else 'FAIL'}): "
                    f"surface Dice {comparison['surface_dice_1mm']['mean']:+.6f}, "
                    f"union Dice {comparison['union_dice']['mean']:+.6f}, "
                    f"ASSD {comparison['assd_mm']['mean']:+.6f} mm, "
                    f"errors/case {comparison['total_errors']['mean']:+.3f}",
                ]
            )
        lines.append("")
    if decision["selected_location_share"] is None:
        lines.append("No ratio passed the pre-registered robustness gate; do not train this hybrid.")
    else:
        lines.append(
            f"The smallest robust ratio is {decision['selected_location_share']:.0%}; "
            "this authorizes one matched five-epoch pilot only."
        )
    lines.extend(
        [
            "",
            "This sweep uses deterministic fold-0 training cases. It does not make a held-out performance claim.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not args.location_shares or any(
        not 0 < share < 1 or not math.isfinite(share)
        for share in args.location_shares
    ):
        raise ValueError("Every location share must be finite and strictly between 0 and 1.")
    if sorted(set(args.location_shares)) != list(args.location_shares):
        raise ValueError("Location shares must be unique and strictly increasing.")
    if args.minimum_passing_update_count < 1 or args.minimum_passing_update_count > len(
        args.repair_update_rms
    ):
        raise ValueError("The required passing count is incompatible with update magnitudes.")
    paths = sorted(args.input_dir.glob("*.npz"))
    if args.max_cases is not None:
        paths = paths[: args.max_cases]
    if not paths:
        raise FileNotFoundError(f"No bundles found in {args.input_dir}.")
    device = common._device(args.device)
    args.output_dir.mkdir(parents=True)
    gradient_rows, valid_paths, skipped = _first_pass(paths, args, device)
    if len(gradient_rows) != len(paths):
        raise RuntimeError("The pre-registered sweep requires every selected case to be valid.")
    calibration, coefficients = _calibration_and_coefficients(gradient_rows, args)
    metric_rows, alignment_rows = _counterfactual(
        valid_paths, args, device, coefficients
    )
    summary = _summarize(metric_rows, args)
    decision = _decision(summary, args)
    gradient_summary = {
        "valid_case_count": len(gradient_rows),
        "skipped_case_count": len(skipped),
        **{
            field: common._summary([float(row[field]) for row in gradient_rows])
            for field in (
                "band_location_cosine",
                "location_residual_after_band",
                "band_location_voxel_overlap_fraction",
                "band_location_conflict_fraction",
                "band_location_conflict_mass_fraction",
            )
        },
    }
    manifest = args.input_dir / "manifest.json"
    source_files = (
        Path(__file__).resolve(),
        HERE / "run_hybrid_feasibility.py",
        HERE / "surface_normal_ordinal" / "counterfactual.py",
        HERE / "surface_normal_ordinal" / "geometry.py",
        HERE / "surface_normal_ordinal" / "hybrid.py",
        HERE / "surface_normal_ordinal" / "io.py",
        HERE / "surface_normal_ordinal" / "metrics.py",
        HERE / "surface_normal_ordinal" / "objective.py",
    )
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
                str(path.relative_to(HERE)): sha256(path) for path in source_files
            }
        },
        "input": {
            "directory": str(args.input_dir.resolve()),
            "manifest": str(manifest.resolve()),
            "manifest_sha256": sha256(manifest),
            "case_count": len(paths),
        },
        "gradient_summary": gradient_summary,
        "calibration": calibration,
        "candidate_coefficients": coefficients,
        "counterfactual_summary": summary,
        "decision": decision,
        "skipped": skipped,
    }
    _write_csv(args.output_dir / "gradient_diagnostics.csv", gradient_rows)
    _write_csv(args.output_dir / "component_alignment.csv", alignment_rows)
    _write_csv(args.output_dir / "counterfactual_metrics.csv", metric_rows)
    write_json(args.output_dir / "sweep_summary.json", payload)
    (args.output_dir / "REPORT.md").write_text(_report(payload), encoding="utf-8")
    print(json.dumps({"decision": decision, "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
