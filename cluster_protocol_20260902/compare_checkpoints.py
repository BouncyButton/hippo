#!/usr/bin/env python3
"""Paired decoded-boundary comparison for the matched one-cut pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import mean, median


REPO_ROOT = Path.cwd().resolve()
if not (REPO_ROOT / "thesis" / "Surface-normal ordinal LogLTN").is_dir():
    REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = REPO_ROOT / "thesis" / "Surface-normal ordinal LogLTN"
sys.path.insert(0, str(AUDIT_ROOT))

from surface_normal_ordinal.io import load_case_bundle  # noqa: E402
from surface_normal_ordinal.metrics import segmentation_metrics  # noqa: E402


FLOAT_METRICS = (
    "union_dice",
    "anterior_dice",
    "posterior_dice",
    "surface_dice_1mm",
    "surface_dice_2mm",
    "assd_mm",
    "hd95_mm",
)
COUNT_METRICS = (
    "foreground_fp",
    "foreground_fn",
    "ap_swaps",
    "predicted_foreground_voxels",
    "predicted_components",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-bundles", type=Path, required=True)
    parser.add_argument("--onecut-bundles", type=Path, required=True)
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--onecut-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object in {path}.")
    return payload


def summarize(values: list[float]) -> dict[str, float]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Comparison metrics must be finite and nonempty.")
    return {
        "mean": mean(values),
        "median": median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifests = {
        "control": read_json(args.control_bundles / "manifest.json"),
        "onecut": read_json(args.onecut_bundles / "manifest.json"),
    }
    for key in ("dataset", "fold", "split", "spatial_size", "resize", "amp", "pkl_sha256", "splits_json_sha256"):
        if manifests["control"].get(key) != manifests["onecut"].get(key):
            raise ValueError(f"Bundle manifests disagree on {key}.")
    paths = {
        "control": {path.stem: path for path in args.control_bundles.glob("*.npz")},
        "onecut": {path.stem: path for path in args.onecut_bundles.glob("*.npz")},
    }
    if not paths["control"] or paths["control"].keys() != paths["onecut"].keys():
        raise ValueError("Control and one-cut bundle case sets must match and be nonempty.")

    rows: list[dict[str, object]] = []
    for case_name in sorted(paths["control"]):
        control = load_case_bundle(paths["control"][case_name])
        onecut = load_case_bundle(paths["onecut"][case_name])
        if (
            control.case_name != onecut.case_name
            or control.labels.shape != onecut.labels.shape
            or control.spacing != onecut.spacing
            or not (control.labels == onecut.labels).all()
        ):
            raise ValueError(f"Ground-truth bundle mismatch for {case_name}.")
        control_metrics = segmentation_metrics(
            control.labels, control.logits.argmax(axis=0), control.spacing
        )
        onecut_metrics = segmentation_metrics(
            onecut.labels, onecut.logits.argmax(axis=0), onecut.spacing
        )
        row: dict[str, object] = {"case_name": case_name}
        for metric in (*FLOAT_METRICS, *COUNT_METRICS):
            control_value = control_metrics[metric]
            onecut_value = onecut_metrics[metric]
            row[f"control_{metric}"] = control_value
            row[f"onecut_{metric}"] = onecut_value
            row[f"delta_{metric}"] = onecut_value - control_value
        rows.append(row)

    comparisons: dict[str, dict[str, object]] = {}
    for metric in FLOAT_METRICS:
        control_values = [float(row[f"control_{metric}"]) for row in rows]
        onecut_values = [float(row[f"onecut_{metric}"]) for row in rows]
        deltas = [float(row[f"delta_{metric}"]) for row in rows]
        comparisons[metric] = {
            "control": summarize(control_values),
            "onecut": summarize(onecut_values),
            "delta_onecut_minus_control": summarize(deltas),
            "onecut_higher_cases": sum(value > 0 for value in deltas),
            "onecut_lower_cases": sum(value < 0 for value in deltas),
            "tied_cases": sum(value == 0 for value in deltas),
        }
    for metric in COUNT_METRICS:
        control_total = sum(int(row[f"control_{metric}"]) for row in rows)
        onecut_total = sum(int(row[f"onecut_{metric}"]) for row in rows)
        comparisons[metric] = {
            "control_total": control_total,
            "onecut_total": onecut_total,
            "delta_onecut_minus_control": onecut_total - control_total,
        }

    run_metrics = {
        "control": read_json(args.control_run / "final_metrics.json"),
        "onecut": read_json(args.onecut_run / "final_metrics.json"),
    }
    best_control = float(run_metrics["control"]["best_val_dice_hard"])
    best_onecut = float(run_metrics["onecut"]["best_val_dice_hard"])
    onecut_skipped_keys = [
        key
        for key in run_metrics["onecut"]
        if "outer_onecut_skipped_patient" in key and key.endswith("_mean")
    ]
    skipped_anomaly = any(float(run_metrics["onecut"][key]) > 0 for key in onecut_skipped_keys)
    surface_delta = float(
        comparisons["surface_dice_1mm"]["delta_onecut_minus_control"]["mean"]
    )
    gates = {
        "best_hard_dice_within_0.001": best_onecut >= best_control - 0.001,
        "mean_surface_dice_1mm_improved": surface_delta > 0,
        "no_skipped_patient_anomaly": not skipped_anomaly,
    }
    gates["advance_to_longer_run"] = all(gates.values())

    fieldnames = list(rows[0])
    with (args.output_dir / "paired_case_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema_version": 1,
        "case_count": len(rows),
        "manifests": {
            name: {
                "path": str((directory / "manifest.json").resolve()),
                "sha256": sha256(directory / "manifest.json"),
                "checkpoint": manifests[name]["checkpoint"],
                "checkpoint_sha256": manifests[name]["checkpoint_sha256"],
            }
            for name, directory in (
                ("control", args.control_bundles),
                ("onecut", args.onecut_bundles),
            )
        },
        "best_val_dice_hard": {
            "control": best_control,
            "onecut": best_onecut,
            "delta_onecut_minus_control": best_onecut - best_control,
        },
        "comparisons": comparisons,
        "gates": gates,
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    direction = "PASS" if gates["advance_to_longer_run"] else "DO NOT ADVANCE"
    lines = [
        "# Matched outer one-cut pilot comparison",
        "",
        f"Decision: **{direction}**",
        "",
        f"- Cases: {len(rows)}",
        f"- Best hard Dice delta: {best_onecut - best_control:+.6f}",
        f"- Mean union Dice delta: {comparisons['union_dice']['delta_onecut_minus_control']['mean']:+.6f}",
        f"- Mean 1-mm surface Dice delta: {surface_delta:+.6f}",
        f"- Mean 2-mm surface Dice delta: {comparisons['surface_dice_2mm']['delta_onecut_minus_control']['mean']:+.6f}",
        f"- Mean ASSD delta: {comparisons['assd_mm']['delta_onecut_minus_control']['mean']:+.6f} mm",
        f"- Mean HD95 delta: {comparisons['hd95_mm']['delta_onecut_minus_control']['mean']:+.6f} mm",
        f"- Foreground FP delta: {comparisons['foreground_fp']['delta_onecut_minus_control']:+d}",
        f"- Foreground FN delta: {comparisons['foreground_fn']['delta_onecut_minus_control']:+d}",
        f"- A/P swap delta: {comparisons['ap_swaps']['delta_onecut_minus_control']:+d}",
        "",
        "## Gates",
        "",
        *[f"- {'PASS' if passed else 'FAIL'}: `{name}`" for name, passed in gates.items() if name != "advance_to_longer_run"],
        "",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"decision": direction, "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
