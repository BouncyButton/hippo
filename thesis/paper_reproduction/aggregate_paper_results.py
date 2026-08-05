#!/usr/bin/env python3
"""Aggregate completed paper-reproduction folds into Table I and Table II reports."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def summarize(values: list[float]) -> dict[str, float | int]:
    return {"mean": float(np.mean(values)), "std_population": float(np.std(values, ddof=0)), "n": len(values)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true", help="Allow incomplete fold cells; never use for a paper table.")
    args = parser.parse_args()

    groups: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    manifests: set[str] = set()
    for config_path in sorted(args.runs_root.glob("*/config.json")):
        if config_path.parent.name.startswith("epsilon_"):
            continue
        metrics_path = config_path.parent / "final_metrics.json"
        if not metrics_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        run = config["run"]
        if (
            float(run.get("volume_epsilon", 5000.0)) != 5000.0
            or run.get("volume_grounding", "paper-hard") != "paper-hard"
        ):
            continue
        manifest_path = config_path.parent / "dataset_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing dataset manifest for {config_path.parent}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["manifest_sha256"]
        manifests.add(manifest)
        groups[(float(run["train_fraction"]), run["method"])].append(
            {"config": config, "metrics": metrics, "manifest": manifest}
        )

    if len(manifests) > 1:
        raise RuntimeError("Refusing to aggregate runs made from different Decathlon data manifests.")
    common_protocols = []
    environments = []
    for records in groups.values():
        for item in records:
            run = item["config"]["run"]
            common_protocols.append({key: value for key, value in run.items() if key not in {"fold", "method", "train_fraction"}})
            environment_path = Path(item["config"]["args"]["output_dir"]) / "environment.json"
            if not environment_path.is_file():
                raise FileNotFoundError(f"Missing environment metadata: {environment_path}")
            environments.append(json.loads(environment_path.read_text(encoding="utf-8")))
    if common_protocols and any(value != common_protocols[0] for value in common_protocols[1:]):
        raise RuntimeError("Refusing to aggregate runs with different seed/epoch/batch/optimizer/model protocols.")
    if environments and any(value != environments[0] for value in environments[1:]):
        raise RuntimeError("Refusing to aggregate runs from different software or hardware environments.")
    expected_keys = {(fraction, method) for fraction in (1.0, 0.25, 0.05) for method in ("baseline", "ltn")}
    if not args.allow_partial and set(groups) != expected_keys:
        missing = sorted(expected_keys - set(groups))
        raise RuntimeError(f"Incomplete experiment matrix; missing cells: {missing}")
    for key, records in groups.items():
        folds = [int(item["config"]["run"]["fold"]) for item in records]
        if len(folds) != len(set(folds)):
            raise RuntimeError(f"Duplicate folds in {key}: {folds}")
        if not args.allow_partial and sorted(folds) != [1, 2, 3, 4, 5]:
            raise RuntimeError(f"Expected folds 1..5 in {key}, found {folds}")
        protocol = [{k: v for k, v in item["config"]["run"].items() if k not in {"fold", "method"}} for item in records]
        if any(value != protocol[0] for value in protocol[1:]):
            raise RuntimeError(f"Mixed protocol configuration in {key}")

    table_i: dict[str, Any] = {}
    table_ii: dict[str, Any] = {}
    for (fraction, method), records in sorted(groups.items()):
        key = f"fraction_{fraction:g}_{method}"
        table_i[key] = {
            "fraction": fraction,
            "method": method,
            "folds": sorted(int(item["config"]["run"]["fold"]) for item in records),
            "dice_all_classes": summarize([item["metrics"]["dice_all_classes"] for item in records]),
            "dice_foreground": summarize([item["metrics"]["dice_foreground"] for item in records]),
        }
    ground_truth_by_fold: dict[int, dict[str, float]] = {}
    for method in ("baseline", "ltn"):
        for item in groups.get((1.0, method), []):
            fold = int(item["config"]["run"]["fold"])
            current = {name: float(item["metrics"][name]) for name in ("ground_truth_connectedness", "ground_truth_nested", "ground_truth_volume_similarity")}
            previous = ground_truth_by_fold.get(fold)
            if previous is not None and any(abs(current[name] - previous[name]) > 1e-10 for name in current):
                raise RuntimeError(f"Ground-truth structural metrics differ across methods for fold {fold}")
            ground_truth_by_fold[fold] = current
    if ground_truth_by_fold:
        table_ii["ground_truth"] = {
            name: summarize([value[name] for value in ground_truth_by_fold.values()])
            for name in ("ground_truth_connectedness", "ground_truth_nested", "ground_truth_volume_similarity")
        }
    for method in ("baseline", "ltn"):
        records = groups.get((1.0, method), [])
        if records:
            table_ii[method] = {
                name: summarize([item["metrics"][f"prediction_{name}"] for item in records])
                for name in ("connectedness", "nested", "volume_similarity")
            }

    payload = {"table_i": table_i, "table_ii_full_fraction": table_ii}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = ["# Paper reproduction aggregate", "", "## Table I", "", "| Fraction | Method | Dice (all classes) | Dice (foreground) | Folds |", "| ---: | --- | ---: | ---: | --- |"]
    for row in table_i.values():
        all_dice = row["dice_all_classes"]
        foreground = row["dice_foreground"]
        lines.append(
            f"| {row['fraction']:g} | {row['method']} | {all_dice['mean']:.4f} ± {all_dice['std_population']:.4f} | "
            f"{foreground['mean']:.4f} ± {foreground['std_population']:.4f} | {row['folds']} |"
        )
    lines.extend(["", "## Table II — full-fraction structural metrics", ""])
    for key, row in table_ii.items():
        lines.extend([f"### {key}", ""])
        for metric, summary in row.items():
            lines.append(f"- {metric}: `{summary['mean']:.4f} ± {summary['std_population']:.4f}` (n={summary['n']})")
        lines.append("")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
