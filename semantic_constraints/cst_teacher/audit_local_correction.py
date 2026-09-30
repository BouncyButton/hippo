#!/usr/bin/env python3
"""Explain held-out voxel edits from the local corrector (diagnostic only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .local_corrector import LocalResidualCorrector, case_dice
from .run_local_correction_study import load_cases, predict_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", type=Path, required=True)
    parser.add_argument("--splits-json", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--inference-dir", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-val-cases", type=int, default=0)
    return parser.parse_args()


def edit_counts(baseline: np.ndarray, proposal: np.ndarray, target: np.ndarray, confidence: np.ndarray) -> dict:
    if baseline.shape != proposal.shape or baseline.shape != target.shape or baseline.shape != confidence.shape:
        raise ValueError("voxel audit arrays must align")
    changed = baseline != proposal
    corrected = changed & (baseline != target) & (proposal == target)
    introduced = changed & (baseline == target) & (proposal != target)
    wrong_to_wrong = changed & (baseline != target) & (proposal != target)
    bins = {
        "lt_0.6": changed & (confidence < 0.6),
        "0.6_to_0.8": changed & (confidence >= 0.6) & (confidence < 0.8),
        "ge_0.8": changed & (confidence >= 0.8),
    }
    transitions = np.zeros((3, 3), dtype=np.int64)
    for before in range(3):
        for after in range(3):
            transitions[before, after] = np.count_nonzero(changed & (baseline == before) & (proposal == after))
    bin_outcomes = {
        name: {
            "changed": int(mask.sum()),
            "corrected": int(np.count_nonzero(mask & corrected)),
            "introduced": int(np.count_nonzero(mask & introduced)),
            "wrong_to_wrong": int(np.count_nonzero(mask & wrong_to_wrong)),
        }
        for name, mask in bins.items()
    }
    return {
        "changed": int(changed.sum()),
        "corrected": int(corrected.sum()),
        "introduced": int(introduced.sum()),
        "wrong_to_wrong": int(wrong_to_wrong.sum()),
        "confidence_bins": bin_outcomes,
        "transitions": transitions.tolist(),
    }


def main() -> None:
    args = parse_args()
    names, contexts, labels, baselines, _, _ = load_cases(args)
    report = json.loads((args.study_dir / "report.json").read_text(encoding="utf-8"))
    if report["fold"] != args.fold or names.tolist() != next(iter(report["variants"].values()))["case_names"]:
        raise ValueError("study artifact does not match the data being audited")
    index = {name: position for position, name in enumerate(names)}
    output = {"schema": "semantic_constraints.cst_teacher.local_edit_audit.v1", "fold": args.fold, "variants": {}}
    for variant, study in report["variants"].items():
        aggregate = {"changed": 0, "corrected": 0, "introduced": 0, "wrong_to_wrong": 0}
        confidence_bins = {
            name: {key: 0 for key in aggregate}
            for name in ("lt_0.6", "0.6_to_0.8", "ge_0.8")
        }
        transitions = np.zeros((3, 3), dtype=np.int64)
        cases = []
        for split in study["outer_splits"]:
            checkpoint = torch.load(split["model_artifact"], map_location="cpu", weights_only=True)
            if checkpoint["variant"] != variant or checkpoint["fold"] != args.fold:
                raise ValueError("wrong correction artifact")
            test = np.asarray([index[name] for name in split["test_patients"]], dtype=np.int64)
            if set(checkpoint["training_patients"]) & set(split["test_patients"]):
                raise ValueError("training/test patient overlap")
            model = LocalResidualCorrector().to(args.device)
            model.load_state_dict(checkpoint["state_dict"])
            proposed = predict_cases(model, contexts, test, variant=variant, device=args.device, batch_size=args.batch_size)
            for patient in test:
                truth = np.moveaxis(labels[patient], 0, 1)
                confidence = contexts[patient, :, 11:14].astype(np.float32).max(axis=1).transpose(1, 0, 2)
                values = edit_counts(baselines[patient], proposed[int(patient)], truth, confidence)
                values["case_name"] = str(names[patient])
                values["dice_gain"] = case_dice(truth, proposed[int(patient)]) - case_dice(truth, baselines[patient])
                cases.append(values)
                for key in aggregate:
                    aggregate[key] += values[key]
                for bin_name in confidence_bins:
                    for count_name in aggregate:
                        confidence_bins[bin_name][count_name] += values["confidence_bins"][bin_name][count_name]
                transitions += np.asarray(values["transitions"])
        aggregate["beneficial_edit_fraction"] = aggregate["corrected"] / max(1, aggregate["changed"])
        aggregate["new_error_per_corrected_voxel"] = aggregate["introduced"] / max(1, aggregate["corrected"])
        output["variants"][variant] = {"aggregate": aggregate, "confidence_bins": confidence_bins,
                                       "transitions": transitions.tolist(), "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({name: {"aggregate": value["aggregate"], "confidence_bins": value["confidence_bins"],
                             "transitions": value["transitions"]} for name, value in output["variants"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
