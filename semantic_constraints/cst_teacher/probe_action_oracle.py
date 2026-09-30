#!/usr/bin/env python3
"""Bound the utility of foreground-logit shifts on QC-selected slices.

Oracle variants explicitly read labels and are diagnostic upper bounds only.
They must never be used as deployable correction or reported as model gains.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import uniform_positions
from .evaluate_predictions import prediction_map
from .probe_error_direction import load_qc_scores
from .slice_qc import load_patient_features


STRENGTHS = (-1.0, -0.5, -0.25, 0.25, 0.5, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", type=Path, required=True)
    parser.add_argument("--splits-json", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--inference-dir", type=Path, required=True)
    parser.add_argument("--reference-features", type=Path, required=True)
    parser.add_argument("--error-types", type=Path, required=True)
    parser.add_argument("--qc-oof-csv", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def biased_prediction(probabilities: np.ndarray, strength: float) -> np.ndarray:
    """Shift foreground-vs-background log odds; retain anterior/posterior odds."""
    if probabilities.shape[0] != 3:
        raise ValueError("expected three class probabilities")
    weights = np.asarray((1.0, np.exp(strength), np.exp(strength)), dtype=np.float32)
    return (probabilities * weights[:, None, None]).argmax(axis=0).astype(np.int8)


def foreground_dice(reference: np.ndarray, predicted: np.ndarray) -> float:
    scores = []
    for label in (1, 2):
        real = reference == label
        guess = predicted == label
        scores.append((2 * np.count_nonzero(real & guess) + 1e-8) / (real.sum() + guess.sum() + 1e-8))
    return float(np.mean(scores))


def bootstrap_gain(values: np.ndarray, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    samples = [values[rng.integers(0, len(values), size=len(values))].mean() for _ in range(1000)]
    return {
        "mean": float(values.mean()),
        "lower95": float(np.quantile(samples, 0.025)),
        "upper95": float(np.quantile(samples, 0.975)),
        "harmed_patients": int(np.sum(values < -1e-8)),
        "improved_patients": int(np.sum(values > 1e-8)),
    }


def evaluate_case(
    labels: np.ndarray,
    probabilities: np.ndarray,
    positions: np.ndarray,
    selected: np.ndarray,
    error_types: np.ndarray,
) -> dict[str, float | int]:
    if probabilities.shape != (3, *labels.shape) or len(positions) != len(selected) or len(selected) != len(error_types):
        raise ValueError("case labels, probabilities, slice selection, and targets must align")
    original = probabilities.argmax(axis=0).astype(np.int8)
    signed = original.copy()
    oracle = original.copy()
    actionable = 0
    oracle_better = 0
    selected_count = 0
    for index, position in enumerate(positions):
        if not selected[index]:
            continue
        selected_count += 1
        plane_probs = probabilities[:, :, position, :]
        truth = labels[:, position, :]
        if error_types[index] in (1, 2):
            actionable += 1
            direction = 0.5 if error_types[index] == 1 else -0.5
            signed[:, position, :] = biased_prediction(plane_probs, direction)
        best_score = foreground_dice(truth, original[:, position, :])
        best_prediction = original[:, position, :]
        for strength in STRENGTHS:
            candidate = biased_prediction(plane_probs, strength)
            score = foreground_dice(truth, candidate)
            if score > best_score + 1e-9:
                best_score, best_prediction = score, candidate
        if not np.array_equal(best_prediction, original[:, position, :]):
            oracle_better += 1
            oracle[:, position, :] = best_prediction
    baseline = foreground_dice(labels, original)
    return {
        "selected_slices": selected_count,
        "selected_directional_slices": actionable,
        "slices_with_beneficial_oracle_action": oracle_better,
        "baseline_dice": baseline,
        "perfect_sign_fixed_strength_dice_gain": foreground_dice(labels, signed) - baseline,
        "best_per_slice_oracle_dice_gain": foreground_dice(labels, oracle) - baseline,
        "perfect_sign_changed_voxels": int(np.count_nonzero(signed != original)),
        "oracle_changed_voxels": int(np.count_nonzero(oracle != original)),
    }


def main() -> None:
    args = parse_args()
    from .train_teacher import build_loaders

    reference = load_patient_features(args.reference_features, require_target=True)
    with np.load(args.error_types, allow_pickle=False) as target_bank:
        names = np.asarray(target_bank["case_names"]).astype(str)
        errors = np.asarray(target_bank["error_type"], dtype=np.int8)
    if not np.array_equal(names, reference.names) or errors.shape != reference.target.shape:
        raise ValueError("target bank is not aligned with feature bank")
    selections = []
    for path in args.qc_oof_csv:
        scores = load_qc_scores(path, names, reference.slices, args.fold)
        top_count = int(np.ceil(scores.size * 0.20))
        mask = np.zeros(scores.size, dtype=bool)
        mask[np.argpartition(scores.ravel(), -top_count)[-top_count:]] = True
        selections.append(mask.reshape(scores.shape))
    loader_args = argparse.Namespace(
        pkl=args.pkl, splits_json=args.splits_json, fold=args.fold,
        spatial_size=(64, 64, 64), set_size=reference.slices, slab_depth=3,
        inplane_size=32, cache_dataset=False, max_train_cases=0,
        max_val_cases=0, batch_size=16, num_workers=0,
    )
    _, loader = build_loaders(loader_args)
    paths = prediction_map(args.inference_dir)
    per_seed = [[] for _ in selections]
    for patient in range(len(loader.dataset.base_dataset)):
        item = loader.dataset.base_dataset[patient]
        name = str(item["case_name"])
        if name != names[patient]:
            raise ValueError("validation patient order changed")
        labels = np.asarray(item["label"])[0].astype(np.int8)
        probabilities = np.load(paths[name], allow_pickle=False)
        positions = uniform_positions(labels.shape[1], reference.slices).numpy()
        for seed, selected in enumerate(selections):
            result = evaluate_case(labels, probabilities, positions, selected[patient], errors[patient])
            per_seed[seed].append({"case_name": name, **result})
    report = {
        "schema": "semantic_constraints.cst_teacher.foreground_bias_oracle.v1",
        "fold": args.fold,
        "strengths": STRENGTHS,
        "warning": "All action choices use ground-truth masks; oracle gains are upper bounds, not deployable improvements.",
        "seeds": [],
    }
    for seed, patients in enumerate(per_seed):
        sign_gain = np.asarray([item["perfect_sign_fixed_strength_dice_gain"] for item in patients])
        oracle_gain = np.asarray([item["best_per_slice_oracle_dice_gain"] for item in patients])
        report["seeds"].append({
            "seed": seed,
            "patients": patients,
            "selected_slices": int(sum(item["selected_slices"] for item in patients)),
            "selected_directional_slices": int(sum(item["selected_directional_slices"] for item in patients)),
            "slices_with_beneficial_oracle_action": int(sum(item["slices_with_beneficial_oracle_action"] for item in patients)),
            "perfect_sign_fixed_strength_gain": bootstrap_gain(sign_gain, 20260921 + args.fold * 10 + seed),
            "best_per_slice_oracle_gain": bootstrap_gain(oracle_gain, 20260931 + args.fold * 10 + seed),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"fold": args.fold, "seeds": [
        {key: value for key, value in item.items() if key != "patients"} for item in report["seeds"]
    ]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
