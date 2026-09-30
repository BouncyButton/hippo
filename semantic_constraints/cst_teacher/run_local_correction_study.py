#!/usr/bin/env python3
"""Patient-held-out local residual correction with an inner no-harm screen."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .data import uniform_positions
from .evaluate_predictions import prediction_map
from .local_corrector import (
    CST_CHANNELS,
    LocalResidualCorrector,
    apply_slice_policy,
    case_dice,
    correction_loss,
    make_slice_contexts,
)
from .risk_probe import grouped_folds
from .slice_qc import load_patient_features, ridge_fit, ridge_predict


POLICY_FRACTIONS = (0.0, 0.20, 0.40, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", type=Path, required=True)
    parser.add_argument("--splits-json", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--inference-dir", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--max-val-cases", type=int, default=0, help="Smoke-only case limit; zero is full cohort")
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--variants", choices=("image_mask", "cst"), nargs="+", default=("image_mask", "cst"))
    return parser.parse_args()


def load_cases(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load only validation patients; never train on Swin/CST training cases."""
    from .train_teacher import build_loaders

    feature_bank = load_patient_features(args.features, require_target=True)
    names = feature_bank.names[:args.max_val_cases] if args.max_val_cases else feature_bank.names
    build_args = argparse.Namespace(
        pkl=args.pkl, splits_json=args.splits_json, fold=args.fold,
        spatial_size=(64, 64, 64), set_size=feature_bank.slices, slab_depth=3,
        inplane_size=32, cache_dataset=False, max_train_cases=0,
        max_val_cases=args.max_val_cases, batch_size=16, num_workers=0,
    )
    _, loader = build_loaders(build_args)
    paths = prediction_map(args.inference_dir)
    with np.load(args.features, allow_pickle=False) as feature_arrays:
        relationships = np.asarray(feature_arrays["slice_relationship_residuals"], dtype=np.float32)
    relationships = relationships.reshape(len(feature_bank.names), feature_bank.slices, -1)
    if relationships.shape[-1] != 9:
        raise ValueError("expected nine per-slice CST relationship features")
    contexts, labels, baselines, actual_names = [], [], [], []
    for patient in range(len(loader.dataset.base_dataset)):
        item = loader.dataset.base_dataset[patient]
        name = str(item["case_name"])
        if name != names[patient]:
            raise ValueError("validation patient order differs from frozen feature bank")
        actual_names.append(name)
        image = np.asarray(item["image"], dtype=np.float32)
        truth = np.asarray(item["label"])[0].astype(np.uint8)
        probabilities = np.load(paths[name], allow_pickle=False).astype(np.float32)
        positions = uniform_positions(truth.shape[1], feature_bank.slices).numpy()
        # The nine relationship values are residual(2), |residual|(2),
        # Swin profile(2), CST expected profile(2), normalized position(1).
        teacher_profile = relationships[patient, :, 6:8]
        contexts.append(make_slice_contexts(image, probabilities, teacher_profile, positions, use_cst=True))
        labels.append(np.moveaxis(truth, 1, 0))
        baselines.append(probabilities.argmax(axis=0).astype(np.uint8))
    if not np.array_equal(actual_names, names):
        raise ValueError("loaded case names differ from feature-bank names")
    return (
        names,
        np.stack(contexts),
        np.stack(labels),
        np.stack(baselines),
        feature_bank.features["combined"][:len(names)],
        feature_bank.target[:len(names)],
    )


def predict_cases(model: LocalResidualCorrector, contexts: np.ndarray, patients: np.ndarray, *, variant: str, device: str, batch_size: int) -> dict[int, np.ndarray]:
    model.eval()
    predictions = {}
    with torch.no_grad():
        for patient in patients:
            patient = int(patient)
            slices = []
            for start in range(0, contexts.shape[1], batch_size):
                block = torch.from_numpy(contexts[patient, start:start + batch_size].astype(np.float32)).to(device)
                if variant == "image_mask":
                    block[:, CST_CHANNELS] = 0
                logits, _ = model(block)
                slices.append(logits.argmax(dim=1).cpu().numpy().astype(np.uint8))
            predictions[patient] = np.moveaxis(np.concatenate(slices), 0, 1)
    return predictions


def train_model(
    contexts: np.ndarray,
    labels: np.ndarray,
    baselines: np.ndarray,
    inner_train: np.ndarray,
    inner_val: np.ndarray,
    *,
    variant: str,
    device: str,
    batch_size: int,
    max_epochs: int,
    seed: int,
) -> tuple[LocalResidualCorrector, dict]:
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = LocalResidualCorrector().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-3)
    patient_indices = np.repeat(inner_train, contexts.shape[1])
    slice_indices = np.tile(np.arange(contexts.shape[1]), len(inner_train))
    rng = np.random.default_rng(seed)
    best_dice, best_state, best_epoch, stale = -float("inf"), None, 0, 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for selection in np.array_split(rng.permutation(len(patient_indices)), max(1, int(np.ceil(len(patient_indices) / batch_size)))):
            chosen_patients, chosen_slices = patient_indices[selection], slice_indices[selection]
            block = torch.from_numpy(contexts[chosen_patients, chosen_slices].astype(np.float32)).to(device)
            if variant == "image_mask":
                block[:, CST_CHANNELS] = 0
            target = torch.from_numpy(labels[chosen_patients, chosen_slices].astype(np.int64)).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, residual = model(block)
            loss = correction_loss(logits, residual, block[:, 11:14], target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        proposed = predict_cases(model, contexts, inner_val, variant=variant, device=device, batch_size=batch_size)
        validation_dice = float(np.mean([
            case_dice(np.moveaxis(labels[index], 0, 1), proposed[int(index)]) for index in inner_val
        ]))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "inner_validation_dice": validation_dice})
        if validation_dice > best_dice + 1e-5:
            best_dice, best_epoch, stale = validation_dice, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 3:
                break
    if best_state is None:
        raise AssertionError("no corrector checkpoint selected")
    model.load_state_dict(best_state)
    baseline_dice = float(np.mean([
        case_dice(np.moveaxis(labels[index], 0, 1), baselines[index]) for index in inner_val
    ]))
    return model, {
        "best_epoch": best_epoch,
        "best_inner_validation_dice": best_dice,
        "baseline_inner_validation_dice": baseline_dice,
        "history": history,
    }


def risk_full_axis(risk_sampled: np.ndarray, y_size: int) -> np.ndarray:
    positions = uniform_positions(y_size, len(risk_sampled)).numpy()
    return np.interp(np.arange(y_size), positions, risk_sampled).astype(np.float32)


def evaluate_policies(
    patients: np.ndarray,
    predictions: dict[int, np.ndarray],
    labels: np.ndarray,
    baselines: np.ndarray,
    sampled_risk: np.ndarray,
) -> dict[float, np.ndarray]:
    result = {}
    for fraction in POLICY_FRACTIONS:
        values = []
        for patient in patients:
            patient = int(patient)
            truth = np.moveaxis(labels[patient], 0, 1)
            risk = risk_full_axis(sampled_risk[patient], truth.shape[1])
            candidate = apply_slice_policy(baselines[patient], predictions[patient], risk, fraction)
            values.append(case_dice(truth, candidate) - case_dice(truth, baselines[patient]))
        result[fraction] = np.asarray(values)
    return result


def select_policy(inner_gain: dict[float, np.ndarray]) -> float:
    """Inner-only acceptance screen: positive mean and ≤25% harmed cases."""
    candidates = []
    for fraction in POLICY_FRACTIONS[1:]:
        values = inner_gain[fraction]
        harm = int(np.sum(values < -1e-5))
        if values.mean() > 1e-5 and harm <= int(np.floor(0.25 * len(values))):
            candidates.append((float(values.mean()), fraction))
    return max(candidates)[1] if candidates else 0.0


def summarize_gain(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "improved_patients": int(np.sum(values > 1e-5)),
        "harmed_patients": int(np.sum(values < -1e-5)),
        "unchanged_patients": int(np.sum(np.abs(values) <= 1e-5)),
    }


def run(args: argparse.Namespace) -> dict:
    names, contexts, labels, baselines, features, target = load_cases(args)
    if args.outer_folds < 2 or args.outer_folds > len(names) or args.max_epochs < 1 or args.batch_size < 1:
        raise ValueError("invalid fold or training limits")
    partitions = grouped_folds(names, args.outer_folds, args.seed + args.fold)
    output = {"schema": "semantic_constraints.cst_teacher.local_correction_study.v1", "fold": args.fold,
              "patients": len(names), "outer_folds": args.outer_folds, "variants": {},
              "warning": "Patient-held-out MSD discovery; not an untouched external-cohort test."}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for variant in args.variants:
        reports = []
        heldout_ungated = np.full(len(names), np.nan, dtype=np.float32)
        heldout_accepted = np.full(len(names), np.nan, dtype=np.float32)
        for outer, test_names in enumerate(partitions):
            test = np.flatnonzero(np.isin(names, test_names))
            remaining = np.flatnonzero(~np.isin(names, test_names))
            rng = np.random.default_rng(args.seed + args.fold * 100 + outer)
            shuffled = rng.permutation(remaining)
            validation_count = max(2, int(np.ceil(len(remaining) * 0.20)))
            inner_val, inner_train = shuffled[:validation_count], shuffled[validation_count:]
            if len(inner_train) < 2 or set(names[test]) & set(names[inner_train]) or set(names[test]) & set(names[inner_val]):
                raise ValueError("insufficient or leaked patient split")
            model, training = train_model(
                contexts, labels, baselines, inner_train, inner_val,
                variant=variant, device=args.device, batch_size=args.batch_size,
                max_epochs=args.max_epochs, seed=args.seed + args.fold * 100 + outer,
            )
            # This QC head is fitted only on inner-training patients. The inner
            # validation cases choose the acceptance policy; outer test labels
            # are read only after both model and policy have been locked.
            risk_head = ridge_fit(features[inner_train], target[inner_train], alpha=1.0)
            sampled_risk = ridge_predict(risk_head, features)
            inner_prediction = predict_cases(model, contexts, inner_val, variant=variant, device=args.device, batch_size=args.batch_size)
            inner_gain = evaluate_policies(inner_val, inner_prediction, labels, baselines, sampled_risk)
            chosen = select_policy(inner_gain)
            test_prediction = predict_cases(model, contexts, test, variant=variant, device=args.device, batch_size=args.batch_size)
            test_gain = evaluate_policies(test, test_prediction, labels, baselines, sampled_risk)
            heldout_ungated[test] = test_gain[1.0]
            heldout_accepted[test] = test_gain[chosen]
            model_path = args.output_dir / f"{variant}_outer{outer}.pt"
            torch.save({
                "schema": "semantic_constraints.cst_teacher.local_corrector.v1",
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "variant": variant,
                "fold": args.fold,
                "outer": outer,
                "training_patients": names[inner_train].tolist(),
                "selection_patients": names[inner_val].tolist(),
                "selected_epoch": training["best_epoch"],
                "acceptance_fraction": chosen,
            }, model_path)
            split = {
                "outer": outer,
                "train_patients": names[inner_train].tolist(),
                "inner_validation_patients": names[inner_val].tolist(),
                "test_patients": names[test].tolist(),
                "training": training,
                "inner_policy_gain": {str(key): summarize_gain(value) for key, value in inner_gain.items()},
                "selected_policy_fraction": chosen,
                "test_ungated_gain": summarize_gain(test_gain[1.0]),
                "test_accepted_gain": summarize_gain(test_gain[chosen]),
                "model_artifact": str(model_path),
            }
            reports.append(split)
            (args.output_dir / f"{variant}_outer{outer}.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
            print(json.dumps({"variant": variant, "fold": args.fold, "outer": outer,
                              "epoch": training["best_epoch"], "policy": chosen,
                              "ungated": split["test_ungated_gain"], "accepted": split["test_accepted_gain"]}), flush=True)
        if not np.isfinite(heldout_ungated).all() or not np.isfinite(heldout_accepted).all():
            raise AssertionError("not every patient received an outer-test result")
        output["variants"][variant] = {
            "ungated": summarize_gain(heldout_ungated),
            "accepted": summarize_gain(heldout_accepted),
            "case_names": names.tolist(),
            "ungated_case_gains": heldout_ungated.tolist(),
            "accepted_case_gains": heldout_accepted.tolist(),
            "outer_splits": reports,
        }
        (args.output_dir / "report.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    with (args.output_dir / "patient_gains.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("case_name", "variant", "ungated_dice_gain", "accepted_dice_gain"))
        for variant, value in output["variants"].items():
            for index, name in enumerate(names):
                writer.writerow((name, variant, value["ungated_case_gains"][index], value["accepted_case_gains"][index]))
    return output


def main() -> None:
    args = parse_args()
    report = run(args)
    print(json.dumps({"fold": report["fold"], "patients": report["patients"],
                      "variants": {name: {key: result[key] for key in ("ungated", "accepted")}
                                   for name, result in report["variants"].items()}}, indent=2))


if __name__ == "__main__":
    main()
