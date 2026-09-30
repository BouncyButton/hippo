#!/usr/bin/env python3
"""Compare matched coronal, sagittal, axial, and tri-view CST cut regressors."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from ..evaluate_predictions import prediction_map
from ..train_teacher import build_loaders
from .cut_model import CSTCutRegressor, CutSetDataset, VIEW_SPECS, best_cut_from_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", type=Path, required=True)
    parser.add_argument("--splits-json", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--inference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--views", nargs="+", choices=tuple(VIEW_SPECS), default=tuple(VIEW_SPECS))
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def foreground_dice(reference: np.ndarray, prediction: np.ndarray) -> float:
    scores = []
    for label in (1, 2):
        real = reference == label
        guess = prediction == label
        scores.append((2 * np.count_nonzero(real & guess) + 1e-8) / (real.sum() + guess.sum() + 1e-8))
    return float(np.mean(scores))


def relabel_by_cut(prediction: np.ndarray, cut: float) -> np.ndarray:
    """Apply one planar A/P cut to the existing Swin foreground, for audit."""
    if prediction.ndim != 3:
        raise ValueError("expected (X,Y,Z) Swin hard labels")
    y = np.arange(prediction.shape[1])[None, :, None]
    return np.where(prediction > 0, np.where(y > round(cut), 1, 2), 0).astype(np.int8)


def predict_cut(model: CSTCutRegressor, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    predicted = []
    with torch.no_grad():
        for batch in loader:
            value = model(batch["slabs"].to(device), batch["metadata"].to(device))
            predicted.extend(value.cpu().tolist())
    return np.asarray(predicted, dtype=np.float32)


def fit_one(
    name: str,
    train_cases: list,
    val_cases: list,
    inner_train: np.ndarray,
    inner_val: np.ndarray,
    *,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    specification = VIEW_SPECS[name]
    training = CutSetDataset(train_cases, specification)
    held_out = CutSetDataset(val_cases, specification)
    train_loader = DataLoader(Subset(training, inner_train.tolist()), batch_size=args.batch_size, shuffle=True)
    inner_loader = DataLoader(Subset(training, inner_val.tolist()), batch_size=args.batch_size, shuffle=False)
    held_loader = DataLoader(held_out, batch_size=args.batch_size, shuffle=False)
    device = torch.device(args.device)
    torch.manual_seed(args.seed + args.fold)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + args.fold)
    model = CSTCutRegressor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_error = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            predicted = model(batch["slabs"].to(device), batch["metadata"].to(device))
            loss = F.smooth_l1_loss(predicted, batch["cut_target"].to(device), beta=0.03)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        inner_pred = predict_cut(model, inner_loader, device)
        inner_target = np.asarray([training.items[index]["cut_target"].item() for index in inner_val])
        mae = float(np.mean(np.abs(inner_pred - inner_target)))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "inner_mae_voxels": mae * 64})
        if mae < best_error - 1e-4:
            best_error, best_epoch, best_state, stale = mae, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise AssertionError("no checkpoint was selected")
    model.load_state_dict(best_state)
    predictions = predict_cut(model, held_loader, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / f"{name}.pt"
    torch.save({
        "schema": "semantic_constraints.cst_teacher.multiview.cut_regressor.v1",
        "fold": args.fold,
        "view": name,
        "specification": specification,
        "state_dict": {key: value.cpu() for key, value in best_state.items()},
        "training_patients": [training.items[index]["case_name"] for index in inner_train],
        "inner_validation_patients": [training.items[index]["case_name"] for index in inner_val],
        "selected_epoch": best_epoch,
    }, checkpoint)
    return predictions, {
        "view": name,
        "tokens": sum(specification.values()),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "inner_mae_voxels": best_error * 64,
        "history": history,
        "checkpoint": str(checkpoint),
    }


def evaluate(
    val_cases: list,
    targets: np.ndarray,
    predictions: dict[str, np.ndarray],
    train_cases: list,
    inner_train: np.ndarray,
    inference_dir: Path,
) -> tuple[dict, list[dict]]:
    paths = prediction_map(inference_dir)
    train_cut = []
    train_fraction = []
    for index in inner_train:
        label = np.asarray(train_cases[index]["label"])[0].astype(np.int8)
        cut, _, _ = best_cut_from_labels(label)
        active = np.flatnonzero((label > 0).any(axis=(0, 2)))
        train_cut.append((cut + 0.5) / label.shape[1])
        train_fraction.append((cut + 0.5 - active[0]) / (active[-1] - active[0] + 1))
    median_cut = float(np.median(train_cut))
    median_fraction = float(np.median(train_fraction))
    rows = []
    for patient, item in enumerate(val_cases):
        name = str(item["case_name"])
        label = np.asarray(item["label"])[0].astype(np.int8)
        probabilities = np.load(paths[name], allow_pickle=False)
        if probabilities.shape != (3, *label.shape):
            raise ValueError(f"Swin probability shape differs for {name}")
        swin = probabilities.argmax(axis=0).astype(np.int8)
        active = np.flatnonzero((swin > 0).any(axis=(0, 2)))
        bbox_cut = (active[0] + median_fraction * (active[-1] - active[0] + 1) - 0.5) if len(active) else median_cut * label.shape[1] - 0.5
        try:
            swin_cut = best_cut_from_labels(swin)[0]
        except ValueError:
            swin_cut = bbox_cut
        cuts = {
            "fixed_coordinate": median_cut * label.shape[1] - 0.5,
            "swin_union_bbox": bbox_cut,
            "swin_ap_cut": swin_cut,
            **{view: float(value[patient] * label.shape[1] - 0.5) for view, value in predictions.items()},
        }
        truth_y = float(targets[patient] * label.shape[1] - 0.5)
        baseline_dice = foreground_dice(label, swin)
        rows.append({
            "case_name": name,
            "true_cut_y": truth_y,
            "swin_dice": baseline_dice,
            "cuts": cuts,
            "absolute_error_voxels": {key: abs(value - truth_y) for key, value in cuts.items()},
            "relabel_dice_delta": {
                key: foreground_dice(label, relabel_by_cut(swin, value)) - baseline_dice
                for key, value in cuts.items() if key != "fixed_coordinate"
            },
        })
    metrics = {}
    for key in rows[0]["cuts"]:
        errors = np.asarray([row["absolute_error_voxels"][key] for row in rows])
        entry = {
            "mae_voxels": float(errors.mean()),
            "median_absolute_error_voxels": float(np.median(errors)),
            "within_one_voxel": float(np.mean(errors <= 1)),
            "within_two_voxels": float(np.mean(errors <= 2)),
        }
        if key != "fixed_coordinate":
            delta = np.asarray([row["relabel_dice_delta"][key] for row in rows])
            entry.update({"mean_relabel_dice_delta": float(delta.mean()), "patients_harmed_by_relabel": int((delta < -1e-8).sum())})
        metrics[key] = entry
    return {"training_median_cut_normalized": median_cut, "training_median_bbox_fraction": median_fraction, "metrics": metrics}, rows


def main() -> None:
    args = parse_args()
    if args.max_epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("training limits must be positive")
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    loader_args = argparse.Namespace(
        pkl=args.pkl, splits_json=args.splits_json, fold=args.fold,
        spatial_size=(64, 64, 64), set_size=32, slab_depth=3,
        inplane_size=32, cache_dataset=False, max_train_cases=0,
        max_val_cases=0, batch_size=16, num_workers=0,
    )
    train_loader, val_loader = build_loaders(loader_args)
    train_cases = [train_loader.dataset.base_dataset[index] for index in range(len(train_loader.dataset.base_dataset))]
    val_cases = [val_loader.dataset.base_dataset[index] for index in range(len(val_loader.dataset.base_dataset))]
    rng = np.random.default_rng(args.seed + args.fold)
    permutation = rng.permutation(len(train_cases))
    inner_val_count = max(20, int(np.ceil(len(train_cases) * 0.20)))
    inner_val, inner_train = permutation[:inner_val_count], permutation[inner_val_count:]
    model_predictions = {}
    model_reports = {}
    targets = np.asarray([
        (best_cut_from_labels(np.asarray(item["label"])[0].astype(np.int8))[0] + 0.5)
        / np.asarray(item["label"]).shape[2]
        for item in val_cases
    ])
    for name in args.views:
        print(f"[VIEW] fold={args.fold} view={name}", flush=True)
        predicted, model_report = fit_one(name, train_cases, val_cases, inner_train, inner_val, args=args)
        model_predictions[name] = predicted
        model_reports[name] = model_report
        print(json.dumps({"view": name, "epoch": model_report["best_epoch"], "inner_mae_voxels": model_report["inner_mae_voxels"]}), flush=True)
    summary, rows = evaluate(val_cases, targets, model_predictions, train_cases, inner_train, args.inference_dir)
    report = {
        "schema": "semantic_constraints.cst_teacher.multiview_cut_study.v1",
        "fold": args.fold,
        "configuration": {"views": args.views, "seed": args.seed, "max_epochs": args.max_epochs, "patience": args.patience, "batch_size": args.batch_size},
        "train_patients": len(inner_train),
        "inner_validation_patients": len(inner_val),
        "outer_validation_patients": len(val_cases),
        "models": model_reports,
        "summary": summary,
        "patients": rows,
        "warning": "Folds 0 and 1 were previously studied; outer validation here is held out of this cut model, not a prospective untouched cohort. Relabel Dice is a diagnostic, not a deployed correction.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"fold": args.fold, "metrics": summary["metrics"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
