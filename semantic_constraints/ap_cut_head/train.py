#!/usr/bin/env python3
"""Train a frozen-Swin auxiliary cut head; evaluate localization without edits."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from semantic_constraints.cst_teacher.evaluate_predictions import prediction_map
from semantic_constraints.cst_teacher.multiview.cut_model import best_cut_from_labels
from semantic_constraints.cst_teacher.train_teacher import build_loaders, seed_everything

from .model import APCutHead, soft_cut_targets


class FrozenSwinCutDataset(Dataset):
    """Deterministic 64³ images, frozen Swin probabilities, and label-derived cuts."""

    def __init__(self, base_dataset: Dataset, inference_dir: Path):
        paths = prediction_map(inference_dir)
        self.items: list[dict] = []
        for index in range(len(base_dataset)):
            case = base_dataset[index]
            name = str(case["case_name"])
            if name not in paths:
                raise ValueError(f"no frozen Swin probabilities for {name}")
            image = torch.as_tensor(case["image"], dtype=torch.float32)
            label = np.asarray(case["label"])[0].astype(np.int8)
            probabilities = np.load(paths[name], allow_pickle=False).astype(np.float32)
            if image.shape != (1, 64, 64, 64) or probabilities.shape != (3, 64, 64, 64):
                raise ValueError(f"unexpected image/probability shape for {name}")
            cut, high, wrong = best_cut_from_labels(label)
            if not high:
                raise ValueError(f"anterior orientation reversed for {name}")
            swin = probabilities.argmax(axis=0).astype(np.int8)
            try:
                swin_cut = best_cut_from_labels(swin)[0]
            except ValueError:
                swin_cut = 31
            self.items.append({
                "case_name": name,
                "image": image,
                "probabilities": torch.from_numpy(probabilities),
                "foreground": torch.from_numpy((label > 0).astype(np.float32)),
                "true_cut": cut,
                "swin_cut": swin_cut,
                "nonplanar_voxels": wrong,
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", type=Path, required=True)
    parser.add_argument("--splits-json", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--train-inference-dir", type=Path, required=True)
    parser.add_argument("--val-inference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=("mri_swin", "swin_only"), default=("mri_swin", "swin_only"))
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--target-sigma", type=float, default=0.75)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    parser.add_argument("--max-train-cases", type=int, default=0, help="Smoke-test limit; zero uses all cases")
    parser.add_argument("--max-val-cases", type=int, default=0, help="Smoke-test limit; zero uses all cases")
    return parser.parse_args()


@torch.no_grad()
def predict(model: APCutHead, loader: DataLoader, device: torch.device, sigma: float) -> list[dict]:
    model.eval()
    rows = []
    for batch in loader:
        scores = model(batch["image"].to(device), batch["probabilities"].to(device), batch["swin_cut"].to(device))
        distribution = scores.softmax(dim=1)
        target = soft_cut_targets(batch["true_cut"].to(device), scores.shape[1], sigma)
        cross_entropy = -(target * F.log_softmax(scores, dim=1)).sum(dim=1)
        for index, name in enumerate(batch["case_name"]):
            rows.append({
                "case_name": name,
                "true_cut": int(batch["true_cut"][index]),
                "swin_cut": int(batch["swin_cut"][index]),
                "predicted_cut": int(scores[index].argmax()),
                "confidence": float(distribution[index].max()),
                "target_cross_entropy": float(cross_entropy[index]),
                "nonplanar_voxels": int(batch["nonplanar_voxels"][index]),
            })
    return rows


def summarize(rows: list[dict], training_median_cut: int) -> dict:
    if not rows:
        raise ValueError("no cases to summarize")
    true = np.asarray([row["true_cut"] for row in rows])
    swin = np.asarray([row["swin_cut"] for row in rows])
    predicted = np.asarray([row["predicted_cut"] for row in rows])
    prior_error = np.abs(true - training_median_cut)
    swin_error = np.abs(true - swin)
    predicted_error = np.abs(true - predicted)
    atypical = prior_error >= 2
    return {
        "cases": len(rows),
        "coordinate_prior_mae": float(prior_error.mean()),
        "swin_cut_mae": float(swin_error.mean()),
        "head_cut_mae": float(predicted_error.mean()),
        "head_exact_rate": float(np.mean(predicted_error == 0)),
        "head_within_one_rate": float(np.mean(predicted_error <= 1)),
        "mean_head_minus_swin_abs_error": float((predicted_error - swin_error).mean()),
        "patients_better_than_swin": int(np.sum(predicted_error < swin_error)),
        "patients_worse_than_swin": int(np.sum(predicted_error > swin_error)),
        "atypical_cases": int(atypical.sum()),
        "atypical_head_mae": float(predicted_error[atypical].mean()) if atypical.any() else None,
        "atypical_swin_mae": float(swin_error[atypical].mean()) if atypical.any() else None,
        "mean_target_cross_entropy": float(np.mean([row["target_cross_entropy"] for row in rows])),
    }


def fit_arm(
    arm: str,
    training: FrozenSwinCutDataset,
    validation: FrozenSwinCutDataset,
    inner_train: np.ndarray,
    inner_val: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    train_loader = DataLoader(Subset(training, inner_train.tolist()), batch_size=args.batch_size, shuffle=True)
    inner_loader = DataLoader(Subset(training, inner_val.tolist()), batch_size=args.batch_size)
    outer_loader = DataLoader(validation, batch_size=args.batch_size)
    device = torch.device(args.device)
    seed_everything(args.seed + args.fold)
    model = APCutHead(image_enabled=(arm == "mri_swin")).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_loss, best_epoch, best_state, stale = float("inf"), 0, None, 0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            image = batch["image"].to(device)
            probabilities = batch["probabilities"].to(device)
            cut = batch["true_cut"].to(device)
            scores = model(image, probabilities, batch["swin_cut"].to(device))
            target = soft_cut_targets(cut, scores.shape[1], args.target_sigma)
            loss = -(target * F.log_softmax(scores, dim=1)).sum(dim=1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        inner_rows = predict(model, inner_loader, device, args.target_sigma)
        inner_loss = float(np.mean([row["target_cross_entropy"] for row in inner_rows]))
        inner_mae = float(np.mean([abs(row["predicted_cut"] - row["true_cut"]) for row in inner_rows]))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "inner_cross_entropy": inner_loss, "inner_cut_mae": inner_mae})
        print(json.dumps({"arm": arm, **history[-1]}), flush=True)
        if inner_loss < best_loss - 1e-4:
            best_loss, best_epoch, best_state, stale = inner_loss, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise AssertionError("no cut-head checkpoint selected")
    model.load_state_dict(best_state)
    outer_rows = predict(model, outer_loader, device, args.target_sigma)
    median_cut = int(np.rint(np.median([training.items[index]["true_cut"] for index in inner_train])))
    report = {
        "arm": arm,
        "selected_epoch": best_epoch,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_median_cut": median_cut,
        "inner_validation": summarize(predict(model, inner_loader, device, args.target_sigma), median_cut),
        "outer_validation": summarize(outer_rows, median_cut),
        "history": history,
        "patients": outer_rows,
    }
    torch.save({
        "schema": "semantic_constraints.ap_cut_head.v1",
        "fold": args.fold,
        "arm": arm,
        "state_dict": {key: value.cpu() for key, value in best_state.items()},
        "selected_epoch": best_epoch,
        "target_sigma": args.target_sigma,
        "training_patients": [training.items[index]["case_name"] for index in inner_train],
        "inner_validation_patients": [training.items[index]["case_name"] for index in inner_val],
    }, args.output_dir / f"{arm}.pt")
    return report


def main() -> None:
    args = parse_args()
    if args.max_epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.target_sigma <= 0:
        raise ValueError("training limits and target sigma must be positive")
    if len(set(args.arms)) != len(args.arms):
        raise ValueError("duplicate experiment arms")
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loader_args = argparse.Namespace(
        pkl=args.pkl, splits_json=args.splits_json, fold=args.fold,
        spatial_size=(64, 64, 64), set_size=32, slab_depth=3, inplane_size=32,
        cache_dataset=False, max_train_cases=args.max_train_cases,
        max_val_cases=args.max_val_cases, batch_size=16, num_workers=0,
    )
    train_loader, val_loader = build_loaders(loader_args)
    training = FrozenSwinCutDataset(train_loader.dataset.base_dataset, args.train_inference_dir)
    validation = FrozenSwinCutDataset(val_loader.dataset.base_dataset, args.val_inference_dir)
    train_names = {item["case_name"] for item in training.items}
    val_names = {item["case_name"] for item in validation.items}
    if train_names & val_names:
        raise ValueError("train/outer-validation patient overlap")
    rng = np.random.default_rng(args.seed + args.fold)
    indices = rng.permutation(len(training))
    inner_val_count = min(max(20, int(np.ceil(0.2 * len(training)))), len(training) - 1)
    if inner_val_count < 1:
        raise ValueError("at least two training patients are required")
    inner_val, inner_train = indices[:inner_val_count], indices[inner_val_count:]
    arms = {}
    for arm in args.arms:
        arms[arm] = fit_arm(arm, training, validation, inner_train, inner_val, args)
        print(json.dumps({"arm": arm, "outer_validation": arms[arm]["outer_validation"]}, indent=2), flush=True)
    report = {
        "schema": "semantic_constraints.ap_cut_head.study.v1",
        "fold": args.fold,
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "training_cases": len(inner_train),
        "inner_validation_cases": len(inner_val),
        "outer_validation_cases": len(validation),
        "arms": arms,
        "warning": "Frozen Swin probabilities for inner training are in-sample; outer patients are disjoint. Fold 0/1 have been studied repeatedly and are development data. No segmentation edits were made.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
