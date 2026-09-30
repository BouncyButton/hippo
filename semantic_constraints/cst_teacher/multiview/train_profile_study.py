#!/usr/bin/env python3
"""Train matched view-specific CST profile teachers and test real Swin signals."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ..descriptors import DESCRIPTOR_NAMES
from ..evaluate_predictions import prediction_map
from ..losses import descriptor_quantile_loss, slice_profile_loss
from ..model import CSTDescriptorTeacher
from ..train_teacher import build_loaders
from .cut_model import ProfileSetDataset, VIEW_SPECS, make_probability_view_profiles
from .train_cut_study import foreground_dice


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
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def profile_loss(output, batch: dict, device: torch.device) -> torch.Tensor:
    descriptors = batch["descriptors"].to(device)
    profiles = batch["profiles"].to(device)
    return 0.25 * descriptor_quantile_loss(output.descriptor_quantiles, descriptors) + 4.0 * slice_profile_loss(
        output.slice_profiles, profiles, loss_kind="smooth_l1"
    )


def evaluate_inner(model: CSTDescriptorTeacher, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            output = model(batch["slabs"].to(device), batch["metadata"].to(device))
            losses.append((float(profile_loss(output, batch, device).detach()), len(batch["case_name"])))
    return sum(loss * count for loss, count in losses) / sum(count for _, count in losses)


def fit_one(name: str, train_cases: list, val_cases: list, inner_train: np.ndarray, inner_val: np.ndarray, args: argparse.Namespace) -> dict:
    specification = VIEW_SPECS[name]
    training = ProfileSetDataset(train_cases, specification)
    held_out = ProfileSetDataset(val_cases, specification)
    train_loader = DataLoader(Subset(training, inner_train.tolist()), batch_size=args.batch_size, shuffle=True)
    inner_loader = DataLoader(Subset(training, inner_val.tolist()), batch_size=args.batch_size, shuffle=False)
    held_loader = DataLoader(held_out, batch_size=args.batch_size, shuffle=False)
    device = torch.device(args.device)
    torch.manual_seed(args.seed + args.fold)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + args.fold)
    model = CSTDescriptorTeacher(metadata_dim=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_loss, best_epoch, best_state, stale = float("inf"), 0, None, 0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(batch["slabs"].to(device), batch["metadata"].to(device))
            loss = profile_loss(output, batch, device)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = evaluate_inner(model, inner_loader, device)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "inner_loss": validation})
        if validation < best_loss - 1e-4:
            best_loss, best_epoch, best_state, stale = validation, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise AssertionError("no profile checkpoint was selected")
    model.load_state_dict(best_state)
    model.eval()
    names, case_embeddings, element_embeddings, teacher_profiles, target_profiles, quantiles, targets = [], [], [], [], [], [], []
    with torch.no_grad():
        for batch in held_loader:
            output = model(batch["slabs"].to(device), batch["metadata"].to(device))
            names.extend(batch["case_name"])
            case_embeddings.append(output.case_embedding.cpu().numpy())
            element_embeddings.append(output.element_embeddings.cpu().numpy())
            teacher_profiles.append(output.slice_profiles.cpu().numpy())
            target_profiles.append(batch["profiles"].numpy())
            quantiles.append(output.descriptor_quantiles.cpu().numpy())
            targets.append(batch["descriptors"].numpy())
    case_embeddings = np.concatenate(case_embeddings)
    element_embeddings = np.concatenate(element_embeddings)
    teacher_profiles = np.concatenate(teacher_profiles)
    target_profiles = np.concatenate(target_profiles)
    quantiles = np.concatenate(quantiles)
    targets = np.concatenate(targets)
    if names != [str(item["case_name"]) for item in val_cases]:
        raise ValueError("held-out patient order changed")
    if teacher_profiles.shape[1] != sum(specification.values()):
        raise ValueError("teacher profile cardinality mismatch")
    paths = prediction_map(args.inference_dir)
    disagreement, dice_error, swin_profiles = [], [], []
    for index, item in enumerate(val_cases):
        label = np.asarray(item["label"])[0].astype(np.int8)
        probabilities = np.load(paths[names[index]], allow_pickle=False)
        swin_profile = make_probability_view_profiles(torch.from_numpy(probabilities), specification).numpy()
        swin_profiles.append(swin_profile)
        disagreement.append(float(np.mean(np.abs(swin_profile - teacher_profiles[index]))))
        dice_error.append(1 - foreground_dice(label, probabilities.argmax(axis=0)))
    swin_profiles = np.stack(swin_profiles)
    disagreement = np.asarray(disagreement)
    dice_error = np.asarray(dice_error)
    group_metrics = {}
    offset = 0
    for view, count in specification.items():
        group_metrics[view] = {
            "profile_mae": float(np.mean(np.abs(teacher_profiles[:, offset:offset + count] - target_profiles[:, offset:offset + count]))),
            "swin_disagreement_vs_case_error_r": float(np.corrcoef(
                np.mean(np.abs(swin_profiles[:, offset:offset + count] - teacher_profiles[:, offset:offset + count]), axis=(1, 2)),
                dice_error,
            )[0, 1]),
        }
        offset += count
    cover = (targets >= quantiles[..., 0]) & (targets <= quantiles[..., 2])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / f"{name}.pt"
    torch.save({
        "schema": "semantic_constraints.cst_teacher.multiview.profile_teacher.v1",
        "fold": args.fold, "view": name, "specification": specification,
        "state_dict": {key: value.cpu() for key, value in best_state.items()},
        "training_patients": [training.items[index]["case_name"] for index in inner_train],
        "inner_validation_patients": [training.items[index]["case_name"] for index in inner_val],
        "selected_epoch": best_epoch,
    }, checkpoint_path)
    features_path = args.output_dir / f"{name}_val_features.npz"
    np.savez_compressed(
        features_path,
        case_names=np.asarray(names),
        case_embeddings=case_embeddings,
        element_embeddings=element_embeddings,
        teacher_profiles=teacher_profiles,
        swin_profiles=swin_profiles,
        profile_targets=target_profiles,
        descriptor_quantiles=quantiles,
        descriptor_targets=targets,
    )
    return {
        "view": name,
        "tokens": sum(specification.values()),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "selected_epoch": best_epoch,
        "inner_best_loss": best_loss,
        "history": history,
        "outer_profile_mae": float(np.mean(np.abs(teacher_profiles - target_profiles))),
        "outer_group_metrics": group_metrics,
        "outer_anterior_interval_coverage": float(cover[:, 0].mean()),
        "outer_descriptor_interval_coverage": dict(zip(DESCRIPTOR_NAMES, cover.mean(axis=0).tolist())),
        "outer_disagreement_vs_case_error_r": float(np.corrcoef(disagreement, dice_error)[0, 1]),
        "checkpoint": str(checkpoint_path),
        "val_features": str(features_path),
    }


def main() -> None:
    args = parse_args()
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
    reports = []
    for name in args.views:
        print(f"[VIEW] fold={args.fold} view={name}", flush=True)
        item = fit_one(name, train_cases, val_cases, inner_train, inner_val, args)
        reports.append(item)
        print(json.dumps({
            "view": name, "epoch": item["selected_epoch"],
            "profile_mae": item["outer_profile_mae"],
            "disagreement_error_r": item["outer_disagreement_vs_case_error_r"],
        }), flush=True)
    report = {
        "schema": "semantic_constraints.cst_teacher.multiview_profile_study.v1",
        "fold": args.fold,
        "train_patients": len(inner_train),
        "inner_validation_patients": len(inner_val),
        "outer_validation_patients": len(val_cases),
        "configuration": {"seed": args.seed, "views": args.views, "max_epochs": args.max_epochs, "patience": args.patience},
        "models": reports,
        "warning": "Previously studied MSD folds; this tests matched view objectives but is not a prospective untouched-cohort result.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"fold": args.fold, "models": [
        {key: value for key, value in item.items() if key not in ("history", "outer_descriptor_interval_coverage", "checkpoint", "val_features")}
        for item in reports
    ]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
