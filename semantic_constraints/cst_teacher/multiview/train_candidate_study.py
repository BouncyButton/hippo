#!/usr/bin/env python3
"""Rank A/P interfaces in a whole-hippocampus mask from matched MRI views."""

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
from .candidate_ranker import (
    CSTCandidateRanker,
    CandidateSetDataset,
    build_candidate_batch,
    candidate_cuts_around,
)
from .cut_model import VIEW_SPECS, best_cut_from_labels
from .train_cut_study import foreground_dice, relabel_by_cut


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", type=Path, required=True)
    parser.add_argument("--splits-json", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--inference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--views", nargs="+", choices=tuple(VIEW_SPECS), default=tuple(VIEW_SPECS))
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def score_candidates(model, batch, cuts, specification, device):
    elements, metadata = build_candidate_batch(batch, cuts, specification)
    scores = model(elements.to(device), metadata.to(device))
    return scores.reshape(cuts.shape)


def inner_validation(model, loader, specification, device):
    model.eval()
    losses, successes, count = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            targets = batch["true_cut_y"].numpy()
            cuts = candidate_cuts_around(targets, batch["union"].shape[2], (-4, -2, 0, 2, 4))
            scores = score_candidates(model, batch, cuts, specification, device)
            target = torch.full((len(targets),), 2, dtype=torch.long, device=device)
            losses += float(F.cross_entropy(scores, target, reduction="sum"))
            successes += int((scores.argmax(dim=1) == 2).sum())
            count += len(targets)
    return losses / count, successes / count


def fit_one(name, train_cases, val_cases, inner_train, inner_val, args):
    specification = VIEW_SPECS[name]
    training = CandidateSetDataset(train_cases, specification)
    held_out = CandidateSetDataset(val_cases, specification)
    train_loader = DataLoader(Subset(training, inner_train.tolist()), batch_size=args.batch_size, shuffle=True)
    inner_loader = DataLoader(Subset(training, inner_val.tolist()), batch_size=args.batch_size, shuffle=False)
    held_loader = DataLoader(held_out, batch_size=1, shuffle=False)
    device = torch.device(args.device)
    torch.manual_seed(args.seed + args.fold)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + args.fold)
    rng = np.random.default_rng(args.seed + args.fold)
    model = CSTCandidateRanker().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_loss, best_epoch, best_state, stale = float("inf"), 0, None, 0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            targets = batch["true_cut_y"].numpy()
            shifts = rng.choice(np.array([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]), len(targets))
            negative = np.clip(targets + shifts, 0, batch["union"].shape[2] - 2)
            cuts = np.stack((targets, negative), axis=1)
            optimizer.zero_grad(set_to_none=True)
            scores = score_candidates(model, batch, cuts, specification, device)
            loss = F.softplus(scores[:, 1] - scores[:, 0]).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        validation, accuracy = inner_validation(model, inner_loader, specification, device)
        history.append({"epoch": epoch, "train_pair_loss": float(np.mean(losses)),
                        "inner_listwise_loss": validation, "inner_top1_accuracy": accuracy})
        if validation < best_loss - 1e-4:
            best_loss, best_epoch, best_state, stale = validation, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise AssertionError("no candidate-ranker checkpoint selected")
    model.load_state_dict(best_state)
    model.eval()
    prediction_paths = prediction_map(args.inference_dir)
    rows = []
    with torch.no_grad():
        for batch in held_loader:
            patient = batch["case_name"][0]
            ground_truth = np.asarray(val_cases[len(rows)]["label"])[0].astype(np.int8)
            probabilities = np.load(prediction_paths[patient], allow_pickle=False)
            swin = probabilities.argmax(axis=0).astype(np.int8)
            if swin.shape != ground_truth.shape:
                raise ValueError(f"probability/label shape mismatch for {patient}")
            try:
                swin_cut = best_cut_from_labels(swin)[0]
            except ValueError:
                active = np.flatnonzero((swin > 0).any(axis=(0, 2)))
                swin_cut = int(np.median(active)) if len(active) else (swin.shape[1] - 1) // 2
            batch["union"] = torch.from_numpy(swin > 0).unsqueeze(0)
            cuts = candidate_cuts_around([swin_cut], swin.shape[1])
            scores = score_candidates(model, batch, cuts, specification, device).cpu().numpy()[0]
            selected = int(cuts[0, np.argmax(scores)])
            true_cut = int(batch["true_cut_y"][0])
            baseline = foreground_dice(ground_truth, swin)
            rows.append({
                "case_name": patient, "true_cut_y": true_cut, "swin_cut_y": swin_cut,
                "chosen_cut_y": selected, "chosen_offset": selected - swin_cut,
                "baseline_dice": baseline,
                "relabel_dice_delta": foreground_dice(ground_truth, relabel_by_cut(swin, selected)) - baseline,
                "scores": scores.tolist(),
            })
    errors = np.array([abs(row["chosen_cut_y"] - row["true_cut_y"]) for row in rows])
    swin_errors = np.array([abs(row["swin_cut_y"] - row["true_cut_y"]) for row in rows])
    deltas = np.array([row["relabel_dice_delta"] for row in rows])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / f"{name}_candidate.pt"
    torch.save({
        "schema": "semantic_constraints.cst_teacher.multiview.candidate_ranker.v1",
        "fold": args.fold, "view": name, "specification": specification,
        "state_dict": {key: value.cpu() for key, value in best_state.items()},
        "training_patients": [training.items[index]["case_name"] for index in inner_train],
        "inner_validation_patients": [training.items[index]["case_name"] for index in inner_val],
        "selected_epoch": best_epoch,
    }, checkpoint)
    return {
        "view": name, "tokens": sum(specification.values()),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "selected_epoch": best_epoch, "inner_listwise_loss": best_loss,
        "inner_top1_accuracy": next(item["inner_top1_accuracy"] for item in history if item["epoch"] == best_epoch),
        "outer_swin_cut_mae": float(swin_errors.mean()), "outer_chosen_cut_mae": float(errors.mean()),
        "outer_exact_cut_rate": float(np.mean(errors == 0)),
        "outer_mean_relabel_dice_delta": float(deltas.mean()),
        "outer_harmed_patients": int(np.sum(deltas < -1e-8)),
        "outer_improved_patients": int(np.sum(deltas > 1e-8)),
        "chosen_offset_histogram": {str(offset): int(np.sum(np.array([row["chosen_offset"] for row in rows]) == offset)) for offset in range(-5, 6)},
        "history": history, "patients": rows, "checkpoint": str(checkpoint),
    }


def main():
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
        result = fit_one(name, train_cases, val_cases, inner_train, inner_val, args)
        reports.append(result)
        print(json.dumps({key: result[key] for key in (
            "view", "selected_epoch", "inner_top1_accuracy", "outer_swin_cut_mae",
            "outer_chosen_cut_mae", "outer_mean_relabel_dice_delta", "outer_harmed_patients"
        )}), flush=True)
    report = {
        "schema": "semantic_constraints.cst_teacher.multiview_candidate_study.v1",
        "fold": args.fold, "configuration": vars(args) | {"pkl": str(args.pkl),
            "splits_json": str(args.splits_json), "inference_dir": str(args.inference_dir),
            "output_dir": str(args.output_dir)},
        "train_patients": len(inner_train), "inner_validation_patients": len(inner_val),
        "outer_validation_patients": len(val_cases), "models": reports,
        "warning": "GT unions train the synthetic ranking task; held-out evaluation uses Swin unions. Existing folds were previously studied and are not prospective untouched cohorts.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"fold": args.fold, "models": [{key: value for key, value in item.items()
        if key not in ("history", "patients")} for item in reports]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
