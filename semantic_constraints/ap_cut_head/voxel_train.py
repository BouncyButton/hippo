#!/usr/bin/env python3
"""Probe literal cut-voxel labels on frozen Swin decoder features."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from monai.networks.nets import SwinUNETR
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from semantic_constraints.cst_teacher.train_teacher import build_loaders, seed_everything

from .model import soft_cut_targets
from .train import FrozenSwinCutDataset, summarize
from .voxel_model import CutVoxelHead, candidate_cut_scores, cut_band_target, cut_voxel_loss


class DecoderFeatureDataset(Dataset):
    """Cache frozen final-decoder tensors so head fitting is cheap and reproducible."""

    def __init__(self, source: FrozenSwinCutDataset, backbone: SwinUNETR, device: torch.device):
        self.items = []
        captured: list[torch.Tensor] = []

        def hook(_module, _inputs, output):
            captured.append(output)

        handle = backbone.decoder1.register_forward_hook(hook)
        try:
            backbone.eval()
            with torch.no_grad():
                for index, item in enumerate(source.items):
                    image = item["image"].unsqueeze(0).to(device)
                    backbone(image)
                    if len(captured) != 1:
                        raise RuntimeError("expected one final-decoder feature tensor per Swin forward")
                    feature = captured.pop()
                    if feature.shape[2:] != image.shape[2:]:
                        raise ValueError(f"decoder feature spatial shape differs for {item['case_name']}")
                    self.items.append({
                        "case_name": item["case_name"],
                        "feature": feature[0].detach().to(device="cpu", dtype=torch.float16),
                        "probabilities": item["probabilities"],
                        "foreground": item["foreground"],
                        "true_cut": item["true_cut"],
                        "swin_cut": item["swin_cut"],
                        "nonplanar_voxels": item["nonplanar_voxels"],
                    })
                    if (index + 1) % 25 == 0:
                        print(f"[FEATURES] {index + 1}/{len(source)}", flush=True)
        finally:
            handle.remove()
        if not self.items:
            raise ValueError("no decoder features cached")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", type=Path, required=True)
    parser.add_argument("--splits-json", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-inference-dir", type=Path, required=True)
    parser.add_argument("--val-inference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--target-sigma", type=float, default=0.75)
    parser.add_argument("--voxel-weight", type=float, default=0.25)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def load_backbone(checkpoint: Path, device: torch.device) -> SwinUNETR:
    backbone = SwinUNETR(in_channels=1, out_channels=3, use_checkpoint=False).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    backbone.load_state_dict(state)
    backbone.eval().requires_grad_(False)
    return backbone


def scores_from_batch(model: CutVoxelHead, batch: dict, device: torch.device) -> torch.Tensor:
    feature = batch["feature"].to(device=device, dtype=torch.float32)
    heatmap = model(feature)
    probabilities = batch["probabilities"].to(device=device, dtype=torch.float32)
    return candidate_cut_scores(heatmap, probabilities[:, 1:3].sum(dim=1))


@torch.no_grad()
def predict(model: CutVoxelHead, loader: DataLoader, device: torch.device, sigma: float) -> list[dict]:
    model.eval()
    rows = []
    for batch in loader:
        scores = scores_from_batch(model, batch, device)
        target = soft_cut_targets(batch["true_cut"].to(device), scores.shape[1], sigma)
        entropy = -(target * F.log_softmax(scores, dim=1)).sum(dim=1)
        confidence = scores.softmax(dim=1).max(dim=1).values
        for index, name in enumerate(batch["case_name"]):
            rows.append({
                "case_name": name,
                "true_cut": int(batch["true_cut"][index]),
                "swin_cut": int(batch["swin_cut"][index]),
                "predicted_cut": int(scores[index].argmax()),
                "confidence": float(confidence[index]),
                "target_cross_entropy": float(entropy[index]),
                "nonplanar_voxels": int(batch["nonplanar_voxels"][index]),
            })
    return rows


def fit(
    training: DecoderFeatureDataset,
    validation: DecoderFeatureDataset,
    inner_train: np.ndarray,
    inner_val: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    device = torch.device(args.device)
    train_loader = DataLoader(Subset(training, inner_train.tolist()), batch_size=args.batch_size, shuffle=True)
    inner_loader = DataLoader(Subset(training, inner_val.tolist()), batch_size=args.batch_size)
    outer_loader = DataLoader(validation, batch_size=args.batch_size)
    seed_everything(args.seed + args.fold)
    channels = int(training.items[0]["feature"].shape[0])
    model = CutVoxelHead(channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_loss, best_epoch, best_state, stale = float("inf"), 0, None, 0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            feature = batch["feature"].to(device=device, dtype=torch.float32)
            probabilities = batch["probabilities"].to(device=device, dtype=torch.float32)
            foreground = batch["foreground"].to(device=device, dtype=torch.float32)
            cuts = batch["true_cut"].to(device)
            heatmap = model(feature)
            scores = candidate_cut_scores(heatmap, probabilities[:, 1:3].sum(dim=1))
            target_distribution = soft_cut_targets(cuts, scores.shape[1], args.target_sigma)
            location_loss = -(target_distribution * F.log_softmax(scores, dim=1)).sum(dim=1).mean()
            band_target = cut_band_target(foreground, cuts)
            band_loss = cut_voxel_loss(heatmap, foreground, band_target)
            loss = location_loss + args.voxel_weight * band_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        inner_rows = predict(model, inner_loader, device, args.target_sigma)
        inner_loss = float(np.mean([row["target_cross_entropy"] for row in inner_rows]))
        inner_mae = float(np.mean([abs(row["predicted_cut"] - row["true_cut"]) for row in inner_rows]))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "inner_cross_entropy": inner_loss, "inner_cut_mae": inner_mae})
        print(json.dumps(history[-1]), flush=True)
        if inner_loss < best_loss - 1e-4:
            best_loss, best_epoch, best_state, stale = inner_loss, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise AssertionError("no voxel-head checkpoint selected")
    model.load_state_dict(best_state)
    median_cut = int(np.rint(np.median([training.items[index]["true_cut"] for index in inner_train])))
    outer_rows = predict(model, outer_loader, device, args.target_sigma)
    torch.save({
        "schema": "semantic_constraints.ap_cut_head.voxel.v1",
        "fold": args.fold,
        "state_dict": {key: value.cpu() for key, value in best_state.items()},
        "decoder_channels": channels,
        "selected_epoch": best_epoch,
        "training_patients": [training.items[index]["case_name"] for index in inner_train],
        "inner_validation_patients": [training.items[index]["case_name"] for index in inner_val],
    }, args.output_dir / "voxel_head.pt")
    return {
        "selected_epoch": best_epoch,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_median_cut": median_cut,
        "inner_validation": summarize(predict(model, inner_loader, device, args.target_sigma), median_cut),
        "outer_validation": summarize(outer_rows, median_cut),
        "history": history,
        "patients": outer_rows,
    }


def main() -> None:
    args = parse_args()
    if args.max_epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.target_sigma <= 0 or args.voxel_weight < 0:
        raise ValueError("invalid training configuration")
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loader_args = argparse.Namespace(
        pkl=args.pkl, splits_json=args.splits_json, fold=args.fold,
        spatial_size=(64, 64, 64), set_size=32, slab_depth=3,
        inplane_size=32, cache_dataset=False, max_train_cases=0,
        max_val_cases=0, batch_size=16, num_workers=0,
    )
    train_loader, val_loader = build_loaders(loader_args)
    training_source = FrozenSwinCutDataset(train_loader.dataset.base_dataset, args.train_inference_dir)
    validation_source = FrozenSwinCutDataset(val_loader.dataset.base_dataset, args.val_inference_dir)
    if {item["case_name"] for item in training_source.items} & {item["case_name"] for item in validation_source.items}:
        raise ValueError("train/validation patient overlap")
    device = torch.device(args.device)
    backbone = load_backbone(args.checkpoint, device)
    print("[FEATURES] caching training decoder tensors", flush=True)
    training = DecoderFeatureDataset(training_source, backbone, device)
    print("[FEATURES] caching validation decoder tensors", flush=True)
    validation = DecoderFeatureDataset(validation_source, backbone, device)
    del backbone, training_source, validation_source
    if device.type == "cuda":
        torch.cuda.empty_cache()
    rng = np.random.default_rng(args.seed + args.fold)
    indices = rng.permutation(len(training))
    count = max(20, int(np.ceil(0.2 * len(training))))
    inner_val, inner_train = indices[:count], indices[count:]
    result = fit(training, validation, inner_train, inner_val, args)
    report = {
        "schema": "semantic_constraints.ap_cut_head.voxel_study.v1",
        "fold": args.fold,
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "training_cases": len(inner_train),
        "inner_validation_cases": len(inner_val),
        "outer_validation_cases": len(validation),
        "result": result,
        "warning": "Frozen in-sample Swin training features; disjoint outer patients on previously studied development folds. No segmentation edits were made.",
    }
    (args.output_dir / "voxel_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(result["outer_validation"], indent=2), flush=True)


if __name__ == "__main__":
    main()
