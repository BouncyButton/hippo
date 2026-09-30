"""Frozen early-stopped SwinUNETR with an independently supervised boundary head.

This diagnostic does not smooth surfaces, alter the segmentation, or train the
backbone. It tests whether image-conditioned boundary predictions contribute
correct information where the baseline segmentation disagrees with them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.data import DataLoader, Dataset
from monai.networks.nets import SwinUNETR

from baselines.swin_unetr.swin_unetr import (
    _build_monai_dataset_from_pkl,
    _load_pkl_dataframe,
    _load_splits_json,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_targets(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = labels[:, 0] > 0
    present = mask.any(-1)
    lower = mask.float().argmax(-1)
    upper = mask.shape[-1] - 1 - mask.flip(-1).float().argmax(-1)
    return present, lower.long(), upper.long()


class BoundaryHead(nn.Module):
    """Predict two categorical z locations and column presence from image/features."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(25, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv3d(32, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.edges = nn.Conv3d(32, 2, 1)
        self.presence = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 1, 1)
        )

    def forward(self, image: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        signal = self.stem(torch.cat((image, features), dim=1))
        pooled = torch.cat((signal.mean(-1), signal.amax(-1)), dim=1)
        return self.presence(pooled).squeeze(1), self.edges(signal)


class FrozenBaseline(nn.Module):
    def __init__(self, checkpoint: Path) -> None:
        super().__init__()
        self.model = SwinUNETR(in_channels=1, out_channels=3, use_checkpoint=False)
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False))
        self.features: torch.Tensor | None = None
        self.model.out.register_forward_pre_hook(self._capture)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()

    def _capture(self, _module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        value = args[0]
        self.features = value.as_tensor() if hasattr(value, "as_tensor") else value

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.model(image)
        assert self.features is not None and self.features.shape[1] == 24
        return logits, self.features


def decoded_means(edge_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z = torch.arange(edge_logits.shape[-1], device=edge_logits.device, dtype=torch.float32)
    means = (edge_logits.float().softmax(-1) * z).sum(-1)
    return means[:, 0], means[:, 1]


def head_loss(output: tuple[torch.Tensor, torch.Tensor], labels: torch.Tensor) -> tuple[torch.Tensor, dict]:
    presence_logits, edge_logits = output
    present, lower, upper = label_targets(labels)
    p = presence_logits.float().sigmoid()
    bce = F.binary_cross_entropy_with_logits(presence_logits.float(), present.float())
    soft_dice = 1 - (2 * (p * present).sum() + 1) / (p.sum() + present.sum() + 1)
    presence_loss = bce + 0.5 * soft_dice
    logp = edge_logits.float().log_softmax(-1)
    lower_nll = -logp[:, 0].gather(-1, lower.unsqueeze(-1)).squeeze(-1)
    upper_nll = -logp[:, 1].gather(-1, upper.unsqueeze(-1)).squeeze(-1)
    n = present.sum().clamp_min(1)
    edge_nll = ((lower_nll + upper_nll) * present).sum() / (2 * n)
    mu_lower, mu_upper = decoded_means(edge_logits)
    position = (((mu_lower - lower).abs() + (mu_upper - upper).abs()) * present).sum() / (2 * n * 64)
    crossing = ((mu_lower - mu_upper).relu() * present).sum() / (n * 64)
    total = presence_loss + 0.25 * edge_nll + 0.25 * position + 0.25 * crossing
    return total, {
        "presence": float(presence_loss.detach()),
        "edge_nll": float(edge_nll.detach()),
        "position": float(position.detach()),
        "crossing": float(crossing.detach()),
    }


def hard_head_mask(output: tuple[torch.Tensor, torch.Tensor], threshold: float = 0.5) -> torch.Tensor:
    presence_logits, edge_logits = output
    lower, upper = decoded_means(edge_logits)
    lo = torch.minimum(lower, upper).round().long().clamp(0, 63)
    hi = torch.maximum(lower, upper).round().long().clamp(0, 63)
    z = torch.arange(64, device=lo.device)
    present = presence_logits.float().sigmoid() >= threshold
    return present[..., None] & (z >= lo[..., None]) & (z <= hi[..., None])


def bounds(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    occupied = mask.any(-1)
    lower = mask.argmax(-1)
    upper = 63 - mask[..., ::-1].argmax(-1)
    return occupied, lower, upper


def dice(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(2 * np.count_nonzero(pred & truth) / (np.count_nonzero(pred) + np.count_nonzero(truth)))


def case_metrics(name: str, cohort: str, truth: np.ndarray, seg: np.ndarray, head: np.ndarray) -> dict:
    truth_occ, truth_lo, truth_hi = bounds(truth)
    seg_occ, seg_lo, seg_hi = bounds(seg)
    head_occ, head_lo, head_hi = bounds(head)
    row = {
        "case": name, "cohort": cohort,
        "baseline_union_dice": dice(seg, truth),
        "head_union_dice": dice(head, truth),
        "voxels_fixed": int(np.count_nonzero((seg != truth) & (head == truth))),
        "voxels_broken": int(np.count_nonzero((seg == truth) & (head != truth))),
    }
    row["head_minus_baseline_union_dice"] = row["head_union_dice"] - row["baseline_union_dice"]
    for prefix, occupied, lower, upper in (
        ("baseline", seg_occ, seg_lo, seg_hi), ("head", head_occ, head_lo, head_hi)
    ):
        common = occupied & truth_occ
        errors = np.concatenate((np.abs(lower[common] - truth_lo[common]),
                                 np.abs(upper[common] - truth_hi[common])))
        row.update({
            f"{prefix}_presence_tp": int(common.sum()),
            f"{prefix}_presence_fp": int(np.count_nonzero(occupied & ~truth_occ)),
            f"{prefix}_presence_fn": int(np.count_nonzero(~occupied & truth_occ)),
            f"{prefix}_edge_mae_common": float(errors.mean()) if len(errors) else math.nan,
            f"{prefix}_edge_within_one_common": float(np.mean(errors <= 1)) if len(errors) else math.nan,
        })
    presence_disagree = seg_occ ^ head_occ
    both = seg_occ & head_occ
    edge_disagree = both & ((seg_lo != head_lo) | (seg_hi != head_hi))
    seg_error = np.abs(seg_lo - truth_lo) + np.abs(seg_hi - truth_hi)
    head_error = np.abs(head_lo - truth_lo) + np.abs(head_hi - truth_hi)
    edge_eval = edge_disagree & truth_occ
    row.update({
        "presence_disagreement_columns": int(presence_disagree.sum()),
        "head_presence_wins": int(np.count_nonzero(presence_disagree & (head_occ == truth_occ))),
        "baseline_presence_wins": int(np.count_nonzero(presence_disagree & (seg_occ == truth_occ))),
        "edge_disagreement_reference_columns": int(edge_eval.sum()),
        "head_edge_wins": int(np.count_nonzero(edge_eval & (head_error < seg_error))),
        "baseline_edge_wins": int(np.count_nonzero(edge_eval & (seg_error < head_error))),
        "edge_ties": int(np.count_nonzero(edge_eval & (seg_error == head_error))),
    })
    return row


def summarize(rows: list[dict]) -> dict:
    rng = np.random.default_rng(0)
    out = {}
    for cohort in ("inner_val", "outer_val"):
        items = [row for row in rows if row["cohort"] == cohort]
        delta = np.asarray([row["head_minus_baseline_union_dice"] for row in items])
        samples = delta[rng.integers(0, len(delta), size=(2000, len(delta)))].mean(1)
        out[cohort] = {
            "cases": len(items),
            "baseline_union_dice_mean": float(np.mean([row["baseline_union_dice"] for row in items])),
            "head_union_dice_mean": float(np.mean([row["head_union_dice"] for row in items])),
            "head_minus_baseline_union_dice_mean": float(delta.mean()),
            "head_minus_baseline_union_dice_ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
            "head_better_cases": int(np.count_nonzero(delta > 0)),
            "baseline_better_cases": int(np.count_nonzero(delta < 0)),
            "head_edge_mae_common_case_mean": float(np.mean([row["head_edge_mae_common"] for row in items])),
            "baseline_edge_mae_common_case_mean": float(np.mean([row["baseline_edge_mae_common"] for row in items])),
        }
        for key in ("baseline_presence_tp", "baseline_presence_fp", "baseline_presence_fn",
                    "head_presence_tp", "head_presence_fp", "head_presence_fn",
                    "presence_disagreement_columns", "head_presence_wins", "baseline_presence_wins",
                    "edge_disagreement_reference_columns", "head_edge_wins", "baseline_edge_wins", "edge_ties",
                    "voxels_fixed", "voxels_broken"):
            out[cohort][key] = int(sum(row[key] for row in items))
    return out


def loaders(pkl: Path, splits: Path):
    dataset = _build_monai_dataset_from_pkl(_load_pkl_dataframe(pkl), "MSD", 3, spatial_size=(64, 64, 64))
    fold = _load_splits_json(splits)[0]
    train_names = np.asarray(sorted(fold["train"]))
    rng = np.random.default_rng(0)
    rng.shuffle(train_names)
    inner_names = set(train_names[:42].tolist())
    fit_names = set(train_names[42:].tolist())
    outer_names = set(fold["val"])
    assert len(fit_names) == 166 and len(inner_names) == 42 and len(outer_names) == 52

    def make(names: set[str], *, shuffle: bool = False, batch_size: int = 2):
        items = [item for item in dataset.data if item["case_name"] in names]
        assert len(items) == len(names)
        return DataLoader(Dataset(items, dataset.transform), batch_size=batch_size,
                          shuffle=shuffle, num_workers=0)

    return (make(fit_names, shuffle=True), make(inner_names), make(inner_names, batch_size=1),
            make(outer_names, batch_size=1),
            {"fit": sorted(fit_names), "inner_val": sorted(inner_names), "outer_val": sorted(outer_names)})


def plain(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    moved = tensor.to(device, non_blocking=True)
    return moved.as_tensor() if hasattr(moved, "as_tensor") else moved


@torch.no_grad()
def evaluate_loss(backbone: FrozenBaseline, head: BoundaryHead, loader, device: torch.device) -> float:
    head.eval()
    total = 0.0
    count = 0
    for batch in loader:
        image = plain(batch["image"], device)
        label = plain(batch["label"], device)
        with torch.cuda.amp.autocast():
            _, features = backbone(image)
            output = head(image, features)
            loss, _ = head_loss(output, label)
        total += float(loss) * len(image)
        count += len(image)
    return total / count


@torch.no_grad()
def evaluate_cases(backbone: FrozenBaseline, head: BoundaryHead, loader, device: torch.device, cohort: str) -> list[dict]:
    head.eval()
    rows = []
    for batch in loader:
        assert len(batch["case_name"]) == 1
        image = plain(batch["image"], device)
        with torch.cuda.amp.autocast():
            logits, features = backbone(image)
            output = head(image, features)
        seg = (logits.argmax(1)[0] > 0).cpu().numpy()
        proposed = hard_head_mask(output)[0].cpu().numpy()
        truth = (batch["label"][0, 0] > 0).numpy()
        rows.append(case_metrics(str(batch["case_name"][0]), cohort, truth, seg, proposed))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pkl", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    train, val, val_cases, outer_cases, split = loaders(args.pkl, args.splits)
    config = {
        "protocol": "frozen early-stopped fold-0 baseline; image-conditioned head only; no smoothing/link/fusion",
        "baseline_checkpoint": str(args.checkpoint.resolve()),
        "baseline_sha256": sha256(args.checkpoint),
        "pkl_sha256": sha256(args.pkl),
        "splits_sha256": sha256(args.splits),
        "source_sha256": sha256(Path(__file__)),
        "seed": 0, "max_epochs": 30, "patience": 6, "batch_size": 2,
        "optimizer": "AdamW", "learning_rate": 0.0003, "weight_decay": 0.0001,
        "selection": "lowest inner-val head loss", "presence_threshold": 0.5,
        "split": split,
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    backbone = FrozenBaseline(args.checkpoint).to(device)
    head = BoundaryHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.0003, weight_decay=0.0001)
    scaler = torch.cuda.amp.GradScaler()
    if args.smoke:
        batch = next(iter(train))
        image = plain(batch["image"], device)
        label = plain(batch["label"], device)
        with torch.cuda.amp.autocast():
            logits, features = backbone(image)
            output = head(image, features)
            loss, parts = head_loss(output, label)
        loss.backward()
        assert torch.isfinite(loss) and head.edges.weight.grad is not None
        assert head.presence[-1].weight.grad is not None
        assert all(parameter.grad is None for parameter in backbone.parameters())
        print(json.dumps({"smoke": "ok", "loss": float(loss), "parts": parts,
                          "logits_shape": list(logits.shape), "feature_shape": list(features.shape),
                          "head_edge_shape": list(output[1].shape),
                          "gpu_max_allocated": torch.cuda.max_memory_allocated()}), flush=True)
        return
    best = math.inf
    best_epoch = 0
    stale = 0
    for epoch in range(1, 31):
        head.train()
        losses = []
        for batch in train:
            image = plain(batch["image"], device)
            label = plain(batch["label"], device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                _, features = backbone(image)
                output = head(image, features)
                loss, _ = head_loss(output, label)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        val_loss = evaluate_loss(backbone, head, val, device)
        if val_loss < best - 1e-6:
            best = val_loss
            best_epoch = epoch
            stale = 0
            torch.save({"epoch": epoch, "head": head.state_dict(), "val_loss": val_loss}, args.output / "head_best.pt")
        else:
            stale += 1
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "inner_val_loss": val_loss,
                  "best_epoch": best_epoch, "stale": stale}
        with (args.output / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if stale >= 6:
            break
    state = torch.load(args.output / "head_best.pt", map_location=device, weights_only=False)
    head.load_state_dict(state["head"])
    rows = evaluate_cases(backbone, head, val_cases, device, "inner_val")
    rows.extend(evaluate_cases(backbone, head, outer_cases, device, "outer_val"))
    with (args.output / "case_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"best_epoch": best_epoch, "best_inner_val_loss": best,
               "head_checkpoint_sha256": sha256(args.output / "head_best.pt"),
               "cohorts": summarize(rows)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
