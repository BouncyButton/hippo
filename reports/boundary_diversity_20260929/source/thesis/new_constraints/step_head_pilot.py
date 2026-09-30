"""Matched fold-0 pilot of image-conditioned sagittal contour supervision.

The only selection cohort is an inner split of the outer-training patients.
The outer fold-0 validation set is opened only after the best checkpoint is fixed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.data import DataLoader, Dataset
from monai.networks.nets import SwinUNETR
from scipy.ndimage import binary_erosion, distance_transform_edt

from baselines.swin_unetr.swin_unetr import (
    _build_monai_dataset_from_pkl,
    _load_pkl_dataframe,
    _load_splits_json,
    build_optimizer_and_scheduler,
)
from thesis.new_constraints.sagittal_step_fit import SagittalStepContourLoss
from thesis.new_constraints.supervised import build_supervised_loss


ARMS = ("A", "B", "C", "D")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def targets(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Presence and z extremes; empty columns have sentinel -1, never an edge target."""
    mask = labels.squeeze(1) > 0
    present = mask.any(-1)
    z = torch.arange(mask.shape[-1], device=mask.device)
    lower = torch.where(mask, z, mask.shape[-1]).amin(-1)
    upper = torch.where(mask, z, -1).amax(-1)
    return present, torch.where(present, lower, -1), upper


def same_run_pairs(present: torch.Tensor) -> torch.Tensor:
    """True for ordered occupied y pairs without an absent y between them."""
    # [B,X,Y,Y]; cumulative absent count separates disconnected runs.
    gaps = (~present).long().cumsum(-1)
    return (
        present[..., :, None]
        & present[..., None, :]
        & (gaps[..., :, None] == gaps[..., None, :])
        & torch.triu(torch.ones((present.shape[-1],) * 2, device=present.device, dtype=torch.bool), 1)
    )


class ContourHead(nn.Module):
    """Full-resolution decoder feature -> presence and two z distributions."""

    def __init__(self, channels: int = 24) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(channels, 32, 3, padding=1), nn.GELU())
        self.edge = nn.Conv3d(32, 2, 1)
        self.presence = nn.Conv2d(32, 1, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.stem(features)
        return self.presence(f.mean(-1)).squeeze(1), self.edge(f)


class PilotModel(nn.Module):
    def __init__(self, arm: str) -> None:
        super().__init__()
        self.backbone = SwinUNETR(in_channels=1, out_channels=3, use_checkpoint=True)
        self.head = ContourHead() if arm in {"C", "D"} else None
        self.features: torch.Tensor | None = None
        self.backbone.out.register_forward_pre_hook(self._capture)

    def _capture(self, _module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        self.features = args[0].as_tensor() if hasattr(args[0], "as_tensor") else args[0]

    def forward(self, image: torch.Tensor):
        logits = self.backbone(image)
        if self.head is None:
            return logits, None
        assert self.features is not None
        return logits, self.head(self.features)


def decoded_edges(edge_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z = torch.arange(edge_logits.shape[-1], device=edge_logits.device, dtype=torch.float32)
    means = (edge_logits.float().softmax(-1) * z).sum(-1)
    return torch.minimum(means[:, 0], means[:, 1]), torch.maximum(means[:, 0], means[:, 1])


def head_loss(head_output, labels: torch.Tensor, structured: bool) -> tuple[torch.Tensor, dict]:
    presence_logits, edge_logits = head_output
    present, lower, upper = targets(labels)
    n = present.sum().clamp_min(1)
    # Balance occupied and absent columns without inventing edges for absent ones.
    pos = F.softplus(-presence_logits.float())
    neg = F.softplus(presence_logits.float())
    presence_loss = 0.5 * ((pos * present).sum() / n + (neg * ~present).sum() / (~present).sum().clamp_min(1))
    logp = edge_logits.float().log_softmax(-1)
    edge_ce = 0.5 * (F.nll_loss(logp[:, 0].reshape(-1, logp.shape[-1]), lower.clamp_min(0).reshape(-1), reduction="none").reshape_as(lower)
                     + F.nll_loss(logp[:, 1].reshape(-1, logp.shape[-1]), upper.clamp_min(0).reshape(-1), reduction="none").reshape_as(upper))
    edge_loss = (edge_ce * present).sum() / n
    lo, hi = decoded_edges(edge_logits)
    position_loss = (((lo - lower).abs() + (hi - upper).abs()) * present).sum() / (2 * n * edge_logits.shape[-1])
    transition_loss = edge_logits.new_zeros(())
    if structured:
        pair = same_run_pairs(present)
        py = torch.stack((lo, hi), 2)
        ty = torch.stack((lower, upper), 2).float()
        pred_delta = py[..., None, :] - py[..., :, None]
        true_delta = ty[..., None, :] - ty[..., :, None]
        distance = torch.arange(present.shape[-1], device=present.device)
        weight = torch.exp(-((distance[None, :] - distance[:, None]).abs().float() - 1).clamp_min(0) / 4)
        pair_weight = pair[:, :, None].float() * weight
        # All ordered positions in each connected run interact. Target upward
        # excursions are allowed by the hinge; false upward motion is costly.
        fit = F.smooth_l1_loss(pred_delta, true_delta, reduction="none", beta=1.0)
        upward = F.softplus((pred_delta - true_delta.clamp_min(0)) * 2) / 2
        transition_loss = ((fit + 0.25 * upward) * pair_weight).sum() / (pair_weight.sum() * 2).clamp_min(1) / edge_logits.shape[-1]
    loss = presence_loss + edge_loss + position_loss + (32.0 * transition_loss if structured else 0.0)
    return loss, {"presence": float(presence_loss.detach()), "edge": float(edge_loss.detach()),
                  "position": float(position_loss.detach()), "transition": float(transition_loss.detach())}


def dice_score(pred: torch.Tensor, truth: torch.Tensor, cls: int | None = None) -> float:
    a = pred > 0 if cls is None else pred == cls
    b = truth > 0 if cls is None else truth == cls
    denom = int(a.sum() + b.sum())
    return float(2 * (a & b).sum() / denom) if denom else 1.0


def plain(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    moved = tensor.to(device, non_blocking=True)
    return moved.as_tensor() if hasattr(moved, "as_tensor") else moved


def hd95(pred: np.ndarray, truth: np.ndarray) -> float:
    a, b = pred.astype(bool), truth.astype(bool)
    if not a.any() and not b.any():
        return 0.0
    if not a.any() or not b.any():
        return float(max(pred.shape))
    sa = a ^ binary_erosion(a)
    sb = b ^ binary_erosion(b)
    distances = np.r_[distance_transform_edt(~sb)[sa], distance_transform_edt(~sa)[sb]]
    return float(np.percentile(distances, 95))


def contour_metrics(pred: np.ndarray, truth: np.ndarray, prefix: str) -> dict[str, float | int]:
    p = torch.from_numpy(pred.astype(np.uint8))[None, None]
    t = torch.from_numpy(truth.astype(np.uint8))[None, None]
    pp, pl, pu = (x.squeeze().numpy() for x in targets(p))
    tp, tl, tu = (x.squeeze().numpy() for x in targets(t))
    tp = tp.astype(bool); pp = pp.astype(bool)
    both = pp & tp
    d = {f"{prefix}_presence_tp": int(both.sum()),
         f"{prefix}_presence_fp": int((pp & ~tp).sum()),
         f"{prefix}_presence_fn": int((~pp & tp).sum()),
         f"{prefix}_edge_columns": int(both.sum())}
    for name, pe, te in (("lower", pl, tl), ("upper", pu, tu)):
        e = np.abs(pe[both] - te[both])
        d.update({f"{prefix}_{name}_mae": float(e.mean()) if len(e) else math.nan,
                  f"{prefix}_{name}_exact": float((e == 0).mean()) if len(e) else math.nan,
                  f"{prefix}_{name}_within1": float((e <= 1).mean()) if len(e) else math.nan})
    # Compare change locations via nearest annotated transition within a run.
    loc_errors = []
    tails = []
    for x in range(tp.shape[0]):
        ys = np.flatnonzero(tp[x])
        if not len(ys):
            continue
        for y in (ys[0], ys[-1]):
            if pp[x, y]:
                tails.extend((abs(pl[x, y] - tl[x, y]), abs(pu[x, y] - tu[x, y])))
        for pe, te in ((pl, tl), (pu, tu)):
            valid = both[x, 1:] & both[x, :-1]
            truth_changes = np.flatnonzero((np.abs(np.diff(te[x])) >= 1) & valid) + 1
            pred_changes = np.flatnonzero((np.abs(np.diff(pe[x])) >= 1) & valid) + 1
            if len(truth_changes) and len(pred_changes):
                loc_errors.extend(np.min(np.abs(pred_changes[:, None] - truth_changes[None]), axis=1).tolist())
            elif len(truth_changes):
                loc_errors.extend([tp.shape[-1]] * len(truth_changes))
    d[f"{prefix}_transition_location_mae"] = float(np.mean(loc_errors)) if loc_errors else math.nan
    d[f"{prefix}_tail_apex_edge_mae"] = float(np.mean(tails)) if tails else math.nan
    return d


def head_mask(output, probabilities: np.ndarray) -> np.ndarray:
    presence_logits, edge_logits = output
    presence = (presence_logits.float().sigmoid() >= 0.5).squeeze().cpu().numpy()
    lo, hi = decoded_edges(edge_logits)
    lo = lo.round().long().squeeze().cpu().numpy()
    hi = hi.round().long().squeeze().cpu().numpy()
    z = np.arange(probabilities.shape[-1])[None, None, :]
    union = presence[..., None] & (z >= lo[..., None]) & (z <= hi[..., None])
    cls = np.argmax(probabilities[1:], axis=0) + 1
    return np.where(union, cls, 0).astype(np.uint8)


def case_row(arm: str, name: str, probability: np.ndarray, truth: np.ndarray, head: np.ndarray | None) -> dict:
    pred = probability.argmax(0).astype(np.uint8)
    row = {"arm": arm, "patient": name, "union_dice": dice_score(torch.from_numpy(pred), torch.from_numpy(truth)),
           "posterior_dice": dice_score(torch.from_numpy(pred), torch.from_numpy(truth), 1),
           "anterior_dice": dice_score(torch.from_numpy(pred), torch.from_numpy(truth), 2),
           "hd95": hd95(pred > 0, truth > 0)}
    row.update(contour_metrics(pred, truth, "seg"))
    if head is not None:
        row.update(contour_metrics(head, truth, "head"))
        seg_occ = (pred > 0).any(-1)
        head_occ = (head > 0).any(-1)
        ref_occ = (truth > 0).any(-1)
        _, sl, su = (x.squeeze().numpy() for x in targets(torch.from_numpy(pred)[None, None]))
        _, hl, hu = (x.squeeze().numpy() for x in targets(torch.from_numpy(head)[None, None]))
        _, rl, ru = (x.squeeze().numpy() for x in targets(torch.from_numpy(truth)[None, None]))
        disagree = (seg_occ != head_occ) | (seg_occ & head_occ & ((sl != hl) | (su != hu)))
        penalty = pred.shape[-1]
        seg_error = np.where(ref_occ & seg_occ, np.abs(sl - rl) + np.abs(su - ru),
                             np.where(ref_occ | seg_occ, penalty, 0))
        head_error = np.where(ref_occ & head_occ, np.abs(hl - rl) + np.abs(hu - ru),
                              np.where(ref_occ | head_occ, penalty, 0))
        row["disagreement_columns"] = int(disagree.sum())
        row["head_column_wins"] = int(((head_error < seg_error) & disagree).sum())
        row["seg_column_wins"] = int(((seg_error < head_error) & disagree).sum())
        row["column_ties"] = int(((seg_error == head_error) & disagree).sum())
        # Correction uses only MRI-conditioned head and predicted class probabilities.
        fixed = (pred != truth) & (head == truth)
        broken = (pred == truth) & (head != truth)
        row.update({"corrected_union_dice": dice_score(torch.from_numpy(head), torch.from_numpy(truth)),
                    "voxels_fixed": int(fixed.sum()), "voxels_broken": int(broken.sum())})
    return row


def loaders(args, root: Path):
    df = _load_pkl_dataframe(args.pkl)
    ds = _build_monai_dataset_from_pkl(df, "MSD", 3, spatial_size=(64, 64, 64))
    outer = _load_splits_json(args.splits)[0]
    train_names, dev_names = set(outer["train"]), set(outer["val"])
    assert len(train_names) == 208 and len(dev_names) == 52 and not train_names & dev_names
    rng = np.random.default_rng(0)
    shuffled = np.array(sorted(train_names))
    rng.shuffle(shuffled)
    inner_names = set(shuffled[:42].tolist())
    fit_names = train_names - inner_names
    names = {item["case_name"] for item in ds.data}
    assert train_names | dev_names <= names and len(fit_names) == 166
    split = {"seed": 0, "train": sorted(fit_names), "inner_val": sorted(inner_names), "outer_dev": sorted(dev_names)}
    (root / "split.json").write_text(json.dumps(split, indent=2))
    def make(selected, shuffle=False, batch=2):
        items = [item for item in ds.data if item["case_name"] in selected]
        return DataLoader(Dataset(items, ds.transform), batch_size=batch, shuffle=shuffle, num_workers=0)
    return make(fit_names, True), make(inner_names), make(dev_names, batch=1)


@torch.no_grad()
def selection_dice(model, loader, device):
    model.eval()
    scores = []
    for batch in loader:
        pred = model(plain(batch["image"], device))[0].argmax(1).cpu()
        truth = batch["label"].squeeze(1)
        for p, t in zip(pred, truth):
            for cls in (1, 2):
                if (t == cls).any():
                    scores.append(dice_score(p, t, cls))
    return float(np.mean(scores))


def calibrate(model, loader, dice_loss, step_loss, arm, device):
    """Choose weight for auxiliary/segmentation decoder-feature gradient RMS = 0.1."""
    if arm == "A":
        return 0.0, {"target_gradient_ratio": 0.0}
    model.train()
    dice_norm, aux_norm = [], []
    for index, batch in enumerate(loader):
        if index == 4:
            break
        images, labels = plain(batch["image"], device), plain(batch["label"], device)
        logits, head = model(images)
        feature = model.features
        assert feature is not None
        dice = dice_loss(logits, labels)
        aux = step_loss(logits, labels).loss if arm == "B" else head_loss(head, labels, arm == "D")[0]
        gd = torch.autograd.grad(dice, feature, retain_graph=True)[0]
        ga = torch.autograd.grad(aux, feature)[0]
        dice_norm.append(float(gd.float().square().mean().sqrt()))
        aux_norm.append(float(ga.float().square().mean().sqrt()))
        model.zero_grad(set_to_none=True)
    weight = min(10.0, 0.1 * float(np.median(dice_norm)) / max(float(np.median(aux_norm)), 1e-12))
    return weight, {"target_gradient_ratio": 0.1, "dice_rms": dice_norm, "aux_rms": aux_norm, "weight": weight}


def atomic_save(payload, path):
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def train_arm(args, arm, root, train_loader, val_loader, dev_loader, device):
    arm_dir = root / arm
    arm_dir.mkdir(exist_ok=True)
    if (arm_dir / "done.json").exists():
        return
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    model = PilotModel(arm).to(device)
    dice = build_supervised_loss("dice")
    step = SagittalStepContourLoss()
    weight, calibration = calibrate(model, train_loader, dice, step, arm, device)
    (arm_dir / "calibration.json").write_text(json.dumps(calibration, indent=2))
    # Calibration consumes a shuffled TRAIN iterator for B/C/D. Restart the
    # sampler identically for every arm before the first optimizer step.
    random.seed(1000); np.random.seed(1000); torch.manual_seed(1000); torch.cuda.manual_seed_all(1000)
    optimizer, scheduler = build_optimizer_and_scheduler("adamw_0.01", model, 50, learning_rate=1e-4,
                                                          weight_decay=1e-5, step_size=20, adamw_gamma=0.5)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    latest = arm_dir / "latest.pt"
    start, best, best_epoch, stale = 1, -1.0, 0, 0
    if latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"]); scaler.load_state_dict(state["scaler"])
        start, best, best_epoch, stale = state["epoch"] + 1, state["best"], state["best_epoch"], state["stale"]
        torch.set_rng_state(state["torch_rng"].cpu())
        np.random.set_state(state["numpy_rng"]); random.setstate(state["python_rng"])
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
    started = time.perf_counter()
    for epoch in range(start, 51):
        if stale >= 8:
            break
        model.train()
        totals = []
        for batch in train_loader:
            images, labels = plain(batch["image"], device), plain(batch["label"], device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits, head = model(images)
                base = dice(logits, labels)
                aux = base.new_zeros(()) if arm == "A" else step(logits, labels).loss if arm == "B" else head_loss(head, labels, arm == "D")[0]
                loss = base + weight * aux
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite training loss {arm} epoch {epoch}")
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            totals.append((float(loss.detach()), float(base.detach()), float(aux.detach())))
        score = selection_dice(model, val_loader, device)
        if not math.isfinite(score):
            raise FloatingPointError("nonfinite inner validation Dice")
        if score > best:
            best, best_epoch, stale = score, epoch, 0
            best_payload = {"epoch": epoch, "model": model.state_dict(), "inner_val_dice": score}
            atomic_save(best_payload, arm_dir / f"best_epoch_{epoch:03d}.pt")
            atomic_save(best_payload, arm_dir / "best.pt")
        else:
            stale += 1
        scheduler.step()
        record = {"arm": arm, "epoch": epoch, "train_total": float(np.mean(totals, axis=0)[0]),
                  "train_dice": float(np.mean(totals, axis=0)[1]), "train_aux": float(np.mean(totals, axis=0)[2]),
                  "inner_val_dice": score, "best_epoch": best_epoch, "stale": stale,
                  "elapsed_seconds": time.perf_counter() - started}
        with (arm_dir / "history.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        atomic_save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "best": best,
                     "best_epoch": best_epoch, "stale": stale, "torch_rng": torch.get_rng_state(),
                     "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                     "cuda_rng": torch.cuda.get_rng_state_all()}, latest)
        print(json.dumps(record), flush=True)
    selected_path = arm_dir / f"best_epoch_{best_epoch:03d}.pt"
    selected = torch.load(selected_path, map_location=device, weights_only=False)
    model.load_state_dict(selected["model"]); model.eval()
    rows = []
    if dev_loader is not None:
        prob_dir = arm_dir / "probabilities"
        prob_dir.mkdir(exist_ok=True)
        with torch.no_grad():
            for batch in dev_loader:
                name = str(batch["case_name"][0])
                logits, head = model(plain(batch["image"], device))
                probability = logits.float().softmax(1)[0].cpu().numpy().astype(np.float16)
                truth = batch["label"][0, 0].numpy().astype(np.uint8)
                head_prediction = head_mask(head, probability.astype(np.float32)) if head is not None else None
                np.savez_compressed(prob_dir / f"{name}.npz", probability=probability,
                                    hard_mask=probability.argmax(0).astype(np.uint8),
                                    **({"head_mask": head_prediction,
                                        "head_presence_probability": head[0].float().sigmoid()[0].cpu().numpy().astype(np.float16),
                                        "head_edge_probability": head[1].float().softmax(-1)[0].cpu().numpy().astype(np.float16)}
                                       if head_prediction is not None else {}))
                rows.append(case_row(arm, name, probability.astype(np.float32), truth, head_prediction))
        keys = sorted({key for row in rows for key in row})
        with (arm_dir / "case_metrics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(rows)
    done = {"arm": arm, "best_epoch": best_epoch, "inner_val_dice": best,
            "checkpoint": str(selected_path.resolve()), "checkpoint_sha256": sha256(selected_path),
            "aux_weight": weight, "runtime_seconds": time.perf_counter() - started,
            "dev_case_count": len(rows),
            "dev_union_dice_mean": float(np.mean([r["union_dice"] for r in rows])) if rows else None}
    (arm_dir / "done.json").write_text(json.dumps(done, indent=2))
    print(json.dumps(done), flush=True)


def summarize(root: Path):
    rows = []
    for arm in ARMS:
        with (root / arm / "case_metrics.csv").open() as f:
            rows.extend(csv.DictReader(f))
    keys = sorted({key for row in rows for key in row})
    with (root / "case_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(rows)
    by_arm = {arm: {r["patient"]: r for r in rows if r["arm"] == arm} for arm in ARMS}
    rng = np.random.default_rng(0)
    out = {}
    def paired(left: str, right: str, field: str):
        names = sorted(set(by_arm[left]) & set(by_arm[right]))
        diff = np.array([float(by_arm[left][n][field]) - float(by_arm[right][n][field]) for n in names])
        diff = diff[np.isfinite(diff)]
        if len(diff) == 0:
            return {"mean": math.nan, "ci95": [math.nan, math.nan], "n": 0}
        boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)])
        return {"mean": float(diff.mean()), "ci95": np.percentile(boot, [2.5, 97.5]).tolist(), "n": len(diff)}
    for arm in ARMS:
        for metric in ("union_dice", "posterior_dice", "anterior_dice", "hd95"):
            values = np.array([float(r[metric]) for r in by_arm[arm].values()])
            out[f"{arm}_{metric}_mean"] = float(values.mean())
        if arm != "A":
            out[f"{arm}_minus_A_union_dice"] = paired(arm, "A", "union_dice")
        if arm in {"C", "D"}:
            for field in ("disagreement_columns", "head_column_wins", "seg_column_wins", "column_ties", "voxels_fixed", "voxels_broken"):
                out[f"{arm}_{field}_total"] = sum(int(r[field]) for r in by_arm[arm].values())
            for edge in ("lower", "upper"):
                vals = np.array([float(r[f"head_{edge}_mae"]) - float(r[f"seg_{edge}_mae"])
                                 for r in by_arm[arm].values()])
                vals = vals[np.isfinite(vals)]
                boot = np.array([vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(2000)])
                out[f"{arm}_head_minus_seg_{edge}_mae"] = {
                    "mean": float(vals.mean()), "ci95": np.percentile(boot, [2.5, 97.5]).tolist(), "n": len(vals)}
            out[f"{arm}_corrected_minus_raw_union_dice"] = {
                "mean": float(np.mean([float(r["corrected_union_dice"]) - float(r["union_dice"])
                                  for r in by_arm[arm].values()]))}
    out["D_minus_C_union_dice"] = paired("D", "C", "union_dice")
    for field in ("head_lower_mae", "head_upper_mae", "seg_lower_mae", "seg_upper_mae"):
        out[f"D_minus_C_{field}"] = paired("D", "C", field)
    (root / "summary.json").write_text(json.dumps(out, indent=2))
    lines = ["# Sagittal contour head pilot", "", "Exploratory MSD fold-0 DEV cohort (52 previously studied cases). Checkpoints selected only on 42 inner-validation patients from the 208 outer-training cases.", "",
             "| Arm | Selected epoch | Inner-val Dice | DEV union Dice | DEV posterior Dice | DEV anterior Dice | HD95 (voxels) | Auxiliary weight | Runtime (h) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        done = json.loads((root / arm / "done.json").read_text())
        lines.append(f"| {arm} | {done['best_epoch']} | {done['inner_val_dice']:.4f} | {out[f'{arm}_union_dice_mean']:.4f} | {out[f'{arm}_posterior_dice_mean']:.4f} | {out[f'{arm}_anterior_dice_mean']:.4f} | {out[f'{arm}_hd95_mean']:.3f} | {done['aux_weight']:.6g} | {done['runtime_seconds']/3600:.2f} |")
    lines.extend(["", "Paired patient bootstrap intervals are percentile 95% intervals with 2,000 resamples. A positive Dice change favors the first arm; a negative edge MAE change favors the first arm.", ""])
    for arm in ("B", "C", "D"):
        metric = out[f"{arm}_minus_A_union_dice"]
        lines.append(f"- {arm} vs A union Dice: {metric['mean']:+.4f} (95% interval {metric['ci95'][0]:+.4f} to {metric['ci95'][1]:+.4f}; n={metric['n']}).")
    metric = out["D_minus_C_union_dice"]
    lines.append(f"- D vs C union Dice: {metric['mean']:+.4f} (95% interval {metric['ci95'][0]:+.4f} to {metric['ci95'][1]:+.4f}; n={metric['n']}).")
    for arm in ("C", "D"):
        for edge in ("lower", "upper"):
            metric = out[f"{arm}_head_minus_seg_{edge}_mae"]
            lines.append(f"- {arm} head minus its segmentation {edge} edge MAE: {metric['mean']:+.3f} voxels (95% interval {metric['ci95'][0]:+.3f} to {metric['ci95'][1]:+.3f}; n={metric['n']}).")
        lines.append(f"- {arm} disagreement: {out[f'{arm}_disagreement_columns_total']} columns; head closer {out[f'{arm}_head_column_wins_total']}, segmentation closer {out[f'{arm}_seg_column_wins_total']}, tied {out[f'{arm}_column_ties_total']}; correction fixed {out[f'{arm}_voxels_fixed_total']} and broke {out[f'{arm}_voxels_broken_total']} voxels.")
    lines.extend(["", "The head correction sets foreground exactly where predicted presence is at least 0.5 and rounded ordered expected z edges enclose the voxel; added voxels take the larger predicted foreground class probability. No reference occupancy is used at inference.",
                  "", "The D transition loss compares signed edge displacement for every ordered y pair inside each connected annotated run, with exp(-(distance-1)/4) weighting, SmoothL1 error, and a soft upward hinge allowing annotated upward excursions. The term is multiplied by 32; total auxiliary weight is calibrated from TRAIN decoder-feature gradients to a 0.1 auxiliary/Dice RMS ratio.",
                  "", "Fold 0 is exploratory DEV, not independent confirmation. Full per-patient results and paired changes are in case_metrics.csv and summary.json; selected checkpoints and hashes are in each arm's done.json, and source/data hashes in config.json."])
    (root / "result.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve()
    dependencies = (
        "baselines.swin_unetr.swin_unetr",
        "thesis.new_constraints.sagittal_step_fit",
        "thesis.new_constraints.directional_steps",
        "thesis.new_constraints.constraint_result",
        "thesis.new_constraints.supervised",
    )
    source_manifest = {str(source): sha256(source)}
    for module_name in dependencies:
        path = Path(importlib.import_module(module_name).__file__).resolve()
        source_manifest[str(path)] = sha256(path)
    config = {"source_manifest": source_manifest, "pkl_sha256": sha256(args.pkl), "splits_sha256": sha256(args.splits),
              "seed": 0, "outer_fold": 0, "train_count": 166, "inner_val_count": 42, "outer_dev_count": 52,
              "recipe": {"loss": "MONAI DiceLoss(to_onehot_y=True,softmax=True)", "epochs": 50,
                         "batch_size": 2, "optimizer": "AdamW", "learning_rate": 1e-4, "weight_decay": 1e-5,
                         "step_size": 20, "gamma": 0.5, "patience": 8, "min_delta": 0,
                         "augmentation": "none", "selection": "inner-val foreground-class hard Dice"},
              "head": {"presence_threshold": 0.5, "edge_decode": "ordered expected z, rounded for mask",
                       "transition": "32 times all ordered y pairs in each connected annotated run; exponentially distance-weighted SmoothL1 of signed edge displacement error plus 0.25 Softplus(2*(predicted rise - max(annotated rise,0)))/2; normalized by z length"}}
    cfg = args.output / "config.json"
    if cfg.exists():
        assert json.loads(cfg.read_text()) == config, "source/data/config changed during resume"
    else:
        cfg.write_text(json.dumps(config, indent=2))
    train, val, dev = loaders(args, args.output)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.smoke:
        batch = next(iter(train))
        images, labels = plain(batch["image"], device), plain(batch["label"], device)
        model = PilotModel("D").to(device)
        logits, head = model(images)
        loss = build_supervised_loss("dice")(logits, labels) + head_loss(head, labels, True)[0]
        assert torch.isfinite(loss)
        loss.backward()
        assert model.backbone.swinViT.patch_embed.proj.weight.grad is not None
        print(json.dumps({"smoke": "ok", "loss": float(loss), "gpu_max_allocated": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0}), flush=True)
        return
    for arm in ARMS:
        train_arm(args, arm, args.output, train, val, dev, device)
    summarize(args.output)


if __name__ == "__main__":
    main()
