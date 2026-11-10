#!/usr/bin/env python3
"""Train one of the 27 fixed-data, fixed-compute baseline replicates."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import pydoc
import random
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "baselines/nnUNet"))

import monai
import numpy as np
import torch
from monai.losses import DiceLoss
from monai.networks.nets import UNet, SwinUNETR
from torch.utils.data import DataLoader, Dataset
from baselines.swin_unetr.swin_unetr import _build_monai_dataset_from_pkl, _load_pkl_dataframe
_augmentation_spec = importlib.util.spec_from_file_location(
    "compute_matched_augmentation", REPO / "thesis/new_constraints/training_augmentation.py")
_augmentation_module = importlib.util.module_from_spec(_augmentation_spec)
_augmentation_spec.loader.exec_module(_augmentation_module)
augment_mild = _augmentation_module.augment_mild

DEFAULT_PROTOCOL = REPO / "experiments/compute_matched_baselines_20261003"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def save_checkpoint(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def write_metrics(path: Path, history: list[dict]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    os.replace(temporary, path)


def array_identity(index: int) -> tuple[str, int, int]:
    if not 0 <= index < 27:
        raise ValueError("Array index must be in 0..26")
    return ("unet", "swinunetr", "nnunet")[index // 9], (index % 9) // 3, index % 3


def validation_steps(max_steps: int, count: int = 20) -> list[int]:
    return sorted({math.ceil(k * max_steps / count) for k in range(1, count + 1)})


def learning_rate(model: str, step: int, max_steps: int, initial: float,
                  nnunet_warmup_fraction: float = 0.0) -> float:
    """Budget-aware LR for the upcoming update, indexed from zero."""
    if model == "nnunet":
        if not 0 <= nnunet_warmup_fraction < 1:
            raise ValueError("nnU-Net warmup fraction must be in [0, 1)")
        warmup = min(max_steps - 1, math.ceil(nnunet_warmup_fraction * max_steps))
        if step < warmup:
            return initial * (step + 1) / warmup
        return initial * (1 - (step - warmup) / (max_steps - warmup)) ** 0.9
    warmup = max(1, math.ceil(0.05 * max_steps))
    if step < warmup:
        return initial * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup - 1)
    return initial * (0.01 + 0.99 * (1 + math.cos(math.pi * progress)) / 2)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model(name: str, architecture: dict) -> torch.nn.Module:
    if name == "unet":
        return UNet(spatial_dims=3, in_channels=1, out_channels=3,
                    channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2)
    if name == "swinunetr":
        return SwinUNETR(in_channels=1, out_channels=3, use_checkpoint=False, drop_rate=0.0)
    from dynamic_network_architectures.architectures.unet import PlainConvUNet
    kwargs = dict(architecture["kwargs"])
    for key in ("conv_op", "norm_op", "nonlin"):
        kwargs[key] = pydoc.locate(kwargs[key])
        if kwargs[key] is None:
            raise ImportError(f"Cannot locate architecture operation {key}")
    network = PlainConvUNet(**kwargs)
    network.apply(network.initialize)
    return network


def build_loss_and_optimizer(name: str, model: torch.nn.Module, protocol: dict):
    if name != "nnunet":
        loss = DiceLoss(to_onehot_y=True, softmax=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["initial_lrs"][name],
                                      weight_decay=protocol["adamw_weight_decay"], foreach=False)
        return loss, optimizer
    from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
    from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
    from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
    loss = DC_and_CE_loss({"batch_dice": True, "smooth": 1e-5, "do_bg": False, "ddp": False}, {},
                          dice_class=MemoryEfficientSoftDiceLoss)
    weights = np.array([1.0, 0.5, 0.25, 0.0])
    loss = DeepSupervisionWrapper(loss, weights / weights.sum())
    momentum = protocol.get("nnunet_sgd_momentum", 0.99)
    if not 0 < momentum < 1:
        raise ValueError("nnU-Net Nesterov momentum must be in (0, 1)")
    optimizer = torch.optim.SGD(model.parameters(), lr=protocol["initial_lrs"][name],
                                weight_decay=protocol["sgd_weight_decay"],
                                momentum=momentum, nesterov=True, foreach=False)
    return loss, optimizer


class PreparedPatients(Dataset):
    """Cache the historical normalization, padding and central-crop transform."""
    def __init__(self, dataframe, names: list[str]):
        subset = dataframe[dataframe["subject_id"].astype(str).isin(names)]
        if len(subset) != len(names):
            raise ValueError("Missing or duplicate requested patient IDs in pickle")
        for _, row in subset.iterrows():
            image, label = np.asarray(row["image_data"]), np.asarray(row["label_data"])
            if image.ndim != 3 or image.shape != label.shape or not np.isfinite(image).all():
                raise ValueError("Invalid paired raw image/label shape or intensity")
            if not np.isfinite(label).all() or not np.equal(label, np.rint(label)).all() or not set(np.unique(label)).issubset({0, 1, 2}):
                raise ValueError("Labels must be finite integers in {0,1,2}")
        dataset = _build_monai_dataset_from_pkl(subset, "MSD", 3, spatial_size=(64, 64, 64), do_resize=False)
        by_name = {}
        for item in dataset:
            for key in ("image", "label"):
                if hasattr(item[key], "as_tensor"):
                    item[key] = item[key].as_tensor()
                if tuple(item[key].shape) != (1, 64, 64, 64):
                    raise ValueError("Preprocessed inputs differ from profiled shape")
            by_name[item["case_name"]] = item
        self.items = [by_name[name] for name in names]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


@torch.no_grad()
def evaluate(model, dataset, device, amp: bool) -> dict:
    model.eval()
    records = []
    for batch in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
        image = batch["image"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext():
            logits = model(image)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        probabilities = logits.float().softmax(1)
        if not torch.isfinite(probabilities).all():
            raise FloatingPointError("Non-finite validation output")
        prediction = probabilities.argmax(1)
        label = batch["label"].to(device)[:, 0]
        scores = {}
        for c, name in [(1, "anterior"), (2, "posterior")]:
            gt = label == c
            hard = prediction == c
            soft = probabilities[:, c]
            if gt.sum() == 0:
                raise ValueError("Foreground class missing in a validation case")
            scores[name + "_hard_dice"] = float(2 * (hard & gt).sum() / (hard.sum() + gt.sum()))
            scores[name + "_soft_dice"] = float(2 * (soft * gt).sum() / (soft.sum() + gt.sum()))
        records.append({"case_name": batch["case_name"][0], **scores})
    if not records:
        raise ValueError("No validation cases")
    means = {key: float(np.mean([r[key] for r in records])) for key in records[0] if key != "case_name"}
    means["macro_hard_dice"] = (means["anterior_hard_dice"] + means["posterior_hard_dice"]) / 2
    means["macro_soft_dice"] = (means["anterior_soft_dice"] + means["posterior_soft_dice"]) / 2
    return {"summary": means, "cases": records}


def run(args):
    protocol = json.loads((args.protocol_dir / "protocol.json").read_text())
    schedule = protocol["models"][args.model]
    budget, cost = protocol["budget_flops"], schedule["flops_per_update"]
    nnunet_warmup_fraction = protocol.get("nnunet_warmup_fraction", 0.0)
    max_steps = budget // cost
    if args.smoke_steps:
        max_steps = min(max_steps, args.smoke_steps)
    if args.seed not in protocol["seeds"] or args.fold not in protocol["folds"]:
        raise ValueError("Run identity outside frozen comparison protocol")
    for name, path in [("splits", args.protocol_dir / "splits_10patients.json"),
                       ("nnunet_architecture", args.protocol_dir / "nnunet_architecture.json")]:
        if digest(path) != protocol[name + "_sha256"]:
            raise ValueError(f"Frozen {name} artifact changed")
    if not args.smoke_steps and digest(args.pkl) != protocol["pkl_sha256"]:
        raise ValueError("Dataset hash differs from original reference")
    if monai.__version__ != protocol["required_versions"]["monai"]:
        raise RuntimeError("MONAI version differs from profiled model")
    if args.model == "nnunet":
        version = importlib.metadata.version("dynamic-network-architectures")
        parts = tuple(int(x) for x in version.split(".")[:3])
        if not (0, 4, 1) <= parts < (0, 5, 0):
            raise RuntimeError("Profile-compatible dynamic-network-architectures 0.4.x required")
    device = torch.device(args.device)
    amp = device.type == "cuda"
    if amp and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("CUDA bfloat16 support required for the production protocol")
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    architecture = json.loads((args.protocol_dir / "nnunet_architecture.json").read_text())
    model = build_model(args.model, architecture).to(device)
    if sum(p.numel() for p in model.parameters()) != schedule["parameters"]:
        raise ValueError("Model parameter count differs from profiled architecture")
    criterion, optimizer = build_loss_and_optimizer(args.model, model, protocol)
    splits = json.loads((args.protocol_dir / "splits_10patients.json").read_text())
    split = splits[args.fold]
    if len(split["train"]) != 10 or len(set(split["train"])) != 10 or set(split["train"]) & set(split["val"]):
        raise ValueError("Expected ten distinct training patients and disjoint validation")
    dataframe = _load_pkl_dataframe(args.pkl)
    train = PreparedPatients(dataframe, split["train"])
    validation = PreparedPatients(dataframe, split["val"])
    augmentation_rng = torch.Generator().manual_seed(args.seed + 2000)
    configuration = {"model": args.model, "fold": args.fold, "seed": args.seed,
                     "budget_flops": budget, "flops_per_update": cost, "max_steps": max_steps,
                     "smoke_test": bool(args.smoke_steps), "protocol_sha256": digest(args.protocol_dir / "protocol.json"),
                     "dataset_sha256": digest(args.pkl), "train_cases": split["train"], "validation_cases": split["val"],
                     "torch": torch.__version__, "monai": monai.__version__,
                     "dynamic_network_architectures": importlib.metadata.version("dynamic-network-architectures") if args.model == "nnunet" else None,
                     "augmentation": protocol["augmentation"], "script_sha256": digest(Path(__file__)),
                     "precision": protocol["precision"], "initial_lr": protocol["initial_lrs"][args.model],
                     "nnunet_warmup_fraction": nnunet_warmup_fraction if args.model == "nnunet" else 0.0,
                     "nnunet_sgd_momentum": optimizer.param_groups[0]["momentum"] if args.model == "nnunet" else None,
                     "selection": protocol["checkpoint_selection"]}
    config_path = args.output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise ValueError("Resume configuration differs from saved run")
    atomic_json(config_path, configuration)
    if (args.output / "completion.json").exists():
        completed = json.loads((args.output / "completion.json").read_text())
        if completed["config_sha256"] != digest(config_path) or any(
            completed[key] != digest(args.output / filename) for key, filename in [
                ("checkpoint_best_sha256", "checkpoint_best.pt"),
                ("checkpoint_latest_sha256", "checkpoint_latest.pt"), ("metrics_sha256", "metrics.csv"),
                ("budget_ledger_sha256", "budget_ledger.json"),
                ("validation_best_sha256", "validation_best.json")]):
            raise ValueError("Completed-run artifact binding changed")
        print("Verified completed run; no further training", flush=True)
        return
    step, best_score, best_step = 0, -1.0, 0
    permutation, epoch, cursor = [], 0, 0
    losses, history = [], []
    ledger_path = args.output / "budget_ledger.json"
    charged = 0
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
        if ledger["config_sha256"] != digest(config_path):
            raise ValueError("Budget ledger belongs to a different configuration")
        charged = ledger["charged_attempts"]
        if not args.resume:
            raise ValueError("Existing budget ledger requires --resume")
    if not isinstance(charged, int) or not 0 <= charged <= max_steps:
        raise ValueError("Invalid charged training-attempt count")
    if args.resume and (args.output / "checkpoint_latest.pt").exists():
        checkpoint = torch.load(args.output / "checkpoint_latest.pt", map_location=device, weights_only=False)
        if checkpoint["configuration"] != configuration:
            raise ValueError("Checkpoint configuration differs")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step, best_score, best_step = checkpoint["step"], checkpoint["best_score"], checkpoint["best_step"]
        epoch, permutation, cursor = checkpoint["epoch"], checkpoint["permutation"], checkpoint["cursor"]
        history, losses = checkpoint["history"], checkpoint["losses"]
        if not step <= checkpoint["charged_attempts"] <= charged:
            raise ValueError("Checkpoint and budget ledger disagree")
        # Restore the selected model from the committed checkpoint transaction.
        # A crash while publishing the separate best file cannot change selection.
        save_checkpoint(args.output / "checkpoint_best.pt", checkpoint["best_checkpoint"])
        atomic_json(args.output / "validation_best.json", checkpoint["best_validation"])
        atomic_json(args.output / "validation_latest.json", checkpoint["latest_validation"])
        write_metrics(args.output / "metrics.csv", history)
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if amp:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
        augmentation_rng.set_state(checkpoint["augmentation_rng"].cpu())
    elif (args.output / "checkpoint_latest.pt").exists():
        raise ValueError("Existing checkpoint requires --resume")
    milestones = validation_steps(max_steps, protocol["validation_milestones"])
    started = time.perf_counter()

    def persist_validation(lr):
        nonlocal best_score, best_step, losses
        result = evaluate(model, validation, device, amp)
        score = result["summary"]["macro_hard_dice"]
        improved = score > best_score
        if improved:
            best_score, best_step = score, step
            best_checkpoint = {"configuration": configuration, "model": model.state_dict(),
                               "step": step, "charged_attempts": charged, "score": score}
            best_validation = result
        else:
            best_checkpoint = torch.load(args.output / "checkpoint_best.pt", map_location=device, weights_only=False)
            best_validation = json.loads((args.output / "validation_best.json").read_text())
        row = {"step": step, "charged_attempts": charged, "patient_passes": step / 10,
               "estimated_training_flops": charged * cost, "compute_fraction": charged * cost / budget,
               "mean_train_loss": float(np.mean(losses)) if losses else None,
               "learning_rate": lr, **result["summary"]}
        history.append(row)
        losses = []
        payload = {"configuration": configuration, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "step": step, "charged_attempts": charged, "best_score": best_score, "best_step": best_step,
                   "best_checkpoint": best_checkpoint, "best_validation": best_validation,
                   "latest_validation": result,
                   "epoch": epoch, "permutation": permutation, "cursor": cursor,
                   "history": history, "losses": losses, "torch_rng": torch.get_rng_state(),
                   "cuda_rng": torch.cuda.get_rng_state_all() if amp else [],
                   "augmentation_rng": augmentation_rng.get_state()}
        save_checkpoint(args.output / "checkpoint_latest.pt", payload)
        save_checkpoint(args.output / "checkpoint_best.pt", best_checkpoint)
        atomic_json(args.output / "validation_best.json", best_validation)
        atomic_json(args.output / "validation_latest.json", result)
        write_metrics(args.output / "metrics.csv", history)
        print(json.dumps({"model": args.model, "fold": args.fold, "seed": args.seed, **row}), flush=True)

    while charged < max_steps:
        if cursor == len(permutation):
            # Epoch-indexed independent shuffle makes resume exact without prefetching.
            permutation = torch.randperm(10, generator=torch.Generator().manual_seed(args.seed + 1000 + epoch)).tolist()
            epoch += 1
            cursor = 0
        batch = train[permutation[cursor]]
        images, labels = batch["image"].unsqueeze(0), batch["label"].unsqueeze(0)
        if protocol["augmentation"] == "mild_v1":
            images, labels, _ = augment_mild(images, labels, generator=augmentation_rng)
        images, labels = images.to(device), labels.to(device)
        lr = learning_rate(args.model, charged, max_steps, protocol["initial_lrs"][args.model], nnunet_warmup_fraction)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        model.train()
        # Reserve a full step before forward. Attempts lost to interruption remain
        # charged on resume, so replay never silently increases the compute cap.
        charged += 1
        atomic_json(ledger_path, {"config_sha256": digest(config_path), "charged_attempts": charged,
                                 "charged_training_flops": charged * cost})
        with torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext():
            output = model(images)
            if args.model == "nnunet":
                targets = [torch.nn.functional.interpolate(labels.float(), size=o.shape[2:], mode="nearest").long() for o in output]
                loss = criterion(output, targets)
            else:
                loss = criterion(output, labels) + 0.2 * (output.float().reshape(-1)[0] * 0.0)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        if args.model == "nnunet":
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12, foreach=False, error_if_nonfinite=True)
        optimizer.step()
        step += 1
        cursor += 1
        losses.append(float(loss.detach()))
        if charged in milestones:
            persist_validation(lr)
    if not history or history[-1]["charged_attempts"] != charged:
        persist_validation(learning_rate(args.model, max(0, charged - 1), max_steps,
                                         protocol["initial_lrs"][args.model], nnunet_warmup_fraction))
    atomic_json(args.output / "completion.json", {"status": "smoke_complete" if args.smoke_steps else "complete",
                "model": args.model, "fold": args.fold, "seed": args.seed, "updates": step,
                "complete_patient_pass_epochs": step // 10, "additional_updates": step % 10,
                "budget_flops": budget, "training_attempts_charged": charged,
                "discarded_attempts": charged - step, "consumed_training_flops": charged * cost,
                "unused_training_flops": budget - charged * cost, "selected_step": best_step,
                "best_validation_hard_dice": best_score, "seconds_this_invocation": time.perf_counter() - started,
                "config_sha256": digest(config_path), "checkpoint_best_sha256": digest(args.output / "checkpoint_best.pt"),
                "checkpoint_latest_sha256": digest(args.output / "checkpoint_latest.pt"),
                "metrics_sha256": digest(args.output / "metrics.csv"),
                "budget_ledger_sha256": digest(ledger_path),
                "validation_best_sha256": digest(args.output / "validation_best.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("unet", "swinunetr", "nnunet"), required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--pkl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=0)
    args = parser.parse_args()
    if args.smoke_steps < 0 or args.threads < 1:
        parser.error("Smoke steps must be nonnegative and threads positive")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args)
        except BaseException:
            atomic_json(args.output / "failure.json", {"traceback": traceback.format_exc()})
            raise


if __name__ == "__main__":
    main()
