#!/usr/bin/env python3
"""Optimization-only training cap; calibration accounted for separately."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import fcntl
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
import monai

import train_compute_matched_baselines as base
from scripts.capped_depth_delay.objectives import ARMS, calibration_case, select_weights, training_loss, geometry_weights, augmented_affine
from thesis.new_constraints.signed_distance import signed_distance_maps
from optimization90 import SCOPE
from scripts.capped_depth_delay.schedule import charge_for_step, training_flops, sdf_scale_for_step

REPO = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = REPO / "experiments/compute_matched_constraints_20pct_20261004"
ARTIFACTS = {
    "config_sha256": "config.json", "checkpoint_best_sha256": "checkpoint_best.pt",
    "checkpoint_latest_sha256": "checkpoint_latest.pt", "metrics_sha256": "metrics.csv",
    "budget_ledger_sha256": "budget_ledger.json", "validation_best_sha256": "validation_best.json",
    "calibration_sha256": "calibration.json",
}


def model_digest(model):
    hasher = hashlib.sha256()
    for name, value in model.state_dict().items():
        hasher.update(name.encode())
        hasher.update(value.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def calibrate(model, patients, arm, device, amp, native_criterion=None, protocol=None):
    mode = model.training
    anchor = model_digest(model)
    saved_torch = torch.get_rng_state()
    saved_cuda = torch.cuda.get_rng_state_all() if amp else []
    rows = []
    try:
        model.eval()
        for patient in patients:
            sdf = signed_distance_maps(patient["label"][None]).to(device) if "sdf" in arm else None
            image = patient["image"][None].to(device)
            label = patient["label"][None].to(device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16) if amp else torch.no_grad():
                raw = model(image)
            depth = (geometry_weights(label, protocol['native_geometry'][patient['case_name']]['model_affine'], protocol['depth_geometry'])
                     if 'depth' in arm else None)
            row = calibration_case(raw, label, arm, sdf, native_criterion, depth)
            row["case_name"] = patient["case_name"]
            rows.append(row)
        weights, statistics = select_weights(rows, arm)
        if model_digest(model) != anchor:
            raise ValueError("Calibration changed model parameters or buffers")
        return {"status": "complete", "weights": weights, "statistics": statistics,
                "cases": rows, "anchor_model_sha256": anchor,
                "target_ratio": .1, "p95_cap": .5,
                "scope": ("All ten training volumes, no augmentation or validation; native Dice+CE gradient including the full-resolution deep-supervision weight versus constraint gradients at detached FP32 full-resolution logits; other native heads retained" if native_criterion is not None else
                          "All training patients, no augmentation or validation; gradients with respect to detached FP32 logits")}
    finally:
        model.train(mode)
        torch.set_rng_state(saved_torch)
        if amp:
            torch.cuda.set_rng_state_all(saved_cuda)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((args.protocol_dir / "protocol.json").read_text())
    schedule = protocol["models"][args.model]["objectives"][args.objective]
    budget = protocol["budget_flops"]
    if protocol.get("budget_scope") != SCOPE:
        raise ValueError("Optimization-only accounting must be explicit")
    max_steps = min(schedule["max_steps"], args.smoke_steps) if args.smoke_steps else schedule["max_steps"]
    if args.seed not in protocol["seeds"] or args.fold not in protocol["folds"]:
        raise ValueError("Identity outside campaign protocol")
    for key, filename in (("splits_sha256", "splits_10patients.json"),
                          ("nnunet_architecture_sha256", "nnunet_architecture.json")):
        if base.digest(args.protocol_dir / filename) != protocol[key]:
            raise ValueError("Frozen campaign artifact changed")
    if not args.smoke_steps and base.digest(args.pkl) != protocol["pkl_sha256"]:
        raise ValueError("Dataset hash changed")
    if monai.__version__ != "1.6.0":
        raise ValueError("Use the profiled MONAI 1.6.0")
    device = torch.device(args.device)
    amp = device.type == "cuda"
    if amp and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise ValueError("Allocated BF16-capable GPU required")
    base.seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    architecture = json.loads((args.protocol_dir / "nnunet_architecture.json").read_text())
    model = base.build_model(args.model, architecture).to(device)
    if sum(p.numel() for p in model.parameters()) != protocol["models"][args.model]["parameters"]:
        raise ValueError("Model differs from profiled architecture")
    criterion, optimizer = base.build_loss_and_optimizer(args.model, model, protocol)
    native_criterion = criterion if args.model == "nnunet" else None
    splits = json.loads((args.protocol_dir / "splits_10patients.json").read_text())
    split = splits[args.fold]
    if len(set(split["train"])) != 10 or set(split["train"]) & set(split["val"]):
        raise ValueError("Expected ten distinct training patients and disjoint validation")
    dataframe = base._load_pkl_dataframe(args.pkl)
    train = base.PreparedPatients(dataframe, split["train"])
    validation = base.PreparedPatients(dataframe, split["val"])
    augmentation_rng = torch.Generator().manual_seed(args.seed + 2000)
    configuration = {
        "budget_scope": SCOPE,
        "model": args.model, "model_identity": protocol["model_identity"], "objective": args.objective, "fold": args.fold, "seed": args.seed,
        "lr_horizon_updates": protocol["lr_horizon_updates"], "budget_flops": budget, "schedule": schedule, "max_steps": max_steps,
        "smoke_test": bool(args.smoke_steps), "protocol_sha256": base.digest(args.protocol_dir / "protocol.json"),
        "dataset_sha256": base.digest(args.pkl), "train_cases": split["train"], "validation_cases": split["val"],
        "torch": torch.__version__, "monai": monai.__version__, "augmentation": protocol["augmentation"],
        "script_sha256": base.digest(Path(__file__)),
        "objective_source_sha256": base.digest(REPO / "scripts/compute_constraint_objectives.py"),
        "shared_source_sha256": base.digest(REPO / "scripts/train_compute_matched_baselines.py"),
        "sdf_source_sha256": base.digest(REPO / "thesis/new_constraints/signed_distance.py"),
        "precision": protocol["precision"], "initial_lr": protocol["initial_lrs"][args.model],
        "constraint_warmup_updates": protocol["constraint_warmup_updates"],
        "sdf_prefix_updates": schedule["sdf_prefix_updates"],
        "sdf_timing_sha256": base.digest(REPO / "scripts/capped_depth_delay/schedule.py"),
        "combined_objective_sha256": base.digest(REPO / "scripts/capped_depth_delay/objectives.py"),
        "depth_geometry_sha256": base.digest(REPO / "scripts/depth_supervision/geometry.py"),
        "selection": protocol["checkpoint_selection"],
    }
    if native_criterion is not None:
        configuration.update(native_loss="Dice+CE with unchanged deep supervision; auxiliary constraints only at full resolution",
                             native_objective_sha256=base.digest(REPO / "scripts/nnunet_multi_objective.py"),
                             nnunet_sgd_momentum=optimizer.param_groups[0]["momentum"],
                             nnunet_warmup_fraction=protocol.get("nnunet_warmup_fraction", 0.0))
    config_path = args.output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise ValueError("Resume configuration differs")
    base.atomic_json(config_path, configuration)
    if (args.output / "completion.json").exists():
        done = json.loads((args.output / "completion.json").read_text())
        if any(done[k] != base.digest(args.output / filename) for k, filename in ARTIFACTS.items()):
            raise ValueError("Completed artifact binding changed")
        print("Verified completed run", flush=True)
        return
    ledger_path = args.output / "budget_ledger.json"
    ledger = {"config_sha256": base.digest(config_path), "charged_attempts": 0,
              "training_flops": 0, "calibration_flops": 0, "calibration_attempts": 0}
    if ledger_path.exists():
        if not args.resume:
            raise ValueError("Existing ledger requires --resume")
        ledger = json.loads(ledger_path.read_text())
        if ledger["config_sha256"] != base.digest(config_path):
            raise ValueError("Ledger configuration differs")
    if any(not isinstance(ledger[k], int) or ledger[k] < 0 for k in
           ("charged_attempts", "training_flops", "calibration_flops", "calibration_attempts")):
        raise ValueError("Invalid budget ledger")
    if ledger["training_flops"] > budget:
        raise ValueError("Budget already exceeded")
    expected_training = training_flops(ledger["charged_attempts"], schedule)
    if (ledger["charged_attempts"] > max_steps or ledger["training_flops"] != expected_training
            or ledger["calibration_flops"] != ledger["calibration_attempts"] * schedule["calibration_flops"]):
        raise ValueError("Ledger arithmetic differs from the frozen schedule")
    step, best_score, best_step, epoch, cursor = 0, -1., 0, 0, 0
    permutation, losses, history = [], [], []
    best_checkpoint = best_validation = latest_validation = None
    calibration_path = args.output / "calibration.json"
    weights = {}
    latest = args.output / "checkpoint_latest.pt"
    if latest.exists():
        if not args.resume:
            raise ValueError("Existing checkpoint requires --resume")
        cp = torch.load(latest, map_location=device, weights_only=False)
        if cp["configuration"] != configuration or cp["charged_attempts"] > ledger["charged_attempts"]:
            raise ValueError("Checkpoint/ledger differs")
        model.load_state_dict(cp["model"])
        optimizer.load_state_dict(cp["optimizer"])
        step, best_score, best_step = cp["step"], cp["best_score"], cp["best_step"]
        epoch, permutation, cursor = cp["epoch"], cp["permutation"], cp["cursor"]
        history, losses = cp["history"], cp["losses"]
        best_checkpoint, best_validation, latest_validation = cp["best_checkpoint"], cp["best_validation"], cp["latest_validation"]
        if best_checkpoint is not None:
            base.save_checkpoint(args.output / "checkpoint_best.pt", best_checkpoint)
            base.atomic_json(args.output / "validation_best.json", best_validation)
        if latest_validation is not None:
            base.atomic_json(args.output / "validation_latest.json", latest_validation)
        if history:
            base.write_metrics(args.output / "metrics.csv", history)
        torch.set_rng_state(cp["torch_rng"].cpu())
        if amp:
            torch.cuda.set_rng_state_all([v.cpu() for v in cp["cuda_rng"]])
        augmentation_rng.set_state(cp["augmentation_rng"].cpu())
    if calibration_path.exists():
        report = json.loads(calibration_path.read_text())
        if report["config_sha256"] != base.digest(config_path):
            raise ValueError("Calibration belongs to another run")
        if "calibration_sha256" in ledger and ledger["calibration_sha256"] != base.digest(calibration_path):
            raise ValueError("Calibration artifact changed")
        weights = report["weights"]
    elif args.objective == "dice":
        base.atomic_json(calibration_path, {"status": "not_required", "weights": {},
                                           "config_sha256": base.digest(config_path)})
    if calibration_path.exists():
        ledger["calibration_sha256"] = base.digest(calibration_path)
    milestones = base.validation_steps(max_steps, protocol["validation_milestones"])
    started = time.perf_counter()

    def commit():
        payload = {"configuration": configuration, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "step": step, "charged_attempts": ledger["charged_attempts"], "best_score": best_score, "best_step": best_step,
                   "best_checkpoint": best_checkpoint, "best_validation": best_validation, "latest_validation": latest_validation,
                   "epoch": epoch, "permutation": permutation, "cursor": cursor,
                   "history": history, "losses": losses, "torch_rng": torch.get_rng_state(),
                   "cuda_rng": torch.cuda.get_rng_state_all() if amp else [],
                   "augmentation_rng": augmentation_rng.get_state()}
        base.save_checkpoint(latest, payload)
        if best_checkpoint is not None:
            base.save_checkpoint(args.output / "checkpoint_best.pt", best_checkpoint)
            base.atomic_json(args.output / "validation_best.json", best_validation)
        if latest_validation is not None:
            base.atomic_json(args.output / "validation_latest.json", latest_validation)
        if history:
            base.write_metrics(args.output / "metrics.csv", history)

    def validate(lr):
        nonlocal best_score, best_step, best_checkpoint, best_validation, latest_validation, losses
        result = base.evaluate(model, validation, device, amp)
        score = result["summary"]["macro_hard_dice"]
        if score > best_score:
            best_score, best_step = score, step
            best_checkpoint = {"configuration": configuration,
                               "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                               "step": step, "score": score}
            best_validation = result
        latest_validation = result
        row = {"step": step, "charged_attempts": ledger["charged_attempts"], "patient_passes": step / 10,
               "estimated_training_flops": ledger["training_flops"], "calibration_flops": ledger["calibration_flops"],
               "total_flops": ledger["training_flops"] + ledger["calibration_flops"],
               "compute_fraction": ledger["training_flops"] / budget,
               "mean_train_loss": float(np.mean(losses)) if losses else None, "learning_rate": lr,
               **result["summary"]}
        history.append(row)
        losses = []
        commit()
        print(json.dumps({"model": args.model, "objective": args.objective, "fold": args.fold, "seed": args.seed, **row}), flush=True)

    while ledger["charged_attempts"] < max_steps:
        charged = ledger["charged_attempts"]
        if args.objective != "dice" and charged >= schedule["prefix_updates"] and not calibration_path.exists():
            commit()  # Save the model before any calibration attempt.
            cal_cost = schedule["calibration_flops"]
            ledger["calibration_flops"] += cal_cost
            ledger["calibration_attempts"] += 1
            base.atomic_json(ledger_path, ledger)
            if native_criterion is None:
                report = calibrate(model, train, args.objective, device, amp, protocol=protocol)
            else:
                report = calibrate(model, train, args.objective, device, amp, native_criterion, protocol)
            if [r["case_name"] for r in report["cases"]] != split["train"]:
                raise ValueError("Calibration case order differs from frozen training subset")
            report.update(config_sha256=base.digest(config_path), anchor_successful_updates=step,
                          anchor_training_attempts=charged, charged_flops=cal_cost)
            base.atomic_json(calibration_path, report)
            ledger["calibration_sha256"] = base.digest(calibration_path)
            base.atomic_json(ledger_path, ledger)
            weights = report["weights"]
            print(json.dumps({"calibration": "complete", "model": args.model, "objective": args.objective,
                              "fold": args.fold, "seed": args.seed, "weights": weights}), flush=True)
            commit()
        cost = charge_for_step(charged, schedule)
        if ledger["training_flops"] + cost > budget:
            break
        if cursor == len(permutation):
            permutation = torch.randperm(10, generator=torch.Generator().manual_seed(args.seed + 1000 + epoch)).tolist()
            epoch += 1
            cursor = 0
        batch = train[permutation[cursor]]
        image, label = batch["image"][None], batch["label"][None]
        original_affine = np.asarray(protocol['native_geometry'][batch['case_name']]['model_affine'])
        current_affine = original_affine
        if protocol["augmentation"] == "mild_v1":
            saved_augmentation = augmentation_rng.get_state()
            image, label, augmentation_statistics = base.augment_mild(image, label, generator=augmentation_rng)
            current_affine = augmented_affine(original_affine, saved_augmentation, augmentation_statistics, label.shape[-3:])
        active_arm = "dice" if charged < schedule["prefix_updates"] else args.objective
        if charged < schedule["sdf_prefix_updates"]:
            active_arm = active_arm.replace("_sdf", "")
        depth = geometry_weights(label, current_affine, protocol['depth_geometry']).to(device) if 'depth' in active_arm else None
        sdf = signed_distance_maps(label).to(device) if "sdf" in active_arm else None
        image, label = image.to(device), label.to(device)
        effective_max = max_steps
        lr = base.learning_rate(args.model, min(charged, effective_max - 1), effective_max, protocol["initial_lrs"][args.model])
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        model.train()
        ledger["charged_attempts"] += 1
        ledger["training_flops"] += cost
        base.atomic_json(ledger_path, ledger)
        scale = .2 if active_arm == "dice" else min(1., (charged - schedule["prefix_updates"] + 1) / protocol["constraint_warmup_updates"])
        with torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext():
            logits = model(image)
            loss = training_loss(logits, label, active_arm, weights, scale, sdf,
                                 native_criterion, depth,
                                 sdf_scale_for_step(charged, schedule) if 'sdf' in active_arm else 0.)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        if native_criterion is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12, foreach=False, error_if_nonfinite=True)
        optimizer.step()
        step += 1
        cursor += 1
        losses.append(float(loss.detach()))
        if ledger["charged_attempts"] in milestones:
            validate(lr)
        elif ledger["charged_attempts"] == schedule["prefix_updates"]:
            commit()
    if not history or history[-1]["charged_attempts"] != ledger["charged_attempts"]:
        validate(base.learning_rate(args.model, max(0, min(ledger["charged_attempts"] - 1, max_steps - 1)), max_steps,
                                    protocol["initial_lrs"][args.model]))
    total = ledger["training_flops"] + ledger["calibration_flops"]
    if not args.smoke_steps and (budget - ledger["training_flops"] >= schedule["flops_per_update"] or
                                args.objective != "dice" and not calibration_path.exists()):
        raise ValueError("Production run ended before exhausting its allowance")
    base.atomic_json(args.output / "completion.json", {
        "status": "smoke_complete" if args.smoke_steps else "complete", "model": args.model,
        "objective": args.objective, "fold": args.fold, "seed": args.seed, "updates": step,
        "training_attempts_charged": ledger["charged_attempts"], "discarded_attempts": ledger["charged_attempts"] - step,
        "calibration_attempts": ledger["calibration_attempts"], "calibration_flops": ledger["calibration_flops"],
        "consumed_training_flops": ledger["training_flops"], "consumed_total_flops": total,
        "unused_flops": budget - ledger["training_flops"], "budget_flops": budget, "budget_scope": SCOPE,
        "selected_step": best_step, "best_validation_hard_dice": best_score,
        "seconds_this_invocation": time.perf_counter() - started,
        **{key: base.digest(args.output / name) for key, name in ARTIFACTS.items()},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("unet", "swinunetr", "nnunet"), required=True)
    parser.add_argument("--objective", choices=ARMS, required=True)
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
        parser.error("Invalid smoke limit or thread count")
    try:
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(args)
    except BaseException:
        failure = traceback.format_exc()
        # Emit to both streams before attempting a possibly quota-blocked write.
        print(failure, file=sys.stdout, flush=True)
        print(failure, file=sys.stderr, flush=True)
        try:
            base.atomic_json(args.output / "failure.json", {"traceback": traceback.format_exc()})
        except OSError:
            pass
        raise


if __name__ == "__main__":
    main()
