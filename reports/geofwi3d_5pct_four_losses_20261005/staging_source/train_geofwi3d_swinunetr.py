"""Train native 3D SwinUNETR with fixed-update low-data budgets and whole-volume Dice."""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
import traceback

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "geofwi3d-matplotlib"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import torch.nn.functional as F

from geofwi3d_data import FRACTIONS, SHAPE, load_case, sha256, validate_splits, write_json
from geofwi3d_losses import METHODS, signed_distance_map, signed_distance_loss, signed_face_edge_loss


def make_model(checkpointing: bool = True) -> torch.nn.Module:
    from monai.networks.nets import SwinUNETR
    options = dict(in_channels=1, out_channels=2, feature_size=24, spatial_dims=3,
                   use_checkpoint=checkpointing, drop_rate=0.0, attn_drop_rate=0.0)
    if "img_size" in inspect.signature(SwinUNETR).parameters:
        options["img_size"] = SHAPE
    return SwinUNETR(**options)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sample_index(position: int, size: int, seed: int) -> int:
    """Visit every selected volume before repeating; order is resumable by index."""
    cycle, offset = divmod(position, size)
    rng = np.random.default_rng(np.random.SeedSequence([seed, cycle, 7301]))
    return int(rng.permutation(size)[offset])


def augmented(image: np.ndarray, label: np.ndarray, seed: int, position: int, enabled: bool):
    if enabled:
        rng = np.random.default_rng(np.random.SeedSequence([seed, position, 9109]))
        # Preserve geological depth: spatial augmentation is in the XY plane only.
        for axis in [0, 1]:
            if rng.random() < .5:
                image, label = np.flip(image, axis), np.flip(label, axis)
        turns = int(rng.integers(4))
        image, label = np.rot90(image, turns, (0, 1)), np.rot90(label, turns, (0, 1))
    return np.ascontiguousarray(image), np.ascontiguousarray(label)


class CaseStore:
    def __init__(self, root: Path, cases: dict, cache: bool = True):
        self.root, self.cases, self.cache = root, cases, cache
        self.arrays = {}
        self.distance_fields = {}

    def get(self, name: str):
        if name not in self.arrays:
            arrays = load_case(self.root, self.cases[name])
            if self.cache:
                self.arrays[name] = arrays
            return arrays
        return self.arrays[name]


    def get_sdf(self, name: str, clip: float) -> np.ndarray:
        key = (name, clip)
        if key not in self.distance_fields:
            self.distance_fields[key] = signed_distance_map(self.get(name)[1], clip)
        return self.distance_fields[key]


def band_support(labels: torch.Tensor, steps: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Six-connected supports excluding artificial observation-window faces."""
    foreground = labels.bool()
    kernel = torch.zeros((1, 1, 3, 3, 3), device=labels.device)
    kernel[0, 0, 1, 1, :] = 1
    kernel[0, 0, 1, :, 1] = 1
    kernel[0, 0, :, 1, 1] = 1
    eroded, dilated = foreground, foreground
    with torch.no_grad():
        for _ in range(steps):
            eroded = F.conv3d(eroded.float(), kernel, padding=1) == 7
            dilated = F.conv3d(dilated.float(), kernel, padding=1) > 0
        valid = torch.zeros_like(foreground)
        valid[:, :, steps:-steps, steps:-steps, steps:-steps] = True
        inner, outer = foreground & ~eroded & valid, ~foreground & dilated & valid
    return inner, outer


def band_loss(logits: torch.Tensor, labels: torch.Tensor, steps: int = 2,
              supports=None) -> torch.Tensor:
    """Balanced inner/outer BCE, excluding artificial observation-window faces."""
    foreground = labels.bool()
    inner, outer = band_support(labels, steps) if supports is None else supports
    log_odds = logits[:, 1:2].float() - logits[:, 0:1].float()
    voxel_loss = F.binary_cross_entropy_with_logits(log_odds, foreground.float(), reduction="none")
    # Normalize each side independently, then average only defined sides.
    values = []
    for case in range(logits.shape[0]):
        sides = [voxel_loss[case][support[case]].mean() for support in [inner, outer] if support[case].any()]
        values.append(torch.stack(sides).mean() if sides else logits[case].sum() * 0)
    return torch.stack(values).mean()


def dice_counts(prediction: np.ndarray, target: np.ndarray) -> dict:
    prediction, target = prediction.astype(bool), target.astype(bool)
    intersection = int(np.count_nonzero(prediction & target))
    predicted, truth = int(prediction.sum()), int(target.sum())
    return {"dice": 2 * intersection / (predicted + truth) if predicted + truth else 1.0,
            "intersection": intersection, "predicted_voxels": predicted, "target_voxels": truth}


@torch.inference_mode()
def evaluate(model, store: CaseStore, ids: list[str], device: torch.device, amp: bool) -> tuple[dict, list]:
    model.eval()
    rows = []
    for name in ids:
        image, label = store.get(name)
        inputs = torch.from_numpy(image[None, None]).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(inputs)
        if tuple(logits.shape) != (1, 2, *SHAPE):
            raise RuntimeError("Model output is not on the complete native voxel grid")
        if not torch.isfinite(logits).all():
            raise RuntimeError("Nonfinite validation logits")
        prediction = logits.argmax(1)[0].cpu().numpy()
        rows.append({"case_id": name, **dice_counts(prediction, label)})
    if len(rows) != len(ids) or {row["case_id"] for row in rows} != set(ids):
        raise RuntimeError("Validation did not cover every case exactly once")
    scores = np.array([row["dice"] for row in rows])
    return {"mean_dice": float(scores.mean()), "std_dice": float(scores.std()),
            "min_dice": float(scores.min()), "max_dice": float(scores.max()),
            "validation_volumes": len(ids), "scope": "mean foreground hard Dice over whole native 96³ volumes"}, rows


def atomic_checkpoint(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def write_rows(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def checked_protocol(protocol: dict) -> None:
    positive = ["max_updates", "validate_every", "early_stopping_patience", "batch_size", "warmup_updates"]
    if any(int(protocol[key]) <= 0 for key in positive):
        raise ValueError("Update, batch, validation and patience settings must be positive")
    if not 0 <= protocol["min_updates"] <= protocol["max_updates"]:
        raise ValueError("Invalid minimum training duration")
    if protocol["warmup_updates"] >= protocol["max_updates"]:
        raise ValueError("Warmup must finish before the update cap")
    scheduler_updates = int(protocol.get("scheduler_updates", protocol["max_updates"]))
    if scheduler_updates < protocol["max_updates"]:
        raise ValueError("Learning-rate schedule must cover the entire training budget")
    if protocol["learning_rate"] <= 0 or protocol["weight_decay"] < 0:
        raise ValueError("Invalid optimizer parameters")
    if protocol["band_weight"] < 0 or protocol["min_delta"] < 0 or protocol["band_warmup_updates"] < 1:
        raise ValueError("Invalid auxiliary/stopping settings")
    if protocol.get("base_loss", "foreground Dice + cross entropy") != "foreground Dice + cross entropy":
        raise ValueError("Unknown supervised base loss")
    for kind in ("sdf", "edge"):
        if protocol.get(kind + "_weight", .1) < 0 or protocol.get(kind + "_warmup_updates", 500) < 1:
            raise ValueError("Invalid additional auxiliary settings")
    if not math.isfinite(protocol.get("sdf_clip_voxels", 8.0)) or protocol.get("sdf_clip_voxels", 8.0) <= 0:
        raise ValueError("Invalid SDF distance clip")


def learning_rate_factor(update: int, protocol: dict) -> float:
    """Allow a short budget to reproduce the prefix of a longer LR schedule."""
    warmup = protocol["warmup_updates"]
    horizon = protocol.get("scheduler_updates", protocol["max_updates"])
    if update < warmup:
        return (update + 1) / warmup
    return .01 + .99 * .5 * (1 + math.cos(math.pi * (update-warmup) / (horizon-warmup)))


def train_run(root: Path, manifest: dict, protocol: dict, output: Path, fraction: str,
              method: str, device: torch.device) -> dict:
    from monai.losses import DiceCELoss
    output.mkdir(parents=True, exist_ok=True)
    completed = output / "completion.json"
    invocation_started = time.monotonic()
    seed = int(protocol["model_seed"])
    ids = manifest["splits"]["training_subsets"][fraction]
    validation = manifest["splits"]["validation"]
    identity = {"manifest_sha256": sha256(root / "manifest.json"), "protocol": protocol,
                "fraction": fraction, "method": method, "training_case_ids": ids,
                "validation_case_ids": validation,
                "source_sha256": {Path(__file__).name: sha256(Path(__file__)),
                                  "geofwi3d_data.py": sha256(Path(__file__).with_name("geofwi3d_data.py")),
                                  "geofwi3d_losses.py": sha256(Path(__file__).with_name("geofwi3d_losses.py"))}}
    config = output / "run_config.json"
    if config.exists() and json.loads(config.read_text()) != identity:
        raise RuntimeError("Run inputs or source changed; choose a new output directory")
    if completed.exists():
        record = json.loads(completed.read_text())
        for name, expected in record["artifact_sha256"].items():
            if sha256(output / name) != expected:
                raise RuntimeError(f"Completed run artifact changed: {name}")
        return record
    write_json(config, identity)
    seed_everything(seed)
    model = make_model(protocol["activation_checkpointing"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["learning_rate"], weight_decay=protocol["weight_decay"])
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
    terms = set(method.split("_")) - {"baseline"}
    store = CaseStore(root, manifest["cases"], protocol["cache_volumes"])
    best, best_update, reference, bad_checks, update = -1.0, 0, -1.0, 0, 0
    elapsed, history = 0.0, []
    latest = output / "latest.pt"
    if latest.exists():
        checkpoint = torch.load(latest, map_location=device, weights_only=False)
        if checkpoint["identity"] != identity:
            raise RuntimeError("Checkpoint belongs to another dataset or training policy")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        update, best, best_update = checkpoint["update"], checkpoint["best"], checkpoint["best_update"]
        reference, bad_checks, history = checkpoint["reference"], checkpoint["bad_checks"], checkpoint["history"]
        elapsed = checkpoint["elapsed_seconds"]
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([item.cpu() for item in checkpoint["cuda_rng"]])
        del checkpoint
    interval_started = time.monotonic()
    interval_loss, interval_samples, updates_since_check = 0.0, 0, 0
    interval_components = {name: 0.0 for name in ("supervised", "bands", "sdf", "edge")}
    should_stop = bool(update >= protocol["min_updates"] and bad_checks >= protocol["early_stopping_patience"])
    while update < protocol["max_updates"] and not should_stop:
        model.train()
        images, labels, distance_fields = [], [], []
        for batch_index in range(protocol["batch_size"]):
            position = update * protocol["batch_size"] + batch_index
            name = ids[sample_index(position, len(ids), seed)]
            image, label = augmented(*store.get(name), seed, position, protocol["augmentation"])
            images.append(image)
            labels.append(label)
            if "sdf" in terms:
                field = store.get_sdf(name, protocol.get("sdf_clip_voxels", 8.0))
                transformed, _ = augmented(field, field, seed, position, protocol["augmentation"])
                distance_fields.append(transformed)
        x = torch.from_numpy(np.stack(images)[:, None]).to(device)
        y = torch.from_numpy(np.stack(labels)[:, None].astype(np.int64)).to(device)
        target_sdf = torch.from_numpy(np.stack(distance_fields)[:, None]).to(device) if distance_fields else None
        factor = learning_rate_factor(update, protocol)
        for group in optimizer.param_groups:
            group["lr"] = protocol["learning_rate"] * factor
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=protocol["amp"] and device.type == "cuda"):
            logits = model(x)
            supervised = loss_function(logits, y)
            loss = supervised
            components = {"supervised": supervised}
            supports = band_support(y) if terms & {"bands", "edge"} else None
            if "bands" in terms:
                weight = protocol["band_weight"] * min((update+1) / protocol["band_warmup_updates"], 1)
                components["bands"] = band_loss(logits, y, supports=supports)
                loss = loss + weight * components["bands"]
            if "sdf" in terms:
                weight = protocol.get("sdf_weight", .1) * min((update+1) / protocol.get("sdf_warmup_updates", 500), 1)
                components["sdf"] = signed_distance_loss(logits, y, target_sdf)
                loss = loss + weight * components["sdf"]
            if "edge" in terms:
                weight = protocol.get("edge_weight", .1) * min((update+1) / protocol.get("edge_warmup_updates", 500), 1)
                components["edge"] = signed_face_edge_loss(logits, y, supports[0] | supports[1])
                loss = loss + weight * components["edge"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite training loss at update {update+1}")
        loss.backward()  # bfloat16 has the float32 exponent range and needs no loss scaling
        torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0, error_if_nonfinite=True)
        optimizer.step()
        update += 1
        updates_since_check += 1
        interval_loss += float(loss.detach())
        for name, value in components.items():
            interval_components[name] += float(value.detach())
        interval_samples += len(images)
        if update % 10 == 0:
            write_json(output / "status.json", {"state": "training", "update": update,
                       "max_updates": protocol["max_updates"], "best_validation_dice": best if best >= 0 else None,
                       "training_volumes": len(ids), "validation_volumes": len(validation),
                       "active_fraction": fraction, "method": method})
        if update % protocol["validate_every"] != 0 and update < protocol["max_updates"]:
            continue
        if device.type == "cuda":
            torch.cuda.synchronize()
        training_seconds = time.monotonic() - interval_started
        check_started = time.monotonic()
        metrics, rows = evaluate(model, store, validation, device, protocol["amp"] and device.type == "cuda")
        validation_seconds = time.monotonic() - check_started
        elapsed += training_seconds + validation_seconds
        score = metrics["mean_dice"]
        if score > best:
            best, best_update = score, update
            atomic_checkpoint(output / "best.pt", {"model": model.state_dict(), "identity": identity,
                              "update": update, "validation": metrics})
            write_json(output / "validation_best.json", {"selected_update": update, **metrics})
            write_rows(output / "validation_per_image.csv", rows)
        if score > reference + protocol["min_delta"]:
            reference, bad_checks = score, 0
        else:
            bad_checks += 1
        should_stop = update >= protocol["min_updates"] and bad_checks >= protocol["early_stopping_patience"]
        row = {"update": update, "mean_train_loss": interval_loss / updates_since_check,
               **{f"mean_{name}_loss": value / updates_since_check for name, value in interval_components.items()},
               "validation_mean_dice": score, "best_validation_dice": best,
               "training_seconds": training_seconds, "validation_seconds": validation_seconds,
               "interval_updates": updates_since_check, "interval_samples": interval_samples,
               "learning_rate": optimizer.param_groups[0]["lr"], "bad_validation_checks": bad_checks,
               "elapsed_seconds": elapsed}
        history.append(row)
        write_rows(output / "history.csv", history)
        atomic_checkpoint(latest, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "identity": identity, "update": update,
                          "best": best, "best_update": best_update, "reference": reference,
                          "bad_checks": bad_checks, "history": history, "elapsed_seconds": elapsed,
                          "torch_rng": torch.get_rng_state(),
                          "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []})
        write_json(output / "status.json", {"state": "validated", "update": update,
                   "validation": metrics, "best_validation_dice": best, "best_update": best_update,
                   "early_stopping_triggered": should_stop})
        print(json.dumps({"fraction": fraction, "method": method, **row}), flush=True)
        interval_loss, interval_samples, updates_since_check = 0.0, 0, 0
        interval_components = {name: 0.0 for name in interval_components}
        interval_started = time.monotonic()
    selected = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(selected["model"])
    metrics, rows = evaluate(model, store, validation, device, protocol["amp"] and device.type == "cuda")
    if not math.isclose(metrics["mean_dice"], best, abs_tol=1e-6):
        raise RuntimeError("Saved best checkpoint does not reproduce its selection Dice")
    write_rows(output / "validation_per_image.csv", rows)
    write_json(output / "validation_best.json", {"selected_update": best_update, **metrics})
    artifacts = ["run_config.json", "best.pt", "latest.pt", "history.csv", "validation_best.json", "validation_per_image.csv"]
    result = {"state": "complete", "fraction": fraction, "method": method, "seed": seed,
              "training_volumes": len(ids), "validation_volumes": len(validation),
              "selected_update": best_update, "stopped_update": update,
              "stop_reason": "early_stopping" if should_stop else "update_cap",
              "validation": metrics, "elapsed_training_and_selection_seconds": elapsed,
              "invocation_wall_seconds": time.monotonic() - invocation_started,
              "scope": "selected validation results; no test evaluation",
              "artifact_sha256": {name: sha256(output / name) for name in artifacts}}
    write_json(completed, result)
    write_json(output / "status.json", result)
    del model, optimizer, store, selected
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--fractions", nargs="+", choices=list(FRACTIONS), default=list(FRACTIONS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=["baseline"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        protocol = json.loads(args.protocol.read_text())
        checked_protocol(protocol)
        manifest = json.loads((args.root / "manifest.json").read_text())
        validate_splits(manifest["splits"], manifest["cases"])
        torch.set_num_threads(protocol["cpu_threads"])
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device.type == "cuda" and protocol["amp"] and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This fixed-precision protocol requires a CUDA GPU with bfloat16 support")
        for fraction in args.fractions:
            for method in args.methods:
                write_json(args.output / "status.json", {"state": "running", "active_fraction": fraction, "active_method": method})
                result = train_run(args.root.resolve(), manifest, protocol,
                                   args.output / f"{fraction}_{method}", fraction, method, device)
                results.append(result)
                write_json(args.output / "summary.json", {"state": "running", "runs": results})
        summary = {"state": "complete", "runs": results, "success_metric": "validation mean foreground Dice"}
        write_json(args.output / "summary.json", summary)
        write_json(args.output / "status.json", summary)
    except Exception as error:
        write_json(args.output / "failure.json", {"state": "failed", "error": str(error),
                   "traceback": traceback.format_exc(), "completed_runs": results})
        raise


if __name__ == "__main__":
    main()
