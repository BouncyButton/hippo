#!/usr/bin/env python3
"""Quantify learning-curve saturation and spatial errors of the saved U-Net runs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import time

import numpy as np
from scipy import ndimage


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def file_hash(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def plateau_analysis(curves):
    parsed = []
    for run in curves:
        rows = list(csv.DictReader(io.StringIO(run["csv"])))
        steps = np.array([int(r["step"]) for r in rows])
        scores = np.array([float(r["macro_hard_dice"]) for r in rows])
        if parsed:
            assert np.array_equal(steps, parsed[0]["steps"])
        parsed.append({"fold": run["fold"], "seed": run["seed"], "steps": steps,
                       "scores": scores, "loss": np.array([float(r["mean_train_loss"]) for r in rows])})
    scores = np.stack([r["scores"] for r in parsed])
    best = np.maximum.accumulate(scores, axis=1)
    final_best = scores.max(axis=1)
    remaining = final_best[:, None] - best
    milestones = []
    for i, step in enumerate(parsed[0]["steps"]):
        milestones.append({"step": int(step), "epoch": float(step / 10),
                           "mean_raw_dice": float(scores[:, i].mean()),
                           "sd_raw_dice": float(scores[:, i].std(ddof=1)),
                           "mean_best_so_far_dice": float(best[:, i].mean()),
                           "mean_remaining_gain": float(remaining[:, i].mean()),
                           "max_remaining_gain": float(remaining[:, i].max()),
                           "mean_train_loss": float(np.mean([r["loss"][i] for r in parsed]))})
    thresholds = {}
    for threshold in (0.01, 0.005, 0.002, 0.001):
        index = next(i for i, row in enumerate(milestones) if row["mean_remaining_gain"] <= threshold)
        per_run = []
        for r, gains in zip(parsed, remaining):
            j = next(i for i, gain in enumerate(gains) if gain <= threshold)
            per_run.append({"fold": r["fold"], "seed": r["seed"],
                            "epoch": float(r["steps"][j] / 10), "step": int(r["steps"][j])})
        thresholds[str(threshold)] = {"aggregate_first_epoch": milestones[index]["epoch"],
                                     "aggregate_first_step": milestones[index]["step"],
                                     "mean_remaining_gain": milestones[index]["mean_remaining_gain"],
                                     "per_run": per_run,
                                     "per_run_median_epoch": float(np.median([r["epoch"] for r in per_run]))}
    return {"definition": "Retrospective first validation milestone whose mean remaining best-checkpoint improvement is no larger than the declared absolute Dice threshold; not an early-stopping rule or a formal change-point test.",
            "runs": len(parsed), "validation_spacing_epochs": [60.6, 60.7],
            "milestones": milestones, "thresholds": thresholds,
            "best_checkpoints": [{"fold": r["fold"], "seed": r["seed"],
                                  "step": int(r["steps"][r["scores"].argmax()]),
                                  "epoch": float(r["steps"][r["scores"].argmax()] / 10),
                                  "dice": float(r["scores"].max())} for r in parsed]}


def geometry(label):
    """GT surface voxels and Euclidean distances in the unchanged input grid."""
    structure = ndimage.generate_binary_structure(3, 1)
    fg = label > 0
    outer = fg & ~ndimage.binary_erosion(fg, structure=structure, border_value=0)
    if not outer.any():
        raise ValueError("Ground-truth foreground is empty")
    anterior, posterior = label == 1, label == 2
    interface = (anterior & ndimage.binary_dilation(posterior, structure=structure)) | (
        posterior & ndimage.binary_dilation(anterior, structure=structure))
    return {"foreground": fg, "outer_surface": outer,
            "outer_distance": ndimage.distance_transform_edt(~outer),
            "interface_surface": interface,
            "interface_distance": ndimage.distance_transform_edt(~interface) if interface.any() else np.full(label.shape, np.inf)}


def dice(truth, predicted):
    denominator = int(truth.sum() + predicted.sum())
    return 2 * int((truth & predicted).sum()) / denominator if denominator else 1.0


def spatial_counts(label, prediction, geo=None):
    if label.shape != prediction.shape or label.ndim != 3:
        raise ValueError("Expected equal 3-D arrays")
    if not set(np.unique(label)).issubset({0, 1, 2}) or not set(np.unique(prediction)).issubset({0, 1, 2}):
        raise ValueError("Only labels 0,1,2 are supported")
    geo = geometry(label) if geo is None else geo
    gt, proposed = geo["foreground"], prediction > 0
    fp, fn = ~gt & proposed, gt & ~proposed
    swap = gt & proposed & (label != prediction)
    error = label != prediction
    assert np.array_equal(error, fp | fn | swap)
    outer_near = geo["outer_distance"] <= 2
    interface_near = geo["interface_distance"] <= 2
    counts = {"total_errors": int(error.sum()), "foreground_fp": int(fp.sum()),
              "foreground_fn": int(fn.sum()), "anterior_posterior_swap": int(swap.sum()),
              "gt_foreground": int(gt.sum()), "pred_foreground": int(proposed.sum()),
              "gt_inner_shell_2_population": int((gt & outer_near).sum()),
              "gt_deep_foreground_population": int((gt & ~outer_near).sum()),
              "gt_inner_shell_2_errors": int((gt & outer_near & error).sum()),
              "gt_deep_foreground_errors": int((gt & ~outer_near & error).sum()),
              "outer_only_errors": int((error & outer_near & ~interface_near).sum()),
              "interface_only_errors": int((error & ~outer_near & interface_near).sum()),
              "outer_and_interface_errors": int((error & outer_near & interface_near).sum()),
              "neither_band_errors": int((error & ~outer_near & ~interface_near).sum())}
    for truth in range(3):
        for pred in range(3):
            counts[f"confusion_{truth}_{pred}"] = int(((label == truth) & (prediction == pred)).sum())
    for name, mask in (("all", error), ("fp", fp), ("fn", fn), ("swap", swap)):
        for threshold in (0, 1, 2, 3):
            counts[f"{name}_outer_le_{threshold}"] = int((mask & (geo["outer_distance"] <= threshold)).sum())
    for threshold in (0, 1, 2, 3):
        counts[f"swap_interface_le_{threshold}"] = int((swap & (geo["interface_distance"] <= threshold)).sum())
    pred_surface = proposed & ~ndimage.binary_erosion(proposed, structure=ndimage.generate_binary_structure(3, 1), border_value=0)
    if pred_surface.any():
        gt_to_pred = ndimage.distance_transform_edt(~pred_surface)[geo["outer_surface"]]
        pred_to_gt = geo["outer_distance"][pred_surface]
        symmetric = np.concatenate((gt_to_pred, pred_to_gt))
        surface = {"mean_symmetric_surface_distance_voxels": float(symmetric.mean()),
                   "surface_distance_95_voxels": float(np.percentile(symmetric, 95)),
                   "surface_match_fraction_1_voxel": float((symmetric <= 1).mean()),
                   "surface_match_fraction_2_voxels": float((symmetric <= 2).mean())}
    else:
        surface = {key: None for key in ("mean_symmetric_surface_distance_voxels", "surface_distance_95_voxels",
                                        "surface_match_fraction_1_voxel", "surface_match_fraction_2_voxels")}
    return {"counts": counts, "dice_anterior": dice(label == 1, prediction == 1),
            "dice_posterior": dice(label == 2, prediction == 2),
            "dice_foreground_union": dice(gt, proposed), **surface}


def aggregate_spatial(records):
    counts = {key: sum(r["counts"][key] for r in records) for key in records[0]["counts"]}
    total = counts["total_errors"]
    fraction = lambda a, b: a / b if b else None
    return {"case_run_evaluations": len(records), "counts": counts,
            "error_composition": {key: fraction(counts[key], total) for key in ("foreground_fp", "foreground_fn", "anterior_posterior_swap")},
            "distance_fractions": {key: fraction(value, total if key.startswith("all_") else
                                  counts["foreground_fp"] if key.startswith("fp_") else
                                  counts["foreground_fn"] if key.startswith("fn_") else counts["anterior_posterior_swap"])
                                   for key, value in counts.items() if "_le_" in key},
            "region_error_shares": {key: fraction(counts[key], total) for key in
                                   ("outer_only_errors", "interface_only_errors", "outer_and_interface_errors", "neither_band_errors")},
            "gt_inner_shell_population_share": fraction(counts["gt_inner_shell_2_population"], counts["gt_foreground"]),
            "inner_shell_foreground_error_rate": fraction(counts["gt_inner_shell_2_errors"], counts["gt_inner_shell_2_population"]),
            "deep_foreground_error_rate": fraction(counts["gt_deep_foreground_errors"], counts["gt_deep_foreground_population"]),
            "mean_macro_dice": float(np.mean([(r["dice_anterior"] + r["dice_posterior"]) / 2 for r in records])),
            "mean_foreground_union_dice": float(np.mean([r["dice_foreground_union"] for r in records])),
            "mean_symmetric_surface_distance_voxels": float(np.mean([r["mean_symmetric_surface_distance_voxels"] for r in records if r["mean_symmetric_surface_distance_voxels"] is not None])),
            "mean_case_surface_distance_95_voxels": float(np.mean([r["surface_distance_95_voxels"] for r in records if r["surface_distance_95_voxels"] is not None]))}


def plot_curves(curves, analysis, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs = np.array([r["epoch"] for r in analysis["milestones"]])
    raw = np.array([r["mean_raw_dice"] for r in analysis["milestones"]])
    best = np.array([r["mean_best_so_far_dice"] for r in analysis["milestones"]])
    sd = np.array([r["sd_raw_dice"] for r in analysis["milestones"]])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for run in curves:
        rows = list(csv.DictReader(io.StringIO(run["csv"])))
        axes[0].plot(epochs, [float(r["macro_hard_dice"]) for r in rows], color="0.75", lw=0.8)
    for ax in axes:
        ax.plot(epochs, raw, color="#006e87", marker="o", markersize=3, label="Mean validation Dice")
        ax.plot(epochs, best, color="#bc5b16", linestyle="--", label="Mean best so far")
        ax.fill_between(epochs, raw - sd, raw + sd, color="#006e87", alpha=0.1, label="Descriptive SD")
        for threshold, color in (("0.01", "#777777"), ("0.005", "#455a64")):
            epoch = analysis["thresholds"][threshold]["aggregate_first_epoch"]
            ax.axvline(epoch, color=color, linestyle=":", alpha=0.7)
        ax.set_xlabel("Patient-pass epochs (10 optimizer updates each)")
        ax.set_ylabel("Validation macro hard Dice")
        ax.grid(alpha=0.2)
    axes[0].set_title("All nine U-Net runs")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[1].set_xlim(110, 1230)
    axes[1].set_ylim(0.765, 0.825)
    axes[1].set_title("Late gains: same curves, expanded Dice scale")
    fig.savefig(output / "learning_curves.png", dpi=180)
    fig.savefig(output / "learning_curves.pdf")
    plt.close(fig)


def run_spatial(args):
    import torch
    spec = importlib.util.spec_from_file_location("frozen_unet_trainer", args.run_root / "source/scripts/train_compute_matched_baselines.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    torch.set_num_threads(args.threads)
    protocol_dir = args.run_root / "source/experiments/compute_matched_baselines_20261003"
    protocol = json.loads((protocol_dir / "protocol.json").read_text())
    assert file_hash(args.pkl) == protocol["pkl_sha256"]
    assert file_hash(protocol_dir / "splits_10patients.json") == protocol["splits_sha256"]
    splits = json.loads((protocol_dir / "splits_10patients.json").read_text())
    dataframe = trainer._load_pkl_dataframe(args.pkl)
    all_records, runs = [], []
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA bfloat16 support required")
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    for fold in range(3):
        dataset = trainer.PreparedPatients(dataframe, splits[fold]["val"])
        geometries = [geometry(item["label"][0].numpy()) for item in dataset]
        for seed in range(3):
            run = args.run_root / "unet" / f"fold{fold}" / f"seed{seed}" / "run"
            completion = json.loads((run / "completion.json").read_text())
            checkpoint_path = run / "checkpoint_best.pt"
            assert file_hash(checkpoint_path) == completion["checkpoint_best_sha256"]
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            config = json.loads((run / "config.json").read_text())
            assert checkpoint["configuration"] == config
            assert config["validation_cases"] == splits[fold]["val"]
            assert config["script_sha256"] == file_hash(Path(trainer.__file__))
            model = trainer.build_model("unet", {}).to(device).eval()
            model.load_state_dict(checkpoint["model"])
            records, examples = [], []
            with torch.no_grad():
                for index, (item, geo) in enumerate(zip(dataset, geometries)):
                    image = item["image"][None].to(device)
                    if args.device == "cuda":
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            logits = model(image)
                    else:
                        logits = model(image)
                    probabilities = logits.float().softmax(1)
                    assert torch.isfinite(probabilities).all()
                    prediction = probabilities.argmax(1)[0].cpu().numpy().astype(np.uint8)
                    label = item["label"][0].numpy().astype(np.uint8)
                    record = {"fold": fold, "seed": seed, "case_name": item["case_name"],
                              **spatial_counts(label, prediction, geo)}
                    records.append(record)
                    if fold == 0 and seed == 0:
                        examples.append((index, label, prediction))
            stats = aggregate_spatial(records)
            reference = completion["best_validation_hard_dice"]
            delta = stats["mean_macro_dice"] - reference
            if abs(delta) > 0.001:
                raise ValueError(f"Replayed Dice differs by {delta:.6f} for fold {fold} seed {seed}")
            runs.append({"fold": fold, "seed": seed, "selected_step": checkpoint["step"],
                         "reference_cuda_bfloat16_dice": reference, "replay_dice_delta": delta, **stats})
            all_records.extend(records)
            if examples:
                ordering = np.argsort([(r["dice_anterior"] + r["dice_posterior"]) / 2 for r in records])
                for name, position in (("worst", 0), ("median", len(ordering) // 2)):
                    idx = int(ordering[position])
                    index, label, pred = examples[idx]
                    np.savez_compressed(args.output / f"example_{name}.npz", label=label, prediction=pred)
                    write_json(args.output / f"example_{name}.json", {"selection": name + " macro Dice within fold0 seed0", "validation_index": index, "record": records[idx]})
            print(json.dumps({"fold": fold, "seed": seed, "dice": stats["mean_macro_dice"],
                              "dice_delta": delta, "errors": stats["counts"]["total_errors"]}), flush=True)
            del model, checkpoint, examples
    result = {"model": "unet", "scope": "All nine selected best checkpoints on all 52 validation cases per fold; no training or changed weights",
              "precision": "CUDA bfloat16 matching original validation" if args.device == "cuda" else "CPU FP32; replay difference verified within 0.001 absolute Dice for every run",
              "distance_convention": "Euclidean voxel-center distance to six-connected GT foreground surface or two-sided anterior/posterior interface; distances are input-grid voxels, not millimetres. Boundary voxels have distance zero. Near bands overlap.",
              "surface_convention": "Symmetric nearest-surface-voxel distance, with voxel counts rather than physical surface-area weighting",
              "aggregation": "Voxel counts pooled across 468 case/run evaluations; repeated seeds are not independent cases. Dice and surface distances averaged per case/run.",
              "source_sha256": file_hash(Path(__file__)), "training_source_sha256": file_hash(Path(trainer.__file__)),
              "dataset_sha256": protocol["pkl_sha256"], "torch": torch.__version__,
              "seconds": time.perf_counter() - started, "runs": runs, "pooled": aggregate_spatial(all_records)}
    write_json(args.output / "spatial_summary.json", result)
    flat = [{k: v for k, v in record.items() if k != "counts"} | record["counts"] for record in all_records]
    with (args.output / "spatial_cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    curves = commands.add_parser("curves")
    curves.add_argument("--input", type=Path, required=True)
    curves.add_argument("--output", type=Path, required=True)
    spatial = commands.add_parser("spatial")
    spatial.add_argument("--run-root", type=Path, required=True)
    spatial.add_argument("--pkl", type=Path, required=True)
    spatial.add_argument("--output", type=Path, required=True)
    spatial.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    spatial.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.command == "curves":
        args.output.mkdir(parents=True, exist_ok=True)
        data = json.loads(args.input.read_text())
        result = plateau_analysis(data)
        write_json(args.output / "plateau_summary.json", result)
        plot_curves(data, result, args.output)
        print(json.dumps({key: {k: v for k, v in value.items() if k != "per_run"}
                          for key, value in result["thresholds"].items()}, indent=2))
    else:
        if args.threads < 1:
            parser.error("threads must be positive")
        run_spatial(args)


if __name__ == "__main__":
    main()
