"""GPU-only official checkpoint export and frozen-loss audit; never trains."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from monai.data import Dataset

from baselines.swin_unetr.swin_unetr import _build_monai_dataset_from_pkl, _load_pkl_dataframe
from .audit_followup import (
    _aggregate_metrics, build_parser, file_sha256, run_audit, segmentation_metrics,
)
from .equivariance import translate_3d
from .source_bootstrap import source_digest
from .teacher import DEFAULT_TEACHER_SHIFTS, TranslationTeacherKLLoss
from .train_swinunetr_constraints import build_swinunetr, _segmentation_dice_sums


def publish(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This audit requires CUDA; no CPU fallback.")
    torch.set_num_threads(4)
    device = torch.device("cuda")
    started = time.monotonic()
    run_dir, out = args.run_dir.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    initial_source = source_digest()
    config = json.loads((run_dir / "config.json").read_text())
    spec = config["run"]
    manifest = json.loads((run_dir / "completion_manifest.json").read_text())
    final = json.loads((run_dir / "final_metrics.json").read_text())
    if not (manifest["status"] == "complete" and manifest["epoch"] == 50
            and spec["constraint_set"] == "none" and spec["fold"] == 0
            and spec["seed"] == 0 and spec["batch_size"] == 1 and spec["amp"]
            and spec.get("supervised_loss", "dice") == "dice"
            and spec.get("ce_weight", 0) == 0):
        raise ValueError("Not the official Family-B epoch-50 Dice-only seed-0 control.")
    checkpoint = run_dir / "MSD_fold0/model.pt"
    expected_model = "8c66f93525145be4efce4eb3578ffab39d1ede8035f728684d0355a62298d7f9"
    if file_sha256(checkpoint) != expected_model or manifest["artifacts"]["MSD_fold0/model.pt"] != expected_model:
        raise ValueError("Official model hash mismatch.")
    if file_sha256(run_dir / "final_metrics.json") != manifest["artifacts"]["final_metrics.json"]:
        raise ValueError("Completed baseline metric artifact mismatch.")
    if manifest["run"] != spec:
        raise ValueError("Completion manifest run differs from configuration.")
    for key, digest_key in (("pkl", "pkl_sha256"), ("splits_json", "splits_json_sha256")):
        if file_sha256(Path(config[key])) != config[digest_key]:
            raise ValueError(f"Official {key} data hash changed.")
    split = json.loads(Path(config["splits_json"]).read_text())[0]
    dataset = _build_monai_dataset_from_pkl(
        _load_pkl_dataframe(config["pkl"]), "MSD", 3,
        spatial_size=tuple(spec["spatial_size"]), do_resize=spec["resize"],
    )
    by_name = {row["case_name"]: row for row in dataset.data}
    if len(by_name) != len(dataset.data):
        raise ValueError("Duplicate dataset cases.")
    val = Dataset([by_name[name] for name in sorted(split["val"])], dataset.transform)
    if len(val) != 52:
        raise ValueError("Official audit requires all 52 held-out cases.")
    geometry = {}
    for cohort in ("train", "val"):
        rows = []
        for name in split[cohort]:
            label = np.asarray(by_name[name]["label"])
            a, p = (label == 1).sum(axis=(0, 2)), (label == 2).sum(axis=(0, 2))
            mixed = (a > 0) & (p > 0)
            rows.append({"case_id": name, "mixed_slices": int(mixed.sum()),
                         "minority_voxels": int(np.minimum(a, p).sum())})
        geometry[cohort] = {"cases": rows, "nonplanar_cases": sum(r["mixed_slices"] > 0 for r in rows),
                            "minority_voxels": sum(r["minority_voxels"] for r in rows)}
    model = build_swinunetr(tuple(spec["spatial_size"]), 3, device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    model.requires_grad_(False).eval()
    teacher_loss = TranslationTeacherKLLoss()
    generator = torch.Generator().manual_seed(20260905)
    cache_dirs = {name: out / name for name in ("cache_full", "cache_two")}
    for directory in cache_dirs.values():
        directory.mkdir()
    cases = []
    hard_sum, soft_sum, dice_count = 0.0, 0.0, 0
    for index, item in enumerate(val):
        image = item["image"].as_tensor() if hasattr(item["image"], "as_tensor") else item["image"]
        label = item["label"].as_tensor() if hasattr(item["label"], "as_tensor") else item["label"]
        image, label = image[None].to(device), label[None].to(device)
        name = item["case_name"]
        with torch.no_grad(), torch.autocast("cuda", enabled=True):
            base = model(image)
            views = [model(translate_3d(image, shift)) for shift in DEFAULT_TEACHER_SHIFTS]
        full = teacher_loss.build_from_cached(views, DEFAULT_TEACHER_SHIFTS)
        selected = torch.randperm(len(views), generator=generator)[:2].tolist()
        pair_shifts = tuple(DEFAULT_TEACHER_SHIFTS[i] for i in selected)
        two = teacher_loss.build_from_cached([views[i] for i in selected], pair_shifts)
        # Identity plus the 12 valid nonzero views; zero-padded predictions are excluded.
        tta = (full.probabilities * full.view_counts + base.float().softmax(1)) / (full.view_counts + 1)
        soft, hard, n = _segmentation_dice_sums(base, label, 3)
        soft_sum += soft
        hard_sum += hard
        dice_count += n
        two_decoding = torch.where(two.valid_mask, two.probabilities, base.float().softmax(1))
        metrics = {"baseline": segmentation_metrics(base[0].float(), label[0, 0]),
                   "teacher_full": segmentation_metrics(full.probabilities[0], label[0, 0]),
                   "teacher_two": segmentation_metrics(two_decoding[0], label[0, 0]),
                   "tta_13": segmentation_metrics(tta[0], label[0, 0])}
        ap_p = base[:, 1:3].float().softmax(1)
        fg = label[:, 0] != 0
        conditional_wrong = ap_p.argmax(1) + 1 != label[:, 0]
        ap_conf = ap_p.max(1).values
        ap = {"gt_foreground_voxels": int(fg.sum()),
              "conditional_ap_saturated_fraction": float((ap_conf[fg] >= .99).float().mean()),
              "conditional_ap_wrong_voxels": int((conditional_wrong & fg).sum()),
              "conditional_ap_wrong_saturated_voxels": int((conditional_wrong & fg & (ap_conf >= .99)).sum())}
        arrays = dict(logits=base[0].cpu().numpy(), labels=label[0, 0].cpu().numpy())
        paths = {}
        for tag, chosen in (("cache_full", list(range(len(views)))), ("cache_two", selected)):
            path = cache_dirs[tag] / (name + ".npz")
            np.savez_compressed(path, **arrays,
                teacher_logits=torch.cat([views[i] for i in chosen]).cpu().numpy(),
                shifts=np.array([DEFAULT_TEACHER_SHIFTS[i] for i in chosen], dtype=np.int64))
            paths[tag] = {"path": str(path), "sha256": file_sha256(path)}
        cases.append({"case_id": name, "metrics": metrics, "ap_conditional": ap,
                      "two_view_shifts": pair_shifts, "cache": paths})
        print(json.dumps({"phase": "export", "case": index + 1, "of": 52,
                          "baseline": metrics["baseline"]["macro_dice"],
                          "elapsed_seconds": time.monotonic() - started}), flush=True)
    reproduced = hard_sum / dice_count
    expected = final["val_dice_hard"]
    reproduction = {"expected": expected, "actual": reproduced, "absolute_difference": abs(reproduced - expected),
                    "pass": abs(reproduced - expected) < 1e-5}
    exports = {"run_dir": str(run_dir), "model_sha256": expected_model,
               "config_sha256": file_sha256(run_dir / "config.json"), "official_source_sha256": spec["source_sha256"],
               "audit_source_sha256": initial_source, "job_id": os.getenv("SLURM_JOB_ID"),
               "gpu": torch.cuda.get_device_name(), "model_inference": "CUDA autocast float16",
               "diagnostic_arithmetic": "CUDA float32 logit leaves; float64 gradient summaries",
               "two_view_decoding": "retain baseline outside union of valid shifted-view support",
               "split_sha256": config["splits_json_sha256"], "pkl_sha256": config["pkl_sha256"],
               "reproduction": reproduction, "pickle_geometry": geometry, "cases": cases,
               "prediction_summary": {variant: _aggregate_metrics([c["metrics"][variant] for c in cases])
                                      for variant in ("baseline", "teacher_full", "teacher_two", "tta_13")}}
    publish(out / "export_manifest.json", exports)
    if not reproduction["pass"]:
        raise RuntimeError(f"Official baseline reproduction failed: {reproduction}")
    del model, base, views, full, two, tta
    gc.collect()
    torch.cuda.empty_cache()
    report_paths = []
    for filename, cache, constraint, loss in (
        ("teacher_full_dice.json", "cache_full", "teacher", "dice"),
        ("teacher_full_dice_ce.json", "cache_full", "teacher", "dice_ce"),
        ("teacher_two_dice.json", "cache_two", "teacher", "dice"),
        ("ap_dice_ce.json", "cache_full", "ap_cut", "dice_ce"),
    ):
        arguments = ["--input-dir", str(cache_dirs[cache]), "--output", str(out / filename),
                     "--purpose", "diagnostic", "--constraint", constraint,
                     "--supervised-loss", loss, "--ce-weight", "1", "--device", "cuda",
                     "--max-cases", "52", "--checkpoint", str(checkpoint)]
        if constraint == "ap_cut":
            arguments.extend(["--ap-axis", "1", "--ap-anterior-side", "high"])
        report = run_audit(build_parser().parse_args(arguments))
        report_paths.append({"path": str(out / filename), "sha256": file_sha256(out / filename),
                             "valid_cases": report["valid_cases"]})
        print(json.dumps({"phase": "audit_complete", "report": filename, "valid": report["valid_cases"],
                          "elapsed_seconds": time.monotonic() - started}), flush=True)
        del report
        gc.collect()
        torch.cuda.empty_cache()
    if source_digest() != initial_source or file_sha256(checkpoint) != expected_model:
        raise RuntimeError("Source/checkpoint changed during audit.")
    publish(out / "completion.json", {"status": "complete", "reports": report_paths,
        "export_manifest_sha256": file_sha256(out / "export_manifest.json"),
        "elapsed_seconds": time.monotonic() - started, "source_sha256": initial_source})


if __name__ == "__main__":
    main()
