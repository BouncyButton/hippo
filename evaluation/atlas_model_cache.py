"""Export native frozen logits, checking old CPU predictions and label geometry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.atlas_registration import CACHE, REPORT, native, save_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=("train", "development"), required=True)
    args = parser.parse_args()
    from baselines.swin_unetr.swin_unetr import _build_monai_dataset_from_pkl, _load_pkl_dataframe
    from thesis.new_constraints.train_swinunetr_constraints import build_swinunetr
    torch.set_num_threads(4)
    cohort = json.loads((REPORT / "cohort.json").read_text())
    names = cohort["calibration"] + cohort["assessment"] if args.subset == "train" else cohort["development"]
    directory = ROOT / "experiments/augmentation_family_b_20260907/checkpoints"
    canonical = json.loads((directory / "CANONICAL_BASELINES.json").read_text())
    checkpoint = directory / canonical["unaugmented"]["path"]
    if sha(checkpoint) != canonical["unaugmented"]["sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    pkl = ROOT / "datasets/Dataset101_MSD/msd_hippocampus_full.pkl"
    local_manifest_path = ROOT / "experiments/context_increment_20260923/cache/unaugmented/manifest.json"
    local_manifest = json.loads(local_manifest_path.read_text())
    if sha(pkl) != local_manifest["pkl_sha256"] or sha(checkpoint) != local_manifest["checkpoint_sha256"]:
        raise ValueError("Local CPU descriptor/checkpoint hash mismatch")
    dataset = _build_monai_dataset_from_pkl(_load_pkl_dataframe(pkl), "MSD", 3,
        spatial_size=(64, 64, 64), do_resize=False)
    indices = {entry["case_name"]: i for i, entry in enumerate(dataset.data)}
    model = build_swinunetr((64, 64, 64), 3, torch.device("cpu"), activation_checkpointing=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    provenance = dict(checkpoint_sha256=sha(checkpoint), pkl_sha256=sha(pkl),
        canonical_cluster_pkl_sha256=canonical["shared"]["pkl_sha256"],
        verified_local_manifest_sha256=sha(local_manifest_path),
        exporter_sha256=sha(__file__), torch=torch.__version__, device="cpu", threads=4,
        model_source_sha256=sha(ROOT / "baselines/swin_unetr/swin_unetr.py"),
        builder_source_sha256=sha(ROOT / "thesis/new_constraints/train_swinunetr_constraints.py"))
    folder = CACHE / "network"
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != provenance:
        raise ValueError("Network cache provenance mismatch")
    save_json(manifest, provenance)
    evidence = {}
    for index, name in enumerate(names):
        destination = folder / f"{name}.npz"
        if destination.exists():
            continue
        item = dataset[indices[name]]
        truth = item["label"][0].numpy().astype(np.uint8)
        with torch.inference_mode():
            logits = model(item["image"][None].float())[0].numpy()
        prediction = logits.argmax(0).astype(np.uint8)
        old_path = ROOT / "experiments/graph_partition_20260923/cache" / f"{name}.npz"
        if not old_path.exists():
            old_path = ROOT / "experiments/uncal_fold_early_stopping_20260921/voxel_audit/baseline_seed0/error_maps" / f"{name}.npz"
        with np.load(old_path, allow_pickle=False) as old:
            key = "truth" if "truth" in old else "ground_truth"
            np.testing.assert_array_equal(truth, old[key])
            np.testing.assert_array_equal(prediction, old["prediction"])
        label = native(name, labels=True)
        if any(a > b for a, b in zip(label.shape, truth.shape)):
            raise ValueError("Native data exceeds model crop")
        crop = tuple(slice((b - a) // 2, (b - a) // 2 + a) for a, b in zip(label.shape, truth.shape))
        np.testing.assert_array_equal(truth[crop], label)
        # Padding predictions, if any, must be counted rather than silently lost.
        native_support = np.zeros(truth.shape, bool)
        native_support[crop] = True
        outside_predictions = int(((prediction > 0) & ~native_support).sum())
        if outside_predictions:
            raise ValueError(f"Nonzero foreground in padded area: {name}")
        np.savez_compressed(destination, logits=logits[(slice(None),) + crop], truth=label.astype(np.uint8),
                            old_cache_sha256=sha(old_path))
        evidence[name] = dict(old_cache_sha256=sha(old_path), native_truth_match=True,
                              old_hard_prediction_match=True, outside_foreground=outside_predictions)
        print(f"Frozen logits {index+1}/{len(names)}: {name}", flush=True)
    old_evidence = REPORT / f"network_verification_{args.subset}.json"
    if old_evidence.exists():
        evidence = {**json.loads(old_evidence.read_text()), **evidence}
    save_json(old_evidence, evidence)


if __name__ == "__main__":
    main()
