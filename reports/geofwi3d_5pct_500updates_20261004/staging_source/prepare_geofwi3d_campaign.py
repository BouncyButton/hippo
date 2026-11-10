"""Freeze selected low-data runs without submitting a cluster job."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys

from geofwi3d_data import sha256, write_json

SOURCES = ["download_geofwi3d.py", "geofwi3d_data.py", "prepare_geofwi3d.py",
           "train_geofwi3d_swinunetr.py", "check_geofwi3d_pipeline.py",
           "prepare_geofwi3d_campaign.py", "submit_geofwi3d.py", "run_geofwi3d_swinunetr.sbatch"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--max-updates", type=int, default=5000)
    parser.add_argument("--scheduler-updates", type=int, help="LR horizon; defaults to the update cap")
    parser.add_argument("--min-updates", type=int, default=2000)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--methods", nargs="+", choices=["baseline", "bands"], default=["baseline"])
    parser.add_argument("--fractions", nargs="+", choices=["5pct", "20pct", "100pct"], default=["5pct", "20pct", "100pct"])
    parser.add_argument("--budget-selection-note", help="Provenance for choosing this training budget")
    args = parser.parse_args()
    root, campaign = args.root.resolve(), args.campaign.resolve()
    if (campaign / "source_manifest.json").exists() or (campaign / "submission.json").exists():
        raise RuntimeError("Campaign already frozen; choose a new directory")
    preflight = json.loads(args.preflight.read_text())
    if preflight.get("state") != "passed" or preflight.get("manifest_sha256") != sha256(root / "manifest.json"):
        raise RuntimeError("Dataset preflight does not match the current manifest")
    if not (preflight.get("model_smoke") or {}).get("optimizer_updated_weights"):
        raise RuntimeError("A native 96³ model smoke check is required before freezing")
    campaign.mkdir(parents=True, exist_ok=True)
    source = campaign / "source"
    source.mkdir(exist_ok=True)
    for name in SOURCES:
        shutil.copy2(Path(__file__).parent / name, source / name)
    protocol = {"version": 1, "model": "3D SwinUNETR", "feature_size": 24, "model_seed": args.model_seed,
                "from_scratch": True, "native_shape": [96, 96, 96],
                "success_metric": "validation mean foreground hard Dice", "test_evaluation": False,
                "max_updates": args.max_updates, "min_updates": args.min_updates,
                "scheduler_updates": args.scheduler_updates if args.scheduler_updates is not None else args.max_updates,
                "validate_every": args.validate_every, "early_stopping_patience": args.patience,
                "min_delta": .0005, "learning_rate": 1e-4, "weight_decay": 1e-5,
                "warmup_updates": 100, "batch_size": 1, "amp": True, "amp_dtype": "bfloat16",
                "activation_checkpointing": True, "cache_volumes": True, "cpu_threads": 4,
                "augmentation": True, "augmentation_policy": "XY flips and 90-degree rotations; preserve Z depth",
                "base_loss": "foreground Dice + cross entropy", "methods": args.methods,
                "band_weight": .1, "band_warmup_updates": 500,
                "band_policy": "optional two-voxel side-balanced BCE; omit observation-cube faces",
                "budget_policy": "same maximum optimizer updates and validation opportunities for every fraction",
                "checkpoint_selection": "exact highest mean validation Dice; early stopping uses min_delta separately"}
    if args.budget_selection_note:
        protocol["budget_selection_note"] = args.budget_selection_note
    from train_geofwi3d_swinunetr import checked_protocol
    checked_protocol(protocol)
    write_json(campaign / "protocol.json", protocol)
    write_json(campaign / "campaign.json", {"dataset_root": str(root), "fractions": args.fractions,
               "methods": args.methods, "python": sys.executable, "submission_requested": False})
    shutil.copy2(root / "manifest.json", campaign / "manifest.json")
    if args.preflight.resolve() != (campaign / "preflight.json").resolve():
        shutil.copy2(args.preflight, campaign / "preflight.json")
    dependencies = {name: version(name) for name in ["torch", "monai", "numpy", "scipy", "nibabel"]}
    write_json(campaign / "dependencies.json", dependencies)
    files = [str(path.relative_to(campaign)) for path in source.iterdir() if path.is_file()]
    files += ["protocol.json", "campaign.json", "manifest.json", "preflight.json", "dependencies.json"]
    write_json(campaign / "source_manifest.json", {"sha256": {name: sha256(campaign / name) for name in sorted(files)}})
    print(json.dumps({"state": "prepared", "campaign": str(campaign), "protocol": protocol}, indent=2))


if __name__ == "__main__":
    main()
