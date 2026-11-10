"""Verify prepared labels, split isolation and a native 3D forward/backward pass."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from geofwi3d_data import load_case, sha256, validate_splits, write_json


def check(root: Path, model_smoke: bool = False) -> dict:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    validate_splits(manifest["splits"], manifest["cases"])
    for name, case in manifest["cases"].items():
        for field in ["image", "label"]:
            if sha256(root / case[field]) != case[field + "_sha256"]:
                raise RuntimeError(f"Changed prepared data: {name}/{field}")
        image, label = load_case(root, case)
        if label.sum() != case["salt_voxels"] or image.min() < -1 or image.max() > 1:
            raise RuntimeError(f"Prepared arrays differ from the recorded label/normalization audit: {name}")
    result = {"state": "passed", "manifest_sha256": sha256(manifest_path),
              "cases": len(manifest["cases"]), "validation_cases": len(manifest["splits"]["validation"]),
              "training_subsets": {key: len(value) for key, value in manifest["splits"]["training_subsets"].items()},
              "all_image_label_hashes_verified": True, "simulation_seed_isolation_verified": True,
              "whole_native_volume_shape": [96, 96, 96], "model_smoke": None}
    if model_smoke:
        import torch
        from monai.losses import DiceCELoss
        from train_geofwi3d_swinunetr import make_model, seed_everything
        torch.set_num_threads(2)
        seed_everything(0)
        name = manifest["splits"]["training_subsets"]["5pct"][0]
        image, label = load_case(root, manifest["cases"][name])
        model = make_model(checkpointing=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        x = torch.from_numpy(image[None, None])
        y = torch.from_numpy(label[None, None].astype(np.int64))
        before = next(model.parameters()).detach().clone()
        logits = model(x)
        assert tuple(logits.shape) == (1, 2, 96, 96, 96)
        loss = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)(logits, y)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        assert not torch.equal(before, next(model.parameters()).detach())
        result["model_smoke"] = {"device": "cpu", "native_input": list(x.shape),
                                 "native_output": list(logits.shape), "loss": float(loss.detach()),
                                 "optimizer_updated_weights": True, "training_case_id": name}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-smoke", action="store_true")
    args = parser.parse_args()
    result = check(args.root.resolve(), args.model_smoke)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
