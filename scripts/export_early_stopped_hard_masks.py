#!/usr/bin/env python3
"""Export compact hard masks from a selected SwinUNETR checkpoint."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from monai.data import Dataset as MonaiDataset
from monai.data import DataLoader
from monai.networks.nets import SwinUNETR

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "baselines/swin_unetr"))
from swin_unetr import _build_monai_dataset_from_pkl, _load_pkl_dataframe, _load_splits_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pkl", type=Path, required=True)
    parser.add_argument("--splits-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    df = _load_pkl_dataframe(args.pkl)
    dataset = _build_monai_dataset_from_pkl(df, "MSD", 3, spatial_size=(64, 64, 64))
    selected = set(_load_splits_json(args.splits_json)[args.fold][args.split])
    items = [item for item in dataset.data if item["case_name"] in selected]
    if {item["case_name"] for item in items} != selected:
        raise RuntimeError("Selected cases do not match the dataset")
    loader = DataLoader(MonaiDataset(items, transform=dataset.transform), batch_size=args.batch_size)
    device = torch.device("cuda")
    model = SwinUNETR(in_channels=1, out_channels=3, use_checkpoint=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = 0
    with torch.no_grad():
        for batch in loader:
            hard = model(batch["image"].to(device)).argmax(dim=1).cpu().numpy().astype(np.uint8)
            for prediction in hard:
                name = items[index]["case_name"]
                np.savez_compressed(args.output_dir / f"{name}.npz", prediction=prediction)
                index += 1
                if index % 10 == 0:
                    print(f"Saved {index}/{len(items)} cases", flush=True)
    with (args.output_dir / "manifest.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("case_name", "checkpoint", "fold", "split"))
        writer.writerows((item["case_name"], str(args.checkpoint), args.fold, args.split) for item in items)
    print(f"Saved {index} compact masks to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
