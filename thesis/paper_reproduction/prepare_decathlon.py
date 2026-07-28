#!/usr/bin/env python3
"""Download and validate the official MONAI Task04_Hippocampus dataset once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from monai.apps import DecathlonDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--task", default="Task04_Hippocampus")
    parser.add_argument("--expected-samples", type=int, default=0)
    args = parser.parse_args()

    args.data_root.mkdir(parents=True, exist_ok=True)
    dataset = DecathlonDataset(root_dir=str(args.data_root), task=args.task, section="training", download=True)
    if args.expected_samples and len(dataset) != args.expected_samples:
        raise RuntimeError(f"Expected {args.expected_samples} samples, downloaded {len(dataset)}.")
    print(json.dumps({"task": args.task, "data_root": str(args.data_root), "training_samples": len(dataset)}))


if __name__ == "__main__":
    main()
