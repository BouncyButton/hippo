#!/usr/bin/env python3
"""Profile the four matched-compute objectives for U-Net or SwinUNETR."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import profile_unet_constraint_budget as audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("unet", "swinunetr"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if audit.monai.__version__ != "1.6.0" or not audit.torch.__version__.startswith("2.2.2"):
        raise RuntimeError("Use the original profiling runtime")
    audit.torch.set_num_threads(4)
    audit.reference.supplementary_flops = audit.constraint_supplement
    original = json.loads((audit.ROOT / "experiments/compute_matched_baselines_20261003/protocol.json").read_text())
    official = json.loads((audit.ROOT / "experiments/compute_budget_reference.json").read_text())
    baseline_cost = original["models"][args.model]["flops_per_update"]
    budget = official["budget_flops"]
    constructor = audit.matched.build_network
    audit.matched.build_network = lambda _name, recipe: constructor(args.model, recipe)
    args.output.mkdir(parents=True, exist_ok=True)
    profiles = []
    for arm in audit.ARMS:
        for geometry in ("ellipsoid", "box"):
            row = audit.profile_arm(arm, geometry)
            row["model"] = args.model
            profiles.append(row)
            print(json.dumps({k: row[k] for k in ("model", "arm", "geometry", "flops_per_update")}), flush=True)
    assert all(p["flops_per_update"] == baseline_cost for p in profiles if p["arm"] == "dice")
    schedules = {}
    for arm in audit.ARMS:
        costs = {p["flops_per_update"] for p in profiles if p["arm"] == arm}
        assert len(costs) == 1
        schedules[arm] = audit.schedule(costs.pop(), budget)
        schedules[arm]["overhead_percent"] = 100 * (schedules[arm]["flops_per_update"] / baseline_cost - 1)
    sources = [Path(__file__), Path(audit.__file__), audit.PROFILE_ROOT / "profile_reference.py",
               audit.PROFILE_ROOT / "profile_matched_models.py",
               audit.ROOT / "thesis/new_constraints/bands/outer_boundary.py",
               audit.ROOT / "thesis/new_constraints/edge_consistency.py",
               audit.ROOT / "thesis/new_constraints/signed_distance.py",
               audit.ROOT / "scripts/compute_constraint_objectives.py",
               audit.ROOT / "thesis/new_constraints/constraint_result.py"]
    result = {
        "model": args.model, "budget_flops": budget,
        "input_shape": [1, 1, 64, 64, 64], "training_patients": 10,
        "runtime": {"torch": audit.torch.__version__, "monai": audit.monai.__version__, "device": "cpu"},
        "parameters": profiles[0]["parameters"],
        "source_sha256": {str(p.relative_to(audit.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "schedules": schedules, "profiles": profiles,
    }
    (args.output / (args.model + "_profiles.json")).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(schedules, indent=2), flush=True)


if __name__ == "__main__":
    main()
