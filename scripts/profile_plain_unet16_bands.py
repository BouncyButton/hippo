"""Measure the width-16 pilot's bands update and training-only calibration cost."""
import hashlib
import json
from pathlib import Path

import profile_budget_bands as profiler
from profile_intermediate_baselines import intermediate_supplement
from baselines.simple_segmentation import build_candidate
from compute_constraint_objectives import scheduled_costs


def main():
    audit = profiler.audit
    if audit.monai.__version__ != "1.6.0" or not audit.torch.__version__.startswith("2.2.2"):
        raise ValueError("Use the established profiling runtime")
    audit.torch.set_num_threads(4)
    audit.reference.supplementary_flops = intermediate_supplement
    # Rebind the helper's U-Net branch without changing the historical width-48 profiler.
    profiler.build_unet48 = lambda: build_candidate("basic_unet16")
    root = Path(__file__).resolve().parents[1]
    directory = root / "experiments/plain_unet16_lr3e4_20261005/dice"
    protocol = json.loads((directory / "protocol.json").read_text())
    rows = []
    for arm, calibration in [("dice", False), ("dice_bands", False), ("dice_bands", True)]:
        for geometry in ("ellipsoid", "box"):
            row = profiler.profile("unet", directory, arm, geometry, calibration)
            rows.append(row)
            print(json.dumps({k: row[k] for k in ["arm", "geometry", "calibration", "flops"]}), flush=True)
    costs = {}
    for key, arm, calibration in [("baseline", "dice", False), ("bands", "dice_bands", False),
                                  ("calibration_per_case", "dice_bands", True)]:
        values = {row["flops"] for row in rows if row["arm"] == arm and row["calibration"] == calibration}
        if len(values) != 1:
            raise ValueError("Geometry changed the declared arithmetic count")
        costs[key] = values.pop()
    if (costs["baseline"] != protocol["models"]["unet"]["flops_per_update"]
            or any(row["parameters"] != 1438819 for row in rows)):
        raise ValueError("Architecture differs from the completed pilot")
    costs["calibration_total"] = 10 * costs["calibration_per_case"] + 1024
    schedule = scheduled_costs(protocol["budget_flops"], costs["baseline"], costs["bands"],
                               costs["calibration_total"], 50)
    paths = [Path(__file__), Path(profiler.__file__), root / "baselines/simple_segmentation.py",
             root / "scripts/profile_intermediate_baselines.py", Path(audit.__file__),
             audit.PROFILE_ROOT / "profile_reference.py", audit.PROFILE_ROOT / "profile_matched_models.py",
             root / "scripts/train_compute_matched_baselines.py", root / "scripts/compute_constraint_objectives.py",
             root / "scripts/train_compute_matched_constraints.py",
             root / "thesis/new_constraints/bands/outer_boundary.py",
             root / "thesis/new_constraints/constraint_result.py"]
    result = {"status": "profiled_only", "candidate": "basic_unet16", "baseline_job_id": 684625,
              "budget_flops": protocol["budget_flops"], "costs": costs, "schedule": schedule,
              "profiles": rows, "parameters": 1438819,
              "baseline_protocol_sha256": hashlib.sha256((directory / "protocol.json").read_bytes()).hexdigest(),
              "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              "runtime": {"torch": audit.torch.__version__, "monai": audit.monai.__version__},
              "calibration_policy": "After 50 baseline updates, calibrate detached-logit gradients on all ten training volumes, charged within budget; median ratio 0.1, p95 cap 0.5; ramp bands over 50 constrained updates.",
              "purpose": "Prepare the authorized conditional fallback; no clinical training performed."}
    target = root / "reports/plain_unet16_bands_20261005"
    target.mkdir(parents=True, exist_ok=True)
    (target / "profiles.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"costs": costs, "schedule": schedule}), flush=True)


if __name__ == "__main__":
    main()
