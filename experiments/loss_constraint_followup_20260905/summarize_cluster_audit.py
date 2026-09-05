"""Read scalar outputs of a completed GPU audit; no model/data loading."""
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
completion = json.loads((root / "completion.json").read_text())
export = json.loads((root / "export_manifest.json").read_text())
summary = {"completion": completion, "reproduction": export["reproduction"],
           "job_id": export["job_id"], "gpu": export["gpu"],
           "model_sha256": export["model_sha256"],
           "prediction_summary": export["prediction_summary"],
           "pickle_geometry": {k: {name: value for name, value in group.items() if name != "cases"}
                               for k, group in export["pickle_geometry"].items()}}
base = export["prediction_summary"]["baseline"]
comparisons = {}
for variant in ("teacher_full", "teacher_two", "tta_13"):
    treatment = export["prediction_summary"][variant]
    deltas = [c["metrics"][variant]["macro_dice"] - c["metrics"]["baseline"]["macro_dice"] for c in export["cases"]]
    comparisons[variant] = {
        "delta_case_mean_macro": treatment["per_case_mean_macro_dice"] - base["per_case_mean_macro_dice"],
        "delta_case_mean_union": treatment["per_case_mean_union_dice"] - base["per_case_mean_union_dice"],
        "better_cases": sum(d > 1e-12 for d in deltas), "worse_cases": sum(d < -1e-12 for d in deltas),
        "median_case_delta": statistics.median(deltas),
        **{f"delta_{k}": treatment[k] - base[k] for k in ("fp", "fn", "swaps", "total_errors")},
    }
summary["comparisons"] = comparisons
aps = [c["ap_conditional"] for c in export["cases"]]
n = sum(c["gt_foreground_voxels"] for c in aps)
wrong = sum(c["conditional_ap_wrong_voxels"] for c in aps)
summary["ap_confidence"] = {
    "foreground_voxels": n,
    "saturated_fraction": sum(c["gt_foreground_voxels"] * c["conditional_ap_saturated_fraction"] for c in aps) / n,
    "wrong_voxels": wrong,
    "wrong_saturated_fraction": sum(c["conditional_ap_wrong_saturated_voxels"] for c in aps) / wrong,
}
reports = {}
for filename in ("teacher_full_dice.json", "teacher_full_dice_ce.json", "teacher_two_dice.json", "ap_dice_ce.json"):
    data = json.loads((root / filename).read_text())
    valid = [c for c in data["cases"] if c["status"] == "valid"]
    reference = data["baseline_valid_cases"]["per_case_mean_macro_dice"]
    report = {"runtime": data["runtime"], "valid": data["valid_cases"], "skipped": data["skipped_cases"],
              "baseline_case_mean_macro": reference,
              "diagnostic_weight": data["diagnostic_auxiliary_weight"],
              "gradient_medians": {k: statistics.median(c["gradients"][k] for c in valid if c["gradients"][k] is not None)
                  for k in ("supervised_rms", "auxiliary_rms", "cosine", "auxiliary_over_supervised_rms")},
              "repairs": {direction: {step: {"delta_case_mean_macro": m["per_case_mean_macro_dice"] - reference,
                    "fp": m["fp"], "fn": m["fn"], "swaps": m["swaps"]} for step, m in steps.items()}
                  for direction, steps in data["repair_summary_same_valid_cases"].items()}}
    if filename == "teacher_full_dice.json":
        diagnostics = [c["calibration_diagnostics"] for c in valid]
        calibration = {}
        for stratum in ("whole", "gt_foreground", "gt_boundary", "incorrect"):
            prefix = "calibration/" + stratum + "/"
            count = sum(c[prefix + "voxel_count"] for c in diagnostics)
            calibration[stratum] = {"voxel_count": count}
            for metric in ("nll", "brier", "saturation_fraction", "confidence", "accuracy"):
                calibration[stratum][metric] = sum(c[prefix + metric] * c[prefix + "voxel_count"] for c in diagnostics if c[prefix + metric] is not None) / count
        summary["calibration_pooled"] = calibration
    reports[filename] = report
summary["audit_reports"] = reports
print(json.dumps(summary, indent=2, allow_nan=False))
