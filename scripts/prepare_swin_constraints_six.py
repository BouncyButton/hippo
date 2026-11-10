"""Freeze six SwinUNETR schedules from the measured update and calibration costs."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compute_constraint_objectives import scheduled_costs

ARMS = ["dice", "dice_bands", "dice_sdf", "dice_bands_sdf", "dice_bands_edge", "dice_bands_edge_sdf"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parent = ROOT / "experiments/compute_matched_constraints_band_20261004"
    directory = ROOT / "experiments/compute_matched_swin_six_20261004"
    reports = ROOT / "reports/compute_matched_swin_six_20261004"
    updates_path = reports / "swinunetr_profiles.json"
    calibration_path = reports / "swinunetr_calibration_profiles.json"
    updates = json.loads(updates_path.read_text())
    calibration = json.loads(calibration_path.read_text())
    for profile in (updates, calibration):
        assert profile["model"] == "swinunetr"
        for name, sha in profile["source_sha256"].items():
            assert digest(ROOT / name) == sha, name
    p = json.loads((parent / "protocol.json").read_text())
    reference = json.loads((ROOT / "experiments/compute_budget_reference.json").read_text())
    assert p["budget_flops"] == reference["budget_flops"] == 12207124453303
    assert updates["budget_flops"] == p["budget_flops"]
    baseline_cost = updates["schedules"]["dice"]["flops_per_update"]
    assert baseline_cost == 155012136022 and updates["parameters"] == 15703029
    # Preserve the U-Net prefix's compute fraction, rounding to the closest
    # feasible SwinUNETR update. Use the same duration for the auxiliary ramp.
    prefix = max(1, round(50 * reference["anchor_flops_per_update"] / baseline_cost))
    assert prefix == 3
    objectives = {}
    for arm in ARMS:
        cost = updates["schedules"][arm]["flops_per_update"]
        schedule = scheduled_costs(p["budget_flops"], baseline_cost, cost, calibration["costs"][arm], prefix)
        schedule.update(flops_per_update=cost, baseline_flops_per_update=baseline_cost)
        objectives[arm] = schedule
    p.update(name="compute_matched_swin_six_fold0_seed0", constraint_warmup_updates=prefix,
             pilot_scope="Exactly six requested SwinUNETR objectives; fold 0, seed 0, from scratch.",
             calibration_policy="3 Dice-only prefix updates, followed by calibration on the ten unaugmented training patients. Same sequential gradient-ratio targets and caps as U-Net. All calibration charged within B_ref.",
             constraint_scale="After 3 Dice prefix updates, ramp auxiliary weight from 1/3 to 1 over 3 constrained updates; all ramp updates charged at their full objective cost.",
             models={"swinunetr": {"parameters": updates["parameters"], "objectives": objectives}},
             prefix_policy={"method": "Preserve U-Net prefix compute fraction, nearest integer update",
                            "unet_prefix_updates": 50, "unet_update_flops": reference["anchor_flops_per_update"],
                            "swin_prefix_updates": prefix,
                            "unet_prefix_budget_fraction": 50 * reference["anchor_flops_per_update"] / p["budget_flops"],
                            "swin_prefix_budget_fraction": prefix * baseline_cost / p["budget_flops"],
                            "ramp_updates": prefix},
             parent_unet_protocol_sha256=digest(parent / "protocol.json"))
    p["profile_evidence"].update(update_profiles_sha256=digest(updates_path),
                                 calibration_profiles_sha256=digest(calibration_path),
                                 update_profiles=str(updates_path.relative_to(ROOT)),
                                 calibration_profiles=str(calibration_path.relative_to(ROOT)))
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("splits_10patients.json", "nnunet_architecture.json", "split_provenance.json", "budget_selection.json"):
        shutil.copyfile(parent / name, directory / name)
    (directory / "protocol.json").write_text(json.dumps(p, indent=2) + "\n")
    rows = ["# Six SwinUNETR runs at the official compute reference", "",
            "Fold 0, seed 0; ten training patients, 52 validation patients; 64³, batch size one.",
            "Each run starts from scratch and has the same 12,207.124453303 GFLOP cap, including training-only calibration.",
            "AdamW: initial LR 1e-4, weight decay 1e-5; 5% linear warmup and cosine decay to 1e-6 over that arm's allotted updates. No early stopping.",
            "Constraints use a three-update Dice prefix and three-update ramp, approximately preserving the U-Net compute fraction. Full-volume classwise SDF and original pooled edge loss are unchanged.",
            "", "| Objective | Updates | Patient-pass epochs | Calibration GFLOPs | Consumed GFLOPs |", "|---|---:|---:|---:|---:|"]
    for arm, s in objectives.items():
        rows.append(f"| {arm} | {s['max_steps']} | {s['patient_pass_epochs']:.1f} | {s['calibration_flops']/1e9:.6f} | {s['consumed_flops']/1e9:.6f} |")
    (directory / "README.md").write_text("\n".join(rows) + "\n")
    print(json.dumps(objectives, indent=2))


if __name__ == "__main__":
    main()
