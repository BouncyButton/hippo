"""Verify profiles and export step-capped schedules without launching training."""
import csv
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
REFERENCE = json.loads((OUT / "profile_cpu.json").read_text())
REFERENCE_BUDGET = REFERENCE["estimated_total_B_ref"]
ROUNDED_BUDGET = 116259 * 10**9
BUDGET = ROUNDED_BUDGET  # Enforce the user's stated maximum literally.
checks = {}
rows = []
for model, recipe in [("unet", "matched"), ("nnunet", "native"), ("swinunetr", "matched")]:
    meta = json.loads((OUT / f"matched_{model}_{recipe}_meta.json").read_text())
    if model == "swinunetr":
        checks[model] = meta["flops_per_update"] == REFERENCE["estimated_total_flops_per_step"]
        prof = meta
    else:
        cpu = json.loads((OUT / f"matched_{model}_{recipe}_cpu.json").read_text())
        checks[model] = all(meta[k] == cpu[k] for k in ["parameters", "core_flops_per_stage", "supplementary_estimate_per_stage", "flops_per_update"])
        prof = cpu
    assert checks[model], f"Meta/CPU mismatch for {model}"
    f = prof["flops_per_update"]
    steps = BUDGET // f
    full, extra = divmod(steps, 10)
    assert steps * f <= BUDGET < (steps + 1) * f
    row = {"model": model, "recipe": recipe, "batch_size": 1, "input_shape": [64, 64, 64],
           "training_patients": 10, "classes": 3, "parameters": prof["parameters"],
           "flops_per_update": f, "gflops_per_update": f / 1e9,
           "max_steps": steps, "complete_patient_pass_epochs": full,
           "additional_updates": extra, "patient_pass_equivalents": steps / 10,
           "loop_epochs_if_10_steps_per_epoch": full + bool(extra),
           "stop_in_last_epoch_after_updates": extra if extra else 10,
           "consumed_gflops": steps * f / 1e9,
           "unused_gflops": (BUDGET - steps * f) / 1e9,
           "max_steps_if_literal_116259_gflops": ROUNDED_BUDGET // f,
           "max_steps_if_unrounded_original_reference": REFERENCE_BUDGET // f,
           "nnunet_native_250_update_blocks": steps // 250 if model == "nnunet" else None,
           "nnunet_remaining_updates_after_250_blocks": steps % 250 if model == "nnunet" else None,
           "architecture_file": "nnunet_matched64_architecture_native.json" if model == "nnunet" else None}
    rows.append(row)
payload = {"budget_flops": BUDGET, "budget_gflops": BUDGET / 1e9,
           "original_reference_flops": REFERENCE_BUDGET,
           "literal_rounded_cap_flops": ROUNDED_BUDGET,
           "profiles_verified": checks, "schedules": rows,
           "epoch_definition": "Ten patient visits at batch one. A true dataset pass requires sampling each patient once; nnU-Net random patches only yield exposure equivalents.",
           "budget_enforcement": "Check global_step < max_steps before every training step; stop mid-epoch if needed. Constraints require a new profiled cost.",
           "normalization": "No gradient accumulation; all trainable; 64^3 input; no activation checkpointing. Estimates exclude validation, preprocessing and CUDA GradScaler."}
(OUT / "training_schedules.json").write_text(json.dumps(payload, indent=2) + "\n")
with (OUT / "training_schedules.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({"budget_gflops": payload["budget_gflops"], "verified": checks,
                  "schedules": [{k:r[k] for k in ["model", "gflops_per_update", "max_steps", "complete_patient_pass_epochs", "additional_updates", "max_steps_if_literal_116259_gflops"]} for r in rows]}, indent=2))
