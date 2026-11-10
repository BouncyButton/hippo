"""Select a reference from complete Dice-only U-Net-48 schedules."""
import argparse
import json
import math
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys

from run_budget_band_pilot import digest, verify_completed, write_json


def next_updates(plan, history):
    if any(plan["target_min"] <= r["final_dice"] <= plan["target_max"] for r in history):
        return None
    if len(history) >= plan["max_pilot_runs"]:
        raise RuntimeError("Pilot limit reached without reaching the final-Dice target band")
    if not history:
        return plan["initial_updates"]
    for r in history:
        if not math.isfinite(r["final_dice"]) or not 0 <= r["final_dice"] <= 1:
            raise ValueError("Invalid pilot score")
    lower = [r["updates"] for r in history if r["final_dice"] < plan["target_min"]]
    upper = [r["updates"] for r in history if r["final_dice"] > plan["target_max"]]
    if lower and upper:
        lo, hi = max(lower), min(upper)
        if hi <= lo or hi - lo < 2:
            raise RuntimeError("Observed pilot curve cannot be bracketed further; inspect outcomes")
        candidate = (lo + hi) // 2
    elif upper:
        candidate = min(upper) // 2
        if candidate < plan["min_updates"]:
            raise RuntimeError("Minimum candidate update count reached")
    else:
        candidate = 2 * max(lower)
        if candidate > plan["max_updates"]:
            raise RuntimeError("Maximum candidate update count reached")
    if any(r["updates"] == candidate for r in history):
        raise RuntimeError("Repeated candidate update count")
    return candidate


def allowance(budget, cost):
    updates = budget // cost
    return {"flops_per_update": cost, "gflops_per_update": cost / 1e9,
            "max_updates": updates, "patient_pass_epochs": updates / 10,
            "consumed_flops": updates * cost, "unused_flops": budget - updates * cost}


def verify_imported_source(source, expected, prior):
    """Permit only explicitly hash-bound generated protocols from a failed staging."""
    exceptions = prior.get("generated_protocol_exceptions", {})
    for name, actual_sha in exceptions.items():
        assert re.fullmatch(r"experiments/unet48_budget_candidates/candidate[0-9]{2}/protocol\.json", name), name
        assert name in expected and expected[name] != actual_sha, name
        path = source / name
        assert digest(path) == actual_sha, name
        assert json.loads(path.read_text())["selection_plan_sha256"] == prior["plan_sha256"]
    for name, sha in expected.items():
        assert digest(source / name) == exceptions.get(name, sha), name


def validate_origin_manifest(expected):
    """Generated candidates must never be part of the immutable source manifest."""
    if any(name.startswith("experiments/unet48_budget_candidates/") for name in expected):
        raise ValueError("Frozen source includes generated candidate protocols; stage a clean source snapshot")


def imported_history(plan, template):
    """Verify earlier candidates against their own immutable plan and runtime."""
    prior = plan.get("imported_search")
    if prior is None:
        return []
    root = Path(prior["root"])
    source = root / "source"
    old_plan_path = source / prior["plan_relative_path"]
    assert digest(old_plan_path) == prior["plan_sha256"]
    old_plan = json.loads(old_plan_path.read_text())
    assert old_plan["baseline_costs"] == plan["baseline_costs"]
    assert (old_plan["fold"], old_plan["seed"], old_plan["lr_policy"]) == (plan["fold"], plan["seed"], plan["lr_policy"])
    assert json.loads((old_plan_path.parent / "template_protocol.json").read_text()) == template
    verify_imported_source(source, json.loads((root / "runtime_origin_manifest.json").read_text()), prior)
    history_path = root / "pilot_history.json"
    assert digest(history_path) == prior["history_sha256"]
    rows = json.loads(history_path.read_text())
    assert [row["completion_sha256"] for row in rows] == prior["completion_sha256"]
    for row in rows:
        output, directory = Path(row["output"]), Path(row["protocol_dir"])
        assert output.is_relative_to(root) and directory.is_relative_to(source)
        final, _ = verify_completed(output, directory, source, "dice")
        assert final == row["final_dice"] and digest(output / "completion.json") == row["completion_sha256"]
        assert json.loads((directory / "protocol.json").read_text())["selection_plan_sha256"] == prior["plan_sha256"]
    return [{**row, "imported_root": str(root), "imported_plan_sha256": prior["plan_sha256"]} for row in rows]


def run(args):
    root, source = args.root, args.root / "source"
    plan = json.loads(args.plan.read_text())
    template_path = args.plan.parent / "template_protocol.json"
    template = json.loads(template_path.read_text())
    assert digest(template_path) == plan["template_protocol_sha256"]
    linear_head = template.get("unet_output_head") == "linear"
    assert template["models"]["unet"]["parameters"] == (43234201 if linear_head else 43234202)
    if linear_head:
        assert template["unet_factory_sha256"] == digest(source / "baselines/residual_unet.py")
        assert template["unet_lr_policy"] == "absolute_warmup_constant"
    assert template["architecture_adapter_sha256"] == digest(source / "scripts/train_unet48_budget_pilot.py")
    assert digest(args.pkl) == template["pkl_sha256"]
    expected = json.loads((root / "runtime_origin_manifest.json").read_text())
    validate_origin_manifest(expected)

    def verify_source():
        for name, sha in expected.items():
            if digest(source / name) != sha:
                raise ValueError("Frozen source changed: " + name)

    verify_source()
    history_path = root / "pilot_history.json"
    earlier = imported_history(plan, template)
    history = json.loads(history_path.read_text()) if history_path.exists() else earlier
    assert history[:len(earlier)] == earlier
    for row in history[len(earlier):]:
        final, _ = verify_completed(Path(row["output"]), Path(row["protocol_dir"]), source, "dice")
        assert final == row["final_dice"] and digest(Path(row["output"]) / "completion.json") == row["completion_sha256"]
        assert json.loads((Path(row["protocol_dir"]) / "protocol.json").read_text())["selection_plan_sha256"] == digest(args.plan)
    environment = {**os.environ, "PYTHONPATH": str(source) + ":" + str(source / "scripts")}
    while True:
        updates = next_updates(plan, history)
        if updates is None:
            break
        index = len(history)
        cost = plan["baseline_costs"]["unet_width48"]
        budget = updates * cost
        directory = source / "experiments/unet48_budget_candidates" / f"candidate{index:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("splits_10patients.json", "split_provenance.json", "nnunet_architecture.json"):
            shutil.copyfile(args.plan.parent / name, directory / name)
        protocol = {**template, "budget_flops": budget, "name": f"unet48_budget_candidate{index:02d}",
                    "selection_plan_sha256": digest(args.plan)}
        protocol["models"] = {"unet": {**template["models"]["unet"], "objectives": {
            "dice": {"max_steps": updates, "prefix_updates": 0, "constraint_updates": updates,
                     "calibration_flops": 0, "consumed_flops": budget, "unused_flops": 0,
                     "patient_pass_epochs": updates / 10, "flops_per_update": cost,
                     "baseline_flops_per_update": cost}}}}
        write_json(directory / "protocol.json", protocol)
        output = root / "pilots" / f"candidate{index:02d}" / "run"
        print(json.dumps({"pilot_start": index, "updates": updates, "epochs": updates / 10,
                          "budget_gflops": budget / 1e9}), flush=True)
        verify_source()
        subprocess.run([sys.executable, str(source / "scripts/train_unet48_budget_pilot.py"),
                        "--model", "unet", "--objective", "dice", "--fold", "0", "--seed", "0",
                        "--pkl", str(args.pkl), "--output", str(output), "--protocol-dir", str(directory),
                        "--threads", str(args.threads), "--resume"], env=environment, check=True)
        final, done = verify_completed(output, directory, source, "dice")
        entry = {"candidate": index, "updates": updates, "budget_flops": budget,
                 "final_dice": final, "best_dice": done["best_validation_hard_dice"],
                 "output": str(output), "protocol_dir": str(directory),
                 "completion_sha256": digest(output / "completion.json")}
        history.append(entry)
        write_json(history_path, history)
        print(json.dumps({"pilot_complete": entry}), flush=True)
    chosen = next(r for r in history if plan["target_min"] <= r["final_dice"] <= plan["target_max"])
    schedules = {name: allowance(chosen["budget_flops"], cost) for name, cost in plan["baseline_costs"].items()}
    selection = {**chosen, "status": "selected", "target_band": [plan["target_min"], plan["target_max"]],
                 "selection_plan_sha256": digest(args.plan), "baseline_allowances": schedules,
                 "pilot_total_flops": sum(r["budget_flops"] for r in history),
                 "additional_search_flops": sum(r["budget_flops"] for r in history if r != chosen),
                 "selection_metric": plan["selection_metric"],
                 "notes": ["Only final validation Dice of from-scratch width-48 Dice baselines enters budget selection.",
                           "Baseline allowances for the other models are calculated, not measured performance.",
                           "Exploratory fold 0, seed 0; freeze budget for subsequent comparisons."]}
    write_json(root / "budget_selection.json", selection)
    reference = {"status": "selected_reference", "adopted_on": plan["client_date"],
                 "budget_flops": chosen["budget_flops"], "budget_gflops": chosen["budget_flops"] / 1e9,
                 "anchor_model": "unet_width48", "anchor_model_kwargs": template["models"]["unet"]["model_kwargs"],
                 "anchor_parameters": template["models"]["unet"]["parameters"],
                 "unet_output_head": template.get("unet_output_head", "legacy_normalized"),
                 "anchor_flops_per_update": plan["baseline_costs"]["unet_width48"],
                 "baseline_max_updates": chosen["updates"], "baseline_patient_pass_epochs": chosen["updates"] / 10,
                 "baseline_final_dice": chosen["final_dice"], "baseline_best_dice": chosen["best_dice"],
                 "training_patients": 10, "validation_patients": 52, "fold": 0, "seed": 0,
                 "input_shape": [1, 1, 64, 64, 64], "selection_plan_sha256": digest(args.plan),
                 "selected_completion_sha256": chosen["completion_sha256"],
                 "selected_protocol_sha256": digest(Path(chosen["protocol_dir"]) / "protocol.json"),
                 "parent_reference_sha256": plan["previous_reference_sha256"],
                 "accounting": template["accounting"], "learning_rate_policy": template["adamw_schedule"],
                 "budget_selection": selection["notes"], "baseline_allowances": schedules}
    write_json(root / "compute_budget_reference.json", reference)
    write_json(root / "source_manifest.json", {p.relative_to(source).as_posix(): digest(p)
                                               for p in source.rglob("*") if p.is_file()})
    print(json.dumps({"BUDGET_SELECTED": selection}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--pkl", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    run(parser.parse_args())
