"""Select a larger budget using only baseline Dice, then run the frozen ablation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ARTIFACTS = {
    "config_sha256": "config.json", "checkpoint_best_sha256": "checkpoint_best.pt",
    "checkpoint_latest_sha256": "checkpoint_latest.pt", "metrics_sha256": "metrics.csv",
    "budget_ledger_sha256": "budget_ledger.json", "validation_best_sha256": "validation_best.json",
    "calibration_sha256": "calibration.json",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def schedules(template, budget):
    result = {}
    for arm, old in template["models"]["unet"]["objectives"].items():
        prefix, baseline, cost, calibration = (
            old["prefix_updates"], old["baseline_flops_per_update"],
            old["flops_per_update"], old["calibration_flops"])
        tail = (budget - prefix * baseline - calibration) // cost
        if tail < 1:
            raise ValueError("Budget cannot fit calibration and a training update")
        updates = prefix + tail
        consumed = prefix * baseline + calibration + tail * cost
        assert consumed <= budget < consumed + cost
        result[arm] = {**old, "max_steps": updates, "constraint_updates": tail,
                       "consumed_flops": consumed, "unused_flops": budget - consumed,
                       "patient_pass_epochs": updates / 10}
    return result


def next_budget(plan, history):
    """Bounded, baseline-only bracket search; no monotonicity claim is made."""
    low, high = plan["base_budget_flops"], None
    if not history:
        numerator, denominator = plan.get("first_multiplier", [3, 2])
        return low * numerator // denominator
    for row in history:
        score, budget = row["final_dice"], row["budget_flops"]
        if not 0 <= score <= 1:
            raise ValueError("Invalid baseline Dice")
        if plan["target_min"] <= score <= plan["target_max"]:
            return None
        if score < plan["target_min"]:
            low = max(low, budget)
        elif high is None or budget < high:
            high = budget
    if len(history) >= plan["max_pilot_runs"]:
        raise RuntimeError("Pilot limit reached without a baseline in the target band")
    if high is not None:
        if high <= low:
            raise RuntimeError("Observed budget/Dice curve is not bracketable; inspect the pilot")
        candidate = (low + high) // 2
    else:
        candidates = [plan["base_budget_flops"] * n // d
                      for n, d in plan["escalation_multipliers"]]
        candidate = next((b for b in candidates if b > low), None)
        if candidate is None:
            raise RuntimeError("Maximum pilot budget reached without entering the target band")
    if candidate <= plan["base_budget_flops"] or any(r["budget_flops"] == candidate for r in history):
        raise RuntimeError("Budget search did not advance")
    return candidate


def verify_completed(run, protocol_dir, source, arm):
    done = json.loads((run / "completion.json").read_text())
    config = json.loads((run / "config.json").read_text())
    protocol = json.loads((protocol_dir / "protocol.json").read_text())
    schedule = protocol["models"]["unet"]["objectives"][arm]
    if (done["status"] != "complete" or config["smoke_test"]
            or (done["model"], done["objective"], done["fold"], done["seed"]) != ("unet", arm, 0, 0)
            or config["schedule"] != schedule or config["budget_flops"] != protocol["budget_flops"]
            or config["protocol_sha256"] != digest(protocol_dir / "protocol.json")
            or config["dataset_sha256"] != protocol["pkl_sha256"]):
        raise ValueError("Completed run differs from its candidate protocol")
    for key, filename in ARTIFACTS.items():
        if done[key] != digest(run / filename):
            raise ValueError("Changed completed artifact: " + filename)
    for key, name in (("script_sha256", "scripts/train_compute_matched_constraints.py"),
                      ("objective_source_sha256", "scripts/compute_constraint_objectives.py"),
                      ("shared_source_sha256", "scripts/train_compute_matched_baselines.py"),
                      ("sdf_source_sha256", "thesis/new_constraints/signed_distance.py")):
        if config[key] != digest(source / name):
            raise ValueError("Training source differs")
    ledger = json.loads((run / "budget_ledger.json").read_text())
    if (done["updates"] != schedule["max_steps"] or done["discarded_attempts"] != 0
            or ledger["charged_attempts"] != done["updates"]
            or done["consumed_total_flops"] != schedule["consumed_flops"]
            or done["consumed_total_flops"] > protocol["budget_flops"]):
        raise ValueError("Pilot must finish its full schedule without discarded updates")
    with (run / "metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if int(rows[-1]["step"]) != schedule["max_steps"]:
        raise ValueError("Missing final validation")
    final = float(rows[-1]["macro_hard_dice"])
    if not 0 <= final <= 1:
        raise ValueError("Invalid final validation Dice")
    return final, done


def run(args):
    root, source = args.root, args.root / "source"
    plan = json.loads(args.plan.read_text())
    template = json.loads((args.plan.parent / "template_protocol.json").read_text())
    if (digest(args.plan.parent / "template_protocol.json") != plan["template_protocol_sha256"]
            or plan["baseline_costs"]["unet"] != template["models"]["unet"]["objectives"]["dice"]["flops_per_update"]
            or not plan["base_final_dice"] < plan["target_min"] <= plan["target_max"] <= 1):
        raise ValueError("Selection plan differs from its baseline evidence or loss costs")
    expected = json.loads((root / "runtime_origin_manifest.json").read_text())

    def verify_source():
        for relative, sha in expected.items():
            if digest(source / relative) != sha:
                raise ValueError("Frozen runtime changed: " + relative)

    verify_source()
    if digest(args.pkl) != template["pkl_sha256"]:
        raise ValueError("Dataset differs from the reference")
    environment = {**os.environ, "PYTHONPATH": str(source) + ":" + str(source / "scripts")}
    history = json.loads((root / "pilot_history.json").read_text()) if (root / "pilot_history.json").exists() else []
    for entry in history:
        output, protocol_dir = Path(entry["output"]), Path(entry["protocol_dir"])
        final, _ = verify_completed(output, protocol_dir, source, "dice")
        if (entry["completion_sha256"] != digest(output / "completion.json")
                or entry["final_dice"] != final
                or json.loads((protocol_dir / "protocol.json").read_text())["selection_plan_sha256"] != digest(args.plan)):
            raise ValueError("Recorded pilot or selection plan changed")
    chosen = None
    while chosen is None:
        budget = next_budget(plan, history)
        if budget is None:
            chosen = next(r for r in history if plan["target_min"] <= r["final_dice"] <= plan["target_max"])
            break
        index = len(history)
        protocol_dir = source / "experiments/budget_band_pilot" / f"candidate{index:02d}"
        protocol_dir.mkdir(parents=True, exist_ok=True)
        for name in ("splits_10patients.json", "split_provenance.json", "nnunet_architecture.json"):
            shutil.copy2(args.plan.parent / name, protocol_dir / name)
        definition = {"budget_flops": budget, "parent_budget_flops": plan["base_budget_flops"],
                      "selection_plan_sha256": digest(args.plan), "candidate_index": index,
                      "selection_metric": "Final validation macro hard Dice of Dice-only U-Net"}
        write_json(protocol_dir / "reference_definition.json", definition)
        protocol = {**template, "name": f"unet_budget_band_candidate{index:02d}", "budget_flops": budget,
                    "budget_selection": "Baseline-only bounded search; no constraint scores enter selection",
                    "selection_plan_sha256": digest(args.plan),
                    "reference_sha256": digest(protocol_dir / "reference_definition.json"),
                    "reference_artifact": "reference_definition.json"}
        protocol["models"] = {"unet": {**template["models"]["unet"], "objectives": schedules(template, budget)}}
        write_json(protocol_dir / "protocol.json", protocol)
        output = root / "pilots" / f"candidate{index:02d}" / "run"
        print(json.dumps({"pilot_start": index, "budget_gflops": budget / 1e9,
                          "updates": protocol["models"]["unet"]["objectives"]["dice"]["max_steps"]}), flush=True)
        verify_source()
        command = [sys.executable, str(source / "scripts/train_compute_matched_constraints.py"),
                   "--model", "unet", "--objective", "dice", "--fold", "0", "--seed", "0",
                   "--pkl", str(args.pkl), "--output", str(output), "--protocol-dir", str(protocol_dir),
                   "--threads", str(args.threads), "--resume"]
        subprocess.run(command, env=environment, check=True)
        final, done = verify_completed(output, protocol_dir, source, "dice")
        entry = {"candidate": index, "budget_flops": budget, "final_dice": final,
                 "best_dice": done["best_validation_hard_dice"], "updates": done["updates"],
                 "protocol_dir": str(protocol_dir), "output": str(output),
                 "completion_sha256": digest(output / "completion.json")}
        history.append(entry)
        write_json(root / "pilot_history.json", history)
        print(json.dumps({"pilot_complete": entry}), flush=True)

    protocol_dir = Path(chosen["protocol_dir"])
    protocol = json.loads((protocol_dir / "protocol.json").read_text())
    baseline_allowances = {model: {"flops_per_update": cost, "max_updates": chosen["budget_flops"] // cost,
                                  "patient_pass_epochs": (chosen["budget_flops"] // cost) / 10}
                           for model, cost in plan["baseline_costs"].items()}
    selection = {**chosen, "target_band": [plan["target_min"], plan["target_max"]],
                 "selection_plan_sha256": digest(args.plan), "baseline_allowances": baseline_allowances,
                 "unet_constraint_schedules": protocol["models"]["unet"]["objectives"],
                 "pilot_total_flops": sum(json.loads((Path(r["output"]) / "completion.json").read_text())["consumed_total_flops"] for r in history),
                 "additional_search_flops": sum(json.loads((Path(r["output"]) / "completion.json").read_text())["consumed_total_flops"] for r in history if r != chosen),
                 "notes": ["Budget selection uses baseline final scores only; all pilot costs are reported separately.",
                           "Selected pilot is reused as the comparison baseline; count it once when totaling actual compute.",
                           "Fold 0, seed 0 is exploratory. Freeze before additional folds/seeds; use untouched test data for confirmatory claims.",
                           "Every objective starts from the same seeded initialization, not the selected baseline checkpoint."]}
    write_json(root / "budget_selection.json", selection)
    reference = {"status": "official_reference", "adopted_on": plan["client_date"],
                 "budget_flops": chosen["budget_flops"], "budget_gflops": chosen["budget_flops"] / 1e9,
                 "parent_budget_flops": plan["base_budget_flops"], "anchor_model": "unet",
                 "anchor_flops_per_update": plan["baseline_costs"]["unet"],
                 "baseline_max_updates": chosen["updates"], "baseline_patient_pass_epochs": chosen["updates"] / 10,
                 "baseline_final_dice": chosen["final_dice"], "training_patients": 10,
                 "input_shape": [1, 1, 64, 64, 64], "selection_plan_sha256": digest(args.plan),
                 "reference_definition_sha256": digest(protocol_dir / "reference_definition.json"),
                 "selected_protocol_sha256": digest(protocol_dir / "protocol.json"),
                 "accounting": template["accounting"], "budget_selection": selection["notes"],
                 "learning_rate_policy": template["adamw_schedule"]}
    write_json(root / "compute_budget_reference.json", reference)
    # Preserve the selected pilot as the comparison baseline, without extra training.
    baseline = root / "unet/dice/fold0/seed0/run"
    baseline.parent.mkdir(parents=True, exist_ok=True)
    if not baseline.exists():
        baseline.symlink_to(Path(chosen["output"]), target_is_directory=True)
    elif baseline.resolve() != Path(chosen["output"]).resolve():
        raise ValueError("Comparison baseline points to a different pilot")
    write_json(root / "source_manifest.json", {str(p.relative_to(source)): digest(p)
                                               for p in source.rglob("*") if p.is_file()})
    print(json.dumps({"BUDGET_FROZEN": selection}), flush=True)
    for arm in protocol["models"]["unet"]["objectives"]:
        if arm == "dice":
            continue
        verify_source()
        output = root / "unet" / arm / "fold0/seed0/run"
        print("START objective=" + arm, flush=True)
        subprocess.run([sys.executable, str(source / "scripts/train_compute_matched_constraints.py"),
                        "--model", "unet", "--objective", arm, "--fold", "0", "--seed", "0",
                        "--pkl", str(args.pkl), "--output", str(output), "--protocol-dir", str(protocol_dir),
                        "--threads", str(args.threads), "--resume"], env=environment, check=True)
    # Reuse the already tested campaign verifier with the selected protocol path.
    verification = ("from pathlib import Path; import sys; "
                    "import summarize_compute_matched_constraints as report; "
                    "report.DEFAULT_PROTOCOL=Path(sys.argv[2]); report.summarize(Path(sys.argv[1]))")
    subprocess.run([sys.executable, "-c", verification, str(root), str(protocol_dir)],
                   env=environment, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--pkl", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    run(parser.parse_args())
