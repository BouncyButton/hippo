"""Freeze a paired, twelve-run stronger-constraint pilot from the verified suite."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from scripts.geofwi_cluster.campaign import CAP, read, write, sha
from scripts.geofwi_constraints.prepare import ROOT, replace

PARENT_NAME = "geofwi3d_constraints_3fold_3seed_20261020"
PARENT = ROOT / "experiments" / PARENT_NAME
NAME = "geofwi3d_constraint_strength_20261020"
STAGE = ROOT / "experiments" / NAME
REPORT = ROOT / "reports" / NAME
REMOTE = Path("/mnt/beegfsstudents/home/3160552") / NAME
DESIGN = {
    "unetr_vit_b16": dict(arm="dice_bands_edge",
                          variants={"x2": [0.2, 1.0], "x4": [0.4, 2.0]}),
    "nnunet": dict(arm="dice_bands_sdf_depth",
                   variants={"x1p5": [0.9, 1.125], "x2": [1.2, 1.5]}),
}


def prepare():
    assert not (STAGE / "submission.json").exists(), "Submitted stages are immutable"
    STAGE.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    inventory = read(PARENT / "manifest.json")
    common = {"baseline_runtime.py", "worker_entry.py", "campaign.py", "splits.json",
              "data_manifest.json", "baseline_results.json", "cpu_audit_basic.json",
              "cpu_audit_unetr.json", "run.sbatch"}
    for name, digest in inventory.items():
        if name.startswith("bundles/") or name in common:
            assert sha(PARENT / name) == digest, name
            dest = STAGE / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PARENT / name, dest)
    if not (STAGE / "data").exists():
        (STAGE / "data").symlink_to((PARENT / "data").resolve(), target_is_directory=True)
    snapshot = read(ROOT / "reports" / PARENT_NAME / "stopped_snapshot.json")
    completed = {r["id"]: r for r in snapshot["rows"]}
    parent_tasks = {t["id"]: t for t in read(PARENT / "plan.json")["tasks"]}
    tasks, controls, calibrations, differences = [], {}, {}, {}
    # One model per lane; modest and stronger settings each retain all three seeds.
    for level in range(2):
        for seed in range(3):
            for model, design in DESIGN.items():
                strength, (target, cap) = list(design["variants"].items())[level]
                arm = design["arm"]
                origin_id = f"GeoFWI3D_{model}_{arm}_f0s{seed}"
                assert origin_id in completed
                controls[origin_id] = completed[origin_id]
                calibrations[origin_id] = snapshot["calibrations"][origin_id]
                task = deepcopy(parent_tasks[origin_id])
                task.update(id=f"{origin_id}_strength_{strength}", strength=strength,
                            control_id=origin_id, calibration_target_ratio=target,
                            calibration_p95_cap=cap)
                folder = STAGE / "protocols" / task["id"]
                folder.mkdir(parents=True, exist_ok=True)
                origin = PARENT / "protocols" / origin_id
                for file in origin.iterdir():
                    if file.is_file() and file.name != "protocol.json":
                        assert sha(file) == inventory[str(file.relative_to(PARENT))]
                        shutil.copyfile(file, folder / file.name)
                write(folder / "dataset.json", dict(
                    dataset_root=str(REMOTE / "data"), manifest=str(REMOTE / "data/fold_0_manifest.json")))
                before = read(origin / "protocol.json")
                after = deepcopy(before)
                after.update(name=NAME, baseline_run_id=task["id"],
                             data=str(REMOTE / "protocols" / task["id"] / "dataset.json"),
                             pkl_sha256=sha(folder / "dataset.json"),
                             calibration_target_ratio=target, calibration_p95_cap=cap,
                             strength=strength, constraint_control=completed[origin_id])
                if model == "unetr_vit_b16":
                    after["rescue_arm"] = task["id"]
                diff = {k: dict(before=before.get(k), after=after[k])
                        for k in after if before.get(k) != after[k]}
                assert set(diff) <= {"name", "baseline_run_id", "data", "pkl_sha256",
                                     "calibration_target_ratio", "calibration_p95_cap",
                                     "strength", "constraint_control", "rescue_arm"}
                assert after["models"] == before["models"]  # Same charged cost/schedule.
                assert after["calibration_target_ratio"] > before["calibration_target_ratio"]
                assert after["calibration_p95_cap"] > before["calibration_p95_cap"]
                differences[task["id"]] = diff
                write(folder / "protocol.json", after)
                tasks.append(task)
    write(STAGE / "controls.json", dict(rows=list(controls.values()),
                                       parent_stage=str(REMOTE.parent / PARENT_NAME)))
    write(STAGE / "control_calibrations.json", calibrations)
    write(STAGE / "protocol_differences.json", differences)
    write(STAGE / "design.json", DESIGN)
    write(STAGE / "plan.json", dict(
        name=NAME, workers=2, cap=CAP, tasks=tasks, reused_rows=[], total_cells=12,
        new_runs=12, baseline_runs=0, baselines_reused=6, constraint_controls_reused=6,
        parent_manifest_sha256=sha(PARENT / "manifest.json"),
        design="Fold 0 seeds 0–2; two higher calibration target/cap settings per model; all outcomes retained",
        selection="Best completed three-seed mean arm per model, selected on fold 0. Exploratory tuning; folds 1/2 remain confirmation data.",
        strength_definition="Multipliers apply to calibration target and p95 cap, not necessarily every sequentially calibrated coefficient.",
        storage=dict(minimum_free_gib=54, estimated_additional_peak_gib=46,
                     laptop_required=False, archive="archives/",
                     policy="Verify cluster metadata archives before retiring completed weights; preserve interrupted recovery"),
        continuation="Cluster Slurm self-submission; no collector or laptop dependency"))
    controller = STAGE / "campaign.py"
    replace(controller, 'return t["model"], t["fold"], t["seed"], t["arm"]',
            'return t["model"], t["fold"], t["seed"], t["arm"], t["strength"]')
    replace(controller,
            'expected = {(m, f, s, a) for m in MODELS for f in range(3) for s in range(3) for a in ARMS}',
            'design = read(stage / "design.json")\n'
            '    expected = {(m, 0, s, d["arm"], v) for m, d in design.items() for s in range(3) for v in d["variants"]}')
    replace(controller, 'assert len(cells) == len(expected) == 468',
            'assert len(cells) == len(expected) == 12\n'
            '    assert plan["total_cells"] == 12 and plan["baseline_runs"] == 0')
    replace(controller, 'assert p["objective"] == t["arm"] != "dice"',
            'assert p["objective"] == t["arm"] != "dice"\n'
            '        assert [p["calibration_target_ratio"], p["calibration_p95_cap"]] == design[t["model"]]["variants"][t["strength"]]\n'
            '        assert p["strength"] == t["strength"]')
    replace(controller, 'complete = len(rows) == 468',
            'controls = {r["id"]: r for r in read(stage / "controls.json")["rows"]}\n'
            '    for r in rows:\n'
            '        control = controls[r["control_id"]]\n'
            '        r["control_final_dice"] = control["final_dice"]\n'
            '        r["delta_vs_original_constraints_pp"] = 100 * (r["final_dice"] - control["final_dice"])\n'
            '    complete = len(rows) == 12')
    replace(controller, 'required_cells=468', 'required_cells=12')
    replace(controller, 'baselines_reused=36', 'baselines_reused=6, constraint_controls_reused=6')
    replace(controller, 'assert usage["free_gib"] >= 24', 'assert usage["free_gib"] >= 54')
    # Check each strength's actual calibration/loss before its first run.
    replace(STAGE / "baseline_runtime.py", 'f"{lane_id}_{t[\'model\']}_{t[\'arm\']}"',
            'f"{lane_id}_{t[\'model\']}_{t[\'arm\']}_{t[\'strength\']}"')
    replace(STAGE / "run.sbatch", "hippo-geofwi-constraints", "hippo-geofwi-strength")
    write(STAGE / "source_provenance.json", dict(
        parent_manifest_sha256=sha(PARENT / "manifest.json"),
        unchanged_training_sources={n: d for n, d in inventory.items() if n.startswith("bundles/") or n == "worker_entry.py"},
        controller_changes="Twelve-cell strength identity, paired controls, stronger preflight isolation and quota gate"))
    print(json.dumps(dict(stage=str(STAGE), runs=len(tasks), design=DESIGN)))


def freeze():
    checks = read(STAGE / "strength_checks.json")
    assert checks["status"] == "passed" and checks["optimizer_updates"] == 0
    write(STAGE / "cpu_contract_checks.json", dict(
        status="passed", optimizer_updates=0, strength_checks_sha256=sha(STAGE / "strength_checks.json"),
        inherited_objective_and_geometry_audit=read(PARENT / "cpu_contract_checks.json"),
        controller_tests=10))
    shutil.copyfile(__file__, STAGE / "prepare_strength.py")
    inventory = {str(p.relative_to(STAGE)): sha(p) for p in sorted(STAGE.rglob("*"))
                 if p.is_file() and "data" not in p.relative_to(STAGE).parts
                 and "__pycache__" not in p.parts and p.name != "manifest.json"}
    write(STAGE / "manifest.json", inventory)
    subprocess.run([str(ROOT / ".venv/bin/python"), str(STAGE / "campaign.py"),
                    "verify", "--stage", str(STAGE)], check=True)
    payload = REPORT / "frozen_payload.tar.gz"
    with tarfile.open(payload, "w:gz", compresslevel=1) as tar:
        for name in sorted(inventory):
            tar.add(STAGE / name, arcname=name, recursive=False)
        tar.add(STAGE / "manifest.json", arcname="manifest.json", recursive=False)
    write(REPORT / "preparation.json", dict(
        status="verified_ready_to_submit", manifest_sha256=sha(STAGE / "manifest.json"),
        payload_sha256=sha(payload), payload_bytes=payload.stat().st_size,
        new_runs=12, baseline_runs=0, laptop_required=False))
    for name in ("plan.json", "design.json", "strength_checks.json", "controls.json"):
        shutil.copyfile(STAGE / name, REPORT / name)


if __name__ == "__main__":
    import sys
    freeze() if "--freeze" in sys.argv else prepare()
