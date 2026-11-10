"""Freeze the 468-cell constraint matrix; never schedule a baseline."""
from copy import deepcopy
from pathlib import Path
import json
import shutil
from itertools import zip_longest

from scripts.geofwi_cluster.campaign import CAP, MODELS, read, write, sha

ROOT = Path(__file__).resolve().parents[2]
NAME = "geofwi3d_constraints_3fold_3seed_20261020"
PARENT = ROOT / "reports/geofwi3d_frozen_3fold_3seed_20261020"
REMOTE = Path("/mnt/beegfsstudents/home/3160552") / NAME
REPORT = ROOT / "reports" / NAME
ARMS = ("dice_bands", "dice_sdf", "dice_bands_sdf", "dice_bands_edge",
        "dice_bands_edge_sdf", "dice_depth", "dice_bands_depth", "dice_edge_depth",
        "dice_sdf_depth", "dice_bands_edge_depth", "dice_bands_sdf_depth",
        "dice_edge_sdf_depth", "dice_bands_edge_sdf_depth")


def replace(path, old, new, count=1):
    text = path.read_text()
    assert text.count(old) == count, (path, old, text.count(old))
    path.write_text(text.replace(old, new))


def schedule(costs, arm, policy, model):
    base = costs["dice"]["update_flops"]
    prefix = ramp = 9 if model == "nnunet" else 50
    host = 2 if model == "unetr_vit_b16" else 0
    full = costs[arm]["update_flops"] + host + int("depth" in arm) + 2 * int("sdf" in arm)
    early = costs[arm.replace("_sdf", "")]["update_flops"] + host + int("depth" in arm)
    late = prefix
    if policy == "delayed_5_of_9":
        candidate = prefix + max(0, CAP * 5 // 9 - prefix * base) // early
        late = min((candidate, candidate + 1), key=lambda n: (abs(prefix * base + (n-prefix)*early-CAP*5//9), n))
    else:
        early = full
    cal = costs[arm]["calibration_flops"]
    def consumed(n):
        return min(n, prefix)*base + max(0, min(n, late)-prefix)*early + max(0, n-late)*full
    maximum = 0
    while consumed(maximum + 1) + cal <= CAP:
        maximum += 1
    return dict(prefix_updates=prefix, sdf_prefix_updates=late, baseline_flops_per_update=base,
                early_flops_per_update=early, flops_per_update=full, calibration_flops=cal,
                constraint_warmup_updates=ramp, sdf_warmup_updates=ramp, budget_scope="optimization_only",
                cap_includes_calibration=True, max_steps=maximum, consumed_training_flops=consumed(maximum),
                consumed_flops=consumed(maximum), consumed_total_flops=consumed(maximum)+cal,
                unused_flops=CAP-consumed(maximum)-cal, patient_pass_epochs=maximum/19)


def balance_lanes(tasks):
    """Spread each model across both GPUs, keeping the broad loss checks first."""
    lanes = [[], []]
    for task in tasks:
        lane = min(range(2), key=lambda i: (sum(t["model"]==task["model"] for t in lanes[i]), len(lanes[i])))
        lanes[lane].append(task)
    if len(lanes[0]) < len(lanes[1]):
        lanes.reverse()
    assert len(lanes[0])-len(lanes[1]) <= 1
    for model in MODELS:
        assert abs(sum(t["model"]==model for t in lanes[0])-sum(t["model"]==model for t in lanes[1])) <= 1
    return [task for pair in zip_longest(*lanes) for task in pair if task is not None]


def prepare():
    stage = ROOT / "experiments" / NAME
    assert not (stage / "submission.json").exists(), "Cannot alter a submitted stage"
    stage.mkdir(parents=True, exist_ok=True)
    frozen = PARENT / "frozen"
    # Copy only frozen inventory; exclude generated import caches and data.
    for name, expected in read(frozen / "manifest.json").items():
        if not name.startswith("bundles/"):
            continue
        assert sha(frozen / name) == expected
        dst = stage / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(frozen / name, dst)
    for name in ("splits.json", "data_manifest.json"):
        shutil.copyfile(frozen / name, stage / name)
    for child in ("unetr", "basic"):
        shutil.copyfile(PARENT / f"cpu_audit_{child}.json", stage / f"cpu_audit_{child}.json")
    shutil.copyfile(PARENT / "final_results.json", stage / "baseline_results.json")
    shutil.copyfile(ROOT / "scripts/geofwi_cluster/campaign.py", stage / "baseline_runtime.py")
    for name in ("campaign.py", "worker_entry.py"):
        shutil.copyfile(Path(__file__).with_name(name), stage / name)
    shutil.copyfile(Path(__file__).with_name("objective.py"), stage / "bundles/unetr/constraint_objective.py")
    if not (stage / "data").exists():
        (stage / "data").symlink_to((frozen / "data").resolve(), target_is_directory=True)
    # Explicitly scoped changes to frozen sources; original bundles stay untouched.
    basic = stage / "bundles/basic"
    unetr = stage / "bundles/unetr"
    replace(basic/"worker.py", "exp.configure(p,'dice')", "exp.configure(p,p['objective'])")
    replace(basic/"worker.py", "objective='dice',fold=", "objective=p['objective'],fold=")
    core = basic / "source/scripts/cross_dataset_campaign/capped_core.py"
    replace(core, 'ledger["training_flops"] > budget', 'ledger["training_flops"] + ledger["calibration_flops"] > budget')
    replace(core, 'ledger["training_flops"] + cost > budget', 'ledger["training_flops"] + ledger["calibration_flops"] + cost > budget')
    replace(core, 'cal_cost = schedule["calibration_flops"]',
            'cal_cost = schedule["calibration_flops"]\n            if ledger["calibration_attempts"] or ledger["training_flops"] + cal_cost > budget:\n                raise ValueError("Calibration replay or cap overrun")')
    replace(core, 'budget - ledger["training_flops"] >=', 'budget - total >=')
    replace(core, '"unused_flops": budget - ledger["training_flops"]', '"unused_flops": budget - total, "cap_includes_calibration": True')
    core = unetr / "source/scripts/cross_dataset_campaign/capped_core.py"
    replace(core, 'image, label = augment(image, label, args.seed, charged)',
            'image, label = augment(image, label, args.seed, charged)\n            from constraint_objective import xy_affine\n            current_affine = xy_affine(original_affine, args.seed, charged, label.shape[-3])')
    for bundle in (basic, unetr):
        geometry_source = bundle/"source/scripts/depth_supervision/geometry.py"
        replace(geometry_source,
                'if np.linalg.norm(normal) < .1:\n        raise ValueError("Longitudinal axis nearly parallel to anatomical left-right; inspect geometry")',
                'if np.linalg.norm(normal) < .1:\n'
                '        # GeoFWI3D has no anatomical axes. Preserve source axis 0\n'
                '        # whenever defined; otherwise use the least parallel source\n'
                '        # axis, with deterministic axis-index tie breaking.\n'
                '        right = np.eye(3)[int(np.argmin(np.abs(longitudinal)))]\n'
                '        normal = right - (right @ longitudinal) * longitudinal')
    runtime = stage / "baseline_runtime.py"
    # The verifier is replaced by campaign.verify; the shared result validator
    # must select the actual arm.
    replace(runtime, "['objectives']['dice']", "['objectives'][t['arm']]", count=2)
    replace(runtime, 'if STOP or (stage/\'STOP\').exists():',
            'if STOP or (stage/\'STOP\').exists() or time.time() >= float(os.environ.get("HIPPO_ALLOCATION_DEADLINE", "inf")):')
    replace(runtime, 'f"{lane_id}_{t[\'model\']}"', 'f"{lane_id}_{t[\'model\']}_{t[\'arm\']}"')
    replace(runtime, "'--kill-on-bad-exit=1'", "'--kill-on-bad-exit=0'")
    start = runtime.read_text().index("        write(state,dict(status='waiting_for_verified_archive'")
    end = runtime.read_text().index("        archive.retire(out)", start)
    text = runtime.read_text()
    runtime.write_text(text[:start] + text[end:])
    # A fresh run environment must never inherit an old Colab half-budget gate.
    replace(runtime, "cmd=[sys.executable,'-u',str(stage/'worker_entry.py')",
            'env.pop("HIPPO_STAGE_LIMIT", None)\n        cmd=[sys.executable,\'-u\',str(stage/\'worker_entry.py\')')
    # Data/reference and source provenance are retained in the immutable stage.
    baseline_rows = read(stage / "baseline_results.json")["rows"]
    splits = read(stage / "splits.json")
    geometry = {c: dict(model_affine=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
                for split in splits for c in split["train"] + split["val"]}
    profiles, parents = {}, {}
    for m in MODELS:
        origin = ROOT / ("reports/adni_seismic_ce60_followon_20261014/frozen/protocols/GeoFWI3D_nnunet"
                        if m == "nnunet" else f"experiments/cross_dataset_full_20261014/protocols/GeoFWI3D_{m}_f0_target10")
        parents[m] = read(origin / "protocol.json")
        profiles[m] = read(REPORT / "unetr_profile.json") if m == "unetr_vit_b16" else read(origin / "profile.json")
    tasks = []
    for fold in range(3):
        for seed in range(3):
            for arm in ARMS:
                for model in MODELS:
                    if (model,fold,seed,arm) == ("unetr_vit_b16",0,0,"dice_bands"):
                        continue
                    child = "unetr" if model == "unetr_vit_b16" else "basic"
                    origin = frozen / "protocols" / f"GeoFWI3D_{model}_baseline_f1s{seed}"
                    p = deepcopy(read(origin / "protocol.json"))
                    identity = f"GeoFWI3D_{model}_{arm}_f{fold}s{seed}"
                    folder = stage / "protocols" / identity
                    folder.mkdir(parents=True, exist_ok=True)
                    write(folder / "dataset.json", dict(dataset_root=str(REMOTE / "data"), manifest=str(REMOTE / "data/fold_0_manifest.json")))
                    shutil.copyfile(stage / "splits.json", folder / "splits_10patients.json")
                    shutil.copyfile(origin / "nnunet_architecture.json", folder / "nnunet_architecture.json")
                    write(folder / "profile.json", profiles[model])
                    parent = parents[model]
                    policies = parent["sdf_policy"]
                    sched = schedule(profiles[model]["costs"], arm, policies[arm], model)
                    reference = next(r for r in baseline_rows if (r["model"],r["fold"],r["seed"]) == (model,fold,seed))
                    p.update(name=NAME, objective=arm, folds=[fold], seeds=[seed], baseline_run_id=identity,
                             constraints=True, requested_arm_order=[arm], native_geometry=geometry,
                             data=str(REMOTE/"protocols"/identity/"dataset.json"), pkl_sha256=sha(folder/"dataset.json"),
                             profile_sha256=sha(folder/"profile.json"), sdf_policy=policies,
                             calibration_target_ratio=.6 if model=="nnunet" else .1,
                             calibration_p95_cap=.75 if model=="nnunet" else .5,
                             constraint_warmup_updates=sched["constraint_warmup_updates"],
                             baseline_reference=reference, cap_includes_calibration=True,
                             budget_policy="Fixed total training plus calibration cap; full budget, no halfway gate",
                             constraint_geometry="Binary BANDS, EDGE, SDF and source-voxel DEPTH; frozen model-specific SDF timing",
                             depth_axis_policy="Source axis0 projected perpendicular to PC1; if norm<0.1, least-parallel canonical source axis (index ties). Fixed annotations only.")
                    p["models"][p["trainer_alias"]]["objectives"] = {arm: sched}
                    if child == "unetr":
                        p["rescue_arm"] = identity
                        p["rescue_validation_steps"] = sorted({min(sched["max_steps"], CAP*i//10//sched["flops_per_update"])
                                                              for i in range(1,11)} | {sched["max_steps"]})
                    write(folder / "protocol.json", p)
                    tasks.append(dict(id=identity,kind="conventional",dataset="GeoFWI3D",model=model,
                                      fold=fold,seed=seed,arm=arm,bundle=child,augmentation=p["augmentation"],
                                      updates=sched["max_steps"]))
    # Put the two largest/four-component loss preflights first, then interleave.
    heavy = [t for t in tasks if t["fold"]==0 and t["seed"]==0 and t["arm"]=="dice_bands_edge_sdf_depth"]
    tasks = balance_lanes(heavy + [t for t in tasks if t not in heavy])
    reused = ROOT / "reports/geofwi3d_unetr_lr6_20261019/bands_completed_evidence"
    shutil.copytree(reused, stage/"reused/unetr_bands_f0s0", dirs_exist_ok=True)
    done = read(reused/"completion.json")
    expected = schedule(profiles["unetr_vit_b16"]["costs"],"dice_bands","not_applicable","unetr_vit_b16")
    assert done["consumed_total_flops"] == expected["consumed_total_flops"] and done["updates"] == expected["max_steps"]
    reused_row = dict(model="unetr_vit_b16",fold=0,seed=0,arm="dice_bands",
                      final_dice=read(reused/"validation_latest.json")["summary"]["macro_hard_dice"],
                      total_flops=done["consumed_total_flops"], updates=done["updates"],
                      source="reused/unetr_bands_f0s0", archive_manifest_sha256=sha(reused/"archive_manifest.json"))
    write(stage/"plan.json",dict(name=NAME,workers=2,cap=CAP,tasks=tasks,reused_rows=[reused_row],
          total_cells=468,new_runs=len(tasks),baselines_reused=36,baseline_runs=0,
          storage=dict(minimum_free_gib=24,estimated_peak_logical_gib=14,mirrored_allowance_gib=28,
                       retained_metadata_allowance_gib=2,archive="archives/",laptop_required=False,
                       policy="Explicit user override: cluster-resident verified archives; retire completed bulky artifacts per run"),
          continuation="Slurm self-submission after clean 22-hour boundary; no laptop service; unknown/failed outcomes stop with recovery preserved",
          selection="Fixed folds/seeds; all final-budget results retained irrespective of Dice",
          parent_manifest_sha256=sha(frozen/"manifest.json")))
    batch = (frozen/"run.sbatch").read_text().replace("hippo-geofwi-3f3s","hippo-geofwi-constraints")
    (stage/"run.sbatch").write_text(batch)
    changed = {}
    for name, old_sha in read(frozen/"manifest.json").items():
        p = stage/name
        if name.startswith("bundles/") and p.exists() and sha(p)!=old_sha:
            changed[name] = dict(before=old_sha, after=sha(p))
    write(stage/"source_patch_manifest.json",dict(changed=changed, new_objective_sha256=sha(stage/"bundles/unetr/constraint_objective.py")))
    write(stage/"manifest.json",{str(p.relative_to(stage)):sha(p) for p in sorted(stage.rglob("*"))
                                if p.is_file() and "data" not in p.relative_to(stage).parts
                                and "__pycache__" not in p.parts and p.name not in ("manifest.json","cpu_contract_checks.json")})
    print(json.dumps(dict(stage=str(stage),tasks=len(tasks),reused=1,baselines_excluded=36)))


if __name__ == "__main__":
    prepare()
