"""Build separate full-pool scratch controls from the archived low-data training code."""
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

D = Path(__file__).resolve().parent
ROOT = D.parents[1]
REPORTS = ROOT / "reports"
PARENTS = {
    "HFH": REPORTS / "hfh_colab_exploration_20261111/baseline_v2/bundle",
    "ADNI": REPORTS / "adni_colab_seeds014_drive_optimized_20261018/package/seed4_bundle",
    "MSD": REPORTS / "msd_colab_seeds0220_recovery_20261018/package/seed20_bundle",
    "MNI": REPORTS / "mni_winning_recipes_matrix2030_53t_0519_20261109/frozen",
}


def dump(value):
    return (json.dumps(value, indent=2) + "\n").encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source_for_control(path):
    code = path.read_text()
    # Native affine is only used by geometry losses or mild affine augmentation.
    # These controls use Dice with either no augmentation or the archived XY flips.
    old = "np.asarray(protocol['native_geometry'][batch['case_name']]['model_affine'])"
    assert code.count(old) == 1
    code = code.replace(old, f"({old} if 'depth' in args.objective or protocol['augmentation']=='mild_v1' else None)")
    old = "        step += 1\n        cursor += 1"
    assert code.count(old) == 1
    code = code.replace(old, '        step += 1\n        with (args.output / "control_exposure.jsonl").open("a") as f:\n            f.write(json.dumps(dict(step=step, case=batch["case_name"], lr=lr))+"\\n")\n        cursor += 1')
    compile(code, str(path), "exec")
    return code.encode()


def build(ds):
    parent = PARENTS[ds]
    manifest = json.loads((D / "manifest.json").read_text())
    splits = json.loads((D / "splits.json").read_text())
    with (D / "scratch_recipe_comparison.csv").open() as f:
        refs = {r["cell"]: r for r in csv.DictReader(f)}
    out = D / "matched_controls"
    out.mkdir(exist_ok=True)
    jobs = {}
    with zipfile.ZipFile(out / (ds + "_matched.zip"), "w", zipfile.ZIP_DEFLATED) as bundle:
        if ds == "HFH":
            bundle.write(parent/"data_index.json","paper_data_index.json")
            crops={f.name:sha(f.read_bytes()) for f in sorted((parent/"data").glob("*.npz"))}
            assert len(crops)==50
            bundle.writestr("hfh_data_sha256.json",dump(crops))
            plateau = (ROOT/"scripts/colab_hfh/nnunet_seed0_plateau_worker.py").read_text()
            fn = next(n for n in ast.parse(plateau).body if isinstance(n,ast.FunctionDef) and n.name=="plateau_lr")
            bundle.writestr("archived_hfh_schedule.py", ast.get_source_segment(plateau,fn)+"\n")
        for f in sorted((parent / "source").rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc":
                n = f.relative_to(parent).as_posix()
                if n == "source/scripts/cross_dataset_campaign/capped_core.py":
                    bundle.writestr(n, source_for_control(f))
                else:
                    bundle.write(f, n)
        if ds in ("MSD", "ADNI"):
            for name in ["runtime.py", "entry.py", "initialization_reference.json"]:
                bundle.write(parent / name, name)
        for key, row in manifest.items():
            if row["dataset"] != ds or row["model"] == "medsam3":
                continue
            ref = refs[key]
            location = ref["few_recipe_source"].split("!")
            init_hash = None
            metric_lr = {}
            if len(location) == 2:
                with zipfile.ZipFile(ROOT / location[0]) as old:
                    raw = old.read(location[1])
                    init_hash = json.loads(old.read("results/initialization.json"))["model_state_sha256"]
                    metrics = csv.DictReader(io.StringIO(old.read("results/metrics.csv").decode()))
                    metric_lr = {r["step"]: float(r["learning_rate"]) for r in metrics}
                    config = json.loads(old.read("results/config.json"))
                    assert config["fold"] == row["partition"] and config["seed"] == row["seed"]
            else:
                raw = (ROOT / location[0]).read_bytes()
            assert sha(raw) == ref["few_protocol_sha256"], key
            protocol = json.loads(raw)
            split = splits[row["split"]]
            full = split["train"]
            val = split["validation"]
            assert len(full) == len(set(full)) and not set(full) & set(val)
            fold = row["partition"]
            saved = json.loads((ROOT / row["archived_validation_source"]["file"]).read_text())[fold]
            assert saved["val"] == val and set(saved["train"]) <= set(full)
            if len(location) == 2:
                assert config["validation_cases"] == val and set(config["train_cases"]) == set(saved["train"])
            alias = protocol["trainer_alias"]
            schedule = protocol["models"][alias]["objectives"]["dice"]
            updates = schedule["max_steps"]
            assert protocol["batch_size"] == 1 and updates * schedule["flops_per_update"] <= protocol["budget_flops"]
            source_arch = (ROOT / row["archived_protocol"]["file"]).with_name("nnunet_architecture.json").read_bytes()
            assert sha(source_arch) == protocol["nnunet_architecture_sha256"]
            sets = [dict(train=[], val=[]) for _ in range(max(3,fold+1))]
            sets[fold] = dict(train=full, val=val)
            split_bytes = dump(sets)
            protocol["train_count"] = len(full)
            protocol["splits_sha256"] = sha(split_bytes)
            schedule["patient_pass_epochs"] = updates / len(full)
            if not init_hash:
                init_hash = protocol.get("baseline_reference", {}).get("initialization_sha256")
            prefix = "protocols/" + key + "/"
            for name, content in (("protocol.json",dump(protocol)),("archived_protocol.json",raw),
                                  ("splits_10patients.json",split_bytes),("nnunet_architecture.json",source_arch)):
                bundle.writestr(prefix + name, content)
            jobs[key] = dict(dataset=ds, model=row["model"], fold=fold, seed=row["seed"], updates=updates,
                             eligible=len(full), feasible=updates>=len(full), prior_lr=metric_lr,
                             initialization_sha256=init_hash, protocol_sha256=sha(dump(protocol)),
                             source=ref["few_recipe_source"], source_sha256=sha(raw))
        assert len(jobs) == 36
        bundle.writestr("jobs.json", dump(jobs))
        bundle.write(D / "run_matched_controls.py", "run_matched_controls.py")
        if ds == "MSD":
            bundle.write(D / "fast_msd_storage.py", "fast_msd_storage.py")
    print(ds, sum(r["feasible"] for r in jobs.values()), "native full-coverage runs within cap")


if __name__ == "__main__":
    import sys
    for dataset in sys.argv[1:] or PARENTS:
        build(dataset)
