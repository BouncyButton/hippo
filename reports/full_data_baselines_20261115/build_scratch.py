"""Package full-data runs using the original Colab dependencies and trainer."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

D = Path(__file__).resolve().parent
ROOT = D.parents[1]
HFH = ROOT / "reports/hfh_colab_exploration_20261111/bundle"
MSD = ROOT / "reports/msd_colab_seeds0220_recovery_20261018/package/seed20_bundle"


def read(p):
    return json.loads(p.read_text())


def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2)+"\n")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def replace_once(s, old, new):
    if s.count(old) != 1:
        raise ValueError("Frozen trainer anchor differs: "+old[:70])
    return s.replace(old, new)


def build(ds, target):
    target.mkdir(exist_ok=True, parents=True)
    keys = {k:r for k,r in read(D/"manifest.json").items() if r["dataset"]==ds and r["model"]!="medsam3"}
    split = read(D/"splits.json")
    with tempfile.TemporaryDirectory(dir=target) as stage:
        b = Path(stage)
        source = b/"source"
        shutil.copytree(HFH/"source",source,ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copytree(HFH/"compatibility",b/"compatibility",ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copy2(MSD/"requirements-colab.txt", b/"requirements.txt")
        shutil.copy2(D/"scratch_worker.py",b/"worker.py")
        core = source/"scripts/cross_dataset_campaign/capped_core.py"
        trainer = replace_once(core.read_text(), "    check_no_augmentation(protocol)",
            '    if args.objective != "dice" or protocol["augmentation"] != "mild_v1":\n        raise ValueError("Expected ordinary supervised full-data training")')
        lo,hi=trainer.index("        original_affine ="), trainer.index("        active_arm =",trainer.index("        original_affine ="))
        trainer = trainer[:lo]+'        image, label, augmentation_statistics = base.augment_mild(image, label, generator=augmentation_rng)\n'+trainer[hi:]
        trainer = replace_once(trainer, "        depth = geometry_weights(label, current_affine, protocol['depth_geometry']).to(device) if 'depth' in active_arm else None","        depth = None")
        trainer = replace_once(trainer, "        step += 1\n        cursor += 1",
            '        step += 1\n        with (args.output / "exposure.jsonl").open("a") as trace:\n            trace.write(json.dumps(dict(update=step,case=batch["case_name"],augmentation=augmentation_statistics,loss=float(loss.detach()),lr=lr))+"\\n")\n        cursor += 1')
        core.write_text(trainer)
        if ds=="HFH":
            shutil.copytree(HFH/"data",b/"data")
            data = b/"data_index.json"
            shutil.copy2(HFH/"data_index.json",data)
        else:
            files = {"ADNI":"datasets/Dataset103_ADNI/adni_hippocampus_full.pkl","MSD":"datasets/Dataset101_MSD/msd_hippocampus_full_cluster_20260906.pkl",
                     "MNI":"experiments/mni_other_baselines_20261106/data/MNI.pkl"}
            data = b/"data.pkl"
            shutil.copy2(ROOT/files[ds],data)
        data_sha = sha(data)
        for key,row in keys.items():
            folder = b/"protocols"/key
            old = ROOT/row["archived_protocol"]["file"]
            p = read(old)
            assert p["pkl_sha256"]==data_sha
            a=p["trainer_alias"]
            s=split[row["split"]]
            # The trainer indexes by original fold even though the run consumes only one.
            splits=[dict(train=[],val=[]) for _ in range(3)]
            splits[row["partition"]] = dict(train=s["train"],val=s["validation"])
            write(folder/"splits_10patients.json",splits)
            shutil.copy2(old.with_name("nnunet_architecture.json"),folder/"nnunet_architecture.json")
            p["archive_recipe_differences"] = dict(few_data_lr=p["initial_lrs"][a], full_data_lr=row["lr"],
                few_data_augmentation=p["augmentation"], full_data_augmentation="mild_v1",
                few_data_train_count=p["train_count"], full_data_train_count=len(s["train"]))
            if ds=="MNI":
                reference=(ROOT/"experiments/unetr_subfields_cap_20261106/protocols/MNI/protocol.json" if row["model"]=="unetr_vit_b16"
                           else ROOT/"experiments/mni_other_baselines_20261106/protocols"/row["model"]/"protocol.json")
                ordinary=read(reference)
                assert ordinary["models"][a]["parameters"]==p["models"][a]["parameters"]
                native=ordinary["models"][a]["objectives"]["dice"].copy()
                cost=native["flops_per_update"]
                cap=p["budget_flops"]
                updates=cap//cost
                native.update(max_steps=updates,consumed_training_flops=updates*cost,consumed_flops=updates*cost,
                              consumed_total_flops=updates*cost,unused_flops=cap-updates*cost)
                p["archive_recipe_differences"].update(few_data_objective=p["training_loss"],full_data_objective=ordinary["training_loss"],
                    few_data_lr_horizon=p["lr_horizon_updates"],full_data_lr_horizon=ordinary["lr_horizon_updates"],
                    ordinary_profile=str(reference.relative_to(ROOT)),ordinary_profile_sha256=sha(reference))
                p["models"][a]["objectives"]["dice"]=native
                p.update(training_loss=ordinary["training_loss"],lr_horizon_updates=ordinary["lr_horizon_updates"])
                row.update(target_updates=updates,full_data_flops_per_update=cost)
            p.update(augmentation="mild_v1", train_count=len(s["train"]),validation_count=len(s["validation"]),
                data_file=data.name, splits_sha256=sha(folder/"splits_10patients.json"),
                folds=[row["partition"]],seeds=[row["seed"]],requested_arm_order=["dice"])
            p["initial_lrs"][a]=row["lr"]
            p["models"][a]["objectives"]["dice"]["patient_pass_epochs"]=row["target_updates"]/len(s["train"])
            p.pop("native_geometry",None)
            write(folder/"protocol.json",p)
        write(b/"cells.json",keys)
        write(b/"checksums.json",{str(f.relative_to(b)):sha(f) for f in sorted(b.rglob("*")) if f.is_file()})
        archive=target/f"full_data_{ds}.zip"
        with zipfile.ZipFile(archive.with_suffix(".tmp"),"w",zipfile.ZIP_DEFLATED,compresslevel=2) as z:
            for f in sorted(b.rglob("*")):
                if f.is_file(): z.write(f,f.relative_to(b))
        archive.with_suffix(".tmp").replace(archive)
        print(ds,archive.stat().st_size,sha(archive),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("dataset",choices=["ADNI","MSD","HFH","MNI"])
    parser.add_argument("target",type=Path)
    args=parser.parse_args()
    build(args.dataset,args.target)
