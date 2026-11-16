"""Fine-tune MSD MedSAM3 on each official 208-crop training fold, with 52 held out."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
VM = Path("/content")
MEDICAL = VM / "full_data_MSD_medical"
BUNDLE = VM / "full_data_MSD"
SUFFIX = "_standard_fold"


def digest(path):
    with path.open("rb") as stream:
        checksum = hashlib.sha256()
        for block in iter(lambda: stream.read(1024*1024), b""):
            checksum.update(block)
        return checksum.hexdigest()


def build(destination):
    data = HERE.parents[1] / "datasets/Dataset101_MSD"
    official = json.loads((data / "splits_final.json").read_bytes())
    inventory = {}
    historic = set()
    jobs = {}
    with zipfile.ZipFile(HERE / "MSD_medical_protocols.zip") as old, zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as out:
        for name in sorted(n for n in old.namelist() if n.startswith("protocols/")):
            p = json.loads(old.read(name))
            historic.update(p["data_hashes"])
            fold = official[p["fold"]]
            train, val = list(fold["train"]), list(fold["val"])
            assert len(set(train)) == len(train) == 208 and len(set(val)) == len(val) == 52
            assert set(train).isdisjoint(val) and set(p["evaluation_cases"]) <= set(val)
            assert set(p["train_cases"]) <= set(train)
            if not inventory:
                for case in sorted(train + val):
                    for filename in ("imagesTr/"+case+"_0000.nii.gz", "labelsTr/"+case+".nii.gz"):
                        inventory[filename] = digest(data / filename)
            assert all(inventory[n] == h for n,h in p["data_hashes"].items())
            key = p["name"] + SUFFIX
            p.update(name=key, train_cases=train, evaluation_cases=val, train_crops=208, train_subjects=None,
                     data="/content/standard_msd_medsam/data", data_hashes=inventory,
                     paired_evaluation_cases=p["evaluation_cases"], official_fold_sha256=digest(data/"splits_final.json"),
                     support_selection="All 208 official training crops. The 52 official validation crops are held out.",
                     sampling="400 adjacent-slice draws. Native uniform sampling with duplicate draws replaced only to cover omitted crops.")
            p["archive_recipe_differences"].update(full_data_train=208)
            # The inherited study documents pilot selection. Preserve its source separately from this new split.
            for field in ("selection", "overlap_policy", "evaluation_union", "excluded_pilot_cases", "subset_seed"):
                p.pop(field,None)
            out.writestr("protocols/"+key+".json", json.dumps(p, indent=2))
            out.writestr("references/"+key+".json", old.read(name))
            jobs[key] = dict(fold=p["fold"],seed=p["seed"],paired_validation=len(p["paired_evaluation_cases"]))
        assert len(jobs) == 9 and len(inventory)==520 and len(historic)==406
        for filename in sorted(set(inventory)-historic):
            out.write(data / filename, "data/"+filename)
        out.writestr("jobs.json",json.dumps(jobs,indent=2))
        out.write(data / "splits_final.json","splits_final.json")
        for filename in ("standard_msd_medsam.py","medical_worker.py","covered_medical.py"):
            out.write(HERE / filename,filename)
    print(destination, destination.stat().st_size, digest(destination))


def run(keys):
    from continue_colab import existing, publish, session, validate_output
    jobs=json.loads((HERE/"jobs.json").read_bytes())
    original_splits=json.loads((HERE/"splits_final.json").read_bytes())
    drive=session()
    with (HERE/"queue.lock").open("a") as guard:
        fcntl.flock(guard,fcntl.LOCK_NB|fcntl.LOCK_EX)
        for key in keys or sorted(jobs):
            assert key in jobs
            spec=HERE/"protocols"/(key+".json")
            p=json.loads(spec.read_bytes())
            fold=original_splits[p["fold"]]
            assert p["train_cases"]==fold["train"] and p["evaluation_cases"]==fold["val"]
            assert digest(HERE/"splits_final.json")==p["official_fold_sha256"]
            for name,checksum in p["data_hashes"].items():
                target=HERE/"data"/name
                if not target.exists():
                    target.parent.mkdir(parents=True,exist_ok=True)
                    os.link(MEDICAL/"data"/name,target)
                assert digest(target)==checksum, name
            path=HERE/(key+"_results.zip")
            if existing(drive,path.name):
                continue
            out=HERE/"work"/key
            out.mkdir(parents=True,exist_ok=True)
            (HERE/"recovery").mkdir(exist_ok=True)
            src=BUNDLE/"source"
            env=dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,[HERE,BUNDLE/"compatibility",src,src/"scripts",src/"baselines/nnUNet"])),
                     PYTHONUNBUFFERED="1",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",WANDB_MODE="disabled")
            if not (out/"completion.json").exists():
                print("TRAIN",key,208,52,flush=True)
                with (out/"training.log").open("a") as log:
                    subprocess.run([str(BUNDLE/"venv/bin/python"),str(HERE/"covered_medical.py"),"--worker",
                                    str(spec),str(HERE/"work"),str(HERE/"recovery")],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            score,count=validate_output(key,HERE,True)
            assert count==208
            metrics=json.loads((out/"evaluation.json").read_bytes())["records"]
            validation={r["case_name"]:r for r in metrics if r["split"]=="validation"}
            paired=mean(validation[c]["macro_hard_dice"] for c in p["paired_evaluation_cases"])
            proof=dict(key=key,train=208,validation=52,paired_cases=len(p["paired_evaluation_cases"]),updates=200,draws=400,
                       dice_all_52=score,dice_paired=paired,official_split_sha256=p["official_fold_sha256"])
            (out/"standard_fold_receipt.json").write_text(json.dumps(proof,indent=2))
            if not path.exists():
                with zipfile.ZipFile(path.with_suffix(".partial"),"w",zipfile.ZIP_DEFLATED) as archive:
                    for item in sorted(out.iterdir()):
                        if item.suffix in (".log",".json",".jsonl",".csv"):archive.write(item,item.name)
                    for file in ("standard_msd_medsam.py","covered_medical.py","medical_worker.py","splits_final.json"):
                        archive.write(HERE/file,file)
                    archive.write(HERE/"references"/(key+".json"),"historical_pool_protocol.json")
                path.with_suffix(".partial").replace(path)
            with zipfile.ZipFile(path) as archive:
                assert archive.testzip() is None
            result=publish(drive,key,path)
            print("VERIFIED",proof,result["id"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--build",type=Path)
    parser.add_argument("keys",nargs="*")
    args=parser.parse_args()
    if args.build:
        build(args.build)
    else:
        run(args.keys)
