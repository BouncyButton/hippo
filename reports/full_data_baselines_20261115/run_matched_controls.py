"""Run and verify native-resolution, budget-limited full-coverage paper controls."""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def result_kind(row):
    if "original_stop" in row:
        return "native_full_pass_budget_exception", "_native_pass_results.zip"
    if row.get("training_patch"):
        return "training_patch_full_coverage", "_training_patch_results.zip"
    if row.get("development_peak_lr"):
        return "development_peak_lr_full_coverage", "_lr_development_results.zip"
    return "paper_matched_full_coverage", "_paper_matched_results.zip"


def train(key, original, out):
    row = read(HERE/"jobs.json")[key]
    folder = HERE/"protocols"/key
    protocol = read(folder/"protocol.json")
    assert digest(folder/"protocol.json") == row["protocol_sha256"] and row["feasible"]
    ds = row["dataset"]
    data = HERE/"paper_data_index.json" if ds=="HFH" else original/"data.pkl"
    if ds=="HFH":
        assert digest(data)==protocol["pkl_sha256"]
        assert read(data)["cases"]==read(original/"data_index.json")["cases"]
        for name,checksum in read(HERE/"hfh_data_sha256.json").items():
            assert digest(original/"data"/name)==checksum, name
    args = argparse.Namespace(model=protocol["trainer_alias"], objective="dice", seed=row["seed"],
        fold=row["fold"], pkl=data, protocol_dir=folder, output=out, threads=4, resume=True, device="cuda", smoke_steps=0)
    if ds in ("MSD","ADNI"):
        cmd = [sys.executable, str(HERE/"entry.py"), "--model", args.model, "--objective", "dice",
               "--fold", str(args.fold), "--seed", str(args.seed), "--protocol-dir", str(folder),
               "--pkl", str(data), "--output", str(out), "--resume"]
        recovery=out.parent.parent/"recovery"/key
        recovery.mkdir(parents=True,exist_ok=True)
        environment = dict(os.environ, HIPPO_COLAB_RECOVERY=str(recovery))
        if ds == "MSD" and (HERE / "fast_msd_storage.py").is_file():
            from fast_msd_storage import run as run_with_staged_checkpoints
            run_with_staged_checkpoints(cmd, environment, out)
        else:
            subprocess.run(cmd, env=environment, check=True)
    elif ds=="MNI":
        from scripts.mni_constraints.winning_matrix2030 import recipe
        recipe.train(args)
    else:
        from scripts.cross_dataset_campaign import conventional as exp
        from scripts.adni_constraints import objectives as binary
        from scripts.colab_hfh import baseline
        exp.configure(protocol, "dice")
        binary.bind()
        exp.base.PreparedPatients = baseline.PreparedPatients
        exp.base.evaluate = baseline.evaluate
        exp.base._load_pkl_dataframe = lambda path: dict(read(path), data=str(original/"data"))
        exp.base.build_loss_and_optimizer = exp.ORIGINAL_LOSS_OPTIMIZER
        if "seed0_plateau_schedule" in protocol:
            schedule = protocol["seed0_plateau_schedule"]
            assert schedule == dict(original_prefix_updates=20,plateau_lr=.07,decay_start=60,horizon=90,power=.9)
            from archived_hfh_schedule import plateau_lr
            exp.base.learning_rate = lambda name, step, maximum, initial: plateau_lr(step)
        rate=exp.base.learning_rate
        for step, expected in row["prior_lr"].items():
            observed=rate(args.model,int(step)-1,row["updates"],protocol["initial_lrs"][args.model])
            assert math.isclose(observed,expected,rel_tol=1e-11,abs_tol=1e-13), (key,step,observed,expected)
        builder=exp.base.build_model
        def checked_builder(name, architecture):
            model=builder(name,architecture)
            fingerprint=exp.trainer.model_digest(model)
            assert fingerprint==row["initialization_sha256"], (key,"initialization mismatch",fingerprint)
            (out/"initialization.json").write_text(json.dumps(dict(model_state_sha256=fingerprint)))
            return model
        exp.base.build_model=checked_builder
        exp.trainer.run(args)


def pack(key,out):
    row=read(HERE/"jobs.json")[key]
    folder=HERE/"protocols"/key
    p=read(folder/"protocol.json")
    split=read(folder/"splits_10patients.json")[row["fold"]]
    done=read(out/"completion.json")
    c=read(out/"config.json")
    assert c["protocol_sha256"]==row["protocol_sha256"] and c["augmentation"]==p["augmentation"]
    assert c["train_cases"]==split["train"] and c["validation_cases"]==split["val"]
    assert c["model_identity"]==row["model"] and c["seed"]==row["seed"] and c["fold"]==row["fold"]
    assert done["status"]=="complete" and done["updates"]==row["updates"] and not done["discarded_attempts"]
    assert done["consumed_training_flops"] <= p["budget_flops"] and not c["smoke_test"]
    for kind in ("latest","best"):
        assert digest(out/f"checkpoint_{kind}.pt")==done[f"checkpoint_{kind}_sha256"]
    trace=[json.loads(line) for line in (out/"control_exposure.jsonl").read_text().splitlines()]
    assert [r["step"] for r in trace]==list(range(1,row["updates"]+1))
    assert set(x["case"] for x in trace)==set(split["train"])
    patch = row.get("training_patch")
    if patch:
        assert all(r["patch_shape"] == patch and r["source_shape"] == [64, 64, 64] for r in trace)
        assert all(all(0 <= i <= 64-n for i,n in zip(r["origin"], patch)) for r in trace)
        assert all(0 < r["foreground_retained"] <= r["foreground_total"] for r in trace)
        profile = read(HERE / "profiles" / (row["model"] + ".json"))
        assert profile["flops_per_update"] * len(trace) == done["consumed_total_flops"]
    for filename,names in (("training_final.json",split["train"]),("validation_latest.json",split["val"])):
        result=read(out/filename)
        assert sorted(r["case_name"] for r in result["cases"])==sorted(names)
        assert all("foreground_fn" in r and "foreground_fp" in r for r in result["cases"])
        if patch:
            assert all(sum(sum(counts) for counts in r["confusion"]) == 64**3 for r in result["cases"])
    score=read(out/"validation_latest.json")["summary"]["macro_hard_dice"]
    assert math.isfinite(score) and 0<=score<=1
    arm, suffix = result_kind(row)
    receipt=dict(key=key,arm=arm,reference=row["source"],reference_sha256=row["source_sha256"],
        protocol_sha256=row["protocol_sha256"],updates=done["updates"],charged_flops=done["consumed_training_flops"],
        eligible=len(split["train"]),observed=len({r["case"] for r in trace}),validation_cases=len(split["val"]),dice=score)
    if patch:
        receipt.update(training_patch=patch,validation_shape=[64,64,64],per_update_flops=profile["flops_per_update"],
                       reference_native_updates=row["native_updates"])
    if row.get("development_peak_lr"):
        receipt.update(development_peak_lr=row["development_peak_lr"], original_peak_lr=row["original_peak_lr"])
    if "original_stop" in row:
        cut=row["original_stop"]
        assert len(trace)==len(split["train"]) and len({x["case"] for x in trace})==len(trace)
        assert all(x["shape"]==[64,64,64] for x in trace)
        assert all(x["lr"]==trace[cut-1]["lr"] for x in trace[cut:])
        assert done["consumed_total_flops"]==len(trace)*p["models"][p["trainer_alias"]]["objectives"]["dice"]["flops_per_update"]
        receipt.update(original_stop=cut, original_cap=row["original_cap"], native_shape=[64,64,64],
                       extra_updates=len(trace)-cut, optimizer_lr_at_original_stop=trace[cut-1]["lr"])
    (out/"matched_receipt.json").write_text(json.dumps(receipt,indent=2))
    path=HERE/(key+suffix)
    with zipfile.ZipFile(path.with_suffix(".partial"),"w",zipfile.ZIP_DEFLATED) as z:
        for item in sorted(out.iterdir()):
            if item.is_file() and item.suffix in (".json",".csv",".jsonl",".log"):z.write(item,item.name)
        for item in folder.iterdir():z.write(item,"protocols/"+item.name)
        if patch:
            z.write(HERE / "profiles" / (row["model"] + ".json"), "patch_profile.json")
            z.write(HERE / "source/scripts/cross_dataset_campaign/training_patch.py", "training_patch.py")
        if "original_stop" in row:
            z.write(HERE/"source/scripts/cross_dataset_campaign/capped_core.py","source/capped_core.py")
            z.write(Path(__file__),"source/run_matched_controls.py")
    path.with_suffix(".partial").replace(path)
    with zipfile.ZipFile(path) as z:assert z.testzip() is None
    return path,receipt


def run(keys,original,cloud):
    if cloud:
        sys.path.append("/content")
        from continue_colab import existing,publish,session
        drive=session()
    guard_name = keys[0] + ".lock" if not cloud and len(keys) == 1 else "runner.lock"
    with (HERE/guard_name).open("a") as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        jobs=read(HERE/"jobs.json")
        for key in keys or [k for k,r in jobs.items() if r["feasible"]]:
            assert jobs[key]["feasible"], (key,"A full pass exceeds this run's cap")
            _, suffix = result_kind(jobs[key])
            if cloud and existing(drive,key+suffix):continue
            out=HERE/"work"/key
            out.mkdir(parents=True,exist_ok=True)
            if not (out/"completion.json").exists():
                env=dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,[original/"compatibility",HERE/"source",
                    HERE/"source/scripts",HERE/"source/baselines/nnUNet"])),OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",
                    OPENBLAS_NUM_THREADS="4",nnUNet_compile="false",WANDB_MODE="disabled")
                print("TRAIN",key,flush=True)
                with (out/"training.log").open("a") as log:
                    subprocess.run([sys.executable,"-u",str(Path(__file__).resolve()),"--train",key,"--original",str(original)],
                                   check=True,env=env,stdout=log,stderr=subprocess.STDOUT)
            path,receipt=pack(key,out)
            if cloud:
                proof=publish(drive,key,path)
                receipt.update(drive_id=proof["id"],sha256=proof["sha256Checksum"])
            print(json.dumps(receipt),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--original",type=Path,required=True)
    parser.add_argument("--train")
    parser.add_argument("--cloud",action="store_true")
    parser.add_argument("keys",nargs="*")
    args=parser.parse_args()
    if args.train:train(args.train,args.original.resolve(),HERE/"work"/args.train)
    else:run(args.keys,args.original.resolve(),args.cloud)
