"""Set up an assigned full-fold worker from the original public MSD release."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request

HERE=Path(__file__).resolve().parent
PUBLIC="https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar"


def digest(file):
    with file.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def link_directory(target, source):
    assert source.is_dir()
    if not target.exists():
        target.symlink_to(source,target_is_directory=True)
    assert target.is_dir()


def stage(ds, fold):
    home=Path("/content")
    local=home/("full_data_"+ds)
    assets=home/("full_data_"+ds+"_medical")
    model=home/"full_data_MSD"
    medical=home/"full_data_MSD_medical"
    for folder in (model,medical):
        folder.mkdir(exist_ok=True)
    for name in ("venv","source","compatibility"):
        link_directory(model/name,local/name)
    for name in ("upstream","weights"):
        link_directory(medical/name,assets/name)
    for n,h in json.loads((HERE/"helper_source_hashes.json").read_bytes()).items():
        assert digest(model/n)==h,n
    key=f"MSD_medsam3_f{fold}_s17_standard_fold"
    p=json.loads((HERE/"protocols"/(key+".json")).read_bytes())
    assert digest(home/"full_baseline_medical_inputs/sam3.pt")==p["base_sha256"]
    assert digest(Path(p["lora_weights"]))==p["lora_sha256"]
    expected=p["data_hashes"]
    raw=HERE/"data"
    missing=[name for name in expected if not (raw/name).exists()]
    if missing:
        print("Downloading the public Task04 archive",flush=True)
        with tempfile.NamedTemporaryFile(dir=HERE) as cache:
            urllib.request.urlretrieve(PUBLIC,cache.name)
            with tarfile.open(cache.name) as t:
                members={m.name.removeprefix("./"):m for m in t if m.isfile()}
                for name in missing:
                    source="Task04_Hippocampus/"+name.replace("_0000.nii.gz",".nii.gz")
                    target=raw/name
                    target.parent.mkdir(parents=True,exist_ok=True)
                    payload=t.extractfile(members[source]).read()
                    assert hashlib.sha256(payload).hexdigest()==expected[name],name
                    temp=target.with_suffix(".partial")
                    temp.write_bytes(payload)
                    temp.replace(target)
    assert all(digest(raw/n)==h for n,h in expected.items()) and len(expected)==520
    print("VERIFIED original public data, released weights and trainer",ds,fold,flush=True)


def start(ds, fold):
    here=Path("/content")
    run="ADNI_medsam3_f1_s"+("0" if ds=="ADNI" else "1")+"_coverage_control"
    waiting=here/"covered_medical_ADNI/work"/run/"completion.json"
    keys=[f"MSD_medsam3_f{fold}_s{s}_standard_fold" for s in (17,83,191)]
    assert json.loads((HERE/"assignment.json").read_text())==keys
    while not waiting.is_file():
        time.sleep(45)
    while int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())<15000:
        time.sleep(45)
    print("Training assignment",keys,flush=True)
    os.execv(sys.executable,[sys.executable,"-u",str(HERE/"standard_msd_medsam.py"),*keys])


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("dataset",choices=("HFH","ADNI"))
    parser.add_argument("fold",type=int,choices=(1,3))
    parser.add_argument("--start",action="store_true")
    args=parser.parse_args()
    assert (args.dataset,args.fold) in (("HFH",3),("ADNI",1))
    stage(args.dataset,args.fold)
    if args.start:
        start(args.dataset,args.fold)
