"""Prepare replacement Colab runtimes from the checked VM-only uploads."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

from googleapiclient.http import MediaIoBaseDownload
from continue_colab import session

HASHES = {
    "HFH": ("d06a8b06c9ca51f1dfdb1645e4f2433b21e9aa029649534a0c0b6105c8247ced",
            "77110cad6b2978d8be1c27b7010e023a07e96202f71a15a6b7c6ba73a493d95d"),
    "ADNI": ("c238b6ddc12867c612db2ee2f13d47c717b3feabaea9a62396d686f17b5cded9",
             "20affa490f014b15d28596207d19bc65f6933426a987a93a3db3804738ca4f99"),
    "MSD": ("f7ddc121bdaaf80504065a4eeb09518335e0ebf70bbe1e3805135087b774c89c",
            "9502a5628192ed40ed1b195084a578daf82018ec73d1a74397a7f33b7845c167"),
}


def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def extract(path, dest, expected):
    assert digest(path)==expected, f"Input has not finished uploading: {path}"
    with zipfile.ZipFile(path) as z:
        assert all(not Path(n).is_absolute() and ".." not in Path(n).parts for n in z.namelist())
        z.extractall(dest)


def prepare(ds):
    top=Path("/content")
    scratch=top/f"full_data_{ds}"
    medical=top/f"full_data_{ds}_medical"
    extract(top/f"full_data_{ds}.zip",scratch,HASHES[ds][0])
    extract(top/f"{ds}_medical_inputs.zip",medical,HASHES[ds][1])
    for filename,checksum in json.loads((scratch/"checksums.json").read_text()).items():
        assert digest(scratch/filename)==checksum, filename
    venv=scratch/"venv"
    subprocess.run([sys.executable,"-m","venv","--without-pip","--system-site-packages",str(venv)],check=True)
    py=str(venv/"bin/python")
    for requirements in [scratch/"requirements.txt",medical/"requirements.txt"]:
        subprocess.run([py,"-m","pip","install","--disable-pip-version-check","-r",str(requirements)],check=True)
    print("READY_FOR_SCRATCH",ds,flush=True)
    prepare_weights(ds)


def prepare_weights(ds):
    top=Path("/content")
    medical=top/f"full_data_{ds}_medical"
    protocol=json.loads(next((medical/"protocols").glob("*.json")).read_text())
    assert digest(medical/"weights/best_lora_weights.pt")==protocol["lora_sha256"]
    base=top/"full_baseline_medical_inputs/sam3.pt"
    base.parent.mkdir(exist_ok=True)
    if not base.exists() or digest(base)!=protocol["base_sha256"]:
        drive=session()
        matches=drive.files().list(q="trashed=false and name='sam3.pt' and 'me' in owners",
            fields="files(id,size,sha256Checksum)",pageSize=100).execute(num_retries=4)["files"]
        match=next(f for f in matches if f["sha256Checksum"]==protocol["base_sha256"])
        download=base.with_suffix(".partial")
        with download.open("wb") as fh:
            request=MediaIoBaseDownload(fh,drive.files().get_media(fileId=match["id"]),chunksize=16*1024**2)
            done=False
            while not done:
                _,done=request.next_chunk(num_retries=5)
        assert digest(download)==protocol["base_sha256"]
        download.replace(base)
    print("READY_FOR_MEDICAL",ds,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("dataset",choices=sorted(HASHES))
    prepare(parser.parse_args().dataset)
