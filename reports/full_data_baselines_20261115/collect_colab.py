"""Reconcile Drive-verified Colab receipts with the frozen 180-cell manifest."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import zipfile

D=Path(__file__).resolve().parent


def merge(index):
    manifest_file=D/"manifest.json"
    splits=json.loads((D/"splits.json").read_text())
    checked={}
    start=json.loads(manifest_file.read_text())
    for key,proof in index.items():
        row=start[key]
        assert row["dataset"] in ("ADNI","MSD","HFH") and proof["updates"]==row["target_updates"]
        assert 0<=proof["final_dice"]<=1 and 0<proof["unique_crops_sampled"]<=row["eligible_crops"]
        current=dict(status="verified_complete",final_dice=proof["final_dice"],last_verified_update=proof["updates"],
            unique_crops_sampled=proof["unique_crops_sampled"],runtime_archive_sha256=proof["sha256"],
            remote_archive_status="second_account_verified_readback",drive_result_id=proof["id"],
            archive="https://drive.google.com/open?id="+proof["id"])
        path=D/"archive"/"colab"/(key+"_results.zip")
        if path.exists():
            assert hashlib.sha256(path.read_bytes()).hexdigest()==proof["sha256"]
            with zipfile.ZipFile(path) as archive:
                assert archive.testzip() is None
                done=json.loads(archive.read("completion.json"))
                split=splits[row["split"]]
                if row["model"]=="medsam3":
                    conf=json.loads(archive.read("protocol.json"))
                    assert conf["train_cases"]==split["train"] and conf["evaluation_cases"]==split["validation"]
                    assert math.isclose(done["validation_dice"],proof["final_dice"])
                else:
                    conf=json.loads(archive.read("config.json"))
                    assert sorted(conf["train_cases"])==sorted(split["train"])
                    assert sorted(conf["validation_cases"])==sorted(split["validation"])
                current["archive"]=str(path.relative_to(D.parents[1]))
        if row.get("runtime_archive_sha256") and row["runtime_archive_sha256"]!=proof["sha256"]:
            raise ValueError("Conflicting existing result: "+key)
        checked[key]=current
    fresh=json.loads(manifest_file.read_text())
    for key,change in checked.items():
        fresh[key].update(change)
    temp=manifest_file.with_name("manifest.colab.tmp")
    temp.write_text(json.dumps(fresh,indent=2,allow_nan=False)+"\n")
    os.replace(temp,manifest_file)
    subprocess.run([sys.executable,str(D/"comparisons.py")],check=True)
    return len(checked)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("index",type=Path)
    args=parser.parse_args()
    print("Reconciled",merge(json.loads(args.index.read_text())),"verified Colab cells")
