"""Maintain a single verified index of the full-data results saved by the Colab workers."""
import hashlib
import io
import json
import math
from pathlib import Path
import re
import time
import zipfile

from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from continue_colab import FOLDER, existing, session

PATTERN=re.compile(r"(MSD|ADNI|HFH)_(nnunet|basic_unet16|unetr_vit_b16|swinunetr|medsam3)_f[0-3]_s[0-9]+_results\.zip")


def row(drive, file):
    data=drive.files().get_media(fileId=file["id"]).execute(num_retries=5)
    assert int(file["size"])==len(data) and hashlib.sha256(data).hexdigest()==file["sha256Checksum"]
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert z.testzip() is None
        complete=json.loads(z.read("completion.json"))
        assert complete["status"]=="complete"
        if "_medsam3_" in file["name"]:
            p=json.loads(z.read("protocol.json"))
            evaluation=z.read("evaluation.json")
            assert hashlib.sha256(evaluation).hexdigest()==complete["evaluation_sha256"]
            records=json.loads(evaluation)["records"]
            groups={"training_support":p["train_cases"],"validation":p["evaluation_cases"]}
            for split,names in groups.items():
                assert sorted(x["case_name"] for x in records if x["split"]==split)==sorted(names)
            assert all("foreground_fp" in x and "foreground_fn" in x for x in records)
            trace=json.loads(z.read("slab_schedule.json"))
            assert len(trace)==p["updates"]*p["accumulation"]
            crops={x[0] for x in trace}
            assert crops<=set(p["train_cases"])
            dice=complete["validation_dice"]
        else:
            p=json.loads(z.read("config.json"))
            results=json.loads(z.read("validation_latest.json"))
            train=json.loads(z.read("training_final.json"))
            records=results["cases"]+train["cases"]
            assert sorted(x["case_name"] for x in results["cases"])==sorted(p["validation_cases"])
            assert sorted(x["case_name"] for x in train["cases"])==sorted(p["train_cases"])
            assert all("foreground_fp" in x and "foreground_fn" in x and "confusion" in x for x in records)
            assert complete["discarded_attempts"]==0
            crops=set(json.loads(z.read("coverage.json"))["sampled_crops"])
            dice=results["summary"]["macro_hard_dice"]
        assert math.isfinite(dice)
        return dict(id=file["id"],sha256=file["sha256Checksum"],bytes=len(data),updates=complete["updates"],
                    final_dice=dice,unique_crops_sampled=len(crops))


def main():
    out=Path("/content/full_data_colab_index.json")
    drive=session()
    previous=existing(drive,out.name)
    index=json.loads(drive.files().get_media(fileId=previous["id"]).execute()) if previous else {}
    while len(index)<135:
        try:
            files=[]
            token=None
            while True:
                page=drive.files().list(q=f"'{FOLDER}' in parents and trashed=false",pageSize=1000,pageToken=token,
                    fields="files(id,name,size,sha256Checksum),nextPageToken,incompleteSearch").execute(num_retries=5)
                assert not page.get("incompleteSearch")
                files.extend(f for f in page["files"] if PATTERN.fullmatch(f["name"]))
                token=page.get("nextPageToken")
                if not token:break
            assert len({f["name"] for f in files})==len(files)
            new=dict(index)
            for file in files:
                key=file["name"][:-len("_results.zip")]
                if key in index:
                    assert index[key]["id"]==file["id"] and index[key]["sha256"]==file["sha256Checksum"]
                else:new[key]=row(drive,file)
            if new!=index:
                out.write_text(json.dumps(new,indent=2,sort_keys=True)+"\n")
                media=MediaFileUpload(str(out),mimetype="application/json")
                prior=existing(drive,out.name)
                call=drive.files().update(fileId=prior["id"],media_body=media,fields="id") if prior else drive.files().create(body={"name":out.name,"parents":[FOLDER]},media_body=media,fields="id")
                fid=call.execute(num_retries=4)["id"]
                assert drive.files().get_media(fileId=fid).execute(num_retries=4)==out.read_bytes()
                index=new
                print("VERIFIED_INDEX",len(index),fid,flush=True)
        except HttpError as err:
            print("Drive retry",str(err)[:240],flush=True)
        time.sleep(45)


if __name__=="__main__":
    main()
