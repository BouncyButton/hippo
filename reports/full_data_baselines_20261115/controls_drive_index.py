"""Inventory every page of the campaign folder for off-runtime archive recovery."""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import time
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from continue_colab import FOLDER, existing, session


NAME = re.compile(r"(?:HFH|MSD|ADNI)_(?:(?:nnunet|swinunetr|basic_unet16|unetr_vit_b16)_f[0-3]_s\d+_(?:paper_matched|training_patch|lr_development|recovery_replicate)|medsam3_f[0-3]_s\d+(?:_coverage_control)?)_results\.zip")


def main():
    path = Path("/content/full_data_controls_inventory.json")
    guard = path.with_suffix(".lock").open("a")
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    drive = session()
    prior = existing(drive, path.name)
    old = json.loads(drive.files().get_media(fileId=prior["id"]).execute(num_retries=5)) if prior else {}
    while True:
        try:
            current = {}
            token = None
            while True:
                page = drive.files().list(q=f"'{FOLDER}' in parents and trashed=false",
                    fields="nextPageToken,incompleteSearch,files(id,name,sha256Checksum,size)",pageSize=1000,
                    pageToken=token).execute(num_retries=5)
                assert not page.get("incompleteSearch")
                for f in page["files"]:
                    if NAME.fullmatch(f["name"]):
                        assert f["name"] not in current and re.fullmatch("[a-f0-9]{64}",f["sha256Checksum"])
                        current[f["name"]] = dict(id=f["id"],sha256=f["sha256Checksum"],size=int(f["size"]))
                token = page.get("nextPageToken")
                if not token:
                    break
            assert all(current.get(k) == v for k,v in old.items()), "Published archive changed or disappeared"
            if current != old:
                path.write_text(json.dumps(current,sort_keys=True,indent=2)+"\n")
                media = MediaFileUpload(str(path),mimetype="application/json")
                if prior:
                    call = drive.files().update(fileId=prior["id"],media_body=media,fields="id,sha256Checksum")
                else:
                    call = drive.files().create(body=dict(name=path.name,parents=[FOLDER]),media_body=media,
                                                fields="id,sha256Checksum")
                prior = call.execute(num_retries=5)
                assert prior["sha256Checksum"] == hashlib.sha256(path.read_bytes()).hexdigest()
                old = current
                print("Published inventory",len(current),prior["id"],flush=True)
        except (HttpError,OSError) as exc:
            print("Inventory read retry",type(exc).__name__,flush=True)
        time.sleep(60)


if __name__ == "__main__":
    main()
