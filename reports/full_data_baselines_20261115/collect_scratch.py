"""Acknowledge one verified MNI scratch archive and release the run's bulky files."""
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from hashlib import sha256

D=Path(__file__).resolve().parent
sys.path.insert(0,str(D/"mni_adapter"))
from collect import HOST, STAGE, REPO, remote_python, read, write, digest, verify_archive


def collect(key):
    rows=read(D/"manifest.json")
    r=rows[key]
    assert r["dataset"]=="MNI" and r["model"]!="medsam3"
    out=STAGE/"runs"/key
    dest=D/"archive"/key
    dest.mkdir(parents=True,exist_ok=True)
    subprocess.run(["rsync","-rt","--exclude=*.pt","--exclude=*.tmp",f"{HOST}:{out}/",str(dest)+"/"],check=True)
    proof=read(dest/"archive_manifest.json")
    verify_archive(dest,proof)
    assert proof["task"]["id"]==key
    with zipfile.ZipFile(D/"MNI_scratch_protocols.zip") as z:
        inputs=json.loads(z.read("checksums.json"))
        for name,expected in inputs.items():
            if name.startswith("source/"):
                content=(z.read(name) if name in z.namelist() else
                         (REPO/"reports/hfh_colab_exploration_20261111/bundle"/name).read_bytes())
                assert sha256(content).hexdigest()==expected
        config=read(dest/"config.json")
        assert config["protocol_sha256"]==sha256(z.read(f"protocols/{key}/protocol.json")).hexdigest()
    s=read(D/"splits.json")[r["split"]]
    assert config["train_cases"]==s["train"] and config["validation_cases"]==s["validation"]
    done=read(dest/"completion.json")
    cov=read(dest/"coverage.json")
    assert done["updates"]==r["target_updates"] and done["discarded_attempts"]==0
    ack=dict(status="verified",source_verified=True,durable_archive=str(dest),manifest_sha256=digest(dest/"archive_manifest.json"),
             source_archive=str(D/"MNI_scratch_protocols.zip"),source_archive_sha256=digest(D/"MNI_scratch_protocols.zip"))
    write(dest/"archive_acknowledgment.json",ack)
    remote_python("""
import json,sys
from pathlib import Path
from scripts.cross_dataset_campaign.common import write, digest
from scripts.cross_dataset_campaign.storage import retire
out=Path(sys.argv[1]);ack=json.loads(sys.argv[2]);assert digest(out/'archive_manifest.json')==ack['manifest_sha256']
write(out/'archive_acknowledgment.json',ack);retire(out)
""",out,json.dumps(ack))
    subprocess.run(["rsync","-rt",f"{HOST}:{out}/cleanup.json",str(dest)+"/"],check=True)
    assert read(dest/"cleanup.json")["status"]=="complete"
    rows=read(D/"manifest.json")
    rows[key].update(status="verified_complete",job_or_runtime="slurm:"+str(read(dest/"allocation.json")["job"]),
        final_dice=read(dest/"validation_latest.json")["summary"]["macro_hard_dice"],unique_crops_sampled=len(cov["sampled_crops"]),
        unique_subjects_sampled=len({c.rsplit('_',1)[0] for c in cov["sampled_crops"]}),last_verified_update=done["updates"],
        archive=str(dest.relative_to(REPO)),wall_seconds=cov["elapsed_seconds"])
    write(D/"manifest.json",rows)
    subprocess.run([sys.executable,str(D/"comparisons.py")],check=True)


if __name__=="__main__":
    collect(sys.argv[1])
