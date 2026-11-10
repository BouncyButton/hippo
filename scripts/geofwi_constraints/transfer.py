"""Transfer a verified frozen payload; never submits training."""
from pathlib import Path
import json
import shlex
import subprocess
import tarfile
from scripts.geofwi_constraints.prepare import ROOT, NAME, REMOTE, REPORT
from scripts.geofwi_cluster.campaign import read, write, sha

SSH=["ssh","-o","BatchMode=yes","-o","ConnectTimeout=20","bocconi-cluster"]
PYTHON="/home/3160552/.conda/envs/hippocampus/bin/python"
REMOTE_TIMEOUT=300


def remote(code, *args):
    return subprocess.check_output(SSH+[shlex.join([PYTHON,"-c",code,*map(str,args)])],text=True,timeout=REMOTE_TIMEOUT)


def main():
    preparation=read(REPORT/"preparation.json")
    payload=REPORT/"frozen_payload.tar.gz"
    assert sha(payload)==preparation["payload_sha256"]
    check="""from pathlib import Path
import json,sys
s=Path(sys.argv[1])
print(json.dumps(dict(exists=s.exists(),files=[p.name for p in s.iterdir()] if s.exists() else [])))
"""
    state=json.loads(remote(check,REMOTE))
    if state["exists"]:
        # Only a verified, never-submitted preparation may receive final edits.
        old_text=remote("import sys; print(open(sys.argv[1]).read())",REMOTE/"manifest.json")
        import hashlib
        old_digest=remote("import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())",REMOTE/"manifest.json").strip()
        if old_digest != preparation["manifest_sha256"]:
            assert read(REPORT/"transfer_receipt.json")["manifest_sha256"]==old_digest
            old=json.loads(old_text)
            current=read(ROOT/"experiments"/NAME/"manifest.json")
            changed=[name for name,value in current.items() if old.get(name)!=value]
            assert set(old)<=set(current), "Final patch must not remove frozen files"
            patch_path=REPORT/"final_transfer_patch.tar.gz"
            with tarfile.open(patch_path,"w:gz") as tar:
                for name in changed+["manifest.json"]:
                    tar.add(ROOT/"experiments"/NAME/name,arcname=name,recursive=False)
            code="""import hashlib,sys,tarfile
from pathlib import Path
s=Path(sys.argv[1])
assert not (s/'submission.json').exists() and not (s/'submissions').exists() and not (s/'runs').exists()
assert hashlib.sha256((s/'manifest.json').read_bytes()).hexdigest()==sys.argv[2]
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|gz') as tar:
    tar.extractall(s,filter='data')
print('final preparation patch installed')
"""
            with patch_path.open("rb") as stream:
                subprocess.run(SSH+[shlex.join([PYTHON,"-c",code,str(REMOTE),old_digest])],
                               stdin=stream,check=True,timeout=REMOTE_TIMEOUT)
        result=subprocess.check_output(SSH+[shlex.join([PYTHON,str(REMOTE/"campaign.py"),"verify","--stage",str(REMOTE)])],text=True,timeout=REMOTE_TIMEOUT)
    else:
        remote("from pathlib import Path; import sys; Path(sys.argv[1]).mkdir()",REMOTE)
        with payload.open("rb") as stream:
            subprocess.run(SSH+[shlex.join(["tar","-xz","-C",str(REMOTE)])],stdin=stream,check=True,timeout=REMOTE_TIMEOUT)
        data=REMOTE.parent/"geofwi3d_frozen_3fold_3seed_20261020/data"
        remote("from pathlib import Path; import sys; Path(sys.argv[1]).symlink_to(Path(sys.argv[2]),target_is_directory=True)",REMOTE/"data",data)
        result=subprocess.check_output(SSH+[shlex.join([PYTHON,str(REMOTE/"campaign.py"),"verify","--stage",str(REMOTE)])],text=True,timeout=REMOTE_TIMEOUT)
    digest=remote("import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())",REMOTE/"manifest.json").strip()
    assert digest==preparation["manifest_sha256"]
    receipt=dict(status="verified",remote_stage=str(REMOTE),manifest_sha256=digest,verification=json.loads(result),
                 laptop_collector=False)
    write(REPORT/"transfer_receipt.json",receipt)
    print(json.dumps(receipt))


if __name__=="__main__":
    main()
