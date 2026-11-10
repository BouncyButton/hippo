"""Optional archive-only mirror using the verified cluster address and host key."""
import json
import shlex
import subprocess
from scripts.cross_dataset_campaign import collect
from scripts.mni_medsam3_cluster.constraint_mirror import snapshot

SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',
     '-o','HostName=10.35.5.3','-o','HostKeyAlias=slogin.hpc.unibocconi.it',
     '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3',
     '-o','ControlMaster=auto','-o','ControlPersist=600',
     '-o','ControlPath=/tmp/hippo-mni-matrix-%C','bocconi-cluster']


def remote(arguments):
    return subprocess.check_output([*SSH,shlex.join([str(v) for v in arguments])],timeout=180)


def upload_json(value,path):
    code='import sys,os; from pathlib import Path; p=Path(sys.argv[1]); t=p.with_name(p.name+".tmp"); t.write_text(sys.stdin.read()); os.replace(t,p)'
    subprocess.run([*SSH,shlex.join(['python3','-c',code,str(path)])],input=json.dumps(value).encode(),check=True,timeout=180)


if __name__=='__main__':
    collect.remote=remote
    collect.upload_json=upload_json
    collect.snapshot=snapshot
    collect.resume_after_archive=lambda *args:None
    collect.main()
