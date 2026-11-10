"""Avoid retransmitting frozen data/source and verified archives already on BeeGFS."""
import io
import json
import shlex
import subprocess
import tarfile
from scripts.cross_dataset_campaign.common import read,digest,REMOTE,write
from scripts.launch_unetr_subfields import remote
from scripts.mni_constraints.seed5_all_models import PARENT,RECOVERY


def assemble(stage,dest,report,extra_donors=None):
    manifest=read(stage/'manifest.json');parent=read(stage/'parent_manifest.json')
    donors={};upload=[]
    for name,sha in manifest.items():
        donor=(extra_donors or {}).get(name)
        if donor is not None:pass
        elif parent.get(name)==sha:donor=REMOTE/PARENT/name
        elif name=='parent_manifest.json':donor=REMOTE/PARENT/'manifest.json'
        elif name.startswith('protocols/'):
            if parent.get(name)==sha:donor=REMOTE/PARENT/name
        elif name.startswith('reused_runs/'):
            donor=REMOTE/RECOVERY/'runs'/name.removeprefix('reused_runs/')
        elif name.startswith('seed1_controls/'):
            donor=REMOTE/PARENT/'runs'/name.removeprefix('seed1_controls/')
        # Acknowledgments are pretty-printed locally but compact JSON remotely.
        # Preserve the exact durable-archive bytes bound by this stage manifest.
        if name.endswith('/archive_acknowledgment.json'):donor=None
        if donor is not None:donors[name]=str(donor)
        else:upload.append(name)
    blob=io.BytesIO()
    with tarfile.open(fileobj=blob,mode='w:gz') as tar:
        for name in [*upload,'manifest.json','control.json']:
            tar.add(stage/name,arcname=name,recursive=False)
        payload=json.dumps(donors).encode();info=tarfile.TarInfo('deployment_donors.json');info.size=len(payload)
        tar.addfile(info,io.BytesIO(payload))
    assert not json.loads(remote(['python3','-c','import json,sys;from pathlib import Path;print(json.dumps((Path(sys.argv[1])/"submission.json").exists()))',dest]))
    subprocess.run(['ssh','-o','BatchMode=yes','bocconi-cluster',shlex.join(['tar','-xzf','-','-C',str(dest)])],input=blob.getvalue(),check=True,timeout=180)
    code='''import json,sys,hashlib,shutil,os
from pathlib import Path
p=Path(sys.argv[1]);sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
assert not (p/'submission.json').exists()
assert sha(p/'manifest.json')==sys.argv[2]
manifest=json.loads((p/'manifest.json').read_text());donors=json.loads((p/'deployment_donors.json').read_text())
for name,source in donors.items():
 f=p/name;src=Path(source)
 assert sha(src)==manifest[name],str(src)
 if f.exists() and sha(f)==manifest[name]:continue
 f.parent.mkdir(parents=True,exist_ok=True);tmp=f.with_name(f.name+'.deploy_tmp');shutil.copy2(src,tmp);os.replace(tmp,f)
for name,value in manifest.items():assert sha(p/name)==value,name
print(json.dumps(dict(status='verified',files=len(manifest),copied_from_cluster=len(donors),manifest_sha256=sys.argv[2])))'''
    result=json.loads(remote(['python3','-c',code,dest,digest(stage/'manifest.json')],timeout=600))
    result['uploaded_compressed_bytes']=len(blob.getvalue());write(report/'deployment_verification.json',result)
    return result
