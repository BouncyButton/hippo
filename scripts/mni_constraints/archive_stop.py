"""Hash-verify scheduler and control records for the cancelled MNI study."""
import base64
import hashlib
import json
from scripts.launch_unetr_subfields import remote
from scripts.cross_dataset_campaign.common import write,REMOTE
from scripts.mni_constraints.campaign import ROOT,NAME


def main():
    code='''import json,sys,hashlib,base64
from pathlib import Path
p=Path(sys.argv[1]);assert json.loads((p/'control.json').read_text())['pause']
files=[p/n for n in ('status.json','control.json','failure.json','submission.json','successor.json')]
files+=list(p.glob('slurm-*'))+list((p/'submissions').glob('*.json'))
result={}
for f in files:
 if f.exists():
  assert f.is_file() and not f.is_symlink()
  b=f.read_bytes();result[str(f.relative_to(p))]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),data=base64.b64encode(b).decode())
print(json.dumps(result))'''
    rows=json.loads(remote(['python3','-c',code,REMOTE/NAME]))
    dest=ROOT/'reports'/NAME/'job_logs';manifest={}
    for name,item in rows.items():
        data=base64.b64decode(item['data'])
        assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
        p=dest/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
        manifest[name]={k:v for k,v in item.items() if k!='data'}
    write(dest/'manifest.json',dict(status='verified',files=manifest))
    print('Archived and verified',len(manifest),'scheduler/control records')


if __name__=='__main__':main()
