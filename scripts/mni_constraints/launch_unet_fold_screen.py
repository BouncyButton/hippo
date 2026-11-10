"""Deploy fold-first scheduling and relocate exact interrupted trajectories."""
import json
import subprocess
import time
from scripts.launch_unetr_subfields import remote
from scripts.cross_dataset_campaign.common import read,write,digest,REMOTE
from scripts.mni_constraints.unet_fold_screen import ROOT,NAME,MODULE,PREVIOUS,RECOVER,verify
from scripts.mni_constraints.stage_seed5_remote import assemble


def main():
    stage=ROOT/'experiments'/NAME;dest=REMOTE/NAME;report=ROOT/'reports'/NAME
    plan=verify(stage);report.mkdir(parents=True,exist_ok=True)
    assert not (report/'submission.json').exists(),'Already submitted'
    oldenv=':'.join(str(REMOTE/PREVIOUS/p) for p in ('source','source/scripts','source/baselines/nnUNet'))
    code='''import json,sys,subprocess
from pathlib import Path
from scripts.cross_dataset_campaign.storage import quota
assert json.loads((Path(sys.argv[1])/'control.json').read_text())['pause']
assert not subprocess.check_output(['squeue','-h','-u','3160552'],text=True).strip()
q=quota();assert q['free_gib']>=16;print(json.dumps(q))'''
    write(report/'initial_quota.json',json.loads(remote(['env','PYTHONPATH='+oldenv,'python3','-c',code,REMOTE/PREVIOUS])))
    remote(['python3','-c','import sys;from pathlib import Path;p=Path(sys.argv[1]);assert not (p/"submission.json").exists();p.mkdir(exist_ok=True)',dest])
    old=read(stage/'previous_manifest.json')
    donors={n:str(REMOTE/PREVIOUS/n) for n,h in read(stage/'manifest.json').items() if old.get(n)==h}
    donors['previous_manifest.json']=str(REMOTE/PREVIOUS/'manifest.json')
    print(json.dumps(assemble(stage,dest,report,donors)),flush=True)
    env=':'.join(str(dest/p) for p in ('source','source/scripts','source/baselines/nnUNet'))
    command=['env','PYTHONPATH='+env,plan['conventional_python'],'-m',MODULE]
    print('Running zero-update preflight',flush=True)
    output=remote(command+['preflight','--stage',dest],timeout=900)
    write(report/'preflight.json',json.loads(output.strip().splitlines()[-1]))
    code='''import json,sys,os,hashlib,time
from pathlib import Path
import torch
old,new=map(Path,sys.argv[1:3]);ids=json.loads(sys.argv[3]);rows=[]
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert read(old/'control.json')['pause']
assert sha(old/'protocols/basic_unet16_dice/protocol.json')==sha(new/'protocols/basic_unet16_dice/protocol.json')
for identifier in ids:
 src=old/'runs'/identifier;dst=new/'runs'/identifier
 if dst.exists():
  rows.append(read(dst/'relocation.json'));continue
 assert src.exists() and not (src/'completion.json').exists()
 cp=torch.load(src/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
 cfg=read(src/'config.json');ledger=read(src/'budget_ledger.json')
 assert cp['configuration']==cfg and cp['step']==cp['charged_attempts']==ledger['charged_attempts']==1
 assert ledger['config_sha256']==sha(src/'config.json')
 before={f.name:sha(f) for f in src.iterdir() if f.is_file()}
 dst.parent.mkdir(exist_ok=True);os.rename(src,dst)
 assert all(sha(dst/n)==h for n,h in before.items())
 row={'original':str(src),'relocated':str(dst),'hashes':before,'saved_updates':cp['step'],'replay_gap':0,'time':time.time()}
 (dst/'relocation.json').write_text(json.dumps(row,indent=2));rows.append(row)
(old/'seed11_relocation.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows))'''
    moved=json.loads(remote([plan['conventional_python'],'-c',code,REMOTE/PREVIOUS,dest,json.dumps(RECOVER)],timeout=600))
    write(report/'seed11_relocation.json',moved)
    write(ROOT/'reports'/PREVIOUS/'seed11_relocation.json',moved)
    with (report/'collector.log').open('a') as log:
        process=subprocess.Popen(['caffeinate','-i',str(ROOT/'.venv/bin/python'),'-u','-m',
            'scripts.mni_constraints.collect_seed_recovery','--stage',str(dest),'--local-stage',str(stage),
            '--destination',str(report)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(report/'collector_process.json',dict(pid=process.pid,started_at=time.time()))
    receipt=json.loads(remote(command+['submit','--stage',dest]));write(report/'submission.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
