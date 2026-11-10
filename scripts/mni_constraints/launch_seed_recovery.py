"""Deploy the frozen recovery block, relocate abandoned runs, and submit once."""
import json
import subprocess
import time
from scripts.launch_unetr_subfields import remote
from scripts.cross_dataset_campaign.common import read,write,REMOTE,digest
from scripts.mni_constraints.seed_recovery import ROOT,NAME,MODULE,PARENT,RETIREMENTS,verify


def main():
    stage=ROOT/'experiments'/NAME;dest=REMOTE/NAME;report=ROOT/'reports'/NAME
    plan=verify(stage);report.mkdir(parents=True,exist_ok=True)
    assert not (report/'submission.json').exists(),'Already submitted'
    code='''import json,subprocess,sys
from pathlib import Path
from scripts.cross_dataset_campaign.storage import quota
assert not subprocess.check_output(['squeue','-h','-u','3160552'],text=True).strip()
assert json.loads((Path(sys.argv[1])/'control.json').read_text())['pause']
usage=quota();assert usage['free_gib']>=16
print(json.dumps(usage))'''
    parent_env=':'.join(str(REMOTE/PARENT/p) for p in ('source','source/scripts','source/baselines/nnUNet'))
    write(report/'initial_quota.json',json.loads(remote(['env','PYTHONPATH='+parent_env,'python3','-c',code,REMOTE/PARENT])))
    code='import sys;from pathlib import Path;p=Path(sys.argv[1]);assert not (p/"submission.json").exists();p.mkdir(exist_ok=True)'
    remote(['python3','-c',code,dest])
    subprocess.run(['rsync','-az','--exclude=__pycache__',str(stage)+'/',f'bocconi-cluster:{dest}/'],check=True)
    env=':'.join(str(dest/p) for p in ('source','source/scripts','source/baselines/nnUNet'))
    command=['env','PYTHONPATH='+env,plan['conventional_python'],'-m',MODULE]
    print('Running remote zero-update preflight',flush=True)
    output=remote(command+['preflight','--stage',dest],timeout=900)
    write(report/'preflight.json',json.loads(output.strip().splitlines()[-1]))
    # Atomic relocation on the same filesystem preserves bytes and avoids a bulky copy.
    # The old stage stays paused; its immutable source is also archived locally.
    code='''import json,sys,os,hashlib,time
from pathlib import Path
old,new=map(Path,sys.argv[1:3]);ids=json.loads(sys.argv[3]);rows=[]
assert json.loads((old/'control.json').read_text())['pause']
for identifier in ids:
 src=old/'runs'/identifier;dst=new/'runs'/identifier
 if not dst.exists():
  assert src.exists() and not (src/'completion.json').exists()
  before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in src.glob('*.pt')}
  dst.parent.mkdir(exist_ok=True);os.rename(src,dst)
  assert all(hashlib.sha256((dst/n).read_bytes()).hexdigest()==h for n,h in before.items())
  row=dict(original=str(src),relocated=str(dst),checkpoints=before,time=time.time())
  (dst/'relocation.json').write_text(json.dumps(row,indent=2))
 else:row=json.loads((dst/'relocation.json').read_text())
 rows.append(row)
(old/'interrupted_run_relocation.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows))'''
    write(report/'interrupted_run_relocation.json',json.loads(remote(['python3','-c',code,REMOTE/PARENT,dest,json.dumps(RETIREMENTS)],timeout=300)))
    with (report/'collector.log').open('a') as log:
        process=subprocess.Popen(['caffeinate','-i',str(ROOT/'.venv/bin/python'),'-u','-m',
            'scripts.mni_constraints.collect_seed_recovery','--stage',str(dest),'--local-stage',str(stage),
            '--destination',str(report)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(report/'collector_process.json',dict(pid=process.pid,started_at=time.time()))
    receipt=json.loads(remote(command+['submit','--stage',dest]))
    write(report/'submission.json',receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
