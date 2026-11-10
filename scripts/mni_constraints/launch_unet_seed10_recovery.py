"""Deploy the verified seed10-first block and its finite archive collector."""
import json
import subprocess
import time
from scripts.launch_unetr_subfields import remote
from scripts.cross_dataset_campaign.common import read, write, REMOTE
from scripts.mni_constraints.unet_seed10_recovery import ROOT, NAME, MODULE, PREVIOUS, verify
from scripts.mni_constraints.stage_seed5_remote import assemble


def main():
    stage=ROOT/'experiments'/NAME; dest=REMOTE/NAME; report=ROOT/'reports'/NAME
    plan=verify(stage); report.mkdir(parents=True,exist_ok=True)
    assert not (report/'submission.json').exists(), 'Already submitted'
    env=':'.join(str(REMOTE/PREVIOUS/p) for p in ('source','source/scripts','source/baselines/nnUNet'))
    code='''import json,subprocess,sys
from pathlib import Path
from scripts.cross_dataset_campaign.storage import quota
assert json.loads((Path(sys.argv[1])/'control.json').read_text())['pause']
assert not subprocess.check_output(['squeue','-h','-u','3160552'],text=True).strip()
usage=quota();assert usage['free_gib']>=16
print(json.dumps(usage))'''
    write(report/'initial_quota.json',json.loads(remote(['env','PYTHONPATH='+env,'python3','-c',code,REMOTE/PREVIOUS])))
    remote(['python3','-c','import sys;from pathlib import Path;p=Path(sys.argv[1]);assert not (p/"submission.json").exists();p.mkdir(exist_ok=True)',dest])
    print(json.dumps(assemble(stage,dest,report,read(stage/'deployment_known_donors.json'))),flush=True)
    env=':'.join(str(dest/p) for p in ('source','source/scripts','source/baselines/nnUNet'))
    command=['env','PYTHONPATH='+env,plan['conventional_python'],'-m',MODULE]
    print('Running zero-update preflight',flush=True)
    output=remote(command+['preflight','--stage',dest],timeout=900)
    write(report/'preflight.json',json.loads(output.strip().splitlines()[-1]))
    with (report/'collector.log').open('a') as log:
        process=subprocess.Popen(['caffeinate','-i',str(ROOT/'.venv/bin/python'),'-u','-m',
            'scripts.mni_constraints.collect_seed_recovery','--stage',str(dest),'--local-stage',str(stage),
            '--destination',str(report)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(report/'collector_process.json',dict(pid=process.pid,started_at=time.time()))
    receipt=json.loads(remote(command+['submit','--stage',dest]))
    write(report/'submission.json',receipt); print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
