"""Deploy and submit the finite common-seed5 check after pausing seed search."""
import json
import subprocess
import time
from scripts.launch_unetr_subfields import remote
from scripts.cross_dataset_campaign.common import read,write,REMOTE
from scripts.mni_constraints.seed5_all_models import ROOT,NAME,MODULE,RECOVERY,verify


def main():
    stage=ROOT/'experiments'/NAME;dest=REMOTE/NAME;report=ROOT/'reports'/NAME
    plan=verify(stage);report.mkdir(parents=True,exist_ok=True)
    assert not (report/'submission.json').exists(),'Already submitted'
    parent_env=':'.join(str(REMOTE/RECOVERY/p) for p in ('source','source/scripts','source/baselines/nnUNet'))
    code='''import json,sys,subprocess
from pathlib import Path
from scripts.cross_dataset_campaign.storage import quota
p=Path(sys.argv[1]);assert json.loads((p/'control.json').read_text())['pause']
assert not subprocess.check_output(['squeue','-h','-u','3160552'],text=True).strip()
q=quota();assert q['free_gib']>=16;print(json.dumps(q))'''
    write(report/'initial_quota.json',json.loads(remote(['env','PYTHONPATH='+parent_env,'python3','-c',code,REMOTE/RECOVERY])))
    code='import sys;from pathlib import Path;p=Path(sys.argv[1]);assert not (p/"submission.json").exists();p.mkdir(exist_ok=True)'
    remote(['python3','-c',code,dest])
    from scripts.mni_constraints.stage_seed5_remote import assemble
    print(json.dumps(assemble(stage,dest,report)),flush=True)
    env=':'.join(str(dest/p) for p in ('source','source/scripts','source/baselines/nnUNet'))
    command=['env','PYTHONPATH='+env,plan['conventional_python'],'-m',MODULE]
    print('Running remote zero-update seed5 preflight',flush=True)
    output=remote(command+['preflight','--stage',dest],timeout=900)
    write(report/'preflight.json',json.loads(output.strip().splitlines()[-1]))
    with (report/'collector.log').open('a') as log:
        process=subprocess.Popen(['caffeinate','-i',str(ROOT/'.venv/bin/python'),'-u','-m',
            'scripts.mni_constraints.collect_seed5_all_models','--stage',str(dest),'--local-stage',str(stage),
            '--destination',str(report)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(report/'collector_process.json',dict(pid=process.pid,started_at=time.time()))
    receipt=json.loads(remote(command+['submit','--stage',dest]));write(report/'submission.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
