"""Stage the targeted strength test; submit only after seed recovery completes."""
import argparse
import json
import subprocess
import time
from scripts.launch_unetr_subfields import remote
from scripts.cross_dataset_campaign.common import read,write,REMOTE
from scripts.mni_constraints.unetr_strength import ROOT,NAME,MODULE,verify


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['stage','submit']);args=parser.parse_args()
    stage=ROOT/'experiments'/NAME;dest=REMOTE/NAME;report=ROOT/'reports'/NAME
    plan=verify(stage);report.mkdir(parents=True,exist_ok=True)
    env=':'.join(str(dest/p) for p in ('source','source/scripts','source/baselines/nnUNet'))
    command=['env','PYTHONPATH='+env,plan['conventional_python'],'-m',MODULE]
    assert not (report/'submission.json').exists(),'Already submitted'
    if args.action=='stage':
        code='import sys;from pathlib import Path;p=Path(sys.argv[1]);assert not (p/"submission.json").exists();p.mkdir(exist_ok=True)'
        remote(['python3','-c',code,dest])
        subprocess.run(['rsync','-az','--exclude=__pycache__',str(stage)+'/',f'bocconi-cluster:{dest}/'],check=True)
        print('Running remote zero-update strength preflight',flush=True)
        output=remote(command+['preflight','--stage',dest],timeout=900)
        write(report/'preflight.json',json.loads(output.strip().splitlines()[-1]));print('Remote preflight passed',flush=True)
        return
    code='''import json,sys,subprocess
from pathlib import Path
from scripts.cross_dataset_campaign.storage import quota
p=Path(sys.argv[1]);assert json.loads((p/'selection.json').read_text())['complete']
assert json.loads((p/'status.json').read_text())['status']=='complete'
assert not subprocess.check_output(['squeue','-h','-u','3160552'],text=True).strip()
q=quota();assert q['free_gib']>=16;print(json.dumps(q))'''
    write(report/'initial_quota.json',json.loads(remote(['env','PYTHONPATH='+env,'python3','-c',code,plan['prerequisite_stage']])))
    with (report/'collector.log').open('a') as log:
        process=subprocess.Popen(['caffeinate','-i',str(ROOT/'.venv/bin/python'),'-u','-m',
            'scripts.mni_constraints.collect_seed_recovery','--stage',str(dest),'--local-stage',str(stage),
            '--destination',str(report)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(report/'collector_process.json',dict(pid=process.pid,started_at=time.time()))
    receipt=json.loads(remote(command+['submit','--stage',dest]));write(report/'submission.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
