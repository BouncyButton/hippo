"""Verify quota, deploy a finite lower-cap pilot, and start its archive collector."""
import argparse
import json
import importlib
from pathlib import Path
import subprocess
import time

from scripts.launch_unetr_subfields import remote
from scripts.cross_dataset_campaign.common import read,write,REMOTE
from scripts.mni_constraints.budget_pilot import ROOT,MODULE,verify


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True)
    parser.add_argument('--module',choices=[MODULE,'scripts.mni_constraints.headroom_confirm','scripts.mni_constraints.lower_cap_pairs','scripts.mni_constraints.class_aware_pilot','scripts.mni_constraints.adaptive_pilot','scripts.mni_constraints.sample_size','scripts.mni_constraints.shared43_study'],default=MODULE)
    args=parser.parse_args()
    stage=ROOT/'experiments'/args.name;dest=REMOTE/args.name;report=ROOT/'reports'/args.name
    plan=importlib.import_module(args.module).verify(stage);report.mkdir(parents=True,exist_ok=True)
    assert not (report/'submission.json').exists(), 'Already submitted'
    parent=REMOTE/'mni_constraint_target_search_f0s0_20261107'
    parent_env=':'.join(str(parent/p) for p in ('source','source/scripts','source/baselines/nnUNet'))
    code='''import json,subprocess,sys
from pathlib import Path
from scripts.cross_dataset_campaign.storage import quota
q=subprocess.check_output(['squeue','-h','-u','3160552','-o','%i|%T|%j'],text=True)
rows=[r.split('|') for r in q.splitlines() if r.strip()]
assert len(rows)<=1 and all(r[1]=='RUNNING' and r[2]=='hippo-mni-budget' for r in rows),q
assert json.loads((Path(sys.argv[1])/'control.json').read_text())['pause']
usage=quota();assert usage['free_gib']>=float(sys.argv[2])
print(json.dumps(dict(queue=rows,quota=usage)))'''
    try:
        write(report/'access_status.json',json.loads(remote(['env','PYTHONPATH='+parent_env,'python3','-c',code,parent,str(plan['storage']['minimum_free_gib'])])))
        # Do not overwrite a previously submitted or differently frozen campaign.
        code='import sys;from pathlib import Path;p=Path(sys.argv[1]);assert not p.exists(),p;p.mkdir()'
        remote(['python3','-c',code,dest])
        subprocess.run(['rsync','-az','--exclude=__pycache__',str(stage)+'/',f'bocconi-cluster:{dest}/'],check=True)
        env=':'.join(str(dest/p) for p in ('source','source/scripts','source/baselines/nnUNet'))
        command=['env','PYTHONPATH='+env,plan['conventional_python'],'-m',args.module]
        preflight_output=remote(command+['preflight','--stage',dest],timeout=600)
        write(report/'preflight.json',json.loads(preflight_output.strip().splitlines()[-1]))
        collector_module=('scripts.mni_constraints.collect_shared43_study' if args.module=='scripts.mni_constraints.shared43_study' else 'scripts.mni_constraints.collect_sample_size' if args.module=='scripts.mni_constraints.sample_size' else 'scripts.mni_constraints.collect_adaptive' if args.module=='scripts.mni_constraints.adaptive_pilot' else 'scripts.mni_constraints.collect_class_aware' if args.module=='scripts.mni_constraints.class_aware_pilot'
                          else 'scripts.cross_dataset_campaign.collect')
        with (report/'collector.log').open('a') as log:
            process=subprocess.Popen(['caffeinate','-i',str(ROOT/'.venv/bin/python'),'-u','-m',
                collector_module,'--stage',str(dest),'--local-stage',str(stage),
                '--destination',str(report)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(report/'collector_process.json',dict(pid=process.pid,started_at=time.time()))
        receipt=json.loads(remote(command+['submit','--stage',dest]))
        write(report/'submission.json',receipt);print(json.dumps(receipt),flush=True)
    except subprocess.CalledProcessError as exc:
        print(exc.stderr,flush=True)
        raise


if __name__=='__main__':main()
