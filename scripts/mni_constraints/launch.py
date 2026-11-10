"""Deploy the frozen MNI study, start its finite collector, submit one job."""
import argparse
import json
import subprocess
import time
from scripts.launch_unetr_subfields import remote
from scripts.cross_dataset_campaign.common import read,write,REMOTE
from scripts.mni_constraints.campaign import NAME,MODULE,ROOT,verify


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage-only',action='store_true');parser.add_argument('--submit-only',action='store_true')
    args=parser.parse_args();assert not(args.stage_only and args.submit_only)
    stage=ROOT/'experiments'/NAME;dest=REMOTE/NAME;report=ROOT/'reports'/NAME
    plan=verify(stage);report.mkdir(parents=True,exist_ok=True)
    if (report/'submission.json').exists():raise ValueError('Already submitted')
    rows=remote(['squeue','-h','-u','3160552','-o','%i|%T']).strip().splitlines()
    if not args.stage_only and (len(rows)>1 or any(r.split('|')[1] not in ('RUNNING','COMPLETING') for r in rows)):
        raise ValueError('No queue slot')
    if not args.submit_only:
        remote(['mkdir','-p',dest])
        subprocess.run(['rsync','-az','--exclude=__pycache__',str(stage)+'/',f'bocconi-cluster:{dest}/'],check=True)
    env=':'.join(str(dest/p) for p in ('source','source/scripts','source/baselines/nnUNet'))
    command=['env','PYTHONPATH='+env,plan['conventional_python'],'-m',MODULE]
    if not args.submit_only:print(remote(command+['preflight','--stage',dest],timeout=900),flush=True)
    write(report/'preflight.json',json.loads(remote(['cat',dest/'preflight.json'])))
    if args.stage_only:
        print('Staged and verified; no job submitted.',flush=True);return
    with (report/'collector.log').open('a') as log:
        process=subprocess.Popen(['caffeinate','-i',str(ROOT/'.venv/bin/python'),'-u','-m','scripts.cross_dataset_campaign.collect',
            '--stage',str(dest),'--local-stage',str(stage),'--destination',str(report)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(report/'collector_process.json',dict(pid=process.pid,time=time.time()))
    receipt=json.loads(remote(command+['submit','--stage',dest]));write(report/'submission.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
