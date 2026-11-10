"""Deploy one explicitly prepared MNI batch; preserve two-job account limit."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from scripts.launch_unetr_subfields import remote
from scripts.cross_dataset_campaign.common import read,write,REMOTE
from scripts.unetr_subfields.cap_extension import ROOT,verify


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True)
    group=parser.add_mutually_exclusive_group();group.add_argument('--stage-only',action='store_true');group.add_argument('--submit-only',action='store_true')
    args=parser.parse_args()
    stage=ROOT/'experiments'/args.name;dest=REMOTE/args.name;report=ROOT/'reports'/args.name
    plan=verify(stage);report.mkdir(parents=True,exist_ok=True)
    if (report/'submission.json').exists():raise ValueError('Already submitted')
    rows=remote(['squeue','-h','-u','3160552','-o','%i|%T']).strip().splitlines()
    if not args.stage_only and (len(rows)>1 or any(r.split('|')[1] not in ('RUNNING','COMPLETING') for r in rows)):raise ValueError('No free queue slot')
    parent=REMOTE/'mni_matched_lr_dice_ce_20261106'
    code='''import shutil,sys
from pathlib import Path
p,parent=map(Path,sys.argv[1:3]);assert not (p/'submission.json').exists()
p.mkdir(parents=True,exist_ok=True)
for part in ('source','data'):shutil.copytree(parent/part,p/part,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))'''
    if not args.submit_only:
        remote(['python3','-c',code,dest,parent])
        subprocess.run(['rsync','-az','--exclude=__pycache__',str(stage)+'/',f'bocconi-cluster:{dest}/'],check=True)
    env=':'.join(str(dest/p) for p in ('source','source/scripts','source/baselines/nnUNet'))
    command=['env','PYTHONPATH='+env,plan['conventional_python'],'-m','scripts.unetr_subfields.cap_extension']
    if not args.submit_only:print(remote(command+['preflight','--stage',dest],timeout=600),flush=True)
    write(report/'preflight.json',json.loads(remote(['cat',dest/'preflight.json'])))
    if args.stage_only:
        print('Staged and verified; no job submitted.',flush=True);return
    log=(report/'collector.log').open('a')
    process=subprocess.Popen(['caffeinate','-i',str(ROOT/'.venv/bin/python'),'-u','-m','scripts.cross_dataset_campaign.collect',
        '--stage',str(dest),'--local-stage',str(stage),'--destination',str(report)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    log.close();write(report/'collector_process.json',dict(pid=process.pid,time=time.time()))
    receipt=json.loads(remote(command+['submit','--stage',dest]));write(report/'submission.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
