"""Deploy the fresh cap-matched MNI Dice+CE rerun."""
import json
import re
import subprocess
import time
from scripts.launch_unetr_subfields import remote
from scripts.cross_dataset_campaign.common import REMOTE,read,write,digest
from scripts.unetr_subfields.matched_lr_job import ROOT,NAME,PARENT,verify


def main():
    stage=ROOT/'experiments'/NAME;report=ROOT/'reports'/NAME;destination=REMOTE/NAME;parent=REMOTE/PARENT
    plan=verify(stage)
    if (report/'submission.json').exists():raise ValueError('Already submitted')
    if remote(['squeue','-h','-u','3160552','-o','%i']).strip():raise ValueError('Prior allocation/continuation must exit')
    state=json.loads(remote(['cat',parent/'status.json']))
    if state['status']!='paused' or not json.loads(remote(['cat',parent/'control.json']))['pause']:raise ValueError('Prior sweep must remain paused')
    write(report/'parent_status.json',state)
    quota=remote(['beegfs','quota','list-usage','--uids','current','--pool','pool_home'])
    match=re.search(r'(\d+(?:\.\d+)?)GiB/(\d+(?:\.\d+)?)GiB',quota)
    if not match or float(match[2])-float(match[1])<plan['storage']['minimum_free_gib']:raise ValueError('Insufficient quota headroom')
    write(report/'prelaunch.json',dict(quota=quota,storage=plan['storage'],time=time.time()))
    code='''import hashlib,shutil,sys
from pathlib import Path
parent,p=map(Path,sys.argv[1:3]);assert not (p/'submission.json').exists()
assert not (p/'manifest.json').exists() or hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest()==sys.argv[3]
p.mkdir(parents=True,exist_ok=True)
shutil.copytree(parent/'source',p/'source',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
(p/'data').mkdir(exist_ok=True);shutil.copy2(parent/'data/MNI.pkl',p/'data/MNI.pkl')'''
    remote(['python3','-c',code,parent,destination,digest(stage/'manifest.json')])
    subprocess.run(['rsync','-az','--exclude=__pycache__',str(stage)+'/',f'bocconi-cluster:{destination}/'],check=True)
    env=':'.join(str(destination/name) for name in ('source','source/scripts','source/baselines/nnUNet'))
    command=['env','PYTHONPATH='+env,plan['conventional_python'],'-m','scripts.unetr_subfields.matched_lr_job']
    print(remote(command+['preflight','--stage',destination],timeout=600),flush=True)
    python=ROOT/'.venv/bin/python'
    for label,args in [('collector',['-m','scripts.cross_dataset_campaign.collect','--stage',str(destination),'--local-stage',str(stage),'--destination',str(report)]),
                       ('summary',['-m','scripts.unetr_subfields.baseline_report','--report',str(report),'--watch'])]:
        log=(report/(label+'.log')).open('a')
        process=subprocess.Popen(['caffeinate','-i',str(python),'-u',*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        log.close();write(report/(label+'_process.json'),dict(pid=process.pid,started_at=time.time(),python=str(python)))
    receipt=json.loads(remote(command+['submit','--stage',destination]));write(report/'submission.json',receipt);print(json.dumps(receipt,indent=2),flush=True)

if __name__=='__main__':main()
