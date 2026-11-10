"""One allocation runs the six cells sequentially, retaining verified results."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from scripts.depth_supervision.utils import atomic_json, digest
from .train import read, verify_completed

STOP = False


def stop(signum, frame):
    global STOP
    STOP = True


def check(stage, pkl):
    for name, sha in read(stage/'freeze.json').items():
        assert digest(stage/name)==sha, name
    for name, sha in read(stage/'source_manifest.json').items():
        assert digest(stage/'source'/name)==sha, name
    p=read(stage/'plan.json')
    assert digest(pkl)==p['pkl_sha256']
    assert len(p['split']['train'])==len(set(p['split']['train']))==10
    assert len(p['split']['val'])==len(set(p['split']['val']))==52
    assert not set(p['split']['train']) & set(p['split']['val'])
    assert p['max_epochs']==75 and p['early_stopping']==dict(min_epochs=25,patience=8,min_delta=.0005)
    assert p['augmentation']=='none' and len(p['runs'])==6
    return p


def report(root, rows, state, current=None):
    compact=[{k:r[k] for k in ['model','objective','fold','seed','best_dice','final_dice',
             'best_epoch','completed_epochs','optimizer_updates','training_seconds']} for r in rows]
    comparison=[]
    for model in ['nnunet','unetr_vit_b16','basic_unet16']:
        pair=[r for r in compact if r['model']==model]
        if len(pair)==2:
            baseline=next(r for r in pair if r['objective']=='dice')
            winner=next(r for r in pair if r['objective']!='dice')
            comparison.append(dict(model=model,baseline=baseline,constrained=winner,
                                   gain_pp=100*(winner['best_dice']-baseline['best_dice'])))
    atomic_json(root/'results.json',dict(state=state,completed=len(rows),planned=6,
        current=current,runs=compact,comparison=comparison,job_id=os.environ.get('SLURM_JOB_ID')))
    if compact:
        with (root/'results.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(compact[0]));writer.writeheader();writer.writerows(compact)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--pkl',type=Path,required=True)
    args=parser.parse_args()
    p=check(args.stage,args.pkl)
    import torch,monai
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert torch.__version__==p['runtime_versions']['torch'] and monai.__version__==p['runtime_versions']['monai']
    root=args.root;root.mkdir(parents=True,exist_ok=True)
    with (root/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        binding=read(args.stage/'freeze.json')
        if (root/'binding.json').exists():assert read(root/'binding.json')==binding
        else:atomic_json(root/'binding.json',binding)
        signal.signal(signal.SIGUSR1,stop)
        rows=[];started=time.monotonic()
        report(root,rows,'preflight')
        if not (root/'gpu_preflight.json').exists():
            from .tests import gpu_preflight
            gpu_preflight(p,args.pkl,root/'gpu_preflight.json')
        for entry in p['runs']:
            dest=root/'runs'/entry['id']
            if not (dest/'completion.json').exists():
                if STOP or time.monotonic()-started>23*3600:
                    report(root,rows,'paused_walltime',entry);return
                quota=subprocess.check_output(['beegfs','quota','list-usage','--uids','current','--pool','pool_home'],text=True)
                match=re.search(r'(\d+(?:\.\d+)?)GiB/(\d+(?:\.\d+)?)GiB',quota)
                if not match or float(match[2])-float(match[1])<6:
                    report(root,rows,'paused_storage',entry);return
                report(root,rows,'training',entry)
                try:
                    subprocess.run([sys.executable,'-m','scripts.epoch_winner75.train',
                        '--root',str(root),'--plan',str(args.stage/'plan.json'),
                        '--pkl',str(args.pkl),'--run-id',entry['id']],check=True)
                except BaseException as error:
                    atomic_json(root/'failure.json',dict(run=entry,error=repr(error)))
                    report(root,rows,'failed',entry);raise
            record=verify_completed(dest)
            assert 25<=record['completed_epochs']<=75
            if entry['objective']!='dice':
                baseline=next(r for r in rows if r['model']==entry['model'] and r['objective']=='dice')
                assert record['initialization_sha256']==baseline['initialization_sha256'], 'Paired initialization differs'
                assert record['warmup_sha256']==baseline['warmup_sha256'], 'Paired first five epochs differ'
            rows.append(record)
            report(root,rows,'running')
            # Both checkpoint files and durable metrics were verified above;
            # retire only these explicitly named completed-run artifacts.
            retired={n:record['artifacts'][n] for n in ['best.pt','latest.pt']}
            atomic_json(dest/'retired_checkpoints.json',retired)
            for name in retired:(dest/name).unlink(missing_ok=True)
            verify_completed(dest)
        report(root,rows,'complete')


if __name__=='__main__':
    main()
