"""Manage the matched ADNI two-sample fold 1/3 replication and verified archives."""
import argparse
import ast
import shlex
import subprocess

from scripts.cross_dataset_campaign.common import REMOTE, read, write, digest
from scripts.cross_dataset_campaign.adni_foundation_folds_job import NAME, MODULE, ROOT, verify
from scripts.cross_dataset_campaign.manage_msd_ce_constraints import collect
from scripts.cross_dataset_campaign.storage import verify_archive


def remote(command):
    stage=REMOTE/NAME
    env=':'.join(str(stage/p) for p in ('source','source/scripts','source/baselines/nnUNet'))
    args=['env','PYTHONPATH='+env,'python3','-m',MODULE,command,'--stage',str(stage)]
    return subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','bocconi-cluster',shlex.join(args)],text=True)


def report():
    root=ROOT/'reports'/NAME;plan=verify(root/'frozen');rows=[];initializations={};schedules={}
    for task in plan['tasks']:
        run=root/'runs'/task['id']
        if not (run/'archive_acknowledgment.json').exists():continue
        manifest=read(run/'archive_manifest.json');verify_archive(run,manifest)
        ack=read(run/'archive_acknowledgment.json')
        if ack['manifest_sha256']!=digest(run/'archive_manifest.json') or ack['source_manifest_sha256']!=digest(root/'frozen/manifest.json'):
            raise ValueError('Archive provenance differs')
        done=read(run/'completion.json');p=read(run/'protocol.json')
        expected={**read(root/'frozen/protocols'/task['protocol']/'protocol.json'),'arm':task['arm']}
        if p!=expected or done['updates']!=200 or read(run/'runtime_validation.json')['status']!='passed':
            raise ValueError('Foundation result recipe differs')
        gpu=read(run/'gpu_worker.json')
        if gpu['status']!='passed' or gpu['visible_devices']!=1:
            raise ValueError('GPU worker binding was not verified')
        e=read(run/'evaluation.json');cal=read(run/'calibration.json')
        train=read(run/'training_summary.json')
        if train['frozen_before']!=train['frozen_after'] or train['adapter_before']==train['adapter_after']:
            raise ValueError('Frozen model or trained adapter check failed')
        if task['protocol'] in initializations and initializations[task['protocol']]!=train['adapter_before']:
            raise ValueError('Matched runs used different initial adapters')
        if task['protocol'] in schedules and schedules[task['protocol']]!=train['schedule_sha256']:
            raise ValueError('Matched runs used different samples')
        initializations[task['protocol']]=train['adapter_before'];schedules[task['protocol']]=train['schedule_sha256']
        if task['arm']!='dice' and (cal['target']!=.1 or cal['p95_cap']!=.5
                or any(v['weighted_p95']>.50000001 for v in cal['statistics'].values())):
            raise ValueError('Constraint calibration differs from original recipe')
        clean=read(run/'cleanup.json') if (run/'cleanup.json').exists() else {}
        errors={s:{k:sum(r[k] for r in e['records'] if r['split']==s) for k in ('foreground_fn','foreground_fp')}
            for s in ('training_support','validation')}
        rows.append(dict(**task,validation_dice_percent=100*done['validation_dice'],training_dice_percent=100*done['training_dice'],
            calibration=cal,errors=errors,cleanup_status=clean.get('status','pending'),released_bytes=clean.get('released_file_bytes',0)))
    for row in rows:
        baseline=next((x for x in rows if x['protocol']==row['protocol'] and x['arm']=='dice'),None)
        row['gain_over_baseline_percentage_points']=None if baseline is None else row['validation_dice_percent']-baseline['validation_dice_percent']
    complete=len(rows)==plan['num_runs'] and all(r['cleanup_status']=='complete' for r in rows)
    result=dict(status='complete' if complete else 'partial',completed=len(rows),total=plan['num_runs'],runs=rows,
        released_bytes=sum(r['released_bytes'] for r in rows),primary_checkpoint='final_update',augmentation='none',folds=[1,3],seed=0)
    write(root/'comparison.json',result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['queue','collect','status','report']);args=parser.parse_args()
    if args.command=='collect':print(collect(NAME))
    elif args.command=='report':
        r=report();print({k:r[k] for k in ('status','completed','total','released_bytes')})
    elif args.command=='status':print(remote('status'))
    else:
        receipt=ast.literal_eval(remote('submit').strip())
        for root in ('experiments','reports'):
            write(ROOT/root/NAME/'submission.json',receipt)
            write(ROOT/root/NAME/'submissions'/f'{receipt["job_id"]}.json',receipt)
        print(dict(submission=receipt,collector=collect(NAME)))


if __name__=='__main__':main()
