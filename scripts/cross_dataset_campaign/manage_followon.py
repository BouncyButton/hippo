"""Status, durable collectors, and graceful pause/resume for the combined chain."""
import argparse
import shlex
import subprocess

from scripts.cross_dataset_campaign.common import REMOTE, read, write
from scripts.cross_dataset_campaign.prepare import ROOT
from scripts.cross_dataset_campaign.prepare_followon import NAME, MODULE, PREDECESSOR
from scripts.cross_dataset_campaign.manage_msd_ce_constraints import collect
from scripts.cross_dataset_campaign.storage import verify_archive


def report():
    root=ROOT/'reports'/NAME;plan=read(root/'frozen/plan.json');rows=[]
    for task in plan['tasks']:
        run=root/'runs'/task['id']
        if not (run/'archive_acknowledgment.json').exists():continue
        verify_archive(run,read(run/'archive_manifest.json'));done=read(run/'completion.json')
        if task['kind']=='foundation':
            result=read(run/'evaluation.json');validation=[r for r in result['records'] if r['split']=='validation']
            val=result['mean_dice']['validation'];train=result['mean_dice']['training_support']
        else:
            result=read(run/'validation_latest.json');validation=result['cases']
            val=result['summary']['macro_hard_dice'];train=read(run/'training_final.json')['summary']['macro_hard_dice']
            config=read(run/'config.json')
            if config['augmentation']!='none' or config['smoke_test'] or done['discarded_attempts']:
                raise ValueError('Invalid comparison run')
        cleanup=read(run/'cleanup.json') if (run/'cleanup.json').exists() else {}
        rows.append(dict(**task,validation_dice_percent=100*val,training_dice_percent=100*train,
            updates=done['updates'],errors={k:sum(r[k] for r in validation) for k in ('foreground_fn','foreground_fp')},
            calibration=read(run/'calibration.json'),cleanup_status=cleanup.get('status','pending'),
            released_bytes=cleanup.get('released_file_bytes',0)))
    baselines={r['dataset']+'/'+r['model']:r['validation_dice_percent'] for r in rows if r['arm']=='dice'}
    for row in rows:
        key=row['dataset']+'/'+row['model'];base=baselines.get(key)
        row['gain_over_baseline_percentage_points']=None if base is None else row['validation_dice_percent']-base
    result=dict(status='complete' if len(rows)==126 and all(r['cleanup_status']=='complete' for r in rows) else 'partial',
        completed=len(rows),total=126,baseline_scores=baselines,runs=rows,
        released_bytes=sum(r['released_bytes'] for r in rows),
        interpretation='Single fold/seed, final checkpoint primary; 60% is a sequential per-auxiliary target, subject to the 75% p95 ceiling.')
    write(root/'comparison.json',result);return {k:result[k] for k in ('status','completed','total','baseline_scores','released_bytes')}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['status','pause','resume','collect','report'])
    command=parser.parse_args().command
    if command=='collect':
        for name in (PREDECESSOR,NAME):print(name,collect(name))
    elif command=='report':print(report())
    else:
        stage=REMOTE/NAME;env=':'.join(str(stage/p) for p in ('source','source/scripts','source/baselines/nnUNet'))
        args=['env','PYTHONPATH='+env,'python3','-m',MODULE,command,'--stage',str(stage)]
        subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','bocconi-cluster',shlex.join(args)],check=True)
        if command=='resume':
            for name in (PREDECESSOR,NAME):print(name,collect(name))


if __name__=='__main__':main()
