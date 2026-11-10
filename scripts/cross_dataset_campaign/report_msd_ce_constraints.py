"""Report verified target60 MSD results relative to fresh per-model baselines."""
import argparse
from pathlib import Path

from scripts.cross_dataset_campaign.common import read, write
from scripts.cross_dataset_campaign.storage import verify_archive


def report(root):
    plan=read(root/'frozen/plan.json')
    rows=[]
    for task in plan['tasks']:
        run=root/'runs'/task['id']
        if not (run/'archive_acknowledgment.json').exists():
            continue
        verify_archive(run,read(run/'archive_manifest.json'))
        done,config=read(run/'completion.json'),read(run/'config.json')
        if config['augmentation']!='none' or config['smoke_test'] or done['discarded_attempts']:
            raise ValueError('Invalid run for comparison')
        validation=read(run/'validation_latest.json')
        calibration=read(run/'calibration.json')
        errors={name:sum(c[name] for c in validation['cases']) for name in ('foreground_fn','foreground_fp')}
        errors['ap_swaps']={name:sum(c['ap_swaps'][name] for c in validation['cases'])
            for name in ('anterior_to_posterior','posterior_to_anterior')}
        cleanup=read(run/'cleanup.json') if (run/'cleanup.json').exists() else {}
        rows.append(dict(**task,validation_dice_percent=100*validation['summary']['macro_hard_dice'],
            training_dice_percent=100*read(run/'training_final.json')['summary']['macro_hard_dice'],
            validation_summary=validation['summary'],errors=errors,updates=done['updates'],
            best_validation_dice_percent=100*done['best_validation_hard_dice'],
            consumed_training_flops=done['consumed_training_flops'],
            calibration=calibration,cleanup_status=cleanup.get('status','pending'),
            released_bytes=cleanup.get('released_file_bytes',0)))
    baselines={r['model']:r['validation_dice_percent'] for r in rows if r['arm']=='dice'}
    for row in rows:
        row['gain_over_baseline_percentage_points']=(row['validation_dice_percent']-baselines[row['model']]
            if row['model'] in baselines else None)
    result=dict(status='complete' if len(rows)==len(plan['tasks']) and all(r['cleanup_status']=='complete' for r in rows) else 'partial',
        completed=len(rows),total=len(plan['tasks']),dataset='MSD',fold=0,seed=0,augmentation='none',
        train_crops=10,validation_crops=52,baseline='Established per-model Dice+CE',
        target=.6,p95_cap=.75,budget_flops=plan['budget_flops_by_dataset']['MSD'],
        primary_checkpoint='final_budget',baseline_scores=baselines,runs=rows,
        released_bytes=sum(r['released_bytes'] for r in rows),
        interpretation='Single fold/seed evidence. Compare constraints with their matched Dice+CE baseline; historical augmented or Dice-only runs do not isolate the calibration target.')
    write(root/'comparison.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    result=report(parser.parse_args().root)
    print({k:result[k] for k in ('status','completed','total','baseline_scores','released_bytes')})
