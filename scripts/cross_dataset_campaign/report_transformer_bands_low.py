"""Verified low-target bands results against matched fresh MSD controls."""
import csv
from pathlib import Path

from scripts.cross_dataset_campaign.common import read, write
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.cross_dataset_campaign.transformer_bands_low_job import ROOT, NAME, PARENT


def report():
    root = ROOT/'reports'/NAME
    plan = read(root/'frozen/plan.json')
    rows = []
    for task in plan['tasks']:
        run = root/'runs'/task['id']
        if not (run/'archive_acknowledgment.json').exists():
            continue
        model = task['model']
        base = ROOT/'reports'/PARENT/'runs'/f'MSD_{model}_ce_dice_target60_f0s0'
        high = ROOT/'reports'/PARENT/'runs'/f'MSD_{model}_ce_dice_bands_target60_f0s0'
        for folder in (run, base, high):
            verify_archive(folder, read(folder/'archive_manifest.json'))
        done, config, calibration = (read(run/n) for n in ('completion.json','config.json','calibration.json'))
        old_config = read(high/'config.json')
        assert config['augmentation']=='none' and not config['smoke_test'] and not done['discarded_attempts']
        for key in ('model_identity','dataset_sha256','train_cases','validation_cases','max_steps',
                    'lr_horizon_updates','initial_lr','precision','torch','monai','schedule'):
            assert config[key]==old_config[key], key
        assert read(run/'initialization.json')==read(base/'initialization.json')==read(high/'initialization.json')
        assert done['updates']==old_config['max_steps']
        assert done['consumed_training_flops']<=config['budget_flops']
        high_calibration = read(high/'calibration.json')
        assert calibration['anchor_model_sha256']==high_calibration['anchor_model_sha256'], 'Different calibration anchor'
        assert calibration['target_ratio']==task['target'] and calibration['p95_cap']==.75
        # These targets should not hit the ceiling: same model at the same prefix.
        assert abs(calibration['weights']['bands']/high_calibration['weights']['bands']-task['target']/.6)<1e-6
        old_rows = {int(r['step']):r for r in csv.DictReader((high/'metrics.csv').open())}
        prefix = []
        for row in csv.DictReader((run/'metrics.csv').open()):
            step = int(row['step'])
            if step<=50:
                for key in ('mean_train_loss','learning_rate','macro_hard_dice','anterior_hard_dice','posterior_hard_dice'):
                    assert row[key]==old_rows[step][key], (step,key)
                prefix.append(step)
        val = read(run/'validation_latest.json')
        baseline_score = 100*read(base/'validation_latest.json')['summary']['macro_hard_dice']
        high_score = 100*read(high/'validation_latest.json')['summary']['macro_hard_dice']
        score = 100*val['summary']['macro_hard_dice']
        cleanup = read(run/'cleanup.json') if (run/'cleanup.json').exists() else {}
        rows.append(dict(**task,validation_dice_percent=score,baseline_dice_percent=baseline_score,
            bands60_dice_percent=high_score,gain_over_baseline_points=score-baseline_score,
            gain_over_bands60_points=score-high_score,
            training_dice_percent=100*read(run/'training_final.json')['summary']['macro_hard_dice'],
            foreground_fn=sum(c['foreground_fn'] for c in val['cases']),
            foreground_fp=sum(c['foreground_fp'] for c in val['cases']),
            ap_swaps={k:sum(c['ap_swaps'][k] for c in val['cases']) for k in ('anterior_to_posterior','posterior_to_anterior')},
            calibration=calibration,matching_calibration_anchor=True,matching_prefix_updates=prefix,
            cleanup_status=cleanup.get('status','pending'),released_bytes=cleanup.get('released_file_bytes',0)))
    result=dict(status='complete' if len(rows)==4 and all(r['cleanup_status']=='complete' for r in rows) else 'partial',
        completed=len(rows),total=4,runs=rows,augmentation='none',fold=0,seed=0,
        baseline='Same per-model Dice+CE',train_crops=10,validation_crops=52,
        budget_flops=plan['budget_flops_by_dataset']['MSD'],
        interpretation='Single fold/seed strength ablation; final FLOP-budget checkpoint is primary',
        released_bytes=sum(r['released_bytes'] for r in rows))
    write(root/'comparison.json',result)
    return result


if __name__=='__main__':
    import json
    print(json.dumps(report(),indent=2))
