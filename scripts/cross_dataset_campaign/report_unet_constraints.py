"""Verify archived 3D UNet constraints against their matched Dice+CE baseline."""
import argparse
from pathlib import Path

from scripts.cross_dataset_campaign.common import read, write
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.cross_dataset_campaign.probe_unet_constraints_job import PROBE


def report(root):
    baseline = root.parent/'msd_other_models_loss_pairs_20261014/runs/MSD_basic_unet16_dice_ce_f0s0'
    verify_archive(baseline,read(baseline/'archive_manifest.json'))
    base_config=read(baseline/'config.json')
    base_init=read(baseline/'initialization.json')
    base_validation=read(baseline/'validation_latest.json')
    score=100*base_validation['summary']['macro_hard_dice']
    rows=[]
    plan=read(root/'frozen/plan.json')
    for task in plan['tasks']:
        path=root/'runs'/task['id']
        if not (path/'archive_acknowledgment.json').exists():continue
        verify_archive(path,read(path/'archive_manifest.json'))
        config,done,recipe,calibration=[read(path/f'{name}.json') for name in ('config','completion','runtime_recipe','calibration')]
        validation,training=read(path/'validation_latest.json'),read(path/'training_final.json')
        for key in ('dataset_sha256','train_cases','validation_cases','torch','monai','lr_horizon_updates','initial_lr','model_identity','augmentation','precision','fold','seed'):
            if config[key]!=base_config[key]:raise ValueError('Baseline configuration differs: '+key)
        if read(path/'initialization.json')!=base_init:raise ValueError('Baseline initialization differs')
        if (recipe['weight_ce']!=1 or calibration['base_gradient']!='Dice+CE'
                or recipe['target']!=task['target'] or config['smoke_test']
                or done['updates']!=config['max_steps'] or done['discarded_attempts']):
            raise ValueError('Incorrect or incomplete constrained run')
        if {r['case_name'] for r in validation['cases']}!=set(base_config['validation_cases']):
            raise ValueError('Validation coverage differs')
        if {r['case_name'] for r in training['cases']}!=set(base_config['train_cases']):
            raise ValueError('Training coverage differs')
        if {r['case_name'] for r in calibration['cases']}!=set(base_config['train_cases']):
            raise ValueError('Calibration used an incorrect cohort')
        errors={k:sum(r[k] for r in validation['cases']) for k in ('foreground_fp','foreground_fn')}
        errors.update({k:sum(r['ap_swaps'][k] for r in validation['cases']) for k in ('anterior_to_posterior','posterior_to_anterior')})
        value=100*validation['summary']['macro_hard_dice']
        rows.append(dict(id=task['id'],objective=task['arm'],target=task['target'],
            updates=done['updates'],validation_dice_percent=value,gain_vs_dice_ce_baseline_pp=value-score,
            training_dice_percent=100*training['summary']['macro_hard_dice'],
            anterior_dice_percent=100*validation['summary']['anterior_hard_dice'],
            posterior_dice_percent=100*validation['summary']['posterior_hard_dice'],
            voxel_errors=errors,weights=calibration['weights'],calibration_statistics=calibration['statistics'],
            consumed_training_flops=done['consumed_training_flops'],
            archive_verified=True,cleanup=read(path/'cleanup.json')['status'] if (path/'cleanup.json').exists() else 'pending'))
    result=dict(status='complete' if len(rows)==len(plan['tasks']) else 'partial',dataset='MSD',model='basic_unet16',
        fold=0,seed=0,augmentation='none',baseline_dice_ce_percent=score,
        baseline_updates=base_config['max_steps'],budget_flops=base_config['budget_flops'],runs=rows,
        methodology='All constraints added to the identical Dice+CE baseline; final-budget checkpoints compared. Calibration uses training-only gradients, sequentially bands, edge, depth. Single fold/seed exploratory comparison.',
        other_models_and_exploration='held')
    write(root/'comparison.json',result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('reports')/PROBE)
    args=parser.parse_args()
    import json
    print(json.dumps(report(args.root),indent=2))


if __name__=='__main__':main()
