"""Verify archived paired results and summarize the effect of adding CE."""
import argparse
import csv
from pathlib import Path

from scripts.cross_dataset_campaign.common import read, write
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.cross_dataset_campaign.probe_other_losses_job import MODELS, PROBE


def report(root):
    rows, pairs = [], []
    for identity in MODELS:
        completed = []
        for label, weight_ce in (('dice', 0), ('dice_ce', 1)):
            path = root/'runs'/f'MSD_{identity}_{label}_f0s0'
            if not (path/'archive_acknowledgment.json').exists():
                continue
            verify_archive(path, read(path/'archive_manifest.json'))
            config, done = read(path/'config.json'), read(path/'completion.json')
            validation, training = read(path/'validation_latest.json'), read(path/'training_final.json')
            recipe = read(path/'runtime_recipe.json')
            assert done['updates'] == config['max_steps'] and done['discarded_attempts'] == 0
            assert config['augmentation'] == 'none' and not config['smoke_test']
            assert recipe['weight_ce'] == weight_ce and recipe['dice_include_background']
            assert len(validation['cases']) == 52 and {r['case_name'] for r in validation['cases']} == set(config['validation_cases'])
            assert len(training['cases']) == 10 and {r['case_name'] for r in training['cases']} == set(config['train_cases'])
            errors = {k:sum(r[k] for r in validation['cases']) for k in ('foreground_fp', 'foreground_fn')}
            errors.update({k:sum(r['ap_swaps'][k] for r in validation['cases'])
                for k in ('anterior_to_posterior', 'posterior_to_anterior')})
            row = dict(model=identity, weight_ce=weight_ce, updates=done['updates'],
                validation_dice_percent=100*validation['summary']['macro_hard_dice'],
                training_dice_percent=100*training['summary']['macro_hard_dice'],
                anterior_dice_percent=100*validation['summary']['anterior_hard_dice'],
                posterior_dice_percent=100*validation['summary']['posterior_hard_dice'],
                voxel_errors=errors, consumed_training_flops=done['consumed_training_flops'],
                archive_verified=True, cleanup=read(path/'cleanup.json')['status'] if (path/'cleanup.json').exists() else 'pending')
            rows.append(row)
            with (path/'metrics.csv').open() as stream:
                curve = list(csv.DictReader(stream))
            completed.append((row, config, read(path/'initialization.json'), curve))
        if len(completed) == 2:
            (a, ca, ia, ma), (b, cb, ib, mb) = completed
            assert ia == ib
            for key in ('dataset_sha256', 'train_cases', 'validation_cases', 'torch', 'monai',
                        'lr_horizon_updates', 'precision', 'initial_lr', 'model_identity', 'max_steps'):
                assert ca[key] == cb[key], (identity, key)
            lr_a = {r['step']:r['learning_rate'] for r in ma}
            lr_b = {r['step']:r['learning_rate'] for r in mb}
            assert all(lr_a[k] == lr_b[k] for k in lr_a.keys() & lr_b.keys())
            pairs.append(dict(model=identity, initialization_sha256=ia['model_state_sha256'],
                dice_only_validation_percent=a['validation_dice_percent'],
                dice_ce_validation_percent=b['validation_dice_percent'],
                validation_gain_percentage_points=b['validation_dice_percent']-a['validation_dice_percent'],
                dice_only_training_percent=a['training_dice_percent'],
                dice_ce_training_percent=b['training_dice_percent'], updates_each=a['updates']))
    result = dict(status='complete' if len(pairs) == 3 else 'partial', dataset='MSD', fold=0, seed=0,
        augmentation='none', budget_flops_each=31976450667090, runs=rows, pairs=pairs,
        comparison='Original model-specific MONAI Dice and AdamW preserved; only CE coefficient toggled 0/1. Final budget checkpoints are primary.',
        scope='One matched seed/fold; no inference of cross-fold replication.', exploration='paused')
    write(root/'comparison.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('reports')/PROBE)
    args = parser.parse_args()
    import json
    print(json.dumps(report(args.root), indent=2))


if __name__ == '__main__':
    main()
