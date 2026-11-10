"""Verify and compare the mixed-loss MSD sweep against its fresh baselines."""
from pathlib import Path

from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.cross_dataset_campaign.msd_mixed_losses_job import NAME, ROOT, verify


def report(root=None):
    root = Path(root) if root else ROOT/'reports'/NAME
    plan = verify(root/'frozen')
    rows, configs, initializations = [], {}, {}
    for task in plan['tasks']:
        run = root/'runs'/task['id']
        if not (run/'archive_acknowledgment.json').exists():
            continue
        verify_archive(run, read(run/'archive_manifest.json'))
        protocol_dir = root/'frozen/protocols'/task['protocol']
        p = read(protocol_dir/'protocol.json')
        config, done, recipe, calibration = (read(run/n) for n in
            ('config.json', 'completion.json', 'runtime_recipe.json', 'calibration.json'))
        native = task['model']=='nnunet'
        expected = dict(model=task['model'], objective=task['arm'], weight_dice=1,
            weight_ce=int(native), dice_include_background=not native,
            native_deep_supervision=native, augmentation='none', target=task['target'],
            p95_cap=.75 if native else .5, calibration_base=task['baseline_loss'],
            primary_checkpoint='final_budget', protocol_sha256=digest(protocol_dir/'protocol.json'))
        if any(recipe.get(k)!=v for k,v in expected.items()):
            raise ValueError('Runtime recipe differs: '+task['id'])
        split = read(protocol_dir/'splits_10patients.json')[0]
        if (config['augmentation']!='none' or config['smoke_test'] or done['discarded_attempts']
                or config['fold']!=0 or config['seed']!=0
                or config['train_cases']!=split['train'] or config['validation_cases']!=split['val']
                or config['dataset_sha256']!=p['pkl_sha256']
                or done['updates']!=config['max_steps']
                or done['consumed_training_flops']>p['budget_flops']):
            raise ValueError('Unmatched or incomplete run: '+task['id'])
        schedule = p['models'][p['trainer_alias']]['objectives'][task['arm']]
        if config['schedule']!=schedule:
            raise ValueError('Actual FLOP schedule differs: '+task['id'])
        init = read(run/'initialization.json')['model_state_sha256']
        if init!=p['baseline_reference']['initialization_sha256']:
            raise ValueError('Initialization differs: '+task['id'])
        if task['arm']!='dice' and (calibration['target_ratio']!=task['target']
                or calibration['p95_cap']!=expected['p95_cap']
                or calibration['base_gradient']!=task['baseline_loss']):
            raise ValueError('Calibration recipe differs: '+task['id'])
        validation = read(run/'validation_latest.json')
        cleanup = read(run/'cleanup.json') if (run/'cleanup.json').exists() else {}
        errors = {k:sum(c[k] for c in validation['cases']) for k in ('foreground_fn','foreground_fp')}
        errors['ap_swaps'] = {k:sum(c['ap_swaps'][k] for c in validation['cases'])
            for k in ('anterior_to_posterior','posterior_to_anterior')}
        rows.append(dict(**task, validation_dice_percent=100*validation['summary']['macro_hard_dice'],
            training_dice_percent=100*read(run/'training_final.json')['summary']['macro_hard_dice'],
            validation_summary=validation['summary'], errors=errors, updates=done['updates'],
            best_validation_dice_percent=100*done['best_validation_hard_dice'],
            consumed_training_flops=done['consumed_training_flops'], calibration=calibration,
            cleanup_status=cleanup.get('status','pending'), released_bytes=cleanup.get('released_file_bytes',0),
            recorded_seconds=done['seconds_this_invocation']))
        configs[task['id']], initializations[task['id']] = config, init
    baselines = {r['model']:r for r in rows if r['arm']=='dice'}
    for row in rows:
        baseline = baselines.get(row['model'])
        row['gain_over_baseline_percentage_points'] = None
        if baseline:
            a,b = configs[row['id']],configs[baseline['id']]
            for key in ('model_identity','dataset_sha256','train_cases','validation_cases',
                    'initial_lr','lr_horizon_updates','precision','torch','monai'):
                if a[key]!=b[key]:raise ValueError('Baseline mismatch: '+key)
            if initializations[row['id']]!=initializations[baseline['id']]:
                raise ValueError('Baseline initialization differs')
            row['gain_over_baseline_percentage_points'] = row['validation_dice_percent']-baseline['validation_dice_percent']
    result = dict(status='complete' if len(rows)==56 and all(r['cleanup_status']=='complete' for r in rows) else 'partial',
        completed=len(rows),total=56,dataset='MSD',fold=0,seed=0,augmentation='none',
        train_crops=10,validation_crops=52,budget_flops=plan['budget_flops_by_dataset']['MSD'],
        primary_checkpoint='final_budget', baseline_scores={m:r['validation_dice_percent'] for m,r in baselines.items()},
        model_recipes={m:dict(baseline_loss='Dice+CE' if m=='nnunet' else 'Dice',
            target=.6 if m=='nnunet' else .1,p95_cap=.75 if m=='nnunet' else .5)
            for m in ('nnunet','swinunetr','unetr_vit_b16','basic_unet16')},
        runs=rows,released_bytes=sum(r['released_bytes'] for r in rows),
        interpretation='Single fold/seed exploratory comparison. Constraint gains use fresh matched per-model baselines; baseline loss differs across models, so model rankings do not isolate architecture.')
    write(root/'comparison.json',result)
    return result


if __name__=='__main__':
    result=report()
    print({k:result[k] for k in ('status','completed','total','baseline_scores','released_bytes')})
