"""Verify ADNI archives and compare every condition with its fresh baseline."""
from pathlib import Path

from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.cross_dataset_campaign.adni_mixed_losses_job import NAME, ROOT, verify


def report(root=None):
    root=Path(root) if root else ROOT/'reports'/NAME
    plan=verify(root/'frozen');rows=[];configs={};initializations={}
    for task in plan['tasks']:
        run=root/'runs'/task['id']
        if not (run/'archive_acknowledgment.json').exists():continue
        verify_archive(run,read(run/'archive_manifest.json'))
        folder=root/'frozen/protocols'/task['protocol'];p=read(folder/'protocol.json')
        config,done,recipe,calibration=(read(run/n) for n in ('config.json','completion.json','runtime_recipe.json','calibration.json'))
        native=task['model']=='nnunet'
        expected=dict(dataset='ADNI',model=task['model'],objective=task['arm'],weight_dice=1,
            weight_ce=int(native),dice_include_background=not native,native_deep_supervision=native,
            augmentation='none',target=task['target'],p95_cap=.75 if native else .5,
            calibration_base=task['baseline_loss'],primary_checkpoint='final_budget',
            ap_swaps_applicable=False,protocol_sha256=digest(folder/'protocol.json'))
        if any(recipe.get(k)!=v for k,v in expected.items()):raise ValueError('ADNI runtime recipe differs')
        split=read(folder/'splits_10patients.json')[0]
        if (config['augmentation']!='none' or config['smoke_test'] or done['discarded_attempts']
                or (config['fold'],config['seed'])!=(0,0) or config['train_cases']!=split['train']
                or config['validation_cases']!=split['val'] or config['dataset_sha256']!=p['pkl_sha256']
                or done['updates']!=config['max_steps'] or done['consumed_training_flops']>p['budget_flops']
                or config['schedule']!=p['models'][p['trainer_alias']]['objectives'][task['arm']]):
            raise ValueError('ADNI run is not a complete matched scientific result')
        if task['arm']!='dice' and (calibration['target_ratio']!=task['target']
                or calibration['p95_cap']!=expected['p95_cap'] or calibration['base_gradient']!=task['baseline_loss']):
            raise ValueError('ADNI calibration recipe differs')
        validation=read(run/'validation_latest.json');cleanup=read(run/'cleanup.json') if (run/'cleanup.json').exists() else {}
        rows.append(dict(**task,validation_dice_percent=100*validation['summary']['macro_hard_dice'],
            training_dice_percent=100*read(run/'training_final.json')['summary']['macro_hard_dice'],
            best_validation_dice_percent=100*done['best_validation_hard_dice'],validation_summary=validation['summary'],
            errors={k:sum(c[k] for c in validation['cases']) for k in ('foreground_fn','foreground_fp')},
            updates=done['updates'],consumed_training_flops=done['consumed_training_flops'],calibration=calibration,
            cleanup_status=cleanup.get('status','pending'),released_bytes=cleanup.get('released_file_bytes',0),
            recorded_seconds=done['seconds_this_invocation']))
        configs[task['id']]=config
        initializations[task['id']]=read(run/'initialization.json')['model_state_sha256']
    baselines={r['model']:r for r in rows if r['arm']=='dice'}
    for row in rows:
        baseline=baselines.get(row['model']);row['gain_over_baseline_percentage_points']=None
        if baseline:
            a,b=configs[row['id']],configs[baseline['id']]
            for key in ('model_identity','dataset_sha256','train_cases','validation_cases','initial_lr','lr_horizon_updates','precision','torch','monai'):
                if a[key]!=b[key]:raise ValueError('Matched ADNI baseline differs: '+key)
            if initializations[row['id']]!=initializations[baseline['id']]:raise ValueError('ADNI initializations differ')
            row['gain_over_baseline_percentage_points']=row['validation_dice_percent']-baseline['validation_dice_percent']
    result=dict(status='complete' if len(rows)==56 and all(r['cleanup_status']=='complete' for r in rows) else 'partial',
        completed=len(rows),total=56,dataset='ADNI',fold=0,seed=0,augmentation='none',requested_fraction=.05,
        training_subjects=5,train_crops=10,validation_subjects=27,validation_crops=54,
        budget_flops=plan['budget_flops_by_dataset']['ADNI'],primary_checkpoint='final_budget',
        ap_swaps_applicable=False,ap_swaps_reason='ADNI labels contain one hippocampus foreground class',
        baseline_scores={m:r['validation_dice_percent'] for m,r in baselines.items()},runs=rows,
        released_bytes=sum(r['released_bytes'] for r in rows),
        interpretation='Single fold/seed exploratory comparison. Use fresh matched per-model baselines; loss recipes differ across models, so architecture-only rankings are not identified.')
    write(root/'comparison.json',result)
    return result


if __name__=='__main__':
    result=report();print({k:result[k] for k in ('status','completed','total','baseline_scores','released_bytes')})
