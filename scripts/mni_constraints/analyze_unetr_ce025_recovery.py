"""Compare archived foreground/background behavior of matched base-loss recipes."""
import json
import csv
from pathlib import Path
from scripts.cross_dataset_campaign.common import read,write,digest
from scripts.mni_constraints import unetr_plain_ce025_study as study
from scripts.mni_constraints.analyze_unetr_constraint_pilot import evaluation
from scripts.unetr_subfields.cap_extension import lr_at


def details(folder):
    metrics=evaluation(folder)
    phases={}
    for phase,name in [('training','training_final.json'),('validation','validation_latest.json')]:
        result=read(folder/name)
        matrix=[[sum(c['confusion'][i][j] for c in result['cases']) for j in range(4)] for i in range(4)]
        phases[phase]=dict(macro_dice_percent=100*result['summary']['macro_hard_dice'],
            ca4_dice_percent=100*result['summary']['CA4_DG_hard_dice'],
            union_dice_percent=100*result['summary']['whole_hippocampus_hard_dice'],
            background_voxels=sum(matrix[0]),background_as_ca4=matrix[0][3],
            true_ca4=sum(matrix[3]),predicted_ca4=sum(r[3] for r in matrix),ca4_true_positive=matrix[3][3])
    return dict(archive=str(folder),manifest_sha256=digest(folder/'archive_manifest.json'),metrics=metrics,phases=phases)


def main():
    reports=study.ROOT/'reports';destination=reports/study.NAME;rows=[];pending=[]
    for seed,fold in [(5,0),(19,0)]:
        current=destination/'runs'/f'MNI_unetr_plain_ce025_dice_f{fold}s{seed}'
        if not (current/'cleanup.json').exists():pending.append(dict(seed=seed,fold=fold));continue
        prior=reports/'mni_unetr_plain_diceonly53t_20261109/runs'/f'MNI_unetr_plain_diceonly_dice_f{fold}s{seed}'
        old=reports/'mni_unetr_plain_subfields53t_20261109/runs'/f'MNI_unetr_plain_dice_f{fold}s{seed}'
        controls=[('ce0',prior),('ce025',current)]
        if (old/'cleanup.json').exists():controls.append(('ce1',old))
        current_cfg=read(current/'config.json');current_model=read(current/'model_recipe.json')
        measurements={}
        for label,folder in controls:
            cfg=read(folder/'config.json');model=read(folder/'model_recipe.json')
            for key in ('seed','fold','train_cases','validation_cases','initial_lr','lr_horizon_updates',
                        'dataset_sha256','script_sha256','objective_source_sha256','augmentation','torch','monai'):
                assert cfg[key]==current_cfg[key],(label,key)
            for key in ('model','declared','kwargs','actual_dropout_modules','parameters'):
                assert model[key]==current_model[key],(label,key)
            with (folder/'metrics.csv').open() as stream:curve=list(csv.DictReader(stream))
            assert curve
            for point in curve:
                expected=lr_at(int(point['step'])-1,cfg['lr_horizon_updates'],cfg['initial_lr'],'unetr_vit_b16','hold_half')
                assert abs(float(point['learning_rate'])-expected)<1e-12,(label,point['step'])
            measurements[label]=details(folder)
        for phase in ('training','validation'):
            assert len({m['phases'][phase]['true_ca4'] for m in measurements.values()})==1
            assert len({m['phases'][phase]['background_voxels'] for m in measurements.values()})==1
        rows.append(dict(seed=seed,fold=fold,recipes=measurements,
            validation_macro_change_vs_ce0=measurements['ce025']['phases']['validation']['macro_dice_percent']-measurements['ce0']['phases']['validation']['macro_dice_percent'],
            validation_background_ca4_fp_change_vs_ce0=measurements['ce025']['phases']['validation']['background_as_ca4']-measurements['ce0']['phases']['validation']['background_as_ca4']))
    output=dict(rows=rows,pending=pending,source_sha256=digest(Path(__file__)),
        scope='Final-budget matched unconstrained baselines only; original auxiliary definitions untouched. These observations assess loss-recipe robustness, not the separate constrained-minus-baseline nine-pair criterion.')
    write(destination/'base_loss_false_positive_recovery.json',output)
    print(json.dumps(dict(verified_pairs=len(rows),pending=pending,summary=[dict(seed=r['seed'],fold=r['fold'],validation_macro_change_vs_ce0=r['validation_macro_change_vs_ce0'],validation_background_ca4_fp_change_vs_ce0=r['validation_background_ca4_fp_change_vs_ce0']) for r in rows])))


if __name__=='__main__':main()
