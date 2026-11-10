"""Compare quarter-CE cosine against matched original-CE cosine experiments."""
import json
from pathlib import Path

from scripts.cross_dataset_campaign.common import read,write,digest
from scripts.mni_constraints import unetr_ce025_cosine_study as study
from scripts.mni_constraints.analyze_unetr_constraint_pilot import evaluation


def main():
    report=study.ROOT/'reports'/study.NAME
    result=read(report/'ce025_cosine_results.json');rows=[];missing=[]
    for current in result['rows']:
        arm,seed,fold=current['arm'],current['seed'],current['fold']
        old_stage='mni_unetr_plain_cosine53t_20261109'
        old_report=study.ROOT/'reports'/old_stage
        old=old_report/'runs'/f'MNI_unetr_plain_cosine_{arm}_f{fold}s{seed}'
        baseline=old_report/'runs'/f'MNI_unetr_plain_cosine_dice_f{fold}s{seed}'
        if not (old/'cleanup.json').exists() or not (baseline/'cleanup.json').exists():
            missing.append(dict(arm=arm,seed=seed,fold=fold));continue
        old_score,old_base=evaluation(old),evaluation(baseline)
        folder=report/'runs'/current['id']
        cfg,prior=read(folder/'config.json'),read(old/'config.json')
        for key in ('seed','fold','train_cases','validation_cases','initial_lr','lr_horizon_updates',
                    'dataset_sha256','script_sha256','objective_source_sha256','augmentation','torch','monai'):
            assert cfg[key]==prior[key],(current['id'],key)
        model,old_model=read(folder/'model_recipe.json'),read(old/'model_recipe.json')
        for key in ('model','declared','kwargs','actual_dropout_modules','parameters'):
            assert model[key]==old_model[key],key
        protocol=report/'frozen/protocols'/current['protocol']/'protocol.json'
        previous=old_report/'frozen/protocols'/('plain_cosine_'+arm)/'protocol.json'
        assert cfg['protocol_sha256']==digest(protocol) and prior['protocol_sha256']==digest(previous)
        p,q=read(protocol),read(previous)
        for key in ('lr_policy','initial_lrs','lr_horizon_updates','lr_schedule_details',
                    'dropout_rate','augmentation','native_geometry','calibration_target_ratio','calibration_p95_cap'):
            assert p[key]==q[key],key
        assert p['loss_recipe']==study.runtime.LOSS_RECIPE
        assert read(folder/'base_loss_recipe.json')['dense_cross_entropy_in_base'] is True
        assert read(folder/'base_loss_recipe.json')['dense_cross_entropy_weight']==.25
        assert read(folder/'runtime_recipe.json')['objective_sha256']==read(old/'runtime_recipe.json')['objective_sha256']
        old_gain=old_score['validation_percent']-old_base['validation_percent']
        rows.append(dict(candidate=current['candidate'],seed=seed,fold=fold,
            previous_constraint_gain_pp=old_gain,new_constraint_gain_pp=current['gain_pp'],
            constraint_interaction_pp=current['gain_pp']-old_gain,
            baseline_loss_change_pp=current['baseline']['validation_percent']-old_base['validation_percent'],
            constrained_loss_change_pp=current['current']['validation_percent']-old_score['validation_percent'],
            previous_constraint_archive_sha256=digest(old/'archive_manifest.json'),
            previous_baseline_archive_sha256=digest(baseline/'archive_manifest.json')))
    output=dict(rows=rows,missing_previous_cells=missing,
        input_result_sha256=digest(report/'ce025_cosine_results.json'),analysis_source_sha256=digest(Path(__file__)),
        scope='Same architecture/dropout,seed/fold,data,post-warmup cosine LR and original auxiliaries. Dense CE reduced from1.0to0.25 in base; each loss recipe has fresh training-only5% calibration. Profiles account for loss arithmetic; both have409updates under samecap.',
        caveat='Development comparisons on overlapping folds, only available exact matched historical cells. Positive constraint interaction alone does not establish higher absolute Dice.')
    write(report/'cosine_base_loss_interactions.json',output);print(json.dumps(rows))


if __name__=='__main__':main()
