"""Separate LR main effects from constraint-gain changes on matching cells."""
import json
from pathlib import Path

from scripts.cross_dataset_campaign.common import read,write,digest
from scripts.mni_constraints import unetr_plain_cosine_study as study
from scripts.mni_constraints.analyze_unetr_constraint_pilot import evaluation


def main():
    report=study.ROOT/'reports'/study.NAME
    result=read(report/'plain_cosine_results.json');rows=[];missing=[]
    for current in result['rows']:
        arm,seed,fold=current['arm'],current['seed'],current['fold']
        prior_stage=('mni_unetr_plain_subfields53t_20261109' if arm=='dice_sdf_depth'
                     else 'mni_unetr_simple_recipe53t_20261108')
        old=study.ROOT/'reports'/prior_stage/'runs'/f'MNI_unetr_plain_{arm}_f{fold}s{seed}'
        if not (old/'cleanup.json').exists() or current['prior_plain_baseline'] is None:
            missing.append(dict(arm=arm,seed=seed,fold=fold));continue
        baseline=Path(current['prior_plain_baseline']['archive'])
        old_score,old_base=evaluation(old),evaluation(baseline)
        cfg,prior=read(report/'runs'/current['id']/'config.json'),read(old/'config.json')
        for key in ('seed','fold','train_cases','validation_cases','initial_lr','lr_horizon_updates',
                    'dataset_sha256','script_sha256','objective_source_sha256','augmentation','torch','monai'):
            assert cfg[key]==prior[key],(current['id'],key)
        assert read(old/'model_recipe.json')==read(report/'runs'/current['id']/'model_recipe.json')
        old_gain=old_score['validation_percent']-old_base['validation_percent']
        rows.append(dict(candidate=current['candidate'],seed=seed,fold=fold,
            previous_constraint_gain_pp=old_gain,new_constraint_gain_pp=current['gain_pp'],
            constraint_interaction_pp=current['gain_pp']-old_gain,
            baseline_lr_effect_pp=current['baseline']['validation_percent']-old_base['validation_percent'],
            constrained_lr_effect_pp=current['current']['validation_percent']-old_score['validation_percent'],
            previous_constraint_archive_sha256=digest(old/'archive_manifest.json'),
            previous_baseline_archive_sha256=digest(baseline/'archive_manifest.json')))
    output=dict(rows=rows,missing_previous_cells=missing,input_result_sha256=digest(report/'plain_cosine_results.json'),
        analysis_source_sha256=digest(Path(__file__)),
        scope='Same seed/fold,plain dropout0/noaug,DiceCE base,original losses and calibrated5%targets. LR policy differs; fresh training-only calibration follows each policy. No checkpoint selection or training changes.',
        caveat='Development comparisons; include only exact matched historical cells. A positive new constraint gain need not represent an improvement over the previous constraint gain or absolute Dice.')
    write(report/'lr_interactions.json',output);print(json.dumps(rows))


if __name__=='__main__':main()
