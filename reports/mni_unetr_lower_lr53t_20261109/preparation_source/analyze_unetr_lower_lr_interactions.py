"""Separate peak-LR effects on baseline Dice from changes in constraint gain."""
from pathlib import Path
import json

from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.mni_constraints import unetr_lower_lr_study as study
from scripts.mni_constraints.analyze_unetr_constraint_pilot import evaluation


def main():
    report = study.ROOT/'reports'/study.NAME
    prior_report = study.ROOT/'reports/mni_unetr_plain_cosine53t_20261109'
    result = read(report/'lower_lr_results.json')
    rows, missing = [], []
    for current in result['rows']:
        arm, seed, fold = current['arm'], current['seed'], current['fold']
        old = prior_report/'runs'/f'MNI_unetr_plain_cosine_{arm}_f{fold}s{seed}'
        baseline = prior_report/'runs'/f'MNI_unetr_plain_cosine_dice_f{fold}s{seed}'
        if not (old/'cleanup.json').exists() or not (baseline/'cleanup.json').exists():
            missing.append(dict(candidate=current['candidate'], seed=seed, fold=fold))
            continue
        old_score, old_base = evaluation(old), evaluation(baseline)
        folder = report/'runs'/current['id']
        current_score = evaluation(folder)
        cfg, prior = read(folder/'config.json'), read(old/'config.json')
        for key in ('seed', 'fold', 'objective', 'train_cases', 'validation_cases',
                    'lr_horizon_updates', 'schedule', 'dataset_sha256', 'script_sha256',
                    'objective_source_sha256', 'augmentation', 'torch', 'monai'):
            assert cfg[key] == prior[key], (current['id'], key)
        assert cfg['initial_lr'] == current['peak_lr'] and prior['initial_lr'] == .0006
        model, old_model = read(folder/'model_recipe.json'), read(old/'model_recipe.json')
        for key in ('model', 'declared', 'kwargs', 'actual_dropout_modules', 'parameters'):
            assert model[key] == old_model[key], key
        new_protocol = report/'frozen/protocols'/current['protocol']/'protocol.json'
        old_protocol = prior_report/'frozen/protocols'/('plain_cosine_'+arm)/'protocol.json'
        assert cfg['protocol_sha256'] == digest(new_protocol)
        assert prior['protocol_sha256'] == digest(old_protocol)
        p, q = read(new_protocol), read(old_protocol)
        for key in ('lr_policy', 'lr_horizon_updates', 'lr_schedule_details',
                    'adamw_weight_decay', 'dropout_rate', 'augmentation', 'native_geometry',
                    'loss_arguments', 'loss_weights', 'loss_source_sha256', 'constraint_warmup_updates',
                    'calibration_target_ratio', 'calibration_p95_cap', 'constraints', 'constraint_recipe'):
            assert p[key] == q[key], key
        assert p['initial_lrs'] == {'unet':current['peak_lr']}
        assert q['initial_lrs'] == {'unet':.0006} and p['lr_policy'] == 'cap_matched'
        assert read(folder/'runtime_recipe.json')['objective_sha256'] == read(old/'runtime_recipe.json')['objective_sha256']
        old_gain = old_score['validation_percent']-old_base['validation_percent']
        rows.append(dict(candidate=current['candidate'], seed=seed, fold=fold,
            peak_lr=current['peak_lr'], previous_peak_lr=.0006,
            previous_constraint_gain_pp=old_gain, new_constraint_gain_pp=current['gain_pp'],
            constraint_interaction_pp=current['gain_pp']-old_gain,
            baseline_lr_effect_pp=current['baseline']['validation_percent']-old_base['validation_percent'],
            constrained_lr_effect_pp=current_score['validation_percent']-old_score['validation_percent'],
            previous_constraint_archive_sha256=old_score['archive_manifest_sha256'],
            previous_baseline_archive_sha256=old_base['archive_manifest_sha256'],
            current_constraint_archive_sha256=current_score['archive_manifest_sha256']))
    output = dict(rows=rows, missing_previous_cells=missing,
        input_result_sha256=digest(report/'lower_lr_results.json'),
        analysis_source_sha256=digest(Path(__file__)),
        scope='Same seed/fold, data, plain model, original DiceCE1 and auxiliaries, AdamW, cosine schedule and FLOPs cap. Only peak LR changes. Each recipe has a fresh matched baseline and training-only calibration.',
        caveat='Available development comparisons only; absent historical cells are not imputed. A better baseline and a larger constraint gain are separate outcomes.')
    write(report/'peak_lr_interactions.json', output)
    print(json.dumps(rows))


if __name__ == '__main__':
    main()
