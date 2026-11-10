"""Separate base CE-weight effects from the matched constraint comparisons."""
import json
from pathlib import Path
from statistics import mean

from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.mni_constraints import unetr_ce025_cosine_study as study
from scripts.mni_constraints.analyze_unetr_constraint_pilot import evaluation


def main():
    report = study.ROOT/'reports'/study.NAME
    prior_report = study.ROOT/'reports/mni_unetr_plain_cosine53t_20261109'
    result = read(report/'ce025_cosine_results.json')
    rows, missing = [], []
    for current in result['archived_runs']:
        if current['role'] != 'baseline':
            continue
        seed, fold = current['seed'], current['fold']
        folder = report/'runs'/current['id']
        prior = prior_report/'runs'/f'MNI_unetr_plain_cosine_dice_f{fold}s{seed}'
        if not (prior/'cleanup.json').exists():
            missing.append(dict(seed=seed, fold=fold))
            continue
        new_score, old_score = evaluation(folder), evaluation(prior)
        cfg, old_cfg = read(folder/'config.json'), read(prior/'config.json')
        for key in ('seed', 'fold', 'objective', 'train_cases', 'validation_cases',
                    'initial_lr', 'lr_horizon_updates', 'schedule', 'dataset_sha256',
                    'script_sha256', 'objective_source_sha256', 'augmentation', 'torch', 'monai'):
            assert cfg[key] == old_cfg[key], (current['id'], key)
        new_protocol = report/'frozen/protocols'/current['protocol']/'protocol.json'
        old_protocol = prior_report/'frozen/protocols/plain_cosine_dice/protocol.json'
        assert cfg['protocol_sha256'] == digest(new_protocol)
        assert old_cfg['protocol_sha256'] == digest(old_protocol)
        p, q = read(new_protocol), read(old_protocol)
        for key in ('lr_policy', 'lr_schedule_details', 'initial_lrs', 'adamw_weight_decay',
                    'dropout_rate', 'augmentation', 'pkl_sha256', 'splits_sha256', 'native_geometry'):
            assert p[key] == q[key], key
        assert p['lr_policy'] == 'cap_matched'
        assert p['loss_arguments'] == dict(q['loss_arguments'], lambda_ce=.25)
        assert q['loss_arguments']['lambda_ce'] == 1.
        effective = read(folder/'base_loss_recipe.json')
        assert effective['arguments'] == p['loss_arguments']
        assert effective['dense_cross_entropy_weight'] == .25
        assert effective['adapter_sha256'] == p['base_loss_adapter_sha256']
        new_model, old_model = read(folder/'model_recipe.json'), read(prior/'model_recipe.json')
        for key in ('model', 'declared', 'kwargs', 'actual_dropout_modules', 'parameters'):
            assert new_model[key] == old_model[key], key
        rows.append(dict(seed=seed, fold=fold,
            ce1_dice=old_score['validation_percent'], ce025_dice=new_score['validation_percent'],
            change_pp=new_score['validation_percent']-old_score['validation_percent'],
            training_change_pp=new_score['training_percent']-old_score['training_percent'],
            subfield_change_pp={key:new_score['per_subfield_percent'][key]-value
                               for key,value in old_score['per_subfield_percent'].items()},
            archive_sha256=[old_score['archive_manifest_sha256'], new_score['archive_manifest_sha256']]))
    output = dict(rows=rows, missing_original_ce_baselines=missing,
        mean_change_pp=mean(r['change_pp'] for r in rows) if rows else None,
        input_sha256=digest(report/'ce025_cosine_results.json'), source_sha256=digest(Path(__file__)),
        scope='Unconstrained baseline comparisons at the same post-warmup cosine schedule, model, data, optimizer, and cap. CE coefficient changes from1 to.25. These changes are not constraint gains.',
        metadata_note='The frozen quarter-CE protocol retains legacy loss_weights and loss_source_sha256 fields from its parent. Effective loss is enforced by loss_arguments plus the protocol-bound adapter and recorded in base_loss_recipe.json; its actual CE coefficient is.25. Frozen protocols are not edited retroactively.',
        caveat='Available development pairs only; stronger hold-half baselines are a separate comparison.')
    write(report/'cosine_baseline_loss_comparison.json', output)
    print(json.dumps(dict(rows=rows, mean_change_pp=output['mean_change_pp'])))


if __name__ == '__main__':
    main()
