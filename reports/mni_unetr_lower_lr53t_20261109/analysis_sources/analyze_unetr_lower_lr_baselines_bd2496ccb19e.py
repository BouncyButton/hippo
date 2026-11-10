"""Compare reduced-LR controls with the identical plain LR0.0006 recipe."""
import hashlib
import json
from pathlib import Path
from statistics import mean

from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.mni_constraints import unetr_lower_lr_study as study
from scripts.mni_constraints.analyze_unetr_constraint_pilot import evaluation


def main():
    report = study.ROOT / 'reports' / study.NAME
    prior = study.ROOT / 'reports/mni_unetr_plain_cosine53t_20261109'
    result_bytes = (report / 'lower_lr_results.json').read_bytes()
    result = json.loads(result_bytes)
    rows, missing = [], []
    for item in result['archived_runs']:
        if item['role'] != 'baseline':
            continue
        seed, fold = item['seed'], item['fold']
        old = prior / 'runs' / f'MNI_unetr_plain_cosine_dice_f{fold}s{seed}'
        if not (old / 'cleanup.json').exists():
            missing.append(dict(recipe=item['recipe'], seed=seed, fold=fold))
            continue
        current = report / 'runs' / item['id']
        cfg, old_cfg = read(current / 'config.json'), read(old / 'config.json')
        for key in ('seed', 'fold', 'objective', 'train_cases', 'validation_cases',
                    'lr_horizon_updates', 'schedule', 'dataset_sha256', 'script_sha256',
                    'objective_source_sha256', 'augmentation', 'torch', 'monai'):
            assert cfg[key] == old_cfg[key], (item['id'], key)
        assert cfg['initial_lr'] == item['peak_lr'] and old_cfg['initial_lr'] == .0006
        new_protocol = report / 'frozen/protocols' / item['protocol'] / 'protocol.json'
        old_protocol = prior / 'frozen/protocols/plain_cosine_dice/protocol.json'
        assert cfg['protocol_sha256'] == digest(new_protocol)
        assert old_cfg['protocol_sha256'] == digest(old_protocol)
        p, q = read(new_protocol), read(old_protocol)
        for key in ('lr_policy', 'lr_horizon_updates', 'lr_schedule_details',
                    'adamw_weight_decay', 'dropout_rate', 'augmentation',
                    'loss_arguments', 'loss_weights', 'loss_source_sha256'):
            assert p[key] == q[key], (item['id'], key)
        model, old_model = read(current / 'model_recipe.json'), read(old / 'model_recipe.json')
        for key in ('model', 'declared', 'kwargs', 'actual_dropout_modules', 'parameters'):
            assert model[key] == old_model[key], key
        new_score, old_score = evaluation(current), evaluation(old)
        rows.append(dict(
            recipe=item['recipe'], seed=seed, fold=fold, peak_lr=item['peak_lr'],
            baseline_dice=new_score['validation_percent'],
            previous_baseline_dice=old_score['validation_percent'],
            validation_change_pp=new_score['validation_percent']-old_score['validation_percent'],
            training_change_pp=new_score['training_percent']-old_score['training_percent'],
            subfield_change_pp={key: value-old_score['per_subfield_percent'][key]
                               for key, value in new_score['per_subfield_percent'].items()},
            current_archive_sha256=new_score['archive_manifest_sha256'],
            previous_archive_sha256=old_score['archive_manifest_sha256'],
        ))
    summary = []
    for recipe in sorted({row['recipe'] for row in rows}):
        group = [row for row in rows if row['recipe'] == recipe]
        summary.append(dict(recipe=recipe, pairs=len(group),
                            mean_validation_change_pp=mean(row['validation_change_pp'] for row in group),
                            improved_pairs=sum(row['validation_change_pp'] > 0 for row in group)))
    write(report / 'peak_lr_baseline_comparison.json', dict(
        rows=rows, summary=summary, missing_previous_cells=missing,
        input_result_sha256=hashlib.sha256(result_bytes).hexdigest(),
        analysis_source_sha256=digest(Path(__file__)),
        scope='Matched same-fold/seed/data, plain DiceCE1 AdamW cosine controls; only peak LR changes.',
        caveat='Available development comparisons with overlapping folds. Missing controls are not imputed; this measures baseline effects, not constraint effects.',
    ))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
