"""Compare archived cosine baselines across CE weights without mixing constraint gains."""
import json
from itertools import combinations
from pathlib import Path
from statistics import mean

from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.mni_constraints.analyze_unetr_constraint_pilot import evaluation
from scripts.mni_constraints.unetr_ce050_cosine_study import ROOT, NAME, PAIRS


STAGES = (
    (1., 'mni_unetr_plain_cosine53t_20261109', 'plain_cosine'),
    (.25, 'mni_unetr_ce025_cosine53t_20261109', 'ce025_cosine'),
    (.5, NAME, 'ce050_cosine'),
)


def baseline(weight, name, recipe, seed, fold):
    report = ROOT/'reports'/name
    folder = report/'runs'/f'MNI_unetr_{recipe}_dice_f{fold}s{seed}'
    if not (folder/'cleanup.json').exists():
        return None
    score = evaluation(folder)  # Verifies the complete durable archive.
    cfg = read(folder/'config.json')
    protocol_file = report/'frozen/protocols'/f'{recipe}_dice/protocol.json'
    protocol = read(protocol_file)
    assert cfg['protocol_sha256'] == digest(protocol_file)
    assert cfg['objective'] == 'dice' and cfg['seed'] == seed and cfg['fold'] == fold
    assert protocol['loss_arguments']['lambda_ce'] == weight
    assert protocol['lr_policy'] == 'cap_matched'
    model = read(folder/'model_recipe.json')
    if weight != 1.:
        effective = read(folder/'base_loss_recipe.json')
        assert effective['arguments'] == protocol['loss_arguments']
        assert effective['dense_cross_entropy_weight'] == weight
        for key, protocol_key in (('adapter_sha256', 'base_loss_adapter_sha256'),
                                  ('loss_source_sha256', 'base_loss_source_sha256')):
            assert effective[key] == protocol[protocol_key]
        suffix = 'ce025' if weight == .25 else 'ce050'
        assert effective['adapter_sha256'] == digest(
            report/'frozen/source/scripts/mni_constraints'/f'unetr_{suffix}_cosine_runtime.py')
        assert effective['loss_source_sha256'] == digest(
            report/'frozen/source/scripts/unetr_subfields'/f'foreground_dice_{suffix}.py')
    return dict(weight=weight, score=score, config=cfg, protocol=protocol, model=model)


def main():
    comparisons, missing = [], []
    for seed, fold in PAIRS:
        available = []
        for weight, name, recipe in STAGES:
            item = baseline(weight, name, recipe, seed, fold)
            if item is None:
                missing.append(dict(seed=seed, fold=fold, ce_weight=weight))
            else:
                available.append(item)
        for old, new in combinations(available, 2):
            for key in ('seed', 'fold', 'objective', 'train_cases', 'validation_cases',
                        'initial_lr', 'lr_horizon_updates', 'schedule', 'dataset_sha256',
                        'script_sha256', 'objective_source_sha256', 'augmentation', 'torch', 'monai'):
                assert old['config'][key] == new['config'][key], (seed, fold, key)
            for key in ('lr_policy', 'lr_schedule_details', 'initial_lrs', 'adamw_weight_decay',
                        'dropout_rate', 'augmentation', 'pkl_sha256', 'splits_sha256', 'native_geometry'):
                assert old['protocol'][key] == new['protocol'][key], key
            assert new['protocol']['loss_arguments'] == dict(
                old['protocol']['loss_arguments'], lambda_ce=new['weight'])
            for key in ('model', 'declared', 'kwargs', 'actual_dropout_modules', 'parameters'):
                assert old['model'][key] == new['model'][key], key
            a, b = old['score'], new['score']
            comparisons.append(dict(seed=seed, fold=fold,
                from_ce=old['weight'], to_ce=new['weight'],
                from_validation_dice=a['validation_percent'], to_validation_dice=b['validation_percent'],
                change_pp=b['validation_percent']-a['validation_percent'],
                training_change_pp=b['training_percent']-a['training_percent'],
                subfield_change_pp={k:b['per_subfield_percent'][k]-v
                                   for k, v in a['per_subfield_percent'].items()},
                archive_sha256=[a['archive_manifest_sha256'], b['archive_manifest_sha256']]))
    summary = []
    for before, after in ((1., .25), (1., .5), (.25, .5)):
        rows = [r for r in comparisons if (r['from_ce'], r['to_ce']) == (before, after)]
        summary.append(dict(from_ce=before, to_ce=after, pairs=len(rows),
            mean_change_pp=mean(r['change_pp'] for r in rows) if rows else None,
            positive_pairs=sum(r['change_pp'] > 0 for r in rows)))
    output = dict(comparisons=comparisons, summary=summary, missing_baselines=missing,
        scope='Archived final-budget unconstrained controls only. Same model, data, optimizer, cosine schedule and cap; CE coefficient differs. These are base-recipe effects, not constraint gains.',
        caveat='Available development pairs on overlapping folds; no missing result is imputed. Quarter-CE legacy loss_weights metadata is superseded by verified effective adapter and loss_arguments.',
        source_sha256=digest(Path(__file__)))
    write(ROOT/'reports'/NAME/'cosine_ce_baseline_comparison.json', output)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
