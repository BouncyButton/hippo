"""Compare completed matched cells of the CE-weight by LR-schedule study."""
from pathlib import Path
import json

from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.mni_constraints import unetr_ce025_cosine_study as study
from scripts.mni_constraints.analyze_unetr_constraint_pilot import evaluation


CONDITIONS = (
    ('ce1_hold', 'mni_unetr_simple_recipe53t_20261108', 'plain', 'hold_half', 1.),
    ('ce1_cosine', 'mni_unetr_plain_cosine53t_20261109', 'plain_cosine', 'cap_matched', 1.),
    ('ce025_hold', 'mni_unetr_plain_ce02553t_20261109', 'plain_ce025', 'hold_half', .25),
    ('ce025_cosine', study.NAME, 'ce025_cosine', 'cap_matched', .25),
)


def main():
    report = study.ROOT/'reports'/study.NAME
    result = read(report/'ce025_cosine_results.json')
    rows = []
    for current in result['rows']:
        seed, fold, arm = current['seed'], current['fold'], current['arm']
        cells, reference_cfg, reference_model = {}, None, None
        for key, name, recipe, policy, ce_weight in CONDITIONS:
            root = study.ROOT/'reports'/name
            base = root/'runs'/f'MNI_unetr_{recipe}_dice_f{fold}s{seed}'
            constrained = root/'runs'/f'MNI_unetr_{recipe}_{arm}_f{fold}s{seed}'
            if not all((p/'cleanup.json').exists() for p in (base, constrained)):
                continue
            values = []
            for folder, objective in ((base, 'dice'), (constrained, arm)):
                score = evaluation(folder)
                cfg, model = read(folder/'config.json'), read(folder/'model_recipe.json')
                protocol_file = root/'frozen/protocols'/f'{recipe}_{objective}'/'protocol.json'
                protocol = read(protocol_file)
                assert cfg['protocol_sha256'] == digest(protocol_file)
                assert protocol['lr_policy'] == policy
                assert cfg['objective'] == objective
                assert cfg['schedule'] == protocol['models']['unet']['objectives'][objective]
                if reference_cfg is None:
                    reference_cfg, reference_model = cfg, model
                for field in ('seed', 'fold', 'train_cases', 'validation_cases', 'initial_lr',
                              'lr_horizon_updates', 'dataset_sha256', 'script_sha256',
                              'objective_source_sha256', 'augmentation', 'torch', 'monai'):
                    assert cfg[field] == reference_cfg[field], (folder, field)
                for field in ('model', 'declared', 'kwargs', 'actual_dropout_modules', 'parameters'):
                    assert model[field] == reference_model[field], (folder, field)
                if ce_weight == .25:
                    assert read(folder/'base_loss_recipe.json')['dense_cross_entropy_weight'] == .25
                values.append(score)
            cells[key] = dict(baseline_dice=values[0]['validation_percent'],
                constrained_dice=values[1]['validation_percent'],
                gain_pp=values[1]['validation_percent']-values[0]['validation_percent'],
                archive_sha256=[s['archive_manifest_sha256'] for s in values])
        interaction = None
        if len(cells) == 4:
            interaction = ((cells['ce025_cosine']['gain_pp']-cells['ce025_hold']['gain_pp'])
                           -(cells['ce1_cosine']['gain_pp']-cells['ce1_hold']['gain_pp']))
        rows.append(dict(seed=seed, fold=fold, arm=arm, cells=cells,
            complete_four_cell_comparison=len(cells)==4,
            ce_by_lr_interaction_in_constraint_gain_pp=interaction))
    output = dict(rows=rows, input_sha256=digest(report/'ce025_cosine_results.json'),
        source_sha256=digest(Path(__file__)),
        scope='Completed final-budget matched comparisons only; same data/model/optimizer/initial LR/cap. Compare baseline and constrained absolute scores separately from their paired gain. Calibration is training-only and freshly estimated per base recipe.',
        caveat='Adaptive development comparisons on overlapping folds; incomplete historical cells stay missing. A four-cell interaction on one pair is not evidence of nine-pair consistency.')
    write(report/'ce_lr_factorial.json', output)
    print(json.dumps(rows))


if __name__ == '__main__':
    main()
