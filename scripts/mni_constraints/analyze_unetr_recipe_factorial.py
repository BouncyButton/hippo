"""Separate base-recipe effects from constraint interactions on matched pairs."""
import json
from pathlib import Path

from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.mni_constraints import unetr_simple_recipe_study as study
from scripts.mni_constraints.analyze_unetr_constraint_pilot import evaluation

BASES = ('original', 'noaug', 'nodrop', 'plain')


def scalar_metrics(result):
    return {**{k:result[k] for k in ('validation_percent','training_percent','union_percent')},
            'training_validation_gap_pp':result['training_percent']-result['validation_percent'],
            **result['per_subfield_percent']}


def contrasts(cells):
    """Every subtraction uses the same seed/fold and metric; missing stays missing."""
    metrics=next(iter(cells.values())).keys()
    def difference(left, right):
        if left not in cells or right not in cells:
            return None
        return {metric:cells[left][metric]-cells[right][metric] for metric in metrics}
    gains={(base,arm):difference((base,arm),(base,'dice'))
           for base in BASES for arm in study.ARMS[1:]}
    def gain_difference(left, right, arm):
        a,b=gains[left,arm],gains[right,arm]
        return {metric:a[metric]-b[metric] for metric in metrics} if a is not None and b is not None else None
    main={
        'remove_augmentation_at_dropout01':difference(('noaug','dice'),('original','dice')),
        'remove_augmentation_at_dropout0':difference(('plain','dice'),('nodrop','dice')),
        'remove_dropout_with_augmentation':difference(('nodrop','dice'),('original','dice')),
        'remove_dropout_without_augmentation':difference(('plain','dice'),('noaug','dice')),
    }
    interactions=[]
    for arm in study.ARMS[1:]:
        a=gain_difference('noaug','original',arm)
        b=gain_difference('plain','nodrop',arm)
        interactions.append(dict(arm=arm,
            constraint_gain_by_base={base:gains[base,arm] for base in BASES},
            augmentation_removal_effect_on_constraint_gain_at_dropout01=a,
            augmentation_removal_effect_on_constraint_gain_at_dropout0=b,
            dropout_removal_effect_on_constraint_gain_with_augmentation=gain_difference('nodrop','original',arm),
            dropout_removal_effect_on_constraint_gain_without_augmentation=gain_difference('plain','noaug',arm),
            three_way_interaction={metric:b[metric]-a[metric] for metric in metrics} if a is not None and b is not None else None))
    return dict(baseline_recipe_effects=main,constraint_interactions=interactions)


def main():
    report=study.ROOT/'reports'/study.NAME
    frozen=report/'frozen'
    result=read(report/'simple_recipe_results.json')
    rows=[]
    controls={(r['seed'],r['fold']):r for r in read(frozen/'controls.json')}
    old={(r['seed'],r['fold'],r['arm']):r for r in read(frozen/'old_constraints.json')}
    for seed,fold in study.PAIRS:
        records=[]
        def append(base,arm,folder):
            records.append(dict(base=base,arm=arm,metrics=scalar_metrics(evaluation(folder)),
                                archive_manifest_sha256=digest(folder/'archive_manifest.json')))
        append('original','dice',frozen/'controls'/controls[seed,fold]['name'])
        for arm in study.ARMS[1:]:
            reference=old.get((seed,fold,arm))
            if reference:
                append('original',arm,frozen/'old_constraints'/reference['name'])
        current=[r for r in result['archived_runs'] if (r['seed'],r['fold'])==(seed,fold)]
        if not current:
            continue
        for run in current:
            append(run['recipe'],run['arm'],report/'runs'/run['id'])
        cells={(r['base'],r['arm']):r['metrics'] for r in records}
        assert len(cells)==len(records)
        rows.append(dict(seed=seed,fold=fold,complete_factorial=len(cells)==12,cells=records,**contrasts(cells)))
    output=dict(rows=rows,
        factors=dict(augmentation=['xy_d4_v1','none'],dropout=[.1,0.],constraints=list(study.ARMS)),
        scope='Same-pair factorial contrasts at final409updates under53.305T. Original constraints with fresh training-only5%calibration; all other base settings fixed. Missing cells remain missing; no averaging over different selected subsets.',
        caveat='Development-validation comparisons on overlapping folds. These contrasts quantify recipe interactions, not a proven biological or optimizer mechanism.',
        input_result_sha256=digest(report/'simple_recipe_results.json'),analysis_source_sha256=digest(Path(__file__)))
    write(report/'recipe_factorial_results.json',output)
    print(json.dumps([dict(seed=r['seed'],fold=r['fold'],cells=len(r['cells']),complete_factorial=r['complete_factorial'],
        baseline_effects={k:v['validation_percent'] if v is not None else None for k,v in r['baseline_recipe_effects'].items()},
        constraint_gains={c['arm']:{k:v['validation_percent'] if v is not None else None for k,v in c['constraint_gain_by_base'].items()} for c in r['constraint_interactions']}) for r in rows]))


if __name__=='__main__':
    main()
