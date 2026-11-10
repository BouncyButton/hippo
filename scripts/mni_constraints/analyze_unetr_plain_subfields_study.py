"""Audit fresh same-recipe controls and absolute performance of simplifications."""
import csv
import json
from pathlib import Path
from statistics import mean

from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.mni_constraints import unetr_plain_subfields_study as study
from scripts.mni_constraints.analyze_unetr_constraint_pilot import evaluation
from scripts.mni_constraints.analyze_unetr_readiness_study import prefix_check, stats
from scripts.mni_constraints.unetr_recipe_pair_selection import select


def audit_run(folder, task, frozen):
    current = evaluation(folder)
    cfg = read(folder/'config.json')
    protocol_file = frozen/'protocols'/task['protocol']/'protocol.json'
    protocol = read(protocol_file)
    assert cfg['protocol_sha256'] == digest(protocol_file)
    assert cfg['script_sha256'] == digest(frozen/study.CORE)
    assert cfg['schedule'] == protocol['models']['unet']['objectives'][task['arm']]
    assert cfg['objective'] == task['arm']
    for key in ('seed', 'fold', 'augmentation'):
        assert cfg[key] == task[key], (task['id'], key)
    model = read(folder/'model_recipe.json')
    assert model['declared'] == protocol['unetr_simplification']
    assert model['kwargs'] == study.runtime.model_kwargs(task['dropout_rate']) or model['kwargs'] == {
        k:list(v) if isinstance(v, tuple) else v for k,v in study.runtime.model_kwargs(task['dropout_rate']).items()}
    assert model['parameters'] == 92685572
    assert model['actual_dropout_modules'] and all(
        p == task['dropout_rate'] for p in model['actual_dropout_modules'].values())
    assert model['runtime_source_sha256'] == digest(frozen/'source/scripts/mni_constraints/unetr_simple_recipe_runtime.py')
    cal = None
    if task['role'] == 'constraint':
        cal = read(folder/'calibration.json')
        assert cal['target_ratio'] == .05 and cal['p95_cap'] == .75
        assert cal['anchor_successful_updates'] == 102
        assert [c['case_name'] for c in cal['cases']] == cfg['train_cases']
        assert set(cal['weights']) == set(task['arm'].split('_')[1:])
        assert all(s['weighted_median_ratio'] <= .0500001 and s['weighted_p95_ratio'] <= .7500001
                   for s in cal['statistics'].values())
    return dict(**task, current=current, calibration=cal,
                training_validation_gap_pp=current['training_percent']-current['validation_percent'])


def main():
    report = study.ROOT/'reports'/study.NAME
    frozen = report/'frozen'
    plan = read(frozen/'plan.json')
    references = {(c['seed'],c['fold']):frozen/'controls'/c['name'] for c in read(frozen/'controls.json')}
    old_baselines = {pair:evaluation(folder) for pair,folder in references.items()}
    old_constraints = {(r['arm'],r['seed'],r['fold']):evaluation(frozen/'old_constraints'/r['name'])
                       for r in read(frozen/'old_constraints.json')}
    archived = []
    for task in plan['tasks']:
        folder = report/'runs'/task['id']
        if (folder/'cleanup.json').exists():
            row = audit_run(folder,task,frozen)
            old = old_baselines[task['seed'],task['fold']]
            row['old_winning_baseline'] = old
            row['absolute_gain_vs_old_baseline_pp'] = row['current']['validation_percent']-old['validation_percent']
            archived.append(row)
    by_id = {r['id']:r for r in archived}
    completed = {r['id']:read(report/'runs'/r['id']/'validation_latest.json')['summary']['macro_hard_dice'] for r in archived}
    selected = select(plan['tasks'],completed,study.CANDIDATES)
    rows = []
    for pair in selected['rows']:
        item = by_id[pair['id']]
        baseline = by_id[pair['baseline_id']]['current']
        current = item['current']
        folder, reference = report/'runs'/item['id'],report/'runs'/pair['baseline_id']
        cfg, bc = read(folder/'config.json'),read(reference/'config.json')
        for key in ('seed','fold','train_cases','validation_cases','initial_lr','lr_horizon_updates',
                    'dataset_sha256','script_sha256','objective_source_sha256','augmentation','torch','monai'):
            assert cfg[key] == bc[key], (item['id'],key)
        assert read(folder/'model_recipe.json') == read(reference/'model_recipe.json')
        prior = old_constraints.get((item['arm'],item['seed'],item['fold']))
        old_gain = (prior['validation_percent']-item['old_winning_baseline']['validation_percent']) if prior else None
        rows.append(dict(**item,baseline=baseline,gain_pp=pair['gain_pp'],
            prefix_replay=prefix_check(folder,reference,102),
            training_gain_pp=current['training_percent']-baseline['training_percent'],
            baseline_training_validation_gap_pp=baseline['training_percent']-baseline['validation_percent'],
            union_gain_pp=current['union_percent']-baseline['union_percent'],
            subfield_gain_pp={k:current['per_subfield_percent'][k]-baseline['per_subfield_percent'][k]
                              for k in baseline['per_subfield_percent']},
            error_delta={k:current[k]-baseline[k] for k in ('fn','fp','swaps')},
            old_recipe_constraint_gain_pp=old_gain,
            constraint_interaction_pp=pair['gain_pp']-old_gain if old_gain is not None else None))
    summary = []
    for candidate in selected['summary']:
        group = [r for r in rows if r['candidate']==candidate['candidate']]
        summary.append(dict(candidate,all_pairs=stats(group),
            mean_absolute_gain_vs_old_baseline_pp=mean(r['absolute_gain_vs_old_baseline_pp'] for r in group) if group else None,
            minimum_absolute_gain_vs_old_baseline_pp=min((r['absolute_gain_vs_old_baseline_pp'] for r in group),default=None),
            mean_new_baseline_gain_vs_old_baseline_pp=mean(r['baseline']['validation_percent']-r['old_winning_baseline']['validation_percent'] for r in group) if group else None))
    live = read(report/'live.json').get('status.json',{})
    complete = live.get('status') == 'complete' and live.get('completed') == len(archived)
    if complete:
        assert selected['complete'] and live['selected']['winner'] == selected['selected']['winner']
    result = dict(status='complete' if complete else 'partial',
        outcome=selected['outcome'] if complete else 'running',winner=selected['selected']['winner'],
        rows=rows,archived_runs=archived,summary=summary,
        scope='Original SDF/depth composition under plain base, paired against identical-recipe controls (three reused). Original delayed5% constraints;409updates;53.305T cap. Absolute comparison against old winning baseline reported separately.',
        caveat='Adaptive development selection on overlapping folds. Gains from lowering a baseline alone do not establish improved absolute performance.',
        analysis_source_sha256=digest(Path(__file__)))
    write(report/'plain_subfields_results.json',result)
    flat = [{k:r[k] for k in ('candidate','seed','fold','gain_pp','absolute_gain_vs_old_baseline_pp',
                              'training_validation_gap_pp','baseline_training_validation_gap_pp','constraint_interaction_pp')} for r in rows]
    if flat:
        with (report/'paired_results.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print(json.dumps(dict(status=result['status'],outcome=result['outcome'],winner=result['winner'],
                         completed_runs=len(archived),summary=summary)))
    if complete:
        from scripts.mni_constraints.analyze_unetr_delayed_combinations import plot
        plot(report,[dict(r,arm=r['candidate']) for r in rows],arms=study.CANDIDATES,pairs=study.PAIRS,
             title='UNETR plain base: original SDF/depth gain vs SAME-recipe baseline\n* development pilot; blank cells not run',stem='plain_subfields_results')


if __name__ == '__main__':
    main()
