"""Verify archives, identical baseline prefixes, and nine-pair readiness outcomes."""
import csv
import json
from pathlib import Path
from statistics import mean

from scripts.cross_dataset_campaign.common import read,write,digest
from scripts.mni_constraints import unetr_readiness_study as study
from scripts.mni_constraints.analyze_unetr_constraint_pilot import evaluation
from scripts.mni_constraints.unetr_readiness import TRAINING_FIELDS


def prefix_check(current,baseline,prefix):
    def curve(folder):
        with (folder/'metrics.csv').open() as file:
            return {int(r['step']):r for r in csv.DictReader(file) if int(r['step'])<=prefix}
    actual,reference=curve(current),curve(baseline)
    common=sorted(set(actual)&set(reference));assert common
    error=max(abs(float(actual[s][k])-float(reference[s][k])) for s in common for k in TRAINING_FIELDS)
    assert error<1e-9, ('Unmatched unconstrained training prefix',current.name,error)
    return dict(shared_checks=common,maximum_training_dice_error=error)


def stats(rows):
    return dict(pairs=len(rows),mean_gain_pp=mean(r['gain_pp'] for r in rows) if rows else None,
                minimum_gain_pp=min((r['gain_pp'] for r in rows),default=None),
                improved_pairs=sum(r['gain_pp']>0 for r in rows))


def main():
    report=study.ROOT/'reports'/study.NAME;frozen=report/'frozen';plan=read(frozen/'plan.json')
    references={(c['seed'],c['fold']):frozen/'controls'/c['name'] for c in read(frozen/'controls.json')}
    baselines={pair:evaluation(folder) for pair,folder in references.items()};rows=[]
    for task in plan['tasks']:
        folder=report/'runs'/task['id']
        if not (folder/'cleanup.json').exists():continue
        pair=(task['seed'],task['fold']);reference=references[pair]
        baseline=baselines[pair];current=evaluation(folder)
        cfg=read(folder/'config.json');bc=read(reference/'config.json')
        for key in ('seed','fold','train_cases','validation_cases','initial_lr','lr_horizon_updates',
                    'dataset_sha256','script_sha256','objective_source_sha256','augmentation','torch','monai'):
            assert cfg[key]==bc[key],(task['id'],key)
        derived=study.derive(reference,task['threshold'])
        p=read(frozen/'protocols'/task['protocol']/'protocol.json');assert p['readiness']==derived
        assert cfg['objective']==task['arm'] and cfg['schedule']['prefix_updates']==derived['update']
        cal=read(folder/'calibration.json')
        assert cal['target_ratio']==.05 and cal['p95_cap']==.75 and cal['anchor_successful_updates']==derived['update']
        assert [c['case_name'] for c in cal['cases']]==cfg['train_cases']
        assert set(cal['weights'])==set(task['arm'].split('_')[1:])
        assert all(s['weighted_median_ratio']<=.0500001 and s['weighted_p95_ratio']<=.7500001 for s in cal['statistics'].values())
        rows.append(dict(**task,baseline=baseline,current=current,readiness=derived,
            prefix_replay=prefix_check(folder,reference,derived['update']),
            gain_pp=current['validation_percent']-baseline['validation_percent'],
            training_gain_pp=current['training_percent']-baseline['training_percent'],
            union_gain_pp=current['union_percent']-baseline['union_percent'],
            subfield_gain_pp={k:current['per_subfield_percent'][k]-baseline['per_subfield_percent'][k]
                              for k in baseline['per_subfield_percent']},
            error_delta={k:current[k]-baseline[k] for k in ('fn','fp','swaps')},
            calibration_weights=cal['weights'],calibration_statistics=cal['statistics']))
    summary=[]
    for c in study.CANDIDATES:
        group=[r for r in rows if r['candidate']==c['key']]
        summary.append(dict(candidate=c['key'],arm=c['arm'],threshold=c['threshold'],all_pairs=stats(group),
            pilots=stats([r for r in group if r['pilot']]),new_confirmation=stats([r for r in group if not r['pilot']]),
            all_nine_improved=len(group)==9 and all(r['gain_pp']>0 for r in group)))
    live=read(report/'live.json').get('status.json',{})
    complete=live.get('status')=='complete' and live.get('completed')==len(rows)
    winners=[s['candidate'] for s in summary if s['all_nine_improved']]
    if complete:
        assert sum(r['pilot'] for r in rows)==16
        assert live['selected']['winner']==(winners[0] if winners else None)
    result=dict(status='complete' if complete else 'partial',outcome='nine_pair_success' if winners else 'exhausted' if complete else 'running',
        winner=winners[0] if winners else None,rows=rows,summary=summary,
        scope='Same409-update53.305T base recipe; original losses; training-only readiness50/60%,target5%,33-update ramp.',
        caveats=['Development-validation exploration, overlapping folds; no untouched test claim.',
                 'One training-only trigger algorithm per candidate; no seed/fold-specific threshold or target.',
                 'All nine improvement is required; retain all attempted candidates.'],analysis_source_sha256=digest(Path(__file__)))
    write(report/'readiness_results.json',result)
    flat=[dict(candidate=r['candidate'],arm=r['arm'],seed=r['seed'],fold=r['fold'],pilot=r['pilot'],
        onset=r['readiness']['update'],baseline_dice=r['baseline']['validation_percent'],
        constraint_dice=r['current']['validation_percent'],gain_pp=r['gain_pp'],
        training_gain_pp=r['training_gain_pp'],union_gain_pp=r['union_gain_pp'],
        fn_delta=r['error_delta']['fn'],fp_delta=r['error_delta']['fp'],swaps_delta=r['error_delta']['swaps']) for r in rows]
    if flat:
        with (report/'paired_results.csv').open('w') as file:
            writer=csv.DictWriter(file,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print(json.dumps(dict(status=result['status'],outcome=result['outcome'],winner=result['winner'],summary=summary)))
    if complete:plot(report,rows)


def plot(report,rows):
    from scripts.mni_constraints import analyze_unetr_delayed_combinations as plotting
    plotting.plot(report,[dict(r,arm=r['candidate']) for r in rows],arms=study.KEYS,pairs=study.PAIRS,
                  title='UNETR training-readiness timing: Dice gain vs matched baseline\n* development pilot; blank cells not run',
                  stem='readiness_results')


if __name__=='__main__':main()
