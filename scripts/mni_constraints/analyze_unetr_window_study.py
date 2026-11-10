"""Verify finite-window trials against matched baseline and persistent references."""
import csv
import json
from pathlib import Path
from statistics import mean

from scripts.cross_dataset_campaign.common import read,write,digest
from scripts.mni_constraints import unetr_window_study as study
from scripts.mni_constraints.analyze_unetr_constraint_pilot import evaluation
from scripts.mni_constraints.analyze_unetr_readiness_study import prefix_check,stats
from scripts.mni_constraints.unetr_sequential_selection import select

RESULT_FILENAME='window_results.json'
PLOT_STEM='window_results'
TITLE='UNETR finite constraint windows: gain vs matched baseline\n* development pilot; blank cells not run'
SCOPE='Original constraint definitions; same409 updates and53.305T cap; linear auxiliary fade with extra scalar work charged.'

def main():
    report=study.ROOT/'reports'/study.NAME;frozen=report/'frozen';plan=read(frozen/'plan.json')
    references={(c['seed'],c['fold']):frozen/'controls'/c['name'] for c in read(frozen/'controls.json')}
    baselines={pair:evaluation(folder) for pair,folder in references.items()}
    persistent={(r['arm'],r['seed'],r['fold']):frozen/'persistent_references'/r['name']
                for r in read(frozen/'persistent_references.json')}
    rows=[]
    for task in plan['tasks']:
        folder=report/'runs'/task['id']
        if not (folder/'cleanup.json').exists():continue
        pair=task['seed'],task['fold'];reference=references[pair]
        baseline,current=baselines[pair],evaluation(folder)
        cfg,bc=read(folder/'config.json'),read(reference/'config.json')
        for key in ('seed','fold','train_cases','validation_cases','initial_lr','lr_horizon_updates',
                    'dataset_sha256','objective_source_sha256','augmentation','torch','monai'):
            assert cfg[key]==bc[key],(task['id'],key)
        assert cfg['script_sha256']==digest(frozen/study.CORE)
        assert bc['script_sha256']==digest(frozen/'reference_source/capped_core.py')
        protocol=read(frozen/'protocols'/task['protocol']/'protocol.json')
        assert cfg['protocol_sha256']==digest(frozen/'protocols'/task['protocol']/'protocol.json')
        assert cfg['schedule']==protocol['models']['unet']['objectives'][task['arm']]
        assert cfg['max_steps']==409
        assert protocol['auxiliary_envelope'][-1]==(0. if study.EXPECT_ZERO_FINAL else 1.)
        cal=read(folder/'calibration.json')
        target=protocol['calibration_target_ratio']
        assert cal['target_ratio']==target and cal['p95_cap']==.75 and cal['anchor_successful_updates']==102
        assert [c['case_name'] for c in cal['cases']]==cfg['train_cases']
        assert set(cal['weights'])==set(task['arm'].split('_')[1:])
        assert all(s['weighted_median_ratio']<=target+1e-7 and s['weighted_p95_ratio']<=.7500001 for s in cal['statistics'].values())
        replay=None;previous=persistent.get((task['arm'],*pair))
        if previous:
            original=read(previous/'calibration.json')
            for key in ('anchor_model_sha256','weights','statistics','cases'):
                assert cal[key]==original[key],(task['id'],'persistent calibration',key)
            replay=prefix_check(folder,previous,task['taper_start']+1)
        unconstrained_prefix=cfg['schedule']['sdf_prefix_updates'] if task['arm']=='dice_sdf' else 102
        rows.append(dict(**task,baseline=baseline,current=current,
            prefix_replay=prefix_check(folder,reference,unconstrained_prefix),persistent_prefix_replay=replay,
            gain_pp=current['validation_percent']-baseline['validation_percent'],
            training_gain_pp=current['training_percent']-baseline['training_percent'],
            union_gain_pp=current['union_percent']-baseline['union_percent'],
            subfield_gain_pp={k:current['per_subfield_percent'][k]-baseline['per_subfield_percent'][k]
                              for k in baseline['per_subfield_percent']},
            error_delta={k:current[k]-baseline[k] for k in ('fn','fp','swaps')},calibration_weights=cal['weights']))
    selected=select(plan['tasks'],rows,study.KEYS)
    summary=[]
    for c in study.CANDIDATES:
        group=[r for r in rows if r['candidate']==c['key']]
        summary.append(dict(candidate=c['key'],arm=c['arm'],all_pairs=stats(group),
            rejected=any(r['gain_pp']<=0 for r in group),all_nine_improved=len(group)==9 and all(r['gain_pp']>0 for r in group)))
    live=read(report/'live.json').get('status.json',{})
    complete=live.get('status')=='complete' and live.get('completed')==len(rows)
    if complete:
        assert selected['complete'] and live['selected']['winner']==selected['selected']['winner']
    result=dict(status='complete' if complete else 'partial',outcome=selected['outcome'] if complete else 'running',
        winner=selected['selected']['winner'],rows=rows,summary=summary,
        scope=SCOPE,
        caveat='Adaptive development-validation selection across overlapping folds, not untouched-test generalization.',
        analysis_source_sha256=digest(Path(__file__)))
    write(report/RESULT_FILENAME,result)
    flat=[{k:r[k] for k in ('candidate','seed','fold','gain_pp','training_gain_pp','union_gain_pp')} for r in rows]
    if flat:
        with (report/'paired_results.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print(json.dumps(dict(status=result['status'],outcome=result['outcome'],winner=result['winner'],summary=summary)))
    if complete:
        from scripts.mni_constraints.analyze_unetr_delayed_combinations import plot
        plot(report,[dict(r,arm=r['candidate']) for r in rows],arms=study.KEYS,pairs=study.PAIRS,
             title=TITLE,stem=PLOT_STEM)


if __name__=='__main__':main()
