"""Archive-verified paired outcomes for the adaptive combination search."""
import csv
import json
from statistics import mean
from pathlib import Path
from scripts.cross_dataset_campaign.common import read,write,digest
from scripts.mni_constraints.unetr_delayed_combinations import ROOT,NAME,ARMS,PAIRS,PILOTS
from scripts.mni_constraints.analyze_unetr_constraint_pilot import evaluation


def main():
    report=ROOT/'reports'/NAME;frozen=report/'frozen'
    plan=read(frozen/'plan.json');controls=read(frozen/'controls.json')
    references={(c['seed'],c['fold']):frozen/'controls'/c['name'] for c in controls}
    baselines={pair:evaluation(folder) for pair,folder in references.items()}
    rows=[]
    for task in plan['tasks']:
        folder=report/'runs'/task['id']
        if not (folder/'cleanup.json').exists():continue
        current=evaluation(folder);pair=(task['seed'],task['fold']);baseline=baselines[pair]
        cfg=read(folder/'config.json');bc=read(references[pair]/'config.json')
        for key in ('seed','fold','train_cases','validation_cases','initial_lr','lr_horizon_updates',
                    'dataset_sha256','script_sha256','objective_source_sha256','augmentation','torch','monai'):
            assert cfg[key]==bc[key],(task['id'],key)
        assert cfg['objective']==task['arm'] and cfg['schedule']['prefix_updates']==102
        cal=read(folder/'calibration.json')
        assert cal['target_ratio']==.05 and cal['p95_cap']==.75 and cal['anchor_successful_updates']==102
        assert [c['case_name'] for c in cal['cases']]==cfg['train_cases']
        assert set(cal['weights'])==set(task['arm'].split('_')[1:])
        assert all(s['weighted_median_ratio']<=.0500001 and s['weighted_p95_ratio']<=.7500001 for s in cal['statistics'].values())
        rows.append(dict(**task,baseline=baseline,current=current,
            gain_pp=current['validation_percent']-baseline['validation_percent'],
            training_gain_pp=current['training_percent']-baseline['training_percent'],
            union_gain_pp=current['union_percent']-baseline['union_percent'],
            subfield_gain_pp={k:current['per_subfield_percent'][k]-baseline['per_subfield_percent'][k]
                              for k in baseline['per_subfield_percent']},
            error_delta={k:current[k]-baseline[k] for k in ('fn','fp','swaps')},
            calibration_weights=cal['weights'],calibration_statistics=cal['statistics']))
    summary=[]
    for arm in ARMS:
        group=[r for r in rows if r['arm']==arm];pilots=[r for r in group if r['pilot']]
        remaining=[r for r in group if not r['pilot']]
        def stats(xs):
            return dict(pairs=len(xs),mean_gain_pp=mean(r['gain_pp'] for r in xs) if xs else None,
                minimum_gain_pp=min((r['gain_pp'] for r in xs),default=None),improved_pairs=sum(r['gain_pp']>0 for r in xs))
        summary.append(dict(arm=arm,all_pairs=stats(group),pilots=stats(pilots),new_confirmation=stats(remaining),
                            all_nine_improved=len(group)==9 and all(r['gain_pp']>0 for r in group)))
    live=read(report/'live.json').get('status.json',{})
    complete=live.get('status')=='complete' and live.get('completed')==len(rows)
    winners=[s['arm'] for s in summary if s['all_nine_improved']]
    if complete:
        assert len([r for r in rows if r['pilot']])==12
        assert live['selected']['winner']==(winners[0] if winners else None)
    result=dict(status='complete' if complete else 'partial',outcome='nine_pair_success' if winners else 'exhausted' if complete else 'running',
        winner=winners[0] if winners else None,rows=rows,summary=summary,
        scope='Fixed original combinations,target5%,prefix102,ramp33,SDF nearest5/9 cap; all409 updates at53.305T.',
        caveats=['Four development pilot pairs select candidates; five additional pairs confirm; folds overlap.',
                 'All nine improvement is the user criterion, not an untouched-test statistical claim.',
                 'No seed/fold-specific recipes or result-dependent seed replacement.'],
        analysis_source_sha256=digest(Path(__file__)))
    write(report/'combination_results.json',result)
    flat=[dict(arm=r['arm'],seed=r['seed'],fold=r['fold'],pilot=r['pilot'],
        baseline_dice=r['baseline']['validation_percent'],constraint_dice=r['current']['validation_percent'],
        gain_pp=r['gain_pp'],training_gain_pp=r['training_gain_pp'],union_gain_pp=r['union_gain_pp'],
        fn_delta=r['error_delta']['fn'],fp_delta=r['error_delta']['fp'],swaps_delta=r['error_delta']['swaps']) for r in rows]
    if flat:
        with (report/'paired_results.csv').open('w') as file:
            writer=csv.DictWriter(file,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print(json.dumps(dict(status=result['status'],outcome=result['outcome'],winner=result['winner'],summary=summary)))
    if complete:plot(report,rows)


def plot(report,rows,*,arms=None,pairs=None,title=None,stem='combination_results'):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    arms=ARMS if arms is None else arms
    pairs=PAIRS if pairs is None else pairs
    values=np.full((len(arms),len(pairs)),np.nan)
    for r in rows:values[arms.index(r['arm']),pairs.index((r['seed'],r['fold']))]=r['gain_pp']
    limit=max(.1,float(np.nanmax(np.abs(values))))
    fig,ax=plt.subplots(figsize=(12,4))
    cmap=plt.get_cmap('RdBu').copy();cmap.set_bad('#eeeeee')
    im=ax.imshow(values,cmap=cmap,norm=TwoSlopeNorm(0,vmin=-limit,vmax=limit),aspect='auto')
    for i in range(len(arms)):
        for j in range(len(pairs)):
            v=values[i,j];ax.text(j,i,f'{v:+.2f}' if np.isfinite(v) else '—',ha='center',va='center',
                                 color='white' if np.isfinite(v) and abs(v)>.6*limit else 'black')
    ax.set_xticks(range(len(pairs)),[f's{s}/f{f}'+('*' if (s,f) in PILOTS else '') for s,f in pairs])
    ax.set_yticks(range(len(arms)),[a.replace('dice_','').replace('_',' + ') for a in arms])
    ax.set_title(title or 'UNETR original combinations: Dice gain vs matched baseline\n* development pilot; blank cells not run')
    fig.colorbar(im,ax=ax,label='Dice points');fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(report/f'{stem}.{suffix}',dpi=150)
    plt.close(fig)


if __name__=='__main__':main()
