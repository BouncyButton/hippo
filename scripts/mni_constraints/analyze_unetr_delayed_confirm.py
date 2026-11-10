"""Paired, archive-verified results for the frozen UNETR confirmation."""
import csv
import json
from statistics import mean, median, stdev

from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.mni_constraints.unetr_delayed_confirm import ROOT, NAME, PARENT, PILOT, SEEDS, CAP
from scripts.mni_constraints.analyze_unetr_constraint_pilot import evaluation


def folder_for(seed, fold, condition):
    if (seed,fold)==(19,0):
        return (ROOT/'reports'/PARENT/'runs/MNI_unetr_vit_b16_dice_f0s19' if condition=='baseline'
                else ROOT/'reports'/PILOT/'runs/MNI_unetr_delayed_t05_f0s19')
    return ROOT/'reports'/NAME/'runs'/f'MNI_unetr_{condition}_f{fold}s{seed}'


def statistics(rows):
    if not rows:
        return dict(pairs=0)
    gains=[r['gain_pp'] for r in rows]
    return dict(pairs=len(rows),baseline_mean=mean(r['baseline']['validation_percent'] for r in rows),
        delayed_mean=mean(r['delayed']['validation_percent'] for r in rows),mean_gain_pp=mean(gains),
        median_gain_pp=median(gains),gain_sd_pp=stdev(gains) if len(rows)>1 else None,
        improved_pairs=sum(g>0 for g in gains),gain_range_pp=[min(gains),max(gains)],
        mean_union_gain_pp=mean(r['union_gain_pp'] for r in rows),
        total_error_delta={k:sum(r['error_delta'][k] for r in rows) for k in ('fn','fp','swaps')})


def main():
    report=ROOT/'reports'/NAME
    rows=[]
    for seed in SEEDS:
        for fold in range(3):
            baseline_folder=folder_for(seed,fold,'baseline')
            constrained_folder=folder_for(seed,fold,'delayed_t05')
            if not all((f/'cleanup.json').exists() for f in (baseline_folder,constrained_folder)):
                continue
            baseline,delayed=evaluation(baseline_folder),evaluation(constrained_folder)
            bc,dc=read(baseline_folder/'config.json'),read(constrained_folder/'config.json')
            assert bc['seed']==dc['seed']==seed and bc['fold']==dc['fold']==fold
            for key in ('train_cases','validation_cases','initial_lr','lr_horizon_updates','dataset_sha256',
                        'script_sha256','objective_source_sha256','augmentation','torch','monai'):
                assert bc[key]==dc[key],(seed,fold,key)
            assert len(bc['train_cases'])==20 and len(bc['validation_cases'])==10
            s=dc['schedule'];assert s['prefix_updates']==102 and s['constraint_warmup_updates']==33
            assert bc['initial_lr']==.0006 and bc['lr_horizon_updates']==409
            cal=read(constrained_folder/'calibration.json')
            assert cal['target_ratio']==.05 and cal['p95_cap']==.75
            assert cal['anchor_successful_updates']==cal['anchor_training_attempts']==102
            assert [r['case_name'] for r in cal['cases']]==dc['train_cases']
            assert cal['statistics']['bands']['weighted_median_ratio']<=.0500001
            assert cal['statistics']['bands']['weighted_p95_ratio']<=.7500001
            rows.append(dict(seed=seed,fold=fold,pilot_reused=(seed,fold)==(19,0),
                baseline=baseline,delayed=delayed,
                gain_pp=delayed['validation_percent']-baseline['validation_percent'],
                training_gain_pp=delayed['training_percent']-baseline['training_percent'],
                union_gain_pp=delayed['union_percent']-baseline['union_percent'],
                subfield_gain_pp={k:delayed['per_subfield_percent'][k]-baseline['per_subfield_percent'][k]
                                  for k in baseline['per_subfield_percent']},
                error_delta={k:delayed[k]-baseline[k] for k in ('fn','fp','swaps')},
                calibration_weight=cal['weights']['bands'],calibration_statistics=cal['statistics']['bands']))
    new=[r for r in rows if not r['pilot_reused']]
    result=dict(status='complete' if len(rows)==9 else 'partial',rows=rows,
        all_nine=statistics(rows),new_confirmation_only=statistics(new),
        by_seed={str(s):statistics([r for r in rows if r['seed']==s]) for s in SEEDS},
        by_fold={str(f):statistics([r for r in rows if r['fold']==f]) for f in range(3)},
        all_nine_improved=len(rows)==9 and all(r['gain_pp']>0 for r in rows),
        recipe=dict(target=.05,p95_cap=.75,prefix_updates=102,ramp_updates=33,
                    updates=409,optimization_cap=CAP,training_subjects=10,training_crops=20),
        limitations=['Seed19/fold0 selected the recipe and is reused; eight new pairs are reported separately.',
                     'Overlapping folds and development validation do not constitute independent test samples.',
                     'This tests original union bands only; no automatic extrapolation to other constraint combinations.'],
        other_sweep_paused=True,analysis_source_sha256=digest(__import__('pathlib').Path(__file__)))
    write(report/'confirmation_results.json',result)
    flat=[dict(seed=r['seed'],fold=r['fold'],pilot_reused=r['pilot_reused'],
        baseline_dice=r['baseline']['validation_percent'],delayed_dice=r['delayed']['validation_percent'],
        gain_pp=r['gain_pp'],training_gain_pp=r['training_gain_pp'],union_gain_pp=r['union_gain_pp'],
        fn_delta=r['error_delta']['fn'],fp_delta=r['error_delta']['fp'],swaps_delta=r['error_delta']['swaps'],
        calibration_weight=r['calibration_weight']) for r in rows]
    if flat:
        with (report/'paired_results.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print(json.dumps(dict(status=result['status'],rows=flat,all_nine=result['all_nine'],
                         new_confirmation_only=result['new_confirmation_only'],all_nine_improved=result['all_nine_improved'])))
    if len(rows)==9:
        plot(report,rows)


def plot(report,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.6))
    labels=[f"s{r['seed']} / f{r['fold']}"+(' *' if r['pilot_reused'] else '') for r in rows]
    x=list(range(len(rows)))
    axes[0].plot(x,[r['baseline']['validation_percent'] for r in rows],marker='o',label='Baseline')
    axes[0].plot(x,[r['delayed']['validation_percent'] for r in rows],marker='o',label='Delayed 5% bands')
    axes[0].set(ylabel='Final validation macro Dice (%)',title='Same 53.305T cap / 409 updates')
    axes[0].legend()
    bars=axes[1].bar(x,[r['gain_pp'] for r in rows],color=['#269579' if r['gain_pp']>0 else '#b25266' for r in rows])
    axes[1].axhline(0,color='black',lw=.8)
    axes[1].bar_label(bars,labels=[f"{r['gain_pp']:+.2f}" for r in rows],padding=3,fontsize=8)
    axes[1].margins(y=.20)
    axes[1].set(ylabel='Delayed bands − baseline (Dice points)',title='Every matched pair retained')
    for ax in axes:
        ax.set_xticks(x,labels,rotation=50,ha='right');ax.grid(axis='y',alpha=.2)
    fig.suptitle('UNETR delayed 5% confirmation · * reused selection pilot')
    fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(report/f'confirmation_results.{suffix}',dpi=150)
    plt.close(fig)


if __name__=='__main__':main()
