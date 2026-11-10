"""Verify and compare the two bands-strength trials against saved controls."""
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=Path.cwd()
    report=root/'reports/adni_nnunet_bands_feedback_20261013'
    original=root/'reports/adni_constraints_noaug_5pct_fold0_seed0_20261013/comparison_20261013/results/nnunet'
    results=report/'results'
    receipt=read(report/'sweep_completion.json')
    assert receipt['status']=='complete' and receipt['runs']==2
    assert digest(results/'summary.json')==receipt['summary_sha256']
    sources=[('Baseline',0.,original/'dice/fold0/seed0/run'),
             ('Bands 10%',.1,original/'dice_bands/fold0/seed0/run'),
             ('Bands 20%',.2,results/'bands_target20'),
             ('Bands 40%',.4,results/'bands_target40')]
    rows=[];curves=[];reference=None;band_anchor=None
    for name,target,path in sources:
        config,done=read(path/'config.json'),read(path/'completion.json')
        for key,filename in [('config_sha256','config.json'),('metrics_sha256','metrics.csv'),
            ('budget_ledger_sha256','budget_ledger.json'),('calibration_sha256','calibration.json'),
            ('validation_best_sha256','validation_best.json')]:
            assert digest(path/filename)==done[key]
        final=read(path/'validation_latest.json')
        assert digest(path/'validation_latest.json')==read(path/'retention.json')['validation_latest_sha256']
        assert config['augmentation']=='none' and config['fold']==config['seed']==0
        assert done['status']=='complete' and done['updates']==90 and done['discarded_attempts']==0
        assert done['consumed_training_flops']<=31976450667090
        population=(config['train_cases'],config['validation_cases'],config['dataset_sha256'])
        if reference is None:reference=population
        assert population==reference
        with (path/'metrics.csv').open(newline='') as f:
            curve=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
        assert len(curve)==20 and curve[-1]['step']==90
        allin_endpoint=next(r for r in reversed(curve) if r['total_flops']<=31976450667090)
        score=final['summary']['hippocampus_hard_dice']
        assert score==curve[-1]['macro_hard_dice']
        counts=[]
        for case in final['cases']:
            predicted,truth=case['predicted_voxels'],case['target_voxels']
            intersection=round(case['hippocampus_hard_dice']*(predicted+truth)/2)
            counts.append((intersection/predicted if predicted else 0.,intersection/truth,
                           predicted-intersection,truth-intersection))
        precision,recall,false_positive,false_negative=np.mean(counts,axis=0).tolist()
        calibration=read(path/'calibration.json')
        if target:
            assert calibration['anchor_successful_updates']==9 and calibration['target_ratio']==target
            if band_anchor is None:band_anchor=calibration['anchor_model_sha256']
            assert calibration['anchor_model_sha256']==band_anchor
            assert calibration['statistics']['bands']['weighted_p95_ratio']<=.5
        rows.append(dict(name=name,calibration_target=target,final_dice=score,best_dice=done['best_validation_hard_dice'],
            selected_step=done['selected_step'],updates=90,training_gflops=done['consumed_training_flops']/1e9,
            calibration_gflops=done['calibration_flops']/1e9,
            band_weight=calibration.get('weights',{}).get('bands',0),
            mean_precision=precision,mean_recall=recall,
            mean_false_positive_voxels=false_positive,mean_false_negative_voxels=false_negative,
            last_recorded_training_plus_calibration_cap_step=int(allin_endpoint['step']),
            last_recorded_training_plus_calibration_cap_dice=allin_endpoint['macro_hard_dice'],
            final_gain_vs_baseline_pp=100*(score-rows[0]['final_dice']) if rows else 0,
            validation_sha256=digest(path/'validation_latest.json')))
        curves.append(curve)
    for row in rows:row['gain_vs_original_bands_pp']=100*(row['final_dice']-rows[1]['final_dice'])
    for curve in curves:assert [r['step'] for r in curve]==[r['step'] for r in curves[0]]
    comparison=dict(status='verified',job_id=read(report/'submission.json')['job_id'],runs=rows,
        conclusion='Exploratory single-seed strength ablation; unchanged MSD 9/9 timing, subset, optimizer, learning-rate horizon, and binary loss.',
        optimization_cap_gflops=31976.45066709,calibration_excluded_from_cap=True)
    (report/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    os.environ.setdefault('MPLCONFIGDIR','/private/tmp/hippo-adni-comparison-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,5.7))
    fig.subplots_adjust(top=.74,bottom=.19,left=.07,right=.97,wspace=.24)
    colors=['#374151','#159895','#db8618','#be4257']
    for i,(row,curve,color) in enumerate(zip(rows,curves,colors)):
        x=[r['estimated_training_flops']/1e12 for r in curve]
        y=[100*r['macro_hard_dice'] for r in curve]
        axes[0].plot(x,y,color=color,lw=2,label=f'{row["name"]}: {100*row["final_dice"]:.2f}%')
        if i:
            gain=100*(np.array(y)/100-np.array([r['macro_hard_dice'] for r in curves[0]]))
            axes[1].plot([r['step'] for r in curve],gain,color=color,lw=2,label=row['name'])
    axes[0].set(xlabel='Optimization compute (1,000 GFLOPs)',ylabel='Validation Dice (%)',ylim=(0,85))
    axes[1].set(xlabel='Optimizer updates',ylabel='Dice gain vs baseline (percentage points)')
    axes[1].axhline(0,color='#9ca3af',lw=1)
    for ax in axes:ax.grid(alpha=.18);ax.legend(frameon=False,fontsize=9)
    fig.suptitle('ADNI nnUNet: does stronger bands feedback help?',x=.07,y=.96,ha='left',fontsize=17,fontweight='bold')
    fig.text(.07,.875,'5% data · fold 0 · seed 0 · no augmentation\nSame MSD timing: calibrate at update 9, activate at 10, full weight at 18. Only the bands target changes.',va='top',fontsize=10.5,linespacing=1.5)
    fig.text(.07,.055,'All conditions complete 90 updates below the 31,976.45 GFLOPs optimization cap; calibration is additional.\nPercentages name the initial median auxiliary/native logit-gradient target, with a fixed 0.5 p95 ceiling.',fontsize=9,linespacing=1.5)
    fig.savefig(report/'bands_strength_comparison.png',dpi=180)
    fig.savefig(report/'bands_strength_comparison.pdf')
    plt.close(fig)
    print(json.dumps(comparison,indent=2))


if __name__=='__main__':main()
