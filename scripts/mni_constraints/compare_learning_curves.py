"""Compare archived validation curves at common optimization-FLOP fractions."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
NAME='mni_constraint_target_search_f0s0_20261107'
OUT=ROOT/'reports/mni_bands_learning_curves_20261107'
BASES={'nnunet':'mni_nnunet_cap8_full_20261106',
       'swinunetr':'mni_swin_log_dice_cap8_full_20261106',
       'unetr_vit_b16':'mni_unetr_geo_cap8_20261106',
       'basic_unet16':'mni_unet_cap8_20261106'}
LABELS={'nnunet':'nnUNet','swinunetr':'SwinUNETR','unetr_vit_b16':'UNETR','basic_unet16':'UNet'}
GRID=np.arange(15,96,5)/100


def read(p):return json.loads(p.read_text())


def curve(folder):
    path=folder/'metrics.csv';sha=hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha==read(folder/'archive_manifest.json')['metadata']['metrics.csv']['sha256']
    with path.open() as handle:rows=list(csv.DictReader(handle))
    # A resumed run may repeat an evaluation at the same charged update.
    by_step={int(r['charged_attempts']):r for r in rows}
    rows=[by_step[k] for k in sorted(by_step)]
    x=np.array([float(r['compute_fraction']) for r in rows])
    y=np.array([100*float(r['macro_hard_dice']) for r in rows])
    assert (np.diff(x)>0).all() and x[0]<=GRID[0] and x[-1]>=GRID[-1]
    final=100*read(folder/'validation_latest.json')['summary']['macro_hard_dice']
    assert abs(y[-1]-final)<1e-10
    return dict(x=x,y=y,source=str(path.relative_to(ROOT)),sha256=sha,
                steps=[int(r['charged_attempts']) for r in rows])


def paired(folder, baseline):
    a,b=curve(folder),curve(baseline)
    ay,by=np.interp(GRID,a['x'],a['y']),np.interp(GRID,b['x'],b['y'])
    delta=ay-by;i=int(delta.argmax())
    return dict(fractions=GRID.tolist(),baseline_dice=by.tolist(),bands_dice=ay.tolist(),
        delta_pp=delta.tolist(),final_delta_pp=float(a['y'][-1]-b['y'][-1]),
        peak_sampled_delta_pp=float(delta[i]),peak_fraction=float(GRID[i]),
        mean_sampled_gain_15_to_40=float(delta[GRID<=.4].mean()),
        positive_samples=int((delta>0).sum()),samples=len(GRID),
        final_baseline_dice=float(b['y'][-1]),final_bands_dice=float(a['y'][-1]),
        raw_baseline={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in b.items()},
        raw_bands={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in a.items()})


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    stage=ROOT/'reports'/NAME
    baseline={m:ROOT/'reports'/n/'runs'/f'MNI_{m}_f0s0' for m,n in BASES.items()}
    tasks=[dict(model=r['model'],target=r['target'],folder=Path(r['archive']),reused=True)
           for r in read(stage/'frozen/constraint_reuse.json')]
    for folder in sorted((stage/'runs').glob('MNI*')):
        if not (folder/'cleanup.json').exists():continue
        c=read(folder/'config.json')
        if c['objective']!='dice_bands':continue
        tasks.append(dict(model=c['model_identity'],target=read(folder/'calibration.json')['target_ratio'],
                          folder=folder,reused=False))
    pilot=[]
    for task in sorted(tasks,key=lambda r:(r['model'],r['target'])):
        pilot.append(dict(model=task['model'],target=task['target'],reused=task['reused'],
                          **paired(task['folder'],baseline[task['model']])))
    historical={}
    for model,name in BASES.items():
        rows=[]
        for fold in range(3):
            for seed in range(3):
                p=ROOT/'reports/mni_constraints14_cap8_20261107/runs'/f'MNI_{model}_dice_bands_f{fold}s{seed}'
                b=ROOT/'reports'/name/'runs'/f'MNI_{model}_f{fold}s{seed}'
                rows.append(dict(fold=fold,seed=seed,**paired(p,b)))
        deltas=np.array([r['delta_pp'] for r in rows]);avg=deltas.mean(0);i=int(avg.argmax())
        historical[model]=dict(target=.1,runs=rows,mean_delta_pp=avg.tolist(),
            sd_delta_pp=deltas.std(0,ddof=1).tolist(),
            mean_final_delta_pp=float(np.mean([r['final_delta_pp'] for r in rows])),
            peak_mean_delta_pp=float(avg[i]),peak_fraction=float(GRID[i]),
            positive_runs_at_peak=int((deltas[:,i]>0).sum()))
    result=dict(status='verified',metric='Validation foreground-subfield macro hard Dice, percentage points',
        matching='Linear interpolation on actual optimization FLOPs at 15%,20%,...,95% of the shared cap; final scores use actual final-budget checkpoints without interpolation. Calibration FLOPs excluded consistently with the experiment cap.',
        caveats=['Saved baseline milestones round down and constrained milestones round up; common-fraction values are estimates between nearby recorded checkpoints.',
                 'Peak gains are descriptive post-hoc maxima, not independent evidence or an approved stopping rule.',
                 'Current target sweep is fold0/seed0; separate historical10% analysis includes all3folds and3seeds.'],
        pilot_runs=pilot,historical_target10=historical)
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={.05:'#9a5b12',.1:'#147d92',.2:'#a0459d',.4:'#5965bb',.6:'#d45c36',.8:'#3a8a55'}
    fig,axes=plt.subplots(2,2,figsize=(11,7.5),sharex=True)
    for ax,(model,label) in zip(axes.flat,LABELS.items()):
        ax.axhline(0,color='#263238',lw=1)
        ax.axvspan(10,20,color='#dde4eb',alpha=.5)
        for r in pilot:
            if r['model']!=model:continue
            ax.plot([*GRID*100,100],[*r['delta_pp'],r['final_delta_pp']],
                color=colors[r['target']],marker='o',markersize=3,lw=1.7,label=f"{r['target']:.0%} target")
        ax.set_title(label,loc='left',fontweight='bold');ax.set_xlim(10,101)
        ax.set_ylabel('Dice gain over baseline (pp)');ax.grid(alpha=.17)
        ax.legend(fontsize=8,frameon=False)
    for ax in axes[-1]:ax.set_xlabel('Optimization FLOP budget used (%)')
    fig.suptitle('MNI bands: does an early advantage survive?',fontsize=16,fontweight='bold')
    fig.text(.5,.018,'Fold 0 / seed 0 · bands only · completed runs · matched-FLOP interpolation · shaded area: constraint ramp',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.95));fig.savefig(OUT/'pilot_gains.png',dpi=180);fig.savefig(OUT/'pilot_gains.pdf');plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,4.5))
    for model,color in zip(BASES,['#d45c36','#147d92','#a0459d','#5965bb']):
        r=historical[model]
        ax.plot([*GRID*100,100],[*r['mean_delta_pp'],r['mean_final_delta_pp']],marker='o',markersize=3,color=color,label=LABELS[model])
    ax.axhline(0,color='#263238',lw=1);ax.grid(alpha=.17);ax.legend(frameon=False,ncol=4)
    ax.set(xlabel='Optimization FLOP budget used (%)',ylabel='Mean paired Dice gain (pp)',
           title='Earlier MNI study: 10% bands target across all 3 folds × 3 seeds')
    fig.tight_layout();fig.savefig(OUT/'nine_run_mean_gains.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(status='verified',completed_pilot_curves=len(pilot),historical_curves=36,output=str(OUT)),indent=2))


if __name__=='__main__':main()
