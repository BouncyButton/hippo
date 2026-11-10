"""Compare the finite 60% nnUNet sweep with verified 10% and 40% results."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from calibration60_pilot import NAME, PARENT, PREVIOUS, BANDS, CAP
from verified_archive import read, write, digest


def label(arm):
    return arm.removeprefix('dice_').replace('_',' + ').replace('sdf','signed distance').replace('depth','depth CE')


def audit(directory, target):
    manifest = read(directory/'archive_manifest.json')
    for name, item in manifest['metadata'].items():
        assert digest(directory/name) == item['sha256'], (directory,name)
    assert read(directory/'cleanup.json')['status'] == 'complete'
    done, config = read(directory/'completion.json'), read(directory/'config.json')
    cal = read(directory/'calibration.json')
    assert done['status'] == 'complete' and done['discarded_attempts'] == 0
    assert config['augmentation'] == 'none' and config['fold'] == config['seed'] == 0
    assert cal['target_ratio'] == target and cal['p95_cap'] == (.75 if target==.6 else .5)
    assert done['consumed_training_flops'] <= CAP
    return done, config, read(directory/'validation_latest.json')['summary']['hippocampus_hard_dice']


def history(directory):
    with (directory/'metrics.csv').open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    root = Path.cwd()
    report = root/'reports'/NAME
    plan = read(report/'plan.json')
    assert digest(root/'experiments'/PREVIOUS/'manifest.json') == plan['previous_manifest_sha256']
    original = root/'reports'/PARENT/'comparison_20261013/results/nnunet'
    baseline = original/'dice/fold0/seed0/run'
    assert digest(baseline/'completion.json') == plan['references']['dice']['completion_sha256']
    baseline_dice = plan['references']['dice']['final_dice']
    rows, paths = [], {}
    for spec in plan['runs']:
        arm = spec['objective']
        old = original/arm/'fold0/seed0/run'
        forty = (root/'reports'/BANDS/'results/bands_target40') if arm=='dice_bands' else root/'reports'/PREVIOUS/'results'/arm
        sixty = report/'results'/arm
        assert digest(old/'completion.json') == plan['references'][arm]['completion_sha256']
        assert digest(old/'validation_latest.json') == plan['references'][arm]['validation_sha256']
        done10, config10 = read(old/'completion.json'), read(old/'config.json')
        done40, config40, dice40 = audit(forty,.4)
        done60, config60, dice60 = audit(sixty,.6)
        for key in ('consumed_training_flops','calibration_flops','updates','budget_scope'):
            assert done10[key] == done40[key] == done60[key], (arm,key)
        for key in ('train_cases','validation_cases','fold','seed','dataset_sha256','schedule',
                    'lr_horizon_updates','precision','initial_lr','nnunet_sgd_momentum'):
            assert config10[key] == config40[key] == config60[key], (arm,key)
        errors60 = read(sixty/'evaluation_diagnostics.json')['evaluations']['latest']
        errors40 = read(forty/'evaluation_diagnostics.json')['evaluations']['latest']
        rows.append(dict(objective=arm,label=label(arm),dice10=plan['references'][arm]['final_dice'],
            dice40=dice40,dice60=dice60,gain_vs_40_pp=100*(dice60-dice40),
            gain_vs_10_pp=100*(dice60-plan['references'][arm]['final_dice']),
            gain_vs_baseline_pp=100*(dice60-baseline_dice),best_dice60=done60['best_validation_hard_dice'],
            selected_update60=done60['selected_step'],stopping_update=done60['updates'],
            optimization_gflops=done60['consumed_training_flops']/1e9,calibration_gflops=done60['calibration_flops']/1e9,
            fn60=errors60['totals']['fn_voxels'],fp60=errors60['totals']['fp_voxels'],
            fn40=errors40['totals']['fn_voxels'],fp40=errors40['totals']['fp_voxels']))
        paths[arm] = {40:forty,60:sixty}
    assert len(rows)==13
    best60, best40 = max(rows,key=lambda r:r['dice60']), max(rows,key=lambda r:r['dice40'])
    result = dict(status='complete',baseline_dice=baseline_dice,conditions=rows,best60=best60,best40=best40,
        gains_over_40=sum(r['gain_vs_40_pp']>0 for r in rows),gains_over_10=sum(r['gain_vs_10_pp']>0 for r in rows),
        gains_over_baseline=sum(r['gain_vs_baseline_pp']>0 for r in rows),
        protocol='5% ADNI, fold 0, seed 0, no augmentation, original timing and optimizer. 60% median target with p95 ceiling 0.75, versus 40%/0.5 and 10%/0.5.',
        limitations=['Single seed and fold; exploratory comparison across 13 conditions.',
                     'Optimization-only cap; identical per-arm calibration compute is accounted separately.',
                     'Binary whole-hippocampus labels; A/P swaps are not applicable.'])
    write(report/'comparison.json',result)
    with (report/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(16,8.5),gridspec_kw={'width_ratios':[1.25,1]})
    y=np.arange(13)
    for i,row in enumerate(rows):
        axes[0].plot([100*row[k] for k in ('dice10','dice40','dice60')],[i]*3,color='#c2cbd4',linewidth=1.5,zorder=1)
    for key,color,name in [('dice10','#899cac','10%'),('dice40','#176e83','40%'),('dice60','#b65b13','60%')]:
        axes[0].scatter([100*r[key] for r in rows],y,s=50,color=color,label=name,zorder=2)
    axes[0].axvline(100*baseline_dice,color='#b23b36',linestyle='--',label='Baseline')
    axes[0].set_yticks(y,[r['label'] for r in rows]);axes[0].invert_yaxis()
    axes[0].set_xlabel('Final mean validation Dice (%)')
    axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.10),ncol=4,fontsize=9)
    h=history(baseline)
    axes[1].plot([float(r['estimated_training_flops'])/1e12 for r in h],
                 [100*float(r['macro_hard_dice']) for r in h],color='#777777',label='Baseline')
    for arm,color in zip(dict.fromkeys([best40['objective'],best60['objective']]),['#176e83','#b65b13']):
        for target,style in [(40,'--'),(60,'-')]:
            h=history(paths[arm][target])
            axes[1].plot([float(r['estimated_training_flops'])/1e12 for r in h],
                         [100*float(r['macro_hard_dice']) for r in h],color=color,linestyle=style,
                         label=label(arm)+f' {target}%')
    axes[1].set_xlabel('Optimization compute (TFLOPs)');axes[1].set_ylabel('Mean validation Dice (%)')
    axes[1].legend(fontsize=8,loc='lower right');axes[1].grid(alpha=.15)
    fig.suptitle('ADNI · nnUNet · 5% · fold 0 · seed 0 · no augmentation\n10%, 40%, and 60% calibration · same 31,976 GFLOPs optimization cap')
    fig.tight_layout();fig.savefig(report/'calibration_comparison.png',dpi=180);fig.savefig(report/'calibration_comparison.pdf')
    plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k!='conditions'},indent=2))


if __name__=='__main__': main()
