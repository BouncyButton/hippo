"""Audit and compare matched 10%/40% ADNI nnUNet calibration endpoints."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from verified_archive import read, digest, write, diagnostics
from heavy_calibration_pilot import NAME, PARENT, BANDS, CAP


def history(path):
    with (path/'metrics.csv').open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    root = Path.cwd()
    report = root/'reports'/NAME
    plan = read(report/'plan.json')
    prior = root/'reports'/PARENT/'comparison_20261013/results/nnunet'
    baseline = prior/'dice/fold0/seed0/run'
    baseline_dice = read(baseline/'validation_latest.json')['summary']['hippocampus_hard_dice']
    diagnostics(baseline)
    baseline_errors = read(baseline/'evaluation_diagnostics.json')['evaluations']['latest']
    rows, pending = [], []
    for arm in plan['references']:
        if arm == 'dice': continue
        old = prior/arm/'fold0/seed0/run'
        new = (root/'reports'/BANDS/'results/bands_target40') if arm == 'dice_bands' else report/'results'/arm
        assert digest(old/'completion.json') == plan['references'][arm]['completion_sha256']
        assert digest(old/'validation_latest.json') == plan['references'][arm]['validation_sha256']
        if args.allow_partial and not (new/'cleanup.json').exists():
            pending.append(arm)
            continue
        manifest = read(new/'archive_manifest.json')
        for name,item in manifest['metadata'].items(): assert digest(new/name)==item['sha256'],name
        assert read(new/'cleanup.json')['status']=='complete'
        completion = read(new/'completion.json')
        original_completion = read(old/'completion.json')
        for key in ('consumed_training_flops', 'calibration_flops', 'updates', 'budget_scope'):
            assert completion[key] == original_completion[key], (arm, key)
        assert completion['consumed_training_flops'] <= CAP and completion['discarded_attempts']==0
        config, original_config = read(new/'config.json'), read(old/'config.json')
        assert config['augmentation']=='none'
        for key in ('train_cases', 'validation_cases', 'fold', 'seed', 'dataset_sha256', 'schedule',
                    'lr_horizon_updates', 'precision', 'initial_lr', 'nnunet_sgd_momentum'):
            assert config[key] == original_config[key], (arm, key)
        assert read(new/'calibration.json')['target_ratio']==.4
        if arm=='dice_bands': assert digest(new/'completion.json')==plan['reused_bands40']['completion_sha256']
        final = read(new/'validation_latest.json')['summary']['hippocampus_hard_dice']
        error = read(new/'evaluation_diagnostics.json')['evaluations']['latest']
        diagnostics(old)
        old_error = read(old/'evaluation_diagnostics.json')['evaluations']['latest']
        rows.append(dict(objective=arm, label=arm.removeprefix('dice_').replace('_',' + ').replace('sdf','signed distance').replace('depth','depth CE'),
            original_dice=plan['references'][arm]['final_dice'], heavy_dice=final,
            gain_vs_baseline_pp=100*(final-baseline_dice),
            gain_vs_original_pp=100*(final-plan['references'][arm]['final_dice']),
            best_dice=completion['best_validation_hard_dice'], selected_update=completion['selected_step'],
            stopping_update=completion['updates'], optimization_gflops=completion['consumed_training_flops']/1e9,
            calibration_gflops=completion['calibration_flops']/1e9,
            fn_voxels=error['totals']['fn_voxels'],fp_voxels=error['totals']['fp_voxels'],
            original_fn_voxels=old_error['totals']['fn_voxels'],original_fp_voxels=old_error['totals']['fp_voxels'],
            fn_rate=error['fn_rate'],fp_fraction=error['fp_fraction'],reused=(arm=='dice_bands')))
    assert len(rows)+len(pending)==13 and (args.allow_partial or len(rows)==13)
    winner=max(rows,key=lambda r:r['heavy_dice'])
    comparison=dict(status='in_progress' if pending else 'complete',pending=pending,
        completed_conditions=len(rows),total_conditions=13,
        baseline_dice=baseline_dice,baseline_errors=baseline_errors['totals'],
        target_ratio=.4,original_target_ratio=.1,p95_cap=.5,
        improves_over_own_original=sum(r['gain_vs_original_pp']>0 for r in rows),
        improves_over_baseline=sum(r['gain_vs_baseline_pp']>0 for r in rows),
        best=winner,conditions=rows,
        limitations=['One fold, one seed, five training subjects; exploratory selection among 13 variants.',
                     'Optimization-only FLOPs matched; calibration and evaluation tracked separately.',
                     'Binary whole-hippocampus ADNI; anterior/posterior swaps not applicable.'])
    prefix = 'interim_' if args.allow_partial else ''
    write(report/(prefix+'comparison.json'),comparison)
    with (report/(prefix+'comparison.csv')).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(15,7.5),gridspec_kw={'width_ratios':[1.2,1]})
    y=np.arange(len(rows))
    for i,row in enumerate(rows):
        axes[0].plot([100*row['original_dice'],100*row['heavy_dice']],[i,i],color='#c2cbd4',linewidth=2,zorder=1)
    axes[0].scatter([100*r['original_dice'] for r in rows],y,s=50,color='#899cac',label='10% target',zorder=2)
    axes[0].scatter([100*r['heavy_dice'] for r in rows],y,s=60,color='#176e83',label='40% target',zorder=3)
    axes[0].axvline(100*baseline_dice,color='#b23b36',linestyle='--',label='Unaugmented baseline')
    axes[0].set_yticks(y,[r['label'] for r in rows]);axes[0].invert_yaxis()
    values=[100*r[k] for r in rows for k in ('original_dice','heavy_dice')]
    axes[0].set_xlim(min(values)-.7,max(values)+.7)
    axes[0].set_xlabel('Final mean validation Dice (%)')
    axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.10),ncol=2,fontsize=9)
    chosen=['dice', 'dice_bands',winner['objective']]
    chosen=list(dict.fromkeys(chosen))
    colors=['#777777','#176e83','#a34c16']
    for arm,color in zip(chosen,colors):
        label=arm.removeprefix('dice_').replace('_',' + ').replace('depth','depth CE')
        old=prior/arm/'fold0/seed0/run'
        h=history(old)
        axes[1].plot([float(r['estimated_training_flops'])/1e12 for r in h],
            [100*float(r['macro_hard_dice']) for r in h],color=color,
            linestyle='-' if arm=='dice' else '--',label='Baseline' if arm=='dice' else label+' 10%')
        if arm!='dice':
            new=(root/'reports'/BANDS/'results/bands_target40') if arm=='dice_bands' else report/'results'/arm
            h=history(new)
            axes[1].plot([float(r['estimated_training_flops'])/1e12 for r in h],
                [100*float(r['macro_hard_dice']) for r in h],color=color,label=label+' 40%')
    axes[1].set_xlabel('Optimization compute (TFLOPs)');axes[1].set_ylabel('Mean validation Dice (%)')
    axes[1].legend(fontsize=8,loc='lower right');axes[1].grid(alpha=.15)
    status = f' · Interim: {len(rows)}/13 constraint conditions' if pending else ''
    fig.suptitle('ADNI · nnUNet · 5% · fold 0 · seed 0 · no augmentation'+status+'\n40% sequential calibration, p95 ceiling 0.5 · same 31,976 GFLOPs optimization cap')
    fig.tight_layout();fig.savefig(report/(prefix+'calibration_comparison.png'),dpi=180);fig.savefig(report/(prefix+'calibration_comparison.pdf'))
    plt.close(fig)
    print(json.dumps({k:v for k,v in comparison.items() if k!='conditions'},indent=2))


if __name__=='__main__': main()
