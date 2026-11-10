"""Audit downloaded ADNI records and plot final-budget constraint comparisons."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

MODELS = {
    'swinunetr': 'SwinUNETR',
    'unetr_vit_b16': 'UNETR',
    'basic_unet16': 'UNet3D',
    'nnunet': 'nnUNet',
}
OBJECTIVES = [
    'dice', 'dice_bands', 'dice_sdf', 'dice_bands_sdf',
    'dice_bands_edge', 'dice_bands_edge_sdf', 'dice_depth',
    'dice_bands_depth', 'dice_edge_depth', 'dice_sdf_depth',
    'dice_bands_edge_depth', 'dice_bands_sdf_depth',
    'dice_edge_sdf_depth', 'dice_bands_edge_sdf_depth',
]
TERMS = {'bands': 'Bands', 'sdf': 'Signed distance',
         'edge': 'Original edge', 'depth': 'Depth CE'}
CAP = 31976450667090


def label(objective):
    return 'Baseline' if objective == 'dice' else ' + '.join(
        TERMS[x] for x in objective.split('_')[1:])


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.results, args.output
    output.mkdir(parents=True, exist_ok=True)
    summary = read(root / 'summary.json')
    if (output/'sweep_completion.json').exists():
        receipt=read(output/'sweep_completion.json')
        assert receipt['status']=='complete' and receipt['completed_runs']==56
        assert sha(root/'summary.json')==receipt['summary_sha256']
        assert summary['status']=='complete' and summary['completed_runs']==56
    rows, curves, cases = [], {}, {}
    common_identity = None
    for source in summary['runs']:
        model, objective = source['model'], source['objective']
        path = root / model / objective / 'fold0/seed0/run'
        config, done = read(path / 'config.json'), read(path / 'completion.json')
        assert sha(path / 'completion.json') == source['completion_sha256']
        for key, filename in {
            'config_sha256': 'config.json', 'metrics_sha256': 'metrics.csv',
            'budget_ledger_sha256': 'budget_ledger.json',
            'validation_best_sha256': 'validation_best.json',
            'calibration_sha256': 'calibration.json',
        }.items():
            assert sha(path / filename) == done[key], (model, objective, key)
        assert sha(path / 'validation_latest.json') == source['validation_latest_sha256']
        assert config['augmentation'] == 'none' and config['fold'] == config['seed'] == 0
        assert config['model_identity'] == model and config['objective'] == objective
        assert not config['smoke_test'] and config['budget_scope'] == 'optimization_only'
        assert config['budget_flops'] == done['budget_flops'] == CAP
        assert done['status'] == 'complete' and done['discarded_attempts'] == 0
        assert done['updates'] == config['max_steps']
        assert done['consumed_training_flops'] == config['schedule']['consumed_training_flops'] <= CAP
        assert done['consumed_total_flops'] == done['consumed_training_flops'] + done['calibration_flops']
        identity = (config['dataset_sha256'], tuple(config['train_cases']), tuple(config['validation_cases']))
        if common_identity is None:
            common_identity = identity
        assert identity == common_identity
        assert len(config['train_cases']) == 10 and len(config['validation_cases']) == 54
        with (path / 'metrics.csv').open(newline='') as stream:
            curve = [{k: float(v) for k, v in x.items()} for x in csv.DictReader(stream)]
        assert len(curve) == 20 and curve[-1]['step'] == done['updates']
        assert all(a['step'] < b['step'] for a, b in zip(curve, curve[1:]))
        final = read(path / 'validation_latest.json')
        case_rows = final['cases']
        assert {x['case_name'] for x in case_rows} == set(config['validation_cases'])
        score = float(np.mean([x['hippocampus_hard_dice'] for x in case_rows]))
        assert abs(score - source['final_dice']) < 1e-12
        assert abs(score - curve[-1]['macro_hard_dice']) < 1e-12
        assert abs(max(x['macro_hard_dice'] for x in curve) - source['best_dice']) < 1e-12
        assert curve[-1]['estimated_training_flops'] == done['consumed_training_flops']
        replay_path = path / 'recovery_provenance.json'
        replay = read(replay_path)['extra_training_flops'] if replay_path.exists() else 0
        assert replay / 1e9 == source['recovery_extra_gflops']
        # Conservative sensitivity: last recorded validation within the cap when
        # calibration and any replay are also charged. No interpolation is used.
        eligible = [x for x in curve if x['total_flops'] +
                    (replay if x['step'] > 147 else 0) <= CAP]
        last_allin = eligible[-1]
        row = dict(model=model, model_label=MODELS[model], objective=objective,
                   constraint=label(objective), final_dice=score,
                   best_dice=source['best_dice'], selected_step=source['selected_step'],
                   updates=done['updates'], optimization_gflops=done['consumed_training_flops']/1e9,
                   calibration_gflops=done['calibration_flops']/1e9,
                   replay_gflops=replay/1e9,
                   training_plus_calibration_plus_replay_gflops=(done['consumed_total_flops']+replay)/1e9,
                   last_recorded_allin_cap_step=int(last_allin['step']),
                   last_recorded_allin_cap_dice=last_allin['macro_hard_dice'],
                   last_recorded_allin_cap_gflops=(last_allin['total_flops']+
                       (replay if last_allin['step']>147 else 0))/1e9)
        rows.append(row)
        curves[model, objective] = curve
        grouped = {}
        for case in case_rows:
            grouped.setdefault(case['subject_id'], []).append(case['hippocampus_hard_dice'])
        assert len(grouped) == 27 and all(len(v) == 2 for v in grouped.values())
        cases[model, objective] = {k: np.mean(v) for k, v in grouped.items()}

    assert len(rows) == summary['completed_runs'] == len({(r['model'],r['objective']) for r in rows})
    if summary['status']=='complete':
        assert {(r['model'],r['objective']) for r in rows} == {(m,o) for m in MODELS for o in OBJECTIVES}
    model_summaries = []
    for model in MODELS:
        own = [r for r in rows if r['model'] == model]
        baseline = next(r for r in own if r['objective'] == 'dice')
        for row in own:
            row['final_gain_pp'] = 100*(row['final_dice']-baseline['final_dice'])
            row['best_gain_pp'] = 100*(row['best_dice']-baseline['best_dice'])
            row['last_recorded_allin_cap_gain_pp'] = 100*(row['last_recorded_allin_cap_dice']-baseline['final_dice'])
            subject = cases[model, row['objective']]
            delta = [subject[s]-v for s,v in cases[model, 'dice'].items()]
            row['subjects_improved'] = int(sum(x>0 for x in delta))
            row['subjects_total'] = 27
        constrained = [r for r in own if r['objective'] != 'dice']
        winner = max(constrained, key=lambda r:r['final_dice'])
        best_selected = max(constrained, key=lambda r:r['best_dice'])
        band_rows = [r for r in constrained if 'bands' in r['objective']]
        model_summaries.append(dict(model=model, baseline=baseline, winner=winner,
            best_selected_checkpoint=best_selected,
            constraints_completed=len(constrained),
            positive_conditions=sum(r['final_gain_pp']>0 for r in constrained),
            bands_gain_range_pp=[min(r['final_gain_pp'] for r in band_rows),max(r['final_gain_pp'] for r in band_rows)],
            training_gflops_range=[min(r['optimization_gflops'] for r in own),max(r['optimization_gflops'] for r in own)],
            calibration_gflops_range=[min(r['calibration_gflops'] for r in constrained),max(r['calibration_gflops'] for r in constrained)]))
    results = dict(completed_runs=len(rows), total_runs=56, training_cap_gflops=CAP/1e9,
        scope='optimization_only', augmentation='none', fold=0, seed=0,
        train_crops=10, validation_crops=54, validation_subjects=27,
        primary_metric='Final-budget mean whole-hippocampus hard Dice across 54 validation crops',
        selection_caveat='Winners chosen from 13 variants on this validation fold; no independent test or seed replication.',
        allin_sensitivity_caveat='Last recorded validation below cap including calibration/replay; coarse 20-point validation grid, unequal unused budgets; excludes evaluation and profiling costs.',
        summary_sha256=sha(root/'summary.json'), models=model_summaries, runs=rows)
    (output/'comparison.json').write_text(json.dumps(results,indent=2)+'\n')
    with (output/'all_conditions.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

    os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/hippo-adni-comparison-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    objectives=OBJECTIVES[1:]
    matrix=np.full((len(objectives),len(MODELS)),np.nan)
    for j,m in enumerate(MODELS):
        for i,o in enumerate(objectives):
            r=next((r for r in rows if r['model']==m and r['objective']==o),None)
            if r:matrix[i,j]=r['final_gain_pp']
    fig,ax=plt.subplots(figsize=(11.8,8.6))
    fig.subplots_adjust(left=.49,right=.93,top=.80,bottom=.13)
    lim=max(1,float(np.nanmax(np.abs(matrix))))
    im=ax.imshow(matrix,cmap='RdBu',norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),aspect='auto')
    ax.set_xticks(range(4),list(MODELS.values()));ax.xaxis.tick_top()
    ax.set_yticks(range(len(objectives)),[label(o) for o in objectives])
    ax.tick_params(length=0,pad=9)
    for i in range(len(objectives)):
        for j in range(4):
            v=matrix[i,j]
            text='Pending' if np.isnan(v) else f'{v:+.2f}'
            if list(MODELS)[j]=='unetr_vit_b16' and objectives[i]=='dice_edge_depth':text+='*'
            ax.text(j,i,text,ha='center',va='center',color='white' if abs(v)>lim*.66 else '#111827',fontsize=11)
    ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,len(objectives),1),minor=True)
    ax.grid(which='minor',color='white',linewidth=2);ax.tick_params(which='minor',length=0)
    fig.suptitle('ADNI: constraint benefit at the training FLOPs cap',x=.055,y=.96,ha='left',fontsize=18,fontweight='bold')
    bases='  |  '.join(f'{s["baseline"]["model_label"]} {100*s["baseline"]["final_dice"]:.2f}' for s in model_summaries)
    fig.text(.055,.905,'Final Dice gain in percentage points vs each unaugmented baseline\nBaseline Dice (%): '+bases,fontsize=11,linespacing=1.7,va='top')
    fig.text(.055,.065,'5% training data · fold 0 · seed 0 · no augmentation · cap 31,976.45 GFLOPs\nCalibration is additional. * UNETR edge + depth also incurred 1,561.29 GFLOPs of recovery replay.\nSingle-seed validation results; combinations were compared on the same 27 validation subjects.',fontsize=9,linespacing=1.5)
    fig.savefig(output/'constraint_gains.png',dpi=180);fig.savefig(output/'constraint_gains.pdf');plt.close(fig)

    fig,axs=plt.subplots(2,2,figsize=(11.8,8.0),sharex=True,sharey=True)
    fig.subplots_adjust(top=.79,bottom=.13,wspace=.15,hspace=.31,left=.08,right=.97)
    colors={'baseline':'#374151','bands':'#159895','winner':'#c44e52'}
    for ax,ms in zip(axs.flat,model_summaries):
        model=ms['model'];winner=ms['winner']['objective']
        for objective,name,color in [('dice','Baseline',colors['baseline']),('dice_bands','+ Bands',colors['bands']),
                                      (winner,'Best final variant',colors['winner'])]:
            c=curves[model,objective]
            ax.plot([r['estimated_training_flops']/1e12 for r in c],
                    [100*r['macro_hard_dice'] for r in c],label=name,color=color,lw=2)
        ax.set_title(f'{MODELS[model]}: {ms["winner"]["final_gain_pp"]:+.2f} pp',loc='left',fontweight='bold')
        ax.grid(alpha=.18);ax.set_xlim(0,CAP/1e12*1.02);ax.set_ylim(0,85)
        ax.axvline(CAP/1e12,color='#9ca3af',lw=1,ls=':')
        ax.legend(loc='lower right',fontsize=8,frameon=False)
    for ax in axs[:,0]:ax.set_ylabel('Validation Dice (%)')
    for ax in axs[1,:]:ax.set_xlabel('Optimization compute (1,000 GFLOPs)')
    fig.suptitle('Constraints at matched optimization compute',x=.08,y=.965,ha='left',fontsize=18,fontweight='bold')
    fig.text(.08,.905,'Unaugmented ADNI · same 5% subset, fold 0 and seed 0\nBest final variants selected after this sweep; these are validation curves.',fontsize=11,linespacing=1.6,va='top')
    nn_winner = next(s['winner']['constraint'] for s in model_summaries if s['model']=='nnunet')
    fig.text(.08,.035,'Final-budget comparison; calibration/evaluation costs are outside the optimization cap.\nBest: SwinUNETR / UNet3D all terms; UNETR bands + edge + depth; nnUNet '+nn_winner+'.',fontsize=8.5,linespacing=1.5)
    fig.savefig(output/'learning_curves.png',dpi=180);fig.savefig(output/'learning_curves.pdf');plt.close(fig)
    print(json.dumps(dict(completed_runs=len(rows), verified_records=True,
        winners=[dict(model=s['model'],objective=s['winner']['objective'],
                      baseline=s['baseline']['final_dice'],final=s['winner']['final_dice'],
                      gain_pp=s['winner']['final_gain_pp']) for s in model_summaries]),indent=2))


if __name__=='__main__':
    main()
