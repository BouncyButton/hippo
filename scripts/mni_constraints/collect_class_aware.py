"""Finite archive collector followed by the requested three-arm comparison."""
import argparse
import csv
import json
from pathlib import Path
from scripts.cross_dataset_campaign import collect
from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.mni_constraints.class_aware_pilot import tasks, evaluation, CAP


def finish(destination):
    frozen = destination/'frozen'; rows = []; sources = {}
    for task in tasks():
        model = task['model']; run = destination/'runs'/task['id']
        verify_archive(run, read(run/'archive_manifest.json'))
        assert read(run/'cleanup.json')['status'] == 'complete'
        sources[task['id']] = digest(run/'archive_manifest.json')
        conditions = {arm:evaluation(frozen/'controls'/model/arm) for arm in ('dice','dice_bands')}
        conditions['class_aware_bands'] = evaluation(run)
        current = conditions['class_aware_bands']['validation']
        delta = {arm:{k:100*(current[k]-v['validation'][k]) for k in current if k.endswith('hard_dice')}
                 for arm,v in conditions.items() if arm!='class_aware_bands'}
        rows.append(dict(model=model, conditions=conditions, delta_pp=delta))
    result = dict(status='complete', fold=0, seed=0, budget_flops=CAP, comparison=rows,
                  source_manifests=sources, inference='Exploratory single-fold single-seed comparison; no tuning or expansion.')
    write(destination/'comparison.json', result)
    lines = ['# MNI class-aware outer-band pilot', '',
        'Fold 0, seed 0; same 2× optimization cap and baseline recipes. Final-budget scores, percentages.', '',
        '| Model | Baseline val | Original bands val | Class-aware val | Δ vs baseline | Δ vs bands | Class-aware train |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        c=row['conditions']; d=row['delta_pp']; m=row['model']
        lines.append(f"| {m} | {100*c['dice']['validation']['macro_hard_dice']:.2f} | {100*c['dice_bands']['validation']['macro_hard_dice']:.2f} | {100*c['class_aware_bands']['validation']['macro_hard_dice']:.2f} | {d['dice']['macro_hard_dice']:+.2f} | {d['dice_bands']['macro_hard_dice']:+.2f} | {100*c['class_aware_bands']['training']['macro_hard_dice']:.2f} |")
    lines += ['', '## Error counts and subfields', '', '| Model / condition | FN | FP | Internal swaps | CA1–3 Dice | Subiculum Dice | CA4–DG Dice | Union Dice |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        for arm, v in row['conditions'].items():
            scores=' | '.join(f"{100*v['validation'][k]:.2f}" for k in ('CA1_3_hard_dice','subiculum_hard_dice','CA4_DG_hard_dice','whole_hippocampus_hard_dice'))
            lines.append(f"| {row['model']} / {arm} | {v['fn']} | {v['fp']} | {v['internal_swaps']} | {scores} |")
    lines += ['', 'Categorical targets on existing outer-contour bands only; no added internal-interface support or class reweighting. Two fresh runs and four verified controls. Checkpoints retired only after verified durable archival.', '', '![Training and validation curves](learning_curves.png)', '']
    (destination/'comparison.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
    for i,task in enumerate(tasks()):
        model=task['model']
        display_model={'swinunetr':'SwinUNETR','unetr_vit_b16':'UNETR'}[model]
        folders={arm:frozen/'controls'/model/arm for arm in ('dice','dice_bands')}
        folders['class_aware_bands']=destination/'runs'/task['id']
        for arm,folder in folders.items():
            with (folder/'metrics.csv').open() as stream:
                records={int(r['charged_attempts']):r for r in csv.DictReader(stream)}
            records=[records[k] for k in sorted(records)]
            for j,key in enumerate(('training_macro_hard_dice','macro_hard_dice')):
                axes[i,j].plot([100*float(r['compute_fraction']) for r in records],
                    [100*float(r[key]) for r in records],label={'dice':'Baseline','dice_bands':'Original bands','class_aware_bands':'Class-aware bands'}[arm])
                axes[i,j].set_title(display_model+' — '+('training' if j==0 else 'validation'))
                axes[i,j].set_ylabel('Dice (%)'); axes[i,j].grid(alpha=.2)
        axes[i,1].legend(fontsize=8, loc='lower right', bbox_to_anchor=(.98,.02), borderaxespad=0)
    for ax in axes[-1]:ax.set_xlabel('Optimization FLOP cap used (%)')
    fig.tight_layout();fig.savefig(destination/'learning_curves.png',dpi=160);fig.savefig(destination/'learning_curves.pdf');plt.close(fig)
    return result


def main():
    parser=argparse.ArgumentParser()
    for name in ('stage','local-stage','destination'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    collect.serve(args.stage,args.local_stage,args.destination)
    state=read(args.destination/'live.json')
    if state.get('status.json',{}).get('status')=='complete':
        finish(args.destination)
        env=':'.join(str(args.stage/p) for p in ('source','source/scripts','source/baselines/nnUNet'))
        quota=json.loads(collect.remote(['env','PYTHONPATH='+env,'python3','-c',
            'import json;from scripts.cross_dataset_campaign.storage import quota;print(json.dumps(quota()))']))
        write(args.destination/'final_quota.json',quota)
        print(json.dumps(dict(status='complete',comparison=str(args.destination/'comparison.md'),free_gib=quota['free_gib'])),flush=True)


if __name__=='__main__':main()
