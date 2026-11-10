"""Summarize achieved training headroom at one shared cap without selecting on validation."""
import argparse
import csv
import json
from pathlib import Path

from scripts.cross_dataset_campaign.common import read,write,digest
from scripts.cross_dataset_campaign.storage import verify_archive


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parents[2];report=root/'reports'/args.name
    plan=read(report/'frozen/plan.json');rows=[]
    for task in plan['tasks']:
        run=report/'runs'/task['id']
        if not (run/'completion.json').exists():continue
        verify_archive(run,read(run/'archive_manifest.json'))
        cfg=read(run/'config.json');done=read(run/'completion.json')
        training_report=read(run/'training_final.json')
        validation_report=read(run/'validation_latest.json')
        training=training_report['summary']['macro_hard_dice']
        validation=validation_report['summary']['macro_hard_dice']
        assert training==done['final_training_hard_dice'] and cfg['budget_flops']==plan['budget_flops']
        with (run/'metrics.csv').open() as stream:curve=list(csv.DictReader(stream))
        assert abs(float(curve[-1]['training_macro_hard_dice'])-training)<1e-12
        rows.append(dict(model=task['model'],fold=task['fold'],seed=task['seed'],training_dice=training,
            validation_dice=validation,training_validation_gap_pp=100*(training-validation),
            minimum_training_subfield_dice=min(v['dice'] for case in training_report['cases'] for k,v in case['per_class'].items() if k!='background'),
            minimum_validation_subfield_dice=min(v['dice'] for case in validation_report['cases'] for k,v in case['per_class'].items() if k!='background'),
            within_target=.85<=training<=.90,over_target=training>.90,under_target=training<.85,
            updates=done['updates'],optimization_flops=done['consumed_training_flops'],
            curve=[{k:float(x[k]) for k in ('compute_fraction','training_macro_hard_dice','macro_hard_dice')} for x in curve],
            source_archive_sha256=digest(run/'archive_manifest.json'),cleaned=(run/'cleanup.json').exists()))
    result=dict(status='complete' if len(rows)==len(plan['tasks']) else 'partial',cap=plan['budget_flops'],multiplier=plan['multiplier'],
        target=[.85,.90],selection_uses_validation=False,runs=rows,
        all_models_within_target=len(rows)==4 and all(r['within_target'] for r in rows),
        all_planned_models_within_target=len(rows)==len(plan['tasks']) and all(r['within_target'] for r in rows),
        caveat='Singlefold/seed cap pilot. In-run milestones use the final-cap LR horizon; they are not equivalent to fresh runs with a shorter matched LR schedule.')
    write(report/'training_headroom.json',result)
    if len(rows)==4:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
        for ax,r in zip(axes.flat,rows):
            curve=r['curve'];x=[100*c['compute_fraction'] for c in curve]
            ax.axhspan(85,90,color='#75ad7c',alpha=.22,label='85–90% target')
            ax.plot(x,[100*c['training_macro_hard_dice'] for c in curve],color='#2459a6',label='Training')
            ax.plot(x,[100*c['macro_hard_dice'] for c in curve],color='#b96528',ls='--',label='Validation')
            ax.set_title(r['model']);ax.set_ylim(0,100);ax.grid(alpha=.15)
            ax.set_ylabel('Macro Dice (%)');ax.set_xlabel('Shared pilot cap consumed (%)')
            ax.legend(fontsize=8,loc='lower right')
        fig.suptitle(f'MNI: shared {plan["multiplier"]:g}× historical cap — fold 0, seed 0')
        fig.tight_layout();fig.savefig(report/'training_headroom.png',dpi=180);plt.close(fig)
    print(json.dumps({**result,'runs':[{k:v for k,v in r.items() if k!='curve'} for r in rows]},indent=2))


if __name__=='__main__':main()
