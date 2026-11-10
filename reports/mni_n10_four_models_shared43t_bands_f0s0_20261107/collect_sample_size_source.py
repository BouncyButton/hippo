"""Finite collector and learning-curve summaries for the subject-count study."""
import argparse
import csv
import json
from pathlib import Path
import shutil
from scripts.cross_dataset_campaign import collect
from scripts.cross_dataset_campaign.common import read,write,digest
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.mni_constraints.class_aware_pilot import evaluation


def finish(destination):
    frozen=destination/'frozen';plan=read(frozen/'plan.json');rows=[]
    counts=plan['training_subject_counts']
    models=plan.get('comparison_models',plan['models']);conditions=plan.get('conditions',('baseline','original_bands','adaptive'))
    entries=[(task,destination/'runs'/task['id'],'') for task in plan['tasks']]
    if plan.get('reuse_pilot_baselines'):
        entries.extend((item['task'],frozen/'reused_baselines'/item['task']['id'],'pilot_baseline') for item in read(frozen/'pilot_baseline_reuse.json'))
    if plan.get('matched_reuse_count'):
        entries.extend((item['task'],frozen/'reused_comparisons'/item['task']['id'],'matched_run') for item in read(frozen/'matched_reuse.json'))
    for task,run,reused in entries:
        verify_archive(run,read(run/'archive_manifest.json'))
        assert read(run/'cleanup.json')['status']=='complete'
        with (run/'metrics.csv').open() as stream:curve=list(csv.DictReader(stream))
        curve=[{k:float(r[k]) for k in ('compute_fraction','estimated_training_flops','training_macro_hard_dice','macro_hard_dice','patient_passes','learning_rate')} for r in curve]
        scores=evaluation(run);last=curve[-1];at75=min(curve,key=lambda r:abs(r['compute_fraction']-.75))
        rows.append(dict(**task,scores=scores,curve=curve,reused_pilot_baseline=reused=='pilot_baseline',reused_archive=bool(reused),
            train_val_gap_pp=100*(last['training_macro_hard_dice']-last['macro_hard_dice']),
            last_quarter_train_change_pp=100*(last['training_macro_hard_dice']-at75['training_macro_hard_dice']),
            last_quarter_val_change_pp=100*(last['macro_hard_dice']-at75['macro_hard_dice']),
            final_vs_best_val_pp=100*(last['macro_hard_dice']-max(r['macro_hard_dice'] for r in curve)),
            archive_manifest_sha256=digest(run/'archive_manifest.json')))
    references=[]
    for model in models:
        for variant in conditions:
            folder=frozen/'controls'/model/variant
            if not folder.exists():continue
            references.append(dict(model=model,train_subjects=5,condition=variant,scores=evaluation(folder),
                budget_flops=read(folder/'config.json')['budget_flops'],
                same_cap=read(folder/'config.json')['budget_flops']==plan['budget_flops']))
    pairs=[]
    if plan['phase']=='comparison':
        for model in models:
            for count in counts:
                pair={r['condition']:r for r in rows if r['model']==model and r['train_subjects']==count}
                pairs.append(dict(model=model,train_subjects=count,
                    validation_gain_pp={arm:100*(pair[arm]['scores']['validation']['macro_hard_dice']-pair['baseline']['scores']['validation']['macro_hard_dice']) for arm in conditions if arm!='baseline'},
                    training_gain_pp={arm:100*(pair[arm]['scores']['training']['macro_hard_dice']-pair['baseline']['scores']['training']['macro_hard_dice']) for arm in conditions if arm!='baseline'}))
    result=dict(status='complete',phase=plan['phase'],budget_flops=plan['budget_flops'],rows=rows,references=references,pairs=pairs,
        interpretation='Pilot curves calibrate a single shared cap; no comparison arm is launched by this collector. Validation is development data, not an untouched test. Final-budget scores primary.')
    result['analysis_source_sha256']=digest(Path(__file__))
    shutil.copy2(Path(__file__),destination/'collect_sample_size_source.py')
    write(destination/'comparison.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(models),len(counts),figsize=(6*len(counts),4*len(models)),sharex=True,squeeze=False)
    for i,m in enumerate(models):
        for j,n in enumerate(counts):
            ax=axes[i,j]
            for row in rows:
                if row['model']!=m or row['train_subjects']!=n:continue
                x=[100*r['compute_fraction'] for r in row['curve']]
                for key,style in [('training_macro_hard_dice','--'),('macro_hard_dice','-')]:
                    ax.plot(x,[100*r[key] for r in row['curve']],style,label=row['condition']+(' train' if style=='--' else ' val'))
            ax.set_title(f'{m} — {n} training subjects');ax.set_ylabel('Dice (%)');ax.grid(alpha=.2);ax.legend(fontsize=8,loc='lower right')
    for ax in axes[-1]:ax.set_xlabel('Optimization cap consumed (%)')
    fig.tight_layout();fig.savefig(destination/'learning_curves.png',dpi=150);fig.savefig(destination/'learning_curves.pdf');plt.close(fig)
    print(json.dumps(dict(phase=plan['phase'],results=[dict(model=r['model'],subjects=r['train_subjects'],arm=r['condition'],
        train=100*r['scores']['training']['macro_hard_dice'],validation=100*r['scores']['validation']['macro_hard_dice'],gap=r['train_val_gap_pp']) for r in rows])),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    for name in ('stage','local-stage','destination'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();collect.serve(args.stage,args.local_stage,args.destination)
    if read(args.destination/'live.json').get('status.json',{}).get('status')=='complete':
        finish(args.destination)
        env=':'.join(str(args.stage/p) for p in ('source','source/scripts','source/baselines/nnUNet'))
        code='import json,sys;from pathlib import Path;from scripts.cross_dataset_campaign.storage import quota;p=Path(sys.argv[1]);print(json.dumps(dict(quota=quota(),cleanup={f.name:json.loads(f.read_text()) for f in p.glob("cache_cleanup_*.json")})))'
        final=json.loads(collect.remote(['env','PYTHONPATH='+env,'python3','-c',code,str(args.stage)]))
        plan=read(args.destination/'frozen/plan.json')
        if plan['phase']=='comparison' and plan.get('paired_baseline_predictions',not plan.get('reuse_pilot_baselines',False)):
            assert len(final['cleanup'])==len(plan['models'])*len(plan['training_subject_counts'])
        write(args.destination/'final_storage.json',final)
        print(json.dumps(dict(status='complete',free_gib=final['quota']['free_gib'])),flush=True)


if __name__=='__main__':main()
