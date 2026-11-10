"""Finite archival and final comparison for the adaptive-band pilot."""
import argparse
import csv
import json
from pathlib import Path
from scripts.cross_dataset_campaign import collect
from scripts.cross_dataset_campaign.common import read,write,digest
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.mni_constraints.adaptive_pilot import tasks,MODELS,VARIANTS,CAP
from scripts.mni_constraints.class_aware_pilot import evaluation


def finish(destination):
    rows=[]; sources={}
    for task in tasks():
        run=destination/'runs'/task['id']; verify_archive(run,read(run/'archive_manifest.json'))
        assert read(run/'cleanup.json')['status']=='complete'
        sources[task['id']]=digest(run/'archive_manifest.json')
        for q in (25,50,75,100): assert read(run/f'spatial_{q:03d}.json')['status']=='complete'
    for m in MODELS:
        conditions={v:evaluation(destination/'runs'/f'MNI_{m}_{v}_f0s0') for v in VARIANTS}
        for arm in ('dice','dice_bands'):
            conditions['archived_'+arm]=evaluation(destination/'frozen/controls'/m/arm)
        deltas={v:{ref:100*(conditions[v]['validation']['macro_hard_dice']-conditions[ref]['validation']['macro_hard_dice'])
                   for ref in ('baseline','archived_dice_bands')} for v in ('weighted','adaptive')}
        rows.append(dict(model=m,conditions=conditions,delta_pp=deltas,
            baseline_replay_delta_pp=100*(conditions['baseline']['validation']['macro_hard_dice']-conditions['archived_dice']['validation']['macro_hard_dice'])))
    result=dict(status='complete',fold=0,seed=0,budget_flops=CAP,comparison=rows,source_manifests=sources,
        caveat='Exploratory single-fold single-seed development comparison. Error tradeoffs accepted; absolute validation Dice primary. No automatic expansion.')
    write(destination/'comparison.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,2,figsize=(12,14),sharex=True)
    for i,m in enumerate(MODELS):
        folders={v:destination/'runs'/f'MNI_{m}_{v}_f0s0' for v in VARIANTS}
        folders['original_bands']=destination/'frozen/controls'/m/'dice_bands'
        for v,folder in folders.items():
            with (folder/'metrics.csv').open() as f: records=list(csv.DictReader(f))
            for j,key in enumerate(('training_macro_hard_dice','macro_hard_dice')):
                axes[i,j].plot([100*float(r['compute_fraction']) for r in records],[100*float(r[key]) for r in records],label=v)
                axes[i,j].set_title(m+' — '+('training' if j==0 else 'validation'))
                axes[i,j].set_ylabel('Dice (%)'); axes[i,j].grid(alpha=.2)
        axes[i,1].legend(fontsize=8,loc='lower right')
    for ax in axes[-1]: ax.set_xlabel('Optimization FLOP cap used (%)')
    fig.tight_layout(); fig.savefig(destination/'learning_curves.png',dpi=150); fig.savefig(destination/'learning_curves.pdf'); plt.close(fig)
    return result


def main():
    parser=argparse.ArgumentParser()
    for name in ('stage','local-stage','destination'): parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(); collect.serve(args.stage,args.local_stage,args.destination)
    if read(args.destination/'live.json').get('status.json',{}).get('status')=='complete':
        finish(args.destination)
        env=':'.join(str(args.stage/p) for p in ('source','source/scripts','source/baselines/nnUNet'))
        code='import json,sys;from pathlib import Path;from scripts.cross_dataset_campaign.storage import quota;p=Path(sys.argv[1]);print(json.dumps(dict(quota=quota(),cleanup={f.name:json.loads(f.read_text()) for f in p.glob("cache_cleanup_*.json")})))'
        final=json.loads(collect.remote(['env','PYTHONPATH='+env,'python3','-c',code,str(args.stage)]))
        assert len(final['cleanup'])==4
        write(args.destination/'final_storage.json',final)
        print(json.dumps(dict(status='complete',comparison=str(args.destination/'comparison.json'),free_gib=final['quota']['free_gib'])),flush=True)


if __name__=='__main__': main()
