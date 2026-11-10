"""Build reviewable seed and paired-target tables from verified durable archives."""
import csv
import argparse
import json
from pathlib import Path
from statistics import mean
from scripts.cross_dataset_campaign.common import read,write,digest
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.mni_constraints.seed_recovery import ROOT,NAME,CAP


def seed_summary(name=NAME):
    report=ROOT/'reports'/name;live=read(report/'live.json');state=live['selection.json']
    rows=[];matched=[]
    for row in state['rows']:
        paths=[];values=[];training=[]
        for fold in range(3):
            identifier=f"MNI_{row['model']}_dice_f{fold}s{row['seed']}"
            folder=report/'frozen/known_runs'/identifier
            if not folder.exists():folder=report/'runs'/identifier
            verify_archive(folder,read(folder/'archive_manifest.json'))
            assert read(folder/'cleanup.json')['status']=='complete'
            values.append(100*read(folder/'validation_latest.json')['summary']['macro_hard_dice'])
            training.append(100*read(folder/'training_final.json')['summary']['macro_hard_dice'])
            config=read(folder/'config.json')
            paths.append(dict(fold=fold,archive=str(folder),archive_manifest_sha256=digest(folder/'archive_manifest.json'),
                              protocol_sha256=config['protocol_sha256'],schedule=config['schedule'],
                              train_cases=config['train_cases'],validation_cases=config['validation_cases']))
        selected=row['seed'] in state['selected'][row['model']]
        rows.append(dict(model=row['model'],seed=row['seed'],fold0_pct=values[0],fold1_pct=values[1],fold2_pct=values[2],
                    mean_pct=mean(values),minimum_fold_pct=min(values),training_mean_pct=mean(training),
                    passes_mean50=mean(values)>=50,passes_every_fold50=min(values)>=50,selected=selected))
        if selected:matched.append(dict(model=row['model'],seed=row['seed'],folds=paths))
    result=dict(status='complete' if state['complete'] else 'interim',budget_flops=CAP,rows=rows,
                selected=state['selected'],rule=state['rule'],caveat=state['caveat'],
                screened_candidates=state.get('decisions',[]),
                recovery_completed_runs=len(list((report/'runs').glob('*/completion.json'))),
                full14_study_resumed=False,analysis_source_sha256=digest(Path(__file__)))
    write(report/'seed_summary.json',result)
    with (report/'seed_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(report/'matched_seed_manifest.json',dict(status=result['status'],budget_flops=CAP,models=state['selected'],
          baseline_archives=matched,fold_policy='Preserve original folds0,1,2 and fixed10 training subjects; use same model-specific seed for every matched condition.',
          excluded_seed_attempts=[dict(model=r['model'],seed=r['seed']) for r in rows if not r['selected']],
          screened_candidates=state.get('decisions',[]),
          selection_caveat=state['caveat']))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--name',default=NAME)
    seed_summary(parser.parse_args().name)
