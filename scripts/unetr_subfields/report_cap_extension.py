"""Verify archived MNI cap batches and report fixed-fold, fixed-seed results."""
import argparse
from pathlib import Path
import statistics
from scripts.cross_dataset_campaign.common import read,write,digest
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.unetr_subfields.contracts import ROOT,CAP


def summarize(report):
    frozen=report/'frozen';plan=read(frozen/'plan.json');protocol=read(frozen/'protocols/model/protocol.json')
    for name,sha in read(frozen/'manifest.json').items():
        if digest(frozen/name)!=sha:raise ValueError('Source archive mismatch: '+name)
    rows=[]
    for task in plan['tasks']:
        r=report/'runs'/task['id']
        if not (r/'cleanup.json').exists():continue
        manifest=read(r/'archive_manifest.json');verify_archive(r,manifest)
        ack=read(r/'archive_acknowledgment.json');done=read(r/'completion.json');config=read(r/'config.json')
        assert ack['status']=='verified' and ack['manifest_sha256']==digest(r/'archive_manifest.json')
        assert ack['source_manifest_sha256']==digest(frozen/'manifest.json')
        assert config['protocol_sha256']==digest(frozen/'protocols/model/protocol.json')
        schedule=protocol['models'][protocol['trainer_alias']]['objectives'][task['arm']]
        assert done['status']=='complete' and done['updates']==schedule['max_steps'] and done['discarded_attempts']==0
        assert done['consumed_training_flops']==schedule['consumed_training_flops']<=plan['budget_flops']
        assert config['dataset_sha256']==protocol['pkl_sha256'] and (config['fold'],config['seed'])==(task['fold'],task['seed'])
        splits=read(frozen/'protocols/model/splits_10patients.json');assert config['train_cases']==splits[task['fold']]['train'] and config['validation_cases']==splits[task['fold']]['val']
        val=read(r/'validation_latest.json')['summary'];train=read(r/'training_final.json')['summary']
        rows.append(dict(model=plan['model'],fold=task['fold'],seed=task['seed'],validation_dice=val['macro_hard_dice'],
            training_dice=train['macro_hard_dice'],subfields={c:val[c+'_hard_dice'] for c in protocol['classes'][1:]},
            whole_hippocampus_dice=val['whole_hippocampus_hard_dice'],updates=done['updates'],
            released_bytes=read(r/'cleanup.json')['released_file_bytes']))
    seed_means={str(seed):statistics.mean(r['validation_dice'] for r in rows if r['seed']==seed)
                for seed in range(3) if {r['fold'] for r in rows if r['seed']==seed}=={0,1,2}}
    complete=len(rows)==len(plan['tasks'])
    minimum_subfield=min((v for r in rows for v in r['subfields'].values()),default=None)
    class_collapse_free=complete and minimum_subfield is not None and minimum_subfield>=.10
    target_met=(not plan['pilot'] and complete and len(seed_means)==3 and
                all(r['validation_dice']>=.45 for r in rows) and all(v>=.50 for v in seed_means.values()) and class_collapse_free)
    summary=dict(name=plan['name'],model=plan['model'],pilot=plan['pilot'],complete=complete,completed=len(rows),
        planned=len(plan['tasks']),target_met=target_met,budget_flops=plan['budget_flops'],historical_cap_multiplier=plan['budget_flops']/CAP,
        seed_means=seed_means,mean_dice=statistics.mean(r['validation_dice'] for r in rows) if rows else None,
        minimum_dice=min((r['validation_dice'] for r in rows),default=None),
        minimum_subfield_dice=minimum_subfield,class_collapse_free=class_collapse_free,
        released_bytes=sum(r['released_bytes'] for r in rows),runs=rows,
        source_manifest_sha256=digest(frozen/'manifest.json'),dataset_sha256=protocol['pkl_sha256'],splits_sha256=protocol['splits_sha256'])
    write(report/'summary.json',summary)
    return summary


def finalize(names, destination):
    """Only publish success after all36 archives and common-budget bindings pass."""
    results=[summarize(ROOT/'reports'/name) for name in names]
    assert {r['model'] for r in results}=={'nnunet','swinunetr','unetr_vit_b16','basic_unet16'} and len(results)==4
    assert all(r['target_met'] and len(r['runs'])==9 for r in results)
    for key in ('budget_flops','dataset_sha256','splits_sha256'):
        assert len({r[key] for r in results})==1,key
    value=dict(status='complete',dataset='MNI',training_subjects_per_fold=5,training_crops_per_fold=10,
        validation_subjects_per_fold=5,folds=[0,1,2],seeds=[0,1,2],budget_flops=results[0]['budget_flops'],
        historical_cap_multiplier=results[0]['historical_cap_multiplier'],completed_runs=36,
        primary_metric='Mean per-volume foreground-subfield macro hard Dice at final budget',
        selection_caveat='Development validation informed cap and recipe selection; these are not untouched test estimates.',
        released_final_run_bytes=sum(r['released_bytes'] for r in results),models=results)
    write(destination/'final_results.json',value)
    return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('names',nargs='+');parser.add_argument('--finalize',type=Path);args=parser.parse_args()
    if args.finalize:
        result=finalize(args.names,args.finalize)
        print(dict(status=result['status'],runs=result['completed_runs'],budget_flops=result['budget_flops']));return
    for name in args.names:
        result=summarize(ROOT/'reports'/name)
        print({k:v for k,v in result.items() if k not in ('runs','source_manifest_sha256','dataset_sha256','splits_sha256')})


if __name__=='__main__':main()
