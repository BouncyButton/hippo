"""Archive seed19 checks and write a verified common-seed decision."""
import argparse
import csv
from pathlib import Path
from statistics import mean
from scripts.cross_dataset_campaign import collect
from scripts.cross_dataset_campaign.common import read,write,digest
from scripts.cross_dataset_campaign.storage import verify_archive


def finish(report):
    frozen=report/'frozen';plan=read(frozen/'plan.json');rows=[]
    entries=[(t,report/'runs'/t['id']) for t in plan['tasks']]
    entries += [(t,frozen/'reused_runs'/t['id']) for t in read(frozen/'reused_runs.json')]
    entries += [(t,frozen/'controls'/t['id']) for t in read(frozen/'controls.json')]
    for task,folder in entries:
        verify_archive(folder,read(folder/'archive_manifest.json'))
        assert read(folder/'cleanup.json')['status']=='complete'
        rows.append(dict(model=task['model'],seed=task['seed'],fold=task['fold'],
            validation_pct=100*read(folder/'validation_latest.json')['summary']['macro_hard_dice'],
            training_pct=100*read(folder/'training_final.json')['summary']['macro_hard_dice'],
            archive=str(folder),archive_manifest_sha256=digest(folder/'archive_manifest.json')))
    summary=[]
    for model in plan['models']:
        for seed in (0,5,19):
            group=sorted((r for r in rows if r['model']==model and r['seed']==seed),key=lambda r:r['fold'])
            assert len(group)==3;values=[r['validation_pct'] for r in group]
            summary.append(dict(model=model,seed=seed,validation_by_fold_pct=values,
                mean_validation_pct=mean(values),passes_mean50=mean(values)>=50,passes_every_fold50=min(values)>=50))
    result=dict(status='complete',budget_flops=plan['budget_flops'],summary=summary,rows=rows,
        seed19_qualifies_for_all_models=all(r['passes_mean50'] for r in summary if r['seed']==19),
        seed19_passes_every_fold_for_all_models=all(r['passes_every_fold50'] for r in summary if r['seed']==19),
        proposed_common_seeds=[0,5,19],interpretation=plan['selection_caveat'],analysis_source_sha256=digest(Path(__file__)))
    write(report/'seed19_comparison.json',result)
    with (report/'common_seeds_all_folds.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(sorted(rows,key=lambda r:(r['model'],r['seed'],r['fold'])))
    return result


def main():
    parser=argparse.ArgumentParser()
    for name in ('stage','local-stage','destination'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();collect.serve(args.stage,args.local_stage,args.destination)
    if read(args.destination/'live.json').get('status.json',{}).get('status')=='complete':finish(args.destination)


if __name__=='__main__':main()
