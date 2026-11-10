"""Archive the six missing runs and compare the common seed5 against seed1."""
import argparse
import csv
from pathlib import Path
from statistics import mean
from scripts.cross_dataset_campaign import collect
from scripts.cross_dataset_campaign.common import read,write,digest
from scripts.cross_dataset_campaign.storage import verify_archive


def finish(report):
    frozen=report/'frozen';plan=read(frozen/'plan.json')
    entries=[(t,report/'runs'/t['id']) for t in plan['tasks']]
    entries += [(t,frozen/'reused_runs'/t['id']) for t in read(frozen/'reused_runs.json')]
    rows=[]
    for t,folder in entries:
        verify_archive(folder,read(folder/'archive_manifest.json'))
        assert read(folder/'cleanup.json')['status']=='complete'
        control=frozen/'seed1_controls'/f"MNI_{t['model']}_dice_f{t['fold']}s1"
        verify_archive(control,read(control/'archive_manifest.json'))
        val=100*read(folder/'validation_latest.json')['summary']['macro_hard_dice']
        old=100*read(control/'validation_latest.json')['summary']['macro_hard_dice']
        rows.append(dict(model=t['model'],fold=t['fold'],seed=5,validation_pct=val,
                    training_pct=100*read(folder/'training_final.json')['summary']['macro_hard_dice'],
                    seed1_validation_pct=old,gain_vs_seed1_pp=val-old,
                    archive=str(folder),archive_manifest_sha256=digest(folder/'archive_manifest.json')))
    summary=[]
    for m in plan['models']:
        group=sorted((r for r in rows if r['model']==m),key=lambda r:r['fold']);assert len(group)==3
        vals=[r['validation_pct'] for r in group]
        summary.append(dict(model=m,seed5_by_fold_pct=vals,seed5_mean_pct=mean(vals),
                    seed1_mean_pct=mean(r['seed1_validation_pct'] for r in group),
                    mean_change_pp=mean(r['gain_vs_seed1_pp'] for r in group),
                    passes_mean50=mean(vals)>=50,passes_every_fold50=min(vals)>=50))
    result=dict(status='complete',budget_flops=plan['budget_flops'],summary=summary,rows=rows,
                seed5_qualifies_for_all_models=all(r['passes_mean50'] for r in summary),
                seed5_passes_every_fold_for_all_models=all(r['passes_every_fold50'] for r in summary),
                interpretation=plan['selection_caveat'],analysis_source_sha256=digest(Path(__file__)))
    write(report/'seed5_comparison.json',result)
    with (report/'seed5_all_folds.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(sorted(rows,key=lambda r:(r['model'],r['fold'])))
    return result


def main():
    parser=argparse.ArgumentParser()
    for name in ('stage','local-stage','destination'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();collect.serve(args.stage,args.local_stage,args.destination)
    if read(args.destination/'live.json').get('status.json',{}).get('status')=='complete':finish(args.destination)


if __name__=='__main__':main()
