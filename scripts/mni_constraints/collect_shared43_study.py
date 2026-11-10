"""Finite archive collector and paired nine-cell summaries for the43T study."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

from scripts.cross_dataset_campaign import collect
from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.cross_dataset_campaign.storage import verify_archive


def finish(destination):
    frozen=destination/'frozen';plan=read(frozen/'plan.json')
    entries=[(t,destination/'runs'/t['id'],False) for t in plan['tasks']]
    entries += [(x['task'],frozen/'reused_runs'/x['task']['id'],True) for x in read(frozen/'reused_runs.json')]
    rows=[]
    for task,folder,reused in entries:
        verify_archive(folder,read(folder/'archive_manifest.json'))
        assert read(folder/'cleanup.json')['status']=='complete'
        val=read(folder/'validation_latest.json')['summary']['macro_hard_dice']
        train=read(folder/'training_final.json')['summary']['macro_hard_dice']
        rows.append(dict(**task,validation_dice=val,training_dice=train,gap_pp=100*(train-val),
            reused=reused,archive_manifest_sha256=digest(folder/'archive_manifest.json')))
    assert len(rows)==504 and len({r['id'] for r in rows})==504
    baselines={(r['model'],r['fold'],r['seed']):r for r in rows if r['arm']=='dice'}
    summary=[]
    for model in plan['models']:
        for arm in plan['conditions']:
            cells=[r for r in rows if r['model']==model and r['arm']==arm]
            assert len(cells)==9
            gains=[100*(r['validation_dice']-baselines[(model,r['fold'],r['seed'])]['validation_dice']) for r in cells]
            scores=[100*r['validation_dice'] for r in cells]
            summary.append(dict(model=model,arm=arm,validation_mean_pct=mean(scores),validation_sd_pct=stdev(scores),
                training_mean_pct=100*mean(r['training_dice'] for r in cells),gain_mean_pp=mean(gains),
                gain_sd_pp=stdev(gains),improved_cells=sum(x>0 for x in gains),paired_gains_pp=gains))
    write(destination/'comparison.json',dict(status='complete',budget_flops=plan['budget_flops'],rows=rows,summary=summary,
        interpretation='Final-budget paired comparisons on nine fixed development fold/seed cells. Overlapping folds are not independent samples.',
        analysis_source_sha256=digest(Path(__file__))))
    with (destination/'summary.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=[k for k in summary[0] if k!='paired_gains_pp'])
        writer.writeheader();writer.writerows({k:v for k,v in r.items() if k!='paired_gains_pp'} for r in summary)


def main():
    parser=argparse.ArgumentParser()
    for name in ('stage','local-stage','destination'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    collect.serve(args.stage,args.local_stage,args.destination)
    if read(args.destination/'live.json').get('status.json',{}).get('status')=='complete':
        finish(args.destination)
        env=':'.join(str(args.stage/p) for p in ('source','source/scripts','source/baselines/nnUNet'))
        code='import json;from scripts.cross_dataset_campaign.storage import quota;print(json.dumps(quota()))'
        write(args.destination/'final_storage.json',json.loads(collect.remote(['env','PYTHONPATH='+env,'python3','-c',code])))
        print(json.dumps(dict(status='complete',cells=504)),flush=True)


if __name__=='__main__':main()
