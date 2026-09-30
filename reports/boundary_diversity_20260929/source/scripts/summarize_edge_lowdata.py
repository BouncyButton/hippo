#!/usr/bin/env python3
"""Summarize all three paired low-data seeds without pooling them as new cases."""
import argparse
import json
from pathlib import Path
import statistics
from probe_edge_consistency import save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--candidate',choices=('edge','weighted'),default='edge')
    parser.add_argument('--fold',type=int,default=0)
    args=parser.parse_args()
    seeds=[]
    for seed in (0,1,2):
        root=args.root/f'seed{seed}'
        assert json.loads((root/'completion.json').read_text())['status']=='complete'
        learning=json.loads((root/'learning_curves.json').read_text())
        row={'fold':args.fold,'seed':seed,'learning_curves':learning,'selected_checkpoints':{},'splits':{}}
        for split in ('train','validation'):
            summary=json.loads((root/split/'audit/summary.json').read_text())
            row['splits'][split]={model:{'macro_dice':s['mean_case_metrics']['macro_dice']['mean'],
                'balanced_boundary_error':s['mean_case_metrics']['balanced_boundary_error']['mean'],
                'boundary_counts':s['pooled_regions']['boundary']} for model,s in summary.items()}
        row['selected_checkpoints']=json.loads((root/'validation/audit/checkpoints.json').read_text())
        seeds.append(row)
    aggregate={}
    model_names=tuple(seeds[0]['splits']['validation'])
    for model in model_names:
        values=[s['splits']['validation'][model]['macro_dice'] for s in seeds]
        aggregate[model]={'mean_validation_dice':statistics.mean(values),'sample_sd_across_seeds':statistics.stdev(values),
            'mean_best_logged_dice':statistics.mean(s['learning_curves']['models'][model]['best_dice'] for s in seeds)}
    for ref in (name for name in model_names if name!=args.candidate):
        deltas=[s['splits']['validation'][args.candidate]['macro_dice']-s['splits']['validation'][ref]['macro_dice'] for s in seeds]
        aggregate[args.candidate+'_minus_'+ref]={'mean_validation_dice_delta':statistics.mean(deltas),'per_seed_deltas':deltas}
    save(args.root/'SUMMARY.json',{'status':'complete','seeds':seeds,'aggregate':aggregate,
        'candidate':args.candidate,'fold':args.fold,
        'scope':f'One fixed ten-case subset, fold {args.fold}, three initialization seeds. Same 52 development cases across seeds; no independent-case multiplication. Follow each learning-curve report for hardware/storage timing limits.'})
    print(json.dumps({'all_seeds_complete':True,'aggregate':aggregate}),flush=True)


if __name__=='__main__':main()
