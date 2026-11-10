"""Summarize archived Dice, spatial coverage and gradient drift for the pilot."""
import csv
import json
from pathlib import Path
import shutil
import numpy as np
from scripts.cross_dataset_campaign.common import read,write,digest
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.mni_constraints.adaptive_pilot import NAME,ROOT,MODELS,VARIANTS,CAP
from scripts.mni_constraints.class_aware_pilot import evaluation


def main():
    report=ROOT/'reports'/NAME; rows=[]
    for model in MODELS:
        controls={arm:evaluation(report/'frozen/controls'/model/arm) for arm in ('dice','dice_bands')}
        for variant in VARIANTS:
            run=report/'runs'/f'MNI_{model}_{variant}_f0s0'
            if not (run/'cleanup.json').exists(): continue
            verify_archive(run,read(run/'archive_manifest.json'))
            scores=evaluation(run); spatial=[]
            for q in (25,50,75,100):
                data=read(run/f'spatial_{q:03d}.json')
                for split in ('training','validation'):
                    cases=[r for r in data['cases'] if r['split']==split]
                    counts={error:{key:sum(r['errors'][error][key] for r in cases)
                                   for key in ('total','original_band','adaptive_band')}
                            for error in ('fn','fp','internal_swaps')}
                    external={key:sum(counts[e][key] for e in ('fn','fp')) for key in ('total','original_band','adaptive_band')}
                    item=dict(quarter=q,actual_fraction=data['actual_fraction'],split=split,counts=counts,
                        external_coverage_original=external['original_band']/max(external['total'],1),
                        adaptive_support_fraction=sum(r['support_voxels']['adaptive'] for r in cases)/sum(r['support_voxels']['volume'] for r in cases),
                        original_support_fraction=sum(r['support_voxels']['original'] for r in cases)/sum(r['support_voxels']['volume'] for r in cases))
                    if q==100:
                        check=read(run/('training_final.json' if split=='training' else 'validation_latest.json'))
                        lookup={c['case_name']:c for c in check['cases']}
                        for row in cases:
                            assert row['errors']['fn']['total']==lookup[row['case_name']]['foreground_fn']
                            assert row['errors']['fp']['total']==lookup[row['case_name']]['foreground_fp']
                    if variant!='baseline':
                        item['paired_baseline']={key:sum(r['paired_baseline'][key] for r in cases) for key in cases[0]['paired_baseline']}
                        if split=='training':
                            probes=[r['gradient_probe'][variant] for r in cases]
                            ratios=[r['weighted_norm_ratio'] for r in probes]
                            item['gradient_probe']=dict(median_weighted_ratio=float(np.median(ratios)),
                                p95_weighted_ratio=float(np.percentile(ratios,95)),
                                mean_fraction_energy_on_external_errors=float(np.mean([r['fraction_energy_on_external_errors'] for r in probes])))
                    spatial.append(item)
            with (run/'metrics.csv').open() as stream: curve=list(csv.DictReader(stream))
            baseline_file=report/'runs'/f'MNI_{model}_baseline_f0s0/metrics.csv'
            with baseline_file.open() as stream: baseline_curve=list(csv.DictReader(stream))
            curve_comparison=[]
            for q in (25,50,75,100):
                a=min(curve,key=lambda r:abs(float(r['compute_fraction'])-q/100))
                b=min(baseline_curve,key=lambda r:abs(float(r['compute_fraction'])-q/100))
                curve_comparison.append(dict(quarter=q,variant_fraction=float(a['compute_fraction']),baseline_fraction=float(b['compute_fraction']),
                    validation_delta_pp=100*(float(a['macro_hard_dice'])-float(b['macro_hard_dice'])),
                    training_dice=100*float(a['training_macro_hard_dice']),validation_dice=100*float(a['macro_hard_dice'])))
            rows.append(dict(model=model,variant=variant,scores=scores,spatial=spatial,curve_comparison=curve_comparison,
                delta_vs_archived_pp={arm:100*(scores['validation']['macro_hard_dice']-controls[arm]['validation']['macro_hard_dice']) for arm in controls},
                archive_manifest_sha256=digest(run/'archive_manifest.json')))
    result=dict(status='complete' if len(rows)==12 else 'partial',runs=len(rows),fold=0,seed=0,budget_flops=CAP,rows=rows,
        selection='Final-budget validation macro Dice primary; no FN/FP veto or fixed training margin.',
        limitations=['One development fold and seed; exploratory, not a generalization estimate.',
            'Adaptive support covers current errors by construction; retrospective validation coverage is not a learning result.',
            'Gradient drift is measured on unaugmented eval-mode training cases, not noisy optimization minibatches.'],
        analysis_source_sha256=digest(Path(__file__)))
    write(report/'diagnostic_summary.json',result)
    shutil.copy2(Path(__file__),report/'summarize_adaptive_source.py')
    print(json.dumps(dict(status=result['status'],runs=len(rows),results=[dict(model=r['model'],variant=r['variant'],
        train=100*r['scores']['training']['macro_hard_dice'],val=100*r['scores']['validation']['macro_hard_dice'],
        delta_vs_bands=r['delta_vs_archived_pp']['dice_bands']) for r in rows])))


if __name__=='__main__': main()
