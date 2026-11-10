"""Audit all pilot results and compare against both exact archived controls."""
import math
from pathlib import Path
from statistics import mean

from scripts.cross_dataset_campaign.common import read,write,digest
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.mni_constraints.unet_bands_timing_strength import ROOT,NAME,VARIANTS,CAP

METRICS=('macro_hard_dice','CA4_DG_hard_dice','CA1_3_hard_dice','subiculum_hard_dice','whole_hippocampus_hard_dice')


def load(folder):
    manifest=read(folder/'archive_manifest.json');verify_archive(folder,manifest)
    ack=read(folder/'archive_acknowledgment.json')
    assert ack['status']=='verified' and ack['manifest_sha256']==digest(folder/'archive_manifest.json')
    cfg=read(folder/'config.json');done=read(folder/'completion.json')
    assert done['status']=='complete' and not done['discarded_attempts']
    assert done['consumed_training_flops']<=cfg['budget_flops']==CAP
    assert done['updates']==done['training_attempts_charged']==cfg['max_steps']
    assert read(folder/'evaluation_diagnostics.json')['status']=='complete'
    return dict(config=cfg,done=done,validation=read(folder/'validation_latest.json'),
                training=read(folder/'training_final.json'),calibration=read(folder/'calibration.json'),
                archive_sha256=digest(folder/'archive_manifest.json'),archive=str(folder),
                cleaned=(folder/'cleanup.json').exists() and read(folder/'cleanup.json')['status']=='complete')


def analyze():
    stage=ROOT/'experiments'/NAME;report=ROOT/'reports'/NAME;plan=read(stage/'plan.json')
    controls={};rows=[];loaded={}
    for c in read(stage/'controls.json'):
        controls[c['seed'],c['arm']]=load(stage/'controls'/c['id'])
    for variant in VARIANTS:
        for seed in (0,5):
            if variant=='early10':run=controls[seed,'dice_bands']
            else:
                folder=report/'runs'/f'MNI_unet_{variant}_f1s{seed}'
                if not (folder/'archive_acknowledgment.json').exists():continue
                run=load(folder)
                assert run['config']['protocol_sha256']==digest(stage/'protocols'/variant/'protocol.json')
            baseline=controls[seed,'dice'];original=controls[seed,'dice_bands']
            for key in ('train_cases','validation_cases','dataset_sha256','initial_lr','augmentation','shared_source_sha256','objective_source_sha256'):
                assert run['config'][key]==baseline['config'][key],key
            cal=run['calibration'];prefix=470 if VARIANTS[variant][0] else 76
            assert cal['anchor_successful_updates']==cal['anchor_training_attempts']==prefix
            assert cal['target_ratio']==VARIANTS[variant][1] and cal['p95_cap']==.75
            assert set(c['case_name'] for c in cal['cases'])==set(run['config']['train_cases'])
            assert cal['config_sha256']==digest(Path(run['archive'])/'config.json')
            v=run['validation']['summary'];b=baseline['validation']['summary'];old=original['validation']['summary']
            rows.append(dict(variant=variant,seed=seed,fold=1,origin='reused' if variant=='early10' else 'new',
                dice=100*v['macro_hard_dice'],baseline_dice=100*b['macro_hard_dice'],gain_pp=100*(v['macro_hard_dice']-b['macro_hard_dice']),
                gain_vs_original_bands_pp=100*(v['macro_hard_dice']-old['macro_hard_dice']),
                ca4_dice=100*v['CA4_DG_hard_dice'],baseline_ca4_dice=100*b['CA4_DG_hard_dice'],
                per_metric_gain_pp={k:100*(v[k]-b[k]) for k in METRICS},
                training_gain_pp={k:100*(run['training']['summary'][k]-baseline['training']['summary'][k]) for k in METRICS},
                calibration_weight=cal['weights']['bands'],calibration_statistics=cal['statistics'],
                updates=run['done']['updates'],optimization_flops=run['done']['consumed_training_flops'],
                calibration_flops=run['done']['calibration_flops'],archive=run['archive'],archive_sha256=run['archive_sha256'],cleaned=run['cleaned']))
            loaded[variant,seed]=run
    anchor_checks=[]
    for timing in ('early','late'):
        for seed in (0,5):
            a,b=loaded.get((timing+'10',seed)),loaded.get((timing+'05',seed))
            if not a or not b:continue
            ca,cb=a['calibration'],b['calibration']
            assert ca['anchor_model_sha256']==cb['anchor_model_sha256']
            assert math.isclose(ca['weights']['bands'],2*cb['weights']['bands'],rel_tol=1e-10)
            anchor_checks.append(dict(timing=timing,seed=seed,identical_calibration_anchor=True,weight_halved=True))
    summary=[]
    for variant in VARIANTS:
        group=[r for r in rows if r['variant']==variant]
        summary.append(dict(variant=variant,pairs=len(group),positive_pairs=sum(r['gain_pp']>0 for r in group),
            mean_gain_pp=mean(r['gain_pp'] for r in group) if group else None,
            minimum_gain_pp=min((r['gain_pp'] for r in group),default=None),
            both_pairs_improved=len(group)==2 and all(r['gain_pp']>0 for r in group)))
    complete=len(rows)==8 and all(r['cleaned'] for r in rows)
    result=dict(status='complete' if complete else 'running',rows=rows,summary=summary,
        calibration_anchor_checks=anchor_checks,baseline_controls={str(s):dict(dice=100*controls[s,'dice']['validation']['summary']['macro_hard_dice'],ca4_dice=100*controls[s,'dice']['validation']['summary']['CA4_DG_hard_dice']) for s in (0,5)},
        source_sha256=digest(Path(__file__)),manifest_sha256=digest(stage/'manifest.json'),
        caveat='Two development pairs. Original bands/base unchanged. Factorial constrained cells all938updates with identical LR schedules; archived baseline939updates under same cap. No automatic promotion to the full matrix.')
    write(report/'pilot_results.json',result)
    return result


if __name__=='__main__':
    import json
    result=analyze();print(json.dumps(dict(status=result['status'],summary=result['summary'],rows=[{k:r[k] for k in ('variant','seed','dice','gain_pp','gain_vs_original_bands_pp','ca4_dice')} for r in result['rows']])))
