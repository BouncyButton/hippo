"""Read-only audit of frozen bands schedules and archived segmentation errors."""
import hashlib
import json
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[2]
NAME = 'mni_constraint_target_search_f0s0_20261107'
OUT = ROOT / 'reports/mni_bands_implementation_audit_20261107'
BASELINES = {
    'nnunet': 'mni_nnunet_cap8_full_20261106',
    'swinunetr': 'mni_swin_log_dice_cap8_full_20261106',
    'unetr_vit_b16': 'mni_unetr_geo_cap8_20261106',
    'basic_unet16': 'mni_unet_cap8_20261106',
}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluation(folder):
    path = folder / 'validation_latest.json'
    manifest = read(folder / 'archive_manifest.json')
    assert digest(path) == manifest['metadata'][path.name]['sha256']
    data = read(path)
    matrix = [[sum(c['confusion'][i][j] for c in data['cases'])
               for j in range(4)] for i in range(4)]
    fn = sum(row[0] for row in matrix[1:])
    fp = sum(matrix[0][1:])
    swaps = sum(matrix[i][j] for i in range(1, 4) for j in range(1, 4) if i != j)
    assert fn == sum(c['foreground_fn'] for c in data['cases'])
    assert fp == sum(c['foreground_fp'] for c in data['cases'])
    assert fn + fp + swaps == sum(map(sum, matrix)) - sum(matrix[i][i] for i in range(4))
    return dict(run=folder.name, source=str(path.relative_to(ROOT)), sha256=digest(path),
                cases=len(data['cases']), fn=fn, fp=fp, swaps=swaps,
                external_share=(fn+fp)/(fn+fp+swaps),
                macro_dice=data['summary']['macro_hard_dice'],
                union_dice=data['summary']['whole_hippocampus_hard_dice'],
                confusion=matrix)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stage = ROOT / 'experiments' / NAME
    models = {}
    for model, name in BASELINES.items():
        runs = [evaluation(p.parent) for p in sorted((ROOT/'reports'/name/'runs').glob('*/validation_latest.json'))]
        assert len(runs) == 9
        fn, fp, swaps = (sum(r[k] for r in runs) for k in ('fn', 'fp', 'swaps'))
        cm = [[sum(r['confusion'][i][j] for r in runs) for j in range(4)] for i in range(4)]
        internal_pairs = {f'{i}<->{j}': cm[i][j]+cm[j][i]
                          for i in range(1, 4) for j in range(i+1, 4)}
        models[model] = dict(runs=runs, fn=fn, fp=fp, swaps=swaps,
            external_share=(fn+fp)/(fn+fp+swaps),
            fn_share_of_external=fn/(fn+fp), fp_share_of_external=fp/(fn+fp),
            external_share_range=[min(r['external_share'] for r in runs), max(r['external_share'] for r in runs)],
            mean_macro_dice=mean(r['macro_dice'] for r in runs),
            mean_union_dice=mean(r['union_dice'] for r in runs),
            internal_pairs=internal_pairs, confusion=cm)

    historical = []
    for row in read(ROOT/'reports/thesis_all_variant_errors_20261023/run_errors.json'):
        if row['campaign'] not in ('MSD_conventional', 'ADNI_conventional') or row['arm'] != 'baseline':
            continue
        assert digest(ROOT/row['source']) == row['sha256']
        historical.append(row)
    historical_summary = {}
    for dataset in ('MSD', 'ADNI'):
        historical_summary[dataset] = {}
        for model in BASELINES:
            rows = [r for r in historical if r['campaign']==dataset+'_conventional' and r['model']==model]
            assert len(rows)==9
            fn, fp, swaps = (sum(r[k] or 0 for r in rows) for k in ('fn','fp','swaps'))
            historical_summary[dataset][model] = dict(runs=9, fn=fn, fp=fp, swaps=swaps,
                external_share=(fn+fp)/(fn+fp+swaps), fn_share_of_external=fn/(fn+fp))

    schedules = {}
    for dataset, folder in [('MNI', NAME), ('MSD','msd_mixed_losses_nnunet60_others10_20261014'),
                            ('ADNI','adni_mixed_losses_nnunet60_others10_20261014')]:
        schedules[dataset] = {}
        for model in BASELINES:
            protocol = model + '_dice_bands_t' + ('60' if model=='nnunet' else '20') if dataset=='MNI' else model
            path = ROOT/'experiments'/folder/'protocols'/protocol/'protocol.json'
            p = read(path); s = p['models'][p['trainer_alias']]['objectives']['dice_bands']
            prefix, ramp, maximum = s['prefix_updates'], p['constraint_warmup_updates'], s['max_steps']
            schedules[dataset][model] = dict(prefix=prefix, ramp=ramp, first_active_update=prefix+1,
                first_full_update=prefix+ramp, total_updates=maximum,
                full_strength_updates=maximum-prefix-ramp+1,
                full_strength_fraction=(maximum-prefix-ramp+1)/maximum,
                ramp_weighted_update_equivalent=sum(min(1., (n-prefix)/ramp) for n in range(prefix+1,maximum+1)),
                protocol=str(path.relative_to(ROOT)), protocol_sha256=digest(path),
                gradient_target=p['calibration_target_ratio'], p95_cap=p['calibration_p95_cap'],
                lr_horizon=p['lr_horizon_updates'], budget=p['budget_flops'])

    constrained = []
    calibrations = []
    for folder in sorted((ROOT/'reports'/NAME/'runs').glob('MNI*')):
        if not (folder/'cleanup.json').exists():
            continue
        row = evaluation(folder); config = read(folder/'config.json'); model=config['model_identity']
        baseline=next(r for r in models[model]['runs'] if r['run'].endswith('f0s0'))
        row.update(model=model, objective=config['objective'],
            delta={k:row[k]-baseline[k] for k in ('fn','fp','swaps','macro_dice','union_dice')})
        constrained.append(row)
        c=read(folder/'calibration.json'); stats=c['statistics']['bands']
        target, cap = c['target_ratio'],c['p95_cap']; weight=c['weights']['bands']
        expected=min(target/stats['median_raw_ratio'], cap/stats['p95_raw_ratio'])
        assert abs(weight-expected)<1e-12
        assert c['anchor_successful_updates']==config['schedule']['prefix_updates']
        cosine=[]
        for case in c['cases']:
            gram=case['gradient_pair_means']
            cosine.append(gram['base:bands']/(gram['base:base']*gram['bands:bands'])**.5)
        calibrations.append(dict(run=folder.name, target=target, cap=cap, weight=weight,
            actual_median_ratio=stats['weighted_median_ratio'], actual_p95_ratio=stats['weighted_p95_ratio'],
            anchor_update=c['anchor_successful_updates'], median_logit_cosine=median(cosine),
            ceiling_binding=cap/stats['p95_raw_ratio']<target/stats['median_raw_ratio']))

    band_hashes={dataset:digest(ROOT/'experiments'/folder/'source/thesis/new_constraints/bands/outer_boundary.py')
        for dataset,folder in [('MNI',NAME),('MSD','msd_mixed_losses_nnunet60_others10_20261014'),
                              ('ADNI','adni_mixed_losses_nnunet60_others10_20261014')]}
    assert len(set(band_hashes.values()))==1
    result=dict(status='verified', baseline_scope='Final-budget validation; 3 folds x 3 seeds per model',
        error_definition='Each wrong voxel counted once: FG->BG FN, BG->FG FP, or FG->different-FG internal swap. External_share is (FN+FP)/(FN+FP+swaps), not spatial distance or contribution to macro Dice.',
        spatial_limitation='Archived confusion matrices do not establish distance to the contour or fraction of errors in the two-voxel band. Baseline predictions/checkpoints were retired under storage policy; no baseline replay performed.',
        class_ids={'0':'background','1':'CA1_3','2':'subiculum','3':'CA4_DG'},
        models=models, historical_baseline_errors=historical_summary, schedules=schedules,
        completed_constrained_comparisons=constrained, calibration_checks=calibrations,
        identical_band_source_sha256=band_hashes,
        calibration_limitation='One-time training-only full-resolution logit-gradient calibration; static coefficients, not a maintained parameter-gradient or optimizer-update ratio.',
        metadata_issue='MNI protocol calibration_policy is inherited as None: baseline only. Frozen executable and calibration receipts correctly use protocol target/cap; this is stale descriptive metadata, not disabled calibration.')
    (OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='verified',baseline_runs=36,constrained_runs=len(constrained),
        external_share={m:r['external_share'] for m,r in models.items()},output=str(OUT/'audit.json')),indent=2))


if __name__=='__main__':
    main()
