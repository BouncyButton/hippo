"""Rebuild GeoFWI3D thesis tables from the final fixed 504-cell inventory."""
from pathlib import Path
import csv
import hashlib
import json
import statistics as st

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'reports/geofwi3d_reweighted_full_suite_20261021'
CAP = 182178126543160
MODELS = ['nnunet', 'swinunetr', 'unetr_vit_b16', 'basic_unet16']
ARMS = ['dice', 'dice_bands', 'dice_sdf', 'dice_bands_sdf', 'dice_bands_edge',
        'dice_bands_edge_sdf', 'dice_depth', 'dice_bands_depth', 'dice_edge_depth',
        'dice_sdf_depth', 'dice_bands_edge_depth', 'dice_bands_sdf_depth',
        'dice_edge_sdf_depth', 'dice_bands_edge_sdf_depth']

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    source = read(SOURCE / 'final_summary.json')
    expected = {(m,a,f,s) for m in MODELS for a in ARMS for f in range(3) for s in range(3)}
    rows = [dict(r, arm='dice') for r in source['baselines']] + [dict(r) for r in source['rows']]
    keys = [(r['model'],r['arm'],r['fold'],r['seed']) for r in rows]
    assert len(keys) == len(set(keys)) == 504 and set(keys) == expected
    needed_hashes = {r['archive_manifest_sha256'] for r in rows if 'archive_manifest_sha256' in r}
    archive_map = {}
    for folder in sorted((ROOT / 'reports').glob('geofwi3d_*')):
        for manifest in folder.rglob('archive_manifest.json'):
            digest = sha(manifest)
            if digest in needed_hashes:
                archive_map.setdefault(digest, manifest.parent)
    missing = needed_hashes - archive_map.keys()
    assert all(r['arm']=='dice' and r['fold']==0 for r in rows
               if r.get('archive_manifest_sha256') in missing)
    baseline_only = []
    verified_files = 0
    unavailable_metadata = []
    case_count = 0
    split_inventory = {}
    for row in rows:
        if row.get('archive_manifest_sha256') in missing:
            baseline_only.append({k:row[k] for k in ('model','fold','seed','source','archive_manifest_sha256')})
            assert row['total_flops'] <= CAP
            row.update(archive=None,validation_fn=None,validation_fp=None,
                       augmentation='xy_d4_v1' if row['model']=='unetr_vit_b16' else 'none')
            continue
        if 'archive_manifest_sha256' in row:
            archive = archive_map[row['archive_manifest_sha256']]
        elif row['arm']=='dice':
            archive = ROOT / 'reports/geofwi3d_frozen_3fold_3seed_20261020/runs' / row['id']
        else:
            archive = SOURCE / 'completed_evidence/archives' / row['id']
        manifest = read(archive / 'archive_manifest.json')
        for name, info in manifest['metadata'].items():
            path = archive / name
            if not path.exists():
                assert row['arm']=='dice' and row['fold']==0
                unavailable_metadata.append(str(path.relative_to(ROOT)))
                continue
            assert path.stat().st_size == info['bytes'] and sha(path) == info['sha256'], path
            verified_files += 1
        val = read(archive / 'validation_latest.json')
        assert len(val['cases']) == 96
        dices = []
        for case in val['cases']:
            cm = case['confusion']
            tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
            assert tn + fp + fn + tp == 96 ** 3
            dice = 2 * tp / (2 * tp + fp + fn)
            assert abs(dice - case['macro_hard_dice']) < 1e-10
            assert fp == case['foreground_fp'] and fn == case['foreground_fn']
            dices.append(dice)
        score = st.mean(dices)
        assert abs(score - row['final_dice']) < 1e-10
        assert abs(score - val['summary']['macro_hard_dice']) < 1e-10
        case_count += len(dices)
        done = read(archive / 'completion.json')
        assert done['status'] == 'complete' and done['updates'] == row['updates']
        assert done['consumed_total_flops'] == row['total_flops'] <= CAP
        assert done['consumed_training_flops'] + done['calibration_flops'] == row['total_flops']
        row.update(archive=str(archive.relative_to(ROOT)),
                   validation_fn=sum(c['foreground_fn'] for c in val['cases']),
                   validation_fp=sum(c['foreground_fp'] for c in val['cases']),
                   calibration_flops=done['calibration_flops'],
                   training_flops=done['consumed_training_flops'])
        config = read(archive / 'config.json')
        train, validation = config['train_cases'], config['validation_cases']
        assert len(train) == 19 and len(validation) == 96 and not set(train) & set(validation)
        assert set(validation) == {c['case_name'] for c in val['cases']}
        pair = {'train': train, 'val': validation}
        if row['fold'] in split_inventory:
            assert split_inventory[row['fold']] == pair
        else:
            split_inventory[row['fold']] = pair
        row['augmentation'] = config['augmentation']
        assert row['augmentation'] == ('xy_d4_v1' if row['model'] == 'unetr_vit_b16' else 'none')
        row['initial_lr'] = config['initial_lr']
        if row['arm'] != 'dice':
            cal = read(archive / 'calibration.json')
            assert len(cal['cases']) == 19
            row['calibration_target'] = cal['target_ratio']
            row['calibration_p95_cap'] = cal['p95_cap']
    baselines = {(r['model'],r['fold'],r['seed']):r for r in rows if r['arm']=='dice'}
    for r in rows:
        base = baselines[r['model'],r['fold'],r['seed']]
        r['paired_gain_pp'] = 100 * (r['final_dice'] - base['final_dice'])
        if 'delta_dice_percentage_points' in r:
            assert abs(r['paired_gain_pp'] - r['delta_dice_percentage_points']) < 1e-10
    summaries = []
    for model in MODELS:
        for arm in ARMS:
            group = [r for r in rows if (r['model'],r['arm'])==(model,arm)]
            folds = [st.mean(r['final_dice']*100 for r in group if r['fold']==f) for f in range(3)]
            gains = [st.mean(r['paired_gain_pp'] for r in group if r['fold']==f) for f in range(3)]
            summaries.append(dict(model=model,arm=arm,mean=st.mean(folds),fold_sd=st.stdev(folds),
                                  fold_means=folds,delta=st.mean(gains),fold_gain_sd=st.stdev(gains),fold_gains=gains,
                                  positive_pairs=sum(r['paired_gain_pp']>0 for r in group),
                                  mean_validation_fn=st.mean(r['validation_fn'] for r in group) if all(r['validation_fn'] is not None for r in group) else None,
                                  mean_validation_fp=st.mean(r['validation_fp'] for r in group) if all(r['validation_fp'] is not None for r in group) else None,
                                  updates_min=min(r['updates'] for r in group),updates_max=max(r['updates'] for r in group)))
    for model in MODELS:
        for arm in source['models'][model]['arms']:
            s = next(s for s in summaries if (s['model'],s['arm'])==(model,arm['arm']))
            assert abs(s['mean']-arm['mean'])<1e-10 and abs(s['delta']-arm['delta'])<1e-10
    def write_csv(name, records):
        columns = list(dict.fromkeys(k for r in records for k in r))
        with (OUT/name).open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader();writer.writerows(records)
    write_csv('all_504_runs.csv',rows)
    write_csv('all_56_condition_summaries.csv',summaries)
    (OUT/'results.json').write_text(json.dumps({'rows':rows,'summaries':summaries,'splits':split_inventory},indent=2)+'\n')
    audit = dict(status='passed',runs=504,conditions=56,metadata_files_verified=verified_files,
                 validation_case_confusions_recomputed=case_count,cap_flops=CAP,
                 raw_run_archives_rechecked=len(rows)-len(baseline_only),
                 baseline_metadata_not_available_locally=unavailable_metadata,
                 saved_summary_baselines_without_local_raw_archive=baseline_only,
                 cap_includes_calibration=True,uncapped_costs='Evaluation, preflight and recipe-development computation',
                 dispersion='Sample SD of three seed-averaged fold means; not nine-cell SD',
                 source_sha256=sha(SOURCE/'final_summary.json'),
                 original_result_validation=read(SOURCE/'final_result_validation.json'),
                 validation_sets_pairwise_disjoint=all(not set(split_inventory[i]['val'])&set(split_inventory[j]['val'])
                                                     for i in range(3) for j in range(i)),
                 distinct_support_scenes=len(set(c for s in split_inventory.values() for c in s['train'])),
                 caveat='Recipes and calibration strengths selected after exploratory validation; no independent test set.')
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))
    for m in MODELS:
        print(m, max((s for s in summaries if s['model']==m),key=lambda x:x['mean']))

if __name__ == '__main__':
    main()
