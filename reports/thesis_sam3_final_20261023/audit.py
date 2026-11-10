"""Verify final SAM3 archives and rebuild thesis statistics without inference."""
from pathlib import Path
from collections import Counter, defaultdict
from statistics import mean, stdev
import csv
import hashlib
import json
import zipfile

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CAMPAIGN = ROOT / 'reports/sam3_geofwi_f013_s012_126_20261019'
SPLIT = CAMPAIGN / 'split_cluster_20261022'
FINAL = SPLIT / 'final_results_20261023'
OLD = ROOT / 'reports/background_dice_archive_recovery_20261023'
METRICS = ['foreground_dice', 'background_dice', 'dice_including_background']

def read(p):
    return json.loads(p.read_text())

def digest(b):
    return hashlib.sha256(b).hexdigest()

def write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2) + '\n')

def writecsv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)

def fold_stats(rows, metric):
    vals = {str(f): mean(r[metric] for r in rows if int(r['fold']) == f)
            for f in sorted({int(r['fold']) for r in rows})}
    return mean(vals.values()), stdev(vals.values()), vals

def main():
    for n, h in read(FINAL / 'artifact_checksums.json').items():
        assert digest((FINAL / n).read_bytes()) == h, n
    final = read(FINAL / 'results.json')
    published = {r['task']: r for r in final['records']}
    plan = read(CAMPAIGN / 'bundle/plan.json')['tasks']
    assert len(plan) == len(published) == 126
    assert set(published) == {r['id'] for r in plan}
    archives = {}
    state = read(SPLIT / 'restart_colab_20261022/completion/campaign_state.json')
    for p in OLD.glob('remote/SAM3_colab_archives/*.zip'):
        with zipfile.ZipFile(p) as z:
            n = next(x for x in z.namelist() if x.endswith('/archive_manifest.json'))
            rid = json.loads(z.read(n))['task']['id']
            assert rid not in archives
            archives[rid] = p
    assert len(archives) == 76
    runs, cases, provenance, missing_metadata = [], [], [], []
    checked = 0
    signatures = defaultdict(list)
    fold_sets = defaultdict(set)
    for task in plan:
        rid = task['id']
        if rid in archives:
            p = archives[rid]
            with zipfile.ZipFile(p) as z:
                prefix = f'work/{rid}/'
                blobs = {n[len(prefix):]: z.read(n) for n in z.namelist() if n.startswith(prefix)}
            receipt = read(OLD / 'extracted/SAM3' / rid / 'recovery_receipt.json')
            assert digest(p.read_bytes()) == receipt['sha256']
            remote = state['objects']['completed/' + rid]
            assert remote['local_sha256'] == remote['appProperties']['sha256'] == receipt['sha256']
            assert state['completed'][rid]['updates'] == 25
            platform = 'Colab'
        else:
            p = SPLIT / 'cluster_results/runs' / rid
            blobs = {str(f.relative_to(p)): f.read_bytes() for f in p.rglob('*') if f.is_file()}
            platform = 'cluster'
            ack = json.loads(blobs['archive_acknowledgment.json'])
            assert ack['status'] == 'verified' and ack['source_verified']
            assert ack['manifest_sha256'] == digest(blobs['archive_manifest.json'])
        manifest = json.loads(blobs['archive_manifest.json'])
        assert manifest['task']['id'] == rid
        metadata = manifest.get('metadata', manifest.get('files', {}))
        for n, info in metadata.items():
            if n in blobs:
                assert digest(blobs[n]) == (info['sha256'] if isinstance(info, dict) else info), (rid, n)
                checked += 1
            else:
                missing_metadata.append([rid, n])
        required = ['evaluation.json', 'completion.json', 'protocol.json', 'training_summary.json',
                    'runtime_recipe.json', 'slab_schedule.json', 'training_inventory.json']
        assert all(n in blobs and n in metadata for n in required)
        c, protocol, train, runtime = [json.loads(blobs[n]) for n in
            ['completion.json', 'protocol.json', 'training_summary.json', 'runtime_recipe.json']]
        assert c['status'] == 'complete' and c['updates'] == train['updates'] == protocol['updates'] == 25
        for field in ['fold', 'seed', 'arm']:
            assert c[field] == task[field]
        assert c['protocol_sha256'] == digest(blobs['protocol.json'])
        assert train['frozen_before'] == train['frozen_after']
        assert train['augmentation'] == protocol['augmentation'] == 'none'
        assert train['schedule_sha256'] == digest(blobs['slab_schedule.json'])
        assert len(protocol['train_cases']) == 5 and len(protocol['evaluation_cases']) == 96
        assert not set(protocol['train_cases']) & set(protocol['evaluation_cases'])
        fold_sets[task['fold']].add((tuple(protocol['train_cases']), tuple(protocol['evaluation_cases'])))
        signatures[task['fold'], task['seed']].append({
            'id': rid, 'adapter': train['adapter_before'], 'schedule': train['schedule_sha256'],
            'base': protocol['base_sha256'], 'frozen': train['frozen_before'],
            'inventory': digest(blobs['training_inventory.json'])})
        data = json.loads(blobs['evaluation.json'])['records']
        assert len(data) == 101
        groups = defaultdict(list)
        seen = set()
        for rec in data:
            cm = rec['confusion']
            assert len(cm) == 2 and all(len(a) == 2 for a in cm)
            assert all(isinstance(v, int) and v >= 0 for a in cm for v in a)
            assert sum(map(sum, cm)) == 96**3
            identity = rec['split'], rec['case_name']
            assert identity not in seen
            seen.add(identity)
            dice = []
            for k in range(2):
                den = sum(cm[k]) + sum(a[k] for a in cm)
                dice.append(2 * cm[k][k] / den if den else 1.)
            assert abs(dice[1] - rec['macro_hard_dice']) < 1e-12
            for k, label in enumerate(['background', 'salt']):
                assert abs(dice[k] - rec['per_class'][label]['dice']) < 1e-12
            assert rec['foreground_fn'] == cm[1][0] and rec['foreground_fp'] == cm[0][1]
            row = {'run_id': rid, 'fold': task['fold'], 'seed': task['seed'], 'arm': task['arm'],
                   'case': rec['case_name'], 'split': rec['split'], 'foreground_dice': dice[1],
                   'background_dice': dice[0], 'dice_including_background': mean(dice),
                   'fn': cm[1][0], 'fp': cm[0][1]}
            groups[rec['split']].append(row)
            cases.append(row)
        assert {r['case'] for r in groups['training_support']} == set(protocol['train_cases'])
        assert {r['case'] for r in groups['validation']} == set(protocol['evaluation_cases'])
        val = groups['validation']
        row = {'campaign': 'GeoFWI3D_SAM3', 'model': 'SAM3', 'run_id': rid, 'fold': task['fold'],
               'seed': task['seed'], 'arm': task['arm'], 'platform': platform,
               **{m: mean(r[m] for r in val) for m in METRICS},
               'training_dice': mean(r['foreground_dice'] for r in groups['training_support']),
               'fn': mean(r['fn'] for r in val), 'fp': mean(r['fp'] for r in val)}
        for k in ['validation_dice', 'training_dice']:
            actual = row['foreground_dice'] if k == 'validation_dice' else row[k]
            assert abs(actual - c[k]) < 1e-12 and abs(actual - published[rid][k]) < 1e-12
        for k in ['fn', 'fp']:
            assert sum(r[k] for r in val) == c['foreground_' + k] == published[rid]['foreground_' + k]
        assert published[rid]['platform'] == platform
        runs.append(row)
        provenance.append({'run_id': rid, 'source': str(p.relative_to(ROOT)),
            'evaluation_sha256': digest(blobs['evaluation.json']),
            'frozen_unchanged': True, 'gpu': runtime['binding']['environment']['gpu'],
            'source_sha256': runtime.get('source_sha256'),
            'elapsed_seconds_recorded': train.get('elapsed_seconds'),
            'repeated_update_attempts_upper_bound': published[rid]['repeated_update_attempts_upper_bound']})
    assert all(len(v) == 1 for v in fold_sets.values())
    pairing = []
    for cell, values in signatures.items():
        assert len(values) == 14
        for k in ['adapter', 'schedule', 'base', 'frozen', 'inventory']:
            distinct = {v[k] for v in values}
            assert len(distinct) == 1, (cell, k, values)
        pairing.append({'fold': cell[0], 'seed': cell[1], **values[0]})
    base = {(r['fold'], r['seed']): r for r in runs if r['arm'] == 'dice'}
    for r in runs:
        r['delta'] = r['foreground_dice'] - base[r['fold'], r['seed']]['foreground_dice']
    summaries = []
    for item in final['summary']:
        arm = item['recipe']; rs = [r for r in runs if r['arm'] == arm]
        assert {(r['fold'], r['seed']) for r in rs} == {(f,s) for f in [0,1,3] for s in [0,1,2]}
        row = {'arm': arm, 'runs': 9, 'positive': sum(r['delta'] > 0 for r in rs)}
        for m in [*METRICS, 'training_dice', 'delta', 'fn', 'fp']:
            row[m], row[m+'_fold_sd'], row[m+'_fold_means'] = fold_stats(rs, m)
        assert abs(row['foreground_dice']*100-item['mean_dice_percent']) < 1e-10
        assert abs(row['delta']*100-item['mean_delta_pp']) < 1e-10
        assert row['positive'] == item['wins']
        summaries.append(row)
    # Retain the earlier recovery as a dated snapshot, replacing SAM3 only in this supplement.
    with (OLD/'background_dice_runs.csv').open() as f:
        combined = [r for r in csv.DictReader(f) if r['campaign'] != 'GeoFWI3D_SAM3']
    for r in combined:
        for m in METRICS: r[m] = float(r[m])
        for m in ['fold', 'seed']: r[m] = int(r[m])
    combined += runs
    conditions = defaultdict(list)
    for r in combined: conditions[r['campaign'],r['model'],r['arm']].append(r)
    effects, flips, selected = [], [], []
    for group in sorted({k[:2] for k in conditions}):
        keys = [k for k in conditions if k[:2] == group]
        means = {k: {m: fold_stats(conditions[k],m)[0] for m in METRICS} for k in keys}
        bk = next(k for k in keys if k[2] in ['dice', 'baseline'])
        b = {(r['fold'],r['seed']):r for r in conditions[bk]}
        best = max(keys,key=lambda k:means[k]['foreground_dice'])
        best_all = max(keys,key=lambda k:means[k]['dice_including_background'])
        selected.append({'campaign': group[0], 'model': group[1], 'foreground_best': best[2],
                         'all_class_best': best_all[2], 'background_improves': means[best]['background_dice'] > means[bk]['background_dice']})
        for k in keys:
            assert len(conditions[k]) == 9
            if k == bk: continue
            changes = []
            for r in conditions[k]:
                d = {m:r[m]-b[r['fold'],r['seed']][m] for m in METRICS}
                changes.append(d)
                if d[METRICS[0]]*d[METRICS[2]] < 0:
                    flips.append({'campaign':k[0],'model':k[1],'arm':k[2], 'fold':r['fold'],'seed':r['seed'], **d})
            effects.append({'campaign':k[0],'model':k[1],'arm':k[2],
                **{'delta_'+m:means[k][m]-means[bk][m] for m in METRICS},
                **{'positive_'+m:sum(d[m]>0 for d in changes) for m in METRICS}})
    sensitivity = {'comparisons':len(effects),'individual_pairs':len(effects)*9,
        'positive_foreground':sum(r['delta_foreground_dice']>0 for r in effects),
        'positive_all_class':sum(r['delta_dice_including_background']>0 for r in effects),
        'mean_sign_changes':[r for r in effects if r['delta_foreground_dice']*r['delta_dice_including_background']<0],
        'individual_sign_changes':flips,'selected':selected}
    assert len(effects) == 193 and sensitivity['positive_foreground'] == sensitivity['positive_all_class'] == 179
    assert not sensitivity['mean_sign_changes'] and len(flips) == 6
    assert len(selected) == 15 and all(r['foreground_best'] == r['all_class_best'] and r['background_improves'] for r in selected)
    for r in selected:
        e = next(e for e in effects if (e['campaign'],e['model'],e['arm']) == (r['campaign'],r['model'],r['foreground_best']))
        expected = 8 if (r['campaign'],r['model']) in [('GeoFWI3D_conventional','unetr_vit_b16'),('GeoFWI3D_SAM3','SAM3')] else 9
        assert e['positive_foreground_dice'] == e['positive_dice_including_background'] == expected
    audit = {'runs':len(runs),'platforms':dict(Counter(r['platform'] for r in runs)),
        'volume_records':len(cases),'validation_records':sum(r['split']=='validation' for r in cases),
        'metadata_hashes_checked':checked,'unavailable_manifest_metadata':missing_metadata,
        'colab_archive_hashes_matched_to_cloud_receipts':76,'cluster_verified_archive_acknowledgements':50,
        'pairing_cells_verified':len(pairing),'all_frozen_parameter_hashes_unchanged':True,
        'final_summary_scores_verified':True,'summary_sd_note':'Recomputed SD across three seed-averaged fold means; upstream recipe_summary SD is across nine runs.',
        'combined_recovered_runs':len(combined),'combined_validation_records':124860-116*96+126*96,
        'repeated_attempts_upper_bound':sum(p['repeated_update_attempts_upper_bound'] for p in provenance),
        'gpu_counts':dict(Counter(p['gpu'] for p in provenance)),
        'source_identities':dict(Counter(p['source_sha256'] for p in provenance)),
        'sources':{str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in [FINAL/'results.json',CAMPAIGN/'bundle/plan.json',OLD/'background_dice_runs.csv']},
        'issues':[]}
    write('verification.json',audit);write('sam3_conditions.json',summaries)
    write('sam3_provenance.json',provenance);write('paired_identity_checks.json',pairing)
    write('metric_sensitivity.json',sensitivity)
    writecsv('sam3_runs.csv',runs);writecsv('sam3_cases.csv',cases)
    writecsv('background_metric_comparison.csv',effects)
    print(json.dumps(audit,indent=2))
    for r in summaries:
        print(r['arm'],*[round(r[m]*100,4) for m in ['foreground_dice','foreground_dice_fold_sd','delta','delta_fold_sd','training_dice','background_dice','dice_including_background']],r['positive'])
    print('SENSITIVITY',json.dumps(sensitivity,indent=2))

if __name__ == '__main__':
    main()
