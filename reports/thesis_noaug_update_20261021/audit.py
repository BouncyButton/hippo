"""Rebuild the retained no-augmentation thesis results, preserving campaign baselines."""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import math
import statistics as st

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CAP = 31976450667090
MODELS = ['nnunet', 'swinunetr', 'unetr_vit_b16', 'basic_unet16']
ARMS = ['baseline', 'bands', 'sdf', 'bands_sdf', 'bands_edge', 'bands_edge_sdf',
        'depth', 'bands_depth', 'edge_depth', 'sdf_depth', 'bands_edge_depth',
        'bands_sdf_depth', 'edge_sdf_depth', 'bands_edge_sdf_depth']
sources = {}
rows = []
checks = defaultdict(int)

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def readcsv(relative):
    p = ROOT / relative
    sources[relative] = digest(p)
    return list(csv.DictReader(p.open()))

def arm_name(a):
    return 'baseline' if a == 'dice' else a.removeprefix('dice_')

for dataset, path in [('MSD', 'reports/msd_3fold_3seed_final_20261020/all_504_runs.csv'),
                      ('ADNI', 'reports/adni_colab_final_20261019/all_504_runs.csv')]:
    data = readcsv(path)
    assert len(data) == 504
    for r in data:
        row = dict(study=dataset+'_capped', dataset=dataset, model=r['model'],
                   arm=arm_name(r['arm']), fold=int(r['fold']), seed=int(r['seed']),
                   dice=float(r['dice']), training_dice=float(r['training_dice']),
                   updates=int(r['updates']), training_flops=int(r['training_flops']),
                   calibration_flops=int(r['calibration_flops']),
                   fn=float(r['fn']), fp=float(r['fp']),
                   swaps=float(r['ap_swaps']) if dataset == 'MSD' else None,
                   error_unit='pooled validation voxels per run', source_id=r['id'])
        assert row['training_flops'] <= CAP
        rows.append(row)
        checks['capped_rows_within_budget'] += 1

data = readcsv('reports/medsam3_msd_colab_final_20261019/all_108_runs.csv')
assert len(data) == 108
for r in data:
    assert r['status'] == 'complete' and r['augmentation'] == 'none'
    assert int(r['updates']) == 200 and r['primary_checkpoint'] == 'final_update'
    rows.append(dict(study='MSD_MedSAM3', dataset='MSD', model='MedSAM3', arm=r['arm'],
                     fold=int(r['fold']), seed=int(r['seed']), dice=100*float(r['validation_dice']),
                     training_dice=100*float(r['training_dice']), updates=200,
                     training_flops=None, calibration_flops=None,
                     fn=float(r['foreground_fn']), fp=float(r['foreground_fp']),
                     swaps=float(r['ap_swaps']), error_unit='mean validation voxels per crop',
                     source_id=r['id']))

# The original MSD baseline/B/B+E experiment is complete and remains a
# separate study key: its baseline must never be replaced by the Colab one.
preserved_path = ROOT / 'reports/medsam3_msd_colab_108_3fold_3seed_noaug_20261015/bundle/existing_results.json'
preserved = json.loads(preserved_path.read_text())
sources[str(preserved_path.relative_to(ROOT))] = digest(preserved_path)
provenance = preserved['provenance']
source = ROOT / provenance['source']
assert digest(source) == provenance['source_sha256']
sources[provenance['source']] = digest(source)
aggregate = provenance['records']['analysis/aggregate.json']['summary']
assert aggregate['status'] == 'complete' and not aggregate['pilot_seed46_included']
assert len(preserved['runs']) == 27
for r in preserved['runs']:
    assert r['audit_status'] == 'passed' and r['updates'] == 200
    cell = next(v for v in aggregate['runs'] if (v['fold'], v['seed']) == (r['fold'], r['seed']))
    assert cell['status'] == 'passed'
    for split, field in [('validation', 'validation_dice'), ('train', 'training_dice')]:
        assert math.isclose(r[field], cell['summary'][split][r['arm']]['ap_mean_dice'], abs_tol=1e-12)
    for field in ['foreground_fn', 'foreground_fp', 'ap_swaps']:
        assert math.isclose(r[field], cell['summary']['validation'][r['arm']][field], abs_tol=1e-12)
    record = provenance['records'][f"results/fold{r['fold']}_seed{r['seed']}/results.json"]['summary']
    train = record['training'][r['arm']]
    assert record['status'] == 'complete' and record['protocol']['updates'] == 200
    assert train['frozen_before'] == train['frozen_after']
    assert train['adapter_before'] != train['adapter_after']
    checks['preserved_msd_medsam3_runs_reconciled_with_passed_audit'] += 1
    rows.append(dict(study='MSD_MedSAM3_original', dataset='MSD', model='MedSAM3', arm=r['arm'],
                     fold=r['fold'], seed=r['seed'], dice=100*r['validation_dice'],
                     training_dice=100*r['training_dice'], updates=200,
                     training_flops=None, calibration_flops=None,
                     fn=r['foreground_fn'], fp=r['foreground_fp'], swaps=r['ap_swaps'],
                     error_unit='mean validation voxels per crop', source_id=r['id']))

adni_root = ROOT / 'reports/adni_medsam3_2sample_3fold_3seed_14arms_20261018'
schedules = defaultdict(set)
supports = defaultdict(set)
runtimes = []
hardware = set()
for p in sorted((adni_root/'runs').iterdir()):
    if not p.is_dir():
        continue
    manifest = json.loads((p/'archive_manifest.json').read_text())
    for filename, expected in manifest['metadata'].items():
        f = p/filename
        assert f.stat().st_size == expected['bytes'], f
        assert digest(f) == expected['sha256'], f
        checks['adni_medsam3_archive_file_hashes_verified'] += 1
    sources[str((p/'archive_manifest.json').relative_to(ROOT))] = digest(p/'archive_manifest.json')
    protocol = json.loads((p/'protocol.json').read_text())
    train = json.loads((p/'training_summary.json').read_text())
    evaluation = json.loads((p/'evaluation.json').read_text())
    assert protocol['augmentation'] == train['augmentation'] == 'none'
    assert protocol['updates'] == train['updates'] == 200
    assert protocol['primary_checkpoint'] == 'final_update'
    assert train['frozen_before'] == train['frozen_after']
    assert train['adapter_before'] != train['adapter_after']
    assert len(protocol['train_cases']) == 2 and len(protocol['evaluation_cases']) == 54
    assert not set(protocol['train_cases']) & set(protocol['evaluation_cases'])
    schedules[(protocol['fold'], protocol['seed'])].add(train['schedule_sha256'])
    supports[protocol['fold']].add(tuple(protocol['train_cases']))
    split_records = defaultdict(list)
    for r in evaluation['records']:
        cm = r['confusion']
        tp, fp, fn = cm[1][1], cm[0][1], cm[1][0]
        d = 2*tp/(2*tp+fp+fn)
        assert math.isclose(d, r['macro_hard_dice'], abs_tol=1e-12)
        assert fp == r['foreground_fp'] and fn == r['foreground_fn']
        split_records[r['split']].append(r)
        checks['adni_medsam3_case_confusions_verified'] += 1
    for split, rr in split_records.items():
        assert math.isclose(st.mean(r['macro_hard_dice'] for r in rr),
                            evaluation['mean_dice'][split], abs_tol=1e-12)
    assert len(split_records['validation']) == 54 and len(split_records['training_support']) == 2
    vv = split_records['validation']
    rows.append(dict(study='ADNI_MedSAM3', dataset='ADNI', model='MedSAM3',
                     arm=arm_name(protocol['arm']), fold=protocol['fold'], seed=protocol['seed'],
                     dice=100*evaluation['mean_dice']['validation'],
                     training_dice=100*evaluation['mean_dice']['training_support'], updates=200,
                     training_flops=None, calibration_flops=None,
                     fn=st.mean(r['foreground_fn'] for r in vv),
                     fp=st.mean(r['foreground_fp'] for r in vv), swaps=None,
                     error_unit='mean validation voxels per crop', source_id=p.name))
    runtimes.append(train['elapsed_seconds'])
    hardware.add((p/'runtime_training_environment.json').read_text())
assert all(len(v) == 1 for v in schedules.values())
assert all(len(v) == 1 for v in supports.values())
assert len(rows) == 1269
assert len({(r['study'], r['model'], r['arm'], r['fold'], r['seed']) for r in rows}) == 1269
expected_cells = {
    'MSD_capped': ([0, 1, 2], [0, 2, 20]),
    'ADNI_capped': ([0, 1, 2], [0, 1, 4]),
    'MSD_MedSAM3': ([0, 1, 3], [17, 83, 191]),
    'MSD_MedSAM3_original': ([0, 1, 3], [17, 83, 191]),
    'ADNI_MedSAM3': ([0, 1, 3], [0, 1, 2]),
}

groups = defaultdict(list)
baselines = {(r['study'],r['model'],r['fold'],r['seed']):r for r in rows if r['arm']=='baseline'}
for r in rows:
    b = baselines[r['study'],r['model'],r['fold'],r['seed']]
    r['gain_pp'] = r['dice'] - b['dice']
    r['role'] = ('additional_matched_control' if r['study']=='MSD_MedSAM3' and r['arm']=='baseline'
                 else 'planned_objective_result')
    groups[r['study'],r['model'],r['arm']].append(r)
summaries = []
for (study, model, arm), rr in groups.items():
    assert len(rr) == 9
    ff, seeds = expected_cells[study]
    assert {(r['fold'], r['seed']) for r in rr} == {(f, seed) for f in ff for seed in seeds}
    folds = sorted({r['fold'] for r in rr})
    fm = [st.mean(r['dice'] for r in rr if r['fold']==f) for f in folds]
    fg = [st.mean(r['gain_pp'] for r in rr if r['fold']==f) for f in folds]
    b = groups[study,model,'baseline']
    row = dict(study=study,model=model,arm=arm,n=9,dice=st.mean(fm),sd_fold=st.stdev(fm),
               gain_pp=st.mean(fg),sd_fold_gain=st.stdev(fg),
               training_dice=st.mean(r['training_dice'] for r in rr),
               wins=sum(r['gain_pp']>0 for r in rr),positive_folds=sum(x>0 for x in fg),
               min_gain=min(r['gain_pp'] for r in rr),max_gain=max(r['gain_pp'] for r in rr),
               folds=folds,fold_dice=fm,fold_gain=fg,min_updates=min(r['updates'] for r in rr),
               max_updates=max(r['updates'] for r in rr),error_unit=rr[0]['error_unit'])
    row['role'] = rr[0]['role']
    for key in ['fn','fp','swaps']:
        row[key] = st.mean(r[key] for r in rr) if rr[0][key] is not None else None
        row[key+'_change_pct'] = (100*(row[key]/st.mean(r[key] for r in b)-1)
                                 if row[key] is not None else None)
    for key in ['training_flops','calibration_flops']:
        row[key] = st.mean(r[key] for r in rr) if rr[0][key] is not None else None
    summaries.append(row)

def savecsv(name, records):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)

assert len(summaries) == 141
assert {r['arm'] for r in rows if r['study'].startswith('MSD_MedSAM3')} == set(ARMS)
savecsv('all_1269_runs.csv',rows)
savecsv('all_141_condition_summaries.csv',summaries)
primary_rows = [r for r in rows if r['role']=='planned_objective_result']
primary_summaries = [r for r in summaries if r['role']=='planned_objective_result']
assert len(primary_rows)==1260 and len(primary_summaries)==140
savecsv('planned_1260_runs.csv',primary_rows)
savecsv('planned_140_condition_summaries.csv',primary_summaries)
(OUT/'results.json').write_text(json.dumps(dict(rows=rows,summaries=summaries),indent=2)+'\n')
audit=dict(total_runs=len(rows),total_conditions=len(summaries),checks=dict(checks),
           planned_objective_runs=len(primary_rows),planned_objective_conditions=len(primary_summaries),
           additional_matched_control_runs=len(rows)-len(primary_rows),
           sources_sha256=sources,
           adni_medsam3_supports={str(f):list(next(iter(v))) for f,v in supports.items()},
           adni_medsam3_training_seconds=dict(min=min(runtimes),median=st.median(runtimes),max=max(runtimes)),
           limitations=['Conventional MSD seeds 0/2/20 and ADNI 0/1/4 were selected after observing earlier seeds; exploratory.',
                        'MSD MedSAM3 covers all 14 objectives across two completed campaigns (27 original, 108 fresh); each has its own matched baseline.',
                        'The 27 original MSD MedSAM3 rows match preserved passed audits and source-report hash; deleted raw model tensors are not re-audited.',
                        'Existing final audits retained for conventional and MSD MedSAM3 sources; not every original payload rehashed here.',
                        'ADNI MedSAM3 metadata hashes and per-case Dice recomputed for all 126 archives; deleted tensors are not available.'])
(OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
for study in sorted({r['study'] for r in rows}):
    print(study)
    for model in MODELS+['MedSAM3']:
        ss=[s for s in summaries if s['study']==study and s['model']==model]
        if not ss:continue
        b=next(s for s in ss if s['arm']=='baseline');best=max(ss,key=lambda s:s['dice'])
        print(model,'baseline',round(b['dice'],4),'best',best['arm'],round(best['dice'],4),
              'gain',round(best['gain_pp'],4),'wins',best['wins'],'fold gains',best['fold_gain'],
              'FN/FP/swaps changes',[best[k+'_change_pct'] for k in ['fn','fp','swaps']])
print(json.dumps({k:v for k,v in audit.items() if k!='sources_sha256'},indent=2))
