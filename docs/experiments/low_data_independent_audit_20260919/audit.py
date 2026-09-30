"""Read-only audit of archived experiments; write derived evidence beside this script.

No training, remote access, or modification of the original handoff/results.
Uses Python's standard library and reconstructs hard Dice from confusion counts.
"""
import csv
import hashlib
import json
import math
import statistics as st
from pathlib import Path

EXP = Path('/Users/filippofocaccia/.codex/worktrees/5063/hippo/experiments')
OUT = Path(__file__).resolve().parent
REMOTE = Path('/mnt/beegfsstudents/home/3160552')
NAMES = [
    'final_four_fold_report_20260918',
    'low_data_noaug_25pct_fold0_seed0_20260918',
    'low_data_noaug_12p5pct_fold0_seed0_20260918',
    'low_data_noaug_5pct_200ep_es_fold0_seed0_20260918',
    'low_data_noaug_5pct_200ep_es_fold0_seeds12_20260918',
    'low_data_noaug_5pct_400ep_es_fold0_seed2_20260918',
]


def read(p):
    return json.loads(p.read_text())


def rows(p):
    with p.open() as f:
        return list(csv.DictReader(f))


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def close(a, b, tol=2e-7):
    assert abs(a-b) < tol, (a, b)


def dice(cm):
    return st.mean(2 * cm[k][k] / (sum(cm[k]) + sum(r[k] for r in cm)) for k in (1, 2))


def quantile(x, q):
    x = sorted(x)
    i = (len(x)-1) * q
    lo = math.floor(i)
    hi = math.ceil(i)
    return x[lo] + (i-lo) * (x[hi]-x[lo])


def main():
    checks = {'hashes': [], 'evaluations': 0, 'case_scores': 0,
              'max_case_dice_error': 0, 'runs': 0, 'calibrations': 0}
    results = []
    histories = {}
    cohorts = {}
    specifications = {}
    for name in NAMES:
        verified = EXP / name / 'verified'
        root = verified / 'results'
        manifest = read(verified / 'FILES.json')
        for entry in manifest:
            path = root / entry['relative']
            assert path.stat().st_size == entry['bytes'], path
            assert digest(path) == entry['sha256'], path
        checks['hashes'].append({'archive': name, 'verified_files': len(manifest)})
        remote = read(verified / 'REMOTE_AUDIT.json')
        assert not remote['failed_checks']
        records = {}
        for rec in remote['runs']:
            path = Path(rec['run'])
            if path.parent.name != 'runs':
                continue
            arm = 'bands' if path.name.startswith('bands') else 'baseline'
            cfg = rec['config']
            spec = cfg['run']
            seed, fold = spec['seed'], spec['fold']
            local = root / path.relative_to(REMOTE)
            assert read(local / 'config.json') == cfg
            history = rows(local / 'metrics.csv')
            scores = [float(r['val_dice_hard']) for r in history]
            best_epoch = max(range(len(scores)), key=scores.__getitem__) + 1
            assert best_epoch == rec['completion']['selected_epoch']
            assert len(history) == rec['completion']['epoch']
            ref, ref_epoch = -math.inf, 0
            for epoch, row in enumerate(history, 1):
                assert int(row['epoch']) == epoch
                lr = spec['learning_rate'] * spec['adamw_gamma'] ** ((epoch-1)//spec['step_size'])
                close(float(row['learning_rate']), lr, 1e-12)
                close(float(row['train_loss']), float(row['train_supervised_loss']) +
                      float(row['constraint_scale']) * float(row['train_constraint_loss']))
                close(float(row['constraint_scale']), min(epoch/5, 1), 1e-12)
                if arm == 'baseline':
                    assert float(row['train_constraint_loss']) == 0
                else:
                    close(float(row['train_constraint_loss']), spec['constraint_config']['bands_weight'] *
                          float(row['train_outer_boundary_band_raw_loss']), 2e-8)
                if scores[epoch-1] > ref + spec['early_stopping_min_delta']:
                    ref, ref_epoch = scores[epoch-1], epoch
                stopped = (epoch >= spec['early_stopping_min_epochs'] and
                           epoch-ref_epoch >= spec['early_stopping_patience'])
                assert not stopped or epoch == len(history)
            assert stopped or len(history) == spec['epochs']
            base = local.parent.parent
            audit_dir = base / 'audits' / f'seed{seed}'
            if not audit_dir.exists():
                audit_dir = base / 'bands_audit'
            evaluations = {}
            for kind in ('best', 'latest'):
                ev = read(audit_dir / f'{arm}_{kind}.json')
                assert ev['epoch'] == (best_epoch if kind == 'best' else len(history))
                for split in ('train', 'validation'):
                    cases = ev[split]['cases']
                    expected_n = cfg['train_samples'] if split == 'train' else cfg['validation_samples']
                    assert len(cases) == expected_n
                    for case in cases:
                        error = abs(dice(case['confusion']) - case['dice_hard'])
                        checks['max_case_dice_error'] = max(error, checks['max_case_dice_error'])
                        assert error < 2e-7
                        checks['case_scores'] += 1
                    rebuilt = st.mean(dice(c['confusion']) for c in cases)
                    close(rebuilt, ev[split]['dice_hard'])
                    summed = [[sum(c['confusion'][i][j] for c in cases) for j in range(3)] for i in range(3)]
                    assert summed == ev[split]['confusion']
                assert not ({c['case_name'] for c in ev['train']['cases']} &
                            {c['case_name'] for c in ev['validation']['cases']})
                evaluations[kind] = ev
                checks['evaluations'] += 1
            if arm == 'bands':
                cal = read(root / Path(spec['bands_calibration']).relative_to(REMOTE))
                d = [c['dice_gradient_rms'] for c in cal['cases']]
                b = [c['band_gradient_rms_unconditional'] for c in cal['cases']]
                weight = min(.10*st.median(d)/st.median(b), .50*st.median(d)/quantile(b,.95))
                close(weight, spec['constraint_config']['bands_weight'], 1e-12)
                train_ids = {c['case_name'] for c in evaluations['best']['train']['cases']}
                assert set(cal['training_case_ids']) <= train_ids
                checks['calibrations'] += 1
            records[fold,seed,arm] = (cfg, history, evaluations)
            histories[name,fold,seed,arm] = history
            specifications[name,fold,seed,arm] = spec
            cohorts[name,fold,seed,arm] = {
                split:sorted(c['case_name'] for c in evaluations['best'][split]['cases'])
                for split in ('train','validation')}
            checks['runs'] += 1
        for fold,seed in sorted({(f,s) for f,s,a in records}):
            bcfg,bhist,bev = records[fold,seed,'baseline']
            ncfg,nhist,nev = records[fold,seed,'bands']
            assert cohorts[name,fold,seed,'baseline'] == cohorts[name,fold,seed,'bands']
            assert bcfg['execution_provenance'].get('cuda_device_name') == ncfg['execution_provenance'].get('cuda_device_name')
            diff = [k for k in bcfg['run'] if bcfg['run'][k] != ncfg['run'][k]]
            assert set(diff) == {'constraint_set','constraint_config','bands_calibration','bands_calibration_sha256'}
            assert {k for k in bcfg['run']['constraint_config'] if bcfg['run']['constraint_config'][k] != ncfg['run']['constraint_config'][k]} == {'bands_weight'}
            item = {'experiment':name, 'fold':fold,'seed':seed,'cases':bcfg['train_samples'],
                    'schedule':bcfg['run']['epochs'], 'step_size':bcfg['run']['step_size'],
                    'pair_config_differences':diff, 'arms':{}}
            for arm,cfg,hist,ev in [('baseline',bcfg,bhist,bev),('bands',ncfg,nhist,nev)]:
                item['arms'][arm] = {
                    'best_dice_pct':100*st.mean(dice(c['confusion']) for c in ev['best']['validation']['cases']),
                    'final_dice_pct':100*st.mean(dice(c['confusion']) for c in ev['latest']['validation']['cases']),
                    'best_epoch':ev['best']['epoch'], 'stop_epoch':len(hist),
                    'updates':len(hist)*cfg['train_samples'],
                    'lr_sum_normalized':sum(float(r['learning_rate'])/1e-4*cfg['train_samples'] for r in hist),
                    'final_train_loss':float(hist[-1]['train_supervised_loss']),
                    'first_reach_epoch':{str(t):next((int(r['epoch']) for r in hist if float(r['val_dice_hard'])*100 >= t),None) for t in (74,76,78,79,80)},
                    'device':cfg['execution_provenance'].get('cuda_device_name'),
                }
            for kind in ('best','latest'):
                bm = {c['case_name']:dice(c['confusion']) for c in bev[kind]['validation']['cases']}
                nm = {c['case_name']:dice(c['confusion']) for c in nev[kind]['validation']['cases']}
                assert bm.keys() == nm.keys()
                item[kind+'_gain_pp'] = 100*st.mean(nm[k]-bm[k] for k in bm)
                item[kind+'_improved_cases'] = sum(nm[k]>bm[k] for k in bm)
            if name == NAMES[0]:
                saved = next(x for x in read(EXP/name/'SUMMARY.json')['pairs'] if x['fold']==fold and x['seed']==seed)
            else:
                saved = read(EXP/name/'RESULTS.json')
                if 'per_seed' in saved:
                    saved = saved['per_seed'][str(seed)]
                saved = {'best_gain_pp':saved['comparisons']['best']['gain_pp'],
                         'latest_gain_pp':saved['comparisons']['latest']['gain_pp']}
            for k in ('best_gain_pp','latest_gain_pp'):
                close(item[k],saved[k],2e-5)
            results.append(item)
    old = [r for r in results if r['experiment']==NAMES[0]]
    fold0 = [r for r in old if r['fold']==0]
    uniform200 = [r for r in results if r['schedule']==200]
    mixed = [r for r in results if r['schedule']==200 and r['seed']!=2 or r['schedule']==400]
    def aggregate(rr):
        return {'pairs':len(rr), 'best_gain_pp':st.mean(r['best_gain_pp'] for r in rr),
                'final_gain_pp':st.mean(r['latest_gain_pp'] for r in rr),
                **{arm:{'best_mean_pct':st.mean(r['arms'][arm]['best_dice_pct'] for r in rr),
                        'best_descriptive_sd_across_pairs_pp':st.stdev(r['arms'][arm]['best_dice_pct'] for r in rr)} for arm in ('baseline','bands')}}
    c10,c26,c52 = [cohorts[n,0,0,'baseline'] for n in (NAMES[0],NAMES[2],NAMES[1])]
    assert set(c10['train']) < set(c26['train']) < set(c52['train'])
    assert c10['validation'] == c26['validation'] == c52['validation']
    for r in results:
        if r['schedule'] > 75:
            assert cohorts[r['experiment'],0,r['seed'],'baseline'] == c10
            current = specifications[r['experiment'],0,r['seed'],'baseline']
            original = specifications[NAMES[0],0,r['seed'],'baseline']
            differences = {k for k in original if original[k] != current[k]} - {'runtime_sha256','execution_sha256'}
            assert differences == {'epochs','step_size','early_stopping_min_epochs'}
    assert len({s['source_sha256'] for s in specifications.values()}) == 1
    checks['long_baseline_config_changes_only_schedule_and_execution_provenance'] = True
    checks['same_frozen_source_digest_all_production_runs'] = True
    checks['nested_fraction_subsets_and_constant_validation'] = True
    checks['long_runs_use_same_ten_case_subset'] = True
    handoff = read(EXP/'low_data_convergence_findings_20260919/FINDINGS.json')
    matched = []
    name_by_n = {10:NAMES[0],26:NAMES[2],52:NAMES[1]}
    for p in handoff['matched_training_budget_analysis']['points']:
        n,center = p['cases'],p['center_epoch']
        name = name_by_n[n]
        b,nr = [histories[name,0,0,a] for a in ('baseline','bands')]
        window = range(center-2,center+3)
        gain = 100*st.mean(float(nr[e-1]['val_dice_hard'])-float(b[e-1]['val_dice_hard']) for e in window)
        close(gain,p['mean_gain_pp'],1e-10)
        lr_sum = sum(float(r['learning_rate'])/1e-4*n for r in b[:center])
        close(lr_sum,p['lr_budget'],1e-10)
        matched.append({**p,'window_epochs':[center-2,center+2],
                        'window_update_range':[(center-2)*n,(center+2)*n]})
    timings = []
    for name in NAMES[3:]:
        root = EXP/name/'verified/results'
        for log in root.glob('*/*.out'):
            groups = []
            for line in log.read_text().splitlines():
                try:
                    r = json.loads(line)
                except (ValueError,TypeError):
                    continue
                if not isinstance(r,dict) or 'timing_epoch' not in r:
                    continue
                if r['timing_epoch'] == 1:
                    groups.append([])
                groups[-1].append(r)
            if len(groups) != 3 or len(groups[1]) != 5:
                continue
            seed_candidates = [r for r in results if r['experiment']==name and
                               len(groups[0])==r['arms']['baseline']['stop_epoch'] and
                               len(groups[2])==r['arms']['bands']['stop_epoch']]
            # A seed-specific filename disambiguates identical schedule lengths.
            seed_candidates = [r for r in seed_candidates if f'seed{r["seed"]}' in log.name or len(seed_candidates)==1]
            if len(seed_candidates)!=1:
                continue
            pair = seed_candidates[0]
            entry = {'experiment':name,'seed':pair['seed'],'log':str(log),
                     'calibration_source_epoch_seconds':sum(r['epoch_seconds'] for r in groups[1]),
                     'limitation':'Epoch timings only: excludes startup, gradient calibration, final evaluation, and queue time.',
                     'arms':{}}
            for arm,group in [('baseline',groups[0]),('bands',groups[2])]:
                targets = pair['arms'][arm]['first_reach_epoch']
                entry['arms'][arm] = {
                    'median_training_seconds_after_epoch1':st.median(r['training_seconds'] for r in group[1:]),
                    'logged_epoch_seconds_to_target':{t:sum(r['epoch_seconds'] for r in group[:ep]) if ep else None for t,ep in targets.items()}}
            timings.append(entry)
    payload = {'scope':'Local archived artifacts only; no independent GPU training/inference or current cluster check.',
               'checks':checks,'pairs':results,'matched_budget_windows':matched,
               'exploratory_timings':timings,'aggregate':{'four_fold_75':aggregate(old),
               'fold0_75':aggregate(fold0),'fold0_uniform_200':aggregate(uniform200),
               'fold0_adaptive_200_200_400':aggregate(mixed)}}
    (OUT/'AUDIT.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({'checks':checks,'aggregate':payload['aggregate']},indent=2))


if __name__ == '__main__':
    main()
