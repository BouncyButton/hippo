"""Summarize the explicitly limited fold-0, three-seed paired pilot."""
import argparse
import json
from pathlib import Path
import numpy as np

KEYS = ['macro_dice','union_dice','all_fp','all_fn','inner_disagree','outer_disagree',
        'inner_both_correct','outer_both_correct','cross_correct_transition',
        'inner_soft_squared_error','outer_soft_squared_error','cross_soft_squared_error',
        'all_soft_squared_error','foreground_components_6','foreground_outside_largest_6']
HIGHER = {'macro_dice','union_dice','inner_both_correct','outer_both_correct','cross_correct_transition'}


def mean_available(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def summarize(root):
    index = {}
    for seed in range(3):
        pair = root / f'fold0/seed{seed}'
        assert json.loads((pair/'completion.json').read_text())['status'] == 'complete'
        for arm in ('pooled','separated'):
            output = pair/arm
            for horizon, path in [('selected',output/'selected_cases.json'),
                                  ('epoch60',output/'coherence/epoch_060.json')]:
                rows = json.loads(path.read_text())
                for r in rows:
                    key = (horizon, r['split'], seed, arm, r['case_name'])
                    assert key not in index
                    index[key] = r['metrics']
    result = {'status':'complete','folds':[0],'seeds':[0,1,2], 'comparisons':{},
        'limitations':'Exploratory development-fold pilot. Case bootstrap averages repeated seeds first and conditions on fitted models. It excludes training-subset, retraining, checkpoint-selection and new-fold uncertainty; case files are not verified independent patients.'}
    rng = np.random.default_rng(20260929)
    for horizon, split in [('selected','validation'),('selected','train'),('epoch60','validation')]:
        names = sorted({k[-1] for k in index if k[:2] == (horizon,split)})
        assert len(names) == (52 if split=='validation' else 10)
        n = len(names)
        draws = rng.integers(0,n,size=(10000,n))
        contrast = {}
        for metric in KEYS:
            a = {arm: np.array([[index[horizon,split,s,arm,name][metric] for name in names]
                                for s in range(3)],float) for arm in ('pooled','separated')}
            assert all(np.isfinite(v).all() for v in a.values())
            d = a['separated'] - a['pooled']
            samples = d.mean(0)[draws].mean(1)
            contrast[metric] = dict(pooled=float(a['pooled'].mean()),separated=float(a['separated'].mean()),
                delta=float(d.mean()), interval95=np.quantile(samples,[.025,.975]).tolist(),
                seed_deltas=d.mean(1).tolist(),
                seeds_improved=int(((d.mean(1)>0) if metric in HIGHER else (d.mean(1)<0)).sum()))
        result['comparisons'][horizon+'_'+split] = contrast
    checks = result['comparisons']['selected_validation']
    result['descriptive_success_checks'] = {
        'inner_disagreement_lower':checks['inner_disagree']['delta']<0,
        'FN_no_higher':checks['all_fn']['delta']<=0,
        'FP_no_higher':checks['all_fp']['delta']<=0,
        'boundary_transitions_no_lower':checks['cross_correct_transition']['delta']>=0,
        'macro_Dice_no_lower':checks['macro_dice']['delta']>=0}
    telemetry = []
    for seed in range(3):
        for arm in ('pooled','separated'):
            for path in sorted((root/f'fold0/seed{seed}'/arm/'gradients').glob('epoch_*.json')):
                row = json.loads(path.read_text())
                telemetry.append(dict(seed=seed,arm=arm,epoch=row['epoch'],
                    parameter_edge_over_base=mean_available([r['parameters']['edge_over_base'] for r in row['cases']]),
                    logit_edge_over_base=mean_available([r['logits']['edge_over_base'] for r in row['cases']]),
                    parameter_inner_dice_cosine=mean_available([r['parameters']['cosines']['dice__inner'] for r in row['cases']]),
                    parameter_outer_dice_cosine=mean_available([r['parameters']['cosines']['dice__outer'] for r in row['cases']]),
                    parameter_cross_dice_cosine=mean_available([r['parameters']['cosines']['dice__cross'] for r in row['cases']])))
    result['gradient_trajectories'] = telemetry
    original = json.loads((root/'original_coherence.json').read_text())
    original_rows = original['rows']
    assert len(original_rows) == 3*3*62
    result['historical_validation'] = {}
    for method in ('dice','bands','edge'):
        rows = [r for r in original_rows if r['method']==method and r['split']=='validation']
        assert len(rows)==3*52
        result['historical_validation'][method] = {k:float(np.mean([r['metrics'][k] for r in rows])) for k in KEYS}
    result['historical_provenance'] = original['provenance']
    result['learning_curves'] = []
    for seed in range(3):
        pair = root/f'fold0/seed{seed}'
        historic = json.loads((pair/'historical_curves.json').read_text())
        histories = {m:{int(r['epoch']):float(r['val_dice_hard']) for r in v['rows']} for m,v in historic.items()}
        for arm in ('pooled','separated'):
            done = json.loads((pair/arm/'completion.json').read_text())
            rows = [json.loads(p.read_text()) for p in sorted((pair/arm/'epochs').glob('*.json'))]
            histories[arm] = {r['epoch']:r['val_dice_hard'] for r in rows if r['epoch']<=done['stopped_epoch']}
        common = sorted(set.intersection(*(set(h) for h in histories.values())))
        assert 60 in common
        result['learning_curves'].append(dict(seed=seed,histories=histories,common_last_epoch=max(common),
            common_last_dice={m:h[max(common)] for m,h in histories.items()},
            epoch60_dice={m:h[60] for m,h in histories.items()},
            pooled_minus_original_edge_common_epochs={str(e):histories['pooled'][e]-histories['edge'][e] for e in sorted(histories['pooled'].keys() & histories['edge'].keys())}))
    (root/'SUMMARY.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines = ['# Separated edge rules: fold 0, three seeds','',
        'Paired training from scratch: pooled edges versus separately averaged inner, outer and crossing rules. '
        'Identical data, initialization seeds, bands coefficients and training settings; training-only matched median logit-gradient budget. '
        'Equal rule weights do not imply equal parameter-gradient strengths.','',
        'No further folds are submitted automatically.','']
    lines += ['## Original 75-epoch-cap reference runs','',
        'Same fold-0 case set, three seeds and original selected checkpoints. Historical spatial metrics come from the completed coherence audit; small repeat-inference drift was disclosed there. '
        'The rerun keeps the original edge coefficient. Its trajectory is compared with the original edge run to expose any reproducibility difference.','',
        '| Metric | Original baseline | Original bands | Original bands+edge | Pooled rerun | Separated |',
        '|---|---:|---:|---:|---:|---:|']
    for metric in KEYS:
        old=[result['historical_validation'][m][metric] for m in ('dice','bands','edge')]
        new=[result['comparisons']['selected_validation'][metric][m] for m in ('pooled','separated')]
        lines.append('| '+metric+' | '+' | '.join(f'{v:.6g}' for v in old+new)+' |')
    lines += ['']
    for name in ('selected_validation','epoch60_validation','selected_train'):
        lines += ['## '+name.replace('_',' '),'',
                  '| Metric | Pooled | Separated | Difference | Conditional 95% interval | Seeds improved |',
                  '|---|---:|---:|---:|---|---:|']
        for metric,r in result['comparisons'][name].items():
            lo,hi = r['interval95']
            lines.append(f"| {metric} | {r['pooled']:.6g} | {r['separated']:.6g} | {r['delta']:+.6g} | [{lo:+.6g}, {hi:+.6g}] | {r['seeds_improved']}/3 |")
        lines += ['']
    lines += ['The success checks below describe mean directions; they are not significance tests or guarantees.','']
    lines += [f'- {k}: {v}' for k,v in result['descriptive_success_checks'].items()]
    lines += ['',result['limitations'],'',
        'Gradient trajectories and per-case parameter/logit norms and cosines are retained alongside the results. '
        'A negative cosine indicates local conflict at the probe point, not a demonstrated harmful AdamW update.','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    summarize(args.root)
