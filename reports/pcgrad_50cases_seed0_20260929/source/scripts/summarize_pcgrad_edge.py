"""Paired PCGrad pilot report, with selected and common epoch-60 endpoints."""
import argparse
import json
from pathlib import Path
import numpy as np
from summarize_separated_edge import KEYS,HIGHER


def summarize(root,previous):
    result=dict(status='complete',folds=[0],seeds=[0,1,2],arms=['sum','pcgrad'],comparisons={},
        limitations='Exploratory reused fold 0. Paired case bootstrap averages the three seeds first, conditions on fitted/selected models, and does not capture new-fold, new-training-set, retraining or selection uncertainty. Case-file independence is not verified patient independence.')
    index={}
    for seed in range(3):
        pair=root/f'fold0/seed{seed}'
        assert json.loads((pair/'completion.json').read_text())['status']=='complete'
        for arm in ('sum','pcgrad'):
            for endpoint,path in [('selected',pair/arm/'selected_cases.json'),('epoch60',pair/arm/'coherence/epoch_060.json')]:
                for row in json.loads(path.read_text()):
                    key=(endpoint,row['split'],seed,arm,row['case_name'])
                    assert key not in index
                    index[key]=row['metrics']
    rng=np.random.default_rng(20260930)
    for endpoint,split in [('selected','validation'),('epoch60','validation'),('selected','train')]:
        names=sorted({k[-1] for k in index if k[:2]==(endpoint,split)})
        n=len(names);assert n==(52 if split=='validation' else 10)
        draw=rng.integers(0,n,size=(10000,n))
        contrast={}
        for metric in KEYS:
            data={a:np.array([[index[endpoint,split,s,a,c][metric] for c in names] for s in range(3)],float) for a in ('sum','pcgrad')}
            assert all(np.isfinite(v).all() for v in data.values())
            delta=data['pcgrad']-data['sum']
            bootstrap=delta.mean(0)[draw].mean(1)
            contrast[metric]=dict(sum=float(data['sum'].mean()),pcgrad=float(data['pcgrad'].mean()),
                delta=float(delta.mean()),interval95=np.quantile(bootstrap,[.025,.975]).tolist(),
                seed_deltas=delta.mean(1).tolist(),seeds_improved=int(((delta.mean(1)>0) if metric in HIGHER else (delta.mean(1)<0)).sum()))
        result['comparisons'][endpoint+'_'+split]=contrast
    result['update_summaries']=[]
    result['learning_curves']=[]
    for seed in range(3):
        histories={}
        for arm in ('sum','pcgrad'):
            path=root/f'fold0/seed{seed}/{arm}'
            complete=json.loads((path/'completion.json').read_text())
            epochs=[json.loads(p.read_text()) for p in sorted((path/'epochs').glob('*.json'))]
            histories[arm]={r['epoch']:r['val_dice_hard'] for r in epochs if r['epoch']<=complete['stopped_epoch']}
            records=[r for p in sorted((path/'updates').glob('*.json')) for r in json.loads(p.read_text())['steps']]
            assert len(records)==10*complete['stopped_epoch']
            finite=[r for r in records if r['finite']]
            frequency={}
            for i,j in [(0,1),(0,2),(0,3),(0,4),(2,3),(2,4),(3,4)]:
                eligible=[r for r in finite if r['task_gram'][i][i]>0 and r['task_gram'][j][j]>0]
                frequency[f'{i}_{j}']=dict(negative=sum(r['task_gram'][i][j]<0 for r in eligible),count=len(eligible))
            ratios=[r['projected_over_original_norm'] for r in finite if r['projected_over_original_norm'] is not None]
            cosines=[r['cosine_with_original'] for r in finite if r['cosine_with_original'] is not None]
            result['update_summaries'].append(dict(seed=seed,arm=arm,updates=len(records),
                skipped_updates=sum(r['optimizer_step_skipped'] for r in records),
                fraction_updates_projected=float(np.mean([r['projection_count']>0 for r in finite])) if finite else None,
                median_projected_norm_ratio=float(np.median(ratios)) if ratios else None,
                median_cosine_to_original=float(np.median(cosines)) if cosines else None,
                gradient_conflicts=frequency,selected_epoch=complete['selected_epoch'],stopped_epoch=complete['stopped_epoch']))
        old=previous/f'fold0/seed{seed}/separated'
        rows=[json.loads(p.read_text()) for p in sorted((old/'epochs').glob('*.json'))]
        histories['previous_separated']={r['epoch']:r['val_dice_hard'] for r in rows}
        common=sorted(set.intersection(*(set(h) for h in histories.values())))
        assert 60 in common
        result['learning_curves'].append(dict(seed=seed,histories=histories,common_last_epoch=common[-1],
            common_last_dice={a:h[common[-1]] for a,h in histories.items()},
            sum_control_vs_previous_max_absolute_dice_drift=max(abs(histories['sum'][e]-histories['previous_separated'][e]) for e in common)))
    result['previous_pilot']=json.loads((previous/'SUMMARY.json').read_text())['comparisons']['selected_validation']
    result['conflict_confirmation']=json.loads((root/'CONFLICT_GATE.json').read_text())
    (root/'SUMMARY.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# PCGrad: fold 0, three seeds','',
        'Primary contrast: five weighted task gradients summed normally versus symmetric PCGrad on the same five gradients. '
        'Original separated-rule coefficients, training data, initialization, AMP, AdamW, learning-rate schedule and 75-epoch-cap stopping/selection policy are preserved. '
        'Projection order uses a separate reproducible RNG. No post-projection norm restoration is applied.','']
    for endpoint in ('selected_validation','epoch60_validation','selected_train'):
        lines+=['## '+endpoint.replace('_',' '),'',
                '| Metric | Sum control | PCGrad | Difference | Conditional 95% interval | Seeds improved |',
                '|---|---:|---:|---:|---|---:|']
        for metric,row in result['comparisons'][endpoint].items():
            lo,hi=row['interval95']
            lines.append(f"| {metric} | {row['sum']:.6g} | {row['pcgrad']:.6g} | {row['delta']:+.6g} | [{lo:+.6g},{hi:+.6g}] | {row['seeds_improved']}/3 |")
        lines+=['']
    lines+=['## Projection and numerical checks','',
            '| Seed | Arm | Updates | AMP skips | Fraction projected | Median norm ratio | Median cosine to sum |',
            '|---:|---|---:|---:|---:|---:|---:|']
    for r in result['update_summaries']:
        lines.append('| '+ ' | '.join(str(r[k]) for k in ('seed','arm','updates','skipped_updates','fraction_updates_projected','median_projected_norm_ratio','median_cosine_to_original'))+' |')
    lines+=['',result['limitations'],'',
        'Read coherence with correctness: fewer neighbor disagreements can coexist with more missed foreground. '
        'Evaluate FN, FP, correct boundary transitions and Dice jointly. Conflicts after sequential PCGrad projections can remain or reappear. '
        'Projection summaries describe pre-AdamW gradients, not guaranteed per-task improvement under the actual optimizer step.','',
        'No further folds or alternative gradient algorithms are launched automatically.','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    p.add_argument('--previous',type=Path,default=Path('/mnt/beegfsstudents/home/3160552/separated_edge_20260929_01'))
    a=p.parse_args();summarize(a.root,a.previous)
