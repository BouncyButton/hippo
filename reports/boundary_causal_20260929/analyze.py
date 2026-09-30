"""Summarize matched pilot case metrics; never use voxels as replications."""
from pathlib import Path
import json
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parent
METRICS = ['macro_dice','union_dice','boundary_union_errors','all_fp','all_fn','ap_swaps',
    'balanced_boundary_error','inner_both_correct','outer_both_correct','cross_correct_transition',
    'bands_bce','bands_truth','assd_voxels','surface_dice_1voxel']


def read(p): return json.loads(p.read_text())


def summarize(rows):
    result = {}
    for split in ('train','validation'):
        rr = [r for r in rows if r['split']==split]
        assert len(rr)==(50 if split=='train' else 52)
        assert len({r['case_name'] for r in rr})==len(rr)
        keys = set.intersection(*(set(r['metrics']) for r in rr))
        result[split] = dict(cases=len(rr), mean={k:float(np.mean([r['metrics'][k] for r in rr]))
            for k in METRICS if k in keys and all(r['metrics'][k] is not None for r in rr)},
            totals={k:sum(r['metrics'][k] for r in rr) for k in
                ('boundary_union_errors','all_fp','all_fn','ap_swaps','all_wrong') if k in keys})
    return result


def pair(a, b):
    """Candidate B minus control A; a paired case interval is descriptive."""
    results={}
    for split in ('train','validation'):
        aa={r['case_name']:r['metrics'] for r in a if r['split']==split}
        bb={r['case_name']:r['metrics'] for r in b if r['split']==split}
        assert set(aa)==set(bb)
        results[split]={}
        for k in METRICS:
            if not all(k in aa[n] and k in bb[n] for n in aa): continue
            d=np.array([bb[n][k]-aa[n][k] for n in sorted(aa)])
            rng=np.random.default_rng(20260929)
            means=d[rng.integers(len(d),size=(10000,len(d)))].mean(1)
            results[split][k]=dict(mean_change=float(d.mean()),
                descriptive_case_interval95=np.quantile(means,[.025,.975]).tolist(),
                cases_decrease=int((d<0).sum()),cases_equal=int((d==0).sum()),cases_increase=int((d>0).sum()))
    return results


def main():
    result=dict(models={},comparisons={},fixed_epochs={},source_hashes={},
        uncertainty='Intervals resample paired cases within this reused validation fold. They omit checkpoint selection, study selection, training-seed variability, and unknown subject dependence. Epochs are not independent replications.')
    records={}
    for folder in sorted((ROOT/'results').glob('seed*/*')):
        done=folder/'completion.json'
        if not done.exists(): continue
        key=f'{folder.parent.name}/{folder.name}'
        cp=read(done); rows=read(folder/'selected_cases.json')
        result['source_hashes'][str(done.relative_to(ROOT))]=hashlib.sha256(done.read_bytes()).hexdigest()
        result['source_hashes'][str((folder/'selected_cases.json').relative_to(ROOT))]=hashlib.sha256((folder/'selected_cases.json').read_bytes()).hexdigest()
        assert result['source_hashes'][str((folder/'selected_cases.json').relative_to(ROOT))]==cp['selected_cases_sha256']
        result['models'][key]=dict(selected_epoch=cp['selected_epoch'],stopped_epoch=cp['stopped_epoch'],
            selected_updates=cp['selected_epoch']*50,stopped_updates=cp['stopped_epoch']*50,**summarize(rows))
        records[key]=rows
        result['fixed_epochs'][key]={p.stem:summarize(read(p)) for p in sorted((folder/'audits').glob('epoch_*.json'))}
    old=ROOT.parent/'pcgrad_50cases_seed0_20260929/supplement/results/pcgrad50.json'
    prior=read(old)
    rows=[]
    for r in prior['cases']:
        m=r['recomputed_metrics'].copy()
        m.update(boundary_union_errors=m['inner_fn']+m['outer_fp'],
            bands_bce=r['bands']['case_loss'], bands_truth=r['bands']['truth'],
            assd_voxels=r['surface']['symmetric_mean_surface_distance_voxels'])
        rows.append(dict(split=r['split'],case_name=r['case_name'],metrics=m))
    records['seed0/pcgrad_historical']=rows
    result['models']['seed0/pcgrad_historical']=dict(selected_epoch=prior['selected_epoch'],**summarize(rows))
    for seed in (0,1,2):
        for a,b in [('sum','sum_aug'),('dice','dice_aug'),('dice','sum'),('dice_aug','sum_aug'),('pcgrad_historical','sum')]:
            ka,kb=f'seed{seed}/{a}',f'seed{seed}/{b}'
            if ka in records and kb in records:
                result['comparisons'][f'{kb}_minus_{a}']=pair(records[ka],records[kb])
    (ROOT/'RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Matched boundary pilot: current completed results','',
        'Case means for scores; validation error counts are totals across52 cases. All use identity-input audits. No incomplete arm is summarized.','',
        '| Model | Epoch best/stop | Train Dice% | Val Dice% | Val union Dice% | Shell FP+FN | Val cross correctness% | Val bands BCE |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for key,r in result['models'].items():
        t,v=r['train']['mean'],r['validation']['mean']
        lines.append(f"| {key} | {r['selected_epoch']}/{r.get('stopped_epoch','historical')} | {100*t['macro_dice']:.4f} | {100*v['macro_dice']:.4f} | {100*v['union_dice']:.4f} | {r['validation']['totals']['boundary_union_errors']} | {100*v['cross_correct_transition']:.3f} | {v['bands_bce']:.4f} |")
    lines+=['','Pairwise effects (candidate minus control):','']
    for name,c in result['comparisons'].items():
        lines.append(f'## {name}')
        lines.append('')
        for metric in ('macro_dice','boundary_union_errors','assd_voxels','bands_bce'):
            if metric not in c['validation']:continue
            r=c['validation'][metric]
            lines.append(f"- {metric}: {r['mean_change']:.6f}, descriptive paired-case95% interval {r['descriptive_case_interval95']}")
        lines.append('')
    lines += [result['uncertainty'],'']
    (ROOT/'CURRENT_RESULTS.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:12]))


if __name__=='__main__':main()
