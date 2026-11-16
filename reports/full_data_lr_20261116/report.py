"""Verify compact result archives and report the whole cohort without picking seeds."""
import csv,hashlib,json,math,statistics,zipfile
from collections import defaultdict
from pathlib import Path
D=Path(__file__).resolve().parent
plan=json.loads((D/'plan.json').read_bytes())
old={r['cell']:r for r in csv.DictReader((D.parent/'full_data_baselines_20261115/paired_comparisons.csv').open())}
groups=defaultdict(list)
for k,r in plan.items():
    base=old[k];score=None
    if r['status']=='reuse':score=float(base['development_peak_lr_0_001_pct'])
    else:
        f=D/'archive'/(k+'_full_lr_v1_results.zip')
        if f.is_file():
            with zipfile.ZipFile(f) as z:
                assert z.testzip() is None
                rec=json.loads(z.read('lr_contrast_receipt.json'))
                assert rec['key']==k and json.loads(z.read('lr_contrast_plan.json'))==r
                assert rec['plan_sha256']==hashlib.sha256(json.dumps(r,separators=(',',':')).encode()).hexdigest()
                score=100*rec['paired_dice']; assert math.isfinite(score) and 0<=score<=100
    ref=base['native_pass_full_pct'] or base['paper_matched_full_pct']
    groups[r['dataset'],r['model']].append((score,float(ref) if ref else None))
rows=['# Full-data learning-rate results','', 'Dice in percent. Means are shown only after all nine paired runs are archived. Original full-data controls and higher-rate development runs remain separate.','','| Dataset | Model | Higher-rate archives | Original full | Higher-rate full | Change (pp) |','|---|---|---:|---:|---:|---:|']
for (ds,m),v in sorted(groups.items()):
    n=sum(s is not None for s,_ in v)
    avg=lambda a:f'{statistics.mean(a):.2f}'
    s=avg([s for s,b in v]) if n==9 else '—'
    b=avg([b for s,b in v]) if all(b is not None for s,b in v) else '—'
    d=avg([s-b for s,b in v]) if n==9 and b!='—' else '—'
    rows.append(f'| {ds} | {m} | {n}/9 | {b} | {s} | {d} |')
tmp=D/'results.partial';tmp.write_text('\n'.join(rows)+'\n');tmp.replace(D/'results.md')
