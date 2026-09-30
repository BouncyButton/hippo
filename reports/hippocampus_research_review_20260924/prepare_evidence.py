"""Normalize preserved experiment records without rerunning or selecting models."""
from pathlib import Path
import json
import hashlib
import statistics as st

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
def read(path):
    return json.loads((ROOT/path).read_text())

consolidated_path = 'experiments/consolidated_convergence_20260920/CONSOLIDATED.json'
original = read(consolidated_path)
edge = json.loads((OUT/'evidence/edge_audits.json').read_text())
runs = []
for source in original['runs']:
    best = next(c for c in source['checkpoints'] if c['key']=='best')
    last = next(c for c in source['checkpoints'] if c['key']=='latest')
    r = {k: source[k] for k in ('budget','train_cases','fold','seed','augmentation','last_logged_epoch','early_stopping_patience','step_size','path')}
    r.update(method='Baseline' if source['arm']=='none' else 'Bands', selected=best['epoch'],
             splits=best['summary'], final_splits=last['summary'], checkpoints=source['checkpoints'],
             evidence=consolidated_path)
    runs.append(r)

def from_audit(a):
    m=a['mean_case_metrics'];p=a['pooled_regions'];whole=p['all']
    return {'dice_pct':100*m['macro_dice']['mean'],'fp':whole['fp'],'fn':whole['fn'],
            'swaps':whole['ap_swap'],'errors':whole['class_errors'],'cases':a['case_count'],
            'outer_band_fp':p['outer']['fp'],'inner_band_fn':p['inner']['fn'],
            'far_fp':p['far_background']['fp'],'core_fn':p['deep_foreground']['fn'],
            'boundary_union_errors':p['boundary']['union_errors'],
            'balanced_boundary_error':m['balanced_boundary_error']['mean']}

for key,x in edge.items():
    fold,seed=map(int,key.split('/'))
    for method,name in [('Baseline','dice'),('Bands','bands'),('Bands + edge','edge')]:
        matches=[r for r in runs if r['budget']==75 and r['train_cases']==10 and r['fold']==fold and r['seed']==seed and r['augmentation']=='none' and r['method']==method]
        l=x['learning']['models'][name]
        if matches:r=matches[0]
        else:
            r=dict(budget=75,train_cases=10,fold=fold,seed=seed,augmentation='none',method=method,
                   early_stopping_patience=8,step_size=20,path=x['root']+'/run',checkpoints=[])
            runs.append(r)
        r.update(selected=l['best_epoch'],last_logged_epoch=l['stopped_epoch'],
                 splits={'train_clean':from_audit(x['splits']['train'][name]),'validation':from_audit(x['splits']['validation'][name])},
                 evidence='evidence/edge_audits.json',learning=l)

for r in runs:
    matches=[b for b in runs if all(b[k]==r[k] for k in ('budget','train_cases','fold','seed')) and b['augmentation']=='none' and b['method']=='Baseline']
    r['delta_unaug_baseline_pp']=r['splits']['validation']['dice_pct']-matches[0]['splits']['validation']['dice_pct'] if matches else None
    r['optimizer_updates_at_stop']=r['last_logged_epoch']*r['train_cases']
    for split,s in r['splits'].items():
        s.setdefault('errors',s['fp']+s['fn']+s['swaps'])
        assert s['errors']==s['fp']+s['fn']+s['swaps'],(r['path'],split)
        s['error_shares_pct']={k:100*s[k]/s['errors'] if s['errors'] else 0 for k in ('fp','fn','swaps')}
        s['errors_per_case']=s['errors']/s['cases']

runs.sort(key=lambda r:(r['budget'],r['train_cases'],r['fold'],r['seed'],r['augmentation']!='none',{'Baseline':0,'Bands':1,'Bands + edge':2}[r['method']]))
for i,r in enumerate(runs,1):r['id']=f'R{i:02d}'

full=read('experiments/baseline_replication_20260916/three_seed_comparison/comparison.json')['records']
full_edge=read('docs/experiments/edge_consistency_20260924/GENERALIZATION_RESULTS.json')
contour=json.loads((OUT/'evidence/contour_inventory.json').read_text())['files']['robustness_summary.json']
component=read('experiments/presence_decisive_20260923/results_audit_666651/summary.json')
full_models={name:{split:from_audit(full_edge[f'{split}/summary.json'][name]) for split in ('train','validation')} for name in ('dice','bands','edge')}
sources=[consolidated_path,'experiments/baseline_replication_20260916/three_seed_comparison/comparison.json',
         'docs/experiments/edge_consistency_20260924/GENERALIZATION_RESULTS.json',
         'experiments/presence_decisive_20260923/results_audit_666651/summary.json',
         'thesis/new_constraints/edge_consistency.py','thesis/new_constraints/bands/outer_boundary.py']
manifest={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
result={'runs':runs,'full_aug_comparison':full,'full_edge':full_models,'contour':contour,
        'contour_components':component,'source_hashes':manifest,'counts':{'runs':len(runs),'master75':sum(r['budget']==75 for r in runs)}}
(OUT/'evidence/normalized.json').write_text(json.dumps(result,indent=2)+'\n')
print(result['counts'])
for f in (0,1,2):
    print('fold',f,[(m,st.mean(r['splits']['validation']['dice_pct'] for r in runs if r['budget']==75 and r['train_cases']==10 and r['fold']==f and r['augmentation']=='none' and r['method']==m)) for m in ('Baseline','Bands','Bands + edge')])
