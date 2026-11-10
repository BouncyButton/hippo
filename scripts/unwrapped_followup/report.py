"""Keep matched loss, seed, and gate comparisons separate."""
import json
from pathlib import Path
import numpy as np
from scripts.depth_supervision.utils import atomic_json


def report(p,root):
    records={};paths={};pairs=[]
    for fraction,arms in p['pairs'].items():
        for arm in ['dice','depth',*arms]:
            key=f'prior/{fraction}/{arm}';path=Path(p['prior_root'])/'runs'/fraction/arm
            records[key]=json.loads((path/'completion.json').read_text());paths[key]=path
    for e in p['runs']:
        path=root/'runs'/e['id']
        if (path/'completion.json').exists():
            records[e['id']]=json.loads((path/'completion.json').read_text());paths[e['id']]=path
    def entry(f,group,seed,arm,gate='none'):
        return next(e['id'] for e in p['runs'] if (e['fraction'],e['group'],e['seed'],e['arm'],e['gate'])==(f,group,seed,arm,gate))
    for f,arms in p['pairs'].items():
        for arm in arms:
            pairs.append(('remove_depth',f'prior/{f}/{arm}',entry(f,'dice_ablation',0,arm)))
            pairs.append(('dice_branch_vs_native',f'prior/{f}/dice',entry(f,'dice_ablation',0,arm)))
            for mode in ('entropy','mean_entropy'):
                pairs.append(('gate_vs_original',f'prior/{f}/{arm}',entry(f,'uncertainty',0,arm,mode)))
            pairs.append(('spatial_vs_mean_gate',entry(f,'uncertainty',0,arm,'mean_entropy'),entry(f,'uncertainty',0,arm,'entropy')))
        pairs.append(('dice_only_unfold_vs_plane',entry(f,'dice_ablation',0,arms[0]),entry(f,'dice_ablation',0,arms[1])))
        for seed in (1,2):
            for arm in arms:pairs.append(('replication_vs_depth',entry(f,'replication',seed,'depth'),entry(f,'replication',seed,arm)))
            pairs.append(('replication_unfold_vs_plane',entry(f,'replication',seed,arms[0]),entry(f,'replication',seed,arms[1])))
    comparisons=[]
    for group,reference,candidate in pairs:
        if reference not in records or candidate not in records:continue
        ca,cb=[{v['case_name']:v for v in json.loads((paths[k]/'validation_best.json').read_text())['cases']} for k in (reference,candidate)]
        assert set(ca)==set(cb)
        delta=np.array([cb[n]['macro_dice']-ca[n]['macro_dice'] for n in sorted(ca)])
        boots=delta[np.random.default_rng(0).integers(len(delta),size=(2000,len(delta)))].mean(1)
        comparisons.append(dict(group=group,reference=reference,candidate=candidate,
             delta_dice_pp=float(delta.mean()*100),patient_bootstrap95_pp=(np.quantile(boots,[.025,.975])*100).tolist(),
             cases_improved=int((delta>0).sum()),cases_worsened=int((delta<0).sum()),
             error_deltas={k:records[candidate]['validation'][k]-records[reference]['validation'][k] for k in ('missed','extra','swaps')}))
    replications={}
    for f,arms in p['pairs'].items():
        data={}
        for arm in ['depth',*arms]:
            keys=[f'prior/{f}/{arm}']+[entry(f,'replication',s,arm) for s in (1,2)]
            if all(k in records for k in keys):
                scores=[records[k]['validation']['macro_dice']*100 for k in keys]
                data[arm]=dict(seeds=[0,1,2],dice_pct=scores,mean=float(np.mean(scores)),sample_sd=float(np.std(scores,ddof=1)))
        replications[f]=data
    result=dict(protocol_sha256=p['identity_sha'],new_completed=sum(e['id'] in records for e in p['runs']),
         new_requested=len(p['runs']),runs={k:{a:v[a] for a in ('completed_epochs','best_epoch','validation')} for k,v in records.items()},
         comparisons=comparisons,seed_replication=replications,
         uncertainty='Case bootstraps condition on validation-selected checkpoints; seed replication shares frozen seed-0 guides and coefficients. This is development validation, not an independent test.')
    atomic_json(root/'comparison.json',result)
    return result
