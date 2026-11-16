"""Freeze the best independently audited complete BAND recipes for budget study."""
import hashlib
import argparse
from copy import deepcopy
import json
from pathlib import Path

from scripts.colab_hfh.lower_budget_protocol import VERSION,native_payload
from scripts.colab_hfh.constraint_schedule import make_payload


def rescale_schedule(specification, numerator, denominator):
    """Predeclare timing proportional to budget; preserve all loss strengths."""
    scaled=deepcopy(specification)
    scaled['prefix']=specification['prefix']*numerator//denominator
    scaled['ramp']=max(1,(specification['ramp']*numerator+denominator-1)//denominator)
    return scaled


def build(ratio=(3,4),scale_schedule=False):
    root=Path(__file__).resolve().parents[2];export=root/'reports/hfh_colab_exploration_20261111'
    read=lambda path:json.loads(path.read_text())
    base=read(export/'bands_v1_payload.json');matrix=read(export/'constraint_matrix_plan.json')
    numerator,denominator=ratio
    if not 0<numerator<denominator or 100*numerator%denominator:
        raise ValueError('Use an integer percentage strictly between0and100')
    version=f'nnunet_budget{100*numerator//denominator}'+('_scaled' if scale_schedule else '')+'_v1'
    study=dict(version=version,ratio=list(ratio))
    low=native_payload(base,study);recipes={}
    for arm in ('dice_bands','dice_bands_depth'):
        candidates=[]
        for path in (export/'constraints_analysis').glob('nnunet__'+arm+'__r*/candidate_audit.json'):
            audit=read(path)
            if audit['summary']['complete'] and len(audit['results'])==9:
                candidates.append((audit['summary']['mean_delta_pp'],path,audit))
        _,path,audit=max(candidates,key=lambda item:item[0])
        from scripts.colab_hfh.constraint_search import recipes as declared_recipes
        index=int(path.parent.name.rsplit('__r',1)[1])
        specification=declared_recipes('nnunet',matrix,arm)[index]
        full=make_payload(base,'nnunet',arm,matrix['cost_references']['nnunet']['canonical_costs'],
                          **specification,screening=False)['protocols']['nnunet']['protocol.json']
        if hashlib.sha256((json.dumps(full,indent=2)+'\n').encode()).hexdigest()!=audit['protocol_sha256']:
            raise ValueError('Selected recipe differs from the independently audited full-budget protocol')
        parent_specification=deepcopy(specification)
        if scale_schedule:
            specification=rescale_schedule(specification,numerator,denominator)
        payload=make_payload(low,'nnunet',arm,matrix['cost_references']['nnunet']['canonical_costs'],
                             **specification,screening=False)
        p=payload['protocols']['nnunet']['protocol.json']
        schedule=p['models']['nnunet']['objectives'][arm]
        if schedule['prefix_updates']+schedule['constraint_warmup_updates']>=schedule['max_steps']:
            raise ValueError('Budget leaves no fully active constrained updates; scale onset/ramp explicitly')
        recipes[arm]=dict(parent_recipe=path.parent.name,specification=specification,
            parent_specification=parent_specification,
            full_budget_summary=audit['summary'],parent_audit_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            full_budget_results=audit['results'],
            lower_budget_schedule=p['models']['nnunet']['objectives'][arm])
    sources=matrix['calibration_implementation']['source_sha256'].copy()
    for name in ('lower_budget_protocol.py','lower_budget_baseline_campaign.py','lower_budget_search.py'):
        sources[name]=hashlib.sha256((root/'scripts/colab_hfh'/name).read_bytes()).hexdigest()
    plan=dict(version=version,model='nnunet',folds=[0,1,2],seeds=[0,1,2],expected_runs=27,
        budget_flops=low['protocols']['nnunet']['protocol.json']['budget_flops'],
        native_updates=low['protocols']['nnunet']['protocol.json']['models']['nnunet']['objectives']['dice']['max_steps'],
        lr_horizon_updates=90,ratio=list(ratio),recipes=recipes,
        scale_constraint_schedule=scale_schedule,
        canonical_costs=matrix['cost_references']['nnunet']['canonical_costs'],source_sha256=sources,
        hypothesis='Lower compute may change the native/constraint advantage; retain absolute Dice and all paired outcomes.',
        matching=('Same full-budget LR trajectory, initialization, splits, native loss, optimizer and loss targets. '
            + ('Constraint prefix=floor(parent prefix*budget fraction); ramp=ceil(parent ramp*budget fraction). '
               'This jointly tests reduced compute and earlier constraint timing, not truncation alone. '
               if scale_schedule else 'Constraint onset/ramp unchanged; budget truncation only. ')
            + 'Training-only recalibration; all nine paired outcomes and absolute scores retained.'))
    target=export/'lower_budget_plan.json';target.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(path=str(target),budget=plan['budget_flops'],updates=plan['native_updates'],
        recipes={arm:r['parent_recipe'] for arm,r in recipes.items()})))
    return plan


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--fraction',nargs=2,type=int,default=[3,4])
    parser.add_argument('--scale-schedule',action='store_true')
    args=parser.parse_args();build(args.fraction,args.scale_schedule)
