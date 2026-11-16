"""Require >=1 Dice point on every fixed seed/fold pair for each target model."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'bundle/source'))
from scripts.colab_hfh.bands_search import candidate_payload,read,write

MODELS=('nnunet','swinunetr')
MIN_DELTA=1.0


def qualifies(rows):
    return (len(rows)==9 and not any(r.get('control_kind') for r in rows)
            and {(r['fold'],r['seed']) for r in rows}=={(f,s) for f in range(3) for s in range(3)}
            and all(r['delta_dice_pp']>=MIN_DELTA and r['band_dice']>=.5 for r in rows))


def specifications(model):
    # Fixed candidate order, common recipe across all nine pairs.
    if model=='swinunetr':
        initial=[(.5,.3,50,50,.75),(.75,.3,50,50,.75),(.75,.6,50,50,1.0)]
    else:
        initial=[(.75,.3,45,9,.75),(.75,.3,22,9,.75),(.75,.6,45,9,1.0)]
    n=90 if model=='nnunet' else 206
    for inner in (.9,.65,.75,.5):
        for target in (.3,.6,1.0):
            for fraction in (.25,.5,.75):
                item=(inner,target,int(n*fraction),max(5,int(n*.1)),max(.75,1.5*target))
                if item not in initial:initial.append(item)
    if model=='swinunetr':
        # Append, never renumber existing attempts. Balanced target .3 nearly
        # passed f2s1; weighting the inner side .75 weakened f2s0.
        initial.extend([(.5,.6,50,50,1.0),(.5,1.0,50,50,1.5),(.5,.4,50,50,.75)])
    prefix,ramp=(22,9) if model=='nnunet' else (50,50)
    for gamma in (2,1):
        for inner in (.5,.75,.9):
            for target in (.3,.6):
                initial.append((inner,target,prefix,ramp,1.0,gamma))
    for inner in (.5,.9,.75):
        for target in (.3,.6,.1):
            initial.append((inner,target,prefix,ramp,1.0,0,1))
    if model=='nnunet':
        initial.extend([(.9,.3,22,9,.75,0,2,True),(.9,.3,22,9,.75,0,1,True),
                        (.75,.3,22,9,.75,0,2,True),(.5,.3,22,9,.75,0,2,True),
                        (.9,.3,45,9,.75,0,2,True),(.9,.6,45,9,1.,0,2,True)])
        initial.extend([(.9,.4,22,9,.75,0,1,True),(.9,.6,22,9,1.,0,1,True),
                        (.9,.2,22,9,.75,0,1,True)])
        initial.extend([(.9,.3,22,9,.75,0,2,False,18),(.9,.6,22,9,1.,0,2,False,18),
                        (.9,.3,22,9,.75,0,2,False,27),(.9,.3,22,9,.75,0,1,True,18)])
        initial.extend([(.9,t,0,45,1.5) for t in (.3,.6,1.)])
        initial.extend([(.5,t,0,45,1.5) for t in (.3,.6)])
        # s66 cleared all fold-2 seeds but regressed on fold0. Test whether
        # releasing that exact intervention preserves its early benefit.
        initial.extend([(.9,.2,22,9,.75,0,1,True,tail) for tail in (18,27,9)])
        initial.extend([(.9,.2,22,9,.75,0,1,True,0,.5,45),
                        (.9,.2,22,9,.75,0,1,True,0,.75,45),
                        (.9,.2,22,9,.75,0,1,True,0,.5,67),
                        (.9,.3,22,9,.75,0,2,False,0,.5,45)])
        # Zero auxiliary weight isolates the SGD reset on two diagnostic pairs;
        # it can never be selected as a BAND winner. Mild nonzero arms follow.
        initial.append((.5,0.,22,9,.75,0,2,True))
        initial.extend([(.5,t,22,9,.75,0,2,True) for t in (.05,.01,.1)])
    else:
        initial.extend([(.5,.3,50,50,.75,0,2,False,41),(.5,.6,50,50,1.,0,2,False,41),
                        (.5,1.,50,50,1.5,0,2,False,82),(.5,.3,103,20,.75,0,2,False,41)])
        initial.extend([(.5,t,0,50,1.5) for t in (.3,.6,1.)])
        initial.append((.5,.6,0,103,1.5))
        # Early target .3 passed fold2 and reached +.984pp on fold0s1;
        # doubling it weakened that pair. Refine strength and onset ramp.
        initial.extend([(.5,t,0,50,1.5) for t in (.4,.35,.25)])
        initial.extend([(.5,.3,0,ramp,1.5) for ramp in (25,75,103)])
    return initial


def candidate_order(model):
    priority=([0,1,39,40,41,42,43,54,55,31,34,37,63,64,67,68,72,73,71,74,75,76,69,65,30,33,36,56,48,12,15,2] if model=='swinunetr'
              else [0,1,2,3,12,37,38,49,52,58,59,64,66,65,76,77,78,71,79,83,84,85,86,60,61,80,81,82,74,41,72,67,68,69,73,70,62,63,50,53,39,40,42,16,7,15,4,13])
    return priority+[i for i in range(len(specifications(model))) if i not in priority]


def make_payload(base,model,spec):
    inner,target,prefix,ramp,cap,*extra=spec
    gamma=extra[0] if extra else 0
    steps=extra[1] if len(extra)>1 else 2
    reset=extra[2] if len(extra)>2 else False
    release=extra[3] if len(extra)>3 else 0
    final_inner=extra[4] if len(extra)>4 else None
    side_start=extra[5] if len(extra)>5 else None
    payload=candidate_payload(base,model,target,prefix,ramp)
    payload.update(strict_search=True,min_pair_delta_pp=MIN_DELTA,folds=[2,0,1],seeds=[0,1,2])
    if model=='swinunetr':
        payload['seeds']=[1,0,2]
        payload['pair_order_rationale']='Seed 1 on fold 2 repeatedly rejected otherwise promising candidates; test it first, retain all nine required pairs.'
        if prefix == 0:
            payload['folds']=[0,2,1]
            payload['pair_order_rationale']='Early activation passed all fold2 seeds but failed fold0seed1 at targets .3 and .6. Screen that pair first; all nine remain required.'
    elif reset and steps==1:
        payload['seeds']=[2,0,1]
        payload['pair_order_rationale']='Narrow/reset candidate s59 passed seeds0/1 but missed seed2; test seed2 first, retain all nine required pairs.'
    if final_inner is not None:
        payload['folds']=[0,2,1];payload['seeds']=[2,0,1]
        payload['pair_order_rationale']='Rebalance the s66 foreground correction to address its fold0seed2 excess false positives; screen that failed pair first, retain all nine.'
    p=payload['protocols'][model]['protocol.json'];p.update(band_inner_weight=inner,calibration_p95_cap=cap)
    if gamma:p['band_focal_gamma']=gamma
    if steps!=2:p['band_steps']=steps
    if reset:
        if model!='nnunet':raise ValueError('SGD reset is only defined for nnU-Net')
        p['band_reset_momentum']=True
    if release:p['band_release_updates']=release
    if final_inner is not None:
        maximum=p['models'][p['trainer_alias']]['objectives']['dice_bands']['max_steps']
        if not prefix <= side_start < maximum-1:raise ValueError('Side transition must follow calibration and fit the run')
        p.update(band_final_inner_weight=final_inner,band_side_transition_start=side_start)
    p['name']+=f'_inner{inner}_cap{cap}_strict1pp'
    if gamma:p['name']+=f'_focal{gamma}'
    if steps!=2:p['name']+=f'_steps{steps}'
    if reset:p['name']+='_reset_sgd_momentum'
    if release:p['name']+=f'_release{release}'
    if final_inner is not None:p['name']+=f'_finalinner{final_inner}_from{side_start}'
    p['endpoint']='Final budget; reject any individual pair below +1 Dice point; validation-selected exploratory recipe'
    if target == 0:
        if not reset or model!='nnunet':raise ValueError('Zero-strength arm is only an SGD diagnostic')
        payload.pop('min_pair_delta_pp')
        payload.update(diagnostic_control=True,pairs=['nnunet_f2_s0','nnunet_f0_s2'],folds=[2,0,1],seeds=[0,1,2])
        p['control_kind']='sgd_reset_without_auxiliary_gradient'
        p['endpoint']='Two-pair diagnostic control, not eligible as a BAND winner; complete both outcomes regardless of direction'
    p['band_definition']=f'Two-voxel-step inner/outer BCE; inner weight {inner}, outer weight {1-inner}; separately labelled side-weighted BAND ablation'
    if gamma:p['band_definition']+=f'; focal multiplier (1-p_t)^{gamma} with full gradient, separately labelled focal BAND'
    if steps!=2:
        p['band_definition']=p['band_definition'].replace('Two-voxel-step', 'One-voxel-step')
        p['band_width_flops_note']='One-step morphology conservatively charged at the original two-step cost; no extra updates credited.'
    if reset:p['band_definition']+='; coupled training-transition ablation: reset SGD momentum at first BAND update, retain weights/LR/data order; original steady-state SGD cost conservatively charged'
    if release:p['band_definition']+=f'; linearly taper BAND scale to zero over final {release} updates, native loss unchanged'
    if final_inner is not None:p['band_definition']+=f'; linearly shift inner weight from {inner} to {final_inner} between update-index {side_start} and the final update; outer weight remains its complement; calibration uses initial side weight'
    if target == 0:p['band_definition']+='; ZERO auxiliary coefficient isolates the SGD transition; dummy auxiliary/calibration computation is conservatively charged'
    s=p['models'][p['trainer_alias']]['objectives']['dice_bands']
    if inner!=.5:
        # Conservative scalar-arithmetic allowance for the additional weighted
        # sum and its backward, per batch-one call. Spatial reductions unchanged.
        s['early_flops_per_update']+=16;s['flops_per_update']+=16;s['calibration_flops']+=160
        p['variant_extra_scalar_flops_allowance']=16
    if gamma:
        # Binary foreground grid 64^3, batch one. <=32 extra arithmetic ops
        # per voxel covers sigmoid, p_t, gamma 1/2 modulation and backward.
        # Native zero-gamma forward remains unchanged. Calibration has ten crops.
        focal_cost=32*64**3
        s['early_flops_per_update']+=focal_cost;s['flops_per_update']+=focal_cost
        s['calibration_flops']+=10*focal_cost
        p['focal_extra_flops_per_update_allowance']=focal_cost
    if release:
        s['early_flops_per_update']+=16;s['flops_per_update']+=16
        p['release_host_flops_per_active_update_allowance']=16
    if final_inner is not None:
        s['early_flops_per_update']+=32;s['flops_per_update']+=32
        p['side_transition_extra_flops_per_active_update_allowance']=32
    cost=prefix*s['baseline_flops_per_update']+(s['max_steps']-prefix)*s['flops_per_update']
    assert cost<=p['budget_flops']
    s.update(consumed_training_flops=cost,consumed_flops=cost,consumed_total_flops=cost+s['calibration_flops'],unused_flops=p['budget_flops']-cost)
    return payload


def run(root):
    from bands_campaign_strict import Drive
    base=read(root/'bands_v1_payload.json');work=root/'bands_strict_search';work.mkdir(exist_ok=True)
    path=work/'search_status.json'
    drive=Drive('results_band_strict_search')
    if not path.exists():
        files=json.loads(drive.call('lsjson',drive.folder,'--files-only','--hash'))
        found=[f for f in files if f['Name']=='search_status.json']
        if found:
            if len(found)!=1:raise ValueError('Ambiguous saved search')
            drive.call('copyto',drive.path('search_status.json'),str(path),'--checksum')
            if hashlib.md5(path.read_bytes()).hexdigest()!=found[0]['Hashes'].get('MD5',found[0]['Hashes'].get('md5')):raise ValueError('Search checksum mismatch')
    state=read(path) if path.exists() else dict(status='searching',criterion='Every individual pair >= +1.0 Dice pp, all nine scores >=50%',minimum_delta_pp=MIN_DELTA,attempts={},winners={})
    if state['minimum_delta_pp']!=MIN_DELTA:raise ValueError('Frozen acceptance threshold changed')
    for model,winner in state['winners'].items():
        if not qualifies(winner['results']):raise ValueError('Saved winner does not meet strict criterion')
        for r in winner['results']:
            if r['baseline_archive_sha256']!=base['references'][f'{model}_f{r["fold"]}_s{r["seed"]}']['archive_sha256']:raise ValueError('Saved controls differ')
    drive.upload(Path(__file__))
    def persist():write(path,state);drive.upload(path)
    state['status']='searching';persist()
    # Interleave models so neither waits behind the other's full candidate bank.
    plans={m:specifications(m) for m in MODELS}
    orders={m:candidate_order(m) for m in MODELS}
    state['search_order']=orders
    state['priority_rationale']='Early Swin target .3 reached +.984pp on fold0 after passing fold2; .6 weakened fold0. Refine around .3 and ramp length. nnU-Net terminal release weakened stress-pair gains; prioritize early activation next. All previous definitions and results are retained.'
    persist()
    for rank in range(max(map(len,orders.values()))):
        for model in MODELS:
            if model in state['winners'] or rank>=len(orders[model]):continue
            index=orders[model][rank]
            if (work/'STOP').exists():state['status']='paused';persist();return
            spec=plans[model][index];name=f'{model}_s{index:02d}'
            record=state['attempts'].setdefault(name,dict(model=model,specification=spec,status='pending'))
            if list(record['specification'])!=list(spec):raise ValueError('Frozen candidate definition changed')
            if record['status'] in ('rejected','diagnostic_complete'):continue
            payload_path=work/(name+'_payload.json');write(payload_path,make_payload(base,model,spec))
            state['current']=name;record['status']='running';persist()
            print('STRICT START',name,'inner,target,prefix,ramp,cap[,focal_gamma,steps]=',spec,flush=True)
            trial=work/name
            try:
                subprocess.run([sys.executable,str(root/'bands_campaign_strict.py'),'--root',str(root),'--payload',str(payload_path.relative_to(root)),
                    '--work',str(trial.relative_to(root)),'--folder','results_band_strict_'+name,'--hours','6'],check=True)
            except (subprocess.CalledProcessError, OSError) as error:
                record.update(status='runtime_error',error=str(error))
                state.update(status='runtime_error',current=name)
                persist()
                raise
            result=read(trial/'campaign_status.json')
            if result['status'] not in ('rejected_pair','complete_requested_matrix'):raise RuntimeError('Strict trial interrupted: '+name)
            rows=result['results']
            diagnostic=spec[1]==0
            record.update(results=rows,status='diagnostic_complete' if diagnostic else ('winner' if qualifies(rows) else 'rejected'))
            if record['status']=='winner':state['winners'][model]=dict(recipe=name,specification=spec,results=rows)
            print('STRICT RESULT',name,record['status'],'pairs',len(rows),'minimum delta',min(r['delta_dice_pp'] for r in rows),flush=True)
            persist()
    state['status']='both_models_strictly_improved' if len(state['winners'])==2 else 'needs_next_hypothesis'
    state.pop('current',None);persist();print('STRICT STATUS',state['status'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    with (args.root/'bands_recipe_search.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args.root)
