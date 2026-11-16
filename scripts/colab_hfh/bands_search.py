"""Sequential, fully logged HFH BAND recipe search against frozen controls."""
import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def qualifies(rows):
    """Require all nine cells >=50%, a positive overall and every-fold mean."""
    if len(rows) != 9 or {(r['fold'], r['seed']) for r in rows} != {(f,s) for f in range(3) for s in range(3)}:
        return False
    return (all(r['band_dice'] >= .5 for r in rows)
            and sum(r['delta_dice_pp'] for r in rows) > 0
            and all(sum(r['delta_dice_pp'] for r in rows if r['fold'] == f) > 0 for f in range(3)))


def candidate_payload(base, model, target, prefix=None, ramp=None):
    payload = deepcopy(base)
    payload['rates'] = {model: base['rates'][model]}
    payload['protocols'] = {model: payload['protocols'][model]}
    payload['references'] = {k:v for k,v in payload['references'].items() if k.startswith(model+'_f')}
    # Fixed, declared stress-first order; recipe selection is exploratory.
    payload['folds'] = [2,0,1]
    payload['seeds'] = [2,0,1]
    protocol = payload['protocols'][model]['protocol.json']
    schedule = protocol['models'][protocol['trainer_alias']]['objectives']['dice_bands']
    if prefix is not None:
        schedule['prefix_updates'] = schedule['sdf_prefix_updates'] = prefix
    if ramp is not None:
        schedule['constraint_warmup_updates'] = schedule['sdf_warmup_updates'] = ramp
        protocol['constraint_warmup_updates'] = ramp
    n, start = schedule['max_steps'], schedule['prefix_updates']
    cost = start*schedule['baseline_flops_per_update']+(n-start)*schedule['flops_per_update']
    assert cost <= protocol['budget_flops'] and start < n
    schedule.update(consumed_training_flops=cost, consumed_flops=cost,
                    consumed_total_flops=cost+schedule['calibration_flops'],
                    unused_flops=protocol['budget_flops']-cost)
    protocol['calibration_target_ratio'] = target
    protocol['name'] = f'HFH_BAND_target{target}_prefix{start}_ramp{schedule["constraint_warmup_updates"]}'
    protocol['endpoint'] = 'Final-budget comparison; validation-selected recipe, all attempted outcomes retained'
    payload['version'] = protocol['name']
    return payload


def recipes(model, base):
    protocol = base['protocols'][model]['protocol.json']
    schedule = protocol['models'][protocol['trainer_alias']]['objectives']['dice_bands']
    n = schedule['max_steps']
    strengths = [.1,.05,.2,.025,.01] if model == 'nnunet' else [.025,.01,.05,.2,.005]
    result = [(t,None,None) for t in strengths]
    for fraction in (.25,.5,.75):
        for target in (.05,.1,.2):
            result.append((target,int(n*fraction),max(5,int(n*.1))))
    return result


def run(root):
    base = read(root/'bands_v1_payload.json')
    work = root/'bands_recipe_search';work.mkdir(exist_ok=True)
    status_path = work/'search_status.json'
    had_local_state=status_path.exists()
    state = read(status_path) if had_local_state else dict(status='waiting_for_v1',attempts={},winners={})
    write(status_path,state)
    # Do not overlap the already-running original campaign.
    with (root/'bands_campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
    original_path=root/'bands_v1/campaign_status.json'
    if not original_path.exists() or read(original_path)['status']!='complete_all_36':
        subprocess.run([sys.executable,str(root/'bands_campaign_search.py'),'--root',str(root)],check=True)
    original = read(original_path)
    if original['status'] != 'complete_all_36':
        raise RuntimeError('Finish/audit original 36-run campaign before recipe search')
    from bands_campaign_search import Drive
    drive = Drive('results_band_recipe_search')
    if not had_local_state:
        entries=json.loads(drive.call('lsjson',drive.folder,'--files-only','--hash'))
        matches=[entry for entry in entries if entry['Name']=='search_status.json']
        if len(matches)>1:raise ValueError('Ambiguous archived recipe-search state')
        if matches:
            recovered=work/'restored_search_status.json'
            drive.call('copyto',drive.path('search_status.json'),str(recovered),'--checksum')
            expected=matches[0]['Hashes'].get('MD5',matches[0]['Hashes'].get('md5'))
            if recovered.stat().st_size!=matches[0]['Size'] or hashlib.md5(recovered.read_bytes()).hexdigest()!=expected:
                raise ValueError('Archived recipe-search state checksum mismatch')
            state=read(recovered)
            for model,winner in state['winners'].items():
                if not qualifies(winner['results']):raise ValueError('Archived winner fails qualification')
                for row in winner['results']:
                    key=f'{model}_f{row["fold"]}_s{row["seed"]}'
                    if row['baseline_archive_sha256']!=base['references'][key]['archive_sha256']:
                        raise ValueError('Archived winner uses different controls')
            write(status_path,state)
    drive.upload(Path(__file__))
    def persist():
        write(status_path,state)
        drive.upload(status_path)
    for model in base['rates']:
        rows = [r for r in original['results'] if r['model']==model]
        if qualifies(rows):
            state['winners'][model] = dict(recipe='bands_v1',results=rows)
    state['status']='searching';persist()
    for model in sorted(base['rates'],key=lambda m:(m!='basic_unet16',m!='nnunet',m)):
        if model in state['winners']:continue
        candidates=list(enumerate(recipes(model,base)))
        priority=[0,9,6,8,12,1,2,3,4,5,7,10,11,13]
        candidates.sort(key=lambda item:priority.index(item[0]))
        for index,(target,prefix,ramp) in candidates:
            if (work/'STOP').exists():
                state['status']='paused';persist();return
            name=f'{model}_r{index:02d}'
            payload=candidate_payload(base,model,target,prefix,ramp)
            payload_path=work/(name+'_payload.json')
            trial=work/name
            trial_status=trial/'campaign_status.json'
            pilots=[f'{model}_f2_s{s}' for s in (2,0,1)]
            record=state['attempts'].setdefault(name,dict(model=model,target=target,prefix=prefix,ramp=ramp,status='pilot'))
            if record['status'] in ('rejected_pilot','rejected_full'):continue
            payload['pairs']=None if record['status']=='full' else pilots
            payload['reject_nonpositive_fold']=record['status']=='full'
            write(payload_path,payload)
            state['current']=name;persist()
            while True:
                print('SEARCH START',name,'phase',record['status'],'target',target,'prefix',prefix,'ramp',ramp,flush=True)
                command=[sys.executable,str(root/'bands_campaign_search.py'),'--root',str(root),
                         '--payload',str(payload_path.relative_to(root)),'--work',str(trial.relative_to(root)),
                         '--folder','results_band_search_'+name,'--hours','6']
                subprocess.run(command,check=True)
                result=read(trial_status)
                if result['status'] not in ('complete_requested_matrix','complete_all_36','rejected_completed_fold'):
                    raise RuntimeError('Candidate did not complete: '+name)
                rows=result['results'];record['results']=rows
                record['all_completed_pairs']=[read(path) for path in sorted((trial/'archives').glob('*/paired_comparison.json'))]
                mean=sum(r['delta_dice_pp'] for r in rows)/len(rows)
                print('SEARCH RESULT',name,len(rows),'mean delta',mean,flush=True)
                if record['status']=='pilot':
                    if mean>0 and all(r['band_dice']>=.5 for r in rows):
                        record['status']='full';payload['pairs']=None;payload['reject_nonpositive_fold']=True;write(payload_path,payload);persist();continue
                    record['status']='rejected_pilot'
                else:
                    record['status']='winner' if qualifies(rows) else 'rejected_full'
                    if record['status']=='winner':state['winners'][model]=dict(recipe=name,results=rows)
                persist();break
            if model in state['winners']:break
    state['status']='all_models_improved' if len(state['winners'])==4 else 'needs_next_hypothesis'
    state.pop('current',None);persist()
    print('SEARCH STATUS',state['status'],'winners',list(state['winners']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    with (args.root/'bands_recipe_search.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args.root)
