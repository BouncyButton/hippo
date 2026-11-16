"""Separate 27-run lower-compute comparison: native, BAND, BAND+depth."""
import argparse
from copy import deepcopy
import csv
import fcntl
import hashlib
import re
from pathlib import Path
import subprocess
import sys
import time

from scripts.colab_hfh.bands_campaign import read,write
from scripts.colab_hfh.constraint_drive import Drive
from scripts.colab_hfh.constraint_schedule import make_payload
from scripts.colab_hfh.lower_budget_protocol import VERSION,native_payload,references_from_native


def run(args):
    root=args.root
    base=read(root/'bands_v1_payload.json');plan=read(root/'lower_budget_plan.json')
    version=plan['version']
    if not re.fullmatch(r'nnunet_budget[0-9]+(?:_scaled)?_v[0-9]+',version):
        raise ValueError('Invalid lower-budget study identifier')
    work=root/version;work.mkdir(exist_ok=True)
    for name,expected in plan['source_sha256'].items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Lower-budget source differs: '+name)
    prior=work/'plan.json'
    if prior.exists() and read(prior)!=plan:raise ValueError('Frozen lower-budget plan changed')
    write(prior,plan)
    drive=Drive('results_'+version);drive.upload(prior)
    deadline=time.time()+args.hours*3600;state=dict(status='running',version=version,
        scope=['nnunet'],objectives=['dice','dice_bands','dice_bands_depth'],
        folds=[0,1,2],seeds=[0,1,2],expected_runs=27,completed_phases=[])
    status=work/'search_status.json'
    def save():write(status,state);drive.upload(status)
    def campaign(name,module,payload):
        payload_path=work/(name+'_payload.json')
        if payload_path.exists() and read(payload_path)!=payload:raise ValueError('Frozen lower-budget payload changed')
        write(payload_path,payload)
        if (work/'STOP').exists() or time.time()>=deadline:
            state['status']='paused';save();return False
        state['current_phase']=name;save()
        subprocess.run([sys.executable,'-m',module,'--root',str(root),'--payload',str(payload_path),
            '--work',str(work/name),'--folder','results_'+version+'_'+name,
            '--hours',str(max(.01,(deadline-time.time())/3600))],check=True)
        outcome=read(work/name/'campaign_status.json')
        if outcome['status']!='complete_requested_matrix':
            state['status']='paused';save();return False
        if len(outcome['results'])!=9 or {(r['fold'],r['seed']) for r in outcome['results']}!={(f,s) for f in range(3) for s in range(3)}:
            raise ValueError('Lower-budget phase requires all nine identities')
        state['completed_phases'].append(name);save();return True
    try:
        native=native_payload(base,plan)
        if not campaign('native','scripts.colab_hfh.lower_budget_baseline_campaign',native):return
        references=references_from_native(work/'native',native)
        low=deepcopy(native);low['references']=references
        write(work/'lower_budget_references.json',references);drive.upload(work/'lower_budget_references.json')
        for arm in ('dice_bands','dice_bands_depth'):
            spec=plan['recipes'][arm]['specification']
            payload=make_payload(low,'nnunet',arm,plan['canonical_costs'],**spec,screening=False)
            payload['lower_budget_parent_recipe']=plan['recipes'][arm]
            if not campaign(arm,'scripts.colab_hfh.constraint_campaign',payload):return
        rows=[]
        for arm in ('dice_bands','dice_bands_depth'):
            for row in read(work/arm/'campaign_status.json')['results']:
                key=f"nnunet_f{row['fold']}_s{row['seed']}"
                full=base['references'][key]['result']['validation_dice']
                parent=next(r for r in plan['recipes'][arm]['full_budget_results']
                            if (r['fold'],r['seed'])==(row['fold'],row['seed']))
                rows.append(dict(objective=arm,fold=row['fold'],seed=row['seed'],
                    full_budget_native_dice=full,lower_budget_native_dice=row['baseline_dice'],
                    full_budget_constraint_dice=parent['constraint_dice'],
                    lower_budget_constraint_dice=row['constraint_dice'],
                    constraint_gain_over_lower_native_pp=row['delta_dice_pp'],
                    full_budget_constraint_gain_pp=parent['delta_dice_pp'],
                    change_in_constraint_gain_pp=row['delta_dice_pp']-parent['delta_dice_pp'],
                    constraint_absolute_change_vs_full_pp=100*(row['constraint_dice']-parent['constraint_dice']),
                    constraint_gain_over_full_native_pp=100*(row['constraint_dice']-full),
                    native_change_vs_full_pp=100*(row['baseline_dice']-full),
                    optimization_flops=row['optimization_flops'],calibration_flops=row['calibration_flops']))
        table=work/'paired_budget_comparison.csv'
        with table.open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        summary={arm:dict(pairs=9,mean_gain_pp=sum(r['constraint_gain_over_lower_native_pp'] for r in rows if r['objective']==arm)/9,
            worst_gain_pp=min(r['constraint_gain_over_lower_native_pp'] for r in rows if r['objective']==arm),
            mean_absolute_dice=sum(r['lower_budget_constraint_dice'] for r in rows if r['objective']==arm)/9)
            for arm in ('dice_bands','dice_bands_depth')}
        write(work/'comparison_summary.json',dict(groups=summary,rows=rows,
            interpretation=plan['matching']+' Gains versus a weaker control alone do not establish better absolute segmentation or full-budget improvement. Exploratory validation; all outcomes retained.'))
        drive.upload(table);drive.upload(work/'comparison_summary.json')
        state.update(status='complete_27_runs',summary=summary);state.pop('current_phase',None);save()
        print('LOWER-BUDGET COMPARISON COMPLETE',summary,flush=True)
    except Exception as error:
        state.update(status='runtime_error',error=str(error));save();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--hours',type=float,default=8);args=parser.parse_args()
    with (args.root/'bands_recipe_search.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(args)
