"""Render the complete matrix without filling missing cells with pilot results."""
import csv
import json
import statistics
from pathlib import Path


def matrix_rows(plan, base, state):
    rows = []
    for item in plan['matrix']:
        model, arm, fold, seed = (item[k] for k in ('model','arm','fold','seed'))
        reference = base['references'][f'{model}_f{fold}_s{seed}']
        control = reference['result']['validation_dice']
        row = dict(model=model, objective=arm, fold=fold, seed=seed, baseline_dice=control,
                   final_dice=None, delta_pp=None, recipe=None, status='pending',recipe_qualification=None,
                   baseline_fp=sum(r['foreground_fp'] for r in reference['validation']['cases']),
                   baseline_fn=sum(r['foreground_fn'] for r in reference['validation']['cases']),
                   final_fp=None,final_fn=None,updates=None,optimization_flops=None,
                   calibration_flops=None,post_training_diagnostic_flops_allowance=0,
                   parameter_gradient_diagnostic_flops_allowance=0,weights_json=None,
                   coefficient_policy='fixed_training_calibration',online_sdf_target=None,
                   online_sdf_probe_flops_in_optimization=0,band_side_policy='fixed',
                   band_side_extra_scalar_flops_per_update=0)
        if arm == 'dice':
            row.update(final_dice=control, delta_pp=0., recipe='native_control', status='verified')
            row['recipe_qualification']='native_control'
            row['coefficient_policy']='not_applicable'
            row.update(final_fp=row['baseline_fp'],final_fn=row['baseline_fn'],
                       updates=reference['result']['updates'],
                       optimization_flops=reference['result']['optimization_flops'],
                       calibration_flops=0,weights_json='{}')
        elif item['status'] == 'verified_existing_result':
            result = item['result']
            row.update(final_dice=result['band_dice'],delta_pp=result['delta_dice_pp'],
                       recipe=result['recipe'],status='verified')
            row['recipe_qualification']='all_nine_passed'
            row.update(final_fp=result['band_foreground_fp'],final_fn=result['band_foreground_fn'],
                       updates=result['updates'],optimization_flops=result['optimization_flops'],
                       calibration_flops=result['calibration_flops'],weights_json=json.dumps({'bands':result['weight']}))
        else:
            key = model+'__'+arm
            selected = state.get('selected',{}).get(key)
            if selected:
                row['recipe_qualification']=selected['qualification']
                name = selected.get('recipe') or f'{key}__r{selected["candidate"]:02d}'
                attempt = state['attempts'][name]
                if attempt.get('full_matrix_complete'):
                    matches = [r for r in attempt['results'] if r['fold']==fold and r['seed']==seed]
                    if len(matches) != 1:
                        raise ValueError('Completed matrix has a missing/duplicate identity')
                    result = matches[0]
                    row.update(final_dice=result['constraint_dice'],delta_pp=result['delta_dice_pp'],
                               recipe=name,status='verified')
                    row.update(final_fp=result['constraint_foreground_fp'],final_fn=result['constraint_foreground_fn'],
                               updates=result['updates'],optimization_flops=result['optimization_flops'],
                               calibration_flops=result['calibration_flops'],weights_json=json.dumps(result['weights'],sort_keys=True))
                    row['post_training_diagnostic_flops_allowance']=result.get('post_training_diagnostic_flops_allowance',0)
                    row['parameter_gradient_diagnostic_flops_allowance']=result.get('parameter_gradient_diagnostic_flops_allowance',0)
                    row['band_side_policy']=result.get('band_side_policy','fixed')
                    row['band_side_extra_scalar_flops_per_update']=result.get('band_side_extra_scalar_flops_per_update',0)
                    if result.get('online_sdf_strength'):
                        online=result['online_sdf_strength']
                        row.update(coefficient_policy='online_training_SDF; weights_json_is_anchor_only',
                            online_sdf_target=online['target'],
                            online_sdf_probe_flops_in_optimization=online['probe_optimization_flops_allowance'])
                else:
                    row.update(recipe=name,status='selected_recipe_incomplete')
        rows.append(row)
    if len(rows) != 504:
        raise ValueError('Matrix does not have all 504 declared cells')
    return rows


def write_report(root):
    read = lambda path: json.loads(path.read_text())
    state_path = root/'constraint_search/search_status.json'
    state = read(state_path) if state_path.exists() else {}
    rows = matrix_rows(read(root/'constraint_matrix_plan.json'),read(root/'bands_v1_payload.json'),state)
    directory = root/'constraint_search';directory.mkdir(exist_ok=True)
    path = directory/'full_constraint_matrix.csv'
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    groups=[]
    for model,arm in dict.fromkeys((r['model'],r['objective']) for r in rows):
        group=[r for r in rows if r['model']==model and r['objective']==arm]
        completed=[r for r in group if r['status']=='verified']
        full=len(completed)==9
        deltas=[r['delta_pp'] for r in completed]
        groups.append(dict(model=model,objective=arm,completed=len(completed),
            mean_delta_pp=sum(deltas)/9 if full else None,minimum_delta_pp=min(deltas) if full else None,
            sd_delta_pp=statistics.stdev(deltas) if full else None,
            mean_final_dice=statistics.mean(r['final_dice'] for r in completed) if full else None,
            sd_final_dice=statistics.stdev(r['final_dice'] for r in completed) if full else None,
            positive_pairs=sum(r['delta_pp']>0 for r in completed) if full else None,
            meaningful_pairs=sum(r['delta_pp']>=1. and r['final_dice']>=.5 for r in completed) if full else None,
            meaningful_all_nine=full and arm!='dice' and min(deltas)>=1.
                and min(r['final_dice'] for r in completed)>=.5,
            mean_target_met=full and arm!='dice' and sum(deltas)/9>=1.
                and min(r['final_dice'] for r in completed)>=.5,
            all_nine_deltas_pp=[dict(fold=r['fold'],seed=r['seed'],delta_pp=r['delta_pp']) for r in completed],
            worst_pair=min((dict(fold=r['fold'],seed=r['seed'],delta_pp=r['delta_pp']) for r in completed),
                           key=lambda r:r['delta_pp']) if completed else None))
    summary=dict(conditions=504,verified=sum(r['status']=='verified' for r in rows),
        missing=sum(r['status']!='verified' for r in rows),
        status='complete' if all(r['status']=='verified' for r in rows) else 'incomplete',
        interpretation='Exploratory validation-selected results; rejected screening trials are separate.',groups=groups)
    summary_path=directory/'matrix_summary.json';summary_path.write_text(json.dumps(summary,indent=2)+'\n')
    return summary,path,summary_path
