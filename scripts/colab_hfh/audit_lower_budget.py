"""Independent local audit of the 27-run budget study; no partial-set means."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import zipfile

from scripts.colab_hfh.lower_budget_protocol import native_payload,MATCHED_FIELDS
from scripts.colab_hfh.summarize_constraints import verify
from scripts.colab_hfh.constraint_schedule import make_payload


def native_reference(path,base,study=None):
    with zipfile.ZipFile(path) as z:
        get=lambda name:json.loads(z.read('results/'+name))
        for name,expected in get('archive_manifest.json').items():
            if Path(name).name!=name or hashlib.sha256(z.read('results/'+name)).hexdigest()!=expected:
                raise ValueError('Native result manifest mismatch')
        protocol_bytes=z.read('protocol/protocol.json');protocol=json.loads(protocol_bytes)
        expected=native_payload(base,study)['protocols']['nnunet']['protocol.json']
        if protocol!=expected:raise ValueError('Native lower-budget protocol differs')
        config=get('config.json');done=get('completion.json');result=get('result.json')
        key=f"nnunet_f{config['fold']}_s{config['seed']}";full=base['references'][key]
        if path.stem!=key or config['protocol_sha256']!=hashlib.sha256(protocol_bytes).hexdigest():
            raise ValueError('Native protocol/identity binding differs')
        if any(config[k]!=full['configuration'][k] for k in MATCHED_FIELDS):
            raise ValueError('Native trajectory or split differs')
        if get('initialization.json')!=full['initialization']:raise ValueError('Native initialization differs')
        schedule=protocol['models']['nnunet']['objectives']['dice'];steps=schedule['max_steps']
        cost=schedule['flops_per_update'];used=steps*cost
        if (done['status']!='complete' or done['discarded_attempts'] or done['updates']!=steps
                or done['training_attempts_charged']!=steps or done['consumed_training_flops']!=used
                or done['calibration_flops']!=0 or not used<=protocol['budget_flops']<used+cost
                or result['updates']!=steps or result['optimization_flops']!=used):
            raise ValueError('Native endpoint/accounting differs')
        for key_name,filename in (('budget_ledger_sha256','budget_ledger.json'),('metrics_sha256','metrics.csv')):
            if hashlib.sha256(z.read('results/'+filename)).hexdigest()!=done[key_name]:
                raise ValueError('Native completion binding differs')
        validation,training=get('validation_latest.json'),get('training_final.json')
        for evaluation,names in ((validation,config['validation_cases']),(training,config['train_cases'])):
            if len(evaluation['cases'])!=10 or {c['case_name'] for c in evaluation['cases']}!=set(names):
                raise ValueError('Native final evaluation cases differ')
        history=list(csv.DictReader(io.StringIO(z.read('results/metrics.csv').decode())))
        score=validation['summary']['macro_hard_dice']
        if (not math.isfinite(score) or not 0<=score<=1 or result['validation_dice']!=score
                or int(history[-1]['charged_attempts'])!=steps
                or not math.isclose(float(history[-1]['macro_hard_dice']),score,abs_tol=1e-12)):
            raise ValueError('Native final score differs from bound evaluation')
        return key,dict(result=result,initialization=get('initialization.json'),configuration=config,
            validation=validation,training=training,archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            baseline_protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest(),precision=base['precision'])


def audit(directory,export,plan_path=None):
    base=json.loads((export/'bands_v1_payload.json').read_text())
    plan=json.loads((plan_path or export/'lower_budget_plan.json').read_text())
    refs={}
    for path in sorted((directory/'native').glob('*.zip')):
        key,reference=native_reference(path,base,plan)
        if key in refs:raise ValueError('Duplicate native seed/fold')
        refs[key]=reference
    rows=[]
    for arm in ('dice_bands','dice_bands_depth'):
        expected_protocol=make_payload(native_payload(base,plan),'nnunet',arm,plan['canonical_costs'],
            **plan['recipes'][arm]['specification'],screening=False)['protocols']['nnunet']['protocol.json']
        for path in sorted((directory/arm).glob('*.zip')):
            if len(refs)!=9:raise ValueError('Constraint audit needs all nine lower-budget controls')
            with zipfile.ZipFile(path) as zipped:
                if json.loads(zipped.read('protocol/protocol.json'))!=expected_protocol:
                    raise ValueError('Transferred constrained recipe differs from frozen budget plan')
            result=verify(path,refs)
            if result['model']!='nnunet' or result['objective']!=arm:raise ValueError('Constraint identity differs')
            key=f"nnunet_f{result['fold']}_s{result['seed']}"
            parent=next(r for r in plan['recipes'][arm]['full_budget_results']
                        if (r['fold'],r['seed'])==(result['fold'],result['seed']))
            full=base['references'][key]['result']['validation_dice']
            rows.append(dict(objective=arm,fold=result['fold'],seed=result['seed'],
                full_native_dice=full,lower_native_dice=result['baseline_dice'],
                full_constraint_dice=parent['constraint_dice'],lower_constraint_dice=result['constraint_dice'],
                lower_constraint_gain_pp=result['delta_dice_pp'],full_constraint_gain_pp=parent['delta_dice_pp'],
                change_in_gain_pp=result['delta_dice_pp']-parent['delta_dice_pp'],
                absolute_constraint_change_pp=100*(result['constraint_dice']-parent['constraint_dice']),
                optimization_flops=result['optimization_flops'],calibration_flops=result['calibration_flops'],
                lower_native_fp=result['baseline_foreground_fp'],lower_native_fn=result['baseline_foreground_fn'],
                lower_constraint_fp=result['constraint_foreground_fp'],lower_constraint_fn=result['constraint_foreground_fn'],
                archive_sha256=result['archive_sha256']))
    expected={(f,s) for f in range(3) for s in range(3)}
    groups={}
    for arm in ('dice_bands','dice_bands_depth'):
        subset=[r for r in rows if r['objective']==arm]
        identities={(r['fold'],r['seed']) for r in subset}
        if len(identities)!=len(subset):raise ValueError('Duplicate constrained seed/fold')
        complete=len(subset)==9 and identities==expected
        mean=statistics.mean(r['lower_constraint_gain_pp'] for r in subset) if complete else None
        groups[arm]=dict(pairs=len(subset),complete=complete,mean_gain_pp=mean,
            positive_pairs=sum(r['lower_constraint_gain_pp']>0 for r in subset),
            fold_mean_gain_pp={str(f):statistics.mean(r['lower_constraint_gain_pp'] for r in subset if r['fold']==f)
                               for f in range(3)} if complete else None,
            worst_gain_pp=min((r['lower_constraint_gain_pp'] for r in subset),default=None),
            mean_lower_dice_pct=100*statistics.mean(r['lower_constraint_dice'] for r in subset) if complete else None,
            mean_change_in_gain_pp=statistics.mean(r['change_in_gain_pp'] for r in subset) if complete else None,
            mean_absolute_change_pp=statistics.mean(r['absolute_constraint_change_pp'] for r in subset) if complete else None,
            mean_gain_native_ge50_pp=(statistics.mean(r['lower_constraint_gain_pp'] for r in subset
                if r['lower_native_dice']>=.5) if complete and any(r['lower_native_dice']>=.5 for r in subset) else None),
            mean_gain_native_lt50_pp=(statistics.mean(r['lower_constraint_gain_pp'] for r in subset
                if r['lower_native_dice']<.5) if complete and any(r['lower_native_dice']<.5 for r in subset) else None),
            both_control_and_constraint_all_ge50=complete and all(
                min(r['lower_native_dice'],r['lower_constraint_dice'])>=.5 for r in subset),
            meets_lower_budget_mean_target=complete and mean>=1 and all(r['lower_constraint_dice']>=.5 for r in subset))
    native_rows=[dict(fold=r['result']['fold'],seed=r['result']['seed'],dice=r['result']['validation_dice'],
        full_dice=base['references'][key]['result']['validation_dice'],
        training_dice=r['training']['summary']['macro_hard_dice'],
        validation_soft_dice=r['validation']['summary']['macro_soft_dice'],
        training_soft_dice=r['training']['summary']['macro_soft_dice'],
        foreground_fp=sum(c['foreground_fp'] for c in r['validation']['cases']),
        foreground_fn=sum(c['foreground_fn'] for c in r['validation']['cases']),
        archive_sha256=r['archive_sha256']) for key,r in refs.items()]
    result=dict(verified_native_pairs=len(refs),verified_constraint_pairs=len(rows),complete=len(refs)==9 and len(rows)==18,
        native_mean_dice_pct=100*statistics.mean(r['dice'] for r in native_rows) if len(refs)==9 else None,
        native_minimum_dice_pct=100*min(r['dice'] for r in native_rows) if len(refs)==9 else None,
        native_below50_pairs=[dict(fold=r['fold'],seed=r['seed'],dice_pct=100*r['dice'])
                             for r in native_rows if r['dice']<.5],
        groups=groups,native=native_rows,rows=rows,
        interpretation='Independent archive audit. Fixed nine pairs, no partial-set mean. Lower-budget gain is separate from absolute performance and full-budget matrix qualification. Native-floor subgroups are descriptive only; all-nine mean remains the target and failed controls are never excluded.')
    (directory/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    if rows:
        with (directory/'independent_pairs.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--export',type=Path,default=Path('reports/hfh_colab_exploration_20261111'))
    parser.add_argument('--plan',type=Path)
    args=parser.parse_args();result=audit(args.directory,args.export,args.plan)
    print(json.dumps({k:v for k,v in result.items() if k not in ('native','rows')},indent=2))
