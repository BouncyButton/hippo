"""Validate complete multi-constraint comparisons before retiring checkpoints."""
import math
from pathlib import Path

from scripts.colab_hfh import runtime
from scripts.colab_hfh.constraint_calibration import components
from scripts.colab_hfh.constraint_schedule import training_flops


def paired_result(output, reference, protocol):
    read = lambda name: runtime.read(output/name)
    arm = protocol['active_objective']
    done, config, calibration = read('completion.json'), read('config.json'), read('calibration.json')
    schedule = protocol['models'][protocol['trainer_alias']]['objectives'][arm]
    if done['objective'] != arm or done['discarded_attempts'] or done['updates'] != schedule['max_steps']:
        raise ValueError('Incomplete objective or budget endpoint')
    for key in ('model_identity', 'seed', 'fold', 'dataset_sha256', 'train_cases', 'validation_cases',
                'lr_horizon_updates', 'initial_lr', 'augmentation', 'precision'):
        if config[key] != reference['configuration'][key]:
            raise ValueError('Matched baseline setting differs: '+key)
    if read('initialization.json') != reference['initialization']:
        raise ValueError('Initialization differs from matched baseline')
    if (done['consumed_training_flops'] != training_flops(done['updates'], schedule)
            or done['consumed_training_flops'] > protocol['budget_flops']
            or training_flops(done['updates']+1, schedule) <= protocol['budget_flops']):
        raise ValueError('Optimization allowance does not match the frozen schedule')
    if (done['calibration_attempts'] != 1 or done['calibration_flops'] != schedule['calibration_flops']
            or calibration['status'] != 'complete'
            or calibration['method'] != protocol['constraint_calibration']):
        raise ValueError('Calibration accounting or method differs')
    weights = calibration['weights']
    if set(weights) != set(components(arm)) or any(not math.isfinite(v) or v <= 0 for v in weights.values()):
        raise ValueError('Invalid auxiliary coefficients')
    if [row['case_name'] for row in calibration['cases']] != config['train_cases']:
        raise ValueError('Calibration cases differ from training split')
    stats = calibration['statistics']['joint']
    if (stats['worst_ramp_median_ratio'] > protocol['constraint_joint_target']+1e-9
            or stats['worst_ramp_p95_ratio'] > protocol['constraint_joint_cap']+1e-9):
        raise ValueError('Joint calibration exceeds the declared ramp envelope')
    validation, training = read('validation_latest.json'), read('training_final.json')
    score = validation['summary']['macro_hard_dice']
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('Invalid final Dice')
    baseline = reference['result']['validation_dice']
    paired = dict(model=config['model_identity'], objective=arm, fold=config['fold'], seed=config['seed'],
        baseline_dice=baseline, constraint_dice=score, delta_dice_pp=100*(score-baseline),
        improved=score > baseline, constraint_training_dice=training['summary']['macro_hard_dice'],
        baseline_training_dice=reference['training']['summary']['macro_hard_dice'], weights=weights,
        updates=done['updates'], baseline_updates=reference['configuration']['max_steps'],
        optimization_flops=done['consumed_training_flops'], calibration_flops=done['calibration_flops'],
        baseline_archive_sha256=reference['archive_sha256'],
        baseline_protocol_sha256=reference['baseline_protocol_sha256'],
        comparison_scope='Optimization FLOP cap matched; unchanged native loss/optimizer/LR horizon; final-budget endpoint; exploratory validation selection')
    for name in ('foreground_fp', 'foreground_fn'):
        paired['constraint_'+name] = sum(row[name] for row in validation['cases'])
        paired['baseline_'+name] = sum(row[name] for row in reference['validation']['cases'])
    if (output/'gradient_strength_final.json').exists():
        diagnostic=read('gradient_strength_final.json')
        if (diagnostic['optimization_updates']!=0
                or diagnostic['diagnostic_flops_allowance']!=2*schedule['calibration_flops']
                or diagnostic['weights']!=weights
                or diagnostic['status'] not in ('complete','unavailable')):
            raise ValueError('Post-training diagnostic accounting differs')
        if diagnostic['status']=='complete':
            from scripts.colab_hfh.constraint_diagnostics import fixed_weight_statistics
            if ([r['case_name'] for r in diagnostic['cases']]!=config['train_cases']
                    or diagnostic['anchor_model_sha256']!=calibration['anchor_model_sha256']):
                raise ValueError('Post-training diagnostic cases or anchor differ')
            for key,cases in (('anchor',calibration['cases']),('final',diagnostic['cases'])):
                expected=fixed_weight_statistics(cases,arm,weights)
                for term,values in expected.items():
                    for statistic,value in values.items():
                        if not math.isclose(value,diagnostic[key][term][statistic],rel_tol=1e-9,abs_tol=1e-12):
                            raise ValueError('Post-training diagnostic statistics differ')
        paired['post_training_diagnostic_flops_allowance']=diagnostic['diagnostic_flops_allowance']
        paired['training_calibration_diagnostic_flops_allowance']=(done['consumed_total_flops']
            +diagnostic['diagnostic_flops_allowance'])
        paired['gradient_diagnostic_status']=diagnostic['status']
    if (output/'parameter_gradient_final.json').exists():
        from scripts.colab_hfh.parameter_gradient_diagnostics import validate_report
        parameter_probe=read('parameter_gradient_final.json')
        final_sha=(read('gradient_strength_final.json').get('final_model_sha256')
                   if (output/'gradient_strength_final.json').exists() else None)
        allowance=validate_report(parameter_probe,config,schedule,weights,final_sha)
        paired['parameter_gradient_diagnostic_flops_allowance']=allowance
        paired['parameter_gradient_diagnostic_status']=parameter_probe['status']
        paired['training_calibration_diagnostic_flops_allowance']=(done['consumed_total_flops']
            +paired.get('post_training_diagnostic_flops_allowance',0)+allowance)
    if protocol.get('online_sdf_gradient_ratio') is not None:
        from scripts.colab_hfh.constraint_online_sdf import validate_trace
        paired['online_sdf_strength']=validate_trace(output,protocol,schedule,weights['sdf'])
    if protocol.get('band_adaptive_sides'):
        if protocol.get('variant_extra_scalar_flops_allowance') != 64 or 'bands' not in components(arm):
            raise ValueError('Adaptive BAND arithmetic declaration differs')
        paired['band_side_policy']='training_case_detached_BCE_ratio_clipped_0.1_0.9'
        paired['band_side_extra_scalar_flops_per_update']=64
    runtime.write(output/'paired_comparison.json', paired)
    return paired


def archive(output, durable, model, fold, seed, protocol, reference):
    p = runtime.read(protocol/'protocol.json')
    paired = paired_result(output, reference, p)
    source = Path(runtime.__file__).read_text()
    anchor = "schedule = p['models'][p['trainer_alias']]['objectives']['dice']"
    if source.count(anchor) != 1:
        raise ValueError('Archive integration changed')
    source = source.replace(anchor, "schedule = p['models'][p['trainer_alias']]['objectives'][p['active_objective']]")
    namespace = {'__name__': 'hfh_constraint_archive_checks', '__file__': __file__}
    exec(compile(source, __file__, 'exec'), namespace)
    namespace['archive'](output, durable, 'baselines', model, fold, seed, protocol)
    return paired
