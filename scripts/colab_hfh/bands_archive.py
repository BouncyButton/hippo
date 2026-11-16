"""Validate a matched BAND run before invoking the existing archive checks."""
import json
import math
from pathlib import Path
from scripts.colab_hfh import runtime


def valid_band_weight(protocol, weight):
    if not math.isfinite(weight):
        return False
    if protocol.get('control_kind'):
        return (protocol['control_kind']=='sgd_reset_without_auxiliary_gradient'
                and protocol.get('band_reset_momentum') is True
                and protocol.get('calibration_target_ratio')==0 and weight==0)
    return weight>0


def paired_result(output, reference, protocol):
    read=lambda name:json.loads((output/name).read_text())
    done,config,calibration=read('completion.json'),read('config.json'),read('calibration.json')
    if done['objective']!='dice_bands' or done['discarded_attempts']:
        raise ValueError('Expected complete, successful BAND updates')
    for key in ('model_identity','seed','fold','dataset_sha256','train_cases','validation_cases',
                'lr_horizon_updates','initial_lr','augmentation','precision','max_steps'):
        if config[key]!=reference['configuration'][key]:
            raise ValueError('Matched baseline setting differs: '+key)
    if read('initialization.json')!=reference['initialization']:
        raise ValueError('Initialization differs from matched baseline')
    if calibration['status']!='complete' or not valid_band_weight(protocol,calibration['weights']['bands']):
        raise ValueError('Invalid training-only BAND coefficient or diagnostic control')
    if [r['case_name'] for r in calibration['cases']]!=config['train_cases']:
        raise ValueError('Calibration did not use exactly the frozen training cases')
    if calibration['target_ratio']!=protocol['calibration_target_ratio'] or calibration['p95_cap']!=protocol['calibration_p95_cap']:
        raise ValueError('Calibration target changed')
    validation,training=read('validation_latest.json'),read('training_final.json')
    score=validation['summary']['macro_hard_dice']
    if not math.isfinite(score):raise ValueError('Nonfinite final Dice')
    baseline=reference['result']['validation_dice']
    def errors(result,key):return sum(c[key] for c in result['cases'])
    paired=dict(model=config['model_identity'],fold=config['fold'],seed=config['seed'],
        baseline_dice=baseline,band_dice=score,delta_dice_pp=100*(score-baseline),
        improved=score>baseline,band_training_dice=training['summary']['macro_hard_dice'],
        baseline_training_dice=reference['training']['summary']['macro_hard_dice'],
        weight=calibration['weights']['bands'],updates=done['updates'],
        optimization_flops=done['consumed_training_flops'],calibration_flops=done['calibration_flops'],
        baseline_archive_sha256=reference['archive_sha256'],baseline_protocol_sha256=reference['baseline_protocol_sha256'])
    for key in ('foreground_fp','foreground_fn'):
        paired['band_'+key]=errors(validation,key);paired['baseline_'+key]=errors(reference['validation'],key)
    if protocol.get('band_reset_momentum'):
        transition=read('band_optimizer_reset.json')
        expected=protocol['models'][protocol['trainer_alias']]['objectives']['dice_bands']['prefix_updates']+1
        if transition['attempt']!=expected or transition['cleared_state_entries']<=0:
            raise ValueError('Missing or mistimed SGD transition')
        paired['training_transition']=transition['policy']
        paired['comparison_scope']='BAND plus SGD transition versus original native baseline'
    if protocol.get('band_release_updates'):
        paired['band_release_updates']=protocol['band_release_updates']
    if 'band_final_inner_weight' in protocol:
        transition=read('band_side_transition.json')
        maximum=protocol['models'][protocol['trainer_alias']]['objectives']['dice_bands']['max_steps']
        if (abs(transition['final_inner_weight']-protocol['band_final_inner_weight'])>1e-12
                or transition['final_update_index']!=maximum-1
                or transition['start_update_index']!=protocol['band_side_transition_start']):
            raise ValueError('Missing or mistimed BAND side transition')
        paired.update(band_initial_inner_weight=protocol['band_inner_weight'],
                      band_final_inner_weight=protocol['band_final_inner_weight'],
                      band_side_transition_start=protocol['band_side_transition_start'])
    if protocol.get('control_kind'):
        if paired['weight'] != 0:raise ValueError('Optimizer diagnostic must have zero auxiliary coefficient')
        paired.update(control_kind=protocol['control_kind'],
                      comparison_scope='SGD reset only versus original native baseline; zero auxiliary gradient; not a BAND result')
    runtime.write(output/'paired_comparison.json',paired)
    return paired


def archive(output,durable,model,fold,seed,protocol,reference):
    p=runtime.read(protocol/'protocol.json')
    paired=paired_result(output,reference,p)
    source=Path(runtime.__file__).read_text()
    anchor="schedule = p['models'][p['trainer_alias']]['objectives']['dice']"
    if source.count(anchor)!=1:raise ValueError('Archive source changed')
    source=source.replace(anchor,"schedule = p['models'][p['trainer_alias']]['objectives']['dice_bands']")
    namespace={'__name__':'hfh_band_archive_checks','__file__':__file__}
    exec(compile(source,__file__,'exec'),namespace)
    namespace['archive'](output,durable,'baselines',model,fold,seed,protocol)
    return paired
