"""Frozen nine-pair gain-only replication; no held-out recipe tuning."""
from copy import deepcopy
import inspect
import random

from scripts.colab_hfh import medsam_augmentation_components as component
from scripts.colab_hfh import medsam_intensity_augmentation as augmentation

CAMPAIGN='medsam_gain125_3fold3seed_v1'
ARMS=('dice','dice_bands','dice_edge_depth')
PAIRS=tuple((seed,fold) for fold in range(3) for seed in range(3))


def sample_prefix(protocol, slice_counts):
    rng=random.Random(protocol['seed']);cases=list(protocol['train_cases'])
    result=[]
    for _ in range(250):
        case=rng.choice(cases);result.append([case,rng.randrange(slice_counts[case])])
    return result


def make_protocol(reference, pilot, schedule, arm):
    if arm not in ARMS or (reference['seed'],reference['fold']) not in PAIRS:
        raise ValueError('Outside authorized three-arm nine-pair study')
    if reference['updates']!=200 or reference['accumulation']!=2 or len(schedule)!=250:
        raise ValueError('Original200-update reference and125-update prefix required')
    p=deepcopy(pilot);p.pop('arm',None)
    for key in ('augmentation_control_archive','augmentation_control_sha256','matched_native_archive','matched_native_sha256'):
        p.pop(key,None)
    for key in ('fold','seed','name','train_cases','evaluation_cases','support_subjects'):
        p[key]=deepcopy(reference[key])
    for key in ('base_sha256','lora_sha256','dataset_sha256','accumulation','learning_rate','final_learning_rate','warmup_updates','calibration_seed'):
        if reference[key]!=pilot[key]:raise ValueError('Original recipe differs: '+key)
    train_subjects={c.rsplit('_h',1)[0] for c in p['train_cases']}
    val_subjects={c.rsplit('_h',1)[0] for c in p['evaluation_cases']}
    if len(train_subjects)!=5 or len(val_subjects)!=5 or train_subjects & val_subjects:
        raise ValueError('Subject leakage or split size changed')
    if any(case not in p['train_cases'] for case,z in schedule):raise ValueError('Sampling includes held-out case')
    p.update(updates=125,frozen_sample_schedule=schedule,requested_arms=[arm],
        baseline_archive_sha256=None,augmentation=component.component_spec('gain'),
        calibration_policy='Fresh original training-only gradient calibration for this fold and seed; unchanged pilot targets,cap,and20-update ramp',
        comparison='Gain-only native versus BAND versus EDGE+DEPTH; all nine pairs retained;125 updates; original LR and uniform sample prefix; hardware recorded')
    return p


def check_protocol(protocol):
    if protocol['augmentation']!=component.component_spec('gain') or protocol['updates']!=125 or (protocol['seed'],protocol['fold']) not in PAIRS:
        raise ValueError('Gain replication scope changed')


def install(engine, protocol, output):
    check_protocol(protocol)
    # Dedicated worker: broaden only the old pilot's fold/seed guard.
    augmentation.check_protocol=check_protocol
    component.install(engine,protocol,output)


def install_calibration(engine, protocol, output):
    """Keep full_runtime fresh calibration, then the proven zero-update logging audit."""
    import torch
    engine.write(output/'execution_hardware.json',dict(gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
        bf16=torch.cuda.is_bf16_supported(),deterministic=torch.are_deterministic_algorithms_enabled(),
        tf32=torch.backends.cuda.matmul.allow_tf32))
    source=inspect.getsource(component.install_calibration)
    needle='    original_aug.install_calibration(engine, protocol, output)'
    if source.count(needle)!=1:raise ValueError('Audited calibration hook changed')
    source=source.replace(needle,'    # Preserve the fresh full_runtime.calibrate installed by the worker.')
    namespace=dict(component.__dict__);exec(compile(source,__file__,'exec'),namespace)
    namespace['install_calibration'](engine,protocol,output)
    original=engine.calibrate
    def calibrate(model,criterion,parameters,slabs,current,arm,destination):
        result=original(model,criterion,parameters,slabs,current,arm,destination)
        expected=[[c,z] for c,z in slabs.calibration_schedule(protocol['calibration_slabs_per_volume'])]
        if arm!='dice':
            actual=[[r['case'],r['anchor']] for r in result['rows']]
            if actual!=expected or len(actual)!=40:raise ValueError('Training-only calibration schedule differs')
            if any(c not in protocol['train_cases'] or c in protocol['evaluation_cases'] for c,z in actual):
                raise ValueError('Calibration leakage')
        engine.write(output/'calibration_scope.json',dict(policy=protocol['calibration_policy'],
            fold=protocol['fold'],seed=protocol['seed'],train_cases=protocol['train_cases'],
            validation_cases=protocol['evaluation_cases'],anchors=expected,optimizer_updates=0))
        return result
    engine.calibrate=calibrate
