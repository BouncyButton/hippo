"""Apply the recorded augmentation and calibrate the adapter on its support cases."""
import inspect

from scripts.colab_hfh import medsam_augmentation_components as component

from scripts.colab_hfh import medsam_intensity_augmentation as augmentation

PAIRS=tuple((seed,fold) for fold in range(3) for seed in range(3))

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
