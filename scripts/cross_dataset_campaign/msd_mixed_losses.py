"""MSD constraints with native nnUNet Dice+CE and the other models' Dice loss."""
import argparse
from functools import partial
import hashlib
from pathlib import Path
import json
import time

from scripts.cross_dataset_campaign import conventional as experiment
from scripts.cross_dataset_campaign.common import ARMS, MODELS, read, write, digest
from scripts.capped_depth_delay import objectives as original

calibration_case = original.calibration_case


def training_loss(outputs, labels, arm, weights, scale, sdf=None,
                  criterion=None, depth=None, sdf_scale=0., *, sdf_policy='delayed_5_of_9'):
    if criterion is not None and (criterion.loss.weight_ce != 1 or criterion.loss.weight_dice != 1):
        raise ValueError('Native nnUNet Dice+CE coefficients must both be one')
    if 'sdf' in arm and sdf_policy == 'original_shared_ramp':
        return original.full_loss(outputs, labels, arm, weights, scale, criterion, sdf, depth)
    return original.training_loss(outputs, labels, arm, weights, scale, sdf, criterion, depth, sdf_scale)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--protocol-dir', type=Path, required=True)
    parser.add_argument('--objective', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cuda','cpu'), default='cuda')
    args, _ = parser.parse_known_args()
    p = read(args.protocol_dir/'protocol.json')
    native = p['model_identity']=='nnunet'
    target, ceiling = (.6,.75) if native else (.1,.5)
    if (p['dataset']!='MSD' or p['model_identity'] not in MODELS or p['augmentation']!='none'
            or p['weight_ce']!=int(native) or args.objective not in ARMS
            or p['calibration_target_ratio']!=target or p['calibration_p95_cap']!=ceiling):
        raise ValueError('Unexpected mixed-loss MSD recipe')
    experiment.configure(p,args.objective)
    # Restoring this builder preserves native foreground Dice+CE/deep supervision
    # for nnUNet; its other branch already uses the established MONAI Dice loss.
    experiment.base.build_loss_and_optimizer = experiment.ORIGINAL_LOSS_OPTIMIZER
    experiment.trainer.training_loss = partial(training_loss,sdf_policy=p['sdf_policy'][args.objective])
    experiment.trainer.calibration_case = calibration_case
    original_calibrate = experiment.trainer.calibrate
    def calibrate(*args, **kwargs):
        report = original_calibrate(*args, **kwargs)
        report.update(base_gradient='Dice+CE' if native else 'Dice',
            scope='All ten training crops only; sequential auxiliary calibration relative to the accumulated weighted logit gradient; native full-resolution deep-supervision coefficient retained for nnUNet')
        return report
    experiment.trainer.calibrate = calibrate
    builder = experiment.base.build_model
    def build_model(name, architecture):
        model = builder(name,architecture)
        h = hashlib.sha256()
        for key,value in sorted(model.state_dict().items()):
            h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
        if h.hexdigest()!=p['baseline_reference']['initialization_sha256']:
            raise ValueError('Initialization differs from the matched baseline')
        write(args.output/'initialization.json',dict(model_state_sha256=h.hexdigest()))
        return model
    experiment.base.build_model = build_model
    write(args.output/'runtime_recipe.json',dict(dataset='MSD',model=p['model_identity'],
        objective=args.objective,weight_dice=1,weight_ce=int(native),dice_include_background=not native,
        native_deep_supervision=native,augmentation='none',constraints=args.objective.split('_')[1:],
        calibration_base='Dice+CE' if native else 'Dice',target=target,p95_cap=ceiling,
        sdf_policy=p['sdf_policy'][args.objective],baseline_reference=p['baseline_reference'],
        runtime_acceleration=p.get('runtime_acceleration',{}),
        primary_checkpoint='final_budget',protocol_sha256=digest(args.protocol_dir/'protocol.json')))
    acceleration=None
    if p.get('runtime_acceleration',{}).get('enabled'):
        from scripts.cross_dataset_campaign.msd_runtime_fast import RuntimeAcceleration
        acceleration=RuntimeAcceleration(experiment,p,args.device).install()
    started=time.perf_counter()
    try:
        experiment.trainer.main()
    finally:
        if acceleration:
            record=dict(**acceleration.stats,trainer_wall_seconds=time.perf_counter()-started,
                recorded_at=time.time(),runtime_acceleration=p['runtime_acceleration'])
            write(args.output/'performance.json',record)
            with (args.output/'performance_invocations.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            acceleration.close()


if __name__=='__main__':
    main()
